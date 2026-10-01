/* Host half of ds4_qwen4exp_mm.h: the cell map and the per-forward row
 * planner.  All device work is thin tensor writes; the arithmetic that
 * decides WHERE a row lands lives here so the kernels stay shape-only. */
#include "ds4_qwen4exp_mm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

ds4_qwen4exp_mm *ds4_qwen4exp_mm_create(uint32_t n_ctx, uint32_t pool_size,
                                       uint32_t row_cap, uint32_t n_embd,
                                       const int32_t sections[4]) {
    if (!n_ctx || !pool_size || !row_cap || !n_embd || !sections) return NULL;
    ds4_qwen4exp_mm *mm = calloc(1, sizeof(*mm));
    if (!mm) return NULL;
    mm->n_ctx = n_ctx;
    mm->pool_size = pool_size;
    mm->row_cap = row_cap;
    mm->n_embd = n_embd;
    const uint32_t n_blocks = n_ctx / pool_size;
    mm->cell_pos   = malloc((size_t)n_ctx * sizeof(int32_t));
    mm->cell_blk   = malloc((size_t)n_ctx * sizeof(int32_t));
    mm->blk_cells  = calloc((size_t)n_blocks * pool_size, sizeof(int32_t));
    mm->blk_filled = calloc((size_t)n_blocks, sizeof(int32_t));
    mm->dirty      = malloc((size_t)row_cap * sizeof(int32_t));
    mm->pend_pos   = malloc((size_t)row_cap * 4u * sizeof(int32_t));
    mm->pend_embd  = malloc((size_t)row_cap * n_embd * sizeof(float));
    mm->pend_map   = malloc((size_t)row_cap * sizeof(int32_t));
    mm->d_cell_pos  = ds4_gpu_tensor_alloc((uint64_t)n_ctx * 4u);
    mm->d_cell_blk  = ds4_gpu_tensor_alloc((uint64_t)n_ctx * 4u);
    mm->d_blk_filled= ds4_gpu_tensor_alloc((uint64_t)n_blocks * 4u);
    mm->d_blk_cells = ds4_gpu_tensor_alloc((uint64_t)n_blocks * pool_size * 4u);
    mm->d_row_pos   = ds4_gpu_tensor_alloc((uint64_t)row_cap * 4u * 4u);
    mm->d_row_embd  = ds4_gpu_tensor_alloc((uint64_t)row_cap * n_embd * 4u);
    mm->d_embd_map  = ds4_gpu_tensor_alloc((uint64_t)row_cap * 4u);
    mm->d_dirty     = ds4_gpu_tensor_alloc((uint64_t)row_cap * 4u);
    mm->d_sections  = ds4_gpu_tensor_alloc(4u * sizeof(int32_t));
    /* rope sections never change; upload once.  Failure is caught by the
     * callers' tensor-size checks (an empty tensor fails them). */
    if (mm->d_sections)
        ds4_gpu_tensor_write(mm->d_sections, 0, sections,
                             4u * sizeof(int32_t));
    if (!mm->cell_pos || !mm->cell_blk || !mm->blk_cells || !mm->blk_filled ||
        !mm->dirty || !mm->d_cell_pos || !mm->d_cell_blk ||
        !mm->d_blk_filled || !mm->d_blk_cells || !mm->d_row_pos ||
        !mm->d_row_embd || !mm->d_embd_map || !mm->d_dirty ||
        !mm->d_sections || !mm->pend_pos || !mm->pend_embd ||
        !mm->pend_map) {
        ds4_qwen4exp_mm_free(mm);
        return NULL;
    }
    ds4_qwen4exp_mm_reset(mm);
    return mm;
}

/* Host-only ctor for unit tests: identical planner/cell-map state, no
 * device tensors.  mm_upload() skips NULL d_* tensors. */
ds4_qwen4exp_mm *ds4_qwen4exp_mm_create_host(uint32_t n_ctx,
                                           uint32_t pool_size,
                                           uint32_t row_cap,
                                           uint32_t n_embd,
                                           const int32_t sections[4]) {
    (void)sections;
    if (!n_ctx || !pool_size || !row_cap || !n_embd) return NULL;
    ds4_qwen4exp_mm *mm = calloc(1, sizeof(*mm));
    if (!mm) return NULL;
    mm->n_ctx = n_ctx;
    mm->pool_size = pool_size;
    mm->row_cap = row_cap;
    mm->n_embd = n_embd;
    const uint32_t n_blocks = n_ctx / pool_size;
    mm->cell_pos   = malloc((size_t)n_ctx * sizeof(int32_t));
    mm->cell_blk   = malloc((size_t)n_ctx * sizeof(int32_t));
    mm->blk_cells  = calloc((size_t)n_blocks * pool_size, sizeof(int32_t));
    mm->blk_filled = calloc((size_t)n_blocks, sizeof(int32_t));
    mm->dirty      = malloc((size_t)row_cap * sizeof(int32_t));
    mm->pend_pos   = malloc((size_t)row_cap * 4u * sizeof(int32_t));
    mm->pend_embd  = malloc((size_t)row_cap * n_embd * sizeof(float));
    mm->pend_map   = malloc((size_t)row_cap * sizeof(int32_t));
    if (!mm->cell_pos || !mm->cell_blk || !mm->blk_cells ||
        !mm->blk_filled || !mm->dirty || !mm->pend_pos ||
        !mm->pend_embd || !mm->pend_map) {
        ds4_qwen4exp_mm_free(mm);
        return NULL;
    }
    for (uint32_t i = 0; i < mm->n_ctx; i++) mm->cell_pos[i] = -1;
    memset(mm->cell_blk, 0, (size_t)mm->n_ctx * sizeof(int32_t));
    return mm;
}

void ds4_qwen4exp_mm_free(ds4_qwen4exp_mm *mm) {
    if (!mm) return;
    free(mm->cell_pos); free(mm->cell_blk);
    free(mm->pend_pos); free(mm->pend_embd); free(mm->pend_map);
    free(mm->blk_cells); free(mm->blk_filled); free(mm->dirty);
    ds4_gpu_tensor_free(mm->d_cell_pos);
    ds4_gpu_tensor_free(mm->d_cell_blk);
    ds4_gpu_tensor_free(mm->d_blk_filled);
    ds4_gpu_tensor_free(mm->d_blk_cells);
    ds4_gpu_tensor_free(mm->d_row_pos);
    ds4_gpu_tensor_free(mm->d_row_embd);
    ds4_gpu_tensor_free(mm->d_embd_map);
    ds4_gpu_tensor_free(mm->d_dirty);
    ds4_gpu_tensor_free(mm->d_sections);
    free(mm);
}

void ds4_qwen4exp_mm_reset(ds4_qwen4exp_mm *mm) {
    if (!mm) return;
    mm->active = 0;
    mm->n_rows = 0;
    mm->t = 0;
    mm->mtp_off = 0;
    mm->img_open = 0;
    mm->dirty_n = 0;
    const uint32_t n_blocks = mm->n_ctx / mm->pool_size;
    if (mm->cell_pos) {
        for (uint32_t i = 0; i < mm->n_ctx; i++) mm->cell_pos[i] = -1;
        memset(mm->cell_blk, 0, (size_t)mm->n_ctx * sizeof(int32_t));
        memset(mm->blk_cells, 0,
               (size_t)n_blocks * mm->pool_size * sizeof(int32_t));
        memset(mm->blk_filled, 0, (size_t)n_blocks * sizeof(int32_t));
        /* The device mirrors must agree with the host mirrors before the
         * first mm forward reads them. */
        ds4_gpu_tensor_write(mm->d_cell_pos, 0, mm->cell_pos,
                             (uint64_t)mm->n_ctx * 4u);
        ds4_gpu_tensor_write(mm->d_cell_blk, 0, mm->cell_blk,
                             (uint64_t)mm->n_ctx * 4u);
        ds4_gpu_tensor_write(mm->d_blk_filled, 0, mm->blk_filled,
                             (uint64_t)n_blocks * 4u);
        ds4_gpu_tensor_write(mm->d_blk_cells, 0, mm->blk_cells,
                             (uint64_t)n_blocks * mm->pool_size * 4u);
    }
    mm->cell_up_lo = mm->cell_up_hi = 0;
    mm->blk_up_lo = mm->blk_up_hi = 0;
    mm->filled_up_lo = mm->filled_up_hi = 0;
}

static void mm_touch(uint32_t *lo, uint32_t *hi, uint32_t idx, int first) {
    if (first || idx < *lo) *lo = idx;
    if (idx + 1u > *hi) *hi = idx + 1u;
}

int ds4_qwen4exp_mm_plan(ds4_qwen4exp_mm *st,
                         const ds4_qwen4exp_mm_slice *slices,
                         uint32_t n_slices,
                         const int32_t *is_image,
                         uint32_t n, uint32_t *t,
                         int32_t *pos_out, float *embd_out,
                         int32_t *embd_map,
                         uint32_t *pos_advance,
                         char *err, size_t errlen) {
    if (!st || !is_image || !pos_out || !embd_map || !t || !pos_advance ||
        (n_slices && !slices) || !n || n > st->row_cap) {
        if (err && errlen) snprintf(err, errlen, "qwen4exp mm: bad plan call");
        return 0;
    }
    if ((uint64_t)st->n_rows + n > st->n_ctx) {
        if (err && errlen) snprintf(err, errlen, "qwen4exp mm: %u rows overflow the context",
                 n + st->n_rows);
        return 0;
    }
    uint32_t cur_t = *t;
    uint32_t slice_at = 0;   /* index into slices */
    uint32_t slice_row = 0;  /* rows consumed of current slice */
    uint32_t n_embd = 0;
    st->dirty_n = 0;
    int up_first = 1;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t slot = st->n_rows + i;
        int32_t *pp = pos_out + 4u * i;
        if (is_image[i]) {
            if (slice_at >= n_slices) {
                if (err && errlen) snprintf(err, errlen, "qwen4exp mm: image row without slice");
                return 0;
            }
            const ds4_qwen4exp_mm_slice *sl = &slices[slice_at];
            if (slice_row >= sl->n_rows) {
                if (err && errlen) snprintf(err, errlen, "qwen4exp mm: slice overrun");
                return 0;
            }
            /* In-flight grid continuation or a fresh one?  A slice is always
             * part of ONE image: n_rows rows of its nx*ny grid. */
            if (!st->img_open) {
                st->img_t0 = cur_t;
                st->img_nx = sl->nx;
                st->img_ny = sl->ny;
                st->img_total = sl->nx * sl->ny;
                st->img_done = sl->embd_skip;
                st->img_open = 1;
            }
            if (sl->nx != st->img_nx || sl->ny != st->img_ny ||
                st->img_done >= st->img_total) {
                if (err && errlen) snprintf(err, errlen, "qwen4exp mm: inconsistent grid");
                return 0;
            }
            const uint32_t k = st->img_done;
            pp[0] = (int32_t)st->img_t0;
            pp[1] = (int32_t)(st->img_t0 + k / sl->nx);   /* h = row */
            pp[2] = (int32_t)(st->img_t0 + k % sl->nx);   /* w = col */
            pp[3] = 0;
            embd_map[i] = (int32_t)n_embd;
            if (embd_out && sl->embd) {
                memcpy(embd_out + (size_t)n_embd * st->n_embd,
                       sl->embd + (size_t)(sl->embd_skip + slice_row) *
                                  st->n_embd,
                       (size_t)st->n_embd * sizeof(float));
            }
            n_embd++;
            st->img_done++;
            slice_row++;
            if (slice_row == sl->n_rows) { slice_at++; slice_row = 0; }
            if (st->img_done == st->img_total) {
                cur_t = st->img_t0 +
                        (sl->nx > sl->ny ? sl->nx : sl->ny);
                st->img_open = 0;
            }
            st->active = 1;
        } else {
            pp[0] = pp[1] = pp[2] = (int32_t)cur_t;
            pp[3] = 0;
            embd_map[i] = -1;
            cur_t++;
        }
        /* Cell map update: logical t for this cell, position-block member. */
        const uint32_t p = (uint32_t)pp[0];
        const uint32_t b = p / st->pool_size;
        st->cell_pos[slot] = (int32_t)p;
        st->cell_blk[slot] = (int32_t)b;
        st->blk_cells[b * st->pool_size + (p % st->pool_size)] =
            (int32_t)slot;
        st->blk_filled[b]++;
        mm_touch(&st->cell_up_lo, &st->cell_up_hi, slot, up_first);
        mm_touch(&st->blk_up_lo, &st->blk_up_hi, b * st->pool_size,
                 up_first);
        mm_touch(&st->blk_up_lo, &st->blk_up_hi,
                 b * st->pool_size + st->pool_size - 1u, up_first);
        mm_touch(&st->filled_up_lo, &st->filled_up_hi, b, up_first);
        up_first = 0;
        /* A block whose membership changed needs repooling.  Dedup by the
         * fact that positions within a row range are dense per axis, so the
         * same block repeats consecutively at pool_size granularity. */
        if (st->dirty_n == 0 ||
            st->dirty[st->dirty_n - 1] != (int32_t)b) {
            int dup = 0;
            for (uint32_t d = 0; d < st->dirty_n; d++)
                if (st->dirty[d] == (int32_t)b) { dup = 1; break; }
            if (!dup) {
                if (st->dirty_n >= st->row_cap) {
                    if (err && errlen) snprintf(err, errlen,
                             "qwen4exp mm: dirty-block list overflow");
                    return 0;
                }
                st->dirty[st->dirty_n++] = (int32_t)b;
            }
        }
    }
    if (slice_at != n_slices || slice_row != 0) {
        if (err && errlen) snprintf(err, errlen, "qwen4exp mm: slice/row count mismatch");
        return 0;
    }
    *pos_advance = cur_t - *t;
    *t = cur_t;
    return 1;
}

void ds4_qwen4exp_mm_commit(ds4_qwen4exp_mm *st, uint32_t n) {
    if (st) st->n_rows += n;
}

int ds4_qwen4exp_mm_upload(ds4_qwen4exp_mm *st,
                           const int32_t *pos, const float *embd,
                           const int32_t *embd_map, uint32_t n,
                           char *err, size_t errlen) {
    if (!st || !pos || !embd_map || !n) {
        if (err && errlen) snprintf(err, errlen, "qwen4exp mm: bad upload call");
        return 0;
    }
    if (st->d_row_pos &&
        (!ds4_gpu_tensor_write(st->d_row_pos, 0, pos,
                              (uint64_t)n * 4u * sizeof(int32_t)) ||
         !ds4_gpu_tensor_write(st->d_embd_map, 0, embd_map,
                              (uint64_t)n * sizeof(int32_t)))) {
        if (err && errlen) snprintf(err, errlen, "qwen4exp mm: row upload failed");
        return 0;
    }
    /* Compacted embedding rows only; the map is already up. */
    uint32_t n_embd = 0;
    for (uint32_t i = 0; i < n; i++)
        if (embd_map[i] >= 0) n_embd++;
    if (n_embd && embd && st->d_row_embd &&
        !ds4_gpu_tensor_write(st->d_row_embd, 0, embd,
                              (uint64_t)n_embd * st->n_embd *
                              sizeof(float))) {
        if (err && errlen) snprintf(err, errlen, "qwen4exp mm: embedding upload failed");
        return 0;
    }
    if (st->d_cell_pos && st->cell_up_lo < st->cell_up_hi &&
        (!ds4_gpu_tensor_write(
             st->d_cell_pos, (uint64_t)st->cell_up_lo * 4u,
             st->cell_pos + st->cell_up_lo,
             (uint64_t)(st->cell_up_hi - st->cell_up_lo) * 4u) ||
         !ds4_gpu_tensor_write(
             st->d_cell_blk, (uint64_t)st->cell_up_lo * 4u,
             st->cell_blk + st->cell_up_lo,
             (uint64_t)(st->cell_up_hi - st->cell_up_lo) * 4u))) {
        if (err && errlen) snprintf(err, errlen, "qwen4exp mm: cell map upload failed");
        return 0;
    }
    if (st->d_blk_cells && st->blk_up_lo < st->blk_up_hi &&
        !ds4_gpu_tensor_write(
            st->d_blk_cells, (uint64_t)st->blk_up_lo * 4u,
            st->blk_cells + st->blk_up_lo,
            (uint64_t)(st->blk_up_hi - st->blk_up_lo) * 4u)) {
        if (err && errlen) snprintf(err, errlen, "qwen4exp mm: block map upload failed");
        return 0;
    }
    if (st->d_blk_filled && st->filled_up_lo < st->filled_up_hi &&
        !ds4_gpu_tensor_write(
            st->d_blk_filled, (uint64_t)st->filled_up_lo * 4u,
            st->blk_filled + st->filled_up_lo,
            (uint64_t)(st->filled_up_hi - st->filled_up_lo) * 4u)) {
        if (err && errlen) snprintf(err, errlen, "qwen4exp mm: fill counts upload failed");
        return 0;
    }
    if (st->dirty_n && st->d_dirty &&
        !ds4_gpu_tensor_write(st->d_dirty, 0, st->dirty,
                              (uint64_t)st->dirty_n * sizeof(int32_t))) {
        if (err && errlen) snprintf(err, errlen, "qwen4exp mm: dirty list upload failed");
        return 0;
    }
    st->cell_up_lo = st->cell_up_hi = 0;
    st->blk_up_lo = st->blk_up_hi = 0;
    st->filled_up_lo = st->filled_up_hi = 0;
    return 1;
}
