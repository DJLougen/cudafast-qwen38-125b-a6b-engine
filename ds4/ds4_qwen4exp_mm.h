/*
 * Qwen4exp multimodal (image) input -- the "cell map" half of M-RoPE.
 *
 * Text qwen4exp treats position and KV slot as the same integer.  Images
 * break that: llama.cpp's MTMD_POS_TYPE_MROPE gives every image token a
 * (t, h, w) position with grid coordinates, and the position axis only
 * advances once per image by max(nx, ny).  Downstream:
 *
 *   - RoPE positions are per-row (t, h, w) triples, not pos0 + row.
 *   - The QSA indexer tape, K/V caches and attention key lists index by
 *     CELL (dense slot), while pooled blocks, completeness and the
 *     tail-bias rule index by POSITION.  llama.cpp keeps a blk_cells map
 *     (position -> cell, last write wins) plus a filled[] count per block;
 *     this module mirrors both.
 *   - Attention visibility for query row i is the slot prefix [0, slot_i]:
 *     cells are appended in raster order, which for a same-t grid is the
 *     visibility order too.  The dense QSA attention kernel is unchanged;
 *     sparse rows get their cell list from the mm select kernel, which
 *     emits cell ids instead of positions.
 *
 * Row descriptors are built on the host per forward (pos: 4 x int32 per
 * row -- t, h, w, unused -- plus an embedding-row map), uploaded once, and
 * consumed by the mm kernel variants in ds4_cuda_qwen4exp_mm.cu.
 */
#ifndef DS4_QWEN4EXP_MM_H
#define DS4_QWEN4EXP_MM_H

#include <stdint.h>
#include <stddef.h>
#include "ds4_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_qwen4exp_mm {
    int      active;               /* 1 once any image row was planned */
    uint32_t n_rows;               /* dense cells written (slot cursor) */
    uint32_t t;                    /* shared M-RoPE position cursor; diverges
                                    * from n_rows after an image (advances
                                    * by max(nx,ny) instead of n_tokens) */
    int      mtp_off;              /* 1: MTP disabled for this session */

    /* host cell map (position domain), updated incrementally */
    int32_t *cell_pos;             /* n_ctx: logical t per cell, -1 empty */
    int32_t *cell_blk;             /* n_ctx: pos block of each cell */
    int32_t *blk_cells;            /* (n_ctx/r)*r: pos -> cell, last wins */
    int32_t *blk_filled;           /* n_ctx/r: writes per block */
    uint32_t n_ctx;
    uint32_t pool_size;            /* indexer compress ratio (4) */
    uint32_t n_embd;

    /* in-flight image grid when one spans a chunk boundary */
    uint32_t img_t0, img_nx, img_ny, img_done, img_total;
    int      img_open;

    /* device tensors */
    ds4_gpu_tensor *d_cell_pos;
    ds4_gpu_tensor *d_cell_blk;
    ds4_gpu_tensor *d_blk_filled;
    ds4_gpu_tensor *d_blk_cells;
    ds4_gpu_tensor *d_row_pos;     /* i32 4 * row_cap */
    ds4_gpu_tensor *d_row_embd;    /* f32 n_embd * row_cap (compacted) */
    ds4_gpu_tensor *d_embd_map;    /* i32 row_cap: image row -> embd idx */
    ds4_gpu_tensor *d_dirty;       /* i32 row_cap: dirty block ids */
    ds4_gpu_tensor *d_sections;    /* i32[4] rope section sizes (t,h,w,e) */
    uint32_t        row_cap;

    /* per-forward plan staging: the caller (ds4.c) fills these between
     * plan and upload; graph.inc reads them inside the forward. */
    int32_t *pend_pos;             /* 4*row_cap */
    float   *pend_embd;            /* n_embd*row_cap */
    int32_t *pend_map;             /* row_cap */

    /* per-forward upload bookkeeping, set by ds4_qwen4exp_mm_plan */
    uint32_t dirty_n;              /* dirty blocks queued in dirty[] */
    int32_t *dirty;                /* host dirty list, row_cap entries */
    uint32_t cell_up_lo, cell_up_hi;     /* cell_pos/cell_blk range */
    uint32_t blk_up_lo, blk_up_hi;       /* blk_cells pos range */
    uint32_t filled_up_lo, filled_up_hi; /* blk_filled block range */
} ds4_qwen4exp_mm;

/* The planner's input view of one vision span, so this header stays free
 * of ds4.h.  `embd` points at the span's embedding row `skip` (0-based
 * within the grid) and must hold at least n_rows * n_embd floats past it. */
typedef struct {
    uint32_t      prompt_lo;
    uint32_t      n_rows;
    uint32_t      nx, ny;
    const float  *embd;
    uint32_t      embd_skip;
} ds4_qwen4exp_mm_slice;

ds4_qwen4exp_mm *ds4_qwen4exp_mm_create(uint32_t n_ctx, uint32_t pool_size,
                                       uint32_t row_cap, uint32_t n_embd,
                                       const int32_t sections[4]);

/* Host-only variant for unit tests: same planner/cell-map state, no device
 * tensors (upload() then only writes d_* when non-NULL). */
ds4_qwen4exp_mm *ds4_qwen4exp_mm_create_host(uint32_t n_ctx, uint32_t pool_size,
                                           uint32_t row_cap, uint32_t n_embd,
                                           const int32_t sections[4]);
void ds4_qwen4exp_mm_free(ds4_qwen4exp_mm *mm);
void ds4_qwen4exp_mm_reset(ds4_qwen4exp_mm *mm);

/* Plan `n` rows: pos_out gets 4*n positions (t,h,w,0), embd_map[i] is -1
 * for token rows or the compacted embd_out row index, cell-map mirrors are
 * updated and the device upload ranges recorded on `state`.  `*t` is the
 * logical-t cursor (in/out); *pos_advance gets the logical t consumed.
 * slices[] lists the image rows inside this chunk in order; is_image[i]
 * marks image rows.  Returns 0 with *err on error. */
int ds4_qwen4exp_mm_plan(ds4_qwen4exp_mm *state,
                         const ds4_qwen4exp_mm_slice *slices,
                         uint32_t n_slices,
                         const int32_t *is_image,
                         uint32_t n, uint32_t *t,
                         int32_t *pos_out, float *embd_out,
                         int32_t *embd_map,
                         uint32_t *pos_advance,
                         char *err, size_t errlen);

/* Advance the slot cursor after a successful forward. */
void ds4_qwen4exp_mm_commit(ds4_qwen4exp_mm *state, uint32_t n);

/* Upload the planned per-forward state to the device tensors.  Runs
 * BEFORE the forward's command batch opens, like the token upload. */
int ds4_qwen4exp_mm_upload(ds4_qwen4exp_mm *state,
                           const int32_t *pos, const float *embd,
                           const int32_t *embd_map, uint32_t n,
                           char *err, size_t errlen);

/* ---- CUDA entry points (ds4_cuda_qwen4exp_mm.cu; DS4_NO_GPU builds must
 * not call them) ---- */
int ds4_gpu_qwen4exp_mm_embd_rows(
        ds4_gpu_tensor *hyper, const ds4_gpu_tensor *embd,
        const ds4_gpu_tensor *embd_map, uint32_t n_tokens, uint32_t n_embd,
        uint32_t n_hc, uint32_t row_cap);
int ds4_gpu_qwen4exp_mm_prep_q(
        ds4_gpu_tensor *q, ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *doubled, const ds4_gpu_tensor *weight,
        const ds4_gpu_tensor *inv_freq, const ds4_gpu_tensor *row_pos,
        const ds4_gpu_tensor *sections,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t rot_dim, float eps, float weight_offset);
int ds4_gpu_qwen4exp_mm_prep_kv(
        ds4_gpu_tensor *k_cache, ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *raw_k, const ds4_gpu_tensor *raw_v,
        const ds4_gpu_tensor *weight, const ds4_gpu_tensor *inv_freq,
        const ds4_gpu_tensor *row_pos, const ds4_gpu_tensor *sections,
        const ds4_gpu_tensor *d_pos,
        uint32_t n_tokens, uint32_t n_head_kv, uint32_t head_dim,
        uint32_t rot_dim, uint32_t cache_cap, float eps,
        float weight_offset);
int ds4_gpu_qwen4exp_mm_rope_idxq(
        ds4_gpu_tensor *idx_q, const ds4_gpu_tensor *inv_freq,
        const ds4_gpu_tensor *row_pos, const ds4_gpu_tensor *sections,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t rot_dim);
int ds4_gpu_qwen4exp_mm_pool_update(
        ds4_gpu_tensor *pool, ds4_gpu_tensor *tape,
        const ds4_gpu_tensor *raw_k, const ds4_gpu_tensor *k_norm_weight,
        const ds4_gpu_tensor *inv_freq,
        const ds4_gpu_tensor *d_pos, const ds4_gpu_tensor *d_dirty,
        const ds4_gpu_tensor *d_blk_cells,
        uint32_t n_tokens, uint32_t n_dirty, uint32_t cache_cap,
        uint32_t head_dim, uint32_t pool_size, uint32_t rot_dim,
        float eps, float weight_offset);
int ds4_gpu_qwen4exp_mm_indexer_scores(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *pool, const ds4_gpu_tensor *blk_filled,
        const ds4_gpu_tensor *row_pos,
        uint32_t n_tokens, uint32_t n_blocks, uint32_t n_head,
        uint32_t head_dim, uint32_t pool_size, uint32_t pos_hi);
int ds4_gpu_qwen4exp_mm_select(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *counts,
        const ds4_gpu_tensor *topk, const ds4_gpu_tensor *cell_pos,
        const ds4_gpu_tensor *cell_blk, const ds4_gpu_tensor *blk_filled,
        const ds4_gpu_tensor *row_pos,
        uint32_t n_tokens, uint32_t n_blocks, uint32_t top_k,
        uint32_t pool_size, uint32_t max_selected, uint32_t n_cells,
        uint32_t slot0);

#ifdef __cplusplus
}
#endif
#endif
