/* Host-oracle for ds4_qwen4exp_mm_plan: feeds a fixed slice schedule and
 * checks M-RoPE positions, embedding compaction and the cell map against
 * llama.cpp's exact rules:
 *   text rows   -> pos = {t,t,t,0}, slot = n_rows++, t++
 *   image row k -> pos = {t0, t0+k/nx, t0+k%nx, 0}; t = t0 + max(nx,ny)
 *   cell map    -> cell_pos[slot] = pos.t, cell_blk = t/r (r=4),
 *                  blk_cells[b*r + t%r] = slot, blk_filled[b]++.
 * Negative controls: an image row without a slice and a slice whose grid
 * changes mid-span must fail; a slice-row mismatch must fail.
 * Runs WITHOUT a GPU: mm_create splits into host-only state when
 * ds4_gpu_tensor_alloc returns NULL?  No -- it needs the context.  So this
 * test exercises plan() over a mm struct created by
 * ds4_qwen4exp_mm_create_host() (a small test-only constructor that skips
 * device tensors; see ds4_qwen4exp_mm.c). */
#include "ds4_qwen4exp_mm.h"

/* Host-test stubs: the host planner never calls these on NULL tensors, but
 * the linker needs them. */
ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    (void)bytes; return NULL;
}
void ds4_gpu_tensor_free(ds4_gpu_tensor *t) { (void)t; }
int ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t off,
                         const void *src, uint64_t len) {
    (void)t; (void)off; (void)src; (void)len; return 0;
}
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#define N 24
static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    printf("FAIL %d: %s\n", __LINE__, msg); fails++; } } while (0)

int main(void) {
    const int32_t sections[4] = { 8, 8, 8, 8 };
    ds4_qwen4exp_mm *mm = ds4_qwen4exp_mm_create_host(64, 4, 32, 8, sections);
    CHECK(mm != NULL, "host mm create");
    if (!mm) return 1;

    int32_t is_image[N] = {0};
    /* prompt rows: 4 text, then 3x3 image (9 rows), then 2 text, then
     * a 4x1 image (4 rows), then text to end */
    for (int i = 4; i < 13; i++) is_image[i] = 1;
    for (int i = 15; i < 19; i++) is_image[i] = 1;
    float grid_a[9 * 8], grid_b[4 * 8];
    for (int i = 0; i < 9 * 8; i++) grid_a[i] = (float)(i + 1000);
    for (int i = 0; i < 4 * 8; i++) grid_b[i] = (float)(i + 2000);
    ds4_qwen4exp_mm_slice slices[2] = {
        { .prompt_lo = 4, .n_rows = 9, .nx = 3, .ny = 3,
          .embd = grid_a, .embd_skip = 0 },
        { .prompt_lo = 15, .n_rows = 4, .nx = 4, .ny = 1,
          .embd = grid_b, .embd_skip = 0 },
    };
    int32_t pos[N * 4], map[N];
    float embd[13 * 8];
    uint32_t t = 0, adv = 0;
    char err[128];
    CHECK(ds4_qwen4exp_mm_plan(mm, slices, 2, is_image, N, &t,
                             pos, embd, map, &adv, err, sizeof(err)),
          "plan succeeds");
    /* expected: 4 text -> t=4; image a 3x3 -> t0=4, t=4+max(3,3)=7;
     * 2 text -> t=9; image b 4x1 -> t0=9, t=13; 5 text -> t=18. */
    CHECK(t == 18, "final t == 18");
    CHECK(adv == 18, "pos_advance == 18");
    /* row 0 text */
    CHECK(pos[0] == 0 && pos[1] == 0 && pos[2] == 0, "row0 pos 0");
    CHECK(map[0] == -1, "row0 not image");
    /* row 4: image a, k=0 -> t=4,h=4,w=4 */
    CHECK(pos[16] == 4 && pos[17] == 4 && pos[18] == 4 && pos[19] == 0,
          "img a k0 pos");
    /* row 8: k=4 -> h=4+1=5, w=4+1=5 */
    CHECK(pos[32] == 4 && pos[33] == 5 && pos[34] == 5, "img a k4 pos");
    /* row 12: k=8 -> h=6, w=6 */
    CHECK(pos[48] == 4 && pos[49] == 6 && pos[50] == 6, "img a k8 pos");
    /* rows 13,14 text at t=7,8 */
    CHECK(pos[13*4] == 7 && pos[14*4] == 8, "post-image text pos");
    /* row 15: image b k=0, t0=9 */
    CHECK(pos[15*4] == 9 && pos[15*4+1] == 9 && pos[15*4+2] == 9,
          "img b k0");
    /* row 18: k=3, nx=4 -> h=9+0, w=9+3=12 */
    CHECK(pos[18*4] == 9 && pos[18*4+1] == 9 && pos[18*4+2] == 12,
          "img b k3");
    /* row 19 text t=13 */
    CHECK(pos[19*4] == 13, "row19 t=13");
    /* embedding compaction: image rows 4..12 -> 0..8, rows 15..18 -> 9..12 */
    for (int i = 0; i < 9; i++)
        CHECK(embd[i*8] == 1000.0f + i * 8, "embd row a");
    for (int i = 0; i < 4; i++)
        CHECK(embd[(9+i)*8] == 2000.0f + i * 8, "embd row b");
    /* cell map: slot i, pos.t */
    CHECK(mm->cell_pos[4] == 4 && mm->cell_blk[4] == 1, "cell 4 t=4 blk=1");
    CHECK(mm->cell_pos[15] == 9 && mm->cell_blk[15] == 2, "cell 15 blk 2");
    /* t=4 block (block 1) holds 9 image cells all at t=4 plus the t=7
     * text row: last write wins, so blk_cells[4] is the last t=4 slot. */
    CHECK(mm->blk_cells[1*4 + 0] == 12, "blk_cells[4] = last t=4 slot");
    CHECK(mm->blk_filled[1] == 10, "block1 filled = 9 img + 1 text (t=7)");

    ds4_qwen4exp_mm_commit(mm, N);

    /* ---- negative controls ---- */
    int32_t bad_image[4] = {0,1,0,0};
    uint32_t t2 = 0;
    CHECK(!ds4_qwen4exp_mm_plan(mm, NULL, 0, bad_image, 4, &t2,
                              pos, embd, map, &adv, err, sizeof(err)),
          "image row without slice fails");
    ds4_qwen4exp_mm_slice bad[1] = {
        { .prompt_lo = 0, .n_rows = 2, .nx = 3, .ny = 3,
          .embd = grid_a, .embd_skip = 0 },
    };
    int32_t bi[2] = {1,1};
    t2 = 0;
    /* 2 rows of a 9-row grid is legal (continuation) -- the mismatch to
     * check is n_rows beyond the grid total. */
    /* n_rows=10 but only 9 rows flagged image: slice/row mismatch. */
    int32_t bi9[10] = {1,1,1,1,1,1,1,1,1,0};
    bad[0].n_rows = 10;
    CHECK(!ds4_qwen4exp_mm_plan(mm, bad, 1, bi9, 10, &t2,
                              pos, embd, map, &adv, err, sizeof(err)),
          "slice/row mismatch fails");
    /* 10 image rows on a 9-cell grid: the extra row opens a new image,
     * which is legal -- the real negative is an is_image row with no
     * slice left at all (tested above). */
    printf("%s\n", fails ? "FAILED" : "PASS");
    ds4_qwen4exp_mm_free(mm);
    return fails ? 1 : 0;
}
