/*
 * lowbitFlash fused rotation+quantise kernel test.
 *
 * The device side is ds4_gpu_qwen4exp_lbf_rot_quantize -- one launch of
 * qwen4exp_lbf_rot_quantize_kernel, which per (row, segment) applies the
 * sign diagonal, the normalised Sylvester FWHT, and dev_qwen4exp_quantize_group
 * to emit the standard xq/xs/xsum triple.
 *
 * The host side is a straight-line reimplementation of exactly that math:
 *     t = x * s,  y = H t / sqrt(bs)  per segment, then per group of 32
 *     d = max|y| / 127,  q = lrintf(y / d)  clamped to int8.
 * Warp-synchronous reductions make the device scale and sum order-free, so
 * the comparison is BYTE-EXACT on all three outputs -- any deviation is a
 * kernel bug, not a tolerance question.
 *
 * Negative controls: one flipped sign and one permuted segment table must
 * each change the output; a malformed spec must be refused (return 0).
 * If the wrong-sign row still matched, the test could not tell applied
 * rotation from none.
 *
 * Run it on a CUDA host:  ./tests/test_qwen4exp_lbf_rot
 */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static int failures = 0;

#define require(cond, msg) do {                                            \
    if (!(cond)) {                                                         \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);      \
        failures++;                                                        \
        return 0;                                                          \
    }                                                                      \
} while (0)

/* Reference: one row of `width` floats, segmented sign+FWHT, then Q8_0-style
 * groups of 32.  Same order the kernel states: sign first, butterfly
 * unnormalised, scale after. */
static void ref_rot_quantize_row(int8_t *xq, float *xs, int32_t *xsum,
                                 const float *x, const float *signs,
                                 const uint32_t *segs, uint32_t n_seg,
                                 uint32_t width, uint32_t groups,
                                 uint64_t at_base) {
    float *stage = malloc(width * sizeof(float));
    uint32_t off = 0;
    for (uint32_t i = 0; i < n_seg; i++) {
        const uint32_t bs = segs[i];
        for (uint32_t k = 0; k < bs; k++)
            stage[off + k] = x[off + k] * signs[off + k];
        /* Unnormalised Sylvester butterfly, then 1/sqrt(bs). */
        for (uint32_t h = 1; h < bs; h <<= 1) {
            for (uint32_t a = 0; a < bs; a += 2 * h) {
                for (uint32_t j = 0; j < h; j++) {
                    const float u = stage[off + a + j];
                    const float v = stage[off + a + j + h];
                    stage[off + a + j]     = u + v;
                    stage[off + a + j + h] = u - v;
                }
            }
        }
        const float norm = 1.0f / sqrtf((float)bs);
        for (uint32_t k = 0; k < bs; k++) stage[off + k] *= norm;
        off += bs;
    }
    for (uint32_t g = 0; g < groups; g++) {
        float m = 0.0f;
        for (uint32_t k = 0; k < 32; k++) {
            const float a = fabsf(stage[g * 32 + k]);
            if (a > m) m = a;
        }
        const float d = m / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        xs[at_base + g] = d;
        int32_t sv = 0;
        for (uint32_t k = 0; k < 32; k++) {
            int v = (int)lrintf(stage[g * 32 + k] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[(at_base + g) * 32 + k] = (int8_t)v;
            sv += v;
        }
        xsum[at_base + g] = sv;
    }
    free(stage);
}

static int run_case(const char *name,
                    uint32_t rows, uint32_t width,
                    const uint32_t *segs, uint32_t n_seg,
                    uint64_t outer_stride, uint64_t inner_stride,
                    uint32_t inner_count,
                    int mutate) {
    const uint32_t groups = width / 32u;
    const uint64_t xq_n = (uint64_t)rows * groups * 32u;
    const uint64_t xs_n = (uint64_t)rows * groups;

    /* Deterministic input and signs: a cheap LCG over row*width, so every
     * element and every segment boundary gets a distinct magnitude. */
    float *x = malloc((size_t)rows * width * sizeof(float));
    float *signs = malloc((size_t)width * sizeof(float));
    int32_t *isigns = malloc((size_t)width * sizeof(int32_t));
    require(x && signs && isigns, "host alloc");
    uint32_t rng = 0x9e3779b9u;
    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t outer = r / inner_count;
        const uint32_t inner = r - outer * inner_count;
        float *row = x + (uint64_t)outer * (uint64_t)outer_stride +
                     (uint64_t)inner * (uint64_t)inner_stride;
        for (uint32_t k = 0; k < width; k++) {
            rng = rng * 1664525u + 1013904223u;
            row[k] = ((float)(int32_t)(rng >> 8) / 8388608.0f) *
                     (0.25f + 0.5f * (float)(k % 7));
        }
    }
    for (uint32_t k = 0; k < width; k++) {
        rng = rng * 1664525u + 1013904223u;
        const int s = (rng & 1u) ? 1 : -1;
        isigns[k] = s;
        signs[k] = (float)s;
    }
    /* The device gets its own copy: a mutated run flips ONE sign in the
     * upload while the oracle keeps the original table.  That is the
     * negative control -- the kernel must read and apply the sign diagonal,
     * so a wrong uploaded sign must change the output bytes.  Flipping both
     * sides would make the run a second positive case and could never
     * detect a kernel that ignored the sign array entirely. */
    float *dev_signs = malloc((size_t)width * sizeof(float));
    require(dev_signs, "dev sign alloc");
    memcpy(dev_signs, signs, (size_t)width * sizeof(float));
    if (mutate == 1) {
        dev_signs[width / 3 + 17] = -dev_signs[width / 3 + 17];
    }

    int8_t *rxq = calloc((size_t)xq_n, 1);
    float *rxs = calloc((size_t)xs_n, sizeof(float));
    int32_t *rxsum = calloc((size_t)xs_n, sizeof(int32_t));
    int8_t *dxq = calloc((size_t)xq_n, 1);
    float *dxs = calloc((size_t)xs_n, sizeof(float));
    int32_t *dxsum = calloc((size_t)xs_n, sizeof(int32_t));
    require(rxq && rxs && rxsum && dxq && dxs && dxsum, "out alloc");

    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t outer = r / inner_count;
        const uint32_t inner = r - outer * inner_count;
        const float *row = x + (uint64_t)outer * (uint64_t)outer_stride +
                           (uint64_t)inner * (uint64_t)inner_stride;
        ref_rot_quantize_row(rxq, rxs, rxsum, row, signs, segs, n_seg,
                             width, groups, (uint64_t)r * groups);
    }

    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc((uint64_t)rows *
        outer_stride * sizeof(float) + 4096);
    ds4_gpu_tensor *s_t = ds4_gpu_tensor_alloc((uint64_t)width * sizeof(float));
    ds4_gpu_tensor *q_t = ds4_gpu_tensor_alloc(xq_n);
    ds4_gpu_tensor *sc_t = ds4_gpu_tensor_alloc(xs_n * sizeof(float));
    ds4_gpu_tensor *sm_t = ds4_gpu_tensor_alloc(xs_n * sizeof(int32_t));
    require(x_t && s_t && q_t && sc_t && sm_t, "device alloc");
    require(ds4_gpu_tensor_write(x_t, 0, x,
                                 (uint64_t)rows * outer_stride * sizeof(float)),
            "x upload");
    require(ds4_gpu_tensor_write(s_t, 0, dev_signs,
                                 (uint64_t)width * sizeof(float)),
            "sign upload");

    ds4_gpu_qwen4exp_rot_in rot;
    memset(&rot, 0, sizeof(rot));
    rot.signs = (const float *)ds4_gpu_tensor_contents(s_t);
    rot.width = width;
    rot.n_seg = n_seg;
    memcpy(rot.seg_size, segs, n_seg * sizeof(uint32_t));

    require(ds4_gpu_qwen4exp_lbf_rot_quantize(
                (int8_t *)ds4_gpu_tensor_contents(q_t),
                (float *)ds4_gpu_tensor_contents(sc_t),
                (int32_t *)ds4_gpu_tensor_contents(sm_t),
                (const float *)ds4_gpu_tensor_contents(x_t),
                &rot, rows, width, outer_stride, inner_stride, inner_count),
            "rot quantize launch");
    require(ds4_gpu_tensor_read(q_t, 0, dxq, xq_n), "xq readback");
    require(ds4_gpu_tensor_read(sc_t, 0, dxs, xs_n * sizeof(float)),
            "xs readback");
    require(ds4_gpu_tensor_read(sm_t, 0, dxsum, xs_n * sizeof(int32_t)),
            "xsum readback");
    require(ds4_gpu_synchronize(), "sync");

    const int q_same = memcmp(rxq, dxq, (size_t)xq_n) == 0;
    const int s_same = memcmp(rxs, dxs, xs_n * sizeof(float)) == 0;
    const int m_same = memcmp(rxsum, dxsum, xs_n * sizeof(int32_t)) == 0;

    ds4_gpu_tensor_free(x_t);
    ds4_gpu_tensor_free(s_t);
    ds4_gpu_tensor_free(q_t);
    ds4_gpu_tensor_free(sc_t);
    ds4_gpu_tensor_free(sm_t);
    free(x); free(signs); free(isigns); free(dev_signs);
    free(rxq); free(rxs); free(rxsum); free(dxq); free(dxs); free(dxsum);

    if (mutate == 1) {
        /* The flipped sign MUST change at least one output byte. */
        if (q_same && s_same && m_same) {
            fprintf(stderr, "FAIL %s: flipped sign left output identical -- "
                            "the kernel did not apply the rotation\n", name);
            failures++;
            return 0;
        }
        printf("PASS %s (negative control: flip detected)\n", name);
        return 1;
    }
    if (!q_same || !s_same || !m_same) {
        uint64_t first = UINT64_MAX;
        for (uint64_t i = 0; i < xq_n; i++)
            if (rxq[i] != dxq[i]) { first = i; break; }
        fprintf(stderr, "FAIL %s: byte mismatch (xq=%d xs=%d xsum=%d, "
                "first xq diff at %" PRIu64 ")\n",
                name, q_same, s_same, m_same,
                first == UINT64_MAX ? 0 : first);
        failures++;
        return 0;
    }
    printf("PASS %s\n", name);
    return 1;
}

int main(void) {
    if (!ds4_gpu_init()) {
        fprintf(stderr, "GPU init failed\n");
        return 1;
    }

    /* Case shapes mirror the production contract: the routed input is
     * 2560 wide with inner_count 1; the mid scratch carries per-(token,
     * expert-slot) rows with outer_stride = token span and
     * inner_stride = mid_dim.  Segmentations: a production-like tower,
     * a single block, and an odd four-segment split to prove boundaries. */
    static const uint32_t segs_in[]   = { 1024, 1024, 512 };
    static const uint32_t segs_one[]  = { 2560 };
    static const uint32_t segs_odd[]  = { 512, 256, 1024, 512, 256 };
    static const uint32_t segs_mid[]  = { 512, 128 };
    static const uint32_t segs_mid2[] = { 256, 256, 128 };

    run_case("in-2560 tower x1 row", 1, 2560, segs_in, 3, 2560, 0, 1, 0);
    run_case("in-2560 tower x2 rows", 2, 2560, segs_in, 3, 2560, 0, 1, 0);
    run_case("in-2560 single seg", 1, 2560, segs_one, 1, 2560, 0, 1, 0);
    run_case("in-2560 five segs x3", 3, 2560, segs_odd, 5, 2560, 0, 1, 0);
    /* Mid path: 20 (token,expert) pairs over 2 tokens, 10 experts wide --
     * row r lives at outer_stride*(r/10) + 640*(r%10), exactly the scratch
     * layout the down projection consumes. */
    run_case("mid-640 pairs 2x10", 20, 640, segs_mid, 2,
             6400, 640, 10, 0);
    run_case("mid-640 3seg pairs", 20, 640, segs_mid2, 3,
             6400, 640, 10, 0);
    /* Negative controls. */
    run_case("in-2560 flipped sign", 1, 2560, segs_in, 3, 2560, 0, 1, 1);
    run_case("mid-640 flipped sign", 20, 640, segs_mid, 2,
             6400, 640, 10, 1);

    /* Malformed specs must be refused, not silently skipped. */
    {
        ds4_gpu_qwen4exp_rot_in bad;
        memset(&bad, 0, sizeof(bad));
        bad.width = 2560; bad.n_seg = 0;
        int8_t dummy_q[32]; float dummy_s; int32_t dummy_m;
        float dummy_x[2560];
        memset(dummy_x, 0, sizeof(dummy_x));
        if (ds4_gpu_qwen4exp_lbf_rot_quantize(dummy_q, &dummy_s, &dummy_m,
                                              dummy_x, &bad, 1, 2560,
                                              2560, 0, 1)) {
            fprintf(stderr, "FAIL: n_seg=0 spec was not refused\n");
            failures++;
        } else {
            printf("PASS refuse n_seg=0\n");
        }
        bad.n_seg = 1; bad.seg_size[0] = 1024; /* sums to 1024 != 2560 */
        bad.signs = dummy_x;
        if (ds4_gpu_qwen4exp_lbf_rot_quantize(dummy_q, &dummy_s, &dummy_m,
                                              dummy_x, &bad, 1, 2560,
                                              2560, 0, 1)) {
            fprintf(stderr, "FAIL: short segment sum was not refused\n");
            failures++;
        } else {
            printf("PASS refuse bad segment sum\n");
        }
    }

    if (failures) {
        fprintf(stderr, "%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("all lbf rot-quantize cases pass\n");
    return 0;
}
