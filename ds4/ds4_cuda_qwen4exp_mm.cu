/*
 * Qwen4exp multimodal kernels -- the per-row-position variants the M-RoPE
 * (IMROPE) image path needs, mirroring llama.cpp's ggml_rope_multi +
 * llama_memory_hybrid_idx cell-map semantics.
 *
 * POSITION vs SLOT.  Every kernel below takes both where the distinction
 * matters:
 *
 *   - "slot" (d_pos / slot0 + token) is the dense cell index: K/V cache
 *     rows, the indexer tape, and the attention selected[] entries are all
 *     cell ids.
 *   - "row_pos" (4 x i32 per row: t, h, w, unused) is the M-RoPE position
 *     the attention Q/K and the indexer query rotate by, and t is also the
 *     logical position the indexer cell map keys on.
 *
 * Axis rule (llama.cpp ggml-cuda rope.cu rope_multi, is_imrope): for pair
 * sector s = (iw/2) % sect_dims with sect_dims = s0+s1+s2+s3:
 *     s%3==0 && s < 3*s0 -> pos plane 0 (t)
 *     s%3==1 && s < 3*s1 -> pos plane 1 (h)
 *     s%3==2 && s < 3*s2 -> pos plane 2 (w)
 *     else               -> pos plane 3 (unused)
 * The pair (d, d+rot_half) rotates by theta = pos[axis] * inv_freq[d],
 * matching the text kernels' layout.
 *
 * Cell map (llama.cpp set_input_qsa): blk_cells[b*r + p%r] = cell, last
 * write wins; blk_filled[b] counts WRITES (not unique positions); the
 * pool gathers mean(tape[blk_cells[b]]) over all r slots even when a
 * block is short (unwritten slots hold cell 0); a block is scoreable iff
 * blk_filled[b] >= r or it is the query's tail block (+1e9 bias).
 */
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "ds4_cuda_qwen4exp.cuh"
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define MM_STREAM ds4_cuda_qwen4exp_decode_stream()

static int mm_cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "ds4: qwen4exp-mm %s failed: %s\n", what,
            cudaGetErrorString(err));
    return 0;
}
static int mm_tensor_has(const ds4_gpu_tensor *t, uint64_t elems,
                         uint64_t esz) {
    return t && t->ptr && (uint64_t)t->bytes >= elems * esz;
}
static uint32_t mm_threads(uint32_t value) {
    uint32_t nth = 1;
    while (nth * 2u <= value && nth * 2u <= 1024u) nth *= 2;
    return nth;
}
static __device__ float mm_blk_sum(float *shared, uint32_t tid,
                                   uint32_t nth) {
    for (uint32_t s = nth >> 1; s > 0; s >>= 1) {
        if (tid + s < nth) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    return shared[0];
}
/* M-RoPE axis + angle for pair index d of a row. */
static __device__ float mm_theta(const int32_t *row_pos, uint32_t token,
                                 uint32_t d, const float *inv_freq,
                                 const int32_t *sections) {
    const int32_t sect_dims =
        sections[0] + sections[1] + sections[2] + sections[3];
    const int32_t s = (int32_t)d % sect_dims;
    int axis;
    if (s % 3 == 1 && s < 3 * sections[1])      axis = 1; /* h */
    else if (s % 3 == 2 && s < 3 * sections[2]) axis = 2; /* w */
    else if (s % 3 == 0 && s < 3 * sections[0]) axis = 0; /* t */
    else                                      axis = 3;
    return (float)row_pos[(size_t)token * 4u + axis] * inv_freq[d];
}

/* ---------- embedding rows ------------------------------------------------
 * hyper[row][h][d] = embd[embd_map[row]][d] for image rows; token rows are
 * left at whatever the ordinary gather wrote. */
__global__ static void mm_embd_rows_kernel(
        float *hyper, const float *embd, const int32_t *embd_map,
        uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t per_row = (uint64_t)n_hc * n_embd;
    if (gid >= (uint64_t)n_tokens * per_row) return;
    const uint32_t token = (uint32_t)(gid / per_row);
    const int32_t src = embd_map[token];
    if (src < 0) return;
    const uint32_t d = (uint32_t)(gid % n_embd);
    hyper[gid] = embd[(uint64_t)src * n_embd + d];
}
extern "C" int ds4_gpu_qwen4exp_mm_embd_rows(
        ds4_gpu_tensor *hyper, const ds4_gpu_tensor *embd,
        const ds4_gpu_tensor *embd_map, uint32_t n_tokens, uint32_t n_embd,
        uint32_t n_hc, uint32_t row_cap) {
    if (!n_tokens || !n_embd || !n_hc ||
        !mm_tensor_has(hyper, (uint64_t)n_tokens * n_hc * n_embd, 4u) ||
        !mm_tensor_has(embd_map, n_tokens, 4u) ||
        !mm_tensor_has(embd, (uint64_t)row_cap * n_embd, 4u)) return 0;
    const uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    mm_embd_rows_kernel<<<(unsigned)((n + 255u) / 256u), 256u, 0,
        MM_STREAM>>>((float *)hyper->ptr, (const float *)embd->ptr,
                     (const int32_t *)embd_map->ptr, n_tokens, n_embd,
                     n_hc);
    return mm_cuda_ok(cudaGetLastError(), "mm embed rows");
}

/* ---------- Q prep (norm + mrope) -----------------------------------------
 * Same arithmetic as qwen4exp_qsa_prep_joint_kernel<0|3>: per-head RMS norm
 * of the first head_dim of each (q|gate) half, then rope.  Positions come
 * from row_pos instead of p0+token. */
template<int WithGate>
__global__ static void mm_prep_q_kernel(
        const float *doubled, const float *qw, const float *inv_freq,
        float *q_out, float *gate_out,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t rot_dim, float eps, float q_offset,
        const int32_t *row_pos, const int32_t *sections) {
    extern __shared__ float mm_prep_shared[];
    const uint32_t token = blockIdx.y, tid = threadIdx.x, nth = blockDim.x;
    const uint32_t head = blockIdx.x;
    if (head >= n_head || token >= n_tokens) return;
    const uint32_t width = n_head * head_dim;
    const uint64_t at = (uint64_t)token * width + head * head_dim + tid;
    const uint64_t src =
        (uint64_t)token * 2u * width + head * 2u * head_dim + tid;
    float raw = 0.0f;
    if (tid < head_dim) {
        raw = doubled[src];
        if (WithGate) gate_out[at] = doubled[src + head_dim];
    }
    mm_prep_shared[tid] = tid < head_dim ? raw * raw : 0.0f;
    __syncthreads();
    const float sum = mm_blk_sum(mm_prep_shared, tid, nth);
    const float inv = rsqrtf(sum / (float)head_dim + eps);
    __syncthreads();
    if (tid < head_dim)
        mm_prep_shared[tid] = raw * inv * (q_offset + qw[tid]);
    __syncthreads();
    const uint32_t half = rot_dim / 2u;
    if (tid < half) {
        const float theta = mm_theta(row_pos, token, tid, inv_freq,
                                     sections);
        const float c = cosf(theta), s = sinf(theta);
        const float x1 = mm_prep_shared[tid], x2 = mm_prep_shared[tid + half];
        mm_prep_shared[tid] = x1 * c - x2 * s;
        mm_prep_shared[tid + half] = x2 * c + x1 * s;
    }
    __syncthreads();
    if (tid < head_dim) q_out[at] = mm_prep_shared[tid];
}
extern "C" int ds4_gpu_qwen4exp_mm_prep_q(
        ds4_gpu_tensor *q, ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *doubled, const ds4_gpu_tensor *weight,
        const ds4_gpu_tensor *inv_freq, const ds4_gpu_tensor *row_pos,
        const ds4_gpu_tensor *sections,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t rot_dim, float eps, float weight_offset) {
    if (!n_tokens || !n_head || !head_dim || !rot_dim ||
        rot_dim > head_dim || (rot_dim % 2u) ||
        !mm_tensor_has(q, (uint64_t)n_tokens * n_head * head_dim, 4u) ||
        !mm_tensor_has(doubled, 2u * (uint64_t)n_tokens * n_head * head_dim,
                       4u) ||
        !mm_tensor_has(weight, head_dim, 4u) ||
        !mm_tensor_has(inv_freq, rot_dim / 2u, 4u) ||
        !mm_tensor_has(row_pos, (uint64_t)n_tokens * 4u, 4u) ||
        !mm_tensor_has(sections, 4u, 4u)) return 0;
    const dim3 grid(n_head, n_tokens);
    const uint32_t nth = mm_threads(head_dim);
    if (gate) {
        mm_prep_q_kernel<1><<<grid, nth, nth * sizeof(float), MM_STREAM>>>(
            (const float *)doubled->ptr, (const float *)weight->ptr,
            (const float *)inv_freq->ptr, (float *)q->ptr,
            (float *)gate->ptr, n_tokens, n_head, head_dim, rot_dim, eps,
            weight_offset, (const int32_t *)row_pos->ptr,
            (const int32_t *)sections->ptr);
    } else {
        mm_prep_q_kernel<0><<<grid, nth, nth * sizeof(float), MM_STREAM>>>(
            (const float *)doubled->ptr, (const float *)weight->ptr,
            (const float *)inv_freq->ptr, (float *)q->ptr, NULL,
            n_tokens, n_head, head_dim, rot_dim, eps, weight_offset,
            (const int32_t *)row_pos->ptr, (const int32_t *)sections->ptr);
    }
    return mm_cuda_ok(cudaGetLastError(), "mm q prep");
}

/* ---------- KV prep: norm + mrope on K, V stored raw ----------------------
 * Same arithmetic as the KV half of the joint kernel; cache row is the
 * SLOT (d_pos + token), rope position is row_pos. */
__global__ static void mm_prep_kv_kernel(
        const float *raw_k, const float *raw_v, const float *kw,
        const float *inv_freq, float *k_cache, float *v_cache,
        uint32_t n_tokens, uint32_t n_head_kv, uint32_t head_dim,
        uint32_t rot_dim, uint32_t cache_cap, float eps, float k_offset,
        const int32_t *row_pos, const int32_t *sections,
        const uint32_t *d_pos) {
    extern __shared__ float mm_kv_shared[];
    const uint32_t token = blockIdx.y, tid = threadIdx.x, nth = blockDim.x;
    const uint32_t head = blockIdx.x;
    if (head >= n_head_kv || token >= n_tokens) return;
    const uint32_t width = n_head_kv * head_dim;
    const uint64_t at = (uint64_t)token * width + head * head_dim + tid;
    const uint32_t slot = (d_pos ? *d_pos : 0u) + token;
    float raw = 0.0f;
    if (tid < head_dim) {
        raw = raw_k[at];
        if (slot < cache_cap)
            v_cache[(uint64_t)slot * width + head * head_dim + tid] =
                raw_v[at];
    }
    mm_kv_shared[tid] = tid < head_dim ? raw * raw : 0.0f;
    __syncthreads();
    const float sum = mm_blk_sum(mm_kv_shared, tid, nth);
    const float inv = rsqrtf(sum / (float)head_dim + eps);
    __syncthreads();
    if (tid < head_dim)
        mm_kv_shared[tid] = raw * inv * (k_offset + kw[tid]);
    __syncthreads();
    const uint32_t half = rot_dim / 2u;
    if (tid < half) {
        const float theta = mm_theta(row_pos, token, tid, inv_freq,
                                     sections);
        const float c = cosf(theta), s = sinf(theta);
        const float x1 = mm_kv_shared[tid], x2 = mm_kv_shared[tid + half];
        mm_kv_shared[tid] = x1 * c - x2 * s;
        mm_kv_shared[tid + half] = x2 * c + x1 * s;
    }
    __syncthreads();
    if (tid < head_dim && slot < cache_cap)
        k_cache[(uint64_t)slot * width + head * head_dim + tid] =
            mm_kv_shared[tid];
}
extern "C" int ds4_gpu_qwen4exp_mm_prep_kv(
        ds4_gpu_tensor *k_cache, ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *raw_k, const ds4_gpu_tensor *raw_v,
        const ds4_gpu_tensor *weight, const ds4_gpu_tensor *inv_freq,
        const ds4_gpu_tensor *row_pos, const ds4_gpu_tensor *sections,
        const ds4_gpu_tensor *d_pos,
        uint32_t n_tokens, uint32_t n_head_kv, uint32_t head_dim,
        uint32_t rot_dim, uint32_t cache_cap, float eps,
        float weight_offset) {
    if (!n_tokens || !n_head_kv || !head_dim || !rot_dim ||
        rot_dim > head_dim || (rot_dim % 2u) ||
        !mm_tensor_has(raw_k, (uint64_t)n_tokens * n_head_kv * head_dim,
                       4u) ||
        !mm_tensor_has(raw_v, (uint64_t)n_tokens * n_head_kv * head_dim,
                       4u) ||
        !mm_tensor_has(k_cache, (uint64_t)cache_cap * n_head_kv * head_dim,
                       4u) ||
        !mm_tensor_has(v_cache, (uint64_t)cache_cap * n_head_kv * head_dim,
                       4u) ||
        !mm_tensor_has(weight, head_dim, 4u) ||
        !mm_tensor_has(inv_freq, rot_dim / 2u, 4u) ||
        !mm_tensor_has(row_pos, (uint64_t)n_tokens * 4u, 4u) ||
        !mm_tensor_has(sections, 4u, 4u)) return 0;
    const dim3 grid(n_head_kv, n_tokens);
    const uint32_t nth = mm_threads(head_dim);
    mm_prep_kv_kernel<<<grid, nth, nth * sizeof(float), MM_STREAM>>>(
        (const float *)raw_k->ptr, (const float *)raw_v->ptr,
        (const float *)weight->ptr, (const float *)inv_freq->ptr,
        (float *)k_cache->ptr, (float *)v_cache->ptr, n_tokens, n_head_kv,
        head_dim, rot_dim, cache_cap, eps, weight_offset,
        (const int32_t *)row_pos->ptr, (const int32_t *)sections->ptr,
        d_pos ? (const uint32_t *)d_pos->ptr : NULL);
    return mm_cuda_ok(cudaGetLastError(), "mm kv prep");
}

/* ---------- indexer query rope ------------------------------------------- */
__global__ static void mm_rope_idxq_kernel(
        float *x, const float *inv_freq, uint32_t n_tokens, uint32_t n_head,
        uint32_t head_dim, uint32_t rot_dim,
        const int32_t *row_pos, const int32_t *sections) {
    const uint32_t rot_half = rot_dim / 2u;
    const uint32_t per_token = n_head * rot_half;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (uint64_t)n_tokens * per_token) return;
    const uint32_t token = (uint32_t)(gid / per_token);
    const uint32_t lane = (uint32_t)(gid - (uint64_t)token * per_token);
    const uint32_t head = lane / rot_half;
    const uint32_t d = lane % rot_half;
    float *vec = x + ((uint64_t)token * n_head + head) * head_dim;
    const float theta = mm_theta(row_pos, token, d, inv_freq, sections);
    const float c = cosf(theta), s = sinf(theta);
    const float x1 = vec[d], x2 = vec[d + rot_half];
    vec[d] = x1 * c - x2 * s;
    vec[d + rot_half] = x2 * c + x1 * s;
}
extern "C" int ds4_gpu_qwen4exp_mm_rope_idxq(
        ds4_gpu_tensor *idx_q, const ds4_gpu_tensor *inv_freq,
        const ds4_gpu_tensor *row_pos, const ds4_gpu_tensor *sections,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t rot_dim) {
    if (!n_tokens || !n_head || !head_dim || !rot_dim ||
        !mm_tensor_has(idx_q, (uint64_t)n_tokens * n_head * head_dim, 4u) ||
        !mm_tensor_has(inv_freq, rot_dim / 2u, 4u) ||
        !mm_tensor_has(row_pos, (uint64_t)n_tokens * 4u, 4u) ||
        !mm_tensor_has(sections, 4u, 4u)) return 0;
    const uint64_t n = (uint64_t)n_tokens * n_head * (rot_dim / 2u);
    mm_rope_idxq_kernel<<<(unsigned)((n + 255u) / 256u), 256u, 0,
        MM_STREAM>>>((float *)idx_q->ptr, (const float *)inv_freq->ptr,
                     n_tokens, n_head, head_dim, rot_dim,
                     (const int32_t *)row_pos->ptr,
                     (const int32_t *)sections->ptr);
    return mm_cuda_ok(cudaGetLastError(), "mm indexer q rope");
}

/* ---------- tape append ----------------------------------------------------
 * tape[slot] = raw indexer K row; slot = d_pos + token.  (Cell-indexed,
 * not position-indexed: that is the whole difference from the text
 * kernel.) */
__global__ static void mm_tape_append_kernel(
        const float *raw_k, float *tape, uint32_t n_tokens,
        uint32_t head_dim, uint32_t cache_cap, const uint32_t *d_pos) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (uint64_t)n_tokens * head_dim) return;
    const uint32_t token = (uint32_t)(gid / head_dim);
    const uint32_t d = (uint32_t)(gid - (uint64_t)token * head_dim);
    const uint32_t slot = *d_pos + token;
    if (slot >= cache_cap) return;
    tape[(uint64_t)slot * head_dim + d] = raw_k[gid];
}
extern "C" int ds4_gpu_qwen4exp_mm_tape_append(
        ds4_gpu_tensor *tape, const ds4_gpu_tensor *raw_k,
        const ds4_gpu_tensor *d_pos,
        uint32_t n_tokens, uint32_t head_dim, uint32_t cache_cap) {
    if (!n_tokens || !head_dim || !d_pos ||
        !mm_tensor_has(tape, (uint64_t)cache_cap * head_dim, 4u) ||
        !mm_tensor_has(raw_k, (uint64_t)n_tokens * head_dim, 4u)) return 0;
    const uint64_t n = (uint64_t)n_tokens * head_dim;
    mm_tape_append_kernel<<<(unsigned)((n + 255u) / 256u), 256u, 0,
        MM_STREAM>>>((const float *)raw_k->ptr, (float *)tape->ptr,
                     n_tokens, head_dim, cache_cap,
                     (const uint32_t *)d_pos->ptr);
    return mm_cuda_ok(cudaGetLastError(), "mm tape append");
}

/* ---------- pooled-block recompute -----------------------------------------
 * For every dirty block b:  vec = mean_s tape[blk_cells[b*r + s]]
 *                           vec = rms_norm(vec) * (w + offset)
 *                           rope at position b*r (all four planes equal)
 *                           pool[b] = vec
 * This is exactly llama.cpp's pooled update; unwritten member slots hold
 * cell 0 by construction, matching the reference.  Positions in blk_cells
 * are POSITIONS; pool is roped at the block's first position. */
__global__ static void mm_pool_kernel(
        float *tape, const float *weight, const float *inv_freq,
        float *pool, const int32_t *dirty, uint32_t n_dirty,
        const int32_t *blk_cells,
        uint32_t head_dim, uint32_t pool_size, uint32_t rot_dim,
        uint32_t cache_cap, float eps, float weight_offset) {
    extern __shared__ float mm_pool_shared[];
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    if (blockIdx.x >= n_dirty) return;
    const int32_t bid = dirty[blockIdx.x];
    if (bid < 0) return;
    const uint32_t block = (uint32_t)bid;
    if ((uint64_t)(block + 1u) * pool_size > cache_cap) return;
    float *vec = mm_pool_shared;
    float *scratch = mm_pool_shared + head_dim;
    for (uint32_t d = tid; d < head_dim; d += nth) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < pool_size; s++) {
            const int32_t cell = blk_cells[block * pool_size + s];
            const uint32_t c = cell < 0 ? 0u : (uint32_t)cell;
            acc += tape[(uint64_t)c * head_dim + d];
        }
        vec[d] = acc / (float)pool_size;
    }
    __syncthreads();
    float partial = 0.0f;
    for (uint32_t d = tid; d < head_dim; d += nth)
        partial += vec[d] * vec[d];
    scratch[tid] = partial;
    const float sum = mm_blk_sum(scratch, tid, nth);
    const float inv = rsqrtf(sum / (float)head_dim + eps);
    for (uint32_t d = tid; d < head_dim; d += nth)
        vec[d] = vec[d] * inv * (weight_offset + weight[d]);
    __syncthreads();
    const uint32_t half = rot_dim / 2u;
    if (tid < half) {
        const float theta = (float)(block * pool_size) * inv_freq[tid];
        const float c = cosf(theta), s = sinf(theta);
        const float x1 = vec[tid], x2 = vec[tid + half];
        vec[tid] = x1 * c - x2 * s;
        vec[tid + half] = x2 * c + x1 * s;
    }
    __syncthreads();
    for (uint32_t d = tid; d < head_dim; d += nth)
        pool[(uint64_t)block * head_dim + d] = vec[d];
}
extern "C" int ds4_gpu_qwen4exp_mm_pool_update(
        ds4_gpu_tensor *pool, ds4_gpu_tensor *tape,
        const ds4_gpu_tensor *raw_k, const ds4_gpu_tensor *k_norm_weight,
        const ds4_gpu_tensor *inv_freq,
        const ds4_gpu_tensor *d_pos, const ds4_gpu_tensor *d_dirty,
        const ds4_gpu_tensor *d_blk_cells,
        uint32_t n_tokens, uint32_t n_dirty, uint32_t cache_cap,
        uint32_t head_dim, uint32_t pool_size, uint32_t rot_dim,
        float eps, float weight_offset) {
    if (!n_tokens || !head_dim || !pool_size || !rot_dim ||
        rot_dim > head_dim || !d_pos ||
        !mm_tensor_has(tape, (uint64_t)cache_cap * head_dim, 4u) ||
        !mm_tensor_has(pool, (uint64_t)(cache_cap / pool_size) * head_dim,
                       4u) ||
        !mm_tensor_has(raw_k, (uint64_t)n_tokens * head_dim, 4u) ||
        !mm_tensor_has(k_norm_weight, head_dim, 4u) ||
        !mm_tensor_has(inv_freq, rot_dim / 2u, 4u) ||
        !mm_tensor_has(d_blk_cells, (uint64_t)cache_cap, 4u)) return 0;
    const uint64_t n = (uint64_t)n_tokens * head_dim;
    mm_tape_append_kernel<<<(unsigned)((n + 255u) / 256u), 256u, 0,
        MM_STREAM>>>((const float *)raw_k->ptr, (float *)tape->ptr,
                     n_tokens, head_dim, cache_cap,
                     (const uint32_t *)d_pos->ptr);
    if (!mm_cuda_ok(cudaGetLastError(), "mm tape append")) return 0;
    if (n_dirty) {
        if (!mm_tensor_has(d_dirty, n_dirty, 4u)) return 0;
        const uint32_t nth = mm_threads(head_dim);
        mm_pool_kernel<<<n_dirty, nth,
            (size_t)(head_dim + nth) * sizeof(float), MM_STREAM>>>(
                (float *)tape->ptr, (const float *)k_norm_weight->ptr,
                (const float *)inv_freq->ptr, (float *)pool->ptr,
                (const int32_t *)d_dirty->ptr, n_dirty,
                (const int32_t *)d_blk_cells->ptr,
                head_dim, pool_size, rot_dim, cache_cap, eps,
                weight_offset);
    }
    return mm_cuda_ok(cudaGetLastError(), "mm pool update");
}

/* ---------- block scores with mm bias --------------------------------------
 * scores[i][b] = sum_h relu(q[i][h] . pool[b]) / sqrt(head_dim)
 *              + (b*r >= tail_start(i) ? 1e9
 *                 : blk_filled[b] < r ? -inf : 0)
 * tail_start(i) = (t_i + 1)/r*r with t_i the row's logical position.
 * Blocks whose whole range lies past pos_hi (no populated cell can be a
 * member anyway) get -inf. */
__global__ static void mm_indexer_scores_kernel(
        const float *q, const float *pool, const int32_t *blk_filled,
        float *scores, uint32_t n_tokens, uint32_t n_blocks,
        uint32_t n_head, uint32_t head_dim, uint32_t pool_size,
        uint32_t pos_hi, const int32_t *row_pos, float scale) {
    extern __shared__ float mm_sc_shared[];
    const uint32_t block = blockIdx.x, token = blockIdx.y;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    if (token >= n_tokens || block >= n_blocks) return;
    const uint32_t t_i = (uint32_t)row_pos[(size_t)token * 4u];
    const uint32_t tail_start = ((t_i + 1u) / pool_size) * pool_size;
    float bias;
    if ((uint64_t)block * pool_size >= tail_start &&
        block * pool_size <= pos_hi) {
        bias = 1e9f;
    } else if ((uint64_t)block * pool_size > pos_hi ||
               blk_filled[block] < (int32_t)pool_size) {
        bias = -INFINITY;
    } else {
        bias = 0.0f;
    }
    if (bias == -INFINITY) {
        if (tid == 0)
            scores[(uint64_t)token * n_blocks + block] = -INFINITY;
        return;
    }
    const float *pb = pool + (uint64_t)block * head_dim;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qv = q + ((uint64_t)token * n_head + h) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = tid; d < head_dim; d += nth)
            dot += qv[d] * pb[d];
        mm_sc_shared[tid] = dot;
        __syncthreads();
        const float s = mm_blk_sum(mm_sc_shared, tid, nth);
        acc += fmaxf(s, 0.0f);
        __syncthreads();
    }
    if (tid == 0)
        scores[(uint64_t)token * n_blocks + block] =
            acc * scale + (bias == 1e9f ? 1e9f : 0.0f);
}
extern "C" int ds4_gpu_qwen4exp_mm_indexer_scores(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *pool, const ds4_gpu_tensor *blk_filled,
        const ds4_gpu_tensor *row_pos,
        uint32_t n_tokens, uint32_t n_blocks, uint32_t n_head,
        uint32_t head_dim, uint32_t pool_size, uint32_t pos_hi) {
    if (!n_tokens || !n_blocks || !n_head || !head_dim || !pool_size ||
        !mm_tensor_has(scores, (uint64_t)n_tokens * n_blocks, 4u) ||
        !mm_tensor_has(q, (uint64_t)n_tokens * n_head * head_dim, 4u) ||
        !mm_tensor_has(pool, (uint64_t)n_blocks * head_dim, 4u) ||
        !mm_tensor_has(blk_filled, n_blocks, 4u) ||
        !mm_tensor_has(row_pos, (uint64_t)n_tokens * 4u, 4u)) return 0;
    const uint32_t nth = mm_threads(head_dim);
    mm_indexer_scores_kernel<<<dim3(n_blocks, n_tokens), nth,
        nth * sizeof(float), MM_STREAM>>>(
            (const float *)q->ptr, (const float *)pool->ptr,
            (const int32_t *)blk_filled->ptr, (float *)scores->ptr,
            n_tokens, n_blocks, n_head, head_dim, pool_size, pos_hi,
            (const int32_t *)row_pos->ptr, 1.0f / sqrtf((float)head_dim));
    return mm_cuda_ok(cudaGetLastError(), "mm indexer scores");
}

/* ---------- cell selection --------------------------------------------------
 * Per row i: candidates are populated cells j with
 *     j <= slot_i                                        (KQ visibility,
 *                                                         slot order)
 * and ( cell_pos[j] >= tail_start(i)                      tail, always taken
 *   || (blk_filled[cell_blk[j]] >= r && cell_blk[j] in    member of a
 *        topk[i])                                          selected block)
 * emitting the first width = min(n_cells, top_k*r + r - 1) in slot order.
 *
 * Phase A sorts the row's top_k block ids (bitonic, as the text select
 * does); phase B scans cells in chunks with a block-wide exclusive count
 * for ordered compaction. */
__global__ static void mm_select_kernel(
        const int32_t *topk, const int32_t *cell_pos,
        const int32_t *cell_blk, const int32_t *blk_filled,
        int32_t *selected, int32_t *counts,
        uint32_t n_tokens, uint32_t n_blocks, uint32_t top_k,
        uint32_t pool_size, uint32_t max_selected, uint32_t n_cells,
        uint32_t slot0, const int32_t *row_pos) {
    extern __shared__ int32_t mm_sel_ids[];
    __shared__ uint32_t mm_sel_carry;
    __shared__ uint32_t mm_sel_chunk[1024 / 32 + 1];
    const uint32_t token = blockIdx.x;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    if (token >= n_tokens) return;
    /* Phase A: bitonic-sort the top_k ids so membership is a binary
     * search. */
    const int32_t sentinel = 0x7fffffff;
    uint32_t sort_width = 1;
    while (sort_width < top_k) sort_width *= 2;
    for (uint32_t i = tid; i < sort_width; i += nth) {
        int32_t b = sentinel;
        if (i < top_k) {
            const int32_t cand = topk[(uint64_t)token * top_k + i];
            if (cand >= 0 && (uint32_t)cand < n_blocks) b = cand;
        }
        mm_sel_ids[i] = b;
    }
    __syncthreads();
    for (uint32_t k = 2u; k <= sort_width; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0u; j >>= 1) {
            for (uint32_t i = tid; i < sort_width; i += nth) {
                const uint32_t ixj = i ^ j;
                if (ixj > i) {
                    const bool up = (i & k) == 0u;
                    if ((up && mm_sel_ids[i] > mm_sel_ids[ixj]) ||
                        (!up && mm_sel_ids[i] < mm_sel_ids[ixj])) {
                        const int32_t tmp = mm_sel_ids[i];
                        mm_sel_ids[i] = mm_sel_ids[ixj];
                        mm_sel_ids[ixj] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }
    const uint32_t t_i = (uint32_t)row_pos[(size_t)token * 4u];
    const uint32_t tail_start = ((t_i + 1u) / pool_size) * pool_size;
    /* KQ visibility in slot order: populated cells j <= slot0 + token. */
    const uint32_t slot_hi = slot0 + token;
    int32_t *dst = selected + (uint64_t)token * max_selected;
    if (tid == 0) mm_sel_carry = 0;
    __syncthreads();
    const uint32_t width = max_selected;
    for (uint32_t base = 0; base < n_cells; base += nth) {
        const uint32_t j = base + tid;
        int take = 0;
        if (j < n_cells && j <= slot_hi) {
            const int32_t p = cell_pos[j];
            if (p >= 0) {
                if ((uint32_t)p >= tail_start) {
                    take = 1;
                } else {
                    const int32_t b = cell_blk[j];
                    if (b >= 0 && blk_filled[b] >= (int32_t)pool_size) {
                        /* binary search b in sorted topk ids */
                        uint32_t lo = 0, hi = sort_width;
                        while (lo < hi) {
                            const uint32_t mid = (lo + hi) >> 1;
                            if (mm_sel_ids[mid] < b) lo = mid + 1;
                            else hi = mid;
                        }
                        take = (lo < sort_width && mm_sel_ids[lo] == b);
                    }
                }
            }
        }
        /* ordered compaction: warp ballots -> block scan */
        const uint32_t lane = tid & 31u, warp = tid >> 5;
        const uint32_t mask = __ballot_sync(0xffffffffu, take);
        if (lane == 0) mm_sel_chunk[warp] = __popc(mask);
        __syncthreads();
        if (tid == 0) {
            uint32_t run = mm_sel_carry;
            for (uint32_t w = 0; w < nth / 32u; w++) {
                const uint32_t c = mm_sel_chunk[w];
                mm_sel_chunk[w] = run;
                run += c;
            }
            mm_sel_carry = run;
        }
        __syncthreads();
        if (take) {
            const uint32_t off =
                mm_sel_chunk[warp] + __popc(mask & ((1u << lane) - 1u));
            if (off < width) dst[off] = (int32_t)j;
        }
        __syncthreads();
        if (mm_sel_carry >= width) break;
    }
    if (tid == 0)
        counts[token] = (int32_t)(mm_sel_carry < width ? mm_sel_carry
                                                      : width);
}
extern "C" int ds4_gpu_qwen4exp_mm_select(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *counts,
        const ds4_gpu_tensor *topk, const ds4_gpu_tensor *cell_pos,
        const ds4_gpu_tensor *cell_blk, const ds4_gpu_tensor *blk_filled,
        const ds4_gpu_tensor *row_pos,
        uint32_t n_tokens, uint32_t n_blocks, uint32_t top_k,
        uint32_t pool_size, uint32_t max_selected, uint32_t n_cells,
        uint32_t slot0) {
    if (!n_tokens || !n_blocks || !top_k || !pool_size || !n_cells ||
        !mm_tensor_has(selected, (uint64_t)n_tokens * max_selected, 4u) ||
        !mm_tensor_has(counts, n_tokens, 4u) ||
        !mm_tensor_has(topk, (uint64_t)n_tokens * top_k, 4u) ||
        !mm_tensor_has(cell_pos, n_cells, 4u) ||
        !mm_tensor_has(cell_blk, n_cells, 4u) ||
        !mm_tensor_has(blk_filled, n_blocks, 4u) ||
        !mm_tensor_has(row_pos, (uint64_t)n_tokens * 4u, 4u)) return 0;
    uint32_t sort_width = 1;
    while (sort_width < top_k) sort_width *= 2;
    const uint32_t nth = 1024u;
    mm_select_kernel<<<n_tokens, nth,
        (size_t)sort_width * sizeof(int32_t), MM_STREAM>>>(
            (const int32_t *)topk->ptr, (const int32_t *)cell_pos->ptr,
            (const int32_t *)cell_blk->ptr,
            (const int32_t *)blk_filled->ptr, (int32_t *)selected->ptr,
            (int32_t *)counts->ptr, n_tokens, n_blocks, top_k, pool_size,
            max_selected, n_cells, slot0,
            (const int32_t *)row_pos->ptr);
    return mm_cuda_ok(cudaGetLastError(), "mm indexer select");
}
