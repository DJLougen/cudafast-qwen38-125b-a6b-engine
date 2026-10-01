/* qwen4exp-vision-encode -- sidecar image encoder for ds4's qwen4exp vision
 * path.  Runs the mmproj (clip/qwen3vl_merger) tower with the exact
 * preprocessor and encoder llama.cpp's mtmd uses, so the embedding rows it
 * writes are by construction what llama.cpp would feed the decoder.
 *
 * Two modes:
 *   one-shot: qwen4exp-vision-encode <mmproj.gguf> <image-file>
 *   daemon:   qwen4exp-vision-encode <mmproj.gguf> --daemon
 *             reads   u32 path_len | path bytes   per request on stdin
 *             replies u32 'Q8V1'|n_tok|nx|ny|embd | f32 rows on stdout
 *             u32 0 = clean shutdown.  The daemon keeps the mmproj mapped
 *             (and on the GPU) across requests -- per-request cost is decode
 *             + encode only.
 *
 * The clip tower runs on the CUDA backend when compiled in
 * (cp.use_gpu = true); ~0.9 GB VRAM.
 * Diagnostics go to stderr only; any failure exits nonzero / replies with
 * magic 0x51384552 ("Q8ER") + u32 msg_len + msg in daemon mode.
 */
#include "clip.h"
#include "clip-impl.h"
#include "mtmd-image.h"
#include "stb/stb_image.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define DS4V_MAGIC 0x51385631u
#define DS4V_ERR   0x51384552u

static clip_ctx * g_ctx = nullptr;

static void write_u32(uint32_t v) {
    unsigned char b[4] = { (unsigned char)(v), (unsigned char)(v >> 8),
                           (unsigned char)(v >> 16), (unsigned char)(v >> 24) };
    fwrite(b, 1, 4, stdout);
}

static bool read_u32(uint32_t * v) {
    unsigned char b[4];
    if (fread(b, 1, 4, stdin) != 4) return false;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return true;
}

static int encode_one(const std::vector<unsigned char> & bytes,
                      const char * tag) {
    int w = 0, h = 0, nc = 0;
    unsigned char * rgb = stbi_load_from_memory(
        bytes.data(), (int)bytes.size(), &w, &h, &nc, 3);
    if (!rgb) {
        fprintf(stderr, "%s: image decode failed: %s\n", tag,
                stbi_failure_reason());
        return 1;
    }
    clip_image_u8 img;
    img.set_size({w, h}, false);
    img.cpy_buf(std::vector<uint8_t>(rgb, rgb + (size_t)w * h * 3));
    stbi_image_free(rgb);
    mtmd_image_preprocessor_dyn_size preproc(g_ctx);
    mtmd_image_preproc_out out;
    try {
        out = preproc.preprocess(img);
    } catch (const std::exception & e) {
        fprintf(stderr, "%s: preprocess failed: %s\n", tag, e.what());
        return 1;
    }
    if (out.entries.size() != 1) {
        fprintf(stderr, "%s: expected one image entry, got %zu\n",
                tag, out.entries.size());
        return 1;
    }
    clip_image_f32_batch batch;
    batch.is_audio = false;
    batch.entries.push_back(out.entries[0]);
    const int nx = clip_n_output_tokens_x(g_ctx, &batch.entries[0]);
    const int ny = clip_n_output_tokens_y(g_ctx, &batch.entries[0]);
    const int n_tokens = nx * ny;
    const int embd_dim = clip_n_mmproj_embd(g_ctx);
    std::vector<float> embd((size_t)n_tokens * (size_t)embd_dim);
    if (!clip_image_batch_encode(g_ctx, 0, &batch, embd)) {
        fputs("encode failed\n", stderr);
        return 1;
    }
    if (n_tokens <= 0 || embd_dim <= 0 ||
        embd.size() != (size_t)n_tokens * (size_t)embd_dim) {
        fprintf(stderr, "%s: bad output shape %d x %d x %d\n", tag, nx, ny,
                embd_dim);
        return 1;
    }
    write_u32(DS4V_MAGIC);
    write_u32((uint32_t)n_tokens);
    write_u32((uint32_t)nx);
    write_u32((uint32_t)ny);
    write_u32((uint32_t)embd_dim);
    fwrite(embd.data(), sizeof(float), embd.size(), stdout);
    if (ferror(stdout)) {
        fprintf(stderr, "%s: stdout write failed\n", tag);
        return 1;
    }
    fprintf(stderr, "%s: %d tokens (%dx%d grid, embd %d)\n",
            tag, n_tokens, nx, ny, embd_dim);
    return 0;
}

static int encode_path(const char * path, const char * tag) {
    std::vector<unsigned char> bytes;
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "%s: cannot open %s\n", tag, path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > (long)(256u << 20)) {
        fprintf(stderr, "%s: bad image size %ld\n", tag, sz);
        fclose(f);
        return 1;
    }
    bytes.resize((size_t)sz);
    if (fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
        fprintf(stderr, "%s: short read on %s\n", tag, path);
        fclose(f);
        return 1;
    }
    fclose(f);
    return encode_one(bytes, tag);
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr,
                "usage: %s <mmproj.gguf> <image-file>\n"
                "       %s <mmproj.gguf> --daemon\n", argv[0], argv[0]);
        return 2;
    }
    clip_context_params cp{};
    cp.use_gpu = !getenv("DS4V_CPU");       /* CUDA clip; ~0.9 GiB VRAM */
    cp.warmup  = false;
    cp.image_min_tokens = 0;
    cp.image_max_tokens = 0;
    clip_init_result res = clip_init(argv[1], cp);
    g_ctx = res.ctx_v;
    if (!g_ctx || !clip_has_vision_encoder(g_ctx)) {
        fprintf(stderr, "%s: mmproj load failed\n", argv[0]);
        return 1;
    }
    if (!strcmp(argv[2], "--daemon")) {
        write_u32(0x52334459u); /* "RDY3" ready marker */
        fflush(stdout);
        for (;;) {
            uint32_t len;
            if (!read_u32(&len)) break;          /* stdin closed: clean exit */
            if (len == 0) break;                 /* shutdown word */
            if (len > 4096) {
                write_u32(DS4V_ERR);
                write_u32(12);
                fwrite("path too long", 1, 12, stdout);
                fflush(stdout);
                continue;
            }
            std::string path(len, 0);
            if (fread(path.data(), 1, len, stdin) != len) break;
            if (encode_path(path.c_str(), argv[0]) != 0) {
                /* emit an error block so the caller never hangs */
                write_u32(DS4V_ERR);
                write_u32(13);
                fwrite("encode failed", 1, 13, stdout);
            }
            fflush(stdout);
        }
        clip_free(g_ctx);
        return 0;
    }
    int rc = encode_path(argv[2], argv[0]);
    clip_free(g_ctx);
    return rc;
}
