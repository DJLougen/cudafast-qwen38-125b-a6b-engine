/* qwen4exp-vision-encode -- sidecar image encoder for ds4's qwen4exp vision
 * path.  Runs the mmproj (clip/qwen3vl_merger) tower with the exact
 * preprocessor and encoder llama.cpp's mtmd uses, so the embedding rows it
 * writes are by construction what llama.cpp would feed the decoder.
 *
 * Usage:
 *   qwen4exp-vision-encode <mmproj.gguf> <image-file>
 *   qwen4exp-vision-encode <mmproj.gguf> -            # image bytes on stdin
 *
 * stdout (binary, little-endian):
 *   u32 magic = 0x51385631 ("Q8V1")
 *   u32 n_tokens      (rows to splice into the prompt)
 *   u32 nx, u32 ny    (merged grid dims for mrope h/w positions)
 *   u32 embd          (2560 = text hidden size)
 *   f32 n_tokens * embd
 * Diagnostics go to stderr only; any failure exits nonzero.
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

static void write_u32(uint32_t v) {
    unsigned char b[4] = { (unsigned char)(v), (unsigned char)(v >> 8),
                           (unsigned char)(v >> 16), (unsigned char)(v >> 24) };
    fwrite(b, 1, 4, stdout);
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <mmproj.gguf> <image-file|->\n", argv[0]);
        return 2;
    }
    const char * mmproj = argv[1];
    const char * image  = argv[2];

    std::vector<unsigned char> bytes;
    if (!strcmp(image, "-")) {
        unsigned char buf[65536];
        for (;;) {
            size_t n = fread(buf, 1, sizeof(buf), stdin);
            bytes.insert(bytes.end(), buf, buf + n);
            if (n < sizeof(buf)) break;
        }
        if (bytes.empty()) {
            fprintf(stderr, "%s: empty image on stdin\n", argv[0]);
            return 1;
        }
    } else {
        FILE * f = fopen(image, "rb");
        if (!f) {
            fprintf(stderr, "%s: cannot open %s\n", argv[0], image);
            return 1;
        }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz <= 0 || sz > (long)(256u << 20)) {
            fprintf(stderr, "%s: bad image size %ld\n", argv[0], sz);
            fclose(f);
            return 1;
        }
        bytes.resize((size_t)sz);
        if (fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
            fprintf(stderr, "%s: short read on %s\n", argv[0], image);
            fclose(f);
            return 1;
        }
        fclose(f);
    }

    int w = 0, h = 0, nc = 0;
    unsigned char * rgb = stbi_load_from_memory(
        bytes.data(), (int)bytes.size(), &w, &h, &nc, 3);
    if (!rgb) {
        fprintf(stderr, "%s: image decode failed: %s\n", argv[0],
                stbi_failure_reason());
        return 1;
    }

    clip_context_params cp{};
    cp.use_gpu = false;      /* CPU only: no GPU contention with ds4 */
    cp.warmup  = false;
    cp.image_min_tokens = 0;
    cp.image_max_tokens = 0;
    clip_init_result res = clip_init(mmproj, cp);
    clip_ctx * ctx = res.ctx_v;
    if (!ctx) {
        fprintf(stderr, "%s: mmproj load failed\n", argv[0]);
        stbi_image_free(rgb);
        return 1;
    }
    if (!clip_has_vision_encoder(ctx)) {
        fprintf(stderr, "%s: mmproj has no vision encoder\n", argv[0]);
        stbi_image_free(rgb);
        clip_free(ctx);
        return 1;
    }

    clip_image_u8 img;
    img.set_size({w, h}, false);
    img.cpy_buf(std::vector<uint8_t>(rgb, rgb + (size_t)w * h * 3));
    stbi_image_free(rgb);

    mtmd_image_preprocessor_dyn_size preproc(ctx);
    mtmd_image_preproc_out out;
    try {
        out = preproc.preprocess(img);
    } catch (const std::exception & e) {
        fprintf(stderr, "%s: preprocess failed: %s\n", argv[0], e.what());
        clip_free(ctx);
        return 1;
    }
    if (out.entries.size() != 1) {
        fprintf(stderr, "%s: expected one image entry, got %zu\n",
                argv[0], out.entries.size());
        clip_free(ctx);
        return 1;
    }
    clip_image_f32_batch batch;
    batch.is_audio = false;
    batch.entries.push_back(out.entries[0]);

    const int nx = clip_n_output_tokens_x(ctx, &batch.entries[0]);
    const int ny = clip_n_output_tokens_y(ctx, &batch.entries[0]);
    const int n_tokens = nx * ny;
    const int embd_dim = clip_n_mmproj_embd(ctx);
    /* clip_encode copies into a caller-sized buffer only. */
    std::vector<float> embd((size_t)n_tokens * (size_t)embd_dim);
    if (!clip_image_batch_encode(ctx, 0 /* n_threads: lib default */, &batch,
                                 embd)) {
        fputs("encode failed\n", stderr);
        clip_free(ctx);
        return 1;
    }
    if (n_tokens <= 0 || embd_dim <= 0 ||
        embd.size() != (size_t)n_tokens * (size_t)embd_dim) {
        fprintf(stderr, "%s: bad output shape %d x %d x %d (embd %zu)\n",
                argv[0], nx, ny, embd_dim, embd.size());
        clip_free(ctx);
        return 1;
    }
    clip_free(ctx);

    write_u32(DS4V_MAGIC);
    write_u32((uint32_t)n_tokens);
    write_u32((uint32_t)nx);
    write_u32((uint32_t)ny);
    write_u32((uint32_t)embd_dim);
    fwrite(embd.data(), sizeof(float), embd.size(), stdout);
    if (ferror(stdout)) {
        fprintf(stderr, "%s: stdout write failed\n", argv[0]);
        return 1;
    }
    fprintf(stderr, "%s: %d tokens (%dx%d grid, embd %d)\n",
            argv[0], n_tokens, nx, ny, embd_dim);
    return 0;
}
