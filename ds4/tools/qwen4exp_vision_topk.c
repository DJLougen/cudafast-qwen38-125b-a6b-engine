/* qwen4exp-vision-topk -- first-token top-K logits for a multimodal prompt.
 * Mirrors ds4-server's qwen4exp ChatML render (thinking=MAX) + image splice:
 *   <|im_start|>system\n<reasoning instructions>\n\n<|im_end|>\n
 *   <|im_start|>user\n<image tokens><question><|im_end|>\n
 *   <|im_start|>assistant\n<think>\n
 * usage: qwen4exp-vision-topk <model.gguf> <mmproj.gguf> <image> <question>
 */
#include "../ds4.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model> <mmproj> <image> <question>\n",
                argv[0]);
        return 2;
    }
    const int reps = argc > 5 ? atoi(argv[5]) : 1;
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path  = argv[1];
    opt.vision_path = argv[2];
    opt.backend     = DS4_BACKEND_CUDA;
    opt.context_size = 8192;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) {
        fputs("engine open failed\n", stderr);
        return 1;
    }
    if (!ds4_engine_has_vision(e)) {
        fputs("vision not ready\n", stderr);
        return 1;
    }
    char err[512] = {0};
    ds4_vision_embedding emb;
    memset(&emb, 0, sizeof(emb));
    if (!ds4_engine_vision_encode_file(e, argv[3], &emb, err, sizeof(err))) {
        fprintf(stderr, "encode: %s\n", err);
        return 1;
    }
    /* Render exactly like render_qwen4exp_chat_prompt_text + the server's
     * marker splice: tokenize up to the image, append vision tokens, then
     * tokenize the tail. */
    ds4_tokens tok;
    memset(&tok, 0, sizeof(tok));
    ds4_vision_span span;
    memset(&span, 0, sizeof(span));
    static const char pre[] =
        "<|im_start|>system\n"
        "Reasoning effort is set to xhigh. Please think carefully through the task, "
        "validate key assumptions, consider plausible alternatives, and prioritize "
        "correctness, consistency, and clarity in the final answer."
        "<|im_end|>\n"
        "<|im_start|>user\n";
    ds4_tokenize_rendered_chat(e, pre, &tok);
    if (!ds4_prompt_append_vision(e, &tok, &span, &emb, err, sizeof(err))) {
        fprintf(stderr, "append_vision: %s\n", err);
        return 1;
    }
    char tail[8192];
    snprintf(tail, sizeof(tail),
             "%s<|im_end|>\n<|im_start|>assistant\n<think>\n", argv[4]);
    ds4_tokenize_rendered_chat(e, tail, &tok);
    printf("prompt_tokens=%d image_span_start=%u image_tokens=%u grid=%ux%u\n",
           tok.len, span.token_start,
           span.embedding.token_count, span.embedding.grid_width,
           span.embedding.grid_height);
    if (getenv("DS4V_DUMP_TOKENS")) {
        for (int i = 0; i < tok.len; i++) {
            size_t dl = 0;
            char *d = ds4_token_text(e, tok.v[i], &dl);
            printf("tok[%d]=%d \"%.*s\"\n", i, tok.v[i],
                   d ? (int)dl : 0, d ? d : "");
            free(d);
        }
    }
    for (int rep = 0; rep < reps; rep++) {
    printf("=== rep %d\n", rep);
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, 8192) != 0 || !s) {
        fputs("session create failed\n", stderr);
        return 1;
    }
    if (ds4_session_sync_multimodal(s, &tok, &span, 1, err, sizeof(err))
            != 0) {
        fprintf(stderr, "sync: %s\n", err);
        return 1;
    }
    const int top = ds4_session_argmax(s);
    size_t tlen = 0;
    char *tt = ds4_token_text(e, top, &tlen);
    printf("argmax=%d text=\"%.*s\"\n", top, tt ? (int)tlen : 0, tt ? tt : "");
    if (tt) free(tt);
    /* Full first-token logits fingerprint: FNV-1a over the f32 logits the
     * model just produced. */
    {
        static float *lg = NULL;
        static int lg_cap = 0;
        const int nv = ds4_engine_vocab_size(e);
        if (nv > lg_cap) {
            free(lg);
            lg = malloc((size_t)nv * sizeof(float));
            lg_cap = nv;
        }
        if (lg && ds4_session_copy_logits(s, lg, nv) == nv) {
            const unsigned char *p = (const unsigned char *)lg;
            uint64_t h = 1469598103934665603ull;
            for (uint64_t i = 0; i < (uint64_t)nv * 4u; i++) {
                h ^= p[i];
                h *= 1099511628211ull;
            }
            printf("logits_fnv=%016llx\n", (unsigned long long)h);
        }
    }
    ds4_token_score scores[8];
    const int k = ds4_session_top_logprobs(s, scores, 8);
    for (int i = 0; i < k; i++) {
        size_t len = 0;
        char *txt = ds4_token_text(e, scores[i].id, &len);
        printf("%d\t%.4f\t%.4f\t\"%.*s\"\n", scores[i].id,
               scores[i].logit, scores[i].logprob,
               txt ? (int)len : 0, txt ? txt : "");
        if (txt) free(txt);
    }
    /* --gen N: greedy-argmax N tokens and hash the id stream. */
    {
        const char *ge = getenv("DS4V_GEN");
        const int gen = ge ? atoi(ge) : 0;
        if (gen > 0) {
            uint64_t h = 1469598103934665603ull;
            int cur = top;
            for (int i = 0; i < gen; i++) {
                const uint32_t id = (uint32_t)cur;
                h ^= id;
                h *= 1099511628211ull;
                printf("gen[%d]=%d\n", i, cur);
                if (ds4_session_eval(s, cur, err, sizeof(err)) != 0) {
                    fprintf(stderr, "eval: %s\n", err);
                    return 1;
                }
                cur = ds4_session_argmax(s);
            }
            printf("gen_fnv=%016llx\n", (unsigned long long)h);
        }
    }
    ds4_session_free(s);
    }
    ds4_engine_close(e);
    return 0;
}
