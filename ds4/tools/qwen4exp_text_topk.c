/* qwen4exp-text-topk: first-token top-K for a TEXT-ONLY prompt (same render),
 * to bisect mm-path vs engine-wide nondeterminism. */
#include "../ds4.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <model> <question>\n", argv[0]);
        return 2;
    }
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path  = argv[1];
    opt.backend     = DS4_BACKEND_CUDA;
    opt.context_size = 8192;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) return 1;
    char err[512] = {0};
    ds4_tokens tok;
    memset(&tok, 0, sizeof(tok));
    char text[8192];
    snprintf(text, sizeof(text),
        "<|im_start|>system\n"
        "Reasoning effort is set to xhigh. Please think carefully through the task, "
        "validate key assumptions, consider plausible alternatives, and prioritize "
        "correctness, consistency, and clarity in the final answer.<|im_end|>\n"
        "<|im_start|>user\n%s<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n", argv[2]);
    ds4_tokenize_rendered_chat(e, text, &tok);
    printf("prompt_tokens=%d\n", tok.len);
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, 8192) != 0 || !s) return 1;
    if (ds4_session_sync(s, &tok, err, sizeof(err)) != 0) {
        fprintf(stderr, "sync: %s\n", err);
        return 1;
    }
    const int top = ds4_session_argmax(s);
    size_t tlen = 0;
    char *tt = ds4_token_text(e, top, &tlen);
    printf("argmax=%d text=\"%.*s\"\n", top, tt ? (int)tlen : 0, tt ? tt : "");
    if (tt) free(tt);
    ds4_token_score scores[8];
    const int k = ds4_session_top_logprobs(s, scores, 8);
    for (int i = 0; i < k; i++) {
        size_t len = 0;
        char *txt = ds4_token_text(e, scores[i].id, &len);
        printf("%d\t%.6f\t%.6f\t\"%.*s\"\n", scores[i].id,
               scores[i].logit, scores[i].logprob,
               txt ? (int)len : 0, txt ? txt : "");
        free(txt);
    }
    ds4_session_free(s);
    ds4_engine_close(e);
    return 0;
}
