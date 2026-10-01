# MOONEY — what this fork is

This is the **fast runtime for
[Qwen3.8-Flash-Next-Mooney](https://huggingface.co/DJLougen/Qwen3.8-Flash-Next-Mooney)** —
a 180B-parameter MoE checkpoint compressed to 92 GB on disk / about 39 GiB in
memory, which runs on **one NVIDIA DGX Spark** (GB10).

The branch is `lbf/pq2-rot`, on top of
[`Layr-Labs/cudafast-qwen38-125b-a6b-engine`](https://github.com/Layr-Labs/cudafast-qwen38-125b-a6b-engine)
@ `5707d4f2` (which vendors `Layr-Labs/ds4`, itself a port of `antirez/ds4`).
Our delta adds what Mooney's GGUF needs:

- `PQ2_0` (GGML type 142) ternary routed-expert tensors in the loader and
  CUDA kernels;
- `lowbitflash.rot.*` rotation metadata (fail-closed parsing) plus fused
  segmented-FWHT rotation kernels;
- Q8_0 PLE n-gram tables (lazily read from SSD);
- MTP speculative decoding with the `mtp-Qwen3.8-Flash-Next.gguf` (Q8_0) draft head
  (`--mtp-model` / `--mtp-draft`);
- ChatML/qwen4exp serving fixes and GB10 (sm_121) build fixes.

Measured on a DGX Spark: **45.8 tok/s** decode short-context and **47.7 tok/s**
at 4k with the MTP head (draft depth 1); 33.3 / 30.6 tok/s serial — 1.39×/1.43×
faster than the 4-bit UD-Q4_K_XL build on this same engine. See the model card
for the full numbers and quality evaluation.

## Easiest path: one-command setup

[`DJLougen/mooney-spark`](https://github.com/DJLougen/mooney-spark) builds this
engine, downloads the model with per-file sha256 verification, and writes a
launcher:

```bash
git clone https://github.com/DJLougen/mooney-spark && cd mooney-spark
./setup_spark.sh
~/mooney-spark/launch/serve_ds4.sh   # OpenAI-compatible, http://127.0.0.1:8000/v1
```

## Build manually on a DGX Spark (GB10)

```bash
make -C ds4 cuda-spark CUDA_ARCH=sm_121 -j8
```

`CUDA_ARCH=sm_121` is **mandatory** — upstream's default arch emits PTX that
`ptxas` rejects on GB10. Requires CUDA 13.x toolkit and driver ≥ 580.159.03.

Run:

```bash
ds4/ds4-server \
  -m Qwen3.8-Flash-Next-Mooney-PQ2_0-00001-of-00004.gguf \
  --vision mmproj-Qwen3.8-Flash-Next-Mooney.gguf \
  --mtp-model mtp-Qwen3.8-Flash-Next.gguf --mtp-draft 2 \
  --cuda --ctx 32768 --host 127.0.0.1 --port 8000
```

(`--mtp-draft 2` = draft depth 1. Text only for now — image input support is in
progress; the [prism-llama.cpp](https://github.com/DJLougen/prism-llama.cpp)
fork serves images today.)

## Credits and license

MIT, unchanged. The engine descends from
[Layr-Labs/ds4](https://github.com/Layr-Labs/ds4) /
[antirez/ds4](https://github.com/antirez/ds4) via the
[cuda.fast](https://github.com/Layr-Labs/cudafast-qwen38-125b-a6b-engine)
benchmark engine. Quantization and evaluation compute for the Mooney release
was generously provided by [Lambda](https://lambda.ai) — thanks to
[Zach Mueller](https://x.com/TheZachMueller).
