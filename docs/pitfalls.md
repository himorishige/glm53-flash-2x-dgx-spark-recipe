# Pitfalls we hit (2026-08-31 → 2026-09-01, 2× DGX Spark, GB10 / sm_121)

Every item below cost us at least one boot cycle (15 min) and, in one case, a physical power cycle of both
machines. All are reproduced from our own logs; community references are given where they exist.

## 1. The day-0 stock image cannot serve this model on GB10

`vllm/vllm-openai:glm53-flash-arm64-cu130` (vLLM `0.1.dev20051+g487ecf187`, Docker Hub 2026-08-26) loads the
weights, does the marlin repack and completes the NCCL rendezvous over the QSFP link — and then dies in
warm-up with

```
RuntimeError: concat_and_cache_mla, /workspace/csrc/libtorch_stable/cache_kernels.cu:866, pe_dim must be 64 for fp8_ds_mla
```

for every `--kv-cache-dtype` we tried (`fp8`, `fp8_e4m3`, `auto`). GLM-5.3-Flash is NoPE
(`qk_rope_head_dim = 0`), and the DeepSeek-sparse-attention indexer cache path always uses the `fp8_ds_mla`
layout, which asserts `pe_dim == 64`. This is the fault Libertai documents on the model card; every
"it works on 2× Spark" report we could find runs a patched image (tonyd2wild v8/v9/v11) or a plugin.

**Fix**: use `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (this recipe) and keep the SM121 top-k
patch mounted (§4). Three failed smokes are in the git history of this recipe's origin.

## 2. compose `--env-file`: an inline comment after an EMPTY value becomes the value

```
SPEC_MODEL=                     # e.g. /models/GLM-5.3-Flash-DFlash2
```

rendered `SPEC_MODEL="# e.g. /models/GLM-5.3-Flash-DFlash2"` and the JSON we build for
`--speculative-config` was garbage; `vllm serve` exited immediately and our health loop waited 92 minutes
on a dead container. Comments after a non-empty value are stripped correctly, so a dry run with the
value overridden did not reveal it. Keep empty-valued lines comment-free, and make your health check
abort when the container is `exited` (`scripts/health.sh` does).

## 3. `flashinfer_cutlass` at `--gpu-memory-utilization 0.85` froze BOTH hosts

During weight load the two Sparks stopped answering ping and ssh simultaneously and needed a physical
power cycle. Libertai's `env.glm53.example` says cutlass needs a utilisation floor around 0.79–0.80 and that
0.82/0.88 triggered a host-wide OOM; marlin at 0.85 already sits at 117/121 GiB used. We did not retry
cutlass. Do not run FlashInfer MoE backends unattended on GB10.

## 4. Past ~24K prompt tokens the stock top-k kernel dies

`persistent_topk` requests ~128 KB of shared memory; GB10 has 101,376 B. tonyd2wild's
`sparse_attn_indexer_kpool_sm121.py` (two-line diff: `pool_topk` initialised to -1, and
`persistent_topk` only on parts with ≥ 78 SMs) is bind-mounted over the image's file by
`docker-compose.yml`. With it, 32K / 131K / 200K prompts returned the needle every time.

## 5. Host preparation that is not optional

- `vm.swappiness=0` on both nodes (does not survive a reboot). With the default 60 the marlin repack can
  push the kernel into a UVM livelock; with swap fully off the worker dies during the repack.
- `sync; echo 3 > /proc/sys/vm/drop_caches` before every launch, and a cache-flusher loop while the shards
  load (`scripts/start-*.sh` start one for 25 min): the driver does not reclaim page cache by itself.
- Start the worker (rank 1, `--headless`) first, wait ~25 s, then the head. Always tear down BOTH ranks
  (`scripts/stop-both.sh`); a half-alive pair leaves one GPU at 100 %.
- Make sure no other model is loaded/unloaded during boot (an Ollama unload during vLLM's memory
  profiling makes the profiler assert on unified memory).

## 6. `--kv-cache-memory-bytes` vs `--kv-cache-memory`

Both spellings appear in community launch lines; the flag is `--kv-cache-memory-bytes` and argparse
accepts the unambiguous prefix. We pin 9 GiB (MTP) / 3–4 GiB (DFlash2), which is what vLLM itself
reported as available.

## 7. Speculative decoding vs sampling parameters

vLLM rejects `min_p` (and `logit_bias`) while speculative decoding is on:
`"The min_p and logit_bias sampling parameters are not yet supported with speculative decoding."` (400).
`top_k` is accepted.

## 8. Benchmark hygiene on a 2-node vLLM

- Prefix caching is on by default. A benchmark that reuses a prompt (even as request 0 of a C=2 batch)
  measures the cache, not prefill — our first C=2 long-context rows showed "3,922 tok/s prefill". Seed every
  request separately (`bench/longctx.py` does).
- Speculative decode tok/s depends on the text being generated: when the model paraphrases the haystack
  the drafter's acceptance jumps and decode reads 50–70 tok/s. Use TTFT / prefill as the long-context
  metrics and a fixed prompt for decode comparisons.
- A tmux `new -d "cmd"` runs under a non-login shell: tools installed in `~/.local/bin` (uv) are not on
  PATH. Wrap with `bash -lc`.
- `/v1/models` answers 200 even when the engine is dead; probe with a real completion.

## 9. Replacement characters (U+FFFD) in Japanese output

About 2 per 10,000 characters of Japanese prose come back with one kanji replaced by U+FFFD
(e.g. 検査範囲 → 検査�囲囲, 現実的 → �実的). Decoding the returned token IDs offline with the checkpoint's
`tokenizer.json` reproduces it at the same position, so the model emitted an invalid byte-fallback
sequence — it is not the detokenizer. It happens with MTP-4, DFlash2 and without speculation, with fp8 and bf16 KV, and neither `top_k 40`
nor `min_p 0.05` removes it. The same probe on a single Spark with the 2-bit GGUF under llama.cpp produced
zero. The mechanism is the tokenizer: many Japanese kanji are spelled as a 2-byte fragment token plus a
1-byte continuation token (測 = `e6b8` + `ac`), and the model sometimes skips the continuation.
`patches/utf8_guard_lp.py` (a logits processor, `LOGITS_PROCESSORS=utf8_guard_lp:Utf8GuardLogitsProcessor`)
forbids the invalid continuations; it cannot be combined with speculative decoding in vLLM. Attribution
table and measured effect in `docs/measurements.md` §4.
