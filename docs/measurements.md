# Measurements (2026-09-01)

Hardware: 2× NVIDIA DGX Spark (GB10, 121 GiB unified memory each), QSFP 200 GbE direct link
(RoCE v2, MTU 9000, `nccl-tests all_reduce` 12.1 GB/s), driver 580.159.03, Docker 29.2 / compose v5.
Model: `LibertAIDAI/GLM-5.3-Flash-NVFP4` rev `caca4e6a` (181 GiB, 120 shards, sha256-verified on both nodes).
Image: `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (digest `4def0ef6…`) + SM121 top-k patch.
Day-1 config A (`cluster.env`): marlin / 262,144 ctx / `--max-num-seqs 2` / MTP k=4 / `--enforce-eager` /
`fp8_e4m3` KV / `--max-num-batched-tokens 4096` / util 0.85 / 9 GiB KV pin. Boot (start-head → `/v1/models`)
≈ 15 min every time (890–990 s), including the FlashInfer autotune that the persisted `~/.cache/glm53-vllm`
does not shorten.

## 1. One-factor sweep (`scripts/tune-sweep.sh`)

Forced 256 output tokens, greedy; C=1 is the median of 3 runs. `code_gen` = effective tok/s over 5 Japanese
spec-to-code problems at `reasoning_effort=low` (all 5/5 correct in every row). "used" = `free -g` while serving.

| config | change vs A | boot s | C=1 tok/s | TTFT s | C=max aggregate (C) | code_gen tok/s | head / worker used GB |
| --- | --- | --: | --: | --: | --: | --: | --- |
| mtp0 | no speculation | 868 | 14.55 | 0.231 | 27.2 (2) | 12.1 | 114 / 114 |
| **A-base** | MTP k=4 | — | 25.67 | 0.362 | 37.0 (2) | 22.3 | 117 / 116 |
| mtp3 | MTP k=3 | 926 | 27.78 | 0.251 | 30.6 (2) | 22.4 | 117 / 117 |
| mtp5 | MTP k=5 | 927 | 24.53 | 0.385 | 25.5 (2) | 22.6 | 117 / 117 |
| **dflash7** | DFlash2 drafter k=7, KV pin 3 GiB | 990 | **30.84** | 0.236 | **38.2 (2)** | **28.2** | 112 / 111 |
| s6 | seqs 6 @ 262K | 933 | 25.59 | 0.362 | 60.0 (6) | — | 118 / 117 |
| **s8** | seqs 8 @ 262K | 934 | 27.24 | 0.361 | **76.4 (8)** | — | 117 / 117 |
| kvbf16 | KV bf16 (`auto`) | 923 | 25.62 | 0.350 | 30.2 (2) | — | 117 / 117 |
| ctx65k-s8 | 65,536 ctx, seqs 8, util-sized KV | 932 | 27.99 | 0.352 | 76.0 (8) | — | 113 / 112 |
| **P** | 65K / seqs 8 / DFlash2 k=7 / KV pin 4.1 GiB | 892 | **31.32** | 0.335 | 62.7 (8) | **29.5** | 112 / 111 |
| cutlass | `--moe-backend flashinfer_cutlass` @ util 0.85 | — | **both hosts froze** (power cycle) | | | | |

Reading: speculation is +76–91 % (14.6 → 25–28); the DFlash2 drafter adds another +20 % single-stream and
+27 % on code; k=3 is the fastest MTP single-stream but k=4 has the best C=2 aggregate; 8 slots fit at
262K on the KDA state cache and give 76 tok/s aggregate (MTP-4) — with the drafter the aggregate peaks
around C=4 (70.8) and drops at C=8 (58.9) because the drafter competes for compute; KV dtype does not move
single-stream speed (fp8 wins at C=2); context length does not move speed at all.

Recommended configurations:

| use | config | why |
| --- | --- | --- |
| single user, long context | **S** = 262K / seqs 2 / DFlash2 k=7 / KV pin 3 GiB | 30.8 tok/s, 200K prompts with needle recall |
| a few interactive users | **P** = 65K / seqs 8 / DFlash2 k=7 / KV pin 4.1 GiB | 31–32 tok/s single, 70 tok/s aggregate at C=4 |
| many concurrent users | 262K / seqs 8 / MTP k=4 | 76 tok/s aggregate at C=8, 27 tok/s single |

## 2. Bench on config P (same harness as the author's single-Spark 2-bit GGUF article)

Japanese tasks, code generation, tool calls, code fix and agent-fitness probes come from a private
benchmark kit (not included). Numbers below are the kit's machine-scored results; the single-Spark column is
`unsloth/GLM-5.3-Flash-GGUF` UD-Q2_K_XL on llama.cpp (17.7 tok/s).

| item | 2× Spark NVFP4 (P) | 1× Spark 2-bit GGUF |
| --- | --- | --- |
| throughput C=1 / C=8 aggregate | 32.3 / 58.9 (76.0 with MTP-4) | 17.7 / 64.0 |
| Japanese 10 tasks, effort=max (8192 budget) | 7/9 machine-checked, 13,833 tokens, **344 s** | 6/9, 14,708 tokens, 927 s |
| effort=low / high | 7/9 in 43 s / 5/9 in 60 s | 7/9 in 98 s / 7/9 in 141 s |
| code generation (max / high / low / 16k) | 5/5 solved, 49/49 tests in every row | 4/5 (8192) – 5/5 |
| tool calls | 5/5 single, parallel calls in one message, `tool_choice=required` honoured, pseudo-FS ✅ | same |
| code fix (3-bug mini repo) | 5/5, 6 turns, 17 tool calls, **54 s** | 5/5, 5 turns, 72 s |
| agent-fitness probes | only `tool_selection_at_scale` fails (all 3 effort levels) | low/high also fail `long_system_prompt` |
| multi-turn probes A (10 turns) / B (5) / C (12K-char doc + 8) | 354 s / 118 s / 62 s (effort=low), all recall checks pass | 433 s / 209 s / 88 s |

The one task that exhausts the 8192 budget at effort=max (a summary with proper nouns) does so on both
machines; a separate 16,384-budget row is measured for it.

## 3. Long context × concurrency on config S (`bench/longctx.py`)

Each request gets its own haystack (unique header + shuffled filler, sized with `/tokenize`) with a
passphrase buried at 50 % depth; 512 forced output tokens, greedy, `reasoning_effort=low`. The needle column is
a separate non-forced request asking for the passphrase.

| prompt tokens | TTFT s | prefill tok/s | decode tok/s | needle |
| --: | --: | --: | --: | :--: |
| 2,048 | 1.5 | 1,391 | 27.5 | ✅ |
| 8,192 | 5.4 | 1,510 | 71.1* | ✅ |
| 32,768 | 22.2 | 1,473 | 50.9* | ✅ |
| 131,072 | 90.2 | 1,452 | 21.9 | ✅ |
| 200,000 | 139.2 | 1,439 | 21.5 | ✅ |

\* the drafter's acceptance jumps when the model paraphrases the haystack — decode with speculation is
content-dependent; TTFT and prefill are the long-context metrics.

| prompt tokens | C=2 TTFT s | C=2 prefill tok/s | aggregate tok/s |
| --: | --: | --: | --: |
| 2,048 | 2.8 | 743 | 51.3 |
| 8,192 | 8.8 | 975 | 24.1 |
| 32,768 | 34.5 | 1,056 | 15.2 |
| 131,072 | 140.6 | 1,069 | 5.1 |
| 200,000 | 218.9 | 1,052 | 3.3 |

Two simultaneous 200K prompts (400K tokens of context) complete; neither host froze at any size.

## 4. U+FFFD attribution (Japanese prose, `kabeuchi` probe ≈ 28K chars, temperature 1.0 / top_p 0.95)

| condition | U+FFFD | chars | per 10K |
| --- | --: | --: | --: |
| DFlash2 k=7 (S) | 7 | 27,757 | 2.5 |
| MTP k=4 | 5 | 27,823 | 1.8 |
| MTP k=4 + `top_k 40` | 13 | 27,475 | 4.7 |
| no speculation | 2 | 11,840 | 1.7 |
| no speculation + `min_p 0.05` | 9 / 2 | 11,787 / 14,635 | 7.6 / 1.4 |
| no speculation + bf16 KV | see README (last run) | | |
| 1× Spark 2-bit GGUF, llama.cpp (same probe) | 0 | 28,000+ | 0 |

Token IDs returned with `return_token_ids` and decoded offline with the checkpoint's `tokenizer.json`
(`bench/garble_ids.py`) reproduce the replacement character at the same position in 3/3 samples: the
model emits an invalid byte-fallback sequence; the detokenizer is not at fault.
