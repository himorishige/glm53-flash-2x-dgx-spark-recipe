# Measurements (2026-09-01; checkpoint switched to RedHatAI and re-measured 2026-09-02)

Narrative, the failed attempts and the verdict by use are in the article:
[I measured the value of running 320B GLM-5.3-Flash on 2 DGX Spark units to see if having 2 units is worth it](https://dev.classmethod.jp/en/articles/dgx-spark-2node-glm-5-3-flash-nvfp4-vllm/).

Hardware: 2× NVIDIA DGX Spark (GB10, 121 GiB unified memory each), QSFP 200 GbE direct link
(RoCE v2, MTU 9000, `nccl-tests` all_reduce 12.14 GB/s / all_gather 11.57 GB/s), driver 580.159.03,
Docker 29.2 / compose v5.
Model: `RedHatAI/GLM-5.3-Flash-NVFP4` rev `36c184c6` (184.3 GiB, 11 shards, compressed-tensors, sha256-verified on
both nodes) — the default since 2026-09-02. The initial release measured `LibertAIDAI/GLM-5.3-Flash-NVFP4` rev
`caca4e6a` (181 GiB, ModelOpt); §1–§4 keep those numbers for the record, §0 is the re-measurement.
Image: `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (20.7 GB, digest `4def0ef6…`; vLLM 0.1.dev20051,
torch 2.13.0+cu130, flashinfer 0.6.17, NCCL 2.29.7, CUDA 13.0) + SM121 top-k patch.
Sweep baseline = day-1 config A (`A-base` in `scripts/tune-configs.txt`; the shipped `cluster.env.example`
now defaults to MTP k=3, see §0): marlin / 262,144 ctx / `--max-num-seqs 2` / MTP k=4 / `--enforce-eager` /
`fp8_e4m3` KV / `--max-num-batched-tokens 4096` / util 0.85 / 9 GiB KV pin. Boot (start-head → `/v1/models`,
the worker started 25 s earlier) ≈ 15 min every time (868–990 s), including the FlashInfer autotune that the persisted
`~/.cache/glm53-vllm` does not shorten.

## 0. Re-measurement on the RedHatAI checkpoint (2026-09-02)

Drop-in swap: same image, same flags, only the model path changed; the KV pool is profiler-sized instead of
pinned (7.48 GiB = 1,151,844 tokens = 4.39 full-262K streams). Boot 542–608 s (was 868–990 s; 11 large shards
load faster). Everything below is measured with `scripts/tune-sweep.sh` / `bench/longctx.py` / the §2 kit
unless noted.

| config | C=1 tok/s | TTFT s | C=2 aggregate | code_gen tok/s | LibertAI (C=1 / C=2 / cg) |
| --- | --: | --: | --: | --- | --- |
| no speculation | 14.60 | 0.228 | 27.50 | 12.00 (5/5) | 14.55 / 27.19 / 12.06 |
| **MTP k=3 (new default)** | **26.47** | 0.249 | 29.77 | 21.97 (5/5) | 27.78 / 30.63 / 22.40 |
| MTP k=4 | 24.10 | 0.358 | 27.99 | 22.63 (5/5) | 25.67 / 37.03 / 22.27 |
| MTP k=5 | 22.63 | 0.263 | 26.35 | 21.75 (5/5) | 24.53 / 25.52 / 22.58 |
| DFlash2 k=7 | 25.57 | 0.352 | 26.51 | **27.74** (4/5) | 30.84 / 38.15 / 28.24 (5/5) |
| 262K / seqs 8 / MTP k=4 | 25.26 | 0.256 | 26.25 | — | C=4 57.70, C=8 **76.19** (was 56.3 / 76.4) |
| 65K / seqs 8 / MTP k=4 | 25.26 | 0.254 | 36.23 | — | C=4 45.77, C=8 75.25 (was 74.5 / 76.0) |
| KV bf16 (MTP k=4) | **cannot boot** | | | | one 262K request needs 3.62 GiB vs 3.54 GiB available |

Reading: the no-speculation denominator is identical, so the base model speed did not change. What changed is
speculation: the external DFlash2 drafter loses its single-stream lead (30.84 → 25.57 — the drafter was
trained against the unquantized weights, and its draft acceptance appears to drop on this quantization; not
directly measured), while the bundled MTP head barely moves. **Single-stream winner is now MTP k=3; DFlash2
still wins effective code-generation tok/s** (drafts hit more often on code). Speculation × C=2 gains shrink
across the board. Multi-user aggregate and long context are unchanged.

Long context (DFlash2 serve; prefill does not depend on the speculation method): prefill 1,369–1,480 tok/s
flat to 200K, TTFT 139.4 s at 200K, needle 5/5, two simultaneous 200K prompts complete in 292 s
(TTFT 209.7 s, 1,070 tok/s prefill per stream).

Japanese U+FFFD (the §4 defect): **0 across all three configs** — 0/23,930 chars (no speculation),
0/27,086 (DFlash2), 0/27,258 (MTP-4) — plus zero in the whole §2 harness output and 30 multi-turn probe
turns. The fragile fragment-path words (測 / 範 / 現実 …) appear 40+ times intact. Same probes, same Japanese
ratio as §4. tonyd2wild reports the same flip (4/9/8 → 0/0/0) with a Korean probe. The UTF-8 guard is no
longer needed on this checkpoint.

Quality harness on config P (same stages as §2): Japanese 10 tasks effort=max 6/9 in 413.1 s (16,384 budget:
8/9 in 577.5 s), low 7/9 in 45.7 s, high 6/9 in 47.3 s; code generation 5/5 (49/49) at max / low / 16k and
4/5 (47/49) at high — the slugify edge case flips between runs, low/max/16k pass it; tool calls 5/5 with
`tool_choice=required` honoured; code fix 5/5 in 38.1 s (5 turns, 10 invocations); agent-fitness:
`tool_selection_at_scale` fails at all three efforts as before, and `language_anchoring` dropped one case at
max (single run, not repeated). Multi-turn probes A/B/C: 385.8 / 108.7 / 57.5 s at effort=low, all recall
checks pass.

### Verdict by use (2026-09-02, RedHatAI checkpoint; default = MTP k=3)

| use | config | speed | verdict | evidence and conditions |
| --- | --- | --- | :--: | --- |
| one user, incl. Japanese | **default** = 262K / seqs 2 / MTP k=3 | 26.5 tok/s single | ✅ usable | Japanese verdicts match the single-Spark set at 2.2× the speed; 0 U+FFFD; commercial-safe (no drafter) |
| one user, code-heavy | 262K / seqs 2 / DFlash2 k=7 | 27.7 tok/s effective on code (25.6 single) | ✅ usable | code generation 5/5 (49/49), code fix 5/5 in 38.1 s; drafter is CC-BY-NC-ND |
| long documents, multi-turn | either | TTFT 139.4 s at 200K, prefill 1,369–1,480 tok/s | ✅ usable | needle recalled 2K–200K, multi-turn recall checks all pass |
| several to many users | 262K / seqs 8 / MTP k=4 | 25.3 single, 57.7 at C=4, 76.2 at C=8 | ✅ usable | 14.4 tok/s per user at C=4, 9.5 at C=8; pool holds 4.4 full-262K streams |
| resident agent | default | — | 🟡 conditional | tool-heavy clients still fail tool-selection-at-scale (probe verdict) |

## 1. One-factor sweep (`scripts/tune-sweep.sh`)

Forced 256 output tokens, greedy; C=1 is the median of 3 runs. `code_gen` = effective tok/s over 5 Japanese
spec-to-code problems at `reasoning_effort=low` (all 5/5 correct in every row) — it comes from the private kit of §2;
`scripts/tune-sweep.sh` reproduces every other column. "used" = `free -g` while serving.

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

### Throughput × concurrency (forced 256 tokens, greedy)

| config | C=1 | C=2 | C=4 | C=8 | source |
| --- | --: | --: | --: | --: | --- |
| 262K / seqs 2 / MTP k=4 (A-base) | 25.7 | 37.0 | — | — | `tune-A-base-throughput.json` |
| 262K / seqs 8 / MTP k=4 (s8) | 27.2 | 28.9 | 56.3 (15.5 per stream) | **76.4** (12.6 per stream) | `tune-s8-throughput.json` |
| 262K / seqs 2 / DFlash2 k=7 (**S**) | **30.8** | 38.1 | — | — | `tune-dflash7-throughput.json` |
| 65K / seqs 8 / DFlash2 k=7 (P, the §2 bench serve) | 32.3 | 42.9 | 70.8 | 58.9 (15.9 per stream) | `glm53-nvfp4-throughput.json` |

Eight slots do not cost single-stream speed. With the drafter the aggregate peaks at C=4 and falls at C=8;
with MTP-4 it keeps climbing to 76.4 (the single Spark's llama.cpp gives 64.0 at C=8). So: one user →
DFlash2, shared → MTP-4.

### Verdict by use (S is the default in `cluster.env.example`)

"Usable" = no harness task lost, latency fits the use, contracts such as `tool_choice` honoured, no broken
characters. "Conditional" = one of those comes with a caveat.

| use | config | speed | verdict | evidence and conditions |
| --- | --- | --- | :--: | --- |
| one user writing / fixing code | **S** = 262K / seqs 2 / DFlash2 k=7 / KV pin 3 GiB | 30.8–32.3 tok/s single, 28.2 tok/s effective on code generation | ✅ usable | code generation 5/5 (49/49 tests), code fix 5/5 in 54 s, tool calls 5/5 with `required` honoured; only tool selection from ~40 tools is weak |
| one user reading long documents, multi-turn | **S** | TTFT 139 s at 200K, prefill 1,391–1,510 tok/s | ✅ usable | needle recalled at 2K–200K, all 5 recall checks in the 10-turn probe pass; budget 2+ min for the first token at 200K |
| several to many users | 262K / seqs 8 / MTP k=4 / KV pin 9 GiB | 27.2 tok/s single, 56.3 aggregate at C=4, 76.4 at C=8 | ✅ usable | 15.5 tok/s per user at C=4, 12.6 at C=8 (the single Spark gives 8.4 at C=8); commercial-safe (no drafter) |
| Japanese that must not lose a character | no speculation + `LOGITS_PROCESSORS=utf8_guard_lp:Utf8GuardLogitsProcessor` | 14.6 tok/s single | 🟡 conditional | 0 U+FFFD in 24,590 chars, recall checks pass; the 10-turn probe goes from 354 s (DFlash2) to 479 s |
| resident agent | **S** | — | 🟡 conditional | a Hermes-Agent-class harness is fine; tool-heavy clients (opencode, OpenClaw) fail the tool-selection-at-scale probe at all three effort levels — a probe verdict, the harnesses themselves were not run |

Config P (65K / seqs 8 / DFlash2 k=7 / KV pin 4.1 GiB) is the serve used for the single-Spark comparison in §2
(same 65K / 8-slot shape as the llama.cpp baseline). It is not a separate recommendation: context length does
not change speed, and for shared use MTP-4 is both faster in aggregate and commercially usable.

## 2. Bench on config P (same harness as the author's single-Spark 2-bit GGUF article)

Japanese tasks, code generation, tool calls, code fix and agent-fitness probes come from a private
benchmark kit (not included). Numbers below are the kit's machine-scored results; the single-Spark column is
`unsloth/GLM-5.3-Flash-GGUF` UD-Q2_K_XL on llama.cpp (17.7 tok/s) from
[I tried running the 320B GLM-5.3-Flash on a single DGX Spark to measure the dividing line of practical usability](https://dev.classmethod.jp/en/articles/dgx-spark-glm-5-3-flash-first-touch/)
(2026-08-30). The 2× serve uses 65K / 8 slots to match that baseline's shape (context length does not move
speed, see §1); the one condition that could not be matched is the KV dtype (fp8_e4m3 here, f16 on llama.cpp).

| item | 2× Spark NVFP4 (P) | 1× Spark 2-bit GGUF |
| --- | --- | --- |
| throughput C=1 (TTFT) / C=8 aggregate (per stream) | 32.3 (0.26 s) / 58.9 (15.9) — 76.4 with MTP-4 at 262K, 76.0 at 65K | 17.7 (0.33 s) / 64.0 (8.4) |
| Japanese 10 tasks, effort=max (8192 budget) | 7/9 machine-checked, 13,833 tokens, **344 s** | 6/9, 14,708 tokens, 927 s |
| effort=low / high | 7/9 in 43 s / 5/9 in 60 s | 7/9 in 98 s / 7/9 in 141 s |
| code generation (max / high / low / 16k) | 5/5 solved, 49/49 tests in every row | 4/5 (8192) – 5/5 |
| tool calls | 5/5 single, parallel calls in one message, `tool_choice=required` honoured, pseudo-FS ✅ | same |
| code fix (3-bug mini repo) | 5/5, 6 turns, 17 tool calls, **54 s** | 5/5, 5 turns, 10 tool calls, 72 s |
| agent-fitness probes | only `tool_selection_at_scale` fails (all 3 effort levels) | low/high also fail `long_system_prompt` |
| multi-turn probes A (10 turns) / B (5) / C (12K-char doc + 8) | 354 s / 118 s / 62 s (effort=low), all recall checks pass | 433 s / 209 s / 88 s |

The one task that exhausts the 8192 budget at effort=max (a summary with proper nouns) does so on both
machines; with a 16,384 budget the 2× serve scores 6/9 in 431 s. The failing Japanese tasks are the same
set on both machines (the 50-character limit, the required word, the proper nouns), and the single run at
effort=high (5/9 vs 7/9) sits inside the run-to-run spread seen on the single Spark (7/9, 7/9, 6/9 across
three repeats), so it is not read as a capability gap.

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

Two simultaneous 200K prompts (400K tokens of context) both complete in 310 s; neither host froze at any
size. `--max-model-len` was 262K, so 1M was not attempted.

## 4. U+FFFD attribution (Japanese prose, `kabeuchi` probe ≈ 28K chars, temperature 1.0 / top_p 0.95)

| condition | U+FFFD | chars | per 10K |
| --- | --: | --: | --: |
| DFlash2 k=7 (S) | 7 | 27,757 | 2.5 |
| MTP k=4 | 5 | 27,823 | 1.8 |
| MTP k=4 + `top_k 40` | 13 | 27,475 | 4.7 |
| no speculation | 2 | 11,840 | 1.7 |
| no speculation + `min_p 0.05` | 9 / 2 | 11,787 / 14,635 | 7.6 / 1.4 |
| no speculation + bf16 KV (`auto`) | 14 | 27,937 | 5.0 |
| **no speculation + `patches/utf8_guard_lp.py`** | **0** | 24,590 | **0.0** |
| 1× Spark 2-bit GGUF, llama.cpp (same probe) | 0 | 28,000+ | 0 |

Token IDs returned with `return_token_ids` and decoded offline with the checkpoint's `tokenizer.json`
(`bench/garble_ids.py`) reproduce the replacement character at the same position in 3/3 samples: the
model emits an invalid byte sequence; the detokenizer is not at fault. Neither speculation, the sampler
(`top_k` / `min_p`) nor the KV dtype changes the rate. The `tokenizer.json` is byte-identical (md5) to
`zai-org/GLM-5.3-Flash`.

**Mechanism.** GLM-5.3's byte-level BPE has no single token for many Japanese shinjitai kanji: 測 is
`e6b8` + `ac`, 範 is `e7af` + `84`, 陥 `e999` + `a5`, 継 `e7b6` + `99`, 拡 `e68b` + `a1`, 遅 `e981` + `85` …
(a 2-byte fragment followed by a 1-byte continuation token; 1,095 of the 154,820 vocab entries are not valid
UTF-8 on their own). The failures are exactly these words — 現実 broke 11 times in 66 occurrences (現 is one token,
実 is `e5ae9f`, but the sequence is emitted through the fragment path), 効果測定 5 times, 許容範囲, 陥る,
桁, 継ぎ, 併用, 拡大, 毀損, 拠 — while 現場 (two whole tokens) never broke in 95 occurrences. The model
emits the 2-byte fragment and then skips the 1-byte continuation, jumping to the next character (hence
the duplicated 囲 in 範�囲囲). Which layer makes that continuation decision go wrong (NVFP4 marlin path,
v11 kernels) is not isolated; on a single Spark with the 2-bit GGUF under llama.cpp the same probe is clean.

**Mitigation.** `patches/utf8_guard_lp.py` is a vLLM v1 logits processor that masks, at every step, the
tokens that would make the byte stream invalid (after a token ending mid-character only the matching
continuation bytes are allowed; at a boundary, continuation-leading tokens are forbidden). Enable with
`LOGITS_PROCESSORS=utf8_guard_lp:Utf8GuardLogitsProcessor`; vLLM rejects custom logits processors while
speculative decoding is on, so this trades the drafter's speed for valid Japanese: **0 replacement characters in
24,590 chars**, recall checks in the multi-turn probes all pass, Japanese ratio ≥ 0.945, and the target words
(現実 ×15, 測定 ×14, 範囲 ×11, 継 ×4, 桁 ×2) are spelled correctly; decode runs at the no-speculation 14.6 tok/s,
and the 10-turn multi-turn probe takes 478.6 s instead of 353.8 s with DFlash2. Use the guard where a missing
character is unacceptable (contracts, procedures); keep speculation for chat and code.
