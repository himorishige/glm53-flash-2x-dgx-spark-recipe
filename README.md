# GLM-5.3-Flash (320B-A18B) on 2× DGX Spark — a recipe book

[日本語版 README](README.ja.md)

Config-as-code, measured numbers and the pitfalls for serving `LibertAIDAI/GLM-5.3-Flash-NVFP4` (181 GiB)
with vLLM tensor-parallel across two NVIDIA DGX Spark (GB10, 121 GiB each) over the QSFP link.
Everything here was run end to end on 2026-08-31 → 2026-09-01; the tables in `docs/measurements.md` are the
raw outputs of the scripts in this repository.

**What you get (config S, single user):** 30.8 tok/s decode, 1.4–1.5K tok/s prefill flat up to 200K,
needle recall at 200K, 15-minute boot. **Config P (a few users):** 31–32 tok/s single, 70 tok/s aggregate at
C=4. **MTP-4 with 8 slots:** 76 tok/s aggregate at C=8. Single Spark with the 2-bit GGUF was 17.7 tok/s.

## Just use this (as of 2026-09-01)

**Default = config S.** It is what `cluster.env.example` launches with no overrides.

| you want | run | measured |
| --- | --- | --- |
| **one user, long documents (default)** | `cluster.env` as shipped: 262K ctx / 2 slots / DFlash2 drafter k=7 / fp8 KV / marlin | 30.8 tok/s single, prefill ≈1.4K tok/s flat to 200K, needle at 200K ✅ |
| several people at once | `OVERRIDES="SPEC_METHOD=mtp MTP_NUM_TOKENS=4 KV_CACHE_MEMORY_BYTES=9663676416 MAX_NUM_SEQS=8"` | 27 tok/s single, 76 tok/s aggregate at C=8 |
| Japanese text that must be byte-exact | `OVERRIDES="MTP_NUM_TOKENS=0 LOGITS_PROCESSORS=utf8_guard_lp:Utf8GuardLogitsProcessor"` | 0 broken kanji (default: ≈2 per 10K chars), 14.6 tok/s |

```bash
cp cluster.env.example cluster.env      # set HEAD_HOST / WORKER_HOST / QSFP IPs / NIC names for your pair
scripts/sync-files.sh
scripts/start-worker.sh && sleep 25 && scripts/start-head.sh && scripts/health.sh   # ≈15 min to READY
# … use http://127.0.0.1:8888/v1 on the head (served model name glm-5.3-flash-nvfp4) …
scripts/stop-both.sh                    # always both ranks
```

Do not change: `--block-size 2304`, `--language-model-only`, `--moe-backend marlin`, `--gpu-memory-utilization 0.85`,
`--tool-call-parser glm47`, `--reasoning-parser deepseek_r1`, the v11 image, the top-k patch mount. Do not try
`flashinfer_cutlass` or utilisation 0.90 unattended (both hosts froze). Note: the DFlash2 drafter is CC-BY-NC-ND —
for commercial use take the MTP-4 row (26–27 tok/s) instead.

## TL;DR — what actually matters

1. **The stock day-0 image does not work on GB10.** `vllm/vllm-openai:glm53-flash-arm64-cu130` dies in
   warm-up with `pe_dim must be 64 for fp8_ds_mla` whatever KV dtype you pass (NoPE model + DSA indexer).
   Use tonyd2wild's `sm121-v11-dflash2` image and mount the SM121 top-k patch (both in this recipe).
2. **Speculation is the lever.** No spec 14.6 → MTP-4 25.7 → DFlash2 drafter 30.8 tok/s single-stream.
   CUDA graphs, KV dtype and context length barely move the needle on GB10.
3. **8 slots fit at 262K**, and give 76 tok/s aggregate with MTP-4. With the drafter the aggregate peaks at
   C≈4.
4. **Do not run FlashInfer MoE backends unattended at util 0.85** — `flashinfer_cutlass` froze both hosts
   until a power cycle. marlin is the backend here.
5. **Host prep is part of the recipe**: `vm.swappiness=0`, drop caches, worker-first launch, always tear
   down both ranks.
6. **Japanese prose comes back with ~2 replacement characters per 10K chars** (a kanji becomes U+FFFD).
   Cause: the tokenizer spells many Japanese kanji as a 2-byte fragment + a 1-byte continuation token and
   the model sometimes skips the continuation — independent of speculation, KV dtype and sampler.
   **Fix**: `patches/utf8_guard_lp.py`, a logits processor that forbids invalid continuations → 0 in 24,590
   chars, quality unchanged; it only works without speculative decoding (vLLM limitation), i.e. at 14.6 tok/s.
   Details in `docs/pitfalls.md` §9 / `docs/measurements.md` §4.

## Hardware and software

| | head (rank 0) | worker (rank 1) |
| --- | --- | --- |
| machine | DGX Spark, GB10, 121 GiB, driver 580.159.03 | same |
| QSFP | `enp1s0f1np1` 192.168.200.14, RDMA `rocep1s0f1`, MTU 9000 | 192.168.200.13 |
| runs | vLLM API server on 127.0.0.1:8888 | `--headless` worker |
| residents during the runs | nothing | one small container (≈ 100 MB) |

Docker 29 + compose v5, `nvidia-container-toolkit`, `tmux`, `rsync`, `python3` (stdlib only for the probes),
`hf` CLI and `uv` for downloads. All commands are issued from a Mac that can ssh to both nodes
(`node1` / `node2` aliases in `~/.ssh/config`); the nodes do not need keys for each other — inter-node
copies go through a read-only rsync daemon bound to the QSFP address.

## Layout

```
docker-compose.yml        symmetric service; rank-specific values are passed by scripts/start-*.sh;
                          the SM121 top-k patch is bind-mounted here
cluster.env.example       every knob (image, checkpoint revision, fabric, vLLM flags). Copy to cluster.env
patches/                  sparse_attn_indexer_kpool_sm121.py (tonyd2wild, Apache-2.0 — see NOTICE), utf8_guard_lp.py (UTF-8 guard)
scripts/                  download / verify / start / health / gate / stop / sweep / long-context loop
bench/                    throughput_probe.py, longctx.py, garble_tokens.py, garble_ids.py (stdlib)
docs/measurements.md      every number we measured, with conditions
docs/pitfalls.md          what broke, why, and the fix
results/ja/               the original Japanese result tables
```

## Step by step

### 0. Host preparation (both nodes)

```bash
sudo sysctl -w vm.swappiness=0            # resets on reboot; keep swap ON
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
ip link show enp1s0f1np1 | grep mtu       # 9000
```

Verify the fabric once: `scripts/nccl-check.sh` on the worker runs `all_gather_perf` / `all_reduce_perf`
(we saw 11.6 / 12.1 GB/s bus bandwidth; the June baseline with MTU 1500 was 1.5 GB/s).

### 1. Files, image, checkpoint

```bash
cp cluster.env.example cluster.env        # edit HEAD_HOST / WORKER_HOST / IPs / NIC names
scripts/sync-files.sh                     # mirrors compose, env, scripts, bench, patches to ~/glm53-cluster/ on both nodes

# head (node2): pull the image and the checkpoint once over the WAN (tmux)
ssh node2 'tmux new -d -s glm-img ~/glm53-cluster/scripts/pull-image.sh'
ssh node2 'tmux new -d -s glm-dl  ~/glm53-cluster/scripts/download-nvfp4.sh'   # 181 GiB; HF_HUB_DISABLE_XET=1
ssh node2 '~/glm53-cluster/scripts/download-dflash2.sh'                         # 2.2 GB drafter (CC-BY-NC-ND)

# worker (node1): copy over the QSFP through the rsync daemon (no ssh keys needed between nodes)
ssh node2 '~/glm53-cluster/scripts/rsyncd-node2.sh start'
ssh node1 'tmux new -d -s glm-sync ~/glm53-cluster/scripts/sync-checkpoint.sh'  # 181 GiB in ≈ 4 min
ssh node2 '~/glm53-cluster/scripts/save-image.sh $IMAGE' && ssh node1 '~/glm53-cluster/scripts/load-image-rsyncd.sh $IMAGE'
ssh node1 'rsync -a rsync://192.168.200.14:8873/dflash2/ ~/models/GLM-5.3-Flash-DFlash2/'

# verify (sha256 of all 123 LFS files against the Hub tree; ≈ 4 min per node)
ssh node2 'python3 ~/glm53-cluster/scripts/verify-checkpoint.py ~/models/GLM-5.3-Flash-NVFP4 \
  --repo LibertAIDAI/GLM-5.3-Flash-NVFP4 --revision caca4e6a4ebbd66f159d3d2fc256683fd6e27177 --sha256 --out ~/glm53-cluster/results'
```

`scripts/resume-after-dl.sh` and `scripts/prep-v11.sh` chain the steps above unattended and are idempotent.

### 2. First smoke

```bash
scripts/smoke.sh          # stop-both → drop caches → start-worker → 25 s → start-head → health → gates → results/smoke-<date>.md
```

Gates (`scripts/gate-2node.sh`, run on the head): G0 logs (RoCE NIC picked up, no NCCL WARN), G1
`/v1/models`, G2 a math prompt at default and `low` effort (checks for degenerate repetition and for
`<think>` leaking into content), G3 a `get_weather` tool call through the `glm47` parser.

### 3. Launch a configuration

Every knob in `cluster.env` can be overridden per launch:

```bash
# config S — single user / long context
OVERRIDES="SPEC_METHOD=dflash SPEC_MODEL=/models/GLM-5.3-Flash-DFlash2 MTP_NUM_TOKENS=7 KV_CACHE_MEMORY_BYTES=3221225472" \
  scripts/start-worker.sh && sleep 25 && OVERRIDES="…same…" scripts/start-head.sh && scripts/health.sh

# config P — a few interactive users
OVERRIDES="MAX_MODEL_LEN=65536 MAX_NUM_SEQS=8 KV_CACHE_MEMORY_BYTES=4445787956 SPEC_METHOD=dflash SPEC_MODEL=/models/GLM-5.3-Flash-DFlash2 MTP_NUM_TOKENS=7" …

# many users — MTP-4 with 8 slots at 262K
OVERRIDES="MAX_NUM_SEQS=8" …

# byte-exact Japanese — no speculation + UTF-8 guard
OVERRIDES="MTP_NUM_TOKENS=0 LOGITS_PROCESSORS=utf8_guard_lp:Utf8GuardLogitsProcessor" …

scripts/stop-both.sh      # ALWAYS: head down, then worker down
```

`MTP_NUM_TOKENS=0` drops `--speculative-config`, `ENFORCE_EAGER=0` drops `--enforce-eager`,
`KV_CACHE_MEMORY_BYTES=` (empty) lets vLLM size the KV pool from `--gpu-memory-utilization`,
`FLASHINFER_AUTOTUNE=0` passes `--kernel-config` with autotune/cutedsl warm-up off.

### 4. Measure

```bash
scripts/tune-sweep.sh                                   # scripts/tune-configs.txt → results/tuning/TUNING.md
SIZES=2048,8192,32768,131072,200000 CONC=1,2 scripts/longctx-run.sh s   # staged long context, host check between sizes
ssh node2 'python3 ~/glm53-cluster/bench/garble_ids.py --model glm-5.3-flash-nvfp4 --tokenizer ~/models/GLM-5.3-Flash-NVFP4/tokenizer.json --out /tmp/garble.json'  # needs `uv run --with tokenizers`
```

## The effective `vllm serve` line (config S)

```
vllm serve /models/GLM-5.3-Flash-NVFP4 --served-model-name glm-5.3-flash-nvfp4 --host 127.0.0.1 --port 8888
  --tensor-parallel-size 2 --distributed-executor-backend mp --nnodes 2 --node-rank 0 --master-addr 192.168.200.14 --master-port 25000
  --language-model-only --kv-cache-dtype fp8_e4m3 --block-size 2304 --kv-cache-memory-bytes 3221225472
  --max-num-batched-tokens 4096 --max-model-len 262144 --max-num-seqs 2 --gpu-memory-utilization 0.85
  --moe-backend marlin --enforce-eager --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser deepseek_r1
  --speculative-config '{"method":"dflash","model":"/models/GLM-5.3-Flash-DFlash2","num_speculative_tokens":7}' --trust-remote-code
```

(rank 1 adds `--headless`; NCCL / UCX / OMPI interface variables are in `docker-compose.yml`.)

## Credits

- Checkpoint: [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) (MIT), whose
  model card documents the NoPE/sparse-MLA fault and the `glm47` / `deepseek_r1` parser choices.
- Image, top-k patch and the DFlash2 launch line: [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark).
- Drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (CC-BY-NC-ND).
- Community threads that shaped the sweep: NVIDIA Developer Forums 381429, 381433, 381534, 381541, 381543, 381703;
  [amasu/glm53-flash-cluster](https://github.com/amasu/glm53-flash-cluster), [kingjones30/GLM-5.3-Flash-2x-DGX-Spark](https://github.com/kingjones30/GLM-5.3-Flash-2x-DGX-Spark),
  [Libertai/glm53-flash-vllm-gb10](https://github.com/Libertai/glm53-flash-vllm-gb10).

MIT — see `LICENSE` and `NOTICE`.
