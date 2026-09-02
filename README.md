# GLM-5.3-Flash (320B-A18B) on 2× DGX Spark — recipe

[日本語版 README](README.ja.md)

Config-as-code for serving `RedHatAI/GLM-5.3-Flash-NVFP4` (184.3 GiB) with vLLM tensor-parallel across two
NVIDIA DGX Spark (GB10, 121 GiB each) over the QSFP link. Every number here was measured on this exact
setup on 2026-09-01/02 (`docs/measurements.md`). The initial default, the LibertAIDAI checkpoint, intermittently
breaks CJK multi-byte characters (`docs/pitfalls.md` #9); the default switched to RedHatAI on 2026-09-02.

The write-up behind the recipe — what did not boot, how the Japanese U+FFFD was tracked down to the
tokenizer, and a verdict table by use — is on DevelopersIO:
[I measured the value of running 320B GLM-5.3-Flash on 2 DGX Spark units to see if having 2 units is worth it](https://dev.classmethod.jp/en/articles/dgx-spark-2node-glm-5-3-flash-nvfp4-vllm/) (published 2026-09-01, re-measured on the RedHat checkpoint 2026-09-02).

## Just use this

**Default = MTP k=3 (262K / 2 slots)** — what `cluster.env.example` launches with no overrides.

| you want | run | measured |
| --- | --- | --- |
| **one user: code, long documents, multi-turn, Japanese (default)** | `cluster.env` as shipped: 262K ctx / 2 slots / MTP k=3 / fp8 KV / marlin | 26.5 tok/s single, 0 U+FFFD across 24–27K-char Japanese probes × 3 configs, commercial-safe (bundled head) |
| best effective code-generation speed | `OVERRIDES="SPEC_METHOD=dflash SPEC_MODEL=/models/GLM-5.3-Flash-DFlash2 MTP_NUM_TOKENS=7"` | 27.7 tok/s effective on code generation (25.6 single). The drafter is CC-BY-NC-ND (non-commercial) |
| several to many people at once | `OVERRIDES="SPEC_METHOD=mtp MTP_NUM_TOKENS=4 MAX_NUM_SEQS=8"` | 25.3 tok/s single, 57.7 aggregate at C=4 (14.4 per user), 76.2 at C=8 (9.5 per user) |

Long context recalls the needle at every size from 2K to 200K, with prefill ≈1.4K tok/s flat to 200K
(TTFT 139 s at 200K). Prefill does not depend on the speculation method (measured on the DFlash2 serve, 2026-09-02).

```bash
cp cluster.env.example cluster.env      # HEAD_HOST / WORKER_HOST / QSFP IPs / NIC names for your pair
scripts/sync-files.sh                   # mirror compose, env, scripts, bench, patches to ~/glm53-cluster/ on both nodes
scripts/start-worker.sh && sleep 25 && scripts/start-head.sh && scripts/health.sh   # ≈10 min to READY
# → OpenAI-compatible API on the head: http://127.0.0.1:8888/v1, model name glm-5.3-flash-nvfp4
scripts/stop-both.sh                    # always both ranks
```

Fixed values (already in `cluster.env.example`, leave them): image `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`,
the SM121 top-k patch mount (in `docker-compose.yml`), `--block-size 2304` (the page size the fp8 paged MQA
needs — without it output corrupts with no error), `--language-model-only` (text only by default; images are one knob
away, see [Images](#images) — the vision tower itself is 1.05 GiB), `--moe-backend marlin`, `--gpu-memory-utilization 0.85`, `--kv-cache-dtype fp8_e4m3`, `--enforce-eager`,
`--tool-call-parser glm47` (`glm` silently drops tool calls), `--reasoning-parser deepseek_r1`. Context length does
not move speed (262K vs 65K: same tok/s), so there is no reason to shrink it. The KV pool is not pinned — the
profiler sizes it (7.48 GiB = 1,151,844 tokens = 4.4 full-262K streams on this checkpoint). The default uses the
bundled MTP head, so there is no drafter dependency and it is commercial-safe as shipped.

## Hardware and software

| | head (rank 0) | worker (rank 1) |
| --- | --- | --- |
| machine | DGX Spark, GB10, 121 GiB, driver 580.159.03 | same |
| QSFP | `enp1s0f1np1` 192.168.200.14, RDMA `rocep1s0f1`, MTU 9000 | 192.168.200.13 |
| runs | vLLM API server on 127.0.0.1:8888 | `--headless` worker |

Docker 29 + compose v5, `nvidia-container-toolkit`, `tmux`, `rsync`, `python3` (the probes are stdlib; only
`bench/garble_ids.py` needs `tokenizers`), `hf` CLI and `uv` for downloads, and passwordless `sudo` on both nodes
(the scripts set `vm.swappiness` and drop caches with `sudo -n`). All commands run from a Mac that can ssh to both
nodes (`node1` / `node2` aliases in `~/.ssh/config`); the nodes need no keys for each other — inter-node copies go
through a read-only rsync daemon on the QSFP address. The QSFP addresses and NIC names in the table are this
pair's: put yours in `cluster.env` (`HEAD_IP` / `WORKER_IP` / `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA`) — every
script reads them from there.

## Step by step

### 0. Host preparation (both nodes, before every launch)

```bash
sudo sysctl -w vm.swappiness=0            # resets on reboot; keep swap ON
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
ip link show enp1s0f1np1 | grep mtu       # 9000
```

`scripts/nccl-check.sh` (worker) runs `all_gather_perf` / `all_reduce_perf` once to confirm the fabric
(11.6 / 12.1 GB/s bus bandwidth here). It is optional and expects an MPI build of nccl-tests in `~/nccl-tests`
(NCCL in `~/nccl`) on the worker plus worker → head ssh (agent forwarding).

### 1. Files, image, checkpoint, drafter

```bash
cp cluster.env.example cluster.env && scripts/sync-files.sh

# head (node2): fetch once over the WAN (tmux)
ssh node2 'tmux new -d -s glm-img ~/glm53-cluster/scripts/pull-image.sh'
ssh node2 'tmux new -d -s glm-dl  ~/glm53-cluster/scripts/download-nvfp4.sh'   # 184.3 GiB
ssh node2 '~/glm53-cluster/scripts/download-dflash2.sh'                         # 2.2 GB drafter

# worker (node1): copy over the QSFP through the rsync daemon
ssh node2 '~/glm53-cluster/scripts/rsyncd-node2.sh start'
ssh node1 'tmux new -d -s glm-sync ~/glm53-cluster/scripts/sync-checkpoint.sh'  # 184.3 GiB in ≈3 min
IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
ssh node2 "~/glm53-cluster/scripts/save-image.sh $IMAGE" && ssh node1 "~/glm53-cluster/scripts/load-image-rsyncd.sh $IMAGE"
ssh node1 'rsync -a rsync://192.168.200.14:8873/dflash2/ ~/models/GLM-5.3-Flash-DFlash2/'   # 192.168.200.14 = your HEAD_IP

# verify the checkpoint (sha256 of every LFS file against the Hub tree, ≈4 min per node)
ssh node2 'python3 ~/glm53-cluster/scripts/verify-checkpoint.py ~/models/GLM-5.3-Flash-NVFP4 \
  --repo RedHatAI/GLM-5.3-Flash-NVFP4 --revision 36c184c6cda000a481711306df5adde42f63321a --sha256 --out ~/glm53-cluster/results'
```

`scripts/resume-after-dl.sh` and `scripts/prep-v11.sh` chain these steps unattended (idempotent).

### 2. First smoke

```bash
scripts/smoke.sh          # stop-both → drop caches → start-worker → 25 s → start-head → health → gates → results/smoke-<date>.md
```

Gates (`scripts/gate-2node.sh` on the head): G0 logs (RoCE NIC picked up, no NCCL WARN), G1 `/v1/models`,
G2 a math prompt at default and `low` effort, G3 a `get_weather` tool call through the `glm47` parser.

### 3. Launch, switch, stop

Every knob in `cluster.env` can be overridden per launch with `OVERRIDES="K=V …"` on `start-worker.sh`
and `start-head.sh` (same string on both). `MTP_NUM_TOKENS=0` drops speculation, `ENFORCE_EAGER=0` drops
`--enforce-eager`, `KV_CACHE_MEMORY_BYTES=` (empty) lets vLLM size the KV pool. Always finish with
`scripts/stop-both.sh` (head, then worker).

### 4. Measure your own pair

```bash
scripts/tune-sweep.sh                                   # scripts/tune-configs.txt → results/tuning/TUNING.md
SIZES=2048,8192,32768,131072,200000 CONC=1,2 scripts/longctx-run.sh s   # staged long context with a host check between sizes
```

## The effective `vllm serve` line (default = MTP k=3, rank 0)

```
vllm serve /models/GLM-5.3-Flash-NVFP4 --served-model-name glm-5.3-flash-nvfp4 --host 127.0.0.1 --port 8888
  --tensor-parallel-size 2 --distributed-executor-backend mp --nnodes 2 --node-rank 0 --master-addr 192.168.200.14 --master-port 25000
  --language-model-only --kv-cache-dtype fp8_e4m3 --block-size 2304
  --max-num-batched-tokens 4096 --max-model-len 262144 --max-num-seqs 2 --gpu-memory-utilization 0.85
  --moe-backend marlin --enforce-eager --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser deepseek_r1
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --trust-remote-code
```

Rank 1 adds `--headless`; NCCL / UCX / OMPI interface variables are in `docker-compose.yml`.

## Images

GLM-5.3-Flash ships a vision tower and the RedHat checkpoint keeps it: 1.05 GiB BF16 per rank, left unquantized
by the recipe, and its `chat_template.jinja` already emits `<|begin_of_image|><|image|><|end_of_image|>`. The
default is text only because that is what every number here was measured with. To accept images:

```bash
OV="LANGUAGE_MODEL_ONLY=0 MAX_MODEL_LEN=131072"
OVERRIDES="$OV" scripts/start-worker.sh && sleep 25 && OVERRIDES="$OV" scripts/start-head.sh && scripts/health.sh
```

`LANGUAGE_MODEL_ONLY=0` drops `--language-model-only` and appends `VISION_ARGS` from `cluster.env`
(`--limit-mm-per-prompt {"image":4} --mm-processor-kwargs {"max_image_tokens":2048}`: up to 4 images per prompt,
≈1.6 MP each before the processor shrinks them). The "15.7 GiB" that circulated for this image is not the tower;
it is warmup profiling with the unbounded default of 8,000 tokens per image. Bounded like this the multimodal
profile costs about 1 GiB of KV pool. Measured 2026-09-02 on the 8-slot MTP k=4 profile: boot 621 s, KV pool
3.31 GiB at 131K (262K needs 2.23 GiB, so 262K fits too), ViT attention auto-selects FLASH_ATTN, and three
vision probes from the author's bench kit (bar-chart values, Japanese OCR, PPE enumeration with a hallucination
watch) score chart 4/4, OCR CER 0.0, PPE fail — the same 2/3 as the single-Spark GGUF run — at 2.5 s / 4.4 s per
image (single Spark: 7.8 / 13.5). Send images as OpenAI `image_url` parts (data URL or http). Video is on at
vLLM's default limit of 1 but unmeasured. Details in `docs/measurements.md` §5.

## Layout

```
docker-compose.yml        symmetric service; rank-specific values come from scripts/start-*.sh; patches are bind-mounted here
cluster.env.example       every knob — defaults are MTP k=3 (262K / 2 slots)
patches/                  sparse_attn_indexer_kpool_sm121.py (tonyd2wild, Apache-2.0 — see NOTICE), utf8_guard_lp.py (UTF-8 guard for LibertAI-checkpoint users)
scripts/                  download / verify / start / health / gate / stop / sweep / long-context loop
bench/                    throughput_probe.py, longctx.py, garble_tokens.py, garble_ids.py (stdlib; garble_ids.py needs tokenizers)
docs/measurements.md      every number we measured, with conditions — 11-config tuning sweep, throughput × concurrency, bench vs a single Spark, long context, U+FFFD attribution, verdict by use
docs/pitfalls.md          what did not work and why (image, KV dtype, MoE backends, env-file, benchmark hygiene, Japanese U+FFFD, flags that fail silently)
results/ja/               the original Japanese result tables
```

## Credits

- Checkpoint: [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) (no license declared on the HF repo; the base model is MIT — default since 2026-09-02) / initial default [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) (MIT, see `docs/pitfalls.md` #9)
- Image, top-k patch and the DFlash2 launch line: [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)
- Drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (CC-BY-NC-ND)
- Single-Spark baseline (2-bit GGUF on llama.cpp) used in the comparison: [I tried running the 320B GLM-5.3-Flash on a single DGX Spark to measure the dividing line of practical usability](https://dev.classmethod.jp/en/articles/dgx-spark-glm-5-3-flash-first-touch/) (2026-08-30)
- Community measurements that shaped the sweep: NVIDIA Developer Forums 381429, 381433, 381534, 381541, 381543, 381703;
  [amasu/glm53-flash-cluster](https://github.com/amasu/glm53-flash-cluster), [kingjones30/GLM-5.3-Flash-2x-DGX-Spark](https://github.com/kingjones30/GLM-5.3-Flash-2x-DGX-Spark),
  [Libertai/glm53-flash-vllm-gb10](https://github.com/Libertai/glm53-flash-vllm-gb10)

MIT — see `LICENSE` and `NOTICE`.
