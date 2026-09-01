# GLM-5.3-Flash（320B-A18B）を DGX Spark 2 台で動かすレシピブック

[English README](README.md)

`LibertAIDAI/GLM-5.3-Flash-NVFP4`（181 GiB）を、NVIDIA DGX Spark（GB10、121 GiB）2 台の QSFP 直結で
vLLM の TP=2 として動かすための config as code・実測値・落とし穴集です。2026-08-31 〜 09-01 に端から端まで
実走した結果で、`docs/measurements.md` の表はこのリポジトリのスクリプトの出力そのものです。

**構成 S（1 人で使う）**: decode 30.8 tok/s、prefill 1.4〜1.5K tok/s が 200K まで平坦、200K の合言葉を正答、
起動 15 分。**構成 P（数人）**: 単発 31〜32 tok/s、C=4 で合計 70 tok/s。**MTP-4 × 8 slots**: C=8 で合計 76 tok/s。
1 台の 2-bit GGUF は 17.7 tok/s でした。

## まずはこれで OK（2026-09-01 時点）

**既定 = 構成 S。** `cluster.env.example` をそのまま使えば、上書きなしでこの構成が上がります。

| やりたいこと | 起動 | 実測 |
| --- | --- | --- |
| **1 人で使う・長い文書を読ませる（既定）** | `cluster.env` のまま: 262K 文脈 / 2 slots / DFlash2 drafter k=7 / fp8 KV / marlin | 単発 30.8 tok/s、prefill ≈1.4K tok/s が 200K まで平坦、200K の合言葉 ✅ |
| 何人かで同時に使う | `OVERRIDES="SPEC_METHOD=mtp MTP_NUM_TOKENS=4 KV_CACHE_MEMORY_BYTES=9663676416 MAX_NUM_SEQS=8"` | 単発 27 tok/s、C=8 合計 76 tok/s |
| 日本語をバイト単位で正確に出したい | `OVERRIDES="MTP_NUM_TOKENS=0 LOGITS_PROCESSORS=utf8_guard_lp:Utf8GuardLogitsProcessor"` | 漢字の破損ゼロ（既定は ≈2 個 / 1 万字）、14.6 tok/s |

```bash
cp cluster.env.example cluster.env      # HEAD_HOST / WORKER_HOST / QSFP の IP / NIC 名を自分の 2 台に合わせる
scripts/sync-files.sh
scripts/start-worker.sh && sleep 25 && scripts/start-head.sh && scripts/health.sh   # READY まで約 15 分
# … head の http://127.0.0.1:8888/v1 を使う（served model name は glm-5.3-flash-nvfp4）…
scripts/stop-both.sh                    # 必ず両 rank
```

変えないもの: `--block-size 2304`、`--language-model-only`、`--moe-backend marlin`、`--gpu-memory-utilization 0.85`、
`--tool-call-parser glm47`、`--reasoning-parser deepseek_r1`、v11 image、top-k パッチの mount。`flashinfer_cutlass` と
util 0.90 は無人で試さない（両ホストが凍結）。注意: DFlash2 drafter は CC-BY-NC-ND なので、商用は MTP-4 の行（26〜27 tok/s）を使う。

## 要点

1. **day-0 の stock image は GB10 で動きません。** `vllm/vllm-openai:glm53-flash-arm64-cu130` は KV dtype に
   関係なく warmup で `pe_dim must be 64 for fp8_ds_mla` を出して落ちます（NoPE モデル × DSA indexer）。
   tonyd2wild の `sm121-v11-dflash2` image + SM121 top-k パッチ（同梱）を使います。
2. **効くのは投機デコードです。** 投機なし 14.6 → MTP-4 25.7 → DFlash2 drafter 30.8 tok/s（単発）。
   CUDA graph・KV dtype・文脈長は GB10 ではほぼ効きません。
3. **262K のまま 8 slots が載り**、MTP-4 で合計 76 tok/s。drafter を付けると合計は C≈4 が峰。
4. **FlashInfer 系 MoE backend を util 0.85 で無人運用しない**——`flashinfer_cutlass` で両ホストが凍結し、
   物理再起動になりました。backend は marlin です。
5. **ホスト準備もレシピの一部**: `vm.swappiness=0`、drop_caches、worker → head の順、両 rank を必ず落とす。
6. **日本語の散文で 1 万字あたり約 2 文字が置換文字（U+FFFD）になります。** 原因は tokenizer: 日本語の
   新字体の多くが「2 バイト断片 + 1 バイト継続」の 2 トークンで綴られ、モデルが継続を飛ばすことがある
   （投機・KV dtype・サンプラーに依らず発生）。**対策**は `patches/utf8_guard_lp.py`（不正な継続を遮る
   logits processor）で、24,590 字でゼロ・品質そのまま。vLLM の制約で投機デコードとは併用できず 14.6 tok/s。
   詳細は `docs/pitfalls.md` §9 と `docs/measurements.md` §4。

## 構成

| | head（rank 0） | worker（rank 1） |
| --- | --- | --- |
| 機体 | DGX Spark、GB10、121 GiB、driver 580.159.03 | 同左 |
| QSFP | `enp1s0f1np1` 192.168.200.14、RDMA `rocep1s0f1`、MTU 9000 | 192.168.200.13 |
| 役割 | vLLM API サーバ 127.0.0.1:8888 | `--headless` worker |

Docker 29 + compose v5、nvidia-container-toolkit、tmux、rsync、python3（プローブは stdlib のみ）、ダウンロードに
`hf` CLI と `uv`。操作はすべて両ノードへ ssh できる Mac から行い（`~/.ssh/config` の `node1` / `node2`）、
ノード間のコピーは QSFP アドレスに bind した読み取り専用 rsync デーモン経由なので、ノード同士の鍵は不要です。

## ディレクトリ

```
docker-compose.yml        対称な service。rank 固有の値は scripts/start-*.sh が渡す。SM121 top-k パッチをここで bind mount
cluster.env.example       全 knob（image / checkpoint revision / fabric / vLLM フラグ）。cluster.env にコピーして使う
patches/                  sparse_attn_indexer_kpool_sm121.py（tonyd2wild、Apache-2.0 — NOTICE 参照）、utf8_guard_lp.py（UTF-8 ガード）
scripts/                  取得 / 照合 / 起動 / health / ゲート / 停止 / スイープ / 段階長文
bench/                    throughput_probe.py, longctx.py, garble_tokens.py, garble_ids.py（stdlib）
docs/measurements.md      実測値と条件のすべて（英語）
docs/pitfalls.md          何が壊れ、なぜで、どう直したか（英語）
results/ja/               日本語の元の結果表（チューニング / 本戦 / 長文）
```

## 手順

### 0. ホスト準備（両ノード）

```bash
sudo sysctl -w vm.swappiness=0            # reboot で戻る。swap 自体は ON のまま
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
ip link show enp1s0f1np1 | grep mtu       # 9000
```

fabric は最初に一度 `scripts/nccl-check.sh`（worker 発の `all_gather_perf` / `all_reduce_perf`）で確認します
（実測 11.6 / 12.1 GB/s。MTU 1500 の 6 月時点は 1.5 GB/s）。

### 1. ファイル・image・checkpoint

```bash
cp cluster.env.example cluster.env        # HEAD_HOST / WORKER_HOST / IP / NIC 名を自環境に
scripts/sync-files.sh                     # compose・env・scripts・bench・patches を両ノードの ~/glm53-cluster/ へ

# head（node2）: image と checkpoint を WAN で 1 回だけ取得（tmux）
ssh node2 'tmux new -d -s glm-img ~/glm53-cluster/scripts/pull-image.sh'
ssh node2 'tmux new -d -s glm-dl  ~/glm53-cluster/scripts/download-nvfp4.sh'   # 181 GiB、HF_HUB_DISABLE_XET=1
ssh node2 '~/glm53-cluster/scripts/download-dflash2.sh'                         # drafter 2.2 GB（CC-BY-NC-ND）

# worker（node1）: QSFP の rsync デーモン経由で複製（ノード間の ssh 鍵は不要）
ssh node2 '~/glm53-cluster/scripts/rsyncd-node2.sh start'
ssh node1 'tmux new -d -s glm-sync ~/glm53-cluster/scripts/sync-checkpoint.sh'  # 181 GiB が約 4 分
ssh node2 '~/glm53-cluster/scripts/save-image.sh $IMAGE' && ssh node1 '~/glm53-cluster/scripts/load-image-rsyncd.sh $IMAGE'
ssh node1 'rsync -a rsync://192.168.200.14:8873/dflash2/ ~/models/GLM-5.3-Flash-DFlash2/'

# 照合（LFS 123 本の sha256 を Hub の tree と突き合わせ。1 ノード約 4 分）
ssh node2 'python3 ~/glm53-cluster/scripts/verify-checkpoint.py ~/models/GLM-5.3-Flash-NVFP4 \
  --repo LibertAIDAI/GLM-5.3-Flash-NVFP4 --revision caca4e6a4ebbd66f159d3d2fc256683fd6e27177 --sha256 --out ~/glm53-cluster/results'
```

`scripts/resume-after-dl.sh` と `scripts/prep-v11.sh` は上の手順を無人でつなぐ冪等スクリプトです。

### 2. 初回スモーク

```bash
scripts/smoke.sh          # stop-both → drop_caches → start-worker → 25 秒 → start-head → health → ゲート → results/smoke-<date>.md
```

ゲート（`scripts/gate-2node.sh`、head で実行）: G0 ログ（RoCE NIC を掴んだか、NCCL WARN なし）、G1 `/v1/models`、
G2 数学 1 問を既定 effort と `low` で（反復と `<think>` の content 漏れを検出）、G3 `glm47` parser での tool call。

### 3. 構成を起動する

`cluster.env` の全 knob は起動時に上書きできます。

```bash
# 構成 S — 1 人・長文
OVERRIDES="SPEC_METHOD=dflash SPEC_MODEL=/models/GLM-5.3-Flash-DFlash2 MTP_NUM_TOKENS=7 KV_CACHE_MEMORY_BYTES=3221225472" \
  scripts/start-worker.sh && sleep 25 && OVERRIDES="…同じ…" scripts/start-head.sh && scripts/health.sh

# 構成 P — 数人で対話
OVERRIDES="MAX_MODEL_LEN=65536 MAX_NUM_SEQS=8 KV_CACHE_MEMORY_BYTES=4445787956 SPEC_METHOD=dflash SPEC_MODEL=/models/GLM-5.3-Flash-DFlash2 MTP_NUM_TOKENS=7" …

# 多人数 — 262K のまま MTP-4 × 8 slots
OVERRIDES="MAX_NUM_SEQS=8" …

# 日本語をバイト単位で正確に — 投機なし + UTF-8 ガード
OVERRIDES="MTP_NUM_TOKENS=0 LOGITS_PROCESSORS=utf8_guard_lp:Utf8GuardLogitsProcessor" …

scripts/stop-both.sh      # 必ず head → worker の順で落とす
```

`MTP_NUM_TOKENS=0` で `--speculative-config` を外し、`ENFORCE_EAGER=0` で `--enforce-eager` を外し、
`KV_CACHE_MEMORY_BYTES=`（空）で KV プールを `--gpu-memory-utilization` 任せにし、`FLASHINFER_AUTOTUNE=0` で
autotune / cutedsl warmup を切る `--kernel-config` を渡します。

### 4. 測る

```bash
scripts/tune-sweep.sh                                   # scripts/tune-configs.txt → results/tuning/TUNING.md
SIZES=2048,8192,32768,131072,200000 CONC=1,2 scripts/longctx-run.sh s   # 段階長文。サイズ間で両ホストを確認
ssh node2 'python3 ~/glm53-cluster/bench/garble_ids.py --model glm-5.3-flash-nvfp4 --tokenizer ~/models/GLM-5.3-Flash-NVFP4/tokenizer.json --out /tmp/garble.json'  # `uv run --with tokenizers` で
```

## 実効の `vllm serve` 行（構成 S）

```
vllm serve /models/GLM-5.3-Flash-NVFP4 --served-model-name glm-5.3-flash-nvfp4 --host 127.0.0.1 --port 8888
  --tensor-parallel-size 2 --distributed-executor-backend mp --nnodes 2 --node-rank 0 --master-addr 192.168.200.14 --master-port 25000
  --language-model-only --kv-cache-dtype fp8_e4m3 --block-size 2304 --kv-cache-memory-bytes 3221225472
  --max-num-batched-tokens 4096 --max-model-len 262144 --max-num-seqs 2 --gpu-memory-utilization 0.85
  --moe-backend marlin --enforce-eager --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser deepseek_r1
  --speculative-config '{"method":"dflash","model":"/models/GLM-5.3-Flash-DFlash2","num_speculative_tokens":7}' --trust-remote-code
```

（rank 1 は `--headless` が付く。NCCL / UCX / OMPI のインターフェース指定は `docker-compose.yml`。）

## クレジット

- checkpoint: [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)（MIT）。
  NoPE / sparse-MLA の不具合と `glm47` / `deepseek_r1` の parser 選択はこのモデルカードの記述どおりでした
- image・top-k パッチ・DFlash2 の起動行: [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)
- drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)（CC-BY-NC-ND）
- スイープの設計に使ったコミュニティの実測: NVIDIA Developer Forums 381429 / 381433 / 381534 / 381541 / 381543 / 381703、
  [amasu/glm53-flash-cluster](https://github.com/amasu/glm53-flash-cluster)、[kingjones30/GLM-5.3-Flash-2x-DGX-Spark](https://github.com/kingjones30/GLM-5.3-Flash-2x-DGX-Spark)、
  [Libertai/glm53-flash-vllm-gb10](https://github.com/Libertai/glm53-flash-vllm-gb10)

MIT — `LICENSE` と `NOTICE` を参照。
