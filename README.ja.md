# GLM-5.3-Flash（320B-A18B）を DGX Spark 2 台で動かすレシピ

[English README](README.md)

`RedHatAI/GLM-5.3-Flash-NVFP4`（184.3 GiB）を、NVIDIA DGX Spark（GB10、121 GiB）2 台の QSFP 直結で vLLM の
TP=2 として動かすための config as code です。数値はすべてこの構成で 2026-09-01〜02 に実測したものです（`docs/measurements.md`）。
初出の既定だった LibertAIDAI 版 checkpoint は日本語の多バイト文字を壊すことがあり（`docs/pitfalls.md` #9）、2026-09-02 に RedHatAI 版へ差し替えました。

このレシピの裏側（起動できなかった image、日本語の U+FFFD を tokenizer まで追った経緯、用途別の判定表）は DevelopersIO の記事にまとめています:
[320B の GLM-5.3-Flash を DGX Spark 2 台で動かして 2 台にする価値を測ってみた](https://dev.classmethod.jp/articles/dgx-spark-2node-glm-5-3-flash-nvfp4-vllm/)（2026-09-01 公開、09-02 に RedHat 版で全数値を改稿）

## まずはこれで OK

**既定 = MTP k=3（262K / 2 slots）。** `cluster.env.example` をそのまま使えば、上書きなしでこの構成が上がります。

| やりたいこと | 起動 | 実測 |
| --- | --- | --- |
| **1 人で使う: コード、長い文書、壁打ち、日本語（既定）** | `cluster.env` のまま: 262K 文脈 / 2 slots / MTP k=3 / fp8 KV / marlin | 単発 26.5 tok/s、日本語 24〜27K 字 × 3 条件で U+FFFD 0、内蔵ヘッドなので商用可 |
| コード生成の実効速度を最優先する | `OVERRIDES="SPEC_METHOD=dflash SPEC_MODEL=/models/GLM-5.3-Flash-DFlash2 MTP_NUM_TOKENS=7"` | コード生成の実効 27.7 tok/s（単発 25.6）。drafter は CC-BY-NC-ND（非商用） |
| 数人から多人数で同時に使う | `OVERRIDES="SPEC_METHOD=mtp MTP_NUM_TOKENS=4 MAX_NUM_SEQS=8"` | 単発 25.3 tok/s、C=4 合計 57.7（1 人あたり 14.4）、C=8 合計 76.2（1 人あたり 9.5） |

長文は 2K〜200K の全サイズで合言葉を正答し、prefill ≈1.4K tok/s が 200K まで平坦（200K の TTFT 139 秒）。
prefill は投機に依存しないため構成間で共通です（実測は DFlash2 構成、2026-09-02）。

```bash
cp cluster.env.example cluster.env      # HEAD_HOST / WORKER_HOST / QSFP の IP / NIC 名を自分の 2 台に合わせる
scripts/sync-files.sh                   # compose・env・scripts・bench・patches を両ノードの ~/glm53-cluster/ へ
scripts/start-worker.sh && sleep 25 && scripts/start-head.sh && scripts/health.sh   # READY まで約 10 分
# → head の http://127.0.0.1:8888/v1 に OpenAI 互換 API、model 名は glm-5.3-flash-nvfp4
scripts/stop-both.sh                    # 必ず両 rank
```

固定値（`cluster.env.example` に入っている。触らない）: image `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`、
SM121 top-k パッチの mount（`docker-compose.yml`）、`--block-size 2304`（fp8 の paged MQA が要求する page サイズ。外すとエラーなしに出力が壊れる）、
`--language-model-only`（マルチモーダルの front-end を読まない。無いと 15.7 GiB 余計に食う）、`--moe-backend marlin`、
`--gpu-memory-utilization 0.85`、`--kv-cache-dtype fp8_e4m3`、`--enforce-eager`、`--tool-call-parser glm47`（`glm` だと tool call が黙って消える）、
`--reasoning-parser deepseek_r1`。文脈長は速度に効かない（262K と 65K で同じ tok/s）ので、削る理由はありません。
KV プールは pin せず profiler に任せます（この checkpoint で 7.48 GiB = 1,151,844 トークン = 262K 満杯 4.4 本）。
既定は内蔵 MTP ヘッドなので drafter への依存はなく、そのまま商用可です。

## 構成

| | head（rank 0） | worker（rank 1） |
| --- | --- | --- |
| 機体 | DGX Spark、GB10、121 GiB、driver 580.159.03 | 同左 |
| QSFP | `enp1s0f1np1` 192.168.200.14、RDMA `rocep1s0f1`、MTU 9000 | 192.168.200.13 |
| 役割 | vLLM API サーバ 127.0.0.1:8888 | `--headless` worker |

Docker 29 + compose v5、nvidia-container-toolkit、tmux、rsync、python3（プローブは stdlib。`bench/garble_ids.py` だけ `tokenizers` が要る）、
ダウンロードに `hf` CLI と `uv`、両ノードでパスワードなしの `sudo`（スクリプトが `sudo -n` で `vm.swappiness` と drop_caches を触る）。
操作はすべて両ノードへ ssh できる Mac から行い（`~/.ssh/config` の `node1` / `node2`）、
ノード間のコピーは QSFP アドレスの読み取り専用 rsync デーモン経由なので、ノード同士の鍵は不要です。
表の QSFP アドレスと NIC 名はこの 2 台のもので、自分の値は `cluster.env`（`HEAD_IP` / `WORKER_IP` / `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA`）に書きます。
スクリプトはすべてそこから読みます。

## 手順

### 0. ホスト準備（両ノード、毎回の起動前）

```bash
sudo sysctl -w vm.swappiness=0            # reboot で戻る。swap 自体は ON のまま
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
ip link show enp1s0f1np1 | grep mtu       # 9000
```

fabric は最初に一度 `scripts/nccl-check.sh`（worker）で確認します（`all_gather_perf` / `all_reduce_perf`、ここでは 11.6 / 12.1 GB/s）。
任意の手順で、worker に MPI 版 nccl-tests（`~/nccl-tests`、NCCL は `~/nccl`）と worker → head の ssh（agent forwarding）が要ります。

### 1. ファイル・image・checkpoint・drafter

```bash
cp cluster.env.example cluster.env && scripts/sync-files.sh

# head（node2）: WAN で 1 回だけ取得（tmux）
ssh node2 'tmux new -d -s glm-img ~/glm53-cluster/scripts/pull-image.sh'
ssh node2 'tmux new -d -s glm-dl  ~/glm53-cluster/scripts/download-nvfp4.sh'   # 184.3 GiB
ssh node2 '~/glm53-cluster/scripts/download-dflash2.sh'                         # drafter 2.2 GB

# worker（node1）: QSFP の rsync デーモン経由で複製
ssh node2 '~/glm53-cluster/scripts/rsyncd-node2.sh start'
ssh node1 'tmux new -d -s glm-sync ~/glm53-cluster/scripts/sync-checkpoint.sh'  # 184.3 GiB が約 3 分
IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
ssh node2 "~/glm53-cluster/scripts/save-image.sh $IMAGE" && ssh node1 "~/glm53-cluster/scripts/load-image-rsyncd.sh $IMAGE"
ssh node1 'rsync -a rsync://192.168.200.14:8873/dflash2/ ~/models/GLM-5.3-Flash-DFlash2/'   # 192.168.200.14 = 自分の HEAD_IP

# 照合（全 LFS ファイルの sha256 を Hub の tree と突き合わせ。1 ノード約 4 分）
ssh node2 'python3 ~/glm53-cluster/scripts/verify-checkpoint.py ~/models/GLM-5.3-Flash-NVFP4 \
  --repo RedHatAI/GLM-5.3-Flash-NVFP4 --revision 36c184c6cda000a481711306df5adde42f63321a --sha256 --out ~/glm53-cluster/results'
```

`scripts/resume-after-dl.sh` と `scripts/prep-v11.sh` は上の手順を無人でつなぐ冪等スクリプトです。

### 2. 初回スモーク

```bash
scripts/smoke.sh          # stop-both → drop_caches → start-worker → 25 秒 → start-head → health → ゲート → results/smoke-<date>.md
```

ゲート（`scripts/gate-2node.sh`、head で実行）: G0 ログ（RoCE NIC を掴んだか、NCCL WARN なし）、G1 `/v1/models`、
G2 数学 1 問を既定 effort と `low` で、G3 `glm47` parser での tool call。

### 3. 起動・切り替え・停止

`cluster.env` の全 knob は `start-worker.sh` / `start-head.sh` に同じ `OVERRIDES="K=V …"` を付けて起動時に上書きできます。
`MTP_NUM_TOKENS=0` で投機を外し、`ENFORCE_EAGER=0` で `--enforce-eager` を外し、`KV_CACHE_MEMORY_BYTES=`（空）で KV プールを
vLLM 任せにします。終わりは必ず `scripts/stop-both.sh`（head → worker）。

### 4. 自分の 2 台で測る

```bash
scripts/tune-sweep.sh                                   # scripts/tune-configs.txt → results/tuning/TUNING.md
SIZES=2048,8192,32768,131072,200000 CONC=1,2 scripts/longctx-run.sh s   # 段階長文。サイズ間で両ホストを確認
```

## 実効の `vllm serve` 行（既定 = MTP k=3、rank 0）

```
vllm serve /models/GLM-5.3-Flash-NVFP4 --served-model-name glm-5.3-flash-nvfp4 --host 127.0.0.1 --port 8888
  --tensor-parallel-size 2 --distributed-executor-backend mp --nnodes 2 --node-rank 0 --master-addr 192.168.200.14 --master-port 25000
  --language-model-only --kv-cache-dtype fp8_e4m3 --block-size 2304
  --max-num-batched-tokens 4096 --max-model-len 262144 --max-num-seqs 2 --gpu-memory-utilization 0.85
  --moe-backend marlin --enforce-eager --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser deepseek_r1
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --trust-remote-code
```

rank 1 は `--headless` が付きます。NCCL / UCX / OMPI のインターフェース指定は `docker-compose.yml`。

## ディレクトリ

```
docker-compose.yml        対称な service。rank 固有の値は scripts/start-*.sh が渡す。パッチはここで bind mount
cluster.env.example       全 knob。既定値 = MTP k=3（262K / 2 slots）
patches/                  sparse_attn_indexer_kpool_sm121.py（tonyd2wild、Apache-2.0 — NOTICE 参照）、utf8_guard_lp.py（LibertAI 版利用者向けの UTF-8 ガード）
scripts/                  取得 / 照合 / 起動 / health / ゲート / 停止 / スイープ / 段階長文
bench/                    throughput_probe.py, longctx.py, garble_tokens.py, garble_ids.py（stdlib。garble_ids.py だけ tokenizers）
docs/measurements.md      実測値と条件のすべて — 11 構成のスイープ、並列数別の合計、1 台との対比、長文、U+FFFD の帰属、用途別の判定（英語）
docs/pitfalls.md          動かなかったものと理由 — image、KV dtype、MoE backend、env-file、ベンチの衛生、日本語の U+FFFD、黙って壊れるフラグ（英語）
results/ja/               日本語の元の結果表（チューニング / 本戦 / 長文）
```

## クレジット

- checkpoint: [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)（HF リポジトリにライセンス宣言なし、base は MIT。2026-09-02 から既定）/ 初出の既定 [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)（MIT、`docs/pitfalls.md` #9）
- image・top-k パッチ・DFlash2 の起動行: [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)
- drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)（CC-BY-NC-ND）
- 比較に使った 1 台の基準（2-bit GGUF、llama.cpp）: [320B の GLM-5.3-Flash を DGX Spark 1 台で動かして実用の分かれ目を測ってみた](https://dev.classmethod.jp/articles/dgx-spark-glm-5-3-flash-first-touch/)（2026-08-30）
- スイープの設計に使ったコミュニティの実測: NVIDIA Developer Forums 381429 / 381433 / 381534 / 381541 / 381543 / 381703、
  [amasu/glm53-flash-cluster](https://github.com/amasu/glm53-flash-cluster)、[kingjones30/GLM-5.3-Flash-2x-DGX-Spark](https://github.com/kingjones30/GLM-5.3-Flash-2x-DGX-Spark)、
  [Libertai/glm53-flash-vllm-gb10](https://github.com/Libertai/glm53-flash-vllm-gb10)

MIT — `LICENSE` と `NOTICE` を参照。
