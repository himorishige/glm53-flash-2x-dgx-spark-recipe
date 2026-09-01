# 長文 × 並列（構成 S: 262K / seqs 2 / DFlash2 k=7 / fp8_e4m3 KV、tonyd2wild v11 + top-k patch）

`bench/longctx.py`。各リクエストは固有の干し草（セッション ID + シャッフル filler、`/tokenize` で目標長）に合言葉を 50% 深度で埋め、強制長 512（greedy、effort=low）で TTFT / prefill / decode を取る。needle は C=1 の非強制リクエスト（effort=low、max 1024）で合言葉が返るか。
decode は投機（DFlash2）の受理率に依存し、干し草を復唱するような出力では跳ね上がる（8K / 32K の C=1）。長文の主指標は TTFT と prefill。

## C=1（プロンプト長に対する TTFT / prefill / decode / needle）

| prompt tok | 実測 tok | TTFT s | prefill tok/s | decode tok/s | needle |
| --: | --: | --: | --: | --: | :--: |
| 2,048 | 2,022 | 1.5 | 1,391 | 27.5 | ✅ |
| 8,192 | 8,140 | 5.4 | 1,510 | 71.1 | ✅ |
| 32,768 | 32,751 | 22.2 | 1,473 | 50.9 | ✅ |
| 131,072 | 130,975 | 90.2 | 1,452 | 21.9 | ✅ |
| 200,000 | 200,342 | 139.2 | 1,439 | 21.5 | ✅ |

## C=2（2 本同時、固有プロンプト × 2）

| prompt tok | TTFT s (median) | prefill tok/s (median) | decode tok/s (median) | aggregate tok/s | wall s |
| --: | --: | --: | --: | --: | --: |
| 2,048 | 2.8 | 743 | 32.8 | 51.3 | 20 |
| 8,192 | 8.8 | 975 | 21.0 | 24.1 | 43 |
| 32,768 | 34.5 | 1,056 | 19.6 | 15.2 | 67 |
| 131,072 | 140.6 | 1,069 | 48.4 | 5.1 | 201 |
| 200,000 | 218.9 | 1,052 | 36.2 | 3.3 | 310 |

## 並列（構成別、Phase 1 の throughput.py: 強制長 256 / greedy、C=1 は 3 回中央値）

| 構成 | C=1 | C=2 | C=4 | C=6 | C=8 | 出典 |
| --- | --: | --: | --: | --: | --: | --- |
| 262K / seqs 2 / MTP-4（A） | 25.7 | 37.0 | — | — | — | `tune-A-base-throughput.json` |
| 262K / seqs 6 / MTP-4 | 25.6 | 26.4 | 57.6 | 60.0 | — | `tune-s6-throughput.json` |
| 262K / seqs 8 / MTP-4 | 27.2 | 28.9 | 56.3 | — | 76.4 | `tune-s8-throughput.json` |
| 65K / seqs 8 / MTP-4 | 28.0 | 28.9 | 74.5 | — | 76.0 | `tune-ctx65k-s8-throughput.json` |
| 262K / seqs 2 / DFlash2 k=7（S） | 30.8 | 38.1 | — | — | — | `tune-dflash7-throughput.json` |
| 65K / seqs 8 / DFlash2 k=7（P） | 32.3 | 42.9 | 70.8 | — | 58.9 | `glm53-nvfp4-throughput.json` |
