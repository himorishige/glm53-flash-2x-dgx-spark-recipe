# vLLM tuning sweep — GLM-5.3-Flash NVFP4 / 2× DGX Spark TP=2

1 因子ずつ day-1 構成 A（`cluster.env`）から変える。throughput は強制長 256（legacy サンプリング temp 0）、C=1 は 3 回の中央値。
`code_gen low` は 5 問の実効 tok/s（受理率が効く実ワークロード、effort=low）。boot は start-head から `/v1/models` 応答まで。
gate は G1（/v1/models）+ G2（effort=low の数学 1 問、反復・`<think>` 漏れ検出）+ G3（tool call glm47）。

| config | overrides | boot s | gate | C=1 tok/s | TTFT s | C=max agg tok/s (C) | code_gen low tok/s | head used GB | worker used GB | note |
| --- | --- | --: | --- | --: | --: | --: | --: | --: | --: | --- |
| A-base | `—` | reused | PASS | 25.67 | 0.362 | 37.03 (2) | 22.27 (5/5) | 117 | 116 | day-1: marlin / 262K / seqs 2 / MTP-4 / eager / fp8 KV / batched 4096 / util 0.85 / KV 9 GiB |
| mtp0 | `MTP_NUM_TOKENS=0` | 868 | PASS | 14.55 | 0.231 | 27.19 (2) | 12.06 (5/5) | 114 | 114 | MTP off — the denominator (community 14.3-14.6 tok/s) |
| mtp5 | `MTP_NUM_TOKENS=5` | 927 | PASS | 24.53 | 0.385 | 25.52 (2) | 22.58 (5/5) | 117 | 117 | model card value (myron4 24.7-30.3) |
| mtp3 | `MTP_NUM_TOKENS=3` | 926 | PASS | 27.78 | 0.251 | 30.63 (2) | 22.40 (5/5) | 117 | 117 | Libertai value (24.0), 1-node draft 3 was the sweet spot |
| dflash7 | `SPEC_METHOD=dflash SPEC_MODEL=/models/GLM-5.3-Flash-DFlash2 MTP_NUM_TOKENS=7 KV_CACHE_MEMORY_BYTES=3221225472` | 990 | PASS | 30.84 | 0.236 | 38.15 (2) | 28.24 (5/5) | 112 | 111 | DFlash2 drafter k=7 (tonyd2wild: code 46.9, prose ~30; KV pin 3 GiB per their v11 launch) |
| s6 | `MAX_NUM_SEQS=6` | 933 | PASS | 25.59 | 0.362 | 60.04 (6) | — | 118 | 117 | 262K with 6 slots (tonyd2wild) |
| s8 | `MAX_NUM_SEQS=8` | 934 | PASS | 27.24 | 0.361 | 76.38 (8) | — | 117 | 117 | 262K with 8 slots (MiaAI) — if it boots, the bench keeps 262K without dropping context |
| cutlass | `MOE_BACKEND=flashinfer_cutlass` | FAIL | host-freeze | | | | | | | 07:21-07:25 に両ノードが ping/ssh 不通（host global OOM と推定、Libertai は cutlass の util 床 0.79-0.80 と報告。0.85 で流した判断ミス）。物理再起動待ち |
| kvbf16 | `KV_CACHE_DTYPE=auto` | 923 | PASS | 25.62 | 0.35 | 30.2 (2) | — | 117 | 117 | bf16 KV (Libertai: 1.04-1.14x faster, half the capacity) |
| ctx65k-s8 | `MAX_MODEL_LEN=65536 MAX_NUM_SEQS=8 KV_CACHE_MEMORY_BYTES=` | 932 | PASS | 27.99 | 0.352 | 76.0 (8) | — | 113 | 112 | fallback parity with the 1-node profile if s8 @262K does not boot |
| P-65k-s8-dflash7 | `MAX_MODEL_LEN=65536 MAX_NUM_SEQS=8 KV_CACHE_MEMORY_BYTES=4445787956 SPEC_METHOD=dflash SPEC_MODEL=/models/GLM-5.3-Flash-DFlash2 MTP_NUM_TOKENS=7` | 892 | PASS | 31.32 | 0.335 | 62.67 (8) | 29.48 (5/5) | 112 | 111 | bench config P: parity with glm-5-3-flash-gguf (65536 / 8) + DFlash2 k=7 |

## 読み方と採用（2026-09-01 10:10、単因子スイープ完了時点）

実測はすべて tonyd2wild v11 image + SM121 top-k mount。day-0 stock image は起動不可（別記）。C=1 は強制長 256 / greedy の 3 回中央値、code_gen は effort=low の 5 問実効 tok/s。

| 因子 | 結果 | 判断 |
| --- | --- | --- |
| MTP off → k=3 / 4 / 5 | 14.55 → 27.78 / 25.67 / 24.53（code_gen 12.1 → 22.4 / 22.3 / 22.6） | 投機は +76〜91%。k=3 が単発最速、k=4 が C2 合計最良（37.0）、k=5 は並列で落ちる（25.5）。1 台 llama.cpp の「draft 3 が最適」と同じ傾向 |
| DFlash2 drafter k=7 | **30.84** / C2 38.2 / code_gen **28.2** | 全指標で MTP を上回る（MTP-4 比 C1 +20%、code +27%）。drafter 2.2 GB、KV pin 3 GiB で head 112 GB |
| seqs 2 → 6 → 8（262K のまま） | C1 25.6 / 25.6 / 27.2、合計 37.0 (2) / 60.0 (6) / 76.4 (8) | KDA 状態キャッシュは 262K × 8 を収容。単発は劣化なし。1 台 llama.cpp の C8 64.0 を超える |
| KV fp8_e4m3 → bf16 | C1 25.67 → 25.62、C2 37.0 → 30.2 | 単発に差なし、並列は fp8 が上。fp8_e4m3 を採用 |
| 262K → 65K（seqs 8） | C1 27.2 → 28.0、C8 76.4 → 76.0、head 117 → 113 GB | 文脈長は速度に効かない。1 台との比較条件（65536 / 8）を落とさずに済む |
| flashinfer_cutlass（util 0.85） | **両ノード凍結**（物理再起動） | 無人では回さない。再試行は util ≤ 0.78 で手動で立ち会う場合のみ |
| 未測（任意枠） | graphs（+1〜3% 報告）/ batched 8192・2048 / noautotune / b12x / util 0.90 | 凍結リスクか期待値が小さいため後回し |

**採用**

- **構成 S（速度・長文）**: `dflash7` = 262K / seqs 2 / DFlash2 k=7 / KV pin 3 GiB / marlin / eager / fp8_e4m3。長文 × 並列に使う
- **構成 P（本戦・1 台比較）**: `P-65k-s8-dflash7` = 65536 / seqs 8 / DFlash2 k=7 / KV pin 4.14 GiB。**C1 31.32 / code_gen 29.5 / C8 合計 62.7**。並列合計は MTP-4（76.0）が上なので、多人数向けの表には ctx65k-s8（MTP-4）の行も併記する
- 近接（±3% 以内）は「差なし」と書く: mtp4 vs kvbf16、s6 vs A-base、ctx65k-s8 vs s8

## 品質ゲート追加（2026-09-01 11:20、本戦の結果から）

構成 P（v11 + DFlash2 k=7 + fp8_e4m3 KV）の壁打ちプローブ出力 27,757 字に **U+FFFD（置換文字）が 7 個**（許容�囲 / �実的 / �ります / X �線 / 効果�定 — 漢字の UTF-8 バイト列がトークン境界で壊れた形）。同じプローブで 1 台 2-bit llama.cpp は 28,000 字超でゼロ。quality_jp（約 1.5K 字 × 4 本）ではゼロ。帰属候補は DFlash2 drafter（バイトトークン境界での受理）/ fp8 KV / NVFP4 / v11 image。MTP-4（→ 投機なし → KV bf16）の順に切り分け、**日本語用途の採用構成は「U+FFFD ゼロ」を条件に決める**（速度だけで決めない）。

### 文字化けの帰属（2026-09-01 12:20 時点）

| 条件（すべて v11 + marlin + fp8_e4m3 KV、262K / seqs 2、temp 1.0 / top_p 0.95） | U+FFFD | 字数 | /10K 字 |
| --- | --: | --: | --: |
| DFlash2 k=7（構成 S、壁打ち A/B） | 7 | 27,757 | 2.5 |
| MTP-4 | 5 | 27,823 | 1.8 |
| 1 台 2-bit GGUF / llama.cpp（同プローブ、参考） | 0 | 28,000+ | 0 |

- 投機の種類に依らず出る → drafter 単独が原因ではない。壊れるのは漢字の UTF-8 バイト列（範 / 現 / 陥 / 線 / 測 / 拠 / ペ / 桁）で、バイトフォールバックのトークンが不正な並びで採られている形
- llama.cpp の既定サンプラーは `min_p 0.05` と `top_k 40`（`llama-server --help`）。1 台版はこの 2 つの刈り込みが暗黙で効いていた。vLLM の既定は `min_p 0` / `top_k -1`（刈り込みなし）
- vLLM は投機デコード中の `min_p` を 400 で拒否する（"The min_p and logit_bias sampling parameters are not yet supported with speculative decoding"）→ 投機と両立する `top_k 40` を先に検証（結果は下に追記）
| MTP-4 + `top_k 40`（llama.cpp 既定相当の刈り込み） | 13 | 27,475 | 4.7 |

- `top_k 40` でも消えない（むしろ増、ただし n が小さく揺れの範囲）→ 不正なバイトトークンは裾ではなく主要な候補として選ばれている。残る候補は (a) vLLM の増分 detokenizer が正しいトークン列を壊して表示している、(b) fp8 KV / NVFP4 marlin / v11 パッチカーネルの数値誤差で logits が歪んでいる。`bench/garble_tokens.py`（logprobs の `bytes` からトークン列を自前で復号）で (a) か (b) かを判定（結果は下に追記）

**判定（2026-09-01 12:45、`bench/garble_tokens.py`）: U+FFFD は vLLM の増分 detokenizer の表示バグ。** `logprobs` の `bytes` からトークン列を自前で UTF-8 復号すると文字化けはゼロ（例: content「現実的な検査�囲囲」→ 復号「現実的な検査範囲」。置換文字に加えて後続文字の重複まで出ており、境界処理の崩れ）。モデルが選んだトークン列は正しく、NVFP4 / DFlash2 / fp8 KV の品質問題ではない。1 台 llama.cpp でゼロだったのは detokenizer が別実装のため。記事では「この per-model image（vLLM 0.1.dev20051 系）で GLM-5.3-Flash の日本語出力に約 2 個 / 1 万字の割合で出る表示バグ。トークン列は正しい」と書き、上流への報告候補にする。本戦の品質表（機械判定）はトークン列に基づく採点ではなく content 文字列の判定なので、置換文字が判定語に当たった場合だけ影響し得る（今回の不合格課題の原因は 1 台版と同じ文字数自己申告・指定語の名指しで、置換文字ではない）

**訂正（2026-09-01 13:08、`bench/garble_ids.py`）: 上の「detokenizer の表示バグ」判定は誤り。** `return_token_ids` で受けたトークン ID を checkpoint 同梱の `tokenizer.json`（`tokenizers` ライブラリ）でオフライン復号しても、content と同じ位置に U+FFFD が出る（3/3 例: 引き�ぎ / 目視と�用 / 検査�囲囲）。`logprobs` の `bytes` は detokenizer 経由の差分で、生の語彙バイトではなかった。よって **不正なバイト列はモデルが選んだトークン列に含まれている**（バイトフォールバックのトークンが不完全な並びで採られている）。残る切り分け: 投機なし（MTP off）で出るか / `min_p 0.05`（投機なしなら指定可）で消えるか → 追試（結果は下に追記）

| 投機なし（MTP off） | 2 | 11,840 | 1.7 |
| 投機なし + `min_p 0.05`（low / high） | 9 / 2 | 11,787 / 14,635 | 7.6 / 1.4 |

- 投機なしでも同率で出る → 投機は無関係。`min_p 0.05` も効かない（llama.cpp の既定サンプラーで消えていた仮説は棄却）。残る候補は fp8 KV lane / NVFP4（marlin）/ v11 カーネル。KV bf16 の結果は末尾に追記

| 投機なし + KV bf16 | 14 | 27,937 | 5.0 |
| **投機なし + UTF-8 ガード（`patches/utf8_guard_lp.py`）** | **0** | 24,590 | **0.0** |

**機構（2026-09-01 15:00 確定）**: GLM-5.3 の tokenizer（zai-org 本家と md5 一致）は byte-level BPE で、日本語の新字体の多くに 1 文字トークンが無く「2 バイト断片 + 1 バイト継続」の 2 トークンで綴る（測 = `e6b8` + `ac`、範 = `e7af` + `84`、陥 = `e999` + `a5`、継 = `e7b6` + `99`、拡 = `e68b` + `a1`）。壊れる語はまさにこれ（現実 66 回中 11、効果測定 5、許容範囲、陥る、桁、継ぎ、併用、拡大、毀損、拠）で、1 トークンずつの「現場」は 95 回ゼロ。モデルが 2 バイト断片の直後に 1 バイト継続を飛ばして次の文字へ行く。KV dtype・投機・サンプラーは無関係。どの層（NVFP4 marlin / v11 カーネル）で継続判断が崩れるかは未分離。対策は UTF-8 ガード logits processor（不正な継続を遮る）で、投機なし限定（vLLM が投機中のカスタム LP を拒否）。ガード下でも想起チェック全問 OK、日本語比率 ≥ 0.945、狙いの語は正しく綴られる

## RedHat 版再測（2026-09-02、checkpoint 差し替え）

checkpoint を `RedHatAI/GLM-5.3-Flash-NVFP4` rev `36c184c6` に差し替え、KV は pin せず profiler（7.48 GiB = 1,151,844 tok）。他は同一手順。

| config | overrides | boot s | gate | C=1 tok/s | TTFT s | C=max agg tok/s (C) | code_gen low tok/s | head used GB | worker used GB | note |
| --- | --- | --: | --- | --: | --: | --: | --: | --: | --: | --- |
| rh-dflash7 | `MODEL_DIR=/models/GLM-5.3-Flash-NVFP4-RedHat KV_CACHE_MEMORY_BYTES= SPEC_METHOD=dflash SPEC_MODEL=/models/GLM-5.3-Flash-DFlash2 MTP_NUM_TOKENS=7` | 542 | PASS | 25.57 | 0.352 | 26.51 (2) | 27.74 (4/5) | 115 | 113 | recipe config S on RedHatAI 36c184c6 (LibertAI row: 30.84 / TTFT 0.236 / C=2 38.15 / codegen 28.24) |
| rh-s8 | `MODEL_DIR=/models/GLM-5.3-Flash-NVFP4-RedHat KV_CACHE_MEMORY_BYTES= SPEC_METHOD=mtp MTP_NUM_TOKENS=4 MAX_NUM_SEQS=8` | 580 | PASS | 25.26 | 0.256 | 76.19 (8) | — | 114 | 113 | recipe multi-user row on RedHatAI (LibertAI row: 27.2 / C=4 56.3 / C=8 76.4; KV was a 9 GiB pin, now profiler) |
| rh-mtp0 | `MODEL_DIR=/models/GLM-5.3-Flash-NVFP4-RedHat KV_CACHE_MEMORY_BYTES= MTP_NUM_TOKENS=0` | 549 | PASS | 14.6 | 0.228 | 27.5 (2) | 12.00 (5/5) | 113 | 112 | article 投機なし (LibertAI: 14.55 / 0.231 / C2 27.19 / cg 12.06) |
| rh-mtp3 | `MODEL_DIR=/models/GLM-5.3-Flash-NVFP4-RedHat KV_CACHE_MEMORY_BYTES= SPEC_METHOD=mtp MTP_NUM_TOKENS=3` | 574 | PASS | 26.47 | 0.249 | 29.77 (2) | 21.97 (5/5) | 113 | 112 | article MTP k=3 (LibertAI: 27.78 / 0.251 / C2 30.63 / cg 22.40) |
| rh-mtp4-s2 | `MODEL_DIR=/models/GLM-5.3-Flash-NVFP4-RedHat KV_CACHE_MEMORY_BYTES= SPEC_METHOD=mtp MTP_NUM_TOKENS=4` | 574 | PASS | 24.1 | 0.358 | 27.99 (2) | 22.63 (5/5) | 113 | 112 | article MTP k=4 day-1 (LibertAI: 25.67 / 0.362 / C2 37.03 / cg 22.27) |
| rh-mtp5 | `MODEL_DIR=/models/GLM-5.3-Flash-NVFP4-RedHat KV_CACHE_MEMORY_BYTES= SPEC_METHOD=mtp MTP_NUM_TOKENS=5` | 603 | PASS | 22.63 | 0.263 | 26.35 (2) | 21.75 (5/5) | 113 | 113 | article MTP k=5 (LibertAI: 24.53 / 0.385 / C2 25.52 / cg 22.58) |
| rh-kvbf16 | `MODEL_DIR=/models/GLM-5.3-Flash-NVFP4-RedHat KV_CACHE_MEMORY_BYTES= SPEC_METHOD=mtp MTP_NUM_TOKENS=4 KV_CACHE_DTYPE=auto` | FAIL | boot | | | | | | | article KV bf16 (LibertAI: 25.62 / C2 30.2) |
| rh-ctx65k-s8 | `MODEL_DIR=/models/GLM-5.3-Flash-NVFP4-RedHat KV_CACHE_MEMORY_BYTES= SPEC_METHOD=mtp MTP_NUM_TOKENS=4 MAX_MODEL_LEN=65536 MAX_NUM_SEQS=8` | 608 | PASS | 25.26 | 0.254 | 75.25 (8) | — | 113 | 113 | article 65K s8 (LibertAI: 28.0 / 28.9 / 74.5 / C8 76.0) |

- 単発首位は MTP k=3（26.47）。DFlash2 は 30.84 → 25.57 に失速（コード実効 27.74 は首位のまま）。投機なしの分母は一致
- KV bf16 は 262K 1 本分 3.62 GiB に対し available 3.54 GiB で boot 不可
- 65K s8 の C=2/C=4 は 36.23 / 45.77、262K s8 は 26.25 / 57.70（C=8 は 75.25 / 76.19）
