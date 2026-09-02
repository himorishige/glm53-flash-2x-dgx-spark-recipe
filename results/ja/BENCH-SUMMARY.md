<!-- 著者の非公開ベンチキット（日本語 10 課題 / コード生成 5 問 / tool calls / code fix / agent-fitness）の機械集計。
     ハーネス本体はこのリポジトリに含まれない。構成 P（65K / seqs 8 / DFlash2 k=7）で 2026-09-01 に測定。 -->

## 速度

| 構成 | C=1 tok/s | TTFT s | C=8 aggregate | C=8 per-stream | スロット |
| --- | --: | --: | --: | --: | --: |
| GLM-5.3-Flash (NVFP4, vLLM TP=2) | 32.34 | 0.261 | 58.85 | 15.9 | 8 |

## 日本語

| 構成 | 機械判定 | 50字制約 | 自由記述 | コード生成 | 数値推論 | 指定語 | JSON形式 | 固有名詞 | 数値転記 | 敬語 | 長文抽出 | 出力tok | 秒 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | --: | --: |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max）〈glm53-nvfp4-quality-jp-16k〉 | 6/9 | ❌ | — | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | 16178 | 431.3 |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=high） | 5/9 | ❌ | — | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | 1389 | 59.9 |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=low） | 7/9 | ❌ | — | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | 1057 | 43.4 |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max）〈glm53-nvfp4-quality-jp〉 | 7/9 | ✅ | — | ✅ | ✅ | ❌ | ✅ | ❌* | ✅ | ✅ | ✅ | 13833 | 344.3 |

## コード生成

| 構成 | 完答 | テスト | 問題別 |
| --- | :--: | :--: | --- |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max）〈glm53-nvfp4-codegen-16k〉 | 5/5 | 49/49 | 11/11 16/16 8/8 6/6 8/8 |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=high） | 5/5 | 49/49 | 11/11 16/16 8/8 6/6 8/8 |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=low） | 5/5 | 49/49 | 11/11 16/16 8/8 6/6 8/8 |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max）〈glm53-nvfp4-codegen〉 | 5/5 | 49/49 | 11/11 16/16 8/8 6/6 8/8 |

## ツール呼び出し

| 構成 | 単発 | 複数呼び出しの形 | tool_choice=required | 擬似FS | wire format |
| --- | :--: | --- | --- | :--: | --- |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max） | 5/5 | parallel_in_one_message | 受理して 1 件 | ✅ | xml |

## コード修正

| 構成 | 直したテスト | ターン | ツール実行 | 秒 | 出力tok | 停止理由 |
| --- | :--: | --: | --: | --: | --: | --- |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max） | 5/5 | 6 | 17 | 54.1 | 1263 | final_answer |

## エージェント適性

### プローブ

| 構成 | 多ツール選択 | 長いsystem prompt | テンプレ安定 | 言語固定 | 過剰呼び出し | ターン往復 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=high） | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=low） | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max） | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |

### ハーネス別の判定

| 構成 | ハーネス | 判定 | 引っかかったプローブ |
| --- | --- | :--: | --- |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=high） | opencode | 厳しい | tool_selection_at_scale |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=high） | hermes-agent | 使える | — |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=high） | openclaw | 厳しい | tool_selection_at_scale |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=low） | opencode | 厳しい | tool_selection_at_scale |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=low） | hermes-agent | 使える | — |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=low） | openclaw | 厳しい | tool_selection_at_scale |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max） | opencode | 厳しい | tool_selection_at_scale |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max） | hermes-agent | 使える | — |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max） | openclaw | 厳しい | tool_selection_at_scale |

> これはハーネスが依存する能力のプローブであり、ハーネス本体を動かした結果ではない。実地の裏取りは `harness-gate/` を使う。

## 測定条件のメモ

- `glm53-nvfp4-agent-high.json` / bench=agent_fitness / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `glm53-nvfp4-agent-low.json` / bench=agent_fitness / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `glm53-nvfp4-agent.json` / bench=agent_fitness / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `glm53-nvfp4-codefix.json` / bench=code_fix / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `glm53-nvfp4-codegen-16k.json` / bench=code_gen / sampling=vendor / max_tokens=16384 / kv=fp8_e4m3
- `glm53-nvfp4-codegen-high.json` / bench=code_gen / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `glm53-nvfp4-codegen-low.json` / bench=code_gen / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `glm53-nvfp4-codegen.json` / bench=code_gen / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `glm53-nvfp4-quality-jp-16k.json` / bench=quality_jp / sampling=vendor / max_tokens=16384 / kv=fp8_e4m3 / set=full
- `glm53-nvfp4-quality-jp-high.json` / bench=quality_jp / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3 / set=full
- `glm53-nvfp4-quality-jp-low.json` / bench=quality_jp / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3 / set=full
- `glm53-nvfp4-quality-jp.json` / bench=quality_jp / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3 / set=full
- `glm53-nvfp4-throughput.json` / bench=throughput / sampling=legacy / max_tokens=256 / slots=8 / kv=fp8_e4m3
- `glm53-nvfp4-toolcalls.json` / bench=tool_calls / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `kabeuchi-glm53-nvfp4-high.json` / bench=kabeuchi_probe / sampling=?
- `kabeuchi-glm53-nvfp4-low.json` / bench=kabeuchi_probe / sampling=?

<!-- 未測定のベンチ: code_fix, code_gen, quality_jp, throughput, tool_calls, vision -->

<!-- RedHat 版再測（2026-09-02）。構成 P（65K / seqs 8 / DFlash2 k=7、KV profiler）、checkpoint RedHatAI 36c184c6。 -->

## 速度

| 構成 | C=1 tok/s | TTFT s | C=8 aggregate | C=8 per-stream | スロット |
| --- | --: | --: | --: | --: | --: |
| GLM-5.3-Flash (NVFP4, vLLM TP=2) | 26.05 | 0.346 | 59.42 | 13.38 | 8 |

## 日本語

| 構成 | 機械判定 | 50字制約 | 自由記述 | コード生成 | 数値推論 | 指定語 | JSON形式 | 固有名詞 | 数値転記 | 敬語 | 長文抽出 | 出力tok | 秒 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | --: | --: |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max）〈glm53-nvfp4-rh-quality-jp-16k〉 | 8/9 | ✅ | — | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | 21385 | 577.5 |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=high） | 6/9 | ❌ | — | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | 1086 | 47.3 |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=low） | 7/9 | ❌ | — | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | 1044 | 45.7 |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max）〈glm53-nvfp4-rh-quality-jp〉 | 6/9 | ❌ | — | ✅ | ✅ | ❌ | ✅ | ❌* | ✅ | ✅ | ✅ | 14700 | 413.1 |

## コード生成

| 構成 | 完答 | テスト | 問題別 |
| --- | :--: | :--: | --- |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max）〈glm53-nvfp4-rh-codegen-16k〉 | 5/5 | 49/49 | 11/11 16/16 8/8 6/6 8/8 |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=high） | 4/5 | 47/49 | 9/11 16/16 8/8 6/6 8/8 |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=low） | 5/5 | 49/49 | 11/11 16/16 8/8 6/6 8/8 |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max）〈glm53-nvfp4-rh-codegen〉 | 5/5 | 49/49 | 11/11 16/16 8/8 6/6 8/8 |

## ツール呼び出し

| 構成 | 単発 | 複数呼び出しの形 | tool_choice=required | 擬似FS | wire format |
| --- | :--: | --- | --- | :--: | --- |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max） | 5/5 | parallel_in_one_message | 受理して 1 件 | ✅ | xml |

## コード修正

| 構成 | 直したテスト | ターン | ツール実行 | 秒 | 出力tok | 停止理由 |
| --- | :--: | --: | --: | --: | --: | --- |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max） | 5/5 | 5 | 10 | 38.1 | 1067 | final_answer |

## エージェント適性

### プローブ

| 構成 | 多ツール選択 | 長いsystem prompt | テンプレ安定 | 言語固定 | 過剰呼び出し | ターン往復 |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=high） | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=low） | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max） | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |

### ハーネス別の判定

| 構成 | ハーネス | 判定 | 引っかかったプローブ |
| --- | --- | :--: | --- |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=high） | opencode | 厳しい | tool_selection_at_scale |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=high） | hermes-agent | 使える | — |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=high） | openclaw | 厳しい | tool_selection_at_scale |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=low） | opencode | 厳しい | tool_selection_at_scale |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=low） | hermes-agent | 使える | — |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=low） | openclaw | 厳しい | tool_selection_at_scale |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max） | opencode | 厳しい | tool_selection_at_scale |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max） | hermes-agent | 厳しい | language_anchoring |
| GLM-5.3-Flash (NVFP4, vLLM TP=2)（effort=max） | openclaw | 厳しい | tool_selection_at_scale |

> これはハーネスが依存する能力のプローブであり、ハーネス本体を動かした結果ではない。実地の裏取りは `harness-gate/` を使う。

## 測定条件のメモ

- `glm53-nvfp4-rh-agent-high.json` / bench=agent_fitness / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `glm53-nvfp4-rh-agent-low.json` / bench=agent_fitness / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `glm53-nvfp4-rh-agent.json` / bench=agent_fitness / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `glm53-nvfp4-rh-codefix.json` / bench=code_fix / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `glm53-nvfp4-rh-codegen-16k.json` / bench=code_gen / sampling=vendor / max_tokens=16384 / kv=fp8_e4m3
- `glm53-nvfp4-rh-codegen-high.json` / bench=code_gen / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `glm53-nvfp4-rh-codegen-low.json` / bench=code_gen / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `glm53-nvfp4-rh-codegen.json` / bench=code_gen / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `glm53-nvfp4-rh-quality-jp-16k.json` / bench=quality_jp / sampling=vendor / max_tokens=16384 / kv=fp8_e4m3 / set=full
- `glm53-nvfp4-rh-quality-jp-high.json` / bench=quality_jp / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3 / set=full
- `glm53-nvfp4-rh-quality-jp-low.json` / bench=quality_jp / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3 / set=full
- `glm53-nvfp4-rh-quality-jp.json` / bench=quality_jp / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3 / set=full
- `glm53-nvfp4-rh-throughput.json` / bench=throughput / sampling=legacy / max_tokens=256 / slots=8 / kv=fp8_e4m3
- `glm53-nvfp4-rh-toolcalls.json` / bench=tool_calls / sampling=vendor / max_tokens=8192 / kv=fp8_e4m3
- `kabeuchi-glm53-nvfp4-rh-high.json` / bench=kabeuchi_probe / sampling=?
- `kabeuchi-glm53-nvfp4-rh-low.json` / bench=kabeuchi_probe / sampling=?

<!-- 未測定のベンチ: code_fix, code_gen, quality_jp, throughput, tool_calls, vision -->
