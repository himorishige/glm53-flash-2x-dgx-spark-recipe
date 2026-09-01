#!/usr/bin/env python3
"""Rigorous attribution of U+FFFD: compare vLLM's `content` with an offline decode of the returned
token IDs (`return_token_ids: true`, a vLLM extension) using the checkpoint's tokenizer.json.
Run on node2:  uv run --with tokenizers python bench/garble_ids.py --tokenizer ~/models/GLM-5.3-Flash-NVFP4/tokenizer.json ...
  content has U+FFFD, offline decode clean  → vLLM incremental detokenizer artefact (display bug)
  offline decode also has U+FFFD            → the model emitted an invalid byte sequence (model/numerics)
"""
import argparse, json, os, sys, urllib.request
from tokenizers import Tokenizer

PROMPTS = [
    "食品工場の外観検査カメラ導入について、現実的な範囲、測定方法、陥りやすい失敗、根拠となる数値の桁を含めて、日本語で 1500 字程度の実務メモを書いてください。",
    "製造ラインの予防保全を計画する担当者向けに、現場オペレーターの役割分担、実現可能な期限、効果測定の判定基準を、日本語で 1500 字程度で論じてください。",
    "設備稼働監視システムの要件定義書をレビューする観点を、現実性・整合性・測定可能性の 3 軸で、日本語で 1500 字程度にまとめてください。",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888"); ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1600); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = Tokenizer.from_file(os.path.expanduser(a.tokenizer))
    report = {"model": a.model, "tokenizer": a.tokenizer, "samples": []}; hits = 0
    for r in range(a.rounds):
        for p in PROMPTS:
            body = {"model": a.model, "messages": [{"role": "user", "content": p}], "max_tokens": a.max_tokens,
                    "temperature": 1.0, "top_p": 0.95, "return_token_ids": True,
                    "chat_template_kwargs": {"reasoning_effort": "low"}}
            req = urllib.request.Request(a.base_url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as resp:
                d = json.load(resp)
            ch = d["choices"][0]; content = ch["message"].get("content") or ""
            ids = ch.get("token_ids") or (ch.get("message") or {}).get("token_ids")
            if not ids:
                print("NO_TOKEN_IDS (return_token_ids unsupported in this build); choice keys:", list(ch.keys())); return 2
            decoded = tok.decode(ids, skip_special_tokens=True)
            # decoded includes reasoning + content; compare only the content part when the parser split them
            reasoning = ch["message"].get("reasoning") or ch["message"].get("reasoning_content") or ""
            n_c, n_d = content.count("�"), decoded.count("�")
            ex_c = content[max(0, content.find("�")-16):content.find("�")+10] if n_c else ""
            ex_d = decoded[max(0, decoded.find("�")-16):decoded.find("�")+10] if n_d else ""
            # where does the content's broken char sit in the offline decode? show the offline text at that phrase
            probe = ""
            if n_c:
                i = content.find("�"); left = content[max(0, i-8):i]
                j = decoded.find(left) if left else -1
                probe = decoded[j:j+20] if j >= 0 else "(phrase not found)"
            s = {"round": r, "prompt": p[:24], "finish": ch.get("finish_reason"), "ids": len(ids), "content_fffd": n_c,
                 "decoded_fffd": n_d, "content_excerpt": ex_c, "decoded_excerpt": ex_d, "decoded_at_content_garble": probe,
                 "reasoning_chars": len(reasoning)}
            report["samples"].append(s); hits += n_c
            print(f"[r{r}] finish={s['finish']} ids={len(ids)} content U+FFFD={n_c} offline-decode U+FFFD={n_d}", flush=True)
            if n_c: print("   content:", repr(ex_c), "| offline at same phrase:", repr(probe), flush=True)
            if n_d: print("   offline garble:", repr(ex_d), flush=True)
            json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=2)
            if hits >= 3: print("GARBLE_IDS_DONE enough samples"); return 0
    print("GARBLE_IDS_DONE"); return 0

if __name__ == "__main__":
    sys.exit(main())
