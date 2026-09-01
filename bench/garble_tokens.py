#!/usr/bin/env python3
"""Is the U+FFFD a detokenizer artefact or a real invalid byte sequence? — stdlib only.

Ask for long Japanese prose with logprobs=true (OpenAI format carries per-token `bytes`), find
replacement characters in `content`, and rebuild the text from the raw token bytes ourselves.
  - rebuilt text clean  → vLLM's incremental detokenizer produced the U+FFFD (display bug)
  - rebuilt text broken → the model really emitted an invalid UTF-8 byte sequence (model/numerics)
"""
import argparse, json, sys, urllib.request

PROMPTS = [
    "食品工場の外観検査カメラ導入について、現実的な範囲、測定方法、陥りやすい失敗、根拠となる数値の桁を含めて、日本語で 1500 字程度の実務メモを書いてください。",
    "製造ラインの予防保全を計画する担当者向けに、現場オペレーターの役割分担、実現可能な期限、効果測定の判定基準を、日本語で 1500 字程度で論じてください。",
    "設備稼働監視システムの要件定義書をレビューする観点を、現実性・整合性・測定可能性の 3 軸で、日本語で 1500 字程度にまとめてください。",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888"); ap.add_argument("--model", required=True)
    ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--max-tokens", type=int, default=1800)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = {"model": a.model, "samples": []}
    hits = 0
    for r in range(a.rounds):
        for p in PROMPTS:
            body = {"model": a.model, "messages": [{"role": "user", "content": p}], "max_tokens": a.max_tokens,
                    "temperature": 1.0, "top_p": 0.95, "logprobs": True, "top_logprobs": 0,
                    "chat_template_kwargs": {"reasoning_effort": "low"}}
            req = urllib.request.Request(a.base_url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as resp:
                d = json.load(resp)
            ch = d["choices"][0]; content = ch["message"].get("content") or ""
            toks = (ch.get("logprobs") or {}).get("content") or []
            raw = bytearray()
            for t in toks:
                b = t.get("bytes")
                if b is not None:
                    raw.extend(bytes(b))
                else:
                    raw.extend((t.get("token") or "").encode("utf-8", "surrogatepass"))
            rebuilt = raw.decode("utf-8", "replace")
            n_content = content.count("�"); n_rebuilt = rebuilt.count("�")
            # locate the first garble in content and show the token bytes around it
            ctx = []
            if n_content:
                i = content.index("�")
                # find token index covering position i by cumulative decode of token strings
                pos = 0
                for k, t in enumerate(toks):
                    s = t.get("token") or ""
                    if pos + len(s) > i:
                        ctx = [{"token": toks[j].get("token"), "bytes": toks[j].get("bytes")} for j in range(max(0, k-3), min(len(toks), k+4))]
                        break
                    pos += len(s)
            sample = {"round": r, "prompt": p[:30], "content_fffd": n_content, "rebuilt_fffd": n_rebuilt,
                      "tokens": len(toks), "has_bytes_field": any(t.get("bytes") is not None for t in toks),
                      "content_excerpt": content[max(0, content.find("�")-20):content.find("�")+12] if n_content else "",
                      "rebuilt_excerpt": rebuilt[max(0, rebuilt.find("�")-20):rebuilt.find("�")+12] if n_rebuilt else "",
                      "token_context": ctx}
            report["samples"].append(sample); hits += n_content
            print(f"[r{r}] content U+FFFD={n_content} rebuilt U+FFFD={n_rebuilt} tokens={len(toks)} bytes_field={sample['has_bytes_field']}", flush=True)
            if n_content:
                print("   content:", repr(sample["content_excerpt"]), "| rebuilt:", repr(sample["rebuilt_excerpt"]) if n_rebuilt else "(clean)", flush=True)
                print("   tokens:", json.dumps(ctx, ensure_ascii=False)[:400], flush=True)
            json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=2)
            if hits >= 4:
                print("GARBLE_TOKENS_DONE enough samples"); return
    print("GARBLE_TOKENS_DONE")

if __name__ == "__main__":
    sys.exit(main())
