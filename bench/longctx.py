#!/usr/bin/env python3
"""Long-context / concurrency probe against an OpenAI-compatible vLLM endpoint — stdlib only.

For each prompt size: build a distinct haystack of the requested token count (measured with the
server's /tokenize, chat template included), bury a passphrase at 50 % depth, then

  1) forced-length run: C concurrent streaming requests, 512 output tokens forced with
     ignore_eos + min_tokens → per request TTFT, prefill tok/s (= prompt_tokens / TTFT),
     decode tok/s (= (completion_tokens - 1) / (t_end - t_first)); aggregate = Σ completion / wall
  2) needle run (C=1, effort=low, not forced): does the answer contain the passphrase?

Every request gets its own haystack (random session header + shuffled filler), so prefix caching
cannot serve a repeat. One JSON per invocation; the Mac wrapper runs one size per call so it can
check both hosts between sizes (the "host freeze on heavy prefill" report).

  python3 bench/longctx.py --base-url http://127.0.0.1:8888 --model glm-5.3-flash-nvfp4 \
      --sizes 2048,8192,32768 --concurrency 1,2 --out results/longctx-2k-32k.json
"""
import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.request

FILLER = [
    "第{n}工場の{m}号ラインでは、{d}月の稼働率が{p}パーセントで、停止要因の上位は段取り替えと材料待ちだった。",
    "品質保証部の集計によると、{d}月の不良率は{q}パーセントで、前月比で{r}ポイント改善している。",
    "設備{n}の振動センサーは{k}Hz 帯の成分が増えており、ベアリングの交換時期が近いと推定される。",
    "在庫システムの棚卸し結果では、部品番号 P-{n}{m} の実棚が帳簿より{r}個少なかった。",
    "現場のリーダー会議では、{d}月{m}日に外観検査カメラの試験運用を{n}台で始めることが決まった。",
    "エネルギー使用量は{p}万kWh で、空調とコンプレッサーが全体の{q}割を占めている。",
    "The line supervisor reported that shift {n} produced {p} units with {r} rejects during week {m}.",
    "Maintenance ticket {n}{m} was closed after replacing the {k} mm belt on conveyor {d}.",
    "保守記録によれば、{d}月に実施した予防保全は{n}件で、そのうち{m}件が計画外の追加作業に発展した。",
    "出荷計画では第{d}週に{p}パレットを{n}便のトラックで運ぶ予定で、積み付け率は{q}パーセントである。",
]


def tokenize(base, model, text):
    body = {"model": model, "messages": [{"role": "user", "content": text}],
            "add_generation_prompt": True}
    req = urllib.request.Request(base + "/tokenize", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["count"]


def make_sentence(rng):
    t = rng.choice(FILLER)
    return t.format(n=rng.randint(1, 9), m=rng.randint(10, 99), d=rng.randint(1, 12),
                    p=rng.randint(50, 99), q=rng.randint(1, 9), r=rng.randint(1, 40),
                    k=rng.randint(100, 900))


def build_haystack(base, model, target, seed):
    """Return (prompt_text, passphrase, measured_tokens). Passphrase at 50 % depth."""
    rng = random.Random(seed)
    passphrase = f"{rng.choice(['青い鯨', '銀の歯車', '朝霧', '第七倉庫', '琥珀'])}-{rng.randint(1000, 9999)}"
    header = f"セッション ID: {rng.randint(10**7, 10**8)}。以下は工場の日次報告の抜粋である。\n"
    question = "\n\n上の文書のどこかに「合言葉は「…」です。」という一文があります。その合言葉だけを答えてください。"
    needle = f"合言葉は「{passphrase}」です。"
    # first estimate tokens/sentence with a 200-sentence sample
    sample = [make_sentence(rng) for _ in range(200)]
    per = tokenize(base, model, header + "\n".join(sample) + question) / 200.0
    n = max(4, int((target - 60) / per))
    sentences = [make_sentence(rng) for _ in range(n)]
    for _ in range(3):  # converge on the target within ~1 %
        mid = len(sentences) // 2
        text = header + "\n".join(sentences[:mid] + [needle] + sentences[mid:]) + question
        count = tokenize(base, model, text)
        if abs(count - target) <= max(16, target // 100):
            break
        delta = int((target - count) / per)
        if delta > 0:
            sentences += [make_sentence(rng) for _ in range(delta)]
        else:
            sentences = sentences[:max(4, len(sentences) + delta)]
    return text, passphrase, count


def stream_forced(base, model, prompt, out_tokens, effort):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": out_tokens, "min_tokens": out_tokens, "ignore_eos": True,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = None; usage = None; deltas = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    deltas += 1
    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    ptoks = (usage or {}).get("prompt_tokens")
    ctoks = (usage or {}).get("completion_tokens") or deltas
    ttft = t_first - t0
    dec = (t_end - t_first)
    return {"ttft_s": round(ttft, 3), "prompt_tokens": ptoks, "completion_tokens": ctoks,
            "prefill_tok_s": round(ptoks / ttft, 1) if ptoks and ttft > 0 else None,
            "decode_tok_s": round((ctoks - 1) / dec, 2) if dec > 0 and ctoks > 1 else None,
            "wall_s": round(t_end - t0, 2), "server_reported_usage": usage is not None,
            "forced_length_held": ctoks == out_tokens}


def needle_check(base, model, prompt, passphrase, effort, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95}
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    wall = time.perf_counter() - t0
    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    return {"found": passphrase in content, "finish_reason": d["choices"][0].get("finish_reason"),
            "content_head": content[:120], "reasoning_chars": len(reasoning),
            "completion_tokens": (d.get("usage") or {}).get("completion_tokens"),
            "wall_s": round(wall, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", required=True)
    ap.add_argument("--sizes", default="2048,8192,32768")
    ap.add_argument("--concurrency", default="1,2")
    ap.add_argument("--out-tokens", type=int, default=512)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--needle-max-tokens", type=int, default=1024)
    ap.add_argument("--no-needle", action="store_true")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    report = {"model": args.model, "base_url": base, "out_tokens": args.out_tokens,
              "effort": args.effort, "temperature_forced": 0.0, "sizes": []}
    for size in [int(s) for s in args.sizes.split(",") if s]:
        for c in [int(x) for x in args.concurrency.split(",") if x]:
            # seed per (size, concurrency, index): otherwise request 0 of C=2 repeats the C=1 prompt and hits the
            # prefix cache (2026-09-01: 32K C=2 showed prefill 3,922 tok/s for that reason)
            prompts = [build_haystack(base, args.model, size, args.seed + size * 100 + c * 10 + i) for i in range(c)]
            results = [None] * c; errors = [None] * c

            def worker(i):
                try:
                    results[i] = stream_forced(base, args.model, prompts[i][0], args.out_tokens, args.effort)
                except Exception as e:  # noqa: BLE001
                    errors[i] = repr(e)
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(c)]
            w0 = time.perf_counter()
            for t in threads: t.start()
            for t in threads: t.join()
            wall = time.perf_counter() - w0
            ok = [r for r in results if r]
            entry = {"target_tokens": size, "measured_tokens": [p[2] for p in prompts], "concurrency": c,
                     "ok": len(ok), "errors": [e for e in errors if e], "requests": results,
                     "wall_s": round(wall, 2)}
            if ok:
                entry.update({
                    "ttft_s_median": statistics.median(r["ttft_s"] for r in ok),
                    "prefill_tok_s_median": statistics.median(r["prefill_tok_s"] or 0 for r in ok),
                    "decode_tok_s_median": statistics.median(r["decode_tok_s"] or 0 for r in ok),
                    "aggregate_tok_s": round(sum(r["completion_tokens"] for r in ok) / wall, 2),
                })
            if c == 1 and not args.no_needle and ok:
                try:
                    entry["needle"] = needle_check(base, args.model, prompts[0][0], prompts[0][1],
                                                   args.effort, args.needle_max_tokens)
                except Exception as e:  # noqa: BLE001
                    entry["needle"] = {"error": repr(e)}
            report["sizes"].append(entry)
            print(f"[{size} x C={c}] ok={len(ok)}/{c} TTFT {entry.get('ttft_s_median')}s "
                  f"prefill {entry.get('prefill_tok_s_median')} tok/s decode {entry.get('decode_tok_s_median')} tok/s "
                  f"agg {entry.get('aggregate_tok_s')} needle={entry.get('needle', {}).get('found', '-')}",
                  flush=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
