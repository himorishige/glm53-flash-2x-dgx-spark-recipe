#!/usr/bin/env python3
"""Forced-length throughput probe for an OpenAI-compatible endpoint — stdlib only.

Streams `--out-tokens` tokens (ignore_eos + min_tokens, greedy) at each concurrency level and reports
single-stream tok/s, TTFT and aggregate tok/s. Numbers are only comparable between rows measured with
the same prompt, output length and server slot count (--max-num-seqs >= the highest C).

  python3 bench/throughput_probe.py --model glm-5.3-flash-nvfp4 --concurrency 1,2,4,8 --out results/throughput.json
"""
import argparse, json, statistics, sys, threading, time, urllib.request

PROMPT = ("Explain how a transformer language model generates text, step by step, "
          "covering tokenization, attention, and decoding. Be thorough and concrete.")

def one(base, model, out_tokens, effort):
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": out_tokens,
            "min_tokens": out_tokens, "ignore_eos": True, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = None; usage = None; n = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            p = line[5:].strip()
            if p == "[DONE]": break
            try: ch = json.loads(p)
            except json.JSONDecodeError: continue
            if ch.get("usage"): usage = ch["usage"]
            for c in ch.get("choices") or []:
                d = c.get("delta") or {}
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    if t_first is None: t_first = time.perf_counter()
                    n += 1
    t_end = time.perf_counter(); t_first = t_first or t_end
    comp = (usage or {}).get("completion_tokens") or n
    return {"ttft_s": t_first - t0, "completion_tokens": comp, "decode_s": t_end - t_first,
            "tok_s": (comp - 1) / (t_end - t_first) if t_end > t_first and comp > 1 else 0.0,
            "usage_reported": usage is not None, "forced_length_held": comp == out_tokens}

def level(base, model, c, out_tokens, effort):
    res = [None] * c; errs = [None] * c
    def w(i):
        try: res[i] = one(base, model, out_tokens, effort)
        except Exception as e: errs[i] = repr(e)
    th = [threading.Thread(target=w, args=(i,)) for i in range(c)]
    t0 = time.perf_counter(); [t.start() for t in th]; [t.join() for t in th]; wall = time.perf_counter() - t0
    ok = [r for r in res if r]
    out = {"concurrency": c, "ok": len(ok), "errors": [e for e in errs if e], "wall_s": round(wall, 2)}
    if ok:
        out.update({"ttft_s_median": round(statistics.median(r["ttft_s"] for r in ok), 3),
                    "per_stream_tok_s_median": round(statistics.median(r["tok_s"] for r in ok), 2),
                    "aggregate_tok_s": round(sum(r["completion_tokens"] for r in ok) / wall, 2),
                    "usage_reported": all(r["usage_reported"] for r in ok),
                    "forced_length_held": all(r["forced_length_held"] for r in ok)})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888"); ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", default="1,2,4,8"); ap.add_argument("--out-tokens", type=int, default=256)
    ap.add_argument("--n", type=int, default=3, help="repeat C=1 this many times, report the median")
    ap.add_argument("--effort", default=None); ap.add_argument("--slots", type=int, default=None, help="server --max-num-seqs (recorded)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); base = a.base_url.rstrip("/")
    report = {"model": a.model, "base_url": base, "out_tokens": a.out_tokens, "prompt": PROMPT, "slots": a.slots, "levels": []}
    for c in [int(x) for x in a.concurrency.split(",") if x]:
        if a.slots and c > a.slots: print(f"! C={c} exceeds --slots {a.slots}: you would be measuring a queue", file=sys.stderr)
        if c == 1:
            runs = [level(base, a.model, 1, a.out_tokens, a.effort) for _ in range(a.n)]
            ok = [r for r in runs if r.get("ok")]
            e = {"concurrency": 1, "runs": runs,
                 "single_tok_s_median": round(statistics.median(r["per_stream_tok_s_median"] for r in ok), 2) if ok else None,
                 "ttft_s_median": round(statistics.median(r["ttft_s_median"] for r in ok), 3) if ok else None}
            print(f"[C=1] single tok/s (median of {a.n}): {e['single_tok_s_median']}  TTFT {e['ttft_s_median']}s")
        else:
            e = level(base, a.model, c, a.out_tokens, a.effort)
            print(f"[C={c}] aggregate tok/s: {e.get('aggregate_tok_s')}  per-stream {e.get('per_stream_tok_s_median')}  TTFT med {e.get('ttft_s_median')}s  ok={e['ok']}/{c}")
        report["levels"].append(e)
    if a.out:
        json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=2); print(f"wrote {a.out}")

if __name__ == "__main__":
    sys.exit(main())
