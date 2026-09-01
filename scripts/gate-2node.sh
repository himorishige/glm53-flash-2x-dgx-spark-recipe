#!/usr/bin/env bash
# GLM-5.3-Flash NVFP4 / vLLM TP=2 first-smoke gates. Run on the head (node2):
#   BASE=http://127.0.0.1:8888 MODEL_NAME=glm-5.3-flash-nvfp4 ./gate-2node.sh
#
# G0: head/worker logs show the QSFP rendezvous and the RoCE NIC, no address errors
# G1: /v1/models lists the served name
# G2: one math prompt — non-empty content, no degenerate repetition ("locklock…"),
#     reasoning separated from content, tok/s recorded (default effort and effort=low)
# G3: one tool call (get_weather) — parser glm47 must yield tool_calls
# (G4 needle 32K→200K is Phase 1, not run here.)
set -uo pipefail
BASE="${BASE:-http://127.0.0.1:8888}"
MODEL_NAME="${MODEL_NAME:-glm-5.3-flash-nvfp4}"
CONTAINER="${CONTAINER:-glm53-vllm-glm53-1}"
FAIL=0

echo "=== G0: logs ==="
docker logs --tail 3000 "$CONTAINER" 2>&1 > /tmp/gate-head.log
for pat in 'Using \[0\]rocep1s0f1' 'Unable to find address' 'NCCL WARN' 'Traceback'; do
  n=$(grep -c -E "$pat" /tmp/gate-head.log || true); echo "  head log: '$pat' x$n"
done
if grep -q -E 'Unable to find address' /tmp/gate-head.log; then echo "G0 FAIL: address resolution error"; FAIL=1; else echo "G0 PASS (rendezvous/NIC lines above are informational)"; fi

echo "=== G1: /v1/models ==="
if curl -sf --max-time 30 "$BASE/v1/models" | python3 -c "import json,sys; d=json.load(sys.stdin); ids=[m['id'] for m in d['data']]; print('models:', ids); sys.exit(0 if '$MODEL_NAME' in ids else 1)"; then
  echo "G1 PASS"
else
  echo "G1 FAIL: served name not listed"; exit 1
fi

echo "=== G2: math prompt (default effort, then effort=low) ==="
# G2_EFFORTS: space-separated, "default" = no kwarg. The sweep uses G2_EFFORTS=low (max thinks for minutes).
for effort in ${G2_EFFORTS:-default low}; do
[ "$effort" = "default" ] && effort=""
BASE="$BASE" MODEL_NAME="$MODEL_NAME" EFFORT="$effort" python3 - <<'PY' || FAIL=1
import json, os, re, time, urllib.request
base, model, effort = os.environ["BASE"], os.environ["MODEL_NAME"], os.environ["EFFORT"]
body = {"model": model,
        "messages": [{"role": "user", "content": "x^2 - 7x + 10 = 0 の解を求めてください。最後に「答え: 」で始めて答えだけ書いてください。"}],
        "max_tokens": int(os.environ.get("G2_MAX_TOKENS", "6144")), "temperature": 1.0, "top_p": 0.95}
if effort:
    body["chat_template_kwargs"] = {"reasoning_effort": effort}
req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=900) as r:
    d = json.load(r)
dt = time.time() - t0
msg = d["choices"][0]["message"]
content = msg.get("content") or ""
reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
usage = d.get("usage", {})
comp = usage.get("completion_tokens", 0)
print(f"  effort={effort or 'default'} finish={d['choices'][0].get('finish_reason')} content={len(content)} chars reasoning={len(reasoning)} chars completion_tokens={comp} wall={dt:.1f}s ~{comp/dt if dt else 0:.1f} tok/s")
print("  content head:", content[:160].replace("\n", " "))
bad = []
if not content.strip():
    bad.append("empty content")
if re.search(r"(lock){3,}", content) or re.search(r"(.{1,8})\1{4,}", content):
    bad.append("degenerate repetition")
if "<think>" in content:
    bad.append("<think> leaked into content (try REASONING_PARSER=glm45)")
if "2" not in content or "5" not in content:
    bad.append("answer 2 and 5 not found")
if bad:
    print("G2 FAIL:", "; ".join(bad)); raise SystemExit(1)
print("G2 PASS")
PY
done

echo "=== G3: tool call ==="
BASE="$BASE" MODEL_NAME="$MODEL_NAME" python3 - <<'PY' || FAIL=1
import json, os, urllib.request
base, model = os.environ["BASE"], os.environ["MODEL_NAME"]
body = {"model": model,
        "messages": [{"role": "user", "content": "東京の現在の天気を調べてください。"}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "指定した都市の現在の天気を取得する",
                   "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "都市名"}}, "required": ["city"]}}}],
        "max_tokens": 2048, "temperature": 1.0, "top_p": 0.95,
        "chat_template_kwargs": {"reasoning_effort": "low"}}
req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r:
    d = json.load(r)
msg = d["choices"][0]["message"]
calls = msg.get("tool_calls") or []
print("  tool_calls:", json.dumps(calls, ensure_ascii=False)[:300])
if not calls or calls[0]["function"]["name"] != "get_weather":
    print("G3 FAIL: no/wrong tool call (content:", str(msg.get("content"))[:200], ")"); raise SystemExit(1)
print("G3 PASS")
PY

echo
if [ "$FAIL" = "0" ]; then echo "GATE RESULT: ALL PASS"; else echo "GATE RESULT: FAIL — read the logs, do not trust the verdict alone"; exit 1; fi
