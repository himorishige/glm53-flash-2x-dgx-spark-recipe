#!/usr/bin/env bash
# Mac: first smoke (config A from cluster.env, or OVERRIDES=/EXTRA_COMPOSE= for another config).
# stop-both → drop caches → start-worker → 25 s → start-head → health → full gate (G0-G3, G2 default+low) →
# results/smoke-<date>.md. On failure: logs.sh → stop-both.sh. Markers: SMOKE_PASS / SMOKE_FAIL.
#   nohup scripts/smoke.sh > results/smoke-run.log 2>&1 &
set -uo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$DIR/cluster.env"
BASE="http://127.0.0.1:${PORT}"; TS=$(date +%Y%m%d-%H%M); OUT="$DIR/results/smoke-$TS.md"
n1() { ssh -o BatchMode=yes "$WORKER_HOST" "$@"; }
n2() { ssh -o BatchMode=yes "$HEAD_HOST" "$@"; }
echo "SMOKE_START $TS overrides=[${OVERRIDES:-}] extra=[${EXTRA_COMPOSE:-}]"
"$DIR/scripts/stop-both.sh" >/dev/null 2>&1
for h in "$WORKER_HOST" "$HEAD_HOST"; do ssh -o BatchMode=yes "$h" "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; free -g | sed -n 2p"; done
"$DIR/scripts/start-worker.sh" || { echo "SMOKE_FAIL start-worker"; exit 1; }
sleep 25
"$DIR/scripts/start-head.sh" || { echo "SMOKE_FAIL start-head"; "$DIR/scripts/stop-both.sh"; exit 1; }
t0=$(date +%s)
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-180}" "$DIR/scripts/health.sh" 2>&1 | tee "$DIR/results/smoke-$TS.health.log"; hrc=${PIPESTATUS[0]}
boot=$(( $(date +%s) - t0 ))
if [ "$hrc" != "0" ]; then
  echo "SMOKE_FAIL boot after ${boot}s"; n2 "docker logs --tail 300 glm53-vllm-glm53-1 2>&1" > "$DIR/results/smoke-$TS.head-tail.log"; n1 "docker logs --tail 300 glm53-vllm-glm53-1 2>&1" > "$DIR/results/smoke-$TS.worker-tail.log"
  "$DIR/scripts/logs.sh"; "$DIR/scripts/stop-both.sh"; exit 1
fi
n2 "BASE=$BASE MODEL_NAME=$SERVED_MODEL_NAME ~/glm53-cluster/scripts/gate-2node.sh" 2>&1 | tee "$DIR/results/gate-smoke-$TS.log"; grc=${PIPESTATUS[0]}
cmd=$(n2 "docker inspect glm53-vllm-glm53-1 --format '{{json .Config.Cmd}}'")
evidence=$(n2 "docker logs glm53-vllm-glm53-1 2>&1 | grep -iE 'KV cache|Maximum concurrency|mamba|hybrid|state cache|autotun|cuda graph|speculative|moe backend|Using .*backend|NCCL INFO NET/IB|rocep1s0f1' | head -n 30")
m1=$(n1 'free -g | sed -n 2p'); m2=$(n2 'free -g | sed -n 2p')
{
  echo "# Smoke $TS — GLM-5.3-Flash NVFP4 / vLLM TP=2 (node2 head, node1 worker)"; echo
  echo "- overrides: \`${OVERRIDES:-none}\` extra_compose: \`${EXTRA_COMPOSE:-none}\`"
  echo "- boot (start-head → /v1/models): **${boot} s**"; echo "- gate: $([ "$grc" = "0" ] && echo PASS || echo FAIL) (see gate-smoke-$TS.log)"
  echo "- head \`free -g\`: $m2"; echo "- worker \`free -g\`: $m1"; echo
  echo "## Effective command"; echo; echo '```'; echo "$cmd"; echo '```'; echo
  echo "## Boot evidence (head log)"; echo; echo '```'; echo "$evidence"; echo '```'; echo
  echo "## Gate output"; echo; echo '```'; grep -E "effort=|content head|tool_calls|G[0-9] (PASS|FAIL)|GATE RESULT|head log" "$DIR/results/gate-smoke-$TS.log"; echo '```'
} > "$OUT"
n2 "docker compose -p glm53 ps" >> "$OUT" 2>/dev/null
echo "wrote $OUT"
if [ "$grc" = "0" ]; then echo "SMOKE_PASS boot=${boot}s"; else echo "SMOKE_FAIL gate (pair left running for inspection; run scripts/logs.sh && scripts/stop-both.sh)"; exit 1; fi
