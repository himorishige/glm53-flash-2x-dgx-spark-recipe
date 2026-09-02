#!/usr/bin/env bash
# Mac: wait for the head's OpenAI API (up to 90 min: cold boot is 14–17 min, FlashInfer autotune
# on the first run can be longer), then send one minimal chat. Tails head-side errors while waiting.
set -uo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$DIR/cluster.env"
ATTEMPTS="${WAIT_ATTEMPTS:-180}"; EVERY="${WAIT_SECONDS:-30}"
start=$(date +%s)
for i in $(seq 1 "$ATTEMPTS"); do
  if ssh -o BatchMode=yes "$HEAD_HOST" "curl -fsS --max-time 5 http://127.0.0.1:${PORT}/v1/models" >/tmp/glm53-models.json 2>/dev/null; then
    echo "[health] API up after $(( $(date +%s) - start )) s"; cat /tmp/glm53-models.json; echo
    ssh -o BatchMode=yes "$HEAD_HOST" "curl -fsS --max-time 180 http://127.0.0.1:${PORT}/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":16,\"temperature\":0}'"
    echo; exit 0
  fi
  # fast-fail: a container that already exited will never answer — do not wait 90 min on a corpse
  for pair in "$HEAD_HOST:head" "$WORKER_HOST:worker"; do
    hh=${pair%%:*}; role=${pair##*:}
    st=$(ssh -o BatchMode=yes "$hh" "docker inspect -f '{{.State.Status}}' glm53-vllm-glm53-1 2>/dev/null" || echo unknown)
    if [ "$st" = "exited" ] || [ "$st" = "dead" ]; then
      echo "[health] $role container is $st — aborting wait"
      ssh -o BatchMode=yes "$hh" "docker logs --tail 40 glm53-vllm-glm53-1 2>&1 | tail -n 12"
      exit 2
    fi
  done
  if (( i % 4 == 0 )); then
    echo "[health] $(( $(date +%s) - start )) s: waiting… (head errors, if any:)"
    ssh -o BatchMode=yes "$HEAD_HOST" "docker logs --tail 200 glm53-vllm-glm53-1 2>&1 | grep -E 'Traceback|NCCL WARN|Unable to find address|Error|error' | tail -n 3" || true
    for h in "$HEAD_HOST" "$WORKER_HOST"; do ssh -o BatchMode=yes "$h" "hostname; free -g | sed -n 2p"; done
  fi
  sleep "$EVERY"
done
echo "[health] timed out" >&2; exit 1
