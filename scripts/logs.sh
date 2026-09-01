#!/usr/bin/env bash
# Mac: collect both ranks' compose logs, memory, dmesg tail into logs/<timestamp>/ (run before stop-both.sh on failure).
set -uo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$DIR/cluster.env"
ts=$(date +%Y%m%d-%H%M%S); out="$DIR/logs/$ts"; mkdir -p "$out"
for pair in "$HEAD_HOST:head" "$WORKER_HOST:worker"; do
  h=${pair%%:*}; role=${pair##*:}
  ssh -o BatchMode=yes "$h" "bash -lc 'cd ~/glm53-cluster && COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 VLLM_HOST_IP=0.0.0.0 docker compose -p glm53 --env-file cluster.env -f docker-compose.yml logs --no-color --tail 2000 2>&1'" > "$out/$role.log" || true
  ssh -o BatchMode=yes "$h" "free -g; echo; docker stats --no-stream 2>/dev/null; echo; dmesg 2>/dev/null | tail -n 50" > "$out/$role.sys.txt" 2>&1 || true
done
echo "[logs] saved to $out"; ls -la "$out"
