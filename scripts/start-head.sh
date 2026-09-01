#!/usr/bin/env bash
# Mac: start the head (rank 0, API server) on node2. Run ~25 s after start-worker.sh.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$DIR/cluster.env"
# OVERRIDES="K=V K2=V2" is injected in front of `docker compose` on the node; shell env wins over
# --env-file during compose interpolation, so this is the per-launch tuning path (no spaces in values).
OVERRIDES="${OVERRIDES:-}"
EXTRA_COMPOSE="${EXTRA_COMPOSE:-}"   # optional extra `-f` override files
echo "[head] overrides=[$OVERRIDES] extra_compose=[$EXTRA_COMPOSE]"
echo "[head] $HEAD_HOST rank=0 VLLM_HOST_IP=$HEAD_IP"
ssh -o BatchMode=yes "$HEAD_HOST" "bash -lc 'cd ~/glm53-cluster && \
  if [ -n \"\$(docker compose -p glm53 ps -q 2>/dev/null)\" ]; then echo \"[head] glm53 containers still exist — run stop-both.sh first\"; exit 2; fi; \
  if ss -ltn | grep -q -E \":(${MASTER_PORT}|${PORT}) \"; then echo \"[head] port ${MASTER_PORT}/${PORT} still listening — run stop-both.sh first\"; exit 2; fi; \
  date -Is > logs/head-up.ts; echo \"$OVERRIDES\" > logs/head-overrides.txt'"
ssh -o BatchMode=yes "$HEAD_HOST" "bash -lc 'cd ~/glm53-cluster && mkdir -p ~/.cache/glm53-vllm && \
  env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1 \
  NODE_RANK=0 HEADLESS= VLLM_HOST_IP=$HEAD_IP $OVERRIDES \
  docker compose -p glm53 --env-file cluster.env -f docker-compose.yml $EXTRA_COMPOSE up -d && docker compose -p glm53 ps'"
# cache flusher sidecar (tonyd2wild cache_flusher.sh): keep Cached < 40 GiB for 25 min while the shards load —
# the GB10 driver does not reclaim page cache by itself (NV_ERR_NO_MEMORY after the load, forum 381429).
ssh -o BatchMode=yes "$HEAD_HOST" 'nohup bash -c "end=\$((SECONDS+1500)); while [ \$SECONDS -lt \$end ]; do c=\$(awk \"/^Cached:/{print int(\\\$2/1048576)}\" /proc/meminfo); [ \"\${c:-0}\" -gt 40 ] && { sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; }; sleep 5; done" > ~/glm53-cluster/logs/cache-flusher.log 2>&1 &'
