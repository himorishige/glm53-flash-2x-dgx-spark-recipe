#!/usr/bin/env bash
# Mac: tear down BOTH ranks (head first, then worker). Always run this before any relaunch;
# a half-alive TP=2 pair leaves one GPU at 100% forever (forum #358755). `down` is safe here:
# project glm53 owns nothing shared (network_mode: host).
set -uo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$DIR/cluster.env"
for pair in "$HEAD_HOST:head" "$WORKER_HOST:worker"; do
  h=${pair%%:*}; role=${pair##*:}
  echo "[$role] down on $h"
  ssh -o BatchMode=yes "$h" "bash -lc 'cd ~/glm53-cluster && COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 VLLM_HOST_IP=0.0.0.0 docker compose -p glm53 --env-file cluster.env -f docker-compose.yml down -t 60; \
    docker ps --format \"{{.Names}}\" | grep -c glm53 | sed \"s/^/[$role] glm53 containers left: /\"; \
    ss -ltn | grep -E \":(${PORT}|${MASTER_PORT}) \" || echo \"[$role] ports ${PORT}/${MASTER_PORT} closed\"; free -g | sed -n 2p'" || true
done
