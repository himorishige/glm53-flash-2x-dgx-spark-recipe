#!/usr/bin/env bash
# node1: copy the image from node2 over the QSFP link (no second WAN pull).
# Uses node1's ssh alias `node2` (its QSFP address), authenticated by the Mac's forwarded agent,
# so start it from an agent-forwarding session (tmux keeps it running afterwards).
set -euo pipefail
# shellcheck disable=SC1091
source ~/glm53-cluster/cluster.env
IMAGE="${1:-$IMAGE}"   # optional: copy another image (e.g. tonyd2wild v11)
mkdir -p ~/glm53-cluster/logs
ssh node2 "docker save $IMAGE" | docker load 2>&1 | tee ~/glm53-cluster/logs/load-image.log
echo "LOAD_DONE rc=${PIPESTATUS[1]} $(date -Is)" >> ~/glm53-cluster/logs/load-image.log
docker image inspect --format 'Id={{.Id}}' "$IMAGE" | tee -a ~/glm53-cluster/logs/load-image.log
