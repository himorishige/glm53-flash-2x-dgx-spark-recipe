#!/usr/bin/env bash
# node1: pull an image tar from node2's rsync daemon ([images] module) and docker load it.
#   load-image-rsyncd.sh ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
set -euo pipefail
IMAGE="${1:?usage: load-image-rsyncd.sh <image>}"
TAG=$(echo "$IMAGE" | tr '/:' '__'); mkdir -p ~/glm53-cluster/images ~/glm53-cluster/logs
if docker image inspect "$IMAGE" >/dev/null 2>&1; then echo "already present: $(docker image inspect --format '{{.Id}}' "$IMAGE")"; exit 0; fi
rsync -a --partial --inplace --info=progress2 "rsync://192.168.200.14:8873/images/$TAG.tar" ~/glm53-cluster/images/ 2>&1 | tail -n 2
docker load -i ~/glm53-cluster/images/$TAG.tar 2>&1 | tee -a ~/glm53-cluster/logs/load-image.log
echo "LOAD_DONE rc=${PIPESTATUS[0]} $TAG $(date -Is)" >> ~/glm53-cluster/logs/load-image.log
docker image inspect --format 'Id={{.Id}}' "$IMAGE" | tee -a ~/glm53-cluster/logs/load-image.log
rm -f ~/glm53-cluster/images/$TAG.tar
