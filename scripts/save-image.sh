#!/usr/bin/env bash
# node2: docker save <image> to ~/glm53-cluster/images/<tag>.tar so node1 can pull it over the rsync daemon (no ssh/agent).
#   save-image.sh ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
set -euo pipefail
IMAGE="${1:?usage: save-image.sh <image>}"
TAG=$(echo "$IMAGE" | tr '/:' '__'); mkdir -p ~/glm53-cluster/images ~/glm53-cluster/logs
OUT=~/glm53-cluster/images/$TAG.tar
if [ -s "$OUT" ] && grep -q "SAVE_DONE rc=0 $TAG" ~/glm53-cluster/logs/save-image.log 2>/dev/null; then echo "already saved $OUT"; exit 0; fi
docker save "$IMAGE" -o "$OUT"; rc=$?
echo "SAVE_DONE rc=$rc $TAG $(date -Is) $(stat -c %s "$OUT")" >> ~/glm53-cluster/logs/save-image.log; ls -la "$OUT"
