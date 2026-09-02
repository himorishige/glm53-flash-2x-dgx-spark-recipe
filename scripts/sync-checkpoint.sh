#!/usr/bin/env bash
# node1: pull the checkpoint from node2 over the QSFP link (rsync, resumable).
# Default transport is node2's rsync daemon (scripts/rsyncd-node2.sh start) — no ssh, so it runs
# unattended in tmux. SRC=node2:models/GLM-5.3-Flash-NVFP4/ switches back to ssh (needs the Mac's
# agent forwarded into this session).
set -euo pipefail
# shellcheck disable=SC1091
source ~/glm53-cluster/cluster.env   # HEAD_IP = the rsync daemon's address
SRC="${SRC:-rsync://${HEAD_IP}:8873/glm53/}"
mkdir -p ~/models/GLM-5.3-Flash-NVFP4 ~/glm53-cluster/logs
echo "SYNC_START $(date -Is) src=$SRC" >> ~/glm53-cluster/logs/sync-checkpoint.log
rsync -aH --partial --inplace --info=progress2 --exclude '.cache/' \
  "$SRC" ~/models/GLM-5.3-Flash-NVFP4/ 2>&1 | tee -a ~/glm53-cluster/logs/sync-checkpoint.log
echo "SYNC_DONE rc=${PIPESTATUS[0]} $(date -Is)" >> ~/glm53-cluster/logs/sync-checkpoint.log
du -sb ~/models/GLM-5.3-Flash-NVFP4 | tee -a ~/glm53-cluster/logs/sync-checkpoint.log
