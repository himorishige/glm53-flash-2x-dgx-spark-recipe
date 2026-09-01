#!/bin/bash
# node2: download the NVFP4 checkpoint (pinned revision) with Xet disabled — Xet hangs
# deterministically on this host (decisions 2026-08-30). Run inside tmux (session glm-dl).
# Resume after a stall: kill the tmux session and run this script again (--local-dir resumes).
export HF_HUB_DISABLE_XET=1
# shellcheck disable=SC1091
source ~/glm53-cluster/cluster.env
mkdir -p ~/models/GLM-5.3-Flash-NVFP4 ~/glm53-cluster/logs
echo "DL_START $(date -Is) rev=$MODEL_REVISION" >> ~/glm53-cluster/logs/download.log
~/.local/bin/hf download "$MODEL_REPO" --revision "$MODEL_REVISION" \
  --local-dir ~/models/GLM-5.3-Flash-NVFP4 --max-workers 4 2>&1 | tee -a ~/glm53-cluster/logs/download.log
echo "DL_DONE rc=${PIPESTATUS[0]} $(date -Is)" >> ~/glm53-cluster/logs/download.log
