#!/usr/bin/env bash
# node2: one-line status for remote polling. Fields (space-separated):
#   pull_rc(last PULL_DONE rc or "none") image_present(0/1) chain_started dl_rc(last DL_DONE rc or "none") model_bytes tmux_sessions
set -uo pipefail
p=$(grep PULL_DONE ~/glm53-cluster/logs/pull-image.log 2>/dev/null | tail -n 1 | sed -n 's/.*rc=\([0-9]*\).*/\1/p')
img=$(docker image inspect vllm/vllm-openai:glm53-flash-arm64-cu130 >/dev/null 2>&1 && echo 1 || echo 0)
c=$(grep -c CHAIN_STARTED_DL ~/glm53-cluster/logs/download.log 2>/dev/null || true)
d=$(grep DL_DONE ~/glm53-cluster/logs/download.log 2>/dev/null | tail -n 1 | sed -n 's/.*rc=\([0-9]*\).*/\1/p')
b=$(du -sb ~/models/GLM-5.3-Flash-NVFP4 2>/dev/null | cut -f1 || true)
s=$(tmux ls 2>/dev/null | cut -d: -f1 | paste -sd, - || true)
echo "${p:-none} ${img} ${c:-0} ${d:-none} ${b:-0} ${s:-none}"
