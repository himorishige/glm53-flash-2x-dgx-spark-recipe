#!/usr/bin/env bash
# node2: pull the vLLM image once over the WAN (tmux session glm-img). Log: ~/glm53-cluster/logs/pull-image.log
set -euo pipefail
# shellcheck disable=SC1091
source ~/glm53-cluster/cluster.env
IMAGE="${1:-$IMAGE}"   # optional: pull another image (e.g. tonyd2wild v11) — log goes to pull-image-<tag>.log
LOG=~/glm53-cluster/logs/pull-image$( [ -n "${1:-}" ] && echo "-$(echo "$1" | tr '/:' '__')" ).log
mkdir -p ~/glm53-cluster/logs
docker pull "$IMAGE" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
# Only a successful pull writes PULL_DONE (downstream chains grep for it); failures are marked separately.
if [ "$rc" -eq 0 ]; then echo "PULL_DONE rc=0 $(date -Is)" >> "$LOG"; else echo "PULL_FAILED rc=$rc $(date -Is)" >> "$LOG"; exit "$rc"; fi
docker image inspect --format 'Id={{.Id}} RepoDigests={{.RepoDigests}} ENTRYPOINT={{.Config.Entrypoint}} CMD={{.Config.Cmd}}' "$IMAGE" | tee -a "$LOG"
