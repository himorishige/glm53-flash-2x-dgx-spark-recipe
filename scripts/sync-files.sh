#!/usr/bin/env bash
# Mirror compose / env / scripts from this directory to ~/glm53-cluster/ on both nodes.
# Run from the Mac. logs/ and results/ are never synced (they flow the other way).
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$DIR/cluster.env"
for h in "$WORKER_HOST" "$HEAD_HOST"; do
  echo "[sync] -> $h:~/glm53-cluster/"
  ssh -o BatchMode=yes "$h" 'mkdir -p ~/glm53-cluster/scripts ~/glm53-cluster/bench ~/glm53-cluster/logs ~/glm53-cluster/results ~/glm53-cluster/patches'
  rsync -a --delete --exclude 'logs/' --exclude 'results/' --exclude '.gitignore' \
    "$DIR/docker-compose.yml" "$DIR/cluster.env" "$DIR/scripts" "$DIR/bench" "$DIR/patches" "$h:glm53-cluster/"
done
echo "[sync] done"
