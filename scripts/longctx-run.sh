#!/usr/bin/env bash
# Mac: staged long-context loop — one size per head-side invocation, both hosts checked between sizes.
#   SIZES="2048,8192,32768,131072,200000" CONC="1,2" scripts/longctx-run.sh <label>
set -uo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$DIR/cluster.env"
LABEL="${1:-s}"; SIZES="${SIZES:-2048,8192,32768,131072}"; CONC="${CONC:-1,2}"; BASE="http://127.0.0.1:${PORT}"
mkdir -p "$DIR/results"
hosts_ok() { for h in "$HEAD_HOST" "$WORKER_HOST"; do
    out=$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "free -g | awk '/^Mem:/{print \$7}'" 2>/dev/null) || { echo "[longctx] $h unresponsive"; return 1; }
    echo "[longctx] $h avail=${out}G"; done; }
for size in ${SIZES//,/ }; do
  hosts_ok || { echo "LONGCTX_ABORT before $size"; exit 1; }
  echo "=== longctx $size x C=$CONC $(date +%FT%T)"
  ssh -o BatchMode=yes "$HEAD_HOST" "python3 ~/glm53-cluster/bench/longctx.py --base-url $BASE --model $SERVED_MODEL_NAME --sizes $size --concurrency $CONC --out /tmp/longctx-$LABEL-$size.json" 2>&1 | tee -a "$DIR/results/longctx-$LABEL.log"
  rsync -a "$HEAD_HOST:/tmp/longctx-$LABEL-$size.json" "$DIR/results/" 2>/dev/null
done
hosts_ok; echo "LONGCTX_DONE $LABEL $(date +%FT%T)"
