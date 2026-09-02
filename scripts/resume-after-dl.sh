#!/usr/bin/env bash
# Mac: after the node2 download finished — sync to node1 over the QSFP, sha256-verify both nodes, host prep.
# Idempotent; every step is skipped when its
# evidence already exists. Run under nohup; markers: SYNC_OK / VERIFY_OK / CHECK_OK / SMOKE_READY / RESUME_FAIL.
#   nohup scripts/resume-after-dl.sh > results/resume-after-dl.log 2>&1 &
set -uo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$DIR/cluster.env"
EXPECTED=197881157135   # HF blob total of rev 36c184c6 (21 files, 184.3 GiB); du -sb may add a few KB of directory entries
n1() { ssh -o BatchMode=yes "$WORKER_HOST" "$@"; }
n2() { ssh -o BatchMode=yes "$HEAD_HOST" "$@"; }
fail() { echo "RESUME_FAIL $*"; exit 1; }
echo "RESUME_START $(date +%FT%T)"

# 0. download evidence
read -r _ _ _ dl bytes _ <<< "$(n2 '~/glm53-cluster/scripts/status-node2.sh')"
[ "$dl" = "0" ] || fail "download not finished (dl_rc=$dl)"
[ "$bytes" = "$EXPECTED" ] || echo "[resume] WARNING node2 bytes=$bytes expected=$EXPECTED (verify will decide)"

# 1. rsync node2 -> node1 over the QSFP rsync daemon
if n1 "grep -q 'SYNC_DONE rc=0' ~/glm53-cluster/logs/sync-checkpoint.log 2>/dev/null && [ \"\$(du -sb ~/models/GLM-5.3-Flash-NVFP4 | cut -f1)\" = \"$EXPECTED\" ]"; then
  echo "SYNC_OK (already)"
else
  n2 '~/glm53-cluster/scripts/rsyncd-node2.sh start' || fail "rsyncd start"
  n1 'tmux has-session -t glm-sync 2>/dev/null || tmux new -d -s glm-sync "~/glm53-cluster/scripts/sync-checkpoint.sh"'
  t0=$(date +%s)
  for _ in $(seq 1 240); do   # up to 2 h
    if n1 "grep -q 'SYNC_DONE' ~/glm53-cluster/logs/sync-checkpoint.log 2>/dev/null"; then break; fi
    sleep 30
  done
  rc=$(n1 "grep SYNC_DONE ~/glm53-cluster/logs/sync-checkpoint.log | tail -n 1 | sed -n 's/.*rc=\([0-9]*\).*/\1/p'")
  b1=$(n1 'du -sb ~/models/GLM-5.3-Flash-NVFP4 | cut -f1')
  echo "[resume] sync rc=${rc:-none} bytes=$b1 in $(( $(date +%s) - t0 )) s"
  [ "${rc:-1}" = "0" ] && [ "$b1" = "$EXPECTED" ] && echo "SYNC_OK" || fail "sync (rc=${rc:-none} bytes=$b1)"
fi

# 2. sha256 on node2 (≈4 min) -> results/SHA256SUMS + verify.json
if n2 '[ -s ~/glm53-cluster/results/SHA256SUMS ] && python3 -c "import json,os;d=json.load(open(os.path.expanduser(\"~/glm53-cluster/results/verify.json\")));exit(0 if d[\"sha256_checked\"] and not d[\"problems\"] else 1)" 2>/dev/null'; then
  echo "VERIFY_OK (already)"
else
  n2 'tmux has-session -t glm-verify 2>/dev/null || tmux new -d -s glm-verify "python3 ~/glm53-cluster/scripts/verify-checkpoint.py ~/models/GLM-5.3-Flash-NVFP4 --repo RedHatAI/GLM-5.3-Flash-NVFP4 --revision 36c184c6cda000a481711306df5adde42f63321a --sha256 --out ~/glm53-cluster/results > ~/glm53-cluster/logs/verify.log 2>&1"'
  t0=$(date +%s)
  for _ in $(seq 1 120); do   # up to 1 h
    if n2 '[ -f ~/glm53-cluster/results/verify.json ]'; then break; fi
    sleep 30
  done
  n2 'cat ~/glm53-cluster/results/verify.json' | tee "$DIR/results/verify-node2.json"
  n2 'python3 -c "import json, os; d=json.load(open(os.path.expanduser(\"~/glm53-cluster/results/verify.json\"))); raise SystemExit(0 if d[\"sha256_checked\"] and not d[\"problems\"] else 1)"' \
    && echo "VERIFY_OK in $(( $(date +%s) - t0 )) s" || fail "verify (see results/verify-node2.json)"
fi

# 3. node1 checks against node2's SHA256SUMS (via the Mac: two small copies)
rsync -a "$HEAD_HOST:glm53-cluster/results/SHA256SUMS" "$DIR/results/SHA256SUMS" && rsync -a "$DIR/results/SHA256SUMS" "$WORKER_HOST:glm53-cluster/results/SHA256SUMS" || fail "SHA256SUMS copy"
n1 'tmux has-session -t glm-check 2>/dev/null || tmux new -d -s glm-check "cd ~/models/GLM-5.3-Flash-NVFP4 && sha256sum -c ~/glm53-cluster/results/SHA256SUMS > ~/glm53-cluster/results/sha256-check.log 2>&1; echo CHECK_DONE rc=\$? >> ~/glm53-cluster/results/sha256-check.log"'
t0=$(date +%s)
for _ in $(seq 1 120); do
  if n1 "grep -q CHECK_DONE ~/glm53-cluster/results/sha256-check.log 2>/dev/null"; then break; fi
  sleep 30
done
n1 'grep -c ": OK$" ~/glm53-cluster/results/sha256-check.log; grep -vE ": OK$" ~/glm53-cluster/results/sha256-check.log | head' | tee "$DIR/results/sha256-check-node1.txt"
n1 "grep -q 'CHECK_DONE rc=0' ~/glm53-cluster/results/sha256-check.log" && echo "CHECK_OK in $(( $(date +%s) - t0 )) s" || fail "node1 sha256sum -c"
n2 '~/glm53-cluster/scripts/rsyncd-node2.sh stop' >/dev/null 2>&1

# 4. pre-smoke host prep + state: swappiness 60 -> 0 (marlin repack UVM livelock, forum 381429), drop page cache
for h in "$WORKER_HOST" "$HEAD_HOST"; do
  ssh -o BatchMode=yes "$h" "sudo -n sysctl -w vm.swappiness=0 >/dev/null && sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' && echo \"\$(hostname) swappiness=\$(sysctl -n vm.swappiness) caches dropped\"" || echo "[resume] WARNING host prep failed on $h"
done
for h in "$WORKER_HOST" "$HEAD_HOST"; do ssh -o BatchMode=yes "$h" "hostname; free -g | sed -n 2p; ss -ltn | grep -E ':(${PORT}|${MASTER_PORT}) ' || echo ports-free; ollama ps 2>/dev/null | tail -n +2 | grep -q . && echo OLLAMA_MODEL_LOADED || echo ollama-idle"; done
echo "SMOKE_READY $(date +%FT%T)"
