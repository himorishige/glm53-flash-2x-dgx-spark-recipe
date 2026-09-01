#!/usr/bin/env bash
# Mac: one-factor-at-a-time sweep. Per row: stop-both → start-worker → 25 s → start-head → health → gate-2node
# (G1 + G2 effort=low + G3) → bench/throughput_probe.py on the head → results/tuning/<name>.* + a row in TUNING.md.
#   scripts/tune-sweep.sh [configs-file]      ONLY="mtp0,s8" …   REUSE_RUNNING=1 …   KEEP_LAST=1 …
set -uo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$DIR/cluster.env"
CONFIGS="${1:-$DIR/scripts/tune-configs.txt}"; OUT="$DIR/results/tuning"; mkdir -p "$OUT"
BASE="http://127.0.0.1:${PORT}"
exec > >(tee -a "$OUT/sweep-$(date +%Y%m%d-%H%M%S).log") 2>&1
[ -f "$OUT/TUNING.md" ] || printf '| config | overrides | boot s | gate | C=1 tok/s | TTFT s | C=max agg tok/s (C) | head used GB | worker used GB | note |\n| --- | --- | --: | --- | --: | --: | --: | --: | --: | --- |\n' > "$OUT/TUNING.md"
hs() { ssh -o BatchMode=yes "$HEAD_HOST" "$@"; }
used() { ssh -o BatchMode=yes "$1" "free -g | awk '/^Mem:/{print \$3}'" 2>/dev/null || echo "?"; }
settle() { for h in "$HEAD_HOST" "$WORKER_HOST"; do ssh -o BatchMode=yes "$h" "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" 2>/dev/null || true; done
  for _ in $(seq 1 36); do a=$(ssh -o BatchMode=yes "$HEAD_HOST" "free -g | awk '/^Mem:/{print \$7}'" 2>/dev/null || echo 0); b=$(ssh -o BatchMode=yes "$WORKER_HOST" "free -g | awk '/^Mem:/{print \$7}'" 2>/dev/null || echo 0); [ "${a:-0}" -ge 100 ] && [ "${b:-0}" -ge 100 ] && return 0; sleep 10; done; }
first=1
while IFS='|' read -r -u 3 name ov slots conc note; do   # fd 3: ssh inside the loop must not eat the config file
  name=$(echo "${name:-}" | xargs); case "$name" in ''|'#'*) continue;; esac
  ov=$(echo "${ov:-}" | xargs); slots=$(echo "${slots:-2}" | xargs); conc=$(echo "${conc:-1,2}" | xargs); note=$(echo "${note:-}" | xargs)
  if [ -n "${ONLY:-}" ] && ! echo ",$ONLY," | grep -q ",$name,"; then continue; fi
  echo; echo "=== CONFIG $name [$ov] slots=$slots conc=$conc $(date +%FT%T)"
  boot="reused"
  if [ "$first" = "1" ] && [ "${REUSE_RUNNING:-0}" = "1" ] && hs "curl -fsS --max-time 5 $BASE/v1/models" >/dev/null 2>&1; then :; else
    "$DIR/scripts/stop-both.sh" >/dev/null 2>&1; settle
    OVERRIDES="$ov" "$DIR/scripts/start-worker.sh" || { echo "CONFIG_FAIL $name start-worker"; continue; }
    sleep 25
    OVERRIDES="$ov" "$DIR/scripts/start-head.sh" || { echo "CONFIG_FAIL $name start-head"; "$DIR/scripts/stop-both.sh" >/dev/null 2>&1; continue; }
    t0=$(date +%s)
    if ! WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-120}" "$DIR/scripts/health.sh" > "$OUT/$name.health.log" 2>&1; then
      echo "CONFIG_FAIL $name boot"; "$DIR/scripts/logs.sh" >/dev/null 2>&1; "$DIR/scripts/stop-both.sh" >/dev/null 2>&1
      printf '| %s | `%s` | FAIL | boot | | | | | | %s |\n' "$name" "${ov:-—}" "$note" >> "$OUT/TUNING.md"; continue
    fi
    boot=$(( $(date +%s) - t0 ))
  fi
  first=0
  hs "docker inspect glm53-vllm-glm53-1 --format '{{json .Config.Cmd}}'" > "$OUT/$name.cmd.json" 2>/dev/null
  hu=$(used "$HEAD_HOST"); wu=$(used "$WORKER_HOST")
  hs "BASE=$BASE MODEL_NAME=$SERVED_MODEL_NAME G2_EFFORTS=low G2_MAX_TOKENS=2048 ~/glm53-cluster/scripts/gate-2node.sh" > "$OUT/$name.gate.log" 2>&1
  grep -q "GATE RESULT: ALL PASS" "$OUT/$name.gate.log" && gate=PASS || gate=FAIL
  hs "python3 ~/glm53-cluster/bench/throughput_probe.py --base-url $BASE --model $SERVED_MODEL_NAME --concurrency $conc --slots $slots --n 3 --out /tmp/tune-$name.json" > "$OUT/$name.throughput.log" 2>&1
  rsync -a "$HEAD_HOST:/tmp/tune-$name.json" "$OUT/tune-$name-throughput.json" 2>/dev/null
  row=$(python3 - "$OUT/tune-$name-throughput.json" <<'PY'
import json, sys
try:
    lv = json.load(open(sys.argv[1]))["levels"]; l1 = [l for l in lv if l["concurrency"] == 1]
    c1 = l1[0]["single_tok_s_median"] if l1 else "?"; tt = l1[0]["ttft_s_median"] if l1 else "?"
    lm = [l for l in lv if l["concurrency"] > 1 and l.get("ok")]; m = max(lm, key=lambda l: l["concurrency"]) if lm else None
    print(f"{c1}|{tt}|{m['aggregate_tok_s'] if m else '?'}{' (%d)' % m['concurrency'] if m else ''}")
except Exception as e: print(f"err:{type(e).__name__}|?|?")
PY
)
  IFS='|' read -r c1 tt agg <<< "$row"
  printf '| %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s |\n' "$name" "${ov:-—}" "$boot" "$gate" "$c1" "$tt" "$agg" "$hu" "$wu" "$note" >> "$OUT/TUNING.md"
  echo "CONFIG_DONE $name boot=$boot gate=$gate c1=$c1 ttft=$tt agg=$agg"
done 3< "$CONFIGS"
[ "${KEEP_LAST:-0}" = "1" ] || "$DIR/scripts/stop-both.sh" >/dev/null 2>&1
echo "SWEEP_DONE $(date +%FT%T)"
