#!/usr/bin/env bash
# node2: print download growth every 5 minutes; flag STALLED when the byte count has not
# grown for 10 minutes (the hang symptom is "process alive, 0 MB/s at the same offset").
set -uo pipefail
DIR=~/models/GLM-5.3-Flash-NVFP4
last=0; stalled=0
while true; do
  now=$(du -sb "$DIR" 2>/dev/null | cut -f1); now=${now:-0}
  if grep -q DL_DONE ~/glm53-cluster/logs/download.log 2>/dev/null; then echo "$(date -Is) DONE bytes=$now"; break; fi
  if [ "$now" -gt "$last" ]; then stalled=0; else stalled=$((stalled+1)); fi
  echo "$(date -Is) bytes=$now delta=$((now-last)) $( [ "$stalled" -ge 2 ] && echo STALLED )"
  last=$now
  sleep 300
done
