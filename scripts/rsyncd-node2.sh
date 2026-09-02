#!/usr/bin/env bash
# node2: serve ~/models/GLM-5.3-Flash-NVFP4 and ~/glm53-cluster/results read-only over the QSFP link with an
# rsync daemon (no ssh, no agent forwarding — node1 has no key for node2, and the Mac's ssh agent may be
# locked overnight). Bound to HEAD_IP, hosts allow = WORKER_IP only (both from cluster.env).
#   rsyncd-node2.sh start | stop | status
set -uo pipefail
# shellcheck disable=SC1091
source ~/glm53-cluster/cluster.env   # HEAD_IP / WORKER_IP
CONF=~/glm53-cluster/rsyncd.conf; PIDF=~/glm53-cluster/logs/rsyncd.pid; RPORT=8873
case "${1:-status}" in
  start)
    mkdir -p ~/glm53-cluster/logs
    cat > "$CONF" <<CONF
pid file = $PIDF
lock file = $HOME/glm53-cluster/logs/rsyncd.lock
log file = $HOME/glm53-cluster/logs/rsyncd.log
port = $RPORT
address = $HEAD_IP
use chroot = no
max connections = 4
[glm53]
  path = $HOME/models/GLM-5.3-Flash-NVFP4
  read only = yes
  hosts allow = $WORKER_IP
[dflash2]
  path = $HOME/models/GLM-5.3-Flash-DFlash2
  read only = yes
  hosts allow = $WORKER_IP
[images]
  path = $HOME/glm53-cluster/images
  read only = yes
  hosts allow = $WORKER_IP
[results]
  path = $HOME/glm53-cluster/results
  read only = yes
  hosts allow = $WORKER_IP
CONF
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "rsyncd already running pid=$(cat "$PIDF")"; exit 0; fi
    rsync --daemon --config="$CONF" && sleep 1 && echo "rsyncd started pid=$(cat "$PIDF") on $HEAD_IP:$RPORT";;
  stop)
    if [ -f "$PIDF" ]; then kill "$(cat "$PIDF")" 2>/dev/null && echo "rsyncd stopped"; rm -f "$PIDF"; else echo "rsyncd not running"; fi;;
  status)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "running pid=$(cat "$PIDF")"; ss -ltn | grep ":$RPORT " || true; else echo "not running"; fi;;
esac
