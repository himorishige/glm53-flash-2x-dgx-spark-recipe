#!/usr/bin/env bash
# node2: serve ~/models/GLM-5.3-Flash-NVFP4 and ~/glm53-cluster/results read-only over the QSFP link with an
# rsync daemon (no ssh, no agent forwarding — node1 has no key for node2, and the Mac's 1Password agent may be
# locked at 02:00). Bound to 192.168.200.14, hosts allow = node1's QSFP address only.
#   rsyncd-node2.sh start | stop | status
set -uo pipefail
CONF=~/glm53-cluster/rsyncd.conf; PIDF=~/glm53-cluster/logs/rsyncd.pid; PORT=8873
case "${1:-status}" in
  start)
    mkdir -p ~/glm53-cluster/logs
    cat > "$CONF" <<CONF
pid file = $PIDF
lock file = $HOME/glm53-cluster/logs/rsyncd.lock
log file = $HOME/glm53-cluster/logs/rsyncd.log
port = $PORT
address = 192.168.200.14
use chroot = no
max connections = 4
[glm53]
  path = $HOME/models/GLM-5.3-Flash-NVFP4
  read only = yes
  hosts allow = 192.168.200.13
[dflash2]
  path = $HOME/models/GLM-5.3-Flash-DFlash2
  read only = yes
  hosts allow = 192.168.200.13
[images]
  path = $HOME/glm53-cluster/images
  read only = yes
  hosts allow = 192.168.200.13
[results]
  path = $HOME/glm53-cluster/results
  read only = yes
  hosts allow = 192.168.200.13
CONF
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "rsyncd already running pid=$(cat "$PIDF")"; exit 0; fi
    rsync --daemon --config="$CONF" && sleep 1 && echo "rsyncd started pid=$(cat "$PIDF") on 192.168.200.14:$PORT";;
  stop)
    if [ -f "$PIDF" ]; then kill "$(cat "$PIDF")" 2>/dev/null && echo "rsyncd stopped"; rm -f "$PIDF"; else echo "rsyncd not running"; fi;;
  status)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "running pid=$(cat "$PIDF")"; ss -ltn | grep ":$PORT " || true; else echo "not running"; fi;;
esac
