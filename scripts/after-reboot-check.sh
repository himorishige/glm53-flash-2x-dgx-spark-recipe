#!/usr/bin/env bash
# Mac: post power-cycle checklist for both nodes (run before relaunching anything).
# Re-applies vm.swappiness=0 (reboot resets it to 60), drops caches, and prints the facts that must hold:
# no glm53 containers, GPU idle, ports PORT/MASTER_PORT free, QSFP MTU 9000 + link up, checkpoints intact.
set -uo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$DIR/cluster.env"
for h in "$WORKER_HOST" "$HEAD_HOST"; do
  echo "=== $h"
  if ! timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" true 2>/dev/null; then echo "UNREACHABLE"; continue; fi
  ssh -o BatchMode=yes "$h" "bash -lc '
    echo \"up since: \$(uptime -s)  load: \$(cut -d\" \" -f1-3 /proc/loadavg)\"
    sudo -n sysctl -w vm.swappiness=0 >/dev/null && sudo -n sh -c \"sync; echo 3 > /proc/sys/vm/drop_caches\" && echo \"swappiness=\$(sysctl -n vm.swappiness) caches dropped\"
    free -g | sed -n 2,3p
    echo \"glm53 containers: \$(docker ps -a --format \"{{.Names}} {{.Status}}\" | grep -c glm53)\"
    echo \"running: \$(docker ps --format \"{{.Names}}\" | paste -sd, -)\"
    echo \"ports: \$(ss -ltn | grep -E \":(${PORT}|${MASTER_PORT}) \" || echo free)\"
    echo \"gpu procs: \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)\"
    ip link show ${NCCL_SOCKET_IFNAME} | head -1 | grep -oE \"mtu [0-9]+|state [A-Z]+\" | paste -sd\" \" -
    ip -4 addr show ${NCCL_SOCKET_IFNAME} | grep -oE \"inet [0-9.]+\"
    echo \"checkpoint: \$(du -sb ~/models/GLM-5.3-Flash-NVFP4 2>/dev/null | cut -f1) B  drafter: \$(du -sb ~/models/GLM-5.3-Flash-DFlash2 2>/dev/null | cut -f1) B\"
    docker image inspect --format \"v11 image: {{.Id}}\" $IMAGE 2>/dev/null || echo \"v11 image: MISSING\"
    ls ~/glm53-cluster/patches/ 2>/dev/null | paste -sd, -
    dmesg 2>/dev/null | grep -iE \"out of memory|oom-kill|nvrm|xid|hung task\" | tail -n 5
  '"
done
echo "=== QSFP ping node1 -> node2"; ssh -o BatchMode=yes "$WORKER_HOST" "ping -c 2 -W 2 -q $HEAD_IP | tail -n 2" 2>/dev/null
