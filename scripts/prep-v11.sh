#!/usr/bin/env bash
# Mac: image + drafter prerequisites, safe to run concurrently with resume-after-dl.sh (WAN vs QSFP).
#   node2: pull tonyd2wild v11 image (~20 GB WAN) + download the DFlash2 drafter (2.2 GB WAN)
#   node1: pull both over the rsync daemon (image tar -> docker load; drafter dir)
# Idempotent. Markers: V11_NODE2_OK / DRAFTER_NODE2_OK / V11_NODE1_OK / DRAFTER_NODE1_OK / PREP_V11_DONE / PREP_V11_FAIL
#   nohup scripts/prep-v11.sh > results/prep-v11.log 2>&1 &
set -uo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$DIR/cluster.env"
V11="${V11:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2}"
V11_DIGEST="${IMAGE_DIGEST_EXPECTED:-sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6}"   # tonyd2wild README, 2026-08-31
TAG=$(echo "$V11" | tr '/:' '__')
n1() { ssh -o BatchMode=yes "$WORKER_HOST" "$@"; }
n2() { ssh -o BatchMode=yes "$HEAD_HOST" "$@"; }
echo "PREP_V11_START $(date +%FT%T)"
# node2 WAN jobs (tmux, both at once — the drafter is small)
n2 "docker image inspect $V11 >/dev/null 2>&1 || tmux has-session -t glm-v11 2>/dev/null || tmux new -d -s glm-v11 '~/glm53-cluster/scripts/pull-image.sh $V11'"
n2 "grep -q 'DFLASH_DONE rc=0' ~/glm53-cluster/logs/download-dflash2.log 2>/dev/null || tmux has-session -t glm-dfl 2>/dev/null || tmux new -d -s glm-dfl '~/glm53-cluster/scripts/download-dflash2.sh'"
for _ in $(seq 1 180); do   # up to 90 min
  img=$(n2 "docker image inspect $V11 >/dev/null 2>&1 && echo 1 || echo 0")
  dfl=$(n2 "grep -q 'DFLASH_DONE rc=0' ~/glm53-cluster/logs/download-dflash2.log 2>/dev/null && echo 1 || echo 0")
  [ "$img" = "1" ] && [ "$dfl" = "1" ] && break
  sleep 30
done
[ "$img" = "1" ] && echo "V11_NODE2_OK $(n2 "docker image inspect --format '{{.Id}} {{.RepoDigests}}' $V11")" || { echo "PREP_V11_FAIL v11 pull (node2)"; n2 "tail -n 5 ~/glm53-cluster/logs/pull-image-$TAG.log"; exit 1; }
n2 "docker image inspect --format '{{.RepoDigests}}' $V11 | grep -q $V11_DIGEST && echo '[prep] digest matches expected' || echo '[prep] WARNING digest differs from expected value'"
[ "$dfl" = "1" ] && echo "DRAFTER_NODE2_OK $(n2 'du -sb ~/models/GLM-5.3-Flash-DFlash2 | cut -f1') B" || { echo "PREP_V11_FAIL drafter download"; exit 1; }
# node1 over the QSFP daemon
n2 '~/glm53-cluster/scripts/rsyncd-node2.sh start' >/dev/null
n2 "~/glm53-cluster/scripts/save-image.sh $V11" | tail -n 1
n1 "~/glm53-cluster/scripts/load-image-rsyncd.sh $V11" | tail -n 2 && echo "V11_NODE1_OK" || { echo "PREP_V11_FAIL load (node1)"; exit 1; }
n1 "mkdir -p ~/models/GLM-5.3-Flash-DFlash2 && rsync -a --partial --inplace rsync://${HEAD_IP}:8873/dflash2/ ~/models/GLM-5.3-Flash-DFlash2/ && du -sb ~/models/GLM-5.3-Flash-DFlash2" && echo "DRAFTER_NODE1_OK"
n2 'rm -f ~/glm53-cluster/images/*.tar'
n1 "docker image inspect --format 'Id={{.Id}}' $V11"; n2 "docker image inspect --format 'Id={{.Id}}' $V11"
echo "PREP_V11_DONE $(date +%FT%T)"
