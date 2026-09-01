#!/usr/bin/env bash
# node1: 2-node NCCL sanity over the QSFP/RoCE link with the MPI build of nccl-tests.
# Validates link / GID / env before vLLM; the container's own NCCL is checked at first boot.
# Needs the Mac's forwarded agent for node1 -> node2 ssh (start from such a session).
set -uo pipefail
# shellcheck disable=SC1091
source ~/glm53-cluster/cluster.env
export LD_LIBRARY_PATH="$HOME/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"   # libnccl.so.2 is not on the default path
COMMON=(-np 2 -H "${WORKER_IP}:1,${HEAD_IP}:1"
  --mca plm_rsh_agent "ssh -o StrictHostKeyChecking=accept-new"
  --mca oob_tcp_if_include "$NCCL_SOCKET_IFNAME" --mca btl_tcp_if_include "$NCCL_SOCKET_IFNAME"
  -x LD_LIBRARY_PATH
  -x "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" -x "UCX_NET_DEVICES=$UCX_NET_DEVICES" -x "OMPI_MCA_btl_tcp_if_include=$NCCL_SOCKET_IFNAME"
  -x NCCL_IB_DISABLE=0 -x NCCL_NET=IB -x "NCCL_IB_HCA=$NCCL_IB_HCA" -x "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
  -x NCCL_CROSS_NIC=1 -x NCCL_CUMEM_ENABLE=0 -x NCCL_DEBUG=INFO)
mkdir -p ~/glm53-cluster/logs
{
  echo "=== all_gather_perf $(date -Is) ==="
  mpirun "${COMMON[@]}" "$HOME/nccl-tests/build/all_gather_perf" -b 8M -e 1G -f 2 -g 1
  echo "=== all_reduce_perf $(date -Is) ==="
  mpirun "${COMMON[@]}" "$HOME/nccl-tests/build/all_reduce_perf" -b 8M -e 1G -f 2 -g 1
} 2>&1 | tee ~/glm53-cluster/logs/nccl-check.log
grep -E 'Using \[0\]|Avg bus bandwidth|GPU Direct RDMA' ~/glm53-cluster/logs/nccl-check.log | sort -u
