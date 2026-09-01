#!/usr/bin/env bash
# node-side: verify the pulled/loaded vLLM image without starting a server.
# Runs the checks with the GPU attached so Triton / CUDA imports behave as at serve time.
# Output goes to stdout (tee it into results/ from the caller).
set -uo pipefail
# shellcheck disable=SC1091
source ~/glm53-cluster/cluster.env
echo "=== host: $(hostname) $(date -Is) ==="
docker image inspect --format 'Id={{.Id}}' "$IMAGE"
docker image inspect --format 'RepoDigests={{.RepoDigests}}' "$IMAGE"
docker image inspect --format 'Entrypoint={{.Config.Entrypoint}}' "$IMAGE"
docker image inspect --format 'Size={{.Size}}' "$IMAGE"
cat > /tmp/glm53-image-check.sh <<'EOF'
set -uo pipefail
echo "vllm bin: $(command -v vllm)"
python3 - <<'PY'
import importlib
def v(m):
    try:
        mod = importlib.import_module(m); return getattr(mod, "__version__", "?")
    except Exception as e:
        return f"import failed: {type(e).__name__}: {e}"
print("vllm", v("vllm"))
print("torch", v("torch"))
print("triton", v("triton"))
print("flashinfer", v("flashinfer"))
try:
    import torch
    print("nccl", torch.cuda.nccl.version(), "cuda", torch.version.cuda, "gpu", torch.cuda.is_available())
except Exception as e:
    print("torch cuda probe failed:", e)
PY
echo "--- flags present in vllm serve --help ---"
vllm serve --help 2>&1 | grep -o -E -- '--(kv-cache-memory-bytes|language-model-only|moe-backend|headless|nnodes|node-rank|master-addr|master-port|speculative-config|reasoning-parser|tool-call-parser|block-size|distributed-executor-backend)' | sort -u
echo "--- moe-backend choices ---"
vllm serve --help 2>&1 | grep -A3 -- '--moe-backend' | head -n 6
echo "--- reasoning parsers containing glm/deepseek ---"
vllm serve --help 2>&1 | grep -A6 -- '--reasoning-parser' | grep -o -E 'glm[0-9a-z_]*|deepseek_r1[0-9a-z_]*' | sort -u | paste -sd, -
echo "--- tool parsers containing glm ---"
vllm serve --help 2>&1 | grep -A6 -- '--tool-call-parser' | grep -o -E 'glm[0-9a-z_]*' | sort -u | paste -sd, -
EOF
docker run --rm --gpus all --entrypoint /bin/bash -v /tmp/glm53-image-check.sh:/tmp/check.sh:ro "$IMAGE" -lc 'bash /tmp/check.sh'
