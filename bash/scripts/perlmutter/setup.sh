#!/usr/bin/env bash
# Run once on Perlmutter (a login node is fine) after stage.sh: builds both
# cache-server variants and installs the Python side into $VERL_PY's environment.
# veRL, vLLM and torch must already be in that environment (see README.md).
set -euo pipefail
source "$(dirname "$0")/env.sh"
command -v cargo >/dev/null || module load rust/stable 2>/dev/null || true
command -v cargo >/dev/null || { echo "cargo missing: curl https://sh.rustup.rs -sSf | sh -s -- -y --no-modify-path" >&2; exit 1; }
for variant in main fixes; do
  (cd "$TVROOT/variants/$variant/tvcache-rust" && cargo build --release --quiet) && echo "built tvcache-rust ($variant)"
done
"$VERL_PY" -m pip install -q -e "$TVROOT/variants/fixes/tvclient" -e "$TVROOT/tvcache-workloads" \
  -e "$TVROOT/tvcache-workloads/correctness" pyyaml pandas pyarrow httpx docker
"$VERL_PY" -c "import verl, vllm, tvclient, tvverify, tvcache_workloads.bash.verl.session; print('python side ok: verl', verl.__version__, 'vllm', vllm.__version__)"
[[ -f "$MODEL/config.json" ]] || "$VERL_PY" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir=sys.argv[1])" "$MODEL"
echo "model at $MODEL"
mkdir -p "$HOME/bin"
[[ -x "$COMPOSE_CMD" ]] || echo "next: put a docker-compose binary at $COMPOSE_CMD (github.com/docker/compose releases, linux-x86_64)"
