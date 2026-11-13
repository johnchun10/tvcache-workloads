#!/usr/bin/env bash
# Run once on Perlmutter (a login node is fine) after bootstrap.sh, and again
# whenever bootstrap.sh pulled new commits: builds the cache server and
# installs the Python side into $VERL_PY's environment. veRL, vLLM and torch
# must already be in that environment (see README.md).
set -euo pipefail
source "$(dirname "$0")/env.sh"
command -v cargo >/dev/null || module load rust/stable 2>/dev/null || true
command -v cargo >/dev/null || { echo "cargo missing: curl https://sh.rustup.rs -sSf | sh -s -- -y --no-modify-path" >&2; exit 1; }
[[ -d "$(variant_dir)/tvclient" ]] || { echo "no $(variant_dir)/tvclient; run bootstrap.sh" >&2; exit 1; }

# Every build under variants/ (bootstrap.sh clones `combined`; `main` and
# `fixes` are only there if you cloned them yourself).
for server in "$TVROOT"/variants/*/tvcache-rust; do
  (cd "$server" && cargo build --release --quiet) && echo "built tvcache-rust ($(basename "$(dirname "$server")"))"
done

# env.sh puts the chosen variant's tvclient first on PYTHONPATH, so this
# install only has to bring tvclient's own dependencies.
"$VERL_PY" -m pip install -q -e "$(variant_dir)/tvclient" -e "$TVROOT/tvcache-workloads" \
  -e "$TVROOT/tvcache-workloads/correctness" pyyaml pandas pyarrow httpx docker "sentence-transformers>=2.3"
"$VERL_PY" -c "import verl, vllm, tvclient, tvverify, tvcache_workloads.bash.verl.session; print('python side ok: verl', verl.__version__, 'vllm', vllm.__version__)"

[[ -f "$MODEL/config.json" ]] || "$VERL_PY" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir=sys.argv[1])" "$MODEL"
echo "model at $MODEL"

# The embedding models contraction can use: download them into the shared
# cache now and embed one command with each, so a model that cannot load
# fails here and not inside a GPU job.
"$VERL_PY" - <<'PY'
from tvcache_workloads.bash.embedding import EMBEDDERS, CommandEmbedder

for alias, (model, _, _) in EMBEDDERS.items():
    try:
        vector = CommandEmbedder(alias).encode("ls -la /app")
        print(f"embedder {alias} ok: {model}, {len(vector)} dimensions")
    except Exception as error:
        print(f"embedder {alias} FAILED ({model}): {type(error).__name__}: {error}")
PY

mkdir -p "$HOME/bin"
[[ -x "$COMPOSE_CMD" ]] || echo "next: put a docker-compose binary at $COMPOSE_CMD (github.com/docker/compose releases, linux-x86_64)"
