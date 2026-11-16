#!/usr/bin/env bash
# Run once on Perlmutter (a login node is fine) after bootstrap.sh, and again
# whenever bootstrap.sh pulled new commits: builds the cache server and
# installs the Python side into $VERL_PY's environment. veRL, vLLM and torch
# must already be in that environment (see GUIDE.md).
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
  -e "$TVROOT/tvcache-workloads/correctness" pyyaml pandas pyarrow httpx docker
"$VERL_PY" -c "import verl, vllm, tvclient, tvverify, tvcache_workloads.bash.verl.session; print('python side ok: verl', verl.__version__, 'vllm', vllm.__version__)"

[[ -f "$MODEL/config.json" ]] || "$VERL_PY" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir=sys.argv[1])" "$MODEL"
echo "model at $MODEL"

# The embedding models contraction can use get their own environment, since
# EmbeddingGemma 2 needs transformers >= 5.19 and veRL needs < 5.11
# (ISSUES.md #28). A run starts one service from it (embedding.py). CPU
# builds: the service does not use a GPU. The model's processor also wants
# torchvision and Pillow, though only text is embedded.
[[ -x "$EMBED_PY" ]] || "$VERL_PY" -m venv "$(dirname "$(dirname "$EMBED_PY")")"
"$EMBED_PY" -m pip install -q --upgrade pip
"$EMBED_PY" -m pip install -q torch==2.15.0 torchvision --index-url https://download.pytorch.org/whl/cpu
"$EMBED_PY" -m pip install -q sentence-transformers==6.3.0 transformers==5.23.0 Pillow
# Download each model into the shared cache now and embed one command with
# it, so a model that cannot load fails here and not inside a GPU job.
"$EMBED_PY" "$TVROOT/tvcache-workloads/src/tvcache_workloads/bash/embedding.py" check ||
  echo "an embedder cannot load (see FAILED above); a run that picks it stops at startup"

mkdir -p "$HOME/bin"
[[ -x "$COMPOSE_CMD" ]] || echo "next: put a docker-compose binary at $COMPOSE_CMD (github.com/docker/compose releases, linux-x86_64)"
