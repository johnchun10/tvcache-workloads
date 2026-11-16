#!/usr/bin/env bash
# Run on a login node after bootstrap.sh, and again whenever bootstrap.sh
# pulled new commits. Installs what is missing and skips what is there:
#   $TVROOT/bin/{uv,docker-compose}  the manager runs on uv; podman-hpc has no compose
#   $TVROOT/verl-env                 conda env with veRL 0.9.1 (about 15 min the first time)
#   $TVROOT/embed-env                the embedding models (they need a newer transformers)
# and builds the cache server and downloads the models. Everything goes under
# $TVROOT; your own uv, docker-compose, conda or Python installs are not used
# or changed (env.sh puts $TVROOT/bin first on PATH).
set -euo pipefail
source "$(dirname "$0")/env.sh"

UV_VERSION=0.13.8 COMPOSE_VERSION=v2.29.7
mkdir -p "$TVROOT/bin"
if [[ ! -x "$TVROOT/bin/uv" ]]; then
  # The release tarball, not the install script: the script edits shell rc
  # files and leaves a non-executable `env` next to uv (ISSUES.md #27).
  curl -sSfL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" |
    tar -xz -C "$TVROOT/bin" --strip-components=1
fi
if [[ ! -x "$COMPOSE_CMD" ]]; then
  [[ "$COMPOSE_CMD" == "$TVROOT/bin/docker-compose" ]] || { echo "COMPOSE_CMD=$COMPOSE_CMD does not exist" >&2; exit 1; }
  curl -sSfL -o "$COMPOSE_CMD.part" \
    "https://github.com/docker/compose/releases/download/$COMPOSE_VERSION/docker-compose-linux-x86_64"
  chmod +x "$COMPOSE_CMD.part" && mv "$COMPOSE_CMD.part" "$COMPOSE_CMD"
fi
echo "tools ok: $("$TVROOT/bin/uv" --version), docker-compose $("$COMPOSE_CMD" version --short)"

if [[ ! -x "$VERL_PY" ]]; then
  [[ "$VERL_PY" == "$TVROOT/verl-env/bin/python" ]] || { echo "VERL_PY=$VERL_PY does not exist" >&2; exit 1; }
  command -v conda >/dev/null || module load conda 2>/dev/null || true
  command -v conda >/dev/null || { echo "conda missing: module load conda, then run this again" >&2; exit 1; }
  # conda-forge only, so a personal ~/.condarc or the defaults channel's terms
  # prompt cannot change or stop the install; package cache on scratch, not $HOME.
  CONDA_PKGS_DIRS="$TVROOT/cache/conda-pkgs" \
    conda create -q -y -p "$TVROOT/verl-env" --override-channels -c conda-forge python=3.12
fi
if ! "$VERL_PY" -c "import verl, vllm" 2>/dev/null; then
  "$VERL_PY" -m pip install -q --upgrade pip
  "$VERL_PY" -m pip install "verl[vllm]==0.9.1"
fi
"$VERL_PY" -c "import verl, sys; v = verl.__version__; v == '0.9.1' or print(f'warning: VERL_PY has verl {v}; the integration was written against 0.9.1', file=sys.stderr)"
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

echo "setup done; next: bash probe_podman.sh"
