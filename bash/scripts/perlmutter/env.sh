# Sourced by the Perlmutter scripts. Override any of these before sourcing.
#
# Layout under $TVROOT (cloned by bootstrap.sh):
#   tvcache-workloads/  container-manager-public/
#   variants/combined/{tvclient,tvcache-rust}   contraction + correctness fixes
# Optional older builds, if you clone them there yourself:
#   variants/main/...   TVCache main (unmodified)
#   variants/fixes/...  the correctness-fixes branches alone
#
# Two separate choices:
#   CACHE            none | tvcache | contraction: what a run does
#   TVCACHE_VARIANT  which build of tvclient + tvcache-rust it uses
# CACHE=none starts no cache server (the with/without baseline); it still
# needs a tvclient for its base classes. CACHE=contraction needs a build with
# contraction (combined).
export TVROOT="${TVROOT:-$SCRATCH/tvcache}"
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"           # uv, docker-compose (batch jobs too)
export CACHE="${CACHE:-tvcache}"
export TVCACHE_VARIANT="${TVCACHE_VARIANT:-combined}"
if [[ "$TVCACHE_VARIANT" == nocache ]]; then CACHE=none; TVCACHE_VARIANT=main; fi   # the old spelling
if [[ "$CACHE" == none ]]; then export TVCACHE_USE_CACHE=0; else export TVCACHE_USE_CACHE=1; fi
export TASKS_ROOT="${TASKS_ROOT:-$TVROOT/container-manager-public/tasks_easy_network_skip}"
export MODEL="${MODEL:-$SCRATCH/hf/Qwen2.5-1.5B-Instruct}"
export VERL_PY="${VERL_PY:-$TVROOT/verl-env/bin/python}"                      # a python with verl, vllm and tvcache-workloads installed

# Containers: podman-hpc instead of Docker (see container-manager-public
# terminal_bench/terminal/container_cli.py). The socket and compose command
# are confirmed by probe_podman.sh; adjust here if the probe says otherwise.
export CONTAINER_CLI="${CONTAINER_CLI:-podman-hpc}"
export PODMAN_SOCKET="${PODMAN_SOCKET:-${XDG_RUNTIME_DIR:-/tmp/$USER}/tvcache-podman.sock}"
export DOCKER_HOST="${DOCKER_HOST:-unix://$PODMAN_SOCKET}"
export COMPOSE_CMD="${COMPOSE_CMD:-$HOME/bin/docker-compose}"
# compose v2 builds through a BuildKit container, which rootless podman-hpc
# refuses (crun cannot create /sys/fs/cgroup/docker). The classic builder goes
# through the podman API instead.
export DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0

# Cache server and container manager (all on the job's node)
export TVCACHE_HOST="${TVCACHE_HOST:-127.0.0.1}"
export TVCACHE_PORT="${TVCACHE_PORT:-7878}"              # TVCache main always uses 7878
export FLASK_PORT="${FLASK_PORT:-9901}"
export TVCACHE_MANAGER_URL="${TVCACHE_MANAGER_URL:-http://127.0.0.1:$FLASK_PORT}"

export TVCACHE_CONTRACTION_PORT_DEFAULT=7879             # the contraction coordinator's control port

variant_dir() { echo "$TVROOT/variants/$TVCACHE_VARIANT"; }
# tvverify builds and starts its own debug server from here (it is not next to
# tvcache-workloads on Perlmutter), so it also needs cargo.
export TVCACHE_RUST_DIR="$(variant_dir)/tvcache-rust"
command -v cargo >/dev/null || module load rust/stable 2>/dev/null || true
# $HOME on compute nodes does not support flock (os error 524), which uv and
# cargo both need; scratch does.
export UV_CACHE_DIR="$TVROOT/uv-cache" CARGO_HOME="$TVROOT/cargo-home"
# Same for every other cache the job touches (HF datasets locks its cache dir).
export XDG_CACHE_HOME="$TVROOT/cache" HF_HOME="$TVROOT/cache/huggingface"
export VLLM_CACHE_ROOT="$TVROOT/cache/vllm" TRITON_CACHE_DIR="$TVROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$TVROOT/cache/inductor" MPLCONFIGDIR="$TVROOT/cache/matplotlib" CUDA_CACHE_PATH="$TVROOT/cache/nv"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR"
# The variant's tvclient wins over any installed one, in this shell and in Ray workers.
export PYTHONPATH="$(variant_dir)/tvclient${PYTHONPATH:+:$PYTHONPATH}"

wait_for_port() {  # wait_for_port PORT SECONDS NAME
  for _ in $(seq 1 "$2"); do
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && return 0
    sleep 1
  done
  echo "$3 did not open port $1 within $2 s" >&2
  return 1
}
