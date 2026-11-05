#!/usr/bin/env bash
# Compute-node check: which filesystems support the file locks uv and cargo take.
#   srun -q shared_interactive -C cpu -c 4 -t 00:05:00 -A <project> bash probe_locks.sh
set -u
source "$(dirname "$0")/env.sh"
for dir in "$HOME" "$SCRATCH" /tmp; do
  f="$dir/.tvprobe_lock_$$"
  if python3 -c "import fcntl,sys; fcntl.flock(open(sys.argv[1],'w'), fcntl.LOCK_EX)" "$f" 2>/dev/null; then
    echo "PASS  flock on $dir"; else echo "FAIL  flock on $dir"; fi
  rm -f "$f"
done
cd "$TVROOT/container-manager-public"
UV_CACHE_DIR=/tmp/$USER-uv uv sync --frozen -q && echo "PASS  uv sync with UV_CACHE_DIR=/tmp" || echo "FAIL  uv sync with UV_CACHE_DIR=/tmp"
cd "$TVCACHE_RUST_DIR"
cargo build --quiet --offline 2>&1 | tail -2; echo "cargo (default CARGO_HOME) exit ${PIPESTATUS[0]}"
CARGO_HOME=$SCRATCH/.cargo-home cargo build --quiet --offline 2>&1 | tail -2; echo "cargo (CARGO_HOME on scratch) exit ${PIPESTATUS[0]}"
