#!/usr/bin/env bash
# Run on a Perlmutter login node. Clones (or updates) the four repositories a
# run needs into $TVROOT (default $SCRATCH/tvcache), each at `main` of our forks:
#
#   tvcache-workloads/                 this repository
#   container-manager-public/          the container manager (podman-hpc support is in main)
#   variants/combined/tvclient/        graph contraction, the correctness fixes, the fork bank
#   variants/combined/tvcache-rust/    the same, server side
#
#   bash bootstrap.sh          # then: bash setup.sh
#
# Starting from nothing:
#   git clone https://github.com/johnchun10/tvcache-workloads.git $SCRATCH/tvcache/tvcache-workloads
#   bash $SCRATCH/tvcache/tvcache-workloads/bash/scripts/perlmutter/bootstrap.sh
#
# Running it again updates every clone to the newest commit of its branch,
# and moves a clone that is still on an older branch over to `main`.
# Override a source with OWNER, WORKLOADS_REF, MANAGER_REF or TVCACHE_REF.
set -euo pipefail
TVROOT="${TVROOT:-$SCRATCH/tvcache}"
OWNER="${OWNER:-johnchun10}"
WORKLOADS_REF="${WORKLOADS_REF:-main}"
MANAGER_REF="${MANAGER_REF:-main}"
TVCACHE_REF="${TVCACHE_REF:-main}"

sync_repo() {  # sync_repo <repository> <branch> <directory>
  local url="https://github.com/$OWNER/$1.git" branch="$2" dir="$3"
  if [[ -d "$dir/.git" ]]; then
    git -C "$dir" remote set-url origin "$url"
    git -C "$dir" fetch -q origin "$branch"
    git -C "$dir" checkout -q -B "$branch" "origin/$branch"
  elif [[ -e "$dir" ]]; then
    echo "$dir exists but is not a git clone (an old stage.sh copy?); move it away and run this again" >&2
    exit 1
  else
    mkdir -p "$(dirname "$dir")"
    git clone -q -b "$branch" "$url" "$dir"
  fi
  echo "$(basename "$dir"): $branch @ $(git -C "$dir" rev-parse --short HEAD)"
}

sync_repo tvcache-workloads "$WORKLOADS_REF" "$TVROOT/tvcache-workloads"
sync_repo container-manager-public "$MANAGER_REF" "$TVROOT/container-manager-public"
sync_repo tvclient "$TVCACHE_REF" "$TVROOT/variants/combined/tvclient"
sync_repo tvcache-rust "$TVCACHE_REF" "$TVROOT/variants/combined/tvcache-rust"

[[ -f "$TVROOT/container-manager-public/terminal_bench/terminal/container_cli.py" ]] || {
  echo "container-manager-public@$MANAGER_REF has no podman-hpc support (terminal_bench/terminal/container_cli.py)" >&2; exit 1; }
[[ -f "$TVROOT/variants/combined/tvclient/tvclient/contraction.py" && -f "$TVROOT/variants/combined/tvclient/tvclient/observer.py" ]] || {
  echo "tvclient@$TVCACHE_REF is missing contraction or the correctness fixes" >&2; exit 1; }
echo "everything is under $TVROOT; next: bash $TVROOT/tvcache-workloads/bash/scripts/perlmutter/setup.sh"
