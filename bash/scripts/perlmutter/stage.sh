#!/usr/bin/env bash
# SUPERSEDED by bootstrap.sh, which clones the repositories on Perlmutter
# itself. Kept for copying uncommitted working trees from a laptop; it stages
# the older main/fixes builds, not the combined one bootstrap.sh uses.
# Run on the Mac (after `sshproxy -u <user>`): copies everything the pilot
# needs to $SCRATCH/tvcache on Perlmutter.
#   REMOTE=<user>@perlmutter.nersc.gov bash stage.sh
# Working trees of tvcache-workloads and container-manager-public as they are
# now, plus two TVCache variants straight from git (no uncommitted edits):
#   variants/main   tvclient + tvcache-rust main
#   variants/fixes  their correctness-fixes branches
set -euo pipefail
REMOTE="${REMOTE:?set REMOTE=<user>@perlmutter.nersc.gov}"
SSH=(ssh -i "${SSH_KEY:-$HOME/.ssh/nersc}")
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"     # the directory holding all five repos
DEST="$("${SSH[@]}" "$REMOTE" 'echo $SCRATCH')/tvcache"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

for variant in main:main fixes:correctness-fixes; do
  name="${variant%%:*}"; ref="${variant#*:}"
  for repo in tvclient tvcache-rust; do
    mkdir -p "$STAGE/variants/$name/$repo"
    git -C "$ROOT/$repo" archive "$ref" | tar -x -C "$STAGE/variants/$name/$repo"
  done
done

EXCLUDES=(--exclude .venv --exclude __pycache__ --exclude .pytest_cache --exclude '*.egg-info' --exclude .DS_Store
          --exclude target --exclude tasks-old --exclude runs --exclude rollouts --exclude .git)
"${SSH[@]}" "$REMOTE" "mkdir -p '$DEST'"
rsync -az --delete -e "${SSH[*]}" "${EXCLUDES[@]}" "$STAGE/variants" "$REMOTE:$DEST/"
rsync -az --delete -e "${SSH[*]}" "${EXCLUDES[@]}" "$ROOT/tvcache-workloads" "$ROOT/container-manager-public" "$REMOTE:$DEST/"
echo "staged to $REMOTE:$DEST"
