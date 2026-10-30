#!/usr/bin/env bash
# Starts container-manager-public's HTTP server for the Bash workload.
# Run it from the cold-snapshots branch of our fork for snapshot_mode="cold".
#
# ENABLE_CACHING=1 is required: it makes forks carry cwd and exported
# variables (tmux-resurrect replay). server.py itself defaults it to 0;
# the repository's own run_server.sh defaults it to 1.
set -euo pipefail

MANAGER_REPO="${MANAGER_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)/container-manager-public}"
if [[ ! -f "$MANAGER_REPO/rlhf_server/server.py" ]]; then
  echo "container-manager-public not found at $MANAGER_REPO (set MANAGER_REPO)" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running (start Docker Desktop first)" >&2
  exit 1
fi

cd "$MANAGER_REPO"
# --frozen: install exactly the committed lockfile and never rewrite uv.lock.
uv sync --frozen
cd rlhf_server
export TASKS_ROOT="${TASKS_ROOT:-../tasks}"
export LOCAL_RESURRECT_PATH="${LOCAL_RESURRECT_PATH:-$PWD/tmux-res}"
export FLASK_PORT="${FLASK_PORT:-9901}"
export ENABLE_CACHING=1
exec uv run --frozen server.py
