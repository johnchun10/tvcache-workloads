#!/usr/bin/env bash
# Starts the tvcache-rust server on 0.0.0.0:7878.
# Eviction is off so experiments are not perturbed by background deletion;
# pass other flags through, e.g. --eviction-policy=ttl_and_watermark.
set -euo pipefail

CACHE_REPO="${CACHE_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)/tvcache-rust}"
cd "$CACHE_REPO"
if [[ $# -eq 0 ]]; then
  set -- --eviction-policy=none
fi
exec cargo run --release -- "$@"
