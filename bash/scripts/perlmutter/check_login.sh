#!/usr/bin/env bash
# Reruns tvverify on a finished run's dump on a login node (free, no GPU):
# C1 and the cache vs no-cache comparison only, since snapshots live in the
# GPU node's /tmp and are gone once the job ends. Dumps are cumulative, so the
# last one covers every earlier one.
#
#   tmux new -d -s tvcheck 'TVCACHE_VARIANT=main bash check_login.sh $RUN [dump.json]'
#
# Writes $RUN/check-login/<dump>.json and .log, and $RUN/report_login.md.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
RUN="$1"
DUMP="${2:-$(ls "$RUN"/dumps/dump_*.json | tail -1)}"
CHECK_NODES="${CHECK_NODES:-20}"
OUT="$RUN/check-login"
mkdir -p "$OUT"
name="$(basename "$DUMP" .json)"

PIDS=()
cleanup() {
  pkill -u "$USER" -f rlhf_server/server.py 2>/dev/null || true
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT

$CONTAINER_CLI system service --time=0 "unix://$PODMAN_SOCKET" > "$OUT/podman.log" 2>&1 & PIDS+=($!)
MANAGER_REPO="$TVROOT/container-manager-public" bash "$TVROOT/tvcache-workloads/bash/scripts/start_manager.sh" \
  > "$OUT/manager.log" 2>&1 & PIDS+=($!)
wait_for_port "$FLASK_PORT" 600 "container manager" || exit 1

echo "$(date +%FT%T) checking $DUMP (--no-snapshots, $CHECK_NODES nodes per task)"
(cd "$TVROOT/tvcache-workloads" && "$VERL_PY" -u -m tvverify check-graph \
   --adapter tvcache_workloads.bash.verify:TerminalBenchAdapter \
   --dump "$DUMP" --max-replay-nodes "$CHECK_NODES" --no-snapshots --json "$OUT/$name.json") \
   > "$OUT/$name.log" 2>&1
echo "$(date +%FT%T) tvverify exited $?"
"$VERL_PY" -m tvverify report "$OUT"/*.json > "$RUN/report_login.md" 2>/dev/null || true
echo "report: $RUN/report_login.md"
echo "containers left: $($CONTAINER_CLI ps -aq | wc -l)"
