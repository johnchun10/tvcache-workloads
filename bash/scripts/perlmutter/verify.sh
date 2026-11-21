#!/usr/bin/env bash
# tvverify on a finished run, apart from the training job: replays cached
# commands from fresh containers and compares them with what the run's cache
# held (C1), and compares the snapshots the run saved with replays of their
# paths (C2a). Needs one node and no GPU. submit_hetjob.sh queues it as a CPU
# job that starts when the training job ends (verify.sbatch); by hand, on a
# login node or inside an allocation:
#
#   bash verify.sh $RUN                            # the final dump, with the run's saved snapshots
#   bash verify.sh $RUN $RUN/dumps/epoch_002.json  # another dump: outputs only, its snapshots may be gone
#   NO_SNAPSHOTS=1 bash verify.sh $RUN             # outputs only: skips loading the snapshot archive
#
# The run's settings come from $RUN/config.env (cache build, task directory,
# CHECK_NODES; a CHECK_NODES in the environment wins). Dumps are cumulative,
# so the final one covers the earlier ones.
#
# Writes $RUN/check/<dump>.json and .log and $RUN/check/report.md, rewrites
# $RUN/report.md with the findings, and logs to $RUN/logs/verify-*.log.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[[ $# -ge 1 && -d "$1" ]] || { sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2; exit 2; }
RUN="$(cd "$1" && pwd)"
[[ -f "$RUN/config.env" ]] || { echo "no $RUN/config.env: not a run directory" >&2; exit 2; }
asked_nodes="${CHECK_NODES:-}"
# The run's own settings, exported: env.sh and tvverify read some of them.
set -a
# shellcheck disable=SC1091
source "$RUN/config.env"
set +a
[[ -z "$asked_nodes" ]] || CHECK_NODES="$asked_nodes"
CHECK_NODES="${CHECK_NODES:-8}"
# shellcheck source=env.sh
source "$HERE/env.sh"
mkdir -p "$RUN/logs" "$RUN/check"
say() { echo "$*" | tee -a "$RUN/logs/run.txt"; }

if [[ "$TVCACHE_USE_CACHE" != 1 ]]; then say "tvverify: the run used no cache, so there is nothing to check"; exit 0; fi
DUMPS=("$RUN"/dumps/dump_*.json)
[[ -e "${DUMPS[0]}" ]] || { say "tvverify: no dumps in $RUN/dumps (did training start?), so there is nothing to check"; exit 1; }
FINAL="${DUMPS[${#DUMPS[@]}-1]}"
DUMP="${2:-$FINAL}"
[[ -f "$DUMP" ]] || { echo "no such dump: $DUMP" >&2; exit 2; }
name="$(basename "$DUMP" .json)"

PIDS=()
LOADED=0
cleanup() {
  # Snapshots loaded here go again (on a login node the image store outlives
  # this script); $RUN/snapshots keeps them for another check.
  if [[ "$LOADED" == 1 ]]; then
    "$VERL_PY" - "$RUN/snapshots/snapshots.json" > /dev/null 2>&1 <<'PY' || true
import asyncio, json, os, sys
from tvcache_workloads.bash.manager_client import ManagerClient
async def go():
    client = ManagerClient(os.environ["TVCACHE_MANAGER_URL"])
    for record in json.load(open(sys.argv[1]))["snapshots"]:
        try:
            await client.delete_snapshot(record["snapshot_id"])
        except Exception:
            pass  # not loaded, or already deleted
asyncio.run(go())
PY
  fi
  pkill -u "$USER" -f rlhf_server/server.py 2>/dev/null || true
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT

# 1. podman's Docker API socket and a container manager on this node, reading
#    the run's task directory
$CONTAINER_CLI system service --time=0 "unix://$PODMAN_SOCKET" > "$RUN/logs/verify-podman.log" 2>&1 & PIDS+=($!)
MANAGER_REPO="$TVROOT/container-manager-public" bash "$HERE/../start_manager.sh" \
  > "$RUN/logs/verify-manager.log" 2>&1 & PIDS+=($!)
wait_for_port "$FLASK_PORT" 600 "container manager" || exit 1

# 2. the run's snapshots: the training job saved their images and records to
#    $RUN/snapshots, because its own image store went with its node
flags=(--no-snapshots)
if [[ "$DUMP" != "$FINAL" ]]; then
  how="outputs only (not the final dump, so its snapshots may be gone)"
elif [[ "${NO_SNAPSHOTS:-0}" == 1 ]]; then
  how="outputs only (NO_SNAPSHOTS=1)"
elif [[ ! -f "$RUN/snapshots/snapshots.json" ]]; then
  how="outputs only (the run saved no snapshots)"
elif loaded="$("$VERL_PY" - "$RUN/snapshots" 2>&1 <<'PY'
import asyncio, os, sys, time
from tvcache_workloads.bash.manager_client import ManagerClient
async def go():
    started = time.time()
    result = await ManagerClient(os.environ["TVCACHE_MANAGER_URL"]).load_snapshots(sys.argv[1])
    print(f"{result['snapshots']} snapshots loaded in {time.time() - started:.0f} s"
          + (f", {len(result['missing'])} could not be" if result["missing"] else ""))
    sys.exit(0 if result["snapshots"] else 1)
asyncio.run(go())
PY
)"; then
  flags=(); LOADED=1
  how="outputs and snapshots ($loaded)"
else
  how="outputs only (the snapshots did not load: $(tail -n 1 <<<"$loaded"))"
fi

# 3. tvverify. It starts its own cache server from the dump.
if [[ -f "$RUN/state.json" ]]; then export TVCACHE_STATE_FILE="$RUN/state.json"; fi   # contraction: what the state comparison skips per task
say "tvverify: checking $name, $CHECK_NODES nodes per task, $how"
started=$SECONDS
(cd "$HERE/../../.." && "$VERL_PY" -u -m tvverify check-graph \
   --adapter tvcache_workloads.bash.verify:TerminalBenchAdapter \
   --dump "$DUMP" --max-replay-nodes "$CHECK_NODES" ${flags[@]+"${flags[@]}"} --json "$RUN/check/$name.json") \
   > "$RUN/check/$name.log" 2>&1
status=$?
say "tvverify: exited $status after $(( (SECONDS - started) / 60 )) min"

# 4. the reports: tvverify's own, and the run's one report with it included
"$VERL_PY" -m tvverify report "$RUN"/check/*.json > "$RUN/check/report.md" 2>/dev/null || true
"$VERL_PY" -m tvcache_workloads.bash.verl.stats "$RUN" >> "$RUN/logs/stats.log" 2>&1 || true
say "report: $RUN/report.md"
echo "containers left: $($CONTAINER_CLI ps -aq | wc -l)"
# Findings are in the report; only a check that produced nothing fails the job.
[[ -s "$RUN/check/$name.json" ]] || exit 1
