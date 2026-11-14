#!/usr/bin/env bash
# Submits a run as a heterogeneous job (hetjob.sbatch): a CPU node for the
# containers, their snapshots, the cache server, manager, fork bank and
# contraction, and a GPU node for veRL. No terminal needs to stay open. On a
# login node, from any directory:
#
#   bash submit_hetjob.sh shakedown                                  # the preset as written
#   CACHES="tvcache contraction" bash submit_hetjob.sh shakedown      # both, back to back in one job
#   TIME=02:00:00 bash submit_hetjob.sh main20-easy-med --threshold 0.95
#   SPLIT=30 TOTAL=200 bash submit_hetjob.sh main20-easy-med          # parts of SPLIT steps, chained
#
# Everything after the preset goes to run.sh (presets/, --cache, --epochs,
# ...). CACHES runs the job once per cache mode, each in its own
# $BASE-<cache> with a fresh cache server, and writes the testing plan's
# ratio table to $BASE-compare.md when it has both tvcache and contraction.
#
# Account, QOS and time limit are set per component in hetjob.sbatch (options
# given to sbatch on the command line would reach only the first component);
# TIME=hh:mm:ss submits a copy of it with that limit on both components.
#
# With SPLIT set, part k trains to k*SPLIT steps; each later part waits for the
# previous one (afterany, since a part that trained but timed out in tvverify
# still left its checkpoint) and resumes from its checkpoints/. Each part has
# its own RUN; the cache restarts empty in each part.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[[ $# -ge 1 && "$1" != -* ]] || { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2; exit 2; }
PRESET="$1"; shift
[[ -f "$HERE/presets/$PRESET.env" ]] || { echo "no preset '$PRESET' in $HERE/presets" >&2; exit 2; }
bash "$HERE/run.sh" "$PRESET" "$@" --dry-run > /dev/null   # reject bad flags now, not in the queue
CACHES="${CACHES:-}"
SPLIT="${SPLIT-}"
TOTAL="${TOTAL:-}"
[[ -z "$SPLIT" || -n "$TOTAL" ]] || { echo "SPLIT needs TOTAL (all steps)" >&2; exit 2; }
SLURM_LOGS="${SLURM_LOGS:-$SCRATCH/tvcache/runs/slurm}"
BASE="${BASE:-$SCRATCH/tvcache/runs/$(date +%Y%m%d-%H%M%S)-$PRESET-het}"
mkdir -p "$SLURM_LOGS"
SCRIPT="$HERE/hetjob.sbatch"
if [[ -n "${TIME:-}" ]]; then
  SCRIPT="$SLURM_LOGS/hetjob-${TIME//:/}.sbatch"
  sed "s/-t 05:00:00/-t $TIME/" "$HERE/hetjob.sbatch" > "$SCRIPT"
fi

submit() {  # submit <run dir> <total steps or ""> <previous part's run dir or ""> <dependency or "">
  local args=(--parsable -J "tv-het-$(basename "$1")" -o "$SLURM_LOGS/%x-%j.out" --export=ALL)
  [[ -n "$4" ]] && args+=(--dependency="afterany:$4")
  mkdir -p "$1"
  : > "$1/run_args"   # one run.sh flag per line
  (( ${#RUN_ARGS[@]} )) && printf '%s\n' "${RUN_ARGS[@]}" > "$1/run_args"
  HERE="$HERE" RUN="$1" PRESET="$PRESET" CACHES="$CACHES" TOTAL_STEPS="$2" RESUME_BASE="$3" \
    sbatch "${args[@]}" "$SCRIPT" | cut -d';' -f1
}

RUN_ARGS=("$@")
if [[ -z "$SPLIT" ]]; then
  job=$(submit "$BASE" "$TOTAL" "" "")
  echo "job $job: $PRESET ${CACHES:+(caches: $CACHES) }RUN=$BASE"
  exit 0
fi
prev_job="" prev_run="" part=1
for (( steps=SPLIT; ; steps+=SPLIT )); do
  (( steps > TOTAL )) && steps=$TOTAL
  run="$BASE-part$part"
  job=$(submit "$run" "$steps" "$prev_run" "$prev_job")
  echo "job $job: part $part, steps to $steps, RUN=$run${prev_job:+ (after $prev_job)}"
  (( steps >= TOTAL )) && break
  prev_job=$job prev_run=$run part=$(( part + 1 ))
done
