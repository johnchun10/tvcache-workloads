#!/usr/bin/env bash
# Submits a run as a two-node job (hetjob.sbatch): a CPU node for the
# containers, snapshots, cache server, manager, fork bank and contraction, and
# a GPU node for veRL. Run it on a login node; no terminal needs to stay open.
# For each run that uses a cache it also queues tvverify, as a one-node CPU job
# that starts when the training job ends (verify.sbatch; not with --no-verify).
#
#   bash submit_hetjob.sh shakedown
#   bash submit_hetjob.sh main20-easy-med --cache all --time 08:00:00
#   bash submit_hetjob.sh main20-easy-med --threshold 0.97 --embedder embeddinggemma2
#   bash submit_hetjob.sh main20-easy-med --split 30 --total 200
#
# Job flags (this script):
#   --time hh:mm:ss     time limit for both nodes (default 05:00:00, in hetjob.sbatch).
#                       tvverify's job has 02:00:00; VERIFY_TIME=hh:mm:ss changes it
#   --cache all         none, tvcache and contraction back to back in one job,
#                       each in its own <run>-<mode> with an empty cache, plus
#                       <run>-compare.md (one mode: --cache none|tvcache|contraction)
#   --split N --total M train M steps in parts of N, one job per part, each
#                       resuming from the previous part's checkpoint
# Every other flag is a run.sh flag (--threshold, --epochs, --top-k, ...).
#
# The account and QOS are the #SBATCH -A/-q lines in hetjob.sbatch (one pair
# per node; sbatch options on the command line would reach only the first).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[[ $# -ge 1 && "$1" != -* ]] || { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2; exit 2; }
PRESET="$1"; shift
[[ -f "$HERE/presets/$PRESET.env" ]] || { echo "no preset '$PRESET' in $HERE/presets" >&2; exit 2; }
TIME="" CACHES="" SPLIT="" TOTAL="" RUN_ARGS=()
need() { [[ $# -ge 2 ]] || { echo "$1 needs a value" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --time)   need "$@"; TIME="$2"; shift 2 ;;
    --cache)  need "$@"; if [[ "$2" == all ]]; then CACHES="none tvcache contraction"; else RUN_ARGS+=("$1" "$2"); fi; shift 2 ;;
    --split)  need "$@"; SPLIT="$2"; shift 2 ;;
    --total)  need "$@"; TOTAL="$2"; shift 2 ;;
    *)        RUN_ARGS+=("$1"); shift ;;
  esac
done
[[ "$SPLIT$TOTAL" =~ ^[0-9]*$ ]] || { echo "--split and --total take a number of steps" >&2; exit 2; }
[[ -z "$SPLIT" || -n "$TOTAL" ]] || { echo "--split needs --total (all steps)" >&2; exit 2; }
[[ -z "$TIME" || "$TIME" =~ ^[0-9]+:[0-9]{2}:[0-9]{2}$ ]] || { echo "--time must look like hh:mm:ss (got '$TIME')" >&2; exit 2; }
settings="$(bash "$HERE/run.sh" "$PRESET" ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} --dry-run)"   # reject bad flags now, not in the queue
# What tvverify's job needs to know: is it on, and which cache does a single-mode run use
VERIFY_ON=0; [[ "$settings" == *"tvverify on"* ]] && VERIFY_ON=1
ONE_CACHE="$(sed -n 's/^  cache: \([a-z]*\).*/\1/p' <<<"$settings")"
VERIFY_TIME="${VERIFY_TIME:-}"
[[ -z "$VERIFY_TIME" || "$VERIFY_TIME" =~ ^[0-9]+:[0-9]{2}:[0-9]{2}$ ]] || { echo "VERIFY_TIME must look like hh:mm:ss (got '$VERIFY_TIME')" >&2; exit 2; }
tools="${TVROOT:-$SCRATCH/tvcache}/bin"
[[ -x "$tools/uv" && -x "${COMPOSE_CMD:-$tools/docker-compose}" ]] ||
  { echo "uv or docker-compose missing from $tools: run setup.sh (it installs them there)" >&2; exit 1; }
SLURM_LOGS="${SLURM_LOGS:-$SCRATCH/tvcache/runs/slurm}"
BASE="${BASE:-$SCRATCH/tvcache/runs/$(date +%Y%m%d-%H%M%S)-$PRESET-het}"
mkdir -p "$SLURM_LOGS"
SCRIPT="$HERE/hetjob.sbatch"
if [[ -n "$TIME" ]]; then
  SCRIPT="$SLURM_LOGS/hetjob-${TIME//:/}.sbatch"
  sed "s/-t 05:00:00/-t $TIME/" "$HERE/hetjob.sbatch" > "$SCRIPT"
fi
VERIFY_SCRIPT="$HERE/verify.sbatch"
if [[ -n "$VERIFY_TIME" ]]; then
  VERIFY_SCRIPT="$SLURM_LOGS/verify-${VERIFY_TIME//:/}.sbatch"
  sed "s/-t 02:00:00/-t $VERIFY_TIME/" "$HERE/verify.sbatch" > "$VERIFY_SCRIPT"
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

submit_verify() {  # submit_verify <run dir base> <training job>: one tvverify job per run that has a cache
  [[ "$VERIFY_ON" == 1 ]] || return 0
  local modes=(${CACHES:-$ONE_CACHE}) mode run job
  for mode in "${modes[@]}"; do
    [[ "$mode" != none ]] || continue
    run="$1"; (( ${#modes[@]} > 1 )) && run="$1-$mode"   # hetjob.sbatch's naming
    job=$(HERE="$HERE" RUN="$run" sbatch --parsable -J "tv-verify-$(basename "$run")" -o "$SLURM_LOGS/%x-%j.out" \
            --export=ALL --dependency="afterany:$2" "$VERIFY_SCRIPT" | cut -d';' -f1)
    echo "job $job: tvverify for RUN=$run (after $2)"
  done
}

if [[ -z "$SPLIT" ]]; then
  job=$(submit "$BASE" "$TOTAL" "" "")
  echo "job $job: $PRESET ${CACHES:+(caches: $CACHES) }RUN=$BASE"
  submit_verify "$BASE" "$job"
  exit 0
fi
prev_job="" prev_run="" part=1
for (( steps=SPLIT; ; steps+=SPLIT )); do
  (( steps > TOTAL )) && steps=$TOTAL
  run="$BASE-part$part"
  job=$(submit "$run" "$steps" "$prev_run" "$prev_job")
  echo "job $job: part $part, steps to $steps, RUN=$run${prev_job:+ (after $prev_job)}"
  submit_verify "$run" "$job"
  (( steps >= TOTAL )) && break
  prev_job=$job prev_run=$run part=$(( part + 1 ))
done
