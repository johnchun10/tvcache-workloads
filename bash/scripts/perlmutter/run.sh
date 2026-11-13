#!/usr/bin/env bash
# Starts one run from a preset, with any setting overridden by a flag.
#
#   bash run.sh shakedown                                   # the preset as written
#   bash run.sh main20-easy-med --cache none                # no cache at all
#   bash run.sh main20-easy-med --cache tvcache             # TVCache, no contraction
#   bash run.sh main20-easy-med --threshold 0.95 --embedder jina-code
#   bash run.sh shakedown --prebuild                        # login node: only build the task images
#   bash run.sh shakedown --dry-run                         # print the settings and stop
#
# Presets are the files in presets/ (shakedown, main20-easy-med). Inside a GPU
# allocation, e.g.
#   sbatch -q shared -C gpu -G 2 -c 64 -t 08:00:00 -A <project> --wrap "bash run.sh main20-easy-med"
#
# Flags:
#   --cache none|tvcache|contraction   no cache / TVCache / TVCache with graph contraction
#   --threshold X                      contraction: similarity threshold, e.g. 0.8 0.9 0.95
#   --embedder minilm|jina-code        contraction: embedding model (or a sentence-transformers name)
#   --budget N                         contraction: live sandboxes all contraction jobs share
#   --epochs N  --rollouts N  --batch N  --max-tokens N
#   --verify | --no-verify             run tvverify on the cache afterwards, or not
#   --tasks "a b c"                    replace the preset's task list
#   --mode train|sample                sample = rollouts only, no training
#   --name LABEL                       appended to the run directory's name
#   --prebuild                         build the task images through the manager and exit
#   --dry-run                          print the resolved settings and exit
# Everything else train_pilot.sh reads (MODEL, N_GPUS, SNAPSHOT_THRESHOLD, RUN,
# TVCACHE_VARIANT, ...) still comes from the environment.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
[[ $# -ge 1 && "$1" != -* ]] || { usage >&2; exit 2; }
PRESET="$1"; shift
PRESET_FILE="$HERE/presets/$PRESET.env"
[[ -f "$PRESET_FILE" ]] || { echo "no preset '$PRESET'; have: $(cd "$HERE/presets" && ls *.env | sed 's/\.env$//' | tr '\n' ' ')" >&2; exit 2; }
# shellcheck disable=SC1090
source "$PRESET_FILE"

MODE="${MODE:-train}"; NAME=""; DRY_RUN=0
need() { [[ $# -ge 2 ]] || { echo "$1 needs a value" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache)      need "$@"; CACHE="$2"; shift 2 ;;
    --threshold)  need "$@"; THRESHOLD="$2"; shift 2 ;;
    --embedder)   need "$@"; EMBEDDER="$2"; shift 2 ;;
    --budget)     need "$@"; BUDGET="$2"; shift 2 ;;
    --epochs)     need "$@"; EPOCHS="$2"; shift 2 ;;
    --rollouts)   need "$@"; ROLLOUTS="$2"; shift 2 ;;
    --batch)      need "$@"; BATCH="$2"; shift 2 ;;
    --max-tokens) need "$@"; MAX_TOKENS="$2"; shift 2 ;;
    --tasks)      need "$@"; TASKS="$2"; shift 2 ;;
    --mode)       need "$@"; MODE="$2"; shift 2 ;;
    --name)       need "$@"; NAME="$2"; shift 2 ;;
    --verify)     VERIFY=1; shift ;;
    --no-verify)  VERIFY=0; shift ;;
    --prebuild)   MODE=prebuild; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$CACHE" in none|tvcache|contraction) ;; *) echo "--cache must be none, tvcache or contraction (got '$CACHE')" >&2; exit 2 ;; esac
case "$MODE" in train|sample|prebuild) ;; *) echo "--mode must be train or sample (got '$MODE')" >&2; exit 2 ;; esac
TASKS="$(echo $TASKS)"   # one line, single spaces
[[ -n "$TASKS" ]] || { echo "the preset has no TASKS" >&2; exit 2; }

LABEL="$PRESET-$CACHE"
[[ "$CACHE" == contraction ]] && LABEL="$LABEL-t$THRESHOLD-$EMBEDDER"
[[ "$MODE" == train ]] || LABEL="$LABEL-$MODE"
[[ -z "$NAME" ]] || LABEL="$LABEL-$NAME"

export PRESET TASKS CACHE THRESHOLD EMBEDDER BUDGET EPOCHS ROLLOUTS BATCH MAX_TOKENS VERIFY MODE
export RUN_LABEL="$LABEL"

echo "preset $PRESET -> $LABEL"
echo "  tasks ($(echo "$TASKS" | wc -w | tr -d ' ')): $TASKS"
echo "  cache: $CACHE$([[ "$CACHE" == contraction ]] && echo " (threshold $THRESHOLD, embedder $EMBEDDER, budget $BUDGET sandboxes)")"
echo "  epochs $EPOCHS, rollouts/task $ROLLOUTS, tasks/batch $BATCH, max tokens/rollout $MAX_TOKENS, tvverify $([[ "$VERIFY" == 1 ]] && echo on || echo off), mode $MODE"
[[ "$DRY_RUN" == 1 ]] && exit 0
exec bash "$HERE/train_pilot.sh"
