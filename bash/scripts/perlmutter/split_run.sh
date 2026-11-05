#!/usr/bin/env bash
# The training pilot as two interactive-QOS allocations (4 h maximum each)
# on a full GPU node, back to back: part 1 trains steps 1..SPLIT, part 2
# resumes from part 1's last checkpoint and trains to TOTAL. Each part has its
# own RUN (dumps, checks and logs restart at 000); the cache restarts empty in
# part 2, and its snapshots do not carry over (they live in the node's /tmp).
#
# Run it on a login node inside tmux, since salloc must stay attached:
#   tmux new -d -s tvsplit 'TVCACHE_VARIANT=main bash split_run.sh'
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VARIANT="${TVCACHE_VARIANT:-main}"
SPLIT="${SPLIT:-30}"
TOTAL="${TOTAL:-60}"
ACCOUNT="${ACCOUNT:-m4999}"
BASE="${BASE:-$SCRATCH/tvcache/runs/$(date +%Y%m%d-%H%M%S)-train-$VARIANT}"
ALLOC=(-q interactive -C gpu -N 1 -G 4 -t 04:00:00 -A "$ACCOUNT")
# One task with the whole node: every CPU, GPU and all the memory, since the
# /tmp RAM disk that holds podman's images counts toward it.
STEP=(-N 1 -n 1 -c 128 --gpus=4 --mem=0 --cpu-bind=none)

part() {  # part <name> <total steps>
  local run="$BASE-$1"
  mkdir -p "$run"
  echo "$(date +%FT%T) $1: salloc, steps to $2, RUN=$run"
  salloc "${ALLOC[@]}" -J "tv-train-$VARIANT-$1" \
    srun "${STEP[@]}" env TVCACHE_VARIANT="$VARIANT" MODE=train TOTAL_STEPS="$2" RUN="$run" \
    bash "$HERE/train_pilot.sh" > "$run/part.out" 2>&1
  echo "$(date +%FT%T) $1: exited $?"
}

part part1 "$SPLIT"
CKPT="$BASE-part1/checkpoints"
if [[ "$(cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null)" == "$SPLIT" ]]; then
  mkdir -p "$BASE-part2"
  cp -al "$CKPT" "$BASE-part2/" 2>/dev/null || cp -r "$CKPT" "$BASE-part2/"
  part part2 "$TOTAL"
else
  echo "$(date +%FT%T) part1 did not checkpoint step $SPLIT; part2 not started"
  exit 1
fi
