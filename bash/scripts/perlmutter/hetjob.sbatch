#!/usr/bin/env bash
# One run as a non-interactive heterogeneous job: component 0 is a CPU node
# (podman, containers and their snapshots, cache server, manager, sidecar with
# the fork bank, contraction coordinator, tvverify), component 1 a GPU node
# (veRL). Slurm starts both together. Submit it with submit_hetjob.sh, which
# sets HERE, RUN, PRESET and the run.sh flags ($RUN/run_args).
#
# CACHES="tvcache contraction" runs each cache mode in turn, each in its own
# $RUN-<cache> with a fresh cache server (an empty graph); empty = the
# preset's own mode, in $RUN.
#SBATCH -J tv-het
#SBATCH -A m4999 -q regular -t 05:00:00
#SBATCH -C cpu -N 1
#SBATCH hetjob
#SBATCH -A m4999 -q regular -t 05:00:00
#SBATCH -C gpu -N 1 -G 4
set -uo pipefail
HERE="${HERE:?submit with submit_hetjob.sh (sets HERE)}"
RUN_BASE="${RUN:?submit with submit_hetjob.sh (sets RUN)}"
PRESET="${PRESET:?submit with submit_hetjob.sh (sets PRESET)}"
ARGS=()
[[ -s "$RUN_BASE/run_args" ]] && mapfile -t ARGS < "$RUN_BASE/run_args"
CACHES="${CACHES:-}"
echo "$(date +%FT%T) job $SLURM_JOB_ID: cpu $SLURM_JOB_NODELIST_HET_GROUP_0, gpu $SLURM_JOB_NODELIST_HET_GROUP_1, preset $PRESET ${ARGS[*]+${ARGS[*]}}${CACHES:+, caches: $CACHES}"

modes=(${CACHES:-""})
last="${modes[${#modes[@]}-1]}"
status=0
for cache in "${modes[@]}"; do
  RUN="$RUN_BASE"; flags=(); suffix=""
  if [[ -n "$cache" ]]; then flags=(--cache "$cache"); [[ ${#modes[@]} -gt 1 ]] && suffix="-$cache"; fi
  RUN="$RUN_BASE$suffix"
  # A later part of a split run resumes from the same mode's previous part.
  resume=""; [[ -n "${RESUME_BASE:-}" ]] && resume="$RESUME_BASE$suffix/checkpoints"
  mkdir -p "$RUN"
  rm -f "$RUN/client.done"
  echo "$(date +%FT%T) ${cache:-preset}: RUN=$RUN"
  # Whole node to each role: every CPU and all memory (the /tmp RAM disk with
  # podman's images counts toward it). CPU nodes have 256 hardware threads,
  # GPU nodes 128.
  srun --het-group=0 -N 1 -n 1 -c 256 --mem=0 --cpu-bind=none \
    env ROLE=server RUN="$RUN" RESUME_FROM="$resume" \
    bash "$HERE/run.sh" "$PRESET" ${ARGS[@]+"${ARGS[@]}"} ${flags[@]+"${flags[@]}"} > "$RUN/server.out" 2>&1 &
  SERVER=$!
  srun --het-group=1 -N 1 -n 1 -c 128 --gpus=4 --mem=0 --cpu-bind=none \
    env ROLE=client RUN="$RUN" RESUME_FROM="$resume" \
    bash "$HERE/run.sh" "$PRESET" ${ARGS[@]+"${ARGS[@]}"} ${flags[@]+"${flags[@]}"} > "$RUN/client.out" 2>&1
  client_status=$?
  # The client writes client.done itself; this covers a client that died
  # first, so the server never waits forever.
  [[ -e "$RUN/client.done" ]] || echo "$client_status" > "$RUN/client.done"
  echo "$(date +%FT%T) ${cache:-preset}: client exited $client_status"
  # After the last mode's training the GPU node only idles while the server
  # runs tvverify; give it back (probe_hetjob.sbatch showed the CPU component
  # keeps running).
  if [[ "${RELEASE_GPU:-1}" == 1 && "$cache" == "$last" ]]; then
    scancel "${SLURM_JOB_ID}+1" && echo "$(date +%FT%T) released the GPU component"
  fi
  wait "$SERVER"
  echo "$(date +%FT%T) ${cache:-preset}: server exited $?"
  (( client_status != 0 )) && status=$client_status
done

# Both modes done: the testing plan's ratios (contraction / tvcache).
if [[ " $CACHES " == *" tvcache "* && " $CACHES " == *" contraction "* ]]; then
  source "$HERE/env.sh"
  "$VERL_PY" -m tvcache_workloads.bash.verl.stats --compare "$RUN_BASE-tvcache" "$RUN_BASE-contraction" \
    > "$RUN_BASE-compare.md" 2>&1 || true
fi
exit "$status"
