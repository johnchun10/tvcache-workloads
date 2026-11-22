#!/usr/bin/env bash
# One run as a non-interactive heterogeneous job: component 0 is a CPU node
# (podman, containers and their snapshots, cache server, manager, sidecar with
# the fork bank, contraction coordinator), component 1 the GPUs (veRL): half
# of a node with 80 GB GPUs, in the shared QOS, which is charged as half a
# node. GPU_NODE=full at submit_hetjob.sh asks for a whole 40 GB node instead
# (it rewrites the two #SBATCH lines after "hetjob" below). The GPUs are asked
# for with --gpus, not --gpus-per-task: the shared queue counted the latter as
# one GPU and refused 64 cores for it.
# Slurm starts both together. Submit it with submit_hetjob.sh, which sets
# HERE, RUN, PRESET and the run.sh flags ($RUN/run_args), and queues tvverify
# as a job of its own for afterwards (verify.sbatch).
#
# CACHES (set by submit_hetjob.sh --cache all) runs each cache mode in turn, each in
# its own $RUN-<cache> with a fresh cache server (an empty graph); empty = the
# preset's own mode, in $RUN.
#SBATCH -J tv-het
#SBATCH -A m4999 -q regular -t 05:00:00
#SBATCH -C cpu -N 1
#SBATCH hetjob
#SBATCH -A m4999 -q shared -t 05:00:00
#SBATCH -C gpu&hbm80g -n 1 -c 64 --gpus=2
set -uo pipefail
HERE="${HERE:?submit with submit_hetjob.sh (sets HERE)}"
RUN_BASE="${RUN:?submit with submit_hetjob.sh (sets RUN)}"
PRESET="${PRESET:?submit with submit_hetjob.sh (sets PRESET)}"
ARGS=()
[[ -s "$RUN_BASE/run_args" ]] && mapfile -t ARGS < "$RUN_BASE/run_args"
CACHES="${CACHES:-}"
echo "$(date +%FT%T) job $SLURM_JOB_ID: cpu $SLURM_JOB_NODELIST_HET_GROUP_0, gpu $SLURM_JOB_NODELIST_HET_GROUP_1, preset $PRESET ${ARGS[*]+${ARGS[*]}}${CACHES:+, caches: $CACHES}"

modes=(${CACHES:-""})
status=0
# The GPU component's step: every GPU it was given, with a quarter of the
# node's 128 hardware threads per GPU (what the shared QOS allocates).
GPUS="${N_GPUS:-2}"
gpu_step=(-n 1 -c $(( 32 * GPUS )) --gpus="$GPUS" --cpu-bind=none)
if (( GPUS == 4 )); then gpu_step=(-N 1 "${gpu_step[@]}" --mem=0); fi   # a whole node: all its memory too
for cache in "${modes[@]}"; do
  RUN="$RUN_BASE"; flags=(); suffix=""
  if [[ -n "$cache" ]]; then flags=(--cache "$cache"); [[ ${#modes[@]} -gt 1 ]] && suffix="-$cache"; fi
  RUN="$RUN_BASE$suffix"
  # A later part of a split run resumes from the same mode's previous part.
  resume=""; [[ -n "${RESUME_BASE:-}" ]] && resume="$RESUME_BASE$suffix/checkpoints"
  mkdir -p "$RUN"
  rm -f "$RUN/client.done"
  echo "$(date +%FT%T) ${cache:-preset}: RUN=$RUN"
  # The whole CPU node to the server: every CPU and all memory (the /tmp RAM
  # disk with podman's images counts toward it; 256 hardware threads).
  # /usr/bin/env, not env: srun runs the first file of that name on PATH, and
  # the uv installer leaves a non-executable ~/.local/bin/env (ISSUES.md #27).
  srun --het-group=0 -N 1 -n 1 -c 256 --mem=0 --cpu-bind=none \
    /usr/bin/env ROLE=server RUN="$RUN" RESUME_FROM="$resume" \
    bash "$HERE/run.sh" "$PRESET" ${ARGS[@]+"${ARGS[@]}"} ${flags[@]+"${flags[@]}"} > "$RUN/server.out" 2>&1 &
  SERVER=$!
  srun --het-group=1 "${gpu_step[@]}" \
    /usr/bin/env ROLE=client RUN="$RUN" RESUME_FROM="$resume" \
    bash "$HERE/run.sh" "$PRESET" ${ARGS[@]+"${ARGS[@]}"} ${flags[@]+"${flags[@]}"} > "$RUN/client.out" 2>&1
  client_status=$?
  # The client writes client.done itself; this covers a client that died
  # first, so the server never waits forever.
  [[ -e "$RUN/client.done" ]] || echo "$client_status" > "$RUN/client.done"
  echo "$(date +%FT%T) ${cache:-preset}: client exited $client_status"
  # The server ends soon after the client: it writes the final dump, saves
  # the snapshots for tvverify and stops. (It used to run tvverify here, with
  # the GPU node cancelled meanwhile, which left an incomplete hetjob that
  # Slurm killed at its next restart: ISSUES.md #33.)
  wait "$SERVER"; server_status=$?   # kept first: $(date) below would reset $?
  echo "$(date +%FT%T) ${cache:-preset}: server exited $server_status"
  (( client_status != 0 )) && status=$client_status
done

# Both modes done: the testing plan's ratios (contraction / tvcache).
if [[ " $CACHES " == *" tvcache "* && " $CACHES " == *" contraction "* ]]; then
  source "$HERE/env.sh"
  "$VERL_PY" -m tvcache_workloads.bash.verl.stats --compare "$RUN_BASE-tvcache" "$RUN_BASE-contraction" \
    > "$RUN_BASE-compare.md" 2>&1 || true
fi
exit "$status"
