#!/usr/bin/env bash
# One Terminal-Bench run on one Perlmutter GPU node: GRPO with veRL, every bash
# command through TVCache (or not, or with graph contraction), then tvverify on
# the cache it built. Start it through run.sh, which sets these from a preset:
#
#   bash run.sh shakedown
#   bash run.sh main20-easy-med --cache tvcache
#
# or set the variables yourself (the old way still works):
#   MODE=sample bash train_pilot.sh   # sampling test: rollouts only, no training (val_only)
#   MODE=train  bash train_pilot.sh   # GRPO training
#   CACHE=none ...                    # no cache at all: the with/without baseline
#   CACHE=contraction THRESHOLD=0.9 EMBEDDER=minilm ...
#
# Run it inside an allocation, e.g.
#   sbatch -q shared -C gpu -G 2 -c 64 -t 08:00:00 -A <project> --wrap "bash run.sh main20-easy-med"
# Half a node for its RAM (about 120 GB): podman's image store on GPU nodes is
# /tmp, a RAM disk, and a quarter node ran out of memory (ISSUES.md #16).
# veRL still uses one GPU (N_GPUS).
# Everything (cache server, manager, containers, veRL) stays on this node, so
# snapshots never have to move between nodes. Results land in $RUN.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

MODE="${MODE:-train}"                  # sample | train | prebuild (build the task images and exit)
EPOCHS="${EPOCHS:-5}"
N_GPUS="${N_GPUS:-1}"
ROLLOUTS="${ROLLOUTS:-4}"              # GRPO group size (the paper: 8)
BATCH="${BATCH:-4}"                    # tasks per step (the paper's 4)
MAX_TOKENS="${MAX_TOKENS:-2048}"       # tokens per rollout after the prompt (model output plus tool output)
MAX_TURNS="${MAX_TURNS:-30}"           # tool calls per rollout
VERIFY="${VERIFY:-1}"                  # 1 = tvverify check-graph on the dumps afterwards
CHECK_NODES="${CHECK_NODES:-20}"       # tvverify nodes replayed per task
THRESHOLD="${THRESHOLD:-0.9}"          # contraction: similarity threshold
EMBEDDER="${EMBEDDER:-minilm}"         # contraction: minilm | jina-code | a sentence-transformers model name
BUDGET="${BUDGET:-32}"                 # contraction: live sandboxes all jobs share
TASKS="${TASKS:-}"                     # task names; empty = every task in $TASKS_ROOT
TASK_SOURCES="${TASK_SOURCES:-$TVROOT/container-manager-public/tasks_easy_network_skip $TVROOT/container-manager-public/tasks_medium_network_skip}"
RUN="${RUN:-$TVROOT/runs/$(date +%Y%m%d-%H%M%S)-${RUN_LABEL:-$MODE-$CACHE}}"
WORKLOADS="$TVROOT/tvcache-workloads"
VERL_PKG="$WORKLOADS/src/tvcache_workloads/bash/verl"
mkdir -p "$RUN/logs"
export TVCACHE_NAMESPACE="$(basename "$RUN")"   # a fresh trie per run
# Fixed snapshot threshold (seconds of un-snapshotted runtime), so misses go
# through the restore path often. Empty = tvclient's adaptive policy, which
# took 4 snapshots in 686 commands in the sampling test.
export TVCACHE_SNAPSHOT_THRESHOLD="${SNAPSHOT_THRESHOLD-1.0}"
# The trace needs tvclient's observer hooks (the fixes and combined builds).
if [[ "$VERIFY" == 1 && "$TVCACHE_USE_CACHE" == 1 && -f "$(variant_dir)/tvclient/tvclient/observer.py" ]]; then
  export TVCACHE_TRACE_DIR="$RUN/trace"
else
  export TVCACHE_TRACE_DIR=""
fi
if [[ "$CACHE" == contraction && ! -f "$(variant_dir)/tvclient/tvclient/contraction.py" ]]; then
  echo "CACHE=contraction needs a tvclient with contraction; $(variant_dir)/tvclient has none (use TVCACHE_VARIANT=combined)" >&2
  exit 1
fi
[[ "$TVCACHE_USE_CACHE" == 0 || -x "$(variant_dir)/tvcache-rust/target/release/tvcache" || "$MODE" == prebuild ]] || {
  echo "no cache server at $(variant_dir)/tvcache-rust/target/release/tvcache; run setup.sh" >&2; exit 1; }

# The task set: a list of names is copied into $RUN/tasks, which then is the
# one directory both the dataset builder and the container manager read.
EXCLUDE_TASKS="${EXCLUDE_TASKS-fix-git accelerate-maximal-square}"
if [[ -n "$TASKS" ]]; then
  rm -rf "$RUN/tasks"; mkdir -p "$RUN/tasks"
  for task in $TASKS; do
    source_dir=""
    for root in $TASK_SOURCES; do
      [[ -f "$root/$task/task.yaml" ]] && { source_dir="$root/$task"; break; }
    done
    [[ -n "$source_dir" ]] || { echo "task '$task' not found under: $TASK_SOURCES" >&2; exit 1; }
    cp -R "$source_dir" "$RUN/tasks/$task"
  done
  export TASKS_ROOT="$RUN/tasks"
  EXCLUDE_TASKS=""
fi
TASK_NAMES="$(cd "$TASKS_ROOT" && for d in */; do d="${d%/}"; if [[ -f "$d/task.yaml" && " $EXCLUDE_TASKS " != *" $d "* ]]; then printf '%s ' "$d"; fi; done)"

{
  echo "run: $RUN"
  echo "mode $MODE, cache $CACHE (build $TVCACHE_VARIANT), snapshot threshold ${TVCACHE_SNAPSHOT_THRESHOLD:-adaptive}"
  [[ "$CACHE" == contraction ]] && echo "contraction: threshold $THRESHOLD, embedder $EMBEDDER, budget $BUDGET sandboxes"
  echo "epochs $EPOCHS, rollouts/task $ROLLOUTS, tasks/batch $BATCH, max tokens/rollout $MAX_TOKENS, max turns $MAX_TURNS, tvverify $VERIFY"
  echo "tasks ($(echo $TASK_NAMES | wc -w | tr -d ' ')): $TASK_NAMES"
} | tee "$RUN/logs/run.txt"
for name in PRESET MODE CACHE TVCACHE_VARIANT THRESHOLD EMBEDDER BUDGET EPOCHS ROLLOUTS BATCH MAX_TOKENS MAX_TURNS VERIFY \
            CHECK_NODES N_GPUS MODEL TASKS_ROOT TASK_NAMES TVCACHE_SNAPSHOT_THRESHOLD; do
  printf '%s=%q\n' "$name" "${!name-}"
done > "$RUN/config.env"

PIDS=()
cleanup() {
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT

# 0. memory logs, once a minute.
#    memory.csv: this job's cgroup memory (includes the /tmp RAM disk), the
#    size of podman's image store, and how many containers and images exist.
#    containers.csv: memory per running container, where podman can report it.
GRAPHROOT="$($CONTAINER_CLI info --format '{{.Store.GraphRoot}}' 2>/dev/null || echo /tmp)"
CGROUP_MEM="/sys/fs/cgroup$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null || true)/memory.current"
(set +eu; echo "time,cgroup_mem_gb,graphroot_gb,containers,images"
 while true; do
   mem=$(awk '{printf "%.1f", $1/2^30}' "$CGROUP_MEM" 2>/dev/null || echo NA)
   root=$(du -s --block-size=1G "$GRAPHROOT" 2>/dev/null | cut -f1)
   echo "$(date +%FT%T),$mem,${root:-NA},$($CONTAINER_CLI ps -q 2>/dev/null | wc -l),$($CONTAINER_CLI images -q 2>/dev/null | wc -l)"
   sleep 60
 done) > "$RUN/logs/memory.csv" 2>/dev/null & PIDS+=($!)
(set +eu; echo "time,container,mem_usage"
 while true; do
   now="$(date +%FT%T)"
   $CONTAINER_CLI stats --no-stream --format '{{.Name}},{{.MemUsage}}' 2>/dev/null | sed "s/^/$now,/"
   sleep 60
 done) > "$RUN/logs/containers.csv" 2>/dev/null & PIDS+=($!)

# 1. podman's Docker API socket, the cache server (default eviction:
#    ttl_and_watermark, as in real training), the container manager.
#    CACHE=none starts no cache server and no sidecar.
$CONTAINER_CLI system service --time=0 "unix://$PODMAN_SOCKET" > "$RUN/logs/podman.log" 2>&1 & PIDS+=($!)
if [[ "$TVCACHE_USE_CACHE" == 1 && "$MODE" != prebuild ]]; then
  "$(variant_dir)/tvcache-rust/target/release/tvcache" --port="$TVCACHE_PORT" > "$RUN/logs/cache.log" 2>&1 & PIDS+=($!)
fi
MANAGER_REPO="$TVROOT/container-manager-public" bash "$WORKLOADS/bash/scripts/start_manager.sh" > "$RUN/logs/manager.log" 2>&1 & PIDS+=($!)
[[ "$TVCACHE_USE_CACHE" == 1 && "$MODE" != prebuild ]] && wait_for_port "$TVCACHE_PORT" 60 "tvcache"
wait_for_port "$FLASK_PORT" 600 "container manager"

if [[ "$MODE" == prebuild ]]; then
  # Builds every task image once through the manager, to find tasks that do
  # not build before paying for GPU time. Fine on a login node.
  "$VERL_PY" "$HERE/prebuild.py" "$TASKS_ROOT" --manager-url "$TVCACHE_MANAGER_URL" --parallel 4 \
    --out "$RUN/prebuild.json" | tee -a "$RUN/logs/run.txt"
  exit 0
fi

# 2. the sidecar: periodic DUMP_ALL, and the one snapshot drainer
SIDECAR=""
if [[ "$TVCACHE_USE_CACHE" == 1 ]]; then
  "$VERL_PY" "$HERE/cache_sidecar.py" --dump-dir "$RUN/dumps" --port "$TVCACHE_PORT" \
    --manager-url "$TVCACHE_MANAGER_URL" --interval "${DUMP_INTERVAL:-600}" > "$RUN/logs/sidecar.log" 2>&1 &
  SIDECAR=$!
fi

# 3. contraction: calibrate the state comparison per task, then start the
#    coordinator that contracts a task whenever none of its rollouts is running
COORDINATOR=""
if [[ "$CACHE" == contraction ]]; then
  # What "the same state" covers: more than /app, plus the installed software
  # (state.py). The check below records, per task, the paths that differ
  # between identical sandboxes anyway, and the tasks that cannot be compared.
  export TVCACHE_STATE_ROOTS="${TVCACHE_STATE_ROOTS:-/app /root /etc /tmp}"
  export TVCACHE_STATE_PACKAGES="${TVCACHE_STATE_PACKAGES:-1}"
  # shellcheck disable=SC2086
  "$VERL_PY" -m tvcache_workloads.bash.state_check --tasks $TASK_NAMES --out "$RUN/state.json" \
    --manager-url "$TVCACHE_MANAGER_URL" 2>&1 | tee "$RUN/logs/state_check.log" | tee -a "$RUN/logs/run.txt" || {
      echo "state check: no task can be compared, so there is nothing to contract; stopping" | tee -a "$RUN/logs/run.txt"; exit 1; }
  export TVCACHE_STATE_FILE="$RUN/state.json"
  export TVCACHE_EMBEDDER="$EMBEDDER"
  export TVCACHE_CONTRACTION_PORT="${TVCACHE_CONTRACTION_PORT:-$TVCACHE_CONTRACTION_PORT_DEFAULT}"
  mkdir -p "$RUN/contraction"
  "$VERL_PY" -m tvcache_workloads.bash.contraction serve --threshold "$THRESHOLD" --max-sandboxes "$BUDGET" \
    --port "$TVCACHE_CONTRACTION_PORT" --cache-port "$TVCACHE_PORT" --manager-url "$TVCACHE_MANAGER_URL" \
    --events "$RUN/contraction/events.jsonl" > "$RUN/logs/contraction.log" 2>&1 &
  COORDINATOR=$!
  wait_for_port "$TVCACHE_CONTRACTION_PORT" 60 "contraction coordinator"
else
  export TVCACHE_EMBEDDER="" TVCACHE_CONTRACTION_PORT=""
fi

# 4. veRL
# Tasks that cannot build on Perlmutter are in EXCLUDE_TASKS: fix-git clones a
# GitHub repo that is gone (404); accelerate-maximal-square's base image has
# uids above the rootless podman range.
# shellcheck disable=SC2086
"$VERL_PY" -m tvcache_workloads.bash.verl.dataset "$TASKS_ROOT" "$RUN/tasks.parquet" --exclude $EXCLUDE_TASKS
COMMON=(
  algorithm.adv_estimator=grpo
  data.train_files="$RUN/tasks.parquet" data.val_files="$RUN/tasks.parquet"
  data.train_batch_size="$BATCH" data.max_prompt_length=2048 data.max_response_length="$MAX_TOKENS"
  data.filter_overlong_prompts=True
  actor_rollout_ref.model.path="$MODEL"
  # flash-attn is not installed (no wheel for torch 2.11 + CUDA 13), so use
  # PyTorch's SDPA. Remove-padding packs sequences and needs flash-attn's
  # varlen kernels to keep them apart, so it goes off too. vLLM is unaffected.
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  actor_rollout_ref.model.use_remove_padding=False
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH" actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.n="$ROLLOUTS" actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
  actor_rollout_ref.rollout.multi_turn.enable=True actor_rollout_ref.rollout.multi_turn.format=hermes
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$VERL_PKG/configs/tools.yaml"
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$VERL_PKG/configs/agent_loops.yaml"
  actor_rollout_ref.rollout.agent.default_agent_loop=tvcache_tool_agent
  reward.custom_reward_function.path="$VERL_PKG/reward.py" reward.custom_reward_function.name=compute_score
  trainer.nnodes=1 trainer.n_gpus_per_node="$N_GPUS" trainer.logger=[console]
  trainer.project_name=tvcache trainer.experiment_name="$(basename "$RUN")"
  trainer.default_local_dir="$RUN/checkpoints" trainer.save_freq=10 trainer.max_actor_ckpt_to_keep=1 trainer.resume_mode=auto
)
if [[ "$MODE" == sample ]]; then
  EXTRA=(trainer.val_only=True trainer.val_before_train=True
         actor_rollout_ref.rollout.val_kwargs.n="$ROLLOUTS" actor_rollout_ref.rollout.val_kwargs.do_sample=True
         actor_rollout_ref.rollout.val_kwargs.temperature=1.0)
else
  EXTRA=(trainer.total_epochs="$EPOCHS" trainer.val_before_train=False trainer.test_freq=-1)
fi
set +e
"$VERL_PY" -m verl.trainer.main_ppo "${COMMON[@]}" "${EXTRA[@]}" 2>&1 | tee "$RUN/logs/verl.log"
VERL_STATUS=${PIPESTATUS[0]}
set -e
echo "veRL exited $VERL_STATUS" | tee -a "$RUN/logs/run.txt"
echo "peak memory: $(awk -F, 'NR>1 && $2!="NA" && $2+0>m {m=$2+0} END {print m+0}' "$RUN/logs/memory.csv") GB (job cgroup), most containers at once: $(awk -F, 'NR>1 && $4+0>m {m=$4+0} END {print m+0}' "$RUN/logs/memory.csv")" | tee -a "$RUN/logs/run.txt"

if [[ "$TVCACHE_USE_CACHE" != 1 ]]; then
  # no cache: nothing to verify and no snapshots to release
  echo "no cache: skipped tvverify and snapshot release" | tee -a "$RUN/logs/run.txt"
  echo "containers left: $($CONTAINER_CLI ps -q 2>/dev/null | wc -l)" | tee -a "$RUN/logs/run.txt"
  exit "$VERL_STATUS"
fi

# 5. stop contraction (it finishes the merge it is applying and stops its
#    sandboxes), and add up what it did
if [[ -n "$COORDINATOR" ]]; then
  kill -TERM "$COORDINATOR" 2>/dev/null || true; wait "$COORDINATOR" 2>/dev/null || true
  if [[ -s "$RUN/contraction/events.jsonl" ]]; then
    "$VERL_PY" -m tvcache_workloads.bash.contraction summary "$RUN/contraction/events.jsonl" \
      --json "$RUN/contraction/summary.json" | tee -a "$RUN/logs/run.txt" || true
  fi
fi

# 6. final dump (the sidecar dumps once more on SIGTERM), then tvverify while
#    the manager still holds the snapshots: C2a on the final trie, and
#    --no-snapshots on earlier dumps, whose snapshots may have been evicted
kill -TERM "$SIDECAR"; wait "$SIDECAR" || true
DUMPS=("$RUN"/dumps/dump_*.json)
FINAL="${DUMPS[${#DUMPS[@]}-1]}"
if [[ "$VERIFY" == 1 ]]; then
  mkdir -p "$RUN/check"
  for dump in "${DUMPS[@]}"; do
    name="$(basename "$dump" .json)"
    snapshots_flag=""; [[ "$dump" != "$FINAL" ]] && snapshots_flag="--no-snapshots"
    # shellcheck disable=SC2086
    (cd "$WORKLOADS" && "$VERL_PY" -m tvverify check-graph --adapter tvcache_workloads.bash.verify:TerminalBenchAdapter \
       --dump "$dump" --max-replay-nodes "$CHECK_NODES" $snapshots_flag --json "$RUN/check/$name.json") \
       > "$RUN/check/$name.log" 2>&1 || true
  done
  "$VERL_PY" -m tvverify report "$RUN"/check/*.json > "$RUN/report.md" || true
  echo "report: $RUN/report.md" | tee -a "$RUN/logs/run.txt"
else
  echo "tvverify: skipped (VERIFY=0)" | tee -a "$RUN/logs/run.txt"
fi

# 7. delete every snapshot this run's trie still holds (the sidecar is stopped,
#    so this is the only drainer now)
"$VERL_PY" - "$FINAL" <<'PY'
import asyncio, json, os, sys
from tvcache_workloads.bash.cache_admin import release_task_snapshots
from tvcache_workloads.bash.sandbox import SandboxConfig, TerminalBenchSandbox
sandbox = TerminalBenchSandbox.with_config(SandboxConfig(manager_url=os.environ["TVCACHE_MANAGER_URL"]))
port = int(os.environ["TVCACHE_PORT"])
async def go():
    for task in json.load(open(sys.argv[1])):
        released = await release_task_snapshots(task, sandbox, port=port)
        print(f"{task}: released {len(released)} snapshots")
asyncio.run(go())
PY
leftover=$($CONTAINER_CLI images --filter label=com.docker.compose.project=tvcache-snapshot -q | wc -l)
echo "snapshot images left: $leftover" | tee -a "$RUN/logs/run.txt"
echo "containers left: $($CONTAINER_CLI ps -q 2>/dev/null | wc -l)" | tee -a "$RUN/logs/run.txt"
exit "$VERL_STATUS"
