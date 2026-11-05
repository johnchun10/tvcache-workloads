#!/usr/bin/env bash
# The Terminal-Bench pilot on one Perlmutter GPU node: GRPO with veRL, every
# bash command through TVCache, then tvverify on the cache it built.
#
#   MODE=sample bash train_pilot.sh   # sampling test: 8 samples per task, no training (val_only)
#   MODE=train  bash train_pilot.sh   # the pilot: 5 epochs of GRPO
#   TVCACHE_VARIANT=main ...          # against TVCache main instead of the fix branches
#
# Run it inside an allocation, e.g.
#   sbatch -q shared -C gpu -G 2 -c 64 -t 08:00:00 -A <project> --wrap "MODE=train bash train_pilot.sh"
# Half a node for its RAM (about 120 GB): podman's image store on GPU nodes is
# /tmp, a RAM disk, and a quarter node ran out of memory (ISSUES.md #16).
# veRL still uses one GPU (N_GPUS).
# Everything (cache server, manager, containers, veRL) stays on this node, so
# snapshots never have to move between nodes. Results land in $RUN.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

MODE="${MODE:-train}"                  # sample | train
EPOCHS="${EPOCHS:-5}"
N_GPUS="${N_GPUS:-1}"
ROLLOUTS="${ROLLOUTS:-4}"              # GRPO group size (the paper: 8)
BATCH="${BATCH:-4}"                    # tasks per step (the paper's 4)
CHECK_NODES="${CHECK_NODES:-20}"       # tvverify nodes replayed per task
RUN="${RUN:-$TVROOT/runs/$(date +%Y%m%d-%H%M%S)-$MODE-$TVCACHE_VARIANT}"
WORKLOADS="$TVROOT/tvcache-workloads"
VERL_PKG="$WORKLOADS/src/tvcache_workloads/bash/verl"
mkdir -p "$RUN/logs"
export TVCACHE_NAMESPACE="$(basename "$RUN")"   # a fresh trie per run
# Fixed snapshot threshold (seconds of un-snapshotted runtime), so misses go
# through the restore path often. Empty = tvclient's adaptive policy, which
# took 4 snapshots in 686 commands in the sampling test.
export TVCACHE_SNAPSHOT_THRESHOLD="${SNAPSHOT_THRESHOLD-1.0}"
# The trace needs tvclient's observer hooks, which only the fix branches have.
if [[ "$TVCACHE_VARIANT" == fixes ]]; then export TVCACHE_TRACE_DIR="$RUN/trace"; else export TVCACHE_TRACE_DIR=""; fi
echo "run: $RUN (variant $TVCACHE_VARIANT, mode $MODE, snapshot threshold ${TVCACHE_SNAPSHOT_THRESHOLD:-adaptive})" | tee "$RUN/logs/run.txt"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT

# 0. memory log, once a minute: this job's cgroup memory (includes the /tmp
#    RAM disk) and the size of podman's image store
GRAPHROOT="$($CONTAINER_CLI info --format '{{.Store.GraphRoot}}' 2>/dev/null || echo /tmp)"
CGROUP_MEM="/sys/fs/cgroup$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)/memory.current"
(set +eu; echo "time,cgroup_mem_gb,graphroot_gb,containers,images"
 while true; do
   mem=$(awk '{printf "%.1f", $1/2^30}' "$CGROUP_MEM" 2>/dev/null || echo NA)
   root=$(du -s --block-size=1G "$GRAPHROOT" 2>/dev/null | cut -f1)
   echo "$(date +%FT%T),$mem,${root:-NA},$($CONTAINER_CLI ps -q 2>/dev/null | wc -l),$($CONTAINER_CLI images -q 2>/dev/null | wc -l)"
   sleep 60
 done) > "$RUN/logs/memory.csv" 2>/dev/null & PIDS+=($!)

# 1. podman's Docker API socket, the cache server (default eviction:
#    ttl_and_watermark, as in real training), the container manager
$CONTAINER_CLI system service --time=0 "unix://$PODMAN_SOCKET" > "$RUN/logs/podman.log" 2>&1 & PIDS+=($!)
"$(variant_dir)/tvcache-rust/target/release/tvcache" --port="$TVCACHE_PORT" > "$RUN/logs/cache.log" 2>&1 & PIDS+=($!)
MANAGER_REPO="$TVROOT/container-manager-public" bash "$WORKLOADS/bash/scripts/start_manager.sh" > "$RUN/logs/manager.log" 2>&1 & PIDS+=($!)
wait_for_port "$TVCACHE_PORT" 60 "tvcache"
wait_for_port "$FLASK_PORT" 600 "container manager"

# 2. the sidecar: periodic DUMP_ALL, and the one snapshot drainer
"$VERL_PY" "$HERE/cache_sidecar.py" --dump-dir "$RUN/dumps" --port "$TVCACHE_PORT" \
  --manager-url "$TVCACHE_MANAGER_URL" --interval "${DUMP_INTERVAL:-600}" > "$RUN/logs/sidecar.log" 2>&1 &
SIDECAR=$!

# 3. veRL
# Tasks that cannot build on Perlmutter: fix-git clones a GitHub repo that is
# gone (404); accelerate-maximal-square's base image has uids above the
# rootless podman range.
EXCLUDE_TASKS="${EXCLUDE_TASKS:-fix-git accelerate-maximal-square}"
# shellcheck disable=SC2086
"$VERL_PY" -m tvcache_workloads.bash.verl.dataset "$TASKS_ROOT" "$RUN/tasks.parquet" --exclude $EXCLUDE_TASKS
COMMON=(
  algorithm.adv_estimator=grpo
  data.train_files="$RUN/tasks.parquet" data.val_files="$RUN/tasks.parquet"
  data.train_batch_size="$BATCH" data.max_prompt_length=2048 data.max_response_length=2048
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
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=30 actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048
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

# 4. final dump (the sidecar dumps once more on SIGTERM), then tvverify while
#    the manager still holds the snapshots: C2a on the final trie, and
#    --no-snapshots on earlier dumps, whose snapshots may have been evicted
kill -TERM "$SIDECAR"; wait "$SIDECAR" || true
mkdir -p "$RUN/check"
DUMPS=("$RUN"/dumps/dump_*.json)
FINAL="${DUMPS[-1]}"
for dump in "${DUMPS[@]}"; do
  name="$(basename "$dump" .json)"
  flags=(); [[ "$dump" != "$FINAL" ]] && flags=(--no-snapshots)
  (cd "$WORKLOADS" && "$VERL_PY" -m tvverify check-graph --adapter tvcache_workloads.bash.verify:TerminalBenchAdapter \
     --dump "$dump" --max-replay-nodes "$CHECK_NODES" "${flags[@]}" --json "$RUN/check/$name.json") \
     > "$RUN/check/$name.log" 2>&1 || true
done
"$VERL_PY" -m tvverify report "$RUN"/check/*.json > "$RUN/report.md" || true

# 5. delete every snapshot this run's trie still holds (the sidecar is stopped,
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
echo "report: $RUN/report.md" | tee -a "$RUN/logs/run.txt"
leftover=$($CONTAINER_CLI images --filter label=com.docker.compose.project=tvcache-snapshot -q | wc -l)
echo "snapshot images left: $leftover" | tee -a "$RUN/logs/run.txt"
exit "$VERL_STATUS"
