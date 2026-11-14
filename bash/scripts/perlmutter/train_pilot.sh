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
# veRL still uses one GPU (N_GPUS). Results land in $RUN.
#
# ROLE picks what runs on this node:
#   all     everything on one GPU node (cache server, manager, containers,
#           veRL), so snapshots never move between nodes
#   server  CPU node: podman, cache server, manager, sidecar (dumps, fork bank),
#           contraction coordinator, then tvverify and snapshot release once
#           the client is done. Containers and snapshots stay here.
#   client  GPU node: veRL only. It waits for $RUN/server.ready (the server's
#           hostname) and sends cache, manager and control traffic there.
# server and client share one RUN (on scratch) and are started together by
# hetjob.sbatch. The handshake files are $RUN/server.ready and $RUN/client.done
# (veRL's exit status; hetjob.sbatch writes it too, in case the client dies
# before it can).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

ROLE="${ROLE:-all}"                    # all | server | client
MODE="${MODE:-train}"                  # sample | train | prebuild (build the task images and exit)
EPOCHS="${EPOCHS:-5}"
TOTAL_STEPS="${TOTAL_STEPS:-}"         # stop early after this many steps (empty = all epochs)
RESUME_FROM="${RESUME_FROM:-}"         # a previous part's checkpoints/ dir, copied into $RUN before veRL starts
SEED="${SEED:-42}"                     # the dataloader's shuffle
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
FORK_BANK="${FORK_BANK:-1}"            # TVCache's prewarming: warm roots for the next batch, pre-forked hot snapshots
TOP_K="${TOP_K:-10}"                   # fork bank: hot snapshots pre-forked per batch
TASKS="${TASKS:-}"                     # task names; empty = every task in $TASKS_ROOT
TASK_SOURCES="${TASK_SOURCES:-$TVROOT/container-manager-public/tasks_easy_network_skip $TVROOT/container-manager-public/tasks_medium_network_skip}"
RUN="${RUN:-$TVROOT/runs/$(date +%Y%m%d-%H%M%S)-${RUN_LABEL:-$MODE-$CACHE}}"
WORKLOADS="$TVROOT/tvcache-workloads"
VERL_PKG="$WORKLOADS/src/tvcache_workloads/bash/verl"
case "$ROLE" in all|server|client) ;; *) echo "ROLE must be all, server or client, not '$ROLE'" >&2; exit 2 ;; esac
[[ "$MODE" != prebuild || "$ROLE" == all ]] || { echo "MODE=prebuild runs with ROLE=all" >&2; exit 2; }
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
[[ "$TVCACHE_USE_CACHE" == 0 || "$ROLE" == client || -x "$(variant_dir)/tvcache-rust/target/release/tvcache" || "$MODE" == prebuild ]] || {
  echo "no cache server at $(variant_dir)/tvcache-rust/target/release/tvcache; run setup.sh" >&2; exit 1; }
# The fork bank needs a tvclient whose ForkBank can follow the batches
# (set_tasks) and be served to other processes.
if [[ "$TVCACHE_USE_CACHE" != 1 || "$MODE" != train ]]; then
  FORK_BANK=0
elif [[ "$FORK_BANK" == 1 ]] && ! grep -q "def set_tasks" "$(variant_dir)/tvclient/tvclient/fork_bank.py" 2>/dev/null; then
  echo "fork bank: off ($(variant_dir)/tvclient has no ForkBank.set_tasks)" | tee -a "$RUN/logs/run.txt"
  FORK_BANK=0
fi
if [[ "$FORK_BANK" == 1 ]]; then
  export TVCACHE_CONTROL_PORT="${TVCACHE_CONTROL_PORT:-$TVCACHE_CONTROL_PORT_DEFAULT}"
else
  export TVCACHE_CONTROL_PORT=""
fi
# Read by rollouts (they embed and gate on the client) and by the coordinator.
if [[ "$CACHE" == contraction ]]; then
  export TVCACHE_EMBEDDER="$EMBEDDER"
  export TVCACHE_CONTRACTION_PORT="${TVCACHE_CONTRACTION_PORT:-$TVCACHE_CONTRACTION_PORT_DEFAULT}"
else
  export TVCACHE_EMBEDDER="" TVCACHE_CONTRACTION_PORT=""
fi

# The task set: a list of names is copied into $RUN/tasks, which then is the
# one directory both the dataset builder and the container manager read. Only
# the node running the manager copies; the client reads it after server.ready.
EXCLUDE_TASKS="${EXCLUDE_TASKS-fix-git accelerate-maximal-square}"
if [[ -n "$TASKS" && "$ROLE" == client ]]; then
  export TASKS_ROOT="$RUN/tasks"
  EXCLUDE_TASKS=""
elif [[ -n "$TASKS" ]]; then
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
if [[ -n "$TASKS" && "$ROLE" == client ]]; then
  TASK_NAMES="$TASKS"
else
  TASK_NAMES="$(cd "$TASKS_ROOT" && for d in */; do d="${d%/}"; if [[ -f "$d/task.yaml" && " $EXCLUDE_TASKS " != *" $d "* ]]; then printf '%s ' "$d"; fi; done)"
fi

{
  echo "run: $RUN (role $ROLE on $(hostname))"
  echo "mode $MODE, cache $CACHE (build $TVCACHE_VARIANT), snapshot threshold ${TVCACHE_SNAPSHOT_THRESHOLD:-adaptive}"
  [[ "$CACHE" == contraction ]] && echo "contraction: threshold $THRESHOLD, embedder $EMBEDDER, budget $BUDGET sandboxes"
  [[ "$FORK_BANK" == 1 ]] && echo "fork bank: $ROLLOUTS warm roots per task of the next batch, top $TOP_K hot snapshots"
  echo "epochs $EPOCHS, rollouts/task $ROLLOUTS, tasks/batch $BATCH, max tokens/rollout $MAX_TOKENS, max turns $MAX_TURNS, tvverify $VERIFY, seed $SEED"
  echo "tasks ($(echo $TASK_NAMES | wc -w | tr -d ' ')): $TASK_NAMES"
} | tee -a "$RUN/logs/run.txt"
if [[ "$ROLE" != client ]]; then
  for name in PRESET MODE CACHE TVCACHE_VARIANT THRESHOLD EMBEDDER BUDGET FORK_BANK TOP_K EPOCHS ROLLOUTS BATCH MAX_TOKENS \
              MAX_TURNS VERIFY CHECK_NODES N_GPUS MODEL SEED TASKS_ROOT TASK_NAMES TVCACHE_SNAPSHOT_THRESHOLD; do
    printf '%s=%q\n' "$name" "${!name-}"
  done > "$RUN/config.env"
fi

PIDS=()
cleanup() {
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT

# 0. memory logs, once a minute.
#    memory.csv: this job's cgroup memory (includes the /tmp RAM disk), the
#    size of podman's image store, how many containers and images exist, and
#    how many sandboxes (compose projects: a Terminal-Bench sandbox is one).
#    containers.csv: memory per running container, where podman can report it
#    (podman-hpc gives every container the same value; ISSUES.md #29).
#    The client role has no containers; its memory goes to memory-client.csv.
CGROUP_MEM="/sys/fs/cgroup$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null || true)/memory.current"
if [[ "$ROLE" == client ]]; then
  (set +eu; echo "time,cgroup_mem_gb"
   while true; do echo "$(date +%FT%T),$(awk '{printf "%.1f", $1/2^30}' "$CGROUP_MEM" 2>/dev/null || echo NA)"; sleep 60; done
  ) > "$RUN/logs/memory-client.csv" 2>/dev/null & PIDS+=($!)
else
  GRAPHROOT="$($CONTAINER_CLI info --format '{{.Store.GraphRoot}}' 2>/dev/null || echo /tmp)"
  (set +eu; echo "time,cgroup_mem_gb,graphroot_gb,containers,images,sandboxes"
   while true; do
     mem=$(awk '{printf "%.1f", $1/2^30}' "$CGROUP_MEM" 2>/dev/null || echo NA)
     root=$(du -s --block-size=1G "$GRAPHROOT" 2>/dev/null | cut -f1)
     sandboxes=$($CONTAINER_CLI ps --format '{{index .Labels "com.docker.compose.project"}}' 2>/dev/null | sort -u | grep -c .)
     echo "$(date +%FT%T),$mem,${root:-NA},$($CONTAINER_CLI ps -q 2>/dev/null | wc -l),$($CONTAINER_CLI images -q 2>/dev/null | wc -l),$sandboxes"
     sleep 60
   done) > "$RUN/logs/memory.csv" 2>/dev/null & PIDS+=($!)
  (set +eu; echo "time,container,mem_usage"
   while true; do
     now="$(date +%FT%T)"
     $CONTAINER_CLI stats --no-stream --format '{{.Name}},{{.MemUsage}}' 2>/dev/null | sed "s/^/$now,/"
     sleep 60
   done) > "$RUN/logs/containers.csv" 2>/dev/null & PIDS+=($!)
fi

start_services() {
  # 1. podman's Docker API socket, the cache server (default eviction:
  #    ttl_and_watermark, as in real training), the container manager. Both
  #    servers listen on 0.0.0.0, so a client on another node can reach them.
  #    CACHE=none starts no cache server and no sidecar.
  #    (|| return: under `if !` in the server role, set -e does not apply here)
  $CONTAINER_CLI system service --time=0 "unix://$PODMAN_SOCKET" > "$RUN/logs/podman.log" 2>&1 & PIDS+=($!)
  if [[ "$TVCACHE_USE_CACHE" == 1 && "$MODE" != prebuild ]]; then
    "$(variant_dir)/tvcache-rust/target/release/tvcache" --port="$TVCACHE_PORT" > "$RUN/logs/cache.log" 2>&1 & PIDS+=($!)
  fi
  MANAGER_REPO="$TVROOT/container-manager-public" bash "$WORKLOADS/bash/scripts/start_manager.sh" > "$RUN/logs/manager.log" 2>&1 & PIDS+=($!)
  if [[ "$TVCACHE_USE_CACHE" == 1 && "$MODE" != prebuild ]]; then wait_for_port "$TVCACHE_PORT" 60 "tvcache" || return 1; fi
  wait_for_port "$FLASK_PORT" 600 "container manager" || return 1

  if [[ "$MODE" == prebuild ]]; then
    # Builds every task image once through the manager, to find tasks that do
    # not build before paying for GPU time. Fine on a login node.
    "$VERL_PY" "$HERE/prebuild.py" "$TASKS_ROOT" --manager-url "$TVCACHE_MANAGER_URL" --parallel 4 \
      --out "$RUN/prebuild.json" | tee -a "$RUN/logs/run.txt"
    exit 0
  fi

  # 2. the sidecar: periodic DUMP_ALL, the one snapshot drainer, and (with the
  #    fork bank) the control port the trainer and rollouts use
  SIDECAR=""
  if [[ "$TVCACHE_USE_CACHE" == 1 ]]; then
    local bank=()
    [[ "$FORK_BANK" == 1 ]] && bank=(--control-port "$TVCACHE_CONTROL_PORT" --top-k "$TOP_K" --roots-per-task "$ROLLOUTS")
    "$VERL_PY" "$HERE/cache_sidecar.py" --dump-dir "$RUN/dumps" --port "$TVCACHE_PORT" \
      --manager-url "$TVCACHE_MANAGER_URL" --interval "${DUMP_INTERVAL:-600}" ${bank[@]+"${bank[@]}"} \
      > "$RUN/logs/sidecar.log" 2>&1 &
    SIDECAR=$!
    if [[ "$FORK_BANK" == 1 ]]; then wait_for_port "$TVCACHE_CONTROL_PORT" 120 "fork bank control port" || return 1; fi
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
        echo "state check: no task can be compared, so there is nothing to contract; stopping" | tee -a "$RUN/logs/run.txt"; return 1; }
    export TVCACHE_STATE_FILE="$RUN/state.json"
    mkdir -p "$RUN/contraction"
    # 0.0.0.0: rollouts on a client node hold and release tasks here too
    "$VERL_PY" -m tvcache_workloads.bash.contraction serve --threshold "$THRESHOLD" --max-sandboxes "$BUDGET" \
      --host 0.0.0.0 --port "$TVCACHE_CONTRACTION_PORT" --cache-port "$TVCACHE_PORT" --manager-url "$TVCACHE_MANAGER_URL" \
      --events "$RUN/contraction/events.jsonl" > "$RUN/logs/contraction.log" 2>&1 &
    COORDINATOR=$!
    wait_for_port "$TVCACHE_CONTRACTION_PORT" 60 "contraction coordinator" || return 1
  fi
}

connect_to_server() {
  # Wait for the server node to announce itself, then send everything there.
  local waited=0
  until [[ -s "$RUN/server.ready" ]]; do
    if [[ -e "$RUN/server.failed" ]]; then echo "server failed: $(cat "$RUN/server.failed")" >&2; exit 1; fi
    (( waited >= ${SERVER_WAIT:-1800} )) && { echo "no $RUN/server.ready after $waited s" >&2; exit 1; }
    sleep 10; waited=$(( waited + 10 ))
  done
  local host; host="$(cat "$RUN/server.ready")"
  export TVCACHE_HOST="$host" TVCACHE_MANAGER_URL="http://$host:$FLASK_PORT"
  if [[ "$TVCACHE_USE_CACHE" == 1 ]]; then wait_for_port "$TVCACHE_PORT" 60 "tvcache" "$host"; fi
  wait_for_port "$FLASK_PORT" 60 "container manager" "$host"
  if [[ -n "$TVCACHE_CONTROL_PORT" ]]; then wait_for_port "$TVCACHE_CONTROL_PORT" 60 "fork bank control port" "$host"; fi
  if [[ -n "$TVCACHE_CONTRACTION_PORT" ]]; then wait_for_port "$TVCACHE_CONTRACTION_PORT" 60 "contraction coordinator" "$host"; fi
  echo "client: cache $TVCACHE_HOST:$TVCACHE_PORT, manager $TVCACHE_MANAGER_URL" | tee -a "$RUN/logs/run.txt"
}

run_verl() {
  # 4. veRL
  if [[ -n "$RESUME_FROM" && ! -e "$RUN/checkpoints" ]]; then
    cp -al "$RESUME_FROM" "$RUN/checkpoints" 2>/dev/null || cp -r "$RESUME_FROM" "$RUN/checkpoints"
  fi
  # Tasks that cannot build on Perlmutter are in EXCLUDE_TASKS: fix-git clones a
  # GitHub repo that is gone (404); accelerate-maximal-square's base image has
  # uids above the rootless podman range.
  # shellcheck disable=SC2086
  "$VERL_PY" -m tvcache_workloads.bash.verl.dataset "$TASKS_ROOT" "$RUN/tasks.parquet" --exclude $EXCLUDE_TASKS
  COMMON=(
    algorithm.adv_estimator=grpo
    data.train_files="$RUN/tasks.parquet" data.val_files="$RUN/tasks.parquet"
    data.train_batch_size="$BATCH" data.max_prompt_length=2048 data.max_response_length="$MAX_TOKENS"
    data.filter_overlong_prompts=True data.seed="$SEED"
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
    trainer.default_local_dir="$RUN/checkpoints" trainer.save_freq="${SAVE_FREQ:-10}" trainer.max_actor_ckpt_to_keep=1 trainer.resume_mode=auto
  )
  # TVCache's step hooks (fork bank warm-up one batch ahead, step log): verl/trainer.py
  local entry=(-m verl.trainer.main_ppo)
  [[ "$MODE" == train ]] && { entry=(-m tvcache_workloads.bash.verl.train); COMMON+=(trainer.v1.trainer_mode=tvcache_sync); }
  if [[ "$MODE" == sample ]]; then
    EXTRA=(trainer.val_only=True trainer.val_before_train=True
           actor_rollout_ref.rollout.val_kwargs.n="$ROLLOUTS" actor_rollout_ref.rollout.val_kwargs.do_sample=True
           actor_rollout_ref.rollout.val_kwargs.temperature=1.0)
  else
    EXTRA=(trainer.total_epochs="$EPOCHS" trainer.val_before_train=False trainer.test_freq=-1)
    # Stop after this many steps in total (veRL checkpoints on the last step), to
    # split a run over several allocations. Each part needs its own RUN, since
    # dumps, checks and logs restart at 000; RESUME_FROM copies the previous
    # part's checkpoints/ into the new RUN and resume_mode=auto picks them up.
    if [[ -n "$TOTAL_STEPS" ]]; then EXTRA+=(trainer.total_training_steps="$TOTAL_STEPS"); fi
  fi
  set +e
  "$VERL_PY" "${entry[@]}" "${COMMON[@]}" "${EXTRA[@]}" 2>&1 | tee "$RUN/logs/verl.log"
  VERL_STATUS=${PIPESTATUS[0]}
  set -e
  echo "veRL exited $VERL_STATUS" | tee -a "$RUN/logs/run.txt"
}

finish() {
  echo "peak memory: $(awk -F, 'NR>1 && $2!="NA" && $2+0>m {m=$2+0} END {print m+0}' "$RUN/logs/memory.csv") GB (job cgroup), most containers at once: $(awk -F, 'NR>1 && $4+0>m {m=$4+0} END {print m+0}' "$RUN/logs/memory.csv")" | tee -a "$RUN/logs/run.txt"

  if [[ "$TVCACHE_USE_CACHE" != 1 ]]; then
    # no cache: nothing to verify and no snapshots to release
    echo "no cache: skipped tvverify and snapshot release" | tee -a "$RUN/logs/run.txt"
    echo "containers left: $($CONTAINER_CLI ps -q 2>/dev/null | wc -l)" | tee -a "$RUN/logs/run.txt"
    return 0
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

  # 6. final dump (the sidecar dumps once more on SIGTERM, and stops the fork
  #    bank's unused sandboxes), then tvverify while the manager still holds
  #    the snapshots: C2a on the final trie, and --no-snapshots on earlier
  #    dumps, whose snapshots may have been evicted
  kill -TERM "$SIDECAR"; wait "$SIDECAR" || true
  DUMPS=("$RUN"/dumps/dump_*.json)
  FINAL="${DUMPS[${#DUMPS[@]}-1]}"
  if [[ "$VERIFY" == 1 ]]; then
    # The final dump goes first: it is the only check that needs the snapshots,
    # which vanish with the node, and each replay takes ~9 s, so the 49-task
    # dump at CHECK_NODES=20 alone takes hours. Earlier dumps follow, newest
    # first, with whatever time the allocation has left (they can be rerun on
    # a login node afterwards); report.md is rewritten after each one.
    mkdir -p "$RUN/check"
    FINAL_CHECK_NODES="${FINAL_CHECK_NODES:-8}"
    for (( i=${#DUMPS[@]}-1; i>=0; i-- )); do
      dump="${DUMPS[$i]}"
      name="$(basename "$dump" .json)"
      flags=(--max-replay-nodes "$FINAL_CHECK_NODES")
      [[ "$dump" != "$FINAL" ]] && flags=(--max-replay-nodes "$CHECK_NODES" --no-snapshots)
      (cd "$WORKLOADS" && "$VERL_PY" -m tvverify check-graph --adapter tvcache_workloads.bash.verify:TerminalBenchAdapter \
         --dump "$dump" "${flags[@]}" --json "$RUN/check/$name.json") \
         > "$RUN/check/$name.log" 2>&1 || true
      "$VERL_PY" -m tvverify report "$RUN"/check/*.json > "$RUN/report.md" 2>/dev/null || true
    done
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
  # 8. the testing plan's metrics (metrics.md / metrics.json)
  "$VERL_PY" -m tvcache_workloads.bash.verl.stats "$RUN" > "$RUN/logs/stats.log" 2>&1 || true
  leftover=$($CONTAINER_CLI images --filter label=com.docker.compose.project=tvcache-snapshot -q | wc -l)
  echo "snapshot images left: $leftover" | tee -a "$RUN/logs/run.txt"
  echo "containers left: $($CONTAINER_CLI ps -q 2>/dev/null | wc -l)" | tee -a "$RUN/logs/run.txt"
}

VERL_STATUS=0
case "$ROLE" in
  all)
    export TVCACHE_HOST=127.0.0.1
    start_services
    run_verl
    finish
    exit "$VERL_STATUS"
    ;;
  server)
    rm -f "$RUN/server.ready" "$RUN/server.failed"
    if ! start_services; then echo "services did not start on $(hostname)" > "$RUN/server.failed"; exit 1; fi
    echo "$(hostname)" > "$RUN/server.ready.tmp" && mv "$RUN/server.ready.tmp" "$RUN/server.ready"
    echo "server: ready on $(hostname); waiting for $RUN/client.done" | tee -a "$RUN/logs/run.txt"
    until [[ -e "$RUN/client.done" ]]; do sleep 30; done
    VERL_STATUS="$(cat "$RUN/client.done")"
    echo "server: client done (status $VERL_STATUS)" | tee -a "$RUN/logs/run.txt"
    finish
    ;;
  client)
    connect_to_server
    run_verl
    echo "$VERL_STATUS" > "$RUN/client.done"
    exit "$VERL_STATUS"
    ;;
esac
