# Running the TVCache Terminal-Bench pilot on Perlmutter

A step-by-step guide, from a new NERSC account to a finished training run
with a tvverify report. `README.md` in this directory is the short
reference; `ISSUES.md` has the history of what broke during bring-up.
Sections 1 to 4 and 6 were done for real on 2026-11-02/03. The flow in
sections 5, 7 and 8 (`run.sh`, the two-node job, graph contraction) was
rehearsed off Perlmutter and had not yet run there in its current form when
this was written; the shakedown in section 7 is its first real test.

What runs: GRPO training with veRL 0.9.1 on Terminal-Bench tasks,
Qwen2.5-1.5B-Instruct, with every bash command going through TVCache,
TVCache with graph contraction, or no cache. Afterwards `tvverify
check-graph` checks the cache the run built, and the run's `report.md` has
the outcome, the testing plan's metrics and tvverify's findings in one file. A run is one job on two nodes: a CPU node for the containers,
cache server, container manager, fork bank and contraction, and a GPU node
for veRL. It can also run on a single GPU node.

---

## 0. What you need

- A NERSC account with MFA set up, and a project with GPU hours (we used
  `m4999`). Replace `<user>` and `<project>` below.
- Nothing on your laptop: the code is cloned from git on Perlmutter itself.
  `bootstrap.sh` clones `main` of the four `johnchun10` forks
  (`tvcache-workloads`, `container-manager-public`, `tvclient`,
  `tvcache-rust`). No other branch is needed.

## 1. Log in from your laptop

Install NERSC's `sshproxy` client, then in a **real terminal** (it prompts
for password + OTP and needs a TTY; it does not work through Claude Code's
`!`):

    sshproxy -u <user>          # writes ~/.ssh/nersc, valid 24 h

Add an SSH host so every later command reuses one connection (one MFA prompt
per 24 h):

    # ~/.ssh/config
    Host perlmutter
      HostName perlmutter.nersc.gov
      User <user>
      IdentityFile ~/.ssh/nersc
      IdentitiesOnly yes
      ControlMaster auto
      ControlPath ~/.ssh/cm-%r@%h:%p
      ControlPersist 1h

Check: `ssh perlmutter 'echo $SCRATCH'` prints `/pscratch/sd/<x>/<user>`.

## 2. Get the code

On a Perlmutter login node:

    git clone https://github.com/johnchun10/tvcache-workloads.git $SCRATCH/tvcache/tvcache-workloads
    bash $SCRATCH/tvcache/tvcache-workloads/bash/scripts/perlmutter/bootstrap.sh

This puts everything under `$SCRATCH/tvcache`:

    $SCRATCH/tvcache/
      tvcache-workloads/          container-manager-public/
      variants/combined/{tvclient,tvcache-rust}   # main: contraction, the correctness fixes, the fork bank

Run `bootstrap.sh` again to pick up new commits, then `setup.sh` again.
Runs live in `$SCRATCH/tvcache/runs`.

## 3. One-time setup on a login node

Login nodes are free; do all of this there (`ssh perlmutter`).

**uv** (the container manager uses it):

    curl -LsSf https://astral.sh/uv/install.sh | sh     # installs to ~/.local/bin

**docker-compose** (podman-hpc has no compose of its own; this talks to
podman's Docker API socket):

    mkdir -p ~/bin
    curl -sSfL -o ~/bin/docker-compose \
      https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64
    chmod +x ~/bin/docker-compose

**veRL environment** (about 15 min; lives on scratch):

    module load conda
    conda create -q -y -p $SCRATCH/tvcache/verl-env python=3.12
    $SCRATCH/tvcache/verl-env/bin/python -m pip install --upgrade pip
    $SCRATCH/tvcache/verl-env/bin/python -m pip install "verl[vllm]==0.9.1"

We got verl 0.9.1, vLLM 0.24, torch 2.11 (CUDA 13); the GPU driver is 580.
There is no `flash-attn` wheel for that combination; the scripts use SDPA
instead (see FAQ).

**Servers, Python packages, model:**

    cd $SCRATCH/tvcache/tvcache-workloads/bash/scripts/perlmutter
    bash setup.sh

This builds the cache server with cargo (`module load rust/stable`),
installs `tvclient`, `tvcache-workloads`, `tvverify` and
`sentence-transformers` into the veRL env, downloads Qwen2.5-1.5B-Instruct
to `$SCRATCH/hf/`, and loads both embedding models once. Check for
`python side ok: verl 0.9.1 ...` and `embedder ... ok` for each model.

## 4. Probe podman (the gate)

    bash probe_podman.sh

Every line should say PASS, and in particular **docker API socket** and
**SDK lists the probe container**: the manager's Docker SDK calls and
docker-compose both need that socket. If those fail, stop: nothing after
this works. Optionally confirm on a compute node (about 0.05 node-hours):

    podman-hpc migrate docker.io/library/ubuntu:22.04
    srun -q shared_interactive -C cpu -c 16 -t 00:20:00 -A <project> bash probe_compute.sh

## 5. Prebuild the task images (login node)

Finds tasks whose image does not build before you pay for GPU time.

    cd $SCRATCH/tvcache/tvcache-workloads/bash/scripts/perlmutter
    bash run.sh shakedown --prebuild            # or main20-easy-med

It starts podman's service and the manager, builds every task of the preset
once, prints `ok` or `FAIL` per task, and stops. Replace a task that fails
with `--tasks "..."` or by editing the preset. Afterwards `podman-hpc ps -a`
should be empty; if not, `podman-hpc rm -af`.

The original pilot built 49 of the 51 easy tasks; `fix-git` and
`accelerate-maximal-square` fail on Perlmutter. Builds took a median of 51 s.

Note: images are **node-local**, so a GPU job still builds each task once on
its own node. Prebuilding is for catching failures, not for warming the
compute node.

## 6. Smoke test (login node)

    source env.sh
    LOG=$TVROOT/runs/smoke-login; rm -rf $LOG; mkdir -p $LOG
    $CONTAINER_CLI system service --time=0 "unix://$PODMAN_SOCKET" > $LOG/podman.log 2>&1 & SVC=$!
    "$(variant_dir)/tvcache-rust/target/release/tvcache" --port=$TVCACHE_PORT > $LOG/cache.log 2>&1 & CACHE=$!
    MANAGER_REPO=$TVROOT/container-manager-public bash ../start_manager.sh > $LOG/manager.log 2>&1 &
    wait_for_port $TVCACHE_PORT 60 cache && wait_for_port $FLASK_PORT 300 manager
    (cd $TVROOT/tvcache-workloads && \
      $VERL_PY -m tvverify check --adapter tvcache_workloads.bash.verify:HelloWorldAdapter --rollouts 5)
    pkill -u $USER -f server.py; kill $CACHE $SVC

Expect contract 11/11 (1 skip), cache PASS, replay PASS, and no leftover
containers or snapshot images.

## 7. Shakedown

One easy and one medium task, small settings, contraction and the fork bank
on: checks that every part runs before the long runs. From a login node:

    cd $SCRATCH/tvcache/tvcache-workloads/bash/scripts/perlmutter
    sbatch -o $SCRATCH/tvcache/runs/slurm/%x-%j.out probe_hetjob.sbatch   # optional, 10 min: do the two nodes start and reach each other?
    bash submit_hetjob.sh shakedown

`submit_hetjob.sh` prints the job id and the run directory. The account and
QOS are the `#SBATCH -A ... -q ...` lines in `hetjob.sbatch`; edit them for
another project. Afterwards read, in the run directory:

- `report.md`: the one report. Settings, veRL's exit code, peak memory and
  leftovers; reward and step time per epoch; hit rates, graph sizes and
  contraction rounds per epoch; tvverify's findings;
- `logs/run.txt`, `server.out`, `client.out`: the run's log and each node's
  output, if something failed.

The same shakedown on one GPU node, if the two-node job does not start:

    sbatch -q shared -C gpu -G 2 -c 64 -t 02:00:00 -A <project> -J tv-shakedown \
      -o $SCRATCH/tvcache/runs/slurm/%x-%j.out --wrap "bash run.sh shakedown"

## 8. The evaluation runs

The same preset three ways, in one job, each mode with an empty cache:

    CACHES="none tvcache contraction" TIME=<hh:mm:ss> bash submit_hetjob.sh main20-easy-med

or one job per mode (`bash submit_hetjob.sh main20-easy-med --cache tvcache`).
Then the contraction settings:

    bash submit_hetjob.sh main20-easy-med --threshold 0.95 --embedder jina-code

- `none` runs every command in the rollout's own container: no cache server,
  dumps or tvverify; its `report.md` has only the outcome and the training
  part. The baseline for reward and step time.
- `tvcache` is the cache without contraction. The baseline for the testing
  plan's ratios.
- `contraction` adds embeddings, the per-task state check and the contraction
  coordinator. Sweep `--threshold` (0.8, 0.9, 0.95) and `--embedder` (`minilm`,
  `jina-code`).

With both `tvcache` and `contraction` in `CACHES`, the job writes the testing
plan's ratio table to `<run>-compare.md`.

Each node gets a whole node's memory. On a single GPU node the 49-task pilot
needed the whole node (`-N 1 -G 4 -c 128 --mem=0`); a quarter and a half node
ran out of memory (FAQ).

**While it runs:**

    squeue --me
    tail -f $SCRATCH/tvcache/runs/slurm/tv-het-*-<jobid>.out
    RUN=$(ls -td $SCRATCH/tvcache/runs/*-main20-* | head -1)
    tail -3 $RUN/logs/memory.csv            # memory once a minute
    tail -3 $RUN/contraction/events.jsonl   # contraction runs only

**Long runs:** `SPLIT=30 TOTAL=<steps> bash submit_hetjob.sh main20-easy-med`
trains in parts of 30 steps, one job per part, each resuming from the previous
part's checkpoint. Each part has its own run directory and starts with an
empty cache.

## 9. What a run leaves in `$RUN`

`README.md` has the full table. Read `report.md`: it has the settings and
outcome, training per epoch, the testing plan's metrics and tvverify's
findings. Put runs side by side, including one without a cache, with

    $VERL_PY compare_runs.py $TVROOT/runs/<a> $TVROOT/runs/<b> ... --plot charts/

Copy results to a laptop:

    rsync -az --exclude checkpoints perlmutter:'$SCRATCH/tvcache/runs/<run>' ./runs/

## 10. Configuration

`run.sh` flags cover the settings that change between runs (`README.md`
lists them; `bash run.sh <preset> --dry-run` prints what a run would use).
Everything else comes from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `TVCACHE_VARIANT` | `combined` | which build under `variants/` to use (`bootstrap.sh` clones only `combined`) |
| `FORK_BANK` | 1 | prewarming; 0 turns it off, and the cache metrics in `report.md` are then not split by epoch |
| `TOP_K` | 10 | fork bank: most used snapshots kept restored |
| `SEED` | 42 | the dataloader's shuffle |
| `N_GPUS` | 1 | GPUs veRL uses |
| `SNAPSHOT_THRESHOLD` | `1.0` | fixed snapshot threshold in seconds; empty = tvclient's adaptive policy (rarely snapshots) |
| `CHECK_NODES` | 20 | tvverify nodes replayed per task |
| `DUMP_INTERVAL` | 600 | seconds between cache dumps |
| `MAX_TURNS` | 30 | tool calls per rollout |
| `MODEL` | `$SCRATCH/hf/Qwen2.5-1.5B-Instruct` | model path |
| `RUN` | `runs/<time>-<preset>-<cache mode>` | output directory; reuse to resume |
| `VERL_PY` | `$TVROOT/verl-env/bin/python` | Python with veRL, vLLM, tvclient, tvverify |
| `CONTAINER_CLI` / `COMPOSE_CMD` / `DOCKER_HOST` | `podman-hpc` / `~/bin/docker-compose` / podman socket | container runtime used by the manager |
| `TVCACHE_PORT` / `FLASK_PORT` / `TVCACHE_CONTRACTION_PORT` / `TVCACHE_CONTROL_PORT` | 7878 / 9901 / 7879 / 7880 | cache server, manager, contraction coordinator, fork bank |

## 11. Budget

| Step | Cost |
|---|---|
| Setup, probe, prebuild, smoke test (login node) | free |
| Compute-node probe | ~0.05 node-hours |
| Sampling test, half node | ~0.5–1 |
| One run of the original 49-task pilot, half node, 4.5–5 h | ~2.5 |
| Two-node shakedown, 2 tasks (an earlier version of the job) | ~0.8 |

A `main20-easy-med` run has not been timed yet. The two-node job is charged
for both nodes; it gives the GPU node back as soon as training ends. Check
usage with `sshare -A <project>` or Iris.

---

## FAQ / troubleshooting

**`sshproxy.sh: command not found`.** The current client is a binary named
`sshproxy`: `sshproxy -u <user>`.

**`sshproxy`: `Device not configured (os error 6)`.** You ran it without a
TTY (e.g. through Claude Code's `!`). Run it in a normal terminal.

**`docker-compose build`: `crun: create /sys/fs/cgroup/docker: Permission denied`.**
Compose v2 builds through a BuildKit container, which rootless podman-hpc
cannot run. `env.sh` sets `DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0`;
make sure it is sourced. If a `buildx_buildkit_default` container is left
over: `podman-hpc rm -f buildx_buildkit_default`.

**Every `/start` fails with `lchown ... invalid argument`.** Files copied
into containers carried your host uid, which rootless podman cannot map.
Fixed in `main` of our container-manager-public fork, which `bootstrap.sh`
clones.

**Every `/start` returns 500 with `UnboundLocalError: ... 'e'`.** Old
manager without the `last_error` fix; same answer: use the fork's `main`.

**The manager log is missing the real error.** Its output was
block-buffered and lost when it was killed. `start_manager.sh` sets
`PYTHONUNBUFFERED=1`; use it rather than starting `server.py` by hand.

**Dozens of containers left running after failed starts.** The manager's
`/start` retries do not clean up (known, not fixed). `podman-hpc rm -af`.

**`fix-git` or `accelerate-maximal-square` does not build.** Expected.
`fix-git` clones a GitHub repo that is gone (404, fails everywhere);
`accelerate-maximal-square`'s base image has uids beyond rootless podman's
range. Neither is in the presets.

**`RuntimeError: tvcache-rust not found` from tvverify.** It looks next to
`tvcache-workloads`; on Perlmutter it is under `variants/`. `env.sh` sets
`TVCACHE_RUST_DIR` and loads `rust/stable`; source it.

**Contract check fails: restored cwd is `/app` instead of `/app/work`.**
The tmux-resurrect plugin (`container-manager-public/rlhf_server/tmux-res`)
is missing from the manager checkout, so cwd and exported variables are
never saved. This fails silently in the manager; only
tvverify's contract check catches it.

**Job dies early: `container manager did not open port 9901`, manager log
says `Could not acquire lock for ~/.cache/uv ... os error 524`.**
`$HOME` on compute nodes does not support `flock`. `env.sh` moves uv, cargo,
Hugging Face, vLLM, Triton, Inductor and CUDA caches to scratch. If a new
tool hits error 524, point its cache at `$TVROOT/cache` too.
`probe_locks.sh` shows which filesystems support locks.

**`fcntl.flock ... OSError: [Errno 524]` while veRL loads the dataset.**
Same cause (HF `datasets` cache); same fix (`HF_HOME` in `env.sh`).

**`FlashAttention2 has been toggled on, but ... not installed`.** veRL
defaults to flash-attn, which has no wheel for torch 2.11 + CUDA 13.
`train_pilot.sh` overrides `attn_implementation=sdpa` and turns off
`use_remove_padding` (it needs flash-attn's varlen kernels). vLLM is not
affected.

**Slurm `OUT_OF_MEMORY` after ~50 min (MaxRSS ~57 GB).** On GPU nodes
podman's image store is `/tmp`, a RAM disk that counts against the job's
memory, and every task image plus snapshots lands there. A half node
(`-G 2 -c 64`, 112 GB) ran out too in training (ISSUES.md #17). Use the
two-node job, which keeps the containers on the CPU node, or a whole GPU
node. Watch `logs/memory.csv`; if it still climbs toward the limit, raise
`SNAPSHOT_THRESHOLD` (fewer snapshot images) or run fewer tasks.

**`podman-hpc --version` fails.** It has no such flag; use
`podman-hpc version`.

**Do I need to rebuild images on the GPU node?** Yes, automatically:
images are node-local (`/images` NVMe on login nodes, `/tmp` RAM on compute
nodes), and the manager runs a compose build on every `/start`, so a GPU
job pays each task's first build on its own node. `podman-hpc migrate
<image>` makes a login-built image visible on compute nodes, but we have
not tried using that to skip those builds.

**tvverify flags `ls -l /tmp` or `wget`/`apt-get` commands.** Seen in the
sampling test (`vimscript-vim-quine`, `build-stp`). Most likely
nondeterminism (timestamps, network), not a cache bug; check the node in
both dumps before reporting it.

**Snapshots almost never happen.** tvclient's adaptive policy took 4
snapshots in 686 commands. The pilot sets a fixed 1 s threshold so the
restore path gets exercised; set `SNAPSHOT_THRESHOLD=` (empty) for the
adaptive policy.

**Snapshot images left over after a run.** `logs/run.txt` reports the count.
Remove them with
`podman-hpc rmi $(podman-hpc images --filter label=com.docker.compose.project=tvcache-snapshot -q)`.
