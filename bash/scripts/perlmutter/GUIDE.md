# Running the TVCache Terminal-Bench pilot on Perlmutter

> **Newer than this guide:** the code now comes from git with `bootstrap.sh`
> instead of `stage.sh`, and runs start with `run.sh <preset>` (see
> `README.md`). The account setup, podman probe, veRL environment and FAQ
> below still apply as written.

A step-by-step guide, from a laptop with the repos checked out to a finished
training run with a tvverify report. `README.md` in this directory is the
short reference; `ISSUES.md` has the full history of what broke during
bring-up. Everything here was done for real on 2026-11-02/03.

What runs: GRPO training with veRL 0.9.1 on 49 of the 51 easy
Terminal-Bench tasks (`container-manager-public/tasks_easy_network_skip`),
Qwen2.5-1.5B-Instruct, every bash command going through TVCache. Afterwards
`tvverify check-graph` checks the cache the run built. Everything (cache
server, container manager, containers, veRL) runs on a single node.

---

## 0. What you need

- A NERSC account with MFA set up, and a project with GPU hours (we used
  `m4999`). Replace `<user>` and `<project>` below.
- On your laptop, the four repos side by side in one folder, with these
  branches present locally (`stage.sh` copies them from here):

  | Repo | Branch |
  |---|---|
  | `tvcache-workloads` | `perlmutter-pilot` (checked out; its working tree is copied as is) |
  | `container-manager-public` | `podman-hpc` (checked out; the working tree is copied as is) |
  | `tvclient` | `main` and `correctness-fixes` (both exported with `git archive`) |
  | `tvcache-rust` | `main` and `correctness-fixes` (same) |

  Only committed state of `tvclient` / `tvcache-rust` is staged, so
  uncommitted edits there never reach Perlmutter.

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

## 2. Copy the code over

From `tvcache-workloads/bash/scripts/perlmutter` on the laptop:

    REMOTE=perlmutter bash stage.sh

This puts everything under `$SCRATCH/tvcache` on Perlmutter:

    $SCRATCH/tvcache/
      tvcache-workloads/          container-manager-public/
      variants/main/{tvclient,tvcache-rust}     # TVCache main, unmodified
      variants/fixes/{tvclient,tvcache-rust}    # correctness-fixes branches

Rerun `stage.sh` after every local change; it uses `rsync --delete`, so
files you created inside those directories on Perlmutter get removed (runs
live in `$SCRATCH/tvcache/runs`, which is safe).

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

This builds both `tvcache-rust` variants with cargo (`module load
rust/stable`), installs `tvclient`, `tvcache-workloads` and `tvverify` into
the veRL env, and downloads Qwen2.5-1.5B-Instruct to `$SCRATCH/hf/`. It
ends with `python side ok: verl 0.9.1 ...`.

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

    source env.sh
    LOG=$TVROOT/runs/prebuild-login; mkdir -p $LOG
    $CONTAINER_CLI system service --time=0 "unix://$PODMAN_SOCKET" > $LOG/podman.log 2>&1 & SVC=$!
    MANAGER_REPO=$TVROOT/container-manager-public bash ../start_manager.sh > $LOG/manager.log 2>&1 &
    wait_for_port $FLASK_PORT 300 manager
    $VERL_PY prebuild.py $TASKS_ROOT --parallel 4 --out $LOG/prebuild.json
    pkill -u $USER -f server.py; kill $SVC

Expect `49/51 built`; `fix-git` and `accelerate-maximal-square` fail and are
already excluded (`EXCLUDE_TASKS` in `train_pilot.sh`). Builds took a median
of 51 s, max 287 s (`png-generation`). Afterwards `podman-hpc ps -a` should
be empty; if not, `podman-hpc rm -af`.

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

Must match the Mac: contract 11/11 (1 skip), cache PASS, replay PASS, and no
leftover containers or snapshot images. Prefix with `TVCACHE_VARIANT=main`
to smoke-test the base cache.

## 7. Sampling test (GPU, about 1 node-hour)

Rollouts only, no training: checks the whole pipeline end to end.

    cd $SCRATCH/tvcache/tvcache-workloads/bash/scripts/perlmutter
    mkdir -p $SCRATCH/tvcache/runs/slurm
    sbatch -q shared -C gpu -G 2 -c 64 -t 02:00:00 -A <project> -J tv-sample \
      -o $SCRATCH/tvcache/runs/slurm/%x-%j.out --wrap "MODE=sample bash train_pilot.sh"

Ours took about 43 min of rollouts (297 `/start`, 686 `/execute`), then
tvverify: 118/118 nodes replayed, 0 cache vs no-cache mismatches.

## 8. The pilot (GPU, about 2.5 node-hours per variant)

    sbatch -q shared -C gpu -G 2 -c 64 -t 08:00:00 -A <project> -J tv-train-main \
      -o $SCRATCH/tvcache/runs/slurm/%x-%j.out --wrap "TVCACHE_VARIANT=main MODE=train bash train_pilot.sh"

    sbatch -q shared -C gpu -G 2 -c 64 -t 08:00:00 -A <project> -J tv-train-fixes \
      -o $SCRATCH/tvcache/runs/slurm/%x-%j.out --wrap "TVCACHE_VARIANT=fixes MODE=train bash train_pilot.sh"

- `main` is the **base cache** (unmodified TVCache, no trace hooks). Its
  tvverify report is the baseline: what goes wrong with no fixes applied.
- `fixes` runs the `correctness-fixes` branches and also writes a
  per-lookup trace.
- `nocache` runs the same training with **no cache at all**: every command
  executes in the rollout's own container (`TVCACHE_USE_CACHE=0`,
  `DirectExecutor` in `bash/verl/direct.py`). No cache server, sidecar,
  dumps or tvverify. Pair it with `main` or `fixes` for a with/without
  TVCache comparison:

      sbatch -q shared -C gpu -G 2 -c 64 -t 08:00:00 -A <project> -J tv-train-nocache \
        -o $SCRATCH/tvcache/runs/slurm/%x-%j.out --wrap "TVCACHE_VARIANT=nocache MODE=train bash train_pilot.sh"

Use **half a node** (`-G 2 -c 64`, about 120 GB RAM) even though veRL uses
one GPU: a quarter node ran out of memory (FAQ). `sbatch` on the `shared`
QoS keeps running if your SSH drops.

**While it runs:**

    squeue --me
    tail -f $SCRATCH/tvcache/runs/slurm/tv-train-*-<jobid>.out
    RUN=$(ls -td $SCRATCH/tvcache/runs/*-train-* | head -1)
    tail -3 $RUN/logs/memory.csv            # memory once a minute; should stay well under 120 GB
    grep -c "POST /execute" $RUN/logs/manager.log

**Resume** after a timeout: resubmit with the same `RUN=<path>`. Checkpoints
are saved every 10 steps; the cache restarts empty.

## 9. What a run leaves in `$RUN`

| Path | What |
|---|---|
| `report.md` | tvverify's combined report: bugs, noise, inconclusive, per task. Start here |
| `logs/run.txt` | summary: variant, mode, veRL exit code, leftover snapshot images |
| `logs/verl.log`, `cache.log`, `manager.log`, `podman.log`, `sidecar.log` | per-process logs |
| `logs/memory.csv` | cgroup memory, podman store size, containers, images per minute |
| `dumps/dump_*.json` | full cache dumps every 10 min and at the end |
| `check/dump_*.json`, `.log` | `check-graph` per dump; only the last one checks snapshots (C2a) |
| `trace/` | per-lookup trace (`fixes` only) |
| `checkpoints/` | veRL checkpoints |

Copy results to the laptop:

    rsync -az --exclude checkpoints perlmutter:'$SCRATCH/tvcache/runs/<run>' ./runs/

## 10. Configuration

All set in `env.sh` or at the top of `train_pilot.sh`; override on the
command line (`EPOCHS=1 MODE=train bash train_pilot.sh`).

| Variable | Default | Meaning |
|---|---|---|
| `TVCACHE_VARIANT` | `fixes` | `main` (base cache), `fixes`, or `nocache` (no cache: the baseline) |
| `MODE` | `train` | `sample` (val only) or `train` |
| `EPOCHS` / `ROLLOUTS` / `BATCH` | 5 / 4 / 4 | GRPO epochs, group size, tasks per step |
| `N_GPUS` | 1 | GPUs veRL uses |
| `SNAPSHOT_THRESHOLD` | `1.0` | fixed snapshot threshold in seconds; empty = tvclient's adaptive policy (rarely snapshots) |
| `CHECK_NODES` | 20 | tvverify nodes replayed per task |
| `DUMP_INTERVAL` | 600 | seconds between cache dumps |
| `EXCLUDE_TASKS` | `fix-git accelerate-maximal-square` | tasks left out of the dataset |
| `MODEL` | `$SCRATCH/hf/Qwen2.5-1.5B-Instruct` | model path |
| `TASKS_ROOT` | `.../tasks_easy_network_skip` | task directory |
| `RUN` | `runs/<time>-<mode>-<variant>` | output directory; reuse to resume |
| `VERL_PY` | `$TVROOT/verl-env/bin/python` | Python with veRL, vLLM, tvclient, tvverify |
| `CONTAINER_CLI` / `COMPOSE_CMD` / `DOCKER_HOST` | `podman-hpc` / `~/bin/docker-compose` / podman socket | container runtime used by the manager |
| `TVCACHE_PORT` / `FLASK_PORT` | 7878 / 9901 | cache server / manager ports (TVCache `main` is hard-wired to 7878) |

## 11. Budget

| Step | Cost |
|---|---|
| Setup, probe, prebuild, smoke test (login node) | free |
| Compute-node probe | ~0.05 node-hours |
| Sampling test, half node | ~0.5–1 |
| One pilot variant, half node, 4.5–5 h | ~2.5 |

Both variants plus the sampling test fit in about 6–7 node-hours. Check
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
Fixed in the `podman-hpc` branch of container-manager-public; make sure
that branch is the one checked out when you run `stage.sh`.

**Every `/start` returns 500 with `UnboundLocalError: ... 'e'`.** Old
manager without the `last_error` fix; same answer: use `podman-hpc`.

**The manager log is missing the real error.** Its output was
block-buffered and lost when it was killed. `start_manager.sh` sets
`PYTHONUNBUFFERED=1`; use it rather than starting `server.py` by hand.

**Dozens of containers left running after failed starts.** The manager's
`/start` retries do not clean up (known, not fixed). `podman-hpc rm -af`.

**`fix-git` or `accelerate-maximal-square` does not build.** Expected.
`fix-git` clones a GitHub repo that is gone (404, fails everywhere);
`accelerate-maximal-square`'s base image has uids beyond rootless podman's
range. Both are in `EXCLUDE_TASKS`.

**`RuntimeError: tvcache-rust not found` from tvverify.** It looks next to
`tvcache-workloads`; on Perlmutter it is under `variants/`. `env.sh` sets
`TVCACHE_RUST_DIR` and loads `rust/stable`; source it.

**Contract check fails: restored cwd is `/app` instead of `/app/work`.**
The tmux-resurrect plugin (`container-manager-public/rlhf_server/tmux-res`)
was not staged, so cwd and exported variables are never saved. Don't
exclude `tmux-res` in `stage.sh`. This fails silently in the manager; only
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
memory, and every task image plus snapshots lands there. Use half a node
(`-G 2 -c 64`). Watch `logs/memory.csv`; if it still climbs toward the
limit, raise `SNAPSHOT_THRESHOLD` (fewer snapshot images) or run fewer
tasks.

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
