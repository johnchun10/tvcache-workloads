# Terminal-Bench pilot on NERSC Perlmutter

GRPO training with veRL on the 51 easy Terminal-Bench tasks the TVCache paper
used (`container-manager-public/tasks_easy_network_skip`), with every bash
command going through TVCache, then `tvverify check-graph` on the cache it
built. Pilot settings: Qwen2.5-1.5B-Instruct, 4 rollouts per task, batch
4, 5 epochs (the paper: Qwen3-4B, 8 rollouts, 10 epochs).

Step-by-step instructions with every command, configuration and an FAQ are
in `GUIDE.md`.

Perlmutter has no Docker daemon, so the container manager drives
**podman-hpc** (`CONTAINER_CLI`, `COMPOSE_CMD`, `DOCKER_HOST`; see `env.sh`).

## Order

| Step | Where | Command |
|---|---|---|
| 0. Log in | Mac | `sshproxy -u <user>` (in a real terminal) |
| 1. Stage the code | Mac | `REMOTE=<user>@perlmutter.nersc.gov bash stage.sh` |
| 2. Probe podman (**gate**) | login node first (free), then a short shared CPU job | `bash probe_podman.sh`; if it passes, confirm under `salloc -q shared_interactive -C cpu -c 16 -t 00:20:00 -A <project>` |
| 3. Build servers, install Python side | login node | `bash setup.sh` |
| 4. Prebuild the 51 task images | login node (NERSC allows `podman-hpc build` there) | start the manager (as in `train_pilot.sh` step 1), then `$VERL_PY prebuild.py $TASKS_ROOT` |
| 5. Smoke test | compute node | from `tvcache-workloads`: `$VERL_PY -m tvverify check --adapter tvcache_workloads.bash.verify:HelloWorldAdapter --rollouts 5`; must match the Mac run (contract 11/11, cache PASS, replay PASS) |
| 6. Sampling test | GPU node | `MODE=sample bash train_pilot.sh` (about 20 min) |
| 7. Pilot | GPU node | `MODE=train bash train_pilot.sh`, once per variant (below) |

Step 2 is the gate: if the **docker API socket** or **SDK** items fail, stop.
The manager's Docker SDK calls and `docker-compose` both need that socket.

GPU jobs for steps 6–7 go in with `sbatch` on the `shared` QoS (it keeps
running if SSH drops), from this directory:

    sbatch -q shared -C gpu -G 2 -c 64 -t 08:00:00 -A <project> \
      -o $SCRATCH/tvcache/runs/slurm/%x-%j.out --wrap "MODE=train bash train_pilot.sh"

Half a node (`-G 2 -c 64`) for its RAM, about 120 GB; veRL uses one GPU
(`N_GPUS=1`). On GPU nodes podman's image store is `/tmp`, a RAM disk, and a
quarter node (`-G 1 -c 32`, about 60 GB) ran out of memory (ISSUES.md #16).
`$RUN/logs/memory.csv` logs the job's memory once a minute.

Snapshots: `train_pilot.sh` sets a fixed 1 s threshold
(`SNAPSHOT_THRESHOLD`; empty for tvclient's adaptive policy), so misses
often go through the restore path.

## Budget

The project has about 10 node-hours. Probe, prebuild and smoke test run on
the login node (free). Sampling test: rollouts took about 43 min. One pilot
variant: about 4.5–5 h on half a node, about 2.5 node-hours (more if the
snapshot threshold slows rollouts), so about 5 for both. Checkpoints are
saved every 10 steps; rerun with the same `RUN=` to resume (the cache
restarts empty, so later steps begin cold). Check usage with
`sshare -A <project>` or Iris.

## veRL environment

`VERL_PY` must be a Python with veRL 0.9.1 (the version the integration was
written against), vLLM and torch. A prebuilt veRL container image, run through
podman-hpc, is the least painful option; a conda env from NERSC's
`pytorch` module plus `pip install verl[vllm]==0.9.1` also works. `setup.sh`
then installs tvclient, tvcache-workloads and tvverify into it.

## Variants

`TVCACHE_VARIANT=fixes` (default) runs the `correctness-fixes` branches of
tvclient and tvcache-rust, with a per-lookup trace (`$RUN/trace`).
`TVCACHE_VARIANT=main` runs unmodified TVCache `main` (no trace: it has no
observer hooks). Running both shows whether bugs B1–B3 hit during real
training, and whether the fixes hold.
`TVCACHE_VARIANT=nocache` runs the same training with no cache at all
(`TVCACHE_USE_CACHE=0`; every command runs in the rollout's own container,
no cache server, sidecar or tvverify): the baseline for a with/without
TVCache comparison against `main` or `fixes`.

## What a run leaves in `$RUN`

| Path | What |
|---|---|
| `report.md` | tvverify's combined report: bugs, noise, inconclusive, per task |
| `check/dump_*.json`, `.log` | check-graph results per dump (the last dump also checks snapshots, C2a) |
| `dumps/dump_*.json` | DUMP_ALL every `DUMP_INTERVAL` s (default 600) and at the end |
| `trace/` | TraceRecorder JSONL (fixes variant only) |
| `logs/` | veRL, cache server, manager, podman, sidecar; `run.txt` has the summary and leftover snapshot count |

Knobs: `MODE`, `EPOCHS`, `N_GPUS`, `ROLLOUTS`, `BATCH`, `CHECK_NODES`,
`DUMP_INTERVAL`, `MODEL`, `TASKS_ROOT`, `TVCACHE_VARIANT`.
