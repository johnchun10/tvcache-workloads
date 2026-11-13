# Terminal-Bench runs on NERSC Perlmutter

GRPO training with veRL on Terminal-Bench tasks, with every bash command going
through TVCache, through TVCache with graph contraction, or through no cache at
all, then `tvverify check-graph` on the cache the run built. Everything (cache
server, container manager, containers, veRL) runs on one GPU node.

`GUIDE.md` has the original step-by-step bring-up with an FAQ, and `ISSUES.md`
what broke along the way. This file is the short reference.

Perlmutter has no Docker daemon, so the container manager drives
**podman-hpc** (`CONTAINER_CLI`, `COMPOSE_CMD`, `DOCKER_HOST`; see `env.sh`).

## Setting up

All on a login node (free). `$TVROOT` is `$SCRATCH/tvcache` unless you set it.

    git clone -b contraction-runs https://github.com/johnchun10/tvcache-workloads.git $SCRATCH/tvcache/tvcache-workloads
    cd $SCRATCH/tvcache/tvcache-workloads/bash/scripts/perlmutter
    bash bootstrap.sh     # clones the other three repositories at the right branches
    bash setup.sh         # builds the cache server, installs the Python side, downloads the models

`bootstrap.sh` lays out:

| Directory | Repository and branch |
|---|---|
| `tvcache-workloads/` | this one |
| `container-manager-public/` | `main` (includes podman-hpc support) |
| `variants/combined/tvclient/`, `variants/combined/tvcache-rust/` | `contraction-plus-fixes`: graph contraction plus the correctness fixes |

Run both scripts again to pick up new commits. First time on a new account,
also do the podman probe and the veRL environment from `GUIDE.md` (steps 2–4).

## Running

    bash run.sh <preset> [flags]

| Preset | Tasks | Defaults |
|---|---|---|
| `shakedown` | 1 easy + 1 medium | contraction at 0.9 with `minilm`; 2 epochs, 4 rollouts per task, 1 task per batch |
| `main20-easy-med` | 10 easy + 10 medium | contraction at 0.9 with `minilm`; 5 epochs, 8 rollouts per task, 4 tasks per batch |

| Flag | Meaning |
|---|---|
| `--cache none\|tvcache\|contraction` | no cache / TVCache / TVCache with graph contraction |
| `--threshold X` | contraction: similarity threshold (the sweep: 0.8, 0.9, 0.95) |
| `--embedder minilm\|jina-code` | contraction: `all-MiniLM-L6-v2` or `jina-embeddings-v2-base-code` |
| `--budget N` | contraction: live sandboxes all contraction jobs share (32) |
| `--epochs N`, `--rollouts N`, `--batch N` | epochs, rollouts per task, tasks per training step |
| `--max-tokens N` | tokens per rollout after the prompt (2048) |
| `--verify`, `--no-verify` | run tvverify on the cache afterwards, or skip it |
| `--tasks "a b c"` | replace the preset's task list |
| `--mode sample` | rollouts only, no training |
| `--prebuild` | only build the task images through the manager, then exit |
| `--dry-run` | print the resolved settings and exit |

The order for a new set of tasks:

    bash run.sh shakedown --prebuild        # login node: do the images build?
    # then, in a GPU allocation:
    sbatch -q shared -C gpu -G 2 -c 64 -t 02:00:00 -A <project> \
      -o $SCRATCH/tvcache/runs/slurm/%x-%j.out --wrap "bash run.sh shakedown"

and the same two commands with `main20-easy-med` (give it `-t 08:00:00`). The
three-way comparison is three runs of the same preset:

    bash run.sh main20-easy-med --cache none
    bash run.sh main20-easy-med --cache tvcache
    bash run.sh main20-easy-med --cache contraction --threshold 0.9 --embedder minilm

Half a node (`-G 2 -c 64`) for its RAM, about 120 GB; veRL uses one GPU
(`N_GPUS=1`). On GPU nodes podman's image store is `/tmp`, a RAM disk, and a
quarter node ran out of memory (ISSUES.md #16).

## What contraction adds to a run

1. **State check** (`state.json`, `logs/state_check.log`). Contraction merges
   two cached commands when their sandboxes compare equal: same files under
   `/app`, `/root`, `/etc` and `/tmp`, same exported variables, same current
   directory, same installed packages. Before training, three sandboxes in the
   same state are compared per task; paths that differ anyway are noise and
   are skipped for that task. A task whose sandboxes cannot be made to compare
   equal is trained on but never contracted.
2. **Coordinator** (`logs/contraction.log`). Contraction may only work on a
   task while none of its rollouts is running. Rollouts tell the coordinator
   when they start and finish with a task; it contracts the tasks that are
   free, sharing `--budget` sandboxes among them.
3. **Embeddings.** Every executed command is embedded on the CPU and stored
   with its cache entry; the server pairs similar commands for comparison.

## What a run leaves in `$RUN`

| Path | What |
|---|---|
| `logs/run.txt` | the settings, then the outcome: veRL's exit code, peak memory, contraction totals, leftover snapshots |
| `config.env` | every setting the run used |
| `tasks/`, `tasks.parquet` | the task directories and dataset the run used |
| `logs/memory.csv` | once a minute: the job's memory, the image store's size, containers and images |
| `logs/containers.csv` | once a minute: memory per running container (where podman reports it) |
| `contraction/events.jsonl`, `summary.json` | every pause, resume, listing and merge with a timestamp; totals and counts over time |
| `state.json` | per task: comparable or not, and the paths skipped as noise |
| `dumps/dump_*.json` | DUMP_ALL every `DUMP_INTERVAL` s (default 600) and at the end |
| `report.md`, `check/` | tvverify's report and its per-dump results (the last dump also checks snapshots) |
| `trace/` | per-lookup trace, when tvverify is on |
| `logs/` | veRL, cache server, manager, podman, sidecar, contraction |

Compare runs with

    $VERL_PY compare_runs.py $TVROOT/runs/<a> $TVROOT/runs/<b> ... --plot charts/

and read one task's cached commands, merged ones marked `=`, with
`$VERL_PY dump_tree.py $TVROOT/runs/<a> hello-world`.

## Other settings

Everything `run.sh` does not cover still comes from the environment (see
`env.sh` and the top of `train_pilot.sh`): `MODEL`, `N_GPUS`,
`SNAPSHOT_THRESHOLD` (fixed 1 s; empty for tvclient's adaptive policy),
`CHECK_NODES`, `DUMP_INTERVAL`, `MAX_TURNS`, `RUN` (reuse to resume from a
checkpoint; the cache restarts empty), `TVCACHE_VARIANT` (`combined`, or
`main` / `fixes` if you cloned those builds into `variants/`).

## veRL environment

`VERL_PY` must be a Python with veRL 0.9.1 (the version the integration was
written against), vLLM and torch. A conda env from NERSC's `pytorch` module
plus `pip install verl[vllm]==0.9.1` works. `setup.sh` then installs tvclient,
tvcache-workloads, tvverify and sentence-transformers into it.
