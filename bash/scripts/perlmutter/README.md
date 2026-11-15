# Terminal-Bench runs on NERSC Perlmutter

GRPO training with veRL on Terminal-Bench tasks, with every bash command going
through TVCache, through TVCache with graph contraction, or through no cache at
all, then `tvverify check-graph` on the cache the run built and the testing
plan's metrics in `metrics.md`.

A run is normally two nodes in one job: a CPU node for the containers, their
snapshots, the cache server, the container manager, the fork bank and graph
contraction, and a GPU node for veRL. Everything can also run on one GPU node.

`GUIDE.md` has the original step-by-step bring-up with an FAQ, and `ISSUES.md`
what broke along the way. This file is the short reference.

Perlmutter has no Docker daemon, so the container manager drives
**podman-hpc** (`CONTAINER_CLI`, `COMPOSE_CMD`, `DOCKER_HOST`; see `env.sh`).

## Setting up

All on a login node (free). `$TVROOT` is `$SCRATCH/tvcache` unless you set it.

    git clone https://github.com/johnchun10/tvcache-workloads.git $SCRATCH/tvcache/tvcache-workloads
    cd $SCRATCH/tvcache/tvcache-workloads/bash/scripts/perlmutter
    bash bootstrap.sh     # clones or updates all four repositories, each at main of our forks
    bash setup.sh         # builds the cache server, installs the Python side, downloads the models

`bootstrap.sh` lays out:

| Directory | Repository (`main` of `github.com/johnchun10/...`) |
|---|---|
| `tvcache-workloads/` | this one |
| `container-manager-public/` | podman-hpc support, cold snapshots, snapshot sizes |
| `variants/combined/tvclient/`, `variants/combined/tvcache-rust/` | graph contraction, the correctness fixes, the fork bank |

Everything a run needs is in `main`; no other branch is used. Run both scripts
again to pick up new commits (a clone still on an older branch is moved to `main`). First time on a new account,
also do the podman probe and the veRL environment from `GUIDE.md` (steps 2–4).

## Running

    bash submit_hetjob.sh <preset> [flags]     # login node: queues the two-node job

| Preset | Tasks | Defaults |
|---|---|---|
| `shakedown` | 1 easy + 1 medium | contraction at 0.9 with `minilm`; 2 epochs, 4 rollouts per task, 1 task per batch |
| `main20-easy-med` | 10 easy + 10 medium | contraction at 0.9 with `minilm`; 5 epochs, 8 rollouts per task, 4 tasks per batch |

The order for a new set of tasks, all from a login node:

    bash run.sh shakedown --prebuild           # do the task images build? (free)
    bash submit_hetjob.sh shakedown            # every part runs once, small
    CACHES="none tvcache contraction" TIME=<hh:mm:ss> bash submit_hetjob.sh main20-easy-med

`CACHES` runs the preset once per cache mode, back to back in one job, each in
its own `$RUN-<mode>` with an empty cache. With both `tvcache` and
`contraction` in the list it also writes the testing plan's ratio table to
`$RUN-compare.md`. Without `CACHES` the job runs the preset's own mode.

`submit_hetjob.sh` reads these from the environment:

| Variable | Meaning |
|---|---|
| `CACHES` | cache modes to run in turn, e.g. `"tvcache contraction"` |
| `TIME` | time limit for both nodes (default 05:00:00, set in `hetjob.sbatch`) |
| `SPLIT`, `TOTAL` | train in parts of `SPLIT` steps up to `TOTAL`, one job per part, each resuming from the previous part's checkpoint |
| `BASE` | the run directory (default `$SCRATCH/tvcache/runs/<time>-<preset>-het`) |

The account and QOS are the `#SBATCH -A ... -q ...` lines in `hetjob.sbatch`
(one pair per node); edit them for another project. `probe_hetjob.sbatch`
checks in 10 minutes that a CPU+GPU job starts and the nodes reach each other.

Everything after the preset is passed to `run.sh`:

| Flag | Meaning |
|---|---|
| `--cache none\|tvcache\|contraction` | no cache / TVCache / TVCache with graph contraction |
| `--threshold X` | contraction: similarity threshold (the sweep: 0.8, 0.9, 0.95) |
| `--embedder minilm\|jina-code` | contraction: `all-MiniLM-L6-v2` or `jina-embeddings-v2-base-code` |
| `--budget N` | contraction: live sandboxes all contraction jobs share (32) |
| `--epochs N`, `--rollouts N`, `--batch N` | epochs, rollouts per task, tasks per training step |
| `--max-tokens N` | tokens per rollout after the prompt (2048) |
| `--verify`, `--no-verify` | run tvverify on the cache afterwards, or skip it. Skipping it also turns the trace off, and `metrics.md` then has no hit rates or call times |
| `--tasks "a b c"` | replace the preset's task list |
| `--mode sample` | rollouts only, no training |
| `--prebuild` | only build the task images through the manager, then exit |
| `--dry-run` | print the resolved settings and exit |

### On one node instead

`run.sh` is also the single-node launcher: inside a GPU allocation it runs
everything on that node.

    sbatch -q shared -C gpu -G 2 -c 64 -t 02:00:00 -A <project> \
      -o $SCRATCH/tvcache/runs/slurm/%x-%j.out --wrap "bash run.sh shakedown"

Half a node should do for the two-task shakedown. The 49-task pilot needed
the whole node (`-N 1 -G 4 -c 128 --mem=0`): on GPU nodes podman's image
store is `/tmp`, a RAM disk that counts toward the job's memory, and both a
quarter and a half node ran out of memory (ISSUES.md #16, #17). The two-node
job avoids this by keeping the containers on the CPU node.

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

## What the fork bank adds to a run

The fork bank is TVCache's prewarming. It is on for every training run with a
cache (`FORK_BANK=0` turns it off). While one batch rolls out, it starts one
container per rollout for the next batch's tasks and restores the `TOP_K`
(10) most used snapshots, and rollouts take these ready-made sandboxes instead
of starting their own. It lives in the sidecar, which also logs where every
training step begins and ends (`control.jsonl`); `metrics.md` needs that log
to split its numbers by epoch, so with the fork bank off they are not split.

Contraction and the fork bank have separate sandbox limits: `--budget` for
contraction, and rollouts per task times tasks per batch, plus `TOP_K`, for
the fork bank.

## What a run leaves in `$RUN`

| Path | What |
|---|---|
| `metrics.md`, `metrics.json` | the testing plan's metrics per task and epoch, and per contraction round (`verl/stats.py`) |
| `logs/run.txt` | the settings, then the outcome: veRL's exit code, peak memory, contraction totals, leftover snapshots |
| `config.env` | every setting the run used |
| `tasks/`, `tasks.parquet` | the task directories and dataset the run used |
| `logs/memory.csv` | once a minute: the job's memory, the image store's size, containers, images and sandboxes |
| `logs/containers.csv` | once a minute: memory per running container (podman-hpc reports one shared value; ISSUES.md #25) |
| `contraction/events.jsonl`, `summary.json` | every pause, resume, listing, merge and finished round with a timestamp; totals and counts over time |
| `control.jsonl` | step starts and ends, each task's idle window and graph size, fork bank warm-ups and hand-outs |
| `state.json` | per task: comparable or not, and the paths skipped as noise |
| `dumps/dump_*.json`, `epoch_*.json` | DUMP_ALL every `DUMP_INTERVAL` s (default 600) and at the end; and as each epoch ended |
| `report.md`, `check/` | tvverify's report and its per-dump results (the last dump also checks snapshots) |
| `trace/` | per-lookup trace and one line per rollout, when tvverify is on |
| `logs/` | veRL, cache server, manager, podman, sidecar, contraction |
| `server.out`, `client.out` | two-node job: each node's output |

## Reading the results

- **`metrics.md`** is the report for the testing plan's numbers: hit rates,
  time saved, graph and snapshot sizes, contraction rounds. It is written at
  the end of every run with a cache. Method against baseline:
  `$VERL_PY -m tvcache_workloads.bash.verl.stats --compare <tvcache run> <contraction run>`
  (the two-node job does this itself when `CACHES` has both).
- **`compare_runs.py`** puts runs side by side, including a run without a
  cache: reward, step time, manager requests, memory, and charts. Its hit
  rates are read from `metrics.json`, so the two reports agree.

      $VERL_PY compare_runs.py $TVROOT/runs/<a> $TVROOT/runs/<b> ... --plot charts/

- **`report.md`** is tvverify's correctness report.
- `$VERL_PY dump_tree.py $TVROOT/runs/<a> hello-world` prints one task's
  cached commands, merged ones marked `=`.

## Other settings

Everything `run.sh` does not cover still comes from the environment (see
`env.sh` and the top of `train_pilot.sh`): `MODEL`, `N_GPUS`,
`SNAPSHOT_THRESHOLD` (fixed 1 s; empty for tvclient's adaptive policy),
`CHECK_NODES`, `DUMP_INTERVAL`, `MAX_TURNS`, `SEED`, `FORK_BANK` (1), `TOP_K`
(10), `RUN` (reuse to resume from a checkpoint; the cache restarts empty),
`TVCACHE_VARIANT` (`combined` is the build `bootstrap.sh` clones; another name
only if you cloned another build into `variants/` yourself).

## veRL environment

`VERL_PY` must be a Python with veRL 0.9.1 (the version the integration was
written against), vLLM and torch. A conda env from NERSC's `pytorch` module
plus `pip install verl[vllm]==0.9.1` works. `setup.sh` then installs tvclient,
tvcache-workloads, tvverify and sentence-transformers into it.
