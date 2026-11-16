# Terminal-Bench runs on Perlmutter

A run is GRPO training with veRL on Terminal-Bench tasks, with every bash
command going through no cache, TVCache, or TVCache with graph contraction.
Afterwards tvverify checks the cache the run built, and everything ends up in
one file, `<run>/report.md`.

A run is one Slurm job on two nodes: a **CPU node** for the containers,
snapshots, cache server, container manager, fork bank and contraction, and a
**GPU node** for veRL. Slurm starts both, so you only work on a login node.

First time? Do the one-time setup in `GUIDE.md`. It also has fixes for
known errors.

## Where things run

| What | Where | Command (from this directory) |
|---|---|---|
| Update the code | login node | `bash bootstrap.sh && bash setup.sh` |
| Check the task images build | login node (free) | `bash run.sh <preset> --prebuild` |
| Print a run's settings | login node | `bash run.sh <preset> [flags] --dry-run` |
| Start a run | login node | `bash submit_hetjob.sh <preset> [flags]` |
| Compare runs | login node | `$VERL_PY compare_runs.py <run> <run> ...` |

`$TVROOT` is `$SCRATCH/tvcache`. Runs go in `$TVROOT/runs/`, and Slurm
output goes in `$TVROOT/runs/slurm/`.

## Running

For a new set of tasks:

    bash run.sh shakedown --prebuild                  # do the images build?
    bash submit_hetjob.sh shakedown                   # small run: does every part work?
    bash submit_hetjob.sh main20-easy-med --cache all --time 08:00:00

| Preset | Tasks | Settings |
|---|---|---|
| `shakedown` | 1 easy + 1 medium | contraction at 0.9 with `minilm`; 2 epochs, 4 rollouts per task, 1 task per step |
| `main20-easy-med` | 10 easy + 10 medium | contraction at 0.9 with `minilm`; 5 epochs, 8 rollouts per task, 4 tasks per step |

Presets are in `presets/`. Any flag below overrides them.

### Flags

`submit_hetjob.sh` takes every flag. `run.sh` takes all of them except
`--time`, `--split`/`--total` and `--cache all`, which only make sense for a
queued job.

| Flag | Meaning |
|---|---|
| `--cache none\|tvcache\|contraction\|all` | no cache / TVCache / TVCache with graph contraction / all three back to back in one job, each with an empty cache, plus `<run>-compare.md` comparing them |
| `--time hh:mm:ss` | time limit for both nodes (default 05:00:00) |
| `--split N --total M` | train M steps in parts of N, one job per part, each resuming from the previous part's checkpoint. The cache starts empty in each part |
| `--threshold X` | contraction: how similar two commands must be to be compared for merging (table below) |
| `--embedder minilm\|embeddinggemma2` | contraction: the model that scores command similarity |
| `--budget N` | contraction: sandboxes it may use at once (32) |
| `--top-k N` | fork bank: most-used snapshots restored ahead of each step (10) |
| `--snapshot-threshold S\|adaptive` | snapshot after commands slower than S seconds (1.0), or tvclient's adaptive policy (rarely snapshots) |
| `--epochs N`, `--rollouts N`, `--batch N` | epochs, rollouts per task, tasks per training step |
| `--max-tokens N` | tokens per rollout after the prompt (2048) |
| `--tasks "a b c"` | replace the preset's task list |
| `--mode sample` | rollouts only, no training |
| `--no-verify` | skip tvverify. This also turns off the trace, so `report.md` then has no hit rates |
| `--name LABEL` | added to the run directory's name |
| `--prebuild`, `--dry-run` | build the task images and exit / print the settings and exit |

Rarely changed settings come from the environment: `MODEL`, `N_GPUS` (1),
`MAX_TURNS` (30), `SEED` (42), `FORK_BANK` (1; 0 turns prewarming off), `CHECK_NODES`
(20), `DUMP_INTERVAL` (600 s), and `RUN` (reuse a run directory to resume from
its checkpoint). See `env.sh` and the top of `train_pilot.sh`. The Slurm
account and QOS are the `#SBATCH -A/-q` lines in `hetjob.sbatch`, one pair
per node.

### Contraction thresholds

The two embedders score on different scales. These figures come from 12
hand-written command pairs, so treat them as a rough guide:

| | `minilm` | `embeddinggemma2` |
|---|---|---|
| same command written two ways | 0.79–0.97 | 0.96–0.99 |
| different commands that look alike | 0.60–0.98 | 0.92–0.98 |
| unrelated commands | 0.00 | 0.77 |
| thresholds to sweep | 0.8, 0.9, 0.95 | 0.95, 0.97, 0.98 |

### On one GPU node instead

`run.sh` called inside a GPU allocation runs everything on that node:

    sbatch -q regular -C gpu -N 1 -G 4 -c 128 --mem=0 -t 05:00:00 -A <project> \
      -o $SCRATCH/tvcache/runs/slurm/%x-%j.out --wrap "bash run.sh shakedown"

Use a whole node. On GPU nodes podman stores images in `/tmp`, which is RAM
and counts against the job's memory. Half a node ran out of memory on the
49-task pilot.

## Reading `report.md`

It is written when training ends, and again after tvverify. To rebuild it:
`$VERL_PY -m tvcache_workloads.bash.verl.stats <run>`.

**Training**
- **Reward per rollout:** 1 if the task's tests passed, 0 if they failed, -1
  if the model never made a valid tool call.
- **Learning signal:** GRPO learns only from a task whose rollouts score
  differently from each other. All-pass or all-fail gives no signal.

**Cache, per task and epoch**
- **lookups:** the commands the model ran. Each is looked up with its full
  command history.
- **cache_hit_rate:** lookups answered from the cache, with nothing run.
- **new_seq_hit_rate:** hits on a command history that was never stored
  itself but was merged with one that was. This is what contraction adds.
- **lpm_hit_rate:** of the misses, how many resumed from a saved snapshot
  (the longest prefix match), so only the remaining commands ran.
- **new_prefix_lpm_hit_rate:** the same, for snapshots reached through a
  merged prefix.
- **lpm_time_saved_pct:** command time skipped thanks to those resumes.
- **nodes:** one per stored command, with its history.
- **graph_depth:** the longest stored command chain.
- **snapshots:** saved container states. One is only taken after a command
  slower than `--snapshot-threshold`.
- **merged_nodes_in_graph:** nodes contraction merged with another.
- **resumes:** where a rollout's container came from after a miss:
  - `fresh`: a new container started from the task's image
  - `live`: a running container already in the right state
  - `fork`: restored from a snapshot
- **from bank:** the container was prewarmed by the fork bank.

**Contraction and the fork bank**
- **Contraction rounds:** contraction runs on a task only while none of its
  rollouts is active. It pairs similar commands by embedding score, replays
  them in sandboxes, and merges the ones that end in the same state (same
  files, environment variables, current directory and packages).
- **The state check:** before training, three identical sandboxes per task
  are compared. Paths that still differ count as noise for that task (in
  `state.json`), and a task that can't be compared is never contracted.
- **Preempted:** the round stopped because a rollout came back.
- **Fork bank:** TVCache's prewarming. While one step runs, it starts
  containers for the next step's tasks and restores the `--top-k` most-used
  snapshots.

**Correctness (tvverify)**
- tvverify replays cached nodes from a fresh container and compares them with
  what the cache holds.
- **bug:** the cache disagrees with the replay.
- **noise:** two fresh replays already disagree with each other, for example
  timestamps or random keys.
- **inconclusive:** the replay couldn't run.

## Other files in a run

| Path | What |
|---|---|
| `logs/run.txt`, `server.out`, `client.out` | the run's log and each node's output; look here first if a run failed |
| `config.env`, `metrics.json` | every setting used; the report's numbers as data |
| `logs/memory.csv` | memory, image store size and containers, once a minute |
| `contraction/events.jsonl` | every contraction pause, merge and round |
| `dumps/`, `check/` | cache dumps; tvverify's results |

**Several runs side by side:**

    $VERL_PY compare_runs.py $TVROOT/runs/<a> $TVROOT/runs/<b> --plot charts/

With one `tvcache` run and one `contraction` run, it also prints the testing
plan's ratio table.

**One task's cached commands:**

    $VERL_PY dump_tree.py $TVROOT/runs/<a> hello-world

Merged commands are marked `=`.

**Cost:** a two-node shakedown is about 0.8 node-hours. A `main20-easy-med`
run hasn't been timed yet. Check usage with `sshare -A <project>` or Iris.
