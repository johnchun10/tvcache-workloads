# Contraction runs: handoff (2026-11-14)

Context for whoever continues this work: what was built, where it lives, what is
verified, and what to do next. The source of truth for what to measure is
`[TVCache] Testing Plan.md` in the parent `tvcache` folder (not in any repo).

## 1. Where things are

All on the forks under `github.com/johnchun10`. Runs happen on NERSC Perlmutter from
these remote branches; local checkouts on the Mac are not what runs.

| Repo | Branch | State |
|---|---|---|
| `tvclient` | `contraction-plus-fixes` (head `b38afdc`) | graph contraction (upstream PR #4, includes embedding and sandbox-equals) merged with the correctness fixes (upstream PR #5) |
| `tvcache-rust` | `contraction-plus-fixes` (head `d472f77`) | graph contraction (upstream PR #2) merged with the correctness fixes (upstream PR #3) |
| `tvcache-workloads` | `contraction-runs` | three commits on top of `nocache-mode`: tvverify dump reading, workload contraction wiring, launcher and presets |
| `container-manager-public` | `main` | includes the podman-hpc PR; nothing from this work |

Upstream is `github.com/abhivijay96/{tvclient,tvcache-rust}`. No pull request is open
for any of the three branches. `nocache-mode` is not merged into `tvcache-workloads`
`main` yet (it merges without conflicts).

## 2. What was built

**Combined client and server.** The two PR sets conflicted (13 hunks in the server, 8 in
the client). Resolved to keep both behaviours. Decisions made by the project owner:

1. A contraction merge may leave a node with one run's value and another run's snapshot.
   Accepted, because a merge only happens after the two states compared equal.
2. One `Sandbox.equals` (the fixes' digest-based default); the duplicate was removed.
3. A refused duplicate insert still attaches its embedding. Left as is.

Review verdict on the fixes: B1, B2, B3, B5 are real bugs; B4 is an unreachable panic
(hardening only). B3's fix relies on a failed call leaving the same state as a
successful one, which holds for the bash workload.

**tvcache-workloads (`contraction-runs`).**

| Piece | File | What it does |
|---|---|---|
| Launcher | `bash/scripts/perlmutter/run.sh`, `presets/*.env` | `run.sh <preset> [flags]`; presets `shakedown` and `main20-easy-med`; cache modes `none`, `tvcache`, `contraction` |
| Engine | `bash/scripts/perlmutter/train_pilot.sh`, `env.sh` | task lists copied into the run folder, state check, coordinator start and stop, `config.env`, memory logs |
| Setup | `bootstrap.sh`, `setup.sh` | clone the four repos at their branches; build, install, check both embedders load |
| State fingerprint | `src/tvcache_workloads/bash/state.py` | read-only probe; files under `/app /root /etc /tmp`, exported variables, cwd, installed packages (wide policy, contraction runs only; other runs keep `/app`) |
| State check | `.../bash/state_check.py` | before a run, compares three identical sandboxes per task, records noise paths, marks uncomparable tasks |
| Coordinator | `.../bash/contraction.py` | owns tvclient's `Contractor`; rollouts hold a task from first command to end of tests; events to `contraction/events.jsonl`; `summary` subcommand |
| Embedders | `.../bash/embedding.py` | `minilm`, `jina-code`; on CPU; embeds the command text |
| Session wiring | `.../bash/verl/session.py` | `TVCACHE_EMBEDDER`, `TVCACHE_CONTRACTION_PORT` |
| Dump reading | `correctness/tvverify/model/dump.py`, `cache_admin.node_children` | both dump shapes; merged calls become aliases in tvverify |

The 20 tasks in `main20-easy-med` were chosen by reading Dockerfiles and tests: one small
container, no network needed to solve, work kept under `/app`, tests check files rather
than running services. Only `hello-world` and `openssl-selfsigned-cert` were actually built.

## 3. What is verified and what is not

Verified on the Mac:
- 119 workload unit tests and 55 tvverify tests on the combined build; same suites pass on
  the fixes-only build. 31 Rust tests on the combined server.
- Real containers (Docker, `hello-world`): three equivalent commands merged, a similar but
  different one kept apart, a later rollout hit the cache through the merged node;
  tvverify replayed the merged tree (its one finding was `ls -la` timestamp noise).
- A dress rehearsal of `run.sh` in all three cache modes with stand-ins for containers and
  veRL.

Not verified:
- Anything on Perlmutter: podman-hpc, real veRL, the GPU node. The shakedown is the first
  real test.
- Loading the real embedding models (`setup.sh` checks this; `jina-code` needs remote code).
- 18 of the 20 tasks' images (`run.sh <preset> --prebuild` reports failures).

## 4. TODO 1: metrics the testing plan asks for

Checked against `[TVCache] Testing Plan.md` on 2026-11-14. The plan wants most metrics
per task and per epoch or per contraction round; today most are run-wide totals.
Assumption: the two "recently added" metrics are the last two of the main list (merged
nodes equivalent to replay; rollouts hitting the maximum length).

| Plan metric | Status | Notes |
|---|---|---|
| Cache hit rate | Partial | run-wide from dumps (`compare_runs.py`); per task is in the dumps but not reported |
| Hit rate of new sequences | Missing | derivable from the trace (`lookup` and `executed` events); nothing computes it |
| LPM hit rate | Partial | LPM hit count only, no rate |
| % execution time saved by LPM hits | Missing | derivable from trace `lpm_depth` plus per-call runtimes |
| LPM hit rate of new prefixes | Missing | as above |
| Median tool-call execution time | Missing | trace has `runtime_s` for executed calls only; hits are not timed |
| Nodes per epoch | Partial | per 10-minute dump, not aligned to epochs; per task only in raw dumps |
| Snapshots per epoch | Partial | same |
| Snapshot size in bytes per epoch | Missing | only the whole image store's size is logged |
| [M] Preempted per round | Partial | derivable: `job_end` after `pause` means preempted; not reported |
| [M] Depth reached per round | Tracked | `job_end.depth` in the events |
| [M] Graph depth per round | Missing | |
| [M] Contraction walltime per round | Partial | derivable from `resume` to `job_end` |
| [M] Idle time per round | Partial | derivable from `resume` (minus grace) to the next `pause` |
| [M] Similar pairs per round | Tracked | `groups` events; the run total recounts a pair each round |
| [M] Similar pairs decided per round | Missing | lives in tvclient's `EquivalenceTracker`, not exposed |
| [M] Sandboxes compared per round | Partial | comparisons are counted, not distinct sandboxes |
| [M] Sandboxes restored per round | Mismatch | `restores` counts only builds from a snapshot; the plan counts every on-demand build |
| [M] Nodes whose sandboxes were restored | Missing | |
| [M] Tool calls replayed per round | Tracked | `job_end.replays` |
| [M] Nodes merged per round | Tracked | `apply` events |
| [M] Comparisons that found equivalence | Missing | needs tvclient to report it |
| [M] Merged nodes equivalent to replay (new) | Partial | tvverify replays aliases, but samples by hit count, not merged-first |
| Rollouts hitting max length (new) | Missing | veRL may log a clip ratio per step; unverified |
| Live sandboxes over time, training vs contraction | Partial | total container count per minute; no split |
| Memory of live sandboxes over time | Partial | job total per minute; per container only if podman reports it |

Most of the "Missing" contraction metrics need counters added inside tvclient's
`contraction.py` (decided pairs, equal comparisons, restored nodes, graph depth), which
is where the teammate's metrics PRs below probably overlap. Do the review first.

## 5. TODO 2: review the teammate's PRs and plan the merge

Pushed by a teammate, likely overlapping this work; at least one probably builds on
`contraction-plus-fixes`:

- `github.com/johnchun10/tvcache-workloads/pull/4`: prewarming, metrics, reserving CPU and GPU
- `github.com/johnchun10/tvclient/pull/1`: prewarming, metrics
- `github.com/johnchun10/container-manager-public/pull/3`: metrics

What to do:
1. Fetch each PR head (`git fetch <fork url> pull/<n>/head:<name>`; `gh` is not installed,
   the public API works with curl). Find each one's base and what it is stacked on.
2. Compare with `contraction-runs` and `contraction-plus-fixes`. Expected overlap:
   metrics (this branch's `contraction.py` events, `compare_runs.py`, the memory logs in
   `train_pilot.sh`), the run scripts (`run.sh`, `train_pilot.sh`, `env.sh`, presets), and
   prewarming (this branch runs no prewarming; the sidecar only drains snapshots).
3. Trial-merge in scratch clones, run the tests (recipe below), and report: what conflicts,
   which side to keep per area, the merge order, and which testing-plan metrics the PRs
   already cover (update the table in section 4).
4. Do not push or merge without the owner's go-ahead.

Test recipe (paths relative to `tvcache-workloads`, with a combined `tvclient` checkout):

    PYTHONPATH=src:correctness:<tvclient> python -m pytest -q                      # unit
    TVCACHE_PORT=<port> PYTHONPATH=... python -m pytest -q -m cache_integration    # needs the combined server on <port>
    TVVERIFY_PATCHED=1 TVCACHE_RUST_DIR=<tvcache-rust> PYTHONPATH=correctness:<tvclient> python -m pytest -q correctness/tests

## 6. TODO 3: documentation still to update

Done: `bash/scripts/perlmutter/README.md` and `GUIDE.md` (both current), module
docstrings, this file, `AGENTS.md`. `stage.sh` was removed (replaced by `bootstrap.sh`).
In the parent `tvcache` folder, `TVCACHE_PROJECT_KNOWLEDGE.md` was updated and the old
`PERLMUTTER_PILOT_GUIDE.md` copy deleted.

Still to do:
- `correctness/README.md` does not mention contracted dumps or aliases.
- No PR descriptions exist for the three branches.
- `stage.sh` was deleted and `GUIDE.md` rewritten on this branch; the teammate's
  tvcache-workloads PR may touch the same files, so expect that in the merge.

## 7. First run on Perlmutter

    cd $SCRATCH/tvcache/tvcache-workloads && git fetch origin contraction-runs && git checkout contraction-runs
    cd bash/scripts/perlmutter && bash bootstrap.sh && bash setup.sh
    bash run.sh shakedown --prebuild                      # login node
    sbatch -q shared -C gpu -G 2 -c 64 -t 02:00:00 -A <project> --wrap "bash run.sh shakedown"

Then `main20-easy-med` with `--cache none`, `--cache tvcache`, and `--cache contraction`
across thresholds 0.8, 0.9, 0.95 and both embedders.
