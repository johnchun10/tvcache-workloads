# Runs on Perlmutter: handoff (updated 2026-11-14, everything in `main`)

Context for whoever continues this work: where things are, what is verified, and what
is left. The source of truth for what to measure is `[TVCache] Testing Plan.md` in the
parent `tvcache` folder (not in any repo). How to run is in
`bash/scripts/perlmutter/README.md`.

## 1. Where things are

Everything a run needs is in `main` of the four forks under `github.com/johnchun10`.
`bootstrap.sh` clones `main` of each. Runs happen on NERSC Perlmutter from these
remotes; local checkouts on the Mac are not what runs.

| Repo | What its `main` holds beyond upstream |
|---|---|
| `tvclient` | graph contraction (upstream PR #4) and the correctness fixes (upstream PR #5), merged; the fork bank changes and per-round contraction stats |
| `tvcache-rust` | graph contraction (upstream PR #2) and the correctness fixes (upstream PR #3), merged |
| `container-manager-public` | cold snapshots, podman-hpc support, the bytes each snapshot adds |
| `tvcache-workloads` | the launcher and presets, contraction wiring, the fork bank wiring, `verl/stats.py`, the CPU+GPU job |

Upstream is `github.com/abhivijay96/{tvclient,tvcache-rust}`; its `main` has none of
this yet, so the forks' `main` is ahead of upstream.

Older branches on the forks (`contraction-runs`, `contraction-plus-fixes`,
`nocache-mode`, `cold-snapshots`) are fully contained in `main` and can be deleted.

## 2. What was built

**Combined client and server.** The contraction and fixes PR sets conflicted (13 hunks
in the server, 8 in the client). Resolved to keep both behaviours. Decisions made by the
project owner:

1. A contraction merge may leave a node with one run's value and another run's snapshot.
   Accepted, because a merge only happens after the two states compared equal.
2. One `Sandbox.equals` (the fixes' digest-based default); the duplicate was removed.
3. A refused duplicate insert still attaches its embedding. Left as is.

Review verdict on the fixes: B1, B2, B3, B5 are real bugs; B4 is an unreachable panic
(hardening only). B3's fix relies on a failed call leaving the same state as a
successful one, which holds for the bash workload.

**tvcache-workloads.**

| Piece | File | What it does |
|---|---|---|
| Two-node launcher | `bash/scripts/perlmutter/submit_hetjob.sh`, `hetjob.sbatch` | queues one job with a CPU node (containers, cache, manager, fork bank, contraction) and a GPU node (veRL); runs `run.sh` once per node; `CACHES`, `SPLIT`, `TIME` |
| Launcher | `run.sh`, `presets/*.env` | `run.sh <preset> [flags]`; presets `shakedown` and `main20-easy-med`; cache modes `none`, `tvcache`, `contraction`; also the single-node launcher |
| Engine | `train_pilot.sh`, `env.sh` | `ROLE=all\|server\|client`; task lists copied into the run folder, state check, sidecar and coordinator start and stop, `config.env`, memory logs, tvverify, `stats.py` |
| Setup | `bootstrap.sh`, `setup.sh` | clone the four repos at `main`; build, install, check both embedders load |
| State fingerprint | `src/tvcache_workloads/bash/state.py` | read-only probe; files under `/app /root /etc /tmp`, exported variables, cwd, installed packages (wide policy, contraction runs only; other runs keep `/app`) |
| State check | `.../bash/state_check.py` | before a run, compares three identical sandboxes per task, records noise paths, marks uncomparable tasks |
| Coordinator | `.../bash/contraction.py` | owns tvclient's `Contractor`; rollouts hold a task from first command to end of tests; events to `contraction/events.jsonl`; `summary` subcommand |
| Embedders | `.../bash/embedding.py` | `minilm`, `jina-code`; on CPU; embeds the command text |
| Fork bank | `cache_sidecar.py`, `.../bash/control.py`, `verl/trainer.py`, `verl/train.py` | the sidecar owns tvclient's `ForkBank` and serves it on port 7880; the trainer warms it one batch ahead and logs step boundaries to `control.jsonl` |
| Metrics | `.../bash/verl/stats.py` | `metrics.md` and `metrics.json` per task and epoch and per contraction round, at the end of every run with a cache; `--compare` for the ratio table |
| Run comparison | `compare_runs.py` | runs side by side (reward, step time, memory, charts); hit rates read from `metrics.json` |
| Dump reading | `correctness/tvverify/model/dump.py`, `cache_admin.node_children` | both dump shapes; merged calls become aliases in tvverify |

The 20 tasks in `main20-easy-med` were chosen by reading Dockerfiles and tests: one small
container, no network needed to solve, work kept under `/app`, tests check files rather
than running services. Only `hello-world` and `openssl-selfsigned-cert` were actually built.

## 3. What is verified and what is not

Verified on the Mac, with `main` of all four repos:
- 176 unit and tvverify tests pass, and the 31 Rust tests of `tvcache-rust`.
- `bootstrap.sh` clones `main` of the four repos and moves a clone on an older branch
  over to `main`.
- `compare_runs.py` and `stats.py` give the same hit rate on a synthetic run.

Verified earlier, before the fork bank and the two-node job were merged:
- Real containers (Docker, `hello-world`): three equivalent commands merged, a similar
  but different one kept apart, a later rollout hit the cache through the merged node;
  tvverify replayed the merged tree (its one finding was `ls -la` timestamp noise).
- A dress rehearsal of `run.sh` in all three cache modes with stand-ins for containers
  and veRL.

On Perlmutter, reported with the fork bank work: a 2-task shakedown on the two-node job
with an earlier version of the code (before it was rebuilt on `run.sh` and contraction),
a login-node smoke test with contraction and the fork bank and a scripted client in
place of veRL, and a 10-minute probe that the two-node job starts.

Not verified:
- Any run of the current `main` on Perlmutter with real veRL, on two nodes or one.
  The shakedown is the first real test.
- Loading the real embedding models (`setup.sh` checks this; `jina-code` needs remote code).
- 18 of the 20 tasks' images (`run.sh <preset> --prebuild` reports failures).
- `CACHES` with `none` in the two-node job (read from the code, not run).

## 4. Metrics the testing plan asks for

Checked against `[TVCache] Testing Plan.md` on 2026-11-14. All of these are in
`metrics.md` unless marked.

| Plan metric | Status |
|---|---|
| Cache hit rate; hit rate of new sequences | Covered |
| LPM hit rate; LPM hit rate of new prefixes | Covered |
| % execution time saved by LPM hits | Covered |
| Median tool-call execution time | Covered (lookup to return, hits included) |
| Nodes, snapshots, snapshot bytes per epoch | Covered (a dump at each epoch end; sizes from the manager log) |
| [M] Preempted; depth reached; graph depth per round | Covered |
| [M] Contraction walltime; idle time per round | Covered |
| [M] Similar pairs; pairs decided per round | Covered |
| [M] Sandboxes restored; nodes restored per round | Covered (`sandboxes_built`, `nodes_built`) |
| [M] Tool calls replayed; nodes merged per round | Covered |
| [M] Comparisons that found equivalence | Covered |
| Rollouts hitting max length | Covered (`length_truncated`, `turn_capped`) |
| [M] Sandboxes compared per round | Partial: comparisons are counted, not distinct sandboxes |
| [M] Merged nodes equivalent to replay | Partial: tvverify samples by hit count, not merged nodes first |
| Live sandboxes over time, training vs contraction | Partial: total only, no split |
| Memory of live sandboxes over time | Partial: job total; podman-hpc reports one shared value per container |

`metrics.md` needs two things on, and `logs/run.txt` says so when one is off: the trace
(off with `--no-verify`; then no hit rates or call times) and the fork bank (off with
`FORK_BANK=0`; then no split by epoch, graph sizes or idle windows).

## 5. Open tasks, in order

1. **Shakedown on Perlmutter** (section 6). Nothing below is worth doing before it.
2. **Fork bank counts fake hits.** tvclient's `ForkBank` looks up each hot snapshot with
   `find`, which counts a cache hit on that node; `lease_sandbox` exists for this and
   counts nothing. Change `fork_bank.py` in tvclient. `metrics.md` is not affected (it
   reads the trace); tvverify's sampling by hit count and the fork bank's own hot list
   are. Found by reading the code, not measured.
3. **Metrics still to build:** the merged-first replay check, the training vs
   contraction split of live sandboxes, distinct sandboxes compared.
4. **Decouple the metrics from the two switches:** trace without tvverify, and the step
   log without the fork bank.
5. **Simplify the settings:** `FORK_BANK`, `TOP_K`, `SEED` and the split settings have
   no `run.sh` flag; the `variants/` folder layer is no longer needed with one build.
6. **`contraction summary`** repeats totals that `metrics.md` has per round. Remove it
   once the `round` events are seen in a real run.
7. `correctness/README.md` does not mention contracted dumps or aliases.

Test recipe (paths relative to `tvcache-workloads`, with `tvclient` and `tvcache-rust`
checkouts of the forks' `main`):

    TVCACHE_RUST_DIR=<tvcache-rust> PYTHONPATH=src:correctness:<tvclient> python -m pytest -q correctness/tests tests
    TVCACHE_PORT=<port> PYTHONPATH=... python -m pytest -q -m cache_integration    # needs the server on <port>

## 6. First run on Perlmutter

On a login node:

    git clone https://github.com/johnchun10/tvcache-workloads.git $SCRATCH/tvcache/tvcache-workloads   # first time only
    cd $SCRATCH/tvcache/tvcache-workloads/bash/scripts/perlmutter
    bash bootstrap.sh && bash setup.sh
    bash run.sh shakedown --prebuild
    bash submit_hetjob.sh shakedown

Then read `logs/run.txt`, `metrics.md` and `report.md` in the run directory. If the
two-node job does not start, the same shakedown on one GPU node:

    sbatch -q shared -C gpu -G 2 -c 64 -t 02:00:00 -A <project> --wrap "bash run.sh shakedown"

After a clean shakedown: `main20-easy-med` with `CACHES="none tvcache contraction"`,
then thresholds 0.8, 0.9, 0.95 and both embedders.
