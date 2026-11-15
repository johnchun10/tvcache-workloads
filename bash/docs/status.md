# Bash workload: status (2026-11-14)

What is verified, which metrics of the testing plan a run reports, and what is still
open. How to run is in `bash/scripts/perlmutter/README.md`. The testing plan itself
(`[TVCache] Testing Plan.md`) is kept outside the repos.

## Verified

Off Perlmutter, with `main` of our four forks:

- 178 unit and tvverify tests, and the 31 Rust tests of `tvcache-rust`.
- `bootstrap.sh` clones `main` of the four repos and moves a clone on an older branch
  over to `main`.
- `submit_hetjob.sh` against a stand-in scheduler.
- The run report on synthetic runs, with and without a cache; `compare_runs.py` shows
  the same hit rates as the report.

Earlier, before the fork bank and the two-node job were added:

- Real containers (Docker, `hello-world`): three equivalent commands merged, a similar
  but different one kept apart, a later rollout hit the cache through the merged node;
  tvverify replayed the merged tree (its one finding was `ls -la` timestamp noise).
- `run.sh` in all three cache modes with stand-ins for containers and veRL.

On Perlmutter:

- The 49-task pilot with TVCache on one GPU node (`bash/scripts/perlmutter/ISSUES.md`).
- A 2-task shakedown on the two-node job with an earlier version of the code, before it
  was rebuilt on `run.sh` and contraction; a login-node smoke test with contraction and
  the fork bank and a scripted client in place of veRL; a 10-minute probe that the
  two-node job starts.

## Not verified

- Any run of the current `main` on Perlmutter with real veRL, on two nodes or one. The
  shakedown (`bash submit_hetjob.sh shakedown`) is the first real test.
- The embedding service and its environment on Perlmutter (`setup.sh` builds and checks
  it). On a laptop both models load, and rollouts' code gets vectors from the service.
- Which threshold suits `embeddinggemma2`. Its scores sit much higher than `minilm`'s
  (`bash/scripts/perlmutter/README.md`); 0.95 to 0.98 is a guess from 12 command pairs.
- 18 of the 20 task images of `main20-easy-med` (`run.sh <preset> --prebuild` reports
  failures). The tasks were chosen by reading Dockerfiles and tests: one small container,
  no network needed to solve, work kept under `/app`, tests check files.
- `CACHES` with `none` in the two-node job (read from the code, not run).

## Metrics the testing plan asks for

All of these are in the run's `report.md` (and `metrics.json`) unless marked.

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
| The ratio table | Covered (`compare_runs.py`, or `stats.py --compare`) |
| [M] Sandboxes compared per round | **Partial**: comparisons are counted, not distinct sandboxes |
| [M] Merged nodes equivalent to replay | **Partial**: tvverify replays merged calls, but samples nodes by hit count, not merged nodes first, and the report has no count of them |
| Live sandboxes over time, training vs contraction | **Partial**: total only, no split |
| Memory of live sandboxes over time | **Partial**: the job's total; podman-hpc reports one shared value per container (ISSUES.md #25) |

The cache metrics need two things on, and `logs/run.txt` says so when one is off: the
trace (off with `--no-verify`; then no hit rates or call times) and the fork bank (off
with `FORK_BANK=0`; then no split by epoch, graph sizes or idle windows).

## Open tasks, in order

1. **Shakedown on Perlmutter.** Nothing below is worth doing before it.
2. **The four partial metrics above.** Distinct sandboxes compared and the training vs
   contraction split need counters in tvclient; the merged-first check is a sampling
   change in tvverify; per-sandbox memory needs something other than `podman stats`.
3. **Fork bank counts fake hits.** tvclient's `ForkBank` looks up each hot snapshot with
   `find`, which counts a cache hit on that node; `lease_sandbox` exists for this and
   counts nothing. Change `fork_bank.py` in tvclient. The report is not affected (it
   reads the trace); tvverify's sampling by hit count and the fork bank's own hot list
   are. Found by reading the code, not measured.
4. **Decouple the metrics from the two switches:** trace without tvverify, and the step
   log without the fork bank.
5. **Simplify the settings:** `FORK_BANK`, `TOP_K`, `SEED` and the split settings have
   no `run.sh` flag; the `variants/` folder layer is no longer needed with one build.
6. **`contraction summary`** (`contraction/summary.json`, printed into `logs/run.txt`)
   repeats totals the report has per round. Remove it once the `round` events are seen
   in a real run.
