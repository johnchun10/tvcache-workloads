# Verification record — 2026-10-30

## Setup

| Component | Version |
|---|---|
| tvclient | `45d8c7d` (main, unmodified) |
| tvcache-rust | `a584261` (main, unmodified), `--release`, `--eviction-policy=none` |
| container-manager-public | our fork, branch `cold-snapshots` @ `70658ce` (local commit on upstream `51a90bb`), `ENABLE_CACHING=1`, `uv sync --frozen` |
| Terminal-Bench | vendored in the manager repo (`terminal-bench` 0.2.17) |
| Task | `hello-world` (existing task, image `t-bench/python-3-13:20250620`) |
| Machine | macOS 27.2, Apple Silicon (arm64), Docker Desktop 29.4.0 (8 GB VM, 10 CPUs) |
| Toolchain | Python 3.13.18 (uv-managed), uv 0.13.6, cargo/rustc 1.98.0, httpx 0.28.1 |

## Test tiers

| Tier | What is real | What is faked | Result |
|---|---|---|---|
| Unit (`uv run pytest`) | nothing | manager (`tests/bash/fake_manager.py`), HTTP | 78 passed |
| Cache integration (`-m cache_integration`) | tvcache-rust, tvclient executor | manager (same test double) | 12 passed, 2 skipped |
| End-to-end (`-m e2e`) | everything: cache, manager, Docker | nothing | 4 passed (lifecycle + trace, each in cold and parked mode) |
| Manager unit (`tests/unit/snapshots` in the manager repo) | nothing | `docker` calls | 6 passed |

The 2 skipped tests are the embedding-switch test in each mode. They need
tvclient PR #2, and pass against it: with the PR client and tvcache-rust PR
#1, the cache-integration tier gave 7 passed in parked-only mode before cold
mode was added.

## Bug found by the real tests

`docker commit` copies the container's compose project label, and
`docker compose down --rmi all` deletes images with that label. So stopping
the original terminal deleted its cold snapshots. Fixed by relabeling at
commit time (see `snapshot-design.md`). The e2e tests now also check that the
set of Docker images is unchanged after a run.

## Example run (`bash/examples/run_trace.py`, cold mode, snapshot after every stateful call)

| Rollout | Step hits | Test | Test hit | Wall time | Snapshot / restore (s) |
|---|---|---|---|---|---|
| solve | miss ×3 | passed | no | 13.0 s | 3 snapshots, 1.73 total |
| solve-again | hit ×3 | passed | **yes** | 0.0 s | none |
| wrong-content | hit, miss, miss | failed | no | 9.3 s | restore 0.51 |
| inspect-prefix | hit, miss, miss | failed | no | 9.7 s | restore 0.50 (same snapshot) |
| equivalent-rewrite | hit, miss, miss | passed | no | 9.2 s | restore 0.51 |

Cleanup deleted 9 snapshots. Afterwards there were no task containers and no task images.
For comparison, the parked mode took about 1.3 s per snapshot and about 1.0–1.3 s per
restore, and divergent rollouts took about 13 s.

## Not verified

- Tasks other than `hello-world`; multi-container tasks; tasks with volumes.
- Concurrency: parallel rollouts on one manager, ForkBank, eviction under load.
- The adaptive snapshot policy end to end (runs used a fixed tiny threshold).
- The real Jina embedding model (only a stub embedder was tested).
- Linux hosts.
- Command timeouts against the real manager (only the test double).
