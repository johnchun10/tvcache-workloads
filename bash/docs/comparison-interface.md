# Environment comparison: boundary with Shawn's work

Shawn owns the correctness and sandbox-comparison work. This repo
**does not implement or fake an equality check**. `TerminalBenchSandbox` has
no `equals` method. If tvclient PR #3 lands, the base class default
(`NotImplementedError`) applies until a real comparator exists.

## What this repo provides for comparison work

- **Real live Bash environments on demand:** `TerminalBenchSandbox` lets you
  start, run commands in, snapshot, restore, and stop real Terminal-Bench
  containers. A cold snapshot is a set of Docker images, which can be
  inspected without running anything, or restored to get a live sandbox to
  compare.
- **Pairs of states worth comparing:** the `hello-world` trace has
  `solve` and `equivalent-rewrite`, which write the same file with different
  commands (`echo 'Hello, world!' > …` vs `printf 'Hello, world!\n' > …`).
  Exact-match caching treats them as different branches. Whether they're
  equivalent is for the comparator to decide; nothing here asserts it.
- **Container identity:** a live id `term_<uuid>` maps to Docker containers
  whose names contain `<uuid>` (for example
  `hello-world_<uuid>_test_container`). This is how the e2e tests check for
  leaks. It is a manager naming convention, not an API.

## State inventory for one Bash sandbox

A comparator has to say which of these it covers and which it excludes:

| State | Survives a snapshot here? | Notes |
|---|---|---|
| Filesystem of each compose container | yes | Should include mounted volumes and exclude volatile paths (`/proc`, `/tmp`, logs, caches?) |
| cwd, exported variables | yes (`ENABLE_CACHING=1`) | Saved by tmux-resurrect to `/root/target_env.sh` |
| Unexported variables, functions, aliases, shell options | no | |
| Running and background processes | no | |
| Terminal / pane contents | n/a | Observations are cached per call, not as state |
| Task services (DBs, servers) in other containers | filesystem only | |

The design notes that two equal states can still produce different command
output, and that nodes keep each tool call's own output. Here the cache
stores per-call observations, so merging nodes doesn't merge outputs.

## Open questions to settle with Shawn (not decided here)

1. **Where comparison runs:** in Python, by collecting state through manager
   operations or `docker exec`/`docker diff`; or as a new manager endpoint
   (which would replace `/hash`)?
2. **What "equal" covers for Bash:** filesystem only, or also cwd/env? Which
   paths are excluded?
3. **Read-only access:** collecting state by typing commands into the
   agent's shell changes shell history and `$?`. Should comparison use
   `docker exec` outside the agent's tmux session instead?
4. **Interface:** John's `Sandbox.equals(other)` is async and compares two
   *live* sandboxes. Snapshot handles refuse `execute`. Should the comparator
   restore a handle temporarily, or compare snapshot images directly?
5. **Cost reporting:** comparison time should be measured with the same event
   hook (`SandboxConfig.on_event`) used for snapshot and restore costs.
