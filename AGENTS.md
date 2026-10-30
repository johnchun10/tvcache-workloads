# AGENTS.md

Context for coding agents working in this repo.

## What this repo is

`tvcache-workloads` connects workloads to TVCache, a cache for agent tool calls. A cache
hit returns the stored result; a miss resumes from the deepest saved snapshot and runs
only the missing commands. Only the Bash workload (Terminal-Bench) exists so far. SQL is
planned and should go in its own subpackage (`src/tvcache_workloads/sql/`, `sql/`, `tests/sql/`).

## The system (separate repos, cloned side by side under one parent folder)

| Repo | Role | Notes |
|---|---|---|
| `tvclient` (abhivijay96/tvclient) | Python client. `TvcacheToolExecutor` decides hit, miss, or resume | Installed editable from `../tvclient`. Defines the `ToolCall` and `Sandbox` interfaces. |
| `tvcache-rust` (abhivijay96/tvcache-rust) | Cache server, TCP on :7878 | Stores results, the call tree, and snapshot ids. Never touches Docker. |
| `container-manager-public` (our fork johnchun10/..., branch `cold-snapshots`) | Flask server on :9901 that runs Terminal-Bench containers | Endpoints: `/start /execute /get_pane_logs /fork /test /stop`, plus our `/snapshot /restore /delete_snapshot`. Built only for Terminal-Bench. |

Data flow: runner -> `TvcacheToolExecutor` -> cache server. On a miss, the executor calls
`TerminalBenchSandbox`, which sends HTTP requests to the manager, which runs Docker.

tvclient and tvcache-rust need no changes for this repo. Don't edit them from here.
Changes to the manager go on our fork.

## Code map (`src/tvcache_workloads/bash/`)

- `tool_call.py`: `BashToolCall`. The cache key is `{"tool": "bash", "command": <exact text>}`. Commands are stateful unless `mutates=False`.
- `sandbox.py`: `TerminalBenchSandbox(Sandbox)`.
  - `start`: fresh terminal, or restore from a handle.
  - `fork`: snapshot.
  - `execute`: runs a command, or `/test` for a `TestToolCall`.
  - `stop`: stops a terminal or deletes a snapshot.
  - Configure it with `with_config(SandboxConfig(...))`, because tvclient constructs sandboxes as `cls(task_name=, env_id=)`.
- `manager_client.py`: async httpx client with per-endpoint timeouts and typed errors (`ManagerError`, `CommandTimeout`, `ManagerUnavailable`, `ManagerProtocolError`).
- `observation.py`: wraps each command in BEGIN/END `printf` markers and extracts the output from the pane log.
- `runner.py`: `run_rollout(...)` sends the cumulative history through the executor and records hits, executions, and timings. Supports `embed_tool_calls`, which needs tvclient PR #2.
- `../cache_admin.py`: shared `release_task_snapshots` marks a task's snapshots for eviction, then drains them. The drain is global across the cache server.

## Invariants (don't break)

- **Snapshot handles have prefixes.** `tb-snap:<id>` is a cold snapshot (manager images); `tb-parked:<terminal>` is a parked running copy. Dispatch on the prefix, not on the current config.
- **Handles are not live terminals.** `execute()` and `fork()` refuse handles, and `start()` refuses a plain live id. Never treat a running container as a snapshot.
- **Results starting with `Failed to execute` are never cached by tvclient.** Use them for environment-side failures. Raise instead when the shell state is unknown, for example on a timeout.
- **Always pass the full command history** to `lookup_or_execute`.
- **Snapshot state limits.** A snapshot keeps files, cwd, and exported variables (the manager needs `ENABLE_CACHING=1`). It loses processes, unexported variables, and volumes.
- **Output format.** Output is terminal-rendered text: blank lines dropped, no exit code, wrapped at 160 columns.
- **Environment equality is not implemented.** Don't add a placeholder `equals`.

## Manager quirks to remember

- `/execute` returns only a status. Output comes from `/get_pane_logs`, which is cumulative and includes the parent's history after a fork.
- The manager appends `; tmux wait -S done` to every command, which hides `$?`. It also joins multi-line commands with ` && `.
- `/stop` with an unknown id returns 500. `/fork` can hang forever on failure, so always use client-side timeouts.
- `docker compose down --rmi all` runs on every stop. That's why each restore gets its own image tag and snapshot images get their own Compose project label.

## Commands

```bash
uv sync
bash/scripts/start_cache.sh            # cache server (cargo, in ../tvcache-rust)
bash/scripts/start_manager.sh          # manager (needs Docker running)
uv run pytest                          # unit tests, all mocked (tests/bash/fake_manager.py)
uv run pytest -m cache_integration     # needs the cache server
uv run pytest -m e2e -s                # needs both servers and Docker; runs cold and parked modes
uv run python bash/examples/run_trace.py bash/traces/hello-world.json --snapshot-threshold 0.001
uvx ruff check src tests bash --select E,F,I --line-length 110
```

In tests, use a fresh cache namespace per run: task id `"hello-world::<name>"`. The part
before `::` is the Terminal-Bench task name.

## Conventions

- Keep the separation: workload code lives here; client fixes go in tvclient, server fixes in tvcache-rust, container fixes in the manager fork.
- Label tests clearly as mocked, real-cache, or real end-to-end. The real tiers fail rather than skip when a service is down.
- After e2e runs, check that no containers or images were leaked (`docker ps -a`, `docker images`).
- Docs should be short and plain. Don't put people's names in docs.
