# Bash / Terminal-Bench workload

Runs Terminal-Bench tasks through TVCache: `tvclient` decides what to execute,
and this package executes it in real containers via `container-manager-public`.

```text
runner ──> tvclient ──TCP──> tvcache-rust          (cached results + snapshot ids)
              │
              │ only when something must actually run
              v
   TerminalBenchSandbox ──HTTP──> container-manager-public ──> Docker
```

| Piece | File |
|---|---|
| `BashToolCall`: one command, the cache-key unit | `src/tvcache_workloads/bash/tool_call.py` |
| `TerminalBenchSandbox`: tvclient `Sandbox` over HTTP | `src/tvcache_workloads/bash/sandbox.py` |
| HTTP client with bounded requests and typed errors | `src/tvcache_workloads/bash/manager_client.py` |
| Per-command output extraction | `src/tvcache_workloads/bash/observation.py` |
| Runner for saved command sequences | `src/tvcache_workloads/bash/runner.py` |
| Snapshot cleanup through cache eviction | `src/tvcache_workloads/bash/cache_admin.py` |
| Controlled traces for the `hello-world` task | `bash/traces/hello-world.json` |
| Runnable example | `bash/examples/run_trace.py` |

Design notes: [snapshot handling](docs/snapshot-design.md) ·
[environment comparison boundary](docs/comparison-interface.md) ·
[verification record](docs/verification.md)

## Setup (once)

Expected layout, with each repository cloned separately:

```text
tvcache/
  tvclient/                   # github.com/abhivijay96/tvclient
  tvcache-rust/               # github.com/abhivijay96/tvcache-rust
  container-manager-public/   # our fork: github.com/johnchun10/container-manager-public,
                              #   branch cold-snapshots
  tvcache-workloads/          # this folder
```

The `cold-snapshots` branch adds `/snapshot`, `/restore`, and `/delete_snapshot`
to the manager (see [snapshot handling](docs/snapshot-design.md)). With the
original manager, pass `--snapshot-mode parked`, or set
`SandboxConfig(snapshot_mode="parked")`.

Requirements: Docker running (Docker Desktop on macOS), [uv](https://docs.astral.sh/uv/),
and a Rust toolchain. The manager needs Python ≥ 3.12; uv downloads it.

```bash
cd tvcache-workloads
uv sync            # creates .venv with tvclient (editable, from ../tvclient) and httpx
```

## Run the services (two terminals)

```bash
bash/scripts/start_cache.sh      # tvcache-rust on :7878 (eviction off)
bash/scripts/start_manager.sh    # container-manager-public on :9901 with ENABLE_CACHING=1
```

`start_manager.sh` uses `uv sync --frozen`, so the manager's `uv.lock` is never
rewritten. The first `/start` of a task builds its image, which takes longer.

## Run the example

```bash
uv run python bash/examples/run_trace.py bash/traces/hello-world.json \
    --snapshot-threshold 0.001 --output results/run.json
```

Each run uses a fresh cache namespace (`hello-world::<random>`), so it starts
uncached. Pass `--namespace NAME` to reuse one. It prints per-step HIT/EXEC,
observations, test results, and snapshot/restore/test costs. At the end it
deletes the run's snapshots (`--keep-snapshots` to keep them).
Omit `--snapshot-threshold` to use tvclient's adaptive policy.

### Embeddings (for the semantic cache)

`--embed` (or `run_rollout(..., embed_tool_calls=True)`) stores an embedding
of each executed command in the cache. It requires the embedding PRs, **tvclient #2
together with tvcache-rust #1**: client and server must be upgraded together,
because a mismatched pair breaks the connection or hangs. It also needs
`uv pip install sentence-transformers "transformers<5"`. By default it uses the
same model as the tvcache-rust embedding demo (`jinaai/jina-embeddings-v2-base-code`); change it with
`--embedding-model`. Without `--embed`, everything works on tvclient with or
without the PRs.

## Tests

| Command | Real | Faked |
|---|---|---|
| `uv run pytest` | nothing | manager and HTTP (test double) |
| `uv run pytest -m cache_integration` | tvcache-rust + tvclient | manager (test double) |
| `uv run pytest -m e2e -s` | everything, with real Docker containers | nothing |

Real-service tiers **fail** (rather than skip) when a service is down.
Override endpoints with `TVCACHE_HOST`, `TVCACHE_PORT`, `TB_MANAGER_URL`.

## Using it from code

```python
from tvclient import TvcacheToolExecutor
from tvcache_workloads.bash import BashToolCall, SandboxConfig, TerminalBenchSandbox

Sandbox = TerminalBenchSandbox.with_config(SandboxConfig(manager_url="http://127.0.0.1:9901"))
history = []
async with TvcacheToolExecutor(Sandbox, BashToolCall, "hello-world") as ex:
    for cmd in ["cd /app", "echo 'Hello, world!' > hello.txt"]:
        history.append(BashToolCall(cmd))          # always pass the cumulative history
        print(await ex.lookup_or_execute(history))
    print(await ex.test(history))                  # '{"passed": true}'
```

A task id `"<task>::<namespace>"` runs Terminal-Bench task `<task>` under its
own cache tree.

## What an observation is

The text the command printed to the terminal: stdout and stderr interleaved,
without the prompt or the command echo, and with tmux wrapping at 160 columns.
Cached and uncached runs return the same format, because the cached value
*is* the stored uncached observation.

Known differences from raw stdout, all caused by the manager:

- Blank lines are dropped.
- The **exit status is not available**: the manager appends `; tmux wait -S done`.
- Output longer than tmux's scrollback (about 2000 rows) can't be located. It is
  returned as `Failed to execute: …`, which tvclient never caches.

Each command runs as three blocking `/execute` calls (BEGIN marker, the
command unchanged, END marker) plus one `/get_pane_logs` call, about 0.3 s
overhead in total. The manager joins multi-line commands with ` && `, so
heredocs and multi-line scripts don't keep their Bash meaning.

## Errors

| Situation | Behavior |
|---|---|
| Command timeout (HTTP 504), manager unreachable, or HTTP error during a command | Raises. tvclient discards that sandbox, since the shell's state is unknown. |
| Output couldn't be extracted | Returns `Failed to execute: …`: the rollout continues, nothing is cached |
| Test run broke (manager error) | Returns `Failed to execute tests: …`, not cached |
| Executing in a snapshot handle, or "restoring" a live id | `SnapshotContractError` |
| Malformed manager response | `ManagerProtocolError` |

`TestToolCall` maps to `/test`, which returns `{"passed": bool}` as a JSON
string. `/test` installs `uv` and pytest in the container and needs network
access.

## Scope

- Commands are treated as state-mutating unless constructed with `mutates=False`.
  There's no automatic read-only detection.
- Snapshots are cold images by default (`parked` running copies as a fallback).
  Either way they keep files, cwd, and exported variables, but not processes or
  unexported shell state. See [snapshot-design.md](docs/snapshot-design.md).
- Out of scope: SQL, semantic-cache / graph contraction, environment
  comparison, JEV.
