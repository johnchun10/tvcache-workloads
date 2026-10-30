# Bash / Terminal-Bench workload

Runs Terminal-Bench tasks through TVCache. `tvclient` decides what needs to run;
`TerminalBenchSandbox` runs it in real containers through `container-manager-public`.

```text
runner -> tvclient -> tvcache-rust                (cached results, snapshot ids)
             |
             v  (only on a cache miss)
   TerminalBenchSandbox -> container-manager-public -> Docker
```

Code is in `src/tvcache_workloads/bash/`: `tool_call.py` (one command), `sandbox.py`
(the connector), `runner.py` (runs saved command lists), `observation.py` (extracts
each command's output). Example traces are in `bash/traces/`.

## Setup

Clone these side by side:

```text
tvcache/
  tvclient/                   github.com/abhivijay96/tvclient
  tvcache-rust/               github.com/abhivijay96/tvcache-rust
  container-manager-public/   github.com/johnchun10/container-manager-public (branch cold-snapshots)
  tvcache-workloads/          this repo
```

You need Docker, [uv](https://docs.astral.sh/uv/), and Rust. Then:

```bash
uv sync
bash/scripts/start_cache.sh      # terminal 1: cache server on :7878
bash/scripts/start_manager.sh    # terminal 2: container manager on :9901
```

## Run the example

```bash
uv run python bash/examples/run_trace.py bash/traces/hello-world.json --snapshot-threshold 0.001
```

It prints whether each step was a cache hit or executed, its output, and timings,
then deletes its snapshots. Useful options:

- `--snapshot-mode parked` works with the original manager, which lacks the snapshot endpoints.
- `--embed` stores command embeddings. This needs tvclient PR #2 and tvcache-rust PR #1, upgraded together, plus `sentence-transformers`.

## Tests

```bash
uv run pytest                        # fast, everything mocked
uv run pytest -m cache_integration   # real cache server, fake manager
uv run pytest -m e2e -s              # everything real (Docker)
```

## Using it from code

```python
Sandbox = TerminalBenchSandbox.with_config(SandboxConfig(manager_url="http://127.0.0.1:9901"))
history = []
async with TvcacheToolExecutor(Sandbox, BashToolCall, "hello-world") as ex:
    for cmd in ["cd /app", "echo 'Hello, world!' > hello.txt"]:
        history.append(BashToolCall(cmd))      # always pass the full history
        print(await ex.lookup_or_execute(history))
    print(await ex.test(history))              # '{"passed": true}'
```

## Things to know

- A command's output is what it printed to the terminal. Blank lines are dropped, and the exit code isn't available.
- Very long output (over about 2000 lines) can't be captured. It's returned as `Failed to execute`, and that result is never cached.
- Every command is assumed to change state unless created with `mutates=False`.
- Snapshots keep files, the current directory, and exported variables. They don't keep running programs or unexported variables. See [snapshot design](docs/snapshot-design.md).
- Environment comparison isn't implemented here; `TerminalBenchSandbox` has no `equals`.
