# tvverify: correctness framework for TVCache

Checks that TVCache serves what direct execution would: every value and
snapshot in a task's cache trie must match a fresh run of the same calls.

## Run it

Expects `tvclient` and `tvcache-rust` checked out next to `tvcache-workloads`:

```bash
cd tvcache-workloads/correctness
uv venv .venv && uv pip install --python .venv/bin/python pytest -e ../../tvclient -e .

.venv/bin/python -m tvverify selftest              # P1, P2, P2b below, ~10 s
.venv/bin/python -m tvverify selftest --seeds 200
.venv/bin/python -m pytest tests                   # about 50 tests; starts its own Rust servers
```

Needs `cargo`: the tests build and launch `tvcache-rust`, found at
`$TVCACHE_RUST_DIR` or in a `tvcache-rust` directory beside a parent directory.

### Unmodified or patched TVCache

tvverify runs against any `tvclient` and `tvcache-rust`: it uses whichever
`tvclient` is installed and whichever `tvcache-rust` it finds. On `main`
(unmodified) it catches bugs B1, B2 and B3: the tests written for them are
marked `known_bug` and report as expected failures (`xfail`), and
`selftest` fails P1 and P2b. With the fixes (branch `correctness-fixes` of
both repos) everything passes; set `TVVERIFY_PATCHED=1` to make the
`known_bug` tests required:

```bash
TVVERIFY_PATCHED=1 .venv/bin/python -m pytest tests
```

| Needs the patched repos | Why |
|---|---|
| `TraceRecorder` (`tvverify/trace`) | uses the executor's `observer=` hooks; its test is skipped otherwise |
| Several test servers at once | an unmodified server has no `--port` and always listens on 7878, so run one tvverify process at a time |

Everything else, including checking a cache trie by fresh-root replay
(`check --suite replay`), works on the unmodified code.

The bugs the `known_bug` tests catch on `main`:

| Bug | Where | What goes wrong |
|---|---|---|
| B1 | `tvclient` executor | A rollout can resume from a sandbox that already contains the current call, and runs it again on its own result: the cache stores a wrong value. |
| B2 | `tvcache-rust` insert | Values are last-writer-wins but snapshots first-writer-wins, so a node's value and its snapshot can come from different runs. |
| B3 | `tvclient` executor | A failed stateful call is never inserted but stays in the sandbox history, so every deeper insert in that rollout is silently refused. |

## What is checked

| Suite | Claim | Result when it holds |
|---|---|---|
| P1 | The Rust server agrees with an independent Python model (`RefTrie`) on random protocol operations | no diffs |
| P2 | The cached executor returns exactly what a cache-free executor returns, on random rollouts, sequential and concurrent | no mismatches |
| P2b | A node's cached value and its snapshot come from the same run | no incoherent nodes, no orphaned snapshots |

Exact TVCache is only expected to match direct execution when tools are
deterministic and snapshots capture all state (the paper's assumptions A1/A2).
Tests show what happens otherwise (`test_hidden_process_state_...`).

## Layout

- `tvverify/model`: `RefTrie` (reference semantics), `GraphExport` (the trie format the validators read, built from DUMP_ALL)
- `tvverify/validators`: `structural` (C4), `offline` (C1, C2a, C3, A2 against fresh-root replay)
- `tvverify/replay`: fresh-root replay engine, the independent oracle
- `tvverify/compare`: output comparators (exact, normalized, SQL result sets, test verdicts)
- `tvverify/digest`: layered state digests
- `tvverify/trace`: records what the cache did (`TraceRecorder`, attached as `observer=` on the executor)
- `tvverify/adapters`: the workload interface (`base.WorkloadAdapter`) and mock environments (`mock_list`, `mock_fs`)
- `tvverify/testing`: the suites (`sandbox_contract`, `exact_vs_nocache`, `ref_vs_server`), the test server launcher and fixtures

## Bring your own sandbox and workload

A *sandbox* is the environment calls run in; a *workload* is the set of tools, tasks
and rollouts you want checked. You supply both: the sandbox as a `Sandbox`
subclass, the workload as a `WorkloadAdapter` that points at it.

1. Copy `templates/my_workload.py` and fill in the TODOs. Required: a `ToolCall`
   (`to_dict`), a `Sandbox` (`start`, `fork`, `execute`, `stop`, and either
   `equals` or `state_digest`), and a `WorkloadAdapter` (`tool_call_from_key`,
   `tasks`, and `sample_rollouts` or `scripted_paths`). Everything else has a default.
2. Check the sandbox alone (no server needed):

   ```bash
   .venv/bin/python -m tvverify check --adapter my_workload:MyAdapter --suite contract
   ```

   It checks that `equals` notices state changes, that a forked snapshot restores
   to the source's state, that restored sandboxes are independent, and that
   calls round-trip through their cache keys. Methods you have not written yet
   show up as `not implemented: ...`.
3. Run everything: `... check --adapter my_workload:MyAdapter`
   - `cache`: rollouts through the cache return what direct execution returns.
   - `replay`: every cached node matches a fresh-root replay (C1, C2a, C3).
   - `--replays N` (default 2): a mismatch is **noise** if fresh replays of the
     same call disagree with each other, otherwise it is a **BUG**. Only bugs and
     contract failures give a non-zero exit code.

4. Check a trie someone else built, e.g. a training run's DUMP_ALL saved to a file:

   ```bash
   .venv/bin/python -m tvverify check-graph --adapter my_workload:MyAdapter --dump dump.json --json result.json
   .venv/bin/python -m tvverify report results/*.json > report.md
   ```

   Per task it replays `--max-replay-nodes` nodes (default 20): the most-served
   half (cache hits plus resumes from their snapshot) and a seeded random sample
   of the rest. Pass `--no-snapshots` when the dump's snapshots are gone; the
   snapshot check (C2a) is then skipped. A replay that cannot run (`Failed to
   execute`, or an exception) is **inconclusive**: reported, but no exit code.

`equals(other)` compares two live sandboxes (diff two filesystems, compare two
database dumps). `state_digest()` returns hashed components instead; it is
faster and says *what* differed, and the default `equals` is built on it. If
you implement neither, the snapshot state check (C2a) cannot run.

### Plugging in a workload

Subclass `tvverify.adapters.base.WorkloadAdapter` and set `sandbox_class` and
`tool_call_class`. Nothing else in the framework depends on the workload.

| Method | | What it is for |
|---|---|---|
| `tool_call_from_key(key)` | required | Rebuild a `ToolCall` from its cache key (`serialize()` output); replay uses it. |
| `tasks()` | required | The tasks to check; each gets its own cache trie. |
| `sample_rollouts(task_id, rng, count)` | required, or give `scripted_paths` | Call lists to run through the cache. Make them share prefixes. |
| `scripted_paths(task_id)` | alternative | Known-good call sequences; `sample_rollouts` draws from them. |
| `comparator_for(key)` | optional | How two outputs are judged equal. Default is exact; use a normalizing or semantic comparator (`tvverify.compare`) when outputs legitimately vary. |
| `tool_class(key)` | optional | Label for grouping results, e.g. `sql:select`. |
| `digest_policy(task_id)` / `probe_state(...)` | optional | What the state digest covers, for sandboxes that implement `state_digest`. |

`ToyFsAdapter` (`tvverify/adapters/toyfs_adapter.py`) is a complete worked example.
