"""Run a saved Bash trace through TVCache against real Terminal-Bench containers.

Needs a running tvcache-rust server and container-manager-public (see
bash/README.md). Example:

    uv run python bash/examples/run_trace.py bash/traces/hello-world.json \
        --namespace demo1 --snapshot-threshold 0.001 --output results/demo1.json

--embed stores an embedding of every executed command in the cache. It needs
sentence-transformers (and transformers >= 5.19 for embeddinggemma2).
"""

import argparse
import asyncio
import json
import sys
import uuid
from collections import Counter
from pathlib import Path

from tvcache_workloads.bash import (
    SandboxConfig,
    TerminalBenchSandbox,
    load_trace_file,
    run_rollout,
)
from tvcache_workloads.bash.cache_admin import release_task_snapshots
from tvcache_workloads.bash.sandbox import NAMESPACE_SEPARATOR


def _load_embedder(name: str) -> None:
    from tvclient import set_embedder

    from tvcache_workloads.bash.embedding import CommandEmbedder

    embedder = CommandEmbedder(name)
    # Load now: the executor only logs a warning if embedding fails later.
    print(f"loaded {embedder.model_name} (dimension {len(embedder.encode('warm up'))})")
    set_embedder(embedder)


def _summarize(record) -> None:
    print(f"\n== rollout {record.name!r} ({record.seconds:.1f}s)")
    for step in record.steps:
        status = "HIT " if step.cache_hit else f"EXEC x{step.executions}"
        observation = step.observation.replace("\n", "\\n")
        print(f"  [{status}] {step.seconds:6.2f}s  $ {step.command}")
        print(f"           -> {observation[:100]!r}")
    if record.test_result is not None:
        status = "HIT " if record.test_cache_hit else f"EXEC x{record.test_executions}"
        print(f"  [{status}] test -> {record.test_result}")
    ops = Counter(event["op"] for event in record.sandbox_events)
    costs = {
        op: round(sum(e["seconds"] for e in record.sandbox_events if e["op"] == op), 2)
        for op in ops
    }
    print(f"  sandbox ops: {dict(ops)}  seconds: {costs}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace", type=Path)
    parser.add_argument("--namespace", default=None,
                        help="cache namespace; default is a fresh random one, so the run starts uncached")
    parser.add_argument("--manager-url", default="http://127.0.0.1:9901")
    parser.add_argument("--cache-host", default="127.0.0.1")
    parser.add_argument("--cache-port", type=int, default=7878)
    parser.add_argument("--snapshot-threshold", type=float, default=None,
                        help="fixed snapshot threshold in seconds; omit for tvclient's adaptive policy")
    parser.add_argument("--command-timeout", type=float, default=120.0)
    parser.add_argument("--snapshot-mode", choices=["cold", "parked"], default="cold",
                        help="cold needs our fork of the manager (main); parked works with any manager")
    parser.add_argument("--keep-snapshots", action="store_true",
                        help="keep the snapshots after the run instead of deleting them")
    parser.add_argument("--output", type=Path, default=None, help="write per-step JSON results here")
    parser.add_argument("--embed", action="store_true",
                        help="store an embedding of each executed command")
    parser.add_argument("--embedding-model", default="minilm",
                        help="minilm, embeddinggemma2 or a sentence-transformers model name")
    args = parser.parse_args()

    trace = load_trace_file(args.trace)
    if args.embed:
        _load_embedder(args.embedding_model)
    namespace = args.namespace or uuid.uuid4().hex[:8]
    task_id = f"{trace['task']}{NAMESPACE_SEPARATOR}{namespace}"
    sandbox_class = TerminalBenchSandbox.with_config(
        SandboxConfig(
            manager_url=args.manager_url,
            command_timeout_sec=args.command_timeout,
            snapshot_mode=args.snapshot_mode,
        )
    )
    print(f"task {trace['task']!r}, cache task id {task_id!r}, snapshot mode {args.snapshot_mode}")

    records = []
    mismatches = []
    try:
        for rollout in trace["rollouts"]:
            record = await run_rollout(
                rollout["name"],
                rollout["commands"],
                sandbox_class=sandbox_class,
                task_id=task_id,
                cache_host=args.cache_host,
                cache_port=args.cache_port,
                snapshot_threshold_seconds=args.snapshot_threshold,
                embed_tool_calls=args.embed,
            )
            records.append(record)
            _summarize(record)
            expected = rollout.get("expect_test_passed")
            if expected is not None and record.test_passed is not expected:
                mismatches.append(f"{record.name}: expected passed={expected}, got {record.test_result}")
    finally:
        if not args.keep_snapshots:
            released = await release_task_snapshots(
                task_id, sandbox_class, args.cache_host, args.cache_port
            )
            print(f"\nreleased {len(released)} snapshot(s)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {
                "task_id": task_id,
                "trace": str(args.trace),
                "snapshot_mode": args.snapshot_mode,
                "embedding_model": args.embedding_model if args.embed else None,
                "rollouts": [r.to_dict() for r in records],
            },
            indent=2,
        ))
        print(f"wrote {args.output}")

    for mismatch in mismatches:
        print(f"UNEXPECTED TEST RESULT: {mismatch}", file=sys.stderr)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
