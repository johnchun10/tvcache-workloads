"""Run saved SQL rollout traces through TVCache."""

import argparse
import asyncio
import json
import sys
import uuid
from collections import Counter
from pathlib import Path

from tvcache_workloads.cache_admin import release_task_snapshots
from tvcache_workloads.sql import SQLSandboxConfig, SQLiteSandbox, load_trace_file, run_rollout


def summarize(record) -> None:
    print(f"\n== rollout {record.name!r} ({record.seconds:.2f}s)")
    for step in record.steps:
        status = "HIT" if step.cache_hit else f"EXEC x{step.executions}"
        query = " ".join(step.call["query"].split())
        print(f"  [{status:8}] {step.seconds:6.3f}s  {query[:90]}")
        print(f"             -> {step.observation[:160]}")
    status = "HIT" if record.test_cache_hit else f"EXEC x{record.test_executions}"
    print(f"  [{status:8}] test passed={record.test_passed}: {record.test_result}")
    operations = Counter(event["op"] for event in record.sandbox_events)
    print(f"  sandbox operations: {dict(operations)}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--namespace", default=None, help="cache namespace; default is random")
    parser.add_argument("--cache-host", default="127.0.0.1")
    parser.add_argument("--cache-port", type=int, default=7878)
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--snapshot-threshold", type=float, default=0.005)
    parser.add_argument("--simulated-sql-latency", type=float, default=0.02)
    parser.add_argument("--keep-snapshots", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    trace = load_trace_file(args.trace)
    namespace = args.namespace or uuid.uuid4().hex[:8]
    task_id = f"{trace['task']}::{namespace}"
    config = SQLSandboxConfig(
        database_dir=args.snapshot_dir or SQLiteSandbox.config.database_dir,
        simulated_sql_latency_seconds=args.simulated_sql_latency,
    )
    sandbox_class = SQLiteSandbox.with_config(config)
    print(f"task {trace['task']!r}, cache task id {task_id!r}")

    records = []
    mismatches = []
    try:
        for rollout in trace["rollouts"]:
            record = await run_rollout(
                rollout["name"],
                rollout["calls"],
                sandbox_class=sandbox_class,
                task_id=task_id,
                cache_host=args.cache_host,
                cache_port=args.cache_port,
                snapshot_threshold_seconds=args.snapshot_threshold,
            )
            records.append(record)
            summarize(record)
            expected = rollout.get("expect_test_passed")
            if expected is not None and record.test_passed is not expected:
                mismatches.append(
                    f"{record.name}: expected passed={expected}, got {record.test_result}"
                )
    finally:
        if not args.keep_snapshots:
            released = await release_task_snapshots(
                task_id, sandbox_class, args.cache_host, args.cache_port
            )
            print(f"\nreleased {len(released)} database snapshot(s)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "trace": str(args.trace),
                    "rollouts": [record.to_dict() for record in records],
                },
                indent=2,
            )
        )
        print(f"wrote {args.output}")

    for mismatch in mismatches:
        print(f"UNEXPECTED TEST RESULT: {mismatch}", file=sys.stderr)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))