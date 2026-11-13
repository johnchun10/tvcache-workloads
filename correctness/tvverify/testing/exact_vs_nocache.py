"""P2 / P2b: the cached executor must serve exactly what direct execution
would, and a node's value must come from the same run as its snapshot."""

import asyncio
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from tvclient import TvcacheToolExecutor, TVCacheClient
from tvclient.tool_call import ToolCall

from tvverify.adapters.base import TEST_KEY, WorkloadAdapter
from tvverify.adapters.mock_fs import ToyCall, ToyFsSandbox, rollout_set
from tvverify.adapters.toyfs_adapter import ToyFsAdapter
from tvverify.model.dump import plain_task
from tvverify.testing.nocache import NoCacheToolExecutor


@dataclass
class Mismatch:
    task: str
    rollout: int
    step: int
    calls: str
    cached: str
    expected: str

    def __str__(self) -> str:
        return (
            f"{self.task} rollout {self.rollout} step {self.step}: cached {self.cached!r} "
            f"!= expected {self.expected!r} for {self.calls}"
        )


async def _one_rollout(
    port, adapter: WorkloadAdapter, task, rollout_idx, calls, threshold, observer=None
) -> List[Mismatch]:
    """One rollout through the cached executor and a cache-free executor, call
    by call. Outputs are judged by the adapter's comparators, so noise
    normalizers apply."""
    mismatches: List[Mismatch] = []

    def same(key: str, got: str, want: str) -> bool:
        return adapter.comparator_for(key).compare(got, want, key=key).equal

    # observer= only when asked for: an unmodified tvclient does not take it
    extra = {"observer": observer} if observer is not None else {}
    cached = TvcacheToolExecutor(
        adapter.sandbox_class, adapter.tool_call_class, task, port=port, adaptive=False,
        fixed_threshold_seconds=threshold, **extra,
    )
    plain = NoCacheToolExecutor(adapter.sandbox_class, adapter.tool_call_class, task)
    async with cached, plain:
        for i in range(len(calls)):
            got = await cached.lookup_or_execute(calls[: i + 1])
            want = await plain.lookup_or_execute(calls[: i + 1])
            if not same(calls[i].serialize(), got, want):
                mismatches.append(Mismatch(task, rollout_idx, i, repr(calls[: i + 1]), got, want))
        got = await cached.test(calls)
        want = await plain.test(calls)
        if not same(TEST_KEY, got, want):
            mismatches.append(Mismatch(task, rollout_idx, len(calls), repr(calls) + "+TEST", got, want))
    return mismatches


async def run_rollouts_vs_nocache(
    port: int,
    adapter: WorkloadAdapter,
    task: str,
    rollouts: Sequence[Sequence[ToolCall]],
    *,
    threshold: float = 1e-9,
    concurrent: bool = False,
    observer=None,
) -> List[Mismatch]:
    """The generic P2 check: any workload's rollouts through the cache versus
    direct execution, plus a check that no sandbox lease leaked."""
    mismatches: List[Mismatch] = []
    if concurrent:
        results = await asyncio.gather(
            *[_one_rollout(port, adapter, task, i, list(c), threshold, observer) for i, c in enumerate(rollouts)]
        )
        for r in results:
            mismatches += r
    else:
        for i, calls in enumerate(rollouts):
            mismatches += await _one_rollout(port, adapter, task, i, list(calls), threshold, observer)

    async with TVCacheClient(port=port) as client:
        refs = (await client.dump_all()).get(task, {}).get("sandbox_references", {})
    if refs:
        mismatches.append(Mismatch(task, -1, -1, "end of run", f"leaked leases {refs}", "none"))
    return mismatches


async def run_exact_vs_nocache(
    port: int,
    seed: int,
    *,
    rollouts: int = 6,
    max_len: int = 5,
    concurrent: bool = False,
    hidden: bool = False,
    observer=None,
) -> List[Mismatch]:
    """The toy-environment P2 run. Noise is excluded by construction: with
    deterministic tools any mismatch is a cache bug."""
    rng = random.Random(seed)
    ToyFsSandbox.reseed(seed)
    task = f"p2-{seed}"
    threshold = rng.choice([1e-9, 0.001, 1000.0])
    sets = rollout_set(rng, rollouts, max_len, hidden=hidden)

    return await run_rollouts_vs_nocache(
        port, ToyFsAdapter(), task, sets, threshold=threshold, concurrent=concurrent, observer=observer
    )


def _collect(node: dict, path: List[str], out: list) -> None:
    for key, child in node["children"].items():
        out.append((path + [key], child))
        _collect(child, path + [key], out)


async def run_value_snapshot_coherence(port: int, trials: int = 10) -> List[str]:
    """P2b: two rollouts race to cache the same noisy stateful call. Whatever
    value the cache keeps, the snapshot next to it must be the run that
    produced it (the file in the snapshot holds the same random token)."""
    problems: List[str] = []
    for trial in range(trials):
        task = f"p2b-{trial}"
        call = ToyCall("stamp", "f")
        ToyFsSandbox.reseed(trial)
        snapshots_before = set(ToyFsSandbox.snapshots)

        async def rollout() -> None:
            ex = TvcacheToolExecutor(
                ToyFsSandbox, ToyCall, task, port=port, adaptive=False, fixed_threshold_seconds=1e-9
            )
            async with ex:
                await ex.lookup_or_execute([call])

        await asyncio.gather(rollout(), rollout())

        async with TVCacheClient(port=port) as client:
            dump = (await client.dump_all())[task]
        nodes: list = []
        _collect(plain_task(dump)["root"], [], nodes)
        stored_ids = {n["sandbox_id"] for _, n in nodes if n["sandbox_id"]}
        for path, node in nodes:
            if node["sandbox_id"] is None:
                continue
            snapshot = ToyFsSandbox.snapshots.get(node["sandbox_id"])
            in_snapshot = snapshot.files.get("/f") if snapshot else None
            if in_snapshot != node["value"]:
                problems.append(
                    f"trial {trial}: node value {node['value']!r} but its snapshot holds {in_snapshot!r}"
                )
        leaked = (set(ToyFsSandbox.snapshots) - snapshots_before) - stored_ids
        if leaked:
            problems.append(f"trial {trial}: orphan snapshots were not discarded: {sorted(leaked)}")
    return problems
