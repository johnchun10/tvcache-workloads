"""P1: random protocol operations against the Rust server and RefTrie.

Both must answer every request identically and end with identical state."""

import random
from typing import List

from tvclient import TVCacheClient
from tvclient.protocol import SandboxEvictionRecord

from tvverify.model.ref_trie import RefTrie

KEYS = ["a", "b", "c"]
VALUES = ["v0", "v1", "v2", None]


async def run_ref_vs_server(port: int, seed: int, steps: int = 250) -> List[str]:
    rng = random.Random(seed)
    task = f"p1-{seed}"
    model = RefTrie()
    problems: List[str] = []
    leases: List[str] = []
    all_sandboxes: List[tuple] = []

    def check(step: int, what: str, got: object, want: object) -> None:
        if got != want:
            problems.append(f"seed {seed} step {step}: {what}: server {got!r} != model {want!r}")

    async with TVCacheClient(port=port) as client:
        for step in range(steps):
            roll = rng.random()
            path = [rng.choice(KEYS) for _ in range(rng.randint(1, 4))]

            if roll < 0.45:
                value = rng.choice(VALUES)
                sandbox = f"s{seed}-{step}" if rng.random() < 0.5 else None
                runtime = rng.randint(0, 50)
                got = await client.insert(task, path, value, sandbox, runtime=runtime)
                want = model.insert(path, value, sandbox, runtime)
                check(step, f"insert {path} {value!r} {sandbox!r}", (got.value_inserted, got.sandbox_inserted), want)
                if want[1]:
                    all_sandboxes.append((list(path), sandbox))
            elif roll < 0.75:
                got = await client.find_unmanaged(task, path)
                want = model.find(path)
                check(step, f"find {path}", (got.value, got.sandbox_path, got.sandbox_id), (want.value, want.sandbox_path, want.sandbox_id))
                if got.sandbox_id is not None:
                    leases.append(got.sandbox_id)
            elif roll < 0.88 and leases:
                sandbox = leases.pop(rng.randrange(len(leases)))
                got = await client.unref_sandbox(task, sandbox)
                check(step, f"unref {sandbox}", got, model.unref(sandbox))
            elif roll < 0.95 and all_sandboxes:
                node_path, sandbox = rng.choice(all_sandboxes)
                if rng.random() < 0.2:
                    sandbox = "wrong-id"  # mismatched id: victim recorded, node untouched
                await client.mark_sandboxes_evicted([SandboxEvictionRecord(task, node_path, sandbox)])
                model.mark_evicted(node_path, sandbox)
            else:
                got = sorted(s.sandbox_id for s in await client.drain_reclaimable_sandboxes() if s.task_id == task)
                check(step, "drain", got, model.drain())

        dump = await client.dump_all()
        if task in dump:
            problems += [f"seed {seed} final: {p}" for p in model.diff_against_dump(dump[task])]
        elif model.counters()["number_of_nodes"]:
            problems.append(f"seed {seed} final: task missing from dump")
    return problems
