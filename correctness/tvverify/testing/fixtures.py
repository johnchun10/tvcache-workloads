"""A known cache trie for tests: toy rollouts run through the real cached
executor against a live server, then exported."""

from dataclasses import dataclass
from typing import List, Sequence

from tvclient import TvcacheToolExecutor

from tvverify.adapters.mock_fs import ToyCall, ToyFsSandbox
from tvverify.adapters.toyfs_adapter import ToyFsAdapter
from tvverify.model.graph import GraphExport, export_task_graph

C = ToyCall
wa_x, wb_y, cd_d, export1 = C("write", "a", "x"), C("write", "b", "y"), C("cd", "/d"), C("export", "K", "1")
read_a, read_b, ls, printenv_k = C("read", "a"), C("read", "b"), C("ls"), C("printenv", "K")

# Shared prefixes, stateful calls with snapshots, stateless leaves; every
# rollout also ends with the task's test via executor.test().
ROLLOUTS: List[List[ToyCall]] = [
    [wa_x, wb_y, read_a, ls],
    [wa_x, wb_y, read_b],
    [wa_x, cd_d, ls],
    [wb_y, read_b],
    [wa_x, export1, printenv_k],
]


@dataclass
class ExactFixture:
    graph: GraphExport
    adapter: ToyFsAdapter

    def snapshot_exists(self, sandbox_id: str) -> bool:
        return sandbox_id in ToyFsSandbox.snapshots


async def build_exact_fixture(port: int, rollouts: Sequence[Sequence[ToyCall]] = ROLLOUTS) -> ExactFixture:
    """Runs the rollouts (snapshotting after every stateful call) and exports
    the trie. Clears ALL ToyFs sandboxes and snapshots first."""
    ToyFsSandbox.reset()
    adapter = ToyFsAdapter()
    for calls in rollouts:
        ex = TvcacheToolExecutor(ToyFsSandbox, ToyCall, "fixture", port=port, adaptive=False, fixed_threshold_seconds=1e-9)
        async with ex:
            for k in range(len(calls)):
                await ex.lookup_or_execute(list(calls[: k + 1]))
            await ex.test(list(calls))
    graph = await export_task_graph(port, "fixture", adapter, ToyFsSandbox.root_image_id)
    return ExactFixture(graph, adapter)
