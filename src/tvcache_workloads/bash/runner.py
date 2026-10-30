"""Runs saved Bash command sequences (rollouts) through TVCache.

Each rollout gets its own TvcacheToolExecutor and submits the *cumulative*
history at every step, as tvclient requires. Commands come from a trace file
rather than an LLM, so runs are reproducible.
"""

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Type, Union

from tvclient import TvcacheToolExecutor

from tvcache_workloads.bash.sandbox import TerminalBenchSandbox
from tvcache_workloads.bash.tool_call import BashToolCall


@dataclass
class StepRecord:
    index: int
    command: str
    observation: str
    # True when the value came from the cache with no sandbox execution.
    cache_hit: bool
    # Calls executed to answer this step: 0 on a hit, 1 for a plain miss,
    # more when a stale or restored sandbox replayed missing history first.
    executions: int
    seconds: float


@dataclass
class RolloutRecord:
    name: str
    task_id: str
    steps: List[StepRecord] = field(default_factory=list)
    test_result: Optional[str] = None
    test_cache_hit: Optional[bool] = None
    test_executions: Optional[int] = None
    seconds: float = 0.0
    # Sandbox lifecycle events (start/restore/snapshot/execute/test/stop).
    sandbox_events: List[dict] = field(default_factory=list)

    @property
    def test_passed(self) -> Optional[bool]:
        if self.test_result is None or self.test_result.startswith("Failed to execute"):
            return None
        return json.loads(self.test_result)["passed"]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self) | {"test_passed": self.test_passed}


def load_trace_file(path: Union[str, Path]) -> dict:
    """Reads a trace file: {"task": ..., "rollouts": [{"name", "commands", ...}]}."""
    trace = json.loads(Path(path).read_text())
    if "task" not in trace or not isinstance(trace.get("rollouts"), list):
        raise ValueError(f"{path}: expected keys 'task' and 'rollouts'")
    for rollout in trace["rollouts"]:
        if not rollout.get("name") or not isinstance(rollout.get("commands"), list):
            raise ValueError(f"{path}: every rollout needs 'name' and a 'commands' list")
    return trace


def _as_call(command: Union[str, BashToolCall]) -> BashToolCall:
    return command if isinstance(command, BashToolCall) else BashToolCall(command)


async def run_rollout(
    name: str,
    commands: Sequence[Union[str, BashToolCall]],
    *,
    sandbox_class: Type[TerminalBenchSandbox],
    task_id: str,
    cache_host: str = "127.0.0.1",
    cache_port: int = 7878,
    run_test: bool = True,
    snapshot_threshold_seconds: Optional[float] = None,
    embed_tool_calls: bool = False,
) -> RolloutRecord:
    """Runs one rollout and returns what each step observed and cost.

    snapshot_threshold_seconds=None uses tvclient's adaptive policy (snapshot
    once un-checkpointed runtime exceeds ~2x the measured fork cost). A small
    fixed value snapshots after nearly every stateful command, which makes
    restore paths deterministic for demos and tests.

    embed_tool_calls=True makes tvclient embed each executed stateful command
    and store the vector with it in the cache. It needs tvclient's embedding
    support (tvclient PR #2 with tvcache-rust PR #1) and an embedder set with
    tvclient.set_embedder() in this process.
    """
    record = RolloutRecord(name=name, task_id=task_id)
    sandbox = sandbox_class.with_config(
        dataclasses.replace(sandbox_class.config, on_event=record.sandbox_events.append)
    )
    options = (
        {"adaptive": True}
        if snapshot_threshold_seconds is None
        else {"adaptive": False, "fixed_threshold_seconds": snapshot_threshold_seconds}
    )
    if embed_tool_calls:
        # Passed only when on, so the default also works with a tvclient that
        # predates embeddings.
        options["embed_tool_calls"] = True
    history: List[BashToolCall] = []
    rollout_start = time.perf_counter()

    async with TvcacheToolExecutor(
        sandbox, BashToolCall, task_id, host=cache_host, port=cache_port, **options
    ) as executor:
        for index, command in enumerate(commands):
            history.append(_as_call(command))
            executed_before = executor.total_executions
            started = time.perf_counter()
            observation = await executor.lookup_or_execute(history)
            executions = executor.total_executions - executed_before
            record.steps.append(
                StepRecord(
                    index=index,
                    command=history[-1].command,
                    observation=observation,
                    cache_hit=executions == 0,
                    executions=executions,
                    seconds=time.perf_counter() - started,
                )
            )

        if run_test:
            executed_before = executor.total_executions
            record.test_result = await executor.test(history)
            record.test_executions = executor.total_executions - executed_before
            record.test_cache_hit = record.test_executions == 0

    record.seconds = time.perf_counter() - rollout_start
    return record
