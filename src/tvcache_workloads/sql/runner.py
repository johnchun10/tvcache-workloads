"""Run saved SQL call traces through TVCache."""

import asyncio
import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Type, Union

from tvclient import TvcacheToolExecutor

from tvcache_workloads.sql.sandbox import SQLiteSandbox
from tvcache_workloads.sql.tool_call import SQLToolCall


@dataclass
class StepRecord:
    index: int
    call: dict
    observation: str
    cache_hit: bool
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
    sandbox_events: List[dict] = field(default_factory=list)

    @property
    def test_passed(self) -> Optional[bool]:
        if self.test_result is None or self.test_result.startswith("Failed to execute"):
            return None
        return json.loads(self.test_result)["passed"]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self) | {"test_passed": self.test_passed}


def load_trace_file(path: Union[str, Path]) -> dict:
    trace = json.loads(Path(path).read_text())
    if not isinstance(trace.get("task"), str) or not isinstance(trace.get("rollouts"), list):
        raise ValueError(f"{path}: expected string 'task' and list 'rollouts'")
    for rollout in trace["rollouts"]:
        if not rollout.get("name") or not isinstance(rollout.get("calls"), list):
            raise ValueError(f"{path}: every rollout needs a name and a calls list")
        for call in rollout["calls"]:
            SQLToolCall.from_dict(call)
    return trace


async def run_rollout(
    name: str,
    calls: Sequence[Union[dict, SQLToolCall]],
    *,
    task_id: str,
    sandbox_class: Type[SQLiteSandbox] = SQLiteSandbox,
    cache_host: str = "127.0.0.1",
    cache_port: int = 7878,
    snapshot_threshold_seconds: Optional[float] = None,
) -> RolloutRecord:
    record = RolloutRecord(name=name, task_id=task_id)
    sandbox = sandbox_class.with_config(
        dataclasses.replace(sandbox_class.config, on_event=record.sandbox_events.append)
    )
    options = (
        {"adaptive": True}
        if snapshot_threshold_seconds is None
        else {"adaptive": False, "fixed_threshold_seconds": snapshot_threshold_seconds}
    )
    history: List[SQLToolCall] = []
    rollout_start = time.perf_counter()

    async with TvcacheToolExecutor(
        sandbox, SQLToolCall, task_id, host=cache_host, port=cache_port, **options
    ) as executor:
        for index, value in enumerate(calls):
            call = value if isinstance(value, SQLToolCall) else SQLToolCall.from_dict(value)
            history.append(call)
            executions_before = executor.total_executions
            started = time.perf_counter()
            observation = await executor.lookup_or_execute(history)
            executions = executor.total_executions - executions_before
            record.steps.append(
                StepRecord(
                    index=index,
                    call=call.to_dict(),
                    observation=observation,
                    cache_hit=executions == 0,
                    executions=executions,
                    seconds=time.perf_counter() - started,
                )
            )

        executions_before = executor.total_executions
        record.test_result = await executor.test(history)
        record.test_executions = executor.total_executions - executions_before
        record.test_cache_hit = record.test_executions == 0

    record.seconds = time.perf_counter() - rollout_start
    return record


def run(coroutine):
    """Synchronous helper for trace scripts and tests."""
    return asyncio.run(coroutine)