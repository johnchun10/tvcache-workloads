"""ToolAgentLoop plus the end of a Terminal-Bench trajectory: run the task's
tests on the final state, close the TVCache session, and record the result
in extra_fields for the reward function (reward.py).

Registered by configs/agent_loops.yaml under the name `tvcache_tool_agent`.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

from tvcache_workloads.bash.verl.reward import CALLS_KEY, TEST_KEY
from tvcache_workloads.bash.verl.session import SESSIONS


def record_rollout(record: dict) -> None:
    """Appends one line per finished trajectory to <TVCACHE_TRACE_DIR>/rollouts-<pid>.jsonl
    (stats.py counts truncated rollouts from these)."""
    trace_dir = os.environ.get("TVCACHE_TRACE_DIR")
    if not trace_dir:
        return
    path = Path(trace_dir) / f"rollouts-{os.getpid()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


class TvcacheToolAgentLoop(ToolAgentLoop):
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        started = time.time()
        output = await super().run(sampling_params, **kwargs)
        session = SESSIONS.pop(output.extra_fields)
        if session is None:  # no well-formed bash call in the whole trajectory
            output.extra_fields[CALLS_KEY] = 0
        else:
            output.extra_fields[CALLS_KEY] = len(session.history)
            output.extra_fields["tb_failed_calls"] = session.failed_calls
            output.extra_fields[TEST_KEY] = await session.finish()
        tools_kwargs = kwargs.get("tools_kwargs") or {}
        task = (tools_kwargs.get("bash", {}).get("create_kwargs", {}) or {}).get("task_id")
        record_rollout({
            "t_start": started,
            "t_end": time.time(),
            "task": task if session is None else session.task,
            "rollout": None if session is None else getattr(session.executor, "rollout_id", None),
            "num_turns": getattr(output, "num_turns", None),
            "response_len": len(getattr(output, "response_ids", None) or []),
            "max_response_len": getattr(self, "response_length", None),
            "max_assistant_turns": getattr(self, "max_assistant_turns", None),
            # Stopped by the response-length cap (veRL truncates to it).
            "length_truncated": bool(getattr(self, "response_length", None))
            and len(getattr(output, "response_ids", None) or []) >= self.response_length,
            "calls": output.extra_fields.get(CALLS_KEY, 0),
            "failed_calls": output.extra_fields.get("tb_failed_calls", 0),
            "test": output.extra_fields.get(TEST_KEY),
        })
        return output
