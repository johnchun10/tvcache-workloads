"""ToolAgentLoop plus the end of a Terminal-Bench trajectory: run the task's
tests on the final state, close the TVCache session, and record the result
in extra_fields for the reward function (reward.py).

Registered by configs/agent_loops.yaml under the name `tvcache_tool_agent`.
"""

from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

from tvcache_workloads.bash.verl.reward import CALLS_KEY, TEST_KEY
from tvcache_workloads.bash.verl.session import SESSIONS


class TvcacheToolAgentLoop(ToolAgentLoop):
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        output = await super().run(sampling_params, **kwargs)
        session = SESSIONS.pop(output.extra_fields)
        if session is None:  # no well-formed bash call in the whole trajectory
            output.extra_fields[CALLS_KEY] = 0
            return output
        output.extra_fields[CALLS_KEY] = len(session.history)
        output.extra_fields["tb_failed_calls"] = session.failed_calls
        output.extra_fields[TEST_KEY] = await session.finish()
        return output
