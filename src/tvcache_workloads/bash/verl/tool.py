"""The `bash` tool for veRL's ToolAgentLoop: every command goes through TVCache.

Configured in configs/tools.yaml (class_name points here). The task comes
from the dataset row's tools_kwargs (bash.create_kwargs.task_id); the
trajectory's session (executor and history) comes from SESSIONS. Use it with
TvcacheToolAgentLoop, which runs the tests and closes the session at the end.
"""

from typing import Any, Optional

from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema, ToolResponse

from tvcache_workloads.bash.verl.session import SESSIONS, CacheSettings

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run one shell command in the task's Linux container and return its terminal output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command to run."}},
            "required": ["command"],
        },
    },
}


class TvcacheBashTool(BaseTool):
    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        super().__init__(config, tool_schema or OpenAIFunctionToolSchema(**TOOL_SCHEMA))
        self.settings = CacheSettings.from_config(config or {})

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs["agent_data"]
        task = agent_data.tools_kwargs.get(self.name, {}).get("create_kwargs", {}).get("task_id")
        command = parameters.get("command")
        if not task:
            return ToolResponse(text="error: this trajectory has no task_id (dataset tools_kwargs)"), 0.0, {}
        if not isinstance(command, str) or not command.strip():
            return ToolResponse(text="error: `command` must be a non-empty string"), 0.0, {}
        session = SESSIONS.for_trajectory(agent_data.extra_fields, task, self.settings)
        output = await session.run(command)
        limit = self.settings.max_output_chars
        if len(output) > limit:
            output = output[: limit // 2] + "\n...[output truncated]...\n" + output[-limit // 2 :]
        return ToolResponse(text=output), 0.0, {}
