"""DirectExecutor: runs a trajectory's commands straight in a sandbox, with
no cache. The baseline for "with vs without TVCache" training runs.

It has the three methods RolloutSession uses on tvclient's
TvcacheToolExecutor (lookup_or_execute, test, close), so the session, tool
and agent loop stay the same; make_session picks it when
CacheSettings.use_cache is False (TVCACHE_USE_CACHE=0).

Like tvclient's executor, a raised execution discards the sandbox, since its
shell state is unknown. The next call starts a fresh sandbox and replays the
whole history first, which is what an uncached rollout has to do too.
"""

import logging
from typing import List, Optional

from tvclient import Sandbox, TestToolCall, ToolCall


class DirectExecutor:
    def __init__(self, sandbox_class: type, task_name: str):
        self.sandbox_class = sandbox_class
        self.task_name = task_name
        self.sandbox: Optional[Sandbox] = None
        self.depth = 0  # how many calls of the history the live sandbox has run
        self.total_calls = 0
        self.total_executions = 0
        self.logger = logging.getLogger(__name__)

    async def lookup_or_execute(self, tool_commands: List[ToolCall]) -> str:
        """Executes the last call after the ones before it. Nothing is cached."""
        self.total_calls += 1
        history, current = tool_commands[:-1], tool_commands[-1]
        if self.sandbox is None or self.depth > len(history):
            await self._discard()
            self.sandbox = self.sandbox_class(task_name=self.task_name)
            await self.sandbox.start()
            self.depth = 0
        value = ""
        for call in list(history[self.depth :]) + [current]:
            value = await self._execute(call)
            self.depth += 1
        return value

    async def test(self, tool_call_history: List[ToolCall]) -> str:
        """Runs the task's tests on the final state, then stops the sandbox."""
        try:
            return await self.lookup_or_execute(list(tool_call_history) + [TestToolCall("")])
        finally:
            await self._discard()

    async def close(self) -> None:
        await self._discard()

    async def _execute(self, call: ToolCall) -> str:
        self.total_executions += 1
        try:
            return await self.sandbox.execute(call)
        except Exception:
            self.logger.debug("execution raised; discarding sandbox %s", self.sandbox.get_id())
            await self._discard()
            raise

    async def _discard(self) -> None:
        sandbox, self.sandbox, self.depth = self.sandbox, None, 0
        if sandbox is None:
            return
        try:
            await sandbox.stop()
        except Exception as error:  # the run goes on; the manager may leak one container
            self.logger.warning("could not stop sandbox %s: %s", sandbox.get_id(), error)
