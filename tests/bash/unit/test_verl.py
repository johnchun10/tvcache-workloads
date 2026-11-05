"""The veRL integration without veRL: the few veRL classes it touches are stubbed
with the signatures of veRL 0.9.1 (verl/tools/base_tool.py,
verl/experimental/agent_loop/tool_agent_loop.py), and TVCache is a fake executor."""

import asyncio
import json
import sys
import types

import pytest

pytest.importorskip("yaml")

from tvcache_workloads.bash.verl import reward  # noqa: E402
from tvcache_workloads.bash.verl.dataset import TOOL_NAME, task_rows  # noqa: E402
from tvcache_workloads.bash.verl.session import (  # noqa: E402
    SESSION_KEY,
    CacheSettings,
    RolloutSession,
    SessionRegistry,
)


class FakeExecutor:
    def __init__(self, test_output='{"passed": true}'):
        self.calls, self.closed, self.test_output = [], False, test_output

    async def lookup_or_execute(self, history):
        self.calls.append([c.command for c in history])
        if history[-1].command == "boom":
            raise TimeoutError("command timed out")
        return f"ran {history[-1].command}"

    async def test(self, history):
        return self.test_output

    async def close(self):
        self.closed = True


@pytest.fixture
def stub_verl(monkeypatch):
    """Minimal stand-ins for the veRL modules tool.py and agent_loop.py import."""

    class ToolResponse:
        def __init__(self, text=None):
            self.text = text

    class OpenAIFunctionToolSchema:
        def __init__(self, **schema):
            self.function = types.SimpleNamespace(name=schema["function"]["name"])

    class BaseTool:
        def __init__(self, config, tool_schema):
            self.config, self.tool_schema = config, tool_schema
            self.name = tool_schema.function.name

    class AgentLoopOutput:
        def __init__(self, extra_fields):
            self.extra_fields = extra_fields

    class ToolAgentLoop:
        script = []  # (tool, args) calls the fake model makes

        def __init__(self, tool):
            self.tool = tool

        async def run(self, sampling_params, **kwargs):
            agent_data = types.SimpleNamespace(tools_kwargs=kwargs["tools_kwargs"], extra_fields={})
            for args in self.script:
                await self.tool.execute("instance", args, agent_data=agent_data)
            return AgentLoopOutput(dict(agent_data.extra_fields))

    modules = {
        "verl": types.ModuleType("verl"),
        "verl.tools": types.ModuleType("verl.tools"),
        "verl.tools.base_tool": types.ModuleType("verl.tools.base_tool"),
        "verl.experimental": types.ModuleType("verl.experimental"),
        "verl.experimental.agent_loop": types.ModuleType("verl.experimental.agent_loop"),
        "verl.experimental.agent_loop.agent_loop": types.ModuleType("verl.experimental.agent_loop.agent_loop"),
        "verl.experimental.agent_loop.tool_agent_loop": types.ModuleType("verl.experimental.agent_loop.tool_agent_loop"),
    }
    modules["verl.tools.base_tool"].__dict__.update(BaseTool=BaseTool, OpenAIFunctionToolSchema=OpenAIFunctionToolSchema, ToolResponse=ToolResponse)
    modules["verl.experimental.agent_loop.agent_loop"].AgentLoopOutput = AgentLoopOutput
    modules["verl.experimental.agent_loop.tool_agent_loop"].ToolAgentLoop = ToolAgentLoop
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("tvcache_workloads.bash.verl.tool", "tvcache_workloads.bash.verl.agent_loop"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return ToolAgentLoop


def fake_registry(executors):
    def factory(task, settings):
        executor = FakeExecutor()
        executors.append((task, executor))
        return RolloutSession(task, executor)

    return SessionRegistry(factory)


def score(extra_info):
    return reward.compute_score("terminal_bench", "", "", extra_info)["score"]


def test_reward_follows_the_papers_scheme():
    assert score({}) == -1.0
    assert score({"tb_tool_calls": 0}) == -1.0
    assert score({"tb_tool_calls": 3, "tb_test": '{"passed": false}'}) == 0.0
    assert score({"tb_tool_calls": 3, "tb_test": "Failed to execute: boom"}) == 0.0
    assert score({"tb_tool_calls": 3, "tb_test": '{"passed": true}'}) == 1.0
    full = reward.compute_score("terminal_bench", "", "", {"tb_tool_calls": 3, "tb_failed_calls": 1, "tb_test": '{"passed": true}'})
    assert full == {"score": 1.0, "tool_calls": 3, "failed_calls": 1, "passed": 1.0}


def test_dataset_rows_carry_the_instruction_and_task_id(tmp_path):
    (tmp_path / "fix-git").mkdir()
    (tmp_path / "fix-git" / "task.yaml").write_text("instruction: |-\n  Fix the repo.\nauthor_name: x\n")
    (tmp_path / "not-a-task").mkdir()
    rows = task_rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["prompt"][-1] == {"role": "user", "content": "Fix the repo."}
    assert row["extra_info"]["tools_kwargs"] == {TOOL_NAME: {"create_kwargs": {"task_id": "fix-git"}}}


def test_settings_fall_back_to_environment(monkeypatch):
    monkeypatch.setenv("TVCACHE_PORT", "7999")
    monkeypatch.setenv("TVCACHE_NAMESPACE", "pilot1")
    settings = CacheSettings.from_config({"type": "native", "manager_url": "http://node:9901"})
    assert (settings.cache_port, settings.manager_url) == (7999, "http://node:9901")
    assert settings.cache_task_id("fix-git") == "fix-git::pilot1"


def test_snapshot_threshold_switches_the_executor_to_a_fixed_policy(monkeypatch):
    from tvcache_workloads.bash.verl import session as session_module

    made = []
    monkeypatch.setattr(session_module, "TvcacheToolExecutor", lambda *a, **kw: made.append(kw) or FakeExecutor())
    session_module.make_session("t", CacheSettings())
    monkeypatch.setenv("TVCACHE_SNAPSHOT_THRESHOLD", "1.0")
    session_module.make_session("t", CacheSettings.from_config({}))
    assert "adaptive" not in made[0]  # default: tvclient's adaptive policy
    assert (made[1]["adaptive"], made[1]["fixed_threshold_seconds"]) == (False, 1.0)


def test_session_runs_calls_in_order_and_survives_a_failed_call():
    session = RolloutSession("t", FakeExecutor())

    async def go():
        # a turn's parallel calls still run one at a time, in order
        return await asyncio.gather(session.run("cd /app"), session.run("boom"), session.run("ls"))

    outputs = asyncio.run(go())
    assert outputs[0] == "ran cd /app" and outputs[1].startswith("Failed to execute") and outputs[2] == "ran ls"
    assert session.executor.calls[-1] == ["cd /app", "boom", "ls"]
    assert session.failed_calls == 1
    assert json.loads(asyncio.run(session.finish())) == {"passed": True} and session.executor.closed


def test_tool_and_agent_loop_share_one_session_per_trajectory(stub_verl, monkeypatch):
    from tvcache_workloads.bash.verl import agent_loop, tool

    executors = []
    registry = fake_registry(executors)
    monkeypatch.setattr(tool, "SESSIONS", registry)
    monkeypatch.setattr(agent_loop, "SESSIONS", registry)
    bash = tool.TvcacheBashTool({"type": "native"})
    stub_verl.script = [{"command": "mkdir -p /app"}, {"command": "echo hi > /app/x"}]
    kwargs = {"tools_kwargs": {"bash": {"create_kwargs": {"task_id": "hello-world"}}}}

    output = asyncio.run(agent_loop.TvcacheToolAgentLoop(bash).run({}, **kwargs))

    assert [task for task, _ in executors] == ["hello-world"]  # one executor for the whole trajectory
    task, executor = executors[0]
    assert executor.calls == [["mkdir -p /app"], ["mkdir -p /app", "echo hi > /app/x"]] and executor.closed
    assert output.extra_fields["tb_tool_calls"] == 2 and output.extra_fields["tb_test"] == '{"passed": true}'
    assert SESSION_KEY in output.extra_fields and registry.sessions == {}
    assert score(output.extra_fields) == 1.0


def test_a_trajectory_without_tool_calls_scores_minus_one(stub_verl, monkeypatch):
    from tvcache_workloads.bash.verl import agent_loop, tool

    registry = fake_registry([])
    monkeypatch.setattr(agent_loop, "SESSIONS", registry)
    stub_verl.script = []
    output = asyncio.run(agent_loop.TvcacheToolAgentLoop(tool.TvcacheBashTool({"type": "native"})).run({}, tools_kwargs={}))
    assert output.extra_fields["tb_tool_calls"] == 0
    assert score(output.extra_fields) == -1.0


def test_tool_rejects_bad_arguments_without_running_anything(stub_verl, monkeypatch):
    from tvcache_workloads.bash.verl import tool

    executors = []
    monkeypatch.setattr(tool, "SESSIONS", fake_registry(executors))
    bash = tool.TvcacheBashTool({"type": "native"})
    data = types.SimpleNamespace(tools_kwargs={"bash": {"create_kwargs": {"task_id": "t"}}}, extra_fields={})
    response, _, _ = asyncio.run(bash.execute("i", {"command": "  "}, agent_data=data))
    assert response.text.startswith("error") and executors == []
    no_task = types.SimpleNamespace(tools_kwargs={}, extra_fields={})
    response, _, _ = asyncio.run(bash.execute("i", {"command": "ls"}, agent_data=no_task))
    assert "task_id" in response.text and executors == []


# ---- no-cache baseline (DirectExecutor) ----------------------------------

from tvclient import TestToolCall as SentinelTestCall  # noqa: E402  (renamed so pytest does not collect it)

from tvcache_workloads.bash.verl.direct import DirectExecutor  # noqa: E402


class FakeSandbox:
    """Stands in for TerminalBenchSandbox: one shared log of starts, commands, tests, stops."""

    log = []
    flaky_failures_left = 1  # "flaky" raises once, then works

    def __init__(self, task_name, env_id=None):
        self.task_name, self.env_id = task_name, env_id

    def get_id(self):
        return self.env_id

    async def start(self):
        self.env_id = f"live-{sum(1 for e in FakeSandbox.log if e[0] == 'start')}"
        FakeSandbox.log.append(("start", self.env_id))

    async def execute(self, call):
        if isinstance(call, SentinelTestCall):
            FakeSandbox.log.append(("test", self.env_id))
            return '{"passed": true}'
        if call.command == "flaky" and FakeSandbox.flaky_failures_left > 0:
            FakeSandbox.flaky_failures_left -= 1
            raise TimeoutError("command timed out")
        FakeSandbox.log.append(("run", self.env_id, call.command))
        return f"ran {call.command}"

    async def stop(self):
        FakeSandbox.log.append(("stop", self.env_id))


@pytest.fixture
def fake_sandbox():
    FakeSandbox.log, FakeSandbox.flaky_failures_left = [], 1
    return FakeSandbox


def test_direct_executor_runs_every_command_in_one_sandbox_and_tests_the_final_state(fake_sandbox):
    session = RolloutSession("t", DirectExecutor(fake_sandbox, "t"))

    async def go():
        outputs = [await session.run("cd /app"), await session.run("ls")]
        return outputs, await session.finish()

    outputs, test_output = asyncio.run(go())
    assert outputs == ["ran cd /app", "ran ls"] and json.loads(test_output) == {"passed": True}
    assert fake_sandbox.log == [
        ("start", "live-0"), ("run", "live-0", "cd /app"), ("run", "live-0", "ls"), ("test", "live-0"), ("stop", "live-0"),
    ]
    assert (session.executor.total_calls, session.executor.total_executions) == (3, 3)
    assert session.executor.sandbox is None  # finish() stopped it


def test_direct_executor_discards_the_sandbox_on_a_raised_call_and_replays_the_history(fake_sandbox):
    session = RolloutSession("t", DirectExecutor(fake_sandbox, "t"))

    async def go():
        return [await session.run("cd /app"), await session.run("flaky"), await session.run("ls")]

    outputs = asyncio.run(go())
    assert outputs[0] == "ran cd /app" and outputs[1].startswith("Failed to execute") and outputs[2] == "ran ls"
    assert session.failed_calls == 1
    assert fake_sandbox.log == [
        ("start", "live-0"), ("run", "live-0", "cd /app"), ("stop", "live-0"),  # flaky raised: sandbox discarded
        ("start", "live-1"), ("run", "live-1", "cd /app"), ("run", "live-1", "flaky"), ("run", "live-1", "ls"),  # replayed
    ]


def test_direct_executor_tests_an_empty_trajectory_in_a_fresh_sandbox(fake_sandbox):
    session = RolloutSession("t", DirectExecutor(fake_sandbox, "t"))
    assert json.loads(asyncio.run(session.finish())) == {"passed": True}
    assert fake_sandbox.log == [("start", "live-0"), ("test", "live-0"), ("stop", "live-0")]


def test_use_cache_false_makes_the_session_run_without_the_cache(monkeypatch):
    from tvcache_workloads.bash.verl import session as session_module

    cached = []
    monkeypatch.setattr(session_module, "TvcacheToolExecutor", lambda *a, **kw: cached.append(kw) or FakeExecutor())

    assert CacheSettings().use_cache is True
    assert isinstance(session_module.make_session("t", CacheSettings()).executor, FakeExecutor)  # default: the cache
    assert isinstance(session_module.make_session("t", CacheSettings(use_cache=False)).executor, DirectExecutor)
    for env in ("0", "false", "no"):
        monkeypatch.setenv("TVCACHE_USE_CACHE", env)
        assert CacheSettings.from_config({}).use_cache is False
    monkeypatch.setenv("TVCACHE_USE_CACHE", "1")
    assert CacheSettings.from_config({}).use_cache is True
    assert CacheSettings.from_config({"use_cache": False}).use_cache is False  # YAML bool wins over the env
    assert len(cached) == 1  # the no-cache sessions never built a TvcacheToolExecutor
