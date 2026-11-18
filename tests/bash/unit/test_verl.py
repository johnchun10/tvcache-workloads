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


def test_dataset_skips_snapshot_task_dirs(tmp_path):
    # The manager's snapshot/restore copies (<task>_<uuid4>) that a run cut
    # off by its time limit leaves next to the real task directories.
    for name in ("build-stp", "build-stp_1957b802-7a7d-4d71-83cb-8028fabe77f6", "my_task"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "task.yaml").write_text("instruction: Build it.\n")
    assert [r["extra_info"]["task_id"] for r in task_rows(tmp_path)] == ["build-stp", "my_task"]


def test_control_port_hands_the_executor_the_sidecars_fork_bank(monkeypatch):
    from tvcache_workloads.bash.control import RemoteForkGenerator
    from tvcache_workloads.bash.verl import session as session_module

    banks = []

    class BankedExecutor(FakeExecutor):
        def set_fork_bank(self, bank):
            banks.append(bank)

    monkeypatch.setattr(session_module, "TvcacheToolExecutor", lambda *a, **kw: BankedExecutor())
    monkeypatch.setattr(session_module, "_fork_bank", None)

    session_module.make_session("t", CacheSettings())  # no control port: no bank
    assert banks == []
    monkeypatch.setenv("TVCACHE_CONTROL_PORT", "7880")
    settings = CacheSettings.from_config({})
    assert settings.control_port == 7880
    session_module.make_session("t", settings)
    session_module.make_session("u", settings)
    assert len(banks) == 2 and isinstance(banks[0], RemoteForkGenerator) and banks[0] is banks[1]  # one per process


def test_a_timed_out_command_ends_the_trajectory_without_replaying_it():
    # Seen in training: every later call and the tests replayed the hung
    # command and waited out its timeout again.
    from tvcache_workloads.bash.manager_client import CommandTimeout
    from tvcache_workloads.bash.verl.session import AFTER_TIMEOUT, NOT_TESTED

    class Executor:
        def __init__(self):
            self.calls, self.tests, self.closed = [], 0, False

        async def lookup_or_execute(self, history):
            self.calls.append([c.command for c in history])
            if history[-1].command == "cat":
                raise CommandTimeout("/execute", 504, '{"error":"Command timed out after 30.0 seconds"}')
            return f"ran {history[-1].command}"

        async def test(self, history):
            self.tests += 1
            return json.dumps({"passed": True})

        async def close(self):
            self.closed = True

    session = RolloutSession("t", Executor())

    async def go():
        return [await session.run(c) for c in ("cd /app", "cat", "ls", "pwd")], await session.finish()

    outputs, test = asyncio.run(go())
    assert outputs[0] == "ran cd /app"
    assert outputs[1].startswith("Failed to execute: CommandTimeout")
    assert outputs[2] == outputs[3] == AFTER_TIMEOUT
    assert session.executor.calls == [["cd /app"], ["cd /app", "cat"]]  # nothing ran after the timeout
    assert (session.timed_out, session.failed_calls, len(session.history)) == (True, 3, 2)
    assert test == NOT_TESTED and session.executor.tests == 0 and session.executor.closed
    # the reward is what a failed test gives: the model called the tool, the task is not solved
    score = reward.compute_score("terminal_bench", "", "", {"tb_tool_calls": len(session.history), "tb_test": test})
    assert (score["score"], score["passed"]) == (0.0, 0.0)


def test_the_command_timeout_comes_from_the_environment(monkeypatch):
    assert CacheSettings().command_timeout_sec == 30.0
    monkeypatch.setenv("TVCACHE_COMMAND_TIMEOUT", "45")
    assert CacheSettings.from_config({}).command_timeout_sec == 45.0


# --- trainer.py: the step hooks, driven the way veRL 0.9.1's trainer_base.py
# drives them (step -> prepare_step -> _next_train_batch -> one
# _fetch_one_gen_batch per prompt, then on_sample_end, then _save_checkpoint)


class FakeLoader:
    """One prompt per batch, like the dataloader veRL builds for this mode."""

    def __init__(self, tasks):
        self.tasks, self.fetched = tasks, 0

    def __iter__(self):
        for task in self.tasks:
            self.fetched += 1
            yield {"extra_info": [{"task_id": task}], "raw_prompt": [task]}

    def state_dict(self):
        return {"fetched": self.fetched}


class FakeData(dict):
    __getattr__ = dict.__getitem__


@pytest.fixture
def trainer_run(monkeypatch):
    """Runs TvcacheSyncTrainer for some steps; returns what it sent and did."""

    class PPOTrainerSync:
        def _next_train_batch(self):
            data = self.config.data
            fetches = data.train_batch_size // (data.get("gen_batch_size", None) or data.train_batch_size)
            return [task for _ in range(fetches) for task in self._fetch_one_gen_batch()["raw_prompt"]]

        def prepare_step(self):
            self.submitted.append(self._next_train_batch())
            return {}

        def on_sample_end(self):
            pass

        def _save_checkpoint(self):
            self.saved.append(self.train_dataloader.state_dict())

    modules = {name: types.ModuleType(name) for name in (
        "verl", "verl.trainer", "verl.trainer.ppo", "verl.trainer.ppo.v1", "verl.trainer.ppo.v1.trainer_base",
        "verl.trainer.ppo.v1.trainer_sync", "verl.utils", "verl.utils.tensordict_utils")}
    modules["verl.trainer.ppo.v1.trainer_base"].register_trainer = lambda name: (lambda cls: cls)
    modules["verl.trainer.ppo.v1.trainer_sync"].PPOTrainerSync = PPOTrainerSync
    modules["verl.utils.tensordict_utils"].get_tensordict = lambda batch: batch
    modules["verl.utils"].tensordict_utils = modules["verl.utils.tensordict_utils"]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "tvcache_workloads.bash.verl.trainer", raising=False)
    from tvcache_workloads.bash.verl import trainer as trainer_module

    def run(tasks, batch, epochs, refill_at=None):
        calls = []
        monkeypatch.setattr(trainer_module.control, "call", lambda op, **fields: calls.append((op, fields)) or {})
        trainer = trainer_module.TvcacheSyncTrainer()
        trainer.config = types.SimpleNamespace(data=FakeData(train_batch_size=batch, gen_batch_size=1))
        trainer.train_dataloader, trainer.train_dataloader_it = FakeLoader(tasks), None
        trainer.steps_per_epoch = len(tasks) // batch
        trainer.total_training_steps = trainer.steps_per_epoch * epochs
        trainer.submitted, trainer.saved, trainer.global_steps = [], [], 0
        for step in range(1, trainer.total_training_steps + 1):
            trainer.global_steps = step
            trainer.prepare_step()
            if step == refill_at:  # veRL replaces an evicted group mid-step
                trainer.submitted[-1] += trainer._fetch_one_gen_batch()["raw_prompt"]
            trainer.on_sample_end()
            trainer._save_checkpoint()
        return trainer, calls

    return run


def test_trainer_sends_one_begin_warm_and_end_per_step_with_all_its_tasks(trainer_run):
    tasks = [f"t{i}" for i in range(8)]
    trainer, calls = trainer_run(tasks, batch=4, epochs=2)  # 4 steps of 4 tasks
    steps = [tasks[:4], tasks[4:], tasks[:4], tasks[4:]]
    assert trainer.submitted == steps  # veRL gets the dataloader's order, nothing skipped or repeated
    expected = [("warm", {"tasks": steps[0], "wait": True})]
    for i, current in enumerate(steps):
        step, epoch, last = i + 1, i // 2, i == 3
        expected.append(("begin", {"step": step, "epoch": epoch, "tasks": current}))
        if not last:  # the next step's tasks warm while this one rolls out, and this step's roots are kept
            expected.append(("warm", {"tasks": steps[i + 1], "keep": current, "wait": False}))
        expected.append(("end", {"step": step, "epoch": epoch, "tasks": current, "last": last}))
    assert calls == expected


def test_trainer_checkpoint_does_not_count_the_peeked_step(trainer_run):
    trainer, _ = trainer_run([f"t{i}" for i in range(8)], batch=4, epochs=2)
    # After step n the run has trained 4n prompts; the loader itself is one step ahead until the last.
    assert trainer.saved == [{"fetched": 4}, {"fetched": 8}, {"fetched": 12}, {"fetched": 16}]
    assert trainer.train_dataloader.state_dict() == {"fetched": 16}  # the override is gone


def test_trainer_counts_a_prompt_added_mid_step_as_that_steps_task(trainer_run):
    tasks = [f"t{i}" for i in range(8)]
    trainer, calls = trainer_run(tasks, batch=2, epochs=1, refill_at=1)
    assert trainer.submitted[0] == ["t0", "t1", "t2"]
    ends = [fields["tasks"] for op, fields in calls if op == "end"]
    assert ends[0] == ["t0", "t1", "t2"]
    # No prompt is lost or repeated; later steps just start one prompt further on.
    assert [task for step in trainer.submitted for task in step] == tasks + ["t0"]
