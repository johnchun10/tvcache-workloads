"""Unit tests (no services): what a contraction run adds. The state probe and
its calibration, the coordinator that pauses contraction for rollouts, the
rollout's side of it, and the dump readers for a server with contraction."""

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fake_manager import FakeManager

from tvcache_workloads.bash import BashToolCall, SandboxConfig, TerminalBenchSandbox
from tvcache_workloads.bash import state
from tvcache_workloads.bash.cache_admin import node_children, snapshot_nodes
from tvcache_workloads.bash.contraction import ContractionGate, Coordinator, EventLog, summarize
from tvcache_workloads.bash.state_check import calibrate_task, noisy_paths
from tvcache_workloads.bash.verl.session import CacheSettings, RolloutSession

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "bash" / "scripts" / "perlmutter"
WIDE = state.WatchPolicy(roots=state.WIDE_ROOTS, excludes=state.RUNTIME_EXCLUDES, packages=True)

# One merged node (two calls), one plain node and a grandchild, as a server
# with contraction dumps them.
CONTRACTED = {
    "root": {
        "children": [
            {
                "children": [
                    {"children": [], "sandbox_id": None, "settled": False,
                     "tool_calls": {"cat f": {"value": "hi", "runtime": 3, "embedding": [0.0, 1.0]}},
                     "number_of_cache_hits": 2, "number_of_lpm_hits": 0}
                ],
                "sandbox_id": "snap-1", "settled": True,
                "tool_calls": {
                    "echo hi > f": {"value": "", "runtime": 5, "embedding": [1.0, 0.0]},
                    "echo hi > /app/f": {"value": "", "runtime": 7, "embedding": [1.0, 0.0]},
                },
                "number_of_cache_hits": 1, "number_of_lpm_hits": 1,
            },
            {"children": [], "sandbox_id": "snap-2", "settled": False,
             "tool_calls": {"echo other > g": {"value": "", "runtime": 1, "embedding": None}},
             "number_of_cache_hits": 0, "number_of_lpm_hits": 0},
        ],
        "sandbox_id": None, "tool_calls": {}, "settled": False,
        "number_of_cache_hits": 0, "number_of_lpm_hits": 0,
    },
    "number_of_nodes": 3, "number_of_sandboxes": 2, "size_in_bytes": 0,
    "sandbox_references": {}, "reclaimable_sandboxes": [],
}


def sandbox_class(fake):
    return TerminalBenchSandbox.with_config(
        SandboxConfig(manager_url="http://fake", command_timeout_sec=5, transport=fake.transport())
    )


# --- the state probe ---------------------------------------------------------

@pytest.mark.parametrize("command", [
    state.probe_command(state.WatchPolicy()),
    state.probe_command(WIDE, manifest=False),
    state.probe_command(WIDE.with_learned(["/tmp/it's [odd]*"], ["NOISY"])),
    state.listing_command(WIDE, 301),
])
def test_probe_lines_are_one_line_of_valid_bash(command):
    assert "\n" not in command and " cd " not in f" {command}"
    if shutil.which("bash"):
        checked = subprocess.run(["bash", "-n", "-c", command], capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr


def test_wide_policy_skips_runtime_files_and_hashes_packages():
    command = state.probe_command(WIDE, manifest=False)
    assert "-path '/etc/hosts'" in command and "-path '/tmp/tmux-*'" in command
    assert "tvv-h pkg" in command and "dpkg-query" in command
    assert "tvv-m" not in command  # no manifest when only the hashes are wanted
    assert "tvv-h pkg" not in state.probe_command(state.WatchPolicy())


def test_policy_comes_from_the_environment_and_the_state_file(monkeypatch, tmp_path):
    assert state.WatchPolicy.base_from_env().roots == ("/app",)  # tvverify's narrow default
    learned = tmp_path / "state.json"
    learned.write_text(json.dumps({"tasks": {"hello-world": {"paths": ["/tmp/a*b"], "env": ["X"], "comparable": True}}}))
    monkeypatch.setenv("TVCACHE_STATE_ROOTS", "/app /etc /tmp")
    monkeypatch.setenv("TVCACHE_STATE_PACKAGES", "1")
    monkeypatch.setenv("TVCACHE_STATE_FILE", str(learned))
    policy = state.WatchPolicy.for_task("hello-world")
    assert policy.roots == ("/app", "/etc", "/tmp") and policy.packages
    assert "/etc/hosts" in policy.excludes and "/root/.cache" not in policy.excludes  # only under watched roots
    assert "/tmp/a\\*b" in policy.excludes and "X" in policy.ignored_env  # the path is matched literally
    assert state.WatchPolicy.for_task("other-task").excludes == state.WatchPolicy.base_from_env().excludes
    assert policy.policy_hash() != state.WatchPolicy.base_from_env().policy_hash()


def test_equals_compares_state_not_history():
    fake = FakeManager()
    Sandbox = sandbox_class(fake)

    async def go():
        a, b, c = Sandbox(task_name="hello-world"), Sandbox(task_name="hello-world"), Sandbox(task_name="hello-world")
        await asyncio.gather(a.start(), b.start(), c.start())
        await a.execute(BashToolCall("echo hi > f"))
        await b.execute(BashToolCall("echo hi > /app/f"))  # another command, same state
        await c.execute(BashToolCall("echo other > f"))
        return await a.equals(b), await a.equals(c), await a.state_digest()

    same, different, digest = asyncio.run(go())
    assert same is True and different is False
    assert set(digest["components"]) >= {"cwd", "env", "fs"} and not digest["components"]["oldpwd"]["strict"]
    assert digest["components"]["fs"]["manifest"]  # state_digest() keeps the manifest for diffs


def test_tool_call_round_trips_through_its_cache_key():
    from tvclient import TestToolCall as Sentinel

    call = BashToolCall("echo 'a b' > f")
    again = BashToolCall.from_dict(json.loads(call.serialize()))
    assert again.serialize() == call.serialize() and again.will_mutate_state()
    assert isinstance(BashToolCall.from_dict(Sentinel().to_dict()), Sentinel)


# --- calibration -------------------------------------------------------------

def test_noisy_paths_drop_what_a_skipped_directory_covers():
    listings = [{"/tmp/x": "d755", "/tmp/x/a": "111", "/app/f": "aaa", "/etc/l -> t1": "l777"},
                {"/tmp/x/a": "222", "/app/f": "aaa", "/etc/l -> t2": "l777"}]
    assert noisy_paths(listings) == ["/etc/l", "/tmp/x"]


def test_calibration_learns_runtime_noise_and_checks_the_result():
    fake = FakeManager(noise_paths=("/etc/machine-id", "/tmp/session/lock"))
    result = asyncio.run(calibrate_task("hello-world", sandbox_class(fake), WIDE))
    assert result["comparable"] is True, result["reason"]
    assert result["paths"] == ["/etc/machine-id", "/tmp/session/lock"]
    assert fake.terminals == {} and fake.snapshots == {}  # everything it started is gone


def test_calibration_marks_a_task_it_cannot_compare():
    fake = FakeManager(noise_paths=tuple(f"/tmp/n{i}" for i in range(state.MAX_LEARNED_PATHS + 1)))
    result = asyncio.run(calibrate_task("hello-world", sandbox_class(fake), WIDE))
    assert result["comparable"] is False and "paths differ" in result["reason"]
    failed = asyncio.run(calibrate_task("no-such-task", sandbox_class(fake), WIDE))
    assert failed["comparable"] is False and failed["reason"]


# --- the coordinator ---------------------------------------------------------

class FakeContractor:
    """tvclient's Contractor interface, recording what the coordinator asks."""

    def __init__(self):
        self.log, self.jobs = [], set()

    def start(self, task):
        self.log.append(("start", task))
        self.jobs.add(task)

    async def stop(self, task):
        if task in self.jobs:
            self.log.append(("stop", task))
            self.jobs.discard(task)

    async def close(self):
        self.log.append(("close", ""))
        self.jobs.clear()


def coordinator(tmp_path, **options):
    contractor = FakeContractor()
    return contractor, Coordinator(contractor, EventLog(tmp_path / "events.jsonl"), grace_seconds=0.01, **options)


def test_a_task_is_contracted_only_while_no_rollout_holds_it(tmp_path):
    async def go():
        contractor, c = coordinator(tmp_path)
        first, second = await c.acquire("t"), await c.acquire("t")
        await c.release("t", first)
        await asyncio.sleep(0.05)
        assert contractor.jobs == set()  # the second rollout still holds it
        await c.release("t", second)
        await asyncio.sleep(0.05)
        assert contractor.jobs == {"t"}
        await c.acquire("t")  # the next batch: the job is stopped before this returns
        assert contractor.jobs == set()
        await c.close()
        return contractor.log

    assert asyncio.run(go()) == [("start", "t"), ("stop", "t"), ("close", "")]
    events = [json.loads(line)["event"] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events == ["resume", "pause"]


def test_a_rollout_that_returns_within_the_grace_period_cancels_the_start(tmp_path):
    async def go():
        contractor, c = coordinator(tmp_path)
        c.grace_seconds = 0.2
        await c.release("t", await c.acquire("t"))
        hold = await c.acquire("t")
        await asyncio.sleep(0.3)
        assert contractor.log == []
        await c.release("t", hold)
        await c.close()

    asyncio.run(go())


def test_uncomparable_tasks_are_never_contracted_and_dead_rollouts_expire(tmp_path):
    async def go():
        now = [0.0]
        contractor, c = coordinator(tmp_path, skip=frozenset({"bad"}), lease_seconds=10, clock=lambda: now[0])
        await c.release("bad::run1", await c.acquire("bad::run1"))  # the cache task id carries the namespace
        await c.acquire("dead")  # this rollout never releases
        await asyncio.sleep(0.05)
        c.sweep()
        await asyncio.sleep(0.05)
        assert contractor.jobs == set()
        now[0] = 11.0  # past the lease
        c.sweep()
        await asyncio.sleep(0.05)
        assert contractor.jobs == {"dead"}
        await c.close()

    asyncio.run(go())


def test_rollouts_reach_the_coordinator_over_its_port(tmp_path):
    async def go():
        contractor, c = coordinator(tmp_path)
        server = await asyncio.start_server(c.handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        gate = ContractionGate("127.0.0.1", port, timeout=5)
        hold = await gate.acquire("t")
        assert hold and c.held("t")
        await gate.release("t", hold)
        await asyncio.sleep(0.05)
        assert contractor.jobs == {"t"}
        server.close()
        await server.wait_closed()
        await c.close()
        # Nothing listening means no Contractor is running: the rollout goes on.
        assert await ContractionGate("127.0.0.1", port, timeout=5).acquire("t") is None

    asyncio.run(go())


class RecordingGate:
    def __init__(self):
        self.calls = []

    async def acquire(self, task):
        self.calls.append(("acquire", task))
        return "hold-1"

    async def release(self, task, hold):
        self.calls.append(("release", task, hold))


class EchoExecutor:
    async def lookup_or_execute(self, history):
        return history[-1].command

    async def test(self, history):
        return '{"passed": true}'

    async def close(self):
        pass


def test_a_rollout_holds_its_task_from_its_first_command_until_its_tests():
    gate = RecordingGate()
    session = RolloutSession("hello-world", EchoExecutor(), gate=gate, cache_task="hello-world::run1")

    async def go():
        await session.run("ls")
        await session.run("pwd")
        return await session.finish()

    assert asyncio.run(go()) == '{"passed": true}'
    assert gate.calls == [("acquire", "hello-world::run1"), ("release", "hello-world::run1", "hold-1")]


def test_settings_read_the_embedder_and_the_coordinator_port(monkeypatch):
    assert CacheSettings.from_config({}).embedder == "" and CacheSettings.from_config({}).contraction_port == 0
    monkeypatch.setenv("TVCACHE_EMBEDDER", "minilm")
    monkeypatch.setenv("TVCACHE_CONTRACTION_PORT", "7879")
    settings = CacheSettings.from_config({})
    assert settings.embedder == "minilm" and settings.contraction_port == 7879


def test_event_totals_and_timeline(tmp_path):
    now = [100.0]
    log = EventLog(tmp_path / "events.jsonl", clock=lambda: now[0])
    log.write("groups", "a::r", depth=1, groups=1, children=3, pairs=2)
    now[0] = 150.0
    log.write("apply", "a::r", depth=1, classes=1, settled=2, merged_nodes=1, merged_classes=1, skipped_calls=0)
    now[0] = 200.0
    log.write("job_end", "a::r", depth=2, comparisons=2, merges=1, restores=2, replays=1, unavailable=0)
    summary = summarize(tmp_path / "events.jsonl", buckets=4)
    assert summary["totals"]["similar_pairs_listed"] == 2 and summary["totals"]["merged_classes"] == 1
    assert summary["totals"]["comparisons"] == 2 and summary["per_task"]["a"]["jobs"] == 1
    assert [b["merged_classes"] for b in summary["over_time"]] == [0, 0, 1, 1]  # cumulative
    assert [b["similar_pairs_listed"] for b in summary["over_time"]] == [2, 2, 2, 2]


# --- dumps from a server with contraction ------------------------------------

def test_dump_readers_handle_both_shapes():
    plain = {"root": {"children": {"ls": {"children": {}, "value": "f", "runtime": 2, "sandbox_id": "s"}}}}
    assert node_children(plain["root"]) == [({"ls": {"value": "f", "runtime": 2}}, plain["root"]["children"]["ls"])]
    assert snapshot_nodes({"t": plain}, "t") == [(["ls"], "s")]
    calls = [sorted(c) for c, _ in node_children(CONTRACTED["root"])]
    assert calls == [["echo hi > /app/f", "echo hi > f"], ["echo other > g"]]
    # A merged node is named by any of its calls; one path per snapshot.
    assert sorted(snapshot_nodes({"t": CONTRACTED}, "t")) == [(["echo hi > /app/f"], "snap-1"), (["echo other > g"], "snap-2")]


def test_tvverify_reads_a_merged_node_as_one_node_with_aliases():
    pytest.importorskip("tvverify")
    from tvverify.model.dump import is_contracted, plain_task
    from tvverify.model.graph import from_dumpall
    from tvverify.validators.structural import check_structure, recount

    assert is_contracted(CONTRACTED) and not is_contracted(plain_task(CONTRACTED))
    graph = from_dumpall({"t": CONTRACTED})["t"]
    by_key = {node.key: node for node in graph.nodes.values() if node.key}
    primary, alias = by_key["echo hi > /app/f"], by_key["echo hi > f"]
    assert alias.alias_of == primary.node_id and primary.alias_of is None
    assert (primary.runtime_ms, alias.runtime_ms) == (7, 5)  # each call keeps its own value and runtime
    assert alias.sandbox_id == primary.sandbox_id == "snap-1"  # replaying the alias checks the merge
    assert by_key["cat f"].path == ["echo hi > /app/f", "cat f"]  # children hang under the first call
    assert recount(graph) == {"number_of_nodes": 3, "number_of_sandboxes": 2}
    assert check_structure(graph) == []


def test_compare_runs_counts_merged_calls(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    try:
        import compare_runs
    finally:
        sys.path.remove(str(SCRIPTS))
    dump = tmp_path / "dump_000.json"
    dump.write_text(json.dumps({"t": CONTRACTED}))
    stats = compare_runs.cache_stats(dump)
    assert (stats["nodes"], stats["calls"], stats["hits"], stats["runtime_ms"]) == (3, 4, 3, 16)
    (tmp_path / "config.env").write_text("CACHE=contraction\nTHRESHOLD=0.9\nEMBEDDER=minilm\n")
    assert compare_runs.series_label(tmp_path) == "Contraction (0.9, minilm)"


# --- the launcher ------------------------------------------------------------

@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_run_sh_resolves_presets_and_flags():
    def dry(*args):
        return subprocess.run(["bash", str(SCRIPTS / "run.sh"), *args, "--dry-run"], capture_output=True, text=True)

    shakedown = dry("shakedown")
    assert shakedown.returncode == 0, shakedown.stderr
    assert "tasks (2): hello-world openssl-selfsigned-cert" in shakedown.stdout
    assert "cache: contraction (threshold 0.9, embedder minilm, budget 32 sandboxes)" in shakedown.stdout
    assert "epochs 2, rollouts/task 4, tasks/batch 1, max tokens/rollout 2048" in shakedown.stdout

    main20 = dry("main20-easy-med", "--cache", "tvcache", "--epochs", "3", "--no-verify")
    assert "tasks (20):" in main20.stdout and "main20-easy-med-tvcache" in main20.stdout
    assert "epochs 3, rollouts/task 8, tasks/batch 4" in main20.stdout and "tvverify off" in main20.stdout
    assert "threshold" not in main20.stdout

    sweep = dry("main20-easy-med", "--threshold", "0.95", "--embedder", "jina-code")
    assert "main20-easy-med-contraction-t0.95-jina-code" in sweep.stdout
    assert dry("nope").returncode == 2 and dry("shakedown", "--cache", "maybe").returncode == 2
