"""Unit tests (no services): what a contraction run adds. The state probe and
its calibration, the coordinator that pauses contraction for rollouts, the
rollout's side of it, and the dump readers for a server with contraction."""

import asyncio
import json
import os
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
    assert (stats["nodes"], stats["calls"], stats["runtime_ms"]) == (3, 4, 16)
    assert "hits" not in stats  # hit rates come from metrics.json, not the server's counters
    (tmp_path / "config.env").write_text("CACHE=contraction\nTHRESHOLD=0.9\nEMBEDDER=minilm\n")
    assert compare_runs.series_label(tmp_path) == "Contraction (0.9, minilm)"


def test_compare_runs_reads_hit_rates_from_metrics_json(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    try:
        import compare_runs
    finally:
        sys.path.remove(str(SCRIPTS))
    assert compare_runs.load_metrics(tmp_path) is None  # no metrics.json and no trace
    assert compare_runs.hits_per_epoch(None) == []
    (tmp_path / "metrics.json").write_text(json.dumps({"tasks": {
        "a": {"0": {"lookups": 10, "cache_hit_rate": 0.2}, "1": {"lookups": 10, "cache_hit_rate": 0.5}},
        "b": {"0": {"lookups": 30, "cache_hit_rate": 0.0}, "1": {"nodes": 4}},  # epoch 1: graph size only
    }}))
    rows = compare_runs.hits_per_epoch(compare_runs.load_metrics(tmp_path))
    assert [(r["epoch"], r["hits"], r["lookups"]) for r in rows] == [(1, 2, 40), (2, 5, 10)]
    assert rows[0]["hit rate"] == pytest.approx(0.05) and rows[1]["hit rate"] == pytest.approx(0.5)


# --- the launcher ------------------------------------------------------------

@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_run_sh_resolves_presets_and_flags():
    def dry(*args):
        return subprocess.run(["bash", str(SCRIPTS / "run.sh"), *args, "--dry-run"], capture_output=True, text=True)

    shakedown = dry("shakedown")
    assert shakedown.returncode == 0, shakedown.stderr
    assert "tasks (2): hello-world openssl-selfsigned-cert" in shakedown.stdout
    assert "cache: contraction (threshold 0.9, embedder minilm, budget 32 sandboxes)" in shakedown.stdout
    assert "epochs 2, rollouts/task 4, tasks/batch 1, max tokens/rollout 4096" in shakedown.stdout

    main20 = dry("main20-easy-med", "--cache", "tvcache", "--epochs", "3", "--no-verify")
    assert "tasks (20):" in main20.stdout and "main20-easy-med-tvcache" in main20.stdout
    assert "epochs 3, rollouts/task 8, tasks/batch 4" in main20.stdout and "tvverify off" in main20.stdout
    assert "(threshold" not in main20.stdout
    assert "snapshot threshold: 1.0" in main20.stdout

    sweep = dry("main20-easy-med", "--threshold", "0.97", "--embedder", "embeddinggemma2")
    assert "main20-easy-med-contraction-t0.97-embeddinggemma2" in sweep.stdout
    assert dry("nope").returncode == 2 and dry("shakedown", "--cache", "maybe").returncode == 2

    assert "snapshot threshold: 0.5" in dry("shakedown", "--snapshot-threshold", "0.5").stdout
    assert "snapshot threshold: adaptive" in dry("shakedown", "--snapshot-threshold", "adaptive").stdout
    assert dry("shakedown", "--snapshot-threshold", "soon").returncode == 2

    assert "fork bank top-k 4" in dry("shakedown", "--top-k", "4").stdout
    assert dry("shakedown", "--top-k", "many").returncode == 2
    assert "only works with submit_hetjob.sh" in dry("shakedown", "--cache", "all").stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_submit_hetjob_rejects_bad_flags_before_queueing(tmp_path):
    """Bad job flags and bad run flags fail at once, before sbatch is called."""
    def submit(*args):
        env = {**os.environ, "SCRATCH": str(tmp_path)}
        return subprocess.run(["bash", str(SCRIPTS / "submit_hetjob.sh"), "shakedown", *args],
                              capture_output=True, text=True, env=env)

    assert "hh:mm:ss" in submit("--time", "2h").stderr
    assert "--cache must be none, tvcache, contraction or all" in submit("--cache", "maybe").stderr
    assert "--split needs --total" in submit("--split", "3").stderr
    assert "unknown option: --bogus" in submit("--bogus").stderr  # passed to run.sh, which rejects it
    assert not (tmp_path / "tvcache").exists()  # nothing was submitted


def test_submit_hetjob_queues_tvverify_after_each_cached_run(tmp_path):
    """Training and tvverify are separate jobs: every run that has a cache gets
    a one-node tvverify job that starts when its training job ends."""
    tools = tmp_path / "tvcache" / "bin"
    tools.mkdir(parents=True)
    fake = tmp_path / "fakebin"
    fake.mkdir()
    for path in (tools / "uv", tools / "docker-compose"):
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
    # A stand-in sbatch: records what it was asked and prints the next job id.
    (fake / "sbatch").write_text(
        "#!/bin/sh\n"
        'n=$(( $(cat "$SCRATCH/n" 2>/dev/null || echo 100) + 1 )); echo $n > "$SCRATCH/n"\n'
        'echo "$n RUN=$RUN CACHES=${CACHES:-} $*" >> "$SCRATCH/sbatch.log"\n'
        'echo "$n;cluster"\n'
    )
    (fake / "sbatch").chmod(0o755)

    def submit(*args, **extra):
        (tmp_path / "sbatch.log").unlink(missing_ok=True)
        env = {**os.environ, "SCRATCH": str(tmp_path), "PATH": f"{fake}:{os.environ['PATH']}", "BASE": str(tmp_path / "run"), **extra}
        result = subprocess.run(["bash", str(SCRIPTS / "submit_hetjob.sh"), "shakedown", *args],
                                capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr
        log = (tmp_path / "sbatch.log").read_text().splitlines()
        return result.stdout, [line.split(" ", 3) for line in log]

    # One cached run: the training job, then tvverify for the same run directory.
    out, jobs = submit("--cache", "tvcache")
    (train_id, train_run, _, train_args), (verify_id, verify_run, _, verify_args) = jobs
    assert train_run == verify_run == f"RUN={tmp_path / 'run'}"
    assert train_args.endswith("hetjob.sbatch") and "--dependency" not in train_args
    assert f"--dependency=afterany:{train_id}" in verify_args and verify_args.endswith("verify.sbatch")
    assert f"job {verify_id}: tvverify for RUN={tmp_path / 'run'} (after {train_id})" in out

    # No cache, or tvverify turned off: nothing to check, so no second job.
    assert len(submit("--cache", "none")[1]) == 1
    assert len(submit("--cache", "tvcache", "--no-verify")[1]) == 1

    # All three modes in one training job: one tvverify job for each mode that has a cache.
    _, jobs = submit("--cache", "all")
    assert [job[1] for job in jobs] == [f"RUN={tmp_path / 'run'}", f"RUN={tmp_path / 'run'}-tvcache", f"RUN={tmp_path / 'run'}-contraction"]
    assert jobs[0][2] == "CACHES=none"  # the rest of "none tvcache contraction" is in the args column
    assert all(f"--dependency=afterany:{jobs[0][0]}" in job[3] for job in jobs[1:])

    # A split run: each part's tvverify waits for its own part.
    _, jobs = submit("--cache", "tvcache", "--split", "2", "--total", "4")
    kinds = ["verify" if job[3].endswith("verify.sbatch") else "train" for job in jobs]
    assert kinds == ["train", "verify", "train", "verify"]
    assert f"--dependency=afterany:{jobs[0][0]}" in jobs[1][3] and f"--dependency=afterany:{jobs[2][0]}" in jobs[3][3]
    assert jobs[1][1].endswith("-part1") and jobs[3][1].endswith("-part2")

    # A longer limit for tvverify's job is written into a copy of its job file.
    _, jobs = submit("--cache", "tvcache", VERIFY_TIME="03:30:00")
    script = Path(jobs[1][3].split()[-1])
    assert "-t 03:30:00" in script.read_text() and "-t 02:00:00" not in script.read_text()


def test_submit_hetjob_asks_for_half_an_80gb_node_or_a_whole_node(tmp_path):
    """The GPU component is two 80 GB GPUs in the shared QOS (charged as half
    a node) unless GPU_NODE=full asks for a whole 40 GB node; veRL's GPU count
    follows."""
    tools = tmp_path / "tvcache" / "bin"
    tools.mkdir(parents=True)
    fake = tmp_path / "fakebin"
    fake.mkdir()
    for path in (tools / "uv", tools / "docker-compose"):
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
    (fake / "sbatch").write_text(
        "#!/bin/sh\n"
        'for last; do :; done\n'
        'case "$last" in *hetjob*) echo "$N_GPUS $last" >> "$SCRATCH/sbatch.log";; esac\n'
        'echo "101;cluster"\n'
    )
    (fake / "sbatch").chmod(0o755)

    def submit(*args, **extra):
        (tmp_path / "sbatch.log").unlink(missing_ok=True)
        env = {k: v for k, v in os.environ.items() if k != "N_GPUS"}  # the script's own default, whatever this shell has
        env.update(SCRATCH=str(tmp_path), PATH=f"{fake}:{os.environ['PATH']}", BASE=str(tmp_path / "run"), **extra)
        result = subprocess.run(["bash", str(SCRIPTS / "submit_hetjob.sh"), "shakedown", "--cache", "tvcache", *args],
                                capture_output=True, text=True, env=env)
        if result.returncode != 0:
            return result, None, None
        gpus, script = (tmp_path / "sbatch.log").read_text().split()
        gpu_lines = Path(script).read_text().split("#SBATCH hetjob\n")[1].splitlines()[:2]
        return result, gpus, gpu_lines

    result, gpus, gpu_lines = submit()
    assert gpus == "2" and "(GPUs: 2, half node)" in result.stdout
    assert gpu_lines == ["#SBATCH -A m4999 -q shared -t 05:00:00", "#SBATCH -C gpu&hbm80g -n 1 -c 64 --gpus=2"]

    result, gpus, gpu_lines = submit("--time", "01:00:00", GPU_NODE="full")
    assert gpus == "4" and "(GPUs: 4, full node)" in result.stdout
    assert gpu_lines == ["#SBATCH -A m4999 -q regular -t 01:00:00", "#SBATCH -C gpu -N 1 -G 4"]

    result, _, _ = submit(GPU_NODE="quarter")
    assert result.returncode == 2 and "GPU_NODE must be half or full" in result.stderr
    # The step's size follows the GPU count: a quarter of the node's threads per GPU.
    job = (SCRIPTS / "hetjob.sbatch").read_text()
    assert 'gpu_step=(-n 1 -c $(( 32 * GPUS )) --gpus="$GPUS" --cpu-bind=none)' in job and 'srun --het-group=1 "${gpu_step[@]}"' in job


def test_coordinator_logs_how_many_sandboxes_contraction_has_running(tmp_path):
    """The split of live sandboxes into contraction's and training's comes
    from here: a `live` event each time the coordinator's count changes."""
    from tvcache_workloads.bash.contraction import EventLog, live_counter

    path = tmp_path / "events.jsonl"
    on_event = live_counter(EventLog(path))
    on_event({"op": "start", "seconds": 1.0, "env_id": "term_a"})
    on_event({"op": "restore", "seconds": 1.0, "handle": "tb-snap:snap_1", "env_id": "term_b", "mode": "cold"})
    on_event({"op": "execute", "seconds": 0.1})                       # not a change
    on_event({"op": "stop", "seconds": 0.5, "env_id": "tb-snap:snap_1"})  # deleting a snapshot: nothing live ended
    on_event({"op": "stop", "seconds": 0.5, "env_id": "term_a"})
    on_event({"op": "stop", "seconds": 0.5, "env_id": "term_b"})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["event"], r["sandboxes"]) for r in records] == [("live", 1), ("live", 2), ("live", 1), ("live", 0)]
    # train_pilot.sh reads the last one with grep and sed, so the spelling matters.
    assert '"event": "live"' in path.read_text() and '"sandboxes": 0' in path.read_text().splitlines()[-1]


def test_sandbox_memory_adds_up_processes_in_other_pid_namespaces(tmp_path):
    """sandbox_memory.py: containers have their own PID namespaces, so the
    sandboxes' memory is the PSS of every process outside this one's."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        import sandbox_memory
    finally:
        sys.path.remove(str(SCRIPTS))

    def process(pid, namespace, pss_kb=None):
        folder = tmp_path / str(pid) / "ns"
        folder.mkdir(parents=True)
        os.symlink(f"pid:[{namespace}]", folder / "pid")
        if pss_kb is not None:
            (tmp_path / str(pid) / "smaps_rollup").write_text(f"Rss:  {pss_kb * 2} kB\nPss:  {pss_kb} kB\nPss_Anon: 1 kB\n")

    process("self", 4026531836)
    process(1, 4026531836, 9000)        # this node's own processes: not counted
    process(200, 4026532500, 1500)      # a container
    process(201, 4026532500, 500)
    process(300, 4026532777, 3000)      # another container
    process(301, 4026532777)            # ended before it could be read
    (tmp_path / "meminfo").write_text("MemTotal: 1 kB\n")
    assert sandbox_memory.sandbox_pss_kb(tmp_path) == 5000
    assert sandbox_memory.sandbox_pss_kb(tmp_path / "missing") == 0
