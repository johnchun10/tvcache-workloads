"""Plugs the Bash workload into tvverify, TVCache's correctness framework.

    uv sync --group verify
    uv run python -m tvverify check --adapter tvcache_workloads.bash.verify:HelloWorldAdapter

Needs the container manager on :9901 (bash/scripts/start_manager.sh). tvverify
starts its own cache server, so the one on :7878 is not used.

State comparison: the sandbox's state_digest() (state.py) runs one read-only
probe command in the container and hashes the watched files, the exported
variables and the current directory there. OLDPWD is an advisory component:
the manager restores the cwd by replaying `cd`, which overwrites OLDPWD, so a
restored snapshot never matches it (found by `tvverify check --suite
contract`). Advisory components are reported in diffs but do not decide
equality. The probe also prints a capped per-entry manifest, so a mismatch
says which file or variable differs. Which paths are watched comes from the
TVCACHE_STATE_* variables (state.py), so a contraction run is checked with the
same policy that decided its merges.
"""

import json
import os
from pathlib import Path
from typing import ClassVar, List, Sequence

from tvclient import TestToolCall, ToolCall
from tvverify.adapters.base import TaskSpec, WorkloadAdapter
from tvverify.compare.base import Comparator, ExactComparator
from tvverify.compare.normalizers import NormalizedComparator

from tvcache_workloads.bash import state
from tvcache_workloads.bash.sandbox import SandboxConfig, TerminalBenchSandbox, tb_task_name
from tvcache_workloads.bash.state import ADVISORY, IGNORED_ENV, MANIFEST_LIMIT, ProbeError, parse_probe  # noqa: F401
from tvcache_workloads.bash.tool_call import BashToolCall

TRACES_DIR = Path(__file__).resolve().parents[3] / "bash" / "traces"
STRICT = state.WatchPolicy().strict


def probe_command(roots: Sequence[str]) -> str:
    """The probe line for these roots (see state.probe_command)."""
    return state.probe_command(state.WatchPolicy(roots=tuple(roots)))


class VerifiableSandbox(TerminalBenchSandbox):
    """The sandbox tvverify drives. TerminalBenchSandbox already has
    state_digest() and equals(); the name is kept for existing adapters."""


def _sandbox_class() -> type:
    """VerifiableSandbox talking to the run's manager (TVCACHE_MANAGER_URL),
    or to the default address when the variable is unset."""
    url = os.environ.get("TVCACHE_MANAGER_URL")
    return VerifiableSandbox.with_config(SandboxConfig(manager_url=url)) if url else VerifiableSandbox


class BashAdapter(WorkloadAdapter):
    """What every Bash workload shares: calls rebuilt from cache keys, and
    output comparison. A key holds only the command, so a rebuilt call always
    counts as state-changing (BashToolCall's default)."""

    name = "bash"
    sandbox_class = _sandbox_class()
    tool_call_class = BashToolCall

    def tool_call_from_key(self, key: str) -> ToolCall:
        if self.is_test_key(key):
            return TestToolCall("")
        data = json.loads(key)
        return BashToolCall(data["command"])

    def tool_class(self, key: str) -> str:
        return "test" if self.is_test_key(key) else "bash"

    def comparator_for(self, key: str) -> Comparator:
        # Terminal text carries container hostnames, ids and timestamps; the
        # test result is JSON ({"passed": ...}) and must match exactly.
        return ExactComparator() if self.is_test_key(key) else NormalizedComparator()


class TraceAdapter(BashAdapter):
    """A Bash workload whose rollouts come from a trace file
    (bash/traces/<task>.json)."""

    trace_file: ClassVar[str] = ""

    def _trace(self) -> dict:
        return json.loads((TRACES_DIR / self.trace_file).read_text())

    def tasks(self) -> Sequence[TaskSpec]:
        task = self._trace()["task"]
        return [TaskSpec(task, root_image_id=task)]

    def scripted_paths(self, task_id: str) -> List[List[ToolCall]]:
        return [[BashToolCall(c) for c in r["commands"]] for r in self._trace()["rollouts"]]


class HelloWorldAdapter(TraceAdapter):
    trace_file = "hello-world.json"


class TerminalBenchAdapter(BashAdapter):
    """Any Terminal-Bench task, for checking a cache trie someone else built
    (a training run's DUMP_ALL):

        tvverify check-graph --adapter tvcache_workloads.bash.verify:TerminalBenchAdapter --dump epoch_1.json

    check-graph takes the task ids from the dump; every call is rebuilt from
    its cache key, so no trace is needed. For `check`, which drives its own
    rollouts, use a TraceAdapter instead. TVVERIFY_TASKS (comma-separated)
    lists tasks for code that asks the adapter."""

    def tasks(self) -> Sequence[TaskSpec]:
        ids = [t.strip() for t in os.environ.get("TVVERIFY_TASKS", "").split(",") if t.strip()]
        return [TaskSpec(t, root_image_id=tb_task_name(t)) for t in ids]
