"""Plugs the Bash workload into tvverify, TVCache's correctness framework.

    uv sync --group verify
    uv run python -m tvverify check --adapter tvcache_workloads.bash.verify:HelloWorldAdapter

Needs the container manager on :9901 (bash/scripts/start_manager.sh). tvverify
starts its own cache server, so the one on :7878 is not used.

State comparison: VerifiableSandbox.state_digest() runs one read-only probe
command in the container and hashes, inside the container,
  fs   every file, directory and symlink under the watched roots (type, mode,
       link target, content)
  env  exported variables, except bookkeeping (_, SHLVL, PWD, TMUX*, HOSTNAME)
  cwd  the current directory
and records OLDPWD as an advisory component: the manager restores the cwd by
replaying `cd`, which overwrites OLDPWD, so a restored snapshot never matches
it (found by `tvverify check --suite contract`). Advisory components are
reported in diffs but do not decide equality.
Equality uses the strict hashes only. The probe also prints a capped per-entry
manifest, so a mismatch says which file or variable differs. Like snapshots, it
cannot see processes or unexported variables.
"""

import json
import os
from pathlib import Path
from typing import ClassVar, List, Optional, Sequence, Tuple

from tvclient import TestToolCall, ToolCall
from tvverify.adapters.base import TaskSpec, WorkloadAdapter
from tvverify.compare.base import Comparator, ExactComparator
from tvverify.compare.normalizers import NormalizedComparator
from tvverify.digest.schema import DigestComponent, StateDigest, sha256_json

from tvcache_workloads.bash.sandbox import TerminalBenchSandbox, tb_task_name
from tvcache_workloads.bash.tool_call import BashToolCall

TRACES_DIR = Path(__file__).resolve().parents[3] / "bash" / "traces"
# Bookkeeping that differs between terminals without being task state
# (HOSTNAME is the container id). OLDPWD is state, but is checked separately
# as an advisory component (see the module docstring).
IGNORED_ENV = ("_", "SHLVL", "PWD", "OLDPWD", "TMUX", "TMUX_PANE", "HOSTNAME")
STRICT = ("cwd", "env", "fs")
ADVISORY = ("oldpwd",)
MANIFEST_LIMIT = 300


class ProbeError(Exception):
    """The state probe's output could not be read."""


def probe_command(roots: Sequence[str]) -> str:
    """One line (the manager joins multi-line commands with &&) that prints
    `tvv-h <component> <hash>` lines and capped `tvv-m <component> ...`
    manifest lines. It reads state only: absolute paths, no cd, no writes."""
    quoted = " ".join(f"'{r}'" for r in roots)
    skip = " ".join(IGNORED_ENV)
    env_lines = (
        f'for v in $(compgen -e | sort); do [[ " {skip} " == *" $v "* ]] && continue; '
        'printf "%s=%s\\n" "$v" "${!v}"; done'
    )
    tree = f"find {quoted} -xdev \\( -type d -o -type f -o -type l \\) -printf '%y %m %p -> %l\\n' 2>/dev/null | sort"
    contents = f"find {quoted} -xdev -type f -print0 2>/dev/null | sort -z | xargs -0 -r sha256sum 2>/dev/null"
    return "; ".join([
        "printf 'tvv-h cwd %s\\n' \"$(pwd | sha256sum | cut -c1-16)\"",
        "printf 'tvv-m cwd %s\\n' \"$(pwd)\"",
        "printf 'tvv-h oldpwd %s\\n' \"$(printf %s \"$OLDPWD\" | sha256sum | cut -c1-16)\"",
        f"printf 'tvv-h env %s\\n' \"$({env_lines} | sha256sum | cut -c1-16)\"",
        f"{env_lines} | while IFS='=' read -r k val; do "
        "printf 'tvv-m env %s %s\\n' \"$(printf %s \"$val\" | sha256sum | cut -c1-12)\" \"$k\"; "
        f"done | head -{MANIFEST_LIMIT}",
        f"printf 'tvv-h fs %s\\n' \"$({{ {tree}; {contents}; }} | sha256sum | cut -c1-16)\"",
        f"{contents} | awk '{{print \"tvv-m fs \" substr($1,1,12) \" \" substr($0,67)}}' | head -{MANIFEST_LIMIT}",
        f"{tree} | awk '$1!=\"f\" {{line=substr($0, index($0,$3)); sub(/ -> $/, \"\", line); print \"tvv-m fs \" $1 $2 \" \" line}}' | head -{MANIFEST_LIMIT}",
    ])


def parse_probe(output: str) -> Tuple[dict, dict]:
    """(component -> hash, component -> {entry: value}) from probe output.
    tmux wraps long rows; a row that does not start a record continues the
    previous one."""
    records: List[str] = []
    for row in output.splitlines():
        if row.startswith("tvv-"):
            records.append(row)
        elif records:
            records[-1] += row
    hashes, manifests = {}, {name: {} for name in STRICT + ADVISORY}
    for record in records:
        parts = record.split(" ", 3)
        if parts[0] == "tvv-h" and len(parts) == 3:
            hashes[parts[1]] = parts[2]
        elif parts[0] == "tvv-m" and parts[1] == "cwd":
            manifests["cwd"]["cwd"] = record.split(" ", 2)[2] if len(parts) > 2 else ""
        elif parts[0] == "tvv-m" and len(parts) == 4:
            # env: <hash> <name>; fs: <hash or type+mode> <path>
            manifests[parts[1]][parts[3]] = parts[2]
    missing = set(STRICT) - set(hashes)
    if missing:
        raise ProbeError(f"state probe printed no hash for {sorted(missing)}: {output[:300]!r}")
    return hashes, manifests


class VerifiableSandbox(TerminalBenchSandbox):
    """TerminalBenchSandbox plus a state digest, for tvverify only. The cache
    never calls state_digest(), so production runs are unaffected."""

    watch_roots: ClassVar[Tuple[str, ...]] = ("/app",)

    async def state_digest(self) -> Optional[dict]:
        output = await self.execute(BashToolCall(probe_command(self.watch_roots)))
        if output.startswith("Failed to execute"):
            raise ProbeError(output)
        hashes, manifests = parse_probe(output)
        components = {
            name: DigestComponent(name, hashes[name], name in STRICT, manifests[name])
            for name in STRICT + ADVISORY
            if name in hashes
        }
        return StateDigest(
            components,
            policy_hash=sha256_json({"roots": list(self.watch_roots), "ignored_env": IGNORED_ENV})[:16],
            root_image_id=tb_task_name(self.task_name),
        ).to_dict()


class BashAdapter(WorkloadAdapter):
    """What every Bash workload shares: calls rebuilt from cache keys, and
    output comparison. A key holds only the command, so a rebuilt call always
    counts as state-changing (BashToolCall's default)."""

    name = "bash"
    sandbox_class = VerifiableSandbox
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
