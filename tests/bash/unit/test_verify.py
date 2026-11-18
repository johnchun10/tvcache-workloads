"""Unit tests (no services): the tvverify adapter and its state probe parser.
Skipped unless the `verify` dependency group is installed."""

import pytest

pytest.importorskip("tvverify")

from tvclient import TestToolCall  # noqa: E402

from tvcache_workloads.bash import BashToolCall  # noqa: E402
from tvcache_workloads.bash.verify import (  # noqa: E402
    HelloWorldAdapter,
    ProbeError,
    parse_probe,
    probe_command,
)

OUTPUT = """tvv-h cwd ad1405e0d73051b4
tvv-m cwd /app/work
tvv-h oldpwd 0123456789abcdef
tvv-h env 36f5a88562606cea
tvv-m env fcde2b2edba5 FOO
tvv-h fs 05290a8b38cfb781
tvv-m fs 98ea6e4f216f /app/hello.txt
tvv-m fs d755 /app/a-very-long-direc
tory-name-wrapped-by-tmux"""


def test_parse_probe_reads_hashes_and_rejoins_wrapped_rows():
    hashes, manifests = parse_probe(OUTPUT)
    assert hashes == {
        "cwd": "ad1405e0d73051b4", "oldpwd": "0123456789abcdef",
        "env": "36f5a88562606cea", "fs": "05290a8b38cfb781",
    }
    assert manifests["cwd"] == {"cwd": "/app/work"}
    assert manifests["env"] == {"FOO": "fcde2b2edba5"}
    assert manifests["fs"]["/app/a-very-long-directory-name-wrapped-by-tmux"] == "d755"


def test_oldpwd_is_advisory_so_restores_can_still_match():
    from tvcache_workloads.bash.verify import ADVISORY, STRICT

    assert "oldpwd" in ADVISORY and "oldpwd" not in STRICT


def test_parse_probe_rejects_output_without_hashes():
    with pytest.raises(ProbeError):
        parse_probe("bash: find: command not found")


def test_probe_is_one_line_and_never_changes_directory():
    command = probe_command(("/app",))
    assert "\n" not in command
    assert " cd " not in f" {command}"


def test_keys_round_trip_including_the_test_call():
    adapter = HelloWorldAdapter()
    call = BashToolCall("echo 'Hello, world!' > /app/hello.txt")
    assert adapter.tool_call_from_key(call.serialize()).serialize() == call.serialize()
    assert isinstance(adapter.tool_call_from_key(TestToolCall("").serialize()), TestToolCall)


def test_rollouts_come_from_the_trace_in_order():
    adapter = HelloWorldAdapter()
    task = adapter.tasks()[0].task_id
    assert task == "hello-world"
    rollouts = adapter.sample_rollouts(task, None, 5)
    assert [c.command for c in rollouts[0]] == [c.command for c in rollouts[1]]  # solve, solve-again


def test_terminal_bench_adapter_rebuilds_calls_from_keys_and_names_tasks(monkeypatch):
    from tvcache_workloads.bash.verify import TerminalBenchAdapter

    adapter = TerminalBenchAdapter()
    key = BashToolCall("cd /app && ls").serialize()
    call = adapter.tool_call_from_key(key)
    assert call.serialize() == key and call.will_mutate_state()
    assert adapter.is_test_key(adapter.tool_call_from_key(TestToolCall("").serialize()).serialize())
    monkeypatch.setenv("TVVERIFY_TASKS", "hello-world::run7, fix-git")
    assert [(t.task_id, t.root_image_id) for t in adapter.tasks()] == [("hello-world::run7", "hello-world"), ("fix-git", "fix-git")]



def _file_hashes(root):
    """What the probe's file-content pipeline prints for root, run for real."""
    import subprocess

    from tvcache_workloads.bash import state

    command = state._contents(state.WatchPolicy(roots=(str(root),)))
    done = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=True)
    return dict(reversed(line.split("  ", 1)) for line in done.stdout.splitlines())


@pytest.mark.skipif(
    not all(__import__("shutil").which(tool) for tool in ("bash", "git", "sha256sum")),
    reason="needs bash, git and sha256sum",
)
def test_git_index_counts_by_what_it_stages_not_by_its_bytes(tmp_path):
    # Seen in training (git-workflow-hack): a snapshot and a fresh replay of
    # `git init; git add .` differed in /app/.git/index alone.
    import hashlib
    import os
    import subprocess

    def git(repo, *args):
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       check=True, capture_output=True)

    repos = []
    for name, mtime in (("a", 1_700_000_000), ("b", 1_600_000_000)):
        repo = tmp_path / name / "app"
        repo.mkdir(parents=True)
        (repo / "site.html").write_text("<p>hello</p>\n")
        os.utime(repo / "site.html", (mtime, mtime))  # the index records this
        git(repo, "init", "-q")
        git(repo, "add", ".")
        repos.append(repo)
    a, b = repos
    raw = [hashlib.sha256((r / ".git" / "index").read_bytes()).hexdigest() for r in repos]
    assert raw[0] != raw[1]  # same staged content, different bytes
    hashes_a, hashes_b = _file_hashes(a), _file_hashes(b)
    index_a, index_b = hashes_a[f"{a}/.git/index"], hashes_b[f"{b}/.git/index"]
    assert index_a == index_b and index_a != raw[0]
    assert hashes_a[f"{a}/site.html"] == hashlib.sha256(b"<p>hello</p>\n").hexdigest()  # other files: content

    # staging different content, or another file, is a different state
    (b / "site.html").write_text("<p>changed</p>\n")
    git(b, "add", ".")
    assert _file_hashes(b)[f"{b}/.git/index"] != index_a
    (a / "new.txt").write_text("x\n")
    assert _file_hashes(a)[f"{a}/.git/index"] == index_a  # untracked: the index is unchanged
    git(a, "add", "new.txt")
    assert _file_hashes(a)[f"{a}/.git/index"] != index_a

    # an index git cannot read falls back to its bytes
    (b / ".git" / "index").write_bytes(b"not an index")
    assert _file_hashes(b)[f"{b}/.git/index"] == hashlib.sha256(b"not an index").hexdigest()
