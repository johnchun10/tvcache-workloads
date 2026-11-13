"""What counts as a sandbox's state, and the read-only probe that hashes it.

Two users:
  - graph contraction (tvclient's Contractor) merges two cached commands when
    TerminalBenchSandbox.equals() says they left the container in the same
    state, so this decides which merges happen;
  - tvverify compares a restored snapshot with a fresh replay.

The probe is one shell line run in the container. It prints
`tvv-h <component> <hash>` lines and, when asked, capped `tvv-m <component> ...`
manifest lines that say which file or variable differs. Components:

  fs   every file, directory and symlink under the watched roots (type, mode,
       link target, content), minus the excluded paths
  env  exported variables, except bookkeeping (_, SHLVL, PWD, TMUX*, HOSTNAME)
  cwd  the current directory
  pkg  installed software: the dpkg package list and the names in
       /usr/local/bin and the Python and Node package directories
  oldpwd  advisory only: restoring a snapshot replays `cd`, which overwrites it

Equality uses the strict components (everything but oldpwd). Like snapshots,
the probe cannot see processes or unexported variables.

The policy (roots, excludes, ignored variables) comes from the environment so
every process of a run agrees on it:

  TVCACHE_STATE_ROOTS     space-separated roots. Default: /app
  TVCACHE_STATE_EXCLUDE   space-separated `find -path` patterns to skip, added
                          to RUNTIME_EXCLUDES
  TVCACHE_STATE_PACKAGES  1 adds the pkg component. Default: 0
  TVCACHE_STATE_FILE      JSON written by state_check.py: per task, the paths
                          and variables found to differ between two sandboxes
                          in the same state, and whether the task can be
                          compared at all

The defaults are the narrow policy tvverify has always used. Contraction runs
set the wide one (WIDE_ROOTS, packages on) and calibrate it per task first.
"""

import hashlib
import json
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

# Bookkeeping that differs between terminals without being task state
# (HOSTNAME is the container id). OLDPWD is state, but is checked separately
# as an advisory component (see the module docstring).
IGNORED_ENV = ("_", "SHLVL", "PWD", "OLDPWD", "TMUX", "TMUX_PANE", "HOSTNAME")
ADVISORY = ("oldpwd",)
MANIFEST_LIMIT = 300

NARROW_ROOTS = ("/app",)
WIDE_ROOTS = ("/app", "/root", "/etc", "/tmp")
# Written by the container runtime or the manager, not by the task's commands:
# name resolution files (podman generates them per container), the manager's
# tmux-resurrect copy and its saved state, tmux's socket directory, caches.
RUNTIME_EXCLUDES = (
    "/etc/hosts",
    "/etc/hostname",
    "/etc/resolv.conf",
    "/etc/mtab",
    "/root/tmux_resurrect",
    "/root/target_env.sh",  # the manager's saved environment, rewritten on every restore
    "/root/.tmux",
    "/root/.cache",
    "/root/.bash_history",
    "/tmp/tmux-*",
)
PACKAGE_DIRS = (
    "/usr/local/bin",
    "/usr/local/lib/python3*/site-packages",
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/node_modules",
)
# More learned excludes than this and the probe line gets unwieldy.
MAX_LEARNED_PATHS = 40


class ProbeError(Exception):
    """The state probe's output could not be read."""


@dataclass(frozen=True)
class WatchPolicy:
    roots: Tuple[str, ...] = NARROW_ROOTS
    excludes: Tuple[str, ...] = ()
    ignored_env: Tuple[str, ...] = IGNORED_ENV
    packages: bool = False

    @property
    def strict(self) -> Tuple[str, ...]:
        return ("cwd", "env", "fs") + (("pkg",) if self.packages else ())

    def with_learned(self, paths: Sequence[str], env: Sequence[str]) -> "WatchPolicy":
        """Adds paths (literal, not patterns) and variables found to be noise."""
        return replace(
            self,
            excludes=self.excludes + tuple(escape_pattern(p) for p in paths),
            ignored_env=self.ignored_env + tuple(v for v in env if v not in self.ignored_env),
        )

    def policy_hash(self) -> str:
        description = {
            "roots": list(self.roots),
            "excludes": list(self.excludes),
            "ignored_env": list(self.ignored_env),
            "packages": self.packages,
        }
        return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()[:16]

    @staticmethod
    def base_from_env() -> "WatchPolicy":
        """The run's policy before any per-task calibration."""
        roots = tuple(os.environ.get("TVCACHE_STATE_ROOTS", "").split()) or NARROW_ROOTS
        extra = tuple(os.environ.get("TVCACHE_STATE_EXCLUDE", "").split())
        relevant = tuple(
            p for p in RUNTIME_EXCLUDES if any(p == r or p.startswith(r.rstrip("/") + "/") for r in roots)
        )
        packages = os.environ.get("TVCACHE_STATE_PACKAGES", "0").strip().lower() in {"1", "true", "yes", "on"}
        return WatchPolicy(roots=roots, excludes=relevant + extra, packages=packages)

    @staticmethod
    def for_task(task: str) -> "WatchPolicy":
        """The run's policy plus what state_check.py learned about this task."""
        policy = WatchPolicy.base_from_env()
        learned = load_calibration().get(task)
        if learned:
            policy = policy.with_learned(learned.get("paths", []), learned.get("env", []))
        return policy


_calibration_cache: Dict[str, Dict[str, dict]] = {}


def load_calibration() -> Dict[str, dict]:
    """task -> {"paths": [...], "env": [...], "comparable": bool, "reason": str}
    from TVCACHE_STATE_FILE; empty when the variable is unset."""
    path = os.environ.get("TVCACHE_STATE_FILE", "")
    if not path:
        return {}
    if path not in _calibration_cache:
        with open(path) as handle:
            _calibration_cache[path] = json.load(handle).get("tasks", {})
    return _calibration_cache[path]


def quote(text: str) -> str:
    """text as one single-quoted shell word."""
    return "'" + text.replace("'", "'\\''") + "'"


def escape_pattern(path: str) -> str:
    """A `find -path` pattern that matches exactly this path."""
    return "".join("\\" + c if c in "*?[]\\" else c for c in path)


def _find(policy: WatchPolicy, tail: str) -> str:
    roots = " ".join(quote(r) for r in policy.roots)
    if not policy.excludes:
        return f"find {roots} -xdev {tail}"
    prune = " -o ".join(f"-path {quote(p)}" for p in policy.excludes)
    return f"find {roots} -xdev \\( {prune} \\) -prune -o {tail}"


def _tree(policy: WatchPolicy) -> str:
    """type, mode, path and link target of every entry, sorted."""
    entries = "\\( -type d -o -type f -o -type l \\) -printf '%y %m %p -> %l\\n'"
    return _find(policy, entries) + " 2>/dev/null | sort"


def _contents(policy: WatchPolicy) -> str:
    """`<sha256>  <path>` of every regular file, sorted by path."""
    return _find(policy, "-type f -print0") + " 2>/dev/null | sort -z | xargs -0 -r sha256sum 2>/dev/null"


def _env_lines(policy: WatchPolicy) -> str:
    skip = " ".join(policy.ignored_env)
    return (
        f'for v in $(compgen -e | sort); do [[ " {skip} " == *" $v "* ]] && continue; '
        'printf "%s=%s\\n" "$v" "${!v}"; done'
    )


def _packages() -> str:
    dirs = " ".join(PACKAGE_DIRS)  # unquoted on purpose: one of them is a glob
    return "{ dpkg-query -W -f '${Package} ${Version}\\n' 2>/dev/null; ls -1 " + dirs + " 2>/dev/null; }"


# Manifest lines: `tvv-m fs <12 hex of content> <path>` for files, and
# `tvv-m fs <type><mode> <path>[ -> <target>]` for directories and symlinks.
_FILE_MANIFEST = "awk '{print \"tvv-m fs \" substr($1,1,12) \" \" substr($0,67)}'"
_OTHER_MANIFEST = (
    "awk '$1!=\"f\" {line=substr($0, index($0,$3)); sub(/ -> $/, \"\", line); "
    "print \"tvv-m fs \" $1 $2 \" \" line}'"
)


def probe_command(policy: WatchPolicy, manifest: bool = True) -> str:
    """One line (the manager joins multi-line commands with &&). It reads
    state only: absolute paths, no cd, no writes."""
    tree, contents, env_lines = _tree(policy), _contents(policy), _env_lines(policy)
    parts = [
        "printf 'tvv-h cwd %s\\n' \"$(pwd | sha256sum | cut -c1-16)\"",
        "printf 'tvv-h oldpwd %s\\n' \"$(printf %s \"$OLDPWD\" | sha256sum | cut -c1-16)\"",
        f"printf 'tvv-h env %s\\n' \"$({env_lines} | sha256sum | cut -c1-16)\"",
        f"printf 'tvv-h fs %s\\n' \"$({{ {tree}; {contents}; }} | sha256sum | cut -c1-16)\"",
    ]
    if policy.packages:
        parts.append(f"printf 'tvv-h pkg %s\\n' \"$({_packages()} | sha256sum | cut -c1-16)\"")
    if manifest:
        parts += [
            "printf 'tvv-m cwd %s\\n' \"$(pwd)\"",
            f"{env_lines} | while IFS='=' read -r k val; do "
            "printf 'tvv-m env %s %s\\n' \"$(printf %s \"$val\" | sha256sum | cut -c1-12)\" \"$k\"; "
            f"done | head -{MANIFEST_LIMIT}",
            f"{contents} | {_FILE_MANIFEST} | head -{MANIFEST_LIMIT}",
            f"{tree} | {_OTHER_MANIFEST} | head -{MANIFEST_LIMIT}",
        ]
    return "; ".join(parts)


def listing_command(policy: WatchPolicy, start: int, count: int = MANIFEST_LIMIT) -> str:
    """Lines start..start+count-1 (1-based) of the full fs manifest, for
    paging through trees too large for one capture. Ends with `tvv-h page`
    so an empty page is still a readable answer."""
    listing = f"{{ {_contents(policy)} | {_FILE_MANIFEST}; {_tree(policy)} | {_OTHER_MANIFEST}; }}"
    return f"{listing} | sed -n '{start},{start + count - 1}p'; printf 'tvv-h page %s\\n' {start}"


def parse_probe(output: str, required: Sequence[str] = ("cwd", "env", "fs")) -> Tuple[dict, dict]:
    """(component -> hash, component -> {entry: value}) from probe output.
    tmux wraps long rows; a row that does not start a record continues the
    previous one."""
    records: List[str] = []
    for row in output.splitlines():
        if row.startswith("tvv-"):
            records.append(row)
        elif records:
            records[-1] += row
    hashes: Dict[str, str] = {}
    manifests: Dict[str, Dict[str, str]] = {}
    for record in records:
        parts = record.split(" ", 3)
        if parts[0] == "tvv-h" and len(parts) == 3:
            hashes[parts[1]] = parts[2]
        elif parts[0] == "tvv-m" and len(parts) > 1 and parts[1] == "cwd":
            manifests.setdefault("cwd", {})["cwd"] = record.split(" ", 2)[2] if len(parts) > 2 else ""
        elif parts[0] == "tvv-m" and len(parts) == 4:
            # env: <hash> <name>; fs: <hash or type+mode> <path>
            manifests.setdefault(parts[1], {})[parts[3]] = parts[2]
    missing = set(required) - set(hashes)
    if missing:
        raise ProbeError(f"state probe printed no hash for {sorted(missing)}: {output[:300]!r}")
    return hashes, manifests


def digest_dict(
    hashes: Dict[str, str],
    manifests: Dict[str, Dict[str, str]],
    policy: WatchPolicy,
    root_image_id: str = "",
) -> dict:
    """A state digest in the plain-dict form tvclient's Sandbox.state_digest()
    returns (the same shape as tvverify's StateDigest.to_dict())."""
    names = [n for n in policy.strict + ADVISORY if n in hashes]
    return {
        "version": "1",
        "policy_hash": policy.policy_hash(),
        "root_image_id": root_image_id,
        "probe_seconds": 0.0,
        "components": {
            name: {"hash": hashes[name], "strict": name in policy.strict, "manifest": manifests.get(name, {})}
            for name in names
        },
    }


def strict_hashes(digest: Optional[dict]) -> Optional[Dict[str, str]]:
    """The hashes that decide equality, or None when there is no digest."""
    if digest is None:
        return None
    return {n: c["hash"] for n, c in digest["components"].items() if c.get("strict", True)}
