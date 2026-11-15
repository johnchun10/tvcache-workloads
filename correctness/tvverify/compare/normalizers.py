"""Named output normalizers and the NormalizedComparator.

Each rule is (name, function). Rules must only remove *noise*: every rule has a
self-test in tests/test_comparators.py showing a real difference survives it.
"""

import re
from typing import Callable, List, Sequence, Tuple

from tvverify.compare.base import CompareResult, Comparator, short_diff

Rule = Tuple[str, Callable[[str], str]]


def _sub(pattern: str, repl: str, flags: int = 0) -> Callable[[str], str]:
    compiled = re.compile(pattern, flags)
    return lambda text: compiled.sub(repl, text)


def _collapse_carriage_returns(text: str) -> str:
    # A progress bar redraws one line with \r; only the last redraw is visible.
    return "\n".join(line.split("\r")[-1] for line in text.replace("\r\n", "\n").split("\n"))


def _strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.split("\n")).rstrip("\n")


def _drop_lines(prefixes: Sequence[str]) -> Callable[[str], str]:
    pattern = re.compile(r"^\s*(?:" + "|".join(re.escape(p) for p in prefixes) + r").*$\n?", re.M)
    return lambda text: pattern.sub("", text)


def _sort_runs(prefix: str) -> Callable[[str], str]:
    """Sorts each run of consecutive lines that start with prefix. pip prints
    its "Requirement already satisfied" lines in an order that changes from
    run to run; the set of lines (names and versions) still has to match."""

    def sort(text: str) -> str:
        out: List[str] = []
        run: List[str] = []
        for line in text.split("\n"):
            if line.lstrip().startswith(prefix):
                run.append(line)
                continue
            out.extend(sorted(run))
            run = []
            out.append(line)
        out.extend(sorted(run))
        return "\n".join(out)

    return sort


_MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
_DAYS = "Mon|Tue|Wed|Thu|Fri|Sat|Sun"

DEFAULT_RULES: List[Rule] = [
    ("ansi", _sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "")),
    ("carriage_return", _collapse_carriage_returns),
    # Container hostnames are 12 hex chars and appear in the shell prompt.
    ("prompt_host", _sub(r"(\w+)@[0-9a-f]{12}:", r"\1@HOST:")),
    ("container_id", _sub(r"\b[0-9a-f]{64}\b", "<ID64>")),
    ("hex_address", _sub(r"\b0x[0-9a-fA-F]{6,}\b", "0x<ADDR>")),
    ("tmp_name", _sub(r"/tmp/tmp[A-Za-z0-9_]{8}\b", "/tmp/tmp<RAND>")),
    (
        "iso_timestamp",
        _sub(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?", "<TS>"),
    ),
    (
        "date_default",
        _sub(rf"\b(?:{_DAYS}) (?:{_MONTHS}) +\d{{1,2}} \d{{2}}:\d{{2}}:\d{{2}}(?: [A-Z]{{3,5}})? \d{{4}}", "<TS>"),
    ),
    ("duration", _sub(r"\bin \d+(?:\.\d+)?s\b", "in <T>s")),
    ("time_builtin", _sub(r"\b(real|user|sys)\s+\d+m\d+(?:\.\d+)?s", r"\1 <T>")),
    ("job_pid", _sub(r"(?<=\[\d\] )\d+\b", "<PID>")),
    ("pytest_header", _drop_lines(["platform ", "rootdir:", "plugins:", "cachedir:", "configfile:"])),
    ("pip_progress", _drop_lines(["Downloading ", "Using cached ", "━", "──"])),
    ("pip_requirement_order", _sort_runs("Requirement already satisfied: ")),
    ("apt_progress", _drop_lines(["Get:", "Hit:", "Ign:", "Fetched "])),
    ("trailing_whitespace", _strip_trailing_whitespace),
]


class NormalizedComparator(Comparator):
    name = "normalized"

    def __init__(self, rules: Sequence[Rule] = tuple(DEFAULT_RULES)):
        self.rules = list(rules)

    def normalize(self, text: str) -> Tuple[str, Tuple[str, ...]]:
        """Returns the normalized text and the names of rules that changed it."""
        changed = []
        for name, fn in self.rules:
            new = fn(text)
            if new != text:
                changed.append(name)
            text = new
        return text, tuple(changed)

    def compare(self, cached: str, fresh: str, *, key: str = "") -> CompareResult:
        if cached == fresh:
            return CompareResult(True, "L0")
        a, rules_a = self.normalize(cached)
        b, rules_b = self.normalize(fresh)
        if a == b:
            needed = tuple(sorted(set(rules_a) | set(rules_b)))
            return CompareResult(True, "L1", needed)
        return CompareResult(False, None, diff=short_diff(a, b))
