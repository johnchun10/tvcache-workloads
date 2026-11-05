"""Output comparators.

Equality of two tool outputs is judged at a level:
  L0  exact bytes
  L1  after named normalizers (timestamps, PIDs, ANSI codes, ...)
  L2  semantic (SQL result sets, per-test verdicts, reward)
A CompareResult records the level at which the outputs became equal, and which
normalizers were needed, so noise can be attributed to a source.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class CompareResult:
    equal: bool
    level: Optional[str]  # level at which they were equal; None if not equal
    normalizers_needed: Tuple[str, ...] = ()
    diff: Optional[str] = None


class Comparator(ABC):
    name: str = "comparator"

    @abstractmethod
    def compare(self, cached: str, fresh: str, *, key: str = "") -> CompareResult:
        ...


class ExactComparator(Comparator):
    name = "exact"

    def compare(self, cached: str, fresh: str, *, key: str = "") -> CompareResult:
        if cached == fresh:
            return CompareResult(True, "L0")
        return CompareResult(False, None, diff=short_diff(cached, fresh))


def short_diff(a: str, b: str, width: int = 80) -> str:
    """The first differing region of two strings, for reports."""
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    start = max(0, i - 10)
    return f"@{i}: {a[start:start + width]!r} != {b[start:start + width]!r}"
