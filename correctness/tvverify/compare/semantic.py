"""L2 comparators: SQL result sets and test verdicts."""

import json
import re
from typing import Any, Callable, List, Optional, Tuple

from tvverify.compare.base import CompareResult, Comparator, short_diff

# parse(output) -> ("rows", [row, ...]) | ("error", message) | ("raw", text)
SqlParse = Callable[[str], Tuple[str, Any]]


def parse_json_rows(output: str) -> Tuple[str, Any]:
    """Default SQL output parser: a JSON list of rows, or an error string."""
    try:
        data = json.loads(output)
    except (ValueError, TypeError):
        return ("raw", output)
    if isinstance(data, list):
        return ("rows", [tuple(r) if isinstance(r, list) else (r,) for r in data])
    if isinstance(data, dict) and "error" in data:
        return ("error", str(data["error"]))
    return ("raw", output)


def has_top_level_order_by(sql: str) -> bool:
    """True if the statement has an ORDER BY outside parentheses and strings.
    An ORDER BY inside a subquery does not order the final result."""
    depth = 0
    in_string: Optional[str] = None
    i = 0
    text = sql
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == in_string:
                in_string = None
        elif ch in "'\"`":
            in_string = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and re.match(r"order\s+by\b", text[i:], re.I) and (
            i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
        ):
            return True
        i += 1
    return False


def has_limit_without_order(sql: str) -> bool:
    """LIMIT without a top-level ORDER BY picks an arbitrary subset: the query
    itself is nondeterministic, so disagreement says nothing about the cache."""
    return bool(re.search(r"\blimit\b", sql, re.I)) and not has_top_level_order_by(sql)


class SqlResultSetComparator(Comparator):
    name = "sql_result_set"

    def __init__(
        self,
        parse: SqlParse = parse_json_rows,
        query_of_key: Callable[[str], str] = lambda key: key,
        float_digits: int = 6,
    ):
        self.parse = parse
        self.query_of_key = query_of_key
        self.float_digits = float_digits

    def _canon_row(self, row: tuple) -> tuple:
        return tuple(round(v, self.float_digits) if isinstance(v, float) else v for v in row)

    def compare(self, cached: str, fresh: str, *, key: str = "") -> CompareResult:
        if cached == fresh:
            return CompareResult(True, "L0")
        kind_a, a = self.parse(cached)
        kind_b, b = self.parse(fresh)
        if kind_a != kind_b:
            return CompareResult(False, None, diff=short_diff(cached, fresh))
        if kind_a == "rows":
            rows_a = [self._canon_row(r) for r in a]
            rows_b = [self._canon_row(r) for r in b]
            ordered = has_top_level_order_by(self.query_of_key(key))
            if not ordered:
                rows_a, rows_b = sorted(map(repr, rows_a)), sorted(map(repr, rows_b))
            if rows_a == rows_b:
                return CompareResult(True, "L2")
            return CompareResult(False, None, diff=short_diff(repr(rows_a), repr(rows_b)))
        if kind_a == "error" and a == b:
            return CompareResult(True, "L2")
        return CompareResult(False, None, diff=short_diff(cached, fresh))


# parse(output) -> (reward, {test_name: passed})
TestParse = Callable[[str], Tuple[float, dict]]


class TestResultComparator(Comparator):
    """L1: the per-test pass/fail vectors agree. L2: the rewards agree.
    Equal vectors imply equal rewards; equal rewards with different vectors is
    a flaky-test signature that the report counts separately."""

    __test__ = False  # not a pytest class
    name = "test_result"

    def __init__(self, parse: TestParse):
        self.parse = parse

    def compare(self, cached: str, fresh: str, *, key: str = "") -> CompareResult:
        if cached == fresh:
            return CompareResult(True, "L0")
        reward_a, vec_a = self.parse(cached)
        reward_b, vec_b = self.parse(fresh)
        if vec_a == vec_b and reward_a == reward_b:
            return CompareResult(True, "L1")
        if reward_a == reward_b:
            return CompareResult(True, "L2", ("verdict_vector",))
        return CompareResult(False, None, diff=f"reward {reward_a} != {reward_b}")
