"""SQL statements and SmallBank-inspired call constructors."""

import json
from typing import Sequence

from tvclient import ToolCall

READ = "read"
WRITE = "write"
MODES = (READ, WRITE)

BALANCE_SQL = """SELECT a.custid, a.name, c.bal_cents, s.bal_cents,
       c.bal_cents + s.bal_cents AS total_cents
FROM accounts AS a
JOIN checking AS c ON c.custid = a.custid
JOIN savings AS s ON s.custid = a.custid
WHERE a.custid = ?"""

ALL_BALANCES_SQL = """SELECT a.custid, a.name, c.bal_cents, s.bal_cents,
       c.bal_cents + s.bal_cents AS total_cents
FROM accounts AS a
JOIN checking AS c ON c.custid = a.custid
JOIN savings AS s ON s.custid = a.custid
ORDER BY a.custid"""

SEND_PAYMENT_SQL = """UPDATE checking
SET bal_cents = bal_cents + CASE custid
    WHEN ? THEN ?
    WHEN ? THEN ?
END
WHERE custid IN (?, ?)
  AND (SELECT bal_cents FROM checking WHERE custid = ?) >= ?"""

SEND_PAYMENT_SQL_ALTERNATE = """UPDATE checking
SET bal_cents = bal_cents + CASE
    WHEN custid = ? THEN ?
    WHEN custid = ? THEN ?
END
WHERE (custid = ? OR custid = ?)
  AND (SELECT bal_cents FROM checking WHERE custid = ?) >= ?"""


class SQLToolCall(ToolCall):
    """One parameterized SQL statement; exact SQL text is part of the cache key."""

    def __init__(
        self,
        query: str,
        params: Sequence[object] = (),
        *,
        mode: str = READ,
        expected_rowcount: int | None = None,
    ):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if expected_rowcount is not None and mode != WRITE:
            raise ValueError("expected_rowcount is only valid for write calls")

        self.query = query
        self.params = tuple(params)
        self.mode = mode
        self.expected_rowcount = expected_rowcount
        # Fail early if a value cannot be represented in a stable cache key.
        json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, value: dict) -> "SQLToolCall":
        return cls(
            query=value["query"],
            params=value.get("params", []),
            mode=value.get("mode", READ),
            expected_rowcount=value.get("expected_rowcount"),
        )

    def to_dict(self) -> dict:
        return {
            "tool": "sql",
            "query": self.query,
            "params": list(self.params),
            "mode": self.mode,
            "expected_rowcount": self.expected_rowcount,
        }

    def will_mutate_state(self) -> bool:
        return self.mode == WRITE


def balance_call(custid: int) -> SQLToolCall:
    return SQLToolCall(BALANCE_SQL, (custid,))


def all_balances_call() -> SQLToolCall:
    return SQLToolCall(ALL_BALANCES_SQL)


def send_payment_call(
    sender: int,
    recipient: int,
    amount_cents: int,
    *,
    alternate_spelling: bool = False,
) -> SQLToolCall:
    if sender == recipient:
        raise ValueError("sender and recipient must be different accounts")
    if amount_cents <= 0:
        raise ValueError("amount_cents must be positive")

    query = SEND_PAYMENT_SQL_ALTERNATE if alternate_spelling else SEND_PAYMENT_SQL
    return SQLToolCall(
        query,
        (
            sender,
            -amount_cents,
            recipient,
            amount_cents,
            sender,
            recipient,
            sender,
            amount_cents,
        ),
        mode=WRITE,
        expected_rowcount=2,
    )