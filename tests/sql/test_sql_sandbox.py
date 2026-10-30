"""SQLite sandbox tests; these need no TVCache server or external service."""

import asyncio
import json

import pytest
from tvclient import TestToolCall as SentinelTestCall

from tvcache_workloads.sql import (
    SQLSandboxConfig,
    SQLToolCall,
    SQLiteSandbox,
    SnapshotContractError,
    all_balances_call,
    balance_call,
    send_payment_call,
)


def run(coroutine):
    return asyncio.run(coroutine)


@pytest.fixture
def Sandbox(tmp_path):
    return SQLiteSandbox.with_config(
        SQLSandboxConfig(database_dir=tmp_path, simulated_sql_latency_seconds=0)
    )


def test_fresh_database_has_seeded_accounts_and_balances(Sandbox):
    async def scenario():
        sandbox = Sandbox(task_name="smallbank")
        await sandbox.start()
        result = json.loads(await sandbox.execute(all_balances_call()))
        await sandbox.stop()
        return result

    result = run(scenario())
    assert result["rows"] == [
        [1, "Alice", 10_000, 5_000, 15_000],
        [2, "Bob", 20_000, 7_000, 27_000],
        [3, "Carol", 30_000, 9_000, 39_000],
    ]


def test_transfer_updates_both_checking_accounts_and_preserves_total(Sandbox):
    async def scenario():
        sandbox = Sandbox(task_name="smallbank")
        await sandbox.start()
        result = await sandbox.execute(send_payment_call(1, 2, 1_000))
        balances = json.loads(await sandbox.execute(all_balances_call()))["rows"]
        test = json.loads(await sandbox.execute(SentinelTestCall()))
        await sandbox.stop()
        return result, balances, test

    result, balances, test = run(scenario())
    assert json.loads(result) == {"rowcount": 2}
    assert balances[0][2] == 9_000 and balances[1][2] == 21_000
    assert test["passed"] is True
    assert test["total_cents"] == 81_000


def test_snapshot_restore_is_durable_repeatable_and_isolated(Sandbox, tmp_path):
    async def scenario():
        parent = Sandbox(task_name="smallbank")
        await parent.start()
        await parent.execute(send_payment_call(1, 2, 1_000))
        snapshot = await parent.fork()
        handle = snapshot.get_id()
        await parent.execute(send_payment_call(2, 3, 500))

        # A newly configured sandbox class can restore the path-based handle.
        other_worker = SQLiteSandbox.with_config(
            SQLSandboxConfig(database_dir=tmp_path, simulated_sql_latency_seconds=0)
        )
        first = other_worker(task_name="smallbank", env_id=handle)
        second = other_worker(task_name="smallbank", env_id=handle)
        await first.start()
        await second.start()
        await first.execute(send_payment_call(1, 3, 700))

        result = (
            json.loads(await parent.execute(all_balances_call()))["rows"],
            json.loads(await first.execute(all_balances_call()))["rows"],
            json.loads(await second.execute(all_balances_call()))["rows"],
        )
        await first.stop()
        await second.stop()
        await snapshot.stop()
        await parent.stop()
        return result, handle

    (parent_rows, first_rows, second_rows), handle = run(scenario())
    assert handle.startswith("sql-snap:")
    assert parent_rows[1][2] == 20_500 and parent_rows[2][2] == 30_500
    assert first_rows[0][2] == 8_300 and first_rows[2][2] == 30_700
    assert second_rows[0][2] == 9_000 and second_rows[1][2] == 21_000


def test_equivalent_sql_spellings_have_same_result_and_state(Sandbox):
    async def run_call(alternate):
        sandbox = Sandbox(task_name="smallbank")
        await sandbox.start()
        result = await sandbox.execute(
            send_payment_call(1, 2, 1_000, alternate_spelling=alternate)
        )
        balances = json.loads(await sandbox.execute(balance_call(1)))
        await sandbox.stop()
        return result, balances

    standard = run(run_call(False))
    alternate = run(run_call(True))
    assert standard == alternate
    assert standard[0] == '{"rowcount": 2}'
    assert standard[1]["rows"][0][2:] == [9_000, 5_000, 14_000]


def test_read_mode_cannot_mutate_and_failed_transfer_rolls_back(Sandbox):
    async def scenario():
        sandbox = Sandbox(task_name="smallbank")
        await sandbox.start()
        denied = await sandbox.execute(
            SQLToolCall("UPDATE checking SET bal_cents = 0", mode="read")
        )
        failed = await sandbox.execute(send_payment_call(1, 2, 99_999_999))
        balances = json.loads(await sandbox.execute(balance_call(1)))
        await sandbox.stop()
        return denied, failed, balances

    denied, failed, balances = run(scenario())
    assert denied.startswith("Failed to execute SQL:")
    assert failed.startswith("Failed to execute SQL: expected 2 changed rows")
    assert balances["rows"][0][2:] == [10_000, 5_000, 15_000]


def test_snapshot_handles_must_be_inside_configured_directory(Sandbox, tmp_path):
    sandbox = Sandbox(task_name="smallbank", env_id="sql-snap:/etc/passwd")
    with pytest.raises(SnapshotContractError, match="outside the configured"):
        run(sandbox.start())