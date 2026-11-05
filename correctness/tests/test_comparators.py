from tvverify.compare.base import ExactComparator
from tvverify.compare.normalizers import NormalizedComparator
from tvverify.compare.semantic import (
    SqlResultSetComparator,
    TestResultComparator,
    has_limit_without_order,
    has_top_level_order_by,
)

N = NormalizedComparator()


def same(a, b):
    return N.compare(a, b)


def test_exact():
    assert ExactComparator().compare("a", "a").level == "L0"
    assert not ExactComparator().compare("a", "b").equal


# ---- each normalizer: removes noise, and does NOT hide a real difference ----

def test_ansi_codes():
    assert same("\x1b[31mred\x1b[0m done", "red done").equal
    assert not same("\x1b[31mred\x1b[0m", "blue").equal


def test_progress_bar_carriage_returns():
    assert same("10%\r50%\r100%\ndone", "100%\ndone").equal
    assert not same("10%\r50%\ndone", "100%\ndone").equal


def test_prompt_hostname():
    assert same("root@3f2a9c0d1e5b:/app# ls", "root@a1b2c3d4e5f6:/app# ls").equal
    assert not same("root@3f2a9c0d1e5b:/app# ls", "root@3f2a9c0d1e5b:/tmp# ls").equal
    assert not same("alice@3f2a9c0d1e5b:/app#", "bob@3f2a9c0d1e5b:/app#").equal


def test_timestamps_and_durations():
    assert same("built at 2025-01-02T03:04:05Z ok", "built at 2026-12-31 23:59:59 ok").equal
    assert same("Mon Jan  2 03:04:05 UTC 2025", "Tue Feb 10 11:12:13 UTC 2026").equal
    assert same("5 passed in 0.12s", "5 passed in 3.40s").equal
    assert same("real\t0m1.234s", "real\t0m9.876s").equal
    # plain numbers and versions are not timestamps
    assert not same("version 1.2.3", "version 1.2.4").equal
    assert not same("5 passed in 0.12s", "4 passed in 0.12s").equal
    assert not same("exit code 2024", "exit code 2025").equal


def test_addresses_and_ids():
    assert same("at 0x7f3a9c001234", "at 0x7fffdeadbeef").equal
    assert not same("at 0x12", "at 0x34").equal  # short hex is data, not an address
    assert same("/tmp/tmpab12cd34/x", "/tmp/tmpzz99yy88/x").equal
    assert not same("/tmp/other/x", "/tmp/tmpzz99yy88/x").equal
    h1, h2 = "a" * 64, "b" * 64
    assert same(f"sha {h1}", f"sha {h2}").equal


def test_pytest_and_package_manager_chatter():
    a = "platform linux -- Python 3.11\nrootdir: /app\n3 passed in 0.1s\n"
    b = "platform linux -- Python 3.12\nrootdir: /other\n3 passed in 9.9s\n"
    assert same(a, b).equal
    assert not same("1 failed, 2 passed in 0.1s", "3 passed in 0.1s").equal
    assert same("Downloading foo-1.0.whl\nInstalled foo", "Installed foo").equal
    assert not same("Installed foo", "Installed bar").equal


def test_trailing_whitespace():
    assert same("a  \nb\n\n", "a\nb").equal
    assert not same("a b", "a  b").equal


def test_attribution_names_the_rules_that_were_needed():
    result = same("x 5 passed in 0.1s\n", "x 5 passed in 7.7s")
    assert result.equal and result.level == "L1"
    assert "duration" in result.normalizers_needed


# ---- SQL ----

def sql(rows):
    import json

    return json.dumps(rows)


def test_sql_unordered_result_sets_compare_as_multisets():
    c = SqlResultSetComparator(query_of_key=lambda k: "select a from t")
    assert c.compare(sql([[1], [2]]), sql([[2], [1]])).equal
    assert not c.compare(sql([[1], [2]]), sql([[1], [1]])).equal  # multiset, not set
    assert not c.compare(sql([[1]]), sql([[1], [1]])).equal


def test_sql_order_by_makes_row_order_significant():
    c = SqlResultSetComparator(query_of_key=lambda k: "select a from t order by a")
    assert not c.compare(sql([[1], [2]]), sql([[2], [1]])).equal
    assert c.compare(sql([[1], [2]]), sql([[1], [2]])).equal


def test_sql_float_rounding_and_errors():
    c = SqlResultSetComparator(query_of_key=lambda k: "select 1")
    assert c.compare(sql([[0.1 + 0.2]]), sql([[0.3]])).equal
    assert not c.compare(sql([[0.31]]), sql([[0.3]])).equal
    e = '{"error": "no such table"}'
    assert c.compare(e, '{"error": "no such table"}').equal
    assert not c.compare(e, '{"error": "syntax error"}').equal
    assert not c.compare(e, sql([[1]])).equal


def test_order_by_detection_is_top_level_only():
    assert has_top_level_order_by("SELECT a FROM t ORDER BY a")
    assert has_top_level_order_by("select a from t order  by a desc limit 3")
    assert not has_top_level_order_by("select a from (select a from t order by a) s")
    assert not has_top_level_order_by("select 'order by' as x from t")
    assert not has_top_level_order_by("select border_by from t")
    assert has_limit_without_order("select a from t limit 3")
    assert not has_limit_without_order("select a from t order by a limit 3")


def test_test_result_comparator_levels():
    def parse(out):
        reward, _, vec = out.partition("|")
        return float(reward), dict(item.split("=") for item in vec.split(",") if item)

    c = TestResultComparator(parse)
    assert c.compare("1|a=1,b=1", "1|a=1,b=1").equal
    assert c.compare("1|a=1,b=1", "1|b=1,a=1").level == "L1"
    flaky = c.compare("1|a=1,b=0", "1|a=0,b=1")
    assert flaky.equal and "verdict_vector" in flaky.normalizers_needed
    assert not c.compare("1|a=1", "0|a=0").equal
