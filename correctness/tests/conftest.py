# Uses whichever tvclient and tvverify are installed in the environment, so
# the same tests run against an unmodified or a patched tvclient.
import os

import pytest

from tvverify.adapters.mock_list import MockListSandbox
from tvverify.testing.server import RustServer


@pytest.fixture
def server():
    """A fresh server per test: no shared cache state between tests."""
    with RustServer() as running:
        yield running


@pytest.fixture(autouse=True)
def reset_mock():
    MockListSandbox.reset()
    yield


@pytest.fixture(scope="session")
def exact_fixture():
    """The trie built by real rollouts against a live server, plus the ToyFs
    snapshots it refers to. Built once; the server is not needed after."""
    import asyncio

    from tvverify.testing.fixtures import build_exact_fixture

    with RustServer() as running:
        return asyncio.run(build_exact_fixture(running.port))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "known_bug(name): fails on an unmodified TVCache because of bug <name> "
        "(see the README); must pass when TVVERIFY_PATCHED=1",
    )


def pytest_collection_modifyitems(config, items):
    # On an unmodified TVCache these tests catch real bugs, so their failures
    # are expected. With TVVERIFY_PATCHED=1 (the fixes applied) they must pass.
    if os.environ.get("TVVERIFY_PATCHED") == "1":
        return
    for item in items:
        marker = item.get_closest_marker("known_bug")
        if marker is not None:
            item.add_marker(pytest.mark.xfail(reason=f"bug {marker.args[0]} in unmodified TVCache", strict=False))

