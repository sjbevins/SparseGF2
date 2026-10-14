"""Shared test configuration.

Stim is the reference oracle for this test suite. If it is missing, the whole
suite fails at collection time instead of silently skipping comparisons (the v2
suite lost half its coverage that way).
"""

import os
import zlib

import numpy as np
import pytest

try:
    import stim  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The SparseGF2 test suite requires Stim, which is the reference every "
        "result is checked against. Install the test extras: pip install -e '.[test]'"
    ) from exc


@pytest.fixture
def rng(request) -> np.random.Generator:
    """A generator seeded from the test's node id, so every test is reproducible
    and different tests draw different numbers."""
    # crc32, not hash(): str hashes are salted per process and would not repeat.
    return np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))


# ---------------------------------------------------------------------------
# No silent skips. With SPARSEGF2_NO_SKIPS=1 (set in CI) any skipped test fails
# the session, except the docs test on a machine without bash.
# ---------------------------------------------------------------------------

_ALLOWED_SKIP_REASONS = {"no working bash"}
_skips: list[tuple[str, str]] = []


def _skip_reason(report) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple):  # (path, line, "Skipped: reason")
        return str(longrepr[2]).removeprefix("Skipped: ")
    return str(longrepr)


def pytest_runtest_logreport(report):
    if report.skipped and not hasattr(report, "wasxfail"):
        _skips.append((report.nodeid, _skip_reason(report)))


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("SPARSEGF2_NO_SKIPS") != "1":
        return
    bad = [(n, r) for n, r in _skips if r not in _ALLOWED_SKIP_REASONS]
    if bad:
        print("\nSPARSEGF2_NO_SKIPS=1 and these tests were skipped:")
        for nodeid, reason in bad:
            print(f"  {nodeid}: {reason}")
        if session.exitstatus == 0:  # keep codes such as 2 (interrupted) and 5 (no tests)
            session.exitstatus = 1
