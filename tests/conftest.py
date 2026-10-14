"""Shared test configuration.

Stim is the reference oracle for this test suite. If it is missing, the whole
suite fails at collection time instead of silently skipping comparisons (the v2
suite lost half its coverage that way).
"""

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
