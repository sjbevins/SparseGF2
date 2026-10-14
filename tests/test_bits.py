"""The 64-bit word primitives: compiled intrinsics versus Python, and the JIT-free fallbacks."""

import os
import subprocess
import sys
import textwrap

import numpy as np
from numba import njit

from sparsegf2._kernel.bits import ctz, popcount


@njit
def _popcount(x):
    return popcount(x)


@njit
def _ctz(x):
    return ctz(x)


def _words():
    fixed = [1, 2**63, 2**64 - 1, 0x5555555555555555, 0xAAAAAAAAAAAAAAAA]
    rng = np.random.default_rng(12345)
    random = [int(w) for w in rng.integers(0, 2**64, size=1000, dtype=np.uint64)]
    return fixed + random


def test_popcount_matches_python():
    for w in [0, *_words()]:
        assert int(_popcount(np.uint64(w))) == w.bit_count()


def test_ctz_matches_lowest_set_bit():
    for w in _words():
        assert int(_ctz(np.uint64(w))) == (w & -w).bit_length() - 1


_JIT_FREE = textwrap.dedent(
    """
    import numba
    assert numba.config.DISABLE_JIT
    import numpy as np
    from sparsegf2 import Simulator
    from sparsegf2._kernel.bits import ctz, popcount

    rng = np.random.default_rng(12345)
    words = [1, 2**63, 2**64 - 1, 0x5555555555555555, 0xAAAAAAAAAAAAAAAA]
    words += [int(w) for w in rng.integers(0, 2**64, size=1000, dtype=np.uint64)]
    assert int(popcount(np.uint64(0))) == 0
    for w in words:
        assert int(popcount(np.uint64(w))) == w.bit_count(), w
        assert int(ctz(np.uint64(w))) == (w & -w).bit_length() - 1, w
    try:
        ctz(np.uint64(0))
    except ValueError:
        pass
    else:
        raise SystemExit("ctz(0) did not raise ValueError")

    sim = Simulator(2)
    sim.h(0)
    sim.cx(0, 1)
    assert sim.entropy([0]) == 1
    print("ok")
    """
)


def test_fallbacks_without_jit():
    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    env.pop("NUMBA_BOUNDSCHECK", None)
    done = subprocess.run(
        [sys.executable, "-c", _JIT_FREE],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "ok"
