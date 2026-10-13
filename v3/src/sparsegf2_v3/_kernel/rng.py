"""A small, fast, reproducible PRNG for use inside compiled kernels.

We use xoshiro256** (Blackman and Vigna, 2018): 256 bits of state, period
2**256 - 1, excellent statistical quality, and a handful of integer
instructions per draw. Every simulator and every trajectory of an ensemble owns
its own state array, so results never depend on thread scheduling.

State is a length-4 ``uint64`` array. Seed it from a ``numpy.random.SeedSequence``
with :func:`seed_state` (Python side), never with all zeros.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_U = np.uint64


def seed_state(seed) -> np.ndarray:
    """Return a fresh xoshiro256** state from an int, a SeedSequence, or None.

    ``None`` draws OS entropy. The result is never all zeros.
    """
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    state = ss.generate_state(4, dtype=np.uint64)
    if not state.any():  # pragma: no cover (probability 2**-256)
        state[0] = 1
    return state


@njit(cache=True, inline="always")
def _rotl(x, k):
    return (x << _U(k)) | (x >> _U(64 - k))


@njit(cache=True)
def next_u64(s):
    """Advance the state and return 64 random bits."""
    result = _rotl(s[1] * _U(5), 7) * _U(9)
    t = s[1] << _U(17)
    s[2] ^= s[0]
    s[3] ^= s[1]
    s[1] ^= s[2]
    s[0] ^= s[3]
    s[2] ^= t
    s[3] = _rotl(s[3], 45)
    return result


@njit(cache=True)
def next_bit(s):
    """One uniformly random bit (0 or 1) as int64."""
    return np.int64(next_u64(s) >> _U(63))


@njit(cache=True)
def next_double(s):
    """A uniform float in [0, 1) with 53 random bits."""
    return np.float64(next_u64(s) >> _U(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def next_below(s, bound):
    """A uniform integer in [0, bound) for 1 <= bound < 2**32 (Lemire's method, unbiased)."""
    b = _U(bound)
    x = next_u64(s) >> _U(32)
    m = x * b
    low = m & _U(0xFFFFFFFF)
    if low < b:
        threshold = (_U(0x100000000) - b) % b
        while low < threshold:
            x = next_u64(s) >> _U(32)
            m = x * b
            low = m & _U(0xFFFFFFFF)
    return np.int64(m >> _U(32))
