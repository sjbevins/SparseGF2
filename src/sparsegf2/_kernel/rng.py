"""A small, fast, reproducible PRNG for use inside compiled kernels.

We use xoshiro256** (Blackman and Vigna, 2018): 256 bits of state, period
2**256 - 1, excellent statistical quality, and a handful of integer
instructions per draw. Every simulator and every trajectory of an ensemble owns
its own state array, so results never depend on thread scheduling.

State is a length-4 ``uint64`` array. Seed it from a ``numpy.random.SeedSequence``
with :func:`seed_state` (Python side), never with all zeros.

Streams (randomness contract 2)
-------------------------------
Many independent streams come from one 128-bit key ``(key0, key1)`` (from
:func:`stream_key` or a run's ``SeedSequence``) and a 64-bit stream index. With
``mix64`` the splitmix64 output function and ``sm(x) = mix64(x + GOLDEN)``, the
output of one splitmix64 step from state ``x`` (``GOLDEN = 0x9E3779B97F4A7C15``,
all arithmetic mod 2**64):

* :func:`stream_state`: ``x = key0 ^ sm(index)``, ``gamma = sm(key1)``, and
  state word ``j`` (``j = 0..3``) is ``mix64(x + (j + 1) * gamma)``: four
  splitmix64 steps from ``x`` whose increment comes from ``key1``. For a fixed
  index, distinct keys always give distinct states (``x + gamma`` and
  ``x + 2 gamma`` are the ``mix64``-inverses of words 0 and 1, and give back
  the key). A change of ``key0`` or of the index changes ``x`` and so every
  word; a change of ``key1`` changes every word unless it changes ``gamma``
  by a multiple of 2**62 (chance about 2**-62, which is why ``key1`` passes
  through ``sm`` rather than being added as it is: a raw top bit would
  change only two words). ``gamma = 0`` (one ``key1`` in 2**64) makes the
  four words equal, a valid state. An all-zero state (never seen) becomes
  ``[1, 0, 0, 0]``.
* :func:`mix`: ``mix(a, b, c) = sm(sm(sm(a) ^ b) ^ c)``, a 64-bit stream index
  from three words (chained splitmix64; one-to-one in each word when the other
  two are fixed). The ensemble engine gives each sampled-probe evaluation the
  stream ``mix(trajectory, probe-name hash, layer)``.

Contract 1 used ``x = key0 ^ (index * 0xD1B54A32D192ED03) ^ key1`` and four
splitmix64 steps from ``x``, so only ``key0 ^ key1`` mattered.

Changing a stream: Numba keys each cached function on a hash of its own
source file, and a cached caller keeps the callees it was compiled with. A
change to these functions must therefore also change the text of every
module whose compiled code calls them (``ensemble.py``, ``program.py``,
``entropy.py``; a new timestamp is not enough), or a warm cache keeps the old
streams there. Bump ``mipt.RANDOMNESS_CONTRACT`` too.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_U = np.uint64


def seed_state(seed) -> np.ndarray:
    """Return a fresh xoshiro256** state from an int, a sequence of ints, a SeedSequence, or None.

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


_GOLDEN = 0x9E3779B97F4A7C15


@njit(cache=True, inline="always")
def _mix64(z):
    """The splitmix64 output function (a bijection of 64-bit words)."""
    z = (z ^ (z >> _U(30))) * _U(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> _U(27))) * _U(0x94D049BB133111EB)
    return z ^ (z >> _U(31))


@njit(cache=True, inline="always")
def _sm(x):
    """The output of one splitmix64 step from state ``x``: ``mix64(x + GOLDEN)``."""
    return _mix64(x + _U(_GOLDEN))


@njit(cache=True)
def mix(a, b, c):
    """A 64-bit stream index from three words: ``sm(sm(sm(a) ^ b) ^ c)`` (module docstring)."""
    return _sm(_sm(_sm(_U(a)) ^ _U(b)) ^ _U(c))


@njit(cache=True)
def stream_state(key, index, out):
    """Fill ``out`` (length 4) with the xoshiro state of stream ``index`` under ``key``.

    ``key`` is a length-2 uint64 array (from :func:`stream_key`). Streams for
    different indices are independent, and stream ``index`` depends only on
    ``(key, index)``, so trajectory/shot ``i`` is reproducible on its own and
    regardless of how work is split across threads. The derivation uses all
    128 bits of the key (module docstring, randomness contract 2).
    """
    x = key[0] ^ _sm(_U(index))
    gamma = _sm(key[1])
    for j in range(4):
        x = x + gamma
        out[j] = _mix64(x)
    if out[0] == _U(0) and out[1] == _U(0) and out[2] == _U(0) and out[3] == _U(0):
        out[0] = _U(1)  # pragma: no cover


def stream_key(seed) -> np.ndarray:
    """A 128-bit key for :func:`stream_state` from an int, SeedSequence, or None."""
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    return ss.generate_state(2, dtype=np.uint64)
