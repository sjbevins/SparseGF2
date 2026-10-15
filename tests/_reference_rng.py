"""Pure-Python reference for the kernel's random number generator.

An independent transcription of the published algorithms, written with Python ints
and explicit ``& MASK64``. It never imports ``sparsegf2._kernel``.

* xoshiro256** 1.0 (Blackman and Vigna, 2018, https://prng.di.unimi.it/).
* splitmix64 (Steele, Lea and Flood; Vigna's reference ``splitmix64.c``).
* Lemire's nearly divisionless bounded integers (Lemire, 2019, "Fast random integer
  generation in an interval"), here on 32-bit draws.

Where the kernel's conventions go beyond those algorithms, this module states them as
contract conventions (``mipt.RANDOMNESS_CONTRACT``). The draw conventions are the same
in contracts 1 and 2:

* ``next_bit`` is the top bit of a draw;
* ``next_double`` is the top 53 bits of a draw times 2**-53;
* ``next_below`` takes ``x = next_u64() >> 32`` and applies Lemire's method with
  32-bit words (the rejection threshold is computed only when ``low32(m) < bound``).

The stream derivation ``stream_state(key, index)`` differs, so it takes ``contract``.
With ``mix64`` the splitmix64 output function and ``sm(x) = mix64(x + GOLDEN)`` the
output of one splitmix64 step from state ``x`` (all arithmetic mod 2**64):

* contract 1: ``x = key0 ^ (index * 0xD1B54A32D192ED03) ^ key1``, then four splitmix64
  steps from ``x``. Only ``key0 ^ key1`` matters (kernel finding S1).
* contract 2: ``x = key0 ^ sm(index)`` and ``gamma = sm(key1)``; word ``j``
  (``j = 0..3``) is ``mix64(x + (j + 1) * gamma)``. For a fixed index this is one-to-one
  in the 128-bit key: ``(x, gamma)`` follows from the key one-to-one, and
  ``x + gamma = mix64^-1(word 0)``, ``x + 2 gamma = mix64^-1(word 1)`` give it back.

In both, an all-zero state becomes ``[1, 0, 0, 0]``. Contract 2 adds ``mix(a, b, c) =
sm(sm(sm(a) ^ b) ^ c)``, the stream index of one sampled-probe evaluation.
"""

from __future__ import annotations

MASK64 = (1 << 64) - 1

# The contract this module models by default (the package's current ``RANDOMNESS_CONTRACT``).
# Every older contract stays available through ``stream_state(..., contract=c)``.
RANDOMNESS_CONTRACT_MODEL = 2
CONTRACTS = (1, 2)

_GOLDEN = 0x9E3779B97F4A7C15
_STREAM_MULTIPLIER = 0xD1B54A32D192ED03


def mix64(z: int) -> int:
    """The splitmix64 output function (a bijection of 64-bit words)."""
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def splitmix64(x: int) -> tuple[int, int]:
    """One splitmix64 step: return ``(new state, output)``."""
    x = (x + _GOLDEN) & MASK64
    return x, mix64(x)


def mix(a: int, b: int, c: int) -> int:
    """``sm(sm(sm(a) ^ b) ^ c)`` with ``sm(x)`` the output of a splitmix64 step from ``x``."""
    x = splitmix64(a & MASK64)[1]
    x = splitmix64(x ^ (b & MASK64))[1]
    return splitmix64(x ^ (c & MASK64))[1]


def stream_state(
    key: tuple[int, int], index: int, contract: int = RANDOMNESS_CONTRACT_MODEL
) -> list[int]:
    """The four xoshiro words of stream ``index`` under ``key`` (see the module docstring)."""
    key0, key1, index = key[0] & MASK64, key[1] & MASK64, index & MASK64
    words = []
    if contract == 1:
        x = key0 ^ ((index * _STREAM_MULTIPLIER) & MASK64) ^ key1
        for _ in range(4):
            x, z = splitmix64(x)
            words.append(z)
    elif contract == 2:
        x = key0 ^ splitmix64(index)[1]
        gamma = splitmix64(key1)[1]
        for _ in range(4):
            x = (x + gamma) & MASK64
            words.append(mix64(x))
    else:
        raise ValueError(f"unknown randomness contract {contract}; known: {CONTRACTS}")
    if not any(words):
        words[0] = 1
    return words


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & MASK64


class Xoshiro:
    """xoshiro256** with the contract draw conventions (the same in every contract)."""

    def __init__(self, state: list[int]):
        if len(state) != 4:
            raise ValueError("state must have 4 words")
        self.s = [int(w) & MASK64 for w in state]
        self.rejections = 0  # draws next_below discarded (for tests of the rejection path)

    def next_u64(self) -> int:
        s = self.s
        result = (_rotl((s[1] * 5) & MASK64, 7) * 9) & MASK64
        t = (s[1] << 17) & MASK64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl(s[3], 45)
        return result

    def next_bit(self) -> int:
        return self.next_u64() >> 63

    def next_double(self) -> float:
        return (self.next_u64() >> 11) * 2.0**-53

    def next_below(self, bound: int) -> int:
        """Lemire on 32-bit words; ``1 <= bound < 2**32``."""
        if not 1 <= bound < 1 << 32:
            raise ValueError("bound must satisfy 1 <= bound < 2**32")
        x = self.next_u64() >> 32
        m = x * bound
        low = m & 0xFFFFFFFF
        if low < bound:
            threshold = ((1 << 32) - bound) % bound
            while low < threshold:
                self.rejections += 1
                x = self.next_u64() >> 32
                m = x * bound
                low = m & 0xFFFFFFFF
        return m >> 32
