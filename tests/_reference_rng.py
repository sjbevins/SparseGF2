"""Pure-Python reference for the kernel's random number generator.

An independent transcription of the published algorithms, written with Python ints
and explicit ``& MASK64``. It never imports ``sparsegf2._kernel``.

* xoshiro256** 1.0 (Blackman and Vigna, 2018, https://prng.di.unimi.it/).
* splitmix64 (Steele, Lea and Flood; Vigna's reference ``splitmix64.c``).
* Lemire's nearly divisionless bounded integers (Lemire, 2019, "Fast random integer
  generation in an interval"), here on 32-bit draws.

Where the kernel's conventions go beyond those algorithms, this module states them as
"the contract-1 convention" (``RANDOMNESS_CONTRACT = 1``):

* ``next_bit`` is the top bit of a draw;
* ``next_double`` is the top 53 bits of a draw times 2**-53;
* ``next_below`` takes ``x = next_u64() >> 32`` and applies Lemire's method with
  32-bit words (the rejection threshold is computed only when ``low32(m) < bound``);
* ``stream_state`` mixes ``(key, index)`` into one 64-bit word and expands it with four
  splitmix64 outputs. That mixing ignores part of the 128-bit key (kernel finding S1:
  only ``key0 ^ key1`` matters). Task 15 changes it under contract 2, and this module
  then gets a second model next to ``RANDOMNESS_CONTRACT_MODEL``'s current entry.
"""

from __future__ import annotations

MASK64 = (1 << 64) - 1

# Which stream mixing this module models (the ``RANDOMNESS_CONTRACT`` it belongs to).
RANDOMNESS_CONTRACT_MODEL = 1

_GOLDEN = 0x9E3779B97F4A7C15
_STREAM_MULTIPLIER = 0xD1B54A32D192ED03


def splitmix64(x: int) -> tuple[int, int]:
    """One splitmix64 step: return ``(new state, output)``."""
    x = (x + _GOLDEN) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return x, z ^ (z >> 31)


def stream_state(key: tuple[int, int], index: int) -> list[int]:
    """The four xoshiro words of stream ``index`` under ``key`` (the contract-1 convention)."""
    x = (key[0] ^ ((index * _STREAM_MULTIPLIER) & MASK64) ^ key[1]) & MASK64
    words = []
    for _ in range(4):
        x, z = splitmix64(x)
        words.append(z)
    if not any(words):
        words[0] = 1
    return words


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & MASK64


class Xoshiro:
    """xoshiro256** with the contract-1 draw conventions."""

    def __init__(self, state: list[int]):
        if len(state) != 4:
            raise ValueError("state must have 4 words")
        self.s = [int(w) & MASK64 for w in state]

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
                x = self.next_u64() >> 32
                m = x * bound
                low = m & 0xFFFFFFFF
        return m >> 32
