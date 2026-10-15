"""Erasure counts, lost information and the first-failure law against a brute-force GF(2) oracle.

The oracle (``_gf2_oracle``) uses no sparsegf2 code. The fixtures are codes with logical
qubits and a small, known distance whose n or s crosses a 64-bit word boundary, where a
packing bug would hide.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import stim

import _gf2_oracle as oracle
from sparsegf2 import analysis
from sparsegf2.codes import StabilizerCode

FIVE_QUBIT = ["XZZX_", "_XZZX", "X_XZZ", "ZX_XZ"]


def _bits(pauli: str) -> np.ndarray:
    n = len(pauli)
    row = np.zeros(2 * n, dtype=np.uint8)
    for j, c in enumerate(pauli):
        if c in "XY":
            row[j] = 1
        if c in "ZY":
            row[n + j] = 1
    return row


# -- the oracle itself ----------------------------------------------------------------------


def test_oracle_on_the_four_two_two_code():
    gens = [_bits("XXXX"), _bits("ZZZZ")]
    assert oracle.lost_information(gens, 4, [0]) == 0
    assert oracle.lost_information(gens, 4, [0, 1]) == 2
    assert oracle.lost_information(gens, 4, range(4)) == 4
    assert oracle.lost_information(gens, 4, []) == 0
    assert oracle.uncorrectable_counts(gens, 4, 2) == [0, 6]
    assert oracle.stabilizer_dim_on(gens, 4, [0, 1]) == 0
    assert oracle.stabilizer_dim_on(gens, 4, range(4)) == 2
    assert oracle.normalizer_dim_on(gens, 4, [0, 1]) == 2  # X0X1 and Z0Z1
    pairs = [(0b1111, 0), (0, 0b1111)]  # (x_bits, z_bits) as Python integers
    assert oracle.uncorrectable_counts(pairs, 4, 2) == [0, 6]


def test_oracle_on_the_five_qubit_code():
    gens = [_bits(g) for g in FIVE_QUBIT]
    assert oracle.uncorrectable_counts(gens, 5, 3) == [0, 0, 10]
    assert oracle.lost_information(gens, 5, [0, 1, 2]) == 2


def test_oracle_on_the_repetition_code():
    # bit-flip repetition code Z_i Z_{i+1} on 5 qubits, k = 1: Z_q is a logical operator
    # supported on any single qubit, so every nonempty erasure loses the phase information
    n = 5
    gens = [_bits("".join("Z" if j in (i, i + 1) else "_" for j in range(n))) for i in range(n - 1)]
    assert oracle.lost_information(gens, n, [0]) == 1
    assert oracle.lost_information(gens, n, [1, 3]) == 1
    assert oracle.lost_information(gens, n, range(n)) == 2
    assert oracle.uncorrectable_counts(gens, n, 3) == [5, 10, 10]


def test_oracle_identities_agree_with_enumeration_of_the_group():
    rng = np.random.default_rng(3)
    n = 6
    for _ in range(4):
        t = stim.Tableau.random(n)
        s = int(rng.integers(2, 6))
        gens = [np.concatenate(t.z_output(i).to_numpy()).astype(np.uint8) for i in range(s)]
        elements = oracle.all_stabilizer_elements(gens, n)
        assert len(elements) == 2**s
        norm = oracle.normalizer(gens, n)
        nspan = oracle.all_stabilizer_elements(
            [np.array([(v >> j) & 1 for j in range(2 * n)]) for v in norm], n
        )
        for m in range(n + 1):
            for a in itertools.combinations(range(n), m):
                mask = oracle._mask(a, n)
                outside = ~mask & ((1 << 2 * n) - 1)
                s_a = sum(1 for v in elements if v & outside == 0)
                n_a = sum(1 for v in nspan if v & outside == 0)
                assert s_a == 2 ** oracle.stabilizer_dim_on(gens, n, a)
                assert n_a == 2 ** oracle.normalizer_dim_on(gens, n, a)
        # a pure state: entropy of a half equals the number of generators crossing the cut
        pure = [np.concatenate(t.z_output(i).to_numpy()).astype(np.uint8) for i in range(n)]
        assert oracle.entropy(pure, n, range(n)) == 0
        assert oracle.entropy(pure, n, []) == 0
        assert oracle.entropy(pure, n, [0, 1, 2]) == oracle.entropy(pure, n, [3, 4, 5])


# -- fixtures spanning several words --------------------------------------------------------


def _block_tail(n_blocks: int, frozen: int, tail: int, tail_s: int, seed: int):
    """Generators of [[4,2,2]] blocks, frozen qubits and a random-Clifford code, permuted."""
    n = 4 * n_blocks + frozen + tail
    rows = []
    for b in range(n_blocks):
        for p in ("XXXX", "ZZZZ"):
            row = np.zeros(2 * n, dtype=np.uint8)
            for j in range(4):
                q = 4 * b + j
                row[q if p[0] == "X" else n + q] = 1
            rows.append(row)
    for f in range(frozen):
        row = np.zeros(2 * n, dtype=np.uint8)
        row[n + 4 * n_blocks + f] = 1  # Z on a frozen qubit
        rows.append(row)
    if tail:
        t = stim.Tableau.random(tail)
        off = 4 * n_blocks + frozen
        for i in range(tail_s):
            x, z = t.z_output(i).to_numpy()
            row = np.zeros(2 * n, dtype=np.uint8)
            row[off : off + tail] = x
            row[n + off : n + off + tail] = z
            rows.append(row)
    perm = np.random.default_rng(seed).permutation(n)
    gens = np.array(rows, dtype=np.uint8).reshape(len(rows), 2 * n)
    out = np.zeros_like(gens)
    out[:, perm] = gens[:, :n]  # qubit q moves to perm[q]
    out[:, n + perm] = gens[:, n:]
    return out


# (n, s, blocks, frozen): the tail is a random Clifford code on the remaining qubits
_FIXTURES = [
    (70, 18, 2, 3),  # k = 52
    (70, 63, 0, 0),  # k = 7, pure random: large distance
    (70, 64, 0, 0),  # s a multiple of 64
    (70, 65, 1, 2),
    (130, 64, 3, 5),  # s a multiple of 64
    (130, 128, 1, 0),  # s a multiple of 64
]


@pytest.fixture(scope="module", params=_FIXTURES, ids=lambda p: f"n{p[0]}_s{p[1]}")
def fixture(request):
    n, s, blocks, frozen = request.param
    tail = n - 4 * blocks - frozen
    tail_s = s - 2 * blocks - frozen
    if tail_s > tail:  # more tail generators than tail qubits: fewer blocks would be needed
        raise AssertionError("bad fixture parameters")
    seed = 1000 * n + s
    gens = _block_tail(blocks, frozen, tail, tail_s, seed)
    code = StabilizerCode(gens)
    assert code.num_qubits == n and code.num_stabilizers == s
    assert code.num_logical == n - s > 0
    return n, s, gens, code


def test_fixtures_are_not_trivial():
    counts = []
    for n, s, blocks, frozen in _FIXTURES:
        tail = n - 4 * blocks - frozen
        gens = _block_tail(blocks, frozen, tail, s - 2 * blocks - frozen, 1000 * n + s)
        assert StabilizerCode(gens).num_logical > 0
        counts.append(oracle.uncorrectable_counts(gens, n, 2))
    assert any(c[1] == 0 for c in counts)
    assert any(c[1] > 0 for c in counts)


def test_counts_equal_the_oracle_exactly(fixture):
    n, _, gens, code = fixture
    expected = oracle.uncorrectable_counts(gens, n, 2)
    assert code.uncorrectable_counts(2).tolist() == expected


def test_counts_to_three_on_the_word_aligned_fixture():
    n, s, blocks, frozen = 70, 64, 0, 0
    gens = _block_tail(blocks, frozen, n, s, 1000 * n + s)
    code = StabilizerCode(gens)
    assert code.num_logical == 6
    assert code.uncorrectable_counts(3).tolist() == oracle.uncorrectable_counts(gens, n, 3)


def test_lost_information_and_correctability_on_random_regions(fixture):
    n, _, gens, code = fixture
    rng = np.random.default_rng(n * 7919 + len(gens))
    sizes = [*range(1, 9), *range(n - 8, n + 1)]
    reference = oracle.Code(gens, n)
    for _ in range(200):
        m = int(rng.choice(sizes))
        region = sorted(int(q) for q in rng.choice(n, m, replace=False))
        expected = reference.lost(region)
        assert code.lost_information(region) == expected
        assert code.is_correctable(region) == (expected == 0)


# -- the first-failure law on block codes with closed forms ---------------------------------


def _exact_failure(blocks: int, frozen: int) -> np.ndarray:
    """P_C(m), m = 0..n: m erased qubits fail iff two of them lie in one [[4,2,2]] block."""
    n = 4 * blocks + frozen
    p = np.zeros(n + 1)
    for m in range(n + 1):
        ok = sum(
            math.comb(blocks, j) * 4**j * math.comb(frozen, m - j)
            for j in range(min(m, blocks) + 1)
            if m - j <= frozen
        )
        p[m] = 1 - ok / math.comb(n, m)
    return p


def _block_code(blocks: int, frozen: int, seed: int):
    n = 4 * blocks + frozen
    return _block_tail(blocks, frozen, 0, 0, seed), n


# (blocks, frozen): n = 63, 64, 65, 127, 128, 129; s = 2 * blocks + frozen
_LAW_CASES = [(15, 3), (16, 0), (16, 1), (31, 3), (32, 0), (32, 1)]


@pytest.mark.parametrize("blocks,frozen", _LAW_CASES, ids=lambda v: str(v))
def test_first_failure_law_is_exact_on_block_codes(blocks, frozen):
    gens, n = _block_code(blocks, frozen, seed=blocks * 100 + frozen)
    s = 2 * blocks + frozen
    code = StabilizerCode(gens)
    assert (code.num_qubits, code.num_stabilizers, code.num_logical) == (n, s, 2 * blocks)
    samples = 20_000
    d = code.first_failure_sizes(samples, seed=blocks * 1000 + frozen)
    assert d.min() >= 2 and d.max() <= blocks + 1  # pigeonhole: D <= blocks + 1
    p = _exact_failure(blocks, frozen)
    hat = analysis.failure_by_size(d, n)
    # sigma of a binomial proportion at each m, computed here: sqrt(p (1 - p) / samples);
    # the bound is 5 sigma, and where p is 0 or 1 the match must be exact
    sigma = np.sqrt(p * (1 - p) / samples)
    degenerate = (p == 0) | (p == 1)
    np.testing.assert_array_equal(hat[degenerate], p[degenerate])
    assert np.all(np.abs(hat - p)[~degenerate] <= 5 * sigma[~degenerate])


def test_the_law_cases_include_a_stabilizer_count_that_is_a_multiple_of_64():
    assert any((2 * b + f) % 64 == 0 for b, f in _LAW_CASES)
    assert sorted(4 * b + f for b, f in _LAW_CASES) == [63, 64, 65, 127, 128, 129]
