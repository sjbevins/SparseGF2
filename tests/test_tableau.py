"""PhaseFreeTableau and the GF(2) row-space kernels, against independent references.

The references (``_rowspace_oracle``) are plain Python on integers: a row is an int whose bit
``j`` is column ``j``, in the canonical order ``x_0 .. x_{N-1}, z_0 .. z_{N-1}`` (so ``x_q`` is
bit ``q`` and ``z_q`` is bit ``N + q``). Random stabilizer states come from Stim, driven by
gates chosen with numpy (fixed seeds). Nothing of sparsegf2 computes an expected value; every
check is exact.
"""

import hashlib
import itertools
import struct

import numpy as np
import pytest
import stim

from _rowspace_oracle import (
    bits,
    dense_rows,
    entropy_by_enumeration,
    int_rows,
    log2_exact,
    pack,
    packed_position,
    physical_rows,
    rank,
    rref,
    rref_by_characterization,
    span,
    stim_joint_state,
    stim_state,
    support,
    symplectic,
    unpack,
)
from sparsegf2 import Circuit, PhaseFreeTableau, Simulator
from sparsegf2._kernel import rowspace
from sparsegf2.tableau import MAGIC

# -- the canonical form ------------------------------------------------------------------------


def random_rows(rng, n: int, count: int) -> np.ndarray:
    """Random rows with dependent, repeated and zero rows mixed in."""
    rows = rng.integers(0, 2, size=(count, 2 * n), dtype=np.uint8)
    if count >= 2:
        dependent = (rows[0] ^ rows[-1])[None, :]
        rows = np.concatenate([rows, dependent, rows[:1], np.zeros((1, 2 * n), np.uint8)])
    return rows[rng.permutation(rows.shape[0])]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 31, 63, 64, 65, 100, 130])
def test_canonical_form_is_the_reduced_row_echelon_form(n, rng):
    for count in sorted({0, 1, 3, n, 2 * n, 2 * n + 3}):
        rows = random_rows(rng, n, count)
        tab = PhaseFreeTableau(rows, num_qubits=n)
        expected = rref(int_rows(rows), range(2 * n))
        assert tab.num_qubits == n and tab.rank == len(expected)
        np.testing.assert_array_equal(tab.canonical(), dense_rows(expected, 2 * n))
        if n <= 4:  # the unique basis with the defining property, found by enumeration
            assert int_rows(tab.canonical()) == rref_by_characterization(int_rows(rows))
        np.testing.assert_array_equal(tab.words, pack(tab.canonical(), n))
        assert not tab.words.flags.writeable


def invertible(rng, size: int) -> np.ndarray:
    """A random invertible GF(2) matrix: unit lower times unit upper triangular, permuted."""
    lower = np.tril(rng.integers(0, 2, size=(size, size)), -1) + np.eye(size, dtype=np.int64)
    upper = np.triu(rng.integers(0, 2, size=(size, size)), 1) + np.eye(size, dtype=np.int64)
    return ((lower @ upper) % 2)[rng.permutation(size)]


@pytest.mark.parametrize("n", [1, 3, 17, 64, 70])
def test_every_basis_of_a_space_gives_the_same_bytes_and_digest(n, rng):
    base = random_rows(rng, n, n)
    rank = len(rref(int_rows(base), range(2 * n)))
    basis = dense_rows(rref(int_rows(base), range(2 * n)), 2 * n).astype(np.int64)
    other = (invertible(rng, rank) @ basis) % 2  # another basis of the same space
    extra = (rng.integers(0, 2, size=(3, rank)) @ basis) % 2  # dependent rows
    other = np.concatenate([other, extra, np.zeros((2, 2 * n), np.int64)])
    other = other[rng.permutation(other.shape[0])]
    a, b = PhaseFreeTableau(base, num_qubits=n), PhaseFreeTableau(other, num_qubits=n)
    c = PhaseFreeTableau.from_words(pack(other.astype(np.uint8), n), n)
    assert a.same_space(b) and a == b and b == c and hash(a) == hash(c)
    assert a.to_bytes() == b.to_bytes() == c.to_bytes()
    assert a.digest() == b.digest() == hashlib.sha256(a.to_bytes()).hexdigest()


def test_different_spaces_differ(rng):
    for n in (2, 9, 64, 66):
        base = rng.integers(0, 2, size=(n // 2 + 1, 2 * n), dtype=np.uint8)
        known = rref(int_rows(base), range(2 * n))
        outside = next(
            v
            for v in rng.integers(0, 2, size=(50, 2 * n), dtype=np.uint8)
            if len(rref([*known, *int_rows(v[None])], range(2 * n))) > len(known)
        )
        a = PhaseFreeTableau(base, num_qubits=n)
        bigger = PhaseFreeTableau(np.concatenate([base, outside[None]]), num_qubits=n)
        smaller = PhaseFreeTableau(dense_rows(known[1:], 2 * n), num_qubits=n)
        digests = {a.digest(), bigger.digest(), smaller.digest()}
        assert len(digests) == 3 and not a.same_space(bigger) and a != smaller
        # the same rows on one more qubit (idle) are another space
        wider = np.zeros((base.shape[0], 2 * n + 2), np.uint8)
        wider[:, :n], wider[:, n + 1 : 2 * n + 1] = base[:, :n], base[:, n:]
        assert PhaseFreeTableau(wider).digest() != a.digest()


def test_pauli_strings_dense_rows_and_words_agree(rng):
    rows = stim_state(7, rng)
    text = ["".join("_XZY"[x + 2 * z] for x, z in zip(r[:7], r[7:], strict=True)) for r in rows]
    signed = ["-" + t if i % 2 else "+" + t for i, t in enumerate(text)]  # signs are ignored
    assert PhaseFreeTableau(text) == PhaseFreeTableau(rows) == PhaseFreeTableau(signed)
    assert PhaseFreeTableau(rows) == PhaseFreeTableau.from_words(pack(rows, 7), 7)


def test_empty_and_zero_rows():
    tab = PhaseFreeTableau(np.zeros((3, 8), np.uint8))
    assert tab.rank == 0 and tab.num_qubits == 4 and tab.canonical().shape == (0, 8)
    assert tab.to_bytes() == MAGIC + struct.pack("<II", 4, 0)
    assert PhaseFreeTableau([], num_qubits=4) == tab
    assert tab.entropy([0, 2]) == 2 and tab.entropy(range(4)) == 4  # maximally mixed
    with pytest.raises(ValueError, match="num_qubits"):
        PhaseFreeTableau([])


# -- bytes and digest ------------------------------------------------------------------------


def reference_bytes(n: int, rows: list[int]) -> bytes:
    """The documented format, written out: magic, N and rank (uint32 LE), then each row's 2N
    bits in canonical column order, most significant bit first, zero padded to whole bytes."""
    out = bytearray(b"SGF2RS1\x00" + n.to_bytes(4, "little") + len(rows).to_bytes(4, "little"))
    width = (2 * n + 7) // 8
    for r in rows:
        value = 0
        for j in range(2 * n):
            value |= ((r >> j) & 1) << (8 * width - 1 - j)
        out += value.to_bytes(width, "big")
    return bytes(out)


def test_byte_format_and_digest(rng):
    bell = PhaseFreeTableau(["XX", "ZZ"])
    assert bell.to_bytes() == b"SGF2RS1\x00" + b"\x02\x00\x00\x00" * 2 + bytes([0xC0, 0x30])
    for n in (1, 3, 5, 64, 67):  # rows of 1, 1, 2, 16 and 17 bytes (ceil(2N / 8))
        rows = random_rows(rng, n, 2 * n)
        tab = PhaseFreeTableau(rows, num_qubits=n)
        data = reference_bytes(n, rref(int_rows(rows), range(2 * n)))
        assert tab.to_bytes() == data
        assert tab.digest() == hashlib.sha256(data).hexdigest()
        back = PhaseFreeTableau.from_bytes(data)
        assert back == tab and back.to_bytes() == data and back.digest() == tab.digest()


def test_from_bytes_refuses_anything_but_canonical_bytes(rng):
    n = 5
    rows = rref(int_rows(stim_state(n, rng)), range(2 * n))
    good = reference_bytes(n, rows)
    assert PhaseFreeTableau.from_bytes(good).rank == n
    bad = {
        "magic": b"SGF2RS2\x00" + good[8:],
        "short": good[:-1],
        "long": good + b"\x00",
        "header": good[:12],
        "rank": good[:12] + struct.pack("<I", 11) + good[16:],
        "padding": good[:-1] + bytes([good[-1] | 1]),  # 10 bits per row: 6 padding bits
        "order": reference_bytes(n, rows[::-1]),  # the rows, not in pivot order
        "reduced": reference_bytes(n, [rows[0] ^ rows[1], *rows[1:]]),  # a pivot not cleared
        "dependent": reference_bytes(n, [*rows, rows[0] ^ rows[1]]),
    }
    for name, data in bad.items():
        try:
            PhaseFreeTableau.from_bytes(data)
        except ValueError as exc:
            assert "canonical row space" in str(exc), name
        else:
            pytest.fail(f"{name}: accepted")


def test_from_words_checks_its_input():
    with pytest.raises(ValueError, match="shape"):
        PhaseFreeTableau.from_words(np.zeros((2, 3), np.uint64), 5)
    with pytest.raises(ValueError, match="beyond qubit 4"):
        PhaseFreeTableau.from_words(np.array([[1 << 5, 0]], dtype=np.uint64), 5)
    with pytest.raises(ValueError, match="non-negative"):
        PhaseFreeTableau.from_words(np.array([[-1, 0]]), 5)
    with pytest.raises(ValueError, match="at least 1"):
        PhaseFreeTableau.from_words(np.zeros((0, 2), np.uint64), 0)
    words = np.array([[0b11, 0], [0, 0b11]], dtype=np.uint64)
    assert PhaseFreeTableau.from_words(words, 2) == PhaseFreeTableau(["XX", "ZZ"])
    np.testing.assert_array_equal(words, [[0b11, 0], [0, 0b11]])  # the input is not touched


# -- the kernels -------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 7, 40, 70])
def test_rref_kernel_for_any_column_order(n, rng):
    for count in (1, n, 2 * n + 2):
        rows = random_rows(rng, n, count)
        order = rng.permutation(2 * n)
        cols = np.array([packed_position(int(c), n) for c in order], dtype=np.int64)
        words = pack(rows, n)
        pivots = np.full(words.shape[0], -1, dtype=np.int64)
        rank = rowspace.rref(words, cols, pivots)
        expected = rref(int_rows(rows), order)
        assert rank == len(expected)
        np.testing.assert_array_equal(unpack(words[:rank], n), dense_rows(expected, 2 * n))
        assert not words[rank:].any()  # every column is listed: the other rows are zero
        firsts = [next(i for i, c in enumerate(order) if (r >> int(c)) & 1) for r in expected]
        assert pivots[:rank].tolist() == firsts


@pytest.mark.parametrize("n", [3, 33, 80])
def test_rank_of_columns_kernel(n, rng):
    rows = random_rows(rng, n, n + 1)
    words = pack(rows, n)
    for size in (0, 1, n // 2, 2 * n):
        chosen = rng.choice(2 * n, size=size, replace=False)
        mask = sum(1 << int(c) for c in chosen)
        expected = len(rref([r & mask for r in int_rows(rows)], range(2 * n)))
        cols = np.array([packed_position(int(c), n) for c in chosen], dtype=np.int64)
        scratch = np.empty_like(words)
        assert rowspace.rank_of_columns(words, words.shape[0], cols, scratch) == expected


# -- isotropy, entropies and the physical code ---------------------------------------------------


def test_isotropy_against_the_symplectic_form(rng):
    for n in (1, 4, 9, 64, 65):
        rows = stim_state(n, rng)
        tab = PhaseFreeTableau(rows)
        ints = int_rows(tab.canonical())
        assert all(symplectic(a, b, n) == 0 for a, b in itertools.combinations(ints, 2))
        assert tab.is_isotropic() and tab.is_pure() and tab.rank == n
        grown = tab
        while grown.rank == n:  # a row outside the span (each try succeeds w.p. 1 - 2**-n)
            extra = rng.integers(0, 2, size=(1, 2 * n), dtype=np.uint8)
            grown = PhaseFreeTableau(np.concatenate([rows, extra]))
        ints = int_rows(grown.canonical())
        commute = all(symplectic(a, b, n) == 0 for a, b in itertools.combinations(ints, 2))
        # a pure state's group is maximal isotropic: a Pauli outside it anticommutes with some
        # element, so the grown space is not isotropic
        assert not commute and not grown.is_isotropic()
        with pytest.raises(ValueError, match="stabilizer group"):
            grown.entropy([0])
        with pytest.raises(ValueError, match="stabilizer group"):
            grown.physical_code(1)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8])
def test_entropy_of_pure_and_mixed_states_against_enumeration(n, rng):
    for drop in (0, 1, n // 2):  # pure (drop 0), then mixed: fewer generators
        rows = stim_state(n, rng)[: n - drop]
        tab = PhaseFreeTableau(rows)
        elements = span(int_rows(rows))
        assert len(elements) == 2**tab.rank and tab.is_pure() == (drop == 0)
        for size in range(n + 1):
            for region in itertools.combinations(range(n), size):
                assert tab.entropy(region) == entropy_by_enumeration(elements, n, region), region


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_mutual_information_of_every_disjoint_pair(n, rng):
    rows = stim_state(n, rng)
    tab = PhaseFreeTableau(rows)
    elements = span(int_rows(rows))

    def s(region):
        return entropy_by_enumeration(elements, n, region)

    for labels in itertools.product((0, 1, 2), repeat=n):  # each qubit in A, B or neither
        a = [q for q in range(n) if labels[q] == 1]
        b = [q for q in range(n) if labels[q] == 2]
        assert tab.mutual_information(a, b) == s(a) + s(b) - s(a + b)
    with pytest.raises(ValueError, match="disjoint"):
        tab.mutual_information([0], [0, 1])
    with pytest.raises(ValueError, match="distinct"):
        tab.entropy([1, 1])
    with pytest.raises(ValueError, match=r"in \[0, "):
        tab.entropy([n])


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_reference_information_of_joint_states(n, rng):
    """The physical code, I(A:R) and the identities of v1.4 §5.4 and §5.8, by enumeration."""
    system, reference = list(range(n)), list(range(n, 2 * n))
    big = 2 * n
    for measured in [[], [0], list(range(n)), *[list(rng.permutation(n)[: n // 2])]]:
        rows = stim_joint_state(n, rng, measured)
        tab = PhaseFreeTableau(rows)
        assert tab.is_pure() and tab.num_qubits == big
        elements = span(int_rows(rows))
        ref_mask = sum(1 << q for q in reference)
        low = (1 << n) - 1
        # S_phys: the elements with no reference part, restricted to the system qubits
        physical = {
            (g & low) | (((g >> big) & low) << n)
            for g in elements
            if support(g, big) & ref_mask == 0
        }
        r = log2_exact(len(physical))
        k = n - r
        assert k == entropy_by_enumeration(elements, big, reference)  # k = S(R)
        phys = tab.physical_tableau(n)
        assert phys.num_qubits == n and phys.rank == r
        np.testing.assert_array_equal(
            phys.canonical(), dense_rows(rref(sorted(physical), range(2 * n)), 2 * n)
        )
        code = tab.physical_code(n)
        assert code.num_stabilizers == r and code.num_logical == k

        def r_of(region, physical=physical):  # dim of the physical elements supported in A
            mask = sum(1 << q for q in region)
            return log2_exact(sum(1 for p in physical if support(p, n) & ~mask == 0))

        for size in range(n + 1):
            for a in itertools.combinations(system, size):
                b = [q for q in system if q not in a]
                s_a = entropy_by_enumeration(elements, big, a)
                s_ar = entropy_by_enumeration(elements, big, [*a, *reference])
                info = s_a + k - s_ar
                assert tab.mutual_information(a, reference) == info
                assert info == k + 2 * len(a) - n - r_of(a) + r_of(b)  # v1.4 §5.4
                assert info + tab.mutual_information(b, reference) == 2 * k  # v1.4 §5.8
                assert code.lost_information(a) == info


@pytest.mark.parametrize(
    ("n", "pairs"),
    [
        (33, None),  # N = 66: one word per system half, two per joint half
        (40, None),  # N = 80
        (70, None),  # N = 140: three words per joint half, two per system half
        (64, [17]),  # one reference: N = 65, two joint words against one system word
        (65, [0]),  # N = 66, two words for both
    ],
)
def test_multiword_joint_states_against_integer_ranks(n, pairs, rng):
    """The physical code, entropies and I(A:R) when a half row spans several 64-bit words
    (every purification state with n >= 33), against the integer references: S_phys by the
    reference-first reduction, ranks by another elimination. Without enumeration (2^N
    elements), the identities are those of v1.4 §5.4 and §5.8 with ranks: S(X) = |X| - N +
    rank(S|_Xbar) (Proposition 9 of docs/study/03), r(A) = r - rank(W_phys|_{V \\ A})."""
    measured = list(rng.permutation(n)[: n // 3])
    rows = stim_joint_state(n, rng, measured, pairs)
    big = rows.shape[0]
    tab = PhaseFreeTableau(rows)
    assert tab.num_qubits == big and tab.is_pure()
    np.testing.assert_array_equal(tab.words, pack(tab.canonical(), big))
    ints = int_rows(rows)
    system, reference = list(range(n)), list(range(n, big))

    def s(region):  # Proposition 9 on the integer rows (any region of the N qubits)
        rest = [q for q in range(big) if q not in set(region)]
        mask = sum((1 << q) | (1 << (big + q)) for q in rest)
        return len(region) - big + rank([g & mask for g in ints])

    physical = physical_rows(ints, n, big)
    r = len(physical)
    ref_mask = sum((1 << q) | (1 << (big + q)) for q in reference)
    assert r == big - rank([g & ref_mask for g in ints])  # dim U_V = N - rank(U|_R)
    k = n - r
    phys = tab.physical_tableau(n)
    assert phys.num_qubits == n and phys.rank == r
    np.testing.assert_array_equal(phys.canonical(), dense_rows(physical, 2 * n))
    code = tab.physical_code(n)
    assert code.num_stabilizers == r and code.num_logical == k
    assert tab.entropy(reference) == k == s(reference)  # k = S(R)
    assert 0 < k < len(reference) or len(reference) == 1, k  # an informative state

    def r_of(region):  # dim of the physical elements supported in A: r - rank(W|_{V \ A})
        outside = [q for q in system if q not in set(region)]
        mask = sum((1 << q) | (1 << (n + q)) for q in outside)
        return r - rank([w & mask for w in physical])

    for size in [1, 2, n // 3, n // 2, n - 1, big // 2 + 3, big - 1]:  # both sides of N / 2
        region = sorted(int(q) for q in rng.choice(big, size=size, replace=False))
        assert tab.entropy(region) == s(region), region
    for _ in range(20):
        a = sorted(int(q) for q in rng.choice(n, size=int(rng.integers(n + 1)), replace=False))
        b = [q for q in system if q not in set(a)]
        info = tab.mutual_information(a, reference)
        assert info == s(a) + s(reference) - s([*a, *reference])
        assert info == k + 2 * len(a) - n - r_of(a) + r_of(b)  # v1.4 §5.4
        assert info + tab.mutual_information(b, reference) == 2 * k  # v1.4 §5.8
        assert code.lost_information(a) == info


def test_physical_code_without_a_reference_is_every_row(rng):
    rows = stim_state(6, rng)
    tab = PhaseFreeTableau(rows)
    assert tab.physical_tableau(6) is tab
    code = tab.physical_code(6)
    assert code.num_qubits == 6 and code.num_logical == 0
    with pytest.raises(ValueError, match=r"in \[1, 6\]"):
        tab.physical_tableau(7)


def test_from_simulator_drops_signs_and_outcomes(rng):
    n = 6
    lines = [f"H {q}" for q in range(n)]
    for _ in range(30):
        a, b = (int(q) for q in rng.choice(n, size=2, replace=False))
        lines.append(f"CX {a} {b}")
        lines.append(f"{['H', 'S', 'S_DAG'][int(rng.integers(3))]} {a}")
        if rng.random() < 0.3:
            lines.append(f"M {b}")
    text = "\n".join(lines)
    states = []
    for phases, seed in ((True, 1), (True, 2), (False, 0)):
        sim = Simulator(n, phases=phases, seed=seed)
        sim.run(Circuit(text))
        states.append(PhaseFreeTableau.from_simulator(sim))
    stim_sim = stim.TableauSimulator(seed=5)
    stim_sim.set_num_qubits(n)
    stim_sim.do(stim.Circuit(text))
    expected = np.array([bits(p) for p in stim_sim.canonical_stabilizers()])
    assert states[0] == states[1] == states[2] == PhaseFreeTableau(expected)
