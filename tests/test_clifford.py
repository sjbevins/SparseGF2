"""Clifford tableaux and the canonical 1- and 2-qubit Clifford groups, checked against Stim."""

import numpy as np
import pytest
import stim

from sparsegf2 import Clifford, PauliString, num_cliffords, symplectic_group
from sparsegf2.clifford import _is_symplectic


def random_pauli(n, rng, any_phase=True):
    phase = int(rng.integers(0, 4)) if any_phase else 0
    return PauliString.from_bits(rng.integers(0, 2, n), rng.integers(0, 2, n), phase)


# -- the groups ------------------------------------------------------------------


@pytest.mark.parametrize("k, size", [(1, 6), (2, 720)])
def test_symplectic_group_size_and_validity(k, size):
    group = symplectic_group(k)
    assert group.shape == (size, 2 * k, 2 * k)
    assert all(_is_symplectic(m) for m in group)
    assert len({m.tobytes() for m in group}) == size


def test_symplectic_group_is_sorted_canonically():
    # Canonical order: increasing integer value of the row-major bits.
    group = symplectic_group(2)
    keys = [int("".join(map(str, m.flatten())), 2) for m in group]
    assert keys == sorted(keys)


@pytest.mark.parametrize("k", [1, 2])
def test_symplectic_group_matches_stim(k):
    ours = {m.tobytes() for m in symplectic_group(k)}
    theirs = {
        Clifford.from_stim(t).symplectic_matrix().tobytes()
        for t in stim.Tableau.iter_all(k, unsigned=True)
    }
    assert ours == theirs


def test_num_cliffords():
    assert num_cliffords(1) == 24
    assert num_cliffords(2) == 11520


@pytest.mark.parametrize("k", [1, 2])
def test_canonical_index_enumerates_the_whole_group(k):
    ours = [Clifford.from_index(k, i) for i in range(num_cliffords(k))]
    assert len(set(ours)) == num_cliffords(k)
    theirs = {Clifford.from_stim(t) for t in stim.Tableau.iter_all(k)}
    assert set(ours) == theirs
    # index() inverts from_index()
    assert [c.index() for c in ours] == list(range(num_cliffords(k)))


def test_from_index_rejects_out_of_range():
    with pytest.raises(ValueError):
        Clifford.from_index(2, 11520)
    with pytest.raises(ValueError):
        Clifford.from_index(1, -1)


def test_index_layout_documented():
    # symplectic part is index // 4**k; sign bit j negates image j (X0, X1, Z0, Z1)
    c = Clifford.from_index(2, 5 * 16 + 0b0101)
    np.testing.assert_array_equal(c.symplectic_matrix(), symplectic_group(2)[5])
    assert list(c.sign_bits()) == [1, 0, 1, 0]


# -- action on Paulis -------------------------------------------------------------


def test_conjugation_matches_stim_for_random_2q_cliffords(rng):
    for _ in range(300):
        c = Clifford.from_index(2, int(rng.integers(num_cliffords(2))))
        p = random_pauli(2, rng)
        assert str(c(p)) == str(c.to_stim()(p.to_stim()))


@pytest.mark.parametrize("k", [3, 5])
def test_conjugation_matches_stim_for_larger_cliffords(k, rng):
    for _ in range(50):
        t = stim.Tableau.random(k)
        c = Clifford.from_stim(t)
        p = random_pauli(k, rng)
        assert str(c(p)) == str(t(p.to_stim()))


def test_composition_matches_stim(rng):
    for _ in range(200):
        k = int(rng.integers(1, 4))
        t1, t2 = stim.Tableau.random(k), stim.Tableau.random(k)
        assert Clifford.from_stim(t1).then(Clifford.from_stim(t2)) == Clifford.from_stim(
            t1.then(t2)
        )


def test_inverse_matches_stim(rng):
    for _ in range(200):
        t = stim.Tableau.random(int(rng.integers(1, 5)))
        c = Clifford.from_stim(t)
        assert c.inverse() == Clifford.from_stim(t.inverse())
        assert c.then(c.inverse()) == Clifford.identity(len(t))


def test_stim_round_trip(rng):
    for _ in range(100):
        t = stim.Tableau.random(int(rng.integers(1, 6)))
        assert Clifford.from_stim(t).to_stim() == t


def test_conjugation_table_agrees_with_call():
    for index in range(0, num_cliffords(2), 37):
        c = Clifford.from_index(2, index)
        out_codes, sign_flips = c.conjugation_table()
        for code in range(16):
            xs = [code & 1, (code >> 2) & 1]
            zs = [(code >> 1) & 1, (code >> 3) & 1]
            img = c(PauliString.from_bits(xs, zs))
            out = int(out_codes[code])
            assert [out & 1, (out >> 2) & 1] == list(img.xs)
            assert [(out >> 1) & 1, (out >> 3) & 1] == list(img.zs)
            assert int(sign_flips[code]) == img.phase // 2


# -- validation -----------------------------------------------------------------


def test_rejects_non_hermitian_image():
    with pytest.raises(ValueError, match="Hermitian"):
        Clifford.from_strings(xs=["+iZ"], zs=["+X"])


def test_rejects_bad_commutation():
    with pytest.raises(ValueError, match="commutation"):
        Clifford.from_strings(xs=["+Z"], zs=["+Z"])
    with pytest.raises(ValueError, match="commutation"):
        Clifford.from_strings(xs=["+X_", "+X_"], zs=["+Z_", "+_Z"])


def test_from_symplectic_rejects_non_symplectic():
    with pytest.raises(ValueError, match="symplectic"):
        Clifford.from_symplectic(np.eye(4, dtype=np.uint8)[[0, 0, 2, 3]])
