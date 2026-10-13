"""PauliString: parsing, products with exact phases, commutation.

Checked two independent ways: against stim.PauliString, and against dense
matrix products (which need no Stim at all).
"""

import numpy as np
import pytest
import stim

from sparsegf2_v3 import PauliString


def random_pauli(n: int, rng: np.random.Generator, any_phase: bool = True) -> PauliString:
    xs = rng.integers(0, 2, n)
    zs = rng.integers(0, 2, n)
    phase = int(rng.integers(0, 4)) if any_phase else 2 * int(rng.integers(0, 2))
    return PauliString.from_bits(xs, zs, phase)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("XYZ", "+XYZ"),
        ("+X_Z", "+X_Z"),
        ("IXI", "+_X_"),
        ("-ZZ", "-ZZ"),
        ("iX", "+iX"),
        ("+iY", "+iY"),
        ("-iXZ", "-iXZ"),
        ("", "+"),
        ("xyz", "+XYZ"),
    ],
)
def test_parse_and_print(text, expected):
    assert str(PauliString(text)) == expected


def test_print_matches_stim(rng):
    for _ in range(200):
        p = random_pauli(int(rng.integers(0, 12)), rng)
        assert str(p) == str(stim.PauliString(str(p)))


def test_invalid_character_raises():
    with pytest.raises(ValueError, match="IXYZ_"):
        PauliString("XQZ")


def test_accessors():
    p = PauliString("-X_YZ")
    assert len(p) == 4
    assert p.weight == 3
    assert p.phase == 2
    assert p.sign == -1
    assert p.is_hermitian
    assert [p[q] for q in range(4)] == ["X", "I", "Y", "Z"]
    assert list(p.xs) == [1, 0, 1, 0]
    assert list(p.zs) == [0, 0, 1, 1]


def test_immutable():
    p = PauliString("XZ")
    with pytest.raises(ValueError):
        p.xs[0] = 0


def test_product_matches_stim(rng):
    for _ in range(500):
        n = int(rng.integers(1, 15))
        a, b = random_pauli(n, rng), random_pauli(n, rng)
        assert str(a * b) == str(a.to_stim() * b.to_stim())


def test_product_matches_dense_matrices(rng):
    # Independent of Stim: compare with explicit matrix multiplication.
    for _ in range(200):
        n = int(rng.integers(1, 4))
        a, b = random_pauli(n, rng), random_pauli(n, rng)
        np.testing.assert_allclose((a * b).to_matrix(), a.to_matrix() @ b.to_matrix())


def test_commutes_matches_stim(rng):
    for _ in range(500):
        n = int(rng.integers(1, 15))
        a, b = random_pauli(n, rng), random_pauli(n, rng)
        assert a.commutes(b) == a.to_stim().commutes(b.to_stim())


def test_commutes_matches_dense_matrices(rng):
    for _ in range(200):
        n = int(rng.integers(1, 4))
        a, b = random_pauli(n, rng), random_pauli(n, rng)
        ma, mb = a.to_matrix(), b.to_matrix()
        assert a.commutes(b) == np.allclose(ma @ mb, mb @ ma)


def test_negation_and_phase():
    p = PauliString("XY")
    assert str(-p) == "-XY"
    assert str(p.times_phase(1)) == "+iXY"
    assert str(p.times_phase(3).unsigned()) == "+XY"


def test_equality_and_hash():
    assert PauliString("+XZ") == PauliString("XZ")
    assert PauliString("+XZ") != PauliString("-XZ")
    assert len({PauliString("XZ"), PauliString("+XZ"), PauliString("ZX")}) == 2


def test_stim_round_trip(rng):
    for _ in range(100):
        p = random_pauli(int(rng.integers(0, 10)), rng)
        assert PauliString.from_stim(p.to_stim()) == p


def test_single_and_identity():
    assert str(PauliString.single(4, 2, "Y")) == "+__Y_"
    assert str(PauliString.identity(3)) == "+___"
    with pytest.raises(ValueError):
        PauliString.single(2, 5, "X")


def test_predicates_return_python_bools():
    p, q = PauliString("XZ"), PauliString("ZX")
    assert type(p.commutes(q)) is bool
    assert type(p.is_hermitian) is bool
    assert type(p.weight) is int
