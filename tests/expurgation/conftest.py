"""Shared builders for the expurgation tests.

Provides the two worked examples of the companion implementation notes
as fixtures, plus a brute-force distance helper:

* **Example A**: the 4-qubit toy chain ``H(0), CX(0,1), CX(1,2),
  CX(2,3)`` with data inputs ``K = {2, 3}`` (checks ``XXXX`` and
  ``ZZII``, brute-force distance 1).
* **Example B**: a depth-2 brickwork encoding on the 4-cycle with
  ``K = {2, 3}``, given directly by its tableau rows; erasing
  ``{0, 1}`` yields ``r_M = 3`` with lightest witness ``YYII``, and
  measuring that witness buys a genuine distance increase (1 to 2).
"""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2 import SparseGF2
from sparsegf2.expurgation import StabilizerCode

_XZ = {"I": (0, 0), "Z": (0, 1), "X": (1, 0), "Y": (1, 1)}


def pauli_row(word: str) -> np.ndarray:
    """Turn a Pauli word like ``"IXXY"`` into an ``[x | z]`` uint8 row."""
    n = len(word)
    row = np.zeros(2 * n, dtype=np.uint8)
    for q, letter in enumerate(word):
        x, z = _XZ[letter]
        row[q] = x
        row[n + q] = z
    return row


def brute_force_distance(code: StabilizerCode) -> int | None:
    """Exact code distance by enumeration: the lightest Pauli with zero
    syndrome and nontrivial logical action. ``None`` when ``k == 0``.
    Only viable for tiny ``n`` (cost ``4**n``)."""
    n = code.n
    best: int | None = None
    for word in range(1, 4**n):
        qubits: list[int] = []
        letters: list[int] = []
        v = word
        for q in range(n):
            xz = v & 3
            v >>= 2
            if xz:
                qubits.append(q)
                letters.append(xz)
        if best is not None and len(qubits) >= best:
            continue
        syndrome, logical = code.commutation_bits(qubits, letters)
        if not syndrome.any() and logical.any():
            best = len(qubits)
    return best


@pytest.fixture
def distance():
    return brute_force_distance


@pytest.fixture
def example_a_code() -> StabilizerCode:
    sim = SparseGF2(4)
    sim.apply_h(0)
    sim.apply_cx(0, 1)
    sim.apply_cx(1, 2)
    sim.apply_cx(2, 3)
    return StabilizerCode.from_encoding(sim, [2, 3])


@pytest.fixture
def example_b_code() -> StabilizerCode:
    rows = [
        # Destabilizers 0..3
        "IXXY",  # X-bar_0
        "IXXI",  # X-bar_1
        "YYZY",  # logical X-bar_a
        "ZZZY",  # logical X-bar_b
        # Stabilizers 4..7
        "XXXX",  # check Z-bar_0
        "XXYZ",  # check Z-bar_1
        "IZZI",  # logical Z-bar_a
        "YIIY",  # logical Z-bar_b
    ]
    symp = np.stack([pauli_row(w) for w in rows])
    sim = SparseGF2.from_symplectic(symp)
    return StabilizerCode.from_encoding(sim, [2, 3])
