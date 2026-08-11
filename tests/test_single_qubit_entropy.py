"""Fast singleton-entropy observable used by single-reference production."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2 import (
    InvalidArgumentError,
    SparseGF2,
    entanglement_entropy,
    single_qubit_entropy,
)
from sparsegf2.circuits.picture import setup_picture
from sparsegf2.core.symplectic import random_symplectic_2q


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("use_numba", [False, True])
def test_single_qubit_entropy_matches_general_observable(hybrid: bool, use_numba: bool) -> None:
    rng = np.random.default_rng(20260811)
    sim, _ = setup_picture(
        "single_ref",
        8,
        rng=np.random.default_rng(7),
        use_numba=use_numba,
        hybrid=hybrid,
    )
    for _ in range(24):
        q0, q1 = rng.choice(8, size=2, replace=False)
        sim.apply_gate_2q(int(q0), int(q1), random_symplectic_2q(rng=rng))
        measured = np.flatnonzero(rng.random(8) < 0.2)
        for qubit in measured:
            sim.measure_z(int(qubit), rng=rng)
        for qubit in range(sim.n):
            assert single_qubit_entropy(sim, qubit) == entanglement_entropy(sim, [qubit])


def test_single_qubit_entropy_product_and_bell_states() -> None:
    assert single_qubit_entropy(SparseGF2(4), 0) == 0
    sim, spec = setup_picture("single_ref", 4)
    assert single_qubit_entropy(sim, int(spec.reference_qubits[0])) == 1
    sim.measure_z(3, rng=np.random.default_rng(0))
    assert single_qubit_entropy(sim, int(spec.reference_qubits[0])) == 0


@pytest.mark.parametrize("qubit", [True, 1.5, -1, 4])
def test_single_qubit_entropy_rejects_invalid_qubits(qubit: object) -> None:
    with pytest.raises(InvalidArgumentError):
        single_qubit_entropy(SparseGF2(4), qubit)  # type: ignore[arg-type]


class _ProtocolOnlyBellPair:
    """Minimal backend that exercises the protocol-only extraction path."""

    n = 2

    def to_symplectic(self) -> np.ndarray:
        # Destabilizers XX, XI; stabilizers ZZ, XX span a Bell state.
        return np.asarray(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
            ],
            dtype=np.uint8,
        )


def test_single_qubit_entropy_protocol_only_backend() -> None:
    backend = _ProtocolOnlyBellPair()
    assert single_qubit_entropy(backend, 0) == entanglement_entropy(backend, [0]) == 1
