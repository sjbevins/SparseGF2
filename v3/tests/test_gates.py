"""The gate library must agree with Stim gate-for-gate, and with textbook unitaries."""

import numpy as np
import pytest
import stim

from sparsegf2_v3 import Clifford, PauliString, all_gates, gate

# Stim gates that are unitary but take a variable number of targets or are
# padding instructions; they are handled by the circuit layer, not this table.
_VARIABLE_ARITY = {"SPP", "SPP_DAG", "MPAD"}


def stim_unitary_gates():
    seen = {}
    for g in stim.gate_data().values():
        if g.is_unitary and g.name not in _VARIABLE_ARITY:
            seen[g.name] = g
    return [seen[k] for k in sorted(seen)]


@pytest.mark.parametrize("stim_gate", stim_unitary_gates(), ids=lambda g: g.name)
def test_every_stim_unitary_gate_is_defined_identically(stim_gate):
    ours = gate(stim_gate.name)
    assert ours.name == stim_gate.name
    assert ours.kind == "unitary"
    assert ours.num_qubits == len(stim_gate.tableau)
    assert ours.clifford == Clifford.from_stim(stim_gate.tableau)
    assert set(ours.aliases) == set(stim_gate.aliases) - {stim_gate.name}


def test_no_gates_beyond_stim():
    stim_names = {g.name for g in stim_unitary_gates()}
    assert {g.name for g in all_gates()} == stim_names


@pytest.mark.parametrize("stim_gate", stim_unitary_gates(), ids=lambda g: g.name)
def test_conjugation_table_matches_stim(stim_gate):
    ours = gate(stim_gate.name).clifford
    k = ours.num_qubits
    out_codes, sign_flips = ours.conjugation_table()
    for code in range(4**k):
        xs = [(code >> (2 * q)) & 1 for q in range(k)]
        zs = [(code >> (2 * q + 1)) & 1 for q in range(k)]
        expected = stim_gate.tableau(PauliString.from_bits(xs, zs).to_stim())
        exp_code = sum(
            (int(expected.to_numpy()[0][q]) | (int(expected.to_numpy()[1][q]) << 1)) << (2 * q)
            for q in range(k)
        )
        assert int(out_codes[code]) == exp_code
        assert int(sign_flips[code]) == (1 if expected.sign == -1 else 0)


# Independent check that does not rely on Stim: conjugate Pauli matrices by the
# textbook unitary (qubit 0 = most significant kron factor = first target).
_S2 = np.sqrt(2)
TEXTBOOK = {
    "H": np.array([[1, 1], [1, -1]]) / _S2,
    "S": np.diag([1, 1j]),
    "S_DAG": np.diag([1, -1j]),
    "SQRT_X": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2,
    "SQRT_X_DAG": np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2,
    "SQRT_Y": np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2,
    "SQRT_Y_DAG": np.array([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]]) / 2,
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
    "CX": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
    "CY": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]),
    "CZ": np.diag([1, 1, 1, -1]),
    "SWAP": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
    "ISWAP": np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]),
    "ISWAP_DAG": np.array([[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]]),
    "SQRT_ZZ": np.diag([1, 1j, 1j, 1]),
    "SQRT_ZZ_DAG": np.diag([1, -1j, -1j, 1]),
}


@pytest.mark.parametrize("name", sorted(TEXTBOOK))
def test_matches_textbook_unitary(name):
    u = TEXTBOOK[name].astype(complex)
    c = gate(name).clifford
    k = c.num_qubits
    for j in range(k):
        for pauli, image in (("X", c.x_image(j)), ("Z", c.z_image(j))):
            p = PauliString.single(k, j, pauli).to_matrix()
            np.testing.assert_allclose(u @ p @ u.conj().T, image.to_matrix(), atol=1e-12)


def test_lookup_is_case_insensitive_and_accepts_aliases():
    assert gate("cnot") is gate("CX")
    assert gate("zcx") is gate("CX")
    assert gate("sqrt_z").name == "S"
    assert gate("h_xz").name == "H"


def test_unknown_gate_raises_helpful_error():
    with pytest.raises(KeyError, match="all_gates"):
        gate("TOFFOLI")


def test_dagger_pairs_are_inverses():
    for name in ["S", "SQRT_X", "SQRT_Y", "ISWAP", "SQRT_XX", "SQRT_YY", "SQRT_ZZ"]:
        c = gate(name).clifford
        assert c.then(gate(name + "_DAG").clifford) == Clifford.identity(c.num_qubits)
