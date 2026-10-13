"""Circuit: parsing, printing, validation and conversion, checked against Stim."""

import numpy as np
import pytest
import stim

from sparsegf2_v3 import (
    Circuit,
    Instruction,
    RepeatBlock,
    target_combiner,
    target_inv,
    target_rec,
    target_x,
    target_z,
)

STIM_GENERATED = [
    ("repetition_code:memory", {"distance": 5, "rounds": 4}),
    ("surface_code:rotated_memory_x", {"distance": 3, "rounds": 3}),
    ("surface_code:rotated_memory_z", {"distance": 5, "rounds": 2}),
    ("surface_code:unrotated_memory_z", {"distance": 3, "rounds": 2}),
    ("color_code:memory_xyz", {"distance": 3, "rounds": 2}),
]
NOISE = {
    "after_clifford_depolarization": 0.001,
    "before_round_data_depolarization": 0.002,
    "before_measure_flip_probability": 0.003,
    "after_reset_flip_probability": 0.004,
}


@pytest.mark.parametrize("code, params", STIM_GENERATED, ids=[c for c, _ in STIM_GENERATED])
def test_stim_generated_circuits_round_trip(code, params):
    original = stim.Circuit.generated(code, **params, **NOISE)
    ours = Circuit.from_stim(original)
    assert stim.Circuit(str(ours)) == original
    assert ours.to_stim() == original
    assert ours.num_qubits == original.num_qubits
    assert ours.num_measurements == original.num_measurements
    assert ours.num_detectors == original.num_detectors
    assert ours.num_observables == original.num_observables
    assert ours.num_ticks == original.num_ticks


def test_text_normalisation_and_comments():
    c = Circuit(
        """
        # a comment
        h 0 1          # lower case is fine, like Stim
        CNOT 0 1
        MPP X0*Z1 Y2   # two products
        REPEAT 2 {
            M !0
            DETECTOR(1, 2.5) rec[-1]
        }
        """
    )
    assert str(c) == "\n".join(
        [
            "H 0 1",
            "CX 0 1",
            "MPP X0*Z1 Y2",
            "REPEAT 2 {",
            "    M !0",
            "    DETECTOR(1, 2.5) rec[-1]",
            "}",
        ]
    )
    assert Circuit(str(c)) == c
    assert c.num_measurements == 4
    assert c.num_detectors == 2


def test_append_api_and_targets():
    c = Circuit()
    c.append("H", [0])
    c.append("CX", [0, 1])
    c.append("M", [target_inv(0), 1])
    c.append("MPP", [target_x(0), target_combiner(), target_z(1)])
    c.append("CZ", [target_rec(-1), 2])
    c.append("X_ERROR", [2], [0.25])
    c.append("OBSERVABLE_INCLUDE", [target_rec(-2)], [0])
    assert (
        str(c)
        == "H 0\nCX 0 1\nM !0 1\nMPP X0*Z1\nCZ rec[-1] 2\nX_ERROR(0.25) 2\nOBSERVABLE_INCLUDE(0) rec[-2]"
    )
    assert stim.Circuit(str(c)).num_measurements == c.num_measurements == 3
    assert isinstance(c[0], Instruction)
    assert c.num_observables == 1


def test_repeat_and_multiplication():
    body = Circuit("H 0\nM 0")
    c = body * 3
    assert isinstance(c[0], RepeatBlock)
    assert c.num_measurements == 3
    assert len(list(c.flattened())) == 6
    assert str(body * 1) == str(body)
    assert len(body * 0) == 0
    joined = body + Circuit("X 1")
    assert joined.num_qubits == 2


@pytest.mark.parametrize(
    "text, message",
    [
        ("FOO 0", "unknown gate"),
        ("CX 0 1 2", "even number"),
        ("CX 0 0", "must differ"),
        ("H !0", "inverted"),
        ("H rec[-1]", "classically controlled"),
        ("CX 0 rec[-1]", "control first"),
        ("X_ERROR 0", "takes"),
        ("X_ERROR(1.5) 0", "probabilities"),
        ("PAULI_CHANNEL_1(0.5, 0.4, 0.3) 0", "sum to at most 1"),
        ("MPP X0**Z1", "misplaced"),
        ("MPP X0*", "ends with"),
        ("M X0", "Pauli"),
        ("DETECTOR 0", "rec"),
        ("REPEAT 2 {\nH 0", "unterminated"),
        ("}", "unmatched"),
        ("M sweep[0]", "sweep"),
        ("HERALDED_ERASE(0.1) 0", "unknown gate"),
        ("CLIFFORD_2Q(1) 0 1 2 3", "one index per application"),
        ("CLIFFORD_2Q(11520) 0 1", "indices"),
        ("rec[-1]", "unknown gate"),
        ("H 0 X1x", "cannot parse target"),
    ],
)
def test_invalid_circuits_raise_helpful_errors(text, message):
    with pytest.raises(ValueError, match=message):
        Circuit(text)


def test_error_messages_carry_line_numbers():
    with pytest.raises(ValueError, match="line 3"):
        Circuit("H 0\nCX 0 1\nCX 2")


def test_clifford_index_instructions_expand_for_stim(rng):
    c = Circuit()
    pairs = [0, 1, 2, 3, 1, 2]
    indices = [int(i) for i in rng.integers(0, 11520, 3)]
    c.append("CLIFFORD_2Q", pairs, indices)
    c.append("CLIFFORD_1Q", [0, 3], [5, 17])
    assert str(c).startswith("CLIFFORD_2Q(")
    expanded = c.to_stim()
    # The expansion must implement exactly the indexed Cliffords.
    from sparsegf2_v3 import Clifford

    ref = stim.TableauSimulator()
    ref.set_num_qubits(4)
    for j, index in enumerate(indices):
        ref.do_tableau(Clifford.from_index(2, index).to_stim(), pairs[2 * j : 2 * j + 2])
    for q, index in zip([0, 3], [5, 17], strict=True):
        ref.do_tableau(Clifford.from_index(1, index).to_stim(), [q])
    other = stim.TableauSimulator()
    other.set_num_qubits(4)
    other.do_circuit(expanded)
    assert other.current_inverse_tableau() == ref.current_inverse_tableau()


def test_tags_are_preserved():
    c = Circuit("H[my tag] 0")
    assert str(c) == "H[my tag] 0"
    assert stim.Circuit(str(c)) == stim.Circuit("H[my tag] 0")


def test_num_qubits_includes_coords_and_mpp():
    assert Circuit("QUBIT_COORDS(0, 0) 7").num_qubits == 8
    assert Circuit("MPP X0*Z9").num_qubits == 10
    assert Circuit().num_qubits == 0


def test_float_arguments_survive_round_trip():
    text = "DEPOLARIZE1(0.001) 0\nX_ERROR(1e-05) 1"
    c = Circuit(text)
    assert stim.Circuit(str(c)) == stim.Circuit(text)
    np.testing.assert_allclose(c[1].args, [1e-05])
