"""crosscheck: SparseGF2 against Stim on one circuit."""

import pytest

import sparsegf2 as sg
from sparsegf2 import crosscheck

GHZ_TICKS = "H 0\nTICK\nCX 0 1\nTICK\nCX 1 2\nTICK\nM 0\nTICK\nH 1\nTICK"


@pytest.mark.parametrize("phases", [True, False])
def test_stabilizers_match_at_every_tick(phases):
    out = crosscheck.compare_stabilizers(sg.Circuit(GHZ_TICKS), phases=phases, seed=4)
    assert out == {"match": True, "ticks_compared": 5, "first_mismatch": None}


def test_selected_ticks():
    out = crosscheck.compare_stabilizers(sg.Circuit(GHZ_TICKS), ticks=[1, 3], seed=1)
    assert out["ticks_compared"] == 2 and out["match"]


def test_measurements_detectors_and_observables_match():
    text = "H 0\nCX 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\nMX 0"
    out = crosscheck.compare_measurements(sg.Circuit(text), seed=7)
    assert out["match"] and out["measurements"] == 3 and out["detectors"] == 1


def test_a_difference_is_caught(monkeypatch):
    real = crosscheck._sparsegf2_stabilizers

    def broken(circuit, num_qubits, phases, coins):
        rows = real(circuit, num_qubits, phases, coins)
        rows[-1] = ["+" + "X" * (len(rows[-1][0]) - 1)] + rows[-1][1:]
        return rows

    monkeypatch.setattr(crosscheck, "_sparsegf2_stabilizers", broken)
    out = crosscheck.compare_stabilizers(sg.Circuit(GHZ_TICKS), seed=4)
    assert not out["match"] and out["first_mismatch"]["tick"] == 4


def test_noise_is_refused():
    with pytest.raises(ValueError, match="noise-free"):
        crosscheck.compare_stabilizers(sg.Circuit("H 0\nX_ERROR(0.1) 0\nTICK"))
