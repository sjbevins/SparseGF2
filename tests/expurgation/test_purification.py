"""The purification bridge: extracting the system code of a monitored run."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2 import SparseGF2, code_dimension, from_bell_purification
from sparsegf2.core.linalg_gf2 import gf2_rank
from sparsegf2.core.symplectic import random_symplectic_2q
from sparsegf2.errors import InvalidArgumentError
from sparsegf2.expurgation import (
    ExpurgationConfig,
    expurgate,
    expurgation_candidates,
    from_purification,
    sample_erasure,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _monitored_purification(m: int, seed: int, p: float = 0.2, layers: int | None = None):
    """A Bell purification of m system qubits pushed through a random
    monitored circuit (gates and Z measurements on the system only)."""
    rng = np.random.default_rng(seed)
    sim = from_bell_purification(m, rng=np.random.default_rng(seed + 1))
    for _ in range(layers if layers is not None else 3 * m):
        a, b = rng.choice(m, size=2, replace=False).tolist()
        sim.apply_gate_2q(int(a), int(b), random_symplectic_2q(rng))
        for q in range(m):
            if rng.random() < p:
                sim.measure_z(q)
    return sim


def _purification_side_distance(sim: SparseGF2, m: int) -> int | None:
    """Independent distance: the lightest system Pauli whose measurement
    on a fresh copy lowers code_dimension (targeted purification)."""
    k0 = code_dimension(sim, m)
    if k0 == 0:
        return None
    best: int | None = None
    for word in range(1, 4**m):
        qubits: list[int] = []
        letters: list[int] = []
        v = word
        for q in range(m):
            xz = v & 3
            v >>= 2
            if xz:
                qubits.append(q)
                letters.append(xz)
        if best is not None and len(qubits) >= best:
            continue
        copy = sim.copy()
        copy.measure_pauli(qubits, letters)
        if code_dimension(copy, m) == k0 - 1:
            best = len(qubits)
    return best


# ----------------------------------------------------------------------
# Extraction correctness
# ----------------------------------------------------------------------


@pytest.mark.parametrize("m, seed, p", [(4, 0, 0.15), (6, 1, 0.2), (8, 2, 0.3), (6, 3, 0.05)])
def test_k_matches_code_dimension(m, seed, p):
    sim = _monitored_purification(m, seed, p)
    code = from_purification(sim)
    assert code.n == m
    assert code.k == code_dimension(sim, m)
    assert len(code.check_pairs()) == m - code.k
    # Checks come first, logical pairs after.
    assert code.check_pairs().tolist() == list(range(m - code.k))


@pytest.mark.parametrize("m, seed", [(4, 10), (6, 11)])
def test_checks_span_system_supported_stabilizers(m, seed):
    sim = _monitored_purification(m, seed)
    code = from_purification(sim)
    n_tot = sim.n
    stab_block = sim.to_symplectic()[n_tot:]
    # Embed each extracted check back into the 2*n_tot-column layout and
    # confirm it lies in the row span of the original stabilizer block.
    for i in range(len(code.check_pairs())):
        row = code.sim.to_symplectic()[code.n + i]
        embedded = np.zeros(2 * n_tot, dtype=np.uint8)
        embedded[:m] = row[: code.n]
        embedded[n_tot : n_tot + m] = row[code.n :]
        assert gf2_rank(np.concatenate([stab_block, embedded[None, :]])) == gf2_rank(stab_block)


@pytest.mark.parametrize("m, seed, p", [(4, 20, 0.15), (4, 21, 0.3), (4, 22, 0.05)])
def test_distance_matches_purification_side(m, seed, p, distance):
    sim = _monitored_purification(m, seed, p)
    code = from_purification(sim)
    assert distance(code) == _purification_side_distance(sim, m)


def test_fresh_bell_purification_has_no_checks():
    sim = from_bell_purification(3)
    code = from_purification(sim)
    assert code.k == 3
    assert len(code.check_pairs()) == 0


def test_fully_measured_system_is_all_checks():
    sim = from_bell_purification(3)
    for q in range(3):
        sim.measure_z(q)
    code = from_purification(sim)
    assert code.k == 0
    assert len(code.check_pairs()) == 3


def test_custom_system_qubits_layout():
    # Bell pairs (0,1) and (2,3): system {0, 2}, reference {1, 3}.
    sim = SparseGF2(4)
    sim.apply_h(0)
    sim.apply_cx(0, 1)
    sim.apply_h(2)
    sim.apply_cx(2, 3)
    code = from_purification(sim, system_qubits=[0, 2])
    assert code.n == 2
    assert code.k == 2
    assert len(code.check_pairs()) == 0


# ----------------------------------------------------------------------
# Targeted purification: replaying a candidate on the original tableau
# ----------------------------------------------------------------------


def test_candidate_replay_lowers_code_dimension():
    m = 8
    sim = _monitored_purification(m, seed=41, p=0.1)
    code = from_purification(sim)
    assert code.k >= 2  # seed chosen so the run is not fully purified
    rng = np.random.default_rng(0)
    for _ in range(20):
        erased = sample_erasure(m, rng, count=3)
        cands = expurgation_candidates(code, erased)
        if cands:
            break
    assert cands, "no candidate found; pick different seeds"
    g = cands[0]
    k_before = code_dimension(sim, m)
    # System qubit i of the code is qubit i of the original (default split).
    pair = sim.measure_pauli(list(g.qubits), list(g.letters))
    assert pair is not None
    assert code_dimension(sim, m) == k_before - 1


# ----------------------------------------------------------------------
# End to end: expurgate an extracted monitored-circuit code
# ----------------------------------------------------------------------


def test_expurgate_extracted_code_distance_monotone(distance):
    sim = _monitored_purification(4, seed=40, p=0.08)
    code = from_purification(sim)
    assert code.k >= 1
    d_before = distance(code)
    result = expurgate(
        code,
        ExpurgationConfig(erasure_count=2, max_rounds=6, validation_patterns=12, seed=4),
    )
    assert result.recovery_after >= result.recovery_before
    assert code.k == result.k_final
    if code.k > 0:
        assert distance(code) >= d_before


def test_expurgate_extracted_code_accounting():
    sim = _monitored_purification(10, seed=50, p=0.15)
    code = from_purification(sim)
    k0 = code.k
    result = expurgate(
        code,
        ExpurgationConfig(erasure_count=2, k_target=max(1, k0 - 3), max_rounds=30, seed=5),
    )
    assert result.k_final == k0 - len(result.measured_pairs)
    assert len(code.gauge_pairs()) == len(result.measured_pairs)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_purification_validation():
    with pytest.raises(InvalidArgumentError):
        from_purification(SparseGF2(5))  # odd n, no system_qubits
    sim = SparseGF2(4)
    with pytest.raises(InvalidArgumentError):
        from_purification(sim, system_qubits=[])
    with pytest.raises(InvalidArgumentError):
        from_purification(sim, system_qubits=[0, 0])
    with pytest.raises(InvalidArgumentError):
        from_purification(sim, system_qubits=[4])
