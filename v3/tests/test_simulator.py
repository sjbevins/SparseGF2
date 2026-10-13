"""The Simulator against Stim: kernel tables, unitary evolution, and lock-step fuzzing."""

import numpy as np
import pytest
import stim

from _lockstep import Lockstep
from sparsegf2_v3 import Clifford, PauliString, all_gates, gate
from sparsegf2_v3._kernel.tables import ANF1, ANF2, LIN1, LIN2
from sparsegf2_v3.errors import PhaseFreeError
from sparsegf2_v3.simulator import Simulator

# Sizes chosen to cross the 64-bit word boundaries of the packed layout.
SIZES = [1, 2, 3, 5, 8, 31, 63, 64, 65, 127, 128, 129]


# -- the kernel's gate tables -------------------------------------------------------------


def _table_image(k: int, index: int, v: int) -> tuple[list[int], int]:
    """Apply the kernel tables to the unsigned input with basis-order bits ``v``."""
    lin, anf_table = (LIN1, ANF1) if k == 1 else (LIN2, ANF2)
    d = 2 * k
    sym, s = divmod(index, 4**k)
    m = (lin[sym] != 0).astype(int)
    bits = [(v >> j) & 1 for j in range(d)]
    out = [sum(bits[i] * m[i, j] for i in range(d)) % 2 for j in range(d)]
    anf = int(anf_table[sym])
    for j in range(d):
        if (s >> j) & 1:
            anf ^= 1 << (1 << j)
    sign = 0
    for mono in range(1 << d):
        if (anf >> mono) & 1 and all(bits[j] for j in range(d) if (mono >> j) & 1):
            sign ^= 1
    return out, sign


@pytest.mark.parametrize("k", [1, 2])
def test_kernel_tables_reproduce_every_clifford(k):
    d = 2 * k
    for index in range(24 if k == 1 else 11520):
        c = Clifford.from_index(k, index)
        for v in range(1 << d):
            xs = [(v >> j) & 1 for j in range(k)]
            zs = [(v >> (k + j)) & 1 for j in range(k)]
            img = c(PauliString.from_bits(xs, zs))
            out, sign = _table_image(k, index, v)
            assert out == list(img.xs) + list(img.zs)
            assert sign == img.phase // 2


# -- unitary evolution: exact generator-by-generator agreement ----------------------------


def _forward_tableau_matches(sim: Simulator, ref: stim.TableauSimulator) -> None:
    fwd = ref.current_inverse_tableau().inverse()
    n = sim.num_qubits
    stabs = sim.stabilizers(canonical=False)
    destabs = sim.destabilizers()
    for i in range(n):
        assert str(stabs[i]) == str(fwd.z_output(i)), f"stabilizer {i}"
        assert str(destabs[i])[1:] == str(fwd.x_output(i))[1:], f"destabilizer {i}"


@pytest.mark.parametrize("n", SIZES)
def test_random_unitary_circuits_match_stim_exactly(n, rng):
    sim = Simulator(n)
    ref = stim.TableauSimulator()
    ref.set_num_qubits(n)
    gates_1q = [g.name for g in all_gates() if g.num_qubits == 1]
    gates_2q = [g.name for g in all_gates() if g.num_qubits == 2]
    for step in range(300):
        if n == 1 or rng.random() < 0.4:
            name, targets = gates_1q[rng.integers(len(gates_1q))], [int(rng.integers(n))]
        else:
            name = gates_2q[rng.integers(len(gates_2q))]
            targets = [int(t) for t in rng.choice(n, 2, replace=False)]
        sim.do(name, *targets)
        ref.do(stim.CircuitInstruction(name, targets))
        if step % 50 == 49:
            _forward_tableau_matches(sim, ref)
    _forward_tableau_matches(sim, ref)


@pytest.mark.parametrize("name", [g.name for g in all_gates()])
def test_every_gate_on_random_states(name, rng):
    n = 6
    info = gate(name)
    for _ in range(5):
        t = stim.Tableau.random(n)
        sim = Simulator.from_tableau(t)
        ref = stim.TableauSimulator()
        ref.set_inverse_tableau(t.inverse())
        targets = [int(q) for q in rng.choice(n, info.num_qubits, replace=False)]
        sim.do(name, *targets)
        ref.do(stim.CircuitInstruction(name, targets))
        _forward_tableau_matches(sim, ref)


def test_all_two_qubit_cliffords_on_a_random_state():
    t = stim.Tableau.random(4)
    for index in range(0, 11520, 7):
        sim = Simulator.from_tableau(t)
        ref = stim.TableauSimulator()
        ref.set_inverse_tableau(t.inverse())
        sim.clifford_2q(index, 3, 1)
        ref.do_tableau(Clifford.from_index(2, index).to_stim(), [3, 1])
        _forward_tableau_matches(sim, ref)


def test_from_tableau_accepts_clifford_and_stim():
    t = stim.Tableau.random(5)
    a = Simulator.from_tableau(t)
    b = Simulator.from_tableau(Clifford.from_stim(t))
    assert [str(p) for p in a.stabilizers()] == [str(p) for p in b.stabilizers()]


# -- lock-step fuzzing with measurements ------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
def test_lockstep_from_zero_state(n, rng):
    ops = 400 if n <= 65 else 200
    ls = Lockstep(n, rng)
    ls.run(ops, check_every=1 if n <= 65 else 10)
    assert ls.num_random > 0


@pytest.mark.parametrize("n", [2, 5, 13, 64, 65, 100])
def test_lockstep_from_random_states(n, rng):
    ls = Lockstep(n, rng, initial=stim.Tableau.random(n))
    ls.run(150, check_every=5)
    assert ls.num_random + ls.num_deterministic > 0


def test_lockstep_many_small_seeds(rng):
    # Many short runs on tiny systems hit unusual corner cases quickly.
    for _ in range(200):
        n = int(rng.integers(1, 6))
        Lockstep(n, rng).run(30)


def test_lockstep_sees_both_outcome_types(rng):
    ls = Lockstep(8, rng)
    ls.run(2000, check_every=50)
    assert ls.num_random > 50
    assert ls.num_deterministic > 50


# -- API behaviour -------------------------------------------------------------------------


def test_ghz_example():
    sim = Simulator(3, seed=1)
    sim.h(0)
    sim.cx(0, 1, 1, 2)
    assert [str(p) for p in sim.stabilizers()] == ["+XXX", "+Z_Z", "+_ZZ"]
    assert sim.peek_z(0) == 0
    m = sim.measure(0)
    assert sim.peek_z(1) == sim.peek_z(2) == (1 - 2 * m)
    assert sim.measure(1) == sim.measure(2) == m


def test_bell_pair_outcomes_are_correlated():
    # The v2 simulator failed this: it returned independent coins.
    for seed in range(50):
        sim = Simulator(2, seed=seed)
        sim.h(0)
        sim.cx(0, 1)
        assert sim.measure(0) == sim.measure(1)


def test_phase_free_mode_never_invents_outcomes():
    sim = Simulator(2, phases=False, seed=0)
    sim.h(0)
    sim.cx(0, 1)
    assert sim.measure(0) is None
    assert sim.measure_x(1) is None
    assert sim.measure_pauli("Z0*Z1") is None
    assert sim.measure_many(0, 1) is None
    for call in (lambda: sim.peek_z(0), lambda: sim.peek_pauli("X0"), sim.destabilizers):
        with pytest.raises(PhaseFreeError, match="phases=True"):
            call()


def test_coin_forces_random_outcomes_only():
    sim = Simulator(1, seed=0)
    sim.h(0)
    assert sim.measure(0, coin=1) == 1  # random: forced
    assert sim.measure(0, coin=0) == 1  # now deterministic: coin ignored
    with pytest.raises(ValueError):
        sim.measure(0, coin=2)


def test_seed_reproducibility_and_stream_independence_from_coins():
    def outcomes(force_first):
        sim = Simulator(10, seed=1234)
        for q in range(10):
            sim.h(q)
        first = sim.measure(0, coin=1 if force_first else None)
        return first, [sim.measure(q) for q in range(1, 10)]

    a, rest_a = outcomes(False)
    b, rest_b = outcomes(False)
    _, rest_c = outcomes(True)
    assert (a, rest_a) == (b, rest_b)
    assert rest_a == rest_c  # forcing one outcome does not shift later randomness


def test_copy_is_independent():
    sim = Simulator(4, seed=5)
    sim.h(0)
    other = sim.copy()
    other.cx(0, 1)
    assert [str(p) for p in sim.stabilizers()] != [str(p) for p in other.stabilizers()]
    assert sim.measure(0) == sim.copy().measure(0)


def test_measure_pauli_parsing_and_inversion():
    sim = Simulator(3, seed=0)
    assert sim.measure_pauli("Z0*Z1") == 0
    assert sim.measure_pauli("!Z0*Z1") == 1
    assert sim.peek_pauli("Z2*Z2") == 1  # repeated qubit multiplies to identity
    assert sim.measure_pauli(PauliString("-ZZ_")) == 1
    with pytest.raises(ValueError, match="Hermitian"):
        sim.measure_pauli("X0*Z0")  # = -iY: not Hermitian
    with pytest.raises(ValueError, match="bad Pauli term"):
        sim.measure_pauli("Q0")
    with pytest.raises(IndexError):
        sim.measure_pauli("Z7")


def test_argument_validation():
    with pytest.raises(ValueError):
        Simulator(0)
    sim = Simulator(3)
    with pytest.raises(IndexError):
        sim.h(3)
    with pytest.raises(ValueError, match="differ"):
        sim.cx(1, 1)
    with pytest.raises(ValueError, match="even number"):
        sim.cx(0, 1, 2)
    with pytest.raises(KeyError):
        sim.do("NOT_A_GATE", 0)
    with pytest.raises(ValueError):
        sim.clifford_2q(11520, 0, 1)


def test_gate_methods_exist_for_every_name_and_alias():
    sim = Simulator(2)
    for info in all_gates():
        for name in (info.name, *info.aliases):
            assert callable(getattr(sim, name.lower()))
    sim.cnot(0, 1)
    sim.sqrt_z(0)
    sim.h_xz(1)


def test_apply_clifford_object():
    sim = Simulator(2)
    ref = Simulator(2)
    c = Clifford.from_index(2, 777)
    sim.apply(c, 1, 0)
    ref.clifford_2q(777, 1, 0)
    assert [str(p) for p in sim.stabilizers()] == [str(p) for p in ref.stabilizers()]


# -- entropy: checked against an explicit state vector (no Stim, no rank formula) --------


def _statevector_entropy(stabilizers: list[PauliString], subsystem: list[int], n: int) -> float:
    projector = np.eye(2**n, dtype=complex)
    for p in stabilizers:
        projector = projector @ (np.eye(2**n) + p.to_matrix()) / 2
    eigvals, eigvecs = np.linalg.eigh(projector)
    psi = eigvecs[:, np.argmax(eigvals)]
    rest = [q for q in range(n) if q not in subsystem]
    psi = psi.reshape([2] * n).transpose(subsystem + rest).reshape(2 ** len(subsystem), -1)
    s = np.linalg.svd(psi, compute_uv=False) ** 2
    s = s[s > 1e-12]
    return float(-(s * np.log2(s)).sum())


@pytest.mark.parametrize("phases", [True, False])
def test_entropy_matches_state_vector(phases, rng):
    n = 6
    for _ in range(25):
        t = stim.Tableau.random(n)
        signed = Simulator.from_tableau(t)  # supplies signed stabilizers for the state vector
        sim = Simulator.from_tableau(t, phases=phases)
        q = int(rng.integers(n))
        signed.measure(q, coin=0)
        sim.measure(q, coin=0)
        stabilizers = signed.stabilizers()
        for _ in range(4):
            k = int(rng.integers(1, n))
            a = sorted(int(v) for v in rng.choice(n, k, replace=False))
            assert sim.entropy(a) == pytest.approx(
                _statevector_entropy(stabilizers, a, n), abs=1e-9
            )


def test_entropy_edge_cases():
    sim = Simulator(4)
    assert sim.entropy([]) == 0
    assert sim.entropy(range(4)) == 0
    sim.h(0)
    sim.cx(0, 1)
    assert sim.entropy([0]) == 1
    assert sim.entropy([0, 2]) == 1
    assert sim.entropy([0, 1]) == 0
    assert sim.entropy([1, 2, 3]) == 1  # uses the complement
