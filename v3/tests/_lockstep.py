"""Lock-step comparison of sparsegf2_v3.Simulator against stim.TableauSimulator.

A random operation is applied to three simulators at once: Stim (the
reference), ours with phases, and ours without phases. Random measurement
outcomes are taken from our phased simulator and forced onto Stim with its
``postselect_*`` methods, so the two must then agree *exactly*:

* every deterministic outcome must match Stim's;
* whether each measurement was random must match (Stim: ``peek_* == 0``);
* the canonical stabilizer list must equal ``canonical_stabilizers()``, with
  signs for the phased simulator and without for the phase-free one.

``check_every`` controls how often the (comparatively slow) full-state
comparison runs; outcomes and randomness are checked on every operation.
"""

from __future__ import annotations

import numpy as np
import stim

from sparsegf2_v3 import Clifford, all_gates
from sparsegf2_v3._kernel.rng import next_bit
from sparsegf2_v3.simulator import Simulator

GATES_1Q = [g.name for g in all_gates("unitary") if g.num_qubits == 1]
GATES_2Q = [g.name for g in all_gates("unitary") if g.num_qubits == 2]

_PEEK = {"Z": "peek_z", "X": "peek_x", "Y": "peek_y"}
_POST = {"Z": "postselect_z", "X": "postselect_x", "Y": "postselect_y"}
_MEASURE = {"Z": "measure", "X": "measure_x", "Y": "measure_y"}
_RESET = {"Z": "reset", "X": "reset_x", "Y": "reset_y"}


class Lockstep:
    def __init__(self, n: int, rng: np.random.Generator, initial: stim.Tableau | None = None):
        self.n = n
        self.rng = rng
        seed = int(rng.integers(2**31))
        if initial is None:
            self.ref = stim.TableauSimulator(seed=seed)
            self.ref.set_num_qubits(n)
            self.ph = Simulator(n, phases=True, seed=seed)
            self.pf = Simulator(n, phases=False, seed=seed + 1)
        else:
            self.ref = stim.TableauSimulator(seed=seed)
            self.ref.set_inverse_tableau(initial.inverse())
            self.ph = Simulator.from_tableau(initial, phases=True, seed=seed)
            self.pf = Simulator.from_tableau(initial, phases=False, seed=seed + 1)
        self.num_random = 0
        self.num_deterministic = 0

    # -- operations ---------------------------------------------------------------------

    def gate(self, name: str, targets: list[int]) -> None:
        self.ph.do(name, *targets)
        self.pf.do(name, *targets)
        self.ref.do(stim.CircuitInstruction(name, targets))

    def clifford_2q(self, index: int, a: int, b: int) -> None:
        self.ph.clifford_2q(index, a, b)
        self.pf.clifford_2q(index, a, b)
        self.ref.do_tableau(Clifford.from_index(2, index).to_stim(), [a, b])

    def _check_measurement(self, det_ph: bool, det_pf: bool, peek: int, outcome: int) -> None:
        assert det_ph == (peek != 0), "phased simulator disagrees with Stim on randomness"
        assert det_pf == (peek != 0), "phase-free simulator disagrees with Stim on randomness"
        if peek != 0:
            assert outcome == (1 if peek == -1 else 0), "deterministic outcome differs from Stim"
            self.num_deterministic += 1
        else:
            self.num_random += 1

    def measure(self, basis: str, q: int) -> None:
        det_ph = self.ph.is_deterministic(f"{basis}{q}")
        det_pf = self.pf.is_deterministic(f"{basis}{q}")
        peek = getattr(self.ref, _PEEK[basis])(q)
        assert getattr(self.ph, _PEEK[basis])(q) == peek, "peek differs from Stim"
        outcome = getattr(self.ph, _MEASURE[basis])(q)
        assert getattr(self.pf, _MEASURE[basis])(q) is None
        self._check_measurement(det_ph, det_pf, peek, outcome)
        if peek == 0:
            getattr(self.ref, _POST[basis])(q, desired_value=bool(outcome))

    def measure_pauli(self, terms: list[tuple[str, int]], invert: bool) -> None:
        text = ("!" if invert else "") + "*".join(f"{p}{q}" for p, q in terms)
        ps = stim.PauliString(self.n)
        for p, q in terms:
            ps[q] = p
        if invert:
            ps = -ps
        det_ph = self.ph.is_deterministic(text)
        det_pf = self.pf.is_deterministic(text)
        peek = self.ref.peek_observable_expectation(ps)
        assert self.ph.peek_pauli(text) == peek, f"peek_pauli({text}) differs from Stim"
        outcome = self.ph.measure_pauli(text)
        assert self.pf.measure_pauli(text) is None
        self._check_measurement(det_ph, det_pf, peek, outcome)
        if peek == 0:
            self.ref.postselect_observable(ps, desired_value=bool(outcome))

    def reset(self, basis: str, q: int) -> None:
        # Our reset collapses with the next bit of its stream; predict it and
        # force Stim's internal collapse to the same outcome before resetting.
        coin = int(next_bit(self.ph._rng.copy()))
        if getattr(self.ref, _PEEK[basis])(q) == 0:
            getattr(self.ref, _POST[basis])(q, desired_value=bool(coin))
        getattr(self.ref, _RESET[basis])(q)
        getattr(self.ph, _RESET[basis])(q)
        getattr(self.pf, _RESET[basis])(q)

    def random_op(self) -> str:
        n, rng = self.n, self.rng
        r = rng.random()
        if r < 0.25:
            name = GATES_1Q[rng.integers(len(GATES_1Q))]
            self.gate(name, [int(rng.integers(n))])
            return name
        if n >= 2 and r < 0.45:
            a, b = (int(v) for v in rng.choice(n, 2, replace=False))
            name = GATES_2Q[rng.integers(len(GATES_2Q))]
            self.gate(name, [a, b])
            return name
        if n >= 2 and r < 0.65:
            a, b = (int(v) for v in rng.choice(n, 2, replace=False))
            self.clifford_2q(int(rng.integers(11520)), a, b)
            return "CLIFFORD_2Q"
        if r < 0.82:
            basis = "ZXY"[rng.integers(3)]
            self.measure(basis, int(rng.integers(n)))
            return "M" + basis
        if r < 0.93:
            k = int(rng.integers(1, min(n, 5) + 1))
            qubits = rng.choice(n, k, replace=False)
            terms = [("XYZ"[rng.integers(3)], int(q)) for q in qubits]
            self.measure_pauli(terms, bool(rng.integers(2)))
            return "MPP"
        basis = "ZXY"[rng.integers(3)]
        self.reset(basis, int(rng.integers(n)))
        return "R" + basis

    # -- state comparison -------------------------------------------------------------

    def check_state(self) -> None:
        expected = [str(p) for p in self.ref.canonical_stabilizers()]
        got = [str(p) for p in self.ph.stabilizers()]
        assert got == expected, "phased stabilizers differ from Stim"
        got_pf = [str(p)[1:] for p in self.pf.stabilizers()]
        assert got_pf == [s[1:] for s in expected], "phase-free stabilizers differ from Stim"
        for p in self.pf.stabilizers():
            assert str(p).startswith("+"), "phase-free simulator produced a sign"

    def run(self, num_ops: int, check_every: int = 1) -> list[str]:
        history = []
        for i in range(num_ops):
            history.append(self.random_op())
            if (i + 1) % check_every == 0 or i == num_ops - 1:
                try:
                    self.check_state()
                except AssertionError as exc:
                    raise AssertionError(f"{exc} after ops {history[-5:]} (op #{i})") from exc
        return history
