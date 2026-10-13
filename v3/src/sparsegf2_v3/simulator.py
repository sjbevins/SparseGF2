"""The :class:`Simulator`: a stabilizer state you apply gates to and measure.

The method names follow ``stim.TableauSimulator`` wherever an equivalent
exists, so code written for Stim reads the same here::

    sim = Simulator(3, seed=1)
    sim.h(0)
    sim.cx(0, 1, 1, 2)          # two-qubit gates take target pairs, like Stim
    sim.measure(2)              # 0 or 1
    sim.peek_z(0)               # +1 / -1, or 0 if a measurement would be random

Every one- and two-qubit Clifford gate in :func:`sparsegf2_v3.all_gates` is
available as a lowercase method (``sim.sqrt_x_dag(3)``, ``sim.iswap(0, 1)``,
...), plus :meth:`Simulator.do` for applying a gate by name.

Modes
-----
``phases=True`` (default)
    Stabilizers, destabilizers and signs are tracked (the Aaronson-Gottesman
    tableau). Measurement outcomes are physical and agree exactly with Stim.
``phases=False``
    Only the unsigned stabilizer group is stored. That fixes every entanglement
    quantity exactly at roughly half the cost, but outcomes do not exist:
    measurements return ``None`` and :meth:`peek_z` and friends raise
    :class:`~sparsegf2_v3.errors.PhaseFreeError`.

Randomness
----------
Every collapse (each measurement, and each reset) consumes exactly one random
bit from the simulator's own stream, whether or not its outcome turns out to be
random. Pass ``coin=`` to force the outcome of a random measurement (it is
ignored when the outcome is deterministic). Because the stream advances the same
way regardless, forcing one outcome never shifts the randomness of later ones.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable

import numpy as np

from sparsegf2_v3._compile import compile_circuit
from sparsegf2_v3._kernel import tableau as _k
from sparsegf2_v3._kernel.program import run_program
from sparsegf2_v3._kernel.rng import next_bit, seed_state
from sparsegf2_v3._kernel.tables import (
    ANF1,
    ANF2,
    IDENTITY_SYM1,
    IDENTITY_SYM2,
    LIN1,
    LIN2,
)
from sparsegf2_v3.circuit import Circuit
from sparsegf2_v3.clifford import Clifford
from sparsegf2_v3.errors import PhaseFreeError
from sparsegf2_v3.gates import all_gates, gate
from sparsegf2_v3.pauli import PauliString, pauli_product_phase
from sparsegf2_v3.result import RunResult

__all__ = ["Simulator"]

_CODE_OF = {"X": 1, "Z": 2, "Y": 3}
_X, _Z, _Y = 1, 2, 3


def _pack_columns(rows: np.ndarray, words: int) -> np.ndarray:
    """Pack a (num_rows, n) 0/1 matrix into (n, words) uint64 column bitsets."""
    num_rows, n = rows.shape
    padded = np.zeros((n, words * 64), dtype=np.uint8)
    padded[:, :num_rows] = rows.T
    packed = np.packbits(padded, axis=1, bitorder="little")
    return packed.view(np.dtype("<u8")).astype(np.uint64)


class Simulator:
    """An n-qubit stabilizer state, initially |0...0>.

    Parameters
    ----------
    num_qubits
        Number of qubits (at least 1).
    phases
        Track signs and destabilizers (default ``True``). See the module
        docstring for what changes with ``phases=False``.
    seed
        Seed for the simulator's random stream: an int, a
        ``numpy.random.SeedSequence``, or ``None`` for fresh OS entropy. The
        seed actually used is available as :attr:`seed`.
    """

    def __init__(self, num_qubits: int, *, phases: bool = True, seed=None):
        n = operator.index(num_qubits)
        if n < 1:
            raise ValueError(f"num_qubits must be at least 1, got {n}")
        self._n = n
        self._phases = bool(phases)
        self._ws = (n + 63) // 64
        width = 2 * self._ws if self._phases else self._ws
        self._tab = np.zeros((2 * n + 1, width), dtype=np.uint64)
        _k.init_zero(self._tab, n, self._ws, self._phases)
        self._seed = (
            seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        )
        self._rng = seed_state(self._seed)
        self._alloc_scratch()

    def _alloc_scratch(self) -> None:
        width = self._tab.shape[1]
        self._anti = np.zeros(width, dtype=np.uint64)
        self._supp = np.zeros(max(self._n, self._ws), dtype=np.int64)
        self._lo = np.zeros(self._ws, dtype=np.uint64)
        self._hi = np.zeros(self._ws, dtype=np.uint64)
        self._qbuf = np.zeros(1, dtype=np.int64)
        self._cbuf = np.zeros(1, dtype=np.int64)

    # -- constructors -------------------------------------------------------------

    @classmethod
    def from_tableau(cls, tableau, *, phases: bool = True, seed=None) -> Simulator:
        """The state ``C |0...0>`` for an n-qubit Clifford ``C``.

        ``tableau`` is a :class:`~sparsegf2_v3.Clifford` or a ``stim.Tableau``.
        Stabilizer ``i`` becomes ``C Z_i C^dagger`` and destabilizer ``i``
        becomes ``C X_i C^dagger``, exactly as in ``stim.TableauSimulator``
        after ``set_inverse_tableau(tableau.inverse())``.
        """
        if isinstance(tableau, Clifford):
            n = tableau.num_qubits
            x2x = np.array([tableau.x_image(i).xs for i in range(n)], dtype=np.uint8)
            x2z = np.array([tableau.x_image(i).zs for i in range(n)], dtype=np.uint8)
            z2x = np.array([tableau.z_image(i).xs for i in range(n)], dtype=np.uint8)
            z2z = np.array([tableau.z_image(i).zs for i in range(n)], dtype=np.uint8)
            z_signs = np.array([tableau.z_image(i).phase // 2 for i in range(n)], dtype=np.uint8)
        elif hasattr(tableau, "to_numpy"):  # stim.Tableau
            x2x, x2z, z2x, z2z, _x_signs, z_signs = tableau.to_numpy()
            n = len(tableau)
        else:
            raise TypeError(f"expected a Clifford or stim.Tableau, got {type(tableau).__name__}")
        sim = cls(n, phases=phases, seed=seed)
        sim._load(
            np.asarray(z2x, np.uint8),
            np.asarray(z2z, np.uint8),
            np.asarray(z_signs, np.uint8),
            np.asarray(x2x, np.uint8),
            np.asarray(x2z, np.uint8),
        )
        return sim

    def _load(self, sx, sz, ssign, dx, dz) -> None:
        n, ws = self._n, self._ws
        tab = self._tab
        tab[:, :] = 0
        tab[0 : 2 * n : 2, :ws] = _pack_columns(sx, ws)
        tab[1 : 2 * n : 2, :ws] = _pack_columns(sz, ws)
        if self._phases:
            tab[0 : 2 * n : 2, ws:] = _pack_columns(dx, ws)
            tab[1 : 2 * n : 2, ws:] = _pack_columns(dz, ws)
            tab[2 * n, :ws] = _pack_columns(ssign[:, None], ws)[0]

    # -- basic properties -----------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        return self._n

    @property
    def phases(self) -> bool:
        """Whether signs and destabilizers are tracked."""
        return self._phases

    @property
    def seed(self) -> np.random.SeedSequence:
        """The ``SeedSequence`` the random stream was built from (its ``.entropy`` reproduces it)."""
        return self._seed

    def copy(self) -> Simulator:
        """An independent copy, including the current position of the random stream."""
        other = object.__new__(Simulator)
        other._n = self._n
        other._phases = self._phases
        other._ws = self._ws
        other._tab = self._tab.copy()
        other._seed = self._seed
        other._rng = self._rng.copy()
        other._alloc_scratch()
        return other

    def __repr__(self) -> str:
        return f"Simulator(num_qubits={self._n}, phases={self._phases})"

    def _state(self) -> tuple[np.ndarray, int, int]:
        """(tableau array, num_qubits, stabilizer words): for the observables module."""
        return self._tab, self._n, self._ws

    # -- validation helpers -------------------------------------------------------------

    def _qubit(self, q) -> int:
        q = operator.index(q)
        if not 0 <= q < self._n:
            raise IndexError(f"qubit {q} out of range for a {self._n}-qubit simulator")
        return q

    def _require_phases(self, what: str) -> None:
        if not self._phases:
            raise PhaseFreeError(
                f"{what} needs sign tracking, but this simulator was created with "
                "phases=False. Create it with phases=True (the default)."
            )

    def _coin(self, coin) -> int:
        drawn = int(next_bit(self._rng))  # always advance the stream
        if coin is None:
            return drawn
        if coin not in (0, 1):
            raise ValueError(f"coin must be 0 or 1, got {coin!r}")
        return int(coin)

    def _pauli_arrays(self, pauli) -> tuple[np.ndarray, np.ndarray, int]:
        """Parse a Pauli product into (qubits, codes, invert)."""
        if isinstance(pauli, PauliString):
            if len(pauli) != self._n:
                raise ValueError(
                    f"PauliString has {len(pauli)} qubits but the simulator has {self._n}"
                )
            if not pauli.is_hermitian:
                raise ValueError(f"can only measure Hermitian Paulis, got {pauli}")
            support = np.flatnonzero(pauli.xs | pauli.zs)
            codes = pauli.xs[support].astype(np.int64) + 2 * pauli.zs[support].astype(np.int64)
            return support.astype(np.int64), codes, pauli.phase // 2
        if not isinstance(pauli, str):
            raise TypeError(f"expected a PauliString or a string like 'X0*Z3', got {pauli!r}")
        text = pauli.strip()
        invert = 0
        if text.startswith("!"):
            invert = 1
            text = text[1:]
        acc: dict[int, tuple[int, int]] = {}
        phase = 0
        for token in filter(None, (t.strip() for t in text.split("*"))):
            letter, index = token[0].upper(), token[1:]
            if letter not in _CODE_OF or not index.isdigit():
                raise ValueError(f"bad Pauli term {token!r} in {pauli!r}; expected e.g. 'X0*Z3'")
            q = self._qubit(int(index))
            code = _CODE_OF[letter]
            x2, z2 = code & 1, code >> 1
            x1, z1 = acc.get(q, (0, 0))
            phase += pauli_product_phase(
                np.array([x1]), np.array([z1]), np.array([x2]), np.array([z2])
            )
            acc[q] = (x1 ^ x2, z1 ^ z2)
        if phase % 2:
            raise ValueError(f"{pauli!r} is not Hermitian (repeated qubits multiply to +-i)")
        invert ^= (phase % 4) // 2
        items = sorted((q, x + 2 * z) for q, (x, z) in acc.items() if x or z)
        qs = np.array([q for q, _ in items], dtype=np.int64)
        codes = np.array([c for _, c in items], dtype=np.int64)
        return qs, codes, invert

    # -- gates ---------------------------------------------------------------------------

    def _apply_1q_index(self, index: int, qubits: Iterable[int]) -> None:
        tab, ws, ph = self._tab, self._ws, self._phases
        for q in qubits:
            _k.apply_1q(tab, self._qubit(q), index, LIN1, ANF1, IDENTITY_SYM1, ws, ph)

    def _apply_2q_index(self, index: int, qubits: tuple[int, ...]) -> None:
        if len(qubits) % 2:
            raise ValueError(f"two-qubit gates need an even number of targets, got {len(qubits)}")
        tab, ws, ph = self._tab, self._ws, self._phases
        for j in range(0, len(qubits), 2):
            a, b = self._qubit(qubits[j]), self._qubit(qubits[j + 1])
            if a == b:
                raise ValueError(f"two-qubit gate targets must differ, got ({a}, {b})")
            _k.apply_2q(tab, a, b, index, LIN2, ANF2, IDENTITY_SYM2, ws, ph)

    def do(self, name: str, *targets: int) -> None:
        """Apply the gate called ``name`` (any Stim Clifford gate name or alias).

        One-qubit gates are applied to each target; two-qubit gates to
        consecutive target pairs, exactly like a line of a Stim circuit:
        ``sim.do("CX", 0, 1, 2, 3)`` applies CX(0,1) then CX(2,3).
        """
        k, index = _gate_index(name)
        if k == 1:
            self._apply_1q_index(index, targets)
        else:
            self._apply_2q_index(index, targets)

    def clifford_1q(self, index: int, *qubits: int) -> None:
        """Apply the one-qubit Clifford with canonical index ``index`` (0..23) to each qubit."""
        index = operator.index(index)
        if not 0 <= index < 24:
            raise ValueError(f"one-qubit Clifford index must be in [0, 24), got {index}")
        self._apply_1q_index(index, qubits)

    def clifford_2q(self, index: int, *qubits: int) -> None:
        """Apply the two-qubit Clifford with canonical index ``index`` (0..11519) to target pairs.

        ``Clifford.from_index(2, index)`` describes the gate; a uniform random
        ``index`` is a uniform random two-qubit Clifford.
        """
        index = operator.index(index)
        if not 0 <= index < 11520:
            raise ValueError(f"two-qubit Clifford index must be in [0, 11520), got {index}")
        self._apply_2q_index(index, qubits)

    def apply(self, clifford: Clifford, *qubits: int) -> None:
        """Apply a one- or two-qubit :class:`~sparsegf2_v3.Clifford` to the given qubit(s)."""
        k = clifford.num_qubits
        if k not in (1, 2):
            raise ValueError(f"only 1- and 2-qubit Cliffords can be applied directly, got k={k}")
        if len(qubits) != k:
            raise ValueError(f"a {k}-qubit Clifford needs {k} target qubits, got {len(qubits)}")
        if k == 1:
            self._apply_1q_index(clifford.index(), qubits)
        else:
            self._apply_2q_index(clifford.index(), qubits)

    # -- measurement ---------------------------------------------------------------------

    def _measure_1q(self, qubit, code: int, coin):
        q = self._qubit(qubit)
        c = self._coin(coin)
        r = _k.measure_1q(
            self._tab, self._n, self._ws, self._phases, q, code, c,
            self._anti, self._supp, self._lo, self._hi, self._qbuf, self._cbuf,
        )  # fmt: skip
        return (r & 1) if self._phases else None

    def measure(self, qubit: int, *, coin: int | None = None) -> int | None:
        """Measure ``qubit`` in the Z basis.

        Returns 0 (eigenvalue +1) or 1 (eigenvalue -1), or ``None`` in
        phase-free mode. ``coin`` forces the outcome if it is random.
        """
        return self._measure_1q(qubit, _Z, coin)

    def measure_x(self, qubit: int, *, coin: int | None = None) -> int | None:
        """Measure ``qubit`` in the X basis (like Stim's ``MX``)."""
        return self._measure_1q(qubit, _X, coin)

    def measure_y(self, qubit: int, *, coin: int | None = None) -> int | None:
        """Measure ``qubit`` in the Y basis (like Stim's ``MY``)."""
        return self._measure_1q(qubit, _Y, coin)

    def measure_many(self, *qubits: int) -> list[int] | None:
        """Measure several qubits in the Z basis, in order (like Stim's ``measure_many``)."""
        out = [self._measure_1q(q, _Z, None) for q in qubits]
        return out if self._phases else None

    def measure_pauli(self, pauli, *, coin: int | None = None) -> int | None:
        """Measure a Pauli product, e.g. ``"X0*Z3*Y5"`` or ``"!Z0*Z1"`` (Stim ``MPP`` syntax).

        ``pauli`` may also be a :class:`PauliString` on all qubits (sign +1 or
        -1). The return value and ``coin`` refer to the operator as written,
        so measuring ``"!Z0"`` returns 1 where ``"Z0"`` would return 0.
        """
        qs, codes, invert = self._pauli_arrays(pauli)
        c = self._coin(coin)
        r = _k.measure_pauli(
            self._tab, self._n, self._ws, self._phases, qs, codes, c ^ invert,
            self._anti, self._supp, self._lo, self._hi,
        )  # fmt: skip
        return ((r & 1) ^ invert) if self._phases else None

    # -- resets ----------------------------------------------------------------------------

    def _reset(self, qubits, measure_code: int, flip_code: int) -> None:
        for qubit in qubits:
            q = self._qubit(qubit)
            c = self._coin(None)
            r = _k.measure_1q(
                self._tab, self._n, self._ws, self._phases, q, measure_code, c,
                self._anti, self._supp, self._lo, self._hi, self._qbuf, self._cbuf,
            )  # fmt: skip
            if self._phases and (r & 1):
                _k.apply_pauli(self._tab, q, flip_code, self._ws)

    def reset(self, *qubits: int) -> None:
        """Reset each qubit to |0> (Stim's ``R``)."""
        self._reset(qubits, _Z, _X)

    def reset_x(self, *qubits: int) -> None:
        """Reset each qubit to |+> (Stim's ``RX``)."""
        self._reset(qubits, _X, _Z)

    def reset_y(self, *qubits: int) -> None:
        """Reset each qubit to |i> (Stim's ``RY``)."""
        self._reset(qubits, _Y, _X)

    # -- non-destructive queries -------------------------------------------------------

    def _peek(self, qs, codes, invert, what: str) -> int:
        self._require_phases(what)
        v = _k.peek_pauli(self._tab, self._n, self._ws, qs, codes, self._anti, self._supp)
        return -v if invert else v

    def peek_z(self, qubit: int) -> int:
        """+1 or -1 if a Z measurement of ``qubit`` is deterministic, else 0 (phases mode)."""
        q = self._qubit(qubit)
        return self._peek(np.array([q]), np.array([_Z]), 0, "peek_z")

    def peek_x(self, qubit: int) -> int:
        """+1 or -1 if an X measurement of ``qubit`` is deterministic, else 0 (phases mode)."""
        q = self._qubit(qubit)
        return self._peek(np.array([q]), np.array([_X]), 0, "peek_x")

    def peek_y(self, qubit: int) -> int:
        """+1 or -1 if a Y measurement of ``qubit`` is deterministic, else 0 (phases mode)."""
        q = self._qubit(qubit)
        return self._peek(np.array([q]), np.array([_Y]), 0, "peek_y")

    def peek_pauli(self, pauli) -> int:
        """Expectation value of a Pauli product: +1, -1, or 0 if random (phases mode).

        Same as Stim's ``peek_observable_expectation``; ``pauli`` is given as in
        :meth:`measure_pauli`.
        """
        qs, codes, invert = self._pauli_arrays(pauli)
        return self._peek(qs, codes, invert, "peek_pauli")

    def is_deterministic(self, pauli) -> bool:
        """Whether measuring ``pauli`` (as in :meth:`measure_pauli`) would give a fixed outcome.

        Works in both modes: in phase-free mode the *value* of a deterministic
        outcome is unknown, but whether it is deterministic is not.
        """
        qs, codes, _ = self._pauli_arrays(pauli)
        return bool(_k.is_deterministic(self._tab, self._ws, qs, codes, self._anti))

    # -- state export ---------------------------------------------------------------------

    def stabilizer_arrays(self, *, canonical: bool = False):
        """Stabilizers as numpy arrays ``(xs, zs, signs)``, each row one generator.

        ``xs[i, q]`` and ``zs[i, q]`` are the x and z bits of generator ``i`` on
        qubit ``q``; ``signs[i]`` is 1 for a minus sign (always 0 in phase-free
        mode). With ``canonical=True`` the generators are put in Stim's
        canonical form, so two simulators in the same state give identical arrays.
        """
        xs, zs, signs = _k.rows_of(self._tab, self._n, 0, self._n)
        if canonical:
            _k.canonicalize(xs, zs, signs, self._phases)
        return xs, zs, signs

    def stabilizers(self, *, canonical: bool = True) -> list[PauliString]:
        """The stabilizer generators as :class:`PauliString` objects.

        By default they are in Stim's canonical form, so the list equals
        ``stim.TableauSimulator.canonical_stabilizers()`` for the same state
        (without signs in phase-free mode, where every sign reads ``+``).
        """
        xs, zs, signs = self.stabilizer_arrays(canonical=canonical)
        return [PauliString.from_bits(xs[i], zs[i], 2 * int(signs[i])) for i in range(self._n)]

    def destabilizers(self) -> list[PauliString]:
        """The destabilizer generators (phases mode), unsigned; destabilizer i pairs with stabilizer i."""
        self._require_phases("destabilizers")
        xs, zs, _ = _k.rows_of(self._tab, self._n, 64 * self._ws, self._n)
        return [PauliString.from_bits(xs[i], zs[i], 0) for i in range(self._n)]

    # -- circuits --------------------------------------------------------------------------

    def run(self, circuit: Circuit, *, coins=None, observe=None) -> RunResult:
        """Execute a :class:`~sparsegf2_v3.Circuit` on this state, in one compiled call.

        Parameters
        ----------
        circuit
            The circuit. It may use fewer qubits than the simulator, not more.
        coins
            Optional 0/1 array with one entry per measurement result: the
            *recorded* value to produce if that measurement is random
            (deterministic results ignore it). Use it for post-selection or to
            replay a recorded run. Resets still draw from the simulator's stream.
        observe
            Optional callback ``observe(sim, tick)`` called after every ``TICK``
            instruction, with the simulator paused in its current state. Its
            return values are collected in ``RunResult.observations`` as
            ``(tick, value)`` pairs. This is how to measure stabilizer-tableau
            properties as a circuit evolves. The callback should only read the
            state.

        Returns
        -------
        RunResult
            ``measurements`` (phases mode), ``random``, ``detectors``,
            ``observables`` and ``observations``.
        """
        program = compile_circuit(circuit)
        if program.num_qubits > self._n:
            raise ValueError(
                f"circuit uses {program.num_qubits} qubits but the simulator has {self._n}"
            )
        m = program.num_measurements
        record = np.zeros(m, dtype=np.uint8)
        random_flags = np.zeros(m, dtype=np.uint8)
        if coins is None:
            coin_array = np.zeros(0, dtype=np.uint8)
            use_coins = False
        else:
            coin_array = np.asarray(coins).astype(np.int64).ravel()
            if coin_array.shape != (m,):
                raise ValueError(
                    f"coins must have one entry per measurement ({m}), got {coin_array.shape}"
                )
            if np.any((coin_array != 0) & (coin_array != 1)):
                raise ValueError("coins must be 0 or 1")
            coin_array = coin_array.astype(np.uint8)
            use_coins = True
        observations: list = []
        position, done, tick = 0, 0, 0
        while True:
            position, done, at_tick = run_program(
                self._tab, self._n, self._ws, self._phases, program.ops, program.params,
                program.mpp_qubits, program.mpp_codes, position, observe is not None,
                coin_array, use_coins, self._rng, record, random_flags, done,
                LIN1, ANF1, IDENTITY_SYM1, LIN2, ANF2, IDENTITY_SYM2,
                self._anti, self._supp, self._lo, self._hi, self._qbuf, self._cbuf, False,
            )  # fmt: skip
            if not at_tick:
                break
            observations.append((tick, observe(self, tick)))
            tick += 1
        return RunResult(self._phases, program, record, random_flags, observations)

    # -- entanglement ---------------------------------------------------------------------

    def entropy(self, qubits: Iterable[int]) -> int:
        """Entanglement entropy (in bits) of the given qubits.

        Computed as ``rank(stabilizers restricted to A) - |A|`` (Fattal et al.,
        quant-ph/0406168), using whichever of A and its complement is smaller.
        """
        a = np.unique(np.fromiter((self._qubit(q) for q in qubits), dtype=np.int64))
        if a.size == 0 or a.size == self._n:
            return 0
        if 2 * a.size > self._n:
            mask = np.ones(self._n, dtype=bool)
            mask[a] = False
            a = np.flatnonzero(mask).astype(np.int64)
        if a.size == 1:
            return int(_k.single_qubit_entropy(self._tab, int(a[0]), self._ws))
        scratch = np.empty((2 * a.size, self._ws), dtype=np.uint64)
        return int(_k.columns_rank(self._tab, a, self._ws, scratch)) - int(a.size)


# -- gate methods ---------------------------------------------------------------------------


_GATE_INDEX: dict[str, tuple[int, int]] = {}


def _gate_index(name: str) -> tuple[int, int]:
    """(arity, canonical Clifford index) for a gate name, cached."""
    key = name.upper()
    cached = _GATE_INDEX.get(key)
    if cached is None:
        info = gate(key)
        if info.kind != "unitary":
            raise ValueError(
                f"{info.name} is not a unitary gate; use the measure/reset methods "
                "or Simulator.run(circuit) for non-unitary instructions"
            )
        cached = (info.num_qubits, info.clifford.index())
        _GATE_INDEX[key] = cached
    return cached


def _make_gate_method(canonical: str, k: int, index: int):
    if k == 1:

        def method(self, *qubits: int) -> None:
            self._apply_1q_index(index, qubits)

        method.__doc__ = f"Apply {canonical} to each target qubit."
    else:

        def method(self, *qubits: int) -> None:
            self._apply_2q_index(index, qubits)

        method.__doc__ = f"Apply {canonical} to each consecutive pair of target qubits."
    return method


# One lowercase method per gate name and alias (sim.h, sim.cx, sim.cnot, sim.sqrt_x_dag, ...).
for _info in all_gates("unitary"):
    _k_arity, _index = _info.num_qubits, _info.clifford.index()
    for _name in (_info.name, *_info.aliases):
        _method = _make_gate_method(_info.name, _k_arity, _index)
        _method.__name__ = _name.lower()
        _method.__qualname__ = f"Simulator.{_name.lower()}"
        setattr(Simulator, _name.lower(), _method)
del _info, _k_arity, _index, _name, _method
