"""Compile a :class:`~sparsegf2.circuit.Circuit` into flat arrays for the kernel.

The compiled :class:`Program` is what :meth:`Simulator.run` executes in a single
Numba call. ``REPEAT`` blocks are unrolled (up to ``max_ops`` operations), gate
names become canonical Clifford indices, MPP products are stored in side
arrays, and detectors/observables become lists of absolute measurement indices.

Operation table (``ops[i] = [opcode, a, b, c, d]``):

=============  ==========================================================
OP_1Q          a = qubit, b = 1-qubit Clifford index
OP_2Q          a, b = qubits, c = 2-qubit Clifford index
OP_MEASURE     a = qubit, b = basis code, c = flags, d = flip-prob slot
OP_MPP         a = start, b = length (into mpp arrays), c = flags, d = slot
OP_RESET       a = qubit, b = basis code
OP_NOISE1      a = qubit, b = channel, c = first parameter slot
OP_NOISE2      a, b = qubits, c = channel, d = first parameter slot
OP_FEEDBACK    a = absolute record index, b = qubit, c = Pauli code
OP_TICK        (a checkpoint for ``Simulator.run(..., observe=...)``)
=============  ==========================================================

Basis/Pauli codes are X = 1, Z = 2, Y = 3. Measurement flags: bit 0 = record
the inverted result, bit 1 = reset after measuring (MR, MRX, MRY). A slot of
-1 means "no measurement noise".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sparsegf2.circuit import Circuit, Instruction, _combine_product, _split_products

OP_1Q = 1
OP_2Q = 2
OP_MEASURE = 3
OP_MPP = 4
OP_RESET = 5
OP_NOISE1 = 6
OP_NOISE2 = 7
OP_FEEDBACK = 8
OP_TICK = 9

FLAG_INVERT = 1
FLAG_RESET = 2

CHANNELS = {
    "X_ERROR": 0,
    "Y_ERROR": 1,
    "Z_ERROR": 2,
    "DEPOLARIZE1": 3,
    "PAULI_CHANNEL_1": 4,
    "DEPOLARIZE2": 5,
    "PAULI_CHANNEL_2": 6,
}

_CODE = {"X": 1, "Z": 2, "Y": 3}
_FEEDBACK_PAULI = {"CX": 1, "CY": 3, "CZ": 2}

DEFAULT_MAX_OPS = 50_000_000


@dataclass(eq=False)
class Program:
    """A circuit compiled for the kernel. Treat as read-only."""

    ops: np.ndarray  # (num_ops, 5) int64
    params: np.ndarray  # float64 noise / flip probabilities
    mpp_qubits: np.ndarray  # int64
    mpp_codes: np.ndarray  # int64
    num_qubits: int
    num_measurements: int
    num_ticks: int
    detector_ptr: np.ndarray  # CSR offsets, length num_detectors + 1
    detector_idx: np.ndarray  # absolute measurement indices
    observable_ptr: np.ndarray
    observable_idx: np.ndarray

    _reference: np.ndarray | None = None

    def reference_record(self) -> np.ndarray:
        """The noiseless reference record: every random collapse biased to +1, as in Stim.

        Detection events and observable flips are reported relative to it, as
        Stim's samplers do; computed once and cached. Collapses happen in target
        order. Stim does the same in the Z basis, but collapses the random
        targets of one X- or Y-basis instruction (``MX``, ``MY``, ``MRX``,
        ``MRY``, ``RX``, ``RY``, ``MXX``, ``MYY``) in ascending qubit order, so
        when such an instruction acts on several entangled qubits (``MY 1 0``)
        the record can differ from Stim's ``reference_sample()``. Deterministic
        detectors and observables (fixed in every noiseless run) are
        unaffected; the per-shot values of non-deterministic ones can differ
        from Stim's, with the same distribution.
        """
        if self._reference is None:
            from sparsegf2._kernel.program import run_program
            from sparsegf2._kernel.tableau import init_zero
            from sparsegf2._kernel.tables import (
                ANF1,
                ANF2,
                IDENTITY_SYM1,
                IDENTITY_SYM2,
                LIN1,
                LIN2,
            )

            n = max(self.num_qubits, 1)
            ws = (n + 63) // 64
            tab = np.zeros((2 * n + 1, 2 * ws), dtype=np.uint64)
            init_zero(tab, n, ws, True)
            record = np.zeros(self.num_measurements, dtype=np.uint8)
            flags = np.zeros(self.num_measurements, dtype=np.uint8)
            run_program(
                tab, n, ws, True, self.ops, self.params, self.mpp_qubits, self.mpp_codes,
                0, False, np.zeros(0, dtype=np.uint8), False, np.ones(4, dtype=np.uint64),
                record, flags, 0, LIN1, ANF1, IDENTITY_SYM1, LIN2, ANF2, IDENTITY_SYM2,
                np.zeros(2 * ws, dtype=np.uint64), np.zeros(max(n, ws), dtype=np.int64),
                np.zeros(ws, dtype=np.uint64), np.zeros(ws, dtype=np.uint64),
                np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), True,
            )  # fmt: skip
            self._reference = record
        return self._reference

    def detection_events(self, records: np.ndarray) -> np.ndarray:
        """Detector parities XOR their reference values (Stim's detection events)."""
        ref = parity_of(self.reference_record(), self.detector_ptr, self.detector_idx)
        return parity_of(records, self.detector_ptr, self.detector_idx) ^ ref

    def observable_flips(self, records: np.ndarray) -> np.ndarray:
        """Observable parities XOR their reference values (as in Stim's samplers)."""
        ref = parity_of(self.reference_record(), self.observable_ptr, self.observable_idx)
        return parity_of(records, self.observable_ptr, self.observable_idx) ^ ref

    @property
    def num_detectors(self) -> int:
        return len(self.detector_ptr) - 1

    @property
    def num_observables(self) -> int:
        return len(self.observable_ptr) - 1


class _Builder:
    def __init__(self, max_ops: int):
        self.max_ops = max_ops
        self.current: Instruction | None = None  # for error messages
        self.ops: list[tuple[int, ...]] = []
        self.params: list[float] = []
        self.mpp_q: list[int] = []
        self.mpp_c: list[int] = []
        self.num_meas = 0
        self.num_ticks = 0
        self.top_qubit = -1
        self.detectors: list[list[int]] = []
        self.observables: dict[int, list[int]] = {}

    def emit(self, *fields: int) -> None:
        if len(self.ops) >= self.max_ops:
            raise ValueError(
                f"circuit unrolls to more than {self.max_ops:,} operations; if this is "
                "intended, raise the limit: Simulator.run(circuit, max_ops=...) or "
                "sample(circuit, shots, max_ops=...)"
            )
        self.ops.append((*fields, *([0] * (5 - len(fields)))))

    def slot(self, values) -> int:
        start = len(self.params)
        self.params.extend(float(v) for v in values)
        return start

    def qubit(self, q: int) -> int:
        self.top_qubit = max(self.top_qubit, q)
        return q

    def record_index(self, lookback: int) -> int:
        idx = self.num_meas + lookback
        if idx < 0:
            raise ValueError(
                f"rec[{lookback}] in '{self.current}' refers to a measurement before the start "
                f"of the circuit (only {self.num_meas} recorded before it)"
            )
        return idx

    def instruction(self, ins: Instruction) -> None:
        self.current = ins
        info = ins.info
        kind = info.kind
        name = info.name
        t = ins.targets

        if kind == "unitary":
            assert info.clifford is not None, "unitary gates carry their Clifford"
            index = info.clifford.index()
            if info.num_qubits == 1:
                for tg in t:
                    self.emit(OP_1Q, self.qubit(tg.value), index)
            else:
                for a, b in zip(t[::2], t[1::2], strict=True):
                    if a.kind == "rec" or b.kind == "rec":
                        rec, q = (a, b) if a.kind == "rec" else (b, a)
                        self.emit(
                            OP_FEEDBACK,
                            self.record_index(rec.value),
                            self.qubit(q.value),
                            _FEEDBACK_PAULI[name],
                        )
                    else:
                        self.emit(OP_2Q, self.qubit(a.value), self.qubit(b.value), index)
        elif kind == "clifford_index":
            k = info.num_qubits
            for j, clifford_number in enumerate(ins.args):
                if k == 1:
                    self.emit(OP_1Q, self.qubit(t[j].value), int(clifford_number))
                else:
                    self.emit(
                        OP_2Q,
                        self.qubit(t[2 * j].value),
                        self.qubit(t[2 * j + 1].value),
                        int(clifford_number),
                    )
        elif kind in ("measure", "measure_reset"):
            slot = self.slot(ins.args) if ins.args else -1
            flags_base = FLAG_RESET if kind == "measure_reset" else 0
            assert info.basis is not None, "measurements carry a basis"
            for tg in t:
                flags = flags_base | (FLAG_INVERT if tg.inverted else 0)
                self.emit(OP_MEASURE, self.qubit(tg.value), _CODE[info.basis], flags, slot)
                self.num_meas += 1
        elif kind == "measure_pair":
            slot = self.slot(ins.args) if ins.args else -1
            assert info.basis is not None, "measurements carry a basis"
            code = _CODE[info.basis]
            for a, b in zip(t[::2], t[1::2], strict=True):
                start = len(self.mpp_q)
                self.mpp_q.extend([self.qubit(a.value), self.qubit(b.value)])
                self.mpp_c.extend([code, code])
                inv = FLAG_INVERT if (a.inverted ^ b.inverted) else 0
                self.emit(OP_MPP, start, 2, inv, slot)
                self.num_meas += 1
        elif kind == "measure_product":
            slot = self.slot(ins.args) if ins.args else -1
            for factors in _split_products(t):
                self._emit_product(factors, slot)
        elif kind == "reset":
            assert info.basis is not None, "resets carry a basis"
            for tg in t:
                self.emit(OP_RESET, self.qubit(tg.value), _CODE[info.basis])
        elif kind == "noise":
            channel = CHANNELS[name]
            slot = self.slot(ins.args)
            if info.num_qubits == 1:
                for tg in t:
                    self.emit(OP_NOISE1, self.qubit(tg.value), channel, slot)
            else:
                for a, b in zip(t[::2], t[1::2], strict=True):
                    self.emit(OP_NOISE2, self.qubit(a.value), self.qubit(b.value), channel, slot)
        elif name == "TICK":
            self.emit(OP_TICK)
            self.num_ticks += 1
        elif name == "DETECTOR":
            self.detectors.append(sorted(self.record_index(tg.value) for tg in t))
        elif name == "OBSERVABLE_INCLUDE":
            self.observables.setdefault(int(ins.args[0]), []).extend(
                self.record_index(tg.value) for tg in t
            )
        elif name == "QUBIT_COORDS":
            for tg in t:
                self.qubit(tg.value)
        # SHIFT_COORDS: no effect on execution

    def _emit_product(self, factors, slot: int) -> None:
        """One MPP product: combine repeated qubits exactly, then emit."""
        acc, invert = _combine_product(factors)
        for f in factors:
            self.qubit(f.value)
        start = len(self.mpp_q)
        for q in sorted(acc):
            x, z = acc[q]
            if x or z:
                self.mpp_q.append(q)
                self.mpp_c.append(x + 2 * z)
        self.emit(OP_MPP, start, len(self.mpp_q) - start, FLAG_INVERT if invert else 0, slot)
        self.num_meas += 1

    def build(self) -> Program:
        ops = np.array(self.ops, dtype=np.int64).reshape(-1, 5)
        det_ptr = np.zeros(len(self.detectors) + 1, dtype=np.int64)
        det_ptr[1:] = np.cumsum([len(d) for d in self.detectors])
        det_idx = np.array([i for d in self.detectors for i in d], dtype=np.int64)
        num_obs = (max(self.observables) + 1) if self.observables else 0
        obs_lists = [self.observables.get(k, []) for k in range(num_obs)]
        obs_ptr = np.zeros(num_obs + 1, dtype=np.int64)
        obs_ptr[1:] = np.cumsum([len(o) for o in obs_lists])
        obs_idx = np.array([i for o in obs_lists for i in o], dtype=np.int64)
        return Program(
            ops=ops,
            params=np.array(self.params, dtype=np.float64),
            mpp_qubits=np.array(self.mpp_q, dtype=np.int64),
            mpp_codes=np.array(self.mpp_c, dtype=np.int64),
            num_qubits=self.top_qubit + 1,
            num_measurements=self.num_meas,
            num_ticks=self.num_ticks,
            detector_ptr=det_ptr,
            detector_idx=det_idx,
            observable_ptr=obs_ptr,
            observable_idx=obs_idx,
        )


def compile_circuit(circuit: Circuit, *, max_ops: int = DEFAULT_MAX_OPS) -> Program:
    """Compile ``circuit`` (cached on the circuit until it, or a REPEAT body in it, is modified).

    A cached program is reused whenever it fits within ``max_ops``.
    """
    if not isinstance(circuit, Circuit):
        raise TypeError(_not_a_circuit(circuit))
    key = circuit._state_key()
    cached = getattr(circuit, "_compiled", None)
    if cached is not None and cached[0] == key and len(cached[1].ops) <= max_ops:
        return cached[1]
    builder = _Builder(max_ops)
    for ins in circuit.flattened():
        builder.instruction(ins)
    program = builder.build()
    circuit._compiled = (key, program)
    return program


def _not_a_circuit(obj) -> str:
    what = f"{type(obj).__module__.split('.')[0]}.{type(obj).__name__}"
    hint = ""
    if what.startswith("stim."):
        hint = "; convert it once with sparsegf2.Circuit.from_stim(stim_circuit)"
        what = "a stim.Circuit" if type(obj).__name__ == "Circuit" else f"a {what}"
    elif isinstance(obj, str):
        hint = "; parse Stim circuit text with sparsegf2.Circuit(text)"
        what = "a str"
    else:
        what = f"a {type(obj).__name__}"
    return f"expected a sparsegf2.Circuit, got {what}{hint}"


def parity_of(record: np.ndarray, ptr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """XOR of record bits over each CSR group (detectors / observables).

    ``record`` may be 1-D (one shot) or 2-D (shots x measurements).
    """
    groups = len(ptr) - 1
    if record.ndim == 1:
        out = np.zeros(groups, dtype=np.uint8)
        for g in range(groups):
            # numpy's stubs lack the plain reduce(array, initial=...) overload
            out[g] = np.bitwise_xor.reduce(  # type: ignore[call-overload, unused-ignore]
                record[idx[ptr[g] : ptr[g + 1]]], initial=0
            )
        return out
    out = np.zeros((record.shape[0], groups), dtype=np.uint8)
    for g in range(groups):
        sel = idx[ptr[g] : ptr[g + 1]]
        if sel.size:
            # numpy's stubs lack the plain reduce(array, axis=...) overload
            out[:, g] = np.bitwise_xor.reduce(record[:, sel], axis=1)  # type: ignore[call-overload, unused-ignore]
    return out
