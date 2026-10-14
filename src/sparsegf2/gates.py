"""The gate library: every Clifford unitary Stim knows, under Stim's names.

Gate names, aliases, qubit-ordering and sign conventions are identical to Stim
(https://github.com/quantumlib/Stim/blob/main/doc/gates.md), so a circuit
written for Stim means the same thing here. The definitions below are plain
data (generator images, written the way Stim's documentation writes them);
Stim is *not* needed at runtime. The test suite checks every entry against
``stim.gate_data()``.

Reading an entry: ``"CX": (["+XX", "+_X"], ["+Z_", "+ZZ"])`` says that
conjugating by CX sends ``X_0 -> X_0 X_1``, ``X_1 -> X_1``, ``Z_0 -> Z_0`` and
``Z_1 -> Z_0 Z_1``, where qubit 0 is the first target (the control).

The registry also describes the non-unitary instructions of Stim's circuit
language that SparseGF2 supports (measurements, resets, Pauli noise,
annotations) and two SparseGF2 extensions, ``CLIFFORD_1Q`` and ``CLIFFORD_2Q``,
which name a Clifford by its canonical index.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from sparsegf2.clifford import Clifford

__all__ = ["KINDS", "GateInfo", "all_gates", "gate"]

# name: (x images, z images, aliases)
_UNITARY_1Q: dict[str, tuple[list[str], list[str], tuple[str, ...]]] = {
    "I": (["+X"], ["+Z"], ()),
    "X": (["+X"], ["-Z"], ()),
    "Y": (["-X"], ["-Z"], ()),
    "Z": (["-X"], ["+Z"], ()),
    "H": (["+Z"], ["+X"], ("H_XZ",)),
    "H_XY": (["+Y"], ["-Z"], ()),
    "H_YZ": (["-X"], ["+Y"], ()),
    "H_NXY": (["-Y"], ["-Z"], ()),
    "H_NXZ": (["-Z"], ["-X"], ()),
    "H_NYZ": (["-X"], ["-Y"], ()),
    "S": (["+Y"], ["+Z"], ("SQRT_Z",)),
    "S_DAG": (["-Y"], ["+Z"], ("SQRT_Z_DAG",)),
    "SQRT_X": (["+X"], ["-Y"], ()),
    "SQRT_X_DAG": (["+X"], ["+Y"], ()),
    "SQRT_Y": (["-Z"], ["+X"], ()),
    "SQRT_Y_DAG": (["+Z"], ["-X"], ()),
    "C_XYZ": (["+Y"], ["+X"], ()),
    "C_ZYX": (["+Z"], ["+Y"], ()),
    "C_NXYZ": (["-Y"], ["-X"], ()),
    "C_XNYZ": (["-Y"], ["+X"], ()),
    "C_XYNZ": (["+Y"], ["-X"], ()),
    "C_NZYX": (["-Z"], ["-Y"], ()),
    "C_ZNYX": (["+Z"], ["-Y"], ()),
    "C_ZYNX": (["-Z"], ["+Y"], ()),
}

_UNITARY_2Q: dict[str, tuple[list[str], list[str], tuple[str, ...]]] = {
    "II": (["+X_", "+_X"], ["+Z_", "+_Z"], ()),
    "CX": (["+XX", "+_X"], ["+Z_", "+ZZ"], ("CNOT", "ZCX")),
    "CY": (["+XY", "+ZX"], ["+Z_", "+ZZ"], ("ZCY",)),
    "CZ": (["+XZ", "+ZX"], ["+Z_", "+_Z"], ("ZCZ",)),
    "XCX": (["+X_", "+_X"], ["+ZX", "+XZ"], ()),
    "XCY": (["+X_", "+XX"], ["+ZY", "+XZ"], ()),
    "XCZ": (["+X_", "+XX"], ["+ZZ", "+_Z"], ()),
    "YCX": (["+XX", "+_X"], ["+ZX", "+YZ"], ()),
    "YCY": (["+XY", "+YX"], ["+ZY", "+YZ"], ()),
    "YCZ": (["+XZ", "+YX"], ["+ZZ", "+_Z"], ()),
    "SWAP": (["+_X", "+X_"], ["+_Z", "+Z_"], ()),
    "ISWAP": (["+ZY", "+YZ"], ["+_Z", "+Z_"], ()),
    "ISWAP_DAG": (["-ZY", "-YZ"], ["+_Z", "+Z_"], ()),
    "SQRT_XX": (["+X_", "+_X"], ["-YX", "-XY"], ()),
    "SQRT_XX_DAG": (["+X_", "+_X"], ["+YX", "+XY"], ()),
    "SQRT_YY": (["-ZY", "-YZ"], ["+XY", "+YX"], ()),
    "SQRT_YY_DAG": (["+ZY", "+YZ"], ["-XY", "-YX"], ()),
    "SQRT_ZZ": (["+YZ", "+ZY"], ["+Z_", "+_Z"], ()),
    "SQRT_ZZ_DAG": (["-YZ", "-ZY"], ["+Z_", "+_Z"], ()),
    "CXSWAP": (["+XX", "+X_"], ["+_Z", "+ZZ"], ()),
    "SWAPCX": (["+_X", "+XX"], ["+ZZ", "+Z_"], ()),
    "CZSWAP": (["+ZX", "+XZ"], ["+_Z", "+Z_"], ("SWAPCZ",)),
}


# Non-unitary instructions, by kind. Values: (aliases, qubits per application,
# basis or None, allowed argument counts or None for "any number").
_OTHER: dict[str, tuple[str, tuple[str, ...], int, str | None, tuple[int, ...] | None]] = {
    # measurements; the optional argument is the result-flip probability
    "M": ("measure", ("MZ",), 1, "Z", (0, 1)),
    "MX": ("measure", (), 1, "X", (0, 1)),
    "MY": ("measure", (), 1, "Y", (0, 1)),
    "MR": ("measure_reset", ("MRZ",), 1, "Z", (0, 1)),
    "MRX": ("measure_reset", (), 1, "X", (0, 1)),
    "MRY": ("measure_reset", (), 1, "Y", (0, 1)),
    "MXX": ("measure_pair", (), 2, "X", (0, 1)),
    "MYY": ("measure_pair", (), 2, "Y", (0, 1)),
    "MZZ": ("measure_pair", (), 2, "Z", (0, 1)),
    "MPP": ("measure_product", (), 0, None, (0, 1)),
    # resets
    "R": ("reset", ("RZ",), 1, "Z", (0,)),
    "RX": ("reset", (), 1, "X", (0,)),
    "RY": ("reset", (), 1, "Y", (0,)),
    # Pauli noise channels
    "X_ERROR": ("noise", (), 1, None, (1,)),
    "Y_ERROR": ("noise", (), 1, None, (1,)),
    "Z_ERROR": ("noise", (), 1, None, (1,)),
    "DEPOLARIZE1": ("noise", (), 1, None, (1,)),
    "PAULI_CHANNEL_1": ("noise", (), 1, None, (3,)),
    "DEPOLARIZE2": ("noise", (), 2, None, (1,)),
    "PAULI_CHANNEL_2": ("noise", (), 2, None, (15,)),
    # annotations (no effect on the state)
    "TICK": ("annotation", (), 0, None, (0,)),
    "DETECTOR": ("annotation", (), 0, None, None),
    "OBSERVABLE_INCLUDE": ("annotation", (), 0, None, (1,)),
    "QUBIT_COORDS": ("annotation", (), 0, None, None),
    "SHIFT_COORDS": ("annotation", (), 0, None, None),
    # SparseGF2 extensions: a Clifford named by its canonical index, one
    # argument per application (see Clifford.from_index). Not part of Stim's
    # language; Circuit.to_stim() expands them into Stim gates.
    "CLIFFORD_1Q": ("clifford_index", (), 1, None, None),
    "CLIFFORD_2Q": ("clifford_index", (), 2, None, None),
}

KINDS = (
    "unitary",
    "measure",
    "measure_reset",
    "measure_pair",
    "measure_product",
    "reset",
    "noise",
    "annotation",
    "clifford_index",
)


@dataclass(frozen=True)
class GateInfo:
    """Description of one instruction.

    Attributes
    ----------
    name
        Canonical (Stim) name, e.g. ``"CX"``.
    aliases
        Other accepted names, e.g. ``("CNOT", "ZCX")``.
    kind
        One of :data:`KINDS`: ``"unitary"``, ``"measure"`` (M, MX, MY),
        ``"measure_reset"`` (MR, MRX, MRY), ``"measure_pair"`` (MXX, MYY, MZZ),
        ``"measure_product"`` (MPP), ``"reset"`` (R, RX, RY), ``"noise"``,
        ``"annotation"`` (TICK, DETECTOR, ...) or ``"clifford_index"`` (the
        SparseGF2 extensions CLIFFORD_1Q and CLIFFORD_2Q).
    num_qubits
        Qubits per application: 1 or 2 (``CX 0 1 2 3`` applies CX twice), or 0
        when targets are not grouped that way (MPP, annotations).
    clifford
        The signed tableau of a unitary gate, else ``None``.
    basis
        ``"X"``, ``"Y"`` or ``"Z"`` for measurements and resets, else ``None``.
    num_args
        Allowed numbers of parenthesized arguments, or ``None`` if any number
        is allowed.
    """

    name: str
    aliases: tuple[str, ...]
    kind: str
    num_qubits: int
    clifford: Clifford | None = None
    basis: str | None = None
    num_args: tuple[int, ...] | None = (0,)


@functools.cache
def _registry() -> dict[str, GateInfo]:
    reg: dict[str, GateInfo] = {}
    for table, k in ((_UNITARY_1Q, 1), (_UNITARY_2Q, 2)):
        for name, (xs, zs, aliases) in table.items():
            info = GateInfo(name, aliases, "unitary", k, Clifford.from_strings(xs, zs))
            for key in (name, *aliases):
                reg[key] = info
    for name, (kind, aliases, k, basis, num_args) in _OTHER.items():
        info = GateInfo(name, aliases, kind, k, None, basis, num_args)
        for key in (name, *aliases):
            reg[key] = info
    return reg


def gate(name: str) -> GateInfo:
    """Look up an instruction by name or alias (case-insensitive, like Stim).

    >>> gate("cnot").name
    'CX'
    >>> gate("MZ").kind
    'measure'
    """
    try:
        return _registry()[name.upper()]
    except KeyError:
        raise KeyError(f"unknown gate {name!r}; see sparsegf2.gates.all_gates()") from None


def all_gates(kind: str | None = None) -> list[GateInfo]:
    """Every distinct instruction (optionally only one kind), sorted by name.

    >>> len(all_gates("unitary"))
    46
    """
    if kind is not None and kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    unique = {info.name: info for info in _registry().values()}
    return [unique[k] for k in sorted(unique) if kind is None or unique[k].kind == kind]
