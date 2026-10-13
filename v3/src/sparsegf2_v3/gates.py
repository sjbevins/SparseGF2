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

Measurements, resets, noise channels and annotations (M, MPP, R, DEPOLARIZE1,
DETECTOR, ...) are added in later milestones; see ``docs/DESIGN.md``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from sparsegf2_v3.clifford import Clifford

__all__ = ["GateInfo", "all_gates", "gate"]

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


@dataclass(frozen=True)
class GateInfo:
    """Description of one gate.

    Attributes
    ----------
    name
        Canonical (Stim) name, e.g. ``"CX"``.
    aliases
        Other accepted names, e.g. ``("CNOT", "ZCX")``.
    kind
        ``"unitary"`` for now; measurement, reset, noise and annotation kinds
        arrive with the circuit layer.
    num_qubits
        Qubits per application: 1 or 2. ``CX 0 1 2 3`` applies CX twice.
    clifford
        The gate's signed tableau.
    """

    name: str
    aliases: tuple[str, ...]
    kind: str
    num_qubits: int
    clifford: Clifford


@functools.cache
def _registry() -> dict[str, GateInfo]:
    reg: dict[str, GateInfo] = {}
    for table, k in ((_UNITARY_1Q, 1), (_UNITARY_2Q, 2)):
        for name, (xs, zs, aliases) in table.items():
            info = GateInfo(name, aliases, "unitary", k, Clifford.from_strings(xs, zs))
            for key in (name, *aliases):
                reg[key] = info
    return reg


def gate(name: str) -> GateInfo:
    """Look up a gate by name or alias (case-insensitive, like Stim).

    >>> gate("cnot").name
    'CX'
    """
    try:
        return _registry()[name.upper()]
    except KeyError:
        raise KeyError(f"unknown gate {name!r}; see sparsegf2_v3.gates.all_gates()") from None


def all_gates() -> list[GateInfo]:
    """Every distinct gate, sorted by canonical name."""
    unique = {info.name: info for info in _registry().values()}
    return [unique[k] for k in sorted(unique)]
