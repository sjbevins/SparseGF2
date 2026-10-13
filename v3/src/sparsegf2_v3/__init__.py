"""SparseGF2 v3: a stabilizer-circuit simulator for monitored Clifford circuits.

This is a ground-up rebuild of SparseGF2, developed milestone by milestone and
verified against Stim at every step. See ``docs/DESIGN.md`` for the plan and
``README.md`` for what works today.

Milestone 1 (this version) provides the foundations every later layer uses:

* :class:`PauliString`: Pauli operators with exact phases (Stim conventions).
* :class:`Clifford`: small Clifford unitaries as signed tableaux, with a stable
  integer index for every 1- and 2-qubit Clifford.
* :func:`gate`: the gate library, every Clifford unitary Stim knows, under
  Stim's names.
"""

from sparsegf2_v3._version import __version__
from sparsegf2_v3.clifford import (
    Clifford,
    num_cliffords,
    random_clifford_indices,
    symplectic_group,
)
from sparsegf2_v3.gates import GateInfo, all_gates, gate
from sparsegf2_v3.pauli import PauliString

__all__ = [
    "Clifford",
    "GateInfo",
    "PauliString",
    "__version__",
    "all_gates",
    "gate",
    "num_cliffords",
    "random_clifford_indices",
    "symplectic_group",
]
