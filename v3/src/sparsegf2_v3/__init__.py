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
from sparsegf2_v3.circuit import (
    Circuit,
    Instruction,
    RepeatBlock,
    Target,
    target_combiner,
    target_inv,
    target_rec,
    target_x,
    target_y,
    target_z,
)
from sparsegf2_v3.clifford import (
    Clifford,
    num_cliffords,
    random_clifford_indices,
    symplectic_group,
)
from sparsegf2_v3.errors import PhaseFreeError
from sparsegf2_v3.gates import GateInfo, all_gates, gate
from sparsegf2_v3.pauli import PauliString
from sparsegf2_v3.result import RunResult, SampleResult
from sparsegf2_v3.sampling import sample
from sparsegf2_v3.simulator import Simulator

__all__ = [
    "Circuit",
    "Clifford",
    "GateInfo",
    "Instruction",
    "PauliString",
    "PhaseFreeError",
    "RepeatBlock",
    "RunResult",
    "SampleResult",
    "Simulator",
    "Target",
    "__version__",
    "all_gates",
    "gate",
    "num_cliffords",
    "random_clifford_indices",
    "sample",
    "symplectic_group",
    "target_combiner",
    "target_inv",
    "target_rec",
    "target_x",
    "target_y",
    "target_z",
]
