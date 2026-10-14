"""SparseGF2: a fast stabilizer-circuit simulator for monitored Clifford circuits.

Verified against Stim. The main pieces:

* :class:`Simulator`, :class:`Circuit` and :func:`sample`: single circuits in
  the Stim circuit language, with or without signs (``phases``).
* :mod:`~sparsegf2.observables`: entanglement, code and erasure quantities
  of a simulator's state.
* :mod:`~sparsegf2.mipt`: ensembles of random monitored circuits at full
  speed (graph collections, reference pictures, probes on their own
  checkpoints), and :mod:`~sparsegf2.graphs` for the interaction graphs.
* :mod:`~sparsegf2.codes`: :class:`StabilizerCode`, the code a circuit
  generated, for erasure analysis after the run.
* :mod:`~sparsegf2.analysis`: estimators (survival, erasure failure
  probability Phi(e), tolerance e_delta, means with errors).
* :class:`PauliString`, :class:`Clifford` and :func:`gate`: Pauli operators,
  small Cliffords with stable indices, and the gate library under Stim's names.
"""

from sparsegf2 import analysis, codes, graphs, mipt, observables
from sparsegf2._version import __version__
from sparsegf2.circuit import (
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
from sparsegf2.clifford import (
    Clifford,
    num_cliffords,
    random_clifford_indices,
    symplectic_group,
)
from sparsegf2.codes import StabilizerCode
from sparsegf2.errors import PhaseFreeError
from sparsegf2.gates import GateInfo, all_gates, gate
from sparsegf2.pauli import PauliString
from sparsegf2.result import RunResult, SampleResult
from sparsegf2.sampling import sample
from sparsegf2.simulator import Simulator

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
    "StabilizerCode",
    "Target",
    "__version__",
    "all_gates",
    "analysis",
    "codes",
    "gate",
    "graphs",
    "mipt",
    "num_cliffords",
    "observables",
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
