"""SparseGF2 v3: a fast stabilizer-circuit simulator for monitored Clifford circuits.

Verified against Stim. The main pieces:

* :class:`Simulator`, :class:`Circuit` and :func:`sample`: single circuits in
  the Stim circuit language, with or without signs (``phases``).
* :mod:`~sparsegf2_v3.observables`: entanglement, code and erasure quantities
  of a simulator's state.
* :mod:`~sparsegf2_v3.mipt`: ensembles of random monitored circuits at full
  speed (graph collections, reference pictures, probes on their own
  checkpoints), and :mod:`~sparsegf2_v3.graphs` for the interaction graphs.
* :mod:`~sparsegf2_v3.codes`: :class:`StabilizerCode`, the code a circuit
  generated, for erasure analysis after the run.
* :mod:`~sparsegf2_v3.analysis`: estimators (survival, erasure failure
  probability Phi(e), tolerance e_delta, means with errors).
* :class:`PauliString`, :class:`Clifford` and :func:`gate`: Pauli operators,
  small Cliffords with stable indices, and the gate library under Stim's names.
"""

from sparsegf2_v3 import analysis, codes, graphs, mipt, observables
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
from sparsegf2_v3.codes import StabilizerCode
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
