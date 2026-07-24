"""SparseGF2: phase-free sparse stabilizer simulator over GF(2).

A self-contained, Stim-free phase-free stabilizer simulator that other
packages (notably the ``circuits`` MIPT runner) build on top of.

What you get
============

* :class:`SparseGF2`: the simulator core. ``SparseGF2(n)`` is a pure
  ``|0^n⟩`` state on exactly ``n`` qubits (no implicit purification).

* **Gate application**: :meth:`SparseGF2.apply_h`, :meth:`apply_s`,
  :meth:`apply_sqrt_x`, :meth:`apply_cx`, :meth:`apply_cz`,
  :meth:`apply_swap`, plus :meth:`apply_gate_1q` / :meth:`apply_gate_2q`
  for arbitrary user-supplied symplectic matrices.

* **Measurement**: :meth:`SparseGF2.measure_z`, :meth:`measure_x`,
  :meth:`measure_y` + the determinism predicate
  :meth:`is_deterministic_z`.

* **Reset / projection**: :meth:`SparseGF2.reset_z`, :meth:`reset_x`,
  :meth:`reset_y`.

* **State extraction**: :meth:`SparseGF2.to_symplectic`
  (the raw ``(2n, 2n)`` ``[X | Z]`` matrix) and
  :meth:`SparseGF2.canonical_form` (GF(2) RREF of the stabilizer
  block, the canonical representative for state-equality testing).

* **Factories**: :func:`from_zero_state`, :func:`from_bell_purification`.

* **Native ``Sp(2n, F_2)`` machinery**, in
  :mod:`sparsegf2.core.symplectic`: :func:`enumerate_symplectic_group`,
  :func:`random_symplectic_2q`, :func:`random_symplectic` for arbitrary
  ``n`` (no runtime Stim dependency).

* **Observables**, in :mod:`sparsegf2.core.observables`:
  :func:`subsystem_rank`, :func:`entanglement_entropy`,
  :func:`mutual_information`, :func:`tripartite_mutual_info`,
  :func:`code_dimension`, :func:`generator_weights`,
  :func:`stabilizer_weight_spectrum`,
  :func:`average_stabilizer_weight`.

* **Named gate constants**: :data:`H_SYMP`, :data:`S_SYMP`,
  :data:`SQRT_X_SYMP`, :data:`CX_SYMP`, :data:`CZ_SYMP`, :data:`SWAP_SYMP`,
  :data:`PAULI_I/X/Y/Z`.

Subpackages built on the core
=============================

* :mod:`sparsegf2.circuits`: graph-defined random-Clifford + measurement
  circuits (the MIPT workhorse): ``simulate(CircuitConfig(...))``, pictures,
  graphs (incl. ``from_networkx`` for arbitrary geometry), gating / matching /
  measurement modes, plus text and visual circuit inspection.
* :mod:`sparsegf2.analysis`: named observables, online/offline analysis of
  final tableaux, and the parameter-:func:`~sparsegf2.analysis.sweep` driver
  with on-disk (parquet / HDF5) output.
* :mod:`sparsegf2.expurgation`: the Gullans et al. expurgation algorithm
  (targeted measurement of uncorrectable error operators) run natively on
  the tableau, with role bookkeeping, exact erasure recovery, and the loop
  driver.

Contracts
=========

* **Stim parity** at the level of the **stabilizer subspace**
  (GF(2) RREF of rows ``n..2n-1`` of ``[X | Z]``).
  ``SparseGF2.canonical_form()`` returns this RREF.
* **Phase-free.** Sign bits are not tracked; methods like
  :meth:`measure_z` return ``0`` for every deterministic outcome.
* **No runtime Stim import.** Stim is a test-time cross-checker only;
  no path under ``src/`` ever calls ``import stim``.
"""

import logging as _logging

from sparsegf2._version import __version__
from sparsegf2.core.observables import (
    Subsystem,
    average_stabilizer_weight,
    code_dimension,
    code_rate,
    complement,
    contiguous_distance,
    entanglement_entropy,
    generator_weights,
    mutual_information,
    stabilizer_weight_spectrum,
    subsystem_rank,
    tripartite_mutual_info,
)
from sparsegf2.core.sparse_tableau import (
    CX_SYMP,
    CZ_SYMP,
    H_SYMP,
    PAULI_I,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    S_SYMP,
    SQRT_X_SYMP,
    SWAP_SYMP,
    SparseGF2,
    from_bell_purification,
    from_zero_state,
)
from sparsegf2.core.symplectic import (
    enumerate_sp4,
    enumerate_symplectic_group,
    is_symplectic,
    random_symplectic,
    random_symplectic_2q,
    random_symplectic_2q_batch,
    random_symplectic_batch,
    symplectic_form,
    symplectic_group_order,
    symplectic_product,
)
from sparsegf2.errors import (
    InvalidArgumentError,
    SimulatorBackendError,
    SparseGF2Error,
    SymplecticConditionError,
    TableauCorruption,
)

__all__ = [
    # Core class
    "SparseGF2",
    # Factories
    "from_bell_purification",
    "from_zero_state",
    # Named symplectic constants
    "CX_SYMP",
    "CZ_SYMP",
    "H_SYMP",
    "PAULI_I",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "S_SYMP",
    "SQRT_X_SYMP",
    "SWAP_SYMP",
    # Sp(2n, F_2) machinery
    "enumerate_sp4",
    "enumerate_symplectic_group",
    "is_symplectic",
    "random_symplectic",
    "random_symplectic_2q",
    "random_symplectic_2q_batch",
    "random_symplectic_batch",
    "symplectic_form",
    "symplectic_group_order",
    "symplectic_product",
    # Observables
    "Subsystem",
    "average_stabilizer_weight",
    "code_dimension",
    "code_rate",
    "complement",
    "contiguous_distance",
    "entanglement_entropy",
    "generator_weights",
    "mutual_information",
    "stabilizer_weight_spectrum",
    "subsystem_rank",
    "tripartite_mutual_info",
    # Exception hierarchy
    "InvalidArgumentError",
    "SimulatorBackendError",
    "SparseGF2Error",
    "SymplecticConditionError",
    "TableauCorruption",
    "__version__",
]

# Library logger. SparseGF2 itself never logs at runtime (no INFO/DEBUG
# spam in hot paths), but downstream packages (notably the circuits
# package) can attach handlers to this parent to route diagnostic
# messages through a known namespace. ``NullHandler`` suppresses the
# "No handlers could be found" warning that older logging APIs would
# emit when nothing is attached.
_logger = _logging.getLogger("sparsegf2")
_logger.addHandler(_logging.NullHandler())
