"""Core simulator: tableau data structures, Clifford kernels, observables.

This subpackage exposes a **superset** of the top-level
:mod:`sparsegf2` public API. Everything in :mod:`sparsegf2` is re-exported
here verbatim:

    >>> from sparsegf2.core import SparseGF2, entanglement_entropy
    >>> # ... same object as:
    >>> from sparsegf2 import SparseGF2, entanglement_entropy

In addition, :mod:`sparsegf2.core` exports a small set of
**extensibility / advanced** symbols that aren't part of the top-level
API but that alternative-backend authors or low-level callers may want:

* :class:`SimulatorProtocol`: structural protocol observables couple to;
  satisfy this in an alternative simulator backend.
* :func:`gf2_rref`, :func:`gf2_rank`, :func:`gf2_kernel_basis`: the
  GF(2) linear-algebra primitives shared between observables and the
  symplectic sampler.

These are intentionally not in the top-level :data:`sparsegf2.__all__`
to keep the canonical user-facing surface minimal. Treat them as
semi-public: stable across minor versions, but not the place we'd
prioritize documentation polish.
"""

from sparsegf2.core._protocol import SimulatorProtocol
from sparsegf2.core.linalg_gf2 import gf2_kernel_basis, gf2_rank, gf2_rref
from sparsegf2.core.observables import (
    average_stabilizer_weight,
    code_dimension,
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

__all__ = [
    # Simulator
    "SparseGF2",
    "SimulatorProtocol",
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
    # Sp(2n, F_2)
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
    "average_stabilizer_weight",
    "code_dimension",
    "entanglement_entropy",
    "generator_weights",
    "mutual_information",
    "stabilizer_weight_spectrum",
    "subsystem_rank",
    "tripartite_mutual_info",
    # Linear algebra primitives (advanced)
    "gf2_kernel_basis",
    "gf2_rank",
    "gf2_rref",
]
