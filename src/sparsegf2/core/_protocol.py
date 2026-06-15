"""Structural typing for the simulator surface.

Defines :class:`SimulatorProtocol`, the minimal attribute/method set
that :mod:`sparsegf2.core.observables` (and any future alternative
simulator) needs to consume.

Why a Protocol and not the concrete `SparseGF2` class? So that:

* an alternative simulator backend (e.g. a `DenseGF2`, a
  `BatchedSparseGF2`, a Stim-backed shim for cross-checks) can satisfy
  the observable functions' contract without inheriting from
  :class:`SparseGF2`;
* the observable functions document their actual coupling surface
  (just ``n``, ``to_symplectic()``, and ``supp_len``) rather than
  importing the concrete class only for type annotations;
* static analyzers can verify alternative simulators satisfy the
  protocol structurally.

The protocol is intentionally **minimal**. Anything observables need
from a simulator instance must be listed here; the rest of the
``SparseGF2`` API is implementation detail.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class SimulatorProtocol(Protocol):
    """Minimal surface a stabilizer simulator must expose to be queried by observables.

    Implementing classes must provide:

    * ``n``: number of qubits in the system.
    * ``supp_len``: a length-``2n`` integer array; ``supp_len[r]`` is
      the support weight (number of non-identity Paulis) of generator
      row ``r``. Observables index into **both halves**: rows ``0..n-1``
      are destabilizers, rows ``n..2n-1`` are stabilizers. Backends
      must populate the full array (``generator_weights`` returns it
      verbatim; ``stabilizer_weight_spectrum`` and
      ``average_stabilizer_weight`` only touch the stab half).
    * ``to_symplectic()``: return the full ``(2n, 2n)`` ``[X | Z]``
      symplectic matrix as a fresh ``uint8`` ndarray.

    The :class:`sparsegf2.SparseGF2` class implements this protocol
    automatically (it's structural, not inheritance-based). Verify with
    ``isinstance(sim, SimulatorProtocol)`` at runtime.

    .. note::
       ``@runtime_checkable`` verifies that the named attributes/methods
       are **present**, not that their types match. A class with
       ``n = "wrong"`` will still pass ``isinstance(..., SimulatorProtocol)``.
       For actual signature/dtype enforcement, rely on a static checker
       (mypy/pyright) at development time, or test by calling the
       protocol methods directly.
    """

    @property
    def n(self) -> int: ...

    @property
    def supp_len(self) -> NDArray[np.integer]: ...

    def to_symplectic(self) -> NDArray[np.uint8]: ...


@runtime_checkable
class SubsystemFastExtractor(Protocol):
    """Optional protocol for backends that can extract a subsystem's
    stabilizer-restricted ``[X | Z]`` block without materializing the
    full ``(2n, 2n)`` symplectic matrix.

    A backend signals support by providing
    :meth:`_subsystem_xz_block(A) -> NDArray[uint8]` returning a
    ``(n, 2 * len(A))`` GF(2) matrix whose left half is the X-bits and
    right half is the Z-bits of stabilizer rows on qubits ``A``.

    :func:`sparsegf2.subsystem_rank` checks for this protocol via
    ``isinstance(sim, SubsystemFastExtractor)`` and uses it as the fast
    path when present; backends that only satisfy the base
    :class:`SimulatorProtocol` go through ``to_symplectic()``.

    This is the **only** sanctioned way for an alternative backend to
    speed up observables; duck-typing on attribute names like ``plt``
    is unsafe because semantics can collide.
    """

    @property
    def n(self) -> int: ...

    def _subsystem_xz_block(self, A: NDArray[np.int64]) -> NDArray[np.uint8]: ...
