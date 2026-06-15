"""Exception hierarchy for the SparseGF2 simulator.

Every exception raised from ``sparsegf2.*`` runtime code inherits from
:class:`SparseGF2Error`, so downstream callers (notably the circuits
package) can do::

    try:
        ...
    except sparsegf2.SparseGF2Error:
        ...

and catch every package-internal error without falsely matching against
unrelated ``ValueError`` / ``RuntimeError`` instances raised by, say,
numpy or the user's own code.

Each concrete error also multi-inherits from the standard exception that
makes sense semantically (``ValueError`` for bad inputs,
``RuntimeError`` for invariant violations) so existing ``except
ValueError`` callers continue to work.
"""

from __future__ import annotations


class SparseGF2Error(Exception):
    """Base for every exception raised by :mod:`sparsegf2`."""


class InvalidArgumentError(SparseGF2Error, ValueError):
    """A public method received an argument that violates its contract.

    Examples: a negative qubit count, a subsystem with duplicate
    qubits, ``A`` and ``B`` overlapping in a mutual-information call, or an
    unknown ``pivot_mode``.
    """


class TableauCorruption(SparseGF2Error, RuntimeError):
    """A four-data-structure invariant of the sparse tableau was broken.

    This is an *internal* error: it means the simulator's bookkeeping
    arrays (``plt``, ``supp_q``, ``inv``, ``inv_x``, and their position
    maps) disagree about which generator touches which qubit. A user
    shouldn't be able to trigger this; if it fires, file an issue.
    """


class SymplecticConditionError(SparseGF2Error, ValueError):
    """An input matrix failed the symplectic condition
    :math:`S\\,\\Omega\\,S^T \\equiv \\Omega \\pmod 2`.

    Raised when a caller passes a non-symplectic matrix to
    :meth:`SparseGF2.apply_gate_1q` or :meth:`apply_gate_2q`. The
    canonical fix is to sample from the native ``Sp(2n, F_2)`` sampler
    in :mod:`sparsegf2.core.symplectic` instead of building the matrix
    by hand.
    """


class SimulatorBackendError(SparseGF2Error, RuntimeError):
    """A backend-level invariant of the simulator was broken.

    Examples: a Numba kernel signature mismatch at runtime, a sampler
    producing a non-symplectic output, an LUT cache miss with a
    malformed cache key. These should never happen in shipped code.
    """


__all__ = [
    "SparseGF2Error",
    "InvalidArgumentError",
    "TableauCorruption",
    "SymplecticConditionError",
    "SimulatorBackendError",
]
