"""Cached two-qubit Clifford (symplectic) table for the circuits package.

The circuits runner applies a *random* two-qubit Clifford on every gate
pair. "Random Clifford" here means a uniformly-random element of the
**symplectic group** :math:`\\mathrm{Sp}(4, \\mathbb{F}_2)`. The phase-free
simulator only tracks the symplectic part, so the 11,520 sign-decorated
Clifford operators collapse to their 720 symplectic representatives, and
sampling uniformly over the 720 is exactly what we want.

The table is built natively, as a one-line wrapper around the core's
:func:`sparsegf2.core.symplectic.enumerate_sp4`, which constructs all 720
matrices (nested symplectic-basis construction, no Stim) and caches them
module-side with double-checked locking.

So this module adds **no** new computation or caching of its own; it
exists only to give the circuits package a stable, intention-revealing
name (``sp4_table``) and a single import site for the table. If a future
need arises (e.g. an integer-indexed Koenig-Smolin enumeration, or an
``n``-qubit table), it lands behind this same function.
"""

from __future__ import annotations

import numpy as np

from sparsegf2.core.symplectic import enumerate_sp4

# Number of symplectic representatives of the 2-qubit Clifford group.
# |Sp(4, F_2)| = (2^4 - 1) * 2^3 * (2^2 - 1) * 2^1 = 15 * 8 * 3 * 2 = 720.
# (The full sign-decorated Clifford group has 11,520 = 720 * 16 elements;
#  the factor 16 is the four sign choices per qubit, invisible to the
#  phase-free simulator.)
SP4_SIZE: int = 720


def sp4_table() -> np.ndarray:
    """Return the cached ``(720, 4, 4)`` uint8 table of ``Sp(4, F_2)``.

    Each ``table[k]`` is a 4×4 GF(2) symplectic matrix in the
    ``(X_qi, X_qj, Z_qi, Z_qj)`` basis, ready to hand straight to
    :meth:`sparsegf2.SparseGF2.apply_gate_2q`. The array is the core's
    cached, **read-only** view. Do not mutate it; copy first if you must.

    The first call anywhere in the process materializes the table (a few
    milliseconds); every subsequent call is a lock-free dictionary-free
    return of the same array, so a runner can call this once per instance
    without amortization concerns.

    Returns
    -------
    numpy.ndarray
        Shape ``(720, 4, 4)``, dtype ``uint8``, read-only.
    """
    return enumerate_sp4()


__all__ = ["SP4_SIZE", "sp4_table"]
