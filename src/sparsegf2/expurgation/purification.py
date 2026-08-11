"""Bridge from the purification picture to the encoding presentation.

A monitored-circuit run in the purification picture leaves a pure
stabilizer state on system plus reference qubits. The code it encodes
lives on the system alone: the checks are the state's stabilizers
supported entirely on the system, and the ``k = S(system)`` logical
qubits are the degrees of freedom still entangled with the reference
(Gullans and Huse, PRX **10**, 041020 (2020)). :func:`from_purification`
extracts that code into the presentation the expurgation machinery
uses: a fresh ``n_system``-qubit tableau whose pairs are labeled check
or logical.

The extraction is GF(2) linear algebra. The stabilizer combinations
with no reference support are the kernel of the reference-column map;
a symplectic Gram-Schmidt completes those checks with destabilizers
and hyperbolic logical pairs into a full canonical basis, which loads
through :meth:`SparseGF2.from_symplectic`. Any completion gives valid
logical representatives, and the quantities expurgation consumes
(syndromes, ``r_M``, recovery, distance) do not depend on that choice.

Measuring an extracted candidate back on the *original* purification
tableau (with the qubit indices mapped through ``system_qubits``) is
the same operation: it lowers ``code_dimension`` by exactly one, which
is the targeted-purification identity the source paper points out.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from sparsegf2.core.linalg_gf2 import gf2_kernel_basis
from sparsegf2.core.observables import entanglement_entropy
from sparsegf2.core.sparse_tableau import SparseGF2
from sparsegf2.errors import InvalidArgumentError, SimulatorBackendError
from sparsegf2.expurgation.roles import (
    ROLE_CHECK,
    ROLE_LOGICAL,
    StabilizerCode,
    _exact_integer_array,
)


def _sp(u: np.ndarray, v: np.ndarray, n: int) -> int:
    """Symplectic product of two ``[x | z]`` rows (int accumulators)."""
    return (
        int(
            u[:n].astype(np.int64) @ v[n:].astype(np.int64)
            + u[n:].astype(np.int64) @ v[:n].astype(np.int64)
        )
        & 1
    )


def _symplectic_gram_schmidt(checks: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Complete independent commuting rows into a full canonical basis.

    Returns ``(destabs, stabs)``, each ``(n, 2n)``: stab row ``i`` and
    destab row ``i`` are a hyperbolic pair, distinct pairs commute, and
    the first ``checks.shape[0]`` stab rows span exactly the input
    space (each is the input row plus possibly earlier input rows).
    """
    queue = [checks[i].copy() for i in range(checks.shape[0])]
    pool = [row.copy() for row in np.eye(2 * n, dtype=np.uint8)]
    stabs: list[np.ndarray] = []
    destabs: list[np.ndarray] = []
    while len(stabs) < n:
        if queue:
            v = queue.pop(0)
        else:
            while pool and not pool[0].any():
                pool.pop(0)
            if not pool:
                raise SimulatorBackendError(
                    "symplectic completion exhausted its pool; the check rows "
                    "were not independent and commuting"
                )
            v = pool.pop(0)
            if all(_sp(v, u, n) == 0 for u in pool):
                # v lies in the span of the pairs already emitted; discard.
                continue
        partner = None
        for j, u in enumerate(pool):
            if _sp(v, u, n) == 1:
                partner = pool.pop(j)
                break
        if partner is None:
            raise SimulatorBackendError(
                "no symplectic partner found; the check rows were not independent and commuting"
            )
        for u in queue + pool:
            coeff_v = _sp(u, partner, n)
            coeff_w = _sp(u, v, n)
            if coeff_v:
                u ^= v
            if coeff_w:
                u ^= partner
        stabs.append(v)
        destabs.append(partner)
    return np.stack(destabs), np.stack(stabs)


def from_purification(
    sim: SparseGF2, system_qubits: Iterable[int] | None = None, **sim_kwargs
) -> StabilizerCode:
    """Extract the system code of a purification tableau for expurgation.

    Parameters
    ----------
    sim
        A :class:`SparseGF2` holding a purification-picture state
        (for example a monitored circuit run on
        :func:`sparsegf2.from_bell_purification`). The instance is only
        read; the returned code gets a fresh simulator.
    system_qubits
        The system qubit indices. ``None`` (allowed only for even
        ``sim.n``) means the standard convention: the first half is the
        system, the second half the reference. Any subset works, so the
        single-reference picture and permuted layouts are covered.
    **sim_kwargs
        Forwarded to the extracted code's ``SparseGF2`` (e.g.
        ``use_numba``, ``pivot_mode``).

    Returns
    -------
    StabilizerCode
        An ``[[n_system, k]]`` code view with ``k = S(system)``:
        check pairs first (the system-supported stabilizers), then the
        logical pairs. Code qubit ``i`` corresponds to original qubit
        ``sorted(system_qubits)[i]``; map candidate supports through
        that list to act on the original tableau.
    """
    n_tot = sim.n
    if system_qubits is None:
        if n_tot % 2:
            raise InvalidArgumentError(
                "system_qubits is required when sim.n is odd (no default half split)"
            )
        system = np.arange(n_tot // 2, dtype=np.int64)
    else:
        system = _exact_integer_array(system_qubits, name="system_qubits")
        if system.size == 0:
            raise InvalidArgumentError("system_qubits must be non-empty")
        if system.min() < 0 or system.max() >= n_tot:
            raise InvalidArgumentError(
                f"system_qubits must be in [0, n={n_tot}); got "
                f"min={system.min()}, max={system.max()}"
            )
        if np.unique(system).shape[0] != system.shape[0]:
            raise InvalidArgumentError(f"system_qubits must not repeat: got {system.tolist()}")
        system = np.sort(system)
    reference = np.setdiff1d(np.arange(n_tot, dtype=np.int64), system, assume_unique=True)

    stab_block = sim.to_symplectic()[n_tot:].astype(np.uint8)
    ref_cols = np.concatenate([reference, reference + n_tot])
    # Stabilizer combinations with zero reference support: the kernel of
    # the reference-column map c -> c @ G_ref.
    kernel = gf2_kernel_basis(stab_block[:, ref_cols].T)
    sys_supported = (kernel.astype(np.int64) @ stab_block.astype(np.int64)) % 2
    sys_cols = np.concatenate([system, system + n_tot])
    other_cols = np.setdiff1d(np.arange(2 * n_tot), sys_cols, assume_unique=True)
    if sys_supported[:, other_cols].any():
        raise SimulatorBackendError("kernel combination has reference support; this is a bug")
    checks = sys_supported[:, sys_cols].astype(np.uint8)

    n_sys = system.shape[0]
    n_s = checks.shape[0]
    k = n_sys - n_s
    if k != entanglement_entropy(sim, system):
        raise SimulatorBackendError(
            f"extracted k={k} disagrees with S(system)="
            f"{entanglement_entropy(sim, system)}; this is a bug"
        )
    destabs, stabs = _symplectic_gram_schmidt(checks, n_sys)
    code_sim = SparseGF2.from_symplectic(np.concatenate([destabs, stabs], axis=0), **sim_kwargs)
    roles = np.full(n_sys, ROLE_LOGICAL, dtype=np.uint8)
    roles[:n_s] = ROLE_CHECK
    return StabilizerCode(code_sim, roles)
