"""Entanglement, code and erasure-protection observables of a stabilizer state.

Every function takes a :class:`~sparsegf2_v3.Simulator` (either mode: all of
these depend only on the unsigned stabilizer group) and explicit lists of
qubits. Entropies are in bits and are exact integers.

Pictures, in plain terms
------------------------
Code properties of monitored circuits are measured by entangling the system
with reference qubits that never interact again (Gullans and Huse, PRX 10,
041020). With system qubits ``S`` and reference qubits ``R`` in a pure joint
state:

* the **code dimension** (number of logical qubits still protected) is
  ``k = S(R)``;
* erasing a set ``E`` of system qubits destroys ``I(R:E)`` bits of the
  reference's correlations: the erasure is **correctable** iff ``I(R:E) = 0``,
  and a decoder that knows ``E`` recovers the logical state with probability
  ``2**-I(R:E)`` (each lost bit randomizes one logical Pauli);
* the **contiguous code distance** is the length of the shortest contiguous
  run of system qubits whose erasure is not correctable (Li and Fisher,
  PRB 103, 104306).

The functions never assume a picture: you pass which qubits are the system and
which are the reference. :mod:`sparsegf2_v3.mipt` builds the standard setups.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from sparsegf2_v3._kernel import entropy as _e
from sparsegf2_v3._kernel.rng import stream_key
from sparsegf2_v3.simulator import Simulator

__all__ = [
    "code_dimension",
    "contiguous_code_distance",
    "entropy",
    "entropy_profile",
    "erasure_is_correctable",
    "lost_logical_information",
    "mutual_information",
    "random_erasure_statistics",
    "recovery_probability",
    "stabilizer_weights",
    "tripartite_information",
]


def _qubits(sim: Simulator, qubits: Iterable[int], what: str = "qubits") -> np.ndarray:
    arr = np.fromiter((int(q) for q in qubits), dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() >= sim.num_qubits):
        raise IndexError(f"{what} out of range for a {sim.num_qubits}-qubit simulator")
    if np.unique(arr).size != arr.size:
        raise ValueError(f"{what} contain duplicates: {arr.tolist()}")
    return arr


def _work(sim: Simulator):
    n = sim.num_qubits
    ws = (n + 63) // 64
    return (
        np.zeros(n, dtype=np.bool_),
        np.zeros(n, dtype=np.int64),
        np.zeros((2 * (n // 2 + 1), ws), dtype=np.uint64),
    )


def entropy(sim: Simulator, region: Iterable[int]) -> int:
    """Entanglement entropy S(A) in bits (same as :meth:`Simulator.entropy`)."""
    return sim.entropy(region)


def mutual_information(sim: Simulator, a: Iterable[int], b: Iterable[int]) -> int:
    """I(A:B) = S(A) + S(B) - S(AB) for disjoint regions A and B."""
    a_arr, b_arr = _qubits(sim, a, "region A"), _qubits(sim, b, "region B")
    if np.intersect1d(a_arr, b_arr).size:
        raise ValueError("regions A and B must be disjoint")
    return sim.entropy(a_arr) + sim.entropy(b_arr) - sim.entropy(np.concatenate([a_arr, b_arr]))


def tripartite_information(
    sim: Simulator, a: Iterable[int], b: Iterable[int], c: Iterable[int]
) -> int:
    """I3(A:B:C) = S(A)+S(B)+S(C) - S(AB) - S(AC) - S(BC) + S(ABC), for disjoint regions.

    Negative and of order one at a measurement-induced transition; widely used
    to locate the critical point (Zabalo et al., PRB 101, 060301).
    """
    regions = [_qubits(sim, r, name) for r, name in ((a, "A"), (b, "B"), (c, "C"))]
    if sum(r.size for r in regions) != np.unique(np.concatenate(regions)).size:
        raise ValueError("regions A, B and C must be disjoint")
    ra, rb, rc = regions
    s = sim.entropy
    return (
        s(ra) + s(rb) + s(rc)
        - s(np.concatenate([ra, rb])) - s(np.concatenate([ra, rc])) - s(np.concatenate([rb, rc]))
        + s(np.concatenate([ra, rb, rc]))
    )  # fmt: skip


def entropy_profile(sim: Simulator, order: Iterable[int] | None = None) -> np.ndarray:
    """Entropies of all prefixes of an ordering: ``out[k] = S(order[:k])``.

    With the default order ``0, 1, ..., n-1`` this is the entanglement entropy
    of every bipartition of an open chain (``out`` has length n + 1). Computed
    incrementally in one pass, ``O(n * rank * n / 64)``.
    """
    arr = (
        np.arange(sim.num_qubits, dtype=np.int64) if order is None else _qubits(sim, order, "order")
    )
    tab, n, ws = sim._state()
    out = np.zeros(arr.size + 1, dtype=np.int64)
    _e.prefix_entropies(tab, n, ws, arr, out)
    return out


def code_dimension(sim: Simulator, reference: Iterable[int]) -> int:
    """Number of logical qubits held by the system: k = S(R) (reference picture)."""
    return sim.entropy(_qubits(sim, reference, "reference"))


def lost_logical_information(
    sim: Simulator, erased: Iterable[int], reference: Iterable[int]
) -> int:
    """I(R:E): bits of logical information destroyed by erasing the qubits ``erased``.

    0 means the erasure is correctable; the maximum is ``2 * code_dimension``.
    """
    e_arr = _qubits(sim, erased, "erased qubits")
    r_arr = _qubits(sim, reference, "reference")
    if np.intersect1d(e_arr, r_arr).size:
        raise ValueError("erased qubits must not include reference qubits")
    tab, n, ws = sim._state()
    mask_buf, comp_buf, scratch = _work(sim)
    union = np.zeros(n, dtype=np.int64)
    return int(
        _e.mutual_information_with(tab, n, ws, r_arr, e_arr, mask_buf, comp_buf, scratch, union)
    )


def erasure_is_correctable(sim: Simulator, erased: Iterable[int], reference: Iterable[int]) -> bool:
    """True iff erasing ``erased`` loses no logical information (I(R:E) = 0)."""
    return lost_logical_information(sim, erased, reference) == 0


def recovery_probability(sim: Simulator, erased: Iterable[int], reference: Iterable[int]) -> float:
    """Probability that an optimal decoder, told which qubits were erased, recovers the logical state.

    Equal to ``2**-I(R:E)`` for a stabilizer code (each lost bit randomizes one
    logical Pauli operator); 1 for a correctable erasure.
    """
    return 2.0 ** -lost_logical_information(sim, erased, reference)


def random_erasure_statistics(
    sim: Simulator,
    system: Iterable[int],
    reference: Iterable[int],
    sizes: Iterable[int],
    samples: int,
    *,
    seed=None,
) -> dict[str, np.ndarray]:
    """Erase uniformly random subsets of the system and record what is lost.

    For each erasure size ``m`` in ``sizes``, ``samples`` subsets of ``m``
    system qubits are drawn without replacement. Returns a dict of arrays
    indexed by size:

    * ``"lost"``: all I(R:E) values, shape ``(len(sizes), samples)``;
    * ``"correctable_fraction"``: fraction of erasures with I(R:E) = 0;
    * ``"mean_recovery_probability"``: mean of ``2**-I(R:E)``;
    * ``"sizes"``: the sizes.

    Sweeping ``m`` gives the erasure threshold of the dynamically generated
    code (cf. Gullans et al., PRX 11, 031066).
    """
    sys_arr = _qubits(sim, system, "system")
    r_arr = _qubits(sim, reference, "reference")
    if np.intersect1d(sys_arr, r_arr).size:
        raise ValueError("system and reference qubits must be disjoint")
    size_arr = np.fromiter((int(m) for m in sizes), dtype=np.int64)
    if size_arr.size and (size_arr.min() < 0 or size_arr.max() > sys_arr.size):
        raise ValueError(f"erasure sizes must be between 0 and {sys_arr.size}")
    if samples < 1:
        raise ValueError("samples must be positive")
    tab, n, ws = sim._state()
    lost = np.zeros((size_arr.size, int(samples)), dtype=np.int64)
    _e.random_erasure_stats(
        tab, n, ws, sys_arr, r_arr, size_arr, int(samples), stream_key(seed), lost
    )
    return {
        "sizes": size_arr,
        "lost": lost,
        "correctable_fraction": (lost == 0).mean(axis=1),
        "mean_recovery_probability": (2.0 ** -lost.astype(float)).mean(axis=1),
    }


def contiguous_code_distance(
    sim: Simulator, system: Iterable[int], reference: Iterable[int], *, periodic: bool = True
) -> int:
    """Shortest contiguous run of ``system`` (in the order given) whose erasure is not correctable.

    Runs wrap around the end when ``periodic`` (a ring). Returns 0 when the
    reference holds no logical information (code dimension 0).
    """
    sys_arr = _qubits(sim, system, "system")
    r_arr = _qubits(sim, reference, "reference")
    if np.intersect1d(sys_arr, r_arr).size:
        raise ValueError("system and reference qubits must be disjoint")
    tab, n, ws = sim._state()
    return int(_e.contiguous_distance(tab, n, ws, sys_arr, r_arr, bool(periodic)))


def stabilizer_weights(sim: Simulator) -> np.ndarray:
    """Number of qubits each stabilizer generator acts on (in the simulator's current basis).

    This depends on the choice of generators, not only on the state; use it as
    a sparsity diagnostic. ``stabilizer_weights(sim).mean()`` is the average
    generator weight.
    """
    xs, zs, _ = sim.stabilizer_arrays()
    return (xs | zs).sum(axis=1).astype(np.int64)
