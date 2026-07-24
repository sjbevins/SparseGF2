"""Random-encoding convenience builder for standalone expurgation runs.

The expurgation machinery does not care where its tableau came from
(any :class:`sparsegf2.SparseGF2` plus a role labeling works). This
module supplies the one construction the source paper uses everywhere,
so the package is usable without the circuits layer: a depth-``d``
random Clifford circuit on ``n`` qubits applied to
:math:`|0^n\\rangle`, with the logical inputs on a chosen set of
qubits.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from sparsegf2.core.sparse_tableau import SparseGF2
from sparsegf2.core.symplectic import random_symplectic_2q
from sparsegf2.errors import InvalidArgumentError
from sparsegf2.expurgation.roles import StabilizerCode

#: Accepted ``geometry`` values for :func:`random_encoding`.
GEOMETRIES = ("brickwork", "all_to_all")


def random_encoding(
    n: int,
    data_qubits: Iterable[int],
    depth: int,
    *,
    geometry: str = "brickwork",
    rng: np.random.Generator | None = None,
    **sim_kwargs,
) -> StabilizerCode:
    """Encode a random ``[[n, k]]`` code and return its code view.

    Applies ``depth`` layers of uniform random ``Sp(4, F_2)`` two-qubit
    Cliffords to a fresh ``SparseGF2(n)``:

    * ``geometry="brickwork"``: one-dimensional open chain, gates on
      ``(0, 1), (2, 3), ...`` in even layers and ``(1, 2), (3, 4), ...``
      in odd layers;
    * ``geometry="all_to_all"``: a fresh uniform random perfect
      matching of the qubits each layer (one qubit idles when ``n`` is
      odd).

    Parameters
    ----------
    n
        Number of physical qubits.
    data_qubits
        The logical input positions (the set ``K``); the remaining
        qubits are the frozen :math:`|0\\rangle` inputs that become
        checks.
    depth
        Number of gate layers. ``0`` returns the trivial encoding.
    geometry
        ``"brickwork"`` or ``"all_to_all"``.
    rng
        Generator for the random gates (and matchings). ``None`` gives
        a fresh nondeterministic generator.
    **sim_kwargs
        Forwarded to :class:`sparsegf2.SparseGF2` (e.g. ``use_numba``,
        ``pivot_mode``).

    Returns
    -------
    StabilizerCode
        The encoded code view: pairs on ``data_qubits`` are logical,
        the rest are checks.
    """
    if depth < 0:
        raise InvalidArgumentError(f"depth must be non-negative, got {depth}")
    if geometry not in GEOMETRIES:
        raise InvalidArgumentError(f"geometry={geometry!r} not in {GEOMETRIES}")
    if rng is None:
        rng = np.random.default_rng()
    sim = SparseGF2(n, **sim_kwargs)
    for layer in range(depth):
        if geometry == "brickwork":
            start = layer % 2
            pairs = [(i, i + 1) for i in range(start, n - 1, 2)]
        else:
            perm = rng.permutation(n)
            pairs = [(int(perm[2 * i]), int(perm[2 * i + 1])) for i in range(n // 2)]
        for a, b in pairs:
            sim.apply_gate_2q(a, b, random_symplectic_2q(rng))
    return StabilizerCode.from_encoding(sim, data_qubits)
