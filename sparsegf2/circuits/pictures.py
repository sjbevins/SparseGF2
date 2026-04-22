"""
Physics-picture initialization for graph-defined circuits.

MVP supports only the ``purification`` picture: 2n qubits (n system + n
reference), initialized as n Bell pairs; the circuit acts only on the system
qubits. ``SparseGF2(n)`` already lands in this state via its constructor's
``_init_bell_pairs`` pass, so the picture initializer here is a thin wrapper
that exists mainly to give future pictures (``single_ref``, ``pure_state``)
a uniform entry point.

Future pictures
---------------
When added, each picture will be a function ``init_<name>(n, **kwargs)``
returning an initialized :class:`SparseGF2`-compatible simulator, plus an
entry in :data:`PICTURES` so the validator and CLI see the new option.
"""
from __future__ import annotations

from typing import Tuple

from sparsegf2 import SparseGF2


PICTURES: Tuple[str, ...] = ("purification",)


def init_picture(
    picture: str,
    n: int,
    *,
    use_min_weight_pivot: bool = True,
    check_inputs: bool = False,
    hybrid_mode: bool = True,
) -> SparseGF2:
    """Initialize a stabilizer simulator in the named physics picture.

    Parameters
    ----------
    picture : str
        One of :data:`PICTURES`. MVP supports only ``"purification"``.
    n : int
        Number of system qubits.
    use_min_weight_pivot, check_inputs, hybrid_mode
        Forwarded to :class:`SparseGF2`. Defaults match the production
        settings used by ``graph-construction/runner.py``.

    Returns
    -------
    SparseGF2
        A simulator whose state is the initial state of the requested
        picture. For purification this is n Bell pairs between system
        qubits ``0..n-1`` and reference qubits ``n..2n-1``.
    """
    if picture != "purification":
        raise ValueError(
            f"picture must be one of {PICTURES}; got {picture!r}. "
            "Other pictures (single_ref, pure_state) are deferred post-MVP."
        )
    return SparseGF2(
        n,
        use_min_weight_pivot=use_min_weight_pivot,
        check_inputs=check_inputs,
        hybrid_mode=hybrid_mode,
    )


__all__ = ["PICTURES", "init_picture"]
