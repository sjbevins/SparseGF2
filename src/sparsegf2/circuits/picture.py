"""Physics "pictures": how the initial state and order parameter are set up.

A *picture* fixes three things before any gates run:

1. **The initial simulator state**: how many physical qubits, and in what
   stabilizer state they start.
2. **The system / reference partition**: which physical qubits are the
   "system" we drive, and which (if any) are a reference we read an order
   parameter off of.
3. **The order parameter**: the scalar the runner extracts at the end.

The runner *always* records the half-cut entanglement entropy; a picture's
``order_parameter`` field names the *additional* picture-specific scalar
(if any) it also records. Three pictures are implemented:

- ``pure_state``: ``n`` qubits in :math:`|0^n\\rangle`, no reference.
  ``order_parameter = "none"``, so only the half-cut entropy is recorded.
  This is the simplest MIPT setup and the golden path.
- ``purification``: ``2n`` qubits, one Bell pair per system qubit
  (system ``i`` ↔ reference ``i+n``), built by the core's
  :func:`sparsegf2.from_bell_purification`. ``order_parameter =
  "code_dimension"``, so *in addition to* the half-cut entropy, the runner
  records the **code dimension** ``k = S(system)``, the number of system
  qubits still entangled with the reference (the Gullans-Huse purification
  order parameter).
- ``single_ref``: ``n + 1`` qubits: the system plus **one** reference
  qubit Bell-paired with system qubit ``n-1`` (via ``H(n-1); CX(n-1, n)``).
  ``order_parameter = "ref_entropy"``, so the runner records
  ``S(reference) ∈ {0, 1}``: 1 while the reference stays entangled with the
  system, 0 once monitored dynamics purifies it. The cheapest MIPT probe
  (a single tracked bit rather than the full code dimension).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from sparsegf2 import SparseGF2
from sparsegf2.circuits.ops import Op, apply_named_gate
from sparsegf2.errors import InvalidArgumentError


class Picture(StrEnum):
    """Physics picture selector.

    A :class:`StrEnum`, so ``Picture.PURE_STATE == "pure_state"`` is True
    and the value serializes straight to JSON. ``CircuitConfig`` accepts
    either the enum member or the bare string.
    """

    PURE_STATE = "pure_state"
    PURIFICATION = "purification"
    SINGLE_REF = "single_ref"


@dataclass(frozen=True)
class PictureSpec:
    """Immutable description of a picture's qubit layout and order parameter.

    The runner reads this (never re-derives the layout) so the picture is
    the single source of truth for "which qubits are system vs reference"
    and "what scalar do I report".

    Attributes
    ----------
    picture : Picture
        Which picture this describes.
    n_system : int
        Number of system qubits (the ``n`` the user asked for).
    total_qubits : int
        Physical qubit count in the simulator (``n`` for pure_state,
        ``2n`` for purification, ``n+1`` for single_ref).
    system_qubits : ndarray[int64]
        The system qubit indices (always ``0..n_system-1``).
    reference_qubits : ndarray[int64]
        The reference qubit indices (empty for pure_state; ``n..2n-1`` for
        purification; ``[n]`` for single_ref).
    order_parameter : {"none", "code_dimension", "ref_entropy"}
        Which observable the runner extracts *in addition to* the half-cut
        entropy (always recorded). ``"code_dimension"`` → ``k = S(system)``
        (purification); ``"ref_entropy"`` → ``S(reference) ∈ {0, 1}``
        (single_ref); ``"none"`` → half-cut entropy only (pure_state).
    """

    picture: Picture
    n_system: int
    total_qubits: int
    system_qubits: NDArray[np.int64]
    reference_qubits: NDArray[np.int64]
    order_parameter: Literal["none", "code_dimension", "ref_entropy"]


def setup_picture(
    picture: Picture | str,
    n_system: int,
    *,
    rng: np.random.Generator | None = None,
    pivot_mode: str | None = None,
    use_numba: bool | None = None,
    hybrid: bool = False,
) -> tuple[SparseGF2, PictureSpec]:
    """Construct the initial simulator and its :class:`PictureSpec`.

    Parameters
    ----------
    picture : Picture or str
        The picture to set up. A bare string is coerced to :class:`Picture`.
    n_system : int
        Number of system qubits (``>= 1``; even is required by the built-in
        graphs, but that is the config's concern, not the picture's).
    rng : numpy.random.Generator, optional
        RNG handed to the :class:`~sparsegf2.SparseGF2` instance; it becomes
        the simulator's measurement-outcome stream. Pass a seeded generator
        for reproducible trajectories.
    pivot_mode : str, optional
        Forwarded to :class:`~sparsegf2.SparseGF2` (``"min_weight"`` keeps
        the tableau sparser; ``"first"`` is cheaper per measurement).
    use_numba : bool, optional
        Forwarded to :class:`~sparsegf2.SparseGF2` (JIT vs pure-Python
        kernels; ``None`` auto-detects).
    hybrid : bool, optional
        Forwarded to :class:`~sparsegf2.SparseGF2`: enable the sparse/dense
        hybrid path (much faster on volume-law / low-measurement circuits).

    Returns
    -------
    (SparseGF2, PictureSpec)
        The freshly-initialized simulator and the layout/order-parameter
        description the runner will use.

    Raises
    ------
    InvalidArgumentError
        If ``picture`` is unknown or ``n_system < 1``.
    """
    picture = _coerce_picture(picture)
    if n_system < 1:
        raise InvalidArgumentError(f"n_system must be >= 1; got {n_system}")

    total_qubits, reference_qubits, order_parameter, ops = _picture_layout(picture, n_system)
    sim = SparseGF2(
        total_qubits, rng=rng, pivot_mode=pivot_mode, use_numba=use_numba, hybrid=hybrid
    )
    # Apply the picture's declarative setup recipe (the SAME ops the inspector
    # shows). This is the single source of truth: construction and
    # visualization can never drift.
    for op in ops:
        apply_named_gate(sim, op.label, op.qubits)
    spec = PictureSpec(
        picture=picture,
        n_system=n_system,
        total_qubits=total_qubits,
        system_qubits=np.arange(n_system, dtype=np.int64),
        reference_qubits=reference_qubits,
        order_parameter=order_parameter,
    )
    return sim, spec


def _coerce_picture(picture: Picture | str) -> Picture:
    """Coerce a string/enum to :class:`Picture`, raising InvalidArgumentError."""
    try:
        return Picture(picture)
    except ValueError as exc:
        valid = [p.value for p in Picture]
        raise InvalidArgumentError(f"picture must be one of {valid}; got {picture!r}") from exc


def _picture_layout(picture: Picture, n_system: int):
    """Return ``(total_qubits, reference_qubits, order_parameter, setup_ops)``.

    The single per-picture description that both :func:`setup_picture` (which
    applies ``setup_ops``) and :func:`setup_ops` (which exposes them to the
    inspector) read from. **To add a new picture, add a branch here**, and its
    deterministic construction then renders in the inspector automatically.
    """
    if picture is Picture.PURE_STATE:
        return n_system, np.empty(0, dtype=np.int64), "none", []

    if picture is Picture.PURIFICATION:
        ops: list[Op] = []
        for i in range(n_system):
            ref = i + n_system
            ops.append(Op("gate", (i,), "H", detail={"role": "setup"}))
            ops.append(
                Op(
                    "gate",
                    (i, ref),
                    "CX",
                    detail={"role": "setup", "note": f"Bell pair: system {i} ↔ reference {ref}"},
                )
            )
        refs = np.arange(n_system, 2 * n_system, dtype=np.int64)
        return 2 * n_system, refs, "code_dimension", ops

    if picture is Picture.SINGLE_REF:
        # Bell-pair the reference (index n) with the LAST system qubit (n-1)
        # via H(n-1); CX(n-1, n), a deliberate choice to keep the reference
        # adjacent to the system boundary.
        ref = n_system
        ops = [
            Op("gate", (n_system - 1,), "H", detail={"role": "setup"}),
            Op(
                "gate",
                (n_system - 1, ref),
                "CX",
                detail={
                    "role": "setup",
                    "note": f"Bell pair: system {n_system - 1} ↔ reference {ref}",
                },
            ),
        ]
        return n_system + 1, np.array([ref], dtype=np.int64), "ref_entropy", ops

    # Unreachable: _coerce_picture rejects anything not in the enum.
    raise AssertionError(f"Unhandled picture {picture!r}")


def setup_ops(picture: Picture | str, n_system: int) -> list[Op]:
    """Return the **deterministic** construction gates for ``picture``.

    The list of :class:`~sparsegf2.circuits.ops.Op` that :func:`setup_picture`
    applies to prepare the initial state (the Bell-pair gates for
    purification / single_ref; empty for pure_state). The inspector renders
    exactly this list, so what you see is what was applied.
    """
    picture = _coerce_picture(picture)
    if n_system < 1:
        raise InvalidArgumentError(f"n_system must be >= 1; got {n_system}")
    return _picture_layout(picture, n_system)[3]


__all__ = ["Picture", "PictureSpec", "setup_picture", "setup_ops"]
