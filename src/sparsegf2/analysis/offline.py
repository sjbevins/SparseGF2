"""Offline analysis: run named analyses over saved final tableaux.

The *online* path (``simulate(..., analyses=[...])``) computes analyses at the
end of each circuit and discards the tableau. The *offline* path instead saves
the compact ``(2N, 2N)`` symplectic (``simulate(..., save_tableau=True)``),
collects a batch of them, and analyzes later, here.

Both paths call the **same** resolved analyses on a :class:`SparseGF2`. The only
difference is that offline rebuilds the simulator from the saved tableau via
:meth:`SparseGF2.from_symplectic`. Because every analysis is a phase-free
function of the stabilizer subspace, the round-trip is lossless: offline values
equal online values bit-for-bit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from sparsegf2.analysis.registry import Analysis, resolve_analyses
from sparsegf2.circuits.picture import PictureSpec
from sparsegf2.circuits.records import SampleRecord
from sparsegf2.core.sparse_tableau import SparseGF2
from sparsegf2.errors import InvalidArgumentError


def reconstruct(
    source: SampleRecord | np.ndarray,
    *,
    pivot_mode: str | None = None,
    use_numba: bool | None = None,
    rng: np.random.Generator | None = None,
) -> SparseGF2:
    """Rebuild a live ``SparseGF2`` from a saved symplectic.

    ``source`` is a :class:`SampleRecord` (its ``final_tableau`` is used) or a
    raw ``(2N, 2N)`` ``[X|Z]`` array. The ``rng`` is irrelevant to analysis
    results (no analysis draws randomness, the measurements already happened),
    so it defaults to a fresh generator.

    The ``hybrid`` flag is deliberately *not* forwarded: this rebuilds a tableau
    only to run read-only analyses (no further gates/measurements), so the
    sparse/dense switching never triggers and a dense mirror would be pure
    allocation overhead.
    """
    symp = source.final_tableau if isinstance(source, SampleRecord) else source
    if symp is None:
        raise InvalidArgumentError(
            "no final_tableau to reconstruct from: run with save_tableau=True"
        )
    return SparseGF2.from_symplectic(
        np.asarray(symp, dtype=np.uint8),
        pivot_mode=pivot_mode,
        use_numba=use_numba,
        rng=rng,
    )


def analyze(
    source: SparseGF2 | SampleRecord | np.ndarray,
    picture_spec: PictureSpec,
    analyses: Sequence[str | Analysis] | Mapping[str, Analysis],
    *,
    pivot_mode: str | None = None,
    use_numba: bool | None = None,
) -> dict[str, Any]:
    """Run named/custom analyses on a single live or reconstructed state.

    ``source`` may be a live :class:`SparseGF2`, a :class:`SampleRecord` with a
    saved ``final_tableau``, or a raw ``(2N, 2N)`` array. ``picture_spec`` is the
    :class:`~sparsegf2.circuits.PictureSpec` the state was produced under.
    """
    sim = (
        source
        if isinstance(source, SparseGF2)
        else reconstruct(source, pivot_mode=pivot_mode, use_numba=use_numba)
    )
    return {name: fn(sim, picture_spec) for name, fn in resolve_analyses(analyses).items()}


def analyze_tableaux(
    tableaux: Iterable[np.ndarray],
    picture_spec: PictureSpec,
    analyses: Sequence[str | Analysis] | Mapping[str, Analysis],
    *,
    pivot_mode: str | None = None,
    use_numba: bool | None = None,
) -> list[dict[str, Any]]:
    """Run the same analyses over a collection of saved ``(2N, 2N)`` tableaux.

    Returns one ``{name: value}`` dict per tableau, in order. All tableaux are
    assumed to share ``picture_spec`` (true for one sweep cell / picture).
    """
    resolved = resolve_analyses(analyses)
    rows: list[dict[str, Any]] = []
    for symp in tableaux:
        sim = reconstruct(symp, pivot_mode=pivot_mode, use_numba=use_numba)
        rows.append({name: fn(sim, picture_spec) for name, fn in resolved.items()})
    return rows


__all__ = ["analyze", "analyze_tableaux", "reconstruct"]
