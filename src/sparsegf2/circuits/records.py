"""The per-sample output record.

:class:`SampleRecord` is what one circuit realization contributes to a
run. It is the **stable contract** the future ``sparsegf2.analysis``
package will consume, so the rule is: optional fields may be *added*
later without breaking readers, but existing fields keep their meaning.

The record has a **union shape**: every picture produces the same
dataclass, with picture-specific observables left ``None`` when they don't
apply. ``purification`` fills ``code_dimension``; ``pure_state`` leaves it
``None``; both fill ``entropy_half_cut``. A reader can therefore load a
mixed batch without branching on the picture to know the columns.

We do **not** carry bit-packed ``x``/``z``/sign arrays: the
:meth:`sparsegf2.SparseGF2.to_symplectic` ``(2N, 2N)`` matrix is the
canonical state snapshot, optionally stored in ``final_tableau``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sparsegf2.circuits.picture import Picture


@dataclass
class SampleRecord:
    """Everything one sample contributes to a run.

    Attributes
    ----------
    sample_seed : int
        The per-sample seed offset used (identity of this realization).
    picture : Picture
        Which picture produced this record.
    total_layers : int
        Number of measured layers actually executed.
    total_gates : int
        Total two-qubit gates applied.
    total_measurements : int
        Total single-qubit Z measurements applied.
    gate_to_meas_ratio_expected : float
        ``config.expected_gate_to_meas_ratio()`` for this cell.
    gate_to_meas_ratio_actual : float
        ``total_gates / total_measurements`` (``inf`` if no measurements).
    code_dimension : int or None
        ``k = S(system)``, purification picture only; ``None`` otherwise.
    ref_entropy : int or None
        ``S(reference) ∈ {0, 1}``, single_ref picture only; ``None`` otherwise.
    entropy_half_cut : int or None
        Half-cut entanglement entropy ``S(0..n/2-1)``, all pictures.
    runtime_total_s : float
        Wall-clock seconds for the run (gates + measurements + observables).
    final_tableau : ndarray[uint8] or None
        ``(2N, 2N)`` ``[X|Z]`` snapshot from
        :meth:`~sparsegf2.SparseGF2.to_symplectic`, only when the runner is
        asked to save it.
    analyses : dict[str, Any]
        Results of any named/custom analyses requested via ``analyses=`` on
        :func:`~sparsegf2.circuits.simulate`. Empty when none were requested.
        Keys are analysis names; values are the returned scalars / ndarrays.
        Additive to (not a replacement for) the fixed observables above; see
        :mod:`sparsegf2.analysis`.
    """

    # identity
    sample_seed: int
    picture: Picture
    # diagnostics
    total_layers: int
    total_gates: int
    total_measurements: int
    gate_to_meas_ratio_expected: float
    gate_to_meas_ratio_actual: float
    # observables (union shape, None when not applicable to the picture)
    code_dimension: int | None = None
    ref_entropy: int | None = None
    entropy_half_cut: int | None = None
    # runtime
    runtime_total_s: float = 0.0
    # adaptive depth: the measured layer at which the order parameter first
    # reached its absorbing value (k=0 / S=0) under depth_mode='until_purified';
    # None when not run in that mode or never absorbed within the cap.
    purified_at_layer: int | None = None
    # tableau-density diagnostic: a_bar = SparseGF2.active_count() (the mean
    # number of generators touching a qubit), time-averaged over the measured
    # layers. None when the circuit had no measured layers. Generating-set
    # dependent (like the weight diagnostics), not a physical observable.
    mean_active_generators: float | None = None
    # per-layer order-parameter trajectory when record_time_series=True
    time_series: list[int] | None = None
    # optional side-payload
    final_tableau: NDArray[np.uint8] | None = None
    # named/custom analysis results (additive; empty unless requested)
    analyses: dict[str, Any] = field(default_factory=dict)


__all__ = ["SampleRecord"]
