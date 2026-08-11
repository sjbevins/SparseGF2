"""The per-sample output record.

:class:`SampleRecord` is what one circuit realization contributes to a
run. It is the **stable contract** consumed by ``sparsegf2.analysis``,
so the rule is: optional fields may be *added*
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
    graph_name : str or None
        Name of the resolved graph topology (identifies the family and, for
        parametrized/stochastic families, its parameters and realization seed).
    graph6 : str or None
        graph6 string of the resolved graph, saved whenever the string spec and
        ``n`` alone cannot reconstruct the geometry: stochastic realizations,
        and any prebuilt :class:`GraphTopology` / networkx-adapted graph passed
        as ``graph_spec`` (the record otherwise carries only its name). ``None``
        for deterministic string specs (``"cycle"``, ...), which the spec and
        ``n`` fully determine.
    runtime_total_s : float
        Wall-clock seconds for the run (gates + measurements + observables).
    purified_at_layer : int or None
        Layer at which the runner detected a zero reference order parameter.
        Exact under ``depth_mode="until_purified"``; with a sparse
        ``CHECKPOINT_STOP`` callback this is the first stopping checkpoint
        observed at zero and therefore an upper bound on the exact transition.
    mean_active_generators : float or None
        Mean tableau active-generator count over the measured layers.
    time_series : list[int] or None
        Per-layer reference order parameter when ``record_time_series=True``.
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
    checkpoint_tableaux : dict[int, ndarray[uint8]] or None
        ``{measured_layer: (2N, 2N) [X|Z] tableau}`` captured at the depths
        requested via ``checkpoint_layers=`` on the runner, each an independent
        :meth:`~sparsegf2.SparseGF2.to_symplectic` snapshot taken *after* that
        measured layer's gates and measurements. ``None`` when no requested
        checkpoint was reached. If the final executed layer is requested, its
        snapshot equals ``final_tableau`` when both are asked for.
    checkpoint_values : dict[int, Any] or None
        ``{measured_layer: callback_result}`` captured when the runner is given
        ``checkpoint_callback``. ``None`` when no requested checkpoint produced
        a stored value. A callback that returns ``CHECKPOINT_STOP`` requests an
        early stop; that sentinel is not stored as a value.
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
    # graph provenance: name always; graph6 only for stochastic graphs, whose
    # realization is not recoverable from (spec, n) alone (deterministic graphs
    # leave it None to keep records lean).
    graph_name: str | None = None
    graph6: str | None = None
    # runtime
    runtime_total_s: float = 0.0
    # The measured layer at which the runner detected the absorbing value
    # (k=0 / S=0). This is the exact first-zero layer under
    # depth_mode='until_purified'. With CHECKPOINT_STOP it is the first stopping
    # checkpoint observed at zero, hence an upper bound on the unobserved exact
    # transition layer. None when no zero was detected.
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
    # named/custom analysis results (additive; empty unless requested). Keep
    # every new additive field after this one so the pre-checkpoint positional
    # constructor layout remains backward compatible.
    analyses: dict[str, Any] = field(default_factory=dict)
    # optional depth checkpoints: {measured_layer: (2N, 2N) tableau} captured at
    # the layers requested via checkpoint_layers= on the runner (dynamics studies).
    checkpoint_tableaux: dict[int, NDArray[np.uint8]] | None = None
    # optional depth-checkpoint values: {measured_layer: callback_result}, filled
    # instead of checkpoint_tableaux when a checkpoint_callback is given (compute an
    # observable on the live state at each depth without saving/reconstructing it).
    checkpoint_values: dict[int, Any] | None = None


__all__ = ["SampleRecord"]
