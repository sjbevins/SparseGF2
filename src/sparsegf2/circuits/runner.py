"""Simulation runner: execute one circuit realization and collect a record.

The runner is the thin layer between the :class:`CircuitBuilder` (which
deterministically produces the layer schedule) and the
:class:`~sparsegf2.SparseGF2` simulator (which applies it). It walks the
layers, applies each two-qubit Clifford via
:meth:`~sparsegf2.SparseGF2.apply_gate_2q` and each measurement via
:meth:`~sparsegf2.SparseGF2.measure_z`, then reads the picture's order
parameter and packs a :class:`SampleRecord`.

Two RNG streams, by design:

- **Circuit construction** lives on the :class:`CircuitBuilder`, seeded
  ``base_seed + sample_seed``.
- **Measurement outcomes** live on the :class:`~sparsegf2.SparseGF2`
  instance, seeded *independently* here so that the random measurement
  coins are decoupled from the circuit realization (re-run the same
  circuit with different outcomes by changing only this stream).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sparsegf2 import code_dimension, entanglement_entropy, random_symplectic
from sparsegf2.circuits import _clifford_table
from sparsegf2.circuits.config import CircuitConfig
from sparsegf2.circuits.picture import setup_picture
from sparsegf2.circuits.records import SampleRecord
from sparsegf2.circuits.scheduler import CircuitBuilder
from sparsegf2.core.sparse_tableau import _build_2q_lut
from sparsegf2.errors import InvalidArgumentError

# Distinguishes the measurement-outcome SeedSequence from the builder's
# single-int seed, so the two streams are independent for every sample
# (a two-element seed never hashes to a plain ``base+sample`` int).
_MEAS_STREAM_KEY = 0x6D656173  # "meas"
_SCRAMBLE_STREAM_KEY = 0x73637262  # "scrb", independent stream for the global scramble Clifford

# Quiet by default; a crash log (sparsegf2.analysis.crash_log) is what attaches a
# handler and a level. INFO records the phases of every simulation; DEBUG (the
# "trace" detail level) records every individual layer.
_log = logging.getLogger(__name__)


# The per-index gate LUT list depends only on the gate table (+ how many of its
# entries are in play), not on the cell, and every cell in a sweep shares the
# same process-cached table. Memoize the list by table content so a 1600-cell
# sweep builds it once instead of re-probing ~720 LUTs per cell.
_GATE_LUTS_BY_TABLE: dict[bytes, list[np.ndarray]] = {}


def _gate_luts_for(table: NDArray[np.uint8], n_cliffords: int) -> list[np.ndarray]:
    key = np.ascontiguousarray(table[:n_cliffords]).tobytes()
    luts = _GATE_LUTS_BY_TABLE.get(key)
    if luts is None:
        luts = [_build_2q_lut(table[i]) for i in range(n_cliffords)]
        _GATE_LUTS_BY_TABLE[key] = luts
    return luts


def _cfg_brief(cfg: CircuitConfig) -> str:
    """A one-line, log-friendly summary of the cell being simulated."""
    graph = getattr(cfg.graph_spec, "name", cfg.graph_spec)
    return (
        f"picture={cfg.picture} n={cfg.n} p={cfg.p} graph={graph} "
        f"gating={cfg.gating_mode} matching={cfg.matching_mode} "
        f"meas={cfg.measurement_mode} depth={cfg.depth_mode}x{cfg.depth_factor}"
    )


def _order_parameter(sim, spec) -> int:
    """The picture's order parameter (absorbing value 0): used for
    ``until_purified`` early-stop and the recorded time series."""
    if spec.order_parameter == "code_dimension":
        return int(code_dimension(sim, spec.n_system))
    if spec.order_parameter == "ref_entropy":
        return int(entanglement_entropy(sim, spec.reference_qubits))
    # pure_state has no absorbing parameter (config forbids until_purified /
    # record_time_series there); fall back to the half-cut entropy.
    return int(entanglement_entropy(sim, range(spec.n_system // 2))) if spec.n_system >= 2 else 0


def simulate(
    config: CircuitConfig,
    *,
    sample_seed: int = 0,
    analyses: Sequence[str | Callable[[Any, Any], Any]]
    | Mapping[str, Callable[[Any, Any], Any]]
    | None = None,
    save_tableau: bool = False,
) -> SampleRecord:
    """Run one sample and return its :class:`SampleRecord`.

    Thin convenience wrapper around :class:`SimulationRunner`.

    Parameters
    ----------
    analyses
        Optional named/custom analyses to compute on the final state
        (``record.analyses``). A list mixing registered names and callables
        ``fn(sim, spec)->value``, or a ``{name: fn}`` mapping. See
        :mod:`sparsegf2.analysis`. Computed at end-of-circuit; the tableau is
        then discarded (online analysis).
    save_tableau
        If true, also store the ``(2N, 2N)`` symplectic in
        ``record.final_tableau`` for later offline analysis.
    """
    return SimulationRunner(config).run(sample_seed, analyses=analyses, save_tableau=save_tableau)


class SimulationRunner:
    """Run circuit realizations of one :class:`CircuitConfig`.

    Construct once, call :meth:`run` per sample. The ``Sp(4, F_2)`` table
    is fetched once at construction (it is process-cached, so this is
    essentially free) and shared across runs.

    Parameters
    ----------
    config : CircuitConfig
        The cell configuration.
    clifford_table : ndarray, optional
        A ``(>=n_cliffords, 4, 4)`` uint8 table to sample gates from.
        Defaults to the cached :func:`sparsegf2.circuits._clifford_table.sp4_table`.
        Injectable for tests / custom gate sets without monkeypatching.
    """

    def __init__(
        self,
        config: CircuitConfig,
        clifford_table: NDArray[np.uint8] | None = None,
    ) -> None:
        self.config = config
        self.clifford_table = (
            clifford_table if clifford_table is not None else _clifford_table.sp4_table()
        )
        if len(self.clifford_table) < config.n_cliffords:
            raise InvalidArgumentError(
                f"clifford_table has {len(self.clifford_table)} entries but "
                f"config.n_cliffords={config.n_cliffords}"
            )
        # Each gate's lookup table, prebuilt once per distinct gate table (see
        # _gate_luts_for). The runner replays gates by index from this fixed
        # table, so re-keying the symplectic on every gate (a tobytes + cache
        # lookup) is pure waste; with the LUTs prebuilt the inner loop calls
        # SparseGF2._dispatch_2q directly.
        self._gate_luts = _gate_luts_for(self.clifford_table, config.n_cliffords)

    def run(
        self,
        sample_seed: int = 0,
        *,
        analyses: Sequence[str | Callable[[Any, Any], Any]]
        | Mapping[str, Callable[[Any, Any], Any]]
        | None = None,
        save_tableau: bool = False,
    ) -> SampleRecord:
        """Execute one realization and return its record."""
        cfg = self.config
        # Clifford indices are drawn in [0, cfg.n_cliffords) and __init__ prebuilds
        # one LUT per index in that range, so every index is in range for `luts`.
        luts = self._gate_luts

        t0 = time.perf_counter()
        _log.info("simulate seed=%d: %s", sample_seed, _cfg_brief(cfg))

        # Independent measurement-outcome stream (see module docstring).
        sim_rng = np.random.default_rng([cfg.base_seed + sample_seed, _MEAS_STREAM_KEY])
        sim, spec = setup_picture(
            cfg.picture,
            cfg.n,
            rng=sim_rng,
            pivot_mode=cfg.pivot_mode,
            use_numba=cfg.use_numba,
            hybrid=cfg.hybrid,
        )

        builder = CircuitBuilder(cfg, sample_seed=sample_seed)

        # --- Global scramble: one maximally-scrambling random Clifford on the
        # system qubits (a one-shot alternative/complement to the gate-only
        # warmup). cfg.scramble_qubits() is the single source of truth for which
        # qubits it covers (all system qubits, or all but the reference's partner
        # for single_ref with scramble_entangled_qubit=False), shared with the
        # inspector so the drawn diagram matches what runs. Drawn from an
        # independent construction sub-stream so toggling it does not perturb the
        # gate-placement or measurement RNG sequences. ---
        scr_qubits = cfg.scramble_qubits()
        if scr_qubits is not None and len(scr_qubits) >= 1:
            scr_rng = np.random.default_rng([cfg.base_seed + sample_seed, _SCRAMBLE_STREAM_KEY])
            k = len(scr_qubits)
            sim.apply_clifford(random_symplectic(k, scr_rng), qubits=scr_qubits)
            _log.info(
                "global scramble applied on %d system qubit%s%s",
                k,
                "" if k == 1 else "s",
                " (reference partner held out)" if k < spec.n_system else "",
            )

        # --- Warmup (gate-only) phase: scramble before measuring. ---
        n_warmup = 0
        for wlayer in builder.warmup_layers_iter():
            for g, (qi, qj) in enumerate(wlayer.gate_pairs):
                ci = int(wlayer.cliff_indices[g])
                sim._dispatch_2q(qi, qj, luts[ci])
            n_warmup += 1
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug("warmup layer %d: %d gates", n_warmup, wlayer.n_gates)
        if n_warmup:
            _log.info("warmup done: %d gate-only layers", n_warmup)

        # --- Measured phase. ---
        # When tracking the order parameter (until_purified early-stop and/or a
        # recorded time series), read it after each measured layer.
        track_op = cfg.depth_mode == "until_purified" or cfg.record_time_series
        time_series: list[int] | None = [] if cfg.record_time_series else None
        purified_at: int | None = None
        # The order parameter is an entanglement entropy and the references are
        # never gated, so it is invariant under the gates and can only change after
        # a measurement. Seed it once post-warmup; in the loop it is recomputed only
        # on layers that measured, and (without a time series) only once the
        # measurement budget below is spent.
        last_op: int | None = _order_parameter(sim, spec) if track_op else None

        total_gates = 0
        total_measurements = 0
        total_layers = 0
        # Check-skipping for until_purified without a recorded time series:
        # the order parameter is an entanglement entropy with the reference, so a
        # single-qubit measurement changes it by at most 1, and measurements on
        # system qubits can never increase it (LOCC monotonicity). Hence after a
        # check reads k = K > 0, the order parameter stays > 0 until at least K
        # further measurements have occurred, so every intermediate rank would be
        # wasted. We re-check only once the measurement budget is spent. tau stays
        # EXACT: at the first layer where the cumulative count reaches K, every
        # earlier layer provably had k > 0, so if the check now reads 0 this layer
        # is the true first-zero layer; if it reads K' > 0 the budget resets to K'.
        # (With record_time_series the value is needed every layer, so no skipping;
        # the bit-packed rank keeps that affordable.)
        skip_checks = time_series is None
        meas_since_check = 0
        # Tableau-density diagnostic: a_bar (mean generators per qubit) averaged
        # over the measured layers. An O(2n) sum per layer (popcount in hybrid
        # dense mode), negligible next to the layer's gates.
        abar_sum = 0.0
        for layer in builder.layers():
            total_layers += 1
            for g, (qi, qj) in enumerate(layer.gate_pairs):
                ci = int(layer.cliff_indices[g])
                sim._dispatch_2q(qi, qj, luts[ci])
            total_gates += layer.n_gates
            for q in layer.meas_qubits:
                sim.measure_z(q)  # outcome drawn from sim_rng (sim._rng)
            total_measurements += layer.n_measurements
            abar_sum += sim.active_count()

            if track_op:
                if layer.n_measurements > 0:  # gates can't move it; only a measurement can
                    if skip_checks:
                        meas_since_check += layer.n_measurements
                        if last_op is None or meas_since_check >= last_op:
                            last_op = _order_parameter(sim, spec)
                            meas_since_check = 0
                    else:
                        last_op = _order_parameter(sim, spec)
                if time_series is not None:
                    time_series.append(last_op)
                if cfg.depth_mode == "until_purified" and last_op == 0:
                    purified_at = total_layers
                    _log.info("purified: order parameter hit 0 at measured layer %d", total_layers)
                    break  # absorbing state reached, stop early
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(
                    "measured layer %d: %d gates, %d measurements%s",
                    total_layers,
                    layer.n_gates,
                    layer.n_measurements,
                    f", order_param={last_op}" if track_op else "",
                )

        # --- Observables. The half-cut entropy is recorded for every picture;
        # the picture's order_parameter names the extra scalar. When the order
        # parameter was already tracked per layer, reuse the final value
        # (``_order_parameter`` returns exactly code_dimension / ref_entropy for
        # those pictures) instead of recomputing it. ---
        if track_op and skip_checks and meas_since_check > 0:
            # The loop exited (depth cap) with measurements still inside the
            # check budget: last_op is an upper bound, not the final value.
            # One refresh keeps the recorded observables exact.
            last_op = _order_parameter(sim, spec)
        half = range(cfg.n // 2)
        entropy_half = int(entanglement_entropy(sim, half)) if cfg.n >= 2 else 0
        code_dim: int | None = None
        ref_ent: int | None = None
        if spec.order_parameter == "code_dimension":
            code_dim = last_op if last_op is not None else int(code_dimension(sim, spec.n_system))
        elif spec.order_parameter == "ref_entropy":
            ref_ent = (
                last_op
                if last_op is not None
                else int(entanglement_entropy(sim, spec.reference_qubits))
            )

        # --- Ratios. ---
        actual_ratio = total_gates / total_measurements if total_measurements > 0 else float("inf")

        record = SampleRecord(
            sample_seed=int(sample_seed),
            picture=spec.picture,
            total_layers=total_layers,
            total_gates=total_gates,
            total_measurements=total_measurements,
            gate_to_meas_ratio_expected=float(cfg.expected_gate_to_meas_ratio()),
            gate_to_meas_ratio_actual=float(actual_ratio),
            code_dimension=code_dim,
            ref_entropy=ref_ent,
            entropy_half_cut=entropy_half,
            runtime_total_s=time.perf_counter() - t0,
            purified_at_layer=purified_at,
            mean_active_generators=(abar_sum / total_layers) if total_layers > 0 else None,
            time_series=time_series,
        )
        _log.info(
            "simulate done seed=%d: %d layers, %d gates, %d measurements, "
            "half_cut=%d code_dim=%s ref_ent=%s in %.3fs",
            sample_seed,
            total_layers,
            total_gates,
            total_measurements,
            entropy_half,
            code_dim,
            ref_ent,
            record.runtime_total_s,
        )
        # --- Named/custom analyses (online): compute now, then discard sim. ---
        if analyses is not None:
            # Lazy import keeps `circuits` importable without the `analysis`
            # layer and breaks the circuits<->analysis import cycle.
            from sparsegf2.analysis.registry import resolve_analyses

            record.analyses = {
                name: fn(sim, spec) for name, fn in resolve_analyses(analyses).items()
            }
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug("analyses computed: %s", dict(record.analyses))

        if save_tableau:
            record.final_tableau = sim.to_symplectic()
            _log.debug("final tableau saved (%dx%d)", 2 * cfg.n, 2 * cfg.n)
        return record


__all__ = ["simulate", "SimulationRunner"]
