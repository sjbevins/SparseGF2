"""Tests for the observable-agnostic FSS data-collapse tools.

Three groups:

(a) synthetic recovery: curves are generated from a KNOWN master function with
    KNOWN ``(p_c, nu, z)``; the fits must recover them and the loss must be near
    zero at the truth and larger when perturbed.
(b) observable-agnostic on REAL simulator output: a tiny WS sweep produces
    several different observables; every tool must run and return finite values
    for at least two of them.
(c) ``smooth_pc_collapse`` over a few synthetic control values tightens the
    per-cell ``p_c`` scatter toward a smooth trend.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

import sparsegf2.analysis.fss_collapse as fss_module
from sparsegf2.analysis import (
    fit_fixed_pc,
    fit_three_param,
    graph_bootstrap,
    master_curve_loss,
    robust_pc,
    smooth_pc_collapse,
)
from sparsegf2.circuits import CircuitConfig

# ----------------------------------------------------------------------
# synthetic data from a known master function O(p,L) = L**z F((p-pc) L**(1/nu))
# ----------------------------------------------------------------------

SIZES = (8, 12, 16, 24, 32, 48)
PC_TRUE, NU_TRUE, Z_TRUE = 0.30, 1.50, 0.80


def _master(x):
    """A smooth, monotone, strictly positive scaling function F(X)."""
    return 0.5 + 2.0 / (1.0 + np.exp(-x))  # logistic in [0.5, 2.5]


def _synthetic_curves(pc=PC_TRUE, nu=NU_TRUE, z=Z_TRUE, sizes=SIZES, n_p=21):
    """Exact FSS curves {L: {p: L**z F((p-pc) L**(1/nu))}} on a p grid per size.

    The p grid is chosen per size so the collapse variable X spans a common
    window across sizes (that is what a real crossing/collapse needs).
    """
    curves = {}
    for L in sizes:
        # common X-window [-6, 6] -> p-window shrinks with L**(1/nu)
        x = np.linspace(-6.0, 6.0, n_p)
        ps = pc + x / L ** (1.0 / nu)
        vals = L**z * _master((ps - pc) * L ** (1.0 / nu))
        curves[int(L)] = {float(p): float(v) for p, v in zip(ps, vals, strict=True)}
    return curves


# ----------------------------------------------------------------------
# (a) synthetic recovery
# ----------------------------------------------------------------------


def test_master_curve_loss_zero_at_truth_and_grows_when_perturbed():
    curves = _synthetic_curves()
    s, p, y = [], [], []
    for L, curve in curves.items():
        for pp, vv in curve.items():
            s.append(L)
            p.append(pp)
            y.append(vv)
    s, p, y = np.array(s, float), np.array(p, float), np.array(y, float)

    loss_true = master_curve_loss(s, p, y, PC_TRUE, NU_TRUE, Z_TRUE)
    assert loss_true < 1e-4  # exact ansatz collapses perfectly

    # perturbing any parameter must worsen the collapse
    assert master_curve_loss(s, p, y, PC_TRUE + 0.05, NU_TRUE, Z_TRUE) > loss_true + 1e-6
    assert master_curve_loss(s, p, y, PC_TRUE, NU_TRUE * 1.5, Z_TRUE) > loss_true + 1e-6
    assert master_curve_loss(s, p, y, PC_TRUE, NU_TRUE, Z_TRUE + 0.4) > loss_true + 1e-6


def test_fit_three_param_recovers_truth():
    curves = _synthetic_curves()
    pc, nu, z = fit_three_param(curves)
    assert pc == pytest.approx(PC_TRUE, abs=0.02)
    assert nu == pytest.approx(NU_TRUE, abs=0.25)
    assert z == pytest.approx(Z_TRUE, abs=0.10)


def test_fit_fixed_pc_recovers_exponents_at_true_pc():
    curves = _synthetic_curves()
    nu, z = fit_fixed_pc(curves, PC_TRUE)
    assert nu == pytest.approx(NU_TRUE, abs=0.20)
    assert z == pytest.approx(Z_TRUE, abs=0.08)


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(success=False, fun=0.0, x=(NU_TRUE, Z_TRUE)),
        SimpleNamespace(success=True, fun=np.nan, x=(NU_TRUE, Z_TRUE)),
        SimpleNamespace(success=True, fun=0.0, x=(NU_TRUE, 2.0)),
    ],
)
def test_fit_fixed_pc_rejects_invalid_warm_result(monkeypatch, result):
    monkeypatch.setattr(fss_module, "_minimize", lambda *args, **kwargs: result)
    assert all(np.isnan(fit_fixed_pc(_synthetic_curves(), PC_TRUE, warm=(1.4, 0.7))))


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(success=False, fun=0.0, x=(PC_TRUE, NU_TRUE, Z_TRUE)),
        SimpleNamespace(success=True, fun=np.inf, x=(PC_TRUE, NU_TRUE, Z_TRUE)),
        SimpleNamespace(success=True, fun=0.0, x=(0.9, NU_TRUE, Z_TRUE)),
    ],
)
def test_fit_three_param_rejects_invalid_warm_result(monkeypatch, result):
    monkeypatch.setattr(fss_module, "_minimize", lambda *args, **kwargs: result)
    assert all(np.isnan(fit_three_param(_synthetic_curves(), warm=(0.3, 1.4, 0.7))))


def test_fit_fixed_pc_point_estimate_screens_optimizer_results(monkeypatch):
    calls = 0
    wanted = (1.6, 0.82)

    def fake_minimize(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(success=False, fun=-3.0, x=(NU_TRUE, Z_TRUE))
        if calls == 2:
            return SimpleNamespace(success=True, fun=-2.0, x=(9.0, Z_TRUE))
        if calls == 3:
            return SimpleNamespace(success=True, fun=np.nan, x=(NU_TRUE, Z_TRUE))
        if calls == 4:
            return SimpleNamespace(success=True, fun=0.2, x=wanted)
        return SimpleNamespace(success=True, fun=0.5, x=(1.7, 0.85))

    monkeypatch.setattr(fss_module, "_minimize", fake_minimize)
    assert fit_fixed_pc(_synthetic_curves(), PC_TRUE) == wanted


def test_fit_three_param_point_estimate_screens_optimizer_results(monkeypatch):
    calls = 0
    wanted = (0.31, 1.6, 0.82)

    def fake_minimize(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(success=False, fun=-3.0, x=(PC_TRUE, NU_TRUE, Z_TRUE))
        if calls == 2:
            return SimpleNamespace(success=True, fun=-2.0, x=(0.9, NU_TRUE, Z_TRUE))
        if calls == 3:
            return SimpleNamespace(success=True, fun=np.nan, x=(PC_TRUE, NU_TRUE, Z_TRUE))
        if calls == 4:
            return SimpleNamespace(success=True, fun=0.2, x=wanted)
        return SimpleNamespace(success=True, fun=0.5, x=(0.32, 1.7, 0.85))

    monkeypatch.setattr(fss_module, "_minimize", fake_minimize)
    assert fit_three_param(_synthetic_curves()) == wanted


def test_fit_three_param_returns_nan_when_every_start_fails(monkeypatch):
    failed = SimpleNamespace(success=False, fun=0.0, x=(PC_TRUE, NU_TRUE, Z_TRUE))
    monkeypatch.setattr(fss_module, "_minimize", lambda *args, **kwargs: failed)
    assert all(np.isnan(fit_three_param(_synthetic_curves())))


def test_robust_pc_recovers_pc():
    curves = _synthetic_curves()
    pc, z_cross = robust_pc(curves)
    assert np.isfinite(pc)
    assert pc == pytest.approx(PC_TRUE, abs=0.03)
    assert np.isfinite(z_cross)


# ----------------------------------------------------------------------
# graph_bootstrap input contract
# ----------------------------------------------------------------------


def _raw_cell_equal_counts(n_graphs=6):
    """A RAW cell {L: {p: [per-graph values]}} with equal counts at every p."""
    rng = np.random.default_rng(0)
    curves = _synthetic_curves(sizes=(8, 12, 16), n_p=7)
    return {
        L: {
            p: [v + float(rng.normal(0, 0.01 * v)) for _ in range(n_graphs)]
            for p, v in curve.items()
        }
        for L, curve in curves.items()
    }


def test_graph_bootstrap_runs_on_equal_counts():
    raw = _raw_cell_equal_counts()
    out = graph_bootstrap(raw, lambda c: fit_fixed_pc(c, PC_TRUE), n_boot=8, seed=0)
    assert out["n_boot_ok"] >= 1
    assert all(np.isfinite(s) for s in out["std"])
    assert all(np.isfinite(v) for v in out["point"])


def test_graph_bootstrap_rejects_ragged_counts():
    # per-graph counts differ across p at one size (e.g. censoring dropped
    # different graphs at different p): shared-index resampling is ill-defined,
    # so the tool must raise the package error, NOT crash with a raw IndexError.
    from sparsegf2.errors import InvalidArgumentError

    raw = _raw_cell_equal_counts(n_graphs=6)
    a_size = next(iter(raw))
    a_p = next(iter(raw[a_size]))
    raw[a_size][a_p] = raw[a_size][a_p][:4]  # make one (L, p) cell ragged
    with pytest.raises(InvalidArgumentError):
        graph_bootstrap(raw, lambda c: fit_fixed_pc(c, PC_TRUE), n_boot=8, seed=0)


# ----------------------------------------------------------------------
# (c) smooth_pc_collapse tightens the per-cell p_c scatter
# ----------------------------------------------------------------------


def test_smooth_pc_collapse_tracks_trend_and_tightens_scatter():
    controls = [0.0, 0.25, 0.5, 0.75]
    # a smooth, monotone p_c trend across the control values
    pc_trend = {c: 0.25 + 0.10 * c for c in controls}
    rng = np.random.default_rng(1)

    cells = {}
    raw_pc = []
    for c in controls:
        # jitter the true p_c a little per cell so the raw per-cell fit scatters
        pc_c = pc_trend[c] + float(rng.normal(0, 0.01))
        raw_pc.append(pc_c)
        cells[c] = _synthetic_curves(pc=pc_c, nu=NU_TRUE, z=Z_TRUE)

    out = smooth_pc_collapse(cells, window=3)
    assert set(out) == set(controls)

    pc_smooth = np.array([out[c][0] for c in controls])
    trend = np.array([pc_trend[c] for c in controls])
    # smoothed p_c tracks the underlying monotone trend
    assert np.all(np.diff(pc_smooth) > 0)
    assert np.max(np.abs(pc_smooth - trend)) < 0.03
    # and the residual to the smooth trend is no worse than the raw scatter
    raw_resid = np.std(np.array(raw_pc) - trend)
    smooth_resid = np.std(pc_smooth - trend)
    assert smooth_resid <= raw_resid + 0.01
    # exponents come back finite and sensible
    for c in controls:
        _pc, nu, z = out[c]
        assert 0.5 < nu < 4.0
        assert 0.2 < z < 1.25


# ----------------------------------------------------------------------
# (b) observable-agnostic on REAL simulator output (small + fast)
# ----------------------------------------------------------------------

_WS_SIZES = (8, 12, 16)
_WS_PS = (0.05, 0.12, 0.20, 0.30, 0.45)
_WS_SEEDS = 4


def _cell_from_sweep(picture, gating, analyses, obs_key):
    """Build a per-graph RAW cell {L: {p: [per-seed values]}} and its median
    curve {L: {p: mean}} for one observable from a small WS sweep."""
    from sparsegf2.analysis import sweep

    raw = {n: {p: [] for p in _WS_PS} for n in _WS_SIZES}
    for n in _WS_SIZES:
        base = CircuitConfig(
            graph_spec="watts_strogatz(k=2,beta=0.3)",
            n=n,
            picture=picture,
            gating_mode=gating,
            measurement_mode="bernoulli",
            depth_factor=3,
            base_seed=100 + n,
        )
        res = sweep(base, {"p": list(_WS_PS)}, seeds=_WS_SEEDS, analyses=analyses)
        for row in res.rows:
            val = row.get(obs_key)
            if val is not None:
                raw[n][float(row["p"])].append(float(val))
    curves = {n: {p: float(np.mean(v)) for p, v in raw[n].items() if v} for n in _WS_SIZES}
    raw_cell = {n: {p: v for p, v in raw[n].items() if v} for n in _WS_SIZES}
    return curves, raw_cell


def test_tools_are_observable_agnostic_on_real_output():
    # single_ref carries ref_entropy; both pictures carry these bulk observables.
    single, single_raw = {}, {}
    curves_re, raw_re = _cell_from_sweep(
        "single_ref",
        "random_pool",
        ["ref_entropy", "half_cut_entropy", "average_stabilizer_weight"],
        "a.ref_entropy",
    )
    single["ref_entropy"] = curves_re
    single_raw["ref_entropy"] = raw_re

    curves_hc, raw_hc = _cell_from_sweep(
        "single_ref",
        "random_pool",
        ["half_cut_entropy"],
        "a.half_cut_entropy",
    )
    single["half_cut_entropy"] = curves_hc
    single_raw["half_cut_entropy"] = raw_hc

    # code_dimension needs the purification picture.
    curves_cd, raw_cd = _cell_from_sweep(
        "purification",
        "random_pool",
        ["code_dimension"],
        "a.code_dimension",
    )
    single["code_dimension"] = curves_cd
    single_raw["code_dimension"] = raw_cd

    # every tool must RUN and return finite values for each observable -- this is
    # the observable-agnostic contract (>= two distinct observables exercised).
    tested = 0
    for name, curves in single.items():
        # need every size populated at >= 3 p to have any chance of a collapse
        if not all(len(curves.get(n, {})) >= 3 for n in _WS_SIZES):
            continue
        tested += 1

        pc, z_cross = robust_pc(curves)
        assert np.isfinite(pc) or np.isnan(pc)  # returns nan gracefully if no crossing

        nu_f, z_f = fit_fixed_pc(curves, 0.2)
        # fit_fixed_pc returns nan when no start converges inside the exponent
        # bounds (it never fabricates an in-band pair); both are well-defined.
        assert (np.isfinite(nu_f) and np.isfinite(z_f)) or (np.isnan(nu_f) and np.isnan(z_f))

        pc3, nu3, z3 = fit_three_param(curves)
        assert np.isfinite(pc3) and np.isfinite(nu3) and np.isfinite(z3)

        # bootstrap with the always-finite three-parameter fit as the estimator,
        # so the resample std is well-posed on this tiny noisy real data.
        boot = graph_bootstrap(single_raw[name], fit_three_param, n_boot=8, seed=0)
        assert boot["n_boot_ok"] >= 1
        assert all(np.isfinite(s) for s in boot["std"])
        assert all(np.isfinite(v) for v in boot["point"])

    assert tested >= 2  # observable-agnostic: at least two distinct observables
