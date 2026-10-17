"""Tests of sparsegf2.study.fss (specification v1.4 §5.9, §9; design §7).

The collapse quality is checked against a loop implementation of Houdayer and Hartmann's own
formulas (PRB 70, 014418 (2004): the raw sums K, Kx, Ky, Kxx, Kxy of a weighted line, not the
centered form the code uses); the optimizer on standard test functions with known minima; the
fits on synthetic data generated from a known scaling form, exact or with a graph hierarchy
whose errors are known. Seeds are fixed and every statistical tolerance states its false-alarm
probability.
"""

import json
import math
from fractions import Fraction

import numpy as np
import pytest

from sparsegf2.study import fss
from sparsegf2.study import stats as st

BOX = {"p_c": (0.1, 0.3), "z": (0.0, 2.0), "nu": (0.5, 3.0)}
TRUTH = (0.2, 1.0, 1.5)
SIZES = (16, 32, 64, 128)


def _f1(x):
    return 2.0 / (1.0 + np.exp(x)) + 0.5


def _f2(x):
    return np.exp(-0.5 * x) + 0.2


def _exact(params=TRUTH, f=_f1, sizes=SIZES, rates=None, rel_se=0.01, name="y"):
    """Noise-free points of y = n^z F((p - p_c) n^(1/nu)) with relative errors ``rel_se``."""
    pc, z, nu = params
    rates = np.linspace(0.1, 0.3, 41) if rates is None else rates
    n = np.repeat(sizes, rates.size)
    p = np.tile(rates, len(sizes))
    y = n**z * f((p - pc) * n ** (1 / nu))
    return fss.ScalingData(n=n, p=p, y=y, se=rel_se * y, observable=name, beta="0.1")


def _hierarchical(
    params=TRUTH, f=_f1, sizes=SIZES, rates=None, graphs=20, sg=0.05, sp=0.03, seed=0, name="y"
):
    """Graph means with a graph effect shared by every rate of a size (the coupled rates) and
    rate-specific noise; y and se are their nested means and graph-cluster errors."""
    pc, z, nu = params
    rates = np.linspace(0.1, 0.3, 21) if rates is None else rates
    rng = np.random.default_rng(seed)
    means, nn, pp = [], [], []
    for n in sizes:
        effect = rng.normal(0, sg, graphs)
        for p in rates:
            truth = n**z * f((p - pc) * n ** (1 / nu))
            values = truth * (1 + effect + rng.normal(0, sp, graphs))
            means.append(st.mean_over_graphs(np.arange(graphs), values))
            nn.append(n)
            pp.append(p)
    return fss.scaling_data(
        np.array(nn), np.array(pp), means, observable=name, beta="0.1", label="exploratory"
    )


# ---------------------------------------------------------------------------------------------
# the collapse quality
# ---------------------------------------------------------------------------------------------


def _reference_chi2(n, p, y, se, pc, z, nu):
    """Houdayer-Hartmann by loops, with their raw weighted-line formulas."""
    pts = [
        (int(a), (b - pc) * a ** (1 / nu), c * a ** (-z), d * a ** (-z))
        for a, b, c, d in zip(n.tolist(), p.tolist(), y.tolist(), se.tolist(), strict=True)
    ]
    by_size = {}
    for size, x, yy, e in pts:
        by_size.setdefault(size, []).append((x, yy, e))
    for size in by_size:
        by_size[size].sort(key=lambda t: t[0])
    out = []
    for size, xi, yi, ei in pts:
        chosen = []
        for other, lst in by_size.items():
            if other == size:
                continue
            xs = [t[0] for t in lst]
            pair = None
            for j in range(len(lst) - 1):  # the last pair x_j <= x_i <= x_{j+1}
                if xs[j] <= xi <= xs[j + 1]:
                    pair = j
            if pair is not None:
                chosen += [lst[pair], lst[pair + 1]]
            elif len(lst) == 1 and xs[0] == xi:
                chosen.append(lst[0])
        if not chosen:
            out.append(math.nan)
            continue
        w = [1 / e**2 for _, _, e in chosen]
        k = sum(w)
        kx = sum(wi * x for wi, (x, _, _) in zip(w, chosen, strict=True))
        ky = sum(wi * yy for wi, (_, yy, _) in zip(w, chosen, strict=True))
        kxx = sum(wi * x * x for wi, (x, _, _) in zip(w, chosen, strict=True))
        kxy = sum(wi * x * yy for wi, (x, yy, _) in zip(w, chosen, strict=True))
        delta = k * kxx - kx * kx
        if all(x == chosen[0][0] for x, _, _ in chosen):
            master, var = ky / k, 1 / k
        else:
            master = (kxx * ky - kx * kxy) / delta + xi * (k * kxy - kx * ky) / delta
            var = (kxx - 2 * xi * kx + xi * xi * k) / delta
        out.append((yi - master) ** 2 / (ei**2 + var))
    return np.array(out)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_collapse_cost_matches_the_houdayer_hartmann_formulas(seed):
    rng = np.random.default_rng(seed)
    sizes = rng.choice([8, 12, 16, 24, 32, 48, 64], size=int(rng.integers(2, 6)), replace=False)
    n, p = [], []
    for size in sizes:
        rates = np.sort(rng.uniform(0.0, 0.4, int(rng.integers(1, 12))))
        n += [int(size)] * rates.size
        p += rates.tolist()
    n, p = np.array(n), np.array(p)
    y = rng.uniform(0.5, 3.0, n.size)
    se = rng.uniform(0.01, 0.2, n.size)
    data = fss.ScalingData(n=n, p=p, y=y, se=se)
    for _ in range(5):
        pc, z, nu = rng.uniform(0.0, 0.4), rng.uniform(-1, 2), rng.uniform(0.5, 3)
        cc = fss.collapse_cost(data, pc, z, nu)
        ref = _reference_chi2(n, p, y, se, pc, z, nu)
        assert np.array_equal(np.isnan(cc.chi2), np.isnan(ref))
        ok = ~np.isnan(ref)
        assert np.allclose(cc.chi2[ok], ref[ok], rtol=1e-8, atol=1e-12)
        assert cc.points_used == ok.sum() and cc.total_points == n.size
        if ok.any():
            assert cc.cost == pytest.approx(ref[ok].mean(), rel=1e-8)
        else:
            assert math.isnan(cc.cost)


def test_the_bracketing_rule_at_the_edges():
    # with p_c = 0, z = 0, nu = 1 the abscissa is x = p n, exact for these binary rates:
    # size 8 at x = 0, 1, 2; size 16 at x = 0, 3; size 32 with one point at x = 3
    n = np.array([8, 8, 8, 16, 16, 32])
    p = np.array([0.0, 0.125, 0.25, 0.0, 0.1875, 0.09375])
    y = np.array([1.0, 2.0, 3.0, 1.0, 4.0, 4.5])
    se = np.full(6, 0.5)
    cc = fss.collapse_cost(fss.ScalingData(n=n, p=p, y=y, se=se), 0.0, 0.0, 1.0)
    # size 8 (x = 0, 1, 2) lies inside size 16's [0, 3]: each point meets the line 1 + x
    # through size 16's two points, so chi2 = 0 (size 32's single point at x = 3 brackets
    # none of them). Size 16 at x = 0 meets size 8's first pair: chi2 = 0. Size 16 at x = 3
    # is above size 8's largest x = 2 but equals size 32's single point, which enters alone:
    # chi2 = (4 - 4.5)^2 / (1/4 + 1/4). Size 32 at x = 3 equals size 16's largest x, so size
    # 16's last two points give the line 1 + x, worth 4 there with variance 1/4 (the line
    # passes through a point): chi2 = (4.5 - 4)^2 / (1/4 + 1/4).
    assert cc.bracketed.tolist() == [True] * 6
    assert cc.chi2.tolist() == pytest.approx([0, 0, 0, 0, 0.5, 0.5], abs=1e-12)
    # no extrapolation: a size beyond the other's range brackets nothing
    apart = fss.ScalingData(
        n=np.array([8, 8, 16]), p=np.array([0.0, 0.125, 0.5]), y=np.ones(3), se=np.ones(3)
    )
    out = fss.collapse_cost(apart, 0.0, 0.0, 1.0)
    assert out.bracketed.tolist() == [False, False, False] and math.isnan(out.cost)


def test_a_linear_scaling_function_collapses_perfectly():
    # local lines are exact for a linear F, so the quality is 0 to rounding at the truth
    rates = np.linspace(0.15, 0.25, 11)
    data = _exact(f=lambda x: 1.0 + 0.4 * x, rates=rates)
    cc = fss.collapse_cost(data, *TRUTH)
    assert cc.cost < 1e-20 and cc.points_used >= data.size // 2
    assert fss.collapse_cost(data, 0.21, 1.0, 1.5).cost > 1.0


def test_the_quality_does_not_depend_on_the_units_of_y():
    data = _exact()
    scaled = fss.ScalingData(n=data.n, p=data.p, y=data.y * 1e6, se=data.se * 1e6)
    for params in [TRUTH, (0.21, 0.9, 1.3)]:
        assert fss.collapse_cost(scaled, *params).cost == pytest.approx(
            fss.collapse_cost(data, *params).cost, rel=1e-10
        )


def test_scaled_coordinates():
    data = _exact()
    x, y, dy = fss.scaled_coordinates(data, *TRUTH)
    assert np.allclose(x, (data.p - 0.2) * data.n ** (1 / 1.5))
    assert np.allclose(y, _f1(x)) and np.allclose(dy, 0.01 * y)
    with pytest.raises(ValueError, match="nu positive"):
        fss.scaled_coordinates(data, 0.2, 1.0, 0.0)
    with pytest.raises(ValueError, match="nu positive"):
        fss.collapse_cost(data, math.nan, 1.0, 1.0)


def test_extreme_parameters_give_no_quality_instead_of_overflow():
    data = _exact()
    cc = fss.collapse_cost(data, 0.2, -400.0, 1.0)  # n^400 overflows
    assert math.isnan(cc.cost) and cc.points_used == 0
    assert math.isnan(fss.collapse_cost(data, 0.2, 1.0, 1e-3).cost)  # n^1000


# ---------------------------------------------------------------------------------------------
# the bounded Nelder-Mead
# ---------------------------------------------------------------------------------------------


def _rosenbrock(v):
    return (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2


def _himmelblau(v):
    return (v[0] ** 2 + v[1] - 11) ** 2 + (v[0] + v[1] ** 2 - 7) ** 2


def _booth(v):
    return (v[0] + 2 * v[1] - 7) ** 2 + (2 * v[0] + v[1] - 5) ** 2


@pytest.mark.parametrize(
    ("fun", "bounds", "minimum"),
    [
        (_rosenbrock, [(-2, 2), (-1, 3)], (1.0, 1.0)),
        (_himmelblau, [(0, 5), (0, 5)], (3.0, 2.0)),  # the box keeps one of its four minima
        (_booth, [(-10, 10), (-10, 10)], (1.0, 3.0)),
        (
            lambda v: float(np.sum((v - np.array([0.3, -0.2, 0.7])) ** 2)),
            [(0, 1), (-1, 1), (0, 1)],
            (0.3, -0.2, 0.7),
        ),
    ],
)
def test_the_optimizer_finds_known_minima(fun, bounds, minimum):
    res = fss.minimize_bounded(fun, bounds, seed=1)
    assert np.allclose(res.x, minimum, atol=1e-6), res.x
    assert res.fun < 1e-10 and res.converged and not res.at_bound.any()
    assert res.starts == 9


def test_a_minimum_outside_the_box_lands_on_its_bounds():
    res = fss.minimize_bounded(lambda v: (v[0] - 3) ** 2 + (v[1] + 2) ** 2, [(0, 1), (0, 1)])
    assert np.allclose(res.x, (1.0, 0.0), atol=1e-8)
    assert res.at_bound.tolist() == [True, True]
    assert res.fun == pytest.approx(4 + 4, abs=1e-8)


def test_fixed_coordinates_and_infinite_regions():
    res = fss.minimize_bounded(lambda v: (v[0] - 0.7) ** 2 + (v[1] - 2) ** 2, [(0, 1), (5, 5)])
    assert res.x[1] == 5.0 and res.x[0] == pytest.approx(0.7, abs=1e-7)
    assert not res.at_bound[1]
    fixed = fss.minimize_bounded(lambda v: float(v.sum()), [(1, 1), (2, 2)])
    assert fixed.x.tolist() == [1.0, 2.0] and fixed.fun == 3.0 and fixed.evaluations == 1

    def walled(v):
        return math.inf if v[0] < 0.5 else (v[0] - 0.7) ** 2

    res = fss.minimize_bounded(walled, [(0, 1)], seed=3)
    assert res.x[0] == pytest.approx(0.7, abs=1e-7)
    nowhere = fss.minimize_bounded(lambda v: math.nan, [(0, 1)], starts=2)
    assert nowhere.fun == math.inf and not nowhere.converged


def test_the_optimizer_is_deterministic_given_its_seed():
    a = fss.minimize_bounded(_himmelblau, [(-5, 5), (-5, 5)], seed=7)
    b = fss.minimize_bounded(_himmelblau, [(-5, 5), (-5, 5)], seed=7)
    assert np.array_equal(a.x, b.x) and a.fun == b.fun and a.evaluations == b.evaluations
    # Himmelblau has four global minima in this box: any is a valid answer
    assert a.fun < 1e-10


def test_the_budget_is_reported():
    res = fss.minimize_bounded(_rosenbrock, [(-2, 2), (-1, 3)], starts=0, max_evaluations=20)
    assert not res.converged and res.evaluations <= 20 * 3 + 10


@pytest.mark.parametrize(
    ("bounds", "kwargs", "match"),
    [
        ([(0, 1), (2, 1)], {}, "lo <= hi"),
        ([(0, math.inf)], {}, "finite"),
        ([(0, 1, 2)], {}, "pairs"),
        ("ab", {}, "pairs of numbers|pairs"),
        ([(0, 1)], {"starts": -1}, "non-negative"),
        ([(0, 1)], {"max_evaluations": 0}, "positive"),
        ([(0, 1)], {"initial": [2.0]}, "outside the box"),
        ([(0, 1)], {"initial": [0.5, 0.5]}, "coordinates"),
    ],
)
def test_the_optimizer_refuses_bad_input(bounds, kwargs, match):
    with pytest.raises(ValueError, match=match):
        fss.minimize_bounded(lambda v: 0.0, bounds, **kwargs)


# ---------------------------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------------------------


def test_the_fit_recovers_the_parameters_of_exact_data():
    fit = fss.fit_collapse(_exact(), bounds=BOX)
    assert fit.status == "ok" and fit.usable and fit.message == ""
    # dense rates: the interpolation bias moves the optimum by far less than these bounds
    assert fit.p_c == pytest.approx(0.2, abs=1e-3)
    assert fit.z == pytest.approx(1.0, abs=1e-2)
    assert fit.nu == pytest.approx(1.5, abs=2e-2)
    assert fit.cost < 0.01 and fit.points_used >= fit.min_points
    assert fit.sizes == SIZES and fit.observable == "y" and fit.beta == "0.1"
    assert fit.bounds == ((0.1, 0.3), (0.0, 2.0), (0.5, 3.0))
    assert fit.min_points == math.ceil(164 / 2) and fit.min_points_setting is None
    assert fit.params.tolist() == [fit.p_c, fit.z, fit.nu]
    json.dumps(fit.to_dict())


def test_the_fit_recovers_the_parameters_within_its_bootstrap_errors():
    # 20 graphs, a 5% graph effect shared by the rates of a size and 3% rate noise. The
    # calibration of 07-finite-size-scaling.md §11 (not run here: 200 fits, 40 bootstraps of 40
    # replicates) gave fit standard deviations 0.013, 0.038, 0.086 for (p_c, z, nu), no visible
    # bias, and bootstrap errors of 1.04, 0.98 and 1.11 of them on average, with a 5th
    # percentile over datasets of 0.82, 0.76 and 0.69. Averaging P(|Z| > 4 r) over the measured
    # per-dataset ratios r gives a false alarm of about 2e-3 for "each parameter within 4
    # bootstrap errors" (it was about 1.2e-2 with the earlier local refits, whose errors were low).
    data = _hierarchical(seed=101)
    boot = fss.bootstrap_fit(data, bounds=BOX, replicates=40, seed=2, starts=4)
    assert boot.fit.status == "ok" and boot.failed == 0
    err = np.abs(boot.fit.params - np.array(TRUTH))
    assert np.all(err <= 4 * boot.se), (boot.fit.params, boot.se)
    assert np.all(boot.se > 0) and np.all(boot.lower < boot.upper)
    assert boot.level == 0.95 and boot.at_bound == 0
    assert boot.replicates.shape == (40, 3) and len(boot.statuses) == 40
    json.dumps(boot.to_dict())


def test_the_bootstrap_errors_match_the_true_spread_of_the_fits():
    # v1.4 §7.4 ("Analysis": interval coverage of synthetic controls). M = 80 datasets of a
    # small design (sizes 16, 32, 64; 9 rates; 20 graphs; known (p_c, z, nu)) give the true
    # spread s_F of each estimator: Houdayer-Hartmann, polynomial and their difference. The
    # first K = 12 are also bootstrapped by compare_methods (B = 19), whose replicates are those
    # of bootstrap_fit for each method (test_compare_methods_bootstraps_like_bootstrap_fit).
    # R = sqrt(mean over K of se^2) / s_F estimates the ratio of the bootstrap error to the true
    # one. Checked: 0.6 <= R <= 2 for each parameter of each method (a bootstrap more than 1.67
    # times too narrow, or twice too wide, fails), and R >= 0.6 for the difference. Measured on
    # 24 independent repeats of this test (other seeds): log R had means (-0.15, -0.13, -0.11)
    # (Houdayer-Hartmann), (-0.07, -0.07, 0.07) (polynomial), (0.37, 0.47, 0.40) (difference:
    # conservative) and standard deviations 0.08-0.26; a normal model of log R puts the false
    # alarm at about 3e-3 (union over the nine bounds), and no repeat failed. Errors shrunk 2x
    # fail in every repeat for either method (only in 25% for the difference alone).
    rates = np.linspace(0.12, 0.28, 9)
    fits = {"hh": [], "poly": []}
    se = {"hh": [], "poly": [], "diff": []}
    for i in range(80):
        data = _hierarchical(sizes=(16, 32, 64), rates=rates, seed=9000 + i)
        boot = i < 12
        cm = fss.compare_methods(
            data, bounds=BOX, starts=0, replicates=19 if boot else 0, seed=7 + i
        )
        assert cm.houdayer_hartmann.usable and cm.polynomial.usable
        fits["hh"].append(cm.houdayer_hartmann.params)
        fits["poly"].append(cm.polynomial.params)
        if boot:
            assert cm.failed <= 2 and cm.reselect_window and cm.replicate_starts == 0
            se["hh"].append(cm.se_houdayer_hartmann)
            se["poly"].append(cm.se_polynomial)
            se["diff"].append(cm.se_difference)
    hh, poly = np.array(fits["hh"]), np.array(fits["poly"])
    spread = {
        "hh": hh.std(axis=0, ddof=1),
        "poly": poly.std(axis=0, ddof=1),
        "diff": (poly - hh).std(axis=0, ddof=1),
    }
    ratio = {k: np.sqrt(np.mean(np.array(v) ** 2, axis=0)) / spread[k] for k, v in se.items()}
    for k in ("hh", "poly"):
        assert np.all((ratio[k] >= 0.6) & (ratio[k] <= 2.0)), (k, ratio[k])
    assert np.all(ratio["diff"] >= 0.6), ratio["diff"]


def test_compare_methods_bootstraps_like_bootstrap_fit():
    # compare_methods resamples with bootstrap_fit's draws and fits each replicate by the same
    # procedure, so (without failed replicates) its errors are bootstrap_fit's
    data = _hierarchical(sizes=(16, 32, 64), rates=np.linspace(0.12, 0.28, 9), seed=8)
    cm = fss.compare_methods(data, bounds=BOX, starts=1, replicates=4, seed=6)
    hh = fss.bootstrap_fit(data, bounds=BOX, starts=1, replicates=4, seed=6)
    poly = fss.bootstrap_fit(data, bounds=BOX, method="polynomial", starts=1, replicates=4, seed=6)
    assert cm.failed == 0 and hh.failed == 0 and poly.failed == 0
    assert np.array_equal(cm.se_houdayer_hartmann, hh.se)
    assert np.array_equal(cm.se_polynomial, poly.se)
    assert np.allclose(
        cm.se_difference, (poly.replicates - hh.replicates).std(axis=0, ddof=1), rtol=1e-12
    )


def test_the_bootstrap_resamples_graphs_once_per_size_for_every_rate():
    data = _hierarchical(sizes=(16, 32), rates=np.linspace(0.15, 0.25, 3), graphs=6, seed=3)
    draws = fss._draws_by_size([data], 25, seed=4)
    assert sorted(draws) == [16, 32]
    labels, d16 = draws[16]
    assert np.array_equal(d16, st.graph_bootstrap(labels, 25, seed=[4, 16]))
    ystar, sestar = fss._replicate_values(data, draws)
    for i in range(data.size):
        lab, d = draws[int(data.n[i])]
        expected = data.graph_means[i][np.searchsorted(lab, d)].mean(axis=1)
        assert np.allclose(ystar[:, i], expected, rtol=1e-14)
    # pairing: the rates of a size share the draws, so a constant offset between two rates'
    # graph means survives every replicate exactly
    shifted = list(data.graph_means)
    shifted[1] = shifted[0] + 0.5
    paired = fss.ScalingData(
        n=data.n,
        p=data.p,
        y=np.array([gm.mean() for gm in shifted]),
        se=data.se,
        graphs=data.graphs,
        graph_means=tuple(shifted),
    )
    ys, _ = fss._replicate_values(paired, draws)
    assert np.allclose(ys[:, 1] - ys[:, 0], 0.5, rtol=0, atol=1e-12)


def _replicate_data(data, replicates, seed):
    """The bootstrap's replicate data sets, built from its documented draws (one per size)."""
    ystar, sestar = fss._replicate_values(data, fss._draws_by_size([data], replicates, seed))
    return [
        fss.ScalingData(n=data.n, p=data.p, y=ystar[b], se=sestar[b]) for b in range(replicates)
    ]


def test_every_bootstrap_replicate_is_fitted_like_the_data():
    # the bootstrap describes the estimator only if each replicate goes through the same
    # procedure: fit_collapse with the same bounds, starts and seed (the bootstrap's seed draws
    # both the graphs and the fit's random starts) on the replicate's values,
    # not a local search from the data's fit (which stays near it on the rough Houdayer-Hartmann
    # quality and makes the errors too small, 07 §11)
    data = _hierarchical(sizes=(16, 32, 64), rates=np.linspace(0.12, 0.28, 9), seed=8)
    boot = fss.bootstrap_fit(data, bounds=BOX, replicates=6, seed=4, starts=2)
    assert boot.replicate_starts == 2 and boot.reselect_window is None and boot.selected is None
    for b, rep in enumerate(_replicate_data(data, 6, 4)):
        again = fss.fit_collapse(rep, bounds=BOX, starts=2, seed=4)
        assert np.array_equal(boot.replicates[b], again.params), b
        assert boot.costs[b] == again.cost and boot.statuses[b] == again.status
    # fewer random starts per replicate when asked (an approximation of the procedure)
    cheap = fss.bootstrap_fit(data, bounds=BOX, replicates=3, seed=4, starts=2, replicate_starts=0)
    assert cheap.replicate_starts == 0
    for b, rep in enumerate(_replicate_data(data, 3, 4)):
        assert np.array_equal(
            cheap.replicates[b], fss.fit_collapse(rep, bounds=BOX, starts=0, seed=4).params
        )


def test_polynomial_replicates_draw_their_own_windows():
    # ruling R-FSS (design §11): the polynomial bootstrap re-selects the window in every
    # replicate, as the fit did for the data (first drawn at the replicate's Houdayer-Hartmann
    # fit, redrawn at its own fit until it settles), so the window's variability is in the error
    data = _hierarchical(seed=100)
    boot = fss.bootstrap_fit(data, bounds=BOX, method="polynomial", replicates=6, seed=3, starts=0)
    assert boot.reselect_window is True and boot.selected.shape == (6, data.size)
    for b, rep in enumerate(_replicate_data(data, 6, 3)):
        again = fss.fit_collapse(rep, bounds=BOX, method="polynomial", starts=0, seed=3)
        assert np.array_equal(boot.replicates[b], again.params), b
        assert np.array_equal(boot.selected[b], again.selected)
        assert boot.window_stable[b] == again.window_stable
        if again.window_stable:  # the replicate's window is the one drawn at its own fit
            assert np.array_equal(fss.window_points(data, *again.params), boot.selected[b])
    moved = [b for b in range(6) if not np.array_equal(boot.selected[b], boot.fit.selected)]
    assert moved, "every replicate kept the data fit's window"
    # reselect_window=False keeps the data fit's window in every replicate (never redrawn)
    fixed = fss.bootstrap_fit(
        data, bounds=BOX, method="polynomial", replicates=3, seed=3, starts=0, reselect_window=False
    )
    assert fixed.reselect_window is False and fixed.window_stable is None
    assert all(np.array_equal(fixed.selected[b], fixed.fit.selected) for b in range(3))
    json.dumps(boot.to_dict())


def test_bootstrap_settings_are_checked():
    data = _hierarchical(sizes=(16, 32), rates=np.linspace(0.15, 0.25, 3), graphs=6, seed=3)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="replicate_starts"):
            fss.bootstrap_fit(data, bounds=BOX, replicates=2, replicate_starts=bad)
        with pytest.raises(ValueError, match="replicate_starts"):
            fss.compare_methods(data, bounds=BOX, replicates=2, replicate_starts=bad)
    for bad in ("yes", 1, None):
        with pytest.raises(ValueError, match="reselect_window"):
            fss.bootstrap_fit(data, bounds=BOX, replicates=2, reselect_window=bad)
    other = _hierarchical(sizes=(16, 32), rates=np.linspace(0.15, 0.25, 3), graphs=6, seed=4)
    other = fss.ScalingData(**{**vars(other), "observable": "b"})
    with pytest.raises(ValueError, match="replicate_starts"):
        fss.compare_shared_independent([data, other], bounds=BOX, replicates=2, replicate_starts=-2)
    with pytest.raises(ValueError, match="reselect_window"):
        fss.compare_shared_independent([data, other], bounds=BOX, replicates=2, reselect_window=0)


def test_bootstrap_points_must_share_their_graphs():
    data = _hierarchical(sizes=(16, 32), rates=np.linspace(0.15, 0.25, 3), graphs=6, seed=3)
    graphs = list(data.graphs)
    graphs[1] = graphs[1] + 100
    broken = fss.ScalingData(
        n=data.n, p=data.p, y=data.y, se=data.se, graphs=tuple(graphs), graph_means=data.graph_means
    )
    with pytest.raises(ValueError, match="do not have the same graphs"):
        fss.bootstrap_fit(broken, bounds=BOX, replicates=5)
    bare = fss.ScalingData(n=data.n, p=data.p, y=data.y, se=data.se)
    with pytest.raises(ValueError, match="per-graph values"):
        fss.bootstrap_fit(bare, bounds=BOX, replicates=5)


def test_fits_are_deterministic_given_the_seed():
    data = _hierarchical(seed=5)
    a = fss.fit_collapse(data, bounds=BOX, starts=3, seed=11)
    b = fss.fit_collapse(data, bounds=BOX, starts=3, seed=11)
    assert a.params.tolist() == b.params.tolist() and a.cost == b.cost
    boot1 = fss.bootstrap_fit(data, bounds=BOX, replicates=4, seed=1, starts=1)
    boot2 = fss.bootstrap_fit(data, bounds=BOX, replicates=4, seed=1, starts=1)
    assert np.array_equal(boot1.replicates, boot2.replicates)


def test_the_fit_does_not_depend_on_the_units_or_on_duplicated_points():
    data = _exact(rates=np.linspace(0.1, 0.3, 21))
    base = fss.fit_collapse(data, bounds=BOX, starts=3)
    scaled = fss.ScalingData(n=data.n, p=data.p, y=data.y * 1e8, se=data.se * 1e8)
    big = fss.fit_collapse(scaled, bounds=BOX, starts=3)
    assert np.allclose(big.params, base.params, rtol=1e-6, atol=1e-8)
    twice = fss.ScalingData(
        n=np.tile(data.n, 2), p=np.tile(data.p, 2), y=np.tile(data.y, 2), se=np.tile(data.se, 2)
    )
    dup = fss.fit_collapse(twice, bounds=BOX, starts=3)
    assert dup.status == "ok"
    assert dup.p_c == pytest.approx(0.2, abs=2e-3) and dup.nu == pytest.approx(1.5, abs=5e-2)


def test_a_fixed_parameter_stays_fixed():
    fit = fss.fit_collapse(_exact(), bounds={"p_c": (0.1, 0.3), "z": (1.0, 1.0), "nu": (0.5, 3.0)})
    assert fit.z == 1.0 and fit.p_c == pytest.approx(0.2, abs=1e-3)
    assert fit.status == "ok" and "z" not in fit.at_bound


def test_a_truth_outside_the_box_is_reported_at_its_bound():
    fit = fss.fit_collapse(_exact(), bounds={"p_c": (0.1, 0.3), "z": (0.0, 2.0), "nu": (2.0, 3.0)})
    assert fit.status == "at_bound" and fit.at_bound == ("nu",) and "widen" in fit.message
    assert fit.nu == pytest.approx(2.0, abs=1e-6) and fit.usable


def test_too_few_sizes_or_points_give_a_status_not_a_crash():
    one = _exact(sizes=(32,))
    fit = fss.fit_collapse(one, bounds=BOX)
    assert fit.status == "too_few_sizes" and not fit.usable and math.isnan(fit.p_c)
    assert "at least two sizes" in fit.message
    # two sizes whose scaled ranges cannot overlap anywhere in the box
    apart = fss.ScalingData(
        n=np.array([10] * 4 + [1000] * 4),
        p=np.array([0.1, 0.13, 0.16, 0.2, 0.5, 0.53, 0.56, 0.6]),
        y=np.ones(8),
        se=np.full(8, 0.1),
    )
    box = {"p_c": (0.0, 0.05), "z": (0.0, 1.0), "nu": (0.9, 1.0)}
    fit = fss.fit_collapse(apart, bounds=box)
    assert fit.status == "too_few_points" and "do not overlap" in fit.message
    assert math.isnan(fit.cost) and fit.points_used == 0
    tiny = fss.fit_collapse(_exact(sizes=(16, 32), rates=np.array([0.2])), bounds=BOX)
    assert tiny.status == "too_few_points" and "fewer points" in tiny.message
    # the dependent tools report the same instead of failing
    loso = fss.leave_one_size_out(_exact(sizes=(16, 32)), bounds=BOX, starts=1)
    assert [f.status for f in loso.fits] == ["too_few_sizes", "too_few_sizes"]
    assert np.isnan(loso.held_out_cost).all() and np.isnan(loso.spread).all()
    diag = fss.collapse_diagnostics(apart, fit)
    assert diag.coverage == 0.0 and "no parameters" in diag.notes[0]


def test_min_points_is_respected():
    data = _exact(rates=np.linspace(0.1, 0.3, 11))
    fit = fss.fit_collapse(data, bounds=BOX, min_points=40, starts=2)
    assert fit.min_points == 40 and fit.min_points_setting == 40
    assert fit.points_used >= 40
    strict = fss.fit_collapse(data, bounds=BOX, min_points=44, starts=2)
    assert strict.status == "too_few_points"
    with pytest.raises(ValueError, match="min_points"):
        fss.fit_collapse(data, bounds=BOX, min_points=0)
    with pytest.raises(ValueError, match="min_points"):
        fss.fit_collapse(data, bounds=BOX, min_points=2.5)


@pytest.mark.parametrize(
    ("bounds", "match"),
    [
        ({"p_c": (0, 1), "z": (0, 1)}, "missing \\['nu'\\]"),
        ({"p_c": (0, 1), "z": (0, 1), "nu": (0.5, 1), "a": (0, 1)}, "unknown \\['a'\\]"),
        ({"p_c": (0, 1), "z": (0, 1), "nu": (0.0, 1)}, "nu must be positive"),
        ([(0, 1), (0, 1)], "3 \\(lo, hi\\) pairs"),
    ],
)
def test_fit_bounds_are_checked(bounds, match):
    with pytest.raises(ValueError, match=match):
        fss.fit_collapse(_exact(), bounds=bounds)


def test_fit_settings_are_checked():
    data = _exact()
    with pytest.raises(ValueError, match="starts"):
        fss.fit_collapse(data, bounds=BOX, starts=-1)
    with pytest.raises(ValueError, match="max_evaluations"):
        fss.fit_collapse(data, bounds=BOX, max_evaluations=0)
    with pytest.raises(ValueError, match="max_evaluations"):
        fss.fit_collapse(data, bounds=BOX, max_evaluations=1.5)
    with pytest.raises(ValueError, match="seed"):
        fss.fit_collapse(data, bounds=BOX, seed=-3)


# ---------------------------------------------------------------------------------------------
# shared exponents
# ---------------------------------------------------------------------------------------------


def test_the_shared_fit_recovers_a_common_set():
    a = _exact(f=_f1, name="tau1")
    b = _exact(f=_f2, name="T_cyc")
    shared = fss.fit_shared([a, b], bounds=BOX)
    assert shared.status == "ok" and shared.observables == ("tau1", "T_cyc")
    assert shared.p_c == pytest.approx(0.2, abs=1e-3)
    assert shared.z == pytest.approx(1.0, abs=1e-2)
    assert shared.nu == pytest.approx(1.5, abs=2e-2)
    assert max(shared.costs) < 0.01 and len(shared.points_used) == 2
    json.dumps(shared.to_dict())


def test_the_comparison_prefers_independent_exponents_for_different_nu():
    # nu = 1.2 against 2.5, 2% graph effects: in development the statistic was 5.1 against a
    # null whose largest of 39 replicates was 0.52; with 19 replicates the p-value is 1/20
    rates = np.linspace(0.12, 0.28, 17)
    a = _hierarchical((0.2, 1.0, 1.2), _f1, rates=rates, sg=0.02, sp=0.02, seed=11, name="a")
    b = _hierarchical((0.2, 1.0, 2.5), _f2, rates=rates, sg=0.02, sp=0.02, seed=12, name="b")
    cmp = fss.compare_shared_independent([a, b], bounds=BOX, replicates=19, seed=5, starts=4)
    assert cmp.prefers == "independent" and cmp.p_value == pytest.approx(1 / 20)
    assert cmp.delta > 4 * np.nanmax(cmp.null) and cmp.failed == 0
    assert cmp.independent[0].nu == pytest.approx(1.2, abs=0.15)
    assert cmp.independent[1].nu == pytest.approx(2.5, abs=0.4)
    assert cmp.delta == pytest.approx(cmp.shared.cost - cmp.independent_cost)
    json.dumps(cmp.to_dict())


@pytest.mark.parametrize("method", ["houdayer_hartmann", "polynomial"])
def test_the_comparison_keeps_its_size_when_the_parameters_are_shared(method):
    # negative control: K = 4 pairs of observables generated with the same (p_c, z, nu) and
    # different scaling functions, so v1.4 §5.9's shared hypothesis holds. If the bootstrap null
    # were exact (delta exchangeable with its B replicates), each p-value would be uniform on
    # {1/(B+1), ..., 1}, so at least a uniform U on (0, 1]; a conservative null makes it larger.
    # The pairs are independent (own data, own draws), so P(sum p <= s) <= s^K / K! for s <= 1
    # (Irwin-Hall). The mean p-value must be at least 0.15: false alarm at most
    # 0.6^4 / 24 = 5.4e-3 under a valid null. In development the p-values were (0.8, 0.9, 0.2,
    # 0.7) with Houdayer-Hartmann. A degenerate null (every delta* = 0) puts every p-value at
    # the smallest, 1/(B+1) = 0.1, and fails; a null shrunk tenfold gave a mean of 0.27 and
    # passes, so test_null_replicates_are_the_comparison_on_mapped_resampled_data checks the
    # null replicates exactly.
    rates = np.linspace(0.12, 0.28, 9)
    sizes = (16, 32, 64)
    pvalues = []
    for k in range(4):
        a = _hierarchical(TRUTH, _f1, sizes=sizes, rates=rates, seed=300 + k, name="a")
        b = _hierarchical(TRUTH, _f2, sizes=sizes, rates=rates, seed=400 + k, name="b")
        cmp = fss.compare_shared_independent(
            [a, b], bounds=BOX, method=method, replicates=9, seed=50 + k, starts=0
        )
        assert cmp.failed <= 1 and cmp.prefers in ("shared", "independent")
        pvalues.append(cmp.p_value)
    assert np.mean(pvalues) >= 0.15, pvalues


@pytest.mark.parametrize("method", ["houdayer_hartmann", "polynomial"])
def test_null_replicates_are_the_comparison_on_mapped_resampled_data(method):
    # 07 §8.3, written out here with the public fits: replicate b maps each observable onto the
    # shared parameters (p' = p_c^s + (p - p_c^o) n^(1/nu_o - 1/nu_s), y' = y n^(z_s - z_o)),
    # resamples its graphs with graph_bootstrap(labels_n, B, [seed, n]) (the same draws for both
    # observables and every rate), and computes delta* exactly as delta: fit_shared and
    # fit_collapse with the comparison's settings. A null pushed down in any way (scaled,
    # frozen at the shared parameters, a wrong mapping or draw) changes some replicate.
    rates = np.linspace(0.12, 0.28, 9)
    sizes = (16, 32, 64)
    a = _hierarchical(TRUTH, _f1, sizes=sizes, rates=rates, graphs=8, seed=301, name="a")
    b = _hierarchical(TRUTH, _f2, sizes=sizes, rates=rates, graphs=8, seed=401, name="b")
    reps, seed, starts = 3, 9, 1
    cmp = fss.compare_shared_independent(
        [a, b], bounds=BOX, method=method, replicates=reps, seed=seed, starts=starts
    )
    assert cmp.failed == 0 and cmp.replicate_starts == starts
    theta_s = cmp.shared.params
    for r in range(reps):
        mapped = []
        for d, own in zip((a, b), cmp.independent, strict=True):
            n = d.n.astype(np.float64)
            factor = n ** (theta_s[1] - own.z)
            p_new = theta_s[0] + (d.p - own.p_c) * n ** (1.0 / own.nu - 1.0 / theta_s[2])
            y, se = np.empty(d.size), np.empty(d.size)
            for i in range(d.size):
                draw = st.graph_bootstrap(d.graphs[i], reps, seed=[seed, int(d.n[i])])[r]
                vals = d.graph_means[i][np.searchsorted(d.graphs[i], draw)]
                y[i] = vals.sum() / vals.size
                dev = vals - y[i]
                se[i] = np.sqrt((dev * dev).sum() / (vals.size * (vals.size - 1)))
            mapped.append(
                fss.ScalingData(
                    n=d.n, p=p_new, y=y * factor, se=se * factor, observable=d.observable
                )
            )
        alone = [
            fss.fit_collapse(m, bounds=BOX, method=method, starts=starts, seed=seed) for m in mapped
        ]
        shared = fss.fit_shared(mapped, bounds=BOX, method=method, starts=starts, seed=seed)
        if method == "polynomial":
            delta = shared.chi2 - sum(f.chi2 for f in alone)
        else:
            pooled = sum(f.cost * f.points_used for f in alone) / sum(f.points_used for f in alone)
            delta = shared.cost - pooled
        assert cmp.null[r] == pytest.approx(delta, rel=1e-12, abs=1e-12), (r, cmp.null[r], delta)


def _mapped(data, own, shared):
    n = data.n.astype(float)
    factor = n ** (shared[1] - own[1])
    return fss.ScalingData(
        n=data.n,
        p=shared[0] + (data.p - own[0]) * n ** (1 / own[2] - 1 / shared[2]),
        y=data.y * factor,
        se=data.se * factor,
    )


def test_the_null_mapping_moves_each_observable_onto_the_shared_parameters():
    # mapped data have, at the shared parameters, the scaled coordinates the observable has at
    # its own: the collapse quality is unchanged (checked at generic parameters, away from the
    # jumps of the quality, where rounding of the coordinates could switch a bracket) ...
    a = _hierarchical(seed=21, name="a")
    shared = np.array([0.21, 0.95, 1.4])
    generic = np.array([0.1937, 1.031, 1.462])
    mapped = _mapped(a, generic, shared)
    x0, y0, _ = fss.scaled_coordinates(a, *generic)
    x1, y1, _ = fss.scaled_coordinates(mapped, *shared)
    assert np.allclose(x1, x0, rtol=1e-12, atol=1e-12) and np.allclose(y1, y0, rtol=1e-12)
    assert fss.collapse_cost(mapped, *shared).cost == pytest.approx(
        fss.collapse_cost(a, *generic).cost, rel=1e-9
    )
    # ... along the slice p_c = p_c^s the map (z, nu) -> (z + z_o - z_s, nu'), with
    # 1/nu' = 1/nu + 1/nu_o - 1/nu_s, carries the mapped quality onto the original one at other
    # (z, nu) too (Proposition 6's second claim; polynomial chi2 on a fixed set as well) ...
    sel = fss.polynomial_cost(a, *generic).selected
    for z, nu in [(0.9, 1.3), (1.1, 1.7), (0.97, 1.52)]:
        z_o = z + generic[1] - shared[1]
        nu_o = 1.0 / (1.0 / nu + 1.0 / generic[2] - 1.0 / shared[2])
        assert fss.collapse_cost(mapped, shared[0], z, nu).cost == pytest.approx(
            fss.collapse_cost(a, generic[0], z_o, nu_o).cost, rel=1e-9
        )
        assert fss.polynomial_cost(mapped, shared[0], z, nu, selected=sel).chi2 == pytest.approx(
            fss.polynomial_cost(a, generic[0], z_o, nu_o, selected=sel).chi2, rel=1e-9
        )
    # ... and when the observable's own parameters are its fit, the refit of the mapped data
    # lands near the shared set, not exactly on it: off that slice the mapped data are no
    # reparametrization of the original ones (nu_o != nu_s), so with noise their own optimum
    # moves a little (07 §8.3; here by less than 2e-3, against a bootstrap error near 0.013 for p_c)
    own = fss.fit_collapse(a, bounds=BOX, starts=3)
    refit = fss.fit_collapse(_mapped(a, own.params, shared), bounds=BOX, initial=shared, starts=0)
    assert np.allclose(refit.params, shared, atol=2e-3)
    assert refit.cost == pytest.approx(own.cost, rel=0.05)


def test_shared_tools_check_their_inputs():
    a = _exact(name="a")
    with pytest.raises(ValueError, match="at least two observables"):
        fss.fit_shared([a], bounds=BOX)
    with pytest.raises(ValueError, match="distinct"):
        fss.fit_shared([a, a], bounds=BOX)
    other = fss.ScalingData(n=a.n, p=a.p, y=a.y, se=a.se, observable="b", beta="0.2")
    with pytest.raises(ValueError, match="different beta"):
        fss.fit_shared([a, other], bounds=BOX)
    with pytest.raises(ValueError, match="ScalingData"):
        fss.fit_shared([a, 1.0], bounds=BOX)
    b = fss.ScalingData(n=a.n, p=a.p, y=a.y, se=a.se, observable="b", beta="0.1")
    with pytest.raises(ValueError, match="per-graph values"):
        fss.compare_shared_independent([a, b], bounds=BOX, replicates=3)
    h1, h2 = _hierarchical(seed=1, name="h1"), _hierarchical(seed=2, name="h2")
    with pytest.raises(ValueError, match="alpha"):
        fss.compare_shared_independent([h1, h2], bounds=BOX, alpha=1.5)
    with pytest.raises(ValueError, match="replicates"):
        fss.compare_shared_independent([h1, h2], bounds=BOX, replicates=0)


def test_a_comparison_without_parameters_is_undetermined():
    one = _hierarchical(sizes=(32,), seed=1, name="a")
    two = _hierarchical(sizes=(32,), seed=2, name="b")
    cmp = fss.compare_shared_independent([one, two], bounds=BOX, replicates=3)
    assert cmp.prefers == "undetermined" and math.isnan(cmp.p_value)
    assert "no parameters" in cmp.message and cmp.failed == 3


# ---------------------------------------------------------------------------------------------
# leave one size out, diagnostics
# ---------------------------------------------------------------------------------------------


def test_leaving_one_size_out_of_exact_data_gives_the_same_parameters():
    loso = fss.leave_one_size_out(_exact(rates=np.linspace(0.1, 0.3, 31)), bounds=BOX, starts=3)
    assert loso.sizes == SIZES and loso.params.shape == (4, 3)
    assert all(f.status == "ok" for f in loso.fits)
    assert np.allclose(loso.params[:, 0], 0.2, atol=2e-3)
    assert np.allclose(loso.params[:, 1], 1.0, atol=2e-2)
    assert np.allclose(loso.params[:, 2], 1.5, atol=5e-2)
    assert np.all(loso.spread < [3e-3, 3e-2, 8e-2])
    # each left-out size collapses onto the others within its 1% errors
    assert np.all(loso.held_out_cost < 0.05) and np.all(loso.held_out_points > 0)
    json.dumps(loso.to_dict())


def test_diagnostics_report_coverage_drift_and_the_fractions_passed_through():
    data = _exact(rates=np.linspace(0.1, 0.3, 21))
    censored = np.where(data.p < 0.15, 0.3, 0.0)
    horizon = np.where(data.n == 128, 0.1, 0.0)
    marked = fss.ScalingData(
        n=data.n,
        p=data.p,
        y=data.y,
        se=data.se,
        estimator="restricted mean to T_max",
        label="confirmatory",
        censored_fraction=censored,
        horizon_fraction=horizon,
    )
    fit = fss.fit_collapse(marked, bounds=BOX, starts=3)
    diag = fss.collapse_diagnostics(marked, fit)
    cc = fss.collapse_cost(marked, fit.p_c, fit.z, fit.nu)
    assert np.array_equal(diag.bracketed, cc.bracketed)
    assert diag.coverage == pytest.approx(cc.points_used / marked.size)
    assert diag.coverage_by_size[16] == 1.0 and diag.coverage_by_size[128] < 1.0
    assert diag.drift_fit is not None and diag.drift_fit.sizes == (32, 64, 128)
    assert np.allclose(diag.drift, diag.drift_fit.params - fit.params)
    assert np.all(np.abs(diag.drift) < [3e-3, 3e-2, 8e-2])
    assert diag.max_censored_fraction == 0.3 and diag.max_horizon_fraction == 0.1
    assert np.array_equal(diag.censored_fraction, censored)
    assert diag.label == "confirmatory" and diag.estimator == "restricted mean to T_max"
    text = " ".join(diag.notes)
    assert "censored fractions up to 0.3" in text and "horizon" in text and "not bracketed" in text
    json.dumps(diag.to_dict())


def test_drift_in_units_of_the_bootstrap_errors():
    data = _hierarchical(seed=7)
    boot = fss.bootstrap_fit(data, bounds=BOX, replicates=6, seed=1, starts=2)
    diag = fss.collapse_diagnostics(data, boot.fit, bootstrap=boot, starts=1)
    assert np.allclose(diag.drift_in_se, diag.drift / boot.se)
    small = fss.collapse_diagnostics(
        _exact(sizes=(16, 32)), fss.fit_collapse(_exact(sizes=(16, 32)), bounds=BOX, starts=1)
    )
    assert small.drift is None and "fewer than three sizes" in " ".join(small.notes)
    with pytest.raises(ValueError, match="CollapseFit"):
        fss.collapse_diagnostics(data, boot)


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------


def test_scaling_data_from_graph_means():
    means = [
        st.mean_over_graphs([0, 1, 2], [1.0, 2.0, 3.0]),
        st.mean_over_graphs([0, 1, 2], [2.0, 2.0, 5.0]),
    ]
    data = fss.scaling_data([8, 8], [0.1, 0.2], means, beta=0.1, label="exploratory")
    assert data.y.tolist() == [2.0, 3.0] and data.beta == "0.1"
    assert data.se[0] == pytest.approx(float(means[0].se))
    assert data.graphs[0].tolist() == [0, 1, 2] and data.size == 2
    assert data.sizes.tolist() == [8]
    sub = data.subset([True, False])
    assert sub.size == 1 and sub.graphs[0].tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="one entry per point"):
        fss.scaling_data([8], [0.1, 0.2], means)
    with pytest.raises(ValueError, match="no defined mean"):
        fss.scaling_data([8], [0.1], [st.mean_over_graphs([0], [1.0])])
    with pytest.raises(ValueError, match="scalar GraphMean"):
        fss.scaling_data([8], [0.1], [st.mean_over_graphs([0, 1], np.ones((2, 2)))])
    with pytest.raises(ValueError, match="boolean mask"):
        data.subset([True])
    with pytest.raises(ValueError, match="no points"):
        data.subset([False, False])
    json.dumps(data.to_dict())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n": [[8]]}, "non-empty one-dimensional"),
        ({"n": [8.5]}, "integer system sizes"),
        ({"n": [0]}, "positive"),
        ({"p": [0.1, 0.2]}, "one entry per point"),
        ({"p": [math.inf]}, "p must be finite"),
        ({"y": [math.nan]}, "y must be finite"),
        ({"se": [0.0]}, "se must be positive"),
        ({"censored_fraction": [1.5]}, "in \\[0, 1\\]"),
        ({"graphs": ([0, 1],)}, "go together"),
        ({"graphs": ([0, 1], [0]), "graph_means": ([1.0, 2.0], [1.0])}, "one array per point"),
        ({"graphs": ([1, 0],), "graph_means": ([1.0, 2.0],)}, "increasing"),
        ({"graphs": ([0, 1],), "graph_means": ([1.0, np.inf],)}, "finite or NaN"),
        ({"graphs": ([0, 1],), "graph_means": ([1.0],)}, "one entry per point"),
    ],
)
def test_scaling_data_refuses_bad_input(kwargs, match):
    args = {"n": [8], "p": [0.1], "y": [1.0], "se": [0.1]}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        fss.ScalingData(**args)


def test_scaling_data_refuses_a_y_that_is_not_the_mean_of_its_graph_means():
    # the bootstrap recomputes every replicate as the mean of the drawn graph means, so the y it
    # fits must be that mean; a median (2 here) would be fitted as one estimator and resampled
    # as another
    labels = (np.array([0, 1, 2]),)
    values = (np.array([1.0, 2.0, 6.0]),)

    def make(y, gm=values):
        return fss.ScalingData(n=[8], p=[0.1], y=[y], se=[1.0], graphs=labels, graph_means=gm)

    assert make(3.0).y[0] == 3.0
    assert make(3.0 * (1 + 1e-15)).y[0] > 3.0  # rounding of another summation order is fine
    with pytest.raises(ValueError, match=r"not the mean of graph_means\[0\].*median"):
        make(2.0)
    # undefined graph means are left out, as nested_mean(nan_policy="omit") leaves them out
    assert make(1.5, (np.array([1.0, 2.0, np.nan]),)).y[0] == 1.5
    with pytest.raises(ValueError, match="no defined value"):
        make(1.0, (np.full(3, np.nan),))
    # data built from nested means are consistent by construction, NaN-omitting ones included
    g, c, r = np.repeat([0, 1, 2], 2), np.tile([0, 1], 3), np.zeros(6, dtype=int)
    m = st.nested_mean(g, c, r, np.array([1.0, np.nan, 2.0, 4.0, 0.5, 0.25]), nan_policy="omit")
    assert fss.scaling_data([8], [0.1], [m]).y[0] == float(m.mean)
    # without graph values any estimate can be fitted (no bootstrap)
    assert fss.ScalingData(n=[8], p=[0.1], y=[2.0], se=[1.0], estimator="KM median").y[0] == 2.0


# ---------------------------------------------------------------------------------------------
# the less common paths
# ---------------------------------------------------------------------------------------------


def test_a_spent_budget_is_reported_as_not_converged():
    fit = fss.fit_collapse(_exact(), bounds=BOX, starts=0, max_evaluations=30)
    assert fit.status == "not_converged" and fit.usable and "budget" in fit.message


def test_overflowing_scaled_values_give_no_quality():
    data = _exact()
    huge = fss.ScalingData(n=data.n, p=data.p, y=data.y * 1e300, se=data.se)
    assert math.isnan(fss.collapse_cost(huge, 0.2, -3.0, 1.5).cost)  # 3e302 * 128^3 overflows
    json.dumps(fss.collapse_cost(data, *TRUTH).to_dict())


def test_replicates_that_cannot_be_refitted_are_counted():
    # one graph per point: every replicate error is undefined, so no replicate has points
    data = _exact(rates=np.linspace(0.1, 0.3, 11), name="a")
    lone = fss.ScalingData(
        n=data.n,
        p=data.p,
        y=data.y,
        se=data.se,
        observable="a",
        beta="0.1",
        graphs=tuple(np.array([0]) for _ in range(data.size)),
        graph_means=tuple(np.array([v]) for v in data.y),
    )
    boot = fss.bootstrap_fit(lone, bounds=BOX, replicates=3, seed=(1, 2), starts=1)
    assert boot.fit.usable and boot.failed == 3
    assert boot.statuses == ("too_few_points",) * 3 and np.isnan(boot.se).all()
    other = fss.ScalingData(
        n=lone.n,
        p=lone.p,
        y=lone.y * 1.1,
        se=lone.se,
        observable="b",
        beta="0.1",
        graphs=lone.graphs,
        graph_means=tuple(gm * 1.1 for gm in lone.graph_means),
    )
    cmp = fss.compare_shared_independent([lone, other], bounds=BOX, replicates=3, starts=1)
    assert cmp.prefers == "undetermined" and cmp.failed == 3
    assert "every bootstrap refit failed" in cmp.message
    # the same with the polynomial method, and for the comparison of the two methods
    pboot = fss.bootstrap_fit(lone, bounds=BOX, method="polynomial", replicates=3, starts=1)
    assert pboot.fit.usable and pboot.failed == 3
    pcmp = fss.compare_shared_independent(
        [lone, other], bounds=BOX, method="polynomial", replicates=3, starts=1
    )
    assert pcmp.prefers == "undetermined" and pcmp.failed == 3
    methods = fss.compare_methods(lone, bounds=BOX, replicates=3, starts=1)
    assert methods.failed == 3 and np.isnan(methods.se_difference).all()
    single = fss.bootstrap_fit(_hierarchical(sizes=(32,), seed=1), bounds=BOX, replicates=2)
    assert single.statuses == ("too_few_sizes",) * 2 and single.failed == 2
    with pytest.raises(ValueError, match="replicates must be an integer >= 2"):
        fss.bootstrap_fit(lone, bounds=BOX, replicates=1)


def test_a_failed_drift_refit_is_noted():
    data = _exact(sizes=(16, 32, 64), rates=np.linspace(0.1, 0.3, 7))
    fit = fss.fit_collapse(data, bounds=BOX, min_points=15, starts=2)  # the subset has 14 points
    assert fit.usable
    diag = fss.collapse_diagnostics(data, fit)
    assert diag.drift is None and diag.drift_fit.status == "too_few_points"
    assert any("refit without n = 16 failed" in note for note in diag.notes)


def test_less_common_input_checks():
    with pytest.raises(ValueError, match="one-dimensional"):
        fss.ScalingData(n=[8], p=[[0.1]], y=[1.0], se=[0.1])
    assert fss.ScalingData(n=[8], p=[0.1], y=[1.0], se=[0.1], beta=0.25).beta == "0.25"
    with pytest.raises(ValueError, match="need parameter names"):
        fss.minimize_bounded(lambda v: 0.0, {"a": (0, 1)})


# ---------------------------------------------------------------------------------------------
# the polynomial master curve (ruling R-FSS)
# ---------------------------------------------------------------------------------------------


def _exact_least_squares(x, big_y, sigma, degree):
    """chi2 and coefficients of the weighted polynomial least squares, in exact rationals
    (normal equations solved by Gauss-Jordan elimination)."""
    m = degree + 1
    rows = [[xi**k / si for k in range(m)] for xi, si in zip(x, sigma, strict=True)]
    rhs = [yi / si for yi, si in zip(big_y, sigma, strict=True)]
    normal = [[sum(r[a] * r[b] for r in rows) for b in range(m)] for a in range(m)]
    vector = [sum(r[a] * v for r, v in zip(rows, rhs, strict=True)) for a in range(m)]
    for col in range(m):
        pivot = next(i for i in range(col, m) if normal[i][col] != 0)
        normal[col], normal[pivot] = normal[pivot], normal[col]
        vector[col], vector[pivot] = vector[pivot], vector[col]
        for i in range(m):
            if i != col and normal[i][col] != 0:
                f = normal[i][col] / normal[col][col]
                normal[i] = [a - f * b for a, b in zip(normal[i], normal[col], strict=True)]
                vector[i] -= f * vector[col]
    coef = [vector[i] / normal[i][i] for i in range(m)]
    resid = [v - sum(c * r[k] for k, c in enumerate(coef)) for r, v in zip(rows, rhs, strict=True)]
    return sum(e * e for e in resid), coef, resid


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_polynomial_cost_is_the_exact_weighted_least_squares(degree):
    # nu = 1 and z = 1: x = (p - p_c) n and Y = y / n are exact rationals of the float inputs
    rng = np.random.default_rng(degree)
    n = np.repeat([8, 16, 32], 5)
    p = np.tile(np.linspace(0.1, 0.3, 5), 3)
    y = rng.uniform(1, 30, n.size)
    se = rng.uniform(0.1, 1.0, n.size)
    data = fss.ScalingData(n=n, p=p, y=y, se=se)
    everything = np.ones(n.size, dtype=bool)
    cost = fss.polynomial_cost(data, 0.19, 1.0, 1.0, degree=degree, selected=everything)
    x = [(Fraction(float(pi)) - Fraction(0.19)) * int(ni) for pi, ni in zip(p, n, strict=True)]
    big_y = [Fraction(float(yi)) / int(ni) for yi, ni in zip(y, n, strict=True)]
    sigma = [Fraction(float(si)) / int(ni) for si, ni in zip(se, n, strict=True)]
    chi2, coef, resid = _exact_least_squares(x, big_y, sigma, degree)
    assert cost.chi2 == pytest.approx(float(chi2), rel=1e-10)
    assert np.allclose(cost.coefficients, [float(c) for c in coef], rtol=1e-8, atol=1e-10)
    assert np.allclose(cost.residuals, [float(e) for e in resid], rtol=1e-8, atol=1e-10)
    assert cost.dof == n.size - degree - 1 and cost.points_used == n.size
    assert cost.chi2_per_dof == pytest.approx(cost.chi2 / cost.dof)
    json.dumps(cost.to_dict())


def test_the_window_rules():
    data = _exact(rates=np.linspace(0.1, 0.3, 11))
    x = fss.scaled_coordinates(data, *TRUTH)[0]
    overlap = fss.window_points(data, *TRUTH)
    lo = max(x[data.n == s].min() for s in SIZES)
    hi = min(x[data.n == s].max() for s in SIZES)
    assert np.array_equal(overlap, (x >= lo) & (x <= hi))
    assert overlap[data.n == 16].all()  # the smallest size's range is the overlap here
    assert np.array_equal(fss.window_points(data, *TRUTH, window=1.0), np.abs(x) <= 1.0)
    # p_c far below every rate: the sizes' scaled ranges no longer overlap
    assert not fss.window_points(data, 0.0, 1.0, 0.6).any()
    for bad in ("all", 0.0, -1.0, math.inf, True):
        with pytest.raises(ValueError, match="window must be"):
            fss.window_points(data, *TRUTH, window=bad)


def test_the_polynomial_fit_recovers_exact_parameters():
    data = _exact(rates=np.linspace(0.1, 0.3, 21))
    for window in ("overlap", 1.0):
        fit = fss.fit_collapse(data, bounds=BOX, method="polynomial", window=window, starts=3)
        assert fit.status == "ok" and fit.method == "polynomial" and fit.degree == 3
        assert np.allclose(fit.params, TRUTH, atol=[1e-5, 1e-4, 1e-3]), fit.params
        assert fit.window_stable and fit.window_iterations >= 1
        assert fit.dof == fit.points_used - 4 - 3 and fit.chi2_per_dof == pytest.approx(fit.cost)
        assert fit.cost < 1e-3  # 1% errors and a nearly cubic F over the window
        # the window is self-consistent: drawn again at the fit, it is the set that was used
        again = fss.window_points(data, fit.p_c, fit.z, fit.nu, window=window)
        assert np.array_equal(again, fit.selected)
        assert fit.window_range[0] <= fit.window_range[1] and len(fit.coefficients) == 4
        json.dumps(fit.to_dict())


def test_the_polynomial_objective_is_continuous_where_the_quality_jumps():
    # on a fixed point set the objective is a smooth function of the parameters: steps of 1e-9
    # change it by at most about its local slope times the step, across a measured rate
    # (0.18), where the Houdayer-Hartmann quality jumps
    data = _hierarchical(seed=1)
    fit = fss.fit_collapse(data, bounds=BOX, method="polynomial", starts=3)

    def poly(pc, z=fit.z, nu=fit.nu):
        return fss.polynomial_cost(data, pc, z, nu, selected=fit.selected).chi2

    h = 1e-9
    slope = abs(poly(0.18 + 1e-5) - poly(0.18 - 1e-5)) / 2e-5
    steps = [abs(poly(0.18 + (k + 1) * h) - poly(0.18 + k * h)) for k in range(-3, 3)]
    assert max(steps) <= 2 * slope * h + 1e-12, (steps, slope)
    hh_jump = abs(
        fss.collapse_cost(data, 0.18 + h, fit.z, fit.nu).cost
        - fss.collapse_cost(data, 0.18 - h, fit.z, fit.nu).cost
    )
    assert hh_jump > 1e-3
    for dz, dnu in ((1e-9, 0.0), (0.0, 1e-9)):  # and in z and nu
        assert abs(poly(fit.p_c, fit.z + dz, fit.nu + dnu) - poly(fit.p_c)) < 1e-5


def test_the_degree_changes_the_answer_until_it_is_enough():
    # a steep master curve over a wide window (x_max = 2): a low degree cannot follow it, so
    # chi2/dof is large and nu biased; enough degree brings both back
    steep = _exact(f=lambda x: 2.0 / (1.0 + np.exp(3.0 * x)) + 0.5, rates=np.linspace(0.1, 0.3, 21))
    scan = fss.polynomial_degree_scan(
        steep, bounds=BOX, degrees=(2, 3, 9), window=2.0, initial=TRUTH, starts=2
    )
    assert scan.degrees == (2, 3, 9) and scan.params.shape == (3, 3)
    assert scan.chi2_per_dof[0] > 100 * scan.chi2_per_dof[2]
    assert abs(scan.params[0, 2] - 1.5) > 0.05  # degree 2: nu biased
    assert np.allclose(scan.params[2], TRUTH, atol=[1e-3, 1e-2, 2e-2])  # degree 9: back
    assert np.all(scan.spread >= 0) and scan.dof.tolist() == [f.dof for f in scan.fits]
    json.dumps(scan.to_dict())
    with pytest.raises(ValueError, match="degree must be"):
        fss.polynomial_degree_scan(steep, bounds=BOX, degrees=(0,))
    with pytest.raises(ValueError, match="must not be empty"):
        fss.polynomial_degree_scan(steep, bounds=BOX, degrees=())


def test_chi2_per_dof_is_about_one_for_independent_errors():
    # independent normal errors of known size (no graph effect): over 20 datasets the mean of
    # chi2/dof (about 40 degrees of freedom each) is 1 within 4 of its standard errors
    # (measured 0.94 +- 0.06). The standard error is estimated from the same 20 values, so the
    # statistic is about Student's t with 19 degrees of freedom: false alarm
    # P(|t_19| > 4) = 7.7e-4 if the window and the fit added nothing (not the normal 6e-5)
    n = np.repeat(SIZES, 21)
    p = np.tile(np.linspace(0.1, 0.3, 21), len(SIZES))
    truth = n * _f1((p - 0.2) * n ** (1 / 1.5))
    values = []
    for seed in range(20):
        rng = np.random.default_rng(500 + seed)
        se = 0.02 * truth
        data = fss.ScalingData(n=n, p=p, y=truth + rng.normal(0, 1, n.size) * se, se=se)
        fit = fss.fit_collapse(data, bounds=BOX, method="polynomial", initial=TRUTH, starts=2)
        values.append(fit.chi2_per_dof)
    values = np.array(values)
    assert abs(values.mean() - 1) < 4 * values.std(ddof=1) / math.sqrt(values.size)


def test_the_two_methods_agree_on_clean_and_on_noisy_data():
    clean = fss.compare_methods(_exact(rates=np.linspace(0.1, 0.3, 21)), bounds=BOX, replicates=0)
    assert clean.houdayer_hartmann.usable and clean.polynomial.usable
    assert np.all(np.abs(clean.difference) < [1e-3, 1e-2, 2e-2])
    assert np.isnan(clean.se_difference).all() and clean.replicates == 0
    noisy = fss.compare_methods(_hierarchical(seed=7), bounds=BOX, replicates=20, seed=3, starts=4)
    # in development z = (-0.75, 0.65, 0.79). se_difference is the standard deviation of 20
    # replicate differences, so with right errors and methods that agree in distribution each z
    # is about Student's t with 19 degrees of freedom: P(|t_19| > 4) = 7.7e-4 per parameter,
    # 2.3e-3 for the three (not the normal tail's 6e-5). The methods' nu biases differ a little
    # (07 §11); a mean shift of one paired error raises the nu term to about 6e-3.
    assert noisy.failed == 0 and np.all(np.abs(noisy.z) < 4), noisy.z
    # pairing: both methods fit the same resampled graphs, so their replicates are positively
    # correlated and the paired error is below the unpaired sqrt(se_hh^2 + se_poly^2) (it need
    # not be below each method's own error: the polynomial's window changes are not shared)
    unpaired = np.hypot(noisy.se_houdayer_hartmann, noisy.se_polynomial)
    assert np.all(noisy.se_difference < unpaired), (noisy.se_difference, unpaired)
    assert np.allclose(noisy.difference, noisy.polynomial.params - noisy.houdayer_hartmann.params)
    json.dumps(noisy.to_dict())


def test_too_few_points_for_the_polynomial_give_a_status():
    data = _exact(rates=np.linspace(0.1, 0.3, 7))
    # degree 9 needs 10 distinct rates of one size inside the window
    fit = fss.fit_collapse(data, bounds=BOX, method="polynomial", degree=9, starts=1)
    assert fit.status == "too_few_points" and "distinct rates" in fit.message
    assert not fit.usable and math.isnan(fit.cost)
    one = fss.fit_collapse(_exact(sizes=(32,)), bounds=BOX, method="polynomial")
    assert one.status == "too_few_sizes"
    # a window holding one size only: p_c far below the rates separates the sizes' ranges
    wide = {"p_c": (0.0, 0.3), "z": (0.0, 2.0), "nu": (0.5, 3.0)}
    apart = fss.fit_collapse(
        data, bounds=wide, method="polynomial", initial=[0.0, 1.0, 0.6], starts=0
    )
    assert apart.status == "too_few_points" and "two sizes" in apart.message
    # 2 sizes x 4 rates all in a wide window, a cubic and 3 parameters: one degree of freedom;
    # one point less: none
    tiny = _exact(sizes=(16, 32), rates=np.array([0.19, 0.195, 0.2, 0.205]))
    few = fss.fit_collapse(
        tiny, bounds=BOX, method="polynomial", window=1.0, initial=TRUTH, starts=0
    )
    assert few.status == "ok" and few.dof == 1 and few.points_used == 8
    fewer = fss.fit_collapse(
        tiny.subset((tiny.n == 16) | (tiny.p != 0.205)),
        bounds=BOX,
        method="polynomial",
        window=1.0,
        initial=TRUTH,
        starts=0,
    )
    assert fewer.status == "too_few_points" and "degrees of freedom" in fewer.message
    # without 4 distinct rates in any size the master curve is not determined
    flat = fss.fit_collapse(
        tiny.subset(tiny.p != 0.205),
        bounds=BOX,
        method="polynomial",
        window=1.0,
        initial=TRUTH,
        starts=0,
    )
    assert flat.status == "too_few_points" and "distinct rates" in flat.message


def test_the_polynomial_method_runs_through_every_tool():
    a = _exact(f=_f1, rates=np.linspace(0.1, 0.3, 21), name="tau1")
    b = _exact(f=_f2, rates=np.linspace(0.1, 0.3, 21), name="T_cyc")
    shared = fss.fit_shared([a, b], bounds=BOX, method="polynomial", starts=2)
    assert shared.status == "ok" and shared.method == "polynomial"
    assert np.allclose(shared.params, TRUTH, atol=[1e-4, 1e-3, 1e-2])
    assert len(shared.selected) == 2 and shared.dof > 0 and max(shared.costs) < 1e-3
    loso = fss.leave_one_size_out(a, bounds=BOX, method="polynomial", starts=2)
    assert all(f.method == "polynomial" for f in loso.fits)
    usable = [k for k, f in enumerate(loso.fits) if f.usable]
    assert usable and np.allclose(loso.params[usable], TRUTH, atol=[1e-3, 1e-2, 3e-2])
    held = loso.held_out_cost[usable]
    assert np.all(held[~np.isnan(held)] < 1.0)
    # these refits keep the full fit's window minus a size by design: they never redraw it, so
    # whether it would settle is not checked (None), not claimed (True)
    assert all(f.window_stable is None and f.window_iterations == 1 for f in loso.fits)
    fit = fss.fit_collapse(a, bounds=BOX, method="polynomial", starts=2)
    diag = fss.collapse_diagnostics(a, fit)
    assert np.array_equal(diag.bracketed, fit.selected)
    assert diag.coverage == pytest.approx(fit.points_used / a.size)
    assert np.isnan(diag.chi2[~fit.selected]).all() and diag.drift_fit.method == "polynomial"
    assert diag.drift_fit.window_stable is None and fit.window_stable is True
    assert "outside the polynomial window" in " ".join(diag.notes)
    assert json.loads(json.dumps(loso.to_dict()))["fits"][0]["window_stable"] is None
    rates = np.linspace(0.12, 0.28, 17)
    h1 = _hierarchical((0.2, 1.0, 1.2), _f1, rates=rates, sg=0.02, sp=0.02, seed=11, name="a")
    h2 = _hierarchical((0.2, 1.0, 2.5), _f2, rates=rates, sg=0.02, sp=0.02, seed=12, name="b")
    cmp = fss.compare_shared_independent(
        [h1, h2],
        bounds=BOX,
        method="polynomial",
        replicates=9,
        seed=5,
        starts=2,
        alpha=0.1,
        replicate_starts=0,
    )
    # in development: delta 181 against a null whose largest of 9 replicates was 30
    assert cmp.method == "polynomial" and cmp.failed == 0
    assert cmp.reselect_window is True and cmp.replicate_starts == 0
    assert cmp.prefers == "independent" and cmp.p_value == pytest.approx(1 / 10)
    assert cmp.delta == pytest.approx(cmp.shared.chi2 - sum(f.chi2 for f in cmp.independent))


def test_polynomial_settings_are_checked():
    data = _exact()
    with pytest.raises(ValueError, match="method must be one of"):
        fss.fit_collapse(data, bounds=BOX, method="spline")
    with pytest.raises(ValueError, match="degree must be"):
        fss.fit_collapse(data, bounds=BOX, method="polynomial", degree=11)
    with pytest.raises(ValueError, match="window must be"):
        fss.fit_collapse(data, bounds=BOX, method="polynomial", window="wide")
    with pytest.raises(ValueError, match="window_iterations"):
        fss.fit_collapse(data, bounds=BOX, method="polynomial", window_iterations=0)
    with pytest.raises(ValueError, match="boolean mask"):
        fss.polynomial_cost(data, *TRUTH, selected=[True])
    undefined = fss.polynomial_cost(data, 0.2, -400.0, 1.0)
    assert math.isnan(undefined.chi2) and undefined.coefficients is None


def test_polynomial_paths_without_a_usable_answer():
    data = _exact(rates=np.linspace(0.1, 0.3, 11))
    # n^(1/nu) overflows: the window is empty and the objective undefined
    assert not fss.window_points(data, 0.2, 1.0, 1e-3).any()
    assert math.isnan(fss.polynomial_cost(data, 0.2, 1.0, 1e-3).chi2)
    # n^(-z) overflows over the whole box: no parameters anywhere
    hopeless = {"p_c": (0.1, 0.3), "z": (-500.0, -400.0), "nu": (0.5, 3.0)}
    fit = fss.fit_collapse(data, bounds=hopeless, method="polynomial", initial=[0.2, -450, 1.5])
    assert fit.status == "too_few_points" and "undefined" in fit.message
    # a shared fit inherits the failure of an observable fitted alone
    other = _exact(sizes=(32,), name="lonely")
    shared = fss.fit_shared([data, other], bounds=BOX, method="polynomial", starts=1)
    assert shared.status == "too_few_sizes" and "lonely" in shared.message
    # user starts join the independent ones
    twin = _exact(f=_f2, rates=np.linspace(0.1, 0.3, 11), name="twin")
    started = fss.fit_shared([data, twin], bounds=BOX, method="polynomial", initial=TRUTH, starts=0)
    assert started.status == "ok" and np.allclose(started.params, TRUTH, atol=[1e-3, 1e-2, 3e-2])
    # leave-one-size-out when the full polynomial fit fails: each part is fitted afresh
    loso = fss.leave_one_size_out(
        _exact(sizes=(16, 32), rates=np.linspace(0.1, 0.3, 7)),
        bounds=BOX,
        method="polynomial",
        degree=9,
        starts=1,
    )
    assert not loso.full.usable
    assert [f.status for f in loso.fits] == ["too_few_sizes", "too_few_sizes"]
    # the methods side by side on data that cannot be fitted
    single = _hierarchical(sizes=(32,), seed=2)
    cmp = fss.compare_methods(single, bounds=BOX, replicates=3)
    assert cmp.failed == 3 and np.isnan(cmp.z).all()


def test_an_unsettled_window_is_reported():
    # one window fit from a reference far from the optimum: the window drawn again at the fit
    # differs from the one used, and the fit says so
    data = _exact(rates=np.linspace(0.1, 0.3, 21))
    fit = fss.fit_collapse(
        data,
        bounds=BOX,
        method="polynomial",
        initial=[0.26, 1.0, 1.5],
        starts=0,
        window_iterations=1,
    )
    assert fit.usable and fit.window_stable is False and fit.window_iterations == 1
    again = fss.window_points(data, fit.p_c, fit.z, fit.nu)
    assert not np.array_equal(again, fit.selected)
    diag = fss.collapse_diagnostics(data, fit)
    assert any("did not settle" in note for note in diag.notes)
    settled = fss.fit_collapse(
        data, bounds=BOX, method="polynomial", initial=[0.26, 1.0, 1.5], starts=0
    )
    assert settled.window_stable and settled.window_iterations > 1
    assert np.allclose(settled.params, TRUTH, atol=[1e-4, 1e-3, 1e-2])
