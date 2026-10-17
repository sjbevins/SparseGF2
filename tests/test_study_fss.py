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
    # 20 graphs, a 5% graph effect shared by the rates of a size and 3% rate noise. A Monte
    # Carlo over 40 such datasets (07-finite-size-scaling.md §10, not run here) gave fit standard
    # deviations 0.016, 0.043, 0.11 for (p_c, z, nu), no visible bias, and bootstrap errors
    # 0.85-0.91 of them. Each parameter within 4 bootstrap errors (at least 3.4 true ones):
    # false alarm about 2e-3 overall (more if these 40-replicate errors come out low).
    data = _hierarchical(seed=101)
    boot = fss.bootstrap_fit(data, bounds=BOX, replicates=40, seed=2, starts=4)
    assert boot.fit.status == "ok" and boot.failed == 0
    err = np.abs(boot.fit.params - np.array(TRUTH))
    assert np.all(err <= 4 * boot.se), (boot.fit.params, boot.se)
    assert np.all(boot.se > 0) and np.all(boot.lower < boot.upper)
    assert boot.level == 0.95 and boot.at_bound == 0
    assert boot.replicates.shape == (40, 3) and len(boot.statuses) == 40
    json.dumps(boot.to_dict())


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
        n=data.n, p=data.p, y=data.y, se=data.se, graphs=data.graphs, graph_means=tuple(shifted)
    )
    ys, _ = fss._replicate_values(paired, draws)
    assert np.allclose(ys[:, 1] - ys[:, 0], 0.5, rtol=0, atol=1e-12)


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
    # ... and when the observable's own parameters are its fit, the refit of the mapped data
    # lands on the shared set
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
        graph_means=lone.graph_means,
    )
    cmp = fss.compare_shared_independent([lone, other], bounds=BOX, replicates=3, starts=1)
    assert cmp.prefers == "undetermined" and cmp.failed == 3
    assert "every bootstrap refit failed" in cmp.message
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
