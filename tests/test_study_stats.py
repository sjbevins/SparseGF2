"""Tests of sparsegf2.study.stats (specification v1.4 §§5.8, 5.9, 6.3, 6.4).

The references are written here, independently of the production code: plain loops over
dictionaries for the nested estimator, exact Fractions for the Kaplan-Meier products, closed
forms and tables for Student's t, finite differences for the cluster influences. Every
statistical test states its tolerance and its false-alarm probability; seeds are fixed.
"""

import json
import math
from collections import defaultdict
from fractions import Fraction

import numpy as np
import pytest

from sparsegf2 import analysis
from sparsegf2.study import stats as st

# ---------------------------------------------------------------------------------------------
# independent references
# ---------------------------------------------------------------------------------------------


def _reference_nested(g, c, r, values):
    """Nested means by dictionaries: (mean, {g: f_g}, se), NaN values left out."""
    tree = defaultdict(lambda: defaultdict(list))
    for gi, ci, _ri, v in zip(g.tolist(), c.tolist(), r.tolist(), values.tolist(), strict=True):
        if not math.isnan(v):
            tree[gi][ci].append(v)
    graph_means = {}
    for gi, circuits in tree.items():
        circuit_means = [sum(vals) / len(vals) for vals in circuits.values() if vals]
        if circuit_means:
            graph_means[gi] = sum(circuit_means) / len(circuit_means)
    vals = list(graph_means.values())
    mean = sum(vals) / len(vals)
    gcount = len(vals)
    se = (
        math.sqrt(sum((v - mean) ** 2 for v in vals) / (gcount * (gcount - 1)))
        if gcount > 1
        else math.nan
    )
    return mean, graph_means, se


def _hierarchy(rng, graphs, max_circuits=4, max_replicas=3):
    """An unbalanced allocation: labels g, c, r (r up to max_replicas - 1)."""
    g, c, r = [], [], []
    for gi in range(graphs):
        for ci in range(1 + int(rng.integers(max_circuits))):
            for ri in range(1 + int(rng.integers(max_replicas))):
                g.append(gi)
                c.append(ci)
                r.append(ri)
    return np.array(g), np.array(c), np.array(r)


def _reference_km(times, observed, weights):
    """Exact (Fraction) weighted Kaplan-Meier: list of (t_j, S(t_j))."""
    w = [Fraction(x) for x in weights]
    out, s = [], Fraction(1)
    for tj in sorted({t for t, e in zip(times, observed, strict=True) if e}):
        at_risk = sum(wi for wi, t in zip(w, times, strict=True) if t >= tj)
        events = sum(wi for wi, t, e in zip(w, times, observed, strict=True) if e and t == tj)
        s *= 1 - events / at_risk
        out.append((tj, s))
    return out


def _chi2_sf_even(x, dof):
    """P(X >= x) for X chi-square with an even number of degrees of freedom."""
    half = x / 2.0
    term, total = 1.0, 1.0
    for k in range(1, dof // 2):
        term *= half / k
        total += term
    return math.exp(-half) * total


# ---------------------------------------------------------------------------------------------
# the nested estimator (v1.4 §6.3)
# ---------------------------------------------------------------------------------------------


def test_every_graph_has_equal_weight_whatever_its_allocation():
    # graph 0: ten circuits measuring 0; graph 1: one circuit measuring 1
    g = np.array([0] * 10 + [1])
    c = np.array(list(range(10)) + [0])
    r = np.zeros(11, dtype=int)
    v = np.array([0.0] * 10 + [1.0])
    res = st.nested_mean(g, c, r, v)
    assert float(res.mean) == 0.5  # the leaf mean would be 1/11
    assert np.mean(v) == pytest.approx(1 / 11)
    assert res.graphs.tolist() == [0, 1]
    assert res.graph_means.tolist() == [0.0, 1.0]
    assert res.circuits_per_graph.tolist() == [10, 1]
    assert res.trajectories_per_graph.tolist() == [10, 1]
    assert res.num_graphs == 2 and res.num_circuits == 11 and res.num_trajectories == 11
    assert res.replicas_per_circuit.tolist() == [1] * 11
    assert float(res.se) == 0.5  # sqrt(((0 - .5)^2 + (1 - .5)^2) / (2 * 1))


def test_replicas_are_averaged_before_circuits():
    # circuit (0, 0) has three replicas, circuit (0, 1) one: f_0 = (mean(1, 2, 3) + 10) / 2
    g = np.array([0, 0, 0, 0, 1])
    c = np.array([0, 0, 0, 1, 0])
    r = np.array([0, 1, 2, 0, 0])
    v = np.array([1.0, 2.0, 3.0, 10.0, 4.0])
    res = st.nested_mean(g, c, r, v)
    assert res.graph_means.tolist() == [6.0, 4.0]
    assert float(res.mean) == 5.0
    assert res.replicas_per_circuit.tolist() == [3, 1, 1]
    assert res.circuit_keys.tolist() == [[0, 0], [0, 1], [1, 0]]


def test_nested_mean_matches_a_loop_reference_on_unbalanced_hierarchies():
    rng = np.random.default_rng(11)
    for graphs in (2, 3, 17):
        g, c, r = _hierarchy(rng, graphs)
        v = rng.normal(size=g.size) + g * 0.3
        res = st.nested_mean(g, c, r, v)
        mean, graph_means, se = _reference_nested(g, c, r, v)
        assert float(res.mean) == pytest.approx(mean, rel=1e-13)
        assert float(res.se) == pytest.approx(se, rel=1e-12)
        assert res.graph_means.tolist() == pytest.approx(
            [graph_means[k] for k in sorted(graph_means)], rel=1e-13
        )


def test_a_balanced_allocation_gives_the_plain_mean():
    rng = np.random.default_rng(12)
    g = np.repeat(np.arange(6), 6)
    c = np.tile(np.repeat(np.arange(3), 2), 6)
    r = np.tile([0, 1], 18)
    v = rng.normal(size=36)
    assert float(st.nested_mean(g, c, r, v).mean) == pytest.approx(v.mean(), rel=1e-13)


def test_the_estimator_is_unbiased_and_its_cluster_error_estimates_the_exact_variance():
    # F_gcr = mu + a_g + b_gc + e_gcr with normal components and an unbalanced allocation fixed
    # in advance (C_g = 1..4, R_gc = 1..3). mu_hat is a linear combination of normals, so it is
    # exactly normal with the variance below; the replicates are the columns of one call.
    mu, sa, sb, se_leaf = 2.0, 0.5, 0.3, 1.0
    graphs, reps = 20, 4000
    g, c, r = [], [], []
    for gi in range(graphs):
        for ci in range(1 + gi % 4):
            for ri in range(1 + (gi + ci) % 3):
                g.append(gi)
                c.append(ci)
                r.append(ri)
    g, c, r = np.array(g), np.array(c), np.array(r)
    var_exact = 0.0
    for gi in range(graphs):
        cg = 1 + gi % 4
        leaf = sum(1.0 / (1 + (gi + ci) % 3) for ci in range(cg))
        var_exact += sa**2 + sb**2 / cg + se_leaf**2 * leaf / cg**2
    var_exact /= graphs**2

    rng = np.random.default_rng(2024)
    a = rng.normal(0, sa, size=(graphs, reps))
    b = rng.normal(0, sb, size=(graphs, 4, reps))
    values = mu + a[g] + b[g, c] + rng.normal(0, se_leaf, size=(g.size, reps))
    res = st.nested_mean(g, c, r, values)
    est, se = res.mean, res.se
    assert est.shape == (reps,)

    # (i) unbiased: the mean of reps exact normals; |z| <= 4.5 has false-alarm probability 7e-6
    z = (est.mean() - mu) / math.sqrt(var_exact / reps)
    assert abs(z) <= 4.5, z
    # (ii) its variance is the exact one: (reps-1) s^2 / var is chi-square(reps-1); within
    # 5 standard deviations (sqrt(2/(reps-1)) relative), false alarm about 1e-6
    ratio = est.var(ddof=1) / var_exact
    assert abs(ratio - 1) <= 5 * math.sqrt(2 / (reps - 1)), ratio
    # (iii) E[SE^2] = Var(mu_hat) exactly for independent graphs (06-statistics.md §3); the mean
    # of reps values within 4.5 of its standard errors (CLT), false alarm about 7e-6
    se2 = se**2
    z = (se2.mean() - var_exact) / (se2.std(ddof=1) / math.sqrt(reps))
    assert abs(z) <= 4.5, z
    # (iv) Student-t intervals on G - 1 = 19 degrees of freedom: coverage measured 0.948 here;
    # the window [0.93, 0.97] is 5.1 binomial standard deviations (0.0035) from it
    lo, hi = res.interval(0.95)
    coverage = float(np.mean((lo <= mu) & (mu <= hi)))
    assert 0.93 <= coverage <= 0.97, coverage


def test_the_result_does_not_depend_on_the_order_or_chunking_of_the_rows():
    rng = np.random.default_rng(13)
    g, c, r = _hierarchy(rng, 12)
    v = rng.normal(size=(g.size, 3))
    v[rng.random(size=v.shape) < 0.1] = np.nan
    base = st.nested_mean(g, c, r, v, nan_policy="omit")
    perm = rng.permutation(g.size)
    chunks = np.array_split(perm, 4)
    order = np.concatenate([chunks[2], chunks[0], chunks[3], chunks[1]])
    other = st.nested_mean(g[order], c[order], r[order], v[order], nan_policy="omit")
    for name in ("mean", "se", "graph_means", "contributing_graphs", "contributing_trajectories"):
        assert np.array_equal(getattr(base, name), getattr(other, name), equal_nan=True), name


def test_trailing_axes_are_independent_columns():
    rng = np.random.default_rng(14)
    g, c, r = _hierarchy(rng, 7)
    v = rng.normal(size=(g.size, 3, 2))
    res = st.nested_mean(g, c, r, v)
    assert res.mean.shape == (3, 2) and res.graph_means.shape == (7, 3, 2)
    for i in range(3):
        for j in range(2):
            one = st.nested_mean(g, c, r, v[:, i, j])
            assert float(one.mean) == res.mean[i, j]
            assert float(one.se) == res.se[i, j]


def test_undefined_values_are_refused_unless_omitted_which_changes_the_target():
    # graph 0: circuits 1, NaN, NaN, NaN (defined once); graph 1: 3, 3, 3, 3
    g = np.repeat([0, 1], 4)
    c = np.tile(np.arange(4), 2)
    r = np.zeros(8, dtype=int)
    v = np.array([1.0, np.nan, np.nan, np.nan, 3.0, 3.0, 3.0, 3.0])
    with pytest.raises(ValueError, match="nan_policy='omit'"):
        st.nested_mean(g, c, r, v)
    res = st.nested_mean(g, c, r, v, nan_policy="omit")
    # the equal-graph mean of the graphs' defined means, not E[F | defined] = 13/5
    assert float(res.mean) == 2.0
    assert int(res.contributing_graphs) == 2 and int(res.contributing_trajectories) == 5
    mean, _, se = _reference_nested(g, c, r, v)
    assert float(res.mean) == mean and float(res.se) == pytest.approx(se)
    # a graph with no defined value drops out
    v2 = v.copy()
    v2[0] = np.nan
    res2 = st.nested_mean(g, c, r, v2, nan_policy="omit")
    assert float(res2.mean) == 3.0 and int(res2.contributing_graphs) == 1
    assert math.isnan(float(res2.se))
    assert math.isnan(res2.graph_means[0])


def test_one_graph_has_no_graph_ensemble_error():
    g = np.zeros(4, dtype=int)
    c = np.arange(4)
    r = np.zeros(4, dtype=int)
    res = st.nested_mean(g, c, r, np.array([1.0, 2.0, 3.0, 4.0]))
    assert float(res.mean) == 2.5
    assert math.isnan(float(res.se))
    lo, hi = res.interval()
    assert math.isnan(float(lo)) and math.isnan(float(hi))
    assert math.isnan(float(st.graph_cluster_se([2.5])))


@pytest.mark.parametrize(
    ("g", "c", "r", "values", "match"),
    [
        ([0, 0], [0, 0], [0, 0], [1.0, 2.0], "appears more than once"),
        ([0, -1], [0, 0], [0, 0], [1.0, 2.0], "non-negative"),
        ([0, 0.5], [0, 0], [0, 1], [1.0, 2.0], "integer labels"),
        ([True, False], [0, 0], [0, 0], [1.0, 2.0], "integer labels"),
        ([[0, 1]], [0, 0], [0, 0], [1.0, 2.0], "one-dimensional"),
        ([0, 1], [0], [0, 0], [1.0, 2.0], "one entry per trajectory"),
        ([], [], [], [], "no trajectories"),
        ([0, 1], [0, 0], [0, 0], [1.0], "one row per trajectory"),
        ([0, 1], [0, 0], [0, 0], [1.0, np.inf], "finite or NaN"),
        (["a", "b"], [0, 0], [0, 0], [1.0, 2.0], "integer labels"),
    ],
)
def test_nested_mean_refuses_bad_input(g, c, r, values, match):
    with pytest.raises(ValueError, match=match):
        st.nested_mean(np.array(g), np.array(c), np.array(r), np.array(values, dtype=float))


def test_nested_mean_refuses_an_unknown_nan_policy():
    with pytest.raises(ValueError, match="nan_policy"):
        st.nested_mean([0], [0], [0], [1.0], nan_policy="ignore")


def test_integer_valued_float_labels_are_accepted():
    res = st.nested_mean(np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.zeros(2), [1.0, 3.0])
    assert float(res.mean) == 2.0


def test_graph_cluster_se_formula():
    gm = np.array([1.0, 2.0, 4.0, np.nan])
    expected = math.sqrt(((1 - 7 / 3) ** 2 + (2 - 7 / 3) ** 2 + (4 - 7 / 3) ** 2) / (3 * 2))
    assert float(st.graph_cluster_se(gm)) == pytest.approx(expected, rel=1e-14)
    with pytest.raises(ValueError, match="one entry per graph"):
        st.graph_cluster_se(1.0)


def test_mean_over_graphs_sorts_labels_and_refuses_bad_input():
    res = st.mean_over_graphs([3, 1, 2], [3.0, 1.0, 2.0])
    assert res.graphs.tolist() == [1, 2, 3] and res.graph_means.tolist() == [1.0, 2.0, 3.0]
    assert float(res.mean) == 2.0
    with pytest.raises(ValueError, match="distinct"):
        st.mean_over_graphs([1, 1], [1.0, 2.0])
    with pytest.raises(ValueError, match="one entry per graph label"):
        st.mean_over_graphs([1, 2], [1.0])
    with pytest.raises(ValueError, match="no graphs"):
        st.mean_over_graphs([], [])
    with pytest.raises(ValueError, match="finite or NaN"):
        st.mean_over_graphs([1], [np.inf])


# ---------------------------------------------------------------------------------------------
# paired rate differences (v1.4 §6.4)
# ---------------------------------------------------------------------------------------------


def _coupled_rates(rng, graphs, reps=None):
    """Two rates on the same graphs: a large shared graph effect, small rate-specific noise."""
    g = np.repeat(np.arange(graphs), 3)
    c = np.tile(np.arange(3), graphs)
    r = np.zeros(g.size, dtype=int)
    shape = (graphs,) if reps is None else (graphs, reps)
    effect = rng.normal(0, 1.0, size=shape)[g]
    noise = (g.size,) if reps is None else (g.size, reps)
    low = 1.0 + effect + rng.normal(0, 0.2, size=noise)
    high = 1.5 + effect + rng.normal(0, 0.2, size=noise)
    return g, c, r, low, high


def test_paired_differences_are_formed_within_graphs_first():
    rng = np.random.default_rng(21)
    g, c, r, low, high = _coupled_rates(rng, 30)
    a, b = st.nested_mean(g, c, r, low), st.nested_mean(g, c, r, high)
    d = st.paired_difference(a, b)
    ref_a, ref_b = _reference_nested(g, c, r, low)[1], _reference_nested(g, c, r, high)[1]
    diffs = [ref_b[k] - ref_a[k] for k in sorted(ref_a)]
    m = sum(diffs) / len(diffs)
    se = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) * (len(diffs) - 1)))
    assert float(d.mean) == pytest.approx(m, rel=1e-12)
    assert float(d.se) == pytest.approx(se, rel=1e-12)
    assert float(d.mean) == pytest.approx(float(b.mean) - float(a.mean), rel=1e-12)
    # the shared graph effect cancels: far below the error that ignores the pairing
    assert float(d.se) < 0.3 * math.hypot(float(a.se), float(b.se))


def test_paired_intervals_cover_and_unpaired_ones_overcover():
    # 3000 replicates of 30 graphs: the paired t-interval covers the true difference 0.5 near
    # 95% (measured 0.944; the window [0.93, 0.97] is 3.5 binomial standard deviations (0.004)
    # away, a false alarm of about 2e-4); combining the two errors as if independent
    # overcovers (measured 1.000)
    rng = np.random.default_rng(22)
    g, c, r, low, high = _coupled_rates(rng, 30, reps=3000)
    a, b = st.nested_mean(g, c, r, low), st.nested_mean(g, c, r, high)
    d = st.paired_difference(a, b)
    lo, hi = d.interval(0.95)
    coverage = float(np.mean((lo <= 0.5) & (hi >= 0.5)))
    assert 0.93 <= coverage <= 0.97, coverage
    half = st.student_t_quantile(0.975, 29) * np.hypot(a.se, b.se)
    unpaired = float(np.mean(np.abs(d.mean - 0.5) <= half))
    assert unpaired > 0.99, unpaired


def test_paired_difference_checks_the_graphs():
    a = st.mean_over_graphs([0, 1, 2], [1.0, 2.0, 3.0])
    b = st.mean_over_graphs([1, 2, 3], [2.5, 3.5, 9.0])
    with pytest.raises(ValueError, match=r"\[0\] are only in the first and \[3\] only"):
        st.paired_difference(a, b)
    d = st.paired_difference(a, b, graphs="common")
    assert d.graphs.tolist() == [1, 2] and float(d.mean) == 0.5
    with pytest.raises(ValueError, match="no graph in common"):
        st.paired_difference(a, st.mean_over_graphs([7], [1.0]), graphs="common")
    with pytest.raises(ValueError, match="different shapes"):
        st.paired_difference(a, st.mean_over_graphs([0, 1, 2], np.ones((3, 2))))
    with pytest.raises(ValueError, match="'same' or 'common'"):
        st.paired_difference(a, a, graphs="all")
    with pytest.raises(ValueError, match="two GraphMean"):
        st.paired_difference(a, 1.0)


def test_a_graph_undefined_at_either_rate_drops_out_of_the_pairs():
    a = st.mean_over_graphs([0, 1, 2], [1.0, np.nan, 3.0])
    b = st.mean_over_graphs([0, 1, 2], [2.0, 5.0, np.nan])
    d = st.paired_difference(a, b)
    assert int(d.contributing_graphs) == 1 and float(d.mean) == 1.0


# ---------------------------------------------------------------------------------------------
# the graph bootstrap
# ---------------------------------------------------------------------------------------------


def test_graph_bootstrap_is_deterministic_and_draws_labels():
    a = st.graph_bootstrap(10, 50, seed=7)
    assert a.shape == (50, 10) and a.dtype == np.int64
    assert np.array_equal(a, st.graph_bootstrap(10, 50, seed=7))
    assert not np.array_equal(a, st.graph_bootstrap(10, 50, seed=8))
    assert np.array_equal(
        st.graph_bootstrap(10, 50, seed=(7, 64)), st.graph_bootstrap(10, 50, seed=[7, 64])
    )
    labels = st.graph_bootstrap(np.array([30, 10, 20]), 40, seed=1)
    assert set(np.unique(labels).tolist()) <= {10, 20, 30}
    # the stream is numpy's raw PCG64 output, which numpy keeps stable: pin the first draws
    assert st.graph_bootstrap(10, 1, seed=7)[0].tolist() == a[0].tolist()


def test_graph_bootstrap_pins_its_stream():
    # raw PCG64(SeedSequence(7)) words reduced mod 10: a change of stream fails here
    raw = np.random.PCG64(np.random.SeedSequence(7)).random_raw(5)
    assert st.graph_bootstrap(5, 1, seed=7)[0].tolist() == (raw % np.uint64(5)).tolist()


def test_graph_bootstrap_draws_are_uniform():
    # Pearson's statistic over 11 labels (10 degrees of freedom) of 110,000 draws; its p-value
    # above 1e-6 fails with probability 1e-6 for a uniform sampler
    draws = st.graph_bootstrap(11, 10_000, seed=3)
    counts = np.bincount(draws.reshape(-1), minlength=11)
    expected = draws.size / 11
    stat = float(np.sum((counts - expected) ** 2 / expected))
    assert _chi2_sf_even(stat, 10) > 1e-6, stat


def test_the_bounded_sampler_rejects_the_top_words(monkeypatch):
    # bound 3: 2^64 mod 3 = 1, so the word 2^64 - 1 is rejected and replaced by the next
    # acceptable one (the fake hands single words out from the end of its list)
    top = np.uint64(2**64 - 1)

    class Fake:
        def __init__(self, seed):
            self.words = [np.uint64(4), top, top, np.uint64(9), np.uint64(7), top]

        def random_raw(self, size=None):
            if size is None:
                return self.words.pop()
            out, self.words = np.array(self.words[:size], dtype=np.uint64), self.words[size:]
            return out

    monkeypatch.setattr(np.random, "PCG64", Fake)
    # draws 4, (top -> top -> 7), (top -> 9): residues mod 3
    assert st._bounded_integers(0, 3, 3).tolist() == [1, 1, 0]


def test_uniform_stream_is_deterministic_and_in_range():
    u = st._uniforms((1, 2), 1000)
    assert np.array_equal(u, st._uniforms([1, 2], 1000))
    assert u.min() >= 0.0 and u.max() < 1.0
    raw = np.random.PCG64(np.random.SeedSequence([1, 2])).random_raw(3)
    assert u[:3].tolist() == [(int(x) >> 11) * 2.0**-53 for x in raw]


@pytest.mark.parametrize(
    ("graphs", "replicates", "seed", "match"),
    [
        (0, 5, 1, "at least 1"),
        ([1, 1], 5, 1, "distinct"),
        ([], 5, 1, "not be empty"),
        (5, 0, 1, "positive integer"),
        (5, 5, -1, "non-negative integer"),
        (5, 5, 1.5, "non-negative integer"),
        (5, 5, [], "empty sequence"),
        (5, 5, True, "non-negative integer"),
    ],
)
def test_graph_bootstrap_refuses_bad_input(graphs, replicates, seed, match):
    with pytest.raises(ValueError, match=match):
        st.graph_bootstrap(graphs, replicates, seed)


def test_bootstrap_replicates_equal_the_estimator_on_resampled_rows():
    rng = np.random.default_rng(31)
    g, c, r = _hierarchy(rng, 8)
    v = rng.normal(size=g.size) + g
    res = st.nested_mean(g, c, r, v)
    draws = st.graph_bootstrap(res.graphs, 20, seed=5)
    reps = res.bootstrap(draws)
    for b in range(20):
        rows, new_g = [], []
        for slot, label in enumerate(draws[b].tolist()):
            idx = np.flatnonzero(g == label)
            rows.extend(idx.tolist())
            new_g.extend([slot] * idx.size)
        rows = np.array(rows)
        again = st.nested_mean(np.array(new_g), c[rows], r[rows], v[rows])
        assert reps[b] == pytest.approx(float(again.mean), rel=1e-13)


def test_the_same_draws_keep_rates_paired():
    rng = np.random.default_rng(32)
    g, c, r, low, high = _coupled_rates(rng, 25)
    a, b = st.nested_mean(g, c, r, low), st.nested_mean(g, c, r, high)
    draws = st.graph_bootstrap(a.graphs, 200, seed=9)
    diff = st.paired_difference(a, b)
    assert np.allclose(
        b.bootstrap(draws) - a.bootstrap(draws), diff.bootstrap(draws), rtol=0, atol=1e-13
    )
    # and the bootstrap spread of the paired difference is the small one
    assert np.std(diff.bootstrap(draws)) < 0.3 * np.std(a.bootstrap(draws))


def test_the_bootstrap_variance_matches_the_cluster_error():
    # Var*(mean of G resampled graph means) = (G - 1)/G * SE^2 exactly over all resamples; with
    # 4000 replicates its estimate has relative standard deviation sqrt(2/4000) = 0.022 and the
    # window 0.1 is 4.5 of them (false alarm about 1e-5); measured ratio 1.020
    rng = np.random.default_rng(33)
    gm = rng.normal(size=40)
    res = st.mean_over_graphs(np.arange(40), gm)
    reps = res.bootstrap(st.graph_bootstrap(40, 4000, seed=4))
    ratio = np.var(reps, ddof=1) / (float(res.se) ** 2 * 39 / 40)
    assert abs(ratio - 1) < 0.1, ratio


def test_bootstrap_draws_must_come_from_the_estimate_graphs():
    res = st.mean_over_graphs([0, 1, 2], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"labels \[5\] are not among"):
        res.bootstrap(np.array([[0, 1, 5]]))
    with pytest.raises(ValueError, match="integer array"):
        res.bootstrap(np.array([0, 1, 2]))


def test_percentile_interval_is_numpys_linear_quantile_without_nan():
    rng = np.random.default_rng(34)
    x = rng.normal(size=(501, 3))
    lo, hi = st.percentile_interval(x, 0.9)
    assert np.allclose(lo, np.quantile(x, 0.05, axis=0, method="linear"), rtol=0, atol=1e-14)
    assert np.allclose(hi, np.quantile(x, 0.95, axis=0, method="linear"), rtol=0, atol=1e-14)
    x[:7, 1] = np.nan
    lo, hi = st.percentile_interval(x, 0.9)
    clean = x[7:, 1]
    assert lo[1] == pytest.approx(np.quantile(clean, 0.05, method="linear"), abs=1e-14)
    x[:, 2] = np.nan
    lo, hi = st.percentile_interval(x, 0.9)
    assert math.isnan(lo[2]) and math.isnan(hi[2])
    lo, hi = st.percentile_interval(np.arange(11.0))  # q = (1 - 0.95) / 2 carries its rounding
    assert (float(lo), float(hi)) == pytest.approx((0.25, 9.75), abs=1e-14)
    with pytest.raises(ValueError, match="level"):
        st.percentile_interval(x, 1.5)
    with pytest.raises(ValueError, match="at least one replicate"):
        st.percentile_interval(np.zeros(0))


# ---------------------------------------------------------------------------------------------
# Student's t
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.6, 0.9, 0.975, 0.995, 0.9999])
def test_student_t_quantile_against_closed_forms(q):
    assert st.student_t_quantile(q, 1) == pytest.approx(math.tan(math.pi * (q - 0.5)), rel=1e-12)
    two = (2 * q - 1) / math.sqrt(2 * q * (1 - q))
    assert st.student_t_quantile(q, 2) == pytest.approx(two, rel=1e-12)


@pytest.mark.parametrize(
    ("q", "dof", "value"),
    [  # standard tables (to the digits shown)
        (0.975, 5, 2.570581835636314),
        (0.975, 10, 2.228138851986274),
        (0.975, 19, 2.093024054408309),
        (0.975, 30, 2.042272456301238),
        (0.975, 100, 1.983971518523552),
        (0.995, 9, 3.249835541592151),
        (0.95, 4, 2.131846786326649),
    ],
)
def test_student_t_quantile_against_tables(q, dof, value):
    assert st.student_t_quantile(q, dof) == pytest.approx(value, rel=1e-9)
    assert st.student_t_quantile(1 - q, dof) == pytest.approx(-value, rel=1e-9)


def test_student_t_quantile_integrates_the_density():
    # Simpson's rule on the density of t(7) from 0 to t_q gives q - 1/2
    nu, q = 7, 0.9
    t = st.student_t_quantile(q, nu)
    const = math.gamma((nu + 1) / 2) / (math.sqrt(nu * math.pi) * math.gamma(nu / 2))
    xs = np.linspace(0, t, 2001)
    f = const * (1 + xs**2 / nu) ** (-(nu + 1) / 2)
    h = xs[1] - xs[0]
    simpson = h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())
    assert simpson == pytest.approx(q - 0.5, abs=1e-12)
    assert st.student_t_quantile(0.5, 3) == 0.0


@pytest.mark.parametrize(("q", "dof"), [(0.0, 3), (1.0, 3), (0.5, 0), (0.5, 2.5), (0.5, True)])
def test_student_t_quantile_refuses_bad_input(q, dof):
    with pytest.raises(ValueError):
        st.student_t_quantile(q, dof)


def test_interval_level_is_checked():
    res = st.mean_over_graphs([0, 1], [1.0, 2.0])
    with pytest.raises(ValueError, match="level"):
        res.interval(0)


# ---------------------------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------------------------


def test_kaplan_meier_on_a_hand_computed_example():
    # times 1, 2, 2+, 3, 4+, 5: S = 5/6, (5/6)(4/5), (2/3)(2/3), 0
    t = np.array([1, 2, 2, 3, 4, 5.0])
    e = np.array([1, 1, 0, 1, 0, 1], dtype=bool)
    km = st.weighted_kaplan_meier(t, e, np.ones(6, dtype=int))
    assert km.times.tolist() == [1.0, 2.0, 3.0, 5.0]
    assert km.survival.tolist() == pytest.approx([5 / 6, 2 / 3, 4 / 9, 0.0], rel=1e-15)
    assert km.num_at_risk.tolist() == [6, 5, 3, 1]
    assert km.num_events.tolist() == [1, 1, 1, 1]
    assert km.median == 3.0 and km.median_upper == 3.0
    # int_0^5 S = 1 + 5/6 + 2/3 + 2 * 4/9
    assert km.restricted_mean == pytest.approx(61 / 18, rel=1e-15)
    assert km.mean is None and km.censored == 2 and km.trajectories == 6
    # each trajectory its own cluster: at t = 1 no censoring yet, so the error is that of a
    # mean of the six indicators 1[tau > 1]: sqrt(sum (x - 5/6)^2 / (6 * 5)) = 1/6
    assert km.se[0] == pytest.approx(1 / 6, rel=1e-14)
    assert km.se[-1] == 0.0
    assert km.survival_at([0.5, 1.0, 2.5, 4.9, 5.0, 6.0]).tolist() == pytest.approx(
        [1.0, 5 / 6, 2 / 3, 4 / 9, 0.0, 0.0]  # the curve reached 0, so S(6) = 0 is known
    )
    assert km.se_at(0.5) == 0.0 and km.se_at(1.5) == pytest.approx(1 / 6)
    # a curve that stays above 0 is unknown past the largest follow-up time
    open_end = st.weighted_kaplan_meier(t[:5], e[:5], np.ones(5, dtype=int))
    assert open_end.horizon == 4.0 and math.isnan(open_end.survival_at(4.5))
    assert math.isnan(open_end.se_at(4.5))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_weighted_kaplan_meier_against_exact_fractions(seed):
    rng = np.random.default_rng(seed)
    n = 30
    t = rng.integers(0, 12, n).astype(float)
    e = rng.random(n) < 0.6
    w = rng.integers(1, 9, n)
    km = st.weighted_kaplan_meier(t, e, w)
    ref = _reference_km(t.tolist(), e.tolist(), w.tolist())
    assert km.times.tolist() == [x for x, _ in ref]
    assert km.survival.tolist() == pytest.approx([float(s) for _, s in ref], rel=1e-14, abs=1e-300)
    total = sum(w)
    for j, tj in enumerate(km.times):
        assert km.at_risk[j] == pytest.approx(w[t >= tj].sum() / total, rel=1e-14)
        assert km.events[j] == pytest.approx(w[(t == tj) & e].sum() / total, rel=1e-12, abs=1e-15)


def test_a_survival_of_exactly_one_half_decides_the_median_exactly():
    # the float product of these factors is 0.5000000000000001 at t = 3; exactly it is 1/2
    t = np.array([3, 4, 2, 2, 1, 2, 5, 5.0])
    e = np.array([1, 1, 1, 0, 1, 0, 1, 0], dtype=bool)
    w = np.array([4, 3, 5, 4, 4, 3, 5, 2])
    ref = dict(_reference_km(t.tolist(), e.tolist(), w.tolist()))
    assert ref[3.0] == Fraction(1, 2)
    plain = 1.0
    for tj in (1.0, 2.0, 3.0):
        y = w[t >= tj].sum()
        d = w[(t == tj) & e].sum()
        plain *= (y - d) / y
    assert plain > 0.5  # a plain float rule would miss the median at t = 3
    km = st.weighted_kaplan_meier(t, e, w)
    assert km.exact_weights
    assert km.survival[2] == 0.5
    assert km.median == 3.0 and km.median_upper == 4.0
    # integer-valued float weights are exact too; non-integer weights give the float curve
    assert st.weighted_kaplan_meier(t, e, w.astype(float)).median == 3.0
    assert not st.weighted_kaplan_meier(t, e, w / 7.0).exact_weights


def test_equal_weights_reproduce_the_unweighted_estimator():
    rng = np.random.default_rng(41)
    t = rng.integers(0, 30, 400).astype(float)
    e = rng.random(400) < 0.7
    km = st.weighted_kaplan_meier(t, e, np.ones(400, dtype=int))
    ref = analysis.kaplan_meier(t, e)
    assert np.array_equal(km.times, ref["times"])
    assert np.allclose(km.survival, ref["survival"], rtol=1e-13, atol=0)
    assert np.array_equal(km.num_at_risk, ref["at_risk"])
    seven = st.weighted_kaplan_meier(t, e, np.full(400, 7))
    assert np.array_equal(seven.survival, km.survival)
    assert np.allclose(seven.se, km.se, rtol=1e-12)


def test_nested_kaplan_meier_gives_each_graph_equal_weight():
    # graph 0: four circuits, all purified at t = 1; graph 1: one circuit, censored at 10
    g = np.array([0, 0, 0, 0, 1])
    c = np.array([0, 1, 2, 3, 0])
    r = np.zeros(5, dtype=int)
    t = np.array([1, 1, 1, 1, 10.0])
    e = np.array([1, 1, 1, 1, 0], dtype=bool)
    km = st.nested_kaplan_meier(g, c, r, t, e)
    assert km.survival.tolist() == [0.5]  # unweighted: 1/5
    assert km.median == 1.0 and km.median_upper is None
    assert st.graph_equal_weights(g, c, r).tolist() == [1, 1, 1, 1, 4]
    assert km.clusters == 2


def test_graph_equal_weights_are_integers_with_equal_graph_totals():
    rng = np.random.default_rng(42)
    g, c, r = _hierarchy(rng, 9)
    w = st.graph_equal_weights(g, c, r)
    assert w.dtype == np.int64
    totals = {int(x): int(w[g == x].sum()) for x in np.unique(g)}
    assert len(set(totals.values())) == 1
    for gi in np.unique(g):
        circuits = np.unique(c[g == gi])
        shares = {int(ci): int(w[(g == gi) & (c == ci)].sum()) for ci in circuits}
        assert len(set(shares.values())) == 1  # each circuit the same share of its graph


def _single_horizon_sample(rng, graphs=25, horizon=12):
    g, c, r = _hierarchy(rng, graphs, max_circuits=3, max_replicas=2)
    hazard = 0.08 * np.exp(rng.normal(0, 0.5, graphs))[g]
    tau = rng.geometric(np.clip(hazard, 1e-6, 1))
    observed = tau <= horizon
    times = np.where(observed, tau, horizon).astype(float)
    return g, c, r, tau, times, observed


def test_with_one_horizon_the_curve_is_the_nested_mean_of_indicators():
    # censoring only at T_max: the product telescopes (06-statistics.md §5.2), so S(t) and its
    # error are the nested mean and v1.4 §6.4 error of 1[tau > t], and the restricted mean to
    # T_max those of min(tau, T_max)
    rng = np.random.default_rng(43)
    g, c, r, tau, times, observed = _single_horizon_sample(rng)
    km = st.nested_kaplan_meier(g, c, r, times, observed)
    assert km.censored > 0 and km.mean is None
    for j, tj in enumerate(km.times):
        ind = st.nested_mean(g, c, r, (tau > tj).astype(float))
        assert km.survival[j] == pytest.approx(float(ind.mean), rel=1e-12, abs=1e-15)
        assert km.se[j] == pytest.approx(float(ind.se), rel=1e-9, abs=1e-15)
    capped = st.nested_mean(g, c, r, np.minimum(tau, 12).astype(float))
    assert km.restricted_horizon == 12.0
    assert km.restricted_mean == pytest.approx(float(capped.mean), rel=1e-12)
    assert km.restricted_mean_se == pytest.approx(float(capped.se), rel=1e-9)


def test_without_censoring_the_mean_is_given_and_is_the_nested_mean():
    rng = np.random.default_rng(44)
    g, c, r, tau, times, observed = _single_horizon_sample(rng, horizon=10_000)
    assert observed.all()
    km = st.nested_kaplan_meier(g, c, r, times, observed)
    plain = st.nested_mean(g, c, r, times)
    assert km.mean == pytest.approx(float(plain.mean), rel=1e-12)
    assert km.mean_se == pytest.approx(float(plain.se), rel=1e-9)
    assert km.restricted_mean == pytest.approx(km.mean, rel=1e-12)


def test_cluster_influences_are_the_derivatives_of_the_estimate():
    # heterogeneous censoring, float weights, clusters: the error formula G/(G-1) sum_g U_g^2,
    # with U_g computed here by central differences along a scaling of cluster g's weights
    rng = np.random.default_rng(45)
    n, groups = 80, 9
    t = rng.integers(1, 15, n).astype(float)
    e = rng.random(n) < 0.6
    w = rng.uniform(0.5, 2.0, n)
    cl = rng.integers(0, groups, n)
    km = st.weighted_kaplan_meier(t, e, w, clusters=cl, horizon=13.0)
    eps = 1e-6
    fd_s, fd_rm = [], []
    for gi in range(groups):
        up = w * np.where(cl == gi, 1 + eps, 1.0)
        down = w * np.where(cl == gi, 1 - eps, 1.0)
        a = st.weighted_kaplan_meier(t, e, up, clusters=cl, horizon=13.0)
        b = st.weighted_kaplan_meier(t, e, down, clusters=cl, horizon=13.0)
        fd_s.append((a.survival - b.survival) / (2 * eps))
        fd_rm.append((a.restricted_mean - b.restricted_mean) / (2 * eps))
    fd_s, fd_rm = np.array(fd_s), np.array(fd_rm)
    assert np.allclose(fd_s.sum(axis=0), 0, atol=1e-7)  # the influences sum to zero
    se = np.sqrt(groups / (groups - 1) * np.sum(fd_s**2, axis=0))
    assert np.allclose(km.se, se, rtol=1e-5, atol=1e-9)
    rm_se = math.sqrt(groups / (groups - 1) * np.sum(fd_rm**2))
    assert km.restricted_mean_se == pytest.approx(rm_se, rel=1e-5)
    # the chunked accumulation does not depend on the chunk size
    order = np.lexsort((w, cl, ~e, t))
    ts, es, ws, cs = t[order], e[order], w[order], cl[order]
    total = ws.sum()
    args = (ts, es, ws / total, cs, groups, km.times, km.at_risk, km.events)
    kept = km.at_risk - km.events
    whole = np.vstack([u for _, u in st._km_influence_chunks(*args, kept, km.survival)])
    small = np.vstack([u for _, u in st._km_influence_chunks(*args, kept, km.survival, chunk=2)])
    assert np.allclose(whole, small, rtol=0, atol=1e-15)
    assert np.allclose(whole, fd_s, rtol=1e-5, atol=1e-9)


def test_censored_geometric_times_give_the_known_median_and_restricted_mean():
    # P(tau > t) = 0.9^t: S(6) = 0.531, S(7) = 0.478, so the median is 7. With 20,000 draws
    # and light censoring the standard error near t = 6-7 is below 0.004, and a median other
    # than 7 needs a 6-sigma error (false alarm below 1e-9). The restricted mean to 30 is
    # (1 - 0.9^30) / 0.1 exactly; 4.5 standard errors allow a false alarm of 7e-6.
    rng = np.random.default_rng(46)
    n = 20_000
    tau = rng.geometric(0.1, n)
    cens = np.minimum(rng.geometric(0.02, n), 40)
    times = np.minimum(tau, cens).astype(float)
    observed = tau <= cens
    km = st.weighted_kaplan_meier(times, observed, np.ones(n, dtype=int))
    assert km.median == 7.0
    km30 = st.weighted_kaplan_meier(times, observed, np.ones(n, dtype=int), horizon=30.0)
    exact = (1 - 0.9**30) / 0.1
    assert abs(km30.restricted_mean - exact) <= 4.5 * km30.restricted_mean_se, (
        km30.restricted_mean,
        exact,
    )
    # S(t) itself within 4.5 standard errors at a few times
    for tt in (3, 6, 10, 20):
        assert abs(km.survival_at(tt) - 0.9**tt) <= 4.5 * km.se_at(tt)


def test_exponential_times_give_the_known_median():
    # median ln 2 / 0.2 = 3.466; 4000 draws with exponential censoring (rate 0.05): the
    # median's standard deviation is about 0.09 (sd of S at the median / density there), and
    # the tolerance 0.4 is 4.4 of them (false alarm about 1e-5)
    rng = np.random.default_rng(47)
    n = 4000
    tau = rng.exponential(5.0, n)
    cens = rng.exponential(20.0, n)
    km = st.weighted_kaplan_meier(np.minimum(tau, cens), tau <= cens, np.ones(n, dtype=int))
    assert abs(km.median - math.log(2) / 0.2) < 0.4, km.median


def test_kaplan_meier_errors_cover_under_graph_clustering():
    # graphs carry a frailty (hazard 0.06 or 0.16 with probability 1/2 each), circuits share
    # it; censoring at random depths 3..40, capped at the horizon 20 (an atom there, as in the
    # study, where every trajectory not purified by T_max is censored at T_max).
    # True S(t) = (0.94^t + 0.84^t)/2.
    # 2000 replicates of 60 graphs x 3 circuits: Student-t intervals at t = 5 and for the
    # restricted mean to 20 cover the truth near 95% (measured 0.946 and 0.953); the window
    # [0.93, 0.97] is at least 3.2 binomial standard deviations (0.0049) from them (false
    # alarm about 1e-3)
    rng = np.random.default_rng(48)
    graphs, circuits, reps = 60, 3, 2000
    g = np.repeat(np.arange(graphs), circuits)
    c = np.tile(np.arange(circuits), graphs)
    r = np.zeros(g.size, dtype=int)
    q = st.student_t_quantile(0.975, graphs - 1)
    s_true = (0.94**5 + 0.84**5) / 2
    rm_true = sum((0.94**t + 0.84**t) / 2 for t in range(20))
    hit_s = hit_rm = 0
    for _ in range(reps):
        hazard = np.where(rng.random(graphs) < 0.5, 0.06, 0.16)[g]
        tau = rng.geometric(hazard)
        cens = np.minimum(rng.integers(3, 41, g.size), 20)
        times = np.minimum(tau, cens).astype(float)
        km = st.nested_kaplan_meier(g, c, r, times, tau <= cens, horizon=20.0)
        hit_s += abs(km.survival_at(5.0) - s_true) <= q * km.se_at(5.0)
        hit_rm += abs(km.restricted_mean - rm_true) <= q * km.restricted_mean_se
    assert 0.93 <= hit_s / reps <= 0.97, hit_s / reps
    assert 0.93 <= hit_rm / reps <= 0.97, hit_rm / reps


def test_a_sample_without_events_has_survival_one():
    km = st.weighted_kaplan_meier([5.0, 5.0, 3.0], [0, 0, 0], [1, 1, 1])
    assert km.times.size == 0 and km.survival.size == 0
    assert km.median is None and km.median_upper is None
    assert km.restricted_mean == 5.0
    assert km.survival_at([0.0, 5.0]).tolist() == [1.0, 1.0]
    assert km.se_at(2.0) == 0.0


def test_events_at_depth_zero_and_a_curve_that_reaches_zero():
    km = st.weighted_kaplan_meier([0.0, 0.0, 2.0, 2.0], [1, 1, 1, 1], [1, 1, 1, 1])
    assert km.survival.tolist() == [0.5, 0.0]
    assert km.median == 0.0 and km.median_upper == 2.0
    assert km.restricted_mean == 1.0  # E[min(tau, 2)] = (0 + 0 + 2 + 2) / 4
    assert km.mean == 1.0 and km.se.tolist()[1] == 0.0
    # S = 0 is known beyond the last time, so a later horizon is allowed and adds no area
    later = st.weighted_kaplan_meier([0.0, 0.0, 2.0, 2.0], [1, 1, 1, 1], [1, 1, 1, 1], horizon=9.0)
    assert later.restricted_mean == 1.0 and later.restricted_horizon == 9.0
    assert later.survival_at(5.0) == 0.0 and later.se_at(5.0) == 0.0


def test_one_cluster_has_no_error():
    km = st.weighted_kaplan_meier([1.0, 2.0], [1, 0], [1, 1], clusters=[0, 0])
    assert km.clusters == 1 and math.isnan(km.se[0]) and math.isnan(km.restricted_mean_se)


def test_near_half_values_are_decided_by_the_exact_products():
    # s = 0.5 in floats while the exact product is just below or just above 1/2
    s = np.array([0.5])
    st._snap_half(s, np.array([2**52 - 1]), np.array([2**53]))
    assert s[0] == np.nextafter(0.5, 0.0)
    s = np.array([0.5])
    st._snap_half(s, np.array([2**52 + 1]), np.array([2**53]))
    assert s[0] == np.nextafter(0.5, 1.0)
    s = np.array([np.nextafter(0.5, 1.0)])
    st._snap_half(s, np.array([3]), np.array([6]))
    assert s[0] == 0.5
    s = np.array([0.75, 0.25])  # far from 1/2: untouched
    st._snap_half(s, np.array([3, 1]), np.array([4, 3]))
    assert s.tolist() == [0.75, 0.25]


def test_weights_too_large_for_exact_sums_fall_back_to_floats(monkeypatch):
    monkeypatch.setattr(st, "_EXACT_WEIGHT_LIMIT", 8)
    km = st.weighted_kaplan_meier([1.0, 2.0, 3.0], [1, 1, 0], [3, 3, 3])
    assert not km.exact_weights and km.survival.tolist() == pytest.approx([2 / 3, 1 / 3])


@pytest.mark.parametrize(
    ("times", "observed", "weights", "kwargs", "match"),
    [
        ([1.0, -1.0], [1, 0], [1, 1], {}, "sentinel"),
        ([1.0, np.inf], [1, 0], [1, 1], {}, "finite and non-negative"),
        ([[1.0]], [1], [1], {}, "one-dimensional"),
        ([], [], [], {}, "empty"),
        ([1.0, 2.0], [1], [1, 1], {}, "one entry per trajectory"),
        ([1.0, 2.0], [1, 2], [1, 1], {}, "booleans"),
        ([1.0, 2.0], [1, 0], [1], {}, "weights must have one entry"),
        ([1.0, 2.0], [1, 0], [1, 0], {}, "positive and finite"),
        ([1.0, 2.0], [1, 0], ["a", "b"], {}, "numbers"),
        ([1.0, 2.0], [1, 0], [1, 1], {"clusters": [0]}, "clusters must have one entry"),
        ([1.0, 2.0], [1, 0], [1, 1], {"horizon": 3.0}, "horizon must lie in"),
        ([1.0, 2.0], [1, 0], [1, 1], {"horizon": -1.0}, "horizon must lie in"),
    ],
)
def test_weighted_kaplan_meier_refuses_bad_input(times, observed, weights, kwargs, match):
    with pytest.raises(ValueError, match=match):
        st.weighted_kaplan_meier(times, observed, weights, **kwargs)


def test_nested_kaplan_meier_checks_lengths():
    with pytest.raises(ValueError, match="one entry per trajectory"):
        st.nested_kaplan_meier([0, 1], [0, 0], [0, 0], [1.0], [1])


def test_graph_equal_weights_refuse_an_overflowing_allocation(monkeypatch):
    monkeypatch.setattr(st, "_EXACT_WEIGHT_LIMIT", 10)
    with pytest.raises(ValueError, match="least common multiple"):
        st.graph_equal_weights([0, 0, 1, 1, 1], [0, 1, 0, 1, 2], [0, 0, 0, 0, 0])


# ---------------------------------------------------------------------------------------------
# censoring summaries
# ---------------------------------------------------------------------------------------------


def test_censoring_summary_uses_graph_equal_weights():
    # graph 0: three censored circuits; graph 1: one observed circuit
    g = np.array([0, 0, 0, 1])
    c = np.array([0, 1, 2, 0])
    r = np.zeros(4, dtype=int)
    t = np.array([20.0, 20.0, 20.0, 4.0])
    e = np.array([0, 0, 0, 1], dtype=bool)
    s = st.censoring_summary(g, c, r, t, e)
    assert s.censored_fraction == 0.5 and s.raw_censored_fraction == 0.75
    assert s.observed_fraction == 0.5
    assert (s.trajectories, s.observed, s.censored, s.graphs) == (4, 1, 3, 2)
    assert s.horizon == 20.0 and s.censoring_times.tolist() == [20.0]
    assert s.graphs_all_censored == 1 and s.graphs_any_censored == 1
    assert s.censored_fraction_se == 0.5
    assert json.loads(json.dumps(s.to_dict()))["censoring_times"] == [20.0]
    with pytest.raises(ValueError, match="one entry per trajectory"):
        st.censoring_summary(g, c, r, t[:2], e)


# ---------------------------------------------------------------------------------------------
# peak times (v1.4 §5.8)
# ---------------------------------------------------------------------------------------------


def _peaks():
    # two graphs, one circuit each; checkpoints 0, 1, 2, 4, 8
    g = np.array([0, 1])
    c = np.zeros(2, dtype=int)
    r = np.zeros(2, dtype=int)
    cps = np.array([0, 1, 2, 4, 8.0])
    traces = np.array([[3, 0, 5, 0, 0], [3, 0, 0, 0, 4.0]])
    return g, c, r, cps, traces


def test_the_three_peak_statistics_are_different_numbers():
    g, c, r, cps, traces = _peaks()
    tstar = np.array([2.0, 8.0])  # the single maximizers
    at_h = np.array([False, True])
    s = st.peak_time_summaries(
        g, c, r, tstar, at_horizon=at_h, traces=traces, checkpoints=cps, horizon_policy="include"
    )
    assert float(s.arithmetic.mean) == 5.0
    assert s.geometric == pytest.approx(4.0)  # exp((ln 2 + ln 8) / 2)
    # the mean trace over the positive checkpoints is (0, 2.5, 0, 2): its argmax is 2, and the
    # value 3 at t = 0 does not count
    assert s.mean_trace.mean.tolist() == [3.0, 0.0, 2.5, 0.0, 2.0]
    assert s.argmax == 2.0 and s.argmax_set.tolist() == [2.0] and s.argmax_at_horizon is False
    assert s.horizon_fraction == 0.5 and s.missing_fraction == 0.0
    assert "lower bounds" in s.note
    assert s.geometric <= float(s.arithmetic.mean)  # Jensen
    json.dumps(s.to_dict())


def test_horizon_peaks_follow_the_declared_policy():
    g, c, r, cps, traces = _peaks()
    tstar = np.array([2.0, 8.0])
    at_h = np.array([False, True])
    refused = st.peak_time_summaries(g, c, r, tstar, at_horizon=at_h)
    assert refused.arithmetic is None and refused.geometric is None
    assert "horizon" in refused.note and refused.horizon_policy == "refuse"
    excluded = st.peak_time_summaries(g, c, r, tstar, at_horizon=at_h, horizon_policy="exclude")
    assert (
        float(excluded.arithmetic.mean) == 2.0 and int(excluded.arithmetic.contributing_graphs) == 1
    )
    assert math.isnan(excluded.geometric_se)
    none_left = st.peak_time_summaries(
        g, c, r, tstar, at_horizon=[True, True], horizon_policy="exclude"
    )
    assert none_left.arithmetic is None and "no trajectory" in none_left.note
    clean = st.peak_time_summaries(g, c, r, tstar, at_horizon=[False, False])
    assert float(clean.arithmetic.mean) == 5.0 and clean.note == ""


def test_missing_peaks_are_left_out_and_counted():
    g = np.array([0, 0, 1, 1])
    c = np.array([0, 1, 0, 1])
    r = np.zeros(4, dtype=int)
    tstar = np.array([np.nan, 4.0, 16.0, 16.0])
    s = st.peak_time_summaries(g, c, r, tstar, at_horizon=np.zeros(4, dtype=bool))
    assert s.missing_fraction == 0.25  # graph 0 has half its circuits without a peak
    assert float(s.arithmetic.mean) == 10.0  # (4 + 16) / 2: each graph's defined peaks
    assert s.geometric == pytest.approx(8.0)
    assert s.geometric_se == pytest.approx(8.0 * float(s.log_mean.se))


def test_argmax_replicates_follow_the_graph_draws():
    g, c, r, cps, traces = _peaks()
    s = st.peak_time_summaries(
        g, c, r, np.array([2.0, 8.0]), at_horizon=[False, True], traces=traces, checkpoints=cps
    )
    reps = s.argmax_replicates(np.array([[0, 0], [1, 1], [0, 1]]))
    assert reps.tolist() == [2.0, 8.0, 2.0]
    plain = st.peak_time_summaries(g, c, r, np.array([2.0, 8.0]), at_horizon=[False, False])
    with pytest.raises(ValueError, match="needs the traces"):
        plain.argmax_replicates(np.array([[0, 1]]))


def test_a_trace_tied_at_the_horizon_is_flagged():
    g = np.array([0, 1])
    c = np.zeros(2, dtype=int)
    r = np.zeros(2, dtype=int)
    cps = np.array([1.0, 2.0, 4.0])
    traces = np.array([[1, 3, 3], [1, 3, 3.0]])
    s = st.peak_time_summaries(
        g, c, r, np.array([2.83, 2.83]), at_horizon=[True, True], traces=traces, checkpoints=cps
    )
    assert s.argmax == 2.0 and s.argmax_set.tolist() == [2.0, 4.0] and s.argmax_at_horizon


def test_a_trace_undefined_at_every_positive_checkpoint_has_no_argmax():
    cps = np.array([0.0, 1.0, 2.0])
    traces = np.array([[1.0, np.nan, np.nan], [1.0, np.nan, np.nan]])
    s = st.peak_time_summaries(
        [0, 1],
        [0, 0],
        [0, 0],
        [np.nan, np.nan],
        at_horizon=[False, False],
        traces=traces,
        checkpoints=cps,
        trace_nan_policy="omit",
    )
    assert s.argmax is None and s.argmax_set is None and s.argmax_at_horizon is None
    assert s.arithmetic is None and s.missing_fraction == 1.0
    assert math.isnan(s.argmax_replicates(np.array([[0, 1]]))[0])


def test_plain_conversion_handles_nested_containers():
    assert st._plain({"a": np.float64("nan"), 2: [np.int64(3), (np.bool_(True), "x")]}) == {
        "a": None,
        "2": [3, [True, "x"]],
    }


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"horizon_policy": "drop"}, "horizon_policy"),
        ({"peak_times": [0.0, 2.0]}, "positive"),
        ({"peak_times": [[1.0], [2.0]]}, "one value per trajectory"),
        ({"at_horizon": [1, 2]}, "booleans"),
        ({"traces": np.ones((2, 3))}, "need their checkpoints"),
        ({"checkpoints": [1.0, 2.0]}, "without traces"),
        ({"traces": np.ones((2, 3)), "checkpoints": [1.0, 2.0]}, "must be \\(trajectories"),
        ({"traces": np.ones((2, 2)), "checkpoints": [2.0, 1.0]}, "strictly increasing"),
        ({"traces": np.array([[1.0, np.nan], [1.0, 1.0]]), "checkpoints": [1.0, 2.0]}, "NaN"),
    ],
)
def test_peak_time_summaries_refuse_bad_input(kwargs, match):
    args = {"peak_times": [1.0, 2.0], "at_horizon": [False, False]}
    args.update(kwargs)
    peak_times = args.pop("peak_times")
    with pytest.raises(ValueError, match=match):
        st.peak_time_summaries([0, 1], [0, 0], [0, 0], peak_times, **args)


def test_results_serialize_to_json():
    rng = np.random.default_rng(51)
    g, c, r = _hierarchy(rng, 5)
    res = st.nested_mean(g, c, r, rng.normal(size=g.size))
    data = json.loads(json.dumps(res.to_dict()))
    assert data["mean"] == pytest.approx(float(res.mean))
    assert data["nan_policy"] == "raise" and len(data["graph_means"]) == 5
    km = st.weighted_kaplan_meier([1.0, 2.0, 3.0], [1, 1, 0], [1, 1, 1])
    out = json.loads(json.dumps(km.to_dict()))
    assert out["median"] == 2.0 and out["mean"] is None and out["variance"] == st.KM_VARIANCE
    single = st.mean_over_graphs([0], [1.0]).to_dict()
    assert single["se"] is None
