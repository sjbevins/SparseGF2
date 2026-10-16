"""The nearest-neighbour brickwork baseline of the geometry-correlated erasures, against the literature.

**Circuit** (the literature's monitored Clifford brickwork): the periodic ring ``graphs.cycle(n)``
with ``schedule="matching"`` (even bonds, then odd bonds), uniform random two-qubit Cliffords,
every system qubit measured in Z with probability p after every layer, ``4n`` layers, the
purification picture (every system qubit Bell-paired with a reference qubit, so the code holds
``k = S(R)`` logical qubits), observed at the final layer. p_c ~ 0.16 in this convention
(0.1593(5), Gullans and Huse, PRX 10, 041020 (2020)). For contrast, Watts-Strogatz circuits
(k = 4, beta in {0.1, 1}, one graph per trajectory, ``schedule="random_edges"``, ``n / 2`` gates
per layer) at the same n and p.

**Measured per trajectory** (n in {32, 64, 128, 256}, p in {0.04, 0.08, 0.12, 0.3}):

* ``P_{C,J}(m)`` for J in {0, 0.5, 1, 2}: the in-run probe ``correlated_erasure`` on the ring
  (the exact ring sampler, every size m = 1..n-1); on a Watts-Strogatz circuit both on its own
  interaction graph (the swap chain with replica exchange, sizes every n/32, 4 sweeps between
  records) and on the ring of its labels (``geometry="cycle"``, exact).
* ``P_{C,inf}(m)``, the ring's exact J -> inf limit: the failing fraction of the n intervals of
  length m (``StabilizerCode.interval_failure_by_size``; the minimum-boundary regions of the ring
  are exactly the intervals, a theorem checked in ``tests/test_brickwork_correlated.py``).
* ``P_C(m)`` (uniform) also from first-failure orderings with exact counts for small m
  (``erasure_first_failure``, ``uncorrectable_region_counts``), and the uniform distance bounds
  (``analysis.distance_bounds``).
* Three contiguous lengths, with the definitions kept apart:

  - ``d_cont``: the smallest failing interval (``I(A:R) > 0``), per code with k > 0. This is
    Li and Fisher's algebraic definition and Gullans and Huse's ``l_min = min_x l_x``;
  - Gullans and Huse's site-averaged length ``<l> = (1/n) sum_x l_x``, ``l_x`` the shortest
    failing interval containing x, with k = 0 counted as 0 (their convention);
  - Li and Fisher's numerical ``|A|*``: the interval length at which the ensemble mean of
    ``I(A:R)`` over intervals reaches 1 bit (their epsilon = ln 2 nats).

  The ring also runs at n = 512 and 1024 (``--scaling-sizes``) for these lengths only (no
  correlated probe), to extend the fits to Li and Fisher's largest L.

**Reported:** q+ (the fraction holding a code), ``<k>/n`` (against Gullans and Huse's
near-critical form ``7.3 (p_c - p)^1.28``), the three lengths and the distance bounds, the
fitted exponents of the lengths in n (against Li-Fisher's ``gamma_1 ~ 0.38``, Li-Vijay-Fisher's
1/3, Gullans-Huse's ``a`` in [1/3, 1/2]), ``P_{C,J}(m)`` at selected sizes, the probability that a
ring region at coupling J is one interval (how far a finite J is from the limit), and at
n = 64 and 128 ``Phi_{C,J}(e)`` and ``e_delta(J)`` (means over codes with k > 0, bootstrap
errors). Each table names its seeds and parameters.

    python benchmarks/correlated_erasure_baseline.py                      # committed: ~20 min, 32 threads
    python benchmarks/correlated_erasure_baseline.py --trajectories 1024 --ws-trajectories 256  # ~1.5 h
    python benchmarks/correlated_erasure_baseline.py --quick --out <dir>  # smoke test, seconds
    python benchmarks/correlated_erasure_baseline.py --from-json benchmarks/results/correlated_erasure_baseline.json

``--from-json`` measures nothing: it redoes the fits and the markdown from a results file.
Writes ``correlated_erasure_baseline.json`` and ``.md`` into ``--out`` (default
``benchmarks/results/``). One private kernel is used, ``_kernel.entropy.erasure_profile_rows``
(``I(A:R)`` of every interval from one start in one pass), for Li and Fisher's mean profile.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numba
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from thread_scaling import cpu_info  # noqa: E402

import sparsegf2 as sg  # noqa: E402
from sparsegf2 import analysis, erasures, graphs, mipt  # noqa: E402
from sparsegf2._kernel.entropy import erasure_profile_rows  # noqa: E402

SEED = 2026_10_16
SIZES = (32, 64, 128, 256)
SCALING_SIZES = (512, 1024)  # ring only, contiguous lengths only
RATES = (0.04, 0.08, 0.12, 0.3)
MIXED = (0.04, 0.08, 0.12)  # below p_c: the scaling fits
COUPLINGS = (0.0, 0.5, 1.0, 2.0)
BETAS = (0.1, 1.0)
CHAIN_THIN = 4  # sweeps between recorded regions of the Watts-Strogatz swap chain
PHI_SIZES = (64, 128)
DELTAS = (0.01, 0.1)
EPSILONS = (0.01, 0.02, 0.05, 0.1, 0.2)
SELECTED = (1, 2, 4, 8, 12, 16)  # sizes m = n * s / 32
LARGE_N = 128  # the second fit uses n >= LARGE_N only
PROFILE_CAP = 192  # Li-Fisher's mean interval information is computed for lengths up to this
GH = {"p_c": 0.1593, "nu": 1.28, "A": 7.3}  # Gullans and Huse, Fig. 2 (near-critical fit)
GH_MIN_P = 0.1  # the near-critical form is read from here to p_c; it gives 0.69 at p = 0
LITERATURE: dict[str, Any] = {
    "li_fisher_gamma1": 0.38,  # PRB 103, 104306 (2021): |A|* (mean I = 1 bit), p = 0.04-0.12
    "li_vijay_fisher_beta": 1 / 3,  # PRX Quantum 4, 010331 (2023): DPRE roughness exponent
    "gullans_huse_a": (1 / 3, 1 / 2),  # PRX 10, 041020 (2020) App. F: <l> ~ L^a, deep mixed phase
}
UNIFORM = "uniform (orderings)"
INF = "J = inf"


# ---------------------------------------------------------------------------
# per-code quantities
# ---------------------------------------------------------------------------


def interval_lengths(first: np.ndarray, n: int) -> tuple[int, float]:
    """``(d_cont, <l>)`` of one code from its window first failures L(s) (k > 0).

    l_x, the shortest failing interval containing x, is min over starts s of
    max(L(s), (x - s) mod n + 1): from s the windows of length >= L(s) fail, and the shortest
    one containing x has length (x - s) mod n + 1.
    """
    s = np.arange(n)
    reach = (s[None, :] - s[:, None]) % n + 1  # [start, x]
    l_x = np.maximum(first[:, None], reach).min(axis=0)
    assert l_x.min() == first.min()
    return int(first.min()), float(l_x.mean())


def interval_information(code, cap: int) -> np.ndarray:
    """``I(A:R)`` averaged over the n intervals A of each length l = 0..cap (private kernel)."""
    n = code.num_qubits
    total = np.zeros(cap + 1)
    lost = np.zeros(cap + 1, dtype=np.int64)
    base = np.arange(n, dtype=np.int64)
    for s in range(n):
        seq = np.ascontiguousarray(np.roll(base, -s)[:cap])
        erasure_profile_rows(code._tab, code._ws, seq, code._marked, lost)
        total += lost
    return total / n


def crossing(profile: np.ndarray, level: float = 1.0) -> float:
    """The length at which a mean profile first reaches ``level``, linearly interpolated."""
    above = np.flatnonzero(profile >= level)
    if not above.size:
        return math.nan
    j = int(above[0])
    if j == 0:
        return 0.0
    lo, hi = profile[j - 1], profile[j]
    return float(j - 1 + (level - lo) / (hi - lo))


def mean_se(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"mean": None, "se": None, "count": 0}
    se = float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else None
    return {"mean": float(v.mean()), "se": se, "count": int(v.size)}


# ---------------------------------------------------------------------------
# one case: run, then summarize
# ---------------------------------------------------------------------------


def window(n: int) -> list[int]:
    stride = max(1, n // 32)
    return list(range(stride, n, stride))


def selected_sizes(n: int) -> list[int]:
    return sorted({max(1, n * s // 32) for s in SELECTED})


def _uniform_probes(n: int, cfg: dict) -> list[mipt.Probe]:
    w = analysis.exhaustive_cutoff(n, cfg["budget"])
    return [
        mipt.erasure_first_failure(samples=cfg["orderings"]),
        mipt.uncorrectable_region_counts(max_size=w, name="counts"),
    ]


def ring_spec(n: int, p: float, cfg: dict, correlated: bool) -> mipt.MonitoredCircuit:
    ce = [mipt.correlated_erasure(COUPLINGS, samples=cfg["samples"], name="ring")]
    return mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        layers=4 * n,
        measurement_rate=p,
        schedule="matching",
        reference="purification",
        checkpoints="final",
        probes=[
            mipt.code_dimension(),
            *(ce if correlated else []),
            *_uniform_probes(n, cfg),
            mipt.code_snapshot(),
        ],
    )


def ws_graphs(n: int, beta: float, count: int) -> list[np.ndarray]:
    return [
        graphs.watts_strogatz(n, 4, beta, seed=[SEED, n, round(100 * beta), i])
        for i in range(count)
    ]


def ws_spec(n: int, p: float, beta: float, cfg: dict) -> mipt.MonitoredCircuit:
    bank: Any = ws_graphs(n, beta, cfg["ws_trajectories"])  # a collection: one graph each
    return mipt.MonitoredCircuit(
        num_system=n,
        edges=bank,
        layers=4 * n,
        measurement_rate=p,
        schedule="random_edges",
        reference="purification",
        checkpoints="final",
        probes=[
            mipt.code_dimension(),
            mipt.correlated_erasure(
                COUPLINGS, sizes=window(n), samples=cfg["samples"], thin=CHAIN_THIN, name="graph"
            ),
            mipt.correlated_erasure(
                COUPLINGS, samples=cfg["samples"], geometry="cycle", name="ring"
            ),
            *_uniform_probes(n, cfg),
            mipt.code_snapshot(),
        ],
    )


def case_seed(family: str, n: int, p: float, beta: float | None) -> list[int]:
    return [SEED, {"ring": 0, "ws": 1}[family], n, round(1000 * p), round(100 * (beta or 0))]


def full_curves(res: mipt.EnsembleResult, name: str, k: np.ndarray, n: int):
    """Per code ``(trajectories, couplings, n + 1)`` curves of a probe, gaps interpolated."""
    _, sizes, values = res.correlated(name)
    return analysis.correlated_failure_curve(values[:, -1], sizes, n, holds_code=k)


def curve_table(per_code: dict[str, np.ndarray], k: np.ndarray, sizes: list[int]) -> dict:
    table = {}
    for label, curve in per_code.items():
        mean, se, _ = analysis.mean_over_codes(curve, k)
        table[label] = {
            "mean": [float(mean[m]) for m in sizes],
            "se": [float(se[m]) for m in sizes],
        }
    return table


def phi_table(curves: dict[str, np.ndarray], n: int, cfg: dict, rng) -> dict:
    """Phi_J(e) of the mean curves and e_delta(J) with bootstrap errors over codes."""
    labels = list(curves)
    stack = np.stack([curves[label] for label in labels], axis=1)  # (codes, labels, n + 1)
    mean = stack.mean(axis=0)
    out: dict = {"epsilons": list(EPSILONS), "labels": labels, "codes": int(stack.shape[0])}
    out["phi"] = analysis.failure_probability_from_curve(mean, n, list(EPSILONS)).tolist()
    boot = rng.integers(0, stack.shape[0], size=(cfg["bootstrap"], stack.shape[0]))
    resampled = stack[boot].mean(axis=1)  # (B, labels, n + 1)
    tolerances = {}
    for delta in DELTAS:
        value, monotone = analysis.tolerance_from_curve(mean, n, delta, return_monotone=True)
        spread = analysis.tolerance_from_curve(resampled, n, delta)
        tolerances[str(delta)] = {
            "value": np.asarray(value).tolist(),
            "se": np.std(spread, axis=0, ddof=1).tolist(),
            "monotone": np.asarray(monotone).tolist(),
        }
    out["tolerance"] = tolerances
    return out


def summarize(res, spec, family: str, n: int, p: float, beta, cfg: dict, rng) -> dict:
    k = res.values["code_dimension"][:, -1].astype(np.int64)
    holds = k > 0
    t = k.size
    first = res.values["erasure_first_failure"][:, -1].astype(np.int64)
    counts = res.values["counts"][:, -1]
    hybrid = analysis.hybrid_failure_by_size(first, n, exact_counts=counts)
    low, high = analysis.distance_bounds(first, n, exact_counts=counts)

    cap = min(n, PROFILE_CAP)
    intervals = np.zeros((t, n + 1))
    d_cont = np.zeros(t, dtype=np.int64)
    gh_l = np.zeros(t)
    profiles = np.zeros((t, cap + 1))
    for i in np.flatnonzero(holds):
        code = res.code(int(i), 0)
        wff = code.window_first_failure()
        intervals[i, 1:] = np.cumsum(np.bincount(wff, minlength=n + 2))[1 : n + 1] / n
        d_cont[i], gh_l[i] = interval_lengths(wff, n)
        profiles[i] = interval_information(code, cap)

    per_code = {UNIFORM: hybrid}
    filled = 0
    if "ring" in {q.name for q in spec.probes}:
        ring_curve, ring_filled = full_curves(res, "ring", k, n)
        filled = int(ring_filled.sum())
        per_code.update({f"J = {c:g}": ring_curve[:, j] for j, c in enumerate(COUPLINGS)})
    per_code[INF] = intervals
    chosen = selected_sizes(n)

    # Li-Fisher |A|*: the mean (over all trajectories, k = 0 as 0) interval information = 1 bit
    mean_profile = profiles.mean(axis=0)
    star = crossing(mean_profile)
    boot = rng.integers(0, t, size=(cfg["bootstrap"], t))
    star_boot = np.array([crossing(profiles[b].mean(axis=0)) for b in boot])
    star_boot = star_boot[np.isfinite(star_boot)]
    head = min(cap + 1, 4 * math.ceil(star) + 2) if math.isfinite(star) else min(cap + 1, 33)
    exact_d = holds & (low == high)
    summary = {
        "family": family,
        "beta": beta,
        "n": n,
        "p": p,
        "correlated": len(per_code) > 2,
        "trajectories": t,
        "seed": case_seed(family, n, p, beta),
        "layers": 4 * n,
        "elapsed_seconds": float(res.elapsed_seconds),
        "q_plus": float(holds.mean()),
        "k_over_n": mean_se(k / n),
        "d_cont": mean_se(d_cont[holds]),
        "gh_l": mean_se(gh_l),  # k = 0 counted as 0 (Gullans and Huse)
        "gh_l_code": mean_se(gh_l[holds]),
        "gh_l_min": mean_se(d_cont),  # min_x l_x with k = 0 as 0
        "lf_star": {
            "value": star,
            "se": float(np.std(star_boot, ddof=1)) if star_boot.size > 1 else None,
        },
        "lf_profile_head": mean_profile[:head].round(4).tolist(),  # mean I(interval of l : R)
        "distance": {
            "low": mean_se(low[holds]),
            "high": mean_se(high[holds]),
            "exact_fraction": float(exact_d[holds].mean()) if holds.any() else None,
            "d_below_d_cont_fraction": (
                float((low[exact_d] < d_cont[exact_d]).mean()) if exact_d.any() else None
            ),
            "high_minus_d_cont": mean_se(high[holds] - d_cont[holds]),
        },
        "sizes": chosen,
        "curves": curve_table(per_code, k, chosen),
        "probe_interpolated_sizes": filled,
    }
    if family == "ws":
        graph_curve, graph_filled = full_curves(res, "graph", k, n)
        graph = {f"J = {c:g}": graph_curve[:, j] for j, c in enumerate(COUPLINGS)}
        summary["graph_curves"] = curve_table(graph, k, chosen)
        summary["graph_sizes_measured"] = int((~graph_filled).sum())
        sampler = spec.correlated_sampler("graph")
        summary["graph_sampler"] = {
            key: sampler.get(key) for key in ("method", "rungs", "max_step", "burn_in", "thin")
        }
        per_code.update({f"graph {label}": v for label, v in graph.items() if label != "J = 0"})
    if summary["correlated"] and n in cfg["phi_sizes"] and holds.sum() >= 2:
        summary["phi"] = phi_table({key: v[holds] for key, v in per_code.items()}, n, cfg, rng)
    return summary


def mixing_check(n: int, beta: float, cfg: dict) -> dict:
    """The swap chain on the first Watts-Strogatz graph, graph side, with the probe's settings.

    The chain does not depend on the code or p, so this check covers every p of (beta, n): the
    regions recorded by ``erasures.sample_regions`` (same kernel, ladder and settings), their
    warnings, exchange acceptance, and the J = 0 mean boundary against the exact uniform law.
    """
    edges = ws_graphs(n, beta, 1)[0]
    warned: list[str] = []
    weakest, rungs, worst_b0 = 1.0, 0, 0.0
    for m in window(n):
        draw = erasures.sample_regions(
            edges, n, m, COUPLINGS, cfg["samples"], thin=CHAIN_THIN, seed=[SEED, 7, n, m]
        )
        warned.extend(f"m = {m}: {w}" for w in draw.warnings)
        finite = draw.exchange_acceptance[np.isfinite(draw.exchange_acceptance)]
        if finite.size:
            weakest = min(weakest, float(finite.min()))
        rungs = max(rungs, int(draw.ladder.size))
        b0_mean, b0_var = erasures.uniform_boundary_moments(edges, n, m)
        b0 = draw.boundaries[0]
        worst_b0 = max(worst_b0, abs(b0.mean() - b0_mean) / math.sqrt(b0_var / b0.size))
    return {
        "n": n,
        "beta": beta,
        "sizes": len(window(n)),
        "warnings": warned,
        "min_exchange_acceptance": weakest,
        "max_rungs": rungs,
        "j0_boundary_max_z": worst_b0,
    }


def interval_probability(n: int) -> list[dict]:
    """How close a coupling J is to the ring's J -> inf limit: P(one interval) = n e^{-2J} / Z."""
    rows = []
    for m in sorted({max(1, n // 8), n // 4, n // 2}):
        row: dict = {"n": n, "m": m}
        for coupling in (*COUPLINGS[1:], 4.0, 8.0):
            values, probs = erasures.ring_boundary_distribution(n, m, coupling)
            row[f"J = {coupling:g}"] = float(probs[values == 2].sum())
        row["J_x"] = 0.5 * math.log(max((m - 1) * (n - m - 1) / 2, 1.0))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------


def power_fit(ns, means, ses) -> dict | None:
    """Weighted least squares of log Q = a + gamma log n (sigma_log = se / mean)."""
    rows = [
        (n, m, s)
        for n, m, s in zip(ns, means, ses, strict=True)
        if m is not None and s is not None and m > 0 and s > 0 and math.isfinite(m)
    ]
    if len(rows) < 2:
        return None
    x = np.log([r[0] for r in rows])
    y = np.log([r[1] for r in rows])
    sigma = np.array([r[2] / r[1] for r in rows])
    w = 1 / sigma**2
    design = np.stack([np.ones_like(x), x], axis=1)
    cov = np.linalg.inv(design.T @ (design * w[:, None]))
    coef = cov @ (design.T @ (w * y))
    resid = (y - design @ coef) / sigma
    dof = len(rows) - 2
    chi2 = float(resid @ resid)
    scale = max(1.0, math.sqrt(chi2 / dof)) if dof > 0 else 1.0
    local = [
        {
            "n": [int(rows[i][0]), int(rows[i + 1][0])],
            "slope": float((y[i + 1] - y[i]) / (x[i + 1] - x[i])),
            "se": float(math.hypot(sigma[i], sigma[i + 1]) / (x[i + 1] - x[i])),
        }
        for i in range(len(rows) - 1)
    ]
    return {
        "sizes": [int(r[0]) for r in rows],
        "gamma": float(coef[1]),
        "se": float(math.sqrt(cov[1, 1]) * scale),
        "chi2_per_dof": chi2 / dof if dof > 0 else None,
        "amplitude": float(math.exp(coef[0])),
        "local": local,
    }


QUANTITIES = {
    "d_cont": ("d_cont", "mean", "se"),
    "gh_l": ("gh_l", "mean", "se"),
    "gh_l_min": ("gh_l_min", "mean", "se"),
    "lf_star": ("lf_star", "value", "se"),
}


def fits(cases: list[dict]) -> list[dict]:
    out = []
    keys = sorted(
        {(c["family"], c["beta"], c["p"]) for c in cases},
        key=lambda x: (x[0], x[1] or 0.0, x[2]),
    )
    for family, beta, p in keys:
        if p not in MIXED:
            continue
        rows = sorted(
            (c for c in cases if (c["family"], c["beta"], c["p"]) == (family, beta, p)),
            key=lambda c: c["n"],
        )
        entry: dict = {"family": family, "beta": beta, "p": p}
        for name, (key, m_key, s_key) in QUANTITIES.items():
            for suffix, chosen in (("", rows), ("_large", [c for c in rows if c["n"] >= LARGE_N])):
                entry[name + suffix] = power_fit(
                    [c["n"] for c in chosen],
                    [c[key][m_key] for c in chosen],
                    [c[key][s_key] for c in chosen],
                )
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------


def fmt(value, se=None, digits=3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    text = f"{value:.{digits}f}"
    if se is not None and math.isfinite(se):
        text += f" ± {se:.{digits}f}"
    return text


def gh_rate(p: float) -> float | None:
    """The Gullans-Huse near-critical guide to the code rate, or None outside its range."""
    return GH["A"] * (GH["p_c"] - p) ** GH["nu"] if GH_MIN_P <= p < GH["p_c"] else None


def prob_text(x: float) -> str:
    """One format for every probability of the table: four significant digits, kept
    (0.01000, 0.5360, 0.9999), and scientific where it is small (6.690e-05)."""
    return f"{x:#.4g}"


def fit_text(f: dict | None) -> str:
    if not f:
        return "-"
    chi = f" (χ²/dof {f['chi2_per_dof']:.1f})" if f["chi2_per_dof"] is not None else ""
    return f"{f['gamma']:.2f} ± {f['se']:.2f}{chi}"


def local_text(f: dict | None) -> str:
    if not f:
        return "-"
    return ", ".join(f"{s['slope']:.2f}±{s['se']:.2f}" for s in f["local"])


def sigma_text(value: float, target: float, se: float) -> str:
    z = (value - target) / se
    return f"{z:+.1f} σ" + (" (consistent)" if abs(z) <= 2 else "")


def _case(cases, family, n, p, beta=None):
    for c in cases:
        if (c["family"], c["n"], c["p"], c["beta"]) == (family, n, p, beta):
            return c
    return None


def verdicts(r: dict) -> list[str]:
    """What agrees with the literature, what does not, and what is new, from the numbers."""
    cases = r["cases"]
    ring = [c for c in cases if c["family"] == "ring"]
    full = [c for c in ring if c["correlated"]]
    big = max(c["n"] for c in ring)
    big_full = max(c["n"] for c in full)
    lines = []
    q = {c["p"]: c["q_plus"] for c in ring if c["n"] == big}
    lines.append(
        f"* **Phase (agrees with p_c ~ 0.16):** at n = {big}, depth 4n, the fraction of "
        "trajectories holding a code is "
        + ", ".join(f"{q[p]:.2f} at p = {p:g}" for p in sorted(q))
        + "."
    )
    rate = {c["p"]: c["k_over_n"] for c in ring if c["n"] == big}
    if 0.12 in rate:
        pred = gh_rate(0.12)
        assert pred is not None
        outside = [p for p in sorted(rate) if gh_rate(p) is None and p < GH["p_c"]]
        ratio = rate[0.12]["mean"] / pred
        lines.append(
            f"* **Code rate:** ⟨k⟩/n = {fmt(rate[0.12]['mean'], rate[0.12]['se'], 4)} at p = 0.12, "
            f"n = {big}, against {pred:.3f} from Gullans-Huse's near-critical form "
            f"7.3 (p_c - p)^1.28, a ratio of {ratio:.2f}. The form is a guide to their Fig. 2 (A ≈ 7.3 "
            "is a caption value; ν = 1.28(2) alone moves it by ±6.5% at p = 0.12), so this agrees "
            "at the level the form can be read."
            + (
                f" At p = {' and '.join(f'{p:g}' for p in outside)} the form is outside its range "
                f"(it gives {GH['A'] * GH['p_c'] ** GH['nu']:.2f} at p = 0), so those are not "
                "compared (the guide column of section 1 shows '-')."
                if outside
                else ""
            )
        )
    for f in [f for f in r["fits"] if f["family"] == "ring"]:
        parts = []
        for name, label in (("lf_star", "|A|*"), ("gh_l", "⟨l⟩"), ("d_cont", "d_cont")):
            every, large = f[name], f.get(name + "_large")
            if not every:
                continue
            text = f"{label}: {fit_text(every)}"
            best = large or every
            if large:
                text += f", n >= {LARGE_N}: {fit_text(large)}"
            if name == "lf_star":
                text += (
                    f" (against Li-Fisher's 0.38: {sigma_text(best['gamma'], 0.38, best['se'])}; "
                    f"against 1/3: {sigma_text(best['gamma'], 1 / 3, best['se'])})"
                )
            if name == "gh_l":
                lo, hi = LITERATURE["gullans_huse_a"]
                where = (
                    "inside"
                    if lo - 2 * best["se"] <= best["gamma"] <= hi + 2 * best["se"]
                    else ("above" if best["gamma"] > hi else "below")
                )
                text += f" ({where} Gullans-Huse's [1/3, 1/2] within 2 σ)"
            parts.append(text)
        sizes = (f["lf_star"] or f["gh_l"] or f["d_cont"] or {}).get("sizes", [])
        lines.append(
            f"* **Contiguous lengths, p = {f['p']:g}** (exponents in n, n = "
            f"{', '.join(map(str, sizes))}): " + "; ".join(parts) + "."
        )
    c8 = _case(cases, "ring", big, 0.08) or _case(cases, "ring", big_full, 0.08)
    if c8 and c8["d_cont"]["mean"]:
        lines.append(
            "* **Definitions matter:** at n = "
            f"{c8['n']}, p = 0.08, |A|* = {fmt(c8['lf_star']['value'], c8['lf_star']['se'], 1)} "
            f"(mean information crosses 1 bit), ⟨l⟩ = {fmt(c8['gh_l']['mean'], c8['gh_l']['se'], 1)} "
            f"(shortest failing interval through a site), d_cont = "
            f"{fmt(c8['d_cont']['mean'], c8['d_cont']['se'], 1)} (shortest failing interval "
            "anywhere): Li-Fisher's crossing tracks the site-averaged length, not the per-code "
            "minimum. Li and Fisher average depths 6L-8L and reach L = 1024; this run uses depth 4n "
            f"(Gullans-Huse's) and n <= {big}, so effective exponents at these sizes are compared."
        )
    strict = [
        c for c in full if c["p"] == 0.08 and c["distance"]["d_below_d_cont_fraction"] is not None
    ]
    if strict:
        text = ", ".join(
            f"n = {c['n']}: {c['distance']['d_below_d_cont_fraction']:.2f} (d known for "
            f"{100 * c['distance']['exact_fraction']:.0f}% of the codes)"
            for c in strict
        )
        lines.append(
            "* **d < d_cont (new numbers for Li-Fisher's inequality d <= d_cont):** at p = 0.08, "
            "among the codes whose distance d the exact counts and orderings pin down (low = high), "
            f"the fraction whose smallest uncorrectable set is not an interval: {text}. Those codes "
            "are the ones with small d, so this is not the fraction over all codes; for the rest "
            "the bounds leave d between d low (the exact counts' cutoff plus one) and d high (the "
            "smallest of the orderings' first failures, a weak bound that grows like n here)."
        )
    li = LITERATURE["li_fisher_gamma1"]
    for f in [f for f in r["fits"] if f["family"] == "ring" and f["p"] == 0.12]:
        fit = f.get("lf_star_large") or f["lf_star"]
        last = (fit or {}).get("local", [])
        if fit and last:
            tail = last[-1]
            xi = (GH["p_c"] - 0.12) ** -GH["nu"]
            scope = f"n >= {LARGE_N}" if f.get("lf_star_large") else "every n"
            z = (fit["gamma"] - li) / fit["se"]
            z_fit = (tail["slope"] - fit["gamma"]) / tail["se"]
            z_li = (tail["slope"] - li) / tail["se"]
            undecided = abs(z_fit) <= 2 and abs(z_li) <= 2
            lines.append(
                f"* **p = 0.12, an open question:** the {scope} exponent of |A|* is "
                f"{fit_text(fit)}, {abs(z):.1f} σ {'above' if z > 0 else 'below'} Li-Fisher's "
                f"{li:g} (their single value for p = 0.04-0.12). Two possible causes are not "
                "tested here: a near-critical crossover (correlation length "
                f"(p_c - p)^-ν ≈ {xi:.0f} sites) and the shallower depth (4n here, against their "
                f"6L-8L). The last local slope of |A|* (n = {tail['n'][0]} to {tail['n'][1]}) is "
                f"{tail['slope']:.2f} ± {tail['se']:.2f}, {abs(z_fit):.1f} σ from "
                f"{fit['gamma']:.2f} and {abs(z_li):.1f} σ from {li:g}"
                + (", so it is consistent with either value." if undecided else ".")
            )
    gaps = []
    for f in [f for f in r["fits"] if f["family"] == "ring"]:
        a, b = f.get("d_cont_large") or f["d_cont"], f.get("gh_l_large") or f["gh_l"]
        if a and b:
            diff = a["gamma"] - b["gamma"]
            gaps.append(
                f"p = {f['p']:g}: {diff:+.2f} ({diff / math.hypot(a['se'], b['se']):+.1f} σ)"
            )
    if gaps:
        lines.append(
            "* **⟨l_min⟩ against ⟨l⟩:** Gullans-Huse find that ⟨l_min⟩ (here d_cont) scales like ⟨l⟩; "
            f"the exponent of d_cont minus that of ⟨l⟩ (n >= {LARGE_N}) is " + ", ".join(gaps) + "."
        )
    labels = ("J = 0", "J = 0.5", "J = 1", "J = 2", INF)
    steps = rises = 0
    for c in full:
        if c["q_plus"] == 0:
            continue
        for i in range(len(c["sizes"])):
            means = [c["curves"][label]["mean"][i] for label in labels]
            ses = [c["curves"][label]["se"][i] for label in labels]
            for a in range(len(labels) - 1):
                steps += 1
                rises += means[a + 1] >= means[a] - 2 * math.hypot(ses[a], ses[a + 1])
    cells, falls = [], []  # (n, p, m, J = 0 mean, J = 2 mean, increase, its standard error)
    for c in full:
        if c["q_plus"] == 0:
            continue
        zero, two = c["curves"]["J = 0"], c["curves"]["J = 2"]
        for i, m in enumerate(c["sizes"]):
            if None in (zero["mean"][i], two["mean"][i], zero["se"][i], two["se"][i]):
                continue
            cell = (
                c["n"], c["p"], m, zero["mean"][i], two["mean"][i],
                two["mean"][i] - zero["mean"][i], math.hypot(zero["se"][i], two["se"][i]),
            )  # fmt: skip
            cells.append(cell)
            if 2 * m == c["n"] and cell[5] < 0:
                falls.append(cell)
    if steps and cells:
        up = [x for x in cells if x[5] > 2 * x[6]]
        top = max(cells, key=lambda x: x[5])
        text = (
            "* **New: clustered erasures hurt the brickwork codes.** From J = 0 to J = 2 the mean "
            f"P_{{C,J}}(m) rises by more than 2 standard errors in {len(up)} of {len(cells)} "
            f"(n, p, m) cells of the selected sizes (k > 0), by as much as {top[5]:.3f} "
            f"(n = {top[0]}, p = {top[1]:g}, m = {top[2]}: {top[3]:.3f} -> {top[4]:.3f}); along "
            f"J = 0, 0.5, 1, 2, inf no step falls by more than 2 standard errors in {rises} of "
            f"{steps} steps."
        )
        if falls:
            near = " near p_c" if all(x[1] >= GH_MIN_P for x in falls) else ""
            text += (
                f" At m = n/2{near} the mean instead falls slightly with J from 0 to 2: "
                + "; ".join(
                    f"n = {n}, p = {p:g}, m = {m}: {a:.4f} -> {b:.4f} ({d / s:+.1f} standard errors)"
                    for n, p, m, a, b, d, s in falls
                )
                + ". "
            )
        else:
            text += " "
        text += (
            "ε_δ(J) falls with J (section 5). The permutation-invariant control (all-to-all "
            "circuits, `tests/test_brickwork_correlated.py`) shows no J dependence, as it must."
        )
        lines.append(text)
    j_rows = []
    for c in full:
        if c["p"] not in MIXED or c["q_plus"] == 0 or c["n"] != big_full:
            continue
        zero = c["curves"]["J = 0"]["mean"]
        i = int(np.argmin([abs(v - 0.3) for v in zero]))  # the most informative selected size
        vals = [c["curves"][label]["mean"][i] for label in ("J = 0", "J = 1", "J = 2", INF)]
        j_rows.append((c["n"], c["p"], c["sizes"][i], vals))
    if j_rows:
        text = "; ".join(
            f"p = {p:g}, m = {m}: " + " / ".join(f"{v:.3f}" for v in vals)
            for n, p, m, vals in j_rows
        )
        jx = next(
            (
                row
                for row in r["interval_probability"]
                if (row["n"], row["m"]) == (big_full, big_full // 4)
            ),
            None,
        )
        limit = (
            f" (section 4: at n = {big_full}, m = {big_full // 4} a region drawn at J = 2 is one "
            f"interval with probability {jx['J = 2']:.1e}; one interval and the two-run regions "
            f"weigh the same at J_x = {jx['J_x']:.2f}, which grows like ln n)"
            if jx
            else ""
        )
        lines.append(
            f"* **New: uniform against clustered erasures at n = {big_full}** (no published "
            "monitored-circuit numbers exist to compare): P_{C,J}(m) for J = 0 / 1 / 2 / inf at the "
            f"selected size where J = 0 is nearest 0.3, {text}. A finite J stays far from the "
            f"interval limit at large n{limit}."
        )
    tol = [c for c in full if "phi" in c and c["p"] == 0.08 and c["family"] == "ring"]
    for c in tol:
        ph = c["phi"]
        i0, iinf = ph["labels"].index("J = 0"), ph["labels"].index(INF)
        parts = []
        for delta, e in ph["tolerance"].items():
            parts.append(
                f"ε_{delta} = {fmt(e['value'][i0], e['se'][i0], 4)} (J = 0, independent erasure) "
                f"against {fmt(e['value'][iinf], e['se'][iinf], 4)} (J = inf, one interval of "
                "Bin(n, ε) qubits)"
            )
        lines.append(f"* **Tolerance, n = {c['n']}, p = 0.08:** " + "; ".join(parts) + ".")
    trend = []
    for p in MIXED:
        rows = sorted(
            (c for c in full if "phi" in c and c["p"] == p and c["family"] == "ring"),
            key=lambda c: c["n"],
        )
        if len(rows) >= 2:
            a, b = rows[0], rows[-1]
            pick = []
            for label in ("J = 0", INF):
                i = a["phi"]["labels"].index(label)
                ta, tb = a["phi"]["tolerance"]["0.1"], b["phi"]["tolerance"]["0.1"]
                pick.append(f"{label}: {ta['value'][i]:.3f} -> {tb['value'][i]:.3f}")
            trend.append(f"p = {p:g} ({', '.join(pick)})")
    if trend:
        lines.append(
            "* **New: the tolerances move apart with n.** ε_0.1 from n = "
            f"{min(PHI_SIZES)} to {max(PHI_SIZES)}: " + "; ".join(trend) + ". Under independent "
            "erasure the tolerance grows with n (two sizes only; consistent with a finite erasure "
            "threshold despite a sub-extensive distance, the degeneracy Gullans and Huse point "
            "to); under one interval of Bin(n, ε) qubits it shrinks, as an interval of about εn "
            "qubits outgrows the contiguous lengths, which grow sub-linearly (section 2)."
        )
    ws = [c for c in cases if c["family"] == "ws" and c["correlated"] and c["p"] == 0.12]
    contrast = []
    for c in sorted(ws, key=lambda c: (c["beta"], c["n"])):
        if c["n"] != big_full or c["q_plus"] == 0:
            continue
        zero = c["graph_curves"]["J = 0"]["mean"]
        i = int(np.argmin([abs(v - 0.3) for v in zero]))
        ring_row = [c["curves"][k]["mean"][i] for k in ("J = 0", "J = 2", INF)]
        graph_row = [c["graph_curves"][k]["mean"][i] for k in ("J = 0", "J = 2")]
        contrast.append(
            f"β = {c['beta']:g}, m = {c['sizes'][i]}: label ring J = 0 / 2 / inf "
            + " / ".join(f"{v:.3f}" for v in ring_row)
            + ", interaction graph J = 0 / 2 "
            + " / ".join(f"{v:.3f}" for v in graph_row)
        )
    if contrast:
        lines.append(
            f"* **Watts-Strogatz contrast (n = {big_full}, p = 0.12):** "
            + "; ".join(contrast)
            + ". At β = 1 the labels carry no geometry, so erasures clustered on the label ring "
            "see a nearly relabeling-invariant ensemble and P does not move with J; on the "
            "circuit's own interaction graph clustering still hurts."
        )
    lines.append(
        "* **Could not be compared:** the amplitudes of the lengths (figures only in the papers), "
        "p_c itself (no I3 crossing is run here), and any random-region (J = 0) or "
        "correlated-erasure statistic of monitored brickwork codes (none found in the literature "
        "survey of `docs/observables/correlated-erasure.md`, section 11). The random-code "
        "formula of Gullans et al. is checked in "
        "`tests/test_brickwork_correlated.py` instead, with its own definition of failure, "
        "1 - E[2^-I(A:R)]."
    )
    return lines


def curve_rows(out: list[str], c: dict, rows) -> None:
    out += [
        "| curve | " + " | ".join(f"m = {m}" for m in c["sizes"]) + " |",
        "|---|" + "---|" * len(c["sizes"]),
    ]
    for label, row in rows:
        cells = " | ".join(fmt(v, s, 3) for v, s in zip(row["mean"], row["se"], strict=True))
        out.append(f"| {label} | {cells} |")
    out.append("")


def render_markdown(r: dict) -> str:
    cfg = r["settings"]
    env = r["environment"]
    cases = r["cases"]
    out = [
        "# Correlated-erasure baseline: the brickwork ring against the literature",
        "",
        f"Generated by `benchmarks/correlated_erasure_baseline.py` on {env['date']} "
        f"({env.get('cpu_model')}, {env['threads']} threads, sparsegf2 {env['sparsegf2']}); "
        f"{r['total_seconds'] / 60:.1f} minutes{' (quick smoke run)' if r['quick'] else ''}. "
        "Physics checks with exact answers (the J -> inf intervals, J = 0, the samplers, the "
        "random-code formula) are in `tests/test_brickwork_correlated.py`.",
        "",
        '**Protocol.** Ring `cycle(n)`, `schedule="matching"` (brickwork), uniform two-qubit '
        "Cliffords, Z measurements at rate p after every layer, 4n layers, purification picture, "
        f"final layer. n = {cfg['sizes']} (and {cfg['scaling_sizes']} for the contiguous lengths "
        f"only), p = {cfg['rates']}; {cfg['trajectories']} trajectories per ring (n, p), "
        f"{cfg['ws_trajectories']} per Watts-Strogatz (β, n, p). Seeds: "
        f"`run(seed=[{SEED}, family, n, round(1000 p), round(100 β)])`, family 0 = ring, "
        "1 = Watts-Strogatz, whose graph i is "
        f"`watts_strogatz(n, 4, β, seed=[{SEED}, n, round(100 β), i])`. Probe "
        f"`correlated_erasure(couplings={list(COUPLINGS)}, samples={cfg['samples']})` "
        "(ring: exact sampler, every size); uniform P_C(m) also from "
        f"`erasure_first_failure(samples={cfg['orderings']})` with "
        f"`uncorrectable_region_counts(max_size=exhaustive_cutoff(n, {cfg['budget']}))`. "
        "Means over codes with k > 0 unless said; ± is one standard error.",
        "",
        "## Summary",
        "",
        *verdicts(r),
        "",
        "## 1. Codes, distances and contiguous lengths (ring)",
        "",
        "q+: fraction with k > 0. d_cont: smallest failing interval (k > 0). ⟨l⟩: Gullans-Huse "
        "site-averaged shortest failing interval containing x, k = 0 counted as 0. \\|A\\|*: "
        "Li-Fisher, the length where the mean I(interval : R) over all trajectories reaches 1 bit "
        "(bootstrap error). d low / high: uniform distance bounds (exact counts up to the cutoff, "
        "then the smallest first failure); 'd exact': the fraction where they meet; "
        "'d < d_cont': the fraction of those whose smallest uncorrectable set is not an interval. "
        "GH: Gullans-Huse's near-critical guide to ⟨k⟩/n, shown only inside the range where it is "
        f"read (p from {GH_MIN_P:g} to p_c = {GH['p_c']:g}), '-' elsewhere.",
        "",
        "| n | p | q+ | ⟨k⟩/n | GH 7.3(p_c-p)^1.28 | d_cont | ⟨l⟩ | \\|A\\|* | d low | d high "
        "| d exact | d < d_cont |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in [c for c in cases if c["family"] == "ring"]:
        d = c["distance"]
        pred = gh_rate(c["p"])
        out.append(
            f"| {c['n']} | {c['p']:g} | {c['q_plus']:.2f} | "
            f"{fmt(c['k_over_n']['mean'], c['k_over_n']['se'])} | "
            f"{fmt(pred) if pred else '-'} | {fmt(c['d_cont']['mean'], c['d_cont']['se'], 2)} | "
            f"{fmt(c['gh_l']['mean'], c['gh_l']['se'], 2)} | "
            f"{fmt(c['lf_star']['value'], c['lf_star']['se'], 2)} | "
            f"{fmt(d['low']['mean'], None, 2)} | {fmt(d['high']['mean'], None, 2)} | "
            f"{fmt(d['exact_fraction'], None, 2)} | {fmt(d['d_below_d_cont_fraction'], None, 2)} |"
        )
    out += [
        "",
        "## 2. Scaling with n (mixed phase)",
        "",
        "Weighted least squares of log Q against log n (the error is scaled by sqrt(χ²/dof) when "
        "that exceeds 1), and the local slopes between neighbouring n. Literature: Li-Fisher "
        "γ1 ≈ 0.38 for \\|A\\|* (p = 0.04, 0.08, 0.12; L up to 1024; depths 6L-8L); "
        "Li-Vijay-Fisher β = 1/3 (directed polymer theory); Gullans-Huse a ∈ [1/3, 1/2] for ⟨l⟩ "
        "deep in the mixed phase and ⟨l⟩ ∝ L near p_c, with ⟨l_min⟩ (= d_cont, k = 0 as 0) "
        "scaling like ⟨l⟩. Watts-Strogatz rows use the qubit labels' ring.",
        "",
        f"| geometry | p | length | exponent, all n | exponent, n ≥ {LARGE_N} | local slopes "
        "(neighbouring n) | literature |",
        "|---|---|---|---|---|---|---|",
    ]
    literature = {
        "lf_star": "Li-Fisher 0.38; Li-Vijay-Fisher 1/3",
        "gh_l": "Gullans-Huse 1/3 to 1/2 (deep), ~1 (near p_c)",
        "d_cont": "as ⟨l_min⟩",
        "gh_l_min": "Gullans-Huse: as ⟨l⟩",
    }
    names = {"lf_star": "\\|A\\|*", "gh_l": "⟨l⟩", "d_cont": "d_cont", "gh_l_min": "⟨l_min⟩"}
    for f in r["fits"]:
        geometry = "ring" if f["family"] == "ring" else f"WS β = {f['beta']:g} (labels)"
        for key, name in names.items():
            out.append(
                f"| {geometry} | {f['p']:g} | {name} | {fit_text(f[key])} | "
                f"{fit_text(f.get(key + '_large'))} | {local_text(f[key])} | {literature[key]} |"
            )
    out += [
        "",
        "## 3. P_{C,J}(m) on the ring: uniform, correlated, contiguous",
        "",
        "Mean over codes with k > 0. 'uniform (orderings)': first-failure orderings with exact "
        "counts; J = 0 to 2: the probe (exact ring sampler); J = inf: the failing fraction of the "
        "n intervals (exact). Sizes m = n/32, n/16, n/8, n/4, 3n/8, n/2.",
        "",
    ]
    for c in [c for c in cases if c["family"] == "ring" and c["q_plus"] > 0 and c["correlated"]]:
        out += [f"**n = {c['n']}, p = {c['p']:g}** ({c['d_cont']['count']} codes)", ""]
        curve_rows(out, c, c["curves"].items())
    out += [
        "## 4. How far a finite J is from the interval limit",
        "",
        "Exact, from the ring's run distribution N(n, m, r) e^{-2Jr} / Z: the probability that a "
        "region drawn at J is a single interval. J_x = (1/2) ln((m-1)(n-m-1)/2) is where one "
        "interval and all two-run regions weigh about the same; it grows like ln n, so a fixed J "
        "moves away from the contiguous limit as n grows.",
        "",
        "| n | m | J = 0.5 | J = 1 | J = 2 | J = 4 | J = 8 | J_x |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in r["interval_probability"]:
        out.append(
            f"| {row['n']} | {row['m']} | {prob_text(row['J = 0.5'])} | {prob_text(row['J = 1'])} | "
            f"{prob_text(row['J = 2'])} | {prob_text(row['J = 4'])} | {prob_text(row['J = 8'])} | "
            f"{row['J_x']:.2f} |"
        )
    out += [
        "",
        "## 5. Φ_{C,J}(ε) and ε_δ(J)",
        "",
        "Φ_J(ε) = Σ_m C(n,m) ε^m (1-ε)^(n-m) P_J(m) of the mean curve over codes with k > 0 "
        "(the binomially mixed correlated channel; J = 0 is independent erasure); ε_δ(J) = "
        "sup{ε : Φ_J(ε) ≤ δ}, bootstrap errors over codes. 'monotone': whether the mean curve "
        "is nondecreasing in m (P_J(m) need not be for J ≠ 0, and a sampled curve need not be).",
        "",
    ]
    for c in [c for c in cases if "phi" in c]:
        ph = c["phi"]
        name = "ring" if c["family"] == "ring" else f"WS β = {c['beta']:g}"
        out += [
            f"**{name}, n = {c['n']}, p = {c['p']:g}** ({ph['codes']} codes)",
            "",
            "| curve | "
            + " | ".join(f"Φ({e:g})" for e in ph["epsilons"])
            + " | "
            + " | ".join(f"ε_{d}" for d in ph["tolerance"])
            + " | monotone |",
            "|---|" + "---|" * (len(ph["epsilons"]) + len(ph["tolerance"]) + 1),
        ]
        for i, label in enumerate(ph["labels"]):
            phis = " | ".join(f"{v:.3g}" for v in ph["phi"][i])
            tols = " | ".join(fmt(t["value"][i], t["se"][i], 4) for t in ph["tolerance"].values())
            mono = "yes" if all(t["monotone"][i] for t in ph["tolerance"].values()) else "no"
            out.append(f"| {label} | {phis} | {tols} | {mono} |")
        out.append("")
    ws = [c for c in cases if c["family"] == "ws"]
    if ws:
        out += [
            "## 6. Watts-Strogatz contrast (k = 4)",
            "",
            'One graph per trajectory, `schedule="random_edges"`, n/2 gates per layer, otherwise '
            "as above. 'graph': erasures correlated on the circuit's own interaction graph (swap "
            f"chain with replica exchange, sizes every n/32, {CHAIN_THIN} sweeps between records, "
            "linear interpolation between sizes for Φ); 'ring': on the ring of the qubit labels "
            "(exact sampler); J = inf on the ring: the label intervals. d_cont: along the labels.",
            "",
            "| β | n | p | q+ | ⟨k⟩/n | d_cont (labels) | ⟨l⟩ | \\|A\\|* | d low | d high |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for c in ws:
            out.append(
                f"| {c['beta']:g} | {c['n']} | {c['p']:g} | {c['q_plus']:.2f} | "
                f"{fmt(c['k_over_n']['mean'], c['k_over_n']['se'])} | "
                f"{fmt(c['d_cont']['mean'], c['d_cont']['se'], 2)} | "
                f"{fmt(c['gh_l']['mean'], c['gh_l']['se'], 2)} | "
                f"{fmt(c['lf_star']['value'], c['lf_star']['se'], 2)} | "
                f"{fmt(c['distance']['low']['mean'], None, 2)} | "
                f"{fmt(c['distance']['high']['mean'], None, 2)} |"
            )
        out.append("")
        for c in [c for c in ws if c["q_plus"] > 0]:
            out += [
                f"**β = {c['beta']:g}, n = {c['n']}, p = {c['p']:g}** ({c['d_cont']['count']} codes)",
                "",
            ]
            rows = [(f"ring {k}", v) for k, v in c["curves"].items()]
            rows += [(f"graph {k}", v) for k, v in c["graph_curves"].items()]
            curve_rows(out, c, rows)
        out += [
            "**Graph-side mixing check** of the swap chain (the first graph of each collection, "
            "every probe size, the probe's settings; the chain depends on the graph, the size and "
            "the settings only, not on p or the code). 'J=0 boundary z': the largest deviation of "
            "the J = 0 mean boundary from its exact value, in standard errors.",
            "",
            "| β | n | sizes | max rungs | min exchange acceptance | J=0 boundary z | warnings |",
            "|---|---|---|---|---|---|---|",
        ]
        for mc in r["mixing"]:
            out.append(
                f"| {mc['beta']:g} | {mc['n']} | {mc['sizes']} | {mc['max_rungs']} | "
                f"{mc['min_exchange_acceptance']:.2f} | {mc['j0_boundary_max_z']:.1f} | "
                f"{len(mc['warnings'])} |"
            )
        out += [
            "",
            "Only J = 0 is compared with an exact value, and with a standard error of independent "
            "draws although the records come from a chain; the J = 2 numbers of the graph rows "
            "above and of the summary rest on the absence of warnings, which is no proof of "
            "equilibrium.",
        ]
        warned = [w for mc in r["mixing"] for w in mc["warnings"]]
        if warned:
            out += ["", f"Warnings ({len(warned)}; the first five):", ""]
            out += [f"* {w}" for w in warned[:5]]
        out.append("")
    out += [
        "## Definitions, kept apart",
        "",
        "* Failure here is F_C(A) = 1[I(A:R) > 0] (some logical information is lost). Gullans, "
        "Krastanov, Huse, Jiang and Flammia (PRX 11, 031066, 2021) define the failure probability "
        "as 1 - E[2^-I(A:R)] (one minus the optimal recovery probability, their Eqs. 18 and 23); "
        "at the critical size of a random [[64, 32]] code theirs is 0.390 and ours 0.711, from "
        "the same random-matrix rank law.",
        "* d_cont (smallest failing interval) is Li-Fisher's algebraic contiguous distance and "
        "Gullans-Huse's l_min; Li-Fisher's numbers are for \\|A\\|*, the crossing of the mean "
        "information; Gullans-Huse's ⟨l⟩ averages the shortest failing interval through each site.",
        "",
    ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, bool | np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating | float):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def run_case(family: str, n: int, p: float, beta, cfg: dict, rng, correlated=True) -> dict:
    if family == "ring":
        spec = ring_spec(n, p, cfg, correlated)
        count = cfg["trajectories"]
    else:
        spec = ws_spec(n, p, beta, cfg)
        count = cfg["ws_trajectories"]
    start = time.perf_counter()
    res = spec.run(count, seed=case_seed(family, n, p, beta))
    summary = summarize(res, spec, family, n, p, beta, cfg, rng)
    summary["seconds"] = time.perf_counter() - start
    label = "ring" if family == "ring" else f"WS beta={beta:g}"
    d = summary["d_cont"]["mean"]
    star = summary["lf_star"]["value"]
    print(
        f"  {label:12s} n={n:4d} p={p:<5g} q+={summary['q_plus']:.2f} "
        f"d_cont={fmt(d, None, 2)} |A|*={fmt(star, None, 2)} ({summary['seconds']:.1f} s)",
        flush=True,
    )
    return summary


def write(results: dict, out: Path) -> None:
    # newline="\n": the same bytes on every platform (the repository keeps LF)
    (out / "correlated_erasure_baseline.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8", newline="\n"
    )
    (out / "correlated_erasure_baseline.md").write_text(
        render_markdown(results), encoding="utf-8", newline="\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(HERE / "results"), help="output directory")
    ap.add_argument("--quick", action="store_true", help="tiny sizes, for a smoke test")
    ap.add_argument("--sizes", type=int, nargs="*", default=None, help="system sizes n")
    ap.add_argument(
        "--scaling-sizes", type=int, nargs="*", default=None,
        help="extra ring sizes for the contiguous lengths only (default 512 1024)",
    )  # fmt: skip
    ap.add_argument("--rates", type=float, nargs="*", default=None, help="measurement rates p")
    ap.add_argument("--betas", type=float, nargs="*", default=None, help="Watts-Strogatz beta")
    ap.add_argument("--trajectories", type=int, default=256, help="ring trajectories per (n, p)")
    ap.add_argument(
        "--ws-trajectories",
        type=int,
        default=64,
        help="Watts-Strogatz trajectories per (beta, n, p)",
    )
    ap.add_argument("--samples", type=int, default=100, help="regions per (J, m) in the probe")
    ap.add_argument("--orderings", type=int, default=200, help="first-failure orderings")
    ap.add_argument("--budget", type=int, default=50_000, help="exact-count region budget")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples")
    ap.add_argument("--threads", type=int, default=None, help="numba threads (default: all)")
    ap.add_argument("--note", default="", help="a sentence about the machine's state")
    ap.add_argument(
        "--from-json",
        default=None,
        help="measure nothing: redo the fits and the markdown from this results file",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.from_json:
        results = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        results["fits"] = jsonable(fits(results["cases"]))
        write(results, out)
        print(f"rederived from {args.from_json}; wrote {out}", flush=True)
        return
    if args.threads:
        numba.set_num_threads(args.threads)
    sizes = list(args.sizes or ((12, 16, 24) if args.quick else SIZES))
    scaling = args.scaling_sizes
    if scaling is None:
        scaling = [32] if args.quick else list(SCALING_SIZES)
    trajectories = 8 if args.quick else args.trajectories
    cfg = {
        "sizes": sizes,
        "scaling_sizes": [n for n in scaling if n not in sizes],
        "rates": list(args.rates or ((0.08, 0.3) if args.quick else RATES)),
        "betas": list(args.betas if args.betas is not None else ((0.1,) if args.quick else BETAS)),
        "trajectories": trajectories,
        "ws_trajectories": 8 if args.quick else args.ws_trajectories,
        "samples": 20 if args.quick else args.samples,
        "orderings": 20 if args.quick else args.orderings,
        "budget": args.budget,
        "bootstrap": 20 if args.quick else args.bootstrap,
        "phi_sizes": [n for n in sizes if n in PHI_SIZES] or [max(sizes)],
    }
    start = time.perf_counter()
    rng = np.random.default_rng([SEED, 99])
    env = {
        "date": time.strftime("%Y-%m-%d"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        **cpu_info(),
        "logical_processors": os.cpu_count(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "sparsegf2": sg.__version__,
        "threads": numba.get_num_threads(),
        "note": args.note,
    }
    print(f"{env['cpu_model']}; {env['threads']} threads; sizes {sizes}", flush=True)
    cases = []
    for n in sizes:
        for p in cfg["rates"]:
            cases.append(run_case("ring", n, p, None, cfg, rng))
    for n in cfg["scaling_sizes"]:
        for p in cfg["rates"]:
            cases.append(run_case("ring", n, p, None, cfg, rng, correlated=False))
    for beta in cfg["betas"]:
        for n in sizes:
            for p in cfg["rates"]:
                cases.append(run_case("ws", n, p, beta, cfg, rng))
    mixing = [mixing_check(n, beta, cfg) for beta in cfg["betas"] for n in sizes]
    results = {
        "environment": env,
        "quick": args.quick,
        "seed": SEED,
        "settings": cfg,
        "couplings": list(COUPLINGS),
        "literature": LITERATURE,
        "gullans_huse_rate": GH,
        "cases": cases,
        "mixing": mixing,
        "interval_probability": [row for n in sizes for row in interval_probability(n)],
    }
    results["fits"] = fits(cases)
    results["total_seconds"] = time.perf_counter() - start
    results = jsonable(results)
    write(results, out)
    print(f"total {results['total_seconds'] / 60:.1f} min; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
