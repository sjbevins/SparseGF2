"""Shared helpers of the study service-layer, CLI and plot tests (test-only).

* :data:`MANIFEST`: a tiny study (n = 6, 8; beta = 0.2; three rates; three graphs of two
  circuits; two graphs per unit, so twelve units) that runs in about a second.
* :func:`build`: a store made and run through ``sparsegf2.study.api``.
* Independent references, written here from the definitions and sharing no code with
  ``sparsegf2.study.stats``: :func:`nested_reference` (v1.4 §6.3 and §6.4 by dictionaries and
  loops), :func:`km_reference` (the product-limit estimator of equally weighted trajectories, the
  weights a balanced design gives every trajectory) and :func:`pauli_strings` (generators from
  the canonical 0/1 rows).
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2.study import api as sa

MANIFEST: dict[str, Any] = {
    "experiment_id": "ws-api-test",
    "conditions": [{"n": [6, 8], "beta": ["0.2"], "p": ["0.1", "0.2", "0.3"]}],
    "depth": {"layers": {"per_n": 4}, "checkpoints": {"count": 5}},
    "trials": {"graphs": 3, "circuits_per_graph": 2},
    "seeds": {"master_graph_seed": 5, "master_circuit_seed": 6},
    "observables": {
        "tableau_snapshots": True,
        "contiguous_distance": True,
        "uniform_erasure": {"samples": 12, "exact_budget": 0},
    },
    "storage": {"graphs_per_unit": 2},
    "analysis": {"peak": {"bootstrap": 0}},
}


def manifest(**changes: Any) -> dict[str, Any]:
    """:data:`MANIFEST` with top-level sections replaced."""
    out = copy.deepcopy(MANIFEST)
    out.update(copy.deepcopy(changes))
    return out


def build(directory: Path, m: dict[str, Any] | None = None, *, run: bool = True) -> str:
    store = str(directory)
    sa.init(MANIFEST if m is None else m, store)
    if run:
        sa.run(store, num_threads=1)
    return store


def nested_reference(g: Any, c: Any, values: Any) -> tuple[float, float, int]:
    """The equal-graph-weight mean of v1.4 §6.3 (one replica per circuit) and the graph-cluster
    standard error of §6.4: the mean over graphs of each graph's mean over its circuits, and
    ``sqrt(sum_g (f_g - mu)^2 / (G (G - 1)))``; NaN values are left out (a graph without a
    value leaves the mean)."""
    per: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for gg, cc, v in zip(np.asarray(g).tolist(), np.asarray(c).tolist(),
                         np.asarray(values, dtype=float).tolist(), strict=True):  # fmt: skip
        if not math.isnan(v):
            per[int(gg)][int(cc)].append(v)
    graph_means = []
    for circuits in per.values():
        circuit_means = [sum(vs) / len(vs) for vs in circuits.values() if vs]
        if circuit_means:
            graph_means.append(sum(circuit_means) / len(circuit_means))
    count = len(graph_means)
    mu = sum(graph_means) / count
    if count < 2:
        return mu, math.nan, count
    se = math.sqrt(sum((f - mu) ** 2 for f in graph_means) / (count * (count - 1)))
    return mu, se, count


def km_reference(times: Any, observed: Any) -> tuple[list[float], list[float]]:
    """The product-limit estimate ``S(t_j) = prod_{i <= j} (1 - d_i / n_i)`` at the distinct
    event times, every trajectory with the same weight."""
    t = np.asarray(times, dtype=float)
    e = np.asarray(observed, dtype=bool)
    surv, out_t, out_s = 1.0, [], []
    for time in sorted(set(t[e].tolist())):
        at_risk = int(np.sum(t >= time))
        deaths = int(np.sum((t == time) & e))
        surv *= 1.0 - deaths / at_risk
        out_t.append(time)
        out_s.append(surv)
    return out_t, out_s


def km_median(times: Any, observed: Any) -> float | None:
    """``min{t : S(t) <= 1/2}`` of :func:`km_reference` (None if the curve stays above 1/2)."""
    for t, s in zip(*km_reference(times, observed), strict=True):
        if s <= 0.5 + 1e-12:
            return t
    return None


def pauli_strings(dense: Any) -> list[str]:
    """Pauli strings from 0/1 rows ``(s, 2N)`` (X bits, then Z bits), written letter by letter."""
    rows = np.asarray(dense, dtype=int)
    n = rows.shape[1] // 2
    out = []
    for row in rows:
        letters = []
        for q in range(n):
            x, z = row[q], row[n + q]
            letters.append({(0, 0): "_", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}[(x, z)])
        out.append("".join(letters))
    return out


def without_timing(obj: Any) -> Any:
    """``obj`` without the fields that measure time (they differ between two calls)."""
    timing = {"elapsed_seconds", "plan_seconds", "trajectories_per_second", "seconds"}
    if isinstance(obj, dict):
        return {k: without_timing(v) for k, v in obj.items() if k not in timing}
    if isinstance(obj, list):
        return [without_timing(v) for v in obj]
    return obj
