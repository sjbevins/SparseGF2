"""Honest runtime estimates, and graph collections bounded before they are built.

``estimate_ensemble`` and the ``max_seconds`` guard of ``run_ensemble`` price a run with
``api._trajectory_us`` (its probe terms are ``api._probe_us``) and ``api._setup_us``. The
estimate is meant to be deliberately pessimistic: the fit aims at about 3 times the one-thread
cost measured on the fitting machine (``benchmarks/probe_cost.py``, results in
``benchmarks/results/probe_cost.md``). Here every probe kind is timed at n = 512, one
trajectory on one thread, in the state that makes it expensive, and must cost no more than
its estimate. (JIT-free, the kernels run as Python, so only the estimates are checked.)

**Timings that hold on slower or busy machines.** A CI runner may be several times slower
than the fitting machine, or busy. Each measured time is therefore converted to fitting-machine
seconds: it is multiplied by ``REFERENCE_SECONDS / t_ref``, where ``t_ref`` is the time of a
fixed reference workload measured in the same test and ``REFERENCE_SECONDS`` its time on the
fitting machine. The reference is a GF(2) elimination written in this file, the inner loop of
the entropy kernels on a cache-resident matrix like the probes' at n = 512, so it does not
change when the package does. Every time is a median over repeats, and the fit's factor of
about 3 (3.5 or more at n = 512 for every case here) leaves room for the reference and the
probes to scale differently between machines.
"""

import math
import statistics
import time

import numba
import numpy as np
import pytest

from sparsegf2 import api, graphs

# ---------------------------------------------------------------------------
# every probe kind costs no more than its estimate (n = 512, one thread)
# ---------------------------------------------------------------------------

# Median seconds of _reference_workload() on the fitting machine (AMD Ryzen 9 9950X3D,
# Numba 0.68, one thread, 2026-10-15; ten medians of five runs spanned 2.76 to 2.86 ms).
REFERENCE_SECONDS = 2.79e-3


@numba.njit(cache=False)
def _eliminate(rows):
    """Rank of the bitset rows of ``rows`` (destroyed): the entropy kernels' elimination."""
    m, words = rows.shape
    rank = 0
    for i in range(m):
        lead_w = -1
        for w in range(words):
            if rows[i, w] != 0:
                lead_w = w
                break
        if lead_w < 0:
            continue
        rank += 1
        word = rows[i, lead_w]
        lead = word & (~word + np.uint64(1))
        for j in range(i + 1, m):
            if rows[j, lead_w] & lead:
                for w in range(lead_w, words):
                    rows[j, w] ^= rows[i, w]
    return rank


_REFERENCE_ROWS = np.random.default_rng(2026).integers(
    0, np.iinfo(np.uint64).max, size=(1024, 16), dtype=np.uint64, endpoint=True
)


def _reference_workload() -> int:
    return _eliminate(_REFERENCE_ROWS.copy())


def _one_trajectory(mc):
    return lambda: mc.run(1, seed=7, num_threads=1)


def _median_seconds(fn, runs: int) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def _specs(n: int, probe: dict, state: str, evaluations: int) -> tuple[dict, dict]:
    """The spec of a state without probes, and with ``evaluations`` evaluations of ``probe``.

    The states are those of benchmarks/probe_cost.py. ``single`` (k = 1, erasures that fail
    only at about n / 2 qubits) and ``purification`` (k = n, a volume-law state on 2n qubits)
    are scrambled states; layers without gates or measurements (``random_edges`` with no
    gates, rate 0) then leave the state as it is, so one probe at every layer evaluates it
    repeatedly. ``monitored`` is the purification state after n / 2 measured layers (an
    intermediate code), evaluated by copies of the probe at its final layer.
    """
    if state == "monitored":
        base = {
            "num_system": n,
            "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": [21, n]},
            "layers": n // 2,
            "measurement_rate": 0.05,
            "schedule": "random_edges",
            "reference": "purification",
            "scramble_gates": 2 * n * math.ceil(math.log2(n)),
            "checkpoints": "final",
        }
        return base, dict(base, probes=[{**probe, "name": f"p{i}"} for i in range(evaluations)])
    base = {
        "num_system": n,
        "graph": {"family": "cycle"},
        "layers": evaluations - 1,
        "measurement_rate": 0.0,
        "schedule": "random_edges",
        "gates_per_layer": 0,
        "reference": state,
        "scramble_gates": 2 * n * math.ceil(math.log2(n)),
        "checkpoints": "every_layer",
    }
    return base, dict(base, probes=[probe])


def _cases(n: int) -> list[tuple[dict, str, int]]:
    """(probe, state, evaluations): every probe kind in the state where it costs most, and the
    early-stopping erasure scans also on a code from measured dynamics. The evaluations make
    each measurement at least about a millisecond at n = 512."""
    q = n // 4
    counts = "uncorrectable_region_counts"
    return [
        ({"kind": "code_dimension"}, "purification", 1024),
        ({"kind": "entropy", "region": f"0:{n}"}, "purification", 4),
        ({"kind": "half_cut_entropy"}, "single", 16),
        ({"kind": "mutual_information", "a": f"0:{q}", "b": f"{2 * q}:{3 * q}"}, "single", 8),
        ({"kind": "tripartite_information"}, "purification", 2),
        ({"kind": "lost_information", "region": f"0:{n // 2}"}, "single", 8),
        ({"kind": "contiguous_distance"}, "single", 1),
        ({"kind": "contiguous_distance"}, "monitored", 2),
        ({"kind": "entropy_profile"}, "purification", 4),
        ({"kind": "erasure_lost_profile", "samples": 10}, "purification", 1),
        ({"kind": "erasure_first_failure", "samples": 100}, "single", 1),
        ({"kind": "erasure_first_failure", "samples": 100}, "monitored", 2),
        ({"kind": counts, "max_size": 1}, "purification", 2),
        ({"kind": counts, "max_size": 2}, "purification", 2),
        ({"kind": counts, "max_size": 3}, "monitored", 1),
        ({"kind": "code_snapshot"}, "single", 4),
    ]


def test_every_probe_costs_at_most_its_estimate():
    n = 16 if numba.config.DISABLE_JIT else 512  # JIT-free kernels time the interpreter
    rows = []
    reference_times = []
    for probe, state, evaluations in _cases(n):
        base_spec, probed_spec = _specs(n, probe, state, evaluations)
        base, probed = api._build_spec(base_spec), api._build_spec(probed_spec)
        estimate = (api._trajectory_us(probed) - api._trajectory_us(base)) * 1e-6 / evaluations
        assert math.isfinite(estimate) and estimate > 0, probe
        if numba.config.DISABLE_JIT:
            continue
        reference_times.append(_median_seconds(_reference_workload, 5))
        _one_trajectory(probed)()  # warm-up (compiles nothing new; fills the spec's caches)
        _one_trajectory(base)()
        runs = 3 if _median_seconds(_one_trajectory(probed), 1) > 0.2 else 7
        with_probe = _median_seconds(_one_trajectory(probed), runs)
        without = _median_seconds(_one_trajectory(base), runs)
        rows.append((probe, state, estimate, (with_probe - without) / evaluations))
    if numba.config.DISABLE_JIT:
        return
    reference_times.append(_median_seconds(_reference_workload, 5))
    speed = REFERENCE_SECONDS / statistics.median(reference_times)  # < 1 on a slower machine
    lines = [
        f"{p['kind']} {p.get('max_size', '')} ({state}): estimate {est * 1e3:.3g} ms, measured "
        f"{meas * 1e3:.3g} ms here = {meas * speed * 1e3:.3g} ms on the fitting machine "
        f"(ratio {est / max(meas * speed, 1e-12):.2f})"
        for p, state, est, meas in rows
    ]
    under = [
        line for line, (_, _, est, meas) in zip(lines, rows, strict=True) if est < meas * speed
    ]
    assert not under, "estimate below the measured cost:\n" + "\n".join(under)


# ---------------------------------------------------------------------------
# graph collections are bounded before any graph is generated
# ---------------------------------------------------------------------------

SPEC = {"num_system": 16, "layers": 2, "measurement_rate": 0.1, "reference": "single"}
RANDOM_FAMILIES = ("watts_strogatz", "newman_watts", "erdos_renyi", "random_regular")


def _no_generation(monkeypatch) -> None:
    def generated(*args, **kwargs):
        raise AssertionError("a graph was generated")

    for name in (*RANDOM_FAMILIES, "complete", "cycle"):
        monkeypatch.setattr(graphs, name, generated)


def _refused_everywhere(spec: dict, message: str, tmp_path) -> None:
    with pytest.raises(ValueError, match=message):
        api.estimate_ensemble(spec)
    with pytest.raises(ValueError, match=message):
        api.run_ensemble(spec, seed=1, max_seconds=60)
    with pytest.raises(ValueError, match=message):
        api.start_ensemble_job(spec, output_dir=str(tmp_path / "job"), seed=1)
    assert not (tmp_path / "job").exists()


def test_a_graph_collection_too_large_is_refused_before_any_graph_is_generated(
    monkeypatch, tmp_path
):
    # A Watts-Strogatz graph has n k / 2 edges of 16 bytes: at n = 4096 and k = 4, 131,072
    # bytes, so 100,000 graphs take 13.1 GB and 2 GB hold 15,258 of them.
    n, realizations = 4096, 100_000
    fit = int(api.MAX_GRAPH_GB * 1e9 // (16 * n * 4 // 2))
    assert fit == 15_258
    _no_generation(monkeypatch)
    bank = {
        "family": "watts_strogatz",
        "k": 4,
        "beta": 0.1,
        "seed": 1,
        "realizations": realizations,
    }
    spec = dict(SPEC, num_system=n, graph=bank)
    message = (
        rf"graph\.realizations = 100,000 graphs on num_system = 4,096 qubits would take about "
        rf"13\.1 GB .* at most {fit:,} realizations fit at num_system = 4,096"
    )
    _refused_everywhere(spec, message, tmp_path)


@pytest.mark.parametrize(
    ("graph", "n", "what"),
    [
        ({"family": "complete"}, 65_536, "edges"),
        ({"family": "erdos_renyi", "p": 1e-4, "seed": 1}, 20_000, "pairs drawn"),
        ({"family": "erdos_renyi", "p": 1e-4, "seed": 1, "realizations": 2}, 20_000, "pairs drawn"),
    ],
)
def test_one_graph_too_large_is_refused_before_it_is_generated(
    monkeypatch, tmp_path, graph, n, what
):
    # Both families hold (or draw) every one of the n (n - 1) / 2 pairs, 16 bytes each.
    largest = max(m for m in range(2, n + 1) if 16 * (m * (m - 1) // 2) <= api.MAX_GRAPH_GB * 1e9)
    _no_generation(monkeypatch)
    message = (
        rf"graph \({graph['family']}\): one graph on num_system = {n:,} qubits takes about .* "
        rf"{what} .* at most num_system = {largest:,} fits"
    )
    _refused_everywhere(dict(SPEC, num_system=n, graph=graph), message, tmp_path)


def test_a_graph_collection_that_fits_runs(monkeypatch):
    # Watts-Strogatz at n = 16, k = 4: 32 edges, 512 bytes; a limit that holds three of them
    monkeypatch.setattr(api, "MAX_GRAPH_GB", 3.5 * 512 / 1e9)
    bank = {"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 3, "realizations": 3}
    out = api.run_ensemble(dict(SPEC, graph=bank), seed=1)
    assert out["num_trajectories"] == 3
    with pytest.raises(ValueError, match="at most 3 realizations fit at num_system = 16"):
        api.estimate_ensemble(dict(SPEC, graph=dict(bank, realizations=4)))


@pytest.mark.parametrize(
    ("family", "params"),
    [
        ("cycle", {}),
        ("path", {}),
        ("complete", {}),
        ("watts_strogatz", {"k": 4, "beta": 0.3}),
        ("watts_strogatz", {"k": 6, "beta": 1.0}),
        ("newman_watts", {"k": 4, "p": 1.0}),
        ("erdos_renyi", {"p": 0.5}),
        ("erdos_renyi", {"p": 0.05}),
        ("random_regular", {"d": 3}),
    ],
)
def test_the_edge_bound_covers_every_generated_graph(family, params):
    for n in (8, 32, 64):
        kept, drawn = api._graph_edges(family, params, n)
        assert kept <= drawn
        seeds = range(5) if family in RANDOM_FAMILIES else [None]
        for seed in seeds:
            extra = {} if seed is None else {"seed": seed}
            edges = getattr(graphs, family)(n, **params, **extra)
            assert len(edges) <= kept, (family, params, n, seed)


def test_absurd_inputs_give_finite_estimates():
    huge = {
        "num_system": 4096,
        "graph": {"family": "cycle"},
        "layers": 10**9,
        "measurement_rate": 0.5,
        "reference": "purification",
        "checkpoints": "final",
        "probes": [
            {"kind": "erasure_lost_profile", "samples": api.MAX_INT},
            {"kind": "contiguous_distance"},
            {"kind": "uncorrectable_region_counts", "max_size": 4000},
        ],
    }
    est = api.estimate_ensemble(huge, 1000)
    assert est["estimated_seconds"] is not None and est["estimated_seconds"] > 1e100
    assert not est["fits_sync_limit"]
    for graph in (
        {"family": "random_regular", "d": 10**6},
        {"family": "erdos_renyi", "p": 1.0},
        {"family": "watts_strogatz", "k": 2**70},
    ):
        assert math.isfinite(api._generation_us(graph, api.MAX_QUBITS))
