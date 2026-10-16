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

import functools
import math
import re
import statistics
import time
import tracemalloc

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
    correlated = {
        "kind": "correlated_erasure",
        "couplings": [0, 1],
        "sizes": [n // 16, n // 2],
        "samples": 20,
    }
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
        ({"kind": "erasure_lost_profile", "samples": 10}, "single", 1),
        ({"kind": "erasure_first_failure", "samples": 100}, "single", 1),
        ({"kind": "erasure_first_failure", "samples": 100}, "monitored", 2),
        ({"kind": counts, "max_size": 1}, "purification", 2),
        ({"kind": counts, "max_size": 2}, "purification", 2),
        ({"kind": counts, "max_size": 3}, "monitored", 1),
        ({"kind": "code_snapshot"}, "single", 4),
        # the swap chain on the Watts-Strogatz graph, and the exact ring sampler
        ({**correlated, "burn_in": 100}, "monitored", 1),
        (correlated, "purification", 1),
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


def _every_thread(mc, threads: int):
    return lambda: mc.run(threads, seed=7, num_threads=threads)


def test_one_trajectory_per_thread_costs_at_most_its_estimate():
    """One trajectory per thread on every thread: ``estimate_ensemble``'s kernel term against
    the wall time, for the probes with the smallest all-threads margins in
    benchmarks/results/probe_cost.md (entropy_profile and counts at w = 2, n = 512).

    A busy thread runs slower than a thread alone (all-core clocks, SMT siblings, shared
    memory bandwidth); ``api._parallel_threads`` prices each of T busy threads at
    1 / (1 + 0.04 (T - 1)) of a lone one, and that model is what this checks. The measured
    time is normalised with the one-thread reference workload, as above. At 32 threads on
    the fitting machine, with other processes running, the estimate was 4.0 and 3.8 times
    the normalised wall time. A runner with 2 to 4 threads cannot make this flaky: the model
    gives a busy thread 0.89 or more of a lone one there, so even SMT siblings at half speed
    leave these probes' one-thread margin (3.5 or more) at 1.75 or more.
    """
    threads = numba.get_num_threads()
    n = 16 if numba.config.DISABLE_JIT else 512  # JIT-free kernels time the interpreter
    rows = []
    reference_times = []
    counts = {"kind": "uncorrectable_region_counts", "max_size": 2}
    for probe, evaluations in (({"kind": "entropy_profile"}, 4), (counts, 2)):
        base_spec, probed_spec = _specs(n, probe, "purification", evaluations)
        base, probed = api._build_spec(base_spec), api._build_spec(probed_spec)
        # the kernel term of estimate_ensemble (its overhead and setup cancel)
        estimate = api._estimate_seconds({}, probed, threads) - api._estimate_seconds(
            {}, base, threads
        )
        assert math.isfinite(estimate) and estimate > 0, probe
        if numba.config.DISABLE_JIT:
            continue
        reference_times.append(_median_seconds(_reference_workload, 5))
        _every_thread(probed, threads)()  # warm-up: compiled, and every thread started
        _every_thread(base, threads)()
        with_probe = _median_seconds(_every_thread(probed, threads), 5)
        without = _median_seconds(_every_thread(base, threads), 5)
        rows.append((probe, estimate, with_probe - without))
    if numba.config.DISABLE_JIT:
        return
    reference_times.append(_median_seconds(_reference_workload, 5))
    speed = REFERENCE_SECONDS / statistics.median(reference_times)
    lines = [
        f"{p['kind']} on {threads} threads: estimate {est * 1e3:.3g} ms, wall {wall * 1e3:.3g} "
        f"ms here = {wall * speed * 1e3:.3g} ms on the fitting machine "
        f"(ratio {est / max(wall * speed, 1e-12):.2f})"
        for p, est, wall in rows
    ]
    under = [line for line, (_, est, wall) in zip(lines, rows, strict=True) if est < wall * speed]
    assert not under, "estimate below the measured wall time:\n" + "\n".join(under)


# ---------------------------------------------------------------------------
# graph collections are bounded before any graph is generated
# ---------------------------------------------------------------------------

SPEC = {"num_system": 16, "layers": 2, "measurement_rate": 0.1, "reference": "single"}
RANDOM_FAMILIES = ("watts_strogatz", "newman_watts", "erdos_renyi", "random_regular")


def _no_generation(monkeypatch) -> None:
    """Every family raises when called; each keeps its signature, which the API reads."""
    for name in (*RANDOM_FAMILIES, "complete", "cycle"):

        @functools.wraps(getattr(graphs, name))
        def generated(*args, **kwargs):
            raise AssertionError("a graph was generated")

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
    ("graph", "n"),
    [
        ({"family": "complete"}, 65_536),
        ({"family": "erdos_renyi", "p": 1e-4, "seed": 1}, 20_000),
        ({"family": "erdos_renyi", "p": 1e-4, "seed": 1, "realizations": 2}, 20_000),
        # 72 million pairs: 1.2 GB at 16 bytes each, but generating them allocates 3.5 GB
        ({"family": "erdos_renyi", "p": 1e-4, "seed": 1}, 12_000),
    ],
)
def test_one_graph_too_large_is_refused_before_it_is_generated(monkeypatch, tmp_path, graph, n):
    # Both families build every one of the n (n - 1) / 2 pairs: triu_indices, the stacked
    # pairs and their int64 copy, 48 bytes per pair at the peak (measured with tracemalloc,
    # see test_the_generation_bound_covers_what_generation_allocates), plus 64 KiB.
    limit = api.MAX_GRAPH_GB * 1e9
    largest = max(m for m in range(2, n + 1) if 48 * (m * (m - 1) // 2) + 65_536 <= limit)
    assert largest == 9129
    _no_generation(monkeypatch)
    message = (
        rf"graph \({graph['family']}\): one graph on num_system = {n:,} qubits takes about "
        rf".* GB while it is generated \(.* pairs drawn at about 48 bytes each; .*\): at most "
        rf"num_system = {largest:,} fits"
    )
    _refused_everywhere(dict(SPEC, num_system=n, graph=graph), message, tmp_path)


@pytest.mark.parametrize(
    ("graph", "accepted"),
    [
        # a degree so large that no num_system the family accepts (k < n) fits
        ({"family": "watts_strogatz", "k": 20_000, "beta": 0.1, "seed": 1}, None),
        ({"family": "newman_watts", "k": 20_000, "p": 0.1, "seed": 1}, None),
        ({"family": "random_regular", "d": 30_001, "seed": 1}, None),
        # one that fits at a smaller n, which must be one the family accepts
        ({"family": "watts_strogatz", "k": 1000, "beta": 0.1, "seed": 1}, lambda m: m > 1000),
        # an odd degree needs an even n (n * d even)
        ({"family": "random_regular", "d": 1001, "seed": 1}, lambda m: m > 1001 and m % 2 == 0),
    ],
)
def test_the_suggested_num_system_is_one_the_family_accepts(monkeypatch, graph, accepted):
    _no_generation(monkeypatch)
    spec = dict(SPEC, num_system=65_536, graph=graph)
    with pytest.raises(ValueError, match="takes about") as info:
        api.estimate_ensemble(spec, 1)
    found = re.search(r"at most num_system = ([\d,]+) fits", str(info.value))
    if accepted is None:
        assert found is None and "no num_system fits these parameters" in str(info.value)
        return
    assert found is not None, str(info.value)
    m = int(found.group(1).replace(",", ""))
    assert accepted(m), m
    # there the bound passes, so generation starts (and the patched family raises)
    with pytest.raises(AssertionError, match="a graph was generated"):
        api.estimate_ensemble(dict(spec, num_system=m), 1)


def test_a_graph_collection_that_fits_runs(monkeypatch):
    # Watts-Strogatz at n = 16, k = 4: 32 edges, 512 bytes kept per graph; a limit that holds
    # 200 of them (and generating one graph, 512 bytes per edge plus 64 KiB, fits too)
    monkeypatch.setattr(api, "MAX_GRAPH_GB", 200.5 * 512 / 1e9)
    bank = {"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 3, "realizations": 200}
    out = api.run_ensemble(dict(SPEC, graph=bank), seed=1)
    assert out["num_trajectories"] == 200
    with pytest.raises(ValueError, match="at most 200 realizations fit at num_system = 16"):
        api.estimate_ensemble(dict(SPEC, graph=dict(bank, realizations=201)))


GRAPH_CASES = [
    ("cycle", {}),
    ("path", {}),
    ("complete", {}),
    ("lattice_2d", {"periodic": True}),
    ("lattice_2d", {"periodic": False}),
    ("watts_strogatz", {"k": 4, "beta": 0.3}),
    ("watts_strogatz", {"k": 6, "beta": 1.0}),
    ("newman_watts", {"k": 4, "p": 1.0}),
    ("erdos_renyi", {"p": 0.5}),
    ("erdos_renyi", {"p": 0.05}),
    ("random_regular", {"d": 3}),
]


def _generate(family: str, params: dict, n: int, seed: int | None):
    if family == "lattice_2d":  # four rows: rows * cols = n
        return graphs.lattice_2d(4, n // 4, **params)
    extra = {"seed": seed} if family in RANDOM_FAMILIES and seed is not None else {}
    return getattr(graphs, family)(n, **params, **extra)


@pytest.mark.parametrize(("family", "params"), GRAPH_CASES)
def test_the_edge_bound_covers_every_generated_graph(family, params):
    for n in (8, 32, 64):
        bound = {"rows": 4, "cols": n // 4, **params} if family == "lattice_2d" else params
        kept, drawn = api._graph_edges(family, bound, n)
        assert kept <= drawn
        for seed in range(5) if family in RANDOM_FAMILIES else [None]:
            edges = _generate(family, params, n, seed)
            assert len(edges) <= kept, (family, params, n, seed)


# ---------------------------------------------------------------------------
# the estimate says where it can be too low
# ---------------------------------------------------------------------------

SCAN_SPEC = {
    "num_system": 256,
    "graph": {"family": "cycle"},
    "layers": 10,
    "measurement_rate": 0.1,
    "reference": "purification",
    "checkpoints": 5,
}


@pytest.mark.parametrize(
    "probes",
    [
        [{"kind": "erasure_first_failure"}],
        [{"kind": "contiguous_distance"}],
        [{"kind": "contiguous_distance"}, {"kind": "erasure_first_failure", "checkpoints": 10}],
    ],
)
def test_purification_scans_carry_their_worst_case_in_a_note(probes):
    spec = dict(SCAN_SPEC, probes=[{"kind": "code_dimension"}, *probes])
    notes = api.estimate_ensemble(spec, 10)["notes"]
    assert len(notes) == 1
    kinds = sorted(p["kind"] for p in probes)
    assert notes[0].startswith(" and ".join(kinds))
    assert "near n / 2" in notes[0]
    shown = re.search(r"up to about (\S+) times", notes[0]).group(1)
    assert float(shown) > 1  # the single-reference price is the higher one
    # one sentence, with the factor rounded: to a whole number from 10 on, else to 0.1
    assert ". " not in notes[0] and "\n" not in notes[0]
    assert re.fullmatch(r"\d+" if float(shown) >= 10 else r"\d\.\d", shown), shown


@pytest.mark.parametrize(
    "spec",
    [
        dict(SCAN_SPEC, reference="single", probes=[{"kind": "erasure_first_failure"}]),
        dict(SCAN_SPEC, reference="single", probes=[{"kind": "contiguous_distance"}]),
        dict(SCAN_SPEC, probes=[{"kind": "code_dimension"}, {"kind": "entropy_profile"}]),
        dict(SCAN_SPEC, probes=[{"kind": "uncorrectable_region_counts", "max_size": 2}]),
    ],
)
def test_other_specs_have_no_note(spec):
    assert "notes" not in api.estimate_ensemble(spec, 10)


@pytest.mark.parametrize(("family", "params"), GRAPH_CASES)
def test_the_generation_bound_covers_what_generation_allocates(family, params):
    # tracemalloc sees numpy's arrays and the interpreter's objects alike
    _generate(family, params, 64, 1)  # anything imported or compiled on first use
    for n in (8, 64, 512):
        bound = {"rows": 4, "cols": n // 4, **params} if family == "lattice_2d" else params
        tracemalloc.start()
        try:
            _generate(family, params, n, 1)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak <= api._generation_bytes(family, bound, n), (family, params, n, peak)


def test_random_regular_is_priced_at_its_own_max_tries():
    # d = 7 needs about exp(12) = 162,755 pairings on average, above the default cap of 1000
    default = api._generation_us({"family": "random_regular", "d": 7}, 64)
    allowed = api._generation_us({"family": "random_regular", "d": 7, "max_tries": 10**6}, 64)
    assert allowed > 100 * default
    for cap in (10**400, math.inf):  # absurd caps stay finite
        assert math.isfinite(
            api._generation_us({"family": "random_regular", "d": 60_000, "max_tries": cap}, 65_536)
        )


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


# ---------------------------------------------------------------------------
# thread counts and the layout of the saved result
# ---------------------------------------------------------------------------

LAYOUT_SPEC = {
    "num_system": 8,
    "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 1, "realizations": 3},
    "circuits_per_graph": 2,
    "layers": 12,
    "measurement_rate": 0.2,
    "schedule": "random_edges",
    "edge_weights": {"family": "power_law", "alpha": 2.0},
    "reference": "single",
    "checkpoints": 4,
    "probes": [
        {"kind": "code_dimension"},
        {"kind": "entropy_profile", "checkpoints": "final"},
        {"kind": "erasure_first_failure", "samples": 5},
        {"kind": "code_snapshot", "checkpoints": "final"},
    ],
}


def test_the_estimate_keeps_its_keys_and_adds_a_layout():
    est = api.estimate_ensemble(LAYOUT_SPEC, 6)
    assert {
        "estimated_seconds",
        "trajectories",
        "result_megabytes",
        "threads",
        "fits_sync_limit",
        "sync_limit_seconds",
        "layout",
    } <= set(est)
    assert "by_threads" not in est
    assert est["layout"]["files"] == ["result.npz", "spec.json", "provenance.json"]
    assert "benchmark_ensemble" in est["layout"]["note"]


def test_the_estimate_for_given_thread_counts():
    spec = {
        "num_system": 64,
        "graph": {"family": "cycle"},
        "layers": 200,
        "measurement_rate": 0.1,
        "reference": "single",
        "checkpoints": "final",
        "probes": [{"kind": "code_dimension"}],
    }
    est = api.estimate_ensemble(spec, 100, threads=[32, 1, 4])
    assert [row["threads"] for row in est["by_threads"]] == [32, 1, 4]
    seconds = {row["threads"]: row["estimated_seconds"] for row in est["by_threads"]}
    assert seconds[1] > seconds[4] > seconds[32] > 0
    # the plain estimate is the one at the current thread count
    here = api.estimate_ensemble(spec, 100, threads=[numba.get_num_threads()])
    assert here["by_threads"][0]["estimated_seconds"] == est["estimated_seconds"]
    # trajectories are the unit of parallelism: threads beyond them are idle
    few = api.estimate_ensemble(spec, 2, threads=[2, 64])["by_threads"]
    assert few[0]["estimated_seconds"] == few[1]["estimated_seconds"]


@pytest.mark.parametrize("threads", [0, -1, "4", 4, [], [0], [1, 0], [True], [1.5], ["2"]])
def test_the_estimate_refuses_bad_thread_counts(threads):
    with pytest.raises(ValueError, match="threads"):
        api.estimate_ensemble(LAYOUT_SPEC, 6, threads=threads)


@pytest.mark.parametrize(
    "spec",
    [
        LAYOUT_SPEC,
        {
            "num_system": 6,
            "graph": {"family": "cycle"},
            "layers": 6,
            "measurement_rate": 0.1,
            "reference": "none",
            "checkpoints": "final",
            "probes": [{"kind": "half_cut_entropy"}],
        },
        {
            "num_system": 70,  # two words per row
            "graph": {"family": "cycle"},
            "layers": 4,
            "measurement_rate": 0.1,
            "reference": "purification",
            "checkpoints": 2,
            "probes": [
                {"kind": "code_dimension"},
                {"kind": "code_snapshot", "checkpoints": "final"},
            ],
        },
    ],
)
def test_the_layout_matches_the_saved_file(spec, tmp_path):
    trajectories = 6
    layout = api.estimate_ensemble(spec, trajectories)["layout"]
    api.run_ensemble(spec, trajectories, seed=1, output_dir=str(tmp_path / "run"))
    with np.load(tmp_path / "run" / "result.npz") as z:
        saved = {name: z[name] for name in z.files if name != "__meta__"}
    stated = {row["name"]: row for row in layout["arrays"]}
    assert set(stated) == set(saved)
    for name, array in saved.items():
        assert stated[name]["shape"] == list(array.shape), name
        assert stated[name]["dtype"] == str(array.dtype), name
        assert stated[name]["megabytes"] == pytest.approx(array.nbytes / 1e6, rel=1e-5, abs=1e-6)
    has_codes = any(p["kind"] == "code_snapshot" for p in spec["probes"])
    assert ("upper bound" in layout["note"]) == has_codes


def test_the_layout_values_fit_the_memory_figure():
    est = api.estimate_ensemble(LAYOUT_SPEC, 6)
    values = sum(r["megabytes"] for r in est["layout"]["arrays"] if r["name"].startswith("value::"))
    assert 0 < values <= est["result_megabytes"]
