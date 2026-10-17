"""The cost of ingesting a unit, without changing a stored value (T3-F02, T3-F09).

* ``observables.bootstrap_peak`` takes the plateau peak of all its replicates at once; every
  replicate's ``T*`` and horizon flag must be exactly those of the replicate-by-replicate loop
  it replaces (``integrated_robustness`` and ``peak`` per replicate, written out here), on
  random traces with every edge case: codes without logical qubits, one ordering (no error),
  ``z = 0``, exact counts, peaks at the horizon.
* A unit's peak memory stays within ``_bytes_per_trajectory`` per trajectory plus a fixed cost
  of the unit (the threads that encode its records); the estimate covers the engine's arrays.
"""

from __future__ import annotations

import gc
import math
import tracemalloc

import numpy as np
import pytest

import _study_store_support as sup
from sparsegf2.study import observables as ob
from sparsegf2.study import runner
from sparsegf2.study.identity import build_graph


def _loop(times, ff, n, *, z, replicates, seed, exact_counts, horizon):
    """The replicate loop of the bootstrap of T*: each replicate's orderings resampled with the
    catalog's draws, D(0) and its error recomputed, the plateau peak taken, one at a time."""
    d = np.asarray(ff, dtype=np.int64)
    e, s = d.shape
    picks = ob._resample_indices(ob._bootstrap_entropy(seed), replicates * e * s, s)
    picks = picks.reshape(replicates, e, s)
    resampled = np.take_along_axis(np.broadcast_to(d, (replicates, e, s)), picks, axis=2)
    t_star, at_horizon = [], []
    for i in range(replicates):
        rob = ob.integrated_robustness(resampled[i], n, exact_counts=exact_counts)
        p = ob.peak(times, rob.value, rob.se, rule="plateau", z=z, horizon=horizon)
        t_star.append(p.t_star)
        at_horizon.append(p.at_horizon)
    return np.array(t_star), np.array(at_horizon)


def _case(rng):
    n = int(rng.integers(5, 40))
    times = np.unique(np.concatenate([[0], rng.integers(1, 200, size=int(rng.integers(1, 13)))]))
    if rng.random() < 0.1:
        times = times[times > 0][:1]  # one positive checkpoint
    e = times.size
    k = np.sort(rng.integers(0, 3, size=e))[::-1]
    if rng.random() < 0.25:
        k[:] = 0  # no code anywhere: every peak missing
    s = 1 if rng.random() < 0.1 else int(rng.integers(2, 30))  # one ordering: no error
    ff = np.empty((e, s), dtype=np.int64)
    for j in range(e):
        if k[j] == 0:
            ff[j] = n + 1
        else:
            ff[j] = rng.integers(int(rng.integers(1, n + 1)), n + 1, size=s)
    counts = None
    if rng.random() < 0.6:
        w = int(rng.integers(1, min(n, 3) + 1))
        counts = np.zeros((e, w))
        for j in range(e):
            for m in range(1, w + 1):
                if k[j] > 0 and ff[j].min() <= m:
                    counts[j, m - 1] = rng.integers(1, math.comb(n, m) + 1)
    z = float(rng.choice([0.0, 0.5, 1.0, 2.0]))
    horizon = int(times[-1]) + int(rng.integers(0, 2))
    return times, ff, n, k, counts, z, horizon


def test_the_bootstrap_of_t_star_equals_the_replicate_loop():
    rng = np.random.default_rng(20261017)
    seen = {"missing": 0, "horizon": 0, "found": 0}
    for _ in range(250):
        times, ff, n, k, counts, z, horizon = _case(rng)
        replicates = int(rng.integers(1, 50))
        seed = [int(x) for x in rng.integers(0, 2**32, size=3)]
        boot = ob.bootstrap_peak(times, ff, n, z=z, replicates=replicates, seed=seed,
                                 exact_counts=counts, k_at=k, horizon=horizon)  # fmt: skip
        t_star, at_horizon = _loop(times, ff, n, z=z, replicates=replicates, seed=seed,
                                   exact_counts=counts, horizon=horizon)  # fmt: skip
        assert np.array_equal(boot.t_star, t_star, equal_nan=True)  # bit for bit
        assert np.array_equal(boot.at_horizon, at_horizon)
        seen["missing"] += int(np.isnan(t_star).sum())
        seen["horizon"] += int(at_horizon.sum())
        seen["found"] += int(np.isfinite(t_star).sum())
    assert min(seen.values()) > 50, seen  # every kind of replicate occurred


def test_the_plateau_of_many_rows_is_the_plateau_of_each_row():
    rng = np.random.default_rng(7)
    times = np.array([0, 1, 2, 4, 8, 16])
    rows = rng.choice([0.0, 1.0, 2.5, 3.0, 3.0, 4.0], size=(400, times.size))
    errors = rng.choice([0.0, 0.1, 0.5, 1.0], size=rows.shape)
    rows[5] = np.nan
    rows[6, 2] = -1.0
    rows[7, 3] = np.inf
    errors[8, 1] = np.nan
    rows[9] = 0.0
    for z in (0.0, 1.0, 2.0):
        t_star, at_horizon = ob._plateau_peaks(times, rows, errors, z, 16)
        for i in range(rows.shape[0]):
            p = ob.peak(times, rows[i], errors[i], rule="plateau", z=z, horizon=16)
            assert t_star[i] == p.t_star or (math.isnan(t_star[i]) and math.isnan(p.t_star))
            assert at_horizon[i] == p.at_horizon, (i, z)


def _unit_store(tmp_path, name, n=40, graphs=16):
    m = sup.manifest(
        experiment_id="ingest",
        conditions=[{"n": [n], "beta": ["0.25"], "p": ["0.1"]}],
        depth={"layers": {"fixed": 60}, "checkpoints": {"count": 8}},
        trials={"graphs": graphs, "circuits_per_graph": 2, "circuit_index_stride": 2},
        observables={**sup.BASE["observables"], "uniform_erasure": {"samples": 40,
                                                                     "exact_budget": n}},
        storage={"graphs_per_unit": graphs},
        analysis={"peak": {"bootstrap": 20}},
    )  # fmt: skip
    return sup.build(tmp_path / name, m, run=False)


def test_a_unit_stays_within_its_memory_estimate(tmp_path):
    n = 128  # snapshots dominate a unit's memory here
    with _unit_store(tmp_path, "warm", n, 32) as warm:
        runner.run(warm, num_threads=2)  # compile and cache before measuring
    with _unit_store(tmp_path, "s", n, 32) as store:
        units = runner.plan(store)
        assert len(units) == 1 and units[0].trajectories == 64
        req = runner.requirements(store.revisions(), "F", n, "0.25")
        per = runner._bytes_per_trajectory(req)
        # the engine's own arrays of one trajectory are part of the estimate
        manifest = store.manifest
        graph = build_graph(n, "0.25", 0, manifest.seeds.master_graph_seed)
        spec = runner._unit_spec(manifest, "F", n, units[0].beta, units[0].p, [graph.edges],
                                 runner.engine_probes(req))  # fmt: skip
        result = spec.run(indices=[0], seed=list(manifest.run_seed(n, "0.25", "0.1")))
        engine = sum(int(w.nbytes) + int(c.nbytes) for w, c in result.snapshots.values())
        engine += sum(int(np.asarray(v).nbytes) for v in result.values.values())
        assert engine <= per
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        runner.run(store, units=units, num_threads=2)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        # the budget's share plus the unit's fixed cost (the pool of two threads, the records'
        # objects); the measured fixed cost is about 0.4 MiB per thread
        assert peak <= per * units[0].trajectories + 4 * 2**20, (peak, per)
        # the engine's snapshot rows alone (the former estimate) do not bound it
        assert peak > engine * units[0].trajectories + 4 * 2**20, (peak, engine)
        assert store.check() == []


@pytest.mark.parametrize("threads", [1, 3])
def test_the_pool_of_threads_changes_no_record(tmp_path, threads):
    with _unit_store(tmp_path, "a") as a, _unit_store(tmp_path, "b") as b:
        runner.run(a, num_threads=threads)
        runner.run(b, num_threads=4)
        assert sup.records(a) == sup.records(b)
        assert a.logical_digest() == b.logical_digest()
