"""The in-run probe ``correlated_erasure``: P_{C,G,J}(m) per trajectory and checkpoint.

At each evaluation, for each size m in ``sizes``, the probe draws ``samples`` regions of m
system qubits at each coupling J from Q_J(A | m) = exp(-J b_G(A)) / Z_m(J) (b_G: the edges
of the erasure graph G leaving A) with the graph-side sampler (``erasures.sample_regions``:
the same kernels, ladder and draw order) from a stream of its own, and records the
fraction of them whose erasure loses logical information. Checked here:

1. The values are exactly those fractions: the regions recomputed with the public
   graph-side sampler from the documented per-size stream (``stream_state(key_probe,
   mix(mix(t, name hash, layer), m, 0))``), evaluated on the saved code; on the ring
   (exact sampler), on a Watts-Strogatz graph (swap chain with the automatic ladder) and
   with ``geometry="cycle"`` and ``"path"`` over the qubit labels.
2. A plain-Python transcription of the draws (``_reference_rng``, T1's transcriptions of
   the chain and the exact samplers, the GF(2) oracle on the saved code) gives the same
   values, so the documented protocol is the whole story.
3. Statistics: the values estimate the exact P_J(m) (boundary-resolved counts on the saved
   code) within errors: binomial tails for the exact ring sampler, the effective sample
   size of the very regions used for the chain; at J = 0 the exact uniform P_C(m)
   (``uncorrectable_counts``); and they agree with ``StabilizerCode.correlated_failure``.
4. k = 0 gives 0; purified trajectories (``stop_when_purified``) record 0.
5. Reproducibility: the same seed, replay (also with narrower ``sizes`` or alone), chunks,
   thread counts; a second correlated probe does not change the first.
6. Storage, ``EnsembleResult.correlated``, save and load, protocol comparison (a resume
   with other couplings is refused), the JSON interface (summaries, conditioning on a
   code, truncation, capabilities) and validation messages.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numba
import numpy as np
import pytest

import _gf2_oracle as gf2
import _protocol_oracle as oracle
import _reference_rng as ref
from _erasure_oracle import _reference_chain, _reference_exact
from sparsegf2 import erasures, graphs, mipt
from sparsegf2._kernel import rng as krng

JIT_FREE = bool(numba.config.DISABLE_JIT)


def _size(compiled, plain):
    """JIT-free (plain-Python kernels, several hundred times slower) runs use less work."""
    return plain if JIT_FREE else compiled


N = 16
COUPLINGS = (1.5, 0.0, -0.5)  # unsorted on purpose: the layout follows the given order
SIZES = (2, 5, 8, 11, 14)


def _probe(**kw) -> mipt.Probe:
    base = dict(couplings=COUPLINGS, sizes=SIZES, samples=_size(40, 12))
    base.update(kw)
    return mipt.correlated_erasure(**base)


def _ring_spec(probe=None, n=N, **kw) -> mipt.MonitoredCircuit:
    """Brickwork on the ring (exact samples): codes with a few logical qubits and distance 2+."""
    base = dict(
        num_system=n,
        edges=graphs.cycle(n),
        layers=48,
        measurement_rate=0.15,
        schedule="matching",
        reference="purification",
        checkpoints=[24, 48],
        probes=[mipt.code_dimension(), probe or _probe(), mipt.code_snapshot()],
    )
    base.update(kw)
    return mipt.MonitoredCircuit(**base)


WS_EDGES = graphs.watts_strogatz(N, 4, 0.3, seed=3)


def _ws_spec(probe=None, n=N, **kw) -> mipt.MonitoredCircuit:
    """A Watts-Strogatz interaction graph: the swap chain with the automatic ladder."""
    base = dict(
        num_system=n,
        edges=WS_EDGES if n == N else graphs.watts_strogatz(n, 4, 0.3, seed=3),
        layers=40,
        measurement_rate=0.15,
        schedule="random_edges",
        reference="purification",
        checkpoints=[20, 40],
        probes=[mipt.code_dimension(), probe or _probe(burn_in=30, thin=2), mipt.code_snapshot()],
    )
    base.update(kw)
    return mipt.MonitoredCircuit(**base)


def _size_state(result: mipt.EnsembleResult, trajectory: int, name: str, layer: int, m: int):
    """The xoshiro state the probe draws size ``m`` from (the documented derivation)."""
    _, key = mipt._keys(result.seed_sequence)
    index = krng.mix(np.int64(trajectory), np.uint64(mipt._name_hash(name)), np.int64(layer))
    out = np.zeros(4, dtype=np.uint64)
    krng.stream_state(key, np.uint64(krng.mix(np.uint64(index), np.int64(m), np.int64(0))), out)
    return out


def _erasure_graph(spec: mipt.MonitoredCircuit, probe: mipt.Probe, trajectory: int):
    if probe.geometry == "cycle":
        return graphs.cycle(spec.num_system)
    if probe.geometry == "path":
        return graphs.path(spec.num_system)
    return spec.graphs[0 if spec.num_graphs == 1 else trajectory // spec.circuits_per_graph]


def _graph_side(monkeypatch, state, edges, n, m, probe: mipt.Probe) -> erasures.RegionSample:
    """``erasures.sample_regions`` with the probe's settings, drawing from ``state``."""
    monkeypatch.setattr(erasures, "seed_state", lambda seed: state.copy())
    return erasures.sample_regions(
        edges, n, m, list(probe.couplings), probe.samples, burn_in=probe.burn_in,
        thin=probe.thin, tempering=probe.tempering, method=probe.method,
        ladder=probe.ladder if isinstance(probe.ladder, str) else list(probe.ladder),
        max_step=probe.max_step, seed=0,
    )  # fmt: skip


def _check_fractions(monkeypatch, spec, result, name="correlated_erasure") -> dict:
    """Every recorded value equals the failing fraction of the graph-side regions."""
    probe = next(p for p in spec.probes if p.name == name)
    couplings, sizes, values = result.correlated(name)
    assert couplings.tolist() == list(probe.couplings) and sizes.tolist() == list(probe.sizes)
    k = result.values["code_dimension"]
    seen = {"codes": 0, "partial": 0, "methods": set()}
    for row, t in enumerate(result.trajectories.tolist()):
        for e, layer in enumerate(result.layers(name).tolist()):
            if k[row, e] == 0:
                assert not values[row, e].any()
                continue
            code = result.code(row, e)
            assert code.num_logical == k[row, e]
            seen["codes"] += 1
            for i, m in enumerate(sizes.tolist()):
                state = _size_state(result, t, name, layer, m)
                edges = _erasure_graph(spec, probe, t)
                drawn = _graph_side(monkeypatch, state, edges, spec.num_system, m, probe)
                seen["methods"].add(drawn.method)
                fails = code.region_failures(drawn.regions.reshape(-1, m))
                fraction = fails.reshape(len(couplings), probe.samples).mean(axis=1)
                np.testing.assert_array_equal(values[row, e, :, i], fraction, err_msg=f"{t} {m}")
                seen["partial"] += int(np.count_nonzero((fraction > 0) & (fraction < 1)))
    return seen


# ---------------------------------------------------------------------------------------
# 1. the values are the failing fractions of the graph-side regions
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", ["ring", "ws_chain", "single_reference", "cycle_geometry", "path_geometry"]
)
def test_values_are_the_failing_fractions_of_the_graph_side_regions(case, monkeypatch):
    if case == "ring":
        spec = _ring_spec()
        method = "exact"
    elif case == "ws_chain":
        spec = _ws_spec()
        method = "chain"
    elif case == "single_reference":  # one reference qubit with a random partner: k in {0, 1}
        spec = _ws_spec(reference="single", measurement_rate=0.04, checkpoints=[10, 20, 40])
        method = "chain"
    else:
        geometry = case.split("_")[0]
        spec = _ws_spec(_probe(geometry=geometry))
        method = "exact"
    result = spec.run(_size(5, 2), seed=17)
    seen = _check_fractions(monkeypatch, spec, result)
    # not vacuous: codes were probed, some fractions are strictly between 0 and 1, and the
    # sampler is the one the geometry calls for
    assert seen["codes"] >= 2 and seen["partial"] >= 3
    assert seen["methods"] == {method}


def test_chain_ladders_and_explicit_settings_follow_the_graph_side_tool(monkeypatch):
    """Other ladders and settings than the defaults: a given ladder, extra rungs, a
    max_step, no tempering, the chain forced on the ring."""
    probes = [
        _probe(name="given", ladder="given", burn_in=5, thin=3),
        _probe(name="extra", ladder=[0.4, 0.9], burn_in=5),
        _probe(name="step", max_step=0.3, burn_in=5, sizes=[3, 9]),
        _probe(name="lone", tempering=False, burn_in=5, sizes=[4]),
        _probe(name="forced", method="chain", geometry="cycle", burn_in=5, sizes=[6, 10]),
    ]
    spec = _ws_spec(probes=[mipt.code_dimension(), *probes, mipt.code_snapshot()])
    result = spec.run(_size(3, 1), seed=5)
    for p in probes:
        seen = _check_fractions(monkeypatch, spec, result, p.name)
        assert seen["codes"] >= 1 and seen["methods"] == {"chain"}


# ---------------------------------------------------------------------------------------
# 2. the protocol oracle: the draws transcribed in plain Python
# ---------------------------------------------------------------------------------------


def _oracle_values(spec, probe, seed, trajectory, layer, generators) -> np.ndarray:
    """``(c, s)`` failing fractions rebuilt without the package's kernels or samplers."""
    n = spec.num_system
    edges = np.asarray(_erasure_graph(spec, probe, trajectory))
    code = gf2.Code(list(generators), n)
    given = list(probe.couplings)
    order = sorted(range(len(given)), key=lambda i: given[i])
    requested = [given[i] for i in order]
    index = ref.mix(trajectory, oracle.probe_name_hash(probe.name), layer)
    out = np.zeros((len(given), len(probe.sizes)))
    for i, m in enumerate(probe.sizes):
        state = ref.stream_state(oracle.probe_key(seed), ref.mix(index, m, 0))
        if probe.geometry in ("cycle", "path") and probe.method != "chain":
            regions, _, _ = _reference_exact(
                probe.geometry, n, m, requested, probe.samples, list(range(n)), state
            )
        else:  # the automatic ladder of sample_regions: 0 and rungs no further apart than
            # 1 / sqrt(Var_0 b)
            var0 = erasures.uniform_boundary_moments(edges, n, m)[1]
            rungs = erasures.coupling_ladder(requested, 1 / math.sqrt(var0)).tolist()
            record = [requested.index(j) if j in requested else -1 for j in rungs]
            regions, _, _, _ = _reference_chain(
                edges.tolist(), n, m, rungs, record, probe.samples, probe.burn_in, probe.thin,
                True, state,
            )  # fmt: skip
        for q, g in enumerate(order):
            out[g, i] = np.mean([code.lost(list(r)) > 0 for r in regions[q].tolist()])
    return out


@pytest.mark.parametrize("geometry", ["interaction", "cycle", "path"])
def test_protocol_oracle_rebuilds_the_values_in_plain_python(geometry):
    n = 8
    probe = mipt.correlated_erasure(
        [0.9, -0.4, 0.0], sizes=[2, 5, 3], samples=5, burn_in=3, thin=2, geometry=geometry
    )
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.watts_strogatz(n, 4, 0.3, seed=1),
        layers=10,
        measurement_rate=0.08,
        schedule="random_edges",
        reference="purification",
        checkpoints=[5, 10],
        probes=[mipt.code_dimension(), probe, mipt.code_snapshot()],
    )
    seed = 31
    result = spec.run(3, seed=seed)
    values = result.correlated(probe.name)[2]
    checked = 0
    for t in range(3):
        for e, layer in enumerate(result.layers(probe.name).tolist()):
            if result.values["code_dimension"][t, e] == 0:
                continue
            generators = result.code(t, e).stabilizer_matrix()
            expected = _oracle_values(spec, probe, seed, t, layer, generators)
            np.testing.assert_array_equal(values[t, e], expected, err_msg=f"t={t} layer={layer}")
            checked += 1
    assert checked >= 3 and 0 < values.mean() < 1


# ---------------------------------------------------------------------------------------
# 3. statistics: the exact P_J(m), the uniform P_C(m), correlated_failure
# ---------------------------------------------------------------------------------------


def _two_sided_binomial(count: int, trials: int, p: float) -> float:
    """The two-sided tail probability of ``count`` successes in ``trials`` at rate ``p``."""
    pmf = [math.comb(trials, x) * p**x * (1 - p) ** (trials - x) for x in range(trials + 1)]
    low, high = sum(pmf[: count + 1]), sum(pmf[count:])
    return min(1.0, 2 * min(low, high))


def test_exact_ring_samples_estimate_the_exact_correlated_failure():
    n = _size(N, 10)  # plain Python enumerates 2^n subsets per code below
    couplings = [0.0, 1.0, 3.0, -1.0]
    samples = _size(300, 60)
    probe = mipt.correlated_erasure(couplings, samples=samples)  # every size 1..n-1
    spec = _ring_spec(probe, n, checkpoints=[48])
    result = spec.run(_size(6, 3), seed=8)
    _, sizes, values = result.correlated(probe.name)
    assert sizes.tolist() == list(range(1, n))
    ring = graphs.cycle(n)
    tails, compared = [], 0
    for t in range(result.num_trajectories):
        k = result.values["code_dimension"][t, 0]
        if k == 0:
            assert not values[t, 0].any()
            continue
        code = result.code(t, 0)
        exact = code.boundary_failure_counts(ring, n - 1).exact(couplings)
        # J = 0 is the uniform P_C(m), from the exhaustive counts
        uniform = code.uncorrectable_counts(n - 1) / [math.comb(n, m) for m in sizes]
        np.testing.assert_allclose(exact[0], uniform, rtol=1e-12)
        for c in range(len(couplings)):
            for i, m in enumerate(sizes.tolist()):
                p = exact[c, i] if c else uniform[i]
                count = round(values[t, 0, c, i] * samples)
                if p in (0.0, 1.0):
                    assert count == p * samples, (t, couplings[c], m)
                    continue
                tails.append(_two_sided_binomial(count, samples, p))
                compared += 1
    assert compared >= _size(40, 10)
    # each of these independent binomial tests at 1e-7: a false alarm in the whole test has
    # probability below 1e-4
    assert min(tails) > 1e-7, min(tails)
    # and not vacuous: J moves the answer
    assert np.abs(values[:, 0, 2] - values[:, 0, 0]).max() > 0.1


def test_chain_estimates_the_exact_correlated_failure_within_its_effective_errors(monkeypatch):
    n = _size(N, 10)  # plain Python enumerates 2^n subsets per code below
    couplings = [0.0, 1.0, 2.5]
    samples = _size(1000, 80)
    sizes = [2, 4, 7, 10, 13] if n == N else [2, 5, 8]
    probe = mipt.correlated_erasure(couplings, sizes=sizes, samples=samples, thin=2)
    spec = _ws_spec(probe, n, checkpoints=[40])
    edges = spec.graphs[0]
    result = spec.run(_size(4, 2), seed=12)
    _, sizes, values = result.correlated(probe.name)
    worst, compared = 0.0, 0
    for t in range(result.num_trajectories):
        if result.values["code_dimension"][t, 0] == 0:
            continue
        code = result.code(t, 0)
        exact = code.boundary_failure_counts(edges, n - 1).exact(couplings)
        for i, m in enumerate(sizes.tolist()):
            state = _size_state(result, t, probe.name, 40, m)
            drawn = _graph_side(monkeypatch, state, edges, n, m, result.spec.probes[1])
            fails = code.region_failures(drawn.regions.reshape(-1, m)).reshape(3, samples)
            for c in range(3):
                p = exact[c, m - 1]
                assert values[t, 0, c, i] == fails[c].mean()
                if p in (0.0, 1.0):
                    assert values[t, 0, c, i] == p
                    continue
                series = fails[c].astype(float)
                ess = erasures.chain_diagnostics(series)["ess"] if series.std() else samples
                se = math.sqrt(p * (1 - p) / ess)
                gap = abs(values[t, 0, c, i] - p)
                # 5 standard errors, plus 3 / ESS for the discreteness at tiny p (T2's rule)
                assert gap <= 5 * se + 3 / ess, (t, couplings[c], m, values[t, 0, c, i], p, ess)
                if ess * p * (1 - p) >= 10:
                    worst = max(worst, gap / se)
                    compared += 1
    assert compared >= _size(15, 0)
    assert worst < 5


def test_values_agree_with_correlated_failure_on_the_saved_codes():
    n = _size(N, 10)  # plain Python enumerates 2^n subsets per code below
    couplings = [-0.5, 2.0]
    samples = _size(400, 60)
    probe = mipt.correlated_erasure(couplings, sizes=[3, 6, 9] if n == 10 else [3, 6, 9, 12],
                                    samples=samples)  # fmt: skip
    result = _ring_spec(probe, n, checkpoints=[48]).run(_size(4, 3), seed=21)
    _, sizes, values = result.correlated(probe.name)
    ring = graphs.cycle(n)
    compared = 0
    for t in range(result.num_trajectories):
        if result.values["code_dimension"][t, 0] == 0:
            continue
        code = result.code(t, 0)
        other = code.correlated_failure(ring, couplings, sizes, 4 * samples, seed=t)
        exact = code.boundary_failure_counts(ring, n - 1).exact(couplings)[:, sizes - 1]
        p = np.clip(exact, 1e-12, 1 - 1e-12)
        se = np.sqrt(p * (1 - p) * (1 / samples + 1 / (4 * samples)))
        assert np.all(np.abs(values[t, 0] - other.failure) <= 5 * se + 2 / samples)
        compared += 1
    assert compared >= 1


# ---------------------------------------------------------------------------------------
# 4. no code, no failure
# ---------------------------------------------------------------------------------------


def test_purified_trajectories_record_zero():
    # 300 samples: more failures than a uint8 counts (the kernels must not count in it)
    probe = _probe(couplings=[0.0], sizes=[2, 14], samples=300, checkpoints=[0, 20, 40])
    spec = _ring_spec(
        layers=40,
        measurement_rate=0.5,
        checkpoints="every_layer",
        stop_when_purified=True,
        probes=[mipt.code_dimension(), probe],
    )
    result = spec.run(4, seed=2)
    assert (result.purification_layer >= 0).all()  # every trajectory purified and stopped
    _, _, values = result.correlated("correlated_erasure")
    k = result.values["code_dimension"][:, [0, 20, 40]]
    # at layer 0 every system qubit is maximally entangled with its reference (k = n), so
    # every region fails
    assert (k[:, 0] == N).all() and (values[:, 0] == 1.0).all()
    assert (k[:, 1:] == 0).any() and not values[k == 0].any()  # k = 0, before or after a stop
    assert mipt._probe_fill(spec.probes[1], N) == 0.0


# ---------------------------------------------------------------------------------------
# 5. reproducibility
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ws_run():
    spec = _ws_spec()
    return spec, spec.run(_size(6, 2), seed=40)


def test_same_seed_same_values_other_seed_other_values(ws_run):
    spec, result = ws_run
    again = spec.run(result.num_trajectories, seed=40)
    np.testing.assert_array_equal(
        again.values["correlated_erasure"], result.values["correlated_erasure"]
    )
    other = spec.run(result.num_trajectories, seed=41)
    assert not np.array_equal(
        other.values["correlated_erasure"], result.values["correlated_erasure"]
    )


def test_chunks_and_thread_counts_give_identical_values(ws_run):
    spec, result = ws_run
    count = result.num_trajectories
    threads = min(4, numba.config.NUMBA_NUM_THREADS)
    assert threads > 1 or JIT_FREE
    for run in (
        spec.run(count, seed=40, num_threads=1),
        spec.run(count, seed=40, num_threads=threads),
        mipt.EnsembleResult.concatenate(
            [spec.run(1, seed=40), spec.run(count - 1, seed=40, first_trajectory=1)]
        ),
    ):
        np.testing.assert_array_equal(
            run.values["correlated_erasure"], result.values["correlated_erasure"]
        )


def test_replay_reproduces_the_values_alone_and_with_narrower_sizes(ws_run):
    spec, result = ws_run
    probe = spec.probes[1]
    _, sizes, values = result.correlated(probe.name)
    for t in range(result.num_trajectories):
        for e, layer in enumerate(result.layers(probe.name).tolist()):
            alone = result.replay(t, probes=[probe], checkpoints=[layer])
            np.testing.assert_array_equal(alone.correlated(probe.name)[2][0, 0], values[t, e])
            narrow = dataclasses.replace(probe, sizes=(sizes[3], sizes[1]))
            part = result.replay(t, probes=[narrow], checkpoints=[layer])
            np.testing.assert_array_equal(
                part.correlated(probe.name)[2][0, 0], values[t, e][:, [3, 1]]
            )
    assert values.any()


def test_a_second_correlated_probe_does_not_change_the_first(ws_run):
    spec, result = ws_run
    other = _probe(name="other", couplings=[0.2], sizes=[3], burn_in=4)
    both = dataclasses.replace(spec, probes=[other, *spec.probes]).run(
        result.num_trajectories, seed=40
    )
    np.testing.assert_array_equal(
        both.values["correlated_erasure"], result.values["correlated_erasure"]
    )


# ---------------------------------------------------------------------------------------
# 6. storage, save and load, the protocol, the JSON interface, validation
# ---------------------------------------------------------------------------------------


def test_layout_is_coupling_major_and_the_helper_reshapes_it(ws_run):
    spec, result = ws_run
    flat = result.values["correlated_erasure"]
    couplings, sizes, values = result.correlated()
    assert flat.shape == (result.num_trajectories, 2, len(COUPLINGS) * len(SIZES))
    assert values.shape == (result.num_trajectories, 2, len(COUPLINGS), len(SIZES))
    for c in range(len(COUPLINGS)):
        for i in range(len(SIZES)):
            np.testing.assert_array_equal(values[:, :, c, i], flat[:, :, c * len(SIZES) + i])
    assert couplings.dtype == np.float64 and sizes.dtype == np.int64
    assert mipt._probe_length(spec.probes[1], N) == len(COUPLINGS) * len(SIZES)
    assert spec.probes[1].is_vector
    with pytest.raises(KeyError, match="no correlated_erasure probe named 'code_dimension'"):
        result.correlated("code_dimension")


def test_default_sizes_and_burn_in_resolve_for_each_system_size():
    probe = mipt.correlated_erasure([0.0, 1.0])
    assert probe.sizes == () and probe.burn_in is None
    spec12 = _ring_spec(probe, num_system=12, edges=graphs.cycle(12))
    resolved = spec12.probes[1]
    assert resolved.sizes == tuple(range(1, 12)) and resolved.burn_in == 200
    # a resolved probe resolves again for another n (as a budget does)
    spec64 = _ring_spec(resolved, num_system=64, edges=graphs.cycle(64))
    assert spec64.probes[1].sizes == tuple(range(1, 64))
    assert spec64.probes[1].burn_in == erasures.default_burn_in(64) == 512
    # every size written out is the same probe as the default, and so is the burn-in
    explicit = mipt.correlated_erasure([0.0, 1.0], sizes="1:12", burn_in=200)
    assert _ring_spec(explicit, num_system=12, edges=graphs.cycle(12)).probes[1] == resolved
    # explicit values stay explicit at another n
    kept = _ring_spec(explicit, num_system=64, edges=graphs.cycle(64)).probes[1]
    assert kept.sizes == tuple(range(1, 12)) and kept.burn_in == 200


def test_values_set_by_replace_win_over_the_defaults(ws_run):
    """A probe whose sizes or burn-in were filled in by a spec keeps a value set later with
    dataclasses.replace (it is explicit), and a field left at its default still follows
    another n."""
    probe = mipt.correlated_erasure(COUPLINGS, samples=_size(40, 12), thin=2, name="ce")
    spec = _ws_spec(probe, checkpoints=[40])
    resolved = spec.probes[1]
    assert resolved.sizes == tuple(range(1, N)) and resolved.burn_in == 200
    full = spec.run(_size(3, 1), seed=40)
    narrow = dataclasses.replace(resolved, sizes=(9, 3))
    for t in range(full.num_trajectories):
        part = full.replay(t, probes=[narrow], checkpoints=[40])
        assert part.correlated("ce")[1].tolist() == [9, 3]
        np.testing.assert_array_equal(
            part.correlated("ce")[2][0, 0], full.correlated("ce")[2][t, 0][:, [8, 2]]
        )
    # a probe no spec has seen yet, and the burn-in
    unseen = dataclasses.replace(mipt.correlated_erasure([0.0]), sizes=(2, 3))
    assert _ws_spec(unseen).probes[1].sizes == (2, 3)
    longer = _ws_spec(dataclasses.replace(resolved, burn_in=31)).probes[1]
    assert longer.burn_in == 31 and dict(longer.n_defaults) == {"sizes": tuple(range(1, N))}
    # the sizes left at their default follow another n; the explicit burn-in stays
    other = _ws_spec(longer, n=10).probes[1]
    assert other.sizes == tuple(range(1, 10)) and other.burn_in == 31


def _save_with_old_rules(tmp_path, *, burn_in=None):
    """A run with the default burn-in and the automatic ladder, saved."""
    probe = mipt.correlated_erasure(
        [0.0, 1.5], sizes=[3, 8], samples=_size(30, 8), name="ce", burn_in=burn_in
    )
    spec = _ws_spec(probe, checkpoints=[40])
    result = spec.run(_size(3, 2), seed=9)
    result.save(tmp_path / "run.npz")
    return probe, spec, result


def test_a_saved_run_pins_its_burn_in_and_ladders(tmp_path, monkeypatch):
    """Review M1: changing the default burn-in or the ladder rule after a save changes
    nothing a loaded run does: it keeps its burn-in and ladders, replays exactly (also a
    narrower probe), and a resume or merge under the new rules is another protocol."""
    probe, spec, result = _save_with_old_rules(tmp_path)
    assert spec.probes[1].burn_in == 200 and spec.correlated_sampler("ce")["method"] == "chain"
    monkeypatch.setattr(erasures, "MIN_BURN_IN", 400)  # a new default burn-in ...
    moments = erasures._uniform_moments  # ... and a ladder twice as dense
    monkeypatch.setattr(
        erasures, "_uniform_moments", lambda *a: (moments(*a)[0], 4 * moments(*a)[1])
    )
    fresh = _ws_spec(probe, checkpoints=[40])
    assert fresh.probes[1].burn_in == 400
    assert not mipt._same_plan(fresh._plans["ce"], spec._plans["ce"])  # the rules did change
    loaded = mipt.EnsembleResult.load(tmp_path / "run.npz")
    pinned = loaded.spec.probes[1]
    assert pinned.burn_in == 200 and pinned.n_defaults == () and pinned == spec.probes[1]
    assert mipt._same_plan(loaded.spec._plans["ce"], spec._plans["ce"])
    assert mipt._protocol_difference(loaded.spec, spec) is None
    narrow = dataclasses.replace(pinned, sizes=(8,))
    for t in range(result.num_trajectories):
        again = loaded.replay(t)
        np.testing.assert_array_equal(again.values["ce"][0], result.values["ce"][t])
        part = loaded.replay(t, probes=[narrow], checkpoints=[40])
        np.testing.assert_array_equal(
            part.correlated("ce")[2][0, 0], result.correlated("ce")[2][t, 0][:, [1]]
        )
    assert mipt._protocol_difference(loaded.spec, fresh) == "probes"  # its burn-in is 400
    same_burn_in = _ws_spec(dataclasses.replace(probe, burn_in=200), checkpoints=[40])
    assert same_burn_in.probes[1] == pinned
    assert "ladders" in mipt._protocol_difference(loaded.spec, same_burn_in)
    with pytest.raises(ValueError, match="ladders .* differs"):
        mipt.EnsembleResult.concatenate([loaded, same_burn_in.run(1, seed=9, first_trajectory=5)])


def test_files_without_a_stored_plan_pin_their_burn_in(tmp_path, monkeypatch):
    """A record as the probe's first version wrote it (n_defaults, no plan) loads with the
    burn-in it stored, not the current default; a stored plan that does not fit is refused."""
    _, _, result = _save_with_old_rules(tmp_path)
    path = tmp_path / "run.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["__meta__"]))
    record = meta["probes"][1]
    plan = record.pop("plan")
    record["n_defaults"] = ["burn_in"]
    arrays["__meta__"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    monkeypatch.setattr(erasures, "MIN_BURN_IN", 400)
    loaded = mipt.EnsembleResult.load(path)
    assert loaded.spec.probes[1].burn_in == 200
    np.testing.assert_array_equal(loaded.replay(1).values["ce"][0], result.values["ce"][1])
    record["plan"] = dict(plan, graph_class=[0, 0])  # one graph, two classes listed
    arrays["__meta__"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="stored plan of probe 'ce' does not fit"):
        mipt.EnsembleResult.load(path)


def test_working_memory_of_all_running_threads_is_bounded(monkeypatch, ws_run):
    spec, _ = ws_run
    per_thread = spec.correlated_bytes_per_thread()
    assert per_thread == 8 * len(COUPLINGS) * spec.probes[1].samples * (max(SIZES) + 1)
    assert (
        _ring_spec(
            probe=mipt.code_dimension(), probes=[mipt.code_dimension()]
        ).correlated_bytes_per_thread()
        == 0
    )
    running = min(3, numba.config.NUMBA_NUM_THREADS)
    monkeypatch.setattr(mipt, "MAX_WORKING_BYTES", running * per_thread - 1)
    with pytest.raises(ValueError, match=f"in each of {running} running threads"):
        spec.run(running, seed=1, num_threads=running)
    if running > 1:  # fewer threads, or fewer trajectories than threads, fit
        assert spec.run(running, seed=1, num_threads=running - 1).num_trajectories == running
        assert spec.run(running - 1, seed=1, num_threads=running).num_trajectories == running - 1


def test_save_and_load_round_trip(tmp_path, ws_run):
    spec, result = ws_run
    path = tmp_path / "run.npz"
    result.save(path)
    loaded = mipt.EnsembleResult.load(path)
    assert loaded.spec.probes == spec.probes
    assert mipt._protocol_difference(loaded.spec, spec) is None
    for a, b in zip(loaded.correlated(), result.correlated(), strict=True):
        np.testing.assert_array_equal(a, b)
    # the replay of a loaded run reproduces its values
    again = loaded.replay(1, probes=[spec.probes[1]], checkpoints=[40])
    np.testing.assert_array_equal(again.correlated()[2][0, 0], result.correlated()[2][1, 1])
    # every setting is saved: a probe with non-default ones survives too
    probe = _probe(
        name="settings", ladder=[0.25], max_step=0.5, tempering=True, thin=3, burn_in=7,
        method="chain", geometry="path", samples=3, sizes=[4, 2],
    )  # fmt: skip
    small = _ws_spec(probe, layers=4, checkpoints="final").run(1, seed=1)
    small.save(tmp_path / "small.npz")
    assert mipt.EnsembleResult.load(tmp_path / "small.npz").spec.probes == small.spec.probes
    # the other probes' records hold none of the new keys, and old files load as before
    meta = json.loads(str(np.load(tmp_path / "small.npz")["__meta__"]))
    assert "couplings" not in meta["probes"][0] and "couplings" in meta["probes"][1]


def test_other_couplings_or_settings_are_another_protocol(ws_run):
    spec, result = ws_run
    for change in (
        dict(couplings=(1.5, 0.0, -0.6)),
        dict(couplings=(0.0, 1.5, -0.5)),  # the layout follows the order
        dict(sizes=(2, 5, 8, 11)),
        dict(burn_in=31),
        dict(thin=1),
        dict(tempering=False),
        dict(ladder="given"),
        dict(max_step=0.2),
        dict(geometry="cycle"),
        dict(method="chain"),
        dict(samples=41),
    ):
        probe = dataclasses.replace(spec.probes[1], **change)
        other = dataclasses.replace(spec, probes=[spec.probes[0], probe, spec.probes[2]])
        assert mipt._protocol_difference(spec, other) == "probes", change
        with pytest.raises(ValueError, match="probes differ"):
            mipt.EnsembleResult.concatenate([result, other.run(1, seed=40, first_trajectory=9)])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(couplings=[]), "at least one coupling"),
        (dict(couplings=[0.0, math.inf]), "J -> \\+inf is a limit"),
        (dict(couplings=[math.nan]), "finite"),
        (dict(couplings=[1.0, 1.0]), "distinct"),
        (dict(couplings=[True]), "numbers"),
        (dict(couplings="0.5"), "numbers"),
        (dict(couplings=[[0.5]]), "numbers"),
        (dict(sizes=[]), "at least one size"),
        (dict(sizes=[3, 3]), "distinct"),
        (dict(sizes=[2.5]), "integers"),
        (dict(sizes="3"), "range"),
        (dict(sizes="5:2"), "at least one size"),
        (dict(samples=0), "samples must be at least 1"),
        (dict(thin=0), "thin must be at least 1"),
        (dict(burn_in=-1), "burn_in must be at least 0"),
        (dict(tempering="yes"), "tempering must be True or False"),
        (dict(method="metropolis"), "method must be"),
        (dict(geometry="torus"), "geometry must be"),
        (dict(ladder="dense"), "ladder must be"),
        (dict(ladder=[math.inf]), "finite"),
        (dict(max_step=0), "max_step must be a positive number"),
    ],
)
def test_constructor_validation(kwargs, message):
    with pytest.raises((ValueError, TypeError), match=message):
        mipt.correlated_erasure(**{"couplings": [0.0], **kwargs})


@pytest.mark.parametrize(
    ("probe", "spec_kw", "message"),
    [
        (dict(sizes=[0, 3]), {}, r"sizes must lie in \[1, 15\]"),
        (dict(sizes=[16]), {}, r"sizes must lie in \[1, 15\]"),
        (dict(method="exact"), {}, "method='exact' needs"),
        (dict(geometry="cycle"), dict(num_system=2, edges=[[0, 1]]), "at least 3"),
        (dict(tempering=False, ladder=[0.5]), {}, "only help with tempering=True"),
        (dict(couplings=[0.0, 1e6]), {}, "exchange ladder needs"),
        (dict(samples=10**8, sizes=[8]), {}, "per thread"),
        ({}, dict(reference="none"), "needs a reference"),
    ],
)
def test_spec_validation(probe, spec_kw, message):
    p = mipt.correlated_erasure(**{"couplings": [0.0, 1.0], **probe})
    base = dict(
        num_system=N, edges=WS_EDGES, layers=4, measurement_rate=0.1, reference="purification",
        probes=[p],
    )  # fmt: skip
    base.update(spec_kw)
    with pytest.raises(ValueError, match=message):
        mipt.MonitoredCircuit(**base)


def test_exact_method_and_geometries_that_allow_it(monkeypatch):
    # method="exact" works on the ring and the open chain (and their relabelings), and with
    # geometry cycle/path; the values are the graph side's on the relabeled graphs too
    perm = np.random.default_rng(0).permutation(N)
    cases = (
        (perm[graphs.cycle(N)], "interaction", ["cycle"]),
        (perm[graphs.path(N)], "interaction", ["path"]),
        (WS_EDGES, "cycle", ["cycle"]),
        (WS_EDGES, "path", ["path"]),
    )
    for edges, geometry, kinds in cases:
        p = mipt.correlated_erasure([0.0, 2.0], sizes=[3, 9], samples=6, method="exact",
                                    geometry=geometry)  # fmt: skip
        spec = _ring_spec(p, edges=edges, schedule="random_edges", layers=8, checkpoints="final")
        sampler = spec.correlated_sampler("correlated_erasure")
        assert sampler["method"] == "exact" and sampler["geometries"] == kinds
        result = spec.run(2, seed=1)
        assert result.values["correlated_erasure"].shape == (2, 1, 4)
        assert _check_fractions(monkeypatch, spec, result)["methods"] == {"exact"}


def test_graph_collections_sample_each_trajectory_on_its_own_graph(monkeypatch):
    """A collection mixing chain graphs (two Watts-Strogatz graphs of one degree sequence
    class or another) and exact ones (a relabeled ring, a path): every trajectory's regions
    are the graph side's on its own graph."""
    perm = np.random.default_rng(1).permutation(N)
    bank = [
        graphs.watts_strogatz(N, 4, 0.3, seed=(5, 0)),
        perm[graphs.cycle(N)],
        graphs.watts_strogatz(N, 4, 0.3, seed=(5, 1)),
        graphs.path(N),
    ]
    probe = _probe(burn_in=10, sizes=[3, 8, 12])
    spec = _ws_spec(probe, edges=bank, circuits_per_graph=2, checkpoints=[40])
    sampler = spec.correlated_sampler("correlated_erasure")
    assert sampler["method"] == "mixed" and sampler["exact_graphs"] == 2
    assert sampler["graphs"] == 4 and sampler["geometries"] == ["cycle", "path"]
    result = spec.run(seed=4)
    assert result.num_trajectories == 8
    seen = _check_fractions(monkeypatch, spec, result)
    assert seen["methods"] == {"exact", "chain"} and seen["codes"] >= 4
