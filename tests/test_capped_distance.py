"""The capped contiguous distance probe, contiguous_distance(cap=c): min(d, c) through the engine, the saved
result, replay and the JSON API (the kernel against the full distance: test_reference_rows.py)."""

import numpy as np
import pytest

from sparsegf2 import api, graphs, mipt

N = 32


def _spec(**changes):
    probes = [
        mipt.code_dimension(),
        mipt.contiguous_distance(),
        mipt.contiguous_distance(cap=2),
        mipt.contiguous_distance(cap=3, name="d3"),
        mipt.contiguous_distance(
            cap=12, name="d12"
        ),  # (12 - 1)**2 > 2 N: the full distance, capped
    ]
    kw = dict(
        num_system=N,
        edges=graphs.watts_strogatz(N, 4, 1.0, seed=1),
        layers=60,
        measurement_rate=0.08,
        schedule="random_edges",
        reference="purification",
        checkpoints="every_layer",
        probes=probes,
    )
    return mipt.MonitoredCircuit(**(kw | changes))


@pytest.mark.parametrize(
    "layout",
    [
        {},
        {"periodic": False},
        {"system_order": np.random.default_rng(3).permutation(N).tolist()},
        {"reference": "single", "measurement_rate": 0.02},
    ],
    ids=["ring", "open", "permuted", "single_reference"],
)
def test_the_engine_records_the_full_distance_capped(layout):
    res = _spec(**layout).run(24, seed=5)
    full = res.values["contiguous_distance"]
    for name, cap in (("contiguous_distance[cap=2]", 2), ("d3", 3), ("d12", 12)):
        capped = res.values[name]
        # NaN exactly where there is no code
        assert np.array_equal(np.isnan(capped), np.isnan(full))
        ok = ~np.isnan(full)
        assert np.array_equal(capped[ok], np.minimum(full[ok], cap))
    finite = full[~np.isnan(full)]
    assert finite.size and finite.max() >= 3  # some distances reach the caps
    if not layout:
        assert {1.0, 2.0} <= set(finite) and finite.max() >= 4  # below, at and above the caps


def test_cap_is_checked_and_names_follow_it():
    assert mipt.contiguous_distance().name == "contiguous_distance"
    assert mipt.contiguous_distance().size == 0
    p = mipt.contiguous_distance(cap=2.0)
    assert p.size == 2 and p.name == "contiguous_distance[cap=2]"
    assert mipt.contiguous_distance(cap=4, name="d4").name == "d4"
    for bad in (1, 0, -3):
        with pytest.raises(ValueError, match="cap must be at least 2"):
            mipt.contiguous_distance(cap=bad)
    for bad in (2.5, True, "2"):
        with pytest.raises(TypeError):
            mipt.contiguous_distance(cap=bad)


def test_a_cap_of_at_least_n_is_the_full_distance():
    spec = _spec(probes=[mipt.contiguous_distance(), mipt.contiguous_distance(cap=N, name="dn")])
    res = spec.run(8, seed=2)
    a, b = res.values["contiguous_distance"], res.values["dn"]
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(
        a[~np.isnan(a)], b[~np.isnan(b)]
    )


def test_saved_loaded_and_replayed(tmp_path):
    res = _spec().run(6, seed=9)
    path = tmp_path / "run.npz"
    res.save(path)
    back = mipt.EnsembleResult.load(path)
    assert [(p.name, p.size) for p in back.spec.probes] == [
        (p.name, p.size) for p in res.spec.probes
    ]
    again = back.replay(3)
    for name in ("contiguous_distance[cap=2]", "d3"):
        np.testing.assert_array_equal(again.values[name][0], res.values[name][3])


def test_the_json_api_takes_a_cap_and_prices_it_below_the_full_distance():
    spec = {
        "num_system": N,
        "graph": {"family": "watts_strogatz", "k": 4, "beta": 1.0, "seed": 1},
        "layers": 40,
        "measurement_rate": 0.08,
        "schedule": "random_edges",
        "reference": "purification",
        "checkpoints": "every_layer",
        "probes": [{"kind": "code_dimension"}, {"kind": "contiguous_distance", "cap": 2}],
    }
    out = api.run_ensemble(spec, trajectories=8, seed=1)
    assert "contiguous_distance[cap=2]" in out["probes"]
    assert any("capped at 2" in note for note in out["notes"])
    with pytest.raises(ValueError, match="cap must be at least 2"):
        api.run_ensemble(
            {**spec, "probes": [{"kind": "contiguous_distance", "cap": 1}]}, trajectories=2, seed=1
        )
    big = {
        **spec,
        "num_system": 512,
        "graph": {"family": "watts_strogatz", "k": 4, "beta": 1.0, "seed": 1},
    }
    capped = api.estimate_ensemble(big, trajectories=100)
    full = api.estimate_ensemble(
        {**big, "probes": [{"kind": "code_dimension"}, {"kind": "contiguous_distance"}]},
        trajectories=100,
    )
    assert capped["estimated_seconds"] < full["estimated_seconds"] / 10
    # a cap the kernel settles by the full distance is priced as the full distance
    mc = api._build_spec(big)
    whole = api._probe_us(mipt.contiguous_distance(), mc)
    assert api._probe_us(mipt.contiguous_distance(cap=8), mc) < whole / 10
    assert api._probe_us(mipt.contiguous_distance(cap=34), mc) == whole  # 33**2 > 2 * 512
    assert "cap" in api.capabilities("probes")["probes"]["contiguous_distance"]["params"]
    assert {"cap": 2} in api.capabilities("probes")["probes"]["contiguous_distance"]["forms"]
