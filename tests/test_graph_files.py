"""graphs.save / graphs.load and the 'file' graph family of the JSON API."""

import json

import numpy as np
import pytest

from sparsegf2 import api, graphs


@pytest.mark.parametrize("suffix", [".json", ".npz"])
def test_round_trip_collection_with_weights(tmp_path, suffix):
    gs = [graphs.watts_strogatz(12, 4, 0.3, seed=(0, i)) for i in range(3)]
    ws = [np.linspace(1.0, 2.0, len(g)) for g in gs]
    path = graphs.save(tmp_path / f"c{suffix}", gs, 12, weights=ws)
    n, loaded, lw = graphs.load(path)
    assert n == 12 and len(loaded) == 3
    for a, b, wa, wb in zip(gs, loaded, ws, lw, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(wa, wb)


def test_json_single_graph_in_any_order_is_normalised(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(
        json.dumps({"num_nodes": 4, "edges": [[3, 2], [0, 1], [2, 1]], "weights": [3.0, 1.0, 2.0]}),
        encoding="utf-8",
    )
    n, gs, ws = graphs.load(path)
    np.testing.assert_array_equal(gs[0], [[0, 1], [1, 2], [2, 3]])
    np.testing.assert_allclose(ws[0], [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"num_nodes": 3, "edges": [[0, 3]]}, "node 3 outside"),
        ({"num_nodes": 3, "edges": [[1, 1]]}, "self-loop"),
        ({"num_nodes": 3, "edges": [[0, 1], [1, 0]]}, "repeated edge"),
        ({"edges": [[0, 1]]}, "num_nodes"),
        ({"num_nodes": 3}, "edges"),
    ],
)
def test_validation_errors_name_the_problem(tmp_path, payload, message):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        graphs.load(path)


def test_file_family_in_ensemble_spec(tmp_path):
    gs = [graphs.cycle(8), graphs.complete(8)]
    path = graphs.save(tmp_path / "two.json", gs, 8)
    spec = {
        "num_system": 8,
        "graph": {"family": "file", "path": str(path)},
        "layers": 4,
        "measurement_rate": 0.2,
        "reference": "purification",
        "probes": [{"kind": "code_dimension"}],
    }
    out = api.run_ensemble(spec, seed=1)  # a collection: every graph once
    assert out["num_trajectories"] == 2


def test_file_family_size_mismatch_and_weights_conflict(tmp_path):
    path = graphs.save(tmp_path / "w.npz", [graphs.cycle(8)], 8, weights=[np.ones(8)])
    base = {"graph": {"family": "file", "path": str(path)}, "layers": 2, "measurement_rate": 0.1}
    with pytest.raises(ValueError, match="num_nodes 8"):
        api.estimate_ensemble({**base, "num_system": 10}, 4)
    with pytest.raises(ValueError, match="already holds weights"):
        api.estimate_ensemble(
            {
                **base,
                "num_system": 8,
                "schedule": "random_edges",
                "edge_weights": {"family": "power_law", "alpha": 2.0},
            },
            4,
        )
