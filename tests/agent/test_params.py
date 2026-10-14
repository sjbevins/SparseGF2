"""Vocabularies and the shared parameter groups (flat tool arguments -> JSON API specs)."""

import dataclasses

import pytest

from sparsegf2 import api, mipt
from sparsegf2.agent import params, vocab
from sparsegf2.agent.context import Context


@pytest.fixture
def ctx(tmp_path):
    return Context(tmp_path)


def test_vocab_matches_the_library():
    assert vocab.probe_signatures()["mutual_information"] == ("a", "b")
    assert vocab.probe_signatures()["erasure_first_failure"] == ("samples",)
    assert vocab.observable_signatures()["entropy"] == ("region",)
    assert vocab.graph_families()["watts_strogatz"] == ("k", "beta")
    assert "watts_strogatz" in vocab.random_graph_families()
    assert "cycle" not in vocab.random_graph_families()
    assert vocab.weight_families()["power_law"] == ("alpha", "periodic")
    assert "CX" in vocab.gate_names()["unitary"]


def test_every_monitored_circuit_field_is_mapped():
    fields = {
        f.name for f in dataclasses.fields(mipt.MonitoredCircuit) if not f.name.startswith("_")
    }
    assert set(params.FIELD_MAP) == fields


def test_watts_strogatz_collection_spec(ctx):
    args = {
        "num_system": 16,
        "layers": 8,
        "measurement_rate": 0.1,
        "graph": "watts_strogatz",
        "graph_count": 5,
        "graph_seed": 3,
        "circuits_per_graph": 2,
        "reference_partner": -1,
        "probes": ["code_dimension", "erasure_first_failure(512)@[4, 8]"],
        "checkpoints": "every_layer",
        "schedule": "random_edges",
        "reference": "purification",
    }
    spec = params.ensemble_spec(args, ctx)
    assert spec["graph"] == {
        "family": "watts_strogatz",
        "k": 4,
        "beta": 0.1,
        "seed": 3,
        "realizations": 5,
    }
    assert spec["reference_partner"] == "random"
    assert spec["probes"] == [
        {"kind": "code_dimension"},
        {"kind": "erasure_first_failure", "samples": 512, "checkpoints": [4, 8]},
    ]
    assert spec["checkpoints"] == "every_layer"
    mc = api._build_spec(spec)
    assert mc.num_trajectories == 10


def test_graph_parameters_must_fit_the_family(ctx):
    with pytest.raises(ValueError, match="graph_beta does not apply to graph=cycle"):
        params.graph_dict(
            {"graph": "cycle", "graph_beta": 0.2, "graph_count": 1, "graph_seed": 0}, 8, ctx
        )
    with pytest.raises(ValueError, match="graph_count applies to random families"):
        params.graph_dict({"graph": "cycle", "graph_count": 3, "graph_seed": 0}, 8, ctx)
    with pytest.raises(ValueError, match="graph=file needs graph_file"):
        params.graph_dict({"graph": "file", "graph_count": 1, "graph_seed": 0}, 8, ctx)


def test_probe_and_observable_strings(ctx):
    assert params.probe_dicts(["entropy(0:4, name=left)@final"]) == [
        {"kind": "entropy", "region": "0:4", "name": "left", "checkpoints": "final"}
    ]
    with pytest.raises(ValueError, match="unknown probe 'entrpy'.*entropy"):
        params.probe_dicts(["entrpy(0:4)"])
    assert params.observable_dicts(["expectation(Z0*Z1)", "code_dimension(reference=4:8)"]) == [
        {"kind": "expectation", "pauli": "Z0*Z1"},
        {"kind": "code_dimension", "reference": "4:8"},
    ]


def test_edge_weights_gate_set_and_system_order(ctx):
    args = {
        "num_system": 8,
        "layers": 2,
        "measurement_rate": 0.1,
        "graph": "complete",
        "graph_count": 1,
        "graph_seed": 0,
        "schedule": "random_edges",
        "edge_weights": "power_law(alpha=2.0)",
        "gate_set": ["CX", "3"],
        "system_order": "[7,6,5,4,3,2,1,0]",
    }
    spec = params.ensemble_spec(args, ctx)
    assert spec["edge_weights"] == {"family": "power_law", "alpha": 2.0}
    assert spec["gate_set"] == ["CX", 3]
    assert spec["system_order"] == [7, 6, 5, 4, 3, 2, 1, 0]
    api._build_spec(spec)  # valid for the JSON API
