"""The experiment manifest of the Watts–Strogatz study (``sparsegf2.study.manifest``).

Strict reading (every malformed manifest is refused with a message that names the field),
the canonical form and digest, the resolved conditions, the checkpoint rule of v1.4 §9
(hand-computed schedules, an independent high-precision implementation, property tests),
the keyword set that drives ``MonitoredCircuit``, and the revision rules of design §3.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
from decimal import Decimal, localcontext

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sparsegf2 import mipt
from sparsegf2.study import identity as ident
from sparsegf2.study import manifest as mf
from sparsegf2.study.identity import Token
from sparsegf2.study.manifest import (
    ExperimentManifest,
    check_extension,
    geometric_checkpoints,
    round_half_up_power,
)

PROPERTY = settings(max_examples=200, deadline=None, database=None, derandomize=True)

EXAMPLE_TEXT = """
{"schema": "sparsegf2.study.manifest/1", "experiment_id": "ws-pilot-001",
 "label": "exploratory",
 "conditions": [{"n": [16, 24], "beta": ["0", "0.1"], "p": ["0.1", "0.15", "0.2"]}],
 "depth": {"layers": {"per_n": 8}, "checkpoints": {"count": 12}},
 "trials": {"graphs": 20, "circuits_per_graph": 2, "circuit_index_stride": 8,
            "measurement_replicas": 1},
 "seeds": {"master_graph_seed": 17, "master_circuit_seed": 29, "gate_mode": "shared"},
 "dynamics": {"stop_when_purified": false},
 "observables": {"tableau_snapshots": true, "contiguous_distance": true,
                 "uniform_erasure": {"samples": 200, "exact_budget": 5000},
                 "correlated_erasure": [{"couplings": [0, 1, 2], "sizes": null, "samples": 100,
                                         "thin": 20, "geometry": "interaction"}]},
 "storage": {"graphs_per_unit": 8, "compression_level": 6},
 "resources": {"num_threads": null, "max_unit_trajectories": 4096},
 "analysis": {"peak": {"rule": "plateau", "z": 1.0, "sensitivity_z": [0, 1, 2],
                       "bootstrap": 200},
              "bootstrap": 1000, "censoring": "survival"}}
"""
EXAMPLE = json.loads(EXAMPLE_TEXT)
DELETE = object()


def mutated(path: str, value=DELETE, base=None) -> dict:
    """A deep copy of the example with the entry at ``path`` ("trials.graphs", "conditions.0.n")
    replaced by ``value`` or, for ``DELETE``, removed."""
    data = copy.deepcopy(EXAMPLE if base is None else base)
    *parents, last = path.split(".")
    node = data
    for key in parents:
        node = node[int(key)] if isinstance(node, list) else node[key]
    key = int(last) if isinstance(node, list) else last
    if value is DELETE:
        del node[key]
    else:
        node[key] = value
    return data


@pytest.fixture(scope="module")
def manifest() -> ExperimentManifest:
    return ExperimentManifest.from_json(EXAMPLE_TEXT)


# ---------------------------------------------------------------------------
# reading, the canonical form and the digest
# ---------------------------------------------------------------------------


def test_the_design_example_parses_and_round_trips(manifest):
    assert manifest.experiment_id == "ws-pilot-001" and manifest.label == "exploratory"
    assert manifest.trials.graphs == 20 and manifest.trials.circuits_per_graph == 2
    assert manifest.stride == 8 and manifest.trajectories_per_condition == 40
    assert manifest.seeds.gate_mode == "shared" and manifest.dynamics.stop_when_purified is False
    assert manifest.observables.correlated_erasure[0].couplings == (0.0, 1.0, 2.0)
    assert manifest.observables.correlated_erasure[0].thin == 20
    assert manifest.warnings == ()
    again = ExperimentManifest.from_json(manifest.to_json())
    assert again == manifest and again.to_json() == manifest.to_json()
    assert again.digest() == manifest.digest()
    assert ExperimentManifest.from_json(manifest.to_dict()) == manifest
    assert json.loads(manifest.to_json(indent=2)) == json.loads(manifest.to_json())


def test_the_canonical_text_writes_every_default_and_tokens_as_strings(manifest):
    doc = json.loads(manifest.to_json())
    assert doc["schema"] == "sparsegf2.study.manifest/1"
    assert doc["conditions"] == [{"n": [16, 24], "beta": ["0", "0.1"], "p": ["0.1", "0.15", "0.2"]}]
    assert doc["depth"] == {"layers": {"per_n": 8}, "checkpoints": {"count": 12}}
    assert doc["trials"]["measurement_replicas"] == 1
    assert doc["observables"]["correlated_erasure"][0]["burn_in"] is None
    assert doc["observables"]["correlated_erasure"][0]["name"] is None
    assert " " not in manifest.to_json() and "\n" not in manifest.to_json()
    # sorted keys at every level
    assert manifest.to_json().index('"analysis"') < manifest.to_json().index('"conditions"')


def test_minimal_manifest_gets_the_defaults_of_the_design_example():
    minimal = {
        "experiment_id": "m",
        "conditions": [{"n": [8], "beta": [0.5], "p": [0.1]}],
        "depth": {"layers": {"fixed": 16}, "checkpoints": {"count": 4}},
        "trials": {"graphs": 3, "circuits_per_graph": 2},
        "seeds": {"master_graph_seed": 1, "master_circuit_seed": 2},
    }
    m = ExperimentManifest.from_json(minimal)
    assert m.label == "exploratory" and m.schema == "sparsegf2.study.manifest/1"
    assert m.stride == 2 and m.trials.measurement_replicas == 1  # the stride defaults to C
    assert m.seeds.gate_mode == "shared" and m.dynamics.stop_when_purified is False
    obs = m.observables
    assert obs.tableau_snapshots and obs.contiguous_distance and obs.correlated_erasure == ()
    assert obs.uniform_erasure == mf.UniformErasure(samples=200, exact_budget=5000)
    assert m.storage == mf.Storage(graphs_per_unit=8, compression_level=6)
    assert m.resources == mf.Resources(num_threads=None, max_unit_trajectories=4096)
    assert m.analysis == mf.Analysis(
        peak=mf.Peak("plateau", 1.0, (0.0, 1.0, 2.0), 200), bootstrap=1000, censoring="survival"
    )
    # spelling the defaults out changes nothing, not even the digest
    explicit = ExperimentManifest.from_json(m.to_json())
    assert explicit.digest() == m.digest()


def _shuffled(obj, rng):
    if isinstance(obj, dict):
        items = [(k, _shuffled(v, rng)) for k, v in obj.items()]
        rng.shuffle(items)
        return dict(items)
    if isinstance(obj, list):
        return [_shuffled(v, rng) for v in obj]
    return obj


def test_digest_is_stable_across_key_order_whitespace_and_spelling(manifest):
    digest = manifest.digest()
    rng = random.Random(7)
    for _ in range(10):
        text = json.dumps(_shuffled(EXAMPLE, rng), indent=rng.choice([None, 0, 1, 4]))
        assert ExperimentManifest.from_json(text).digest() == digest
    compact = json.dumps(EXAMPLE, separators=(",", ":"))
    for text in (compact, json.dumps(EXAMPLE, indent=3), "\n\t " + EXAMPLE_TEXT + "\r\n"):
        assert ExperimentManifest.from_json(text).digest() == digest
    # the same conditions spelled with numbers and rationals, in any order, in one block or in
    # several: the digest identifies the experiment, not the factorization of its conditions
    spellings = [
        [{"n": [24, 16], "beta": [0.1, 0], "p": ["0.2", "3/20", 0.1]}],
        [
            {"n": [24], "beta": [0.1, "0"], "p": [0.2, "3/20", "1/10"]},
            {"n": [16], "beta": ["0.0", "1/10"], "p": ["0.10", "0.15", "2/10"]},
        ],
        [
            {"n": [16, 24], "beta": ["0"], "p": ["0.1", "0.15", "0.2"]},
            {"n": [24, 16], "beta": ["0.1"], "p": ["0.1", "0.15", "0.2"]},
        ],
        [
            {"n": [16], "beta": ["0"], "p": ["0.1"]},
            {"n": [16], "beta": ["0"], "p": ["0.15", "0.2"]},
            {"n": [16], "beta": ["0.1"], "p": ["0.1", "0.15", "0.2"]},
            {"n": [24], "beta": ["0", "0.1"], "p": ["0.1", "0.15", "0.2"]},
        ],
    ]
    for blocks in spellings:
        other = ExperimentManifest.from_json(mutated("conditions", blocks))
        assert other.digest() == digest and other == manifest
        assert other.to_dict()["conditions"] == manifest.to_dict()["conditions"]


def test_the_canonical_blocks_group_sizes_that_share_rates_and_betas():
    blocks = [
        {"n": [16], "beta": ["0"], "p": ["0.1", "0.2"]},
        {"n": [24], "beta": ["0"], "p": ["0.1", "0.2"]},
        {"n": [24], "beta": ["0.5"], "p": ["0.3"]},
        {"n": [32], "beta": ["0.5"], "p": ["0.3"]},
    ]
    m = ExperimentManifest.from_json(mutated("conditions", blocks))
    # n = 24 has both betas but the two rate sets differ, so the sizes group by rate set; the
    # blocks are listed in the order of their canonical JSON text
    assert m.to_dict()["conditions"] == [
        {"n": [16, 24], "beta": ["0"], "p": ["0.1", "0.2"]},
        {"n": [24, 32], "beta": ["0.5"], "p": ["0.3"]},
    ]
    assert ExperimentManifest.from_json(m.to_json()) == m


def test_digest_changes_with_every_scientific_setting(manifest):
    digests = {manifest.digest()}
    for path, value in [
        ("seeds.master_graph_seed", 18),
        ("seeds.master_circuit_seed", 30),
        ("seeds.gate_mode", "independent"),
        ("trials.graphs", 21),
        ("trials.circuit_index_stride", 9),
        ("depth.layers", {"per_n": 9}),
        ("depth.checkpoints", {"count": 13}),
        ("label", "confirmatory"),
        ("experiment_id", "ws-pilot-002"),
        ("dynamics.stop_when_purified", True),
        ("observables.uniform_erasure.samples", 201),
        ("analysis.peak.z", 1.5),
        ("conditions.0.p", ["0.1", "0.15"]),
    ]:
        digests.add(ExperimentManifest.from_json(mutated(path, value)).digest())
    assert len(digests) == 14


MINIMAL_TEXT = (
    '{"analysis":{"bootstrap":1000,"censoring":"survival","peak":{"bootstrap":200,"rule":"plateau",'
    '"sensitivity_z":[0.0,1.0,2.0],"z":1.0}},"conditions":[{"beta":["0.5"],"n":[8],"p":["0.1"]}],'
    '"depth":{"checkpoints":{"count":4},"layers":{"fixed":16}},"dynamics":{"stop_when_purified":false},'
    '"experiment_id":"m","label":"exploratory","observables":{"contiguous_distance":true,'
    '"correlated_erasure":[],"tableau_snapshots":true,"uniform_erasure":{"exact_budget":5000,'
    '"samples":200}},"resources":{"max_unit_trajectories":4096,"num_threads":null},'
    '"schema":"sparsegf2.study.manifest/1","seeds":{"gate_mode":"shared","master_circuit_seed":2,'
    '"master_graph_seed":1},"storage":{"compression_level":6,"graphs_per_unit":8},'
    '"trials":{"circuit_index_stride":2,"circuits_per_graph":2,"graphs":3,'
    '"measurement_replicas":1}}'
)


def test_the_canonical_text_of_a_minimal_manifest_is_the_documented_form():
    # a literal written from the schema (sorted keys, no spaces, tokens as strings, every default
    # spelled out), and a digest computed here with hashlib, not by the code under test
    minimal = {
        "experiment_id": "m",
        "conditions": [{"n": [8], "beta": [0.5], "p": [0.1]}],
        "depth": {"layers": {"fixed": 16}, "checkpoints": {"count": 4}},
        "trials": {"graphs": 3, "circuits_per_graph": 2},
        "seeds": {"master_graph_seed": 1, "master_circuit_seed": 2},
    }
    m = ExperimentManifest.from_json(minimal)
    assert m.to_json() == MINIMAL_TEXT
    assert m.digest() == hashlib.sha256(MINIMAL_TEXT.encode("utf-8")).hexdigest()
    assert m.digest() == "49f6a7f590c72379c81ec4156ea62c7e29763abcb9100b993a98583b5f0392c1"


def test_the_digest_of_the_design_example_is_pinned(manifest):
    # a change of this value means the canonical form changed: bump the manifest schema
    assert manifest.digest() == "071f97fc90da2e504f6cf19ec1908658b3d431840f404e374543156c6d4c6278"
    assert hashlib.sha256(manifest.to_json().encode("utf-8")).hexdigest() == manifest.digest()


def test_from_json_reads_text_dicts_and_files(manifest, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(EXAMPLE_TEXT, encoding="utf-8")
    assert ExperimentManifest.from_json(path) == manifest
    assert ExperimentManifest.from_json(EXAMPLE_TEXT) == manifest
    assert ExperimentManifest.from_json(EXAMPLE) == manifest
    with pytest.raises(ValueError, match="pass a pathlib.Path"):
        ExperimentManifest.from_json(str(path))
    with pytest.raises(ValueError, match="cannot read"):
        ExperimentManifest.from_json(tmp_path / "missing.json")


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("{", "not valid JSON"),
        ('{"a": 1,}', "not valid JSON"),
        ('{"experiment_id": "a", "experiment_id": "b"}', "key 'experiment_id' twice"),
        ('{"seeds": {"master_graph_seed": NaN}}', "no NaN"),
        ('{"p": Infinity}', "no Infinity"),
        ("[1, 2]", "must start with"),
        ("null", "must start with"),
    ],
)
def test_bad_json_text_is_refused(text, fragment):
    with pytest.raises(ValueError, match=fragment):
        ExperimentManifest.from_json(text)


def test_a_non_object_is_refused():
    with pytest.raises(ValueError, match="manifest: expected an object, got list"):
        ExperimentManifest.from_json([EXAMPLE])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected an object, got null"):
        ExperimentManifest.from_dict(None)


MALFORMED = [
    # the top level
    (
        "bogus",
        1,
        r"manifest: unknown key\(s\) \['bogus'\]; allowed keys are \['analysis', 'conditions'",
    ),
    ("conditions", DELETE, r"manifest: missing required key\(s\) \['conditions'\]"),
    ("depth", DELETE, r"missing required key\(s\) \['depth'\]"),
    ("trials", DELETE, r"missing required key\(s\) \['trials'\]"),
    ("seeds", DELETE, r"missing required key\(s\) \['seeds'\]"),
    ("experiment_id", DELETE, r"missing required key\(s\) \['experiment_id'\]"),
    ("schema", "sparsegf2.study.manifest/2", "manifest.schema: this version reads"),
    ("experiment_id", "has space", "manifest.experiment_id must be 1 to 64 characters"),
    ("experiment_id", 5, "manifest.experiment_id must be 1 to 64 characters"),
    ("label", "pilot", r"manifest.label: must be one of \['exploratory', 'confirmatory'\]"),
    # conditions
    ("conditions", [], "manifest.conditions: give at least one entry"),
    ("conditions", "n=16", "manifest.conditions: expected a list, got str"),
    ("conditions", [5], r"manifest.conditions\[0\]: expected an object, got int"),
    ("conditions.0.n", DELETE, r"manifest.conditions\[0\]: missing required key\(s\) \['n'\]"),
    ("conditions.0.q", [1], r"manifest.conditions\[0\]: unknown key\(s\) \['q'\]"),
    ("conditions.0.n", [], r"manifest.conditions\[0\].n: give at least one entry"),
    ("conditions.0.n", [16, 4], r"manifest.conditions\[0\].n\[1\]: must be at least 5, got 4"),
    ("conditions.0.n", ["16"], r"manifest.conditions\[0\].n\[0\]: must be an integer"),
    ("conditions.0.n", [16.0], r"manifest.conditions\[0\].n\[0\]: must be an integer, got 16.0"),
    ("conditions.0.n", [True], r"manifest.conditions\[0\].n\[0\]: must be an integer"),
    ("conditions.0.n", [16, 16], r"manifest.conditions\[0\].n: 16 appears twice"),
    ("conditions.0.n", 16, r"manifest.conditions\[0\].n: expected a list, got int"),
    ("conditions.0.beta", ["0.1", 1.5], r"manifest.conditions\[0\].beta\[1\] must lie in \[0, 1\]"),
    ("conditions.0.beta", ["x"], r"manifest.conditions\[0\].beta\[0\] must be a decimal"),
    ("conditions.0.beta", ["0.1", "1/10"], r"manifest.conditions\[0\].beta: 0.1 appears twice"),
    ("conditions.0.p", [-0.1], r"manifest.conditions\[0\].p\[0\] must lie in \[0, 1\]"),
    ("conditions.0.p", [None], r"manifest.conditions\[0\].p\[0\] must be a decimal"),
    ("conditions.0.p", [True], r"manifest.conditions\[0\].p\[0\] must be a probability"),
    ("conditions.0.p", [float("inf")], r"manifest.conditions\[0\].p\[0\] must be a finite"),
    ("conditions.0.p", ["0.1", "0.10"], r"manifest.conditions\[0\].p: 0.1 appears twice"),
    # depth
    ("depth.layers", {}, "manifest.depth.layers: give exactly one of 'per_n', 'fixed' or 'table'"),
    ("depth.layers", {"per_n": 8, "fixed": 3}, "give exactly one of"),
    ("depth.layers", {"per_n": 0}, r"manifest.depth.layers.per_n: must be at least 1, got 0"),
    ("depth.layers", {"per_n": 1.5}, "manifest.depth.layers.per_n: must be an integer"),
    ("depth.layers", {"fixed": -3}, r"manifest.depth.layers.fixed: must be at least 1"),
    ("depth.layers", {"per": 8}, r"manifest.depth.layers: unknown key\(s\) \['per'\]"),
    ("depth.layers", {"table": {}}, "manifest.depth.layers.table: expected an object"),
    ("depth.layers", {"table": [1]}, "manifest.depth.layers.table: expected an object"),
    (
        "depth.layers",
        {"table": {"16": 64}},
        r"manifest.depth.layers.table: no entry for n = \[24\]",
    ),
    (
        "depth.layers",
        {"table": {"16": 64, "24": 96, "99": 5}},
        r"table: n = \[99\] is not in any condition",
    ),
    ("depth.layers", {"table": {"x": 64}}, "keys are sizes n as strings"),
    ("depth.layers", {"table": {16: 64}}, "keys are sizes n as strings"),
    ("depth.layers", {"table": {"16": 0, "24": 5}}, r"table\[16\]: must be at least 1"),
    ("depth.checkpoints", {}, "manifest.depth.checkpoints: give exactly one of 'count' or 'list'"),
    ("depth.checkpoints", {"count": 3, "list": [1]}, "give exactly one of"),
    ("depth.checkpoints", {"count": 0}, r"checkpoints.count: must be at least 1"),
    ("depth.checkpoints", {"count": 1001}, r"checkpoints.count: must be at most 1000"),
    ("depth.checkpoints", {"count": 2.5}, "checkpoints.count: must be an integer"),
    ("depth.checkpoints", {"list": []}, r"checkpoints.list: give at least one entry"),
    ("depth.checkpoints", {"list": [-1, 3]}, r"checkpoints.list\[0\]: must be at least 0"),
    ("depth.checkpoints", {"list": [1, 2.0]}, r"checkpoints.list\[1\]: must be an integer"),
    ("depth.checkpoints", {"list": [1, 129]}, r"checkpoint 129 exceeds T_max = 128 for n = 16"),
    ("depth.checkpoints", {"list": "1,2"}, "checkpoints.list: expected a list"),
    ("depth.layers", DELETE, r"manifest.depth: missing required key\(s\) \['layers'\]"),
    ("depth.checkpoints", DELETE, r"manifest.depth: missing required key\(s\) \['checkpoints'\]"),
    # trials
    ("trials.graphs", 0, r"manifest.trials.graphs: must be at least 1, got 0"),
    ("trials.graphs", "20", "manifest.trials.graphs: must be an integer"),
    ("trials.graphs", DELETE, r"manifest.trials: missing required key\(s\) \['graphs'\]"),
    ("trials.circuits_per_graph", 0, r"circuits_per_graph: must be at least 1"),
    (
        "trials.circuit_index_stride",
        1,
        r"circuit_index_stride: must be at least circuits_per_graph \(2\), got 1",
    ),
    (
        "trials.measurement_replicas",
        2,
        r"measurement_replicas: must be 1, got 2: the native engine draws",
    ),
    ("trials.measurement_replicas", 0, "must be 1, got 0"),
    ("trials.replicas", 1, r"manifest.trials: unknown key\(s\) \['replicas'\]"),
    # seeds
    ("seeds.master_graph_seed", -1, r"master_graph_seed: must be at least 0"),
    ("seeds.master_graph_seed", 2**32, r"master_graph_seed: must be at most 4294967295"),
    ("seeds.master_circuit_seed", 1.5, "master_circuit_seed: must be an integer"),
    ("seeds.master_circuit_seed", 17, "master_graph_seed and master_circuit_seed must differ"),
    ("seeds.gate_mode", "coupled", r"seeds.gate_mode: must be one of \['shared', 'independent'\]"),
    ("seeds.master_circuit_seed", DELETE, r"missing required key\(s\) \['master_circuit_seed'\]"),
    # dynamics
    (
        "dynamics.stop_when_purified",
        "yes",
        "manifest.dynamics.stop_when_purified: must be true or false",
    ),
    ("dynamics.stop_when_purified", 1, "must be true or false"),
    ("dynamics.stop", True, r"manifest.dynamics: unknown key\(s\) \['stop'\]"),
    # observables
    ("observables.tableau_snapshots", 1, "observables.tableau_snapshots: must be true or false"),
    (
        "observables.contiguous_distance",
        "no",
        "observables.contiguous_distance: must be true or false",
    ),
    ("observables.bogus", 1, r"manifest.observables: unknown key\(s\) \['bogus'\]"),
    ("observables.uniform_erasure.samples", 0, r"uniform_erasure.samples: must be at least 1"),
    ("observables.uniform_erasure.exact_budget", -1, r"exact_budget: must be at least 0"),
    (
        "observables.uniform_erasure.exact_budget",
        10,
        r"exact_budget: 10 regions cannot even hold the 24 single-qubit",
    ),
    ("observables.uniform_erasure.shots", 10, r"uniform_erasure: unknown key\(s\) \['shots'\]"),
    ("observables.correlated_erasure", {"couplings": [1]}, "correlated_erasure: expected a list"),
    (
        "observables.correlated_erasure.0.couplings",
        DELETE,
        r"correlated_erasure\[0\]: missing required key\(s\) \['couplings'\]",
    ),
    (
        "observables.correlated_erasure.0.couplings",
        [],
        r"correlated_erasure\[0\].couplings: give at least one entry",
    ),
    ("observables.correlated_erasure.0.couplings", ["a"], r"couplings\[0\]: must be a number"),
    ("observables.correlated_erasure.0.couplings", [True], r"couplings\[0\]: must be a number"),
    (
        "observables.correlated_erasure.0.couplings",
        [1e7],
        r"correlated_erasure\[0\]: .*1e\+?0?6|MAX_ABS_COUPLING|10\^6|1000000",
    ),
    (
        "observables.correlated_erasure.0.couplings",
        [1, 1],
        r"correlated_erasure\[0\]: couplings must be distinct",
    ),
    ("observables.correlated_erasure.0.sizes", "bad", r"sizes: expected null, a list or a range"),
    ("observables.correlated_erasure.0.sizes", [0], r"sizes\[0\]: must be at least 1"),
    (
        "observables.correlated_erasure.0.sizes",
        [3, 3],
        r"correlated_erasure\[0\]: sizes must be distinct",
    ),
    ("observables.correlated_erasure.0.sizes", [24], r"sizes must lie in \[1, 15\] for n = 16"),
    ("observables.correlated_erasure.0.sizes", "1:40", r"sizes must lie in \[1, 15\] for n = 16"),
    ("observables.correlated_erasure.0.sizes", "5:5", "sizes: give at least one size"),
    (
        "observables.correlated_erasure.0.geometry",
        "torus",
        r"geometry: must be one of \['interaction', 'cycle', 'path'\]",
    ),
    (
        "observables.correlated_erasure.0.method",
        "magic",
        r"correlated_erasure\[0\].method: must be one of",
    ),
    ("observables.correlated_erasure.0.thin", 0, r"thin: must be at least 1"),
    ("observables.correlated_erasure.0.samples", 0, r"samples: must be at least 1"),
    ("observables.correlated_erasure.0.burn_in", -1, r"burn_in: must be at least 0"),
    ("observables.correlated_erasure.0.tempering", "yes", r"tempering: must be true or false"),
    ("observables.correlated_erasure.0.ladder", "x", r"ladder: must be one of \['auto', 'given'\]"),
    ("observables.correlated_erasure.0.max_step", "wide", r"max_step: must be a number"),
    (
        "observables.correlated_erasure.0.name",
        "bad name",
        r"correlated_erasure\[0\].name must be 1 to 64",
    ),
    (
        "observables.correlated_erasure.0.extra",
        1,
        r"correlated_erasure\[0\]: unknown key\(s\) \['extra'\]",
    ),
    # storage, resources, analysis
    ("storage.graphs_per_unit", 0, r"storage.graphs_per_unit: must be at least 1"),
    ("storage.compression_level", 10, r"storage.compression_level: must be at most 9"),
    ("storage.compression_level", -1, r"storage.compression_level: must be at least 0"),
    ("storage.path", "x", r"manifest.storage: unknown key\(s\) \['path'\]"),
    ("resources.num_threads", 0, r"resources.num_threads: must be at least 1"),
    ("resources.num_threads", 2.0, "resources.num_threads: must be an integer"),
    ("resources.max_unit_trajectories", 0, r"max_unit_trajectories: must be at least 1"),
    ("analysis.peak.rule", "exact", r"analysis.peak.rule: must be one of \['plateau'\]"),
    ("analysis.peak.z", -1, r"analysis.peak.z: must be at least 0"),
    ("analysis.peak.z", "1", "analysis.peak.z: must be a number"),
    ("analysis.peak.sensitivity_z", [], r"sensitivity_z: give at least one entry"),
    ("analysis.peak.sensitivity_z", [0, -1], r"sensitivity_z\[1\]: must be at least 0"),
    ("analysis.peak.bootstrap", -1, r"analysis.peak.bootstrap: must be at least 0"),
    ("analysis.bootstrap", 1.5, "analysis.bootstrap: must be an integer"),
    ("analysis.censoring", "none", r"analysis.censoring: must be one of \['survival'\]"),
]


@pytest.mark.parametrize(
    "path, value, fragment", MALFORMED, ids=[f"{p}={v!r}"[:60] for p, v, _ in MALFORMED]
)
def test_malformed_manifests_are_refused_naming_the_field(path, value, fragment):
    with pytest.raises(ValueError, match=fragment):
        ExperimentManifest.from_json(mutated(path, value))


def test_malformed_table_includes_enough_cases():
    assert len(MALFORMED) >= 100  # the brief: many cases, parametrized


def test_duplicate_conditions_across_blocks_and_equal_floats_are_refused():
    data = mutated(
        "conditions",
        [{"n": [16], "beta": ["0"], "p": ["0.1", "0.2"]}, {"n": [16], "beta": ["0"], "p": ["0.2"]}],
    )
    with pytest.raises(ValueError, match=r"\(n = 16, beta = 0, p = 0.2\) appears twice"):
        ExperimentManifest.from_json(data)
    near = mutated(
        "conditions",
        [{"n": [16], "beta": ["0.1", "0.1000000000000000055511151231257827"], "p": ["0.1"]}],
    )
    with pytest.raises(ValueError, match="same float64"):
        ExperimentManifest.from_json(near)
    rates = mutated(
        "conditions",
        [{"n": [16], "beta": ["0"], "p": ["0.1", "0.1000000000000000055511151231257827"]}],
    )
    with pytest.raises(ValueError, match="same float64"):
        ExperimentManifest.from_json(rates)


# ---------------------------------------------------------------------------
# conditions, depth
# ---------------------------------------------------------------------------


def test_conditions_are_resolved_in_a_stable_order(manifest):
    got = manifest.conditions()
    assert [(c.n, str(c.beta)) for c in got] == [(16, "0"), (16, "0.1"), (24, "0"), (24, "0.1")]
    assert all([str(p) for p in c.rates] == ["0.1", "0.15", "0.2"] for c in got)
    n, beta, rates = got[1]  # a NamedTuple unpacks as (n, beta, rates)
    assert (n, beta) == (16, Token.parse("0.1")) and rates[0] == Token.parse(0.1)
    assert manifest.rates(24, "0.1") == tuple(Token.parse(p) for p in ("0.1", "0.15", "0.2"))


def test_blocks_with_the_same_size_and_beta_merge_their_rates():
    data = mutated(
        "conditions",
        [
            {"n": [24, 16], "beta": ["1/2", "0"], "p": ["0.3", "1/10"]},
            {"n": [16], "beta": ["0.5"], "p": [0.2, "1"]},
            {"n": [10], "beta": [1], "p": [0]},
        ],
    )
    data["depth"]["layers"] = {"per_n": 4}
    data["observables"]["correlated_erasure"] = []
    data["observables"]["uniform_erasure"] = {"samples": 10, "exact_budget": 100}
    m = ExperimentManifest.from_json(data)
    flat = [(c.n, str(c.beta), [str(p) for p in c.rates]) for c in m.conditions()]
    assert flat == [
        (10, "1", ["0"]),
        (16, "0", ["0.1", "0.3"]),
        (16, "0.5", ["0.1", "0.2", "0.3", "1"]),
        (24, "0", ["0.1", "0.3"]),
        (24, "0.5", ["0.1", "0.3"]),
    ]
    # the order of the blocks and of the lists inside them does not matter
    swapped = copy.deepcopy(data)
    swapped["conditions"].reverse()
    swapped["conditions"][1]["n"].reverse()
    assert ExperimentManifest.from_json(swapped).digest() == m.digest()


def test_layers_rules(manifest):
    assert manifest.layers(16, "0.1") == 128 and manifest.layers(24, 0) == 192
    fixed = ExperimentManifest.from_json(mutated("depth.layers", {"fixed": 100}))
    assert fixed.layers(16, "0") == fixed.layers(24, "0.1") == 100
    table = ExperimentManifest.from_json(mutated("depth.layers", {"table": {"24": 50, "16": 40}}))
    assert (table.layers(16, 0), table.layers(24, 0)) == (40, 50)
    assert table.to_dict()["depth"]["layers"] == {"table": {"16": 40, "24": 50}}


@pytest.mark.parametrize(
    "call, fragment",
    [
        (
            lambda m: m.layers(20, "0"),
            r"\(n = 20, beta = 0\) is not a condition of experiment 'ws-pilot-001'",
        ),
        (lambda m: m.checkpoints(16, "0.2"), r"n = 16, beta = 0.2\) is not a condition"),
        (lambda m: m.rates(99, 0), "is not a condition"),
        (lambda m: m.layers(16.0, 0), "n: must be an integer"),
        (lambda m: m.layers(16, "x"), "beta must be a decimal"),
        (lambda m: m.monitored_circuit_kwargs(16, 0, "0.3"), r"p = 0.3\) is not a condition"),
        (lambda m: m.run_seed("2"), "must lie in"),
    ],
)
def test_unknown_conditions_are_refused(manifest, call, fragment):
    with pytest.raises(ValueError, match=fragment):
        call(manifest)


# ---------------------------------------------------------------------------
# the checkpoint rule (v1.4 section 9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "t_max, count, expected",
    [
        (128, 8, [0, 1, 2, 4, 8, 16, 32, 64, 128]),  # a_j = 2^j
        (64, 7, [0, 1, 2, 4, 8, 16, 32, 64]),
        (64, 4, [0, 1, 4, 16, 64]),  # 64^(1/3) = 4
        (1000, 4, [0, 1, 10, 100, 1000]),
        (100, 3, [0, 1, 10, 100]),
        (9, 3, [0, 1, 3, 9]),
        (25, 3, [0, 1, 5, 25]),
        (8, 4, [0, 1, 2, 4, 8]),
        (10, 4, [0, 1, 2, 5, 10]),  # 10^(1/3) = 2.154, 10^(2/3) = 4.642
        (12, 3, [0, 1, 3, 12]),  # sqrt 12 = 3.46
        (13, 3, [0, 1, 4, 13]),  # sqrt 13 = 3.61
        (24, 3, [0, 1, 5, 24]),  # sqrt 24 = 4.90
        (3, 3, [0, 1, 2, 3]),  # sqrt 3 = 1.73
        (5, 3, [0, 1, 2, 5]),  # sqrt 5 = 2.24
        (2, 5, [0, 1, 2]),  # 2^(1/4) = 1.19, 2^(1/2) = 1.41, 2^(3/4) = 1.68
        (128, 12, [0, 1, 2, 4, 6, 9, 14, 22, 34, 53, 82, 128]),  # 128^(j/11) = 2^(7 j / 11)
        (128, 2, [0, 1, 128]),
        (50, 1, [0, 1, 50]),  # N_T = 1: just 0, 1, T_max
        (50, 2, [0, 1, 50]),
        (1, 1, [0, 1]),
        (1, 7, [0, 1]),
        (2, 1, [0, 1, 2]),
        (3, 2, [0, 1, 3]),
    ],
)
def test_geometric_checkpoints_hand_computed(t_max, count, expected):
    got = geometric_checkpoints(t_max, count)
    assert list(got) == expected
    assert isinstance(got, tuple) and all(type(x) is int for x in got)


def test_the_design_example_checkpoints(manifest):
    assert manifest.checkpoints(16, "0") == (0, 1, 2, 4, 6, 9, 14, 22, 34, 53, 82, 128)
    assert manifest.checkpoints(24, "0.1")[0] == 0 and manifest.checkpoints(24, "0.1")[-1] == 192
    assert manifest.checkpoints(24, "0.1") == geometric_checkpoints(192, 12)
    # the same schedule for every rate of a condition (v1.4 section 1)
    assert manifest.layers(16, "0") == manifest.layers(16, "0.1") == 128


def _decimal_checkpoints(t_max: int, count: int) -> list[int]:
    """The rule of v1.4 section 9 in 120-digit decimal arithmetic, rounding half up."""
    points = {0, 1, t_max}
    if count >= 2:
        with localcontext() as ctx:
            ctx.prec = 120
            for j in range(count):
                a = (Decimal(j) * Decimal(t_max).ln() / Decimal(count - 1)).exp()
                points.add(
                    min(
                        max(int((a + Decimal("0.5")).to_integral_value(rounding="ROUND_FLOOR")), 0),
                        t_max,
                    )
                )
    return sorted(points)


@pytest.mark.parametrize("t_max", [1, 2, 3, 7, 10, 24, 64, 100, 128, 192, 255, 1000, 4096, 10**6])
@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 8, 12, 17, 33, 100])
def test_geometric_checkpoints_equal_high_precision_arithmetic(t_max, count):
    assert list(geometric_checkpoints(t_max, count)) == _decimal_checkpoints(t_max, count)


@PROPERTY
@given(st.integers(1, 50_000), st.integers(1, 60))
def test_checkpoint_invariants(t_max, count):
    cps = geometric_checkpoints(t_max, count)
    assert list(cps) == sorted(set(cps))  # sorted and unique
    assert cps[0] == 0 and cps[-1] == t_max and 1 in cps  # always 0, 1 and T_max
    assert all(0 <= c <= t_max for c in cps)
    # a_0 = 1 and a_(N-1) = T_max are checkpoints already: at most N_T + 1 points with 0
    assert len(cps) <= min(count + 1 if count >= 2 else 3, t_max + 1)
    if count <= 2:
        assert cps == tuple(sorted({0, 1, t_max}))
    # every geometric point is within 1/2 of a_j = T^(j/(N-1)): the schedule is the rounded curve
    if count >= 2:
        with localcontext() as ctx:
            ctx.prec = 60
            ideal = [(Decimal(j) * Decimal(t_max).ln() / (count - 1)).exp() for j in range(count)]
        inner = [c for c in cps if c not in (0,)]
        for a in ideal:
            assert min(abs(Decimal(c) - a) for c in inner) <= Decimal("0.5")
        # the rounded values are monotone in j, so the points are a_0 <= ... <= a_{N-1} rounded
        rounded = [round_half_up_power(t_max, j, count - 1) for j in range(count)]
        assert rounded == sorted(rounded) and rounded[0] == 1 and rounded[-1] == t_max


@PROPERTY
@given(st.integers(1, 20_000), st.integers(2, 30), st.integers(1, 4))
def test_refining_the_count_keeps_every_old_checkpoint(t_max, count, factor):
    # N_T -> k (N_T - 1) + 1 revisits the same real numbers a_j = T^(j / (N_T - 1)): a_(k j) of
    # the new schedule is a_j of the old one, rounded by the same exact rule
    finer = geometric_checkpoints(t_max, factor * (count - 1) + 1)
    assert set(geometric_checkpoints(t_max, count)) <= set(finer)


@PROPERTY
@given(st.integers(1, 10**9), st.integers(1, 40), st.data())
def test_round_half_up_power_is_the_nearest_integer(t_max, q, data):
    j = data.draw(st.integers(0, q))
    got = round_half_up_power(t_max, j, q)
    with localcontext() as ctx:
        ctx.prec = 100
        a = (Decimal(j) * Decimal(t_max).ln() / Decimal(q)).exp()
    assert abs(Decimal(got) - a) <= Decimal("0.5")
    assert got == int((a + Decimal("0.5")).to_integral_value(rounding="ROUND_FLOOR"))


def test_ties_round_upward_and_never_arise_for_integer_horizons():
    # the internal rule: floor(x + 1/2) with x = S^(1/q) / 2 for S = 2^q T^j. Ties need an odd
    # integer q-th root of S: x = 3/2, 5/2, ... (S = 9 and q = 2, S = 25 and q = 2, ...)
    for s, q, expected in [(9, 2, 2), (25, 2, 3), (1, 2, 1), (27, 3, 2), (8, 3, 1), (7**4, 4, 4)]:
        assert mf._half_up_root(s, q) == expected, (s, q)
    # python's round() and numpy's round to even would give 2 for 5/2 and 0 for 1/2
    assert round(2.5) == 2 and round(0.5) == 0 and mf._half_up_root(25, 2) == 3
    # for integer T >= 1 no a = T^(j/q) is a half-integer: the odd root of 2^q T^j would have to
    # divide into a power of two. Checked exhaustively on a range.
    for t_max, q in itertools.product(range(1, 80), range(1, 9)):
        for j in range(q + 1):
            s = 2**q * t_max**j
            r = 0
            while (r + 1) ** q <= s:
                r += 1
            assert not (r**q == s and r % 2 == 1), (t_max, j, q)


def test_geometric_checkpoints_validate_their_arguments():
    for bad in [(0, 3), (-5, 3), (10, 0), (10, 1001), (True, 3), (10, True), (10.0, 3), (10, 3.0)]:
        with pytest.raises(ValueError):
            geometric_checkpoints(*bad)  # type: ignore[arg-type]
    assert geometric_checkpoints(np.int64(64), np.int64(4)) == (0, 1, 4, 16, 64)
    with pytest.raises(ValueError, match="need T"):
        round_half_up_power(0, 0, 1)
    with pytest.raises(ValueError, match="need T"):
        round_half_up_power(5, 4, 3)


def test_explicit_checkpoint_lists_always_include_zero_one_and_horizon():
    listed = ExperimentManifest.from_json(
        mutated("depth.checkpoints", {"list": [64, 5, 5, 20, 128]})
    )
    assert listed.checkpoints(16, "0") == (0, 1, 5, 20, 64, 128)
    assert listed.checkpoints(24, "0.1") == (0, 1, 5, 20, 64, 128, 192)
    assert listed.to_dict()["depth"]["checkpoints"] == {"list": [5, 20, 64, 128]}  # sorted, unique
    assert listed.depth.checkpoints.resolve(7) == (0, 1, 5, 7, 20, 64, 128)  # as given, for any T


def test_a_checkpoint_list_must_fit_every_horizon():
    data = mutated("depth.layers", {"table": {"16": 64, "24": 200}})
    data["depth"]["checkpoints"] = {"list": [70]}
    with pytest.raises(ValueError, match="checkpoint 70 exceeds T_max = 64 for n = 16"):
        ExperimentManifest.from_json(data)
    data["depth"]["checkpoints"] = {"list": [64]}
    ExperimentManifest.from_json(data)


# ---------------------------------------------------------------------------
# the engine keywords, seeds, keys
# ---------------------------------------------------------------------------


def test_monitored_circuit_kwargs_are_the_v14_protocol(manifest):
    kwargs = manifest.monitored_circuit_kwargs(16, "0.1", "0.15")
    assert kwargs == {
        "num_system": 16,
        "circuits_per_graph": 8,
        "layers": 128,
        "measurement_rate": 0.15,
        "reference": "purification",
        "schedule": "random_edges",
        "gates_per_layer": 8,
        "random_orientation": False,
        "scramble_gates": 0,
        "scramble_matchings": 0,
        "scramble_layers": 0,
        "periodic": True,
        "checkpoints": "every_layer",
        "stop_when_purified": False,
    }
    assert "edges" not in kwargs and "probes" not in kwargs  # the runner adds both
    assert kwargs["measurement_rate"] == Token.parse("0.15").float64
    stopping = ExperimentManifest.from_json(mutated("dynamics.stop_when_purified", True))
    assert stopping.monitored_circuit_kwargs(24, 0, "0.2")["stop_when_purified"] is True
    assert stopping.monitored_circuit_kwargs(24, 0, "0.2")["layers"] == 192


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9, 16, 17, 25])
def test_the_kwargs_build_the_study_protocol_for_even_and_odd_sizes(n):
    data = mutated("conditions", [{"n": [n], "beta": ["0.3"], "p": ["0.2"]}])
    data["depth"] = {"layers": {"per_n": 2}, "checkpoints": {"count": 5}}
    data["observables"]["correlated_erasure"] = []
    data["observables"]["uniform_erasure"] = {"samples": 10, "exact_budget": 50}
    m = ExperimentManifest.from_json(data)
    edges = [ident.build_graph(n, "0.3", g, 17).edges for g in range(3)]
    spec = mipt.MonitoredCircuit(**m.monitored_circuit_kwargs(n, "0.3", "0.2"), edges=edges)
    assert spec.num_system == n and spec.gates_per_layer == n // 2  # floor(n / 2), odd n too
    assert spec.circuits_per_graph == 8 and spec.num_trajectories == 24
    assert spec.layers == 2 * n and spec.measurement_rate == 0.2
    assert spec.schedule == "random_edges" and spec.reference == "purification"
    assert spec.random_orientation is False and spec.periodic is True
    assert (spec.scramble_gates, spec.scramble_matchings, spec.scramble_layers) == (0, 0, 0)
    assert spec.checkpoints == "every_layer"
    assert np.array_equal(spec.checkpoint_layers, np.arange(2 * n + 1))
    assert spec.edge_weights is None and len(spec.gate_set) == 11520
    assert spec.stop_when_purified is False and spec.probes == ()


def test_seed_helpers_and_trajectory_keys(manifest):
    assert manifest.graph_seed(3) == (17, 3)
    assert manifest.run_seed("0.1") == 29 == manifest.run_seed("0.2")
    assert manifest.stream_index(3, 1) == 25
    key = manifest.trajectory_key(16, 0.1, "3/20", 3, 1)
    assert str(key) == "ws-pilot-001:n=16:beta=0.1:p=0.15:g=3:c=1:r=0"
    assert key.stream_index(manifest.stride) == manifest.stream_index(3, 1)
    independent = ExperimentManifest.from_json(mutated("seeds.gate_mode", "independent"))
    assert independent.run_seed("0.1") == (29, ident.rate_hash("0.1"))
    assert independent.run_seed("0.1") != independent.run_seed("0.15")
    with pytest.raises(ValueError, match="below the circuit index stride 8"):
        manifest.stream_index(0, 8)
    with pytest.raises(ValueError, match="is not a condition"):
        manifest.trajectory_key(16, 0.1, "0.3", 0, 0)


def test_warnings_flag_allowed_but_doubtful_choices(manifest):
    assert manifest.warnings == ()
    five = mutated("conditions", [{"n": [5, 16], "beta": ["0.1"], "p": ["0.1"]}])
    five["depth"]["layers"] = {"per_n": 8}
    texts = ExperimentManifest.from_json(five).warnings
    assert len(texts) == 1 and "K_5" in texts[0]
    stopping = ExperimentManifest.from_json(mutated("dynamics.stop_when_purified", True))
    assert any("recorded as absent" in w for w in stopping.warnings)
    big = mutated("storage.graphs_per_unit", 4000)
    assert any(
        "above max_unit_trajectories" in w for w in ExperimentManifest.from_json(big).warnings
    )


def test_the_manifest_is_immutable_hashable_and_compared_by_value(manifest):
    again = ExperimentManifest.from_json(EXAMPLE_TEXT)
    assert manifest == again and hash(manifest) == hash(again) and {manifest: 1}[again] == 1
    with pytest.raises(AttributeError):
        manifest.label = "confirmatory"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        manifest.trials.graphs = 3  # type: ignore[misc]
    assert manifest != ExperimentManifest.from_json(mutated("trials.graphs", 21))


# ---------------------------------------------------------------------------
# correlated-erasure settings
# ---------------------------------------------------------------------------


def test_correlated_erasure_settings_build_the_engine_probe(manifest):
    entry = manifest.observables.correlated_erasure[0]
    probe = entry.probe()
    direct = mipt.correlated_erasure(
        [0.0, 1.0, 2.0], None, 100, None, thin=20, geometry="interaction"
    )
    assert probe == direct and probe.name == "correlated_erasure"
    assert entry.probe(name="x", checkpoints=[0, 8]).name == "x"
    full = mutated(
        "observables.correlated_erasure",
        [
            dict(
                couplings=[0.5, -1],
                sizes="2:10:4",
                samples=7,
                thin=3,
                geometry="cycle",
                method="chain",
                burn_in=11,
                tempering=False,
                ladder=[0.25],
                max_step=0.5,
                name="ce",
            )
        ],
    )
    e = ExperimentManifest.from_json(full).observables.correlated_erasure[0]
    expected = mipt.correlated_erasure(
        [0.5, -1.0],
        "2:10:4",
        7,
        "ce",
        burn_in=11,
        thin=3,
        tempering=False,
        method="chain",
        ladder=[0.25],
        max_step=0.5,
        geometry="cycle",
    )
    assert e.probe() == expected
    assert e.expand_sizes(16) == (2, 6) and e.expand_sizes(24) == (2, 6)


def test_correlated_erasure_sizes_forms():
    base = mutated("observables.correlated_erasure", [{"couplings": [1], "sizes": None}])
    default = ExperimentManifest.from_json(base).observables.correlated_erasure[0]
    assert default.expand_sizes(16) == tuple(range(1, 16))
    assert default.expand_sizes(24) == tuple(range(1, 24))
    listed = ExperimentManifest.from_json(
        mutated("observables.correlated_erasure.0.sizes", [4, 2, 9], base=base)
    ).observables.correlated_erasure[0]
    assert listed.sizes == (4, 2, 9) and listed.expand_sizes(16) == (4, 2, 9)  # order as given
    big = mutated("conditions", [{"n": [40], "beta": ["0.1"], "p": ["0.1"]}], base=base)
    ranged = ExperimentManifest.from_json(
        mutated("observables.correlated_erasure.0.sizes", " 1 : 31 : 2 ", base=big)
    ).observables.correlated_erasure[0]
    assert ranged.sizes == "1 : 31 : 2"  # stripped, kept as written
    assert ranged.expand_sizes(40) == tuple(range(1, 31, 2))
    with pytest.raises(ValueError, match=r"sizes must lie in \[1, 15\] for n = 16"):
        ExperimentManifest.from_json(
            mutated("observables.correlated_erasure.0.sizes", "1:31:2", base=base)
        )


def test_several_correlated_erasure_probes_need_distinct_names():
    entries = [
        {"couplings": [0]},
        {"couplings": [1], "sizes": [2]},
        {"couplings": [2], "name": "mine"},
    ]
    m = ExperimentManifest.from_json(mutated("observables.correlated_erasure", entries))
    assert m.observables.probe_names() == ("correlated_erasure_0", "correlated_erasure_1", "mine")
    single = ExperimentManifest.from_json(mutated("observables.correlated_erasure", entries[:1]))
    assert single.observables.probe_names() == ("correlated_erasure",)
    clash = entries + [{"couplings": [3], "name": "mine"}]
    with pytest.raises(ValueError, match="the probe name 'mine' is used twice"):
        ExperimentManifest.from_json(mutated("observables.correlated_erasure", clash))
    clash2 = [{"couplings": [0]}, {"couplings": [1], "name": "correlated_erasure_0"}]
    with pytest.raises(ValueError, match="used twice"):
        ExperimentManifest.from_json(mutated("observables.correlated_erasure", clash2))


def test_uniform_erasure_can_be_switched_off_or_left_without_exact_counts():
    off = ExperimentManifest.from_json(mutated("observables.uniform_erasure", None))
    assert off.observables.uniform_erasure is None
    assert off.to_dict()["observables"]["uniform_erasure"] is None
    assert ExperimentManifest.from_json(off.to_json()) == off
    no_exact = ExperimentManifest.from_json(
        mutated("observables.uniform_erasure", {"samples": 50, "exact_budget": 0})
    )
    assert no_exact.observables.uniform_erasure == mf.UniformErasure(50, 0)


# ---------------------------------------------------------------------------
# revisions
# ---------------------------------------------------------------------------


def load(data) -> ExperimentManifest:
    return ExperimentManifest.from_json(data)


def with_changes(*changes) -> ExperimentManifest:
    data = copy.deepcopy(EXAMPLE)
    for path, value in changes:
        data = mutated(path, value, base=data)
    return load(data)


# the sparse checkpoints of the example at n = 16 (T_max 128) and n = 24 (T_max 192)
OLD_DEPTHS = sorted(set(geometric_checkpoints(128, 12)) | set(geometric_checkpoints(192, 12)))

ALLOWED = {
    "identical": [],
    "more rates": [("conditions.0.p", ["0.1", "0.15", "0.2", "0.25"])],
    "more sizes": [("conditions.0.n", [16, 24, 32])],
    "more betas": [("conditions.0.beta", ["0", "0.1", "0.3"])],
    "a second block": [
        ("conditions", EXAMPLE["conditions"] + [{"n": [40], "beta": ["0.5"], "p": ["0.3"]}])
    ],
    "more graphs": [("trials.graphs", 40)],
    "more circuits up to the stride": [("trials.circuits_per_graph", 8)],
    "raise T_max with an explicit list that keeps the old depths": [
        ("depth.layers", {"per_n": 16}),
        ("depth.checkpoints", {"list": OLD_DEPTHS + [250]}),
    ],
    # N_T -> 2 N_T - 1 refines the schedule: a_(2j) of the new one is a_j of the old one
    "more checkpoints": [("depth.checkpoints", {"count": 23})],
    "a second correlated entry": [
        (
            "observables.correlated_erasure",
            EXAMPLE["observables"]["correlated_erasure"] + [{"couplings": [3], "name": "second"}],
        )
    ],
    "storage": [("storage", {"graphs_per_unit": 4, "compression_level": 9})],
    "resources": [("resources", {"num_threads": 4, "max_unit_trajectories": 100})],
    "analysis": [("analysis", {"peak": {"z": 2.0}, "bootstrap": 10})],
}


@pytest.mark.parametrize("name", list(ALLOWED))
def test_allowed_revisions(manifest, name):
    new = with_changes(*ALLOWED[name])
    assert check_extension(manifest, new) is None


def test_adding_observables_and_a_larger_horizon_are_extensions():
    lean = load(
        mutated(
            "observables",
            {
                "tableau_snapshots": False,
                "contiguous_distance": False,
                "uniform_erasure": None,
                "correlated_erasure": [],
            },
        )
    )
    full = load(EXAMPLE)
    assert check_extension(lean, full) is None  # every observable switched on
    assert check_extension(full, lean) is not None
    # T_max itself is a checkpoint: raising it keeps the old horizon in the explicit list
    short = with_changes(("depth.layers", {"per_n": 4}), ("depth.checkpoints", {"list": [2, 8]}))
    forgot = with_changes(
        ("depth.layers", {"per_n": 8}), ("depth.checkpoints", {"list": [2, 8, 32]})
    )
    longer = with_changes(
        ("depth.layers", {"per_n": 8}), ("depth.checkpoints", {"list": [2, 8, 32, 64, 96]})
    )
    assert "lose the depths [64]" in check_extension(short, forgot)
    assert check_extension(forgot, longer) is None
    assert check_extension(short, longer) is None
    assert check_extension(longer, short) is not None


FORBIDDEN = {
    "experiment_id": (
        [("experiment_id", "ws-pilot-002")],
        "experiment_id changed from 'ws-pilot-001' to 'ws-pilot-002'",
    ),
    "label": ([("label", "confirmatory")], "label changed from 'exploratory' to 'confirmatory'"),
    "graph seed": (
        [("seeds.master_graph_seed", 18)],
        "seeds.master_graph_seed changed from 17 to 18: seeds are frozen",
    ),
    "circuit seed": (
        [("seeds.master_circuit_seed", 30)],
        "seeds.master_circuit_seed changed from 29 to 30",
    ),
    "gate mode": (
        [("seeds.gate_mode", "independent")],
        "seeds.gate_mode changed from 'shared' to 'independent'",
    ),
    "stride": ([("trials.circuit_index_stride", 16)], "circuit_index_stride changed from 8 to 16"),
    "stop_when_purified": (
        [("dynamics.stop_when_purified", True)],
        "dynamics.stop_when_purified changed from False to True",
    ),
    "fewer graphs": (
        [("trials.graphs", 10)],
        "trials.graphs changed from 20 to 10: a revision may add graphs",
    ),
    "fewer circuits": (
        [("trials.circuits_per_graph", 1)],
        "trials.circuits_per_graph changed from 2 to 1",
    ),
    "a rate removed": (
        [("conditions.0.p", ["0.1", "0.2"])],
        "condition (n = 16, beta = 0, p = 0.15) of the old manifest is missing",
    ),
    "a size removed": (
        [("conditions.0.n", [16])],
        "condition (n = 24, beta = 0, p = 0.1) of the old manifest is missing",
    ),
    "a beta removed": (
        [("conditions.0.beta", ["0"])],
        "condition (n = 16, beta = 0.1, p = 0.1) of the old manifest is missing",
    ),
    "lower T_max": (
        [("depth.layers", {"per_n": 4}), ("depth.checkpoints", {"list": [3]})],
        "T_max for n = 16 changed from 128 to 64: a revision may raise T_max",
    ),
    "geometric checkpoints move with T_max": ([("depth.layers", {"per_n": 16})], "lose the depths"),
    "fewer checkpoints": (
        [("depth.checkpoints", {"count": 8})],
        "checkpoints of (n = 16, beta = 0) lose the depths",
    ),
    "snapshots off": (
        [("observables.tableau_snapshots", False)],
        "observables.tableau_snapshots was recorded and cannot be switched off",
    ),
    "distance off": (
        [("observables.contiguous_distance", False)],
        "observables.contiguous_distance was recorded",
    ),
    "uniform changed": (
        [("observables.uniform_erasure.samples", 300)],
        "observables.uniform_erasure changed from",
    ),
    "uniform removed": (
        [("observables.uniform_erasure", None)],
        "observables.uniform_erasure changed from",
    ),
    "correlated changed": (
        [("observables.correlated_erasure.0.couplings", [0, 1, 3])],
        "observables.correlated_erasure[0] ('correlated_erasure') was recorded",
    ),
    "correlated removed": (
        [("observables.correlated_erasure", [])],
        "was recorded and is missing or changed",
    ),
}


@pytest.mark.parametrize("name", list(FORBIDDEN))
def test_forbidden_revisions_name_the_first_forbidden_change(manifest, name):
    changes, fragment = FORBIDDEN[name]
    new = with_changes(*changes)
    message = check_extension(manifest, new)
    assert message is not None and fragment in message, message


def test_the_first_forbidden_change_in_the_documented_order_is_reported(manifest):
    both = with_changes(("trials.graphs", 10), ("seeds.master_graph_seed", 18))
    assert "seeds.master_graph_seed" in check_extension(manifest, both)  # seeds before graphs
    several = with_changes(
        ("conditions.0.p", ["0.1"]),
        ("trials.circuits_per_graph", 1),
        ("observables.tableau_snapshots", False),
    )
    assert "trials.circuits_per_graph" in check_extension(
        manifest, several
    )  # trials before conditions


def test_check_extension_requires_manifests(manifest):
    with pytest.raises(TypeError, match="two ExperimentManifest"):
        check_extension(manifest, EXAMPLE)  # type: ignore[arg-type]


def test_a_chain_of_revisions_is_checked_pairwise(manifest):
    second = with_changes(("trials.graphs", 40), ("conditions.0.p", ["0.1", "0.15", "0.2", "0.25"]))
    third = with_changes(
        ("trials.graphs", 80), ("conditions.0.p", ["0.1", "0.15", "0.2", "0.25", "0.3"])
    )
    assert check_extension(manifest, second) is None and check_extension(second, third) is None
    assert check_extension(manifest, third) is None  # extensions compose
    assert (
        check_extension(third, manifest) is not None
        and check_extension(second, manifest) is not None
    )
