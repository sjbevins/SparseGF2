"""The experiment manifest of the Watts–Strogatz study (``sparsegf2.study.manifest``).

Strict reading (every malformed manifest is refused with a message that names the field),
the canonical form and digest, the resolved conditions, the checkpoint rule of v1.4 §9
(hand-computed schedules, an independent high-precision implementation, property tests),
the keyword set that drives ``MonitoredCircuit``, and the revision rules of design §3 with the
ruling R-CK: a checkpoint list serves every size (entries above a horizon are dropped), a table
gives one list per size, any change of the rule and any larger horizon is an extension, and
``effective_checkpoints`` is the union of the schedules of all revisions.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import itertools
import json
import random
from decimal import Decimal, localcontext

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import _ws_reference as wsref
from sparsegf2 import mipt
from sparsegf2.study import identity as ident
from sparsegf2.study import manifest as mf
from sparsegf2.study.identity import Token
from sparsegf2.study.manifest import (
    ExperimentManifest,
    check_extension,
    effective_checkpoints,
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


def test_a_file_with_a_utf8_byte_order_mark_is_read(manifest, tmp_path):
    # Windows PowerShell 5.1 writes UTF-8 with a byte-order mark (Set-Content -Encoding utf8)
    path = tmp_path / "bom.json"
    path.write_bytes(b"\xef\xbb\xbf" + EXAMPLE_TEXT.encode("utf-8"))
    assert ExperimentManifest.from_json(path) == manifest
    assert ExperimentManifest.from_json(path).digest() == manifest.digest()
    # a file that is not a JSON object is named as such, not as a string to be read as a path
    other = tmp_path / "list.json"
    other.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match=r"list.json does not contain a JSON object") as error:
        ExperimentManifest.from_json(other)
    assert "pathlib.Path" not in str(error.value)
    latin = tmp_path / "latin.json"
    latin.write_bytes(b'{"experiment_id": "caf\xe9"}')
    with pytest.raises(ValueError, match="manifest: cannot read .*latin.json"):
        ExperimentManifest.from_json(latin)


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
    (
        "depth.layers",
        {"table": {"16": 64, "016": 70, "24": 96}},
        r"layers.table: the size n = 16 appears twice",
    ),
    (
        "depth.checkpoints",
        {},
        "manifest.depth.checkpoints: give exactly one of 'count', 'list' or 'table'",
    ),
    ("depth.checkpoints", {"count": 3, "list": [1]}, "give exactly one of"),
    ("depth.checkpoints", {"list": [1], "table": {"16": [1], "24": [1]}}, "give exactly one of"),
    ("depth.checkpoints", {"count": 0}, r"checkpoints.count: must be at least 1"),
    ("depth.checkpoints", {"count": 1001}, r"checkpoints.count: must be at most 1000"),
    ("depth.checkpoints", {"count": 2.5}, "checkpoints.count: must be an integer"),
    (
        "depth.checkpoints",
        {"list": []},
        r"checkpoints.list: give at least one entry \(for just 0, 1 and T_max use",
    ),
    ("depth.checkpoints", {"list": [-1, 3]}, r"checkpoints.list\[0\]: must be at least 0"),
    ("depth.checkpoints", {"list": [1, 2.0]}, r"checkpoints.list\[1\]: must be an integer"),
    ("depth.checkpoints", {"list": "1,2"}, "checkpoints.list: expected a list"),
    ("depth.checkpoints", {"table": {}}, r"checkpoints.table: expected an object"),
    ("depth.checkpoints", {"table": [1]}, r"checkpoints.table: expected an object"),
    ("depth.checkpoints", {"table": {"16": [1, 4]}}, r"checkpoints.table: no entry for n = \[24\]"),
    (
        "depth.checkpoints",
        {"table": {"16": [1], "24": [1], "99": [1]}},
        r"checkpoints.table: n = \[99\] is not in any condition",
    ),
    ("depth.checkpoints", {"table": {"x": [1]}}, "keys are sizes n as strings"),
    ("depth.checkpoints", {"table": {16: [1], "24": [1]}}, "keys are sizes n as strings"),
    (
        "depth.checkpoints",
        {"table": {"16": [1], "016": [2], "24": [1]}},
        r"checkpoints.table: the size n = 16 appears twice",
    ),
    (
        "depth.checkpoints",
        {"table": {"16": [129], "24": [1]}},
        r"checkpoints.table\[16\]: checkpoint 129 exceeds T_max = 128 for n = 16",
    ),
    (
        "depth.checkpoints",
        {"table": {"16": [1], "24": [193]}},
        r"checkpoints.table\[24\]: checkpoint 193 exceeds T_max = 192 for n = 24",
    ),
    (
        "depth.checkpoints",
        {"table": {"16": [-1], "24": [1]}},
        r"checkpoints.table\[16\]\[0\]: must be at least 0",
    ),
    (
        "depth.checkpoints",
        {"table": {"16": [1.5], "24": [1]}},
        r"checkpoints.table\[16\]\[0\]: must be an integer",
    ),
    (
        "depth.checkpoints",
        {"table": {"16": 5, "24": [1]}},
        r"checkpoints.table\[16\]: expected a list",
    ),
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
    ("seeds.master_circuit_seed", -1, r"master_circuit_seed: must be at least 0"),
    ("seeds.master_circuit_seed", 2**32, r"master_circuit_seed: must be at most 4294967295"),
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
    (
        "observables.correlated_erasure.0.couplings",
        [-1],
        r"couplings\[0\]: must be at least 0, got -1: v1.4 section 5.7 defines .* for J >= 0",
    ),
    (
        "observables.correlated_erasure.0.couplings",
        [0, 2, -0.5],
        r"correlated_erasure\[0\].couplings\[2\]: must be at least 0, got -0.5",
    ),
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
    ("observables.correlated_erasure.0.sizes", "5:5", r"sizes: the range '5:5' is empty"),
    ("observables.correlated_erasure.0.sizes", "9:3", r"sizes: the range '9:3' is empty"),
    ("observables.correlated_erasure.0.sizes", "1:9:0", "sizes: a range step must be positive"),
    ("observables.correlated_erasure.0.sizes", "1:5:2:1", r"sizes: expected null, a list"),
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
    *[
        (
            "observables.correlated_erasure.0.name",
            reserved,
            rf"correlated_erasure\[0\].name: '{reserved}' is the name of a probe the runner adds",
        )
        for reserved in mf.RESERVED_PROBE_NAMES
    ],
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
    twin = "0.1000000000000000055511151231257827"  # the exact value of the double nearest 0.1
    near = mutated("conditions", [{"n": [16], "beta": ["0.1", twin], "p": ["0.1"]}])
    with pytest.raises(
        ValueError, match="two tokens of one float64 .* one rewiring probability under two names"
    ):
        ExperimentManifest.from_json(near)
    rates = mutated("conditions", [{"n": [16], "beta": ["0"], "p": ["0.1", twin]}])
    with pytest.raises(ValueError, match="two tokens of one float64 .* one rate under two names"):
        ExperimentManifest.from_json(rates)
    # the reason is the shared float, not shared seeds: under ruling R-SEED the seeds hash the
    # token, so the two betas would even build different graphs from different seeds
    assert Token.parse(twin).float64 == Token.parse("0.1").float64
    a, b = ident.build_graph(16, "0.1", 0, 17), ident.build_graph(16, twin, 0, 17)
    assert a.seed != b.seed and not np.array_equal(a.edges, b.edges)


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
        (lambda m: m.run_seed(16, "0", "2"), "must lie in"),
        (lambda m: m.run_seed(16, "0", "0.3"), r"p = 0.3\) is not a condition"),
        (lambda m: m.run_seed(20, "0", "0.1"), r"\(n = 20, beta = 0, p = 0.1\) is not a condition"),
        (lambda m: m.graph_seed(20, "0", 0), r"\(n = 20, beta = 0\) is not a condition"),
        (lambda m: m.graph_seed(16, "0.2", 0), r"n = 16, beta = 0.2\) is not a condition"),
        (lambda m: m.graph_seed(16, "0", -1), "g must be at least 0"),
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
    # the rule is a function of the horizon and the size: entries above the horizon are dropped
    assert listed.depth.checkpoints.resolve(7, 16) == (0, 1, 5, 7)
    assert listed.depth.checkpoints.resolve(100, 16) == (0, 1, 5, 20, 64, 100)


def test_a_checkpoint_list_serves_every_size_and_drops_what_is_above_its_horizon():
    # ruling R-CK: one list for every size; the entries above a size's T_max are dropped for that
    # size (not an error), and 0, 1 and T_max are always included
    data = mutated("depth.layers", {"table": {"16": 64, "24": 200}})
    data["depth"]["checkpoints"] = {"list": [70, 5, 5, 20, 64, 130, 400]}
    m = ExperimentManifest.from_json(data)
    assert m.checkpoints(16, "0") == (0, 1, 5, 20, 64)  # 70, 130 and 400 dropped; 64 is T_max
    assert m.checkpoints(24, "0.1") == (0, 1, 5, 20, 64, 70, 130, 200)  # 400 dropped, 200 added
    assert m.checkpoints(16, "0.1") == m.checkpoints(16, "0")  # the same for every rate and beta
    assert m.to_dict()["depth"]["checkpoints"] == {"list": [5, 20, 64, 70, 130, 400]}
    assert ExperimentManifest.from_json(m.to_json()) == m
    # 400 is above every horizon: it has no effect, which is flagged but not refused
    assert any("[400]" in w and "no effect" in w for w in m.warnings)
    data["depth"]["checkpoints"] = {"list": [70, 5, 20, 64, 130]}
    assert not any("checkpoints.list" in w for w in ExperimentManifest.from_json(data).warnings)


def test_a_checkpoint_table_gives_one_list_per_size():
    data = mutated("depth.checkpoints", {"table": {"24": [3, 100], "16": [64, 5, 5]}})
    m = ExperimentManifest.from_json(data)
    assert m.checkpoints(16, "0") == (0, 1, 5, 64, 128)  # T_max(16) = 128
    assert m.checkpoints(24, "0.1") == (0, 1, 3, 100, 192)  # T_max(24) = 192
    assert m.to_dict()["depth"]["checkpoints"] == {"table": {"16": [5, 64], "24": [3, 100]}}
    assert ExperimentManifest.from_json(m.to_json()) == m and m.digest() != load(EXAMPLE).digest()
    # an empty list for a size means just 0, 1 and T_max; an entry equal to T_max is allowed
    plain = ExperimentManifest.from_json(
        mutated("depth.checkpoints", {"table": {"16": [], "24": [192]}})
    )
    assert plain.checkpoints(16, "0") == (0, 1, 128) and plain.checkpoints(24, "0") == (0, 1, 192)
    # the table is a function of the size alone
    with pytest.raises(ValueError, match="no entry for n = 20"):
        m.depth.checkpoints.resolve(100, 20)
    with pytest.raises(ValueError, match=r"table\[16\]: checkpoint 64 exceeds T_max = 50"):
        m.depth.checkpoints.resolve(50, 16)


@PROPERTY
@given(
    st.lists(st.integers(0, 300), min_size=1, max_size=10),
    st.integers(1, 200),
    st.sampled_from([5, 8, 16, 24]),
)
def test_the_list_form_is_the_list_below_the_horizon_with_zero_one_and_the_horizon(
    depths, t_max, n
):
    rule = mf.CheckpointRule.from_dict({"list": depths}, "x")
    expected = tuple(sorted({0, 1, t_max} | {d for d in depths if d <= t_max}))
    assert rule.resolve(t_max, n) == expected
    assert rule.resolve(t_max, n) == rule.resolve(t_max, n + 1)  # the same list for every size


@PROPERTY
@given(st.data())
def test_the_table_form_is_the_size_entry_with_zero_one_and_the_horizon(data):
    sizes = data.draw(
        st.lists(st.sampled_from([5, 6, 8, 12, 16]), min_size=1, max_size=4, unique=True)
    )
    t_max = {n: data.draw(st.integers(1, 60)) for n in sizes}
    table = {str(n): data.draw(st.lists(st.integers(0, t_max[n]), max_size=6)) for n in sizes}
    manifest = ExperimentManifest.from_json(
        {
            "experiment_id": "t",
            "conditions": [{"n": sizes, "beta": ["0.1"], "p": ["0.2"]}],
            "depth": {
                "layers": {"table": {str(n): t_max[n] for n in sizes}},
                "checkpoints": {"table": table},
            },
            "trials": {"graphs": 2, "circuits_per_graph": 1},
            "seeds": {"master_graph_seed": 1, "master_circuit_seed": 2},
            "observables": {"uniform_erasure": None},
        }
    )
    for n in sizes:
        assert manifest.checkpoints(n, "0.1") == tuple(sorted({0, 1, t_max[n], *table[str(n)]}))
    assert ExperimentManifest.from_json(manifest.to_json()) == manifest


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
    # graph seeds name (n, beta, g); the example's master graph seed is 17
    assert manifest.graph_seed(16, "0.1", 3) == wsref.graph_seed(17, 16, "0.1", 3)
    assert manifest.graph_seed(24, 0, 0) == wsref.graph_seed(17, 24, "0", 0)
    assert manifest.graph_seed(16, "0.1", 3) == (0x57534731, 17, 16, 2007713733, 3507754741, 3)
    # shared mode: one run seed per (n, beta), the same at every rate; the master circuit seed is 29
    shared = manifest.run_seed(16, "0.1", "0.1")
    assert shared == wsref.run_seed(29, 16, "0.1") == (0x57534331, 29, 16, 2007713733, 3507754741)
    assert all(manifest.run_seed(16, "0.1", p) == shared for p in ("0.1", "0.15", 0.2, "1/5"))
    assert (
        manifest.run_seed(24, "0.1", "0.1") != shared and manifest.run_seed(16, 0, "0.1") != shared
    )
    assert manifest.stream_index(3, 1) == 25
    key = manifest.trajectory_key(16, 0.1, "3/20", 3, 1)
    assert str(key) == "ws-pilot-001:n=16:beta=0.1:p=0.15:g=3:c=1:r=0"
    assert key.stream_index(manifest.stride) == manifest.stream_index(3, 1)
    # independent mode appends the words of the rate: one run seed per rate
    independent = ExperimentManifest.from_json(mutated("seeds.gate_mode", "independent"))
    assert independent.run_seed(16, "0.1", "0.1") == wsref.run_seed(29, 16, "0.1", "0.1")
    assert independent.run_seed(16, "0.1", "0.1") != independent.run_seed(16, "0.1", "0.15")
    assert independent.run_seed(16, "0.1", "0.15")[:5] == shared and len(shared) == 5
    assert independent.graph_seed(16, "0.1", 3) == manifest.graph_seed(
        16, "0.1", 3
    )  # not the mode's
    with pytest.raises(ValueError, match="below the circuit index stride 8"):
        manifest.stream_index(0, 8)
    with pytest.raises(ValueError, match="is not a condition"):
        manifest.trajectory_key(16, 0.1, "0.3", 0, 0)


def test_seeds_depend_on_the_condition_and_only_the_run_seed_on_the_gate_mode(manifest):
    # every (n, beta) has its own graph collection and its own run seed (ruling R-SEED); the rate
    # enters the run seed in independent mode only, and never the graph seed
    independent = ExperimentManifest.from_json(mutated("seeds.gate_mode", "independent"))
    pairs = [(c.n, c.beta) for c in manifest.conditions()]
    assert len(pairs) == 4
    graph = {manifest.graph_seed(n, b, g) for n, b in pairs for g in range(3)}
    assert len(graph) == 12  # four conditions x three graphs, all different
    shared = {manifest.run_seed(n, b, p) for n, b in pairs for p in manifest.rates(n, b)}
    assert len(shared) == 4  # one per condition
    per_rate = {independent.run_seed(n, b, p) for n, b in pairs for p in independent.rates(n, b)}
    assert len(per_rate) == 12  # one per (condition, rate)
    assert not graph & shared and not graph & per_rate  # disjoint domains


def test_equal_master_seeds_are_accepted_because_the_seed_domains_are_disjoint():
    # the first draft refused them (graph 0 was SeedSequence([s, 0]) == SeedSequence(s)); the tags
    # of ruling R-SEED separate the domains whatever the master seeds are
    equal = ExperimentManifest.from_json(mutated("seeds.master_circuit_seed", 17))
    assert equal.seeds.master_graph_seed == equal.seeds.master_circuit_seed == 17
    assert ExperimentManifest.from_json(equal.to_json()) == equal
    assert equal.digest() != ExperimentManifest.from_json(EXAMPLE_TEXT).digest()
    graph = [equal.graph_seed(c.n, c.beta, g) for c in equal.conditions() for g in range(4)]
    runs = [equal.run_seed(c.n, c.beta, c.rates[0]) for c in equal.conditions()]
    assert not set(graph) & set(runs)
    plain = tuple(np.random.SeedSequence(17).generate_state(4).tolist())
    states = [tuple(np.random.SeedSequence(s).generate_state(4).tolist()) for s in graph + runs]
    assert plain not in states and len(set(states)) == len(states)
    # and the conditions of one manifest give distinct seeds (the same graph label, other n or beta)
    assert len(set(graph)) == len(graph)


def test_two_tokens_with_one_seed_hash_are_refused(monkeypatch):
    # The words of beta name the condition in every seed, so two different beta tokens with equal
    # 64-bit hashes (probability 2**-64 per pair) are refused; likewise two rates, in both gate
    # modes: the words of the rate enter the run seed in independent mode and the region seeds of
    # the sampled observables (study/observables.py::region_seed) in both.
    real = mf.token_words
    same = {Token.parse("0.3"): real("0.1")}  # beta = 0.3 gets the words of beta = 0.1
    monkeypatch.setattr(mf, "token_words", lambda t: same.get(Token.parse(t), real(t)))
    with pytest.raises(ValueError, match="beta values 0.1 and 0.3 have the same 64-bit seed hash"):
        ExperimentManifest.from_json(mutated("conditions.0.beta", ["0.1", "0.3"]))
    ExperimentManifest.from_json(EXAMPLE_TEXT)  # the example has no such pair
    same[Token.parse("0.2")] = real("0.15")  # rate 0.2 gets the words of rate 0.15
    for mode in ("shared", "independent"):
        with pytest.raises(ValueError, match="p values 0.15 and 0.2 have the same 64-bit seed"):
            ExperimentManifest.from_json(mutated("seeds.gate_mode", mode))
    # one rate token used at several conditions is one token, not a collision
    ExperimentManifest.from_json(mutated("conditions.0.p", ["0.1", "0.15"]))
    # the real hash has no collision among the tokens of a fine grid (checked, not proved)
    monkeypatch.undo()
    grid = [f"{k}/{d}" for d in (997, 1000) for k in range(d + 1)]
    assert len({mf.token_words(t) for t in grid}) == len({Token.parse(t) for t in grid})


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


def test_direct_construction_and_replace_obey_the_rules_of_reading(manifest):
    # Every section is read back through its from_dict when a manifest is built, so a manifest
    # made in code (dataclasses.replace, direct construction) is refused exactly where reading
    # its canonical JSON would refuse it, and an accepted one reads back as itself.
    replace = dataclasses.replace
    Trials, Seeds = mf.Trials, mf.Seeds
    cases = [
        (
            {"trials": replace(manifest.trials, circuits_per_graph=9)},
            r"manifest.trials.circuit_index_stride: must be at least circuits_per_graph \(9\)",
        ),
        ({"trials": Trials(0, 4, 8, 1)}, r"manifest.trials.graphs: must be at least 1, got 0"),
        (
            {"trials": Trials(20, 2, 8, 3)},
            r"manifest.trials.measurement_replicas: must be 1, got 3: the native engine",
        ),
        ({"seeds": Seeds(2**40, 29)}, r"manifest.seeds.master_graph_seed: must be at most"),
        ({"seeds": Seeds(17, 29, "coupled")}, r"manifest.seeds.gate_mode: must be one of"),
        (
            {"blocks": (mf.ConditionBlock((4,), (Token.parse("0.1"),), (Token.parse("0.1"),)),)},
            r"manifest.conditions\[0\].n\[0\]: must be at least 5, got 4",
        ),
        (
            {"blocks": (mf.ConditionBlock((16,), ("0.1",), (Token.parse("0.1"),)),)},  # type: ignore[arg-type]
            r"manifest.conditions\[0\]: a field of this ConditionBlock has the wrong type",
        ),
        ({"blocks": ()}, r"manifest.conditions: give a non-empty tuple"),
        (
            {"depth": mf.Depth(mf.Layers("per_n", 0), manifest.depth.checkpoints)},
            r"manifest.depth.layers.per_n: must be at least 1, got 0",
        ),
        (
            {"depth": mf.Depth(mf.Layers("per_n", 8, ((16, 3),)), manifest.depth.checkpoints)},
            r"manifest.depth: .* is not in the canonical form that reading it gives",
        ),
        (
            {"depth": mf.Depth(manifest.depth.layers, mf.CheckpointRule("count", count=0))},
            r"manifest.depth.checkpoints.count: must be at least 1, got 0",
        ),
        (
            {"depth": mf.Depth(manifest.depth.layers, mf.CheckpointRule("list", depths=(9, 3)))},
            r"manifest.depth: .* is not in the canonical form",  # an unsorted list
        ),
        (
            {"observables": mf.Observables(uniform_erasure=mf.UniformErasure(0, 5000))},
            r"manifest.observables.uniform_erasure.samples: must be at least 1",
        ),
        (
            {"observables": mf.Observables(correlated_erasure=(mf.CorrelatedErasure((-1.0,)),))},
            r"manifest.observables.correlated_erasure\[0\].couplings\[0\]: must be at least 0",
        ),
        (
            {
                "observables": mf.Observables(
                    correlated_erasure=(mf.CorrelatedErasure((1.0,), name="code_dimension"),)
                )
            },
            r"name: 'code_dimension' is the name of a probe the runner adds",
        ),
        ({"storage": mf.Storage(8, 10)}, r"manifest.storage.compression_level: must be at most 9"),
        ({"resources": mf.Resources(0)}, r"manifest.resources.num_threads: must be at least 1"),
        (
            {"analysis": mf.Analysis(mf.Peak(z=-1.0))},
            r"manifest.analysis.peak.z: must be at least 0",
        ),
        ({"trials": {"graphs": 20}}, r"manifest.trials: expected a Trials, got dict"),
        (
            {
                "protocols": (
                    mf.ProtocolSettings("F"),
                    mf.ProtocolSettings("P", mf.CheckpointRule("count", count=0)),
                )
            },
            r"manifest.protocols.P.checkpoints.count: must be at least 1, got 0",
        ),
        ({"protocols": ("F",)}, r"manifest.protocols\[0\]: expected a ProtocolSettings, got str"),
        ({"label": "pilot"}, r"manifest.label: must be one of"),
    ]
    for changes, fragment in cases:
        with pytest.raises(ValueError, match=fragment):
            replace(manifest, **changes)
    # what replace may change is checked like a revision read from JSON, and reads back as itself
    more = replace(manifest, trials=replace(manifest.trials, graphs=40, circuits_per_graph=8))
    assert ExperimentManifest.from_json(more.to_json()) == more
    assert check_extension(manifest, more) is None and more.trajectories_per_condition == 320
    # direct construction of the whole manifest
    built = ExperimentManifest(
        "built",
        manifest.blocks,
        manifest.depth,
        manifest.trials,
        manifest.seeds,
        observables=mf.Observables(uniform_erasure=None),
    )
    assert ExperimentManifest.from_json(built.to_json()) == built


def test_the_conditions_are_resolved_once_per_manifest(monkeypatch, manifest):
    calls = []
    real = mf._resolve_conditions

    def counting(blocks):
        calls.append(1)
        return real(blocks)

    monkeypatch.setattr(mf, "_resolve_conditions", counting)
    m = ExperimentManifest.from_json(EXAMPLE_TEXT)
    assert len(calls) == 1  # the constructor resolves them, for its checks
    for g in range(50):
        m.trajectory_key(24, "0.1", "0.2", g, 1)
        m.graph_seed(16, "0", g)
        m.run_seed(24, "0", "0.15")
        m.layers(16, "0.1")
        m.checkpoints(24, "0.1")
        m.rates(16, "0")
        m.has_condition(24, "0.1")
        m.monitored_circuit_kwargs(16, "0.1", "0.1")
    assert check_extension(m, m) is None and effective_checkpoints([m, m], 16, "0.1")
    assert len(calls) == 1  # every later question is a look-up
    # the result is a fresh list each time, and the cache is not part of equality or hashing
    first = m.conditions()
    first.clear()
    assert len(m.conditions()) == 4
    assert m == manifest and hash(m) == hash(manifest)
    assert "_resolved_conditions" not in repr(m)


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
                couplings=[0.5, 1.5],
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
        [0.5, 1.5],
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
    assert e.probe() == expected  # the stored expansion builds the probe of the range string
    assert e.sizes == (2, 6) and e.expand_sizes(16) == (2, 6) and e.expand_sizes(24) == (2, 6)


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
    assert ranged.sizes == tuple(range(1, 31, 2))  # a range is stored as its expansion
    assert ranged.expand_sizes(40) == tuple(range(1, 31, 2))
    assert ranged.to_dict()["sizes"] == list(range(1, 31, 2))
    with pytest.raises(ValueError, match=r"sizes must lie in \[1, 15\] for n = 16"):
        ExperimentManifest.from_json(
            mutated("observables.correlated_erasure.0.sizes", "1:31:2", base=base)
        )


def test_equivalent_spellings_of_correlated_sizes_are_one_setting():
    # a range string does not depend on n, so every spelling of the same sizes is one canonical
    # form: one digest, one engine probe, and a respelling is not a change of a recorded setting
    big = mutated("conditions", [{"n": [40], "beta": ["0.1"], "p": ["0.1"]}])
    spellings = ["1:31:2", " 1 : 31 : 2 ", "01:31:2", "1:30:2", list(range(1, 31, 2))]
    manifests = [
        ExperimentManifest.from_json(
            mutated("observables.correlated_erasure.0.sizes", sizes, base=big)
        )
        for sizes in spellings
    ]
    assert len({m.digest() for m in manifests}) == 1
    assert len({m.observables.correlated_erasure[0] for m in manifests}) == 1
    probes = [m.observables.correlated_probes()[0] for m in manifests]
    assert all(probe == probes[0] for probe in probes)
    assert probes[0].sizes == tuple(range(1, 31, 2))
    assert all(check_extension(manifests[0], m) is None for m in manifests)
    # the order of a list is kept: it is the order of the stored values
    reordered = ExperimentManifest.from_json(
        mutated("observables.correlated_erasure.0.sizes", [3, 1, 5], base=big)
    )
    assert reordered.observables.correlated_erasure[0].sizes == (3, 1, 5)
    assert (
        reordered.digest()
        != ExperimentManifest.from_json(
            mutated("observables.correlated_erasure.0.sizes", "1:7:2", base=big)
        ).digest()
    )


def test_several_correlated_erasure_probes_need_distinct_names():
    entries = [
        {"couplings": [0]},
        {"couplings": [1], "sizes": [2]},
        {"couplings": [2], "name": "mine"},
    ]
    m = ExperimentManifest.from_json(mutated("observables.correlated_erasure", entries))
    # an unnamed entry is named by its position alone: the first keeps the engine's default name
    assert m.observables.probe_names() == ("correlated_erasure", "correlated_erasure_1", "mine")
    single = ExperimentManifest.from_json(mutated("observables.correlated_erasure", entries[:1]))
    assert single.observables.probe_names() == ("correlated_erasure",)
    assert [mf.Observables.default_probe_name(i) for i in range(3)] == [
        "correlated_erasure",
        "correlated_erasure_1",
        "correlated_erasure_2",
    ]
    clash = entries + [{"couplings": [3], "name": "mine"}]
    with pytest.raises(ValueError, match="the probe name 'mine' is used twice"):
        ExperimentManifest.from_json(mutated("observables.correlated_erasure", clash))
    clash2 = [{"couplings": [0]}, {"couplings": [1], "name": "correlated_erasure"}]
    with pytest.raises(ValueError, match="the probe name 'correlated_erasure' is used twice"):
        ExperimentManifest.from_json(mutated("observables.correlated_erasure", clash2))
    clash3 = [
        {"couplings": [0]},
        {"couplings": [1]},
        {"couplings": [2], "name": "correlated_erasure_1"},
    ]
    with pytest.raises(ValueError, match="'correlated_erasure_1' is used twice"):
        ExperimentManifest.from_json(mutated("observables.correlated_erasure", clash3))


def test_the_correlated_probes_carry_the_resolved_names_and_fit_beside_the_runners_probes():
    entries = [
        {"couplings": [0]},
        {"couplings": [1], "sizes": [2]},
        {"couplings": [2], "name": "x"},
    ]
    m = ExperimentManifest.from_json(mutated("observables.correlated_erasure", entries))
    probes = m.observables.correlated_probes(checkpoints=[0, 8])
    assert tuple(p.name for p in probes) == m.observables.probe_names()
    assert all(p.checkpoints == (0, 8) for p in probes)
    for probe, entry, name in zip(
        probes, m.observables.correlated_erasure, m.observables.probe_names(), strict=True
    ):
        assert probe == entry.probe(name=name, checkpoints=[0, 8])
    # entry.probe() alone falls back to the engine default: two unnamed entries would clash
    assert [e.probe().name for e in m.observables.correlated_erasure[:2]] == [
        "correlated_erasure",
        "correlated_erasure",
    ]
    # the runner's own probes keep the reserved names, so the engine accepts the whole set
    cps = list(m.checkpoints(16, "0.1"))
    fixed = [
        mipt.code_dimension(),
        mipt.tableau_snapshot("tableau_snapshot", checkpoints=cps),
        mipt.contiguous_distance(checkpoints=cps),
        mipt.erasure_first_failure(10, "erasure_first_failure", checkpoints=cps),
        mipt.uncorrectable_region_counts(
            name="uncorrectable_region_counts", budget=5000, checkpoints=cps
        ),
    ]
    assert {p.name for p in fixed} <= set(mf.RESERVED_PROBE_NAMES)
    assert mipt.code_snapshot("code_snapshot").name in mf.RESERVED_PROBE_NAMES
    edges = [ident.build_graph(16, "0.1", 0, 17).edges]
    spec = mipt.MonitoredCircuit(
        **m.monitored_circuit_kwargs(16, "0.1", "0.1"),
        edges=edges,
        probes=[*fixed, *m.observables.correlated_probes(checkpoints=cps)],
    )
    assert len({p.name for p in spec.probes}) == len(spec.probes) == 8
    # a correlated entry may not take one of those names (the engine would refuse the set)
    for reserved in mf.RESERVED_PROBE_NAMES:
        with pytest.raises(ValueError, match="is the name of a probe the runner adds"):
            ExperimentManifest.from_json(mutated("observables.correlated_erasure.0.name", reserved))


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
    # ruling R-CK: any change of the checkpoint rule, and a larger T_max under any of them
    "raise T_max under the geometric rule": [("depth.layers", {"per_n": 16})],
    "raise T_max and switch to a list": [
        ("depth.layers", {"per_n": 16}),
        ("depth.checkpoints", {"list": [3, 50, 400]}),
    ],
    "raise T_max for one size only": [
        ("depth.layers", {"table": {"16": 128, "24": 400}}),
    ],
    "fewer checkpoints": [("depth.checkpoints", {"count": 8})],
    "more checkpoints": [("depth.checkpoints", {"count": 23})],
    "a list instead of the geometric rule": [("depth.checkpoints", {"list": [3, 50]})],
    "a table instead of the geometric rule": [
        ("depth.checkpoints", {"table": {"16": [3], "24": [50]}})
    ],
    "a second correlated entry": [
        (
            "observables.correlated_erasure",
            EXAMPLE["observables"]["correlated_erasure"] + [{"couplings": [3], "name": "second"}],
        )
    ],
    # appending renames nothing: the recorded entry stays 'correlated_erasure'
    "a second unnamed correlated entry": [
        (
            "observables.correlated_erasure",
            EXAMPLE["observables"]["correlated_erasure"] + [{"couplings": [3]}],
        )
    ],
    "the recorded correlated entry named by its own name": [
        ("observables.correlated_erasure.0.name", "correlated_erasure")
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
    # ruling R-CK: a revision may raise T_max whatever it does with the checkpoint rule, because
    # the schedule used for work is the union of the schedules of all revisions
    short = with_changes(("depth.layers", {"per_n": 4}), ("depth.checkpoints", {"list": [2, 8]}))
    forgot = with_changes(
        ("depth.layers", {"per_n": 8}), ("depth.checkpoints", {"list": [2, 8, 32]})
    )
    longer = with_changes(
        ("depth.layers", {"per_n": 8}), ("depth.checkpoints", {"list": [2, 8, 32, 64, 96]})
    )
    assert check_extension(short, forgot) is None  # the old horizon 64 need not be listed again
    assert check_extension(forgot, longer) is None
    assert check_extension(short, longer) is None
    assert check_extension(longer, short) is not None  # lowering T_max is forbidden
    assert "T_max for n = 16 changed from 128 to 64" in check_extension(forgot, short)


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
    # the horizon is checked size by size: n = 16 keeps its 128, n = 24 falls from 192
    "lower T_max for one size only": (
        [("depth.layers", {"table": {"16": 128, "24": 100}})],
        "T_max for n = 24 changed from 192 to 100: a revision may raise T_max, not lower it",
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
        "observables.correlated_erasure[0] ('correlated_erasure') was recorded and the new "
        "manifest gives this name other settings",
    ),
    "correlated removed": (
        [("observables.correlated_erasure", [])],
        "was recorded and the new manifest has no probe of this name",
    ),
    # a new entry in front takes the position, and so the name, of the recorded one
    "correlated entry inserted in front": (
        [
            (
                "observables.correlated_erasure",
                [{"couplings": [3]}] + EXAMPLE["observables"]["correlated_erasure"],
            )
        ],
        "('correlated_erasure') was recorded and the new manifest names these settings "
        "'correlated_erasure_1'",
    ),
    "correlated entry renamed": (
        [("observables.correlated_erasure.0.name", "renamed")],
        "('correlated_erasure') was recorded and the new manifest names these settings 'renamed'",
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


def correlated(*entries) -> ExperimentManifest:
    return with_changes(("observables.correlated_erasure", list(entries)))


A_ENTRY = {"couplings": [0, 1], "sizes": [2, 4], "samples": 50}
B_ENTRY = {"couplings": [0, 2], "sizes": [2, 4], "samples": 80}


def test_a_recorded_correlated_probe_keeps_its_resolved_name():
    # The resolved name keys a probe's random stream (mipt hashes it) and its stored values
    # (v1.4 section 7.3: preserve probe names and versioned settings). Comparing settings alone
    # would accept every case below that renames a recorded probe.
    one = correlated(A_ENTRY)
    two = correlated(A_ENTRY, B_ENTRY)
    swapped = correlated(B_ENTRY, A_ENTRY)
    assert one.observables.probe_names() == ("correlated_erasure",)
    assert two.observables.probe_names() == ("correlated_erasure", "correlated_erasure_1")
    # appending renames nothing, in any number
    assert check_extension(one, two) is None
    assert check_extension(one, correlated(A_ENTRY, B_ENTRY, {"couplings": [5]})) is None
    assert check_extension(two, correlated(A_ENTRY, B_ENTRY, {"couplings": [5]})) is None
    # naming an entry by the name it already has is not a change
    assert check_extension(one, correlated({**A_ENTRY, "name": "correlated_erasure"})) is None
    assert check_extension(correlated({**A_ENTRY, "name": "correlated_erasure"}), one) is None
    # a swap of two unnamed entries swaps their names between different settings
    message = check_extension(two, swapped)
    assert message is not None
    assert "observables.correlated_erasure[0] ('correlated_erasure') was recorded" in message
    assert "names these settings 'correlated_erasure_1'" in message
    # inserting in front, or naming a recorded entry otherwise, renames it
    assert "names these settings 'correlated_erasure_1'" in check_extension(one, swapped)
    assert "names these settings 'mine'" in check_extension(
        one, correlated({**A_ENTRY, "name": "mine"})
    )
    assert "names these settings 'correlated_erasure'" in check_extension(
        correlated({**A_ENTRY, "name": "mine"}), one
    )
    # the same name with other settings, and a removed entry
    assert "gives this name other settings" in check_extension(one, correlated(B_ENTRY))
    assert "('correlated_erasure_1') was recorded and the new manifest has no probe" in (
        check_extension(two, one)
    )
    # moving a recorded entry is possible by naming it (an unnamed first entry would take the
    # default name, so the entry in front is named too): 'correlated_erasure' stays A's
    moved = correlated({**B_ENTRY, "name": "b"}, {**A_ENTRY, "name": "correlated_erasure"})
    assert moved.observables.probe_names() == ("b", "correlated_erasure")
    assert check_extension(one, moved) is None
    assert (
        "('correlated_erasure_1') was recorded and the new manifest names these settings 'b'"
        in (check_extension(two, moved))
    )
    with pytest.raises(ValueError, match="the probe name 'correlated_erasure' is used twice"):
        correlated(B_ENTRY, {**A_ENTRY, "name": "correlated_erasure"})


def test_a_confirmatory_manifest_freezes_its_conditions_and_analysis():
    # v1.4 section 9: the rate grid P(n, beta) is frozen before confirmatory runs; section 5.8:
    # the peak rule is predeclared. An exploratory manifest may gain conditions (ALLOWED above).
    base = [("label", "confirmatory")]
    confirmed = with_changes(*base)
    for changes, fragment in [
        (
            [("conditions.0.p", ["0.1", "0.15", "0.2", "0.25"])],
            "condition (n = 16, beta = 0, p = 0.25) is new, but the manifest is labeled "
            "'confirmatory': v1.4 section 9 freezes the rate grid",
        ),
        ([("conditions.0.n", [16, 24, 32])], "condition (n = 32, beta = 0, p = 0.1) is new"),
        ([("conditions.0.beta", ["0", "0.1", "0.3"])], "(n = 16, beta = 0.3, p = 0.1) is new"),
        (
            [("analysis.peak.z", 2.0)],
            "analysis changed from {'peak': {'rule': 'plateau', 'z': 1.0",
        ),
        ([("analysis.peak.sensitivity_z", [0, 1])], "a confirmatory manifest freezes its analysis"),
        ([("analysis.bootstrap", 10)], "a confirmatory manifest freezes its analysis rules"),
    ]:
        message = check_extension(confirmed, with_changes(*base, *changes))
        assert message is not None and fragment in message, message
        # the same change is an extension of the exploratory manifest
        assert check_extension(with_changes(), with_changes(*changes)) is None
    # what does not touch the grid or the analysis rules stays allowed
    for changes in [
        [("trials.graphs", 40)],
        [("trials.circuits_per_graph", 8)],
        [("depth.layers", {"per_n": 16})],
        [("depth.checkpoints", {"count": 23})],
        [("storage", {"graphs_per_unit": 4, "compression_level": 9})],
        [("resources", {"num_threads": 4, "max_unit_trajectories": 100})],
        [
            (
                "observables.correlated_erasure",
                EXAMPLE["observables"]["correlated_erasure"] + [{"couplings": [3]}],
            )
        ],
    ]:
        assert check_extension(confirmed, with_changes(*base, *changes)) is None, changes
    # the freeze is the last check (item 8 of document 02, section 3.11): a lower horizon is
    # reported first
    both = with_changes(
        *base,
        ("conditions.0.p", ["0.1", "0.15", "0.2", "0.25"]),
        ("depth.layers", {"per_n": 4}),
    )
    assert "T_max for n = 16 changed from 128 to 64" in check_extension(confirmed, both)
    # and a valid confirmatory chain has its effective schedule
    longer = with_changes(*base, ("depth.layers", {"per_n": 16}))
    assert set(confirmed.checkpoints(16, "0.1")) <= set(
        effective_checkpoints([confirmed, longer], 16, "0.1")
    )
    with pytest.raises(ValueError, match="revision 1 does not extend revision 0: condition"):
        effective_checkpoints(
            [confirmed, with_changes(*base, ("conditions.0.p", ["0.1", "0.15", "0.2", "0.3"]))],
            16,
            "0.1",
        )


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


# ---------------------------------------------------------------------------
# the effective schedule across revisions (ruling R-CK)
# ---------------------------------------------------------------------------


def one_size(
    t_max: int,
    rule: dict,
    *,
    sizes: tuple[int, ...] = (8,),
    betas: tuple[str, ...] = ("0.1",),
    **overrides,
) -> ExperimentManifest:
    """A manifest with the horizon ``t_max`` for every size and the checkpoint ``rule``."""
    data = {
        "experiment_id": "chain",
        "conditions": [{"n": list(sizes), "beta": list(betas), "p": ["0.2"]}],
        "depth": {"layers": {"fixed": t_max}, "checkpoints": rule},
        "trials": {"graphs": 2, "circuits_per_graph": 1},
        "seeds": {"master_graph_seed": 1, "master_circuit_seed": 2},
        "observables": {"uniform_erasure": None},
        **overrides,
    }
    return ExperimentManifest.from_json(data)


def test_the_effective_schedule_is_the_union_of_the_schedules_of_the_revisions():
    first = one_size(16, {"count": 5})  # 16^(j/4) = 2^j
    second = one_size(64, {"count": 4})  # 64^(j/3) = 4^j
    third = one_size(100, {"list": [3, 50, 200]})  # 200 is above T_max = 100: dropped
    assert first.checkpoints(8, "0.1") == (0, 1, 2, 4, 8, 16)
    assert second.checkpoints(8, "0.1") == (0, 1, 4, 16, 64)
    assert third.checkpoints(8, "0.1") == (0, 1, 3, 50, 100)
    # raising T_max under the geometric rule, and any other change of the rule, is an extension
    assert check_extension(first, second) is None and check_extension(second, third) is None
    assert check_extension(first, third) is None
    assert effective_checkpoints([first], 8, "0.1") == (0, 1, 2, 4, 8, 16)  # one revision: its own
    assert effective_checkpoints([first, second], 8, "0.1") == (0, 1, 2, 4, 8, 16, 64)
    everything = (0, 1, 2, 3, 4, 8, 16, 50, 64, 100)
    assert effective_checkpoints([first, second, third], 8, "0.1") == everything
    # the depth 8 of the first revision is not a depth of the second's own schedule, yet it stays
    assert 8 not in second.checkpoints(8, "0.1") and 8 in effective_checkpoints(
        [first, second], 8, "0.1"
    )
    # a token, a string or a float name the same beta
    assert effective_checkpoints([first, second], 8, 0.1) == effective_checkpoints(
        [first, second], 8, "1/10"
    )


def test_a_size_or_beta_absent_from_an_earlier_revision_contributes_nothing_from_it():
    first = one_size(16, {"count": 5}, sizes=(8,), betas=("0.1",))
    second = one_size(64, {"list": [3, 10]}, sizes=(8, 12), betas=("0.1", "0.3"))
    assert check_extension(first, second) is None
    assert effective_checkpoints([first, second], 12, "0.1") == (0, 1, 3, 10, 64)  # n new
    assert effective_checkpoints([first, second], 8, "0.3") == (0, 1, 3, 10, 64)  # beta new
    assert effective_checkpoints([first, second], 8, "0.1") == (0, 1, 2, 3, 4, 8, 10, 16, 64)
    with pytest.raises(ValueError, match=r"\(n = 9, beta = 0.1\) is not a condition"):
        effective_checkpoints([first, second], 9, "0.1")
    with pytest.raises(ValueError, match="revision 2 does not extend revision 1: condition"):
        dropped_beta = one_size(64, {"count": 2}, sizes=(8, 12))  # beta 0.3 would disappear
        effective_checkpoints([first, second, dropped_beta], 8, "0.1")


def test_effective_checkpoints_refuses_what_is_not_an_extension_chain():
    first, second = one_size(16, {"count": 5}), one_size(64, {"count": 4})
    with pytest.raises(
        ValueError,
        match=r"revision 1 does not extend revision 0: T_max for n = 8 changed from 64 to 16",
    ):
        effective_checkpoints([second, first], 8, "0.1")  # the horizon decreases
    reseeded = one_size(64, {"count": 4}, seeds={"master_graph_seed": 1, "master_circuit_seed": 3})
    with pytest.raises(
        ValueError, match="revision 1 does not extend revision 0: seeds.master_circ"
    ):
        effective_checkpoints([first, reseeded], 8, "0.1")
    with pytest.raises(ValueError, match="revision 1 does not extend revision 0: condition"):
        effective_checkpoints([first, one_size(64, {"count": 4}, sizes=(9,))], 9, "0.1")
    third = one_size(32, {"count": 3})
    with pytest.raises(ValueError, match="revision 2 does not extend revision 1"):
        effective_checkpoints([first, second, third], 8, "0.1")  # the pair that breaks is named
    with pytest.raises(ValueError, match="at least one revision"):
        effective_checkpoints([], 8, "0.1")
    with pytest.raises(TypeError, match="revision 1 is not an ExperimentManifest"):
        effective_checkpoints([first, EXAMPLE], 8, "0.1")  # type: ignore[list-item]
    # equal horizons, in either order, or one revision twice, are valid chains
    again = one_size(16, {"list": [3, 5]})
    both = (0, 1, 2, 3, 4, 5, 8, 16)
    assert effective_checkpoints([first, again], 8, "0.1") == both
    assert effective_checkpoints([again, first], 8, "0.1") == both
    assert effective_checkpoints([first, first], 8, "0.1") == first.checkpoints(8, "0.1")


RULE_KINDS = [
    {"count": 3},
    {"count": 12},
    {"list": [7, 90]},
    {"table": {"16": [5], "24": [60]}},
]


RULE_IDS = ["count3", "count12", "list", "table"]


@pytest.mark.parametrize("old_rule", RULE_KINDS, ids=RULE_IDS)
@pytest.mark.parametrize("new_rule", RULE_KINDS, ids=RULE_IDS)
def test_any_change_of_the_checkpoint_rule_is_an_extension(old_rule, new_rule):
    old = with_changes(("depth.checkpoints", old_rule))
    new = with_changes(("depth.checkpoints", new_rule))
    assert check_extension(old, new) is None and check_extension(new, old) is None
    higher = with_changes(("depth.layers", {"per_n": 16}), ("depth.checkpoints", new_rule))
    lower = with_changes(("depth.layers", {"per_n": 4}), ("depth.checkpoints", new_rule))
    assert check_extension(old, higher) is None  # a larger T_max, whatever the rule becomes
    assert "T_max for n = 16 changed from 128 to 64" in check_extension(old, lower)
    assert check_extension(higher, old) is not None  # and a smaller one is refused


POOL = [5, 6, 7, 8, 9, 10, 12, 16]


def chain_manifest(sizes, t_max, rule) -> ExperimentManifest:
    """A manifest of one block with a horizon per size (a layer table) and the rule."""
    return ExperimentManifest.from_json(
        {
            "experiment_id": "chain",
            "conditions": [{"n": sorted(sizes), "beta": ["0.1"], "p": ["0.2"]}],
            "depth": {
                "layers": {"table": {str(n): t_max[n] for n in sizes}},
                "checkpoints": rule,
            },
            "trials": {"graphs": 2, "circuits_per_graph": 1},
            "seeds": {"master_graph_seed": 1, "master_circuit_seed": 2},
            "observables": {"uniform_erasure": None},
        }
    )


@st.composite
def checkpoint_rules(draw, sizes, t_max):
    kind = draw(st.sampled_from(["count", "list", "table"]))
    if kind == "count":
        return {"count": draw(st.integers(1, 12))}
    if kind == "list":  # entries may exceed some or all horizons: they are dropped
        return {"list": draw(st.lists(st.integers(0, 300), min_size=1, max_size=8))}
    return {
        "table": {
            str(n): draw(st.lists(st.integers(0, t_max[n]), max_size=6)) for n in sorted(sizes)
        }
    }


@st.composite
def revision_drafts(draw):
    """Valid chains of 1 to 4 revisions, with what each was drawn from: ``(manifest, sizes,
    horizons)``. Horizons never fall (each size's rises by 0 to 30, independently of the other
    sizes), sizes may be added, any rule."""
    sizes = draw(st.lists(st.sampled_from(POOL), min_size=1, max_size=3, unique=True))
    t_max = {n: draw(st.integers(1, 40)) for n in sizes}
    chain = []
    for _ in range(draw(st.integers(1, 4))):
        rule = draw(checkpoint_rules(sizes, t_max))
        chain.append((chain_manifest(sizes, t_max, rule), tuple(sizes), dict(t_max)))
        t_max = {n: t + draw(st.integers(0, 30)) for n, t in t_max.items()}
        spare = [n for n in POOL if n not in sizes]
        if spare and draw(st.booleans()):
            added = draw(st.sampled_from(spare))
            sizes = sorted([*sizes, added])
            t_max[added] = draw(st.integers(1, 40))
    return chain


def revision_chains():
    """The manifests of :func:`revision_drafts`."""
    return revision_drafts().map(lambda drafts: [m for m, _, _ in drafts])


@PROPERTY
@given(revision_chains())
def test_effective_checkpoints_properties(chain):
    earlier: dict[int, tuple[int, ...]] = {}
    for k in range(len(chain)):
        prefix = chain[: k + 1]
        latest = prefix[-1]
        for n in sorted({c.n for c in latest.conditions()}):
            effective = effective_checkpoints(prefix, n, "0.1")
            horizon = latest.layers(n, "0.1")
            own = [m.checkpoints(n, "0.1") for m in prefix if m.has_condition(n, "0.1")]
            assert list(effective) == sorted(set(effective))  # sorted, unique
            assert effective == tuple(sorted(set().union(*own)))  # the union of the own schedules
            assert all(0 <= d <= horizon for d in effective)  # never above the latest T_max
            assert {0, 1, horizon} <= set(effective)
            assert all(set(schedule) <= set(effective) for schedule in own)
            if k == 0:  # one revision: its own schedule
                assert effective == chain[0].checkpoints(n, "0.1")
            if n in earlier:  # a superset of the effective schedule of every earlier prefix
                assert set(earlier[n]) <= set(effective)
            earlier[n] = effective
    for i in range(len(chain)):  # every pair is an extension: extensions compose
        for j in range(i + 1, len(chain)):
            assert check_extension(chain[i], chain[j]) is None


@PROPERTY
@given(revision_drafts())
def test_a_chain_in_the_wrong_order_is_refused_exactly_when_it_is_not_an_extension(drafts):
    # The oracle is computed from what the revisions were drawn from, not by check_extension:
    # read backward, the second revision extends the first exactly when the first has every
    # size of the second and no size's horizon falls. Every other field is the same in both.
    if len(drafts) < 2:
        return
    (first, first_sizes, first_t), (second, second_sizes, second_t) = drafts[0], drafts[1]
    extends = set(second_sizes) <= set(first_sizes) and all(
        first_t[n] >= second_t[n] for n in second_sizes
    )
    message = check_extension(second, first)
    assert (message is None) == extends, message
    if not extends:
        lowered = [n for n in second_sizes if n in first_sizes and first_t[n] < second_t[n]]
        missing = [n for n in second_sizes if n not in first_sizes]
        expected = (  # check_extension reports the smallest size, conditions before horizons
            f"condition (n = {min(missing)},"
            if missing
            else f"T_max for n = {min(lowered)} changed"
        )
        assert message is not None and expected in message, message
    for n in first_sizes:  # every size of the last revision, not only the first
        if extends:
            own = [m.checkpoints(n, "0.1") for m in (second, first) if m.has_condition(n, "0.1")]
            assert effective_checkpoints([second, first], n, "0.1") == tuple(
                sorted(set().union(*own))
            )
        else:
            with pytest.raises(ValueError, match="revision 1 does not extend revision 0"):
                effective_checkpoints([second, first], n, "0.1")
