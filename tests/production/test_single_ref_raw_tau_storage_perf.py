"""Focused integrity and hot-path tests for raw-tau edge-bank storage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from studies.prl_production.graph_registry.build import build_collection, manifest_payload
from studies.prl_production.graph_registry.spec import smoke_spec
from studies.prl_production.single_ref.raw_tau import providers, storage
from studies.prl_production.single_ref.raw_tau.io import write_deterministic_npz
from studies.prl_production.single_ref.raw_tau.providers import (
    GridGraphProvider,
    WattsStrogatzRegistryProvider,
    builtin_graph_factory,
    load_edge_bank,
    prepare_edge_bank,
)
from studies.prl_production.sweep_spec import (
    GraphCollectionGridSpec,
    GraphParameterGrid,
    ProbabilityGrid,
    ScientificEnvironmentContract,
    SingleReferenceProtocolSpec,
    SingleReferenceSweepSpec,
)


def _cycle_edges(n: int, _parameters, _seed: int):
    return [(index, (index + 1) % n) for index in range(n)]


def _case():
    graphs = GraphCollectionGridSpec(
        name="raw_tau_storage_test_graphs",
        graph_family="cycle",
        generator_name="tests.cycle_edges",
        generator_version="v1",
        sizes=(4,),
        parameter_grid=GraphParameterGrid(),
        graphs_per_cell=2,
        master_seed=123,
    )
    provider = GridGraphProvider(graphs, factory=_cycle_edges)
    protocol = SingleReferenceProtocolSpec(
        n_circuits=3,
        q_scramble=0,
        q_max=1,
        p_grid=ProbabilityGrid("1", "1", "1"),
        master_seed=456,
    )
    sweep = SingleReferenceSweepSpec(
        name="raw_tau_storage_test",
        graph_collection_sha256=graphs.specification_sha256,
        source_fingerprint_sha256="a" * 64,
        environment_contract=ScientificEnvironmentContract("3.test", "2.test", "0.test"),
        protocol=protocol,
    )
    return provider, sweep, tuple(sweep.work_units(graphs))[0]


def test_builtin_factories_reject_lossy_integer_casts_and_invalid_seeds() -> None:
    watts_strogatz = builtin_graph_factory("watts_strogatz")
    with pytest.raises(TypeError, match="k must be an integer"):
        watts_strogatz(8, {"k": 2.5, "beta": 0.2}, 1)
    with pytest.raises(TypeError, match="k must be an integer"):
        watts_strogatz(8, {"k": True, "beta": 0.2}, 1)
    with pytest.raises(TypeError, match="graph seed must be an integer"):
        watts_strogatz(8, {"k": 2, "beta": 0.2}, 1.5)
    with pytest.raises(TypeError, match="graph seed must be an integer"):
        watts_strogatz(8, {"k": 2, "beta": 0.2}, False)
    with pytest.raises(ValueError, match="nonnegative"):
        watts_strogatz(8, {"k": 2, "beta": 0.2}, -1)
    with pytest.raises(ValueError, match="1 <= k < n/2"):
        watts_strogatz(8, {"k": 4, "beta": 0.2}, 1)


def test_registry_provider_requires_the_sealed_ensemble_family(tmp_path: Path) -> None:
    spec = smoke_spec()
    database = tmp_path / "graph_registry.sqlite3"
    summary = build_collection(spec, database)
    payload = manifest_payload(spec, summary)
    payload["database"] = str(database.resolve())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    provider = WattsStrogatzRegistryProvider(manifest)
    assert provider.collection_id == spec.collection_id
    assert len(provider.cells()) == len(spec.sizes) * len(spec.beta_keys)
    first_cell = provider.cells()[0]
    sealed_seeds = provider.graph_seeds(first_cell)

    wrong_environment = json.loads(json.dumps(payload))
    wrong_environment["environment"]["numpy"] += ".different"
    wrong_manifest = tmp_path / "wrong_environment.json"
    wrong_manifest.write_text(json.dumps(wrong_environment), encoding="utf-8")
    with pytest.raises(ValueError, match="recorded graph numpy"):
        WattsStrogatzRegistryProvider(wrong_manifest)

    wrong_k = json.loads(json.dumps(payload))
    wrong_k["graph_k"] += 1
    wrong_k_manifest = tmp_path / "wrong_k.json"
    wrong_k_manifest.write_text(json.dumps(wrong_k), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed ensemble specification"):
        WattsStrogatzRegistryProvider(wrong_k_manifest)

    wrong_beta = json.loads(json.dumps(payload))
    wrong_beta["betas"][0] = 1e-10
    wrong_beta_manifest = tmp_path / "wrong_beta.json"
    wrong_beta_manifest.write_text(json.dumps(wrong_beta), encoding="utf-8")
    with pytest.raises(ValueError, match="beta_key / 1e9 exactly"):
        WattsStrogatzRegistryProvider(wrong_beta_manifest)

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE graphs SET graph_seed = graph_seed + 1
            WHERE graph_id = (SELECT MIN(graph_id) FROM graphs)
            """
        )
    assert provider.graph_seeds(first_cell) == sealed_seeds

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE ensembles SET graph_family = ? WHERE ensemble_key = ?",
            ("watts_strogatz", spec.collection_id),
        )
    with pytest.raises(ValueError, match="watts_strogatz_rewired_circulant"):
        WattsStrogatzRegistryProvider(manifest)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_builtin_factories_reject_invalid_probabilities(value: object) -> None:
    factory = builtin_graph_factory("watts_strogatz")
    error = TypeError if value is True else ValueError
    with pytest.raises(error):
        factory(8, {"k": 2, "beta": value}, 1)


def test_receipt_fast_path_skips_graph_recanonicalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _sweep, _work = _case()
    cell = provider.cells()[0]
    bank_path = Path(prepare_edge_bank(tmp_path, provider, cell))
    receipt_path = bank_path.with_name(f"{bank_path.name}.validated.npz")
    assert receipt_path.is_file()
    bank_before = bank_path.read_bytes()
    receipt_before = receipt_path.read_bytes()
    assert Path(prepare_edge_bank(tmp_path, provider, cell)) == bank_path
    assert bank_path.read_bytes() == bank_before
    assert receipt_path.read_bytes() == receipt_before

    def unexpected_revalidation(_n, _edges):
        raise AssertionError("worker load repeated coordinator graph validation")

    monkeypatch.setattr(providers, "_canonical_edges", unexpected_revalidation)
    bank = load_edge_bank(bank_path, cell)
    assert bank.artifact_sha256 == providers.file_sha256(bank_path)
    assert not bank.graph_seed.flags.writeable
    assert not bank.edge_offsets.flags.writeable
    assert not bank.edges.flags.writeable

    with pytest.raises(AssertionError, match="repeated coordinator"):
        load_edge_bank(bank_path, cell, validation="full")
    assert receipt_path.read_bytes() == receipt_before


def test_receipt_fast_path_detects_edge_bank_mutation(tmp_path: Path) -> None:
    provider, _sweep, _work = _case()
    cell = provider.cells()[0]
    bank_path = Path(prepare_edge_bank(tmp_path, provider, cell))
    payload = bytearray(bank_path.read_bytes())
    payload[len(payload) // 2] ^= 1
    bank_path.write_bytes(payload)

    with pytest.raises(ValueError, match="immutable edge-bank digest"):
        load_edge_bank(bank_path, cell)


def test_full_edge_bank_validation_rejects_lossy_endpoint_narrowing(tmp_path: Path) -> None:
    provider, _sweep, _work = _case()
    cell = provider.cells()[0]
    bank_path = Path(prepare_edge_bank(tmp_path, provider, cell))
    with np.load(bank_path, allow_pickle=False) as data:
        arrays = {key: np.array(data[key], copy=True) for key in data.files}
    oversized = np.asarray(arrays["edges"], dtype=np.int64)
    oversized[0, 0] += 1 << 32
    arrays["edges"] = oversized
    write_deterministic_npz(bank_path, arrays)

    with pytest.raises(ValueError, match="edges must have i4"):
        load_edge_bank(bank_path, cell, validation="full")


def test_checkpoints_reuse_cached_edge_bank_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, sweep, work = _case()
    cell = provider.cells()[0]
    bank_path = Path(prepare_edge_bank(tmp_path, provider, cell)).resolve()
    original_file_sha256 = storage.file_sha256
    original_write_npz = storage.write_deterministic_npz
    hashed_paths: list[Path] = []
    published_paths: list[Path] = []

    def recording_file_sha256(path):
        resolved = Path(path).resolve()
        hashed_paths.append(resolved)
        if resolved == bank_path:
            raise AssertionError("checkpoint validation re-hashed the edge bank")
        return original_file_sha256(path)

    monkeypatch.setattr(storage, "file_sha256", recording_file_sha256)

    def recording_write_npz(path, arrays):
        published_paths.append(Path(path).resolve())
        return original_write_npz(path, arrays)

    monkeypatch.setattr(storage, "write_deterministic_npz", recording_write_npz)
    result = storage.run_work_unit(
        tmp_path,
        sweep,
        work,
        cell,
        bank_path,
        checkpoint_every=1,
    )

    assert result.is_complete
    assert bank_path not in hashed_paths
    assert hashed_paths == [Path(result.path).resolve()]
    assert published_paths == [Path(result.path).resolve()]
    assert not storage.checkpoint_journal_path(tmp_path, work).exists()
    bank = load_edge_bank(bank_path, cell)
    arrays = storage.load_raw_tau_arrays(result.path, sweep, work, cell, bank)
    assert arrays["edge_bank_sha256"].item() == bank.artifact_sha256
    assert arrays["environment_contract_sha256"].item() == (
        sweep.environment_contract.specification_sha256
    )
    assert json.loads(arrays["environment_contract_json"].item()) == (
        sweep.environment_contract.canonical_payload()
    )


def test_partial_run_leaves_only_a_resume_journal(tmp_path: Path) -> None:
    provider, sweep, work = _case()
    cell = provider.cells()[0]
    bank_path = prepare_edge_bank(tmp_path, provider, cell)

    partial = storage.run_work_unit(
        tmp_path,
        sweep,
        work,
        cell,
        bank_path,
        checkpoint_every=2,
        max_new_trajectories=3,
    )

    journal = storage.checkpoint_journal_path(tmp_path, work)
    assert partial.path == str(journal)
    assert partial.completed == 3
    assert journal.is_file()
    assert not storage.raw_tau_path(tmp_path, work).exists()
    with sqlite3.connect(journal) as connection:
        assert connection.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 3


def test_interrupted_batch_replays_only_committed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, sweep, work = _case()
    cell = provider.cells()[0]
    bank_path = prepare_edge_bank(tmp_path, provider, cell)
    original_simulate = storage.simulate_trajectory
    calls = 0

    def interrupt_third(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected interruption")
        return original_simulate(*args, **kwargs)

    monkeypatch.setattr(storage, "simulate_trajectory", interrupt_third)
    with pytest.raises(RuntimeError, match="injected interruption"):
        storage.run_work_unit(
            tmp_path,
            sweep,
            work,
            cell,
            bank_path,
            checkpoint_every=2,
        )

    journal = storage.checkpoint_journal_path(tmp_path, work)
    connection = sqlite3.connect(journal)
    try:
        assert connection.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 2
    finally:
        connection.close()
    monkeypatch.setattr(storage, "simulate_trajectory", original_simulate)
    resumed = storage.run_work_unit(
        tmp_path,
        sweep,
        work,
        cell,
        bank_path,
        checkpoint_every=2,
    )
    assert resumed.is_complete
    assert resumed.newly_completed == 4
    assert not journal.exists()


def test_corrupt_checkpoint_row_is_rejected(tmp_path: Path) -> None:
    provider, sweep, work = _case()
    cell = provider.cells()[0]
    bank_path = prepare_edge_bank(tmp_path, provider, cell)
    storage.run_work_unit(
        tmp_path,
        sweep,
        work,
        cell,
        bank_path,
        max_new_trajectories=1,
    )
    journal = storage.checkpoint_journal_path(tmp_path, work)
    with sqlite3.connect(journal) as connection:
        connection.execute("UPDATE results SET stop_layer = 2")

    with pytest.raises(ValueError, match="inconsistent observed journal row"):
        storage.run_work_unit(tmp_path, sweep, work, cell, bank_path)


def test_final_publish_is_recovered_before_journal_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, sweep, work = _case()
    cell = provider.cells()[0]
    bank_path = prepare_edge_bank(tmp_path, provider, cell)
    original_remove = storage._remove_journal

    def interrupt_cleanup(_path):
        raise RuntimeError("injected post-publish interruption")

    monkeypatch.setattr(storage, "_remove_journal", interrupt_cleanup)
    with pytest.raises(RuntimeError, match="post-publish interruption"):
        storage.run_work_unit(tmp_path, sweep, work, cell, bank_path)
    final_path = storage.raw_tau_path(tmp_path, work)
    journal = storage.checkpoint_journal_path(tmp_path, work)
    final_before = final_path.read_bytes()
    assert journal.is_file()

    monkeypatch.setattr(storage, "_remove_journal", original_remove)
    recovered = storage.run_work_unit(tmp_path, sweep, work, cell, bank_path)
    assert recovered.is_complete
    assert recovered.newly_completed == 0
    assert final_path.read_bytes() == final_before
    assert not journal.exists()
