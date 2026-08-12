"""Configuration, planning, and coordinator safety tests for raw tau production."""

from __future__ import annotations

import json
import os
import sqlite3
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path

import numba
import numpy as np
import pytest
from studies.prl_production.single_ref.benchmark import THREAD_LIMIT_VARIABLES
from studies.prl_production.single_ref.raw_tau import config as config_module
from studies.prl_production.single_ref.raw_tau import run as runner
from studies.prl_production.single_ref.raw_tau.catalog import catalog_path
from studies.prl_production.single_ref.raw_tau.config import (
    SOURCE_FINGERPRINT_PATHS,
    ResolvedRawTauConfig,
    RuntimeConfig,
    load_config,
    source_fingerprint_sha256,
)
from studies.prl_production.single_ref.raw_tau.io import file_sha256, write_deterministic_npz
from studies.prl_production.single_ref.raw_tau.storage import (
    WorkUnitProgress,
    logical_tau_digest,
    raw_tau_path,
)
from studies.prl_production.sweep_spec import ScientificEnvironmentContract


@pytest.fixture(scope="module", autouse=True)
def _restore_process_thread_state():
    """Keep runner tests from leaking process-wide thread limits."""
    original_environment = {name: os.environ.get(name) for name in THREAD_LIMIT_VARIABLES}
    original_numba_threads = numba.get_num_threads()
    try:
        yield
    finally:
        numba.set_num_threads(original_numba_threads)
        for name, value in original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _payload(data_root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "raw_tau_runner_test",
        "graph_source": {
            "kind": "cartesian_builtin",
            "name": "runner_test_graphs",
            "graph_family": "cycle",
            "generator_name": "sparsegf2.circuits.graphs.cycle_graph",
            "generator_version": "test-v1",
            "sizes": [4, 6],
            "parameter_axes": {},
            "graphs_per_cell": 2,
            "master_seed": 11,
        },
        "protocol": {
            "n_circuits": 3,
            "q_scramble": 1,
            "q_max": 2,
            "p_min": "0",
            "p_max": "0.2",
            "delta_p": "0.1",
            "master_seed": 22,
        },
        "runtime": {
            "data_root": str(data_root),
            "workers": 1,
            "checkpoint_every": 2,
            "max_in_flight": 2,
        },
    }


def _write_config(tmp_path: Path, payload: dict[str, object] | None = None) -> Path:
    content = _payload(tmp_path / "raw") if payload is None else payload
    path = tmp_path / "raw_tau.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_load_config_and_plan_have_exact_counts(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    plan = runner.make_plan(config)

    assert config.sweep.protocol.p_grid.canonical_values == ("0", "0.1", "0.2")
    assert plan.cells == 2
    assert plan.p_values == 3
    assert plan.work_units == 6
    assert plan.graphs == 4
    assert plan.circuits_per_graph_p == 3
    assert plan.trajectories == 36
    assert plan.max_layers == 360
    assert plan.max_dynamic_gates == 936
    assert plan.scramble_gates == 180
    assert plan.measurement_trials == 1_872
    assert plan.raw_tau_bytes == 144


def test_config_is_strict_and_runtime_is_execution_only(tmp_path: Path) -> None:
    payload = _payload(tmp_path / "raw")
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        load_config(_write_config(tmp_path, payload))

    payload = _payload(tmp_path / "raw")
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    runtime["workers"] = 3
    runtime["max_in_flight"] = 2
    with pytest.raises(ValueError, match="at least workers"):
        load_config(_write_config(tmp_path, payload))

    data_file = tmp_path / "not_a_directory"
    data_file.write_text("x", encoding="ascii")
    payload = _payload(data_file)
    with pytest.raises(ValueError, match="not a directory"):
        load_config(_write_config(tmp_path, payload))


def test_plan_rejects_invalid_builtin_parameter_cells_before_writes(tmp_path: Path) -> None:
    missing_axis = _payload(tmp_path / "raw_missing")
    graph_source = missing_axis["graph_source"]
    assert isinstance(graph_source, dict)
    graph_source.update(
        {
            "graph_family": "watts_strogatz",
            "generator_name": "sparsegf2.circuits.graphs.watts_strogatz",
            "sizes": [8],
            "parameter_axes": {"k": [2]},
        }
    )
    with pytest.raises(ValueError, match="parameters must be"):
        load_config(_write_config(tmp_path, missing_axis))

    invalid_later_size = _payload(tmp_path / "raw_lattice")
    graph_source = invalid_later_size["graph_source"]
    assert isinstance(graph_source, dict)
    graph_source.update(
        {
            "graph_family": "lattice_2d",
            "generator_name": "sparsegf2.circuits.graphs.lattice_2d",
            "sizes": [4, 6],
            "parameter_axes": {"rows": [2], "cols": [2]},
        }
    )
    with pytest.raises(ValueError, match="equal n=6"):
        load_config(_write_config(tmp_path, invalid_later_size))


def test_source_fingerprint_covers_runner_and_low_level_kernels() -> None:
    assert "studies/prl_production/single_ref/raw_tau/catalog.py" in SOURCE_FINGERPRINT_PATHS
    assert "studies/prl_production/single_ref/raw_tau/run.py" in SOURCE_FINGERPRINT_PATHS
    assert "studies/prl_production/single_ref/shared_io.py" in SOURCE_FINGERPRINT_PATHS
    assert "studies/prl_production/graph_registry/collection.py" in SOURCE_FINGERPRINT_PATHS
    assert "studies/prl_production/graph_registry/database.py" in SOURCE_FINGERPRINT_PATHS
    assert "src/sparsegf2/__init__.py" in SOURCE_FINGERPRINT_PATHS
    assert "src/sparsegf2/core/__init__.py" in SOURCE_FINGERPRINT_PATHS
    assert "src/sparsegf2/core/_protocol.py" in SOURCE_FINGERPRINT_PATHS
    assert "src/sparsegf2/core/linalg_gf2.py" in SOURCE_FINGERPRINT_PATHS
    assert "src/sparsegf2/core/numba_kernels.py" in SOURCE_FINGERPRINT_PATHS
    assert "src/sparsegf2/core/symplectic.py" in SOURCE_FINGERPRINT_PATHS
    first = source_fingerprint_sha256()
    assert first == source_fingerprint_sha256()
    assert len(first) == 64


def test_dependency_contract_changes_the_experiment_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path)
    original = load_config(config_path)
    changed_environment = replace(
        original.sweep.environment_contract,
        numpy=original.sweep.environment_contract.numpy + ".different",
    )
    monkeypatch.setattr(
        config_module,
        "current_scientific_environment_contract",
        lambda: changed_environment,
    )
    changed = load_config(config_path)

    assert changed.sweep.environment_contract == changed_environment
    assert changed.sweep.specification_sha256 != original.sweep.specification_sha256
    assert changed.sweep.experiment_id != original.sweep.experiment_id


def test_plan_cli_is_default_and_write_free(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = tmp_path / "must_not_be_created"
    config_path = _write_config(tmp_path, _payload(data_root))

    assert runner.main(["--config", str(config_path)]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["plan"]["trajectories"] == 36
    assert printed["exact_p_values"] == ["0", "0.1", "0.2"]
    assert printed["protocol"]["q_max"] == 2
    assert (
        printed["protocol_sha256"]
        == config_module.load_config(config_path).sweep.protocol.specification_sha256
    )
    assert printed["graph_source_summary"]["graph_family"] == "cycle"
    assert printed["graph_source_summary"]["sizes"] == [4, 6]
    assert printed["graph_source_summary"]["graphs_per_cell"] == 2
    assert printed["graph_source_summary"]["parameter_values"] == {}
    assert len(printed["graph_source_summary"]["generator_contract_sha256"]) == 64
    assert printed["environment_contract"]["bit_generator"] == "PCG64"
    assert len(printed["environment_contract_sha256"]) == 64
    assert len(printed["source_fingerprint_sha256"]) == 64
    assert not data_root.exists()


def test_run_requires_exact_confirmation_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    dispatched: list[ResolvedRawTauConfig] = []
    monkeypatch.setattr(runner, "run_sweep", lambda resolved: dispatched.append(resolved))

    with pytest.raises(SystemExit, match="confirm-experiment-id"):
        runner.main(["--config", str(config_path), "--run"])
    with pytest.raises(SystemExit, match="confirm-experiment-id"):
        runner.main(
            [
                "--config",
                str(config_path),
                "--run",
                "--confirm-experiment-id",
                "wrong",
            ]
        )
    assert not dispatched

    assert (
        runner.main(
            [
                "--config",
                str(config_path),
                "--run",
                "--workers",
                "2",
                "--confirm-experiment-id",
                config.sweep.experiment_id,
            ]
        )
        == 0
    )
    assert len(dispatched) == 1
    assert dispatched[0].runtime.workers == 2
    assert dispatched[0].runtime.max_in_flight == 4
    assert dispatched[0].sweep.specification_sha256 == config.sweep.specification_sha256


class _ImmediateExecutor:
    """ProcessPoolExecutor test double that preserves Future/wait behavior."""

    starts: list[tuple[int, int]] = []

    def __init__(self, *, max_workers: int, mp_context, initializer, initargs) -> None:
        del mp_context, initializer
        assert len(initargs) == 1
        assert isinstance(initargs[0], ScientificEnvironmentContract)
        self._processes: dict[object, object] = {}
        self._shutdown = False
        self.starts.append((max_workers, int(os.environ["NUMBA_NUM_THREADS"])))

    def submit(self, function, *args) -> Future[WorkUnitProgress]:
        future: Future[WorkUnitProgress] = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:  # pragma: no cover - used by failure-path tests
            future.set_exception(exc)
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        assert wait
        del cancel_futures
        self._shutdown = True


def _completed_task(data_root, _sweep, work, _cell, _bank, _checkpoint):
    total = work.raw_shape[0] * work.raw_shape[1]
    path = raw_tau_path(data_root, work)
    shape = work.raw_shape
    arrays = {
        "cell_sha256": np.str_(work.cell.cell_sha256),
        "p_decimal": np.str_(work.p_decimal),
        "graph_index": np.arange(shape[0], dtype=np.int32),
        "graph_seed": np.arange(shape[0], dtype=np.int64),
        "circuit_index": np.arange(shape[1], dtype=np.int32),
        "tau_p": np.ones(shape, dtype=np.int32),
        "stop_layer": np.ones(shape, dtype=np.int32),
        "event_observed": np.ones(shape, dtype=np.uint8),
        "complete": np.ones(shape, dtype=np.uint8),
        "reference_system_qubit": np.zeros(shape, dtype=np.int32),
    }
    write_deterministic_npz(path, arrays)
    return WorkUnitProgress(
        path=str(path),
        work_sha256=work.work_sha256,
        completed=total,
        total=total,
        events=total,
        censored=0,
        newly_completed=0,
        elapsed_s=0.0,
        artifact_sha256=file_sha256(path),
        logical_result_sha256=logical_tau_digest(arrays),
    )


def test_resume_accepts_runtime_changes_and_audits_each_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(_write_config(tmp_path))
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(runner, "prepare_edge_bank", lambda *_args: "edge-bank.npz")
    monkeypatch.setattr(runner, "_run_task", _completed_task)
    _ImmediateExecutor.starts.clear()

    first_plan = runner.run_sweep(config)
    run_root = (
        config.runtime.data_root / "single_ref" / "raw_tau" / "runs" / config.sweep.experiment_id
    )
    manifest_path = run_root / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    manifest = json.loads(manifest_before)
    assert manifest["schema_version"] == runner.SCIENTIFIC_MANIFEST_SCHEMA_VERSION
    assert "runtime" not in manifest
    assert "environment" not in manifest
    assert manifest["environment_contract"] == config.sweep.environment_contract.canonical_payload()
    assert manifest["environment_contract_sha256"] == (
        config.sweep.environment_contract.specification_sha256
    )
    assert manifest["graph_source_summary"] == runner.graph_source_summary(config)
    with sqlite3.connect(catalog_path(config.runtime.data_root)) as connection:
        complete_rows, logical_rows = connection.execute(
            """
            SELECT COUNT(*), COUNT(logical_result_sha256)
            FROM work_units WHERE status = 'complete'
            """
        ).fetchone()
    assert (complete_rows, logical_rows) == (first_plan.work_units, first_plan.work_units)

    changed = replace(
        config,
        runtime=RuntimeConfig(
            data_root=config.runtime.data_root,
            workers=2,
            checkpoint_every=7,
            max_in_flight=4,
        ),
    )
    assert runner.run_sweep(changed) == first_plan
    assert manifest_path.read_bytes() == manifest_before

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_root / "runtime_history").glob("*.json"))
    ]
    assert len(records) == 4
    assert {record["event"] for record in records} == {"started", "complete"}
    assert len({record["invocation_id"] for record in records}) == 2
    runtime_settings = {
        (
            record["runtime"]["workers"],
            record["runtime"]["checkpoint_every"],
            record["runtime"]["max_in_flight"],
        )
        for record in records
    }
    assert runtime_settings == {(1, 2, 2), (2, 7, 4)}
    for record in records:
        assert record["recorded_at_utc"].endswith("Z")
        assert record["environment"]["pid"] == os.getpid()
        for version_key in ("python", "numpy", "numba", "sparsegf2"):
            assert record["environment"][version_key]
    assert _ImmediateExecutor.starts == [(1, 1), (2, 1)]
    assert json.loads((run_root / "status.json").read_text(encoding="utf-8"))["state"] == (
        "complete"
    )


def test_spawned_runner_is_bit_identical_with_one_or_two_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in THREAD_LIMIT_VARIABLES:
        monkeypatch.setenv(name, os.environ.get(name, "restore-after-test"))

    configs: list[ResolvedRawTauConfig] = []
    for label, workers in (("one", 1), ("two", 2)):
        case_root = tmp_path / label
        case_root.mkdir()
        payload = _payload(case_root / "data")
        graph_source = payload["graph_source"]
        protocol = payload["protocol"]
        runtime = payload["runtime"]
        assert isinstance(graph_source, dict)
        assert isinstance(protocol, dict)
        assert isinstance(runtime, dict)
        graph_source["sizes"] = [4]
        graph_source["graphs_per_cell"] = 1
        protocol.update(
            {
                "n_circuits": 1,
                "q_scramble": 0,
                "q_max": 1,
                "p_min": "0",
                "p_max": "1",
                "delta_p": "1",
            }
        )
        runtime.update(
            {
                "workers": workers,
                "checkpoint_every": 1,
                "max_in_flight": workers,
            }
        )
        config = load_config(_write_config(case_root, payload))
        runner.run_sweep(config)
        configs.append(config)

    first, second = configs
    assert first.sweep.specification_sha256 == second.sweep.specification_sha256
    first_units = runner.build_work_units(first)
    second_units = runner.build_work_units(second)
    assert [work.work_sha256 for _, work in first_units] == [
        work.work_sha256 for _, work in second_units
    ]
    for (_, first_work), (_, second_work) in zip(first_units, second_units, strict=True):
        assert (
            raw_tau_path(first.runtime.data_root, first_work).read_bytes()
            == raw_tau_path(
                second.runtime.data_root,
                second_work,
            ).read_bytes()
        )

    logical_rows = []
    for config in configs:
        with sqlite3.connect(catalog_path(config.runtime.data_root)) as connection:
            logical_rows.append(
                connection.execute(
                    """
                    SELECT work_sha256, logical_result_sha256
                    FROM work_units ORDER BY cell_sha256, p_index
                    """
                ).fetchall()
            )
    assert logical_rows[0] == logical_rows[1]


def test_scientific_manifest_rejects_legacy_data_without_environment_contract(
    tmp_path: Path,
) -> None:
    config = load_config(_write_config(tmp_path))
    expected = runner._manifest_payload(config, runner.make_plan(config))
    legacy = dict(expected)
    legacy["schema_version"] = 1
    legacy.pop("manifest_kind")
    legacy.pop("environment_contract")
    legacy.pop("environment_contract_sha256")
    legacy["runtime"] = {"workers": 99}
    legacy["environment"] = {"python": "old"}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsupported"):
        runner._ensure_scientific_manifest(path, expected)


def test_scientific_manifest_rejects_non_scientific_fields(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    expected = runner._manifest_payload(config, runner.make_plan(config))
    tampered = dict(expected)
    tampered["runtime"] = {"workers": 1}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unknown fields"):
        runner._ensure_scientific_manifest(path, expected)


def test_runner_lock_rejects_nested_live_holder_and_leaves_audit_marker(
    tmp_path: Path,
) -> None:
    experiment_id = "exp_lock_test"
    lock = tmp_path / "single_ref" / "raw_tau" / "runtime" / f"{experiment_id}.lock"
    with runner._runner_lock(tmp_path, experiment_id):
        assert lock.exists()
        with (
            pytest.raises(RuntimeError, match="already held"),
            runner._runner_lock(tmp_path, experiment_id),
        ):
            pass

    assert lock.exists()
    marker = json.loads(lock.read_text(encoding="utf-8"))
    assert marker["state"] == "released"
    assert marker["released_at_utc"].endswith("Z")


def test_runner_lock_ignores_stale_marker_when_os_lock_is_free(tmp_path: Path) -> None:
    experiment_id = "exp_stale_marker_test"
    lock = tmp_path / "single_ref" / "raw_tau" / "runtime" / f"{experiment_id}.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "pid": os.getpid(),
                "state": "locked",
                "acquired_at_utc": "2000-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    with runner._runner_lock(tmp_path, experiment_id):
        pass

    marker = json.loads(lock.read_text(encoding="utf-8"))
    assert marker["state"] == "released"
    assert marker["acquired_at_utc"] != "2000-01-01T00:00:00Z"


def test_thread_limit_environment_is_forced_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in THREAD_LIMIT_VARIABLES:
        monkeypatch.setenv(name, "17")

    runner._set_worker_thread_limits()

    assert {name: os.environ[name] for name in THREAD_LIMIT_VARIABLES} == {
        name: "1" for name in THREAD_LIMIT_VARIABLES
    }


def test_worker_rejects_a_different_scientific_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = ScientificEnvironmentContract("3.actual", "2.actual", "0.actual")
    expected = ScientificEnvironmentContract("3.expected", "2.expected", "0.expected")
    for name in THREAD_LIMIT_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(runner, "current_scientific_environment_contract", lambda: actual)

    with pytest.raises(RuntimeError, match="differs"):
        runner._worker_initializer(expected)


def test_atomic_status_write_retries_transient_windows_reader_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "status.json"
    real_replace = os.replace
    attempts = 0
    delays: list[float] = []

    def flaky_replace(source, target) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated antivirus reader")
        real_replace(source, target)

    monkeypatch.setattr(runner.os, "replace", flaky_replace)
    monkeypatch.setattr(runner.time, "sleep", delays.append)

    runner._atomic_text(destination, "ready\n")

    assert destination.read_text(encoding="utf-8") == "ready\n"
    assert attempts == 3
    assert delays == [0.05, 0.1]
