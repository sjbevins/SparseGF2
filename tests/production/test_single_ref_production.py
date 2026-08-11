from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
from studies.prl_production.campaign import GRAPH_K, production_profile
from studies.prl_production.single_ref import run as production_run
from studies.prl_production.single_ref.engine import (
    PointSpec,
    graph_bank_path,
    point_path,
    prepare_graph_bank,
    run_point,
    simulate_trajectory,
)

from sparsegf2.circuits.graphs import _ws_rewire_edges


def test_production_grid_is_explicit_and_fine() -> None:
    profile = production_profile()
    assert len(profile.betas) == 55
    assert profile.sizes == (32, 48, 64, 96, 128, 160, 192, 256)
    assert profile.n_graphs == 500
    assert all(len(profile.p_by_beta[beta]) == 85 for beta in profile.betas)
    assert all(np.allclose(np.diff(profile.p_by_beta[beta][4:-4]), 0.001) for beta in profile.betas)
    assert profile.n_points == 37_400
    assert profile.n_trajectories == 18_700_000


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n": 7, "beta": 0.1, "p": 0.2, "n_graphs": 1}, "even integer"),
        ({"n": 8, "beta": -0.1, "p": 0.2, "n_graphs": 1}, "beta"),
        ({"n": 8, "beta": 0.1, "p": 1.1, "n_graphs": 1}, "p must"),
        ({"n": 8, "beta": 0.1, "p": 0.2, "n_graphs": 0}, "positive integer"),
    ],
)
def test_point_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        PointSpec(**kwargs)


def test_ws_graphs_have_mean_degree_four() -> None:
    for beta in (0.0, 0.003, 0.01, 0.1, 1.0):
        for graph_index in range(4):
            edges = _ws_rewire_edges(32, GRAPH_K, beta, graph_index)
            assert len(edges) == 32 * GRAPH_K
            assert 2 * len(edges) / 32 == 4


def test_every_layer_first_passage_and_batch_scalar_parity() -> None:
    point = PointSpec(n=8, beta=0.1, p=0.2, n_graphs=1)
    edges = np.asarray(_ws_rewire_edges(8, GRAPH_K, 0.1, 0), dtype=np.int32)
    batch = simulate_trajectory(
        point,
        0,
        edges,
        execution="batch",
        record_trace=True,
        audit_tableau=True,
    )
    scalar = simulate_trajectory(
        point,
        0,
        edges,
        execution="scalar",
        record_trace=True,
        audit_tableau=True,
    )
    assert batch == scalar
    assert batch.s_r_trace is not None
    assert batch.s_r_trace[:-1] == (1,) * (len(batch.s_r_trace) - 1)
    assert batch.s_r_trace[-1] == 0
    assert batch.tau_p == len(batch.s_r_trace)


def test_censoring_and_full_measurement_are_distinct() -> None:
    edges = np.asarray(_ws_rewire_edges(8, GRAPH_K, 0.1, 0), dtype=np.int32)
    censored = simulate_trajectory(
        PointSpec(n=8, beta=0.1, p=0.0, n_graphs=1),
        0,
        edges,
        record_trace=True,
    )
    assert censored.tau_p is None
    assert censored.stop_layer == 64
    assert not censored.event_observed
    assert censored.s_r_trace == (1,) * 64

    observed = simulate_trajectory(
        PointSpec(n=8, beta=0.1, p=1.0, n_graphs=1),
        0,
        edges,
        record_trace=True,
    )
    assert observed.tau_p == 1
    assert observed.stop_layer == 1
    assert observed.event_observed
    assert observed.s_r_trace == (0,)


def test_graph_bank_and_resumed_point_are_byte_reproducible(tmp_path) -> None:
    point = PointSpec(n=8, beta=0.1, p=0.2, n_graphs=3)
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    prepare_graph_bank(root_a, point)
    prepare_graph_bank(root_b, point)
    hash_a = hashlib.sha256(graph_bank_path(root_a, point).read_bytes()).digest()
    hash_b = hashlib.sha256(graph_bank_path(root_b, point).read_bytes()).digest()
    assert hash_a == hash_b

    partial = run_point(
        root_a,
        point,
        checkpoint_every=1,
        record_traces=True,
        max_new_trajectories=1,
    )
    assert partial.completed == 1
    complete = run_point(root_a, point, checkpoint_every=1, record_traces=True)
    assert complete.completed == 3
    output = point_path(root_a, point)
    final_hash = hashlib.sha256(output.read_bytes()).digest()
    repeated = run_point(root_a, point, checkpoint_every=1, record_traces=True)
    assert repeated.newly_completed == 0
    assert hashlib.sha256(output.read_bytes()).digest() == final_hash


def test_runner_lock_blocks_concurrent_writer_and_recovers_stale_pid(
    tmp_path, monkeypatch
) -> None:
    lock = tmp_path / "runner.lock"
    monkeypatch.setattr(production_run, "RUNNER_LOCK", lock)
    with production_run._single_runner_lock():
        owner = json.loads(lock.read_text(encoding="utf-8"))
        assert owner["pid"] == os.getpid()
        with (
            pytest.raises(RuntimeError, match="already active"),
            production_run._single_runner_lock(),
        ):
            pass
    assert not lock.exists()

    lock.write_text(
        json.dumps({"pid": 99_999_999, "token": "stale"}),
        encoding="utf-8",
    )
    with production_run._single_runner_lock():
        owner = json.loads(lock.read_text(encoding="utf-8"))
        assert owner["pid"] == os.getpid()
    assert not lock.exists()


def test_print_run_id_is_write_free(tmp_path, monkeypatch, capsys) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(production_run, "RUNTIME_ROOT", runtime)
    monkeypatch.setattr(production_run, "_source_fingerprint", lambda: "fixed-source")

    assert production_run._main_unlocked(["--profile", "smoke", "--print-run-id"]) == 0
    run_id = capsys.readouterr().out.strip()

    assert re.fullmatch(r"[0-9a-f]{16}", run_id)
    assert not runtime.exists()


def test_detached_scripts_target_the_resolved_run() -> None:
    campaign_root = Path(production_run.__file__).resolve().parents[1]
    launcher = (campaign_root / "run_single_ref.ps1").read_text(encoding="utf-8")
    pause = (campaign_root / "pause_single_ref.ps1").read_text(encoding="utf-8")

    assert '"--run-id", $runId' in launcher
    assert "RedirectStandardOutput = $monitorStdout" in launcher
    assert "RedirectStandardError = $monitorStderr" in launcher
    assert "run_id = $runId" in launcher
    assert '"single_ref_$($runId)_state.json"' in pause
    assert "monitor --run-id $runId" in pause
