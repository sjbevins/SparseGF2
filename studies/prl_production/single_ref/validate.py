"""Bounded numerical and storage validation for the production workflow."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
from studies.prl_production.campaign import GRAPH_K, MEAN_DEGREE, production_profile
from studies.prl_production.single_ref.engine import (
    PointSpec,
    graph_bank_path,
    load_graph_bank,
    point_path,
    prepare_graph_bank,
    run_point,
    simulate_trajectory,
)

from sparsegf2.circuits.graphs import _ws_rewire_edges

REPO_TMP = Path(__file__).resolve().parents[3] / "tmp"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    profile = production_profile()
    assert len(profile.betas) == 55
    assert profile.sizes == (32, 48, 64, 96, 128, 160, 192, 256)
    assert all(len(profile.p_by_beta[beta]) == 85 for beta in profile.betas)
    assert profile.n_points == 37_400
    assert profile.n_trajectories == 18_700_000

    point = PointSpec(n=8, beta=0.1, p=0.2, n_graphs=4)
    edges = np.asarray(_ws_rewire_edges(point.n, GRAPH_K, point.beta, 0), dtype=np.int32)
    assert edges.shape == (point.n * GRAPH_K, 2)
    assert 2 * len(edges) / point.n == MEAN_DEGREE

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
    assert batch.s_r_trace[-1] == 0
    assert batch.tau_p == len(batch.s_r_trace)

    no_measurement = simulate_trajectory(
        PointSpec(n=8, beta=0.1, p=0.0, n_graphs=4),
        0,
        edges,
        record_trace=True,
    )
    assert no_measurement.tau_p is None
    assert no_measurement.stop_layer == 64
    assert no_measurement.s_r_trace == (1,) * 64

    full_measurement = simulate_trajectory(
        PointSpec(n=8, beta=0.1, p=1.0, n_graphs=4),
        0,
        edges,
        record_trace=True,
    )
    assert full_measurement.tau_p == 1
    assert full_measurement.s_r_trace == (0,)

    REPO_TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prl_single_ref_", dir=REPO_TMP) as temp_name:
        temp = Path(temp_name)
        root_a = temp / "a"
        root_b = temp / "b"
        prepare_graph_bank(root_a, point)
        prepare_graph_bank(root_b, point)
        bank_a = graph_bank_path(root_a, point)
        bank_b = graph_bank_path(root_b, point)
        assert _file_hash(bank_a) == _file_hash(bank_b)
        assert np.array_equal(load_graph_bank(root_a, point), load_graph_bank(root_b, point))

        partial = run_point(
            root_a,
            point,
            checkpoint_every=1,
            record_traces=True,
            max_new_trajectories=2,
        )
        assert partial.completed == 2
        resumed = run_point(root_a, point, checkpoint_every=1, record_traces=True)
        assert resumed.completed == point.n_graphs
        assert resumed.newly_completed == 2
        output = point_path(root_a, point)
        completed_hash = _file_hash(output)
        unchanged = run_point(root_a, point, checkpoint_every=1, record_traces=True)
        assert unchanged.newly_completed == 0
        assert _file_hash(output) == completed_hash
        with np.load(output, allow_pickle=False) as data:
            assert np.all(data["complete"] == 1)
            observed = data["event_observed"].astype(bool)
            assert np.all(data["tau_p"][observed] == data["stop_layer"][observed])
            assert np.all(data["tau_p"][~observed] == -1)
            assert np.all(data["stop_layer"][~observed] == point.cap)

    print("single-reference production validation: PASS")
    print(f"production points={profile.n_points:,}")
    print(f"production trajectories={profile.n_trajectories:,}")
    print("mean degree=4, T_max=8n, exact-layer S(R), batch/scalar parity, resume=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
