from __future__ import annotations

import json
from pathlib import Path

import pytest
from studies.prl_production.single_ref.benchmark import (
    THREAD_LIMIT_VARIABLES,
    TrialMetrics,
    choose_circuit_tile,
    estimate_work,
    main,
    recommended_worker_grid,
    summarize_scaling,
)
from studies.prl_production.single_ref.raw_tau import benchmark as raw_benchmark
from studies.prl_production.single_ref.raw_tau.benchmark import (
    ADMISSION_SAFETY_FACTOR,
    PCG64_CONTRACT,
    AdmissionPilot,
    ChunkMetrics,
    _aggregate_trial,
    _circulant_c2_edges,
    _require_thread_limits,
    _verified_environment_contract,
    make_benchmark_work,
    run_admission_pilot,
)
from studies.prl_production.single_ref.raw_tau.config import source_fingerprint_sha256
from studies.prl_production.single_ref.raw_tau.engine import (
    RawTauTrajectoryResult,
    TrajectoryTimings,
)
from studies.prl_production.sweep_spec import ScientificEnvironmentContract


def _trial(*, workers: int, trajectories: int, wall_seconds: float) -> TrialMetrics:
    return TrialMetrics(
        workers=workers,
        wall_seconds=wall_seconds,
        cpu_seconds=workers * wall_seconds * 0.75,
        trajectories=trajectories,
        layers=100 * trajectories,
        scramble_gates=10 * trajectories,
        dynamics_gates=200 * trajectories,
        measurement_trials=400 * trajectories,
        measurements=40 * trajectories,
        events=trajectories // 2,
        peak_rss_bytes=workers * 1_000_000,
    )


def test_work_estimate_counts_every_cartesian_trajectory_and_capped_operation() -> None:
    result = estimate_work(
        (4, 5),
        graph_parameter_cells=2,
        n_graphs=3,
        n_p=4,
        n_circuits=5,
        q_max=2,
        q_scramble=3,
    )

    assert result.n_cells == 4
    assert result.n_trajectories == 240
    assert result.scramble_gates == 3_240
    assert result.capped_layers == 2_160
    assert result.capped_dynamics_gates == 4_320
    assert result.capped_measurement_trials == 9_840
    assert result.raw_tau_bytes == 960


def test_work_estimate_allows_an_explicit_zero_length_scramble() -> None:
    result = estimate_work(
        (8,),
        graph_parameter_cells=1,
        n_graphs=2,
        n_p=3,
        n_circuits=4,
        q_max=5,
        q_scramble=0,
    )
    assert result.scramble_gates == 0


@pytest.mark.parametrize(
    ("seconds", "n_circuits", "expected"),
    [(2.0, 100, 1), (0.25, 100, 4), (0.001, 17, 17)],
)
def test_circuit_tile_targets_one_second_and_stays_bounded(
    seconds: float, n_circuits: int, expected: int
) -> None:
    assert choose_circuit_tile(seconds, n_circuits=n_circuits) == expected


def test_worker_grid_uses_physical_cores_and_retains_nonstandard_endpoint() -> None:
    assert recommended_worker_grid(10) == (1, 2, 4, 8, 10)
    assert recommended_worker_grid(16) == (1, 2, 4, 8, 12, 16)


def test_scaling_summary_uses_median_one_worker_baseline() -> None:
    summary = summarize_scaling(
        [
            _trial(workers=1, trajectories=10, wall_seconds=2.0),
            _trial(workers=1, trajectories=12, wall_seconds=2.0),
            _trial(workers=2, trajectories=20, wall_seconds=2.0),
        ]
    )

    assert summary[0].median_trajectories_per_second == 5.5
    assert summary[0].speedup == 1.0
    assert summary[0].efficiency == 1.0
    assert summary[1].median_trajectories_per_second == 10.0
    assert summary[1].speedup == pytest.approx(10.0 / 5.5)
    assert summary[1].efficiency == pytest.approx(10.0 / 5.5 / 2)


def test_trial_metrics_reject_inconsistent_measurement_counts() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        TrialMetrics(
            workers=1,
            wall_seconds=1.0,
            cpu_seconds=1.0,
            trajectories=1,
            layers=1,
            scramble_gates=1,
            dynamics_gates=1,
            measurement_trials=1,
            measurements=2,
            events=0,
        )


def test_thread_limit_contract_covers_numba_and_common_native_runtimes() -> None:
    assert THREAD_LIMIT_VARIABLES == (
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )


def test_estimate_cli_prints_and_optionally_writes_same_json(tmp_path, capsys) -> None:
    output = tmp_path / "estimate.json"
    assert (
        main(
            [
                "--sizes",
                "4",
                "5",
                "--graph-parameter-cells",
                "2",
                "--graphs",
                "3",
                "--p-count",
                "4",
                "--circuits",
                "5",
                "--q-max",
                "2",
                "--q-scramble",
                "3",
                "--json",
                str(output),
            ]
        )
        == 0
    )

    printed = capsys.readouterr().out
    assert json.loads(printed) == json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(printed)["n_trajectories"] == 240


def test_raw_tau_benchmark_case_uses_exact_v2_protocol() -> None:
    work = make_benchmark_work(n=9, p="0.125", n_circuits=7, q_max=3, q_scramble=2)
    edges = _circulant_c2_edges(9)

    assert work.p_decimal == "0.125"
    assert work.raw_shape == (1, 7)
    assert work.protocol.t_max(9) == 27
    assert work.protocol.scramble_gate_count(9) == 18
    assert edges.shape == (18, 2)
    assert len({tuple(sorted((int(u), int(v)))) for u, v in edges}) == 18


def _profile_result(
    *,
    total_s: float,
    stop_layer: int,
    event_observed: bool,
) -> RawTauTrajectoryResult:
    return RawTauTrajectoryResult(
        tau_p=stop_layer if event_observed else None,
        stop_layer=stop_layer,
        event_observed=event_observed,
        reference_system_qubit=0,
        layers_executed=stop_layer,
        scramble_gates=0,
        dynamic_gates=0,
        measurements=0,
        timings=TrajectoryTimings(0.0, 0.0, 0.0, 0.0, 0.0, total_s),
    )


def test_admission_pilot_profiles_fixed_requested_and_capped_p_zero_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = make_benchmark_work(n=9, p="0.5", n_circuits=4, q_max=2, q_scramble=1)
    edges = _circulant_c2_edges(9)
    calls: list[tuple[str, int, bool]] = []

    def fake_simulate(point, _graph_index, circuit_index, _edges, *, profile=False, **_kwargs):
        calls.append((point.p_decimal, circuit_index, profile))
        if not profile:
            return _profile_result(total_s=0.001, stop_layer=1, event_observed=True)
        if point.p_decimal == "0":
            return _profile_result(
                total_s=0.040 + 0.001 * circuit_index,
                stop_layer=18,
                event_observed=False,
            )
        return _profile_result(
            total_s=0.010 + 0.001 * circuit_index,
            stop_layer=1,
            event_observed=True,
        )

    monkeypatch.setattr(raw_benchmark, "simulate_trajectory", fake_simulate)
    profile, pilot = run_admission_pilot(
        work,
        edges,
        execution="batch",
        hybrid=True,
        pilot_trajectories=3,
        safety_factor=1.5,
    )

    assert profile.timings is not None and profile.timings.total_s == 0.010
    assert pilot.circuit_indices == (0, 1, 2)
    assert pilot.t_max == 18
    assert pilot.capped_stop_layers == (18, 18, 18)
    assert pilot.capped_event_observed == (False, False, False)
    assert pilot.requested_seconds == pytest.approx((0.010, 0.011, 0.012))
    assert pilot.capped_seconds == pytest.approx((0.040, 0.041, 0.042))
    assert pilot.maximum_observed_seconds == pytest.approx(0.042)
    assert pilot.seconds_per_trajectory_guard == pytest.approx(0.063)
    assert pilot.estimated_one_worker_repeat_seconds == pytest.approx(0.252)
    assert [(p, index) for p, index, timed in calls if timed and p == "0"] == [
        ("0", 0),
        ("0", 1),
        ("0", 2),
    ]


def test_admission_aborts_after_first_slow_capped_pilot_before_extra_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = make_benchmark_work(n=9, p="0.5", n_circuits=4, q_max=2, q_scramble=1)
    calls: list[tuple[str, int, bool]] = []

    def fake_simulate(point, _graph_index, circuit_index, _edges, *, profile=False, **_kwargs):
        calls.append((point.p_decimal, circuit_index, profile))
        if not profile:
            return _profile_result(total_s=0.001, stop_layer=1, event_observed=True)
        return _profile_result(total_s=0.1, stop_layer=18, event_observed=False)

    monkeypatch.setattr(raw_benchmark, "simulate_trajectory", fake_simulate)
    with pytest.raises(SystemExit, match="conservative pilot"):
        run_admission_pilot(
            work,
            _circulant_c2_edges(9),
            execution="batch",
            hybrid=True,
            pilot_trajectories=3,
            safety_factor=1.25,
            max_estimated_seconds_per_repeat=0.1,
        )

    assert calls == [("1", 0, False), ("0", 0, True)]


def test_environment_contract_pins_literal_verified_pcg64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _verified_environment_contract()
    assert contract.bit_generator == PCG64_CONTRACT == "PCG64"

    mismatched = ScientificEnvironmentContract(
        python=contract.python,
        numpy=contract.numpy,
        numba=contract.numba,
        bit_generator="NOT_PCG64",
    )
    monkeypatch.setattr(
        raw_benchmark,
        "current_scientific_environment_contract",
        lambda: mismatched,
    )
    with pytest.raises(RuntimeError, match="literal PCG64"):
        _verified_environment_contract()
    with pytest.raises(ValueError, match="must pin PCG64"):
        make_benchmark_work(
            n=5,
            p="0",
            n_circuits=1,
            q_max=1,
            q_scramble=0,
            environment_contract=mismatched,
        )


def test_chunk_aggregation_counts_memory_once_per_worker_process() -> None:
    chunks = [
        ChunkMetrics(10, 0.5, 2, 20, 8, 80, 180, 18, 1, 1_000),
        ChunkMetrics(10, 0.6, 3, 30, 12, 120, 270, 27, 2, 1_200),
        ChunkMetrics(11, 0.4, 1, 10, 4, 40, 90, 9, 0, 900),
    ]

    trial = _aggregate_trial(2, 1.0, chunks)

    assert trial.trajectories == 6
    assert trial.layers == 60
    assert trial.scramble_gates == 24
    assert trial.dynamics_gates == 240
    assert trial.measurement_trials == 540
    assert trial.measurements == 54
    assert trial.events == 3
    assert trial.cpu_seconds == 1.5
    assert trial.peak_rss_bytes == 2_100


def test_benchmark_requires_preimport_thread_limits(monkeypatch) -> None:
    for name in THREAD_LIMIT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="before Python starts"):
        _require_thread_limits()

    for name in THREAD_LIMIT_VARIABLES:
        monkeypatch.setenv(name, "1")
    _require_thread_limits()


def test_raw_benchmark_report_emits_full_scientific_provenance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in THREAD_LIMIT_VARIABLES:
        monkeypatch.setenv(name, "1")
    profile = _profile_result(total_s=0.01, stop_layer=1, event_observed=True)
    admission = AdmissionPilot(
        circuit_indices=(0,),
        requested_p="0.2",
        capped_p="0",
        t_max=5,
        requested_stop_layers=(1,),
        capped_stop_layers=(5,),
        capped_event_observed=(False,),
        requested_seconds=(0.01,),
        capped_seconds=(0.02,),
        maximum_observed_seconds=0.02,
        safety_factor=ADMISSION_SAFETY_FACTOR,
        seconds_per_trajectory_guard=0.025,
        estimated_one_worker_repeat_seconds=0.025,
    )
    trial = _trial(workers=1, trajectories=1, wall_seconds=0.1)
    monkeypatch.setattr(raw_benchmark, "_gate_luts", lambda: None)
    monkeypatch.setattr(
        raw_benchmark,
        "run_admission_pilot",
        lambda *_args, **_kwargs: (profile, admission),
    )
    monkeypatch.setattr(
        raw_benchmark,
        "run_scaling_benchmark",
        lambda *_args, **_kwargs: ((trial,), {1: 0.1}, {1: 1}),
    )

    assert (
        raw_benchmark.main(
            [
                "--n",
                "5",
                "--p",
                "0.2",
                "--q-max",
                "1",
                "--q-scramble",
                "0",
                "--circuits",
                "1",
                "--workers",
                "1",
                "--repetitions",
                "1",
                "--max-estimated-seconds-per-repeat",
                "1",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    expected_work = make_benchmark_work(
        n=5,
        p="0.2",
        n_circuits=1,
        q_max=1,
        q_scramble=0,
    )
    contract = ScientificEnvironmentContract(**payload["environment_contract"])
    assert payload["schema_version"] == 2
    assert payload["source_fingerprint_sha256"] == source_fingerprint_sha256()
    assert payload["protocol_sha256"] == expected_work.protocol.specification_sha256
    assert payload["environment_contract_sha256"] == contract.specification_sha256
    assert payload["environment_contract"]["bit_generator"] == "PCG64"
    assert payload["admission_pilot"]["capped_p"] == "0"
    assert payload["admission_pilot"]["capped_stop_layers"] == [5]
    assert payload["admission_pilot"]["capped_event_observed"] == [False]
    assert payload["admission_pilot"]["seconds_per_trajectory_guard"] == 0.025


def test_powershell_launcher_sets_every_thread_limit_before_python() -> None:
    text = Path("studies/prl_production/benchmark_single_ref.ps1").read_text(encoding="utf-8")
    for name in THREAD_LIMIT_VARIABLES:
        assert f'$env:{name} = "1"' in text
    assert "single_ref.raw_tau.benchmark" in text
