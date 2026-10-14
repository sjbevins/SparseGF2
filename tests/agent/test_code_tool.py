"""code_analyze on the [[4,2,2]] code, a saved trajectory code, an open state, a code file."""

import numpy as np

from sparsegf2 import StabilizerCode, analysis

C422 = ["XXXX", "ZZZZ"]


def test_properties_of_the_422_code(run_tool):
    out = run_tool(
        "code_analyze",
        stabilizers=C422,
        properties=[
            "summary",
            "first_failure(samples=50)",
            "exhaustive_counts(2)",
            "distance_bounds",
            "mean_first_failure",
            "failure_probability(rates=[0.1], delta=0.01)",
            "stabilizers",
            "stabilizer_weights",
            "contiguous_distance",
            "lost_information(0:2)",
            "is_correctable([0])",
        ],
    )
    assert not out.is_error, out.text
    t = out.text
    assert "summary: n=4, k=2, generators=2" in t
    assert "distance_bounds: d = 2" in t and "mean_first_failure: 2" in t
    assert "stabilizers: XXXX ZZZZ" in t and "is_correctable([0]): true" in t
    assert "lost_information(0:2): 2" in t and "contiguous_distance: 2" in t
    phi = analysis.erasure_failure_probability(
        StabilizerCode(C422).first_failure_sizes(512, seed=0),
        4,
        0.1,
        exact_counts=StabilizerCode(C422).uncorrectable_counts(2),
    )
    assert f"{phi:.4g}" in t


def test_code_from_a_trajectory_record_and_a_saved_file(run_tool, ctx):
    rec = run_tool(
        "traj_run",
        num_system=10,
        layers=8,
        measurement_rate=0.1,
        reference="purification",
        seed=2,
    ).data["output_dir"]
    a = run_tool("code_analyze", source=rec, properties=["summary"], save_to="c.code.npz")
    b = run_tool("code_analyze", source="c.code.npz", properties=["summary"])
    assert a.text.split("\n")[1] == b.text.split("\n")[1]


def test_code_of_an_open_state(run_tool):
    run_tool("state_new", num_qubits=4, phases=False)
    run_tool("state_apply", state_id="s1", circuit="H 0\nCX 0 2\nH 1\nCX 1 3")
    out = run_tool("code_analyze", state_id="s1", reference_qubits="2:4", properties=["summary"])
    assert "n=2, k=2, generators=0" in out.text


def test_errors(run_tool):
    assert (
        "exactly one of stabilizers, source or state_id"
        in run_tool("code_analyze", properties=["summary"]).text
    )
    out = run_tool("code_analyze", stabilizers=C422, properties=["distanse"])
    assert out.is_error and "did you mean distance_bounds" in out.text


def test_exhaustive_limit_is_refused_with_the_largest_size(run_tool):
    big = ["X" * 64, "Z" * 64]
    out = run_tool("code_analyze", stabilizers=big, properties=["exhaustive_counts(10)"])
    assert out.is_error and "needs" in out.text and "use max_size <= 8 for n = 64" in out.text
    auto = run_tool(
        "code_analyze", stabilizers=big, properties=["exhaustive_counts(10)"], background="auto"
    )
    assert auto.is_error and "use max_size <=" in auto.text


def test_a_large_estimate_returns_a_job(run_tool, ctx):
    ctx.sync_seconds = 1e-4
    out = run_tool(
        "code_analyze",
        stabilizers=C422,
        properties=["first_failure(samples=50)"],
        background="auto",
    )
    assert not out.is_error and out.data.get("job_id"), out.text


def test_sizes_are_validated_not_truncated(run_tool):
    for call in ("first_failure(samples=0)", "first_failure(samples=2.5)", "exhaustive_counts(0)"):
        out = run_tool("code_analyze", stabilizers=C422, properties=[call])
        assert out.is_error, call
    assert (
        "at least 1"
        in run_tool("code_analyze", stabilizers=C422, properties=["first_failure(samples=0)"]).text
    )
    assert (
        "integer"
        in run_tool(
            "code_analyze", stabilizers=C422, properties=["first_failure(samples=2.5)"]
        ).text
    )


def _record(run_tool, **kw):
    args = dict(num_system=12, layers=10, measurement_rate=0.2, reference="purification")
    args |= dict(trajectory=3, seed=5, checkpoints="[0, 5, 10]") | kw
    return run_tool("traj_run", **args).data["output_dir"]


def test_selecting_an_evaluation_of_a_record(run_tool, ctx):
    from sparsegf2 import mipt
    from sparsegf2.agent.tools.traj import canonical

    rec = _record(run_tool)
    res = mipt.EnsembleResult.load(ctx.root / rec / "result.npz")
    for evaluation in (0, 1, -1):
        out = run_tool(
            "code_analyze", source=rec, evaluation=evaluation, properties=["stabilizers"],
            save_to=f"e{evaluation}.code.npz",
        )  # fmt: skip
        assert not out.is_error, out.text
        saved = run_tool("code_analyze", source=f"e{evaluation}.code.npz", properties=["summary"])
        assert not saved.is_error
        want = canonical(res.code(0, evaluation).stabilizer_matrix())
        with np.load(ctx.root / f"e{evaluation}.code.npz") as z:
            np.testing.assert_array_equal(canonical(z["stabilizers"]), want)
    assert "trajectory 3" in out.text and "layer 10" in out.text
    assert "trajectory 3" in run_tool("code_analyze", source=rec, trajectory=3).text


def test_bad_evaluation_and_trajectory_name_what_exists(run_tool):
    rec = _record(run_tool)
    out = run_tool("code_analyze", source=rec, evaluation=7)
    assert out.is_error and "layers [0, 5, 10]" in out.text and "0..2 or -1" in out.text
    out = run_tool("code_analyze", source=rec, trajectory=4)
    assert out.is_error and "trajectory 4 is not in" in out.text


def test_a_purified_trajectory_has_no_code_later(run_tool):
    rec = _record(
        run_tool,
        measurement_rate=0.9,
        layers=20,
        checkpoints="[0, 10, 20]",
        stop_when_purified=True,
    )
    out = run_tool("code_analyze", source=rec, evaluation=-1)
    assert out.is_error and "had stopped" in out.text and "trajectory 3" in out.text
    assert "layer 20" in out.text and "k = 0" in out.text


def test_a_result_without_codes(run_tool):
    run = run_tool(
        "ensemble_run", num_system=8, layers=4, measurement_rate=0.1, reference="purification",
        trajectories=2, seed=1, output_dir="plain",
    )  # fmt: skip
    assert not run.is_error, run.text
    out = run_tool("code_analyze", source="plain")
    assert out.is_error and "saved no code" in out.text


def test_a_state_file_source(run_tool):
    run_tool("state_new", num_qubits=4, phases=False)
    run_tool("state_apply", state_id="s1", circuit="H 0\nCX 0 2\nH 1\nCX 1 3")
    assert not run_tool("state_save", state_id="s1", path="s.state.npz").is_error
    out = run_tool("code_analyze", source="s.state.npz", reference_qubits="2:4")
    assert "n=2, k=2, generators=0" in out.text
    assert "needs reference_qubits" in run_tool("code_analyze", source="s.state.npz").text


def test_no_logical_qubits_is_said_not_given_as_a_distance(run_tool):
    out = run_tool(
        "code_analyze",
        stabilizers=["ZZ", "XX"],
        properties=["distance_bounds", "contiguous_distance"],
    )
    assert out.text.count("no logical qubits (k = 0)") == 2, out.text
