"""code_analyze on the [[4,2,2]] code, a saved trajectory code, an open state, a code file."""

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
