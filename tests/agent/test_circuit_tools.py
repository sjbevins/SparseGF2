"""circuit_inspect, circuit_run, circuit_convert, circuit_crosscheck."""

GHZ = "H 0\nCX 0 1\nTICK\nCX 1 2\nTICK\nM 0 1 2\nDETECTOR rec[-1] rec[-2]"


def test_inspect_valid_and_invalid(run_tool):
    out = run_tool("circuit_inspect", circuit=GHZ)
    assert out.text.startswith("circuit_inspect: valid") and "num_qubits: 3" in out.text
    bad = run_tool("circuit_inspect", circuit="H 0\nFOO 1")
    assert "line 2: unknown gate 'FOO'" in bad.text and not bad.is_error


def test_run_single_shot_with_observables_per_tick(run_tool):
    out = run_tool(
        "circuit_run",
        circuit=GHZ,
        seed=1,
        observables=["entropy(0:1)", "stabilizers"],
        observe_ticks=True,
    )
    assert not out.is_error, out.text
    assert out.text.startswith("circuit_run: 3 qubits, 3 measurements, 1 shot")
    assert "measurements: " in out.text and "entropy" in out.text


def test_run_many_shots(run_tool):
    out = run_tool("circuit_run", circuit=GHZ, shots=200, seed=2)
    assert "measurement_means" in out.text and "detector_rates" in out.text


def test_circuit_file_and_exactly_one_source(run_tool, ctx):
    (ctx.root / "c.stim").write_text(GHZ, encoding="utf-8")
    assert not run_tool("circuit_inspect", circuit_file="c.stim").is_error
    assert "exactly one of circuit or circuit_file" in run_tool("circuit_inspect").text
    assert "exactly one" in run_tool("circuit_inspect", circuit=GHZ, circuit_file="c.stim").text


def test_convert_expands_cliffords(run_tool, ctx):
    out = run_tool("circuit_convert", circuit="CLIFFORD_2Q(77) 0 1\nM 0", to="stim")
    assert "CLIFFORD_2Q" not in out.text.split("\n", 1)[1]
    run_tool("circuit_convert", circuit=GHZ, to="stim", output_file="out.stim")
    assert (ctx.root / "out.stim").read_text(encoding="utf-8").startswith("H 0")


def test_crosscheck(run_tool):
    out = run_tool("circuit_crosscheck", circuit=GHZ.replace("DETECTOR", "TICK\nDETECTOR"), seed=3)
    assert "stabilizers: match" in out.text and "measurements: match" in out.text
