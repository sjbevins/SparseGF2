"""state_* tools: a Bell pair, queries, save/load, phase-free limits."""


def test_bell_pair_session(run_tool, ctx):
    new = run_tool("state_new", num_qubits=3, seed=4)
    assert new.text.startswith("state_new: s1, 3 qubits")
    applied = run_tool("state_apply", state_id="s1", circuit="H 0\nCX 0 1\nM 0")
    assert "1 measurement" in applied.text and "random: 1" in applied.text
    q = run_tool(
        "state_query",
        state_id="s1",
        observables=["entropy(0:1)", "expectation(Z0*Z1)", "stabilizers", "destabilizers"],
    )
    assert "entropy(0:1): 0" in q.text and "expectation(Z0*Z1): 1" in q.text
    saved = run_tool("state_save", state_id="s1", path="bell.state.npz")
    assert (ctx.root / "bell.state.npz").exists() and not saved.is_error
    again = run_tool("state_new", source="bell.state.npz")
    assert again.text.startswith("state_new: s2")
    s1, s2 = ctx.states.get("s1"), ctx.states.get("s2")
    assert [str(p) for p in s1.stabilizers()] == [str(p) for p in s2.stabilizers()]
    assert "s1" in run_tool("state_list").text
    run_tool("state_close", state_id="s1")
    gone = run_tool("state_query", state_id="s1", observables=["entropy(0:1)"])
    assert "unknown state_id 's1'" in gone.text


def test_phase_free_has_no_expectations(run_tool):
    run_tool("state_new", num_qubits=2, phases=False)
    out = run_tool("state_query", state_id="s1", observables=["expectation(Z0)"])
    assert out.is_error and "needs sign tracking" in out.text


def test_needs_exactly_one_source(run_tool):
    assert "exactly one of num_qubits or source" in run_tool("state_new").text


def test_save_suffix(run_tool):
    run_tool("state_new", num_qubits=1)
    assert ".state.npz" in run_tool("state_save", state_id="s1", path="x.npz").text
