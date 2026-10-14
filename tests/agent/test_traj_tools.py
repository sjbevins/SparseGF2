"""traj_run and traj_probe: one trajectory, its code, depth replay, Stim cross-check."""

import json

import numpy as np

from sparsegf2 import StabilizerCode, graphs, mipt
from sparsegf2.agent.tools.traj import canonical

ARGS = dict(
    num_system=12,
    layers=10,
    measurement_rate=0.2,
    reference="purification",
    trajectory=3,
    seed=5,
    checkpoints="[5, 10]",
)


def test_default_stores_the_engine_code(run_tool, ctx):
    out = run_tool("traj_run", **ARGS)
    assert not out.is_error, out.text
    rec = ctx.root / out.data["output_dir"]
    meta = json.loads((rec / "trajectory.json").read_text(encoding="utf-8"))
    assert meta["trajectory"] == 3 and meta["randomness_contract"] == mipt.RANDOMNESS_CONTRACT
    ens = mipt.MonitoredCircuit(
        num_system=12,
        edges=graphs.cycle(12),
        layers=10,
        measurement_rate=0.2,
        reference="purification",
        checkpoints=[5, 10],
        probes=[mipt.code_snapshot()],
    ).run(5, seed=5)
    mine = mipt.EnsembleResult.load(rec / "result.npz")
    for c in range(2):
        np.testing.assert_array_equal(
            mine.code(0, c).stabilizer_matrix(), ens.code(3, c).stabilizer_matrix()
        )
    assert "layer" in out.text and "k" in out.text


def test_store_code_needs_a_reference(run_tool):
    out = run_tool("traj_run", num_system=8, layers=4, measurement_rate=0.1)
    assert out.is_error and "store=code needs a code" in out.text


def test_store_state_and_reopen(run_tool, ctx):
    out = run_tool("traj_run", **ARGS, store="state")
    rec = ctx.root / out.data["output_dir"]
    assert (rec / "state_L10.state.npz").exists()
    opened = run_tool("state_new", source=f"{out.data['output_dir']}/state_L10.state.npz")
    sim = ctx.states.get(opened.data["state_id"])
    code = StabilizerCode.from_simulator(sim, reference=range(12, 24))
    engine = run_tool("traj_run", **ARGS)  # same trajectory, code stored
    stored = mipt.EnsembleResult.load(ctx.root / engine.data["output_dir"] / "result.npz")
    np.testing.assert_array_equal(
        canonical(code.stabilizer_matrix()), canonical(stored.code(0, 1).stabilizer_matrix())
    )


def test_compare_stim_and_save_circuit(run_tool, ctx):
    import stim

    out = run_tool("traj_run", **ARGS, compare_stim=True, save_circuit=True)
    assert "stim: match" in out.text and "engine vs simulator: match" in out.text
    text = (ctx.root / out.data["output_dir"] / "circuit.stim").read_text(encoding="utf-8")
    stim.Circuit(text)


def test_probe_replays_at_new_depths(run_tool, ctx):
    first = run_tool("traj_run", **ARGS)
    again = run_tool(
        "traj_probe",
        source=first.data["output_dir"],
        probes=["code_dimension", "half_cut_entropy"],
        checkpoints="[2, 5, 10]",
    )
    assert not again.is_error, again.text
    a = mipt.EnsembleResult.load(ctx.root / first.data["output_dir"] / "result.npz")
    b = mipt.EnsembleResult.load(ctx.root / again.data["output_dir"] / "result.npz")
    np.testing.assert_array_equal(
        a.code(0, 1).stabilizer_matrix(), b.code(0, 2).stabilizer_matrix()
    )


def test_probe_refuses_another_randomness_contract(run_tool, ctx):
    first = run_tool("traj_run", **ARGS)
    f = ctx.root / first.data["output_dir"] / "trajectory.json"
    meta = json.loads(f.read_text(encoding="utf-8"))
    meta["randomness_contract"] = 0
    f.write_text(json.dumps(meta), encoding="utf-8")
    out = run_tool("traj_probe", source=first.data["output_dir"], probes=["code_dimension"])
    assert out.is_error and "randomness contract" in out.text


def test_run_refuses_a_directory_holding_another_trajectory(run_tool, ctx):
    first = run_tool("traj_run", **ARGS, output_dir="rec")
    assert not first.is_error, first.text
    f = ctx.root / "rec" / "trajectory.json"
    before = f.read_text(encoding="utf-8")
    out = run_tool("traj_run", **{**ARGS, "seed": 6}, output_dir="rec")
    assert out.is_error
    assert "rec holds a different trajectory record; give another output_dir" in out.text
    assert f.read_text(encoding="utf-8") == before
    again = run_tool("traj_run", **ARGS, output_dir="rec")  # the same trajectory may be rewritten
    assert not again.is_error, again.text
