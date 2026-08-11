"""Tests for the circuit inspector (setup_ops, trace_circuit, render_text)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from sparsegf2 import from_bell_purification
from sparsegf2.circuits import (
    CircuitConfig,
    CircuitTrace,
    Op,
    inspect_circuit,
    setup_ops,
    setup_picture,
    trace_circuit,
)
from sparsegf2.circuits.ops import apply_named_gate
from sparsegf2.errors import InvalidArgumentError

# ----------------------------------------------------------------------
# setup_ops: the declarative construction recipe
# ----------------------------------------------------------------------


def test_setup_ops_pure_state_is_empty():
    assert setup_ops("pure_state", 8) == []


def test_setup_ops_single_ref():
    ops = setup_ops("single_ref", 8)
    assert [(o.label, o.qubits) for o in ops] == [("H", (7,)), ("CX", (7, 8))]
    assert all(o.random is False for o in ops)  # construction is deterministic
    assert "Bell pair" in ops[1].detail["note"]


def test_setup_ops_purification_one_bell_pair_per_system_qubit():
    n = 4
    ops = setup_ops("purification", n)
    # H(i); CX(i, i+n) for each i
    expected = []
    for i in range(n):
        expected += [("H", (i,)), ("CX", (i, i + n))]
    assert [(o.label, o.qubits) for o in ops] == expected


def test_setup_ops_validates():
    with pytest.raises(InvalidArgumentError):
        setup_ops("teleport", 8)
    with pytest.raises(InvalidArgumentError):
        setup_ops("single_ref", 0)


# ----------------------------------------------------------------------
# Single source of truth: setup_picture applies exactly setup_ops, and
# purification still equals the core's from_bell_purification.
# ----------------------------------------------------------------------


def test_setup_picture_applies_setup_ops_no_drift():
    # Build a sim by hand from setup_ops and compare to setup_picture's state.
    from sparsegf2 import SparseGF2

    n = 6
    for picture in ("pure_state", "purification", "single_ref"):
        sim_auto, spec = setup_picture(picture, n)
        sim_manual = SparseGF2(spec.total_qubits)
        for op in setup_ops(picture, n):
            apply_named_gate(sim_manual, op.label, op.qubits)
        assert np.array_equal(sim_auto.to_symplectic(), sim_manual.to_symplectic())


def test_purification_matches_core_factory():
    # The refactored purification setup must produce the identical state to
    # the core's from_bell_purification.
    n = 5
    sim_pic, _ = setup_picture("purification", n)
    sim_core = from_bell_purification(n)
    assert np.array_equal(sim_pic.to_symplectic(), sim_core.to_symplectic())


# ----------------------------------------------------------------------
# trace_circuit: the IR
# ----------------------------------------------------------------------


def test_trace_has_setup_then_layers():
    cfg = CircuitConfig(graph_spec="cycle", n=8, picture="single_ref", p=0.16, depth_factor=4)
    tr = trace_circuit(cfg, sample_seed=0, max_layers=5)
    assert isinstance(tr, CircuitTrace)
    kinds = [s.kind for s in tr.stages]
    assert kinds[0] == "setup"
    assert kinds.count("layer") == 5
    assert tr.total_layers == 32
    assert tr.shown_layers == 5
    assert tr.truncated is True


def test_trace_summary_describes_all_edges_and_exact_depth():
    cfg = CircuitConfig(
        graph_spec="cycle",
        n=8,
        gating_mode="all_edges",
        total_layers_override=3,
    )
    tr = trace_circuit(cfg, sample_seed=0, max_layers=1)
    assert tr.summary["gating"] == "all_edges (8 edges/layer)"
    assert tr.summary["depth"] == "exact override = 3 measured layers"


def test_trace_setup_shows_bell_pair_for_single_ref():
    cfg = CircuitConfig(graph_spec="cycle", n=8, picture="single_ref", p=0.1, depth_factor=2)
    tr = trace_circuit(cfg, max_layers=2)
    setup = tr.stages[0]
    assert [(o.label, o.qubits) for o in setup.ops] == [("H", (7,)), ("CX", (7, 8))]


def test_trace_layer_gates_are_random_with_index():
    cfg = CircuitConfig(graph_spec="cycle", n=8, p=0.1, depth_factor=2)
    tr = trace_circuit(cfg, max_layers=3)
    layer0 = next(s for s in tr.stages if s.kind == "layer")
    gates = [o for o in layer0.ops if o.kind == "gate"]
    assert len(gates) == 4  # brickwork on cycle(8): n/2
    for op in gates:
        assert op.random is True
        assert "clifford_index" in op.detail
        assert op.label == f"C[{op.detail['clifford_index']}]"


def test_trace_measurements_are_marked_random():
    cfg = CircuitConfig(graph_spec="cycle", n=8, p=1.0, depth_factor=2)  # p=1 -> guaranteed meas
    tr = trace_circuit(cfg, max_layers=3)
    meas = [o for s in tr.stages if s.kind == "layer" for o in s.ops if o.kind == "measure"]
    assert meas, "expected measurements at p=1"
    for op in meas:
        assert op.label == "MZ" and op.random is True  # p=1 -> all candidates fire
        assert op.detail["basis"] == "Z" and op.detail["fired"] is True


def test_trace_emits_candidate_ops_with_fired_flag():
    # bernoulli at moderate p: candidates = all n qubits; some fire, some don't.
    cfg = CircuitConfig(
        graph_spec="cycle", n=8, measurement_mode="bernoulli", p=0.4, depth_factor=2
    )
    tr = trace_circuit(cfg, sample_seed=0, max_layers=3)
    layer = next(s for s in tr.stages if s.kind == "layer")
    meas = [o for o in layer.ops if o.kind == "measure"]
    assert sorted(o.qubits[0] for o in meas) == list(range(8))  # all qubits are candidates
    fired = [o for o in meas if o.detail["fired"]]
    not_fired = [o for o in meas if not o.detail["fired"]]
    assert fired and not_fired  # a mix at p=0.4
    assert all(o.label == "MZ" for o in fired)
    assert all(o.label == "MZ?" for o in not_fired)


def test_trace_candidates_superset_of_fired_all_modes():
    for mode in ("bernoulli", "gated", "random_pair"):
        cfg = CircuitConfig(graph_spec="cycle", n=8, measurement_mode=mode, p=0.5, depth_factor=2)
        tr = trace_circuit(cfg, sample_seed=2, max_layers=4)
        for s in tr.stages:
            if s.kind != "layer":
                continue
            cands = {o.qubits[0] for o in s.ops if o.kind == "measure"}
            fired = {o.qubits[0] for o in s.ops if o.kind == "measure" and o.detail["fired"]}
            assert fired <= cands


def test_render_text_shows_fired_and_candidates():
    cfg = CircuitConfig(
        graph_spec="cycle", n=8, measurement_mode="random_pair", p=0.5, depth_factor=2
    )
    out = inspect_circuit(cfg, sample_seed=0, max_layers=2)
    assert "fired [" in out and "candidates [" in out


def test_trace_random_edge_shows_m_gates():
    cfg = CircuitConfig(
        graph_spec="cycle", n=8, gating_mode="random_edge", gates_per_layer=3, p=0.1, depth_factor=1
    )
    tr = trace_circuit(cfg, max_layers=4)
    for s in tr.stages:
        if s.kind == "layer":
            assert sum(o.kind == "gate" for o in s.ops) == 3


def test_trace_reproducible_clifford_indices():
    cfg = CircuitConfig(graph_spec="cycle", n=8, p=0.2, depth_factor=2)
    a = trace_circuit(cfg, sample_seed=4, max_layers=5)
    b = trace_circuit(cfg, sample_seed=4, max_layers=5)
    idx = lambda tr: [  # noqa: E731
        o.detail.get("clifford_index") for s in tr.stages for o in s.ops if o.kind == "gate"
    ]
    assert idx(a) == idx(b)


def test_trace_no_truncation_when_max_layers_large():
    cfg = CircuitConfig(graph_spec="cycle", n=4, p=0.1, depth_factor=1)  # 4 layers
    tr = trace_circuit(cfg, max_layers=100)
    assert tr.truncated is False
    assert tr.shown_layers == tr.total_layers == 4


def test_trace_includes_warmup_when_present():
    cfg = CircuitConfig(
        graph_spec="cycle", n=8, picture="purification", p=0.1, depth_factor=2, warmup_layers=3
    )
    tr = trace_circuit(cfg, max_layers=5, include_warmup=True)
    assert sum(s.kind == "warmup" for s in tr.stages) == 3
    # warmup layers are gate-only (no measurements)
    for s in tr.stages:
        if s.kind == "warmup":
            assert all(o.kind != "measure" for o in s.ops)
    # and can be hidden
    tr2 = trace_circuit(cfg, max_layers=5, include_warmup=False)
    assert sum(s.kind == "warmup" for s in tr2.stages) == 0


# ----------------------------------------------------------------------
# render_text: one view over the IR
# ----------------------------------------------------------------------


def test_render_text_contains_key_markers():
    cfg = CircuitConfig(graph_spec="cycle", n=8, picture="single_ref", p=0.16, depth_factor=4)
    out = inspect_circuit(cfg, sample_seed=0, max_layers=4)
    assert isinstance(out, str)
    assert "single_ref" in out
    assert "setup (deterministic)" in out
    assert "Bell pair" in out  # deterministic construction shown
    assert "random Sp(4)" in out  # random gates flagged
    assert "measure Z" in out
    assert "exp. gate:meas" in out  # the pre-run ratio is surfaced
    assert "more measured layers not shown" in out  # truncation note


def test_render_text_pure_state_setup_is_trivial():
    cfg = CircuitConfig(graph_spec="cycle", n=8, p=0.1, depth_factor=1)
    out = inspect_circuit(cfg, max_layers=2)
    assert "setup (none" in out  # |0..0>, no construction gates


def test_render_all_pictures_and_modes_render():
    # smoke: every picture x gating x measurement renders without error
    for picture in ("pure_state", "purification", "single_ref"):
        for gating, extra in (
            ("brickwork", {}),
            ("random_edge", {"gates_per_layer": 2}),
            ("random_pool", {"gates_per_layer": 2}),
            ("all_edges", {}),
        ):
            for meas in ("bernoulli", "gated", "random_pair"):
                cfg = CircuitConfig(
                    graph_spec="cycle",
                    n=8,
                    picture=picture,
                    gating_mode=gating,
                    measurement_mode=meas,
                    p=0.2,
                    depth_factor=1,
                    **extra,
                )
                out = inspect_circuit(cfg, max_layers=3)
                assert "Circuit inspection" in out


# ----------------------------------------------------------------------
# unknown op kinds render gracefully (extensibility)
# ----------------------------------------------------------------------


def test_unknown_op_kind_renders_via_fallback():
    from sparsegf2.circuits.inspector import CircuitTrace, Stage, render_text

    op = Op(kind="reset", qubits=(2,), label="R0", random=False)
    tr = CircuitTrace(
        summary={"picture": "demo"},
        stages=[Stage(label="layer 0", kind="layer", index=0, ops=[op])],
        total_layers=1,
        shown_layers=1,
        truncated=False,
    )
    out = render_text(tr)
    assert "reset" in out and "R0" in out  # fallback line for an unknown kind


# ----------------------------------------------------------------------
# CLI script
# ----------------------------------------------------------------------


def test_cli_script_runs():
    r = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_circuit.py",
            "--n",
            "8",
            "--picture",
            "single_ref",
            "--layers",
            "3",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    assert "single_ref" in r.stdout
    assert "Bell pair" in r.stdout


def test_cli_script_bad_config_exits_nonzero():
    r = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_circuit.py",
            "--n",
            "7",
            "--gating",
            "brickwork",
            "--matching",
            "round_robin",
        ],  # odd n -> no 1-factorization
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 2
    assert "error:" in r.stderr


# ----------------------------------------------------------------------
# Global scramble: shown in the trace, matching what the runner applies
# ----------------------------------------------------------------------


def test_scramble_qubits_single_source_of_truth():
    """scramble_qubits() is the range the runner and inspector share."""
    inc = CircuitConfig(
        graph_spec="cycle", n=16, picture="single_ref", scramble=True, depth_factor=2
    )
    exc = CircuitConfig(
        graph_spec="cycle",
        n=16,
        picture="single_ref",
        scramble=True,
        scramble_entangled_qubit=False,
        depth_factor=2,
    )
    off = CircuitConfig(
        graph_spec="cycle", n=16, picture="single_ref", scramble=False, depth_factor=2
    )
    assert list(inc.scramble_qubits()) == list(range(16))
    assert list(exc.scramble_qubits()) == list(range(15))  # q15 (ref partner) held out
    assert off.scramble_qubits() is None
    # the flag only affects single_ref
    pur = CircuitConfig(
        graph_spec="cycle",
        n=16,
        picture="purification",
        scramble=True,
        scramble_entangled_qubit=False,
        depth_factor=2,
    )
    assert list(pur.scramble_qubits()) == list(range(16))


def test_trace_includes_scramble_stage():
    """The trace emits a scramble stage as one multi-qubit block over the right
    qubits; absent when scramble is off."""
    cfg = CircuitConfig(
        graph_spec="cycle",
        n=16,
        picture="single_ref",
        scramble=True,
        scramble_entangled_qubit=False,
        depth_factor=2,
    )
    trace = trace_circuit(cfg, sample_seed=0, max_layers=2)
    scr = [s for s in trace.stages if s.kind == "scramble"]
    assert len(scr) == 1
    op = scr[0].ops[0]
    assert op.qubits == tuple(range(15)) and op.label == "scramble" and op.random
    # off -> no scramble stage
    cfg_off = CircuitConfig(graph_spec="cycle", n=16, picture="single_ref", depth_factor=2)
    assert not any(s.kind == "scramble" for s in trace_circuit(cfg_off, sample_seed=0).stages)


def test_inspect_default_seed_varies_realization():
    """sample_seed=None draws a fresh realization each call (so re-running the
    preview shows different random Cliffords); an explicit seed is reproducible."""
    cfg = CircuitConfig(graph_spec="complete", n=16, picture="single_ref", depth_factor=2)
    assert inspect_circuit(cfg, max_layers=4) != inspect_circuit(cfg, max_layers=4)
    assert inspect_circuit(cfg, sample_seed=3, max_layers=4) == inspect_circuit(
        cfg, sample_seed=3, max_layers=4
    )


def test_scramble_entangled_qubit_changes_physics():
    """include vs exclude really run different circuits, with the intended
    physics: held-out keeps the probe pair {q_{n-1}, ref} a pure local Bell pair;
    included mixes it into the bulk."""
    from sparsegf2.core.observables import entanglement_entropy
    from sparsegf2.core.symplectic import random_symplectic

    for excluded, expected in [(False, 2), (True, 0)]:
        sim, _ = setup_picture("single_ref", 16)
        k = 15 if excluded else 16
        sim.apply_clifford(random_symplectic(k, np.random.default_rng(7)), qubits=range(k))
        assert entanglement_entropy(sim, [15, 16]) == expected
