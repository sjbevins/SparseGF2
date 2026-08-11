"""Lock the circuits package public surface (mirror of the core's API test)."""

from __future__ import annotations

import sparsegf2.circuits as circuits

EXPECTED_ALL = {
    # picture
    "Picture",
    "PictureSpec",
    "setup_picture",
    "setup_ops",
    # inspection
    "Op",
    "apply_named_gate",
    "NAMED_GATES",
    "DIRECTED_GATES",
    "Stage",
    "CircuitTrace",
    "trace_circuit",
    "render_text",
    "inspect_circuit",
    "draw_circuit",
    "save_circuit",
    "circuit_to_quantikz",
    # graphs
    "GraphTopology",
    "cycle_graph",
    "complete_graph",
    "from_networkx",
    "from_spec",
    "graph6_encode",
    "lattice_2d",
    "newman_watts",
    "path_graph",
    "watts_strogatz",
    # matching
    "MATCHING_MODES",
    "select_matching",
    "available_modes",
    # measurements
    "MEASUREMENT_MODES",
    "sample_measurements",
    "sample_measurements_detailed",
    # scheduler
    "CircuitLayer",
    "CircuitBuilder",
    # config + records
    "CircuitConfig",
    "SampleRecord",
    # runner
    "CHECKPOINT_STOP",
    "SimulationRunner",
    "simulate",
}


def test_all_matches_expected():
    assert set(circuits.__all__) == EXPECTED_ALL


def test_every_name_in_all_is_importable():
    for name in circuits.__all__:
        assert hasattr(circuits, name), f"missing public symbol: {name}"


def test_no_duplicate_names_in_all():
    assert len(circuits.__all__) == len(set(circuits.__all__))
