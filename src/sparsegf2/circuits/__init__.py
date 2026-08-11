"""``sparsegf2.circuits``: graph-defined random Clifford + measurement circuits.

Builds MIPT-style circuits on top of the phase-free
:class:`sparsegf2.SparseGF2` core: a graph fixes who-interacts-with-whom,
a scheduler places gates and measurements layer by layer, and a runner
executes one realization into a :class:`SampleRecord`.

Golden path::

    from sparsegf2.circuits import simulate, CircuitConfig
    rec = simulate(CircuitConfig(graph_spec="cycle", n=8, p=0.16))

Implemented today: ``pure_state`` / ``purification`` / ``single_ref`` pictures;
``cycle`` / ``complete`` / ``path`` / ``lattice_2d`` / ``newman_watts`` /
``watts_strogatz`` graphs plus ``from_networkx`` for arbitrary simple undirected
geometry; ``brickwork`` /
``random_edge`` / ``random_pool`` / ``all_edges`` gating; ``round_robin`` / ``palette`` /
``fresh`` matchings; ``bernoulli`` / ``gated`` / ``random_pair`` /
``uniform_count`` measurements; ``O(n)`` / ``O(log_n)`` / ``until_purified``
depth. Everything is live code: new pictures / graphs / modes are small local
edits at the ``extension point`` comments, not pre-committed stubs.

No runtime Stim: the two-qubit Clifford table is built natively from
:func:`sparsegf2.core.symplectic.enumerate_sp4`.
"""

from sparsegf2.circuits.config import CircuitConfig
from sparsegf2.circuits.draw import circuit_to_quantikz, draw_circuit, save_circuit
from sparsegf2.circuits.graphs import (
    GraphTopology,
    complete_graph,
    cycle_graph,
    from_networkx,
    from_spec,
    graph6_encode,
    lattice_2d,
    newman_watts,
    path_graph,
    watts_strogatz,
)
from sparsegf2.circuits.inspector import (
    CircuitTrace,
    Stage,
    inspect_circuit,
    render_text,
    trace_circuit,
)
from sparsegf2.circuits.matching import (
    MATCHING_MODES,
    available_modes,
    select_matching,
)
from sparsegf2.circuits.measurements import (
    MEASUREMENT_MODES,
    sample_measurements,
    sample_measurements_detailed,
)
from sparsegf2.circuits.ops import DIRECTED_GATES, NAMED_GATES, Op, apply_named_gate
from sparsegf2.circuits.picture import Picture, PictureSpec, setup_ops, setup_picture
from sparsegf2.circuits.records import SampleRecord
from sparsegf2.circuits.runner import CHECKPOINT_STOP, SimulationRunner, simulate
from sparsegf2.circuits.scheduler import CircuitBuilder, CircuitLayer

__all__ = [
    # picture
    "Picture",
    "PictureSpec",
    "setup_picture",
    "setup_ops",
    # circuit inspection (trace IR + renderers)
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
]
