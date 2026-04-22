"""
sparsegf2.circuits — graph-defined random Clifford circuits for MIPT / code-discovery.

This subpackage is the canonical way to configure, run, and serialize the
random-Clifford + measurement circuits used by the SparseGF2 code-discovery
platform.

Public API
----------
Configuration dataclasses:

- :class:`CircuitConfig` — per-cell knobs (graph, picture, modes, depth, p, n).
- :class:`RunConfig` — sweep-level knobs (sizes, p grid, samples, output).

Graph construction:

- :class:`GraphTopology` — graph + 1-factorization + perfect-matching sampler.
- :func:`cycle_graph`, :func:`complete_graph`, :func:`parse_graph_spec`.

Matching modes (reference doc: ``sparsegf2/circuits/README.md``):

- ``round_robin`` — deterministic cycle through a fixed 1-factorization.
- ``palette``     — uniform random draw from a fixed 1-factorization each layer.
- ``fresh``       — uniform random perfect matching of the graph each layer.

Execution:

- :class:`CircuitBuilder` — consumes a :class:`CircuitConfig`, yields a stream
  of :class:`CircuitLayer` records (gate pairs, Clifford indices, measurement qubits).
- :class:`SimulationRunner` — drives a :class:`SparseGF2` through a builder for
  a single ``(n, p, sample_seed)`` cell; returns a populated :class:`SampleRecord`.
- :class:`SweepDriver` — orchestrates a full ``sizes × p_values × samples`` sweep
  with a worker pool, calls the validator, hands results to :class:`RunWriter`.

Persistence:

- :class:`RunWriter` — writes the standardized run directory tree defined in §3
  (``manifest.json``, ``graph.g6``, Hive-partitioned
  ``data/n=.../p=.../samples.parquet`` and optional ``tableaus.h5`` /
  ``realizations.h5``).

Validation:

- :func:`validate_config` — pre-flight compatibility checker. Hard-fails on
  incompatible ``(graph, matching_mode, n)`` triples before any sample is run.
"""
from __future__ import annotations

from sparsegf2.circuits.graphs import (
    GraphTopology,
    cycle_graph,
    complete_graph,
    parse_graph_spec,
)
from sparsegf2.circuits.matching import (
    MATCHING_MODES,
    select_matching,
)
from sparsegf2.circuits.config import (
    CircuitConfig,
    RunConfig,
    SampleRecord,
)
from sparsegf2.circuits.validator import (
    validate_config,
    ValidationReport,
    CompatibilityError,
)
from sparsegf2.circuits.pictures import (
    PICTURES,
    init_picture,
)
from sparsegf2.circuits.measurements import (
    MEASUREMENT_MODES,
    sample_measurements,
)
from sparsegf2.circuits.builder import (
    CircuitBuilder,
    CircuitLayer,
)
from sparsegf2.circuits.runner import SimulationRunner
from sparsegf2.circuits.writer import RunWriter
from sparsegf2.circuits.driver import SweepDriver

SCHEMA_VERSION = "1.0.0"

__all__ = [
    "SCHEMA_VERSION",
    # graphs
    "GraphTopology",
    "cycle_graph",
    "complete_graph",
    "parse_graph_spec",
    # matching
    "MATCHING_MODES",
    "select_matching",
    # config
    "CircuitConfig",
    "RunConfig",
    "SampleRecord",
    # validation
    "validate_config",
    "ValidationReport",
    "CompatibilityError",
    # pictures
    "PICTURES",
    "init_picture",
    # measurements
    "MEASUREMENT_MODES",
    "sample_measurements",
    # builder/runner/writer/driver
    "CircuitBuilder",
    "CircuitLayer",
    "SimulationRunner",
    "RunWriter",
    "SweepDriver",
]
