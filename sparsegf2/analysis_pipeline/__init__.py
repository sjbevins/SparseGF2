"""
sparsegf2.analysis_pipeline — post-processing over a runs/<run_id>/ tree.

Walks a run directory produced by ``sparsegf2.circuits``, rehydrates each
sample's ``tableaus.h5`` payload into a live :class:`SparseGF2`, and runs a
configurable suite of analyses that write results into
``data/n=.../p=.../analysis/<name>.{parquet,h5}`` plus a run-level
``aggregates.parquet``.

Public API
----------

- :class:`AnalysisConfig`  — driver configuration.
- :func:`run_pipeline`     — main entry point; returns a :class:`PipelineReport`.
- :class:`PipelineReport`  — per-cell, per-analysis status summary.
- :data:`ANALYSIS_REGISTRY` — mapping from analysis name to its module.
- :func:`rehydrate_sim`    — recover a ``SparseGF2`` from packed tableau bits.

Adding a new analysis
---------------------

1. Create ``analyses/<name>.py`` that exports a ``run_cell(cell_dir, tableaus,
   seeds, params, force)`` function.
2. Register it in ``analyses/__init__.py::ANALYSIS_REGISTRY``.

No other files need to change.
"""
from __future__ import annotations

from sparsegf2.analysis_pipeline.config import AnalysisConfig
from sparsegf2.analysis_pipeline.orchestrator import run_pipeline, PipelineReport
from sparsegf2.analysis_pipeline.rehydrate import rehydrate_sim
from sparsegf2.analysis_pipeline.registry import (
    read_cell_registry, write_cell_registry, RegistryEntry,
)
from sparsegf2.analysis_pipeline.analyses import ANALYSIS_REGISTRY

SCHEMA_VERSION = "1.0.0"

__all__ = [
    "SCHEMA_VERSION",
    "AnalysisConfig",
    "run_pipeline",
    "PipelineReport",
    "rehydrate_sim",
    "read_cell_registry",
    "write_cell_registry",
    "RegistryEntry",
    "ANALYSIS_REGISTRY",
]
