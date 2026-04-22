"""
Registry of analysis modules.

Adding a new analysis
---------------------

1. Write ``analyses/<name>.py`` with:

    NAME = "<name>"
    OUTPUT_FILENAME = "<name>.parquet" or "<name>.h5"
    OUTPUT_KIND = "parquet" | "h5"
    CELL_SCOPE = True | False
    EXPENSIVE = True | False            # False => default on; True => opt-out
    DEFAULT_PARAMS = {...}
    def run_cell(ctx: CellContext, params, force) -> CellRunResult        # cell-scope
    # OR
    def run_run(run_dir, params, force) -> RunRunResult                   # run-scope

2. Import it here and add an entry to ``ANALYSIS_REGISTRY``.

The rest of the pipeline (orchestrator, CLI, tests) picks up the new
analysis automatically.
"""
from __future__ import annotations

from sparsegf2.analysis_pipeline.analyses import (
    distances,
    weight_stats,
    entropy_profile,
    weight_spectrum,
    logical_weights,
    aggregates,
)


# Order matters: dependencies should be earlier. aggregates runs last because
# it reads every other analysis's parquet.
ANALYSIS_REGISTRY = {
    distances.NAME:        distances,
    weight_stats.NAME:     weight_stats,
    entropy_profile.NAME:  entropy_profile,
    weight_spectrum.NAME:  weight_spectrum,
    logical_weights.NAME:  logical_weights,
    aggregates.NAME:       aggregates,
}


def cell_scope_analyses():
    return {n: m for n, m in ANALYSIS_REGISTRY.items() if m.CELL_SCOPE}


def run_scope_analyses():
    return {n: m for n, m in ANALYSIS_REGISTRY.items() if not m.CELL_SCOPE}


def cheap_analyses():
    return {n: m for n, m in ANALYSIS_REGISTRY.items() if not m.EXPENSIVE}


def expensive_analyses():
    return {n: m for n, m in ANALYSIS_REGISTRY.items() if m.EXPENSIVE}


__all__ = [
    "ANALYSIS_REGISTRY",
    "cell_scope_analyses",
    "run_scope_analyses",
    "cheap_analyses",
    "expensive_analyses",
]
