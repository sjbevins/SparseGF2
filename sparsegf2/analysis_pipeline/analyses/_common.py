"""
Shared types + conventions for analysis modules.

Every analysis module exposes:

- ``NAME: str``                      — registry key.
- ``OUTPUT_FILENAME: str``           — filename inside ``cell_dir/analysis/``.
- ``OUTPUT_KIND: str``               — "parquet" | "h5".
- ``DEFAULT_PARAMS: dict``           — baseline parameters merged with user overrides.
- ``EXPENSIVE: bool``                — ``True`` to be opt-outable in CLI defaults.
- ``CELL_SCOPE: bool``               — ``True`` if the analysis writes per-cell files.
- ``run_cell(ctx, params, force) -> CellRunResult``  — the compute entry point.

Run-level analyses (e.g. ``aggregates``) set ``CELL_SCOPE = False`` and
expose ``run_run(run_dir, params, force)`` instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CellContext:
    """Everything a cell-scope analysis needs to compute its output."""

    cell_dir: Path                # data/n=XXXX/p=YYYY/
    analysis_dir: Path            # cell_dir / "analysis"
    n: int
    p: float
    seeds: np.ndarray             # int64[S]
    x_stack: np.ndarray           # uint64[S, 2n, ceil(n/64)]
    z_stack: np.ndarray           # uint64[S, 2n, ceil(n/64)]

    @property
    def n_samples(self) -> int:
        return int(self.x_stack.shape[0])


@dataclass
class CellRunResult:
    """Outcome of one cell-scope analysis invocation."""

    name: str
    status: str                   # "computed" | "skipped" | "existing" | "error"
    output_path: Optional[Path] = None
    runtime_s: float = 0.0
    n_samples: int = 0
    params: dict = field(default_factory=dict)
    message: str = ""


@dataclass
class RunRunResult:
    """Outcome of one run-scope analysis invocation (e.g. aggregates)."""

    name: str
    status: str                   # "computed" | "skipped" | "existing" | "error"
    output_path: Optional[Path] = None
    runtime_s: float = 0.0
    params: dict = field(default_factory=dict)
    message: str = ""


__all__ = ["CellContext", "CellRunResult", "RunRunResult"]
