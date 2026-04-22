"""Configuration for the analysis pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class AnalysisConfig:
    """Top-level configuration for a single invocation of :func:`run_pipeline`.

    Parameters
    ----------
    run_dir : Path
        Path to an existing ``runs/<run_id>/`` directory produced by
        ``sparsegf2.circuits``.
    only : list of str, optional
        When set, only the named analyses run (every other cell-level
        analysis is skipped). Mutually exclusive with :attr:`skip`.
    skip : list of str, optional
        When set, every default-on analysis *except* these runs. Mutually
        exclusive with :attr:`only`.
    force : list of str, optional
        Analyses to recompute even when their output file already exists.
    sizes : list of int, optional
        If provided, restrict the pass to cells whose ``n`` is in this list.
        Useful for iterative development.
    p_values : list of float, optional
        If provided, restrict the pass to cells whose ``p`` is within 1e-9
        of a listed value.
    n_workers : int
        Number of parallel worker processes. ``1`` runs inline.
    params : dict
        Per-analysis parameter overrides, keyed by analysis name. E.g.
        ``{"distances": {"d_cont_starts": [0, 0.25, 0.5, 0.75]}}``.
    verbose : bool
        Print per-cell progress lines.
    """

    run_dir: Path
    only: Optional[List[str]] = None
    skip: Optional[List[str]] = None
    force: Optional[List[str]] = None
    sizes: Optional[List[int]] = None
    p_values: Optional[List[float]] = None
    n_workers: int = 1
    params: dict = field(default_factory=dict)
    verbose: bool = True

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        if not self.run_dir.is_dir():
            raise ValueError(f"run_dir does not exist or is not a directory: {self.run_dir}")
        if self.only and self.skip:
            raise ValueError("--only and --skip are mutually exclusive")
        self.only = list(self.only) if self.only else None
        self.skip = list(self.skip) if self.skip else None
        self.force = list(self.force) if self.force else None
        self.sizes = [int(n) for n in self.sizes] if self.sizes else None
        self.p_values = [float(p) for p in self.p_values] if self.p_values else None
        if self.n_workers < 1:
            raise ValueError(f"n_workers must be >= 1; got {self.n_workers}")

    # --------------------------------------------------------------

    def is_selected(self, name: str) -> bool:
        """Whether the named analysis should run given only/skip filters."""
        if self.only is not None:
            return name in self.only
        if self.skip is not None:
            return name not in self.skip
        return True

    def should_force(self, name: str) -> bool:
        return self.force is not None and name in self.force


__all__ = ["AnalysisConfig"]
