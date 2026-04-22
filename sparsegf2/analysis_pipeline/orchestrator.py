"""
Orchestrator — walks a run directory, dispatches cells to analyses.

- Enumerate every ``data/n=XXXX/p=YYYY/`` cell.
- For each cell, load ``tableaus.h5`` once, build a :class:`CellContext`,
  then dispatch every active cell-scope analysis.
- After all cells are processed, run every active run-scope analysis (the
  ``aggregates`` module).
- Emit a :class:`PipelineReport` summarizing what was computed, skipped,
  already-present, or errored.

Parallelism is across cells via :class:`ProcessPoolExecutor`; a single
cell is processed serially across its analyses.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np

from sparsegf2.analysis_pipeline.analyses import (
    ANALYSIS_REGISTRY, cell_scope_analyses, run_scope_analyses,
)
from sparsegf2.analysis_pipeline.analyses._common import (
    CellContext, CellRunResult, RunRunResult,
)
from sparsegf2.analysis_pipeline.config import AnalysisConfig


# Report

@dataclass
class PipelineReport:
    run_dir: Path
    cell_results: List[CellRunResult] = field(default_factory=list)
    run_results: List[RunRunResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    wall_seconds: float = 0.0

    def summary(self) -> str:
        lines: List[str] = [f"Run: {self.run_dir}"]
        counts: Dict[Tuple[str, str], int] = {}
        for r in self.cell_results:
            counts[(r.name, r.status)] = counts.get((r.name, r.status), 0) + 1
        lines.append("")
        lines.append("Cell-scope analyses:")
        names = sorted({k[0] for k in counts})
        for name in names:
            statuses = {s: counts[(n, s)] for (n, s) in counts
                        for _ in [()] if n == name
                        for n_, s_ in [(n, s)] if n_ == name}
            # simpler: reconstruct per-name status counts directly
            status_counts = {s: counts.get((name, s), 0)
                             for s in ("computed", "existing", "skipped", "error")}
            lines.append(f"  {name}: " + "  ".join(
                f"{s}={c}" for s, c in status_counts.items() if c
            ))
        lines.append("")
        lines.append("Run-scope analyses:")
        for r in self.run_results:
            lines.append(f"  {r.name}: {r.status} ({r.runtime_s:.2f}s) "
                         f"{r.message}")
        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for e in self.errors:
                lines.append(f"  - {e}")
        lines.append(f"\nTotal wall: {self.wall_seconds:.2f}s")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# Cell discovery

def _discover_cells(
    run_dir: Path,
    sizes: Optional[List[int]] = None,
    p_values: Optional[List[float]] = None,
) -> List[Path]:
    cells = sorted((run_dir / "data").glob("n=*/p=*"))
    out: List[Path] = []
    p_tol = 1e-9
    for c in cells:
        try:
            n = int(c.parent.name.split("=", 1)[1])
            p = float(c.name.split("=", 1)[1])
        except (IndexError, ValueError):
            continue
        if sizes is not None and n not in sizes:
            continue
        if p_values is not None and not any(abs(p - q) < p_tol for q in p_values):
            continue
        out.append(c)
    return out


def _load_cell_tableaus(cell_dir: Path) -> Optional[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Return (n, seeds, x_stack, z_stack) or None if no tableaus.h5."""
    tpath = cell_dir / "tableaus.h5"
    if not tpath.exists():
        return None
    with h5py.File(tpath, "r") as f:
        n = int(f.attrs["n"])
        seeds = np.asarray(f["sample_seed"][:], dtype=np.int64)
        x_stack = np.asarray(f["x_packed"][:], dtype=np.uint64)
        z_stack = np.asarray(f["z_packed"][:], dtype=np.uint64)
    return n, seeds, x_stack, z_stack


# Worker for per-cell processing

def _process_cell(
    cell_dir_str: str,
    active_names: List[str],
    params_by_name: Dict[str, dict],
    force_set: List[str],
) -> List[CellRunResult]:
    """Run every active cell-scope analysis on one cell (worker entry point)."""
    from sparsegf2.analysis_pipeline.analyses import ANALYSIS_REGISTRY
    cell_dir = Path(cell_dir_str)
    results: List[CellRunResult] = []

    loaded = _load_cell_tableaus(cell_dir)
    if loaded is None:
        # No tableaus.h5 -- can't run anything here
        for name in active_names:
            results.append(CellRunResult(
                name=name, status="skipped", output_path=None,
                n_samples=0, params=params_by_name.get(name, {}),
                message="tableaus.h5 missing; nothing to analyze",
            ))
        return results

    n, seeds, x_stack, z_stack = loaded
    try:
        p = float(cell_dir.name.split("=", 1)[1])
    except (IndexError, ValueError):
        p = float("nan")
    ctx = CellContext(
        cell_dir=cell_dir,
        analysis_dir=cell_dir / "analysis",
        n=n, p=p, seeds=seeds,
        x_stack=x_stack, z_stack=z_stack,
    )

    for name in active_names:
        mod = ANALYSIS_REGISTRY[name]
        if not mod.CELL_SCOPE:
            continue
        params = params_by_name.get(name, {})
        force = name in force_set
        try:
            res = mod.run_cell(ctx, params=params, force=force)
            results.append(res)
        except Exception as e:
            results.append(CellRunResult(
                name=name, status="error", output_path=None,
                n_samples=ctx.n_samples, params=params,
                message=f"{type(e).__name__}: {e}",
            ))
    return results


# Main entry

def _active_cell_analyses(cfg: AnalysisConfig) -> List[str]:
    cell_mods = cell_scope_analyses()
    active: List[str] = []
    for name, mod in cell_mods.items():
        if not cfg.is_selected(name):
            continue
        active.append(name)
    return active


def _active_run_analyses(cfg: AnalysisConfig) -> List[str]:
    run_mods = run_scope_analyses()
    return [name for name in run_mods if cfg.is_selected(name)]


def run_pipeline(cfg: AnalysisConfig) -> PipelineReport:
    """Top-level driver."""
    t_all = time.perf_counter()
    report = PipelineReport(run_dir=cfg.run_dir)

    # Cell-level pass
    active_cell = _active_cell_analyses(cfg)
    cells = _discover_cells(cfg.run_dir, sizes=cfg.sizes, p_values=cfg.p_values)
    if not cells:
        report.errors.append("no cells matched the filter")

    params_by_name = dict(cfg.params)
    force_set = list(cfg.force or [])

    def _promote_cell_errors(cell: Path, results: List[CellRunResult]) -> None:
        """Surface per-analysis 'error' statuses as top-level report.errors.

        Without this promotion, a test like ``assert not report.errors``
        cannot detect an individual analysis that raised — the exception
        would live only inside the ``CellRunResult.message`` of a single
        ``status="error"`` entry.
        """
        for r in results:
            if r.status == "error":
                report.errors.append(
                    f"{cell.parent.name}/{cell.name} [{r.name}] {r.message}"
                )

    if cfg.n_workers <= 1:
        for cell in cells:
            results = _process_cell(str(cell), active_cell, params_by_name, force_set)
            report.cell_results.extend(results)
            _promote_cell_errors(cell, results)
            if cfg.verbose:
                _log_cell(cell, results)
    else:
        with ProcessPoolExecutor(max_workers=cfg.n_workers) as pool:
            fut_to_cell = {
                pool.submit(_process_cell, str(c), active_cell,
                            params_by_name, force_set): c
                for c in cells
            }
            for fut in as_completed(fut_to_cell):
                cell = fut_to_cell[fut]
                try:
                    results = fut.result()
                    report.cell_results.extend(results)
                    _promote_cell_errors(cell, results)
                    if cfg.verbose:
                        _log_cell(cell, results)
                except Exception as e:
                    report.errors.append(
                        f"cell {cell} failed: {type(e).__name__}: {e}"
                    )

    # Run-level pass (serial; always after cells finish)
    for name in _active_run_analyses(cfg):
        mod = ANALYSIS_REGISTRY[name]
        params = params_by_name.get(name, {})
        force = name in force_set
        try:
            res = mod.run_run(cfg.run_dir, params=params, force=force)
            report.run_results.append(res)
            if cfg.verbose:
                print(f"[{name}] {res.status} ({res.runtime_s:.2f}s) {res.message}",
                      flush=True)
        except Exception as e:
            report.errors.append(f"{name}: {type(e).__name__}: {e}")

    # Update run manifest with analysis_summary
    _update_manifest_summary(cfg.run_dir)

    report.wall_seconds = time.perf_counter() - t_all
    return report


# Manifest summary update

def _update_manifest_summary(run_dir: Path) -> None:
    """Scan the tree and update manifest.json::analysis_summary."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return

    summary: Dict[str, List[str]] = {}
    cells = sorted((run_dir / "data").glob("n=*/p=*"))
    for cell in cells:
        key = f"{cell.parent.name}/{cell.name}"
        analyses: List[str] = []
        adir = cell / "analysis"
        if adir.exists():
            for f in adir.iterdir():
                if f.name in ("_registry.json",):
                    continue
                if f.suffix in (".parquet", ".h5"):
                    analyses.append(f.stem)
        summary[key] = sorted(analyses)
    run_level: List[str] = []
    if (run_dir / "aggregates.parquet").exists():
        run_level.append("aggregates")
    manifest["analysis_summary"] = {
        "per_cell": summary,
        "run_level": sorted(run_level),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False))


# Logging

def _log_cell(cell: Path, results: Iterable[CellRunResult]) -> None:
    cell_name = f"{cell.parent.name}/{cell.name}"
    parts = []
    for r in results:
        mark = {"computed": "+", "existing": "=",
                "skipped": "-", "error": "!"}.get(r.status, "?")
        parts.append(f"{mark}{r.name}({r.runtime_s:.2f}s)")
    print(f"  [{cell_name}] " + " ".join(parts), flush=True)


__all__ = ["PipelineReport", "run_pipeline"]
