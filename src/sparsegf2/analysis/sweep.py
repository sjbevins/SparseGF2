"""Parameter-sweep driver over circuit configurations.

:func:`sweep` runs the Cartesian product of a parameter ``grid`` (config fields →
value lists) crossed with ``seeds``, executing one circuit per cell. Two modes,
matching the analysis architecture:

* **online** (default): each cell computes its requested ``analyses`` at the end
  of the circuit and the tableau is discarded, so only a flat row of numbers is
  kept. Memory is O(number of cells).
* **offline** (``collect_tableaus=True``): each cell's final ``(2N, 2N)`` tableau
  is retained (keyed by its parameters + seed) for later analysis via
  :func:`sparsegf2.analysis.analyze_tableaux`.

Results come back as a :class:`SweepResult` with ``.rows`` (one dict per cell),
optional ``.tableaux``, ``.array_results`` (de-flattened array-valued analyses),
and helpers to/from pandas and parquet/HDF5.

Reproducibility: a cell is fully determined by ``(config, sample_seed)``. Sweep
seeds map directly to ``sample_seed``, so a row is identical across re-runs and
across ``n_jobs`` values.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2.circuits.config import CircuitConfig
from sparsegf2.circuits.runner import simulate
from sparsegf2.errors import InvalidArgumentError

_log = logging.getLogger(__name__)

Analyses = (
    Sequence[str | Callable[[Any, Any], Any]] | Mapping[str, Callable[[Any, Any], Any]] | None
)


# ----------------------------------------------------------------------
# Provenance + JSON/columnar-safe coercion
# ----------------------------------------------------------------------


def _jsonsafe(v: Any) -> Any:
    """Coerce a value to something parquet/JSON can store."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if callable(v):
        return getattr(v, "__name__", repr(v))
    return getattr(v, "name", str(v))


def _config_provenance(cfg: CircuitConfig) -> dict[str, Any]:
    """A JSON-safe snapshot of a config for the sweep manifest."""
    return {f.name: _jsonsafe(getattr(cfg, f.name)) for f in fields(cfg)}


def _task_key(overrides: Mapping[str, Any], seed: int) -> str:
    """A filesystem/HDF5-safe key identifying one swept cell."""
    parts = [f"{k}={_jsonsafe(v)}" for k, v in overrides.items()] + [f"seed={seed}"]
    return "__".join(parts).replace("/", "-") or f"seed={seed}"


def _is_array(v: Any) -> bool:
    return isinstance(v, (np.ndarray, list, tuple))


# ----------------------------------------------------------------------
# Grid expansion + per-task execution
# ----------------------------------------------------------------------


def _validate_grid(base_config: CircuitConfig, grid: Mapping[str, Sequence[Any]]) -> None:
    valid = {f.name for f in fields(base_config)}
    bad = [k for k in grid if k not in valid]
    if bad:
        raise InvalidArgumentError(
            f"grid keys are not CircuitConfig fields: {bad}; valid: {sorted(valid)}"
        )


def _expand(
    base_config: CircuitConfig,
    grid: Mapping[str, Sequence[Any]],
    seeds: Sequence[int],
) -> list[tuple[CircuitConfig, dict[str, Any], int]]:
    keys = list(grid)
    tasks: list[tuple[CircuitConfig, dict[str, Any], int]] = []
    combos = itertools.product(*(list(grid[k]) for k in keys)) if keys else [()]
    for combo in combos:
        overrides = dict(zip(keys, combo, strict=True))
        cfg = replace(base_config, **overrides) if overrides else base_config  # re-validates
        for s in seeds:
            tasks.append((cfg, overrides, int(s)))
    return tasks


def count_tasks(grid: Mapping[str, Sequence[Any]] | None, seeds: Sequence[int] | int) -> int:
    """Number of cells :func:`_expand` produces: ``(product of grid sizes) x #seeds``.

    The single source of truth for the cell count, used by :meth:`Study.run` to
    size its whole-study progress bar so the total cannot drift from the actual
    task count.
    """
    n = seeds if isinstance(seeds, int) else len(list(seeds))
    for vals in dict(grid or {}).values():
        n *= len(list(vals))
    return n


def _run_task_indexed(
    idx: int,
    cfg: CircuitConfig,
    overrides: dict[str, Any],
    seed: int,
    analyses: Analyses,
    collect_tableaus: bool,
) -> tuple[int, tuple[dict[str, Any], dict[str, np.ndarray], tuple[str, np.ndarray] | None]]:
    """One cell, tagged with its submission index, which lets the parallel path
    consume completions out of order and still restore submission order."""
    return idx, _run_task(cfg, overrides, seed, analyses, collect_tableaus)


def _run_task(
    cfg: CircuitConfig,
    overrides: dict[str, Any],
    seed: int,
    analyses: Analyses,
    collect_tableaus: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray], tuple[str, np.ndarray] | None]:
    """Execute one cell. Module-level so it is picklable for parallel backends."""
    rec = simulate(cfg, sample_seed=seed, analyses=analyses, save_tableau=collect_tableaus)
    row: dict[str, Any] = {str(k): _jsonsafe(v) for k, v in overrides.items()}
    row.update(
        picture=str(rec.picture),
        sample_seed=rec.sample_seed,
        total_layers=rec.total_layers,
        total_gates=rec.total_gates,
        total_measurements=rec.total_measurements,
        gate_to_meas_ratio_expected=rec.gate_to_meas_ratio_expected,
        gate_to_meas_ratio_actual=rec.gate_to_meas_ratio_actual,
        code_dimension=rec.code_dimension,
        ref_entropy=rec.ref_entropy,
        entropy_half_cut=rec.entropy_half_cut,
        purified_at_layer=rec.purified_at_layer,
        mean_active_generators=rec.mean_active_generators,
        runtime_total_s=rec.runtime_total_s,
    )
    arrays: dict[str, np.ndarray] = {}
    for name, val in rec.analyses.items():
        if _is_array(val):
            arrays[name] = np.asarray(val)
        else:
            row[f"a.{name}"] = _jsonsafe(val)
    tab = (
        (_task_key(overrides, seed), rec.final_tableau)
        if (collect_tableaus and rec.final_tableau is not None)
        else None
    )
    return row, arrays, tab


def _cell_label(overrides: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """The swept parameters of one cell, for the progress bar / log line."""
    label = {str(k): _jsonsafe(v) for k, v in overrides.items()}
    label["seed"] = seed
    return label


def _map_tasks(tasks, analyses, collect_tableaus, n_jobs, progress):
    """Run every cell, driving a bar whose postfix names the running cell's
    ``n``/``p``/seed (serial) or the most recently completed cell (parallel,
    where completions arrive out of order), and logging each cell so a crash log
    shows exactly where a run died."""
    from sparsegf2.analysis._progress import resolve_bar
    from sparsegf2.analysis._runtime import resolve_n_jobs

    total = len(tasks)
    # Turn a negative n_jobs (-1 = all cores) into a concrete count using the
    # scheduler allocation, so an HPC job filling part of a node is not
    # oversubscribed. Non-negative values pass through unchanged.
    n_jobs = resolve_n_jobs(n_jobs)
    # `progress` may be a bool, or a shared bar handed down by Study.run so the
    # simulation and tableau-write phases read as one whole-study meter.
    bar, owned = resolve_bar(progress, total=total, desc="simulating", unit="cell")
    try:
        if n_jobs == 1:
            results = []
            for i, (cfg, ov, s) in enumerate(tasks):
                label = _cell_label(ov, s)
                _log.info("cell %d/%d: %s", i + 1, total, label)
                if bar is not None:
                    bar.set_postfix(label, refresh=True)
                results.append(_run_task(cfg, ov, s, analyses, collect_tableaus))
                if bar is not None:
                    bar.update(1)
            return results

        try:
            from joblib import Parallel, delayed
        except ImportError as e:  # pragma: no cover
            raise InvalidArgumentError(
                "n_jobs>1 needs the optional 'parallel' extra (joblib); "
                "install with: pip install sparsegf2[parallel]"
            ) from e
        _log.info("dispatching %d cells across n_jobs=%d", total, n_jobs)
        # Yield in COMPLETION order so the bar ticks the moment any worker
        # finishes. With submission-order yield, one slow cell at the head of
        # the queue freezes the bar while finished cells pile up behind it,
        # then the bar jumps in a burst (the "stutter" on mixed-duration
        # grids). Each task carries its index so the returned rows are sorted
        # back to submission order: the output is byte-identical to the
        # ordered path. generator_unordered needs joblib >= 1.4; older joblib
        # falls back to the ordered generator.
        try:
            gen = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
                delayed(_run_task_indexed)(i, cfg, ov, s, analyses, collect_tableaus)
                for i, (cfg, ov, s) in enumerate(tasks)
            )
        except ValueError as e:
            # joblib < 1.4 (no generator_unordered): both it and a backend
            # without generator support reject return_as at construction, so
            # this is reached before any task runs. Re-raise anything else.
            if "return_as" not in str(e):  # pragma: no cover (unrelated error)
                raise
            gen = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(_run_task_indexed)(i, cfg, ov, s, analyses, collect_tableaus)
                for i, (cfg, ov, s) in enumerate(tasks)
            )
        indexed_results = []
        for done, (idx, res) in enumerate(gen):
            _cfg, ov, s = tasks[idx]
            label = _cell_label(ov, s)
            _log.info("cell %d/%d done: %s", done + 1, total, label)
            if bar is not None:
                bar.set_postfix(label, refresh=True)
            indexed_results.append((idx, res))
            if bar is not None:
                bar.update(1)
        if len(indexed_results) != total:  # pragma: no cover (backend defect guard)
            raise RuntimeError(
                f"parallel sweep returned {len(indexed_results)}/{total} cells; "
                "results would misalign with the grid/seeds provenance"
            )
        indexed_results.sort(key=lambda pair: pair[0])
        return [res for _idx, res in indexed_results]
    finally:
        if owned and bar is not None:
            bar.close()


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------


@dataclass
class SweepResult:
    """The output of a :func:`sweep`.

    Attributes
    ----------
    rows : list of dict
        One flat row per swept cell: the varied parameters, the run diagnostics,
        the fixed observables, and any *scalar* analyses (keys prefixed ``a.``).
    tableaux : dict of str -> ndarray
        ``task_key -> (2N, 2N)`` final tableau; empty unless ``collect_tableaus``.
    array_results : dict of str -> list
        Array-valued analyses (e.g. ``entropy_profile``), one entry per row.
    grid, seeds, base_config, analyses
        Provenance: the swept axes, the seeds, a JSON-safe base config, and the
        resolved analysis names.
    """

    rows: list[dict[str, Any]]
    tableaux: dict[str, np.ndarray]
    array_results: dict[str, list]
    grid: dict[str, list]
    seeds: list[int]
    base_config: dict[str, Any]
    analyses: list[str]

    def __len__(self) -> int:
        return len(self.rows)

    def to_pandas(self):
        """Return the rows as a pandas DataFrame (lazy pandas import)."""
        import pandas as pd

        return pd.DataFrame(self.rows)

    def _manifest(self) -> dict:
        return {
            "grid": self.grid,
            "seeds": self.seeds,
            "base_config": self.base_config,
            "analyses": self.analyses,
        }

    def to_parquet(self, path: str | Path) -> None:
        """Write the rows to parquet, manifest embedded (lazy pyarrow import)."""
        from sparsegf2.analysis import io

        io.write_parquet(self.rows, path, manifest=self._manifest())

    def save(self, out: str | Path) -> None:
        """Persist the sweep: rows (parquet) to ``out`` + tableaux to a sidecar file."""
        from sparsegf2.analysis import io

        out = Path(out)
        self.to_parquet(out)  # rows + manifest
        if self.tableaux:
            attrs = {k: _attrs_from_key(k) for k in self.tableaux}
            try:
                io.write_tableaux_hdf5(
                    self.tableaux, out.with_suffix(".tableaux.h5"), attrs_by_key=attrs
                )
            except InvalidArgumentError:
                io.write_tableaux_npz(self.tableaux, out.with_suffix(".tableaux.npz"))

    @classmethod
    def load(cls, path: str | Path) -> SweepResult:
        """Load rows (+ manifest) written by :meth:`save`. Tableaux not reloaded."""
        from sparsegf2.analysis import io

        rows, manifest = io.read_parquet(path)
        manifest = manifest or {}
        return cls(
            rows=rows,
            tableaux={},
            array_results={},
            grid=manifest.get("grid", {}),
            seeds=manifest.get("seeds", []),
            base_config=manifest.get("base_config", {}),
            analyses=manifest.get("analyses", []),
        )


def _attrs_from_key(key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in key.split("__"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def sweep(
    base_config: CircuitConfig,
    grid: Mapping[str, Sequence[Any]] | None = None,
    seeds: Sequence[int] | int = 1,
    *,
    analyses: Analyses = None,
    collect_tableaus: bool = False,
    n_jobs: int = 1,
    out: str | Path | None = None,
    progress: bool = False,
) -> SweepResult:
    """Run a parameter sweep and return a :class:`SweepResult`.

    Parameters
    ----------
    base_config : CircuitConfig
        The base cell; ``grid`` overrides selected fields per cell.
    grid : mapping, optional
        ``{config_field: [values]}``. Cartesian product is taken. Keys must be
        ``CircuitConfig`` field names.
    seeds : sequence of int, or int
        Sample seeds per grid point. An int ``R`` means ``range(R)`` (so ``R``
        must be ``>= 1``). An empty seed set is rejected.
    analyses : list/mapping, optional
        Named/custom analyses (see :mod:`sparsegf2.analysis`). Scalar results
        become ``a.<name>`` columns; array results go to ``array_results``.
    collect_tableaus : bool
        If true, retain each cell's final tableau for offline analysis.
    n_jobs : int
        Worker count. ``>1`` needs the ``parallel`` extra (joblib) and a
        **picklable** base config: a string ``graph_spec`` (a prebuilt
        ``GraphTopology`` carries an un-picklable matching-sampler closure) and
        an int ``gates_per_layer`` (a lambda will not pickle). Results are
        returned in deterministic task order regardless of ``n_jobs``.
    out : path, optional
        If set, persist rows (+ tableaux sidecar) to disk after the sweep.
    progress : bool
        If true, show a live ``tqdm`` progress bar over the cells (auto-selects a
        Jupyter widget vs a terminal bar). Optional: nothing is imported when
        false (the headless / HPC path), and a missing tqdm degrades to no bar
        with a warning rather than failing. Needs the ``progress`` extra
        (``pip install sparsegf2[progress]``).
    """
    grid = dict(grid or {})
    _validate_grid(base_config, grid)
    seed_list = list(range(seeds)) if isinstance(seeds, int) else [int(s) for s in seeds]
    if not seed_list:
        raise InvalidArgumentError(
            "seeds must be a positive int or a non-empty sequence of ints; "
            f"got {seeds!r} (an empty seed set runs nothing)"
        )

    tasks = _expand(base_config, grid, seed_list)
    results = _map_tasks(tasks, analyses, collect_tableaus, n_jobs, progress)

    rows = [row for row, _, _ in results]
    array_names = sorted({name for _, arrs, _ in results for name in arrs})
    array_results = {name: [arrs.get(name) for _, arrs, _ in results] for name in array_names}
    tableaux = {tab[0]: tab[1] for _, _, tab in results if tab is not None}

    from sparsegf2.analysis.registry import resolve_analyses

    result = SweepResult(
        rows=rows,
        tableaux=tableaux,
        array_results=array_results,
        grid={k: list(v) for k, v in grid.items()},
        seeds=seed_list,
        base_config=_config_provenance(base_config),
        analyses=list(resolve_analyses(analyses)),
    )
    if out is not None:
        result.save(out)
    return result


__all__ = ["SweepResult", "sweep"]
