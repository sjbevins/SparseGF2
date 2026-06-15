"""Persisted, augmentable studies: keyed rows + saved tableaux on disk.

A :class:`Study` is a directory holding three files:

* ``rows.parquet``  : one **keyed** row per swept cell (the varied parameters,
  the run diagnostics, the fixed observables, and any scalar analyses).
* ``tableaux.h5``   : each cell's final ``(2N, 2N)`` tableau (optional), keyed
  identically to the rows.
* ``manifest.json`` : the grid, seeds, base config, picture, and the list of
  analyses computed so far.

The point is the **augment workflow**. Two ways to get data, both first class:

1. *Online*: :meth:`Study.run` with ``analyses=[...]`` computes the observables
   at the end of each circuit and discards the tableau (memory-light). Use this
   when you already know what you want to measure.
2. *Offline*: :meth:`Study.run` with ``save_tableaux=True`` keeps every final
   tableau. Later, :meth:`Study.augment` reconstructs each tableau, computes
   **new** observables (e.g. ``code_rate``, ``contiguous_distance``), and merges
   the new columns into ``rows.parquet`` keyed by the cell, with no re-simulation. A
   plotting script then picks up the new columns automatically.

The merge is "smart": :meth:`augment` only computes analyses whose columns are
not already present (unless ``force=True``), and joins them on the cell key, so
re-running it is cheap and idempotent.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2.analysis import io
from sparsegf2.analysis._logging import crash_log
from sparsegf2.analysis._progress import make_bar
from sparsegf2.analysis.registry import Analysis, resolve_analyses
from sparsegf2.analysis.sweep import _is_array, _jsonsafe, _task_key, count_tasks, sweep
from sparsegf2.circuits.config import CircuitConfig
from sparsegf2.circuits.picture import setup_picture
from sparsegf2.core.sparse_tableau import SparseGF2
from sparsegf2.errors import InvalidArgumentError

_log = logging.getLogger(__name__)

Analyses = Sequence[str | Analysis] | Mapping[str, Analysis] | None


class Study:
    """A persisted, augmentable sweep on disk (see the module docstring)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rows: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # run + persist
    # ------------------------------------------------------------------

    @classmethod
    def run(
        cls,
        path: str | Path,
        base_config: CircuitConfig,
        grid: Mapping[str, Sequence[Any]] | None = None,
        seeds: Sequence[int] | int = 1,
        *,
        analyses: Analyses = None,
        save_tableaux: bool = True,
        n_jobs: int = 1,
        progress: bool = False,
        log: bool = True,
        log_detail: str = "info",
    ) -> Study:
        """Run a sweep and persist it as a :class:`Study` at ``path``.

        ``save_tableaux=True`` (default) keeps every final tableau so the study
        can be :meth:`augment`-ed later; set it to ``False`` for an online-only
        study that just keeps the computed rows.

        ``progress=True`` shows a **single** live bar for the whole study: it
        advances through every simulated cell (naming the current ``n``/``p``/
        seed) and then through the tableau write, as one 0→100% meter, so the run
        never goes silent and the bar means "fraction of the whole study done".
        ``log=True`` (default) writes a verbose trace to ``<path>/logs`` that is
        kept **only if the run does not finish** (a clean run deletes it);
        ``log_detail="trace"`` adds a per-layer record (the log is a file, not
        console text, so it never floods the notebook).
        """
        study = cls(path)
        study.path.mkdir(parents=True, exist_ok=True)
        with crash_log(study.path / "logs", label="study", detail=log_detail, enabled=log):
            _log.info(
                "Study.run -> %s | grid=%s | seeds=%s | save_tableaux=%s | n_jobs=%d",
                study.path,
                {k: list(v) for k, v in dict(grid or {}).items()},
                seeds,
                save_tableaux,
                n_jobs,
            )
            # One whole-study bar: a step per simulated cell, then a step per
            # tableau written. Shared with sweep + write_tableaux_hdf5 so there is
            # exactly one continuous meter (no second bar, no silent tail). The
            # cell count comes from sweep.count_tasks so it tracks _expand exactly.
            n_cells = count_tasks(grid, seeds)
            total_steps = n_cells + (n_cells if save_tableaux else 0)
            bar = make_bar(total_steps, desc="study", unit="step", progress=progress)
            shared = bar if bar is not None else False
            try:
                res = sweep(
                    base_config,
                    grid,
                    seeds,
                    analyses=analyses,
                    collect_tableaus=save_tableaux,
                    n_jobs=n_jobs,
                    progress=shared,
                )
                study.rows = res.rows
                study.manifest = {
                    # grid values can be rich objects (e.g. GraphTopology); store
                    # their JSON-safe form, exactly as the rows do.
                    "grid": {k: [_jsonsafe(v) for v in vs] for k, vs in res.grid.items()},
                    "seeds": res.seeds,
                    "base_config": res.base_config,
                    "picture": res.base_config.get("picture"),
                    "analyses": list(res.analyses),
                    "has_tableaux": bool(res.tableaux),
                }
                _log.info(
                    "sweep done: %d rows; writing rows.parquet + manifest.json", len(study.rows)
                )
                study._save_rows()
                study._save_manifest()
                if res.tableaux:
                    _log.info("writing %d tableaux to tableaux.h5", len(res.tableaux))
                    io.write_tableaux_hdf5(
                        res.tableaux, study.path / "tableaux.h5", progress=shared
                    )
                _log.info("Study.run complete: %s", study.path)
            finally:
                if bar is not None:
                    bar.close()
        return study

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path) -> Study:
        """Open an existing study directory."""
        study = cls(path)
        rows_path = study.path / "rows.parquet"
        if not rows_path.exists():
            raise InvalidArgumentError(f"{study.path}: not a study (no rows.parquet)")
        study.rows, _ = io.read_parquet(rows_path)
        manifest_path = study.path / "manifest.json"
        study.manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        )
        return study

    # ------------------------------------------------------------------
    # augment: compute new analyses on the saved tableaux, merge by key
    # ------------------------------------------------------------------

    def augment(
        self,
        new_analyses: Analyses,
        *,
        force: bool = False,
        progress: bool = False,
        log: bool = True,
        log_detail: str = "info",
    ) -> Study:
        """Compute ``new_analyses`` on the saved tableaux and merge new columns.

        Reconstructs each cell's tableau, runs the requested analyses, and writes
        the **scalar** results back into ``rows.parquet`` as ``a.<name>`` columns
        keyed by the cell (array-valued analyses go to a ``arrays.npz`` sidecar).
        Only analyses whose columns are not already present are computed, unless
        ``force=True``. Idempotent.

        ``progress=True`` shows a bar over the per-cell reconstruct+analyze loop
        (the silent phase you hit after ``Study.run``); ``log`` / ``log_detail``
        behave as in :meth:`run`, writing a crash-only trace to ``<path>/logs``.
        """
        tab_path = self.path / "tableaux.h5"
        if not tab_path.exists():
            raise InvalidArgumentError(
                "this study has no saved tableaux to augment from; "
                "re-run Study.run(..., save_tableaux=True)"
            )
        resolved = resolve_analyses(new_analyses)
        already = set(self.manifest.get("analyses", []))
        if not force:
            resolved = {name: fn for name, fn in resolved.items() if name not in already}
        if not resolved:
            return self  # nothing new to compute

        with crash_log(self.path / "logs", label="augment", detail=log_detail, enabled=log):
            _log.info("Study.augment: %s on %d rows", list(resolved), len(self.rows))
            tableaux = io.read_tableaux_hdf5(tab_path)
            grid_keys = list(self.manifest.get("grid", {}).keys())
            picture = self.manifest.get("picture") or self.manifest["base_config"].get("picture")
            base_n = self.manifest["base_config"].get("n")
            spec_cache: dict[int, Any] = {}
            arrays: dict[str, np.ndarray] = {}

            bar = make_bar(len(self.rows), desc="augment", unit="cell", progress=progress)
            for i, row in enumerate(self.rows):
                overrides = {k: row[k] for k in grid_keys}
                key = _task_key(overrides, int(row["sample_seed"]))
                if bar is not None:
                    bar.set_postfix({**overrides, "seed": row["sample_seed"]}, refresh=True)
                symp = tableaux.get(key)
                if symp is None:
                    _log.debug(
                        "cell %d/%d: no saved tableau for key=%s", i + 1, len(self.rows), key
                    )
                    if bar is not None:
                        bar.update(1)
                    continue
                n_sys = int(row.get("n", base_n))
                if n_sys not in spec_cache:
                    spec_cache[n_sys] = setup_picture(picture, n_sys)[1]
                sim = SparseGF2.from_symplectic(np.asarray(symp, dtype=np.uint8))
                for name, fn in resolved.items():
                    val = fn(sim, spec_cache[n_sys])
                    if _is_array(val):
                        arrays[f"{name}|{key}"] = np.asarray(val)
                    else:
                        row[f"a.{name}"] = _jsonsafe(val)
                _log.debug("cell %d/%d augmented: %s", i + 1, len(self.rows), key)
                if bar is not None:
                    bar.update(1)
            if bar is not None:
                bar.close()

            self.manifest["analyses"] = list(dict.fromkeys(list(already) + list(resolved)))
            _log.info("augment done: merging %d new columns into rows.parquet", len(resolved))
            self._save_rows()
            self._save_manifest()
            if arrays:
                existing = (
                    io.read_tableaux_npz(self.path / "arrays.npz")
                    if (self.path / "arrays.npz").exists()
                    else {}
                )
                existing.update(arrays)
                io.write_tableaux_npz(existing, self.path / "arrays.npz")
        return self

    # ------------------------------------------------------------------
    # views
    # ------------------------------------------------------------------

    def to_pandas(self):
        """Return the rows as a pandas DataFrame (lazy pandas import)."""
        import pandas as pd

        return pd.DataFrame(self.rows)

    def analysis_columns(self) -> list[str]:
        """The analysis columns (``a.<name>``) currently in the rows."""
        return sorted({c for r in self.rows for c in r if c.startswith("a.")})

    @property
    def has_tableaux(self) -> bool:
        return (self.path / "tableaux.h5").exists()

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return (
            f"Study(path={str(self.path)!r}, rows={len(self.rows)}, "
            f"tableaux={self.has_tableaux}, analyses={self.manifest.get('analyses', [])})"
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _save_rows(self) -> None:
        io.write_parquet(self.rows, self.path / "rows.parquet", manifest=self.manifest)

    def _save_manifest(self) -> None:
        (self.path / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )


__all__ = ["Study"]
