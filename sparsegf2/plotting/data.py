"""
Shared data-loading + filtering + aggregation helpers for plotting primitives.

``_load`` handles:

- single run-directory source or list of run directories
- reading ``samples.parquet`` with Hive partitioning (so ``n``, ``p`` materialize
  as columns automatically)
- left-joining every ``analysis/*.parquet`` present, with their columns
  re-prefixed by the analysis name (e.g. ``distances.d_cont``)
- applying derived-column aliases (``k_over_n``, ``d_cont_over_n``, ...)
- applying a user filter expressed as a polars SQL WHERE fragment

``_aggregate`` groups by the (x, size) pair and returns a DataFrame with
``x``, ``n``, ``mean``, ``err_low``, ``err_high``, ``n_samples`` columns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from sparsegf2.plotting.aliases import DERIVED_ALIASES, resolve_alias
from sparsegf2.plotting.errors import ERROR_METRICS, pick_error_metric


# Column-name sanitizer

def _sanitize(name: str) -> str:
    """Replace dots with underscores for SQL-safe identifiers."""
    return name.replace(".", "_")


def _apply_rename_map(expr: str, rename_map: dict) -> str:
    """Substitute column names inside a string filter expression.

    Only substitutes whole-word (identifier) matches; leaves literals alone.
    """
    import re
    if not rename_map:
        return expr
    # Replace longest first to avoid partial substitutions.
    for old in sorted(rename_map, key=len, reverse=True):
        new = rename_map[old]
        if old == new:
            continue
        # Quote dots in the pattern; match on word boundary + escape.
        pat = re.compile(r"(?<![\w.])" + re.escape(old) + r"(?![\w.])")
        expr = pat.sub(new, expr)
    return expr


# Load

def _load_one(run_dir: Path, include_analyses: bool = True) -> pl.DataFrame:
    """Load one run's samples + analysis columns into a single eager DataFrame."""
    run_dir = Path(run_dir)

    pattern = str(run_dir / "data" / "**" / "samples.parquet")
    df = pl.scan_parquet(pattern, hive_partitioning=True).collect()

    if not include_analyses:
        return df

    # Join every analysis/*.parquet column set present in each cell's analysis dir.
    extras_by_name: dict = {}
    for cell in sorted((run_dir / "data").glob("n=*/p=*")):
        try:
            n = int(cell.parent.name.split("=", 1)[1])
            p = float(cell.name.split("=", 1)[1])
        except (IndexError, ValueError):
            continue
        adir = cell / "analysis"
        if not adir.exists():
            continue
        for f in sorted(adir.glob("*.parquet")):
            name = f.stem
            try:
                df_a = pl.read_parquet(f)
            except Exception:
                continue
            if "sample_seed" not in df_a.columns:
                continue
            rename = {c: f"{name}.{c}" for c in df_a.columns if c != "sample_seed"}
            df_a = df_a.rename(rename)
            df_a = df_a.with_columns([pl.lit(n).alias("n"), pl.lit(p).alias("p")])
            extras_by_name.setdefault(name, []).append(df_a)

    for name, chunks in extras_by_name.items():
        merged = pl.concat(chunks, how="vertical_relaxed")
        df = df.join(merged, on=["n", "p", "sample_seed"], how="left")

    return df


def _load(
    source: Union[Path, str, Iterable[Union[Path, str]]],
    include_analyses: bool = True,
) -> pl.DataFrame:
    """Load one or more run directories."""
    if isinstance(source, (str, Path)):
        source = [source]
    parts = [_load_one(Path(s), include_analyses) for s in source]
    if len(parts) == 1:
        return parts[0]
    return pl.concat(parts, how="diagonal_relaxed")


# Derived columns

def _ensure_column(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Materialize an aliased derived column when missing."""
    if name in df.columns:
        return df
    expr = resolve_alias(name)
    if expr is None:
        return df
    return df.with_columns(expr)


# Filter

def _apply_filter(df: pl.DataFrame, expr: Optional[str]) -> pl.DataFrame:
    """Apply a user filter string to a DataFrame using polars' SQL engine.

    Column names containing dots (``obs.k``, ``distances.d_cont``) are
    temporarily renamed to underscore form so users can write plain SQL
    like ``"obs.k > 0"`` without backtick-quoting.
    """
    if expr is None or not expr.strip():
        return df
    rename_map = {c: _sanitize(c) for c in df.columns if "." in c}
    df2 = df.rename(rename_map) if rename_map else df
    expr_sanitized = _apply_rename_map(expr, rename_map)
    result = df2.sql(f"SELECT * FROM self WHERE {expr_sanitized}")
    inverse = {v: k for k, v in rename_map.items()}
    return result.rename(inverse) if inverse else result


# Aggregate

def _aggregate(
    df: pl.DataFrame,
    x: str,
    y: str,
    error_metric: str = "auto",
) -> pl.DataFrame:
    """Group by ``(n, x)``, return ``(x, n, mean, err_low, err_high, n_samples)``.

    ``error_metric`` follows :func:`sparsegf2.plotting.errors.pick_error_metric`
    semantics: ``"auto"`` → Wilson for binary columns, SEM for continuous.
    """
    df = _ensure_column(df, y)
    if y not in df.columns:
        raise KeyError(
            f"column {y!r} not in DataFrame "
            f"(after alias resolution). Known columns: {df.columns}"
        )
    if x not in df.columns:
        raise KeyError(f"x column {x!r} not in DataFrame. Known columns: {df.columns}")

    # Resolve the concrete error metric using the full (non-grouped) values.
    all_values = df[y].to_numpy().astype(float)
    metric_name = pick_error_metric(error_metric, all_values)
    metric_fn = ERROR_METRICS[metric_name]

    # Compute per-(n, x) aggregates.
    rows = []
    gb = df.group_by(["n", x], maintain_order=True).agg(
        pl.col(y).alias("_values")
    ).sort(["n", x])
    for row in gb.iter_rows(named=True):
        values = np.asarray(row["_values"], dtype=float)
        if values.size == 0:
            continue
        mean = float(np.nanmean(values))
        lo, hi = metric_fn(values)
        rows.append({
            "n": int(row["n"]),
            x: float(row[x]),
            "mean": mean,
            "err_low": float(lo),
            "err_high": float(hi),
            "n_samples": int(values.size),
        })
    return pl.DataFrame(rows, schema={
        "n": pl.Int64, x: pl.Float64, "mean": pl.Float64,
        "err_low": pl.Float64, "err_high": pl.Float64,
        "n_samples": pl.Int64,
    }) if rows else pl.DataFrame({
        "n": [], x: [], "mean": [], "err_low": [], "err_high": [], "n_samples": [],
    })


__all__ = ["_load", "_apply_filter", "_aggregate", "_ensure_column"]
