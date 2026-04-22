"""
``aggregates`` analysis — run-level rollup at ``runs/<run_id>/aggregates.parquet``.

For every ``(n, p)`` cell it loads ``samples.parquet`` plus every available
``analysis/*.parquet``, joins on ``sample_seed``, and emits one row with:

- per-column ``mean``, ``std``, ``sem``, ``q025``, ``q500``, ``q975``
- Wilson lower/upper interval for columns whose values are in ``{0, 1}``
- the same stats conditional on ``obs.k > 0`` (suffix ``_cond_k_gt_0``)
- derived ratios: ``k_over_n``, ``d_cont_over_n``, ``d_min_over_n`` and
  their means + conditional means

The plotting primitive reads this file first when possible for speed; it
only falls back to raw samples when a filter requires sample-level data.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from sparsegf2 import __version__ as _SGF2_VERSION
from sparsegf2.analysis_pipeline.analyses._common import RunRunResult


NAME = "aggregates"
OUTPUT_FILENAME = "aggregates.parquet"
OUTPUT_KIND = "parquet"
CELL_SCOPE = False
EXPENSIVE = False
DEFAULT_PARAMS: Dict = {
    "quantiles": [0.025, 0.5, 0.975],
    "wilson_z": 1.96,
    "conditional_column": "obs.k",
    "conditional_predicate": "> 0",   # "> 0" | ">= 1" etc. — evaluated as col >  value
}


# ══════════════════════════════════════════════════════════════
# Wilson interval
# ══════════════════════════════════════════════════════════════

def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Reference: Wilson, E. B. (1927), "Probable inference, the law of
    succession, and statistical inference", Journal of the American
    Statistical Association 22 (158), 209-212. See Brown, Cai, DasGupta
    (2001), "Interval Estimation for a Binomial Proportion", Statistical
    Science 16, 101-133, for why Wilson is preferred over the normal
    approximation (particularly for small n or proportions near 0 / 1).

    Returns ``(lower, upper)`` both in ``[0, 1]``. Defined even for
    ``k ∈ {0, n}``, where it gives a proper non-degenerate interval.
    """
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = phat + (z * z) / (2.0 * n)
    half = z * math.sqrt(phat * (1.0 - phat) / n + (z * z) / (4.0 * n * n))
    return ((center - half) / denom, (center + half) / denom)


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _is_binary(series: pl.Series) -> bool:
    """Heuristic: a column is "binary" for Wilson-interval purposes iff its
    *dtype* is boolean or an integer type and every value is in ``{0, 1}``.

    Requiring the dtype to be integer prevents false positives where a
    floating-point column (e.g. ``obs.tmi``) happens to contain only ``{0, 1}``
    values in a small sample — such cases are conceptually continuous
    observations, not binomial proportions.
    """
    dtype = series.dtype
    if dtype == pl.Boolean:
        return True
    if not (dtype.is_integer() and dtype.is_numeric()):
        return False
    s = series.drop_nulls()
    if len(s) == 0:
        return False
    return bool(s.is_in([0, 1]).all())


def _numeric_cols(df: pl.DataFrame, skip: Tuple[str, ...] = ("sample_seed",)) -> List[str]:
    return [c for c in df.columns
            if c not in skip and df[c].dtype.is_numeric()]


def _cell_stats_for_column(col: str, series: pl.Series, quantiles: List[float],
                           wilson_z: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = len(series)
    nan = float("nan")
    if n == 0:
        out[f"{col}_mean"] = nan
        out[f"{col}_std"] = nan
        out[f"{col}_sem"] = nan
        for q in quantiles:
            out[f"{col}_q{int(round(q * 1000)):03d}"] = nan
        return out

    values = series.to_numpy().astype(np.float64)
    # Suppress numpy warnings from all-NaN or zero-variance columns; nan-aware
    # reductions produce the desired NaN outputs without user-facing noise.
    with np.errstate(all="ignore"):
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values, ddof=1)) if n > 1 else 0.0
        sem = std / math.sqrt(n) if n > 1 and not math.isnan(std) else 0.0
        out[f"{col}_mean"] = mean
        out[f"{col}_std"] = std
        out[f"{col}_sem"] = sem
        finite = values[np.isfinite(values)]
        for q in quantiles:
            key = f"{col}_q{int(round(q * 1000)):03d}"
            out[key] = float(np.quantile(finite, q)) if len(finite) else nan

    if _is_binary(series):
        k = int(np.sum(values.astype(int)))
        lo, hi = wilson_interval(k, n, z=wilson_z)
        out[f"{col}_wilson_lower"] = lo
        out[f"{col}_wilson_upper"] = hi
    return out


def _load_cell_df(cell_dir: Path) -> Optional[pl.DataFrame]:
    """Load samples.parquet + join with every analysis/*.parquet present."""
    samples_path = cell_dir / "samples.parquet"
    if not samples_path.exists():
        return None
    df = pl.read_parquet(samples_path)

    analysis_dir = cell_dir / "analysis"
    if analysis_dir.exists():
        for extra in sorted(analysis_dir.glob("*.parquet")):
            try:
                extra_df = pl.read_parquet(extra)
            except Exception:
                continue
            if "sample_seed" not in extra_df.columns:
                continue
            # Avoid duplicating runtime_s / method columns across analyses
            rename = {c: f"{extra.stem}.{c}" for c in extra_df.columns
                      if c not in ("sample_seed",)}
            extra_df = extra_df.rename(rename)
            df = df.join(extra_df, on="sample_seed", how="left")
    return df


def _conditional_mask(df: pl.DataFrame, cond_col: str, cond_op: str) -> pl.Series:
    """Return a boolean mask corresponding to ``df[cond_col] <op> <value>``."""
    op = cond_op.strip()
    for prefix in (">=", "<=", "==", "!=", ">", "<"):
        if op.startswith(prefix):
            value_str = op[len(prefix):].strip()
            try:
                value = float(value_str)
            except ValueError:
                return pl.Series(values=[True] * len(df))
            if cond_col not in df.columns:
                return pl.Series(values=[True] * len(df))
            series = df[cond_col].cast(pl.Float64)
            if prefix == ">":
                return series > value
            if prefix == ">=":
                return series >= value
            if prefix == "<":
                return series < value
            if prefix == "<=":
                return series <= value
            if prefix == "==":
                return series == value
            if prefix == "!=":
                return series != value
    return pl.Series(values=[True] * len(df))


# ══════════════════════════════════════════════════════════════
# Derived columns
# ══════════════════════════════════════════════════════════════

def _add_derived_columns(df: pl.DataFrame, n: int) -> pl.DataFrame:
    """Add k_over_n, d_cont_over_n, d_min_over_n columns when source cols exist."""
    adds: List[pl.Expr] = []
    if "obs.k" in df.columns:
        adds.append((pl.col("obs.k").cast(pl.Float64) / float(n)).alias("k_over_n"))
    if "distances.d_cont" in df.columns:
        adds.append((pl.col("distances.d_cont").cast(pl.Float64) / float(n))
                    .alias("d_cont_over_n"))
    if "distances.d_min" in df.columns:
        adds.append((pl.col("distances.d_min").cast(pl.Float64) / float(n))
                    .alias("d_min_over_n"))
    if adds:
        df = df.with_columns(adds)
    return df


# ══════════════════════════════════════════════════════════════
# Per-cell stats
# ══════════════════════════════════════════════════════════════

def _cell_row(
    df: pl.DataFrame, n: int, p: float,
    quantiles: List[float], wilson_z: float,
    cond_col: str, cond_op: str,
) -> Dict[str, object]:
    df = _add_derived_columns(df, n)

    row: Dict[str, object] = {"n": int(n), "p": float(p),
                              "n_samples": int(len(df))}
    if cond_col in df.columns:
        mask = _conditional_mask(df, cond_col, cond_op)
        df_cond = df.filter(mask)
    else:
        df_cond = df.clear()                            # no rows
    row[f"n_samples_cond_{cond_col.replace('.', '_')}{_safe_op(cond_op)}"] = int(len(df_cond))

    for col in _numeric_cols(df):
        stats = _cell_stats_for_column(col, df[col], quantiles, wilson_z)
        row.update(stats)
        if len(df_cond) > 0 and col in df_cond.columns:
            cond_stats = _cell_stats_for_column(
                col, df_cond[col], quantiles, wilson_z)
            for k, v in cond_stats.items():
                row[k + "_cond"] = v
    return row


def _safe_op(op: str) -> str:
    return (op.replace(">", "gt").replace("<", "lt")
              .replace("=", "eq").replace("!", "ne").replace(" ", ""))


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

def run_run(run_dir: Path, params: dict, force: bool = False) -> RunRunResult:
    """Always recomputes (MVP policy per spec §9 — caching is a post-MVP concern).

    The ``force`` parameter is accepted for interface parity with cell-scope
    analyses but has no effect: aggregates is cheap to recompute and the inputs
    (per-cell samples + analysis/*.parquet) may have changed between invocations
    without any mtime on ``aggregates.parquet`` to indicate it.
    """
    run_dir = Path(run_dir)
    output = run_dir / OUTPUT_FILENAME
    merged = {**DEFAULT_PARAMS, **params}
    quantiles = list(merged["quantiles"])
    wilson_z = float(merged["wilson_z"])
    cond_col = str(merged["conditional_column"])
    cond_op = str(merged["conditional_predicate"])

    t0 = time.perf_counter()
    cell_dirs = sorted((run_dir / "data").glob("n=*/p=*"))
    if not cell_dirs:
        return RunRunResult(
            name=NAME, status="error", output_path=None,
            runtime_s=time.perf_counter() - t0, params=merged,
            message="no cells found under run_dir/data",
        )

    rows: List[Dict[str, object]] = []
    for cell in cell_dirs:
        try:
            n_str = cell.parent.name.split("=", 1)[1]
            p_str = cell.name.split("=", 1)[1]
            n = int(n_str)
            p = float(p_str)
        except (IndexError, ValueError):
            continue
        df = _load_cell_df(cell)
        if df is None or len(df) == 0:
            continue
        rows.append(_cell_row(df, n, p, quantiles, wilson_z, cond_col, cond_op))

    if not rows:
        return RunRunResult(
            name=NAME, status="error", output_path=None,
            runtime_s=time.perf_counter() - t0, params=merged,
            message="no cells produced any aggregatable rows",
        )

    # Build the parquet with a union of keys across cells (sparse cells get null)
    all_keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    table_dict: Dict[str, list] = {k: [] for k in all_keys}
    for r in rows:
        for k in all_keys:
            table_dict[k].append(r.get(k))
    # Cast dtype per-column
    table = pa.table(table_dict)
    pq.write_table(table, output)
    runtime = time.perf_counter() - t0

    return RunRunResult(
        name=NAME, status="computed", output_path=output,
        runtime_s=runtime, params=merged,
        message=f"wrote {len(rows)} cell-rows to {output.name}",
    )
