"""
Analysis utilities for the ``single_ref`` MIPT-probe picture.

Two flavours of analysis:

1. **Crossing analysis** (endpoint). Reads ``samples.parquet`` across the
   run, aggregates ``obs.k = S(qubit n)`` by ``(n, p)``, and produces a
   crossing plot of ``<S>`` vs ``p`` for each ``n``. The curves cross at
   the MIPT critical point ``p_c``.

2. **Purification-time analysis** (time-series). When the run was
   executed with ``CircuitConfig.record_time_series=True``, each cell
   carries a ``timeseries.h5`` file holding the per-layer reference-
   qubit entropy. The survival probability ``P(t) = <S(t)>`` (ensemble-
   averaged over samples) decays over time; the characteristic
   purification time ``tau`` is the smallest ``t`` at which ``P(t)``
   drops to ``0.5``. ``tau`` typically scales with system size at and
   near the critical rate (``tau ~ n`` at ``p_c`` in 1+1D MIPT).

The top-level entry point :func:`analyze_single_ref` auto-detects from
``manifest.json`` which analyses are applicable, runs them, and writes
figures to the run directory.

References
----------
Gullans & Huse 2020, Phys. Rev. X 10, 041020 (arXiv:1905.05195).
Skinner, Ruhman, Nahum 2019, Phys. Rev. X 9, 031009 (arXiv:1808.05953).
Fattal, Cubitt, Yamamoto, Bravyi, Chuang 2004, arXiv:quant-ph/0406168.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl


# Manifest / metadata detection

def detect_picture(run_dir: Union[str, Path]) -> str:
    """Return the ``picture`` field recorded in ``manifest.json``.

    Raises FileNotFoundError if no manifest is present, ValueError if it
    cannot be parsed or does not name a picture.
    """
    run_dir = Path(run_dir)
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(f"no manifest.json under {run_dir}")
    try:
        data = json.loads(mpath.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse {mpath}: {exc}") from exc
    # SparseGF2 writes the full config under the top-level ``config`` key;
    # fall back to other plausible shapes for forward-compat.
    for container in (
        data.get("config"),
        data.get("circuit_config"),
        data.get("circuit"),
        data,
    ):
        if isinstance(container, dict):
            picture = container.get("picture")
            if isinstance(picture, str):
                return picture
    raise ValueError(
        f"manifest.json at {mpath} does not record a picture field"
    )


def has_timeseries(run_dir: Union[str, Path]) -> bool:
    """True iff at least one cell has a ``timeseries.h5`` file."""
    run_dir = Path(run_dir)
    return any((run_dir / "data").glob("n=*/p=*/timeseries.h5"))


# Endpoint (crossing) analysis

def load_samples(run_dir: Union[str, Path]) -> pl.DataFrame:
    """Load every ``samples.parquet`` under a run directory (eager, hive)."""
    run_dir = Path(run_dir)
    pattern = str(run_dir / "data" / "**" / "samples.parquet")
    return pl.scan_parquet(pattern, hive_partitioning=True).collect()


def aggregate_entropy(
    df: pl.DataFrame, entropy_col: str = "obs.k"
) -> pl.DataFrame:
    """Aggregate ``obs.k`` by ``(n, p)``: mean, std (ddof=1), SEM, count."""
    if entropy_col not in df.columns:
        raise KeyError(
            f"column {entropy_col!r} not in dataframe; available: "
            f"{sorted(df.columns)}"
        )
    agg = (
        df.group_by(["n", "p"])
        .agg(
            pl.col(entropy_col).mean().alias("S_mean"),
            pl.col(entropy_col).std(ddof=1).alias("S_std"),
            pl.col(entropy_col).count().alias("count"),
        )
        .with_columns(
            (pl.col("S_std") / pl.col("count").cast(pl.Float64).sqrt())
            .alias("S_sem")
        )
        .sort(["n", "p"])
    )
    return agg


def plot_crossing(
    agg: pl.DataFrame,
    *,
    out_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    title: Optional[str] = None,
    ax=None,
):
    """``<S>`` vs ``p`` curves for each system size. Curves intersect near p_c."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
    else:
        fig = ax.figure

    sizes = sorted(int(v) for v in agg["n"].unique().to_list())
    for n_val in sizes:
        sub = agg.filter(pl.col("n") == n_val).sort("p")
        ax.errorbar(
            sub["p"].to_numpy(),
            sub["S_mean"].to_numpy(),
            yerr=sub["S_sem"].to_numpy(),
            marker="o",
            linewidth=1.6,
            capsize=3,
            label=f"n={n_val}",
        )

    ax.set_xlabel("p  (measurement rate)")
    ax.set_ylabel(r"$\langle S \rangle$  (reference-qubit entropy)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.0, color="0.6", linestyle=":", linewidth=0.8)
    ax.axhline(1.0, color="0.6", linestyle=":", linewidth=0.8)
    ax.set_title(title if title is not None else "single_ref MIPT probe")
    ax.legend(title="system size", loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


# Time-series (purification-time) analysis

@dataclass
class CellTimeseries:
    """Per-cell time-series payload loaded from ``timeseries.h5``."""

    n: int
    p: float
    S_of_t: np.ndarray     # uint8 [n_samples, total_layers+1]
    t_axis: np.ndarray     # int32 [total_layers+1]
    sample_seed: np.ndarray  # int64 [n_samples]

    @property
    def P_of_t(self) -> np.ndarray:
        """Ensemble-averaged survival probability P(t) = <S(t)>."""
        return self.S_of_t.mean(axis=0).astype(np.float64)


def load_timeseries(run_dir: Union[str, Path]) -> Dict[Tuple[int, float], CellTimeseries]:
    """Load every ``timeseries.h5`` under the run, keyed by ``(n, p)``."""
    import h5py

    run_dir = Path(run_dir)
    out: Dict[Tuple[int, float], CellTimeseries] = {}
    for cell in sorted((run_dir / "data").glob("n=*/p=*")):
        ts_path = cell / "timeseries.h5"
        if not ts_path.exists():
            continue
        try:
            n = int(cell.parent.name.split("=", 1)[1])
            p = float(cell.name.split("=", 1)[1])
        except (IndexError, ValueError):
            continue
        with h5py.File(ts_path, "r") as f:
            S_of_t = np.asarray(f["S_of_t"][:], dtype=np.uint8)
            t_axis = np.asarray(f["t_axis"][:], dtype=np.int32)
            seeds = np.asarray(f["sample_seed"][:], dtype=np.int64)
        out[(n, p)] = CellTimeseries(
            n=n, p=p, S_of_t=S_of_t, t_axis=t_axis, sample_seed=seeds
        )
    return out


def compute_tau(
    cell: CellTimeseries, *, threshold: float = 0.5
) -> Optional[float]:
    """Characteristic purification time: smallest t with P(t) <= threshold.

    Uses linear interpolation between ``t`` and ``t - 1`` when P crosses
    the threshold continuously. Returns ``None`` if P(t) never falls
    below the threshold within the recorded window.
    """
    P = cell.P_of_t
    if P.size == 0:
        return None
    # Bell-pair initial state: P(0) should be ~1. If the initial value is
    # already below threshold something is off with the data; treat as tau=0.
    if P[0] <= threshold:
        return 0.0
    below = np.where(P <= threshold)[0]
    if below.size == 0:
        return None
    j = int(below[0])
    # Linear-interpolate between (j-1, P_{j-1}) and (j, P_j)
    P_prev, P_cur = P[j - 1], P[j]
    if P_prev == P_cur:
        return float(cell.t_axis[j])
    frac = (P_prev - threshold) / (P_prev - P_cur)
    return float(cell.t_axis[j - 1] + frac * (cell.t_axis[j] - cell.t_axis[j - 1]))


def plot_purification_decay(
    ts: Dict[Tuple[int, float], CellTimeseries],
    *,
    p_values: Optional[List[float]] = None,
    n_values: Optional[List[int]] = None,
    out_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    title: Optional[str] = None,
    ax=None,
):
    """``P(t)`` vs ``t`` for selected ``(n, p)`` cells.

    By default plots every cell; pass ``p_values`` and/or ``n_values`` to
    subset.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
    else:
        fig = ax.figure

    keys = sorted(ts.keys())
    if p_values is not None:
        keys = [k for k in keys if any(abs(k[1] - pv) < 1e-9 for pv in p_values)]
    if n_values is not None:
        keys = [k for k in keys if k[0] in n_values]

    cmap = plt.get_cmap("viridis")
    # color by p within plot (common convention)
    all_p = sorted({p for _, p in keys})
    p_to_color = {p: cmap(i / max(1, len(all_p) - 1)) for i, p in enumerate(all_p)}

    for (n_val, p_val) in keys:
        cell = ts[(n_val, p_val)]
        ax.plot(
            cell.t_axis,
            cell.P_of_t,
            linewidth=1.4,
            color=p_to_color[p_val],
            label=f"n={n_val}, p={p_val:.3f}",
        )

    ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1.0, label="P = 0.5")
    ax.set_xlabel("t  (circuit layer)")
    ax.set_ylabel(r"$P(t) = \langle S(t) \rangle$")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title if title is not None else "single_ref purification decay")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_tau_scaling(
    ts: Dict[Tuple[int, float], CellTimeseries],
    *,
    p: Optional[float] = None,
    out_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    title: Optional[str] = None,
    ax=None,
):
    """``tau`` vs ``n``. If ``p`` is given, restrict to that rate; otherwise
    plot one curve per p value present.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
    else:
        fig = ax.figure

    # Group by p-value.
    by_p: Dict[float, List[Tuple[int, Optional[float]]]] = {}
    for (n_val, p_val), cell in ts.items():
        if p is not None and abs(p_val - p) > 1e-9:
            continue
        tau = compute_tau(cell)
        by_p.setdefault(p_val, []).append((int(n_val), tau))

    for p_val in sorted(by_p):
        rows = sorted(by_p[p_val])
        ns = [n for n, _ in rows]
        taus = [t if t is not None else np.nan for _, t in rows]
        ax.plot(ns, taus, marker="o", linewidth=1.5, label=f"p={p_val:.3f}")

    ax.set_xlabel("n  (system size)")
    ax.set_ylabel(r"$\tau$  (purification time, layers to P=0.5)")
    ax.set_title(title if title is not None else r"single_ref $\tau$ scaling")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


# Auto-detecting top-level entry point

def analyze_single_ref(
    run_dir: Union[str, Path], *, tau_threshold: float = 0.5
) -> Dict[str, Path]:
    """Run the full single_ref analysis on a run directory.

    Auto-detects ``picture`` from ``manifest.json``; raises if the run is
    not ``single_ref``. Always produces ``crossing_plot.png`` from
    ``samples.parquet``. If ``timeseries.h5`` files exist, also produces
    ``purification_decay.png`` and ``tau_scaling.png``.

    Returns a mapping from artefact name to its written path.
    """
    run_dir = Path(run_dir)
    picture = detect_picture(run_dir)
    if picture != "single_ref":
        raise ValueError(
            f"analyze_single_ref: picture in manifest is {picture!r}; "
            "pipeline is only defined for picture='single_ref'"
        )

    out: Dict[str, Path] = {}

    # 1. Crossing plot (endpoint).
    df = load_samples(run_dir)
    agg = aggregate_entropy(df)
    crossing_path = run_dir / "crossing_plot.png"
    plot_crossing(agg, out_path=crossing_path, title=f"{run_dir.name}: crossing")
    out["crossing_plot"] = crossing_path

    # 2. Purification-time plots, if the data is available.
    if has_timeseries(run_dir):
        ts = load_timeseries(run_dir)
        if ts:
            decay_path = run_dir / "purification_decay.png"
            plot_purification_decay(
                ts, out_path=decay_path,
                title=f"{run_dir.name}: P(t) decay",
            )
            out["purification_decay"] = decay_path

            tau_path = run_dir / "tau_scaling.png"
            plot_tau_scaling(
                ts, out_path=tau_path,
                title=rf"{run_dir.name}: $\tau$(n)",
            )
            out["tau_scaling"] = tau_path
    return out


__all__ = [
    "detect_picture",
    "has_timeseries",
    "load_samples",
    "aggregate_entropy",
    "plot_crossing",
    "CellTimeseries",
    "load_timeseries",
    "compute_tau",
    "plot_purification_decay",
    "plot_tau_scaling",
    "analyze_single_ref",
]
