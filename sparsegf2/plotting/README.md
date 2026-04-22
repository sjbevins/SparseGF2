# `sparsegf2.plotting` — declarative plots over run directories

Single MVP primitive, `plot_vs_p`, that loads a scalar column from a run's
`samples.parquet` or `analysis/*.parquet`, aggregates across samples with
optional filtering, and draws one curve per size `n`.

## Quickstart

```python
from sparsegf2.plotting import plot_vs_p

# P(k>0) vs p with Wilson-interval bands (auto-detected for binary columns).
fig, ax = plot_vs_p("runs/<run_id>", y="obs.p_k_gt_0")

# Conditional k/n: restrict to samples where k > 0 before aggregating.
fig, ax = plot_vs_p(
    "runs/<run_id>",
    y="k_over_n",
    filter="obs.k > 0",
    errors="bar",
)

# Only curves for specific sizes, SEM error bands, log-x:
fig, ax = plot_vs_p(
    "runs/<run_id>",
    y="distances.d_cont",
    sizes=[32, 64, 128],
    errors="band",
    error_metric="sem",
    xscale="linear",
    yscale="linear",
    title="Contiguous distance vs measurement rate",
    save="fig.png",
)

# Overlay two runs (same schema):
fig, ax = plot_vs_p(["runs/run_a", "runs/run_b"], y="obs.tmi", errors="band")
```

## Signature

```python
plot_vs_p(
    source,                        # Path | str | iterable of paths
    y,                             # column name OR derived-column alias
    *,
    x="p",                         # column on the x-axis
    sizes=None,                    # list[int] | None (= all)
    errors="band",                 # "band" | "bar" | "none"
    error_metric="auto",           # "auto" | "sem" | "std" | "ci95" | "wilson" | "none"
    filter=None,                   # SQL predicate, e.g. "obs.k > 0"
    label_fmt="n={n}",
    title=None, xlabel=None, ylabel=None,
    xscale="linear", yscale="linear",
    save=None,
    ax=None,                       # inject into an existing Axes
    figsize=(6.0, 4.0),
    style="research",              # built-in preset or None
) -> (Figure, Axes)
```

## Column names and derived-column aliases

`y` can be:

- Any column in `samples.parquet` — e.g. `"obs.k"`, `"obs.tmi"`, `"obs.p_k_gt_0"`.
- Any column in an `analysis/*.parquet`, **prefixed with the file stem** —
  e.g. `"distances.d_cont"`, `"logical_weights.d_logical_min"`,
  `"weight_stats.abar"`.
- A derived-column alias recognized by the plotter:
  - `"k_over_n"`
  - `"d_cont_over_n"`
  - `"d_min_over_n"`
  - `"d_logical_min_over_n"`

Add new aliases by editing `aliases.py::DERIVED_ALIASES`.

## Filter strings

`filter` is a polars SQL `WHERE` fragment applied to the sample-level
DataFrame *before* aggregation. The most important use case is conditional
means:

```python
# Plot <k/n | k > 0> — conditional on the code being non-trivial.
plot_vs_p(run, y="k_over_n", filter="obs.k > 0")
```

Column names that contain dots (`obs.k`, `distances.d_cont`) are handled
transparently — you do not need backtick-quoting. Supported operators:
`>`, `>=`, `<`, `<=`, `=`, `!=`, `AND`, `OR`, `NOT`, etc. — anything polars'
SQL engine accepts.

## Error metrics

| `error_metric` | Behaviour                                                                 |
|----------------|---------------------------------------------------------------------------|
| `"auto"`       | Wilson interval when the column is binary (dtype-integer `{0,1}`); SEM otherwise. |
| `"sem"`        | Standard error of the mean = `std(ddof=1) / sqrt(N)`. Symmetric.           |
| `"std"`        | Sample standard deviation. Symmetric (spread, not statistical uncertainty). |
| `"ci95"`       | 95% non-parametric bootstrap confidence interval (seed=42).                |
| `"wilson"`     | Wilson score interval (correct for binomial proportions at small N).       |
| `"none"`       | Zero-width errors regardless of the `errors` argument.                     |

`errors="band"` uses a semi-transparent fill; `errors="bar"` uses matplotlib
errorbars; `errors="none"` draws just the line.

## Data flow

1. **Load.** Scan `source/data/**/samples.parquet` with `hive_partitioning=True`
   so `n` and `p` materialize as columns. Left-join every per-cell
   `analysis/*.parquet` (each file's columns are prefixed with its stem).
2. **Resolve alias.** If `y` is a known derived-column alias, add it as a new
   polars expression.
3. **Filter.** Apply the user's SQL predicate (dot-containing columns are
   handled via temporary renaming).
4. **Aggregate.** Group by `(n, x)`, compute `mean` and asymmetric error
   half-widths `(err_low, err_high)` using the resolved error metric.
5. **Render.** One curve per size, color from `viridis`, legend as
   `label_fmt`, save to disk if `save=...`.

## Future primitives

The plan is to add sibling primitives in `sparsegf2.plotting.primitives/`:

- `plot_heatmap` — `z(n, p)` or `z(x, y)` with colorbar.
- `plot_histogram` — distribution of a scalar across samples for a cell / group.
- `plot_rate_distance` — Pareto scatter of rate vs distance.
- `plot_collapse` — FSS data collapse `y((p - p_c) * n^(1/nu))`.

Each new primitive should reuse `sparsegf2.plotting.data._load`,
`_apply_filter`, and (where applicable) `_aggregate`.
