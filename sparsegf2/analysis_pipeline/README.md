# `sparsegf2.analysis_pipeline` — post-processing over a run directory

This subpackage walks a `runs/<run_id>/` tree produced by
[`sparsegf2.circuits`](../circuits/README.md), rehydrates each cell's
`tableaus.h5`, runs a configurable suite of analyses, and writes the results
into `data/n=XXXX/p=Y.YYYY/analysis/<name>.{parquet,h5}` plus a run-level
`aggregates.parquet`.

## Quickstart

```bash
# Default: run every analysis (cheap ones + the opt-out expensive ones).
py -3.13 -m sparsegf2.analysis_pipeline runs/<run_id>

# Skip the expensive analyses for a first pass:
py -3.13 -m sparsegf2.analysis_pipeline runs/<run_id> \
    --skip weight_spectrum logical_weights

# Fill in the ones you skipped, later:
py -3.13 -m sparsegf2.analysis_pipeline runs/<run_id>

# Recompute one analysis:
py -3.13 -m sparsegf2.analysis_pipeline runs/<run_id> \
    --only distances --force distances

# Restrict to a subset of cells (useful for iterative development):
py -3.13 -m sparsegf2.analysis_pipeline runs/<run_id> \
    --sizes 32 64 --p-values 0.1 0.15 0.2

# Parallel across cells:
py -3.13 -m sparsegf2.analysis_pipeline runs/<run_id> --workers 8
```

From Python:

```python
from sparsegf2.analysis_pipeline import AnalysisConfig, run_pipeline

cfg = AnalysisConfig(
    run_dir="runs/<run_id>",
    skip=["weight_spectrum"],    # mutually exclusive with `only`
    force=["distances"],
    n_workers=8,
)
report = run_pipeline(cfg)
print(report)
```

## What runs by default

| Analysis key       | Output                                          | Default | Cost      |
|--------------------|-------------------------------------------------|---------|-----------|
| `distances`        | `analysis/distances.parquet`                    | on      | cheap     |
| `weight_stats`     | `analysis/weight_stats.parquet`                 | on      | cheap     |
| `entropy_profile`  | `analysis/entropy_profile.h5`                   | on      | moderate  |
| `weight_spectrum`  | `analysis/weight_spectrum.h5`                   | on (`--skip` to drop) | storage-heavy at n >= 128 |
| `logical_weights`  | `analysis/logical_weights.parquet`              | on (`--skip` to drop) | compute-heavy (combinatorial search) |
| `aggregates`       | `aggregates.parquet` at the run root            | on      | cheap (runs once after cells) |

- Existing output files are **never overwritten** unless `--force <name>` is passed.
- Rerunning the pipeline after adding new tableaus to a cell picks up the gaps automatically.
- Every analysis's metadata (parameters, git hash, runtime, sample count) is recorded in the cell's `analysis/_registry.json`.

## What each analysis contains

### `distances.parquet` (per cell)

One row per sample:

| Column         | Meaning                                                          |
|----------------|------------------------------------------------------------------|
| `sample_seed`  | join key with `samples.parquet`                                  |
| `d_cont`       | smallest contiguous arc A with I(A; R) > 0                       |
| `d_min`        | upper bound on true code distance (exhaustive ≤ 3 + greedy shrink) |
| `d_cont_method`, `d_min_method` | method tag strings for provenance             |
| `runtime_s`    | per-sample compute time                                          |

### `weight_stats.parquet` (per cell)

One row per sample, mirroring the scalars returned by
`sparsegf2.analysis.observables.observe(sim)`:

- Generator-weight moments: `abar, var_a, std_a, cv_a, a_max, a_min, skew_a`
- Pauli-weight moments: `wbar, var_w, std_w, cv_w, w_max, w_min, skew_w`
- Stabilizer/destabilizer split: `wbar_stab, wbar_destab`
- X-only mean weight: `mean_wt_x`
- Consistency check: `weight_mass`, `identity_holds` (sum rule invariant)
- Pivot diagnostics: `pei_proxy`, `wcp_proxy`, `pivot_speedup`, `n_anticommuting_qubits`

### `entropy_profile.h5` (per cell)

Fixed-length `uint16` arrays: for each sample, `S(A_ell)` for ell in 1..n/2 at
a canonical starting position (default 0). Useful for plotting entanglement
profiles and diagnosing phase boundaries.

### `weight_spectrum.h5` (per cell, optional)

Per-generator weights as a `uint16[S, 2n]` matrix. Rows 0..n-1 are
destabilizers, rows n..2n-1 are stabilizers. Storage scales as 2nS uint16s
per cell — flagged expensive on that basis, not CPU.

### `logical_weights.parquet` (per cell, optional)

For each sample, minimum-weight **pure-type** logical operator:

| Column          | Meaning                                                   |
|-----------------|-----------------------------------------------------------|
| `d_logical_x`   | min |A| such that a pure-X operator on A is a logical     |
| `d_logical_y`   | same for pure-Y                                           |
| `d_logical_z`   | same for pure-Z                                           |
| `d_logical_min` | `min(d_logical_x, d_logical_y, d_logical_z)`              |
| `search_depth`  | maximum |A| enumerated (cap = min(max_exhaustive, d_cont))|
| `method`        | method tag; `n+1` in a column means "no pure-T logical of weight <= search_depth was found" |

Columns that exhaust the search without finding a logical are filled with
the sentinel value `n + 1`. For CSS-like emergent codes you'll see tight
`d_logical_x` and `d_logical_z`; strongly non-CSS codes push both towards
the sentinel while `d_min` from `distances.parquet` remains small.

### `aggregates.parquet` (one per run)

One row per `(n, p)` cell, with:

- Partition keys: `n`, `p`, `n_samples`, and `n_samples_cond_<cond>`.
- For every numeric column across samples + analyses:
  - `<col>_mean`, `<col>_std`, `<col>_sem`
  - `<col>_q025`, `<col>_q500`, `<col>_q975` quantiles
  - For binary columns (dtype-integer in `{0, 1}`): `<col>_wilson_lower/upper`
- The same set suffixed `_cond` for rows restricted to `obs.k > 0`.
- Derived columns: `k_over_n`, `d_cont_over_n`, `d_min_over_n` when their
  source columns are present.

The plotting primitive reads this file first when possible. It is always
recomputed on every pipeline invocation (caching
`aggregates.parquet` is deliberately post-MVP).

## Adding a new analysis

1. Create `analyses/<name>.py` that exposes:

   ```python
   NAME = "<name>"
   OUTPUT_FILENAME = "<name>.parquet"   # or ".h5"
   OUTPUT_KIND = "parquet"              # or "h5"
   CELL_SCOPE = True                    # False for run-scope
   EXPENSIVE = False                    # True => opt-outable in CLI default
   DEFAULT_PARAMS = {}

   def run_cell(ctx, params, force=False) -> CellRunResult:
       ...
   ```

2. Register it in `analyses/__init__.py::ANALYSIS_REGISTRY`.

No other changes needed — the orchestrator, CLI, and registry updates
pick up the new analysis automatically.

See `analyses/_common.py` for `CellContext` and `CellRunResult`. See
`analyses/distances.py` for a complete, minimal cell-scope example. See
`analyses/aggregates.py` for a run-scope example (uses `run_run(run_dir, ...)`
and `CELL_SCOPE = False`).

## Rehydration

Persisted tableau bits (`x_packed`, `z_packed`) are loaded back into a live
`SparseGF2` via `rehydrate_sim(n, x_packed, z_packed)`. This means every
analysis can call the existing `sim.compute_subsystem_entropy(...)`,
`observe(sim)`, etc. without reimplementing logic on raw arrays. See
`rehydrate.py`.

## Forward compatibility

- **New columns** can be added to any existing `.parquet` without breaking readers.
- **New datasets** can be added to any existing `.h5` without breaking readers.
- **New analyses** appear as new files in `analysis/`; existing files are
  untouched.
- The per-cell `_registry.json` records exactly what was computed, with
  what parameters, at what code version, so downstream consumers can filter
  by algorithm version if/when the algorithm evolves.

## Version

Registry / manifest use `schema_version = "1.0.0"`.
