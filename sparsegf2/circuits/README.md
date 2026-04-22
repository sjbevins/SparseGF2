# `sparsegf2.circuits` — Graph-defined random-Clifford + measurement circuits

This subpackage is the canonical way to configure, run, and serialize
graph-defined Clifford circuits on top of the SparseGF2 simulator for the
code-discovery platform. This README is the user-facing reference for the
per-layer *matching modes*, which are the most subtle knob in the package.

## What this package is for

The goal of the SparseGF2 code-discovery platform is to find good quantum
error-correcting codes that emerge from measurement-induced phase
transitions in families of graph-defined random Clifford circuits. This
subpackage provides:

1. A uniform configuration object (`CircuitConfig` + `RunConfig`) for
   every knob that defines a sweep.
2. A pre-flight **validator** that refuses to run a sweep when the
   requested matching mode is incompatible with the graph at a given
   size — and tells the user which modes *would* work.
3. A standardized **output directory** (see "The output tree" below) with
   independent slots for metadata, per-sample diagnostics + observables,
   optional tableau snapshots, optional circuit realizations, and
   post-hoc analyses. The slots are Hive-partitioned by `(n, p)` and use
   Parquet for tabular data and HDF5 for arrays.

## Quickstart

```bash
py -3.13 -m sparsegf2.circuits \
    --graph cycle \
    --matching-mode round_robin \
    --sizes 32 64 128 \
    --p-min 0.0 --p-max 0.6 --n-p 50 \
    --samples 500 \
    --depth-mode O_n \
    --depth-factor 8 \
    --save-tableaus \
    --workers 14 \
    --output runs/
```

Or from Python:

```python
from sparsegf2.circuits import CircuitConfig, RunConfig, SweepDriver

circuit = CircuitConfig(
    graph_spec="cycle",
    n=32,                           # overridden per cell by the driver
    matching_mode="round_robin",
    measurement_mode="uniform",
    depth_mode="O(n)",
    depth_factor=8,
    p=0.15,                         # overridden per cell by the driver
    base_seed=42,
)
run_cfg = RunConfig(
    circuit=circuit,
    sizes=[32, 64, 128],
    p_min=0.0, p_max=0.6, n_p=50,
    n_samples_per_cell=500,
    save_tableaus=True,
    n_workers=14,
)
run_dir = SweepDriver(run_cfg).run()
```

## The output tree

Every run produces:

```
runs/<run_id>/
    manifest.json                     run metadata (always)
    graph.g6                          deterministic graph6 (if applicable)
    data/n=<N>/p=<P>/
        samples.parquet               always: identity + diag.* + obs.* columns
        tableaus.h5                   optional: end-of-circuit stabilizer tableaus
        realizations.h5               optional: layer-by-layer circuit trace
        analysis/                     optional, grows post-hoc per analysis
            <name>.parquet | <name>.h5
    index.parquet                     redundant: cell-level summary
```

Load the scalar data with `polars` or `duckdb` and let hive partitioning
push `n` and `p` into the DataFrame for free:

```python
import polars as pl
df = pl.scan_parquet("runs/<id>/data/**/samples.parquet", hive_partitioning=True)
df.filter(pl.col("n") == 128).group_by("p").agg(pl.col("obs.k").mean()).collect()
```

## Matching modes — authoritative reference

This section is the canonical definition of each matching mode, referenced
the authoritative reference for each mode.

In the `matching` gating mode, gates are applied in **matchings** — sets
of edges sharing no vertex — so that every gate in a layer acts on a
disjoint pair of qubits. A **perfect matching** is a matching in which
every vertex participates (so it has exactly `n/2` edges).

For a graph `G` with chromatic index `χ'(G)`, Vizing's theorem gives
`χ'(G) ∈ {Δ(G), Δ(G)+1}`. When `G` is regular and *class 1* (i.e.
`χ'(G) = Δ(G)`), the edges decompose into `Δ(G)` perfect matchings; this
decomposition is called a **1-factorization**. Both the cycle (even `n`)
and the complete graph `K_n` (even `n`) are class 1 and admit
1-factorizations. Odd `n` has no perfect matching for either family.

Each circuit period is `χ'` gate layers long. Within a period, the three
matching modes differ only in *how* the layer-to-matching assignment is
made.

### `matching_mode = "round_robin"`

Compute a canonical 1-factorization `M_0, M_1, …, M_{χ'-1}` once. At
layer `t`, apply matching `M_{t mod χ'}`.

- **Deterministic given `(graph, base_seed)`.**
- **Reproducible**: no per-layer RNG draws affect gate placement.
- **Use when** you want identical gate placements across samples and are
  only varying measurement randomness + Clifford choice.
- **Compatibility**: graph must admit a 1-factorization (regular,
  class 1). For MVP: `cycle` with even `n`, `complete` with even `n`.

### `matching_mode = "palette"`

Compute a canonical 1-factorization `M_0, M_1, …, M_{χ'-1}` once (the
"palette"). At each layer, sample `M_i` uniformly at random from the
palette.

- **Stochastic** but drawn from a finite pre-fixed palette.
- **Reproducible** given `(graph, base_seed, sample_seed)`.
- **Use when** you want stochastic layer order while keeping the *set*
  of realized matchings small and identical across runs.
- **Compatibility**: same as `round_robin` — requires 1-factorization.

### `matching_mode = "fresh"`

At each layer, sample a uniformly random perfect matching of `G` from
*all* perfect matchings of `G`, not from a fixed palette.

- **Maximally stochastic.** No fixed 1-factorization is precomputed;
  different samples see entirely different matchings.
- **Use when** you want the layer-by-layer gate placements to fully
  average over all perfect matchings of the graph (Bentsen-style ATA,
  where this is the literature standard).
- **Implementation for `complete` graph**: sample a uniformly random
  permutation `σ` of `[0, n)`, then pair
  `(σ[0], σ[1]), (σ[2], σ[3]), …`. This is uniform over perfect
  matchings of `K_n` by a standard counting argument.
- **Implementation for `cycle` graph**: the cycle has only *two*
  perfect matchings (even / odd edges), so at each layer flip a coin
  between them. Degenerate with `palette` for cycle.
- **Compatibility**: graph must have at least one perfect matching.

### Compatibility matrix (MVP graphs)

| Graph      | Even `n`                                                                                           | Odd `n`                         |
|------------|----------------------------------------------------------------------------------------------------|---------------------------------|
| `cycle`    | all three modes work; `fresh` and `palette` are degenerate (only 2 matchings exist)                | ✗ no perfect matching           |
| `complete` | all three modes work; all three are genuinely distinct                                             | ✗ no perfect matching           |

If you request a mode that is incompatible with the graph at some size,
the pre-flight validator aborts the run with exit code 2 before any
sample is written, and the error message lists the modes that *would*
work.

## Measurement mode

MVP supports one measurement mode:

- `uniform` — after each gate layer, *every* qubit in `[0, n)` is
  independently measured in the Z-basis with probability `p`. The gate-
  to-measurement ratio is `1 : 2p` (one gate touches two qubits, each
  measured with prob `p`), recorded per-sample in
  `samples.parquet` as `diag.gate_to_meas_ratio_{expected,actual}` for
  cross-checking.

The `gated` measurement mode (candidates restricted to the qubits gated
that layer) is deferred post-MVP.

## Physics picture

MVP supports one picture:

- `purification` — `2n` qubits (`n` system + `n` reference), initialized
  as `n` Bell pairs; the circuit acts only on the system qubits. After
  the circuit, the reference subsystem encodes a QEC code.

The `single_ref` and `pure_state` pictures are deferred post-MVP.

## What's in a `samples.parquet`

Every file has the same columns:

| prefix   | columns                                                                                                                        |
|----------|--------------------------------------------------------------------------------------------------------------------------------|
| identity | `sample_seed`                                                                                                                  |
| `diag.*` | `total_layers`, `total_gates`, `avg_gates_per_layer`, `total_measurements`, `gate_to_meas_ratio_{expected,actual}`, `final_abar`, `runtime_{total,gate_phase,meas_phase}_s` |
| `obs.*`  | `k`, `bandwidth`, `tmi`, `entropy_half_cut`, `p_k_gt_0`                                                                         |

`n` and `p` are not columns — they are encoded in the path
(`data/n=XXXX/p=Y.YYYY/`) and materialize automatically as DataFrame
columns when you read with `hive_partitioning=True`.

## Adding a new analysis later

Post-hoc analyses live in `data/<cell>/analysis/<analysis_name>.parquet`
(scalars per sample) or `.h5` (arrays per sample). Adding a new one
requires no changes anywhere else in the tree: your analysis script
reads `tableaus.h5`, computes its quantity, and writes to
`analysis/<name>.parquet`. Update `analysis/_registry.json` with the
algorithm, parameters, and code hash. Old readers are unaffected.

## Version

The output schema is versioned via `manifest.json:schema_version` (semver).
Current: **1.0.0**.
