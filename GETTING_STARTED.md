# Getting Started with SparseGF2

A walkthrough from a fresh clone to a first MIPT sweep. Covers the core
simulator, the `sparsegf2.circuits` subpackage, the analysis pipeline,
the plotting primitive, and reproduction of the benchmark figures.

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Verify the install](#3-verify-the-install)
4. [Your first simulation](#4-your-first-simulation)
5. [Applying gates](#5-applying-gates)
6. [Measurements](#6-measurements)
7. [Observables](#7-observables)
8. [Hybrid sparse/dense mode](#8-hybrid-sparsedense-mode)
9. [Verifying against Stim](#9-verifying-against-stim)
10. [A random-Clifford circuit using the full 11,520-element group](#10-a-random-clifford-circuit-using-the-full-11520-element-group)
11. [Your first MIPT sweep](#11-your-first-mipt-sweep)
12. [Using the `sparsegf2.circuits` subpackage](#12-using-the-sparsegf2circuits-subpackage)
13. [Reading a run directory](#13-reading-a-run-directory)
14. [The analysis pipeline](#14-the-analysis-pipeline)
15. [The plotting primitive](#15-the-plotting-primitive)
16. [Reproducing the benchmarks](#16-reproducing-the-benchmarks)
17. [Troubleshooting](#17-troubleshooting)
18. [Where to go from here](#18-where-to-go-from-here)

---

## 1. Prerequisites

- **Python 3.11, 3.12, or 3.13.** The CI matrix exercises all three. Python
  3.13 is the primary development target on Windows; 3.11 is the oldest
  supported.
- **Operating system.** Linux, macOS, and Windows are all supported. The
  repository is developed on Windows 11 and regularly exercised on
  Ubuntu-latest via GitHub Actions.
- **Compiler.** Not required. The package is pure Python + Numba JIT; no
  build step is needed beyond `pip install`.

### Required dependencies (automatically installed)

| Package | Why |
|---|---|
| `numpy >= 1.24` | Array backbone |
| `numba >= 0.57` | JIT-compiled hotpaths (gate/measurement kernels, GF(2) rank) |
| `stim >= 1.13` | Ground-truth simulator for RREF parity checks |

### Optional extras

| Extra | Packages | What it unlocks |
|---|---|---|
| `[plotting]` | `matplotlib` | `sparsegf2.plotting.plot_vs_p` and the benchmark plot scripts |
| `[analysis]` | `networkx`, `scipy` | Tanner-graph utilities under `sparsegf2.analysis.tanner_graph` |
| `[circuits]` | `pyarrow`, `h5py`, `polars` | Parquet / HDF5 I/O for `sparsegf2.circuits` and `sparsegf2.analysis_pipeline` |
| `[dev]` | `pytest`, `pytest-cov` | Running the test suite |
| `[all]` | everything above | Recommended for local development |

## 2. Installation

From a fresh clone:

```bash
git clone https://github.com/sjbevins/SparseGF2-v1.git
cd SparseGF2-v1
pip install -e ".[all]"
```

Editable install (`-e`) is recommended while you're exploring — it lets
you hack on the source without reinstalling.

## 3. Verify the install

Import the package and run a few basic tests:

```bash
py -3.13 -c "import sparsegf2; print(sparsegf2.__version__)"
# Expected output: 0.2.0

py -3.13 -m pytest tests/test_stim_rref_verification.py -q
# Expected: several hundred tests pass in ~10s
```

Optionally, run the standalone Stim equivalence suite:

```bash
py -3.13 sanity_checks/verify_vs_stim.py
```

## 4. Your first simulation

```python
from sparsegf2 import SparseGF2, warmup

# One-time JIT compile of every Numba kernel (~3 seconds).
# Without this, the first gate/measurement call will be slow.
warmup()

# 16-qubit system initialized in the purification picture:
# 16 system qubits + 16 reference qubits as 16 Bell pairs.
sim = SparseGF2(n=16)

print("k =", sim.compute_k())            # code dimension (k = n for Bell init)
print("abar =", sim.get_active_count())  # mean generator support
```

Expected output:

```
k = 16
abar = 2.0
```

The initial `abar = 2.0` reflects each generator having support on exactly
one system qubit and one reference qubit (the Bell-pair X_i X_{n+i},
Z_i Z_{n+i}).

## 5. Applying gates

SparseGF2 exposes three gate paths, in decreasing specialization:

### 5.1. Specialized fast kernels (H, S, CX)

```python
sim.apply_h(0)               # Hadamard on qubit 0
sim.apply_s(3)               # S gate on qubit 3
sim.apply_cx_fast(0, 1)      # CNOT with control=0, target=1
```

These bypass the generic LUT-building path and are the fastest option at
low average generator support. Use them whenever the three elementary
Cliffords are enough.

### 5.2. Generic 1-qubit Clifford

A 1-qubit Clifford is a `2 × 2` GF(2) symplectic matrix acting on `(X, Z)`:

```python
import numpy as np
H = np.array([[0, 1], [1, 0]], dtype=np.uint8)   # X ↔ Z
sim.apply_gate_1q(5, H)
```

### 5.3. Generic 2-qubit Clifford

Any 2-qubit Clifford can be applied by passing its `4 × 4` GF(2) symplectic
matrix acting on `(X_qi, X_qj, Z_qi, Z_qj)`:

```python
import numpy as np
CNOT = np.array([
    [1, 1, 0, 0],   # X_qi → X_qi X_qj
    [0, 1, 0, 0],   # X_qj → X_qj
    [0, 0, 1, 0],   # Z_qi → Z_qi
    [0, 0, 1, 1],   # Z_qj → Z_qi Z_qj
], dtype=np.uint8)
sim.apply_gate(0, 1, CNOT)
```

To use Stim's full 2-qubit Clifford group:

```python
import stim
from sparsegf2 import symplectic_from_stim_tableau

for tab in stim.Tableau.iter_all(2):      # all 11,520 elements
    S = symplectic_from_stim_tableau(tab) # 4×4 GF(2) symplectic matrix
    sim.apply_gate(0, 1, S)
```

## 6. Measurements

```python
sim.apply_measurement_z(0)   # Z-basis projective measurement, reset to |0⟩
```

On a generator that anticommutes with `Z_0`, the measurement collapses it;
otherwise the state is unchanged. The `apply_measurement_z` method is the
workhorse for MIPT simulations.

X-basis measurement is available via `apply_measurement_x`. Y-basis
measurement has a known-issue divergence from Stim in some configurations
(see [CHANGELOG.md](CHANGELOG.md)); avoid in the interim.

## 7. Observables

SparseGF2 computes all major scalar observables directly from the
stabilizer tableau without materializing the state vector:

```python
sim.compute_k()                          # code dimension (GF(2) rank)
sim.get_active_count()                   # avg generator support (abar)
sim.compute_bandwidth()                  # max generator bandwidth on C_n
sim.compute_tmi()                        # tripartite mutual information
sim.compute_subsystem_entropy([0, 1, 2]) # S(A) for A = {0, 1, 2}
```

Every one of these is a linear-algebra primitive over GF(2); none of them
traces out a density matrix.

## 8. Hybrid sparse/dense mode

By default, the simulator stays in pure-sparse mode for the whole circuit
— this is fastest when the circuit remains in the area-law phase of the
MIPT. For circuits that pass through the volume-law phase (e.g., a low-*p*
warmup followed by a high-*p* steady state), opt into hybrid mode:

```python
sim = SparseGF2(n=256, hybrid_mode=True)
# The simulator auto-switches to bit-packed dense mode when the average
# generator support crosses n/4, and back to sparse when it drops below n/8.
```

Hybrid mode adds a small per-op check; leave it off for strictly area-law
workloads.

## 9. Verifying against Stim

Correctness is our bar: SparseGF2 and Stim must produce the *identical
stabilizer group* for any circuit. Here's the minimum pattern for
verifying this yourself:

```python
import numpy as np
import stim
from sparsegf2 import SparseGF2, warmup, symplectic_from_stim_tableau

warmup()

def gf2_rref(M):
    """Reduce an integer {0,1} matrix to GF(2) RREF."""
    M = M.copy().astype(np.uint8)
    rank = 0
    for col in range(M.shape[1]):
        for row in range(rank, M.shape[0]):
            if M[row, col]:
                M[[rank, row]] = M[[row, rank]]
                break
        else:
            continue
        for row in range(M.shape[0]):
            if row != rank and M[row, col]:
                M[row] ^= M[rank]
        rank += 1
    return M[:rank]

# Apply the same gate to both simulators.
n = 8
sim_s = SparseGF2(n)
sim_t = stim.TableauSimulator()
for i in range(n):        # Bell-pair init on Stim
    sim_t.h(i); sim_t.cx(i, n + i)

S_h = np.array([[0, 1], [1, 0]], dtype=np.uint8)
sim_s.apply_gate_1q(0, S_h)
sim_t.h(0)

# RREF both stabilizer groups and compare.
mat_s = sim_s.extract_sys_matrix()
fwd = sim_t.current_inverse_tableau().inverse(unsigned=True)
mat_t = np.zeros((2 * n, 2 * n), dtype=np.uint8)
for r in range(2 * n):
    ps = fwd.z_output(r)
    for c in range(n):
        if ps[c] in (1, 2): mat_t[r, c] = 1
        if ps[c] in (2, 3): mat_t[r, n + c] = 1

assert np.array_equal(gf2_rref(mat_s), gf2_rref(mat_t))
print("Stabilizer groups match")
```

For a complete worked example see
[`examples/ex02_stim_verification.py`](examples/ex02_stim_verification.py);
for the full RREF parity suite see `tests/test_stim_rref_verification.py`.

## 10. A random-Clifford circuit using the full 11,520-element group

```python
import numpy as np
import stim
from sparsegf2 import SparseGF2, warmup, symplectic_from_stim_tableau

warmup()

# Enumerate the entire 2-qubit Clifford group deterministically.
cliffords = list(stim.Tableau.iter_all(2))
symps = np.stack([symplectic_from_stim_tableau(t) for t in cliffords])

n = 64
p = 0.20               # area-law
rng = np.random.default_rng(42)
sim = SparseGF2(n)

depth = 8 * n          # 512 layers
for t in range(depth):
    # Brickwork edge selection: even matchings on even layers, odd on odd.
    offset = t % 2
    for q in range(offset, n - 1, 2):
        ci = int(rng.integers(0, len(symps)))
        sim.apply_gate(q, q + 1, symps[ci])
    # Uniform measurement: each of the n system qubits with probability p.
    for q in range(n):
        if rng.random() < p:
            sim.apply_measurement_z(q)

print(f"k = {sim.compute_k()}, abar = {sim.get_active_count():.2f}")
```

At `p = 0.20` (deep in the area-law phase) you should see small `k` and
`abar` of order 1.

## 11. Your first MIPT sweep

A minimum viable phase-transition sweep: fix `n`, vary `p`, count
surviving codes.

See [`examples/ex03_mipt_sweep.py`](examples/ex03_mipt_sweep.py) for a
complete runnable example; the essentials:

```python
import numpy as np
from sparsegf2 import warmup
from sparsegf2.circuits import CircuitConfig
from sparsegf2.circuits.runner import SimulationRunner, get_clifford_table

warmup()
cliff_table = get_clifford_table()

for p in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    cfg = CircuitConfig(
        graph_spec="cycle", n=64,
        matching_mode="round_robin", measurement_mode="uniform",
        depth_mode="O(n)", depth_factor=8,
        p=p, base_seed=42,
    )
    runner = SimulationRunner(cfg, clifford_table=cliff_table, warmup_jit=False)
    ks = [runner.run(sample_seed=s).k for s in range(20)]
    print(f"p={p:.2f}  <k>={np.mean(ks):5.1f}  P(k>0)={np.mean([int(k>0) for k in ks]):.2f}")
```

Expected: `<k>` starts near `n` at small `p`, crosses through order-1
values near `p_c ≈ 0.16`, and drops to 0 by `p ≈ 0.25`.

## 12. Using the `sparsegf2.circuits` subpackage

`sparsegf2.circuits` is the canonical high-level API for building
graph-defined random-Clifford circuits, running parameter sweeps, and
persisting results into a standardized Hive-partitioned directory.

### 12.1. One-shot sweep from Python

```python
from pathlib import Path
from sparsegf2.circuits import CircuitConfig, RunConfig, SweepDriver

circuit = CircuitConfig(
    graph_spec="cycle", n=32,          # placeholder; driver overrides
    matching_mode="round_robin",       # "round_robin" | "palette" | "fresh"
    measurement_mode="uniform",
    depth_mode="O(n)", depth_factor=8,
    p=0.15,                            # placeholder; driver overrides
    base_seed=42,
)
run = RunConfig(
    circuit=circuit,
    sizes=[32, 64, 128],
    p_min=0.05, p_max=0.30, n_p=16,
    n_samples_per_cell=200,
    save_tableaus=True,
    n_workers=8,
    output_root=Path("runs"),
)
run_dir = SweepDriver(run).run()
print("Run written to:", run_dir)
```

### 12.2. Equivalent CLI

```bash
py -3.13 -m sparsegf2.circuits \
    --graph cycle \
    --matching-mode round_robin \
    --sizes 32 64 128 \
    --p-min 0.05 --p-max 0.30 --n-p 16 \
    --samples 200 \
    --depth-mode O_n --depth-factor 8 \
    --save-tableaus \
    --workers 8
```

See [`sparsegf2/circuits/README.md`](sparsegf2/circuits/README.md) for the
full API and the semantics of the three matching modes.

## 13. Reading a run directory

Every run is Hive-partitioned by `(n, p)`, so standard data tools see the
partition keys as real columns without any custom schema.

```python
import polars as pl

df = pl.scan_parquet(
    "runs/<run_id>/data/**/samples.parquet",
    hive_partitioning=True,
).collect()

print(df.select([
    "n", "p", "sample_seed",
    "obs.k", "obs.tmi", "obs.p_k_gt_0",
    "diag.final_abar", "diag.runtime_total_s",
]).head(8))
```

Or, if you prefer SQL, with DuckDB (install separately — it is not a
SparseGF2 dependency: `pip install duckdb`):

```python
import duckdb
out = duckdb.query("""
    SELECT n, p, AVG("obs.k") AS mean_k, AVG("obs.p_k_gt_0") AS frac_survived
    FROM read_parquet('runs/<run_id>/data/**/samples.parquet',
                      hive_partitioning=true)
    GROUP BY n, p ORDER BY n, p
""").fetchall()
```

Tableau snapshots (when you ran with `save_tableaus=True`) live in
`tableaus.h5` per cell and can be loaded with `h5py`:

```python
import h5py
with h5py.File("runs/<run_id>/data/n=0064/p=0.1500/tableaus.h5", "r") as f:
    x_packed = f["x_packed"][:]           # uint64[S, 2n, ceil(n/64)]
    z_packed = f["z_packed"][:]
    seeds = f["sample_seed"][:]
```

## 14. The analysis pipeline

After a sweep produces `tableaus.h5`, the analysis pipeline walks the run
directory, rehydrates each sample back into a live `SparseGF2`, and writes
derived quantities next to the raw data:

```bash
# All default analyses (cheap + the two opt-out expensive ones).
py -3.13 -m sparsegf2.analysis_pipeline runs/<run_id>

# Skip expensive analyses for a first pass.
py -3.13 -m sparsegf2.analysis_pipeline runs/<run_id> \
    --skip weight_spectrum logical_weights

# Fill the skipped ones in later.
py -3.13 -m sparsegf2.analysis_pipeline runs/<run_id>
```

This writes `data/n=.../p=.../analysis/{distances, weight_stats,
entropy_profile, weight_spectrum, logical_weights}.{parquet,h5}` and a
run-level `aggregates.parquet` with per-cell means, quantiles, and Wilson
intervals for binary columns.

See [`sparsegf2/analysis_pipeline/README.md`](sparsegf2/analysis_pipeline/README.md)
for the per-analysis schema and for how to add a new analysis.

## 15. The plotting primitive

`plot_vs_p` loads any scalar column from a run's `samples.parquet` or
`analysis/*.parquet`, aggregates across samples with optional filtering,
and draws one curve per system size:

```python
from sparsegf2.plotting import plot_vs_p

# P(k>0) vs p with auto error detection (Wilson interval for binary columns).
fig, ax = plot_vs_p("runs/<run_id>", y="obs.p_k_gt_0")

# Conditional k/n: restrict to samples with k > 0 before aggregating.
fig, ax = plot_vs_p(
    "runs/<run_id>",
    y="k_over_n",
    filter="obs.k > 0",
    errors="bar",
    error_metric="sem",
    sizes=[32, 64, 128],
    title="Mean code rate | k > 0",
    save="k_over_n.png",
)
```

See [`sparsegf2/plotting/README.md`](sparsegf2/plotting/README.md) for the
full signature, the derived-column alias list, and error-metric options.

## 16. Reproducing the benchmarks

The four canonical benchmarks under `benchmarks/`:

| Script | Reproduces | Runtime |
|---|---|---|
| `benchmark_full_clifford_group.py` | the 46.5× speedup table in the README | hours at n=512 |
| `benchmark_mipt_runtime.py` | the full NN+ATA phase diagram JSON | overnight |
| `benchmark_stim_comparison.py` | the paired Stim vs SparseGF2 runtime JSON | hours |
| `plot_comprehensive.py` | 25 figures from the JSON data (PDF + PNG each) | ~1 minute |

Each of the three data-producing scripts has hardcoded `SIZES` and
`P_VALUES` lists near the top — edit those for a quicker smoke run.
`plot_comprehensive.py` only reads the JSON outputs from the others and
takes under a minute.

## 17. Troubleshooting

### "Numba takes several seconds on my first run"

That's the JIT compile. Call `sparsegf2.warmup()` once up front; after
that every kernel is cached and subsequent calls are microsecond-fast.
Numba also caches compiled kernels to `__pycache__/` across processes,
so the second `python` invocation pays no warmup cost unless you change
kernel source.

### `ModuleNotFoundError: No module named 'pyarrow'` / `'h5py'` / `'polars'`

Install the `circuits` extra:

```bash
pip install -e ".[circuits]"
```

or simply `pip install -e ".[all]"` to pick up every optional extra.

### The simulator says `k = 0` unexpectedly

Every measurement has a chance to remove a generator. In a sufficiently
deep circuit above the MIPT critical point, k drops to 0 — this is the
physics, not a bug. Reduce `p`, increase `n`, or shorten the circuit if
you expected `k > 0`.

### A Y-basis measurement gives a different stabilizer group than Stim

Known issue — see [CHANGELOG.md](CHANGELOG.md). Avoid `apply_measurement_y`
until the fix lands.

## 18. Where to go from here

- **For circuit construction:** the semantics of matching modes and the
  output-tree schema are covered in detail in
  [`sparsegf2/circuits/README.md`](sparsegf2/circuits/README.md).
- **For post-processing:** [`sparsegf2/analysis_pipeline/README.md`](sparsegf2/analysis_pipeline/README.md).
- **For plotting:** [`sparsegf2/plotting/README.md`](sparsegf2/plotting/README.md).
- **For examples:** [`examples/`](examples/) has four runnable scripts of
  increasing scope (basic simulator → Stim verification → MIPT sweep →
  multi-topology comparison).
- **For benchmarks and figures:** [`benchmarks/`](benchmarks/) holds the
  scripts that produce every figure shown in the main README.
