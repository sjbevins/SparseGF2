# Changelog

All notable changes to `sparsegf2` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `CircuitConfig(total_layers_override=T)` for a literal measured-layer cap,
  independent of the depth-mode formula.
- `single_qubit_entropy`, an exact singleton stabilizer-entropy fast path for
  per-layer single-reference first-passage checks.
- Layer checkpoints on `simulate` and `SimulationRunner.run`, including
  read-only live callbacks and the `CHECKPOINT_STOP` sentinel for
  checkpoint-granular early stopping.
- Deterministic `gating_mode="all_edges"`, which fires the graph's full stored
  edge list each layer.
- Degree-preserving Watts--Strogatz graphs through `watts_strogatz(...)` and
  parameterized string specifications such as
  `"watts_strogatz(k=2,beta=0.1)"`, with graph provenance recorded on every
  sample.
- Dtype-preserving analysis-array NPZ I/O and general finite-size-scaling
  collapse tools.
- `SparseGF2.project_pauli_into_pair`, the code-space projection primitive
  needed to expurgate a selected logical pair even when the Pauli is a
  stabilizer of the pure-state tableau used to present the code.
- A resume-safe PRL production workflow for exact-layer single-reference
  purification times on mean-degree-four Watts--Strogatz graphs, including
  deterministic data files, censoring flags, live status, and scalar/batch
  parity validation.
- Manifest-validated live Kaplan--Meier summaries, graph-bootstrap median error
  bars, and preliminary raw-curve plots for the production campaign.
- Bounded three-parameter purification-time collapse tools with a profiled
  smooth master curve, deterministic multistart fitting, and all three
  pairwise profiled loss landscapes.
- A versioned SQLite graph registry for 350,000 independently indexed
  mean-degree-four Watts--Strogatz draws, with deterministic tuple-keyed seed
  derivation, strict provenance, integrity checks, and resume-safe insertion.
- Complete per-graph realized-rewiring and combinatorial-Laplacian algebraic-
  connectivity analyses for the paper graph registry, including compact cell
  summaries, conditional histograms, cumulative normalized-gain estimates,
  deterministic bootstrap intervals, and reproducible figure recipes.
- A graph-family-agnostic raw purification-time workflow over Cartesian graph
  parameter grids, with exact decimal measurement grids, stable named RNG
  streams, per-graph/per-circuit censoring data, plan-first launch guards,
  runtime-attempt auditing, generator- and environment-bound edge banks,
  resume-safe checkpoint journals, and a coordinator-owned SQLite artifact
  catalog with container and logical-result digests.
- A real-kernel single-reference benchmark harness with stage timings,
  one-thread-per-process safeguards, work/storage estimates, and bounded
  Windows process-scaling measurements.

### Changed

- Circuit gates and Z measurements use ordered batched Numba kernels while
  preserving scalar operation order, RNG draws, hybrid mode-switch points, and
  final tableaux.
- Per-sample RNG streams use the pair `(base_seed, sample_seed)` rather than
  their sum. This removes collisions between distinct seed pairs but changes
  seeded trajectories relative to 2.1.0.
- Integer-valued circuit, Pauli, erasure, and expurgation inputs reject Boolean
  aliases and lossy numeric coercions. Measurement rates and probability
  targets must be finite real numbers in their documented ranges.
- Atomic analysis writes use unique same-directory staging paths, so concurrent
  writers cannot clobber one another's temporary payload.

### Fixed

- The `analysis` extra now installs SciPy, which its finite-size-scaling and
  production-collapse routines require; minimal-runtime CI skips only tests for
  optional plotting/scaling dependencies. GitHub Actions now use the Node
  24-based checkout and Python setup releases.
- Expurgation now always consumes exactly one logical pair for every validated
  logically nontrivial measurement. Previously an encoded logical-Z operator
  could be mistaken for a deterministic pure-state measurement, or a
  pre-existing gauge pair could absorb the pivot; either path left `k`
  unchanged and violated the algorithm's accounting and monotonicity contract.
- The optional PDF drawing test now checks for the required LaTeX classes and
  packages, rather than failing merely because `pdflatex` exists without
  `standalone` or `quantikz`.
- Production readers take short, share-safe NPZ snapshots, and production
  workers retry transient Windows destination locks without changing completed
  trajectory data.
- Finite-size-scaling fits reject unsuccessful, nonfinite, and out-of-bounds
  optimizer results instead of reporting fallback parameters as valid fits.
- Production resume validation now rejects inconsistent incomplete rows and
  observed purification times beyond the configured depth cap.

## [2.1.0] - 2026-07-24

Expurgation: the code-surgery algorithm of Gullans, Krastanov, Huse, Jiang, and
Flammia, PRX **11**, 031066 (2021), Sec. VI, implemented natively on the
phase-free tableau (no signed simulator anywhere in the pipeline).

### Core simulator

- **`SparseGF2.measure_pauli(qubits, letters)`**: projective measurement of an
  arbitrary multi-qubit Pauli via the same Aaronson-Gottesman three-step update
  as `measure_z` (anticommuters by parity accumulation, `pivot_mode`-respecting
  stabilizer pivot, decouple / re-pair / install). Returns the pivot pair index,
  or `None` for a deterministic measurement (tableau unchanged, matching
  `measure_z`). Native sparse and hybrid-dense paths; cross-checked against
  Stim's `MPP` at the stabilizer-subspace level.
- **`SparseGF2.pauli_anticommuting_rows(qubits, letters)`**: the underlying
  commutation query (all generator rows anticommuting with a sparse Pauli),
  `O(sum |inv[q]|)` over the support in sparse mode, bit-packed popcounts in
  dense mode.
- **`gf2_eliminate_on_columns`** promoted from an observables-internal helper
  to `sparsegf2.core.linalg_gf2` (single source of truth): forward GF(2)
  elimination on a chosen pivot-column set, shared by `contiguous_distance`
  and the expurgation witness extraction.

### Expurgation (`sparsegf2.expurgation`)

- **`StabilizerCode`**: the (check / logical / gauge) role array over
  destabilizer/stabilizer pairs plus the elementary measure-and-relabel move.
  Tableau-agnostic: couples only to the public `SparseGF2` measurement and
  commutation API, so encoding circuits, `from_symplectic` imports, and
  hand-built tableaux all work; `from_encoding(sim, data_qubits)` covers the
  standard convention.
- **Erasure decoding**: `uncorrectable_matrix` (`M(S, L, e)` assembled from
  per-site commutation queries), `uncorrectable_rank`
  (`r_M = rank M - rank M_S`), exact `recovery_probability` (`2**-r_M`), and
  `expurgation_candidates` (augmented `[M | I]` elimination whose witness bits
  reconstruct the uncorrectable operators, lightest first).
- **Purification bridge**: `from_purification(sim, system_qubits)` extracts the
  system code of a monitored purification-picture tableau (system + reference
  qubits) into the presentation the machinery uses: the system-supported
  stabilizers become the checks (kernel of the reference-column map) and a
  symplectic Gram-Schmidt completes the logical pairs, with `k = S(system)`
  verified internally. Tests pin the extracted brute-force distance against an
  independent purification-side computation and the targeted-purification
  identity (measuring an extracted candidate on the original tableau lowers
  `code_dimension` by exactly one).
- **`expurgate` driver**: the published loop with mid-sequence re-validation,
  gauge and stabilizer strategies, frozen validation patterns (monotone
  before/after recovery comparison), and stopping criteria (`k_target`,
  `recovery_target`, barren rounds, `max_rounds`, `k = 0`), returning a full
  `ExpurgationResult` record; `random_encoding` builds brickwork or all-to-all
  random codes so the package runs standalone.
- **Tests**: the companion notes' worked examples traced verbatim (matrix
  entries, extracted candidates, a brute-force-verified distance increase from
  1 to 2), random-code cross-checks of `M` against dense symplectic products,
  loop accounting and monotonicity, and Stim `MPP` parity for the new kernel.
  See `src/sparsegf2/expurgation/README.md`.

## [2.0.0] - 2026-06-15

A self-contained, phase-free sparse stabilizer simulator over GF(2) with a
graph-circuit layer and an analysis/studies layer. Cross-checked against Stim at
the stabilizer-subspace level, with no runtime Stim dependency.

### Core simulator

- **`SparseGF2`**: phase-free sparse stabilizer tableau. `SparseGF2(n)` is
  exactly `n` qubits in `|0^n⟩`. Inverted-index storage with numba JIT kernels
  and pure-Python fallbacks; `pivot_mode="min_weight"|"first"`.
- **Hybrid sparse/dense mode**: `SparseGF2(n, hybrid=True)` (and
  `CircuitConfig(..., hybrid=True)`). The simulator monitors the tableau's
  stabilizer density (`a_bar`, the average column weight) and switches to a
  bit-packed dense representation while the state is volume-law, where the sparse
  inverted-index bookkeeping costs more than it saves, switching back to sparse
  when it thins. A 2x hysteresis band (enter above `max(n//4, 16)`, leave below
  half) checked every `max(n//2, 32)` ops keeps it from thrashing. Gates and
  measurements then run via dense Stim-style kernels (numba + pure-Python
  mirrors) with no index maintenance: ~2.5-3x faster on volume-law /
  low-measurement-rate circuits (e.g. all-to-all studies at low `p`), and
  ~parity in the area-law regime. The physical state and every gauge-invariant
  observable are identical to the pure-sparse engine; only the basis-dependent
  generator-weight diagnostics may differ, exactly as they do across
  `pivot_mode`. Default is `False`.
- **Gates**: `apply_h`, `apply_s`, `apply_sqrt_x`, `apply_cx`, `apply_cz`,
  `apply_swap`, and generic `apply_gate_1q` / `apply_gate_2q` (validated against
  the symplectic condition on every call).
- **Measurement**: `measure_z`, `measure_x`, `measure_y`, `is_deterministic_z`.
  Deterministic outcomes return `0` (the sign bit is not tracked).
- **Reset/projection**: `reset_z`, `reset_x`, `reset_y`.
- **State extraction**: `to_symplectic()` / `from_symplectic()` (round-trip),
  `canonical_form()` (RREF of the stabilizer subspace, the comparator for state
  equality), and copy / pickle support for checkpointing and trajectory
  branching.
- **Factories**: `from_zero_state(n)`, `from_bell_purification(n_system)`.
- **`Sp(2n, F₂)` toolkit** (`sparsegf2.core.symplectic`): two uniform samplers
  (kernel-basis and Bravyi-Maslov, selected via `random_symplectic(...,
  sampler=...)`), the cached `random_symplectic_2q` fast path, and enumeration
  (`enumerate_sp4`, `enumerate_symplectic_group`).
- **Observables** (`sparsegf2.core.observables`): `subsystem_rank`,
  `entanglement_entropy`, `mutual_information`, `tripartite_mutual_info`,
  `code_dimension`, `code_rate`, `contiguous_distance`, `generator_weights`,
  `stabilizer_weight_spectrum`, `average_stabilizer_weight`, and `active_count()`
  (the mean number of generators touching a qubit, the tableau-density
  diagnostic `a_bar`; dense-aware).
- **GF(2) linear algebra** (`sparsegf2.core.linalg_gf2`): `gf2_rref`, `gf2_rank`,
  `gf2_kernel_basis`, and the bit-packed `gf2_rank_bits` (64 columns per `uint64`
  word; ~7-27x faster at observable-sized matrices and never slower).
  numba-accelerated with pure-Python fallbacks; every observable's subsystem rank
  routes through the bit-packed path.
- **`SimulatorProtocol`**: observables couple against this structural protocol
  rather than the concrete class, so alternative backends can opt in.
- **Exception hierarchy** at `sparsegf2.errors`, each multi-inheriting from the
  fitting standard exception so existing `except ValueError:` callers still work.

### Circuits (`sparsegf2.circuits`)

- Graph-defined random-Clifford + measurement circuits with three pictures
  (pure state, purification, single reference).
- Arbitrary geometry via `from_networkx` plus built-in `cycle` / `complete` /
  `path` / `lattice_2d`.
- Gating modes `brickwork` / `random_edge` / `random_pool` (the last applies
  `O(n)` random edges per layer, default `n/2`), four measurement modes
  (`bernoulli` / `gated` / `random_pair` / `uniform_count`), early-stop
  (`depth_mode="until_purified"`, with an exact `purified_at_layer`) and
  per-layer time series.
- **`scramble_entangled_qubit`** (default `True`): for `single_ref` with
  `scramble=True`, choose whether the global scramble covers the reference's Bell
  partner (qubit `n-1`). `False` holds it out, so the probe qubit stays a
  localized Bell pair until the monitored dynamics spread it. The two settings are
  different single-qubit-probe protocols. `CircuitConfig.scramble_qubits()`
  exposes the scramble support, shared by the runner and the inspector.
- Each circuit run records `mean_active_generators` (`a_bar` averaged over the
  measured layers) on the `SampleRecord` and in the sweep/study rows.
- Text and visual circuit inspection (`inspect_circuit`, `draw_circuit`,
  `scripts/inspect_circuit.py`), reading one shared trace so all three always
  agree. The global `scramble` is drawn as one labeled block over exactly the
  qubits it acts on; the inspector and drawer draw a fresh random realization each
  call (the chosen seed is reported in the summary) and accept an explicit
  `sample_seed` to reproduce one. Diagrams render with the **quantikz2** LaTeX
  package (`circuit_to_quantikz` / `save_circuit` / inline `draw_circuit`); system
  wires are black and the reference register is red, with dashed `setup` / `t_1,
  t_2, …` separators marking the individual timesteps. One example per gating /
  matching / measurement mode lives in `docs/figures/gallery/`.

### Analysis & studies (`sparsegf2.analysis`)

- A registry of named, picture-aware observables usable both online
  (`simulate(..., analyses=[...])`, which computes then discards the tableau) and
  offline (`analyze` / `analyze_tableaux` over saved tableaux); custom
  `fn(sim, spec) -> value` callables work the same way.
- A parameter-`sweep` driver with on-disk parquet/HDF5 output. The parallel sweep
  yields results in completion order internally (`joblib
  return_as="generator_unordered"`, with an ordered fallback for joblib < 1.4)
  and restores submission order by index, so the progress bar ticks the moment any
  worker finishes instead of stalling behind the slowest head-of-queue cell on
  mixed-duration grids.
- The augmentable **`Study`** database: run a sweep saving tableaux, then
  `study.augment([...])` to compute new observables on the saved tableaux and
  merge them in keyed by cell, with no re-simulation (idempotent).
- **Crash-only run logging**: `Study.run` / `Study.augment` write a verbose
  descriptive trace to `<path>/logs`, kept only if the run fails (a clean run
  deletes it); `log_detail="trace"` adds a per-layer record. `Study.run` shows a
  single whole-study progress bar (the current `n`/`p`/seed while simulating, then
  the tableau write) as one 0-100% meter.
- **Plotting** (`plot_study` + `scripts/plot_study.py`): auto-detects every
  numeric observable column (so it picks up `Study.augment` columns), and shows
  uncertainty across seeds: error bars by default, `errorstyle="band"` for a
  shaded band, via `aggregate()`.

### HPC / headless

- `sparsegf2.analysis.available_cores()` resolves the usable core count from the
  scheduler allocation (`SLURM_CPUS_PER_TASK`, `PBS_NCPUS`, the CPU affinity mask,
  then `cpu_count`), and a sweep with `n_jobs=-1` fills that allocation rather than
  the whole node.
- `SPARSEGF2_PROGRESS=0` is a global progress-bar kill-switch for batch logs, so a
  job runs silently regardless of any `progress=True` in the code.
- `Study` output is resumable: chunks already on disk are skipped, so a re-queued
  job never redoes finished work, and the crash-only log names the exact cell if a
  run dies.
- `scripts/hpc_sweep_array.{py,sbatch}` is a SLURM job-array template that runs
  resumable study chunks across nodes, one task per chunk.

### Packaging

- MIT `LICENSE`, `py.typed` marker, project URLs and classifiers, and a CI
  workflow (ruff + pytest) running the full suite on Linux, macOS, and Windows, in
  two install configurations per OS (the bare `numpy`+`numba` floor and the full
  optional-extras set), so the analysis / parallel (`n_jobs>1`) / data / graph
  paths and the Stim-parity cross-checks are all exercised on every platform.
- The runtime floor stays numpy + numba; heavier dependencies (networkx, pyarrow,
  h5py, pandas, joblib, matplotlib) are optional extras (`graph`, `data`,
  `parallel`, `viz`, `analysis`), imported lazily only when used.
- Package-owned text metadata I/O specifies `encoding="utf-8"`, while a
  `.gitattributes` pins LF line endings, so the installed package and its test
  suite behave consistently on Windows, macOS, and Linux. Standalone benchmark
  and research scripts are outside this portability claim.
- Stim is its own `stim` extra, separate from `test`; the suite runs without it
  (the Stim-parity tests skip), so the simulator and tests install on any
  supported Python (3.12+) even when Stim has no wheel for it yet.

### Performance

- ~30x faster than the pure-Python reference via `@njit(inline="always")` on the
  position-map primitives; faster than Stim above `n ≈ 64` on the brick-wall
  benchmark (≈4x at `n = 256`, growing with `n`). `to_symplectic()` is vectorized
  and `canonical_form()` uses a JIT'd GF(2) RREF.
- Two-qubit gate hot loop (the largest single cost, ~54% of a circuit): the
  per-generator LUT is indexed by the packed Pauli pair `(xz_i<<2)|xz_j` and
  returns the packed result, so the inner loop is one lookup on the two `plt`
  bytes with no bit unpack/repack and an early-skip for unaffected generators; and
  the runner prebuilds each gate's LUT once from the fixed gate table instead of
  re-keying the symplectic on every gate.
- `until_purified` / `record_time_series` recompute the order parameter only on
  layers that actually measured (an entanglement entropy is invariant under the
  gates, and the references are never gated) instead of every layer, and skip
  provably redundant checks (the order parameter is non-increasing and drops at
  most 1 per measurement), keeping `purified_at_layer` exact while cutting the
  per-layer rank cost.
- `contiguous_distance` factors the reference out of the stabilizer block once
  (each window is then two small ranks instead of a fresh rank over ~n columns)
  and short-circuits fully purified cells, making it 100-3000x faster at
  `n = 128`, the difference between a multi-minute and a few-second
  `Study.augment`.
