# SparseGF2 roadmap

A short, forward-looking view of where the simulator is and what is planned.
Released history lives in [`CHANGELOG.md`](CHANGELOG.md).

## Where it stands

The simulator is feature-complete for single-trajectory MIPT studies:

- **Core**: phase-free sparse stabilizer tableau over GF(2): single- and
  two-qubit Cliffords, Z/X/Y measurement, factories (`from_zero_state`,
  `from_bell_purification`), reset, and a full `Sp(2n, F₂)` toolkit
  (enumeration + two uniform samplers, kernel-basis and Bravyi-Maslov). Numba
  JIT kernels throughout, cross-checked against Stim at the stabilizer-subspace
  level. An optional **hybrid** mode (`hybrid=True`) mirrors the tableau into a
  bit-packed dense representation and switches to it in the volume-law regime,
  recovering Stim-like per-step speed there while keeping the sparse engine's
  edge under heavy measurement, with identical physical results either way.
- **Circuits**: graph-defined random-Clifford + measurement circuits with three
  pictures (pure state, purification, single reference), arbitrary geometry via
  `from_networkx` plus built-in `cycle` / `complete` / `path` / `lattice_2d` /
  `newman_watts` / `watts_strogatz`, four gating modes (`brickwork`,
  `random_edge`, `random_pool`, `all_edges`), three matching modes, four
  measurement modes, exact purification stopping, literal depth overrides,
  selected depth checkpoints, per-layer time series, and text + visual circuit
  inspection.
- **Observables**: entanglement entropy, mutual information, tripartite mutual
  information, code dimension, code rate, contiguous distance, and generator-weight
  diagnostics.
- **Analysis**: a registry of named, picture-aware analyses usable both online
  (compute at end-of-circuit, discard the tableau) and offline (save tableaux,
  analyze later); a parameter-sweep driver with on-disk output; the augmentable
  `Study` database that lets you add new observables to saved tableaux and
  re-plot without re-simulating; and finite-size-scaling collapse helpers for
  critical-point and exponent estimation, including graph bootstrap support.
- **Expurgation** (`sparsegf2.expurgation`): the Gullans et al. (PRX 11,
  031066) code-surgery loop run natively on the tableau, built on the core's
  general `measure_pauli` kernel: role bookkeeping over pairs, exact erasure
  recovery `2**-r_M`, witness-based candidate extraction, and the driver with
  gauge/stabilizer strategies. Works with any tableau regardless of origin.
  See `src/sparsegf2/expurgation/README.md`.

## Planned

- **Batched simulation**: carry many trajectories in shared storage with batched
  RNG seeding and observable extraction, for cheaper trajectory averaging.
- **Growable per-generator storage**: start `supp_q` / `inv` small and realloc on
  overflow, to cut per-instance memory for area-law trajectories at large `n`.
- **Expurgation sweeps as studies**: record `k`, mean recovery, and candidate
  weights as `Study` columns so expurgation parameter scans (over `n`, depth,
  geometry, erasure model, strategy) run on the existing analysis rails; numba
  mirrors for `measure_pauli` if expurgation ever becomes a hot path.

These are enhancements; none blocks running studies today. The runtime floor
stays numpy + numba; every heavier dependency (networkx, pyarrow, h5py, joblib,
pandas, matplotlib) is an optional extra, imported lazily only when used.
