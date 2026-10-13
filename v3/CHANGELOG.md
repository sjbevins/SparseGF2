# Changelog (SparseGF2 v3)

## 3.0.0.dev5: the monitored-circuit ensemble engine

* `sparsegf2_v3.graphs`: edge-array graphs (`cycle`, `path`, `complete`,
  `lattice_2d`, `watts_strogatz`, `newman_watts`, `erdos_renyi`,
  `random_regular`, `from_networkx`, `from_edges`) with platform-independent
  seeded generation, plus `edge_coloring` (brickwork on any graph).
* `sparsegf2_v3.mipt`: `MonitoredCircuit` protocol spec (pictures: none,
  purification, single reference with fixed or random partner; scrambles;
  schedules: matching, random_edges, all_edges; gate sets; checkpoints; stop at
  purification) and probes (`code_dimension`, `half_cut_entropy`, `entropy`,
  `contiguous_distance`, `lost_information`, `erasure_lost_information`,
  `erasure_correctable_fraction`, `mean_stabilizer_weight`, profiles).
* `spec.run(trajectories, seed=...)` generates and simulates every trajectory in
  compiled code, in parallel; results are independent of thread count and
  chunking (`first_trajectory`, `EnsembleResult.concatenate`), and saved/loaded
  with `EnsembleResult.save/load`. `spec.circuit(t, seed=...)` returns the exact
  circuit of trajectory `t`.
* Tests: engine output equals the general Simulator running the exported
  circuits, probe by probe at every checkpoint, for five protocol variants.

## 3.0.0.dev4: milestone 4, observables and erasure protection

* `sparsegf2_v3.observables`: `entropy`, `mutual_information`,
  `tripartite_information`, `entropy_profile` (all prefix entropies in one
  incremental pass), `code_dimension` (S(R)), `lost_logical_information`
  (I(R:E)), `erasure_is_correctable`, `recovery_probability` (2**-I(R:E)),
  `random_erasure_statistics` (erasure-threshold curves),
  `contiguous_code_distance`, `stabilizer_weights`.
* Compiled kernels for all of the above (incremental GF(2) bases on the column
  bitsets), reusable by the ensemble engine.
* Tests against independent state-vector calculations and brute-force
  definitions, at sizes across 64-bit word boundaries.

## 3.0.0.dev3: milestone 3, circuits and compiled execution

* `Circuit`: Stim's circuit language (parse, print, append, REPEAT, tags),
  lossless `from_stim`/`to_stim`, and validation with line-numbered errors.
  Supports every Clifford gate, `M MX MY MR MRX MRY MXX MYY MZZ MPP R RX RY`,
  inverted results (`!q`), classically controlled Paulis (`CX rec[-1] q`),
  Pauli noise (`X/Y/Z_ERROR`, `DEPOLARIZE1/2`, `PAULI_CHANNEL_1/2`, `M(p)`),
  `DETECTOR`, `OBSERVABLE_INCLUDE`, `QUBIT_COORDS`, `SHIFT_COORDS`, `TICK`.
* Extensions `CLIFFORD_1Q(i...)` / `CLIFFORD_2Q(i...)` apply Cliffords by
  canonical index; `to_stim()` expands them into Stim gates.
* `Simulator.run(circuit, coins=None, observe=None)`: one compiled Numba call;
  `observe(sim, tick)` runs at every TICK to measure the state as it evolves.
  `RunResult` has `measurements`, `random`, `detectors`, `observables`,
  `observations`. Detectors/observables follow Stim's convention (XOR with the
  noiseless reference sample).
* `sample(circuit, shots, seed=..., num_threads=...)`: many shots in parallel;
  results independent of the thread count.
* Tests: Stim-generated QEC circuits round-trip exactly; random noiseless
  circuits replayed with Stim's record reproduce every outcome, final signed
  state and deterministic detector/observable; 11 noise channels match Stim's
  sampler statistically; a noisy d=3 surface code matches Stim's per-detector
  rates.

## 3.0.0.dev2: milestone 2, the kernel and `Simulator`

* Column-major, bit-packed tableau kernel (Numba): one 64-bit word holds one
  Pauli bit of 64 generators, so gates are word-parallel and a measurement finds
  all anticommuting generators in one bitset read.
* `Simulator(n, phases=True|False, seed=...)` with Stim-style methods: every
  Clifford gate by name (`sim.h`, `sim.cx`, `sim.iswap`, ...), `do`,
  `clifford_1q/2q` by canonical index, `apply(Clifford)`, `measure`,
  `measure_x/y`, `measure_many`, `measure_pauli` (MPP syntax), `reset`,
  `reset_x/y`, `peek_z/x/y`, `peek_pauli`, `is_deterministic`, `stabilizers`
  (Stim canonical form), `destabilizers`, `stabilizer_arrays`, `entropy`,
  `copy`, `from_tableau`.
* Outcomes are physical in phases mode; phase-free mode returns `None` and
  raises `PhaseFreeError` for sign-dependent queries. `coin=` forces random
  outcomes; every collapse consumes one bit of the xoshiro256** stream.
* Lock-step tests against `stim.TableauSimulator` with forced outcomes: exact
  signed canonical stabilizers after every operation, deterministic outcomes,
  randomness flags, peeks; random initial states; sizes across 64-bit word
  boundaries; plus state-vector checks of the entropy.

## 3.0.0.dev1: milestone 1, foundations

* Audit of v2 (`docs/AUDIT.md`) and the v3 design proposal (`docs/DESIGN.md`).
* `PauliString`: immutable Pauli operators with exact phases, Stim's notation
  and sign conventions; products, commutation, matrices, Stim conversion.
* `Clifford`: k-qubit Clifford unitaries as signed tableaux; conjugation,
  composition, inverse, symplectic matrix, conjugation tables, Stim conversion.
  Canonical, Stim-independent indexing of the 24 one-qubit and 11,520
  two-qubit Cliffords (`Clifford.from_index`, `Clifford.index`,
  `random_clifford_indices`).
* `gate()` / `all_gates()`: every one- and two-qubit Clifford unitary in Stim
  (46 gates plus aliases), identical names and conventions.
* Test suite (156 tests, plus doctests) checking all of the above against Stim, plus
  Stim-independent checks against dense matrices.
* `benchmarks/prototypes/`: the layout prototypes behind the kernel design.
