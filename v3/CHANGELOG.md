# Changelog (SparseGF2 v3)

## 3.0.0.dev8: fixes from the review

Behaviour changes (results of some existing seeds change):

* **`scramble_layers` with `schedule="matching"`**: the colour cycle now runs on
  from the scramble layers into the measured layers. It used to restart, so
  when `scramble_layers` was 1 more than a multiple of the number of colours
  the first measured layer repeated the last scramble matching and absorbed it
  (`scramble_layers=1` on a brickwork did nothing). Unchanged when
  `scramble_layers` is a multiple of the number of colours (e.g. `2n` on an
  even ring) and for the other schedules.
* The `contiguous_distance` probe records NaN, not 0, where the code
  dimension is 0, so `EnsembleResult.mean` and the API summaries average only
  trajectories that still hold a code (the summary's `count` says how many).

Fixes:

* Checkpoints given as an int64 array are sorted and deduplicated like any
  list (a repeated or out-of-order entry silently skipped every later
  checkpoint and censored purification times).
* `dataclasses.replace(spec, ...)` gives an independent spec: the compiled
  cache is per instance (a derived spec ran the original's graph, partner or
  probes, and could write past its result buffers), and `"final"`,
  `"every_layer"` and strides follow a new `layers`.
* `MonitoredCircuit` normalises numpy scalars (sizes from numpy arrays, numpy
  seeds), so `EnsembleResult.save` no longer fails after the run; it copies
  `system_order`, `gate_set` and `edge_weights` and keeps them read-only.
* `EnsembleResult` records the `SeedSequence` actually used
  (`seed_entropy`, `seed_spawn_key`, `seed_pool_size`; `result.seed_sequence`
  repeats the run, also for `seed=None` and spawned sequences) and the package
  `version`. `save` stores both. `concatenate` refuses results whose spec,
  seed or version differ, or that share trajectories.
* `Circuit.from_stim` reads Stim's instruction objects instead of its 6-digit
  text, so arguments keep full precision (`from_stim(c).to_stim() == c`).
* `Circuit.copy()`, `+`, `+=` and `*` copy `REPEAT` bodies, and editing a
  `REPEAT` body recompiles the circuit before it next runs.
* NaN probabilities are rejected; anti-Hermitian MPP products are rejected
  when the circuit is built, with the line number; `#` inside an
  instruction tag is not a comment, and `REPEAT` tags are kept.
* `Simulator.run` and `sample` raise a `TypeError` that names
  `Circuit.from_stim` for a `stim.Circuit` (and `Circuit(text)` for text),
  take `max_ops`, and name the instruction of a bad `rec[-k]` lookback.
  `measure_pauli`, `peek_pauli` and `is_deterministic` accept `!` on any
  factor, as Stim's MPP does.
* `observables.contiguous_code_distance` returns 0 when no contiguous run is
  uncorrectable (system plus reference need not be pure); observables refuse
  fractional qubit indices; `analysis.kaplan_meier` refuses negative times
  (the `-1` censoring marker) with a message saying what to pass.
* `examples/purification_time.py` saves the result before printing, and
  reports the median as "> T_max" when more than half the trajectories are
  censored (it crashed and lost the run).
* Graph banks: the engine colours graphs only for `schedule="matching"`, and
  `graphs.from_edges` skips re-sorting an already normalised edge array, so
  preparing a Watts-Strogatz bank for `schedule="random_edges"` costs about
  13 us per graph at n = 32 and 20 us at n = 256 (it was 190 us and 970 us).
  `EnsembleResult.elapsed_seconds` now includes that preparation.
* API: list `edge_weights` follow their edges when an explicit edge list is
  normalised (they were applied in the user's order to the sorted edges), and
  are refused with random graph families; `spec.json` stores the spec
  exactly and resumes compare it exactly; resuming keeps the original
  provenance, records each resume, and refuses another package version;
  `estimate_ensemble` counts graph generation and preprocessing, and
  `run_ensemble` reports wall-clock time; the MCP `run_ensemble` builds the
  spec once; phase-free `stabilizers` are unsigned; full profiles are saved
  with `output_dir` and cuts are flagged; oversized summaries show fewer,
  evenly spaced checkpoints (first and last kept); out-of-range integers are
  input errors.
* Documentation: replay with `coins` is exact only when resets act on
  unentangled qubits; the reference record can differ from Stim's for
  multi-target X/Y-basis instructions (deterministic detectors are
  unaffected); circuits are lists of Python objects (DESIGN.md); the
  single-reference tau is O(log L) after a scramble (LITERATURE.md);
  `bernoulli_erasure_curve` needs `1 - correctable_fraction` for the failure
  probability; the PRL production speed-ups over v2 are measured end to end
  (spec construction and graph preparation included): 26–49× per thread,
  69–122× on 4 cores (the README said 27–60× and 90–210×, engine time only);
  the throughput figure labels its phase-free curve; the phase-free margin
  over Stim at n = 512–1024 is 9–15× (not 11–15×).
* `benchmarks/make_figures.py --outdir` writes `tables.md` there too, never
  over the committed copy.

## 3.0.0.dev7: production protocol, cheap code observables, agent interface

* **Graph disorder in the engine.** `MonitoredCircuit(edges=[g0, g1, ...])`
  takes a graph bank; trajectory `t` runs on graph `t mod G` (graphs may have
  different edge counts; `edge_weights` then takes one array per graph).
  `spec.graphs`, `spec.num_graphs`; banks round-trip through
  `EnsembleResult.save/load`. Graph seeds may be tuples, e.g.
  `graphs.watts_strogatz(n, 4, beta, seed=(17, i))`.
* **`scramble_matchings`**: layers of uniformly random all-to-all perfect
  matchings before the dynamics (the PRL production scramble).
* `examples/purification_time.py` runs the PRL single-reference protocol; on
  identical graph banks its purification-time distribution agrees with the v2
  production engine (`benchmarks/crosscheck_v2_production.py`, KS test).
* **Reference rows.** With a reference, the engine prefers reference-free
  pivots and keeps the set of stabilizer rows touching the reference. The
  code dimension is then a popcount and I(R:E) = rank(E) - rank(E over the
  other rows), so no reference column enters an elimination: k(t) at every
  layer costs nothing extra (41 -> 13 us per layer at n = 128, purification
  picture), erasure profiles and contiguous distances are 1.3-2x faster.
  Results are unchanged (the stabilizer group is the same; tests compare the
  new kernels with the generic ones on random codes).
* Removed the `mean_stabilizer_weight` probe: it depended on the generating
  set, not on the state, so its value changed with implementation details.
  `observables.stabilizer_length_distribution` (clipped gauge) is the
  gauge-invariant alternative.
* **LLM interface.** `sparsegf2_v3.api` (JSON-in, JSON-out service layer), the
  `sparsegf2` CLI and the `sparsegf2-mcp` MCP server (`pip install
  "./v3[mcp]"`), with background jobs for long ensembles; `AGENTS.md`,
  `llms.txt`, `docs/LLM_INTERFACE.md`, the repository's `.mcp.json` and a
  Claude Code skill.
* Benchmarks against Stim, Qiskit Aer, Cirq and v2
  (`benchmarks/compare_simulators.py`), physics data and README figures
  (`benchmarks/physics_data.py`, `benchmarks/make_figures.py`).

## 3.0.0.dev6: literature-driven observables

* New probes: `mutual_information`, `tripartite_information` (default: the
  first three quarters of the system), and vector probes `entropy_profile`,
  `erasure_lost_profile`, `erasure_correctable_profile` (I(R:E_m) for every
  erasure size m in one incremental pass). `record_profile` is replaced by the
  `entropy_profile()` probe; vector probes have shape
  `(trajectories, checkpoints, n + 1)`.
* Long-range circuits: `edge_weights` for `schedule="random_edges"` and
  `graphs.power_law_weights` (P(r) ~ r^-alpha, Block et al.).
* `EnsembleResult.survival()`: P(tau > layer) at every checkpoint.
* Observables: `clipped_gauge`, `stabilizer_length_distribution`,
  `erasure_profile`.
* `sparsegf2_v3.analysis`: `kaplan_meier` (Greenwood errors),
  `bernoulli_erasure_curve` (erasure threshold curves from per-size data),
  `mean_and_error`.

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
