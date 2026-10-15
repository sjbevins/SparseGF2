# Changelog

## Unreleased

* **Exact erasure counts:** `StabilizerCode.uncorrectable_counts` and the
  `uncorrectable_region_counts` probe add up each count in int64 while
  C(n, m) < 2**63, so counts above 2**53 no longer carry accumulated rounding
  error. `uncorrectable_counts` now returns int64 whenever every C(n, m) is below
  2**63 (before: 2**53). The probe stores float64, so it holds the exact count
  rounded once, and that value never exceeds float(C(n, m)). Sizes with C(n, m) >= 2**63 use a
  compensated float64 sum.
* **Rounded counts accepted:** `analysis.exact_failure_by_size` accepts float
  counts up to C(n, m) (1 + 1e-12) and clips the ratio to [0, 1]. Counts rounded
  just above C(n, m) (for example n = 200, `max_size` 13) used to raise
  `ValueError` in every hybrid estimator and in the `run_ensemble` summary (CLI
  and MCP).
* **P_C(m) from integer counts:** `analysis.failure_by_size` adds up the integer
  sample counts and divides once, so P_C(n) is exactly 1 for every code with
  k > 0. `junction_dip` no longer reports rounding one ulp below 1 as a dip. Phi
  and mu may change in the last bit.
* **One hybrid estimator:** `analysis.erasure_tolerance` now computes Phi with
  the same function as `erasure_failure_probability`, no longer with a copy of
  its own.
* **Breaking: randomness contract 2.** `mipt.RANDOMNESS_CONTRACT` is now 2.
  Random streams now use all 128 bits of their key (`stream_state` used only
  `key0 ^ key1` before), so a fixed seed gives other random numbers everywhere
  a stream is drawn: the circuits and probe samples of `MonitoredCircuit.run`,
  `replay` and `circuit`, the shot streams of `sample()` (and of `run_circuit`
  with more than one shot), and the subsets of the `random_erasure_statistics`
  and `erasure_profile` observables. A `Simulator`'s own stream, the graph
  families and `StabilizerCode.first_failure_sizes` are unchanged. Results
  saved before this version cannot be extended: `EnsembleResult.concatenate`
  refuses to merge them with new chunks, and `start_ensemble_job` refuses to
  resume or extend their output_dir ("this run was made under randomness
  contract 1; this version uses 2; start a new output_dir"). They still load
  and summarize.
* **Replay reproduces the sampled probes:** every evaluation of
  `erasure_first_failure` or `erasure_lost_profile` draws its orderings from a
  stream of its own, keyed by the trajectory, the probe's name and the layer.
  Their samples no longer depend on the other probes, the checkpoints or the
  earlier evaluations, so `replay(t, probes=..., checkpoints=...)` reproduces
  them, as its documentation always said. Fewer samples are a prefix of more,
  and renaming a probe gives it other orderings.
* **The contract is recorded:** `EnsembleResult.randomness_contract` is saved
  with every result (files without it load as 1), `provenance.json` records
  `randomness_contract` next to the package version, and the `run_ensemble`
  and `load_results` summaries show it.
* **`EnsembleResult.replay(t, probes=..., checkpoints=...)`** replays
  trajectory `t` of a saved run with the run's own spec and seed, and refuses a
  result made under another randomness contract: this version cannot
  reproduce the trajectories of a run made under contract 1.
  `MonitoredCircuit.replay(t, seed=...)`, `circuit(t, seed=...)` and
  rerunning a saved `spec.json` repeat a run only under the contract it was
  made under, and do not check it.
* **Compiled kernels are never stale:** importing the package deletes Numba's
  cached kernels whenever any kernel source, or the Numba or Python version, has
  changed since they were compiled. Numba checks only each function's own file,
  so before this a cached caller could keep running an old callee from another
  file (for example `sample()` with an old `stream_state` after a change to
  `rng.py`). The first import after an update, including this one, recompiles the
  kernels once (10 to 30 s).

## 3.0.0.dev11: distances from counts, graph files and statistics, saved states, replay

* **mu_C and distance bounds** from the saved erasure arrays:
  `analysis.mean_first_failure` (mu_D = E[D] = sum_m (1 - P_C(m)) per code),
  `analysis.distance_bounds` (exact d where the exhaustive counts fix it,
  otherwise `w + 1 <= d <= min_j D_j`), `analysis.hybrid_failure_by_size` (the
  exact-plus-sampled P_C(m) curve) and `analysis.junction_dip` (flags sampling
  noise at the junction of that curve).
* **Graph statistics:** `graphs.stats` (algebraic connectivity, mean path
  length, clustering, degrees; up to 4096 nodes) and `graphs.shortcut_fraction`.
* **Graph files:** `graphs.save` and `graphs.load` write and read graph
  collections as `.json` or `.npz`, and `{"family": "file", "path": ...}` runs an
  ensemble on any saved geometry (a file with several graphs is a collection).
* **Codes:** `StabilizerCode.contiguous_distance` and
  `StabilizerCode.from_simulator`, both checked against the engine's probes.
* **Exact states:** `Simulator.save` and `Simulator.load` restore the tableau,
  the signs and the random-stream position, so a restored simulator continues
  bit for bit.
* **Replay:** `MonitoredCircuit.replay(t, seed=...)` reruns trajectory `t` with
  other probes or checkpoints and matches the original run exactly
  (`mipt.RANDOMNESS_CONTRACT = 1`).
* **Stim cross-check:** `crosscheck.compare_stabilizers` and
  `crosscheck.compare_measurements` compare any circuit with Stim after every
  `TICK`.
* **Planned work, specified but not built:** the general agent interface (tool
  stacks) in `docs/specs/2026-10-13-agent-tool-stacks-design.md` and the reverse
  log in `docs/specs/2026-10-13-reverse-log-design.md` (issue #2).

## 3.0.0.dev10: v3 becomes the `sparsegf2` package

SparseGF2 v3 is promoted to the repository root and is now the only
maintained version.

* **Import and distribution name `sparsegf2`** (was `sparsegf2_v3` /
  `sparsegf2-v3`). Replace `import sparsegf2_v3` with `import sparsegf2`
  and reinstall with `pip install -e ".[mcp]"` from the repository root.
  Saved results load unchanged (`EnsembleResult.load` does not store the
  module name).
* **v2 is deprecated** and no longer in the tree. Its last state is the
  branch `v2-legacy`; the tag `v2.0.0` is unchanged. Do not install v2 and
  3.x in one environment: both import as `sparsegf2`.
* **Layout:** the former `v3/` directory is the repository root (`src/`,
  `tests/`, `docs/`, `examples/`, `benchmarks/`). The layout prototypes in
  `benchmarks/prototypes/`, which imported v2, were removed (they remain in
  the history; see `docs/DESIGN.md`).
* **Windows:** every text file the package, tests and benchmarks read or
  write is opened as UTF-8, so the suite passes under the default Windows
  code page (cp1252) instead of failing in `tests/test_docs.py`.
* **CI** runs on pushes to `main` and `v3` and on pull requests: Ubuntu,
  macOS and Windows on Python 3.12, plus 3.11 (the floor) and 3.14.

## 3.0.0.dev9: erasure probes, graph collections, codes after the run

v3 is now developed for its own simulation campaigns: the v2 production
protocol, the cross-check against v2 and the v2 speed comparisons are gone
(see "Removed"). The old package still lives in the repository root.

Behaviour changes (some existing scripts and results change):

* **Graph collections replace `t mod G`.** `MonitoredCircuit(edges=[g0, g1,
  ...], circuits_per_graph=c)` runs `c` random circuits on each of the `G`
  graphs (default `c = 1`); trajectory `t` uses graph `t // c`, so each
  graph's circuits are consecutive and chunks keep graphs together. The old
  rule, trajectory `t` on graph `t mod G`, is gone everywhere (engine, API,
  examples). With one circuit per graph and as many trajectories as graphs,
  trajectory `t` still runs on graph `t`; asking for more than `G * c`
  trajectories, which used to wrap around, is now an error.
  `spec.num_trajectories` is `G * c` (`None` for a single graph), and
  `spec.run(seed=...)` without a count runs all of them;
  `run(trajectories, seed=..., first_trajectory=...)` still runs chunks.
  `circuits_per_graph` must be 1 for a single graph.
  `EnsembleResult.graph_index` gives each trajectory's graph, and
  `spec.with_edges(edges, edge_weights, circuits_per_graph)` swaps the graphs.
  The same collection (same graph seeds) can be reused at every measurement
  rate; with the same circuit seed the gates and random numbers are shared
  too.
* **Removed probes**: `erasure_lost_information`,
  `erasure_correctable_fraction` and `erasure_correctable_profile` (kinds
  `erasure_mean_lost`, `erasure_correctable`, `erasure_correctable_profile`).
  Use `erasure_first_failure` (P_C(m), the uncorrectable fraction at every
  size m), `uncorrectable_region_counts` (exact P_C(m) for small m),
  `erasure_lost_profile` or `lost_information`.
* **Exact values after a stop.** With `stop_when_purified`, evaluations after
  a trajectory stops hold the probe's exact value for a pure reference instead
  of NaN: `code_dimension`, `lost_information`, `erasure_lost_profile` and
  `uncorrectable_region_counts` 0, `erasure_first_failure` n + 1,
  `code_snapshot` n (no code stored; snapshot count -1). Entanglement probes
  (`entropy`, `half_cut_entropy`, `mutual_information`,
  `tripartite_information`, `entropy_profile`) and `contiguous_distance` stay
  NaN there. Code-rate and erasure averages over all trajectories need no NaN
  handling.

New:

* **Per-probe checkpoints.** Every probe factory takes `checkpoints=`
  (`"final"`, `"every_layer"`, a positive stride or a list of layers; default:
  the spec's `checkpoints`), so a cheap probe can run every layer and an
  expensive one at a few depths. `values[name]` has shape `(trajectories,
  evaluations)` or `(trajectories, evaluations, length)`, one evaluation per
  layer of `result.layers(name)` (also `spec.probe_layers(name)`). The
  purification layer is still detected only at the spec's `checkpoints`.
* **`erasure_first_failure(samples=100)`**: for each of `samples` uniformly
  random orderings of the system qubits, the first-failure size D, the
  smallest m such that erasing the first m qubits leaks logical information
  (I(A:R) > 0); n + 1 when k = 0. The fraction of samples with D <= m
  estimates P_C(m) for every m at once.
* **`uncorrectable_region_counts(max_size=2)`**: the number of uncorrectable
  m-qubit erasures for m = 1..w, by exhaustion (exact P_C(m) = count / C(n, m);
  all zero exactly when the distance exceeds w). Default name
  `uncorrectable_region_counts[w]`.
* **`code_snapshot()`**: saves the code's stabilizer generators at its
  checkpoints (at most one per spec; give it its own, e.g. `"final"`). Its
  value is the number of generators, n - k. `result.snapshots[name]` holds the
  packed generators and counts, `result.code(t, c)` returns a
  `StabilizerCode`, and snapshots are saved and loaded with the result
  (`.npz`).
* **`codes.StabilizerCode`** (also `sparsegf2_v3.StabilizerCode`): a code from
  an (s, 2n) 0/1 array (X bits, then Z bits) or Pauli strings (`"XZZX_"`;
  commuting, not necessarily independent), or from packed rows
  (`from_packed`). `num_qubits`, `num_stabilizers`, `num_logical`,
  `stabilizer_matrix()`, `lost_information(A)` (l(A) = I(A:R)),
  `is_correctable(A)`, `first_failure_sizes(samples, seed=...)` and
  `uncorrectable_counts(max_size)`, computed by the engine's kernels.
* **Erasure estimators in `analysis`**: `failure_by_size` (P_C(m), m = 0..n,
  from first-failure samples), `exact_failure_by_size` (exact P_C(m),
  m = 1..w, from counts), `erasure_failure_probability` (Phi_C(e) =
  sum_m C(n,m) e^m (1-e)^(n-m) P_C(m) = mean_j Pr[Bin(n, e) >= D_j], for
  independent erasure with probability e; with `exact_counts`, the hybrid
  estimator that is exact for m <= w) and `erasure_tolerance`
  (e_delta = sup{e : Phi(e) <= delta} for an ensemble, by bisection).
  Purified codes (D = n + 1) give Phi = 0.
* Examples: `examples/code_and_erasure.py` runs the erasure workflow versus
  depth on a graph collection (`--graphs`, `--circuits-per-graph`);
  `examples/purification_time.py` takes `--graphs`, `--circuits-per-graph` and
  `--graph-seed`.

Interfaces (API, CLI, MCP):

* JSON spec key `circuits_per_graph`; `"realizations": G` in `graph` builds a
  collection (graph `i` from seed `[seed, i]`). `trajectories` may be omitted
  for a collection (all `G * c` run); asking for more than `G * c` is an
  error.
* MCP tools `run_ensemble(spec, trajectories=None, ...)`,
  `estimate_ensemble(spec, trajectories=None)` and
  `start_ensemble_job(spec, output_dir, trajectories=None, ...)` (`output_dir`
  now comes before `trajectories`). CLI `sparsegf2 ensemble`: `--trajectories`
  is optional for graph collections.
* Probes accept `"checkpoints"` in JSON. Summaries show each probe on its own
  layers (a `layers` key appears when they differ from the spec's);
  `erasure_first_failure` and `uncorrectable_region_counts` are summarized as
  P_C(m) curves (index = m) with a `what` description, `code_snapshot` as the
  mean generator count with a note (the codes are saved only with
  `output_dir`).
* `estimate_ensemble` returns `trajectories` and `result_megabytes` and
  counts each probe on its own layers; runs whose result would exceed 8 GB
  (`api.MAX_RESULT_GB`) are refused with a message saying what to change.
* The MCP erasure prompt uses `erasure_first_failure` and
  `uncorrectable_region_counts`.

Performance:

* The erasure kernels work on the columns of the system qubits in a row basis
  adapted to the code (the first s rows span S, all r = n + k rows span
  N(S)). First-failure scans skip the N(S) rank while the S-rank is full and
  build the N(S) basis once, at the first deficient prefix. Exhaustive counting
  prunes failing prefixes, builds N(S) bases only when needed, and finds the
  uncorrectable last qubits of each prefix with hash lookups over the prefix's
  span. Measured one-thread costs per evaluation at n = 256 (codes from random
  circuits): `erasure_first_failure` with 100 samples 3 to 7 ms;
  `uncorrectable_region_counts` about 1 ms for w = 2, 40 to 50 ms for w = 3
  and about 7 s for w = 4 (C(256, 4) = 1.7e8 sets; practical only at a few
  depths); `code_snapshot` under 1 ms. Trajectories run in parallel over all
  cores.

Removed:

* The v2 cross-check and production benchmark:
  `benchmarks/crosscheck_v2_production.py`,
  `benchmarks/results/crosscheck_v2.json`, the production figure
  (`docs/figures/production-*.png`) and table, and every PRL production
  protocol speed-up claim.
* The v2 engine in the simulator comparison (`benchmarks/compare_simulators.py`,
  `benchmarks/results/simulators.json`, the figures and `tables.md`).
* The migration guide, `docs/MIGRATING_FROM_V2.md`.

Documentation: `docs/DESIGN.md` documents the probe registry and how to add a
probe, graph collections, and the erasure kernels; `README.md`, `AGENTS.md`,
`llms.txt` and `docs/LLM_INTERFACE.md` describe the erasure workflow.

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
