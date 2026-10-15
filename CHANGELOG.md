# Changelog

## Unreleased

* **Every MCP failure carries a message:** the server turns every exception into a tool
  error the model can read. A `ValueError` keeps the API's message; anything else reads
  `TypeName: message`, followed by `(internal error: please report it)` unless it is a
  `TypeError` (an input error too). The SDK used to send only `Error executing tool X`
  for anything but a `ValueError`. An internal error's traceback goes to the server's
  log (stderr). The CLI exits with 2 for a `TypeError` as well.
* **Names are checked before they are looked up:** a graph `family`, an `edge_weights`
  family, a probe or observable `kind` and a `capabilities` section that is not a string
  raise `ValueError` naming the field (`graph.family must be a string (one of [...]),
  got ['cycle']`) instead of `TypeError: unhashable type`, which the CLI reported as an
  internal error (exit 1) and MCP as a bare error.
* **`num_system` is bounded:** at most `MAX_QUBITS` (65,536), checked before any graph
  is generated, in `estimate_ensemble` too (`num_system: 10**12` ran out of memory
  generating the graph). `capabilities()["limits"]["max_num_system"]` gives it.
* **Directories that cannot be written or read are input errors:** an `output_dir` that
  cannot be created or written (no permission, a full or read-only drive, a path below a
  file), its lock included, raises `ValueError("output_dir DIR is not writable: ...")` in
  `run_circuit`, `run_ensemble` and `start_ensemble_job`. A saved part that stays
  unreadable after the retries (`result.npz` or a chunk that another program holds, or
  that this process may not read) raises `could not read results from DIR: ...`, naming
  the file, in `start_ensemble_job` as in `load_results`; it raised `PermissionError`
  (CLI exit 1, a bare MCP error). A `spec.json` that is not JSON, holds no object or
  records no seed, and a graph file that cannot be read, are refused naming the file. The
  CLI exits with 2 for a circuit, spec or observables file it cannot read or that is not
  UTF-8.
* **Strict MCP argument types:** integer, boolean and number tool arguments are checked
  as the API checks them. Pydantic's lax default ran one trajectory for
  `trajectories: true` and three for `"3"`, and took `seed: true` as seed 1; these, and
  `"false"` for `phases` or `"inf"` for `max_seconds`, are now refused with a message
  naming the argument. A whole-number float such as `4.0` still counts as 4. The input
  schema is unchanged.
* **Fresh seeds fit in a double:** `seed=None` draws a 53-bit seed (it drew 63 bits), so
  the reported seed survives clients that read JSON numbers as doubles (JavaScript MCP
  clients, Claude Code among them); a larger one came back rounded and repeated another
  run. An explicit seed of any size is still used and reported exactly.
* **Lock and merge details:** `load_results` no longer fails when a merge removes
  `chunks/` while it lists it (a missing `chunks/` holds no chunks, and the chunks are
  listed before `result.npz`, which a merge replaces first). The refusal of a lock taken
  on another system names that system (`pid P on host H (Linux pidns:N)`), since WSL
  and containers share the Windows or host name. A lock whose `created` is not an integer
  counts as having none; a malformed one made a live writer's lock look stale.
* **Kaplan-Meier survival is correctly rounded:** `analysis.kaplan_meier` computes
  `survival` as the exact product of (n_j - d_j) / n_j (`fractions.Fraction`), rounded to
  float64 once, instead of a running float product. A survival of exactly 1/2 used to
  come out as 0.5000000000000001, so the median rule "first time with survival <= 0.5"
  skipped it; the purification-time example no longer needs a tolerance. `stderr` and
  `at_risk` are unchanged.
* **Kaplan-Meier censoring time:** `EnsembleResult.censoring_layer` (the last spec
  checkpoint at which purification was tested, per trajectory) and
  `EnsembleResult.purification_times()` (`(time, observed)` for
  `analysis.kaplan_meier`) replace the recipe `np.where(tau >= 0, tau, final_layer)`,
  which censored at the last layer simulated and so misstated the curve whenever the
  checkpoints ended before `layers`. AGENTS.md, GETTING_STARTED, the example and the
  `kaplan_meier` error message use the new methods. `final_layer` is unchanged.
* **P_C(m) summaries condition on a code:** the `run_ensemble` / `load_results` curves of
  `erasure_first_failure` average only the trajectories that hold a code at each
  evaluation (some sample <= n), and report their number as `count_with_code` (also in
  `vs_depth`). They used to average in the trajectories with k = 0 as P_C = 0, so
  P_C(n) read below 1 and codes looked better protected the more of them were lost.
  `uncorrectable_region_counts` is conditioned the same way when the spec has a
  `code_dimension` probe, or else an `erasure_first_failure` probe, evaluated at each of
  its layers (the same schedule or a denser one); otherwise its `what` line and a note
  say that trajectories without a code count as P_C = 0. A conditioned curve is shown at
  the last evaluation where some trajectory holds a code (with `stop_when_purified` the
  last ones often have none); if none ever does, its mean is `null`.
* **Lost information says it is not conditioned:** the summaries of
  `erasure_lost_profile` and `lost_information` still average every trajectory, those
  without a code counting as 0 bits lost, and now say so in the profile's `what` line
  and a note on each.
* **`tau_median_all` needs a majority:** it is reported only when strictly more than half
  the trajectories purified within the horizon (`null` at exactly half, where the median
  is not determined), and is then the first checkpoint with at most half left.
  `examples/purification_time.py` applies the same rule to its Kaplan-Meier median
  (with a tolerance, since the Kaplan-Meier product can miss 1/2 by an ulp).
* **Notes survive the size cap:** cutting a result to fit keeps every note (it used to
  keep only the first two) and adds the truncation note once.
* **Subsampled vector probes are flagged:** an ensemble curve or `vs_depth` grid that
  leaves sizes or evaluations out sets `truncated: true` with a note, as `run_circuit`
  already did for its profiles. This is routine for the grid once n + 1 > 17. The
  default thinning of checkpoints to 50 is still reported by `checkpoints.subsampled`
  alone, as LLM_INTERFACE now says.
* **Unique observable names in `run_circuit`:** automatic names (`entropy#2`, ...) skip
  every name in use or given explicitly, so one value can no longer overwrite another;
  an explicit name is kept unless two observables are given the same one.
* **Exact tick limit:** the default `tick_stride` and the check on a given one count the
  last tick, so at most `MAX_TICK_ROWS` ticks are evaluated (it was one more at exact
  multiples, e.g. 10,001 of 20,000 ticks).
* **Resume and extend compare the run before computing:** `start_ensemble_job` on an
  existing `output_dir` builds the spec and compares its protocol with the saved run's
  (`mipt._protocol_difference`, which a merge of saved parts requires) before anything
  runs. A spec that only compared equal as text used to be accepted even when its graph
  file had changed: every new chunk then ran on other graphs, the final merge failed,
  and the directory could not be read until chunk files were deleted by hand. A spec
  written differently that builds the same protocol (a cutoff `budget` or per-size
  object resolving to the same w, a range string, a default written out, another path
  to the same graphs) now continues the run with a note; `spec.json` stores the new
  form. The randomness-contract and version refusals still come first, and the version
  is now also read from the saved results, so a missing or unreadable `provenance.json`
  no longer lets another version's results be extended (and then fail at the merge). A
  `spec.json` seed that differs from the saved results' seed is refused up front too.
* **Graph files are recorded:** for `{"family": "file"}`, `spec.json` records
  `graph_file`, the resolved absolute path and the sha256 of the file's bytes. A resume
  whose file has other bytes is refused up front, unless saved results show that it
  holds the same graphs (a regenerated `.npz` differs in its zip timestamps); with no
  results saved, it is refused. A `spec.json` from before this change is checked
  against the saved results alone. The CLI accepts `graph_file` in a run's `spec.json`.
* **A failed merge says what to delete:** when saved parts cannot be merged, the error
  names the chunk files that do not belong to the run in `result.npz` (or the first
  chunk); deleting them makes the earlier trajectories readable and resumable again.
* **`load_results` returns the exact spec:** the stored `spec` is no longer rounded to 6
  significant digits for display (1/3 became 0.333333), so passing it back to
  `start_ensemble_job` extends the run instead of being refused. The refusal of a
  different spec now names a differing key path and both values
  (`measurement_rate is 0.3333333333333333 in its spec.json and 0.333333 here`), chosen
  among the keys that set the protocol field that differs, and that field.
* **Resuming needs no `trajectories`:** without it, `start_ensemble_job` resumes to the
  target stored in `spec.json` (a single-graph run used to demand the count again). This
  holds for a graph collection too: a collection started with fewer than
  `G * circuits_per_graph` trajectories used to be run in full by a resume without a
  count, and is now resumed to its target; only an explicit larger count extends it. A
  new collection run without `trajectories` still runs, and stores, all of its circuits.
* **CLI graph collections from files:** `sparsegf2 ensemble` passes a missing
  `--trajectories` to the API, so a graph file holding several graphs runs all of its
  circuits, as the API does, and a resumed `--chunk-size` run uses its stored target.
  A single graph is refused with the API's message, which now names `--trajectories`
  and every case where the count may be omitted, still exit code 2; a non-object
  `graph` is now exit 2 instead of an internal error.
* **One writer per output directory:** `start_ensemble_job`, `run_ensemble` with an
  `output_dir`, and so the CLI, hold `output_dir/.lock` while they write, and release it
  when they end, also on an error, a cancel or Ctrl-C. The lock appears with its whole
  record (written to a file of the writer's own, then hard-linked, which fails if a lock
  exists; created with `O_CREAT | O_EXCL` where hard links are not supported): pid,
  hostname, start time, the system (`platform.system()` and, on Linux, the pid
  namespace) and, on Windows, the process's creation time. A second writer, in another
  process too, is refused before anything runs: `another process (pid P on host H,
  started T) is writing to DIR; wait for it to finish or cancel it` (`this process
  (...)` for a lock of the same process, and a job of the same process is named, as
  before, also when its lock was deleted by hand). Only jobs of the same process used to
  be refused, so the MCP server's job and a CLI resume of the same directory could both
  write, and with different chunk sizes both failed and left the directory unreadable.
  A lock of this machine (the same hostname and system: WSL and containers share the
  Windows or host name but not its pids) whose process is gone is taken over: checked
  with `os.kill(pid, 0)`, or on Windows with `OpenProcess` and `GetExitCodeProcess`,
  and on Windows also when its pid now names a process with another creation time (the
  kernel's, so clock steps do not matter). A lock without a record is taken over only
  when it is older than 10 s. Another machine's lock always holds.
* **Overlapping chunks merge:** trajectories saved twice (chunks of two writers with
  different chunk sizes, or chunks a merge could not delete) are kept once when a
  directory is read or merged, instead of failing the merge. A part of another run is
  still refused, even when all of its trajectories are covered. Chunks that `result.npz`
  covers entirely are deleted without writing `result.npz` again.
* **Merges survive readers (Windows):** replacing `result.npz` while a reader holds it (a
  notebook calling `EnsembleResult.load`, or `load_results`) is retried with a backoff
  of about 5 s in total, and so is deleting the merged chunks, with one backoff for all
  of them (a chunk still in use is left; it repeats `result.npz`). If `result.npz` is
  still in use, the job ends `completed` with the note `merge pending: result.npz is in
  use; the chunk files hold the data, and the next load_results or start_ensemble_job
  merges them`, instead of `failed`, and the next `load_results` (when no writer holds
  the directory) or `start_ensemble_job` merges them. An error writing the merged file
  itself still fails the job. `load_results` retries a file that a merge is replacing
  in the same way instead of failing with "could not read results". The CLI shows the
  job's notes (this one, and a resume's) under `job.notes`.
* **Temporary files are each writer's own:** `result.npz` and chunks are saved to a
  temporary file named after the writer's pid and a random token, so two writers never
  write into one file; loads never read temporaries, and a merge deletes those that
  killed writers left.

## 3.0.0.dev12: randomness contract 2, exact erasure counts, the exhaustive cutoff by budget

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
* **Cached kernels follow every kernel source:** Numba checks only each
  function's own file, so before this a cached caller could keep running an old
  callee from another file (for example `sample()` with an old `stream_state`
  after a change to `rng.py`). Now every cached kernel is stamped with a
  fingerprint of all kernel sources and the Numba and Python versions. A
  kernel compiled from other sources is recompiled rather than loaded, even one
  saved after an update by a process that had imported the old sources.
  Importing the package also deletes such stale cache files. The first import
  after an update, including this one, recompiles the kernels once (10 to 30 s).
  A process that was already running keeps the code it loaded, so restart it
  (for example the MCP server) to use the new code. If a Numba version lacks the
  private locator list this relies on, or `NUMBA_CACHE_LOCATOR_CLASSES` is set,
  stale files are only deleted. Then an old process can still save a stale
  kernel after an update; deleting the `sparsegf2-kernels.*.sha256` marker
  next to the cache files clears them on the next import.
* **The exhaustive cutoff by budget or per system size:**
  `uncorrectable_region_counts` takes `budget=B`, the number of regions to
  enumerate per evaluation, as an alternative to `max_size`; the cutoff is then
  m*(n) = max{w : sum_{m=1}^{w} C(n, m) <= B}. Either may be one int or a
  mapping from the system size n to a value (`max_size={64: 3, 128: 2}`,
  `budget={64: 20_000, 256: 50_000}`), which the spec resolves for its
  `num_system`, so one probe definition serves a sweep over n. The JSON forms
  are `{"budget": 50000}` and objects keyed by n (`{"64": 3, "128": 2}`); a
  missing n, a budget below n, both arguments, and non-integer keys or values
  are refused with a message that names the field (`probes[i]` in JSON); the
  entries of a mapping are checked against their own n when the probe is built.
  The key stays `uncorrectable_region_counts[w]` with the resolved w (give
  `name` to keep one key at every n), saved results store the resolved w and
  the input (loading refuses a file in which they disagree), and
  `estimate_ensemble` costs the resolved w. New `analysis.exhaustive_cutoff(n, B)`
  and `analysis.exhaustive_region_count(n, w)` (sum_{m=1}^{w} C(n, m), an exact
  int). The default is unchanged: `max_size=2`.

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
