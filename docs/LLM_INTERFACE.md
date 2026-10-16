# The LLM interface: CLI and MCP reference

SparseGF2 has three front ends over one implementation:

```
            +-------------------------+
 Python --> |  sparsegf2.api       |  JSON in, compact JSON out, ValueError on bad input
            +-----------+-------------+
                 ^             ^
   sparsegf2 CLI |             | sparsegf2-mcp (MCP server, stdio)
   (--json)      |             | tools + docs:// resources + prompts
```

The API functions take plain JSON values and return dicts of plain JSON values.
`MAX_OUTPUT_CHARS` (8000) caps the size of a result when it is encoded as JSON. Large
arrays are summarized:

* means and standard errors;
* at most 50 checkpoints per probe, evenly subsampled and always including the first and
  last (fewer when the result would otherwise exceed the size cap, chosen the same way);
* vector probes subsampled to 65 entries (33 when a run has more than two vector probes),
  plus a coarse grid of at most 8 checkpoints by 17 sizes;
* probe names longer than 64 characters abbreviated (regions written as `start:stop`
  ranges, e.g. `entropy[0:1024]`).

The full data can be written to files with `output_dir`. Whenever anything is left out,
the result carries `"truncated": true` and a note says what and how to get it, with one
exception: the default thinning of an ensemble's checkpoints to at most 50 is reported by
`checkpoints.subsampled` alone (showing fewer than 50 to fit the size cap does set
`truncated`). A result cut to fit keeps all of its notes.

For the quick version, read [`../AGENTS.md`](../AGENTS.md). To set up the MCP server step
by step (install, connect a client, a first session), follow
[`MCP_GETTING_STARTED.md`](MCP_GETTING_STARTED.md). This page is the complete reference.

---

## 1. Installing and registering

```sh
pip install -e ".[mcp]"            # sparsegf2 and sparsegf2-mcp on PATH
```

The core package needs only numpy and numba. The `mcp` extra (`mcp>=2.2`,
`pydantic>=2`) is imported only by the MCP server.

### Claude Code

```sh
claude mcp add sparsegf2 -- sparsegf2-mcp
```

The repository root also has a project-scoped [`.mcp.json`](../.mcp.json), so Claude Code
offers the server automatically when it is opened in this repository:

```json
{
  "mcpServers": {
    "sparsegf2": {"command": "sparsegf2-mcp", "args": []}
  }
}
```

A Claude Code skill in [`.claude/skills/sparsegf2/SKILL.md`](../.claude/skills/sparsegf2/SKILL.md)
tells Claude when and how to use the simulator.

### Claude Desktop

Add this to `claude_desktop_config.json` (Settings, Developer, Edit Config). Use the
absolute path that `which sparsegf2-mcp` prints, because Desktop does not inherit your
shell's PATH:

```json
{
  "mcpServers": {
    "sparsegf2": {
      "command": "/absolute/path/to/venv/bin/sparsegf2-mcp",
      "args": []
    }
  }
}
```

### Other MCP clients

Any client that can launch a stdio server works: the command is `sparsegf2-mcp`, or
equivalently `sparsegf2 mcp` or `python -m sparsegf2.mcp_server`.

---

## 2. Concepts shared by every front end

| Concept | Rule |
|---|---|
| circuits | Stim text format. Every Clifford gate, `M MX MY MR MRX MRY MXX MYY MZZ MPP R RX RY`, Pauli noise channels, `TICK DETECTOR OBSERVABLE_INCLUDE QUBIT_COORDS SHIFT_COORDS`, `REPEAT`, plus the SparseGF2 extensions `CLIFFORD_1Q(i)` and `CLIFFORD_2Q(i)` (Clifford by canonical index) |
| seeds | a non-negative int reproduces a run; `null` draws a fresh seed below 2^53, reported back as `seed` (53 bits, so a client that reads JSON numbers as doubles, JavaScript and Claude Code among them, keeps it exactly). An explicit seed of any size is used and reported exactly (as a JSON integer); such a client can send and read only seeds below 2^53 without rounding, so a result whose seed is 2^53 or more carries a note saying so |
| qubit lists | `[0, 1, 2]` or `"0:3"` / `"0:16:2"` (Python slices) |
| phases | `true` (default): signs tracked, outcomes match Stim. `false`: about 2x faster, no outcomes (only `random` flags), same entanglement |
| errors | invalid input raises `ValueError` (API), exits with code 2 (CLI), or returns a tool error (MCP). The message says what to change. Names that must be strings (a graph or weight `family`, a probe or observable `kind`, a `capabilities` section) are checked before they are looked up; an `output_dir` that cannot be created or written is `output_dir DIR is not writable: ...` (or `output_dir DIR cannot be created: P is a file, not a directory` when a parent is a file); a saved file that cannot be read is `could not read results from DIR: ...`, naming the file. Any other failure is a bug, and still carries its type and message (section 5) |
| provenance | every `output_dir` gets `provenance.json`: package, Python, numpy and numba versions, git commit, platform, seed, UTC timestamp, command. Resuming a job keeps that record and appends one entry per resume under `resumes`; a resume with another `sparsegf2` version is refused. A field a resume reads that holds another type (`resumes` not a list, `git_commit` not a string; a hand edit, say) is ignored with a note, as an unreadable `provenance.json` is |

---

## 3. Functions (API, MCP tools, CLI commands)

| API function / MCP tool | CLI | Purpose |
|---|---|---|
| `capabilities(section=None)` | `sparsegf2 capabilities [--section S]` | supported instructions, observables, probes, spec keys, graph families, limits, examples |
| `inspect_circuit(circuit)` | `sparsegf2 inspect FILE\|-` | parse and validate a circuit without running it |
| `run_circuit(circuit, shots, phases, seed, observables, observe_ticks, tick_stride, num_qubits, output_dir)` | `sparsegf2 run FILE\|- [...]` | run a circuit; measurement statistics and observables |
| `run_ensemble(spec, trajectories=None, seed, output_dir, max_seconds=60, num_threads)` | `sparsegf2 ensemble SPEC.json [...]` | ensembles of random monitored circuits |
| `estimate_ensemble(spec, trajectories=None, threads=None)` | (printed by `ensemble`) | pessimistic runtime estimate (per thread count with `threads`), trajectory count, result size, layout of the saved result |
| `benchmark_ensemble(spec, trajectories=None, threads=None, seconds=30, seed=0)` | `sparsegf2 benchmark SPEC.json [--trajectories N] [--threads T ...] [--seconds S] [--seed S]` | measured speed and storage on this machine, projected to other thread counts; writes nothing (section 3.6) |
| `start_ensemble_job(spec, output_dir, trajectories=None, seed, chunk_size, num_threads)` | `sparsegf2 ensemble ... --chunk-size K --output-dir D` | resumable background run |
| `job_status(job_id, wait_seconds)` | (shown as progress) | progress, throughput, ETA |
| `cancel_job(job_id)` | Ctrl-C | stop after the current chunk |
| `list_jobs()` | | jobs in this process |
| `load_results(path)` | `sparsegf2 results PATH` | summarize a saved run (file or directory, including partial chunks) |
| `sample_erasure_regions(graph, size, couplings, samples=200, num_nodes=None, burn_in=None, thin=1, tempering=True, chains=4, seed=None, method="auto", ladder="auto", max_step=None)` | `sparsegf2 regions --family F --n N ... --size M --couplings J... [...]` | read-only: geometry-correlated erasure regions on a graph alone (no circuit), with mixing diagnostics against exact values (section 3.7) |
| `erasure_curves(path, e=None, delta=0.01, layer=None)` | `sparsegf2 curves RUN [--eps E...] [--delta D] [--layer L]` | read-only: Phi(e) and e_delta of a saved run's erasure probes, independent and correlated (section 3.7) |

`trajectories` may be omitted for a collection of graphs (`graph.realizations`, or a
graph file holding several graphs), which then runs all of its circuits, and when
`start_ensemble_job` resumes a run, which then runs to the target stored in its
`spec.json`, a collection's included (section 3.5). The signatures are the MCP tools'. In
Python,
`api.start_ensemble_job(spec, trajectories=None, output_dir=None, ...)` keeps
`trajectories` second, and `output_dir` is still required (pass it by keyword).

### 3.1 `inspect_circuit`

```sh
$ printf 'H 0\nFOO 1\nCX 0 0\n' | sparsegf2 inspect - --json
{"valid": false, "errors": [{"line": 2, "message": "unknown gate 'FOO'; ..."},
                            {"line": 3, "message": "CX targets in a pair must differ, got (0, 0)"}]}
```

Besides syntax, it checks what only shows up when the circuit runs: record lookbacks
such as `rec[-2]` that point before the first measurement (reported with their line
number, checked in the first iteration of each REPEAT body) and circuits that unroll to
more than 50 million operations. REPEAT blocks are never unrolled, so this is instant even
for `REPEAT 1000000000 { ... }`. The CLI exits with code 2 for an invalid circuit, after
printing the errors.

For a valid circuit it returns `num_qubits`, `num_measurements`, `num_detectors`,
`num_observables`, `num_ticks`, `num_operations`, `repeat_blocks`,
`instruction_histogram` (applications per instruction with REPEAT blocks unrolled; the
64 most frequent), `kind_totals`, `extensions`, `stim_compatible` and the normalised
`text` (first 3000 characters).

### 3.2 `run_circuit`

| Argument | Default | Meaning |
|---|---|---|
| `circuit` | required | Stim text |
| `shots` | 1 | `1`: one run of `Simulator(n, phases, seed)`. `>1`: `sparsegf2.sample(circuit, shots, seed=seed, phases=phases)`, in parallel |
| `phases` | `true` | sign tracking |
| `seed` | `null` | see above |
| `observables` | none | list of `{"kind": ..., args..., "name"?: ...}`, evaluated on the final state of the single-shot run. Each value is keyed by its `name`, else by its kind; repeats get `#2`, `#3`, ..., skipping every key already taken and every name given explicitly, so no value overwrites another |
| `observe_ticks` | `false` | also evaluate the observables after every `TICK`, giving a time series (circuits with up to 1,000,000 ticks) |
| `tick_stride` | auto | with `observe_ticks`, evaluate after ticks `0, k, 2k, ...` and the last one; the default is every tick, or the smallest stride that keeps at most 10,000 evaluations, the last tick included |
| `num_qubits` | circuit's | simulate extra idle qubits |
| `output_dir` | none | save `circuit_run.npz` (records, random flags, detectors, observable flips), `circuit.stim`, `observables.json` (every evaluated tick, every profile entry, full precision) and `provenance.json` |

The result always has `num_qubits`, `num_measurements`, `num_detectors`,
`num_observables`, `num_ticks`, `shots`, `phases`, `seed`, `elapsed_seconds` and
`random_fraction`. The rest depends on `shots`:

* `shots=1`: `measurements`, `random` and `detectors` as bitstrings (up to 256 entries),
  plus `observable_flips`.
* `shots>1`: `measurement_mean`, `measurement_means` (up to 64 measurements),
  `first_shots`, `detector_rate_mean`, `detector_rate_max`, `any_detector_fraction`,
  `detector_rates` (up to 64 detectors), `observable_flip_rates`.

**Observables** (see `capabilities(section="observables")`):

| kind | args | value |
|---|---|---|
| `entropy` | `region` | S(A) in bits |
| `mutual_information` | `a`, `b` | I(A:B) |
| `tripartite_information` | `a`, `b`, `c` | I3(A:B:C) |
| `entropy_profile` | `order`? | S(order[:k]) for every k |
| `code_dimension` | `reference` | k = S(R) |
| `lost_logical_information` | `erased`, `reference` | I(R:E); 0 means correctable |
| `erasure_is_correctable` | `erased`, `reference` | bool |
| `recovery_probability` | `erased`, `reference` | 2^-I(R:E) |
| `random_erasure_statistics` | `system`, `reference`, `sizes`, `samples`?, `seed`? | per size: correctable fraction, mean lost bits, mean recovery probability |
| `erasure_profile` | `system`, `reference`, `samples`?, `seed`? | the same for every size 0..\|system\| |
| `contiguous_code_distance` | `system`, `reference`, `periodic`? | shortest uncorrectable contiguous erasure (0 if there is none) |
| `stabilizer_weights` | | mean, min and max generator weight |
| `stabilizer_length_distribution` | `order`? | clipped-gauge length histogram |
| `stabilizers` | | canonical generators (32 qubits or fewer); signed with `phases=true`, unsigned (`"XZ_"`) with `phases=false`, where signs are not tracked |
| `expectation` | `pauli` | <P> = +1, -1 or 0; needs phases |
| `is_deterministic` | `pauli` | bool |

Example: the entanglement of a growing Bell chain, tick by tick:

```sh
$ cat chain.stim
H 0
TICK
CX 0 1
TICK
CX 1 2
TICK
M 0
TICK
$ sparsegf2 run chain.stim --seed 1 --observe-ticks \
    --observables '[{"kind": "entropy", "region": [0]}]' --json
{..., "observables": {"entropy": 0},
 "tick_observables": {"ticks": [0, 1, 2, 3], "values": {"entropy": [0, 1, 1, 0]}, ...}}
```

### 3.3 `run_ensemble` and the spec

`spec` keys are the fields of `sparsegf2.mipt.MonitoredCircuit` (listed with their
defaults by `capabilities(section="ensemble_spec")`). JSON forms:

| Key | JSON form |
|---|---|
| `num_system`, `layers`, `measurement_rate` | required numbers |
| `graph` | `{"family": F, params...}`. `F` is one of `cycle`, `path`, `complete`, `lattice_2d` (`rows`, `cols`, `periodic`), `watts_strogatz` (`k`, `beta`, `seed`), `newman_watts` (`k`, `p`, `seed`), `erdos_renyi` (`p`, `seed`), `random_regular` (`d`, `seed`). Random families require their own `seed`. Add `"realizations": G` (at most 100,000) for graph disorder: a collection of `G` independent graphs, graph `i` from seed `[seed, i]`. `{"family": "file", "path": P}` reads a graph file (`.json` or `.npz`, as `graphs.save` writes it); a file with several graphs is a collection |
| `circuits_per_graph` | int, default 1: random circuits per graph of a collection. Trajectory `t` runs on graph `t // circuits_per_graph`, so the collection has `G * circuits_per_graph` trajectories; omit `trajectories` to run them all (asking for more is an error). Must be 1 for a single graph |
| `edges` | instead of `graph`: `[[a, b], ...]` |
| `edge_weights` | list (one per edge, in the order of `edges`; each weight stays with its edge when the edge list is normalised) or `{"family": "power_law", "alpha": 2.0, "periodic": true}`, for `schedule="random_edges"`. A list needs explicit `edges` or a deterministic family (whose edges come sorted, `a < b`); random families and `realizations` take a weight family |
| `schedule` | `matching` (default), `random_edges`, `all_edges`; `gates_per_layer` for `random_edges` |
| `reference`, `reference_partner` | `none`/`purification`/`single`; an int or `"random"` |
| `scramble_gates`, `scramble_matchings`, `scramble_layers` | ints: random all-to-all gates, layers on random perfect matchings, then gate-only layers on the graph |
| `gate_set` | list of two-qubit gate names (`["CX", "SWAP"]`) or CLIFFORD_2Q indices |
| `checkpoints` | `"final"`, `"every_layer"`, an int stride, or a list of layers: where purification is tested and where probes without their own `checkpoints` are evaluated |
| `probes` | `[{"kind": K, params...}]`; `K` is a `sparsegf2.mipt` probe function (`code_dimension`, `half_cut_entropy`, `entropy` with `region`, `mutual_information` with `a` and `b`, `tripartite_information` (optional `a`, `b`, `c`), `contiguous_distance`, `lost_information` with `region`, `entropy_profile`, `erasure_lost_profile` with `samples` (default 10), `erasure_first_failure` with `samples` (default 100), `uncorrectable_region_counts` with `max_size` w (default 2) or `budget` B, either one integer or an object per system size such as `{"64": 3, "128": 2}` (key `uncorrectable_region_counts[w]`; see below), `code_snapshot`, `correlated_erasure` with `couplings` (required), `sizes`, `samples` (default 100), `geometry` and the sampler's settings (section 3.7)). Every probe takes an optional `name`, its key in the results, and optional `checkpoints`, its own layers in the same forms as the spec's (default: the spec's); `capabilities(section="probes")` lists the other parameters |
| types | checked before running: booleans must be `true`/`false` (not `"false"`), counts must be integers (`4.0` is accepted, `2.5` is not), `measurement_rate` a number in [0, 1], `reference_partner` an int or `"random"`, `num_system` at most `MAX_QUBITS` (65,536, checked before any graph is generated). Errors name the key. The MCP tools check their own arguments the same way (section 5) |
| `stop_when_purified`, `system_order`, `periodic`, `random_orientation` | as in the library |

What each probe records per evaluation (n system qubits), and its value after
`stop_when_purified` has stopped a trajectory:

| kind | value per evaluation | after a stop |
|---|---|---|
| `entropy`, `half_cut_entropy`, `mutual_information`, `tripartite_information` | one number, in bits | NaN |
| `entropy_profile` | n + 1 numbers: S of the first m qubits of `system_order`, m = 0..n | NaN |
| `code_dimension` | k = S(R) | 0 |
| `contiguous_distance` | shortest contiguous run (in `system_order`, wrapping if `periodic`) whose erasure loses logical information; NaN where k = 0 | NaN |
| `lost_information` | I(A:R) for the fixed region A | 0 |
| `erasure_lost_profile` | n + 1 numbers: mean I(A:R) over `samples` random erasures A of each size m = 0..n | 0 |
| `erasure_first_failure` | `samples` numbers: for each random ordering of the system qubits, the smallest m such that erasing the first m loses logical information (n + 1 if k = 0) | n + 1 |
| `uncorrectable_region_counts` | w numbers (w = `max_size`, or the cutoff a `budget` allows): the number of uncorrectable m-qubit erasures for m = 1..w, by exhaustion | 0 |
| `code_snapshot` | n - k, the number of stabilizer generators; the generators are stored in the saved result | n (no code stored) |
| `correlated_erasure` | len(couplings) x len(sizes) numbers, coupling-major: for coupling J and size m, the fraction of `samples` m-qubit regions, drawn with probability proportional to exp(-J b(A)), whose erasure loses logical information (0 if k = 0) | 0 |

All probes except the entanglement ones and `code_snapshot` need a reference. Code
probes are exact after a stop, so their averages need no correction; entanglement probes
and `contiguous_distance` are NaN there and drop out of the means. At most one
`code_snapshot` per spec; give it its own `"checkpoints": "final"`, since codes are large.
The cost of `uncorrectable_region_counts` grows like C(n, w): measured one-thread costs
per evaluation at n = 256 are about 1 ms for w = 2, about 20 ms for w = 3 and about 4 s
for w = 4, against about 2 ms for `erasure_first_failure` with 100 samples
(`benchmarks/results/erasure_cost.md` has every n and w).

The cutoff w = m*(n) of `uncorrectable_region_counts` separates the exact regime of
P_C(m), m <= w, from the sampled one. It takes one of these forms (giving both `max_size`
and `budget` is an error):

| form | w at system size n |
|---|---|
| `{"max_size": 3}` | 3 |
| `{"budget": 50000}` | the largest w with sum_{m=1}^{w} C(n, m) <= B (`analysis.exhaustive_cutoff(n, B)`): 3, 2, 2 and 1 at n = 64, 128, 256 and 512 |
| `{"max_size": {"64": 3, "128": 2}}` | the entry for n |
| `{"budget": {"64": 20000, "256": 50000}}` | the cutoff of the entry for n |

The budget counts regions, not seconds (`analysis.exhaustive_region_count(n, w)` is the
number of regions a cutoff w covers): the time per region depends on n and w. An object
lets one probe serve a sweep over n. Its keys are system sizes written as integers;
a spec whose `num_system` has no entry is refused with the sizes the object covers, and
a budget below n (not even the single-qubit regions) is refused too. The key in the
results is `uncorrectable_region_counts[w]` for the resolved w, so give a `name` to keep
one key at every n. `estimate_ensemble` costs the resolved w, and `spec.json` keeps the
form as given.

The result contains:

* `num_trajectories`, `seed`, `elapsed_seconds` and `trajectories_per_second` (wall clock
  for the whole call, including building the spec and generating its graphs);
* `randomness_contract`: the `mipt.RANDOMNESS_CONTRACT` the run was made under (1 for
  results saved before it was recorded). Only a run made under this version's contract
  can be replayed, resumed or extended;
* `checkpoints`: the spec's layers shown, the total count, and whether they were
  subsampled;
* `probes`, each on its own layers:
  * scalar probes: `mean` and `stderr` per evaluation; `layers` when the probe's layers
    differ from the spec's checkpoints; and `count` when some values are NaN
    (entanglement probes of stopped trajectories, or `contiguous_distance` without a
    code: the distance is undefined there and left out of the mean). `code_snapshot`
    shows the mean number of generators, with a note that the codes are saved only with
    `output_dir`;
  * vector probes: `vector: true`; `what` (what the curve is); `at_layer` (the last
    evaluation that has data; with `stop_when_purified` later ones may be empty),
    `count` (trajectories there), `mean` and `stderr` per size m, and `index` (the sizes
    m) when subsampled; and `vs_depth`, the mean on a grid of at most 8 evaluations
    (`layers`, `count`) by 17 sizes (`index`), which shows how the curve changes with
    depth. A curve or grid that leaves sizes or evaluations out sets `truncated: true`,
    with a note (routine for the grid once n + 1 > 17);
  * the erasure probes are shown as P_C(m), the fraction of m-qubit erasures that are
    uncorrectable: `erasure_first_failure` as the fraction of samples with first-failure
    size at most m, for m = 0..n (`analysis.failure_by_size`), and
    `uncorrectable_region_counts` exactly, count / C(n, m), for m = 0..w with
    P_C(0) = 0 (`analysis.exact_failure_by_size`);
  * both erasure curves are averaged over the trajectories that hold a code (k > 0) at
    each evaluation, and `count_with_code` (in the entry and in `vs_depth`) says how many
    there are. `at_layer` is then the last evaluation with `count_with_code` > 0 (with
    `stop_when_purified` the last ones often have no code left); if no evaluation has a
    code, it is the last evaluation and the mean and `stderr` are `null`.
    `erasure_first_failure` tells the code holders from its samples (with k = 0 every
    sample is n + 1). `uncorrectable_region_counts` cannot (its counts are all 0 both for
    k = 0 and for a distance above w), so it takes them from a `code_dimension` probe
    evaluated at each of its layers (the same schedule or a denser one) or, if there is
    none, from such an `erasure_first_failure` probe (the first one of each kind, found by
    layer, not by name). With neither, the curve is averaged over every trajectory, the
    ones without a code counting as P_C = 0, and its `what` line and a note say so;
  * `erasure_lost_profile` and the scalar `lost_information` are not conditioned on a
    code: they average every trajectory, the ones without a code counting as 0 bits lost,
    and the profile's `what` line and a note on each say so;
* `purification` (with a reference):
  * counts: `trajectories`, `purified`, `censored`, `censored_fraction`, `horizon_layer`;
  * tau statistics: `tau_median_all` (censoring-aware: the first checkpoint layer at which
    at most half the trajectories are left unpurified, given only when strictly more
    than half purified within the horizon, else `null`), `tau_mean_purified`,
    `tau_median_purified`, `tau_min`, `tau_max`, `tau_quantiles_purified` (the 10th,
    25th, 75th and 90th percentiles of the purified trajectories' tau, interpolated
    linearly as `numpy.percentile` does);
  * `survival`: `layers` and `fraction_unpurified`;
* `final_layer_mean`, `measurements_per_trajectory`;
* `notes` when needed (on `stop_when_purified`, `code_snapshot`, `contiguous_distance`,
  erasure and lost-information probes that are not conditioned on a code, subsampled
  vector probes, abbreviated names); a cut to the size limit keeps every note and adds
  its own;
* `files` (with `output_dir`).

The scalar means are `sparsegf2.analysis.mean_and_error` of
`EnsembleResult.values[name]`, whose columns are the layers `EnsembleResult.layers(name)`.
The survival curve equals `EnsembleResult.survival()`.

```sh
$ cat spec.json
{"num_system": 64, "graph": {"family": "cycle"}, "layers": 512,
 "measurement_rate": 0.1, "reference": "single", "checkpoints": 16,
 "probes": [{"kind": "code_dimension"}], "stop_when_purified": true}
$ sparsegf2 ensemble spec.json --trajectories 2000 --seed 7 --output-dir runs/p010
```

`runs/p010/spec.json` stores `{"spec", "trajectories", "seed"}`, so
`sparsegf2 ensemble runs/p010/spec.json` repeats the run exactly. For a graph file it
also stores `graph_file`: the file's resolved absolute path and the sha256 of its bytes,
with which a resume checks that the file still holds the run's graphs. `load_results`
returns the stored `spec` exactly (its floats are not rounded for display), so it can be
passed back to `start_ensemble_job`. A spec longer than about 2000 characters (a long
explicit edge or weight list) is shown as `"too long to show; see the saved files"`;
take the exact spec to pass back from the `"spec"` entry of `spec.json` instead.

**Estimates and limits.** `estimate_ensemble` returns `estimated_seconds`, `trajectories`
(the count that would run), `result_megabytes` (every probe value as a float64, plus the
saved codes), `threads`, `fits_sync_limit` and `sync_limit_seconds`, and `notes` where
the estimate can be too low (see below). With `threads` (a list of positive integers) it
adds `by_threads`, `[{"threads", "estimated_seconds"}, ...]` in the order given; threads
beyond the trajectory count stay idle. It always adds `layout`: `arrays` (`name`, `shape`,
`dtype`, uncompressed `megabytes` of each array that `output_dir` will hold, as
`EnsembleResult.save` writes them: `value::<probe>`, `snapshot::` and `snapshot_counts::`
for a `code_snapshot`, the graph arrays and the run's own), `files` (`result.npz`,
`spec.json`, `provenance.json`) and a `note`. Disk use is lower, because the file is
compressed, and a `snapshot::` array is an upper bound (each code fills only its first
rows); `benchmark_ensemble` and `load_results` measure it. `run_ensemble`,
`estimate_ensemble` and `start_ensemble_job` refuse a run whose result would exceed
`MAX_RESULT_GB` (8 GB); the message suggests giving the large probes
(`erasure_first_failure` samples, `code_snapshot`, vector probes) fewer checkpoints of
their own, or splitting the trajectories over several output directories. They also
refuse a `num_system` above `MAX_QUBITS` (65,536) before any graph is generated, and
graphs that would take more than `MAX_GRAPH_GB` (2 GB), counted from the family's
parameters: the edges a collection keeps (16 bytes each) and the peak while one graph is
generated (48 bytes per pair for `complete` and `erdos_renyi`, which build every pair; up
to 512 bytes per edge for the ring families). A `graph.realizations` collection that is
too large is refused with how many realizations fit at that `num_system`, and one graph
that alone is too large with the largest `num_system` that fits and that the family
accepts (or, if there is none, with advice to choose a smaller degree).

The estimate is deliberately pessimistic. Its probe terms are about 3 times or more the
one-thread costs measured on an AMD Ryzen 9 9950X3D up to n = 2048, in both reference
pictures and in the states that make each probe expensive
([`benchmarks/results/probe_cost.md`](../benchmarks/results/probe_cost.md)), which leaves
room for a machine about twice as slow per thread. Trajectories on many threads at once are
priced as if each busy thread ran at 1 / (1 + 0.04 (T - 1)) of the speed of a thread alone,
0.45 at T = 32: with every logical processor busy (all-core clocks, SMT siblings, shared
memory bandwidth) each thread ran at 0.35 to 0.66 of that speed. The scans that stop at a
code's first failure (`contiguous_distance`, `erasure_first_failure`) are priced for the
worst case of the single-reference picture (k = 1, erasures that fail only at about n / 2
qubits) and for codes of monitored dynamics in the purification picture, so codes of small
distance run well within their estimate; a purification code of rate near 0 whose
erasures fail only near n / 2 would take up to the single-reference price. In the
purification picture, a spec with either scan therefore gets a note in
`estimate_ensemble`'s `notes` saying how many times its estimate such a code can make them
take: the single-reference price over the purification price of those probes at that n
(for example about 3.9 for `erasure_first_failure` and 27 for `contiguous_distance` at
n = 256).

### 3.4 Erasure studies

The erasure probes test the code the circuit has generated at each of their
checkpoints; for the code after the dynamics only, give them `"checkpoints": "final"`.
With a collection of graphs, one call runs every graph with `circuits_per_graph`
circuits:

```json
{"spec": {"num_system": 64,
          "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": 0,
                    "realizations": 50},
          "circuits_per_graph": 4, "layers": 256, "measurement_rate": 0.05,
          "schedule": "random_edges", "reference": "purification", "checkpoints": 32,
          "probes": [{"kind": "code_dimension"},
                     {"kind": "erasure_first_failure", "samples": 100},
                     {"kind": "uncorrectable_region_counts", "max_size": 2},
                     {"kind": "code_snapshot", "checkpoints": "final"}]},
 "seed": 1, "output_dir": "runs/erasure"}
```

The summary gives k per checkpoint and P_C(m) averaged over the trajectories that hold a
code (k > 0; the `code_dimension` probe lets the exact counts be conditioned too). The
failure probability when every qubit is erased independently with probability e,
Phi(e) = sum_m C(n,m) e^m (1-e)^(n-m) P_C(m), and the tolerance
e_delta = sup{e : Phi(e) <= delta} come from the saved arrays:

```python
import numpy as np
from sparsegf2 import StabilizerCode, analysis, mipt

res = mipt.EnsembleResult.load("runs/erasure/result.npz")
n = res.spec.num_system
k = res.values["code_dimension"][:, -1]                       # last checkpoint
first = res.values["erasure_first_failure"][:, -1]            # (trajectories, samples)
counts = res.values["uncorrectable_region_counts[2]"][:, -1]  # (trajectories, 2): m = 1, 2
e = np.array([0.05, 0.1, 0.2])
phi = analysis.erasure_failure_probability(first, n, e, exact_counts=counts).mean(axis=0)
code = k > 0                          # q+ = code.mean(), code rate = k.mean() / n
e10 = analysis.erasure_tolerance(first[code], n, 0.1, exact_counts=counts[code])
c = res.code(0, 0)                    # StabilizerCode of trajectory row 0, first snapshot
more = c.first_failure_sizes(10_000, seed=0)  # more orderings than the run used
StabilizerCode(["XXXX", "ZZZZ"]).uncorrectable_counts(2)      # [0, 6]: the [[4, 2, 2]] code
```

* `erasure_failure_probability(first_failure, n, e, *, exact_counts=None)` is the mean
  over samples of Pr[Bin(n, e) >= D]. With `exact_counts` (shape `(..., w)`, from the
  same checkpoint) the terms m <= w use the exact P_C(m) and the samples contribute
  Pr[Bin(n, e) >= max(D, w + 1)]. A code with k = 0 (D = n + 1) gives 0, so a mean over
  all trajectories counts purified circuits as protected.
* `erasure_tolerance(first_failure, n, delta, *, exact_counts=None)` pools all leading
  axes into one ensemble and bisects. Select the ensemble first, e.g. `k > 0` to
  condition on a code.
* `failure_by_size(first_failure, n)` gives the estimated P_C(m), m = 0..n, and
  `exact_failure_by_size(counts, n)` the exact P_C(m), m = 1..w.
* `StabilizerCode(stabilizers, num_qubits=None)` takes generators as Pauli strings
  (`["XXXX", "ZZZZ"]`) or an (s, 2n) 0/1 array (X bits, then Z bits); they must commute
  and need not be independent. It has `num_qubits`, `num_stabilizers` (the rank s),
  `num_logical` (k = n - s), `stabilizer_matrix()`, `lost_information(A)` (I(A:R)),
  `is_correctable(A)`, `first_failure_sizes(samples, seed=...)` and
  `uncorrectable_counts(max_size)`. `EnsembleResult.code(t, c)` returns the saved code
  of trajectory row `t` at evaluation `c` of the `code_snapshot` probe.

[`../examples/code_and_erasure.py`](../examples/code_and_erasure.py) runs this analysis
versus depth.

### 3.5 Background jobs

```python
from sparsegf2 import api
job = api.start_ensemble_job(spec, 100_000, output_dir="runs/big", seed=3)
api.job_status(job["job_id"], wait_seconds=30)   # status, trajectories_done, eta_seconds
api.load_results("runs/big")                      # works mid-run too (reads saved chunks)
```

* Chunks: `chunk_size` trajectories each (default about 10 estimated seconds, and at
  least 4 per thread). Each chunk is saved atomically to `output_dir/chunks/` when it
  finishes. On completion the chunks are merged into `output_dir/result.npz` and removed.
  On Windows a file another process holds open cannot be replaced or deleted, so the
  merge retries for about 5 s while a reader holds `result.npz` (and once more, for all
  chunks together, while one holds a chunk; a chunk still in use is left, a repeat that
  loads drop). If `result.npz` is still in use, the job ends `completed` with the note
  `merge pending: result.npz is in use; the chunk files hold the data, and the next
  load_results or start_ensemble_job merges them`; nothing is lost, and every read
  includes the chunks. Chunks that `result.npz` already covers are only deleted. Files
  are saved under a temporary name of the writer's own first; reads never see them.
* One writer per directory: a job, and `run_ensemble` with an `output_dir`, hold
  `output_dir/.lock` while they write and release it when they end (also after an error,
  a cancel or Ctrl-C). It appears with its whole record: pid, hostname, start time, the
  system (`platform.system()`, and the pid namespace on Linux) and, on Windows, the
  process's creation time. Another writer, from this process or another one (the MCP
  server and the CLI, say), is refused before anything runs: `another process (pid P on
  host H, started T) is writing to DIR; wait for it to finish or cancel it` (`this
  process (...)` when the lock is this process's own), or `job ID is already writing to
  DIR; poll it or cancel it` for a job of the same process. A lock of this machine whose
  process has ended (a process that was killed, say; on Windows also when its pid now
  names a process with another creation time) is taken over by the next writer. A lock
  of another machine always holds: another hostname, or the same hostname under another
  system (WSL and containers report the Windows or host name, but their pids are
  others), so a Windows writer and a WSL writer of one directory exclude each other.
  The refusal then names the lock's system after the host (`pid P on host H (Linux
  pidns:N)`). A WSL2 distribution, like a container, has a pid namespace of its own,
  which may be a new one after WSL restarts: a lock that a crashed WSL writer left then
  counts as another machine's, and must be deleted by hand. A lock that cannot be
  created (a read-only directory or drive) is `output_dir DIR is not writable: ...`.
  Readers (`load_results`, `EnsembleResult.load`) take no lock, except that
  `load_results` takes it to finish a pending merge (below), which for a large run can
  take minutes.
* Equivalence: trajectory `t` depends only on `(spec, seed, t)`, so a job's result is
  identical to a single `run_ensemble` call. The tests check this.
* Resume and extend: call `start_ensemble_job` again with the same spec and `output_dir`.
  `seed` may be omitted (the stored one is used), and so may `trajectories`: the run then
  resumes to the target stored in `spec.json`, for a graph collection too (a new
  collection run without `trajectories` stores all of its circuits; one started with
  fewer is not grown by a resume). Only a larger `trajectories` extends a run.
  `load_results` returns the stored spec exactly, ready to pass back.
* What is refused, before anything runs, in this order: a run made under another
  randomness contract or SparseGF2 version (read from `provenance.json` and from the saved
  results, so a missing `provenance.json` does not hide it); a graph file whose contents
  changed; a spec that builds another protocol than the saved run (the message names a
  differing key path with both values, such as
  `measurement_rate is 0.3333333333333333 in its spec.json and 0.333333 here`, chosen
  among the keys that set the protocol field that differs, and that field); another
  `seed`, or a `spec.json` seed that is not the one the saved results were made with.
  The protocol, not the spec's text, identifies the run (it is what a merge of saved
  parts requires), so a spec written differently that builds the same protocol
  continues the run, with a note, and `spec.json` then stores the new form: a cutoff
  given as `budget` or as a per-size object that resolves to the same w, a range string
  for a qubit list, a default written out, another path to the same graphs.
* Graph files: the sha256 recorded in `spec.json` must still match the file's bytes. If
  the bytes changed (a regenerated `.npz` differs in its zip timestamps even with the
  same graphs), the run continues only when its saved results show the same graphs, and
  `spec.json` records the new sha256; otherwise, or with no results saved yet, it is
  refused. A `spec.json` written before the hash was recorded is checked against the
  saved results alone.
* A merge that still fails (a directory mixed by an earlier version) names the chunk
  files that do not belong to the run in `result.npz` (or the first chunk); deleting them
  makes the earlier trajectories readable and resumable again.
* Graph collections: in a new run `trajectories` may be omitted (all
  `G * circuits_per_graph` run) and cannot exceed that count. To finish a collection
  started with fewer, pass that count (`estimate_ensemble(spec)["trajectories"]`). Changing `circuits_per_graph` changes the spec, so more
  circuits per graph need a new directory.
* Concurrency: compiled kernels run one at a time (a process-wide lock), so jobs and
  interactive calls interleave chunk by chunk. Every function is safe to call from any
  thread.
* Lifetime: jobs live in the process that started them. After a restart, resume from the
  directory.
* Crash safety: chunks and `result.npz` are written atomically, and trajectories saved
  twice (chunks already merged into `result.npz`, or overlapping chunks of two writers
  with different chunk sizes) are kept once, so a directory stays readable and resumable
  whenever the process dies. `load_results` merges the chunks of a finished run whose
  merge is pending, unless a writer holds the directory.

The CLI runs the same machinery in the foreground:
`sparsegf2 ensemble spec.json --trajectories 100000 --output-dir runs/big --chunk-size 5000`
prints progress to stderr. Ctrl-C stops it after the current chunk, prints the partial
summary and exits with code 130; rerunning the same command resumes it (also after the
process was killed: the rerun takes over the lock the dead process left).

### 3.6 Cost of a run: `benchmark_ensemble` and the storage of a saved run

`estimate_ensemble` is a model. `benchmark_ensemble(spec, trajectories=None, threads=None,
seconds=30, seed=0)` measures. It runs one trajectory on one thread (this compiles the
kernels: `warmup_seconds`), then the first trajectories of the run (with `seed`) on one
thread and on `P = min(max(threads), threads_available, trajectories)` threads, doubling
the count until a run takes `seconds / 4`, so a call takes up to about twice `seconds`.
`threads_available` is the most threads Numba can use in the process, `cpus` the
processors it may run on; `threads` defaults to `[1, threads_available]`. The result has

* `measured`: `[{"threads", "trajectories", "seconds", "trajectories_per_second"}]`;
* `projected`: `[{"threads", "seconds", "basis"}]` for each count in `threads`, the
  seconds of the whole run (the trajectories; building the spec and graphs is extra).
  `basis` is `"measured"` for one thread and `P`, and `"projected"` otherwise: the
  simulator's thread model, calibrated so that it reproduces the `P`-thread rate. A note
  says so when a count exceeds `threads_available`;
* `estimate_ratio`: measured seconds per trajectory at `P` threads over
  `estimate_ensemble`'s, to scale a cheap estimate;
* `storage`: `sample_trajectories`, `disk_megabytes` (the size of `result.npz` for all
  trajectories, measured on saved samples and scaled by what grows with the trajectories)
  and `memory_megabytes`;
* `estimated_seconds` (the plain estimate), `trajectories`, `seed`, `notes`.

It writes nothing: the sample is saved in a temporary folder that is removed. The MCP tool
is read-only and refuses `seconds` above 30. The CLI prints a table without `--json`.

`load_results` adds `storage` for a saved run: `total_megabytes` (every file in the
folder), `files` (the 8 largest: `path` relative to the folder, `megabytes`;
`files_not_shown` counts the rest) and `arrays` (the members of the `.npz` files: `name`,
abbreviated like the summary's probe names, compressed `megabytes`, and `file` for a
single file). With several chunk files the arrays are summed by name
(`arrays_summed_over_files`); at most 20 are listed (the rest as `(other arrays)`).

`benchmark_ensemble` refuses a spec whose least possible measurement (a warm-up and one
trajectory on one thread, and the same on `P` threads) is estimated above 2 x `seconds`,
without running anything; its `notes` say when `disk_megabytes` is rough (a small sample,
or a sample file no larger than the one-trajectory file) and that a background job writes
chunks before merging, so its peak disk use is about twice the final result.

### 3.7 Correlated erasures: the `correlated_erasure` probe and two read-only tools

An m-qubit erasure that clusters on the erasure graph G strikes a region A with
probability exp(-J b(A)) / Z_m(J), b(A) the number of edges of G leaving A: J = 0 is the
uniform case of section 3.4, J > 0 favours compact regions, J < 0 spread-out ones. The
probe estimates P_{C,G,J}(m), the probability that such an erasure fails, for the code
each trajectory holds. The mathematics, the samplers with their proofs, costs and
worked examples are in
[`observables/correlated-erasure.md`](observables/correlated-erasure.md).

```json
{"kind": "correlated_erasure", "couplings": [0, 1, 2], "sizes": "8:57:8", "samples": 100,
 "burn_in": null, "thin": 1, "tempering": true, "method": "auto", "ladder": "auto",
 "max_step": null, "geometry": "interaction", "checkpoints": "final"}
```

* `couplings`: one number or a list of distinct ones, each with |J| <= 10^6 (J -> inf is a
  limit, not a coupling, and is refused with the exact routes to use). `sizes`: a list, a range string `"a:b[:step]"` (Python slice
  semantics) or null for every m = 1..n-1. `geometry`: `"interaction"` (each trajectory's
  own graph, default), `"cycle"` or `"path"` over the qubit labels.
* The sampler is `erasures.sample_regions`: with `method="auto"`, exact independent
  samples when the erasure graph is one cycle or one path through all qubits (any
  labeling), otherwise a Metropolis swap chain per rung of a replica-exchange ladder
  (`ladder="auto"`: the couplings, 0 and rungs at most `max_step` = 1/sqrt(Var_0 b) apart;
  `"given"`; or extra couplings), started from uniform regions, `burn_in` sweeps (null:
  max(200, 8 n)) then `samples` records `thin` sweeps apart. `"chain"` forces the chain,
  `"exact"` refuses other graphs.
* Values: `len(couplings) * len(sizes)` per evaluation, coupling-major;
  `EnsembleResult.correlated(name)` returns `(couplings, sizes, values)` with `values`
  shaped `(trajectories, evaluations, c, s)`. A code with k = 0 gives 0 and draws nothing.
* Summary entry: `what`, `at_layer`, `count`, `count_with_code` (when a `code_dimension`
  or `erasure_first_failure` probe covers the layers; without one the mean is over every
  trajectory, k = 0 counting as 0, with a note), `couplings` and `sizes` (at most 12
  couplings and 17 sizes, evenly spaced with the ends kept, the couplings in increasing
  order of J with the smallest and the largest included when some are left out, fewer with a
  note when the summary must shrink), `mean` and `stderr` (`mean[i][j]` is `couplings[i]` at
  `sizes[j]`), `sampler` (one line) and `vs_depth` (`layers`, `count_with_code`, `sizes`,
  and `mean[i][d][j]` at `couplings[i]`, `layers[d]`, `sizes[j]`). `estimate_ensemble`
  adds `working_megabytes` (the regions the running threads hold at once) and a note.
  For a chain a note says to check its mixing
  with `sample_erasure_regions`: the probe stores only failure fractions, so the summary
  cannot warn about a chain that has not mixed.
* Randomness: each size of each evaluation has its own stream, so a size's values do not
  depend on the other sizes, and `EnsembleResult.replay` reproduces them. A saved run
  pins the sizes, burn-in and replica-exchange ladders it resolved; a resume under rules
  that now resolve otherwise is refused as another protocol.
* Cost: every size of every evaluation runs a fresh sampler with a full burn-in, about
  rungs x (burn_in + samples x thin) x n swap proposals for the chain (one size at
  n = 256 about 0.2 s on one thread, every size about 52 s); exact samples cost little.
  `estimate_ensemble` prices it. Keep `sizes` to `analysis.binomial_window(n, e, tol)` of
  the rates you need and give the probe its own few `checkpoints`.

`sample_erasure_regions` runs the probe's sampler on a graph alone, so its settings can
be checked before a long run (the chain does not depend on the code). `graph` is
`{"family": ..., params}` with `num_nodes` (beside it or inside it),
`{"family": "file", "path": ...}` (the first graph of a file) or
`{"edges": [[a, b], ...], "num_nodes": n}`. It runs `chains` independent chains (chain i
seeded `[seed, i]`) for at most 12 couplings and returns `graph` (with its `geometry`),
`size`, `samples`, `chains`, `method`, for chains `burn_in`, `thin`, `tempering` and
`ladder` (`rungs`, `lowest`, `highest`, `max_step`), `seed`, and

* `couplings`, one row per coupling in the order given: `coupling`, `mean_boundary`,
  `stderr`, `variance`, `acceptance`, `exchange_acceptance`, `tau_int`, `ess`,
  `ess_boundary`, `slowest`, `r_hat`, `r_hat_limit`, `r_hat_at`, `min_boundary_seen`,
  `exact` and `deviation_se`. `stderr = sqrt(variance / ess_boundary)`, the effective
  sample size of the boundary itself; `tau_int` (largest) and `ess` (smallest) are over the
  boundary and the membership of 8 spread nodes, `slowest` says which. `ess` and
  `ess_boundary` are capped at `chains x samples` (the estimated `tau_int` of a short
  series can fall below 1 from noise; `tau_int` is left as estimated). `r_hat` is the
  R-hat of the series (the boundary or a node, `r_hat_at`) furthest above its own
  `r_hat_limit`: the 99% point of the F distribution of the split R-hat for stationary
  draws of that series' autocorrelation, shared over the `2 x 9 x couplings` series
  tested, never below 1.01. On a cycle or a path the samples are exact and independent:
  `tau_int` is 1, `ess` and `ess_boundary` are `chains x samples`, and `r_hat`,
  `r_hat_limit`, `r_hat_at`, `slowest` and the acceptances are null;
* `exact_source` (the ring or path closed form, the enumeration of every region up to
  10^8 of them, or the uniform law at J = 0), `nonincreasing_in_J` and `proposition2`:
  each step to the next coupling against `z_critical`, the normal quantile that shares a
  1% false-alarm chance over the `steps` (3.1 for 12 couplings), with the `worst` step and
  the `failed` ones as `{"J": [from, to], "increase", "z"}`; `False` means the estimates
  are off, not the proposition (measured false alarms with a correct sampler: 1.5% of 400
  calls with 12 couplings 0.001 apart);
* `mixing`, one note per kind with what to change: R-hat above its limit (the chains
  differ more than one stationary distribution allows: not in equilibrium, in different
  regions, or too short for their autocorrelation), an effective sample size below 100 per
  chain (400 for the default four chains),
  exchange acceptance below 5%, a mean more than 5 standard errors from its exact value, a
  failed Proposition-2 step. On correct swap chains the R-hat note fired in about 2% of
  calls at 200 samples (a ring of 32 nodes, a 14-node Watts-Strogatz graph) and in none at
  800; a record too short for its autocorrelation fires it, and thinning clears it.
  An `r_hat` of 1000 stands for infinity (chains stuck apart);
* `warnings` (the sampler's own), `examples` (three regions at the largest coupling),
  `estimated_seconds`, `elapsed_seconds` and `notes`.

The call is refused when its estimate exceeds `max_seconds` (60; the MCP tool has no such
argument). An empty `mixing` list is no proof of mixing: repeat with another seed and a
longer burn-in.

`erasure_curves` reads a saved run (an output directory or `result.npz`) and returns, at
`layer` (default each probe's last evaluation), over the trajectories that hold a code:
`uniform` (per `erasure_first_failure` probe, the independent channel, exact for
m <= w where `uncorrectable_region_counts` was evaluated at the same layer) and
`correlated` (per `correlated_erasure` probe, one row per coupling), each with `count`,
`count_with_code`, `q_plus`, `phi` (`mean`, `stderr` and the number of `codes` used, per
`e`; default eight values log-spaced from 1/n to 1/2) and `e_delta`: `of_mean_phi`, the
tolerance of the ensemble-mean Phi and the headline, with `of_mean_phi_range` (its value
when the mean curve moves by plus and minus its standard error at every size), and
`per_code_mean`, `per_code_median`, `per_code_stderr` of each code's own tolerance (they
describe the spread across codes and are biased upward for noisy curves);
`of_mean_phi_monotone` (false where the mean curve steps down in m: `of_mean_phi` is then
the last crossing of delta); `monotone_fraction` for the correlated rows. A correlated entry
adds `couplings` (at most 12 shown), `sampled_sizes`, `interpolated_sizes`,
`interpolated_mass` (the binomial probability the interpolated sizes carry at each e),
`complete` (that mass below 0.1% at the e asked for), `e_delta_complete` (false, with a
note, when the interpolated sizes carry more than 0.1% at `e_delta` itself, which the
tolerance searches over every e: `e_delta.interpolated_mass_at_e_delta`, per coupling, the
mass at `of_mean_phi` and at both ends of `of_mean_phi_range`; the `uniform` rows have none
of these fields), `binomial_window` and `sampler`. Phi erases each qubit independently only
for the `uniform` rows and for J = 0; a row with J != 0 erases Bin(n, e) qubits as one
region drawn from Q_J. Without a code probe on the layer every trajectory counts
and one that stored only zeros loses nothing (P(n) = 0). The standard errors treat
trajectories as independent; a note warns when they ran on a few graphs (circuits on one
graph are correlated). The work is bounded: the curves are at most 10^8 values (codes x
couplings x (n + 1)) and the per-code tolerances at most 10^10 multiply-adds and 4,000
codes; beyond that evenly spaced codes are used and a note says so (`phi.codes`,
`e_delta.codes`), and `elapsed_seconds` reports the time. Both tools write nothing.

---

## 4. CLI details

* `--json` on every command prints exactly one JSON object on stdout. Logs and progress go
  to stderr. On failure the object is `{"error": {"type", "message"}}`.
* Exit codes: `0` success, `2` invalid input or usage (bad circuit text, including
  `inspect` of an invalid circuit, unknown spec key, missing file, a file that cannot be
  read or is not UTF-8, bad JSON, an output directory that cannot be written:
  `ValidationError`), `1` internal error (`internal error: ...`, any other exception, a
  `TypeError` included: a bug to report), `130` interrupted (Ctrl-C).
* `-` reads the circuit or spec from stdin. `--observables` takes inline JSON or
  `@file.json`.
* `sparsegf2 ensemble` needs `--trajectories` (or a `trajectories` entry in SPEC.json),
  except for a graph collection (`graph.realizations`, or a graph file holding several
  graphs), which runs all of its circuits by default, and for a `--chunk-size` run
  resumed in its `--output-dir`, which runs to the stored target. Without a count the
  spec goes to the API as it is, which refuses a single graph with its own message.
  Without `--chunk-size` it logs the trajectory count and the runtime estimate to stderr
  before running.
* `sparsegf2 benchmark SPEC.json --trajectories N --threads 1 32 [--seconds S] [--seed S]`
  reads the spec as `ensemble` does and calls `benchmark_ensemble` (section 3.6); without
  `--json` it prints a short table.
* `sparsegf2 regions` calls `sample_erasure_regions` (section 3.7). The graph is
  `--graph FILE` (its first graph) or `--family F` with `--n`, `--k`, `--beta`,
  `--graph-seed` and `--graph-param KEY=VALUE` (a JSON value, repeatable); then `--size`,
  `--couplings J [J ...]` (negative values work), `--samples`, `--chains`, `--burn-in`,
  `--thin`, `--tempering/--no-tempering`, `--method`, `--ladder`, `--max-step`, `--seed`
  and `--max-seconds`. For example
  `sparsegf2 regions --family watts_strogatz --n 64 --k 4 --beta 0.1 --graph-seed 1 --size 16 --couplings 0 0.5 1 2 --samples 400 --chains 8 --seed 7`.
  Without `--json` it prints one row per coupling: the mean boundary against its exact
  value, the diagnostics and the mixing notes.
* `sparsegf2 curves RUN [--eps E [E ...]] [--delta D] [--layer L]` calls
  `erasure_curves`; without `--json` it prints Phi(e) and e_delta per channel and
  coupling.
* Without `--json`, the same data is printed as indented `key: value` lines.

---

## 5. MCP details

* Tools: `capabilities`, `inspect_circuit`, `run_circuit`, `run_ensemble`,
  `estimate_ensemble`, `benchmark_ensemble`, `start_ensemble_job`, `job_status`,
  `cancel_job`, `list_jobs`, `load_results`, `sample_erasure_regions` and
  `erasure_curves` (the last two read-only, marked so for the client; section 3.7;
  `sample_erasure_regions` has no `max_seconds` argument and keeps the 60-second
  limit). Each description includes a worked example of its arguments; the spec
  keys, graph families and probe kinds in the `run_ensemble` description are generated
  from the library, so they never go stale. Tools return structured content (the API
  dict) plus the same JSON as compact text, so the text stays within the 8000-character
  budget.
* Ensemble tool signatures: `run_ensemble(spec, trajectories=None, seed=None,
  output_dir=None, max_seconds=60, num_threads=N)`, `estimate_ensemble(spec,
  trajectories=None, threads=None)`, `benchmark_ensemble(spec, trajectories=None,
  threads=None, seconds=30, seed=0)` (read-only; `seconds` at most 30 here, since a call
  takes up to twice that) and `start_ensemble_job(spec, output_dir, trajectories=None,
  seed=None, chunk_size=None, num_threads=N)`.
* `num_threads` (`run_ensemble`, `start_ensemble_job`): the threads the run uses. Its
  default N is the number of threads the server process may use,
  `numba.config.NUMBA_NUM_THREADS`, read when the server is built, so the tool's input
  schema and description show the real number: 16 in Clank's sandbox (its containers set
  `NUMBA_NUM_THREADS=16` to match their 16-CPU limit), every logical processor elsewhere.
  A value above N runs on N (`sparsegf2._parallel.threads` clamps it, as the API does);
  0 or less is refused (`num_threads must be at least 1, got 0`). Results are identical at
  any thread count; only the time changes. Use it to leave cores free on a shared machine,
  or as `estimate_ensemble`'s note advises ("run on at most K threads (num_threads)") when
  `correlated_erasure` regions would not fit in memory on every thread.
* `run_ensemble` refuses runs estimated above `max_seconds` (default 60, must be positive)
  and points to `start_ensemble_job`. The estimate is for the threads the run will use
  (`num_threads`, at most N), counts graph generation and preprocessing (large
  `realizations` collections), and the spec is built once for the estimate and the run. A
  background job's default `chunk_size` (about 10 estimated seconds) is sized for its
  `num_threads` too.
* Errors: invalid input becomes an MCP tool error (`isError: true`) whose text is the
  API's message, so the model can correct the call. Every other failure is a tool error
  too, never the SDK's bare `Error executing tool X`: the model reads
  `Error executing tool X: TypeName: message (internal error: please report it)`, and the
  server logs the traceback on its stderr. The API raises `ValueError` for every input
  error (it turns the library's field-naming `TypeError`s into one), so anything else,
  a `TypeError` included, is a bug: report it rather than retry with other arguments.
* Argument types are strict, as the API's own checks are, and are reported by the SDK's
  schema validation in the same way, naming the argument: an integer (`trajectories`,
  `seed`, `shots`, `tick_stride`, `num_qubits`, `chunk_size`, `num_threads`) must be a JSON integer or a
  whole-number float (`4.0` is 4; `2.5`, `"3"` and `true` are refused), a flag
  (`phases`, `observe_ticks`) `true` or `false` (not `"false"` or `0`), and
  `max_seconds` and `wait_seconds` a number (not `"inf"` or `true`). Pydantic's default
  lax mode used to turn `true` into 1 and `"3"` into 3. The input schema is unchanged.
* Resources: `docs://sparsegf2/quickstart` (Markdown guide) and
  `docs://sparsegf2/capabilities` (JSON).
* Prompts: `study_purification_transition(num_system, graph_family, rates, trajectories)`
  and `study_erasure_protection(num_system, measurement_rate, trajectories)`. The erasure
  prompt asks for `code_dimension`, `erasure_first_failure` and
  `uncorrectable_region_counts` and a report of P_C(m), and points to
  `analysis.erasure_failure_probability` and `erasure_tolerance` for Phi(e) and e_delta.
* Implementation: the official MCP Python SDK v2 (`mcp.server.mcpserver.MCPServer`), over
  stdio. Tools are synchronous functions, which the SDK runs in worker threads, so a long
  call does not block the protocol.
