# The LLM interface: CLI and MCP reference

SparseGF2 v3 has three front ends over one implementation:

```
            +-------------------------+
 Python --> |  sparsegf2_v3.api       |  JSON in, compact JSON out, ValueError on bad input
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
the result carries `"truncated": true` and a note says what and how to get it.

For the quick version, read [`../AGENTS.md`](../AGENTS.md). To set up the MCP server step
by step (install, connect a client, a first session), follow
[`MCP_GETTING_STARTED.md`](MCP_GETTING_STARTED.md). This page is the complete reference.

---

## 1. Installing and registering

```sh
pip install -e "./v3[mcp]"            # sparsegf2 and sparsegf2-mcp on PATH
```

The core package needs only numpy and numba. The `mcp` extra (`mcp>=2.2`,
`pydantic>=2`) is imported only by the MCP server.

### Claude Code

```sh
claude mcp add sparsegf2 -- sparsegf2-mcp
```

The repository root also has a project-scoped [`.mcp.json`](../../.mcp.json), so Claude Code
offers the server automatically when it is opened in this repository:

```json
{
  "mcpServers": {
    "sparsegf2": {"command": "sparsegf2-mcp", "args": []}
  }
}
```

A Claude Code skill in [`.claude/skills/sparsegf2/SKILL.md`](../../.claude/skills/sparsegf2/SKILL.md)
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
equivalently `sparsegf2 mcp` or `python -m sparsegf2_v3.mcp_server`.

---

## 2. Concepts shared by every front end

| Concept | Rule |
|---|---|
| circuits | Stim text format. Every Clifford gate, `M MX MY MR MRX MRY MXX MYY MZZ MPP R RX RY`, Pauli noise channels, `TICK DETECTOR OBSERVABLE_INCLUDE QUBIT_COORDS SHIFT_COORDS`, `REPEAT`, plus the SparseGF2 extensions `CLIFFORD_1Q(i)` and `CLIFFORD_2Q(i)` (Clifford by canonical index) |
| seeds | a non-negative int reproduces a run; `null` draws a fresh 63-bit seed, reported back as `seed` |
| qubit lists | `[0, 1, 2]` or `"0:3"` / `"0:16:2"` (Python slices) |
| phases | `true` (default): signs tracked, outcomes match Stim. `false`: about 2x faster, no outcomes (only `random` flags), same entanglement |
| errors | invalid input raises `ValueError` (API), exits with code 2 (CLI), or returns a tool error (MCP). The message says what to change |
| provenance | every `output_dir` gets `provenance.json`: package, Python, numpy and numba versions, git commit, platform, seed, UTC timestamp, command. Resuming a job keeps that record and appends one entry per resume under `resumes`; a resume with another `sparsegf2_v3` version is refused |

---

## 3. Functions (API, MCP tools, CLI commands)

| API function / MCP tool | CLI | Purpose |
|---|---|---|
| `capabilities(section=None)` | `sparsegf2 capabilities [--section S]` | supported instructions, observables, probes, spec keys, graph families, limits, examples |
| `inspect_circuit(circuit)` | `sparsegf2 inspect FILE\|-` | parse and validate a circuit without running it |
| `run_circuit(circuit, shots, phases, seed, observables, observe_ticks, tick_stride, num_qubits, output_dir)` | `sparsegf2 run FILE\|- [...]` | run a circuit; measurement statistics and observables |
| `run_ensemble(spec, trajectories=None, seed, output_dir, max_seconds=60)` | `sparsegf2 ensemble SPEC.json [...]` | ensembles of random monitored circuits |
| `estimate_ensemble(spec, trajectories=None)` | (printed by `ensemble`) | pessimistic runtime estimate, trajectory count, result size |
| `start_ensemble_job(spec, output_dir, trajectories=None, seed, chunk_size)` | `sparsegf2 ensemble ... --chunk-size K --output-dir D` | resumable background run |
| `job_status(job_id, wait_seconds)` | (shown as progress) | progress, throughput, ETA |
| `cancel_job(job_id)` | Ctrl-C | stop after the current chunk |
| `list_jobs()` | | jobs in this process |
| `load_results(path)` | `sparsegf2 results PATH` | summarize a saved run (file or directory, including partial chunks) |

`trajectories` may be omitted only for a collection of graphs (`graph.realizations`),
which then runs all of its circuits. The signatures are the MCP tools'. In Python,
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
| `shots` | 1 | `1`: one run of `Simulator(n, phases, seed)`. `>1`: `sparsegf2_v3.sample(circuit, shots, seed=seed, phases=phases)`, in parallel |
| `phases` | `true` | sign tracking |
| `seed` | `null` | see above |
| `observables` | none | list of `{"kind": ..., args..., "name"?: ...}`, evaluated on the final state of the single-shot run |
| `observe_ticks` | `false` | also evaluate the observables after every `TICK`, giving a time series (circuits with up to 1,000,000 ticks) |
| `tick_stride` | auto | with `observe_ticks`, evaluate after ticks `0, k, 2k, ...` and the last one; the default is every tick, or the smallest stride that keeps at most 10,000 evaluations |
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

`spec` keys are the fields of `sparsegf2_v3.mipt.MonitoredCircuit` (listed with their
defaults by `capabilities(section="ensemble_spec")`). JSON forms:

| Key | JSON form |
|---|---|
| `num_system`, `layers`, `measurement_rate` | required numbers |
| `graph` | `{"family": F, params...}`. `F` is one of `cycle`, `path`, `complete`, `lattice_2d` (`rows`, `cols`, `periodic`), `watts_strogatz` (`k`, `beta`, `seed`), `newman_watts` (`k`, `p`, `seed`), `erdos_renyi` (`p`, `seed`), `random_regular` (`d`, `seed`). Random families require their own `seed`. Add `"realizations": G` (at most 100,000) for graph disorder: a collection of `G` independent graphs, graph `i` from seed `[seed, i]` |
| `circuits_per_graph` | int, default 1: random circuits per graph of a collection. Trajectory `t` runs on graph `t // circuits_per_graph`, so the collection has `G * circuits_per_graph` trajectories; omit `trajectories` to run them all (asking for more is an error). Must be 1 for a single graph |
| `edges` | instead of `graph`: `[[a, b], ...]` |
| `edge_weights` | list (one per edge, in the order of `edges`; each weight stays with its edge when the edge list is normalised) or `{"family": "power_law", "alpha": 2.0, "periodic": true}`, for `schedule="random_edges"`. A list needs explicit `edges` or a deterministic family (whose edges come sorted, `a < b`); random families and `realizations` take a weight family |
| `schedule` | `matching` (default), `random_edges`, `all_edges`; `gates_per_layer` for `random_edges` |
| `reference`, `reference_partner` | `none`/`purification`/`single`; an int or `"random"` |
| `scramble_gates`, `scramble_matchings`, `scramble_layers` | ints: random all-to-all gates, layers on random perfect matchings, then gate-only layers on the graph |
| `gate_set` | list of two-qubit gate names (`["CX", "SWAP"]`) or CLIFFORD_2Q indices |
| `checkpoints` | `"final"`, `"every_layer"`, an int stride, or a list of layers: where purification is tested and where probes without their own `checkpoints` are evaluated |
| `probes` | `[{"kind": K, params...}]`; `K` is a `sparsegf2_v3.mipt` probe function (`code_dimension`, `half_cut_entropy`, `entropy` with `region`, `mutual_information` with `a` and `b`, `tripartite_information` (optional `a`, `b`, `c`), `contiguous_distance`, `lost_information` with `region`, `entropy_profile`, `erasure_lost_profile` with `samples` (default 10), `erasure_first_failure` with `samples` (default 100), `uncorrectable_region_counts` with `max_size` (default 2; key `uncorrectable_region_counts[w]`), `code_snapshot`). Every probe takes an optional `name`, its key in the results, and optional `checkpoints`, its own layers in the same forms as the spec's (default: the spec's); `capabilities(section="probes")` lists the other parameters |
| types | checked before running: booleans must be `true`/`false` (not `"false"`), counts must be integers (`4.0` is accepted, `2.5` is not), `measurement_rate` a number in [0, 1], `reference_partner` an int or `"random"`. Errors name the key |
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
| `uncorrectable_region_counts` | w = `max_size` numbers: the number of uncorrectable m-qubit erasures for m = 1..w, by exhaustion | 0 |
| `code_snapshot` | n - k, the number of stabilizer generators; the generators are stored in the saved result | n (no code stored) |

All probes except the entanglement ones and `code_snapshot` need a reference. Code
probes are exact after a stop, so their averages need no correction; entanglement probes
and `contiguous_distance` are NaN there and drop out of the means. At most one
`code_snapshot` per spec; give it its own `"checkpoints": "final"`, since codes are large.
The cost of `uncorrectable_region_counts` grows like C(n, w): measured one-thread costs
per evaluation at n = 256 are about 1 ms for w = 2, 40 to 50 ms for w = 3 and about 7 s
for w = 4, against 3 to 7 ms for `erasure_first_failure` with 100 samples.

The result contains:

* `num_trajectories`, `seed`, `elapsed_seconds` and `trajectories_per_second` (wall clock
  for the whole call, including building the spec and generating its graphs);
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
    depth;
  * the erasure probes are shown as P_C(m), the fraction of m-qubit erasures that are
    uncorrectable: `erasure_first_failure` as the fraction of samples with first-failure
    size at most m, for m = 0..n (`analysis.failure_by_size`), and
    `uncorrectable_region_counts` exactly, count / C(n, m), for m = 0..w with
    P_C(0) = 0 (`analysis.exact_failure_by_size`);
* `purification` (with a reference):
  * counts: `trajectories`, `purified`, `censored`, `censored_fraction`, `horizon_layer`;
  * tau statistics: `tau_median_all` (censoring-aware), `tau_mean_purified`,
    `tau_median_purified`, `tau_min`, `tau_max`, `tau_quantiles_purified`;
  * `survival`: `layers` and `fraction_unpurified`;
* `final_layer_mean`, `measurements_per_trajectory`;
* `notes` when needed (on `stop_when_purified`, `code_snapshot`, `contiguous_distance`,
  abbreviated names);
* `files` (with `output_dir`).

The scalar means are `sparsegf2_v3.analysis.mean_and_error` of
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
`sparsegf2 ensemble runs/p010/spec.json` repeats the run exactly.

**Estimates and limits.** `estimate_ensemble` returns `estimated_seconds`, `trajectories`
(the count that would run), `result_megabytes` (every probe value as a float64, plus the
saved codes), `threads`, `fits_sync_limit` and `sync_limit_seconds`. `run_ensemble`,
`estimate_ensemble` and `start_ensemble_job` refuse a run whose result would exceed
`MAX_RESULT_GB` (8 GB); the message suggests giving the large probes
(`erasure_first_failure` samples, `code_snapshot`, vector probes) fewer checkpoints of
their own, or splitting the trajectories over several output directories.

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

The summary gives k and P_C(m) per checkpoint. The failure probability when every qubit
is erased independently with probability e, Phi(e) = sum_m C(n,m) e^m (1-e)^(n-m) P_C(m),
and the tolerance e_delta = sup{e : Phi(e) <= delta} come from the saved arrays:

```python
import numpy as np
from sparsegf2_v3 import StabilizerCode, analysis, mipt

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
from sparsegf2_v3 import api
job = api.start_ensemble_job(spec, 100_000, output_dir="runs/big", seed=3)
api.job_status(job["job_id"], wait_seconds=30)   # status, trajectories_done, eta_seconds
api.load_results("runs/big")                      # works mid-run too (reads saved chunks)
```

* Chunks: `chunk_size` trajectories each (default about 10 estimated seconds, and at
  least 4 per thread). Each chunk is saved atomically to `output_dir/chunks/` when it
  finishes. On completion the chunks are merged into `output_dir/result.npz` and removed.
* Equivalence: trajectory `t` depends only on `(spec, seed, t)`, so a job's result is
  identical to a single `run_ensemble` call. The tests check this.
* Resume and extend: call `start_ensemble_job` again with the same spec and `output_dir`.
  `seed` may be omitted (the stored one is used), and a larger `trajectories` extends a
  finished run. A directory holding a different spec or seed is refused.
* Graph collections: `trajectories` may be omitted (all `G * circuits_per_graph` run) and
  cannot exceed that count. Changing `circuits_per_graph` changes the spec, so more
  circuits per graph need a new directory.
* Concurrency: compiled kernels run one at a time (a process-wide lock), so jobs and
  interactive calls interleave chunk by chunk. Every function is safe to call from any
  thread.
* Lifetime: jobs live in the process that started them. After a restart, resume from the
  directory.
* Crash safety: chunks and `result.npz` are written atomically, and chunks already merged
  into `result.npz` are ignored, so a directory stays readable and resumable whenever the
  process dies.

The CLI runs the same machinery in the foreground:
`sparsegf2 ensemble spec.json --trajectories 100000 --output-dir runs/big --chunk-size 5000`
prints progress to stderr. Ctrl-C stops it after the current chunk, prints the partial
summary and exits with code 130; rerunning the same command resumes it.

---

## 4. CLI details

* `--json` on every command prints exactly one JSON object on stdout. Logs and progress go
  to stderr. On failure the object is `{"error": {"type", "message"}}`.
* Exit codes: `0` success, `2` invalid input or usage (bad circuit text, including
  `inspect` of an invalid circuit, unknown spec key, missing file, bad JSON), `1` internal
  error, `130` interrupted (Ctrl-C).
* `-` reads the circuit or spec from stdin. `--observables` takes inline JSON or
  `@file.json`.
* `sparsegf2 ensemble` needs `--trajectories` (or a `trajectories` entry in SPEC.json),
  except for a graph collection (`graph.realizations`), which runs all of its circuits by
  default. Without `--chunk-size` it logs the trajectory count and the runtime estimate to
  stderr before running.
* Without `--json`, the same data is printed as indented `key: value` lines.

---

## 5. MCP details

* Tools: `capabilities`, `inspect_circuit`, `run_circuit`, `run_ensemble`,
  `estimate_ensemble`, `start_ensemble_job`, `job_status`, `cancel_job`, `list_jobs` and
  `load_results`. Each description includes a worked example of its arguments; the spec
  keys, graph families and probe kinds in the `run_ensemble` description are generated
  from the library, so they never go stale. Tools return structured content (the API
  dict) plus the same JSON as compact text, so the text stays within the 8000-character
  budget.
* Ensemble tool signatures: `run_ensemble(spec, trajectories=None, seed=None,
  output_dir=None, max_seconds=60)`, `estimate_ensemble(spec, trajectories=None)` and
  `start_ensemble_job(spec, output_dir, trajectories=None, seed=None, chunk_size=None)`.
* `run_ensemble` refuses runs estimated above `max_seconds` (default 60, must be positive)
  and points to `start_ensemble_job`. The estimate counts graph generation and
  preprocessing (large `realizations` collections), and the spec is built once for the
  estimate and the run.
* Errors: invalid input becomes an MCP tool error (`isError: true`) whose text is the
  API's message, so the model can correct the call. Argument-type errors are reported by
  the SDK's schema validation in the same way.
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
