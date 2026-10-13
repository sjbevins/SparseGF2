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
* at most 50 checkpoints, evenly subsampled and always including the first and last;
* vector probes subsampled to 65 entries, plus a coarse grid of at most 8 checkpoints by
  17 region sizes;
* probe names longer than 64 characters abbreviated (regions written as `start:stop`
  ranges, e.g. `entropy[0:1024]`).

The full data can be written to files with `output_dir`. Whenever anything is left out,
the result carries `"truncated": true` and a note says what and how to get it.

For the quick version, read [`../AGENTS.md`](../AGENTS.md). This page is the complete
reference.

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
| provenance | every `output_dir` gets `provenance.json`: package, Python, numpy and numba versions, git commit, platform, seed, UTC timestamp, command |

---

## 3. Functions (API, MCP tools, CLI commands)

| API function / MCP tool | CLI | Purpose |
|---|---|---|
| `capabilities(section=None)` | `sparsegf2 capabilities [--section S]` | supported instructions, observables, probes, spec keys, graph families, limits, examples |
| `inspect_circuit(circuit)` | `sparsegf2 inspect FILE\|-` | parse and validate a circuit without running it |
| `run_circuit(circuit, shots, phases, seed, observables, observe_ticks, tick_stride, num_qubits, output_dir)` | `sparsegf2 run FILE\|- [...]` | run a circuit; measurement statistics and observables |
| `run_ensemble(spec, trajectories, seed, output_dir)` | `sparsegf2 ensemble SPEC.json [...]` | ensembles of random monitored circuits |
| `estimate_ensemble(spec, trajectories)` | (printed by `ensemble`) | pessimistic runtime estimate |
| `start_ensemble_job(spec, trajectories, output_dir, seed, chunk_size)` | `sparsegf2 ensemble ... --chunk-size K --output-dir D` | resumable background run |
| `job_status(job_id, wait_seconds)` | (shown as progress) | progress, throughput, ETA |
| `cancel_job(job_id)` | Ctrl-C | stop after the current chunk |
| `list_jobs()` | | jobs in this process |
| `load_results(path)` | `sparsegf2 results PATH` | summarize a saved run (file or directory, including partial chunks) |

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
| `output_dir` | none | save `circuit_run.npz` (records, random flags, detectors, observable flips), `circuit.stim`, `observables.json` (every tick) and `provenance.json` |

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
| `contiguous_code_distance` | `system`, `reference`, `periodic`? | shortest uncorrectable contiguous erasure |
| `stabilizer_weights` | | mean, min and max generator weight |
| `stabilizer_length_distribution` | `order`? | clipped-gauge length histogram |
| `stabilizers` | | canonical generators (32 qubits or fewer) |
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
| `graph` | `{"family": F, params...}`. `F` is one of `cycle`, `path`, `complete`, `lattice_2d` (`rows`, `cols`, `periodic`), `watts_strogatz` (`k`, `beta`, `seed`), `newman_watts` (`k`, `p`, `seed`), `erdos_renyi` (`p`, `seed`), `random_regular` (`d`, `seed`). Random families require their own `seed`. Add `"realizations": G` for graph disorder: `G` independent graphs (graph `i` from seed `[seed, i]`), trajectory `t` runs on graph `t mod G` |
| `edges` | instead of `graph`: `[[a, b], ...]` |
| `edge_weights` | list (one per edge) or `{"family": "power_law", "alpha": 2.0, "periodic": true}`, for `schedule="random_edges"` |
| `schedule` | `matching` (default), `random_edges`, `all_edges`; `gates_per_layer` for `random_edges` |
| `reference`, `reference_partner` | `none`/`purification`/`single`; an int or `"random"` |
| `scramble_gates`, `scramble_matchings`, `scramble_layers` | ints: random all-to-all gates, layers on random perfect matchings, then gate-only layers on the graph |
| `gate_set` | list of two-qubit gate names (`["CX", "SWAP"]`) or CLIFFORD_2Q indices |
| `checkpoints` | `"final"`, `"every_layer"`, an int stride, or a list of layers |
| `probes` | `[{"kind": K, params...}]`; `K` is a `sparsegf2_v3.mipt` probe function (`code_dimension`, `half_cut_entropy`, `entropy` with `region`, `mutual_information` with `a` and `b`, `tripartite_information` (optional `a`, `b`, `c`), `contiguous_distance`, `lost_information` with `region`, `erasure_lost_information` or `erasure_correctable_fraction` with `size` and `samples`, `entropy_profile`, `erasure_lost_profile` or `erasure_correctable_profile` with `samples`). Every probe takes an optional `name`, its key in the results; `capabilities(section="probes")` lists the parameters |
| types | checked before running: booleans must be `true`/`false` (not `"false"`), counts must be integers (`4.0` is accepted, `2.5` is not), `measurement_rate` a number in [0, 1], `reference_partner` an int or `"random"`. Errors name the key |
| `stop_when_purified`, `system_order`, `periodic`, `random_orientation` | as in the library |

The result contains:

* `num_trajectories`, `seed`, `elapsed_seconds`, `trajectories_per_second`;
* `checkpoints`: the layers shown, the total count, and whether they were subsampled;
* `probes`:
  * scalar probes: `mean` and `stderr` per checkpoint, and `count` when trajectories
    stopped early;
  * vector probes: `vector: true`; `at_layer` (the last checkpoint that has data; with
    `stop_when_purified` later ones may be empty), `count` (trajectories there), `mean`
    and `stderr` per region size, and `index` when subsampled; and `vs_depth`, the mean
    on a grid of at most 8 checkpoints (`layers`, `count`) by 17 region sizes (`index`),
    which shows how the profile changes with depth;
* `purification` (with a reference):
  * counts: `trajectories`, `purified`, `censored`, `censored_fraction`, `horizon_layer`;
  * tau statistics: `tau_median_all` (censoring-aware), `tau_mean_purified`,
    `tau_median_purified`, `tau_min`, `tau_max`, `tau_quantiles_purified`;
  * `survival`: `layers` and `fraction_unpurified`;
* `final_layer_mean`, `measurements_per_trajectory`;
* `files` (with `output_dir`).

The probe means are `sparsegf2_v3.analysis.mean_and_error` of
`EnsembleResult.values[name]`. The survival curve equals `EnsembleResult.survival()`.

```sh
$ cat spec.json
{"num_system": 64, "graph": {"family": "cycle"}, "layers": 512,
 "measurement_rate": 0.1, "reference": "single", "checkpoints": 16,
 "probes": [{"kind": "code_dimension"}], "stop_when_purified": true}
$ sparsegf2 ensemble spec.json --trajectories 2000 --seed 7 --output-dir runs/p010
```

`runs/p010/spec.json` stores `{"spec", "trajectories", "seed"}`, so
`sparsegf2 ensemble runs/p010/spec.json` repeats the run exactly.

### 3.4 Background jobs

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
  `seed` may be omitted, and the stored one is used. A directory holding a different spec
  or seed is refused.
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
* `run_ensemble` refuses runs estimated above `max_seconds` (default 60, must be positive)
  and points to `start_ensemble_job`.
* Errors: invalid input becomes an MCP tool error (`isError: true`) whose text is the
  API's message, so the model can correct the call. Argument-type errors are reported by
  the SDK's schema validation in the same way.
* Resources: `docs://sparsegf2/quickstart` (Markdown guide) and
  `docs://sparsegf2/capabilities` (JSON).
* Prompts: `study_purification_transition(num_system, graph_family, rates, trajectories)`
  and `study_erasure_protection(num_system, measurement_rate, trajectories)`.
* Implementation: the official MCP Python SDK v2 (`mcp.server.mcpserver.MCPServer`), over
  stdio. Tools are synchronous functions, which the SDK runs in worker threads, so a long
  call does not block the protocol.
