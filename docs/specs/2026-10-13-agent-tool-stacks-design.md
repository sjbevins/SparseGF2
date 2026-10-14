# The SparseGF2 MCP server: a general agent interface

Status: **proposed (revision 2)**, 2026-10-13. Branch `agent-stacks`.
Related: [`2026-10-13-reverse-log-design.md`](2026-10-13-reverse-log-design.md)
(deferred), [`../DESIGN.md`](../DESIGN.md) (the simulator's architecture).

Revision 2 widens revision 1's scope from the erasure-protection workflow
to the **whole simulator**, and makes growing the interface along with the
simulator a design requirement.

## 1. Goal

An agent driving SparseGF2 through MCP should be able to do **anything a
person can do with the Python library**: prepare and manipulate stabilizer
states gate by gate, run, sample, convert and cross-check Stim-language
circuits, run ensembles of monitored circuits with every protocol option and
every probe, pull out and re-probe single trajectories, analyse any code,
build and characterise interaction graphs (built-in families or any
geometry from a file), and analyse, compare, slice and plot saved results,
with long work running in the background.

The clients are Sam's own local-model agent **Clank** (mid-size models that
change over time) and **Claude Code**. Users pick and save **tool stacks**
(subsets of the tools, with pinned or re-defaulted arguments) to fit a model
and a task.

The interface **grows with the simulator**. Tools are declared in a
registry, their argument vocabularies (gates, observables, probes, graph
families, ...) are generated from the library's own registries, and a
coverage test fails when the library gains a capability the server does not
expose. Every tool is tested by construction.

## 2. Success criteria

1. **Coverage.** Every public capability of the library is reachable
   through some tool. That covers every gate and circuit instruction, every
   observable, probe kind, graph family and edge-weight family, every
   `MonitoredCircuit` field, every `StabilizerCode` method and every
   estimator in `analysis`. A test enforces this against the library's
   registries. Anything deliberately not exposed is listed in the registry's
   `NOT_EXPOSED` table with a reason, and the test reports it.
2. **Flat schemas.** Every tool's input schema has only `string`,
   `integer`, `number` and `boolean` properties, string `enum`s, and
   `array`s of those. There are no nested objects, `$ref`, `$defs`, `oneOf`,
   `anyOf` or `allOf`, and optional parameters are simply not `required`.
   Structured arguments (probes, observables, code properties, metrics,
   checkpoint lists, sweep grids) use one small **call grammar** (section 5.1).
   Enforced by a test.
3. **Size.** Each tool (name, description and schema, as JSON) is at most
   5,000 characters. The `core` bundle is at most 18,000 characters, about
   4,500 tokens. `sparsegf2-mcp --list` and `sparsegf2 stacks show` print the
   size of any stack. Enforced by tests.
4. **Self-contained text results.** The first line names the tool and what
   ran. There is a `next:` line when there is a natural follow-up, and the
   whole result fits `--max-output-chars` (default 4,000). The same data is
   attached as `structuredContent`; nothing depends on a client reading it.
5. **Every tool tested by construction.**
   * Each tool declares examples (argument sets). The suite runs every
     example through the handler, and through a real MCP stdio session for
     every built-in stack.
   * Physics results are checked against the library called directly, and
     against Stim where Stim applies.
6. **Stacks.**
   * Built-in stacks by area, bundles (`core`, `all`), and workflow presets
     shipped as data.
   * Custom stacks that a user saves with `sparsegf2 stacks save` and serves
     with `sparsegf2-mcp --stack NAME`.
7. **Runs in Clank's sandbox.**
   * Starts and answers `tools/list` in under 5 s with no network.
   * Accepts `--quiet` and keeps stdout to JSON-RPC.
   * Confines file access to `--root`.
   * `sparsegf2 warmup` precompiles every kernel.
8. **Reference documentation.** `docs/MCP_TOOLS.md` is generated from the
   registry, and a test fails if it is stale.
9. **Measured usefulness.** `scripts/eval_agent_tools.py` scores tool choice
   and argument validity on a fixed set of requests against any
   OpenAI-compatible endpoint. Not a CI gate; it is how descriptions get
   tuned. Sam then tests the server himself.

## 3. Constraints from the clients

**Clank** (github.com/sjbevins/Clank, read 2026-10-13).
* **Where it runs.** The agent loop is OpenCode inside a per-workspace
  Docker sandbox: Ubuntu 24.04, Python 3.12, no outbound network, half the
  cores, no GPU. Only the workspace is mounted, at `/workspace`.
* **Model.** Qwen3.8-27B on a llama.cpp-family engine, with 45,056 tokens of
  context on the realistic preset.
* **Consequences:**
  * The model sees each tool's `name`, `description` and `inputSchema`, and
    the result's text. Clank does not support `outputSchema`,
    `structuredContent`, annotations, progress, cancellation or
    `tools/list_changed`, so stacks are fixed at server start.
  * Long work must go start, then poll, then fetch.
  * The server must be preinstalled, with Numba's cache somewhere writable
    and persistent (`NUMBA_CACHE_DIR`).
  * Clank's console launches `sparsegf2-mcp --quiet`.
  * OpenCode enables tools per agent or mode by name pattern, so **every
    tool name starts with its stack's prefix**.

**Claude Code** reads the project `.mcp.json` and gets `--stack all`.

## 4. Architecture

```text
sparsegf2-mcp [--stack NAME] [--root DIR] [--stacks-file F] [--quiet] [--max-output-chars N] [--list]
   |
agent/server.py    low-level mcp Server: on_list_tools / on_call_tool for the resolved stack
agent/stacks.py    built-in stacks, bundles, presets (presets.json) and custom stacks -> tools + overrides
agent/registry.py  ToolDef, Param, shared parameter groups, NOT_EXPOSED, vocabularies from the library
agent/grammar.py   the call grammar (section 5.1): parse, validate against a signature, explain errors
agent/render.py    dict -> compact text: tables, "mean+-stderr", next: hints, the output cap
agent/sessions.py  open stabilizer states (state_*) and background jobs (job_*)
agent/tools/*.py   handlers by area: help, state, circuit, ensemble, traj, code, graph,
                   analysis, result, job, plot. JSON in, dict out, raise ValueError("what to change")
```

* **Low-level server.** It uses `mcp.server.lowlevel.Server(on_list_tools=...,
  on_call_tool=...)` (SDK 2.3) with schemas generated from the registry,
  instead of `MCPServer`'s signature-derived ones. That is what keeps
  schemas flat and lets stacks pin or re-default arguments.
* **Registry.**
  * A `Param` is `(name, type, description, default, enum, minimum, maximum,
    items)`.
  * A `ToolDef` is `(name, stack, summary, description, params, handler,
    examples, read_only, long_running)`.
  * Shared parameter groups (section 5.3) are defined once and reused, so
    they are identical everywhere. A parameter name has one meaning and one type across
    the whole interface (a test enforces both), so `reference` is always the picture,
    `trajectories` always a count, and qubit or row selections carry their own names
    (`reference_qubits`, `trajectory_range`, `graph_range`, `evaluation_range`).
* **One implementation.** Handlers build on the library and on `sparsegf2.api`
  (whose spec, probe and observable builders already validate everything)
  rather than duplicating them. The grammar layer turns strings into the
  same dicts the JSON API takes. `api.py` gains no new responsibilities; it
  is already 2,700 lines.
* **Entry point.** `mcp_server.py` stays as `sparsegf2-mcp` and delegates to
  `agent.server.main`. MCP **resources** expose the `help_topic` texts as
  `docs://sparsegf2/<topic>` for clients that read resources. Today's two MCP
  **prompts** become recipes inside `help_topic(recipes)`.

## 5. Conventions shared by all tools

### 5.1 Call grammar

Structured arguments are strings in one grammar. The model writes
something close to a Python call, and the server parses it. It is never
`eval`'d.

```text
item   := name [ "(" args ")" ] [ "@" checkpoints ]      # "@..." only where the tool says so
args   := arg ("," arg)*          arg := value | key "=" value      # positional args first
value  := integer | number | true | false | range | list | word | "quoted string"
range  := int ":" int [ ":" int ]                          # Python slice semantics: 0:8 = 0..7
list   := "[" value ("," value)* "]"
word   := Pauli products and names, e.g. Z0*Z1, XXZZ, random, power_law
checkpoints := final | every_layer | int (a stride) | list of layers
```

| where it is used | examples |
|---|---|
| probes (`ensemble_run`, `traj_run`, `traj_probe`) | `code_dimension`, `entropy(0:8)`, `mutual_information(a=0:4, b=8:12)@every_layer`, `erasure_first_failure(samples=512)@[64,128,256]`, `uncorrectable_region_counts(3)@final` |
| observables (`state_query`, `circuit_run`) | `entropy(0:4)`, `code_dimension(reference=16:20)`, `expectation(Z0*Z1)`, `random_erasure_statistics(system=0:16, reference=16:20, sizes=[1,2,4], samples=50)` |
| code properties (`code_analyze`) | `first_failure(samples=1000)`, `exhaustive_counts(3)`, `failure_probability(rates=[0.01,0.05], delta=0.01)`, `lost_information(0:4)` |
| metrics (`analysis_compare`, `plot_make`) | `mean(code_dimension)`, `fraction(code_dimension, above=0)`, `phi(e=0.05, where=code)`, `e_delta(0.01)`, `mu_D`, `tau_median` |
| edge weights | `power_law(alpha=2.0)` |
| sweep grids (`ensemble_sweep`) | `measurement_rate=[0.05,0.1,0.15]; num_system=[64,128]` |

* **Names come from the library.** The argument names are the library
  function's parameter names. Positional arguments map onto those parameters
  in order.
* **Every name is listed.** `help_list` lists every name with its signature
  and one line of meaning.
* **Errors teach.** A mistake comes back with its position:
  `probes[1] "entropy(regon=0:8)": unknown argument 'regon' at column 9;
  entropy takes (region)`.

### 5.2 Qubits

* System qubits are `0..n-1`, and references follow at `n..`.
* Qubit lists are ranges (`0:8`, `0:16:2`) or lists (`[0, 3, 5]`), inside
  call strings and in string-typed parameters such as `system_order`.

### 5.3 Shared parameter groups

**Graph** (`ensemble_run`, `ensemble_sweep`, `traj_run`, `graph_build`,
`graph_stats`):

* Names are `graph_` plus the library generator's parameter name.
* `graph` (enum generated from `graphs`: `cycle`, `path`, `complete`,
  `lattice_2d`, `watts_strogatz`, `newman_watts`, `erdos_renyi`,
  `random_regular`, `file`; default `cycle`).
* `graph_k` (total degree K for `watts_strogatz` / `newman_watts`, so K/2
  neighbours per side), `graph_beta`, `graph_p` (`newman_watts`,
  `erdos_renyi`), `graph_d` (`random_regular`), `graph_rows`, `graph_cols`,
  `graph_periodic` (`lattice_2d`).
* `graph_file`: a JSON or NPZ collection (section 7.3), any geometry.
* `graph_count` (G realizations, default 1) and `graph_seed` (graph i from
  seed (graph_seed, i); kept separate from the circuit seed).
* A parameter given for a family that does not take it is an error. Defaults
  never are.

**Protocol** (`ensemble_run`, `ensemble_sweep`, `traj_run`): one parameter
per `MonitoredCircuit` field, **under the library's name**, so knowledge
moves freely between Python and the tools:
* `num_system`, `layers` and `measurement_rate` are required.
* The optional fields:
  * `schedule` (enum), `gates_per_layer`, `circuits_per_graph`;
  * `reference` (enum) and `reference_partner` (integer; -1 = random per
    trajectory);
  * `scramble_gates`, `scramble_matchings`, `scramble_layers`;
  * `gate_set` (array of gate names or CLIFFORD_2Q indices) and
    `random_orientation`;
  * `edge_weights` (call string);
  * `stop_when_purified`, `system_order` (qubit-list string) and `periodic`.
* `edges` becomes the graph group. `probes` and `checkpoints` are strings in
  the call grammar.
* The coverage test checks that this mapping stays one to one with the
  dataclass.

**Run:** `seed` (integer; omitted means drawn and reported), `output_dir`
(under `--root`; default `runs/<tool>-<hash of the arguments>`, so the same
request reuses and resumes the same directory), `background` (enum `auto` |
`always` | `never`, on long-running tools, section 5.6).

**Source** (tools that read saved work): `source`, a path under `--root` to
any artifact, recognised by its marker files:

| artifact | files |
|---|---|
| ensemble run | `result.npz`, `spec.json`, `provenance.json`, `chunks/` |
| sweep | `sweep.json` plus one ensemble run per grid point |
| trajectory record | `trajectory.json`, `codes.npz`, optional `state.npz`, `probes.npz`, `circuit.stim` |
| state | `*.state.npz` |
| code | `*.code.npz` |
| graph collection | `*.graphs.json`, `*.graphs.npz` |

### 5.4 Paths

Every path resolves against `--root` (default: the current directory; in
Clank, `/workspace`). Absolute paths must lie inside it, and `..` may not
escape it.

### 5.5 Results and errors

* **Results.** Rendered text comes first, capped at `--max-output-chars`. If
  anything is cut, a final line says what was cut and where the full data is
  (the saved files, `result_get`). The data dict is attached as
  `structuredContent`.
* **Your mistakes.** A `ValueError` from a handler becomes a tool error
  saying what to change, e.g. `graph_k must be even (K/2 neighbours per
  side), got 5`.
* **Internal errors.** Anything else is reported as an internal error with
  its exception type and logged to stderr unless `--quiet`.

### 5.6 Long work

* Tools marked `long_running` take `background`:
  * `ensemble_run`, `ensemble_sweep`, `traj_run`, `traj_probe`,
    `code_analyze`, `circuit_run`, `circuit_crosscheck`.
  * With `auto`, a tool that has a runtime estimate (ensembles, sweeps)
    starts a job when the estimate exceeds `--sync-seconds` (default 60).
  * Tools without an estimate run synchronously under `auto`.
  * `always` and `never` force either way.
* **A job runs the same handler** in a worker thread. `job_status` returns
  progress while it runs, and the tool's rendered result when it is done.
* **Ensembles and sweeps keep today's chunked jobs.** Every chunk is saved
  as soon as it finishes. After a crash, cancel or server restart, calling
  the same tool again resumes, because the default `output_dir` is the
  arguments' hash.
* **Concurrency.** Compute runs under one lock, as today, so synchronous
  calls interleave with job chunks. Analysis of saved results never takes
  the lock.

## 6. Tools

`<stack>_<verb>`, 31 tools in 11 stacks. Parameters in the shared groups of
5.3 are not repeated.

### 6.1 `help`: discovery

| tool | parameters | returns |
|---|---|---|
| `help_topic` | `topic` enum: `overview`, `conventions`, `grammar`, `states`, `circuits`, `ensembles`, `probes`, `trajectories`, `codes`, `erasure`, `purification`, `entanglement`, `graphs`, `analysis`, `files`, `jobs`, `stacks`, `recipes` | a few hundred words of guidance with examples |
| `help_list` | `what` enum: `tools`, `gates`, `instructions`, `observables`, `probes`, `code_properties`, `metrics`, `graph_families`, `edge_weight_families`, `protocol_parameters`, `stacks` | exact names, signatures, one-line meanings; generated from the registries |

### 6.2 `state`: stabilizer states, gate by gate

Open states are held in the server process: at most `--max-states` (16)
states, each up to `api.MAX_QUBITS` qubits. They can be saved to files.

| tool | parameters | returns |
|---|---|---|
| `state_new` | `num_qubits` or `source` (a state file or a trajectory record with a stored state), `phases` (default true), `seed` | `state_id`, n, phases |
| `state_apply` | `state_id`, `circuit` (Stim-language text: gates, measurements, resets, MPP, noise, REPEAT; TICKs ignored) | outcomes of the new measurements (with phases), which were random, counts |
| `state_query` | `state_id`, `observables` (array of call strings: every observable of the JSON API, `expectation(pauli)` included, plus `destabilizers`) | each value; stabilizer lists only up to 32 qubits |
| `state_save` | `state_id`, `path` | the file (`*.state.npz`: the tableau, phases, signs, seed) |
| `state_close` | `state_id` | ok |
| `state_list` | none | open states: id, n, phases, measurements so far |

### 6.3 `circuit`: Stim-language circuits

Every tool takes `circuit` (text) or `circuit_file`.

| tool | parameters | returns |
|---|---|---|
| `circuit_inspect` | the circuit | counts, instruction histogram, or every bad line with its line number |
| `circuit_run` | `shots`, `phases`, `seed`, `observables` (call strings), `observe_ticks`, `tick_stride`, `num_qubits`, `output_dir` | today's `run_circuit`: the record for one shot, per-measurement means and detector / observable flip rates for many, observables at the end and per tick |
| `circuit_convert` | `to` enum `stim` (CLIFFORD_1Q/2Q expanded into Stim gates) or `canonical`, `output_file` | the converted text, or the file |
| `circuit_crosscheck` | `seed`, `at_ticks` (array; default every tick), `compare` enum `stabilizers`, `measurements`, `both` | runs the circuit in SparseGF2 and in Stim and compares the stabilizer groups (canonical forms) at the ticks. With `measurements`, it replays Stim's random outcomes and compares every outcome, detector and observable. Reports `match` or the first mismatch. Needs Stim |

### 6.4 `ensemble`: random monitored circuits

| tool | parameters | returns |
|---|---|---|
| `ensemble_run` | protocol group, graph group, `probes` (array of call strings), `checkpoints` (grammar; default `final`), `trajectories` (required for a single graph; for a collection the default is all G x `circuits_per_graph`), run group, `estimate_only` (boolean) | probe means +- stderr per checkpoint (subsampled to fit), purification statistics, throughput, saved files; or, with `estimate_only`, the runtime and memory estimate |
| `ensemble_sweep` | everything `ensemble_run` takes, plus `grid` (e.g. `measurement_rate=[0.05,0.1]; num_system=[64,128]`: every combination) | always a background job. Each grid point runs as a chunked ensemble in `output_dir/<point>/`; points reuse graph seeds and circuit seeds. `analysis_compare` makes the table |

### 6.5 `traj`: single trajectories

| tool | parameters | returns |
|---|---|---|
| `traj_run` | protocol and graph groups, `trajectory` (index in the ensemble with this seed, default 0), `probes` (optional), `checkpoints` (default `final`), `store` enum `code` (default), `state` or `none`, `compare_stim` (boolean), `save_circuit` (boolean), run group | `n` and `k` at each checkpoint, plus any probes. It stores the code (or the whole state, for `state_new(source=...)`) at each checkpoint in a trajectory record. `compare_stim` checks the stabilizer group against Stim at every checkpoint |
| `traj_probe` | `source` (a trajectory record), `probes`, `checkpoints`, `store` | **replays** the recorded trajectory with new probes at new depths (section 7.4) and reports as `traj_run` |

By default `traj_run` simulates one trajectory and only stores its code
(`store=code` needs a reference: `purification` or `single`).
Every code property is opt-in, through `probes` here or `code_analyze` on
the stored code.

### 6.6 `code`: any stabilizer code

| tool | parameters | returns |
|---|---|---|
| `code_analyze` | the code: `stabilizers` (array of Pauli strings), or `source` (a trajectory record, an ensemble run with `code_snapshot`, a `*.code.npz`, or a `*.state.npz` with `reference_qubits`) with `trajectory` (the trajectory number) and `evaluation`, or `state_id` with `reference_qubits`. `properties` (call strings; default `summary`): `summary` (n, k, generators), `lost_information(region)`, `is_correctable(region)`, `first_failure(samples, seed)`, `failure_by_size(samples)`, `exhaustive_counts(max_size)`, `distance_bounds(samples, max_size)`, `mean_first_failure(samples, max_size)`, `failure_probability(rates, delta, samples, max_size)`, `stabilizer_weights`, `stabilizers`, `contiguous_distance(order)`. `save_to` (optional `*.code.npz`) | each property |

### 6.7 `graph`: interaction graphs

| tool | parameters | returns |
|---|---|---|
| `graph_build` | `num_nodes`, graph group, `output_file`, `show_edges` (boolean; graphs up to 64 nodes) | the saved collection and its summary (count, edges, degrees) |
| `graph_stats` | `num_nodes` and graph group (a family or `graph_file`) | mean +- stderr over the collection of: lambda_2 of the Laplacian, fraction connected, mean shortest-path length, average clustering, degree min / mean / max, and for ring-based families the fraction of shortcut edges |

### 6.8 `analysis`: saved results

Every tool takes `source`. Selections, where they apply:
* `where`: `code` (k > 0 at that evaluation), `purified`, `unpurified`, or
  `<scalar probe> <op> <number>`;
* `trajectory_range` and `graph_range`: ranges.

| tool | parameters | returns |
|---|---|---|
| `analysis_summary` | the source | today's `load_results` summary: every probe vs depth, purification, the stored spec |
| `analysis_probe` | `probe`, `statistic` enum `mean`, `median`, `quantiles`, `histogram`, `fraction_positive`, `depths`, selections | the statistic of any probe vs depth (a vector probe: per index at chosen depths) |
| `analysis_erasure` | `rates`, `delta`, `depths`, selections | per depth: k/n, q+, mu_D, distance bounds, Phi(e) over all circuits and over k > 0, e_delta, P_C(m) (hybrid estimators, section 7.1) |
| `analysis_purification` | `quantiles`, `at_layers` | censoring, tau statistics among purified, tau median over all (when defined), the Kaplan-Meier survival curve |
| `analysis_compare` | `sources` (array) or a sweep `source`, `metrics` (call strings), `depth` (default each source's last) | one row per run, with the parameters that differ between runs as columns, then the metrics. Metrics: `mean(probe)`, `stderr(probe)`, `fraction(probe, above=0)`, `phi(e, where)`, `e_delta(delta)`, `mu_D`, `q_plus`, `tau_median`, `censored_fraction`, `survival(layer)` |

### 6.9 `result`: files

| tool | parameters | returns |
|---|---|---|
| `result_list` | `path` (default `.`), `recursive` | the artifacts under it: kind, one-line description, size |
| `result_get` | `source`, `name` (an array in it), `trajectory_range`, `evaluation_range`, `max_values` (default 200) | raw values, bounded, with shape and dtype |
| `result_merge` | `sources` (ensemble runs of one spec), `output_dir` | the merged run (`EnsembleResult.concatenate`, e.g. chunks run on a cluster) |

### 6.10 `job`: background work

`job_status(job_id, wait_seconds 0..60)`, `job_cancel(job_id)`,
`job_list()`. They cover any long-running tool (section 5.6). A finished
job's status contains the tool's result.

### 6.11 `plot`: figures

`plot_make`:
* **Parameters:** `kind` (enum `vs_depth`, `vs_size`, `vs_parameter`,
  `survival`, `histogram`), `sources`, `quantity` (a probe name or metric
  call string), `depth`, `where`, `output_file` (`.png`), `title`, `logx`,
  `logy`.
* **Behaviour:** it draws one or several runs (overlaid, with error bars)
  and returns the file path.
* **Dependency:** needs matplotlib, from a new optional extra `plot`. The
  stack is left out of `all` when matplotlib is missing, and a note says
  why.

## 7. Library additions

### 7.1 `sparsegf2.analysis`

* **`mean_first_failure(first_failure, num_qubits, *, exact_counts=None)`.**
  * mu_D = sum_{m=0}^{n} (1 - P_C(m)), computed from the same hybrid P_C(m)
    as `erasure_failure_probability`, so mu_D and Phi agree.
  * A k = 0 code gives n + 1.
* **`distance_bounds(first_failure, exact_counts)`.**
  * Returns per code `(low, high)`, with `high = min_j D_j`, because an
    erasure smaller than d is always correctable.
  * With exhaustive counts up to w: if some count is positive, the smallest
    such m is d exactly; otherwise `low = w + 1`.
  * k = 0 gives `(n + 1, n + 1)`, and reports exclude it.
* **The hybrid P_C(m) can dip** at m = w + 1 through sampling noise. The
  estimators stay unbiased; `analysis_erasure` flags a dip and suggests more
  orderings.

### 7.2 `sparsegf2.graphs`: statistics

`stats(edges, n)` returns `lambda2`, `connected`, `mean_path_length`,
`clustering` and the degree range. `shortcut_fraction(edges, n, k)` covers
the ring-based families. Plain NumPy; lambda_2 is dense, so n <= 4096.

### 7.3 `sparsegf2.graphs`: files

`save(path, graphs, num_nodes, weights=None)` and
`load(path) -> (num_nodes, graphs, weights)`.

* **JSON:** `{"num_nodes": n, "graphs": [[[a, b], ...], ...], "weights":
  [[w, ...], ...]?}`. A single graph may use `"edges"` instead of
  `"graphs"`.
* **NPZ:** `num_nodes`, `edges` (E x 2), `ptr` (G + 1 offsets),
  optionally `weights` (E).
* **Validation:** nodes in [0, n), no self-loops, no repeated edges, and
  `num_nodes` must equal the requested size. Errors name the graph and the
  edge.
* **The JSON API** accepts `{"family": "file", "path": ...}` too, so the
  same files work in Python, the CLI and the tools. This is how any geometry
  enters any workflow.

### 7.4 Replay

Trajectory t of `run(seed=s)` is a function of (protocol, s, t) only. On
2026-10-13 it was verified that changing probes or checkpoints leaves even
the generator matrices bit-identical. So a trajectory record is (protocol,
graph collection or its seeds, s, t, package version, randomness contract),
and probing it at new depths means re-running it.

* `MonitoredCircuit.replay(trajectory, *, seed, probes, checkpoints)` returns
  a one-row `EnsembleResult`.
* A test pins the contract.
* `randomness_contract` (now 1) is bumped whenever the engine's random
  streams change, and replays across versions are refused with an
  explanation.
* Cost at n = 256, T = 256, p = 0.1, one thread:
  * the whole trajectory takes 4.5 ms;
  * one first-failure evaluation with M = 512 takes 5.2 ms.

Walking backward from the final tableau is the deferred reverse-log spec.

### 7.5 Single-trajectory state and code export

`traj_run(store="state")` needs the full state, which the ensemble engine
does not keep. The trajectory's circuit (`MonitoredCircuit.circuit`) runs in
a phase-free `Simulator` with an `observe` callback at the requested ticks.
The test checks that this gives the engine's codes exactly. Stored states use
`Simulator.stabilizer_arrays` and come back with `Simulator.from_tableau`.

### 7.6 `StabilizerCode.contiguous_distance`

`StabilizerCode.contiguous_distance(order=None, periodic=True)`: the
smallest contiguous window of `order` (default `0..n-1`) whose erasure is
uncorrectable, so `code_analyze` offers for a saved code what the
`contiguous_distance` probe gives during a run. It is checked against that
probe on the same states.

### 7.7 `sparsegf2.crosscheck`

This module imports Stim lazily.
* `stim_stabilizers(circuit, ticks)` returns canonical stabilizers after the
  given ticks.
* `stim_replay(circuit, seed)` returns the measurement record plus detector
  and observable events, for outcome-by-outcome comparison.

It shares the canonical form with `Simulator.stabilizers(canonical=True)`.
The lock-step helpers in `tests/_lockstep.py` stay test-only.

## 8. Stacks

### 8.1 Built-in stacks, bundles and presets

* **Stacks** are the 11 areas of section 6.
* **Bundles:**
  * `all` (the default; Claude Code) is every stack whose dependencies are
    installed.
  * `core` (for local models; at most 18,000 characters) is `help_topic`,
    `help_list`, `ensemble_run`, `traj_run`, `code_analyze`,
    `analysis_summary`, `analysis_probe`, `analysis_compare`, `result_list`,
    `job_status` and `job_cancel`.
* **Presets** are workflow stacks shipped as data
  (`agent/presets.json`), in exactly the custom-stack format. They are
  examples to copy and change:

| preset | tools | defaults |
|---|---|---|
| `erasure` | `ensemble_run`, `ensemble_sweep`, `traj_run`, `code_analyze`, `analysis_erasure`, `analysis_compare`, `graph_stats`, `result_list`, `job_*`, `help_topic` | `reference=purification`, `schedule=random_edges`, `probes=[code_dimension, erasure_first_failure(samples=512), uncorrectable_region_counts(2)]`, `graph=watts_strogatz`, `graph_k=4`, `graph_count=50`, `circuits_per_graph=4` |
| `purification` | `ensemble_run`, `ensemble_sweep`, `analysis_purification`, `analysis_compare`, `plot_make`, `job_*`, `help_topic` | `reference=single`, `reference_partner=-1`, `stop_when_purified=true`, `checkpoints=every_layer`, `probes=[code_dimension]` |
| `entanglement` | `ensemble_run`, `ensemble_sweep`, `analysis_probe`, `analysis_compare`, `plot_make`, `job_*`, `help_topic` | `probes=[half_cut_entropy, tripartite_information]` |
| `circuits` | `state_*`, `circuit_*`, `code_analyze`, `help_*` | none |

### 8.2 Custom stacks

**File.** `stacks.json` is looked up in three places:
1. `--stacks-file PATH`;
2. `<root>/.sparsegf2/stacks.json` (project);
3. the user file: `%APPDATA%\sparsegf2\stacks.json` on Windows, else
   `$XDG_CONFIG_HOME/sparsegf2/stacks.json` or
   `~/.config/sparsegf2/stacks.json`.

Earlier files win. Custom names may not reuse built-in stack, bundle or
preset names.

```json
{
  "version": 1,
  "stacks": {
    "ws-erasure-128": {
      "description": "WS K=4 erasure studies at n=128",
      "include": ["erasure"],
      "exclude": ["ensemble_sweep"],
      "fixed": {"*": {"num_system": 128, "graph": "watts_strogatz", "graph_k": 4}},
      "defaults": {"ensemble_run": {"graph_count": 200}}
    }
  }
}
```

* **`include`**: stacks, bundles, presets, other custom stacks, or tool
  names.
* **`exclude`**: tool names, removed after `include`.
* **`fixed`**: values that are injected and **removed from the schema**,
  per tool or `"*"` (every tool in the stack that has the parameter). This
  is the strongest lever for smaller models.
* **`defaults`**: new defaults shown in the schema.
* **Defaults merge** from presets and included stacks, with later entries
  winning.
* **`job_status` is added automatically** to any stack with a
  `long_running` tool.
* **Validation:** unknown names, wrong types or enum values, and cycles are
  errors that name the stack.

**CLI.**

```text
sparsegf2 stacks list                          built-ins, bundles, presets, custom: tool counts and sizes
sparsegf2 stacks show NAME                     tools, fixed and default values, tools/list size
sparsegf2 stacks save NAME [--from STACK] [--include A,B] [--exclude T,...]
                     [--fix TOOL.PARAM=VALUE ...] [--default TOOL.PARAM=VALUE ...]
                     [--description TEXT] [--project | --user]
sparsegf2 stacks delete NAME [--project | --user]
```

`TOOL` may be `*`, and `VALUE` is parsed as JSON, falling back to a string.
`save` validates before writing and writes atomically. Stacks change only
through the CLI or the file, never through MCP: a model should not reshape
its own tool access, and changes take effect when the server restarts.

## 9. Growing the interface

This is how the server keeps up with the simulator.

1. **Vocabularies come from the library.** Gate names, instructions,
   observables, probe kinds, graph and edge-weight families, the
   `MonitoredCircuit` fields and the `StabilizerCode` methods are read from
   the library at server start, as `api.capabilities` already does. A new
   probe or graph family appears in the grammar, the enums and `help_list`
   with no server change.
2. **The coverage test.** For each registry it checks that every item is
   reachable through a tool, or listed in `NOT_EXPOSED` with a reason. A new
   `MonitoredCircuit` field with no tool parameter fails the suite until it
   is mapped or explicitly excluded.
3. **Adding a tool** is one handler plus one `ToolDef` in
   `agent/tools/<area>.py`, with examples. Registration puts it in its stack
   and in `all`. The schema, size and example tests then apply to it, and
   regenerating `docs/MCP_TOOLS.md` documents it. `docs/MCP_GETTING_STARTED.md`
   gets an "Adding a tool" checklist.
4. **Versioning.** The interface has its own version (`interface_version`
   in `help_list(tools)` and the server's `instructions`). It is bumped on
   any rename or removal and listed in the CHANGELOG's "Agent interface"
   entries, so Sam's saved stacks and Clank modes can be checked against it.
   `stacks show` warns when a custom stack names a tool or parameter that no
   longer exists.

## 10. Server behaviour

Flags:
* `--stack NAME` (default `all`);
* `--root DIR`;
* `--stacks-file PATH`;
* `--sync-seconds S` (60);
* `--max-output-chars N` (4000);
* `--max-states N` (16);
* `--quiet`;
* `--list` (print the resolved tools and sizes, then exit).

No kernel compiles at startup; building the server takes 0.9 s today.
`sparsegf2 warmup` compiles every kernel once into the Numba cache. The
Clank guide sets `NUMBA_CACHE_DIR` inside the workspace and runs it.

## 11. Testing

1. **Grammar.** Every construct, every error (with columns), round trips
   with the JSON API's dicts, and fuzzing so that no input raises anything
   but `ValueError`.
2. **Registry.** Flatness, size budgets, descriptions with examples,
   identical shared groups, and the coverage test of section 9.
3. **Examples.** Every tool's examples run through the handler, and through
   an MCP stdio session for every built-in stack, bundle and preset. Results
   are under the cap and start with the tool name.
4. **Physics against independent paths:**
   * `ensemble_run` against `MonitoredCircuit` and `analysis` called
     directly;
   * `traj_run` codes against the engine's `code_snapshot` (bit-exact), and
     against Stim (`compare_stim`);
   * `traj_probe` against `traj_run`, and the replay contract;
   * `state_*` against `Simulator` and Stim;
   * `circuit_crosscheck` on circuits with known differences, which it must
     catch;
   * `analysis_erasure` against `analysis` called directly;
   * `mean_first_failure` and `distance_bounds` on codes with known answers
     ([[4,2,2]]: d = 2; repetition-type codes; k = 0);
   * `graph_stats` against closed forms (cycle: lambda_2 = 2 - 2cos(2 pi/n);
     complete graph: lambda_2 = n, clustering 1; ring lattice:
     clustering 3(K - 2)/(4(K - 1))).
5. **Stacks:** resolution, custom files, `fixed` / `defaults` / `*`, every
   validation error, and the CLI round trip.
6. **Jobs:** background runs of each long-running tool, cancel, resume after
   a simulated restart, and sweeps.
7. **Docs:** `docs/MCP_TOOLS.md` is current.
8. **`scripts/eval_agent_tools.py`** (not CI): about 30 requests spanning
   every stack, scored for tool choice and schema-valid arguments, with an
   option to run the calls for real.

## 12. Documentation

* **`docs/MCP_GETTING_STARTED.md`** is rewritten for the general server:
  * stacks, presets and custom stacks;
  * running under Clank: installing into the image or workspace,
    `NUMBA_CACHE_DIR`, `sparsegf2 warmup`, `--root /workspace`, and
    `clank mcp add sparsegf2 -- sparsegf2-mcp --stack core --quiet`;
  * Claude Code;
  * a tour across every stack;
  * "Adding a tool".
* **`docs/MCP_TOOLS.md`** is the generated reference.
* **Also updated:** `AGENTS.md`, `llms.txt`, `docs/LLM_INTERFACE.md`, the
  README's agent section and `.claude/skills/sparsegf2/SKILL.md`.

## 13. Out of scope

* The reverse log: own spec, deferred (issue #2).
* A random-code baseline: needs a uniform n-qubit Clifford sampler, on the
  DESIGN.md roadmap; it arrives through section 9 when the library has it.
* Streamable HTTP transport, dynamic stacks (`tools/list_changed`), and
  editing stacks over MCP.

## 14. Breaking changes

* **Tool names.** Every MCP tool is renamed (`run_circuit` becomes
  `circuit_run`, `run_ensemble` becomes `ensemble_run`, `load_results`
  becomes `analysis_summary`, ...), without aliases.
* **Arguments.** `circuit_run` observables and ensemble probes are call
  strings, and ensemble specs are flat parameters, not a nested `spec`
  object.
* **Prompts.** The two MCP prompts move into `help_topic(recipes)`.
* **Unchanged:** the Python library, the JSON API and the CLI only gain
  functions and commands.
