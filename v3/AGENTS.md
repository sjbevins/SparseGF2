# SparseGF2 v3 for AI agents

This file is for LLM agents (Claude, Codex, and others) that need to run the simulator.
Human-oriented docs are in [`README.md`](README.md) and [`docs/`](docs/).

SparseGF2 is a fast stabilizer (Clifford) circuit simulator built for **monitored random
Clifford circuits**: measurement-induced and purification transitions, the codes these
circuits generate dynamically, and how well those codes survive erasures. It reads Stim's
circuit language and its results are checked against Stim in the test suite.

## Install

```sh
pip install -e "./v3[mcp]"        # from the repository root; [mcp] adds the MCP server
python -c "import sparsegf2_v3; print(sparsegf2_v3.__version__)"
```

This installs two commands: `sparsegf2` (CLI) and `sparsegf2-mcp` (MCP server). The first
simulation after installing or updating compiles the Numba kernels. That takes 10 to 30 s
once; later runs load the compiled kernels from cache.

## Which entry point to use

| You are... | Use | Why |
|---|---|---|
| an agent with MCP tools (Claude Code, Claude Desktop) | the `sparsegf2` MCP server | typed tools, compact JSON results, background jobs |
| an agent with only a shell | `sparsegf2 <command> --json` | one JSON object on stdout; exit code 0 ok, 2 bad input, 1 internal error, 130 interrupted |
| writing Python (scripts, notebooks, studies) | `sparsegf2_v3.api` for JSON in and out, or the library itself (`Simulator`, `Circuit`, `mipt.MonitoredCircuit`) | full arrays, no size limits |

All three share one implementation, [`src/sparsegf2_v3/api.py`](src/sparsegf2_v3/api.py),
so arguments, results and error messages are the same everywhere.

Register the MCP server with Claude Code (the repository's `.mcp.json` already does this
for this project):

```sh
claude mcp add sparsegf2 -- sparsegf2-mcp
```

## Five-minute tour

1. **Discover.** `capabilities` lists instructions, observables, probes, graph families,
   spec keys, limits and examples. If the overview is too long, large sections are replaced
   by their names; `capabilities(section="probes")` returns one section in full.
2. **Check a circuit.** `inspect_circuit(circuit=...)` returns counts and an instruction
   histogram, or every bad line with its line number (including `rec[-k]` lookbacks that
   point before the first measurement, and circuits too large to unroll).
3. **Run a circuit.**

   ```json
   {"circuit": "H 0\nCX 0 1\nTICK\nM 0 1", "shots": 1000, "seed": 1}
   ```

   With `shots=1` you get the measurement record as a bitstring. With more shots you get
   per-measurement means and detector and observable flip rates.
4. **Measure the state as the circuit evolves.** Add `observables` (evaluated on the final
   state) and `observe_ticks: true` (also after every `TICK`; `tick_stride: k` evaluates
   every k-th tick only):

   ```json
   {"circuit": "...", "seed": 1, "observe_ticks": true,
    "observables": [{"kind": "entropy", "region": "0:8"},
                    {"kind": "code_dimension", "reference": "16:20"},
                    {"kind": "random_erasure_statistics", "system": "0:16",
                     "reference": "16:20", "sizes": [1, 2, 4, 8], "samples": 50}]}
   ```

5. **Run an ensemble of random monitored circuits** (this is where the speed is):

   ```json
   {"spec": {"num_system": 64,
             "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": 1},
             "layers": 512, "measurement_rate": 0.1, "schedule": "random_edges",
             "reference": "single", "reference_partner": "random", "scramble_gates": 128,
             "checkpoints": "every_layer", "probes": [{"kind": "code_dimension"}],
             "stop_when_purified": true},
    "trajectories": 1000, "seed": 7, "output_dir": "runs/ws_n64_p010"}
   ```

   The summary has probe means and standard errors at up to 50 checkpoints, purification
   statistics with censoring, and throughput. `output_dir` keeps everything:
   `result.npz`, `spec.json` and `provenance.json`. For a collection of graphs
   (`"realizations"` in `graph`, see below) `trajectories` may be left out: every graph
   then gets `circuits_per_graph` circuits.
6. **Long runs.** `estimate_ensemble` gives a pessimistic runtime estimate, the number of
   trajectories and the memory of the result (`result_megabytes`). Above about 60 s, call
   `start_ensemble_job(spec, output_dir=..., trajectories=...)`, then
   `job_status(job_id, wait_seconds=30)` until it reports `completed`, then
   `load_results(output_dir)`. Every chunk is saved as soon as it finishes. To resume
   after a crash or cancel, or to extend a finished run, call `start_ensemble_job` again
   with the same spec and `output_dir`.

## The protocol behind `run_ensemble`

The spec keys are the parameters of `sparsegf2_v3.mipt.MonitoredCircuit`.

* **Qubit layout:** system qubits are `0..n-1`. Reference qubits follow at `n..`.
* **Pictures:**
  * `"purification"`: every system qubit is Bell-paired with its own reference, so the
    system starts maximally mixed with `n` logical qubits.
  * `"single"`: one reference qubit `n` is Bell-paired with `reference_partner` (an index,
    or `"random"`, drawn per trajectory).
  * `"none"`: the system starts in |0...0>.
* **Scramble (optional):** `scramble_gates` random all-to-all two-qubit Cliffords, then
  `scramble_matchings` layers of gates on random perfect matchings, then
  `scramble_layers` gate-only layers on the graph.
* **Layer:** gates on graph edges, placed by `schedule`:
  * `matching`: edge colouring; on an even ring this is brickwork;
  * `random_edges`: `gates_per_layer` edges drawn with replacement, weighted by
    `edge_weights` if given;
  * `all_edges`.

  After the gates, each system qubit is measured in Z with probability `measurement_rate`.
  A "layer" always means exactly this. It is never rescaled.
* **Checkpoints:** `"final"`, `"every_layer"`, an integer stride, or a list of layers.
  Layer 0 is the state after setup and scramble. Purification is tested at these layers.
* **Graph collections:** `"realizations": G` in `graph` builds `G` graphs, graph `i` from
  seed `[seed, i]`, and `"circuits_per_graph": c` (default 1) runs `c` random circuits on
  each. Trajectory `t` runs on graph `t // c`, so there are `G * c` trajectories; leave
  `trajectories` out to run them all (asking for more is an error). A graph is never
  redrawn during a trajectory. The same graph seed gives the same collection at every
  measurement rate, and the same circuit seed gives the same gates and random numbers
  (the qubits measured at a lower rate are a subset of those measured at a higher one).
* **Probes:** recorded at the spec's checkpoints, or at the probe's own `"checkpoints"`
  (same forms), so a cheap probe can run every layer and an expensive one at a few depths:
  * entanglement: `half_cut_entropy`, `entropy`, `mutual_information`,
    `tripartite_information`, and the vector `entropy_profile` (S of the first m qubits
    of `system_order`, m = 0..n);
  * code (need a reference): `code_dimension` (k = S(R)), `contiguous_distance`,
    `lost_information` (one fixed region), and the vector `erasure_lost_profile` (mean
    bits lost by a random m-qubit erasure, m = 0..n);
  * erasure (need a reference): `erasure_first_failure` (`samples`, default 100) records,
    for each of `samples` random orderings of the system qubits, the first-failure size D:
    the smallest m such that erasing the first m qubits loses logical information (n + 1
    when k = 0). `uncorrectable_region_counts` (`max_size` w, default 2; key
    `uncorrectable_region_counts[w]`) counts the uncorrectable m-qubit erasures for
    m = 1..w by exhaustion, so count / C(n, m) is the exact P_C(m), the fraction of
    m-qubit erasures that are uncorrectable; all counts are 0 exactly when the distance
    exceeds w;
  * `code_snapshot`: saves the code's stabilizer generators; its value is their number,
    n - k. At most one per spec; give it its own `"checkpoints": "final"`, since codes are
    large. The summary shows only the generator count; the codes are in the saved result
    (`output_dir`; `EnsembleResult.code(t, c)` in Python).

  Every probe takes an optional `"name"`, its key in the results. The summary shows a
  vector probe as a curve over m (`index`) at the last checkpoint with data, with a `what`
  line saying what the curve is, plus `vs_depth`, a grid of up to 8 checkpoints by 17
  sizes. The two erasure probes are shown as P_C(m): `erasure_first_failure` as an
  estimate for m = 0..n, `uncorrectable_region_counts` exactly for m = 0..w. A scalar
  probe on its own schedule gets a `layers` key in the summary.
* **Purification time tau:** the first of the spec's checkpoints at which the reference
  is pure (a probe's own checkpoints do not count). `-1` means censored: the reference
  never purified within the horizon.

## Conventions

* **Seeds.** Pass integers to make a run reproducible. `seed: null` draws a fresh seed and
  reports it as `seed` in the result. Graph seeds (inside `graph`) are separate from the
  circuit seed, so graph disorder and circuit disorder can be varied independently.
  Trajectory `t` depends only on `(spec, seed, t)`, so chunked and single runs agree
  exactly.
* **Phases.** `phases=true` (the default) tracks signs: measurement outcomes, detectors and
  expectation values are physical and match Stim. `phases=false` is about twice as fast,
  but has no outcomes: only *which* measurements were random. Entanglement, code and
  erasure quantities are identical in both modes. Ensembles always run phase-free.
* **Qubit lists** are JSON lists `[0, 1, 2]` or range strings `"0:3"` (Python slice
  semantics; `"0:16:2"` selects the even qubits).
* **Units.** Entropies and mutual information are in bits, as exact integers per
  trajectory.
* **Random two-qubit Cliffords** in circuit text: `CLIFFORD_2Q(i) a b` applies Clifford
  number `i` in `[0, 11520)`. A uniformly random `i` gives a uniformly random Clifford.
  `Circuit.to_stim()` expands these into Stim gates.

## Pitfalls

* Never average tau with the `-1` (censored) entries included. Use `censored_fraction`,
  `tau_median_all` (defined when more than half the trajectories purified), the survival
  curve `purification.survival`, or `sparsegf2_v3.analysis.kaplan_meier` on the saved
  arrays. `-1` is a marker, not a time: call
  `kaplan_meier(np.where(tau >= 0, tau, final_layer), tau >= 0)` with
  `tau = result.purification_layer` and `final_layer = result.final_layer` (passing tau
  itself raises an error).
* With `stop_when_purified`, code probes record their exact values for a pure reference
  after a trajectory stops (`code_dimension`, `lost_information`, `erasure_lost_profile`
  and `uncorrectable_region_counts` 0, `erasure_first_failure` n + 1, `code_snapshot` n
  with no code saved), so code averages need no correction. Entanglement probes are NaN
  there and average only the trajectories still running; the per-checkpoint `count` says
  how many there were. `contiguous_distance` averages only trajectories that still hold a
  code (it is NaN wherever the code dimension is 0).
* Results that would need more than 8 GB of memory are refused. `erasure_first_failure`
  stores `samples` numbers per evaluation and `code_snapshot` a whole code, so give them
  few checkpoints of their own. Measured one-thread costs per evaluation at n = 256:
  `erasure_first_failure` with 100 samples 3 to 7 ms; `uncorrectable_region_counts` about
  1 ms for w = 2, 40 to 50 ms for w = 3, about 7 s for w = 4 (practical only at a few
  depths).
* `stabilizers` lists generators only up to 32 qubits. For large states use
  `stabilizer_weights` or `stabilizer_length_distribution`.
* Results are summaries capped at about 8000 characters, and `truncated: true` means
  something was cut. Pass `output_dir` and read the `.npz` file in Python when you need
  every number.
* Background jobs live in the server process. If the process restarts, the job ids are
  lost but the saved chunks are not: call `start_ensemble_job` again with the same
  `output_dir` to resume.
* Errors come back as plain messages that say what to change, for example
  `line 2: unknown gate 'FOO'`, `spec: unknown key(s) ['layrs']; allowed keys are [...]`,
  or a suggestion to use a background job. Fix the input and retry. Do not work around
  the error.

## Erasure analysis in Python

The summaries show P_C(m). The failure probability under independent erasure,
Phi(e) = sum_m C(n,m) e^m (1-e)^(n-m) P_C(m), and the tolerance
e_delta = sup{e : Phi(e) <= delta} are computed from the saved arrays. A run that
records what they need (50 graphs with 4 circuits each, 200 trajectories):

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
p_m = analysis.failure_by_size(more, n)                       # P_C(m), m = 0..n
StabilizerCode(["XXXX", "ZZZZ"]).uncorrectable_counts(2)      # [0, 6]: the [[4, 2, 2]] code
```

With `exact_counts` the terms m <= w are exact and the samples cover m > w. A code with
k = 0 (D = n + 1) has Phi = 0, so the first mean counts purified circuits as protected;
select `k > 0` to condition on a code. `[:, -1]` is each probe's last evaluation
(`res.layers(name)` lists them). `StabilizerCode` takes generators as Pauli strings or an
(s, 2n) 0/1 array (X bits, then Z bits), and has `num_logical`, `lost_information(A)`,
`is_correctable(A)`, `first_failure_sizes(samples, seed=...)` and
`uncorrectable_counts(w)`.
[`examples/code_and_erasure.py`](examples/code_and_erasure.py) runs the whole analysis
versus depth.

## Where things are

| Path | What |
|---|---|
| `src/sparsegf2_v3/api.py` | the JSON service layer (CLI and MCP wrap it) |
| `src/sparsegf2_v3/cli.py` | `sparsegf2` command line |
| `src/sparsegf2_v3/mcp_server.py` | `sparsegf2-mcp` MCP server: tools, `docs://` resources, prompts |
| `src/sparsegf2_v3/mipt.py` | `MonitoredCircuit`, probes, `EnsembleResult` |
| `src/sparsegf2_v3/codes.py` | `StabilizerCode`: erasure analysis of a saved code |
| `src/sparsegf2_v3/analysis.py` | P_C(m), Phi(e), e_delta, Kaplan-Meier, means with errors |
| `src/sparsegf2_v3/observables.py` | entanglement, code and erasure observables of one state |
| `docs/LLM_INTERFACE.md` | full CLI and MCP reference |
| `docs/LITERATURE.md` | which observables the literature uses, and why |
| `docs/DESIGN.md` | architecture and the reasons behind it |

## Testing changes

```sh
cd v3 && python -m pytest          # full suite; Stim is required
ruff check src tests && ruff format src tests
```
