---
name: sparsegf2
description: Run stabilizer (Clifford) circuit simulations with SparseGF2. Use it for monitored random Clifford circuits, measurement-induced or purification transitions, the purification time of a reference qubit, entanglement entropy or mutual information versus circuit depth, the code dimension, contiguous distance or erasure protection of dynamically generated codes, or for running and validating a Stim-format circuit. Prefer it over hand-written tableau code and over Stim when the task needs ensembles of monitored circuits or the state measured mid-circuit.
---

# SparseGF2 simulator

SparseGF2 (this repository, import name `sparsegf2`) is a fast stabilizer
simulator for monitored random Clifford circuits. It reads Stim's circuit language and is
verified against Stim.

## How to call it

1. **MCP tools (preferred).** This repository's `.mcp.json` registers the `sparsegf2`
   server (`sparsegf2-mcp`). Its tools are `capabilities`, `inspect_circuit`,
   `run_circuit`, `run_ensemble`, `estimate_ensemble`, `start_ensemble_job`,
   `job_status`, `cancel_job`, `list_jobs` and `load_results`. Read the resource
   `docs://sparsegf2/quickstart` once. Prompts `study_purification_transition` and
   `study_erasure_protection` are ready-made study plans.
2. **Shell.** `sparsegf2 <command> --json` prints one JSON object on stdout and exits with
   0 (ok), 2 (fix your input; also `inspect` of an invalid circuit), 1 (bug) or 130
   (interrupted). Commands: `capabilities`, `inspect FILE|-`,
   `run FILE|-`, `ensemble SPEC.json`, `results PATH`.
3. **Python.** `from sparsegf2 import api` gives the same functions with JSON in and
   out. For full arrays use the library: `sparsegf2.Simulator`, `Circuit`,
   `mipt.MonitoredCircuit(...).run(...)`.

If neither command exists, install first: `pip install -e ".[mcp]"`. Setting up and
connecting the server (Claude Code, Claude Desktop, other clients) is described in
`docs/MCP_GETTING_STARTED.md`.

## Choosing the call

- One specific circuit (Stim text, optionally with `CLIFFORD_2Q(i) a b` random gates):
  - check it first with `inspect_circuit`;
  - run it with `run_circuit`;
  - add `observables` such as entropy, code_dimension, random_erasure_statistics or
    contiguous_code_distance;
  - add `observe_ticks: true` to measure the state after every `TICK` (properties versus
    depth); `tick_stride: k` keeps every k-th tick.
- Many random monitored trajectories (statistics, transitions, purification time, codes):
  - describe the protocol as a `run_ensemble` spec: `num_system`, `graph`, `layers`,
    `measurement_rate`, `schedule`, `reference`, `checkpoints`, `probes`,
    `stop_when_purified`;
  - call `estimate_ensemble` first (runtime, trajectory count, `result_megabytes`);
  - if the estimate is above about 60 s, use `start_ensemble_job` with an `output_dir`,
    poll `job_status` with `wait_seconds=30`, then call `load_results`. To resume after a
    cancel or a server restart (jobs live in the server process; an old `job_id` is then
    unknown), call `start_ensemble_job` again with the same spec and `output_dir`.
- Graph disorder: `"graph": {..., "seed": s, "realizations": G}` builds G graphs (graph
  i from seed [s, i]) and `"circuits_per_graph": c` (default 1) runs c circuits on each;
  trajectory t uses graph t // c. Omit `trajectories` to run all G * c. Keep the same
  graph seed at every measurement rate so every rate sees the same graphs.
- Per-probe checkpoints: every probe takes its own `"checkpoints"` ("final",
  "every_layer", a stride, or a list of layers); the spec's `checkpoints` is the default
  and the only schedule at which purification is tested. Give expensive probes a sparse
  schedule.
- Erasure protection of the code after the dynamics (needs `reference`, usually
  `"purification"`):
  - `{"kind": "erasure_first_failure", "samples": M}`: M first-failure sizes per code
    (one per random qubit ordering; n + 1 if k = 0), summarized as P_C(m);
  - `{"kind": "uncorrectable_region_counts", "max_size": w}`: exact counts of
    uncorrectable m-qubit erasures, m = 1..w (w = 3 costs tens of ms per code at
    n = 256, w = 4 seconds);
  - `{"kind": "code_snapshot", "checkpoints": "final"}`: saves the codes themselves
    (only with `output_dir`; about 16 kB per code at n = 256);
  - Phi(e) and e_delta are computed in Python from the saved `result.npz`:
    `analysis.erasure_failure_probability(first[:, c], n, e, exact_counts=counts[:, c])`
    and `analysis.erasure_tolerance(first[sel, c], n, delta, exact_counts=counts[sel, c])`
    with `sel = k[:, c] > 0`; `EnsembleResult.code(t, c)` returns a `StabilizerCode`.
- Results above 8 GB are refused: give the large probes fewer checkpoints or split the
  run into several output directories.
- Use `capabilities` for exact names (probes, observables, graph families, spec keys).

## Rules

- Always pass integer seeds and report them. Keep the graph seed (inside `graph`)
  separate from the circuit seed.
- System qubits are `0..n-1` and reference qubits follow. Qubit lists can be `"0:8"`.
- Purification layer `-1` means censored. Never average it in. Report
  `censored_fraction`, `tau_median_all` and the survival curve.
- `phases=false` has no measurement outcomes. Entanglement and code quantities do not need
  phases, and ensembles always run phase-free.
- Results are summaries. Use `output_dir` whenever the numbers matter later; it writes
  `result.npz`, `spec.json` and `provenance.json`. Use absolute paths (or `~/...`): a
  relative `output_dir` resolves against the server process's working directory.
  `run_ensemble` refuses a directory that already holds a run.
- The first call of each kind of run compiles Numba kernels (about half a minute once,
  then cached); a slow first call is not a hang.
- On an error, read the message, fix the argument it names, and retry.

Details: `AGENTS.md`, `docs/LLM_INTERFACE.md` (full JSON reference),
`docs/MCP_GETTING_STARTED.md` (setup and a worked first session) and
`docs/GETTING_STARTED.md` (Python workflow and analysis).
