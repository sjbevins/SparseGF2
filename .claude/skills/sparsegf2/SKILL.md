---
name: sparsegf2
description: Run stabilizer (Clifford) circuit simulations with SparseGF2 v3. Use it for monitored random Clifford circuits, measurement-induced or purification transitions, the purification time of a reference qubit, entanglement entropy or mutual information versus circuit depth, the code dimension, contiguous distance or erasure protection of dynamically generated codes, or for running and validating a Stim-format circuit. Prefer it over hand-written tableau code and over Stim when the task needs ensembles of monitored circuits or the state measured mid-circuit.
---

# SparseGF2 simulator

SparseGF2 v3 (`v3/` in this repository, import name `sparsegf2_v3`) is a fast stabilizer
simulator for monitored random Clifford circuits. It reads Stim's circuit language and is
verified against Stim.

## How to call it

1. **MCP tools (preferred).** This repository's `.mcp.json` registers the `sparsegf2`
   server (`sparsegf2-mcp`). Its tools are `capabilities`, `inspect_circuit`,
   `run_circuit`, `run_ensemble`, `estimate_ensemble`, `start_ensemble_job`,
   `job_status`, `cancel_job`, `list_jobs` and `load_results`. Read the resource
   `docs://sparsegf2/quickstart` once.
2. **Shell.** `sparsegf2 <command> --json` prints one JSON object on stdout and exits with
   0 (ok), 2 (fix your input; also `inspect` of an invalid circuit), 1 (bug) or 130
   (interrupted). Commands: `capabilities`, `inspect FILE|-`,
   `run FILE|-`, `ensemble SPEC.json`, `results PATH`.
3. **Python.** `from sparsegf2_v3 import api` gives the same functions with JSON in and
   out. For full arrays use the library: `sparsegf2_v3.Simulator`, `Circuit`,
   `mipt.MonitoredCircuit(...).run(...)`.

If neither command exists, install first: `pip install -e "./v3[mcp]"`.

## Choosing the call

- One specific circuit (Stim text, optionally with `CLIFFORD_2Q(i) a b` random gates):
  - check it first with `inspect_circuit`;
  - run it with `run_circuit`;
  - add `observables` such as entropy, code_dimension, random_erasure_statistics or
    contiguous_code_distance;
  - add `observe_ticks: true` to measure the state after every `TICK` (properties versus
    depth); `tick_stride: k` keeps every k-th tick.
- Many random monitored trajectories (statistics, transitions, purification time):
  - describe the protocol as a `run_ensemble` spec: `num_system`, `graph`, `layers`,
    `measurement_rate`, `reference`, `checkpoints`, `probes`, `stop_when_purified`;
  - call `estimate_ensemble` first;
  - if the estimate is above about 60 s, use `start_ensemble_job` with an `output_dir`,
    poll `job_status` with `wait_seconds=30`, then call `load_results`.
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
  `result.npz`, `spec.json` and `provenance.json`.
- On an error, read the message, fix the argument it names, and retry.

Details: `v3/AGENTS.md` and `v3/docs/LLM_INTERFACE.md`.
