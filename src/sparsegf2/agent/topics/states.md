# States, gate by gate

The `state` stack holds open stabilizer states in the server process, at most `--max-states` (16) at a time. Each one is a `Simulator`.

Workflow:

1. `state_new(num_qubits=2)` returns a `state_id`. Use `phases=true` (the default) when you need measurement outcomes or expectation values, `phases=false` for about twice the speed when only entanglement and code quantities matter. `source` opens a saved state file, or a trajectory record with a stored state.
2. `state_apply(state_id, circuit)` applies Stim-language text: gates, measurements, resets, MPP, noise, REPEAT. TICK lines are ignored. It returns the outcomes of the new measurements and which of them were random. Outcomes require `phases=true`.
3. `state_query(state_id, observables=[...])` evaluates observables written as call strings, for example `entropy(0:1)`, `code_dimension(reference=2:4)`, `expectation(Z0*Z1)`. Stabilizer lists are shown only up to 32 qubits; use `stabilizer_weights` for large states.
4. `state_save(state_id, path)` writes a `*.state.npz` file; `state_close(state_id)` frees the slot; `state_list` shows what is open.

Example: a Bell pair.

```text
state_new(num_qubits=2, seed=1)            -> state_id s1
state_apply(s1, "H 0\nCX 0 1")
state_query(s1, ["entropy(0:1)", "expectation(Z0*Z1)"])
```

The entropy of qubit 0 is 1 bit, because the pair is maximally entangled, and the expectation of Z0*Z1 is 1. Run `help_list(what="observables")` for every observable.
