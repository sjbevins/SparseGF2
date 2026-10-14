# Conventions

- **Qubits.** System qubits are `0..n-1`. Reference qubits follow at `n..`.
- **Layers.** A layer is the gates followed by the measurements, and is never rescaled. Layer 0 is the state after the setup and the scramble.
- **Pictures** (the `reference` parameter): `none` starts the system in |0...0>; `purification` Bell-pairs every system qubit with its own reference, so the system starts maximally mixed with n logical qubits; `single` Bell-pairs one reference qubit `n` with `reference_partner` (an index, or -1 for random per trajectory).
- **Seeds.** `seed` is the circuit seed and `graph_seed` the graph seed, so graph disorder and circuit disorder vary independently. Pass integers to make a run reproducible. Trajectory t depends only on (protocol, seed, t), so chunked and single runs agree exactly.
- **Phases.** Ensembles are phase-free: they know which measurements were random but not the outcomes. States and circuits default to `phases=true`, which tracks signs, so outcomes and expectation values are physical and match Stim. Entanglement, code and erasure quantities are identical in both modes.
- **Units.** Entropies and mutual information are in bits, as exact integers per trajectory.
- **Qubit lists** are ranges with Python slice semantics (`0:8` is qubits 0 to 7, `0:16:2` the even ones) or lists such as `[0,3,5]`.
- **Paths** are confined to the root folder; relative paths are relative to it. Default output directories are `runs/<tool>-<hash of the arguments>`, so running the same request again resumes the same directory.
- **Tau** is the purification time. It is -1 when censored, meaning the reference never purified within the horizon. Never average tau with the -1 entries; use the censored fraction, the median over all, or the survival curve (see `purification`).
- **Errors** come back as plain messages that say what to change. Fix the input and retry.
