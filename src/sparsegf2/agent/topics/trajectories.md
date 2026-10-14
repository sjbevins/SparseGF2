# Single trajectories

`traj_run` simulates one trajectory of a protocol, the one with index `trajectory` (default 0) of the ensemble that `seed` defines. It takes the protocol and graph parameters of `ensemble_run`.

- `store` chooses what is kept at each checkpoint: `code` (the default; needs a reference, `purification` or `single`), `state` (the whole state, which `state_new(source=...)` can open), or `none`.
- The result reports n and k at each checkpoint, plus any `probes` you ask for. Every code property is opt-in, through `probes` here or `code_analyze` on the stored code.
- `compare_stim=true` checks the stabilizer group against Stim at every checkpoint. `save_circuit=true` writes the trajectory's circuit as `circuit.stim`.
- The output is a trajectory record: `trajectory.json`, `codes.npz`, optionally `state.npz`, `probes.npz` and `circuit.stim`.

`traj_probe(source, probes, checkpoints, store)` replays a recorded trajectory with new probes at new depths and reports like `traj_run`.

## The randomness contract

Trajectory t of a run with seed s is a function of (protocol, s, t) only. Changing the probes or the checkpoints leaves even the generator matrices bit-identical. So a trajectory record is (protocol, graph collection or its seeds, s, t, package version, randomness contract), and probing it at new depths means re-running it, which is cheap: at n = 256 and 256 layers the whole trajectory takes 4.5 ms and one first-failure evaluation with 512 orderings about 5 ms. The contract has a version number; a replay across versions that changed the random streams is refused with an explanation.

Use it to find an interesting trajectory in an ensemble (the one with the smallest first-failure size, say), then look at it closely.
