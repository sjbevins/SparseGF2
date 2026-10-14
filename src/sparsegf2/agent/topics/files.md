# Files and saved work

Everything a tool saves lives under the root folder. Artifacts are recognised by their marker files:

| artifact | files |
|---|---|
| ensemble run | `result.npz`, `spec.json`, `provenance.json`, `chunks/` |
| sweep | `sweep.json` plus one ensemble run per grid point |
| trajectory record | `trajectory.json`, `codes.npz`, optional `state.npz`, `probes.npz`, `circuit.stim` |
| state | `*.state.npz` |
| code | `*.code.npz` |
| graph collection | `*.graphs.json`, `*.graphs.npz` |

Tools of the `result` stack:

- `result_list(path, recursive)`: the artifacts under a folder, with kind, a one-line description and size. Start here when you do not know what exists.
- `result_get(source, name, trajectory_range, evaluation_range, max_values)`: raw values of an array inside an artifact, bounded (200 values by default), with shape and dtype. Use it when a capped summary is not enough.
- `result_merge(sources, output_dir)`: merges ensemble runs of one spec into one run, for example chunks computed on a cluster (`EnsembleResult.concatenate`).

Results are summaries capped at the server's output limit; a result that was cut says so, and the saved files hold every number.

Reading in Python:

```python
from sparsegf2 import mipt

res = mipt.EnsembleResult.load("runs/erasure/result.npz")
k = res.values["code_dimension"][:, -1]  # (trajectories,) at the last checkpoint
```

Stored codes load with `res.code(t, c)` as a `StabilizerCode`. Results that would need more than 8 GB of memory are refused; `ensemble_run(estimate_only=true)` reports the size first.
