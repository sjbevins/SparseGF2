# PRL production campaign

This directory is the reproducible home for simulations, diagnostics, and
publication figures used in the SparseGF2 paper.  Generated trajectory data and
logs stay local; the protocol, code, manifests, status summary, and figure
recipes are version controlled.

The current production campaign is the generalized exact-layer raw
single-reference first-passage workflow. It is documented in
[`single_ref/raw_tau/PROTOCOL.md`](single_ref/raw_tau/PROTOCOL.md), with a
standalone process map in
[`single_ref/raw_tau/FLOWCHART.md`](single_ref/raw_tau/FLOWCHART.md). It supports
the Cartesian product of configured system sizes and named parameter axes for
the source-fingerprinted built-in graph families, as well as an adapter for the
sealed Watts-Strogatz graph registry. The scientific identities and exact
decimal measurement grid are defined in
[`sweep_spec.py`](sweep_spec.py). Production parameters remain unset until the
prelaunch review and the production-granularity checks in
[`single_ref/BENCHMARKS.md`](single_ref/BENCHMARKS.md) are complete.

Directory map:

- `inputs/`: immutable steering inputs, not fit constraints.
- `single_ref/raw_tau/`: generalized graph providers, exact-layer trajectory
  kernel, resumable raw shards, guarded process runner, benchmark harness,
  protocol, and flowchart.
- `single_ref/` outside `raw_tau/`: retained preliminary campaign code. Its
  runner, validator, monitor, and launcher do **not** implement the generalized
  `N_circuits`, graph-registry, or exact-layer protocol and are not production
  entry points for this campaign.
- `analysis/`: survival estimates, live aggregation, scaling fits, and the
  predeclared analysis protocol.
- `graph_registry/`: the persistent 350,000-draw Watts-Strogatz graph catalog,
  extensible invariant/result schema, and its validation protocol.
- `manifests/`: exact run metadata and reproducibility records.
- `data/`: generated graph banks and per-point trajectory arrays (ignored).
- `logs/`: detached-run logs (ignored).
- `figures/raw/`: direct diagnostic plots from unprocessed results.
- `figures/diagnostics/`: loss landscapes, collapses, and audit plots.
- `figures/publication/`: final paper-ready figures.
- `STATUS.md`: human-readable progress report written atomically by the
  monitor.

From the repository root, first request the write-free exact plan. This is the
only current production entry point:

```powershell
studies\prl_production\run_raw_tau.ps1 -Config PATH_TO_REVIEWED_CONFIG.json
```

After reviewing the printed plan, benchmark results, and exact experiment ID,
start or resume only that experiment with explicit confirmation:

```powershell
studies\prl_production\run_raw_tau.ps1 `
  -Config PATH_TO_REVIEWED_CONFIG.json -Run `
  -ConfirmExperimentId EXACT_ID_FROM_PLAN
```

`-Workers N` may override only the audited runtime worker count; it does not
alter the scientific experiment identity. Run status is written under
`DATA_ROOT/single_ref/raw_tau/runs/EXPERIMENT_ID/`. Interrupting the coordinator
leaves committed per-shard journals and completed NPZ files resume-safe; repeat
the same confirmed command to continue. A nonblocking operating-system lock
rejects an accidental second runner for the same experiment.

The coordinator also maintains
`DATA_ROOT/single_ref/raw_tau/catalog.sqlite3`, a versioned SQLite inventory of
the full planned Cartesian grid and every completed raw shard. It verifies each
final path, container SHA-256, and platform-independent logical-result SHA-256
before terminal registration. Raw NPZ arrays remain authoritative and preserve
the `[graph_index, circuit_index]` layout; workers never write the catalog or
the sealed graph registry.

Do not use `single_ref.run`, `run_single_ref.ps1`, or `pause_single_ref.ps1` for
this campaign. They remain only for reproducing the preliminary workflow.
