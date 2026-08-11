# PRL production campaign

This directory is the reproducible home for simulations, diagnostics, and
publication figures used in the SparseGF2 paper.  Generated trajectory data and
logs stay local; the protocol, code, manifests, status summary, and figure
recipes are version controlled.

The first campaign is the single-reference purification-time phase diagram in
`single_ref/`.  It uses Watts-Strogatz rewiring of `C(n, 2)`: every graph has
`2n` edges and mean degree four.  See [PROTOCOL.md](PROTOCOL.md) before
launching a production job.  [PROVENANCE.md](PROVENANCE.md) maps the active
run to its published source commit, and [analysis/METHOD.md](analysis/METHOD.md)
fixes the statistical protocol before the data are complete.

Directory map:

- `inputs/`: immutable steering inputs, not fit constraints.
- `single_ref/`: simulator, campaign runner, validation, and monitor.
- `analysis/`: survival estimates, live aggregation, scaling fits, and the
  predeclared analysis protocol.
- `manifests/`: exact run metadata and reproducibility records.
- `data/`: generated graph banks and per-point trajectory arrays (ignored).
- `logs/`: detached-run logs (ignored).
- `figures/raw/`: direct diagnostic plots from unprocessed results.
- `figures/diagnostics/`: loss landscapes, collapses, and audit plots.
- `figures/publication/`: final paper-ready figures.
- `STATUS.md`: human-readable progress report written atomically by the
  monitor.

From the repository root, use only the repository interpreter:

```powershell
.venv\Scripts\python.exe -m studies.prl_production.single_ref.validate
.venv\Scripts\python.exe -m studies.prl_production.single_ref.run --profile smoke
```

The production profile is intentionally explicit and never the default.  It is
started only after the simulator review, test suite, protocol review, and Git
commit are complete:

```powershell
studies\prl_production\run_single_ref.ps1 -Profile production -Workers 8 -ConfirmProduction
```

The detached launcher records the resolved run ID, its own process IDs, and
separate runner and monitor logs.  To pause it after the latest atomic point
checkpoint, use
`pause_single_ref.ps1 -Force`; the next identical launch resumes incomplete
graph indices.  A single-writer lock rejects an accidental second runner.
