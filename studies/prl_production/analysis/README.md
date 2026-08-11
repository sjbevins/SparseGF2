# Live single-reference analysis

Run these commands from the repository root with the repository interpreter:

```powershell
.venv\Scripts\python.exe -m studies.prl_production.analysis.aggregate --run-id f76def10804a67a6 --bootstrap-resamples 50 --bootstrap-confidence 0.68
.venv\Scripts\python.exe -m studies.prl_production.analysis.plot_live --run-id f76def10804a67a6
.venv\Scripts\python.exe -m studies.prl_production.analysis.fit_beta --run-id f76def10804a67a6 --beta-key 0 --pc-min 0.138 --pc-max 0.153 --nu-min 0.8 --nu-max 2.2 --z-min 0.4 --z-max 1.6
```

The aggregation writes `coverage.csv`, `point_summary.csv`, and
`LIVE_ANALYSIS.md` to
`studies/prl_production/analysis/runs/f76def10804a67a6/live/`. The plotting
step reads only that summary CSV and writes one raw purification-time plot per
available beta, plus `LIVE_PLOTS.md`, to
`studies/prl_production/figures/raw/f76def10804a67a6/`.

All outputs produced while the campaign is running are **preliminary**. Only
fully complete `(beta, n, p)` points enter `point_summary.csv`; absent and
partial points appear only in `coverage.csv`. Unresolved Kaplan-Meier medians
remain lower limits at `T_max=8n`, and the live 50-resample intervals are for
monitoring rather than final inference.

`fit_beta` reads only the validated summary CSV. It jointly fits
`(p_c, nu, z)` with a profiled smooth master curve and atomically writes a
JSON audit record plus raw, collapse, and residual plots under
`figures/diagnostics/<run_id>/beta_<beta_key>/`. Add `--landscapes` to create
all three pairwise loss surfaces; the omitted physical parameter and the
master curve are reoptimized at every grid cell. The command has conservative
hard work limits, and the fit bounds must be stated explicitly for each beta.
Unresolved medians remain visible as lower limits and are never substituted by
the depth cap.

The frozen publication analysis, including the three-parameter scaling fit,
cluster bootstrap, loss profiles, and systematic checks, is specified in
[`METHOD.md`](METHOD.md).
