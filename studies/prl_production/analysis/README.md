# Live single-reference analysis

Run these commands from the repository root with the repository interpreter:

```powershell
.venv\Scripts\python.exe -m studies.prl_production.analysis.aggregate --run-id f76def10804a67a6 --bootstrap-resamples 50 --bootstrap-confidence 0.68
.venv\Scripts\python.exe -m studies.prl_production.analysis.plot_live --run-id f76def10804a67a6
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

The frozen publication analysis, including the three-parameter scaling fit,
cluster bootstrap, loss profiles, and systematic checks, is specified in
[`METHOD.md`](METHOD.md).
