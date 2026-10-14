# Analysis of saved results

Every `analysis_*` tool takes `source`, a path to an ensemble run (or a sweep, for `analysis_compare`). Nothing is re-simulated, and analysis never waits on running jobs.

Selections:

- `where`: `code` (k > 0 at that evaluation), `purified`, `unpurified`, or `<scalar probe> <op> <number>`;
- `trajectory_range` and `graph_range`: ranges such as `0:100`.

Tools:

- `analysis_summary(source)`: every probe versus depth, purification statistics, and the stored spec.
- `analysis_probe(source, probe, statistic, ...)`: a statistic of any probe versus depth. `statistic` is `mean`, `median`, `quantiles`, `histogram`, `fraction_positive` or `depths`. For a vector probe it gives the per-index values at chosen depths.
- `analysis_erasure(source, rates, delta, depths, ...)`: per depth, k/n, q+, mu_D, distance bounds, Phi(e) over all circuits and over k > 0, e_delta and P_C(m). See `erasure`.
- `analysis_purification(source, quantiles, at_layers)`: censoring, tau statistics, the median over all, and the survival curve. See `purification`.
- `analysis_compare(sources or a sweep source, metrics, depth)`: one row per run, the parameters that differ between runs as columns, then the metrics. Metrics are call strings: `mean(probe)`, `stderr(probe)`, `fraction(probe, above=0)`, `phi(e=0.05, where=code)`, `e_delta(0.01)`, `mu_D`, `q_plus`, `tau_median`, `censored_fraction`, `survival(at=256)`. `help_list(what="metrics")` lists them.

Means come with standard errors. Summaries are capped in length; the line at the end of a capped result says where the full data is. To read raw values use `result_get`, or in Python `EnsembleResult.load("runs/x/result.npz")`.
