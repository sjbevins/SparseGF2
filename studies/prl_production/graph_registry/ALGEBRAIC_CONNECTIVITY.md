# Algebraic connectivity and normalized gain

Status: **complete and validated**.

Collection: `ws_c2_paper_v1_cf9d71dc40b2ab1c` (350,000 graphs in 350 cells).

For each graph we compute the second-smallest eigenvalue of the combinatorial
Laplacian.  Each cell first forms the arithmetic graph mean.  The cumulative
normalized gain is then the geometric mean across its stated size set:

\[
g_\lambda(\beta)=\exp\!\left[\frac{1}{|\mathcal N|}
\sum_{n\in\mathcal N}\ln\frac{\overline{\lambda_2(G_{n,\beta})}}
{\lambda_2[C(n,2)]}\right].
\]

Nested size sets: [64], [64, 128], [64, 128, 192], [64, 128, 192, 256].
Error bars use the log-delta SEM; the 68% interval independently resamples
graphs within each cell and reuses those cell resamples across cumulative sets.

The normalized-gain figure is generated locally by the connectivity-analysis
command documented in [README.md](README.md). Raw figures are intentionally
ignored and are not embedded in this version-controlled report.

## Cumulative-size convergence test

| beta | m=1 | m=2 | m=3 | m=4 | m=4 relative to m=3 |
|---:|---:|---:|---:|---:|---:|
| 0.005 | 1.01975 | 1.06340 | 1.12764 | 1.19397 | 5.88% |
| 0.009696137 | 1.05927 | 1.19146 | 1.34514 | 1.51079 | 12.31% |
| 0.029240177 | 1.46802 | 1.97740 | 2.58230 | 3.26331 | 26.37% |
| 0.098468966 | 3.34473 | 5.42945 | 7.89362 | 10.77970 | 36.56% |
| 0.296948151 | 8.23281 | 14.85417 | 23.03533 | 32.70333 | 41.97% |
| 1 | 13.41153 | 25.26144 | 40.25956 | 58.34518 | 44.93% |

The cumulative curves do not converge to a size-independent master curve over
the four available nested sets. At the smallest positive grid point, adding
`n=256` raises the aggregate by 5.88%; at `beta=1` the increase is 44.93%.
The successive `beta=1` ratios are 1.8836, 1.5937, and 1.4492. The relative
drift decreases with set size but remains substantial.

This is not a statistical-resolution problem. The unrewired gap scales as
`lambda_2[C(n,2)] ~ 20 pi^2/n^2`. A rewired gap that approaches a nonzero
large-`n` value therefore produces a normalized single-size gain that grows
approximately as `n^2`. These data support a growing cumulative gain rather
than an unscaled limiting master curve.

## Locally generated artifacts

The following paths are reproducible outputs, not version-controlled files:

- Per-cell summary: `studies/prl_production/data/graph_registry/ws_c2_paper_v1_cf9d71dc40b2ab1c/invariants/graph.algebraic_connectivity.combinatorial_laplacian_v1/algebraic_connectivity_summary.csv`
- Nested normalized gain: `studies/prl_production/data/graph_registry/ws_c2_paper_v1_cf9d71dc40b2ab1c/invariants/graph.algebraic_connectivity.combinatorial_laplacian_v1/normalized_connectivity_gain.csv`
- Per-graph raw values: `studies/prl_production/data/graph_registry/ws_c2_paper_v1_cf9d71dc40b2ab1c/invariants/graph.algebraic_connectivity.combinatorial_laplacian_v1/algebraic_connectivity_raw.npz`
- Run manifest: `studies/prl_production/data/graph_registry/ws_c2_paper_v1_cf9d71dc40b2ab1c/invariants/graph.algebraic_connectivity.combinatorial_laplacian_v1/connectivity_invariant_manifest.json`
- Figure PNG: `studies/prl_production/figures/raw/graph_geometry/ws_c2_paper_v1_cf9d71dc40b2ab1c/algebraic_connectivity_gain_convergence.png`
- Figure PDF: `studies/prl_production/figures/raw/graph_geometry/ws_c2_paper_v1_cf9d71dc40b2ab1c/algebraic_connectivity_gain_convergence.pdf`
- Logical result SHA-256: `2f4152ae245e8a04aeb1e12916617eec4dcdc669096d2b985476d65a8528613b`
