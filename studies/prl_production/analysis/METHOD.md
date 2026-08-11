# Single-reference phase-diagram analysis

This document fixes the analysis protocol before the production data are
complete.  Live outputs are provisional; publication fits use only a frozen,
fully validated snapshot.

## Point estimates

For each `(beta, n, p)` point, the analysis reads `stop_layer`,
`event_observed`, and `complete` together.  It computes the tie-aware
Kaplan-Meier survival curve and defines the median purification time as the
first layer at which the estimated survival probability is at most one half.
If that layer is not reached by `T_max=8n`, the median is recorded as an
unresolved lower bound, never replaced by `8n`.

Raw-curve error bars come from resampling graph indices.  A point reports its
event fraction, censoring fraction, median, bootstrap interval, and whether
the interval is limited by the depth cap.  Shaded variance bands are not used.

## Critical scaling

At each beta, the primary fit is the three-parameter scaling form

```text
tau_50(p,n) = n^z F((p-p_c)n^(1/nu)).
```

Equivalently, the fit models `log(tau_50)-z*log(n)` as one smooth function of
`(p-p_c)n^(1/nu)`.  The nuisance function is profiled out while jointly fitting
`p_c`, `nu`, and `z`.  A bounded coarse search followed by bounded local
refinement is used so one warm start cannot select an arbitrary basin.  The
existing leave-one-size-out collapse score is retained as an independent
cross-check, not as the only estimator.

The primary fit uses the predeclared central p window around each steering
center, identically for every size.  It does not remove points because their
observed median is close to either a cap or an area-law floor.  Unresolved
medians remain visible as lower limits and are excluded from the median fit.
If the fitted critical point is not safely inside the simulated window, that
beta is flagged for more simulation rather than extrapolated.

## Uncertainty and loss geometry

Graph indices are resampled as clusters: one resampled index vector is shared
across every p value at fixed `(beta,n)`, while different sizes are resampled
independently.  Each final bootstrap replicate repeats the Kaplan-Meier
summaries, point selection, and full three-parameter optimization.  Final
results use at least 500 successful replicates and report percentile intervals,
the full covariance, and basin occupancy.  Conditional fixed-`p_c` errors are
not presented as uncertainties of the joint fit.

For every beta, diagnostics include the raw curves, collapse, residuals,
one-dimensional profiled losses, and all three pairwise loss surfaces.  A
pairwise surface minimizes over the omitted parameter; it is not merely a
slice holding that parameter fixed.  Loss changes are diagnostic scores, not
likelihood-ratio confidence levels.  Statistical intervals come from the
graph bootstrap.

## Systematic checks

The final report separates bootstrap uncertainty from a systematic stability
envelope built from:

- several fixed central p windows;
- `n_min` equal to 32, 48, 64, and 96;
- multiple smooth-function complexities;
- leave-one-size-out and held-out-largest-size prediction;
- fits with and without a leading correction to scaling;
- artificial re-censoring of the exact times at `4n` and `6n`.

Agreement between the `6n` and `8n` results is the cap-stability check.  If the
critical median or its upper interval remains unresolved at `8n`, a targeted
deeper campaign is required before reporting `z`.

## Small-world crossover

The graph banks support a separate geometry audit at every `(beta,n)`: realized
rewire count, zero-rewire fraction, algebraic connectivity, and average path
length.  Results are plotted against both beta and the expected shortcut count
`2*n*beta`.  Near `2*n*beta` of order one, drift with `n_min` is treated as a
finite-size geometry crossover and not automatically as a new asymptotic
critical exponent.
