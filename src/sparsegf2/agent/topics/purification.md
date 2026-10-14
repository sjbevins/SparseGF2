# Purification

Use `reference="single"`: one reference qubit `n` is Bell-paired with `reference_partner` (an index, or -1 for random per trajectory). The reference is purified once the monitoring has learned everything about it. `stop_when_purified=true` ends each trajectory at that point, which saves time deep in the pure phase.

- **Purification time tau** is the first of the spec's checkpoints at which the reference is pure. A probe's own checkpoints do not count, so use `checkpoints="every_layer"` or a fine stride to resolve tau.
- **Censoring.** tau = -1 means the reference never purified within the horizon. It is a marker, not a time. Never average tau with the -1 entries included.
- Use instead: `censored_fraction`; `tau_median_all`, defined when more than half the trajectories purified; the survival curve `purification.survival`; or `sparsegf2.analysis.kaplan_meier` on `np.where(tau >= 0, tau, final_layer)` with the event flag `tau >= 0`.

`analysis_purification(source, quantiles, at_layers)` reports the censoring, tau statistics among the purified, the median over all, and the Kaplan-Meier survival curve.

## Finding the transition

Sweep the measurement rate with `ensemble_sweep` (grid `measurement_rate=[0.05,0.1,0.15,0.2]`, optionally times `num_system=[32,64,128]`), then compare with `analysis_compare(metrics=["survival(at=256)", "censored_fraction"])`. Below the critical rate the reference survives to the horizon; above it the survival drops quickly, and the drop sharpens as the system size grows. The rate where curves for different sizes cross marks the transition.

For brickwork on a ring the entanglement transition is near p_c ~ 0.16.
