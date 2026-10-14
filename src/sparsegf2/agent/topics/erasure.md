# Erasure protection

Run the dynamics in the purification picture; the system holds a code C with n qubits and k logical qubits (code rate k/n). Erase every qubit independently with probability e. Erasing a set A destroys l(A) bits of logical information, and the erasure is correctable exactly when l(A) = 0.

- `l(A) = I(A:R) = rank N(S)|_A - rank S|_A`, the bits lost by erasing A.
- Draw a uniformly random ordering pi of the qubits, with nested regions A_m (the first m qubits). The first-failure size is `D_C = min{m : l(A_m) > 0}` (n + 1 when k = 0).
- `P_C(m) = Pr[D_C <= m]`: the probability that a random m-qubit erasure is uncorrectable.
- `Phi_C(e) = E_D Pr[Bin(n, e) >= D]`: the failure probability under erasure rate e, estimated by the mean over M orderings. The ensemble value Phi is the mean of Phi_C.
- The hybrid estimator uses exhaustive counts for m <= w (exact) and the sampled orderings above: `sum_{m<=w} C(n,m) e^m (1-e)^(n-m) P_C(m) + mean_j Pr[Bin(n,e) >= max(D_j, w+1)]`.
- `e_delta = sup{e : Phi(e) <= delta}`: the tolerance.
- `mu_D = sum_{m=0}^{n} (1 - P_C(m))`: the mean first-failure size; a k = 0 code gives n + 1.
- Distance bounds: the distance is at most min_j D_j; with exhaustive counts up to w it is exact if some count is positive, otherwise at least w + 1.

## Workflow

1. `ensemble_run` with `reference="purification"` and probes `code_dimension`, `erasure_first_failure(samples=512)` and `uncorrectable_region_counts(2)`, at a few checkpoints.
2. `analysis_erasure(source, rates=[...], delta=0.01)` for k/n, q+, mu_D, bounds, Phi(e) over all circuits and over k > 0, e_delta and P_C(m).

A code with k = 0 has Phi = 0, so the plain mean counts purified circuits as protected; the k > 0 numbers condition on holding a code. The hybrid P_C(m) can dip at m = w + 1 through sampling noise; use more orderings.
