# Observables, one document each

Each document here takes one family of observables from its definition,
through the formula the simulator evaluates and the proof that it equals the
definition, down to the compiled kernels, and back up to the functions you
call. Each one ends with runnable examples on single trajectories, checked
against an independent brute force, and with pitfalls. Every statement about
the code cites the function it describes (`path::function`, relative to
`src/sparsegf2/`), and the examples are run by `tests/test_docs.py`.

For a shorter tour of the whole simulator, read
[HOW_IT_WORKS.md](../HOW_IT_WORKS.md). For the reasons behind the design, read
[DESIGN.md](../DESIGN.md).

## Index

| observable | definition | engine probe (`mipt`) | on a state or a saved code | from saved results (`analysis`) | document |
|---|---|---|---|---|---|
| entanglement: $S(A)$, half-cut entropy, $I(A:B)$, $I_3(A:B:C)$, entropy profile | $S(A) = -\operatorname{tr}\rho_A \log_2 \rho_A$, and signed sums of it | `entropy(region)`, `half_cut_entropy()`, `mutual_information(a, b)`, `tripartite_information()`, `entropy_profile()` | `observables.entropy` (`Simulator.entropy`), `mutual_information`, `tripartite_information`, `entropy_profile` | `EnsembleResult.mean`, `mean_and_error` (both skip the NaN after `stop_when_purified`) | [entanglement.md](entanglement.md) |
| code dimension $k_C$ | $S(R)$ | `code_dimension()` | `observables.code_dimension`, `StabilizerCode.num_logical` | `EnsembleResult.survival`, `kaplan_meier` | [code-dimension.md](code-dimension.md) |
| lost information $\ell_C(A)$ | $I(A:R)$; correctable iff 0 | `lost_information(region)`, `erasure_lost_profile(samples)` | `observables.lost_logical_information`, `erasure_is_correctable`, `recovery_probability`, `erasure_profile`; `StabilizerCode.lost_information`, `is_correctable` | `bernoulli_erasure_curve` | [lost-information.md](lost-information.md) |
| **new:** weight-one logical operators $N_{\log}$, site information $I(i:R)$, untouched qubits $N_{\mathrm{bare}}(T)$ | $N_{\log} = \lvert\lbrace i : I(i:R) > 0\rbrace\rvert$ (qubits carrying a single-qubit Pauli in $N(S) \setminus S$); $d_{\mathrm{cont}} \ge 2$ iff $k_C > 0$ and $N_{\log} = 0$; $N_{\mathrm{bare}}(T)$: qubits no gate or measurement reached by layer $T$ | `logical_sites()`, `site_information()`; spec option `record_first_touch` (`EnsembleResult.first_touch`) | `StabilizerCode.logical_sites`, `site_information` | `EnsembleResult.bare_counts`, `bare_times`, `kaplan_meier` | [logical-sites.md](logical-sites.md) |
| random-region failure $P_C(m)$ | probability that a uniformly random $m$-qubit erasure is uncorrectable | `erasure_first_failure(samples)` (sampled), `uncorrectable_region_counts(max_size)` (exact), `code_snapshot()` | `StabilizerCode.first_failure_sizes`, `uncorrectable_counts`; `observables.random_erasure_statistics` | `failure_by_size`, `exact_failure_by_size`, `hybrid_failure_by_size`, `junction_dip` | [random-region-erasure.md](random-region-erasure.md) |
| independent erasure $\Phi_C(e)$, $e_\delta$, $\mu_C$, distance bounds | $\Phi_C(e) = \sum_m b_m(e) P_C(m)$ with binomial weights $b_m(e)$; $e_\delta = \sup\lbrace e : \Phi(e) \le \delta\rbrace$; $\mu_C = \mathbb{E}[D]$ | (from the probes above) | (from a saved code, as above) | `erasure_failure_probability`, `erasure_tolerance`, `mean_first_failure`, `distance_bounds` | [random-region-erasure.md](random-region-erasure.md) |
| **new:** geometry-correlated failure $P_{C,G,J}(m)$, $\Phi_{C,G,J}(\varepsilon)$, $\varepsilon_\delta(J)$ | probability that an $m$-qubit region drawn with weight $e^{-J b_G(A)}$ ($b_G$: edges of $G$ leaving $A$) is uncorrectable, and its binomial mixture | `correlated_erasure(couplings, sizes, samples)` (sampled; `EnsembleResult.correlated`) | `StabilizerCode.correlated_failure`, `boundary_failure_counts`, `interval_failure_by_size`, `end_window_failure_by_size`; regions from `erasures.sample_regions` | `exact_correlated_failure`, `correlated_failure_curve`, `failure_probability_from_curve`, `tolerance_from_curve`, `binomial_window`, `mean_over_codes` | [correlated-erasure.md](correlated-erasure.md) |

The JSON forms (CLI and MCP) use the same names: a probe is
`{"kind": "<mipt function>", ...arguments}`, and a single-circuit observable
is `{"kind": "<observables function>", ...arguments}`. Each document lists
its own.

## Shared notation

| symbol | meaning |
|---|---|
| $n$, $[n]$ | number of system qubits, and the system (qubits $0..n-1$) |
| $R$ | the reference qubits, $n..$; purification picture: $n$ of them, single picture: one |
| $C$, $[[n, k_C]]$ | the code the circuit has generated at a checkpoint |
| $S$, $s = n - k_C$ | its stabilizer group and number of independent generators |
| $H$ | an $s \times 2n$ check matrix: X bits of the $n$ qubits, then Z bits |
| $H_A$ | the X and Z columns of $H$ for the qubits in $A$; $\bar A$ is the complement of $A$ in $[n]$ |
| $N(S)$ | the normalizer: Paulis commuting with all of $S$, dimension $n + k_C$ |
| $V_A$, $V\vert_A$ | a subspace's elements supported on $A$, and its projection onto $A$ |
| $D$ | first-failure size along a uniformly random ordering of the qubits |
| $M$ | orderings per evaluation (`samples`) |
| $w = m^{\ast}(n)$ | exhaustive cutoff (`max_size`) |
| $G$, $b_G(A)$ | an interaction graph on the qubits, and the number of its edges with exactly one end in $A$ |
| $J$, $Q_J(A \mid m)$ | the coupling, and the correlated law $e^{-J b_G(A)}/Z_m(J)$ on regions of size $m$ |

## Reading order

1. [entanglement.md](entanglement.md): $S(A)$ and the rank formula that every
   other document builds on, the smaller side and the single-qubit fast path,
   $I(A:B)$, $I_3(A:B:C)$, profiles in one pass, why sign tracking does not
   matter, and the NaN after `stop_when_purified`.
2. [code-dimension.md](code-dimension.md): $k_C$, the reference rows, and why
   the engine never eliminates reference columns.
3. [lost-information.md](lost-information.md): $\ell_C(A)$ for one region,
   correctability, complementarity, and formula (1).
4. [random-region-erasure.md](random-region-erasure.md): $P_C(m)$ in its two
   regimes (exhaustive below the cutoff $m^{\ast}(n)$, sampled above),
   $\Phi_C(e)$, $e_\delta$, $\mu_C$, the distance bounds, costs, error bars and
   the choice of cutoff.
5. [correlated-erasure.md](correlated-erasure.md) (new): erasures that cluster
   on an interaction graph $G$, $Q_J(A \mid m) \propto e^{-J b_G(A)}$. The
   owner's study transcribed and corrected, the Ising form, the exact limits
   (uniform at $J = 0$; intervals on the ring, the two end intervals on the
   open path, minimum-boundary regions elsewhere as $J \to \infty$), the exact
   ring and path samplers and the swap chain with replica exchange, with
   proofs, the estimators and their diagnostics, the exact boundary-resolved
   counts, $\Phi_{C,G,J}$ and $\varepsilon_\delta(J)$, the in-run probe and
   the tools `sample_erasure_regions` and `erasure_curves`, and their cost.
   Read random-region-erasure.md first: $J = 0$ is its case.
6. [logical-sites.md](logical-sites.md) (new): $I(i:R)$ for one qubit from two
   column ranks, the count $N_{\log}$ of qubits carrying a weight-one logical
   operator and why it decides $d_{\mathrm{cont}} \ge 2$, why membership in $S$
   matters, the untouched qubits and their first touches, and the cost (the
   $d = 1$ step of the contiguous distance, for every qubit). Read
   lost-information.md first.

Documents for purification and the contiguous distance will follow the same
structure.
