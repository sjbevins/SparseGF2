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
| code dimension $k_C$ | $S(R)$ | `code_dimension()` | `observables.code_dimension`, `StabilizerCode.num_logical` | `EnsembleResult.survival`, `kaplan_meier` | [code-dimension.md](code-dimension.md) |
| lost information $\ell_C(A)$ | $I(A:R)$; correctable iff 0 | `lost_information(region)`, `erasure_lost_profile(samples)` | `observables.lost_logical_information`, `erasure_is_correctable`, `recovery_probability`, `erasure_profile`; `StabilizerCode.lost_information`, `is_correctable` | `bernoulli_erasure_curve` | [lost-information.md](lost-information.md) |
| random-region failure $P_C(m)$ | probability that a uniformly random $m$-qubit erasure is uncorrectable | `erasure_first_failure(samples)` (sampled), `uncorrectable_region_counts(max_size)` (exact), `code_snapshot()` | `StabilizerCode.first_failure_sizes`, `uncorrectable_counts`; `observables.random_erasure_statistics` | `failure_by_size`, `exact_failure_by_size`, `hybrid_failure_by_size`, `junction_dip` | [random-region-erasure.md](random-region-erasure.md) |
| independent erasure $\Phi_C(e)$, $e_\delta$, $\mu_C$, distance bounds | $\Phi_C(e) = \sum_m b_m(e) P_C(m)$ with binomial weights $b_m(e)$; $e_\delta = \sup\lbrace e : \Phi(e) \le \delta\rbrace$; $\mu_C = \mathbb{E}[D]$ | (from the probes above) | (from a saved code, as above) | `erasure_failure_probability`, `erasure_tolerance`, `mean_first_failure`, `distance_bounds` | [random-region-erasure.md](random-region-erasure.md) |

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

## Reading order

1. [code-dimension.md](code-dimension.md): $k_C$, the reference rows, and why
   the engine never eliminates reference columns.
2. [lost-information.md](lost-information.md): $\ell_C(A)$ for one region,
   correctability, complementarity, and formula (1).
3. [random-region-erasure.md](random-region-erasure.md): $P_C(m)$ in its two
   regimes (exhaustive below the cutoff $m^{\ast}(n)$, sampled above),
   $\Phi_C(e)$, $e_\delta$, $\mu_C$, the distance bounds, costs, error bars and
   the choice of cutoff.

Documents for the entanglement observables, purification and the contiguous
distance will follow the same structure.
