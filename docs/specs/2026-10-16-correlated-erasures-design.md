# Geometry-correlated erasures: design

**Owner's request (2026-10-16):** implement the study "Geometry-correlated erasures in monitored quantum circuits"
(the owner's note, `implementations/Geometry-correlated_erasures_in_monitored_quantum_circuits.pdf`, transcribed and
corrected in `docs/observables/correlated-erasure.md`) inside the simulator: generalized, documented in complete
mathematical detail, checked against the literature, easy for a language model to use, exposed through the MCP server,
tested exhaustively with the nearest-neighbour brickwork ring as the main baseline. Built overnight while the owner
sleeps; every decision below is a ruling recorded with its reason, for the owner to read in the morning.

**Status:** written 2026-10-16 by Claude. Implemented on branch `correlated-erasures` (pull request against `main`;
merged only when the owner says).

## 1. The mathematics (what the code must compute)

Notation: interaction graph `G = (V, E)`, `V = {0, ..., n-1}` (the simulator numbers qubits from 0), finite, simple,
undirected. Region `A ⊆ V`, size `m = |A|`. Edge boundary `∂A = {{u,v} ∈ E : u ∈ A, v ∉ A}`, `b(A) = |∂A|`.

- **Fixed-weight correlated law:** `Q_J(A | m) = exp(-J b(A)) / Z_m(J)`, `Z_m(J) = Σ_{|A|=m} exp(-J b(A))`, for `A` of
  size `m` (zero otherwise). `J` real (the note takes `J >= 0`; we allow any finite real `J`: `J < 0` favours large
  boundaries, spread-out erasures; every identity below holds, only the sign of the monotonicity flips).
- **Ising form:** with `σ_v = -1` for `v ∈ A`, `+1` otherwise, `b(A) = Σ_{{u,v}∈E} (1 - σ_u σ_v)/2`, so
  `exp(-J b(A)) ∝ exp((J/2) Σ_{{u,v}∈E} σ_u σ_v)`: the ferromagnetic (for `J > 0`) Ising model with coupling `K = J/2`
  on `G` at fixed magnetization `Σσ = n - 2m` (the canonical, conserved-order-parameter ensemble).
- **Density of states:** `N_m(b) = #{A : |A| = m, b(A) = b}`; `Z_m(J) = Σ_b N_m(b) e^{-Jb}`; `⟨b⟩ = -∂_J log Z`,
  `∂_J ⟨b⟩ = -Var(b) <= 0` (the note's Proposition 2). `N_m` depends on `G` only.
- **Failure:** `F_C(A) = 1[I(A:R) > 0]` with `I(A:R) = rank(A) - rank_free(A)` as the simulator computes it
  (`lost_information_rows`). `P_{C,J}(m) = Σ_{|A|=m} Q_J(A|m) F_C(A)`; `P_{C,0}(m) = P_C(m)` (the uniform observable).
- **Boundary-resolved failure counts:** `U_m(b) = #{A : |A| = m, b(A) = b, F_C(A) = 1}`. Then, exactly and for every
  `J` at once, `P_{C,J}(m) = Σ_b U_m(b) e^{-Jb} / Σ_b N_m(b) e^{-Jb}`. At `J = 0`: `Σ_b U_m(b) / C(n,m)` = the existing
  exact `P_C(m)` (`uncorrectable_counts`), an identity the tests check. As `J -> +∞`: `U_m(b_min) / N_m(b_min)`, the
  fraction of minimum-boundary sets that fail (Proposition 3).
- **Binomially mixed channel:** `Φ_{C,J}(ε) = Σ_{m=0}^{n} C(n,m) ε^m (1-ε)^{n-m} P_{C,J}(m)`, `P(0) = 0`,
  `P(n) = 1[k > 0]`. At `J = 0` it is the IID channel's `Φ_C(ε)` (Theorem 2, Corollary 2). Tolerance
  `ε_δ(J) = sup{ε : Φ_{C,J}(ε) <= δ}`.
- **Ring `C_n` (n >= 3):** for `1 <= m <= n-1`, `b(A) = 2r` with `r` the number of runs, and
  `N(n,m,r) = (n/r) C(m-1,r-1) C(n-m-1,r-1)` for `1 <= r <= min(m, n-m)`. (The note's first statement of this formula
  has `n/4`, a typo; its proof gives `n/r`; checked by brute force for n = 3..12, every m and r.) Minimum boundary 2,
  attained by exactly the `n` intervals, so `Q_J(·|m) -> uniform on the n intervals` as `J -> ∞`. **Open path `P_n`:**
  minimum boundary 1, attained only by the two intervals that touch an end (checked by brute force): `J -> ∞` gives
  those two, not every interval. **Graphs with shortcuts** (Watts–Strogatz, β > 0): minimum-boundary sets need not be
  contiguous in the qubit order; the docs must say so.
- **Exact ring sampler** (for tests): draw `r` with probability `N(n,m,r) e^{-2Jr} / Z`, a uniform composition of `m`
  into `r` positive parts and of `n-m` into `r` positive parts (stars and bars: a uniform `(r-1)`-subset of the `m-1`
  (resp. `n-m-1`) gaps), and a uniform start vertex; the map from these triples to configurations is exactly `r`-to-1
  (the choice of which run is "first"), so configurations with `r` runs come out uniformly.
- **Metropolis swap chain (Kawasaki dynamics):** propose `u` uniform in `A`, `v` uniform in `V \ A`,
  `A' = A \ {u} ∪ {v}`; `q(A,A') = 1/(m(n-m)) = q(A',A)`; accept with `min(1, e^{-J Δb})`. Detailed balance with
  respect to `Q_J(·|m)` follows from the symmetric proposal. Irreducible on the `m`-subsets for any `G` (the swap graph is
  the Johnson graph `J(n,m)`, connected for `1 <= m <= n-1`, whatever `G` is). Incremental boundary change:
  `Δb = (deg_A(u) - deg_{Ā}(u)) + (deg_{Ā}(v) - deg_A(v)) + 2·1[u ~ v]`, with degrees counted in the current `A`
  (derivation in the docs; the `u ~ v` term keeps the edge `{u,v}`, a boundary edge before and after, unchanged). Cost
  `O(deg u + deg v)` per proposal. A **sweep** is `n` proposals.
- **Replica exchange (parallel tempering)** across the requested couplings (ruling R2): one chain per `J`, sorted;
  after each sweep, propose to exchange the configurations of each adjacent pair `(J_i, J_{i+1})` with probability
  `min(1, exp((J_i - J_{i+1})(b_i - b_{i+1})))` (Swendsen & Wang 1986; Geyer 1991; Hukushima & Nemoto 1996). The
  product of the targets is invariant, so each chain's marginal is exactly `Q_{J_i}`; the `J = 0` chain mixes
  instantly, and exchanges carry its configurations up the ladder. Optional (`tempering`), on by default when there are
  two or more couplings.
- **Starting state:** a uniformly random `m`-subset (an exact sample at `J = 0`, so at `J = 0` there is no burn-in bias
  at all), then `burn_in` sweeps before the first recorded state, and `thin` sweeps between recorded states.
- **Estimators and errors:** per trajectory and per `(J, m)`, `P̂ = (1/S) Σ_s F_C(A_s)` over `S` recorded states of one
  chain (consistent; biased only by incomplete equilibration). Across trajectories the chains are independent, so the
  ensemble mean's standard error from the across-trajectory spread is valid whatever the correlation inside a chain.
  Chain diagnostics (graph-side, code-free): acceptance rates, the integrated autocorrelation time of `b` with Sokal's
  automatic window (Madras & Sokal 1988; Sokal 1997), the effective sample size, and the rank-normalized split R-hat
  over independent chains (Vehtari et al. 2021).

## 2. Rulings (decisions made without the owner, with reasons)

- **R1. The probe values are failure fractions only**, shape `(couplings, sizes)` per evaluation. The chain does not
  depend on the code, so its boundary statistics (Proposition 2 checks, diagnostics) come from the graph-side sampler,
  which runs the very same kernel. Simpler storage; cost if wrong: a later probe can add them.
- **R2. Replica exchange across the requested couplings, on by default.** Kawasaki dynamics coarsens slowly at strong
  coupling (the conserved-order-parameter Ising model); exchanges fix that at no cost to exactness. Cost if wrong: one
  flag (`tempering: false`) restores independent chains.
- **R3. `J` may be any finite real number** (negative: anti-correlated erasures). Generalization the note's algebra
  already supports. `+∞` is not a coupling: the exact `J -> ∞` answers come from `interval_failure_by_size` (ring) and
  `minimum_boundary_regions` (small graphs), never from a chain at "infinite" coupling (which only finds local minima).
- **R4. The erasure geometry** is the trajectory's own interaction graph by default (the note's `G`), or `"cycle"` /
  `"path"` over the qubit labels (the physical layout under a Watts–Strogatz circuit, which is built on that ring).
  Cost if wrong: another geometry (a graph file) is a new option, not a change.
- **R5. Exact small-`m` results by boundary-resolved counts** `U_m(b)` (plus the code-free `N_m(b)`), computed by plain
  enumeration with a boundary per subset: one enumeration gives `P_{C,J}(m)` exactly for every `J`. For `m <= w` with
  `Σ_{m<=w} C(n,m)` within a budget, as the existing exhaustive cutoff. On saved codes first (`StabilizerCode`), and as
  an in-run probe when time allows.
- **R6. `sizes` default to every `m = 1..n-1`** (`Φ` needs the whole curve); a list or a range string narrows them.
  `analysis` fills a full curve: `P(0) = 0`, `P(n) = 1[k > 0]`, exact entries for `m <= w`, sampled sizes, and linear
  interpolation between sampled sizes with a note saying so; `binomial_window(n, ε, tol)` says which sizes matter.
- **R7. No version bump** (3.0.0.dev12): resuming the owner's runs stays possible; the randomness contract stays 2
  (a new sampled probe draws only from its own per-evaluation stream).
- **R9. Exact samplers wherever the erasure graph is a cycle or a path** (detected from its structure, under any
  labeling: one cycle through all `n` vertices, or a Hamiltonian path), chosen automatically (`method="auto"`; `"chain"`
  forces the Metropolis chain, `"exact"` refuses other graphs). Reason (the spectral gap of the ring walk, `docs/observables/correlated-erasure.md` §4.8):
  on the ring at large `J` the swap chain equilibrates the boundary at once but the interval's *position* only diffuses,
  about `m(n-m) / (2(1 - cos(2π/n)))` proposals (≈ 8·10^4 sweeps at n = 256, m = 64); on the open path the two
  minimum-boundary states (the end intervals) are separated by a barrier of `Δb = 1` (an interior interval), so leaving
  an end takes an expected `>= e^J` proposals there (corrected 2026-10-16 from "exponentially large", which the barrier
  is not: it is exponential in `J`, not in `n`). Exact sampling removes
  both problems where the geometry allows it, which covers the brickwork baseline. Path counts (derived and to be checked
  by brute force): a subset with `r` runs, `e ∈ {0,1,2}` of them touching an end, has `b = 2r - e`; the numbers of
  such subsets of size `m` are `C(m-1,r-1)·C(n-m-1,r)` (`e = 0`), `2·C(m-1,r-1)·C(n-m-1,r-1)` (`e = 1`) and
  `C(m-1,r-1)·C(n-m-1,r-2)` (`e = 2`, `r >= 2`), from the compositions of the `n - m` unerased vertices into the
  `r + 1` gaps (interior gaps positive, end gaps zero or positive).
- **R10. Mixing diagnostics follow positions, not only the boundary:** besides `b`, the integrated autocorrelation time
  and R-hat of membership indicators `1[v ∈ A]` for a few spread-out vertices (the slowest reported). A diagnostic on `b`
  alone says a ring chain has mixed long before its intervals have moved.
- **R8. The note's PDF is not committed;** its content goes into Markdown (`docs/observables/correlated-erasure.md`,
  a section transcribing the study), which language models read better. The two typos (the `n/4`, and `Δb = b(A') = b(A)`
  for `b(A') - b(A)`) are corrected there and said so.

## 3. Units

| Unit | What | Depends on |
|---|---|---|
| `_kernel/erasure_chain.py` | Numba: CSR from edges, `Δb`, the swap chain with replica exchange, drawing from the project's xoshiro streams; batch failure evaluation over many regions (`lost_information_rows`); exhaustive boundary-resolved enumeration | `_kernel/rng.py`, `_kernel/entropy.py` |
| `sparsegf2/erasures.py` | Graph side, public: `boundary`, `boundary_counts` (exact `N_m(b)`), `sample_regions` (chains, regions, boundaries, acceptance), `chain_diagnostics` (τ_int, ESS, R-hat), `boundary_moments` (log Z, ⟨b⟩, Var b from `N_m`), `minimum_boundary_regions`, ring closed forms (`ring_run_count`, `ring_boundary_distribution`, `ring_partition_function`) and `sample_ring_regions` (exact) | the kernel module, `graphs` |
| `codes.StabilizerCode` | `correlated_failure(edges, couplings, sizes, samples, ...)`, `boundary_failure_counts(edges, max_size)`, `interval_failure_by_size(order=None, periodic=True)` | `erasures`, kernels |
| `analysis` | `exact_correlated_failure(U, N, couplings)`, `correlated_failure_curve(...)` (a full `P(m)` per `J`, hybrid exact + sampled), `failure_probability_from_curve(P, n, ε)` (`Φ`), `tolerance_from_curve(P, n, δ)` (`ε_δ`), `binomial_window`, ensemble means with errors conditioned on `k > 0` | numpy |
| `mipt` probe `correlated_erasure` | in-run `P̂_{C,J}(m)` per trajectory and checkpoint; its own stream; `k = 0` gives 0; replay exact | kernels |
| `api` / CLI / MCP | the probe in specs, summaries (per-`J` curves conditioned on `k > 0`), capabilities, cost estimates; new read-only tools `sample_erasure_regions` (graph side: diagnostics and exact checks) and `erasure_curves` (`Φ` and `ε_δ` per `J`, uniform and correlated, from a saved run) | all of the above |
| docs | `docs/observables/correlated-erasure.md` (complete mathematics, the study transcribed, algorithms with proofs, kernels, user code, worked examples checked by brute force, literature), AGENTS.md, LLM_INTERFACE.md, the observables index, CHANGELOG, `examples/correlated_erasure.py` | |
| baseline | `tests/test_brickwork_correlated.py` (exact small-`n` brickwork checks, fast), `benchmarks/correlated_erasure_baseline.py` and its report (the physics vs the literature) | |

## 4. What must hold (the tests)

1. `Δb` incremental equals recomputation after every step, `u ~ v` included; the chain keeps `|A| = m`.
2. Stationarity: on small graphs (every subset enumerable), long chains match `Q_J` exactly in distribution (total
   variation and a chi-square test), for `J` in {−1, 0, 0.7, 2}, with and without tempering; at `J = 0` uniform.
3. Ring: `N(n,m,r)` and `Z` closed forms equal brute force; `sample_ring_regions` follows the exact law; the chain's
   `⟨b⟩`, `Var b` on `C_64` agree with the closed forms within errors; at large `J` every recorded state is an interval;
   open path at large `J`: the two end intervals only.
4. Proposition 2: `⟨b⟩` nonincreasing in `J` (exact on small graphs; within errors on `n = 64` Watts–Strogatz).
5. Codes: `Σ_b U_m(b) = uncorrectable_counts` exactly; `U_m(b)` against the brute-force GF(2) oracle; the chain
   estimate at `J = 0` agrees with the exact `P_C(m)`; at large `J` on a ring code it agrees with
   `interval_failure_by_size`; `interval_failure_by_size` against brute force; `Φ_{C,0} = Φ_C`.
6. Probe: reproducible, replay equal, chunked = single, any thread count, `k = 0` gives 0, summaries, JSON and MCP; the
   pinned digests of the existing probes unchanged.
7. Brickwork baseline (nearest-neighbour ring, `schedule="matching"`): every identity above on real monitored-circuit
   codes; `P_{C,∞}(m)` (intervals) against `P_{C,0}(m)`; the contiguous distance; the literature's expectations (the
   memo in the docs' references).
