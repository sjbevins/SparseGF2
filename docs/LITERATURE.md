# Literature review: what SparseGF2 must support

Compiled 2026-10-13 from six parallel literature and tooling sweeps.

**Verification status.** arxiv.org, ar5iv, journals.aps.org, Semantic Scholar and INSPIRE could not be reached from the build sandbox (proxy 403 or DNS failure). Citations were checked against search-engine abstracts plus recollection. A dagger (†) marks anything that has not been checked against the paper's full text: numerical values, model details, journal references, or author lists. Check these before quoting them in a paper. Benchmark numbers in §3.5 were measured locally and are verified.

**Review pass (2026-10-13, completeness critic / fact check).** A second pass checked the citations and formulas against search-engine copies of arXiv and APS abstracts. arxiv.org itself was still unreachable, so full texts were not read. Every formula in §2 and §3.2 was re-derived by hand. Edits are tagged inline: **[corrected]** fixes an error, **[added]** fills a gap, and **[verified]** removes a dagger that the abstract now supports. §3.7 lists every change and the citations that remain unidentified.

---

## 1. Purpose

This document fixes the physics scope of SparseGF2. The research targets are:

- measurement-induced and purification transitions in monitored random Clifford circuits on graphs;
- the dynamically generated codes these circuits produce (code dimension, contiguous distance, erasure protection) as functions of depth;
- the single-reference purification time τ.

The document also covers the general Stim-like protocols, the competitor simulators for fair benchmarking, and the interface an LLM agent needs to drive all of this. Each item below says what must be computed, how it reduces to GF(2) linear algebra on the column-major bit-packed tableau, what it costs, and whether it must run inside the evolution kernel.

### Notation

| Symbol | Meaning |
|---|---|
| n | system qubits |
| n_ref | reference qubits |
| N | tableau rows (stabilizers; destabilizers are added in phase mode) |
| W | ⌈N/64⌉, words per packed column |
| x_q, z_q ∈ GF(2)^N | packed columns of qubit q |
| cols(A) | {x_q, z_q : q ∈ A} |
| rank(A) | GF(2) rank of cols(A) over all rows |
| rank⁰(A) | the same rank restricted to rows with no reference support |
| Rmask | packed bitmask of rows with nonzero reference part |
| k | code dimension (number of logical qubits) |
| r | number of system-only stabilizer generators, r = n − k |

Entropies are in bits.

---

## 2. Prioritized feature table

**Must**

| ID | Feature | Why | How on the tableau (cost) | Run during evolution | Key citations |
|---|---|---|---|---|---|
| M1 | Column-rank entropy engine: any region (non-contiguous allowed), batched, nested-prefix sweeps, linear combinations Σcᵢ S(Aᵢ) | Every entanglement observable is built on it | Pure state: S(A) = rank(A) − \|A\|. Any stabilizer state with N independent rows on m qubits: S(A) = \|A\| − N + rank(Ā). **[added]** Dual form: if the rows are the stored restriction of a purification's generators (they may be linearly dependent, e.g. the environment rows of S2), then S(A) = rank(A) − \|A\| still holds, because only the span of the rows restricted to A enters. Insert columns into a pivot-indexed XOR basis, so no transpose is needed. One region costs O(\|A\|·min(2\|A\|,N)·W). One prefix sweep gives every nested S in O(n·N·W). | y | quant-ph/0406168; 1608.06950; 1901.08092 |
| M2 | Reference qubits as first-class objects (columns gates never touch, plus Rmask). Initial pictures: product, maximally mixed (n Bell pairs), k₀ pairs, single reference at a fixed or random site. Optional scramble of q_scramble·n gates or t₀ layers. | Purification, code and τ observables | Reference Bell rows initialise Rmask. N = n + n_ref. | setup | 1905.05195; 1910.00020; 2010.09775 |
| M3 | Code dimension k(t), code rate k/n, purification events | Order parameter of the purification transition | Pivot rule: pick a reference-free anticommuting row if one exists; only otherwise pick an Rmask row (that row leaves Rmask and k decreases by 1). Then k = popcount(Rmask) − n_ref. Cost O(W) per measurement, O(1) to read. **[corrected]** popcount(Rmask) − n_ref always equals S(R). It equals S(sys), the code dimension, only while system+reference is pure. Once an S2 channel has acted, report I(R:sys)/2 or the coherent information instead (S21). | y | 1905.05195; 2008.10611 |
| M4 | Single-reference S_R(t) and τ with right-censoring and early stop | Scalable local order parameter; the single-reference purification observable | S_R = rank{x_R, z_R} − 1. It equals 1 iff both columns are nonzero and unequal, so the check is O(W). It can change only after a measurement. Store τ (−1 if censored), event_observed, stop_layer, ref site. **[added]** Stopping early is exact: once S_R = 0, R is in a product state with the system, and operations on the system alone cannot re-entangle it. Channels never change S_R either, because S(R) is invariant under any trace-preserving map on the system. So under S2 noise S_R can stay at 1 forever while the information leaks to the environment. τ is then not a protection measure; use the first time I(R:sys) reaches 0 instead. | y | 1910.00020; 1905.05195 |
| M5 | I(A:R) for any erasure set; optimal recovery probability | Erasure protection | I(A:R) = rank(A) − rank⁰(A). Correctable ⇔ I = 0. P_rec = 2^−I. For large A use I(A:R) = 2k − I(Ā:R), which holds only for pure system+reference. **[added]** With S2 environment rows, rank(A) − rank⁰(A) remains exact, but the complement identity becomes I(A:R) + I(Ā:R) ≤ 2S(R). Cost O(\|A\|²·W). | y | 2010.09775; 2007.03822; 1903.05124 |
| M6 | Random-erasure curves: nested profile I(R:E_m) for m = 0..n along a random permutation, giving F(m), breakdown point m\*, mean 2^−I; exact binomial conversion to Bernoulli(e) erasure | Erasure threshold as a function of depth | Two incremental bases (all rows, reference-free rows) along the permutation. The whole curve costs O(n·N·W) per permutation. | y | 2010.09775; quant-ph/9701015 |
| M7 | Contiguous code distance d_cont, position-resolved ℓ\*(x), and the BFS-ball analogue on graphs | Code distance vs depth (user study) | Grow intervals from each start and stop at the first I > 0 (exact, since I is monotone by strong subadditivity). Branch and bound against the current best: ≈ O(n·d²·W). Alternative: clipped gauge (S1). | y | 2007.03822; 2105.13352; 2010.09775 |
| M8 | Named observables: half-cut S, profile S(ℓ), I(A:B), I3 on ring quarters (rotation-averaged), CMI I(A:C\|B) | Locating p_c; data collapse | Built on M1. I3 needs 3 prefix passes plus 1 non-contiguous rank per offset: O(n·N·W) per offset. **[added]** Topological entanglement entropy (Kitaev–Preskill or Levin–Wen combinations) for the measurement-only toric-code and Kitaev models is the same kind of signed sum of region ranks. | y | 1911.00008; 1901.08092; 2504.03472 [verified]; 2004.07243 |
| M9 | In-kernel checkpoint scheduler, stop rules (S_R = 0, k = 0, T_max), raw per-trajectory output | Observables must never leave the kernel | Cost tiers:<br>• every layer: O(1) (k, flags) and O(W) (2–4-column ranks, membership)<br>• sparse or log-spaced checkpoints: O(n·d²·W) for d_cont; O(n·N·W) for profiles, I3 and erasure curves<br>Output is fixed-shape [trajectory, checkpoint] arrays with a sentinel after stop.<br>**[added]** An optional bit-packed tableau snapshot at chosen checkpoints, for a chosen subset of trajectories. Each snapshot copies 2(n + n_ref)·W words (plus signs), and the file is written outside the kernel. This feeds the offline-only analyses: exact distance (S4), logical extraction and expurgation (S3). | y | 1910.00020; 1905.05195 |
| M10 | Gate ensembles: uniform C₂ (720 Sp(4,2) matrices × 16 Pauli frames = 11520 elements); weighted index subsets (Z₂-symmetric, CNOT-only, CZ+H, dual-unitary CZ·SWAP / iSWAP); per-edge application probability | The model's unitaries | Fused 4×4 symplectic map: each new column is the XOR of ≤ 4 old columns, O(W). Phase mode adds a 16-entry sign lookup table. **[added]** A uniform n-qubit Clifford sampler, needed for encoders and global scramblers (S17) and for the random-stabilizer-code baselines in §3.2. Use Bravyi–Maslov canonical form or van den Berg's method, O(n³). | y | 2103.02202; 1406.2170; 2003.09412; 2008.06011; 2102.09164 |
| M11 | Measurements: Bernoulli(p) Z (also X, Y, random basis); Pauli products MPP / MXX / MYY / MZZ | MIPT and measurement-only models | 1. Anticommutation mask a = XOR of ≤ 2w columns, O(w·W).<br>2. Deterministic iff a has no stabilizer bit.<br>3. Otherwise choose a pivot p and XOR a∖{p} into each column in p's support: O(wt(p)·W), no per-row loop. | y | quant-ph/0406196; 2004.09560 |
| M12 | Graph generators returning canonical edge arrays: path/cycle, circulant, Watts–Strogatz, Newman–Watts, complete, hypercubic d = 1–3 (open or periodic), random regular, Erdős–Rényi, power-of-two, Cayley tree | The user's geometries | Graph side, once per graph. One `k_per_side` convention throughout. No networkx dependency. Integers bit-stable across NumPy versions. **[added]** A recorded `graph_disorder` mode:<br>• `quenched_per_study`: one graph for all trajectories;<br>• `quenched_per_trajectory`;<br>• `annealed_per_layer`: the graph is redrawn every layer, which is the control that separates geometry from quenched disorder.<br>Optionally, static site-dependent rates p_i. In 1D, quenched disorder is Harris-relevant for Clifford MIPT (ν ≈ 1.3 < 2/d) and gives infinite-randomness criticality (2205.14002), so a frozen WS/NW graph can show Griffiths effects. | n | Watts–Strogatz 1998; cond-mat/9903357; 2109.10944; 2205.14002 |
| M13 | Layer schedules with recorded time-unit metadata: the user's random edges with replacement; brickwork / edge colouring; uniform random perfect matching on Kₙ; random greedy maximal matching; random-sequential (Poisson); power-law P(r) ∝ r^−α | p_c depends on the convention | Generated in-kernel from a CSR edge bank or alias tables, O(1) per draw. **[added]** Measurement schedules, also recorded:<br>• Bernoulli(p) per site per layer (default);<br>• exactly ⌊pn⌋ sites per layer, which removes the fluctuation in measurement count;<br>• per-gate-endpoint, where each qubit a gate touches is measured with probability p right after that gate (common in all-to-all and random-pair models);<br>• spacetime-modulated p(x, t) (2307.15689†). | y | 1901.08092; 2009.11311; 2104.13372; 2110.14403 |
| M14 | Counter-based per-trajectory RNG streams (graph, circuit, outcome, noise, probe), keyed by (seed, cell, p, graph, circuit, trajectory) | Results independent of thread count, sharding and resume | xoshiro / splitmix / Philox in Numba. Own bounded-integer routine (Lemire), because NEP 19 lets NumPy distribution algorithms change between releases. | y | NEP 19; studies/…/raw_tau/PROTOCOL.md |
| M15 | Compile-time instruction classes: structure-changing / sign-only / outcome-dependent structure | Keeps the phase-free kernel exact and branch-free | phases=False drops sign-only operations (Paulis, Pauli noise, rec-controlled Paulis, the X correction inside a reset), and rejects outcome-dependent structure and signed outputs | y | quant-ph/0406196; 2103.02202 |
| M16 | One mixed-state representation, chosen now (see §3.2) | Purification, erasure channels and logical extraction without technical debt | Rmask rows act as the logical block; symplectic logical pairs are extracted on demand in O(n·k²/64) | y | quant-ph/0406196; 1905.05195 |
| M17 | Signed Pauli peek; ⟨P⟩² membership; snapshot/copy; forced-outcome replay returning log₂p(m) = −#random and a consistency flag | String order, Edwards–Anderson correlator, absorbing state, decoders, cross-entropy, Stim parity | Membership: anticommutation vector, O(w·W). Sign: destabilizer pairing, O(n·W). Copy: memcpy into a per-thread scratch tableau, O(n·W). | y | quant-ph/0406196; 2103.02202; 2004.09509; 2209.00609 |
| M18 | Pauli feedback (CX/CY/CZ rec[-k], XCZ, YCZ, sweep[k]); resets R/RX/RY/MR | Stim generality; control in absorbing-state models | Sign flip on the anticommuting rows, O(W). Phase-free: feedback is a no-op and R ≡ M. | y | 2103.02202; 2211.12526; 2212.13823 |
| M19 | Raw, censor-aware, mergeable output plus manifest (conventions, versions, seeds, spec hash); resume | Millions of trajectories; statistics done afterwards | Structured arrays or npz per batch. No aggregation inside the kernel. | n | 1910.00020 |
| M20 | LLM interface: validated service layer, agent CLI, stdio MCP server, agent docs (§3.6) | User requirement | Thin adapters, bounded outputs, detached jobs | n | MCP spec 2025-11-25 |
| M21 | Fair benchmark harness: simulator-neutral circuit description; adapters for Stim, Aer, Cirq (+ optional); entropy cross-check | README runtime figures | Exported tableaux go through the same compiled GF(2) rank | n | 2103.02202 |

**Should**

| ID | Feature | Why | How (cost) | Evol | Citations |
|---|---|---|---|---|---|
| S1 | Clipped gauge: entropies and I(R:[a,b]) for every interval; stabilizer-length distribution P(ℓ) | All-interval tables at checkpoints | Left echelon form, then right endpoints. Cost O(n·N·W), then O(1) per interval via 2D counts. Two cuts on a ring. | y | 1608.06950; 1901.08092 App. A; 2010.09775 Eq. E7† |
| S2 | In-circuit ERASE / DEPHASE / HERALDED_ERASE as exact channels | Pauli noise never changes a single trajectory's entropy | **[corrected]** The earlier one-liner ("reduce x_q, z_q to ≤ 2 pivot rows and relabel them as environment rows") was ambiguous, and the obvious readings of it are wrong. Erasure equals SWAP(q, e) with half of a fresh Bell pair (e, e′) whose environment columns are never stored. On the stored columns this means appending two rows, X_q and Z_q, tagged `env`: the span of every row restricted to sys+R is unchanged if the old rows keep their q-part. Dephasing appends only Z_q.<br>• All pure-state formulas carry over unchanged: M1 (dual form), M3's Rmask invariant (env rows have no reference support), M5's rank(A) − rank⁰(A), and the measurement update.<br>• Cost: O(W) per event, plus row capacity.<br>• Rows can become dependent, so compact by elimination when N exceeds 2(n + n_ref), the dimension of the commutant.<br>• Phase mode: env rows carry no sign.<br>The alternative, deleting the pivot rows (the mixed-state group, N decreases), forces the mixed-state formula S(A) = \|A\| − N + rank(Ā). It loses the A-only shortcut for I(A:R), because the complement columns are then needed. | y | 2202.12905; 2401.01593; 2409.13801; 2510.07512 |
| S3 | Logical extraction on a set E; symplectic pairing; expurgation driver (port of v2) | The user's earlier Gullans et al. code surgery | Eliminate on Ē, taking S pivots first: O(n²·W). Cross-check: r_M ≡ I(R:E). **[added]** Gullans et al.'s expurgation turns a failure-causing logical into either a stabilizer or a *gauge* operator, giving a subsystem code. The driver therefore needs a gauge-group role for rows, and dressed distance and P_rec must be computed with N(S) replaced by the dressed-logical group. | n | 2010.09775; quant-ph/0508131 |
| S4 | Code distance: exact for n ≲ 40; upper bounds (m\*, d_cont, information-set search) | Checking d_cont | Enumeration or Brouwer–Zimmermann for small n; Prange/Leon trials at O(n³/64) each. **[added]** 2603.22532 (Webster–Jacob–Higgott) benchmarks Brouwer–Zimmermann, connected-cluster, SAT and MIP exact methods, plus heuristics including QDistEvol, and ships the `codeDistance` Python package. It is a candidate optional backend and cross-check. | n | 2203.04262; 2308.15140†; 2603.22532 [verified] |
| S5 | Random-measurement threshold of a generated code | The measurement threshold is ≥ the erasure threshold | Measure along a permutation on a scratch copy and record k: O(n²·W) per pass | y | 2402.00145 |
| S6 | Common random numbers across p (U_{t,q} < p) | Lower variance in crossings and slopes | One uniform draw per site per layer, shared across the p grid (the `common_circuit_disorder` policy in PROTOCOL.md) | y | user PROTOCOL.md |
| S7 | Survival and finite-size-scaling toolkit | τ is right-censored; graph disorder | Kaplan–Meier; censored geometric MLE q̂ = D/Σtᵢ; RMST; two-level bootstrap (graphs, then circuits); crossings; collapse | n | 1910.00020; Kaplan–Meier 1958 |
| S8 | Graph diagnostics per realization: degrees, clustering C, path length ℓ, diameter, λ₂, spectral dimension d_s, chromatic index | Plot against geometry instead of β | BFS O(n·m); eigvalsh O(n³), once per graph | n | 2610.12597† |
| S9 | Region families (ring, BFS, random, Fiedler, degree order); I(i:j) vs graph distance; I(R:B_r(c)) | Non-contiguous structure on graphs | Prefix sweeps; 2- and 4-column ranks at O(W) | y | 2202.11750; 2109.10944 |
| S10 | Stabilizer logarithmic negativity E_N = ½·rank K | Open/noisy circuits; multipartite structure | Eliminate the C columns, restrict to A, take the rank of the commutation matrix: O(n³/64) | y | 2012.00031; 2012.00040†; 2202.12905 |
| S11 | Outcome-conditioned Clifford feedback `IF(rec…) CLIFFORD_2Q(i) a b` (v3-only, phases only) | Adaptive circuits | Evaluate the predicate, then apply the gate. Phase-free mode and Stim export raise. | y | 2211.05162; 2309.04520; 2212.13823 |
| S12 | Twin trajectories for linear cross-entropy χ | MIPT probe without postselection | Two phase-mode tableaux in lock-step; B is forced to A's outcomes and returns k₁, k₂ and consistency | y | 2209.00609; 2403.00938 |
| S13 | MIE / teleportation probes on a snapshot | Experimental protocols | Measure the complement on a scratch copy (outcome-independent for Clifford), then 2–3 ranks | y | 2303.04792; 2110.06963 |
| S14 | Absorbing-state order parameter n₁ = (1/n)Σ(1−⟨Zᵢ⟩)/2; absorption-time stop | Separates p_ent from p_ctrl | Batched peek. \|0ⁿ⟩ test: all x columns zero and all signs +. | y | 2211.12526; 2212.14026; 2212.13823 |
| S15 | Optimal Clifford decoder / learnability; record export with spacetime metadata | Decoders, charge learning | Forced replay plus 1–3 peeks; bit-packed records | y | 2206.12429; 2204.10904; 2311.00058 |
| S16 | Stim QEC annotations: DETECTOR, OBSERVABLE_INCLUDE, REPEAT, MPAD, SPP, HERALDED_* | General Stim-like tool | XORs of record bits; SPP is one row-multiplication pass | n | 2103.02202; 2108.10457 |
| S17 | Floquet (frozen) circuits; encoder generators (p = 0 low-depth codes, all-to-all) | Separate temporal from spatial randomness; code baselines | Compile one period and loop over it | y | 2010.09775; 1312.7646; 2608.18536 (title verified); 2609.25201† [added] |
| S18 | Tree builders (collapse/expansion) with reset, discard and fresh ancilla | Exactly solvable comparison points | Same kernels | y | 2210.07264; 2409.13801 |
| S19 | Inverse-tableau exchange with Stim | Validation, cheap Stim entropies | Stim's inverse-tableau rows are v3's forward columns with x↔z swapped (M⁻¹ = ΩMᵀΩ), so no transpose | y | 2103.02202 |
| S20 **[added]** | Participation entropy, i.e. the entropy of the computational-basis measurement record | A second MIPT order parameter. Its subleading term in L is universal (Sierant–Turkeshi) | For a pure stabilizer state, every Rényi S_q of the Z-basis distribution equals r = rank{x_q : q ∈ system} in bits: the support is an affine subspace of dimension r, with flat amplitudes. The marginal on a subsystem A, with N independent rows, is S_part(A) = \|A\| − N + rank(cols(Ā) ∪ {x_q : q ∈ A}). This is ≥ S(A) and equals rank{x_q} when A is the whole pure system. Cost O(n·N·W). | y | 2109.06882 |
| S21 **[added]** | Information-retention observables under channels: I(R:Q), coherent information I_c = S(Q) − S(RQ), the information-loss time (first t with I(R:Q) = 0), and the noise-scaling protocol q = c/L^α | Erasure protection *during* the dynamics. Noise is a relevant perturbation: entanglement ∼ q^−1/3, protection time ∼ q^−1/2 (uncorrelated noise) (2401.01593). Depth-noise coding transitions sit at depth ∼ 1/p (2510.07512). | All are rank combinations on the S2 env-row tableau: I(R:Q) = rank(Q) − rank⁰(Q); S(RQ) = rank(Q ∪ R) − n − n_ref. Cost O(n·N·W) per checkpoint. | y | 2401.01593; 2401.16631; 2408.16267; 2510.07512; 2304.02664; 2512.18783 |
| S22 **[added]** | Multi-reference and two-reference correlators: I(R₁:R₂) and the joint purification of references at separation r | Spatial structure of the generated code; logical correlations on graphs | 4-column ranks on reference columns, O(W) | y | 1910.00020; 2010.09775 |

**Could**: **[added]** ZX-calculus / circuit-percolation diagnostics for CNOT+SWAP+Bell-measurement brickworks (2502.13211); GHZ extraction / multipartite entanglement statistics (quant-ph/0504208, 2407.03206†); min-cut / percolation bound S_A ≤ mincut(A) (1808.05953); Pauli-frame detector sampler (otherwise delegate to Stim); spacetime-duality maps and postselected projectors (2010.15840, 2103.06873); Majorana helpers via Jordan–Wigner to MPP/SPP (1911.11169); Heisenberg Pauli propagation and Hayden–Preskill decodability (0708.4025, 1710.03363); cluster / large-N Brownian emulation (2104.07688, 2109.00013); logical-weight statistics (2212.10634).

Virtual qubit permutation (PERMUTE/SHIFT) is also a Could, but it is a design-now-or-never decision. It only works if every qubit index goes through an index table from the start (2207.12415).

---

## 3. Areas

### 3.1 MIPT and purification observables

**Three pictures** (1905.05195, 1910.00020):

1. **Pure state** from |0ⁿ⟩. Observables: S(A), profiles, I(A:B), I3.
2. **Purification** from the maximally mixed state (n Bell pairs) or from k₀ = αn pairs. Here S(ρ_sys) = k(t).
3. **Single reference.** S_R ∈ {0, 1}, so ⟨S_R(t)⟩ = P(τ > t).

Stabilizer spectra are flat, so every Rényi entropy equals S(A).

**Phase-free mode is exact for all of these.** Every observable here depends only on the unsigned stabilizer group. Without outcome-dependent feedback, the unsigned group evolves independently of outcome values. Forced and Born-rule measurements therefore give identical entanglement statistics for Clifford circuits. This is reasoning from the sweeps; the forced-measurement-transition (FMPT) distinction of 2009.11311 matters only for non-Clifford gates. A unit test should confirm it.

The user's decision is that **phases=True stays the default**. To support that choice without losing speed:

- `capabilities()`, the docs and every run manifest should state that phases=False is exact for these observables, with the measured speedup, so users and agents opt in knowingly.
- Pauli gates and Pauli noise are exact no-ops in phase-free mode (M15), so noisy or feedback circuits can still be used for entropy studies.

**Reference bookkeeping (M3–M5).** The reference-free-first pivot rule maintains one invariant: the rows with reference support have linearly independent reference parts. From it:

- k = popcount(Rmask) − n_ref;
- I(A:R) = rank(A) − rank⁰(A).

Only A's own columns are needed, once unmasked and once ANDed with ¬Rmask. This identity was derived in the sweeps, not taken from a paper. It must be unit-tested against S_A + S_R − S_AR computed by brute force.

Two consequences:

- Reference columns are needed only for S_R = rank{x_R, z_R} − 1 and for Stim lock-step. Keep them materialised by default, since gates never touch them. Dropping them is a later optimisation, allowed once the tests pass.
- S_R can change only on a layer with ≥ 1 measurement. Check it only then, and stop the trajectory on the event.

**I3** (1911.00008; Gullans–Huse): split the periodic chain into quarters A, B, C, D.

I3 = S_A + S_B + S_C − S_AB − S_BC − S_AC + S_ABC,

with S_ABC = S_D and S_AC = S_BD for pure states. I3 → 0 in the area law, ≈ −0.5 bits at p_c†, and ∝ −L in the volume law. Average over 4–16 rotations.

**Other observables:**

- Antipodal mutual information I(A:B) (1901.08092; interval size L/8†).
- Entanglement profile: S ≈ α·log of the chord length (L/π)·sin(πℓ/L) at p_c.
- Growth from a product state: linear for p < p_c, logarithmic at p_c, with z = 1.
- Stabilizer-length distribution P(ℓ): reported as a power law at criticality (ℓ⁻²†). Sweeps disagree about its form in the volume law (bimodal vs power law†).

**Reference values.** These depend on convention.

- 1D brickwork Clifford with measurement after each gate sublayer: p_c ≈ 0.16, quoted as 0.1593(5) with ν = 1.28(2)†. **[corrected]** The 1910.00020 abstract does not contain these numbers, and the review pass did not find their source. Locate it before quoting.
- Haar: p_c = 0.17(1) from I3 (1911.00008; in the published abstract). **[corrected]** ν ≈ 1.3 appears only in the APS March Meeting 2020 abstract, not in the paper's abstract. The paper says its bulk exponents agree with percolation and stabilizer values.
- **[added]** Gate-set dependence: a brickwork of *dual-unitary* Clifford gates gives ρ_c = 0.2048(5), ν = 1.30(5) (2307.15689†; ρ is a measurement density, so it is not directly comparable to p).
- Clifford and Haar transitions lie in different universality classes (2107.03393).
- Steady state is reached after 2L–4L layers.
- In the mixed phase, τ and the purification time grow as exp(cL). Fidkowski–Haah–Hastings show that the maximally mixed state does purify, but only on a timescale set by the Hilbert-space dimension, i.e. exponential in system size. So "mixed phase" is a statement about times poly(n), and k(t) must be recorded on a depth grid, with censoring.
- **[added]** Expected τ scaling for the single reference, for choosing T_max and checking censoring:
  - p > p_c (pure phase): τ = O(1), an L-independent distribution, only for a reference entangled *locally* into an unscrambled (or steady) state. After a global scramble (all-to-all random gates or matchings, or ~2L gate-only layers, as in the random-edges protocol below, `examples/purification_time.py` and the README figure) the reference's information is spread over the whole system first, and τ grows as O(log L). **[corrected]** Measured with SparseGF2 (ring, p = 0.3, random partner, median τ for L = 8, 16, 32, 64): 3, 3, 3, 3 without a scramble; 6, 9, 13, 16 after 2L scramble layers. Seeing τ grow logarithmically after a scramble is therefore not a sign of being at or below p_c;
  - p = p_c: τ ∝ L^z with z = 1, so S_R(t) collapses against t/L;
  - p < p_c: exp(cL).
  
  Purifying the whole maximally mixed state takes O(log L) in the pure phase†, because O(L) independent local events must all occur (1905.05195, 1910.00020).
- **[added]** Foundational stat-mech mapping, which explains why Clifford and Haar differ: Bao–Choi–Altman (1908.04305) and Jian–You–Vasseur–Ludwig (1908.08051). Reviews: Potter–Vasseur (2111.08018) and Fisher–Khemani–Nahum–Vijay (2207.14280). The 2502.13211 ZX-calculus analysis argues that the Clifford transition in a CNOT/SWAP/Bell-measurement brickwork is a disguised classical percolation transition, so the gate set matters for universality claims.

**The random-edges protocol** (as in `examples/purification_time.py`):

- Each layer draws m = ⌊n/2⌋ edges independently and uniformly *with replacement*, applies a random C₂ to each, and then measures Z on every qubit with probability p.
- S_R is checked after the full layer.
- τ = −1 if censored at T_max = q_max·n. The scramble is q_scramble·n gates.

The gate-to-measurement ratio matches brickwork-with-measurement-per-sublayer, but the layers are not matchings. On any d-regular graph the chance that a given qubit is idle in a layer is (1 − 2/n)^{n/2} → e⁻¹ ≈ 0.37 (my computation), and some qubits receive two or more gates. So p_c is not comparable to 0.159, and every result must record the layer definition.

### 3.2 Codes and erasure protection

**Setup.** Start maximally mixed, or with k reference Bell pairs. The surviving entropy defines a stabilizer code:

- S has n − k generators;
- the normalizer N(S) = ⟨S, logicals⟩ has dimension n + k;
- in the reference picture, the system parts of the Rmask rows together with S span N(S).

Choi et al. (1903.05124) and Fan et al. (2002.12385) view the volume-law phase as a code whose errors are the measurements. Fan et al. obtain the Hamming bound p_c ≤ 0.1893 for qubits. **[corrected]** The bound assumes the volume-law state encodes a Page state in a code; it is a heuristic bound, not a theorem. It comes from p_c·log[(d²−1)(p_c⁻¹−1)] ≤ log[(1−p_c)d], which is saturated at p_c = ½ as d → ∞.

**Exact formulas** (A, E ⊆ system; ranks are over the rows indicated):

```
I(R:E) = S(R) + S(E) − S(RE)                                   (pure system+reference)
       = dim N(S)_E − dim S_E = rank_{N(S)}(E) − rank_S(E)       (Gullans et al. r_M = rank M − rank M_S)
       = k + 2|E| − n + rank_S(Ē) − rank_S(E)                    (stabilizer rows only)
       = rank(E) − rank⁰(E)                                     (Rmask form, derived; test it)
correctable  ⇔  I(R:E) = 0  ⇔  no nontrivial logical supported on E   (Knill–Laflamme / cleaning lemma)
P_rec(E) = |S_E| / |N(S)_E| = 2^{−I(R:E)}                       (optimal decoder; 2010.09775 Eqs. 16–18†)
I(R:E) + I(R:Ē) = 2k ;   0 ≤ I(R:E) ≤ 2·min(|E|, k) ;   I is non-decreasing in E
```

I(R:E) can be odd (Bravyi–Fattal–Gottesman GHZ extraction, quant-ph/0504208). **[added]** The elementary case: E supports a logical X̄ but no logical anticommuting with it, so N(S)_E = ⟨S_E, X̄⟩ and I = 1. Use I(R:E)/2 as "exposed logical qubits" only as a convention, and record it as such.

**[added]** P_rec is the success probability of the optimal decoder averaged over a uniformly random Pauli on E, which is equivalent to replacing E by I/2^{\|E\|}. Equivalently, it is the entanglement fidelity of the best recovery. An erasure E is either correctable (P_rec = 1) or loses at least one logical bit (P_rec ≤ ½).

**Erasure models.** Erasure sets can be Bernoulli(e) per qubit, fixed weight w, contiguous, graph balls, or hub-targeted.

The capacity bound is Q = max(0, 1 − 2e) (quant-ph/9701015). Vanishing failure therefore requires e < (1 − k/n)/2. Random stabilizer codes satisfy F(m) ≲ 2^{2m+k−n}. **[verified by derivation]** This is a union bound: about 4^m Paulis act on E, and each is a nontrivial logical with probability ≈ 2^{−(n−k)}.

The nested profile (M6) computes I(R:E_m) along one random permutation. Each E_m is marginally a uniform m-subset, so

P_fail(e) = Σ_m C(n,m)·e^m·(1−e)^{n−m}·F(m)

holds exactly. One pass therefore gives the whole curve. Error bars come from bootstrapping over permutations and trajectories, since different m on one permutation are correlated.

**Contiguous distance.**

- Definition: d_cont = min{|A| : A an interval (arc on a ring), I(A:R) > 0}. It is undefined when k = 0; store a sentinel.
- Also store ℓ\*(x), the first length from start x with I > 0, and its mean and distribution. Some papers average over positions instead of taking the minimum.
- Ordering: d ≤ d_cont ≤ ⟨ℓ\*(x)⟩.
- Scaling: d_cont diverges sub-extensively in the mixed phase (Li–Fisher 2007.03822). There it is the crossover length at which domain-wall energy and entropy become comparable. **[corrected]** The earlier text attributed "fits slightly above 1/3" to 2107.12376, which is Yang–Li–Fisher–Chen on random stabilizer *tensor networks*. The fitted exponent ≈ 0.38 is Li–Fisher's own Clifford fit (2007.03822), which they say differs from their capillary-wave prediction. The directed-polymer / KPZ picture gives ~L^{1/3} (Li–Vijay–Fisher 2105.13352). Sang–Li–Hsieh–Yoshida (2212.10634) derive ≈ 1/3 from the tail of the stabilizer-length distribution, and report d_cont growing ~√t before it saturates†. Han–Chen (2207.02165) find sublinear power-law d_cont in automaton circuits. Distance is nearly extensive in maximally nonlocal sparse circuits (2109.10944 [verified]). It is set by the spectral dimension on sparse random graphs (2610.12597†; **not found in any search index on 2026-10-13, so do not rely on it**).
- On graphs, replace intervals by BFS balls with deterministic tie-breaking. On complete graphs use only random-erasure statistics.

**Clipped gauge (S1).** First put the generators in left-endpoint echelon form, then reduce right endpoints. For any interval, dim S_[a,b] = #{g : a ≤ l_g, r_g ≤ b}. Therefore

I(R:[a,b]) = #N(S)-generators inside [a,b] − #S-generators inside [a,b],

and every interval is O(1) after an O(n·N·W) gauge.

**Codes from low-depth circuits** (Gullans et al., 2010.09775):

- **[corrected]** For any fixed rate below capacity, depth O(log N) is necessary and sufficient for vanishing erasure failure probability in *every* D ≥ 1. The abstract says "strong evidence", so this is numerical, not proven. Only in the *moderate-deviation* limit, where the gap to capacity shrinks with N, does the depth stay O(log N) for D ≥ 2 but become O(√N) in D = 1. The earlier text stated the D = 1 penalty as if it applied at fixed rate.
- Expurgation measures failure-causing logical operators, turning them into stabilizers (k → k − 1) or gauge operators (a subsystem code), without increasing the maximum check weight. When a logical becomes a stabilizer, distance and recovery cannot decrease, since N(S′)∖S′ ⊆ N(S)∖S **[verified by derivation]**. Expurgated 2D codes reach depth 4–8 with thousands of logical qubits.
- Related: Brown–Fawzi (O(log³ n) depth all-to-all, 1312.7646). 2608.18536, "Good Stabilizer Codes from Shallow Clifford Circuits with Random Matchings": title verified; the author list and the O(log n)/quantum-GV claim are not verified†. Nelson et al. (2311.17985) [verified]: 1D random-circuit codes with fault-tolerant distillation and noisy gates and measurements correct erasures up to ≈ 2%, and their depolarizing threshold tracks the hashing bound.
- **[added]** Noise *during* encoding: Nelson–Rajakumar–Gullans (2510.07512) find a sharp transition at critical depth ~1/p for 2D random Clifford encoders with depolarizing noise after every gate. They compute it from the average coherent information with a stabilizer simulator, which is exactly what S2 + S21 must reproduce. Related coding transitions: boundary dissipation (Lovas–Agrawal–Vijay 2304.02664), coherent-information transitions in noisy monitored circuits (2408.16267), and the review 2512.18783. Kelly et al. (2210.11547) show that coherence bounds the code distance of hybrid-circuit stabilizer codes.
- Exact distance is NP-hard (2203.04262), so offer exact methods for small n and proxies (m\*, d_cont) at scale.
- Lee–Yoshida (2402.00145) [verified]: for stabilizer and subsystem codes, the random-measurement threshold is ≥ the erasure threshold (S5).

**Noise inside the circuit (S2).** A Pauli error at a known location changes only signs, so Pauli unravelings never change any trajectory entropy. Noise that is meant to degrade entanglement must use the exact channel unraveling:

- Z_ERROR(q) ≡ complete dephasing with probability 2q;
- DEPOLARIZE1(p) ≡ erasure (qubit replaced by I/2) with probability 4p/3;
- DEPOLARIZE2(p) ≡ two-qubit erasure with probability 16p/15.

Stim's HERALDED_ERASE(p) records 1 and applies X_ERROR(½) + Z_ERROR(½); support it verbatim, plus `ERASE(p)` with channel semantics.

**[added]** In the exact unraveling, an erasure appends env rows X_q, Z_q and a dephasing appends Z_q (see S2). The coefficients above are exact: (1−p)ρ + (p/3)ΣPρP = (1−4p/3)ρ + (4p/3)·I/2, and similarly 16p/15 for two qubits. Under these channels:

- S_R and popcount(Rmask) − n_ref stop measuring protection (M3, M4);
- report I(R:Q) and the coherent information instead (S21);
- the relevant scaling variable is q·L or q·L^α (2401.16631).

**Representation decision (M16).** The sweeps proposed two encodings:

- explicit references plus Rmask;
- a QuantumClifford-style full symplectic basis: r stabilizers, r destabilizers and k logical pairs (MixedDestabilizer).

Under the pivot rule these carry the same information: the Rmask rows play the role of the logical block. Recommendation:

- Use Rmask in the kernel, with row roles (system stabilizer / reference / environment) so that erasure channels can label environment rows. **[added]** Environment rows are *appended*, not relabelled (S2), so the tableau needs spare row capacity, up to 2(n + n_ref) rows plus a compaction pass. Expurgation (S3) also needs a fourth role, gauge.
- Derive paired logicals on demand by symplectic Gram–Schmidt, for expurgation.
- Do not maintain two representations.

### 3.3 Graph-defined and long-range circuits

**Generators (M12).**

- Watts–Strogatz rewires each circulant edge with probability β. The edge count is preserved; the degree sequence is not.
- Newman–Watts adds shortcuts and never removes edges.
- v2's `watts_strogatz(k)` meant neighbours per side, while `newman_watts(k)` meant total degree (as in networkx). Unify this as `k_per_side`.
- Also provide random d-regular (pairing model with rejection), Erdős–Rényi (geometric skipping), the power-of-two graph i ~ i ± 2ʲ (1805.08215†, 2109.10944), Cayley trees, and clustered graphs.
- Emit graph6 and a connectivity flag for every graph.

**Layer schedules (M13).** Each result must record: gates per qubit per time unit, measurements per qubit per time unit, sublayers per unit, mode parameters, graph hash and seed. Critical p is comparable between papers only after normalising to measurements per gate per qubit.

1. **Random edges with replacement** (the random-edges protocol): ⌊n/2⌋ draws per layer.
2. **Edge colouring / brickwork.** Bipartite graphs need Δ colours (König's theorem): 2 for even rings, 4 for square lattices, 6 for cubic lattices. General graphs need Δ or Δ+1. **[corrected]** That bound is Vizing's theorem (1964); Misra–Gries is the polynomial algorithm that achieves Δ+1 colours. Record whether "one step" means a sublayer or a full sweep, since that changes p_c by about the number of sublayers.
3. **Uniform random perfect matching on Kₙ** via a random permutation (2608.18536†; standard all-to-all†).
4. **Random greedy maximal matching** on irregular graphs. This is *not* uniform: counting perfect matchings is #P-complete (Valiant). v2 mislabelled this sampler as uniform. (Approximately uniform samplers exist for bipartite graphs, via the Jerrum–Sinclair–Vigoda MCMC, but they are far too slow per layer.)
5. **Random-sequential.** Gates on uniform random edges with replacement, n/2 gates per time unit, measurements at rate p. Optionally Gillespie. This emulates all-to-all random-pair and Brownian models (2009.11311†, 2104.07688).
6. **Power-law pairs**, P(r) ∝ r^−α for 1 ≤ r ≤ L/2, via an alias table. Modes are random-site, per-site, or weighted matching, and must be recorded†. Record `alpha_convention` as either probability or coupling amplitude. Brownian papers state the coupling exponent, so α_prob = 2α_coupling (my inference, consistent with α_c ≈ 3 vs 3/2).
   - Block et al. (2104.13372) find conformal, short-range-like criticality above a critical exponent and a continuum of non-conformal classes below it. **[corrected]** The value α ≈ 3 comes from the group's APS meeting abstract; the arXiv abstract gives no number.
   - Otsuka–Seki–Yunoki (2504.03472 [verified], PRB 112, 054301 (2025)), using the QCMI: no transition for α ≲ 2; for 2 ≲ α ≲ 3, transitions with exponents different from nearest-neighbour; the log coefficient for the shortest range is c̃ = 1.519(3). The L range (up to 2048) is unverified†. Sharma et al. (2110.14403) is also cited for the absence of a transition at small α†.
   - Minato et al. (2104.09118) prove sufficient conditions α > d/2 + 1 for free fermions and α > d + 1 for generic systems.
   - α = ∞ must reproduce the nearest-neighbour chain.

**Physics notes.**

- On complete and small-world graphs a random bipartition is crossed by about half of all gates. Bipartite entropy saturates in O(1)–O(log n) layers and carries little information. Use purification, S_R, erasure curves and I3 instead (2009.11311, 1905.05195).
- The sweeps found no primary paper on MIPT on Watts–Strogatz or Newman–Watts graphs (2020–Oct 2026). The review pass ran an extended search and still found none. The nearest work is on long-range circuits, sparse nonlocal circuits (2109.10944, 2202.11750), the graph-theoretic view of typical outputs (2608.03102, Yu-Xuan Zhang and Yu-Xiang Zhang [verified]), and sparse random graphs (2610.12597†, unconfirmed).
- Graph-to-graph fluctuations can dominate near p_c, so error bars need a two-level bootstrap (S7).
- **[added]** A frozen graph is *quenched* disorder. In 1D, static randomness in measurement rates is Harris-relevant and produces an infinite-randomness critical point: S ~ √ℓ, z = ∞, with Griffiths phases on either side (Zabalo et al. 2205.14002, PRB 107, L220204 (2023)). Whether frozen WS/NW shortcuts at small β act the same way is open; they are nonlocal, unlike rate disorder. So compare quenched-per-trajectory against annealed-per-layer graphs (M12) before attributing an effect to small-world geometry. Classically diffusing measurers likewise change the universality class (Ha–Pandey–Gopalakrishnan–Huse 2405.08861).
- 2D/3D lattices: Clifford transitions lie near, but not exactly at, (d+1)-dimensional percolation (2007.02970†, 2012.03857, 2210.11957).
- Trees: Feng–Skinner–Nahum (2210.07264) and Sommers–Huse–Gullans (concatenated codes with an exact heralded-erasure recursion, 2409.13801).

### 3.4 Other protocols (Stim-like generality)

| Family | Examples | Needs | Phases? |
|---|---|---|---|
| Measurement-only | Projective TFIM ZZ vs X (bond percolation at ½); cluster checks; Kitaev XX/YY/ZZ; toric-code stars/plaquettes; XZZX | MPP/MXX/MYY/MZZ inside the ensemble kernel; S_topo; k = dynamically generated logical qubits | no |
| Feedback / absorbing state | measure Z and correct X; control maps; adaptive gates; quantum East model | rec-controlled Paulis, resets, IF-Clifford, signed ⟨Zᵢ⟩, absorption stop | yes |
| Symmetric / structured | Z₂-symmetric, CNOT-only, CZ+H (kicked Ising), dual-unitary | weighted subsets of the 11520-element table. U(1)-symmetric Cliffords only map Zᵢ → +Z_{π(i)}, since ΣZ must be preserved. **[corrected]** 2405.08861 (Ha–Pandey–Gopalakrishnan–Huse, PRB 110, L140301 (2024)) states that a U(1)-conserving restriction of random Clifford circuits eliminates the volume-law phase, so charge sharpening is not Clifford-native. Their workaround, Clifford circuits monitored by classically diffusing "measurers", *is* Clifford-native and is worth supporting. | no |
| Noisy / open | dephasing or erasure at rate q; L^{1/3} negativity | exact channels, mixed states, negativity | no |
| Learnability / XEB / decoding | cross-entropy χ, charge learning, neural-network decoders | separate RNG streams, forced replay, twin runs, signed peeks, record export | yes |
| MIE / teleportation | Google 2D experiment (finite-time teleportation, T_c ≈ 4†) | snapshot, then measure the complement | no |
| Noisy monitored **[added]** | depolarizing / dephasing / erasure at rate q, with q = c/L^α; noisy encoders at depth ~1/p | S2 env-row channels; I(R:Q) and coherent information (S21) | no |
| Floquet monitored **[added]** | frozen Clifford period plus measurements; logical-space localization (2609.25201†) | S17 | no |
| QEC / Floquet memory | surface, colour, repetition, honeycomb | DETECTOR, OBSERVABLE_INCLUDE, REPEAT, SPP, HERALDED_*; mid-circuit code properties | yes |
| Spacetime duality | dual brickworks | postselected projectors (Could) | yes |

Verified against local Stim 1.16:

- Only CX, CY, CZ, XCZ and YCZ accept rec targets; H, SWAP, ISWAP and CXSWAP reject them. IF-Clifford is therefore v3-only and must raise on export.
- TableauSimulator offers copy, peek_observable_expectation, postselect_* and measure_kickback. These are the parity targets for M17.

### 3.5 Simulator landscape and fair benchmarking

**Algorithmic baseline.** Stim stores the inverse tableau. Its rows are v3's forward columns with x↔z swapped, so the two share a layout.

**[corrected]** This list previously wrote W for the SIMD width, which clashes with W = ⌈N/64⌉ in the notation table. Here w is the SIMD register width in bits.

- Stim gates cost O(n/w).
- A deterministic Z measurement costs O(n/w), because Stim stores inverse-frame signs.
- A random measurement costs O(n²/w) and goes through a transposed tableau.
- PyPI wheels always import `stim._stim_sse2` (w = 128), because AVX2 is disabled pending Stim issue #432. State this in the captions.

**Where v3 can legitimately win:**

- a fused 4×4 gate (Stim needs 3.6 instructions on average with the best decomposition, 9.3 via `to_circuit`);
- phase-free mixed states;
- S_R in O(W) and k in O(1);
- circuits generated and observables computed inside the kernel;
- trajectory-parallel `prange`.

Packing 64 trajectories into one word does *not* work here: each trajectory picks a different pivot on random measurements.

| Simulator | Use | Fastest fair driving |
|---|---|---|
| **Stim 1.16** TableauSimulator | headline | Build one `stim.Circuit` per layer from text templates (~50× faster than append loops), using the best 2-qubit decomposition (≤ 5 instructions). Batch M. For S_R use `peek_bloch(R)` ('I' ⇔ entangled; 0.74 µs at n = 1024). For region entropies use `current_inverse_tableau().to_numpy(bit_packed=True)` with v3's compiled rank: S_A = rank([x2x[A]; z2x[A]]) − \|A\|. For multiple cores, a `multiprocessing.Pool` in which each worker builds its own circuits. Avoid: per-gate `do_tableau`, separate `measure(q)` calls (10–20× slower), `inverse()` or `canonical_stabilizers()` at checkpoints. |
| **Qiskit Aer 0.17** method="stabilizer" | headline | Column-major forward tableau with signs. Report engine time (`time_taken`) and wall time; Python conversion costs ~10–15 µs per instruction. A measure or save instruction is *required*, otherwise Aer skips the circuit. For multiple cores set `max_parallel_experiments`. |
| **Cirq 1.7** (CH form) | n ≤ 64 | `CliffordSimulator`, 3.5× faster than `act_on` |
| QuantumClifford.jl via juliacall | optional | MixedDestabilizer, `projectZ!(…; phases=false)`, `fastcolumn` (API untested) |
| PyClifford (Numba, GitHub only) | optional | closest tech stack; untested |
| *Excluded from the headline* | — | qiskit.quantum_info (≈3 ms/op); PennyLane default.clifford and tensorcircuit-ng (both wrap Stim); GPU or frame shot samplers (QuaSARQ, cuStabilizer, STABSim, SymPhase), which sample one fixed circuit and have no state; graph-state simulators (O(d²) with d = Θ(n) in the volume law); qecsim |

**Measured locally** (4-core Xeon, 2.1 GHz; Python 3.13; one trajectory, best of 3):

| Case | Result |
|---|---|
| n = 32, depth 32, p = 0.16 | Stim 0.44 ms; Aer 96 ms wall; Cirq CH 365 ms; Cirq tableau 1.28 s; qiskit.quantum_info 1.69 s. All agree on S_half. |
| Stim, depth = n, p = 0.16, `to_circuit` decomposition | 1.65 ms (n = 64), 9.5 ms (128), 60.6 ms (256), 466 ms (512), 4.72 s (1024) |
| Stim, best decomposition | p = 0: 171–225 ns per 2-qubit Clifford. p = 0.16, n = 512: 455 ms. |
| Stim random measurement | batched: 4.4 / 9.5 / 37 µs at n = 256 / 512 / 1024. One `measure()` call each: 48 / 132 / 781 µs. |
| Deterministic fraction of measurements in steady state | 0.5–1.4% (p = 0.05), 6% (0.16), 14% (0.3), 20% (0.5). Forward-frame signs are therefore acceptable. |

**Protocol (M21).**

- Every simulator receives the same seeded description, layers = [([(a, b, cliff_idx)], [measured qubits])].
- Assert identical entropies for every benchmarked trajectory. Clifford entropies depend only on gates and measurement locations.
- Metric: trajectories per second, reported engine-only and end-to-end, on 1 core and on all cores.
- JIT compile time reported separately. One warm-up run, then median and IQR of ≥ 5 repeats.
- Grid: n ∈ {16, …, 1024}, depth = n (and 4n for τ), p ∈ {0, 0.16, 0.30}.
- Observable workloads: none; S_half at the end; S_R every layer; profile every n/8 layers.
- Label v3 runs phase-free vs phase-tracking. Stim always tracks signs, so phase-tracking is the like-for-like comparison.
- Mixed-state protocols need n + 1 or 2n qubits in Stim and Aer; say so in the caption.

README figure plan:

- A: circuits/s vs n.
- B: ns per gate and µs per measurement.
- C: τ workload and profile-checkpoint workload.
- D: speedup vs n for each p, plus one small-world geometry.

### 3.6 LLM interface (CLI + MCP)

- **Service layer.** One validated layer, `sparsegf2.api`, built on frozen Pydantic specs (EnsembleSpec, SweepSpec, CircuitRunSpec) with `extra='forbid'` and a description on every field. JSON Schema comes from the models, so the CLI, MCP server and docs cannot drift apart.
- **CLI `sparsegf2`.** Built with argparse.
  - Subcommands: info, schema, example, validate, run, ensemble, sweep (`--resume`, `--shard i/N`), status, analyze, plot, bench, selftest, mcp, mcp-config.
  - `--json` prints one envelope on stdout: {ok, command, version, result, artifacts, warnings} or {ok: false, error: {type, message, field, hint}}.
  - Exit codes: 0 ok, 2 usage, 3 runtime, 4 unsupported instruction, 5 budget, 130 interrupted.
  - Logs go to stderr. No prompts. `--help` opens with examples.
- **MCP server.** `mcp>=2.2,<3`. In 2.x, `FastMCP` was renamed `MCPServer`, imported from `mcp.server.mcpserver`; this was verified on 2.2.0 and 2.3.0.
  - Transport: stdio.
  - About 10 tools: capabilities, inspect_circuit, run_circuit, run_ensemble (has `dry_run`; synchronous budget ≤ 90 s), start_sweep / job_status / cancel_job (detached CLI subprocess with status.json heartbeat), analyze_results, plot_results, benchmark.
  - Resources: `sparsegf2://docs/{topic}`.
  - Prompts: MIPT scan, purification time, code vs depth, erasure study.
- **Contracts.**
  - Outputs are Pydantic models, ≤ ~8k characters by default with a hard cap of 40k. Larger data goes to absolute file paths.
  - Errors raise `ToolError("field: message. Hint: …")`. Never raise `MCPError`, which becomes a JSON-RPC error the model may not see.
  - Library code never writes to stdout.
  - Report progress at least every 10 s; progress resets the 30-minute stdio idle timeout.
- **Claude Code limits that set these numbers:** warning at 10k tokens and cap at 25k tokens per tool output; text over 50k characters is moved to a file; descriptions are truncated at 2,048 characters; calls auto-background after 2 minutes.
- **Docs.**
  - AGENTS.md is canonical; CLAUDE.md contains `@AGENTS.md`.
  - llms.txt.
  - docs/OBSERVABLES.md with the exact formulas above.
  - docs/RECIPES.md showing each task as a CLI call, Python code and an MCP call.
  - CLI.md and MCP.md generated from the registry, with a freshness test.
  - `.claude/skills/sparsegf2/SKILL.md` and `.mcp.json`.
- **Tests.**
  - In-process `mcp.Client(server, raise_exceptions=True)` tests for every tool and every error path.
  - A stdio subprocess smoke test, which catches stdout pollution.
  - Schema lints: every parameter described; no root-level anyOf; descriptions ≤ 2,048 characters.
  - CLI/MCP parity: byte-identical summary.json for the same spec and seed.
  - Thread-count determinism.

### 3.7 Review log (added 2026-10-13)

**Corrections**

1. Gullans et al. depth scaling (§3.2). O(log N) suffices in every D at a fixed rate below capacity. The D = 1 vs D ≥ 2 split (O(√N) vs O(log N)) applies only in the moderate-deviation limit.
2. Contiguous-distance exponent (§3.2). The "slightly above 1/3" fit is Li–Fisher's (2007.03822, ≈ 0.38). 2107.12376 is a tensor-network paper and was miscited. 2212.10634 was added for the ≈ 1/3 derivation.
3. S2 erasure implementation. "Relabel pivot rows as environment" was ambiguous: a relabelled pivot row would still constrain qubit q, and deleting it breaks the A-only I(A:R) shortcut. It is replaced by appended env rows X_q, Z_q, which is exact and keeps every pure-state rank formula. Consequences were added to M3, M4, M5 and M16.
4. M3. popcount(Rmask) − n_ref equals S(R). It is the code dimension only when system+reference is pure.
5. M5. The complement identity I(A:R) = 2k − I(Ā:R) needs purity. Under channels it becomes ≤ 2S(R), by weak monotonicity.
6. Zabalo et al. ν ≈ 1.3 comes from the APS meeting abstract, not the paper abstract. The source of p_c = 0.1593(5), ν = 1.28(2) is still unlocated.
7. Block et al. α ≈ 3 comes from the meeting abstract.
8. The Fan et al. bound is assumption-dependent (Page-state encoding).
9. The edge-colouring bound is Vizing's theorem, not Misra–Gries.
10. The SIMD width w was renamed so it no longer clashes with W.
11. 2405.08861 is now described with its actual content and authors.

**Verified from abstracts (daggers removed)**

2504.03472 (Otsuka–Seki–Yunoki); 2311.17985 (≈ 2% erasure); 2109.10944 (nearly extensive d_cont); 2402.00145 (measurement threshold ≥ erasure threshold); 2409.13801 (title, PRR 7, 023086); 2603.22532 (title, authors, `codeDistance` package); 2608.03102 (authors); 2608.18536 (title only); 2607.06683 (title exists); 2007.03822 (journal reference); 2107.12376 (PRB 105, 104306 is correct; the page number really does coincide with Li–Fisher's).

**Identified:** 2510.07512 is Nelson–Rajakumar–Gullans, "Error correction phase transition in noisy random quantum circuits".

**Unidentified (do not cite until checked)**

- 2505.22720: no search hit.
- 2603.10751: no search hit. The only Gerbino paper found is "A Dyson Brownian motion model for weak measurements in chaotic quantum systems", Quantum Rep. 6, 200 (2024), which is a different work.
- 2610.12597: no search hit.
- 2308.13384: no search hit.

**Added features:** S20 (participation entropy); S21 (information retention under channels); S22 (multi-reference correlators); graph-disorder modes (M12); measurement schedules (M13); tableau snapshots (M9); uniform n-qubit Clifford sampler (M10); gauge role for expurgation (S3, M16); topological entanglement entropy (M8); expected τ scaling (§3.1); the quenched-disorder caveat (§3.3).

**Added papers:** 1908.04305, 1908.08051, 2111.08018, 2205.14002, 2109.06882, 2307.15689†, 2502.13211, 2401.16631, 2408.16267, 2510.07512, 2512.18783, 2304.02664, 2210.11547, 2609.25201†, 2008.06011, 2407.03206†.

---

## 4. Typical parameter ranges

| Setting | Sizes | Depth / time | Samples |
|---|---|---|---|
| 1D brickwork Clifford MIPT | L = 8–512 standard; 2048 in 2504.03472†; up to 32768 qubits across d = 1–3 (2210.11957†) | 2L–4L to steady state, then a time window | 10²–10⁴ circuits per (L, p) for entropies and I3, plus spatial and temporal averaging |
| Single-reference τ / survival | user: n = 64–1024 | T_max = q_max·n (an earlier version censored at 8n) | 10⁴–10⁶ trajectories per (L, p) |
| Purification k(t)/n | direct exp(cL) τ only for L ≲ 20–30 | plateaus at t = poly(L): L, 2L, 4L, log grid | 10³–10⁵ |
| p grids | — | — | Δp = 0.005–0.01 near p_c; 0.02–0.05 over [0, 0.5] |
| Contiguous distance | L ≲ 512 | log-spaced checkpoints | 10²–10⁴ circuits, all starts |
| Erasure / low-depth codes | N in the thousands (2D expurgated) | depth O(log N); 4–8 expurgated (2D) | 10²–10⁴ erasure patterns per state |
| 2D / 3D | 2D L ≤ 48 (N = 2304, 2007.02970†); 3D L ~ 10–20† | ~2L | 10³–10⁴ |
| Long-range | L = 16–2048; α ∈ [1, 4] ∪ {∞} | ≤ 4096 steps | — |
| All-to-all / small-world | N = 16–128 is enough to see exponential τ | scrambling ~log n; t_max = 4n–10n | 10²–10⁵ graphs (user's WS registry: 350,000 draws) |
| Measurement-only | 1D L ≲ 512; 2D L = 16–64 | — | — |
| Absorbing state (DP class) | L = 1024–4096 | t ~ L^z, z ≈ 1.58 | — |
| QEC memory | d = 3–25 | d–3d rounds | 10⁶–10⁹ shots (frame sampler) |
| Noisy monitored **[added]** | 1D chains; noise rate q fixed, or q = c/L^α (2401.16631) | until I(R:Q) → 0; protection time ∼ q^−1/2 | —† |
| Static-disorder MIPT **[added]** | 1D Clifford, large L (2205.14002) | z = ∞ at criticality, so use a log time grid | —† |

The sample counts are typical practice, not checked paper by paper†. The user's target, millions of trajectories at n = 256–1024, is 1–3 orders of magnitude beyond typical published studies.

---

## 5. References

† = journal reference, author list or existence not verified against the full text.

**MIPT, purification, reviews.**
- Y. Li, X. Chen, M. P. A. Fisher, PRB 98, 205136 (2018), arXiv:1808.06134.
- Y. Li, X. Chen, M. P. A. Fisher, PRB 100, 134306 (2019), arXiv:1901.08092.
- B. Skinner, J. Ruhman, A. Nahum, PRX 9, 031009 (2019), arXiv:1808.05953.
- A. Nahum, J. Ruhman, S. Vijay, J. Haah, PRX 7, 031016 (2017), arXiv:1608.06950.
- M. J. Gullans, D. A. Huse, PRX 10, 041020 (2020), arXiv:1905.05195.
- M. J. Gullans, D. A. Huse, PRL 125, 070606 (2020), arXiv:1910.00020.
- A. Zabalo, M. J. Gullans, J. H. Wilson, S. Gopalakrishnan, D. A. Huse, J. H. Pixley, PRB 101, 060301(R) (2020), arXiv:1911.00008.
- A. Zabalo et al., "Operator scaling dimensions and multifractality at measurement-induced transitions", arXiv:2107.03393 (PRL 128, 050602 (2022)†).
- S. Choi, Y. Bao, X.-L. Qi, E. Altman, PRL 125, 030505 (2020), arXiv:1903.05124.
- L. Fidkowski, J. Haah, M. B. Hastings, Quantum 5, 382 (2021), arXiv:2008.10611.
- R. Fan, S. Vijay, A. Vishwanath, Y.-Z. You, PRB 103, 174309 (2021), arXiv:2002.12385.
- P. Sierant, M. Schirò, M. Lewenstein, X. Turkeshi, PRB 106, 214316 (2022)†, arXiv:2210.11957.
- P. Sierant, X. Turkeshi, arXiv:2308.13384†. The review pass could not confirm this ID.
- O. Lunt, M. Szyniszewski, A. Pal, PRB 104, 155111 (2021), arXiv:2012.03857.
- X. Turkeshi, R. Fazio, M. Dalmonte, PRB 102, 014315 (2020), arXiv:2007.02970†.
- Y. Li, R. Vasseur, M. P. A. Fisher, A. W. W. Ludwig, arXiv:2110.02988.
- M. P. A. Fisher, V. Khemani, A. Nahum, S. Vijay, Annu. Rev. CMP 14, 335 (2023), arXiv:2207.14280.
- A. G. Moghaddam, K. Pöyhönen, T. Ojanen, PRL 131, 020401 (2023), arXiv:2302.14044.
- C. Noel et al., Nat. Phys. 18, 760 (2022), arXiv:2106.05881.
- J. M. Koh et al., Nat. Phys. 19, 1314 (2023), arXiv:2203.04338†.
- "Universal purification dynamics of monitored Clifford circuits", arXiv:2607.06683 (title verified; authors†).
- ~~Gerbino et al., arXiv:2603.10751~~ **unidentified** (see §3.7).
- Yu-Xuan Zhang, Yu-Xiang Zhang, "Typical output states of monitored random Clifford circuits: a graph-theoretic approach", arXiv:2608.03102 [verified].
- **[added]** Y. Bao, S. Choi, E. Altman, PRB 101, 104301 (2020), arXiv:1908.04305.
- **[added]** C.-M. Jian, Y.-Z. You, R. Vasseur, A. W. W. Ludwig, PRB 101, 104302 (2020), arXiv:1908.08051.
- **[added]** A. C. Potter, R. Vasseur, "Entanglement dynamics in hybrid quantum circuits", arXiv:2111.08018 (review†).
- **[added]** A. Zabalo et al., "Infinite-randomness criticality in monitored quantum dynamics with static disorder", PRB 107, L220204 (2023), arXiv:2205.14002 [verified].
- **[added]** P. Sierant, X. Turkeshi, arXiv:2109.06882, participation entropy at MIPT (PRL 128, 130605 (2022)†).
- **[added]** arXiv:2307.15689†, "Engineering entanglement geometry via spacetime-modulated measurements" (dual-unitary Clifford ρ_c = 0.2048(5)).
- **[added]** E. Buznach, D. Chowdhury, J. Ruhman, PRL 135, 050402 (2025), arXiv:2502.13211.
- **[added]** arXiv:2407.03206†, "Multipartite GHZ entanglement in monitored random Clifford circuits".
- **[added]** arXiv:2609.25201†, "Measurement-induced phase transitions and logical-space localization in Floquet monitored Clifford circuits".

**Stabilizer formalism and algorithms.**
- S. Aaronson, D. Gottesman, PRA 70, 052328 (2004), quant-ph/0406196.
- D. Fattal, T. S. Cubitt, Y. Yamamoto, S. Bravyi, I. L. Chuang, quant-ph/0406168.
- K. M. R. Audenaert, M. B. Plenio, NJP 7, 170 (2005), quant-ph/0505036.
- S. Bravyi, D. Fattal, D. Gottesman, J. Math. Phys. 47, 062106 (2006), quant-ph/0504208.
- R. Koenig, J. A. Smolin, J. Math. Phys. 55, 122202 (2014), arXiv:1406.2170.
- S. Bravyi, D. Maslov, IEEE TIT 67, 4546 (2021), arXiv:2003.09412.
- S. Anders, H. J. Briegel, PRA 73, 022334 (2006), quant-ph/0504117.
- S. Bravyi et al. (CH form), Quantum 3, 181 (2019), arXiv:1808.00128.
- **[added]** E. van den Berg, "A simple method for sampling random Clifford operators", arXiv:2008.06011.

**Codes and erasure.**
- M. J. Gullans, S. Krastanov, D. A. Huse, L. Jiang, S. T. Flammia, PRX 11, 031066 (2021), arXiv:2010.09775.
- Y. Li, M. P. A. Fisher, PRB 103, 104306 (2021), arXiv:2007.03822 [verified].
- Y. Li, S. Vijay, M. P. A. Fisher, PRX Quantum 4, 010331 (2023), arXiv:2105.13352.
- Z.-C. Yang, Y. Li, M. P. A. Fisher, X. Chen, "Entanglement phase transitions in random stabilizer tensor networks", PRB 105, 104306 (2022), arXiv:2107.12376 [verified; tensor networks, not d_cont of circuits].
- B. Yoshida, "Decoding the entanglement structure of monitored quantum circuits", arXiv:2109.08691.
- S. Sang, Z. Li, T. H. Hsieh, B. Yoshida, "Ultrafast entanglement dynamics in monitored quantum circuits", PRX Quantum (2023), arXiv:2212.10634.
- Y. Han, X. Chen, "Entanglement structure in the volume-law phase of hybrid quantum automaton circuits", arXiv:2207.02165.
- D. Lee, B. Yoshida, "Randomly monitored quantum codes", arXiv:2402.00145 [verified].
- **[added]** J. Nelson, J. Rajakumar, M. J. Gullans, "Error correction phase transition in noisy random quantum circuits", arXiv:2510.07512 (formerly listed here unidentified).
- **[added]** I. Lovas, U. Agrawal, S. Vijay, PRX Quantum 5, 030327 (2024), arXiv:2304.02664.
- **[added]** S. P. Kelly, U. Poschinger, F. Schmidt-Kaler, M. P. A. Fisher, J. Marino, SciPost Phys. 15, 250 (2023), arXiv:2210.11547.
- **[added]** "Coherent information phase transition in a noisy quantum circuit", arXiv:2408.16267 (authors†).
- **[added]** S. Liu, S.-K. Jian, S.-X. Zhang, "Noisy monitored quantum circuits" (review), arXiv:2512.18783.
- J. Nelson, G. Bentsen, S. T. Flammia, M. J. Gullans, "Fault-tolerant quantum memory using low-depth random circuit codes", PRR 7, 013040 (2025)†, arXiv:2311.17985 [abstract verified].
- A. S. Darmawan, Y. Nakata, S. Tamiya, H. Yamasaki, PRR 6, 023055 (2024), arXiv:2212.05071.
- G. M. Sommers, D. A. Huse, M. J. Gullans, "Dynamically generated concatenated codes and their phase diagrams", PRR 7, 023086 (2025), arXiv:2409.13801 [verified].
- W. Brown, O. Fawzi: arXiv:1312.7646 (ISIT 2013); arXiv:1307.0632 (CMP 340, 867 (2015)†); arXiv:1210.6644.
- "Good stabilizer codes from shallow Clifford circuits with random matchings", arXiv:2608.18536 (title verified; the author list E. Anand, E. Gorokhovsky, J. Hritz, J. Sun is not†).
- C. H. Bennett, D. P. DiVincenzo, J. A. Smolin, PRL 78, 3217 (1997), quant-ph/9701015.
- M. Grassl, T. Beth, T. Pellizzari, PRA 56, 33 (1997), quant-ph/9610042.
- S. Bravyi, B. Terhal, NJP 11, 043029 (2009)†, arXiv:0810.1983.
- D. Poulin, PRL 95, 230504 (2005), quant-ph/0508131.
- U. Kapshikar, S. Kundu, arXiv:2203.04262.
- L. P. Pryadko, V. A. Shabashov, V. K. Kozin, QDistRnd, arXiv:2308.15140†.
- M. Webster, A. Jacob, O. Higgott, "Distance-finding algorithms for quantum codes and circuits", arXiv:2603.22532 [verified].
- P. Hayden, J. Preskill, JHEP 09 (2007) 120, arXiv:0708.4025.
- B. Yoshida, A. Kitaev, arXiv:1710.03363.
- B. Yoshida, arXiv:2106.15628.
- P. Hosur, X.-L. Qi, D. A. Roberts, B. Yoshida, JHEP 02 (2016) 004, arXiv:1511.04021.
- ~~arXiv:2505.22720~~ **unidentified**: no search hit on 2026-10-13. Remove it unless someone identifies it.

**Geometry, long range, graphs, trees.**
- A. Nahum, S. Roy, B. Skinner, J. Ruhman, PRX Quantum 2, 010352 (2021), arXiv:2009.11311.
- X. Feng, B. Skinner, A. Nahum, PRX Quantum 4, 030333 (2023), arXiv:2210.07264.
- X. Feng, N. Fishchenko, S. Gopalakrishnan, M. Ippoliti, Quantum 9, 1692 (2025), arXiv:2405.13894.
- B. Ferté, X. Cao, arXiv:2312.04284†.
- J. Lopez-Piqueres, B. Ware, R. Vasseur, PRB 102, 064202 (2020), arXiv:2003.01138.
- M. Block, Y. Bao, S. Choi, E. Altman, N. Y. Yao, PRL 128, 010604 (2022), arXiv:2104.13372.
- T. Minato, K. Sugimoto, T. Kuwahara, K. Saito, PRL 128, 010603 (2022), arXiv:2104.09118.
- S. Sharma, X. Turkeshi, R. Fazio, M. Dalmonte, SciPost Phys. Core 5, 023 (2022), arXiv:2110.14403.
- Y. Otsuka, K. Seki, S. Yunoki, "Quantum conditional mutual information as a probe of measurement-induced entanglement phase transitions", PRB 112, 054301 (2025), arXiv:2504.03472 [verified].
- J. Richter, O. Lunt, A. Pal, PRR 5, L012031 (2023), arXiv:2205.06309.
- G. Bentsen, S. Sahu, B. Swingle, PRB 104, 094304 (2021), arXiv:2104.07688.
- S. Sahu, S.-K. Jian, G. Bentsen, B. Swingle, PRB 106, 224305 (2022), arXiv:2109.00013.
- T. Hashizume, G. Bentsen, A. J. Daley, PRR 4, 013174 (2022), arXiv:2109.10944.
- T. Hashizume, S. Kuriyattil, A. J. Daley, G. Bentsen, Symmetry 14, 666 (2022), arXiv:2202.11750.
- G. Bentsen, Y. Gu, A. Lucas, PNAS 116, 6689 (2019)†, arXiv:1805.08215.
- G. Cemin, M. Bukov, A. Solfanelli, arXiv:2610.12597†. Existence was confirmed by only one of two sweeps, and the review pass found no search hit. Treat it as unverified.
- S. Vijay, arXiv:2005.03052.
- D. J. Watts, S. H. Strogatz, Nature 393, 440 (1998).
- M. E. J. Newman, D. J. Watts, Phys. Lett. A 263, 341 (1999), cond-mat/9903357.
- J. Misra, D. Gries, IPL 41, 131 (1992).
- **[added]** V. G. Vizing, Diskret. Analiz 3, 25 (1964) (edge-colouring bound Δ+1).
- L. G. Valiant, TCS 8, 189 (1979).
- A. Steger, N. C. Wormald, CPC 8, 377 (1999).

**Other protocols.**
- M. Ippoliti, M. J. Gullans, S. Gopalakrishnan, D. A. Huse, V. Khemani, PRX 11, 011030 (2021), arXiv:2004.09560.
- A. Lavasani, Y. Alavirad, M. Barkeshli, Nat. Phys. 17, 342 (2021), arXiv:2004.07243.
- A. Lavasani, Y. Alavirad, M. Barkeshli, PRL 127, 235701 (2021), arXiv:2011.06595.
- S. Sang, T. H. Hsieh, PRR 3, 023200 (2021), arXiv:2004.09509.
- N. Lang, H. P. Büchler, PRB 102, 094204 (2020)†, arXiv:2006.09748.
- K. Klocke, M. Buchhold, arXiv:2204.08489.
- V. Sriram, T. Rakovszky, V. Khemani, M. Ippoliti, arXiv:2207.07096†.
- A. Lavasani, Z.-X. Luo, S. Vijay, arXiv:2207.02877†.
- G. Morral-Yepes, F. Pollmann, I. Lovas, arXiv:2302.14551.
- M. B. Hastings, J. Haah, Quantum 5, 564 (2021), arXiv:2107.02194.
- C. Gidney, M. Newman, A. Fowler, M. Broughton, Quantum 5, 605 (2021), arXiv:2108.10457.
- T. Iadecola, S. Ganeshan, J. H. Pixley, J. H. Wilson, PRL 131, 060403 (2023), arXiv:2207.12415.
- C. LeMaire, A. A. Allocca, J. H. Pixley, T. Iadecola, J. H. Wilson, PRB 110, 014310 (2024), arXiv:2309.04520.
- M. Buchhold, T. Müller, S. Diehl, arXiv:2208.10506.
- N. O'Dea, A. Morningstar, S. Gopalakrishnan, V. Khemani, PRB 109, L020304 (2024), arXiv:2211.12526.
- L. Piroli, Y. Li, R. Vasseur, A. Nahum, arXiv:2212.14026.
- P. Sierant, X. Turkeshi, PRL 130, 120402 (2023)†, arXiv:2212.13823.
- V. Ravindranath, Y. Han, Z.-C. Yang, X. Chen, PRB 108, L041103 (2023), arXiv:2211.05162.
- A. J. Friedman, O. Hart, R. Nandkishore, PRX Quantum 4, 040309 (2023), arXiv:2210.07256.
- U. Agrawal et al., PRX 12, 041002 (2022), arXiv:2107.10279.
- F. Barratt et al., PRL 129, 120604 (2022)†, arXiv:2111.09336.
- F. Barratt, U. Agrawal, A. C. Potter, S. Gopalakrishnan, R. Vasseur, PRL 129, 200602 (2022), arXiv:2206.12429.
- J. Ha, A. Pandey, S. Gopalakrishnan, D. A. Huse, "Measurement-induced phase transitions in systems with diffusive dynamics", PRB 110, L140301 (2024), arXiv:2405.08861 [verified].
- Y. Bao, S. Choi, E. Altman, Ann. Phys. 435 (2021), arXiv:2102.09164.
- Y. Li, M. P. A. Fisher, PRB 108, 214302 (2023), arXiv:2108.04274.
- H. Dehghani, A. Lavasani, M. Hafezi, M. J. Gullans, Nat. Commun. 14, 2918 (2023), arXiv:2204.10904.
- Y. Li, Y. Zou, P. Glorioso, E. Altman, M. P. A. Fisher, PRL 130, 220404 (2023), arXiv:2209.00609.
- H. Kamakari et al., PRL 134, 120401 (2025), arXiv:2403.00938.
- M. Ippoliti, V. Khemani, arXiv:2307.15011.
- U. Agrawal et al., PRX 14, 041012 (2024), arXiv:2311.00058.
- Z. Weinstein, Y. Bao, E. Altman, PRL 129, 080501 (2022), arXiv:2202.12905.
- S. Liu, M.-R. Li, S.-X. Zhang, S.-K. Jian, arXiv:2401.01593.
- S. Liu et al., "Noise-induced phase transitions in hybrid quantum circuits", arXiv:2401.16631 (title verified; studies q = p/L^α).
- S. Sang, Y. Li, T. Zhou, X. Chen, T. H. Hsieh, M. P. A. Fisher, PRX Quantum 2, 030313 (2021), arXiv:2012.00031.
- B. Shi, X. Dai, Y.-M. Lu, arXiv:2012.00040†.
- M. Ippoliti, V. Khemani, PRL 126, 060501 (2021)†, arXiv:2010.15840.
- M. Ippoliti, T. Rakovszky, V. Khemani, PRX 12, 011045 (2022), arXiv:2103.06873.
- T.-C. Lu, T. Grover, PRX Quantum 2, 040319 (2021)†, arXiv:2103.06356.
- J. C. Hoke et al., Nature 622, 481 (2023), arXiv:2303.04792.
- Y. Bao, M. Block, E. Altman, arXiv:2110.06963.
- A. Nahum, B. Skinner, PRR 2, 023288 (2020), arXiv:1911.11169.
- J. Iaconis, A. Lucas, X. Chen, PRB 102, 224311 (2020)†, arXiv:2010.02196.

**Simulators and tooling.**
- C. Gidney, "Stim", Quantum 5, 497 (2021), arXiv:2103.02202. Stim issue #432: AVX2 disabled in PyPI wheels.
- Qiskit Aer, `src/simulators/stabilizer/clifford.hpp`.
- Cirq, `stabilizer_state_ch_form.py`.
- QuantumClifford.jl (QuantumSavory).
- PyClifford (github.com/hongyehu/PyClifford).
- W. Fang, M. Ying, SymPhase, arXiv:2311.03906.
- S. Garner et al., STABSim, arXiv:2507.03092.
- M. Osama, D. Thanos, A. Laarman, QuaSARQ, arXiv:2603.14641.
- Qimax, arXiv:2505.03307†.
- Qibo Clifford backend, arXiv:2408.00384.
- Tsim arXiv:2604.01059†, Clifft arXiv:2604.27058†, SymFT arXiv:2607.28600†.
- NVIDIA cuQuantum `cuquantum.stabilizer`.
- PennyLane `default.clifford`.

**LLM interface.**
- MCP Python SDK `mcp` 2.3.0 (PyPI) and v2 docs (py.sdk.modelcontextprotocol.io/v2).
- MCP Specification 2025-11-25.
- Claude Code docs: code.claude.com/docs/en/mcp, /memory, /skills.
- K. Aizawa et al., "Writing effective tools for agents", Anthropic Engineering (2025).
- J. Howard, llms.txt proposal (2024).
- AGENTS.md (agents.md).
- Command Line Interface Guidelines (clig.dev).
- Qiskit MCP servers (github.com/Qiskit/mcp-servers).
- H. Pan et al., arXiv:2508.18489†.
- E. L. Kaplan, P. Meier, JASA 53, 457 (1958).
- **[added]** K. Harada, "Bayesian inference in the scaling analysis of critical phenomena", PRE 84, 056704 (2011)†, arXiv:1102.4149†. A candidate collapse method for S7.
- D. Blackman, S. Vigna, xoshiro256\*\* (2018).
- NumPy NEP 19.
