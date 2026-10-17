# Watts–Strogatz monitored Clifford circuits: focused MIPT study

**Version:** 0.3.0 (2026-10-17)  
**Implementation:** SparseGF2, pinned Git commit and native RNG contract  
**Ensemble definition:** *Watts–Strogatz monitored-Clifford ensemble specification*, v1.4  
**Purpose:** A small, literature-grounded measurement-induced-phase-transition (MIPT) study across rewiring probability \(\beta\), adding the full-reference first-passage family \(\tau_\ell\) and trajectory-level peak depth \(T^*\). This document replaces the expansive v0.2 observable program as the **implementation scope**; it does not alter the circuit ensemble.

## 1. Scientific questions

For the fixed-degree-four initial ring rewired by \(\beta\), determine:

1. The transition location \(p_c(\beta)\), and whether conventional entanglement and full-reference purification diagnostics identify the same transition.
2. The critical exponents \(\nu(\beta)\), \(z(\beta)\), and whether conventional near-1D finite-size scaling survives nonlocal rewiring.
3. Whether the **new** full-reference \(\tau_\ell\) and ensemble-averaged contiguous-distance peak time \(\langle T^*\rangle\) show compatible critical scaling.
4. Whether apparent changes with \(\beta\) track actual finite-graph small-world geometry instead of just nominal rewiring probability.

**Do not implement the v0.2 exploratory catalog.** Implement exactly the observables in §3, the graph measurements in §4, and the fitting/validation procedures in §§5–7. These form the first complete campaign.

## 2. Sampling and two initializations

Use the circuit, graph, and RNG laws of companion v1.4 without modification: \(G\sim\mu^{\mathrm{WS}}_{n,\beta}\), \(n\) physical qubits, \(\lfloor n/2\rfloor\) sequential uniformly sampled edges **with replacement** per layer, uniformly sampled two-qubit Cliffords, and independent physical-site \(Z\) measurements with probability \(p\) after the gate block. Circuit depth \(t\) counts completed gate-plus-measurement layers.

- **F — full-reference run (primary):** initialize \(n\) Bell pairs between physical sites \(S_i\) and reference sites \(R_i\). The joint \(SR\) tableau is a **pure** phase-free stabilizer state, while \(\rho_S\) is generally mixed. Compute \(k(t)=S(R)_t=S(S)_t\) in bits, \(\tau_\ell\), and contiguous-distance peak times from this branch.
- **P — pure-product control (literature comparison):** initialize \(|0\rangle^{\otimes n}\) on physical sites only. Compute conventional pure-state half-chain entropy, tripartite mutual information (TMI), and antipodal mutual information (MI). These **cannot** be obtained from the mixed F physical-state tableau. Pair P and F with the *same ordered gate/measurement-mask history* where replay supports it and test equivalence; maintain separate state identifiers.

All observables are phase-free. Subsystem entropies are exact binary-rank quantities in **bits**: for any stabilizer state on register \(\Omega\) with stabilizer row space \(W_t\),
\[
S_t(A)=|A|-\dim\{w\in W_t:\operatorname{supp}(w)\subseteq A\}.
\]
See Gullans–Huse (2020), Appendix E, Eqs. (E1)–(E3). For pure P, \(S_t(A)=S_t(\Omega\setminus A)\). The F reduced code supplies entropy/erasure probes but must not be called a pure-state trajectory.

## 3. Exact observable set

| ID | Branch | Definition and sampling | Main role |
|---|---|---|---|
| `F.K` | F | Exact \(k(t)\) for **every layer**, its ensemble mean \(\langle k(t)\rangle\), survival \(\Pr(k(t)>\ell)\), and distribution at selected times | Reproduce global mixed-state purification dynamics; characterize information survival |
| `F.TAU` | F | First-passage \(\tau_\ell=\min\{t\ge0:k(t)\le\ell\}\), all \(\ell=0,\dots,n\); event and censor flags | Test time scaling for \(\ell=0,1\) and selected \(\ell\); **not** the published single-reference protocol |
| `F.DCYC` | F | Exact cyclic-contiguous distance \(d_{\rm cyc}(t)\) from original ring order, with declared zero extension if \(k(t)=0\), at **predeclared depth checkpoints** | New robustness-peak analysis using the prior observable |
| `F.TSTAR` | F | Per-trajectory \(T^*\) from the maximum of `F.DCYC`; report \(\langle T^*\rangle\), peak-height statistics and maximizing checkpoints | Test the proposed new characteristic dynamical time |
| `P.HALF` | P | Pure-state half-ring entropy \(S_t(A)\), \(|A|=n/2\), and selected interval entropies at steady-state-validated depths | Conventional entanglement phase characterization |
| `P.TMI` | P | \(I_3(A:B:C)\) for four **equal consecutive ring quarters** (\(4\mid n\)) at late time | Primary conventional estimate of \(p_c\) and \(\nu\) via crossings/collapse |
| `P.MI` | P | \(I(A:B)=S(A)+S(B)-S(AB)\) for two specified small *antipodal* blocks, using ring labels; store block size and position averaging | Additional peak-based critical-point check, especially for nonlocal dynamics |

### 3.1 The literature diagnostics

**(a) TMI — Zabalo et al. (2020), Eqs. (1)–(2), Fig. 1(c) and Fig. 2.** For four consecutive regions \(A,B,C,D\), each containing \(n/4\) sites,
\[
I_3(A:B:C)=S(A)+S(B)+S(C)-S(AB)-S(AC)-S(BC)+S(ABC).
\]
For the short-range steady-state model, a size-independent critical crossing is well described by \(I_3(p,n)\simeq f[(p-p_c)n^{1/\nu}]\). **Replicate the method, not the paper's numerical \(p_c\)**: the Watts–Strogatz graph/gate ensemble differs. Run P to depths sufficient for stabilization; the paper uses \(t=4n\) as its short-range benchmark. At \(\beta>0\), verify the plateau instead of assuming \(4n\) is necessary or sufficient. Do not force the crossing form if the amplitude drifts with \(n\).

**(b) Half-chain entropy — Li, Chen & Fisher (2019); Block et al. (2022), diagnostic discussion and supplementary §I A.** Measure steady-state \(S_{n/2}\) in P, and its dependence on \(n\) and \(p\). Distinguish saturation, logarithmic, subextensive, and extensive behavior *from fits*; do not assume the conventional 1D critical chord-law when \(\beta>0\). The cut is in the **original ring ordering**, not a graph-optimal cut.

**(c) Antipodal MI — Block et al. (2022), main-text Eq. (1), Fig. 3(g), supplement §I B.** Use two predeclared small blocks \(A,B\) opposite each other in ring ordering; initially choose **single-site blocks** \(A=\{i\}\), \(B=\{i+n/2\}\) for even \(n\), averaged over starting site \(i\). Record the exact region convention; validate it against the source geometry rather than claim identical finite-size amplitudes. Locate the peak in \(p\). Test
\[
I_{AB}(p,n)\simeq n^{y_{\rm MI}}f_{\rm MI}[(p-p_c)n^{1/\nu}],
\]
where \(y_{\rm MI}\) is **not** the rewiring parameter \(\beta\). Do not assume \(y_{\rm MI}=0\) for small-world circuits or treat a sharp peak as proof of conformal invariance. Block et al. explicitly warn that the MI-derived \(\nu\) can differ from purification-derived estimates in nonlocal regimes (supplement §I B).

**(d) Global purification — Block et al. (2022), main-text Eq. (3), Fig. 3(c,f); Gullans & Huse (2020).** With F initialization, \(k(t)\) is the entropy of the initial maximally mixed physical system along a conditioned trajectory. Use \(\langle k(t)\rangle\) and \(\Pr(k(t)>\ell)\) over \((p,n,\beta)\); check a two-variable dynamical collapse, as a hypothesis,
\[
\langle k(t)\rangle\stackrel?\simeq \mathcal K_\beta((p-p_c)n^{1/\nu},t/n^z).
\]
Test only over a justified near-critical scaling window. Always report the sampled \(t\) range and whether the data cover the predicted relaxation times.

### 3.2 New purification and peak observables

**(e) Family \(\tau_\ell\).** Calculate \(k(t)\) after every completed layer, not only at sparse tableau checkpoints. For \(\ell\in\{0,1,2,\lfloor n/4\rfloor\}\) when distinct, analyze the survival function \(\Pr(\tau_\ell>t)=\Pr(k(t)>\ell)\), median and predeclared quantiles. Save the full \(\tau_\ell\) vector at no additional circuit cost. When a crossing is not seen by \(T_{\max}\), save a **right-censored** event, not an observed infinity or an observed finite time. A conventional *test* for a chosen quantile is
\[
\tau_{\ell,q}(n,p;\beta)\stackrel?\simeq n^{z_\ell} f_{\ell,q,\beta}[(p-p_c)n^{1/\nu_\ell}],
\]
with \(z_\ell\) fitted. For \(\ell\propto n\), do **not** require the same exponents as fixed \(\ell\). **Do not identify \(\tau_1\) from F with the single-reference purification time** of Hashizume et al. (2022), Eq. (5), or Block et al. (2022), Eq. (2): their probe uses a different initial condition (and scrambling). The *scaling methodology* is adapted here, not their precise observable.

**(f) Contiguous-distance peak and \(\langle T^*\rangle\).** For each F trajectory, at positive checkpoints \(\mathcal T_+\), define \(d(t)=d_{\rm cyc}(t)\), with \(d(t)=0\) after complete purification, and
\[
d^*=\max_{t\in\mathcal T_+}d(t),\quad
\mathcal A=\{t\in\mathcal T_+:d(t)=d^*\},\quad
T^*=\exp\!\left(\frac{1}{|\mathcal A|}\sum_{t\in\mathcal A}\ln t\right).
\]
This is the **trajectory peak log-midpoint** from this study. If \(d^*=0\), mark the peak undefined. If the maximizing set includes \(T_{\max}\), flag the peak as **right-boundary-limited**. Record \(d^*\), \(\mathcal A\), \(T^*\), and \(\{k(t):t\in\mathcal A\}\). Compute \(\langle T^*\rangle\) by the **graph-weighted ensemble mean of individual valid \(T^*\)**, with the invalid and boundary-limited fractions reported. Distinguish it explicitly from \(\arg\max_t\langle d(t)\rangle\) and \(\exp\langle\ln T^*\rangle\). Test
\[
\langle T^*\rangle\stackrel?\simeq n^{z_*}f_{*,\beta}[(p-p_c)n^{1/\nu_*}],
\]
**only when the valid, interior-peak fraction and checkpoint density support this estimand**. Report sensitivity to the checkpoint set. For \(\tau_1\) comparisons, use the actual maximizing checkpoints: \(t\ge\tau_1\iff k(t)\le1\), not \(k(T^*)\), because \(T^*\) generally lies between checkpoints.

The code-distance peak is included here strictly as a dynamical MIPT diagnostic; no geometry-correlated erasure \(J\), integrated \(D(J)\), connected-subgraph enumeration, or other code-protection campaign is required.

## 4. Minimal graph measurements

Save the full labeled edge list and graph seed once per \((n,\beta,g)\). Compute these **four** graph summaries independently of the circuit:

| Graph statistic | Definition | Comparison |
|---|---|---|
| \(q_G\) | \(|E(G)\setminus E(C(n,2))|/(2n)\) | Realized non-original edge fraction versus nominal \(\beta\) |
| \(\bar d_G\) | Mean shortest-path distance over reachable distinct vertex pairs; also store component count | Small-world path contraction and connectedness |
| \(\lambda_2(G)\) | Second-smallest eigenvalue of the **unnormalized** graph Laplacian \(D-A\) (zero when disconnected) | Spectral connectivity versus critical time scales |
| \(\operatorname{Var}_V(\deg_G)\) | Population variance of graph vertex degrees | Degree heterogeneity introduced by rewiring |

These are **covariates**, not assumed to determine the MIPT. For fixed \((n,\beta,p)\), compare graph-level circuit averages of \(\tau_\ell\), \(T^*\), and pure-state TMI against \(\bar d_G\), \(\lambda_2(G)\), and \(q_G\), with uncertainty clustered by independent graphs. A single graph's apparent transition point is a noisy finite-size pseudocritical estimate, not the thermodynamic \(p_c(\beta)\).

## 5. Targeted finite-size-scaling protocol

**Step 1 — pilot.** Use a size set with \(4\mid n\) for TMI (e.g. \(n=32,64,128,256\)), several fixed \(\beta\) values covering \(0\) and the strongly rewired regime, and an exploratory \(p\) grid broad enough to bracket each transition. The numbers are initial computational choices, **not** universal constants. Run enough independent graphs to estimate graph-level variance. Save the circuit inputs and selected phase-free snapshots for both F and P; keep F \(k(t)\) at every layer.

**Step 2 — reproducibility of the literature *method*.** At \(\beta=0\), demonstrate that pure P half-chain entropy and TMI diagnose an entanglement transition in the appropriate qualitative manner. Do not require agreement with published numeric \(p_c\), which depends on the specific circuit layout. At every \(\beta\), compare independent estimates from P TMI crossings, P antipodal MI peaks, F purification survival/quantiles, and F global \(k(t)\) collapse. State clearly which fits are identifiable and which are not.

**Step 3 — fixed-\(\beta\) scaling.** For each \(\beta\), estimate \(p_c\) and \(\nu\) first from the most stable P crossing, if one exists. Independently fit \(z\) and \(\nu\) using F \(\tau_\ell\) distributions and \(\langle k(t)\rangle\) collapse; compare with \(T^*\) fits without forcing agreement. Allow \(z\ne1\) and nonzero MI scaling dimension. Do not treat a failed collapse as evidence of an absent transition without checking depth, finite-size drift, and nonlocal geometry.

**Step 4 — beta dependence and crossover.** Plot the estimated \(p_c(\beta)\), \(\nu(\beta)\), \(z(\beta)\), and \(z_*(\beta)\), always with confidence intervals. Repeat selected small-\(\beta\) points along \(\beta_n=b/n\) sequences (predeclare \(b\)) to distinguish fixed-\(\beta\) behavior from small-world finite-size crossover. Relate the results to saved \(q_G\), \(\bar d_G\), \(\lambda_2\), and degree variance. A smooth curve in \(\beta\) is an inferred interpolation, not a directly observed continuum of parameter values.

**Statistics.** Equal total weight to each independently sampled graph; circuits within a graph are nested samples. Retain rate pairing inside graph-level bootstrap resamples. For F first passages use censor-aware survival/quantiles and/or deterministic longer-depth continuations; never use the mean of only observed crossing times as an unconditional mean. For \(T^*\), bootstrap whole trajectories **within graph blocks**, preserving tied maxima and checkpoint correlation. Assess fit stability by removing smallest sizes, moving depth windows, testing held-out sizes, and comparing shared versus independent exponents.

## 6. Storage and agent acceptance criteria

Persist enough information to compute the seven catalog items without rerunning an existing checkpoint:

- Graph key, exact \(\beta\)/\(p\) tokens, native seeds/trajectory index, pinned SparseGF2 commit/RNG contract, labeled edge list and graph summaries.
- F phase-free joint tableau and physical-code snapshot at sparse declared checkpoints; **full per-layer** \(k(t)\) and all \(\tau_\ell\) event/censor records.
- P **separately evolved** phase-free state snapshots for the same circuit-input realization at late-time-validated depths; region masks for entropy/TMI/MI.
- F cyclic-distance trace, exact maximum/tie set, \(T^*\), and `undefined_peak`/`peak_at_horizon` flags.
- Raw per-trajectory data and graph-level aggregates, not only grand averages; observable definitions/versions and saved analysis scripts.

**Mandatory tests:** (1) \(p=0\) gives F \(k(t)=n\); \(p=1\) gives \(k(1)=0\). (2) F \(k(t)\) is nonincreasing and first-passage times agree with the complete trace. (3) Pure P satisfies \(S(A)=S(\bar A)\) and TMI/MI match a small independent stabilizer oracle. (4) Saved/reloaded phase-free row spaces reproduce observables irrespective of generator basis. (5) Graph generation, phase-free replay, F/P paired circuit inputs, and extension to longer horizons preserve existing trajectory prefixes. (6) \(d_{\rm cyc}\), tie handling, and \(\langle T^*\rangle\) are verified with hand-constructed traces and small-code erasure oracles. (7) Synthetic scaling data validate crossing/collapse software, graph-cluster resampling, and censor handling. (8) An actual small multi-size, multi-\(\beta\), multi-\(p\) run produces all required records, plots and a transparent test report.

**Required plots:** one pure P TMI-crossing/collapse plot, one P antipodal-MI-versus-\(p\) plot, one F \(k(t)\) surface/selected-time collapse, one F \(\tau_\ell/n^z\) family, one F \(\langle T^*\rangle/n^{z_*}\) family, a direct \(T^*\)/\(\tau_1\) comparison on paired trajectories, and \(p_c(\beta)\) with graph summaries. Clearly label empty/insufficient fits instead of inventing exponent estimates.

## 7. Literature and provenance

1. **Zabalo et al. (2020)**, *Critical properties of the measurement-induced transition in random quantum circuits*, [arXiv:1911.00008](https://arxiv.org/pdf/1911.00008), **Eqs. (1)–(2), Figs. 1(c), 2**: pure-state equal-quarter TMI and its steady-state crossing/collapse; their benchmark uses depth \(4L\) in a 1D circuit, not a guarantee for the Watts–Strogatz ensemble.
2. **Li, Chen & Fisher (2019)**, *Measurement-driven entanglement transition in hybrid quantum circuits*, [arXiv:1901.08092](https://arxiv.org/pdf/1901.08092): pure-state volume-/area-law entropy diagnostics and critical logarithmic entropy in local circuits.
3. **Block et al. (2022)**, *Measurement-Induced Transition in Long-Range Interacting Quantum Circuits*, [arXiv:2104.13372](https://arxiv.org/pdf/2104.13372), **main Eqs. (1)–(3), Fig. 3(g–i); supplement §§I B–D, Eqs. (4)–(7)**: antipodal MI scaling, global entropy collapse and **separate single-reference** median purification time. Their power-law interaction graph is not the Watts–Strogatz ensemble; their single-ancilla result is not F \(\tau_1\).
4. **Hashizume, Bentsen & Daley (2022)**, *Measurement-induced phase transitions in sparse nonlocal scramblers*, [arXiv:2109.10944](https://arxiv.org/pdf/2109.10944), **Sec. V, Fig. 4, Eq. (5)**: single-reference purification scaling after prescribed scrambling; methodology adapted but initialization distinct.
5. **Gullans & Huse (2020)**, *Dynamical Purification Phase Transition Induced by Quantum Measurements*, [PRX 10, 041020](https://doi.org/10.1103/PhysRevX.10.041020), **Appendix E, Eqs. (E1)–(E3)**: phase-free stabilizer entropy and global mixed-state purification interpretation.
6. **Olfati-Saber (2005)**, *Ultrafast Consensus in Small-World Networks*, [DOI](https://doi.org/10.1109/ACC.2005.1470321), **Sec. IV, Eq. (8)**: graph rewiring law; edge order and orientation are fixed by pinned SparseGF2 implementation.

**Explicit new analyses:** the all-\(\ell\) full-reference family, the logarithmic tie-midpoint \(T^*\), graph-weighted \(\langle T^*\rangle\), the comparison between these two times, and tests of their shared exponents across \(\beta\). They are proposals to be established numerically, not cited published scaling results.
