# Watts–Strogatz monitored-Clifford ensemble

**Specification version:** `1.4.0`  
**Status:** Mathematical ensemble specification and implementation contract for [SparseGF2](https://github.com/sjbevins/SparseGF2), `3.0.0.dev12` at the reviewed documentation snapshot (pin a Git commit before running). Fixed choices are in §9. New observables and scaling hypotheses are not literature theorems. Unimplemented persistence features are distinguished from supported APIs in §8.1.

## 1. Ensemble hierarchy and experimental parameters

The experiment is defined by nested sampling of **interaction graphs**, **Clifford gate sequences**, **measurement locations**, and **measurement outcomes**, with probability laws specified in §6.1. Simulations are performed for fixed parameter values; parameter-grid points are not themselves random samples.

Let
\[
 n\in\mathbb Z_{\ge1},\qquad V=\{0,\ldots,n-1\},\qquad \beta,p\in[0,1].
\]
The physical qubits are \(Q_i\), and the reference qubits are \(R_i\), for \(i\in V\). The parameter \(\beta\) controls graph rewiring; \(p\) controls projective measurements.

For each \((n,\beta)\), specify a finite measurement-rate grid and depth set:
\[
 \mathcal P(n,\beta)\subseteq[0,1],\qquad
 \mathcal T(n,\beta)=\{t_1<\cdots<t_{N_T}\}\subseteq\mathbb Z_{\ge0},\qquad
 T_{\max}=\max\mathcal T(n,\beta).
\]
Every \(p\in\mathcal P(n,\beta)\) uses the same depth set. A depth \(t\) counts **completed monitored timesteps**, not individual gates. The checkpoint construction is fixed in §9. **The logical-qubit count must be observed at every completed layer** to obtain exact first-passage depths; expensive code and tableau snapshots can remain sparse. In SparseGF2 this requires the *global* `checkpoints="every_layer"` when recording exact `purification_layer`, because purification is detected at global checkpoints (not automatically at arbitrary intermediate layers). Individual expensive probes retain their own sparse checkpoint lists; see §8.1.

The sampling and record hierarchy is

| Level | Object | Identifiers | Random quantity | Conditioned on |
|---|---|---|---|---|
| 0 | Experimental condition | \((n,\beta,p)\) | None | Experiment design |
| 1 | Interaction graph | \((n,\beta,g)\) | \(G_g\) | \((n,\beta)\) |
| 2 | Gate-sequence realization | \((n,\beta,g,c[,p])\) | Ordered Clifford gates | \(G_g\), coupling mode |
| 3 | Monitored trajectory | \((n,\beta,g,c,p,r)\) | Measurement indicators and Born outcomes | Graph, gate sequence, \(p\) |
| 4 | Tableau snapshot | \((n,\beta,g,c,p,r,t)\) | Resulting stabilizer state | Entire preceding trajectory |

Here \(g,c,r\in\mathbb Z_{\ge0}\) label graph, gate-sequence, and measurement-trajectory trials, respectively. The index \(r\) permits multiple independent measurement realizations for the same gate sequence; setting \(r=0\) gives one such realization. Gate sequences and sitewise mask uniforms are intentionally coupled across measurement rates using the existing SparseGF2 random-stream contract. Phase-free dynamics do **not** sample or retain the signed Born outcomes. Conditional Born probabilities are part of the ideal physical ensemble but can be marginalized out for the present observables.

**Sampling principle.** Each graph trial has the same weight in an unconditional graph-ensemble expectation. Samples sharing a graph, gate sequence, or earlier timesteps are correlated and must retain these identifiers.

## 2. Level 1: interaction-graph distribution

### 2.1 Initial circulant graph

Define the simple, undirected circulant graph \(C(n,2)=(V,E_0(n))\) by
\[
 E_0(n)=\Bigl\{\{i,(i+d)\bmod n\}:i\in V,\ d\in\{1,2\},\ (i+d)\bmod n\ne i\Bigr\}.
\]
Edges are unordered pairs; duplicate descriptions are identified. Consequently,
\[
 \deg_{C(n,2)}(i)=\min\{4,n-1\},\qquad
 |E_0(n)|=\frac{n\min\{4,n-1\}}2.
\]
For \(n\ge5\), this is a degree-four ring with \(2n\) edges. For \(n\le5\), it is \(K_n\).

### 2.2 Rewiring law

Let \(L_n=((a_1,b_1),\ldots,(a_m,b_m))\) be a fixed ordered orientation of all *original* edges, each appearing once, where \(m=|E_0(n)|\). Set \(G^{(0)}=C(n,2)\). For \(j=1,\ldots,m\), perform:

1. Draw \(B_j\sim\operatorname{Bernoulli}(\beta)\), independently for each \(j\).
2. If \(B_j=0\), set \(G^{(j)}=G^{(j-1)}\).
3. If \(B_j=1\), construct the eligible target set **from the graph before deleting the original edge**:
   \[
   A_j=V\setminus\bigl(N_{G^{(j-1)}}(a_j)\cup\{a_j\}\bigr).
   \]
4. If \(A_j=\varnothing\), leave the graph unchanged. Otherwise, draw \(b'_j\) uniformly from \(A_j\) and replace \(\{a_j,b_j\}\) by \(\{a_j,b'_j\}\).

The resulting graph is
\[
 G_g=G^{(m)}\sim\mu_{n,\beta}:=\operatorname{WS}(n,\beta;L_n).
\]
This is the random-rewiring operator of Olfati-Saber (2005, §IV, Eq. (8)), with its rewiring probability renamed \(\beta\). The paper does not specify a universal edge order or retained endpoint, so \(L_n\) must be fixed as part of the ensemble definition. The fixed orientation and ordering **is** the native SparseGF2 **offset-major** order: first traverse `d = 1,2`, then `i = 0,...,n−1` within each offset; append `(a,b)=(i,(i+d) mod n)` to `L_n`. For the production domain `n≥6`, these `2n` oriented edges are all distinct and match `graphs._ring_lattice(n, 4)` exactly. The retained endpoint is always `a=i`. This differs from vertex-major ordering and must not be silently changed because sequential rewiring depends on visit order. Abort on an unexpectedly absent original edge. This tie-breaking is a **new implementation convention**, not something asserted by [Olfati-Saber, Eq. (8)](https://doi.org/10.1109/ACC.2005.1470321).

No loops or parallel edges are introduced, \(|E(G_g)|=|E_0(n)|\), and vertex degrees may vary. Graphs are **not** conditioned on connectivity or another measured property. The induced Watts–Strogatz distribution is generally **not uniform over all simple graphs** with the same number of edges; uniformity here refers to each eligible rewiring target conditional on the current graph. At \(\beta=0\), \(G_g=C(n,2)\); for \(n\le5\), rewiring cannot alter the complete graph.

**Tracking.** A graph record is indexed by \((n,\beta,g)\) and contains the graph-generation seed/key, rewiring-law version, ordered original-edge convention, canonical sorted edge list, and edge-list digest. The original graph and generated graph are both reproducible from this record and the algorithm versions.

**Sampling.** For fixed \((n,\beta)\), graph trials \(G_0,\ldots,G_{G_*-1}\) are independent draws from \(\mu_{n,\beta}\) in the ideal ensemble. No graph is selected or rejected on the basis of its code properties. For any graph statistic \(H\), the target is \(\mathbb E_{G\sim\mu_{n,\beta}}[H(G)]\).

## 3. Level 2: Clifford gate sequences conditional on a graph

Fix a sampled graph \(G_g=(V,E_g)\). At every monitored timestep \(t\ge1\), apply an **ordered, sequential** block of \(h(n)\) two-qubit Clifford gates. For every integer \(n\ge1\), the number of gates in each monitored timestep is fixed to \(h(n)=\lfloor n/2\rfloor\). This includes odd system sizes. An edge set may be empty only when \(n=1\), in which case \(h(1)=0\).

For \(a=1,\ldots,h(n)\), independently draw
\[
 e_{g,c,t,a}\sim\operatorname{Uniform}(E_g),\qquad
 Q_{g,c,t,a}\sim\operatorname{Uniform}(\mathrm{Cliff}_2/\mathrm{U}(1)),
\]
where \(|\mathrm{Cliff}_2/\mathrm{U}(1)|=11520\), by [König–Smolin, Eq. (1)](https://arxiv.org/pdf/1406.2170). The quotient by the two-qubit Pauli group has \(720\) elements [König–Smolin, Eqs. (2)–(4)](https://arxiv.org/pdf/1406.2170). Every such phase-free symplectic action has exactly 16 Clifford preimages, so uniform sampling of 720 symplectic elements would give the same **phase-free observable distribution**, but the native simulator still samples 11,520 projective Clifford indices. Draws are independent over gate position and timestep conditional on \(G_g\). Apply \(Q_{g,c,t,a}\) to the two endpoints of \(e_{g,c,t,a}\), **in sampled order**. Repeated edges and intersecting supports are allowed.

For every \(e\in E_g\),
\[
 \Pr(e_{g,c,t,a}=e\mid G_g)=\frac1{|E_g|}.
\]
This is uniform *edge* sampling, not uniform *vertex* sampling; the probability of a qubit being an endpoint is proportional to its degree in \(G_g\). For \(h(n)=0\), no edge draw is required.

The full gate-sequence realization \(U_{g,c}\) is the ordered gate sequence through \(T_{\max}\). SparseGF2 generates successive layers from one **trajectory-specific xoshiro256\*\*** stream, not separately seeded per-layer streams. The public `spec.circuit(trajectory=t, seed=s)` export and `result.replay(...)` permit regeneration. Prefixes are invariant under increasing the horizon, but directly regenerating layer \(t\) alone requires replaying the stream from the start (or an explicitly versioned state checkpoint); do **not** assume native independently addressable layer seeds.

**Tracking.** Record the gate-sequence identifier \((n,\beta,g,c)\) for `gate_mode=shared` or \((n,\beta,g,c,p)\) for `gate_mode=independent`, graph digest, native run-seed and stream contract, Clifford enumeration/version, and local endpoint-order convention. Per-layer records contain \(t\), the trajectory identity and, optionally, the ordered edge/Clifford indices and a digest. Store the native run seed and `RANDOMNESS_CONTRACT` version, not fabricated per-layer keys. Independently addressable layer seeds would require a deliberate new backend randomness contract and must not be silently substituted for the current engine.

**Sampling.** For a fixed graph, distinct \(c\) labels represent independent gate-sequence trials in the ideal ensemble. Gate sequences from separate graphs may be sampled independently, but observations from two circuits on the same graph are not unconditional independent draws from the full graph ensemble.

## 4. Level 3: measurements and Born outcomes conditional on the circuit

Fix \((n,\beta,g,c,p,r)\). After the sequential gate block at timestep \(t\), independently draw measurement indicators
\[
 M_{g,c,p,r,t,i}\sim\operatorname{Bernoulli}(p),\qquad i\in V.
\]
Each selected qubit \(Q_i\) is measured in the \(Z\) basis. Process selected qubits in ascending site order. If the current conditioned state immediately before the measurement is \(|\varphi\rangle\), the probability of outcome \(z\in\{+1,-1\}\) is
\[
 \Pr(O_{t,i}=z\mid |\varphi\rangle,M_{t,i}=1)
 =\langle\varphi|\frac{I+zZ_{Q_i}}2|\varphi\rangle.
\]
The state is updated to the corresponding normalized post-measurement state. These outcome probabilities depend on prior gates and measurement outcomes; Born outcomes are **not** independent fair signs. In the **actual phase-free simulation**, the choice of outcome sign is omitted and the updated unsigned stabilizer subspace is exact for the observables studied. For a stabilizer state, a \(Z\) measurement is either deterministic or has probabilities \((1/2,1/2)\) [Li–Fisher, Eqs. (9)–(11)](https://arxiv.org/pdf/2007.03822). Since this study uses phase-free observables and has **no outcome-conditioned feedback**, its production simulator may update just the binary stabilizer subspace and **does not need to sample measurement outcome signs**. The optional outcome-aware replay mode samples signs according to the displayed Born law. This is an exact specialization of the stabilizer measurement rules, not an approximation.

One monitored timestep consists of **all** \(h(n)\) ordered gates followed by **all** selected measurements. The graph is held fixed throughout the trajectory.

**Tracking.** Each trajectory key \((n,\beta,g,c,p,r)\) identifies a graph, a gate stream, a measurement-rate value and a measurement-mask stream; the coupling modes determine which underlying random draws are shared across rates. Masks are generated from the native per-trajectory stream; store the exact sampled mask or export the exact circuit for audit. In phase-free mode there is no outcome seed, outcome record, or signed quantum trajectory to reconstruct. An optional **separate** signed simulation is required if outcome histories become an observable. Reference qubits are never acted on.

**Sampling.** Conditional on a fixed gate sequence, measurement-mask trials \(r\) are independent in the ideal ensemble; the unobserved Born outcomes are marginalized in the phase-free implementation. For each \(p\), the per-site inclusion probability is exactly \(p\) in the mathematical law. Sharing random numbers *between different values of* \(p\) is an optional variance-reduction coupling, not a change of these marginal distributions (§9).

## 5. Level 4: stabilizer state, code, and snapshots

Initialize the physical and reference registers in
\[
 |\Psi_0\rangle=\bigotimes_{i=0}^{n-1}|\Phi^+\rangle_{Q_iR_i},
 \qquad |\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt2},
\]
with stabilizer generators \(X_{Q_i}X_{R_i}\) and \(Z_{Q_i}Z_{R_i}\) for every \(i\in V\). Let \(|\Psi_t\rangle\) be the normalized conditioned joint stabilizer state after \(t\) monitored timesteps, and let \(\mathsf S_t\) be its signed stabilizer group on \(2n\) qubits.

The induced physical stabilizer group and code parameters are
\[
 \mathsf S_t^{\mathrm{phys}}=
 \{P\in\mathcal P_n:P\otimes I_R\in\mathsf S_t\},
 \quad r_t=\operatorname{rank}\mathsf S_t^{\mathrm{phys}},
 \quad k_t=n-r_t.
\]
The reduced physical state is the maximally mixed state on the induced \([[n,k_t]]\) stabilizer code \(\mathcal C_t\), consistent with the code-state and entropy formulas in [Li–Fisher, Eqs. (5)–(7)](https://arxiv.org/pdf/2007.03822):
\[
 \operatorname{Tr}_R|\Psi_t\rangle\langle\Psi_t|
 =2^{-k_t}\Pi_{\mathcal C_t}.
\]
At \(t=0\), \(r_0=0\) and \(k_0=n\).

**Tracking.** Each requested \(t\in\mathcal T(n,\beta)\) produces a snapshot keyed by \((n,\beta,g,c,p,r,t)\). Store the phase-free \(2n\)-qubit stabilizer row space (or a durable reference), a canonical binary-subspace digest, \(r_t\), \(k_t\), and the algorithm versions. For the present observable catalog, sampling Born outcomes and saving signed generators are optional; phases can be generated in a separate outcome-aware mode. A backend-dependent raw tableau digest need not match across implementations; the canonical phase-free row-space digest must.

**Sampling.** Snapshots at different depths of one trajectory are correlated. A snapshot at \(t\) is a marginal of the same infinite-prefix trajectory and is not a newly sampled circuit.


### 5.1 Phase-free tableau and phase-free observables

For each snapshot time \(t\), let \(\widetilde{\mathsf S}_t\) denote the **phase-free tableau data**, i.e. the binary symplectic representation of the conditioned joint stabilizer group with generator signs removed. Equivalently, \(\widetilde{\mathsf S}_t\) specifies the stabilizer subgroup modulo \(\pm 1\) phases. Any observable that depends only on \(\widetilde{\mathsf S}_t\) is called a **phase-free tableau observable**.

A **trajectory observable** is any observable computed from the family of phase-free snapshots along one trajectory,
\[
 \widetilde{\mathsf S}_{0:T}:=\bigl(\widetilde{\mathsf S}_t\bigr)_{t\in\mathcal T(n,\beta)}.
\]
A snapshot observable uses one \(\widetilde{\mathsf S}_t\); a trajectory observable uses multiple depths from the same trajectory.

For phase-free observables, the relevant persistent object is therefore the dictionary
\[
 \mathcal D_{n,\beta,g,c,p,r}
 :=\{t\mapsto \widetilde{\mathsf S}_t: t\in\mathcal T(n,\beta)\},
\]
keyed by the trajectory identity \((n,\beta,g,c,p,r)\). Observable values may be stored, or recomputed later from \(\mathcal D_{n,\beta,g,c,p,r}\) without resampling the circuit.

### 5.2 Observable catalog: logical-qubit count and first-passage times

The phase-free physical stabilizer rank, surviving logical-qubit count, and code rate are
\[
 r_t=\dim W_t^{\mathrm{phys}},\qquad k_t=n-r_t,\qquad R_t=\frac{k_t}{n}.
\]
Here \(R_t\) denotes **code rate**, not the reference register \(R\); use \(k_t/n\) in ambiguous expressions. The stabilizer-code convention follows [Li–Fisher, Eqs. (5)–(7)](https://arxiv.org/pdf/2007.03822). For local Pauli measurement dynamics without feedback, \(k_t\) is an integer nonincreasing function of completed layer depth \(t\) [Li–Fisher, Eqs. (9)–(11)](https://arxiv.org/pdf/2007.03822).

For \(\ell\in\{0,\ldots,n\}\), define the **exact first-passage depth**
\[
 \boxed{\tau_\ell=\min\{t\in\mathbb Z_{\ge0}: k_t\le\ell\}.}
\]
This is a random variable that can exceed the chosen simulation horizon. During simulation, record the first completed layer satisfying each threshold; if no crossing is observed through \(T_{\max}\), store `event_observed=false` and `censor_depth=T_max`, **not** an observed value \(+\infty\). For \(\ell=n\), \(\tau_n=0\). In particular,
\[
 \tau_1=\min\{t:k_t\le1\},\qquad \tau_0=\min\{t:k_t=0\}.
\]
The first means *at most one* logical qubit remains, not necessarily exactly one; \(\tau_0\) is the full-system purification time. Neither is the single-reference-qubit purification protocol of [Hashizume–Bentsen–Daley, Sec. V, Fig. 4 and Eq. (5)](https://strathprints.strath.ac.uk/79723/7/Hashizume_etal_PRR_2022_Measurement_induced_phase_transitionQ_in_sparse_nonlocal.pdf), which initializes only one encoded reference after a scrambling stage.

For comparison with sparse stored checkpoints one can define the **checkpoint upper-bound time**
\[
 \widehat\tau^{\mathrm{chk}}_\ell=\min\{t\in\mathcal T(n,\beta):k_t\le\ell\}.
\]
Whenever both are observed, \(\tau_\ell\le\widehat\tau^{\mathrm{chk}}_\ell\). All reported primary purification-time analyses use the **exact online** \(\tau_\ell\), never silently substitute the checkpoint statistic. Since \(k_t\) is nonincreasing, recording the \(n+1\) threshold-crossing entries online is sufficient for every \(\ell\), regardless of the checkpoint schedule. The complete per-layer \(k_t\) trace is also retained in compressed run-length or change-point form.

### 5.3 Tableau dictionary and observable storage

For each trajectory key \(\xi=(n,\beta,g,c,p,r)\), store the **checkpoint tableau dictionary**
\[
 \mathcal D_\xi=\{t\mapsto\widetilde{\mathsf S}_t:t\in\mathcal T(n,\beta)\},
\]
containing the full \(2n\)-qubit joint phase-free stabilizer row space at each checkpoint. Separately retain \((t,k_t)\) at all layers (losslessly compressed), \(\{(\ell,\tau_\ell,\mathrm{event\_observed},\mathrm{censor\_depth})\}\), and exact trajectory/RNG/version metadata.

Every phase-free **snapshot** observable can be recomputed exactly from a stored tableau at the same depth; any phase-free observable at an *unsaved* depth requires deterministic replay from a preceding tableau or from initialization. The seed by itself does not replace an existing checkpoint; it enables exact replay using the frozen algorithms. Retain the original tableaus and trial-level data when new observables are added, instead of overwriting existing values. References and physical subsystems must both be present in stored joint row-space data.

### 5.4 Reference-information criterion for a region

For a trajectory at depth \(t\), let \(W_t^{\mathrm{phys}}\subseteq\mathbb F_2^{2n}\) be the row space of the **phase-free** physical stabilizer generators, and set
\[
 r_t(A)=\dim\bigl\{w\in W_t^{\mathrm{phys}}:\operatorname{supp}(w)\subseteq A\bigr\},
 \qquad k_t=n-r_t(V).
\]
For \(A\subseteq V\) and \(B=V\setminus A\), the stabilizer entropy identity [Gullans–Huse, Appendix E, Eqs. (E1)–(E3)](https://journals.aps.org/prx/pdf/10.1103/PhysRevX.10.041020) gives
\[
 S_t(A)=|A|-r_t(A),\qquad S_t(B)=|B|-r_t(B),
\]
and hence
\[
 \boxed{I_t(A:R)= k_t+2|A|-n-r_t(A)+r_t(B).}
\]
All entropies and mutual informations are measured in bits. Define
\[
 f_t(A)=\mathbf 1\{I_t(A:R)>0\}.
\]
The region \(A\) is an exactly correctable **erasure** if and only if \(f_t(A)=0\), by [Li–Fisher, Theorem 1 and Eqs. (12)–(14)](https://arxiv.org/pdf/2007.03822). The binary-rank expression above is **derived here** by substituting the stabilizer entropy formula, not copied from that source. The indicator is monotone under inclusion: \(A\subseteq B\Rightarrow f_t(A)\le f_t(B)\).

For an independent-row generator matrix \(H_t\) of \(W_t^{\mathrm{phys}}\), \(r_t(A)=\operatorname{rank}(H_t)-\operatorname{rank}(H_t|_{V\setminus A})\); here the restriction retains both \(X\) and \(Z\) columns of each site. This specifies a phase-free implementation using binary ranks, without enumerating Pauli operators.

### 5.5 Original-ring contiguous distance (baseline)

Retain the original cyclic ordering \(V=\{0,\ldots,n-1\}\) for every \(\beta\). For \(1\le m\le n\), let
\[
 A_{a,m}=\{(a+j)\bmod n:0\le j<m\},\qquad a\in V.
\]
Define
\[
 d_{\mathrm{cyc}}(t)=
 \begin{cases}
 0,&k_t=0,\\
 \min\{m\in\{1,\ldots,n\}:\exists a\in V,\ f_t(A_{a,m})=1\},&k_t>0.
 \end{cases}
\]
For fixed \(k_t>0\), the minimum exists because \(I_t(V:R)=2k_t>0\). This is the first-failing contiguous-region definition in [Li–Fisher, Sec. II D, Theorem 1 and Eqs. (12)–(14)](https://arxiv.org/pdf/2007.03822), with an **analysis-only zero extension** when \(k_t=0\). SparseGF2's native `contiguous_distance` returns `NaN` in that case; retain the raw value and a `has_code` indicator, and use the zero extension only when explicitly analyzing unconditional peak traces. At \(\beta=0\), this is the ring-contiguous erasure distance; at \(\beta>0\), it measures susceptibility to ring-localized erasure despite the rewired interaction graph. It is **not** a graph-intrinsic metric.

### 5.6 Graph-connected erasure distance (candidate)

For an interaction-graph realization \(G=(V,E)\), let
\[
 \mathcal C_m(G)=\{A\subseteq V:|A|=m,\ G[A]\text{ is connected}\}.
\]
Define
\[
 d_{\mathrm{conn}}(G,t)=
 \begin{cases}
 0,&k_t=0,\\
 \min\{|A|: A\subseteq V,\ G[A]\text{ connected},\ f_t(A)=1\},&k_t>0,
 \end{cases}
\]
with \(+\infty\) if no connected failure set exists. On \(C(n,2)\), connected regions need not be cyclic intervals (e.g. \(\{0,2,4\}\) is connected for \(n\ge7\) but not contiguous in cyclic order). Therefore \(d_{\mathrm{conn}}(C(n,2),t)\) need not equal \(d_{\mathrm{cyc}}(t)\). Exhaustive computation can require exponentially many connected subsets. Uniform sampling of connected subsets of fixed size cannot, in general, be obtained by naive random connected growth: growth histories introduce sampling multiplicities. This is a secondary geometric observable, not the primary recommended large-\(n\) statistic.

### 5.7 Uniform and geometry-correlated erasures

Let \(\Omega_{n,m}=\{A\subseteq V:|A|=m\}\), \(b_G(A)=|\partial_G A|\), and \(\partial_G A=\{\{i,j\}\in E(G):i\in A,j\notin A\}\). Define the fixed-cardinality **region distribution**
\[
 Q_{G,J}(A\mid m)=\frac{\mathbf1_{\{|A|=m\}}e^{-Jb_G(A)}}{Z_{G,m}(J)},
 \qquad Z_{G,m}(J)=\sum_{B\in\Omega_{n,m}}e^{-Jb_G(B)}, \quad J\ge0.
\]
This \(Q\) is a probability law **on subsets**, not a failure probability. For the phase-free code \(\mathcal C_t\), define its erasure-failure indicator \(F_{\mathcal C_t}(A)=\mathbf1\{I_t(A:R)>0\}\) and the fixed-size **failure probability**
\[
 \boxed{P_{\mathcal C_t,G,J}(m)=\sum_{A\in\Omega_{n,m}}Q_{G,J}(A\mid m) F_{\mathcal C_t}(A)}.
\]
These symbols match [SparseGF2's correlated-erasure definitions](https://github.com/sjbevins/SparseGF2/blob/main/docs/observables/correlated-erasure.md), §1.1–1.3. In particular, at \(J=0\), \(Q_{G,0}(A\mid m)=\binom nm^{-1}\) and \(P_{\mathcal C_t,G,0}(m)=P_{\mathcal C_t}(m)\) **exactly**, the standard uniform fixed-cardinality erasure law. Positive \(J\) favors low-boundary regions; as \(J\to\infty\), the law becomes uniform over minimizing-cut sets at that size, which need not be graph-connected.

The mean leaked information at fixed size and geometry bias is
\[
 \bar\ell_{\mathcal C_t,G,J}(m)=\mathbb E_{A\sim Q_{G,J}(\cdot\mid m)}[I_t(A:R)].
\]
For any \(\varepsilon\in[0,1]\), the **binomially mixed channel failure probability** is
\[
 \Phi_{\mathcal C_t,G,J}(\varepsilon)=\sum_{m=0}^{n}\binom nm \varepsilon^m(1-\varepsilon)^{n-m}P_{\mathcal C_t,G,J}(m).
\]
At \(J=0\) this is the usual IID single-qubit erasure channel; \(\varepsilon\) (erasure probability) must not be confused with monitoring rate \(p\). This construction and its verified implementations are described in [SparseGF2, correlated-erasure guide §§1 and 7](https://github.com/sjbevins/SparseGF2/blob/main/docs/observables/correlated-erasure.md); it is a **defined sampling model**, not a formula attributed to the MIPT papers.

Introduce the **integrated first-failure robustness statistic** (study-specific, not the standard code distance):
\[
 D_{\mathcal C_t,G}(J)=
 \begin{cases}0,&k_t=0,\\
 \displaystyle\sum_{m=0}^{n-1}[1-P_{\mathcal C_t,G,J}(m)],&k_t>0.\end{cases}
\]
For \(J=0\), if a uniformly random ordering of \(V\) is erased in its prefix order and \(M_\pi\) is the first failing prefix size, then \(\boxed{D_{\mathcal C_t,G}(0)=\mathbb E_\pi[M_\pi]}\). It can be computed from SparseGF2's **existing** `mipt.erasure_first_failure` probe. For \(J>0\), the \(m\)-distributions are not guaranteed to be marginals of a single nested erasure process; evaluate the fixed-size failure profile through `mipt.correlated_erasure` and sum it. This is a weighted integrated correctability score, **not** generally an expected first-failure depth at \(J>0\).

For uniform-sampled regions the identity
\[
 P_{\mathcal C_t,G,J}(m)=
 \frac{\mathbb E_{A\sim Q_{G,0}(\cdot\mid m)}[e^{-J b_G(A)}F_{\mathcal C_t}(A)]}
 {\mathbb E_{A\sim Q_{G,0}(\cdot\mid m)}[e^{-J b_G(A)}]}
\]
is exact. Finite-sample self-normalized reweighting is generally biased and can have negligible effective sample size for large \(J\); use the existing SparseGF2 exact samplers where applicable, or its replica-exchange chain with mixing diagnostics and independent-chain checks. A single set of \(J\)-coupled samples must not be treated as independent ensembles.

**Recording.** At each probed depth store the code snapshot, \(P(m,J)\) estimates and their sampling method/uncertainty, \(\bar\ell(m,J)\) when requested, \(D(J)\), and the full region-sampling configuration. The study first uses \(d_{\mathrm{cyc}}\) and \(D(0)\); \(J>0\) is a separately identified geometry-response analysis.

### 5.8 Exact rate–robustness identities and peak observables

For every pure conditioned joint system–reference state and \(A\subseteq V\), the complementary region \(B=V\setminus A\) satisfies
\[
 \boxed{I_t(A:R)+I_t(B:R)=2k_t.}
\]
Because \(|\partial_G A|=|\partial_G B|\), taking complementary random regions under \(Q_{G,J}(\cdot\mid m)\) yields
\[
 \boxed{\bar\ell_{\mathcal C_t,G,J}(m)+\bar\ell_{\mathcal C_t,G,J}(n-m)=2k_t.}
\]
If \(k_t>0\), at least one complementary region leaks reference information, so
\[
 P_{\mathcal C_t,G,J}(m)+P_{\mathcal C_t,G,J}(n-m)\ge1,
 \qquad  D_{\mathcal C_t,G}(J)\le\frac{n+1}{2}.
\]
For even \(n\), \(\bar\ell_{\mathcal C_t,G,J}(n/2)=k_t\) exactly at every \(J\). These identities follow algebraically from purity and [Li–Fisher, Eq. (14) and the cleaning-lemma discussion after Eq. (14)](https://arxiv.org/pdf/2007.03822). The bound on \(D\) follows by summing the complementary failure-probability inequality over \(m=1,\ldots,n-1\), giving \(\sum_{m=0}^{n-1}P_{\mathcal C_t,G,J}(m)\ge(n-1)/2\). These identities are mandatory unit tests; they connect the leakage profile and the logical-qubit count **without** imposing a dynamical finite-size-scaling law.

For any selected nonnegative depth-dependent robustness trace \(d(t)\), including \(d_{\mathrm{cyc}}(t)\), \(d_{\mathrm{conn}}(G,t)\) when finite, or \(D_{\mathcal C_t,G}(J)\), define
\[
 d^*=\max_{t\in\mathcal T_+}d(t),\quad
 \mathcal A=\{t\in\mathcal T_+:d(t)=d^*\},\quad
 T^*=\exp\!\left(\frac1{|\mathcal A|}\sum_{t\in\mathcal A}\ln t\right),
\]
where \(\mathcal T_+=\mathcal T(n,\beta)\setminus\{0\}\). Define \(T^*\) as missing when \(d^*=0\) or the trace is otherwise invalid; this avoids assigning a peak time to a fully purified trajectory. For Monte Carlo estimates of \(D\), exact ties must not be inferred from numerical noise; predeclare a bootstrap-supported plateau/tie rule or a peak-fit procedure and report its sensitivity. \(T^*\) generally lies between checkpoints and is **not itself a checkpoint**, so \(k(T^*)\) is undefined until an interpolation/lookup convention is specified. To test the relationship to \(\tau_1\) without this ambiguity, store the actual maximizing checkpoint set \(\mathcal A\) and its values \(\{k_t:t\in\mathcal A\}\); at any checkpoint \(t\), \(t\ge\tau_1\iff k_t\le1\) because the exact \(\tau_1\) is measured across all layers. The arithmetic mean \(\mathbb E[T^*]\), geometric mean \(\exp(\mathbb E[\ln T^*])\), and \(\operatorname*{argmax}_t\mathbb E[d(t)]\) are different statistics. The log-midpoint is **the analysis convention from the present study**, not a formula claimed from the external references.

Record whether \(d^*\) occurs at \(T_{\max}\); such a peak may not have been observed beyond the simulation horizon. Likewise, first-passage thresholds not reached by \(T_{\max}\) are **right-censored**, not observed infinite purification times. Finite-size-scaling fits must report their treatment of censoring, checkpoint resolution, \(\ell\) (for \(\tau_\ell\)), \(J\), and the sample hierarchy. Comparisons of scaling exponents \((p_c,z,\nu)\) should be done separately for each fixed \(\beta\), with graph-cluster bootstrap or equivalent paired resampling and held-out-size collapse diagnostics. A common exponent set for \(\mathbb E[T^*]\) and \(\mathbb E[\tau_\ell]\) is a hypothesis to test, not a consequence of the complementarity identities. If the fraction of censored events does not vanish with the available horizon, do not report a finite observed-sample mean as the ensemble mean.

### 5.9 Finite-size scaling and rate–robustness comparison

For each fixed \(\beta\) and each observable time \(\theta\in\{\tau_\ell,T^*_{\mathrm{cyc}},T^*_{J}\}\), test the **near-critical scaling hypothesis** with explicit accounting for censored purification times and missing or boundary-limited peaks
\[
 \mathbb E[\theta\mid n,\beta,p]\simeq n^{z_\theta(\beta)}
 F_{\theta,\beta}\bigl((p-p_{c,\theta}(\beta))n^{1/\nu_\theta(\beta)}\bigr).
\]
This is the conventional dynamic finite-size-scaling form of [Hashizume–Bentsen–Daley, Eq. (5)](https://strathprints.strath.ac.uk/79723/7/Hashizume_etal_PRR_2022_Measurement_induced_phase_transitionQ_in_sparse_nonlocal.pdf), **generalized here as an unverified ansatz** to the full-reference \(\tau_\ell\) and robustness peak observables. The different scaling functions are not assumed equal. Do not fit \(\mathbb E[\theta\mid\mathrm{event\ observed}]\) as a surrogate for the unconditional mean when censoring is appreciable: the conditioning changes the target and can bias \(p_c,z,\nu\). Report crossing fractions, observability and censoring by \((n,\beta,p)\), use longer deterministic continuations or an explicitly stated survival-based estimator, and restrict power-law collapse to a validated scaling window. A stronger **shared-criticality hypothesis** asserts the same \(p_c,z,\nu\) across selected observables at fixed \(\beta\), to be tested against independently fitted alternatives with uncertainty. This ansatz does not presume a single \(p_c,z,\nu\) for all \(\beta\); small-world finite-size crossover can introduce an additional geometry scale (for example, the number of rewired edges is of order \(n\beta\) before considering failed rewiring attempts).

For each individual trajectory, preserve \(k_t\), the chosen distance/robustness trace, \(T^*\), its maximizing set \(\mathcal A\), and the checkpoint-indexed joint data \(\{(t,k_t,d_t):t\in\mathcal T\}\). Record \(\tau_\ell\) and event statistics \(\Pr(k_t\le\ell\mid t\in\mathcal A)\). Compare the distribution of \(T^*/\tau_1\) only when \(\tau_1\) is observed and positive; also use the simpler joint ordering indicator computed from the maximizing checkpoints. Analyze conditional \(k_t\)-resolved robustness profiles and \(d_t/n\) where meaningful, in addition to time scaling: two times can share critical exponents while differing systematically in their relative positions.

The exact identity \(\bar\ell(m,J)+\bar\ell(n-m,J)=2k_t\) constrains reference-information sharing, but cannot determine the time of a peak. A peak can arise from the competition between information spreading (increasing protection against a given erasure geometry) and measurement-induced logical-qubit loss (decreasing \(k_t\)), as a **physical hypothesis**. Observing matching exponents alone does not prove this mechanism or equality of the two timescales.

**Status:** Historical contiguous distance and uniform-region failure are established observables; the integrated score \(D(J)\), its peak time, and the associated shared-exponent hypothesis are explicitly defined *new analysis choices*. Their critical scaling must be established from data.

## 6. Ensemble expectations and statistically valid estimators

### 6.1 Conditional sampling laws

All laws below refer to the ideal ensemble. Their joint factorization follows from the stipulated conditional independence of gate and mask draws; the **order of their mathematical expectations is interchangeable**, but their physical operations are not. The conditional Born law is separate from this independence statement.

Fix an experimental condition \((n,\beta,p)\) and a depth \(T\). The parameters \(n,\beta,p,T\) are held fixed and are not ensemble random variables. Assume a fixed original-edge ordering and orientation \(L_n\) for the rewiring procedure of §2.

**Level 1 — Graph.** The interaction graph is distributed according to the probability law induced by the Olfati-Saber sequential rewiring algorithm:
\[
 G\sim\mu^{\mathrm{WS}}_{n,\beta}:=\operatorname{Law}\bigl(\operatorname{Rewire}(C(n,2),\beta;L_n)\bigr).
\]
For a graph \(g\), \(\mu^{\mathrm{WS}}_{n,\beta}(g)\) is the total probability of all rewiring histories that generate \(g\). This is **not** the uniform distribution over all simple graphs with \(|E_0(n)|\) edges.

**Level 2 — Conditional gate sequence.** Given \(G=g\), sample \(h(n)\) ordered edge–Clifford pairs per timestep, independently with replacement. Define
\[
 \lambda_g=\operatorname{Uniform}(E(g)),\qquad
 \gamma=\operatorname{Uniform}(\overline{\mathrm{Cliff}}_2),
\]
where \(\overline{\mathrm{Cliff}}_2=\mathrm{Cliff}_2/\mathrm U(1)\) has 11,520 elements. Writing \(U=((e_{t,a},Q_{t,a}))_{1\le t\le T,1\le a\le h(n)}\) for the *random gate schedule* (not the composite physical unitary), its conditional law is
\[
 \kappa_{g,T}=\bigl(\lambda_g\otimes\gamma\bigr)^{\otimes Th(n)},
 \qquad
 \kappa_{g,T}(u)=\prod_{t=1}^{T}\prod_{a=1}^{h(n)}
 \frac{1}{|E(g)|}\frac{1}{11520}.
\]
The last expression applies to schedules supported on \(E(g)\times\overline{\mathrm{Cliff}}_2\). For \(h(n)=0\), the empty gate schedule is deterministic.

**Level 3 — Measurement locations.** Independently of \(G\) and \(U\), sample \(nT\) Bernoulli indicators \(M_{t,i}\sim\operatorname{Bernoulli}(p)\). For \(m\in\{0,1\}^{T\times n}\),
\[
 \nu_{p,T}(m)=\Pr(M=m\mid p)
 =\prod_{t=1}^{T}\prod_{i=0}^{n-1}p^{m_{t,i}}(1-p)^{1-m_{t,i}}
 =p^{|m|}(1-p)^{nT-|m|}.
\]
At \(p=0\) and \(p=1\), interpret the Bernoulli law directly (the all-zero and all-one patterns respectively). This law does not depend on the gate schedule or the realized graph.

**Level 4 — Measurement outcomes.** Given \((g,u,m)\), the distribution of the outcome history \(O\) is determined by the Born rule. Write \(V_t(g,u)\) for the ordered product of all two-qubit gates at timestep \(t\), acting on physical qubits only. For the measured sites in layer \(t\), let
\[
 P_{m_t,o_t}=\prod_{i:m_{t,i}=1}\frac{I+o_{t,i}Z_{Q_i}}{2},
 \qquad o_{t,i}\in\{-1,+1\}.
\]
The identity applies when the measurement set is empty. Set
\[
 K_{g,u,m,o}(T)=P_{m_T,o_T}V_T\cdots P_{m_1,o_1}V_1.
\]
For the initial \(2n\)-qubit Bell-pair state \(|\Psi_0\rangle\), define
\[
 \pi_{g,u,m}(o)
 =\Pr(O=o\mid G=g,U=u,M=m)
 =\bigl\|(K_{g,u,m,o}(T)\otimes I_R)|\Psi_0\rangle\bigr\|^2.
\]
The set \(\mathcal O(m)\) contains the \(\pm1\) assignments at measured locations; impossible histories have probability zero. The measurement completeness relations imply \(\sum_{o\in\mathcal O(m)}\pi_{g,u,m}(o)=1\). In the Clifford-stabilizer setting, each measurement outcome is deterministic or equiprobable *conditional on the immediately preceding state*; outcomes along a history are not assumed independent.

**Joint law.** The probability mass function of a full trajectory is
\[
 \boxed{\Pr(G=g,U=u,M=m,O=o\mid n,\beta,p,T)
 =\mu^{\mathrm{WS}}_{n,\beta}(g)\,\kappa_{g,T}(u)\,
 \nu_{p,T}(m)\,\pi_{g,u,m}(o).}
\]
Only the first three factors describe randomness prescribed externally by the ensemble; the fourth is the state-dependent quantum trajectory law. The initial state and output tableau are deterministic functions of these sampled objects. These laws specify independent ideal draws; seeded pseudorandom streams approximate them computationally.

### 6.2 Ensemble expectations and order of integration

Let \(F_T=F_T(G,U,M,O)\) be an integrable real-valued observable evaluated at depth \(T\). Its target expectation is
\[
 \boxed{\mu_F(n,\beta,p,T)
 =\mathbb E_{G\sim\mu^{\mathrm{WS}}_{n,\beta}}
  \mathbb E_{U\sim\kappa_{G,T}}
  \mathbb E_{M\sim\nu_{p,T}}
  \mathbb E_{O\sim\pi_{G,U,M}}[F_T].}
\]
The Born-averaged observable at fixed \((g,u,m)\) is
\[
 \overline F_T(g,u,m)
 =\sum_{o\in\mathcal O(m)}\pi_{g,u,m}(o)F_T(g,u,m,o).
\]
Conditional on \(G=g\), the gate schedule and measurement mask are independent:
\[
 \Pr(U=u,M=m\mid G=g,p)=\kappa_{g,T}(u)\nu_{p,T}(m).
\]
Therefore, since these sums are finite,
\[
 \begin{aligned}
 \mathbb E_{U\mid G=g}\mathbb E_{M\mid p}[\overline F_T(g,U,M)]
 &=\sum_{u,m}\kappa_{g,T}(u)\nu_{p,T}(m)\overline F_T(g,u,m)\\
 &=\sum_{m,u}\nu_{p,T}(m)\kappa_{g,T}(u)\overline F_T(g,u,m)\\
 &=\mathbb E_{M\mid p}\mathbb E_{U\mid G=g}[\overline F_T(g,U,M)].
 \end{aligned}
\]
Thus **the gate-schedule and measurement-mask averages may be interchanged while keeping the Born average conditioned on both**. This is the finite-space case of Fubini's theorem, not an assertion that the physical unitary and projection operators commute. Without independent sampling (e.g. adaptive gate choices responding to measurement outcomes), the simple product factorization does not apply; one must instead use the actual conditional joint law.

**Outcome-independent stabilizer observables.** Fix \((g,u,m)\) and assume Clifford gates, Pauli measurements, and no outcome-dependent feedback. The binary (phase-free) stabilizer subspace at every timestep is independent of the realized Born outcomes. Clifford conjugation acts on Pauli vectors independently of signs; a Pauli measurement changes the generator subspace according to commutation relations and changes signs according to its result. Induction over all operations proves that the subspace does not depend on \(o\). Hence quantities depending only on this subspace — including \(k_T\), subsystem stabilizer entropies, and erasure correctability — satisfy
\[
 \overline F_T(g,u,m)=F_T(g,u,m,o)
 \quad\text{for every }o\text{ with }\pi_{g,u,m}(o)>0.
\]
The *signed* stabilizer state and code projector generally **do** depend on the outcomes; outcome histories must be sampled according to the Born rule when full conditioned tableaus are required.

**Sampling across experimental conditions.** Different rates \(p\) can share the underlying graph, gate schedule, and optionally per-site uniform variables for measurement-mask generation. This creates dependence between estimates at distinct \(p\) but does not change the marginal law of any fixed-\(p\) ensemble. Post-selection or adaptive stopping changes the relevant sampling/conditioning law and requires an explicitly defined conditional estimator. Values at distinct depths from one trajectory are jointly distributed, not independent.

### 6.3 Sampling allocation and estimator

Let \(G_*\ge1\) denote the number of graph trials. For graph \(g\), sample \(C_g\ge1\) gate-sequence trials. For each \((g,c,p)\), sample \(R_{g,c,p}\ge1\) measurement trajectories. Denote their observable values by \(F_{g,c,p,r}(t)\). Define
\[
 \widehat f_{g,c,p}(t)
 =\frac1{R_{g,c,p}}\sum_{r=0}^{R_{g,c,p}-1}F_{g,c,p,r}(t),
\]
\[
 \widehat f_{g,p}(t)
 =\frac1{C_g}\sum_{c=0}^{C_g-1}\widehat f_{g,c,p}(t),
 \qquad
 \boxed{\widehat\mu_F(n,\beta,p,t)
 =\frac1{G_*}\sum_{g=0}^{G_*-1}\widehat f_{g,p}(t).}
\]
With independent graph trials and a nonadaptive sampling allocation fixed independently of the sampled observable values, this estimator is unbiased for \(\mu_F\) under the stated ensemble distribution. In particular, **each graph receives equal total weight** even when its circuit or measurement-trajectory counts differ. Simply averaging all leaf trajectories instead weights graphs in proportion to the number of leaf observations assigned to them.

If all \(C_g=C\) and all \(R_{g,c,p}=R\), the nested estimator reduces to the arithmetic mean over the \(G_*CR\) values, although those values remain correlated within graph clusters.

### 6.4 Uncertainty and comparisons

Independent graph trials are the outer statistical units. With \(G_*>1\), a graph-cluster standard error is
\[
 \widehat{\operatorname{SE}}(\widehat\mu_F)
 =\sqrt{\frac1{G_*(G_*-1)}
 \sum_{g=0}^{G_*-1}\left(\widehat f_{g,p}(t)-\widehat\mu_F\right)^2}.
\]
This captures variation of the entire graph-level estimator, including nested Monte Carlo noise. It is not estimated correctly by treating every circuit or checkpoint as an independent graph draw. For rate comparisons on shared graphs, form differences **within each graph** first and estimate uncertainty from the graph-level paired differences. If only one graph is sampled, the unconditional graph-ensemble uncertainty cannot be inferred from circuit repeats alone.

Averaging across \(\beta\), \(p\), or \(t\) requires a separately specified scientific target and weighting measure. For nonlinear observables, average the computed observable values, not a nonlinear function of already-averaged tableaus or underlying quantities.

## 7. RNG protocol and reproducible sampling

### 7.1 Mathematical and computational randomness

The ensemble laws in §6 are ideal probability distributions. A seeded pseudorandom simulator realizes a deterministic, finite-precision approximation to those laws; no finite deterministic sequence can be proved independent solely by passing empirical randomness tests. Every reported run must record both the scientific sampling law and its concrete RNG realization.

**Production authority:** the [`sparsegf2.mipt`](https://github.com/sjbevins/SparseGF2/blob/main/src/sparsegf2/mipt.py) **native randomness contract `RANDOMNESS_CONTRACT = 2`** (as inspected in the documented v3 development tree), implemented in [`_kernel/rng.py`](https://github.com/sjbevins/SparseGF2/blob/main/src/sparsegf2/_kernel/rng.py) and [`_kernel/ensemble.py`](https://github.com/sjbevins/SparseGF2/blob/main/src/sparsegf2/_kernel/ensemble.py). It uses a xoshiro256** PRNG with deterministic seed/stream derivation, bounded-integer selection and double-precision draws. The graph generator uses this RNG through `graphs.watts_strogatz` (see §2). **The earlier `HMAC-SHA256-addressed-v1` design is not the SparseGF2 native RNG contract and is not used in this study.** Substitution would produce different graphs, gates and measurement masks and must be treated as a *new* experiment/RNG version, not an invisible optimization.

### 7.2 Realization identity and graph/circuit seeding

Use one immutable experiment manifest `experiment_id` with explicit `master_graph_seed`, `master_circuit_seed`, integer graph labels `g`, circuit labels `c`, and run configuration. For a fixed `(n, beta)` choose `graph_seed(g)` by appending `g` to a versioned graph-seed namespace; use `graphs.watts_strogatz(n, k=4, beta=beta, seed=graph_seed(g))`. **Graph seeds are independent of measurement rate `p`**: construct each collection once and reuse it, with canonical sorted edge lists and hashes.

SparseGF2's graph-collection mode with `circuits_per_graph=C` maps its native global trajectory index to the pair `(g,c)` as
\[
 u=gC+c,\qquad 0\le g<G_*,\quad 0\le c<C.
\]
With the identical graph collection, circuit run seed, schedule, and global index `u`, changing the rate `p` reuses the gates and uniforms used for masks. Native `spec.run(trajectories=..., seed=..., first_trajectory=...)`, `result.replay`, and `spec.circuit(trajectory=u, seed=...)` define exact trajectory identities. The primary campaign uses **one measurement realization per `(g,c,p)`, so `r=0`**. Independent `r>0` measurement-mask replicates *on the same gates* are **not an existing native `mipt.MonitoredCircuit` operation** and must not be represented as such; adding them requires a separately versioned generator extension.

Store `(n, beta_token, p_token, g, c, r=0, u, graph_digest, manifest_digest, run_seed, code_commit, SparseGF2_version, RANDOMNESS_CONTRACT)` for every trajectory. Use exact decimal/rational tokens as identifiers, but also record the actual IEEE-754 float passed to the simulator: mathematically exact Bernoulli probabilities cannot in general be represented by a finite binary floating-point threshold.

### 7.3 Native stream semantics

The trajectory PRNG stream generates gate indices and measurement-mask uniforms **in sequence**, as described in [`_kernel/ensemble.py::gen_layer`](https://github.com/sjbevins/SparseGF2/blob/main/src/sparsegf2/_kernel/ensemble.py): each of the `n//2` gates consumes an edge draw and a Clifford draw (plus optional orientation draw), followed by one mask uniform for every site. Set `random_orientation=False`, unweighted edges, and disable optional scramble (`scramble_gates=0`, `scramble_matchings=0`, `scramble_layers=0`) to match the ensemble exactly.

For two rates `p_1 <= p_2`, common random uniforms guarantee sitewise nested masks `M(p_1) <= M(p_2)` for fixed `(g,c,t,i)`. This is **a coupling across experimental conditions**, not independence of samples at distinct `p`. Within one fixed rate, the intended measurement masks remain independent across sites and layers in the ideal ensemble; the simulator approximates this with its PRNG.

A run ending at layer `T_1` and an otherwise identical run ending at `T_2>=T_1` must produce the same circuit prefix through `T_1`. Changing probe schedules, worker counts, batch partitions, or adding unrelated trial indices must not change that prefix. **The current engine does not expose an independently keyed RNG seed for an arbitrary layer**: a layer is reproducible as part of a trajectory, by replay/export, or by restoring a serialized PRNG state. A future random-access layer generator would require an explicit new randomness contract and reference comparisons, not just a change of seed syntax.

Sampled erasure probes use **separate, keyed probe streams** under `RANDOMNESS_CONTRACT=2`, addressed by trajectory index, probe name, checkpoint depth, and, for correlated erasures, region size. Changing the number of erasure samples or moving a probe does not affect circuit randomness. Preserve probe names and versioned settings when reproducing sampled erasure estimates.

### 7.4 Randomness validation and acceptance gates

| Level | Required deterministic checks | Required distributional checks |
|---|---|---|
| Graph | Same seed produces same ordered/sorted edge list; `beta=0` equals `C(n,2)`; no loop, duplicate or edge-count change; native generator equals independently implemented reference on fixed test seeds | Rewire decisions have rate `beta`, eligible endpoint uniform conditional on current graph; sample graph statistics within Monte Carlo uncertainty |
| Gate schedule | Same `(spec, run_seed, u)` yields identical `spec.circuit` export; order and endpoints correct; prefix/horizon and parallel invariance | `n//2` independent with-replacement edge draws with uniform edge probabilities; uniform projective Clifford indices/720 symplectic classes |
| Masks | Exact prefix and checkpoint invariance; measured masks nested across rates | Bernoulli frequencies, site/time correlation and gate/mask independence diagnostics under fixed graph |
| Phase-free tableau | Canonical binary row-space digest identical after replay; measurement update preserves commuting independent stabilizers; no signed outcome dependence | Small-`n` checks against signed Stim/SparseGF2 after discarding signs; compare exact code/erasure quantities |
| Erasure sampling | Probe result invariant under unrelated probes; changing `samples` preserves draw prefix under a pinned contract | Uniform fixed-size subsets at `J=0`; exact brute-force comparisons for small graphs; chain effective sample sizes, mixing and independent chains for `J>0` |
| Analysis | Nested graph weights, aggregation, censoring and peak flags invariant under chunking; rate-coupled pairs stay aligned | Confidence-interval coverage of synthetic controls; bootstrap over independent graphs; region-sampling uncertainty propagated |

The test suite must include the upstream repository tests (`python -m pytest`, with Stim installed) and an independent small-system implementation of the defining formulas. Save conformance traces, seeds, versions, test counts, tolerances and failing examples; **never** claim these statistical tests prove formal independence or unbiasedness of the PRNG as a mathematical object.

## 8. Data model and lineage

Each record has a stable identity and a parent reference. Numeric parameter values are encoded canonically; grid positions are not identifiers.

| Record | Primary key | Required contents | Parent |
|---|---|---|---|
| `experiment` | `experiment_id` | \(n\) grid, \(\beta\) grid, \(\mathcal P(n,\beta)\), \(\mathcal T(n,\beta)\), trial allocation, graph/circuit seeds, native RNG contract and algorithm versions | — |
| `graph` | \((n,\beta,g)\) | Graph seed, graph-generator version, offset-major original-edge order, sorted edges and digest | Experiment |
| `gate_sequence` | \((n,\beta,g,c[,p])\) according to coupling mode | Native circuit run seed and global trajectory index, graph digest, \(h(n)\), gate coupling and native Clifford enumeration/version | Graph |
| `trajectory` | \((n,\beta,g,c,p,r)\) | Gate-sequence reference, exact rate token and float64, native seed/contract, observed horizon, completion/censoring status; phase-free mode | Gate sequence |
| `layer` | \((n,\beta,g,c,p,r,t)\) | Trajectory seed/index; optional ordered edge/Clifford indices, mask, and phase-free row-space digest; no Born outcomes | Trajectory |
| `snapshot` | \((n,\beta,g,c,p,r,t)\) | Full joint \(2n\)-qubit phase-free row space and RREF digest **only after export extension**; native `code_snapshot` stores physical code, not joint tableau | Trajectory/layer |
| `purification_trace` | `(trajectory_key, dynamics_version)` | Per-layer `code_dimension` vector (or lossless change points), exact first crossings \(\tau_\ell\), censoring flags; full `checkpoints="every_layer"` required | Trajectory |
| `observable_snapshot` | `(snapshot_key, observable_id, observable_version)` | Snapshot-level observable value and any required metadata | Snapshot |
| `observable_trajectory` | `((n, β, g, c, p, r), observable_id, observable_version, observable_parameters)` | Trajectory observable; full maximizing checkpoint set, peak treatment, censoring, depth schedule, and analysis version when applicable | Trajectory / snapshots |
| `region_sample` | `(trajectory_key, t, m, subset_trial)` | Region/probe identifier, subset when saved, sampler seed/contract and geometry, boundary, failure indicator, loss, and chain diagnostics when relevant | Graph / snapshot |
| `geometry_erasure_profile` | `(trajectory_key, t, J, analysis_version)` | \(P_{\mathcal C,G,J}(m)\), \(\bar\ell(m,J)\), \(D(J)\), error estimates, effective sample sizes and method | Region samples |
| `aggregate` | `(n, β, p, observable_id, observable_parameters, analysis_depth_or_horizon, observable_version, estimator_version)` | Graph-wise means, pooled estimate, uncertainty, sample counts, censoring and exclusion policy | Observables |
| `rng_validation` | `(experiment_id, rng_protocol_version, validation_version)` | Native contract/version traces, exact invariant checks, statistical diagnostics, RNG sensitivity, test counts and thresholds | Experiment |

Store the entire *joint* \(2n\)-qubit **phase-free** stabilizer subspace when a full-state checkpoint is requested **only after implementing and testing the export extension described below**. By default, SparseGF2's `code_snapshot()` exports the induced **physical** stabilizer code, sufficient for code-theoretic erasure computations but **not** the full joint tableau. A canonical digest is computed from a row-reduced binary-subspace representation; raw backend tableau buffers can depend on the chosen generator basis. Signed checkpoints may be retained separately for tasks that require phases. Graph and trajectory records must retain the versions needed for bitwise replay. Aggregates do not replace the sample-level records.

### 8.1 SparseGF2 implementation mapping and required extensions

The **maintained** codebase is [sjbevins/SparseGF2](https://github.com/sjbevins/SparseGF2). At the reviewed snapshot the README reports `3.0.0.dev12`, on the path to v3; **pin the exact repository commit, dependency lock and RNG contract before production**. Do not use legacy v2 API examples or reimplement its Clifford simulator. Source references here are *software documentation*, not peer-reviewed equations.

| Study requirement | Existing concrete API | Status |
|---|---|---|
| Graphs \(G_g\sim\mu_{n,\beta}\) | `graphs.watts_strogatz(n, k=4, beta=beta, seed=(graph_master,g))` | Present for `n>=5`, matches offset-major orientation and simple-graph rewiring; validate edge list on golden seeds |
| Graph collection reused across \(p\) | `edges=[...]`, `circuits_per_graph=C`, `result.graph_index` | Present; `u=g*C+c`; never regenerate graph collections for each rate |
| Phase-free 2n-qubit purification initialization | `mipt.MonitoredCircuit(reference="purification")` | Present; ensemble engine **always phase-free**; manual API `Simulator(num_qubits, phases=False)` |
| Fixed-layer sequential gate ensemble | `schedule="random_edges"`, `gates_per_layer=n//2`, `edge_weights=None`, `scramble_gates=0`, `scramble_matchings=0`, `scramble_layers=0` | Present; with replacement; no other pre-scramble; default edge orientation fixed |
| Per-layer \(k_t\) and exact \(\tau_\ell\) | `checkpoints="every_layer"`, `probes=[mipt.code_dimension()]` plus `result.values["code_dimension"]`; `result.purification_layer` records \(\tau_0\) | Present; compute all \(\tau_\ell\) from recorded per-layer k trace. **Sparse global checkpoints would coarsen purification time** |
| Sparse distance/erasure probes | `mipt.contiguous_distance(checkpoints=...)`, `mipt.erasure_first_failure(...)`, `mipt.uncorrectable_region_counts(...)`, `mipt.correlated_erasure(...)` | Present; expensive probes independent schedules; ensure NaN when `k=0` is handled explicitly |
| Physical code saved for retrospective erasure analyses | `mipt.code_snapshot(checkpoints=...)`, `result.code(t,checkpoint_index)`, `StabilizerCode` | Present, sufficient to recompute erasure correctability and region loss at saved depths |
| **Full joint phase-free tableau** at arbitrary checkpoints | No documented equivalent of a joint-tableau snapshot probe in `mipt.MonitoredCircuit` | **Extension required**: serialize exact joint 2n-qubit binary stabilizer row space, qubit order and RREF digest, or replay exported `spec.circuit(...)` with `Simulator(2*n, phases=False)` at specified checkpoints; test against live probes |
| Fixed circuit regeneration | `spec.circuit(trajectory=u, seed=...)`, `result.replay(u, ...)`, `EnsembleResult.save/load`, CLI `provenance.json` | Present for the pinned native RNG contract; a native layer-random-access seed is **not** present |
| Several measurement masks for identical gates | No native independent mask-only replicate selector `r` in compiled ensemble | **Extension required** if `r>0` is used; baseline `r=0` |
| Additional graph-specific postprocessing | `StabilizerCode` code methods, `analysis` package, `erasures` samplers | Present; the new integrated statistic and its peak fitter are analysis code, not new circuit dynamics |

A minimal production configuration is structurally:

```python
from sparsegf2 import graphs, mipt

n, beta, p, T_max, G, C = 64, 0.1, 0.2, 1024, 100, 2
graphs_ = [graphs.watts_strogatz(n, k=4, beta=beta, seed=(17, g)) for g in range(G)]
peak_checkpoints = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
spec = mipt.MonitoredCircuit(
    num_system=n,
    edges=graphs_,
    circuits_per_graph=C,
    layers=T_max,
    measurement_rate=p,
    reference="purification",
    schedule="random_edges",
    gates_per_layer=n // 2,
    random_orientation=False,
    periodic=True,
    checkpoints="every_layer",  # exact tau_ell and k_t
    probes=[
        mipt.code_dimension(),
        mipt.contiguous_distance(checkpoints=peak_checkpoints),
        mipt.erasure_first_failure(samples=100, checkpoints=peak_checkpoints),
        mipt.code_snapshot(checkpoints=peak_checkpoints),
    ],
    stop_when_purified=True,
)
result = spec.run(seed=29)  # graph collections remain fixed at every rate
```

The example illustrates supported public arguments from [SparseGF2's guide](https://github.com/sjbevins/SparseGF2/blob/main/docs/GETTING_STARTED.md), §§3.1–3.9; its seed and grid numbers are **examples**, not adopted science parameters. In the exploratory project, retain the *same* `graphs_` and `seed=29` for every `p` to get paired runs. **Do not represent `mipt.code_snapshot()` as a full joint tableau**. A later joint-snapshot extension should retain the native `RANDOMNESS_CONTRACT`, not silently replace its random circuit generation.

When using `stop_when_purified=True`, the engine fills post-purification code-dimension values with the exact zero, but fills `contiguous_distance` with `NaN`; the post-purification *system state* itself is not evolved, so no unsaved joint-tableau state after stopping can be claimed. If full joint tableau snapshots are required **at every depth even after reference purification**, set `stop_when_purified=False` for that campaign or perform deterministic, separately identified continuations. For code-only observables, early stopping is valid with the recorded zero extension and `has_code` flag.

For long campaigns use `EnsembleResult.save/load` or the chunked CLI; store separate graph collections and a fully resolved JSON manifest alongside each result, including graph/circuit seeds, `p` float, `u`, probe schedules, library commit, `RANDOMNESS_CONTRACT` and hashes. Recompute any derived observable from the saved code/tableau and its versioned extraction algorithm; never conflate a new probe's RNG draws with new graph/circuit samples.

### Validation requirements

1. Graphs have no loops or duplicate edges, and \(|E(G_g)|=|E_0(n)|\); \(\beta=0\) yields \(C(n,2)\).
2. For each fixed graph, sampled gate edges have the uniform-edge law, gates are ordered and sampled with replacement, and \(h(n)=\lfloor n/2\rfloor\) even for odd \(n\).
3. Measurement indicators are independent Bernoulli draws conditional on fixed \(p\) and are independent of gate schedules conditional on \(G\); stabilizer Born outcomes obey their state-dependent conditional probabilities.
4. Only system qubits are acted upon; snapshots satisfy \(k_t=n-r_t\), with \(r_0=0\).
5. Replaying the same identity and versions reproduces graph edges, layer inputs and the phase-free stabilizer row space; in outcome-aware mode it also reproduces Born outcomes and the signed stabilizer group.
6. Changing maximum depth, checkpoint schedule, worker count, or unrelated grid values leaves existing trajectory prefixes unchanged.
7. Each aggregate can be reproduced from its graph-wise, circuit-wise, and measurement-wise underlying samples.
8. Canonical state digests and fixed golden test vectors agree across independent implementations.
9. Statistical validation results, RNG primitive versions, trial addresses, and domain-coupling modes are archived with each accepted run.
10. Native random-edge draws use SparseGF2's `next_below`; masks use `next_double < float(p)`. Verify both against the frozen RNG contract and record float representations. The ideal Bernoulli\((p)\) law is defined mathematically, while actual finite-precision sampling is implementation dependent.
11. Under shared-uniform mask coupling, all sitewise masks are nested across measurement rates, using aligned `(g,c)` and native trajectory index `u`.
12. For every tested subset and complement, \(I_t(A:R)+I_t(V\setminus A:R)=2k_t\); boundary-biased ensemble averages satisfy the identities of §5.8.
13. The observable catalog records missing thresholds, peak-at-boundary flags, and differences between checkpoint peak and interpolated peak summaries.

## 9. Fixed conventions and rationale

All parameters, allocations, seeds, checkpoint depths, exact software versions and derived-observable versions are frozen in a machine-readable experiment manifest. The following conventions are binding unless a new study/protocol version explicitly supersedes them.

| Component | Adopted contract | Mathematical or implementation reason |
|---|---|---|
| Size domain | Graph definition is valid for any integer \(n\ge1\); native SparseGF2 WS generator supports \(n\ge5\); nontrivial WS rewiring requires \(n\ge6\). | The small complete-graph cases are well-defined but have no rewiring candidates. Simulator support is not the same as mathematical definability. |
| Gates per monitored timestep | \(h(n)=\lfloor n/2\rfloor\); sequential, uniform edge draws **with replacement**. | Unambiguous for odd \(n\) and native `schedule="random_edges"` default. |
| Rewiring edge order | Original edges \((i,(i+d)\bmod n)\), **\(d=1,2\) outer loop, \(i=0,\dots,n-1\) inner loop**, retained \(i\). | Matches [`graphs._ring_lattice`](https://github.com/sjbevins/SparseGF2/blob/main/src/sparsegf2/graphs.py) and the conditional rewiring rule of Olfati-Saber [Eq. (8)](https://doi.org/10.1109/ACC.2005.1470321). |
| Gate endpoints | Canonical sorted undirected edge list (low endpoint first); no random orientation (`random_orientation=False`). | Avoids a second unstated random variable and fixes asymmetric gate action. Uniform Clifford distribution makes orientation marginally irrelevant, but it changes seed-level realizations. |
| Clifford distribution | Uniform over 11,520 two-qubit projective Cliffords, with native SparseGF2 catalog indexing in §9.1. | Ensures the physical ensemble is the one specified, even though phase-free observables see only 720 symplectic maps. |
| Initial state | \(n\) Bell pairs with \(n\) references; `reference="purification"`; no pre-scramble. | Exactly the full-reference protocol; not a single-reference or pure-state simulation. |
| Measurements | After all gates, measure each system site in Z independently with rate \(p\); reference never measured. | Matches physical ensemble and engine. In phase-free mode outcome signs are marginalized analytically, not explicitly generated. |
| Gate and mask coupling over \(p\) | Shared gate stream and sitewise threshold uniforms, with identical graph collection and native trajectory index. | Correct marginal laws; paired cross-rate comparisons and nested masks. |
| Replication | Balanced graph count \(G_*\), circuits per graph \(C\), \(r=0\) in native engine. | Distinct graph trials provide outer independent units; native engine has no independent mask-only `r` option. |
| RNG | Native SparseGF2 `RANDOMNESS_CONTRACT=2`, pinned code revision, `xoshiro256**` PRNG and stored seed. | Avoids introducing a second incompatible stream design; deterministic replay from trajectory identity. |
| Checkpoints | \(T_\max\ge1\), requested geometric count \(N_T\). Define \(a_j=\exp(j\ln T_\max/(N_T-1))\) for \(j=0,\ldots,N_T-1\) if \(N_T\ge2\), round to nearest integer with ties upward, clamp, deduplicate, sort and include \(0,1,T_\max\). For \(N_T=1\), include \(0,1,T_\max\). | Deterministic sparse schedule for expensive snapshots; a later change of schedule cannot alter circuit prefixes. |
| First-passage observables | **Global** `checkpoints="every_layer"` and \(k_t\) at all layers, with lossless run-length/change-point compression after capture. | The native engine detects purification only at global checkpoints; exact \(\tau_\ell\) otherwise lost. |
| Purification stopping | `stop_when_purified=True` for code-only campaigns, because erasure/code properties thereafter are exactly zero or undefined; `False` if full joint tableaux are required after purification. | Prevents treating a code snapshot or stopped run as a continued joint state. |
| Native contiguous distance | Store raw `NaN` when \(k=0\); separately compute `zero_extended_distance=0` for a declared unconditional peak statistic. | Do not silently change the ensemble conditional on survival; both targets may be scientifically relevant. |
| Peak depth | Maximize over strictly positive stored checkpoints; log-geometric mean of *exactly tied* maxima only for exact data; stochastic-data plateau rule predeclared; preserve maximizing set and horizon-peak flags. | A log midpoint is not usually a sampled time; exact and stochastic peaks must not be conflated. |
| Primary robustness | Ring-contiguous \(d_\mathrm{cyc}\) plus uniform-region \(P_{\mathcal C}(m)\) and \(D(0)\). | Baseline comparable to earlier work, and graph-independent erasure distribution to isolate the code's effect. |
| Geometry-sensitive follow-up | \(P_{\mathcal C,G,J}(m)\), \(\Phi_{\mathcal C,G,J}(\varepsilon)\), \(D(J)\) for predeclared \(J>0\), mixing and ESS diagnostics. | SparseGF2's implemented graph-boundary sampling; distinct erasure channel from the monitored-circuit rate \(p\). |
| Storage | `EnsembleResult` and `StabilizerCode` snapshots immediately; full joint phase-free tableaus only after verified extension; all files keyed by graph/circuit/depth and code version. | Prevents claiming the existing `code_snapshot` stores reference stabilizers. |
| Rate grid | \(\mathcal P(n,\beta)\) frozen before confirmatory runs; rational/decimal tokens plus actual float bits and exploratory/confirmatory labels. | Controls graph-size crossover window selection and numerical rate provenance. |
| Finite-size scaling | Fit fixed-\(\beta\) models with graph-cluster uncertainty, paired across \(p\), explicit right-censoring and peak-at-horizon policies; test shared vs separate exponents. | The scaling ansatz of Hashizume et al. [Eq. (5)](https://strathprints.strath.ac.uk/79723/7/Hashizume_etal_PRR_2022_Measurement_induced_phase_transitionQ_in_sparse_nonlocal.pdf) is not proved for these new observables. |

### 9.1 Native two-qubit Clifford indexing and phase-free quotient

SparseGF2's [`clifford.py`](https://github.com/sjbevins/SparseGF2/blob/main/src/sparsegf2/clifford.py) fixes the canonical integer index as
\[
 j=16s+b,\qquad s\in\{0,\ldots,719\},\quad b\in\{0,\ldots,15\},
\]
where \(s\) is a symplectic 4-by-4 binary matrix, sorted by its row-major bits as an integer, and \(b\) contains the four signed generator-image bits in the order \((X_0,X_1,Z_0,Z_1)\). Thus the 11,520 projective Cliffords map uniformly, 16-to-1, to 720 phase-free symplectic actions \(\mathrm{Sp}(4,\mathbb F_2)\). The group cardinalities are the standard symplectic/Clifford identities in [König–Smolin, Eqs. (1)–(4)](https://arxiv.org/pdf/1406.2170). **Use this native enumeration**, not an alternative lexicographic signed-Pauli catalog: the latter would change actual seed-to-gate mappings while leaving the abstract uniform distribution unchanged.

### 9.2 Validation acceptance criteria

Require (a) upstream SparseGF2 tests, including Stim comparison; (b) an independently implemented mathematical reference for graph rewiring and small-system phase-free rank/erasure checks; (c) exact replay and parallel invariance tests; (d) cross-rate coupling tests; (e) correct right-censoring for \(\tau_\ell\); (f) complement identities and the finite-size \(D(J)\) bound; (g) saved/reloaded physical-code equality; and, before relying on a joint dictionary, (h) equality of the exported joint row space, its reduced-code stabilizers, and direct simulator probes at all tested depths. Keep benchmark and correctness results distinct; a faster simulator is not automatically an unbiased ensemble simulator.

## References and equation provenance

1. **Olfati-Saber (2005)**, [*Ultrafast Consensus in Small-World Networks*](https://doi.org/10.1109/ACC.2005.1470321), Sec. IV, **Eq. (8)**: sequential edge-rewiring recursion. The paper's target rule is used; the exact ordering/orientation in §9 is a new, explicit convention.
2. **Li and Fisher (2021)**, [*Statistical mechanics of quantum error correcting codes*](https://arxiv.org/pdf/2007.03822), **Eqs. (5)–(7)**: stabilizer code dimension, code state and entropy; **Eqs. (9)–(11)**: measurement stabilizer update; **Theorem 1 and Eqs. (12)–(14)**: subsystem logical information, mutual information and decoupling; Sec. II D and **Theorem 1, Eqs. (12)–(14)**: exact contiguous-region correctability and its mutual-information test. Domain-wall predictions are developed separately in Sec. III; they are not identical to the exact code-distance definition. The fixed-cardinality graph-boundary measure and \(D(J)\) in §5.7 are **new definitions** here, not equations from that paper.
3. **Gullans and Huse (2020)**, [*Dynamical Purification Phase Transition Induced by Quantum Measurements*](https://journals.aps.org/prx/pdf/10.1103/PhysRevX.10.041020), Appendix E, **Eqs. (E1)–(E3)**: stabilizer density matrix, reduced density matrix and subsystem entropy. This paper also establishes the full-system mixed-state purification setting, which differs from a single initially encoded reference.
4. **Hashizume, Bentsen and Daley (2022)**, [*Measurement-induced phase transitions in sparse nonlocal scramblers*](https://strathprints.strath.ac.uk/79723/7/Hashizume_etal_PRR_2022_Measurement_induced_phase_transitionQ_in_sparse_nonlocal.pdf), **Eqs. (2), (4)**: finite-size scaling of other critical observables; **Eq. (5)**: single-reference purification-time scaling; Appendix C **Eqs. (C2)–(C5)**: stabilizer entropies; Sec. VI and Appendix F: *effective* contiguous-distance procedure, distinct from the exact first-failing-segment definition here. Our generalization of Eq. (5) to \(\tau_\ell\) and peak times is **a hypothesis**.
5. **König and Smolin (2014)**, [*How to efficiently select an arbitrary Clifford group element*](https://arxiv.org/pdf/1406.2170), **Eq. (1)**: projective Clifford group cardinality; **Eqs. (2)–(4)**: symplectic representation and quotient. The exact native implementation indexing is defined in `sparsegf2/clifford.py` and frozen in §9.1.

6. **SparseGF2 repository**, [source and README](https://github.com/sjbevins/SparseGF2), [monitored-circuit guide, §§3.1–3.9](https://github.com/sjbevins/SparseGF2/blob/main/docs/GETTING_STARTED.md), [correlated-erasure guide, §§1, 4, 7](https://github.com/sjbevins/SparseGF2/blob/main/docs/observables/correlated-erasure.md), and source files `graphs.py`, `clifford.py`, `mipt.py`, `_kernel/ensemble.py`, `_kernel/rng.py`. **Software contract, not peer-reviewed paper equations.** Pin the exact commit before production; the repository's public `main` can change.

**Derived or study-specific formulas:** the physical-stabilizer rank expression for \(I(A:R)\) (§5.4), complement bounds (§5.8), integrated score \(D(J)\) (§5.7), exact full-reference all-\(\ell\) first-passage recording (§5.2), log-midpoint peak (§5.8), and equal-graph hierarchical estimator (§6). The geometry-correlated erasure distribution in §5.7 is already implemented and documented in SparseGF2; it is not attributed to any of the external MIPT papers. No empirical scaling relationship is asserted as a theorem.
