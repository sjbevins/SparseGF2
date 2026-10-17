# The ensemble and its sampling: the joint law, the Watts–Strogatz graphs, the gates and the measurement masks

How the Watts–Strogatz study draws its randomness: the nested ensemble of graphs, gate
schedules, measurement masks and Born outcomes and its joint law; the sequential rewiring that
defines the graph law and the way the native generator realizes it; the gate law with the
native two-qubit Clifford numbering; the mask law with its float64 thresholds and its nesting
across measurement rates; what the sequential random stream guarantees; and the tests that
check each of these against independent implementations.

This document is written for a reader fluent in the stabilizer formalism and for language
models. Every symbol is defined where it first appears, every formula is the one the code
computes, and every statement about the code names the function it describes as
`path::function` (paths relative to `src/sparsegf2/`, or `tests/` where marked). The Python
blocks run as written, in order, in one namespace, in a few seconds once the kernels are
compiled.

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Notation](#2-notation)
3. [Definitions](#3-definitions)
4. [Propositions and proofs](#4-propositions-and-proofs)
5. [Algorithms](#5-algorithms)
6. [From the mathematics to the code](#6-from-the-mathematics-to-the-code)
7. [Worked examples](#7-worked-examples)
8. [Validation map](#8-validation-map)
9. [Limitations and departures](#9-limitations-and-departures)

---

## 1. Purpose and scope

The study (`docs/misc/watts_strogatz_study_spec_v1_4.md`, cited as "v1.4") defines its ideal
ensemble by nested sampling: an interaction graph $G$, then a gate schedule $\Gamma$ on that
graph, then a measurement mask $M$, then Born outcomes $O$ (v1.4 §1, §6.1). It says the simulator
realizes this with a seeded pseudorandom generator and lists what must be validated (v1.4 §7.4 and
validation requirements 1–3, 6, 10, 11). This document states the ensemble exactly as the code
realizes it, proves every property the code and the records rely on, and maps each claim to the
test that checks it.

It covers

* the four-level hierarchy and the joint law, the proof that the law is normalized, and the
  interchange of the gate and mask averages (v1.4 §1, §4, §6.1, §6.2);
* the circulant graph $C(n,2)$ and the sequential rewiring law $\mu_{n,\beta}$, its native
  realization (offset-major order, the eligible set taken before the deletion, rejection
  sampling with Lemire's `next_below`), and the invariants the graph records and the validation
  rely on: simplicity, the edge count, the absent-edge abort that cannot fire, and the cases
  $\beta = 0$ and $n = 5$ (v1.4 §2, §7.4, §9);
* the gate law, $h(n) = \lfloor n/2\rfloor$ uniform edge draws with replacement and uniform
  Clifford draws, and the 16-to-1 map from the 11,520 projective Cliffords to the 720
  symplectic classes (v1.4 §3, §9.1);
* the mask law, the float64 thresholds against the ideal Bernoulli($p$), the nesting of masks
  under shared uniforms, and independence of masks and gates (v1.4 §4, §7.3, §9);
* the stream facts: prefix invariance under a longer horizon, and independence from probes,
  checkpoints and thread counts (v1.4 §7.3, validation requirement 6).

It changes no probability law, no stream and no Clifford numbering: the generator is the native
`RANDOMNESS_CONTRACT = 2` one and the engine is `_kernel/ensemble.py`. What it adds is the
mathematics, the independent reference implementations and conformance tests
(`tests/_ws_reference.py`, `tests/test_study_sampling.py`), and the graph record
(`study/identity.py::build_graph`, document [02](02-identity-seeds-and-manifest.md)).

## 2. Notation

| symbol | meaning | type | in the code |
|---|---|---|---|
| $n$ | number of system qubits (the study uses $n \ge 5$) | integer | `num_system` |
| $V$ | the vertices $\lbrace 0, \dots, n-1\rbrace$ | set | |
| $C(n,2)$, $E_0(n)$ | the circulant graph and its edge set | graph, set of pairs | `identity.circulant_edges` |
| $L_n$ | the oriented original edges $((a_1,b_1), \dots, (a_m,b_m))$, offset-major | list of $m = 2n$ pairs | `identity.original_edge_list`, `graphs._ring_lattice(n, 4)` |
| $\beta$ | rewiring probability | $[0,1]$ | `beta` |
| $\beta'$ | the realized rewiring probability $P_{\mathrm{real}}(\mathrm{fl}(\beta))$ (Proposition 11): $\beta$ itself for dyadic $\beta$, within $2^{-52}$ of it otherwise | $[0,1]$ | the double `Token.float64` compared with `next_double` |
| $B_j$ | the rewiring decision at visit $j$ | $\lbrace 0,1\rbrace$ | `next_double(rng) < beta` (the float64 of the token) |
| $G^{(j)}$ | the graph after visit $j$ | simple graph | `adj` |
| $N_G(a)$ | the neighbors of $a$ in $G$ | subset of $V$ | `adj[a]` |
| $A_j$ | the eligible targets $V \setminus (N_{G^{(j-1)}}(a_j) \cup \lbrace a_j\rbrace)$ | subset of $V$ | the loop condition `w != a and w not in adj[a]` |
| $b'_j$ | the new endpoint | vertex | `w` |
| $\mu_{n,\beta}$ | the law of $G^{(m)}$ | probability on graphs | `graphs.watts_strogatz` |
| $g$, $G_g$ | a graph label and the graph drawn for it (for the condition $(n, \beta)$) | integer, graph | `identity.graph_seed(master, n, beta, g)` |
| $E_g$ | the edge set of $G_g$, sorted, $a < b$ | set of pairs | the rows of `edges` |
| $h(n)$ | gates per layer, $\lfloor n/2\rfloor$ | integer | `gates_per_layer` |
| $e_{t,a}$, $Q_{t,a}$ | the edge and the Clifford of gate $a$ of layer $t$ | $E_g$, $\lbrace 0, \dots, 11519\rbrace$ | `next_below(rng, len(edges))`, `next_below(rng, 11520)` |
| $\Gamma$ | the gate schedule $((e_{t,a}, Q_{t,a}))_{t,a}$ (v1.4 writes $u$; here $u$ is the trajectory index) | element of $(E_g \times \mathcal K)^{T h(n)}$ | |
| $\mathcal K$, $j$ | the native two-qubit Clifford indices $\lbrace 0, \dots, 11519\rbrace$ | integers | `Clifford.from_index(2, j)` |
| $s$, $b$ | $j = 16 s + b$: symplectic class $s \in \lbrace 0, \dots, 719\rbrace$, sign bits $b \in \lbrace 0, \dots, 15\rbrace$ | integers | `j >> 4`, `j & 15` |
| $p$ | the measurement rate | $[0,1]$ | `measurement_rate` |
| $q_p$ | the float64 nearest to $p$ | double | `Token.float64` |
| $U_{t,i}$ | the mask uniform $k\,2^{-53}$ of site $i$ at layer $t$ | double in $[0,1)$ | `next_double(rng)` |
| $M_{t,i}$ | the measurement indicator, $\mathbf 1\lbrace U_{t,i} < q_p\rbrace$ | $\lbrace 0,1\rbrace$ | the list after `M` in an exported circuit |
| $O$ | the Born outcomes of the measured sites | $\pm 1$ at each measured site | not sampled in phase-free mode |
| $\pi_{g,\gamma,m}(o)$ | the Born probability of the outcomes $o$ | $[0,1]$ | |
| $T$ | the horizon (layers) | integer | `layers` |
| $N = 2^{32}$ | the range of the 32-bit draw behind `next_below` | integer | |
| $\mathrm{rng}$ | an xoshiro256** state | four `uint64` | `_kernel/rng.py` |
| $u$ | the trajectory index; trajectory $u$ runs on graph $\lfloor u/\bar C\rfloor$ for the circuit index stride $\bar C$ (document 02) | integer | `first_trajectory + t` |
| $\lvert A\rvert$ | the size of an eligible set $A$ (in §5.4, of the set of one rewiring event) | integer | `len(v.eligible)` |

## 3. Definitions

### 3.1 The four levels and the joint law (v1.4 §1, §4, §6.1)

Fix $(n, \beta, p, T)$. The ideal ensemble has four levels.

1. **Graph.** $G \sim \mu_{n,\beta}$, the law of the sequential rewiring of §3.3.
2. **Gate schedule.** Given $G = g$ with edge set $E_g$, the schedule is
   $\Gamma = ((e_{t,a}, Q_{t,a}))_{1 \le t \le T,\ 1 \le a \le h(n)}$ with all $T\,h(n)$ pairs
   independent and uniform on $E_g \times \mathcal K$:
   $$
   \kappa_{g,T}(\gamma) = \Big(\frac{1}{\lvert E_g\rvert \cdot 11520}\Big)^{T h(n)}.
   $$
   For $h(n) = 0$ the schedule is empty.
3. **Measurement mask.** $M \in \lbrace 0,1\rbrace^{T \times n}$ with independent entries,
   $$
   \nu_{p,T}(m) = p^{\lvert m\rvert}(1-p)^{nT - \lvert m\rvert}.
   $$
4. **Born outcomes.** With $V_t(g,\gamma)$ the ordered product of the gates of layer $t$ (on the
   system qubits only), $S_t = \lbrace i : m_{t,i} = 1\rbrace$ and, for $o_t \in \lbrace \pm 1\rbrace^{S_t}$,
   $$
   \Pi_{o_t} = \prod_{i \in S_t} \frac{I + o_{t,i} Z_{Q_i}}{2}
   \quad(\Pi = I \text{ if } S_t = \varnothing),
   $$
   the Born probability of the outcomes $o = (o_1, \dots, o_T)$ is
   $$
   \pi_{g,\gamma,m}(o) = \big\lVert \big(\Pi_{o_T} V_T \cdots \Pi_{o_1} V_1 \otimes I_R\big)\,\lvert\Psi_0\rangle \big\rVert^2,
   \qquad \lvert\Psi_0\rangle = \bigotimes_{i=0}^{n-1} \lvert\Phi^+\rangle_{Q_i R_i}.
   $$

The **joint law** (v1.4 §6.1) is the product
$$
\Pr(G = g, \Gamma = \gamma, M = m, O = o) = \mu_{n,\beta}(g)\,\kappa_{g,T}(\gamma)\,\nu_{p,T}(m)\,\pi_{g,\gamma,m}(o).
$$
The first three factors are the randomness the ensemble prescribes; the fourth is the quantum
trajectory law. In phase-free mode the engine does not sample $O$ (document
[03](03-phase-free-tableau.md), Proposition 4, shows that every phase-free quantity is the same for
every outcome).

### 3.2 The circulant graph and the original edge list (v1.4 §2.1, §2.2, §9)

$C(n,2) = (V, E_0(n))$ with
$$
E_0(n) = \big\lbrace \lbrace i, (i+d) \bmod n\rbrace : i \in V,\ d \in \lbrace 1, 2\rbrace,\ (i+d) \bmod n \ne i \big\rbrace,
$$
unordered pairs, duplicates identified (`identity.circulant_edges` builds exactly this set, sorted with
$a < b$). The **oriented original edge list** is the native offset-major order: $d = 1$ then $d = 2$
(outer loop), $i = 0, \dots, n-1$ (inner loop), appending $(a, b) = (i, (i+d) \bmod n)$:
$$
L_n = \big((0,1), (1,2), \dots, (n-1,0),\ (0,2), (1,3), \dots, (n-1,1)\big), \qquad m = \lvert L_n\rvert = 2n \quad (n \ge 5).
$$
`identity.original_edge_list(n)` returns it and equals `graphs._ring_lattice(n, 4)`, the list the
native generator walks. The retained endpoint of a rewired edge is always $a$.

### 3.3 The rewiring law (v1.4 §2.2)

Start from $G^{(0)} = C(n,2)$. For $j = 1, \dots, m$, visit the edge $(a_j, b_j)$ of $L_n$:

1. draw $B_j \sim \mathrm{Bernoulli}(\beta)$, independently of everything else (the draw is made at every
   visit, whatever the state);
2. if $B_j = 0$, set $G^{(j)} = G^{(j-1)}$;
3. if $B_j = 1$, form the eligible set **from the graph before the deletion**,
   $A_j = V \setminus \big(N_{G^{(j-1)}}(a_j) \cup \lbrace a_j\rbrace\big)$;
4. if $A_j = \varnothing$ set $G^{(j)} = G^{(j-1)}$; otherwise draw $b'_j$ uniformly from $A_j$ and set
   $G^{(j)} = G^{(j-1)} - \lbrace a_j, b_j\rbrace + \lbrace a_j, b'_j\rbrace$.

$\mu_{n,\beta}$ is the law of $G^{(m)}$. Three conventions matter because the process is sequential and
the retained endpoint breaks a symmetry. **(i)** The visiting order is offset-major; v1.4 says it must
not be replaced by vertex-major order. **(ii)** The retained endpoint is $a_j = i$. **(iii)** $A_j$ is taken
before the deletion, so $b_j \in N(a_j)$ is never eligible and an edge is never "rewired to itself". v1.4
adds that an absent visited edge aborts the construction; Proposition 4(a) shows this cannot happen for
$n \ge 5$. The law is generally **not** uniform over simple graphs with $2n$ edges (§7, example (a)).

### 3.4 The native realization (`GRAPH_GENERATOR_VERSION = "ws-offset-major-reject-1"`)

The generator is `graphs.py::watts_strogatz(n, 4, beta, seed=...)`. For graph $g$ of the condition
$(n, \beta)$ the study passes the six-word seed
`(GRAPH_SEED_TAG, master, n, lo, hi, g)` of `identity.graph_seed` (document 02, §3.3; $(lo, hi)$ are the
two 32-bit words of a hash of the token of $\beta$), so the seed names the condition (§3.8). The generator
accepts any seed; it uses the xoshiro256** state
`np.random.SeedSequence(seed).generate_state(4, dtype=np.uint64)` (`_kernel/rng.py::seed_state`; an
all-zero state would become $(1,0,0,0)$) and two draws:

* `next_double(rng)` $= k\,2^{-53}$ with $k$ the top 53 bits of `next_u64`, so $B_j = [\texttt{next\_double} < \mathrm{fl}(\beta)]$,
  where $\mathrm{fl}(\beta)$ is the double the study passes (`Token.float64` of the token, document 02, §3.1). Hence
  $\Pr(B_j = 1) = \beta' = P_{\mathrm{real}}(\mathrm{fl}(\beta))$ (Proposition 11): $\beta$ itself when $\beta$ is dyadic with at most 53
  binary digits ($0$, $1/4$, $1/2$, $1$, ...), and within $2^{-52}$ of $\beta$ otherwise ($\beta' - \beta = 8.9\times10^{-17}$ at $\beta = 0.1$);
* `next_below(rng, bound)` (Lemire's method on the top 32 bits of `next_u64`, Proposition 5): a uniform
  integer in $[0, \text{bound})$.

For each visit $(a, b)$ in the order of $L_n$ it draws the double; if it is below $\mathrm{fl}(\beta)$ and
$\deg(a) < n-1$ (equivalently $A_j \ne \varnothing$) it repeats `w = next_below(rng, n)` until
$w \ne a$ and $w \notin N(a)$ (the neighbor set still contains $b$), then deletes $\lbrace a,b\rbrace$ and
adds $\lbrace a,w\rbrace$. **This is rejection sampling from the uniform law on $V$** (Proposition 6).
The *mapping* from generator output to target is part of the generator version: Proposition 8 shows that
another mapping with the same law gives other graphs from the same seed.

### 3.5 The gate law and the Clifford numbering (v1.4 §3, §9.1)

$h(n) = \lfloor n/2\rfloor$ for every $n$, odd included. With `schedule="random_edges"`,
`gates_per_layer=n // 2`, `edge_weights=None` and `random_orientation=False`
(`_kernel/ensemble.py::gen_layer`), each gate draws `e = next_below(rng, |E|)`, indexing the
sorted edge array, and `j = pool[next_below(rng, 11520)]` with `pool = arange(11520)`. The gate acts
on `(a, b) = edges[e]` with $a < b$ (low endpoint first), and the gates of a layer are applied in
the order they were drawn. A pair of draws is made for every gate, so a repeated edge or two gates
with a common qubit are possible.

The native Clifford index is $j = 16 s + b$ (`clifford.py`, module docstring): $s$ is the position of the
gate's $4 \times 4$ symplectic matrix in the list of all 720 elements of $\mathrm{Sp}(4,\mathbb F_2)$ sorted by the
integer whose binary digits are the matrix entries row by row (first entry most significant), and
$b = b_{X_0} + 2 b_{X_1} + 4 b_{Z_0} + 8 b_{Z_1}$ holds the sign bits of the images of $X_0, X_1, Z_0, Z_1$
(`Clifford.from_index`, `Clifford.sign_bits`).

### 3.6 The mask law (v1.4 §4, §7.3, §9)

After the gates of layer $t$, each site $i = 0, \dots, n-1$ draws one $U_{t,i}$ = `next_double(rng)` and is
measured in $Z$ when $U_{t,i} < q_p$, where $q_p$ is the float64 handed to the engine
(`measurement_rate`; document 02, §3.1). The sites are processed in ascending order. The draw is made
for every site at every rate, including $p = 0$ and $p = 1$.

### 3.7 The trajectory stream (v1.4 §7.3)

Trajectory $u$ of a run with seed $\sigma$ draws its circuit from the stream
`stream_state(key_struct, u)`, with `key_struct` the first two words of
`SeedSequence(σ).generate_state(4)` (`mipt.py::_keys`; the last two words key the probe streams). The
study's $\sigma$ is the run seed `identity.run_seed(master, n, beta, gate_mode, p)` of the condition
(document 02, §3.3): five words that name $(n, \beta)$ in `gate_mode="shared"`, the same at every $p$ (§3.8). One
trajectory therefore draws, in this order and nothing else (`_kernel/ensemble.py`, module docstring,
item 4): for each layer $t = 1, \dots, T$, for each gate $a = 1, \dots, h(n)$ an edge index and a Clifford
index, then $n$ mask uniforms. A graph collection assigns trajectory $u$ to graph
$\lfloor u / \texttt{circuits\_per\_graph} \rfloor$ and never draws graph randomness from the circuit
stream.

### 3.8 What is coupled and what is not (v1.4 §1, §6.2, §7.2; design §3 and §11, ruling R-SEED)

The study couples its conditions across the measurement rate $p$ only: for one $(n, \beta)$ the graphs, the
gates and the mask uniforms are the same at every $p$ (so curves across $p$ are paired and the masks are
nested, Proposition 12). It does **not** couple them across $n$ or $\beta$. Finite-size fits and comparisons
between rewiring probabilities treat the samples of different $(n, \beta)$ as independent, and samples built
from one raw stream are not: the same raw draws would feed `next_below(|E|)` at different $|E|$, and the first
rewiring decisions $B_j = [U_j < \beta]$ would be the same draws: the same at two sizes, and nested at two rewiring
probabilities ($[U_j < \beta_1] \le [U_j < \beta_2]$ for $\beta_1 < \beta_2$). (The first draft of the design shared the seed
$(M, g)$ across conditions, as the native collection convention does; ruling R-SEED replaced it.) The seeds are
fixed-length tuples of 32-bit words that name the condition (document 02, §3.3). With the tags
$\tau_G = \texttt{0x57534731}$ and $\tau_C = \texttt{0x57534331}$ (the ASCII codes of `WSG1` and `WSC1`) and
$(\lambda_x, \eta_x)$ the low and high 32-bit words of the 64-bit hash of the token of $x$:

| seed | tuple | words | depends on |
|---|---|---|---|
| graph $g$ of $(n, \beta)$ | $(\tau_G, M_g, n, \lambda_\beta, \eta_\beta, g)$ | 6 | $n$, $\beta$, $g$; not $p$ |
| run, `gate_mode="shared"` | $(\tau_C, M_c, n, \lambda_\beta, \eta_\beta)$ | 5 | $n$, $\beta$; not $p$ |
| run, `gate_mode="independent"` | $(\tau_C, M_c, n, \lambda_\beta, \eta_\beta, \lambda_p, \eta_p)$ | 7 | $n$, $\beta$, $p$ |

Proposition 14 states the consequences, and four tests check them (§8).

## 4. Propositions and proofs

Everything below is derived here unless a citation is given. "Ideal" means the idealization that the
generator's 32-bit and 53-bit draws are independent and uniform; the pseudorandom generator only
approximates it (§9).

### 4.1 The joint law

**Proposition 1 (normalization).** For all $(n, \beta, p, T)$ the joint pmf of §3.1 sums to 1.

*Proof.* $\mu_{n,\beta}$ is the law of a random graph, $\kappa_{g,T}$ is a product of uniform pmfs and
$\nu_{p,T}$ a product of Bernoulli pmfs, so each sums to 1 over its own set. It remains to show
$\sum_o \pi_{g,\gamma,m}(o) = 1$ for fixed $g, \gamma, m$. Write $\psi_0 = \lvert\Psi_0\rangle$ and let every
operator act as itself tensored with $I_R$.

For one layer, the operators $\Pi_{o}$, $o \in \lbrace\pm1\rbrace^{S_t}$, are orthogonal projectors that sum to $I$:
two different outcome vectors differ at a site $i$, where $\frac{I+Z_i}{2}\frac{I-Z_i}{2} = 0$, and
expanding $\prod_{i \in S_t}\big(\frac{I+Z_i}{2} + \frac{I-Z_i}{2}\big) = I$ gives the sum. Hence
$\sum_{o}\lVert \Pi_o \phi\rVert^2 = \lVert\phi\rVert^2$ for every vector $\phi$ (Pythagoras).

Now sum $\lVert \Pi_{o_T} V_T \phi_{T-1}\rVert^2$ over $o_T$, where
$\phi_{T-1} = \Pi_{o_{T-1}} V_{T-1} \cdots \Pi_{o_1} V_1 \psi_0$ does not depend on $o_T$: the result is
$\lVert V_T\phi_{T-1}\rVert^2 = \lVert\phi_{T-1}\rVert^2$ by unitarity. Repeating from layer $T$ down to
layer 1 gives $\lVert\psi_0\rVert^2 = 1$. ∎

*Remark (the chain rule).* The projectors of one layer are applied one site at a time in ascending
order, each followed by renormalization. For any projectors, $\lVert\Pi_k \cdots \Pi_1\psi\rVert^2 =
\prod_k \lVert\Pi_k\psi_{k-1}\rVert^2$ with $\psi_k = \Pi_k\psi_{k-1}/\lVert\Pi_k\psi_{k-1}\rVert$, because the
product telescopes. So $\pi$ is the product of the sequential conditional probabilities
$\langle\varphi\vert\frac{I + zZ_{Q_i}}{2}\vert\varphi\rangle$ of v1.4 §4, where each is $0$, $\frac12$ or $1$ for a
stabilizer state.

**Proposition 2 (interchange of the gate and mask averages; v1.4 §6.2).** For any real function
$F(g, \gamma, m, o)$ put $\overline F(g,\gamma,m) = \sum_o \pi_{g,\gamma,m}(o)\,F(g,\gamma,m,o)$. Then
$$
\mathbb E_{\gamma\sim\kappa_{g,T}}\ \mathbb E_{m\sim\nu_{p,T}}\ \overline F(g,\gamma,m) =
\mathbb E_{m\sim\nu_{p,T}}\ \mathbb E_{\gamma\sim\kappa_{g,T}}\ \overline F(g,\gamma,m).
$$

*Proof.* Conditional on $G = g$ the schedule and the mask are independent, so the pair has the product
pmf $\kappa_{g,T}(\gamma)\nu_{p,T}(m)$ on the finite set (schedules) $\times$ (masks). Both sides are the
finite sum $\sum_{\gamma,m}\kappa_{g,T}(\gamma)\,\nu_{p,T}(m)\,\overline F(g,\gamma,m)$ written in two orders, and a
finite sum can be reordered. ∎ This is the finite case of Fubini's theorem. It says nothing about the
physical operators commuting: $\overline F$ is evaluated for each $(\gamma, m)$ with the layers in their
physical order.

*Consequence.* The simulator may produce the schedule and the mask from one stream, in the order of §3.7,
and may couple its random numbers across different $p$ (Proposition 12) without changing any
fixed-$p$ expectation, because those expectations depend only on the marginal law of $(G, \Gamma, M)$ at
that $p$, which Proposition 12 shows is the product law.

### 4.2 The graphs

**Proposition 3 ($C(n,2)$; v1.4 §2.1).** For every $n \ge 1$, $\lvert E_0(n)\rvert = n\min\lbrace 4, n-1\rbrace / 2$ and every vertex has degree $\min\lbrace4, n-1\rbrace$. For $n \le 5$
it is the complete graph $K_n$. For $n \ge 5$ it is the 4-regular graph with $2n$ edges (odd and even
$n$ alike), and the $2n$ oriented pairs of $L_n$ are pairwise distinct as unordered pairs.

*Proof.* The neighbors of $i$ are $i \pm 1, i \pm 2 \bmod n$ with $i$ itself excluded. For $n \ge 5$ these four
residues are pairwise distinct and different from $i$: the coincidences $1 \equiv -1$, $2 \equiv -2$,
$1 \equiv -2$, $2 \equiv -1$, $1 \equiv 2$ and $\pm1, \pm2 \equiv 0$ need $n \mid 2$, $n \mid 4$, $n \mid 3$, $n \mid 3$, $n \mid 1$ and
$n \mid 1$ or $n \mid 2$, impossible for $n \ge 5$. So the degree is 4 and $\lvert E_0\rvert = 4n/2 = 2n$. For $n = 5$
the four neighbors are all the other vertices: $K_5$ with $\binom52 = 10 = 2n$ edges. For $n \le 4$ the
same residue argument gives $\min\lbrace4,n-1\rbrace$ distinct neighbors (all other vertices). Finally, for
$n \ge 5$ every entry of $L_n$ is an element of $E_0(n)$, every element of $E_0(n)$ arises (by the
definition), and both have $2n$ elements, so the map from entries to edges is onto a set of the same size,
hence one-to-one. ∎ (For $n = 4$ the list has 8 entries but $K_4$ has 6 edges: $\lbrace0,2\rbrace$ and
$\lbrace1,3\rbrace$ occur twice. This is why the study needs $n \ge 5$.)

**Proposition 4 (invariants of the rewiring, $n \ge 5$).** For every sequence of decisions and targets
the rewiring satisfies:

* (a) at visit $j$ the original edge $\lbrace a_j, b_j\rbrace$ is an edge of $G^{(j-1)}$ (so the abort of v1.4 §2.2 never fires);
* (b) every $G^{(j)}$ is a simple graph with $2n$ edges;
* (c) a rewired edge changes the neighbor set of $a_j$: $b'_j \ne a_j$, $b'_j \ne b_j$, and $\lbrace a_j, b'_j\rbrace \notin E(G^{(j-1)})$;
* (d) an edge added at visit $j$ is never deleted later, and neither is an original edge that was not rewired at its own visit;
* (e) every vertex of $G^{(m)}$ has degree at least 2, and the degrees sum to $4n$;
* (f) $\beta = 0$ gives $G^{(m)} = C(n,2)$, and $n = 5$ gives $K_5$ for every $\beta$.

*Proof.* Let $I_j$ be the statement: $G^{(j-1)}$ is simple with $2n$ edges, and it contains
$\lbrace a_i, b_i\rbrace$ for every $i \ge j$. $I_1$ holds: $G^{(0)} = C(n,2)$ is simple with $2n$ edges
(Proposition 3) and contains every original edge. (a) is the second part of $I_j$ for $i = j$.

Assume $I_j$. If visit $j$ changes nothing, $I_{j+1}$ follows. If it rewires, $b'_j \in A_j$ gives
$b'_j \ne a_j$ (no loop) and $b'_j \notin N(a_j)$, so $\lbrace a_j,b'_j\rbrace$ is not an edge of
$G^{(j-1)}$ (the new edge is not parallel to an existing one); $b_j \in N(a_j)$ by (a), so $b'_j \ne b_j$.
Exactly one edge is removed and one new edge added, so the graph stays simple with $2n$ edges. For
$i > j$, $\lbrace a_i,b_i\rbrace \in E(G^{(j-1)})$ by $I_j$ and $\lbrace a_i,b_i\rbrace \ne \lbrace a_j,b_j\rbrace$ by the
distinctness in Proposition 3, so it is still present in $G^{(j)}$: this is $I_{j+1}$. That proves (a)–(c).

(d) The edge added at visit $j$ is not in $G^{(j-1)}$, whereas each $\lbrace a_i,b_i\rbrace$ with $i > j$ is; so the
added edge differs from every later visited edge and is never the one deleted. An original edge is
deleted only at its own visit, and it is visited once.

(e) Vertex $v$ is the retained endpoint of exactly two visits, $(v, v+1)$ (offset 1) and $(v, v+2)$
(offset 2), in this order. After each visit it owns one edge: the original one if not rewired, the added
one otherwise; by (d) both owned edges stay in the graph. They are different edges: let $o_1 = \lbrace v, x_1\rbrace$
be the first. If the first visit did not rewire, $x_1 = v+1 \ne v+2$. If it did, $x_1$ was not a neighbor of
$v$, whereas $v+2$ was (the original edge $\lbrace v,v+2\rbrace$ is present by $I_j$), so $x_1 \ne v+2$.
If the second visit did not rewire its owned edge is $\lbrace v, v+2\rbrace \ne o_1$; if it did, its target is not
a neighbor of $v$, whereas $x_1$ is, so the owned edge differs from $o_1$. Hence $\deg v \ge 2$. The degree
sum is twice the edge count.

(f) With $\beta = 0$ every $B_j = 0$. With $n = 5$ every vertex of $C(5,2) = K_5$ has degree $4 = n-1$, so
$A_j = \varnothing$ at every visit. ∎

*Consequences.* The study validates every graph it builds against (b), (e), (f) (`identity.validate_graph`);
(a) is why the "abort on an absent original edge" of v1.4 §2.2 lives in the independent reference
(`tests/_ws_reference.py::rewire`, which checks it at every visit) and in these tests, not in
`graphs.py`.

**Proposition 5 (Lemire's `next_below` is exactly uniform).** Let $N = 2^{32}$ and $1 \le b < N$. For
$x$ uniform on $[0, N)$ put $m = x b$, $\ell = m \bmod N$ and $t = N \bmod b$. Redrawing $x$ whenever $\ell < t$
and returning $\lfloor m / N\rfloor$ otherwise gives an output uniform on $[0, b)$. Each draw is accepted
with probability $1 - t/N$, and $t < b$.

*Proof.* The kernel (`_kernel/rng.py::next_below`) computes the threshold $(N - b) \bmod b = N \bmod b = t$ only
when $\ell < b$; since $t < b$, an $\ell \ge b$ is accepted in any case, so this is the rule above. Fix
$v \in [0, b)$. The draws with $\lfloor xb/N\rfloor = v$ are those with $x b = vN + \ell$ for an integer
$0 \le \ell < N$, and then $\ell \equiv -vN \pmod b$; conversely each such $\ell$ gives the integer
$x = (vN + \ell)/b < N$. So the draws mapping to $v$ correspond one-to-one to the integers
$\ell \in [0, N)$ in the residue class $r_v = (-vN) \bmod b$. The accepted ones have $\ell \in [t, N)$, an
interval of length $N - t = b\lfloor N/b\rfloor$, a multiple of $b$, so it contains each residue class mod
$b$ exactly $\lfloor N/b\rfloor$ times. Every $v$ is thus produced by exactly $\lfloor N/b\rfloor$ accepted draws
out of $b\lfloor N/b\rfloor$, and the output is uniform. Independent redraws keep it uniform. ∎

The proof is by counting residue classes, so it is checked for the bounds the study uses without
enumerating $2^{32}$ values (§7, example (c)). For $b = 11520$ the rejection probability per draw is
$(N \bmod 11520)/N = 256/2^{32} \approx 6\times10^{-8}$.

**Proposition 6 (rejection sampling gives the uniform eligible target).** Let $A \subseteq V$ with
$\lvert A\rvert = \alpha \ge 1$, and $X_1, X_2, \dots$ i.i.d. uniform on $V$ (Proposition 5, bound $n$). Let
$\tau = \min\lbrace k : X_k \in A\rbrace$. Then $\tau$ is geometric with success probability $\alpha/n$ (finite
almost surely, $\mathbb E\tau = n/\alpha$, $\operatorname{Var}\tau = (1 - \alpha/n)(n/\alpha)^2$), and $X_\tau$ is uniform on $A$
and independent of $\tau$.

*Proof.* For $w \in A$ and $k \ge 1$, $\Pr(\tau = k, X_\tau = w) = (1 - \alpha/n)^{k-1}\cdot\frac1n$. This factors as
$\big[(1 - \alpha/n)^{k-1}\frac{\alpha}{n}\big]\cdot\frac1\alpha$: a geometric law in $k$ times the uniform law on $A$. ∎
So step 4 of §3.3 is realized exactly, and the expected number of `next_below` calls for a rewiring with
eligible set $A$ is $n/\lvert A\rvert$ (the test `test_rejection_sampling_uses_n_over_the_eligible_size_draws_on_average`).

**Proposition 7 (the exact law; consequences).**

* (a) $(G^{(j)})_j$ is a Markov chain on graphs. From $G$, visit $j$ stays with probability $1 - \beta$ (or 1
  if $A_j = \varnothing$) and moves to $G - \lbrace a_j,b_j\rbrace + \lbrace a_j,w\rbrace$ with probability $\beta/\lvert A_j\rvert$ for each $w \in A_j$.
  The law $\mu_{n,\beta}$ is the distribution of $G^{(m)}$ for $G^{(0)} = C(n,2)$, computed exactly by
  propagating the probability of every reachable graph (§5.2).
* (b) The decisions $B_1, \dots, B_m$ are i.i.d. Bernoulli($\beta$), whatever the graph, so the number
  $R = \sum_j B_j$ of rewiring *attempts* is Binomial($2n$, $\beta$), with mean $2n\beta$ and variance
  $2n\beta(1-\beta)$. (Some attempts do nothing: when $A_j = \varnothing$.)
* (b') The native generator compares the uniform with the double $\mathrm{fl}(\beta)$ (§3.4), so in the ideal model its
  decisions are i.i.d. Bernoulli($\beta'$) with $\beta' = P_{\mathrm{real}}(\mathrm{fl}(\beta))$ (Proposition 11), and its graphs have the
  law $\mu_{n,\beta'}$ of (a) at $\beta'$. For dyadic $\beta$ with at most 53 binary digits $\beta' = \beta$; otherwise
  $0 < \lvert\beta' - \beta\rvert < 2^{-52}$ and the total variation distance between $\mu_{n,\beta'}$ and $\mu_{n,\beta}$ is at most
  $2n\lvert\beta' - \beta\rvert < 2n\cdot 2^{-52}$.
* (c) $\mu_{n,\beta}$ is in general not uniform on the simple graphs with $2n$ edges (§7, example (a)).

*Proof.* (a) is the definition of §3.3 read as a chain. (b): $B_j$ is drawn without reference to the state.
(b'): $B_j = [U_j < \mathrm{fl}(\beta)]$ with $U_j$ uniform on the multiples of $2^{-53}$, whose probability is
Proposition 11 with $q = \mathrm{fl}(\beta)$, and the bound $\lvert\beta' - \beta\rvert < 2^{-52}$ is the one proved there. For the
distance, couple the two processes on the same uniforms: draw $V_j$ uniform on $[0,1)$ and set $B_j = [V_j < \beta]$ and
$B'_j = [V_j < \beta']$, with the same target draws. The two graphs can differ only if some $B_j \ne B'_j$, which has
probability $\lvert\beta' - \beta\rvert$ at each of the $m = 2n$ visits, so by the union bound the graphs differ with probability at
most $2n\lvert\beta' - \beta\rvert$, which bounds the total variation distance. (c) is shown by exhibiting the exact law: at
$n = 6$ and $\beta = 1/2$ the most likely graph has probability $0.0228$, ten times the uniform value $1/455$. ∎
The exact-law and moment tests use dyadic $\beta$ ($1/4$, $1/2$, $1$), where $\mu_{n,\beta'} = \mu_{n,\beta}$; the decision tests at
$\beta = 0.1$ cannot resolve a difference of $9\times10^{-17}$.

**Proposition 8 (the mapping from random numbers to targets is part of the generator).** Let
$\mathrm{Rej}$ be the native sampler of §3.4 and $\mathrm{Idx}$ the sampler that draws one
`next_below(len(A))` and returns the sorted $A$ at that position. Both realize the uniform law on $A$
(Propositions 5 and 6), so both produce graphs with law $\mu_{n,\beta}$. They are different functions
of the generator output (different bounds, a different number of generator steps), and a graph
produced from a seed by one is in general not the graph produced by the other.

*Proof.* Equality of laws: each decision and each target is, conditional on the past, as in §3.3.
Different functions: from one 32-bit word $x$ the rejection sampler proposes $v = \lfloor xn/N\rfloor$ and
accepts it only if $v \in A$ (otherwise it draws again, which moves every later generator position), whereas the
index sampler takes the position $\lfloor x\lvert A\rvert/N\rfloor$ of the sorted $A$. Example (d) of §7 builds the same
seeds with both mappings, and none of 200 graphs at $n = 24$ and $\beta = 1/2$ coincides (each has about 24
rewirings). ∎ Hence "the same law" does not give "the same graph from the same seed", and a graph record
is reproducible only under the mapping named by its `generator` field.

### 4.3 The gates

**Proposition 9 (the gate law).** In the ideal model, for a fixed graph with edge set $E_g$:

* (a) the $T\,h(n)$ edge draws are i.i.d. uniform on $E_g$ (with replacement), so
  $\Pr(e_{t,a} = e) = 1/\lvert E_g\rvert$ and a qubit $v$ is an endpoint of a draw with probability
  $\deg_g(v)/\lvert E_g\rvert$ (uniform *edge* sampling, not uniform vertex sampling);
* (b) the number of distinct edges among the $h$ draws of one layer has the occupancy law
  $\Pr(D = d) = \binom{\lvert E_g\rvert}{d}\,d!\,S(h,d)/\lvert E_g\rvert^h$ ($S$ the Stirling numbers of the second kind), in
  particular all $h$ edges differ with probability $\prod_{i<h}(1 - i/\lvert E_g\rvert)$;
* (c) the Clifford indices are i.i.d. uniform on $\lbrace 0, \dots, 11519\rbrace$, independent of the edges and
  of the masks.

*Proof.* (a) and (c): each draw is `next_below` with a fixed bound, uniform by Proposition 5, from
disjoint parts of the stream. (b) is the classical occupancy distribution for $h$ balls in
$\lvert E_g\rvert$ cells; the recursion $P_{k+1}(d) = P_k(d)\,d/\lvert E_g\rvert + P_k(d-1)(\lvert E_g\rvert - d + 1)/\lvert E_g\rvert$ computes it
(`tests/test_study_sampling.py::occupancy_pmf`). ∎

**Proposition 10 (the 16-to-1 map).** The map $j \mapsto (s, b) = (\lfloor j/16\rfloor, j \bmod 16)$ is a bijection from
$\lbrace 0, \dots, 11519\rbrace$ onto $\lbrace 0, \dots, 719\rbrace \times \lbrace 0, \dots, 15\rbrace$. Hence for $j$ uniform on 11,520 values, $s$ is uniform
on 720 values, $b$ is uniform on 16 values, they are independent, and every symplectic class has exactly 16
preimages. In the phase-free engine a gate acts through its class $s$ only.

*Proof.* The bijection is the division with remainder. A uniform variable on a product set under a
bijection onto a product of two sets is a product of uniforms. The identification of $s$ with the
$s$-th sorted element of $\mathrm{Sp}(4,\mathbb F_2)$ is the definition in `clifford.py`; `tests` verify it by brute force
over all $2^{16}$ binary matrices (720 satisfy $M\Omega M^T = \Omega$; v1.4 §9.1 cites König and Smolin,
Eqs. (1)–(4), for $\lvert\mathrm{Sp}(4,\mathbb F_2)\rvert = 720$ and the 16 sign patterns). The kernel applies the linear part of a gate
from `lin2[sym]` with `sym = g >> 4` (`_kernel/tableau.py::apply_2q`); the sign polynomial, which
depends on `g & 15`, is used only when phases are tracked, and the ensemble engine is phase-free. ∎

*Consequence.* Because the integer draw $j$ is itself exactly uniform (Proposition 5 with
$b = 11520$), $\Pr(s) = 16/11520 = 1/720$ exactly under the ideal input: sampling the 11,520 projective
Cliffords and sampling the 720 symplectic classes give the same phase-free ensemble (v1.4 §3), but only
the first reproduces the seeds of the native contract (v1.4 §9.1).

### 4.4 The masks, nesting and the stream

**Proposition 11 (the mask law and its float64 threshold).** Let $q \in [0,1]$ be a double and $U$ =
`next_double`, i.e. $U = k\,2^{-53}$ with $k$ uniform on $\lbrace 0, \dots, 2^{53}-1\rbrace$. Then
$$
P_{\mathrm{real}}(q) := \Pr(U < q) = \frac{\lceil 2^{53} q\rceil}{2^{53}}, \qquad 0 \le P_{\mathrm{real}}(q) - q < 2^{-53}.
$$
For $q = q_p = \mathrm{fl}(p)$ (the double nearest to $p$), $\lvert P_{\mathrm{real}}(q_p) - p\rvert < 2^{-52}$. The
endpoints are exact: $p = 0$ never measures and $p = 1$ always does, and so is every dyadic $p$ with at
most 53 binary digits. For fixed $p$ the masks $M_{t,i} = [U_{t,i} < q_p]$ are independent
Bernoulli($P_{\mathrm{real}}(q_p)$) variables.

*Proof.* $k\,2^{-53} < q \iff k < 2^{53}q \iff k \le \lceil 2^{53}q\rceil - 1$, and there are $\lceil 2^{53}q\rceil$ such $k \ge 0$.
The ceiling exceeds its argument by less than 1, giving the first bound. Rounding to the nearest double moves $p \in [0,1]$ by at most half a unit in the last place,
$\lvert q_p - p\rvert \le 2^{-54}$, so $\lvert P_{\mathrm{real}} - p\rvert < 2^{-53} + 2^{-54} < 2^{-52}$. If $q = 0$ then no $k$ qualifies; if
$q = 1$ every $k < 2^{53}$ does. If $2^{53}q$ is an integer, $P_{\mathrm{real}} = q$. Independence: the $U$'s are
independent. ∎ So a rate token $x$ (document 02) is realized by Bernoulli($P_{\mathrm{real}}$) with
$\lvert P_{\mathrm{real}} - x\rvert < 2^{-52}$, below any statistical resolution (v1.4 §7.2: the ideal law "cannot in general be
represented by a finite binary floating-point threshold").

**Proposition 12 (shared uniforms nest the masks).** In `gate_mode="shared"` (the run seed of the condition
$(n, \beta)$, which is the same for every rate), fix the condition, the graph, the trajectory index $u$ and the
horizon. Then the schedule $\Gamma$ and the
uniforms $U_{t,i}$ are the same at every rate, and for $p_1 \le p_2$ the masks satisfy
$M_{t,i}(p_1) \le M_{t,i}(p_2)$ for all $t, i$. For each fixed $p$ the law of $(G, \Gamma, M)$ is the
product law of §3.1.

*Proof.* In shared mode the run seed does not contain $p$ (document 02, Proposition 6), so trajectory $u$ has the
same stream at every rate. The draws of a trajectory are consumed in the order of §3.7: for each layer, the gate draws, then
exactly $n$ doubles, and the number and order of all draws depend on the stream and on $(n, \lvert E\rvert)$ only (the
rare rejections inside `next_below` depend on the stream, not on $p$): `gen_layer` always draws the doubles,
and $p$ enters only in the comparison $U < q_p$. So $\Gamma$ and $U$ do not depend on $p$.
$p_1 \le p_2$ implies $q_{p_1} \le q_{p_2}$ (rounding is monotone), and $[U < q_{p_1}] \le [U < q_{p_2}]$ pointwise.
The marginal law at one $p$ is Proposition 11 and does not depend on how the $U$ are shared with other rates. ∎
In `gate_mode="independent"` the run seed is derived from the rate (document 02), the streams differ, and
nothing is nested or shared.

**Proposition 13 (independence of masks and gates; of trajectories; prefix invariance).** In the ideal model
the gate draws, the mask uniforms and the draws of different trajectories are mutually independent. Moreover
the first $T_1$ layers of trajectory $u$ are the same circuit for every horizon $T \ge T_1$, and the circuit
does not depend on the probes, the checkpoint schedule, the other trajectories of the run, the graphs of
the collection other than $u$'s own, or the number of threads.

*Proof.* Independence: the quantities are functions of disjoint sets of generator outputs (distinct stream
positions of one stream, or distinct streams), which are independent in the ideal model. Prefix: layer $t$ draws
from the stream position reached after layers $1, \dots, t-1$, by code (`gen_layer`) that does not read the
horizon; the stream is `stream_state(key_struct, u)`, a function of the seed and $u$ only. Probes draw from the
separate key `key_probe` (the last two of the four words of `SeedSequence(σ).generate_state(4)`, §3.7; not words of
the seed tuple $\sigma$ itself), checkpoints only decide when to look, and each
trajectory owns its state array, so scheduling cannot change it (`tests/test_invariance.py`,
`tests/test_randomness_contract.py`). ∎ A layer is not independently addressable: the state after $t-1$ layers
is reached only by replaying them, which is why extending the horizon reruns from layer 0 (design D3).

**Proposition 14 (conditions are independent samples; only the rates of one condition are coupled).** Let the
seed tuples of §3.8 be given and assume, as in the rest of this document, that streams of different `SeedSequence`
entropy are independent (the ideal model; document 02, §9). Then

* (a) for two different conditions $(n, \beta) \ne (n', \beta')$ of one study, the graph collections
  $(G_g)_g$ and $(G'_g)_g$ and the run seeds are four different families of tuples, so the collections and the
  circuit streams of the two conditions are mutually independent;
* (b) within one condition, graph $g$ and graph $g' \ne g$ are independent, and the graph collection is the same
  at every rate (the graph seed has no $p$);
* (c) in `gate_mode="shared"` the rates of one condition use one run seed: the gates and the mask uniforms of
  trajectory $u$ are the same at every rate and the masks are nested (Proposition 12); in `gate_mode="independent"`
  the rates have different run seeds, so their circuits are independent and nothing is nested;
* (d) a graph seed never equals a run seed, whatever the master seeds are, so one master seed may serve both.

*Proof.* The statements (a), (b) and (d) are the injectivity of the seed maps (document 02, Proposition 4): a graph
seed determines $(M_g, n, \beta, g)$ and a run seed $(M_c, n, \beta)$ (and $p$ in independent mode), and a graph seed
and a run seed differ in the first word. Different tuples are different entropy, hence independent streams in the ideal
model, and functions of independent streams are independent. (c) is Proposition 12 for the shared mode, and for the
independent mode the seeds differ by the words of $p$ (document 02, Proposition 6). ∎ *What the tests check.* Independence
cannot be proved for a pseudorandom generator (v1.4 §7.1). The tests look where a shared seed shows at once (§8): if the
first draws were shared, the visit of the first rewiring attempt of graph $g$ would be identical at two sizes, and nested at
two values of $\beta$ (with the decisions $[U_j < \beta]$ on one sequence of uniforms, the first attempt at the larger $\beta$
never comes after the one at the smaller); and the
index of the first gate would be equal at two values of $\beta$ (the same $2n$ values from the same draws), and its
cell $[e/2n, (e+1)/2n)$ would contain the same first draw at two sizes. Each statistic has an exact null law under independence; each
is degenerate under the shared seeds of the first draft, which the tests use as the negative control.

## 5. Algorithms

### 5.1 The graph generator

Input $n \ge 5$, $\beta$, a seed. Output the sorted edge array $(2n, 2)$.

1. Seed an xoshiro256** state from the seed; `adj[v]` $\leftarrow$ the neighbors of $v$ in $C(n,2)$ from $L_n$.
2. For each $(a, b)$ in $L_n$: draw $x$ = `next_double`; if $x \ge \mathrm{fl}(\beta)$ continue.
3. If $\lvert\mathrm{adj}[a]\rvert = n - 1$ continue (empty eligible set).
4. Repeat `w = next_below(n)` until $w \ne a$ and $w \notin \mathrm{adj}[a]$.
5. Delete $b$ from `adj[a]` and $a$ from `adj[b]`; add $w$ to `adj[a]` and $a$ to `adj[w]`.
6. Return the sorted pairs $a < b$ (`graphs.from_edges`).

Cost: $2n$ doubles plus, per rewiring, an expected $n/\lvert A\rvert$ bounded draws (Proposition 6), $O(n\bar d)$ for
the neighbor tests; 60 to 100 microseconds per graph at $n$ between 7 and 24 in this implementation
(measured). Exactness: in the ideal model the output has exactly the law $\mu_{n,\beta'}$ of the realized probability
$\beta' = P_{\mathrm{real}}(\mathrm{fl}(\beta))$, which is $\mu_{n,\beta}$ for dyadic $\beta$ and within total variation $2n\cdot2^{-52}$ of it
otherwise (Proposition 7(b')); the pseudorandom generator only approximates the ideal model (§9). The result is validated by
`identity.validate_graph` (Proposition 4).

### 5.2 The exact law by dynamic programming

For $n \le 7$ the law $\mu_{n,\beta}$ can be computed without random numbers. A graph is the bitmask over the
$\binom n2$ sorted pairs. Keep a dictionary from graph to probability, starting with $\lbrace C(n,2) \mapsto 1\rbrace$. For
each visit $j$ build the next dictionary: for every state $G$ with probability $q$, assert that
$\lbrace a_j,b_j\rbrace \in G$; compute $A_j$ from $G$; add $q(1-\beta)$ to $G$ (all of $q$ if $A_j = \varnothing$) and, for each $w \in A_j$,
$q\beta/\lvert A_j\rvert$ to $G - \lbrace a_j,b_j\rbrace + \lbrace a_j,w\rbrace$. States of probability zero are dropped. The dictionary
after $m$ visits is $\mu_{n,\beta}$, and the assertion firing nowhere proves Proposition 4(a) for that
$n$ by exhaustion. The number of reachable graphs at $\beta = 1/2$ is 1, 428 and 101,593 for $n = 5, 6, 7$, asserted
by the tests, and 24,359,329 for $n = 8$ (measured once, 87 seconds; not in the test suite).
`tests/_ws_reference.py::exact_law` implements it with floats or exact `Fraction`s.

### 5.3 One layer of the engine

`_kernel/ensemble.py::gen_layer`, for the study protocol: for $a = 1, \dots, h(n)$: $e$ = `next_below(rng, |E|)`,
$(x, y)$ = `edges[e]`, $j$ = `pool[next_below(rng, 11520)]`; then for $i = 0, \dots, n-1$: `next_double(rng) < p`
marks site $i$. Cost $O(h(n) + n)$ draws per layer. The export `MonitoredCircuit.circuit(u, seed=...)`
records exactly these draws (the generator functions are shared with the run), as `CLIFFORD_2Q(j, ...) x y ...`
lines and an `M` line of the measured sites before each `TICK`.

### 5.4 The conformance statistics

* **Pearson's chi-square.** For counts $O_c$ with expected counts $E_c$, $X^2 = \sum_c (O_c - E_c)^2/E_c$ over cells
  with $E_c \ge 5$ (smaller cells are pooled), degrees of freedom the number of cells minus 1, and
  $p$-value $Q(\nu/2, X^2/2)$ (the regularized upper incomplete gamma function;
  `tests/_ws_reference.py::chi2_sf`, checked against the closed forms for even $\nu$, $\nu = 1, 3$ and published
  critical values). A test passes when $p > 10^{-6}$.
* **z-tests.** A sum of independent terms with known mean and variance, standardized; passes when $\lvert z\rvert < 4.5$
  (two-sided false-alarm 6.8e-6 under the normal approximation). Means and variances are computed from the exact
  law, never from the sample. This includes the spread at $n = 7$: with $\mu$, $\sigma^2$ and $\mu_4$ the exact mean,
  variance and fourth central moment of a statistic, $V = \frac1N\sum_i (x_i - \mu)^2$ has $\mathbb E V = \sigma^2$ and
  $\operatorname{Var} V = (\mu_4 - \sigma^4)/N$ exactly for independent draws, and $z = (V - \sigma^2)/\sqrt{(\mu_4 - \sigma^4)/N}$.
* **Randomized probability integral transform (PIT).** For a rewiring with eligible set $A$ and target at
  position $r$ of the sorted set, $u = (r + V)/\lvert A\rvert$ with $V$ uniform from an independent generator. Given the past
  the target is uniform on $A$, so $u$ is uniform on $(0,1)$ and independent of the past; the sequence of $u$ over all
  rewirings, whatever the sizes $\lvert A\rvert$, is exactly i.i.d. uniform. A 20-bin chi-square tests it
  (`tests/test_study_sampling.py::target_pit_p`).
* **Correlation families.** With $Y = M - p$, a product $Y_a Y_b$ of two different positions has mean 0 and variance
  $(p(1-p))^2$, and products of different pairs are uncorrelated. Over $N$ independent terms
  $z = \sum Y_aY_b/(\sqrt N\,p(1-p))$ is standard normal, and the squares of a family add up to a chi-square. The
  families are two sites of one layer, one site at lags 1–3, two sites in consecutive layers and neighboring
  trajectories (`correlation_pvalues`). Given the gates the masks are independent Bernoulli($p$), so for any weights
  $c$ that depend on the gates only $z_i = \sum Y c/\sqrt{p(1-p)\sum c^2}$ is standard normal, and the $z_i$ of different sites
  are independent (`gate_dependence_pvalues`). At the level of layers, the measured count $X$ of a layer is
  Binomial($n$, $p$) and the mean $Y$ of the classes (or edge indices) of its $h$ gates has the mean and variance of one
  uniform draw on 720 values (or $\lvert E\rvert$), the variance divided by $h$; under independence the products
  $(X - \mathbb EX)(Y - \mathbb EY)$ of different layers are uncorrelated with variance $\operatorname{Var}X\operatorname{Var}Y$, so
  $z = \sum (X - \mathbb EX)(Y - \mathbb EY)/\sqrt{N\operatorname{Var}X\operatorname{Var}Y}$ has mean 0 and variance 1 exactly
  (`test_masks_are_independent_of_the_gate_draws`; the "next layer" pairs take $Y$ from the following layer).
* **False-alarm accounting.** Every component test is counted: a maximum over 48 positions is 48 z-tests, the
  minimum of six $p$-values is six tests, a Bonferroni-corrected maximum ($k$ times the tail of the largest of $k$
  scores, `frequency_pvalues`) is one $p$-value test. With $\alpha = 10^{-6}$ per $p$-value test and $6.8\times10^{-6}$ per
  z-test, the union bound (valid whatever the dependence between the tests) on the chance that a correct
  implementation fails at least one at fresh seeds is
  $$46 \times 10^{-6} + 149 \times 6.8\times10^{-6} = 1.06\times10^{-3}.$$
  At the fixed seeds the outcome is deterministic. Passing says no deviation was found at the stated sensitivity,
  not that the generator is random (v1.4 §7.4). The inventory (tests of `tests/test_study_sampling.py`, with their
  parametrizations; negative controls, which must *fail* the null, are not counted):

  | test | $p$-value tests | z-tests |
  |---|---|---|
  | `test_chi_square_helper_pools_small_cells_and_detects_bias` (numpy's multinomial) | 1 | |
  | `test_the_mapping_from_random_numbers_to_targets_matters_not_only_the_law` | 1 | |
  | `test_the_native_generator_has_the_exact_law_at_n_6` ($\beta = 1/4, 1/2$) | 2 | |
  | `test_each_plausible_misreading_of_the_rewiring_rule_is_rejected` (the correct reading: one test, repeated on the same graphs in the three parametrizations) | 1 | |
  | `test_the_native_generator_has_the_exact_moments_at_n_7` (3 values of $\beta$, 2 statistics, mean and spread) | | 12 |
  | `test_rewiring_decisions_are_bernoulli_beta` ($\beta = 0.1, 0.5$: the total, the histogram, the 48 positions and their sum of squares, lag 1) | 4 | 100 |
  | `test_targets_are_uniform_on_the_eligible_set_computed_before_the_deletion` (2 values of $\beta$) | 4 | |
  | `test_rejection_sampling_uses_n_over_the_eligible_size_draws_on_average` (2 values of $\beta$) | | 2 |
  | `test_edge_draws_are_uniform_over_the_graphs_edges` (pooled, by position, 24 vertex incidences) | 2 | 24 |
  | `test_edges_are_drawn_with_replacement_in_sampled_order` | 1 | 2 |
  | `test_disjoint_pairs_of_edge_draws_are_independent_and_uniform` | 2 | |
  | `test_clifford_indices_are_uniform_over_11520_and_their_classes_over_720` | 5 | |
  | `test_the_clifford_test_has_the_power_to_see_a_bias` (the fair sample) | 1 | |
  | `test_masks_are_independent_bernoulli_p_at_every_site` | 2 | 1 |
  | `test_masks_have_no_site_or_layer_correlation_beyond_chance` | 6 | |
  | `test_masks_are_independent_of_the_gate_draws` | 3 | 4 |
  | `test_the_correlation_tests_have_the_power_to_see_planted_dependence` (the clean samples) | 11 | |
  | the four tests of Proposition 14 (independent conditions) | | 4 |
  | **total** | **46** | **149** |

  Two further assertions are statistical in principle, with false-alarm probabilities below $10^{-190}$ by the Chernoff
  bound $e^{-N\,\mathrm{KL}}$ for independent Bernoulli trials: that more than 70% of the 28,800 layers of
  `test_edges_are_drawn_with_replacement_in_sampled_order` repeat an edge (expected 77.7%, standard error 0.25%), and
  that more than 19,000 of the 20,000 graphs at $\beta = 0.1$ rewire at least once (expected 19,872). They are not
  counted. The test of Proposition 8 allows 2 coincidences among 200 graphs; it is about the two mappings, not a
  sampling test, and it is not counted either.
* **Sensitivity.** For a relative deviation $\delta$ in every cell, the noncentrality of the chi-square is
  $\lambda = N\delta^2$. The class test ($N = 345{,}600$ draws, 719 degrees of freedom, null standard deviation
  $\sqrt{2\cdot719} = 37.9$) rejects a $\pm5\%$ bias ($\lambda = 864$) at once and cannot see a 1% bias
  ($\lambda = 35$); the test of the 11,520 indices, with 30 expected draws per index and 16 times the degrees
  of freedom, is less sensitive per unit of $\lambda$. Negative controls in the tests show each family
  rejecting a planted deviation (a wrong $\beta$, a biased target, a biased Clifford sample, a wrong rate,
  correlated masks, masks that depend on the gates).

## 6. From the mathematics to the code

| mathematics | code |
|---|---|
| $C(n,2)$, $E_0(n)$ (§3.2) | `study/identity.py::circulant_edges`; test side `tests/_ws_reference.py::circulant_edges` |
| $L_n$ (§3.2) | `study/identity.py::original_edge_list`, `graphs.py::_ring_lattice`; `tests/_ws_reference.py::original_edge_list` |
| $G_g \sim \mu_{n,\beta}$ (§3.3, §3.4) | `graphs.py::watts_strogatz`; record: `study/identity.py::build_graph`, `GraphRecord` |
| Proposition 4(b), (e), (f) | `study/identity.py::validate_graph` |
| independent rewiring (§3.3) | `tests/_ws_reference.py::rewire` (with `mapping="reject"`, `"index"`, `"first"` and `rewire_misreading`) |
| exact law (§5.2) | `tests/_ws_reference.py::exact_law`, `graph_mask`, `mask_edges`, `law_p` |
| `next_double`, `next_below` (§3.4) | `_kernel/rng.py::next_double`, `next_below`; reference `tests/_reference_rng.py::Xoshiro` |
| $h(n)$, edge draw, Clifford draw (§3.5) | `_kernel/ensemble.py::gen_layer`; `mipt.py::MonitoredCircuit` (`gates_per_layer` default $n//2$) |
| $j = 16s + b$ (§3.5) | `clifford.py::Clifford.from_index`, `symplectic_group`; kernel `_kernel/tableau.py::apply_2q` |
| mask $[U < q_p]$ (§3.6) | `_kernel/ensemble.py::gen_layer` (`next_double(rng) < p`); $q_p$ = `identity.Token.float64` |
| streams (§3.7) | `_kernel/rng.py::stream_state`, `mipt.py::_keys`; the export `MonitoredCircuit.circuit` |
| seeds of graphs and runs (§3.8) | `study/identity.py::graph_seed`, `run_seed`, `token_words`; `ExperimentManifest.graph_seed`, `run_seed`; test side `tests/_ws_reference.py::graph_seed`, `run_seed`, `token_words` |
| study protocol keywords | `study/manifest.py::ExperimentManifest.monitored_circuit_kwargs` (document 02) |
| parsing and transcribing circuits in tests | `tests/_ws_reference.py::parse_circuit`, `study_draws` |
| statistics (§5.4) | `tests/_ws_reference.py::chi2_sf`, `chi2_p`, `uniform_p`, `binomial_pmf`; `tests/test_study_sampling.py` helpers |

## 7. Worked examples

The blocks share one namespace and run in order; a block may use the names an earlier one defined.

### (a) The exact law at n = 6, against the native generator

The dynamic program of §5.2 in 25 lines, with exact fractions, and 20,000 graphs from the generator, each seeded
with the study's graph seed `identity.graph_seed(17, n, "0.5", g)` (§3.8).

```python
import itertools
from fractions import Fraction

import numpy as np

from sparsegf2 import graphs, mipt
from sparsegf2.study import identity as ident


def exact_law(n, beta):
    """mu_{n,beta} by dynamic programming over graph states (a bitmask of the n(n-1)/2 pairs)."""
    bit = {p: 1 << i for i, p in enumerate(itertools.combinations(range(n), 2))}

    def b(u, v):
        return bit[(u, v) if u < v else (v, u)]

    order = [(i, (i + d) % n) for d in (1, 2) for i in range(n)]  # offset-major
    start = 0
    for a, c in order:
        start |= b(a, c)
    states = {start: Fraction(1)}
    for a, c in order:
        new = {}
        for s, q in states.items():
            assert s & b(a, c)  # the visited original edge is present
            eligible = [v for v in range(n) if v != a and not s & b(a, v)]
            if not eligible:
                new[s] = new.get(s, 0) + q
                continue
            if beta < 1:
                new[s] = new.get(s, 0) + q * (1 - beta)
            for w in eligible:
                t = (s & ~b(a, c)) | b(a, w)
                new[t] = new.get(t, 0) + q * beta / len(eligible)
        states = new
    return bit, states


n, beta = 6, Fraction(1, 2)
bit, law = exact_law(n, beta)
circulant = {tuple(sorted((i, (i + d) % n))) for d in (1, 2) for i in range(n)}


def new_edges(mask):
    return sum(1 for pair, bt in bit.items() if mask & bt and pair not in circulant)


exact_mean = sum(q * new_edges(m) for m, q in law.items())
exact_sd = float(sum(q * (new_edges(m) - exact_mean) ** 2 for m, q in law.items())) ** 0.5
total, count, seen = 20_000, 0, {}
for g in range(total):
    mask = 0
    for u, v in graphs.watts_strogatz(n, 4, 0.5, seed=ident.graph_seed(17, n, "0.5", g)).tolist():
        mask |= bit[(u, v)]
    seen[mask] = seen.get(mask, 0) + 1
    count += new_edges(mask)

assert sum(law.values()) == 1 and len(law) == 428  # 428 of the 455 simple graphs with 12 edges
assert set(seen) <= set(law)  # the generator never leaves the support
top = max(law, key=law.get)
print("most likely graph: probability", float(law[top]), "(uniform would be", 1 / 455, ")")
print("E[new edges]: exact", float(exact_mean), "sample", count / total, " exact sd", exact_sd)
assert float(law[top]) > 10 * (1 / 455)  # far from uniform (Proposition 7c)
assert abs(count / total - float(exact_mean)) < 4.5 * exact_sd / total**0.5  # a 4.5-sigma z-test
```

### (b) The abort of v1.4 §2.2 cannot fire

The program of example (a) asserts that every visited edge is present in every reachable graph, for
every branch. It is run for $n = 5, 6, 7$, and the distinctness behind Proposition 4 is checked for a
large range.

```python
for size in (5, 6, 7):
    exact_law(size, Fraction(1, 2))  # an AssertionError would mean an absent visited edge
for size in range(5, 300):
    oriented = [(i, (i + d) % size) for d in (1, 2) for i in range(size)]
    assert len({frozenset(e) for e in oriented}) == 2 * size  # Proposition 3: 2n distinct edges
four = [(i, (i + d) % 4) for d in (1, 2) for i in range(4)]
print("n = 4: 8 oriented edges but", len({frozenset(e) for e in four}), "distinct pairs")
```

### (c) Lemire's method is exact for the bounds the study uses

A Python mirror of `_kernel/rng.py::next_below` equals the kernel, and the residue count of
Proposition 5 holds for every bound up to 80 and for the Clifford bounds 720 and 11,520.

```python
from sparsegf2._kernel.rng import next_below, next_double, next_u64, seed_state

N = 2**32


def lemire(state, bound):
    """A pure-Python mirror of _kernel/rng.py::next_below, drawing 64-bit words from `state`."""
    x = int(next_u64(state)) >> 32
    m = x * bound
    low = m & (N - 1)
    if low < bound:
        threshold = (N - bound) % bound
        while low < threshold:
            x = int(next_u64(state)) >> 32
            m = x * bound
            low = m & (N - 1)
    return m >> 32


mirror, kernel = seed_state(7), seed_state(7)
for bound in (1, 2, 3, 5, 6, 24, 40, 100, 11520, 3_000_000_000):
    for _ in range(300):
        assert lemire(mirror, bound) == int(next_below(kernel, bound))


def accepted_per_value(bound):
    """For every value v < bound: the number of 32-bit draws that map to v and are accepted."""
    t = N % bound
    counts = []
    for v in range(bound):
        r = (-v * N) % bound  # the residue class of the low word l for the draws that map to v
        first = t + ((r - t) % bound)  # the smallest l >= t in that class
        counts.append(len(range(first, N, bound)))
    return counts, t


for bound in [*range(1, 80), 720, 11520]:
    counts, t = accepted_per_value(bound)
    assert set(counts) == {N // bound}  # each value: exactly floor(2^32 / bound) draws
    assert bound * (N // bound) + t == N  # the rest, t draws, are rejected
print("rejected fraction at bound 11520:", (N % 11520) / N)
```

### (d) The mapping matters, not only the law

A from-scratch rewiring (v1.4 §2.2) with the two mappings of Proposition 8. The rejection mapping equals
the native generator for every seed; the index mapping has the same law (uniform targets) and equals it for none.

```python
def rewire(n, beta, seed, mapping="reject"):
    rng = seed_state(seed)
    edges = {frozenset((i, (i + d) % n)) for i in range(n) for d in (1, 2)}
    for d in (1, 2):
        for a in range(n):
            b = (a + d) % n
            if next_double(rng) < beta:
                # the eligible set, taken before the original edge is deleted
                eligible = [v for v in range(n) if v != a and frozenset((a, v)) not in edges]
                if not eligible:
                    continue
                if mapping == "reject":  # the native mapping: next_below(n) until eligible
                    while True:
                        w = int(next_below(rng, n))
                        if w in eligible:
                            break
                else:  # the same law, another mapping: one draw below |A|, indexed
                    w = eligible[int(next_below(rng, len(eligible)))]
                edges.remove(frozenset((a, b)))
                edges.add(frozenset((a, w)))
    return np.array(sorted(tuple(sorted(e)) for e in edges))


n, beta, trials = 24, 0.5, 200
equal = {"reject": 0, "index": 0}
for g in range(trials):
    seed = ident.graph_seed(17, n, "0.5", g)
    native = graphs.watts_strogatz(n, 4, beta, seed=seed)
    for mapping in equal:
        equal[mapping] += np.array_equal(rewire(n, beta, seed, mapping), native)
print("graphs equal to the native one, of", trials, ":", equal)
assert equal == {"reject": trials, "index": 0}


def first_targets(mapping, trials=20_000):
    """The target of the first visit (0, 1) when it rewires; eligible = V minus {0, 1, 2, n-2, n-1}."""
    hist = np.zeros(n, dtype=int)
    eligible = [v for v in range(n) if v not in (0, 1, 2, n - 2, n - 1)]
    for g in range(trials):
        rng = seed_state((5, g))
        next_double(rng)  # the Bernoulli draw (taken as a success)
        if mapping == "reject":
            while True:
                w = int(next_below(rng, n))
                if w in eligible:
                    break
        else:
            w = eligible[int(next_below(rng, len(eligible)))]
        hist[w] += 1
    return hist[3 : n - 2], hist.sum() - hist[3 : n - 2].sum()


for mapping in ("reject", "index"):
    hist, outside = first_targets(mapping)
    expected = hist.sum() / (n - 5)
    chi2 = float(((hist - expected) ** 2 / expected).sum())  # 18 degrees of freedom
    print(mapping, "targets per vertex", hist.min(), "to", hist.max(), " chi-square", round(chi2, 1))
    assert outside == 0 and chi2 < 60  # uniform on the 19 eligible vertices (P(chi2_18 > 60) = 2e-6)
```

### (e) The 16-to-1 map

```python
from sparsegf2.clifford import Clifford, symplectic_group

group = symplectic_group(2)
assert len(group) == 720 and 16 * len(group) == 11520
for j in (0, 1, 15, 16, 5000, 11519):
    s, b = divmod(j, 16)
    gate = Clifford.from_index(2, j)
    assert (gate.symplectic_matrix() == group[s]).all()  # the class s is the sorted matrix
    assert gate.sign_bits().tolist() == [(b >> k) & 1 for k in range(4)]  # X0, X1, Z0, Z1
classes = np.bincount([Clifford.from_index(2, j).index() // 16 for j in range(11520)], minlength=720)
assert set(classes.tolist()) == {16}  # sixteen indices per class
```

### (f) Masks: the float64 threshold, nesting, and the prefix

The realized Bernoulli probability for three rates, then a graph, two rates and two horizons on one
trajectory.

```python
for p in (Fraction(1, 10), Fraction(1, 2), Fraction(1, 3)):
    q = Fraction(float(p))  # the float64 threshold, exactly
    real = Fraction(-((-q * 2**53) // 1), 2**53)  # ceil(2^53 q) / 2^53
    print(p, " q - p =", float(q - p), " P_real - p =", float(real - p))
    assert 0 <= real - q < Fraction(1, 2**53) and abs(real - p) < Fraction(1, 2**52)
assert Fraction(float(Fraction(1, 2))) == Fraction(1, 2)  # a dyadic rate is exact

n = 10
edges = ident.build_graph(n, "0.3", 0, 17).edges  # graph 0 of (10, 0.3) under the study's seed
RUN_SEED = ident.run_seed(29, n, "0.3", "shared")  # one run seed for every rate of the condition (10, 0.3)


def exported(p, layers, u=3, seed=RUN_SEED):
    spec = mipt.MonitoredCircuit(
        num_system=n, edges=[edges], circuits_per_graph=4, layers=layers, measurement_rate=p,
        schedule="random_edges", reference="purification", checkpoints="every_layer",
    )
    return str(spec.circuit(u, seed=seed))


def layers_of(text):
    """(gate line, measured set) per dynamics layer of an exported circuit."""
    out, gates, measured, ticks = [], "", set(), 0
    for line in text.splitlines():
        if line == "TICK":
            ticks += 1
            if ticks > 1:
                out.append((gates, measured))
            gates, measured = "", set()
        elif line.startswith("CLIFFORD_2Q"):
            gates = line
        elif line.startswith("M "):
            measured = {int(x) for x in line[2:].split()}
    return out


low, high = layers_of(exported(0.1, 30)), layers_of(exported(0.4, 30))
assert [g for g, _ in low] == [g for g, _ in high]  # the same gates at every rate
assert all(a <= b for (_, a), (_, b) in zip(low, high))  # nested masks (Proposition 12)
print("measured sites:", sum(len(m) for _, m in low), "at p = 0.1,", sum(len(m) for _, m in high), "at p = 0.4, of", 30 * n)
assert layers_of(exported(0.2, 12))[:5] == layers_of(exported(0.2, 5))  # prefix invariance (Proposition 13)
```

### (g) A trajectory from the raw generator

Stream $u$ of the circuit key, drawn by hand in the order of §3.7, equals the exported circuit.

```python
from sparsegf2._kernel.rng import stream_state

n, layers, u, p = 10, 4, 3, 0.3
seed = ident.run_seed(29, n, "0.3", "shared")
edges = ident.build_graph(n, "0.3", 0, 17).edges
spec = mipt.MonitoredCircuit(
    num_system=n, edges=[edges], circuits_per_graph=4, layers=layers, measurement_rate=p,
    schedule="random_edges", gates_per_layer=n // 2, random_orientation=False,
    reference="purification", checkpoints="every_layer",
)
key = np.random.SeedSequence(seed).generate_state(4, dtype=np.uint64)[:2].copy()  # circuit key
state = np.zeros(4, dtype=np.uint64)
stream_state(key, u, state)
lines = ["H " + " ".join(map(str, range(n))), "CX " + " ".join(f"{i} {n + i}" for i in range(n)), "TICK"]
for _ in range(layers):
    gates, indices = [], []
    for _ in range(n // 2):  # per gate: an edge, then a Clifford index
        a, b = edges[int(next_below(state, len(edges)))]
        indices.append(int(next_below(state, 11520)))
        gates.append(f"{a} {b}")
    lines.append(f"CLIFFORD_2Q({', '.join(map(str, indices))}) " + " ".join(gates))
    measured = [i for i in range(n) if next_double(state) < p]  # then one uniform per site
    if measured:
        lines.append("M " + " ".join(map(str, measured)))
    lines.append("TICK")
assert "\n".join(lines) == str(spec.circuit(u, seed=seed))
print("\n".join(lines[3:6]))
```

### (h) Sizes and rewiring probabilities are independent samples (Proposition 14)

The visit of the first rewiring attempt is read from the first draws of a graph seed. Under the study's seeds it is
independent between two sizes, and nested between two values of $\beta$ only by chance; under the native collection
convention of the first draft (one seed $(17, g)$ for every condition) it is the same draw.

```python
import math

num_graphs = 2000


def first_attempt(n, beta, seed):
    """The visit of the first rewiring decision that succeeds (2n if none): the first draws of the seed."""
    rng = seed_state(seed)
    for j in range(2 * n):
        if next_double(rng) < float(beta):
            return j
    return 2 * n


conventions = {
    "study": lambda n, beta, g: ident.graph_seed(17, n, beta, g),  # the condition is in the seed
    "shared": lambda n, beta, g: (17, g),  # one seed per g for every condition
}
q_sizes = sum(0.25 ** (j + 1) for j in range(24)) + 2.0**-49  # P(equal) for first attempts, beta = 1/2, n = 12 and 16
q_betas = 10 / 13  # P(first attempt at beta = 1/2 is not after that at beta = 0.3), up to 1e-14
for name, seed_of in conventions.items():
    equal = sum(
        first_attempt(12, "0.5", seed_of(12, "0.5", g)) == first_attempt(16, "0.5", seed_of(16, "0.5", g))
        for g in range(num_graphs)
    )
    nested = sum(
        first_attempt(16, "0.5", seed_of(16, "0.5", g)) <= first_attempt(16, "0.3", seed_of(16, "0.3", g))
        for g in range(num_graphs)
    )
    z_equal = (equal - num_graphs * q_sizes) / math.sqrt(num_graphs * q_sizes * (1 - q_sizes))
    z_nested = (nested - num_graphs * q_betas) / math.sqrt(num_graphs * q_betas * (1 - q_betas))
    print(f"{name:7s} sizes 12 and 16: equal {equal} of {num_graphs} (z = {z_equal:.1f});"
          f" betas 0.3 and 0.5: nested {nested} (z = {z_nested:.1f})")
    if name == "study":
        assert abs(z_equal) < 4.5 and abs(z_nested) < 4.5  # about a third equal, about 10/13 nested
    else:
        assert equal > 0.99 * num_graphs and nested == num_graphs  # the same draws
```

## 8. Validation map

All tests are in `tests/test_study_sampling.py` unless marked. "z-test" means $\lvert z\rvert < 4.5$; "χ²" means $p > 10^{-6}$.
The sample sizes are the ones in the tests.

| claim | test | what is checked |
|---|---|---|
| chi-square tail (§5.4) | `test_chi_square_tail_matches_closed_forms`, `test_chi_square_helper_pools_small_cells_and_detects_bias` | closed forms ($\nu$ even up to 40, 1, 3), published critical values (relative error $10^{-5}$); a 10% bias is rejected |
| native generator = independent rewiring (v1.4 §7.4, requirement 1) | `test_the_reference_rewiring_equals_the_native_generator_trace_for_trace`, `test_equality_holds_for_every_seed_form`, `test_equality_and_validity_for_arbitrary_seeds` | exact equality of edge arrays: 8,640 graphs ($n = 5, \dots, 40$, $\beta \in \lbrace 0, .05, .3, 1\rbrace$, 60 graph labels, each seeded with the study's graph seed), ten seed forms (integers, lists, pairs, and the five-, six- and seven-word tuples of §3.8), and a hypothesis property over $n \le 60$ and 32-bit master seeds that also compares the seed with the reference's independent derivation |
| Proposition 8 (mapping) | `test_the_mapping_from_random_numbers_to_targets_matters_not_only_the_law` | at most 2 of 200 graphs equal under the index mapping; it passes the uniformity test |
| Proposition 3, 4(b), (e), (f) | `test_graph_properties_hold_at_every_size_and_rewiring_probability`, `test_c_n_2_is_the_complete_graph_up_to_five_vertices_and_a_ring_after`; `tests/test_study_identity.py::test_circulant_edges_follow_the_definition`, `test_original_edge_list_is_offset_major_and_matches_the_native_ring`, `test_validate_graph_*` | exact, $n = 5, \dots, 40$ (odd and even), simple, $2n$ edges, degree $\ge 2$, $\beta = 0$ is $C(n,2)$, $n = 5$ is $K_5$ |
| Proposition 4(a) (abort) | `test_the_abort_fires_on_a_corrupted_order_and_never_on_the_real_one`, `test_no_reachable_state_lacks_the_visited_edge` | the abort fires on a corrupted order; exhaustive over all reachable graphs for $n = 5, 6, 7$ |
| Proposition 7(a) (exact law) | `test_the_native_generator_has_the_exact_law_at_n_6` | χ² over all 428 graphs at $\beta = 1/4, 1/2$ with 100,000 graphs each; a wrong $\beta$ ($+0.05$) is rejected |
| exact law, moments | `test_the_native_generator_has_the_exact_moments_at_n_7` | z-tests of the mean and of the spread (the second moment about the exact mean, with the exact fourth central moment, §5.4) of the degree variance and the new-edge count at $\beta = 1/4, 1/2, 1$, 25,000 graphs; every moment from the exact law; the twelve $\lvert z\rvert$ are below 1.9 |
| three misreadings (§3.3) | `test_each_plausible_misreading_of_the_rewiring_rule_is_rejected` | eligible set after the deletion, vertex-major order, retained endpoint $b$: each rejected at $n = 6$ with 30,000 graphs; the correct reading passes on the same seeds |
| Proposition 7(b) (decisions) | `test_rewiring_decisions_are_bernoulli_beta` | $n = 24$, 20,000 graphs each at $\beta = 0.1, 0.5$: z-test of the attempt total; χ² of the attempt histogram against Binomial(48, $\beta$); χ² and max-z over the 48 positions; z-test of lag-1 products; a wrong $\beta$ is rejected |
| Propositions 5, 6 (targets) | `test_targets_are_uniform_on_the_eligible_set_computed_before_the_deletion`, `test_rejection_sampling_uses_n_over_the_eligible_size_draws_on_average`, `test_a_biased_target_sampler_fails_the_uniformity_test` | χ² of the position among the 19 eligible vertices of each graph's first rewiring; χ² of the randomized PIT over all events; z-test of the number of draws against $\sum n/\lvert A\rvert$; "smallest eligible vertex" rejected |
| study protocol exact (v1.4 §7.3, requirement 10) | `test_the_exported_circuit_is_the_oracle_and_the_direct_transcription`, `test_each_layer_has_floor_n_over_two_gates_on_edges_with_the_low_endpoint_first` | exact: text equals `tests/_protocol_oracle.py` and a transcription of the draws for $n \in \lbrace 5, 6, 9, 16, 17\rbrace$, both gate modes; $\lfloor n/2\rfloor$ gates, $a < b$, on graph edges, for $n = 5, \dots, 11, 16, 17, 24, 25$ |
| Proposition 9 (a)–(b) | `test_edge_draws_are_uniform_over_the_graphs_edges`, `test_edges_are_drawn_with_replacement_in_sampled_order`, `test_disjoint_pairs_of_edge_draws_are_independent_and_uniform` | 345,600 draws over 48 edges: χ² pooled and by position; z-tests of the vertex incidences ($\propto$ degree, 24 tests); occupancy law χ²; collision rates 1/48 within and across layers; 256-cell χ² of disjoint pairs |
| Propositions 5, 9(c), 10 | `test_clifford_indices_are_uniform_over_11520_and_their_classes_over_720`, `test_the_clifford_test_has_the_power_to_see_a_bias`, `test_j_over_16_is_the_symplectic_class_and_j_mod_16_the_sign_bits` | χ² over 11,520 indices (30 each), 720 classes, 16 sign patterns, class $\times$ sign, consecutive pairs; a $\pm5\%$ class bias is rejected; exact: `clifford.py` classes and signs for all 11,520 indices against a brute-force enumeration of $\mathrm{Sp}(4,\mathbb F_2)$ |
| Proposition 11 (mask law) | `test_masks_are_independent_bernoulli_p_at_every_site`, `test_the_extreme_rates_measure_nothing_or_everything` | the largest of the 12 per-site z-scores as a Bonferroni-corrected $p$-value, the χ² of their squares, and a z-test of the total (8,000 layers $\times$ 12 sites); a wrong rate is rejected; $p = 0, 1$ exact |
| masks uncorrelated (v1.4 §7.4) | `test_masks_have_no_site_or_layer_correlation_beyond_chance`, `test_the_correlation_tests_have_the_power_to_see_planted_dependence` | six correlation families, χ² each; planted site and layer dependence rejected |
| masks independent of gates | `test_masks_are_independent_of_the_gate_draws` | three layer offsets of "site touched by a gate" (χ²); four z-tests of the layer count against the mean gate class and edge of the same and of the next layer, standardized with the exact Binomial and uniform moments (§5.4) |
| Proposition 12 (nesting; requirement 11) | `test_masks_are_uniform_below_the_float_threshold_and_nested_across_rates`, `test_nesting_does_not_hold_when_the_gate_mode_is_independent` | exact: 9 rates, 5 aligned $(g,c,u)$: one run seed (five words) for all rates, the same gates, mask $=$ `uniform < float64(p)`, nested chain; independent mode: three seven-word seeds, other gates, not nested |
| Proposition 14 (independent conditions) | `test_graph_collections_of_different_sizes_are_independent_samples`, `test_graph_collections_of_different_betas_are_independent_samples`, `test_circuit_streams_of_different_betas_are_independent_samples`, `test_circuit_streams_of_different_sizes_are_independent_samples`; `tests/test_study_identity.py::test_graph_seeds_and_run_seeds_never_coincide_even_for_equal_master_seeds` | z-tests with exact null laws, each with the shared seed of the first draft as negative control. Sizes 12 and 16, $\beta = 1/2$, 3,000 graphs: first rewiring attempt equal for 1,021 (null $q = 1/3$, $z = 0.81$; shared: 3,000). $\beta = 0.3$ and $0.5$ at $n = 16$: attempt at $0.5$ not after that at $0.3$ for 2,311 of 3,000 (null $10/13$, $z = 0.14$; shared: 3,000). Edge indices at $\beta = 0.1$ and $0.3$, $n = 12$: equal at 372 of 9,600 gate draws (null $1/24$, $z = -1.43$; shared: all). First gate of $n = 12$ and $16$: cells overlapping for 67 of 1,000 trajectories (null $1/16$, $z = 0.59$; shared: all) |
| Proposition 13 (requirement 6) | `test_a_longer_horizon_extends_the_circuit_prefix`, `test_unrelated_choices_leave_the_circuit_unchanged`, `test_the_simulated_code_dimension_has_a_horizon_independent_prefix` | exact: prefixes for three horizon pairs; probes, checkpoints and appended graphs leave the export unchanged; $k_t$ for $t \le 6$ equal at $T = 6$ and $14$ and under sparse checkpoints |

Covered elsewhere and not repeated: `tests/test_protocol.py` (oracle equality for 20 specs and the laws of the
generic draws), `tests/test_invariance.py` (chunks, threads, replay), `tests/test_randomness_contract.py`
(pinned digests of the streams).

## 9. Limitations and departures

* **Float thresholds.** A rate $p$ and a rewiring probability $\beta$ reach the generator as doubles compared with uniforms
  on the multiples of $2^{-53}$, so the realized probabilities are $P_{\mathrm{real}}(\mathrm{fl}(p))$ and
  $\beta' = P_{\mathrm{real}}(\mathrm{fl}(\beta))$, exact for dyadic values and within $2^{-52}$ otherwise (Propositions 7(b') and
  11; v1.4 §7.2). Statements "Bernoulli($p$)" and "$\mu_{n,\beta}$" in this document are exact for those realized values.
* **Ideal versus pseudorandom.** Everything in §4 is exact for independent uniform draws. The xoshiro256**
  generator is a deterministic sequence; the tests show no deviation at the stated sensitivity and prove
  nothing about independence (v1.4 §7.1, §7.4). The streams of trajectories and of graphs come from
  `SeedSequence` mixing, not from a proof of independence.
* **Sizes.** The study needs $n \ge 5$ (the native generator requires $k = 4 < n$); $n = 5$ is $K_5$ and is never rewired
  (v1.4 §9: nontrivial rewiring needs $n \ge 6$). The exact law is computed only for $n \le 7$ (the number of reachable
  graphs grows from 428 at $n = 6$ to 101,593 at $n = 7$ and 24,359,329 at $n = 8$); beyond, only the moments of the
  decisions and targets and the trace equality with the reference are tested.
* **Reference checks.** v1.4 §2.2 says to abort on an absent original edge without saying when. The reference checks
  the visited edge at every visit, with $B_j = 0$ as well, which is the stronger form; Proposition 4(a) holds
  for it.
* **Independence across conditions is a hash property.** The seeds of different conditions differ (document 02,
  Proposition 4), and different `SeedSequence` entropy is treated as independent streams (Proposition 14). That
  is a property of the mixing function, not a theorem; the tests of §8 look at the first draws, where a shared seed
  would show, and nothing more. Only the rates of one condition are coupled (§3.8): comparisons across $n$ or
  $\beta$ are comparisons of independent samples, and a paired analysis across them would need a different design.
* **Departures from v1.4 and the design.** D1 (no measurement replicas $r > 0$), D2 (the circuit index stride), D3
  (depth extension reruns from layer 0, Proposition 13), D4 (`gate_mode="independent"`), D5 (the absent-edge
  abort lives in the reference and the tests, Proposition 4(a)) and D6 (domain-tagged seeds that depend on
  $(n, \beta)$, ruling R-SEED, §3.8) are as listed in the design §9 and §11. No probability
  law, stream or numbering is changed.
* **The mapping is frozen.** A different but equally valid target sampler would give a valid ensemble and other
  graphs; it needs a new `GRAPH_GENERATOR_VERSION` and cannot be mixed with records made under this one
  (Proposition 8).
