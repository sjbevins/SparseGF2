# The observable catalog: first passage, distances, erasure profiles, robustness and peaks

How the Watts–Strogatz study turns the engine's probes and the stored joint tableaux into its
observables: the logical-qubit trace $k_t$ and the exact first-passage depths $\tau_\ell$ with
right-censoring; the ring-contiguous distance $d_{\mathrm{cyc}}$ and the graph-connected distance
$d_{\mathrm{conn}}$; the uniform and geometry-correlated erasure profiles $P_{\mathcal C,G,J}(m)$, the
channel failure $\Phi(\varepsilon)$, the mean leakage $\bar\ell(m, J)$ and the integrated robustness
$D(J)$; the half-cut entropy; and the peak depth $T^*$ of a robustness trace. Each comes with the
proof of every property the code relies on, its algorithm, its estimator and error, and the tests
that check it against an independent enumeration.

This document is written for a reader fluent in the stabilizer formalism and for language models.
Every symbol is defined where it first appears, every formula is the one the code computes, and
every statement about the code names the function it describes as `path::function` (paths relative
to `src/sparsegf2/`). The Python blocks run as written, in order, in one namespace, in a few seconds
once the kernels are compiled.

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

The study specification (`docs/misc/watts_strogatz_study_spec_v1_4.md`, cited as "v1.4") defines a
catalog of observables on the phase-free joint stabilizer state of each monitored trajectory (v1.4
§5.2–§5.8) and the records that hold them (v1.4 §8: `purification_trace`, `observable_snapshot`,
`observable_trajectory`, `geometry_erasure_profile`). This part, `study/observables.py`, implements
that catalog as pure functions of the engine's probe values and of stored snapshots, with a
versioned registry. It implements:

* v1.4 §5.2: $k_t$, the code rate $k_t/n$ and the stabilizer rank $r_t$; the lossless
  change-point form of the trace; the exact first-passage depths $\tau_\ell$ for every
  $\ell = 0, \dots, n$ with right-censoring at the horizon; the checkpoint upper bounds
  $\widehat\tau^{\mathrm{chk}}_\ell$;
* v1.4 §5.3: observables recomputed later from a stored snapshot (the retrospective ones), with
  deterministic seeds for the sampled ones;
* v1.4 §5.4: the reference information $I_t(A:R)$ and the failure indicator $f_t(A)$, through
  `StabilizerCode` and `PhaseFreeTableau` (`docs/study/03-phase-free-tableau.md`);
* v1.4 §5.5: $d_{\mathrm{cyc}}$ raw, with `has_code` and zero-extended, from the engine probe and
  from a stored code;
* v1.4 §5.6: $d_{\mathrm{conn}}$ by exact enumeration, refused above a configurable size;
* v1.4 §5.7: $P_{\mathcal C}(m)$, $P_{\mathcal C,G,J}(m)$, $\Phi_{\mathcal C,G,J}(\varepsilon)$,
  $\bar\ell_{\mathcal C,G,J}(m)$ and $D_{\mathcal C,G}(J)$, with their estimators and errors;
* v1.4 §5.8: the rate–robustness identities (proved in §4 and checked as mandatory tests) and the
  peak observables $d^*$, $\mathcal A$, $T^*$ with the horizon flag and $k_t$ on $\mathcal A$;
* v1.4 validation requirements 12 and 13, and the §9 rows "First-passage observables", "Native
  contiguous distance", "Peak depth", "Primary robustness" and "Geometry-sensitive follow-up".

It does not store anything (the store and runner, `docs/study/05-store-and-reproducibility.md`) and
does not average over trajectories (`docs/study/06-statistics.md`). It changes no probability law
and no random stream: it reads what the engine produced.

## 2. Notation

| symbol | meaning | type | in the code |
|---|---|---|---|
| $n$ | system qubits $Q_0, \dots, Q_{n-1}$ | integer $\ge 1$ | `num_system` |
| $V$ | the system $\lbrace 0, \dots, n-1\rbrace$, in the original ring order | set | `range(n)` |
| $R$ | the reference qubits $n, \dots, 2n-1$ | set | `range(n, 2 * n)` |
| $t$ | completed monitored layers (depth) | integer $\ge 0$ | `t` |
| $T$ | the horizon of a trajectory, $T_{\max}$ | integer | `FirstPassage.horizon`, `Peak.horizon` |
| $\mathcal T$, $\mathcal T_+$ | the checkpoint depths $\mathcal T(n,\beta)$; those $> 0$ | sets of depths | `checkpoints`, `t` of `peak` |
| $\widetilde{\mathsf S}_t$ | the phase-free joint stabilizer group (dimension $2n$) | subspace of $\mathbb F_2^{4n}$ | `PhaseFreeTableau` |
| $W_t$ | the phase-free physical group $W_t^{\mathrm{phys}}$ (elements with no reference support, on $V$) | subspace of $\mathbb F_2^{2n}$ | `StabilizerCode` (its stabilizers) |
| $H_t$ | a generator matrix of $W_t$ with $r_t$ independent rows | $r_t \times 2n$ over $\mathbb F_2$ | `StabilizerCode.stabilizer_matrix()` |
| $r_t$, $k_t$ | $\dim W_t$ and $n - r_t$, the logical qubits | integers | `stabilizer_rank`, `k` |
| $X_A$ | for a subspace $X$ and qubits $A$: its elements supported in $A$ | subspace | |
| $X\vert_A$ | the restriction of $X$ to the $x$ and $z$ coordinates of $A$ | subspace | |
| $N_t$ | the normalizer $W_t^{\perp}$: Paulis commuting with every stabilizer | subspace | |
| $I_t(A:R)$ | $S(A) + S(R) - S(A \cup R)$, bits | integer in $[0, 2k_t]$ | `StabilizerCode.lost_information`, `PhaseFreeTableau.mutual_information` |
| $f_t(A)$ | $\mathbf 1\lbrace I_t(A:R) > 0\rbrace$: the erasure of $A$ fails | 0/1 | `StabilizerCode.region_failures` |
| $\tau_\ell$ | $\min\lbrace t : k_t \le \ell\rbrace$ | depth or censored | `FirstPassage.tau`, `.observed` |
| $\widehat\tau^{\mathrm{chk}}_\ell$ | $\min\lbrace t \in \mathcal T : k_t \le \ell\rbrace$ | depth or censored | `FirstPassage.tau_chk`, `.observed_chk` |
| $A_{a,m}$ | the cyclic interval $\lbrace a, a+1, \dots, a+m-1\rbrace \bmod n$ | set | windows of `contiguous_distance` |
| $d_{\mathrm{cyc}}$, $d^0_{\mathrm{cyc}}$ | the shortest failing interval; its zero extension | integers (raw: NaN at $k = 0$) | `CyclicDistance.raw`, `.zero_extended` |
| $G = (V, E)$ | the trajectory's interaction graph | graph | `edges` |
| $\Omega_{n,m}$ | the $m$-subsets of $V$, $\lbrace A \subseteq V : \lvert A\rvert = m\rbrace$ | family of sets | |
| $\Gamma_G(A)$ | the neighborhood of $A$ in $G$: vertices adjacent to some vertex of $A$ | set | `reach` in `connected_distance` |
| $\mathcal C_m(G)$ | the $m$-subsets $A$ with $G[A]$ connected | family of sets | `connected_distance` levels |
| $d_{\mathrm{conn}}$ | the smallest connected failing region | integer, $0$ or $\infty$ | `connected_distance` |
| $b_G(A)$ | $\lvert\partial_G A\rvert$, the edges with exactly one end in $A$ | integer | `erasures.boundary` |
| $Q_{G,J}(A \mid m)$ | $e^{-J b_G(A)} / Z_{G,m}(J)$ on the $m$-subsets | law on regions | `erasures.sample_regions` |
| $P_{\mathcal C}(m)$, $P_{\mathcal C,G,J}(m)$ | failure probability of a uniform / $Q_{G,J}$ region of size $m$ | probability | `FailureProfile.curve`, `CorrelatedProfile.curve` |
| $\bar\ell(m, J)$ | $\mathbb E_{A \sim Q_{G,J}(\cdot\mid m)} I(A:R)$ | bits | `Leakage.value`, `mean_leakage_exact` |
| $b_m(\varepsilon)$ | $\binom nm \varepsilon^m (1-\varepsilon)^{n-m}$ | probability | `analysis` binomial weights |
| $\Phi(\varepsilon)$ | $\sum_m b_m(\varepsilon) P(m)$ | probability | `ChannelFailure.value` |
| $D(J)$ | $\sum_{m=0}^{n-1} (1 - P_{\mathcal C,G,J}(m))$, $0$ at $k = 0$ | real in $[1, (n+1)/2]$ or 0 | `Robustness.value`, `CorrelatedProfile.robustness` |
| $\pi$, $M_\pi$ | a uniformly random ordering of $V$; its first failing prefix size | permutation; integer | |
| $D_j$ | the engine's first-failure sizes, $j = 1, \dots, S$ ($n+1$: never fails) | integers | `erasure_first_failure` values |
| $S$ | orderings (or regions) per evaluation | integer | `samples` |
| $w$ | the exhaustive cutoff: exact counts for $m = 1, \dots, w$ | integer | `max_exact`, width of `exact_counts` |
| $Y_j$, $Z_j$ | $\max(D_j, w+1) - (w+1)$; $T_\varepsilon(\max(D_j, w+1))$ | per-ordering values | in `integrated_robustness`, `uniform_channel_failure` |
| $T_\varepsilon(d)$ | $\Pr[\mathrm{Bin}(n, \varepsilon) \ge d]$ | probability | `_binomial_tails` |
| $d(t)$, $\mathrm{se}(t)$ | a depth trace at the checkpoints and its standard error | reals | `d`, `se` of `peak` |
| $d^*$, $\mathcal A$, $T^*$ | the maximum over $\mathcal T_+$, the maximizing set, its log midpoint | real, set, real | `Peak.d_star`, `.maximizing_set`, `.t_star` |
| $z$ | the plateau width in standard errors | real $\ge 0$ | `z` |
| $\Phi_{\mathcal N}$ | the standard normal distribution function (not the channel failure $\Phi$) | function | |
| $\tau_{\mathrm{int}}$ | the integrated autocorrelation time of a chain's records | real $\ge 1$ | `erasures.chain_diagnostics` |
| $\xi$ | the trajectory key $(\text{experiment}, n, \beta, p, g, c, r)$ | identity | `identity.TrajectoryKey` |
| $w(x)$ | the 32-bit words (low, high) of `blake2b(x, digest_size=8)` read little-endian | two integers | `observables._text_words` |

Subscripts $t$ are dropped when one depth is meant. All entropies and informations are in bits.

## 3. Definitions

Everything here is stated exactly as implemented. Where v1.4 left a convention open, the choice
and its reason are given in a **Convention** paragraph.

### 3.1 The code and its parameters (v1.4 §5, §5.2)

After $t$ layers the joint system–reference state is pure; its phase-free stabilizer group
$\widetilde{\mathsf S}_t$ has dimension $2n$. Its elements with no reference support, restricted
to $V$, form $W_t$, of dimension $r_t$; the code holds $k_t = n - r_t$ logical qubits, its rate is
$R_t = k_t/n$, and $k_t = S(R)$ (`docs/study/03-phase-free-tableau.md`, Propositions 7 and 8). At
$t = 0$ (Bell pairs) $r_0 = 0$ and $k_0 = n$. `observables.py::code_parameters` returns
$(k_t, k_t/n, n - k_t)$ for an array of $k$ values.

### 3.2 The $k$ trace and its change points (v1.4 §5.2, §5.3)

The trace is $k_0, k_1, \dots, k_T$, one value per completed layer (the engine's `code_dimension`
probe with `checkpoints="every_layer"`). Its **change-point form** is the list of rows
$(t_j, k_{t_j})$: the layer $0$, every layer $t \ge 1$ with $k_t \lt k_{t-1}$, in increasing order,
and the horizon $T$ as a last row when it is not already one of them.

**Convention.** v1.4 asks for a lossless run-length or change-point form. The last row is always
the horizon, so the form alone determines $T$ and the inverse needs nothing else; every other row
is a strict decrease, so the form is canonical (one trace, one array).
`k_change_points` and `k_from_change_points` convert; an increase anywhere is refused with the
layer named, since a trace of one trajectory never increases (Proposition 1).

### 3.3 First passage and the checkpoint bounds (v1.4 §5.2)

For $\ell = 0, \dots, n$,

$$
\tau_\ell = \min\lbrace t \in \lbrace 0, \dots, T\rbrace : k_t \le \ell\rbrace,
\qquad
\widehat\tau^{\mathrm{chk}}_\ell = \min\lbrace t \in \mathcal T : k_t \le \ell\rbrace .
$$

$\tau_\ell$ is **observed** when the set is non-empty, that is when $k_T \le \ell$; otherwise it is
**right-censored at the horizon** $T$: the event happens after $T$ (possibly never). In
particular $\tau_n = 0$, $\tau_0$ is the purification depth of the full reference, and $\tau_1$ is
the first depth with at most one logical qubit (not exactly one).

**Convention.** A censored threshold is never given a value: `FirstPassage.tau[ell]` holds $-1$
as a marker (the engine's convention for `purification_layer`, and the store's, design §4),
`observed[ell]` is False, `to_dict()` writes `None`, and `survival_data(ell)` gives the pair
$(T, \text{False})$ that survival estimators take. `censor_depth` is $T$. The checkpoint bound is
recorded the same way.

### 3.4 Reference information and failure (v1.4 §5.4)

For $A \subseteq V$ and $B = V \setminus A$, with $r_t(X) = \dim (W_t)_X$,

$$
I_t(A:R) = k_t + 2\lvert A\rvert - n - r_t(A) + r_t(B),
\qquad
f_t(A) = \mathbf 1\lbrace I_t(A:R) > 0\rbrace .
$$

Proposition 6 proves this equals $S(A) + S(R) - S(A \cup R)$ of the joint state and
$\dim (N_t)_A - \dim (W_t)_A$, the number of independent logical operators supported in $A$.
`StabilizerCode.lost_information` computes it from the physical code
($2\lvert A\rvert - s - \operatorname{rank} H\vert_A + \operatorname{rank} H\vert_B$, `codes.py`
module docstring) and `PhaseFreeTableau.mutual_information(A, R)` from the joint state.

### 3.5 The ring-contiguous distance (v1.4 §5.5)

With the cyclic intervals $A_{a,m} = \lbrace (a + j) \bmod n : 0 \le j \lt m\rbrace$ of the original
ring order (identity order, periodic, at every $\beta$),

$$
d_{\mathrm{cyc}}(t) = \min\lbrace m \in \lbrace 1, \dots, n\rbrace : \exists a,\ f_t(A_{a,m}) = 1\rbrace \quad (k_t > 0).
$$

The engine records **raw** values: $d_{\mathrm{cyc}}$ where $k_t > 0$, NaN where $k_t = 0$.
`cyclic_distance(raw, k)` returns the raw values, `has_code` $= \mathbf 1\lbrace k_t > 0\rbrace$ and
the **zero extension** $d^0_{\mathrm{cyc}}(t) = d_{\mathrm{cyc}}(t)$ for $k_t > 0$ and $0$ for
$k_t = 0$, the analysis-only form for unconditional peak traces (v1.4 §5.5, §9).
`cyclic_distance_from_code(code)` computes the same from a stored code
(`StabilizerCode.contiguous_distance(periodic=True)`, identity order).

### 3.6 The graph-connected distance (v1.4 §5.6)

With $\mathcal C_m(G) = \lbrace A : \lvert A\rvert = m, G[A] \text{ connected}\rbrace$,

$$
d_{\mathrm{conn}}(G, t) = \begin{cases} 0, & k_t = 0,\\ \min\lbrace \lvert A\rvert : A \in \textstyle\bigcup_m \mathcal C_m(G),\ f_t(A) = 1\rbrace, & k_t > 0,\end{cases}
$$

and $+\infty$ when no connected region fails. `connected_distance(code, edges, max_n=20)`.

**Convention.** v1.4 gives the value $0$ at $k_t = 0$ directly (unlike $d_{\mathrm{cyc}}$, there is
no native probe with a NaN convention to preserve); the function follows v1.4, and the retrospective
record carries `has_code` beside it. Sizes above `max_n` are refused (the enumeration is
exponential); the caller raises `max_n` to accept the cost.

### 3.7 Erasure laws and their failure statistics (v1.4 §5.7)

For a graph $G$, $b_G(A) = \lvert\lbrace\lbrace i, j\rbrace \in E : i \in A, j \notin A\rbrace\rvert$ and
the fixed-size region law at coupling $J$ is

$$
Q_{G,J}(A \mid m) = \frac{\mathbf 1\lbrace\lvert A\rvert = m\rbrace\, e^{-J b_G(A)}}{Z_{G,m}(J)},
\qquad Z_{G,m}(J) = \sum_{\lvert B\rvert = m} e^{-J b_G(B)} .
$$

The failure probability, the mean leakage, the channel failure and the integrated robustness are

$$
P_{\mathcal C,G,J}(m) = \sum_{\lvert A\rvert = m} Q_{G,J}(A \mid m)\, f(A),
\qquad
\bar\ell(m, J) = \sum_{\lvert A\rvert = m} Q_{G,J}(A \mid m)\, I(A:R),
$$

$$
\Phi_{\mathcal C,G,J}(\varepsilon) = \sum_{m=0}^{n} b_m(\varepsilon)\, P_{\mathcal C,G,J}(m),
\qquad
D_{\mathcal C,G}(J) = \begin{cases} 0, & k = 0,\\ \sum_{m=0}^{n-1}\big(1 - P_{\mathcal C,G,J}(m)\big), & k > 0.\end{cases}
$$

At $J = 0$ the law is uniform and $P_{\mathcal C,G,0}(m) = P_{\mathcal C}(m) = \binom nm^{-1}\sum_{\lvert A\rvert = m} f(A)$.
$\varepsilon$ is an erasure probability, not the measurement rate $p$. v1.4 states $J \ge 0$; the
code accepts any $\lvert J\rvert \le 10^6$ (`erasures.MAX_ABS_COUPLING`), and every identity of §4
holds for any real $J$.

### 3.8 The half-cut entropy

$S(A)$ of the system qubits $A = \lbrace 0, \dots, \lfloor n/2\rfloor - 1\rbrace$, from the stored
joint state: `half_cut_entropy(tableau, n)`. **Convention.** The first half of the identity order
with $\lfloor n/2\rfloor$ qubits, which is the region of the engine's `half_cut_entropy` probe
(`mipt.py::MonitoredCircuit._resolve_probe`: `system_order[: n // 2]`), so the two agree.

### 3.9 Peaks (v1.4 §5.8)

For a nonnegative trace $d(t)$ at the checkpoints, over $\mathcal T_+ = \mathcal T \setminus \lbrace 0\rbrace$:

$$
d^* = \max_{t \in \mathcal T_+} d(t), \qquad
T^* = \exp\Big(\frac{1}{\lvert\mathcal A\rvert}\sum_{t \in \mathcal A} \ln t\Big),
$$

with the maximizing set $\mathcal A$ of the rule fixed in advance:

* **exact** (integer or exact traces, such as $d^0_{\mathrm{cyc}}$):
  $\mathcal A = \lbrace t \in \mathcal T_+ : d(t) = d^*\rbrace$;
* **plateau** (Monte Carlo traces, such as $D(0)$ estimated from orderings or $D(J)$ estimated
  from sampled regions, with errors $\mathrm{se}(t)$ and a width $z \ge 0$):
  $\mathcal A = \lbrace t \in \mathcal T_+ : d(t) \ge d^* - z\sqrt{\mathrm{se}(t)^2 + \mathrm{se}(t^\circ)^2}\rbrace$,
  $t^\circ$ the first (smallest) maximizer. The width means $z$ standard deviations only when
  $\mathrm{se}$ are the true standard errors: $D(0)$ from the orderings, and $D(J)$ from exact
  samples (one cycle or one path). For $D(J)$ from the swap chain (every Watts–Strogatz
  interaction graph, at every $J$) the probe's binomial errors are lower bounds and the plateau
  is narrower than it says (§4.6, §9).

$T^*$ is **missing** when $d^* = 0$ or the trace is invalid: no positive checkpoint, a value in
$\mathcal T_+$ that is NaN, infinite or negative, or (plateau) a missing or invalid error.
`at_horizon` says whether $T_{\max}$ is in $\mathcal A$ (False when missing); `k_on_set` lists $k_t$
for $t \in \mathcal A$; `first_argmax` is $t^\circ$, the checkpoint peak; `t_star_is_checkpoint`
says whether the interpolated summary $T^*$ is itself a checkpoint; `se_exact` records whether the
plateau's errors were the true ones (True), known lower bounds (False: a chain-sampled $D(J)$,
`CorrelatedProfile.se_exact`) or not stated (None; always None for the exact rule).

**Conventions.** (i) The plateau uses the first maximizer $t^\circ$ when several estimates tie
exactly (the design leaves it open; ties between Monte Carlo estimates are rare, and the first is
deterministic). (ii) A singleton $\mathcal A = \lbrace t\rbrace$ gives $T^* = t$ exactly (not
$e^{\ln t}$ rounded). (iii) $T^*$ is a checkpoint "to a relative $10^{-12}$". (iv) The entry at
$t = 0$ is ignored (v1.4: $\mathcal T_+$).

### 3.10 The registry, versions and keys (design §5)

An entry `ObservableDef(id, version, level, inputs, params, definition, function, compute, kinds)`
names an observable, the version of its implementation, where it lives (`snapshot`, `trajectory`,
`aggregate`, `scaling`), what it reads (`k_trace`, `joint_tableau`, `physical_code`, `graph`,
`probe:<kind>`, `trace:<id>`), its parameters with their defaults (`REQUIRED` when there is none),
its formula, the function that computes it, for an observable computed later from a stored
snapshot `compute(snapshot, params, entry)`, and the kind of each parameter. A stored value is
keyed by `(id, version, params_key)` with `params_key` the canonical JSON of the resolved
parameters (sorted keys, no spaces, defaults filled in): a changed implementation needs a new
version, and `register` refuses a different entry under an existing `(id, version)`. The 19
built-in entries are listed in §7(h).

**Convention (canonical parameters).** A key must be one text per parameter set, so every
parameter has a kind (`KINDS`; the built-in names in `PARAM_KINDS`) with one canonical form:
`int` (an int or numpy integer; a bool, a float such as `3.0` or a string is refused), `real` (an
int, float or numpy number, finite, stored as a float with $-0.0$ written as $0.0$), `reals` (a
real or a 1-D list of them, stored as a tuple of floats), `bool`, `text`, `sizes` (a range string,
or ints) and `ladder` (a string, or reals); a trailing `?` also allows None. `resolve_params` puts
every given value in that form (and the defaults are stored in it), so $J = 1$, $1.0$ and
`numpy.float64(1)` give one key, couplings `[0, 1, 2]` and `[0.0, 1.0, 2.0]` give one key, and the
design manifest's `analysis.peak` (`sensitivity_z: [0, 1, 2]`) keys as the default. A string where
a number belongs is refused, also by `region_seed`.

**Convention (an entry computes under its own key).** `compute(id, snapshot, params, version=None)`
resolves the entry (the one registered last when `version` is None) and hands it to the entry's
`compute`, so a sampled observable seeds itself with that entry's `id` and `version`
(Proposition 18), never with another version's; the returned `ObservableValue` carries
`observable_id`, `version` and `params_key`, the key the store keeps it under. Code that computes
or stores values names the version: "registered last" changes when a new version is registered.

## 4. Propositions and proofs

Claims taken from v1.4's verified references cite them as v1.4 does; every other claim is derived
here, and the proof is given.

### 4.1 The logical-qubit trace

**Proposition 1 ($k_t$ is nonincreasing).** For Clifford gates on the system and Pauli measurements
of system qubits without outcome-dependent feedback, $k_{t+1} \le k_t$; a single measurement
lowers $k$ by at most 1.

*Proof (derived here from the stabilizer update of Li and Fisher, Eqs. (9)–(11), as cited by v1.4
§4 and §5.2).* Write $U \subseteq \widetilde{\mathsf S}$ for the elements with zero reference
part; restriction to $V$ maps $U$ one to one onto $W$ (`03-phase-free-tableau.md`, Proposition 7),
so $r = \dim U$. *A gate* $C$ on the system acts on phase-free vectors by a symplectic map $M$ of
the system coordinates, the identity on the reference: $\widetilde{\mathsf S}' = (M \oplus I)\widetilde{\mathsf S}$.
It maps vectors with zero reference part to vectors with zero reference part and is invertible,
so $U' = (M \oplus I)U$ and $r' = r$. *A measurement* of $Z_q$, $q \in V$, has the vector $\mu$
(zero reference part). The pure state's group is maximal isotropic, $\widetilde{\mathsf S}^\perp = \widetilde{\mathsf S}$.
If $\mu \in \widetilde{\mathsf S}^\perp$ the outcome is deterministic and nothing changes.
Otherwise $\widetilde{\mathsf S}' = (\widetilde{\mathsf S} \cap \mu^\perp) \oplus \mathrm{span}(\mu)$
(`03-phase-free-tableau.md`, Proposition 4). An element $a + \epsilon\mu$ ($a \in \widetilde{\mathsf S} \cap \mu^\perp$,
$\epsilon \in \mathbb F_2$) has zero reference part exactly when $a$ has, so
$U' = (U \cap \mu^\perp) \oplus \mathrm{span}(\mu)$; the sum is direct because
$\mu \notin \widetilde{\mathsf S}$. $U \cap \mu^\perp$ is the kernel of the linear functional
$\langle \cdot, \mu\rangle$ on $U$, of dimension $r$ or $r - 1$. So $r' \in \lbrace r, r+1\rbrace$
and $k' = n - r' \in \lbrace k - 1, k\rbrace$. A layer is gates followed by measurements. ∎

The same follows from entropy: $k = S(R)$; operations on the system leave $\rho_R$ unchanged, and a
measurement leaves $\rho_R = \sum_o p_o \rho_R^{(o)}$, so by concavity
$\sum_o p_o S(\rho_R^{(o)}) \le S(\rho_R)$; for stabilizer states every outcome has the same
phase-free group (`03-phase-free-tableau.md`, Proposition 4), hence the same entropy, so it cannot
increase on any outcome. The phase-free argument above is the one the code's monotonicity check
relies on.

**Proposition 2 (thresholds and the trace determine each other).** Let $k_0 \ge \dots \ge k_T$.
(a) For each $\ell$, $\lbrace t \le T : k_t \le \ell\rbrace = [\tau_\ell, T]$ when non-empty.
(b) $\tau_\ell$ is nonincreasing in $\ell$, $\tau_\ell = 0$ for $\ell \ge k_0$, and the layer
$t \ge 1$ with $k_t \lt k_{t-1}$ is the first passage of exactly the thresholds
$\ell = k_t, k_t + 1, \dots, k_{t-1} - 1$ (several at once when $k$ drops by more than one).
(c) $k_t = \min\lbrace \ell : \tau_\ell \text{ observed},\ \tau_\ell \le t\rbrace$ for every
$t \le T$. So the $n + 1$ first passages, recorded as their thresholds are crossed at the layers
where $k$ is observed, determine every $\tau_\ell$ exactly, whatever the checkpoint schedule, and
they and the change points determine each other.

*Proof (derived here).* (a) If $k_t \le \ell$ and $t' \ge t$ then $k_{t'} \le k_t \le \ell$.
(b) A larger $\ell$ has a larger set in (a), hence a smaller minimum. $k_0 \le \ell$ gives
$\tau_\ell = 0$. Threshold $\ell$ is first met at $t$ exactly when $k_t \le \ell \lt k_{t-1}$.
(c) By (a), $k_t \le \ell$ iff $\tau_\ell$ is observed and $\tau_\ell \le t$; the smallest such
$\ell$ is $k_t$. ∎

`first_passage` evaluates (b) on the change points; the tests check (a) to (c) against the
definitions on random traces.

**Proposition 3 (checkpoint bounds).** $\widehat\tau^{\mathrm{chk}}_\ell = \min\lbrace c \in \mathcal T : c \ge \tau_\ell\rbrace$;
hence $\tau_\ell \le \widehat\tau^{\mathrm{chk}}_\ell$ whenever both are observed, an observed
$\widehat\tau^{\mathrm{chk}}_\ell$ implies an observed $\tau_\ell$, and when $T \in \mathcal T$ (v1.4
§9 always includes $T_{\max}$) an observed $\tau_\ell$ implies an observed
$\widehat\tau^{\mathrm{chk}}_\ell$. The gap is less than the spacing of the checkpoints around
$\tau_\ell$.

*Proof (derived here).* By Proposition 2(a), $k_c \le \ell$ iff $c \ge \tau_\ell$ (for
$c \le T$), so the checkpoints in $[\tau_\ell, T]$ are exactly those meeting the threshold. ∎

**Proposition 4 (censoring and the zero fill).** (a) For a fixed horizon $T$ (Type I censoring),
threshold $\ell$ contributes the observation $(\tau_\ell, \text{observed})$ when $k_T \le \ell$ and
$(T, \text{censored})$ otherwise; the latter means $\tau_\ell \gt T$, including $\tau_\ell = \infty$
(for example at $p = 0$). The censoring depth does not depend on the trajectory, so it is
non-informative. (b) With `stop_when_purified`, the engine stops at $\tau_0$ and fills
$k_t = 0$ for $t \gt \tau_0$; this is the exact trace of the continued dynamics.

*Proof (derived here).* (a) is the definition of right-censoring at a fixed time. (b) $k$ is
nonincreasing (Proposition 1) and nonnegative, so $k_{\tau_0} = 0$ forces $k_t = 0$ for every
$t \ge \tau_0$. ∎

The tests run each trajectory with and without the stop and find equal traces.

### 4.2 Reference information, correctability and the complement identity

**Lemma 5 (restriction ranks).** For a subspace $X \subseteq \mathbb F_2^{2n}$ with generator
matrix $H$ ($\operatorname{rank} H = \dim X$) and a set of qubits $A$:
(a) $\dim X_A = \dim X - \operatorname{rank}(H\vert_{V \setminus A})$;
(b) if $X$ is isotropic and $N = X^\perp$ (the symplectic complement, all Paulis commuting with $X$),
$\dim N_A = 2\lvert A\rvert - \operatorname{rank}(H\vert_A)$.

*Proof (derived here).* (a) $X_A$ is the kernel of the restriction map $X \to X\vert_{V\setminus A}$.
(b) A vector $v$ supported in $A$ commutes with $h \in X$ iff
$\langle v\vert_A, h\vert_A\rangle_A = 0$, since the form only sees coordinates where $v$ can be
nonzero; so $N_A$ is the orthogonal complement of $X\vert_A$ in $\mathbb F_2^{2\lvert A\rvert}$
under the nondegenerate symplectic form of $A$, of dimension $2\lvert A\rvert - \dim X\vert_A$. ∎

v1.4 §5.4's $r_t(A) = \operatorname{rank}(H_t) - \operatorname{rank}(H_t\vert_{V\setminus A})$ is (a).

**Proposition 6 (three forms of $I(A:R)$).** For the pure joint state, $A \subseteq V$, $B = V \setminus A$,
$s = r_t$:

$$
S(A) + S(R) - S(A \cup R) \;=\; k + 2\lvert A\rvert - n - r(A) + r(B) \;=\; \dim N_A - \dim W_A .
$$

*Proof.* The first equality is `03-phase-free-tableau.md`, Proposition 11 (v1.4 §5.4, which
derives it from Gullans and Huse, Appendix E, Eqs. (E1)–(E3)). For the second (derived here):
Lemma 5 gives $r(A) = s - \operatorname{rank} H\vert_B$, $r(B) = s - \operatorname{rank} H\vert_A$
and $\dim N_A = 2\lvert A\rvert - \operatorname{rank} H\vert_A$, $\dim W_A = s - \operatorname{rank} H\vert_B$.
Then $\dim N_A - \dim W_A = 2\lvert A\rvert - s - \operatorname{rank} H\vert_A + \operatorname{rank} H\vert_B$,
and with $s = n - k$ this is $k + 2\lvert A\rvert - n + \operatorname{rank} H\vert_B - \operatorname{rank} H\vert_A = k + 2\lvert A\rvert - n - r(A) + r(B)$. ∎

$W_A \subseteq N_A$ ($W$ is isotropic), so $I(A:R) = \dim(N_A / W_A)$ counts the independent
logical operators supported in $A$, modulo the stabilizers supported in $A$.

**Proposition 7 (correctability).** $f(A) = 0$ iff the erasure of $A$ is exactly correctable.

*Proof (derived here; v1.4 cites Li and Fisher, Theorem 1 and Eqs. (12)–(14), for the
statement).* By the Knill–Laflamme conditions, an erasure of $A$ (a channel whose Kraus operators
act on $A$) is correctable iff $\Pi O \Pi = c(O)\Pi$ for every operator $O$ on $A$, with $\Pi$ the
code projector; by linearity it suffices to check Paulis $E$ supported in $A$. If $E$ anticommutes
with a stabilizer, $\Pi E \Pi = 0$. If $E$ is a stabilizer up to a phase, $\Pi E \Pi$ is a multiple
of $\Pi$. Otherwise $E \in N \setminus W$ (up to phase) is a nontrivial logical operator: it maps
the code space to itself and acts on it as a non-identity Pauli, so $\Pi E \Pi = E\Pi$ is not a
multiple of $\Pi$. So correctability holds iff $N_A \subseteq W$, i.e. $N_A = W_A$, i.e.
$I(A:R) = 0$ by Proposition 6. ∎

**Proposition 8 (monotonicity).** $A \subseteq A'$ implies $I(A:R) \le I(A':R)$, hence
$f(A) \le f(A')$.

*Proof (derived here).* $N_A \subseteq N_{A'}$, and the map $N_A/W_A \to N_{A'}/W_{A'}$,
$v + W_A \mapsto v + W_{A'}$, is injective: if $v \in N_A \cap W_{A'}$ then $v \in W$ is supported
in $A$, so $v \in W_A$. Take dimensions and use Proposition 6. (Equivalently: data processing for
the mutual information.) ∎

**Proposition 9 (complement identity, v1.4 §5.8).** $I(A:R) + I(V \setminus A : R) = 2k$; in
particular $I(V:R) = 2k$ and, when $k > 0$, $f(A) + f(V\setminus A) \ge 1$.

*Proof.* Add the middle form of Proposition 6 for $A$ and for $B$: the $r$ terms cancel and
$2\lvert A\rvert + 2\lvert B\rvert - 2n = 0$ (`03-phase-free-tableau.md`, Proposition 12). If
$k > 0$ the two informations sum to $2k > 0$, so one is positive. ∎

### 4.3 Distances

**Proposition 10 ($d_{\mathrm{cyc}}$).** Let $k > 0$. (a) The minimum exists and
$1 \le d_{\mathrm{cyc}} \le n$. (b) "Some interval of length $m$ fails" is monotone in $m$, so
bisection over $m$ finds $d_{\mathrm{cyc}}$. (c) $d_{\mathrm{cyc}} \le \lfloor (n+1)/2\rfloor$.

*Proof (derived here).* (a) $A_{0,n} = V$ and $I(V:R) = 2k > 0$ (Proposition 9). (b)
$A_{a,m} \subset A_{a,m+1}$ and $f$ is monotone (Proposition 8). (c) Let $d = d_{\mathrm{cyc}}$:
every interval of length $d - 1$ is correctable ($d - 1 = 0$: the empty set), so by Proposition 9
its complement, an interval of length $n - d + 1$, fails; hence $d \le n - d + 1$. ∎

The zero extension $d^0_{\mathrm{cyc}}$ is an analysis convention (v1.4 §5.5): $0$ where there is
no code, so a purified trajectory has no robustness and its peak is missing (§3.9).

**Proposition 11 ($d_{\mathrm{conn}}$).** (a) For $m \ge 1$,
$\mathcal C_{m+1}(G) = \lbrace A \cup \lbrace v\rbrace : A \in \mathcal C_m(G),\ v \in \Gamma_G(A) \setminus A\rbrace$,
with $\mathcal C_1(G)$ the singletons and $\Gamma_G(A)$ the neighborhood of $A$. (b) If $G$ is connected and $k > 0$, then
$1 \le d_{\mathrm{conn}} \le n$. (c) On $C(n,2)$, $d_{\mathrm{conn}} \le d_{\mathrm{cyc}}$. (d)
Neither bounds the other in general: there are codes with $d_{\mathrm{conn}}(C(8,2)) = 2 \lt d_{\mathrm{cyc}} = 3$
and, on a rewired graph, $d_{\mathrm{conn}} = 3 \gt d_{\mathrm{cyc}} = 2$. (e) $d_{\mathrm{conn}} = \infty$
is possible only for a disconnected $G$.

*Proof (derived here).* (a) "$\supseteq$": adding a neighbor keeps a set connected. "$\subseteq$":
a connected graph on $m+1 \ge 2$ vertices has a spanning tree, whose leaf $v$ can be removed
leaving a connected set $A$ with $v \in \Gamma_G(A)$. (b) $V \in \mathcal C_n(G)$ and $f(V) = 1$
(Proposition 9). (c) On $C(n,2)$ consecutive vertices are adjacent (the offset-1 edges), so every
interval is connected and the minimum is over a larger family. (d) §7(d) runs both codes: the
$[[4,2,2]]$ code on qubits $0, 2, 4, 6$ of 8 (with $Z$ on the odd qubits), where $\lbrace 0, 2\rbrace$
fails and is connected through an offset-2 edge while every interval of length 2 holds one code
qubit; and the $[[4,2,2]]$ code on $0, 1, 4, 5$ on $C(8,2)$ with $\lbrace 0,1\rbrace$ and
$\lbrace 4,5\rbrace$ rewired to $\lbrace 0,3\rbrace$ and $\lbrace 4,7\rbrace$, where the interval
$\lbrace 0, 1\rbrace$ fails but no two code qubits are adjacent. (e) is (b). ∎

v1.4 §5.6 makes the same point with $\lbrace 0, 2, 4\rbrace$, which is connected in $C(n,2)$ but not an
interval: the connected regions are a larger family than the intervals.

### 4.4 Uniform erasures

**Proposition 12 ($P_{\mathcal C}$ and its estimator).** (a) $P_{\mathcal C}(0) = 0$,
$P_{\mathcal C}(n) = \mathbf 1\lbrace k > 0\rbrace$, and $P_{\mathcal C}$ is nondecreasing in $m$.
(b) For a uniformly random ordering $\pi$ with prefixes $A_m(\pi)$ and first-failure size
$M_\pi = \min\lbrace m : f(A_m(\pi)) = 1\rbrace$ ($n + 1$ if none), $\Pr[M_\pi \le m] = P_{\mathcal C}(m)$
for every $m$ at once. (c) With $S$ independent orderings, $\hat P(m) = \frac1S\sum_j \mathbf 1\lbrace D_j \le m\rbrace$
is unbiased with binomial variance $P(1-P)/S$; with exhaustive counts $c_m$ for $m \le w$, the
hybrid curve ($c_m/\binom nm$ for $1 \le m \le w$, $\hat P(m)$ above) is unbiased term by term.
(d) Exactly: $c_m = 0$ implies $D_j \gt m$ for every $j$, and $k = 0$ implies every $D_j = n+1$
and every $c_m = 0$.

*Proof (derived here; the same steps are in [random-region-erasure.md](../observables/random-region-erasure.md) §2.4).*
(a) The empty erasure loses nothing; $I(V:R) = 2k$. Removing a uniformly random element from a
uniform $(m+1)$-subset gives a uniform $m$-subset (each $m$-subset arises from $n - m$ supersets,
each with probability $1/(m+1)$, and $(n-m)/((m+1)\binom n{m+1}) = 1/\binom nm$), and $f$ is
monotone. (b) Exactly $m!(n-m)!$ orderings have a given $m$-subset as prefix, so the prefix is
uniform; the prefixes are nested, so $M_\pi \le m$ iff $f(A_m(\pi)) = 1$ (Proposition 8). (c) Each
indicator has mean $P(m)$ by (b), and the orderings are independent draws of the probe's stream.
(d) $c_m = 0$ means no $m$-subset fails, in particular no prefix of length $m$. ∎

`uniform_failure` and the other uniform estimators refuse data that violate (d), or a code
dimension $k$ that disagrees with the samples ($k > 0$ exactly when the orderings fail).

**Proposition 13 ($D(0) = \mathbb E_\pi[M_\pi]$ and its estimator).** Let $k > 0$.
(a) $D(0) = \sum_{m=0}^{n-1}(1 - P_{\mathcal C}(m)) = \mathbb E_\pi[M_\pi]$.
(b) With $Y_j = \max(D_j, w+1) - (w+1)$,

$$
\hat D = 1 + \sum_{m=1}^{w}\Big(1 - \frac{c_m}{\binom nm}\Big) + \frac1S\sum_{j=1}^{S} Y_j
$$

equals $\sum_{m=0}^{n-1}(1 - \tilde P(m))$ of the hybrid curve $\tilde P$, is unbiased, and has
standard error $s_Y/\sqrt S$ ($s_Y$ the sample standard deviation); without counts ($w = 0$) it is
the mean of the $D_j$. (c) $\operatorname{Var}(Y) \le \operatorname{Var}(M_\pi)$: the exact terms
never add noise.

*Proof (derived here).* (a) $M_\pi$ is an integer in $[1, n]$ ($f(\emptyset) = 0$, $f(V) = 1$), so
$M_\pi = \sum_{m=0}^{n-1}\mathbf 1\lbrace M_\pi \gt m\rbrace$; take expectations and use
Proposition 12(b): $\Pr[M_\pi \gt m] = 1 - P_{\mathcal C}(m)$. (b) For $D_j \le n$,
$\sum_{m=w+1}^{n-1}\mathbf 1\lbrace D_j \gt m\rbrace = \#\lbrace m : w+1 \le m \le D_j - 1\rbrace = Y_j$,
so the sampled terms sum to the mean of the $Y_j$; the $Y_j$ are independent and identically
distributed, which gives the error. (c) $y(x) = \max(x, w+1) - (w+1)$ is 1-Lipschitz, so for an
independent copy $M'$, $\operatorname{Var} y(M) = \tfrac12\mathbb E(y(M) - y(M'))^2 \le \tfrac12\mathbb E(M - M')^2 = \operatorname{Var} M$. ∎

`integrated_robustness` computes $\hat D$ by summing the hybrid curve
(`analysis.hybrid_failure_by_size`) over $m \lt n$; `analysis.mean_first_failure` sums it to
$m = n$, the same for $k > 0$ since $\tilde P(n) = 1$. v1.4 defines $D = 0$ at $k = 0$, where every
$D_j = n + 1$: the function returns 0 with error 0.

**Proposition 14 ($\Phi_{\mathcal C}$ and its estimator).** (a) Erasing each qubit independently
with probability $\varepsilon$ fails with probability $\Phi_{\mathcal C}(\varepsilon) = \sum_m b_m(\varepsilon)P_{\mathcal C}(m)$.
(b) With $T_\varepsilon(d) = \Pr[\mathrm{Bin}(n,\varepsilon) \ge d]$ ($T_\varepsilon(n+1) = 0$) and
$Z_j = T_\varepsilon(\max(D_j, w+1))$,

$$
\hat\Phi(\varepsilon) = \sum_{m=1}^{w} b_m(\varepsilon)\frac{c_m}{\binom nm} + \frac1S\sum_j Z_j
$$

is the binomial mixture of the hybrid curve, unbiased, with standard error $s_Z/\sqrt S$.

*Proof (derived here; [random-region-erasure.md](../observables/random-region-erasure.md) §1.3 and
§4.3).* (a) The erased set has weight $\varepsilon^{\lvert A\rvert}(1-\varepsilon)^{n-\lvert A\rvert}$,
a function of $\lvert A\rvert$ only, so given its size it is uniform. (b)
$\sum_{m \ge w+1} b_m\mathbf 1\lbrace D_j \le m\rbrace = \sum_{m \ge \max(D_j, w+1)} b_m = Z_j$; the
$Z_j$ are i.i.d. ∎

### 4.5 Geometry-correlated erasures

**Proposition 15 (complement symmetry and the bounds of v1.4 §5.8).** For any real $J$:
(a) $Q_{G,J}(V\setminus A \mid n-m) = Q_{G,J}(A \mid m)$, and $P_{\mathcal C,G,0} = P_{\mathcal C}$.
If $k > 0$:
(b) $P_{\mathcal C,G,J}(m) + P_{\mathcal C,G,J}(n-m) \ge 1$ for $0 \le m \le n$;
(c) $1 \le D(J) \le (n+1)/2$.
For every $k$:
(d) $\bar\ell(m, J) + \bar\ell(n-m, J) = 2k$, $\bar\ell(0, J) = 0$, $\bar\ell(n, J) = 2k$, and
$\bar\ell(n/2, J) = k$ for even $n$.

*Proof (derived here; v1.4 states (b)–(d) and the summation step of (c)).* (a) An edge has
exactly one end in $A$ iff it has exactly one end in $V \setminus A$, so
$b_G(V\setminus A) = b_G(A)$; complementation is a bijection from $\Omega_{n,m}$ to
$\Omega_{n,n-m}$, so $Z_{G,n-m}(J) = Z_{G,m}(J)$ and the weights agree. At $J = 0$ every weight is
1 and $Z = \binom nm$. (b) By (a), $P(n-m) = \sum_{\lvert A\rvert = m} Q(A\mid m) f(V\setminus A)$, so
$P(m) + P(n-m) = \sum_A Q(A\mid m)(f(A) + f(V\setminus A)) \ge \sum_A Q(A \mid m) = 1$ by
Proposition 9. (c) Summing (b) over $m = 1, \dots, n-1$ gives $2\sum_{m=1}^{n-1}P(m) \ge n - 1$; with
$P(0) = 0$, $D = n - \sum_{m=0}^{n-1}P(m) \le n - (n-1)/2$. $D \ge 1 - P(0) = 1$ since every term is
nonnegative. (d) As in (b), $\bar\ell(m) + \bar\ell(n-m) = \sum_A Q(A\mid m)(I(A:R) + I(V\setminus A:R)) = 2k$;
$m = n/2$ is its own complement size. ∎

These are the mandatory identities of v1.4 §5.8 and validation requirement 12. The tests check
them exactly on the exact profiles of engine codes (`StabilizerCode.boundary_failure_counts`,
`mean_leakage_exact`) and against the oracle's enumeration of $Q_{G,J}$.

**Proposition 16 (the estimators of $P_{\mathcal C,G,J}$, $D(J)$ and $\Phi_J$).** The engine
probe records, for each coupling $J$ and size $m$, the failing fraction
$\hat P_J(m) = \frac1S\sum_s f(A_s)$ of $S$ regions drawn from $Q_{G,J}(\cdot\mid m)$; each size of
each evaluation draws from its own stream (`RANDOMNESS_CONTRACT = 2`;
[correlated-erasure.md](../observables/correlated-erasure.md) §5.1, §7.4), so the
$\hat P_J(m)$ of different sizes are independent given the code.
(a) For exact samples (one cycle or one path), $S\hat P_J(m) \sim \mathrm{Bin}(S, P_J(m))$. For the
chain the estimate is consistent and its bias is at most the total-variation distance of the
records from $Q_J$ (correlated-erasure.md §5.1).
(b) The full curve $\tilde P_J$ takes $\tilde P(0) = 0$, $\tilde P(n) = \mathbf 1\lbrace k>0\rbrace$, the
sampled sizes, and linear interpolation between the nearest known sizes elsewhere; it is affine in
the sampled values: $\tilde P_J(m) = \sum_j L_{mj}\hat P_J(m_j) + \mathbf 1\lbrace k>0\rbrace u_m$.
So $\hat D(J) = n - \sum_{m \lt n}\tilde P_J(m)$ and $\hat\Phi_J(\varepsilon) = \sum_m b_m\tilde P_J(m)$ are
affine in the $\hat P_J(m_j)$ with weights $a_j = \sum_{m\lt n}L_{mj}$ and $a_j(\varepsilon) = \sum_m b_m(\varepsilon)L_{mj}$, and

$$
\operatorname{Var}\hat D(J) = \sum_j a_j^2\operatorname{Var}\hat P_J(m_j),
\qquad
\operatorname{Var}\hat\Phi_J(\varepsilon) = \sum_j a_j(\varepsilon)^2\operatorname{Var}\hat P_J(m_j).
$$

(c) When every size $1, \dots, n-1$ was sampled ($L$ is the identity on them), $\hat D(J)$ and
$\hat\Phi_J$ are unbiased for exact samples. Otherwise they carry the interpolation error, which
need not vanish: $P_J(m)$ need not be monotone or linear in $m$ for $J \ne 0$
(correlated-erasure.md §6.7).

*Proof (derived here).* (a) Exact samples are i.i.d. from $Q_J$. (b) Linear interpolation and the
fixed end values make the curve affine; a sum of independent terms has the sum of the variances.
(c) Each sampled entry is unbiased and nothing is interpolated. ∎

`correlated_failure` computes the weights $a_j$ by applying the interpolation to the unit vectors
(`analysis.correlated_failure_curve` with the end values set to 0) and uses the binomial variances
$\hat P(1-\hat P)/S$. These are the true variances given the code only for exact samples
(`CorrelatedProfile.se_exact` is True only when the sampler's method is `"exact"`). For the chain
they understate the variance by about the integrated autocorrelation time $\tau_{\mathrm{int}}$
of the records, which the probe does not store, and which grows with $J$ and $n$ and falls with
the chain's `thin` (§9 gives measured factors). An ensemble mean is a different matter: the chains
of different trajectories are independent, so the error of a mean of per-trajectory estimates
computed over independent units contains the chain's variance. In the study those units are the
graphs (v1.4 §6.4): trajectories that share a graph share its disorder, so the valid error is the
graph-cluster error (`stats.graph_cluster_se` of the per-graph means; correlated-erasure.md §5.2:
"average per graph first"), and the plain spread over trajectories is valid only with one circuit
per graph. Neither error contains the bias of an incompletely equilibrated chain (Proposition
16(a)), which every trajectory run with one sampler setting shares.
`complete` is False when a size in $1, \dots, n-1$ was interpolated, and
`ChannelFailure.interpolated_mass` is $\sum_{m \text{ interpolated}} b_m(\varepsilon)$, the share of
$\Phi_J$ that leans on interpolation. §7(f) shows the bias of an interpolated $D(J)$.

**Proposition 17 (the mean-leakage estimator).** With regions $A_1, \dots, A_S$ from
`erasures.sample_regions` at $(m, J)$ and $\hat\ell = \frac1S\sum_s I(A_s:R)$: (a) for exact
samples, $\hat\ell$ is unbiased with standard error $\sigma/\sqrt S$, estimated by $s/\sqrt S$;
(b) for the chain, $\hat\ell$ is consistent, its bias is at most $2k$ times the total-variation
distance of the records from $Q_J$, and the standard error is estimated by $s/\sqrt{\mathrm{ESS}}$
with $\mathrm{ESS} = \min(S/\tau_{\mathrm{int}}, S)$ of the series $I(A_s:R)$.

*Proof (derived here).* (a) i.i.d. draws. (b) $0 \le I \le 2k$, so
$\lvert\mathbb E I(A_s) - \bar\ell\rvert \le 2k\lVert\mathcal L(A_s) - Q_J\rVert_{\mathrm{TV}}$, and the
total-variation argument of correlated-erasure.md §5.1 applies to each record. For a stationary
series with integrated autocorrelation time $\tau_{\mathrm{int}}$ the variance of the mean is
$\sigma^2\tau_{\mathrm{int}}/S$ to leading order; $\tau_{\mathrm{int}}$ is estimated by
`erasures.chain_diagnostics` (Sokal's window), and the cap at $S$ keeps the error at least the
independent one (`codes.py::CorrelatedFailure.failure_ess` uses the same cap). ∎

`mean_leakage_exact` enumerates every $m$-subset with its weight $e^{-J(b - b_0)}$, $b_0$ the
smallest boundary ($J \ge 0$) or the largest ($J \lt 0$), so every exponent is $\le 0$; it is exact
up to rounding.

**Proposition 18 (the seeds of the retrospective samplers).** The seed of the regions of
$(\xi, t, m, J)$ for observable `id` at `version` is the 19-word tuple

$$
\big(\mathtt{WSRO},\ w(\text{experiment\_id}),\ n,\ w(\beta),\ w(p),\ g,\ c,\ r,\ t,\ m,\ \mathrm{bits}(J),\ w(\text{id}),\ w(\text{version})\big)
$$

with $\mathtt{WSRO} = \mathtt{0x5753524F}$, $w(x)$ the two words (low, high) of
`blake2b(x.encode(), digest_size=8)` read little-endian (the repository's convention, design §3,
ruling R-SEED), $\beta$ and $p$ their canonical tokens, $\mathrm{bits}(J)$ the two words (low, high)
of $J$'s IEEE-754 binary64 pattern with $-0.0$ written as $0.0$, and every other field one word
(below $2^{32}$, refused otherwise). (a) Distinct inputs give distinct tuples unless two different
strings of one field share a 64-bit blake2b digest. (b) No region seed equals a graph seed or a run
seed of the study, as a `SeedSequence` entropy, whatever the master seeds. (c) The regions depend
on every field of $\xi$: the same $(g, c)$ at two rates draws independent regions; a new version
draws new ones. (d) The seed of the peak bootstrap of trajectory $\xi$ for observable `id` at
`version` is the 15-word tuple
$\big(\mathtt{WSRB},\ w(\text{experiment\_id}),\ n,\ w(\beta),\ w(p),\ g,\ c,\ r,\ w(\text{id}),\ w(\text{version})\big)$
with $\mathtt{WSRB} = \mathtt{0x57535242}$; (a) and (b) hold for it too, and no bootstrap seed
equals a region seed.

*Proof (derived here).* (a) The fields sit at fixed positions of a tuple of fixed length; the
integer fields and $\mathrm{bits}(J)$ are one to one; only $w$ can collide. (b) The graph and run
seeds start with $\mathtt{0x57534731}$ (`"WSG1"`) and $\mathtt{0x57534331}$ (`"WSC1"`) (design §3), the
region seeds with $\mathtt{0x5753524F}$: the first words differ. Every one of these tuples has at
least 4 words, so `SeedSequence` pads none of them (it pads only entropy shorter than its 4-word
pool), and each field is one word, so no field spills into the next. The generated streams then
coincide only through a collision of `SeedSequence`'s hash. (c) Every field enters the tuple.
(d) The same argument, with the first word $\mathtt{0x57535242}$, distinct from the other three. ∎

`observables.py::region_seed` builds the tuple; `mean_leakage` passes it to
`erasures.sample_regions(..., seed=list(seed))`. Through the registry, `compute` hands the entry
being computed to its `compute` (§3.10), so `id` and `version` are that entry's own: registering a
version 2 beside version 1 leaves every version-1 seed and value unchanged, and the two versions
(or a renamed copy) draw independent regions. `observables.py::bootstrap_seed` builds the tuple of
(d) for `bootstrap_peak`. It has neither $z$ nor the replicate count $B$: the bootstraps at the
widths of `sensitivity_z` resample the same orderings (paired replicates), and $B$ replicates are
the first $B$ of any larger count (the draws run replicate by replicate). **Departure note.** The
T4 task brief first keyed these seeds by the native stream index $u$ (the design table, §5, names
only a "keyed region sampler"); ruling R-SEED (design §3) makes graph and run seeds depend on
$(n, \beta)$, so $u$ alone no longer identifies a trajectory, and the controller asked for the full
key. The words of $\beta$ and $p$ come from `identity.token_words`, the helper the graph and run
seeds use; that helper parses its input as a probability token, so the experiment id, the
observable id and the version go through `observables.py::_text_words`, the same convention for
any string (a test checks that the two agree on tokens).

### 4.6 Peaks

**Proposition 19 (properties of the peak).** (a) $T^* \in [\min\mathcal A, \max\mathcal A]$, and
$T^* = t$ when $\mathcal A = \lbrace t\rbrace$. (b) The plateau rule with $z = 0$ is the exact
rule, and $\mathcal A_z \subseteq \mathcal A_{z'}$ for $z \le z'$. (c) For a checkpoint $t \le T$,
$t \ge \tau_1$ iff $k_t \le 1$, so `k_on_set` answers whether the peak lies after $\tau_1$ without
an interpolated $k(T^*)$.

*Proof (derived here).* (a) A geometric mean lies between the smallest and the largest value.
(b) With $z = 0$ the condition is $d(t) \ge d^*$, i.e. $d(t) = d^*$; the threshold
$d^* - z\sqrt{\cdot}$ decreases in $z$. (c) Proposition 2(a) with $\ell = 1$. ∎

**The plateau rule's statistical meaning (derived here; it assumes true standard errors).**
Conditional on the trajectory (its codes), the estimates at different checkpoints come from
different probe streams (keyed by the layer, `RANDOMNESS_CONTRACT = 2`), so they are independent
and $\hat d(t^\circ) - \hat d(t)$ has variance $\mathrm{Var}\,\hat d(t) + \mathrm{Var}\,\hat d(t^\circ)$.
When $\mathrm{se}$ are the true standard errors, this is $\mathrm{se}(t)^2 + \mathrm{se}(t^\circ)^2$
and a checkpoint stays in $\mathcal A$ unless its shortfall from the maximum exceeds $z$ of these
standard deviations: $\mathcal A$ is the set of checkpoints not significantly below the maximum,
one-sided, at about $\Phi_{\mathcal N}(-z)$ per comparison ($\Phi_{\mathcal N}$ the standard normal
distribution function). Because $t^\circ$ is chosen as the largest estimate, the shortfalls of true
maximizers are biased upward; a union bound controls it: if the true trace has a plateau of $q$
checkpoints with equal errors $\sigma$ (Gaussian), and every other checkpoint is far below, a
plateau checkpoint $j$ is left out only if some plateau checkpoint $i \ne j$ has
$\hat d_i - \hat d_j \gt z\sqrt2\sigma$, an event of probability $\Phi_{\mathcal N}(-z)$, so

$$
\Pr[\text{some plateau checkpoint is left out}] \le q(q-1)\,\Phi_{\mathcal N}(-z),
$$

$6\Phi_{\mathcal N}(-2) = 0.1365$ for $q = 3$, $z = 2$ (the test checks this rate). The rule is
fixed in the manifest before the data are seen (`analysis.peak.rule`, `z`); `peak_sensitivity`
reports it at every `sensitivity_z`, as v1.4 §5.8 asks.

**When the errors are not the true ones.** The statements above hold for $\hat D(0)$ from the
orderings (Proposition 13: $s_Y/\sqrt S$ estimates the true error) and for $\hat D(J)$ from exact
samples. For $\hat D(J)$ from the swap chain, which is every Watts–Strogatz interaction graph at
every $J$ (a degree-4 graph is never one cycle or one path), the binomial `robustness_se` is a
lower bound: if the true standard deviation is $\rho\,\mathrm{se}$ with $\rho \approx
\sqrt{\tau_{\mathrm{int}}} \gt 1$ at every checkpoint, the plateau spans only $z/\rho$ true
standard deviations, each comparison leaves out a true maximizer with probability about
$\Phi_{\mathcal N}(-z/\rho)$, the bound becomes $q(q-1)\Phi_{\mathcal N}(-z/\rho)$, and since $\rho$
changes with $J$, $n$, $\beta$ and `thin`, the effective width of the predeclared rule differs
between conditions (so does that of every `sensitivity_z`). Measured factors are in §9. The
record says which case applies (`Peak.se_exact`, from `CorrelatedProfile.se_exact`); a `thin` near
the chain's $\tau_{\mathrm{int}}$, measured on the graph with `erasures.sample_regions` at the
probe's settings, brings $\rho$ near 1.

**Proposition 20 (the bootstrap of $T^*$ of $D(0)$).** `bootstrap_peak` resamples, for each
replicate and independently at each checkpoint, $S$ of that checkpoint's $S$ orderings with
replacement, keeps the exact counts, recomputes $\hat D(0)$ and its error, and takes the plateau
peak. This is the nonparametric bootstrap of $\hat T^*$ given the codes: the orderings at one
checkpoint are i.i.d., and those of different checkpoints are independent (different streams), so
resampling them independently reproduces their joint law. It measures the Monte Carlo uncertainty
of $T^*$ for one trajectory, not the ensemble's (graphs are resampled by `stats.graph_bootstrap`).
Its seed is given (`None`, fresh entropy, is refused); the study's is `bootstrap_seed` of the
trajectory key and the observable's id and version (Proposition 18(d)), so a stored bootstrap is a
function of the stored data and its key. ∎

There is no such bootstrap for $D(J)$: the correlated probe stores failing fractions, not the
regions, so there is nothing to resample, and a parametric (binomial) bootstrap of the fractions
would inherit the understated chain errors above. `peak_D_J` therefore lists no `bootstrap`
parameter (§9).

## 5. Algorithms

All algorithms are deterministic; the sampled ones take their randomness from the streams named.

**5.1 Change points** (`k_change_points`, `k_from_change_points`). (1) Read $k$ as whole numbers
$\ge 0$ (NaN, fractions and negatives refused, the first named). (2) Refuse the first
$t$ with $k_t \gt k_{t-1}$. (3) Rows: $0$, every $t$ with $k_t \lt k_{t-1}$, and $T$ if missing. The
inverse checks the canonical form (first layer 0, strictly increasing layers, strict decreases
except a final repeat) and repeats each value until the next row. $O(T)$; exact (int64).

**5.2 First passage** (`first_passage`). (1) Change points as in 5.1. (2) $\tau_\ell \gets -1$;
$\tau_\ell \gets 0$ for $\ell \ge k_0$. (3) For each drop from $k_{\mathrm{before}}$ to $k_j$ at
$t_j$: $\tau_\ell \gets t_j$ for $\ell = k_j, \dots, k_{\mathrm{before}} - 1$ (Proposition 2(b)).
(4) With checkpoints: the first checkpoint $\ge \tau_\ell$ by binary search (Proposition 3).
$O(T + n + n\log\lvert\mathcal T\rvert)$; exact.

**5.3 $d_{\mathrm{cyc}}$** (`cyclic_distance`, `cyclic_distance_from_code`). From the probe: check
that the raw values are whole numbers in $[1, n]$ exactly where $k > 0$ and NaN where $k = 0$,
then zero-extend. From a code: `StabilizerCode.contiguous_distance` bisects the window length
(Proposition 10(b)) and tests every window of a length; about $n\log_2 n$ region tests, each two
GF(2) eliminations. Exact.

**5.4 $d_{\mathrm{conn}}$** (`connected_distance`). (1) Refuse $n \gt$ `max_n`. (2) $k = 0$: return
0. (3) Level 1: the singletons, as 64-bit masks. (4) For $m = 1, 2, \dots$: test every region of
the level in one batch (`StabilizerCode.region_failures`); return $m$ at the first failure;
otherwise grow every region by each neighbor outside it and deduplicate (Proposition 11(a)).
(5) Return $\infty$ when a level is empty. Cost: $\sum_{m \le d_{\mathrm{conn}}}\lvert\mathcal C_m(G)\rvert$
region tests plus $O(n)$ bit operations per region; memory one level. Exact; exponential in the
worst case, which is why `max_n` exists.

**5.5 Uniform statistics** (`uniform_failure`, `integrated_robustness`, `uniform_channel_failure`).
(1) Validate the samples (whole numbers in $[1, n+1]$, one code's samples all $n + 1$ or all
$\le n$), the counts and $k$ against them (Proposition 12(d)). (2) The hybrid curve
(`analysis.hybrid_failure_by_size`: integer counts divided once, so the ends are exact). (3)
$\hat D$: the curve summed over $m \lt n$; error from the $Y_j$ (Proposition 13). (4)
$\hat\Phi$: `analysis.erasure_failure_probability`; the tails $T_\varepsilon(d)$ for all $d$ from
the step curves $\mathbf 1\lbrace m \ge d\rbrace$ through `analysis.failure_probability_from_curve`
(binomial weights in log space); error from the $Z_j$ (Proposition 14), formed one rate at a time
(a $(\dots, S)$ array per rate, never the $(\dots, S, q)$ array of all of them, so the memory is a
few copies of the input whatever the number $q$ of rates). Time $O(S + n)$ per code for $\hat D$,
$O(Sq + n)$ per code for $\hat\Phi$ and its error, plus $O(n^2)$ per $\varepsilon$ once. Float64;
unbiased; errors are plug-in estimates (NaN with one ordering, at every rate; `se` always has the
shape of `value`). The rates are copied: the caller's array is neither shared nor made read-only.

**5.6 Correlated statistics** (`correlated_failure`, `correlated_channel_failure`). (1) Validate
shapes, distinct couplings and sizes, fractions in $[0, 1]$ that are multiples of $1/S$ (to
$10^{-9}S$), and zeros where $k = 0$. (2) The curve (`analysis.correlated_failure_curve` with
`holds_code`). (3) $\hat D(J) = \sum_{m\lt n}(1 - \tilde P(m))$, 0 where $k = 0$. (4) Interpolation
weights from the unit vectors; binomial variances; Proposition 16. (5) Carry the sampler's
configuration (`MonitoredCircuit.correlated_sampler(name)`: method, rungs, step, burn-in, thinning,
tempering) as given. $O(c\,s + n)$ per code, $O(s\,n)$ once for the weights.

**5.7 Mean leakage** (`mean_leakage`, `mean_leakage_exact`). Sampled: (1) the seed of
Proposition 18; (2) `erasures.sample_regions(edges, n, m, [J], S, ..., seed=list(seed))` (exact on
a cycle or a path, else the swap chain with replica exchange; correlated-erasure.md §4); (3)
$I(A_s:R)$ for each region; (4) mean, ESS, error (Proposition 17). $k = 0$ or $m = 0$ return 0
without drawing. Cost: the sampler's (correlated-erasure.md §8.1) plus $S$ region evaluations.
Exact: every $m$-subset, its boundary from the membership matrix, $I(A:R)$ per region; refused
above `budget` regions (default two million); $O(\binom nm(\lvert E\rvert + \text{elimination}))$.

**5.8 Peaks** (`peak`, `peak_sensitivity`, `bootstrap_peak`). (1) Validate the checkpoints
(strictly increasing), values, errors, $k$. (2) Restrict to $t \gt 0$; missing with a reason if
invalid. (3) $d^*$, $t^\circ$, the set of the rule, $T^*$ ($e^{\overline{\ln t}}$, a singleton
exactly), the flags (with `se_exact` as given). $O(\lvert\mathcal T\rvert)$. The bootstrap checks
its seed (a non-negative integer or a sequence of them; `None`, floats, strings and `SeedSequence`
objects are refused), draws $B\cdot\lvert\mathcal T\rvert\cdot S$ indices from
`PCG64(SeedSequence(seed)).random_raw` (NumPy's policy, NEP 19, keeps a bit generator's raw stream
unchanged across versions) in the order replicate, checkpoint, sample, as $\lfloor uS\rfloor$ with
$u$ the top 53 bits (bias below $S\cdot 2^{-53}$ per draw), then evaluates all replicates at once.
$O(B\,\lvert\mathcal T\rvert\,S)$. `bootstrap_seed` builds the tuple of Proposition 18(d).

**5.9 The registry** (`register`, `get`, `resolve_params`, `params_key`, `compute`). Entries are
held in a dictionary keyed by `(id, version)`; `get` without a version returns the one registered
last. Building an entry gives every parameter a kind (its own `kinds`, else `PARAM_KINDS`; a
parameter with neither is refused) and stores the defaults in canonical form. `resolve_params`
refuses unknown names (listing the allowed ones) and missing required ones, and puts every given
value in the canonical form of its kind (§3.10), refusing what does not fit; `params_key` is
`identity.canonical_json` of the result. `SnapshotInput` checks its stored data against $n$: the
key's $n$, the code's qubits ($n$) and the joint tableau's ($n$, $n + 1$ or $2n$: no reference, one
reference, purification), so a snapshot paired with another condition's $n$ is refused, not
computed into plausible wrong values. `compute` resolves the entry and the parameters, calls
`entry.compute(snapshot, params, entry)` and returns the value with its key (`observable_id`,
`version`, `params_key`). `catalog_markdown` escapes `|` in every cell (`\|`), so every row has
the header's six cells.

## 6. From the mathematics to the code

| mathematics | code |
|---|---|
| $k_t$, $R_t$, $r_t$ (§3.1) | `study/observables.py::code_parameters` |
| change points (§3.2, 5.1) | `k_change_points`, `k_from_change_points` |
| $\tau_\ell$, censoring, $\widehat\tau^{\mathrm{chk}}_\ell$ (§3.3, Propositions 2–4) | `first_passage`, `FirstPassage` (`tau`, `observed`, `horizon`, `censor_depth`, `tau_chk`, `observed_chk`, `survival_data`, `to_dict`) |
| $I(A:R)$, $f(A)$ (§3.4, Propositions 6–9) | `codes.py::StabilizerCode.lost_information`, `is_correctable`, `region_failures`; `tableau.py::PhaseFreeTableau.mutual_information` |
| $d_{\mathrm{cyc}}$ raw, `has_code`, $d^0_{\mathrm{cyc}}$ (§3.5, Proposition 10) | `cyclic_distance`, `CyclicDistance`; from a code `cyclic_distance_from_code` (`StabilizerCode.contiguous_distance`) |
| $d_{\mathrm{conn}}$ (§3.6, Proposition 11) | `connected_distance` |
| half-cut $S(A)$ (§3.8) | `half_cut_entropy` (`PhaseFreeTableau.entropy`) |
| $P_{\mathcal C}(m)$ hybrid (Proposition 12) | `uniform_failure`, `FailureProfile`; `analysis.hybrid_failure_by_size` |
| $D(0)$ and its error (Proposition 13) | `integrated_robustness`, `Robustness` |
| $\Phi_{\mathcal C}(\varepsilon)$ and its error (Proposition 14) | `uniform_channel_failure`, `ChannelFailure`; `analysis.erasure_failure_probability` |
| $P_{\mathcal C,G,J}(m)$, $D(J)$, errors, completeness (Proposition 16) | `correlated_failure`, `CorrelatedProfile` (`se_exact`: are the errors the true ones); `analysis.correlated_failure_curve` |
| $\Phi_J(\varepsilon)$, interpolated mass (Proposition 16) | `correlated_channel_failure`; `analysis.failure_probability_from_curve` |
| $\bar\ell(m, J)$ sampled (Proposition 17) | `mean_leakage`, `Leakage`; `erasures.sample_regions`, `erasures.chain_diagnostics` |
| $\bar\ell(m, J)$ exact | `mean_leakage_exact` |
| seeds (Proposition 18) | `region_seed`, `REGION_SEED_TAG`, `REGION_SEED_WORDS`, `bootstrap_seed`, `BOOTSTRAP_SEED_TAG`, `BOOTSTRAP_SEED_WORDS`, `_key_words`, `_text_words` |
| $d^*$, $\mathcal A$, $T^*$, flags (§3.9, Proposition 19), the plateau's errors (§4.6) | `peak` (`se_exact`), `Peak` |
| sensitivity, bootstrap (Proposition 20) | `peak_sensitivity`, `bootstrap_peak`, `PeakBootstrap` |
| the registry (§3.10) | `ObservableDef`, `register`, `get`, `catalog`, `catalog_markdown`, `resolve_params`, `params_key`, `compute`, `SnapshotInput`, `ObservableValue`, `LEVELS`, `INPUTS`, `REQUIRED`, `KINDS`, `PARAM_KINDS` |

Stored fields (the design §4 `observations` columns are `value`, `se`, `data`, `method`):

| field | formula |
|---|---|
| `FirstPassage.tau[ell]` | $\tau_\ell$ if $k_T \le \ell$, else $-1$ (censored at `horizon` $= T$) |
| `FirstPassage.tau_chk[ell]` | $\min\lbrace c \in \mathcal T : c \ge \tau_\ell\rbrace$, or $-1$ |
| `CyclicDistance.zero_extended` | $d_{\mathrm{cyc}}$ where $k > 0$, else 0 |
| `FailureProfile.curve[m]`, `.se[m]` | hybrid $\tilde P(m)$; 0 on exact entries, $\sqrt{\tilde P(1-\tilde P)/S}$ on sampled ones |
| `Robustness.value`, `.se` | $\hat D(0)$ of Proposition 13 (0 at $k = 0$); $s_Y/\sqrt S$ |
| `ChannelFailure.value`, `.se`, `.interpolated_mass` | $\hat\Phi(\varepsilon)$; $s_Z/\sqrt S$ or $\sqrt{\sum_j a_j(\varepsilon)^2\hat P_j(1-\hat P_j)/S}$; $\sum_{m \text{ interp.}}b_m(\varepsilon)$ |
| `CorrelatedProfile.robustness`, `.robustness_se`, `.complete` | $\hat D(J)$; $\sqrt{\sum_j a_j^2\hat P_j(1-\hat P_j)/S}$; no size in $1..n-1$ interpolated |
| `Leakage.value`, `.se`, `.ess`, `.seed` | $\hat\ell$; $s/\sqrt{\mathrm{ESS}}$; $S$ or $\min(S/\tau_{\mathrm{int}}, S)$; Proposition 18 |
| `Peak.t_star`, `.maximizing_set`, `.at_horizon`, `.k_on_set`, `.first_argmax`, `.t_star_is_checkpoint`, `.se_exact` | §3.9 |
| `PeakBootstrap.t_star`, `.seed` | $T^*$ per replicate (Proposition 20); the seed used (`bootstrap_seed`) |
| `ObservableValue` of `d_cyc` / `d_conn` / `half_cut_entropy` / `k` | the value; `se` 0 (exact); `data` has `has_code`, `zero_extended`, `k` where relevant |
| `ObservableValue.observable_id`, `.version`, `.params_key` | the key `compute` computed it under (§3.10) |

## 7. Worked examples

### (a) A trace, its change points and its first passages

A drop of 4 at layer 3 and of 2 at layer 7: the thresholds 2 to 5 are all first met at layer 3
(Proposition 2(b)); the checkpoint bound of each is the first checkpoint at or after it
(Proposition 3). A trace that stops at $k = 3$ leaves $\tau_0, \tau_1, \tau_2$ censored at the
horizon, with no value.

```python
import hashlib
import itertools
import math
import struct
from fractions import Fraction

import numpy as np

from sparsegf2 import StabilizerCode, graphs, mipt
from sparsegf2.study import observables as ob
from sparsegf2.study.identity import TrajectoryKey

k = [6, 6, 6, 2, 2, 2, 2, 0]
points = ob.k_change_points(k)
assert points.tolist() == [[0, 6], [3, 2], [7, 0]]
assert ob.k_from_change_points(points).tolist() == k
fp = ob.first_passage(k, checkpoints=[0, 1, 4, 7])
assert fp.tau.tolist() == [7, 7, 3, 3, 3, 3, 0]
assert fp.tau_chk.tolist() == [7, 7, 4, 4, 4, 4, 0]
# Proposition 2(c): the passage times give back the trace
assert all(k[t] == min(ell for ell in range(7) if fp.tau[ell] <= t) for t in range(8))
censored = ob.first_passage([4, 4, 3, 3, 3])
assert censored.tau.tolist() == [-1, -1, -1, 2, 0]
assert censored.survival_data(0) == (4, False)  # right-censored at the horizon T = 4
assert censored.to_dict()["tau"] == [None, None, None, 2, 0]
```

### (b) $D(0) = \mathbb E_\pi[M_\pi]$ over all $6!$ orderings

The $[[4,2,2]]$ code on qubits $0, 2, 3, 5$ of 6 (qubits 1 and 4 frozen by $Z$): every pair of code
qubits fails and no single qubit does, so $P_{\mathcal C} = (0, 0, 2/5, 4/5, 1, 1, 1)$ and
$D(0) = 1 + 1 + 3/5 + 1/5 = 14/5$. The first-failure sizes of all 720 orderings average to exactly
that (Proposition 13(a)), and their distribution function is $P_{\mathcal C}$ (Proposition 12(b)).
With 400 sampled orderings the plain estimate is $2.85 \pm 0.038$; with the exhaustive counts up to
$w = 3$ every size $m \ge 4$ fails, every $Y_j = 0$, and the hybrid estimate is exactly $2.8$ with
error 0 (Proposition 13(b), (c)).

```python
code = StabilizerCode(["X_XX_X", "Z_ZZ_Z", "_Z____", "____Z_"])
n = 6
fails = {
    a: code.lost_information(a) > 0 for m in range(n + 1) for a in itertools.combinations(range(n), m)
}
P = [
    Fraction(sum(fails[a] for a in itertools.combinations(range(n), m)), math.comb(n, m))
    for m in range(n + 1)
]
assert P == [0, 0, Fraction(2, 5), Fraction(4, 5), 1, 1, 1]
first = [
    next(m for m in range(1, n + 1) if fails[tuple(sorted(perm[:m]))])
    for perm in itertools.permutations(range(n))
]
D0 = sum(1 - p for p in P[:n])
assert D0 == Fraction(14, 5) == Fraction(sum(first), len(first))
assert all(Fraction(sum(f <= m for f in first), len(first)) == P[m] for m in range(n + 1))
samples = code.first_failure_sizes(400, seed=1)
plain = ob.integrated_robustness(samples, n)
hybrid = ob.integrated_robustness(samples, n, exact_counts=code.uncorrectable_counts(3))
assert abs(float(plain.value) - 2.8) <= 4.5 * float(plain.se)  # 2.85 +- 0.038
assert float(hybrid.value) == 2.8 and float(hybrid.se) == 0.0
```

### (c) The identities of v1.4 §5.8 on a real code

A Watts–Strogatz trajectory at $n = 8$: the code with the fewest logical qubits at depth 16
($k = 2$). For every one of the 256 regions, $I(A:R)$ from the code equals the joint state's, the
complement identity holds, and correctability is $I = 0$ (Propositions 6, 7, 9). The exact
profiles (every region enumerated) satisfy $P(m) + P(n-m) \ge 1$ and $1 \le D(J) \le 9/2$ (here
$D = 2.796, 2.713, 2.678$ at $J = 0, 1, 3$), and the exact mean leakage satisfies
$\bar\ell(m) + \bar\ell(8-m) = 4$ and $\bar\ell(4) = k = 2$ (Proposition 15).

```python
n = 8
edges = graphs.watts_strogatz(n, 4, 0.2, seed=(17, 0))
spec = mipt.MonitoredCircuit(
    num_system=n,
    edges=edges,
    layers=16,
    measurement_rate=0.08,
    reference="purification",
    schedule="random_edges",
    gates_per_layer=n // 2,
    random_orientation=False,
    checkpoints="every_layer",
    probes=[mipt.code_dimension(), mipt.tableau_snapshot(checkpoints=[16])],
)
result = spec.run(trajectories=6, seed=29)
final_k = result.values["code_dimension"][:, 16]
row = int(np.flatnonzero(final_k == final_k[final_k > 0].min())[0])
tableau, code = result.tableau(row, 0), result.code(row, 0)
k = code.num_logical
assert k == 2
for m in range(n + 1):
    for a in itertools.combinations(range(n), m):
        info = code.lost_information(a)
        assert info == tableau.mutual_information(a, range(n, 2 * n))
        assert info + code.lost_information([q for q in range(n) if q not in a]) == 2 * k
        assert code.is_correctable(a) == (info == 0)
couplings = [0.0, 1.0, 3.0]
exact = code.boundary_failure_counts(edges, n).exact(couplings)  # P_J(m), m = 1..n
curve = np.concatenate([np.zeros((3, 1)), exact], axis=1)  # m = 0..n
assert np.all(curve + curve[:, ::-1] >= 1 - 1e-12)  # P(m) + P(n - m) >= 1
d_exact = (1 - curve[:, :n]).sum(axis=1)
assert np.all((1 <= d_exact) & (d_exact <= (n + 1) / 2))
leak = np.array([ob.mean_leakage_exact(code, edges, m, couplings) for m in range(n + 1)])
assert np.allclose(leak + leak[::-1], 2 * k) and np.allclose(leak[n // 2], k)
```

### (d) $d_{\mathrm{conn}}$ and $d_{\mathrm{cyc}}$ differ both ways

The two codes of Proposition 11(d), and a third whose failing interval $\lbrace 7, 0\rbrace$ wraps
around the ring (read as an open chain, its first failing window would have length 3).

```python
ring8 = graphs.watts_strogatz(8, 4, 0.0, seed=0)  # C(8, 2)
even = StabilizerCode(["X_X_X_X_", "Z_Z_Z_Z_", "_Z______", "___Z____", "_____Z__", "_______Z"])
assert ob.cyclic_distance_from_code(even) == 3.0 and ob.connected_distance(even, ring8) == 2.0
block = StabilizerCode(["XX__XX__", "ZZ__ZZ__", "__Z_____", "___Z____", "______Z_", "_______Z"])
rewired = [tuple(e) for e in ring8.tolist() if tuple(e) not in {(0, 1), (4, 5)}] + [(0, 3), (4, 7)]
assert ob.cyclic_distance_from_code(block) == 2.0 and ob.connected_distance(block, rewired) == 3.0
wraps = StabilizerCode(["X_X__X_X", "Z_Z__Z_Z", "_Z______", "___Z____", "____Z___", "______Z_"])
assert ob.cyclic_distance_from_code(wraps) == 2.0 and wraps.contiguous_distance(periodic=False) == 3
```

### (e) One trajectory, end to end: first passage, distances, $D(0)$, $\Phi$ and the peaks

$n = 14$, $\beta = 0.1$, $p = 0.1$, horizon 56, the geometric checkpoints $0, 1, 2, 4, \dots, 32, 56$.
For every trajectory $\tau_0$ equals the engine's `purification_layer` (or both are censored).
Trajectory 4 goes from $k = 14$ to $k = 1$; $\tau_1 = 10$ while its checkpoint bound is 16, and
$\tau_0$ is censored. Its zero-extended $d_{\mathrm{cyc}}$ is $1, 1, 1, 1, 1, 3, 1, 2$: the exact
peak is $\mathcal A = \lbrace 16\rbrace$, $T^* = 16$, with $k_{16} = 1$, so the peak lies after
$\tau_1$ (Proposition 19(c)) and is not at the horizon. $\hat D(0)$ peaks at the same checkpoint
($6.33 \pm 0.10$; the next largest, $5.70 \pm 0.11$ at the horizon, is 4.0 combined errors below), so
the plateau at $z = 0, 1, 2$ is $\lbrace 16\rbrace$ (with true errors: `se_exact`), and every
replicate of the bootstrap, seeded by `bootstrap_seed` of the trajectory's key (Proposition
18(d)), agrees.

```python
n, layers = 14, 56
cps = [0, 1, 2, 4, 8, 16, 32, 56]
spec = mipt.MonitoredCircuit(
    num_system=n,
    edges=graphs.watts_strogatz(n, 4, 0.1, seed=(17, 0)),
    layers=layers,
    measurement_rate=0.1,
    reference="purification",
    schedule="random_edges",
    gates_per_layer=n // 2,
    random_orientation=False,
    checkpoints="every_layer",
    probes=[
        mipt.code_dimension(),
        mipt.contiguous_distance(checkpoints=cps),
        mipt.erasure_first_failure(300, checkpoints=cps),
        mipt.uncorrectable_region_counts(2, checkpoints=cps),
        mipt.tableau_snapshot(checkpoints=cps),
    ],
)
result = spec.run(trajectories=6, seed=5)
K = result.values["code_dimension"]
for r in range(6):
    assert ob.first_passage(K[r]).tau[0] == result.purification_layer[r]
r = 4
k_at = K[r, cps].astype(int)
assert k_at.tolist() == [14, 11, 9, 5, 2, 1, 1, 1]
passage = ob.first_passage(K[r], checkpoints=cps)
assert passage.tau[1] == 10 and passage.tau_chk[1] == 16 and not passage.observed[0]
dcyc = ob.cyclic_distance(result.values["contiguous_distance"][r], k_at, num_system=n)
assert dcyc.zero_extended.tolist() == [1, 1, 1, 1, 1, 3, 1, 2]
first = result.values["erasure_first_failure"][r]
counts = result.values["uncorrectable_region_counts[2]"][r]
D = ob.integrated_robustness(first, n, exact_counts=counts, k=k_at)
by_rule = ob.peak(cps, dcyc.zero_extended, rule="exact", k_at=k_at)
assert by_rule.maximizing_set.tolist() == [16] and by_rule.t_star == 16.0
assert by_rule.k_on_set.tolist() == [1] and not by_rule.at_horizon and by_rule.t_star_is_checkpoint
assert [round(float(x), 2) for x in (D.value[5], D.se[5], D.value[7], D.se[7])] == [6.33, 0.10, 5.70, 0.11]
plateau = ob.peak(cps, D.value, D.se, rule="plateau", z=1.0, k_at=k_at, se_exact=True)
assert plateau.maximizing_set.tolist() == [16] and plateau.first_argmax == 16 and plateau.se_exact
assert [p.maximizing_set.tolist() for p in ob.peak_sensitivity(cps, D.value, D.se, [0, 1, 2])] == [[16]] * 3
key = TrajectoryKey("ws-demo", n, "0.1", "0.1", 0, 4)
boot_seed = ob.bootstrap_seed(key, "peak_D", "1")
boot = ob.bootstrap_peak(cps, first, n, z=1.0, replicates=200, seed=boot_seed, exact_counts=counts)
assert boot.interval(0.9) == (16.0, 16.0) and boot.missing_fraction == 0.0 and boot.seed == boot_seed
phi = ob.uniform_channel_failure(first, n, [0.05, 0.1], exact_counts=counts, k=k_at)
assert np.all(phi.value[5] < 0.02)  # at the peak depth: Phi(0.05) = 0.0033, Phi(0.1) = 0.017
```

### (f) A correlated profile: complete against interpolated

The same evaluations probed twice on the ring geometry (exact samples, 400 regions), once at every
size $1, \dots, 9$ and once at the odd sizes only. The complete profile's $D(J)$ agrees with the exact
value (from the code's boundary-resolved counts) within its error; the odd-size profile flags
sizes 2, 4, 6, 8 as interpolated, is marked incomplete, and its $D(J)$ is biased (for example 1.78
against the exact 1.59 at $J = 0$ for trajectory 1 at depth 10, where $P_0(1), P_0(2), P_0(3)$ are
$0.5, 0.911, 1$: the chord gives $P_0(2) \approx 0.75$, too low, so $D$ comes out too high). At
$\varepsilon = 0.1$ and $0.3$ the interpolated sizes carry 0.205 and 0.472 of the binomial weight of
$\Phi_J$.

```python
n = 10
cps_f = [10, 20]
spec_f = mipt.MonitoredCircuit(
    num_system=n,
    edges=graphs.watts_strogatz(n, 4, 0.1, seed=(17, 0)),
    layers=20,
    measurement_rate=0.05,
    reference="purification",
    schedule="random_edges",
    gates_per_layer=n // 2,
    random_orientation=False,
    checkpoints="every_layer",
    probes=[
        mipt.code_dimension(),
        mipt.correlated_erasure([0.0, 2.0], None, 400, geometry="cycle", name="every", checkpoints=cps_f),
        mipt.correlated_erasure([0.0, 2.0], "1:10:2", 400, geometry="cycle", name="odd", checkpoints=cps_f),
        mipt.tableau_snapshot(checkpoints=cps_f),
    ],
)
res_f = spec_f.run(trajectories=3, seed=11)
k_f = res_f.values["code_dimension"][:, cps_f]
profiles = {}
for name in ("every", "odd"):
    js, sizes, values = res_f.correlated(name)
    sampler = spec_f.correlated_sampler(name)
    profiles[name] = ob.correlated_failure(values, js, sizes, n, k_f, samples=400, sampler=sampler)
assert profiles["every"].complete and profiles["every"].sampler["method"] == "exact"
assert profiles["every"].se_exact  # exact samples: the binomial errors are the true ones
assert not profiles["odd"].complete and profiles["odd"].interpolated_sizes.tolist() == [2, 4, 6, 8]
for rr in range(3):
    for e in range(2):
        exact = res_f.code(rr, e).boundary_failure_counts(graphs.cycle(n), n).exact([0.0, 2.0])
        d_exact = 1 + (1 - exact[:, : n - 1]).sum(axis=1)
        every = profiles["every"]
        assert np.all(np.abs(every.robustness[rr, e] - d_exact) <= 4.5 * every.robustness_se[rr, e])
assert round(float(profiles["odd"].robustness[1, 0, 0]), 2) == 1.78  # exact D(0) = 1.589
mass = ob.correlated_channel_failure(profiles["odd"], [0.1, 0.3]).interpolated_mass
assert np.allclose(mass, [0.205, 0.4718], atol=5e-5)
```

### (g) Observables computed later from a stored snapshot

The joint tableau of trajectory 4 of (e) at depth 56 ($k = 1$), with its graph and its key:
`compute` gives $k$, the half-cut entropy, $d_{\mathrm{conn}}$ and $d_{\mathrm{cyc}}$, and the mean
leakage by both routes. At $m = 7 = n/2$ the exact route gives $\bar\ell = k = 1$
(Proposition 15(d)); the sampled route (the chain on the Watts–Strogatz graph) gives
$1.095 \pm 0.041$. The value carries its key (the entry's id and version and the canonical
parameters, §3.10), and the seed is the tuple of Proposition 18, rebuilt here by hand.

```python
n, t = 14, 56
key = TrajectoryKey("ws-demo", n, "0.1", "0.1", 0, 4)
snap = ob.SnapshotInput(num_system=n, t=t, key=key, tableau=result.tableau(4, 7), edges=spec.graphs[0])
assert ob.compute("k", snap).value == 1.0
assert ob.compute("half_cut_entropy", snap).value == 2.0
assert ob.compute("d_conn", snap).value == 2.0 and ob.compute("d_cyc", snap).value == 2.0
sampled = ob.compute("mean_leakage", snap, {"m": 7, "J": 1.0, "samples": 200})
exact = ob.compute("mean_leakage", snap, {"m": 7, "J": 1.0, "route": "enumerate"})
assert abs(exact.value - 1.0) < 1e-12 and sampled.method == "chain"
assert abs(sampled.value - exact.value) <= 4 * sampled.se
# the key the value is stored under: the entry's id and version, and canonical parameters (J = 1
# and J = 1.0 are one key)
assert (sampled.observable_id, sampled.version) == ("mean_leakage", "1")
assert sampled.params_key == ob.params_key("mean_leakage", {"m": 7, "J": 1, "samples": 200})


def w(text):
    value = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")
    return [value % 2**32, value // 2**32]


bits = struct.unpack("<Q", struct.pack("<d", 1.0))[0]
by_hand = (0x5753524F, *w("ws-demo"), n, *w("0.1"), *w("0.1"), 0, 4, 0, t, 7)
by_hand += (bits % 2**32, bits // 2**32, *w("mean_leakage"), *w("1"))
assert tuple(sampled.data["seed"]) == by_hand
```

### (h) The catalog

```python
rows = ob.catalog()
assert len(rows) == 19
assert {r["id"] for r in rows if r["retrospective"]} == {
    "k", "code_rate", "stabilizer_rank", "d_cyc", "mean_leakage", "d_conn", "half_cut_entropy"
}
print(ob.catalog_markdown().splitlines()[0])
print(ob.params_key("mean_leakage", {"m": 7, "J": 1.0}))
```

| id | level | inputs | definition (v1.4) |
|---|---|---|---|
| `k`, `code_rate`, `stabilizer_rank` | snapshot (also retrospective) | `k_trace`, `physical_code` | §5.2: $k_t$, $k_t/n$, $r_t$ |
| `k_trace` | trajectory | `k_trace` | §5.2, §5.3: change points |
| `first_passage` | trajectory | `k_trace` | §5.2: $\tau_\ell$, censoring, $\widehat\tau^{\mathrm{chk}}_\ell$ |
| `d_cyc` | snapshot (also retrospective) | `probe:contiguous_distance`, `probe:code_dimension`, `physical_code` | §5.5 |
| `uniform_failure`, `D`, `Phi` | snapshot | `probe:erasure_first_failure`, `probe:uncorrectable_region_counts` | §5.7 at $J = 0$ |
| `correlated_failure`, `D_J`, `Phi_J` | snapshot | `probe:correlated_erasure`, `probe:code_dimension` | §5.7 |
| `mean_leakage` | snapshot (retrospective) | `physical_code`, `graph` | §5.7: $\bar\ell(m, J)$ |
| `d_conn` | snapshot (retrospective) | `physical_code`, `graph` | §5.6 |
| `half_cut_entropy` | snapshot (retrospective) | `joint_tableau` | $S$ of $\lfloor n/2\rfloor$ qubits |
| `peak_d_cyc`, `peak_d_conn` | trajectory | `trace:...`, `k_trace` | §5.8, exact rule |
| `peak_D` | trajectory | `trace:D`, `k_trace`, the uniform probes | §5.8, plateau rule, bootstrap of $T^*$ |
| `peak_D_J` | trajectory | `trace:D_J`, `k_trace` | §5.8, plateau rule (no bootstrap; chain errors are lower bounds, `se_exact`) |

## 8. Validation map

`tests/_study_oracle.py` imports nothing from `sparsegf2`: it enumerates the whole stabilizer group of
a joint state ($S(X) = \lvert X\rvert - \log_2\#\lbrace g : \operatorname{supp} g \subseteq X\rbrace$), every
Pauli commuting with a code ($\dim N_A - \dim W_A$), every subset, every permutation, every cyclic
interval, every connected subset (breadth-first search), and $Q_{G,J}$ by enumeration; random joint
states come from Stim. All tests are in `tests/test_study_observables.py`.

| property | test | what is compared |
|---|---|---|
| registry: one entry per observable, sorted rows, plain values, retrospective set | `test_the_catalog_has_one_entry_per_observable` | ids, levels, inputs, v1.4 sections, the Markdown table (six cells in every row, pipes escaped, the definitions recovered) |
| versions: a changed entry needs a new version | `test_register_get_and_versions` | refusal, a second version, `get` by version, refusals of bad entries |
| canonical parameter keys | `test_parameters_are_resolved_and_keyed_canonically`, `test_numerically_equal_parameters_have_one_key`, `test_entries_declare_the_kind_of_every_parameter` | defaults, unknown and missing names, key equality, `json.dumps` reference; one key for $J$ = 1, 1.0, numpy scalars, $\pm 0$, int and float coupling lists, a scalar or one-element $\varepsilon$, the design manifest's `analysis.peak`; strings, bools, fractional ints and NaN refused; kinds required and defaults canonical |
| parameters a function consumes | `test_peak_parameters_are_only_those_a_function_consumes` | `peak_D` keeps `bootstrap`, `peak_D_J` has none |
| `compute` refusals | `test_compute_refuses_what_it_cannot_compute` | missing inputs, no key, wrong $n$, bad route |
| `compute` passes the entry and keys the value | `test_compute_hands_the_entry_to_its_compute_and_keys_the_value` | the entry and canonical parameters reach `compute`; `observable_id`, `version`, `params_key`; a non-`ObservableValue` refused |
| a snapshot checked against $n$ | `test_a_snapshot_is_checked_against_its_n` | a 12-qubit joint tableau refused at $n$ = 4, 5, 7, 8 and accepted at 6, 11, 12; a 6-qubit code refused at $n$ = 5, 8; types |
| change points (§3.2) | `test_change_points_round_trip_and_are_canonical`, `test_change_points_refuse_what_is_not_a_trace` | hand cases; 300 random traces against a loop; every refusal |
| Propositions 2, 3 | `test_first_passage_matches_its_definition` | 300 random traces: $\tau$, $\widehat\tau^{\mathrm{chk}}$ against $\min$ over layers and checkpoints; $\tau \le \widehat\tau^{\mathrm{chk}}$; $k_t = \min\lbrace\ell : \tau_\ell \le t\rbrace$ (exact) |
| hand-made first passages, censoring, zero fill | `test_first_passage_on_hand_made_traces` | multi-drop, plateaus, censoring, `survival_data`, `to_dict`, change-point input |
| Propositions 1, 4 on engine traces | `test_first_passage_on_engine_traces` (5 cases, $n$ = 5 to 16, odd $n$) | stopped and continued runs give equal traces; $\tau_0$ = `purification_layer`; $\tau_n = 0$; bounds |
| $d_{\mathrm{cyc}}$ forms | `test_cyclic_distance_forms_and_refusals` | raw, `has_code`, zero extension; 6 refusals |
| $d_{\mathrm{cyc}}$ three ways; Proposition 10(c) | `test_cyclic_distance_of_engine_codes_three_ways` (6 cases, $n$ = 3 to 16) | engine probe = stored code = oracle intervals ($n \le 5$); $d \le \lfloor(n+1)/2\rfloor$ |
| periodic windows | `test_cyclic_distance_wraps_around_the_ring` | a wrapping failing interval (2 against 3 open), oracle |
| Propositions 6–9 | `test_reference_information_identities_on_random_codes` (20 Stim states, $n$ = 1 to 5), `test_reference_information_identities_on_engine_codes` (5 cases, $n$ = 3 to 7) | every region: code = joint state = oracle entropies = oracle $\dim N_A - \dim W_A$; complement; correctability; monotonicity (exact) |
| Proposition 15(a)–(c) | `test_complementary_failure_and_robustness_bounds_on_engine_codes` | exact $P_J$ of 20+ engine codes = oracle $Q_{G,J}$ enumeration (to $10^{-12}$); $J = 0$ = uniform; $P(m) + P(n-m) \ge 1$; $D(J) \le (n+1)/2$ |
| Proposition 15(d) | `test_leakage_identities_and_exact_values_on_engine_codes` | exact $\bar\ell$ = oracle (to $10^{-9}$); complement sums; $\bar\ell(n/2) = k$ |
| Proposition 12 | `test_uniform_failure_profile` | hand curves, errors, exact mask, refusals |
| Proposition 13(a), 12(b) | `test_D0_is_the_expected_first_failure_size_exactly` | all $n!$ orderings ($n \le 6$), Fractions: $\Pr[M \le m] = P(m)$, $\mathbb E M = D(0)$, $D \le (n+1)/2$ |
| Proposition 13(b) | `test_integrated_robustness_formulas` | 50 random sample sets: value and error against loops; contradicting counts and $k$ refused |
| $\hat D(0)$ unbiased | `test_D0_estimates_match_the_exact_value`, `test_D0_from_the_engine_probes_matches_the_exact_value` | within 4.5 exact standard deviations; false alarm below $2\cdot10^{-4}$ and $10^{-4}$ (normal approximation); engine counts = oracle counts (exact) |
| Proposition 14 | `test_uniform_channel_failure_formulas`, `test_uniform_channel_failure_matches_the_exact_value` | value and error against loops (exact), `se` of the shape of `value` (also with one ordering and several rates); 48 comparisons within 4.5 exact sd, false alarm below $3.3\cdot10^{-4}$ |
| rates copied; memory bounded in $q$ | `test_channel_failure_never_shares_or_freezes_the_callers_rates`, `test_uniform_channel_failure_memory_does_not_grow_with_the_rate_count` | the caller's array (and the base of a view) stays writable and unchanged in the result; peak traced memory at 40 rates $\le 12\times$ the input (measured 5.1; the former dense array, 82) |
| Proposition 16 | `test_correlated_profile_bookkeeping` | hand curves, $D(J)$ and $\Phi_J$ with the interpolation weights written out, refusals; `se_exact` only for an exact sampler |
| exact correlated samples | `test_correlated_probe_matches_the_exact_law` ($n$ = 6, 7, 8) | every $\hat P_J(m)$: exact binomial $p$-value above $10^{-4}$/comparisons (Bonferroni); $D(J)$, $\Phi_J$ within 4.5 sd; below $9.6\cdot10^{-4}$ in all; `se_exact` |
| the chain, across trajectories | `test_correlated_probe_on_interaction_graphs_is_unbiased_across_trajectories` | mean of $\hat D(J) - D(J)$ over $\ge 40$ trajectories, Student $\lvert t\rvert \le 4$ (false alarm $5.6\cdot10^{-4}$); not `se_exact` |
| Proposition 18(a)–(c) | `test_region_seed_derivation` | the tuple rebuilt in the test (blake2b, struct); 192 distinct seeds over every field; tags; refusals |
| Proposition 18(d), Proposition 20's seed | `test_bootstrap_seed_derivation_and_the_bootstrap_refuses_fresh_entropy` | the tuple rebuilt in the test; 32 distinct seeds over every field; the tag apart from WSRO, WSG1, WSC1; a recorded seed reproduces the bootstrap; `None`, floats, strings, bools, negatives, empty and `SeedSequence` refused |
| versions keep their own seeds | `test_a_registered_version_keeps_its_own_region_seeds` | with a version 2 and a renamed copy registered, version 1's seed and value are unchanged; the three seeds are distinct and each the tuple of its own id and version |
| Proposition 17 | `test_mean_leakage_is_keyed_and_reproducible`, `test_mean_leakage_matches_the_exact_law_with_exact_samples`, `test_mean_leakage_with_the_chain_is_unbiased`, `test_mean_leakage_caps_the_chain_ess_at_the_sample_count` | reproducibility and the regions of the seed; ESS = capped `chain_diagnostics` (a correlated case, and a tempered case whose Sokal ESS is 106.7 > S = 60: ESS = S, error $s/\sqrt S$); exact samples within 4.5 exact sd (12 comparisons, $10^{-4}$); the chain, $\lvert t\rvert \le 4$ over 48 seeds ($5\cdot10^{-4}$) |
| Proposition 11 | `test_connected_distance_against_the_oracle`, `test_connected_and_cyclic_distances_can_differ`, `test_connected_distance_is_at_most_cyclic_on_the_unrewired_graph` | 30+ random codes and graphs (cycles, paths, WS, disconnected) = oracle (with $\infty$); the two examples; $d_{\mathrm{conn}} \le d_{\mathrm{cyc}}$ on $C(n,2)$; refusal above `max_n` |
| half-cut entropy | `test_half_cut_entropy_matches_the_engine_probe_and_the_oracle` | engine probe = stored tableau = oracle ($n \le 7$) |
| retrospective `compute` | `test_compute_from_stored_snapshots_matches_the_engine_probes` | $k$, rate, rank, $d_{\mathrm{cyc}}$, $d_{\mathrm{conn}}$ against the engine probes |
| peaks: exact rule, flags | `test_exact_peaks_ties_log_midpoint_and_horizon`, `test_peaks_missing_and_invalid` | ties, log midpoint, horizon, $k$ on $\mathcal A$, `t_star_is_checkpoint`; every invalid case and refusal |
| Proposition 19(b) | `test_plateau_rule_hand_cases_and_nesting` | hand thresholds; nesting in $z$ and $z = 0$ = exact on 200 random traces |
| the plateau rule as defined, unequal errors | `test_plateau_rule_uses_the_first_maximizers_error`, `test_plateau_rule_matches_its_definition_on_random_traces` | hand cases whose sets depend on $\mathrm{se}(t^\circ)$ (a checkpoint at 0; a first maximizer that is not the smallest depth; a last error unlike $\mathrm{se}(t^\circ)$); 400 random traces (with $t = 0$, ties, several $z$) against the rule written out in the test |
| the plateau's error rate | `test_plateau_rule_keeps_a_true_plateau_with_the_stated_probability` | 4000 trials, a 3-point plateau, $z = 2$: exclusion rate $\le 0.1365 + 4$ sd (false alarm $4\cdot10^{-5}$) |
| the plateau's record of its errors | `test_peaks_record_whether_their_errors_are_true` | `se_exact` as given (None for the exact rule), in `to_dict`, in missing peaks and the sensitivity sweep; True for the $D(0)$ bootstrap |
| Proposition 20 | `test_bootstrap_peak_reproduces_its_documented_draws`, `test_bootstrap_peak_keeps_the_exact_counts_in_every_replicate`, `test_bootstrap_peak_of_noiseless_samples_is_the_estimate` | the draws rebuilt from the documented stream, peaked by the rule written out; noisy samples with counts up to $w = 2$, each replicate's hybrid $\hat D(0)$ and $s_Y/\sqrt S$ rebuilt (the plain estimator changes $T^*$ in 34 of the 60 replicates, and the test requires at least 5, so it tells them apart); noiseless samples; $k = 0$ |
| the module's examples | `src/sparsegf2/study/observables.py` doctests | `--doctest-modules` |

A mutation check (a script outside the repository) planted 35 single-line faults in
`observables.py`, one at a time; the tests killed 34. The survivor sums $D(0)$ to $m = n$ instead
of $n - 1$, which is equivalent: validated inputs force $\tilde P(n) = 1$ when $k > 0$. After the
review, 18 more faults, one per fix (the latest version's seed in a computed value, unnormalized
parameters, string couplings, a bootstrap listed for `peak_D_J`, an unchecked bootstrap seed, a
one-column error with one ordering, shared rates, the dense $(\dots, S, q)$ array, unchecked
snapshot widths (two), no Markdown escaping, four wrong plateau widths ($\mathrm{se}$ indexed
before restricting to $t > 0$, $\sqrt2\,\mathrm{se}(t)$, the last checkpoint's error,
$\mathrm{se}(t)$ twice), replicates without the counts, no ESS cap, no `se_exact` record), were all
killed.

## 9. Limitations and departures

* **No departure from v1.4's laws or definitions.** Nothing here draws circuit randomness or
  changes a probe; the definitions are v1.4's, and §4 proves the properties the code uses. Nothing
  in v1.4 §5.2–§5.8 was found to be mathematically wrong. One remark: v1.4 §5.6 says
  $\lbrace 0, 2, 4\rbrace$ is connected but not contiguous "for $n \ge 7$"; it is so for every $n \ge 5$.
* **Chain errors.** The engine's correlated probe stores failing fractions only, so its binomial
  errors understate the error of chain estimates by about $\sqrt{\tau_{\mathrm{int}}}$, and nothing
  in the run can reveal incomplete mixing; the sampler's configuration is carried
  (`CorrelatedProfile.se_exact` is False for it), and mixing is checked on the graph side with
  `erasures.sample_regions` (correlated-erasure.md §8.3). Measured (a script outside the
  repository: one engine code on a WS($n$, 4, 0.1) graph, the probe's defaults, 100 regions per
  size, every size, 40 independent chain seeds), the spread of $\hat D(J)$ over the seeds divided by
  the mean `robustness_se` was 1.02, 1.28, 2.29 at $J = 0, 1, 2$ for $n = 32$ and 0.88, 1.44, 3.64
  for $n = 64$ with `thin` = 1 (the default), and 1.22, 0.94, 0.81 and 0.99, 0.94, 0.98 with
  `thin` = 20 (each ratio itself uncertain by about 11% with 40 seeds). Across trajectories the
  chains are independent, so the graph-cluster error of an ensemble mean of per-trajectory
  estimates (`stats.graph_cluster_se`, v1.4 §6.4) contains the chain variance and is valid; the
  plain spread over trajectories is valid only with one circuit per graph. Neither contains a
  shared equilibration bias.
* **Interpolated sizes.** $D(J)$ and $\Phi_J$ computed with interpolated sizes are biased by the
  interpolation (§7(f)); they are flagged (`complete`, `interpolated_mass`), not corrected.
* **The plateau rule** is a fixed tie rule, not a calibrated test: its exclusion rate for true
  maximizers depends on the plateau size (§4.6's bound), and it is conditional on the
  trajectory's codes. Its stated meaning needs the true standard errors. For `peak_D_J` on the
  study's interaction graphs (the chain at every $J$) the errors are the binomial lower bounds, so
  with the default `thin` = 1 the plateau at the manifest's $z = 1$ spans about $1/2.3$ to
  $1/3.6$ of a true standard deviation at $J = 2$ (the factors above), true plateau checkpoints are
  left out more often than $q(q-1)\Phi_{\mathcal N}(-z)$, $T^*_J$ moves toward single, noisy maxima,
  `sensitivity_z` understates the rule's fragility, and the effective width differs between
  conditions, which matters when $T^*_J$ enters a finite-size-scaling comparison. The record says
  so (`Peak.se_exact` False); the remedy is a `thin` near the chain's $\tau_{\mathrm{int}}$
  (measured with `erasures.sample_regions` at the probe's settings), as `thin` = 20 achieved above.
* **No bootstrap of $T^*_J$** (a partial departure from v1.4 §5.8, which asks for a
  bootstrap-supported plateau rule for Monte Carlo estimates of $D$). The correlated probe keeps
  failing fractions, not regions, so there is nothing to resample nonparametrically, and a
  parametric binomial bootstrap would inherit the understated chain errors and give false
  confidence on the interaction graphs. `peak_D_J` therefore lists no `bootstrap` (its key would
  otherwise claim replicates that never ran); the uncertainty of $T^*_J$ comes from the
  sensitivity over $z$ and, for ensemble statistics, from the graph-cluster bootstrap
  (`stats.graph_bootstrap`). $T^*$ of $D(0)$ keeps its bootstrap (Proposition 20).
* **$d_{\mathrm{conn}}$** is exponential in the worst case: refused above `max_n` (default 20).
* **`mean_leakage_exact`** evaluates $\binom nm$ regions in Python (refused above two million);
  it is for validation and small systems.
* **Conventions chosen here** (v1.4 left them open): the change-point form with the horizon row;
  the $-1$ marker of censored thresholds; the first maximizer $t^\circ$ in the plateau; the exact
  singleton $T^*$; `t_star_is_checkpoint` to a relative $10^{-12}$; the half-cut region; $\infty$ for
  no connected failing region; the seed tuples of Proposition 18; the canonical parameter forms of
  §3.10.
* **Additions beyond the design table (§5)**, none of which changes a definition: `Phi` and `Phi_J`
  (v1.4 §5.7 and §9, "Geometry-sensitive follow-up"), the exact route of `mean_leakage`
  (`route="enumerate"`), the optional `k` cross-check of the uniform estimators,
  `t_star_is_checkpoint` (validation requirement 13), `se_exact`, `bootstrap_seed`, the key carried
  by `ObservableValue`, and `catalog_markdown`.
* **Seeds (ruling R-SEED).** The retrospective seeds use the full trajectory key, not the stream
  index $u$ the T4 task brief first named (the design table says only "keyed region sampler"). The
  token words come from `identity.token_words`; the other strings use a private helper with the
  same convention (Proposition 18).
