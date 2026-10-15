# Random-region erasure

How SparseGF2 computes $P_C(m)$, $\Phi_C(e)$, $e_\delta$, $\mu_C$ and the
distance bounds of the codes that monitored circuits generate.

This document follows the erasure protection of a dynamically generated code
from its definition, through the formulas and kernels that compute it, to the
functions you call and a worked example on a single trajectory. It is written
for a reader fluent in the stabilizer formalism and GF(2) linear algebra, and
every statement about the code cites the function it describes as
`path::function` (paths relative to `src/sparsegf2/`). It describes version
3.0.0.dev12.

Companion documents: [lost-information.md](lost-information.md) (the lost
information $\ell_C(A)$ of one region, complementarity, the profile probe) and
[code-dimension.md](code-dimension.md) ($k_C = S(R)$ and the reference-row
invariant). [HOW_IT_WORKS.md §9](../HOW_IT_WORKS.md#9-erasure-analysis) is a
shorter overview of the same machinery.

## Contents

1. [Definition](#1-definition)
2. [What the simulator evaluates, and why it equals the definition](#2-what-the-simulator-evaluates-and-why-it-equals-the-definition)
3. [From the kernel to the code you run](#3-from-the-kernel-to-the-code-you-run)
4. [Cost and accuracy](#4-cost-and-accuracy)
5. [Worked example: one Watts–Strogatz trajectory at n = 16](#5-worked-example-one-wattsstrogatz-trajectory-at-n--16)
6. [Pitfalls](#6-pitfalls)

The Python blocks run as written (they are executed by `tests/test_docs.py`)
and take a few seconds together once the kernels are compiled.

---

## 1. Definition

### 1.1 The setting

A monitored circuit acts on $n$ system qubits $[n]$ (labelled $0, \dots, n-1$
in the code). In the **purification picture** every system qubit $i$ starts
Bell-paired with a reference qubit $n + i$; gates and measurements act on the
system only, so the joint state of the system and the reference $R$ stays a
pure stabilizer state. At a checkpoint, the system's reduced state is the
maximally mixed state on the code space of a stabilizer code $C$ with
parameters $[[n, k_C]]$:

* its stabilizer group $S$ is the set of elements of the joint stabilizer group
  that act trivially on $R$;
* $S$ has $s = n - k_C$ independent generators, the rows of a check matrix
  $H \in \mathbb{F}_2^{s \times 2n}$ (the X bits of the $n$ qubits, then the Z
  bits; signs play no role in anything below);
* $R$ purifies the $k_C$ logical qubits: $S(R) = k_C$
  ([code-dimension.md](code-dimension.md)).

In the **single** picture $R$ is one qubit, Bell-paired with one system qubit,
and $k_C \in \lbrace 0, 1\rbrace$. The erasure is applied to the code at the
checkpoint, in thought: evaluating it never changes the trajectory.

For a region $A \subseteq [n]$, write $\bar A = [n] \setminus A$ and $H_A$ for
the $s \times 2|A|$ submatrix of the X and Z columns of the qubits in $A$. The
**lost information** of $A$ is

$$
\ell_C(A) \equiv I(A:R),
$$

and the erasure of $A$ is perfectly correctable if and only if
$\ell_C(A) = 0$ ([lost-information.md](lost-information.md) §1). The simulator
evaluates it through **formula (1)**:

$$
\ell_C(A) = 2|A| - s - \operatorname{rank}(H_A) + \operatorname{rank}(H_{\bar A}),
$$

with ranks over $\mathbb{F}_2$ (derived in §2.1).

### 1.2 $P_C(m)$: a probability over regions, not a channel

Fix the code and a size $m$, and draw $A$ uniformly from the $\binom{n}{m}$
subsets of $[n]$ of size $m$ (the $m$ qubits are distinct: drawn without
replacement). **Formula (2)**:

$$
P_C(m) = \Pr_{A \sim \mathrm{Unif}\binom{[n]}{m}}\big[\ell_C(A) > 0\big] = \frac{1}{\binom{n}{m}} \sum_{A \subseteq [n],\ |A| = m} \mathbf{1}\big[\ell_C(A) > 0\big] = \frac{N_m}{\binom{n}{m}},
$$

where $N_m$ is the number of uncorrectable $m$-subsets. $P_C(m)$ is a
probability over the choice of region for one fixed code. No qubit is erased
independently of the others; the independent per-qubit channel enters only
through $\Phi_C(e)$ below.

Properties used throughout (proofs in §2.3):

* $P_C(0) = 0$, and $P_C(m)$ is non-decreasing in $m$.
* With $d$ the code distance (the smallest size of an uncorrectable region),
  $P_C(m) = 0$ for $m \lt d$ and $P_C(d) = N_d / \binom{n}{d} > 0$.
* $P_C(m) = 1$ for $m > n - d$, and in particular $P_C(n) = 1$ when $k_C > 0$.
* When $k_C = 0$ there is nothing to lose: $P_C(m) = 0$ for every $m$, and
  there is no distance. Ensemble statistics must say whether they include such
  trajectories (§6).

### 1.3 Independent erasure: $\Phi_C(e)$, $e_\delta$ and $\mu_C$

Under independent erasure, each qubit is erased with probability $e$, so the
erased set $E$ has $\Pr[E = A] = e^{|A|}(1-e)^{n-|A|}$. This weight depends on
$A$ only through $|A|$, so conditioned on $|E| = m$ every $m$-subset is equally
likely, and the failure probability is the binomial mixture of $P_C$:

$$
\Phi_C(e) = \Pr_e\big[\ell_C(E) > 0\big] = \sum_{m=0}^{n} \binom{n}{m} e^m (1-e)^{n-m} P_C(m) = \sum_{m=0}^{n} N_m e^m (1-e)^{n-m}.
$$

The binomial weights $b_m(e) = \binom{n}{m} e^m (1-e)^{n-m}$ concentrate near
$m \approx ne$, with width $\sqrt{ne(1-e)}$. As $e \to 0$,
$\Phi_C(e) = N_d e^d + O(e^{d+1})$: the small-$e$ behaviour is fixed by the
exact number of uncorrectable regions of the smallest failing size (§4.4).

For an ensemble of codes (trajectories), $\Phi(e)$ is the mean of $\Phi_C(e)$
over the codes you select (all trajectories, or those with $k_C > 0$), and the
**erasure tolerance** at level $\delta$ is

$$
e_\delta = \sup\lbrace e \in [0, 1] : \Phi(e) \le \delta \rbrace.
$$

The **first-failure size** $D$ (§2.4) is the number of qubits erased, in
uniformly random order, until the first failure. For $k_C > 0$ its mean is

$$
\mu_C = \mathbb{E}[D] = \sum_{m=0}^{n} \big(1 - P_C(m)\big).
$$

### 1.4 Two regimes and the cutoff $m^{\ast}(n)$

$P_C(m)$ is computed in two regimes, separated by a cutoff
$m^{\ast}(n) \in \lbrace 0, \dots, n\rbrace$:

* $0 \le m \le m^{\ast}(n)$: **exact enumeration** of every $m$-subset;
* $m^{\ast}(n) \lt m \le n$: **$M$ uniformly random regions** per size, from
  $M$ random orderings (§2.4).

$m = 0$ is trivially exact ($P_C(0) = 0$). The exhaustive regime tests
**formula (3)**

$$
N_{\mathrm{exact}}(n) = \sum_{m=1}^{m^{\ast}(n)} \binom{n}{m}
$$

regions per code. For $n = 256$:

| $m$ | $\binom{256}{m}$ | cumulative $N_{\mathrm{exact}}$ with $m^{\ast} = m$ |
|---|---|---|
| 1 | 256 | 256 |
| 2 | 32,640 | 32,896 |
| 3 | 2,763,520 | 2,796,416 |
| 4 | 174,792,640 | 177,589,056 |

so $m^{\ast}(256) = 2$ costs 32,896 tests per code. The cutoff can also be set
by an enumeration budget $B$, **formula (4)**:

$$
m^{\ast}(n) = \max\Big\lbrace w : \sum_{m=1}^{w} \binom{n}{m} \le B \Big\rbrace .
$$

For example $B = 50{,}000$ gives $m^{\ast}(64) = 3$, $m^{\ast}(128) = 2$,
$m^{\ast}(256) = 2$ and $m^{\ast}(512) = 1$. In the code, $m^{\ast}(n)$ is the
`max_size` of the probe `mipt.uncorrectable_region_counts` and of
`StabilizerCode.uncorrectable_counts`; it is written $w$ below. A `budget=`
argument on the probe is planned; until then, compute $w$ from the budget
yourself:

```python
import math


def exhaustive_regions(n, w):
    """N_exact(n) of formula (3) for the cutoff w = m*(n)."""
    return sum(math.comb(n, m) for m in range(1, w + 1))


def cutoff_from_budget(n, budget):
    """m*(n) of formula (4): the largest w whose N_exact(n) fits the budget (0 if none)."""
    w = 0
    while w < n and exhaustive_regions(n, w + 1) <= budget:
        w += 1
    return w


assert exhaustive_regions(256, 2) == 32_896
assert [cutoff_from_budget(n, 50_000) for n in (64, 128, 256, 512)] == [3, 2, 2, 1]
```

and pass it on, for example `mipt.uncorrectable_region_counts(max_size=cutoff_from_budget(n, 50_000))`.

---

## 2. What the simulator evaluates, and why it equals the definition

Notation. Paulis on $[n]$ modulo phases are vectors of $\mathbb{F}_2^{2n}$ with
the symplectic form $\langle u, v\rangle$; $S$ is a subspace of dimension $s$,
and its normalizer $N(S) = S^{\perp}$ (the Paulis commuting with all of $S$)
has dimension $2n - s = n + k_C$ and contains $S$. For a subspace $V$ and a
region $A$ there are two different restrictions:

* $V_A = \lbrace v \in V : \operatorname{supp} v \subseteq A\rbrace$, the
  elements of $V$ **supported on** $A$;
* $V|_A$, the **projection** of $V$ onto the coordinates of $A$. Its dimension
  is the rank of the $A$-columns of any generator matrix of $V$, so
  $\dim S|_A = \operatorname{rank}(H_A)$.

The projection onto $\bar A$ has kernel $V_A$, so
$\dim V_A = \dim V - \dim V|_{\bar A}$. In particular
$\dim S_A = s - \operatorname{rank}(H_{\bar A})$.

### 2.1 Formula (1) from the entropies

Let $G$ be the stabilizer group of the pure joint state on $[n] \cup R$
($\dim G = n + |R|$). For any set $X$ of qubits, $S(X) = |X| - \dim G_X$
(Fattal et al., quant-ph/0406168). For a region $X \subseteq [n]$ the elements
of $G$ supported on $X$ act trivially on $R$, so $G_X = S_X$ and

$$
S(X) = |X| - \dim S_X = |X| - s + \operatorname{rank}(H_{\bar X}).
$$

Purity gives $S(AR) = S(\bar A)$ and $S(R) = S([n]) = n - s = k_C$. Hence

$$
I(A:R) = S(A) + S(R) - S(AR) = \big(|A| - s + \operatorname{rank}(H_{\bar A})\big) + (n - s) - \big(n - |A| - s + \operatorname{rank}(H_A)\big),
$$

which is formula (1): $\ell_C(A) = 2|A| - s - \operatorname{rank}(H_A) + \operatorname{rank}(H_{\bar A})$.

### 2.2 The identity the kernels evaluate

**Lemma (duality on a region).** For any subspace $V \subseteq \mathbb{F}_2^{2n}$
and region $A$, the elements of $V^{\perp}$ supported on $A$ are exactly the
symplectic complement of $V|_A$ inside $\mathbb{F}_2^{2|A|}$. So
$\dim (V^{\perp})_A = 2|A| - \dim V|_A$.

*Proof.* If $u$ is supported on $A$, then $\langle u, v\rangle = \langle u|_A, v|_A\rangle$
for every $v$, so $u \perp V$ exactly when $u|_A \perp V|_A$. ∎

With $V = S$ the lemma gives $\dim N(S)_A = 2|A| - \operatorname{rank}(H_A)$,
so formula (1), read as
$\big(2|A| - \operatorname{rank}(H_A)\big) - \big(s - \operatorname{rank}(H_{\bar A})\big)$,
is (i) below. With $V = N(S)$ (whose complement is $S$) it gives
$\dim N(S)|_A = 2|A| - \dim S_A = 2|A| - s + \operatorname{rank}(H_{\bar A})$;
together with $\dim S|_A = \operatorname{rank}(H_A)$ this turns formula (1)
into (ii):

$$
\ell_C(A) = \dim N(S)_A - \dim S_A \qquad \text{(i)}
$$

$$
\ell_C(A) = \dim N(S)|_A - \dim S|_A = \operatorname{rank}\big(N(S)|_A\big) - \operatorname{rank}\big(S|_A\big). \qquad \text{(ii)}
$$

Form (i) is the stabilizer reading of correctability: $\ell_C(A)$ is the
number of independent logical Pauli operators, modulo stabilizers, that can be
supported on $A$, and the erasure is correctable exactly when no nontrivial
logical operator fits in $A$. Form (ii) is **what the kernels compute**, in
the code's notation `l(A) = rank(N(S)|_A) - rank(S|_A)`
(`_kernel/entropy.py::uncorrectable_counts_rows`, `codes.py` module
docstring). Both ranks are ranks of the $2|A|$ columns of $A$, so the cost
depends on $|A|$ and never on $\bar A$.

**Complementarity (the cleaning lemma).** (i) and (ii) together give
$\ell_C(A) + \ell_C(\bar A) = 2k_C$: projecting onto $\bar A$ has kernel
$N(S)_A$ on $N(S)$ and $S_A$ on $S$, so by (ii),
$\ell_C(\bar A) = (n + k_C - \dim N(S)_A) - (s - \dim S_A) = 2k_C - \ell_C(A)$
by (i). The case $\ell_C(A) = 0$ is the cleaning lemma of Bravyi and Terhal:
if no logical operator fits in $A$, every logical operator has a
representative on $\bar A$. (For the pure state on $A\bar A R$ this is also
$I(A:R) + I(\bar A:R) = 2S(R)$.)

**How the kernels' columns relate to $H$.** `_kernel/entropy.py::code_columns`
row-reduces the generators restricted to the system in two passes (§3.2). The
first $s$ echelon rows span $S$: they form $H' = TH$ for some invertible
$s \times s$ matrix $T$. All $r = n + k_C$ echelon rows span $N(S)$: they form
a generator matrix $G_N$ of $N(S)$ whose first $s$ rows are $H'$ and whose
other $2k_C$ rows are logical operators. The function transposes the
echelon, so each qubit $q$ gets an X column and a Z column of $r$ bits:

* the **N(S)-columns** of $q$ are its columns of $G_N$ (all $r$ bits), and
  $\operatorname{rank}\big((G_N)_A\big) = \dim N(S)|_A$;
* the **S-columns** of $q$ are the first $s$ bits of the same columns, i.e. its
  columns of $H'$, and $\operatorname{rank}(H'_A) = \operatorname{rank}(TH_A) = \operatorname{rank}(H_A)$,
  since row operations leave the rank of every set of columns unchanged.

So the kernels evaluate $\ell_C(A) = \operatorname{rank}\big((G_N)_A\big) - \operatorname{rank}(H'_A)$,
which is (ii) and hence formula (1), without ever forming $H_{\bar A}$.

**On the live tableau.** The engine's fixed-region probe evaluates the same
identity on the tableau rows directly,
`I(A:R) = rank(A) - rank_free(A)` (`_kernel/entropy.py::lost_information_rows`).
All rows restricted to the system span $G|_{[n]} = N(S)$ (every element of
$G|_{[n]}$ commutes with $S$, and
$\dim G|_{[n]} = \dim G - \dim G_R = (n + |R|) - (|R| - k_C) = n + k_C$), and
the rows without support on $R$ span $S$ ([code-dimension.md](code-dimension.md) §2).

### 2.3 Monotonicity

**Lemma.** If $A \subseteq B$ then $\ell_C(A) \le \ell_C(B)$.

*Proof.* Data processing: discarding $B \setminus A$ is a channel on $B$, so
$I(A:R) \le I(B:R)$. Algebraically, by (i),
$\ell_C(B) - \ell_C(A) = \dim\big(N(S)_B / N(S)_A\big) - \dim\big(S_B / S_A\big) \ge 0$,
because $S_B \cap N(S)_A = S_A$ makes the map $S_B/S_A \to N(S)_B/N(S)_A$
injective. ∎

Consequences for §1.2. If $B$ is a uniformly random $(m+1)$-subset and $A$ is
$B$ with one uniformly random element removed, then $A$ is a uniformly random
$m$-subset and $\ell_C(A) > 0$ implies $\ell_C(B) > 0$, so
$P_C(m) \le P_C(m+1)$. If $|\bar A| \lt d$ then $\ell_C(\bar A) = 0$ and, by
complementarity, $\ell_C(A) = 2k_C > 0$, so $P_C(m) = 1$ for $m > n - d$.

### 2.4 Sampled regime: first-failure sizes give $P_C(m)$ for every $m$ at once

For an ordering $\pi$ of $[n]$, let $A_m(\pi) = \lbrace \pi_1, \dots, \pi_m\rbrace$
be its prefix of length $m$, and define the **first-failure size**

$$
D(\pi) = \min\lbrace m : \ell_C(A_m(\pi)) > 0 \rbrace, \qquad D = n + 1 \text{ if there is no such } m \text{ (that is, } k_C = 0\text{)}.
$$

**Claim.** If $\pi$ is uniformly random, then $\Pr[D \le m] = P_C(m)$ for every
$m = 0, \dots, n$ simultaneously.

*Proof.*

1. *The prefix is a uniform $m$-subset.* For a fixed $m$-subset $A$, exactly
   $m!(n-m)!$ of the $n!$ orderings have $A_m(\pi) = A$, so
   $\Pr[A_m(\pi) = A] = 1/\binom{n}{m}$.
2. *Nesting and monotonicity make the events coincide.* The prefixes are nested,
   $A_1 \subset A_2 \subset \cdots \subset A_n$. If $D \le m$, then
   $A_D \subseteq A_m$ and $\ell_C(A_m) \ge \ell_C(A_D) > 0$ by §2.3.
   Conversely, if $\ell_C(A_m) > 0$ then $D \le m$ by the definition of $D$ as a
   minimum. So $\lbrace D \le m\rbrace = \lbrace \ell_C(A_m(\pi)) > 0\rbrace$.
3. Taking probabilities: $\Pr[D \le m] = \Pr[\ell_C(A_m(\pi)) > 0] = P_C(m)$
   by step 1 and formula (2). ∎

The probe `erasure_first_failure(samples=M)` records $D_1, \dots, D_M$ for $M$
independent uniform orderings, and the estimator is

$$
\hat P_C(m) = \frac{1}{M} \sum_{j=1}^{M} \mathbf{1}[D_j \le m]
$$

(`analysis.py::failure_by_size`). Its properties follow directly:

* **At each $m$, the $M$ regions are iid uniform $m$-subsets.** Region $j$ is
  the prefix $A_m(\pi_j)$; the orderings are independent, and each prefix is
  uniform by step 1. Each region is drawn without replacement (its qubits are
  distinct).
* **Unbiased, binomial.** $M\hat P_C(m) \sim \mathrm{Bin}(M, P_C(m))$, so
  $\mathbb{E}\hat P_C(m) = P_C(m)$ and the standard error is
  $\sqrt{P_C(m)(1 - P_C(m))/M}$.
* **Positively correlated across $m$.** The regions at different sizes are
  nested prefixes of the same orderings, not independent draws. For
  $m \le m'$, $\mathbf{1}[D \le m] \cdot \mathbf{1}[D \le m'] = \mathbf{1}[D \le m]$, so
  $\operatorname{Cov}\big(\hat P_C(m), \hat P_C(m')\big) = P_C(m)\big(1 - P_C(m')\big)/M \ge 0$.
* **Monotone.** $\hat P_C$ is the empirical distribution function of $D$: it
  is non-decreasing in $m$, $\hat P_C(0) = 0$, and $\hat P_C(n) = 1$ exactly
  when $k_C > 0$ (the integer counts are divided once).
* **Exact at the ends.** If $P_C(m) = 0$ then no $D_j \le m$ and
  $\hat P_C(m) = 0$ exactly; if $P_C(m) = 1$ then $\hat P_C(m) = 1$ exactly.
* **Functions of the whole curve are means over orderings.** Inserting
  $\hat P_C$ into the definitions collapses the sum over $m$:
  $\hat\Phi_C(e) = \frac{1}{M}\sum_j \Pr\big[\mathrm{Bin}(n, e) \ge D_j\big]$
  (`analysis.py::erasure_failure_probability`), and
  $\hat\mu_C = \frac{1}{M}\sum_j D_j$ (`analysis.py::mean_first_failure`). The
  distance satisfies $d \le D_j$ for every $j$, so $\min_j D_j$ is an upper
  bound on $d$ (`analysis.py::distance_bounds`).

The orderings of one evaluation all come from one random stream, drawn one
after another; §3.6 says which stream, and why replaying a trajectory
reproduces them.

### 2.5 Exhaustive regime: every uncorrectable $m$-subset is counted exactly once

`uncorrectable_region_counts(max_size=w)` returns $N_1, \dots, N_w$, so
$N_m/\binom{n}{m}$ is the exact $P_C(m)$ (`analysis.py::exact_failure_by_size`).
The kernel (§3.3) never tests most regions individually, so it is worth
stating why the count is exact.

Write a subset in increasing order, $B = \lbrace b_1 \lt \cdots \lt b_m\rbrace$,
and call $\lbrace b_1, \dots, b_j\rbrace$ its $j$-prefix. Every uncorrectable
$B$ has a unique **shortest failing prefix** $F(B)$ (it exists, since $B$
itself fails), and all shorter prefixes of $B$ are correctable. The walk
visits sorted prefixes depth-first and extends only correctable ones, so it
reaches every sorted set of size at most $w$ whose proper prefixes are all
correctable. Each counting event belongs to one such set $F$ that fails:

* if $|F| \lt w$, the walk adds, for every $i$ with $|F| + i \le w$, the
  $\binom{n - 1 - \max F}{i}$ extensions of $F$ by $i$ larger indices. These are
  exactly the sets of size $|F| + i$ whose $|F|$-prefix is $F$, and for each of
  them $F$ is the shortest failing prefix;
* if $|F| = w$, the last-qubit step (§3.3) counts $F$ itself.

So every uncorrectable $B$ with $|B| \le w$ is counted exactly once, at the
event of $F(B)$. A correctable $B$ is never counted: an extension of a failing
prefix fails (§2.3). The counts are therefore
$N_m = \#\lbrace B : |B| = m,\ \ell_C(B) > 0\rbrace$ exactly, and §3.3 shows
that the integer arithmetic is exact too.

---

## 3. From the kernel to the code you run

### 3.1 Data layout: the tableau and the reference rows

The state of $N$ qubits (system plus reference, $N = n + |R|$) is one
`uint64` array `tab` of shape $(2N + 1, W)$, stored column-major
(`_kernel/tableau.py`, module docstring):

* `tab[2q]` is the X column of qubit $q$: bit $r$ is set when generator $r$
  has X or Y on $q$;
* `tab[2q + 1]` is the Z column of qubit $q$, and the last row holds signs;
* generator $r$ sits at bit `r & 63` of word `r >> 6`, so a column is a bitset
  over the generators. The ensemble engine runs phase-free, with
  $W = \lceil N/64\rceil$ words and no destabilizers.

With a reference, every trajectory also keeps `ref_rows`, the bitset of
stabilizer rows with support on $R$ (`_kernel/entropy.py::reference_rows`).
The engine's measurement keeps the R-parts of these rows linearly independent
(`_kernel/tableau.py::measure_1q_phase_free_tracked`). Three facts follow,
proved in [code-dimension.md](code-dimension.md) §2:

* $k_C = S(R)$ is the number of marked rows minus $|R|$
  (`popcount(ref_rows) - n_ref`);
* the unmarked rows (outside `ref_rows`), restricted to the system, are $s$
  independent generators of $S$;
* all rows restricted to the system span $N(S)$.

A saved code is in the same situation. `codes.py::StabilizerCode` stacks the
generators of $S$ and a basis of $N(S)$ (from
`_kernel/entropy.py::symplectic_complement`) into one column-major tableau
and marks the $N(S)$ rows as if they were reference rows
(`StabilizerCode._init_packed`). The unmarked rows then span $S$ and all rows
span $N(S)$, so the same kernels serve both paths. Redundant generators are
harmless, because `code_columns` discards dependent rows.

### 3.2 Adapted columns (`code_columns`)

`_kernel/entropy.py::code_columns(tab, ws, system, ref_rows)` returns
`(cols, s, r)`:

1. Every tableau row is restricted to the system's $2n$ columns (qubit
   `system[j >> 1]`, X or Z by `j & 1`). The rows outside `ref_rows` are
   inserted first, then the marked rows, into an insertion-order echelon
   (`_small_add`: each stored vector has zeros at the pivots of all earlier
   ones, so one pass in insertion order reduces a new vector). Dependent rows
   are dropped. After the first pass the echelon has $s$ rows (spanning $S$);
   after the second it has $r = n + k_C$ (spanning $N(S)$).
2. The echelon is transposed: row `2i + c` of `cols` is the X (`c = 0`) or Z
   (`c = 1`) column of qubit `system[i]` over the $r$ basis rows, packed in
   $\lceil r/64\rceil$ words. The S-columns are the first $s$ bits of each
   column (a mask on the last word).

Cost: $O(N \cdot r \cdot \lceil 2n/64\rceil)$ word operations for the
elimination, at most about $n^3/8$ in the purification picture ($N = 2n$ rows,
$r \le 2n$). Each erasure kernel calls `code_columns` itself, once per
evaluation.

### 3.3 Exhaustive counts (`uncorrectable_counts_rows`)

`_kernel/entropy.py::uncorrectable_counts_rows(tab, ws, system, ref_rows, max_size, out, out_exact)`
fills `out[m - 1]` (float64) with $N_m$ for $m = 1, \dots, w$, and
`out_exact[m - 1]` (int64) too for every size with $\binom{n}{m} \lt 2^{63}$.
With $r = s$ ($k_C = 0$) it returns zeros at once. Otherwise:

**The walk.** Subsets are increasing index sequences `cand[0..lvl-1]`, walked
depth-first. `lvl` is the length of the prefix being extended and
`cand[lvl]` its next candidate qubit $i$. Each level owns a slice of one stack
of echelon vectors: the S-basis of the prefix of length $j$ is the first
`ftop[j]` vectors of `fb`. Extending a prefix appends at most two vectors, and
backtracking just returns to the shorter `ftop`, so nothing is copied.

**The full-rank shortcut.** Adding qubit $i$ appends its two S-columns. If the
S-rank of the prefix $A$ is then $2|A|$ (full), $A$ is correctable without
looking at $N(S)$: the N(S)-rank is at least the S-rank, because the S-columns
are truncations of the N(S)-columns, and at most $2|A|$.

**The lazy N(S) basis.** Only an S-rank-deficient prefix needs its N(S)-rank
(and the last-qubit step below, when some qubit hits). The N(S)-bases live in
a second stack (`ab`, `atop`), with flags `built[j]`. A deficient prefix
rebuilds the missing levels from the deepest built one, then compares: the
prefix fails exactly when N(S)-rank $>$ S-rank, by (ii).

**Pruning by monotonicity.** When a prefix $A$ whose largest index is $i$
fails, every extension by larger indices fails too (§2.3). The kernel adds
$\binom{n - 1 - i}{j}$ to the count of size $|A| + j$, for every $j$ with
$|A| + j \le w$, and moves to the next candidate without visiting the
subtree.

**The last qubit by hash lookup.** A correctable prefix $P$ of length $w - 1$
is completed without looping over the last qubit $c$. Adding $c$'s columns
$x_c, z_c$ to a span $V$ raises its dimension by $2 - \dim K$, where
$K = \lbrace (a, b) \in \mathbb{F}_2^2 : a x_c + b z_c \in V\rbrace$. So, with
$K_S$ for the S-columns and $K_N$ for the N(S)-columns,

$$
\ell_C(P \cup \lbrace c\rbrace) = \ell_C(P) + \dim K_S - \dim K_N = \dim K_S - \dim K_N .
$$

A relation among full columns survives truncation, so $K_N \subseteq K_S$,
and $P \cup \lbrace c\rbrace$ fails exactly when $K_S \ne K_N$. A subspace of
$\mathbb{F}_2^2$ is fixed by which of its three nonzero elements it contains,
that is, by which of $x_c$, $z_c$ and $y_c = x_c + z_c$ lie in $V$. So:

1. `_column_table` builds an open-addressing hash table of the X, Z and Y
   columns of every qubit (a second table holds the full N(S)-columns).
2. `_span_hits` walks the $2^{\mathrm{rank}_S(P)} \le 4^{w-1}$ elements of the
   S-span of $P$ in Gray-code order (one XOR per step). It looks each up in
   the table and sets bit $j$ of `hits_s[c]` when column $j$ of qubit $c$
   (X, Z, Y) is in the span, for every $c$ after the last prefix qubit. Keys
   are compared in full after a hash match, so collisions cost time, never
   correctness.
3. Most qubits have $K_S = \lbrace 0\rbrace$ (`hits_s[c] == 0`) and cannot
   fail. If any qubit has a hit, the same walk over the N(S)-span, restricted
   to those qubits, fills `hits_n`, and every $c$ with `hits_s[c] != hits_n[c]`
   is counted.

**Exact integer accumulation.** `_binomial_tables` builds $\binom{a}{j}$ for
$a \le n$, $j \le w$ by Pascal's rule in int64, never overflowing (an entry
that would reach $2^{63}$ is marked $-1$). Every size $m$ with
$\binom{n}{m} \lt 2^{63}$ is accumulated exactly in `out_exact` (each addend
counts subsets not counted before, so no partial sum exceeds
$\binom{n}{m}$), and `out[m - 1]` is that integer rounded once to float64.
Only a size with $\binom{n}{m} \ge 2^{63}$ falls back to a compensated
(Neumaier) float64 sum, with relative error below about
$(2w+2) \cdot 2^{-53}$. `StabilizerCode.uncorrectable_counts` returns the
int64 counts whenever every $\binom{n}{m}$, $m \le w$, is below $2^{63}$. An
ensemble result stores the counts as float64, which holds every integer below
$2^{53}$ exactly; that covers every count whenever $\binom{n}{w} \lt 2^{53}$
(for example $\binom{256}{4} \approx 1.7 \cdot 10^{8}$).

**Cost.** At most $\sum_{j \lt w}\binom{n}{j}$ prefixes are visited (fewer
after pruning). Each correctable prefix of length $w - 1$ costs up to
$4^{w-1}$ hash lookups of $\lceil s/64\rceil$ words, an $O(n)$ scan of the hit
flags and, for the few qubits that hit, up to $4^{w-1}$ lookups in the N(S)
table. That is roughly $\binom{n}{w-1}\big(4^{w-1}\lceil r/64\rceil + n\big)$
word operations, against $\binom{n}{w}$ full rank computations for a naive
enumeration. The measured times are in §4.2.

### 3.4 First-failure scans (`first_failure_samples`)

`_kernel/entropy.py::first_failure_samples(tab, ws, ref_rows, k, prng, system, out)`
writes one first-failure size per entry of `out`:

* With `k == 0` every entry is $n + 1$ and **no random numbers are drawn**.
* Otherwise it calls `code_columns`, then for each sample shuffles an index
  array `idx` by Fisher–Yates ($n - 1$ draws of
  `_kernel/rng.py::next_below`, which is exactly uniform: Lemire's method with
  rejection). Each sample shuffles the previous sample's order, starting from
  the identity at every call. A Fisher–Yates shuffle of any fixed arrangement
  is a uniform permutation, independent of the earlier ones, so the
  orderings are iid uniform.
* The scan adds qubit `idx[m]`'s two S-columns to an incremental basis
  (`basis_add`, vectors keyed by their lowest set bit; `basis_clear` empties
  it in time proportional to its size). While the S-rank is $2(m+1)$ the
  prefix is correctable (§3.3), and the N(S) basis is not touched.
* At the first S-rank-deficient prefix, the N(S) basis is built once from the
  whole prefix. Deficiency then persists: the S-rank grows by at most 2 per
  qubit, so it stays below $2(m'+1)$ for every later $m'$. From there both
  bases grow together until the N(S)-rank exceeds the S-rank. That $m + 1$ is
  $D$, and the scan stops.

Cost per ordering: $O(n)$ for the shuffle plus $O(D^2 \lceil r/64\rceil)$ for
the two bases. The scan stops at $D$, which is typically much smaller than $n$.

### 3.5 How the engine dispatches the probes

`_kernel/ensemble.py::run_ensemble` runs the trajectories in parallel
(`prange`). At each layer in the union of all probe schedules, it reads
$k_C$ once as a popcount (`s_ref = popcount(ref_rows) - n_ref`), then calls
`_kernel/ensemble.py::eval_probe` for every probe scheduled at that layer:

| probe kind | kernel call in `eval_probe` | value written |
|---|---|---|
| `erasure_first_failure` | `first_failure_samples(tab, ws, ref_rows, s_ref, prng, order, row)` | $M$ first-failure sizes |
| `uncorrectable_region_counts` | zeros if `s_ref == 0`, else `uncorrectable_counts_rows(tab, ws, order, ref_rows, w, row, exact)` | $N_1, \dots, N_w$, float64 |
| `code_snapshot` | `code_snapshot_rows(tab, n_sys, n, ref_rows, snap_out)` | the generators of $S$ (unmarked rows) into a separate buffer; $s = n - k_C$ into the row |
| `code_dimension` | none | `s_ref` |

The qubits are passed in the spec's `system_order`; the counts do not depend
on it, and the orderings are uniformly random either way. Each probe writes
into its own slice of the trajectory's float64 output row, and
`EnsembleResult.values[name]` reshapes it: `(trajectories, evaluations, M)`
for `erasure_first_failure` and `(trajectories, evaluations, w)` for the
counts, whose key is `uncorrectable_region_counts[w]` unless you name the probe.
With `stop_when_purified`, evaluations after a trajectory's purification
layer hold the exact values for $k_C = 0$: $n + 1$ for every first-failure
sample, 0 for every count, and $n$ for `code_snapshot`, with no code stored
(`mipt.py::_KINDS`, `_probe_fill`).

### 3.6 Per-evaluation random streams

Each run derives two 128-bit keys from its seed: one for the circuit and one
for probe sampling (`mipt.py::_keys`). In 3.0.0.dev12 (randomness contract 2,
`mipt.RANDOMNESS_CONTRACT`), every evaluation of a sampled probe
(`erasure_first_failure`, `erasure_lost_profile`) at layer $L$ of trajectory
$t$ draws only from its own stream, `stream_state(key_probe, mix(t, h, L))`.
Here $h$ is a 64-bit hash of the probe's name (blake2b) and `mix` a fixed
chained splitmix64. The exact definitions are in the module docstrings of
`_kernel/ensemble.py` (Probe streams) and `_kernel/rng.py`. Each evaluation's
orderings start from the identity. Consequences:

* the samples at an evaluation depend only on the seed, the trajectory, the
  probe's name, the layer, $M$ and the state, never on other probes, on the
  other checkpoints, or on earlier evaluations (an evaluation with
  $k_C = 0$ draws nothing, and that no longer shifts anything);
* `MonitoredCircuit.replay(t, seed=..., probes=[...], checkpoints=[L])`
  reproduces the original run's samples at $L$ exactly, so one trajectory can
  be re-examined at new depths;
* fewer samples are a prefix of more (same name, seed, trajectory and layer);
* results written under contract 1 (3.0.0.dev11 and earlier) cannot be merged
  with or extended by dev12 runs.

`StabilizerCode.first_failure_sizes(samples, seed=...)` draws from a stream
seeded by `_kernel/rng.py::seed_state(seed)` and is independent of all of
this.

### 3.7 The Python layers

| layer | function | what it gives |
|---|---|---|
| `mipt` probes | `erasure_first_failure(samples=100)` | $D_1, \dots, D_M$ per evaluation |
| | `uncorrectable_region_counts(max_size=2)` | $N_1, \dots, N_w$ per evaluation |
| | `code_snapshot()` | the code itself; `EnsembleResult.code(t, c)` returns it as a `StabilizerCode` |
| `codes.StabilizerCode` | `uncorrectable_counts(w)`, `first_failure_sizes(M, seed=...)`, `lost_information(A)`, `is_correctable(A)` | the same kernels on a saved code, with any $w$ or $M$ later |
| | `from_simulator(sim, reference)` | the code held by a `Simulator`'s system |
| `observables` (any `Simulator`) | `random_erasure_statistics(sim, system, reference, sizes, samples)` | $I(A:R)$ for `samples` independent uniform $m$-subsets per size (independent across sizes too), from the generic entropy formula |
| | `erasure_profile(sim, system, reference, samples)` | $I(R:A_m)$ along random orderings (nested, like the engine) |
| `analysis` | `failure_by_size(first, n)` | $\hat P_C(m)$, $m = 0..n$, from the samples |
| | `exact_failure_by_size(counts, n)` | exact $P_C(m)$, $m = 1..w$ |
| | `hybrid_failure_by_size(first, n, exact_counts=counts)` | exact for $m \le w$, sampled above |
| | `erasure_failure_probability(first, n, e, exact_counts=counts)` | $\hat\Phi_C(e)$ per code (§4.3) |
| | `erasure_tolerance(first, n, delta, exact_counts=counts)` | $e_\delta$ of the codes passed |
| | `mean_first_failure(first, n, exact_counts=counts)` | $\hat\mu_C$ |
| | `distance_bounds(first, n, exact_counts=counts)` | `(low, high)` on $d$ |
| | `junction_dip(first, n, counts)` | whether sampled $\hat P_C(w+1)$ falls below exact $P_C(w)$ |

All `analysis` functions accept leading axes (one row per code) and return
one result per code, except `erasure_tolerance`, which averages $\Phi_C$ over
all the codes it is given.

**The hybrid estimator.** With exact counts for $m \le w$,
`erasure_failure_probability` uses the exact $P_C(m)$ there and lets each
ordering cover only the sizes above $w$:

$$
\hat\Phi_C(e) = \sum_{m=1}^{w} b_m(e) P_C(m) + \frac{1}{M}\sum_{j=1}^{M} \Pr\big[\mathrm{Bin}(n, e) \ge \max(D_j, w+1)\big]
$$

(`analysis.py::_hybrid_phi`). This is exactly
$\sum_m b_m(e) \tilde P_C(m)$ for the hybrid curve $\tilde P_C$ of
`hybrid_failure_by_size`, and it is unbiased because it is linear in
$\hat P_C$. `mean_first_failure` uses the same curve, and `erasure_tolerance`
the same estimator: it averages over the codes passed, locates the last
crossing of $\delta$ on a grid of 2048 steps in $e$, and refines it by
bisection to about $10^{-12}$. The last crossing is used because the hybrid
estimate need not be monotone in $e$.

**`junction_dip`.** The true $P_C$ is non-decreasing (§2.3). The sampled part
of the hybrid curve is monotone and so is the exact part, but at the junction
a sampled $\hat P_C(w+1)$ can fall below the exact $P_C(w)$ by chance.
`junction_dip` flags this per code. Nothing is clipped, since clipping would
bias the estimators; a dip says to use more orderings.

**`distance_bounds`.** `high` $= \min_j D_j$ (always $\ge d$). Without counts
`low` $= 1$. With counts, the smallest $m$ with $N_m > 0$ is $d$ exactly
(`low = high = m`), and if there is none `low` $= w + 1$. A code with
$k_C = 0$ gets $(n+1, n+1)$: it has no distance, so leave it out.

A code needs no trajectory at all. The five-qubit code has $d = 3$, and every
3-qubit erasure destroys its logical qubit, so every ordering fails at
exactly $D = 3$ and $\Phi_C(e) = \Pr[\mathrm{Bin}(5, e) \ge 3]$:

```python
import math

import numpy as np

from sparsegf2 import StabilizerCode, analysis

five = StabilizerCode(["XZZX_", "_XZZX", "X_XZZ", "ZX_XZ"])
print(five.num_logical, five.uncorrectable_counts(3))  # 1 [ 0  0 10]
first5 = five.first_failure_sizes(500, seed=0)
assert np.all(first5 == 3)
low5, high5 = analysis.distance_bounds(first5, 5, exact_counts=five.uncorrectable_counts(2))
assert (low5, high5) == (3, 3)
e5 = np.array([0.05, 0.1, 0.3])
exact5 = sum(math.comb(5, m) * e5**m * (1 - e5) ** (5 - m) for m in range(3, 6))
assert np.allclose(analysis.erasure_failure_probability(first5, 5, e5), exact5, rtol=0, atol=1e-12)
```

### 3.8 JSON, CLI and MCP

The JSON layer (`api.py`, shared by the `sparsegf2` command and the
`sparsegf2-mcp` server) takes the probes by their `mipt` names and
arguments:

```json
{"probes": [{"kind": "code_dimension"},
            {"kind": "uncorrectable_region_counts", "max_size": 2},
            {"kind": "erasure_first_failure", "samples": 100},
            {"kind": "code_snapshot", "checkpoints": "final"}]}
```

Every probe also takes `"name"` and its own `"checkpoints"`. `run_ensemble`
returns a summary, and so does `load_results` after a background job
(`start_ensemble_job`). For each erasure probe, the summary
holds a `what` line, the mean $P_C(m)$ curve over trajectories and its
standard error at the last checkpoint with data (`at_layer`, `count`, `mean`,
`stderr`), and a `vs_depth` grid of up to 8 checkpoints by 17 sizes:

* `erasure_first_failure`: each trajectory's $\hat P_C(m)$, $m = 0..n$
  (`api.py::_vector_stats`);
* `uncorrectable_region_counts`: each trajectory's exact $P_C(m)$,
  $m = 0..w$, with $P_C(0) = 0$ prepended.

In 3.0.0.dev12 these means are over **all** trajectories, and a trajectory
without a code contributes $P_C \equiv 0$ (§6). Conditioning the summaries on
$k_C > 0$ is planned. With `output_dir`, the full result is saved as
`result.npz`, which `mipt.EnsembleResult.load` reads back, together with
`spec.json` and `provenance.json`. Compute $\Phi$, $e_\delta$, $\mu_C$ and the
distance bounds from that file with `analysis`.

For a single circuit, `run_circuit` evaluates the `observables` forms on the
final state (or at every `TICK`):
`{"kind": "random_erasure_statistics", "system": "0:16", "reference": "16:32", "sizes": [1, 2, 4], "samples": 50}`
and `{"kind": "erasure_profile", "system": "0:16", "reference": "16:32", "samples": 10}`.
`estimate_ensemble` prices the erasure probes pessimistically
(`api.py::_probe_us`). The counts are priced in logarithms, so an absurd
`max_size` gives a huge estimate instead of an overflow.

---

## 4. Cost and accuracy

### 4.1 Complexity

| routine | cost per evaluation |
|---|---|
| `code_columns` | $O(N r \lceil 2n/64\rceil)$ word operations, once per erasure kernel call |
| `uncorrectable_counts_rows`, cutoff $w$ | about $\binom{n}{w-1}(4^{w-1}\lceil r/64\rceil + n)$, after pruning less |
| `first_failure_samples`, $M$ orderings | $M \cdot O(n + D^2\lceil r/64\rceil)$ |
| `code_snapshot_rows` | $O(N n)$; stores $n \cdot 2\lceil n/64\rceil$ words |

Memory per evaluation per trajectory: 8 bytes per first-failure sample, 8
bytes per count, and $16 n \lceil n/64\rceil$ bytes per code snapshot (16 KB at
$n = 256$).

### 4.2 Measured costs

Single-thread cost per evaluation at $n = 256$, for codes generated by random
circuits (AGENTS.md, and the `mipt.uncorrectable_region_counts` docstring):

| probe | cost per evaluation | regions in the exhaustive regime |
|---|---|---|
| `erasure_first_failure`, $M = 100$ | 3 to 7 ms | |
| `uncorrectable_region_counts`, $w = 2$ | about 1 ms | 32,896 |
| `uncorrectable_region_counts`, $w = 3$ | 40 to 50 ms | 2,796,416 |
| `uncorrectable_region_counts`, $w = 4$ | about 7 s | 177,589,056 |

Divided by the regions in the exhaustive regime, the counts cost about 15 to
40 ns per region at $n = 256$ on one thread. So, at $n = 256$,
$N_{\mathrm{exact}} \times 4 \cdot 10^{-8}$ s is a pessimistic planning figure
per code and evaluation. Multiply by trajectories and evaluations, and divide
by the cores in use: the engine runs trajectories in parallel. A dedicated
benchmark of these costs, by $n$ and $w$, on codes from real trajectories, is
planned.

### 4.3 Error bars

* **One code, one size.** $\hat P_C(m)$ has standard error
  $\sqrt{P(1-P)/M}$, estimated with $\hat P$. When $\hat P_C(m) = 0$ from $M$
  orderings, the one-sided 95% bound is $P_C(m) \le 3/M$ (rule of three).
* **One code, a function of the curve.** Do not combine per-$m$ error bars:
  the sizes are correlated (§2.4). $\hat\Phi_C(e)$ and $\hat\mu_C$ are means of
  per-ordering quantities, $f(D_j) = \Pr[\mathrm{Bin}(n, e) \ge \max(D_j, w+1)]$
  (with $w = 0$ when there are no counts) and $D_j$ (or
  $\max(D_j - w - 1, 0)$ with counts), so their standard error is the sample
  standard deviation of those quantities divided by $\sqrt{M}$. The exact part
  adds no variance.
* **An ensemble.** Trajectories are independent, so the mean over $T$ codes and
  its standard error (`analysis.mean_and_error` over the per-code values)
  include both the code-to-code spread and each code's sampling noise. With a
  graph collection and `circuits_per_graph > 1`, trajectories on the same
  graph share its disorder. For error bars that include graph disorder,
  average per graph first (`EnsembleResult.graph_index`) and take the spread
  over graphs.
* **$e_\delta$** has no closed-form error. Resample the codes (a bootstrap over
  trajectories, or over graphs) and recompute `erasure_tolerance`.

### 4.4 Why small erasures need the exhaustive regime, and choosing $m^{\ast}(n)$

The relative standard error of $\hat P_C(m)$ is $\sqrt{(1-P)/(MP)}$. Where
$M P_C(m) \lesssim 1$, the sampled estimate is usually exactly 0. Small sizes
are where this happens. A distance-2 code at $n = 256$ with a single
uncorrectable pair has $P_C(2) = 1/32{,}640$; with $M = 100$ orderings the
chance that any $D_j \le 2$ is about 0.3%. Yet that pair sets the leading term
$\Phi_C(e) \approx N_d e^d = e^2$ as $e \to 0$. The exhaustive regime gives
these $N_m$ exactly, at a cost fixed in advance.

A practical way to choose $m^{\ast}(n)$:

1. **Cost.** Pick the largest $w$ whose time per code (§4.2), multiplied by
   trajectories and evaluations and divided by cores, fits your budget. This
   is formula (4) with $B$ in regions; `cutoff_from_budget` above computes it.
2. **Accuracy.** The sampled regime is informative where $M P_C(m) \gg 1$. If
   $\hat P_C(w+1)$ is 0 or rests on a handful of orderings, either raise $w$
   or raise $M$ (orderings are cheap: §4.2).
3. **Relevance.** At the erasure rates you care about, $b_m(e)$ peaks at
   $m \approx ne$. If $ne \gg w$, the exact part carries little weight in
   $\Phi_C(e)$ and the samples set the error. Choose $M$ from the
   per-ordering standard error of §4.3. The exact part matters most where
   $ne$ is comparable to $w$: small $n$, or the small $e_\delta$ that a small
   $\delta$ gives.

---

## 5. Worked example: one Watts–Strogatz trajectory at n = 16

One trajectory of a circuit on a Watts–Strogatz graph ($n = 16$, degree 4,
rewiring probability $\beta = 0.1$; purification picture, 64 layers of
`random_edges` at measurement rate 0.1, low enough that the code usually
keeps $k_C > 0$). At $n = 16$ all $2^{16}$ regions can be checked
by brute force, so every estimate can be compared with the exact answer. The
example takes the first trajectory whose code has distance at least 2 (so
that the exhaustive regime has something to show), or failing that the first
that holds a code. Which trajectory that is, and its numbers, depend on the
version's random streams, so the printed values are not quoted here. The
assertions are what the example establishes.

**Run, and get the code two ways.** From the engine's snapshot, and from the
exported circuit replayed on a general `Simulator` (whose plain measurement
rule differs from the engine's, but leaves the same stabilizer group):

```python
import math

import numpy as np

from sparsegf2 import Simulator, StabilizerCode, analysis, graphs, mipt, observables

n, M, w = 16, 2000, 3
spec = mipt.MonitoredCircuit(
    num_system=n,
    edges=graphs.watts_strogatz(n, 4, 0.1, seed=1),
    layers=64,
    measurement_rate=0.1,
    schedule="random_edges",
    reference="purification",
    probes=[
        mipt.code_dimension(),
        mipt.uncorrectable_region_counts(max_size=w),
        mipt.erasure_first_failure(samples=M),
        mipt.code_snapshot(),
    ],
)
res = spec.run(16, seed=7)
k_final = res.values["code_dimension"][:, -1]
all_counts = res.values[f"uncorrectable_region_counts[{w}]"][:, -1]  # N_1..N_w, per trajectory
holds_code = k_final > 0
distance_2 = holds_code & (all_counts[:, 0] == 0)  # no single qubit fails: d >= 2
t = int(np.flatnonzero(distance_2 if distance_2.any() else holds_code)[0])
k = int(k_final[t])
code = res.code(t, 0)  # the snapshot at the final layer
counts = all_counts[t]
first = res.values["erasure_first_failure"][t, -1]  # D_1, ..., D_M
print(f"trajectory {t}: k = {k}; engine counts N_1..N_{w} = {counts.astype(int).tolist()}")

sim = Simulator(spec.num_qubits, phases=False)
sim.run(spec.circuit(t, seed=7))
refs = spec.reference_qubits
replayed = StabilizerCode.from_simulator(sim, reference=refs)
assert replayed.num_logical == code.num_logical == k == observables.code_dimension(sim, refs)
assert np.array_equal(replayed.uncorrectable_counts(w), counts)
assert np.array_equal(code.uncorrectable_counts(w), counts)
```

**Brute force with formula (1).** An independent GF(2) rank on integers
(read as bit vectors), the columns of $H$ packed with numpy, and the ranks
$\operatorname{rank}(H_A)$ of all $2^{16}$ regions by a depth-first walk over
subsets. Then $\ell_C(A)$ for every region at once, and the exact $N_m$ and
$P_C(m)$ for every $m$:

```python
def gf2_rank(vectors):
    """Rank over GF(2) of integers read as bit vectors."""
    basis = []  # distinct leading bits, in decreasing order
    for v in vectors:
        for b in basis:
            v = min(v, v ^ b)  # clears b's leading bit from v if v has it
        if v:
            basis.append(v)
            basis.sort(reverse=True)
    return len(basis)


H = code.stabilizer_matrix()  # (s, 2n): X bits of qubits 0..n-1, then Z bits
s = gf2_rank(int(row) for row in H.astype(np.int64) @ (1 << np.arange(2 * n)))
assert s == n - k
col = [int(c) for c in H.T.astype(np.int64) @ (1 << np.arange(H.shape[0]))]  # column j of H


def ranks_of_all_regions():
    """rank[a] = rank(H_A) for every region A, given as a bitmask a."""
    rank = np.zeros(1 << n, dtype=np.int64)

    def extend(a, first_qubit, basis):
        rank[a] = len(basis)
        for q in range(first_qubit, n):
            grown = list(basis)
            for v in (col[q], col[n + q]):  # the X and Z columns of qubit q
                for b in grown:
                    v = min(v, v ^ b)
                if v:
                    grown.append(v)
                    grown.sort(reverse=True)
            extend(a | (1 << q), q + 1, grown)

    extend(0, 0, [])
    return rank


masks = np.arange(1 << n)
size = sum((masks >> q) & 1 for q in range(n))  # |A|
rank_A = ranks_of_all_regions()
rank_Abar = rank_A[(1 << n) - 1 - masks]  # the complement of A
ell = 2 * size - s - rank_A + rank_Abar  # formula (1), every region at once

N = np.bincount(size[ell > 0], minlength=n + 1)  # N_m, m = 0..n
binom = np.array([math.comb(n, m) for m in range(n + 1)])
p_exact = N / binom  # P_C(m), formula (2)
d = int(size[ell > 0].min())
print("N_m, m = 0..n:", N.tolist())
print("distance d =", d)
```

**Agreement.** The brute force against the engine's counts, against
`StabilizerCode.lost_information`, and against the generic entropy formula
$I(A:R) = S(A) + S(R) - S(AR)$ on the replayed state
(`observables.lost_logical_information`, a different code path). It also
checks the structural facts of §1.2 and §2:

```python
assert np.array_equal(N[1 : w + 1], counts)  # exhaustive regime: exact
assert ell.min() == 0 and ell[-1] == 2 * k  # nothing lost by erasing nothing; 2k by everything
assert np.all(ell + ell[(1 << n) - 1 - masks] == 2 * k)  # complementarity
assert np.all(np.diff(p_exact) >= 0) and p_exact[0] == 0 and p_exact[n] == 1
assert np.all(p_exact[n - d + 1 :] == 1)  # every erasure of more than n - d qubits fails

rng = np.random.default_rng(0)
for _ in range(300):
    region = rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False)
    bitmask = sum(1 << int(q) for q in region)
    assert ell[bitmask] == code.lost_information(region)
    assert ell[bitmask] == observables.lost_logical_information(sim, region, refs)
```

**The sampled regime.** $\hat P_C(m)$ from the $M = 2000$ first-failure sizes
agrees with the exact $P_C(m)$ at every $m$ within 5 binomial standard errors
$\sigma_m = \sqrt{P_C(m)(1-P_C(m))/M}$. Where $\sigma_m = 0$
($P_C \in \lbrace 0, 1\rbrace$) the estimate must be exact, and it is (§2.4).
The hybrid curve is the exact curve up to $w$ and the sampled one above:

```python
p_hat = analysis.failure_by_size(first, n)
sigma = np.sqrt(p_exact * (1 - p_exact) / M)
assert np.all(np.abs(p_hat - p_exact) <= 5 * sigma)  # a 5-sigma band, exact where sigma = 0

hybrid = analysis.hybrid_failure_by_size(first, n, exact_counts=counts)
assert np.array_equal(hybrid[: w + 1], p_exact[: w + 1])
assert np.array_equal(hybrid[w + 1 :], p_hat[w + 1 :])
```

**$\Phi_C(e)$, its estimator, $d$ and $\mu_C$.** The exact
$\Phi_C(e) = \sum_m b_m(e) P_C(m)$ comes from the brute-force curve. The
distribution of $D$ follows from the claim of §2.4,
$\Pr[D = m] = P_C(m) - P_C(m-1)$. From it, the mean of the hybrid estimator
equals $\Phi_C(e)$ exactly (unbiasedness, to rounding), and its standard
deviation gives the 5σ band the estimate must fall in:

```python
e = np.array([0.02, 0.05, 0.1, 0.2])
b = np.array([[math.comb(n, m) * x**m * (1 - x) ** (n - m) for m in range(n + 1)] for x in e])
phi_exact = b @ p_exact
p_D = np.diff(np.append(p_exact, 1.0), prepend=0.0)  # Pr[D = m], m = 0..n+1
f = np.array([[row[max(m, w + 1) :].sum() for m in range(n + 2)] for row in b])  # f(D)
f_mean, f_sd = f @ p_D, np.sqrt(np.maximum(f**2 @ p_D - (f @ p_D) ** 2, 0.0))
assert np.allclose(b[:, 1 : w + 1] @ p_exact[1 : w + 1] + f_mean, phi_exact, rtol=0, atol=1e-12)

phi_hat = analysis.erasure_failure_probability(first, n, e, exact_counts=counts)
assert np.all(np.abs(phi_hat - phi_exact) <= 5 * f_sd / math.sqrt(M) + 1e-12)
print("Phi_C(e) exact:", np.round(phi_exact, 6).tolist())

low, high = analysis.distance_bounds(first, n, exact_counts=counts)
assert low <= d <= high and (d > w or low == high == d)
mu_exact = (1 - p_exact).sum()  # mu_C = E[D]
mu_hat = analysis.mean_first_failure(first, n, exact_counts=counts)
mu_sd = math.sqrt(max(p_D @ np.arange(n + 2) ** 2 - mu_exact**2, 0.0))  # sd of D
assert abs(mu_hat - mu_exact) <= 5 * mu_sd / math.sqrt(M) + 1e-9
print("mu_C exact:", round(float(mu_exact), 4))
```

The last band uses the standard deviation of $D$ itself. That bounds the
standard deviation of the hybrid's per-ordering term $\max(D - w - 1, 0)$
(§4.3), since a 1-Lipschitz function of $D$ cannot have a larger variance, so
the band is conservative.

To study a code further, keep it: `code.first_failure_sizes(100_000, seed=1)`
gives more orderings, and `code.uncorrectable_counts(4)` a larger exhaustive
regime, both on the saved snapshot without rerunning the circuit.

---

## 6. Pitfalls

* **Codes with $k_C = 0$.** They have $P_C \equiv 0$, $D = n + 1$ for every
  sample, $\Phi_C = 0$ and no distance. Averages over all trajectories (the
  JSON summaries in 3.0.0.dev12, `EnsembleResult.mean`) mix them in as
  perfect protection. To condition on a code, select the rows with
  $k_C > 0$ (a `code_dimension` probe at the same layers, or any first-failure
  sample $\le n$) before calling `erasure_tolerance`, `mean_first_failure` or
  `distance_bounds`. The fraction with $k_C > 0$ is a result in itself.
* **$P_C(m)$ is not the per-qubit channel.** $P_C(m)$ is at fixed size;
  independent erasure is $\Phi_C(e)$, the binomial mixture. Do not read
  $P_C(ne)$ as the failure probability at rate $e$.
* **The sizes are correlated.** The sampled estimates at different $m$ come
  from the same orderings. Fit or combine them with that in mind, and get
  error bars for $\Phi$ or $\mu$ from the per-ordering values (§4.3).
* **Sampling cannot see rare small failures.** For $m$ with
  $M P_C(m) \lesssim 1$ the estimate is mostly 0 (§4.4). Use the exhaustive
  regime for those sizes, and look at `junction_dip` where the two regimes
  meet.
* **The exhaustive cost grows like $\binom{n}{w-1}4^{w-1}$.** Give large
  cutoffs their own sparse `checkpoints` (for example `"final"`), and check
  `estimate_ensemble` first.
* **Results are float64.** The counts are exact integers stored in float64
  (exact below $2^{53}$, far above any feasible enumeration). First-failure
  sizes are integers in $[1, n+1]$. `analysis` checks both and refuses
  anything else.
* **Single picture.** With one reference qubit, $k_C \le 1$, and the code is
  the one protecting the partner's half of the Bell pair. Its $P_C$ answers
  "is that one logical qubit damaged", and a random partner adds variation
  between trajectories.
* **Contracts.** Results from 3.0.0.dev11 or earlier (randomness contract 1)
  differ from dev12's, circuits included, because the streams changed.
  Results of the two contracts cannot be merged, and an old output directory
  cannot be extended.
* **Replay needs the same spec.** A replayed trajectory reproduces its
  samples only if the probe name, the layer, $M$ and the seed are the same
  (§3.6). Renaming a probe changes its stream.
