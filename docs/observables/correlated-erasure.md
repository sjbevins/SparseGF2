# Geometry-correlated erasure

How SparseGF2 computes the failure probability $P_{C,G,J}(m)$ of a code under
erasures that cluster on an interaction graph $G$, the binomially mixed
channel $\Phi_{C,G,J}(\varepsilon)$ and its tolerance $\varepsilon_\delta(J)$,
and how it samples the erased regions.

This document follows the owner's study "Geometry-correlated erasures in
monitored quantum circuits" from its definitions, through the exact limits and
the samplers with their proofs, to the functions you call and worked examples
checked by brute force. It is written for a reader fluent in the stabilizer
formalism and in Markov chains, and for language models: every symbol is
defined where it first appears, every formula is the one the code computes,
and every statement about the code cites the function it describes as
`path::function` (paths relative to `src/sparsegf2/`).

**What is built.** The graph side (`erasures.py`, `_kernel/erasure_chain.py`):
boundaries, exact densities of states, the exact ring and path samplers and
the swap chain with replica exchange, with diagnostics. The code side
(`codes.py`, `_kernel/erasure_failure.py`): failures of sampled regions,
exact boundary-resolved counts and the $J \to \infty$ limits on saved codes.
The curves (`analysis.py`): exact $P_J(m)$, full curves, $\Phi$ and
$\varepsilon_\delta$. The in-run probe `mipt.correlated_erasure`
(`_kernel/ensemble.py`), and the read-only tools `sample_erasure_regions` and
`erasure_curves` in the JSON service, the CLI (`sparsegf2 regions`,
`sparsegf2 curves`) and the MCP server (§7). Everything here describes that
code, at randomness contract 2.

Companion documents: [random-region-erasure.md](random-region-erasure.md) (the
uniform case $J = 0$ in depth: $P_C(m)$, $\Phi_C(e)$, $e_\delta$, the first-failure
estimator and the exhaustive counts) and [lost-information.md](lost-information.md)
(the lost information $\ell_C(A) = I(A:R)$ of one region and complementarity).

## Contents

1. [The study, transcribed](#1-the-study-transcribed)
2. [The Ising form](#2-the-ising-form)
3. [The limits, precisely](#3-the-limits-precisely)
4. [The samplers, with proofs](#4-the-samplers-with-proofs)
5. [Estimators and their errors](#5-estimators-and-their-errors)
6. [Exact answers](#6-exact-answers)
7. [From the kernels to the code you run](#7-from-the-kernels-to-the-code-you-run)
8. [Cost and accuracy](#8-cost-and-accuracy)
9. [Worked examples](#9-worked-examples)
10. [Pitfalls](#10-pitfalls)
11. [Literature](#11-literature)

The Python blocks run as written: `tests/test_docs.py` executes them in order,
in one namespace, and together they take a few seconds once the kernels are
compiled.

**Notation.**

| symbol | meaning | in the code |
|---|---|---|
| $G = (V, E)$ | the interaction (erasure) graph; the simulator's $V = \lbrace 0, \dots, n-1\rbrace$ | `edges`, an `(E, 2)` integer array (`graphs.py`), repeated edges removed by `graphs.from_edges` |
| $A$, $m$, $\bar A$ | an erased region, its size $m = \lvert A\rvert$, its complement $V \setminus A$ | `region`, `size` |
| $\partial A$, $b(A)$ | the edge boundary and its size | `erasures.boundary` |
| $J$ | the coupling, any finite real number | `couplings` |
| $Q_J(A \mid m)$, $Z_m(J)$ | the correlated law on regions of size $m$ and its partition function | `erasures.sample_regions`, `erasures.boundary_moments` |
| $N_m(b)$ | the number of regions of size $m$ with boundary $b$ (density of states) | `erasures.boundary_counts`, `BoundaryCounts.total` |
| $k$ | the number of logical qubits $k_C = S(R)$ | `StabilizerCode.num_logical` |
| $F_C(A)$ | $\mathbf{1}[I(A:R) > 0]$: erasing $A$ loses logical information | `StabilizerCode.region_failures` |
| $U_m(b)$ | the number of failing regions of size $m$ with boundary $b$ | `BoundaryCounts.failed` |
| $P_{C,G,J}(m)$ | the failure probability under $Q_J$ (also written $P_J(m)$) | `StabilizerCode.correlated_failure`, `BoundaryCounts.exact` |
| $\Phi_{C,G,J}(\varepsilon)$, $\varepsilon_\delta(J)$ | the binomially mixed failure probability and its tolerance | `analysis.failure_probability_from_curve`, `analysis.tolerance_from_curve` |

---

## 1. The study, transcribed

This section is the owner's note (13 pages, kept outside the repository) in
Markdown, section by section. Its statements are kept, and its proofs are
completed where the note leaves steps out. Two corrections are marked
**[corrected]** and listed in §1.9. The note numbers the vertices $1, \dots, n$;
the simulator numbers them $0, \dots, n-1$. The note writes $U_m(A)$ for the
uniform law; this document writes $\mathrm{Unif}_m(A)$, because $U_m(b)$ is the
failure count of §6.

### 1.1 Definitions

**Definition (Interaction graph).** Let $G = (V, E)$ be a finite, undirected,
simple graph with $V = \lbrace 1, \dots, n\rbrace = [n]$. Vertices represent
the $n$ physical qubits; edges represent the allowed two-qubit interactions.

**Definition (Erasure region).** An erasure region is a subset $A \subseteq V$.
A vertex $v$ belongs to $A$ precisely when the corresponding physical qubit is
erased. The size of the region is $m = \lvert A\rvert$.

**Definition (Configuration space).** For fixed $m$,

$$
\Omega_{n,m} = \lbrace A \subseteq V : \lvert A\rvert = m\rbrace, \qquad \lvert\Omega_{n,m}\rvert = \binom{n}{m}.
$$

**Definition (Uniform fixed-weight erasure distribution).** For any
$A \in \Omega_{n,m}$, $\mathrm{Unif}_m(A) = 1/\binom{n}{m}$. Every erased subset
of size $m$ has precisely the same probability, and the distribution is
completely independent of the graph.

**Definition (Edge boundary).** For any $A \subseteq V$, its graph-edge boundary
and its size are

$$
\partial_G A = \lbrace \lbrace u, v\rbrace \in E : u \in A,\ v \notin A\rbrace, \qquad b_G(A) = \lvert \partial_G A\rvert .
$$

**Definition (Partition function).** For fixed $G$, $n$, $m$ and a correlation
parameter $J$ (the note takes $J \ge 0$),

$$
Z_{G,m}(J) = \sum_{A \in \Omega_{n,m}} \exp\big(-J\, b_G(A)\big),
$$

a finite, strictly positive quantity.

**Definition (Geometry-correlated fixed-weight erasures).**

$$
Q_{G,J}(A \mid m) = \frac{\exp\big(-J\, b_G(A)\big)}{Z_{G,m}(J)} \quad \text{for } A \in \Omega_{n,m},
$$

and probability zero for subsets of other sizes. $J = 0$: no geometric
preference. $J > 0$: subsets with smaller graph-edge boundaries are favoured.
The normalization is exact, $\sum_{A \in \Omega_{n,m}} Q_{G,J}(A \mid m) = 1$.
$J$ is dimensionless and only controls the statistical preference for small
boundaries.

### 1.2 Theorem 1 and Corollary 1

**Theorem 1 (Exact uniform-erasure limit).** Let $G$ be any finite undirected
graph with $n$ vertices and $0 \le m \le n$. Then for every $A \in \Omega_{n,m}$,
$Q_{G,0}(A \mid m) = \mathrm{Unif}_m(A) = 1/\binom{n}{m}$.

*Proof.* At $J = 0$ every weight is $e^{0} = 1$, so
$Z_{G,m}(0) = \lvert\Omega_{n,m}\rvert = \binom{n}{m}$. ∎

**Corollary 1 (Exact equality of all fixed-weight observables).** For any
$f : \Omega_{n,m} \to \mathbb{R}$,
$\mathbb{E}_{Q_{G,0}}[f(A)] = \mathbb{E}_{\mathrm{Unif}_m}[f(A)]$.

*Proof.* The two measures are equal (Theorem 1). ∎

### 1.3 Failure probabilities

Let $C \sim \mathcal{E}$ be a stabilizer code from the ensemble $\mathcal{E}$
(here: the code a monitored circuit holds at a checkpoint, its logical
information purified by a reference $R$), and define

$$
F_C(A) = \mathbf{1}\big[I(A:R) > 0\big].
$$

$I(A:R)$ is the mutual information, in bits, between the erased qubits and the
reference: the lost information $\ell_C(A)$ of
[lost-information.md](lost-information.md), and the erasure of $A$ is
correctable exactly when it is 0. The original fixed-weight failure
probability and its correlated version are

$$
P_C(m) = \frac{1}{\binom{n}{m}} \sum_{A \in \Omega_{n,m}} F_C(A), \qquad
P_{C,G,J}(m) = \sum_{A \in \Omega_{n,m}} Q_{G,J}(A \mid m)\, F_C(A).
$$

Theorem 1 gives the exact identity $P_{C,G,0}(m) = P_C(m)$. It requires no
assumption about the code being random, Clifford, monitored or geometrically
local: it is an identity between probability measures.

### 1.4 Independent erasure and the binomially mixed law

**Definition (IID erasure distribution).** Each physical qubit is erased
independently with probability $\varepsilon \in [0, 1]$, so for any
$A \subseteq V$, $\mathbb{P}_{\mathrm{IID},\varepsilon}(A) = \varepsilon^{\lvert A\rvert}(1-\varepsilon)^{n-\lvert A\rvert}$.
The number of erased qubits $M = \lvert A\rvert$ is $\mathrm{Bin}(n, \varepsilon)$,
$\Pr(M = m) = \binom{n}{m}\varepsilon^m(1-\varepsilon)^{n-m}$, and conditional
on $M = m$ every region of size $m$ has the same probability:

$$
\Pr(A \mid M = m) = \frac{\varepsilon^m(1-\varepsilon)^{n-m}}{\binom{n}{m}\varepsilon^m(1-\varepsilon)^{n-m}} = \frac{1}{\binom{n}{m}} \qquad (0 \lt \varepsilon \lt 1).
$$

**Definition (Binomially mixed geometry-correlated erasure distribution).**
First sample $M \sim \mathrm{Bin}(n, \varepsilon)$; then, conditional on
$M = m$, sample $A \sim Q_{G,J}(\cdot \mid m)$. By the product rule, with
$m = \lvert A\rvert$,

$$
\mathbb{P}_{G,J,\varepsilon}(A) = \Pr(M = m)\, Q_{G,J}(A \mid m) = \binom{n}{m}\varepsilon^m(1-\varepsilon)^{n-m}\, \frac{\exp(-J\, b_G(A))}{Z_{G,m}(J)} .
$$

At the endpoints $\varepsilon = 0, 1$ the distribution is understood by its
sampling definition: no qubit, respectively every qubit, is erased (the formula
agrees with the convention $0^0 = 1$).

**Theorem 2 (Exact IID-erasure limit).** For any finite undirected $G$ and
$0 \le \varepsilon \le 1$, $\mathbb{P}_{G,0,\varepsilon}(A) = \mathbb{P}_{\mathrm{IID},\varepsilon}(A)$
for every $A \subseteq V$: at $J = 0$ the whole erasure distribution is that of
erasing each qubit independently with probability $\varepsilon$.

*Proof.* By Theorem 1,
$\mathbb{P}_{G,0,\varepsilon}(A) = \binom{n}{m}\varepsilon^m(1-\varepsilon)^{n-m}/\binom{n}{m}$. ∎

**Corollary 2 (Exact recovery of the original erasure-failure probability).**
For a fixed code $C$ define

$$
\Phi_{C,G,J}(\varepsilon) = \Pr_{A \sim \mathbb{P}_{G,J,\varepsilon}}\big[I(A:R) > 0\big] = \sum_{m=1}^{n} \binom{n}{m}\varepsilon^m(1-\varepsilon)^{n-m}\, P_{C,G,J}(m).
$$

At $J = 0$, $P_{C,G,0}(m) = P_C(m)$, so $\Phi_{C,G,0}(\varepsilon) = \Phi_C(\varepsilon)$:
both the fixed-weight random-region observable and the IID channel failure
observable are recovered exactly at $J = 0$.

*Proof.* Group $\sum_A \mathbb{P}_{G,J,\varepsilon}(A) F_C(A)$ by $m = \lvert A\rvert$.
The term $m = 0$ vanishes because $F_C(\emptyset) = 0$ for every code (erasing
nothing loses nothing; the note says "for a nontrivial code", but it holds for
all). ∎

### 1.5 Propositions 1 and 2

**Proposition 1 (Relative probability of two erasure regions).** For
$A, B \in \Omega_{n,m}$,

$$
\frac{Q_{G,J}(A \mid m)}{Q_{G,J}(B \mid m)} = \exp\big[-J\big(b_G(A) - b_G(B)\big)\big].
$$

*Proof.* $Z_{G,m}(J)$ cancels. ∎ So if $b_G(A) \lt b_G(B)$ and $J > 0$, then
$Q_{G,J}(A \mid m) > Q_{G,J}(B \mid m)$.

**Proposition 2 (Increasing $J$ reduces the expected edge boundary).** With
$\langle b\rangle_{G,J,m} = \sum_{A \in \Omega_{n,m}} Q_{G,J}(A \mid m)\, b_G(A)$,

$$
\langle b\rangle_{G,J,m} = -\frac{\partial}{\partial J}\log Z_{G,m}(J), \qquad
\frac{\partial}{\partial J}\langle b\rangle_{G,J,m} = -\operatorname{Var}_{G,J,m}\big(b_G(A)\big) \le 0 .
$$

*Proof.* $\partial_J Z_{G,m} = -\sum_A b_G(A) e^{-J b_G(A)}$; dividing by
$Z_{G,m}$ gives $\partial_J \log Z_{G,m} = -\langle b\rangle$. Differentiating
again, $\partial_J^2 \log Z_{G,m} = \langle b^2\rangle - \langle b\rangle^2$,
the variance of the boundary size, so
$\partial_J \langle b\rangle = -\operatorname{Var}(b)$. ∎

*Physical interpretation (the note's).* Increasing $J$ concentrates the
erasure ensemble on small-boundary regions, in the sense that the expected
boundary cannot increase. This verifies a sampler: at fixed $G$ and $m$ the
measured mean boundary should be nonincreasing in $J$, up to Monte Carlo
uncertainty. (`erasures.boundary_moments` computes $\log Z$, $\langle b\rangle$
and $\operatorname{Var}(b)$ exactly from $N_m(b)$;
`CorrelatedFailure.mean_boundary` is a sampled estimate.)

### 1.6 Proposition 3

**Proposition 3 (The large-$J$ limit).** Let
$b_{\min}(G, m) = \min_{A \in \Omega_{n,m}} b_G(A)$ and
$\mathcal{M}_{G,m} = \lbrace A \in \Omega_{n,m} : b_G(A) = b_{\min}(G, m)\rbrace$.
Then

$$
\lim_{J \to \infty} Q_{G,J}(A \mid m) =
\begin{cases} 1/\lvert\mathcal{M}_{G,m}\rvert, & A \in \mathcal{M}_{G,m}, \\ 0, & A \notin \mathcal{M}_{G,m}. \end{cases}
$$

*Proof.* Multiply numerator and denominator by $e^{J b_{\min}}$:
$Q_{G,J}(A \mid m) = e^{-J(b_G(A) - b_{\min})} / \sum_{A'} e^{-J(b_G(A') - b_{\min})}$.
Every exponent is $\le 0$; the terms with $b = b_{\min}$ equal 1 and the others
tend to 0, so the denominator tends to $\lvert\mathcal{M}_{G,m}\rvert \ge 1$. ∎

At $J = 0$ every region of size $m$ is equally probable; as $J \to \infty$ only
minimum-boundary regions retain nonzero probability, uniformly.

### 1.7 Exactly solvable example: the nearest-neighbour ring

Let $C_n$ be the cycle on $n \ge 3$ vertices (each connected to its two nearest
neighbours, periodic boundary conditions; $n \ge 3$ makes it a simple graph).

**Boundary size and runs.** A nonempty proper subset $A$ of the ring consists
of $r \ge 1$ disjoint contiguous runs of erased vertices (maximal blocks of
consecutive erased places). Each run has two boundary edges, one entering it
and one leaving it, so

$$
b_{C_n}(A) = 2r \qquad (1 \le \lvert A\rvert \le n - 1).
$$

*Proof, completed.* Going once round the cycle, the places alternate between
the $r$ runs and $r$ gaps (maximal blocks of unerased places): there are as
many gaps as runs because $A$ is proper and nonempty. An edge
$\lbrace i, i+1\rbrace$ is a boundary edge exactly when one end is erased and
the other is not, that is, at each of the $2r$ transitions from a run to a gap
or from a gap to a run. For $n \ge 3$ these are $2r$ distinct edges. ∎

A single contiguous interval has $r = 1$, $b = 2$; four separated sites, none
adjacent to another ($n \ge 8$), have $r = 4$, $b = 8$, so

$$
\frac{Q_{C_n,J}(A_{\text{contiguous}} \mid 4)}{Q_{C_n,J}(A_{\text{separated}} \mid 4)} = \frac{e^{-2J}}{e^{-8J}} = e^{6J}:
$$

for any $J > 0$ a particular contiguous four-site pattern is more probable
than a particular fully separated one.

**Exact combinatorial counting.** Let $N(n, m, r)$ be the number of size-$m$
regions on the labelled ring with exactly $r$ runs. For $1 \le m \le n-1$ and
$1 \le r \le \min(m, n-m)$,

$$
N(n, m, r) = \frac{n}{r}\binom{m-1}{r-1}\binom{n-m-1}{r-1} \qquad \text{[corrected: the note's statement has } n/4\text{; its proof gives } n/r\text{]},
$$

and $N(n, m, r) = 0$ for other $r$ (each run holds at least one erased place
and each gap at least one unerased place, so $r \le m$ and $r \le n - m$).

*Proof.* Count pairs (region, marked run). Such a pair is described by

* the lengths of its $r$ runs, in cyclic order from the marked one: strictly
  positive and summing to $m$;
* the lengths of its $r$ gaps, the gap after each run: strictly positive and
  summing to $n - m$;
* the vertex where the marked run begins.

The number of ways to write $m$ as an ordered sum of $r$ positive integers is
$\binom{m-1}{r-1}$ (stars and bars: choose $r - 1$ of the $m - 1$ places
between consecutive units as cuts); the number of gap compositions is
$\binom{n-m-1}{r-1}$; there are $n$ starting vertices. Every such triple lays
out a valid pair (runs and gaps alternate from the start, the lengths sum to
$n$ so the layout closes up, and the runs are maximal because the gaps are
positive), and every pair comes from exactly one triple. So there are
$n\binom{m-1}{r-1}\binom{n-m-1}{r-1}$ pairs. Every region with $r$ runs appears
in exactly $r$ of them, one per choice of its marked run; dividing by $r$
gives the formula. ∎

**Exact partition function on the ring.** Since a region with $r$ runs has
boundary $2r$,

$$
Z_{C_n,m}(J) = \sum_{r=1}^{\min(m,n-m)} N(n, m, r)\, e^{-2Jr} = \sum_{r=1}^{\min(m,n-m)} \frac{n}{r}\binom{m-1}{r-1}\binom{n-m-1}{r-1} e^{-2Jr},
$$

an exact finite sum for $1 \le m \le n-1$. For $m = 0$ or $m = n$ there is one
configuration, of boundary 0, so $Z_{C_n,0}(J) = Z_{C_n,n}(J) = 1$. At $J = 0$
the run counts partition all size-$m$ subsets, so
$Z_{C_n,m}(0) = \sum_r N(n, m, r) = \binom{n}{m}$.

**Large-$J$ limit on a ring.** For any nonempty proper region the smallest
possible boundary is 2, reached exactly by the single contiguous intervals,
and there are exactly $n$ intervals of each length $1 \le m \lt n$. Therefore
$\lim_{J\to\infty} Q_{C_n,J}(A \mid m) = 1/n$ on the intervals (0 elsewhere):

$$
\text{uniform fixed-weight erasures} \xrightarrow{\ J \uparrow\ } \text{contiguous-interval erasures}.
$$

For finite $J > 0$ the distribution includes both contiguous and
noncontiguous regions, with probabilities set by their boundary sizes.

### 1.8 Sampling $Q_{G,J}(A \mid m)$ without enumerating the regions

$Q_{G,J}$ is known exactly, but its partition function is a sum over
$\binom{n}{m}$ regions. The distribution can be sampled without ever computing
it, by a Metropolis–Hastings Markov chain at fixed erasure weight.

**State space.** Fix $G$, $m$ and $J$. The state space is $\Omega_{n,m}$; a
state $A$ is the current set of erased qubits, and every transition preserves
$\lvert A\rvert = m$.

**Proposal.** Assume $0 \lt m \lt n$. Given the current region $A$:
uniformly select an erased vertex $u \in A$; uniformly select an unerased
vertex $v \notin A$; propose exchanging their erasure statuses,
$A' = (A \setminus \lbrace u\rbrace) \cup \lbrace v\rbrace$. By construction
$\lvert A'\rvert = m$. The probability of proposing a particular allowable swap
is $q(A, A') = 1/(m(n-m))$, and the reverse move has the same probability,
$q(A', A) = q(A, A')$.

**Acceptance.** With the change in boundary size

$$
\Delta b = b_G(A') - b_G(A) \qquad \text{[corrected: the note prints } b_G(A') = b_G(A)\text{]},
$$

the ratio of target probabilities is
$Q_{G,J}(A' \mid m)/Q_{G,J}(A \mid m) = e^{-J\Delta b}$: the unknown partition
function cancels exactly. The Metropolis acceptance probability is
$a_J(A, A') = \min\lbrace 1, e^{-J\Delta b}\rbrace$. A move that decreases the
boundary, or leaves it unchanged, is always accepted; one that increases it is
accepted with probability $e^{-J\Delta b}$. For $J = 0$ every proposed swap is
accepted; for $J > 0$ the chain increasingly prefers configurations with
smaller boundaries.

### 1.9 Corrections, and the brute-force check

The two corrections:

1. **The ring count.** The note states $N(n, m, r) = \frac{n}{4}\binom{m-1}{r-1}\binom{n-m-1}{r-1}$;
   the factor is $n/r$, as its own proof derives. The block below checks the
   corrected formula against brute force for every $n = 3, \dots, 12$, every
   $1 \le m \le n - 1$ and every $r$, and shows a case where $n/4$ is wrong.
2. **The change in boundary** is $\Delta b = b_G(A') - b_G(A)$; the note
   prints $b_G(A') = b_G(A)$.

Typesetting slips fixed above without comment: the garbled ratio
$Q(A_{\text{contiguous}} \mid 4)/Q(A_{\text{separated}} \mid 4) = e^{6J}$, the
missing value in $Z_{C_n,0} = Z_{C_n,n} = 1$, the missing bar in
$1/\lvert\mathcal{M}_{G,m}\rvert$, a missing parenthesis in
$-\partial_J \log Z_{G,m}(J)$, and "large $j$" for large $J$.

What the simulator generalizes: $J$ may be any finite real number. Every
identity above holds for $J \lt 0$ too; Proposition 1 then favours the larger
boundary (spread-out regions), and Proposition 2 still says that
$\langle b\rangle$ falls as $J$ grows. $J \to \pm\infty$ is a limit, never a
coupling (§3). $G$ is any simple graph on the qubit labels $0, \dots, n-1$.

```python
import itertools
import math

import numpy as np

from sparsegf2 import erasures, graphs


def all_regions(n, m):
    """Every m-subset of 0..n-1, one per row, in lexicographic order."""
    rows = list(itertools.combinations(range(n), m))
    return np.array(rows, dtype=np.int64).reshape(len(rows), m)


def boundaries(edges, n, regions):
    """b(A) for every row A of `regions`: the edges with exactly one end in A (plain numpy)."""
    member = np.zeros((len(regions), n), dtype=bool)
    np.put_along_axis(member, regions, True, axis=1)
    return np.count_nonzero(member[:, edges[:, 0]] != member[:, edges[:, 1]], axis=1)


def ring_runs(n, regions):
    """The number of runs (maximal cyclic blocks of erased places) of each region on C_n."""
    member = np.zeros((len(regions), n), dtype=bool)
    np.put_along_axis(member, regions, True, axis=1)
    return np.count_nonzero(member & ~np.roll(member, 1, axis=1), axis=1)


for n in range(3, 13):
    ring = graphs.cycle(n)
    for m in range(1, n):
        regions = all_regions(n, m)
        r = ring_runs(n, regions)
        assert np.array_equal(boundaries(ring, n, regions), 2 * r)  # b = 2r
        observed = np.bincount(r, minlength=n + 1)  # N(n, m, r) by brute force
        for runs in range(n + 1):
            if 1 <= runs <= min(m, n - m):
                pairs = n * math.comb(m - 1, runs - 1) * math.comb(n - m - 1, runs - 1)
                assert pairs % runs == 0 and observed[runs] == pairs // runs  # (n/r) C C
            else:
                assert observed[runs] == 0
            assert erasures.ring_run_count(n, m, runs) == observed[runs]
        assert observed.sum() == math.comb(n, m) and observed[1] == n  # Z(0) = C(n, m); n intervals

# the note's n/4 is a typo: at n = 12, m = 4, r = 2 there are 126 regions, not 63
assert erasures.ring_run_count(12, 4, 2) == 126 == 12 // 2 * math.comb(3, 1) * math.comb(7, 1)
assert 12 / 4 * math.comb(3, 1) * math.comb(7, 1) == 63
```

---

## 2. The Ising form

With $\sigma_v = -1$ for $v \in A$ and $\sigma_v = +1$ otherwise, an edge
$\lbrace u, v\rbrace$ lies in $\partial A$ exactly when $\sigma_u\sigma_v = -1$,
so

$$
b(A) = \sum_{\lbrace u,v\rbrace \in E} \frac{1 - \sigma_u\sigma_v}{2} = \frac{\lvert E\rvert}{2} - \frac{1}{2}\sum_{\lbrace u,v\rbrace \in E}\sigma_u\sigma_v,
\qquad
e^{-J b(A)} = e^{-J\lvert E\rvert/2}\, \exp\Big(\frac{J}{2}\sum_{\lbrace u,v\rbrace \in E}\sigma_u\sigma_v\Big).
$$

The constant cancels in $Q_J$, and $A \mapsto \sigma$ maps $\Omega_{n,m}$ onto
the spin configurations with magnetization
$\sum_v \sigma_v = (n - m) - m = n - 2m$. So $Q_J(\cdot \mid m)$ is the
Boltzmann law of the **Ising model on $G$ with coupling $K = J/2$**
(ferromagnetic for $J > 0$, antiferromagnetic for $J \lt 0$) **at fixed
magnetization $n - 2m$**: the canonical, conserved-order-parameter ensemble.
(`erasures.py` module docstring.)

What the mapping buys:

* **The dynamics has a name and a literature.** The swap move of §1.8 exchanges
  one up spin and one down spin, so it conserves the magnetization: it is
  Kawasaki spin-exchange dynamics (Kawasaki 1966). The textbook Kawasaki move
  exchanges nearest-neighbour spins; the simulator's move exchanges an
  arbitrary (erased, unerased) pair chosen uniformly, a non-local variant
  whose proposal is symmetric because the number of such pairs, $m(n-m)$,
  does not depend on the state. Schemes that pick the pair uniformly among a
  state-dependent set (for example among the boundary edges, whose number is
  $b(A)$) are not symmetric and, without the Hastings factor (the number of
  proposable pairs at $A$ over the number at $A'$), do not sample the
  Boltzmann law (Shida and Henriques 1997).
* **Coarsening at strong coupling.** A conserved order parameter coarsens
  slowly: for local Kawasaki dynamics the domain size grows as $t^{1/3}$
  (Lifshitz–Slyozov–Wagner), against $t^{1/2}$ without conservation (Bray
  1994). Those laws are for local dynamics and do not literally apply to the
  non-local swap, but the mechanism is the same: at large $J$ a region of
  fixed size can reorganize its clusters only by moving single qubits against
  the boundary cost. That is what makes a lone chain slow at large $J$, and
  why the samplers use exact sampling where the graph allows it and replica
  exchange elsewhere (§4).
* **The ring is exactly solvable.** On $C_n$, with a fugacity $z$ per erased
  site, the transfer matrix
  $T = \begin{pmatrix} 1 & z e^{-J} \\ e^{-J} & z\end{pmatrix}$ (rows and
  columns: unerased, erased) gives
  $\sum_{A} z^{\lvert A\rvert} e^{-J b(A)} = \operatorname{tr} T^n$, so
  $Z_{C_n,m}(J)$ is the coefficient of $z^m$ in $\operatorname{tr} T^n$. The
  closed form of §1.7 is that coefficient written out. The ring has no phase
  transition at finite $J$ (a one-dimensional Ising model), but single
  intervals overtake two-run regions near
  $J_\times = \tfrac{1}{2}\ln\big[(m-1)(n-m-1)/2\big]$, where the $r = 2$ and
  $r = 1$ terms of $Z$ are equal ($N(n,m,2)e^{-4J} = N(n,m,1)e^{-2J}$; about
  2.9 at $n = 64$, $m = 16$).

The block checks the transfer-matrix statement against the closed form
(`erasures.py::ring_partition_function`, which works in log space):

```python
def trace_power_coefficients(n, coupling):
    """Coefficients of tr T^n in z, T = [[1, z x], [x, z]] with x = exp(-J)."""
    x = math.exp(-coupling)
    t = [[np.array([1.0]), np.array([0.0, x])], [np.array([x]), np.array([0.0, 1.0])]]
    power = [[np.array([1.0]), np.array([0.0])], [np.array([0.0]), np.array([1.0])]]

    def add(a, b):
        out = np.zeros(max(len(a), len(b)))
        out[: len(a)] += a
        out[: len(b)] += b
        return out

    for _ in range(n):
        power = [
            [add(np.convolve(power[i][0], t[0][j]), np.convolve(power[i][1], t[1][j])) for j in (0, 1)]
            for i in (0, 1)
        ]
    return add(power[0][0], power[1][1])


n, coupling = 12, 0.7
trace = trace_power_coefficients(n, coupling)
for m in range(1, n):
    closed = sum(erasures.ring_run_count(n, m, r) * math.exp(-2 * coupling * r) for r in range(1, n))
    assert math.isclose(trace[m], closed, rel_tol=1e-12)
    assert math.isclose(math.log(closed), erasures.ring_partition_function(n, m, coupling), rel_tol=1e-12)
```

---

## 3. The limits, precisely

### 3.1 $J = 0$

$Q_0$ is uniform on any graph (Theorem 1), so $P_{C,G,0}(m) = P_C(m)$ and
$\Phi_{C,G,0} = \Phi_C$ (Corollary 2). Everything in
[random-region-erasure.md](random-region-erasure.md) applies. In the code,
`BoundaryCounts.exact(0.0)` equals the uniform
`analysis.exact_failure_by_size(code.uncorrectable_counts(w), n)` bit for bit
(`analysis.py::exact_correlated_failure`; checked in §9(c)).

### 3.2 $J \to +\infty$ on any graph, with an explicit rate

By Proposition 3 applied to $F_C$,

$$
P_{C,G,\infty}(m) \equiv \lim_{J\to\infty} P_{C,G,J}(m) = \frac{1}{\lvert\mathcal{M}_{G,m}\rvert}\sum_{A \in \mathcal{M}_{G,m}} F_C(A) = \frac{U_m(b_{\min})}{N_m(b_{\min})},
$$

the failing fraction of the minimum-boundary regions ($U_m$, $N_m$ in §6).
The approach is exponential. Write $\epsilon_J = Q_J(b > b_{\min})$ and
$P^{>}_J(m)$ for the failure probability conditioned on $b > b_{\min}$; then
$P_J(m) = (1 - \epsilon_J) P_\infty(m) + \epsilon_J P^{>}_J(m)$, and since
$P^{>}_J$ and $P_\infty$ lie in $[0, 1]$,

$$
\lvert P_J(m) - P_\infty(m)\rvert \le \epsilon_J = \frac{\sum_{b > b_{\min}} N_m(b) e^{-J(b - b_{\min})}}{\sum_b N_m(b) e^{-J(b - b_{\min})}} \le \frac{\binom{n}{m} - N_m(b_{\min})}{N_m(b_{\min})}\, e^{-J} \qquad (J \ge 0),
$$

because every boundary above $b_{\min}$ is at least $b_{\min} + 1$. On the
ring, $\epsilon_J$ is exactly $1 - $ `erasures.ring_boundary_distribution(n, m, J)[1][0]`
(the weight of $r \ge 2$).

### 3.3 The ring $C_n$: exactly the $n$ intervals

**Claim.** For $n \ge 3$ and $1 \le m \le n - 1$, $b_{\min}(C_n, m) = 2$ and
$\mathcal{M}_{C_n,m}$ is exactly the set of the $n$ intervals
$\lbrace s, s+1, \dots, s+m-1\rbrace \pmod n$, $s = 0, \dots, n-1$.

*Proof.* $b = 2r$ with $r \ge 1$ (§1.7), so $b \ge 2$, with equality exactly
when $r = 1$, that is, when $A$ is a single run: an interval. An interval of
length $m \lt n$ has exactly one first place (the erased place whose
predecessor is unerased), so the $n$ starting places give $n$ distinct
intervals. ∎

Hence $P_{C,C_n,\infty}(m) = \frac{1}{n}\#\lbrace s : \text{the window of length } m \text{ at } s \text{ fails}\rbrace$,
which is `StabilizerCode.interval_failure_by_size(order)[m]` with
`periodic=True` (the default), for the ring laid along `order` (default
$0, \dots, n-1$, which is `graphs.cycle(n)`).

### 3.4 The open path $P_n$: only the two end intervals

Let $P_n$ have the edges $\lbrace i, i+1\rbrace$, $i = 0, \dots, n-2$ (as
`graphs.path(n)`).

**Claim.** For $n \ge 2$ and $1 \le m \le n - 1$, $b_{\min}(P_n, m) = 1$ and
$\mathcal{M}_{P_n,m}$ holds exactly two regions: the prefix
$\lbrace 0, \dots, m-1\rbrace$ and the suffix $\lbrace n-m, \dots, n-1\rbrace$.

*Proof.* Let $A$ (nonempty, proper) have $r \ge 1$ runs, maximal blocks
$[a, c]$ of consecutive erased places. Each run has a left side and a right
side. The left side contributes the boundary edge $\lbrace a-1, a\rbrace$ when
$a > 0$ ($a - 1$ is unerased, by maximality) and nothing when $a = 0$; the
right side contributes $\lbrace c, c+1\rbrace$ when $c \lt n - 1$ and nothing
when $c = n - 1$. Every boundary edge has exactly one erased end, so it is a
side of exactly one run. Hence

$$
b_{P_n}(A) = 2r - e, \qquad e = \#\lbrace \text{runs containing the end place } 0 \text{ or } n - 1\rbrace .
$$

A run containing both end places is the whole path ($m = n$), excluded, so
each run touches at most one end and $e \le \min(r, 2)$. For $r = 1$,
$b = 2 - e \ge 1$, with equality exactly when $e = 1$: the single run touches
an end, so it is the prefix or the suffix (distinct, since $m \lt n$). For
$r \ge 2$, $b \ge 2r - 2 \ge 2$. ∎

So $P_{C,P_n,\infty}(m) = \tfrac{1}{2}\big(F_C(\text{prefix}) + F_C(\text{suffix})\big)$,
which is `StabilizerCode.end_window_failure_by_size(order)[m]`, computed as
$\big(\mathbf{1}[L_{\text{first}} \le m] + \mathbf{1}[L_{\text{last}} \le m]\big)/2$
from the shortest failing prefix and suffix. It is **not**
`interval_failure_by_size(order, periodic=False)`, the uniform law on all
$n - m + 1$ windows of the open order: interior intervals have $b = 2$.

**Counts by runs and ends** (`erasures.py::path_run_count`). The $n - m$
unerased places fill $r + 1$ gaps: gap 0 before the first run, gap $j$ between
runs $j$ and $j + 1$, gap $r$ after the last run. The $r - 1$ inner gaps are
positive; an end gap is empty exactly when the run next to it touches that end.
With $\binom{m-1}{r-1}$ compositions of the run lengths, the number of regions
of size $1 \le m \le n - 1$ with $r$ runs touching $e$ ends is

$$
e = 0:\ \binom{m-1}{r-1}\binom{n-m-1}{r}, \qquad
e = 1:\ 2\binom{m-1}{r-1}\binom{n-m-1}{r-1}, \qquad
e = 2:\ \binom{m-1}{r-1}\binom{n-m-1}{r-2}\ (r \ge 2),
$$

from the compositions of $n - m$ into $r + 1$, $r$ (either end empty: the
factor 2) or $r - 1$ positive parts, each region arising once. (The empty
region is counted as $(r, e) = (0, 0)$ and the full one as $(1, 2)$, one each.)
The second level, $b = 2$, holds the $n - m - 1$ interior intervals
($r = 1$, $e = 0$) and the $m - 1$ regions split into a prefix and a suffix
($r = 2$, $e = 2$).

### 3.5 Graphs with shortcuts: minimizers need not be contiguous

On a Watts–Strogatz graph (`graphs.watts_strogatz(n, k, beta, seed=...)`: a
ring lattice in which every node is joined to its $k/2$ nearest neighbours on
each side, each edge then rewired with probability $\beta$ to a random new
end), the minimum-boundary
regions need not be intervals of the qubit order, for two reasons. With
$k = 4$ the lattice already joins second neighbours, so at $m = 2$ the pair
$\lbrace i, i+2\rbrace$ ties with $\lbrace i, i+1\rbrace$ (both $b = 6$) even
at $\beta = 0$. And a shortcut can make $b_{\min}$ smaller than the boundary
of every interval, so that no interval is a minimizer at all. The block finds
such a graph and shows its minimizers (`erasures.py::minimum_boundary_regions`,
which enumerates every region of size $m$):

```python
def is_interval(region, n):
    """Whether `region` is a block of consecutive labels on the ring 0..n-1 (wrapping)."""
    members = set(region.tolist())
    return any(all((start + j) % n in members for j in range(len(members))) for start in range(n))


n_ws = 16
for ws_seed in range(1, 41):  # the first graph where a shortcut beats every interval
    ws = graphs.watts_strogatz(n_ws, 4, 0.1, seed=ws_seed)
    shortcut = None
    for m in range(2, n_ws // 2 + 1):
        b_min, minimizers = erasures.minimum_boundary_regions(ws, n_ws, m)
        b_interval = min(erasures.boundary(ws, n_ws, [(s + j) % n_ws for j in range(m)]) for s in range(n_ws))
        if b_min < b_interval:
            shortcut = (m, b_min, b_interval, minimizers)
            break
    if shortcut is not None:
        break
assert shortcut is not None
m_short, b_short, b_interval, minimizers = shortcut
assert not any(is_interval(row, n_ws) for row in minimizers)  # no interval reaches b_min
print(f"Watts-Strogatz seed {ws_seed}, m = {m_short}: b_min = {b_short} < {b_interval} (best interval);")
print("minimum-boundary regions:", minimizers.tolist())

for n in range(3, 13):  # the ring and the open path (3.3, 3.4)
    for m in range(1, n):
        b_min, rows = erasures.minimum_boundary_regions(graphs.cycle(n), n, m)
        intervals = sorted(sorted((s + j) % n for j in range(m)) for s in range(n))
        assert b_min == 2 and rows.tolist() == intervals
        b_min, rows = erasures.minimum_boundary_regions(graphs.path(n), n, m)
        assert b_min == 1 and rows.tolist() == [list(range(m)), list(range(n - m, n))]
```

On such graphs `interval_failure_by_size` is not the $J \to \infty$ limit; the
failing fraction of the minimizers is (§3.7).

### 3.6 $J \to -\infty$

The same proof with $b_{\max} = \max_{A \in \Omega_{n,m}} b(A)$ shows that
$Q_J(\cdot \mid m)$ tends to the uniform law on the maximum-boundary regions,
and $P_{C,G,-\infty}(m) = U_m(b_{\max})/N_m(b_{\max})$. On the ring these are
the regions with $\min(m, n - m)$ runs (for $m = n/2$, the two alternating
patterns). `analysis.exact_correlated_failure` handles negative $J$ by
shifting with $b_{\max}$ (§6.3).

### 3.7 Which function gives each limit exactly

| limit | graph | exact route | function |
|---|---|---|---|
| $J = 0$ | any | uniform counts | `analysis.exact_failure_by_size(code.uncorrectable_counts(w), n)`; `BoundaryCounts.exact(0.0)` (identical) |
| $J \to +\infty$ | ring $C_n$, laid along any `order` | the $n$ windows | `StabilizerCode.interval_failure_by_size(order)` (`periodic=True`) |
| $J \to +\infty$ | open path $P_n$, laid along any `order` | the two end windows | `StabilizerCode.end_window_failure_by_size(order)` |
| $J \to \pm\infty$ | any, sizes $m \le w$ with $\sum_{m \le w}\binom{n}{m}$ within the budget | $U_m(b_{\min})/N_m(b_{\min})$ (or $b_{\max}$) | `code.boundary_failure_counts(edges, w)`, then `failed[m-1, b] / total[m-1, b]` at the smallest (largest) $b$ with `total > 0`, or `.exact(1e4)` (`.exact(-1e4)`), which gives the same numbers exactly |
| $J \to +\infty$ | any, one size $m$ with $\binom{n}{m}$ within the budget | the minimizers listed | `code.region_failures(erasures.minimum_boundary_regions(edges, n, m)[1]).mean()` |

A chain at a very large finite coupling is not a route to the limit: it finds
local minima of $b$ and stays in them (§4.8). The couplings of every sampler
must be finite, and $J = \pm\infty$ is refused with a message pointing to these
routes.

### 3.8 Why exact large-$J$ answers need structure or small size

For $m = n/2$, finding $b_{\min}(G, m)$ is the **minimum bisection** problem,
which is NP-hard for general graphs (the abstract of Feige and Krauthgamer
2002 states it; reference in §11). So no general shortcut to
$\mathcal{M}_{G,m}$ is expected: the simulator gives the exact $J \to \infty$
answer on structured graphs (the ring and the path, in closed form and in
$O(n^2)$ column additions) and on small cases by enumeration within a budget.

---

## 4. The samplers, with proofs

### 4.1 Which sampler runs

`erasures.sample_regions(edges, num_nodes, size, couplings, samples, *, burn_in=None, thin=1, tempering=True, seed=None, method="auto", ladder="auto", max_step=None, max_bytes=MAX_RESULT_BYTES)`
(`erasures.py::sample_regions`) draws `samples` regions of size $m$ = `size`
at each coupling (finite and distinct, any order) and returns a
`RegionSample`: `regions` int64 `[c, S, m]` (nodes sorted in each row),
`boundaries` `[c, S]`, `method`, `geometry`, `ladder`, `max_step`,
`acceptance` `[c]`, `exchange_pairs` `[L-1, 2]`, `exchange_acceptance`
`[L-1]`, `warnings`, `seed`, `burn_in`, `thin` and `tempering`
(`erasures.py::RegionSample`). Per-coupling arrays follow `couplings` as
given; per-rung arrays follow the sorted `ladder`. The `method`:

* `"auto"` (default): `"exact"` when the graph is one cycle or one path
  through all $n$ nodes, in any labeling, and `"chain"` otherwise;
* `"exact"`: independent exact samples (§4.2, §4.3), refused for any other
  graph;
* `"chain"`: the Metropolis swap chain (§4.4) with replica exchange (§4.5), on
  any graph.

`StabilizerCode.correlated_failure(edges, couplings, sizes=None, samples=100, *, burn_in=None, thin=1, tempering=True, seed=None, method="auto", ladder="auto", max_step=None)`
(`codes.py::StabilizerCode.correlated_failure`) calls `sample_regions` once
per size $m$ (default every $m = 1, \dots, n-1$) with the seed `[seed, m]`,
evaluates every drawn region's failure exactly
(`StabilizerCode.region_failures`, §6.4), and returns a `CorrelatedFailure`
(§5.1).

**Geometry detection** (`_kernel/erasure_chain.py::geometry_order`, $O(n + \lvert E\rvert)$).
Kind 1 (cycle) when every node has degree 2 and a walk from node 0 visits all
$n$ nodes before it closes; kind 2 (path) when exactly two nodes have degree
1, all others degree 2, and a walk from the smaller end visits all $n$ (a
single node counts as a path); kind 0 otherwise. It returns the nodes in walk
order. Why this suffices: in a graph of maximum degree 2 every component is a
path or a cycle. If every degree is 2 the graph is a disjoint union of cycles,
and it is one cycle through all nodes exactly when the walk from node 0
covers them. If exactly two nodes have degree 1 and none has degree 0, there
is exactly one path component, and it holds all nodes exactly when the walk
from one of its ends covers them. The exact samplers lay their regions along
this order, so any labeling of a ring or path is sampled exactly.

### 4.2 The exact ring sampler

`_kernel/erasure_chain.py::ring_exact_sample` (also
`erasures.sample_ring_regions(n, m, coupling, samples, *, seed)` on
`graphs.cycle(n)`). For each coupling, in the order passed (`sample_regions`
passes them sorted and un-sorts the rows), and each sample:

1. one `next_double` picks $r$ by inverting the cumulative distribution of
   $P(r) = N(n, m, r)e^{-2Jr}/Z_m(J)$, $r = 1, \dots, \min(m, n-m)$, whose
   weights are computed in log space,
   $\log(n/r) + \log\binom{m-1}{r-1} + \log\binom{n-m-1}{r-1} - 2Jr$ (with
   `lgamma`), shifted by their maximum, the last cumulative entry set to 1;
2. $r - 1$ draws give a uniform composition of $m$ into $r$ positive parts
   (`_composition`: a partial Fisher–Yates shuffle picks a uniform
   $(r-1)$-subset of the $m - 1$ cut places);
3. $r - 1$ draws give a uniform composition of $n - m$ into $r$ positive parts;
4. `next_below(n)` gives the place $p$ of the first run; runs and gaps then
   alternate from $p$ round the cycle (places taken mod $n$ along the order).

The recorded boundary is $2r$. $m = 0$ and $m = n$ draw nothing.

**Claim.** Each sample has exactly the law $Q_J(\cdot \mid m)$, independently
of the others.

*Proof.* Given $r$, the triple (run composition, gap composition, $p$) is
uniform over $n\binom{m-1}{r-1}\binom{n-m-1}{r-1}$ values (stars and bars
makes each composition uniform, and `next_below` is exactly uniform). The
layout maps triples onto regions with $r$ runs, exactly $r$ to 1: a triple
maps to $A$ precisely when $p$ is the first place of one of $A$'s runs and the
compositions are $A$'s run and gap lengths read cyclically from that run;
there are $r$ such runs, and they start at different places, so the $r$
preimages are distinct even for a region with rotational symmetry. Hence
$\Pr[A \mid r] = r / \big(n\binom{m-1}{r-1}\binom{n-m-1}{r-1}\big) = 1/N(n, m, r)$,
uniform, and
$\Pr[A] = P(r)/N(n, m, r) = e^{-2Jr}/Z_m(J) = Q_J(A \mid m)$. Each sample uses
its own fresh draws of the stream, so the samples are independent. ∎
(Exact up to the 53-bit resolution of `next_double` and the rounding of the
cumulative weights.)

### 4.3 The exact path sampler

`_kernel/erasure_chain.py::path_exact_sample` (also
`erasures.sample_path_regions` on `graphs.path(n)`). The categories
$(r, e)$, $r = 1, \dots, \min(m, n - m + 1)$ ($r \le m$ because every run
holds an erased place, $r - 1 \le n - m$ because the inner gaps are positive),
$e = 0, 1, 2$, have log weights
$\log\binom{m-1}{r-1} + \log\binom{n-m-1}{r-e} + [e = 1]\log 2 - J(2r - e)$, and
$-\infty$ where the count is 0. Per sample: one `next_double` picks $(r, e)$;
for $e = 1$, one `next_below(2)` picks the touched end (0: `order[0]`, 1:
`order[n-1]`); $r - 1$ draws give the run lengths; the gap lengths are a
uniform composition of $n - m$ into the gaps that must be positive ($r + 1$
of them for $e = 0$, $r$ for $e = 1$, $r - 1$ for $e = 2$); the layout is
gap 0, run 1, gap 1, ..., run $r$, gap $r$ along the order, and the recorded
boundary is $2r - e$.

*Proof that it is exact.* Given $(r, e)$ and, for $e = 1$, the end, the run
and gap lengths along the path determine the region and are determined by it,
so the region is uniform among the regions of that category, whose number is
`path_run_count(n, m, r, e)` (§3.4; for $e = 1$ the two ends hold equally many,
and the end is drawn uniformly). Then
$\Pr[A] = \text{count}\cdot e^{-Jb}/Z \cdot 1/\text{count} = e^{-J b(A)}/Z_m(J)$. ∎

Exact samples are independent, so `burn_in`, `thin`, `tempering`, `ladder` and
`max_step` do not apply (the first two are recorded and ignored), `acceptance`
and `exchange_acceptance` are NaN, and no warning is ever issued.

### 4.4 The Metropolis swap chain

`_kernel/erasure_chain.py::sample_chain`. Each rung keeps its region as a list
of the erased nodes (`inside`, length $m$), a list of the unerased nodes
(`outside`, length $n - m$), a 0/1 mask and its boundary $b$. A proposal draws
a position `a = next_below(m)` in the erased list and `p = next_below(n - m)` in
the unerased list, so $u$ = `inside[a]` and $v$ = `outside[p]` are uniform in
$A$ and in $\bar A$. It computes $\Delta b$ (`swap_delta`, below). When
$J\Delta b > 0$ it draws one `next_double` and accepts when it is below
$e^{-\lvert J\rvert\lvert\Delta b\rvert}$ (from a table over
$0 \le \lvert\Delta b\rvert \le 2\max\deg + 2$); otherwise it accepts without a
draw. That is exactly $\min(1, e^{-J\Delta b})$ for either sign of $J$: when
$J\Delta b > 0$, $-J\Delta b = -\lvert J\rvert\lvert\Delta b\rvert$. On
acceptance $v$ takes $u$'s place in the erased list and $u$ takes $v$'s place in
the unerased list, so the update is $O(1)$. A **sweep** is $n$ proposals.

**The proofs.**

1. *Symmetric proposal.* The $m(n-m)$ pairs $(u, v) \in A \times \bar A$ give
   $m(n-m)$ distinct regions $A'$ ($u$ is the element of $A \setminus A'$ and
   $v$ that of $A' \setminus A$): the neighbours of $A$ in the Johnson graph
   $J(n, m)$, whose vertices are the $m$-subsets and whose edges join subsets
   differing by one swap. So $q(A, A') = 1/(m(n-m)) = q(A', A)$.
2. *Detailed balance.* For neighbours $A \ne A'$,
   $Q_J(A)P(A, A') = \frac{1}{m(n-m)Z}\min\big(e^{-Jb(A)}, e^{-Jb(A')}\big)$,
   which is symmetric in $A$ and $A'$. Summing over $A$ shows that $Q_J$ is
   stationary.
3. *Irreducibility, whatever $G$ is.* $J(n, m)$ is connected for
   $1 \le m \le n - 1$: if $\lvert A \setminus B\rvert = d$, then $d$ swaps,
   each replacing an element of $A \setminus B$ by one of $B \setminus A$, lead
   from $A$ to $B$, and every step has positive probability at finite $J$. The
   graph $G$ enters only through the acceptance.
4. *Aperiodicity.* For $n \ge 3$, $J(n, m)$ contains a triangle: with
   $u, u' \in A$ and $v \notin A$ (when $m \ge 2$), the regions $A$,
   $A - u + v$, $A - u' + v$ are pairwise neighbours; with $v, v' \notin A$
   (when $m = 1$, so $n - m \ge 2$), $A$, $A - u + v$, $A - u + v'$ are. Closed
   walks of length 2 and 3 give period 1. More: one sweep connects any two
   regions with positive probability. Their distance $d \le \min(m, n-m) \le n/2$
   leaves $n - d \ge 2$ further steps, and any integer $\ge 2$ is a sum of 2s
   (there and back) and 3s (round a triangle).
5. *The corner $n = 2$, $m = 1$.* $J(2, 1)$ is a single edge, and
   $b(\lbrace 0\rbrace) = b(\lbrace 1\rbrace)$ (the edge, if present, is a
   boundary edge of both), so $Q_J$ is uniform for every $J$, every proposal is
   accepted, and the chain alternates with period 2. The law is still exact,
   since the start state is uniform. But a sweep is two proposals, so a lone
   chain records its start state every time: its records are not independent,
   and the effective sample size of that constant series means nothing.
   `method="auto"` samples the two-node path exactly; only `method="chain"` or
   the edgeless two-node graph reach this corner.

Time averages along an irreducible finite chain converge to $Q_J$ whatever its
period (the ergodic theorem); aperiodicity is what makes the law of the state
itself converge, which is what a burn-in relies on.

**The incremental boundary change** (`swap_delta`). With
$\deg_A(x) = \lvert N(x) \cap A\rvert$ and $\deg_{\bar A}(x) = \lvert N(x) \setminus A\rvert$
counted in the current $A$, and $\mathbf{1}[u \sim v]$ for an edge between $u$
and $v$,

$$
\Delta b = \big(\deg_A(u) - \deg_{\bar A}(u)\big) + \big(\deg_{\bar A}(v) - \deg_A(v)\big) + 2\cdot\mathbf{1}[u \sim v].
$$

*Derivation.* Only edges at $u$ or $v$ can change status. An edge
$\lbrace u, w\rbrace$, $w \ne v$, is a boundary edge before the swap exactly
when $w \notin A$, and after it exactly when $w \in A$ ($u$ has left; $w$ has
not moved). Before, there are $\deg_{\bar A}(u) - \mathbf{1}[u \sim v]$ of
them (excluding $v$, which is in $\bar A$); after, $\deg_A(u)$. An edge
$\lbrace v, w\rbrace$, $w \ne u$, is a boundary edge before exactly when
$w \in A$ and after exactly when $w \notin A$: $\deg_A(v) - \mathbf{1}[u \sim v]$
before (excluding $u$), $\deg_{\bar A}(v)$ after. The edge $\lbrace u, v\rbrace$,
when present, is a boundary edge before ($u$ in, $v$ out) and after ($v$ in,
$u$ out), so it changes nothing. Summing the differences gives the formula:
the $+2\cdot\mathbf{1}[u \sim v]$ undoes the two places where the degrees
count the edge $\lbrace u, v\rbrace$. In particular
$\lvert\Delta b\rvert \le \deg u + \deg v \le 2\max\deg$. The block checks the
formula against recomputation for every swap of several regions on a graph
with adjacent and non-adjacent pairs:

```python
graph = graphs.watts_strogatz(12, 4, 0.3, seed=2)
neighbours = [set() for _ in range(12)]
for a, c in graph.tolist():
    neighbours[a].add(c)
    neighbours[c].add(a)


def swap_change(region, u, v):
    """Delta b of the formula, degrees counted in the current region."""
    inside, outside = len(neighbours[u] & region), len(neighbours[u] - region)
    v_inside, v_outside = len(neighbours[v] & region), len(neighbours[v] - region)
    return (inside - outside) + (v_outside - v_inside) + 2 * int(v in neighbours[u])


rng = np.random.default_rng(1)
adjacent_pairs = 0
for m in (1, 3, 6, 11):
    for _ in range(3):
        region = set(rng.choice(12, size=m, replace=False).tolist())
        b_before = erasures.boundary(graph, 12, sorted(region))
        for u in region:
            for v in set(range(12)) - region:
                adjacent_pairs += int(v in neighbours[u])
                after = erasures.boundary(graph, 12, sorted(region - {u} | {v}))
                assert swap_change(region, u, v) == after - b_before
assert adjacent_pairs > 0
```

**Cost.** A proposal costs two `next_below` draws, $O(\deg u + \deg v)$ for
$\Delta b$, at most one `next_double` and an $O(1)$ update. Compiled, one core
runs about $5 \cdot 10^7$ proposals per second (Watts–Strogatz $n = 256$,
$k = 4$, $m = 64$, four rungs; measured on the development machine, an AMD
Ryzen 9 9950X3D, 2026-10-16). Recording a region scans the mask, $O(n)$.

### 4.5 Replica exchange

Replica exchange, or parallel tempering (Swendsen and Wang 1986; Hukushima
and Nemoto 1996; Geyer 1991 is commonly credited too; see §11): with
`tempering` and a ladder of $L \ge 2$ rungs $J_0 \lt J_1 \lt \dots \lt J_{L-1}$,
one chain runs per rung. After every sweep, for $i = 0, \dots, L - 2$ in that
order, the states of rungs $i$ and $i + 1$ are exchanged with probability

$$
\min\Big(1, \exp\big((J_i - J_{i+1})(b_i - b_{i+1})\big)\Big),
$$

with one `next_double` drawn only when the exponent
$x = (J_i - J_{i+1})(b_i - b_{i+1})$ is negative (exchange when it is below
$e^x$). The kernel swaps which state buffer each rung points to; nothing is
copied (`_kernel/erasure_chain.py::sample_chain`).

**Claim.** The product law $\Pi = Q_{J_0} \otimes \dots \otimes Q_{J_{L-1}}$ is
invariant, so in equilibrium the marginal of rung $i$ is exactly $Q_{J_i}$; and
for $n \ge 3$ the joint chain converges to $\Pi$ from any start.

*Proof.* Exchanging rungs $i$ and $i+1$ changes $\Pi$ by the factor
$\frac{e^{-J_i b_{i+1}} e^{-J_{i+1} b_i}}{e^{-J_i b_i} e^{-J_{i+1} b_{i+1}}} = e^{(J_i - J_{i+1})(b_i - b_{i+1})}$.
The exchange proposal is its own inverse, hence symmetric, so the Metropolis
rule satisfies detailed balance with respect to $\Pi$. A sweep moves each
coordinate with a $Q_{J_i}$-invariant kernel, so it leaves $\Pi$ invariant, and
so does any composition of these steps. For convergence: by §4.4 (item 4) the
one-sweep kernel $S$ has every entry positive on each coordinate, hence on the
product space. The exchange kernel $E$ preserves $\Pi$, which has full support,
so $\sum_z \Pi(z)E(z, y) = \Pi(y) > 0$ and every $y$ has some $z$ with
$E(z, y) > 0$. Then $(SE)(x, y) \ge S(x, z)E(z, y) > 0$ for all $x, y$: the
one-sweep-plus-exchanges kernel is positive, hence irreducible and
aperiodic. ∎

Why it helps: the rung $J = 0$ starts exactly in $Q_0$ and decorrelates fast,
and accepted exchanges carry its fresh states up the ladder, so a cold rung
need not escape its local minima by itself. The $J = 0$ swap chain is the
Bernoulli–Laplace diffusion: its transition matrix is the adjacency matrix of
$J(n, m)$ divided by $m(n-m)$, whose second eigenvalue (from the Johnson
scheme, $(m-1)(n-m-1) - 1$ for the adjacency matrix) gives the spectral gap
$n/(m(n-m))$ per proposal, so it relaxes within a fraction of a sweep; at
$m = n/2$ it has cutoff at $\tfrac{1}{8}n\log n$ proposals (Levin, Peres and
Wilmer, Exercise 18.3). Only the requested couplings are recorded; the auxiliary rungs run and
exchange but are never stored, and recording fewer rungs changes no draw.
`RegionSample.exchange_acceptance` holds the accepted fraction per ladder pair;
`CorrelatedFailure.min_exchange_acceptance` the smallest over the whole ladder
of each size.

### 4.6 The automatic ladder

`erasures.coupling_ladder(couplings, max_step)` returns the sorted union of the
couplings and 0, with $k = \lceil (b - a)/\texttt{max\_step}\rceil - 1$ rungs
inserted evenly in every gap $(a, b)$, at $a + (b - a)j/(k + 1)$,
$j = 1, \dots, k$, so that no gap exceeds `max_step`; more than
`MAX_LADDER_INSERTED` (256) insertions in all are refused, with what to change.
With the chain and `tempering=True`, `sample_regions` runs:

* `ladder="auto"` (default): `coupling_ladder(requested, max_step)`, with
  `max_step` defaulting to $1/\sqrt{\operatorname{Var}_0(b)}$ (none inserted
  when $\operatorname{Var}_0(b) = 0$; `RegionSample.max_step` is then None);
* `ladder="given"`: exactly the requested couplings;
* a list: the requested couplings plus those.

Without `tempering` the ladder is the requested couplings (a list is refused).
The cost is proportional to the number of rungs (`RegionSample.ladder`,
`CorrelatedFailure.rungs`).

**The spacing rule.** Suppose that at neighbouring rungs $J$ and $J + \Delta J$
the boundary is roughly Gaussian with a common variance $\sigma^2$ and means
differing by $\Delta J\,\sigma^2$ (Proposition 2 to first order). The exchange
exponent is $x = -\Delta J\,(b_J - b_{J+\Delta J})$, so for independent rungs
in equilibrium $x \sim \mathcal{N}(-a, 2a)$ with $a = \Delta J^2\sigma^2$. For
such an $x$, $\Pr[x \ge 0] = \Phi_{\mathcal N}(-\sqrt{a/2})$ and
$\mathbb{E}[e^x; x \lt 0] = e^{-a + a}\Pr[y \lt 0] = \Phi_{\mathcal N}(-\sqrt{a/2})$
with $y \sim \mathcal{N}(a, 2a)$, so the mean acceptance is

$$
\mathbb{E}\big[\min(1, e^{x})\big] = 2\,\Phi_{\mathcal N}\big(-\Delta J\,\sigma/\sqrt{2}\big) = \operatorname{erfc}\big(\Delta J\,\sigma/2\big),
$$

about 0.48 at $\Delta J\,\sigma = 1$ and falling like $e^{-\Delta J^2\sigma^2/4}$.
Hence the rule $\Delta J \lesssim 1/\sigma_J$. This is a Gaussian heuristic, not
a theorem; the default takes $\sigma = \sqrt{\operatorname{Var}_0(b)}$, the
uniform law's, which is exact and cheap. $\operatorname{Var}_J(b)$ is usually
smaller at $J > 0$, so the spacing is then conservative, but near a transition
it can be far larger: for two 9-cliques joined by one edge and $m = 9$,
$\operatorname{Var}_0(b) = 10.7$ but $\operatorname{Var}_{0.3}(b) = 176$. The
exchange warning (§4.9) is the safeguard, and a smaller `max_step` the fix.

**The uniform moments in closed form** (`erasures.uniform_boundary_moments`).
Write $b = \sum_{e \in E} X_e$ with $X_e = \mathbf{1}[e \text{ is cut}]$ (one end
in $A$), and let $D = \sum_v d_v(d_v - 1)$, the number of ordered pairs of
distinct edges sharing a node (in a simple graph two distinct edges share at
most one node). For $A$ uniform among the $m$-subsets, the probability that
given distinct nodes are in or out of $A$ is a ratio of falling factorials:
$k$ given nodes in and $j$ out has probability $m^{\underline k}(n-m)^{\underline j}/n^{\underline{k+j}}$.
Then

$$
\begin{aligned}
P_2 &= \Pr[X_e = 1] = \frac{2m(n-m)}{n(n-1)}, \qquad \mathbb{E}_0[b] = \lvert E\rvert P_2, \\
P_3 &= \Pr[X_e = X_f = 1],\ e, f \text{ sharing a node} = \frac{m(n-m)(n-m-1) + (n-m)m(m-1)}{n(n-1)(n-2)}, \\
P_4 &= \Pr[X_e = X_f = 1],\ e, f \text{ disjoint} = \frac{4m(m-1)(n-m)(n-m-1)}{n(n-1)(n-2)(n-3)}, \\
\mathbb{E}_0[b^2] &= \mathbb{E}_0[b] + D\,P_3 + \big(\lvert E\rvert(\lvert E\rvert - 1) - D\big)P_4, \qquad \operatorname{Var}_0(b) = \mathbb{E}_0[b^2] - \mathbb{E}_0[b]^2 .
\end{aligned}
$$

*Derivation.* $\mathbb{E}_0[b^2] = \sum_{e,f}\mathbb{E}[X_eX_f]$ over ordered pairs.
$e = f$ gives $\mathbb{E}_0[b]$. An edge is cut when one end is in and the
other out: two orders, so $P_2 = 2m(n-m)/n^{\underline 2}$. Edges
$\lbrace u, v\rbrace$, $\lbrace v, w\rbrace$ sharing $v$ are both cut exactly
when $v$ is in and $u, w$ out, or $v$ out and $u, w$ in: $P_3$. Two disjoint
edges are both cut when each has exactly one end in $A$: 4 assignments of 2 in
and 2 out among their 4 nodes, $P_4$. There are $D$ ordered sharing pairs and
$\lvert E\rvert(\lvert E\rvert - 1) - D$ disjoint ones. ∎ The code computes it in
exact rational arithmetic and skips a term with no pairs (so $n \lt 4$, $m = 0$
and $m = n$ need no special case). On the ring, $\mathbb{E}_0[b] = 2m(n-m)/(n-1)$.

The block checks the closed form against enumeration, shows the automatic
ladder and its exchange rates against $\operatorname{erfc}(\Delta J\sigma_0/2)$,
and reproduces the two-clique failure of the default spacing and its fix:

```python
graph = graphs.watts_strogatz(12, 4, 0.2, seed=1)
for m in range(13):
    exact = erasures.boundary_moments(erasures.boundary_counts(graph, 12, m), 0.0)
    mean0, var0 = erasures.uniform_boundary_moments(graph, 12, m)
    assert math.isclose(exact["mean"][0], mean0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(exact["variance"][0], var0, rel_tol=1e-9, abs_tol=1e-9)

ws64 = graphs.watts_strogatz(64, 4, 0.1, seed=1)
mean0, var0 = erasures.uniform_boundary_moments(ws64, 64, 16)
drawn = erasures.sample_regions(ws64, 64, 16, [0.5, 1.0, 2.0], 500, seed=1)
assert drawn.method == "chain" and drawn.max_step == 1 / math.sqrt(var0)
assert np.array_equal(drawn.ladder, erasures.coupling_ladder([0.5, 1.0, 2.0], 1 / math.sqrt(var0)))
assert drawn.burn_in == erasures.default_burn_in(64) == 512 and drawn.warnings == ()
first_gap = drawn.ladder[1] - drawn.ladder[0]
predicted = math.erfc(first_gap * math.sqrt(var0) / 2)  # the Gaussian heuristic at J = 0
assert abs(drawn.exchange_acceptance[0] - predicted) < 0.1
assert drawn.exchange_acceptance.min() > erasures.EXCHANGE_WARNING
print(f"{len(drawn.ladder)} rungs, max_step {drawn.max_step:.3f}; exchange rates",
      np.round(drawn.exchange_acceptance, 2).tolist(), f"(heuristic at J = 0: {predicted:.2f})")

pairs = list(itertools.combinations(range(9), 2))
cliques = np.array(pairs + [(a + 9, c + 9) for a, c in pairs] + [(8, 9)])  # two K_9 and one bridge
moments = erasures.boundary_moments(erasures.boundary_counts(cliques, 18, 9), [0.0, 0.3])
print("two cliques, m = 9: Var_0 =", round(moments["variance"][0], 1), " Var_0.3 =", round(moments["variance"][1], 1))
coarse = erasures.sample_regions(cliques, 18, 9, [0.0, 0.3, 0.6], 2000, seed=1)
assert any("replica exchange between J = 0 and J = 0.3" in w for w in coarse.warnings)
finer = erasures.sample_regions(cliques, 18, 9, [0.0, 0.3, 0.6], 2000, seed=1, max_step=0.05)
assert finer.warnings == () and finer.exchange_acceptance.min() > erasures.EXCHANGE_WARNING
```

### 4.7 Start state, burn-in, thinning, sweeps

* **Start.** Every rung, recorded or not, starts from a uniformly random
  $m$-subset: $m$ draws `next_below(n - t)`, $t = 0, \dots, m-1$, a partial
  Fisher–Yates shuffle of $0, \dots, n-1$; the first $m$ entries are the erased
  list and the rest the unerased list. This is an exact sample of $Q_0$, so a
  lone chain at $J = 0$ is in equilibrium from the start.
* **Sweeps and records.** In a sweep every rung, in increasing $J$, makes $n$
  proposals; then the exchanges run. After `burn_in` sweeps the requested rungs
  are recorded, then again every `thin` sweeps, `samples` records in all:
  `burn_in + (samples - 1) * thin` sweeps. Recording draws nothing. $m = 0$ and
  $m = n$ record the single region `samples` times and draw nothing
  (acceptance NaN).
* **The default burn-in**, `burn_in=None`, is
  `erasures.default_burn_in(n)` $= \max(200, 8n)$ sweeps (`MIN_BURN_IN`,
  `BURN_IN_PER_NODE`). Every rung starts from a uniform region, and the cold
  rungs need a number of sweeps that grows with $n$ to settle. Measured during
  development on Watts–Strogatz graphs ($k = 4$, $\beta = 0.1$, graph seed 1,
  $m = n/4$, couplings 0, 0.5, 1, 2, the automatic ladder, 1,000 records per
  seed): the drift is the mean over 16 seeds of (mean of the first 200
  records − mean of the last 500) of $b$ at $J = 2$, two seed sets per cell.

  | $n$ | burn-in 200 | burn-in $4n$ | burn-in $8n$ |
  |---|---|---|---|
  | 160 | 0.56 ± 0.31, 1.07 ± 0.31 | 0.10 ± 0.17, 0.31 ± 0.20 | |
  | 192 | 1.16 ± 0.25, 1.47 ± 0.29 | 0.76 ± 0.23, 0.47 ± 0.17 | −0.01 ± 0.20, 0.15 ± 0.22 |
  | 224 | 1.60 ± 0.33, 1.71 ± 0.24 | 0.12 ± 0.24, −0.08 ± 0.15 | |
  | 256 | 1.98 ± 0.41, 2.35 ± 0.31 | 0.74 ± 0.32, 0.21 ± 0.22 | −0.14 ± 0.35, 0.06 ± 0.40 |

  At $n = 256$ the mean $b$ at $J = 2$ came out 24.49 (200 sweeps), 23.39
  ($4n$), and 23.06 and 22.99 ($8n$), against long runs of 23.07 ± 0.27
  (4,096 sweeps) and 22.84 ± 0.43 (8,192 sweeps): 200 sweeps leave the mean
  6% high, with no warning. So $4n$ removes most of the bias but leaves a
  measurable drift at $n = 192$ and 256; $8n$ leaves none at either. The
  default must still be checked on a new graph or size: a second run with a
  longer burn-in that agrees, and several seeds.
* **Cost** of one call: $L \cdot n \cdot (\texttt{burn\_in} + (S-1)\,\texttt{thin})$
  proposals for $L$ rungs. For the same Watts–Strogatz family at
  $n = 4096$, $m = n/4$ and couplings 0 to 2, the automatic ladder has 73
  rungs, and the default burn-in alone is about $10^{10}$ proposals: about
  3 minutes per call on one core.
* **Memory.** Only the $c$ requested couplings are stored: $8cS(m+1)$ bytes,
  refused above `max_bytes` (default `MAX_RESULT_BYTES`, 2 GiB) before
  anything runs.

### 4.8 Slow modes, and why diagnostics of $b$ alone mislead

**The ring at large $J$.** Once the region is a single interval ($b = 2$),
for $2 \le m \le n - 2$ exactly two of the $m(n-m)$ proposals keep $b = 2$:
removing the first place and adding the place after the last (a shift by +1),
or the reverse (a shift by −1). Every other swap leaves two runs ($b \ge 4$)
and is accepted with probability at most $e^{-2J}$. So at $J = \infty$ the
interval performs a random walk on $\mathbb{Z}_n$ with steps $\pm 1$, each with
probability $1/(m(n-m))$ per proposal; its spectral gap is
$2(1 - \cos(2\pi/n))/(m(n-m))$, and its relaxation time

$$
t_{\mathrm{rel}} = \frac{m(n-m)}{2\big(1 - \cos(2\pi/n)\big)} \text{ proposals} \approx \frac{m(n-m)\,n}{4\pi^2} \text{ sweeps}
$$

is about $1.2 \cdot 10^3$ sweeps at $n = 64$, $m = 16$ and $8 \cdot 10^4$ at
$n = 256$, $m = 64$. Meanwhile every record has $b = 2$: the boundary series is
constant, so `chain_diagnostics` reports $\tau = 1$, an effective sample size
equal to the number of records and an R-hat of NaN, which looks perfect and
says nothing. The position is the slow variable, and the membership of fixed
nodes follows it: `erasures.region_diagnostics` (§5.3).

**The open path at large $J$.** The two minimum-boundary states (prefix and
suffix, $b = 1$) are not one swap apart when $n - m \ge 2$, and every swap from
the prefix raises $b$ by at least 1, so it is accepted with probability at most
$e^{-J}$: a lone chain stays at an end for at least $e^{J}$ proposals on
average, while $Q_J$ puts nearly half its mass on each end.

**General graphs.** Kawasaki-type coarsening (§2): on a Watts–Strogatz graph
with $n = 256$, $k = 4$, $\beta = 0.1$, $m = 64$, a lone chain at $J = 2$ still
drifts after 200,000 sweeps, and the four rungs 0, 0.5, 1, 2 with 200 sweeps of
burn-in hardly ever exchange and overestimate $\langle b\rangle$ at $J = 2$ by
15–30%; 41 rungs $0, 0.05, \dots, 2$ exchange about 70% of the time and give the
right value (measured during development; `erasures.py::sample_regions`
docstring).

This is why `method="auto"` samples cycles and paths exactly (no chain, no
slow mode), and why the chain runs an automatic ladder by default. The block
shows the ring case: four lone chains at $J = 8$ (no exchange) whose boundary
diagnostics look perfect while the positional R-hat exposes them, and the
exact sampler on the same ring; then the open path:

```python
n, m = 64, 16
ring = graphs.cycle(n)
lone = [erasures.sample_regions(ring, n, m, [8.0], 400, burn_in=500, seed=s, method="chain", tempering=False)
        for s in range(4)]
on_b = erasures.chain_diagnostics(np.stack([r.boundaries[0] for r in lone]))
assert on_b["constant"] and on_b["tau_int"] == 1.0 and math.isnan(on_b["rhat"])  # "perfect", and empty
where = erasures.region_diagnostics(np.stack([r.regions[0] for r in lone]), n)
assert where["rhat_max"] > 1.5 and where["tau_int_max"] > 3  # the regions have not moved
print("lone chains at J = 8: positional R-hat", round(where["rhat_max"], 2), "frozen nodes", where["frozen"])

exact = [erasures.sample_regions(ring, n, m, [8.0], 400, seed=s) for s in range(4)]
assert all(r.method == "exact" and r.geometry == "cycle" for r in exact)
again = erasures.region_diagnostics(np.stack([r.regions[0] for r in exact]), n)
assert again["rhat_max"] < 1.1 and again["frozen"] == []

path12 = graphs.path(12)
stuck = erasures.sample_regions(path12, 12, 4, [8.0], 400, burn_in=500, seed=2, method="chain", tempering=False)
left = np.mean(stuck.regions[0, :, 0] == 0)  # records that are the prefix {0, 1, 2, 3}
both = erasures.sample_regions(path12, 12, 4, [8.0], 400, seed=2)  # exact: both ends
assert left in (0.0, 1.0) and 0.4 < np.mean(both.regions[0, :, 0] == 0) < 0.6
```

### 4.9 Warnings, and why no warning is no proof

A chain run returns plain sentences in `RegionSample.warnings` (each says what
to change; exact samples never warn; Python's `warnings` module is not used):

* an exchange acceptance below `EXCHANGE_WARNING` = 0.1 between two rungs
  ("add couplings between them", or "pass a smaller max_step" when the ladder
  was automatic);
* a swap acceptance below `ACCEPTANCE_WARNING` = 0.01 at a requested coupling
  that no exchange pair with acceptance $\ge 0.1$ reaches;
* from `CORRELATION_MIN_RECORDS` = 50 records on, an integrated
  autocorrelation time of the boundary or of the slowest followed node's
  membership (`region_diagnostics`) above $\max(3,\ 0.1 \times \text{records})$
  (`CORRELATION_WARNING`).

`CorrelatedFailure.warnings` merges the sampler's warnings over sizes, one line
per kind with the sizes it occurred at; `size_warnings` keeps every
`(m, sentence)` pair.

**No warning is no proof of mixing.** The warnings are heuristics from one
run. A bias left by too short a burn-in (the 6% of §4.7 came with no warning),
or a relaxation longer than the run (a lone chain cannot see a slow mode it
never left), can pass all of them. The checks that settle it are a second,
longer run that agrees, and several seeds compared by R-hat (§5.3), on
positions as well as on $b$.

### 4.10 Randomness and reproducibility

The only source of randomness is the xoshiro256** state `seed_state(seed)`
(`_kernel/rng.py`), drawn in the documented orders of `sample_chain` (start
states of every rung, then per sweep each rung's $n$ proposals in increasing
coupling order, then the exchanges), `ring_exact_sample` and
`path_exact_sample` (coupling by coupling in increasing order, sample by
sample). The draw orders are contracts, pinned by digests and by a
plain-Python transcription in `tests/test_erasures.py`. The same arguments and
seed give the same result on every platform; `seed=None` draws a fresh seed
and reports it. `StabilizerCode.correlated_failure` draws size $m$ with the
sequence seed `[seed, m]` (a list seed's entries followed by $m$), so one seed
drives every size and any size can be reproduced alone with
`erasures.sample_regions(..., seed=[seed, m])`.

---

## 5. Estimators and their errors

### 5.1 One code

For one code, size $m$ and coupling $J$, the estimate is the failing fraction
of the $S$ recorded regions,

$$
\hat P_{C,G,J}(m) = \frac{1}{S}\sum_{s=1}^{S} F_C(A_s)
$$

(`CorrelatedFailure.failure`, with the integer `failures`).

* **Exact samples** ($A_s$ iid from $Q_J$): $S\hat P \sim \mathrm{Bin}(S, P_{C,G,J}(m))$,
  unbiased, standard error $\sqrt{P(1-P)/S}$.
* **The chain**: consistent ($\hat P \to P$ as $S \to \infty$, by the ergodic
  theorem) and biased only by incomplete equilibration:

  $$
  \big\lvert \mathbb{E}\hat P - P_{C,G,J}(m)\big\rvert \le \frac{1}{S}\sum_{s}\big\lVert \mathcal{L}(A_s) - Q_J\big\rVert_{\mathrm{TV}} \le \big\lVert \mu_0 K^{\texttt{burn\_in}} - \Pi\big\rVert_{\mathrm{TV}},
  $$

  because $F_C \in \lbrace 0, 1\rbrace$ (so
  $\lvert\mathbb{E}F_C(A_s) - P\rvert \le \lVert\mathcal{L}(A_s) - Q_J\rVert_{\mathrm{TV}}$),
  a marginal is no farther from its target than the joint law, and the
  one-sweep kernel $K$ (sweep and exchanges) never increases the
  total-variation distance to its invariant law, while every record comes at
  least `burn_in` sweeps after the start ($\mu_0$: the joint start law, $\Pi$
  the product law of §4.5). A lone chain at $J = 0$ starts in $Q_0$
  and is exactly unbiased; with replica exchange the $J = 0$ rung also
  receives states from the other rungs, so its records are exact only once
  the joint chain has equilibrated (`codes.py::StabilizerCode.correlated_failure`).
* **Correlated records.** Within one chain $S\hat P$ is not binomial.
  `CorrelatedFailure.failure_ess` gives the effective number of independent
  records: $S$ for exact samples, and for the chain $S/\tau_{\mathrm{int}}$ of
  the series of failure indicators (`chain_diagnostics`; $S$ for a constant
  series and for $k = 0$). The standard error is
  $\sqrt{\hat P(1-\hat P)/\mathrm{ESS}}$, an estimate from the run itself (too
  small when the chain has not mixed). With no failing record (or all
  failing) it gives 0: read such an entry as "below about $3/\mathrm{ESS}$"
  (or above $1 - 3/\mathrm{ESS}$), the rule of three, exact for independent
  samples.
* **Across sizes and couplings.** Each size runs its own sampler on its own
  stream (`[seed, m]`), so the estimates at different sizes are independent
  given the code (unlike the nested orderings of the uniform estimator in
  [random-region-erasure.md](random-region-erasure.md) §2.4). The couplings
  of one size share one ladder run and are correlated through the exchanges.

### 5.2 An ensemble of codes

For codes $C_1, \dots, C_T$ drawn independently from the ensemble, each with
its own independent region stream, the per-code estimates $\hat P_t$ are iid.
Their mean estimates $\mathbb{E}_C[P_{C,G,J}(m)]$ (up to the common burn-in
bias), and the sample standard deviation over $t$ divided by $\sqrt{T}$
(`analysis.mean_and_error`, `analysis.mean_over_codes`) is a valid standard
error whatever the correlation inside each chain:
$\operatorname{Var}\hat P_t = \operatorname{Var}_C(P_C) + \mathbb{E}_C\operatorname{Var}(\hat P_t \mid C)$
is estimated as a whole by the spread. Two conditions:

1. **Independent streams per code.** The in-run probe has them: every
   trajectory, evaluation and size draws from its own stream (§7.4). With
   `correlated_failure` on saved codes, pass a different seed per code (for
   example `seed=[base, t]`). The same seed draws the same regions for every
   code (common random numbers); the spread over codes then misses the variance
   of that one region sample.
2. **Condition on a code.** A code with $k = 0$ fails nowhere, at every $J$.
   `analysis.mean_over_codes(values, holds_code)` averages only the rows with
   $k > 0$ and returns `(mean, std_error, count)`.

Codes that share a graph (a graph collection with several circuits per graph)
share its disorder: average per graph first, as in
[random-region-erasure.md](random-region-erasure.md) §4.3.

### 5.3 Diagnostics, with formulas

`erasures.chain_diagnostics(series)`: `series` is one chain (1-D, at least 2
draws) or independent chains of equal length (chains × draws, at least 4
draws each).

* **Autocorrelation.** Each chain is centred on its own mean (a chain with no
  variation, decided exactly, is centred to zeros), and
  $\hat C(t) = \frac{1}{N}\sum_{s=0}^{N-1-t}\tilde x_s\tilde x_{s+t}$ (by FFT,
  zero-padded) is averaged over chains; $\hat\rho(t) = \hat C(t)/\hat C(0)$.
* **Integrated autocorrelation time, Sokal's window.**
  $\hat\tau(W) = 1 + 2\sum_{t=1}^{W}\hat\rho(t)$, cut at the smallest $W$ with
  $W \ge c\,\hat\tau(W)$, $c$ = `SOKAL_WINDOW` = 5 (or $W = N - 1$ when there is
  none); $\tau_{\mathrm{int}} = \hat\tau(W)$, floored at
  $1/\max(1, \log_{10}\text{total})$ (an effective size at most
  $\text{total}\cdot\max(1, \log_{10}\text{total})$, as Stan caps it). This is emcee's
  normalization: $\tau = 1$ for independent draws and $(1+\rho)/(1-\rho)$ for
  an AR(1) chain. Sokal's own $\tau_{\mathrm{int}} = \tfrac12 + \sum_{t \ge 1}\rho(t)$
  is half of it, so $c = 5$ here is $c = 10$ in Sokal's convention, the
  conservative end of the range commonly used. The automatic window
  underestimates $\tau$ when the correlations decay slowly (Li, Madras and
  Sokal 1995, Appendix C), and an estimate is trustworthy only from runs much
  longer than $\tau$ (emcee: at least about $50\tau$).
* **Effective sample size and error.** $\mathrm{ESS} = \text{total}/\tau_{\mathrm{int}}$,
  `std_error` $= \sqrt{\mathrm{variance}\cdot\tau_{\mathrm{int}}/\text{total}}$,
  `variance` the variance of all draws ($1/\text{total}$).
* **Constant series.** When every draw is equal, `constant` is True, `mean` is
  that value, `variance` and `std_error` are 0, $\tau_{\mathrm{int}} = 1$ and
  ESS = total. That is not evidence of mixing (§4.8): a stuck chain looks the
  same.
* **Rank-normalized split R-hat** (2-D input; Vehtari, Gelman, Simpson,
  Carpenter and Bürkner 2021). Each chain is split into halves (the middle draw
  dropped when $N$ is odd). The draws are replaced by normal scores
  $z = \Phi_{\mathcal N}^{-1}\big((r - 3/8)/(S + 1/4)\big)$ of their average
  ranks $r$ among all $S$ split draws, and with $N'$ draws per half,
  $\hat R = \sqrt{\big(\tfrac{N'-1}{N'}W + B/N'\big)/W}$, $W$ the mean
  within-half variance and $B/N'$ the variance of the half means. The same is
  done for the folded draws $\lvert x - \operatorname{median}\rvert$, and the
  larger value is reported. NaN when every draw is equal; $\infty$ when each
  half is constant but the halves differ.
* **Positional diagnostics** (`erasures.region_diagnostics(regions, n, *, vertices=None)`):
  for followed nodes $v$ (default the 8 nodes $\lfloor kn/8\rfloor$,
  $k = 0, \dots, 7$; every node when $n \lt 8$), the series
  $\mathbf{1}[v \in A_s]$ through `chain_diagnostics`. It returns `vertices`,
  `mean` (the marginals, $m/n$ at $J = 0$ on any graph and on vertex-transitive
  graphs at any $J$), `tau_int`, `ess`, `changes`, `slowest`, `tau_int_max`,
  `ess_min`, `frozen` (nodes whose membership never changed: the run may be too
  short to see the region move) and, for stacked chains, `rhat` and
  `rhat_max`.
* **Acceptances.** `RegionSample.acceptance` (swaps, per requested coupling)
  and `exchange_acceptance` (per ladder pair).

Thresholds and what to do: Vehtari et al. recommend $\hat R \lt 1.01$, at least
four chains, and a rank-normalized ESS above 400, and rely on $\hat R$ only
when each split chain has an ESS of at least 50 (the older Gelman–Rubin
threshold was 1.1). When a check fails: record more and burn in longer; run
several seeds of the same graph, size and couplings and compare them by R-hat,
on $b$ and on positions; lower `max_step` when an exchange rate is below 0.1;
use the exact samplers on cycles and paths. The chain does not depend on the
code, so diagnostics from `sample_regions` carry over to a code evaluation
with the same graph, size, couplings and settings.

The block checks the conventions on an AR(1) chain, whose
$\tau = (1+\rho)/(1-\rho)$ is known, and shows R-hat separating agreeing and
disagreeing chains:

```python
rng = np.random.default_rng(0)
rho, draws = 0.8, 50_000
noise = rng.normal(size=(4, draws))
chains = np.empty_like(noise)
chains[:, 0] = noise[:, 0] / math.sqrt(1 - rho**2)  # start in the stationary law
for t in range(1, draws):
    chains[:, t] = rho * chains[:, t - 1] + noise[:, t]
d = erasures.chain_diagnostics(chains)
assert abs(d["tau_int"] / ((1 + rho) / (1 - rho)) - 1) < 0.1  # tau = 9, emcee's normalization
assert abs(d["ess"] - d["chains"] * d["draws"] / d["tau_int"]) < 1e-6 and d["rhat"] < 1.01
shifted = chains.copy()
shifted[0] += 2 * chains.std()  # one chain sits elsewhere
assert erasures.chain_diagnostics(shifted)["rhat"] > 1.2
```

---

## 6. Exact answers

### 6.1 Boundary-resolved counts

For a code $C$ on the graph $G$, define

$$
N_m(b) = \#\lbrace A : \lvert A\rvert = m,\ b(A) = b\rbrace, \qquad
U_m(b) = \#\lbrace A : \lvert A\rvert = m,\ b(A) = b,\ F_C(A) = 1\rbrace .
$$

$N_m$ depends on the graph only; $U_m$ on the graph and the code.

**Theorem.** For every finite $J$,

$$
P_{C,G,J}(m) = \frac{\sum_b U_m(b)\, e^{-Jb}}{\sum_b N_m(b)\, e^{-Jb}} .
$$

*Proof.* Group both sums of $P_{C,G,J}(m) = \sum_A e^{-Jb(A)}F_C(A) / \sum_A e^{-Jb(A)}$
by the value of $b(A)$. ∎

One enumeration therefore gives $P_{C,G,J}(m)$ exactly for every coupling at
once, with the identities

* $J = 0$: $\sum_b U_m(b)/\binom{n}{m} = P_C(m)$, and $\sum_b U_m(b)$ is the
  uniform count `StabilizerCode.uncorrectable_counts(w)[m-1]`;
* $J \to +\infty$: $U_m(b_{\min})/N_m(b_{\min})$; $J \to -\infty$:
  $U_m(b_{\max})/N_m(b_{\max})$ (§3.2, §3.6);
* **the slope in $J$:** differentiating the ratio,

  $$
  \frac{\partial}{\partial J}P_{C,G,J}(m) = -\Big(\langle F_C\, b\rangle_J - P_{C,G,J}(m)\langle b\rangle_J\Big) = -\operatorname{Cov}_J(F_C, b),
  \qquad \langle F_C\, b\rangle_J = \frac{\sum_b b\, U_m(b)e^{-Jb}}{\sum_b N_m(b)e^{-Jb}} .
  $$

  $P_J(m)$ grows with $J$ exactly where the failing regions have a smaller
  boundary than the average region. Nothing fixes the sign of the covariance,
  and it can change with $J$, so $P_J(m)$ need not be monotone in $J$ (the
  block of §6.7 checks the formula by finite differences).

### 6.2 The enumeration

`StabilizerCode.boundary_failure_counts(edges, max_size, *, budget=None)`
(`codes.py`; kernel `_kernel/erasure_failure.py::boundary_failure_counts`)
returns a `BoundaryCounts` with `failed[m-1, b]` $= U_m(b)$ and
`total[m-1, b]` $= N_m(b)$ for $m = 1, \dots, w$ (`max_size`), int64, and
`.exact(couplings)`, the probabilities of §6.1 as float64 `(c, w)`.

* **Every subset of size $1 \dots w$ exactly once**, depth first in
  lexicographic order of the sorted qubits: the node for
  $\lbrace a_1 \lt \dots \lt a_j\rbrace$ extends the node of its first $j - 1$
  qubits.
* **Boundaries incrementally:**
  $b(P \cup \lbrace v\rbrace) = b(P) + \deg v - 2\lvert N(v) \cap P\rvert$, since
  the $\lvert N(v) \cap P\rvert$ edges from $v$ into $P$ stop being boundary
  edges and the other $\deg v - \lvert N(v) \cap P\rvert$ edges at $v$ become
  boundary edges.
* **Failures incrementally:** two GF(2) bases on the code's adapted columns
  (`_kernel/entropy.py::code_columns`), one of the S-parts and one of the full
  N(S) columns, each node adding its last qubit's two columns;
  $\ell_C(P) = \operatorname{rank}_N - \operatorname{rank}_S$ (form (ii) of
  [random-region-erasure.md](random-region-erasure.md) §2.2). Backing up
  truncates the bases (stack truncation), so nothing is copied.
* **Monotone skip:** once a prefix fails, every region below it in the tree
  fails (§6.7), so its rank work is skipped; its boundary never is. With
  $k = 0$ nothing fails and no rank is computed.
* **Budget:** $\sum_{m=1}^{w}\binom{n}{m}$ may not exceed `budget` (default
  `erasures.ENUMERATION_BUDGET`, two million); a larger count is refused with
  the number and the largest `max_size` that fits
  (`analysis.exhaustive_cutoff`).
* **Columns:** `codes.boundary_columns(edges, n, w)` $= \min\big(\lvert E\rvert, \text{sum of the } w \text{ largest degrees}\big) + 1$,
  because $b(A) \le \sum_{v \in A}\deg v$ and $b(A) \le \lvert E\rvert$. The
  kernel refuses smaller arrays (`ValueError`, in compiled code too), and
  `BoundaryCounts` trims the columns to the largest boundary that occurs, so
  `total[m-1]` is `erasures.boundary_counts(edges, n, m)` padded with zeros.

The graph-side counterparts: `erasures.boundary_counts(edges, n, m, *, budget=None)`
($N_m(b)$ for $b = 0, \dots, b_{\max}(m)$, trimmed so the last entry is
positive; $\binom{n}{m}$ within the budget) and
`erasures.boundary_moments(counts, couplings)` ($\log Z_m(J)$, $\langle b\rangle_J$,
$\operatorname{Var}_J(b)$, each sum shifted by its largest term).

### 6.3 The stable evaluation

`analysis.exact_correlated_failure(failed, total, couplings)` (what
`BoundaryCounts.exact` calls) multiplies numerator and denominator by
$e^{J b_{\mathrm{lo}}}$ when $J \ge 0$, $b_{\mathrm{lo}}$ the smallest $b$ with
$N_m(b) > 0$, and by $e^{J b_{\mathrm{hi}}}$ when $J \lt 0$, $b_{\mathrm{hi}}$ the
largest:

$$
P_{C,G,J}(m) = \frac{\sum_b U_m(b)\, e^{-J(b - b_\ast)}}{\sum_b N_m(b)\, e^{-J(b - b_\ast)}}, \qquad b_\ast = \begin{cases} b_{\mathrm{lo}}, & J \ge 0 \\ b_{\mathrm{hi}}, & J \lt 0 \end{cases}.
$$

Every exponent is then $\le 0$, and the term at $b_\ast$ contributes
$N_m(b_\ast) \ge 1$ to the denominator, so nothing overflows and nothing
divides by a vanishing number, at any finite $J$; boundaries with $N_m(b) = 0$
get weight 0. At $J = \pm 10^4$ every other term underflows to 0 and the limits
of §3 come out exactly. The counts are converted to float64 (exact below
$2^{53}$); leading axes broadcast (one `total` serves every code on the same
graph) and boundary axes of different lengths are padded with zeros. At
$J = 0$ the result equals `analysis.exact_failure_by_size` of the summed counts
bit for bit.

### 6.4 One region: failures, and the complement for large regions

`StabilizerCode.region_failures(regions)` (kernel
`_kernel/erasure_failure.py::region_failures`) returns $F_C(A)$ for each row of
an `(R, m)` array. $I(A:R)$ is `_kernel/entropy.py::lost_information_rows`, the
rank of $A$'s columns over all tableau rows minus their rank over the unmarked
rows, two eliminations of $2m$ columns. A region of more than $n/2$ qubits is
evaluated on its complement, by

$$
I(A:R) + I(\bar A:R) = 2k .
$$

*Proof.* The system and $R$ together are pure, so $S(AR) = S(\bar A)$ and
$S(\bar AR) = S(A)$, and $S(R) = k$. Hence $I(A:R) = S(A) + k - S(\bar A)$ and
$I(\bar A:R) = S(\bar A) + k - S(A)$, which sum to $2k$. (For a code this is
the complementarity of the cleaning lemma; [lost-information.md](lost-information.md).) ∎

So $F_C(A) = 1$ exactly when $I(\bar A:R) \lt 2k$, and every elimination has at
most $2\min(m, n - m)$ columns. Before any region, the kernel checks the `k`
it is given by one elimination of all system columns
($I(\text{system}:R) = 2k$); with $k = 0$ every output is then 0 with no
further elimination.

### 6.5 The $J \to \infty$ limits through windows

`StabilizerCode.window_first_failure(order=None, *, periodic=True)` (kernel
`window_first_failure`) returns, for every start $s$, the length $L(s)$ of the
shortest failing window `order[s], ..., order[s+L-1]` (read cyclically when
`periodic`, stopping at the end of the order otherwise), or the largest
possible length plus one ($n + 1$ on the ring, $n - s + 1$ on the open order)
when none fails. Each start grows its window one qubit at a time: while the
S-rank of the window is full ($2L$) the window is correctable, because
$\operatorname{rank}_N \ge \operatorname{rank}_S$ (the S-columns are
truncations of the N(S)-columns) and $\operatorname{rank}_N \le 2L$; at the first
deficient length the N(S) basis of the window is built and kept up, and the
scan stops at the first $L$ with $\operatorname{rank}_N > \operatorname{rank}_S$.
That is $O(n)$ column additions per start.

Windows from one start are nested and $F_C$ is monotone (§6.7), so the window
of length $m$ at $s$ fails exactly when $L(s) \le m$. Hence:

* `interval_failure_by_size(order)[m]` $= \#\lbrace s : L(s) \le m\rbrace / n$
  on the ring, the $J \to \infty$ limit of §3.3;
* `end_window_failure_by_size(order)[m]` $= \big(\mathbf{1}[L_{\text{first}} \le m] + \mathbf{1}[L_{\text{last}} \le m]\big)/2$,
  with the shortest failing prefix and suffix found by the erasure profile
  along the order and its reverse (`_kernel/entropy.py::erasure_profile_rows`):
  the open path's limit of §3.4;
* entry 0 of both is 0 and entry $n$ is $\mathbf{1}[k > 0]$.

### 6.6 $\Phi_{C,G,J}(\varepsilon)$, $\varepsilon_\delta(J)$ and full curves

**Full curves.** `analysis.correlated_failure_curve(sampled, sizes, n, *, exact=None, holds_code=None)`
returns `(curve, interpolated)`: `curve` of shape `(..., c, n + 1)` and a
boolean mask of length $n + 1$. Each entry comes from the first source that
has it:

1. $P(0) = 0$; $P(n) = \mathbf{1}[k > 0]$, from `holds_code` (booleans or the
   $k$) when given, else from `exact` when it covers $m = n$, else 1 (codes
   already conditioned on $k > 0$);
2. the exact values for $1 \le m \le \min(w, n-1)$ (`exact`, shape `(..., c, w)`);
3. the sampled values (`sampled`, shape `(..., c, s)`, at the distinct `sizes`);
4. linear interpolation between the nearest known sizes,
   $P(m) = P(m_\ell) + t\big(P(m_r) - P(m_\ell)\big)$,
   $t = (m - m_\ell)/(m_r - m_\ell)$ (exactly $P(m_\ell)$ on a plateau), with
   `interpolated[m] = True`.

Interpolated entries are a modelling choice, not measurements: report which
sizes they are. NaN propagates.

**The channel.** `analysis.failure_probability_from_curve(curve, n, e)` computes

$$
\Phi_{C,G,J}(\varepsilon) = \sum_{m=0}^{n}\binom{n}{m}\varepsilon^m(1-\varepsilon)^{n-m}P_{C,G,J}(m)
$$

with the binomial weights in log space (`lgamma`; $\varepsilon = 0$ and 1
exactly), a few million weights at a time, so any $n$ is safe. It is linear in
$P$, so the mean curve over codes gives the mean $\Phi$. At $J = 0$ it equals
`analysis.erasure_failure_probability` given the same exact $P_C(m)$ (to
rounding; §9(c)).

**The tolerance.** `analysis.tolerance_from_curve(curve, n, delta, *, return_monotone=False)`
returns $\varepsilon_\delta = \sup\lbrace\varepsilon \in [0, 1] : \Phi(\varepsilon) \le \delta\rbrace$:
$\Phi$ on a grid of 2,049 points (2,048 steps), the last grid point with
$\Phi \le \delta$, then 50 bisection steps between it and the next point. It
returns 1.0 when $\Phi(1) = P(n) \le \delta$; for $\delta = 0$, 0.0 unless
$P \equiv 0$ (then 1.0); NaN for a curve with a NaN entry; $P(0)$ must be 0.
When $P$ is not monotone it is the supremum up to excursions narrower than a
grid step. With `return_monotone=True` it also returns whether each curve is
nondecreasing in $m$ up to `MONOTONE_SLACK` ($10^{-12}$); where that is False,
$\Phi$ may cross $\delta$ more than once and $\varepsilon_\delta$ is the last
crossing. The tolerance of the mean curve is the tolerance of the
ensemble-mean $\Phi$, not the mean of per-code tolerances.

*Why monotone curves give monotone $\Phi$.* Couple all $\varepsilon$ through
$M_\varepsilon = \sum_i \mathbf{1}[\xi_i \le \varepsilon]$ with $\xi_i$ iid
uniform on $[0, 1]$: $M_\varepsilon \sim \mathrm{Bin}(n, \varepsilon)$ and is
nondecreasing in $\varepsilon$ pointwise. If $P$ is nondecreasing, so is
$\Phi(\varepsilon) = \mathbb{E}P(M_\varepsilon)$, and
$\lbrace\varepsilon : \Phi \le \delta\rbrace = [0, \varepsilon_\delta]$.

**The binomial window.** `analysis.binomial_window(n, e, tol=1e-12)` returns
$(m_{\mathrm{lo}}, m_{\mathrm{hi}})$ with
$\Pr[M \lt m_{\mathrm{lo}}] \le \mathrm{tol}/2$ and
$\Pr[M > m_{\mathrm{hi}}] \le \mathrm{tol}/2$ for $M \sim \mathrm{Bin}(n, e)$,
$m_{\mathrm{lo}}$ as large and $m_{\mathrm{hi}}$ as small as that allows (for an
array of $e$, the union; $e = 0$ gives $(0, 0)$, $e = 1$ gives $(n, n)$,
$\mathrm{tol} = 0$ gives $(0, n)$). Since $0 \le P \le 1$,

$$
\Big\lvert \Phi(\varepsilon) - \sum_{m = m_{\mathrm{lo}}}^{m_{\mathrm{hi}}}\binom{n}{m}\varepsilon^m(1-\varepsilon)^{n-m}P(m)\Big\rvert \le \Pr[M \lt m_{\mathrm{lo}}] + \Pr[M > m_{\mathrm{hi}}] \le \mathrm{tol} :
$$

the sizes outside the window change $\Phi$ by at most `tol`, so they need not
be sampled at that $\varepsilon$.

### 6.7 Monotonicity in $m$, and where it fails

* **$F_C$ is monotone under inclusion:** $A \subseteq B$ implies
  $I(A:R) \le I(B:R)$ (data processing; [random-region-erasure.md](random-region-erasure.md) §2.3).
* **$P_0(m)$ is nondecreasing in $m$:** removing a uniformly random element
  from a uniform $(m+1)$-subset $B$ gives a uniform $m$-subset $A$ (each
  $m$-subset arises from $n - m$ supersets, each removal with probability
  $1/(m+1)$, so $\Pr[A] = (n-m)/\big((m+1)\binom{n}{m+1}\big) = 1/\binom{n}{m}$),
  and $F_C(A) \le F_C(B)$.
* **The ring and path limits are nondecreasing:** $\#\lbrace s : L(s) \le m\rbrace$
  grows with $m$, and so do the two end indicators.
* **For $J \ne 0$ on a general graph, $P_J(m)$ need not be monotone in $m$.**
  The coupling above needs the favoured regions of size $m + 1$ to contain
  those of size $m$, and under $Q_J$ they need not. The block builds a code
  and a graph where $P_\infty(2) = 1$ but $P_\infty(3) = 0$: the $[[4,2,2]]$
  code on qubits 0, 1, 2, 5 (every pair of them fails, no single one does) and
  frozen qubits 3, 4, 6, 7, on the graph made of the edge $\lbrace 0, 1\rbrace$
  and the triangles $\lbrace 2, 3, 4\rbrace$, $\lbrace 5, 6, 7\rbrace$. The only
  region of size 2 with $b = 0$ is $\lbrace 0, 1\rbrace$, which fails; the
  only ones of size 3 with $b = 0$ are the triangles, each holding one code
  qubit, which do not. The tolerance then reports `monotone=False`. A
  natural Watts–Strogatz case is in §9(d).

```python
from sparsegf2 import StabilizerCode, analysis

toy = StabilizerCode(["XXX__X__", "ZZZ__Z__", "___Z____", "____Z___", "______Z_", "_______Z"])
assert toy.num_logical == 2
toy_graph = np.array([(0, 1), (2, 3), (3, 4), (2, 4), (5, 6), (6, 7), (5, 7)])
toy_counts = toy.boundary_failure_counts(toy_graph, 8)  # every region, m = 1..8
couplings = [0.0, 2.0, 1e4]
p_toy = toy_counts.exact(couplings)  # (3, 8): P_J(m), m = 1..8
assert p_toy[2, 1] == 1.0 and p_toy[2, 2] == 0.0  # P_inf(2) = 1, P_inf(3) = 0
assert np.all(np.diff(p_toy[0]) >= 0)  # the uniform curve is monotone
toy_curve, _ = analysis.correlated_failure_curve(np.zeros((3, 0)), [], 8, exact=p_toy, holds_code=2)
eps, monotone = analysis.tolerance_from_curve(toy_curve, 8, 0.1, return_monotone=True)
assert monotone.tolist() == [True, False, False]
print("P_J(m), m = 1..4, at J = 0, 2, inf:", np.round(p_toy[:, :4], 3).tolist())

# the slope in J is minus the covariance of F and b (6.1), by central differences
u, t = toy_counts.failed.astype(float), toy_counts.total.astype(float)
b_axis = np.arange(t.shape[1])
for coupling in (-0.7, 0.4, 1.5):
    weight = np.exp(-coupling * b_axis)
    z = t @ weight
    p, mean_b, fb = (u @ weight) / z, (t @ (b_axis * weight)) / z, (u @ (b_axis * weight)) / z
    h = 1e-5
    slope = (toy_counts.exact(coupling + h)[0] - toy_counts.exact(coupling - h)[0]) / (2 * h)
    assert np.allclose(slope, -(fb - p * mean_b), atol=1e-7)

# stable at extreme couplings, and the binomial window's guarantee
assert np.all(np.isfinite(toy_counts.exact([-1e4, 1e4])))
lo, hi = analysis.binomial_window(256, 0.1, 1e-9)
pmf = np.array([math.comb(256, m) * 0.1**m * 0.9 ** (256 - m) for m in range(257)])
assert pmf[:lo].sum() <= 0.5e-9 and pmf[hi + 1 :].sum() <= 0.5e-9
print("binomial window at n = 256, e = 0.1, tol = 1e-9:", (lo, hi))
```

---

## 7. From the kernels to the code you run

### 7.1 The dispatch table

| compiled kernel (`_kernel/...`) | Python | JSON service, CLI and MCP |
|---|---|---|
| `erasure_chain.py::sample_chain`, `ring_exact_sample`, `path_exact_sample`, `geometry_order` | `erasures.sample_regions` (`RegionSample`), `sample_ring_regions`, `sample_path_regions` | `api.sample_erasure_regions`, `sparsegf2 regions`, MCP tool `sample_erasure_regions` (§7.5) |
| `erasure_chain.py::enumerate_boundaries` | `erasures.boundary_counts`, `minimum_boundary_regions`; `boundary_moments` and `uniform_boundary_moments` in numpy and exact rationals | the exact values `sample_erasure_regions` compares with |
| `erasure_failure.py::region_failures` | `StabilizerCode.region_failures`, `StabilizerCode.correlated_failure` (`CorrelatedFailure`) | (saved codes, Python only) |
| `erasure_failure.py::boundary_failure_counts`, `boundary_columns` | `StabilizerCode.boundary_failure_counts` (`BoundaryCounts`), `codes.boundary_columns` | (saved codes, Python only) |
| `erasure_failure.py::window_first_failure`; `entropy.py::erasure_profile_rows` | `StabilizerCode.window_first_failure`, `interval_failure_by_size`; `end_window_failure_by_size` | (saved codes, Python only) |
| `ensemble.py::correlated_erasure_rows` (the exact samplers or `sample_chain`, then `region_failures`) | the probe `mipt.correlated_erasure`; `EnsembleResult.correlated` | `{"kind": "correlated_erasure", ...}` in `run_ensemble`, `estimate_ensemble`, `start_ensemble_job`; `sparsegf2 ensemble`; the MCP ensemble tools (§7.2, §7.3) |
| (numpy) | `analysis.exact_correlated_failure`, `correlated_failure_curve`, `failure_probability_from_curve`, `tolerance_from_curve`, `binomial_window`, `mean_over_codes` | `api.erasure_curves`, `sparsegf2 curves`, MCP tool `erasure_curves` (§7.5) |

The ensemble engine calls the graph-side kernels themselves, with the same
ladders, so a trajectory's regions are exactly those that
`erasures.sample_regions` would draw from the same random state (§7.4), and
the graph-side tools check the very sampler the probe runs.

### 7.2 The probe

```text
mipt.correlated_erasure(couplings, sizes=None, samples=100, name=None, *, burn_in=None,
                        thin=1, tempering=True, method="auto", ladder="auto",
                        max_step=None, geometry="interaction", checkpoints=None) -> Probe
```

At each of its evaluations, for every coupling $J$ in `couplings` and size $m$
in `sizes`, the probe draws `samples` regions $A_s$ of size $m$ from
$Q_J(\cdot \mid m)$ on the erasure graph and records the failing fraction,
the estimate $\hat P_{C,G,J}(m)$ of §5.1 for the code the trajectory holds
there (`mipt.py::correlated_erasure`):

* `couplings`: one finite number or a list of distinct ones, any order
  ($J = \pm\infty$ is refused, with the exact routes of §3.7);
* `sizes`: distinct sizes in $[1, n-1]$, any order, as a list or a range
  string `"a:b[:step]"` with Python slice semantics (`"2:15:2"` is 2, 4, ..., 14),
  or None (default) for every $m = 1, \dots, n-1$, filled in for the spec's $n$;
* `samples`, `burn_in` (None: `erasures.default_burn_in(n)` $= \max(200, 8n)$
  sweeps), `thin`, `tempering`, `method`, `ladder`, `max_step`: exactly those of
  `erasures.sample_regions` (§4), so a cycle or path is sampled exactly and
  any other graph by the swap chain with its automatic ladder;
* `geometry`, the erasure graph $G$: `"interaction"` (default; the trajectory's
  own interaction graph, the study's $G$; for a graph collection, each
  trajectory's own graph), or `"cycle"` / `"path"` over the qubit labels
  $0, \dots, n-1$ (the physical ring or chain a Watts–Strogatz circuit is built
  on; `"cycle"` needs $n \ge 3$);
* `name` and `checkpoints` as for every probe. It needs a reference.

In JSON the same arguments are keys of the probe object:

```json
{"kind": "correlated_erasure", "couplings": [0, 1, 2], "sizes": "8:57:8", "samples": 100,
 "burn_in": null, "thin": 1, "tempering": true, "method": "auto", "ladder": "auto",
 "max_step": null, "geometry": "interaction", "name": "correlated_erasure",
 "checkpoints": "final"}
```

`ladder` is `"auto"`, `"given"` or a list of extra couplings; every key but
`kind` and `couplings` may be left out. The spec is checked when it is built:
sizes outside $[1, n-1]$, `method="exact"` on a graph that is neither a cycle
nor a path, a list `ladder` without tempering, a ladder of too many rungs, and
a working memory of more than 2 GiB per thread (one size's regions,
$8cS(m+1)$ bytes) are refused there, naming `probes[i]` in JSON; `run` also
refuses more than `mipt.MAX_WORKING_BYTES` (8 GiB) over all running threads.
`MonitoredCircuit.correlated_sampler(name)` says before a run how the probe
will draw: `method` (`"exact"`, `"chain"`, or `"mixed"` for a
collection with both), `graphs`, `exact_graphs`, `geometries`, and for the
chain `rungs` (the fewest and most of any ladder), `max_step`, `burn_in`,
`thin` and `tempering`.

**What it does with $k = 0$.** A code without logical qubits never fails: the
probe records 0 and draws nothing, also for the evaluations after a
`stop_when_purified` stop.

### 7.3 The value layout

`EnsembleResult.values[name]` is float64 of shape
`(trajectories, evaluations, len(couplings) * len(sizes))`, coupling-major:
column `c * len(sizes) + i` is `couplings[c]` at `sizes[i]`, both in the order
given. `EnsembleResult.correlated(name=None)` returns
`(couplings, sizes, values)` with `values` the same array as
`(trajectories, evaluations, c, s)`:

$$
\texttt{values}[t, e, c, i] = \frac{1}{S}\sum_{s=1}^{S} F_C(A_s), \qquad A_1, \dots, A_S \text{ drawn at } J = \texttt{couplings}[c],\ m = \texttt{sizes}[i],
$$

for trajectory row $t$ at evaluation $e$ (the layers are
`EnsembleResult.layers(name)`). `name` may be left out when the spec has one
such probe. The JSON summary of `run_ensemble` and `load_results` shows, at
the last evaluation where some trajectory holds a code: `what`, `at_layer`,
`count`, `count_with_code` (the trajectories with $k > 0$, when a
`code_dimension` or `erasure_first_failure` probe covers the layers; without
one the mean is over every trajectory, $k = 0$ counting as 0, and a note says
so), `couplings` and `sizes` (at most 12 couplings and 17 sizes shown, evenly
spaced with the ends kept; fewer when the summary must shrink, with a note),
`mean` and `stderr` (`mean[i][j]` is `couplings[i]` at `sizes[j]`), `sampler`
(one line: the exact samplers, or the chain's rungs, burn-in and thinning) and
`vs_depth` (`layers`, `count_with_code`, `sizes`, and `mean[i][d][j]` at
`couplings[i]`, `layers[d]`, `sizes[j]`). For a chain, a note says to check
its mixing with `sample_erasure_regions` before a large run: the probe stores
only the failure fractions, so the summary cannot see a chain that has not
mixed (§8.4). `estimate_ensemble` prices the probe and reports
`working_megabytes`, the regions the running threads hold at once.

### 7.4 The random streams, and what a saved run pins

Under randomness contract 2 (`mipt.RANDOMNESS_CONTRACT`), each size of each
evaluation draws from its own stream. With `key_probe` the run's probe key
(words 2 and 3 of the run's `SeedSequence`), $h$ the 64-bit hash of the probe's
name (the first 8 bytes of `blake2b(name, digest_size=8)`, little-endian), $t$
the trajectory index and $L$ the layer, size $m$ draws from

$$
\texttt{stream\_state}\big(\texttt{key\_probe},\ \mathrm{mix}(\mathrm{mix}(t, h, L),\ m,\ 0)\big),
\qquad \mathrm{mix}(a, b, c) = \mathrm{sm}(\mathrm{sm}(\mathrm{sm}(a) \oplus b) \oplus c),
$$

with `sm` splitmix64 (`_kernel/rng.py::mix`; `_kernel/ensemble.py`, module
docstring). **Draw order** (`_kernel/ensemble.py::correlated_erasure_rows`):
for each size in the order given, the stream is set as above, and then exactly
the draws of `erasures.sample_regions` follow: the exact ring or path sampler
with the couplings in increasing order (§4.2, §4.3), or the swap chain on the
size's ladder, every rung started afresh from a uniform region with a full
burn-in (§4.4–§4.7, §4.10). Failures are then evaluated without drawing. With
$k = 0$ nothing is drawn. Consequences:

* a size's values depend only on (seed, trajectory, probe name, layer, size,
  the couplings and sampler settings, the erasure graph and the state), never
  on the other sizes requested, other probes, or earlier evaluations;
* `EnsembleResult.replay(t, probes=[...], checkpoints=[L])` reproduces them,
  also for a probe with a subset of the sizes in another order (the couplings
  may be in any order too);
* nothing is carried from one evaluation to the next (no warm start), which
  is what makes the values reproducible one by one, and is why every
  evaluation pays a full burn-in (§8).

**What a saved run pins.** The draws also depend on what the spec resolved
for its $n$: the default sizes and burn-in, the exact-or-chain class of each
erasure graph, and every replica-exchange ladder (which depend on
`erasures.default_burn_in`, `coupling_ladder` and $\operatorname{Var}_0(b)$).
A saved result stores them (`EnsembleResult.load` keeps the stored sizes,
burn-in, classes and ladders), so a replay of a loaded run reproduces it even
if those rules change in a later version, and a resume or merge under rules
that now resolve otherwise is refused as another protocol. The pinned digest
in `tests/test_randomness_contract.py` freezes the rules of this version.

### 7.5 The two read-only tools

Both are in the JSON service (`api.py`), the CLI and the MCP server, and
neither writes anything.

**`sample_erasure_regions`** checks the sampler on a graph, with no circuit:

```text
api.sample_erasure_regions(graph, size, couplings, samples=200, *, num_nodes=None,
                           burn_in=None, thin=1, tempering=True, chains=4, seed=None,
                           method="auto", ladder="auto", max_step=None, max_seconds=60)
sparsegf2 regions --family watts_strogatz --n 64 --k 4 --beta 0.1 --graph-seed 1 \
    --size 16 --couplings 0 1 2 --samples 400 --chains 8 --seed 7 [--json]
```

`graph` is a family (`{"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": 1}`
with `num_nodes`), a graph file (`{"family": "file", "path": ...}`, its first
graph) or `{"edges": [[a, b], ...], "num_nodes": n}`. It runs `chains`
independent calls of `erasures.sample_regions`, chain $i$ seeded `[seed, i]`,
and returns per coupling (`couplings`, in the order given): `mean_boundary`
with its `stderr` (from the effective sample size), `variance`, `acceptance`,
`exchange_acceptance`, `tau_int`, `ess` and `r_hat` (the slowest of the
boundary and of the membership of a few nodes, named by `slowest`: the
positional diagnostics of §5.3), `min_boundary_seen`, and `exact`
(`mean_boundary`, `variance`) with `deviation_se` where an exact value exists:
the ring or path closed form, the enumeration of every region when
$\binom{n}{m} \le 10^8$, or the uniform law at $J = 0$ (`exact_source` says
which). At the top level: `graph` (with its `geometry`), `method`, the
`ladder` (`rungs`, `lowest`, `highest`, `max_step`), `burn_in`, `thin`,
`tempering`, `seed`, `nonincreasing_in_J` (Proposition 2 on the estimates,
within 3 standard errors), `mixing` (one note per kind of problem, saying what
to change: R-hat above 1.01, an effective sample size below 400, an exchange
acceptance below 5%, a mean more than 5 standard errors from its exact
value), `warnings` (the sampler's own), three `examples` of regions at the
largest coupling, `estimated_seconds`, `elapsed_seconds` and `notes`. An
`r_hat` of 1000 stands for infinity (chains stuck apart). It is refused when
its estimated time exceeds `max_seconds` (the MCP tool keeps the default
60 s). An empty `mixing` list is no proof of equilibrium (§4.9).

**`erasure_curves`** turns a saved run into channels:

```text
api.erasure_curves(path, e=None, delta=0.01, layer=None)
sparsegf2 curves runs/erasure --eps 0.02 0.05 0.1 --delta 0.01 [--layer L] [--json]
```

For each `erasure_first_failure` probe (`uniform`: the independent channel,
exact for $m \le w$ where an `uncorrectable_region_counts` probe was evaluated
at the same layer) and each `correlated_erasure` probe (`correlated`, one row
per coupling), at `layer` (default: each probe's last evaluation), over the
trajectories that hold a code: `count`, `count_with_code`, `q_plus` (their
fraction), `phi` (`mean` and `stderr` of $\Phi(e)$ at each `e`; default eight
values log-spaced from $1/n$ to $1/2$), and `e_delta`: `of_mean_phi`, the
tolerance of the ensemble-mean $\Phi$ (the ensemble's tolerance), and
`per_code_mean`, `per_code_median`, `per_code_stderr` of each code's own
tolerance (they differ: $\Phi$ is linear in $P$, the tolerance is not), with
`monotone_fraction` for the correlated curves (the fraction of codes whose
curve is monotone in $m$, §6.7). A correlated entry also lists
`sampled_sizes`, `interpolated_sizes` and `interpolated_mass` (the binomial
probability the interpolated sizes carry at each $e$; `complete` when it stays
below 0.1%), `binomial_window` (the sizes carrying all but $10^{-6}$ of the
probability at the given $e$: the ones worth sampling), and `sampler`. The
per-code curves are `analysis.correlated_failure_curve` of the stored
fractions (P(0) = 0, P(n) = 1, linear interpolation between sampled sizes),
then `failure_probability_from_curve` and `tolerance_from_curve` (§6.6), so
the tool's numbers are exactly those functions' (§9(e) checks it). Exact
small-$m$ curves need the saved codes (`StabilizerCode.boundary_failure_counts`).

A call of the first on the Watts–Strogatz graph of §3.5 (fast at $n = 16$;
§9(e) calls the second on a saved run):

```python
from sparsegf2 import api

check = api.sample_erasure_regions(
    {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": ws_seed}, 8, [0, 1, 2],
    num_nodes=16, seed=1,
)
assert check["method"] == "chain" and check["exact_source"].startswith("enumeration")
assert check["nonincreasing_in_J"]
for row in check["couplings"]:
    assert abs(row["deviation_se"]) < 5  # the pooled chains against the exact <b>_J
print(f"{check['ladder']['rungs']} rungs; R-hat per J:", [row["r_hat"] for row in check["couplings"]])
print("mixing notes:", check["mixing"])
```

## 8. Cost and accuracy

### 8.1 What an evaluation costs

Per evaluation of a trajectory that holds a code, and per size $m$, the probe
pays

* **the regions:** for the chain,
  $L \cdot \big(\texttt{burn\_in} + (S - 1)\,\texttt{thin}\big) \cdot n$ swap
  proposals for an $L$-rung ladder, at about 20 ns each at mean degree 4 and
  30 ns at degree 8 (one thread); for the exact samplers, $c \cdot S$ samples
  of a few microseconds each;
* **the failures:** $c \cdot S$ regions, each two eliminations of
  $2\min(m, n - m)$ columns, plus one elimination of all $2n$ system columns
  per size (the kernel's check of $k$).

Measured on one thread of the development machine (AMD Ryzen 9 9950X3D,
2026-10-16): one size $m = n/4$, couplings 0, 1, 2, `samples=100`, default
burn-in, purification picture after $2n$ layers at $p = 0.05$; the time with
the probe minus the time without, per evaluation, against the probe's share of
`estimate_ensemble` (`api.py::_correlated_us`):

| erasure graph | $n$ | sampler | measured | estimate |
|---|---|---|---|---|
| Watts–Strogatz, $k = 4$, $\beta = 0.1$ | 64 | chain, 11 rungs, burn-in 512 | 9.0 ms | 29 ms |
| | 128 | chain, 15 rungs, burn-in 1024 | 45 ms | 141 ms |
| | 256 | chain, 19 rungs, burn-in 2048 | 211 ms | 671 ms |
| ring (`cycle`, brickwork circuit) | 64 | exact | 1.1 ms | 4.5 ms |
| | 128 | exact | 3.6 ms | 13 ms |
| | 256 | exact | 11 ms | 48 ms |

For the chain the burn-in dominates: at $n = 256$ it is 19 rungs × 2,048
sweeps × 256 proposals, about $10^7$ proposals, against 19 × 99 × 256 for the
100 records. The default burn-in grows as $8n$ and the rungs about as
$\sqrt{\operatorname{Var}_0(b)} \propto \sqrt{n}$, so one size costs about
$n^{2.5}$ proposals; the exact samplers cost little, and their evaluation is
dominated by the failures. The estimate is pessimistic by design, by 3.1 to
3.2 times for the chain and 3.7 to 4.2 times for the exact samplers here, so
`run_ensemble`'s 60-second guard sends long runs to `start_ensemble_job`.

**Every size.** The default `sizes=None` is every $m = 1, \dots, n-1$, and
each size runs its own sampler with a full burn-in (§7.4). That costs about
$n - 1$ times one size. Measured as above (Watts–Strogatz, couplings 0, 1, 2,
`samples=100`, one evaluation, one thread):

| $n$ | every size $1, \dots, n-1$ | every fourth size of the binomial window for $e \in [0.02, 0.2]$ |
|---|---|---|
| 64 | 0.52 s (estimate 1.8 s) | sizes `"1:31:4"`, 0.06 s (estimate 0.25 s) |
| 256 | 52 s (estimate 164 s) | sizes `"1:85:4"`, 3.4 s (estimate 10.5 s) |

(`analysis.binomial_window(n, [0.02, 0.2], 1e-6)` is $(0, 30)$ at $n = 64$ and
$(0, 84)$ at $n = 256$.) Multiply by trajectories and evaluations and divide
by the threads; `estimate_ensemble` does it for a spec.

### 8.2 Choosing sizes, samples and couplings

* **Sizes.** $\Phi_J(\varepsilon)$ needs $P_J(m)$ only where the binomial
  weights are not negligible: `analysis.binomial_window(n, e, tol)` gives the
  window for the rates $e$ you report (an array of rates gives the union), and
  the sizes outside it change $\Phi$ by at most `tol` (§6.6). Sample a stride
  inside the window (a range string such as `"8:57:4"`) and let
  `correlated_failure_curve` interpolate between, then read
  `interpolated_mass` in `erasure_curves`: where it is not small, $\Phi$ leans
  on the interpolation, which can miss a non-monotone step for $J \ne 0$
  (§9(d)). Small sizes are where sampling fails: a failure probability of
  order $10^{-3}$ needs thousands of samples, and the exact counts of §6 on the
  saved codes (`code_snapshot`, then `boundary_failure_counts`) do better.
* **Samples.** One trajectory's estimate has variance about
  $P(1-P)\,\tau/S$ (§5.1); the ensemble mean's squared standard error is
  $\big(\operatorname{Var}_C P_C + \mathbb{E}_C[P(1-P)\tau/S]\big)/T$ over $T$ codes,
  so more samples help only while the second term dominates. With the chain,
  records are cheap next to the burn-in (100 records against $8n$ sweeps of
  burn-in), so several hundred samples cost little more than 100; with the
  exact samplers the cost is proportional to the samples.
* **`burn_in` and `thin`.** Keep the default burn-in unless a graph-side
  check (§8.3) shows that a shorter one gives the same values; raising
  `thin` lowers the correlation of the records at a cost proportional to it.
* **Couplings and the ladder.** The automatic ladder fills $[\min(J, 0), \max(J, 0)]$
  with rungs at most $1/\sqrt{\operatorname{Var}_0(b)}$ apart, so the cost grows
  with the span of the couplings (about
  $\text{span}\cdot\sqrt{\operatorname{Var}_0(b)}$ rungs) and hardly with their
  number: couplings inside the span are rungs the ladder runs anyway. A large
  $J$ is expensive, and beyond the crossover where minimum-boundary regions
  dominate it adds little: use the exact $J \to \infty$ routes of §3.7 for the
  limit. On a cycle or path (a brickwork circuit, or `geometry="cycle"`) the
  exact samplers cost the same at every $J$.
* **Checkpoints.** Give the probe its own sparse `checkpoints` (`"final"`, or a
  few depths): every evaluation pays the full burn-in again.

### 8.3 Check the mixing before a large run

The chain does not depend on the code, only on the erasure graph, the size,
the couplings and the sampler settings. So its mixing can be checked on the
graph alone, before the run, with exactly the probe's settings:
`sample_erasure_regions` (or `erasures.sample_regions` in Python, with
`region_diagnostics` and `chain_diagnostics`) at a few sizes of the window,
with several chains and seeds. Look at `mixing`, `r_hat` (on the boundary and
on positions), `deviation_se` where an exact value exists, and
`exchange_acceptance`; then repeat with a longer burn-in and another seed and
compare the means. For a graph collection, check a few of its graphs (graph
$i$ has seed `[seed, i]`). The block checks two seeds of a Watts–Strogatz
graph at $n = 64$:

```python
for seed in (1, 2):
    check = api.sample_erasure_regions(
        {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": 1}, 16, [0, 1, 2],
        num_nodes=64, samples=300, seed=seed,
    )
    means = [row["mean_boundary"] for row in check["couplings"]]
    exact0 = check["couplings"][0]["exact"]["mean_boundary"]  # J = 0: the uniform law
    assert check["burn_in"] == 512 and abs(means[0] - exact0) < 5 * check["couplings"][0]["stderr"]
    print(f"seed {seed}: mean b per J {[round(x, 2) for x in means]};", check["mixing"] or "no notes")
```

On this graph the check is not clean at $J = 2$: the four chains of 300
records disagree a little about which nodes their regions hold (an R-hat
above 1.01 on a node's membership), and the effective sample size there is of
the order of 100. The two seeds' mean boundaries at $J = 2$ agree to a few
tenths, so the bias is small, but the positions are the slow variable (§4.8).
Measured with other settings (not run here): `thin=10, samples=500` raised the
effective sample size at $J = 2$ to about 200–400 and brought the R-hat to
1.01–1.02, and a burn-in of 2,048 sweeps instead of 512 did not by itself. The
probe's estimates at $J = 2$ are then noisier per trajectory than `samples`
suggests (the ensemble error bar includes that), and a run that relies on
$J = 2$ should repeat the check with its own settings and compare the probe's
values for two burn-ins on a few trajectories.

### 8.4 What the probe cannot tell you

The probe stores only failure fractions: no boundaries, acceptances or
autocorrelations. Its summary and `erasure_curves` can therefore not warn
about a chain that has not mixed; they say only which sampler ran and point to
the graph-side check. The standard error across trajectories is valid for the
mean of the chains' estimates whatever the correlation inside a chain (each
trajectory has its own streams, §5.2), but it does not include a bias left by
incomplete mixing, which only the check of §8.3 can show. On rings and paths
(exact samples) there is nothing to check.

---

## 9. Worked examples

Each example runs in a fraction of a second once the kernels are compiled,
and checks the simulator against an independent brute force. Which circuit
trajectory an example picks, and its numbers, depend on the version's random
streams, so printed values are not quoted here; the assertions are what the
examples establish.

### (a) The ring's and the path's closed forms, and the exact samplers

The laws of $b$ and the partition functions of §1.7 and §3.4 against direct
sums over every region, for $n = 7, 10, 12$, every $m$ and four couplings; the
path's counts by runs and ends against brute force for $n = 2, \dots, 12$; and
the exact samplers, on a cycle and a path with scrambled labels, against the
exact law of every region. For independent samples the total-variation
distance between the empirical and the exact law satisfies
$\mathbb{E}\,\mathrm{TV} \le \tfrac12\sum_A\sqrt{q_A(1-q_A)/S}$, and changing one
sample moves it by at most $1/S$, so by McDiarmid's inequality it exceeds that
plus $\sqrt{\ln(10^6)/(2S)}$ with probability at most $10^{-6}$:

```python
from collections import Counter

couplings = [-1.3, 0.0, 0.4, 2.0]
for n in (7, 10, 12):
    for edges, law, log_partition in (
        (graphs.cycle(n), erasures.ring_boundary_distribution, erasures.ring_partition_function),
        (graphs.path(n), erasures.path_boundary_distribution, erasures.path_partition_function),
    ):
        for m in range(1, n):
            b = boundaries(edges, n, all_regions(n, m))
            for coupling in couplings:
                weights = np.exp(-coupling * b)
                values, probabilities = law(n, m, coupling)
                assert values.tolist() == sorted(set(b.tolist()))
                brute = np.array([weights[b == v].sum() for v in values]) / weights.sum()
                assert np.allclose(probabilities, brute, rtol=1e-10, atol=0)
                assert math.isclose(log_partition(n, m, coupling), math.log(weights.sum()),
                                    rel_tol=1e-12, abs_tol=1e-12)


def path_runs_and_ends(n, regions):
    """(r, e) of each region on the path 0-1-...-(n-1): runs, and how many ends they contain."""
    member = np.zeros((len(regions), n), dtype=bool)
    np.put_along_axis(member, regions, True, axis=1)
    starts = member.copy()
    starts[:, 1:] &= ~member[:, :-1]  # a run starts where the previous place is unerased
    return np.count_nonzero(starts, axis=1), member[:, 0].astype(int) + member[:, -1]


for n in range(2, 13):
    for m in range(n + 1):
        regions = all_regions(n, m)
        r, e = path_runs_and_ends(n, regions)
        if 0 < m < n:
            assert np.array_equal(boundaries(graphs.path(n), n, regions), 2 * r - e)  # b = 2r - e
        tally = Counter(zip(r.tolist(), e.tolist()))
        for runs in range(n + 1):
            for ends in range(3):
                assert erasures.path_run_count(n, m, runs, ends) == tally.get((runs, ends), 0)

n, m, S = 8, 3, 20_000
regions8 = all_regions(n, m)
index8 = {tuple(row): i for i, row in enumerate(regions8.tolist())}
labels = np.random.default_rng(5).permutation(n)
for geometry, edges in (("cycle", graphs.cycle(n)), ("path", graphs.path(n))):
    relabeled = labels[edges]  # the same ring or path under other labels
    b = boundaries(relabeled, n, regions8)
    drawn = erasures.sample_regions(relabeled, n, m, couplings, S, seed=9)
    assert (drawn.method, drawn.geometry, drawn.warnings) == ("exact", geometry, ())
    for i, coupling in enumerate(couplings):
        q = np.exp(-coupling * (b - b.min()))
        q /= q.sum()
        rows = [index8[tuple(row)] for row in drawn.regions[i].tolist()]
        assert np.array_equal(drawn.boundaries[i], b[rows])
        tv = 0.5 * np.abs(np.bincount(rows, minlength=len(q)) / S - q).sum()
        assert tv <= 0.5 * np.sqrt(q * (1 - q) / S).sum() + math.sqrt(math.log(1e6) / (2 * S))
```

### (b) A small graph: the chain against the exact $Q_J$

On a Watts–Strogatz graph with $n = 10$ (210 regions of size 4) the chain is
the only sampler. Its records at four couplings, negative included, against
the exact law of every region (enumerated) and the exact $\langle b\rangle_J$
(`boundary_moments`). The records of a chain are correlated, so the bounds use
the smallest effective sample size among $b$ and the followed nodes'
memberships in place of $S$: a heuristic version of the bound in (a), not a
theorem. `tests/test_erasures.py` runs the same comparison with chi-square
tests on several graphs, with and without replica exchange.

```python
n, m, S = 10, 4, 20_000
graph10 = graphs.watts_strogatz(n, 4, 0.3, seed=2)
regions10 = all_regions(n, m)
index10 = {tuple(row): i for i, row in enumerate(regions10.tolist())}
b10 = boundaries(graph10, n, regions10)
couplings = [-1.0, 0.0, 1.0, 2.5]
drawn = erasures.sample_regions(graph10, n, m, couplings, S, thin=2, seed=5)
assert drawn.method == "chain" and drawn.warnings == ()
assert drawn.ladder[0] == -1.0 and 0.0 in drawn.ladder  # the automatic ladder covers [-1, 2.5]
moments = erasures.boundary_moments(erasures.boundary_counts(graph10, n, m), couplings)
assert np.all(np.diff(moments["mean"]) < 0)  # Proposition 2: <b> falls as J grows
for i, coupling in enumerate(couplings):
    q = np.exp(-coupling * (b10 - b10.min()))
    q /= q.sum()
    rows = [index10[tuple(row)] for row in drawn.regions[i].tolist()]
    tv = 0.5 * np.abs(np.bincount(rows, minlength=len(q)) / S - q).sum()
    on_b = erasures.chain_diagnostics(drawn.boundaries[i])
    ess = min(on_b["ess"], erasures.region_diagnostics(drawn.regions[i], n)["ess_min"])
    assert tv <= 0.5 * np.sqrt(q * (1 - q) / ess).sum() + math.sqrt(math.log(1e6) / (2 * ess))
    assert abs(on_b["mean"] - moments["mean"][i]) <= 5 * on_b["std_error"]
    print(f"J = {coupling:+.1f}: TV {tv:.3f}, mean b {on_b['mean']:.3f} (exact {moments['mean'][i]:.3f}),"
          f" ESS {ess:.0f} of {S}")
```

### (c) A brickwork ring code, $n = 16$

The nearest-neighbour ring (`graphs.cycle`) with `schedule="matching"`, which
on an even ring is the brickwork (even bonds, then odd bonds), in the
purification picture, $4n$ layers at measurement rate 0.1 (below the critical
rate $p_c \approx 0.16$, §11), and the erasure graph equal to the circuit's
ring. The example takes the first trajectory whose code has $k > 0$ and no
failing single qubit (or, failing that, the first with $k > 0$). The
independent oracle computes $\ell_C(A)$ for all $2^{16}$ regions from the
generators alone, by formula (1) of
[random-region-erasure.md](random-region-erasure.md) and plain-Python GF(2)
ranks.

```python
from sparsegf2 import mipt

n = 16
ring = graphs.cycle(n)
spec = mipt.MonitoredCircuit(
    num_system=n,
    edges=ring,
    schedule="matching",  # brickwork on the even ring
    layers=4 * n,
    measurement_rate=0.1,
    reference="purification",
    checkpoints="final",
    probes=[mipt.code_dimension(), mipt.uncorrectable_region_counts(max_size=1), mipt.code_snapshot()],
)
res = spec.run(8, seed=11)
holds_code = res.values["code_dimension"][:, -1] > 0
no_single = res.values["uncorrectable_region_counts[1]"][:, -1, 0] == 0  # no failing single qubit
assert holds_code.any()
t = int(np.flatnonzero(holds_code & no_single if (holds_code & no_single).any() else holds_code)[0])
code = res.code(t, 0)  # the snapshot at the final layer
k = code.num_logical
print(f"brickwork trajectory {t}: k = {k}, contiguous distance {code.contiguous_distance()}")


def lost_for_every_region(code):
    """(|A|, l(A)) for every region A of the code's qubits, A given by its bitmask.

    l(A) = 2|A| - s - rank(H_A) + rank(H_Abar), with plain-Python GF(2) ranks of the
    generators' columns: independent of the simulator's kernels.
    """
    n = code.num_qubits
    h = code.stabilizer_matrix()  # the generators: X bits of qubits 0..n-1, then Z bits
    col = [int(sum(int(bit) << j for j, bit in enumerate(h[:, c]))) for c in range(2 * n)]
    rank = np.zeros(1 << n, dtype=np.int64)  # rank[a] = rank(H_A)

    def extend(a, first, basis):  # basis: distinct leading bits, in decreasing order
        rank[a] = len(basis)
        for q in range(first, n):
            grown = list(basis)
            for v in (col[q], col[n + q]):  # the X and Z columns of qubit q
                for vector in grown:
                    v = min(v, v ^ vector)  # clears vector's leading bit from v if v has it
                if v:
                    grown.append(v)
                    grown.sort(reverse=True)
            extend(a | (1 << q), q + 1, grown)

    extend(0, 0, [])
    masks = np.arange(1 << n)
    size = sum((masks >> q) & 1 for q in range(n))
    return size, 2 * size - rank[-1] - rank + rank[(1 << n) - 1 - masks]


def boundaries_of_every_region(edges, n):
    """b(A) for every region A, given by its bitmask."""
    member = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(bool)
    return np.count_nonzero(member[:, edges[:, 0]] != member[:, edges[:, 1]], axis=1)


size, ell = lost_for_every_region(code)
b_all = boundaries_of_every_region(ring, n)
assert ell[0] == 0 and ell[-1] == 2 * k

counts = code.boundary_failure_counts(ring, n)  # U_m(b), N_m(b) for m = 1..16
width = counts.total.shape[1]
for m in range(1, n + 1):
    pick = size == m
    assert np.array_equal(np.bincount(b_all[pick], minlength=width), counts.total[m - 1])
    assert np.array_equal(np.bincount(b_all[pick & (ell > 0)], minlength=width), counts.failed[m - 1])
assert np.array_equal(counts.failed.sum(axis=1), code.uncorrectable_counts(n))
for m in range(1, n):  # the ring's density of states in closed form: N_m(2r) = N(n, m, r)
    closed = [erasures.ring_run_count(n, m, b // 2) if b % 2 == 0 else 0 for b in range(width)]
    assert counts.total[m - 1].tolist() == closed
```

**The exact $P_J(m)$ for several couplings**, against the direct sum over all
regions; the identity at $J = 0$; and the interval limit three ways (brute
force over windows, $U_m(2)/N_m(2)$, and `exact` at $J = 10^4$), with the
explicit rate of §3.2 at $J = 4$:

```python
couplings = [-1.0, 0.0, 0.5, 1.0, 2.0, 4.0]
p_exact = counts.exact(couplings)  # (6, 16): P_J(m), m = 1..16
for i, coupling in enumerate(couplings):
    weight = np.exp(-coupling * b_all)
    direct = [weight[(size == m) & (ell > 0)].sum() / weight[size == m].sum() for m in range(1, n + 1)]
    assert np.allclose(p_exact[i], direct, rtol=1e-11, atol=0)
uniform = analysis.exact_failure_by_size(code.uncorrectable_counts(n), n)
assert np.array_equal(p_exact[1], uniform)  # J = 0 is the uniform P_C(m), bit for bit

intervals = code.interval_failure_by_size()  # J -> inf on the ring, m = 0..n
windows = [sum(1 << ((s + j) % n) for j in range(m)) for m in range(1, n) for s in range(n)]
assert np.array_equal(intervals[1:n], (ell[windows].reshape(n - 1, n) > 0).sum(axis=1) / n)
assert np.array_equal(counts.failed[: n - 1, 2] / n, intervals[1:n])  # U_m(2) / N_m(2)
assert np.array_equal(counts.exact(1e4)[0, : n - 1], intervals[1:n])
rate = np.array([1 - erasures.ring_boundary_distribution(n, m, 4.0)[1][0] for m in range(1, n)])
assert np.all(np.abs(p_exact[-1, : n - 1] - intervals[1:n]) <= rate)  # |P_4 - P_inf| <= Q_4(r >= 2)
np.set_printoptions(precision=3, suppress=True, linewidth=120)
print("P_J(m), m = 1..8, rows J =", couplings)
print(p_exact[:, :8])
print("J -> inf (intervals):", intervals[1:9])
```

At a fixed column (fixed $m$) of the printout, the slope in $J$ is
$-\operatorname{Cov}_J(F_C, b)$ (§6.1), and nothing fixes its sign.

**The samplers.** On the ring `correlated_failure` samples exactly
(`method="auto"`); the chain is forced for comparison. Both agree with the
exact values within $4$ standard errors plus $2/\mathrm{ESS}$ (the second
term allows for the discreteness of a count when $P$ is tiny):

```python
sizes = [2, 4, 6, 8]
sampled_couplings = [0.0, 1.0, 4.0]
target = counts.exact(sampled_couplings)[:, [m - 1 for m in sizes]]
for method, expected in (("auto", "exact"), ("chain", "chain")):
    est = code.correlated_failure(ring, sampled_couplings, sizes, 4000, seed=1, method=method)
    assert (est.method, est.geometry) == (expected, "cycle")
    se = np.sqrt(target * (1 - target) / est.failure_ess)
    assert np.all(np.abs(est.failure - target) <= 4 * se + 2 / est.failure_ess)
    print(f"{expected:>5} sampler, m = {sizes}:", np.round(est.failure, 3).tolist())
```

**$\Phi_{C,G,J}$ and $\varepsilon_\delta(J)$.** Every size is exact here, so the
full curve has nothing interpolated; at $J = 0$ the binomially mixed channel
is the independent-erasure $\Phi_C$ of
[random-region-erasure.md](random-region-erasure.md) (Corollary 2), computed
there from exact counts covering every size:

```python
curve, filled = analysis.correlated_failure_curve(
    np.zeros((len(couplings), 0)), [], n, exact=p_exact, holds_code=k
)
assert not filled.any()
e = np.array([0.02, 0.05, 0.1, 0.2])
phi = analysis.failure_probability_from_curve(curve, n, e)  # (couplings, rates)
iid = analysis.erasure_failure_probability(
    code.first_failure_sizes(10, seed=0), n, e, exact_counts=code.uncorrectable_counts(n)
)
assert np.allclose(phi[1], iid, rtol=1e-12, atol=1e-15)  # Phi_{C,G,0} = Phi_C
tolerance, monotone = analysis.tolerance_from_curve(curve, n, 0.1, return_monotone=True)
print("Phi_J(e) at e =", e.tolist())
print(phi)
print("e_0.1(J):", np.round(tolerance, 4).tolist(), "monotone in m:", monotone.tolist())
```

### (d) The same on a Watts–Strogatz graph

The graph of §3.5, on which a shortcut makes the minimum boundary smaller than
that of any interval; a circuit with `schedule="random_edges"` on it; and the
erasure graph equal to the circuit's graph. Here the chain is the only
sampler, the $J \to \infty$ limit is the failing fraction of the minimizers
(not the interval curve), and the full curve mixes exact small sizes, sampled
sizes and interpolated ones:

```python
ws = graphs.watts_strogatz(n, 4, 0.1, seed=ws_seed)
spec_ws = mipt.MonitoredCircuit(
    num_system=n,
    edges=ws,
    schedule="random_edges",
    layers=4 * n,
    measurement_rate=0.1,
    reference="purification",
    checkpoints="final",
    probes=[mipt.code_dimension(), mipt.uncorrectable_region_counts(max_size=1), mipt.code_snapshot()],
)
res_ws = spec_ws.run(8, seed=7)
holds_ws = res_ws.values["code_dimension"][:, -1] > 0
no_single_ws = res_ws.values["uncorrectable_region_counts[1]"][:, -1, 0] == 0
assert holds_ws.any()
t_ws = int(np.flatnonzero(holds_ws & no_single_ws if (holds_ws & no_single_ws).any() else holds_ws)[0])
code_ws = res_ws.code(t_ws, 0)
k_ws = code_ws.num_logical

size, ell = lost_for_every_region(code_ws)
b_all = boundaries_of_every_region(ws, n)
counts_ws = code_ws.boundary_failure_counts(ws, n)
width = counts_ws.total.shape[1]
for m in range(1, n + 1):
    pick = size == m
    assert np.array_equal(np.bincount(b_all[pick], minlength=width), counts_ws.total[m - 1])
    assert np.array_equal(np.bincount(b_all[pick & (ell > 0)], minlength=width), counts_ws.failed[m - 1])

ws_couplings = [0.0, 1.0, 2.0, 4.0]
p_ws = counts_ws.exact(ws_couplings)
for i, coupling in enumerate(ws_couplings):
    weight = np.exp(-coupling * b_all)
    direct = [weight[(size == m) & (ell > 0)].sum() / weight[size == m].sum() for m in range(1, n + 1)]
    assert np.allclose(p_ws[i], direct, rtol=1e-11, atol=0)

# J -> inf: the failing fraction of the minimum-boundary regions, three ways
minimizers_by_size = [erasures.minimum_boundary_regions(ws, n, m)[1] for m in range(1, n)]
limit = np.array([code_ws.region_failures(rows).mean() for rows in minimizers_by_size])
assert np.array_equal(counts_ws.exact(1e4)[0, : n - 1], limit)
for m, rows in zip(range(1, n), minimizers_by_size, strict=True):
    failing = sum(int(ell[sum(1 << int(q) for q in row)] > 0) for row in rows)
    assert failing == limit[m - 1] * len(rows)
assert not any(is_interval(row, n) for row in minimizers_by_size[m_short - 1])  # as in 3.5
print(f"Watts-Strogatz trajectory {t_ws}: k = {k_ws}")
print("J -> inf, minimizers:", np.round(limit[:10], 3))
print("intervals (not the limit here):", np.round(code_ws.interval_failure_by_size()[1:11], 3))
```

**The chain, and a hybrid curve.** The chain at four couplings and five
sizes, against the exact values; then a curve as one would build it for a
large code: exact counts for $m \le 3$, the sampled sizes, and interpolation
in between, flagged:

```python
sizes_ws = [4, 6, 8, 10, 12]
est_ws = code_ws.correlated_failure(ws, ws_couplings, sizes_ws, 2000, seed=3)
assert (est_ws.method, est_ws.geometry) == ("chain", "other")
target = p_ws[:, [m - 1 for m in sizes_ws]]
se = np.sqrt(target * (1 - target) / est_ws.failure_ess)
assert np.all(np.abs(est_ws.failure - target) <= 4 * se + 2 / est_ws.failure_ess)
print("rungs per size:", est_ws.rungs.tolist(), " smallest exchange rate:",
      np.round(est_ws.min_exchange_acceptance, 2).tolist(), " warnings:", est_ws.warnings)

small = code_ws.boundary_failure_counts(ws, 3)  # exact for m = 1..3
hybrid, filled = analysis.correlated_failure_curve(
    est_ws.failure, sizes_ws, n, exact=small.exact(ws_couplings), holds_code=k_ws
)
assert np.flatnonzero(filled).tolist() == [5, 7, 9, 11, 13, 14, 15]  # the interpolated sizes
full, _ = analysis.correlated_failure_curve(np.zeros((4, 0)), [], n, exact=p_ws, holds_code=k_ws)
assert np.array_equal(hybrid[:, :4], full[:, :4])  # m = 0..3: exact in both
e = np.array([0.05, 0.1, 0.2])
print("largest |hybrid - exact| per J:", np.round(np.abs(hybrid - full).max(axis=1), 3).tolist())
print("Phi_J(e), hybrid:\n", analysis.failure_probability_from_curve(hybrid, n, e))
print("Phi_J(e), exact:\n", analysis.failure_probability_from_curve(full, n, e))
print("exact curves monotone in m:", analysis.tolerance_from_curve(full, n, 0.1, return_monotone=True)[1])
```

Interpolation assumes the curve is smooth between sampled sizes; for $J \ne 0$
on a graph with shortcuts the exact curve can step down (§6.7), and the
printed gap shows what interpolation can miss there. Sample the sizes that
matter (the binomial window of §6.6) rather than relying on interpolation.

### (e) A small ensemble with the probe, its summary and `erasure_curves`

A JSON run (as the CLI and MCP send it) on the Watts–Strogatz graph of §3.5:
40 trajectories, the probe on the interaction graph (the chain) at three
couplings and every second size, a second probe on the ring of qubit labels
(exact samples) at its own final checkpoint, and `code_dimension` and
`erasure_first_failure` beside them. The example checks:

* the layout of §7.3 and the summary's means, against the saved arrays;
* replays of one trajectory, with the same probe and with a narrower one whose
  sizes and couplings are in another order (§7.4);
* the probe at $J = 0$ against the uniform first-failure estimator of the same
  codes: the two are independent given the code and both unbiased for
  $P_C(m)$ (§3.1), so their paired difference has mean 0;
* `erasure_curves` against the `analysis` functions applied by hand (§7.5).

```python
run_spec = {
    "num_system": 16,
    "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": ws_seed},
    "layers": 64,
    "measurement_rate": 0.1,
    "schedule": "random_edges",
    "reference": "purification",
    "checkpoints": 32,
    "probes": [
        {"kind": "code_dimension"},
        {"kind": "erasure_first_failure", "samples": 200},
        {"kind": "correlated_erasure", "couplings": [0, 1, 2], "sizes": "2:15:2", "samples": 200},
        {"kind": "correlated_erasure", "name": "on_ring", "couplings": [0, 4], "sizes": [4, 8],
         "samples": 200, "geometry": "cycle", "checkpoints": "final"},
    ],
}
summary = api.run_ensemble(run_spec, trajectories=40, seed=3, output_dir="runs/correlated")
entry = summary["probes"]["correlated_erasure"]
print("samplers:", entry["sampler"], "|", summary["probes"]["on_ring"]["sampler"])

res = mipt.EnsembleResult.load("runs/correlated/result.npz")
n = res.spec.num_system
couplings, sizes, values = res.correlated("correlated_erasure")  # values[t, e, c, i]
assert values.shape == (40, len(res.layers("correlated_erasure")), 3, 7)
assert np.array_equal(res.values["correlated_erasure"][:, :, 1 * 7 + 2], values[:, :, 1, 2])
assert res.spec.correlated_sampler("correlated_erasure")["method"] == "chain"
assert res.spec.correlated_sampler("on_ring") == {
    "method": "exact", "graphs": 1, "exact_graphs": 1, "geometries": ["cycle"]
}
holds = res.values["code_dimension"][:, -1] > 0  # the summary conditions on k > 0
mean, err = analysis.mean_and_error(values[holds, -1], axis=0)
assert entry["count_with_code"] == holds.sum()
assert np.allclose(entry["mean"], mean, rtol=1e-5, atol=1e-6)

t = int(np.flatnonzero(holds)[0])
last = int(res.layers("correlated_erasure")[-1])
same = next(p for p in res.spec.probes if p.name == "correlated_erasure")
assert np.array_equal(res.replay(t, probes=[same], checkpoints=[last]).correlated()[2][0, -1],
                      values[t, -1])
narrow = mipt.correlated_erasure([2, 0, 1], sizes=[8, 4], samples=200)  # same name, other order
_, _, again = res.replay(t, probes=[narrow], checkpoints=[last]).correlated()
assert np.array_equal(again[0, -1], values[t, -1][[2, 0, 1]][:, [3, 1]])  # J = 2, 0, 1; m = 8, 4

first = res.values["erasure_first_failure"][holds, -1]
paired = values[holds, -1, 0] - analysis.failure_by_size(first, n)[:, sizes]  # J = 0 - uniform
d_mean, d_err = analysis.mean_and_error(paired, axis=0)
assert np.all(np.abs(d_mean) <= 4 * d_err + 0.01)

curves = api.erasure_curves("runs/correlated", e=[0.05, 0.1, 0.2], delta=0.05)
row = curves["correlated"]["correlated_erasure"]
per_code, filled = analysis.correlated_failure_curve(values[holds, -1], sizes, n)
e = np.array([0.05, 0.1, 0.2])
phi = analysis.failure_probability_from_curve(per_code, n, e)  # (codes, couplings, rates)
assert np.allclose(row["phi"]["mean"], phi.mean(axis=0), rtol=1e-5)
assert np.allclose(row["e_delta"]["of_mean_phi"],
                   analysis.tolerance_from_curve(per_code.mean(axis=0), n, 0.05), rtol=1e-5)
assert np.flatnonzero(filled).tolist() == list(range(1, 16, 2))  # the odd sizes, interpolated
print("Phi_J(e) per J:", np.round(phi.mean(axis=0), 3).tolist())
print("interpolated mass per e:", row["interpolated_mass"], " notes:", len(curves["notes"]))
```

The uniform entry of `erasure_curves` (`curves["uniform"]`) is the same channel
at $J = 0$, from the first-failure samples; here the interpolated odd sizes
carry about half the binomial mass, which `interpolated_mass` and a note
report. A study would sample every size of the window.

---

## 10. Pitfalls

* **Codes with $k = 0$.** They fail nowhere, at every $J$: $P_J \equiv 0$ and
  $\Phi_J \equiv 0$. Averages over all trajectories mix them in as perfect
  protection. Select $k > 0$ (`analysis.mean_over_codes(values, holds_code)`),
  pass `holds_code` to `correlated_failure_curve` (otherwise $P(n) = 1$ is
  assumed unless `exact` covers $m = n$), and report the fraction with $k > 0$
  separately.
* **Interpolated sizes are not measurements.** `correlated_failure_curve`
  returns the mask of interpolated sizes; say which they are. For $J \ne 0$ the
  true curve can be non-monotone between sampled sizes (§6.7, §9(d)), so sample
  the sizes in the binomial window of the rates you report instead.
* **Mixing at large $J$.** The chain's slow modes are positions, not
  boundaries (§4.8): the ring's interval only diffuses, a lone chain stays at
  one end of the open path for at least $e^{J}$ proposals, and on general
  graphs a coarse ladder or a short burn-in biases the cold rungs without any
  warning. Keep `method="auto"` (exact on
  cycles and paths), the automatic ladder and the default burn-in, read
  `min_exchange_acceptance` and the warnings, and confirm with several seeds and
  a longer run. No warning is no proof (§4.9).
* **$J \to \infty$ is a limit, not a coupling.** Infinite couplings are refused;
  a chain at a huge finite $J$ only finds local minima. Use the exact routes of
  §3.7.
* **The open path.** Its $J \to \infty$ limit is the two end windows,
  `end_window_failure_by_size`, not `interval_failure_by_size(periodic=False)`,
  which weighs all windows equally.
* **Shortcuts, and which graph.** On graphs with shortcuts (or with
  next-nearest-neighbour edges, as Watts–Strogatz with $k = 4$) the minimizers
  need not be intervals, and `interval_failure_by_size` is not the limit
  (§3.5). Decide which graph models the physical layout: the circuit's own
  interaction graph, or the ring of qubit labels on which a Watts–Strogatz
  graph is built (pass `graphs.cycle(n)` as `edges` for the latter).
* **$J \lt 0$.** Negative couplings favour spread-out regions; the limit is the
  maximum-boundary regions and Proposition 2's monotonicity is the same
  statement ($\langle b\rangle$ still falls as $J$ grows, so it rises as $J$ goes
  more negative). The ladder spacing uses $\operatorname{Var}_0(b)$, which can
  misjudge $\operatorname{Var}_J(b)$ at large $\lvert J\rvert$ of either sign:
  watch the exchange warning.
* **Non-monotone curves.** For $J \ne 0$, $P_J(m)$ need not be monotone in $m$
  (§6.7) or in $J$ (§6.1). `tolerance_from_curve` returns the last crossing,
  and with `return_monotone=True` says whether the curve was monotone; report
  that flag with $\varepsilon_\delta$.
* **One seed for many codes.** `correlated_failure` on saved codes with the
  same seed draws the same regions for every code. For ensemble error bars
  give each code its own seed (`seed=[base, t]`, §5.2). The in-run probe needs
  nothing of the kind: every trajectory, evaluation and size has its own
  stream (§7.4).
* **The probe cannot see bad mixing.** It stores only failure fractions, so
  neither its summary nor `erasure_curves` can warn about a chain that has not
  mixed. Check the sampler on the graph first, with the probe's settings and
  several seeds (`sample_erasure_regions`, §8.3).
* **Cost.** Every size of every evaluation pays a full burn-in, about
  $n^{2.5}$ proposals per size on the chain: at $n = 256$ every size at one
  evaluation took 52 s on one thread (§8.1). Give the probe a window of sizes
  and its own few checkpoints, and read `estimate_ensemble` first.
* **When $J$ cannot matter.** If the ensemble of codes is invariant in law
  under every permutation of the qubits while $G$ is fixed (for example, codes
  made by circuits whose gates act on uniformly random pairs of qubits), then $\mathbb{E}_C F_C(A)$ depends on
  $A$ only through $\lvert A\rvert$, so
  $\mathbb{E}_C P_{C,G,J}(m) = \sum_A Q_J(A)\,\mathbb{E}_C F_C(A) = \mathbb{E}_C P_C(m)$
  for every $G$ and $J$: on average the correlation is invisible, and only
  code-to-code fluctuations remain. Geometry matters only for codes whose
  structure is aligned with $G$.
* **The default sizes are every $m = 1, \dots, n-1$**, for the probe and for
  `correlated_failure`, and each size runs its own sampler, burn-in included,
  so the default costs about $n - 1$ sampler runs. Choose sizes with
  `analysis.binomial_window` for the rates you need.
* **Exact enumeration has a budget.** `boundary_failure_counts` refuses more
  than `budget` regions in all (default two million), and
  `minimum_boundary_regions` and `boundary_counts` more than `budget` regions of
  one size; the message gives the count and the largest size that fits.

---

## 11. Literature

The claims below were checked against the sources as noted: "full text" means
the cited statement was found in the paper's full text; "abstract" means only
the abstract or catalog page was read. Nothing here was found that already
studies Gibbs-weighted fixed-size erasure sets $Q_J(A \mid m) \propto e^{-J b_G(A)}$,
or random (non-contiguous) erasure statistics of the codes made by monitored
brickwork Clifford circuits; not finding a source is not proof that none
exists.

**The measurement-induced transition and its Clifford critical rate** (brickwork
of random two-qubit Cliffords on a ring, each qubit measured in $Z$ with
probability $p$ after every layer):

* M. J. Gullans and D. A. Huse, "Dynamical purification phase transitions
  induced by quantum measurements", Phys. Rev. X 10, 041020 (2020),
  [arXiv:1905.05195](https://arxiv.org/abs/1905.05195) (full text):
  $p_c = 0.1593(5)$, $\nu = 1.28(2)$ from the tripartite information of three
  contiguous quarters, $L = 128$–512; the contiguous code length, an upper bound
  on the distance, subextensive deep in the mixed phase.
* A. Zabalo et al., Phys. Rev. B 101, 060301(R) (2020),
  [arXiv:1911.00008](https://arxiv.org/abs/1911.00008) (full text):
  $p_c = 0.154(4)$ from crossings at $L = 16$–24.
* Y. Li, X. Chen and M. P. A. Fisher, Phys. Rev. B 100, 134306 (2019),
  [arXiv:1901.08092](https://arxiv.org/abs/1901.08092) (full text):
  $p_c \approx 0.16$, $\nu \approx 1.3$.
* Y. Li, X. Chen, A. W. W. Ludwig and M. P. A. Fisher, Phys. Rev. B 104, 104305
  (2021), [arXiv:2003.12721](https://arxiv.org/abs/2003.12721) (full text):
  $p_c = 0.1600 \pm 0.0003$.

**Contiguous versus random erasures, and the distance of these codes:**

* Y. Li and M. P. A. Fisher, "Statistical mechanics of quantum error
  correcting codes", Phys. Rev. B 103, 104306 (2021),
  [arXiv:2007.03822](https://arxiv.org/abs/2007.03822) (full text): the
  contiguous code distance $d_{\mathrm{cont}}$ (the shortest contiguous segment
  supporting a logical operator), $d \le d_{\mathrm{cont}}$ by definition,
  $d_{\mathrm{cont}} \propto L^{\gamma_1}$ with $\gamma_1 \approx 0.38$, and the
  complementarity $\ell_A + \ell_{\bar A} = 2k$ (their Theorem 1) used in §6.4.
* Y. Li, S. Vijay and M. P. A. Fisher, "Entanglement domain walls in monitored
  quantum circuits and the directed polymer in a random environment", PRX
  Quantum 4, 010331 (2023), [arXiv:2105.13352](https://arxiv.org/abs/2105.13352)
  (full text): $d_{\mathrm{cont}} \propto L^{1/3}$ from the directed-polymer
  picture.
* M. J. Gullans, S. Krastanov, D. A. Huse, L. Jiang and S. T. Flammia,
  "Quantum coding with low-depth random circuits", Phys. Rev. X 11, 031066
  (2021), [arXiv:2010.09775](https://arxiv.org/abs/2010.09775) (full text):
  uniformly random erasures of unitary random-circuit codes (no measurements);
  in one dimension, regions with excess erasures fail first, and uncorrectable
  events bunch over a distance of the order of the depth: the mechanism by
  which clustered erasures should hurt a local code.
* B. Yoshida, "Decoding the entanglement structure of monitored quantum
  circuits", [arXiv:2109.08691](https://arxiv.org/abs/2109.08691) (full text):
  notes that Li and Fisher studied the contiguous, not the conventional,
  distance.

**The swap chain and its slow mixing:**

* D. A. Levin, Y. Peres and E. L. Wilmer, *Markov Chains and Mixing Times*,
  2nd ed. (AMS, 2017), Section 3.2 and Exercise 18.3,
  [free PDF](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf) (full text):
  the Metropolis chain of a symmetric base chain is reversible for its target
  and needs only ratios; the Bernoulli–Laplace swap has cutoff at
  $\tfrac18 n\log n$.
* K. Kawasaki, "Diffusion constants near the critical point for time-dependent
  Ising models. I", Phys. Rev. 145, 224 (1966) (abstract): spin-exchange
  dynamics.
* Y. Shida and E. Henriques, "Kawasaki dynamics and equilibrium distributions
  in simulations of phase separating systems",
  [arXiv:cond-mat/9703198](https://arxiv.org/abs/cond-mat/9703198) (abstract):
  state-dependent pair choices bias the equilibrium law; exchanges of opposite
  spins at arbitrary distance avoid it.
* A. J. Bray, "Theory of phase ordering kinetics", Adv. Phys. 43, 357 (1994),
  [arXiv:cond-mat/9501089](https://arxiv.org/abs/cond-mat/9501089) (full text):
  $t^{1/3}$ coarsening for a conserved order parameter, $t^{1/2}$ without.
* R. H. Swendsen and J.-S. Wang, "Replica Monte Carlo simulation of spin
  glasses", Phys. Rev. Lett. 57, 2607 (1986) (verified through Wang and
  Swendsen, [arXiv:cond-mat/0407273](https://arxiv.org/abs/cond-mat/0407273));
  K. Hukushima and K. Nemoto, "Exchange Monte Carlo method and application to
  spin glass simulations", J. Phys. Soc. Jpn. 65, 1604 (1996),
  [arXiv:cond-mat/9512035](https://arxiv.org/abs/cond-mat/9512035): replica
  exchange between temperatures. C. J. Geyer, "Markov chain Monte Carlo
  maximum likelihood", Computing Science and Statistics: Proc. 23rd Symposium
  on the Interface, 156 (1991), is commonly credited with parallel tempering
  too (not checked in the paper).
* D. J. Earl and M. W. Deem, "Parallel tempering: theory, applications, and new
  perspectives", Phys. Chem. Chem. Phys. 7, 3910 (2005),
  [arXiv:physics/0508111](https://arxiv.org/abs/physics/0508111) (abstract):
  a review.

**Minimum bisection:**

* U. Feige and R. Krauthgamer, "A polylogarithmic approximation of the minimum
  bisection", SIAM J. Comput. 31(4), 1090 (2002)
  ([PDF](https://www.wisdom.weizmann.ac.il/~robi/papers/FK-bisection2-SIGEST.pdf);
  abstract): finding a minimum bisection is NP-hard (§3.8).

**Autocorrelation times and R-hat:**

* B. Li, N. Madras and A. D. Sokal, J. Stat. Phys. 80, 661 (1995),
  [arXiv:hep-lat/9409003](https://arxiv.org/abs/hep-lat/9409003), Appendix C
  (full text): the automatic window $M \ge c\,\hat\tau(M)$ (from Madras and Sokal
  1988), $c \approx 5$ sufficing for a pure exponential, and underestimates of
  $\tau$ for slowly decaying correlations.
* U. Wolff, Comput. Phys. Commun. 156, 143 (2004),
  [arXiv:hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017) (full text):
  the statistical error of $\hat\tau$, relative error about
  $\sqrt{2(2W+1)/N}$.
* emcee documentation, the autocorrelation tutorial,
  [emcee.readthedocs.io](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/)
  (read): the window with $c = 5$ on $1 + 2\sum\rho$, and chains of about
  $50\tau$ or more.
* A. Vehtari, A. Gelman, D. Simpson, B. Carpenter and P.-C. Bürkner,
  "Rank-normalization, folding, and localization: an improved $\hat R$ for
  assessing convergence of MCMC", Bayesian Analysis 16(2), 667 (2021),
  doi:10.1214/20-BA1221,
  [arXiv:1903.08008](https://arxiv.org/abs/1903.08008) (full text): rank
  normalization with $(r - 3/8)/(S + 1/4)$, split and folded $\hat R$, the
  threshold 1.01, at least four chains, and an ESS above 400.

**The ring's run counts.** OEIS
[A119462](https://oeis.org/A119462) (read): circular binary words of length $n$
with $k$ occurrences of 01, $2\binom{n}{2k}$, which is
$\sum_m N(n, m, k)$; checked against the closed form for $n \le 39$. No
citable text with the refined count at fixed $m$ was found; §1.7 proves it.

**Correlated and burst erasure models in quantum error correction** (context;
none of them defines the Gibbs ensemble used here; abstracts):

* C. T. Chubb and S. T. Flammia, "Statistical mechanical models for quantum
  codes with correlated noise", Ann. Inst. Henri Poincaré D 8, 269 (2021),
  [arXiv:1809.10704](https://arxiv.org/abs/1809.10704).
* R. Klesse and S. Frank, "Quantum error correction in spatially correlated
  quantum noise", Phys. Rev. Lett. 95, 230503 (2005),
  [arXiv:quant-ph/0505153](https://arxiv.org/abs/quant-ph/0505153).
* D. Aharonov, A. Kitaev and J. Preskill, "Fault-tolerant quantum computation
  with long-range correlated noise", Phys. Rev. Lett. 96, 050504 (2006),
  [arXiv:quant-ph/0510231](https://arxiv.org/abs/quant-ph/0510231); J. Preskill,
  "Sufficient condition on noise correlations for scalable quantum computing",
  Quant. Inf. Comput. 13, 181 (2013),
  [arXiv:1207.6131](https://arxiv.org/abs/1207.6131).
* J. Fan, M.-H. Hsieh, H. Chen, H. Chen and Y. Li, "Construction and performance
  of quantum burst error correction codes for correlated errors", ISIT 2018,
  [arXiv:1801.03861](https://arxiv.org/abs/1801.03861): burst (contiguous)
  errors, the code-theory analogue of the $J \to \infty$ limit.
* T. M. Stace, S. D. Barrett and A. C. Doherty, "Thresholds for topological
  codes in the presence of loss", Phys. Rev. Lett. 102, 200501 (2009),
  [arXiv:0904.3556](https://arxiv.org/abs/0904.3556).
* N. H. Nickerson and B. J. Brown, "Analysing correlated noise on the surface
  code using adaptive decoding algorithms", Quantum 3, 131 (2019),
  [arXiv:1712.00502](https://arxiv.org/abs/1712.00502).
