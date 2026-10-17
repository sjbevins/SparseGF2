# The phase-free tableau: snapshots, canonical row spaces and trajectory indices

How SparseGF2 saves the full joint system–reference stabilizer state of a monitored
trajectory, reduces it to a canonical form that does not depend on how it was computed,
identifies it by a digest, and recovers from it the physical code, entropies and the
reference information $I(A:R)$; and how a run simulates an explicit list of trajectories.

This document is written for a reader fluent in the stabilizer formalism and for language
models. Every symbol is defined where it first appears, every formula is the one the code
computes, and every statement about the code names the function it describes as
`path::function` (paths relative to `src/sparsegf2/`). The Python blocks run as written,
in order, in one namespace, in a few seconds once the kernels are compiled.

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

The Watts–Strogatz study (`docs/misc/watts_strogatz_study_spec_v1_4.md`, cited as "v1.4")
asks for the full $2n$-qubit joint phase-free stabilizer row space at every checkpoint of
every trajectory, with a canonical binary-subspace digest (v1.4 §5, §5.1, §5.3, the
`snapshot` record of §8), and lists this as an extension the engine did not have (v1.4 §8.1:
the native `code_snapshot` stores only the physical code). Before a joint dictionary is
relied on, v1.4 §9.2 (h) requires "equality of the exported joint row space, its
reduced-code stabilizers, and direct simulator probes at all tested depths". The study's
runner also needs to run an arbitrary set of trajectories $u$ in one engine call, each
exactly as a range run would give it (v1.4 §7.2–§7.3).

This part implements:

* the probe `mipt.tableau_snapshot`: at each of its evaluations the engine writes every
  stabilizer generator of the joint state over every qubit (`_kernel/ensemble.py`, kernel
  `_kernel/entropy.py::joint_snapshot_rows`);
* `tableau.py::PhaseFreeTableau`: a phase-free stabilizer group held as a canonical GF(2)
  row space (the reduced row echelon form), with a byte format, a SHA-256 digest, the rank,
  isotropy, entropies, $I(A:R)$ and the physical code; its eliminations are compiled
  (`_kernel/rowspace.py`);
* `MonitoredCircuit.run(..., indices=...)`: a run on an explicit array of trajectory
  indices;
* the canonicalization of many snapshots at once, in parallel on every thread
  (`EnsembleResult.tableaux`, `PhaseFreeTableau.from_words_batch`,
  `tableau.py::physical_tableaux`, `tableau.py::to_bytes_batch`), with exactly the results of
  the one-at-a-time methods.

It implements v1.4 §5 (the state, the snapshot record and its digest), §5.1 (the phase-free
tableau), §5.2 ($r_t$, $k_t$), §5.3 (the tableau dictionary's entries), §5.4 (the
physical-rank formula for $I(A:R)$), §5.8 (the complement identity), and the snapshot and
trajectory-identity requirements of §6.2, §7.2, §7.3, §8, §8.1 and §9.2 (h). It changes no
probability law and no random stream (Proposition 13 in §4.8).

## 2. Notation

| symbol | meaning | type | in the code |
|---|---|---|---|
| $n$ | number of system qubits $Q_0, \dots, Q_{n-1}$ | integer $\ge 2$ | `num_system` |
| $V$ | the system, $\lbrace 0, \dots, n-1\rbrace$ | set of qubits | `system_qubits` |
| $R$ | the reference qubits $n, \dots, N-1$ (reference $n + i$ is $R_i$ of v1.4) | set of qubits | `reference_qubits` |
| $N$ | all qubits, $N = n + \lvert R\rvert$ ($2n$ in the purification picture) | integer | `num_qubits` |
| $\mathbb{F}_2$ | the field with two elements; all arithmetic on bits is mod 2 | | XOR, AND on `uint64` words |
| $\varphi(P)$ | the phase-free vector $(x_0, \dots, x_{N-1}, z_0, \dots, z_{N-1})$ of a Pauli $P$ | $\mathbb{F}_2^{2N}$ | a row |
| $\sigma(x, z)$ | the single-qubit Pauli $i^{xz} X^x Z^z$ ($I$, $X$, $Z$, $Y$) | operator | |
| $\lambda$ | a scalar $\pm 1$, $\pm i$ (the phase of a Pauli operator) | complex number | |
| $\langle v, w\rangle$ | the symplectic form $\sum_q (x_q z'_q + z_q x'_q)$ | $\mathbb{F}_2$ | `_kernel/entropy.py::generators_commute` |
| $U^\perp$ | the symplectic complement $\lbrace w : \langle v, w\rangle = 0 \text{ for all } v \in U\rbrace$ of a subspace $U$ | subspace | |
| $\mathsf S$, $\widetilde{\mathsf S}$ | a stabilizer group and its phase-free group $\varphi(\mathsf S)$ | group, subspace | engine tableau rows; `PhaseFreeTableau` |
| $r$ | $\dim \widetilde{\mathsf S}$ | integer | `PhaseFreeTableau.rank` |
| $\pi_A$ | restriction to the qubits $A$: keep $x_q$, $z_q$ for $q \in A$ | linear map $\mathbb{F}_2^{2N} \to \mathbb{F}_2^{2\lvert A\rvert}$ | column subset |
| $\bar A$ | the complement of a set of qubits $A$ among all $N$ qubits | set of qubits | `np.setdiff1d` in `PhaseFreeTableau.entropy` |
| $\widetilde{\mathsf S}\vert_A$, $U\vert_A$ | $\pi_A(\widetilde{\mathsf S})$, the generators restricted to $A$ (likewise for any subspace $U$) | subspace | `_kernel/rowspace.py::rank_of_columns` |
| $\widetilde{\mathsf S}_A$, $U_A$ | $\lbrace v \in \widetilde{\mathsf S} : \operatorname{supp} v \subseteq A\rbrace$ (likewise for $U$) | subspace | (dimension only) |
| $\mathsf S_A$ | the elements of $\mathsf S$ that act as a scalar on $\bar A$ (Proposition 9) | group | |
| $\Pi$, $\rho$, $\rho_A$ | the projector onto a stabilized subspace; the state $2^{-N}\sum_{g \in \mathsf S} g$; its reduced state on $A$ | operators | |
| $G$, $M_G$ | a Clifford gate and its $2N \times 2N$ binary symplectic matrix (Proposition 4) | operator, matrix | `_kernel/tableau.py::apply_2q` |
| $M$, $v_M$ | a measured Pauli and $\varphi(M)$ (Proposition 4) | operator, vector | `_kernel/tableau.py::measure_1q_phase_free_tracked` |
| $\mathcal P_n$ | the Pauli group on the $n$ system qubits (Pauli operators with phases $\pm 1$, $\pm i$) | group | |
| $\mathsf S_t$ | the signed stabilizer group of the joint state after $t$ layers | group | |
| $\mathsf S_t^{\mathrm{phys}}$ | the physical stabilizer group: the $P \in \mathcal P_n$ with $P \otimes I_R \in \mathsf S_t$ | group | `PhaseFreeTableau.physical_code(n)` (signs dropped) |
| $W_t^{\mathrm{phys}}$ | phase-free physical group: elements of $\widetilde{\mathsf S}_t$ with zero reference part, restricted to $V$ | subspace of $\mathbb{F}_2^{2n}$ | `PhaseFreeTableau.physical_tableau(n)` |
| $r_t$, $k_t$ | $\dim W_t^{\mathrm{phys}}$ and $n - r_t$ | integers | `physical_code(n).num_stabilizers`, `.num_logical` |
| $r_t(A)$ | $\dim\lbrace w \in W_t^{\mathrm{phys}} : \operatorname{supp} w \subseteq A\rbrace$ | integer | |
| $S(A)$ | von Neumann entropy of the reduced state on $A$, in bits | integer | `PhaseFreeTableau.entropy` |
| $I(A:B)$ | $S(A) + S(B) - S(A \cup B)$, in bits | integer | `PhaseFreeTableau.mutual_information` |
| $\mathrm{RREF}(\widetilde{\mathsf S})$ | the canonical basis (§3.4) | matrix | `PhaseFreeTableau.canonical()` |
| $d$ | the length of the vectors of a subspace $U \subseteq \mathbb{F}_2^d$ (§3.4, Proposition 5) | integer | `2 * num_qubits` |
| $L(U)$ | the leads (§3.4) of the nonzero elements of $U$ (Proposition 5) | set of columns | `pivots` |
| $J$ | a set of basis rows whose sum is a given element (Propositions 5, 7) | set | |
| $\mathcal B$, $\mathcal B_R$, $\mathcal B_V$ | the RREF basis for the reference-first column order, its rows with a reference lead and its rows with a system lead (Proposition 7) | sets of rows | in `physical_tableau`: `rows[:first]`, `rows[first:rank]` |
| $w_q$ | $\lceil N / 64\rceil$, words per half row (X words, then as many Z words) | integer | `wq` |
| $w_g$ | $\lceil N / 64\rceil$, words per column of the engine tableau (one bit per generator) | integer | `tab.shape[1]` |
| $W$ | words per packed row given to an elimination (§5.1, §5.2): $2 w_q$ for a phase-free row | integer | `rows.shape[1]` |
| $s$ | the number of rows given to an elimination (§5.1) | integer | `rows.shape[0]` |
| $i$, $j$ | indices, local to the passage where they appear; in §3.6, $j$ numbers the generators of the engine tableau, which are the rows of a snapshot | integers | `r` in `joint_snapshot_rows` |
| $C$ | circuits per graph in a collection of graphs | integer | `circuits_per_graph` |
| $u$ | a trajectory index (v1.4 §7.2): it runs graph $\lfloor u / C\rfloor$ with circuit $u \bmod C$ | integer $\ge 0$ | `indices[t]`, `trajectories[t]` |
| $h$ | the 64-bit hash of a probe's name, which keys its streams | integer | `phash[j]` in `_kernel/ensemble.py` |
| $\mathrm{mix}$ | the stream-index mixer of the engine's random numbers | function | `_kernel/rng.py::mix` |
| $m$ | an erasure size (§3.7) | integer | `sizes` |
| $T$, $E$ | the rows and the evaluations of a run's snapshot array (§3.6, §9) | integers | `num_trajectories`, `len(layers(name))` |
| $e$ | an evaluation of a probe (a column of its values) | integer | `e`, `evaluation` |
| $\ell$ | a depth: the number of completed layers | integer | `layer`, `layers(name)[e]` |
| $t$ | a depth in layers (§3.3–§3.5, §4.1–§4.7 and v1.4), or a row of a run's output (§3.6–§3.7, §4.8, §5.6–§5.7, §5.9, the stored fields of §6) | integer | `layer`; `t` |

The letter $t$ has the two meanings of the last row, as in the code: $\mathsf S_t$, $r_t$ and
$k_t$ are taken after $t$ completed layers; in the sections on runs (§3.6–§3.7, §4.8,
§5.6–§5.7, §5.9 and the stored fields of §6), $t$ is a row of the output, as the kernel's
`t` is, and the depth of evaluation $e$ is written $\ell$ = `layers(name)[e]`. A row's
trajectory is always called $u$ = `indices[t]` = `trajectories[t]`.

## 3. Definitions

### 3.1 The phase-free representation

Write the Hermitian single-qubit Paulis as $\sigma(x, z) = i^{xz} X^x Z^z$ for
$x, z \in \lbrace 0, 1\rbrace$: $\sigma(0,0) = I$, $\sigma(1,0) = X$, $\sigma(0,1) = Z$,
$\sigma(1,1) = iXZ = Y$. Every $N$-qubit Pauli operator is
$P = \lambda\, \sigma(x_0, z_0) \otimes \cdots \otimes \sigma(x_{N-1}, z_{N-1})$ with
$\lambda \in \lbrace \pm 1, \pm i\rbrace$, and its *phase-free vector* is

$$
\varphi(P) = (x_0, \dots, x_{N-1}, z_0, \dots, z_{N-1}) \in \mathbb{F}_2^{2N}.
$$

This column order, all X bits and then all Z bits, is the one used everywhere below. The
*symplectic form* is $\langle v, w\rangle = \sum_{q} (x_q z'_q + z_q x'_q) \bmod 2$ for
$v = (x, z)$ and $w = (x', z')$.

### 3.2 Phase-free groups and their row spaces

A *stabilizer group* is an abelian subgroup $\mathsf S$ of the Pauli group with
$-I \notin \mathsf S$; its state (or code space) is the joint $+1$ eigenspace. Its
*phase-free group* is $\widetilde{\mathsf S} = \varphi(\mathsf S) \subseteq \mathbb{F}_2^{2N}$,
the binary symplectic data with the generator signs removed (v1.4 §5.1). By Proposition 2 it
is a subspace, the row space of the vectors of any generating set. A subspace is
*isotropic* when $\langle v, w\rangle = 0$ for all its elements. `PhaseFreeTableau` holds any
subspace; a phase-free stabilizer group is an isotropic one, and a pure state's has
dimension $N$.

For a set of qubits $A$, $\pi_A$ keeps the $2\lvert A\rvert$ coordinates $x_q, z_q$ with
$q \in A$; $\widetilde{\mathsf S}\vert_A = \pi_A(\widetilde{\mathsf S})$ and
$\operatorname{rank}(\widetilde{\mathsf S}\vert_A)$ is its dimension (the rank of the
generator matrix restricted to those columns); $\widetilde{\mathsf S}_A$ is the subspace of
elements supported in $A$ ($x_q = z_q = 0$ for $q \notin A$).

### 3.3 The joint state and the physical code

The study's state (v1.4 §5) is $n$ Bell pairs $\lvert\Phi^+\rangle_{Q_i R_i}$, stabilized by
$X_{Q_i} X_{R_i}$ and $Z_{Q_i} Z_{R_i}$, evolved by the monitored circuit on the system
only. In the engine, system qubit $i$ is qubit $i$ and $R_i$ is qubit $n + i$ (the
purification picture, `reference="purification"`; `reference="single"` has one reference,
qubit $n$, paired with one system qubit; `"none"` has none). $\mathsf S_t$ is the signed
stabilizer group after $t$ completed layers and $\widetilde{\mathsf S}_t$ its phase-free
group. Following v1.4 §5 and §5.4,

$$
\mathsf S_t^{\mathrm{phys}} = \lbrace P \in \mathcal P_n : P \otimes I_R \in \mathsf S_t\rbrace,
\qquad W_t^{\mathrm{phys}} = \varphi(\mathsf S_t^{\mathrm{phys}}), \qquad
r_t = \dim W_t^{\mathrm{phys}}, \qquad k_t = n - r_t,
$$

and for $A \subseteq V$, $r_t(A) = \dim\lbrace w \in W_t^{\mathrm{phys}} : \operatorname{supp} w \subseteq A\rbrace$.

### 3.4 The canonical form

For a subspace $U \subseteq \mathbb{F}_2^{d}$ and a *column order* (a list of the $d$
coordinates), the *lead* of a nonzero $v$ is the first coordinate in the order where
$v$ is 1. A basis $b_1, \dots, b_r$ of $U$ is in *reduced row echelon form* (RREF) when

1. the leads $p_1, \dots, p_r$ of the rows strictly increase in the column order, and
2. every lead column is zero in every other row: the $p_i$-coordinate of $b_j$ is 0 for $j \ne i$.

The *canonical form* of a phase-free group (`PhaseFreeTableau.canonical()`) is its RREF for
the column order $x_0, \dots, x_{N-1}, z_0, \dots, z_{N-1}$, zero rows dropped.
Proposition 5 shows it exists and is unique, so it is a function of the row space alone
(Corollary 6).

*Convention.* v1.4 §5 asks for "a canonical binary-subspace digest" without fixing the
canonical form. The implementation chose the RREF in the column order of $\varphi$ because
it is unique (Proposition 5), cheap ($O(N^3/64)$ word operations, §5.1) and independent of
the generator basis, the row order, the simulator and the signs.

### 3.5 The byte format and the digest

`PhaseFreeTableau.to_bytes()` writes, in order:

| bytes | content |
|---|---|
| 8 | the magic `b"SGF2RS1\x00"` (`tableau.MAGIC`; format 1) |
| 4 | $N$, unsigned, little-endian |
| 4 | the rank $r$, unsigned, little-endian |
| $r \lceil 2N/8\rceil$ | the canonical rows in order; each row's $2N$ bits in the column order of §3.1, packed most significant bit first (`np.packbits(row, bitorder="big")`), the last byte padded with zero bits |

`PhaseFreeTableau.digest()` is the SHA-256 of these bytes as 64 hexadecimal characters: the
canonical phase-free row-space digest of v1.4 §5. A pure state on $N$ qubits takes
$16 + N\lceil N/4\rceil$ bytes: 65,552 at $N = 512$ ($n = 256$).
`PhaseFreeTableau.from_bytes` accepts only canonical bytes: it rebuilds the space and
refuses the input unless writing it back gives the same bytes.

### 3.6 The engine's tableau and the snapshot layout

The engine stores a state column-major (`_kernel/tableau.py`, module docstring): an array
`tab` of shape $(2N + 1, w_g)$ of `uint64`, where `tab[2q]` is the X column of qubit $q$ (bit
$j$ set when generator $j$ has X or Y on $q$), `tab[2q + 1]` the Z column, and
`tab[2N]` the sign row. Generator ("row") $j$ is bit $j \,\&\, 63$ of word $j \gg 6$. The
ensemble engine runs phase-free, so $w_g = \lceil N / 64\rceil$, there are no destabilizers,
and the sign row is never read. A pure state on $N$ qubits has exactly $N$ generators,
rows $0, \dots, N-1$.

A snapshot is the transpose, row-major (`_kernel/entropy.py::joint_snapshot_rows`): row $j$
of the output holds generator $j$ in $2 w_q$ words, $w_q = \lceil N/64\rceil$; $x_q$ is bit
$q \,\&\, 63$ of word $q \gg 6$ and $z_q$ is bit $q \,\&\, 63$ of word $w_q + (q \gg 6)$; bits
beyond qubit $N - 1$ are zero. This is the layout of a `code_snapshot`
(`_kernel/entropy.py::code_snapshot_rows`) extended to all qubits and all rows, and the layout of
`PhaseFreeTableau.from_words` and `PhaseFreeTableau.words`. A run stores the snapshots of
probe `name` as `EnsembleResult.snapshots[name] = (generators, counts)`, with `generators`
of shape $(T, E, N, 2 w_q)$ (trajectories, evaluations) and `counts[t, e]` $= N$, or $-1$
for an evaluation after a `stop_when_purified` stop, where nothing is written.

These raw words are one basis of the row space: the one the simulation produced through
the pivot choices of every measurement. This engine reproduces them bit for bit for the same
spec, seed and trajectory, in any run (Proposition 13; the tests compare raw words across
runs), but another program or another engine version may produce another basis of the same
space. So identities and comparisons across implementations use the canonical form (v1.4
§5: "a backend-dependent raw tableau digest need not match across implementations; the
canonical phase-free row-space digest must").

### 3.7 Trajectory indices

A run of $T$ rows takes an array `indices` of $T$ distinct non-negative integers in any
order. Row $t$ simulates trajectory $u = $ `indices[t]`: it runs on graph
$\lfloor u / C\rfloor$ of a collection with $C$ = `circuits_per_graph` (graph 0 for a single
graph), draws its circuit from stream $u$ of the circuit key, and draws each sampled-probe
evaluation at layer $\ell$ from stream $\mathrm{mix}(u, h, \ell)$ of the probe key, $h$ the
hash of the probe's name (the `correlated_erasure` probe: stream
$\mathrm{mix}(\mathrm{mix}(u, h, \ell), m, 0)$ per size $m$). A range run is the case `indices` $= (f, f+1, \dots, f+T-1)$ with
$f$ = `first_trajectory`. `result.trajectories` is `indices`, in the given order.

## 4. Propositions and proofs

All claims below are derived here unless a citation is given; the stabilizer facts are
standard, and the proofs are written out so that every step can be checked.

### 4.1 The representation

**Proposition 1.** For Paulis $P, Q$ on $N$ qubits:
(a) $\varphi(PQ) = \varphi(P) + \varphi(Q)$;
(b) $PQ = (-1)^{\langle \varphi(P), \varphi(Q)\rangle}\, QP$;
(c) $\varphi(P) = \varphi(Q)$ if and only if $P = \lambda Q$ for some $\lambda \in \lbrace\pm1, \pm i\rbrace$.

*Proof.* On one qubit, $ZX = -XZ$ gives
$X^x Z^z X^{x'} Z^{z'} = (-1)^{z x'} X^{x + x'} Z^{z + z'}$, where $X^2 = Z^2 = I$ lets the
exponents be read mod 2. With $\sigma(x, z) = i^{xz} X^x Z^z$ this is
$\sigma(x, z)\sigma(x', z') = \mu\, \sigma(x \oplus x', z \oplus z')$ for a phase $\mu$, which
is (a) on one qubit; the tensor product applies it qubit by qubit. Exchanging the factors
gives the sign $(-1)^{z' x}$ instead of $(-1)^{z x'}$, so
$\sigma(x,z)\sigma(x',z') = (-1)^{x z' + z x'} \sigma(x',z')\sigma(x,z)$; multiplying over
the qubits gives (b). For (c), $\varphi$ forgets exactly the scalar $\lambda$. ∎

### 4.2 Stabilizer groups as subspaces

**Proposition 2.** Let $\mathsf S$ be a stabilizer group on $N$ qubits. Then $\varphi$ is
injective on $\mathsf S$, $\widetilde{\mathsf S} = \varphi(\mathsf S)$ is an isotropic subspace
with $\lvert \mathsf S\rvert = 2^{\dim \widetilde{\mathsf S}}$, it is the row space of the vectors of
any generating set of $\mathsf S$, and the stabilized subspace has dimension
$2^{N - \dim \widetilde{\mathsf S}}$. In particular a pure state has $\dim \widetilde{\mathsf S} = N$.

*Proof.* If $\varphi(P) = \varphi(Q)$ for $P, Q \in \mathsf S$, then $P = \lambda Q$
(Proposition 1c), so $\lambda I = P Q^{-1} \in \mathsf S$. $\lambda = -1$ is excluded by
$-I \notin \mathsf S$, and $\lambda = \pm i$ too, since then $(\lambda I)^2 = -I \in \mathsf S$;
hence $\lambda = 1$: $\varphi$ is injective on $\mathsf S$. By Proposition 1a the image of a
group is closed under addition, so it is a subspace, and it is spanned by the images of
generators; injectivity gives $\lvert \mathsf S\rvert = \lvert \widetilde{\mathsf S}\rvert = 2^{\dim}$.
Elements of $\mathsf S$ commute, so by Proposition 1b all forms vanish: isotropic. The
projector onto the stabilized subspace is $\Pi = \lvert \mathsf S\rvert^{-1}\sum_{g \in \mathsf S} g$
(each $g$ fixes the subspace, and $\Pi^2 = \Pi$ because $g\mathsf S = \mathsf S$). Every
$g \ne I$ in $\mathsf S$ is a non-identity Pauli up to sign (injectivity), hence traceless, so
$\operatorname{tr}\Pi = 2^N / \lvert \mathsf S\rvert$. A pure state means
$\operatorname{tr}\Pi = 1$, that is $\lvert\mathsf S\rvert = 2^N$. ∎

**Lemma 3 (dimension count).** For any subspace $U \subseteq \mathbb{F}_2^{2N}$ and set of
qubits $A$ with complement $\bar A$:
(a) $\dim U_A = \dim U - \operatorname{rank}(U\vert_{\bar A})$;
(b) $\dim (U^{\perp})_A = 2\lvert A\rvert - \operatorname{rank}(U\vert_A)$, with $U^\perp$ the
symplectic complement in $\mathbb{F}_2^{2N}$;
(c) if $U$ is the phase-free group of a pure state, $U^\perp = U$.

*Proof.* (a) $U_A$ is the kernel of $\pi_{\bar A}$ restricted to $U$; rank–nullity. (b) For
$w \in \mathbb{F}_2^{2\lvert A\rvert}$ let $\bar w \in \mathbb{F}_2^{2N}$ be $w$ extended by zeros.
For every $v$, $\langle \bar w, v\rangle = \langle w, \pi_A v\rangle_A$ (only the coordinates of
$A$ contribute). So $\bar w \in U^\perp$ if and only if $w$ is in the symplectic complement of
$U\vert_A$ inside $\mathbb{F}_2^{2\lvert A\rvert}$, and $w \mapsto \bar w$ maps that complement
onto $(U^\perp)_A$ one to one. The form on $\mathbb{F}_2^{2\lvert A\rvert}$ is nondegenerate, so
the complement has dimension $2\lvert A\rvert - \dim U\vert_A$. (c) Isotropy gives
$U \subseteq U^\perp$, and nondegeneracy gives $\dim U^\perp = 2N - \dim U = N = \dim U$. ∎

### 4.3 Outcome independence (v1.4 §6.2)

**Proposition 4.** In a circuit of Clifford gates and single-qubit Pauli measurements
without feedback, applied to a pure stabilizer state, the phase-free group after every
operation is a function of the initial phase-free group and the operations alone: it does
not depend on the measurement outcomes or on any sign.

*Proof.* By induction over the operations; the state stays pure (a measurement projects a
pure state onto a pure state). *Clifford gate $G$.* $G P G^\dagger$ is a Pauli for every Pauli
$P$, and $\varphi(G P G^\dagger) = M_G \varphi(P)$ for a $2N \times 2N$ binary (symplectic)
matrix $M_G$ that depends on $G$ only: on the two qubits a two-qubit gate acts on, the
images of $X_a, X_b, Z_a, Z_b$ fix the columns of $M_G$, and Proposition 1a extends it
linearly. So $\widetilde{\mathsf S} \mapsto M_G \widetilde{\mathsf S}$. In the engine the
Clifford with index $16s + b$ applies the $4 \times 4$ matrix of its symplectic class
$s$ to the columns of its two qubits (`_kernel/tableau.py::apply_2q`, `sym = g >> 4`); the
four sign bits $b$ only change signs (v1.4 §9.1). *Measurement of a Pauli $M$ with outcome
$\pm 1$.* If $M$ commutes with all of $\mathsf S$, then $\varphi(M) \in \widetilde{\mathsf S}^\perp
= \widetilde{\mathsf S}$ (Lemma 3c), the outcome is determined and the state does not change.
Otherwise let $v_M = \varphi(M)$. The new state is
$\lvert\psi'\rangle \propto (I \pm M)\lvert\psi\rangle$. Every $g \in \mathsf S$ commuting with $M$
stabilizes it ($g(I \pm M)\lvert\psi\rangle = (I \pm M)g\lvert\psi\rangle$), and so does $\pm M$.
The map $g \mapsto \langle \varphi(g), v_M\rangle$ is a nonzero homomorphism
$\mathsf S \to \mathbb{F}_2$ (Proposition 1a and bilinearity), so its kernel has
$2^{N-1}$ elements, and $\pm M$ is not in it (it anticommutes with an element of the
abelian group $\mathsf S$, so it is not in $\mathsf S$). The group they generate is abelian,
does not contain $-I$ (that would put $\mp M$ in $\mathsf S$), has $2^N$ elements and
stabilizes $\lvert\psi'\rangle$. The Paulis that fix $\lvert\psi'\rangle$ form a
stabilizer group in the sense of §3.2: products and inverses of Paulis that fix a vector fix
it; $-I$ and $\pm iI$ fix no nonzero vector; and two of them, $P$ and $Q$, commute, since
$PQ = -QP$ would give $\lvert\psi'\rangle = PQ\lvert\psi'\rangle = -QP\lvert\psi'\rangle =
-\lvert\psi'\rangle$. This group contains the one above and, stabilizing a nonzero vector, has
at most $2^N$ elements (Proposition 2: the stabilized subspace has dimension
$2^{N - \dim}$, at least 1), so the two are equal, and

$$
\widetilde{\mathsf S}' = \big(\widetilde{\mathsf S} \cap v_M^{\perp}\big) \oplus \operatorname{span}(v_M),
$$

which contains no outcome and no sign. ∎

The engine realizes this update with
`_kernel/tableau.py::measure_1q_phase_free_tracked` (`_collapse_phase_free`): the rows
anticommuting with $M$ (bitset `anti`) other than a pivot row $p$ are replaced by their
sum with $p$, which commutes with $M$, and row $p$ by $m$. The $N - 1$ rows that now
commute with $M$ are independent and lie in $\widetilde{\mathsf S} \cap m^\perp$, which has
dimension $N - 1$, so they span it. The tracked version chooses the pivot to keep the
reference rows (`ref_rows`) a basis of their reference parts
([code-dimension.md](../observables/code-dimension.md) §2); every pivot choice gives the same
span, and the canonical form erases the choice. Consequence: the phase-free tableau of a
trajectory is fixed by its graph, gate sequence and measurement mask, which the phase-free
engine samples, and a signed simulator (Stim, or `Simulator(2n, phases=True)`) gives the
same canonical row space whatever outcomes it draws (§8, tests of Stim and of
`from_simulator`).

### 4.4 Uniqueness of the canonical form

**Proposition 5.** Every subspace $U \subseteq \mathbb{F}_2^d$ has, for each column order,
exactly one basis in reduced row echelon form. Its leads are
$L(U) = \lbrace \operatorname{lead}(v) : v \in U \setminus \lbrace 0\rbrace\rbrace$, and its row with lead
$p$ is the unique $v \in U$ with $\operatorname{lead}(v) = p$ and $v$ zero at every other
element of $L(U)$.

*Proof.* *Existence:* Gauss–Jordan elimination (§5.1) applies only row operations to a
generating set, so its nonzero output rows span $U$, and they satisfy conditions 1 and 2
of §3.4 by construction; they are independent because each has a 1 at its own lead where
all others are 0. *Leads.* Let $b_1, \dots, b_r$ be any RREF basis with leads
$p_1 < \dots < p_r$ (in the column order). A nonzero $v \in U$ is $\sum_{i \in J} b_i$ for a
nonempty $J$. Let $i_0$ be the smallest index in $J$. Every $b_i$ with $i \in J$ is zero
before its own lead $p_i \ge p_{i_0}$, so $v$ is zero before $p_{i_0}$; at $p_{i_0}$ only
$b_{i_0}$ has a 1 (condition 2), so $v$ has a 1 there: $\operatorname{lead}(v) = p_{i_0}$.
Hence $L(U) = \lbrace p_1, \dots, p_r\rbrace$, and the leads of any RREF basis are determined
by $U$. *Rows.* If $v, v' \in U$ both have lead $p$ and are zero on $L(U) \setminus \lbrace p\rbrace$,
then $w = v + v' \in U$ is zero on all of $L(U)$ (both are 1 at $p$). A nonzero $w$ would
have its lead in $L(U)$, where it is zero; so $w = 0$ and $v = v'$. The row $b_i$ of any
RREF basis has lead $p_i$ and is zero on the other leads, so it is this unique vector. ∎

**Corollary 6.** Two generating sets span the same space if and only if their canonical
forms are equal, and then their bytes (§3.5) and digests are equal. Two different spaces
on the same $N$ have different canonical forms, hence different bytes; distinct bytes
collide under SHA-256 only with the probability of a hash collision. The bytes include $N$,
so the same rows on another number of qubits are another space.

### 4.5 The physical code

**Proposition 7.** Let $U = \widetilde{\mathsf S}_t$ be the phase-free group of the joint state
and order the columns with the reference columns first ($x_q$ for $q \in R$, then $z_q$ for
$q \in R$) and the system columns after them in the order $x_0, \dots, x_{n-1}, z_0, \dots,
z_{n-1}$. Let $\mathcal B$ be the RREF basis for this order, $\mathcal B_R$ its rows whose
lead is a reference column and $\mathcal B_V$ the others. Then the rows of $\mathcal B_V$ are
zero on $R$, they span $U_V$ (the elements of $U$ with zero reference part), and
$\pi_V$ maps $\operatorname{span} \mathcal B_V$ one to one onto $W_t^{\mathrm{phys}}$. So
$r_t = \lvert \mathcal B_V\rvert$, and $\pi_V(\mathcal B_V)$ is a basis of $W_t^{\mathrm{phys}}$
in reduced row echelon form for the system order $x_0, \dots, x_{n-1}, z_0, \dots, z_{n-1}$
(the canonical order on the $n$ system qubits).

*Proof.* A row of $\mathcal B_V$ has its lead among the system columns, so it is zero on every
earlier column, which includes all reference columns. Let $v \in U_V$, written
$v = \sum_{b \in J} b$ over a subset $J$ of $\mathcal B$. For $b \in \mathcal B_R$ with lead
$p$, the $p$-coordinate of $v$ equals the coefficient of $b$ (condition 2), and it is 0
because $p$ is a reference column; so $J \subseteq \mathcal B_V$ and
$U_V = \operatorname{span} \mathcal B_V$. On $U_V$, $\pi_V$ only drops
coordinates that are zero, so it is injective. Its image is $W_t^{\mathrm{phys}}$: if
$P \in \mathsf S_t^{\mathrm{phys}}$ then $P \otimes I_R \in \mathsf S_t$ and
$\varphi(P \otimes I_R) = (\varphi(P), 0) \in U_V$; conversely an element of $U_V$ is
$\varphi(g)$ for some $g \in \mathsf S_t$ that acts as $\lambda I$ on $R$, so
$g = P \otimes I_R$ for a Pauli $P = \lambda \cdot (\text{system part})$, and
$P \in \mathsf S_t^{\mathrm{phys}}$. Restricting to $V$ keeps the system columns, in the
order $x_0, \dots, x_{n-1}, z_0, \dots, z_{n-1}$ that they have in the column order above, and
drops only zero coordinates; so the restricted rows keep their leads (which increase in that
order) and every lead column stays zero in the other rows: they are in RREF for it. ∎

**Proposition 8.** For the pure joint state, $k_t = n - r_t = S(R) = S(V)$.

*Proof.* Proposition 9 below with $A = V$ gives $S(V) = n - \dim U_V = n - r_t$
(Proposition 7), and $S(R) = S(V)$ for a pure state on $V \cup R$ (Corollary 10). ∎

This is the quantity the `code_dimension` probe reads as `popcount(ref_rows) - |R|`
([code-dimension.md](../observables/code-dimension.md)), and $r_0 = 0$, $k_0 = n$ for the Bell
pairs (v1.4 §5).

### 4.6 Entropies

**Proposition 9.** Let $\widetilde{\mathsf S}$ (dimension $r$) be the phase-free group of a
stabilizer state $\rho = 2^{-N}\sum_{g \in \mathsf S} g$ on $N$ qubits (the maximally mixed
state on the stabilized subspace; a pure state when $r = N$). For a set of qubits $A$,

$$
S(A) = \lvert A\rvert - \dim \widetilde{\mathsf S}_A = \lvert A\rvert - r + \operatorname{rank}\big(\widetilde{\mathsf S}\vert_{\bar A}\big)
$$

bits; it does not depend on the signs.

*Proof.* $\operatorname{Tr}_{\bar A} g = 0$ unless $g$ acts as $\pm I$ on $\bar A$ (a
non-identity Pauli factor is traceless), and then it is $2^{\lvert\bar A\rvert}\, g\vert_A$. So
$\rho_A = 2^{-\lvert A\rvert}\sum_{g \in \mathsf S_A} g\vert_A$, with $\mathsf S_A$ the elements of
$\mathsf S$ acting trivially on $\bar A$, a stabilizer group on $A$ with
$\varphi(\mathsf S_A) = \widetilde{\mathsf S}_A$ (restricted to $A$). By Proposition 2,
$2^{\lvert A\rvert} \rho_A / \lvert\mathsf S_A\rvert$ is the projector onto a subspace of dimension
$2^{\lvert A\rvert - \dim\widetilde{\mathsf S}_A}$, so $\rho_A$ is the maximally mixed state on it
and $S(A) = \lvert A\rvert - \dim\widetilde{\mathsf S}_A$. Lemma 3a gives the second form. ∎

**Corollary 10 (pure states).** If $r = N$, then
$S(A) = \operatorname{rank}(\widetilde{\mathsf S}\vert_A) - \lvert A\rvert = S(\bar A)$.

*Proof.* Lemma 3b and 3c: $\dim \widetilde{\mathsf S}_A = 2\lvert A\rvert - \operatorname{rank}(\widetilde{\mathsf S}\vert_A)$;
substitute in Proposition 9. Lemma 3a gives
$\dim\widetilde{\mathsf S}_A = N - \operatorname{rank}(\widetilde{\mathsf S}\vert_{\bar A})$ as well;
equating the two, $\operatorname{rank}(\widetilde{\mathsf S}\vert_A) - \lvert A\rvert = \operatorname{rank}(\widetilde{\mathsf S}\vert_{\bar A}) - \lvert\bar A\rvert$, which is $S(\bar A)$. ∎

The formula of Corollary 10 is the one the design states (design §2.4) and the one the
engine's entropy kernels use (`_kernel/entropy.py` module docstring, Fattal et al.,
quant-ph/0406168). v1.4 §5.4 takes the form $S_t(A) = \lvert A\rvert - r_t(A)$ for
$A \subseteq V$ from Gullans and Huse, Appendix E, Eqs. (E1)–(E3): that is Proposition 9,
since $\dim (\widetilde{\mathsf S}_t)_A = r_t(A)$ for $A \subseteq V$ (proof of
Proposition 11); for a pure state it equals the form of Corollary 10. `PhaseFreeTableau.entropy` uses Corollary 10 on the smaller side when the
state is pure and Proposition 9 otherwise.

### 4.7 Reference information

**Proposition 11 (v1.4 §5.4).** For the pure joint state and $A \subseteq V$, $B = V \setminus A$,

$$
I(A:R) = S(A) + S(R) - S(A \cup R) = k_t + 2\lvert A\rvert - n - r_t(A) + r_t(B).
$$

*Proof.* The elements of $\widetilde{\mathsf S}_t$ supported in $A \subseteq V$ are elements of
$U_V$, and $\pi_V$ maps them one to one onto the elements of $W_t^{\mathrm{phys}}$ supported in
$A$ (Proposition 7), so $\dim (\widetilde{\mathsf S}_t)_A = r_t(A)$ and Proposition 9 gives
$S(A) = \lvert A\rvert - r_t(A)$; likewise $S(B) = \lvert B\rvert - r_t(B)$. Purity gives
$S(A \cup R) = S(B)$ (Corollary 10) and $S(R) = k_t$ (Proposition 8). So
$I(A:R) = \lvert A\rvert - r_t(A) + k_t - \lvert B\rvert + r_t(B)$, and
$\lvert B\rvert = n - \lvert A\rvert$. ∎

v1.4 says the formula is derived there by substitution; this proof follows the same steps.
`PhaseFreeTableau.mutual_information(A, R)` computes the left side from the joint row space
(three entropies, Proposition 9 or Corollary 10); the tests check it against the right
side with $r_t(A)$ found by enumerating $W_t^{\mathrm{phys}}$ (§8).

**Proposition 12 (v1.4 §5.8).** $I(A:R) + I(V \setminus A : R) = 2k_t$ for every $A \subseteq V$.

*Proof.* Add Proposition 11 for $A$ and for $B$: the $r_t$ terms cancel and
$2\lvert A\rvert + 2\lvert B\rvert - 2n = 0$. ∎

### 4.8 Trajectory indices leave every stream unchanged

**Proposition 13.** For any arrays of distinct trajectory indices, row $t$ of a run on
`indices` equals the row of trajectory `indices[t]` in any other run of the same spec and
seed (a range run, another list, another thread count): the same graph, circuit, mask,
probe samples, snapshots, purification layer, final layer and measurement count.

*Proof.* In `_kernel/ensemble.py::run_ensemble`, the body of the parallel loop for row $t$
reads `u = indices[t]` and the run's read-only inputs (spec arrays, keys, probe tables), and
writes only row $t$ of the outputs. Every random draw of the row comes from a stream that is
a function of the key and $u$ alone: the circuit stream `stream_state(key_struct, u)`, the
sampled-probe streams `stream_state(key_probe, mix(u, phash[j], layer))`, and the
`correlated_erasure` streams derived from that index and the size; the graph is
$\lfloor u / C\rfloor$. Before this change the same expressions read `first + t`, which is
`indices[t]` for a range run, so a range run draws exactly what it drew before (the
pinned digests of `tests/test_randomness_contract.py` are unchanged). The tableau snapshot
draws nothing. Rows share no mutable state, so the result does not depend on which thread
runs which row, or on the order of `indices`. ∎

Two consequences the study relies on (v1.4 §7.3): *prefix invariance*, because layer
$\ell$ of trajectory $u$ consumes its stream after layers $1..\ell-1$ and before layer
$\ell + 1$, so raising `layers` appends draws without changing earlier ones; and
*checkpoint invariance*, because probes never draw from the circuit stream (the sampled
probes have their own streams, the snapshot none), so moving or adding a snapshot does not
change the trajectory.

## 5. Algorithms

All arithmetic is exact (GF(2) on `uint64` words); nothing is approximated and nothing
draws random numbers.

### 5.1 Reduced row echelon form for a column order

`_kernel/rowspace.py::rref(rows, cols, pivots)`. Input: $s$ packed rows of $W$ words
($W = 2 w_q$ for phase-free rows; writable, modified in place), a column order `cols` of packed positions (column $c$ is bit
$c \,\&\, 63$ of word $c \gg 6$), and room for the pivots.

1. $\mathit{rank} \leftarrow 0$.
2. For $j = 0, 1, \dots$ while $\mathit{rank} < s$: let $c$ = `cols[j]`. Find the first row
   $i \ge \mathit{rank}$ with bit $c$ set; if none, continue with $j + 1$.
3. Swap rows $i$ and $\mathit{rank}$. XOR row $\mathit{rank}$ into every other row (above
   and below) with bit $c$ set. Record `pivots[rank] = j`, and increment $\mathit{rank}$.
4. Return $\mathit{rank}$.

*Invariants.* After the step for column $c$ that found a pivot, column $c$ has a single 1,
in the pivot row; after a step that found none, column $c$ is zero in rows
$\mathit{rank}, \dots, s-1$. Later steps add later pivot rows, which are zero in column $c$
(they come from rows $\ge \mathit{rank}$), so both properties persist. At the end rows
$0..\mathit{rank}-1$ are in RREF for `cols` and rows $\mathit{rank}..s-1$ are zero in every
listed column (zero rows when every possibly nonzero column is listed). *Cost:*
$O(\lvert\mathrm{cols}\rvert \cdot s \cdot W)$ word operations: for a pure state on $N$ qubits,
$2N \cdot N \cdot 2\lceil N/64\rceil \approx N^3/16$.

### 5.2 Rank of a set of columns

`_kernel/rowspace.py::rank_of_columns(rows, count, cols, scratch)`: copy the rows into
`scratch` with every column outside `cols` cleared (a projection, which commutes with row
operations), then eliminate (`_rank_in_place`): each row in turn, if nonzero, takes its
lowest set bit as pivot and clears it from the rows after it. A row nonzero at its turn has
a pivot bit no later row has, so those rows are independent and their number is the rank.
*Cost:* $O(\mathit{count} \cdot W \cdot (1 + \mathit{rank}))$.

### 5.3 Canonicalization and serialization

`PhaseFreeTableau.__init__`, `from_words`, `from_simulator`, `from_bytes` all end in
`PhaseFreeTableau._init`:

1. Pack the rows into the snapshot layout of §3.6 (`tableau.py::_dense_to_words`:
   `np.packbits(..., bitorder="little")` per half, read as little-endian `uint64`; or take
   the engine words, checked by `_checked_words`: shape $(s, 2w_q)$, padding bits zero).
2. Run §5.1 with the canonical order (`tableau.py::_canonical_columns`: packed positions
   $0..N-1$, then $64 w_q .. 64 w_q + N - 1$), keep the first $\mathit{rank}$ rows. With no
   rows the rank is 0 and nothing else is built: a 16-byte rank-0 header costs nothing in
   proportion to its $N$, so a corrupted $N$ there cannot make reading it allocate memory in
   proportion to $N$ (the length check of `from_bytes` bounds $N$ only when $r > 0$).

`to_bytes` writes §3.5's header and the rows packed by
`_kernel/rowspace.py::pack_msb_first`: each byte of a half row is one byte of its
little-endian words with its bits reversed (a 256-entry table), and when $N$ is not a
multiple of 8 the Z half, which starts at bit $N$, straddles two output bytes; the result is
`np.packbits(rows, axis=1, bitorder="big")` of the 0/1 rows, which the tests check for every
$N \bmod 8$. The bytes are cached.
`digest` hashes them with `hashlib.sha256`. `from_bytes` checks the magic, $N \ge 1$,
$r \le 2N$, the length, the padding bits, rebuilds the space and refuses it unless
`to_bytes()` reproduces the input exactly.

### 5.4 Entropy, mutual information, isotropy

`PhaseFreeTableau.entropy(A)`: check the qubits (distinct, in $[0, N)$) and isotropy (a
non-isotropic space is no stabilizer group: `ValueError`); if the space is pure
($r = N$, isotropic) and $2\lvert A\rvert \le N$, return
$\operatorname{rank}(\widetilde{\mathsf S}\vert_A) - \lvert A\rvert$ (Corollary 10); otherwise return
$\lvert A\rvert - r + \operatorname{rank}(\widetilde{\mathsf S}\vert_{\bar A})$ (Proposition 9).
Ranks by §5.2 with the columns $x_q, z_q$ of the chosen qubits. `mutual_information(A, B)`
checks that $A$ and $B$ are disjoint and returns $S(A) + S(B) - S(A \cup B)$.
`is_isotropic` evaluates $\langle b_i, b_j\rangle$ for every pair of canonical rows with popcounts
(`_kernel/entropy.py::generators_commute`, $O(r^2 w_q)$), once.

### 5.5 The physical code

`PhaseFreeTableau.physical_tableau(n)`: if $n = N$ return the tableau itself; otherwise run
§5.1 on a copy of the canonical rows with the order "reference X columns, reference Z
columns, system X columns, system Z columns"; the rows whose pivot index is at least
$2(N - n)$ are $\mathcal B_V$ (pivots increase, so they are the last rows); keep the first
$\lceil n/64\rceil$ words of each half (the reference bits of $\mathcal B_V$ are zero) and build a
`PhaseFreeTableau` on $n$ qubits (which canonicalizes again; by Proposition 7 the rows are
already canonical). `physical_code(n)` checks isotropy and returns
`StabilizerCode.from_packed` of those rows: `num_stabilizers` $= r_t$,
`num_logical` $= k_t$.

### 5.6 The snapshot in the engine

`_kernel/entropy.py::joint_snapshot_rows(tab, n_qubits, n_rows, out)`: zero the `n_rows`
output rows; for each qubit $q$ and each half (X, Z), walk the set bits of the column's
stabilizer words ($\lceil n_\mathrm{rows}/64\rceil$ words, so a phase-tracking tableau's
destabilizer words are never read) with count-trailing-zeros, and set bit $q$ of word
$\mathrm{half} \cdot w_q + (q \gg 6)$ in each such row; return `n_rows`. *Cost:*
$O(N \cdot 2w_q + \mathrm{nnz})$ for nnz set bits.

`_kernel/ensemble.py::eval_probe` dispatches `PROBE_TABLEAU_SNAPSHOT` (12) to it with
$n_\mathrm{qubits} = n_\mathrm{rows} = N$ and the row's snapshot slot
`snap[t, evaluation]`; `run_ensemble` records the returned count in `snap_rows[t, c]` and
writes $-1$ there for evaluations after a `stop_when_purified` stop. The recorded value is
$N$ in both cases: a pure state on $N$ qubits has $N$ generators whatever the state
(`mipt._probe_fill`, rule `"qubits"`). `mipt._snapshot_shape` sizes the buffer
$(T, E, N, 2w_q)$; a spec has at most one snapshot probe (`code_snapshot` or
`tableau_snapshot`), so the kernel keeps one buffer.

### 5.7 Runs on indices

`MonitoredCircuit.run(trajectories=None, *, seed, first_trajectory=None, indices=None,
num_threads=None)`: with `indices`, `_trajectory_indices` refuses an array that is not 1-D,
non-integer entries, negative entries, repeated entries (each would be simulated and
counted twice) and, for a collection of graphs, indices at or beyond
`len(edges) * circuits_per_graph`; `trajectories` and `first_trajectory` must then be left
out. Without it, the indices are `first_trajectory .. first_trajectory + trajectories - 1`
as before. The int64 array goes to the kernel in place of the former scalar `first`.

### 5.8 Measured cost

One thread of an AMD Ryzen 9 9950X3D, Numba 0.68, 2026-10-17; purification picture,
Watts–Strogatz graph ($k = 4$, $\beta = 0.1$), `random_edges`. The snapshot cost is the
difference of one trajectory's time with a snapshot at every layer and without, divided by
the evaluations (medians of 7 runs); the canonicalization time is `from_words` of a stored
snapshot (copy, checks and §5.1). "Scrambled" is a dense state ($2n\lceil\log_2 n\rceil$
scramble gates, no measurements); "monitored" is depth $n$ at $p = 0.1$.

| $n$ ($N$) | snapshot per evaluation | canonicalization | bytes (§3.5) | zlib level 6, scrambled | zlib level 6, monitored (median; min) |
|---|---|---|---|---|---|
| 64 (128) | 8 µs scrambled, 4 µs monitored | 0.06 ms, 0.03 ms | 4,112 | about 2,570 | 1,465; 107 |
| 128 (256) | 34 µs, 18 µs | 0.23 ms, 0.15 ms | 16,400 | about 9,380 | 5,258; 230 |
| 256 (512) | 134 µs, 73 µs | 1.04 ms, 0.68 ms | 65,552 | about 35,300 | 17,446; 665 |

The minima are the Bell pairs at layer 0. The engine words of one snapshot take
$8 N \cdot 2 w_q$ bytes (65,536 at $N = 512$) in memory; that is also their uncompressed size
in `result.npz`, which `EnsembleResult.save` writes with `np.savez_compressed`, so on disk
they take less (`load_results` reports the compressed sizes).

The snapshot's cost per $N^2$ depends on the picture: measured by
`tests/test_estimates.py`'s method (one thread, scrambled states), about 0.0005 µs per $N^2$
in the purification picture, whose reference columns stay sparse, and 0.001 µs in the single
and no-reference pictures, whose columns are dense, for $N$ up to 3072. Above $N = 2048$ the
scattered writes alias in the cache when $N$ is a power of two: 0.0027 µs per $N^2$ at
$N = 4096$ and 0.0047 at 8192 (0.0013 at 6144). `api.py::_probe_us` prices a snapshot at
$2 + 0.004 N^2 \max(1, N/2048)^{1.5}$ µs: 3.9 to 16 times the measured cost at every point
measured (4.1 in the single picture at $n = 512$, 7.9 in the purification picture).

Many snapshots at once (§5.9), on 32 threads of the same machine: for a unit at $n = 256$
($N = 512$, Watts–Strogatz $k = 4$, $\beta = 0.1$, $p = 0.1$, 2048 layers) of 256 trajectories
with 13 snapshot evaluations, the joint canonical form, the physical form and both byte
strings take 0.85 to 0.93 ms per snapshot one at a time and 0.17 to 0.18 ms in batches
(about 5 times faster; the engine run itself took 0.62 to 0.70 s, against 2.8 to 3.1 s and
0.58 to 0.60 s); at $n = 128$, 0.23 ms and 0.049 ms (4.6 times; engine 0.11 s). The bytes are
the same.

### 5.9 Many row spaces at once

`PhaseFreeTableau.from_words_batch(words, counts, N)` takes $B$ generator sets, `words` of
shape $(B, R, 2w_q)$ and `counts` (item $b$ is `words[b, :counts[b]]`; a negative count gives
`None`), and `_kernel/rowspace.py::rref_batch` runs §5.1 with the canonical order on every
item in a `prange` loop on every Numba thread. `tableau.py::physical_tableaux(tableaux, n)` runs
§5.5's two eliminations on every item the same way (`physical_rows_batch`), and
`tableau.py::to_bytes_batch(tableaux)` packs every item's bytes (`pack_msb_first_batch`) and
keeps them in the objects, so `digest` does not pack again. `EnsembleResult.tableaux(e)` is
`from_words_batch` on `generators[:, e]` and `counts[:, e]`. The kernels of one row space
(`rref`, `rank_of_columns`, `pack_msb_first`, `_kernel/entropy.py::generators_commute`)
release the GIL, so a pool of Python threads can run them at once; the batch kernels keep it,
as the engine's parallel kernels do, because two parallel launches from two Python threads at
once abort the process under Numba's `workqueue` threading layer (with the GIL held they take
turns).

*Exactness.* Each item goes through the same compiled `rref` and `pack_msb_first` as one
object alone, on the same rows in the same order, so the results are the objects and bytes
of the one-at-a-time methods, bit for bit, at any thread count; the tests compare them.
*Memory.* The input is copied in chunks of at most 64 MB (`tableau._BATCH_BYTES`, at least
one item per chunk); the input is not modified. *Cost:* §5.1 per item, divided among the
threads; the objects themselves are made in a Python loop of $O(1)$ work per item plus a copy
of its canonical rows.

## 6. From the mathematics to the code

| mathematics | code |
|---|---|
| $\varphi$, column order $x_0..x_{N-1}, z_0..z_{N-1}$ (§3.1) | `tableau.py` module docstring; `_canonical_columns` |
| packed words, $w_q$ (§3.6) | `tableau.py::_dense_to_words`, `_words_to_dense`, `PhaseFreeTableau.words` |
| RREF for a column order (§3.4, §5.1) | `_kernel/rowspace.py::rref` |
| $\operatorname{rank}(\widetilde{\mathsf S}\vert_A)$ (§5.2) | `_kernel/rowspace.py::rank_of_columns` |
| canonical form (Proposition 5) | `PhaseFreeTableau.canonical()`, `PhaseFreeTableau._init` |
| bytes, digest (§3.5, §5.3) | `PhaseFreeTableau.to_bytes`, `from_bytes`, `digest`, `tableau.MAGIC`; the packing `_kernel/rowspace.py::pack_msb_first` |
| $r = \dim\widetilde{\mathsf S}$ | `PhaseFreeTableau.rank` |
| isotropy | `PhaseFreeTableau.is_isotropic` (`_kernel/entropy.py::generators_commute`); `is_pure` |
| same row space (Corollary 6) | `PhaseFreeTableau.same_space`, `==` |
| $S(A)$ (Proposition 9, Corollary 10) | `PhaseFreeTableau.entropy` |
| $I(A:B)$, $I(A:R)$ (Proposition 11) | `PhaseFreeTableau.mutual_information` |
| $W_t^{\mathrm{phys}}$, $r_t$, $k_t$ (Proposition 7, 8) | `PhaseFreeTableau.physical_tableau`, `physical_code` |
| engine tableau, snapshot rows (§3.6, §5.6) | `_kernel/entropy.py::joint_snapshot_rows`; `_kernel/ensemble.py::eval_probe` (`PROBE_TABLEAU_SNAPSHOT = 12`) |
| the probe | `mipt.py::tableau_snapshot`, `_KINDS["tableau_snapshot"]`, `_SNAPSHOT_KINDS`, `_snapshot_shape`, `_probe_fill` |
| a stored snapshot as a row space | `mipt.py::EnsembleResult.tableau`; its physical code `EnsembleResult.code` |
| trajectory $u$ of row $t$ (§3.7, Proposition 13) | `_kernel/ensemble.py::run_ensemble` (`u = indices[t]`); `mipt.py::MonitoredCircuit.run(indices=...)`, `_trajectory_indices` |
| a stabilizer state from a simulator, signs dropped | `PhaseFreeTableau.from_simulator` |
| many row spaces at once (§5.9) | `PhaseFreeTableau.from_words_batch`, `tableau.py::physical_tableaux`, `tableau.py::to_bytes_batch`, `mipt.py::EnsembleResult.tableaux`; kernels `_kernel/rowspace.py::rref_batch`, `physical_rows_batch`, `pack_msb_first_batch` |

Stored fields of a run with a `tableau_snapshot` probe named `name`:

| field | formula |
|---|---|
| `values[name][t, e]` | $N$, the number of generators of the joint state (also after a stop) |
| `snapshots[name][0][t, e, i]` | generator $i$ of the joint state of row $t$ at evaluation $e$, in the words of §3.6 (raw: basis as simulated) |
| `snapshots[name][1][t, e]` | $N$, or $-1$ where a stopped trajectory saved nothing |
| `tableau(t, e).digest()` | SHA-256 of the canonical bytes (§3.5) of $\widetilde{\mathsf S}_{\ell}$, the joint state of trajectory $u$ = `trajectories[t]` after $\ell$ = `layers(name)[e]` layers |
| `code(t, e).num_logical` | $k_\ell = n - r_\ell$ of that trajectory at that depth |
| `tableaux(e)[t]` | `tableau(t, e)`, or `None` where row $t$ had stopped and saved nothing |
| `trajectories[t]` | the trajectory index $u$ of row $t$ |

## 7. Worked examples

### (a) A Bell pair, written out

The canonical rows of $\lbrace XX, ZZ\rbrace$ are $(1,1,0,0)$ and $(0,0,1,1)$; packed most
significant bit first they are the bytes `0xC0` and `0x30`. Any other generating set of
the group, such as $\lbrace YY, XX\rbrace$, gives the same bytes.

```python
import hashlib
import itertools

import numpy as np
import stim

from sparsegf2 import Circuit, PhaseFreeTableau, Simulator, graphs, mipt

bell = PhaseFreeTableau(["XX", "ZZ"])
assert bell.canonical().tolist() == [[1, 1, 0, 0], [0, 0, 1, 1]]
data = bell.to_bytes()
assert data == b"SGF2RS1\x00" + (2).to_bytes(4, "little") * 2 + bytes([0xC0, 0x30])
assert bell.digest() == hashlib.sha256(data).hexdigest()
assert PhaseFreeTableau(["YY", "XX"]) == bell and PhaseFreeTableau.from_bytes(data) == bell
assert bell.entropy([0]) == 1 and bell.mutual_information([0], [1]) == 2
```

### (b) Proposition 5 by enumeration

A random space on 4 qubits: its leads are the lowest set bits (in the column order) of its
nonzero elements, and each canonical row is the one element with that lead which is zero
at the other leads. Rows are written as integers, bit $j$ = column $j$.

```python
def as_int(row):
    return sum(1 << j for j, bit in enumerate(row) if bit)


def span(rows):
    out = {0}
    for r in rows:
        out |= {s ^ r for s in out}
    return out


rng = np.random.default_rng(7)
rows = rng.integers(0, 2, size=(5, 8), dtype=np.uint8)
tab = PhaseFreeTableau(rows)
space = span([as_int(r) for r in rows])
leads = sorted({(v & -v).bit_length() - 1 for v in space if v})
assert len(space) == 2**tab.rank and len(leads) == tab.rank
mask = sum(1 << p for p in leads)
for p, row in zip(leads, tab.canonical(), strict=True):
    (only,) = [v for v in space if v & mask == 1 << p and v & ((1 << p) - 1) == 0]
    assert as_int(row) == only
other = (rng.integers(0, 2, size=(3, 5)) @ rows % 2).astype(np.uint8)  # dependent rows
assert PhaseFreeTableau(np.concatenate([rows[::-1], other])).digest() == tab.digest()
```

### (c) Outcome independence

The same circuit on two phase-tracking simulators whose random measurements are forced to
opposite outcomes (`coins`), and on a phase-free one: the outcomes and the signed states
differ, the phase-free groups are the same (Proposition 4).

```python
text = "\n".join(
    ["H 0 1 2 3 4 5", "CX 0 1 2 3 4 5", "M 1 3", "H 1", "CX 1 2 3 4 5 0", "M 0 2 4", "S 3"]
)
records, signed, groups = [], [], []
for phases, coin in ((True, 0), (True, 1), (False, 0)):
    sim = Simulator(6, phases=phases, seed=0)
    result = sim.run(Circuit(text), coins=[coin] * 5)
    if phases:  # a phase-free simulator has no outcomes and no signs at all
        records.append(list(result.measurements))
        signed.append([str(p) for p in sim.stabilizers()])
    groups.append(PhaseFreeTableau.from_simulator(sim))
assert records[0] != records[1] and signed[0] != signed[1]  # other outcomes and states ...
assert groups[0] == groups[1] == groups[2]  # ... the same phase-free group
```

### (d) An engine run checked by brute force

$n = 4$ system qubits Bell-paired with 4 references; every snapshot's group has
$2^{8} = 256$ elements, enumerated here. The checks: $S(A)$ for all 256 subsets of the 8
qubits (Proposition 9), $k_t = n - r_t$ equal to the `code_dimension` probe (Proposition 8),
$I(A:R)$ equal to v1.4 §5.4's formula (Proposition 11) and the complement identity
(Proposition 12).

```python
n = 4
spec = mipt.MonitoredCircuit(
    num_system=n,
    edges=graphs.complete(n),
    layers=6,
    measurement_rate=0.25,
    schedule="random_edges",
    reference="purification",
    checkpoints="every_layer",
    probes=[mipt.code_dimension(), mipt.tableau_snapshot(checkpoints="every_layer")],
)
res = spec.run(3, seed=2)
system, reference = list(range(n)), list(range(n, 2 * n))


def support(g, qubits):  # x bits 0..qubits-1, z bits qubits..2 qubits-1
    return (g | (g >> qubits)) & ((1 << qubits) - 1)


def entropy(group, qubits, region):  # |A| - log2 #{g : supp g within A}
    inside = sum(1 for g in group if support(g, qubits) & ~sum(1 << q for q in region) == 0)
    return len(region) - (inside.bit_length() - 1)


for t, layer in itertools.product(range(3), range(7)):
    tab = res.tableau(t, layer)
    group = span([as_int(r) for r in tab.canonical()])
    assert len(group) == 2 ** (2 * n)
    for size in range(2 * n + 1):
        for region in itertools.combinations(range(2 * n), size):
            assert tab.entropy(region) == entropy(group, 2 * n, region)
    low = (1 << n) - 1
    physical = {  # zero reference part, restricted to the system
        (g & low) | (((g >> (2 * n)) & low) << n)
        for g in group
        if support(g, 2 * n) >> n == 0
    }
    r = len(physical).bit_length() - 1
    k = n - r
    assert k == res.values["code_dimension"][t, layer] == tab.physical_code(n).num_logical
    for size in range(n + 1):
        for a in itertools.combinations(system, size):
            b = [q for q in system if q not in a]
            r_a = entropy(physical, n, a)  # |A| - r_t(A), so r_t(A) = |A| - this
            r_b = entropy(physical, n, b)
            info = tab.mutual_information(a, reference)
            assert info == k + 2 * len(a) - n - (len(a) - r_a) + (len(b) - r_b)
            assert info + tab.mutual_information(b, reference) == 2 * k
```

### (e) The stored state against Stim (v1.4 §9.2 (h))

The exported circuit of trajectory $u$ (Bell pairs, then per layer the gates, the
measurements and a `TICK`) run in Stim's signed `TableauSimulator`: after tick $\ell$, its
stabilizers with the signs dropped span the stored row space of layer $\ell$.

```python
spec = mipt.MonitoredCircuit(
    num_system=9,
    edges=graphs.watts_strogatz(9, 4, 0.3, seed=1),
    layers=8,
    measurement_rate=0.2,
    schedule="random_edges",
    reference="purification",
    probes=[mipt.tableau_snapshot(checkpoints="every_layer")],
)
res = spec.run(seed=5, indices=[4])
signed = stim.TableauSimulator(seed=0)
signed.set_num_qubits(spec.num_qubits)
layer = 0
for ins in spec.circuit(4, seed=5).to_stim().flattened():
    if ins.name != "TICK":
        signed.do(ins)
        continue
    rows = [np.concatenate(p.to_numpy()).astype(np.uint8) for p in signed.canonical_stabilizers()]
    assert PhaseFreeTableau(rows).digest() == res.tableau(0, layer).digest()
    layer += 1
assert layer == 9
```

### (f) A run on explicit indices

Graph collection of 3 graphs with 2 circuits each: trajectory $u$ runs on graph
$\lfloor u/2\rfloor$. A run on `[5, 0, 3]` gives exactly those rows of the full run, sampled
probes and snapshots included (Proposition 13).

```python
bank = [graphs.watts_strogatz(8, 4, 0.2, seed=(3, g)) for g in range(3)]
spec = mipt.MonitoredCircuit(
    num_system=8,
    edges=bank,
    circuits_per_graph=2,
    layers=10,
    measurement_rate=0.1,
    schedule="random_edges",
    reference="purification",
    probes=[mipt.code_dimension(), mipt.erasure_first_failure(5), mipt.tableau_snapshot()],
)
whole = spec.run(seed=1)
part = spec.run(seed=1, indices=[5, 0, 3])
assert part.trajectories.tolist() == [5, 0, 3] and part.graph_index.tolist() == [2, 0, 1]
for name in whole.values:
    assert np.array_equal(part.values[name], whole.values[name][[5, 0, 3]])
for row, u in enumerate([5, 0, 3]):
    assert part.tableau(row, 0) == whole.tableau(u, 0)
```

### (g) Many snapshots at once

The six trajectories of the run above at their one snapshot evaluation, canonicalized,
reduced to their physical codes and serialized in batches (§5.9): the same objects and bytes
as one at a time.

```python
from sparsegf2.tableau import physical_tableaux, to_bytes_batch

joint = whole.tableaux(0)
physical = physical_tableaux(joint, 8)
assert joint == [whole.tableau(t, 0) for t in range(6)]
assert to_bytes_batch(joint) == [whole.tableau(t, 0).to_bytes() for t in range(6)]
assert to_bytes_batch(physical) == [
    whole.tableau(t, 0).physical_tableau(8).to_bytes() for t in range(6)
]
assert [p.rank for p in physical] == [8 - k for k in whole.values["code_dimension"][:, 0]]
```

## 8. Validation map

Every check is exact (integer or bit equality); none is statistical.

| property | test | what is compared |
|---|---|---|
| canonical form = RREF (Proposition 5) | `tests/test_tableau.py::test_canonical_form_is_the_reduced_row_echelon_form` | against a pure-Python Gauss–Jordan on integers (`tests/_rowspace_oracle.py::rref`) for $N$ = 1 to 130, with dependent, repeated and zero rows; for $N \le 4$ also against the characterization by enumeration |
| basis independence (Corollary 6) | `test_tableau.py::test_every_basis_of_a_space_gives_the_same_bytes_and_digest`, `test_pauli_strings_dense_rows_and_words_agree` | random invertible changes of basis, dependent and zero rows, row order, signs, three constructors: equal objects, bytes, digests |
| different spaces differ | `test_tableau.py::test_different_spaces_differ` | a larger, a smaller and a wider space: three digests |
| byte format, digest | `test_tableau.py::test_byte_format_and_digest`, `test_empty_and_zero_rows` | against the format written out bit by bit in the test and against `np.packbits` of the canonical rows, for every $N \bmod 8$ and one to three words per half ($N$ = 1 to 9, 12, 15, 63, 64, 65, 67, 70, 127, 130); `hashlib.sha256` |
| a rank-0 header with a large $N$ | `test_tableau.py::test_a_rank_zero_header_costs_nothing_in_proportion_to_its_N` | $N = 2^{24}$: the rank-0 space, its bytes, and a traced memory peak below 1 MB (537 MB before) |
| `from_bytes` accepts only canonical bytes | `test_tableau.py::test_from_bytes_refuses_anything_but_canonical_bytes` | 9 malformed inputs (magic, length, header, rank, padding, row order, unreduced, dependent) refused; the round trip |
| kernels (§5.1, §5.2) | `test_tableau.py::test_rref_kernel_for_any_column_order`, `test_rank_of_columns_kernel` | random column orders and column subsets, against the integer reference; pivots; zero rows |
| isotropy | `test_tableau.py::test_isotropy_against_the_symplectic_form` | the symplectic form of every pair, computed on integers |
| entropies (Proposition 9, Corollary 10) | `test_tableau.py::test_entropy_of_pure_and_mixed_states_against_enumeration`, `test_mutual_information_of_every_disjoint_pair` | every region of random pure and mixed states from Stim, $N \le 8$, against $\lvert A\rvert - \log_2 \#\lbrace g : \operatorname{supp} g \subseteq A\rbrace$ over all $2^r$ elements |
| physical code, $I(A:R)$, complement (Propositions 7, 8, 11, 12) | `test_tableau.py::test_reference_information_of_joint_states` | Stim joint states, $n \le 4$: $W^{\mathrm{phys}}$ and $k = S(R)$ by enumeration, $I(A:R)$ against v1.4 §5.4's formula with $r(A)$ enumerated, $I(A:R) + I(B:R) = 2k$, `StabilizerCode.lost_information` |
| the same with several words per half row | `test_tableau.py::test_multiword_joint_states_against_integer_ranks` | Stim joint states at $N$ = 66, 80, 140 (purification) and 65, 66 (one reference): the physical rows against the integer reduction with the reference columns first, $r = N - \operatorname{rank}(U\vert_R)$, $S(X)$ of random regions on both sides of $N/2$ against Proposition 9 with integer ranks, 20 random $A$: $I(A:R)$ against those entropies and v1.4 §5.4's formula ($r(A)$ by integer ranks), the complement identity, `lost_information` |
| outcome independence (Proposition 4) | `test_tableau.py::test_from_simulator_drops_signs_and_outcomes` | phase-tracking simulators with different outcomes, a phase-free one and Stim: one row space |
| layer 0 = Bell pairs, $r_0 = 0$, $k_0 = n$ | `tests/test_tableau_snapshot.py::test_layer_zero_holds_the_bell_pairs` | the canonical rows written out in the test |
| every snapshot pure; $k_t = n - r_t$ = `code_dimension` = $S(R)$ | `test_tableau_snapshot.py::test_every_snapshot_is_a_pure_state_whose_code_has_the_probed_dimension` | the raw engine rows: $2n$ independent commuting rows (integer reference), at every layer, $n$ = 5 to 16 and, with two or three words per half row, 33, 40 and 70; odd and even, several rates and $\beta$; the buffer shape $(T, E, N, 2\lceil N/64\rceil)$ |
| physical code = native `code_snapshot` (v1.4 §9.2 (h), "reduced-code stabilizers") | `test_tableau_snapshot.py::test_the_physical_code_is_the_native_code_snapshot` | same seed and trajectories, purification, single and no reference, $n$ = 10, and with several words per joint half row $n$ = 33, 40, 70 (purification), 64, 65 (single) and 65 (none): equal canonical forms (integer reference), equal probe values, $k_t$ = `code_dimension` = $S(R)$, both buffer shapes |
| stored row space = Stim (v1.4 §9.2 (h)) | `test_tableau_snapshot.py::test_snapshots_equal_stim_on_the_exported_circuit` | the exported circuit in Stim's signed simulator, every layer, 13 cases ($n$ = 5 to 16, and with several words per half row $N$ = 66, 80, 140, 65 and 70; odd $n$, rates 0.05 to 0.3, $\beta$ = 0 to 1, three pictures) |
| entropies of engine snapshots | `test_tableau_snapshot.py::test_entropies_of_engine_snapshots_against_enumeration` | $n$ = 3, 4: every region of the $2n$ qubits, §5.4's formula and the complement identity by enumeration |
| the snapshot kernel | `test_tableau_snapshot.py::test_the_kernel_writes_every_stabilizer_row_and_never_a_destabilizer` | `joint_snapshot_rows` on phase-tracking tableaux ($N$ = 3 to 130) against `stabilizer_arrays` |
| indices (Proposition 13) | `test_tableau_snapshot.py::test_a_run_on_indices_equals_those_rows_of_a_range_run`, `test_indices_on_a_single_graph_reach_any_trajectory`, `test_bad_indices_are_refused` | rows 5, 2, 9 of a collection run (a probe of every kind with a stream of its own: `erasure_lost_profile`, `erasure_first_failure` and both `correlated_erasure` branches; snapshots, layers, counts); an index of $10^{12}$; 8 refusals |
| prefix, checkpoint, thread invariance | `test_tableau_snapshot.py::test_prefixes_checkpoints_and_threads_do_not_change_a_trajectory` | layers 40 against 80, moved and dropped snapshots, 1 thread against 4 |
| stops | `test_tableau_snapshot.py::test_after_a_stop_nothing_is_saved` | counts $-1$, value $N$, `tableau` and `code` refuse, the buffer stays zero |
| save, load, concatenate, replay | `test_tableau_snapshot.py::test_save_load_concatenate_and_replay`, `test_a_result_without_the_probe_saves_as_before` | equal tableaux and digests after a round trip; a merged result against the full run; a replayed snapshot; a file without the probe has the same array names as before and the `code_snapshot` shape |
| one snapshot probe per spec; the JSON probe | `test_tableau_snapshot.py::test_one_snapshot_probe_per_spec`, `test_the_json_probe_its_layout_summary_and_saved_file` | refusals; the layout's exact shapes against the saved file ($N$ = 12 and 80: `[3, 2, 80, 4]`); the summary note; `load_results` of the saved run: its probe summaries equal `run_ensemble`'s, and its storage lists the snapshot arrays |
| many row spaces at once (§5.9) | `test_tableau.py::test_batch_canonicalization_equals_one_at_a_time`, `test_batch_canonicalization_checks_its_input`, `test_batch_physical_tableaux_equal_one_at_a_time`, `test_batch_bytes_equal_one_at_a_time`, `test_the_kernels_release_the_gil`; `test_tableau_snapshot.py::test_tableaux_of_an_evaluation_are_the_tableaux_one_by_one` | equal words, objects and bytes to the one-at-a-time methods ($N$ = 1 to 140; with chunks of one item, of a few and of all; on 1, 2 and every thread; with `None` items and stopped rows), the input untouched, refusals; the one-row-space kernels compiled with `nogil`, the batch kernels `parallel` and holding the GIL |
| the snapshot's price | `tests/test_estimates.py::test_every_probe_costs_at_most_its_estimate` | `tableau_snapshot` in the purification and single pictures at $n = 512$, timed against `_probe_us` with the file's normalized method |
| streams unchanged | `tests/test_randomness_contract.py` (unchanged) | the pinned digests of contract 2 |

## 9. Limitations and departures

* **No departure from v1.4.** The probability laws, streams, generators and the Clifford
  enumeration are untouched (Proposition 13; the pinned digests pass).
* **Conventions chosen here** (v1.4 leaves them open): the canonical form (RREF in the
  column order of §3.1), the byte format and SHA-256 (§3.5). A change to either is a new
  format: a new magic.
* **What the digest covers.** The phase-free group and $N$ only: not the signs (by design,
  v1.4 §5.1), not the depth or the trajectory (the store keys them), and not a relabeling
  of the qubits (the same state with the qubits renumbered has another digest).
* **After a stop.** With `stop_when_purified`, the state is not evolved after the stop and
  nothing is saved there (v1.4 §8.1: "no unsaved joint-tableau state after stopping can be
  claimed"); `tableau()` refuses such an evaluation. Campaigns that need snapshots after
  purification set `stop_when_purified=False` (v1.4 §9).
* **Memory.** A run holds all its snapshots in memory: $T \cdot E \cdot 16 N w_q$ bytes,
  64 KiB per snapshot at $N = 512$; `api._result_bytes` counts them and the JSON service
  refuses results above `MAX_RESULT_GB`. Long campaigns run in units (the study's runner).
* **Sizes tested.** The engine-level checks (Stim, the native code, `code_dimension`, the
  JSON layout) run at $N$ up to 140, with one to three 64-bit words per half row and with a
  system half of fewer words than the joint one ($n$ = 33, 40, 64, 65, 70); the kernels and
  the canonical form are checked up to $N = 130$. Study runs at $n \ge 33$ in the
  purification picture use the same index arithmetic with more words.
* **Many snapshots.** Canonicalize many snapshots with `EnsembleResult.tableaux`,
  `physical_tableaux` and `to_bytes_batch` (§5.9), not with a Python loop over `tableau`:
  the results are the same and the work runs on every thread.
* **One snapshot probe per spec** (design §2.1): a `code_snapshot` and a `tableau_snapshot`
  cannot share a run; the tableau's physical code (`EnsembleResult.code`) replaces the
  former.
* **Additions beyond design §2.4**, none of which changes its interface:
  `PhaseFreeTableau.from_simulator`, `physical_tableau`, `is_pure`, `words`, `==` and
  `hash`, `tableau.MAGIC`, and the batch path of §5.9 (`from_words_batch`,
  `physical_tableaux`, `to_bytes_batch`, `EnsembleResult.tableaux`). `entropy` also accepts mixed (non-pure) isotropic spaces,
  with Proposition 9; for a pure state it is the design's
  $\operatorname{rank}(\widetilde{\mathsf S}\vert_A) - \lvert A\rvert$.
