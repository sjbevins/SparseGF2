# Lost information

How SparseGF2 computes $\ell_C(A) = I(A:R)$, the logical information
destroyed by erasing a region $A$, whether that erasure is correctable, and
the mean loss over random regions of every size.

The setting and notation are those of
[random-region-erasure.md](random-region-erasure.md) §1.1: a code $C$ with
parameters $[[n, k_C]]$ on the system qubits $[n]$, stabilizer group $S$ with
$s = n - k_C$ independent generators (check matrix
$H \in \mathbb{F}_2^{s \times 2n}$), normalizer $N(S)$, and a reference $R$
that purifies the $k_C$ logical qubits. That document builds $P_C(m)$ and
$\Phi_C(e)$ on top of $\ell_C$; [code-dimension.md](code-dimension.md) covers
$k_C$ and the reference rows. Function references are `path::function`,
relative to `src/sparsegf2/`. This describes version 3.0.0.dev12.

## Contents

1. [Definition](#1-definition)
2. [What the simulator evaluates, and why it equals the definition](#2-what-the-simulator-evaluates-and-why-it-equals-the-definition)
3. [From the kernel to the code you run](#3-from-the-kernel-to-the-code-you-run)
4. [Cost and accuracy](#4-cost-and-accuracy)
5. [Worked example](#5-worked-example)
6. [Pitfalls](#6-pitfalls)

---

## 1. Definition

For a region $A \subseteq [n]$ of system qubits, the **lost information** is

$$
\ell_C(A) \equiv I(A:R) = S(A) + S(R) - S(AR),
$$

in bits, for the pure joint state of the system and $R$.

**Correctability.** The erasure of $A$ (at known locations) is perfectly
correctable if and only if $\ell_C(A) = 0$. This is the decoupling condition:
$I(A:R) = 0$ means $\rho_{AR} = \rho_A \otimes \rho_R$, so all correlations
with $R$ sit in $\bar A$ and a recovery acting on $\bar A$ restores them. In
stabilizer language (§2), it is the erasure form of the Knill–Laflamme
condition: no nontrivial logical operator is supported on $A$.

**Range and meaning.** $\ell_C(A)$ is an integer with
$0 \le \ell_C(A) \le 2\min(|A|, k_C)$, since
$I(A:R) \le 2\min\big(S(A), S(R)\big)$. It equals the number of independent
logical Pauli operators, modulo stabilizers, that are supported on $A$ (§2):

* $\ell_C(A) = 0$: correctable;
* $\ell_C(A) = 2k_C$: everything is lost (the reference is uncorrelated with
  the remaining qubits $\bar A$);
* odd values occur. For the bit-flip repetition code ($ZZI$, $IZZ$), erasing
  qubit 0 exposes $\bar Z = Z_0$ but not $\bar X$, so $\ell = 1$: the logical
  qubit is dephased, not destroyed.

An optimal decoder told which qubits were erased recovers the logical state
with probability $2^{-\ell_C(A)}$: each lost bit randomizes one logical Pauli
(`observables.py::recovery_probability`).

**Complementarity.** $\ell_C(A) + \ell_C(\bar A) = 2k_C$, which is
$I(A:R) + I(\bar A:R) = 2S(R)$ for the pure state on $A\bar A R$. In
particular, everything is lost by erasing $A$ exactly when its complement is
correctable (the cleaning lemma, [random-region-erasure.md](random-region-erasure.md) §2.2).

**Monotonicity.** $A \subseteq B$ implies $\ell_C(A) \le \ell_C(B)$ (data
processing). This is what makes first-failure sampling work
([random-region-erasure.md](random-region-erasure.md) §2.3–2.4).

**The loss profile.** For each size $m$, the mean loss over uniformly random
$m$-subsets,

$$
\lambda_C(m) = \frac{1}{\binom{n}{m}} \sum_{|A| = m} \ell_C(A),
$$

satisfies $\lambda_C(0) = 0$, $\lambda_C(n) = 2k_C$, and
$\lambda_C(m) + \lambda_C(n - m) = 2k_C$ (the complement of a uniform
$m$-subset is a uniform $(n-m)$-subset). It bounds the failure probability
from above, $P_C(m) \le \lambda_C(m)$, because $\ell_C \ge 1$ whenever it is
positive. The probe `erasure_lost_profile` estimates $\lambda_C(m)$ for every
$m$.

---

## 2. What the simulator evaluates, and why it equals the definition

**Formula (1).** With $S(X) = |X| - \dim S_X$ for system regions and
$\dim S_X = s - \operatorname{rank}(H_{\bar X})$:

$$
\ell_C(A) = 2|A| - s - \operatorname{rank}(H_A) + \operatorname{rank}(H_{\bar A}).
$$

By symplectic duality on $A$ (the elements of $V^{\perp}$ supported on $A$ are
the complement of the projection $V|_A$), this equals both

$$
\ell_C(A) = \dim N(S)_A - \dim S_A \qquad \text{(i)}
$$

$$
\ell_C(A) = \operatorname{rank}\big(N(S)|_A\big) - \operatorname{rank}\big(S|_A\big). \qquad \text{(ii)}
$$

Here $V_A$ is the subspace supported on $A$ and $V|_A$ the projection. (i) is
the count of logical operators on $A$, and (ii) is what the kernels compute.
[random-region-erasure.md](random-region-erasure.md) §2.1–2.2 has the
derivation in full. Complementarity follows from (i) and (ii) together.

The simulator evaluates $\ell_C(A)$ along three paths. All of them compute the
ranks of the $2|A|$ columns of $A$ only, never of $\bar A$.

**(a) The engine, on the live tableau.** The tableau rows generate the
stabilizer group $G$ of the joint state. The marked rows (`ref_rows`, the rows
with support on $R$) have linearly independent R-parts, and the engine keeps
them so ([code-dimension.md](code-dimension.md) §2). Write $\operatorname{rank}(X)$
for the rank of the columns of a qubit set $X$ over all rows, and
$\operatorname{rank}_{\mathrm{free}}(A)$ over the unmarked rows only. Then

$$
\ell_C(A) = \operatorname{rank}(A) - \operatorname{rank}_{\mathrm{free}}(A)
$$

(`_kernel/entropy.py::lost_information_rows`). *Proof.* With $p$ marked rows,
$S(R) = p - |R|$. Count the linear relations among the rows restricted to
$R \cup A$. A relation has zero R-part, so it involves no marked row (their
R-parts are independent); it is a relation among the unmarked rows restricted
to $A$. So the rows restricted to $R \cup A$ have rank
$\operatorname{rank}(R \cup A) = p + \operatorname{rank}_{\mathrm{free}}(A)$. With
$S(X) = \operatorname{rank}(X) - |X|$:

$$
I(A:R) = \big(\operatorname{rank}(A) - |A|\big) + (p - |R|) - \big(p + \operatorname{rank}_{\mathrm{free}}(A) - |A| - |R|\big) = \operatorname{rank}(A) - \operatorname{rank}_{\mathrm{free}}(A). \quad \blacksquare
$$

This is form (ii): all rows restricted to the system span $N(S)$ and the
unmarked rows span $S$.

**(b) Any `Simulator`, without the invariant.**
`observables.py::lost_logical_information(sim, erased, reference)` evaluates
$S(R) + S(E) - S(R \cup E)$ with three region entropies
(`_kernel/entropy.py::mutual_information_with`). Each is
$\operatorname{rank} - |X|$ on whichever of $X$ and its complement is smaller
(`_kernel/entropy.py::region_entropy`). This works on any state, with any
choice of qubits as reference.

**(c) A saved code.** `codes.py::StabilizerCode` stacks the generators of $S$
and a basis of $N(S)$ into one tableau and marks the $N(S)$ rows
(`StabilizerCode._init_packed`). Then `lost_information(A)` calls the same
`lost_information_rows`: over all rows the rank is $\dim N(S)|_A$, over the
unmarked rows $\dim S|_A$. That is form (ii) directly, with no reference
qubits at all.

**The profile, incrementally.** `_kernel/entropy.py::erasure_profile_rows(tab, ws, order, ref_rows, out)`
adds the qubits of an ordering one at a time to two incremental bases (all
rows, and rows outside `ref_rows`, via `basis_add_qubit` and
`basis_add_qubit_without`). It records
$\ell_C(A_m) = \operatorname{rank}(A_m) - \operatorname{rank}_{\mathrm{free}}(A_m)$
for every prefix $A_m$, $m = 0..n$, in one pass. Prefixes of a uniformly random
ordering are uniformly random $m$-subsets
([random-region-erasure.md](random-region-erasure.md) §2.4), so the mean over
orderings is an unbiased estimate of $\lambda_C(m)$ for every $m$ at once.

---

## 3. From the kernel to the code you run

**Data layout.** The column-major tableau `tab` (`tab[2q]`, `tab[2q + 1]`: the
X and Z bitsets of qubit $q$ over the generators) and the bitset `ref_rows`
are described in [random-region-erasure.md](random-region-erasure.md) §3.1.

**Kernels.**

| function (`_kernel/`) | computes | cost |
|---|---|---|
| `entropy.py::lost_information_rows(tab, ws, erased, ref_rows, scratch)` | $\ell_C(A)$, form (a) | two eliminations of $2\lvert A\rvert$ bitsets of $W$ words, $O(\lvert A\rvert^2 W)$ |
| `tableau.py::columns_rank`, `columns_rank_without` | the two ranks (the second with the marked rows masked out) | as above |
| `entropy.py::erasure_profile_rows(tab, ws, order, ref_rows, out)` | $\ell_C(A_m)$ for all prefixes | $O(n \cdot \mathrm{rank} \cdot W)$ per ordering |
| `entropy.py::mutual_information_with` | $S(R) + S(E) - S(R \cup E)$, form (b) | three entropies on the smaller sides |

$W = \lceil N/64\rceil$ words for $N$ qubits in all (system plus reference).

**The engine.** At a checkpoint, `_kernel/ensemble.py::run_ensemble` calls
`_kernel/ensemble.py::eval_probe`, which branches on the probe kind:

* `lost_information`: `row[0] = lost_information_rows(tab, ws, region, ref_rows, erase_scratch)`,
  evaluated whatever $k_C$ is (it is 0 when $k_C = 0$);
* `erasure_lost_profile`: zeros when $k_C = 0$ (no random draws). Otherwise $M$
  times a Fisher–Yates shuffle of the system (`_shuffle`), then
  `erasure_profile_rows`, and the stored row is the mean over the $M$
  orderings, $n + 1$ numbers. Every evaluation draws from its own random
  stream, keyed by the seed, the trajectory, the probe's name and the layer
  ([random-region-erasure.md](random-region-erasure.md) §3.6), so a replayed
  trajectory reproduces it.

With `stop_when_purified`, evaluations after a trajectory's purification
layer hold 0 for both probes, which is exact for $k_C = 0$ (`mipt.py::_KINDS`).

**Python.**

* `mipt.lost_information(region, name=None, *, checkpoints=None)`: one fixed
  region of system qubits. `values[name]` has shape
  `(trajectories, evaluations)`, and the default name is
  `lost_information[...]` with the region listed.
* `mipt.erasure_lost_profile(samples=10, name=None, *, checkpoints=None)`:
  `values[name]` has shape `(trajectories, evaluations, n + 1)`.
* `observables.lost_logical_information(sim, erased, reference)`,
  `erasure_is_correctable`, `recovery_probability` (form (b), any
  `Simulator`). `observables.erasure_profile(sim, system, reference, samples)`
  returns the per-ordering losses `lost` (samples × (n + 1)), their mean and
  the correctable fraction per size.
* `StabilizerCode.lost_information(A)` and `StabilizerCode.is_correctable(A)`
  (form (c)), on a code from `EnsembleResult.code(t, c)`,
  `StabilizerCode.from_simulator`, or your own generators.
* `analysis.bernoulli_erasure_curve(per_size, p)` averages a per-size curve
  over independent erasure with probability $p$. Applied to $\lambda_C(m)$ it
  gives the mean bits lost at rate $p$; applied to a failure curve it gives a
  failure probability.

**JSON (CLI, MCP).** Probes `{"kind": "lost_information", "region": "0:4"}`
and `{"kind": "erasure_lost_profile", "samples": 10}`. A region is a list or a
range string. For a single circuit, `run_circuit` takes
`{"kind": "lost_logical_information", "erased": [0, 1], "reference": "16:32"}`,
`erasure_is_correctable`, `recovery_probability` and
`{"kind": "erasure_profile", "system": "0:16", "reference": "16:32", "samples": 10}`.
The ensemble summary shows the profile as a curve over $m$ (`index`) at the
last checkpoint, plus a `vs_depth` grid.

---

## 4. Cost and accuracy

* $\ell_C(A)$ is an exact integer on every path. A fixed region costs two
  eliminations over its $2|A|$ columns of $W$ words, $O(|A|^2 W)$, which
  `estimate_ensemble` prices at $4 + 0.2|A|W$ µs per evaluation
  (`api.py::_probe_us`).
* `erasure_lost_profile` costs one incremental pass per ordering, priced by
  `estimate_ensemble` at about $M(0.8n + 0.05nW)$ µs per evaluation, a
  pessimistic fit (`api.py::_probe_us`).
* The profile is a mean over $M$ orderings. Each $\hat\lambda_C(m)$ is
  unbiased, and its sampling error is not stored, since only the mean is.
  Across trajectories, the standard error of the ensemble mean
  (`analysis.mean_and_error`) includes it. For per-ordering values on one
  state, use `observables.erasure_profile`, which returns them all.
* As for $P_C$, the sizes are correlated (nested prefixes), and
  $\hat\lambda_C(n) = 2k_C$ holds exactly in every sample.

---

## 5. Worked example

**Small codes, by hand.** The repetition code shows an odd loss, and the
$[[4, 2, 2]]$ code shows complementarity ($\ell(A) + \ell(\bar A) = 2k = 4$):

```python
from sparsegf2 import StabilizerCode

rep = StabilizerCode(["ZZ_", "_ZZ"])  # bit-flip repetition code, k = 1
print([rep.lost_information(a) for a in ([0], [0, 1], [0, 1, 2])])  # [1, 1, 2]
c422 = StabilizerCode(["XXXX", "ZZZZ"])  # [[4, 2, 2]]
print([c422.lost_information(a) for a in ([0], [0, 1], [0, 1, 2])])  # [0, 2, 4]
assert c422.is_correctable([3]) and not c422.is_correctable([2, 3])
```

**One trajectory, four ways.** A Watts–Strogatz circuit with $n = 16$ in the
purification picture, with the fixed region $A = \lbrace 0, 1, 2, 3\rbrace$
probed at every layer. The engine's values are compared with:

* the generic entropy formula on the replayed state, at every `TICK` (form (b));
* the saved code (form (c));
* formula (1) written out with numpy and an independent GF(2) rank.

The trajectory is the first that holds a code at the end. Its numbers depend
on the version's random streams and are not quoted.

```python
import numpy as np

from sparsegf2 import Simulator, graphs, mipt, observables

n = 16
region = [0, 1, 2, 3]
spec = mipt.MonitoredCircuit(
    num_system=n,
    edges=graphs.watts_strogatz(n, 4, 0.1, seed=1),
    layers=64,
    measurement_rate=0.1,
    schedule="random_edges",
    reference="purification",
    checkpoints="every_layer",
    probes=[
        mipt.code_dimension(),
        mipt.lost_information(region, name="ell_A"),
        mipt.erasure_lost_profile(samples=50, checkpoints="final"),
        mipt.code_snapshot(checkpoints="final"),
    ],
)
res = spec.run(16, seed=7)
k_final = res.values["code_dimension"][:, -1]
t = int(np.flatnonzero(k_final > 0)[0])  # the first trajectory that holds a code
k = int(k_final[t])
ell_engine = res.values["ell_A"][t]  # l(A) after every layer 0..64
print(f"trajectory {t}, k = {k}; l(A) by layer:", ell_engine.astype(int).tolist())
assert ell_engine[0] == 2 * len(region)  # at layer 0 every qubit is Bell-paired with R

refs = spec.reference_qubits
sim = Simulator(spec.num_qubits, phases=False)
run = sim.run(
    spec.circuit(t, seed=7),
    observe=lambda s, tick: observables.lost_logical_information(s, region, refs),
)
ell_library = np.array([value for _, value in run.observations])  # tick L = after layer L
assert np.array_equal(ell_library, ell_engine)
```

Formula (1) with numpy, on the code saved at the final layer, checked
against `StabilizerCode` and the replayed state for the fixed region and 200
random ones. The check also covers complementarity, correctability and the
recovery probability:

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


code = res.code(t, 0)
H = code.stabilizer_matrix()  # (s, 2n): X bits, then Z bits
s = gf2_rank(int(row) for row in H.astype(np.int64) @ (1 << np.arange(2 * n)))
col = [int(c) for c in H.T.astype(np.int64) @ (1 << np.arange(H.shape[0]))]


def ell_formula(a):
    """Formula (1): 2|A| - s - rank(H_A) + rank(H_Abar)."""
    abar = [q for q in range(n) if q not in a]
    rank_a = gf2_rank([col[q] for q in a] + [col[n + q] for q in a])
    rank_abar = gf2_rank([col[q] for q in abar] + [col[n + q] for q in abar])
    return 2 * len(a) - s - rank_a + rank_abar


assert s == n - k
assert ell_formula(region) == code.lost_information(region) == ell_engine[-1]
rng = np.random.default_rng(1)
for _ in range(200):
    a = sorted(int(q) for q in rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False))
    abar = [q for q in range(n) if q not in a]
    ell = code.lost_information(a)
    assert ell == ell_formula(a) == observables.lost_logical_information(sim, a, refs)
    assert ell + code.lost_information(abar) == 2 * k  # complementarity
    assert code.is_correctable(a) == (ell == 0)
    assert observables.recovery_probability(sim, a, refs) == 2.0**-ell
```

The loss profile is sampled, so only its exact features are checked: no loss
at $m = 0$, all $2k_C$ bits at $m = n$ (every ordering ends with the whole
system), and a non-decreasing mean, since every ordering's curve is
non-decreasing:

```python
profile = res.values["erasure_lost_profile"][t, 0]  # lambda_C(m), m = 0..n, at the final layer
assert profile[0] == 0 and profile[n] == 2 * k
assert np.all(np.diff(profile) >= 0)
```

---

## 6. Pitfalls

* **Bits, not qubits.** $\ell_C(A)$ runs from 0 to $2k_C$, and losing one bit
  (a dephased logical qubit) is not losing a qubit.
* **System qubits only.** The probe's region must hold system qubits
  (`mipt.MonitoredCircuit` checks it). The `observables` functions require the
  erased qubits and the reference to be disjoint. Their result is the lost
  logical information when the system and the reference together are in a
  pure state, as in the reference pictures.
* **The profile is not a failure probability.** `erasure_lost_profile` is the
  mean number of bits lost, $\lambda_C(m) \ge P_C(m)$. For $P_C(m)$ use
  `erasure_first_failure` and `uncorrectable_region_counts`
  ([random-region-erasure.md](random-region-erasure.md)).
* **One fixed region is one region.** `lost_information` probes the region you
  give, for example a contiguous block of `system_order`. It says nothing about
  random regions of the same size.
* **After purification.** Both probes record 0 once $k_C = 0$, so averages over
  all trajectories count purified ones as losing nothing. Condition on
  $k_C > 0$ when that matters.
