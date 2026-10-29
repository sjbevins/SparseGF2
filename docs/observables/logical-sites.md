# Weight-one logical operators and untouched qubits

How SparseGF2 counts the system qubits that carry a weight-one logical operator,
$N_{\log}$, records $I(i:R)$ for every qubit, and records when each qubit is first
touched by the circuit, from which the number of untouched ("bare") qubits
$N_{\mathrm{bare}}(T)$ follows.

The setting and notation are those of
[random-region-erasure.md](random-region-erasure.md) §1.1; the lost information of a
region is in [lost-information.md](lost-information.md) and the reference rows in
[code-dimension.md](code-dimension.md). Function references are `path::function`,
relative to `src/sparsegf2/`. This describes the code at randomness contract 2
(`mipt.RANDOMNESS_CONTRACT`).

## Contents

1. [Definition](#1-definition)
2. [What the simulator evaluates, and why it equals the definition](#2-what-the-simulator-evaluates-and-why-it-equals-the-definition)
3. [From the kernel to the code you run](#3-from-the-kernel-to-the-code-you-run)
4. [Cost and accuracy](#4-cost-and-accuracy)
5. [Worked example](#5-worked-example)
6. [Pitfalls](#6-pitfalls)

---

## 1. Definition

Let $S$ be the code's stabilizer group at a checkpoint (the elements of the joint
stabilizer group of the system and $R$ that act trivially on $R$), and $N(S)$ its
normalizer, the Paulis on the system that commute with every element of $S$. For a
system qubit $i$,

$$
D_i = \mathbf 1\lbrace \exists P \in \lbrace X_i, Y_i, Z_i\rbrace : P \in N(S) \setminus S \rbrace ,
$$

that is, $D_i = 1$ when some single-qubit Pauli on $i$ commutes with all of $S$ without
being in $S$: a nontrivial logical operator of weight one, supported on $i$. When
$k_C = 0$, $N(S) = S$ and every $D_i = 0$. The number of such qubits is

$$
N_{\log} = \sum_{i=1}^{n} D_i .
$$

**Site information.** Erasing qubit $i$ alone destroys

$$
I(i:R) = \dim N(S)_{\lbrace i\rbrace} - \dim S_{\lbrace i\rbrace} \in \lbrace 0, 1, 2\rbrace
$$

bits of logical information (the cleaning lemma,
[lost-information.md](lost-information.md); $V_{\lbrace i\rbrace}$ is the subgroup of
elements supported on $i$). $N(S)_{\lbrace i\rbrace}$ and $S_{\lbrace i\rbrace}$ are the
single-qubit Paulis on $i$ (with the identity) that commute with $S$ and that lie in
$S$, so $I(i:R) > 0$ exactly when $D_i = 1$, and

$$
N_{\log} = \lvert\lbrace i : I(i:R) > 0\rbrace\rvert .
$$

$N_{\log}$ counts qubits, not bits: it is **not** $\sum_i I(i:R)$. A qubit still
Bell-paired with its own reference (purification picture) has
$N(S)_{\lbrace i\rbrace}$ all four Paulis and $S_{\lbrace i\rbrace} = \lbrace I\rbrace$,
so $I(i:R) = 2$; a qubit with one nontrivial logical type, as in the repetition code, has
$I(i:R) = 1$.

**Distance two.** The contiguous distance $d_{\mathrm{cont}}$ (the shortest run of the
ordering whose erasure is uncorrectable) is 1 exactly when some single qubit fails, so

$$
d_{\mathrm{cont}} \ge 2 \iff k_C > 0 \ \text{and}\ N_{\log} = 0 ,
$$

in any ordering: `contiguous_distance(cap=2)` records 2 exactly where $k_C > 0$ and
$N_{\log} = 0$, and 1 where $N_{\log} > 0$.

**Untouched qubits.** A system qubit is *touched* in a layer when it takes part in a drawn
gate (either end) or is measured; every gate of the scramble (pairs, matchings and
scramble layers) touches at layer 0, and the Bell pairing of the setup does not count.
The first touch $f_q$ is the first layer that touches $q$, and

$$
N_{\mathrm{bare}}(T) = \lvert\lbrace q : f_q > T \text{ or } q \text{ never touched}\rbrace\rvert ,
\qquad
\tau_{\mathrm{bare}} = \min\lbrace T : N_{\mathrm{bare}}(T) = 0\rbrace = \max_q f_q .
$$

In the purification picture an untouched qubit is still in its Bell pair with reference
$n + q$, a tensor factor of the joint state, so $I(q:R) = 2$ and it contributes one
logical qubit: $N_{\mathrm{bare}}(T) \le N_{\log}(T)$ and $N_{\mathrm{bare}}(T) \le k_C(T)$
at every layer. (In the single picture an untouched qubit other than the partner is in
$\lvert 0\rangle$ and carries nothing.)

---

## 2. What the simulator evaluates, and why it equals the definition

**The two ranks.** With the reference-row invariant
([code-dimension.md](code-dimension.md) §2), the rows outside `ref_rows` are a basis of
$S$ and all rows restricted to the system span $N(S)$. For a set $A$ of system qubits,
$\dim V_A = 2\lvert A\rvert - \operatorname{rank}(V^{\perp}\vert_A)$
([random-region-erasure.md](random-region-erasure.md) §2.2), and $S^{\perp} = N(S)$
inside the system, so

$$
I(A:R) = \dim N(S)_A - \dim S_A = \operatorname{rank}(N(S)\vert_A) - \operatorname{rank}(S\vert_A) .
$$

For $A = \lbrace i\rbrace$ the restrictions are qubit $i$'s X column $a$ and Z column $b$
over all rows, and the same columns with the marked rows cleared, $a'$ and $b'$. Two
vectors over GF(2) have rank $[a \ne 0] + [b \ne 0] - [a \ne 0 \wedge b \ne 0 \wedge a = b]$
(two nonzero vectors are dependent exactly when they are equal), so

$$
I(i:R) = \operatorname{rank}(a, b) - \operatorname{rank}(a', b')
$$

needs one pass over the $2W$ words of the two columns
(`_kernel/entropy.py::single_qubit_lost`). `_kernel/entropy.py::site_information_rows`
runs it for $i = 0, \dots, n - 1$, writes every value when asked to and returns the
number above 0, $N_{\log}$.

**Why membership in $S$ matters.** A test of the local rank alone, "some single-qubit
Pauli on $i$ commutes with $S$", is not $D_i$: a measured qubit has $Z_i \in S$, so $Z_i$
commutes with $S$, but it is a stabilizer, not a logical operator. The rank difference
handles it: $Z_i \in S$ puts $b' = b \ne 0$ into both ranks, and $I(i:R) = 0$ unless
another Pauli on $i$ is a logical operator.

**When $k_C = 0$**, $\operatorname{rank}(N(S)\vert_A) = \operatorname{rank}(S\vert_A)$
for every $A$, so every $I(i:R) = 0$; the engine returns 0 at once from the popcount it
already has.

**First touches.** The engine draws each layer as lists, the gates' qubit pairs and the
measured qubits (`_kernel/ensemble.py::gen_layer`), before applying them. After they are
applied, `_kernel/ensemble.py::_touch` gives the layer to every qubit in the lists whose
entry is still -1, and the scramble's lists get layer 0. Layers only increase, so the
first value written is the first touch. Nothing is drawn, so the circuit, its random
numbers and every other output are those of the run without the option.

---

## 3. From the kernel to the code you run

**The engine** (`_kernel/ensemble.py`):

1. `eval_probe` with `PROBE_LOGICAL_SITES` writes
   `site_information_rows(tab, ws, n_sys, ref_rows, empty)` (0 when `s_ref == 0`) into
   `row[0]`; with `PROBE_SITE_INFORMATION` it writes the n values into `row[0:n]`, qubit
   $i$ at index $i$ (label order, not `system_order`).
2. With `out_touch` of shape `(trajectories, n)` (filled with -1), `run_ensemble` calls
   `_touch` after the scramble and after each layer; with width 0 it records nothing.
3. After a `stop_when_purified` stop both probes record 0, the exact value for a pure
   reference (`mipt.py::_KINDS`, rule `"zero"`); after a `stop_when_distance` stop they
   are NaN, as every probe.

**Python.**

* `mipt.logical_sites(name=None, *, checkpoints=None)`: `values["logical_sites"]`,
  `(trajectories, evaluations)`.
* `mipt.site_information(name=None, *, checkpoints=None)`: `values["site_information"]`,
  `(trajectories, evaluations, n)`; `values["site_information"] > 0` are the defect
  bitsets. Give it checkpoints of its own: it stores n numbers per evaluation.
* `MonitoredCircuit(record_first_touch=True)`: `EnsembleResult.first_touch`,
  `(trajectories, n)` int64 with -1 for never touched through `final_layer`;
  `EnsembleResult.bare_counts(layers=None)` returns `(layers, counts)` with
  `counts[t, j]` $= N_{\mathrm{bare}}(\text{layers}[j])$ (NaN past a stop unless it
  was already 0: it never rises); `EnsembleResult.bare_times()` returns
  `(time, observed)` of $\tau_{\mathrm{bare}}$ for `analysis.kaplan_meier` (censored at
  `final_layer`).
* `StabilizerCode.site_information()` and `StabilizerCode.logical_sites()`: the same
  values for a code given by its generators or saved by `code_snapshot`
  (`EnsembleResult.code(t, c)`), with the code's normalizer rows in place of the
  reference rows.

**JSON** (CLI, MCP): the probes `{"kind": "logical_sites"}` and
`{"kind": "site_information", "checkpoints": [...]}`, and the spec key
`"record_first_touch": true`. The summary of `logical_sites` adds `mean_with_code`,
`stderr_with_code`, `count_with_code` and `distance_two_fraction` (the fraction of
trajectories with $k_C > 0$ and $N_{\log} = 0$) when a `code_dimension` probe covers its
layers; `site_information` is shown as the mean $I(i:R)$ per qubit; and a run with first
touches gets a `first_touch` block (`bare_mean`, `fraction_with_bare`, `all_touched`,
`censored`, `tau_bare_median_all`).

---

## 4. Cost and accuracy

* Every value is an exact integer. `logical_sites` and `site_information` read two
  columns of $W = \lceil (n + \lvert R\rvert)/64\rceil$ words per qubit: $2nW$ words per
  evaluation, no elimination, whatever the code. That is exactly what
  `contiguous_distance(cap=2)` spends on a code without a failing qubit; on a code with
  one, cap 2 stops at the first and is cheaper, but says nothing about how many there are.
  The full contiguous distance pays two gauge eliminations, $O(n^3/64)$, and
  `uncorrectable_region_counts(max_size=1)`, which gives the same count, first builds the
  code's adapted columns, at least $O(n(n + \lvert R\rvert))$ bit operations.
  `benchmarks/logical_sites_cost.py` measures all of them beside one dynamics layer.
* First touches cost two tests per gate and one per measured qubit, and 8 n bytes per
  trajectory in the result; off (the default), one test per layer and nothing stored.
* Across trajectories, `analysis.mean_and_error` gives means and standard errors. Means of
  $N_{\log}$ over all trajectories count purified ones as 0; condition on $k_C > 0$ (the
  summary's `mean_with_code`) for the density on surviving codes.

---

## 5. Worked example

A Watts–Strogatz circuit with $n = 6$ in the purification picture, every value recorded
after every layer. The engine's values are compared with a brute force on the replayed
state: every element of the joint stabilizer group is listed, those without support on
$R$ form $S$, and each single-qubit Pauli is tested for commuting with all of $S$ and for
membership in $S$. The printed numbers depend on the version's random streams and are not
quoted. `spec.circuit(t, seed=3)` rebuilds the circuit this run simulated because the run
was made under this version's randomness contract, which the example checks.

```python
import numpy as np

from sparsegf2 import Simulator, StabilizerCode, graphs, mipt

n = 6
spec = mipt.MonitoredCircuit(
    num_system=n,
    edges=graphs.watts_strogatz(n, 4, 0.3, seed=1),
    layers=16,
    measurement_rate=0.15,
    schedule="random_edges",
    reference="purification",
    checkpoints="every_layer",
    probes=[mipt.code_dimension(), mipt.logical_sites(), mipt.site_information(),
            mipt.contiguous_distance(cap=2)],
    record_first_touch=True,
)
res = spec.run(4, seed=3)
k = res.values["code_dimension"]
n_log = res.values["logical_sites"]
sites = res.values["site_information"]  # (4 trajectories, 17 layers, n qubits)
assert np.array_equal(n_log, (sites > 0).sum(axis=2))  # a count of qubits, not of bits
d2 = res.values["contiguous_distance[cap=2]"]
assert np.array_equal(d2[k > 0] == 2, n_log[k > 0] == 0)  # d >= 2 iff k > 0 and N_log = 0
print("N_log after each layer:", n_log[0].astype(int).tolist())
```

The brute force, on the state after every layer of trajectory 0:

```python
def brute_force_sites(sim, n):
    """I(i:R) for every system qubit by listing S and testing every single-qubit Pauli."""
    xs, zs, _ = sim.stabilizer_arrays()
    num = xs.shape[1]
    gens = [int(sum(1 << q for q in np.flatnonzero(x)) | sum(1 << (num + q) for q in np.flatnonzero(z)))
            for x, z in zip(xs, zs)]
    group = [0]
    for g in gens:
        group += [e ^ g for e in group]
    on_r = sum(1 << q for q in range(n, num)) | sum(1 << (num + q) for q in range(n, num))
    s = [e for e in group if e & on_r == 0]  # S: every element without support on R
    members = set(s)
    out = []
    for i in range(n):
        commuting = inside = 0
        for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):  # I, X, Z, Y on qubit i
            # P = X^a Z^b on i anticommutes with e when a z_i(e) + b x_i(e) is odd
            commutes = all(((a & (e >> (num + i))) ^ (b & (e >> i))) & 1 == 0 for e in s)
            commuting += commutes
            inside += ((a << i) | (b << (num + i))) in members
        out.append(int(np.log2(commuting)) - int(np.log2(inside)))
    return out


assert res.randomness_contract == mipt.RANDOMNESS_CONTRACT  # so spec.circuit repeats this run
sim = Simulator(spec.num_qubits, phases=False)
run = sim.run(spec.circuit(0, seed=3), observe=lambda s, tick: brute_force_sites(s, n))
for layer, (_, expected) in enumerate(run.observations):
    assert sites[0, layer].tolist() == expected
```

Codes with known distance, through `StabilizerCode`: the $[[4,2,2]]$ code has no
weight-one logical operator, the repetition code has one (a $Z$) on every qubit, and a
measured qubit ($Z_0 \in S$) carries none although $Z_0$ commutes with $S$:

```python
assert StabilizerCode(["XXXX", "ZZZZ"]).site_information().tolist() == [0, 0, 0, 0]
assert StabilizerCode(["ZZ_", "_ZZ"]).site_information().tolist() == [1, 1, 1]
assert StabilizerCode(["Z___"]).site_information().tolist() == [0, 2, 2, 2]
assert StabilizerCode(["XZZX_", "_XZZX", "X_XZZ", "ZX_XZ"]).logical_sites() == 0  # [[5,1,3]]
```

First touches against the circuit text: the scramble (none here) is layer 0, every
`TICK` ends a layer, and the setup's `H` and `CX` lines do not count. Every untouched
qubit still carries 2 bits, so $N_{\mathrm{bare}} \le N_{\log}$ and
$N_{\mathrm{bare}} \le k_C$:

```python
import re

first = [-1] * n
layer = 0
for line in str(spec.circuit(0, seed=3)).splitlines():
    name, targets = re.match(r"([A-Z_0-9]+)(?:\([^)]*\))?\s*(.*)", line).groups()
    if name == "TICK":
        layer += 1
    elif name in ("CLIFFORD_2Q", "M"):  # gates and measurements; H and CX are the setup
        for q in map(int, targets.split()):
            if first[q] < 0:
                first[q] = layer
assert res.first_touch[0].tolist() == first

layers, bare = res.bare_counts()
assert np.all(bare <= n_log) and np.all(bare <= k)
untouched = (res.first_touch[:, None, :] < 0) | (res.first_touch[:, None, :] > layers[None, :, None])
assert np.all(sites[untouched] == 2)
time, observed = res.bare_times()  # tau_bare, censored at final_layer
print("tau_bare:", time.tolist(), "observed:", observed.tolist())
```

---

## 6. Pitfalls

* **$N_{\log}$ is not $\sum_i I(i:R)$.** A bare qubit contributes 2 bits but one site.
* **A local rank test is not enough.** "Some Pauli on $i$ commutes with $S$" also holds
  for a measured qubit with $Z_i \in S$; the probe subtracts the rank over $S$, which
  removes it (§2).
* **Means include purified trajectories.** `mean("logical_sites")` counts $k_C = 0$ as
  $N_{\log} = 0$, so it falls as codes purify. Purification is not encoding: condition on
  $k_C > 0$ (`mean_with_code` in the summary, or `values[...][k > 0]`), and report the
  fraction with $k_C > 0$ and $N_{\log} = 0$ (`distance_two_fraction`) separately from the
  fraction with $k_C > 0$.
* **Label order.** `site_information` is indexed by qubit label $0..n-1$, not by
  `system_order`; $N_{\log}$ does not depend on the order, and neither does whether
  $d_{\mathrm{cont}} \ge 2$.
* **$N_{\mathrm{bare}} \le N_{\log}$ holds in the purification picture only.** In the
  single picture an untouched qubit is in $\lvert 0\rangle$, not entangled with $R$.
* **First touches past a stop are unknown.** A stopped trajectory keeps -1 for every
  qubit it had not touched by its `final_layer`; `bare_counts` returns NaN after the stop
  unless the count was already 0, and `bare_times` censors there.
* **The first touch is a property of the drawn circuit**, not of the state: a gate whose
  random Clifford happens to act trivially still touches both qubits.
