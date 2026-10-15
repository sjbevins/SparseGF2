# Getting started with SparseGF2

This guide takes you from a fresh clone to running your own simulation
campaigns of monitored random Clifford circuits. It is written for a physicist
who is about to start new studies, and it uses one study throughout as the
running example: **erasure protection of the codes that monitored circuits
generate** (Watts–Strogatz graphs at two rewiring probabilities, a scan of
measurement rates, depths from 1 to 10⁶ layers, first-failure sizes over random
orderings, exhaustive screening of small erasures, and the codes saved for
later). Section 4 runs a scaled-down version of that study end to end in a few
seconds; section 5 turns it into the full-size campaign on a cluster.

Every Python block in this guide runs, in order, as part of the test suite
(`tests/test_docs.py`), except the few marked as full-scale code. They write
files only into a temporary directory. Paste them into a script or a notebook
in the same order and they will run for you too.

**Contents**

1. [Install](#1-install)
2. [A ten-minute tour](#2-a-ten-minute-tour)
3. [Concepts](#3-concepts)
4. [Your first study, end to end](#4-your-first-study-end-to-end)
5. [Scaling up to the full study](#5-scaling-up-to-the-full-study)
6. [Other workflows](#6-other-workflows)
7. [Troubleshooting and FAQ](#7-troubleshooting-and-faq)

Related documents: [`README.md`](../README.md) (overview, figures, performance),
[`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) (how the simulator works inside),
[`MCP_GETTING_STARTED.md`](MCP_GETTING_STARTED.md) (driving it from an AI agent),
[`DESIGN.md`](DESIGN.md) (design decisions and the randomness contract), and the
complete example scripts in [`examples/`](../examples/).

---

## 1. Install

You need **Python 3.11 or newer** and `git`. No C compiler is needed: the
kernels are compiled by Numba, which ships its own LLVM.

```sh
git clone https://github.com/sjbevins/SparseGF2
cd SparseGF2
python -m venv .venv
source .venv/bin/activate                       # Windows: .venv\Scripts\activate
pip install -e ".[test]"
```

`-e` (editable) means the installed package is the source tree itself: a
`git pull` updates it without reinstalling. The extras are:

| extra | adds | for |
|---|---|---|
| `test` | pytest, Stim, matplotlib | the test suite (Stim is the reference it checks against) and the plots in this guide |
| `mcp` | the MCP SDK | the `sparsegf2-mcp` server for AI agents ([MCP guide](MCP_GETTING_STARTED.md)) |
| `lint` | ruff | linting |
| `dev` | all of the above | development |

Combine them as `pip install -e ".[test,mcp]"`. The core simulator needs
only NumPy and Numba. The package imports as `sparsegf2`, and the install
puts two commands on your path: `sparsegf2` (command line) and `sparsegf2-mcp`
(MCP server).

**Check the install.** A one-line smoke test, then the first ensemble (this one
compiles the ensemble kernel, see below):

```sh
python -c "import sparsegf2 as sg; s = sg.Simulator(2); s.h(0); s.cx(0, 1); print(sg.__version__, s.entropy([0]))"
python -c "from sparsegf2 import graphs, mipt; print(mipt.MonitoredCircuit(num_system=16, edges=graphs.cycle(16), layers=32, measurement_rate=0.1, reference='purification', probes=[mipt.code_dimension()]).run(100, seed=1))"
```

The first prints the version and `1` (one bit of entanglement in a Bell pair);
the second prints an `EnsembleResult(trajectories=100, ...)` line. Its
trajectories-per-second figure is tiny the first time, because it includes
the compilation; run the line again and it starts in under a second.

**Run the test suite** (from the repository root):

```sh
python -m pytest
```

There are about 600 tests, doctests included. On the 4-core machine this
guide was written on, the suite takes about a minute once the kernels are
compiled and cached, and about two minutes the very first time, when it also
compiles every kernel (see the next paragraph). All tests should pass; Stim is
required, not optional.

**First-call compilation and the cache.** The simulation kernels are compiled
by Numba the first time each one is called. That takes a few seconds per
kernel (on the machine used for this guide, about 30 s for the ensemble
engine in the smoke test above), once: the machine code
is cached on disk in the `__pycache__` folders next to the source, and every
later process loads it in a fraction of a second. So the first run after
installing or updating is slow and the rest are not. If the source tree is
read-only (a shared install), point the cache somewhere writable with
`export NUMBA_CACHE_DIR=$HOME/.cache/sparsegf2-numba`. A cache built on one
CPU model is not reused on a different one; the kernels simply compile again.

**Threads.** The ensemble engine and `sample` run trajectories (shots) in
parallel over all cores. To limit that, set `NUMBA_NUM_THREADS` before Python
starts, or pass `num_threads=` to a single call:

```sh
NUMBA_NUM_THREADS=2 python -c "import numba; print(numba.get_num_threads())"
```

`spec.run(..., num_threads=k)` can lower the thread count for one run but not
raise it above `NUMBA_NUM_THREADS`. Results never depend on the number of
threads (section 3.6); the benchmark checks this at every thread count.

How it scales (`benchmarks/thread_scaling.py`, results in
`benchmarks/results/thread_scaling.md`; AMD Ryzen 9 9950X3D, 16 cores and 32
logical processors, Numba's `omp` threading layer): speedup over one thread was
12.3 to 13.7 at 16 threads and 17.3 to 21.1 at 32, depending on the workload
(ensembles of erasure probes scale best, the purification-time workload with
its every-layer probe least). Using all 32
logical processors beat 16 threads in every workload, by 1.4 to 1.5 times, so
the default (`NUMBA_NUM_THREADS` equal to the logical processor count) is the
right setting on a machine you have to yourself. On a shared machine, or
when running several jobs side by side, set `NUMBA_NUM_THREADS` to each job's
share of the logical processors. The `omp` and `tbb` layers are safe to call
from several Python threads at once (as the MCP server's background jobs do);
`workqueue` is not.

---

## 2. A ten-minute tour

### A state you act on

`Simulator` is a stabilizer state you apply gates to and measure, with Stim's
gate names and conventions:

```python
import numpy as np

import sparsegf2 as sg
from sparsegf2 import analysis, graphs, mipt
from sparsegf2 import observables as ob

print(sg.__version__)

sim = sg.Simulator(4, seed=1)          # |0000>, signs tracked (phases=True is the default)
sim.h(0)
sim.cx(0, 1, 1, 2, 2, 3)               # two-qubit gates take target pairs, like Stim
print(sim.stabilizers())               # a 4-qubit GHZ state
print(sim.entropy([0, 1]))             # S of qubits {0, 1}: 1 bit
print(ob.tripartite_information(sim, [0], [1], [2]))   # I3 = 1 for a GHZ state
```

Entropies are in bits (they are GF(2) ranks). The README's
[quickstart](../README.md#quickstart) shows more of the `Simulator`,
Stim-language circuits and the observables.

### A first ensemble

Studies of monitored circuits use the **ensemble engine**,
`mipt.MonitoredCircuit`: you describe the protocol once, and the engine
generates every trajectory's random circuit and simulates it inside one
compiled, parallel kernel, recording **probes** at **checkpoint** layers.

Here: 32 qubits on a ring, brickwork random two-qubit Cliffords, each qubit
measured with probability 0.1 per layer, in the **purification picture** (every
system qubit starts Bell-paired with its own reference qubit, so the system
starts maximally mixed and holds 32 logical qubits).

```python
n = 32
spec = mipt.MonitoredCircuit(
    num_system=n,
    edges=graphs.cycle(n),             # the interaction graph: a ring
    layers=4 * n,                      # 128 layers of gates + measurements
    measurement_rate=0.1,
    reference="purification",
    checkpoints=16,                    # evaluate probes at layers 0, 16, 32, ..., 128
    probes=[mipt.code_dimension(), mipt.half_cut_entropy()],
)
result = spec.run(trajectories=200, seed=1)
print(result)
print(result.layers("code_dimension"))             # [  0  16  32 ... 128]
print(result.values["code_dimension"].shape)       # (200, 9): trajectories x checkpoints
rate = result.values["code_dimension"].mean(axis=0) / n
print(np.round(rate, 3))                           # code rate k/n: 1 at layer 0, then falling
```

What the output means:

* `result.values[name]` is a NumPy array with one row per trajectory and one
  column per evaluation of that probe (a third axis for vector probes).
* `code_dimension` is k, the number of logical qubits still protected from
  the measurements: the entropy of the reference, S(R). It starts at n and the
  measurements purify it; k/n is the code rate.
* `half_cut_entropy` is the entanglement entropy of the first half of the
  system (here including its entanglement with the references).
* `result` also records the seed it used, the package version and the spec,
  so `result.save(path)` keeps everything needed to reproduce it.

Erasure probes, graph collections and the rest build on exactly this pattern.

---

## 3. Concepts

The protocol is fixed by the fields of `MonitoredCircuit` (its docstring and
the module docstring of `sparsegf2.mipt` are the reference). This section
explains what each concept means and what the engine guarantees.

### 3.1 System and reference qubits: the three pictures

Qubits `0 .. n-1` are the system; reference qubits follow them. The
`reference` field picks the initial state:

| `reference` | initial state | references | used for |
|---|---|---|---|
| `"none"` (default) | \|0...0⟩ | none | entanglement of a pure state: half-cut entropy, −I₃, entropy profiles |
| `"purification"` | system qubit `i` Bell-paired with reference `n + i` | `n` | the code the circuit generates: k = S(R), code rate, erasure protection |
| `"single"` | one reference `n` Bell-paired with system qubit `reference_partner` (an index, or `"random"` per trajectory) | 1 | purification time of a single qubit of quantum information |

The reference qubits never take part in the dynamics. In the purification
picture the system holds a stabilizer code with n − k stabilizer generators;
erasing a set A of system qubits loses I(A:R) bits of logical information, and
the erasure is correctable exactly when I(A:R) = 0. Code probes need a
reference; entanglement probes work in every picture.

### 3.2 What one layer is

One **layer** is exactly:

1. two-qubit gates on edges of the interaction graph, chosen by the schedule,
   each a uniformly random two-qubit Clifford (or uniform over `gate_set`);
2. a Z measurement of each system qubit, independently with probability
   `measurement_rate` (p).

A layer is never rescaled: depth T means T such layers. Before the dynamics an
optional scramble can run (`scramble_gates` random gates on random pairs,
`scramble_matchings` layers of random all-to-all perfect matchings,
`scramble_layers` gate-only layers of the dynamics). Layer 0 is the state after
setup and scramble.

### 3.3 Schedules

| `schedule` | gates in one layer |
|---|---|
| `"matching"` (default) | one colour class of an edge colouring of the graph, cycling through the colours; on an even ring this is the standard brickwork |
| `"random_edges"` | `gates_per_layer` edges (default n // 2) drawn uniformly with replacement, or with probabilities `edge_weights` |
| `"all_edges"` | every edge of the graph, in order |

The erasure study uses `"random_edges"`: n/2 random edges of a Watts–Strogatz
graph per layer.

### 3.4 Measurement rate

Each system qubit is measured with probability p after each layer's gates. The
engine draws one uniform random number per qubit per layer whatever p is, and
measures the qubit when that number is below p. Two consequences: the gates
do not depend on p, and with the same seed the set of qubits measured at a
lower rate is a subset of the set measured at a higher one (common random
numbers across a rate scan, which makes curves versus p smooth).

### 3.5 Checkpoints and probes

`checkpoints` says after which layers to evaluate the probes: `"final"` (the
default), `"every_layer"`, a positive integer stride k (layers 0, k, 2k, ...
and the last), or an explicit list of layers in [0, layers]. A list is sorted
and deduplicated, so you can pass any set of depths, such as a geometric
sequence.

**Each probe can have its own checkpoints** (`checkpoints=` on any probe
factory, in the same forms), so cheap probes run at every depth you care about
and expensive ones at a sparser set. `result.layers(name)` gives the layers of
probe `name`, which are the columns of `result.values[name]`. Purification
(below) is detected only at the spec's own `checkpoints`.

The probes:

| probe | records (per evaluation) | needs a reference |
|---|---|---|
| `entropy(region)`, `half_cut_entropy()` | S(A) | no |
| `mutual_information(a, b)`, `tripartite_information()` | I(A:B), I₃ (default: the first three quarters of the system) | no |
| `entropy_profile()` | vector, S of the first m system qubits for m = 0..n | no |
| `code_dimension()` | k = S(R) | yes |
| `contiguous_distance()` | shortest contiguous run whose erasure loses information; NaN when k = 0 | yes |
| `lost_information(region)` | I(A:R) for a fixed region A | yes |
| `erasure_lost_profile(samples)` | vector, mean I(A_m:R) over random erasures of each size m = 0..n | yes |
| `erasure_first_failure(samples=M)` | vector of M first-failure sizes, one per random ordering of the qubits (n + 1 when k = 0) | yes |
| `uncorrectable_region_counts(max_size=w)` | vector of w counts of uncorrectable m-qubit erasures, m = 1..w, by exhaustion | yes |
| `code_snapshot()` | the number of stabilizer generators; the generators themselves are saved | no |

Probe names default to the factory's name, except
`uncorrectable_region_counts(w)`, whose name is `"uncorrectable_region_counts[w]"`,
and probes on explicit regions, whose names include the region. Pass `name=` to
choose your own.

### 3.6 Trajectories, seeds and the randomness contract

A **trajectory** is one random circuit and its simulation. The contract:

* Trajectory `t` of `spec.run(..., seed=s)` is a deterministic function of
  `(spec, s, t)` only. It does not depend on the number of threads, on how
  many other trajectories run in the same call, or on the machine.
* Hence `spec.run(count, seed=s, first_trajectory=f)` returns exactly rows
  `f .. f + count - 1` of one big run with the same seed, and
  `EnsembleResult.concatenate` merges such chunks (it refuses chunks with
  different specs, seeds, package versions or randomness contracts, or
  overlapping trajectories).
* The circuit's gates and the measurement mask use one random stream. Each
  evaluation of a sampled probe draws its random orderings from a stream of
  its own under a separate probe key. Adding, removing or rescheduling probes,
  or changing `layers`, does not change the circuits: the state after layer T
  is the same in every run of the same trajectory that reaches layer T, and
  `result.replay(t, probes=..., checkpoints=...)` reproduces the sampled
  probes too.
* With the same seed, runs at different measurement rates share their gates
  and random numbers (section 3.4).
* `seed` can be an integer, a sequence of integers, a
  `numpy.random.SeedSequence`, or `None` (fresh entropy). The seed actually
  used is stored in the result: `result.seed_sequence` repeats the run, as long
  as `result.randomness_contract == mipt.RANDOMNESS_CONTRACT` (a run made under
  another contract cannot be reproduced by this version).
* `spec.circuit(trajectory=t, seed=s)` returns the exact circuit trajectory `t`
  ran, as a Stim-language `Circuit` (section 6.3 replays one).

### 3.7 Graph collections

Pass a list of graphs as `edges` and the ensemble runs on a **collection**:
each graph is evolved with `circuits_per_graph` random circuits (default 1),
and trajectory `t` runs on graph `t // circuits_per_graph`. A graph is fixed
for a whole circuit, never redrawn per layer. Without a trajectory count,
`spec.run(seed=...)` runs all `len(edges) * circuits_per_graph` trajectories;
asking for more is an error. `result.graph_index` gives each trajectory's
graph.

Random graphs take their own seed, separate from the circuit seed, so the same
collection (built with the same graph seeds) can be reused unchanged at every
measurement rate: build it once per β and pass it to every p.

```python
collection = [graphs.watts_strogatz(16, 4, 1.0, seed=(17, g)) for g in range(3)]
demo = mipt.MonitoredCircuit(
    num_system=16, edges=collection, circuits_per_graph=2, layers=8,
    measurement_rate=0.1, schedule="random_edges",
)
demo_result = demo.run(seed=1)                 # all 3 graphs x 2 circuits
print(demo.num_trajectories, demo_result.graph_index)   # 6 [0 0 1 1 2 2]
```

### 3.8 Stopping at purification, censoring and the exact values after a stop

With a reference, the **purification layer** of a trajectory is the first
spec checkpoint at which the reference is pure (k = 0); it is in
`result.purification_layer`, and it is **−1 if the reference never purified**
by the last checkpoint (right-censored; `result.censored` is the boolean mask).

`stop_when_purified=True` ends each trajectory at its purification layer,
which saves the time of simulating a state with no code left. Later
evaluations do not hold garbage; they hold each probe's **exact value for a
pure reference**:

| probe | value after a stop |
|---|---|
| `code_dimension`, `lost_information`, `erasure_lost_profile`, `uncorrectable_region_counts` | 0 |
| `erasure_first_failure` | n + 1 for every sample (no erasure ever fails) |
| `code_snapshot` | n generators; no code is stored (snapshot count −1) |
| entanglement probes, `contiguous_distance` | NaN (not determined: the system keeps evolving) |

So averages of code quantities over all trajectories (the code rate, Φ(e)
including purified circuits) need no special handling, and `result.mean(name)`
skips NaN for the entanglement probes. `result.final_layer` is the last layer
each trajectory simulated.

### 3.9 The result object

`spec.run` returns an `EnsembleResult`:

| attribute / method | what it is |
|---|---|
| `values[name]` | `(trajectories, evaluations)` or `(trajectories, evaluations, length)` float64 array |
| `layers(name)` | the layers of probe `name` (columns of `values[name]`) |
| `checkpoint_layers` | the spec's checkpoint layers |
| `purification_layer`, `censored` | τ per trajectory (−1 = censored), and the censoring mask (`None` without a reference) |
| `final_layer`, `num_measurements` | last layer simulated and number of measurements, per trajectory |
| `trajectories`, `graph_index` | global trajectory indices of the rows, and each one's graph |
| `snapshots[name]` | `(generators, counts)` of the `code_snapshot` probe |
| `code(t, c)` | the code of row `t` at evaluation `c` of the snapshot probe, as a `StabilizerCode` |
| `mean(name)`, `survival()` | NaN-skipping mean per evaluation; survival curve P(τ > layer) |
| `save(path)`, `EnsembleResult.load(path)` | one compressed `.npz` holding values, codes, spec, graphs, seed and version |
| `EnsembleResult.concatenate(results)` | merge chunks of the same spec and seed |
| `seed_sequence`, `version`, `elapsed_seconds` | provenance and timing |

---

## 4. Your first study, end to end

This is the erasure study scaled down so that it runs in a few seconds:
n = 32 instead of 256, 64 circuits per point instead of 1000, depths up to
4096 instead of 10⁶. Everything else is as in the full study.

### 4.1 Plan

* β ∈ {0, 1}: Watts–Strogatz rewiring of the ring lattice C(n, 2) (degree 4).
  One graph collection per β (one graph per circuit), reused at every p.
* A few measurement rates p.
* Depths T from a geometric sequence, used directly as the spec's checkpoints
  (one run to the largest depth gives every smaller depth too, section 3.6).
* Probes: `code_dimension` at every depth; `erasure_first_failure` (M random
  orderings) and `uncorrectable_region_counts(3)` on their own, sparser
  checkpoints (they cost much more); `code_snapshot` at the final layer only.
* `stop_when_purified=True`: a purified circuit has no code, and every code
  probe then records its exact value (section 3.8).
* One result file per (β, p) point.

### 4.2 Run it

```python
import tempfile
from pathlib import Path

workdir = Path(tempfile.mkdtemp(prefix="sparsegf2-guide-"))   # every file of this guide goes here

n = 32
betas = [0.0, 1.0]
rates = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3]
circuits = 64                     # one graph per circuit: 64 graphs per beta
samples = 100                     # M: random orderings per code
w = 3                             # exhaustive screening of all 1-, 2- and 3-qubit erasures
depths = np.unique(np.round(np.geomspace(1, 4096, 80)).astype(np.int64))       # T values
sparse = np.unique(depths[np.searchsorted(depths, np.geomspace(1, depths[-1], 16))])
print(len(depths), "depths;", len(sparse), "for the erasure probes:", sparse)


def make_spec(collection, p):
    return mipt.MonitoredCircuit(
        num_system=n,
        edges=collection,                       # circuits_per_graph=1 (the default)
        layers=int(depths[-1]),
        measurement_rate=p,
        schedule="random_edges",                # n/2 random edges per layer
        reference="purification",
        checkpoints=depths,                     # code_dimension and purification detection
        stop_when_purified=True,
        probes=[
            mipt.code_dimension(),
            mipt.erasure_first_failure(samples=samples, checkpoints=sparse),
            mipt.uncorrectable_region_counts(w, checkpoints=sparse),
            mipt.code_snapshot(checkpoints="final"),
        ],
    )


def point_file(beta, p):
    return workdir / f"erasure_n{n}_beta{beta:g}_p{p:g}.npz"


for beta in betas:
    collection = [graphs.watts_strogatz(n, 4, beta, seed=(17, g)) for g in range(circuits)]
    for p in rates:
        res = make_spec(collection, p).run(seed=1)      # the same circuit seed at every p
        res.save(point_file(beta, p))
        print(f"beta={beta:g} p={p:<5g} {res.num_trajectories} circuits, "
              f"{res.censored.sum()} never purified, {res.elapsed_seconds:.2f} s")
```

The `sparse` depths are taken from `depths` itself, so the erasure probes'
layers are a subset of the `code_dimension` layers: at each of them we know k
and can condition on k > 0.

### 4.3 Analyse it

Load a point back and pull out the arrays:

```python
def load_point(beta, p):
    res = mipt.EnsembleResult.load(point_file(beta, p))
    k = res.values["code_dimension"]                      # (circuits, len(depths))
    first = res.values["erasure_first_failure"]           # (circuits, len(sparse), M)
    counts = res.values[f"uncorrectable_region_counts[{w}]"]   # (circuits, len(sparse), w)
    cols = np.searchsorted(res.layers("code_dimension"), res.layers("erasure_first_failure"))
    return res, k, first, counts, k[:, cols]              # last: k at the erasure layers


res, k, first, counts, k_sparse = load_point(1.0, 0.06)
print(res.layers("erasure_first_failure"))
print(k.shape, first.shape, counts.shape, k_sparse.shape)
```

**Code rate and the fraction of circuits with a code.** The code rate is
r(p, T) = E[k/n] over all circuits (purified circuits count as k = 0, which is
what `stop_when_purified` records), and q₊(p, T) = Pr[k > 0]:

```python
r = {b: np.array([load_point(b, p)[1].mean(axis=0) / n for p in rates]) for b in betas}
q_plus = {b: np.array([(load_point(b, p)[1] > 0).mean(axis=0) for p in rates]) for b in betas}
print("r(p, T = 4096), beta = 1:", np.round(r[1.0][:, -1], 3))
print("q+(p, T = 4096), beta = 1:", np.round(q_plus[1.0][:, -1], 2))
```

**Failure probability under independent erasure.** Each qubit is erased
independently with probability e after the dynamics. For a code C,
Φ_C(e) = Σ_m C(n,m) e^m (1 − e)^(n−m) P_C(m), where P_C(m) is the probability
that a uniformly random m-qubit erasure is uncorrectable.
`analysis.erasure_failure_probability` estimates it from the first-failure
sizes; passing the exhaustive counts makes the m ≤ w terms exact and uses the
samples only for m > w (the **hybrid estimator**). It works on whole arrays:
a `(circuits, evaluations, M)` array of samples gives a
`(circuits, evaluations, len(e))` array.

```python
e0 = [0.05, 0.10, 0.20]
phi = {}           # Phi(e0; p, T) over all circuits (a purified circuit never fails)
phi_code = {}      # Phi(e0; p, T) conditioned on k > 0
for b in betas:
    phi[b] = np.empty((len(rates), len(sparse), len(e0)))
    phi_code[b] = np.full_like(phi[b], np.nan)
    for i, p in enumerate(rates):
        res, k, first, counts, k_sparse = load_point(b, p)
        per_code = analysis.erasure_failure_probability(first, n, e0, exact_counts=counts)
        phi[b][i] = per_code.mean(axis=0)
        for c in range(len(sparse)):
            has_code = k_sparse[:, c] > 0
            if has_code.any():
                phi_code[b][i, c] = per_code[has_code, c].mean(axis=0)
print("Phi(e = 0.1; p, T = 4096), beta = 1:", np.round(phi[1.0][:, -1, 1], 3))
```

**Erasure tolerance.** e_δ = sup{e : Φ(e) ≤ δ}, for the ensemble of codes with
k > 0 at each (p, T); NaN where no circuit holds a code:

```python
delta = 0.10
e_tol = {b: np.full((len(rates), len(sparse)), np.nan) for b in betas}
for b in betas:
    for i, p in enumerate(rates):
        res, k, first, counts, k_sparse = load_point(b, p)
        for c in range(len(sparse)):
            has_code = k_sparse[:, c] > 0
            if has_code.any():
                e_tol[b][i, c] = analysis.erasure_tolerance(
                    first[has_code, c], n, delta, exact_counts=counts[has_code, c])
print("e_0.10(p, T = 4096), beta = 1:", np.round(e_tol[1.0][:, -1], 3))
```

**P_C(m) curves.** `analysis.failure_by_size` turns first-failure sizes into
P_C(m) for m = 0..n (the fraction of orderings with D ≤ m: one ordering answers
every m at once), and `analysis.exact_failure_by_size` turns the counts into
the exact P_C(m) for m = 1..w. Averaged over the codes with k > 0 at the final
depth:

```python
res, k, first, counts, k_sparse = load_point(1.0, 0.06)
has_code = k_sparse[:, -1] > 0
p_sampled = analysis.failure_by_size(first[has_code, -1], n).mean(axis=0)     # m = 0..n
p_exact = analysis.exact_failure_by_size(counts[has_code, -1], n).mean(axis=0)  # m = 1..w
print("sampled P_C(1..3):", np.round(p_sampled[1:w + 1], 3))
print("exact   P_C(1..3):", np.round(p_exact, 3))
print("sampled P_C(4..8):", np.round(p_sampled[4:9], 3))
```

The sampled values agree with the exact ones within sampling error; use the
exact ones where you have them.

**Heatmaps.** With matplotlib, `pcolormesh` over p and T (log scale in T):

```python
import matplotlib

matplotlib.use("Agg")              # draw to files; drop this line in a notebook
import matplotlib.pyplot as plt


def heatmap(maps, layers, label, filename):
    fig, axes = plt.subplots(1, len(betas), figsize=(9, 3.6), sharey=True, layout="constrained")
    for ax, b in zip(axes, betas):
        mesh = ax.pcolormesh(rates, layers, np.ma.masked_invalid(maps[b].T),
                             shading="nearest", cmap="Blues", vmin=0)
        ax.set_yscale("log")
        ax.set_xlabel("measurement rate p")
        ax.set_title(f"β = {b:g}")
    axes[0].set_ylabel("depth T (layers)")
    fig.colorbar(mesh, ax=axes, label=label)
    fig.savefig(workdir / filename, dpi=120)
    plt.close(fig)


heatmap(r, depths, "code rate r = E[k/n]", "rate.png")
heatmap({b: phi[b][:, :, 1] for b in betas}, sparse, "Φ(e = 0.10)", "phi_e010.png")
heatmap(e_tol, sparse, "e_0.10 (codes with k > 0)", "e_tol.png")

fig, ax = plt.subplots(figsize=(4.5, 3.4), layout="constrained")
for i in (0, 2, 4):                       # P_C(m) at the final depth, beta = 1
    res, k, first, counts, k_sparse = load_point(1.0, rates[i])
    has_code = k_sparse[:, -1] > 0
    if has_code.any():
        ax.plot(np.arange(n + 1), analysis.failure_by_size(first[has_code, -1], n).mean(axis=0),
                label=f"p = {rates[i]:g}")
ax.set_xlabel("erased qubits m")
ax.set_ylabel("P_C(m)")
ax.legend()
fig.savefig(workdir / "p_c.png", dpi=120)
plt.close(fig)
print(sorted(f.name for f in workdir.glob("*.png")))
```

Cells where no circuit holds a code are left blank in the e_δ map.

### 4.4 Re-analyse a saved code

The codes saved by `code_snapshot` come back with the result. Take one, check
it against what the engine recorded, and analyse it with many more orderings
than the run used:

```python
res, k, first, counts, k_sparse = load_point(1.0, 0.06)
generators, kept = res.snapshots["code_snapshot"]          # kept[t, 0] = -1 if circuit t purified
t = int(np.flatnonzero(kept[:, 0] >= 0)[0])
code = res.code(t, 0)                                      # a StabilizerCode
print(code, "k =", code.num_logical)
assert code.num_logical == k[t, -1]
assert np.array_equal(code.uncorrectable_counts(w), counts[t, -1])   # same exhaustive counts
more = code.first_failure_sizes(5000, seed=0)              # 5000 orderings instead of 100
print("P_C(m), m = 4..8, 5000 orderings:", np.round(analysis.failure_by_size(more, n)[4:9], 3))
print("Phi_C(0.1) from 5000 orderings:",
      round(float(analysis.erasure_failure_probability(more, n, 0.1,
                                                       exact_counts=counts[t, -1])), 4))
```

`StabilizerCode` also builds codes from Pauli strings
(`StabilizerCode(["XXXX", "ZZZZ"])`) and answers `lost_information(A)` and
`is_correctable(A)` for any set A. That is everything Simulation 1 needs; the
rest is scale.

---

## 5. Scaling up to the full study

### 5.1 The full-size point

The full study, per (β, p) point: n = 256, 1000 circuits (one graph each),
T = `geomspace(1, 1e6, 10000)` rounded to integers, which is 5957 distinct
layers, M = 100 orderings and w = 3. Written as a script that runs **one chunk
of one point** (it is used again in 5.3 and 5.4); save it as
`erasure_point.py`:

```python
"""One chunk of one (beta, p) point of the erasure study (Simulation 1)."""

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

import sparsegf2
from sparsegf2 import graphs, mipt


def study_depths(depth_max=1_000_000, num=10_000):
    """The depths T: geomspace(1, depth_max, num) rounded and deduplicated."""
    return np.unique(np.round(np.geomspace(1, depth_max, num)).astype(np.int64))


def sparse_depths(depths, num=200):
    """About num depths spread geometrically, taken from depths itself."""
    return np.unique(depths[np.searchsorted(depths, np.geomspace(1, depths[-1], num))])


def make_spec(n, beta, p, circuits, depths, sparse, samples=100, max_size=3, graph_seed=17):
    collection = [graphs.watts_strogatz(n, 4, beta, seed=(graph_seed, g)) for g in range(circuits)]
    return mipt.MonitoredCircuit(
        num_system=n,
        edges=collection,
        layers=int(depths[-1]),
        measurement_rate=p,
        schedule="random_edges",
        reference="purification",
        checkpoints=depths,
        stop_when_purified=True,
        probes=[
            mipt.code_dimension(),
            mipt.erasure_first_failure(samples=samples, checkpoints=sparse),
            mipt.uncorrectable_region_counts(max_size, checkpoints=sparse),
            mipt.code_snapshot(checkpoints="final"),
        ],
    )


def git_commit():
    source = Path(sparsegf2.__file__).resolve().parent
    proc = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False)
    return proc.stdout.strip() or None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--chunk", type=int, required=True, help="chunk index, from 0")
    ap.add_argument("--chunk-size", type=int, default=50, help="circuits per chunk")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--circuits", type=int, default=1000, help="circuits (= graphs) per point")
    ap.add_argument("--depth-max", type=int, default=1_000_000)
    ap.add_argument("--num-depths", type=int, default=10_000)
    ap.add_argument("--num-sparse", type=int, default=200)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--max-size", type=int, default=3)
    ap.add_argument("--graph-seed", type=int, default=17)
    ap.add_argument("--seed", type=int, default=1, help="circuit seed: the same for every chunk")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args(argv)

    first = args.chunk * args.chunk_size
    count = min(args.chunk_size, args.circuits - first)
    if count <= 0:
        raise SystemExit(f"chunk {args.chunk} is past the last circuit")
    point = Path(args.out) / f"n{args.n}_beta{args.beta:g}_p{args.p:g}"
    out = point / f"chunk{args.chunk:04d}.npz"
    if out.exists():
        print(f"{out} exists, skipping")
        return out
    point.mkdir(parents=True, exist_ok=True)

    depths = study_depths(args.depth_max, args.num_depths)
    spec = make_spec(args.n, args.beta, args.p, args.circuits, depths,
                     sparse_depths(depths, args.num_sparse), args.samples, args.max_size,
                     args.graph_seed)
    start = time.time()
    res = spec.run(count, seed=args.seed, first_trajectory=first, num_threads=args.threads)
    tmp = point / f"chunk{args.chunk:04d}.tmp.npz"
    res.save(tmp)
    os.replace(tmp, out)                                   # never a half-written chunk
    provenance = {"sparsegf2": sparsegf2.__version__, "git_commit": git_commit(),
                  "args": vars(args), "host": platform.node(),
                  "seconds": round(time.time() - start, 1),
                  "trajectories": [first, first + count - 1]}
    out.with_suffix(".json").write_text(json.dumps(provenance, indent=2))
    print(f"{out}: {count} circuits in {provenance['seconds']} s")
    return out


if __name__ == "__main__":
    main()
```

Every chunk builds the same collection of 1000 graphs (graph `g` from seed
`(17, g)`) and the same spec; it simulates only circuits
`first .. first + count − 1`. That is what lets the chunks merge later.

Run as one process, a whole point would be (shown, not run here):

<!-- not run -->
```python
depths = study_depths()                        # 5957 depths from 1 to 10^6
spec = make_spec(256, 1.0, 0.05, 1000, depths, sparse_depths(depths))
result = spec.run(seed=1)                      # all 1000 circuits; hours, see 5.2
result.save("erasure_n256_beta1_p0.05.npz")
```

### 5.2 Estimate time and memory before running

**Arithmetic from measured costs.** Measured on one thread at n = 256, for
random-circuit codes: a layer of dynamics (n/2 random two-qubit Cliffords plus
the measurements) costs about 30 µs (at p = 0.05; up to about 60 µs at p = 0.2
on β = 1 graphs, so a circuit that never purifies at a higher rate costs up to
twice as much); one evaluation of
`erasure_first_failure` with M = 100 costs about 2 ms,
`uncorrectable_region_counts` costs about 1 ms for w = 2, about 20 ms for w = 3
and about 4 s for w = 4 (`benchmarks/results/erasure_cost.md` has every n and w), and `code_snapshot` under 1 ms. With about 175 sparse
depths (`sparse_depths(depths)` above), one circuit that never purifies costs

| part | cost per circuit (one thread) |
|---|---|
| dynamics, 10⁶ layers × 30 µs | 30 s |
| `erasure_first_failure`, 175 × 2 ms | 0.35 s |
| `uncorrectable_region_counts(3)`, 175 × 20 ms | 3.5 s |
| `code_dimension` at 5957 depths, `code_snapshot` once | well under 1 s |
| **total** | **about 35 s** |

so a point of 1000 circuits is about 10 core-hours, or about 2.5 hours on 4
cores. Circuits that purify stop early (`stop_when_purified`), so points deep
in the pure phase cost far less. w = 4 at the same 175 depths would add
175 × 4 s ≈ 12 minutes *per circuit*: if you want it, give it a handful of
depths of its own.

**Memory** is set by the stored values (float64, 8 bytes each) and codes:

| probe | values per circuit | bytes per circuit |
|---|---|---|
| `code_dimension` at 5957 depths | 5957 | 48 kB |
| `erasure_first_failure`, 175 depths × M = 100 | 17,500 | 140 kB |
| `uncorrectable_region_counts(3)`, 175 depths × 3 | 525 | 4 kB |
| `code_snapshot`, final layer | n × 2 × 4 words | 16 kB |

about 210 kB per circuit, 210 MB per point of 1000 circuits. Had
`erasure_first_failure` run at all 5957 depths, it would be
5957 × 100 × 8 B = 4.8 MB per circuit, **4.8 GB per point**: that is why the
expensive probes get their own sparser checkpoints. The memory is allocated
when the run starts, for the whole call, so chunking (5.3) also bounds it.

**The estimator.** `sparsegf2.api.estimate_ensemble` (the same function the
MCP server and the CLI use) takes the spec as JSON and returns a deliberately
pessimistic runtime for this machine's thread count, plus the result size. In
JSON, `"realizations": 1000` builds the same collection as `make_spec` (graph
`g` from seed `[17, g]`):

```python
from sparsegf2 import api

full_depths = np.unique(np.round(np.geomspace(1, 1e6, 10_000)).astype(np.int64))
full_sparse = np.unique(full_depths[np.searchsorted(full_depths, np.geomspace(1, 1e6, 200))])
json_spec = {
    "num_system": 256,
    "graph": {"family": "watts_strogatz", "k": 4, "beta": 1.0, "seed": 17, "realizations": 1000},
    "layers": 1_000_000,
    "measurement_rate": 0.05,
    "schedule": "random_edges",
    "reference": "purification",
    "checkpoints": full_depths.tolist(),
    "stop_when_purified": True,
    "probes": [
        {"kind": "code_dimension"},
        {"kind": "erasure_first_failure", "samples": 100, "checkpoints": full_sparse.tolist()},
        {"kind": "uncorrectable_region_counts", "max_size": 3, "checkpoints": full_sparse.tolist()},
        {"kind": "code_snapshot", "checkpoints": "final"},
    ],
}
print(len(full_depths), "depths,", len(full_sparse), "sparse depths")
estimate = api.estimate_ensemble(json_spec)
print(estimate)
print(f"about {estimate['estimated_seconds'] / 3600:.1f} h with {estimate['threads']} threads, "
      f"{estimate['result_megabytes']:.0f} MB of results")
```

On the 4-core machine used for this guide it printed about 27,600 s (7.7 h)
and 208 MB; the arithmetic above gives about 3 h, and the estimator is meant to
err on the long side (in spot checks at n = 64–128, measured runs took a third
or less of the estimate).
The estimator refuses specs whose result would exceed 8 GB.

**A pilot.** The best estimate is a measurement: run one small chunk (say 8
circuits) at the lowest and highest p of the scan with the real parameters and
read `res.elapsed_seconds`, or the `seconds` that `erasure_point.py` writes
next to each chunk.

### 5.3 Chunks, seeds and merging

Split each point into chunks of circuits, one process (or cluster job) per
(β, p, chunk). Because trajectory t depends only on (spec, seed, t), the merged
chunks are *exactly* the single big run. Rules:

* every chunk of a point uses the **same spec** (same graphs, depths, probes)
  and the **same seed**; only `first_trajectory` differs;
* use one circuit seed for the whole campaign (or at least the same seed at
  every p of a β): common random numbers across p make the curves smooth;
* name files by what they hold: `runs/n256_beta1_p0.05/chunk0003.npz`;
* use the same installed version for every chunk of a point:
  `concatenate` refuses to mix versions or randomness contracts, because
  different versions need not produce the same trajectories.

Here is the whole loop at toy size, using `main` from the script above: three
chunks of one point, merged and compared with a single run.

```python
toy = ["--beta", "1", "--p", "0.05", "--n", "16", "--circuits", "12", "--chunk-size", "5",
       "--depth-max", "200", "--num-depths", "40", "--num-sparse", "8", "--samples", "20",
       "--out", str(workdir / "runs")]
for chunk in range(3):                          # 12 circuits: chunks of 5, 5 and 2
    main(toy + ["--chunk", str(chunk)])

point_dir = workdir / "runs" / "n16_beta1_p0.05"
merged = mipt.EnsembleResult.concatenate(
    [mipt.EnsembleResult.load(f) for f in sorted(point_dir.glob("chunk*[0-9].npz"))])
merged.save(point_dir / "merged.npz")
print(merged.trajectories)                      # 0 .. 11

toy_depths = study_depths(200, 40)
single = make_spec(16, 1.0, 0.05, 12, toy_depths, sparse_depths(toy_depths, 8), samples=20).run(seed=1)
assert all(np.array_equal(single.values[k], merged.values[k], equal_nan=True) for k in single.values)
print("three chunks == one run")
```

At full size, merge each point once all its chunks exist, then analyse the
merged file exactly as in section 4.3:

<!-- not run -->
```python
point_dir = Path("runs/n256_beta1_p0.05")
parts = sorted(point_dir.glob("chunk*[0-9].npz"))
merged = mipt.EnsembleResult.concatenate([mipt.EnsembleResult.load(f) for f in parts])
assert np.array_equal(merged.trajectories, np.arange(1000)), "missing chunks"
merged.save(point_dir / "merged.npz")
```

### 5.4 On a cluster: one job per (β, p, chunk)

First, on the machine (or a node of the partition) you will run on: install as
in section 1 into a virtual environment, run the test suite once (this also
compiles the kernels into the cache, so the jobs do not each compile them), and
run a tiny chunk of the script:

```sh
python erasure_point.py --beta 1 --p 0.05 --chunk 0 --n 16 --circuits 8 --chunk-size 4 --depth-max 100 --num-depths 30 --num-sparse 6 --out test_runs
```

Then a SLURM array, one task per (β, p, chunk). With 2 values of β, 5 of p and
1000 circuits in chunks of 50, that is 2 × 5 × 20 = 200 tasks. Each chunk is
about 50 × 40 s ≈ 35 core-minutes if no circuit purifies (section 5.2), so 4
CPUs per task finish in about 10 minutes; the time limit below leaves a wide
margin. Save as `erasure.sbatch` and submit with `sbatch erasure.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=erasure
#SBATCH --array=0-199
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

BETAS=(0 1)
PS=(0.01 0.03 0.05 0.07 0.09)          # your measurement rates
CHUNKS=20                               # 1000 circuits / 50 per chunk

i=${SLURM_ARRAY_TASK_ID}
chunk=$(( i % CHUNKS ))
p=${PS[$(( (i / CHUNKS) % ${#PS[@]} ))]}
beta=${BETAS[$(( i / (CHUNKS * ${#PS[@]}) ))]}

export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}   # use exactly the CPUs you were given
source "$HOME/SparseGF2/.venv/bin/activate"
python erasure_point.py --beta "$beta" --p "$p" --chunk "$chunk" --chunk-size 50 --out runs
```

Create `logs/` before submitting (`mkdir -p logs`). A task whose chunk file
already exists exits immediately, so resubmitting the array after a failure or
a timeout reruns only the missing chunks. If your rates differ per β, give
each β its own array (and its own `PS`).

**Provenance.** Every result file records the spec, the graphs, the seed and
`sparsegf2.__version__`; the script also writes a JSON file per chunk with
the git commit, the arguments, the host and the run time. Do not `git pull`
in the middle of a campaign: keep one commit until every chunk is done (or
rerun the affected points), and note the commit in your lab book:

```sh
git -C ~/SparseGF2 rev-parse HEAD
python -c "import sparsegf2; print(sparsegf2.__version__)"
```

---

## 6. Other workflows

### 6.1 Purification time of a single reference qubit

A reference qubit Bell-paired with a random system qubit, on a collection of
Watts–Strogatz graphs; S(R) is tested after every layer and each trajectory
stops when the reference is pure. Censoring is at the last layer.

```python
n_ref = 16
pur = mipt.MonitoredCircuit(
    num_system=n_ref,
    edges=[graphs.watts_strogatz(n_ref, 4, 0.1, seed=(5, g)) for g in range(400)],
    layers=8 * n_ref,
    measurement_rate=0.1,
    schedule="random_edges",
    reference="single",
    reference_partner="random",
    scramble_matchings=16,                  # 16 layers of random all-to-all matchings first
    checkpoints="every_layer",
    stop_when_purified=True,
)
pres = pur.run(seed=2)
tau = pres.purification_layer               # -1 = never purified within 8n layers
print(f"{pres.censored.sum()} of {tau.size} censored at T = {pur.layers}")
layers, surviving = pres.survival()         # P(tau > layer) at every checkpoint, exact here
print("P(tau > 16, 32, 64):", surviving[[16, 32, 64]])
km = analysis.kaplan_meier(np.where(tau >= 0, tau, pres.final_layer), tau >= 0)
print("Kaplan-Meier at the first event times:", np.round(km["survival"][:4], 3))
```

`survival()` is exact for a single run with one horizon; use
`analysis.kaplan_meier` to combine runs with different maximum depths (pass
the censoring time, `final_layer`, for censored entries, never the −1).
[`examples/purification_time.py`](../examples/purification_time.py) is the full
study.

### 6.2 The measurement-induced transition (−I₃)

Pure-state picture, brickwork on a ring, the tripartite information of the
first three quarters at depth 4n:

```python
for size in (16, 32):
    row = []
    for p in (0.05, 0.10, 0.16, 0.25, 0.35):
        mit = mipt.MonitoredCircuit(num_system=size, edges=graphs.cycle(size), layers=4 * size,
                                    measurement_rate=p, probes=[mipt.tripartite_information()])
        mean, err = analysis.mean_and_error(-mit.run(400, seed=1).values["tripartite_information"][:, -1])
        row.append(f"{mean:5.2f} ± {err:4.2f}")
    print(f"n = {size}: -I3 =", "  ".join(row))
```

−I₃ grows with n below the critical rate and vanishes above it; the curves
cross near p ≈ 0.16 for this circuit family (the README's transition figure
shows n up to 256). [`examples/transition_scan.py`](../examples/transition_scan.py)
is the full scan.

### 6.3 A single circuit in Stim's language

`Circuit` reads Stim's circuit language; `Simulator.run` runs it in one
compiled call, `sample` runs many shots in parallel, and `observe` is called
after every `TICK` with the live state:

```python
bell = sg.Circuit("""
H 0
TICK
CX 0 1
TICK
M 0 1
DETECTOR rec[-1] rec[-2]
""")
one = sg.Simulator(2, seed=3).run(bell, observe=lambda s, tick: s.entropy([0]))
print(one.measurements, one.detectors, one.observations)   # observations: [(0, 0), (1, 1)]
shots = sg.sample(bell, shots=10_000, seed=1)
print(shots.measurements.mean(axis=0), shots.detectors.mean())   # about [0.5 0.5] and 0
```

Every ensemble trajectory can be exported this way. Replaying one with
`observe` reproduces, layer by layer, what the engine recorded (tick `t` sees
the state after layer `t`):

```python
small = mipt.MonitoredCircuit(
    num_system=8, edges=graphs.cycle(8), layers=12, measurement_rate=0.2,
    reference="purification", checkpoints="every_layer", probes=[mipt.code_dimension()])
engine = small.run(4, seed=9)
circuit = small.circuit(trajectory=2, seed=9)
replay = sg.Simulator(small.num_qubits, phases=False).run(
    circuit, observe=lambda s, tick: ob.code_dimension(s, reference=small.reference_qubits))
print([k for _, k in replay.observations])
assert [k for _, k in replay.observations] == engine.values["code_dimension"][2].tolist()
```

### 6.4 The command line

`sparsegf2` runs the same things from a shell, with JSON in and out
(`--json` prints one JSON object). An ensemble spec is the JSON form of
`MonitoredCircuit` used in section 5.2. From a scratch directory:

```sh
cat > spec.json <<'EOF'
{"num_system": 32,
 "graph": {"family": "watts_strogatz", "k": 4, "beta": 1.0, "seed": 17, "realizations": 64},
 "layers": 1024, "measurement_rate": 0.05, "schedule": "random_edges",
 "reference": "purification", "checkpoints": [1, 4, 16, 64, 256, 1024],
 "stop_when_purified": true,
 "probes": [{"kind": "code_dimension"},
            {"kind": "erasure_first_failure", "samples": 100, "checkpoints": [64, 1024]},
            {"kind": "uncorrectable_region_counts", "max_size": 3, "checkpoints": [64, 1024]},
            {"kind": "code_snapshot", "checkpoints": "final"}]}
EOF
sparsegf2 ensemble spec.json --seed 1 --output-dir runs/beta1_p0.05
sparsegf2 results runs/beta1_p0.05
```

The output directory holds `result.npz` (an `EnsembleResult`, load it with
`mipt.EnsembleResult.load`), `spec.json` (running `sparsegf2 ensemble` on it
repeats the run exactly, under the same randomness contract) and
`provenance.json` (versions, git commit, seed, randomness contract, command). The summary prints means with standard errors per checkpoint and
the erasure probes as P_C(m) curves.

For long runs, `--chunk-size` saves the run chunk by chunk and resumes where it
stopped when you run the same command again; run it in the background with
`nohup`:

```sh
nohup sparsegf2 ensemble spec.json --seed 1 --output-dir runs/chunked --chunk-size 16 > chunked.log 2>&1 &
wait
sparsegf2 results runs/chunked
```

(`wait` is only there so the next line sees the finished run; in practice you
log out and come back.) The chunked run gives exactly the numbers of the
unchunked one above, and the chunks are merged into `result.npz` when it ends.
`sparsegf2 results` also reads a directory whose job is still running, from the chunks saved so far. `sparsegf2 --help` and
`sparsegf2 ensemble --help` list everything.

### 6.5 From an AI agent

The same functionality is available to AI agents through an MCP server
(`sparsegf2-mcp`), with runtime and memory estimates and background jobs for
long ensembles: see [`MCP_GETTING_STARTED.md`](MCP_GETTING_STARTED.md), and
[`AGENTS.md`](../AGENTS.md) and [`LLM_INTERFACE.md`](LLM_INTERFACE.md) for the
reference.

---

## 7. Troubleshooting and FAQ

**The first call takes many seconds.** That is Numba compiling the kernel,
once per kernel; the result is cached in `__pycache__` (or `NUMBA_CACHE_DIR`).
If it happens on every run, the cache directory is not writable: set
`NUMBA_CACHE_DIR` to a writable path. On a cluster, warm the cache once before
submitting jobs (section 5.4).

**I ran out of memory.** The result is allocated for the whole call:
(circuits) × (Σ over probes of evaluations × values per evaluation) × 8 bytes,
plus 16 kB per stored code at n = 256. Give the vector probes
(`erasure_first_failure`, `entropy_profile`, `erasure_lost_profile`) and
`code_snapshot` fewer checkpoints of their own, and run fewer circuits per call
(chunks, section 5.3). `api.estimate_ensemble` reports `result_megabytes`.

**The job uses every core of a shared node.** Set `NUMBA_NUM_THREADS` to the
number of CPUs you were allocated before Python starts (in SLURM,
`$SLURM_CPUS_PER_TASK`), or pass `num_threads=` to `run`. Parallelism is over
trajectories, so a chunk should have at least as many circuits as threads.

**Some values are NaN after a trajectory stopped, others are not.** With
`stop_when_purified`, code probes record their exact values for a pure
reference after the stop (k = 0, nothing lost, first failure n + 1, no
uncorrectable regions), so averaging them over all circuits is correct.
Entanglement probes and `contiguous_distance` are NaN there because their
values are not determined; `result.mean(name)` and `analysis.mean_and_error`
skip NaN. `contiguous_distance` is also NaN wherever k = 0.

**What does −1 in `purification_layer` mean?** The reference never became
pure up to the last checkpoint: the trajectory is right-censored. Never
average −1 as a time; use `result.survival()` or `analysis.kaplan_meier` with
`np.where(tau >= 0, tau, result.final_layer)` and `observed = tau >= 0`.

**`res.code(t, c)` raises "had stopped".** Circuit `t` purified before that
evaluation, so there is no code to return; its snapshot count is −1.

**`concatenate` refuses my chunks.** They must come from the same spec (same
graphs, depths and probes), the same seed and the same package version, and
cover different trajectories. "different sparsegf2 versions" means the code
changed between chunks (a `git pull` mid-campaign): rerun the affected chunks
with one version. Check a file's version with
`mipt.EnsembleResult.load(path).version`.

**Can I resume an interrupted point?** Yes. With `erasure_point.py`, resubmit:
finished chunks are skipped. With the CLI, rerun the same
`sparsegf2 ensemble ... --chunk-size` command.

**Do results depend on the thread count or on the chunking?** No: trajectory
t depends only on (spec, seed, t), so any thread count and any split into
chunks give the same numbers.

**Where is the physics of the estimators explained?** In the README's
[erasure section](../README.md#4-erasure-protection-of-the-generated-code), the
module docstring of `sparsegf2.analysis`, and [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md)
for how the engine computes them.
