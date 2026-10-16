# Benchmarks and figure data

Everything in the README's performance and physics figures is reproducible
from the scripts here. Run them from the repository root, on an otherwise idle
machine (timings are single-threaded unless stated, so background load skews
them directly).

| script | what it measures | output |
|---|---|---|
| [`compare_simulators.py`](compare_simulators.py) | time per monitored circuit for SparseGF2 (both modes), Stim, Qiskit Aer and Cirq on identical circuits; SparseGF2's ensemble throughput | `results/simulators.json` |
| [`physics_data.py`](physics_data.py) | the three physics figures: −I₃ crossing, single-reference purification, code rate and erasure protection | `results/physics.json` |
| [`why_data.py`](why_data.py) | the two "why this simulator" figures: the cost of measuring the state after every layer (SparseGF2 vs Stim, same circuits), and the cost of erasure screening per code (SparseGF2 kernels vs a compiled from-scratch baseline) | `results/why.json` |
| [`thread_scaling.py`](thread_scaling.py) | wall time, throughput, speedup and parallel efficiency of ensembles (three workloads) and shot sampling (four workloads in all; the run is timed, result digests are not) at 1, 2, 4, 8, 16 and `NUMBA_NUM_THREADS` threads; aborts if results differ between thread counts | `results/thread_scaling.json`, `results/thread_scaling.md` |
| [`erasure_cost.py`](erasure_cost.py) | one-thread seconds per code of the exhaustive erasure counts (w = 1, 2, ... per n and p, through `StabilizerCode` and through the engine probe) and of first-failure orderings (M = 100, 512, 2048), on codes from Watts–Strogatz trajectories, n = 32 to 512; the cutoff m\*(n) and seconds per code of region budgets B; an example study total; levels it does not run are extrapolated and labelled so | `results/erasure_cost.json`, `results/erasure_cost.md` |
| [`probe_cost.py`](probe_cost.py) | one-thread seconds per evaluation of every probe kind, n = 64 to 2048, in the states that make each expensive (single reference with k = 1 and erasures failing near n / 2; purification, k = n; a monitored purification code), set against `estimate_ensemble`'s cost model (`api._trajectory_us`); `--from-json` prices saved measurements with the current model | `results/probe_cost.json`, `results/probe_cost.md` |
| [`correlated_erasure_baseline.py`](correlated_erasure_baseline.py) | the geometry-correlated erasure baseline on nearest-neighbour brickwork ring codes (n = 32 to 256, p = 0.04 to 0.3; the contiguous lengths to n = 1024) against the literature: P_{C,J}(m) for J = 0, 0.5, 1, 2 and the exact J -> inf intervals, q+ and the code rate, the smallest failing interval, Gullans-Huse's ⟨l⟩ and Li-Fisher's \|A\|* with fitted exponents in n, distance bounds, Φ_J(ε) and ε_δ(J); Watts-Strogatz circuits (β = 0.1, 1) for contrast; physics, not timing | `results/correlated_erasure_baseline.json`, `results/correlated_erasure_baseline.md` |
| [`make_figures.py`](make_figures.py) | renders every figure, light and dark, into `docs/figures/`, and the exact numbers into `results/tables.md` (each input JSON is optional) | PNGs, `results/tables.md` |

```sh
pip install -e ".[test]" qiskit-aer cirq matplotlib
python benchmarks/compare_simulators.py
python benchmarks/physics_data.py
python benchmarks/why_data.py          # about 4 minutes, one thread
python benchmarks/thread_scaling.py    # about 15 minutes, all cores; idle machine
python benchmarks/erasure_cost.py      # about 20 minutes, one thread; idle machine
python benchmarks/probe_cost.py        # about 6 minutes, one thread; idle machine
python benchmarks/correlated_erasure_baseline.py  # about 20 minutes, all cores
python benchmarks/make_figures.py
```

`thread_scaling.py --quick --out <dir>` is its smoke test (a few seconds once compiled), and so
are `erasure_cost.py --quick --out <dir>`, `probe_cost.py --quick --out <dir>` and
`correlated_erasure_baseline.py --quick --out <dir>`.

`why_data.py --quick --out <file>` is a smoke test (tiny sizes, under a
minute including compilation); pass that file to `make_figures.py --why <file>
--outdir <dir>` to check the rendering without touching the committed copies.

## How the simulator comparison is kept fair

* **Same circuits.** Every simulator runs identical two-qubit Cliffords and
  identical measurement locations. SparseGF2 applies each random Clifford as
  one fused gate (`CLIFFORD_2Q`), which is part of what it offers. Stim gets
  each Clifford in its tightest form, at most four of its named single-qubit
  Cliffords plus one of CX / ISWAP / SWAP (4.9 gates on average, against 12.7
  for `Tableau.to_circuit("elimination")`), with each layer's gates batched
  into a few instructions; all 11,520 decompositions are checked against the
  Clifford's tableau. Qiskit Aer and Cirq get the H/S/CX elimination form.
* **Each simulator's fastest documented path.** Stim runs one prebuilt
  `stim.Circuit` through `TableauSimulator.do_circuit`, measurements batched
  per layer. Qiskit Aer runs a prebuilt `QuantumCircuit` with
  `method="stabilizer"` (its per-run circuit conversion is part of its API and
  therefore included). Cirq runs `CliffordSimulator.simulate` on a prebuilt
  circuit. SparseGF2 runs a prebuilt `Circuit` whose compilation is cached by a
  warm-up run.
* **One thread each**, median of repeated runs after a warm-up. The ensemble
  rows are labelled with their thread count.
* **Correctness first.** For every size, SparseGF2's final stabilizer group is
  compared with Stim's; the script aborts on any mismatch.
* **Stim's build is recorded.** The PyPI wheel of Stim 1.16 ships only an SSE2
  build; the environment block of the results names the build used. A source
  build with `-march=native` (so Stim's SIMD words use AVX2) was slower on the
  benchmark machine (n = 512: 1.16 s vs 0.97 s), so the PyPI build is the
  baseline.

## The "why" measurements (`why_data.py`)

**Observing every layer** (`why-observe`). The circuit family above (ring,
brickwork, uniform random two-qubit Cliffords, Z measurements at p = 0.16),
n = 32 to 1024, min(2n, 256) layers (`--max-layers`), measured after every
layer: (a) the half-cut entropy S(first n/2 qubits), no reference; (b) the code
dimension k = S(R) with every system qubit Bell-paired with a reference (2n
qubits). Each simulator is timed evolving only and evolving plus measuring.

* **Same circuits.** SparseGF2 runs the `mipt` engine (`num_threads=1`,
  probes at `checkpoints="every_layer"`; evolve only = `checkpoints="final"`,
  no probes). Every circuit of the batch is exported with
  `spec.circuit(t, seed)` and split at its `TICK`s into prebuilt per-layer
  `stim.Circuit`s, each two-qubit Clifford in Stim's tightest form (as in the
  simulator comparison).
* **Stim's fastest route to the observable.** After each layer the
  stabilizer columns come from
  `current_inverse_tableau().to_numpy(bit_packed=True)`: row q of the inverse
  tableau's `z2x` (`x2x`) block is the X (Z) column of qubit q over the
  stabilizers, so no inverse is computed. In the full run this was 10 to 29 times (n = 32) and 510 to 900 times
  (n = 1024) faster
  than `canonical_stabilizers()` or `...inverse().to_numpy()`; all three
  routes are timed once per size and recorded under
  `stim_extraction_options`. The GF(2) rank is a numba-compiled elimination on
  64-bit words, the same algorithm SparseGF2 uses for a region entropy, so
  the difference is data layout and integration, not a slow rank in Python.
  Stim's evolve-only time is one prebuilt circuit through `do_circuit`.
* **What differs by design.** SparseGF2 keeps the reference rows of the
  tableau separated, so k = S(R) is a popcount and costs almost nothing;
  Stim has no such invariant and needs a rank of 2n columns of 2n bits per
  layer. The half-cut entropy is a rank for both. Stim tracks signs and
  samples outcomes; SparseGF2's engine is phase-free (the unsigned group,
  and so every entropy, does not depend on either).
* **Checked.** For every size and circuit, Stim's per-layer values must
  equal SparseGF2's exactly, or the script aborts. The JSON records Stim's
  observed time split into evolve / extract / rank.
* **"Measuring adds" in the figure.** For Stim it is the timed extract +
  rank phases of the observed run (the work measuring adds, measured
  directly). SparseGF2 evaluates probes inside its compiled layer loop, so its
  added time is the observed minus the evolve-only time, shown as "almost
  nothing" when that is under 5 % of the evolve time (timing noise; at
  n = 512 the code-dimension run measured slightly faster than evolve only).
  Stim's evolve-only time (one prebuilt circuit) and the evolve phase of its
  per-layer run can differ: for the code dimension at n = 1024 they were
  4.22 s and 3.46 s per circuit, so the difference of Stim's two totals
  understates its measuring cost there.

**Erasure screening per code** (`why-erasure`). Codes made by the engine
(purification picture, Watts-Strogatz graph k = 4, beta = 0.1, p = 0.05, n/2
layers, `code_snapshot`), n = 64 to 512, three codes per size, loaded as
`StabilizerCode`. Timed: `uncorrectable_counts(2)`, `uncorrectable_counts(3)`
and `first_failure_sizes(100)`, against compiled baselines on the same
compressed columns (`code_columns`) that evaluate
l(A) = rank(N(S)|_A) - rank(S|_A) from scratch: every subset of size at most
w for the counts (no pruning, no incremental bases, no hash lookups), and,
for the first failure, the same orderings (same PRNG draws) searched by
doubling then bisection on the prefix length. The baselines must reproduce the
kernels' counts and first-failure sizes exactly. A baseline whose estimated
time exceeds `--baseline-limit` (120 s) is skipped and recorded as skipped.

## The erasure cost measurements (`erasure_cost.py`)

What choosing the exhaustive cutoff m\*(n) and the number of orderings M
costs, per code, on one thread.

* **Codes from real trajectories.** n = 32 to 512, p = 0.05 and 0.1,
  Watts–Strogatz graphs (k = 4, beta = 0.1, one graph per trajectory),
  purification picture, `random_edges`, 4n layers, `code_snapshot` at the
  final layer; the first 8 trajectories with k > 0 per (n, p). Their k and
  distance bounds are recorded.
* **Two paths.** `StabilizerCode.uncorrectable_counts(w)` and
  `first_failure_sizes(M)` on the saved code; and the engine probe, timed as
  the trajectory rerun with R copies of the probe at its final layer minus the
  same run without probes, divided by R. The engine's counts must equal the
  library's, or the script aborts.
* **How far w goes.** w rises until one evaluation takes over 10 s. A level
  is not run when N_exact(n, w) exceeds 1e9, or when the walk model predicts
  over 90 s per evaluation. Slow levels are measured on the first two to four
  codes only. The next level is extrapolated with the walk model and
  labelled as an extrapolation; a per-code model check beside it shows how
  far to trust it.
* **Derived tables.** m\*(n) and the seconds per code for budgets B = 1e4 to
  1e7, and an example study total that divides by the effective threads of
  `results/thread_scaling.json`. `--from-json <file>` recomputes these from a
  results file without measuring again.
