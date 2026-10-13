# Benchmarks and figure data

Everything in the README's performance and physics figures is reproducible
from the scripts here. Run them from the repository root, on an otherwise idle
machine (timings are single-threaded unless stated, so background load skews
them directly).

| script | what it measures | output |
|---|---|---|
| [`compare_simulators.py`](compare_simulators.py) | time per monitored circuit for SparseGF2 v3 (both modes), Stim, Qiskit Aer, Cirq and SparseGF2 v2 on identical circuits; v3's ensemble throughput | `results/simulators.json` |
| [`crosscheck_v2_production.py`](crosscheck_v2_production.py) | the PRL single-reference protocol through the v2 production engine and v3 on identical graph banks: purification-time distributions (KS test) and trajectories per second | `results/crosscheck_v2.json` |
| [`physics_data.py`](physics_data.py) | the three physics figures: −I₃ crossing, single-reference purification, code rate and erasure protection | `results/physics.json` |
| [`make_figures.py`](make_figures.py) | renders every figure, light and dark, into `v3/docs/figures/` | PNGs |
| [`prototypes/`](prototypes/) | the layout experiments that chose the kernel's data layout (see DESIGN.md §5.1) | printed tables |

```sh
pip install -e "./v3[test]" qiskit-aer cirq matplotlib
python v3/benchmarks/compare_simulators.py
python v3/benchmarks/crosscheck_v2_production.py --n 32 64 128 256 --graphs 2000 500 200 100 \
    --p 0.293 0.253 --out v3/benchmarks/results/crosscheck_v2.json
python v3/benchmarks/physics_data.py
python v3/benchmarks/make_figures.py
```

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
* **Correctness first.** For every size, v3's final stabilizer group is
  compared with Stim's (and v2's); the script aborts on any mismatch.
* **Stim's build is recorded.** The PyPI wheel of Stim 1.16 ships only an SSE2
  build; the environment block of the results names the build used. A source
  build with `-march=native` (so Stim's SIMD words use AVX2) was slower on the
  benchmark machine (n = 512: 1.16 s vs 0.97 s), so the PyPI build is the
  baseline.
