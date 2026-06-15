"""Build + execute ``notebooks/circuits/runner.ipynb`` - the simulation loop,
two independent RNG streams, observables, and the SampleRecord."""

from __future__ import annotations

from _nbtools import build_and_execute, code, md

CELLS = [
    md(r"""# `sparsegf2.circuits.runner` - execute a realization, read the order parameter

[`runner.py`](../../src/sparsegf2/circuits/runner.py) is the thin layer that
walks a `CircuitBuilder`'s layer stream on a `SparseGF2` instance, then reads
the picture's order parameter into a `SampleRecord`. It is where the
[`scheduler`](scheduler.ipynb), [`picture`](picture.ipynb), and
[`clifford_table`](clifford_table.ipynb) pieces meet the core simulator.

## Contents
1. `simulate` and `SimulationRunner`
2. The **two RNG streams** (construction vs. outcome) and the independence trick
3. The run loop - warmup, gates, measurements
4. Observables - half-cut entropy and code dimension
5. The `SampleRecord` union shape
6. Reproducibility, injected tables, `save_tableau`"""),
    md(r"""## 1. `simulate` and `SimulationRunner`

```python
def simulate(config, *, sample_seed=0, save_tableau=False) -> SampleRecord:
    return SimulationRunner(config).run(sample_seed, save_tableau=save_tableau)
```

`simulate` is the one-liner entry point. `SimulationRunner` lets you build
once and run many seeds, sharing the (process-cached) $\mathrm{Sp}(4)$ table;
the table is also **injectable** for tests or custom gate sets."""),
    code(
        "from sparsegf2.circuits import CircuitConfig, Picture, simulate\n"
        "cfg = CircuitConfig(graph_spec='cycle', n=8, picture=Picture.PURE_STATE,\n"
        "                    gating_mode='brickwork', matching_mode='round_robin',\n"
        "                    measurement_mode='bernoulli', p=0.16, depth_factor=4)\n"
        "rec = simulate(cfg, sample_seed=0)\n"
        "print('total_layers       :', rec.total_layers)       # depth_factor*n = 32\n"
        "print('total_gates        :', rec.total_gates)         # 32 * n/2 = 128\n"
        "print('total_measurements :', rec.total_measurements)\n"
        "print('entropy_half_cut   :', rec.entropy_half_cut)    # <= n/2\n"
        "print('code_dimension     :', rec.code_dimension)      # None for pure_state\n"
        "print('ratio exp/act      :', round(rec.gate_to_meas_ratio_expected,3),\n"
        "      '/', round(rec.gate_to_meas_ratio_actual,3))\n"
    ),
    md(r"""## 2. Two RNG streams - and why they're independent

A run uses **two** generators, by design:

- **Circuit construction** - the `CircuitBuilder`, seeded `base_seed +
  sample_seed`. Decides gate pairs, Clifford indices, measured qubits.
- **Measurement outcomes** - lives on the `SparseGF2` instance, seeded
  *independently* in the runner as
  ```python
  np.random.default_rng([base_seed + sample_seed, 0x6D656173])   # "meas"
  ```

Why the two-element seed? A plain integer seed `base+sample` is what the
*builder* uses. If the outcome stream were seeded `base+sample+2` (a tempting
"offset"), it would **collide** across samples: sample 0's outcome seed
(`base+2`) would equal sample 2's *construction* seed (`base+2`), correlating
the two. A **two-element** seed `[base+sample, KEY]` goes through numpy's
`SeedSequence` hash and shares entropy with no plain integer seed - so every
sample's two streams are mutually independent, and independent across
samples. Empirically the first draws are uncorrelated:"""),
    code(
        "import numpy as np\n"
        "KEY = 0x6D656173\n"
        "# construction seed vs outcome seed for the same sample: different streams\n"
        "for s in (0, 1, 2):\n"
        "    constr = np.random.default_rng(42 + s).random()\n"
        "    outcome = np.random.default_rng([42 + s, KEY]).random()\n"
        "    print(f'sample {s}: construction first-draw={constr:.4f}  outcome first-draw={outcome:.4f}')\n"
        "# no collision: outcome seed [42+s, KEY] never equals any plain int seed 42+s'\n"
        "print('two-element seeds never collide with the builder\\'s int seeds.')\n"
    ),
    md(r"""## 3. The run loop

```python
sim_rng = np.random.default_rng([base_seed + sample_seed, _MEAS_STREAM_KEY])
sim, spec = setup_picture(picture, n, rng=sim_rng, pivot_mode=..., use_numba=...)
builder = CircuitBuilder(config, sample_seed)

for wlayer in builder.warmup_layers_iter():          # gate-only pre-scramble
    for (qi, qj), ci in ...: sim.apply_gate_2q(qi, qj, table[ci])

for layer in builder.layers():                       # measured phase
    for (qi, qj), ci in ...: sim.apply_gate_2q(qi, qj, table[ci])
    for q in layer.meas_qubits: sim.measure_z(q)      # outcome <- sim_rng
```

The Clifford index is taken `% len(table)` - a harmless safety net (indices
are already in `[0, n_cliffords) ⊆ [0, len(table))`). `measure_z` draws its
phase-free coin from `sim_rng`, the outcome stream. `total_gates` counts only
the **measured** phase (warmup is pre-scrambling, not part of the reported
depth)."""),
    md(r"""## 4. Observables

After the loop the runner reads, based on `spec.order_parameter`:

- **half-cut entropy** $S(\{0,\dots,n/2-1\})$ - *always*, for every picture.
- **code dimension** $k=S(\text{system})$ - `"code_dimension"` (purification).
- **reference entropy** $S(\text{reference})\in\{0,1\}$ - `"ref_entropy"`
  (single_ref).

All are rank-based, phase-exact quantities (see [`picture`](picture.ipynb)).
Each picture fills its own slot and leaves the others `None`:"""),
    code(
        "purif = CircuitConfig(graph_spec='cycle', n=8, picture='purification', p=0.16, depth_factor=4)\n"
        "sref  = CircuitConfig(graph_spec='cycle', n=8, picture='single_ref', p=0.16, depth_factor=4)\n"
        "rp = simulate(purif, sample_seed=3)\n"
        "rr = simulate(sref, sample_seed=3)\n"
        "rs = simulate(cfg, sample_seed=3)\n"
        "print('purification: code_dimension =', rp.code_dimension, '| ref_entropy =', rp.ref_entropy, '| half =', rp.entropy_half_cut)\n"
        "print('single_ref  : code_dimension =', rr.code_dimension, '| ref_entropy =', rr.ref_entropy, '| half =', rr.entropy_half_cut)\n"
        "print('pure_state  : code_dimension =', rs.code_dimension, '| ref_entropy =', rs.ref_entropy, '| half =', rs.entropy_half_cut)\n"
    ),
    md(r"""## 5. The `SampleRecord` union shape

Every picture returns the **same** dataclass; picture-specific observables
are `None` when they don't apply. So a mixed batch loads under one schema
without branching on the picture. The contract for the future
`sparsegf2.analysis` layer: **optional fields may be added later; required
fields may not.** Required (identity + diagnostics) vs optional (observables,
runtime, side-payload):"""),
    code(
        "import dataclasses\n"
        "from sparsegf2.circuits import SampleRecord\n"
        "req = [f.name for f in dataclasses.fields(SampleRecord) if f.default is dataclasses.MISSING]\n"
        "opt = [f.name for f in dataclasses.fields(SampleRecord) if f.default is not dataclasses.MISSING]\n"
        "print('required:', req)\n"
        "print('optional:', opt)\n"
    ),
    md(r"""## 6. Reproducibility, injected tables, `save_tableau`

Same `(config, sample_seed)` → identical record (both streams are seeded
deterministically). `save_tableau=True` attaches the full
`to_symplectic()` $[X|Z]$ matrix - shape $(2N, 2N)$ for $N$ physical qubits.
And a custom Clifford table can be injected without monkeypatching:"""),
    code(
        "a = simulate(cfg, sample_seed=0); b = simulate(cfg, sample_seed=0)\n"
        "print('reproducible:', (a.total_measurements, a.entropy_half_cut) == (b.total_measurements, b.entropy_half_cut))\n"
        "rt = simulate(purif, sample_seed=0, save_tableau=True)\n"
        "print('purification final_tableau shape:', rt.final_tableau.shape, '= (2*2n, 2*2n) for N=16')\n"
        "from sparsegf2.circuits import SimulationRunner\n"
        "from sparsegf2.circuits._clifford_table import sp4_table\n"
        "runner = SimulationRunner(cfg, clifford_table=sp4_table().copy())\n"
        "print('injected-table run total_gates:', runner.run(0).total_gates)\n"
    ),
    md(r"""## Summary

- `simulate` / `SimulationRunner` execute one realization into a
  `SampleRecord`.
- **Two independent RNG streams** - construction (`base+sample`) and outcome
  (`[base+sample, KEY]`) - decouple the circuit from its measurement coins,
  with a seed trick that avoids cross-sample collisions.
- Observables are the phase-exact rank quantities; the record has a union
  shape so analysis sees one schema.

This is the top of the package. For the architecture and the MIPT framing,
see the [package overview](overview.ipynb)."""),
]

if __name__ == "__main__":
    build_and_execute("runner.ipynb", CELLS)
