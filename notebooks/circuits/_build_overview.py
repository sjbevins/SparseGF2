"""Build + execute ``notebooks/circuits/overview.ipynb`` - the architecture,
data flow, MIPT framing, extension points, and a one-click test run."""

from __future__ import annotations

from _nbtools import build_and_execute, code, md

CELLS = [
    md(r"""# `sparsegf2.circuits` - package overview

This is the **map** for the circuits package: what it is, how the modules
fit together, the physics it computes, and where to extend it. Each module
has its own deep-dive notebook (linked below); start here for the whole
picture.

## What it is

`sparsegf2.circuits` builds **graph-defined random Clifford + measurement
circuits** on top of the phase-free [`SparseGF2`](../../src/sparsegf2/core/sparse_tableau.py)
core. These are the workhorse of **measurement-induced phase transition
(MIPT)** studies: alternate random two-qubit Cliffords (which generate
entanglement) with random projective measurements (which destroy it), and
watch an order parameter cross a transition as the measurement rate $p$
varies.

## The per-module notebooks

1. [`clifford_table`](clifford_table.ipynb) - $\mathrm{Sp}(4,\mathbb{F}_2)$, the gate set (the no-Stim seam)
2. [`graphs`](graphs.ipynb) - connectivity, 1-factorizations, graph6
3. [`matching`](matching.ipynb) - which 1-factor fires each layer
4. [`measurements`](measurements.ipynb) - which qubits get measured
5. [`picture`](picture.ipynb) - initial state + order parameter (FCYBC entropy)
6. [`config`](config.ipynb) - the validated knob-bag
7. [`scheduler`](scheduler.ipynb) - config + seed → layer stream
8. [`runner`](runner.ipynb) - execute + read the order parameter"""),
    md(r"""## Architecture & data flow

```
                         CircuitConfig  (validated knobs; resolves+caches the graph)
                               │
              ┌────────────────┼─────────────────┐
              ▼                ▼                  ▼
        setup_picture     CircuitBuilder     _clifford_table.sp4_table()
        (picture.py)      (scheduler.py)      (Sp(4,F2), 720, cached, no Stim)
              │                │                  │
   (SparseGF2, PictureSpec)    │ yields           │
              │           CircuitLayer ──────────►│
              │           {gate_pairs,            │
              │            cliff_indices, ────────┘ index into the table
              │            meas_qubits}
              ▼                │
        ┌─────────────────  SimulationRunner.run  (runner.py)  ─────────────────┐
        │  for each layer:  apply_gate_2q(qi,qj, table[ci]) ;  measure_z(q)      │
        │  then observables: entanglement_entropy / code_dimension              │
        └───────────────────────────────► SampleRecord (records.py) ────────────┘

  graphs.py  →  matching.py  (pick a 1-factor)
  graphs.py / measurements.py  (no simulator coupling - pure selection logic)
```

Two design seams worth naming:

- **No runtime Stim.** The gate set comes from `enumerate_sp4()` in the
  core, built natively. Verified by a source-grep test.
- **Two RNG streams.** Construction (on the builder) is decoupled from
  measurement outcomes (on the simulator), so a fixed circuit can be replayed
  with different coins."""),
    md(r"""## The golden path, end to end

The simplest complete run: `pure_state` × `cycle` × `brickwork` ×
`bernoulli`. One call, one record."""),
    code(
        "from sparsegf2.circuits import simulate, CircuitConfig\n"
        "rec = simulate(CircuitConfig(graph_spec='cycle', n=8, p=0.16))\n"
        "print(rec)\n"
    ),
    md(r"""## The physics: measurement-induced phase transitions

A monitored Clifford circuit has two competing effects:

- **Cliffords entangle.** Random two-qubit Cliffords spread entanglement
  ballistically; left alone, a subsystem's entropy grows to **volume law**
  ($S(A)\propto|A|$).
- **Measurements disentangle.** Each projective $Z$ measurement collapses a
  qubit, removing entanglement locally.

As the measurement rate $p$ increases past a critical $p_c$, the steady state
switches from **volume-law** (weakly monitored) to **area-law** (strongly
monitored) entanglement - a genuine phase transition in the *trajectory*
ensemble. Two order parameters, both phase-exact rank computations here:

- **`pure_state` → half-cut entropy** $S(\{0,\dots,n/2-1\})$: volume-law
  above $p_c$, area-law below. Grows with $n$ in the entangling phase,
  saturates in the disentangling phase.
- **`purification` → code dimension** $k=S(\text{system})$ (Gullans-Huse):
  the number of system qubits still entangled with a reference. The *time*
  for $k\to0$ diverges at $p_c$ - the purification transition.

A quick sweep of half-cut entropy vs $p$ at fixed $n$ shows the trend (a
proper transition needs finite-size scaling, which is the `analysis` layer's
job - not this package's):"""),
    code(
        "import numpy as np\n"
        "from sparsegf2.circuits import simulate, CircuitConfig\n"
        "n = 16\n"
        "print('    p    mean half-cut S over 10 samples')\n"
        "for p in (0.02, 0.08, 0.16, 0.30, 0.50):\n"
        "    cfg = CircuitConfig(graph_spec='complete', n=n, p=p, depth_factor=6)\n"
        "    S = np.mean([simulate(cfg, sample_seed=s).entropy_half_cut for s in range(10)])\n"
        "    bar = '#' * int(round(S))\n"
        "    print(f'{p:>5.2f} {S:>8.2f}  {bar}')\n"
        "print('\\nHigh entanglement at small p (entangling phase) -> low at large p (monitored).')\n"
    ),
    md(r"""## Design principles (and how to extend)

The package is deliberately **lean**: every line is live code; unimplemented
options are one-line *extension points*, not pre-committed stubs. To add:

| To add… | Edit… | Pattern |
|---|---|---|
| a graph family (`path`, `lattice_2d`, from-networkx) | `graphs.py` `_GRAPH_CONSTRUCTORS` | return a `GraphTopology` (edges, optional 1-factorization + sampler) |
| a picture (`single_ref`) | `picture.py` `Picture` enum + `setup_picture` | add a branch returning `(sim, PictureSpec)` |
| a measurement mode | `measurements.py` `sample_measurements` | add a branch + register in `MEASUREMENT_MODES` |
| a gating mode | `scheduler.py` `_place_gates` | add a branch returning `(pairs, cliff_idx)` |

Invariants any extension must keep: **no runtime Stim**, **phase-free**
(rank-based observables only), validate inputs with **`InvalidArgumentError`**,
and preserve the scheduler's **draw order** for reproducibility."""),
    md(r"""## One-click verification

Run the circuits test suite from inside the notebook - the same 105 tests
that gate every change (no runtime Stim, determinism, every mode end-to-end,
eager validation, observable invariants)."""),
    code(
        "import pathlib\n"
        "import subprocess\n"
        "import sys\n"
        "\n"
        "import sparsegf2\n"
        "\n"
        "# Locate the project root from the installed package, so this cell works\n"
        "# regardless of the kernel's working directory (notebooks have no __file__).\n"
        "# sparsegf2.__file__ -> .../src/sparsegf2/__init__.py\n"
        "PROJECT_ROOT = pathlib.Path(sparsegf2.__file__).resolve().parents[2]\n"
        "r = subprocess.run(\n"
        "    [sys.executable, '-m', 'pytest', 'tests/circuits', '-q', '--no-header'],\n"
        "    cwd=str(PROJECT_ROOT), capture_output=True, text=True,\n"
        ")\n"
        "print('ran circuits test suite from the project root')\n"
        "print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-400:])\n"
        "print('exit code:', r.returncode)\n"
    ),
    md(r"""## Summary

- `sparsegf2.circuits` = graph-defined random Clifford + measurement circuits
  on the phase-free core, for MIPT studies.
- Clean data flow: **config → (picture, builder, table) → runner → record**,
  with no runtime Stim and two decoupled RNG streams.
- Two order parameters, both phase-exact: half-cut entropy and code
  dimension.
- Lean by design - extend at the documented points, keep the invariants.

For the math behind each piece, follow the per-module notebooks linked at
the top. For *why the core simulator is fast* (the sparse-RREF measurement
update), see the forthcoming core-speedup notebook."""),
]

if __name__ == "__main__":
    build_and_execute("overview.ipynb", CELLS, timeout=300)
