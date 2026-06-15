"""Build + execute ``notebooks/master.ipynb`` - the end-to-end walkthrough.

A single, executable record of the SparseGF2 simulator: mathematical
foundations, the sparse stabilizer core, native Sp(2n,F2) sampling, measurement
and observables, the graph-defined circuits package, text + visual circuit
inspection, a performance snapshot, a complete MIPT example, and a one-button
test run. Stim (if installed) is used only as a cross-checker.

Run from the project root::

    .venv/bin/python notebooks/_build_master.py

The script writes ``notebooks/master.ipynb`` after executing every cell, so the
artifact is reproducible from this source.
"""

from __future__ import annotations

import pathlib

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT = HERE / "master.ipynb"


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


CELLS = [
    md(r"""# SparseGF2 - master walkthrough

A single, executable tour of the simulator, top to bottom. Each section spells
out the mathematics, runs the corresponding piece of `sparsegf2`, and (where
[Stim](https://github.com/quantumlib/Stim) is installed) cross-checks against
it. The last section runs the full test suite from inside the notebook.

**SparseGF2** is a phase-free **sparse** stabilizer simulator over
$\mathbb{F}_2$: it tracks only the symplectic part of an Aaronson-Gottesman
tableau, stored sparsely, with Numba-JIT kernels and a native
$\mathrm{Sp}(2n,\mathbb{F}_2)$ sampler (no runtime Stim). On top sits a
`circuits` package for graph-defined random-Clifford + measurement circuits -
the workhorse for measurement-induced-phase-transition (MIPT) studies.

## Contents
1. Setup
2. Mathematical foundations - stabilizer formalism, the symplectic picture, the phase-free contract
3. The simulator core - `SparseGF2`, gates, Stim parity
4. Native $\mathrm{Sp}(2n,\mathbb{F}_2)$ - enumeration + sampling without Stim
5. Measurement
6. Observables - entropy, mutual information, code dimension, weight spectrum
7. The circuits package - graph-defined MIPT circuits
8. Inspecting circuits - text + visual
9. Performance
10. A complete MIPT example - the purification transition
11. Verify everything (pytest)
12. Where to go next"""),
    # ---------------------------------------------------------------
    md("## 1. Setup\n\nImport the package and confirm it's a complete install (the kernel must be one where `sparsegf2` is installed - its `circuits` subpackage and drawer should be present). Stim is optional and used only for cross-checks."),
    code(
        "import pathlib\n"
        "import sys\n"
        "\n"
        "import numpy as np\n"
        "import sparsegf2\n"
        "import sparsegf2.circuits\n"
        "\n"
        "print('sparsegf2', sparsegf2.__version__)\n"
        "assert hasattr(sparsegf2.circuits, 'draw_circuit'), 'incomplete/stale install - check the kernel'\n"
        "print('core + circuits present ✓')\n"
        "\n"
        "ROOT = pathlib.Path(sparsegf2.__file__).resolve().parents[2]\n"
        "try:\n"
        "    import stim\n"
        "    HAVE_STIM = True\n"
        "    sys.path.insert(0, str(ROOT / 'tests'))\n"
        "    from _stim_parity import assert_states_equal, fresh_stim, stim_symplectic\n"
        "    print('stim', stim.__version__, '- cross-checks enabled')\n"
        "except ImportError:\n"
        "    HAVE_STIM = False\n"
        "    print('stim not installed - cross-check cells will be skipped')\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## 2. Mathematical foundations

### 2.1 Stabilizer states
An $n$-qubit **stabilizer state** is the unique simultaneous $+1$ eigenstate of
an abelian group $\mathcal{S}$ of $2^n$ Pauli operators (not containing $-I$);
$n$ independent generators specify it. The Aaronson-Gottesman layout tracks
$2n$ generators - $n$ *destabilizers* and $n$ *stabilizers* - and updates them
in polynomial time under Clifford gates and measurements.

### 2.2 The symplectic representation over $\mathbb{F}_2$
Modulo phase, a Pauli on $n$ qubits is a pair of bit-vectors
$(x,z)\in\mathbb{F}_2^n\times\mathbb{F}_2^n$:
$$P = i^{\delta}\bigotimes_q X_q^{x_q} Z_q^{z_q}.$$
So $I\leftrightarrow(0,0)$, $X\leftrightarrow(1,0)$, $Z\leftrightarrow(0,1)$,
$Y\leftrightarrow(1,1)$. Stacking the $2n$ generators gives the binary matrix
$[X\,|\,Z]$. Two Paulis commute iff their **symplectic inner product**
$x\cdot z' + z\cdot x' \pmod 2$ vanishes, and a Clifford acts as a linear map
preserving it - an element of $\mathrm{Sp}(2n,\mathbb{F}_2)$.

### 2.3 The phase-free contract
SparseGF2 stores **only** $(x,z)$ - it drops the phase $i^\delta$. Everything
built from the GF(2) row span is then computed *exactly*: rank, entanglement
entropy, mutual information, code dimension, weight spectra. What is **not**
available is a signed expectation $\langle\psi|P|\psi\rangle$ (that needs the
sign) - `measure_z` returns `0` for every deterministic outcome regardless of
the physical eigenvalue. For signed expectations, use Stim."""),
    code(
        "from sparsegf2 import SparseGF2, symplectic_form\n"
        "\n"
        "# |0^3>: stabilizers Z0,Z1,Z2 and destabilizers X0,X1,X2 -> [X|Z] is a permuted identity\n"
        "sim = SparseGF2(3)\n"
        "M = sim.to_symplectic()            # (2n, 2n) [X | Z]\n"
        "print('shape', M.shape)\n"
        "print(M)\n"
        "print('\\nsymplectic form Omega (n=2):\\n', symplectic_form(2))\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## 3. The simulator core

`SparseGF2(n)` is a pure $|0^n\rangle$ on exactly $n$ qubits. Gates are the
named Cliffords `apply_h`, `apply_s`, `apply_sqrt_x`, `apply_cx`, `apply_cz`,
`apply_swap`, plus `apply_gate_1q` / `apply_gate_2q` for arbitrary symplectic
matrices. `canonical_form()` returns the GF(2) RREF of the stabilizer block -
the canonical representative used for state equality."""),
    code(
        "sim = SparseGF2(4, rng=np.random.default_rng(0))\n"
        "sim.apply_h(0)\n"
        "for i in range(3):\n"
        "    sim.apply_cx(i, i + 1)        # build a GHZ-like state\n"
        "print('canonical (RREF of stabilizers):')\n"
        "print(sim.canonical_form())\n"
    ),
    md(r"""### Stim parity
Stim is the ground-truth cross-checker. The simulator's contract is **RREF
parity at the level of the stabilizer subspace** - at *every* step. Apply the
same named-gate circuit to both and compare:"""),
    code(
        "if HAVE_STIM:\n"
        "    n = 5\n"
        "    sim = SparseGF2(n)\n"
        "    st = fresh_stim(n)\n"
        "    rng = np.random.default_rng(1)\n"
        "    gates = [('h', 1), ('s', 1), ('cx', 2), ('cz', 2), ('swap', 2)]\n"
        "    for _ in range(40):\n"
        "        name, ar = gates[int(rng.integers(len(gates)))]\n"
        "        if ar == 1:\n"
        "            q = int(rng.integers(n))\n"
        "            getattr(sim, {'h': 'apply_h', 's': 'apply_s'}[name])(q)\n"
        "            getattr(st, name)(q)\n"
        "        else:\n"
        "            a, b = (int(x) for x in rng.choice(n, 2, replace=False))\n"
        "            getattr(sim, {'cx': 'apply_cx', 'cz': 'apply_cz', 'swap': 'apply_swap'}[name])(a, b)\n"
        "            getattr(st, name)(a, b)\n"
        "    assert_states_equal(sim.to_symplectic(), stim_symplectic(st, n), n)\n"
        "    print('RREF stabilizer-subspace parity with Stim after 40 random gates: OK ✓')\n"
        "else:\n"
        "    print('(stim not available - skipped)')\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## 4. Native $\mathrm{Sp}(2n,\mathbb{F}_2)$ - no Stim at runtime

A phase-free Clifford **is** an element of the symplectic group. SparseGF2
generates and samples it natively:
- `enumerate_sp4()` → the cached $(720,4,4)$ table of all of
  $\mathrm{Sp}(4,\mathbb{F}_2)$ (built by a nested symplectic-basis
  construction, no Stim);
- `random_symplectic(n, rng)` → a uniform element of
  $\mathrm{Sp}(2n,\mathbb{F}_2)$ for arbitrary $n$.

$|\mathrm{Sp}(4,\mathbb{F}_2)| = 720$; the full sign-decorated 2-qubit Clifford
group has $11{,}520 = 720\times 16$ elements - the factor 16 is the sign
choices the phase-free simulator can't see, so sampling the 720 uniformly is
exactly "uniform random two-qubit Clifford" for us."""),
    code(
        "from sparsegf2 import enumerate_sp4, is_symplectic, random_symplectic, symplectic_group_order\n"
        "\n"
        "table = enumerate_sp4()\n"
        "print('Sp(4,F2):', table.shape,\n"
        "      '| all symplectic:', all(is_symplectic(table[k]) for k in range(len(table))))\n"
        "print('|Sp(2n,F2)| for n=1,2,3:', [symplectic_group_order(n) for n in (1, 2, 3)])\n"
        "\n"
        "sim = SparseGF2(6, rng=np.random.default_rng(0))\n"
        "rng = np.random.default_rng(2)\n"
        "for _ in range(10):                       # 10 random 2-qubit Cliffords\n"
        "    a, b = (int(x) for x in rng.choice(6, 2, replace=False))\n"
        "    sim.apply_gate_2q(a, b, table[int(rng.integers(720))])\n"
        "S = random_symplectic(8, np.random.default_rng(0))\n"
        "print('random_symplectic(8) is in Sp(16,F2):', is_symplectic(S), S.shape)\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## 5. Measurement

`measure_z(q)` projects onto the $Z_q$ eigenspace and returns a **phase-free**
bit: `0` for a deterministic outcome (the sign that would distinguish
$|0\rangle$ from $|1\rangle$ is not tracked), and a fair coin for a
non-deterministic one. The post-measurement *symplectic* tableau is the unique
projection - matching Stim step for step. `measure_x` / `measure_y` and
`reset_z/x/y` are also available, as is the predicate `is_deterministic_z`."""),
    code(
        "sim = SparseGF2(2, rng=np.random.default_rng(0))\n"
        "print('Z0 deterministic on |00>:', sim.is_deterministic_z(0), '-> measure_z(0) =', sim.measure_z(0))\n"
        "sim.apply_h(0); sim.apply_cx(0, 1)        # Bell state\n"
        "print('Z0 deterministic on Bell:', sim.is_deterministic_z(0))\n"
        "out0 = sim.measure_z(0)\n"
        "out1 = sim.measure_z(1)\n"
        "print('measured (q0,q1):', out0, out1, '-> correlated:', out0 == out1)\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## 6. Observables

For a stabilizer state the **entanglement entropy** of a subsystem $A$ is a
pure rank computation (Fattal-Cubitt-Yamamoto-Bravyi-Chuang 2004):
$$S(A) = \operatorname{rank}_{\mathbb{F}_2}\!\bigl([X\,|\,Z]\big|_A\bigr) - |A|,$$
an integer (in ebits). This is why a stabilizer simulator computes
entanglement in polynomial time. Built on it: `mutual_information`,
`tripartite_mutual_info`, `code_dimension` (the purification order parameter
$k=S(\text{system})$), and the stabilizer **weight spectrum**."""),
    code(
        "from sparsegf2 import (\n"
        "    average_stabilizer_weight, code_dimension, entanglement_entropy,\n"
        "    from_bell_purification, mutual_information,\n"
        ")\n"
        "\n"
        "bell = SparseGF2(2); bell.apply_h(0); bell.apply_cx(0, 1)\n"
        "print('S(one half of a Bell pair) =', entanglement_entropy(bell, [0]), 'ebit')\n"
        "print('I(q0:q1) =', mutual_information(bell, [0], [1]))\n"
        "\n"
        "# Purification: 2n qubits, a Bell pair per system qubit -> code dimension k = n\n"
        "pur = from_bell_purification(6)\n"
        "print('fresh purification code dimension k =', code_dimension(pur, 6), '(= n_system)')\n"
        "print('avg stabilizer weight (fresh purification):', round(average_stabilizer_weight(pur), 2))\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## 7. The circuits package

`sparsegf2.circuits` builds graph-defined random-Clifford + measurement
circuits. One `CircuitConfig` + a `sample_seed` fully (and reproducibly)
determines a realization; `simulate` runs it into a `SampleRecord`.

Knobs: **picture** (`pure_state` / `purification` / `single_ref`), **graph**
(`cycle` / `complete`), **gating** (`brickwork` / `random_edge`, with
`gates_per_layer`), **matching** (`round_robin` / `palette` / `fresh`),
**measurement** (`bernoulli` / `gated` / `random_pair`), plus depth and RNG
controls."""),
    code(
        "from sparsegf2.circuits import CircuitConfig, Picture, simulate\n"
        "\n"
        "cfg = CircuitConfig(graph_spec='cycle', n=16, picture=Picture.PURIFICATION,\n"
        "                    gating_mode='brickwork', measurement_mode='bernoulli',\n"
        "                    p=0.16, depth_factor=8)\n"
        "rec = simulate(cfg, sample_seed=0)\n"
        "print('picture            :', rec.picture)\n"
        "print('total_layers/gates :', rec.total_layers, '/', rec.total_gates)\n"
        "print('total_measurements :', rec.total_measurements)\n"
        "print('code_dimension k   :', rec.code_dimension)\n"
        "print('half-cut entropy   :', rec.entropy_half_cut)\n"
        "print('exp/actual ratio   :', round(rec.gate_to_meas_ratio_expected, 3),\n"
        "      '/', round(rec.gate_to_meas_ratio_actual, 3))\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## 8. Inspecting circuits

Two views of *exactly* what a config builds, so you can verify it. Both read
the same structured trace, so they always agree.

**Text** - exact and scriptable. Note `setup` shows the deterministic Bell-pair
construction, layer gates are random `C[index]` (reproducible), and
measurements list **fired** vs **candidates**:"""),
    code(
        "from sparsegf2.circuits import draw_circuit, inspect_circuit\n"
        "\n"
        "print(inspect_circuit(CircuitConfig(graph_spec='cycle', n=8, picture='single_ref', p=0.16),\n"
        "                      sample_seed=0, max_layers=3))\n"
    ),
    md(r"""**Visual** - a real circuit diagram (needs the `viz` extra / matplotlib).
Blue = deterministic setup gates, orange = random Sp(4) `C[index]`, **solid
green meter = a measurement that fired**, **faint dashed marker = a measurement
candidate that didn't fire**; reference qubits are shaded."""),
    code(
        "%matplotlib inline\n"
        "draw_circuit(CircuitConfig(graph_spec='cycle', n=8, picture='single_ref',\n"
        "                           measurement_mode='bernoulli', p=0.3, depth_factor=2),\n"
        "             sample_seed=0, max_layers=4);\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## 9. Performance

A $Z$-measurement is a **sparse rank-update** costing
$O(\text{anticommuters}\times\text{weight})$, not the dense $O(n^2)$.
Measurements keep the tableau sparse, so the advantage over a dense simulator
**grows with system size and measurement rate** - the regime MIPT studies live
in. A quick illustration (timing a measurement-heavy circuit as $n$ grows; the
per-qubit-per-layer cost stays roughly flat):"""),
    code(
        "import time\n"
        "\n"
        "from sparsegf2.circuits._clifford_table import sp4_table\n"
        "from sparsegf2.circuits.picture import setup_picture\n"
        "from sparsegf2.circuits.scheduler import CircuitBuilder\n"
        "tab = sp4_table()\n"
        "setup_picture(Picture.PURE_STATE, 4)[0].apply_gate_2q(0, 1, tab[5])  # JIT warmup\n"
        "print(f'{\"n\":>5} {\"time_s\":>9} {\"us/qubit-layer\":>16}')\n"
        "for n in (32, 64, 128, 256):\n"
        "    cfg = CircuitConfig(graph_spec='cycle', n=n, p=0.25, depth_factor=4)\n"
        "    sim, _ = setup_picture(Picture.PURE_STATE, n, rng=np.random.default_rng([1, 2]))\n"
        "    t0 = time.perf_counter()\n"
        "    for layer in CircuitBuilder(cfg, 0).layers():\n"
        "        for g, (qi, qj) in enumerate(layer.gate_pairs):\n"
        "            sim.apply_gate_2q(qi, qj, tab[int(layer.cliff_indices[g]) % len(tab)])\n"
        "        for q in layer.meas_qubits:\n"
        "            sim.measure_z(q)\n"
        "    dt = time.perf_counter() - t0\n"
        "    print(f'{n:>5} {dt:>9.4f} {dt / (n * 4 * n) * 1e6:>16.3f}')\n"
        "print('\\nFull SparseGF2-vs-Stim benchmarks (with plots): notebooks/benchmarks.ipynb')\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## 10. A complete MIPT example - the purification transition

Putting it together. In the **purification** picture the order parameter is the
code dimension $k=S(\text{system})$: the number of system qubits still
entangled with the reference. Under monitored dynamics, weak measurement
($p$ small) keeps $k$ large for a long time; strong measurement purifies it to
$k=0$. At fixed depth, $\langle k\rangle$ falls as $p$ rises - a purification
transition. We sweep $p$ and average over realizations:"""),
    code(
        "n = 12\n"
        "ps = np.linspace(0.04, 0.40, 8)\n"
        "n_samples = 40\n"
        "kbar = []\n"
        "for p in ps:\n"
        "    cfg = CircuitConfig(graph_spec='complete', n=n, picture='purification',\n"
        "                        p=float(p), depth_factor=6)\n"
        "    ks = [simulate(cfg, sample_seed=s).code_dimension for s in range(n_samples)]\n"
        "    kbar.append(float(np.mean(ks)))\n"
        "    print(f'p={p:4.2f}  <k> = {kbar[-1]:5.2f}')\n"
        "kbar = np.array(kbar)\n"
    ),
    code(
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(figsize=(7, 4.3))\n"
        "ax.plot(ps, kbar, 'o-', color='#55A868', lw=2, ms=7)\n"
        "ax.set_xlabel('measurement rate  p')\n"
        "ax.set_ylabel(r'mean code dimension  $\\langle k \\rangle = \\langle S(\\mathrm{system})\\rangle$')\n"
        "ax.set_title(f'Purification order parameter vs measurement rate  (complete graph, n={n})')\n"
        "ax.grid(True, ls=':', alpha=0.5)\n"
        "fig.tight_layout(); plt.show()\n"
        "print('Weak monitoring keeps the system entangled with the reference (k large);')\n"
        "print('strong monitoring purifies it (k -> 0). Locating the transition precisely')\n"
        "print('(finite-size scaling over n) is the job of an analysis/sweep layer.')\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## 11. Verify everything

The strongest statement of correctness is the test suite - including dozens of
**step-by-step Stim RREF-parity** cross-checks. Run it from inside the
notebook (takes ~1 minute):"""),
    code(
        "import subprocess\n"
        "import sys\n"
        "r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '--no-header'],\n"
        "                   cwd=str(ROOT), capture_output=True, text=True)\n"
        "print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:])\n"
        "print('exit code:', r.returncode)\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## 12. Where to go next

- [`notebooks/core_speedup.ipynb`](core_speedup.ipynb) - *why* measurements are
  fast, dense vs sparse, with a correctness cross-check.
- [`notebooks/benchmarks.ipynb`](benchmarks.ipynb) - the full
  SparseGF2-vs-Stim benchmark suite, **with plots**.
- [`notebooks/circuits/overview.ipynb`](circuits/overview.ipynb) - the circuits
  package architecture + a per-module deep dive.
- [`notebooks/circuits/circuit_gallery.ipynb`](circuits/circuit_gallery.ipynb)
  - rendered diagrams for every construction.
- [`notebooks/circuits/inspector.ipynb`](circuits/inspector.ipynb) - verifies
  every construction is built correctly (90 configs).

### Summary
- Phase-free **sparse** stabilizer simulator; exact for all rank-based
  observables.
- Native $\mathrm{Sp}(2n,\mathbb{F}_2)$ - **no runtime Stim**.
- A `circuits` package for graph-defined MIPT circuits, with text + visual
  inspection.
- Fast where it matters: large, deep, measurement-heavy circuits."""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    NotebookClient(
        nb, timeout=900, kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    ).execute()
    nbformat.write(nb, OUT)
    print(f"wrote + executed {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
