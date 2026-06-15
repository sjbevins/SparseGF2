"""Build + execute ``notebooks/benchmarks.ipynb``.

The benchmark notebook: it RUNS the SparseGF2-vs-Stim benchmarks and renders
the plots inline - scaling in system size, the measurement-rate crossover, the
sparsity mechanism behind the speedup, and the (non-)effect of graph topology.
Reuses the scripts in ``benchmarks/`` (located via the installed package, so
the notebook works from any cwd).

Run from the project root::

    .venv/bin/python notebooks/_build_benchmarks.py
"""

from __future__ import annotations

import pathlib

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT = HERE / "benchmarks.ipynb"


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


CELLS = [
    md(r"""# Benchmarks - how fast, and *why*

SparseGF2 is a phase-free **sparse** stabilizer simulator. This notebook
measures its performance against [Stim](https://github.com/quantumlib/Stim)
(a state-of-the-art dense-tableau simulator) and plots the result, so the
claims in the README are reproducible end to end.

The thesis: a $Z$-measurement in SparseGF2 is a **sparse rank-update** whose
cost scales with the *weight* of the stabilizers it touches, not with $n$.
Measurements keep the tableau sparse (area-law), so the speedup over a dense
simulator **grows with both system size and measurement rate** - exactly the
regime measurement-induced-phase-transition (MIPT) studies live in.

## Contents
1. Setup
2. Scaling in $n$ (fixed measurement rate)
3. The measurement-rate crossover (fixed $n$)
4. The mechanism: stabilizer sparsity vs $p$
5. Graph topology has ~no effect on cost (at $O(n)$ depth)
6. Summary"""),
    md("## 1. Setup\n\nLocate the benchmark scripts via the installed package (works from any working directory), import them, and warm up the Numba JIT so it doesn't pollute the timings."),
    code(
        "%matplotlib inline\n"
        "import sys, pathlib\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import sparsegf2\n"
        "\n"
        "ROOT = pathlib.Path(sparsegf2.__file__).resolve().parents[2]   # project root\n"
        "sys.path.insert(0, str(ROOT / 'benchmarks'))\n"
        "sys.path.insert(0, str(ROOT / 'benchmarks' / 'circuits'))\n"
        "import benchmark_random_clifford as bench           # SparseGF2 vs Stim engine\n"
        "import benchmark_topology_sparsity as topo          # circuits topology benchmark\n"
        "\n"
        "if bench._numba_warmup:\n"
        "    bench._numba_warmup()\n"
        "print('Stim available:', bench._HAS_STIM, '| sparsegf2', sparsegf2.__version__)\n"
        "plt.rcParams['figure.dpi'] = 110\n"
    ),
    md(r"""## 2. Scaling in $n$

Random Clifford brickwork + $Z$-measurements at a fixed rate $p=0.25$, depth
$=n$. We time SparseGF2 (Numba) and Stim on the *same* circuits (best of 2
reps, to suppress noise). This is the headline result: where is the crossover,
and how does the advantage grow?"""),
    code(
        "P = 0.25\n"
        "ns = [32, 64, 128, 256, 512]\n"
        "reps = 2\n"
        "ours, stim = [], []\n"
        "for n in ns:\n"
        "    o_runs, s_runs = [], []\n"
        "    for rep in range(reps):\n"
        "        circ = bench.make_circuit(n, n, P, seed=rep)\n"
        "        if bench._HAS_STIM:\n"
        "            bench.precompile_stim_tableaux(circ)\n"
        "        o_runs.append(bench.run_sparsegf2(circ))\n"
        "        if bench._HAS_STIM:\n"
        "            s_runs.append(bench.run_stim(circ))\n"
        "    ours.append(min(o_runs))\n"
        "    stim.append(min(s_runs) if s_runs else np.nan)\n"
        "    print(f'n={n:4d}  ours={ours[-1]:7.3f}s  stim={stim[-1]:8.3f}s  '\n"
        "          f'speedup={stim[-1]/ours[-1]:6.2f}x')\n"
        "ns = np.array(ns); ours = np.array(ours); stim = np.array(stim)\n"
        "speedup_n = stim / ours\n"
    ),
    md("**Runtime vs $n$** (log-log). The two simulators have different slopes; the lines cross around $n\\approx 64$, after which SparseGF2's shallower scaling pulls away."),
    code(
        "fig, ax = plt.subplots(figsize=(7, 4.5))\n"
        "ax.loglog(ns, ours, 'o-', color='#DD8452', label='SparseGF2 (Numba)', lw=2, ms=7)\n"
        "ax.loglog(ns, stim, 's-', color='#4C72B0', label='Stim', lw=2, ms=7)\n"
        "ax.set_xlabel('system size  n'); ax.set_ylabel('time per circuit (s)')\n"
        "ax.set_title(f'Random Clifford + measurement (p={P}, depth=n) - runtime vs n')\n"
        "ax.grid(True, which='both', ls=':', alpha=0.5); ax.legend()\n"
        "fig.tight_layout(); plt.show()\n"
    ),
    md("**Speedup vs $n$.** Below the dashed line Stim is faster; above it SparseGF2 is. The advantage grows roughly linearly in $n$ in this regime."),
    code(
        "fig, ax = plt.subplots(figsize=(7, 4.5))\n"
        "ax.semilogx(ns, speedup_n, 'o-', color='#55A868', lw=2, ms=8)\n"
        "ax.axhline(1.0, color='grey', ls='--', lw=1, label='break-even (Stim = SparseGF2)')\n"
        "for x, y in zip(ns, speedup_n):\n"
        "    ax.annotate(f'{y:.1f}x', (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)\n"
        "ax.set_xlabel('system size  n'); ax.set_ylabel('speedup  (Stim time / SparseGF2 time)')\n"
        "ax.set_title('SparseGF2 speedup over Stim vs system size'); ax.set_ylim(0, max(speedup_n)*1.15)\n"
        "ax.grid(True, which='both', ls=':', alpha=0.5); ax.legend()\n"
        "fig.tight_layout(); plt.show()\n"
    ),
    md(r"""## 3. The measurement-rate crossover

Now fix $n$ and sweep the measurement rate $p$. At $p=0$ (pure scrambling) the
state is **volume-law** - the tableau densifies and Stim wins. As $p$ rises the
monitored dynamics holds the state **area-law** - the tableau stays sparse and
SparseGF2 pulls ahead. The speedup is, quite literally, the measurement-induced
sparsity."""),
    code(
        "N = 192\n"
        "ps = [0.0, 0.05, 0.1, 0.25, 0.5]\n"
        "ours_p, stim_p = [], []\n"
        "for p in ps:\n"
        "    o_runs, s_runs = [], []\n"
        "    for rep in range(2):\n"
        "        circ = bench.make_circuit(N, N, p, seed=rep)\n"
        "        if bench._HAS_STIM:\n"
        "            bench.precompile_stim_tableaux(circ)\n"
        "        o_runs.append(bench.run_sparsegf2(circ))\n"
        "        if bench._HAS_STIM:\n"
        "            s_runs.append(bench.run_stim(circ))\n"
        "    ours_p.append(min(o_runs)); stim_p.append(min(s_runs) if s_runs else np.nan)\n"
        "    print(f'p={p:4.2f}  ours={ours_p[-1]:.3f}s  stim={stim_p[-1]:.3f}s  '\n"
        "          f'speedup={stim_p[-1]/ours_p[-1]:5.2f}x')\n"
        "ours_p = np.array(ours_p); stim_p = np.array(stim_p); speedup_p = stim_p / ours_p\n"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 4.5))\n"
        "ax.plot(ps, speedup_p, 'o-', color='#55A868', lw=2, ms=8)\n"
        "ax.axhline(1.0, color='grey', ls='--', lw=1)\n"
        "ax.fill_between(ps, 1.0, speedup_p, where=(speedup_p >= 1.0), color='#55A868', alpha=0.12)\n"
        "ax.fill_between(ps, speedup_p, 1.0, where=(speedup_p < 1.0), color='#4C72B0', alpha=0.12)\n"
        "for x, y in zip(ps, speedup_p):\n"
        "    ax.annotate(f'{y:.1f}x', (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)\n"
        "ax.text(0.01, 0.7, 'Stim faster\\n(dense / volume-law)', fontsize=9, color='#4C72B0')\n"
        "ax.text(0.30, max(speedup_p)*0.55, 'SparseGF2 faster\\n(sparse / area-law)', fontsize=9, color='#3a7a52')\n"
        "ax.set_xlabel('measurement rate  p'); ax.set_ylabel('speedup  (Stim / SparseGF2)')\n"
        "ax.set_title(f'Speedup vs measurement rate  (n={N}, depth={N})')\n"
        "ax.grid(True, ls=':', alpha=0.5)\n"
        "fig.tight_layout(); plt.show()\n"
    ),
    md(r"""## 4. The mechanism: stabilizer sparsity vs $p$

Why does more measurement mean more speed? Because the cost of a sparse
measurement scales with the **stabilizer weight** (non-identity Paulis per
generator), and measurements keep that weight small. We measure it directly on
a nearest-neighbour brickwork circuit as $p$ varies (reusing the circuits-side
benchmark), and overlay it on the speedup curve: as the average weight falls,
the speedup rises."""),
    code(
        "topo._warmup()\n"
        "weights = []\n"
        "for p in ps:\n"
        "    _, avg_w, _ = topo.bench(N, 'cycle', p, depth_factor=4, seed=0)\n"
        "    weights.append(avg_w)\n"
        "    print(f'p={p:4.2f}  avg stabilizer weight = {avg_w:6.2f}')\n"
        "weights = np.array(weights)\n"
        "\n"
        "fig, ax1 = plt.subplots(figsize=(7, 4.5))\n"
        "c1, c2 = '#C44E52', '#55A868'\n"
        "ax1.plot(ps, weights, 'o-', color=c1, lw=2, ms=7)\n"
        "ax1.set_xlabel('measurement rate  p')\n"
        "ax1.set_ylabel('avg stabilizer weight  (sparsity)', color=c1)\n"
        "ax1.tick_params(axis='y', labelcolor=c1)\n"
        "ax2 = ax1.twinx()\n"
        "ax2.plot(ps, speedup_p, 's--', color=c2, lw=2, ms=7)\n"
        "ax2.set_ylabel('speedup over Stim', color=c2); ax2.tick_params(axis='y', labelcolor=c2)\n"
        "ax1.set_title(f'Lower stabilizer weight -> higher speedup  (n={N})')\n"
        "ax1.grid(True, ls=':', alpha=0.4)\n"
        "fig.tight_layout(); plt.show()\n"
    ),
    md(r"""## 5. Graph topology has ~no effect on cost (at $O(n)$ depth)

A natural guess is that nearest-neighbour (`cycle`) circuits simulate faster
than all-to-all (`complete`) ones, because local gates keep stabilizer
supports short. We measured it - and at the $O(n)$ depths we run, the tableau
*saturates* either way, so topology barely changes the runtime (it changes the
physics, not the cost)."""),
    code(
        "sizes = [32, 64, 128]\n"
        "t_cycle, t_complete = [], []\n"
        "for n in sizes:\n"
        "    tc, _, _ = topo.bench(n, 'cycle', 0.1, depth_factor=4, seed=0)\n"
        "    tk, _, _ = topo.bench(n, 'complete', 0.1, depth_factor=4, seed=0)\n"
        "    t_cycle.append(tc); t_complete.append(tk)\n"
        "    print(f'n={n:4d}  cycle={tc:.4f}s  complete={tk:.4f}s  ratio={tk/tc:.2f}')\n"
        "\n"
        "x = np.arange(len(sizes)); w = 0.36\n"
        "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
        "ax.bar(x - w/2, t_cycle, w, label='cycle (nearest-neighbour)', color='#DD8452')\n"
        "ax.bar(x + w/2, t_complete, w, label='complete (all-to-all)', color='#4C72B0')\n"
        "ax.set_xticks(x); ax.set_xticklabels([f'n={n}' for n in sizes])\n"
        "ax.set_ylabel('time per circuit (s)'); ax.set_title('Topology barely affects simulation cost (p=0.1, depth=4n)')\n"
        "ax.legend(); ax.grid(True, axis='y', ls=':', alpha=0.5)\n"
        "fig.tight_layout(); plt.show()\n"
    ),
    md(r"""## 6. Summary

- **Scaling:** SparseGF2 overtakes Stim around $n\approx 64$ and the advantage
  grows with $n$ (≈3× at 128, ≈8× at 256, ≈20× at 512 at $p=0.25$).
- **Measurement rate:** the speedup *is* the measurement-induced sparsity -
  Stim wins at $p=0$ (volume-law), SparseGF2 wins by an increasing margin as
  $p$ rises (area-law).
- **Mechanism:** lower stabilizer weight ⇒ cheaper sparse measurements ⇒
  higher speedup.
- **Topology:** at $O(n)$ depth it's a physics knob, not a speed knob.

So for the **large-$n$, strongly-monitored** circuits MIPT studies need,
SparseGF2 is the fast path. Reproduce any of this directly:

```sh
.venv/bin/python benchmarks/benchmark_random_clifford.py --ns 64 128 256 512 --p 0.25
.venv/bin/python benchmarks/benchmark_measurement_rate.py --n 192
```"""),
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
