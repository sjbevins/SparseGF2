"""Build + execute ``notebooks/analysis.ipynb``.

The end-to-end "run a study" walkthrough: the online workflow (analyze at
end-of-circuit, discard the tableau), the offline workflow (save tableaux,
analyze later), the augment workflow (add new observables to saved tableaux and
re-plot with no re-simulation), and a real purification-transition sweep.

Run from the project root::

    .venv/bin/python notebooks/_build_analysis.py
"""

from __future__ import annotations

import pathlib

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT = HERE / "analysis.ipynb"


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


CELLS = [
    md(r"""# Analysis & studies - running, saving, augmenting, plotting

This notebook is the end-to-end story for getting data **out** of the simulator.
A single circuit is one sample; a *study* is a parameter sweep with persisted,
re-analyzable results. There are two ways to measure observables, and you will
use both:

1. **Online** - compute observables at the end of each circuit and throw the
   tableau away. Memory-light; use it when you already know what you want.
2. **Offline (save then analyze)** - save every final tableau, then analyze it
   later. This is what makes the database *extensible*: run the expensive
   simulations once, then decide on new observables (code rate, contiguous
   distance, …) whenever you like and **merge them in without re-simulating**.

Contents:
1. Online analysis on a single circuit
2. The same analyses, offline, on a saved tableau
3. A `Study`: a persisted parameter sweep
4. **Augmenting** a study with new observables - no re-simulation
5. Plotting (auto-detects every column, including augmented ones)
6. The purification transition, start to finish"""),
    code(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from sparsegf2.circuits import CircuitConfig, simulate\n"
        "from sparsegf2.analysis import Study, analyze, plot_study, ANALYSES\n"
        "\n"
        "print('built-in analyses:')\n"
        "print(', '.join(sorted(ANALYSES)))"
    ),
    md(r"""## 1. Online analysis on a single circuit

Pass `analyses=[...]` to `simulate`. Each name resolves to a picture-aware
observable; the results land in `rec.analyses`. The tableau is built, measured,
read, and discarded - nothing large is kept."""),
    code(
        "cfg = CircuitConfig(graph_spec='cycle', n=24, picture='purification',\n"
        "                    p=0.12, depth_factor=6)\n"
        "rec = simulate(cfg, sample_seed=0,\n"
        "               analyses=['code_dimension', 'code_rate', 'half_cut_entropy'])\n"
        "print('online analyses:', rec.analyses)\n"
        "print('tableau kept?   ', rec.final_tableau is not None)   # online: no"
    ),
    md(r"""You can also pass **your own** callable `fn(sim, spec) -> value` - it is
treated exactly like a built-in, so a custom observable works online and offline
without any special-casing."""),
    code(
        "def heavy_generator_fraction(sim, spec):\n"
        "    '''Fraction of stabilizer generators with weight > n/4.'''\n"
        "    from sparsegf2 import generator_weights\n"
        "    w = generator_weights(sim)\n"
        "    return float((w > sim.n / 4).mean())\n"
        "\n"
        "rec = simulate(cfg, sample_seed=0, analyses=['code_rate', heavy_generator_fraction])\n"
        "print(rec.analyses)"
    ),
    md(r"""## 2. The same analyses, offline, on a saved tableau

Ask `simulate` to keep the final tableau (`save_tableau=True`). Later,
`analyze(...)` runs the same named analyses on that tableau and returns the same
values - the online and offline paths agree by construction."""),
    code(
        "from sparsegf2.circuits.picture import setup_picture\n"
        "\n"
        "rec = simulate(cfg, sample_seed=0, save_tableau=True)\n"
        "symp = rec.final_tableau                      # the (2N, 2N) GF(2) symplectic\n"
        "print('saved tableau shape:', symp.shape)\n"
        "\n"
        "spec = setup_picture('purification', 24)[1]   # the PictureSpec for n_system=24\n"
        "offline = analyze(symp, spec, ['code_dimension', 'code_rate'])\n"
        "online  = simulate(cfg, sample_seed=0,\n"
        "                   analyses=['code_dimension', 'code_rate']).analyses\n"
        "print('offline:', offline)\n"
        "print('online :', online)\n"
        "assert offline == online, 'online and offline must agree'\n"
        "print('online == offline  ✓')"
    ),
    md(r"""## 3. A `Study`: a persisted parameter sweep

`Study.run` expands a grid of config fields × seeds, runs every cell, and writes
a directory of `rows.parquet` (keyed rows) + `tableaux.h5` (the saved tableaux)
+ `manifest.json`. Here we sweep system size `n` and measurement rate `p`,
computing the order parameter (`code_dimension`) online and saving the tableaux
for later."""),
    code(
        "import tempfile, os\n"
        "workdir = tempfile.mkdtemp(prefix='sgf2_study_')\n"
        "path = os.path.join(workdir, 'transition')\n"
        "\n"
        "base = CircuitConfig(graph_spec='cycle', n=16, picture='purification',\n"
        "                     p=0.1, depth_factor=6)\n"
        "study = Study.run(path, base,\n"
        "                  grid={'n': [12, 16, 24], 'p': [0.04, 0.08, 0.12, 0.16, 0.20]},\n"
        "                  seeds=40, analyses=['code_dimension'], save_tableaux=True)\n"
        "print(study)\n"
        "print('columns:', study.analysis_columns())\n"
        "study.to_pandas().head()"
    ),
    md(r"""## 4. Augmenting a study with new observables

Now the payoff. Suppose we later decide we also want the **code rate** ($k/n$)
and the **contiguous distance** (the smallest contiguous region carrying logical
information). We do *not* re-run the simulations - `study.augment([...])`
reconstructs each saved tableau, computes the new observables, and merges the new
columns into `rows.parquet` keyed by cell. It only computes what's missing, so
it's cheap and idempotent."""),
    code(
        "study.augment(['code_rate', 'contiguous_distance'])\n"
        "print('columns now:', study.analysis_columns())\n"
        "study.to_pandas()[['n', 'p', 'sample_seed', 'a.code_dimension',\n"
        "                   'a.code_rate', 'a.contiguous_distance']].head()"
    ),
    code(
        "# re-augmenting is a no-op (the columns already exist) -> idempotent\n"
        "before = study.to_pandas().shape\n"
        "study.augment(['code_rate'])\n"
        "print('shape unchanged on re-augment:', before == study.to_pandas().shape)\n"
        "\n"
        "# reopening from disk shows the augmented columns persisted\n"
        "reopened = Study.open(path)\n"
        "print('persisted columns:', reopened.analysis_columns())"
    ),
    md(r"""## 5. Plotting

`plot_study` auto-detects every numeric observable column and draws one panel per
observable - the order parameter vs `p`, one curve per system size `n`. Because
it auto-detects, the **same call** now includes the columns we just augmented in,
with no edits."""),
    code(
        "fig = plot_study(study)\n"
        "fig.suptitle('Study observables vs measurement rate (one curve per n)', y=1.02)\n"
        "plt.show()"
    ),
    md(r"""The command-line script does the same and is the thing you actually run after a
big sweep finishes:

```sh
python scripts/plot_study.py runs/transition --out transition.png
python scripts/plot_study.py runs/transition --list      # list plottable columns
```"""),
    md(r"""## 6. The purification transition, start to finish

The purification picture's order parameter is the **code dimension**
$k = S(\text{system})$ - the number of system qubits still entangled with the
reference. It starts at $k = n$ (fresh Bell purification) and falls toward $0$ as
measurements purify the system. Plotting $\langle k \rangle / n$ (the code rate)
against $p$ for several $n$ shows the transition: a family of curves that steepen
and cross as $n$ grows."""),
    code(
        "df = study.to_pandas()\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "for n, sub in df.groupby('n'):\n"
        "    g = sub.groupby('p')['a.code_rate'].mean()\n"
        "    ax.plot(g.index, g.to_numpy(), 'o-', label=f'n={n}')\n"
        "ax.set_xlabel('measurement rate p')\n"
        "ax.set_ylabel(r'code rate  $\\langle k \\rangle / n$')\n"
        "ax.set_title('Purification transition')\n"
        "ax.legend(); ax.grid(True, ls=':', alpha=0.5)\n"
        "plt.show()"
    ),
    md(r"""### Tracking the transition layer by layer

For the *dynamical* purification transition you want the order parameter at every
layer, not just at the end. Pass `record_time_series=True` (and optionally
`until_purified=True` to stop early once the system purifies). `rec.time_series`
is then the per-layer code dimension."""),
    code(
        "tcfg = CircuitConfig(graph_spec='cycle', n=24, picture='purification',\n"
        "                     p=0.16, depth_factor=10,\n"
        "                     depth_mode='until_purified', record_time_series=True)\n"
        "trec = simulate(tcfg, sample_seed=1)\n"
        "ts = trec.time_series\n"
        "fig, ax = plt.subplots(figsize=(6, 3.5))\n"
        "ax.plot(range(len(ts)), ts, '-o', ms=3)\n"
        "ax.set_xlabel('layer'); ax.set_ylabel('code dimension k')\n"
        "ax.set_title(f'Purification trajectory (purified at layer {trec.purified_at_layer})')\n"
        "ax.grid(True, ls=':', alpha=0.5)\n"
        "plt.show()"
    ),
    md(r"""## Summary

- **Online** (`simulate(..., analyses=[...])`) computes observables and discards
  the tableau - memory-light, use when the observables are known up front.
- **Offline** (`save_tableau=True` / `Study.run(save_tableaux=True)`) keeps the
  tableaux so you can analyze them later; online and offline agree exactly.
- **`Study`** persists a sweep as keyed rows + tableaux + manifest.
- **`study.augment([...])`** adds new observables to saved tableaux and merges
  them in keyed by cell - no re-simulation, idempotent.
- **`plot_study`** auto-detects every observable column, so augmented data plots
  with no code changes.

Next: build your own study by editing the grid in step 3, or wire a custom
`fn(sim, spec)` analysis (step 1) into `Study.run(analyses=[...])`."""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    ).execute()
    nbformat.write(nb, OUT)
    print(f"wrote + executed {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
