"""Build + execute ``notebooks/circuits/measurements.ipynb``.

A literate, line-by-line walkthrough of
``src/sparsegf2/circuits/measurements.py`` - the module that decides
*which qubits get measured* after each gate layer. Every branch of
``sample_measurements`` is reproduced, explained, and exercised with a
runnable demo, including the empirical basis for the ``.tolist()``
optimization.

Run from the project root::

    .venv/bin/python notebooks/circuits/_build_measurements.py

The script writes the executed notebook to
``notebooks/circuits/measurements.ipynb`` so the artifact is reproducible
from this source. This is the *template* for the other per-module
circuit notebooks.
"""

from __future__ import annotations

import pathlib

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
OUT = HERE / "measurements.ipynb"


def md(text: str):
    return new_markdown_cell(text)


def code(text: str):
    return new_code_cell(text)


CELLS = [
    md(r"""# `sparsegf2.circuits.measurements` - line by line

This notebook **rewrites and explains** every line of
[`src/sparsegf2/circuits/measurements.py`](../../src/sparsegf2/circuits/measurements.py).

## Where this module sits

A graph-defined MIPT circuit alternates two things each layer:

1. a **gate layer** - two-qubit Cliffords on the edges chosen by the
   scheduler, and
2. a **measurement layer** - single-qubit $Z$ measurements on a random
   subset of the $n$ system qubits.

This module owns **step 2's qubit selection**: given the mode, $n$, the
probability $p$, and the gate pairs that just fired, it returns the sorted
list of qubit indices to measure. It does **not** perform the measurement
or decide its outcome - that happens in
`SparseGF2.measure_z`, driven by a *separate* RNG stream (see the runner
notebook). Keeping *selection* and *outcome* apart is a deliberate design
choice: you can replay the same circuit with different measurement coins
by changing only the outcome stream.

The three modes:

| mode | candidate set each layer | RNG draws |
|---|---|---|
| `bernoulli` | all $n$ qubits, each kept w.p. $p$ | one length-$n$ uniform vector |
| `gated` | only qubits the gate layer touched, each kept w.p. $p$ | one length-(#touched) vector |
| `random_pair` | exactly 2 random qubits, each kept w.p. $p$ | a 2-choice + 2 coins |
"""),
    md("## Setup\n\nImport the module and a seeded RNG so every demo below is reproducible."),
    code(
        "import numpy as np\n"
        "from sparsegf2.circuits.measurements import MEASUREMENT_MODES, sample_measurements\n"
        "from sparsegf2.errors import InvalidArgumentError\n"
        "\n"
        "MEASUREMENT_MODES"
    ),
    md(r"""## The mode registry

```python
MEASUREMENT_MODES: tuple[str, ...] = ("bernoulli", "gated", "random_pair")
```

A module-level tuple is the single source of truth for the valid mode
names. `CircuitConfig` validates against it, and `sample_measurements`
rejects anything outside it. The mode is named **`bernoulli`** rather than
`"uniform"` - "uniform" is overloaded (it also suggests *uniform-count*),
and "bernoulli" names exactly what it is: an independent Bernoulli($p$)
trial per qubit.
"""),
    md(r"""## The function signature

```python
def sample_measurements(mode, n, p, gate_pairs, rng) -> list[int]:
```

- `mode` - one of `MEASUREMENT_MODES`.
- `n` - number of **system** qubits (the candidate universe is `0..n-1`).
- `p` - per-qubit keep probability, in $[0, 1]$.
- `gate_pairs` - the pairs that fired this layer; only `gated` reads them.
- `rng` - a `numpy.random.Generator` passed in by the scheduler, so the
  draw order is controlled centrally (load-bearing for reproducibility).

It returns a **sorted, deduplicated** `list[int]`."""),
    md(r"""## Input validation

```python
if mode not in MEASUREMENT_MODES:
    raise InvalidArgumentError(...)
if not 0.0 <= p <= 1.0:
    raise InvalidArgumentError(...)
```

Both raise `InvalidArgumentError`, which multi-inherits `ValueError` - so
existing `except ValueError` handlers still catch it, but the package's
own exception type lets callers distinguish *our* bad-input errors from a
stray numpy `ValueError`. Let's see both fire:"""),
    code(
        "for bad in [('nope', 8, 0.1), ('bernoulli', 8, 1.5)]:\n"
        "    try:\n"
        "        sample_measurements(bad[0], bad[1], bad[2], [], np.random.default_rng(0))\n"
        "    except InvalidArgumentError as e:\n"
        "        print('rejected:', e)\n"
        "        assert isinstance(e, ValueError)  # backward-compatible\n"
    ),
    md(r"""## Branch 1 - `bernoulli`

```python
if mode == "bernoulli":
    draws = rng.random(n)
    return np.nonzero(draws < p)[0].tolist()
```

Line by line:

- `rng.random(n)` draws `n` i.i.d. uniforms in $[0, 1)$ in **one
  vectorized call** - far cheaper than a Python loop of `n` scalar draws.
- `draws < p` is a boolean mask: `True` exactly where that qubit's coin
  landed below $p$ (probability $p$).
- `np.nonzero(...)[0]` returns the indices of the `True` entries, **already
  in ascending order** - so no explicit `sorted()` is needed.
- `.tolist()` converts the numpy index array to a Python `list[int]` in C.

That last step is an optimization a benchmark confirmed: `.tolist()`
beats the previous `sorted(int(q) for q in ...)` comprehension by 3-6×
on this op. Let's reproduce that head-to-head:"""),
    code(
        "import timeit\n"
        "n = 256\n"
        "rng = np.random.default_rng(0)\n"
        "draws = rng.random(n)\n"
        "p = 0.1\n"
        "old = lambda: sorted(int(q) for q in np.nonzero(draws < p)[0])\n"
        "new = lambda: np.nonzero(draws < p)[0].tolist()\n"
        "assert old() == new()  # identical output\n"
        "t_old = timeit.timeit(old, number=20000)\n"
        "t_new = timeit.timeit(new, number=20000)\n"
        "print(f'old (sorted+int comprehension): {t_old*1e3:.1f} ms / 20k')\n"
        "print(f'new (np .tolist):               {t_new*1e3:.1f} ms / 20k')\n"
        "print(f'speedup: {t_old/t_new:.1f}x')\n"
    ),
    md("**Sanity demos** for `bernoulli` - the boundary cases $p=0$ and $p=1$:"),
    code(
        "rng = np.random.default_rng(1)\n"
        "print('p=0 :', sample_measurements('bernoulli', 8, 0.0, [], rng))   # never measure\n"
        "print('p=1 :', sample_measurements('bernoulli', 8, 1.0, [], rng))   # always all qubits\n"
        "out = sample_measurements('bernoulli', 16, 0.5, [], rng)\n"
        "print('p=.5:', out)\n"
        "assert out == sorted(set(out))  # sorted + unique, by construction\n"
    ),
    md(r"""## Branch 2 - `gated`

```python
if mode == "gated":
    candidates = sorted({int(q) for pair in gate_pairs for q in pair})
    if not candidates:
        return []
    draws = rng.random(len(candidates))
    return sorted(candidates[i] for i in range(len(candidates)) if draws[i] < p)
```

Here the candidate universe is **only the qubits this layer's gates
touched**, not all $n$:

- the set comprehension flattens `gate_pairs` into the distinct qubits
  involved (a `set` dedups, e.g. if two gates shared a qubit - which
  brickwork never does, but `random_edge`-style overlaps could);
- `sorted(...)` fixes a deterministic candidate order so the RNG draws map
  to qubits reproducibly;
- if no gates fired (`candidates` empty), return `[]` immediately - no RNG
  is consumed;
- otherwise draw one uniform per candidate and keep those below $p$.

With $p=1$ every touched qubit is measured:"""),
    code(
        "rng = np.random.default_rng(2)\n"
        "pairs = [(0, 1), (4, 5)]\n"
        "print('gated p=1, pairs', pairs, '->', sample_measurements('gated', 8, 1.0, pairs, rng))\n"
        "print('gated, no gates  ->', sample_measurements('gated', 8, 1.0, [], rng))\n"
    ),
    md(r"""## Branch 3 - `random_pair`

```python
if n < 2:
    return []
pair = rng.choice(n, size=2, replace=False)
draws = rng.random(2)
kept = [int(pair[i]) for i in range(2) if draws[i] < p]
return sorted(set(kept))
```

The candidate set is **exactly two distinct qubits**, chosen uniformly
without replacement, each then passed through the same Bernoulli($p$)
gate:

- `n < 2` guards the degenerate case (can't pick 2 distinct qubits);
- `rng.choice(n, size=2, replace=False)` is the uniform 2-subset draw;
- the two coins decide which of the pair survive;
- `sorted(set(...))` keeps the output sorted + unique (the `set` is belt
  and braces - the pair is already distinct).

At $p=1$ both are always kept, so the result has length 2:"""),
    code(
        "rng = np.random.default_rng(3)\n"
        "for _ in range(4):\n"
        "    print(sample_measurements('random_pair', 8, 1.0, [], rng))\n"
        "print('n<2 ->', sample_measurements('random_pair', 1, 1.0, [], rng))\n"
    ),
    md(r"""## Determinism - the contract the scheduler relies on

`sample_measurements` draws from the `rng` you hand it, in a fixed order.
So the **same generator state** produces the **same selection** - this is
what makes a whole sweep reproducible from `(base_seed, sample_seed,
layer_index)`. Two generators seeded identically agree exactly:"""),
    code(
        "a = np.random.default_rng(12345)\n"
        "b = np.random.default_rng(12345)\n"
        "seq_a = [sample_measurements('bernoulli', 32, 0.3, [], a) for _ in range(5)]\n"
        "seq_b = [sample_measurements('bernoulli', 32, 0.3, [], b) for _ in range(5)]\n"
        "assert seq_a == seq_b\n"
        "print('identical across two equally-seeded RNGs:', seq_a == seq_b)\n"
        "print('first layer selection:', seq_a[0])\n"
    ),
    md(r"""## Mean check - does `bernoulli` actually keep $\approx np$ qubits?

A quick empirical confirmation that the Bernoulli($p$) semantics hold: the
average number of measured qubits per layer should sit near $n p$."""),
    code(
        "rng = np.random.default_rng(7)\n"
        "n, p, trials = 64, 0.2, 2000\n"
        "counts = [len(sample_measurements('bernoulli', n, p, [], rng)) for _ in range(trials)]\n"
        "print(f'empirical mean per layer: {np.mean(counts):.3f}   expected n*p = {n*p}')\n"
        "assert abs(np.mean(counts) - n * p) < 0.5\n"
    ),
    md(r"""## Summary

- `measurements.py` is **pure selection logic** - no simulator coupling,
  no measurement outcomes.
- Three modes, one validated entry point, deterministic given the RNG.
- The only optimization (`bernoulli` → `np.nonzero(...).tolist()`) is
  benchmark-justified and semantically identical to the obvious version.

Next module: the **scheduler**, which calls this function once per layer
with the RNG it owns - see `notebooks/circuits/scheduler.ipynb`."""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    nbformat.write(nb, OUT)
    print(f"wrote + executed {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
