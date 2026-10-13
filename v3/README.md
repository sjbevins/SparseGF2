# SparseGF2 v3 (in development)

A ground-up rebuild of SparseGF2: a stabilizer-circuit simulator for monitored
Clifford circuits that is easier to use, faster, and checked against
[Stim](https://github.com/quantumlib/Stim) at every step. It lives beside v2
(`../src/sparsegf2`), which is untouched, under the import name
`sparsegf2_v3`.

* **Why and how:** [`docs/DESIGN.md`](docs/DESIGN.md) (architecture, data layout
  with benchmark evidence, testing strategy, milestones, open decisions).
* **What was wrong with v2:** [`docs/AUDIT.md`](docs/AUDIT.md).
* **History:** [`CHANGELOG.md`](CHANGELOG.md).

## Status: milestone 1 of 8 (foundations)

| milestone | what it adds | status |
|---|---|---|
| M1 | Pauli strings, Clifford tableaux, the gate library | **done** |
| M2 | the simulation kernel and `Simulator` | next |
| M3 | `Circuit` (Stim syntax) and compiled `run()` | |
| M4 | observables (entropies, code dimension, ...) | |
| M5 | performance work and fair benchmarks | |
| M6 | monitored-circuit families (graphs, pictures, purification time) | |
| M7 | QEC circuits (surface codes) | |
| M8 | study runner, migration guide, tutorials | |

## Install and test

From the repository root, in your virtual environment:

```sh
pip install -e "./v3[test]"
cd v3
python -m pytest
```

Expected: `159 passed` in a few seconds (156 tests plus the 3 examples in
docstrings, which pytest also runs). Stim is required for the tests (it is
the reference every result is checked against); if it is missing the suite
fails with an install hint instead of skipping.

## Try it (milestone 1)

```python
import sparsegf2_v3 as sg

# Pauli strings with exact phases, in Stim's notation.
p = sg.PauliString("+XZ_Y")
q = sg.PauliString("-ZZ__")
print(p * q)                      # +iY__Y
print(p.commutes(q))              # False

# Every Clifford gate Stim knows, under Stim's names and conventions.
cx = sg.gate("CNOT")              # aliases and case-insensitive lookup work
print(cx.name, cx.num_qubits)     # CX 2
print(cx.clifford)                # four lines: X0 -> +XX, X1 -> +_X, Z0 -> +Z_, Z1 -> +ZZ
print(cx.clifford(sg.PauliString("+Y_")))   # +YX

# The two-qubit Clifford group, with a stable index for every element.
print(sg.num_cliffords(2))        # 11520
c = sg.Clifford.from_index(2, 4321)
assert c.index() == 4321
print(c.then(c.inverse()) == sg.Clifford.identity(2))   # True

# A uniformly random two-qubit Clifford for each of 5 gates:
import numpy as np
print(sg.random_clifford_indices(2, 5, np.random.default_rng(0)))
```

What the tests check, all against Stim unless noted:

* `tests/test_pauli.py`: parsing, printing, products with phases, and
  commutation, against `stim.PauliString` *and* against explicit matrix
  products (no Stim).
* `tests/test_clifford.py`: the 720-element symplectic group and the
  11,520-element Clifford group match Stim's enumeration exactly; conjugation,
  composition and inverses agree with `stim.Tableau`, for 1 to 5 qubits.
* `tests/test_gates.py`: every one of Stim's 46 one- and two-qubit unitary
  gates has the identical tableau and aliases; conjugation tables agree; 18
  common gates also match their textbook unitary matrices (no Stim).
