"""Build + execute ``notebooks/core_speedup.ipynb``.

The core-speedup explainer: *why* is SparseGF2 fast for random measurements?
Answers it at the CORE level (sparse [X|Z] representation + the measurement
rank-update), with a dense O(n^2) reference implemented inline for a real
head-to-head - correctness (same stabilizer subspace) and an operation-count
comparison that shows where the asymptotic win comes from.

Run from the project root::

    .venv/bin/python notebooks/_build_core_speedup.py
"""

from __future__ import annotations

import pathlib

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT = HERE / "core_speedup.ipynb"


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


CELLS = [
    md(r"""# Why is SparseGF2 fast for random measurements?

The circuits package is just orchestration; the *speed* lives in the
**core** - [`sparse_tableau.py`](../src/sparsegf2/core/sparse_tableau.py) and
its Numba kernels. This notebook explains, and **measures**, the one idea
that makes monitored (measurement-heavy) Clifford circuits cheap to simulate:
a Z-measurement is a **sparse rank-update**, not a dense $O(n^2)$ sweep.

## Contents
1. What a Z-measurement does to the tableau (Aaronson-Gottesman)
2. The dense cost: $O(n^2)$ per measurement - with a reference implementation
3. The sparse representation: why $Z_q$ only touches the generators with $X$ on $q$
4. Correctness: dense and sparse agree on the stabilizer subspace
5. Sparsity is real: stabilizer weight stays bounded under measurement
6. Operation-count head-to-head: dense $a\cdot 2n$ vs sparse $a\cdot w$
7. Wall-clock: full monitored-circuit runtime vs $n$
8. Why this matters for MIPT

**Sources.** Aaronson & Gottesman, *Improved Simulation of Stabilizer
Circuits*, [PRA 70, 052328 (2004)](https://arxiv.org/abs/quant-ph/0406196);
Fattal et al., [quant-ph/0406168](https://arxiv.org/abs/quant-ph/0406168)."""),
    md(r"""## 1. What a Z-measurement does

A stabilizer state on $n$ qubits is tracked by $2n$ generators - $n$
*destabilizers* (rows $0..n{-}1$) and $n$ *stabilizers* (rows $n..2n{-}1$) -
each a phase-free Pauli stored as its $(x,z)$ bits in a $2n\times 2n$ binary
matrix $[X\,|\,Z]$. Measuring $Z_q$:

- A generator **anticommutes** with $Z_q$ iff its $X$-bit on qubit $q$ is set
  (symplectic product $= x_q$). Call the set of such rows the
  *anticommuters*, of size $a$.
- If no **stabilizer** anticommutes, the outcome is deterministic and the
  stabilizer subspace is unchanged.
- Otherwise pick one anticommuting stabilizer as the **pivot** $p$, XOR it
  into every *other* anticommuter (so they now commute with $Z_q$), demote
  the old pivot to a destabilizer, and install $Z_q$ as the new stabilizer.

The post-measurement **stabilizer subspace** is the unique projection onto
the $Z_q$ eigenspace - independent of the outcome *and* of which pivot you
pick. That invariance is what lets us cross-check two implementations."""),
    code(
        "import numpy as np\n"
        "from sparsegf2 import SparseGF2, average_stabilizer_weight\n"
        "from sparsegf2.core.symplectic import enumerate_sp4\n"
        "from sparsegf2.core.linalg_gf2 import gf2_rref\n"
        "\n"
        "TABLE = enumerate_sp4()          # 720 Sp(4,F2) matrices, native (no Stim)\n"
        "\n"
        "def row_weight(M, r, n):\n"
        "    '''Pauli weight of generator r: #qubits where it is non-identity.'''\n"
        "    return int(np.count_nonzero(M[r, :n] | M[r, n:]))\n"
        "\n"
        "print('table:', TABLE.shape)\n"
    ),
    md(r"""## 2. The dense cost - $O(n^2)$ per measurement

The textbook dense CHP measurement does, per measurement:

- **scan** column $q$ of the $X$ block to find anticommuters - $O(n)$;
- **XOR** the pivot row into each of the $\le 2n$ other anticommuters, each
  XOR spanning the full width $2n$ - $O(n)$ per row, $O(n^2)$ total.

Here it is, operating on the dense $[X|Z]$ matrix. The full-width row XORs
(`M[others, :] ^= M[p, :]`) are the $O(n^2)$ cost."""),
    code(
        "def dense_measure_z(M, q, n):\n"
        "    '''Phase-free dense CHP Z-measurement on a (2n,2n) [X|Z] matrix.'''\n"
        "    anti = np.nonzero(M[:, q])[0]              # x-bit at q -> anticommutes with Z_q\n"
        "    stab_anti = anti[anti >= n]\n"
        "    if stab_anti.size == 0:\n"
        "        return M                               # deterministic: subspace unchanged\n"
        "    p = int(stab_anti[0])                      # pivot = first anticommuting stabilizer\n"
        "    others = anti[anti != p]\n"
        "    M[others, :] ^= M[p, :]                    # <-- full-width XORs: O(a * 2n)\n"
        "    M[p - n, :] = M[p, :]                      # old stabilizer -> destabilizer\n"
        "    M[p, :] = 0\n"
        "    M[p, n + q] = 1                            # new stabilizer = Z_q\n"
        "    return M\n"
        "print('dense reference defined')\n"
    ),
    md(r"""## 3. The sparse representation

SparseGF2 stores the *same* bits four complementary ways (see the
[`sparse_tableau` docstring](../src/sparsegf2/core/sparse_tableau.py)). Two
are the keys to a cheap measurement:

- a **per-generator support list** - the qubits where a generator is
  non-identity, so a row's nonzeros are iterated in $O(\text{weight})$, not
  $O(n)$;
- a **per-qubit inverted index of generators with the $X$-bit set**
  (`inv_x[q]`) - so the anticommuters of $Z_q$ are found in $O(a)$ directly,
  with **no $O(n)$ column scan**.

So the sparse measurement does: look up the $a$ anticommuters in `inv_x[q]`
($O(a)$); pick the **minimum-weight** one as pivot (keeps the tableau
sparse); XOR it into the other $a{-}1$ anticommuters, each XOR touching only
the pivot's $w$ nonzeros. Total $\approx a\cdot w$ - versus dense $a\cdot 2n$.
When generators are sparse ($w \ll n$), that is a large win."""),
    md(r"""## 4. Correctness - dense and sparse agree on the stabilizer subspace

Prepare a state, snapshot its $[X|Z]$, then measure the same qubit with both
implementations. The stabilizer blocks (rows $n..2n{-}1$) must have the same
**row span** - checked by comparing their GF(2) RREFs. (Destabilizers may
differ - different pivots - but the physics is in the stabilizer subspace.)"""),
    code(
        "def prepare(n, p, depth, seed):\n"
        "    '''Nearest-neighbour brickwork + Z-measurements at rate p.'''\n"
        "    gate_rng = np.random.default_rng(seed)\n"
        "    sim = SparseGF2(n, rng=np.random.default_rng(seed + 1))\n"
        "    for t in range(depth):\n"
        "        for i in range(t % 2, n - 1, 2):                  # alternating NN pairs\n"
        "            sim.apply_gate_2q(i, i + 1, TABLE[int(gate_rng.integers(0, 720))])\n"
        "        if p > 0:\n"
        "            for qq in np.nonzero(gate_rng.random(n) < p)[0]:\n"
        "                sim.measure_z(int(qq))\n"
        "    return sim\n"
        "\n"
        "def stab_rref(M, n):\n"
        "    return gf2_rref(np.ascontiguousarray(M[n:]))\n"
        "\n"
        "n = 24\n"
        "sim = prepare(n, p=0.1, depth=3 * n, seed=7)\n"
        "M = sim.to_symplectic().copy()\n"
        "q = n // 2\n"
        "# sparse measure on the sim; dense measure on the snapshot\n"
        "sim.measure_z(q)\n"
        "dense_measure_z(M, q, n)\n"
        "same = np.array_equal(stab_rref(sim.to_symplectic(), n), stab_rref(M, n))\n"
        "print('dense and sparse give the SAME stabilizer subspace:', same)\n"
    ),
    md(r"""## 5. Sparsity is real - weight stays bounded under measurement

The sparse cost $a\cdot w$ only beats $a\cdot 2n$ if the generator weight $w$
actually stays small. Measurements *keep it small*: each projection collapses
a qubit and prunes supports. Compare the **average stabilizer weight** in a
strongly-monitored circuit (area law, $p=0.2$) vs pure scrambling (volume
law, $p=0$), as $n$ grows. Under measurement the weight stays roughly flat;
without it, it grows with $n$."""),
    code(
        "print(f'{\"n\":>5} {\"avg weight p=0.2\":>18} {\"avg weight p=0 (scrambled)\":>28}')\n"
        "for n in (16, 32, 48, 64):\n"
        "    s_sparse = prepare(n, p=0.2, depth=4 * n, seed=1)\n"
        "    s_dense = prepare(n, p=0.0, depth=4 * n, seed=1)\n"
        "    print(f'{n:>5} {average_stabilizer_weight(s_sparse):>18.2f} '\n"
        "          f'{average_stabilizer_weight(s_dense):>28.2f}')\n"
    ),
    md(r"""## 6. Operation-count head-to-head

For one measurement on a prepared (monitored) state, count the real work:

- **dense**: $(a-1)\times 2n$ - the non-pivot anticommuters, each XORed at
  full width.
- **sparse**: $(a-1)\times w_{\text{pivot}}$ - same anticommuters, each XOR
  touching only the (min-weight) pivot's nonzeros.

The ratio $\approx 2n / w_{\text{pivot}}$ grows with $n$ because $w$ stays
bounded under measurement. (SparseGF2 also avoids the $O(n)$ column scan via
`inv_x`, a further constant-factor win not even counted here.)"""),
    code(
        "print(f'{\"n\":>5} {\"a\":>4} {\"w_pivot\":>8} {\"dense=a*2n\":>11} {\"sparse=a*w\":>11} {\"ratio\":>7}')\n"
        "for n in (16, 32, 64, 128):\n"
        "    sim = prepare(n, p=0.2, depth=4 * n, seed=3)\n"
        "    M = sim.to_symplectic()\n"
        "    q = n // 2\n"
        "    anti = np.nonzero(M[:, q])[0]\n"
        "    a = anti.size\n"
        "    if a == 0:\n"
        "        print(f'{n:>5} {0:>4}  (deterministic at this q/seed)')\n"
        "        continue\n"
        "    w_pivot = min(row_weight(M, int(r), n) for r in anti)   # min-weight pivot\n"
        "    dense = (a - 1) * 2 * n\n"
        "    sparse = (a - 1) * w_pivot\n"
        "    ratio = dense / max(sparse, 1)\n"
        "    print(f'{n:>5} {a:>4} {w_pivot:>8} {dense:>11} {sparse:>11} {ratio:>6.1f}x')\n"
    ),
    md(r"""## 7. Wall-clock - full monitored-circuit runtime vs $n$

End to end: time a complete $4n$-deep nearest-neighbour monitored circuit at
$p=0.2$. A dense CHP simulator costs $O(n)$ measurements $\times\,O(n^2)$ each
$= O(n^3)$ over the circuit (plus gates). SparseGF2 stays far below that
because both gates and measurements touch only sparse supports. (First call
includes a one-time Numba compile, excluded by a warmup.)"""),
    code(
        "import time\n"
        "prepare(8, 0.2, 8, 0)          # JIT warmup\n"
        "print(f'{\"n\":>5} {\"time_s (4n-deep, p=0.2)\":>24} {\"us / qubit-layer\":>18}')\n"
        "for n in (16, 32, 64, 128):\n"
        "    depth = 4 * n\n"
        "    t0 = time.perf_counter()\n"
        "    prepare(n, p=0.2, depth=depth, seed=5)\n"
        "    dt = time.perf_counter() - t0\n"
        "    per = dt / (n * depth) * 1e6\n"
        "    print(f'{n:>5} {dt:>24.4f} {per:>18.3f}')\n"
    ),
    md(r"""## 8. Why this matters for MIPT

Measurement-induced phase transition studies live in the **strongly
monitored** regime - exactly where this speedup is largest:

- High measurement rate $\Rightarrow$ **area-law** entanglement
  $\Rightarrow$ low stabilizer weight $\Rightarrow$ $w \ll n$ $\Rightarrow$
  the $2n/w$ advantage is at its biggest.
- The order parameters we record (half-cut entropy, code dimension,
  reference entropy) are all $\operatorname{rank}_{\mathbb{F}_2}$ quantities -
  computed on the same sparse tableau, no exponential state vector anywhere.

So the sparse representation is not a micro-optimization; it is what makes
large-$n$, deep, measurement-heavy circuits - the workhorse of MIPT - actually
tractable.

## Summary
- A Z-measurement is a **rank-update**: find the $a$ anticommuters, XOR a
  pivot into them.
- Dense pays $a\cdot 2n$ per measurement and $O(n)$ to find them; sparse pays
  $a\cdot w$ and finds them in $O(a)$ via `inv_x`.
- Measurements keep $w$ bounded, so the ratio $\sim 2n/w$ grows with $n$ -
  verified above on real tableaux.
- This is the engine under the circuits package; see
  [`notebooks/circuits/overview.ipynb`](circuits/overview.ipynb)."""),
]


def main():
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    NotebookClient(
        nb, timeout=300, kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    ).execute()
    nbformat.write(nb, OUT)
    print(f"wrote + executed {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
