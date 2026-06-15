"""Build + execute ``notebooks/circuits/clifford_table.ipynb``.

Deep literate walkthrough of ``_clifford_table.py`` and the symplectic
group $\\mathrm{Sp}(4, \\mathbb{F}_2)$ it exposes: the phase-free reduction
from the Clifford group, the group-order derivation, the nested
symplectic-basis construction, and the no-runtime-Stim design.
"""

from __future__ import annotations

from _nbtools import build_and_execute, code, md

CELLS = [
    md(r"""# `sparsegf2.circuits._clifford_table` - Sp(4, 𝔽₂), the gate set

This notebook explains [`_clifford_table.py`](../../src/sparsegf2/circuits/_clifford_table.py)
**and the mathematics it rests on**. The module is tiny - a cached wrapper -
but it is the seam where "apply a random two-qubit Clifford" becomes "sample
uniformly from a finite group of $4\times4$ matrices over $\mathbb{F}_2$."
Understanding *which* group, and *why*, is the whole point.

## Contents
1. Stabilizer tableaux and the symplectic picture
2. From the Clifford group to the symplectic group (the phase-free reduction)
3. The order $|\mathrm{Sp}(2n,\mathbb{F}_2)|$ - derivation + the nested-basis construction
4. The symplectic form and the symplectic condition
5. The module, line by line (no runtime Stim)
6. How the runner consumes the table

**Primary sources.**
Aaronson & Gottesman, *Improved Simulation of Stabilizer Circuits*,
[Phys. Rev. A 70, 052328 (2004)](https://arxiv.org/abs/quant-ph/0406196);
Koenig & Smolin, [arXiv:1406.2170](https://arxiv.org/abs/1406.2170);
Bravyi & Maslov, [arXiv:2003.09412](https://arxiv.org/abs/2003.09412)."""),
    md(r"""## 1. Stabilizer tableaux and the symplectic picture

A stabilizer state on $n$ qubits is the simultaneous $+1$ eigenstate of $n$
commuting Pauli operators. Track the *whole* stabilizer group by its $2n$
generators (destabilizers + stabilizers in the Aaronson-Gottesman layout).
Each generator is a Pauli string, and modulo phase a Pauli on $n$ qubits is
a pair of bit-vectors $(x, z) \in \mathbb{F}_2^{n} \times \mathbb{F}_2^{n}$:

$$ P = i^{\delta}\, \bigotimes_{q} X_q^{x_q} Z_q^{z_q}, \qquad
   x_q, z_q \in \{0,1\}. $$

So $I\leftrightarrow(0,0)$, $X\leftrightarrow(1,0)$, $Z\leftrightarrow(0,1)$,
$Y\leftrightarrow(1,1)$. SparseGF2 stores **only** $(x,z)$ - it drops the
phase $i^\delta$. That is the *phase-free contract*, and
it is exactly correct for every observable built from the row span of the
tableau: rank, entanglement entropy, code dimension (we prove the entropy
formula in the `picture` notebook).

A Clifford gate $C$ conjugates Paulis to Paulis: $C P C^\dagger$ is again
(a phase times) a Pauli. On the $(x,z)$ bits this conjugation is a **linear
map over $\mathbb{F}_2$** - a $2n\times2n$ matrix $S$ acting by
$(x',z') = (x,z)\,S$. Dropping the phase, *a Clifford is just $S$*."""),
    md(r"""## 2. From Clifford to symplectic - the phase-free reduction

Not every $\mathbb{F}_2$-linear map is a valid Clifford: $C$ must preserve
commutation relations of Paulis. Two Paulis commute iff their *symplectic
inner product* vanishes,

$$ \langle (x,z), (x',z') \rangle_\Omega \;=\; x\cdot z' + z\cdot x' \pmod 2, $$

and Clifford conjugation preserves it. The matrices $S$ that preserve
$\Omega$ are exactly the **symplectic group** $\mathrm{Sp}(2n,\mathbb{F}_2)$.

The full Clifford group (with phases/signs) is larger than
$\mathrm{Sp}(2n,\mathbb{F}_2)$: each symplectic $S$ lifts to several signed
Cliffords. For $n=2$ the Clifford group has $11{,}520$ elements while
$|\mathrm{Sp}(4,\mathbb{F}_2)| = 720$, a factor of $11520/720 = 16$ - the
$4^{n}=16$ sign choices (a $\pm$ on each of the $2n=4$ generators) that the
phase-free simulator cannot see. **Crucially**, projecting the uniform
distribution on the $11{,}520$ Cliffords down to $\mathrm{Sp}(4,\mathbb{F}_2)$
gives the *uniform* distribution on the $720$ symplectics (each is hit by
exactly 16). So for SparseGF2, "uniform random two-qubit Clifford" $=$
"uniform random element of $\mathrm{Sp}(4,\mathbb{F}_2)$." That equivalence
is why this module samples from a 720-element table, not 11,520."""),
    code(
        "import numpy as np\n"
        "from sparsegf2 import enumerate_symplectic_group, symplectic_group_order, is_symplectic\n"
        "from sparsegf2.circuits._clifford_table import SP4_SIZE, sp4_table\n"
        "\n"
        "print('|Sp(2,F2)| =', symplectic_group_order(1))   # 1 qubit\n"
        "print('|Sp(4,F2)| =', symplectic_group_order(2))   # 2 qubits  -> 720\n"
        "print('|Sp(6,F2)| =', symplectic_group_order(3))   # 3 qubits\n"
        "print('11520 / 720 =', 11520 // 720, '(the 4^2 = 16 sign choices)')\n"
        "print('SP4_SIZE   =', SP4_SIZE)\n"
    ),
    md(r"""## 3. The order of $\mathrm{Sp}(2n,\mathbb{F}_2)$ - two ways

**Closed form.**
$$ |\mathrm{Sp}(2n,\mathbb{F}_2)| \;=\; 2^{\,n^2}\prod_{i=1}^{n}\bigl(2^{2i}-1\bigr). $$
For $n=1$: $2^{1}(2^2-1)=2\cdot3=6$. For $n=2$:
$2^{4}(2^2-1)(2^4-1)=16\cdot3\cdot15=720$. ✓

**Constructive count (the nested symplectic-basis construction).** This is
the algorithm `enumerate_symplectic_group` actually runs.
Build a symplectic basis $\{X_1',Z_1',\dots,X_n',Z_n'\}$ greedily:

- **Pick $X_i'$**: any nonzero vector in the symplectic complement
  $V_{i-1}^{\perp}$ of the pairs already chosen - $2^{2(n-i+1)}-1$ choices.
- **Pick $Z_i'$**: any vector in $V_{i-1}^{\perp}$ with
  $\langle X_i', Z_i'\rangle_\Omega = 1$ - $2^{2(n-i+1)-1}$ choices.

The product telescopes to the closed form. For $n=2$ it is
$(2^4-1)\cdot 2^3 \cdot (2^2-1)\cdot 2^1 = 15\cdot8\cdot3\cdot2 = 720$. Let's
confirm both the count and that enumeration yields exactly that many
*distinct* matrices:"""),
    code(
        "def sp_order_closed(n):\n"
        "    out = 2 ** (n * n)\n"
        "    for i in range(1, n + 1):\n"
        "        out *= (2 ** (2 * i) - 1)\n"
        "    return out\n"
        "\n"
        "def sp_order_constructive(n):\n"
        "    out = 1\n"
        "    for i in range(1, n + 1):\n"
        "        k = 2 * (n - i + 1)\n"
        "        out *= (2 ** k - 1) * (2 ** (k - 1))  # choices for X_i', then Z_i'\n"
        "    return out\n"
        "\n"
        "for n in (1, 2, 3):\n"
        "    print(f'n={n}: closed={sp_order_closed(n)}, constructive={sp_order_constructive(n)}, '\n"
        "          f'lib={symplectic_group_order(n)}')\n"
        "\n"
        "tab = enumerate_symplectic_group(2)\n"
        "print('enumerated shape:', tab.shape, '| distinct:', len({m.tobytes() for m in tab}))\n"
    ),
    md(r"""## 4. The symplectic form and the symplectic condition

In the $(x_1,\dots,x_n,z_1,\dots,z_n)$ basis the form is the block matrix
$$ \Omega = \begin{pmatrix} 0 & I_n \\ I_n & 0 \end{pmatrix} \pmod 2 $$
(note: over $\mathbb{F}_2$, $-1=+1$, so the usual $\begin{psmallmatrix}0&I\\-I&0\end{psmallmatrix}$
is this). A matrix $S$ is symplectic iff it preserves $\Omega$:
$$ S\,\Omega\,S^{\top} \equiv \Omega \pmod 2. $$
`apply_gate_2q` checks this on every gate (a $\sim1\,\mu$s safety net against
silently corrupting the tableau into a non-Clifford map). Here it is on a
few table entries - and a non-symplectic matrix correctly fails:"""),
    code(
        "from sparsegf2 import symplectic_form\n"
        "\n"
        "Omega = symplectic_form(2)\n"
        "print('Omega =\\n', Omega)\n"
        "t = sp4_table()\n"
        "S = t[123].astype(int)\n"
        "lhs = (S @ Omega @ S.T) % 2\n"
        "print('S Omega S^T == Omega :', np.array_equal(lhs, Omega), '| is_symplectic:', is_symplectic(t[123]))\n"
        "bad = np.eye(4, dtype=np.uint8); bad[0, 1] = 1  # shear: not symplectic\n"
        "print('tampered matrix is_symplectic:', is_symplectic(bad))\n"
    ),
    md(r"""### The basis convention `apply_gate_2q` expects

The core applies a $4\times4$ symplectic on the **4-vector
$(x_{q_i}, x_{q_j}, z_{q_i}, z_{q_j})$** of every generator supported on
$q_i$ or $q_j$, via the row-vector rule $(x',z')=(x,z)\,S$. The table is
produced in exactly this basis, so a table entry drops straight into
`sim.apply_gate_2q(qi, qj, table[k])` with no relabeling - which is what the
runner does."""),
    md(r"""## 5. The module, line by line

```python
SP4_SIZE: int = 720

def sp4_table() -> np.ndarray:
    return enumerate_sp4()
```

That is the whole module. The design decisions in it:

- **It computes nothing itself.** `enumerate_sp4()` lives in the *core*
  (`sparsegf2.core.symplectic`) and runs the nested-basis construction from
  §3, then caches the result with double-checked locking. This wrapper just
  gives the circuits package a stable, intention-revealing name and a single
  import site. If we ever add an integer-indexed Koenig-Smolin enumeration
  or an $n$-qubit table, it lands behind this same `sp4_table()`.
- **No Stim.** A natural way to build this table would be
  `stim.Tableau.iter_all(2)` - a hard runtime dependency. We avoid that: the
  table is built natively, and we verify no `stim` is imported.
- **Read-only, cached.** The returned array is the core's cached view; the
  first call materializes it (~milliseconds), later calls are a lock-free
  return of the *same* object. Treat it as immutable."""),
    code(
        "import sys\n"
        "t1 = sp4_table()\n"
        "t2 = sp4_table()\n"
        "print('shape/dtype     :', t1.shape, t1.dtype)\n"
        "print('cached identity :', t1 is t2)           # same object, no recompute\n"
        "print('read-only       :', not t1.flags.writeable)\n"
        "print('all symplectic  :', all(is_symplectic(t1[k]) for k in range(len(t1))))\n"
        "print('stim imported?  :', 'stim' in sys.modules)   # never at runtime\n"
    ),
    md(r"""## 6. How the runner consumes the table

The scheduler draws an integer index per gate from
`rng.integers(0, n_cliffords)` (default `n_cliffords = 720 = |\mathrm{Sp}(4)|`),
and the runner applies `table[index]` to the gate's qubit pair. Sampling a
uniform index over the full table is, by §2, a uniform random two-qubit
Clifford as far as any phase-free observable can tell. Here is the entire
data path in miniature:"""),
    code(
        "from sparsegf2 import SparseGF2\n"
        "rng = np.random.default_rng(0)\n"
        "table = sp4_table()\n"
        "sim = SparseGF2(4)\n"
        "for _ in range(8):                 # 8 random 2q Cliffords on a small register\n"
        "    qi, qj = rng.choice(4, size=2, replace=False)\n"
        "    k = int(rng.integers(0, 720))\n"
        "    sim.apply_gate_2q(int(qi), int(qj), table[k])\n"
        "print('applied 8 uniform Sp(4,F2) gates; tableau still valid, n =', sim.n)\n"
    ),
    md(r"""## Summary

- A phase-free Clifford **is** an element of $\mathrm{Sp}(2n,\mathbb{F}_2)$.
- For two qubits that group has $720$ elements; uniform sampling over them
  equals uniform random two-qubit Clifford for every observable we compute.
- The table is built natively (no Stim), cached, read-only.
- `sp4_table()` is the single, stable handle the rest of the package uses.

Next: [`graphs`](graphs.ipynb) - who is allowed to interact with whom, and
the 1-factorization theory behind brickwork layers."""),
]

if __name__ == "__main__":
    build_and_execute("clifford_table.ipynb", CELLS)
