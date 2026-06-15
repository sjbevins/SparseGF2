"""Build + execute ``notebooks/circuits/picture.ipynb`` - stabilizer states,
the FCYBC entropy formula, Bell purification, and the code-dimension order
parameter."""

from __future__ import annotations

from _nbtools import build_and_execute, code, md

CELLS = [
    md(r"""# `sparsegf2.circuits.picture` - initial state & order parameter

[`picture.py`](../../src/sparsegf2/circuits/picture.py) sets up *what state
the circuit starts in* and *what scalar we read off it*. Two pictures:
`pure_state` and `purification`. To understand them we need the
**entanglement-entropy formula for stabilizer states** - which is also the
mathematical heart of why this whole simulator is useful for MIPT.

## Contents
1. Stabilizer states and the $[X\,|\,Z]$ tableau
2. The FCYBC entropy formula $S(A)=\operatorname{rank}_{\mathbb{F}_2}[X|Z]\big|_A-|A|$
3. Why phase-free is exact for entropy
4. `pure_state` - $|0^n\rangle$
5. `purification` - Bell pairs, the code dimension $k=S(\text{system})$
6. `setup_picture` and `PictureSpec`, line by line

**Sources.** Fattal, Cubitt, Yamamoto, Bravyi, Chuang,
[quant-ph/0406168](https://arxiv.org/abs/quant-ph/0406168) (the entropy
formula); Gullans & Huse,
[PRX 10, 041020 (2020)](https://arxiv.org/abs/1905.05195) (purification MIPT);
Aaronson & Gottesman, [quant-ph/0406196](https://arxiv.org/abs/quant-ph/0406196)."""),
    md(r"""## 1. Stabilizer states and the $[X\,|\,Z]$ tableau

An $n$-qubit stabilizer state is fixed by an abelian group $\mathcal{S}$ of
$2^n$ Pauli operators (not containing $-I$); $n$ independent generators
specify it. SparseGF2 stores $2n$ generators in the Aaronson-Gottesman
layout - rows $0..n{-}1$ are *destabilizers*, rows $n..2n{-}1$ are
*stabilizers* - each as its $(x,z)$ bits. Stacked, that is a $2n\times 2n$
binary matrix $[X\,|\,Z]$. `to_symplectic()` returns it; for $|0^n\rangle$
the stabilizers are $Z_1,\dots,Z_n$ and the destabilizers $X_1,\dots,X_n$,
so $[X|Z]$ is a permuted identity:"""),
    code(
        "import numpy as np\n"
        "from sparsegf2 import SparseGF2, entanglement_entropy, subsystem_rank\n"
        "sim = SparseGF2(3)                 # |000>\n"
        "M = sim.to_symplectic()\n"
        "print('shape:', M.shape)           # (2n, 2n) = (6, 6)\n"
        "print(M)\n"
    ),
    md(r"""## 2. The FCYBC entanglement-entropy formula

For a stabilizer state and a subsystem $A$, Fattal-Cubitt-Yamamoto-Bravyi-
Chuang (2004) give the entanglement entropy (in units of $\log_2$, i.e.
*ebits*) as a pure **rank** computation:

$$ \boxed{\,S(A) \;=\; \operatorname{rank}_{\mathbb{F}_2}\!\bigl([X\,|\,Z]\big|_A\bigr)\;-\;|A|\,} $$

where $[X|Z]\big|_A$ keeps only the columns ($X$ and $Z$ bits) belonging to
qubits in $A$, across all $n$ stabilizer-generator rows. The intuition: the
restricted generators span a subspace; the *local* part of the stabilizer
group has rank $|A| - S(A)$, and the deficit from the full $|A|$ counts the
ebits crossing the cut. This is why a **stabilizer** simulator computes
entanglement in polynomial time - it is linear algebra over $\mathbb{F}_2$,
not an exponential Schmidt decomposition. SparseGF2 exposes it as
`entanglement_entropy(sim, A)`; `subsystem_rank` is the rank term. For a
product state like $|0^n\rangle$ every cut has $S=0$:"""),
    code(
        "sim = SparseGF2(8)   # |0^8>, a product state\n"
        "for A in ([0], [0,1,2], range(4)):\n"
        "    A = list(A)\n"
        "    r = subsystem_rank(sim, A)\n"
        "    S = entanglement_entropy(sim, A)\n"
        "    print(f'A={A!s:<12} rank={r:<2} |A|={len(A):<2} S = rank-|A| = {S}')\n"
    ),
    md(r"""Now make a genuinely entangled state - a Bell pair $\tfrac{1}{\sqrt2}
(|00\rangle+|11\rangle)$ via $H_0$ then $\mathrm{CX}_{0\to1}$ - and watch the
formula report exactly **1 ebit** across the cut between the two qubits:"""),
    code(
        "bell = SparseGF2(2)\n"
        "bell.apply_h(0)\n"
        "bell.apply_cx(0, 1)\n"
        "print('S({0}) =', entanglement_entropy(bell, [0]), 'ebit  (maximally entangled pair)')\n"
        "print('[X|Z] of the Bell state:\\n', bell.to_symplectic())\n"
    ),
    md(r"""## 3. Why phase-free is exact for entropy

The formula uses only the **row span** of $[X|Z]$ over $\mathbb{F}_2$ - ranks
are invariant to the Pauli *signs* $i^\delta$ that SparseGF2 drops. So every
quantity built from $\operatorname{rank}_{\mathbb{F}_2}$ - entanglement
entropy, mutual information, code dimension, the weight spectrum - is
computed **exactly** despite the phase-free representation. What is *not*
available is a signed expectation like
$\langle\psi|Z_q|\psi\rangle$ (that needs the sign). The circuits package
only ever asks for rank-based observables, so phase-free is lossless here."""),
    md(r"""## 4. `pure_state` - $|0^n\rangle$

The simplest picture: `setup_picture("pure_state", n)` returns a plain
`SparseGF2(n)` and a `PictureSpec` with no reference subsystem and
`order_parameter="none"`. The runner still records the **half-cut entropy**
$S(\{0,\dots,n/2-1\})$ - the standard MIPT order parameter for a pure
monitored circuit (volume-law vs. area-law as $p$ crosses the threshold)."""),
    code(
        "from sparsegf2.circuits.picture import Picture, PictureSpec, setup_picture\n"
        "sim, spec = setup_picture(Picture.PURE_STATE, 8)\n"
        "print('picture         :', spec.picture)\n"
        "print('total_qubits    :', spec.total_qubits)         # n\n"
        "print('system_qubits   :', spec.system_qubits.tolist())\n"
        "print('reference_qubits:', spec.reference_qubits.tolist())   # empty\n"
        "print('order_parameter :', spec.order_parameter)      # 'none'\n"
        "print('half-cut S(|0^8>) =', entanglement_entropy(sim, range(4)))\n"
    ),
    md(r"""## 5. `purification` - Bell pairs and the code dimension

The Gullans-Huse purification picture entangles each **system** qubit $i$
with a **reference** qubit $i+n$ via a Bell pair, giving a $2n$-qubit state.
`from_bell_purification(n)` builds it (apply $H_i;\mathrm{CX}_{i\to i+n}$ to
$|0^{2n}\rangle$). The order parameter is the **code dimension**

$$ k \;\equiv\; S(\text{system}) \;=\; \operatorname{rank}_{\mathbb{F}_2}[X|Z]\big|_{\text{sys}} - n, $$

the number of system qubits still entangled with the reference - i.e. the
logical qubits the monitored dynamics has *not yet purified*. Fresh out of
the box every system qubit is Bell-paired with its reference, so $k=n$;
measurements drive $k\to0$, and the **time** at which $k$ collapses is the
MIPT purification order parameter. Watch $k=n$ initially, then drop under
measurements:"""),
    code(
        "from sparsegf2 import from_bell_purification, code_dimension\n"
        "sim, spec = setup_picture('purification', 8)\n"
        "print('total_qubits    :', spec.total_qubits)             # 2n = 16\n"
        "print('system_qubits   :', spec.system_qubits.tolist())\n"
        "print('reference_qubits:', spec.reference_qubits.tolist())\n"
        "print('order_parameter :', spec.order_parameter)          # 'code_dimension'\n"
        "print('fresh code dimension k = S(system) =', code_dimension(sim, 8), '(= n, fully entangled)')\n"
        "\n"
        "# Z-measuring a freshly Bell-paired system qubit collapses THAT pair,\n"
        "# dropping k by 1. Measure two qubits per step and watch k: 8->6->4->2->0.\n"
        "measured = 0\n"
        "for batch in range(4):\n"
        "    for q in (2 * batch, 2 * batch + 1):\n"
        "        sim.measure_z(q)\n"
        "        measured += 1\n"
        "    print(f'  after measuring {measured} system qubits: k =', code_dimension(sim, 8))\n"
    ),
    md(r"""## 5b. `single_ref` - one tracked reference qubit

The cheapest MIPT probe: instead of $n$ reference qubits (purification),
use **one**. `setup_picture("single_ref", n)` builds $n+1$ qubits and
Bell-pairs the reference (index $n$) with system qubit $n-1$ via
`H(n-1); CX(n-1, n)` (a deliberate choice to keep the reference adjacent to
the system boundary). The order parameter is `ref_entropy` $= S(\text{reference}) \in
\{0,1\}$: **1** while the reference stays entangled with the system, **0**
once monitored dynamics purifies it. A single bit, much cheaper than the full
code dimension, and the standard single-reference MIPT diagnostic."""),
    code(
        "sim, spec = setup_picture('single_ref', 8)\n"
        "print('total_qubits    :', spec.total_qubits)          # n + 1 = 9\n"
        "print('reference_qubits:', spec.reference_qubits.tolist())   # [8]\n"
        "print('order_parameter :', spec.order_parameter)       # 'ref_entropy'\n"
        "print('fresh S(reference) =', entanglement_entropy(sim, spec.reference_qubits),\n"
        "      '(1: entangled with the system via the Bell pair)')\n"
        "# Measuring the system qubit the reference is paired with collapses it.\n"
        "sim.measure_z(7)\n"
        "print('after measuring qubit 7: S(reference) =',\n"
        "      entanglement_entropy(sim, spec.reference_qubits), '(purified to 0)')\n"
    ),
    md(r"""## 6. `setup_picture` and `PictureSpec`, line by line

```python
@dataclass(frozen=True)
class PictureSpec:
    picture: Picture
    n_system: int
    total_qubits: int
    system_qubits: NDArray[np.int64]
    reference_qubits: NDArray[np.int64]
    order_parameter: Literal["none", "code_dimension"]
```

`PictureSpec` is the **single source of truth** for the qubit layout, so the
runner never re-derives "which qubits are system vs reference." It is frozen
(immutable) - a description, not a mutable state.

```python
def setup_picture(picture, n_system, *, rng=None, pivot_mode=None, use_numba=None):
    picture = Picture(picture)          # coerce + validate (-> InvalidArgumentError)
    if n_system < 1: raise InvalidArgumentError(...)
    kwargs = {"rng": rng, "pivot_mode": pivot_mode, "use_numba": use_numba}
    if picture is Picture.PURE_STATE:    sim = SparseGF2(n_system, **kwargs);            order="none"
    if picture is Picture.PURIFICATION:  sim = from_bell_purification(n_system, **kwargs); order="code_dimension"
```

- `Picture(picture)` accepts the enum or the bare string (it's a `StrEnum`);
  an unknown name becomes a clean `InvalidArgumentError`.
- `rng` becomes the simulator's **measurement-outcome** stream - the runner
  seeds it independently of the circuit-construction RNG.
- `pivot_mode` / `use_numba` are forwarded straight to the core, so the
  simulator's knobs are reachable without re-wrapping.
- The runner reads `spec.order_parameter` to decide whether to also record
  $k$ - `"none"` records only half-cut entropy; `"code_dimension"` records
  $k$ **in addition**. `single_ref` is a documented extension point (one
  reference qubit), deliberately not stubbed."""),
    code(
        "# rng is forwarded -> reproducible measurement outcomes\n"
        "a,_ = setup_picture(Picture.PURE_STATE, 6, rng=np.random.default_rng(99))\n"
        "b,_ = setup_picture(Picture.PURE_STATE, 6, rng=np.random.default_rng(99))\n"
        "for s in (a, b): s.apply_h(0)\n"
        "print('same-seed outcomes match:', [a.measure_z(0) for _ in range(5)] == [b.measure_z(0) for _ in range(5)])\n"
        "# pivot_mode / use_numba are forwarded to the core\n"
        "s,_ = setup_picture(Picture.PURE_STATE, 4, pivot_mode='first', use_numba=False)\n"
        "print('forwarded knobs:', s.pivot_mode, s.use_numba)\n"
    ),
    md(r"""## Summary

- Entanglement of a stabilizer state is **$\operatorname{rank}_{\mathbb{F}_2}$
  minus size** (FCYBC) - polynomial, exact, and phase-independent.
- `pure_state` reports half-cut $S$; `purification` reports the code
  dimension $k=S(\text{system})$, the Gullans-Huse purification parameter.
- `PictureSpec` fixes the layout once; `setup_picture` forwards the core's
  RNG/pivot/numba knobs.

Next: [`config`](config.ipynb) - the validated knob-bag, including the eager
graph/matching compatibility check."""),
]

if __name__ == "__main__":
    build_and_execute("picture.ipynb", CELLS)
