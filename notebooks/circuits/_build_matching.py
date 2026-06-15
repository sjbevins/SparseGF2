"""Build + execute ``notebooks/circuits/matching.ipynb`` - per-layer matching
selection (round_robin / palette / fresh) and its RNG semantics."""

from __future__ import annotations

from _nbtools import build_and_execute, code, md

CELLS = [
    md(r"""# `sparsegf2.circuits.matching` - which 1-factor fires this layer

[`matching.py`](../../src/sparsegf2/circuits/matching.py) answers: given a
graph and a layer index $t$, *which* perfect matching (set of gate pairs)
do we apply? Three policies, differing in determinism and in the
distribution they sample from. The [`graphs`](graphs.ipynb) notebook
established what a 1-factorization is; this one is about *choosing among*
the matchings.

## Contents
1. The three modes and their distributions
2. RNG semantics - who draws, and why the signature is uniform
3. `select_matching`, line by line
4. `available_modes` and the validator
5. Demos: cycling, sampling, and reproducibility"""),
    md(r"""## 1. The three modes

| mode | draws from | each layer |
|---|---|---|
| `round_robin` | the fixed 1-factorization | matching $t \bmod \chi'$ - **deterministic** |
| `palette` | the fixed 1-factorization | one of its $\chi'$ matchings, **uniform** |
| `fresh` | *all* perfect matchings of the graph | a **uniform** random perfect matching |

The subtlety is `palette` vs `fresh`. `palette` is restricted to the
*canonical* 1-factorization - for $C_n$ that is just **2** matchings, for
$K_n$ it is $n-1$. `fresh` samples from the (typically far larger) set of
*all* perfect matchings: $K_8$ has $7\cdot5\cdot3\cdot1 = 105$ perfect
matchings, but its 1-factorization names only 7 of them. So `fresh` explores
more of configuration space; `palette` stays on a fixed schedule's palette;
`round_robin` walks that palette in order."""),
    md(r"""## 2. RNG semantics - who draws

A circuit realization is reproducible only if the RNG is consumed in a
**fixed order**. `select_matching` takes an `rng` even when it won't use it,
so the call site's draw sequence is mode-independent up to this point:

- `round_robin` consumes **zero** random numbers (pure function of $t$);
- `palette` consumes **one** integer draw (which palette entry);
- `fresh` consumes whatever its sampler needs (a permutation for $K_n$).

This matters: the scheduler's documented draw order is *placement →
clifford-indices → measurements*. Because `round_robin` draws nothing at the
placement step, swapping `round_robin`↔`palette` shifts every later draw -
which is exactly why the mode is part of the reproducibility contract."""),
    code(
        "import numpy as np\n"
        "from sparsegf2.circuits.graphs import cycle_graph, complete_graph\n"
        "from sparsegf2.circuits.matching import MATCHING_MODES, select_matching, available_modes\n"
        "print('modes:', MATCHING_MODES)\n"
        "\n"
        "# round_robin draws NOTHING: the RNG state is untouched.\n"
        "g = cycle_graph(8)\n"
        "rng = np.random.default_rng(0)\n"
        "before = rng.bit_generator.state\n"
        "select_matching(g, 'round_robin', 3, rng)\n"
        "print('round_robin left RNG untouched:', rng.bit_generator.state == before)\n"
    ),
    md(r"""## 3. `select_matching`, line by line

```python
def select_matching(graph, mode, layer_index, rng):
    if mode == "round_robin":
        chi = len(graph.one_factorization)
        return list(graph.one_factorization[layer_index % chi])
    if mode == "palette":
        chi = len(graph.one_factorization)
        idx = int(rng.integers(0, chi))
        return list(graph.one_factorization[idx])
    if mode == "fresh":
        return list(graph.fresh_matching_sampler(rng))
    raise InvalidArgumentError(...)
```

- `round_robin`: index the 1-factorization by $t \bmod \chi'$ - cycling
  through the palette in order. For $C_n$ ($\chi'=2$) this alternates the two
  matchings every layer (the textbook brickwork pattern).
- `palette`: draw a uniform index into the same $\chi'$ matchings.
- `fresh`: delegate to the graph's `fresh_matching_sampler` closure, which
  draws a uniform perfect matching (for $K_n$, by randomly pairing a random
  permutation).
- The returned `list(...)` is a **defensive copy** - callers may sort or
  mutate `gate_pairs` without corrupting the graph's canonical
  1-factorization. (Removing it saves only ~3% - not worth the
  aliasing risk; safety wins.)
- Each branch first guards graph compatibility and raises `RuntimeError` if
  violated - *defense in depth*; the config validator already rejects
  incompatible `(graph, mode)` pairs eagerly, so these are unreachable in
  normal use. An unknown `mode` raises `InvalidArgumentError`."""),
    code(
        "# round_robin alternates the two C_8 matchings; palette samples them uniformly\n"
        "g = cycle_graph(8)\n"
        "rng = np.random.default_rng(1)\n"
        "rr = [tuple(map(tuple, select_matching(g, 'round_robin', t, rng))) for t in range(4)]\n"
        "print('round_robin t=0..3 alternates:', rr[0] == rr[2] and rr[1] == rr[3] and rr[0] != rr[1])\n"
        "fac = [sorted(m) for m in g.one_factorization]\n"
        "pal = [sorted(select_matching(g, 'palette', 0, rng)) for _ in range(6)]\n"
        "print('palette draws are all 1-factorization members:', all(m in fac for m in pal))\n"
    ),
    md(r"""### `fresh` explores beyond the palette

On $K_8$, `fresh` can produce matchings the canonical 1-factorization never
lists. Sample many and count distinct matchings - it exceeds $\chi'=7$:"""),
    code(
        "g = complete_graph(8)\n"
        "rng = np.random.default_rng(2)\n"
        "seen = set()\n"
        "for _ in range(500):\n"
        "    m = tuple(sorted(map(tuple, select_matching(g, 'fresh', 0, rng))))\n"
        "    verts = sorted(q for e in m for q in e)\n"
        "    assert verts == list(range(8))               # always a perfect matching\n"
        "    seen.add(m)\n"
        "print('distinct fresh matchings seen:', len(seen), '(palette only has chi_prime =', g.chi_prime, ')')\n"
        "print('K_8 has 7!! = 7*5*3*1 =', 7*5*3*1, 'perfect matchings total')\n"
    ),
    md(r"""## 4. `available_modes` and the validator

```python
def available_modes(graph):
    out = []
    if graph.one_factorization is not None:  out += ["round_robin", "palette"]
    if graph.fresh_matching_sampler is not None:  out += ["fresh"]
    return out
```

The config validator calls this to report *which* modes a graph supports
when an incompatible one is requested - turning a deep `RuntimeError` into a
clear up-front `InvalidArgumentError` listing the alternatives. An even-$n$
graph supports all three; an odd-$n$ graph (no perfect matching) supports
none of the brickwork modes:"""),
    code(
        "print('even C_8 :', available_modes(cycle_graph(8)))\n"
        "print('odd  C_7 :', available_modes(cycle_graph(7)))   # [] - no perfect matching\n"
    ),
    md(r"""## 5. Reproducibility

Same graph, same mode, same RNG seed → identical matching stream. This is
the contract the scheduler and the Stim-parity tests both rely on:"""),
    code(
        "g = complete_graph(6)\n"
        "a = np.random.default_rng(7); b = np.random.default_rng(7)\n"
        "sa = [sorted(select_matching(g, 'fresh', t, a)) for t in range(5)]\n"
        "sb = [sorted(select_matching(g, 'fresh', t, b)) for t in range(5)]\n"
        "print('identical fresh streams under equal seeds:', sa == sb)\n"
    ),
    md(r"""## Summary

- `round_robin` (deterministic cycle), `palette` (uniform over the fixed
  $\chi'$ matchings), `fresh` (uniform over **all** perfect matchings).
- RNG draws: 0 / 1 / sampler-dependent - and the mode is part of the
  reproducibility contract because it shifts the draw sequence.
- Defensive copies protect the graph's canonical data; `available_modes`
  powers the validator's helpful errors.

Next: [`config`](config.ipynb) - the validated bag of knobs that ties graph,
matching, measurement, depth, and picture together."""),
]

if __name__ == "__main__":
    build_and_execute("matching.ipynb", CELLS)
