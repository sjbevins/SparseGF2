"""Build + execute ``notebooks/circuits/config.ipynb`` - the validated
per-cell configuration, its derived quantities, and eager compatibility."""

from __future__ import annotations

from _nbtools import build_and_execute, code, md

CELLS = [
    md(r"""# `sparsegf2.circuits.config` - one cell's knobs, validated

[`config.py`](../../src/sparsegf2/circuits/config.py) holds `CircuitConfig`:
everything that, with a `sample_seed`, determines a circuit realization. Its
job is to **fail fast** - a bad cell raises a clear `InvalidArgumentError` at
construction, never a cryptic error deep in the runner. Sweep orchestration
(many `n`, `p`, seeds → parquet) lives *outside* this package, in the future
`sparsegf2.analysis` layer; a `CircuitConfig` is exactly one cell.

## Contents
1. The fields
2. Eager validation & the error hierarchy
3. Derived: `total_layers` (depth modes)
4. Derived: `total_qubits` (picture-aware)
5. Derived: `expected_gate_to_meas_ratio` (mode-aware, with derivation)
6. Eager **graph/matching compatibility** + graph caching
7. `to_dict` serialization"""),
    md(r"""## 1. The fields

```python
@dataclass
class CircuitConfig:
    graph_spec: str | GraphTopology   # "cycle"/"complete", or a prebuilt graph
    n: int                            # system qubits
    picture: Picture | str = Picture.PURE_STATE
    gating_mode = "brickwork"         # or "random_edge"
    matching_mode = "round_robin"     # or "palette" / "fresh"
    measurement_mode = "bernoulli"    # or "gated" / "random_pair"
    p: float = 0.15
    depth_mode = "O(n)"               # or "O(log_n)" / "until_purified"
    depth_factor: int = 8
    n_cliffords: int = 720            # |Sp(4,F2)| - NOT 11520 (that was phase-aware)
    base_seed: int = 42
    record_time_series: bool = False
    warmup_layers: int = 0
    pivot_mode: str | None = None     # forwarded to SparseGF2
    use_numba: bool | None = None     # forwarded to SparseGF2
```

A couple of deliberate choices: `n_cliffords` defaults to **720**, the order
of $\mathrm{Sp}(4,\mathbb{F}_2)$ - not 11,520, which counts
*sign-decorated* Cliffords the phase-free simulator can't see (see the
[`clifford_table`](clifford_table.ipynb) notebook). `graph_spec` accepts a
name *or* a prebuilt `GraphTopology`. `pivot_mode`/`use_numba` pass straight
through to the core."""),
    code(
        "from sparsegf2.circuits import CircuitConfig, Picture\n"
        "c = CircuitConfig(graph_spec='cycle', n=8)\n"
        "print('picture coerced to enum :', c.picture, '(', type(c.picture).__name__, ')')\n"
        "print('n_cliffords default      :', c.n_cliffords)\n"
        "print('gating/matching/meas     :', c.gating_mode, '/', c.matching_mode, '/', c.measurement_mode)\n"
    ),
    md(r"""## 2. Eager validation & the error hierarchy

`__post_init__` checks **every** field and raises
`InvalidArgumentError` - which multi-inherits `ValueError` (so existing
`except ValueError` still works) but lets callers catch *our* input errors
specifically. A battery of bad values, each rejected at construction:"""),
    code(
        "from sparsegf2.errors import InvalidArgumentError\n"
        "bad = [\n"
        "    dict(n=1), dict(graph_spec=123), dict(picture='teleport'),\n"
        "    dict(gating_mode='nope'), dict(matching_mode='nope'),\n"
        "    dict(measurement_mode='nope'), dict(p=1.5), dict(p=-0.1),\n"
        "    dict(depth_mode='nope'), dict(depth_factor=0),\n"
        "    dict(n_cliffords=0), dict(n_cliffords=99999), dict(warmup_layers=-1),\n"
        "]\n"
        "for kw in bad:\n"
        "    base = dict(graph_spec='cycle', n=8); base.update(kw)\n"
        "    try:\n"
        "        CircuitConfig(**base); print('NOT REJECTED:', kw)\n"
        "    except InvalidArgumentError:\n"
        "        pass\n"
        "print('all', len(bad), 'bad configs rejected with InvalidArgumentError (a ValueError subclass)')\n"
        "print('record_time_series on pure_state is also rejected:')\n"
        "try:\n"
        "    CircuitConfig(graph_spec='cycle', n=8, picture='pure_state', record_time_series=True)\n"
        "except InvalidArgumentError as e:\n"
        "    print('  ', e)\n"
    ),
    md(r"""## 3. `total_layers` - depth in **gates per qubit** (gating-aware)

Depth is measured so circuits are comparable **across gating modes**. The
brickwork-equivalent *base* budget scales with $n$ per `depth_mode`:

$$ T_{\text{base}} = \begin{cases}
\text{depth\_factor}\cdot n & \text{O}(n)\ \text{or until\_purified (a cap)}\\[2pt]
\text{depth\_factor}\cdot\lceil\log_2 n\rceil & \text{O}(\log n)
\end{cases} $$

A **brickwork** layer fires $n/2$ gates and touches every qubit once, so
$T_{\text{base}}$ *is* the gates-per-qubit budget and is returned directly.
But **`random_edge`** fires only $m = $ `gates_per_layer` gates/layer,
touching $2m/n$ of the qubits, so it needs $\tfrac{n}{2m}$ times as many
layers to reach the same budget:

$$ T_{\text{random\_edge}} = \operatorname{round}\!\Big(T_{\text{base}}\cdot\frac{n}{2m}\Big). $$

**This is the fix for the single-edge problem**: at $m=1$, an $O(n)$ layer
count applies only $O(n)$ gates total - far too shallow. The formula gives
$O(n^2)$ layers instead. At $m=n/2$ it matches brickwork. So all gating
styles apply the *same total gate count* at equal `depth_factor`:"""),
    code(
        "import math\n"
        "for dm in ('O(n)', 'O(log_n)'):\n"
        "    c = CircuitConfig(graph_spec='cycle', n=32, depth_mode=dm, depth_factor=4)\n"
        "    print(f'{dm:>9} brickwork: total_layers =', c.total_layers())\n"
        "print('check O(log_n):', 4 * math.ceil(math.log2(32)), '= 4*ceil(log2 32) = 4*5')\n"
        "print()\n"
        "n, df = 8, 2\n"
        "cb = CircuitConfig(graph_spec='cycle', n=n, gating_mode='brickwork', depth_factor=df)\n"
        "c1 = CircuitConfig(graph_spec='cycle', n=n, gating_mode='random_edge', gates_per_layer=1, depth_factor=df)\n"
        "cn = CircuitConfig(graph_spec='cycle', n=n, gating_mode='random_edge', gates_per_layer=n//2, depth_factor=df)\n"
        "gb = lambda c, eg: c.total_layers() * eg\n"
        "print(f'brickwork    : {cb.total_layers():>3} layers x {n//2} gates = {gb(cb, n//2)} gates')\n"
        "print(f'random m=1   : {c1.total_layers():>3} layers x 1 gate  = {gb(c1, 1)} gates  (O(n^2) layers)')\n"
        "print(f'random m=n/2 : {cn.total_layers():>3} layers x {n//2} gates = {gb(cn, n//2)} gates  (matches brickwork)')\n"
    ),
    md(r"""### `gates_per_layer` - the "O(n) edges per step" knob

`gates_per_layer` controls how many random edges `random_edge` fires per
layer (default 1, a single edge). It accepts an int **or a callable** that is
resolved against the config, so the count can scale with $n$ -
`lambda cfg: cfg.n // 2` gives the "O(n) random edges per step" model across
a whole size sweep. It only applies to `random_edge` (brickwork always fires
a full matching); a non-default value with brickwork is rejected."""),
    code(
        "c = CircuitConfig(graph_spec='cycle', n=16, gating_mode='random_edge',\n"
        "                  gates_per_layer=lambda cfg: cfg.n // 2)\n"
        "print('callable resolves to:', c.resolved_gates_per_layer(), '= n/2')\n"
        "from sparsegf2.errors import InvalidArgumentError\n"
        "try:\n"
        "    CircuitConfig(graph_spec='cycle', n=8, gating_mode='brickwork', gates_per_layer=5)\n"
        "except InvalidArgumentError as e:\n"
        "    print('brickwork + gates_per_layer rejected:', e)\n"
    ),
    md(r"""## 4. `total_qubits` - picture-aware

`pure_state` uses $n$ physical qubits; `purification` uses $2n$; `single_ref`
uses $n+1$. The runner trusts the `PictureSpec` for the actual layout, but
`total_qubits()` lets you size things up front."""),
    code(
        "for pic in ('pure_state', 'purification', 'single_ref'):\n"
        "    print(f'{pic:>13} total_qubits:', CircuitConfig(graph_spec='cycle', n=8, picture=pic).total_qubits())\n"
    ),
    md(r"""## 5. `expected_gate_to_meas_ratio` - mode-aware, derived

This is computed **before any simulation runs**, purely from the cell's
knobs, and is recorded on every `SampleRecord` as
`gate_to_meas_ratio_expected` (the runner later compares it to the realized
`gate_to_meas_ratio_actual`). Let $e_g$ = expected gates/layer, $e_m$ =
expected measurements/layer:

- $e_g = n/2$ for `brickwork` (a full matching); $e_g = m =$
  `gates_per_layer` for `random_edge`.
- $e_m = n p$ for `bernoulli`; $=2 e_g p$ for `gated` (each gate touches 2
  qubits, each kept w.p. $p$); $=2p$ for `random_pair` (exactly 2 candidates).

Ratio $= e_g / e_m$ (or $\infty$ when $e_m=0$, e.g. $p=0$). For the golden
path (brickwork + bernoulli) this collapses to the clean $\tfrac{n/2}{np} =
\tfrac{1}{2p}$. The ratio is mode-aware rather than a blanket $1/(2p)$,
which would be wrong for `gated`/`random_pair`/`random_edge`."""),
    code(
        "for gm, mm, p in [('brickwork','bernoulli',0.1), ('random_edge','random_pair',0.25),\n"
        "                  ('brickwork','gated',0.2), ('brickwork','bernoulli',0.0)]:\n"
        "    c = CircuitConfig(graph_spec='cycle', n=16, gating_mode=gm, matching_mode='round_robin',\n"
        "                      measurement_mode=mm, p=p)\n"
        "    print(f'{gm:>11} + {mm:<11} p={p}:  expected ratio = {c.expected_gate_to_meas_ratio()}')\n"
        "print('check brickwork+bernoulli p=0.1 -> 1/(2p) =', 1/(2*0.1))\n"
    ),
    md(r"""## 6. Eager graph/matching compatibility + caching

The headline guarantee: `__post_init__` **resolves the graph and validates
that the matching mode is compatible with it** - so requesting `round_robin`
on an odd-$n$ graph (no 1-factorization) fails *at construction*, with a
message listing the available modes, instead of a `RuntimeError` deep in the
scheduler. The resolved graph is cached on `_graph` and reused by
`CircuitBuilder`, so it is built once, not per sample. (For a future
*stochastic* graph family the builder would re-resolve per seed; the comment
flags this.)"""),
    code(
        "# odd n + brickwork is rejected up front, with help text:\n"
        "try:\n"
        "    CircuitConfig(graph_spec='cycle', n=7, gating_mode='brickwork', matching_mode='round_robin')\n"
        "except InvalidArgumentError as e:\n"
        "    print('rejected:', e)\n"
        "# but random_edge needs only an edge, so odd n is fine:\n"
        "ok = CircuitConfig(graph_spec='cycle', n=7, gating_mode='random_edge')\n"
        "print('random_edge on odd n constructs OK; cached graph:', ok._graph.name)\n"
        "# the builder reuses the cached graph object (no rebuild)\n"
        "from sparsegf2.circuits.scheduler import CircuitBuilder\n"
        "print('builder reuses config._graph:', CircuitBuilder(ok, 0).graph is ok._graph)\n"
    ),
    md(r"""## 7. `to_dict` - serialization for manifests

`to_dict()` returns a JSON-friendly dict: the `Picture` enum becomes its
string value, and a prebuilt `GraphTopology` becomes its `name` (the private
`_graph` cache and closures are *not* fields, so they're excluded
automatically)."""),
    code(
        "from sparsegf2.circuits import complete_graph\n"
        "d = CircuitConfig(graph_spec='cycle', n=8, picture='purification').to_dict()\n"
        "print('picture serialized as:', repr(d['picture']), '| graph_spec:', repr(d['graph_spec']))\n"
        "d2 = CircuitConfig(graph_spec=complete_graph(8), n=8).to_dict()\n"
        "print('prebuilt graph serialized as its name:', repr(d2['graph_spec']))\n"
    ),
    md(r"""## Summary

- One validated cell; everything checked eagerly with `InvalidArgumentError`.
- Derived `total_layers` (depth modes), `total_qubits` (picture-aware),
  `expected_gate_to_meas_ratio` (mode-aware, $1/(2p)$ on the golden path).
- The graph is resolved + compatibility-checked at construction and cached.

Next: [`scheduler`](scheduler.ipynb) - turning a config + seed into the
concrete layer stream."""),
]

if __name__ == "__main__":
    build_and_execute("config.ipynb", CELLS)
