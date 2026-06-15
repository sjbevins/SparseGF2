"""Build + execute ``notebooks/circuits/inspector.ipynb``.

A comprehensive circuit-verification notebook: it **reimplements** the trace
logic inline (so the reader sees the mechanics), checks the inline version
against the library ``trace_circuit`` / ``inspect_circuit``, and then sweeps
**every** construction (pictures × graphs × gating × matching × measurement)
with structural correctness checks so the reader can confirm each circuit is
built correctly.
"""

from __future__ import annotations

from _nbtools import build_and_execute, code, md

CELLS = [
    md(r"""# Inspecting & verifying circuit construction

This notebook lets you **see and verify** exactly what circuit each
`CircuitConfig` builds. It works three ways at once:

1. **Inline reimplementation** - we rebuild the trace logic from scratch in
   plain cells (setup gates, layer gates, measurements), so the mechanics are
   visible, not hidden behind a function call.
2. **Cross-check against the library** - we run the real
   [`trace_circuit`](../../src/sparsegf2/circuits/inspector.py) /
   `inspect_circuit` and assert our inline version produces the *same*
   structured operations. If they ever disagree, a cell fails loudly.
3. **Comprehensive sweep** - we inspect **every** construction (all pictures
   × graphs × gating × matching × measurement modes) and run structural
   checks (gate counts, valid edges, reference qubits never measured, gated
   measurements only on touched qubits, …).

The guiding distinction throughout: **deterministic** construction gates (the
Bell pairs that build purification / single_ref) are shown as the *actual*
gates (`H`, `CX`); **random** two-qubit Cliffords are shown as `C[index]`
(the Sp(4) table index, so you can verify reproducibility); and the **random
selection** of measured qubits is flagged as such."""),
    md("## Setup"),
    code(
        "import numpy as np\n"
        "from sparsegf2 import code_dimension, entanglement_entropy, SparseGF2\n"
        "from sparsegf2.circuits import (\n"
        "    CircuitConfig, CircuitBuilder, trace_circuit, inspect_circuit, setup_ops,\n"
        ")\n"
        "from sparsegf2.circuits.picture import setup_picture\n"
        "from sparsegf2.circuits.ops import apply_named_gate\n"
        "print('ready')\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## Part A - the deterministic setup gates (picture construction)

A *picture* prepares the initial state with a fixed sequence of gates. These
are **deterministic** - the same every run - so the inspector shows the actual
gates. Let's derive them from first principles, inline:

- `pure_state` - nothing; the simulator starts in $|0\dots0\rangle$.
- `purification` - for each system qubit $i$, a Bell pair with reference
  $i+n$: `H(i); CX(i, i+n)`.
- `single_ref` - one Bell pair between the last system qubit $n-1$ and the
  reference $n$: `H(n-1); CX(n-1, n)`."""),
    code(
        "def my_setup_ops(picture, n):\n"
        "    '''Inline reconstruction of each picture's deterministic setup gates.'''\n"
        "    picture = str(picture)\n"
        "    if picture == 'pure_state':\n"
        "        return []\n"
        "    if picture == 'purification':\n"
        "        ops = []\n"
        "        for i in range(n):\n"
        "            ops.append(('H', (i,)))            # put system qubit i in |+>\n"
        "            ops.append(('CX', (i, i + n)))     # entangle it with reference i+n\n"
        "        return ops\n"
        "    if picture == 'single_ref':\n"
        "        return [('H', (n - 1,)), ('CX', (n - 1, n))]   # one Bell pair: (n-1, n)\n"
        "    raise ValueError(picture)\n"
        "\n"
        "# Cross-check: our inline recipe must match the library's setup_ops exactly.\n"
        "for pic in ('pure_state', 'purification', 'single_ref'):\n"
        "    lib = [(o.label, o.qubits) for o in setup_ops(pic, 4)]\n"
        "    mine = my_setup_ops(pic, 4)\n"
        "    assert lib == mine, (pic, lib, mine)\n"
        "    print(f'{pic:>13}: {mine}')\n"
        "print('\\n✓ inline my_setup_ops matches library setup_ops for all pictures')\n"
    ),
    md(r"""### A.2 - the setup gates really build the right state

Showing the gates is one thing; we also verify they *do the right thing*.
Apply our inline setup ops by hand and check the resulting state:

- **purification**: every system qubit entangled with its reference ⇒ code
  dimension $k = S(\text{system}) = n$.
- **single_ref**: the reference is entangled with the system ⇒
  $S(\text{reference}) = 1$, and it is paired specifically with qubit $n-1$
  (measuring $n-1$ should purify it; measuring any other system qubit should
  not)."""),
    code(
        "n = 6\n"
        "# Build purification by hand from the inline recipe, then check k = n.\n"
        "sim = SparseGF2(2 * n)\n"
        "for label, qs in my_setup_ops('purification', n):\n"
        "    apply_named_gate(sim, label, qs)\n"
        "print('purification: code_dimension k =', code_dimension(sim, n), '(expect', n, ')')\n"
        "assert code_dimension(sim, n) == n\n"
        "\n"
        "# single_ref: reference paired with qubit n-1 specifically.\n"
        "sim = SparseGF2(n + 1)\n"
        "for label, qs in my_setup_ops('single_ref', n):\n"
        "    apply_named_gate(sim, label, qs)\n"
        "print('single_ref : S(reference) =', entanglement_entropy(sim, [n]), '(expect 1)')\n"
        "assert entanglement_entropy(sim, [n]) == 1\n"
        "s_after_paired = sim.copy(); s_after_paired.measure_z(n - 1)\n"
        "s_after_other = sim.copy(); s_after_other.measure_z(0)\n"
        "print('  measuring the PAIRED qubit n-1 -> S(ref) =', entanglement_entropy(s_after_paired, [n]), '(purifies)')\n"
        "print('  measuring an OTHER qubit 0     -> S(ref) =', entanglement_entropy(s_after_other, [n]), '(unchanged)')\n"
        "assert entanglement_entropy(s_after_paired, [n]) == 0\n"
        "assert entanglement_entropy(s_after_other, [n]) == 1\n"
        "print('\\n✓ Bell pairs are wired to exactly the right qubits')\n"
    ),
    md("And here is how the library renders that setup (purification, n=4):"),
    code(
        "cfg = CircuitConfig(graph_spec='cycle', n=4, picture='purification', p=0.1, depth_factor=1)\n"
        "tr = trace_circuit(cfg, max_layers=0)\n"
        "setup_stage = tr.stages[0]\n"
        "from sparsegf2.circuits.inspector import _render_stage\n"
        "print('\\n'.join(_render_stage(setup_stage)))\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## Part B - the layer schedule (random Cliffords + measurements)

Each circuit layer comes from the scheduler as a `CircuitLayer` with three
fields: `gate_pairs` (which qubit pairs get a gate), `cliff_indices` (the
**random** Sp(4) table index for each gate), and `meas_qubits` (which qubits
are measured). Let's turn a layer into operations inline, then confirm it
matches the library trace."""),
    code(
        "def my_layer_ops(layer):\n"
        "    '''Inline reconstruction of a layer's operations.'''\n"
        "    ops = []\n"
        "    for g, (qi, qj) in enumerate(layer.gate_pairs):\n"
        "        ci = int(layer.cliff_indices[g])\n"
        "        ops.append(('gate', f'C[{ci}]', (int(qi), int(qj)), True))   # random Sp(4) #ci\n"
        "    for q in layer.meas_qubits:\n"
        "        ops.append(('measure', 'MZ', (int(q),), True))               # random qubit choice\n"
        "    return ops\n"
        "\n"
        "cfg = CircuitConfig(graph_spec='cycle', n=8, p=0.3, depth_factor=2)\n"
        "# Inline: take layer 0 straight from a CircuitBuilder.\n"
        "layer0 = next(CircuitBuilder(cfg, sample_seed=0).layers())\n"
        "mine = my_layer_ops(layer0)\n"
        "# Library: pull layer 0's ops from trace_circuit.\n"
        "tr = trace_circuit(cfg, sample_seed=0, max_layers=1)\n"
        "lib_layer = next(s for s in tr.stages if s.kind == 'layer')\n"
        "lib = [(o.kind, o.label, o.qubits, o.random) for o in lib_layer.ops]\n"
        "for op in mine:\n"
        "    print(op)\n"
        "assert mine == lib\n"
        "print('\\n✓ inline my_layer_ops matches library trace ops exactly')\n"
    ),
    md(r"""### B.2 - random, but reproducible

The Cliffords are *random* (a uniform draw from the 720-element Sp(4) table),
but seeded - so the **same** `(config, sample_seed)` gives the **same**
indices every time, and a different seed gives different ones. The indices in
the trace let you verify this directly."""),
    code(
        "def cliff_indices(cfg, seed, k=4):\n"
        "    tr = trace_circuit(cfg, sample_seed=seed, max_layers=k)\n"
        "    return [o.detail['clifford_index'] for s in tr.stages for o in s.ops if o.kind == 'gate']\n"
        "\n"
        "cfg = CircuitConfig(graph_spec='cycle', n=8, p=0.2, depth_factor=2)\n"
        "print('seed 0 :', cliff_indices(cfg, 0))\n"
        "print('seed 0 :', cliff_indices(cfg, 0), '  (identical - reproducible)')\n"
        "print('seed 1 :', cliff_indices(cfg, 1), '  (different - independent realization)')\n"
        "assert cliff_indices(cfg, 0) == cliff_indices(cfg, 0)\n"
        "assert cliff_indices(cfg, 0) != cliff_indices(cfg, 1)\n"
        "print('\\n✓ same seed reproduces, different seed varies')\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## Part C - a from-scratch renderer (and the library's view)

Putting A and B together: a small inline renderer that prints the setup gates
and the first few layers. Then we show the library's `inspect_circuit` output
for the same config - and assert the underlying operations are identical."""),
    code(
        "def my_render(config, seed=0, max_layers=4):\n"
        "    '''A minimal from-scratch inspector - setup + first N layers.'''\n"
        "    lines = [f'picture={config.picture}  graph={config._graph.name}  '\n"
        "             f'gating={config.gating_mode}  p={config.p}']\n"
        "    sops = my_setup_ops(str(config.picture), config.n)\n"
        "    lines.append('  setup: ' + (', '.join(f'{l}{q}' for l, q in sops) or '(none)'))\n"
        "    for i, layer in zip(range(max_layers), CircuitBuilder(config, seed).layers()):\n"
        "        g = '  '.join(f'C[{int(layer.cliff_indices[j])}]{tuple(layer.gate_pairs[j])}'\n"
        "                      for j in range(len(layer.gate_pairs)))\n"
        "        m = ' '.join(f'q{q}' for q in layer.meas_qubits) or '-'\n"
        "        lines.append(f'  L{i}: gates[ {g} ]  measZ[ {m} ]')\n"
        "    return '\\n'.join(lines)\n"
        "\n"
        "cfg = CircuitConfig(graph_spec='cycle', n=8, picture='single_ref', p=0.16, depth_factor=4)\n"
        "print('################  my_render (inline)  ################')\n"
        "print(my_render(cfg, seed=0, max_layers=4))\n"
        "print('\\n################  library inspect_circuit  ################')\n"
        "print(inspect_circuit(cfg, sample_seed=0, max_layers=4))\n"
    ),
    md("Confirm both views describe the *same* circuit (compare structured ops):"),
    code(
        "def structured(config, seed, k):\n"
        "    out = []\n"
        "    out += [('setup',) + t for t in my_setup_ops(str(config.picture), config.n)]\n"
        "    for i, layer in zip(range(k), CircuitBuilder(config, seed).layers()):\n"
        "        out += [('L%d' % i,) + (op[1], op[2]) for op in my_layer_ops(layer)]\n"
        "    return out\n"
        "\n"
        "def lib_structured(config, seed, k):\n"
        "    tr = trace_circuit(config, sample_seed=seed, max_layers=k)\n"
        "    out = []\n"
        "    for s in tr.stages:\n"
        "        if s.kind == 'setup':\n"
        "            out += [('setup', o.label, o.qubits) for o in s.ops]\n"
        "        elif s.kind == 'layer':\n"
        "            out += [('L%d' % s.index, o.label, o.qubits) for o in s.ops]\n"
        "    return out\n"
        "\n"
        "assert structured(cfg, 0, 4) == lib_structured(cfg, 0, 4)\n"
        "print('✓ inline and library agree on the full operation list')\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## Part D - comprehensive sweep over EVERY construction

Now the thorough part. We inspect every combination of

- **graph**: cycle, complete
- **picture**: pure_state, purification, single_ref
- **gating**: brickwork (× round_robin / palette / fresh) and random_edge
  (× 1 edge / n/2 edges)
- **measurement**: bernoulli, gated, random_pair

and run structural correctness checks on each:

1. setup ops match our inline recipe;
2. every layer has the expected number of gates ($n/2$ brickwork; $m$ for
   random_edge, capped at the edge count);
3. every gate sits on a real graph edge with distinct qubits;
4. **no reference qubit is ever measured** (only system qubits $0..n-1$);
5. `gated` measurements only touch qubits the layer's gates touched;
6. `random_pair` measures at most 2 qubits."""),
    code(
        "def check_construction(config, max_layers=6, seed=0):\n"
        "    '''Return a list of problems (empty list == all checks pass).'''\n"
        "    n = config.n\n"
        "    edges = {tuple(e) for e in config._graph.edges}\n"
        "    tr = trace_circuit(config, sample_seed=seed, max_layers=max_layers)\n"
        "    issues = []\n"
        "    setup = next(s for s in tr.stages if s.kind == 'setup')\n"
        "    if [(o.label, o.qubits) for o in setup.ops] != my_setup_ops(str(config.picture), n):\n"
        "        issues.append('setup mismatch')\n"
        "    if config.gating_mode == 'brickwork':\n"
        "        exp_g = n // 2\n"
        "    else:\n"
        "        exp_g = min(config.resolved_gates_per_layer(), len(edges))\n"
        "    for s in tr.stages:\n"
        "        if s.kind != 'layer':\n"
        "            continue\n"
        "        gates = [o for o in s.ops if o.kind == 'gate']\n"
        "        meas = [o for o in s.ops if o.kind == 'measure']\n"
        "        if len(gates) != exp_g:\n"
        "            issues.append(f'{s.label}: {len(gates)} gates != {exp_g}')\n"
        "        for o in gates:\n"
        "            u, v = o.qubits\n"
        "            if u == v or (min(u, v), max(u, v)) not in edges:\n"
        "                issues.append(f'{s.label}: bad edge {o.qubits}')\n"
        "        for o in meas:\n"
        "            if o.qubits[0] >= n:\n"
        "                issues.append(f'{s.label}: measured REFERENCE qubit {o.qubits[0]}')\n"
        "        if config.measurement_mode == 'gated':\n"
        "            touched = {q for o in gates for q in o.qubits}\n"
        "            if any(o.qubits[0] not in touched for o in meas):\n"
        "                issues.append(f'{s.label}: gated measured an untouched qubit')\n"
        "        if config.measurement_mode == 'random_pair' and len(meas) > 2:\n"
        "            issues.append(f'{s.label}: random_pair measured >2')\n"
        "    return issues\n"
        "print('check_construction defined')\n"
    ),
    code(
        "pictures = ['pure_state', 'purification', 'single_ref']\n"
        "gatings = [('brickwork', {}), ('random_edge', {'gates_per_layer': 1}),\n"
        "           ('random_edge', {'gates_per_layer': 4})]\n"
        "measurements = ['bernoulli', 'gated', 'random_pair']\n"
        "\n"
        "results = {}\n"
        "for graph in ('cycle', 'complete'):\n"
        "    for pic in pictures:\n"
        "        for gate, extra in gatings:\n"
        "            matchings = ['round_robin', 'palette', 'fresh'] if gate == 'brickwork' else ['-']\n"
        "            for mm in matchings:\n"
        "                for meas in measurements:\n"
        "                    kw = dict(graph_spec=graph, n=8, picture=pic, gating_mode=gate,\n"
        "                              measurement_mode=meas, p=0.3, depth_factor=1, **extra)\n"
        "                    if gate == 'brickwork':\n"
        "                        kw['matching_mode'] = mm\n"
        "                    cfg = CircuitConfig(**kw)\n"
        "                    results[(graph, pic, gate, str(extra.get('gates_per_layer', '')), mm, meas)] = \\\n"
        "                        check_construction(cfg)\n"
        "\n"
        "fails = {k: v for k, v in results.items() if v}\n"
        "print(f'Checked {len(results)} constructions (2 graphs x 3 pictures x gating x matching x measurement).')\n"
        "print(f'PASS: {len(results) - len(fails)} / {len(results)}')\n"
        "if fails:\n"
        "    for k, v in fails.items():\n"
        "        print('  FAIL', k, '->', v)\n"
        "else:\n"
        "    print('\\n✓ EVERY construction is structurally correct')\n"
        "assert not fails\n"
    ),
    md("Here is the full pass matrix for the **cycle** graph (the nearest-neighbor model), for eyeballing:"),
    code(
        "print(f\"{'picture':>13} {'gating':>22} {'matching':>11} {'measure':>11}  result\")\n"
        "print('-' * 72)\n"
        "for (graph, pic, gate, m, mm, meas), issues in results.items():\n"
        "    if graph != 'cycle':\n"
        "        continue\n"
        "    g = gate + (f'(m={m})' if gate == 'random_edge' else '')\n"
        "    status = 'PASS' if not issues else 'FAIL'\n"
        "    print(f'{pic:>13} {g:>22} {mm:>11} {meas:>11}  {status}')\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## Part D.2 - full renderings of representative constructions

The checks above are automated; here are full inspector renderings so you can
*read* a representative circuit from each major family and confirm by eye."""),
    code(
        "examples = [\n"
        "    ('pure_state, cycle, brickwork/round_robin, bernoulli',\n"
        "     dict(graph_spec='cycle', n=8, picture='pure_state', p=0.16, depth_factor=2)),\n"
        "    ('purification, cycle, brickwork/round_robin, bernoulli',\n"
        "     dict(graph_spec='cycle', n=6, picture='purification', p=0.16, depth_factor=2)),\n"
        "    ('single_ref, cycle, brickwork/round_robin, bernoulli',\n"
        "     dict(graph_spec='cycle', n=8, picture='single_ref', p=0.16, depth_factor=2)),\n"
        "    ('pure_state, cycle, random_edge (1 edge), bernoulli',\n"
        "     dict(graph_spec='cycle', n=8, gating_mode='random_edge', gates_per_layer=1, p=0.1, depth_factor=2)),\n"
        "    ('pure_state, cycle, random_edge (n/2 edges), bernoulli',\n"
        "     dict(graph_spec='cycle', n=8, gating_mode='random_edge', gates_per_layer=4, p=0.1, depth_factor=2)),\n"
        "    ('pure_state, complete, brickwork/fresh, gated',\n"
        "     dict(graph_spec='complete', n=6, matching_mode='fresh', measurement_mode='gated', p=0.3, depth_factor=2)),\n"
        "    ('pure_state, cycle, brickwork, random_pair',\n"
        "     dict(graph_spec='cycle', n=8, measurement_mode='random_pair', p=0.5, depth_factor=2)),\n"
        "]\n"
        "for title, kw in examples:\n"
        "    print('#' * 78)\n"
        "    print('##', title)\n"
        "    print('#' * 78)\n"
        "    print(inspect_circuit(CircuitConfig(**kw), sample_seed=0, max_layers=3))\n"
        "    print()\n"
    ),
    # ---------------------------------------------------------------
    md(r"""## Part E - targeted correctness guarantees

A few specific invariants worth calling out explicitly, each verified over
many layers / seeds."""),
    code(
        "# (1) Reference qubits are NEVER measured (purification + single_ref).\n"
        "for pic, total in [('purification', 16), ('single_ref', 9)]:\n"
        "    cfg = CircuitConfig(graph_spec='cycle', n=8, picture=pic, p=1.0, depth_factor=2)\n"
        "    tr = trace_circuit(cfg, max_layers=16)\n"
        "    measured = {o.qubits[0] for s in tr.stages for o in s.ops if o.kind == 'measure'}\n"
        "    assert all(q < 8 for q in measured), (pic, measured)\n"
        "    print(f'{pic:>13}: measured qubits {sorted(measured)} -- all < 8 (system only) ✓')\n"
        "\n"
        "# (2) Single-edge random_edge runs O(n^2) layers; n/2 edges matches brickwork.\n"
        "cb = CircuitConfig(graph_spec='cycle', n=8, depth_factor=2)\n"
        "c1 = CircuitConfig(graph_spec='cycle', n=8, gating_mode='random_edge', gates_per_layer=1, depth_factor=2)\n"
        "cn = CircuitConfig(graph_spec='cycle', n=8, gating_mode='random_edge', gates_per_layer=4, depth_factor=2)\n"
        "print(f'\\nbrickwork layers={cb.total_layers()}, single-edge layers={c1.total_layers()} (O(n^2)), '\n"
        "      f'n/2-edge layers={cn.total_layers()}')\n"
        "assert c1.total_layers() == 64 and cn.total_layers() == cb.total_layers() == 16\n"
        "print('✓ depth normalization correct')\n"
        "\n"
        "# (3) gated only measures touched qubits; random_pair measures <= 2.\n"
        "cg = CircuitConfig(graph_spec='cycle', n=8, gating_mode='random_edge', gates_per_layer=3,\n"
        "                   measurement_mode='gated', p=1.0, depth_factor=1)\n"
        "for s in trace_circuit(cg, max_layers=8).stages:\n"
        "    if s.kind == 'layer':\n"
        "        touched = {q for o in s.ops if o.kind == 'gate' for q in o.qubits}\n"
        "        meas = {o.qubits[0] for o in s.ops if o.kind == 'measure'}\n"
        "        assert meas <= touched\n"
        "print('✓ gated measurements are a subset of gate-touched qubits')\n"
        "cp = CircuitConfig(graph_spec='cycle', n=8, measurement_mode='random_pair', p=1.0, depth_factor=2)\n"
        "for s in trace_circuit(cp, max_layers=16).stages:\n"
        "    if s.kind == 'layer':\n"
        "        assert sum(o.kind == 'measure' for o in s.ops) <= 2\n"
        "print('✓ random_pair measures at most 2 qubits per layer')\n"
    ),
    md(r"""## Summary

- The inline reconstructions (`my_setup_ops`, `my_layer_ops`, `my_render`)
  match the library `setup_ops` / `trace_circuit` / `inspect_circuit` exactly
  - so what the inspector shows is what is actually built.
- Setup gates are the real deterministic construction (Bell pairs wired to
  the right qubits, verified by the resulting entropies); layer gates are
  random Sp(4) (reproducible by index); measured qubits are a random,
  system-only selection.
- **Every** construction across graphs × pictures × gating × matching ×
  measurement passes its structural checks, and the targeted invariants
  (reference never measured, depth normalization, gated/random_pair
  semantics) hold.

To inspect any new config yourself:

```python
from sparsegf2.circuits import inspect_circuit, CircuitConfig
print(inspect_circuit(CircuitConfig(graph_spec='cycle', n=8, picture='single_ref', p=0.16), max_layers=15))
```
or from the shell: `python scripts/inspect_circuit.py --n 8 --picture single_ref --layers 15`."""),
]

if __name__ == "__main__":
    build_and_execute("inspector.ipynb", CELLS, timeout=300)
