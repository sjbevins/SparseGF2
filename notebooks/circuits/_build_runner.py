"""Build and execute ``notebooks/circuits/runner.ipynb``."""

from __future__ import annotations

from _nbtools import build_and_execute, code, md

CELLS = [
    md(
        r"""# `sparsegf2.circuits.runner` - execute and analyze a realization

`simulate` is the one-call interface; `SimulationRunner` reuses the Clifford
table and lookup tables across many sample seeds. The runner executes each
`CircuitLayer`, computes the picture's final observables, and returns one
`SampleRecord`.

The current runner also supports named/custom online analyses, final-tableau
saving, depth checkpoints, live checkpoint callbacks, and checkpoint-granular
early stopping through `CHECKPOINT_STOP`.
"""
    ),
    code(
        "from sparsegf2.circuits import CircuitConfig, simulate\n\n"
        "cfg = CircuitConfig(\n"
        "    graph_spec='cycle', n=8, picture='purification', p=0.2,\n"
        "    gating_mode='random_pool', total_layers_override=6, base_seed=23,\n"
        ")\n"
        "record = simulate(\n"
        "    cfg, sample_seed=5, analyses=['code_dimension', 'half_cut_entropy']\n"
        ")\n"
        "print('layers / gates / measurements:', record.total_layers, record.total_gates, record.total_measurements)\n"
        "print('code dimension / half cut:', record.code_dimension, record.entropy_half_cut)\n"
        "print('online analyses:', record.analyses)\n"
    ),
    md(
        r"""## Schema-v2 RNG streams

Three streams have distinct SeedSequence entropy vectors:

- construction: `[base_seed, sample_seed]`;
- measurement outcomes: `[base_seed, sample_seed, 0x6D656173]` (`meas`);
- optional global scramble: `[base_seed, sample_seed, 0x73637262]` (`scrb`).

The tags make the outcome and scramble generators independent of circuit
construction and of each other. Distinct `(base_seed, sample_seed)` pairs never
collapse to the same scalar-sum seed. The scramble has its own stream, so
toggling it leaves the complete scheduled layer sequence unchanged.
"""
    ),
    code(
        "import numpy as np\n"
        "from sparsegf2.circuits import CircuitBuilder\n\n"
        "base, sample = 23, 5\n"
        "streams = {\n"
        "    'construction': [base, sample],\n"
        "    'measurement': [base, sample, 0x6D656173],\n"
        "    'scramble': [base, sample, 0x73637262],\n"
        "}\n"
        "for name, entropy in streams.items():\n"
        "    print(name, np.random.default_rng(entropy).integers(2**32, size=3))\n\n"
        "plain = CircuitConfig(\n"
        "    graph_spec='cycle', n=8, p=0.3, base_seed=base,\n"
        "    total_layers_override=3, scramble=False,\n"
        ")\n"
        "scrambled = CircuitConfig(\n"
        "    graph_spec='cycle', n=8, p=0.3, base_seed=base,\n"
        "    total_layers_override=3, scramble=True,\n"
        ")\n"
        "a = CircuitBuilder(plain, sample).schedule()\n"
        "b = CircuitBuilder(scrambled, sample).schedule()\n"
        "same_schedule = all(\n"
        "    x.gate_pairs == y.gate_pairs\n"
        "    and np.array_equal(x.cliff_indices, y.cliff_indices)\n"
        "    and x.meas_qubits == y.meas_qubits\n"
        "    for x, y in zip(a, b, strict=True)\n"
        ")\n"
        "print('scramble toggle preserves schedule:', same_schedule)\n"
    ),
    md(
        r"""## Tableau checkpoints

`checkpoint_layers` uses 1-based measured depth and records the state after
that layer's gates and measurements. With no callback, the record stores full
symplectic tableaux. Out-of-range indices are ignored. A checkpoint at the
executed final layer equals `final_tableau` when both are requested.
"""
    ),
    code(
        "snapshot = simulate(\n"
        "    cfg, sample_seed=7, save_tableau=True, checkpoint_layers=[2, 6, 99]\n"
        ")\n"
        "print('stored checkpoint layers:', sorted(snapshot.checkpoint_tableaux))\n"
        "print(\n"
        "    'final checkpoint equals final tableau:',\n"
        "    np.array_equal(snapshot.checkpoint_tableaux[6], snapshot.final_tableau),\n"
        ")\n"
    ),
    md(
        r"""## Live callbacks and `CHECKPOINT_STOP`

A read-only callback computes an observable on the live tableau without saving
the tableau itself. Its return values populate `checkpoint_values`. Returning
`CHECKPOINT_STOP` stops after the current checkpoint; the sentinel is checked
before storage, so it never appears as a value. Final observables and optional
online analyses are still computed on the actual stopping state.
"""
    ),
    code(
        "from sparsegf2 import code_dimension\n"
        "from sparsegf2.circuits import CHECKPOINT_STOP\n\n"
        "def read_k(sim, spec, layer):\n"
        "    return int(code_dimension(sim, spec.n_system))\n\n"
        "values = simulate(\n"
        "    cfg, sample_seed=8, checkpoint_layers=[1, 3, 6],\n"
        "    checkpoint_callback=read_k,\n"
        ")\n"
        "print('k at checkpoints:', values.checkpoint_values)\n\n"
        "def stop_at_three(sim, spec, layer):\n"
        "    if layer == 3:\n"
        "        return CHECKPOINT_STOP\n"
        "    return int(code_dimension(sim, spec.n_system))\n\n"
        "early = simulate(\n"
        "    cfg, sample_seed=9, checkpoint_layers=[1, 3, 6],\n"
        "    checkpoint_callback=stop_at_three,\n"
        ")\n"
        "print('early-stop layer:', early.total_layers)\n"
        "print('stored values:', early.checkpoint_values)\n"
        "assert early.total_layers == 3 and 3 not in early.checkpoint_values\n"
    ),
    md(
        r"""## Summary

The runner combines schema-v2 construction, independently tagged outcome and
scramble streams, exact final observables, implemented online analyses, and
nonperturbing checkpoint diagnostics. `CHECKPOINT_STOP` provides cheap
checkpoint-granular stopping while preserving a correct final record.
"""
    ),
]

if __name__ == "__main__":
    build_and_execute("runner.ipynb", CELLS)
