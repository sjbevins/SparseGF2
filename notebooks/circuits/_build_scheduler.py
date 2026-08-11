"""Build and execute ``notebooks/circuits/scheduler.ipynb``."""

from __future__ import annotations

from _nbtools import build_and_execute, code, md

CELLS = [
    md(
        r"""# `sparsegf2.circuits.scheduler` - config plus seed to layers

`CircuitBuilder` is simulator-free. It converts a validated `CircuitConfig`
and one `sample_seed` into a deterministic stream of `CircuitLayer` records.
Each layer contains gate pairs, one symplectic Clifford-table index per gate,
the measurement candidates, and the subset that actually fires.

Schema v2 seeds the construction generator with the pair
`[base_seed, sample_seed]`. The old scalar sum aliased distinct cells such as
`(10, 2)` and `(11, 1)`; pair seeding keeps those streams independent.
"""
    ),
    code(
        "from sparsegf2.circuits import CircuitBuilder, CircuitConfig, from_spec\n\n"
        "cfg = CircuitConfig(\n"
        "    graph_spec='cycle', n=8, p=0.25, total_layers_override=2,\n"
        "    base_seed=10,\n"
        ")\n"
        "layer = CircuitBuilder(cfg, sample_seed=2).schedule()[0]\n"
        "print('gate pairs       :', layer.gate_pairs)\n"
        "print('Clifford indices :', layer.cliff_indices)\n"
        "print('meas candidates  :', layer.meas_candidates)\n"
        "print('meas qubits      :', layer.meas_qubits)\n"
    ),
    md(
        r"""## Four gate-placement modes

The fixed per-layer draw order is load-bearing for reproducibility:

1. place gates (`all_edges` and round-robin brickwork consume no placement
   randomness);
2. draw one Clifford index per gate;
3. select measurement candidates where needed, then draw their Bernoulli
   firing coins.

`random_edge` draws distinct edge indices. `random_pool` draws with replacement,
so repeated edges are allowed and the requested count may exceed `|E|`.
`all_edges` returns the complete stored edge list in its canonical order.
"""
    ),
    code(
        "modes = {\n"
        "    'brickwork': {},\n"
        "    'random_edge': {'gates_per_layer': 2},\n"
        "    'random_pool': {'gates_per_layer': 6},\n"
        "    'all_edges': {},\n"
        "}\n"
        "for mode, extra in modes.items():\n"
        "    local = CircuitConfig(\n"
        "        graph_spec='cycle', n=8, gating_mode=mode, p=0.0,\n"
        "        total_layers_override=1, **extra,\n"
        "    )\n"
        "    first = CircuitBuilder(local, 4).schedule()[0]\n"
        "    print(f'{mode:>11}: {first.n_gates} gates  {first.gate_pairs}')\n"
        "    if mode == 'all_edges':\n"
        "        assert first.gate_pairs == from_spec('cycle', 8).edges\n"
    ),
    md(
        r"""## Four measurement modes

`bernoulli` makes every system qubit eligible. `gated` uses the distinct gate
endpoints. `random_pair` selects two distinct candidates, and `uniform_count`
selects `meas_count` distinct candidates. Candidate selection and firing are
recorded separately so circuit visualizations can show both.
"""
    ),
    code(
        "measurement_modes = {\n"
        "    'bernoulli': {},\n"
        "    'gated': {},\n"
        "    'random_pair': {},\n"
        "    'uniform_count': {'meas_count': 3},\n"
        "}\n"
        "for mode, extra in measurement_modes.items():\n"
        "    local = CircuitConfig(\n"
        "        graph_spec='cycle', n=8, measurement_mode=mode, p=1.0,\n"
        "        total_layers_override=1, **extra,\n"
        "    )\n"
        "    first = CircuitBuilder(local, 5).schedule()[0]\n"
        "    assert first.meas_qubits == first.meas_candidates\n"
        "    print(f'{mode:>13}: candidates={first.meas_candidates}')\n"
    ),
    md(
        r"""## Pair-seeded construction and independent runner streams

Equal scalar sums no longer imply equal construction streams. The runner adds
two independent tagged streams without perturbing this schedule:
`[base_seed, sample_seed, 0x6D656173]` for measurement outcomes and
`[base_seed, sample_seed, 0x73637262]` for the optional global scramble.
Thus toggling `scramble` cannot move gate placement, Clifford, or measurement
candidate draws.
"""
    ),
    code(
        "a_cfg = CircuitConfig(graph_spec='cycle', n=8, base_seed=10, total_layers_override=1)\n"
        "b_cfg = CircuitConfig(graph_spec='cycle', n=8, base_seed=11, total_layers_override=1)\n"
        "a_draws = CircuitBuilder(a_cfg, 2).rng.integers(2**32, size=6)\n"
        "b_draws = CircuitBuilder(b_cfg, 1).rng.integers(2**32, size=6)\n"
        "print('equal scalar sums:', 10 + 2 == 11 + 1)\n"
        "print('pair-seeded streams differ:', not (a_draws == b_draws).all())\n"
    ),
    md(
        r"""## Warmup and measured iterators

`warmup_layers_iter()` yields gate-only prescrambling layers. `layers()` yields
the measured phase lazily, while `schedule()` materializes that phase as a list.
A literal `total_layers_override` controls the measured iterator only.
"""
    ),
    code(
        "local = CircuitConfig(\n"
        "    graph_spec='cycle', n=8, picture='purification',\n"
        "    warmup_layers=2, total_layers_override=3,\n"
        ")\n"
        "builder = CircuitBuilder(local, 9)\n"
        "warm = list(builder.warmup_layers_iter())\n"
        "measured = builder.schedule()\n"
        "print('warmup / measured:', len(warm), '/', len(measured))\n"
        "print('warmup is gate-only:', all(layer.n_measurements == 0 for layer in warm))\n"
    ),
    md(
        r"""## Summary

The scheduler implements all four gate modes and all four measurement modes
with a fixed schema-v2 construction stream. Measurement outcomes and global
scrambling remain separately tagged runner streams.
"""
    ),
]

if __name__ == "__main__":
    build_and_execute("scheduler.ipynb", CELLS)
