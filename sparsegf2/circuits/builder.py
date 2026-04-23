"""
Circuit builder: translates a :class:`CircuitConfig` + ``sample_seed`` into a
deterministic stream of :class:`CircuitLayer` records.

The builder is the single source of truth for the circuit realization.
Both the SparseGF2 runner and the Stim parity tests consume the same
:class:`CircuitLayer` objects so that their executions are bit-for-bit
equivalent modulo simulator internals.

Gating modes:

- ``matching``    — apply a perfect matching every layer (n/2 gates).
- ``random_edge`` — apply exactly one gate per layer on a uniformly-random
  edge of the underlying graph.

RNG usage order per layer (this order MUST NOT change without bumping the
schema version, because it is load-bearing for reproducibility):

    1. gate-placement selection:
         - matching mode: matching selection (``palette`` / ``fresh`` only;
           ``round_robin`` draws nothing).
         - random_edge mode: one edge index sampled uniformly from the
           graph edge list.
    2. Clifford indices for each gate pair.
    3. measurement qubit Bernoullis (mode-specific: ``uniform`` draws n,
       ``gated`` draws len(candidates), ``random_pair`` draws a pair +
       2 Bernoullis).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np

from sparsegf2.circuits.config import CircuitConfig
from sparsegf2.circuits.graphs import GraphTopology, parse_graph_spec
from sparsegf2.circuits.matching import select_matching
from sparsegf2.circuits.measurements import sample_measurements


# CircuitLayer record

@dataclass
class CircuitLayer:
    """One layer of a graph-defined Clifford-measurement circuit."""

    gate_pairs: List[Tuple[int, int]]
    cliff_indices: np.ndarray                         # dtype int64, shape (len(gate_pairs),)
    meas_qubits: List[int]

    @property
    def n_gates(self) -> int:
        return len(self.gate_pairs)

    @property
    def n_measurements(self) -> int:
        return len(self.meas_qubits)


# CircuitBuilder

class CircuitBuilder:
    """Generate the circuit schedule for one ``(n, p, sample_seed)`` realization.

    Parameters
    ----------
    config : CircuitConfig
        All non-sample-seed knobs.
    sample_seed : int
        Per-sample offset added to ``config.base_seed`` to produce the
        sample's deterministic RNG seed.
    """

    def __init__(self, config: CircuitConfig, sample_seed: int = 0) -> None:
        self.config = config
        self.sample_seed = int(sample_seed)
        self.seed = int(config.base_seed) + self.sample_seed
        self.rng = np.random.default_rng(self.seed)
        self.graph: GraphTopology = parse_graph_spec(
            config.graph_spec, config.n, seed=self.seed
        )

    # --------------------------------------------------------------

    def warmup_layers_iter(self) -> Iterator[CircuitLayer]:
        """Yield ``CircuitConfig.warmup_layers`` gate-only layers.

        These apply the same gate schedule as :meth:`layers` but with no
        measurement candidates (``meas_qubits`` is always empty). The RNG
        is consumed before the main ``layers()`` iterator runs, so the
        main layers still begin at their canonical ``t = 0`` for the
        gating scheduler (e.g. ``round_robin``'s matching index cycles
        through its 1-factorization continuously).
        """
        cfg = self.config
        if cfg.warmup_layers <= 0:
            return
        edges = (
            self.graph.edges if cfg.gating_mode == "random_edge" else None
        )
        n_edges = len(edges) if edges is not None else 0
        for t in range(cfg.warmup_layers):
            if cfg.gating_mode == "matching":
                pairs = select_matching(self.graph, cfg.matching_mode, t, self.rng)
            elif cfg.gating_mode == "random_edge":
                if n_edges == 0:
                    pairs = []
                else:
                    j = int(self.rng.integers(0, n_edges))
                    u, v = edges[j]
                    pairs = [(int(u), int(v))]
            else:
                raise RuntimeError(
                    f"Unhandled gating_mode {cfg.gating_mode!r}"
                )
            n_pairs = len(pairs)
            if n_pairs:
                cliff_idx = self.rng.integers(
                    0, cfg.n_cliffords, size=n_pairs, dtype=np.int64
                )
            else:
                cliff_idx = np.zeros(0, dtype=np.int64)
            yield CircuitLayer(
                gate_pairs=pairs,
                cliff_indices=cliff_idx,
                meas_qubits=[],
            )

    def layers(self) -> Iterator[CircuitLayer]:
        """Yield ``CircuitLayer`` objects, one per layer, in simulation order."""
        T = self.config.total_layers()
        cfg = self.config
        edges = (
            self.graph.edges if cfg.gating_mode == "random_edge" else None
        )
        n_edges = len(edges) if edges is not None else 0

        for t in range(T):
            # 1. gate placement
            if cfg.gating_mode == "matching":
                pairs = select_matching(self.graph, cfg.matching_mode, t, self.rng)
            elif cfg.gating_mode == "random_edge":
                if n_edges == 0:
                    pairs = []
                else:
                    j = int(self.rng.integers(0, n_edges))
                    u, v = edges[j]
                    pairs = [(int(u), int(v))]
            else:
                raise RuntimeError(
                    f"Unhandled gating_mode {cfg.gating_mode!r}"
                )

            # 2. Clifford indices
            n_pairs = len(pairs)
            if n_pairs:
                cliff_idx = self.rng.integers(
                    0, cfg.n_cliffords, size=n_pairs, dtype=np.int64
                )
            else:
                cliff_idx = np.zeros(0, dtype=np.int64)

            # 3. measurement qubits
            meas = sample_measurements(
                cfg.measurement_mode, cfg.n, cfg.p, pairs, self.rng
            )

            yield CircuitLayer(gate_pairs=pairs, cliff_indices=cliff_idx, meas_qubits=meas)

    # --------------------------------------------------------------

    def schedule(self) -> List[CircuitLayer]:
        """Eager form of :meth:`layers` — return the whole schedule as a list."""
        return list(self.layers())


__all__ = ["CircuitLayer", "CircuitBuilder"]
