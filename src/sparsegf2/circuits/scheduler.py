"""Circuit scheduler: ``CircuitConfig`` + ``sample_seed`` → a stream of layers.

The :class:`CircuitBuilder` is the single source of truth for a circuit
*realization*: which gate pairs fire, which Clifford index each gets, and
which qubits are measured, layer by layer. It is deliberately
**simulator-free**: it only produces :class:`CircuitLayer` records; the
runner (or a Stim parity test) consumes them. That separation means the
exact same realization can be replayed on different backends.

Gating modes
------------
- ``brickwork``: fire a whole perfect matching each layer (``n/2`` gates),
  chosen by :func:`sparsegf2.circuits.matching.select_matching`.
- ``random_edge``: fire ``config.gates_per_layer`` *distinct* uniformly-random
  graph edges each layer (may share vertices, so not a matching). Default ``m=1``
  (single-edge); ``total_layers`` scales with ``m`` (a single-edge circuit needs
  ``O(n^2)`` layers; see :meth:`CircuitConfig.total_layers`).
- ``random_pool``: fire ``config.gates_per_layer`` uniformly-random edges *with
  replacement* each layer (default ``m=n/2``, holding the gate:measurement ratio
  at brickwork's ``1:2p``; raise the coefficient to change the ratio).
- ``all_edges``: fire every stored graph edge, in order, on every layer. Gate
  placement is deterministic and consumes no RNG.

RNG model (single stream)
-------------------------
This builder uses **one** ``np.random.Generator`` seeded from the **pair**
``[base_seed, sample_seed]`` (schema v2; the v1 scalar ``base_seed +
sample_seed`` collided across cells: ``(b, s)`` and ``(b + k, s - k)`` shared
every stream, so sweeping ``base_seed`` for stochastic-graph averaging while
also sweeping samples silently duplicated trajectories). The stream is
consumed in a fixed per-layer order that is **load-bearing for
reproducibility** and must not change without a schema bump:

    1. gate placement
       - ``brickwork``: matching selection (``palette``/``fresh`` draw;
         ``round_robin`` draws nothing)
       - ``random_edge``: ``m`` distinct edge indices
       - ``random_pool``: ``m`` edge indices with replacement
       - ``all_edges``: no draw
    2. Clifford indices, one per gate pair
    3. measurement-candidate draws (mode-specific count)

The measurement *outcomes* are a **separate** stream living on the
:class:`~sparsegf2.SparseGF2` instance (seeded independently in
``setup_picture``), so circuit construction and measurement results are
decoupled. (A future option is to split (1)/(3) onto their own generators
too; not done by default, to keep the mental model simple.)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from sparsegf2.circuits.config import CircuitConfig
from sparsegf2.circuits.graphs import GraphTopology, from_spec
from sparsegf2.circuits.matching import select_matching
from sparsegf2.circuits.measurements import sample_measurements_detailed
from sparsegf2.errors import InvalidArgumentError


def _normalize_sample_seed(sample_seed: object) -> int:
    """Validate a SeedSequence entropy component without lossy coercion."""
    if not isinstance(sample_seed, (int, np.integer)) or isinstance(sample_seed, (bool, np.bool_)):
        raise InvalidArgumentError(
            "sample_seed must be a non-negative integer; "
            f"got {sample_seed!r} ({type(sample_seed).__name__})"
        )
    sample_seed = int(sample_seed)
    if sample_seed < 0:
        raise InvalidArgumentError(
            f"sample_seed must be a non-negative integer; got {sample_seed!r}"
        )
    return sample_seed


@dataclass
class CircuitLayer:
    """One layer of a graph-defined Clifford+measurement circuit.

    Attributes
    ----------
    gate_pairs : list of (int, int)
        The qubit pairs that get a two-qubit Clifford this layer.
    cliff_indices : ndarray[int64]
        Shape ``(len(gate_pairs),)``; ``cliff_indices[g]`` indexes the
        ``Sp(4, F_2)`` table for ``gate_pairs[g]``.
    meas_qubits : list of int
        Sorted, deduplicated qubits that **fired** (were actually measured in
        Z) this layer, the subset of ``meas_candidates`` that passed the
        Bernoulli(p) coin. The runner measures exactly these.
    meas_candidates : list of int
        Qubits that were *eligible* to be measured this layer (the mode's
        candidate set). ``meas_qubits ⊆ meas_candidates``. Carried for
        visualization; the runner ignores it.
    """

    gate_pairs: list[tuple[int, int]]
    cliff_indices: np.ndarray
    meas_qubits: list[int]
    meas_candidates: list[int] = field(default_factory=list)

    @property
    def n_gates(self) -> int:
        return len(self.gate_pairs)

    @property
    def n_measurements(self) -> int:
        return len(self.meas_qubits)


class CircuitBuilder:
    """Generate the schedule for one ``(config, sample_seed)`` realization.

    Parameters
    ----------
    config : CircuitConfig
        All non-sample-seed knobs (validated already in its ``__post_init__``).
    sample_seed : int
        Non-negative exact integer paired with ``config.base_seed`` as
        ``[base_seed, sample_seed]`` to seed this realization's RNG. Boolean
        aliases and values requiring a lossy integer coercion are rejected.
    """

    def __init__(self, config: CircuitConfig, sample_seed: int = 0) -> None:
        self.config = config
        self.sample_seed = _normalize_sample_seed(sample_seed)
        # Pair-seeded (schema v2): [base_seed, sample_seed] gives every
        # (base_seed, sample_seed) cell its own stream. The v1 scalar sum made
        # (b, s) and (b + k, s - k) bit-identical.
        self.rng = np.random.default_rng([int(config.base_seed), self.sample_seed])
        # Reuse the graph the config already resolved + validated in
        # __post_init__ (``_graph``): avoids a per-sample rebuild and
        # guarantees the executed graph is the one validation checked. The
        # getattr fallback keeps the builder usable with a hand-made config
        # object that skipped __post_init__ (defensive; not the normal path).
        cached = getattr(config, "_graph", None)
        if cached is not None:
            self.graph: GraphTopology = cached
        elif isinstance(config.graph_spec, GraphTopology):
            self.graph = config.graph_spec
        else:
            # base_seed (not base+sample): the stochastic-graph realization is
            # quenched per config, exactly as the normal __post_init__ path
            # resolves it; a per-sample seed here would silently anneal the
            # geometry and diverge from what validation checked.
            self.graph = from_spec(config.graph_spec, config.n, seed=int(config.base_seed))

    # ---- internal: one layer's gate placement + Clifford indices ----

    def _place_gates(self, t: int, edges, n_edges: int):
        """Return ``(gate_pairs, cliff_indices)`` for layer ``t``.

        Random placement modes consume RNG for step 1; deterministic modes do
        not. Step 2 then draws one Clifford index per selected edge.
        """
        cfg = self.config
        if cfg.gating_mode == "brickwork":
            pairs = select_matching(self.graph, cfg.matching_mode, t, self.rng)
        elif cfg.gating_mode == "all_edges":
            # Fire EVERY graph edge each layer, in the graph's fixed edge order.
            # Fully deterministic placement (no RNG consumed in step 1); the RNG
            # is spent only on the per-gate Clifford indices (step 2) and the
            # measurements (step 3).
            pairs = list(self.graph.edges)
        elif cfg.gating_mode in ("random_edge", "random_pool"):
            if n_edges == 0:
                pairs = []
            else:
                # random_edge: m *distinct* random edges (default 1).
                # random_pool: m random edges *with replacement* (default n/2),
                #   so an edge may repeat and m may exceed the edge count, the
                #   "O(n) independent random gates per layer" model.
                m = cfg.resolved_gates_per_layer()
                replace = cfg.gating_mode == "random_pool"
                if not replace:
                    m = min(m, n_edges)  # can't draw more distinct edges than exist
                idx = self.rng.choice(n_edges, size=m, replace=replace)
                pairs = [edges[int(j)] for j in idx]
        else:  # pragma: no cover, config validation forbids this
            raise AssertionError(f"Unhandled gating_mode {cfg.gating_mode!r}")

        n_pairs = len(pairs)
        if n_pairs:
            cliff_idx = self.rng.integers(0, cfg.n_cliffords, size=n_pairs, dtype=np.int64)
        else:
            cliff_idx = np.zeros(0, dtype=np.int64)
        return pairs, cliff_idx

    # ---- public iterators ----

    def warmup_layers_iter(self) -> Iterator[CircuitLayer]:
        """Yield ``config.warmup_layers`` gate-only layers (no measurements).

        Pre-scrambling: the same gate schedule as :meth:`layers` but with an
        empty ``meas_qubits`` every layer. Consumes the RNG before
        :meth:`layers` runs, so the measured phase still begins at the
        scheduler's canonical ``t = 0``.
        """
        cfg = self.config
        if cfg.warmup_layers <= 0:
            return
        edges = self.graph.edges if cfg.gating_mode in ("random_edge", "random_pool") else None
        n_edges = len(edges) if edges is not None else 0
        for t in range(cfg.warmup_layers):
            pairs, cliff_idx = self._place_gates(t, edges, n_edges)
            yield CircuitLayer(gate_pairs=pairs, cliff_indices=cliff_idx, meas_qubits=[])

    def layers(self) -> Iterator[CircuitLayer]:
        """Yield one :class:`CircuitLayer` per measured layer, in run order."""
        cfg = self.config
        T = cfg.total_layers()
        edges = self.graph.edges if cfg.gating_mode in ("random_edge", "random_pool") else None
        n_edges = len(edges) if edges is not None else 0
        for t in range(T):
            pairs, cliff_idx = self._place_gates(t, edges, n_edges)
            count = cfg.resolved_meas_count() if cfg.measurement_mode == "uniform_count" else None
            candidates, fired = sample_measurements_detailed(
                cfg.measurement_mode, cfg.n, cfg.p, pairs, self.rng, count=count
            )
            yield CircuitLayer(
                gate_pairs=pairs,
                cliff_indices=cliff_idx,
                meas_qubits=fired,
                meas_candidates=candidates,
            )

    def schedule(self) -> list[CircuitLayer]:
        """Eager form of :meth:`layers`: the whole measured schedule as a list."""
        return list(self.layers())


__all__ = ["CircuitLayer", "CircuitBuilder"]
