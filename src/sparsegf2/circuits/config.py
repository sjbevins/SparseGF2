"""Configuration for one graph-defined circuit cell.

:class:`CircuitConfig` is the single bag of knobs that, together with a
``sample_seed``, fully determines a circuit realization and its
simulation. Everything is validated eagerly in ``__post_init__`` so a
bad cell fails at construction with a clear
:class:`~sparsegf2.errors.InvalidArgumentError`, not deep inside the
runner.

Sweep-level orchestration (iterating over ``n``, ``p``, many seeds,
writing parquet) deliberately lives *outside* this package. It belongs
to the future ``sparsegf2.analysis`` layer. A ``CircuitConfig`` is one
cell; the analysis package will hold the many-cell driver.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from sparsegf2.circuits.graphs import GraphTopology, from_networkx, from_spec
from sparsegf2.circuits.matching import MATCHING_MODES, available_modes
from sparsegf2.circuits.measurements import MEASUREMENT_MODES
from sparsegf2.circuits.picture import Picture
from sparsegf2.errors import InvalidArgumentError

GATING_MODES = ("brickwork", "random_edge", "random_pool")
DEPTH_MODES = ("O(n)", "O(log_n)", "until_purified")


@dataclass
class CircuitConfig:
    """Per-cell circuit configuration.

    Parameters
    ----------
    graph_spec : str, GraphTopology, or networkx.Graph
        Either a spec name resolved by
        :func:`sparsegf2.circuits.graphs.from_spec` (``"cycle"`` / ``"complete"``
        / ``"path"`` / ``"lattice_2d"`` / ``"newman_watts"``), a prebuilt
        :class:`GraphTopology` (e.g. from :func:`~sparsegf2.circuits.graphs.newman_watts`
        with tuned parameters), or any simple undirected ``networkx.Graph``.
        The stochastic ``newman_watts`` geometry has no 1-factorization, so the
        ``round_robin`` / ``palette`` matching modes are unavailable on it (use
        ``random_edge`` / ``random_pool``, or ``brickwork`` with ``fresh``).
    n : int
        Number of system qubits. The built-in graphs need even ``n`` to admit a
        1-factorization; that is enforced lazily when the graph is built,
        not here.
    picture : Picture or str
        Physics picture (``"pure_state"`` / ``"purification"`` / ``"single_ref"``).
    gating_mode : {"brickwork", "random_edge", "random_pool"}
        ``brickwork`` fires a whole perfect matching each layer (``n/2``
        gates); ``random_edge`` fires ``gates_per_layer`` *distinct* random
        graph edges each layer (default 1); ``random_pool`` fires
        ``gates_per_layer`` random edges **with replacement** (default ``n/2``,
        which holds the gate:measurement ratio at brickwork's ``1:2p``; raise
        the coefficient, e.g. ``n``, to change the ratio deliberately).
    matching_mode : {"round_robin", "palette", "fresh"}
        How brickwork picks its matching each layer (ignored for the
        random-edge modes).
    gates_per_layer : int or callable(CircuitConfig) -> int, optional
        Number of random edges fired per layer in ``random_edge`` /
        ``random_pool`` mode. ``None`` (default) uses the per-mode default
        (``random_edge`` → 1; ``random_pool`` → ``n/2``). A callable is resolved
        against this config so the count can scale with ``n`` (e.g.
        ``lambda cfg: cfg.n``). **Only valid for the random-edge modes**; a
        value with ``brickwork`` raises.
    measurement_mode : {"bernoulli", "gated", "random_pair", "uniform_count"}
        Which qubits become measurement candidates each layer. ``uniform_count``
        samples ``meas_count`` qubits uniformly (default ``n/2``).
    meas_count : int or None, optional
        Number of candidate qubits for ``uniform_count`` measurement (default
        ``n/2``). Only valid with ``measurement_mode='uniform_count'``.
    p : float
        Per-qubit measurement probability, in ``[0, 1]``.
    depth_mode : {"O(n)", "O(log_n)", "until_purified"}
        How total circuit depth scales with ``n``.
    depth_factor : int
        Multiplier in the depth formula (``>= 1``).
    n_cliffords : int
        Size of the Clifford table to sample gate indices from. Defaults to
        ``720 = |Sp(4, F_2)|``, **not** 11,520 (the sign-decorated Clifford
        count, which is invisible to the phase-free simulator).
    base_seed : int
        Base RNG seed; the per-sample seed is ``base_seed + sample_seed``.
    record_time_series : bool
        Record the order parameter after every layer (a future runner
        feature). Forbidden for ``pure_state`` (no reference to track).
    warmup_layers : int
        Number of gate-only pre-scrambling layers before the measured loop
        (``>= 0``).
    scramble : bool
        If true, apply one global random Clifford to the ``n_system`` system
        qubits before the (warmup and) measured phases, a one-shot
        maximally-scrambling unitary, the standard initial condition for the
        single-qubit purification probe. Drawn from an independent construction
        sub-stream (toggling it does not perturb the gate/measurement streams).
        Exact but ``O(n^3)``; for large ``n`` a deep gate-only ``warmup_layers``
        scramble is cheaper (see ``benchmarks/benchmark_scramble.py``).
    scramble_entangled_qubit : bool
        Only relevant for ``single_ref`` with ``scramble=True``. The single
        reference is Bell-paired with system qubit ``n-1``. When true (default)
        the global scramble covers all ``n`` system qubits, so the probe qubit
        is immediately mixed into the bulk. When false the scramble acts on the
        other ``n-1`` system qubits only, leaving qubit ``n-1`` out, so the
        reference stays localized on its partner until the monitored dynamics
        carry the entanglement into the system. The two choices are different
        single-qubit-probe protocols. For ``pure_state`` and ``purification``
        this flag has no effect (the scramble always covers every system qubit).
    pivot_mode : str or None
        Forwarded to :class:`~sparsegf2.SparseGF2` (measurement pivot rule).
    use_numba : bool or None
        Forwarded to :class:`~sparsegf2.SparseGF2` (JIT toggle).
    hybrid : bool
        Forwarded to :class:`~sparsegf2.SparseGF2`: enable the sparse/dense
        hybrid path. Much faster on volume-law / low-measurement-rate circuits
        (the dense regime), where the sparse inverted index costs more than it
        saves. The physical state and every gauge-invariant observable are
        identical either way (only the basis-dependent generator-weight
        diagnostics may differ). Default ``False``.
    """

    graph_spec: str | GraphTopology
    n: int
    picture: Picture | str = Picture.PURE_STATE
    gating_mode: Literal["brickwork", "random_edge", "random_pool"] = "brickwork"
    matching_mode: Literal["round_robin", "palette", "fresh"] = "round_robin"
    gates_per_layer: int | Callable[[CircuitConfig], int] | None = None
    measurement_mode: Literal["bernoulli", "gated", "random_pair", "uniform_count"] = "bernoulli"
    meas_count: int | None = None
    p: float = 0.15
    depth_mode: Literal["O(n)", "O(log_n)", "until_purified"] = "O(n)"
    depth_factor: int = 8
    n_cliffords: int = 720
    base_seed: int = 42
    record_time_series: bool = False
    warmup_layers: int = 0
    scramble: bool = False
    scramble_entangled_qubit: bool = True
    pivot_mode: str | None = None
    use_numba: bool | None = None
    hybrid: bool = False

    def __post_init__(self) -> None:
        # ---- graph_spec ----
        # A str (named family) or a prebuilt GraphTopology pass through. Anything
        # else is adapted as a networkx graph (simple, undirected) via
        # from_networkx, which raises a clear error for invalid objects.
        if not isinstance(self.graph_spec, (str, GraphTopology)):
            self.graph_spec = from_networkx(self.graph_spec)
        # ---- n ----
        if not isinstance(self.n, (int, np.integer)) or self.n < 2:
            raise InvalidArgumentError(f"n must be an integer >= 2; got {self.n!r}")
        self.n = int(self.n)
        if isinstance(self.graph_spec, GraphTopology) and self.graph_spec.n != self.n:
            raise InvalidArgumentError(
                f"graph_spec graph has n={self.graph_spec.n} but config n={self.n}"
            )
        # ---- picture (coerce to enum) ----
        try:
            self.picture = Picture(self.picture)
        except ValueError as exc:
            raise InvalidArgumentError(
                f"picture must be one of {[p.value for p in Picture]}; got {self.picture!r}"
            ) from exc
        # ---- gating_mode ----
        if self.gating_mode not in GATING_MODES:
            raise InvalidArgumentError(
                f"gating_mode must be one of {GATING_MODES}; got {self.gating_mode!r}"
            )
        # ---- matching_mode ----
        if self.matching_mode not in MATCHING_MODES:
            raise InvalidArgumentError(
                f"matching_mode must be one of {MATCHING_MODES}; got {self.matching_mode!r}"
            )
        # ---- gates_per_layer (random-edge modes only) ----
        if self.gates_per_layer is None:
            pass  # per-mode default, resolved lazily (1 for random_edge, n/2 for random_pool)
        elif callable(self.gates_per_layer):
            pass  # resolved lazily against this config (e.g. lambda cfg: cfg.n)
        elif isinstance(self.gates_per_layer, (int, np.integer)):
            if self.gates_per_layer < 1:
                raise InvalidArgumentError(
                    f"gates_per_layer must be >= 1; got {self.gates_per_layer!r}"
                )
            self.gates_per_layer = int(self.gates_per_layer)
        else:
            raise InvalidArgumentError(
                "gates_per_layer must be None, a positive int, or a callable "
                f"(CircuitConfig -> int); got {type(self.gates_per_layer).__name__}"
            )
        # gates_per_layer only applies to the random-edge modes; brickwork always
        # fires a full matching. Reject any value with brickwork rather than
        # silently ignore it.
        if self.gating_mode == "brickwork" and self.gates_per_layer is not None:
            raise InvalidArgumentError(
                "gates_per_layer only applies to gating_mode='random_edge'/'random_pool'; "
                "brickwork always fires a full matching (n/2 gates)"
            )
        # ---- measurement_mode ----
        if self.measurement_mode not in MEASUREMENT_MODES:
            raise InvalidArgumentError(
                f"measurement_mode must be one of {MEASUREMENT_MODES}; got {self.measurement_mode!r}"
            )
        # ---- meas_count (uniform_count only) ----
        if self.meas_count is not None:
            # Mode mismatch first; it is the more actionable error (the user
            # likely meant to also set measurement_mode='uniform_count').
            if self.measurement_mode != "uniform_count":
                raise InvalidArgumentError(
                    "meas_count only applies to measurement_mode='uniform_count'"
                )
            if not isinstance(self.meas_count, (int, np.integer)) or self.meas_count < 1:
                raise InvalidArgumentError(
                    f"meas_count must be a positive int; got {self.meas_count!r}"
                )
            self.meas_count = int(self.meas_count)
            if self.meas_count > self.n:
                raise InvalidArgumentError(f"meas_count={self.meas_count} exceeds n={self.n}")
        # ---- p ----
        if not 0.0 <= self.p <= 1.0:
            raise InvalidArgumentError(f"p must be in [0, 1]; got {self.p}")
        self.p = float(self.p)
        # ---- depth ----
        if self.depth_mode not in DEPTH_MODES:
            raise InvalidArgumentError(
                f"depth_mode must be one of {DEPTH_MODES}; got {self.depth_mode!r}"
            )
        if not isinstance(self.depth_factor, (int, np.integer)) or self.depth_factor < 1:
            raise InvalidArgumentError(
                f"depth_factor must be a positive integer; got {self.depth_factor!r}"
            )
        self.depth_factor = int(self.depth_factor)
        # ---- n_cliffords ----
        if not isinstance(self.n_cliffords, (int, np.integer)) or not (
            1 <= self.n_cliffords <= 720
        ):
            raise InvalidArgumentError(
                f"n_cliffords must be an integer in [1, 720]; got {self.n_cliffords!r}"
            )
        self.n_cliffords = int(self.n_cliffords)
        # ---- base_seed ----
        if not isinstance(self.base_seed, (int, np.integer)):
            raise InvalidArgumentError(f"base_seed must be an integer; got {self.base_seed!r}")
        self.base_seed = int(self.base_seed)
        # ---- record_time_series ----
        if not isinstance(self.record_time_series, bool):
            raise InvalidArgumentError(
                f"record_time_series must be bool; got {self.record_time_series!r}"
            )
        if self.record_time_series and self.picture is Picture.PURE_STATE:
            raise InvalidArgumentError(
                "record_time_series=True needs a reference subsystem to track; "
                "it is not meaningful for picture='pure_state'"
            )
        # ---- until_purified only makes sense with a reference ----
        if self.depth_mode == "until_purified" and self.picture is Picture.PURE_STATE:
            raise InvalidArgumentError(
                "depth_mode='until_purified' needs a reference order parameter; "
                "it is not defined for picture='pure_state'"
            )
        # ---- warmup_layers ----
        if not isinstance(self.warmup_layers, (int, np.integer)) or self.warmup_layers < 0:
            raise InvalidArgumentError(
                f"warmup_layers must be a non-negative integer; got {self.warmup_layers!r}"
            )
        self.warmup_layers = int(self.warmup_layers)

        # ---- scramble ----
        if not isinstance(self.scramble, bool):
            raise InvalidArgumentError(f"scramble must be bool; got {self.scramble!r}")
        if not isinstance(self.scramble_entangled_qubit, bool):
            raise InvalidArgumentError(
                f"scramble_entangled_qubit must be bool; got {self.scramble_entangled_qubit!r}"
            )

        if not isinstance(self.hybrid, bool):
            raise InvalidArgumentError(f"hybrid must be bool; got {self.hybrid!r}")

        # ---- graph + gating/matching compatibility (eager, fail-fast) ----
        # Resolve the graph now so an incompatible (graph, mode) pair raises
        # InvalidArgumentError at construction, not a RuntimeError deep in
        # the scheduler. The resolved graph is cached on ``_graph`` and reused
        # by CircuitBuilder. For a stochastic string spec (``newman_watts``) the
        # ``base_seed`` is the graph realization seed, so the geometry is fixed
        # across a config's trajectories and varies when ``base_seed`` is swept;
        # deterministic specs ignore the seed.
        self._graph: GraphTopology = (
            self.graph_spec
            if isinstance(self.graph_spec, GraphTopology)
            else from_spec(self.graph_spec, self.n, self.base_seed)
        )
        if self.gating_mode == "brickwork":
            needs_factorization = self.matching_mode in ("round_robin", "palette")
            if needs_factorization and not self._graph.has_one_factorization:
                raise InvalidArgumentError(
                    f"matching_mode={self.matching_mode!r} needs a 1-factorization, but "
                    f"graph {self._graph.name!r} has none (e.g. odd n). "
                    f"Available matching modes: {available_modes(self._graph)}"
                )
            if self.matching_mode == "fresh" and not self._graph.has_perfect_matching:
                raise InvalidArgumentError(
                    f"matching_mode='fresh' needs a perfect matching, but graph "
                    f"{self._graph.name!r} has none. "
                    f"Available matching modes: {available_modes(self._graph)}"
                )
        elif self.gating_mode in ("random_edge", "random_pool") and not self._graph.edges:
            raise InvalidArgumentError(
                f"gating_mode={self.gating_mode!r} needs at least one edge, but graph "
                f"{self._graph.name!r} has none"
            )

        # Eagerly validate a callable gates_per_layer for the random-edge modes,
        # so a bad callable fails fast at construction (not deep in the scheduler).
        if self.gating_mode in ("random_edge", "random_pool") and callable(self.gates_per_layer):
            try:
                self.resolved_gates_per_layer()
            except InvalidArgumentError:
                raise
            except Exception as exc:
                raise InvalidArgumentError(
                    f"gates_per_layer callable failed when evaluated against this config: {exc}"
                ) from exc

    # ---- derived quantities ----

    def resolved_gates_per_layer(self) -> int:
        """Resolve :attr:`gates_per_layer` to a concrete positive int.

        ``None`` uses the per-mode default: 1 for ``random_edge`` (a single
        edge), ``n/2`` for ``random_pool`` (matches brickwork's gate count, so
        the ratio stays ``1:2p``). A callable is evaluated against this config
        (so the count can scale with ``n``); an int passes through. Only
        meaningful for the random-edge modes (brickwork fires a full matching).
        """
        g = self.gates_per_layer
        if g is None:
            m = max(1, self.n // 2) if self.gating_mode == "random_pool" else 1
        else:
            m = int(g(self)) if callable(g) else int(g)
        if m < 1:
            raise InvalidArgumentError(f"gates_per_layer resolved to {m}; must be >= 1")
        return m

    def resolved_meas_count(self) -> int:
        """Resolve :attr:`meas_count` for ``uniform_count`` (default ``n/2``)."""
        k = self.meas_count if self.meas_count is not None else max(1, self.n // 2)
        return max(1, min(int(k), self.n))

    def total_layers(self) -> int:
        """Number of measured layers (time steps) for this cell.

        Depth is measured in **gates per qubit**, so circuits are comparable
        across gating modes. The brickwork-equivalent base budget is:

        - ``O(n)`` / ``until_purified`` → ``depth_factor * n``
        - ``O(log_n)``                 → ``depth_factor * ceil(log2 n)``

        A brickwork layer fires ``n/2`` gates and touches every qubit once,
        so the base value *is* the gates-per-qubit budget and is returned
        directly. ``random_edge`` fires ``m = resolved_gates_per_layer()``
        gates/layer, touching only ``2m/n`` of the qubits per layer, so it
        needs ``n/(2m)`` times as many layers to reach the same budget:

        - ``random_edge`` → ``round(base * n / (2m))``.

        Consequence (the reason this is mode-aware): a **single-edge**
        ``random_edge`` circuit (``m=1``) runs ``O(n^2)`` layers; an
        ``O(n)`` layer count is *not* enough at one gate per step. An
        ``m = n/2`` random_edge circuit matches brickwork's layer count.
        """
        if self.depth_mode in ("O(n)", "until_purified"):
            base = max(1, self.depth_factor * self.n)
        elif self.depth_mode == "O(log_n)":
            base = max(1, self.depth_factor * max(1, int(math.ceil(math.log2(self.n)))))
        else:
            raise AssertionError(f"Unhandled depth_mode {self.depth_mode!r}")
        if self.gating_mode in ("random_edge", "random_pool"):
            m = self.resolved_gates_per_layer()
            return max(1, round(base * self.n / (2 * m)))
        return base

    def total_qubits(self) -> int:
        """Physical qubit count for this picture (``n``, ``2n``, or ``n+1``)."""
        if self.picture is Picture.PURE_STATE:
            return self.n
        if self.picture is Picture.PURIFICATION:
            return 2 * self.n
        if self.picture is Picture.SINGLE_REF:
            return self.n + 1
        raise AssertionError(f"Unhandled picture {self.picture!r}")

    def scramble_qubits(self) -> range | None:
        """System qubits the global scramble acts on, or ``None`` when off.

        The single source of truth for the global-scramble support, read by both
        the runner (which applies the Clifford) and the inspector (which draws
        it), so the diagram can never disagree with what runs. Returns
        ``range(n)`` for every picture, except ``single_ref`` with
        ``scramble_entangled_qubit=False``, where it returns ``range(n-1)`` to
        hold the reference's partner (qubit ``n-1``) out of the scramble.
        Returns ``None`` when ``scramble`` is off.
        """
        if not self.scramble:
            return None
        if self.picture is Picture.SINGLE_REF and not self.scramble_entangled_qubit:
            return range(self.n - 1)
        return range(self.n)

    def expected_gate_to_meas_ratio(self) -> float:
        """Expected gates-per-layer ÷ measurements-per-layer (mode-aware).

        Computed *before* any simulation from the cell's knobs, and recorded
        on every :class:`~sparsegf2.circuits.SampleRecord` as
        ``gate_to_meas_ratio_expected`` (the runner compares it against the
        realized ``gate_to_meas_ratio_actual``).

        Expected gates per layer ``eg``: ``n/2`` for brickwork (a full
        matching), ``m = resolved_gates_per_layer()`` for ``random_edge`` /
        ``random_pool``. Expected measurements per layer ``em`` by mode:

        - ``bernoulli``     : ``n * p``       (every qubit, prob p)
        - ``gated``         : ``2 * eg * p``  (see caveat below)
        - ``random_pair``   : ``2 * p``       (exactly 2 candidates)
        - ``uniform_count`` : ``meas_count * p``

        Returns ``inf`` when ``em == 0`` (e.g. ``p == 0``). The ratio is
        mode-aware rather than a blanket ``1/(2p)``.

        .. note::

           The ``gated`` estimate assumes each layer's gates touch ``2 * eg``
           **distinct** qubits. That is exact for a brickwork layer that is a
           *perfect matching*. It is an **over-estimate** of the candidate count
           (so the returned ratio is a lower bound on the realized one) when
           gates share or repeat vertices (``random_edge`` with dense ``m``,
           ``random_pool`` which draws edges *with replacement*) or when a
           brickwork color class is not a perfect matching (path / lattice /
           irregular networkx graphs). The realized ``gate_to_meas_ratio_actual``
           on each :class:`SampleRecord` is always exact.
        """
        eg = (
            self.n / 2.0
            if self.gating_mode == "brickwork"
            else float(self.resolved_gates_per_layer())
        )
        if self.measurement_mode == "bernoulli":
            em = self.n * self.p
        elif self.measurement_mode == "gated":
            em = 2.0 * eg * self.p
        elif self.measurement_mode == "uniform_count":
            em = self.resolved_meas_count() * self.p
        else:  # random_pair
            em = 2.0 * self.p
        return float("inf") if em == 0 else eg / em

    def to_dict(self) -> dict:
        """Serializable dict for manifests.

        The ``Picture`` enum becomes its string value, a prebuilt graph
        becomes its name, and a callable ``gates_per_layer`` is resolved to its
        concrete int for the random-edge modes (callables don't serialize).
        The result round-trips through ``CircuitConfig(**d)`` for every gating
        mode: ``gates_per_layer`` is emitted as ``None`` for ``brickwork`` (which
        rejects any explicit value).
        """
        d = asdict(self)
        d["picture"] = str(self.picture)
        if isinstance(self.graph_spec, GraphTopology):
            d["graph_spec"] = self.graph_spec.name
        d["gates_per_layer"] = (
            self.resolved_gates_per_layer()
            if self.gating_mode in ("random_edge", "random_pool")
            else None
        )
        return d


__all__ = ["GATING_MODES", "DEPTH_MODES", "CircuitConfig"]
