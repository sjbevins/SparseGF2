"""Circuit inspector: turn any construction into a viewable trace.

The inspector answers "what circuit did this config actually build?": it shows
the deterministic setup gates (Bell pairs) and the first $N$ layers of the
random Clifford + measurement schedule, in a human-readable form, marking
clearly what is **random** (the specific two-qubit Cliffords, which qubits
get measured) versus **deterministic** (the picture's construction gates).

Architecture (the part that makes this extensible)
--------------------------------------------------
There are two layers, deliberately separated:

1. :func:`trace_circuit` builds a **structured trace**, a
   :class:`CircuitTrace` of :class:`~sparsegf2.circuits.ops.Op` records, from
   a :class:`CircuitConfig`. It runs **no simulation**; the schedule and the
   picture's :func:`~sparsegf2.circuits.picture.setup_ops` fully determine the
   construction. This is the stable intermediate representation (IR).
2. A **renderer** (:func:`render_text` today) consumes the IR and produces a
   view. New views (a quantikz/LaTeX diagram, JSON, an interactive widget)
   are new functions over the *same* ``CircuitTrace``; they don't touch the
   trace builder.

Extending
---------
- **New gating / matching / measurement mode**: nothing to do, since it already
  emits ``CircuitLayer``, which :func:`_layer_ops` renders generically.
- **New picture**: add its branch to
  :func:`~sparsegf2.circuits.picture._picture_layout`; its deterministic setup
  gates then appear automatically.
- **New operation kind** (reset, mid-circuit feed-forward, …): emit an
  ``Op`` with a new ``kind`` and add one branch to :func:`_render_stage`.
- **New renderer**: write ``render_<format>(trace)`` over ``CircuitTrace``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sparsegf2.circuits.config import CircuitConfig
from sparsegf2.circuits.ops import DIRECTED_GATES, Op
from sparsegf2.circuits.picture import Picture, _picture_layout, setup_ops
from sparsegf2.circuits.scheduler import CircuitBuilder


@dataclass
class Stage:
    """One contiguous block of a circuit trace.

    Attributes
    ----------
    label : str
        Display label, e.g. ``"setup"``, ``"warmup 0"``, ``"layer 3"``.
    kind : str
        ``"setup"`` (deterministic construction), ``"warmup"`` (gate-only
        pre-scramble), or ``"layer"`` (a measured layer).
    index : int or None
        Layer index within its kind (``None`` for setup).
    ops : list of Op
        The operations in this stage, in application order.
    """

    label: str
    kind: str
    index: int | None
    ops: list[Op]


@dataclass
class CircuitTrace:
    """A structured, render-agnostic view of a circuit construction.

    Attributes
    ----------
    summary : dict
        Human-readable config facts (picture, qubit layout, graph, modes,
        depth, expected gate:measurement ratio, seed).
    stages : list of Stage
        Setup, then any warmup layers, then the first measured layers.
    total_layers : int
        Total measured layers the full circuit would run.
    shown_layers : int
        How many measured layers this trace includes.
    truncated : bool
        ``True`` when ``shown_layers < total_layers``.
    n_qubits : int
        Total physical qubits (system + reference), the number of wires a
        diagram renderer draws.
    reference_qubits : tuple of int
        Indices of the reference qubits (empty for pure_state); a renderer
        can shade their wires to distinguish system from reference.
    """

    summary: dict[str, Any]
    stages: list[Stage]
    total_layers: int
    shown_layers: int
    truncated: bool = field(default=False)
    n_qubits: int = 0
    reference_qubits: tuple[int, ...] = ()


# ----------------------------------------------------------------------
# Trace builder (the IR)
# ----------------------------------------------------------------------


def _layer_ops(layer) -> list[Op]:
    """Convert a :class:`CircuitLayer` to its ``Op`` list.

    Two-qubit gates are **random** (a uniform draw from the Sp(4) table); the
    table index is kept in ``detail`` so a reader can verify reproducibility.

    One measure op is emitted per measurement **candidate** (a qubit eligible
    to be measured this layer), with ``detail["fired"]`` recording whether it
    passed its Bernoulli(p) coin and was actually measured. So a renderer can
    show the candidate set alongside the measurements that fired.
    """
    ops: list[Op] = []
    for g, (qi, qj) in enumerate(layer.gate_pairs):
        ci = int(layer.cliff_indices[g])
        ops.append(
            Op(
                kind="gate",
                qubits=(int(qi), int(qj)),
                label=f"C[{ci}]",
                random=True,
                detail={"clifford_index": ci, "family": "Sp(4) random"},
            )
        )
    fired_set = set(layer.meas_qubits)
    # Iterate candidates (fall back to fired-only for layers built without a
    # candidate set, e.g. warmup or hand-made layers).
    candidates = layer.meas_candidates if layer.meas_candidates else layer.meas_qubits
    for q in candidates:
        fired = int(q) in fired_set
        ops.append(
            Op(
                kind="measure",
                qubits=(int(q),),
                label="MZ" if fired else "MZ?",
                random=True,  # which qubits get measured is a random draw
                detail={"basis": "Z", "fired": fired, "selection": "stochastic"},
            )
        )
    return ops


def _summary(config: CircuitConfig, sample_seed: int) -> dict[str, Any]:
    """Build the human-readable config summary for a trace."""
    cfg = config
    n = cfg.n
    if cfg.picture is Picture.PURE_STATE:
        layout = f"{cfg.total_qubits()} qubits (all system)"
    elif cfg.picture is Picture.PURIFICATION:
        layout = f"{cfg.total_qubits()} qubits: system 0–{n - 1}, reference {n}–{2 * n - 1}"
    else:  # SINGLE_REF
        layout = f"{cfg.total_qubits()} qubits: system 0–{n - 1}, reference [{n}]"

    if cfg.gating_mode == "brickwork":
        gating = f"brickwork / {cfg.matching_mode}"
    elif cfg.gating_mode == "all_edges":
        gating = f"all_edges ({len(cfg._graph.edges)} edges/layer)"
    else:  # random_edge | random_pool
        gating = f"{cfg.gating_mode} (gates_per_layer={cfg.resolved_gates_per_layer()})"

    ratio = cfg.expected_gate_to_meas_ratio()
    ratio_str = "∞ (no measurements)" if ratio == float("inf") else f"{ratio:.3g} : 1"

    summary: dict[str, Any] = {
        "picture": f"{cfg.picture}  ({layout})",
        "graph": cfg._graph.name,
        "gating": gating,
        "measurement": f"{cfg.measurement_mode}  p={cfg.p}",
        "depth": (
            f"exact override = {cfg.total_layers()} measured layers"
            if cfg.total_layers_override is not None
            else f"{cfg.depth_mode} ×{cfg.depth_factor} = {cfg.total_layers()} measured layers"
        ),
        "exp. gate:meas": ratio_str,
        "sample_seed": str(sample_seed),
    }
    if cfg.warmup_layers:
        summary["warmup"] = f"{cfg.warmup_layers} gate-only layers"
    scr = cfg.scramble_qubits()
    if scr is not None:
        held = " (reference partner held out)" if len(scr) < cfg.n else ""
        qs_word = "qubit" if len(scr) == 1 else "qubits"
        summary["scramble"] = f"global random Clifford on {len(scr)} system {qs_word}{held}"
    return summary


def trace_circuit(
    config: CircuitConfig,
    *,
    sample_seed: int | None = None,
    max_layers: int = 20,
    include_warmup: bool = True,
) -> CircuitTrace:
    """Build a :class:`CircuitTrace` for ``config`` without running anything.

    Parameters
    ----------
    config : CircuitConfig
        The construction to inspect.
    sample_seed : int or None
        Per-sample seed. ``None`` (default) picks a fresh random realization on
        every call, so re-running the inspector shows different draws of the
        random Cliffords and measured qubits (the chosen seed is reported in the
        trace summary so a preview you like can be reproduced). Pass an int for
        a fixed, reproducible realization.
    max_layers : int
        Maximum number of *measured* layers to include (default 20). Warmup
        layers are also capped at this count.
    include_warmup : bool
        Whether to include the gate-only warmup layers.

    Returns
    -------
    CircuitTrace
    """
    cfg = config
    if sample_seed is None:
        sample_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    builder = CircuitBuilder(cfg, sample_seed=sample_seed)
    stages: list[Stage] = []

    # Setup stage: the deterministic construction (Bell pairs).
    stages.append(
        Stage(label="setup", kind="setup", index=None, ops=list(setup_ops(cfg.picture, cfg.n)))
    )

    # Global scramble stage: one random Clifford over the qubits the runner
    # scrambles (cfg.scramble_qubits() is shared with the runner, so the diagram
    # matches what runs). Drawn as a single multi-qubit block, not decomposed.
    scr_qubits = cfg.scramble_qubits()
    if scr_qubits is not None and len(scr_qubits) >= 1:
        qs = tuple(int(q) for q in scr_qubits)
        scr_op = Op(
            kind="gate",
            qubits=qs,
            label="scramble",
            random=True,
            detail={
                "role": "scramble",
                "note": f"global random Clifford on {len(qs)} qubit{'s' if len(qs) != 1 else ''}",
            },
        )
        stages.append(Stage(label="scramble", kind="scramble", index=None, ops=[scr_op]))

    # Warmup stages (gate-only), capped at max_layers.
    if include_warmup and cfg.warmup_layers:
        for i, layer in enumerate(builder.warmup_layers_iter()):
            if i >= max_layers:
                break
            stages.append(Stage(label=f"warmup {i}", kind="warmup", index=i, ops=_layer_ops(layer)))

    # Measured layers (the main event), first max_layers.
    total_layers = cfg.total_layers()
    shown = 0
    for i, layer in enumerate(builder.layers()):
        if i >= max_layers:
            break
        stages.append(Stage(label=f"layer {i}", kind="layer", index=i, ops=_layer_ops(layer)))
        shown += 1

    # Qubit layout (single source of truth: the picture's _picture_layout).
    n_qubits, ref_arr, _, _ = _picture_layout(cfg.picture, cfg.n)

    return CircuitTrace(
        summary=_summary(cfg, sample_seed),
        stages=stages,
        total_layers=total_layers,
        shown_layers=shown,
        truncated=shown < total_layers,
        n_qubits=n_qubits,
        reference_qubits=tuple(int(q) for q in ref_arr),
    )


# ----------------------------------------------------------------------
# Text renderer (one view over the IR)
# ----------------------------------------------------------------------


def _render_qubits(op: Op) -> str:
    """Render an op's qubit tuple, with a control→target arrow for CX-likes."""
    if len(op.qubits) == 2 and op.label in DIRECTED_GATES:
        return f"q{op.qubits[0]}→q{op.qubits[1]}"
    if len(op.qubits) == 2:
        return f"q{op.qubits[0]},q{op.qubits[1]}"
    return ",".join(f"q{x}" for x in op.qubits)


def _render_stage(stage: Stage) -> list[str]:
    """Render one stage to text lines."""
    if stage.kind == "setup":
        head = "setup (deterministic)" if stage.ops else "setup (none: |0…0⟩)"
        lines = [head]
        for op in stage.ops:
            note = op.detail.get("note")
            line = f"  {op.label:<5} {_render_qubits(op)}"
            if note:
                line += f"   # {note}"
            lines.append(line)
        return lines

    if stage.kind == "scramble":
        op = stage.ops[0]
        qs = op.qubits
        span = f"q{qs[0]}..q{qs[-1]}" if len(qs) > 1 else f"q{qs[0]}"
        return [
            "scramble",
            f"  global random Clifford on {span}  [{len(qs)} qubit{'s' if len(qs) != 1 else ''}]",
        ]

    # warmup / layer (or any future op-bearing stage)
    suffix = " (gate-only pre-scramble)" if stage.kind == "warmup" else ""
    lines = [f"{stage.label}{suffix}"]
    gates = [op for op in stage.ops if op.kind == "gate"]
    meas = [op for op in stage.ops if op.kind == "measure"]
    other = [op for op in stage.ops if op.kind not in ("gate", "measure")]

    if gates:
        rendered = "   ".join(f"{op.label} ({_render_qubits(op)})" for op in gates)
        lines.append(f"  2q gates [random Sp(4)]:  {rendered}")
    if meas:
        fired_qs = " ".join(f"q{op.qubits[0]}" for op in meas if op.detail.get("fired", True))
        cand_qs = " ".join(f"q{op.qubits[0]}" for op in meas)
        lines.append(f"  measure Z  fired [{fired_qs or 'none'}]   candidates [{cand_qs}]")
    elif stage.kind == "layer":
        lines.append("  measure Z  (no candidates this layer)")
    for op in other:  # graceful fallback for unknown op kinds (extensibility)
        lines.append(f"  {op.kind}: {op.label} {_render_qubits(op)}")
    return lines


def render_text(trace: CircuitTrace) -> str:
    """Render a :class:`CircuitTrace` as a readable plain-text block."""
    lines: list[str] = []
    lines.append(
        f"Circuit inspection: first {trace.shown_layers} of {trace.total_layers} measured layers"
    )
    lines.append("=" * 70)
    width = max(len(k) for k in trace.summary)
    for k, v in trace.summary.items():
        lines.append(f"  {k:<{width}} : {v}")
    lines.append("")
    for stage in trace.stages:
        lines.extend(_render_stage(stage))
        lines.append("")
    if trace.truncated:
        lines.append(
            f"… {trace.total_layers - trace.shown_layers} more measured layers not shown "
            f"(raise max_layers to see them)."
        )
    return "\n".join(lines).rstrip() + "\n"


def inspect_circuit(
    config: CircuitConfig,
    *,
    sample_seed: int | None = None,
    max_layers: int = 20,
    include_warmup: bool = True,
) -> str:
    """Trace ``config`` and render it to text (the one-call convenience).

    ``sample_seed=None`` (default) shows a fresh random realization each call;
    pass an int to fix it. The chosen seed is reported in the summary.
    """
    trace = trace_circuit(
        config, sample_seed=sample_seed, max_layers=max_layers, include_warmup=include_warmup
    )
    return render_text(trace)


__all__ = [
    "Op",
    "Stage",
    "CircuitTrace",
    "trace_circuit",
    "render_text",
    "inspect_circuit",
]
