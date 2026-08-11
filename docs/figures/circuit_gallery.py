#!/usr/bin/env python
"""Render the README circuit figure and one quantikz2 diagram per construction mode.

Every gating mode, matching (1-factorization) mode, and measurement mode gets an
example. Per the saturation argument, the brickwork / matching modes use a
*uniform* measurement candidate set (``bernoulli`` / ``uniform_count``): when a
gate touches every qubit each layer, the ``gated`` candidate set is identical to
measuring everything, so ``gated`` is shown with a sparse gating mode
(``random_edge``) instead.

System qubits are black wires, the reference register is red. Output goes to
``docs/figures/circuit_diagram.png`` (the README figure) and
``docs/figures/gallery/<mode>.png``.

Needs a LaTeX install (pdflatex + quantikz) and a PDF->PNG tool (pdftoppm / sips
/ ImageMagick). Run from the repo root:  python docs/figures/circuit_gallery.py
"""

from __future__ import annotations

from pathlib import Path

from sparsegf2.circuits import CircuitConfig, save_circuit

HERE = Path(__file__).resolve().parent
GALLERY = HERE / "gallery"


def cfg(**kw) -> CircuitConfig:
    base = dict(n=6, picture="purification", p=0.4, depth_factor=2)
    base.update(kw)
    return CircuitConfig(**base)


# (filename, caption-ish label, config)
MODES = [
    # --- gating modes ---
    (
        "gating_brickwork",
        cfg(graph_spec="cycle", gating_mode="brickwork", measurement_mode="bernoulli"),
    ),
    (
        "gating_random_edge",
        cfg(graph_spec="complete", gating_mode="random_edge", measurement_mode="gated", p=0.5),
    ),
    (
        "gating_random_pool",
        cfg(graph_spec="complete", gating_mode="random_pool", measurement_mode="bernoulli"),
    ),
    (
        "gating_all_edges",
        cfg(graph_spec="cycle", gating_mode="all_edges", measurement_mode="bernoulli"),
    ),
    # --- matching / 1-factorization modes (brickwork + uniform measure) ---
    (
        "matching_round_robin",
        cfg(graph_spec="cycle", matching_mode="round_robin", measurement_mode="bernoulli"),
    ),
    (
        "matching_palette",
        cfg(graph_spec="cycle", matching_mode="palette", measurement_mode="bernoulli"),
    ),
    (
        "matching_fresh",
        cfg(graph_spec="cycle", matching_mode="fresh", measurement_mode="bernoulli"),
    ),
    # --- measurement modes ---
    (
        "measure_bernoulli",
        cfg(graph_spec="cycle", gating_mode="brickwork", measurement_mode="bernoulli"),
    ),
    (
        "measure_uniform_count",
        cfg(
            graph_spec="cycle",
            gating_mode="brickwork",
            measurement_mode="uniform_count",
            meas_count=2,
        ),
    ),
    (
        "measure_gated",
        cfg(graph_spec="complete", gating_mode="random_edge", measurement_mode="gated", p=0.5),
    ),
    (
        "measure_random_pair",
        cfg(
            graph_spec="complete", gating_mode="random_edge", measurement_mode="random_pair", p=0.5
        ),
    ),
]


def main() -> None:
    GALLERY.mkdir(parents=True, exist_ok=True)
    # Main README figure: a small purification brickwork circuit (red reference register).
    main_cfg = CircuitConfig(graph_spec="cycle", n=4, picture="purification", p=0.3, depth_factor=2)
    print(
        "wrote",
        save_circuit(main_cfg, HERE / "circuit_diagram.png", max_layers=4, dpi=170, sample_seed=0),
    )
    for name, c in MODES:
        print(
            "wrote", save_circuit(c, GALLERY / f"{name}.png", max_layers=5, dpi=200, sample_seed=1)
        )


if __name__ == "__main__":
    main()
