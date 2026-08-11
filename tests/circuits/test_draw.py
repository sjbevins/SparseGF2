"""Tests for the quantikz2 circuit drawer.

Structural (no pixel comparison): every construction must emit valid quantikz2
LaTeX with one wire per qubit and red wires for the reference register, the
column packer must never overlap two ops in a column, and compilation is
exercised when a LaTeX install is present. Visual quality is checked separately.
"""

from __future__ import annotations

import inspect
import subprocess

import pytest

from sparsegf2.circuits import (
    CircuitConfig,
    circuit_to_quantikz,
    draw_circuit,
    save_circuit,
    trace_circuit,
)
from sparsegf2.circuits.draw import _pack_stage, _which
from sparsegf2.errors import InvalidArgumentError


def _latex_dependencies_available():
    """Whether the external compiler and document packages used below exist."""
    if _which("pdflatex") is None:
        return False
    kpsewhich = _which("kpsewhich")
    if kpsewhich is None:
        return False
    try:
        return all(
            subprocess.run(
                [kpsewhich, package],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            ).returncode
            == 0
            for package in ("standalone.cls", "tikzlibraryquantikz2.code.tex")
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


_HAS_LATEX = _latex_dependencies_available()


# ----------------------------------------------------------------------
# the column packer never overlaps
# ----------------------------------------------------------------------


def _intervals_in_column(placements):
    cols: dict[int, list[tuple[int, int]]] = {}
    for op, c in placements:
        cols.setdefault(c, []).append((min(op.qubits), max(op.qubits)))
    return cols


@pytest.mark.parametrize(
    "kw",
    [
        dict(graph_spec="cycle", n=8, p=0.2),
        dict(graph_spec="complete", n=8, p=0.2),  # long-range matchings
        dict(graph_spec="complete", n=8, gating_mode="random_edge", gates_per_layer=4, p=0.3),
        dict(graph_spec="cycle", n=8, picture="purification", p=0.16),
        dict(graph_spec="cycle", n=8, picture="single_ref", p=0.16),
    ],
)
def test_packing_has_no_intra_column_overlap(kw):
    tr = trace_circuit(CircuitConfig(**kw), max_layers=6)
    for stage in tr.stages:
        placements, _ = _pack_stage(stage)
        for _c, intervals in _intervals_in_column(placements).items():
            for i in range(len(intervals)):
                a_lo, a_hi = intervals[i]
                for j in range(i + 1, len(intervals)):
                    b_lo, b_hi = intervals[j]
                    disjoint = a_hi < b_lo or b_hi < a_lo
                    points = a_lo == a_hi and b_lo == b_hi  # both measurements
                    assert disjoint or points, (kw, intervals)


# ----------------------------------------------------------------------
# valid quantikz2, one wire per qubit, red reference wires
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "picture,expected_wires",
    [("pure_state", 8), ("purification", 16), ("single_ref", 9)],
)
def test_quantikz_has_one_wire_per_qubit(picture, expected_wires):
    cfg = CircuitConfig(graph_spec="cycle", n=8, picture=picture, p=0.16, depth_factor=2)
    tex = circuit_to_quantikz(cfg, max_layers=4)
    assert r"\begin{quantikz}" in tex and r"\end{quantikz}" in tex
    assert tex.count(r"\lstick") == expected_wires


def test_reference_wires_are_red_system_wires_are_not():
    pur = circuit_to_quantikz(
        CircuitConfig(graph_spec="cycle", n=6, picture="purification", p=0.2, depth_factor=2),
        max_layers=3,
    )
    assert r"\wire[l][1][red]{q}" in pur  # the reference register is drawn red
    assert "label style={red}" in pur
    # pure_state has no reference register, so nothing is red
    pure = circuit_to_quantikz(
        CircuitConfig(graph_spec="cycle", n=6, picture="pure_state", p=0.2, depth_factor=2),
        max_layers=3,
    )
    assert "red" not in pure


@pytest.mark.parametrize("gating", ["brickwork", "random_edge", "random_pool", "all_edges"])
@pytest.mark.parametrize("measurement", ["bernoulli", "gated", "random_pair", "uniform_count"])
def test_all_modes_emit_quantikz(gating, measurement):
    kw = dict(
        graph_spec="cycle",
        n=8,
        gating_mode=gating,
        measurement_mode=measurement,
        p=0.3,
        depth_factor=1,
    )
    if measurement == "uniform_count":
        kw["meas_count"] = 2
    try:
        cfg = CircuitConfig(**kw)
    except InvalidArgumentError:
        pytest.skip(f"{gating}+{measurement} is not a valid combination")
    assert r"\begin{quantikz}" in circuit_to_quantikz(cfg, max_layers=4)


def test_timesteps_add_labeled_slices():
    cfg = CircuitConfig(graph_spec="cycle", n=4, picture="purification", p=0.3, depth_factor=2)
    tex = circuit_to_quantikz(cfg, max_layers=3)
    assert r"\slice{setup}" in tex  # the setup boundary is marked
    assert r"\slice{$t_{1}$}" in tex  # measured layers become labeled timesteps
    # and the feature switches off
    assert r"\slice" not in circuit_to_quantikz(cfg, max_layers=3, timesteps=False)


def test_accepts_trace_or_config():
    cfg = CircuitConfig(graph_spec="cycle", n=8, p=0.2, depth_factor=2)
    tr = trace_circuit(cfg, max_layers=4)
    assert r"\begin{quantikz}" in circuit_to_quantikz(tr)
    assert r"\begin{quantikz}" in circuit_to_quantikz(cfg, max_layers=4)


def test_rejects_bad_source():
    with pytest.raises(TypeError):
        circuit_to_quantikz("not a config")


# ----------------------------------------------------------------------
# saving: .tex never needs LaTeX; .pdf compiles when LaTeX is present
# ----------------------------------------------------------------------


def test_save_tex_needs_no_latex(tmp_path):
    cfg = CircuitConfig(graph_spec="cycle", n=6, picture="single_ref", p=0.16, depth_factor=2)
    out = save_circuit(cfg, tmp_path / "c.tex", max_layers=4)
    assert out == str(tmp_path / "c.tex")
    assert r"\begin{quantikz}" in (tmp_path / "c.tex").read_text(encoding="utf-8")


def test_save_bad_format_rejected(tmp_path):
    cfg = CircuitConfig(graph_spec="cycle", n=4, p=0.2, depth_factor=2)
    with pytest.raises(InvalidArgumentError):
        save_circuit(cfg, tmp_path / "c.svg", fmt="svg")


@pytest.mark.skipif(not _HAS_LATEX, reason="needs a LaTeX install (pdflatex + quantikz)")
def test_save_pdf_compiles(tmp_path):
    cfg = CircuitConfig(graph_spec="cycle", n=4, picture="purification", p=0.2, depth_factor=2)
    out = save_circuit(cfg, tmp_path / "c.pdf", max_layers=3)
    assert (tmp_path / "c.pdf").exists() and (tmp_path / "c.pdf").stat().st_size > 0
    assert out == str(tmp_path / "c.pdf")


def test_draw_circuit_falls_back_to_latex_without_compiler(monkeypatch):
    # With no LaTeX on PATH, draw_circuit hands back the quantikz source string.
    import sparsegf2.circuits.draw as d

    monkeypatch.setattr(d, "_which", lambda name: None)
    out = draw_circuit(CircuitConfig(graph_spec="cycle", n=6, p=0.2, depth_factor=2), max_layers=3)
    assert isinstance(out, str) and r"\begin{quantikz}" in out


def test_drawer_is_matplotlib_free():
    # The quantikz drawer must not depend on matplotlib at all.
    import sparsegf2.circuits.draw as drawmod

    assert "import matplotlib" not in inspect.getsource(drawmod)


def test_quantikz_renders_scramble_block():
    """The global scramble renders as a multi-qubit \\gate[k]{scramble} block over
    exactly the scrambled qubits, narrowing by one when the reference partner is
    held out."""
    from sparsegf2.circuits import CircuitConfig, circuit_to_quantikz

    # single_ref, n=16, scramble on all 16 system qubits -> a 16-wire block
    cfg = CircuitConfig(
        graph_spec="cycle", n=16, picture="single_ref", scramble=True, depth_factor=2
    )
    tex = circuit_to_quantikz(cfg, sample_seed=0, max_layers=2, standalone=False)
    assert r"\gate[16]{\mathrm{scramble}}" in tex

    # held-out partner -> 15-wire block (q15 left out)
    cfg_held = CircuitConfig(
        graph_spec="cycle",
        n=16,
        picture="single_ref",
        scramble=True,
        scramble_entangled_qubit=False,
        depth_factor=2,
    )
    assert r"\gate[15]{\mathrm{scramble}}" in circuit_to_quantikz(
        cfg_held, sample_seed=0, max_layers=2, standalone=False
    )
