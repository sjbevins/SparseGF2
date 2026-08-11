"""Circuit drawer: a quantikz2 (LaTeX/TikZ) renderer over the trace IR.

This turns a :class:`~sparsegf2.circuits.CircuitConfig` or
:class:`~sparsegf2.circuits.CircuitTrace` into a publication-quality circuit
diagram using the **quantikz2** TikZ library (Kay, *Tutorial on the Quantikz
Package*, `arXiv:1809.03842 <https://arxiv.org/abs/1809.03842>`_).

It is just another renderer over the same IR as the text inspector, so it works
for **every** construction automatically (any picture, graph, gating / matching /
measurement mode):

* :func:`circuit_to_quantikz` returns the LaTeX source (a standalone document by
  default): embed it in a paper, or compile it yourself.
* :func:`save_circuit` compiles it with ``pdflatex`` and writes a ``.pdf`` /
  ``.png`` / ``.tex``.
* :func:`draw_circuit` is the notebook entry point: it compiles to PNG and
  returns an inline image (falling back to the LaTeX string off-notebook).

Wire color follows the physics: **system qubits are black, reference qubits
(purification / single_ref) are red**, so the Bell-paired reference register is
unmistakable. Deterministic construction gates (Bell pairs) and random Sp(4)
Cliffords are boxes; measurements are meters.

Compiling needs a LaTeX install with the ``quantikz`` package (TeX Live ships it)
on ``PATH``; PNG output additionally needs one of ``pdftoppm`` (Poppler/TeX
Live), ``sips`` (macOS), or ImageMagick. No Python build dependency is added; if
LaTeX is absent, :func:`circuit_to_quantikz` still returns the source.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from sparsegf2.circuits.config import CircuitConfig
from sparsegf2.circuits.inspector import CircuitTrace, Stage, trace_circuit
from sparsegf2.circuits.ops import DIRECTED_GATES, Op
from sparsegf2.errors import InvalidArgumentError

_REF_COLOR = "red"  # reference-qubit wires; system wires stay the default black


def _resolve_trace(source, *, sample_seed, max_layers, include_warmup) -> CircuitTrace:
    """Accept a CircuitTrace (used as-is) or a CircuitConfig (traced)."""
    if isinstance(source, CircuitTrace):
        return source
    if isinstance(source, CircuitConfig):
        return trace_circuit(
            source,
            sample_seed=sample_seed,
            max_layers=max_layers,
            include_warmup=include_warmup,
        )
    raise TypeError(
        f"draw_circuit expects a CircuitConfig or CircuitTrace; got {type(source).__name__}"
    )


def _pack_stage(stage: Stage) -> tuple[list[tuple[Op, int]], int]:
    """Assign each op a local column via greedy interval coloring.

    Each gate occupies the wire interval ``[min(qubits), max(qubits)]``; two ops
    share a column only if their intervals are disjoint, so no gate box is ever
    crossed by another gate in the same column. Measurements pack into one column
    after the gates. Returns ``(placements, n_cols)``.
    """
    gates = [o for o in stage.ops if o.kind == "gate"]
    meas = [o for o in stage.ops if o.kind == "measure"]
    other = [o for o in stage.ops if o.kind not in ("gate", "measure")]

    columns: list[list[tuple[int, int]]] = []
    placements: list[tuple[Op, int]] = []
    for op in gates:
        lo, hi = min(op.qubits), max(op.qubits)
        for c, intervals in enumerate(columns):
            if all(hi < a or lo > b for a, b in intervals):
                intervals.append((lo, hi))
                placements.append((op, c))
                break
        else:
            columns.append([(lo, hi)])
            placements.append((op, len(columns) - 1))
    n_gate_cols = len(columns)

    if meas:
        for op in meas:
            placements.append((op, n_gate_cols))
        n_cols = n_gate_cols + 1
    else:
        n_cols = n_gate_cols

    for k, op in enumerate(other):
        placements.append((op, n_cols + k))
    n_cols += len(other)

    return placements, n_cols


# ----------------------------------------------------------------------
# quantikz2 source generation
# ----------------------------------------------------------------------


def _tex_gate_label(label: str) -> str:
    r"""Render an op label for a ``\gate{...}`` (math mode).

    ``C[3]`` becomes ``C_{3}``; a multi-letter word like ``scramble`` is set
    upright with ``\mathrm`` so it does not render as italic concatenated
    symbols; short symbols (``H``, ``S``, ``CX``) pass through.
    """
    m = re.fullmatch(r"C\[(\d+)\]", label)
    if m:
        return f"C_{{{m.group(1)}}}"
    if len(label) > 2 and label.isalpha():
        return rf"\mathrm{{{label}}}"
    return label


def _qubit_label(q: int, is_ref: bool) -> str:
    return rf"q_{{{q}}}\,\mathrm{{(ref)}}" if is_ref else f"q_{{{q}}}"


def _stage_label(stage: Stage) -> str:
    """Short slice label for a trace stage: the measured layers become t_1, t_2…."""
    if stage.kind == "layer":
        return f"$t_{{{(stage.index or 0) + 1}}}$"
    if stage.kind == "warmup":
        return "warmup"
    if stage.kind == "setup":
        return "setup"
    return stage.label


def circuit_to_quantikz(
    source,
    *,
    sample_seed: int | None = None,
    max_layers: int = 12,
    include_warmup: bool = True,
    standalone: bool = True,
    timesteps: bool = True,
    row_sep: str = "0.25cm",
    column_sep: str = "0.32cm",
) -> str:
    r"""Build the quantikz2 LaTeX for a circuit and return it as a string.

    Parameters
    ----------
    source
        A :class:`CircuitConfig` (traced with the kwargs below) or a
        :class:`CircuitTrace` (drawn as-is).
    sample_seed, max_layers, include_warmup
        Forwarded to :func:`~sparsegf2.circuits.trace_circuit` for a config.
    standalone
        When true (default) wrap the ``quantikz`` environment in a complete
        ``standalone`` document (compilable on its own); otherwise return just
        the ``\begin{quantikz}...\end{quantikz}`` body for embedding.
    timesteps
        When true (default) draw a labeled dashed ``\slice`` at each stage
        boundary, so the individual timesteps read off the diagram: ``setup``,
        ``warmup``, then ``t_1, t_2, …`` for the measured layers.
    row_sep, column_sep
        Spacing passed to the ``quantikz`` environment.

    Reference-qubit wires are drawn red, system wires black.
    """
    trace = _resolve_trace(
        source, sample_seed=sample_seed, max_layers=max_layers, include_warmup=include_warmup
    )
    n = trace.n_qubits
    ref = set(trace.reference_qubits)

    # --- assign every op a global column across the stages, and remember each
    #     stage's last column so we can slice the timesteps apart ---
    placements: list[tuple[Op, int]] = []
    gcol = 0
    stage_bounds: list[tuple[str, int]] = []  # (label, last global column)
    for stage in trace.stages:
        place, ncols = _pack_stage(stage)
        for op, c in place:
            placements.append((op, gcol + c))
        if ncols > 0:
            stage_bounds.append((_stage_label(stage), gcol + ncols - 1))
        gcol += max(ncols, 0)
    total_cols = max(gcol, 1)
    # A slice draws between column c and c+1, so it cannot sit on the last
    # column; this also drops the redundant trailing slice at the right edge.
    slice_at = (
        {col: label for label, col in stage_bounds if col < total_cols - 1} if timesteps else {}
    )

    # --- grid[row][col] of cell content; default "" (an auto / drawn wire) ---
    grid = [["" for _ in range(total_cols)] for _ in range(n)]
    for op, c in placements:
        if op.kind == "measure":
            if op.detail.get("fired", True):  # only the measurements that fired
                grid[op.qubits[0]][c] = r"\meter{}"
        elif op.kind == "gate":
            if len(op.qubits) == 1:
                grid[op.qubits[0]][c] = rf"\gate{{{_tex_gate_label(op.label)}}}"
            elif len(op.qubits) > 2:  # multi-qubit block (the global scramble)
                lo, hi = min(op.qubits), max(op.qubits)
                grid[lo][c] = rf"\gate[{hi - lo + 1}]{{{_tex_gate_label(op.label)}}}"
            elif op.label in DIRECTED_GATES:  # CX: control -> target
                ctrl, tgt = op.qubits
                grid[ctrl][c] = rf"\ctrl{{{tgt - ctrl}}}"
                grid[tgt][c] = r"\targ{}"
            else:  # generic 2q Clifford
                a, b = op.qubits
                top, bot = min(a, b), max(a, b)
                lbl = _tex_gate_label(op.label)
                if bot - top == 1:  # adjacent wires: one box spanning both
                    grid[top][c] = rf"\gate[2]{{{lbl}}}"
                else:  # long-range: a box on each endpoint joined by a vertical wire
                    grid[top][c] = rf"\gate{{{lbl}}}\vqw{{{bot - top}}}"
                    grid[bot][c] = rf"\gate{{{lbl}}}"
        else:  # unknown kind -> a labeled box (extensibility fallback)
            grid[op.qubits[0]][c] = rf"\gate{{{_tex_gate_label(op.label)}}}"

    # --- emit rows. Reference rows have auto-wire off (type n) and draw a red
    #     wire explicitly in every cell; system rows use the default black wire. ---
    wire_types = ",".join("n" if q in ref else "q" for q in range(n))
    lines: list[str] = []
    for q in range(n):
        is_ref = q in ref
        cells = []
        for col in range(total_cols):
            content = grid[q][col]
            if q == 0 and col in slice_at:  # one slice per boundary, on the top wire
                content = f"{content}\\slice{{{slice_at[col]}}}"
            if is_ref:  # red wire leftward in this cell, after any gate content
                content = f"{content}\\wire[l][1][{_REF_COLOR}]{{q}}"
            cells.append(content)
        stick = (
            rf"\lstick[label style={{{_REF_COLOR}}}]{{${_qubit_label(q, True)}$}}"
            if is_ref
            else rf"\lstick{{${_qubit_label(q, False)}$}}"
        )
        row = stick + " & " + " & ".join(cells)
        lines.append(row)
    body = " \\\\\n  ".join(lines)
    env = f"\\begin{{quantikz}}[row sep={{{row_sep}}}, column sep={{{column_sep}}}, wire types={{{wire_types}}}]\n  {body}\n\\end{{quantikz}}"

    if not standalone:
        return env
    return (
        "\\documentclass[border=4pt]{standalone}\n"
        "\\usepackage{tikz}\n"
        "\\usetikzlibrary{quantikz2}\n"
        "\\begin{document}\n"
        f"{env}\n"
        "\\end{document}\n"
    )


# ----------------------------------------------------------------------
# Compilation (pdflatex -> pdf, then optionally -> png)
# ----------------------------------------------------------------------

_TEXBIN = "/Library/TeX/texbin"


def _which(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    cand = os.path.join(_TEXBIN, name)
    return cand if os.path.exists(cand) else None


def _compile_pdf(latex: str, workdir: Path) -> Path:
    """Compile ``latex`` to a PDF in ``workdir`` and return the PDF path."""
    pdflatex = _which("pdflatex")
    if pdflatex is None:
        raise InvalidArgumentError(
            "compiling circuit diagrams needs a LaTeX install (pdflatex) with the "
            "quantikz package on PATH; TeX Live provides both. Use "
            "circuit_to_quantikz(...) to get the LaTeX source and compile it elsewhere."
        )
    tex = workdir / "circuit.tex"
    tex.write_text(latex, encoding="utf-8")
    proc = subprocess.run(
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex.name],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    pdf = workdir / "circuit.pdf"
    if proc.returncode != 0 or not pdf.exists():
        log = (
            (workdir / "circuit.log").read_text(encoding="utf-8")
            if (workdir / "circuit.log").exists()
            else ""
        )
        errs = [ln for ln in log.splitlines() if ln.startswith("!")][:5]
        raise InvalidArgumentError(
            "pdflatex failed to compile the diagram"
            + (" (is the quantikz package installed?):\n  " + "\n  ".join(errs) if errs else "")
        )
    return pdf


def _pdf_to_png(pdf: Path, png: Path, dpi: int) -> None:
    """Convert ``pdf`` to ``png`` with the first available rasteriser."""
    ppm = _which("pdftoppm")
    if ppm:  # best quality; -singlefile drops the page-number suffix
        subprocess.run(
            [ppm, "-png", "-r", str(dpi), "-singlefile", str(pdf), str(png.with_suffix(""))],
            check=True,
            capture_output=True,
        )
        return
    if _which("sips"):  # macOS built-in
        subprocess.run(
            ["sips", "-s", "format", "png", str(pdf), "--out", str(png)],
            check=True,
            capture_output=True,
        )
        return
    magick = _which("magick") or _which("convert")
    if magick:
        subprocess.run(
            [magick, "-density", str(dpi), str(pdf), "-quality", "92", str(png)],
            check=True,
            capture_output=True,
        )
        return
    raise InvalidArgumentError(
        "no PDF->PNG rasteriser found (need pdftoppm, sips, or ImageMagick); "
        "save the diagram as .pdf instead, or install one"
    )


def save_circuit(
    source,
    path: str | Path,
    *,
    fmt: str | None = None,
    dpi: int = 200,
    **trace_kwargs,
) -> str:
    """Render ``source`` with quantikz2 and write it to ``path``. Returns ``path``.

    ``fmt`` (or ``path``'s extension) selects the output: ``"tex"`` writes the
    LaTeX source; ``"pdf"`` compiles with ``pdflatex``; ``"png"`` compiles then
    rasterizes at ``dpi``. ``trace_kwargs`` are forwarded to
    :func:`circuit_to_quantikz` (``sample_seed``, ``max_layers``, …).
    """
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".") or "pdf").lower()
    if fmt not in ("tex", "pdf", "png"):
        raise InvalidArgumentError(f"fmt must be 'tex', 'pdf', or 'png'; got {fmt!r}")
    latex = circuit_to_quantikz(source, standalone=True, **trace_kwargs)
    if fmt == "tex":
        path.write_text(latex, encoding="utf-8")
        return str(path)
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        pdf = _compile_pdf(latex, work)
        if fmt == "pdf":
            shutil.copyfile(pdf, path)
        else:
            _pdf_to_png(pdf, path, dpi)
    return str(path)


def draw_circuit(source, *, dpi: int = 170, **trace_kwargs):
    """Draw a circuit with quantikz2 for inline display (the notebook entry point).

    Compiles to PNG and returns an :class:`IPython.display.Image` so a notebook
    renders it inline. Off-notebook (no IPython) or without LaTeX it returns the
    quantikz LaTeX string instead; use :func:`save_circuit` to write a file.
    ``trace_kwargs`` (``sample_seed``, ``max_layers``, ``include_warmup``) are
    forwarded to :func:`circuit_to_quantikz`.
    """
    latex = circuit_to_quantikz(source, standalone=True, **trace_kwargs)
    if _which("pdflatex") is None:
        return latex
    try:
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            pdf = _compile_pdf(latex, work)
            png = work / "circuit.png"
            _pdf_to_png(pdf, png, dpi)
            data = png.read_bytes()
    except InvalidArgumentError:
        return latex
    try:
        from IPython.display import Image
    except ImportError:
        return latex
    return Image(data=data, format="png")


__all__ = ["circuit_to_quantikz", "draw_circuit", "save_circuit"]
