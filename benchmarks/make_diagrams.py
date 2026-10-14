"""Render the explanatory diagrams of docs/HOW_IT_WORKS.md (light and dark PNGs).

Three figures, each written as ``<name>-light.png`` and ``<name>-dark.png``:

* ``tableau``: how a state is stored. The bits are a genuine state: trajectory 0
  of a small monitored circuit with one reference qubit, replayed with the same
  compiled kernels the ensemble engine uses (and checked against the engine's
  own ``code_dimension`` and against ``Simulator`` entropies).
* ``engine``: the ensemble engine, from a ``MonitoredCircuit`` spec to an
  ``EnsembleResult`` and the analysis.
* ``erasure-analysis``: from the code a circuit generated to P_C(m), Phi(e) and
  e_delta. The curves come from a small run with fixed seeds (about a second).

The look (colours, fonts, ``save``) is shared with ``make_figures.py``.

Usage (from the repository root)::

    python benchmarks/make_diagrams.py               # writes docs/figures/
    python benchmarks/make_diagrams.py --outdir /tmp/diagrams
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_figures as mf  # noqa: E402
from make_figures import THEMES, caption, legend, style  # noqa: E402

from sparsegf2 import analysis, graphs, mipt  # noqa: E402
from sparsegf2._kernel import tableau as _tk  # noqa: E402
from sparsegf2._kernel.tables import (  # noqa: E402
    ANF1,
    ANF2,
    IDENTITY_SYM1,
    IDENTITY_SYM2,
    LIN1,
    LIN2,
)
from sparsegf2.simulator import Simulator  # noqa: E402

MONO = "DejaVu Sans Mono"
WIDTH = 8.4  # inches, like the other README figures
FS = 8.5  # body text size (pt)

# Semantic colour slots, the same as in the circuit figure of make_figures.py:
GATE, MEAS, REF = 0, 1, 2  # blue = gates / system, orange = measurement, green = reference


# ---------------------------------------------------------------------------
# drawing helpers (inch coordinates)
# ---------------------------------------------------------------------------


def inch_axes(fig, rect=(0.0, 0.0, 1.0, 1.0)):
    """Axes without decoration whose data units are inches of the figure."""
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    w, h = fig.get_size_inches()
    ax.set_xlim(0, rect[2] * w)
    ax.set_ylim(0, rect[3] * h)
    return ax


def rbox(ax, x, y, w, h, theme, *, edge=None, fill=None, lw=1.2, z=1, radius=0.07):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=theme["box"] if fill is None else fill,
        edgecolor=theme["axis"] if edge is None else edge,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, theme, *, color=None, lw=1.3, style_="-|>", z=3, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style_,
            mutation_scale=11,
            color=theme["ink2"] if color is None else color,
            linewidth=lw,
            zorder=z,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def label(ax, x, y, s, theme, *, size=FS, color="ink2", weight="normal", mono=False, **kw):
    kw.setdefault("va", "top")
    kw.setdefault("ha", "left")
    kw.setdefault("linespacing", 1.4)
    return ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=theme[color],
        fontweight=weight,
        family=MONO if mono else None,
        zorder=5,
        **kw,
    )


def height(fig, artist) -> float:
    """Rendered height of a text artist, in inches."""
    return artist.get_window_extent(renderer=fig.canvas.get_renderer()).height / fig.dpi


def wrap(s: str, width: int) -> str:
    return "\n".join(textwrap.fill(p, width) for p in s.split("\n"))


# ---------------------------------------------------------------------------
# 1. tableau: a genuine state, replayed with the engine's kernels
# ---------------------------------------------------------------------------


def replay(circuit, n_sys: int):
    """Run an exported trajectory circuit exactly as the ensemble engine does.

    Phase-free tableau, gates through the same kernels, and every Z measurement
    through ``measure_1q_phase_free_tracked`` (the reference-row pivot rule).
    Returns the tableau, the reference-row mask, the gates of the last layer and
    the number of measurements.
    """
    n = n_sys + 1
    ws = (n + 63) // 64
    tab = np.zeros((2 * n + 1, ws), dtype=np.uint64)
    _tk.init_zero(tab, n, ws, False)
    ref_rows = np.zeros(ws, dtype=np.uint64)
    anti = np.zeros(ws, dtype=np.uint64)
    supp = np.zeros(max(n, ws), dtype=np.int64)
    h_index, cx_index = mipt._H_INDEX, mipt._CX_INDEX
    refs_done = False
    layer_gates: list[tuple[int, int]] = []
    last_gates: list[tuple[int, int]] = []
    measured = 0
    for ins in circuit.flattened():
        qs = [t.value for t in ins.targets]
        if ins.name == "H":
            for q in qs:
                _tk.apply_1q(tab, q, h_index, LIN1, ANF1, IDENTITY_SYM1, ws, False)
        elif ins.name == "CX":
            for a, b in zip(qs[::2], qs[1::2], strict=True):
                _tk.apply_2q(tab, a, b, cx_index, LIN2, ANF2, IDENTITY_SYM2, ws, False)
        elif ins.name == "CLIFFORD_2Q":
            for (a, b), g in zip(zip(qs[::2], qs[1::2], strict=True), ins.args, strict=True):
                _tk.apply_2q(tab, a, b, int(g), LIN2, ANF2, IDENTITY_SYM2, ws, False)
                layer_gates.append((a, b))
        elif ins.name == "M":
            for q in qs:
                _tk.measure_1q_phase_free_tracked(tab, n, ws, q, 2, anti, supp, ref_rows)
                measured += 1
        elif ins.name == "TICK":
            if not refs_done:  # gates never touch R, so its rows are fixed until a measurement
                for w in range(ws):
                    ref_rows[w] = tab[2 * n_sys, w] | tab[2 * n_sys + 1, w]
                refs_done = True
            if layer_gates:
                last_gates = layer_gates
            layer_gates = []
        else:
            raise ValueError(f"unexpected instruction {ins.name}")
    return tab, ref_rows, last_gates, measured


def bits_of(tab, n):
    """(x, z) as (rows, qubits) 0/1 arrays: generator r on qubit q."""
    x = np.zeros((n, n), dtype=np.uint8)
    z = np.zeros((n, n), dtype=np.uint8)
    for r in range(n):
        w, b = r >> 6, np.uint64(r & 63)
        for q in range(n):
            x[r, q] = (int(tab[2 * q, w]) >> int(b)) & 1
            z[r, q] = (int(tab[2 * q + 1, w]) >> int(b)) & 1
    return x, z


def gf2_rank(m: np.ndarray) -> int:
    m = m.copy() % 2
    rank = 0
    for c in range(m.shape[1]):
        piv = np.flatnonzero(m[rank:, c])
        if len(piv) == 0:
            continue
        p = rank + piv[0]
        m[[rank, p]] = m[[p, rank]]
        for r in np.flatnonzero(m[:, c]):
            if r != rank:
                m[r] ^= m[rank]
        rank += 1
        if rank == m.shape[0]:
            break
    return rank


def tableau_example():
    n_sys, layers, p = 6, 4, 0.25
    spec = mipt.MonitoredCircuit(
        num_system=n_sys,
        edges=graphs.cycle(n_sys),
        layers=layers,
        measurement_rate=p,
        schedule="matching",
        reference="single",
        checkpoints="final",
        probes=[mipt.code_dimension()],
    )
    region = [3, 4, 5]
    for seed in range(1000):
        circ = spec.circuit(0, seed=seed)
        tab, ref_rows, last_gates, measured = replay(circ, n_sys)
        n = n_sys + 1
        x, z = bits_of(tab, n)
        ref = [r for r in range(n) if (int(ref_rows[r >> 6]) >> (r & 63)) & 1]
        k = len(ref) - 1
        adjacent = [g for g in last_gates if abs(g[0] - g[1]) == 1 and max(g) < 3]
        cols = np.concatenate([x[:, region], z[:, region]], axis=1)
        s_a = gf2_rank(cols) - len(region)
        if k != 1 or measured < 3 or not adjacent or s_a != 1:
            continue
        # The engine itself must agree (same seed, trajectory 0) ...
        engine_k = spec.run(1, seed=seed).values["code_dimension"][0, -1]
        assert engine_k == k, (engine_k, k)
        # ... and an independent Simulator run of the same circuit (its own pivot rule).
        sim = Simulator(n, phases=False, seed=0)
        sim.run(circ)
        assert sim.entropy(region) == s_a and sim.entropy([n_sys]) == k
        a, b = sorted(adjacent[0])
        return {
            "x": x,
            "z": z,
            "ref": ref,
            "k": k,
            "n_sys": n_sys,
            "gate": (a, b),
            "region": region,
            "rank": gf2_rank(cols),
            "s_a": s_a,
            "seed": seed,
            "layers": layers,
            "p": p,
            "measured": measured,
            "tab": tab,
        }
    raise RuntimeError("no example state found")


def fig_tableau(theme, mode, ex):
    x, z, ref, n_sys = ex["x"], ex["z"], ex["ref"], ex["n_sys"]
    n = n_sys + 1
    H = 4.9
    fig = plt.figure(figsize=(WIDTH, H))
    ax = inch_axes(fig)
    ink2, muted = theme["ink2"], theme["muted"]
    blue, green = theme["series"][GATE], theme["series"][REF]

    cell, gap = 0.26, 0.05  # cell pitch and surface gap between cells (inches)
    pair_gap, ref_gap = 0.12, 0.22
    x0, y_top = 0.55, H - 1.05  # top-left of the grid

    def col_x(q, xz):  # left edge of the X (xz=0) or Z (xz=1) column of qubit q
        extra = ref_gap if q == n_sys else 0.0
        return x0 + q * (2 * cell + pair_gap) + xz * cell + extra

    def row_y(r):  # bottom edge of generator row r
        return y_top - (r + 1) * cell

    grid_right = col_x(n_sys, 1) + cell
    strings_x = grid_right + 0.25
    band_right = strings_x + 1.12

    # reference-row bands (behind everything)
    for r in ref:
        ax.add_patch(
            Rectangle(
                (x0 - 0.42, row_y(r) + 0.01),
                band_right - (x0 - 0.42),
                cell - 0.02,
                facecolor=green,
                alpha=0.16,
                linewidth=0,
                zorder=0,
            )
        )
    # cells
    for r in range(n):
        for q in range(n):
            for xz, bits in ((0, x), (1, z)):
                on = bits[r, q] == 1
                color = (green if q == n_sys else ink2) if on else theme["grid"]
                ax.add_patch(
                    Rectangle(
                        (col_x(q, xz) + gap / 2, row_y(r) + gap / 2),
                        cell - gap,
                        cell - gap,
                        facecolor=color,
                        linewidth=0,
                        zorder=2,
                    )
                )
        label(ax, x0 - 0.12, row_y(r) + cell / 2, f"g{r}", theme, size=8, ha="right", va="center")
        word = ""
        for q in range(n):
            word += "IXZY"[x[r, q] + 2 * z[r, q]] + (" | " if q == n_sys - 1 else "")
        label(ax, strings_x, row_y(r) + cell / 2, word, theme, size=8.5, mono=True, va="center")
    label(ax, strings_x, y_top + 0.04, "as Paulis", theme, size=7.5, color="muted", va="bottom")
    # column headers
    for q in range(n):
        name = "R" if q == n_sys else f"q{q}"
        cx = col_x(q, 0) + cell
        label(ax, cx, y_top + 0.24, name, theme, size=8.5, color="ink", ha="center", va="bottom")
        for xz, s in ((0, "X"), (1, "Z")):
            label(
                ax, col_x(q, xz) + cell / 2, y_top + 0.04, s, theme, size=7.5, color="muted",
                ha="center", va="bottom",
            )  # fmt: skip
    # gate box around the four columns of (a, b)
    a, b = ex["gate"]
    gx0, gx1 = col_x(a, 0) - 0.05, col_x(b, 1) + cell + 0.05
    rbox(
        ax, gx0, row_y(n - 1) - 0.06, gx1 - gx0, n * cell + 0.5, theme, edge=blue,
        fill="none", lw=1.6, z=3,
    )  # fmt: skip
    label(
        ax, (gx0 + gx1) / 2, row_y(n - 1) - 0.1, f"gate on (q{a}, q{b})", theme, size=8,
        ha="center", color="ink2",
    )  # fmt: skip
    # region A bracket
    reg = ex["region"]
    bx0, bx1 = col_x(reg[0], 0) + 0.02, col_x(reg[-1], 1) + cell - 0.02
    by = row_y(n - 1) - 0.12
    ax.plot([bx0, bx0, bx1, bx1], [by + 0.06, by, by, by + 0.06], color=ink2, lw=1.0, zorder=3)
    label(
        ax, (bx0 + bx1) / 2, by - 0.05,
        f"A = {{q{reg[0]}, q{reg[1]}, q{reg[2]}}}:  S(A) = {ex['rank']} − {len(reg)} = {ex['s_a']}",
        theme, size=8, ha="center",
    )  # fmt: skip
    # reference-row tag
    label(
        ax, band_right, row_y(n - 1) - 0.1, "green bands:\nreference rows", theme,
        size=8, ha="right", color="ink2", linespacing=1.25,
    )  # fmt: skip
    # key for the cells
    kx, ky = x0 - 0.42, H - 0.5
    for i, (color, text) in enumerate(
        ((ink2, "bit 1"), (theme["grid"], "bit 0"), (green, "bit 1 on the reference R"))
    ):
        xx = kx + i * 0.85
        ax.add_patch(Rectangle((xx, ky - 0.07), 0.14, 0.14, facecolor=color, linewidth=0))
        label(ax, xx + 0.2, ky, text, theme, size=8, va="center")

    # one column as it sits in memory: a uint64 word, bit r = generator r
    wy = 0.78
    wx0, wc = 0.55, 0.066
    colname = f"Z column of q{a}"
    bits = [z[r, a] for r in range(n)]
    for i in range(64):
        used = i < n
        color = ink2 if (used and bits[i]) else theme["grid"]
        ax.add_patch(
            Rectangle(
                (wx0 + i * wc + 0.008, wy),
                wc - 0.016,
                0.17,
                facecolor=color,
                alpha=1.0 if used else 0.55,
                linewidth=0,
                zorder=2,
            )
        )
    ax.plot([wx0, wx0 + n * wc], [wy - 0.06] * 2, color=blue, lw=1.4, zorder=3)
    label(ax, wx0, wy - 0.11, f"bits 0–{n - 1}: generators g0–g{n - 1}", theme, size=8)
    label(
        ax, wx0 + 64 * wc, wy - 0.11, f"bits {n}–63: padding (always 0)", theme, size=8,
        ha="right", color="muted",
    )  # fmt: skip
    label(
        ax, wx0, wy + 0.4, f"The {colname} in memory: one uint64 word", theme, size=8.5,
        color="ink", va="top",
    )  # fmt: skip
    # tie the word to its column
    arrow(
        ax, (col_x(a, 1) + cell / 2, by - 0.38), (col_x(a, 1) + cell / 2, wy + 0.47), theme,
        color=muted, lw=1.0,
    )  # fmt: skip

    # explanations
    tx = band_right + 0.3
    tw = 49
    blocks = [
        (
            "Storage",
            "One row per stabilizer generator (6 system qubits + 1 reference R = 7 rows); "
            "per qubit an X and a Z column, Y = both bits. Stored by columns: each column "
            "is a bitset over the generators, 64 generators per uint64 word "
            "(n = 256 takes 4 words).",
        ),
        (
            "Gates",
            f"A two-qubit Clifford on (q{a}, q{b}) rewrites only those 4 columns: each new "
            "column is an XOR of old ones, a whole word (64 generators) at a time.",
        ),
        (
            "Entanglement",
            "S(A) = rank of the 2|A| columns of A − |A|, one elimination over A's columns.",
        ),
        (
            "Code dimension",
            f"Reference rows are the generators that touch R (green). k = (number of "
            f"reference rows) − n_ref = {len(ref)} − 1 = {ex['k']}. A measurement pivots "
            "on an anticommuting row that does not touch R whenever one exists, which "
            "keeps this count exact.",
        ),
    ]
    ty = H - 0.45
    for title, body in blocks:
        label(ax, tx, ty, title, theme, size=FS, color="ink", weight="bold")
        t = label(ax, tx, ty - 0.2, wrap(body, tw), theme, size=8.2)
        ty -= 0.2 + height(fig, t) + 0.16

    ax.text(
        0.0, H - 0.08, "How SparseGF2 stores a state: a column-major stabilizer tableau",
        fontsize=11, fontweight="bold", color=theme["ink"], va="top", ha="left",
    )  # fmt: skip
    caption(
        fig,
        f"A genuine state: trajectory 0 of a {n_sys}-qubit brickwork circuit (cycle graph, "
        f"p = {ex['p']}, {ex['layers']} layers, {ex['measured']} measurements, seed "
        f"{ex['seed']}) with one reference, replayed with the engine's kernels.",
        theme,
    )
    mf.save(fig, "tableau", mode)


# ---------------------------------------------------------------------------
# 2. engine: spec -> compiled arrays -> parallel trajectories -> result
# ---------------------------------------------------------------------------


def fig_engine(theme, mode):
    H = 7.55
    fig = plt.figure(figsize=(WIDTH, H))
    ax = inch_axes(fig)
    blue, orange, green = (theme["series"][i] for i in (GATE, MEAS, REF))
    ink2 = theme["ink2"]

    ax.text(
        0.0, H - 0.08, "The ensemble engine: one spec, many trajectories in parallel",
        fontsize=11, fontweight="bold", color=theme["ink"], va="top", ha="left",
    )  # fmt: skip

    # --- row A: spec and compiled arrays -------------------------------------
    ya0, ya1 = 5.57, 7.07
    rbox(ax, 0.02, ya0, 4.55, ya1 - ya0, theme)
    label(ax, 0.16, ya1 - 0.1, "MonitoredCircuit  (the spec)", theme, color="ink", weight="bold")
    spec_lines = [
        ("graphs", "edges = [g0, …, g(G−1)], circuits_per_graph = c"),
        ("protocol", "reference picture, scramble, layers L,\nschedule, measurement_rate p"),
        ("checkpoints", "layers where purification is tested"),
        ("probes", "what to record, each with its own checkpoints"),
    ]
    yy = ya1 - 0.36
    for key, val in spec_lines:
        label(ax, 0.16, yy, key, theme, size=8, mono=True, color="ink")
        t = label(ax, 1.2, yy, val, theme, size=8.2, linespacing=1.3)
        yy -= height(fig, t) + 0.07

    rbox(ax, 4.95, ya0, 3.43, ya1 - ya0, theme)
    label(ax, 5.09, ya1 - 0.1, "Compiled once into arrays", theme, color="ink", weight="bold")
    label(
        ax, 5.09, ya1 - 0.36,
        "graphs back to back + offsets, edge colourings\n"
        "probe kinds, qubit regions, output offsets\n"
        "union of all checkpoint layers, and which of\n"
        "them test purification",
        theme, size=8.2,
    )  # fmt: skip
    arrow(ax, (4.6, (ya0 + ya1) / 2), (4.92, (ya0 + ya1) / 2), theme)

    # --- row B: the parallel trajectory loop -----------------------------------
    yb0, yb1 = 1.3, 5.23
    rbox(ax, 0.02, yb0, 8.36, yb1 - yb0, theme, fill="none", edge=theme["axis"], lw=1.2)
    label(
        ax, 0.16, yb1 - 0.1, "run(trajectories, seed = s)", theme, color="ink", weight="bold",
        mono=True, size=8.5,
    )  # fmt: skip
    label(
        ax, 2.95, yb1 - 0.1,
        "every trajectory t in parallel over CPU cores (numba prange)", theme, size=8.5,
    )  # fmt: skip
    arrow(ax, (6.66, ya0 - 0.02), (6.66, yb1 + 0.02), theme)

    steps = [
        (
            "own random streams from (s, t), one for the circuit\nand one for probes; graph g = t // c",
            None,
        ),
        ("setup: Bell pairs with the reference(s),\nthen the optional scramble", None),
        ("layer l = 1 … L:\ngates on graph edges, then Z measurements", blue),
        (
            "checkpoint layer? evaluate the probes due here;\nat the spec's checkpoints, test purification (k = 0)",
            orange,
        ),
    ]
    sx0, sw = 0.22, 3.55
    sy = yb1 - 0.42
    centers = []
    for text, edge in steps:
        t = label(ax, sx0 + 0.1, sy - 0.08, text, theme, size=8.2, linespacing=1.3)
        h = height(fig, t) + 0.16
        rbox(
            ax, sx0, sy - h, sw, h, theme, edge=edge if edge else theme["axis"],
            lw=1.4 if edge else 1.0, radius=0.05,
        )  # fmt: skip
        centers.append((sy, sy - h))
        sy -= h + 0.16
    for (_, bot), (top, _) in zip(centers[:-1], centers[1:], strict=True):
        arrow(ax, (sx0 + 0.28, bot), (sx0 + 0.28, top), theme, lw=1.1)
    # loop back from checkpoint to the next layer
    (l_top, l_bot), (c_top, c_bot) = centers[2], centers[3]
    lx = sx0 + sw + 0.17
    ax.plot(
        [sx0 + sw, lx, lx], [(c_top + c_bot) / 2] * 2 + [(l_top + l_bot) / 2],
        color=ink2, lw=1.1, zorder=3,
    )  # fmt: skip
    arrow(ax, (lx, (l_top + l_bot) / 2), (sx0 + sw + 0.01, (l_top + l_bot) / 2), theme, lw=1.1)
    label(
        ax, lx + 0.07, (l_top + c_bot) / 2, "next\nlayer", theme, size=8, va="center",
        color="muted", linespacing=1.25,
    )  # fmt: skip
    label(
        ax, sx0, c_bot - 0.1,
        "ends after layer L, or at purification\nwith stop_when_purified",
        theme, size=8, color="muted", linespacing=1.3,
    )  # fmt: skip

    # right side: trajectory -> graph mapping
    rx = 4.45
    label(ax, rx, yb1 - 0.45, "Trajectories to graphs (G = 3 graphs, c = 2):", theme, size=8.2)
    tile, tgap = 0.42, 0.05
    ty = yb1 - 1.0
    for t in range(6):
        g = t // 2
        x = rx + t * (tile + tgap) + g * 0.12
        rbox(ax, x, ty, tile, 0.3, theme, edge=theme["axis"], lw=1.0, radius=0.04)
        label(ax, x + tile / 2, ty + 0.15, f"t{t}", theme, size=8, ha="center", va="center")
    for g in range(3):
        x = rx + 2 * g * (tile + tgap) + g * 0.12
        ax.plot([x + 0.03, x + 2 * tile + tgap - 0.03], [ty - 0.06] * 2, color=ink2, lw=1.0)
        label(
            ax, x + tile + tgap / 2, ty - 0.1, f"graph {g}", theme, size=8, ha="center",
            color="muted",
        )  # fmt: skip

    # right side: the flat output row of one trajectory
    oy = yb1 - 1.8
    label(ax, rx, oy + 0.4, "Each trajectory writes one flat row of floats:", theme, size=8.2)
    segments = [  # (probe, evaluations, length per evaluation, colour)
        ("code_dimension", 4, 1, theme["series"][0]),
        ("erasure_first_failure", 2, 4, theme["series"][3]),
        ("code_snapshot", 1, 1, theme["series"][4]),
    ]
    unit = 0.17
    x = rx
    for name, evals, length, color in segments:
        start = x
        for _ in range(evals):
            for i in range(length):
                ax.add_patch(
                    Rectangle(
                        (x + 0.012, oy),
                        unit - 0.024,
                        0.22,
                        facecolor=color,
                        alpha=1.0 if i == 0 else 0.55,
                        linewidth=0,
                        zorder=2,
                    )
                )
                x += unit
            x += 0.04
        last = name == "code_snapshot"  # its label goes on a second line, right-aligned
        label(
            ax, x - 0.04 if last else (start + x - 0.04) / 2, oy - (0.25 if last else 0.05),
            name, theme, size=7.5, mono=True, ha="right" if last else "center",
        )  # fmt: skip
        x += 0.12
    t1 = label(
        ax, rx, oy - 0.55,
        wrap(
            "Evaluation c of probe j fills its own slice of the row: one number for a "
            "scalar probe, a block for a vector probe (M first-failure sizes). Code "
            "snapshots go to a separate array.",
            62,
        ),
        theme, size=8, linespacing=1.3,
    )  # fmt: skip
    label(
        ax, rx, oy - 0.55 - height(fig, t1) - 0.14,
        wrap(
            "Purification is tested only at the spec's checkpoints. After a stop, later "
            "evaluations hold the exact values for a pure reference (k = 0, first failure "
            "n + 1), or NaN for entanglement probes.",
            62,
        ),
        theme, size=8, linespacing=1.3,
    )  # fmt: skip

    # --- row C: result and what reads it ---------------------------------------
    yc0, yc1 = 0.05, 1.12
    arrow(ax, (2.0, yb0 - 0.02), (2.0, yc1 + 0.02), theme)
    rbox(ax, 0.02, yc0, 4.55, yc1 - yc0, theme, edge=green, lw=1.4)
    label(ax, 0.16, yc1 - 0.1, "EnsembleResult", theme, color="ink", weight="bold")
    label(
        ax, 0.16, yc1 - 0.36,
        "values[probe]: (trajectories, evaluations[, length])\n"
        "purification_layer (−1 = censored), final_layer\n"
        "snapshots → result.code(t, c), a StabilizerCode",
        theme, size=8.2,
    )  # fmt: skip
    outs = [
        ("analysis", "P_C(m), Φ(e), e_δ, Kaplan-Meier"),
        ("StabilizerCode", "re-analyse a saved code"),
        ("save / load / concatenate", "chunks merge exactly"),
    ]
    oh = 0.3
    for i, (title, body) in enumerate(outs):
        y = yc1 - i * (oh + 0.085) - oh
        rbox(ax, 4.95, y, 3.43, oh, theme, radius=0.05)
        label(ax, 5.07, y + oh / 2, title, theme, size=8, mono=True, color="ink", va="center")
        label(ax, 8.28, y + oh / 2, body, theme, size=8, va="center", ha="right")
        arrow(ax, (4.6, (yc0 + yc1) / 2), (4.93, y + oh / 2), theme, lw=1.0)

    caption(
        fig,
        "Trajectory t depends only on (spec, s, t): run(count, seed=s, first_trajectory=f) "
        "reproduces that slice of one big run, whatever the thread count.",
        theme,
    )
    mf.save(fig, "engine", mode)


# ---------------------------------------------------------------------------
# 3. erasure-analysis: from a code to P_C(m), Phi(e), e_delta
# ---------------------------------------------------------------------------


def erasure_data():
    n, graphs_count, rate, layers, samples, w = 64, 48, 0.1, 128, 100, 3
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=[graphs.watts_strogatz(n, 4, 0.1, seed=(0, g)) for g in range(graphs_count)],
        layers=layers,
        measurement_rate=rate,
        schedule="random_edges",
        reference="purification",
        checkpoints="final",
        probes=[
            mipt.code_dimension(),
            mipt.erasure_first_failure(samples=samples),
            mipt.uncorrectable_region_counts(w),
            mipt.code_snapshot(),
        ],
    )
    res = spec.run(seed=1)
    k = res.values["code_dimension"][:, -1]
    first = res.values["erasure_first_failure"][:, -1]
    counts = res.values[f"uncorrectable_region_counts[{w}]"][:, -1]
    code = k > 0
    first, counts = first[code], counts[code]
    e = np.linspace(0.0, 0.4, 401)
    delta = 0.1
    return {
        "n": n,
        "G": graphs_count,
        "p": rate,
        "layers": layers,
        "M": samples,
        "w": w,
        "k_mean": float(k.mean()),
        "codes": int(code.sum()),
        "pc_each": analysis.failure_by_size(first, n),
        "pc_mean": analysis.failure_by_size(first, n).mean(axis=0),
        "exact": analysis.exact_failure_by_size(counts, n).mean(axis=0),
        "e": e,
        "phi_each": analysis.erasure_failure_probability(first, n, e, exact_counts=counts),
        "phi_mean": analysis.erasure_failure_probability(first, n, e, exact_counts=counts).mean(
            axis=0
        ),
        "delta": delta,
        "e_delta": analysis.erasure_tolerance(first, n, delta, exact_counts=counts),
    }


def fig_erasure_analysis(theme, mode, d):
    H = 4.6
    fig = plt.figure(figsize=(WIDTH, H))
    n = d["n"]
    blue, orange = theme["series"][0], theme["series"][1]
    ink2, muted = theme["ink2"], theme["muted"]

    # left: the code and its probes (inch coordinates)
    lw_in = 2.75
    ax = inch_axes(fig, (0.0, 0.0, lw_in / WIDTH, 1.0))
    top = H - 0.5
    rbox(ax, 0.02, top - 1.08, lw_in - 0.1, 1.08, theme, edge=theme["series"][REF], lw=1.4)
    label(ax, 0.14, top - 0.09, "The code C after the dynamics", theme, color="ink", weight="bold")
    label(
        ax, 0.14, top - 0.33,
        "stabilizers S (n − k generators),\n"
        "normalizer N(S), k logical qubits.\n"
        "Erasing A is correctable iff I(A:R) = 0.",
        theme, size=8.2,
    )  # fmt: skip
    probes = [
        (
            "erasure_first_failure(M)",
            "D_j = first m at which erasing the first m\nqubits of random ordering j fails",
            blue,
        ),
        (
            "uncorrectable_region_counts(w)",
            "every m-qubit erasure, m ≤ w: exact P_C(m)",
            orange,
        ),
        ("code_snapshot()", "the generators, for StabilizerCode later", theme["axis"]),
    ]
    y = top - 1.3
    for name, body, edge in probes:
        lines = body.count("\n") + 1
        h = 0.3 + 0.165 * lines
        rbox(ax, 0.02, y - h, lw_in - 0.1, h, theme, edge=edge, lw=1.4, radius=0.05)
        label(ax, 0.14, y - 0.07, name, theme, size=7.8, mono=True, color="ink")
        label(ax, 0.14, y - 0.28, body, theme, size=8, linespacing=1.3)
        y -= h + 0.14
    arrow(ax, (lw_in / 2 - 0.1, top - 1.1), (lw_in / 2 - 0.1, top - 1.28), theme, lw=1.1)

    # middle: P_C(m)
    gl, gw = (lw_in + 0.55) / WIDTH, 2.1 / WIDTH
    a1 = fig.add_axes((gl, 0.15, gw, 0.6))
    m = np.arange(n + 1)
    for row in d["pc_each"]:
        a1.plot(
            m[1:], np.where(row[1:] > 0, row[1:], np.nan), color=muted, lw=0.6, alpha=0.35, zorder=1
        )
    a1.plot([], [], color=muted, lw=0.8, label=f"one code ({d['M']} orderings)")
    a1.plot(m[1:], d["pc_mean"][1:], color=blue, lw=2.0, label="mean over codes", zorder=3)
    w = d["w"]
    a1.plot(
        np.arange(1, w + 1), d["exact"], "o", color=orange, markersize=6.5, zorder=4,
        label=f"exact, m ≤ {w}",
    )  # fmt: skip
    a1.set_yscale("log")
    a1.set_ylim(3e-3, 1.4)
    a1.set_xlim(0, n)
    a1.set_xticks([0, 16, 32, 48, 64])
    a1.set_xlabel("erased qubits m")
    a1.set_ylabel(r"$P_C(m)$")
    a1.set_title("Failure by erasure size", fontsize=10)
    legend(a1, loc="lower right", fontsize=7.8, handlelength=1.4, borderaxespad=0.2)

    # right: Phi(e) and e_delta
    a2 = fig.add_axes((gl + gw + 0.75 / WIDTH, 0.15, gw, 0.6))
    for row in d["phi_each"]:
        a2.plot(d["e"], row, color=muted, lw=0.6, alpha=0.35, zorder=1)
    a2.plot([], [], color=muted, lw=0.8, label="one code")
    a2.plot(d["e"], d["phi_mean"], color=blue, lw=2.0, label="mean over codes", zorder=3)
    ed, delta = d["e_delta"], d["delta"]
    a2.plot([0, ed], [delta, delta], color=ink2, lw=0.9, zorder=2)
    a2.plot([ed, ed], [0, delta], color=ink2, lw=0.9, zorder=2)
    a2.plot(ed, delta, "o", color=blue, markersize=6.5, zorder=4)
    a2.annotate(
        rf"$e_{{0.1}}$ = {ed:.3f}",
        xy=(ed, delta),
        xytext=(9, -3),
        textcoords="offset points",
        fontsize=8.5,
        color=theme["ink"],
    )
    a2.text(0.005, delta + 0.02, r"$\delta$ = 0.1", fontsize=8, color=ink2, va="bottom")
    a2.set_xlim(0, d["e"][-1])
    a2.set_ylim(0, 1.0)
    a2.set_xlabel("erasure probability e (each qubit)")
    a2.set_ylabel(r"$\Phi(e)$")
    a2.set_title("Failure under independent erasure", fontsize=10)
    legend(a2, loc="upper left", fontsize=7.8, handlelength=1.4, borderaxespad=0.2)

    # formulas
    fig.text(
        gl,
        0.875,
        r"$P_C(m)$ = fraction of m-qubit erasures that fail       "
        r"$\Phi(e) = \sum_m \binom{n}{m}\, e^m (1-e)^{n-m}\, P_C(m)$       "
        r"$e_\delta = \sup\,\{e : \Phi(e) \leq \delta\}$",
        fontsize=8.5,
        color=theme["ink"],
        ha="left",
        va="center",
    )
    fig.text(
        0.0, 0.985, "From a code to its erasure numbers", fontsize=11, fontweight="bold",
        color=theme["ink"], ha="left", va="top",
    )  # fmt: skip
    caption(
        fig,
        f"Computed here: n = {n}, purification picture, {d['G']} Watts-Strogatz graphs (k = 4, "
        f"beta = 0.1, 1 circuit each), p = {d['p']}, {d['layers']} layers, seed 1;\nall "
        f"{d['codes']} codes have k > 0 (mean k = {d['k_mean']:.1f}). Φ uses the exact terms "
        f"for m ≤ {w} (the hybrid estimator).",
        theme,
    )
    mf.save(fig, "erasure-analysis", mode)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="where to write the PNGs (default: docs/figures, the copies the docs use)",
    )
    args = ap.parse_args()
    if args.outdir:
        mf.FIGURES = Path(args.outdir)  # save() reads make_figures.FIGURES at call time
    ex = tableau_example()
    er = erasure_data()
    for mode, theme in THEMES.items():
        style(theme)
        fig_tableau(theme, mode, ex)
        fig_engine(theme, mode)
        fig_erasure_analysis(theme, mode, er)
    print(f"wrote tableau, engine and erasure-analysis figures to {mf.FIGURES}")


if __name__ == "__main__":
    main()
