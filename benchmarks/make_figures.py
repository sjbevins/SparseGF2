"""Render the README figures (light and dark PNGs) from the benchmark results.

Reads ``results/simulators.json`` (from ``compare_simulators.py``),
``results/physics.json`` (from ``physics_data.py``) and ``results/why.json``
(from ``why_data.py``), skipping any that is missing, and writes
``docs/figures/<name>-light.png`` and ``<name>-dark.png``; the README shows
the right one with a ``<picture>`` element. The exact numbers go to
``results/tables.md``.

Usage::

    python benchmarks/make_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "docs" / "figures"

# Colour roles (validated categorical order, and one-hue ordinal ramps per mode).
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
        "ramp4": ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
        "ramp3": ["#86b6ef", "#2a78d6", "#0d366b"],
        "box": "#eef4fc",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
        "ramp4": ["#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"],
        "ramp3": ["#256abf", "#6da7ec", "#cde2fb"],
        "box": "#1f2a38",
    },
}

# Entity -> categorical slot, the same in every figure.
SIMULATORS = [
    ("sparsegf2_phases", "SparseGF2", 0),
    ("sparsegf2_phase_free", "SparseGF2, phase-free", 1),
    ("stim", "Stim", 2),
    ("qiskit_aer", "Qiskit Aer (stabilizer)", 3),
    ("cirq", "Cirq (Clifford)", 4),
]


def style(theme: dict) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": theme["surface"],
            "axes.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "text.color": theme["ink"],
            "axes.labelcolor": theme["ink2"],
            "axes.edgecolor": theme["axis"],
            "axes.titlecolor": theme["ink"],
            "xtick.color": theme["ink2"],
            "ytick.color": theme["ink2"],
            "xtick.labelcolor": theme["ink2"],
            "ytick.labelcolor": theme["ink2"],
            "axes.grid": True,
            "grid.color": theme["grid"],
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.axisbelow": True,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.5,
            "lines.markeredgewidth": 1.5,
            "lines.markeredgecolor": theme["surface"],
            "lines.solid_capstyle": "round",
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
        }
    )


def save(fig, name: str, mode: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}-{mode}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def end_labels(ax, items, theme, *, log=True, min_gap=0.055):
    """Direct labels at the right end of each line, nudged apart vertically.

    ``items`` is a list of (x, y, text). Only labels that end at the same x can
    collide, so each such group is spread out separately, keeping every label
    inside the axes. Labels use the secondary text colour; the line ending
    beside each label carries the series identity.
    """
    if not items:
        return
    lo, hi = ax.get_ylim()
    to = (lambda v: np.log10(v)) if log else (lambda v: v)
    back = (lambda v: 10**v) if log else (lambda v: v)
    span = to(hi) - to(lo)
    gap = min_gap * span
    top = to(hi) - 0.02 * span
    groups: dict[float, list[int]] = {}
    for i, (x, _, _) in enumerate(items):
        groups.setdefault(x, []).append(i)
    for x, members in groups.items():
        order = sorted(members, key=lambda i: to(items[i][1]))
        pos = [to(items[i][1]) for i in order]
        for k in range(1, len(pos)):
            pos[k] = max(pos[k], pos[k - 1] + gap)
        if pos[-1] > top:  # pushed out of the axes: pin the top label, push the rest down
            pos[-1] = top
            for k in range(len(pos) - 2, -1, -1):
                pos[k] = min(pos[k], pos[k + 1] - gap)
        for k, i in enumerate(order):
            ax.annotate(
                items[i][2],
                xy=(x, back(pos[k])),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=9,
                color=theme["ink2"],
                annotation_clip=False,
            )


def legend(ax, **kw):
    leg = ax.legend(**kw)
    for h in leg.legend_handles:
        if hasattr(h, "set_markeredgewidth"):
            h.set_markeredgewidth(0)
    return leg


def log_ticks(axis, values):
    axis.set_major_locator(matplotlib.ticker.FixedLocator(values))
    axis.set_minor_locator(matplotlib.ticker.NullLocator())


def caption(fig, text, theme):
    fig.text(
        0.0,
        -0.02,
        text,
        ha="left",
        va="top",
        fontsize=8,
        color=theme["muted"],
        transform=fig.transFigure,
    )


# ---------------------------------------------------------------------------
# benchmark figures
# ---------------------------------------------------------------------------


def runs_table(sim: dict, name: str, p: float):
    rows = [r for r in sim["runs"] if r["simulator"] == name and abs(r["p"] - p) < 1e-9]
    rows.sort(key=lambda r: r["n"])
    return np.array([r["n"] for r in rows]), np.array([r["seconds"] for r in rows])


def machine_line(env: dict) -> str:
    cpu = env.get("cpu_model") or env.get("processor") or "unknown CPU"
    names = {"stim": "Stim", "qiskit_aer": "Qiskit Aer", "cirq": "Cirq"}
    versions = ", ".join(f"{names[k]} {env[k]}" for k in names if env.get(k))
    if env.get("stim_build"):
        versions = versions.replace(
            f"Stim {env['stim']}", f"Stim {env['stim']} ({env['stim_build'].upper()} build)"
        )
    return f"{cpu}; one thread per simulator; median of repeated runs. {versions}."


def fig_runtime(sim: dict, theme: dict, mode: str, p: float = 0.16) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = []
    for key, label, slot in SIMULATORS:
        n, t = runs_table(sim, key, p)
        if n.size == 0:
            continue
        ax.plot(
            n, t, marker="o", color=theme["series"][slot], label=label, zorder=3 if slot < 2 else 2
        )
        labels.append((n[-1], t[-1], label))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("qubits n  (depth 2n, brickwork random Clifford gates)")
    ax.set_ylabel("seconds per circuit")
    ax.set_title(f"Time to simulate one monitored circuit (measurement rate p = {p})", pad=52)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.3)
    end_labels(ax, labels, theme)
    caption(fig, machine_line(sim["environment"]), theme)
    save(fig, "runtime", mode)


def fig_speedup(sim: dict, theme: dict, mode: str) -> None:
    rates = sorted({r["p"] for r in sim["runs"]})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for ax, (key, title) in zip(
        axes,
        [
            ("sparsegf2_phases", "SparseGF2 (phases tracked)"),
            ("sparsegf2_phase_free", "SparseGF2, phase-free"),
        ],
        strict=True,
    ):
        ramp = theme["ramp3"] if len(rates) <= 3 else theme["ramp4"]
        for p, color in zip(rates, ramp, strict=False):
            n_s, t_s = runs_table(sim, "stim", p)
            n_v, t_v = runs_table(sim, key, p)
            common = np.intersect1d(n_s, n_v)
            speed = t_s[np.isin(n_s, common)] / t_v[np.isin(n_v, common)]
            ax.plot(common, speed, marker="o", color=color, label=f"p = {p}")
        ax.axhline(1.0, color=theme["muted"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        log_ticks(ax.yaxis, [0.5, 1, 2, 5, 10, 20, 50, 100, 200])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("qubits n")
    axes[0].set_ylabel("speed-up over Stim")
    axes[0].annotate(
        "Stim",
        xy=(axes[0].get_xlim()[0], 1.0),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
        color=theme["muted"],
    )
    legend(axes[1], loc="lower right", title="measurement rate", title_fontsize=9)
    fig.suptitle(
        "Speed-up over Stim on the same circuits",
        x=0.0,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=theme["ink"],
    )
    fig.tight_layout()
    caption(fig, machine_line(sim["environment"]), theme)
    save(fig, "speedup", mode)


def fig_throughput(sim: dict, theme: dict, mode: str, p: float = 0.16) -> None:
    ens = sorted((e for e in sim["ensemble"] if abs(e["p"] - p) < 1e-9), key=lambda e: e["n"])
    if not ens:
        return
    threads = ens[0]["threads"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    n_e = np.array([e["n"] for e in ens])
    labels = []
    series = [
        (
            n_e,
            np.array([e["circuits_per_second"] for e in ens]),
            f"SparseGF2 mipt engine, {threads} threads",
            0,
            "-",
        ),
    ]
    n_v, t_v = runs_table(sim, "sparsegf2_phase_free", p)
    series.append((n_v, 1.0 / t_v, "SparseGF2 Simulator.run (phase-free), 1 thread", 1, "-"))
    n_s, t_s = runs_table(sim, "stim", p)
    series.append((n_s, 1.0 / t_s, "Stim, 1 thread", 2, "-"))
    series.append((n_s, threads / t_s, f"Stim, ideal {threads}-process scaling", 2, (0, (4, 3))))
    for n, cps, label, slot, ls in series:
        keep = n <= n_e.max()
        ax.plot(
            n[keep],
            cps[keep],
            marker="o" if ls == "-" else None,
            linestyle=ls,
            color=theme["series"][slot],
            label=label,
        )
        labels.append(
            (
                n[keep][-1],
                cps[keep][-1],
                label.split(",")[0] + ("" if ls == "-" else " ×" + str(threads)),
            )
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("qubits n  (depth 2n)")
    ax.set_ylabel("circuits per second")
    ax.set_title(f"Throughput on fresh random monitored circuits (p = {p})")
    legend(ax, loc="lower left")
    end_labels(ax, labels, theme)
    caption(
        fig,
        "The mipt engine generates and simulates a new random circuit per trajectory, in parallel; "
        "the other curves rerun one prebuilt circuit. "
        + (sim["environment"].get("cpu_model") or ""),
        theme,
    )
    save(fig, "throughput", mode)


# ---------------------------------------------------------------------------
# physics figures
# ---------------------------------------------------------------------------


def fig_transition(phys: dict, theme: dict, mode: str) -> None:
    d = phys["transition"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rates = np.array(d["rates"])
    for n, color in zip(d["sizes"], theme["ramp4"], strict=False):
        y = np.array(d["neg_i3"][str(n)])
        e = np.array(d["neg_i3_err"][str(n)])
        ax.fill_between(rates, y - e, y + e, color=color, alpha=0.18, linewidth=0)
        ax.plot(rates, y, marker="o", color=color, label=f"n = {n}")
    ax.axvline(0.16, color=theme["muted"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(
        "p_c ≈ 0.16",
        xy=(0.16, ax.get_ylim()[1]),
        xytext=(4, -12),
        textcoords="offset points",
        fontsize=8,
        color=theme["muted"],
    )
    ax.set_xlabel("measurement rate p")
    ax.set_ylabel("−I₃ (four quarters of the ring)")
    ax.set_title("Measurement-induced transition: −I₃ curves cross at p_c")
    legend(ax, loc="upper right")
    caption(
        fig,
        "Brickwork random Cliffords on a ring, depth 4n from |0…0⟩; shaded bands are ±1 s.e.",
        theme,
    )
    save(fig, "transition", mode)


def fig_purification(phys: dict, theme: dict, mode: str) -> None:
    d = phys["purification"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for ax, p in zip(axes, d["rates"], strict=True):
        for n, color in zip(d["sizes"], theme["ramp4"], strict=False):
            key = f"{p}/{n}"
            ax.plot(d["layers"][key], d["survival"][key], color=color, label=f"n = {n}")
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        phase = "mixed phase" if p < 0.16 else "pure phase"
        ax.set_title(f"p = {p} ({phase})", fontsize=10)
        ax.set_xlabel("layers t after scrambling")
    axes[0].set_ylabel("P(reference still mixed at t)")
    legend(axes[0], loc="lower left")
    fig.suptitle(
        "Purification of a single reference qubit",
        x=0.0,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=theme["ink"],
    )
    fig.tight_layout()
    caption(
        fig,
        "Reference Bell-paired with a random system qubit, scrambled for 2n gate-only layers; then monitored brickwork dynamics.",
        theme,
    )
    save(fig, "purification", mode)


def fig_erasure(phys: dict, theme: dict, mode: str) -> None:
    d = phys["erasure"]
    n = d["n"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    layers = np.array(d["layers"]) / n
    for p, color in zip(d["rates"], theme["ramp3"], strict=False):
        axes[0].plot(layers, d["code_rate"][str(p)], color=color, label=f"p = {p}")
        axes[1].plot(d["erased_fraction"], d["correctable"][str(p)], color=color, label=f"p = {p}")
    axes[0].set_xlabel("layers t / n")
    axes[0].set_ylabel("code rate k / n")
    axes[0].set_title("Code dimension of the dynamically generated code", fontsize=10)
    legend(axes[0], loc="upper right", title="measurement rate", title_fontsize=9)
    axes[1].set_xlabel("erased fraction m / n")
    axes[1].set_ylabel("P(random erasure is correctable)")
    axes[1].set_title("Erasure protection at t = n", fontsize=10)
    axes[1].set_ylim(-0.02, 1.02)
    fig.suptitle(
        f"Erasure protection in monitored circuits (n = {n}, maximally mixed start)",
        x=0.0,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=theme["ink"],
    )
    fig.tight_layout()
    caption(
        fig,
        "An erasure E is correctable iff I(R:E) = 0 (no logical information leaks into E).",
        theme,
    )
    save(fig, "erasure", mode)


# ---------------------------------------------------------------------------
# "why this simulator" figures
# ---------------------------------------------------------------------------

WHY_SLOTS = {"sparsegf2": ("SparseGF2", 0), "stim": ("Stim", 2)}
ERASURE_TASKS = [
    ("counts2", "every erasure of 1 or 2 qubits"),
    ("counts3", "every erasure of up to 3 qubits"),
    ("first_failure100", "first failure, 100 random orderings"),
]


def why_machine_line(env: dict, with_stim: bool = True) -> str:
    cpu = env.get("cpu_model") or env.get("processor") or "unknown CPU"
    stim = f", Stim {env['stim']}" if with_stim and env.get("stim") else ""
    return f"{cpu}; one thread{stim}, numba {env.get('numba', '?')}."


def _log_axis(axis) -> None:
    """Decade ticks labelled as plain numbers (0.01, 1, 100), no minor labels."""
    axis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=12))
    axis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def _figure_legend(fig, ax, ncol: int) -> None:
    """One legend for all panels, between the figure title and the panel titles."""
    handles, texts = ax.get_legend_handles_labels()
    leg = fig.legend(handles, texts, loc="upper left", bbox_to_anchor=(0.0, 0.94), ncol=ncol,
                     frameon=False, fontsize=9, borderaxespad=0.0)  # fmt: skip
    for h in leg.legend_handles:
        if hasattr(h, "set_markeredgewidth"):
            h.set_markeredgewidth(0)


def _added(times: dict, layers: int) -> str:
    """The time measuring adds per layer, as text.

    Stim's observed run is timed in phases, so its added time is the extract + rank time measured
    directly. SparseGF2 evaluates probes inside the compiled layer loop, so its added time is the
    observed minus the evolve-only time; under 5 % of the evolve time that is timing noise.
    """
    if "phases" in times:
        extra = times["phases"]["extract"] + times["phases"]["rank"]
    else:
        extra = times["observe"] - times["evolve"]
        if extra < 0.05 * times["evolve"]:
            return "almost nothing"
    ms = 1e3 * extra / layers
    if ms >= 1:
        return f"{ms:.1f} ms"
    return f"{ms:.2g} ms" if ms >= 0.01 else f"{1e3 * ms:.2g} µs"


def fig_why_observe(why: dict, theme: dict, mode: str) -> None:
    obs = why["observe"]
    panels = [
        ("half_cut_entropy", "Half-cut entropy S(A) after every layer\n(n qubits)"),
        (
            "code_dimension",
            "Code dimension k = S(R) after every layer\n(n system + n reference qubits)",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 5.4), sharey=True)
    fig.subplots_adjust(top=0.68, bottom=0.11, left=0.08, right=0.87, wspace=0.32)
    for ax, (observable, title) in zip(axes, panels, strict=True):
        rows = sorted(
            (r for r in obs["rows"] if r["observable"] == observable), key=lambda r: r["n"]
        )
        if not rows:
            ax.set_visible(False)
            continue
        n = np.array([r["n"] for r in rows])
        labels = []
        for key, (label, slot) in WHY_SLOTS.items():
            per_layer = {
                what: np.array([r[key][what] / r["layers"] for r in rows]) * 1e3
                for what in ("evolve", "observe")
            }
            color = theme["series"][slot]
            ax.plot(n, per_layer["evolve"], color=color, linewidth=1.4, linestyle=(0, (4, 3)),
                    label=f"{label}, evolve only")  # fmt: skip
            ax.plot(n, per_layer["observe"], color=color, marker="o",
                    label=f"{label}, evolve + measure every layer")  # fmt: skip
            labels.append((n[-1], per_layer["observe"][-1], label))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlabel("system qubits n")
        ax.set_title(title, fontsize=10, pad=32)
        big = rows[-1]
        added = [_added(big[key], big["layers"]) for key in ("sparsegf2", "stim")]
        ax.annotate(
            f"measuring adds, per layer at n = {big['n']}:\n"
            f"SparseGF2 {added[0]}, Stim {added[1]}",
            xy=(0.0, 1.02), xycoords="axes fraction", ha="left", va="bottom",
            fontsize=8.5, color=theme["ink2"],
        )  # fmt: skip
        end_labels(ax, labels, theme)
    axes[0].set_ylabel("milliseconds per layer")
    _log_axis(axes[0].yaxis)
    low = min(line.get_ydata().min() for ax in axes for line in ax.get_lines())
    axes[0].set_ylim(bottom=10 ** np.floor(np.log10(low)))  # a labelled tick below the lowest point
    _figure_legend(fig, axes[0], ncol=2)
    fig.suptitle(
        "Measuring the state while it evolves: cost per layer, same circuits",
        x=0.0, y=0.995, ha="left", va="top", fontsize=11, fontweight="bold", color=theme["ink"],
    )  # fmt: skip
    caption(
        fig,
        f"Monitored brickwork random Clifford circuits on a ring, p = {obs['measurement_rate']}, "
        f"{obs['layers']} layers; the same circuits in both simulators.\n"
        "Stim: per-layer do_circuit, stabilizer columns read from the inverse tableau, compiled "
        "GF(2) rank; its added time is the timed extract + rank (its evolve-only line is a\n"
        "separate run of one prebuilt circuit, so the gap between Stim's lines can understate it).\n"
        "SparseGF2: mipt engine probes; its added time is measured minus evolve-only. k is kept as "
        "a count of reference rows.\n" + why_machine_line(why["environment"]),
        theme,
    )
    save(fig, "why-observe", mode)


def _erasure_series(rows, side: str, task: str):
    """Sizes and the median seconds per code over the codes of each size (only measured ones)."""
    by_n: dict[int, list[float]] = {}
    for r in rows:
        if task in r[side]:
            by_n.setdefault(r["n"], []).append(r[side][task])
    sizes = sorted(by_n)
    return np.array(sizes), np.array([float(np.median(by_n[m])) for m in sizes]), by_n


def fig_why_erasure(why: dict, theme: dict, mode: str) -> None:
    era = why["erasure"]
    rows = era["rows"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.6), sharey=True)
    fig.subplots_adjust(top=0.75, bottom=0.12, left=0.08, right=0.99, wspace=0.12)
    sides = [
        ("kernel", "SparseGF2 StabilizerCode", theme["series"][0]),
        ("baseline", "compiled baseline: both ranks from scratch for every set", theme["muted"]),
    ]
    skipped = sorted({(r["n"], t) for r in rows for t in r["skipped"]})
    for ax, (task, title) in zip(axes, ERASURE_TASKS, strict=True):
        for side, label, color in sides:
            n, med, by_n = _erasure_series(rows, side, task)
            if n.size == 0:
                continue
            for m, values in by_n.items():  # every code, faint
                ax.scatter([m] * len(values), np.array(values) * 1e3, s=14, color=color,
                           alpha=0.35, linewidths=0, zorder=2)  # fmt: skip
            ax.plot(n, med * 1e3, color=color, marker="o", label=label, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlabel("qubits n")
        ax.set_title(title, fontsize=10, pad=20)
        # the ratio at the largest size where both were measured
        nk, tk, _ = _erasure_series(rows, "kernel", task)
        nb, tb, _ = _erasure_series(rows, "baseline", task)
        common = np.intersect1d(nk, nb)
        if common.size:
            m = common[-1]
            ratio = tb[nb == m][0] / tk[nk == m][0]
            ax.annotate(
                f"{_ratio(ratio)} faster at n = {m}" if ratio >= 1.05 else f"{_ratio(ratio)} at n = {m}",
                xy=(0.0, 1.02), xycoords="axes fraction", ha="left", va="bottom",
                fontsize=8.5, color=theme["ink2"],
            )  # fmt: skip
    axes[0].set_ylabel("milliseconds per code")
    _log_axis(axes[0].yaxis)
    _figure_legend(fig, axes[0], ncol=2)
    fig.suptitle(
        "Erasure screening of one code: exact counts and first-failure sizes",
        x=0.0, y=0.995, ha="left", va="top", fontsize=11, fontweight="bold", color=theme["ink"],
    )  # fmt: skip
    note = ""
    if skipped:
        note = " Missing baseline points were estimated above the time limit and not run."
    caption(
        fig,
        f"Codes made by the engine: purification picture, Watts-Strogatz graph (k = 4, beta = 0.1), "
        f"p = {era['measurement_rate']}, n/2 layers; dots are single codes, lines the median.\n"
        "Both sides give identical counts and first-failure sizes (checked)."
        + note
        + " "
        + why_machine_line(why["environment"], with_stim=False),
        theme,
    )
    save(fig, "why-erasure", mode)


# ---------------------------------------------------------------------------
# schematic
# ---------------------------------------------------------------------------


def fig_schematic(theme: dict, mode: str, n: int = 8, depth: int = 7, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.set_axis_off()
    gate = theme["series"][0]
    meas = theme["series"][1]
    ref = theme["series"][2]
    y = {q: n - 1 - q for q in range(n)}
    x0, dx = 1.2, 1.0
    for q in range(n):
        ax.plot(
            [0.35, x0 + depth * dx + 0.3],
            [y[q], y[q]],
            color=theme["axis"],
            linewidth=1.2,
            zorder=1,
        )
        ax.text(0.25, y[q], f"q{q}", ha="right", va="center", fontsize=8, color=theme["ink2"])
    # reference qubit Bell-paired with q2
    yr = n + 0.4
    ax.plot(
        [0.35, x0 + depth * dx + 0.3],
        [yr, yr],
        color=ref,
        linewidth=1.2,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    ax.text(0.25, yr, "R", ha="right", va="center", fontsize=8, color=theme["ink2"])
    ax.plot([0.7, 0.7], [yr, y[2]], color=ref, linewidth=2.0, zorder=2)
    for yy in (yr, y[2]):
        ax.plot(
            0.7, yy, marker="o", markersize=6, color=ref, markeredgecolor=theme["surface"], zorder=3
        )
    ax.text(0.78, (yr + y[2]) / 2 + 1.0, "Bell pair", fontsize=8, color=theme["ink2"], va="center")
    for t in range(depth):
        x = x0 + t * dx
        for a in range(t % 2, n - 1, 2):
            box = FancyBboxPatch(
                (x - 0.28, y[a + 1] - 0.3), 0.56, 1.6, boxstyle="round,pad=0,rounding_size=0.08",
                facecolor=theme["box"], edgecolor=gate, linewidth=1.4, zorder=2,
            )  # fmt: skip
            ax.add_patch(box)
            ax.text(
                x,
                y[a] - 0.5,
                "C",
                ha="center",
                va="center",
                fontsize=8,
                color=theme["ink"],
                zorder=3,
            )
        measured = np.flatnonzero(rng.random(n) < 0.2)
        for q in measured:
            ax.plot(
                x + 0.5,
                y[q],
                marker="o",
                markersize=9,
                color=meas,
                markeredgecolor=theme["surface"],
                markeredgewidth=1.5,
                zorder=4,
            )
    xe = x0 + depth * dx + 0.45
    ax.text(
        xe,
        n / 2 - 0.5,
        "probe the state\nat any layer:\nS(A), I(A:B), I₃,\nk = S(R), d, erasures",
        fontsize=8.5,
        color=theme["ink2"],
        va="center",
    )
    lx = x0
    ax.add_patch(
        FancyBboxPatch(
            (lx - 0.15, -1.55),
            0.3,
            0.45,
            boxstyle="round,pad=0,rounding_size=0.05",
            facecolor=theme["box"],
            edgecolor=gate,
            linewidth=1.4,
        )
    )
    ax.text(
        lx + 0.3, -1.33, "random two-qubit Clifford", fontsize=8, color=theme["ink2"], va="center"
    )
    ax.plot(
        lx + 3.1,
        -1.33,
        marker="o",
        markersize=9,
        color=meas,
        markeredgecolor=theme["surface"],
        markeredgewidth=1.5,
    )
    ax.text(
        lx + 3.35, -1.33, "Z measurement (rate p)", fontsize=8, color=theme["ink2"], va="center"
    )
    ax.plot([lx + 5.6, lx + 6.0], [-1.33, -1.33], color=ref, linewidth=2.0)
    ax.text(lx + 6.15, -1.33, "reference qubit", fontsize=8, color=theme["ink2"], va="center")
    ax.set_xlim(-0.3, xe + 2.6)
    ax.set_ylim(-1.9, yr + 0.6)
    ax.set_title(
        "A monitored random Clifford circuit (one layer = gates, then measurements)", loc="left"
    )
    save(fig, "circuit", mode)


def _ms(seconds: float) -> str:
    """Milliseconds with three significant digits, without scientific notation."""
    ms = seconds * 1e3
    if ms >= 100:
        return f"{ms:,.0f}"
    if ms >= 10:
        return f"{ms:.1f}"
    if ms >= 1:
        return f"{ms:.2f}"
    return f"{ms:.3f}"


def tables(sim: dict | None, why: dict | None = None) -> str:
    """Markdown tables of the benchmark numbers, for the README (exact values from the JSON)."""
    out = []
    if sim:
        rates = sorted({r["p"] for r in sim["runs"]})
        names = {key: label for key, label, _ in SIMULATORS}
        for p in rates:
            sizes = sorted({r["n"] for r in sim["runs"] if abs(r["p"] - p) < 1e-9})
            present = [k for k, _, _ in SIMULATORS if any(r["simulator"] == k for r in sim["runs"])]
            out.append(
                f"**Milliseconds per circuit, p = {p}** (depth 2n; speed-up over Stim in brackets)\n"
            )
            out.append("| n | " + " | ".join(names[k] for k in present) + " |")
            out.append("|---:|" + "---:|" * len(present))
            for n in sizes:
                t = {
                    r["simulator"]: r["seconds"]
                    for r in sim["runs"]
                    if r["n"] == n and abs(r["p"] - p) < 1e-9
                }
                cells = []
                for k in present:
                    if k not in t:
                        cells.append("–")
                    elif k.startswith("sparsegf2") and "stim" in t:
                        cells.append(f"{_ms(t[k])} ({t['stim'] / t[k]:.1f}×)")
                    else:
                        cells.append(_ms(t[k]))
                out.append(f"| {n} | " + " | ".join(cells) + " |")
            out.append("")
        ens = sorted(sim["ensemble"], key=lambda e: (e["p"], e["n"]))
        if ens:
            out.append(
                f"**Ensemble engine throughput** (fresh random circuits, depth 2n, "
                f"{ens[0]['threads']} threads), circuits per second\n"
            )
            sizes = sorted({e["n"] for e in ens})
            ps = sorted({e["p"] for e in ens})
            out.append("| p | " + " | ".join(f"n = {n}" for n in sizes) + " |")
            out.append("|---:|" + "---:|" * len(sizes))
            for p in ps:
                row = {e["n"]: e["circuits_per_second"] for e in ens if e["p"] == p}
                out.append(f"| {p} | " + " | ".join(f"{row[n]:,.0f}" for n in sizes) + " |")
            out.append("")
    if why:
        out.extend(why_tables(why))
    return "\n".join(out)


def _ratio(x: float) -> str:
    return f"{x:.0f}×" if x >= 10 else f"{x:.1f}×"


def why_tables(why: dict) -> list[str]:
    out = []
    obs = why.get("observe", {})
    names = {"half_cut_entropy": "half-cut entropy (n qubits)",
             "code_dimension": "code dimension k = S(R) (2n qubits)"}  # fmt: skip
    for observable, title in names.items():
        rows = sorted((r for r in obs.get("rows", []) if r["observable"] == observable),
                      key=lambda r: r["n"])  # fmt: skip
        if not rows:
            continue
        out.append(
            f"**Measuring the {title} after every layer**, milliseconds per circuit "
            f"(p = {obs['measurement_rate']}, {obs['layers']} layers, one thread; "
            "Stim's observed time split into its phases; in brackets, Stim's evolve + measure "
            "time over SparseGF2's)\n"
        )
        out.append(
            "| n | layers | SparseGF2 evolve | SparseGF2 evolve + measure | Stim evolve "
            "| Stim evolve + measure | Stim evolve / extract / rank |"
        )
        out.append("|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            sg, st = r["sparsegf2"], r["stim"]
            ph = st["phases"]
            split = " / ".join(_ms(ph[k]) for k in ("evolve", "extract", "rank"))
            out.append(
                f"| {r['n']} | {r['layers']} | {_ms(sg['evolve'])} | {_ms(sg['observe'])} | "
                f"{_ms(st['evolve'])} | {_ms(st['observe'])} ({_ratio(st['observe'] / sg['observe'])}) "
                f"| {split} |"
            )
        out.append("")
    era = why.get("erasure", {})
    rows = era.get("rows", [])
    if rows:
        out.append(
            "**Erasure screening per code**, milliseconds (one thread; median over the codes of "
            "each size; baseline = compiled from-scratch evaluation of l(A), speed-up in brackets)\n"
        )
        out.append("| n | codes | mean k | " + " | ".join(t for _, t in ERASURE_TASKS) + " |")
        out.append("|---:|---:|---:|" + "---:|" * len(ERASURE_TASKS))
        for n in sorted({r["n"] for r in rows}):
            group = [r for r in rows if r["n"] == n]
            cells = []
            for task, _ in ERASURE_TASKS:
                kern = float(np.median([r["kernel"][task] for r in group]))
                base = [r["baseline"][task] for r in group if task in r["baseline"]]
                if base:
                    b = float(np.median(base))
                    cells.append(f"{_ms(kern)} vs {_ms(b)} ({_ratio(b / kern)})")
                else:
                    cells.append(f"{_ms(kern)} (baseline skipped)")
            mean_k = np.mean([r["k"] for r in group])
            out.append(f"| {n} | {len(group)} | {mean_k:.0f} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--simulators", default=str(HERE / "results" / "simulators.json"))
    ap.add_argument("--physics", default=str(HERE / "results" / "physics.json"))
    ap.add_argument("--why", default=str(HERE / "results" / "why.json"))
    ap.add_argument(
        "--outdir",
        default=None,
        help="where to write figures and tables.md (default: docs/figures and "
        "benchmarks/results/tables.md, the copies the READMEs use)",
    )
    args = ap.parse_args()
    tables_path = HERE / "results" / "tables.md"
    if args.outdir:
        global FIGURES
        FIGURES = Path(args.outdir)
        tables_path = FIGURES / "tables.md"  # never overwrite the committed tables from a test run
    sim = (
        json.loads(Path(args.simulators).read_text(encoding="utf-8"))
        if Path(args.simulators).exists()
        else None
    )
    phys = (
        json.loads(Path(args.physics).read_text(encoding="utf-8"))
        if Path(args.physics).exists()
        else None
    )
    why = (
        json.loads(Path(args.why).read_text(encoding="utf-8")) if Path(args.why).exists() else None
    )
    for mode, theme in THEMES.items():
        style(theme)
        fig_schematic(theme, mode)
        if sim:
            fig_runtime(sim, theme, mode)
            fig_speedup(sim, theme, mode)
            fig_throughput(sim, theme, mode)
        if phys:
            fig_transition(phys, theme, mode)
            fig_purification(phys, theme, mode)
            fig_erasure(phys, theme, mode)
        if why:
            if why.get("observe", {}).get("rows"):
                fig_why_observe(why, theme, mode)
            fig_why_erasure(why, theme, mode)
    FIGURES.mkdir(parents=True, exist_ok=True)
    tables_path.write_text(tables(sim, why), encoding="utf-8")
    print(f"wrote figures to {FIGURES} and tables to {tables_path}")


if __name__ == "__main__":
    main()
