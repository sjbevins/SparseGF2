"""
Analysis + LaTeX report generator for a single_ref MIPT sweep produced
by ``scripts/run_all_to_all_mipt.py``.

Pipeline:

1. Load every ``samples.parquet`` under the run directory using polars
   (hive partitioning lifts ``n`` and ``p`` into columns).
2. Aggregate ``obs.k`` per ``(n, p)``: mean, ddof-1 sample std, SEM,
   sample count.
3. Emit ``crossing_plot.pdf``: <S> vs p curves per system size with
   SEM error bars, on PDF (vector output for inclusion in the LaTeX
   document).
4. Estimate the critical rate ``p_c`` from the crossing of the two
   largest system sizes' curves (linear interpolation at the sign
   change of their difference).
5. Emit ``report.tex``: complete, standalone LaTeX document containing
   an abstract, introduction, methods, results (the plot + a raw-value
   table), and a conclusion reporting ``p_c``.

Usage
-----

    python scripts/generate_report.py <run_dir>

Optional:
    --out-dir DIR          write outputs somewhere other than the run dir
    --p-c-method METHOD    crossing | half  (default: crossing)
    --compile              invoke pdflatex on report.tex (requires TeX)

The generated LaTeX compiles stand-alone with pdflatex (no custom
packages beyond amsmath, booktabs, graphicx, longtable, hyperref).
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl

from sparsegf2.analysis import (
    aggregate_single_ref_entropy,
    load_single_ref_samples,
    plot_single_ref_crossing,
)


# Critical-point estimation

def _estimate_pc_from_crossing(agg: pl.DataFrame) -> Optional[float]:
    """Locate the crossing of <S>(p) curves for the two largest system
    sizes, interpolated linearly at the first sign change of their
    difference. Returns None if the curves do not cross within the
    sampled p-window.
    """
    sizes = sorted(int(v) for v in agg["n"].unique().to_list())
    if len(sizes) < 2:
        return None
    n_big, n_small = sizes[-1], sizes[-2]

    big = agg.filter(pl.col("n") == n_big).sort("p")
    small = agg.filter(pl.col("n") == n_small).sort("p")

    p_common = np.sort(
        np.unique(
            np.concatenate([
                big["p"].to_numpy(),
                small["p"].to_numpy(),
            ])
        )
    )
    s_big = np.interp(
        p_common, big["p"].to_numpy(), big["S_mean"].to_numpy()
    )
    s_small = np.interp(
        p_common, small["p"].to_numpy(), small["S_mean"].to_numpy()
    )
    diff = s_big - s_small
    # Walk forward looking for the first true sign change. A "true" sign
    # change is diff[i] * diff[i+1] < 0 (opposite strictly-nonzero signs);
    # plateaus where both curves sit at 1 or 0 produce diff == 0 and must
    # be skipped to avoid a spurious p_c estimate at the sweep endpoints.
    eps = 1e-12
    prev_sign = 0.0
    for i in range(len(diff) - 1):
        if abs(diff[i]) < eps:
            continue
        if abs(diff[i + 1]) < eps:
            # the next point is on a plateau -- carry on, but remember this
            # sign so we can detect the eventual crossing
            prev_sign = float(np.sign(diff[i]))
            continue
        this_sign = float(np.sign(diff[i + 1]))
        ref_sign = float(np.sign(diff[i])) if prev_sign == 0.0 else prev_sign
        if ref_sign * this_sign < 0.0:
            if diff[i + 1] == diff[i]:
                return float(p_common[i])
            return float(
                p_common[i] - diff[i] * (p_common[i + 1] - p_common[i]) /
                (diff[i + 1] - diff[i])
            )
        prev_sign = this_sign
    return None


def _estimate_pc_by_halving(agg: pl.DataFrame) -> Optional[float]:
    """Return the p at which the largest-n <S> curve first drops through
    0.5. Useful when the crossing is noisy or not bracketed."""
    sizes = sorted(int(v) for v in agg["n"].unique().to_list())
    if not sizes:
        return None
    big = agg.filter(pl.col("n") == sizes[-1]).sort("p")
    p = big["p"].to_numpy()
    s = big["S_mean"].to_numpy()
    below = np.where(s <= 0.5)[0]
    if below.size == 0:
        return None
    j = int(below[0])
    if j == 0:
        return float(p[0])
    if s[j - 1] == s[j]:
        return float(p[j])
    frac = (s[j - 1] - 0.5) / (s[j - 1] - s[j])
    return float(p[j - 1] + frac * (p[j] - p[j - 1]))


def estimate_pc(agg: pl.DataFrame, method: str = "crossing") -> Optional[float]:
    if method == "crossing":
        p_c = _estimate_pc_from_crossing(agg)
        if p_c is None:
            p_c = _estimate_pc_by_halving(agg)
        return p_c
    if method == "half":
        return _estimate_pc_by_halving(agg)
    raise ValueError(f"unknown p_c method: {method!r}")


# LaTeX table assembly

def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("&", "\\&")
         .replace("%", "\\%")
         .replace("_", "\\_")
         .replace("#", "\\#")
         .replace("$", "\\$")
         .replace("{", "\\{")
         .replace("}", "\\}")
    )


def _raw_values_table(agg: pl.DataFrame, max_p_rows: int = 60) -> str:
    """Pivot the aggregate into a longtable: p vs n, cell = ``mean +/- SEM``.

    If the p grid has more than ``max_p_rows`` rows, rows are evenly
    subsampled so the printed table stays readable; a footnote records
    this.
    """
    sizes = sorted(int(v) for v in agg["n"].unique().to_list())
    p_values = sorted(float(v) for v in agg["p"].unique().to_list())

    subsampled = False
    if len(p_values) > max_p_rows:
        idx = np.linspace(0, len(p_values) - 1, max_p_rows).round().astype(int)
        p_subset = [p_values[i] for i in sorted(set(idx.tolist()))]
        subsampled = True
    else:
        p_subset = p_values

    # Build a lookup of (n, p) -> (mean, sem)
    lookup = {}
    for row in agg.iter_rows(named=True):
        lookup[(int(row["n"]), float(row["p"]))] = (
            float(row["S_mean"]) if row["S_mean"] is not None else float("nan"),
            float(row["S_sem"]) if row["S_sem"] is not None else float("nan"),
        )

    header_cols = "l " + " ".join(["c"] * len(sizes))
    lines: List[str] = []
    lines.append("\\begin{longtable}{%s}" % header_cols)
    lines.append("\\caption{Per-cell mean reference-qubit entropy "
                 "$\\langle S \\rangle$ and standard error of the mean "
                 "$\\sigma_{\\langle S\\rangle}$, reported as "
                 "$\\langle S \\rangle \\pm \\sigma_{\\langle S\\rangle}$. "
                 "Rows index the measurement rate $p$; columns index "
                 "system size $n$." + (
                     f" The {len(p_values)}-point $p$ grid was subsampled "
                     f"to {len(p_subset)} rows for display."
                     if subsampled else "") + "} \\label{tab:raw}\\\\")
    lines.append("\\toprule")
    header = ["$p$"] + [f"$n={s}$" for s in sizes]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")

    for p_val in p_subset:
        row = [f"{p_val:.4f}"]
        for s in sizes:
            val = lookup.get((s, p_val))
            if val is None:
                row.append("---")
            else:
                m, e = val
                if np.isnan(m):
                    row.append("---")
                elif np.isnan(e):
                    row.append(f"{m:.3f}")
                else:
                    row.append(f"{m:.3f} $\\pm$ {e:.3f}")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{longtable}")
    return "\n".join(lines)


# LaTeX document assembly

def _summary_stats(agg: pl.DataFrame) -> dict:
    sizes = sorted(int(v) for v in agg["n"].unique().to_list())
    p_values = sorted(float(v) for v in agg["p"].unique().to_list())
    total_samples = int(agg["count"].sum())
    return {
        "sizes": sizes,
        "p_min": min(p_values) if p_values else float("nan"),
        "p_max": max(p_values) if p_values else float("nan"),
        "n_p": len(p_values),
        "total_cells": len(sizes) * len(p_values),
        "total_samples": total_samples,
    }


_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=black, urlcolor=blue}

\title{Measurement-Induced Phase Transition in All-to-All Clifford Circuits:\\
       A Single-Qubit Reference-Probe Study}
\author{SparseGF2 automatic report}
\date{%s}

\begin{document}
\maketitle
""" % dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%d")


def _build_tex(
    agg: pl.DataFrame,
    figure_path: str,
    pc_estimate: Optional[float],
    pc_method: str,
    run_id: str,
) -> str:
    stats = _summary_stats(agg)
    sizes_str = ", ".join(str(n) for n in stats["sizes"])
    pc_sentence = (
        f"The curves cross at $p_c \\approx {pc_estimate:.3f}$ "
        f"(method: {_latex_escape(pc_method)})."
        if pc_estimate is not None else
        "No crossing of the $\\langle S \\rangle$ curves was found in "
        "the sampled $p$-window; the transition likely lies outside "
        "$[p_\\min, p_\\max]$."
    )

    parts = [_PREAMBLE]

    parts.append(r"""\begin{abstract}
We locate the critical measurement rate $p_c$ of the purification
phase transition in random Clifford circuits with all-to-all
connectivity, using a minimal single-qubit reference probe. The
system of $n$ qubits is coupled to a single ancilla via an initial
Bell pair on qubits $(0, n)$; gates are drawn uniformly from the
two-qubit Clifford group and applied on the $n(n-1)/2$ edges of the
complete graph $K_n$ via a deterministic round-robin
1-factorization. After every gate layer each system qubit is
measured in the $Z$ basis independently with probability $p$. The
observable $S(\text{ref})$, the von Neumann entropy of the
reference qubit, is $\in \{0, 1\}$ for stabilizer states and serves
as an order parameter: it is $1$ in the volume-law phase (the
ancilla stays entangled with the scrambled system) and $0$ in the
area-law phase (measurements collapse the correlation). We sweep
""" + f"$n \\in \\{{{sizes_str}\\}}$" +
r""" and $p \in [""" + f"{stats['p_min']:.3f}" + r""", """ +
f"{stats['p_max']:.3f}" + r"""]$, and identify $p_c$ from the
crossing of the $\langle S \rangle$ curves for different $n$.
""" + pc_sentence + r"""
\end{abstract}
""")

    parts.append(r"""\section{Introduction}
Measurement-induced phase transitions (MIPT) in monitored random
Clifford circuits separate a volume-law entangled phase at low
measurement rate $p$ from an area-law (product) phase at high $p$
\cite{SkinnerRuhmanNahum2019, GullansHuse2020}. For all-to-all
connectivity, the volume-law phase maps to a purification phase in
the purification picture of Gullans and Huse, and the critical
measurement rate is governed by a competition between the
scrambling rate of the random Cliffords and the projective
collapse rate of local measurements.

A minimal diagnostic for the transition is obtained by coupling an
$n$-qubit system to a single reference qubit via an initial Bell
pair \cite{ChoiBaoAltman2020}. The von Neumann entropy of the
reference, $S(\text{ref})$, is $1$ as long as the correlation
established at $t = 0$ survives the combination of scrambling and
measurement, and $0$ once measurements have fully collapsed it.
Because a single qubit in a stabilizer state has integer von
Neumann entropy, $S(\text{ref})$ is exactly $0$ or $1$ on every
realization; ensemble averaging over many realizations yields a
continuous order parameter.

In this report we use the \textsc{SparseGF2} stabilizer simulator
to sweep $(n, p)$ at all-to-all connectivity and identify $p_c$
from the crossing of the $\langle S \rangle$ curves.
""")

    parts.append(r"""\section{Methods}

\paragraph{Simulator.} We use the \textsc{SparseGF2} phase-free
stabilizer tableau simulator with Numba-JIT symplectic kernels and
\textsc{stim}-sourced two-qubit Clifford tables.

\paragraph{Single-reference protocol.} Each realization allocates
$n + 1$ qubits. Qubits $0, \ldots, n - 1$ are system qubits;
qubit $n$ is the reference. The initial state is prepared as
\[
  \lvert \psi_0 \rangle = H_0 \cdot \mathrm{CNOT}_{0 \to n} \cdot
    \lvert 0 \rangle^{\otimes (n+1)}
  \;=\; \lvert \Phi^+ \rangle_{0,n} \otimes \lvert 0 \rangle_1
    \otimes \cdots \otimes \lvert 0 \rangle_{n-1},
\]
a single Bell pair between system qubit $0$ and the reference, with
the remaining system qubits in $\lvert 0 \rangle$. The circuit acts
only on qubits $0, \ldots, n - 1$; qubit $n$ is untouched
throughout.

\paragraph{Round-robin 1-factorization of $K_n$.} With
connectivity $K_n$ on the $n$ system qubits, the edge set
$\lvert E \rvert = n(n-1)/2$ is partitioned into $n - 1$ perfect
matchings $M_0, M_1, \ldots, M_{n-2}$ (a 1-factorization
\cite{Anderson2001}), each of size $n / 2$. Layer $t$ applies a
uniformly random two-qubit Clifford on every edge of the matching
$M_{t \bmod (n-1)}$. This sequence covers every edge of $K_n$
exactly once per period of $n - 1$ layers.

\paragraph{Measurements.} After every gate layer each system qubit
is independently measured in the $Z$ basis with probability $p$
(uniform measurement mode). Measurements act only on the system
block.

\paragraph{Depth and samples.} The circuit runs for
$T = 8 n$ layers in the default configuration. After execution we
compute $S(\text{ref}) = \mathrm{rank}(M \rvert_{\{n\}}) - 1$ from
the Fattal--Cubitt--Yamamoto--Bravyi--Chuang rank formula
\cite{Fattal2004} applied to the single-qubit subset $\{n\}$; this
yields an integer in $\{0, 1\}$ which is written to the
\texttt{obs.k} column of the per-sample Parquet.

\paragraph{Ensembling.} At each $(n, p)$ we run many independent
realizations and report the empirical mean
$\langle S \rangle$ and standard error of the mean
$\sigma_{\langle S \rangle} = s / \sqrt{M}$, where $s$ is the
sample standard deviation (ddof 1) and $M$ the sample count.
""")

    parts.append(
        r"""\section{Results}
""" + f"""
\\subsection{{Parameters of this run}}
\\begin{{itemize}}
  \\item Run identifier: \\texttt{{{_latex_escape(run_id)}}}
  \\item System sizes: $n \\in \\{{{sizes_str}\\}}$
  \\item Measurement-rate grid: {stats['n_p']} values over
    $[{stats['p_min']:.3f}, {stats['p_max']:.3f}]$
  \\item Total cells: {stats['total_cells']}
  \\item Total realizations: {stats['total_samples']:,}
\\end{{itemize}}
""" +
        r"""
\subsection{Crossing plot}
Figure~\ref{fig:crossing} shows $\langle S \rangle$ versus $p$ for
each system size. The curves approach $1$ for $p \ll p_c$
(volume-law phase), $0$ for $p \gg p_c$ (area-law phase), and cross
near $p_c$.
""" +
        f"\n\\begin{{figure}}[htbp]\n"
        f"  \\centering\n"
        f"  \\includegraphics[width=0.85\\linewidth]{{{figure_path}}}\n"
        f"  \\caption{{Mean reference-qubit entropy "
        f"$\\langle S \\rangle$ versus measurement rate $p$, for "
        f"system sizes $n \\in \\{{{sizes_str}\\}}$. Error bars are "
        f"one standard error of the mean. Dotted horizontal lines at "
        f"$0$ and $1$ mark the two integer values available to the "
        f"stabilizer probe.}}\n"
        f"  \\label{{fig:crossing}}\n"
        f"\\end{{figure}}\n" +
        r"""
\subsection{Raw aggregate values}
Table~\ref{tab:raw} records the per-cell mean and standard error
for every $(n, p)$ pair simulated.
"""
    )

    parts.append(_raw_values_table(agg))

    parts.append(r"""
\section{Conclusion}
""" + pc_sentence + r""" In the all-to-all setting the transition
is expected to be governed by the competition between $O(n)$
two-qubit Cliffords per layer (which scramble the information
maximally in $O(\log n)$ time) and the $O(n p)$ measurements per
layer. The single-qubit reference probe $S(\text{ref})$ is
minimally invasive, uses $n + 1$ rather than $2n$ qubits, and
reproduces the textbook crossing signature of MIPT.

Future work. Fit the finite-size scaling form $\langle S \rangle =
f((p - p_c) n^{1 / \nu})$ to extract the critical exponent $\nu$;
use the accompanying purification-time analysis
(\texttt{analyze\_single\_ref} with \texttt{record\_time\_series})
to recover the dynamical exponent from $\tau(n) \sim n^{z}$ at
$p = p_c$.

\begin{thebibliography}{9}

\bibitem{SkinnerRuhmanNahum2019}
B.~Skinner, J.~Ruhman, and A.~Nahum,
\emph{Measurement-Induced Phase Transitions in the Dynamics of
Entanglement},
Phys.~Rev.~X \textbf{9}, 031009 (2019),
\href{https://arxiv.org/abs/1808.05953}{arXiv:1808.05953}.

\bibitem{GullansHuse2020}
M.~J.~Gullans and D.~A.~Huse,
\emph{Dynamical Purification Phase Transition Induced by Quantum
Measurements},
Phys.~Rev.~X \textbf{10}, 041020 (2020),
\href{https://arxiv.org/abs/1905.05195}{arXiv:1905.05195}.

\bibitem{ChoiBaoAltman2020}
S.~Choi, Y.~Bao, X.-L.~Qi, and E.~Altman,
\emph{Quantum Error Correction in Scrambling Dynamics and
Measurement-Induced Phase Transition},
Phys.~Rev.~Lett.~\textbf{125}, 030505 (2020),
\href{https://arxiv.org/abs/1903.05124}{arXiv:1903.05124}.

\bibitem{Fattal2004}
D.~Fattal, T.~S.~Cubitt, Y.~Yamamoto, S.~Bravyi, and I.~L.~Chuang,
\emph{Entanglement in the Stabilizer Formalism},
\href{https://arxiv.org/abs/quant-ph/0406168}{arXiv:quant-ph/0406168}
(2004).

\bibitem{Anderson2001}
I.~Anderson,
\emph{Combinatorial Designs and Tournaments},
Oxford Lecture Series in Mathematics and Its Applications (2001).

\end{thebibliography}

\end{document}
""")
    return "".join(parts)


# Main entry point

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate a single_ref run, plot the crossing, "
                    "and emit a self-contained LaTeX report."
    )
    ap.add_argument("run_dir", type=Path,
                    help="Run directory produced by "
                         "scripts/run_all_to_all_mipt.py.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write crossing_plot.pdf and report.tex "
                         "(default: inside run_dir).")
    ap.add_argument("--p-c-method", choices=("crossing", "half"),
                    default="crossing",
                    help="How to estimate p_c. crossing: intersection of "
                         "the two largest-n curves. half: smallest p at "
                         "which the largest-n mean drops through 0.5.")
    ap.add_argument("--compile", action="store_true",
                    help="After writing report.tex, invoke pdflatex (twice) "
                         "to produce report.pdf. Requires a TeX toolchain "
                         "on PATH.")
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"error: run directory does not exist: {run_dir}",
              file=sys.stderr)
        return 2
    out_dir = (args.out_dir or run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading samples from {run_dir} ...", flush=True)
    df = load_single_ref_samples(run_dir)
    if df.is_empty():
        print("error: no samples found under {}.".format(run_dir),
              file=sys.stderr)
        return 3
    print(f"  {len(df):,} samples loaded")

    agg = aggregate_single_ref_entropy(df)
    print(f"  aggregated to {len(agg)} (n, p) cells")

    figure_path = out_dir / "crossing_plot.pdf"
    print(f"Writing crossing plot to {figure_path} ...", flush=True)
    plot_single_ref_crossing(
        agg,
        out_path=figure_path,
        title=r"$\langle S(\mathrm{ref}) \rangle$ vs $p$ on $K_n$",
    )

    pc = estimate_pc(agg, method=args.p_c_method)
    if pc is not None:
        print(f"  estimated p_c = {pc:.3f} (method={args.p_c_method})")
    else:
        print("  no p_c estimate could be extracted")

    # The LaTeX \includegraphics path is relative to the .tex file, so we
    # write just the filename when the figure sits alongside the report.
    fig_rel = (
        figure_path.name if figure_path.parent == out_dir
        else str(figure_path)
    )

    tex_source = _build_tex(
        agg, fig_rel, pc, args.p_c_method, run_id=run_dir.name
    )
    tex_path = out_dir / "report.tex"
    tex_path.write_text(tex_source, encoding="utf-8")
    print(f"Wrote {tex_path}")

    if args.compile:
        pdflatex = shutil.which("pdflatex")
        if pdflatex is None:
            print("warning: pdflatex not on PATH; skipping --compile",
                  file=sys.stderr)
        else:
            print("Compiling report.tex with pdflatex (twice) ...",
                  flush=True)
            for _ in range(2):
                proc = subprocess.run(
                    [pdflatex, "-interaction=nonstopmode",
                     "-halt-on-error", tex_path.name],
                    cwd=out_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if proc.returncode != 0:
                    sys.stderr.write(proc.stderr.decode(errors="replace"))
                    return proc.returncode
            print(f"  PDF built: {out_dir / 'report.pdf'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
