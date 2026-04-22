#!/usr/bin/env python3
"""
Comprehensive benchmark figure suite for SparseGF2.

Generates ~15 publication-quality figures covering every angle of the
computational phase transition.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent
DATA = BASE / "results" / "mipt_runtime_benchmark.json"
STIM_DATA = BASE / "results" / "stim_comparison.json"
FIG = BASE / "results" / "figures"
FIG.mkdir(exist_ok=True)

with open(DATA) as f:
    raw = json.load(f)
data = [r for r in raw if "error" not in r and r["time"] > 0]

stim_data = []
if STIM_DATA.exists():
    with open(STIM_DATA) as f:
        stim_data = json.load(f)

print(f"Main data: {len(data)} records, Stim data: {len(stim_data)} records")

# ── Aggregation ───────────────────────────────────────────────

def agg_by(records, keys):
    groups = defaultdict(list)
    for r in records:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key_vals, recs in sorted(groups.items()):
        row = dict(zip(keys, key_vals))
        ts = [r["time"] for r in recs]
        ks = [r["k"] for r in recs]
        abars = [r["abar"] for r in recs if r["abar"] > 0]
        row["t_mean"] = np.mean(ts)
        row["t_std"] = np.std(ts)
        row["t_median"] = np.median(ts)
        row["k_mean"] = np.mean(ks)
        row["k_std"] = np.std(ks)
        row["Pk"] = np.mean([1 if k > 0 else 0 for k in ks])
        row["abar_mean"] = np.mean(abars) if abars else 0
        row["abar_std"] = np.std(abars) if abars else 0
        row["rate"] = np.mean([k / row["n"] for k in ks]) if row["n"] > 0 else 0
        row["count"] = len(recs)
        out.append(row)
    return out

agg = agg_by(data, ["model", "n", "p"])

CMAP = plt.cm.viridis
def cn(n, sizes):
    return CMAP(sizes.index(n) / max(1, len(sizes) - 1))

def savefig(fig, name):
    # Write both a publication-quality PDF and a web-friendly PNG so that
    # GitHub renders the figures inline in the README.
    path = FIG / name
    fig.savefig(path, dpi=200, bbox_inches='tight')
    png_path = path.with_suffix('.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {name}")


print("\nGenerating figures...")

# 1. Runtime vs p with P(k>0) (NN)
for model in ["NN", "ATA"]:
    ma = [r for r in agg if r["model"] == model]
    if not ma: continue
    sizes = sorted(set(r["n"] for r in ma))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    for n in sizes:
        rows = sorted([r for r in ma if r["n"] == n], key=lambda r: r["p"])
        ax1.semilogy([r["p"] for r in rows], [r["t_mean"] for r in rows],
                     '-o', color=cn(n, sizes), ms=2.5, lw=1.2, label=f"n={n}")
        ax2.plot([r["p"] for r in rows], [r["Pk"] for r in rows],
                 '-o', color=cn(n, sizes), ms=2.5, lw=1.2, label=f"n={n}")
    ax1.axvline(0.16, color='red', ls=':', alpha=0.4)
    ax1.set_ylabel("Runtime (s)"); ax1.set_title(f"{model}: Runtime vs $p$")
    ax1.legend(fontsize=7, ncol=3, loc='upper right'); ax1.grid(alpha=0.2)
    ax2.axvline(0.16, color='red', ls=':', alpha=0.4)
    ax2.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax2.set_xlabel("$p$"); ax2.set_ylabel("$P(k>0)$"); ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=7, ncol=3); ax2.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, f"fig_runtime_Pk_vs_p_{model.lower()}.pdf")

# 2. abar vs p for multiple n (NN and ATA)
for model in ["NN", "ATA"]:
    ma = [r for r in agg if r["model"] == model]
    if not ma: continue
    sizes = sorted(set(r["n"] for r in ma))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for n in sizes:
        rows = sorted([r for r in ma if r["n"] == n], key=lambda r: r["p"])
        ps = [r["p"] for r in rows]
        abars = [r["abar_mean"] for r in rows]
        ax1.semilogy(ps, abars, '-o', color=cn(n, sizes), ms=2.5, lw=1.2, label=f"n={n}")
        # Normalized abar/n
        ax2.plot(ps, [a / n for a in abars], '-o', color=cn(n, sizes), ms=2.5, lw=1.2, label=f"n={n}")
    for ax in [ax1, ax2]:
        ax.axvline(0.16, color='red', ls=':', alpha=0.4)
        ax.grid(alpha=0.2)
    ax1.set_xlabel("$p$"); ax1.set_ylabel(r"$\bar{a}$")
    ax1.set_title(f"{model}: Active Count vs $p$"); ax1.legend(fontsize=7, ncol=3)
    ax2.set_xlabel("$p$"); ax2.set_ylabel(r"$\bar{a}/n$")
    ax2.set_title(f"{model}: Normalized Active Count"); ax2.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    savefig(fig, f"fig_abar_vs_p_{model.lower()}.pdf")

# 3. Scaling exponent gamma(p) for both models side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, model in enumerate(["NN", "ATA"]):
    ax = axes[idx]
    ma = [r for r in agg if r["model"] == model]
    sizes = sorted(set(r["n"] for r in ma))
    ps_all = sorted(set(r["p"] for r in ma))
    gammas_t, gammas_a, ps_fit = [], [], []
    for p in ps_all:
        rows = {r["n"]: r for r in ma if r["p"] == p}
        ns = sorted([n for n in sizes if n in rows and rows[n]["t_mean"] > 0])
        if len(ns) < 4: continue
        log_n = np.log(np.array(ns, dtype=float))
        log_t = np.log([rows[n]["t_mean"] for n in ns])
        log_a = np.log([max(rows[n]["abar_mean"], 0.1) for n in ns])
        gt, _ = np.polyfit(log_n, log_t, 1)
        ga, _ = np.polyfit(log_n, log_a, 1)
        gammas_t.append(gt); gammas_a.append(ga); ps_fit.append(p)
    ax.plot(ps_fit, gammas_t, 'b-o', ms=4, lw=1.5, label=r'$\gamma_t$: $t \sim n^{\gamma_t}$')
    ax.plot(ps_fit, gammas_a, 'g-s', ms=4, lw=1.5, label=r'$\alpha$: $\bar{a} \sim n^\alpha$')
    ax.axvline(0.16, color='red', ls=':', alpha=0.4, label=r'$p_c$')
    ax.axhline(2.0, color='cyan', ls=':', alpha=0.3, label=r'$\gamma=2$')
    ax.axhline(3.0, color='orange', ls=':', alpha=0.3, label=r'$\gamma=3$')
    ax.axhline(0.0, color='gray', ls=':', alpha=0.2)
    ax.set_xlabel("$p$"); ax.set_ylabel("Exponent")
    ax.set_title(f"{model}: Scaling Exponents"); ax.legend(fontsize=8)
    ax.set_ylim(-0.5, 4.5); ax.grid(alpha=0.2)
fig.tight_layout()
savefig(fig, "fig_scaling_exponents_gamma_alpha.pdf")

# 4. Log-log runtime vs n at selected p
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sel_ps = [0.04, 0.10, 0.155, 0.20, 0.30]
mks = ['v', 's', 'D', 'o', '^']
for idx, model in enumerate(["NN", "ATA"]):
    ax = axes[idx]
    ma = [r for r in agg if r["model"] == model]
    sizes = sorted(set(r["n"] for r in ma))
    for ip, p in enumerate(sel_ps):
        pm = min(set(r["p"] for r in ma), key=lambda x: abs(x-p))
        rows = sorted([r for r in ma if r["p"]==pm], key=lambda r: r["n"])
        ns = [r["n"] for r in rows]; ts = [r["t_mean"] for r in rows]
        if len(ns) < 2: continue
        ls = ':' if p < 0.15 else ('-' if p > 0.18 else '--')
        ax.loglog(ns, ts, ls+mks[ip], ms=5, lw=1.5, label=f"p={pm:.3f}")
    nr = np.array([min(sizes), max(sizes)]); t0 = 1e-4
    ax.loglog(nr, t0*(nr/nr[0])**2, 'k:', alpha=0.15)
    ax.loglog(nr, t0*(nr/nr[0])**3, 'k--', alpha=0.15)
    ax.text(nr[-1]*0.6, t0*(nr[-1]/nr[0])**2*0.4, '$n^2$', fontsize=9, alpha=0.3)
    ax.text(nr[-1]*0.6, t0*(nr[-1]/nr[0])**3*0.4, '$n^3$', fontsize=9, alpha=0.3)
    ax.set_xlabel("$n$"); ax.set_ylabel("Runtime (s)")
    ax.set_title(f"{model}"); ax.legend(fontsize=9); ax.grid(alpha=0.2, which='both')
fig.tight_layout()
savefig(fig, "fig_loglog_runtime_vs_n.pdf")

# 5. Heatmap: runtime(n, p) for NN
for model in ["NN", "ATA"]:
    ma = [r for r in agg if r["model"] == model]
    sizes = sorted(set(r["n"] for r in ma))
    ps = sorted(set(r["p"] for r in ma))
    Z = np.full((len(sizes), len(ps)), np.nan)
    for r in ma:
        i = sizes.index(r["n"]); j = ps.index(r["p"])
        Z[i, j] = r["t_mean"]
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.pcolormesh(ps, range(len(sizes)), Z, norm=LogNorm(), cmap='viridis', shading='nearest')
    ax.set_yticks(range(len(sizes))); ax.set_yticklabels(sizes)
    ax.axvline(0.16, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.set_xlabel("$p$"); ax.set_ylabel("$n$")
    ax.set_title(f"{model}: Runtime Heatmap")
    plt.colorbar(im, ax=ax, label="Runtime (s)")
    fig.tight_layout()
    savefig(fig, f"fig_heatmap_runtime_{model.lower()}.pdf")

# 6. Heatmap: abar(n, p)
for model in ["NN", "ATA"]:
    ma = [r for r in agg if r["model"] == model]
    sizes = sorted(set(r["n"] for r in ma))
    ps = sorted(set(r["p"] for r in ma))
    Z = np.full((len(sizes), len(ps)), np.nan)
    for r in ma:
        i = sizes.index(r["n"]); j = ps.index(r["p"])
        Z[i, j] = r["abar_mean"]
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.pcolormesh(ps, range(len(sizes)), Z, norm=LogNorm(), cmap='inferno', shading='nearest')
    ax.set_yticks(range(len(sizes))); ax.set_yticklabels(sizes)
    ax.axvline(0.16, color='cyan', ls='--', lw=1.5, alpha=0.7)
    ax.set_xlabel("$p$"); ax.set_ylabel("$n$")
    ax.set_title(f"{model}: Active Count $\\bar{{a}}$ Heatmap")
    plt.colorbar(im, ax=ax, label=r"$\bar{a}$")
    fig.tight_layout()
    savefig(fig, f"fig_heatmap_abar_{model.lower()}.pdf")

# 7. Code rate k/n vs p
for model in ["NN", "ATA"]:
    ma = [r for r in agg if r["model"] == model]
    sizes = sorted(set(r["n"] for r in ma))
    fig, ax = plt.subplots(figsize=(10, 5))
    for n in sizes:
        rows = sorted([r for r in ma if r["n"]==n], key=lambda r: r["p"])
        ax.plot([r["p"] for r in rows], [r["rate"] for r in rows],
                '-o', color=cn(n, sizes), ms=2.5, lw=1.2, label=f"n={n}")
    ax.axvline(0.16, color='red', ls=':', alpha=0.4)
    ax.set_xlabel("$p$"); ax.set_ylabel("$k/n$")
    ax.set_title(f"{model}: Code Rate vs $p$")
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, f"fig_code_rate_vs_p_{model.lower()}.pdf")

# 8. Runtime per layer vs p (removes depth factor)
import math
for model in ["NN", "ATA"]:
    ma = [r for r in agg if r["model"] == model]
    sizes = sorted(set(r["n"] for r in ma))
    fig, ax = plt.subplots(figsize=(10, 5))
    for n in sizes:
        rows = sorted([r for r in ma if r["n"]==n], key=lambda r: r["p"])
        if model == "NN":
            D = 8 * n
        else:
            D = 8 * max(1, int(math.ceil(math.log2(n))))
        ax.semilogy([r["p"] for r in rows], [r["t_mean"]/D for r in rows],
                    '-o', color=cn(n, sizes), ms=2.5, lw=1.2, label=f"n={n}")
    ax.axvline(0.16, color='red', ls=':', alpha=0.4)
    ax.set_xlabel("$p$"); ax.set_ylabel("Runtime per layer (s)")
    ax.set_title(f"{model}: Per-Layer Cost vs $p$")
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, f"fig_per_layer_cost_vs_p_{model.lower()}.pdf")

# 9. abar vs n (log-log) at selected p — shows alpha exponent
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, model in enumerate(["NN", "ATA"]):
    ax = axes[idx]
    ma = [r for r in agg if r["model"] == model]
    sizes = sorted(set(r["n"] for r in ma))
    for ip, p in enumerate(sel_ps):
        pm = min(set(r["p"] for r in ma), key=lambda x: abs(x-p))
        rows = sorted([r for r in ma if r["p"]==pm], key=lambda r: r["n"])
        ns = [r["n"] for r in rows]; abars = [r["abar_mean"] for r in rows]
        if len(ns) < 2: continue
        ls = ':' if p < 0.15 else ('-' if p > 0.18 else '--')
        ax.loglog(ns, abars, ls+mks[ip], ms=5, lw=1.5, label=f"p={pm:.3f}")
    nr = np.array([min(sizes), max(sizes)])
    ax.loglog(nr, 3*np.ones_like(nr), 'k:', alpha=0.2)
    ax.text(nr[-1]*0.7, 3.5, '$O(1)$', fontsize=9, alpha=0.4)
    ax.loglog(nr, 0.5*(nr)**0.9, 'k--', alpha=0.15)
    ax.text(nr[-1]*0.5, 0.5*nr[-1]**0.9*0.5, '$n^{0.9}$', fontsize=9, alpha=0.3)
    ax.set_xlabel("$n$"); ax.set_ylabel(r"$\bar{a}$")
    ax.set_title(f"{model}: Active Count Scaling"); ax.legend(fontsize=9)
    ax.grid(alpha=0.2, which='both')
fig.tight_layout()
savefig(fig, "fig_abar_vs_n_loglog.pdf")

# 10. Runtime / n^2 vs p (normalized — flat in area-law = O(n^2))
for model in ["NN", "ATA"]:
    ma = [r for r in agg if r["model"] == model]
    sizes = sorted(set(r["n"] for r in ma))
    fig, ax = plt.subplots(figsize=(10, 5))
    for n in sizes:
        rows = sorted([r for r in ma if r["n"]==n], key=lambda r: r["p"])
        ax.semilogy([r["p"] for r in rows], [r["t_mean"]/(n**2) for r in rows],
                    '-o', color=cn(n, sizes), ms=2.5, lw=1.2, label=f"n={n}")
    ax.axvline(0.16, color='red', ls=':', alpha=0.4)
    ax.set_xlabel("$p$"); ax.set_ylabel("Runtime / $n^2$")
    ax.set_title(f"{model}: Runtime Normalized by $n^2$ (flat in area-law = $O(n^2)$)")
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, f"fig_runtime_over_n2_{model.lower()}.pdf")

# 11. Stim comparison: runtime vs p with speedup
if stim_data:
    sizes_s = sorted(set(r["n"] for r in stim_data))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    for n in sizes_s:
        rows = sorted([r for r in stim_data if r["n"]==n], key=lambda r: r["p"])
        ps = [r["p"] for r in rows]
        col = cn(n, sizes_s)
        ax1.semilogy(ps, [r["t_sparse"] for r in rows], '-o', color=col, ms=3, lw=1.5, label=f"SparseGF2 n={n}")
        ax1.semilogy(ps, [r["t_stim"] for r in rows], '--s', color=col, ms=3, lw=1, alpha=0.5, label=f"Stim n={n}")
        ratios = [r["t_stim"]/r["t_sparse"] for r in rows]
        ax2.plot(ps, ratios, '-o', color=col, ms=3, lw=1.5, label=f"n={n}")
    ax1.axvline(0.16, color='red', ls=':', alpha=0.4)
    ax1.set_ylabel("Runtime (s)"); ax1.set_title("NN: SparseGF2 vs Stim")
    ax1.legend(fontsize=6, ncol=3, loc='center left'); ax1.grid(alpha=0.2)
    ax2.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax2.axvline(0.16, color='red', ls=':', alpha=0.4)
    ax2.set_xlabel("$p$"); ax2.set_ylabel("Speedup (Stim/SparseGF2)")
    ax2.set_yscale('log'); ax2.legend(fontsize=8, ncol=2); ax2.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, "fig_stim_comparison_full.pdf")

# 12. Stim speedup vs n at fixed area-law p
if stim_data:
    fig, ax = plt.subplots(figsize=(8, 5))
    for p_sel in [0.20, 0.25, 0.30, 0.40]:
        pm = min(set(r["p"] for r in stim_data), key=lambda x: abs(x-p_sel))
        rows = sorted([r for r in stim_data if r["p"]==pm], key=lambda r: r["n"])
        ns = [r["n"] for r in rows]
        sp = [r["t_stim"]/r["t_sparse"] for r in rows]
        ax.plot(ns, sp, '-o', ms=5, lw=1.5, label=f"p={pm:.2f}")
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel("$n$"); ax.set_ylabel("Speedup (Stim / SparseGF2)")
    ax.set_title("Area-Law Speedup vs System Size")
    ax.legend(fontsize=10); ax.grid(alpha=0.2)
    ax.set_xscale('log'); ax.set_yscale('log')
    fig.tight_layout()
    savefig(fig, "fig_speedup_vs_n_area_law.pdf")

# 13. Runtime variance across seeds (error bars)
model = "NN"
ma = [r for r in agg if r["model"] == model]
sizes_sel = [64, 128, 256, 512]
fig, ax = plt.subplots(figsize=(10, 5))
for n in sizes_sel:
    rows = sorted([r for r in ma if r["n"]==n], key=lambda r: r["p"])
    ps = [r["p"] for r in rows]
    ts = [r["t_mean"] for r in rows]
    errs = [r["t_std"] for r in rows]
    ax.errorbar(ps, ts, yerr=errs, fmt='-o', ms=3, lw=1, capsize=2,
                label=f"n={n}", alpha=0.8)
ax.set_yscale('log')
ax.axvline(0.16, color='red', ls=':', alpha=0.4)
ax.set_xlabel("$p$"); ax.set_ylabel("Runtime (s)")
ax.set_title("NN: Runtime with Seed Variance (error bars = 1σ)")
ax.legend(fontsize=9); ax.grid(alpha=0.2)
fig.tight_layout()
savefig(fig, "fig_runtime_error_bars_nn.pdf")

# 14. Per-gate cost: runtime / (D * n/2) vs p
for model in ["NN"]:
    ma = [r for r in agg if r["model"] == model]
    sizes = sorted(set(r["n"] for r in ma))
    fig, ax = plt.subplots(figsize=(10, 5))
    for n in sizes:
        rows = sorted([r for r in ma if r["n"]==n], key=lambda r: r["p"])
        D = 8 * n
        n_gates = D * n // 2
        ax.semilogy([r["p"] for r in rows], [r["t_mean"]/n_gates for r in rows],
                    '-o', color=cn(n, sizes), ms=2.5, lw=1.2, label=f"n={n}")
    ax.axvline(0.16, color='red', ls=':', alpha=0.4)
    ax.set_xlabel("$p$"); ax.set_ylabel("Time per gate (s)")
    ax.set_title(f"NN: Per-Gate Cost vs $p$ (should approach $O(\\bar{{a}}/n)$ in area-law)")
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, "fig_per_gate_cost_nn.pdf")

# 15. NN vs ATA comparison at fixed n
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, n_sel in enumerate([64, 256, 512]):
    ax = axes[idx]
    for model, ls, col in [("NN", '-o', 'blue'), ("ATA", '--s', 'orange')]:
        rows = sorted([r for r in agg if r["model"]==model and r["n"]==n_sel],
                      key=lambda r: r["p"])
        if not rows: continue
        ax.semilogy([r["p"] for r in rows], [r["t_mean"] for r in rows],
                    ls, color=col, ms=3, lw=1.5, label=model)
    ax.axvline(0.16, color='red', ls=':', alpha=0.4)
    ax.set_xlabel("$p$"); ax.set_ylabel("Runtime (s)")
    ax.set_title(f"n = {n_sel}")
    ax.legend(fontsize=10); ax.grid(alpha=0.2)
fig.suptitle("NN vs ATA Runtime Comparison", fontsize=14, y=1.02)
fig.tight_layout()
savefig(fig, "fig_nn_vs_ata_comparison.pdf")

# 16. Heatmap: P(k>0) as phase diagram
for model in ["NN", "ATA"]:
    ma = [r for r in agg if r["model"] == model]
    sizes = sorted(set(r["n"] for r in ma))
    ps = sorted(set(r["p"] for r in ma))
    Z = np.full((len(sizes), len(ps)), np.nan)
    for r in ma:
        Z[sizes.index(r["n"]), ps.index(r["p"])] = r["Pk"]
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.pcolormesh(ps, range(len(sizes)), Z, cmap='RdYlBu_r', vmin=0, vmax=1, shading='nearest')
    ax.set_yticks(range(len(sizes))); ax.set_yticklabels(sizes)
    ax.axvline(0.16, color='black', ls='--', lw=2, alpha=0.7)
    ax.set_xlabel("$p$"); ax.set_ylabel("$n$")
    ax.set_title(f"{model}: Phase Diagram — $P(k>0)$")
    plt.colorbar(im, ax=ax, label="$P(k>0)$")
    fig.tight_layout()
    savefig(fig, f"fig_phase_diagram_{model.lower()}.pdf")

# 17. Combined: runtime + abar + Pk on 3-panel figure
for model in ["NN"]:
    ma = [r for r in agg if r["model"] == model]
    sizes = sorted(set(r["n"] for r in ma))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for n in sizes:
        rows = sorted([r for r in ma if r["n"]==n], key=lambda r: r["p"])
        ps = [r["p"] for r in rows]; col = cn(n, sizes)
        ax1.semilogy(ps, [r["t_mean"] for r in rows], '-', color=col, lw=1.2, label=f"n={n}")
        ax2.semilogy(ps, [r["abar_mean"] for r in rows], '-', color=col, lw=1.2, label=f"n={n}")
        ax3.plot(ps, [r["Pk"] for r in rows], '-', color=col, lw=1.2, label=f"n={n}")
    for ax in [ax1, ax2, ax3]:
        ax.axvline(0.16, color='red', ls=':', alpha=0.4)
        ax.grid(alpha=0.2)
    ax1.set_ylabel("Runtime (s)"); ax1.set_title(f"NN Model: Three Views of the MIPT")
    ax1.legend(fontsize=6, ncol=4, loc='upper right')
    ax2.set_ylabel(r"$\bar{a}$"); ax2.legend(fontsize=6, ncol=4, loc='upper right')
    ax3.set_ylabel("$P(k>0)$"); ax3.set_xlabel("$p$")
    ax3.set_ylim(-0.05, 1.05); ax3.legend(fontsize=6, ncol=4)
    fig.tight_layout()
    savefig(fig, "fig_three_panel_mipt_nn.pdf")

print(f"\nAll figures saved to {FIG}")
