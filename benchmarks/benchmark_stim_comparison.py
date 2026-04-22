#!/usr/bin/env python3
"""
Direct SparseGF2 vs Stim runtime comparison at selected system sizes.

Runs identical random Clifford circuits through both simulators and
measures wall-clock time as a function of p. Produces the Stim reference
curve for the runtime vs p plot.

Uses Stim's TableauSimulator.do_tableau_1q/do_tableau_2q for efficient
gate application (avoids circuit string parsing overhead).

Usage:
    cd SparseGF2
    py -3.13 benchmarks/benchmark_stim_comparison.py
"""
import sys, time, json, pickle
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import stim
from sparsegf2.core.sparse_tableau import SparseGF2, warmup
from sparsegf2.gates.clifford import symplectic_from_stim_tableau

OUT_DIR = REPO_ROOT / "benchmarks" / "results"
OUT_DIR.mkdir(exist_ok=True)

# Sizes small enough that Stim finishes in reasonable time
SIZES = [32, 64, 128, 256, 512]
P_VALUES = sorted(set(
    [round(x, 3) for x in np.arange(0.02, 0.10, 0.02)] +
    [round(x, 3) for x in np.arange(0.10, 0.22, 0.01)] +
    [round(x, 3) for x in np.arange(0.22, 0.42, 0.02)]
))
SEEDS = 5


def main():
    print("Building Clifford cache...", flush=True)
    N_CLIFF = 4096
    rng_cliff = np.random.default_rng(12345)
    TABS = []
    SYMP = np.zeros((N_CLIFF, 4, 4), dtype=np.uint8)
    for i in range(N_CLIFF):
        tab = stim.Tableau.random(2)
        TABS.append(tab)
        SYMP[i] = symplectic_from_stim_tableau(tab)
    print(f"  {N_CLIFF} random 2-qubit Cliffords", flush=True)

    warmup()
    print("Ready.\n", flush=True)

    results = []
    for n in SIZES:
        print(f"n = {n}:", flush=True)
        print(f"  {'p':>6} {'sparse':>10} {'stim':>10} {'ratio':>8} {'k':>4}", flush=True)

        for p in P_VALUES:
            ts_sparse = []
            ts_stim = []
            k_val = None

            for seed in range(SEEDS):
                rng = np.random.default_rng(seed)
                T = 8 * n

                # Pre-generate schedule
                schedule = []
                for t in range(T):
                    m = t % 2
                    n_pairs = n // 2
                    ci = rng.integers(0, N_CLIFF, size=n_pairs)
                    pairs = []
                    for j in range(n_pairs):
                        qi = 2*j if m == 0 else 2*j+1
                        qj = 2*j+1 if m == 0 else (2*j+2) % n
                        pairs.append((qi, qj))
                    meas = []
                    for qi, qj in pairs:
                        if rng.random() < p: meas.append(qi)
                        if rng.random() < p: meas.append(qj)
                    schedule.append((pairs, ci, sorted(set(meas))))

                # --- SparseGF2 ---
                sim_s = SparseGF2(n, use_min_weight_pivot=True)
                t0 = time.perf_counter()
                for pairs, ci, meas in schedule:
                    for ip, (qi, qj) in enumerate(pairs):
                        sim_s.apply_gate(qi, qj, SYMP[ci[ip]])
                    for mq in meas:
                        sim_s.apply_measurement_z(mq)
                ts_sparse.append(time.perf_counter() - t0)
                if k_val is None:
                    k_val = sim_s.compute_k()

                # --- Stim ---
                sim_st = stim.TableauSimulator()
                for i in range(n):
                    sim_st.h(i)
                    sim_st.cx(i, n + i)
                t0 = time.perf_counter()
                for pairs, ci, meas in schedule:
                    for ip, (qi, qj) in enumerate(pairs):
                        sim_st.do_tableau(TABS[ci[ip]], [qi, qj])
                    for mq in meas:
                        sim_st.measure(mq)
                        sim_st.reset(mq)
                ts_stim.append(time.perf_counter() - t0)

            t_sp = np.mean(ts_sparse)
            t_st = np.mean(ts_stim)
            ratio = t_sp / t_st if t_st > 0 else float('inf')
            print(f"  {p:>6.3f} {t_sp:>10.4f} {t_st:>10.4f} {ratio:>8.3f} {k_val:>4}", flush=True)

            results.append({
                "n": n, "p": p, "t_sparse": t_sp, "t_stim": t_st,
                "ratio": ratio, "k": int(k_val),
            })

    out = OUT_DIR / "stim_comparison.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
