#!/usr/bin/env python3
"""
SparseGF2 vs Stim benchmark with FULL two-qubit Clifford group (11,520 elements).

Runs identical NN brickwork circuits through both simulators with:
- Full 11,520-element Clifford group (deterministic via stim.Tableau.iter_all(2))
- RREF verification at every (n, p) point proving identical stabilizer groups
- System sizes: n = 64, 128, 256, 512
- Measurement rates: p = 0.02 to 0.40
- Depth: T = 8*n layers (O(n) depth mode, depth factor 8)
- NN brickwork on cycle graph C_n with periodic boundary conditions

Usage:
    cd SparseGF2
    py -3.13 benchmarks/benchmark_full_clifford_group.py
"""
import sys, time, json
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

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

SIZES = [64, 128, 256, 512]

# Fine sweep: dense around p_c ~ 0.16, coarser in the tails
P_VALUES = sorted(set(
    [round(x, 3) for x in np.arange(0.02, 0.10, 0.02)] +   # volume-law tail
    [round(x, 3) for x in np.arange(0.10, 0.22, 0.01)] +   # dense near p_c
    [round(x, 3) for x in np.arange(0.22, 0.42, 0.02)]     # area-law tail
))
# This gives ~25 p values: 0.02, 0.04, 0.06, 0.08, 0.10, 0.11, ..., 0.21, 0.22, 0.24, ..., 0.40
SEEDS_PER_POINT = 2       # Seeds for timing (mean of 2, faster for large n)
RREF_VERIFY_SEED = 0      # Single seed for RREF verification


# ═══════════════════════════════════════════════════════════════
# Build the FULL two-qubit Clifford group
# ═══════════════════════════════════════════════════════════════

def build_full_clifford_group():
    """Build symplectic matrices for ALL 11,520 two-qubit Cliffords."""
    print("Building full two-qubit Clifford group (11,520 elements)...", flush=True)
    t0 = time.perf_counter()
    all_tabs = list(stim.Tableau.iter_all(2))
    n_cliff = len(all_tabs)
    symp = np.zeros((n_cliff, 4, 4), dtype=np.uint8)
    for i, tab in enumerate(all_tabs):
        symp[i] = symplectic_from_stim_tableau(tab)
    dt = time.perf_counter() - t0
    print(f"  {n_cliff} Cliffords built in {dt:.2f}s", flush=True)
    return all_tabs, symp


# ═══════════════════════════════════════════════════════════════
# Circuit schedule generation
# ═══════════════════════════════════════════════════════════════

def generate_schedule(n, p, seed, n_cliff):
    """Generate a NN brickwork circuit schedule.

    Returns list of (pairs, cliff_indices, measurements) per layer.
    """
    rng = np.random.default_rng(seed)
    T = 8 * n  # Depth = 8n layers
    schedule = []
    for t in range(T):
        m = t % 2  # Alternating even/odd matchings
        n_pairs = n // 2
        ci = rng.integers(0, n_cliff, size=n_pairs)
        pairs = []
        for j in range(n_pairs):
            if m == 0:
                qi, qj = 2*j, 2*j+1
            else:
                qi, qj = 2*j+1, (2*j+2) % n
            pairs.append((qi, qj))
        meas = []
        for qi, qj in pairs:
            if rng.random() < p:
                meas.append(qi)
            if rng.random() < p:
                meas.append(qj)
        schedule.append((pairs, ci, sorted(set(meas))))
    return schedule


# ═══════════════════════════════════════════════════════════════
# GF(2) RREF for verification
# ═══════════════════════════════════════════════════════════════

def gf2_rref(mat):
    """Compute reduced row echelon form over GF(2). Returns (rref, rank)."""
    m = mat.copy()
    nrows, ncols = m.shape
    rank = 0
    for col in range(ncols):
        pivot = None
        for row in range(rank, nrows):
            if m[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            m[[rank, pivot]] = m[[pivot, rank]]
        for row in range(nrows):
            if row != rank and m[row, col]:
                m[row] ^= m[rank]
        rank += 1
    return m[:rank], rank


def extract_stim_sys_matrix(sim_st, n):
    """Extract 2n x 2n system sub-matrix from Stim TableauSimulator."""
    N = 2 * n
    fwd = sim_st.current_inverse_tableau().inverse(unsigned=True)
    mat = np.zeros((N, 2 * n), dtype=np.uint8)
    for row in range(N):
        ps = fwd.z_output(row)
        for col in range(n):
            pauli = ps[col]
            if pauli in (1, 2):
                mat[row, col] = 1
            if pauli in (2, 3):
                mat[row, n + col] = 1
    return mat


# ═══════════════════════════════════════════════════════════════
# Run benchmark
# ═══════════════════════════════════════════════════════════════

def run_one(n, p, seed, TABS, SYMP, n_cliff, verify_rref=False):
    """Run one (n, p, seed) point through both simulators.

    Returns (t_sparse, t_stim, k, rref_match).
    """
    schedule = generate_schedule(n, p, seed, n_cliff)

    # --- SparseGF2 ---
    sim_s = SparseGF2(n, use_min_weight_pivot=True)
    t0 = time.perf_counter()
    for pairs, ci, meas in schedule:
        for ip, (qi, qj) in enumerate(pairs):
            sim_s.apply_gate(qi, qj, SYMP[ci[ip]])
        for mq in meas:
            sim_s.apply_measurement_z(mq)
    t_sparse = time.perf_counter() - t0
    k = sim_s.compute_k()

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
    t_stim = time.perf_counter() - t0

    # --- RREF verification ---
    rref_match = None
    if verify_rref:
        mat_s = sim_s.extract_sys_matrix()
        mat_st = extract_stim_sys_matrix(sim_st, n)
        rref_s, rank_s = gf2_rref(mat_s)
        rref_st, rank_st = gf2_rref(mat_st)
        rref_match = (rank_s == rank_st and np.array_equal(rref_s, rref_st))

    return t_sparse, t_stim, k, rref_match


def main():
    TABS, SYMP = build_full_clifford_group()
    n_cliff = len(TABS)

    print("Warming up JIT...", flush=True)
    warmup()
    # Warmup at each size to avoid first-call JIT penalty
    for n in SIZES:
        _ = run_one(n, 0.20, 999, TABS, SYMP, n_cliff, verify_rref=False)
    print("Ready.\n", flush=True)

    # Header
    print("=" * 90)
    print(f"SparseGF2 vs Stim Benchmark — Full 11,520 Clifford Group")
    print(f"NN brickwork on C_n, depth T=8n, {SEEDS_PER_POINT} seeds/point")
    print("=" * 90)

    results = []
    total_verified = 0
    total_passed = 0

    for n in SIZES:
        print(f"\n{'='*70}")
        print(f"  n = {n} (2n = {2*n} qubits, T = {8*n} layers)")
        print(f"{'='*70}")
        print(f"  {'p':>6}  {'SparseGF2':>10}  {'Stim':>10}  {'Speedup':>8}  "
              f"{'k':>4}  {'RREF':>6}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*4}  {'-'*6}")

        for p in P_VALUES:
            # Timing: mean of SEEDS_PER_POINT runs
            ts_sparse = []
            ts_stim = []
            k_val = None
            for seed in range(SEEDS_PER_POINT):
                t_s, t_st, k, _ = run_one(n, p, seed, TABS, SYMP, n_cliff,
                                           verify_rref=False)
                ts_sparse.append(t_s)
                ts_stim.append(t_st)
                if k_val is None:
                    k_val = k

            # RREF verification: single dedicated seed
            _, _, _, rref_ok = run_one(n, p, RREF_VERIFY_SEED, TABS, SYMP,
                                       n_cliff, verify_rref=True)
            total_verified += 1
            if rref_ok:
                total_passed += 1

            t_sp = np.mean(ts_sparse)
            t_st = np.mean(ts_stim)
            speedup = t_st / t_sp if t_sp > 0 else float('inf')
            rref_str = "PASS" if rref_ok else "FAIL"

            print(f"  {p:>6.2f}  {t_sp:>10.4f}  {t_st:>10.4f}  "
                  f"{speedup:>7.1f}x  {k_val:>4}  {rref_str:>6}", flush=True)

            results.append({
                "n": n, "p": p,
                "t_sparse": round(t_sp, 5),
                "t_stim": round(t_st, 5),
                "speedup": round(speedup, 2),
                "k": int(k_val),
                "rref_match": rref_ok,
                "n_cliffords": n_cliff,
                "depth": 8 * n,
                "seeds": SEEDS_PER_POINT,
            })

    # Summary
    print(f"\n{'='*90}")
    print(f"RREF Verification: {total_passed}/{total_verified} PASS "
          f"({'ALL MATCH' if total_passed == total_verified else 'FAILURES DETECTED'})")
    print(f"Clifford group: {n_cliff} elements (complete two-qubit Clifford group)")
    print(f"{'='*90}")

    # Save results
    out_file = OUT_DIR / "benchmark_full_clifford_group.json"
    with open(out_file, "w") as f:
        json.dump({
            "metadata": {
                "n_cliffords": n_cliff,
                "sizes": SIZES,
                "p_values": P_VALUES,
                "seeds_per_point": SEEDS_PER_POINT,
                "depth_mode": "8n",
                "circuit": "NN brickwork on C_n",
                "rref_verified": total_passed == total_verified,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
