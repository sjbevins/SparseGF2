#!/usr/bin/env python3
"""
MIPT runtime benchmark: SparseGF2 scaling across the full phase diagram.

Measures runtime and k as a function of p and n for both NN and ATA models.
Full p sweep for ALL system sizes — no special-casing by phase.

Usage:
    cd SparseGF2
    py -3.13 benchmarks/benchmark_mipt_runtime.py
"""
import sys, time, json, math, pickle
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Configuration ─────────────────────────────────────────────

NN_SIZES  = [32, 48, 64, 96, 128, 192, 256, 384, 512]
ATA_SIZES = [32, 48, 64, 96, 128, 192, 256, 384, 512]

P_VALUES = sorted(set(
    [round(x, 3) for x in np.arange(0.02, 0.10, 0.02)] +
    [round(x, 3) for x in np.arange(0.10, 0.22, 0.005)] +
    [round(x, 3) for x in np.arange(0.22, 0.42, 0.02)]
))

SEEDS_PER_POINT = 10
N_WORKERS = 14
N_CLIFF_CACHE = 4096

OUT_DIR = REPO_ROOT / "benchmarks" / "results"
OUT_DIR.mkdir(exist_ok=True)


# ── Pre-generate Clifford cache ──────────────────────────────

def build_clifford_cache():
    import stim
    from sparsegf2.gates.clifford import symplectic_from_stim_tableau

    cache_path = OUT_DIR / "clifford_cache.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if "symp" in cache and len(cache["symp"]) == N_CLIFF_CACHE:
            print(f"Loaded Clifford cache ({N_CLIFF_CACHE} gates)", flush=True)
            return
    print("Building Clifford cache...", flush=True)
    symp = np.zeros((N_CLIFF_CACHE, 4, 4), dtype=np.uint8)
    for i in range(N_CLIFF_CACHE):
        tab = stim.Tableau.random(2)
        symp[i] = symplectic_from_stim_tableau(tab)
    with open(cache_path, "wb") as f:
        pickle.dump({"symp": symp}, f)
    print(f"  Cached {N_CLIFF_CACHE} Cliffords", flush=True)


# ── Worker function ───────────────────────────────────────────

def _run_batch(batch):
    import sys, time, math, pickle
    import numpy as np
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from sparsegf2.core.sparse_tableau import SparseGF2, warmup

    with open(repo / "benchmarks" / "results" / "clifford_cache.pkl", "rb") as f:
        SYMP = pickle.load(f)["symp"]
    N_CLIFF = len(SYMP)
    warmup()

    results = []
    for model, n, p, seed in batch:
        try:
            rng = np.random.default_rng(seed)
            if model == "NN":
                T = 8 * n
            else:
                T = 8 * max(1, int(math.ceil(math.log2(n))))

            sim = SparseGF2(n, use_min_weight_pivot=True)
            t0 = time.perf_counter()

            for t in range(T):
                if model == "NN":
                    m = t % 2
                    n_pairs = n // 2
                    ci = rng.integers(0, N_CLIFF, size=n_pairs)
                    for j in range(n_pairs):
                        qi = 2*j if m == 0 else 2*j+1
                        qj = 2*j+1 if m == 0 else (2*j+2) % n
                        sim.apply_gate(qi, qj, SYMP[ci[j]])
                    for j in range(n_pairs):
                        qi = 2*j if m == 0 else 2*j+1
                        qj = 2*j+1 if m == 0 else (2*j+2) % n
                        if rng.random() < p: sim.apply_measurement_z(qi)
                        if rng.random() < p: sim.apply_measurement_z(qj)
                else:
                    perm = rng.permutation(n)
                    n_pairs = n // 2
                    ci = rng.integers(0, N_CLIFF, size=n_pairs)
                    for j in range(n_pairs):
                        sim.apply_gate(int(perm[2*j]), int(perm[2*j+1]), SYMP[ci[j]])
                    for q in range(n):
                        if rng.random() < p: sim.apply_measurement_z(q)

            elapsed = time.perf_counter() - t0
            k = sim.compute_k()
            abar = sim.get_active_count()

            results.append({
                "model": model, "n": n, "p": p, "seed": seed,
                "time": elapsed, "k": int(k), "abar": round(abar, 2),
            })
        except Exception as e:
            import traceback
            results.append({
                "model": model, "n": n, "p": p, "seed": seed,
                "time": -1, "k": -1, "abar": -1,
                "error": traceback.format_exc(),
            })
    return results


# ── Main ──────────────────────────────────────────────────────

def main():
    build_clifford_cache()

    batches = []
    for model, sizes in [("NN", NN_SIZES), ("ATA", ATA_SIZES)]:
        for n in sizes:
            for p in P_VALUES:
                batches.append([(model, n, p, s) for s in range(SEEDS_PER_POINT)])

    total = sum(len(b) for b in batches)
    print(f"Total: {total} jobs in {len(batches)} batches", flush=True)
    print(f"Workers: {N_WORKERS}", flush=True)
    print(f"NN sizes: {NN_SIZES}", flush=True)
    print(f"ATA sizes: {ATA_SIZES}", flush=True)
    print(f"p values: {len(P_VALUES)} ({P_VALUES[0]:.3f}..{P_VALUES[-1]:.3f})", flush=True)
    print(flush=True)

    batches.sort(key=lambda b: (b[0][1], b[0][0] == "ATA", b[0][2]))

    results = []
    done = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_batch, b): b[0] for b in batches}
        for f in as_completed(futures):
            label = futures[f]
            try:
                batch_results = f.result()
                results.extend(batch_results)
                for r in batch_results:
                    if "error" in r:
                        print(f"  ERR {r['model']} n={r['n']} p={r['p']}: "
                              f"{r['error'][:80]}", flush=True)
            except Exception as e:
                print(f"  BATCH FAIL {label}: {e}", flush=True)
            done += 1
            if done % 10 == 0 or done == len(batches):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(batches) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(batches)}] {len(results)} ok, "
                      f"{elapsed:.0f}s, ~{eta:.0f}s left", flush=True)

    out = OUT_DIR / "mipt_runtime_benchmark.json"
    with open(out, "w") as f:
        json.dump(results, f)
    print(f"\nSaved {len(results)} results to {out}", flush=True)

    for model in ["NN", "ATA"]:
        mr = [r for r in results if r["model"] == model and "error" not in r]
        if not mr: continue
        print(f"\n{model}:", flush=True)
        for n in sorted(set(r["n"] for r in mr)):
            nr = [r for r in mr if r["n"] == n]
            ts = [r["time"] for r in nr if r["time"] > 0]
            print(f"  n={n:>5}: {len(nr)} runs, avg {np.mean(ts):.4f}s", flush=True)


if __name__ == "__main__":
    main()
