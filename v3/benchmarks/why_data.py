"""Data for the two "why this simulator" figures: observing a state while it evolves, and screening codes.

1. **Observing every layer** (``why-observe``). Monitored random Clifford
   circuits of the family in ``compare_simulators.py`` (ring, brickwork
   matching, uniformly random two-qubit Cliffords, Z measurement of every
   qubit with probability ``p = 0.16``), measured after *every* layer:

   a. the half-cut entanglement entropy ``S(first n/2 qubits)`` (no reference);
   b. the code dimension ``k = S(R)`` in the purification picture (every
      system qubit Bell-paired with a reference qubit, 2n qubits in all).

   For each, SparseGF2 v3 and Stim are timed on the same circuits, one
   thread each, "evolve only" and "evolve + observe every layer".

   * SparseGF2: the ``mipt`` ensemble engine with ``num_threads=1``, probe
     ``half_cut_entropy`` / ``code_dimension`` at ``checkpoints="every_layer"``;
     evolve only = the same spec with ``checkpoints="final"`` and no probes.
     Each circuit of the batch is exported with ``spec.circuit(t, seed)``.
   * Stim: the exported circuit, split at its ``TICK`` s into prebuilt per-layer
     ``stim.Circuit`` objects (each two-qubit Clifford in Stim's tightest form,
     batched per layer, as in ``compare_simulators.py``), run layer by layer
     with ``TableauSimulator.do_circuit``. After each layer the stabilizer
     columns are read from ``current_inverse_tableau().to_numpy(bit_packed=True)``:
     row q of the inverse tableau's ``z2x`` (``x2x``) block is the X (Z) column
     of qubit q over the stabilizers, so no inverse is computed (the other
     routes, ``canonical_stabilizers()`` and ``...inverse().to_numpy()``, are
     timed once per size and recorded; they are 10 to 1000 times slower). The
     GF(2) rank of the region's columns is computed by a numba-compiled
     elimination on 64-bit words, the same algorithm SparseGF2 uses for a
     region entropy. Evolve only = one prebuilt ``stim.Circuit`` of all layers
     through ``do_circuit`` (its fastest path). The per-phase split (evolve,
     extract, rank) of the observed run is recorded.
   * Correctness: for every size and circuit, Stim's per-layer values must
     equal SparseGF2's exactly; the script aborts on any mismatch.

2. **Erasure screening per code** (``why-erasure``). Codes from the engine
   (purification picture, Watts-Strogatz graph k = 4, beta = 0.1, p = 0.05,
   n/2 layers, ``code_snapshot``) loaded as ``StabilizerCode``. Timed:
   ``uncorrectable_counts(2)``, ``uncorrectable_counts(3)`` and
   ``first_failure_sizes(100)``. Baselines, numba-compiled, on the same
   compressed columns (``_kernel.entropy.code_columns``) with
   ``l(A) = rank(N(S)|_A) - rank(S|_A)``:

   * counts: ``l(A)`` evaluated from scratch (one elimination for each of the
     two ranks) for every subset A of size <= w; no pruning, no incremental
     bases, no hash lookups;
   * first failure: for the same orderings as the kernel (same PRNG stream),
     ``l`` of a prefix evaluated from scratch, at m = 1, 2, 4, ... until it is
     positive, then by bisection (which uses only that ``l`` grows with the
     set).

   Baseline counts and first-failure sizes must equal the kernel's (abort on
   mismatch). A baseline case whose estimated time (extrapolated from the
   smaller sizes) exceeds ``--baseline-limit`` seconds is skipped and recorded
   as skipped, never as a number.

Usage (from the repository root, on an idle machine)::

    python v3/benchmarks/why_data.py --out v3/benchmarks/results/why.json
    python v3/benchmarks/why_data.py --quick --out /tmp/why_quick.json   # smoke test

Writes a JSON file with every timing, the workload parameters and the
environment (CPU, versions, thread count).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import compare_simulators as cs  # noqa: E402  (also puts v3/src on the path)

from sparsegf2_v3 import graphs, mipt  # noqa: E402
from sparsegf2_v3._kernel.entropy import code_columns  # noqa: E402
from sparsegf2_v3._kernel.rng import next_below, seed_state  # noqa: E402

ZERO = np.uint64(0)
ONE = np.uint64(1)

# ---------------------------------------------------------------------------
# compiled helpers
# ---------------------------------------------------------------------------


@njit(cache=True)
def _echelon_rank(scratch, m, ws):
    """Rank of the first ``m`` bitset rows of ``scratch`` (destroyed); as in SparseGF2's kernel."""
    rank = 0
    for i in range(m):
        lead_w = -1
        for w in range(ws):
            if scratch[i, w] != ZERO:
                lead_w = w
                break
        if lead_w < 0:
            continue
        rank += 1
        word = scratch[i, lead_w]
        lead = word & (~word + ONE)
        for j in range(i + 1, m):
            if (scratch[j, lead_w] & lead) != ZERO:
                for w in range(lead_w, ws):
                    scratch[j, w] ^= scratch[i, w]
    return rank


@njit(cache=True)
def region_rank(xcols, zcols, qubits, scratch):
    """GF(2) rank of the X and Z stabilizer columns of ``qubits`` (rows of ``xcols``/``zcols``)."""
    ws = xcols.shape[1]
    for i in range(qubits.shape[0]):
        q = qubits[i]
        for w in range(ws):
            scratch[2 * i, w] = xcols[q, w]
            scratch[2 * i + 1, w] = zcols[q, w]
    return _echelon_rank(scratch, 2 * qubits.shape[0], ws)


@njit(cache=True)
def _lost(cols, s, fmask, wf, subset, size, scratch_n, scratch_s):
    """``l(A) = rank(N(S)|_A) - rank(S|_A)`` for the first ``size`` qubits of ``subset``, from scratch."""
    wa = cols.shape[1]
    for i in range(size):
        q = subset[i]
        for c in range(2):
            for v in range(wa):
                scratch_n[2 * i + c, v] = cols[2 * q + c, v]
            for v in range(wf):
                scratch_s[2 * i + c, v] = cols[2 * q + c, v]
            scratch_s[2 * i + c, wf - 1] &= fmask
    rank_n = _echelon_rank(scratch_n, 2 * size, wa)
    rank_s = _echelon_rank(scratch_s, 2 * size, wf) if s > 0 else 0
    return rank_n - rank_s


@njit(cache=True)
def _prepare(tab, ws, system, ref_rows):
    cols, s, r = code_columns(tab, ws, system, ref_rows)
    wf = max((s + 63) // 64, 1)
    fmask = ZERO
    if s > 0:
        fmask = ~ZERO if s % 64 == 0 else (ONE << np.uint64(s % 64)) - ONE
    return cols, s, r, wf, fmask


@njit(cache=True)
def baseline_counts(tab, ws, system, ref_rows, max_size, out):
    """``out[m - 1]`` = number of m-subsets with ``l(A) > 0``, every subset evaluated from scratch."""
    n = system.shape[0]
    cols, s, r, wf, fmask = _prepare(tab, ws, system, ref_rows)
    for m in range(max_size):
        out[m] = 0.0
    if r == s:
        return
    scratch_n = np.zeros((2 * max_size, cols.shape[1]), dtype=np.uint64)
    scratch_s = np.zeros((2 * max_size, wf), dtype=np.uint64)
    subset = np.zeros(max_size, dtype=np.int64)
    for m in range(1, max_size + 1):
        for i in range(m):
            subset[i] = i
        count = 0
        while True:
            if _lost(cols, s, fmask, wf, subset, m, scratch_n, scratch_s) > 0:
                count += 1
            # next m-combination in lexicographic order
            i = m - 1
            while i >= 0 and subset[i] == n - m + i:
                i -= 1
            if i < 0:
                break
            subset[i] += 1
            for j in range(i + 1, m):
                subset[j] = subset[j - 1] + 1
        out[m - 1] = count


@njit(cache=True)
def baseline_first_failure(tab, ws, system, ref_rows, k, prng, out):
    """First-failure sizes for the kernel's orderings (same PRNG draws), prefixes evaluated from scratch.

    Galloping search (m = 1, 2, 4, ...) for a failing prefix, then bisection
    between the last correctable and the first failing size.
    """
    n = system.shape[0]
    if k == 0:
        for i in range(out.shape[0]):
            out[i] = n + 1
        return
    cols, s, r, wf, fmask = _prepare(tab, ws, system, ref_rows)
    scratch_n = np.zeros((2 * n, cols.shape[1]), dtype=np.uint64)
    scratch_s = np.zeros((2 * n, wf), dtype=np.uint64)
    idx = np.arange(n)
    for i in range(out.shape[0]):
        for t in range(n - 1):  # the kernel's Fisher-Yates shuffle, draw for draw
            j = t + next_below(prng, n - t)
            tmp = idx[t]
            idx[t] = idx[j]
            idx[j] = tmp
        lo = 0  # largest size known correctable
        hi = 1
        while True:
            if _lost(cols, s, fmask, wf, idx, hi, scratch_n, scratch_s) > 0:
                break
            lo = hi
            if hi == n:
                hi = n + 1
                break
            hi = min(2 * hi, n)
        if hi <= n:
            while hi - lo > 1:  # l(prefix lo) = 0 < l(prefix hi)
                mid = (lo + hi) // 2
                if _lost(cols, s, fmask, wf, idx, mid, scratch_n, scratch_s) > 0:
                    hi = mid
                else:
                    lo = mid
        out[i] = hi


# ---------------------------------------------------------------------------
# part 1: observing every layer
# ---------------------------------------------------------------------------


def _disjoint_runs(pairs, indices):
    """Split a gate list into consecutive runs of qubit-disjoint pairs (which commute, so batch)."""
    runs, used, cur = [], set(), ([], [])
    for (a, b), idx in zip(pairs, indices, strict=True):
        if a in used or b in used:
            runs.append(cur)
            used, cur = set(), ([], [])
        used.update((a, b))
        cur[0].append((a, b))
        cur[1].append(idx)
    if cur[0]:
        runs.append(cur)
    return runs


def _append_cliffords(c, pairs, indices) -> None:
    """Two-qubit Cliffords in Stim's tightest form, batched per slot (as compare_simulators.build_stim)."""
    for run_pairs, run_indices in _disjoint_runs(pairs, indices):
        parts = [cs.stim_decomposition(idx) for idx in run_indices]
        for slot in range(5):
            groups: dict[str, list[int]] = {}
            for (a, b), part in zip(run_pairs, parts, strict=True):
                name = part[slot]
                if name != "I":
                    groups.setdefault(name, []).extend(
                        [a] if slot in (0, 3) else [b] if slot in (1, 4) else [a, b]
                    )
            for name, targets in groups.items():
                c.append(name, targets)


def stim_layers(circuit) -> list:
    """The exported SparseGF2 circuit as one ``stim.Circuit`` per checkpoint layer (split at TICKs).

    Entry 0 holds the setup (Bell pairs) and scramble, entry t the gates and
    measurements of layer t.
    """
    import stim

    layers = [stim.Circuit()]
    for ins in circuit.flattened():
        targets = [t.value for t in ins.targets]
        if ins.name == "TICK":
            layers.append(stim.Circuit())
        elif ins.name == "CLIFFORD_2Q":
            pairs = list(zip(targets[0::2], targets[1::2], strict=True))
            _append_cliffords(layers[-1], pairs, [int(a) for a in ins.args])
        elif ins.name in ("H", "CX", "M"):
            layers[-1].append(ins.name, targets)
        else:
            raise ValueError(f"unexpected instruction {ins.name} in an exported circuit")
    if len(layers[-1]) == 0:
        layers.pop()  # the circuit ends with a TICK
    return layers


def _columns(sim, words: int):
    """Stabilizer X and Z columns of every qubit, as (N, words) uint64 arrays, without an inverse."""
    ix2x, _, iz2x, _, _, _ = sim.current_inverse_tableau().to_numpy(bit_packed=True)
    if ix2x.shape[1] == 8 * words:
        return iz2x.view(np.uint64), ix2x.view(np.uint64)
    xs = np.zeros((ix2x.shape[0], 8 * words), dtype=np.uint8)
    zs = np.zeros_like(xs)
    xs[:, : iz2x.shape[1]] = iz2x
    zs[:, : ix2x.shape[1]] = ix2x
    return xs.view(np.uint64), zs.view(np.uint64)


def stim_observe(layers, num_qubits: int, region: np.ndarray, offset: int):
    """Run the per-layer circuits; after each, ``rank(region columns) - offset``. Returns values and phase times."""
    import stim

    words = (num_qubits + 63) // 64
    scratch = np.zeros((2 * len(region), words), dtype=np.uint64)
    sim = stim.TableauSimulator(seed=1)
    sim.set_num_qubits(num_qubits)
    values = np.zeros(len(layers), dtype=np.int64)
    t_evolve = t_extract = t_rank = 0.0
    clock = time.perf_counter
    for i, layer in enumerate(layers):
        t0 = clock()
        sim.do_circuit(layer)
        t1 = clock()
        xcols, zcols = _columns(sim, words)
        t2 = clock()
        values[i] = region_rank(xcols, zcols, region, scratch) - offset
        t3 = clock()
        t_evolve += t1 - t0
        t_extract += t2 - t1
        t_rank += t3 - t2
    return values, {"evolve": t_evolve, "extract": t_extract, "rank": t_rank}


def stim_evolve(full, num_qubits: int) -> float:
    import stim

    t = time.perf_counter()
    sim = stim.TableauSimulator(seed=1)
    sim.set_num_qubits(num_qubits)
    sim.do_circuit(full)
    return time.perf_counter() - t


def extraction_options(layers, num_qubits: int, skip_canonical: bool) -> dict:
    """Seconds for one stabilizer extraction by each route, on the state after the whole circuit."""
    import stim

    sim = stim.TableauSimulator(seed=1)
    sim.set_num_qubits(num_qubits)
    for layer in layers:
        sim.do_circuit(layer)
    words = (num_qubits + 63) // 64
    routes = {
        "inverse_tableau_columns": lambda: _columns(sim, words),
        "forward_tableau_rows": lambda: (
            sim.current_inverse_tableau().inverse().to_numpy(bit_packed=True)
        ),
        "canonical_stabilizers": lambda: [p.to_numpy() for p in sim.canonical_stabilizers()],
    }
    out = {}
    for name, fn in routes.items():
        if name == "canonical_stabilizers" and skip_canonical:
            out[name] = None
            continue
        times = []
        for _ in range(3):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
        out[name] = min(times)
    return out


def observe_case(n: int, observable: str, layers: int, p: float, circuits: int, repeats: int):
    reference = "none" if observable == "half_cut_entropy" else "purification"
    probe = mipt.half_cut_entropy if observable == "half_cut_entropy" else mipt.code_dimension
    common = dict(
        num_system=n,
        edges=graphs.cycle(n),
        layers=layers,
        measurement_rate=p,
        schedule="matching",
        reference=reference,
    )
    spec_obs = mipt.MonitoredCircuit(
        **common, checkpoints="every_layer", probes=[probe(checkpoints="every_layer")]
    )
    spec_evo = mipt.MonitoredCircuit(**common, checkpoints="final")
    seed = 1000 + n
    spec_obs.run(1, seed=0, num_threads=1)  # compile and preprocess the graph
    spec_evo.run(1, seed=0, num_threads=1)
    t_obs, t_evo = [], []
    for _ in range(repeats):
        t = time.perf_counter()
        res = spec_obs.run(circuits, seed=seed, num_threads=1)
        t_obs.append((time.perf_counter() - t) / circuits)
        t = time.perf_counter()
        spec_evo.run(circuits, seed=seed, num_threads=1)
        t_evo.append((time.perf_counter() - t) / circuits)
    ours = res.values[observable]  # (circuits, layers + 1)

    num_qubits = spec_obs.num_qubits
    if observable == "half_cut_entropy":
        region = np.arange(n // 2, dtype=np.int64)
        offset = n // 2
    else:
        region = np.arange(n, 2 * n, dtype=np.int64)  # S(R) = rank(R columns) - |R|
        offset = n
    s_evo, s_obs, phases = [], [], []
    for t in range(circuits):
        circuit = spec_obs.circuit(t, seed=seed)
        if t == 0 and str(spec_evo.circuit(t, seed=seed)) != str(circuit):
            raise AssertionError("the evolve-only spec generated a different circuit")
        per_layer = stim_layers(circuit)
        if len(per_layer) != layers + 1:
            raise AssertionError(f"expected {layers + 1} layer blocks, got {len(per_layer)}")
        full = sum(per_layer[1:], per_layer[0].copy())
        stim_evolve(full, num_qubits)  # warm-up
        values, ph = stim_observe(per_layer, num_qubits, region, offset)  # warm-up and check
        if not np.array_equal(values, ours[t].astype(np.int64)):
            bad = int(np.flatnonzero(values != ours[t])[0])
            raise AssertionError(
                f"{observable}, n={n}, circuit {t}: Stim {values[bad]} vs SparseGF2 "
                f"{ours[t][bad]} at layer {bad}"
            )
        evo_times, obs_times, ph_runs = [], [], []
        for _ in range(repeats):
            evo_times.append(stim_evolve(full, num_qubits))
            t0 = time.perf_counter()
            _, ph = stim_observe(per_layer, num_qubits, region, offset)
            obs_times.append(time.perf_counter() - t0)
            ph_runs.append(ph)
        s_evo.append(statistics.median(evo_times))
        s_obs.append(statistics.median(obs_times))
        phases.append(ph_runs[int(np.argsort(obs_times)[len(obs_times) // 2])])
    options = extraction_options(per_layer, num_qubits, skip_canonical=num_qubits > 1024)
    row = {
        "n": n,
        "observable": observable,
        "qubits": num_qubits,
        "layers": layers,
        "evaluations": layers + 1,
        "circuits": circuits,
        "sparsegf2": {
            "evolve": statistics.median(t_evo),
            "observe": statistics.median(t_obs),
        },
        "stim": {
            "evolve": statistics.mean(s_evo),
            "observe": statistics.mean(s_obs),
            "phases": {k: statistics.mean(ph[k] for ph in phases) for k in phases[0]},
        },
        "stim_extraction_options": options,
        "mean_final_value": float(ours[:, -1].mean()),
        "checked": True,
    }
    return row


# ---------------------------------------------------------------------------
# part 2: erasure screening
# ---------------------------------------------------------------------------


def _timed(fn, repeats: int):
    times, value = [], None
    for _ in range(repeats):
        t = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times), value


def erasure_codes(n: int, p: float, layers: int, count: int):
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.watts_strogatz(n, 4, 0.1, seed=(0, n)),
        layers=layers,
        measurement_rate=p,
        schedule="random_edges",
        reference="purification",
        checkpoints="final",
        probes=[mipt.code_dimension(), mipt.code_snapshot()],
    )
    res = spec.run(count, seed=2000 + n)
    return [res.code(t, 0) for t in range(count)]


def erasure_case(code, case_seed, repeats, baseline_ok):
    """Kernel and baseline timings for one code; ``baseline_ok(task)`` says whether to run a baseline."""
    n = code.num_qubits
    tab, ws, system, marked = code._tab, code._ws, code._system, code._marked
    row = {"n": n, "k": code.num_logical, "kernel": {}, "baseline": {}, "skipped": []}
    for w in (2, 3):
        task = f"counts{w}"
        sec, counts = _timed(lambda w=w: code.uncorrectable_counts(w), repeats)
        row["kernel"][task] = sec
        row[f"uncorrectable_counts_{w}"] = [int(c) for c in counts]
        if baseline_ok(task):
            out = np.zeros(w)
            sec_b, _ = _timed(
                lambda w=w, out=out: baseline_counts(tab, ws, system, marked, w, out), 1
            )
            if not np.array_equal(out.astype(np.int64), counts):
                raise AssertionError(f"n={n}: baseline counts {out} != kernel {counts}")
            row["baseline"][task] = sec_b
        else:
            row["skipped"].append(task)
    task = "first_failure100"
    sec, first = _timed(lambda: code.first_failure_sizes(100, seed=case_seed), repeats)
    row["kernel"][task] = sec
    row["mean_first_failure"] = float(first.mean())
    if baseline_ok(task):
        out = np.zeros(100)
        k = code.num_logical
        sec_b, _ = _timed(
            lambda: baseline_first_failure(tab, ws, system, marked, k, seed_state(case_seed), out),
            1,
        )
        if not np.array_equal(out.astype(np.int64), first):
            raise AssertionError(f"n={n}: baseline first-failure sizes differ from the kernel's")
        row["baseline"][task] = sec_b
    else:
        row["skipped"].append(task)
    return row


def _estimate(history: dict, task: str, n: int) -> float | None:
    """Extrapolate a baseline's time to size n from the largest measured size (power law in n)."""
    sizes = sorted(m for m in history.get(task, {}) if m < n)
    if not sizes:
        return None
    if len(sizes) == 1:
        m = sizes[-1]
        power = {"counts2": 2, "counts3": 3, "first_failure100": 3}[task] + 1
        return history[task][m] * (n / m) ** power
    a, b = sizes[-2], sizes[-1]
    power = max(math.log(history[task][b] / history[task][a]) / math.log(b / a), 1.0)
    return history[task][b] * (n / b) ** power


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(HERE / "results" / "why.json"))
    ap.add_argument("--quick", action="store_true", help="tiny sizes, for a smoke test")
    ap.add_argument("--rate", type=float, default=0.16, help="measurement rate of part 1")
    ap.add_argument("--max-layers", type=int, default=256, help="part 1 runs min(2n, this) layers")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--baseline-limit", type=float, default=120.0, help="seconds (estimated)")
    ap.add_argument("--skip", nargs="*", default=[], choices=["observe", "erasure"])
    args = ap.parse_args()
    obs_sizes = [16, 32] if args.quick else [32, 64, 128, 256, 512, 1024]
    era_sizes = [32] if args.quick else [64, 128, 256, 512]
    codes_per_size = 2 if args.quick else 3

    import numba

    numba.set_num_threads(1)
    env = cs.environment()
    env["threads"] = 1
    results = {
        "environment": env,
        "observe": {
            "workload": "ring, brickwork matching, uniform random two-qubit Cliffords, "
            "Z measurement with probability p after each layer",
            "measurement_rate": args.rate,
            "layers": f"min(2n, {args.max_layers})",
            "repeats": args.repeats,
            "times": "seconds per circuit; SparseGF2 median over repeats of a batch, "
            "Stim mean over circuits of the median over repeats",
            "stim_route": "per-layer do_circuit; columns from current_inverse_tableau()"
            ".to_numpy(bit_packed=True); numba GF(2) rank",
            "rows": [],
        },
        "erasure": {
            "codes": "purification picture, Watts-Strogatz k=4 beta=0.1, random_edges "
            "schedule, n/2 layers, final code (code_snapshot)",
            "measurement_rate": 0.05,
            "codes_per_size": codes_per_size,
            "baseline_limit_seconds": args.baseline_limit,
            "times": "seconds per code; kernel median over repeats, baseline one run",
            "rows": [],
        },
    }
    start = time.perf_counter()
    if "observe" not in args.skip:
        for n in obs_sizes:
            layers = min(2 * n, args.max_layers)
            circuits = 4 if n <= 128 else 2
            for observable in ("half_cut_entropy", "code_dimension"):
                row = observe_case(n, observable, layers, args.rate, circuits, args.repeats)
                results["observe"]["rows"].append(row)
                sg, st = row["sparsegf2"], row["stim"]
                print(
                    f"observe {observable:<16} n={n:<5} layers={layers:<4} "
                    f"SparseGF2 {sg['evolve'] * 1e3:9.2f} -> {sg['observe'] * 1e3:9.2f} ms   "
                    f"Stim {st['evolve'] * 1e3:9.2f} -> {st['observe'] * 1e3:9.2f} ms "
                    f"(extract {st['phases']['extract'] * 1e3:.2f}, rank "
                    f"{st['phases']['rank'] * 1e3:.2f})   checked",
                    flush=True,
                )
    if "erasure" not in args.skip:
        warm = erasure_codes(16, 0.05, 8, 1)[0]  # compile the kernels and baselines untimed
        erasure_case(warm, case_seed=0, repeats=1, baseline_ok=lambda task: True)
        history: dict[str, dict[int, float]] = {}
        for n in era_sizes:
            codes = erasure_codes(n, 0.05, n // 2, codes_per_size)
            decided: dict[str, bool] = {}

            def baseline_ok(task, n=n, decided=decided):
                if task not in decided:
                    est = _estimate(history, task, n)
                    decided[task] = est is None or est <= args.baseline_limit
                return decided[task]

            for i, code in enumerate(codes):
                if code.num_logical == 0:
                    print(f"erasure n={n} code {i}: k = 0, skipped", flush=True)
                    continue
                row = erasure_case(
                    code, case_seed=(n, i), repeats=args.repeats, baseline_ok=baseline_ok
                )
                row["code"] = i
                results["erasure"]["rows"].append(row)
                for task, sec in row["baseline"].items():
                    slot = history.setdefault(task, {})
                    slot[n] = max(slot.get(n, 0.0), sec)  # the slowest code, for the estimate
                print(
                    f"erasure n={n:<4} code {i} k={row['k']:<4} "
                    + "  ".join(
                        f"{task} {row['kernel'][task] * 1e3:.2f} ms"
                        + (
                            f" (baseline {row['baseline'][task] * 1e3:.1f} ms)"
                            if task in row["baseline"]
                            else " (baseline skipped)"
                        )
                        for task in row["kernel"]
                    ),
                    flush=True,
                )
    results["total_seconds"] = time.perf_counter() - start
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out} ({results['total_seconds']:.0f} s)")


if __name__ == "__main__":
    main()
