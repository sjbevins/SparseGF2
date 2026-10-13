"""Runtime comparison of stabilizer simulators on monitored random Clifford circuits.

Workload (the standard measurement-induced-transition circuit): ``n`` qubits on a
ring, ``depth = 2 n`` layers; each layer applies uniformly random two-qubit
Cliffords on a brickwork matching (even bonds, then odd bonds), then measures
every qubit in the Z basis with probability ``p``.

Fairness rules:

* Every simulator runs the *same* circuit: identical two-qubit Cliffords and
  identical measurement locations. Simulators without a native random-Clifford
  gate get each Clifford decomposed: Stim gets the tightest form (at most four
  of its named single-qubit Cliffords and one of CX / ISWAP / SWAP, every
  layer's gates grouped into a few batched instructions), the others get Stim's
  ``Tableau.to_circuit("elimination")`` into H, S and CX.
* Each gets the circuit in its own prebuilt, native form; building that form is
  excluded from the timing. Stim runs a prebuilt ``stim.Circuit`` with
  ``do_circuit`` (its fastest path, measurements batched per layer); Qiskit Aer
  runs a prebuilt ``QuantumCircuit`` (its per-run circuit conversion is part of
  its API and therefore included); SparseGF2 runs a prebuilt ``Circuit`` whose
  compilation is cached by a warm-up run.
* Single thread for every simulator; median of ``--repeats`` runs after one
  warm-up.
* Correctness: for every size, SparseGF2 v3's final stabilizer group is checked
  against Stim's (and v2's, when included); the run aborts on any mismatch.

The ensemble rows measure SparseGF2 v3's ``mipt`` engine generating *and*
simulating fresh random circuits of the same family on all cores.

Usage::

    python v3/benchmarks/compare_simulators.py --out v3/benchmarks/results/simulators.json
    python v3/benchmarks/compare_simulators.py --quick       # small sizes, for a smoke test

Writes a JSON file with every timing plus the environment (CPU, versions).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "v3" / "src"))

import sparsegf2_v3 as sg  # noqa: E402
from sparsegf2_v3 import graphs, mipt  # noqa: E402
from sparsegf2_v3.clifford import Clifford, symplectic_group  # noqa: E402

# ---------------------------------------------------------------------------
# the workload
# ---------------------------------------------------------------------------


def make_workload(n: int, p: float, depth: int, seed: int):
    """Gate pairs, Clifford indices and measured qubits for every layer."""
    rng = np.random.default_rng(seed)
    layers = []
    for t in range(depth):
        off = t % 2
        pairs = [(i % n, (i + 1) % n) for i in range(off, n - 1 + off, 2)]
        indices = rng.integers(0, 11520, len(pairs)).tolist()
        measured = np.flatnonzero(rng.random(n) < p).tolist()
        layers.append((pairs, indices, measured))
    return layers


_DECOMP: dict[int, list[tuple[str, tuple[int, ...]]]] = {}


def decomposition(index: int) -> list[tuple[str, tuple[int, ...]]]:
    """H/S/CX gate list implementing the canonical two-qubit Clifford ``index`` on (0, 1)."""
    if index not in _DECOMP:
        circuit = Clifford.from_index(2, index).to_stim().to_circuit("elimination")
        ops = []
        for ins in circuit:
            targets = tuple(t.value for t in ins.targets_copy())
            step = 2 if ins.name == "CX" else 1
            for j in range(0, len(targets), step):
                ops.append((ins.name, targets[j : j + step]))
        _DECOMP[index] = ops
    return _DECOMP[index]


_STIM_DECOMP: dict[int, tuple[str, str, str, str, str]] = {}
_STIM_TABLES: dict = {}


def _stim_tables():
    """Named single-qubit Cliffords, the four cores, and a symplectic-class lookup."""
    if not _STIM_TABLES:
        import itertools

        import stim

        named = {}
        for name, g in stim.gate_data().items():
            if g.is_unitary and g.is_single_qubit_gate and name == g.name:
                named.setdefault(str(g.tableau), name)
        tab1 = {name: stim.Tableau.from_named_gate(name) for name in named.values()}
        cores = {c: stim.Tableau.from_named_gate(c) for c in ("I", "CX", "ISWAP", "SWAP")}
        cores["I"] = stim.Tableau(2)
        reps = {}
        for name, t in tab1.items():
            key = tuple(int(v[0]) for p in (t.x_output(0), t.z_output(0)) for v in p.to_numpy())
            reps.setdefault(key, name)
        classes = {}
        for core_name, core in cores.items():
            for a, b in itertools.product(reps.values(), repeat=2):
                for c, d in itertools.product(reps.values(), repeat=2):
                    t = (tab1[a] + tab1[b]).then(core).then(tab1[c] + tab1[d])
                    classes.setdefault(_symplectic_key(t), (a, b, core_name, c, d))
        assert len(classes) == 720
        _STIM_TABLES.update(named=named, tab1=tab1, cores=cores, classes=classes)
    return _STIM_TABLES


def _symplectic_key(t) -> bytes:
    rows = [t.x_output(0), t.x_output(1), t.z_output(0), t.z_output(1)]
    return np.array([np.concatenate(p.to_numpy()) for p in rows], dtype=np.uint8).tobytes()


def stim_decomposition(index: int) -> tuple[str, str, str, str, str]:
    """Clifford ``index`` on (0, 1) as (right_0, right_1, core, left_0, left_1) Stim gates.

    Applied in that order: single-qubit gates on qubits 0 and 1, the core
    (``I`` / ``CX`` / ``ISWAP`` / ``SWAP``) on (0, 1), single-qubit gates again.
    Checked against the Clifford's tableau.
    """
    if index not in _STIM_DECOMP:
        import stim

        tb = _stim_tables()
        target = Clifford.from_index(2, index).to_stim()
        a, b, core, c, d = tb["classes"][_symplectic_key(target)]
        t = (
            (tb["tab1"][a] + tb["tab1"][b])
            .then(tb["cores"][core])
            .then(tb["tab1"][c] + tb["tab1"][d])
        )
        frame = t.inverse().then(target)  # a Pauli frame: same symplectic part
        c2 = tb["named"][str(tb["tab1"][c].then(_pauli_1q(frame, 0)))]
        d2 = tb["named"][str(tb["tab1"][d].then(_pauli_1q(frame, 1)))]
        check = stim.Circuit("I 0 1")
        for name, q in ((a, [0]), (b, [1]), (core, [0, 1]), (c2, [0]), (d2, [1])):
            check.append(name, q)
        assert stim.Tableau.from_circuit(check) == target, index
        _STIM_DECOMP[index] = (a, b, core, c2, d2)
    return _STIM_DECOMP[index]


def _pauli_1q(frame, q: int):
    """The single-qubit Pauli tableau that ``frame`` (a 2-qubit Pauli frame) applies on qubit ``q``."""
    import stim

    sx = frame.x_output(q).sign.real < 0
    sz = frame.z_output(q).sign.real < 0
    name = {(False, False): "I", (False, True): "X", (True, False): "Z", (True, True): "Y"}[
        (sx, sz)
    ]
    return stim.Tableau.from_named_gate(name)


# ---------------------------------------------------------------------------
# adapters: build(workload) -> runner(), and a final-state hook for checks
# ---------------------------------------------------------------------------


def build_v3(n, layers, phases):
    c = sg.Circuit()
    for pairs, indices, measured in layers:
        c.append("CLIFFORD_2Q", [q for pr in pairs for q in pr], indices)
        if measured:
            c.append("M", measured)

    def run():
        sim = sg.Simulator(n, phases=phases, seed=1)
        sim.run(c)
        return sim

    run()  # compile (cached on the circuit)
    return run


def build_stim(n, layers):
    import stim

    c = stim.Circuit()
    for pairs, indices, measured in layers:
        # gates on disjoint pairs commute, so each of the five decomposition slots
        # becomes one batched instruction per gate name for the whole layer
        parts = [stim_decomposition(idx) for idx in indices]
        for slot in range(5):
            groups: dict[str, list[int]] = {}
            for (a, b), part in zip(pairs, parts, strict=True):
                name = part[slot]
                if name != "I":
                    groups.setdefault(name, []).extend(
                        [a] if slot in (0, 3) else [b] if slot in (1, 4) else [a, b]
                    )
            for name, targets in groups.items():
                c.append(name, targets)
        if measured:
            c.append("M", measured)

    def run():
        sim = stim.TableauSimulator(seed=1)
        sim.set_num_qubits(n)
        sim.do_circuit(c)
        return sim

    return run


def build_qiskit(n, layers):
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    num_meas = sum(len(m) for _, _, m in layers)
    qc = QuantumCircuit(n, max(num_meas, 1))
    k = 0
    for pairs, indices, measured in layers:
        for (a, b), idx in zip(pairs, indices, strict=True):
            for name, tg in decomposition(idx):
                q = [a if t == 0 else b for t in tg]
                if name == "H":
                    qc.h(q[0])
                elif name == "S":
                    qc.s(q[0])
                else:
                    qc.cx(q[0], q[1])
        for q in measured:
            qc.measure(q, k)
            k += 1
    if num_meas == 0:
        qc.measure(0, 0)
    backend = AerSimulator(method="stabilizer", max_parallel_threads=1, seed_simulator=1)

    def run():
        return backend.run(qc, shots=1).result()

    return run


def build_cirq(n, layers):
    import cirq

    qs = cirq.LineQubit.range(n)
    ops = []
    k = 0
    for pairs, indices, measured in layers:
        for (a, b), idx in zip(pairs, indices, strict=True):
            for name, tg in decomposition(idx):
                q = [qs[a] if t == 0 else qs[b] for t in tg]
                ops.append(
                    cirq.H(q[0])
                    if name == "H"
                    else cirq.S(q[0])
                    if name == "S"
                    else cirq.CNOT(q[0], q[1])
                )
        for q in measured:
            ops.append(cirq.measure(qs[q], key=f"m{k}"))
            k += 1
    circuit = cirq.Circuit(ops)
    simulator = cirq.CliffordSimulator(seed=1)

    def run():
        return simulator.simulate(circuit)

    return run


def build_v2(n, layers):
    sys.path.insert(0, str(REPO / "src"))
    from sparsegf2 import SparseGF2
    from sparsegf2.core.sparse_tableau import _build_2q_lut

    group = symplectic_group(2)
    luts = np.stack([_build_2q_lut(m) for m in group]).astype(np.uint8)
    prepared = []
    for pairs, indices, measured in layers:
        prepared.append(
            (
                np.asarray(pairs, dtype=np.int64),
                np.asarray(indices, dtype=np.int64) // 16,  # v2 is phase-free: symplectic part
                np.asarray(measured, dtype=np.int64),
            )
        )

    def run():
        sim = SparseGF2(n, rng=np.random.default_rng(1), hybrid=True)
        for pairs, sym, measured in prepared:
            sim._dispatch_2q_batch(pairs, sym, luts)
            if measured.size:
                sim._measure_z_batch(measured)
        return sim

    run()  # JIT warm-up
    return run


# ---------------------------------------------------------------------------
# correctness checks
# ---------------------------------------------------------------------------


def unsigned_group(paulis) -> list[str]:
    return [str(p)[1:] for p in paulis]


def check(n, layers, include_v2):
    import stim

    sim = build_v3(n, layers, phases=False)()
    ours = unsigned_group(sim.stabilizers())
    ref = build_stim(n, layers)()
    if ours != [str(p)[1:] for p in ref.canonical_stabilizers()]:
        raise AssertionError(f"v3 and Stim disagree at n={n}")
    if include_v2:
        v2 = build_v2(n, layers)()
        sym = v2.canonical_form()
        tab = stim.Tableau.from_stabilizers(
            [
                stim.PauliString(
                    "".join("_XZY"[x + 2 * z] for x, z in zip(row[:n], row[n:], strict=True))
                )
                for row in sym
            ],
            allow_underconstrained=False,
        )
        theirs = [str(p)[1:] for p in tab.to_stabilizers(canonicalize=True)]
        if ours != theirs:
            raise AssertionError(f"v3 and v2 disagree at n={n}")


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def time_runner(run, repeats: int, budget: float) -> float:
    """Median wall time of ``repeats`` runs (fewer if a run exceeds ``budget`` seconds)."""
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        run()
        times.append(time.perf_counter() - t)
        if times[-1] > budget:
            break
    return statistics.median(times)


def environment() -> dict:
    import numba

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "sparsegf2_v3": sg.__version__,
    }
    for mod in ("stim", "qiskit_aer", "cirq"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            info[mod] = None
    if info["stim"]:
        info["stim_build"] = next(
            (m[len("stim._stim_") :] for m in sys.modules if m.startswith("stim._stim_")), "unknown"
        )
        info["stim_path"] = str(Path(sys.modules["stim"].__file__).parent)
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


SIMULATORS = {
    "sparsegf2_v3_phase_free": lambda n, layers: build_v3(n, layers, phases=False),
    "sparsegf2_v3_phases": lambda n, layers: build_v3(n, layers, phases=True),
    "stim": build_stim,
    "qiskit_aer": build_qiskit,
    "cirq": build_cirq,
    "sparsegf2_v2": build_v2,
}
MAX_N = {"cirq": 64, "qiskit_aer": 256}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(HERE / "results" / "simulators.json"))
    ap.add_argument("--sizes", type=int, nargs="*", default=[16, 32, 64, 128, 256, 512, 1024])
    ap.add_argument("--rates", type=float, nargs="*", default=[0.05, 0.16, 0.3])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--budget", type=float, default=120.0, help="skip larger sizes once a run exceeds this (s)"
    )
    ap.add_argument("--skip", nargs="*", default=[], help="simulators to leave out")
    ap.add_argument("--quick", action="store_true", help="sizes 16-64, one repeat")
    ap.add_argument(
        "--ensemble-trajectories",
        type=int,
        default=64,
        help="starting count, doubled until a run is long enough",
    )
    ap.add_argument("--ensemble-seconds", type=float, default=1.0)
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.repeats = [16, 32, 64], 1

    import numba

    numba.set_num_threads(1)
    results = {
        "environment": environment(),
        "workload": "brickwork ring, uniform 2q Cliffords, depth 2n",
        "runs": [],
        "ensemble": [],
    }
    exhausted: set[tuple[str, float]] = set()
    for p in args.rates:
        for n in args.sizes:
            depth = 2 * n
            layers = make_workload(n, p, depth, seed=n)
            check(n, layers, include_v2="sparsegf2_v2" not in args.skip and n <= 512)
            for name, builder in SIMULATORS.items():
                if name in args.skip or n > MAX_N.get(name, 10**9) or (name, p) in exhausted:
                    continue
                try:
                    run = builder(n, layers)
                except ImportError as exc:
                    print(f"  {name}: not installed ({exc})", file=sys.stderr)
                    args.skip.append(name)
                    continue
                seconds = time_runner(run, args.repeats, args.budget)
                results["runs"].append(
                    {"simulator": name, "n": n, "p": p, "depth": depth, "seconds": seconds}
                )
                print(f"p={p:<5} n={n:<5} {name:<26} {seconds:10.4f} s", flush=True)
                if seconds > args.budget:
                    exhausted.add((name, p))
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    for p in args.rates:
        for n in [s for s in args.sizes if s <= 512]:
            spec = mipt.MonitoredCircuit(
                num_system=n,
                edges=graphs.cycle(n),
                layers=2 * n,
                measurement_rate=p,
                schedule="matching",
            )
            spec.run(2, seed=0)
            trajectories = args.ensemble_trajectories
            res = spec.run(trajectories, seed=1)
            while (
                res.elapsed_seconds < args.ensemble_seconds
            ):  # long enough to amortise thread start-up
                trajectories *= 2
                res = spec.run(trajectories, seed=1)
            results["ensemble"].append(
                {
                    "n": n,
                    "p": p,
                    "depth": 2 * n,
                    "threads": numba.get_num_threads(),
                    "trajectories": trajectories,
                    "circuits_per_second": res.trajectories_per_second,
                }
            )
            print(
                f"ensemble p={p} n={n}: {res.trajectories_per_second:,.1f} circuits/s on {numba.get_num_threads()} threads",
                flush=True,
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
