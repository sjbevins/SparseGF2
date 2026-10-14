"""Compare SparseGF2 with Stim on one circuit.

Stim is imported lazily (it is a test dependency, not a runtime one). Random
measurement outcomes are taken from Stim and replayed in SparseGF2 with
``coins``, so with phases the comparison is sign for sign. Circuits with noise
channels are refused: the two simulators draw noise independently.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sparsegf2.circuit import Circuit
from sparsegf2.simulator import Simulator

__all__ = ["compare_measurements", "compare_stabilizers"]


def _stim():
    try:
        import stim
    except ImportError as exc:  # pragma: no cover - exercised only without Stim
        raise ImportError("the Stim cross-check needs Stim: pip install stim") from exc
    return stim


def _check(circuit: Circuit) -> None:
    for ins in circuit.flattened():
        if ins.info.kind == "noise":
            raise ValueError(
                f"the cross-check needs a noise-free circuit; {ins.name} draws noise, which "
                "Stim and SparseGF2 sample independently"
            )


def _stim_run(circuit: Circuit, n: int, seed: int) -> tuple[list[list[str]], np.ndarray]:
    """Stim's canonical stabilizers after every TICK, and its measurement record."""
    stim = _stim()
    sim = stim.TableauSimulator(seed=seed)
    sim.set_num_qubits(n)
    per_tick: list[list[str]] = []
    for ins in circuit.to_stim().flattened():
        if ins.name == "TICK":
            per_tick.append([str(p) for p in sim.canonical_stabilizers()])
        else:
            sim.do(ins)
    return per_tick, np.array(sim.current_measurement_record(), dtype=np.uint8)


def _sparsegf2_stabilizers(
    circuit: Circuit, num_qubits: int, phases: bool, coins: np.ndarray
) -> list[list[str]]:
    sim = Simulator(num_qubits, phases=phases, seed=0)
    result = sim.run(
        circuit,
        coins=coins,
        observe=lambda s, tick: [str(p) for p in s.stabilizers(canonical=True)],
    )
    return [value for _, value in result.observations]


def _unsigned(rows: list[str]) -> list[str]:
    return [r[1:] for r in rows]


def compare_stabilizers(
    circuit: Circuit,
    *,
    num_qubits: int | None = None,
    phases: bool = True,
    ticks=None,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare the canonical stabilizer groups after every ``TICK`` (or the given ticks).

    Signs are compared when ``phases`` (Stim's random outcomes are replayed);
    without phases only the Pauli parts are.
    """
    _check(circuit)
    n = max(circuit.num_qubits, num_qubits or 0, 1)
    stim_rows, record = _stim_run(circuit, n, seed)
    ours = _sparsegf2_stabilizers(circuit, n, phases, record)
    wanted = range(len(stim_rows)) if ticks is None else [int(t) for t in ticks]
    compared = 0
    for t in wanted:
        if not 0 <= t < len(stim_rows):
            raise ValueError(f"tick {t} does not exist; the circuit has {len(stim_rows)} ticks")
        a, b = ours[t], stim_rows[t]
        if not phases:
            a, b = _unsigned(a), _unsigned(b)
        compared += 1
        if a != b:
            return {
                "match": False,
                "ticks_compared": compared,
                "first_mismatch": {"tick": t, "sparsegf2": a, "stim": b},
            }
    return {"match": True, "ticks_compared": compared, "first_mismatch": None}


def compare_measurements(
    circuit: Circuit, *, num_qubits: int | None = None, seed: int = 0
) -> dict[str, Any]:
    """Replay Stim's random outcomes and compare every measurement, detector and observable."""
    _check(circuit)
    n = max(circuit.num_qubits, num_qubits or 0, 1)
    _, record = _stim_run(circuit, n, seed)
    result = Simulator(n, phases=True, seed=0).run(circuit, coins=record)
    ours = {
        "measurements": np.asarray(result.measurements, dtype=np.uint8),
        "detectors": np.asarray(result.detectors, dtype=np.uint8),
        "observables": np.asarray(result.observables, dtype=np.uint8),
    }
    sc = circuit.to_stim()
    events = sc.compile_m2d_converter().convert(
        measurements=record.reshape(1, -1).astype(bool), append_observables=True
    )[0]
    nd = sc.num_detectors
    theirs = {
        "measurements": record,
        "detectors": events[:nd].astype(np.uint8),
        "observables": events[nd:].astype(np.uint8),
    }
    out: dict[str, Any] = {key: int(theirs[key].size) for key in theirs}
    for key in ("measurements", "detectors", "observables"):
        diff = np.flatnonzero(ours[key] != theirs[key])
        if diff.size:
            i = int(diff[0])
            return {
                **out,
                "match": False,
                "first_mismatch": {
                    "kind": key,
                    "index": i,
                    "sparsegf2": int(ours[key][i]),
                    "stim": int(theirs[key][i]),
                },
            }
    return {**out, "match": True, "first_mismatch": None}
