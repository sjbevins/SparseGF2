"""
Standalone QA agent for the ``single_ref`` MIPT-probe picture.

Sweeps a matrix of configurations and verifies:

1. The reference qubit at index ``n`` is NEVER involved in a gate or
   measurement during the circuit. The circuit schedule is inspected
   directly via :class:`CircuitBuilder`.

2. For a single sample the reported reference entropy ``k`` (stored in
   ``SampleRecord.k`` when ``picture="single_ref"``) is always an integer
   in ``{0, 1}`` -- the entropy of a single qubit in a stabilizer state
   is either 0 (pure) or 1 (maximally mixed).

3. Known limits of the MIPT probe:
     - at ``p == 0.0`` (no measurements) the initial Bell pair between
       system qubit 0 and reference qubit n is preserved by any Clifford
       on qubits 0..n-1, so S(qubit n) == 1.
     - at ``p == 1.0`` (every qubit measured every layer) the Bell-pair
       correlation is fully collapsed by the first measurement of qubit
       0, so S(qubit n) == 0.

Matrix: graph in {"complete" (all-to-all), "cycle" (brickwork / 1D)},
size n in {4, 8, 12}, rate p in {0.0, 1.0}.

Runnable both as a pytest target (functions prefixed ``test_``) and as
a standalone script (``python tests/validate_single_ref.py``).
"""
from __future__ import annotations

import sys
from itertools import product
from typing import Iterator, Tuple

from sparsegf2.circuits import (
    CircuitConfig,
    SimulationRunner,
)
from sparsegf2.circuits.builder import CircuitBuilder


# "all-to-all" and "brickwork" in the task vocabulary map to the
# repository's existing graph specs "complete" and "cycle".
GRAPH_LABELS = {
    "complete": "all-to-all",
    "cycle":    "brickwork",
}
SIZES = (4, 8, 12)
RATES = (0.0, 1.0)


def _iter_matrix() -> Iterator[Tuple[str, int, float]]:
    for graph in GRAPH_LABELS:
        for n in SIZES:
            for p in RATES:
                yield graph, n, p


def _assert_ref_qubit_untouched(cfg: CircuitConfig) -> int:
    """Walk the full circuit schedule for ``cfg`` and confirm that qubit
    ``cfg.n`` (the reference) never appears in any gate pair or
    measurement list. Returns the total number of gate+meas ops
    inspected (for reporting).
    """
    builder = CircuitBuilder(cfg, sample_seed=0)
    total = 0
    n = cfg.n
    for layer in builder.layers():
        for qi, qj in layer.gate_pairs:
            assert qi < n and qj < n, (
                f"{cfg.graph_spec}/n={n}/p={cfg.p}: gate on ref qubit "
                f"({qi},{qj}); expected both < {n}"
            )
            total += 1
        for q in layer.meas_qubits:
            assert q < n, (
                f"{cfg.graph_spec}/n={n}/p={cfg.p}: measurement on ref qubit "
                f"q={q}; expected < {n}"
            )
            total += 1
    return total


def _one_sample_k(cfg: CircuitConfig, *, seed: int = 0) -> int:
    runner = SimulationRunner(cfg, warmup_jit=False)
    rec = runner.run(sample_seed=seed)
    k = rec.k
    assert isinstance(k, int), (
        f"{cfg.graph_spec}/n={cfg.n}/p={cfg.p}: k is {type(k).__name__}, "
        "expected int"
    )
    assert k in (0, 1), (
        f"{cfg.graph_spec}/n={cfg.n}/p={cfg.p}: k={k} not in {{0, 1}} "
        "(single-qubit stabilizer entropy must be 0 or 1)"
    )
    return k


def _cfg(graph: str, n: int, p: float) -> CircuitConfig:
    return CircuitConfig(
        graph_spec=graph,
        n=n,
        picture="single_ref",
        matching_mode="round_robin",
        p=p,
        depth_factor=4,
    )


# pytest-style harness

def test_ref_qubit_never_touched():
    """Sweep assertion (1): ref qubit never in any gate or measurement."""
    for graph, n, p in _iter_matrix():
        cfg = _cfg(graph, n, p)
        ops = _assert_ref_qubit_untouched(cfg)
        assert ops > 0, (
            f"{graph}/n={n}/p={p}: circuit had zero ops (did builder fail?)"
        )


def test_k_is_integer_0_or_1():
    """Sweep assertion (2): single-sample k is strictly 0 or 1."""
    for graph, n, p in _iter_matrix():
        _one_sample_k(_cfg(graph, n, p))


def test_known_limits():
    """Sweep assertion (3): p=0 -> k=1; p=1 -> k=0 on all (graph, n)."""
    for graph in GRAPH_LABELS:
        for n in SIZES:
            k0 = _one_sample_k(_cfg(graph, n, p=0.0))
            assert k0 == 1, (
                f"{graph}/n={n}/p=0.0: k={k0}, expected 1 "
                "(no measurements, initial Bell pair preserved)"
            )
            k1 = _one_sample_k(_cfg(graph, n, p=1.0))
            assert k1 == 0, (
                f"{graph}/n={n}/p=1.0: k={k1}, expected 0 "
                "(full measurement collapses Bell-pair correlation)"
            )


# Standalone script entry point

def run_all(verbose: bool = True) -> int:
    """Execute the full QA sweep and return a POSIX exit code."""
    ok = True
    for graph, n, p in _iter_matrix():
        cfg = _cfg(graph, n, p)
        label = GRAPH_LABELS[graph]
        try:
            ops = _assert_ref_qubit_untouched(cfg)
            k = _one_sample_k(cfg)
            # Known limit checks
            if p == 0.0 and k != 1:
                raise AssertionError(
                    f"{graph}/n={n}/p=0.0: expected k=1, got {k}"
                )
            if p == 1.0 and k != 0:
                raise AssertionError(
                    f"{graph}/n={n}/p=1.0: expected k=0, got {k}"
                )
        except AssertionError as exc:
            ok = False
            print(f"FAIL  {label}/n={n}/p={p}: {exc}")
            continue
        if verbose:
            print(
                f"PASS  {label:10s}  n={n:2d}  p={p:.1f}  "
                f"ops_inspected={ops:5d}  k={k}"
            )
    if ok:
        print("\nAll single_ref QA sweep assertions PASSED.")
        return 0
    print("\nSingle_ref QA sweep FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(run_all(verbose=True))
