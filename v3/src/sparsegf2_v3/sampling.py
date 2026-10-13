"""Run one circuit many times, in parallel: :func:`sample`."""

from __future__ import annotations

import operator

import numpy as np

from sparsegf2_v3._compile import compile_circuit
from sparsegf2_v3._kernel.program import run_many
from sparsegf2_v3._kernel.rng import stream_key
from sparsegf2_v3._kernel.tables import ANF1, ANF2, IDENTITY_SYM1, IDENTITY_SYM2, LIN1, LIN2
from sparsegf2_v3._parallel import threads
from sparsegf2_v3.circuit import Circuit
from sparsegf2_v3.result import SampleResult

__all__ = ["sample"]


def sample(
    circuit: Circuit,
    shots: int,
    *,
    seed=None,
    phases: bool = True,
    num_threads: int | None = None,
) -> SampleResult:
    """Run ``circuit`` ``shots`` times from |0...0> and collect the results.

    Shots run in parallel across CPU cores (``num_threads`` limits that). Shot
    ``s`` uses its own random stream derived from ``(seed, s)``, so the result
    is identical whatever the thread count.

    Returns a :class:`~sparsegf2_v3.result.SampleResult` with one row per shot:
    ``measurements``, ``detectors``, ``observables`` (phases mode) and
    ``random`` (both modes).
    """
    shots = operator.index(shots)
    if shots < 0:
        raise ValueError(f"shots must be non-negative, got {shots}")
    program = compile_circuit(circuit)
    n = max(program.num_qubits, 1)
    ws = (n + 63) // 64
    width = 2 * ws if phases else ws
    m = program.num_measurements
    records = np.zeros((shots, m), dtype=np.uint8)
    random_flags = np.zeros((shots, m), dtype=np.uint8)
    key = stream_key(seed)
    if shots:
        with threads(num_threads):
            run_many(
                n, ws, width, bool(phases), program.ops, program.params, program.mpp_qubits,
                program.mpp_codes, key, 0, records, random_flags,
                LIN1, ANF1, IDENTITY_SYM1, LIN2, ANF2, IDENTITY_SYM2,
            )  # fmt: skip
    return SampleResult(bool(phases), program, records, random_flags)
