"""
Measurement-step implementations for graph-defined circuits.

MVP supports one measurement mode:

- ``uniform`` — every qubit independently with probability ``p`` per layer.

Future modes (``gated``, ``gated_with_prob``, ...) will slot in with the same
signature and an entry in :data:`MEASUREMENT_MODES`.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


MEASUREMENT_MODES: Tuple[str, ...] = ("uniform",)


def sample_measurements(
    mode: str,
    n: int,
    p: float,
    gate_pairs: Sequence[Tuple[int, int]],
    rng: np.random.Generator,
) -> List[int]:
    """Choose which qubits to measure after a gate layer.

    Parameters
    ----------
    mode : str
        One of :data:`MEASUREMENT_MODES`.
    n : int
        Number of system qubits (measurement candidates in ``uniform`` mode).
    p : float
        Per-qubit measurement probability.
    gate_pairs : sequence of (int, int)
        The gate pairs applied in this layer. Unused in ``uniform`` mode;
        retained in the signature so future ``gated`` modes have access.
    rng : numpy.random.Generator
        RNG for the Bernoulli draws.

    Returns
    -------
    list of int
        Sorted qubit indices to measure this layer (deduplicated).
    """
    if mode != "uniform":
        raise ValueError(
            f"measurement_mode must be one of {MEASUREMENT_MODES}; got {mode!r}"
        )
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    draws = rng.random(n)
    return sorted(int(q) for q in np.nonzero(draws < p)[0])


__all__ = ["MEASUREMENT_MODES", "sample_measurements"]
