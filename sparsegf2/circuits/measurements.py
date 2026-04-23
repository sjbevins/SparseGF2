"""
Measurement-step implementations for graph-defined circuits.

Three measurement-candidate modes are supported:

- ``uniform`` — every one of the n system qubits is an independent
  candidate each layer. Each is measured in Z with probability p.
- ``gated`` — only qubits touched by this layer's gate(s) are
  candidates. Each is measured in Z with probability p.
- ``random_pair`` — at each layer exactly 2 distinct system qubits are
  sampled uniformly at random (without replacement) as the candidate
  set. Each is then measured with probability p.

The RNG draws are issued in a deterministic order (see
:mod:`sparsegf2.circuits.builder`) so sweeps are bit-for-bit
reproducible from ``(base_seed, sample_seed, layer_index)``.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


MEASUREMENT_MODES: Tuple[str, ...] = ("uniform", "gated", "random_pair")


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
        Number of system qubits.
    p : float
        Per-qubit measurement probability.
    gate_pairs : sequence of (int, int)
        The gate pairs applied in this layer (used by the ``gated`` mode
        to determine the candidate set).
    rng : numpy.random.Generator
        RNG for the Bernoulli draws and, in ``random_pair`` mode, for
        selecting the random pair of candidates.

    Returns
    -------
    list of int
        Sorted, deduplicated qubit indices to measure this layer.
    """
    if mode not in MEASUREMENT_MODES:
        raise ValueError(
            f"measurement_mode must be one of {MEASUREMENT_MODES}; got {mode!r}"
        )
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")

    if mode == "uniform":
        draws = rng.random(n)
        return sorted(int(q) for q in np.nonzero(draws < p)[0])

    if mode == "gated":
        candidates = sorted({int(q) for pair in gate_pairs for q in pair})
        if not candidates:
            return []
        draws = rng.random(len(candidates))
        return sorted(
            candidates[i] for i in range(len(candidates)) if draws[i] < p
        )

    # random_pair: 2 distinct qubits picked uniformly at random.
    if n < 2:
        return []
    pair = rng.choice(n, size=2, replace=False)
    draws = rng.random(2)
    kept = [int(pair[i]) for i in range(2) if draws[i] < p]
    return sorted(set(kept))


__all__ = ["MEASUREMENT_MODES", "sample_measurements"]
