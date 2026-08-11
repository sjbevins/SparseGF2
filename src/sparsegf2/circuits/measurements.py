"""Measurement-candidate selection for graph-defined circuits.

After a gate layer fires, choose which of the ``n`` system qubits to
measure in the Z basis this layer. Four modes:

- ``bernoulli``: every one of the ``n`` qubits is an independent
  candidate; each is measured with probability ``p``. (Named ``bernoulli``
  rather than ``uniform`` to avoid overloading the uniform-count idea.)
- ``gated``: only qubits touched by this layer's gate(s) are
  candidates; each measured with probability ``p``.
- ``random_pair``: exactly 2 distinct qubits are sampled uniformly
  (without replacement) as the candidate set; each measured with prob ``p``.
- ``uniform_count``: exactly ``count`` distinct qubits are sampled uniformly
  (without replacement) as the candidate set; each measured with probability
  ``p``. The default count is ``max(1, n // 2)``.

The RNG draws are issued in a deterministic, mode-specific order (see
:mod:`sparsegf2.circuits.scheduler`) so a sweep is bit-for-bit
reproducible from ``(base_seed, sample_seed, layer_index)``.

These functions decide *which qubits* to measure. The measurement
*outcome* (the phase-free coin) is drawn inside
:meth:`sparsegf2.SparseGF2.measure_z` from the simulator's own RNG, a
deliberate separation so the circuit realization and the measurement
outcomes are independent streams.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sparsegf2.errors import InvalidArgumentError

MEASUREMENT_MODES: tuple[str, ...] = ("bernoulli", "gated", "random_pair", "uniform_count")


def sample_measurements_detailed(
    mode: str,
    n: int,
    p: float,
    gate_pairs: Sequence[tuple[int, int]],
    rng: np.random.Generator,
    *,
    count: int | None = None,
) -> tuple[list[int], list[int]]:
    """Return ``(candidates, fired)`` for this layer's measurements.

    A **candidate** is a qubit *eligible* to be measured this layer (the
    candidate set is mode-specific); a candidate **fires** (is actually
    measured) when it passes its Bernoulli(``p``) coin. So ``fired`` is always
    a subset of ``candidates``.

    - ``bernoulli``: candidates are all ``n`` qubits.
    - ``gated``: candidates are the qubits this layer's gates touched.
    - ``random_pair``: candidates are 2 qubits chosen uniformly at random.
    - ``uniform_count``: candidates are ``count`` qubits chosen uniformly at
      random without replacement (``count`` defaults to ``n // 2``).

    The RNG is consumed in exactly the same order as :func:`sample_measurements`
    (candidate selection, where stochastic, then the per-candidate coins), so
    realizations are reproducible bit-for-bit.

    Returns
    -------
    (list of int, list of int)
        ``(candidates, fired)``, both sorted ascending.

    Raises
    ------
    InvalidArgumentError
        If ``mode`` is invalid or ``p`` is out of ``[0, 1]``.
    """
    if mode not in MEASUREMENT_MODES:
        raise InvalidArgumentError(
            f"measurement_mode must be one of {MEASUREMENT_MODES}; got {mode!r}"
        )
    if not 0.0 <= p <= 1.0:
        raise InvalidArgumentError(f"p must be in [0, 1]; got {p}")

    if mode == "bernoulli":
        # Candidates = every qubit; each fires w.p. p. One vectorized draw of n
        # uniforms; np.nonzero returns ascending indices, .tolist() -> ints.
        candidates = list(range(n))
        draws = rng.random(n)
        fired = np.nonzero(draws < p)[0].tolist()
        return candidates, fired

    if mode == "gated":
        # Candidates = the distinct qubits the gate layer touched.
        candidates = sorted({int(q) for pair in gate_pairs for q in pair})
        if not candidates:
            return [], []
        draws = rng.random(len(candidates))
        fired = sorted(candidates[i] for i in range(len(candidates)) if draws[i] < p)
        return candidates, fired

    if mode == "uniform_count":
        # Candidates = `count` distinct qubits chosen uniformly (consumes RNG),
        # each then passed through the Bernoulli(p) gate. Same draw order as
        # random_pair (selection, then coins).
        k = count if count is not None else max(1, n // 2)
        k = max(0, min(int(k), n))
        if k == 0:
            return [], []
        chosen = rng.choice(n, size=k, replace=False)
        candidates = sorted(int(x) for x in chosen)
        draws = rng.random(k)
        fired = sorted(int(chosen[i]) for i in range(k) if draws[i] < p)
        return candidates, fired

    # random_pair: 2 distinct qubits picked uniformly at random (consumes RNG),
    # then each passed through the same Bernoulli(p) gate.
    if n < 2:
        return [], []
    pair = rng.choice(n, size=2, replace=False)
    candidates = sorted(int(x) for x in pair)
    draws = rng.random(2)
    fired = sorted({int(pair[i]) for i in range(2) if draws[i] < p})
    return candidates, fired


def sample_measurements(
    mode: str,
    n: int,
    p: float,
    gate_pairs: Sequence[tuple[int, int]],
    rng: np.random.Generator,
    *,
    count: int | None = None,
) -> list[int]:
    """Choose which qubits to measure (the **fired** subset) after a gate layer.

    Thin wrapper over :func:`sample_measurements_detailed` returning only the
    qubits that fired (passed the Bernoulli(``p``) coin), the ones the runner
    actually measures. Use :func:`sample_measurements_detailed` to also see the
    candidate set. RNG consumption is identical.

    Parameters
    ----------
    mode : str
        One of :data:`MEASUREMENT_MODES`.
    n : int
        Number of system qubits.
    p : float
        Per-qubit measurement probability, in ``[0, 1]``.
    gate_pairs : sequence of (int, int)
        The gate pairs applied this layer (used by ``gated`` for candidates).
    rng : numpy.random.Generator
        RNG for the Bernoulli draws and (in ``random_pair``) the pair choice.

    Returns
    -------
    list of int
        Sorted, deduplicated qubit indices that fired this layer.

    Raises
    ------
    InvalidArgumentError
        If ``mode`` is invalid or ``p`` is out of ``[0, 1]``. (Subclasses
        ``ValueError``, so ``except ValueError`` still catches it.)
    """
    return sample_measurements_detailed(mode, n, p, gate_pairs, rng, count=count)[1]


__all__ = ["MEASUREMENT_MODES", "sample_measurements", "sample_measurements_detailed"]
