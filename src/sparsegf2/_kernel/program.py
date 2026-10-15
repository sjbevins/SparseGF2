"""The compiled loop that executes a :class:`~sparsegf2._compile.Program`.

Randomness (all from the simulator's xoshiro256** stream, in program order):

* each measurement result consumes one bit (its coin), then one float if it
  has a flip probability;
* each reset consumes one bit;
* each noise target (or target pair) consumes one float.

``coins`` (when ``use_coins``) replaces the drawn coin of measurement ``m`` by
``coins[m]``, which is the *recorded* value the measurement should produce if
its outcome is random (inversion is undone before collapsing). The stream still
advances, so forcing outcomes never shifts the randomness that follows.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from sparsegf2._kernel.rng import next_bit, next_double, stream_state
from sparsegf2._kernel.tableau import (
    apply_1q,
    apply_2q,
    apply_pauli,
    init_zero,
    measure_1q,
    measure_pauli,
)

OP_1Q = 1
OP_2Q = 2
OP_MEASURE = 3
OP_MPP = 4
OP_RESET = 5
OP_NOISE1 = 6
OP_NOISE2 = 7
OP_FEEDBACK = 8
OP_TICK = 9

# Pauli code for each index of Stim's "IXYZ" letter order.
_LETTER_CODE = (0, 1, 3, 2)


@njit(cache=True, inline="always")
def _flip_code(basis_code):
    """A Pauli that anticommutes with the measured basis (used by resets)."""
    return 2 if basis_code == 1 else 1


@njit(cache=True)
def _sample_pauli_1(channel, params, slot, u):
    """Pauli code (0 = none) for a 1-qubit channel given a uniform draw ``u``."""
    if channel <= 2:  # X_ERROR, Y_ERROR, Z_ERROR
        if u < params[slot]:
            return 1 if channel == 0 else (3 if channel == 1 else 2)
        return 0
    if channel == 3:  # DEPOLARIZE1
        p = params[slot]
        if u < p:
            k = min(int(u / p * 3.0), 2)
            return 1 if k == 0 else (3 if k == 1 else 2)
        return 0
    # PAULI_CHANNEL_1(px, py, pz)
    acc = params[slot]
    if u < acc:
        return 1
    acc += params[slot + 1]
    if u < acc:
        return 3
    acc += params[slot + 2]
    if u < acc:
        return 2
    return 0


@njit(cache=True)
def _sample_pauli_2(channel, params, slot, u):
    """Index j in 1..15 of a 2-qubit Pauli in Stim's IX, IY, ..., ZZ order (0 = none)."""
    if channel == 5:  # DEPOLARIZE2
        p = params[slot]
        if u < p:
            return min(int(u / p * 15.0), 14) + 1
        return 0
    acc = 0.0  # PAULI_CHANNEL_2
    for j in range(15):
        acc += params[slot + j]
        if u < acc:
            return j + 1
    return 0


@njit(cache=True)
def run_program(
    tab, n, ws, phases, ops, params, mpp_qubits, mpp_codes,
    start, stop_at_tick, coins, use_coins, rng, record, random_flags, m_start,
    lin1, anf1, ident1, lin2, anf2, ident2, anti, supp, lo, hi, qbuf, cbuf, reference,
):  # fmt: skip
    """Execute ``ops[start:]``; return ``(next_op, measurements_done, stopped_at_tick)``.

    With ``stop_at_tick`` the loop returns right after the next TICK. With
    ``reference`` the run is Stim's noiseless *reference sample*: noise is
    skipped and every collapse picks the +1 eigenstate (no randomness used).
    """
    m = m_start
    i = start
    num_ops = ops.shape[0]
    while i < num_ops:
        op = ops[i, 0]
        if op == OP_2Q:
            apply_2q(tab, ops[i, 1], ops[i, 2], ops[i, 3], lin2, anf2, ident2, ws, phases)
        elif op == OP_1Q:
            apply_1q(tab, ops[i, 1], ops[i, 2], lin1, anf1, ident1, ws, phases)
        elif op == OP_MEASURE:
            q = ops[i, 1]
            code = ops[i, 2]
            flags = ops[i, 3]
            invert = flags & 1
            if reference:
                coin = invert
            else:
                coin = next_bit(rng)
                if use_coins:
                    coin = coins[m]
            r = measure_1q(
                tab, n, ws, phases, q, code, coin ^ invert, anti, supp, lo, hi, qbuf, cbuf
            )
            outcome = r & 1
            random_flags[m] = (r >> 1) & 1
            if (flags & 2) and phases and outcome:
                apply_pauli(tab, q, _flip_code(code), ws)
            bit = outcome ^ invert
            if not reference and ops[i, 4] >= 0 and next_double(rng) < params[ops[i, 4]]:
                bit ^= 1
            record[m] = bit
            m += 1
        elif op == OP_MPP:
            a = ops[i, 1]
            length = ops[i, 2]
            invert = ops[i, 3] & 1
            if reference:
                coin = invert
            else:
                coin = next_bit(rng)
                if use_coins:
                    coin = coins[m]
            r = measure_pauli(
                tab, n, ws, phases, mpp_qubits[a : a + length], mpp_codes[a : a + length],
                coin ^ invert, anti, supp, lo, hi,
            )  # fmt: skip
            random_flags[m] = (r >> 1) & 1
            bit = (r & 1) ^ invert
            if not reference and ops[i, 4] >= 0 and next_double(rng) < params[ops[i, 4]]:
                bit ^= 1
            record[m] = bit
            m += 1
        elif op == OP_RESET:
            q = ops[i, 1]
            code = ops[i, 2]
            coin = 0 if reference else next_bit(rng)
            r = measure_1q(tab, n, ws, phases, q, code, coin, anti, supp, lo, hi, qbuf, cbuf)
            if phases and (r & 1):
                apply_pauli(tab, q, _flip_code(code), ws)
        elif reference and (op == OP_NOISE1 or op == OP_NOISE2):
            pass
        elif op == OP_NOISE1:
            pauli = _sample_pauli_1(ops[i, 2], params, ops[i, 3], next_double(rng))
            if phases and pauli != 0:
                apply_pauli(tab, ops[i, 1], pauli, ws)
        elif op == OP_NOISE2:
            j = _sample_pauli_2(ops[i, 3], params, ops[i, 4], next_double(rng))
            if phases and j != 0:
                pa = _LETTER_CODE[j >> 2]
                pb = _LETTER_CODE[j & 3]
                if pa != 0:
                    apply_pauli(tab, ops[i, 1], pa, ws)
                if pb != 0:
                    apply_pauli(tab, ops[i, 2], pb, ws)
        elif op == OP_FEEDBACK:
            if phases and record[ops[i, 1]] != 0:
                apply_pauli(tab, ops[i, 2], ops[i, 3], ws)
        elif op == OP_TICK and stop_at_tick:
            return i + 1, m, True
        i += 1
    return num_ops, m, False


@njit(parallel=True, cache=True)
def run_many(
    n, ws, width, phases, ops, params, mpp_qubits, mpp_codes, key, first_shot,
    records, random_flags, lin1, anf1, ident1, lin2, anf2, ident2,
):  # fmt: skip
    """Run the program once per row of ``records``, each from |0...0>, in parallel.

    Shot ``s`` uses random stream ``first_shot + s`` of ``key`` (randomness
    contract of :func:`~sparsegf2._kernel.rng.stream_state`), so results do not
    depend on the number of threads.
    """
    shots = records.shape[0]
    for s in prange(shots):
        tab = np.zeros((2 * n + 1, width), dtype=np.uint64)
        init_zero(tab, n, ws, phases)
        rng = np.empty(4, dtype=np.uint64)
        stream_state(key, first_shot + s, rng)
        anti = np.zeros(width, dtype=np.uint64)
        supp = np.zeros(max(n, ws), dtype=np.int64)
        lo = np.zeros(ws, dtype=np.uint64)
        hi = np.zeros(ws, dtype=np.uint64)
        qbuf = np.zeros(1, dtype=np.int64)
        cbuf = np.zeros(1, dtype=np.int64)
        coins = np.zeros(0, dtype=np.uint8)
        run_program(
            tab, n, ws, phases, ops, params, mpp_qubits, mpp_codes, 0, False, coins, False,
            rng, records[s], random_flags[s], 0,
            lin1, anf1, ident1, lin2, anf2, ident2, anti, supp, lo, hi, qbuf, cbuf, False,
        )  # fmt: skip
