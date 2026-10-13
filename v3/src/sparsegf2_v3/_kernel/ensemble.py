"""Compiled engine for ensembles of random monitored circuits.

Each trajectory generates its own random circuit layer by layer from its own
random stream and simulates it immediately (phase-free: entanglement does not
depend on measurement outcomes), evaluating probes at checkpoint layers. The
outer loop over trajectories runs in parallel (``prange``); trajectory ``t``
uses streams ``first + t`` of two keys (circuit structure and probe sampling),
so results never depend on the thread count.

The generator functions (:func:`gen_layer`, :func:`gen_scramble_pairs`) are
shared with :func:`generate_ops`, which records the same draws as operations
instead of applying them. ``MonitoredCircuit.circuit(t)`` is built from those
operations, so an exported circuit is exactly what trajectory ``t`` simulated.

Random draws per trajectory, in order:

1. single-reference partner (one draw, only if chosen at random);
2. scramble pairs: per gate, ``a``, then ``b``, then the gate index;
3. scramble layers, then dynamics layers, each via :func:`gen_layer`: per gate,
   [edge index if edges are sampled], [orientation bit if enabled], gate index;
   then (dynamics only) one uniform per system qubit for the measurement mask.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from sparsegf2_v3._kernel.bits import popcount
from sparsegf2_v3._kernel.entropy import (
    contiguous_distance,
    mutual_information_with,
    prefix_entropies,
    region_entropy,
)
from sparsegf2_v3._kernel.rng import next_below, next_bit, next_double, stream_state
from sparsegf2_v3._kernel.tableau import apply_1q, apply_2q, init_zero, measure_1q

SCHED_MATCHING = 0
SCHED_RANDOM_EDGES = 1
SCHED_ALL_EDGES = 2

REF_NONE = 0
REF_PURIFICATION = 1
REF_SINGLE = 2

PROBE_ENTROPY = 0
PROBE_CODE_DIMENSION = 1
PROBE_CONTIGUOUS_DISTANCE = 2
PROBE_LOST_INFORMATION = 3
PROBE_ERASURE_MEAN_LOST = 4
PROBE_ERASURE_CORRECTABLE = 5
PROBE_MEAN_WEIGHT = 6


@njit(cache=True)
def gen_layer(
    rng, layer, schedule_kind, edges, color_ptr, color_idx, gates_per_layer, pool, n_sys, p,
    with_measurements, orient, ga, gb, gg, mq,
):  # fmt: skip
    """Draw one layer: gates into (ga, gb, gg), measured qubits into mq. Returns (gates, meas)."""
    ng = 0
    num_edges = edges.shape[0]
    base = 0
    if schedule_kind == SCHED_MATCHING:
        c = layer % (color_ptr.shape[0] - 1)
        base = color_ptr[c]
        count = color_ptr[c + 1] - base
    elif schedule_kind == SCHED_RANDOM_EDGES:
        count = gates_per_layer
    else:
        count = num_edges
    for j in range(count):
        if schedule_kind == SCHED_MATCHING:
            e = color_idx[base + j]
        elif schedule_kind == SCHED_RANDOM_EDGES:
            e = next_below(rng, num_edges)
        else:
            e = j
        a = edges[e, 0]
        b = edges[e, 1]
        if orient and next_bit(rng):
            t = a
            a = b
            b = t
        ga[ng] = a
        gb[ng] = b
        gg[ng] = pool[next_below(rng, pool.shape[0])]
        ng += 1
    nm = 0
    if with_measurements:
        for q in range(n_sys):
            if next_double(rng) < p:
                mq[nm] = q
                nm += 1
    return ng, nm


@njit(cache=True)
def gen_scramble_pairs(rng, count, n_sys, pool, ga, gb, gg):
    """``count`` gates on uniformly random distinct pairs of system qubits."""
    for j in range(count):
        a = next_below(rng, n_sys)
        b = next_below(rng, n_sys - 1)
        if b >= a:
            b += 1
        ga[j] = a
        gb[j] = b
        gg[j] = pool[next_below(rng, pool.shape[0])]


@njit(cache=True)
def mean_weight(tab, n, ws):
    """Average number of qubits per stabilizer generator (total support / n)."""
    total = 0
    for q in range(n):
        for w in range(ws):
            total += np.int64(popcount(tab[2 * q, w] | tab[2 * q + 1, w]))
    return total / n


@njit(cache=True)
def eval_probes(
    tab, n, ws, n_sys, refs, kinds, ptr, idx, size_param, sample_param, order, periodic,
    prng, out_row, mask_buf, comp_buf, scratch, union_buf, perm,
):  # fmt: skip
    """Evaluate every probe on the current state into ``out_row``."""
    for j in range(kinds.shape[0]):
        k = kinds[j]
        if k == PROBE_ENTROPY:
            out_row[j] = region_entropy(
                tab, n, ws, idx[ptr[j] : ptr[j + 1]], mask_buf, comp_buf, scratch
            )
        elif k == PROBE_CODE_DIMENSION:
            out_row[j] = region_entropy(tab, n, ws, refs, mask_buf, comp_buf, scratch)
        elif k == PROBE_CONTIGUOUS_DISTANCE:
            out_row[j] = contiguous_distance(tab, n, ws, order, refs, periodic)
        elif k == PROBE_LOST_INFORMATION:
            out_row[j] = mutual_information_with(
                tab, n, ws, refs, idx[ptr[j] : ptr[j + 1]], mask_buf, comp_buf, scratch, union_buf
            )
        elif k == PROBE_ERASURE_MEAN_LOST or k == PROBE_ERASURE_CORRECTABLE:
            m = size_param[j]
            samples = sample_param[j]
            acc = 0.0
            for _ in range(samples):
                for t in range(m):
                    r = t + next_below(prng, n_sys - t)
                    tmp = perm[t]
                    perm[t] = perm[r]
                    perm[r] = tmp
                lost = mutual_information_with(
                    tab, n, ws, refs, perm[:m], mask_buf, comp_buf, scratch, union_buf
                )
                if k == PROBE_ERASURE_MEAN_LOST:
                    acc += lost
                elif lost == 0:
                    acc += 1.0
            out_row[j] = acc / samples
        else:  # PROBE_MEAN_WEIGHT
            out_row[j] = mean_weight(tab, n, ws)


@njit(cache=True)
def _setup(
    tab, n_sys, ref_mode, partner, h_index, cx_index, lin1, anf1, ident1, lin2, anf2, ident2, ws
):
    if ref_mode == REF_PURIFICATION:
        for i in range(n_sys):
            apply_1q(tab, i, h_index, lin1, anf1, ident1, ws, False)
            apply_2q(tab, i, n_sys + i, cx_index, lin2, anf2, ident2, ws, False)
    elif ref_mode == REF_SINGLE:
        apply_1q(tab, partner, h_index, lin1, anf1, ident1, ws, False)
        apply_2q(tab, partner, n_sys, cx_index, lin2, anf2, ident2, ws, False)


@njit(parallel=True, cache=True)
def run_ensemble(
    n_sys, n_ref, ref_mode, fixed_partner, edges, color_ptr, color_idx, schedule_kind, gates_per_layer,
    pool, p, layers, scramble_gates, scramble_layers, orient, checkpoints, stop_purified,
    kinds, ptr, idx, size_param, sample_param, order, periodic, want_profile,
    key_struct, key_probe, first, out_values, out_profile, out_tau, out_final, out_meas,
    h_index, cx_index, lin1, anf1, ident1, lin2, anf2, ident2,
):  # fmt: skip
    """Simulate ``out_values.shape[0]`` trajectories; see the module docstring."""
    num_traj = out_values.shape[0]
    num_ck = checkpoints.shape[0]
    n = n_sys + n_ref
    ws = (n + 63) // 64
    max_gates = max(gates_per_layer, edges.shape[0], scramble_gates, 1)
    for t in prange(num_traj):
        rng = np.empty(4, dtype=np.uint64)
        prng = np.empty(4, dtype=np.uint64)
        stream_state(key_struct, first + t, rng)
        stream_state(key_probe, first + t, prng)
        tab = np.zeros((2 * n + 1, ws), dtype=np.uint64)
        init_zero(tab, n, ws, False)
        anti = np.zeros(ws, dtype=np.uint64)
        supp = np.zeros(max(n, ws), dtype=np.int64)
        lo = np.zeros(ws, dtype=np.uint64)
        hi = np.zeros(ws, dtype=np.uint64)
        qbuf = np.zeros(1, dtype=np.int64)
        cbuf = np.zeros(1, dtype=np.int64)
        ga = np.zeros(max_gates, dtype=np.int64)
        gb = np.zeros(max_gates, dtype=np.int64)
        gg = np.zeros(max_gates, dtype=np.int64)
        mq = np.zeros(n_sys, dtype=np.int64)
        mask_buf = np.zeros(n, dtype=np.bool_)
        comp_buf = np.zeros(n, dtype=np.int64)
        scratch = np.zeros((2 * (n // 2 + 1), ws), dtype=np.uint64)
        union_buf = np.zeros(n, dtype=np.int64)
        perm = np.arange(n_sys).astype(np.int64)
        refs = np.arange(n_sys, n).astype(np.int64)
        profile = np.zeros(order.shape[0] + 1, dtype=np.int64)

        partner = fixed_partner
        if ref_mode == REF_SINGLE and fixed_partner < 0:
            partner = next_below(rng, n_sys)
        _setup(
            tab,
            n_sys,
            ref_mode,
            partner,
            h_index,
            cx_index,
            lin1,
            anf1,
            ident1,
            lin2,
            anf2,
            ident2,
            ws,
        )

        gen_scramble_pairs(rng, scramble_gates, n_sys, pool, ga, gb, gg)
        for j in range(scramble_gates):
            apply_2q(tab, ga[j], gb[j], gg[j], lin2, anf2, ident2, ws, False)
        for s in range(scramble_layers):
            ng, _ = gen_layer(
                rng, s, schedule_kind, edges, color_ptr, color_idx, gates_per_layer, pool, n_sys, p,
                False, orient, ga, gb, gg, mq,
            )  # fmt: skip
            for j in range(ng):
                apply_2q(tab, ga[j], gb[j], gg[j], lin2, anf2, ident2, ws, False)

        tau = -1
        ci = 0
        num_meas = 0
        final = 0
        stopped = False
        for layer in range(layers + 1):
            if layer > 0:
                ng, nm = gen_layer(
                    rng, layer - 1, schedule_kind, edges, color_ptr, color_idx, gates_per_layer, pool,
                    n_sys, p, True, orient, ga, gb, gg, mq,
                )  # fmt: skip
                for j in range(ng):
                    apply_2q(tab, ga[j], gb[j], gg[j], lin2, anf2, ident2, ws, False)
                for j in range(nm):
                    measure_1q(tab, n, ws, False, mq[j], 2, 0, anti, supp, lo, hi, qbuf, cbuf)
                num_meas += nm
                final = layer
            if ci < num_ck and checkpoints[ci] == layer:
                eval_probes(
                    tab, n, ws, n_sys, refs, kinds, ptr, idx, size_param, sample_param, order,
                    periodic, prng, out_values[t, ci], mask_buf, comp_buf, scratch, union_buf, perm,
                )  # fmt: skip
                if want_profile:
                    prefix_entropies(tab, n, ws, order, profile)
                    for k in range(profile.shape[0]):
                        out_profile[t, ci, k] = profile[k]
                if (
                    ref_mode != REF_NONE
                    and tau < 0
                    and region_entropy(tab, n, ws, refs, mask_buf, comp_buf, scratch) == 0
                ):
                    tau = layer
                    stopped = stop_purified
                ci += 1
            if stopped:
                break
        for c in range(ci, num_ck):
            for j in range(out_values.shape[2]):
                out_values[t, c, j] = np.nan
            if want_profile:
                for k in range(out_profile.shape[2]):
                    out_profile[t, c, k] = -1
        out_tau[t] = tau
        out_final[t] = final
        out_meas[t] = num_meas


@njit(cache=True)
def generate_ops(
    n_sys, ref_mode, fixed_partner, edges, color_ptr, color_idx, schedule_kind, gates_per_layer, pool, p,
    layers, scramble_gates, scramble_layers, orient, key_struct, trajectory,
    out_ga, out_gb, out_gg, gate_ptr, out_mq, meas_ptr,
):  # fmt: skip
    """Record trajectory ``trajectory``'s circuit (same draws as :func:`run_ensemble`).

    Gate block 0 holds the scramble pairs, blocks 1..scramble_layers the
    scramble layers, then one block per dynamics layer; ``gate_ptr`` /
    ``meas_ptr`` are CSR offsets (measurement blocks exist for dynamics layers
    only). Returns the reference partner (or -1).
    """
    rng = np.empty(4, dtype=np.uint64)
    stream_state(key_struct, trajectory, rng)
    max_gates = max(gates_per_layer, edges.shape[0], scramble_gates, 1)
    ga = np.zeros(max_gates, dtype=np.int64)
    gb = np.zeros(max_gates, dtype=np.int64)
    gg = np.zeros(max_gates, dtype=np.int64)
    mq = np.zeros(n_sys, dtype=np.int64)
    partner = fixed_partner
    if ref_mode == REF_SINGLE and fixed_partner < 0:
        partner = next_below(rng, n_sys)
    g_pos = 0
    block = 0
    gen_scramble_pairs(rng, scramble_gates, n_sys, pool, ga, gb, gg)
    gate_ptr[0] = 0
    for j in range(scramble_gates):
        out_ga[g_pos] = ga[j]
        out_gb[g_pos] = gb[j]
        out_gg[g_pos] = gg[j]
        g_pos += 1
    block += 1
    gate_ptr[block] = g_pos
    for s in range(scramble_layers):
        ng, _ = gen_layer(
            rng, s, schedule_kind, edges, color_ptr, color_idx, gates_per_layer, pool, n_sys, p,
            False, orient, ga, gb, gg, mq,
        )  # fmt: skip
        for j in range(ng):
            out_ga[g_pos] = ga[j]
            out_gb[g_pos] = gb[j]
            out_gg[g_pos] = gg[j]
            g_pos += 1
        block += 1
        gate_ptr[block] = g_pos
    m_pos = 0
    meas_ptr[0] = 0
    for layer in range(layers):
        ng, nm = gen_layer(
            rng, layer, schedule_kind, edges, color_ptr, color_idx, gates_per_layer, pool, n_sys, p,
            True, orient, ga, gb, gg, mq,
        )  # fmt: skip
        for j in range(ng):
            out_ga[g_pos] = ga[j]
            out_gb[g_pos] = gb[j]
            out_gg[g_pos] = gg[j]
            g_pos += 1
        block += 1
        gate_ptr[block] = g_pos
        for j in range(nm):
            out_mq[m_pos] = mq[j]
            m_pos += 1
        meas_ptr[layer + 1] = m_pos
    return partner
