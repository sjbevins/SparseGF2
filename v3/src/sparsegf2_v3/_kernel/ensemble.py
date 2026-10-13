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

Graphs: ``edges`` holds one or more graphs back to back (graph ``g`` is
``edges[graph_ptr[g]:graph_ptr[g + 1]]``, with its edge colouring and edge
weights stored the same way); trajectory ``t`` runs on graph ``t mod G``. This
is how graph disorder enters (one realization per trajectory) without
touching the random streams.

Random draws per trajectory, in order:

1. single-reference partner (one draw, only if chosen at random);
2. scramble pairs: per gate, ``a``, then ``b``, then the gate index;
3. scramble matchings, per layer: a Fisher-Yates shuffle of the system
   (``n - 1`` draws), then one gate index per pair;
4. scramble layers, then dynamics layers, each via :func:`gen_layer`: per gate,
   [edge: one integer draw, or one uniform draw if edges are weighted, if edges
   are sampled], [orientation bit if enabled], gate index; then (dynamics only)
   one uniform per system qubit for the measurement mask. Scramble layers and
   dynamics layers share one layer count, so the matching schedule's colour
   cycle runs on from the scramble into the dynamics (dynamics layer ``l``,
   counted from 1, uses colour ``(scramble_layers + l - 1) mod C``).

Reference qubits
----------------
With a reference, every trajectory keeps ``ref_rows``, the stabilizer rows with
support on the references, under the invariant of
:func:`~sparsegf2_v3._kernel.tableau.measure_1q_phase_free_tracked`. The code
dimension S(R) is then a popcount, and the lost information I(R:E) of an
erasure costs two column ranks over E only (see the ``*_rows`` routines in
:mod:`~sparsegf2_v3._kernel.entropy`).

Probes
------
Scalar probes write one number per checkpoint. Their qubit sets are stored
flat in ``idx``: probe ``j``'s block starts at ``ptr[j]`` and holds up to three
consecutive regions of lengths ``region_len[j, 0..2]``. Vector probes write
``n_sys + 1`` numbers per checkpoint (entanglement profile, erasure profiles).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from sparsegf2_v3._kernel.bits import popcount
from sparsegf2_v3._kernel.entropy import (
    contiguous_distance_rows,
    erasure_profile_rows,
    lost_information_rows,
    prefix_entropies,
    reference_rows,
    region_entropy,
)
from sparsegf2_v3._kernel.rng import next_below, next_bit, next_double, stream_state
from sparsegf2_v3._kernel.tableau import (
    apply_1q,
    apply_2q,
    init_zero,
    measure_1q_phase_free_tracked,
)

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
PROBE_MUTUAL_INFORMATION = 6
PROBE_TRIPARTITE = 7

VPROBE_ENTROPY_PROFILE = 0
VPROBE_ERASURE_LOST = 1
VPROBE_ERASURE_CORRECTABLE = 2


@njit(cache=True, inline="always")
def _pick_weighted(cum, u):
    """Smallest index e with u < cum[e] (binary search on a cumulative distribution)."""
    lo = 0
    hi = cum.shape[0] - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        if u < cum[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(cache=True)
def gen_layer(
    rng, layer, schedule_kind, edges, edge_cum, color_ptr, color_idx, gates_per_layer, pool, n_sys,
    p, with_measurements, orient, ga, gb, gg, mq,
):  # fmt: skip
    """Draw one layer: gates into (ga, gb, gg), measured qubits into mq. Returns (gates, meas)."""
    ng = 0
    num_edges = edges.shape[0]
    weighted = edge_cum.shape[0] > 0
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
            if weighted:
                e = _pick_weighted(edge_cum, next_double(rng))
            else:
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
def gen_scramble_matching(rng, n_sys, pool, perm, ga, gb, gg):
    """A uniformly random perfect matching of the system (one qubit idles if n is odd).

    Pairs ``(perm[2j], perm[2j+1])`` of a fresh uniform permutation, each with a
    gate drawn from ``pool``. Returns the number of gates, ``n_sys // 2``.
    """
    for i in range(n_sys):
        perm[i] = i
    for i in range(n_sys - 1):
        r = i + next_below(rng, n_sys - i)
        tmp = perm[i]
        perm[i] = perm[r]
        perm[r] = tmp
    for j in range(n_sys // 2):
        ga[j] = perm[2 * j]
        gb[j] = perm[2 * j + 1]
        gg[j] = pool[next_below(rng, pool.shape[0])]
    return n_sys // 2


@njit(cache=True)
def _max_gates(n_sys, graph_ptr, gates_per_layer, scramble_gates):
    """Gate-buffer size: the largest layer any schedule or scramble can produce."""
    m = max(gates_per_layer, scramble_gates, n_sys // 2, 1)
    for g in range(graph_ptr.shape[0] - 1):
        m = max(m, graph_ptr[g + 1] - graph_ptr[g])
    return m


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
def _entropy_of_union(tab, n, ws, idx, starts, lens, which, mask_buf, comp_buf, scratch, union_buf):
    """S of the union of the regions selected by the bit mask ``which`` (bit r = region r)."""
    m = 0
    for r in range(3):
        if (which >> r) & 1:
            for t in range(lens[r]):
                union_buf[m] = idx[starts[r] + t]
                m += 1
    return region_entropy(tab, n, ws, union_buf[:m], mask_buf, comp_buf, scratch)


@njit(cache=True)
def eval_probes(
    tab, n, ws, n_sys, ref_rows, n_ref, s_ref, kinds, ptr, region_len, idx, size_param,
    sample_param, order, periodic, prng, out_row, mask_buf, comp_buf, scratch, union_buf, perm,
    erase_scratch,
):  # fmt: skip
    """Evaluate every scalar probe on the current state into ``out_row``.

    ``s_ref`` is the reference entropy S(R), computed once per checkpoint by the
    caller (it is both the code-dimension probe and the purification test).
    ``erase_scratch`` has shape >= (2 n_sys, ws).
    """
    starts = np.zeros(3, dtype=np.int64)
    lens = np.zeros(3, dtype=np.int64)
    region_buf = np.zeros(n, dtype=np.int64)
    for j in range(kinds.shape[0]):
        k = kinds[j]
        s0 = ptr[j]
        for r in range(3):
            starts[r] = s0
            lens[r] = region_len[j, r]
            s0 += region_len[j, r]
        if k == PROBE_ENTROPY:
            out_row[j] = _entropy_of_union(
                tab, n, ws, idx, starts, lens, 1, mask_buf, comp_buf, scratch, union_buf
            )
        elif k == PROBE_CODE_DIMENSION:
            out_row[j] = s_ref
        elif k == PROBE_CONTIGUOUS_DISTANCE:
            if s_ref == 0:
                out_row[j] = np.nan  # no code, no distance (never averaged as 0)
            else:
                out_row[j] = contiguous_distance_rows(tab, ws, order, ref_rows, n_ref, periodic)
        elif k == PROBE_LOST_INFORMATION:
            for t in range(lens[0]):
                region_buf[t] = idx[starts[0] + t]
            out_row[j] = lost_information_rows(
                tab, ws, region_buf[: lens[0]], ref_rows, erase_scratch
            )
        elif k == PROBE_MUTUAL_INFORMATION or k == PROBE_TRIPARTITE:
            # sum over nonempty subsets T of the regions of (-1)^(|T|+1) S(T)
            num_regions = 2 if k == PROBE_MUTUAL_INFORMATION else 3
            total = 0
            for which in range(1, 1 << num_regions):
                bits = 0
                for r in range(num_regions):
                    bits += (which >> r) & 1
                s = _entropy_of_union(
                    tab, n, ws, idx, starts, lens, which, mask_buf, comp_buf, scratch, union_buf
                )
                total += s if bits % 2 == 1 else -s
            out_row[j] = total
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
                lost = lost_information_rows(tab, ws, perm[:m], ref_rows, erase_scratch)
                if k == PROBE_ERASURE_MEAN_LOST:
                    acc += lost
                elif lost == 0:
                    acc += 1.0
            out_row[j] = acc / samples


@njit(cache=True)
def eval_vector_probes(
    tab, n, ws, n_sys, ref_rows, vkinds, vsamples, order, prng, out_block, perm, work
):
    """Evaluate every vector probe into ``out_block[v, :]`` (length n_sys + 1 each)."""
    for v in range(vkinds.shape[0]):
        k = vkinds[v]
        if k == VPROBE_ENTROPY_PROFILE:
            prefix_entropies(tab, n, ws, order, work)
            for i in range(n_sys + 1):
                out_block[v, i] = work[i]
        else:
            samples = vsamples[v]
            for i in range(n_sys + 1):
                out_block[v, i] = 0.0
            for _ in range(samples):
                for t in range(n_sys - 1):  # full Fisher-Yates shuffle of the system
                    r = t + next_below(prng, n_sys - t)
                    tmp = perm[t]
                    perm[t] = perm[r]
                    perm[r] = tmp
                erasure_profile_rows(tab, ws, perm, ref_rows, work)
                for i in range(n_sys + 1):
                    if k == VPROBE_ERASURE_LOST:
                        out_block[v, i] += work[i]
                    elif work[i] == 0:
                        out_block[v, i] += 1.0
            for i in range(n_sys + 1):
                out_block[v, i] /= samples


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
    n_sys, n_ref, ref_mode, fixed_partner, edges, graph_ptr, edge_cum, color_ptr, color_ptr_ptr,
    color_idx, schedule_kind, gates_per_layer, pool, p, layers, scramble_gates,
    scramble_matchings, scramble_layers, orient, checkpoints,
    stop_purified, kinds, ptr, region_len, idx, size_param, sample_param, vkinds, vsamples, order,
    periodic, key_struct, key_probe, first, out_values, out_vectors, out_tau, out_final, out_meas,
    h_index, cx_index, lin1, anf1, ident1, lin2, anf2, ident2,
):  # fmt: skip
    """Simulate ``out_values.shape[0]`` trajectories; see the module docstring."""
    num_traj = out_values.shape[0]
    num_ck = checkpoints.shape[0]
    n = n_sys + n_ref
    ws = (n + 63) // 64
    max_gates = _max_gates(n_sys, graph_ptr, gates_per_layer, scramble_gates)
    num_graphs = graph_ptr.shape[0] - 1
    for t in prange(num_traj):
        g = (first + t) % num_graphs
        g_edges = edges[graph_ptr[g] : graph_ptr[g + 1]]
        g_cum = edge_cum[graph_ptr[g] : graph_ptr[g + 1]]  # empty when unweighted
        g_cidx = color_idx[graph_ptr[g] : graph_ptr[g + 1]]
        g_cptr = color_ptr[color_ptr_ptr[g] : color_ptr_ptr[g + 1]]
        rng = np.empty(4, dtype=np.uint64)
        prng = np.empty(4, dtype=np.uint64)
        stream_state(key_struct, first + t, rng)
        stream_state(key_probe, first + t, prng)
        tab = np.zeros((2 * n + 1, ws), dtype=np.uint64)
        init_zero(tab, n, ws, False)
        anti = np.zeros(ws, dtype=np.uint64)
        supp = np.zeros(max(n, ws), dtype=np.int64)
        ga = np.zeros(max_gates, dtype=np.int64)
        gb = np.zeros(max_gates, dtype=np.int64)
        gg = np.zeros(max_gates, dtype=np.int64)
        mq = np.zeros(n_sys, dtype=np.int64)
        mask_buf = np.zeros(n, dtype=np.bool_)
        comp_buf = np.zeros(n, dtype=np.int64)
        scratch = np.zeros((2 * (n // 2 + 1), ws), dtype=np.uint64)
        union_buf = np.zeros(n, dtype=np.int64)
        erase_scratch = np.zeros((2 * n_sys, ws), dtype=np.uint64)
        ref_rows = np.zeros(ws, dtype=np.uint64)
        perm = np.arange(n_sys).astype(np.int64)
        mperm = np.zeros(n_sys, dtype=np.int64)
        refs = np.arange(n_sys, n).astype(np.int64)
        work = np.zeros(n_sys + 1, dtype=np.int64)

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
        reference_rows(tab, ws, refs, ref_rows)

        gen_scramble_pairs(rng, scramble_gates, n_sys, pool, ga, gb, gg)
        for j in range(scramble_gates):
            apply_2q(tab, ga[j], gb[j], gg[j], lin2, anf2, ident2, ws, False)
        for _ in range(scramble_matchings):
            ng = gen_scramble_matching(rng, n_sys, pool, mperm, ga, gb, gg)
            for j in range(ng):
                apply_2q(tab, ga[j], gb[j], gg[j], lin2, anf2, ident2, ws, False)
        for s in range(scramble_layers):
            ng, _ = gen_layer(
                rng, s, schedule_kind, g_edges, g_cum, g_cptr, g_cidx, gates_per_layer,
                pool, n_sys, p, False, orient, ga, gb, gg, mq,
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
                # the matching schedule's colour cycle continues from the scramble layers
                ng, nm = gen_layer(
                    rng, scramble_layers + layer - 1, schedule_kind, g_edges, g_cum, g_cptr,
                    g_cidx, gates_per_layer, pool, n_sys, p, True, orient, ga, gb, gg, mq,
                )  # fmt: skip
                for j in range(ng):
                    apply_2q(tab, ga[j], gb[j], gg[j], lin2, anf2, ident2, ws, False)
                for j in range(nm):
                    measure_1q_phase_free_tracked(tab, n, ws, mq[j], 2, anti, supp, ref_rows)
                num_meas += nm
                final = layer
            if ci < num_ck and checkpoints[ci] == layer:
                s_ref = -n_ref
                for w in range(ws):
                    s_ref += np.int64(popcount(ref_rows[w]))
                eval_probes(
                    tab, n, ws, n_sys, ref_rows, n_ref, s_ref, kinds, ptr, region_len, idx,
                    size_param, sample_param, order, periodic, prng, out_values[t, ci], mask_buf,
                    comp_buf, scratch, union_buf, perm, erase_scratch,
                )  # fmt: skip
                eval_vector_probes(
                    tab, n, ws, n_sys, ref_rows, vkinds, vsamples, order, prng, out_vectors[t, ci],
                    perm, work,
                )  # fmt: skip
                if ref_mode != REF_NONE and tau < 0 and s_ref == 0:
                    tau = layer
                    stopped = stop_purified
                ci += 1
            if stopped:
                break
        for c in range(ci, num_ck):
            for j in range(out_values.shape[2]):
                out_values[t, c, j] = np.nan
            for v in range(out_vectors.shape[2]):
                for i in range(out_vectors.shape[3]):
                    out_vectors[t, c, v, i] = np.nan
        out_tau[t] = tau
        out_final[t] = final
        out_meas[t] = num_meas


@njit(cache=True)
def generate_ops(
    n_sys, ref_mode, fixed_partner, edges, graph_ptr, edge_cum, color_ptr, color_ptr_ptr,
    color_idx, schedule_kind, gates_per_layer, pool, p, layers, scramble_gates,
    scramble_matchings, scramble_layers, orient, key_struct, trajectory, out_ga, out_gb, out_gg,
    gate_ptr, out_mq, meas_ptr,
):  # fmt: skip
    """Record trajectory ``trajectory``'s circuit (same draws as :func:`run_ensemble`).

    Gate block 0 holds the scramble pairs, the next ``scramble_matchings``
    blocks the scramble matchings, then ``scramble_layers`` blocks of scramble
    layers, then one block per dynamics layer; ``gate_ptr`` / ``meas_ptr`` are
    CSR offsets (measurement blocks exist for dynamics layers only). Returns
    the reference partner (or -1).
    """
    rng = np.empty(4, dtype=np.uint64)
    stream_state(key_struct, trajectory, rng)
    max_gates = _max_gates(n_sys, graph_ptr, gates_per_layer, scramble_gates)
    g = trajectory % (graph_ptr.shape[0] - 1)
    g_edges = edges[graph_ptr[g] : graph_ptr[g + 1]]
    g_cum = edge_cum[graph_ptr[g] : graph_ptr[g + 1]]  # empty when unweighted
    g_cidx = color_idx[graph_ptr[g] : graph_ptr[g + 1]]
    g_cptr = color_ptr[color_ptr_ptr[g] : color_ptr_ptr[g + 1]]
    mperm = np.zeros(n_sys, dtype=np.int64)
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
    for _ in range(scramble_matchings):
        ng = gen_scramble_matching(rng, n_sys, pool, mperm, ga, gb, gg)
        for j in range(ng):
            out_ga[g_pos] = ga[j]
            out_gb[g_pos] = gb[j]
            out_gg[g_pos] = gg[j]
            g_pos += 1
        block += 1
        gate_ptr[block] = g_pos
    for s in range(scramble_layers):
        ng, _ = gen_layer(
            rng, s, schedule_kind, g_edges, g_cum, g_cptr, g_cidx, gates_per_layer, pool,
            n_sys, p, False, orient, ga, gb, gg, mq,
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
            rng, scramble_layers + layer, schedule_kind, g_edges, g_cum, g_cptr, g_cidx,
            gates_per_layer, pool, n_sys, p, True, orient, ga, gb, gg, mq,
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
