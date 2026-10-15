"""Compiled engine for ensembles of random monitored circuits.

Each trajectory generates its own random circuit layer by layer from its own
random stream and simulates it immediately (phase-free: entanglement does not
depend on measurement outcomes), evaluating probes at checkpoint layers. The
outer loop over trajectories runs in parallel (``prange``); trajectory ``t``
draws its circuit from stream ``first + t`` of the circuit key, and each of its
sampled-probe evaluations from a stream of its own under the probe key (see
Probes), so results never depend on the thread count.

The generator functions (:func:`gen_layer`, :func:`gen_scramble_pairs`) are
shared with :func:`generate_ops`, which records the same draws as operations
instead of applying them. ``MonitoredCircuit.circuit(t)`` is built from those
operations, so an exported circuit is exactly what trajectory ``t`` simulated.

Graphs: ``edges`` holds one or more graphs back to back (graph ``g`` is
``edges[graph_ptr[g]:graph_ptr[g + 1]]``, with its edge colouring and edge
weights stored the same way). With several graphs, trajectory ``t`` runs on
graph ``t // circuits_per_graph``: each graph carries a block of
``circuits_per_graph`` consecutive trajectories. This is how graph disorder
enters without touching the random streams.

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
:func:`~sparsegf2._kernel.tableau.measure_1q_phase_free_tracked`. The code
dimension S(R) is then a popcount, and the lost information I(R:E) of an
erasure costs two column ranks over E only (see the ``*_rows`` routines in
:mod:`~sparsegf2._kernel.entropy`).

Probes
------
Every probe writes ``plen[j]`` numbers each time it is evaluated, into its own
slice of the trajectory's flat output row: evaluation ``c`` of probe ``j``
fills ``out[t, poff[j] + c * plen[j] : ... + plen[j]]``. Probe ``j`` is
evaluated at its own checkpoint layers ``pck[pck_ptr[j]:pck_ptr[j + 1]]``
(sorted); the engine stops at the union ``gck`` of all of them, and tests
purification only where ``detect`` is set. Qubit sets are stored flat in
``idx``: probe ``j``'s block starts at ``ptr[j]`` and holds up to three
consecutive regions of lengths ``region_len[j, 0..2]``. Evaluations that a
trajectory never reaches (``stop_when_purified``) are filled with ``pfill[j]``,
the probe's exact value once the reference is pure, or NaN where that value
is not determined.

Probe streams (randomness contract 2): the sampled probes
(``erasure_lost_profile``, ``erasure_first_failure``) draw at each evaluation
from a fresh stream, ``stream_state(key_probe, mix(first + t, phash[j], L))``
for probe ``j`` evaluated at layer ``L`` of trajectory ``first + t``, where
``mix(a, b, c) = sm(sm(sm(a) ^ b) ^ c)`` is chained splitmix64
(:mod:`~sparsegf2._kernel.rng`) and ``phash[j]`` is the first 8 bytes of
``blake2b(name.encode("utf-8"), digest_size=8)`` of the probe's name, read
little-endian (computed in Python). The engine seeds ``prng`` this way for
every evaluation of a probe with ``sample_param[j] > 0``, which every sampled
kind has. Every evaluation starts its orderings from the identity
(``erasure_lost_profile``: the qubits ``0 .. n - 1``;
``erasure_first_failure``: the positions in ``order``), and its samples are
Fisher-Yates shuffles drawn one after another, each of the previous ordering.
So the samples at one evaluation depend only on (seed, trajectory, probe name,
layer, samples) and the state, never on other probes, on the checkpoints or
on earlier evaluations: replaying a trajectory with other probes or
checkpoints reproduces them, and fewer samples are a prefix of more.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from sparsegf2._kernel.bits import popcount
from sparsegf2._kernel.entropy import (
    code_snapshot_rows,
    contiguous_distance_rows,
    erasure_profile_rows,
    first_failure_samples,
    lost_information_rows,
    prefix_entropies,
    reference_rows,
    region_entropy,
    uncorrectable_counts_rows,
)
from sparsegf2._kernel.rng import mix, next_below, next_bit, next_double, stream_state
from sparsegf2._kernel.tableau import (
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
PROBE_MUTUAL_INFORMATION = 4
PROBE_TRIPARTITE = 5
PROBE_ENTROPY_PROFILE = 6
PROBE_ERASURE_LOST_PROFILE = 7
PROBE_FIRST_FAILURE = 8
PROBE_UNCORRECTABLE_COUNTS = 9
PROBE_CODE_SNAPSHOT = 10


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
def _shuffle(prng, perm):
    """Uniformly random permutation of ``perm`` in place (Fisher-Yates, len - 1 draws)."""
    n = perm.shape[0]
    for t in range(n - 1):
        r = t + next_below(prng, n - t)
        tmp = perm[t]
        perm[t] = perm[r]
        perm[r] = tmp


@njit(cache=True)
def eval_probe(
    j, kind, row, tab, n, ws, n_sys, ref_rows, n_ref, s_ref, ptr, region_len, idx, size_param,
    sample_param, order, periodic, prng, mask_buf, comp_buf, scratch, union_buf, perm,
    erase_scratch, work, snap_out,
):  # fmt: skip
    """Evaluate probe ``j`` (kernel code ``kind``) on the current state into ``row``.

    ``s_ref`` is the reference entropy S(R) = k, read once per checkpoint by
    the caller. Code and erasure probes take their trivial values when k = 0.
    ``snap_out`` receives the code's generators for a snapshot probe. A
    sampled probe draws from ``prng``, which the caller seeds for this
    evaluation alone (module docstring, Probe streams).
    """
    starts = np.zeros(3, dtype=np.int64)
    lens = np.zeros(3, dtype=np.int64)
    s0 = ptr[j]
    for r in range(3):
        starts[r] = s0
        lens[r] = region_len[j, r]
        s0 += region_len[j, r]
    if kind == PROBE_ENTROPY:
        row[0] = _entropy_of_union(
            tab, n, ws, idx, starts, lens, 1, mask_buf, comp_buf, scratch, union_buf
        )
    elif kind == PROBE_CODE_DIMENSION:
        row[0] = s_ref
    elif kind == PROBE_CONTIGUOUS_DISTANCE:
        if s_ref == 0:
            row[0] = np.nan  # no code, no distance (never averaged as 0)
        else:
            row[0] = contiguous_distance_rows(tab, ws, order, ref_rows, n_ref, periodic)
    elif kind == PROBE_LOST_INFORMATION:
        row[0] = lost_information_rows(
            tab, ws, idx[starts[0] : starts[0] + lens[0]], ref_rows, erase_scratch
        )
    elif kind == PROBE_MUTUAL_INFORMATION or kind == PROBE_TRIPARTITE:
        # sum over nonempty subsets T of the regions of (-1)^(|T|+1) S(T)
        num_regions = 2 if kind == PROBE_MUTUAL_INFORMATION else 3
        total = 0
        for which in range(1, 1 << num_regions):
            bits = 0
            for r in range(num_regions):
                bits += (which >> r) & 1
            s = _entropy_of_union(
                tab, n, ws, idx, starts, lens, which, mask_buf, comp_buf, scratch, union_buf
            )
            total += s if bits % 2 == 1 else -s
        row[0] = total
    elif kind == PROBE_ENTROPY_PROFILE:
        prefix_entropies(tab, n, ws, order, work)
        for i in range(n_sys + 1):
            row[i] = work[i]
    elif kind == PROBE_ERASURE_LOST_PROFILE:
        samples = sample_param[j]
        for i in range(n_sys + 1):
            row[i] = 0.0
        if s_ref > 0:
            for i in range(n_sys):  # every evaluation starts from the identity
                perm[i] = i
            for _ in range(samples):
                _shuffle(prng, perm)
                erasure_profile_rows(tab, ws, perm, ref_rows, work)
                for i in range(n_sys + 1):
                    row[i] += work[i]
            for i in range(n_sys + 1):
                row[i] /= samples
    elif kind == PROBE_FIRST_FAILURE:
        first_failure_samples(tab, ws, ref_rows, s_ref, prng, order, row)
    elif kind == PROBE_UNCORRECTABLE_COUNTS:
        if s_ref == 0:
            for m in range(size_param[j]):
                row[m] = 0.0
        else:  # the row is float64: the exact int64 counts arrive rounded once
            exact = np.zeros(size_param[j], dtype=np.int64)
            uncorrectable_counts_rows(tab, ws, order, ref_rows, size_param[j], row, exact)
    else:  # PROBE_CODE_SNAPSHOT: the generators go to snap_out, the row holds their number
        row[0] = code_snapshot_rows(tab, n_sys, n, ref_rows, snap_out)


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
    n_sys, n_ref, ref_mode, fixed_partner, edges, graph_ptr, circuits_per_graph, edge_cum,
    color_ptr, color_ptr_ptr, color_idx, schedule_kind, gates_per_layer, pool, p, layers,
    scramble_gates, scramble_matchings, scramble_layers, orient, gck, detect, stop_purified,
    pkind, plen, poff, pck_ptr, pck, pfill, ptr, region_len, idx, size_param, sample_param,
    order, periodic, key_struct, key_probe, phash, first, out, snap, snap_rows, out_tau,
    out_final, out_meas, h_index, cx_index, lin1, anf1, ident1, lin2, anf2, ident2,
):  # fmt: skip
    """Simulate ``out.shape[0]`` trajectories; see the module docstring.

    ``phash[j]`` (uint64) is the hash of probe ``j``'s name, which keys its
    sampling streams.

    ``snap`` (trajectories, evaluations, n_sys, 2 wsys) and ``snap_rows``
    receive the code snapshots of the (at most one) snapshot probe; rows of
    evaluations a trajectory never reaches are marked -1 in ``snap_rows``.
    """
    num_traj = out.shape[0]
    num_gck = gck.shape[0]
    num_probes = pkind.shape[0]
    n = n_sys + n_ref
    ws = (n + 63) // 64
    max_gates = _max_gates(n_sys, graph_ptr, gates_per_layer, scramble_gates)
    num_graphs = graph_ptr.shape[0] - 1
    for t in prange(num_traj):
        g = 0 if num_graphs == 1 else (first + t) // circuits_per_graph
        g_edges = edges[graph_ptr[g] : graph_ptr[g + 1]]
        g_cum = edge_cum[graph_ptr[g] : graph_ptr[g + 1]]  # empty when unweighted
        g_cidx = color_idx[graph_ptr[g] : graph_ptr[g + 1]]
        g_cptr = color_ptr[color_ptr_ptr[g] : color_ptr_ptr[g + 1]]
        rng = np.empty(4, dtype=np.uint64)
        prng = np.empty(4, dtype=np.uint64)  # seeded per sampled evaluation
        stream_state(key_struct, first + t, rng)
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
        next_eval = np.zeros(num_probes, dtype=np.int64)  # next evaluation index per probe

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
            if ci < num_gck and gck[ci] == layer:
                s_ref = -n_ref
                for w in range(ws):
                    s_ref += np.int64(popcount(ref_rows[w]))
                for j in range(num_probes):
                    c = next_eval[j]
                    if c < pck_ptr[j + 1] - pck_ptr[j] and pck[pck_ptr[j] + c] == layer:
                        base = poff[j] + c * plen[j]
                        if sample_param[j] > 0:  # every sampled kind has samples >= 1
                            stream_state(key_probe, mix(first + t, phash[j], layer), prng)
                        eval_probe(
                            j, pkind[j], out[t, base : base + plen[j]], tab, n, ws, n_sys,
                            ref_rows, n_ref, s_ref, ptr, region_len, idx, size_param,
                            sample_param, order, periodic, prng, mask_buf, comp_buf, scratch,
                            union_buf, perm, erase_scratch, work,
                            snap[t, min(c, snap.shape[1] - 1)],
                        )  # fmt: skip
                        if pkind[j] == PROBE_CODE_SNAPSHOT:
                            snap_rows[t, c] = np.int64(out[t, base])
                        next_eval[j] = c + 1
                if detect[ci] and ref_mode != REF_NONE and tau < 0 and s_ref == 0:
                    tau = layer
                    stopped = stop_purified
                ci += 1
            if stopped:
                break
        for j in range(num_probes):  # evaluations never reached: exact values for k = 0, or NaN
            for c in range(next_eval[j], pck_ptr[j + 1] - pck_ptr[j]):
                base = poff[j] + c * plen[j]
                for i in range(plen[j]):
                    out[t, base + i] = pfill[j]
                if pkind[j] == PROBE_CODE_SNAPSHOT:
                    snap_rows[t, c] = -1
        out_tau[t] = tau
        out_final[t] = final
        out_meas[t] = num_meas


@njit(cache=True)
def generate_ops(
    n_sys, ref_mode, fixed_partner, edges, graph_ptr, circuits_per_graph, edge_cum, color_ptr,
    color_ptr_ptr, color_idx, schedule_kind, gates_per_layer, pool, p, layers, scramble_gates,
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
    g = 0 if graph_ptr.shape[0] == 2 else trajectory // circuits_per_graph
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
