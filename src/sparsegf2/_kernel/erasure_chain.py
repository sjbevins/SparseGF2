"""Compiled kernels for geometry-correlated erasure regions (private).

The graph side of :mod:`sparsegf2.erasures`. Notation: an undirected graph
``G = (V, E)`` with ``V = {0, ..., n-1}``, no self-loops and no repeated edges;
a region ``A`` of ``V`` with ``m = |A|`` and complement ``Ā = V \\ A``; its edge
boundary ``b(A) = #{{u, v} in E : u in A, v not in A}``. The target law at a
real coupling ``J`` is ``Q_J(A) = exp(-J b(A)) / Z_m(J)`` on the ``C(n, m)``
regions of size ``m``.

* :func:`csr`: the graph in compressed sparse rows, neighbours sorted.
* :func:`region_boundary` and :func:`swap_delta`: ``b(A)`` and the change of
  ``b`` when one erased node is swapped for one unerased node.
* :func:`sample_chain`: the Metropolis swap chain (Kawasaki dynamics) at fixed
  ``m``, one chain per coupling, with replica exchange between neighbouring
  couplings. Its draw order (the docstring) is a contract: a probe's stream
  in the ensemble engine is made of it.
* :func:`enumerate_boundaries`: every ``m``-subset in lexicographic order with
  its boundary (the exact density of states ``N_m(b)``, and the subsets of a
  given boundary).
* :func:`geometry_order`: whether the graph is one cycle or one path through
  all nodes (under any labeling), and the nodes in that order.
* :func:`ring_exact_sample` and :func:`path_exact_sample`: independent exact
  samples of ``Q_J`` on such a cycle or path (no Markov chain). Their draw
  orders (the docstrings) are contracts too.

Every random number comes from a xoshiro256** state (:mod:`.rng`) that the
caller passes in and that the kernel advances; nothing else is random.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from sparsegf2._kernel.rng import next_below, next_double


@njit(cache=True)
def csr(n, edges):
    """The graph in compressed sparse rows: ``(indptr int64[n+1], indices int64[2E])``.

    ``edges`` is an ``(E, 2)`` int64 array of distinct pairs without self-loops.
    Distinct matters: a repeated edge would appear twice in both rows, and
    every boundary, ``Δb`` (its ``u ~ v`` flag) and count here would treat it as
    two edges. So build it only from :func:`sparsegf2.graphs.from_edges` output,
    which removes repeats (as :mod:`sparsegf2.erasures` does). The neighbours
    of ``v`` are ``indices[indptr[v]:indptr[v+1]]``, sorted ascending; every
    edge appears in the rows of both its ends. The compiled twin of
    ``graphs._csr``, which does not sort the rows.
    """
    num_edges = edges.shape[0]
    indptr = np.zeros(n + 1, dtype=np.int64)
    for k in range(num_edges):
        indptr[edges[k, 0] + 1] += 1
        indptr[edges[k, 1] + 1] += 1
    for v in range(n):
        indptr[v + 1] += indptr[v]
    fill = indptr[:n].copy()
    indices = np.empty(2 * num_edges, dtype=np.int64)
    for k in range(num_edges):
        a = edges[k, 0]
        b = edges[k, 1]
        indices[fill[a]] = b
        fill[a] += 1
        indices[fill[b]] = a
        fill[b] += 1
    for v in range(n):
        indices[indptr[v] : indptr[v + 1]].sort()
    return indptr, indices


@njit(cache=True)
def region_boundary(indptr, indices, member):
    """``b(A)``: the number of edges with exactly one end in ``A``.

    ``member`` is a uint8 array of length ``n``, 1 for the nodes of ``A``. Each
    boundary edge is counted once, from its end in ``A``. Cost ``O(n + E)``.
    """
    n = member.shape[0]
    b = 0
    for u in range(n):
        if member[u]:
            for j in range(indptr[u], indptr[u + 1]):
                if not member[indices[j]]:
                    b += 1
    return b


@njit(cache=True)
def swap_delta(indptr, indices, member, u, v):
    """``Δb = b(A') - b(A)`` for ``A' = A \\ {u} ∪ {v}``, ``u`` in ``A``, ``v`` not in ``A``.

    With ``deg_A(x) = |N(x) ∩ A|`` and ``deg_Ā(x) = |N(x) \\ A|`` counted in
    the current ``A`` (``member``), and ``1[u ~ v]`` = 1 when ``{u, v}`` is an
    edge::

        Δb = (deg_A(u) - deg_Ā(u)) + (deg_Ā(v) - deg_A(v)) + 2 * 1[u ~ v]

    Derivation: only edges at ``u`` or ``v`` can change status. An edge
    ``{u, w}`` with ``w != v`` is a boundary edge before the swap exactly when
    ``w`` is not in ``A`` and after it exactly when ``w`` is in ``A`` (``u``
    has left, ``w`` stays put): these edges change ``b`` by
    ``deg_A(u) - (deg_Ā(u) - 1[u ~ v])``. An edge ``{v, w}`` with ``w != u`` is
    a boundary edge before exactly when ``w`` is in ``A`` and after exactly when
    it is not (``v`` has entered): ``deg_Ā(v) - (deg_A(v) - 1[u ~ v])``. The
    edge ``{u, v}`` itself, when present, is a boundary edge before (``u`` in,
    ``v`` out) and after (``v`` in, ``u`` out), so it changes nothing; the two
    ``1[u ~ v]`` corrections remove it from ``deg_Ā(u)`` and ``deg_A(v)``,
    which count it. Summing gives the formula. Cost ``O(deg u + deg v)``.
    """
    du = 0  # deg_A(u) - deg_Ā(u)
    adjacent = 0
    for j in range(indptr[u], indptr[u + 1]):
        w = indices[j]
        if member[w]:
            du += 1
        else:
            du -= 1
            if w == v:
                adjacent = 1
    dv = 0  # deg_Ā(v) - deg_A(v)
    for j in range(indptr[v], indptr[v + 1]):
        if member[indices[j]]:
            dv -= 1
        else:
            dv += 1
    return du + dv + 2 * adjacent


@njit(cache=True)
def sample_chain(
    indptr,
    indices,
    n,
    m,
    couplings,
    record,
    samples,
    burn_in,
    thin,
    tempering,
    prng,
    out_regions,
    out_boundary,
    out_moves,
):
    """Sample regions of size ``m`` from ``Q_J`` at each coupling: one Metropolis swap chain each.

    Arguments: the graph in CSR form (:func:`csr`) on ``n`` nodes; ``couplings``
    float64[c], the ladder of rungs, finite and strictly increasing;
    ``record`` int64[c], the output row of each rung, or -1 for an auxiliary
    rung that runs (and exchanges) but is never recorded; ``samples >= 1``,
    ``burn_in >= 0``, ``thin >= 1``; ``tempering`` (bool); ``prng`` a uint64[4]
    xoshiro256** state, the only source of randomness, advanced in place.
    Outputs, filled in full: ``out_regions`` int64[k, samples, m], ``k`` the
    number of recorded rungs (row ``[record[i], s]``: the ``s``-th recorded
    region of the chain at ``couplings[i]``, its nodes sorted ascending),
    ``out_boundary`` int64[k, samples] (their ``b``), ``out_moves`` int64[c, 4]
    (row ``i``: swap proposals and accepted swaps at ``couplings[i]``, then
    exchange attempts and accepted exchanges of the pair ``(i, i+1)``; zero for
    ``i = c-1`` and without tempering). Recording only some rungs changes
    nothing else: the draws, the moves and every recorded row are those of a
    run that records every rung.

    **The chain.** A state is a region of size ``m``, kept as a list of its
    nodes ("erased") and a list of the others ("unerased"), plus a 0/1 mask and
    its boundary. A proposal draws a position ``a`` in the erased list and a
    position ``p`` in the unerased list, uniformly, so ``u`` (erased) and ``v``
    (unerased) are uniform and the proposal ``A -> A \\ {u} ∪ {v}`` has the
    symmetric probability ``1 / (m (n - m))``. It is accepted with probability
    ``min(1, exp(-J Δb))`` (Metropolis; :func:`swap_delta` gives ``Δb``), which
    makes ``Q_J`` stationary (detailed balance); the swaps connect all
    ``m``-subsets (the Johnson graph), whatever ``G`` is. On acceptance ``v``
    takes ``u``'s place in the erased list and ``u`` takes ``v``'s place in the
    unerased list. A sweep is ``n`` proposals.

    **Replica exchange** (``tempering`` and ``c >= 2``): after each sweep, for
    ``i = 0, ..., c-2`` in that order, the states of the chains at
    ``J_i`` and ``J_{i+1}`` are exchanged with probability
    ``min(1, exp((J_i - J_{i+1}) (b_i - b_{i+1})))``, the Metropolis ratio for
    the product law ``Q_{J_0} x ... x Q_{J_{c-1}}``, which therefore stays
    invariant: each chain's marginal is exactly its ``Q_{J_i}``. The exchange
    swaps which state buffer each coupling points to; nothing is copied.

    **Recording.** Each chain starts from a uniformly random ``m``-subset (an
    exact sample of ``Q_0``). After ``burn_in`` sweeps the state of every
    recorded rung is recorded, then again after every further ``thin`` sweeps,
    ``samples`` records in all (``burn_in + (samples - 1) * thin`` sweeps).

    **Draw order (the contract).** All draws come from ``prng``
    (:func:`~sparsegf2._kernel.rng.next_below`, :func:`~.rng.next_double`):

    1. Start states, chain by chain in coupling order (every rung, recorded or
       not): ``m`` draws ``next_below(n - t)``, ``t = 0..m-1``, a partial
       Fisher-Yates shuffle of ``0..n-1`` (swap entries ``t`` and
       ``t + draw``). The first ``m`` entries, in shuffled order, are the erased
       list and the remaining ``n - m`` the unerased list.
    2. Sweeps, one after another. In a sweep, chain by chain in coupling order,
       ``n`` proposals, each ``a = next_below(m)``, then ``p = next_below(n - m)``,
       then, only when ``J Δb > 0`` (the acceptance probability
       ``exp(-|J Δb|)`` is below 1), one ``next_double``: accept when it is
       below ``exp(-|J Δb|)``. Then, with replica exchange, for each pair
       ``(i, i+1)`` in order, one ``next_double`` only when
       ``x = (J_i - J_{i+1}) (b_i - b_{i+1}) < 0``: exchange when it is below
       ``exp(x)``.
    3. Recording draws nothing.

    With ``m = 0`` or ``m = n`` there is a single region (boundary 0): it is
    recorded ``samples`` times, nothing is drawn, and every move count is 0.
    """
    c = couplings.shape[0]
    for i in range(c):
        for k in range(4):
            out_moves[i, k] = 0
    if m == 0 or m == n:
        _single_region(m, out_regions, out_boundary)
        return
    unerased = n - m
    member = np.zeros((c, n), dtype=np.uint8)
    inside = np.empty((c, m), dtype=np.int64)
    outside = np.empty((c, unerased), dtype=np.int64)
    bnd = np.empty(c, dtype=np.int64)
    slot = np.empty(c, dtype=np.int64)  # slot[i]: the state buffer of the chain at couplings[i]
    order = np.empty(n, dtype=np.int64)
    for i in range(c):
        slot[i] = i
        for v in range(n):
            order[v] = v
        for t in range(m):
            j = t + next_below(prng, n - t)
            tmp = order[t]
            order[t] = order[j]
            order[j] = tmp
        for t in range(m):
            inside[i, t] = order[t]
            member[i, order[t]] = 1
        for t in range(unerased):
            outside[i, t] = order[m + t]
        bnd[i] = region_boundary(indptr, indices, member[i])

    # exp(-|J| d) for d = 0..2 maxdeg + 2 (|Δb| <= 2 maxdeg), one row per coupling.
    maxdeg = 0
    for v in range(n):
        maxdeg = max(maxdeg, indptr[v + 1] - indptr[v])
    top = 2 * maxdeg + 2
    table = np.empty((c, top + 1), dtype=np.float64)
    for i in range(c):
        strength = abs(couplings[i])
        for d in range(top + 1):
            table[i, d] = np.exp(-strength * d)
    exchange = tempering and c >= 2

    done = 0
    for r in range(samples):
        target = burn_in + r * thin
        while done < target:
            for i in range(c):
                s = slot[i]
                coupling = couplings[i]
                state = member[s]
                accepted = 0
                for _ in range(n):
                    a = next_below(prng, m)
                    p = next_below(prng, unerased)
                    u = inside[s, a]
                    v = outside[s, p]
                    d = swap_delta(indptr, indices, state, u, v)
                    if coupling * d > 0.0 and not next_double(prng) < table[i, abs(d)]:
                        continue
                    inside[s, a] = v
                    outside[s, p] = u
                    state[u] = 0
                    state[v] = 1
                    bnd[s] += d
                    accepted += 1
                out_moves[i, 0] += n
                out_moves[i, 1] += accepted
            if exchange:
                for i in range(c - 1):
                    s0 = slot[i]
                    s1 = slot[i + 1]
                    x = (couplings[i] - couplings[i + 1]) * (bnd[s0] - bnd[s1])
                    out_moves[i, 2] += 1
                    if x >= 0.0 or next_double(prng) < np.exp(x):
                        slot[i] = s1
                        slot[i + 1] = s0
                        out_moves[i, 3] += 1
            done += 1
        for i in range(c):
            row = record[i]
            if row < 0:
                continue  # an auxiliary rung of the ladder
            s = slot[i]
            k = 0
            for v in range(n):
                if member[s, v]:
                    out_regions[row, r, k] = v
                    k += 1
            out_boundary[row, r] = bnd[s]


@njit(cache=True)
def enumerate_boundaries(indptr, indices, n, m, hist, target, regions):
    """Visit every ``m``-subset of ``0..n-1`` in lexicographic order, with its boundary.

    ``hist[b] += 1`` for every subset of boundary ``b`` (``hist`` needs length
    at least the largest boundary plus one; ``E + 1`` always suffices), so a
    zeroed ``hist`` comes back as the density of states ``N_m(b)``. The subsets
    whose boundary equals ``target`` are written, sorted, to the rows of
    ``regions`` in order, as many as fit; the return value is how many there
    are (0 for ``target < 0``). Boundaries are built incrementally:
    ``b(S ∪ {v}) = b(S) + deg(v) - 2 |N(v) ∩ S|``, from the first position the
    lexicographic successor changes. Draws nothing.
    """
    combo = np.empty(m, dtype=np.int64)
    partial = np.zeros(m + 1, dtype=np.int64)  # partial[k] = b(combo[:k])
    member = np.zeros(n, dtype=np.uint8)
    for k in range(m):
        combo[k] = k
    start = 0
    found = 0
    while True:
        for k in range(start, m):
            v = combo[k]
            inner = 0
            for j in range(indptr[v], indptr[v + 1]):
                inner += member[indices[j]]
            partial[k + 1] = partial[k] + (indptr[v + 1] - indptr[v]) - 2 * inner
            member[v] = 1
        b = partial[m]
        hist[b] += 1
        if b == target:
            if found < regions.shape[0]:
                for k in range(m):
                    regions[found, k] = combo[k]
            found += 1
        i = m - 1
        while i >= 0 and combo[i] == n - m + i:
            i -= 1
        if i < 0:
            return found
        for k in range(i, m):
            member[combo[k]] = 0
        combo[i] += 1
        for k in range(i + 1, m):
            combo[k] = combo[k - 1] + 1
        start = i


@njit(cache=True)
def _composition(total, parts, prng, gaps, cuts):
    """A uniform composition of ``total`` into ``parts`` positive parts (stars and bars).

    The ``parts - 1`` cut points are a uniform subset of the ``total - 1`` gaps
    ``1..total-1`` (a partial Fisher-Yates shuffle, ``parts - 1`` draws
    ``next_below(total - 1 - t)``). Part ``j`` is ``cuts[j+1] - cuts[j]``,
    with ``cuts[0] = 0`` and ``cuts[parts] = total``.
    """
    free = total - 1
    for t in range(free):
        gaps[t] = t + 1
    for t in range(parts - 1):
        j = t + next_below(prng, free - t)
        tmp = gaps[t]
        gaps[t] = gaps[j]
        gaps[j] = tmp
    cuts[0] = 0
    for t in range(parts - 1):
        cuts[t + 1] = gaps[t]
    cuts[1:parts].sort()
    cuts[parts] = total


@njit(cache=True)
def geometry_order(indptr, indices, n):
    """Is the graph one cycle or one path through all ``n`` nodes? ``(order int64[n], kind)``.

    ``kind`` is 1 when every node has degree 2 and the graph is connected (a
    cycle through all nodes, ``n >= 3``), 2 when exactly two nodes have degree
    1, the others degree 2, and the graph is connected (a Hamiltonian path; a
    single node, ``n = 1``, counts as a path), else 0. ``order`` lists the
    nodes along the cycle (from node 0 towards its smaller neighbour) or along
    the path (from its smaller end); for kind 0 it is ``0..n-1``. The labels
    may be anything: the order is read from the edges. Cost ``O(n + E)``.
    """
    order = np.empty(n, dtype=np.int64)
    for v in range(n):
        order[v] = v
    if n == 1:
        return order, 2
    ends = 0
    start = 0
    for v in range(n - 1, -1, -1):
        degree = indptr[v + 1] - indptr[v]
        if degree == 1:
            ends += 1
            start = v
        elif degree != 2:
            return order, 0
    if ends != 0 and ends != 2:
        return order, 0
    kind = 1 if ends == 0 else 2
    seen = np.zeros(n, dtype=np.uint8)
    walk = np.empty(n, dtype=np.int64)
    walk[0] = start
    seen[start] = 1
    previous = -1
    current = start
    for k in range(1, n):
        following = -1
        for j in range(indptr[current], indptr[current + 1]):
            if indices[j] != previous:
                following = indices[j]
                break
        if following < 0 or seen[following]:
            return order, 0  # the walk ended or closed before n nodes: several components
        seen[following] = 1
        walk[k] = following
        previous = current
        current = following
    return walk, kind


@njit(cache=True)
def _log_comb(a, k):
    """``log C(a, k)`` for ``0 <= k <= a``, by ``lgamma``."""
    return math.lgamma(a + 1.0) - math.lgamma(k + 1.0) - math.lgamma(a - k + 1.0)


@njit(cache=True)
def _cumulative(logw, cdf):
    """``cdf[k] = sum_{j <= k} w_j / sum_j w_j`` with ``w = exp(logw - max logw)``; last entry 1.

    Entries of ``logw`` equal to ``-inf`` weigh 0.
    """
    top = -np.inf
    for k in range(logw.shape[0]):
        top = max(top, logw[k])
    total = 0.0
    for k in range(logw.shape[0]):
        if logw[k] > -np.inf:
            total += np.exp(logw[k] - top)
        cdf[k] = total
    for k in range(logw.shape[0]):
        cdf[k] /= total
    cdf[logw.shape[0] - 1] = 1.0


@njit(cache=True)
def _pick(prng, cdf):
    """One ``next_double`` ``x``; the first ``k`` with ``x < cdf[k]`` (the last if none)."""
    x = next_double(prng)
    k = 0
    while k < cdf.shape[0] - 1 and not x < cdf[k]:
        k += 1
    return k


@njit(cache=True)
def _single_region(m, out_regions, out_boundary):
    """``m = 0`` or ``m = n``: every record is ``0..m-1``, boundary 0."""
    for i in range(out_regions.shape[0]):
        for s in range(out_regions.shape[1]):
            out_boundary[i, s] = 0
            for k in range(m):
                out_regions[i, s, k] = k


@njit(cache=True)
def _write_sorted(member, n, out_regions, i, s):
    k = 0
    for v in range(n):
        if member[v]:
            out_regions[i, s, k] = v
            k += 1


@njit(cache=True)
def ring_exact_sample(n, m, couplings, samples, prng, order, out_regions, out_boundary):
    """Independent exact samples of ``Q_J`` on a cycle through all ``n`` nodes, per coupling.

    ``order`` (int64[n]) lists the nodes in cyclic order (:func:`geometry_order`),
    so any labeling works; ``couplings`` float64[c], any finite values, any order.
    Fills ``out_regions[i, s]`` (the erased nodes, sorted) and
    ``out_boundary[i, s] = 2 r`` for coupling ``i`` and sample ``s``.

    On the ring a region with ``r`` runs (maximal blocks of consecutive erased
    places) has ``b = 2 r``, and there are ``N(n, m, r) = (n / r) C(m-1, r-1)
    C(n-m-1, r-1)`` of them, ``r = 1..min(m, n-m)``. So ``r`` is drawn with
    probability ``N(n, m, r) exp(-2 J r) / Z`` (weights in log space,
    ``lgamma``), then the ``r`` run lengths (a uniform composition of ``m``
    into ``r`` positive parts: a uniform ``(r-1)``-subset of the ``m - 1``
    places between consecutive units) and the ``r`` gap lengths (one of
    ``n - m``), and the place of the first run, uniform among the ``n``;
    runs and gaps then alternate round the cycle. Each region with ``r`` runs
    comes from exactly ``r`` of these equally likely triples (one per choice
    of which of its runs is laid first), so given ``r`` it is uniform.

    **Draw order (the contract).** Coupling by coupling in the order given; for
    each sample: one ``next_double`` picks ``r`` (the first ``r`` whose
    cumulative probability exceeds it); ``r - 1`` draws for the run lengths
    and ``r - 1`` for the gap lengths (:func:`_composition`);
    ``next_below(n)`` for the place of the first run, which then covers the
    places ``p, p+1, ...`` (mod ``n``) of ``order``. ``m = 0`` or ``m = n``
    draws nothing.
    """
    if m == 0 or m == n:
        _single_region(m, out_regions, out_boundary)
        return
    rmax = min(m, n - m)
    logw = np.empty(rmax, dtype=np.float64)
    cdf = np.empty(rmax, dtype=np.float64)
    member = np.zeros(n, dtype=np.uint8)
    gaps = np.empty(n, dtype=np.int64)
    run_cuts = np.empty(rmax + 1, dtype=np.int64)
    gap_cuts = np.empty(rmax + 1, dtype=np.int64)
    for i in range(couplings.shape[0]):
        coupling = couplings[i]
        for r in range(1, rmax + 1):
            logw[r - 1] = (
                math.log(n / r)
                + _log_comb(m - 1, r - 1)
                + _log_comb(n - m - 1, r - 1)
                - 2 * coupling * r
            )
        _cumulative(logw, cdf)
        for s in range(samples):
            r = _pick(prng, cdf) + 1
            _composition(m, r, prng, gaps, run_cuts)
            _composition(n - m, r, prng, gaps, gap_cuts)
            place = next_below(prng, n)
            for v in range(n):
                member[v] = 0
            for j in range(r):
                for _ in range(run_cuts[j + 1] - run_cuts[j]):
                    member[order[place % n]] = 1
                    place += 1
                place += gap_cuts[j + 1] - gap_cuts[j]
            _write_sorted(member, n, out_regions, i, s)
            out_boundary[i, s] = 2 * r


@njit(cache=True)
def path_exact_sample(n, m, couplings, samples, prng, order, out_regions, out_boundary):
    """Independent exact samples of ``Q_J`` on a path through all ``n`` nodes, per coupling.

    ``order`` (int64[n]) lists the nodes from one end of the path to the other
    (:func:`geometry_order`), so any labeling works. Fills ``out_regions[i, s]``
    (sorted) and ``out_boundary[i, s]`` as :func:`ring_exact_sample`.

    A region with ``r`` runs of which ``e`` (0, 1 or 2) touch an end of the path
    has ``b = 2 r - e``. Its ``n - m`` unerased places form ``r + 1`` gaps: the
    ``r - 1`` between runs are positive, the two end gaps positive or zero as
    ``e`` says. So the number of such regions is ``C(m-1, r-1) C(n-m-1, r)``
    (``e = 0``: ``n - m`` into ``r + 1`` positive parts),
    ``2 C(m-1, r-1) C(n-m-1, r-1)`` (``e = 1``: either end, ``n - m`` into
    ``r`` parts) and ``C(m-1, r-1) C(n-m-1, r-2)`` (``e = 2``, ``r >= 2``:
    ``n - m`` into the ``r - 1`` inner gaps). ``(r, e)`` is drawn with
    probability proportional to that count times ``exp(-J (2 r - e))``, then
    the end (``e = 1``), the run lengths and the gap lengths uniformly: one
    region per choice, so the region is uniform given ``(r, e)``.

    **Draw order (the contract).** Coupling by coupling in the order given; the
    categories ``(r, e)`` are listed ``r = 1..min(m, n-m+1)``, ``e = 0, 1, 2``
    within each ``r``. For each sample: one ``next_double`` picks ``(r, e)``
    (the first category whose cumulative probability exceeds it); for
    ``e = 1`` one ``next_below(2)`` (0: the run touches ``order[0]``, 1:
    ``order[n-1]``); ``r - 1`` draws for the run lengths (:func:`_composition`
    of ``m`` into ``r``); then the gaps: ``e = 0``, ``n - m`` into ``r + 1``
    parts (``r`` draws); ``e = 1``, into the ``r`` gaps other than the touched
    end's (``r - 1`` draws); ``e = 2``, into the ``r - 1`` inner gaps
    (``r - 2`` draws). Gap 0 comes first, then run 1, gap 1, ..., run ``r``,
    gap ``r``, along ``order``. ``m = 0`` or ``m = n`` draws nothing.
    """
    if m == 0 or m == n:
        _single_region(m, out_regions, out_boundary)
        return
    rmax = min(m, n - m + 1)
    logw = np.empty(3 * rmax, dtype=np.float64)
    cdf = np.empty(3 * rmax, dtype=np.float64)
    member = np.zeros(n, dtype=np.uint8)
    gaps = np.empty(n, dtype=np.int64)
    run_cuts = np.empty(rmax + 1, dtype=np.int64)
    gap_cuts = np.empty(rmax + 2, dtype=np.int64)
    gap = np.zeros(
        rmax + 1, dtype=np.int64
    )  # gap[j]: unerased places before run j+1 (j = r: after run r)
    for i in range(couplings.shape[0]):
        coupling = couplings[i]
        for r in range(1, rmax + 1):
            for e in range(3):
                k = r - e  # C(n-m-1, r), C(n-m-1, r-1), C(n-m-1, r-2)
                if 0 <= k <= n - m - 1:
                    extra = math.log(2.0) if e == 1 else 0.0
                    logw[3 * (r - 1) + e] = (
                        _log_comb(m - 1, r - 1)
                        + _log_comb(n - m - 1, k)
                        + extra
                        - coupling * (2 * r - e)
                    )
                else:
                    logw[3 * (r - 1) + e] = -np.inf
        _cumulative(logw, cdf)
        for s in range(samples):
            category = _pick(prng, cdf)
            r = category // 3 + 1
            e = category % 3
            side = next_below(prng, 2) if e == 1 else 0
            _composition(m, r, prng, gaps, run_cuts)
            if e == 0:
                _composition(n - m, r + 1, prng, gaps, gap_cuts)
                for j in range(r + 1):
                    gap[j] = gap_cuts[j + 1] - gap_cuts[j]
            elif e == 1:
                _composition(n - m, r, prng, gaps, gap_cuts)
                shift = 1 - side  # side 0: gap 0 is empty; side 1: gap r is empty
                gap[0] = 0
                gap[r] = 0
                for j in range(r):
                    gap[j + shift] = gap_cuts[j + 1] - gap_cuts[j]
            else:
                _composition(n - m, r - 1, prng, gaps, gap_cuts)
                gap[0] = 0
                gap[r] = 0
                for j in range(r - 1):
                    gap[j + 1] = gap_cuts[j + 1] - gap_cuts[j]
            for v in range(n):
                member[v] = 0
            place = gap[0]
            for j in range(r):
                for _ in range(run_cuts[j + 1] - run_cuts[j]):
                    member[order[place]] = 1
                    place += 1
                place += gap[j + 1]
            _write_sorted(member, n, out_regions, i, s)
            out_boundary[i, s] = 2 * r - e
