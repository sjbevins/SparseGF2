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
* :func:`ring_regions`: the exact sampler of ``Q_J`` on the ring ``C_n``.

Every random number comes from a xoshiro256** state (:mod:`.rng`) that the
caller passes in and that the kernel advances; nothing else is random.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from sparsegf2._kernel.rng import next_below, next_double


@njit(cache=True)
def csr(n, edges):
    """The graph in compressed sparse rows: ``(indptr int64[n+1], indices int64[2E])``.

    ``edges`` is an ``(E, 2)`` int64 array of distinct pairs without self-loops
    (:func:`sparsegf2.graphs.from_edges` gives one). The neighbours of ``v``
    are ``indices[indptr[v]:indptr[v+1]]``, sorted ascending; every edge
    appears in the rows of both its ends. The compiled twin of
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
    float64[c], finite and strictly increasing; ``samples >= 1``,
    ``burn_in >= 0``, ``thin >= 1``; ``tempering`` (bool); ``prng`` a uint64[4]
    xoshiro256** state, the only source of randomness, advanced in place.
    Outputs, filled in full: ``out_regions`` int64[c, samples, m] (row
    ``[i, s]``: the ``s``-th recorded region of the chain at ``couplings[i]``,
    its nodes sorted ascending), ``out_boundary`` int64[c, samples] (their
    ``b``), ``out_moves`` int64[c, 4] (row ``i``: swap proposals and accepted
    swaps at ``couplings[i]``, then exchange attempts and accepted exchanges of
    the pair ``(i, i+1)``; zero for ``i = c-1`` and without tempering).

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
    exact sample of ``Q_0``). After ``burn_in`` sweeps the state of every chain
    is recorded, then again after every further ``thin`` sweeps, ``samples``
    records in all (``burn_in + (samples - 1) * thin`` sweeps).

    **Draw order (the contract).** All draws come from ``prng``
    (:func:`~sparsegf2._kernel.rng.next_below`, :func:`~.rng.next_double`):

    1. Start states, chain by chain in coupling order: ``m`` draws
       ``next_below(n - t)``, ``t = 0..m-1``, a partial Fisher-Yates shuffle of
       ``0..n-1`` (swap entries ``t`` and ``t + draw``). The first ``m``
       entries, in shuffled order, are the erased list and the remaining ``n - m``
       the unerased list.
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
        for i in range(c):
            for s in range(samples):
                out_boundary[i, s] = 0
                for k in range(m):
                    out_regions[i, s, k] = k
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
            s = slot[i]
            k = 0
            for v in range(n):
                if member[s, v]:
                    out_regions[i, r, k] = v
                    k += 1
            out_boundary[i, r] = bnd[s]


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
def ring_regions(n, m, cdf, samples, prng, out):
    """Exact samples of ``Q_J`` on the ring ``C_n``: ``out[s]`` = the erased nodes, sorted.

    ``cdf[r-1]`` is the probability of at most ``r`` runs of erased nodes,
    ``r = 1..len(cdf)`` (the last entry 1), from
    :func:`sparsegf2.erasures.ring_boundary_distribution`. Per sample: one
    ``next_double`` picks ``r`` (the first ``r`` with the draw below
    ``cdf[r-1]``); ``r - 1`` draws give a uniform composition of ``m`` into the
    ``r`` run lengths and ``r - 1`` more one of ``n - m`` into the ``r`` gap
    lengths (:func:`_composition`); ``next_below(n)`` gives the start of the
    first run. Runs and gaps then alternate around the ring from there. Every
    region with ``r`` runs comes from exactly ``r`` such triples (one per
    choice of which run is the first), so the regions with ``r`` runs are
    uniform. ``m = 0`` or ``m = n`` draws nothing.
    """
    if m == 0 or m == n:
        for s in range(samples):
            for k in range(m):
                out[s, k] = k
        return
    rmax = cdf.shape[0]
    member = np.zeros(n, dtype=np.uint8)
    gaps = np.empty(n, dtype=np.int64)
    run_cuts = np.empty(rmax + 1, dtype=np.int64)
    gap_cuts = np.empty(rmax + 1, dtype=np.int64)
    for s in range(samples):
        x = next_double(prng)
        r = 1
        while r < rmax and not x < cdf[r - 1]:
            r += 1
        _composition(m, r, prng, gaps, run_cuts)
        _composition(n - m, r, prng, gaps, gap_cuts)
        position = next_below(prng, n)
        for v in range(n):
            member[v] = 0
        for j in range(r):
            for _ in range(run_cuts[j + 1] - run_cuts[j]):
                member[position % n] = 1
                position += 1
            position += gap_cuts[j + 1] - gap_cuts[j]
        k = 0
        for v in range(n):
            if member[v]:
                out[s, k] = v
                k += 1
