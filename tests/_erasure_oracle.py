"""Plain-Python transcriptions of the erasure kernels' draw orders (test-only).

Written from the docstrings of ``sparsegf2._kernel.erasure_chain``: ``sample_chain`` (the
swap chain with replica exchange and a record mask) and the exact ring and path samplers
(``ring_exact_sample``, ``path_exact_sample``). They draw from ``_reference_rng`` (a
pure-Python xoshiro256**) and never import the kernels, so a test that requires the kernels,
or a probe built on them, to give the same regions from the same state checks the documented
draw order itself. Shared by ``tests/test_erasures.py`` (the kernels) and
``tests/test_correlated_probe.py`` (the in-run probe's protocol oracle).
"""

from __future__ import annotations

import itertools
import math

import numpy as np

import _reference_rng as ref


def _brute_boundary(edges, region) -> int:
    inside = {int(q) for q in region}
    return sum((int(a) in inside) != (int(b) in inside) for a, b in edges)


def _reference_chain(edges, n, m, couplings, record, samples, burn_in, thin, tempering, state):
    """The chain transcribed from the draw order in ``sample_chain``'s docstring.

    Rung ``i`` is recorded on row ``record[i]`` of the outputs, or not at all for -1.
    """
    gen = ref.Xoshiro([int(w) for w in state])
    c = len(couplings)
    chains = []
    for _ in range(c):
        order = list(range(n))
        for t in range(m):
            j = t + gen.next_below(n - t)
            order[t], order[j] = order[j], order[t]
        chains.append([order[:m], order[m:], _brute_boundary(edges, order[:m])])
    slot = list(range(c))
    moves = np.zeros((c, 4), dtype=np.int64)
    rows = sum(1 for k in record if k >= 0)
    regions = np.zeros((rows, samples, m), dtype=np.int64)
    bounds = np.zeros((rows, samples), dtype=np.int64)
    done = 0
    for r in range(samples):
        while done < burn_in + r * thin:
            for i, coupling in enumerate(couplings):
                chain = chains[slot[i]]
                inside, outside = chain[0], chain[1]
                for _ in range(n):
                    a = gen.next_below(m)
                    p = gen.next_below(n - m)
                    u, v = inside[a], outside[p]
                    d = _brute_boundary(edges, [*inside[:a], v, *inside[a + 1 :]]) - chain[2]
                    moves[i, 0] += 1
                    if coupling * d > 0 and not gen.next_double() < math.exp(-abs(coupling * d)):
                        continue
                    inside[a], outside[p] = v, u
                    chain[2] += d
                    moves[i, 1] += 1
            if tempering and c > 1:
                for i in range(c - 1):
                    x = (couplings[i] - couplings[i + 1]) * (
                        chains[slot[i]][2] - chains[slot[i + 1]][2]
                    )
                    moves[i, 2] += 1
                    if x >= 0 or gen.next_double() < math.exp(x):
                        slot[i], slot[i + 1] = slot[i + 1], slot[i]
                        moves[i, 3] += 1
            done += 1
        for i in range(c):
            if record[i] >= 0:
                regions[record[i], r] = sorted(chains[slot[i]][0])
                bounds[record[i], r] = chains[slot[i]][2]
    return regions, bounds, moves, gen.s


def _reference_composition(gen, total, parts):
    gaps = list(range(1, total))
    for t in range(parts - 1):
        j = t + gen.next_below(total - 1 - t)
        gaps[t], gaps[j] = gaps[j], gaps[t]
    cuts = [0, *sorted(gaps[: parts - 1]), total]
    return [cuts[j + 1] - cuts[j] for j in range(parts)]


def _log_comb(a, k):
    return math.lgamma(a + 1) - math.lgamma(k + 1) - math.lgamma(a - k + 1)


def _reference_log_weights(kind, n, m, coupling):
    """Categories (r, e) and their log weights, as the exact kernels list them."""
    if kind == "cycle":
        cats = [(r, 0) for r in range(1, min(m, n - m) + 1)]
        logw = [
            math.log(n / r) + _log_comb(m - 1, r - 1) + _log_comb(n - m - 1, r - 1)
            - 2 * coupling * r
            for r, _ in cats
        ]  # fmt: skip
        return cats, logw
    cats, logw = [], []
    for r in range(1, min(m, n - m + 1) + 1):
        for e, (k, extra) in enumerate(((r, 0.0), (r - 1, math.log(2)), (r - 2, 0.0))):
            cats.append((r, e))
            if 0 <= k <= n - m - 1:
                runs = _log_comb(m - 1, r - 1)
                logw.append(runs + _log_comb(n - m - 1, k) + extra - coupling * (2 * r - e))
            else:
                logw.append(-math.inf)
    return cats, logw


def _reference_exact(kind, n, m, couplings, samples, order, state):
    """The exact samplers transcribed from the draw order in their docstrings."""
    gen = ref.Xoshiro([int(w) for w in state])
    regions = np.zeros((len(couplings), samples, m), dtype=np.int64)
    bounds = np.zeros((len(couplings), samples), dtype=np.int64)
    for i, coupling in enumerate(couplings):
        cats, logw = _reference_log_weights(kind, n, m, coupling)
        top = max(logw)
        cdf = list(itertools.accumulate(math.exp(x - top) for x in logw))
        cdf = [c / cdf[-1] for c in cdf]
        cdf[-1] = 1.0
        for s in range(samples):
            x = gen.next_double()
            k = 0
            while k < len(cats) - 1 and not x < cdf[k]:
                k += 1
            r, e = cats[k]
            if kind == "cycle":
                runs = _reference_composition(gen, m, r)
                gaps = _reference_composition(gen, n - m, r)
                place = gen.next_below(n)
            else:
                side = gen.next_below(2) if e == 1 else 0
                runs = _reference_composition(gen, m, r)
                if e == 0:
                    gaps = _reference_composition(gen, n - m, r + 1)
                elif e == 1:
                    inner = _reference_composition(gen, n - m, r)
                    gaps = [0, *inner] if side == 0 else [*inner, 0]
                else:
                    gaps = [0, *_reference_composition(gen, n - m, r - 1), 0]
                place, gaps = gaps[0], gaps[1:]
            chosen = []
            for length, gap in zip(runs, gaps, strict=True):
                chosen += [int(order[(place + t) % n]) for t in range(length)]
                place += length + gap
            regions[i, s] = sorted(chosen)
            bounds[i, s] = 2 * r - e
    return regions, bounds, gen.s
