"""Independent references for the Watts–Strogatz study (v1.4 §2.2, §3, §4, §7.3).

Nothing here imports ``sparsegf2._kernel`` or ``sparsegf2.graphs``; the random numbers come
from ``_reference_rng`` (a pure-Python xoshiro256**), so a test that compares these functions
with the package compares two implementations of the same documents.

**The rewiring law** (:func:`rewire`), written from v1.4 §2.2 and nothing else:

1. the explicit edge set of $C(n, 2)$ (:func:`circulant_edges`) and the oriented list
   $L_n$ in offset-major order (:func:`original_edge_list`: $d = 1, 2$ outer, $i = 0..n-1$
   inner, $(a, b) = (i, (i + d) \\bmod n)$);
2. for each visit $j$: $B_j$ = ``next_double() < beta`` (the draw is consumed whether or not
   it succeeds);
3. if $B_j = 1$: the eligible set $A_j = V \\setminus (N(a_j) \\cup \\{a_j\\})$ from the graph
   *before* the deletion, an empty $A_j$ leaves the graph unchanged, otherwise the target is
   drawn from $A_j$ and $\\{a_j, b_j\\}$ is replaced by $\\{a_j, b'_j\\}$;
4. a visited edge that is absent from the current graph aborts (:class:`AbsentOriginalEdge`).
   The reference checks this at every visit, with $B_j = 0$ too.

*The mapping from random numbers to the target matters, not only the law.* The law of step 3
is "uniform on $A_j$". Two samplers with that law consume different random numbers and give
different graphs from one seed: ``mapping="reject"`` (the native one) draws ``next_below(n)``
until the value lies in $A_j$, ``mapping="index"`` draws one ``next_below(|A_j|)`` and takes
that position in the sorted $A_j$. The first draw of ``index`` is a different number
($|A_j| \\ne n$) and uses a different number of generator steps, so after the first
rewiring the two traces differ in the targets *and* in which stream positions the later
Bernoulli decisions read. Equality of graphs from equal seeds (the bitwise replay of a graph
record) therefore needs the mapping of the generator version, not just its law.

**The exact law** (:func:`exact_law`) is a dynamic program over graph states with no random
numbers at all: the probability of every graph the rewiring can produce at small $n$.

**Circuits** (:func:`parse_circuit`, :func:`study_draws`): the text of ``spec.circuit(u, seed)``
parsed into layers, and the same draws transcribed from the protocol of v1.4 §7.3 (per layer,
for each of the $\\lfloor n/2 \\rfloor$ gates one ``next_below(|E|)`` edge draw and one
``next_below(11520)`` Clifford draw, then one ``next_double`` per site).

**Statistics** (:func:`chi2_sf`, :func:`chi2_p`): the chi-square tail by the regularized upper
incomplete gamma function, checked in ``test_study_sampling.py`` against closed forms.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

import _reference_rng as ref
from _protocol_oracle import circuit_key

# ---------------------------------------------------------------------------
# the graph law
# ---------------------------------------------------------------------------


class AbsentOriginalEdge(AssertionError):
    """The abort of v1.4 §2.2: a visited original edge is not in the current graph."""


def circulant_edges(n: int) -> list[frozenset[int]]:
    """$E_0(n)$ as a list of 2-element frozensets: ``{i, (i + d) mod n}`` for ``d`` in 1, 2."""
    edges: set[frozenset[int]] = set()
    for i in range(n):
        for d in (1, 2):
            j = (i + d) % n
            if j != i:
                edges.add(frozenset((i, j)))
    return sorted(edges, key=lambda e: tuple(sorted(e)))


def original_edge_list(n: int) -> list[tuple[int, int]]:
    """$L_n$: ``d = 1, 2`` outer, ``i = 0..n-1`` inner, ``(a, b) = (i, (i + d) mod n)``."""
    return [(i, (i + d) % n) for d in (1, 2) for i in range(n)]


def seed_words(seed: Any) -> list[int]:
    """The xoshiro256** state words of ``seed``: ``SeedSequence(seed).generate_state(4, uint64)``.

    An all-zero state becomes ``[1, 0, 0, 0]`` (the generator's only exception).
    """
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    words = [int(w) for w in ss.generate_state(4, dtype=np.uint64)]
    return words if any(words) else [1, 0, 0, 0]


@dataclass
class Visit:
    """One visit of the sequential rewiring (for the distribution tests)."""

    j: int
    a: int
    b: int
    attempted: bool  # B_j
    eligible: tuple[int, ...] | None  # sorted A_j when attempted
    target: int | None  # b'_j when the graph changed
    draws: int  # next_below calls used for the target


def rewire(
    n: int,
    beta: float,
    seed: Any,
    *,
    order: list[tuple[int, int]] | None = None,
    mapping: str = "reject",
    record: list[Visit] | None = None,
) -> np.ndarray:
    """The sequential rewiring of $C(n, 2)$ (module docstring) as a sorted ``(2n, 2)`` array.

    ``order`` replaces $L_n$ (to test the abort), ``mapping`` is ``"reject"`` or ``"index"``,
    and ``record`` receives one :class:`Visit` per visit.
    """
    if mapping not in ("reject", "index"):
        raise ValueError(mapping)
    rng = ref.Xoshiro(seed_words(seed))
    edges = set(circulant_edges(n))
    for j, (a, b) in enumerate(original_edge_list(n) if order is None else order):
        attempted = rng.next_double() < beta
        if frozenset((a, b)) not in edges:
            raise AbsentOriginalEdge(f"visit {j}: the original edge {{{a}, {b}}} is absent")
        if not attempted:
            if record is not None:
                record.append(Visit(j, a, b, False, None, None, 0))
            continue
        neighbors = {v for v in range(n) if v != a and frozenset((a, v)) in edges}
        eligible = tuple(v for v in range(n) if v != a and v not in neighbors)
        if not eligible:
            if record is not None:
                record.append(Visit(j, a, b, True, eligible, None, 0))
            continue
        draws = 0
        if mapping == "reject":
            allowed = set(eligible)
            while True:
                draws += 1
                w = rng.next_below(n)
                if w in allowed:
                    break
        else:
            draws = 1
            w = eligible[rng.next_below(len(eligible))]
        edges.remove(frozenset((a, b)))
        edges.add(frozenset((a, w)))
        if record is not None:
            record.append(Visit(j, a, b, True, eligible, w, draws))
    return np.array(sorted(tuple(sorted(e)) for e in edges), dtype=np.int64).reshape(-1, 2)


def exact_law(
    n: int, beta: Fraction | float, *, exact: bool = False
) -> dict[int, Fraction | float]:
    """The law $\\mu_{n,\\beta}$ of the rewiring: graph bitmask -> probability (no randomness).

    A graph is the bitmask over the sorted pairs of $\\{0..n-1\\}$ (``pair_index``). The program
    follows v1.4 §2.2 literally: from every state with probability $q$, visit $j$ leaves the
    state with probability $q (1 - \\beta)$ (and with probability $q \\beta$ if $A_j$ is empty)
    and moves to each target of $A_j$ with probability $q \\beta / |A_j|$. It raises
    :class:`AbsentOriginalEdge` if any reachable state lacks the visited edge, which proves
    the abort unreachable for that $n$. ``exact=True`` uses Fractions (``beta`` a Fraction).
    """
    index = pair_index(n)

    def bit(u: int, v: int) -> int:
        return 1 << index[(u, v) if u < v else (v, u)]

    start = 0
    for e in circulant_edges(n):
        u, v = sorted(e)
        start |= bit(u, v)
    one: Fraction | float = Fraction(1) if exact else 1.0
    b: Fraction | float = Fraction(beta) if exact else float(beta)
    states: dict[int, Fraction | float] = {start: one}
    for j, (a, c) in enumerate(original_edge_list(n)):
        nxt: dict[int, Fraction | float] = {}
        for s, q in states.items():
            if not s & bit(a, c):
                raise AbsentOriginalEdge(f"visit {j}: the original edge {{{a}, {c}}} is absent")
            eligible = [v for v in range(n) if v != a and not s & bit(a, v)]
            if not eligible:
                nxt[s] = nxt.get(s, 0) + q
                continue
            nxt[s] = nxt.get(s, 0) + q * (1 - b)
            share = q * b / len(eligible)
            for w in eligible:
                t = (s & ~bit(a, c)) | bit(a, w)
                nxt[t] = nxt.get(t, 0) + share
        states = nxt
    return states


def pair_index(n: int) -> dict[tuple[int, int], int]:
    """The position of each pair ``(u, v)``, ``u < v``, in the sorted list of all pairs."""
    return {p: i for i, p in enumerate(itertools.combinations(range(n), 2))}


def graph_mask(edges: np.ndarray, n: int) -> int:
    """The bitmask (``pair_index``) of an edge array."""
    index = pair_index(n)
    mask = 0
    for u, v in np.asarray(edges).tolist():
        mask |= 1 << index[(min(u, v), max(u, v))]
    return mask


def mask_edges(mask: int, n: int) -> np.ndarray:
    pairs = list(itertools.combinations(range(n), 2))
    return np.array([p for i, p in enumerate(pairs) if mask >> i & 1], dtype=np.int64)


def degree_variance(edges: np.ndarray, n: int) -> float:
    """The population variance of the degree sequence (the mean degree is always 4)."""
    deg = np.bincount(np.asarray(edges).ravel(), minlength=n).astype(float)
    return float(np.mean((deg - deg.mean()) ** 2))


def new_edge_count(edges: np.ndarray, n: int) -> int:
    """$|E \\setminus E_0|$: the edges of the graph that are not edges of $C(n, 2)$."""
    original = {tuple(sorted(e)) for e in circulant_edges(n)}
    return sum(1 for e in np.asarray(edges).tolist() if tuple(e) not in original)


# ---------------------------------------------------------------------------
# exported circuits
# ---------------------------------------------------------------------------

_CLIFFORD_LINE = re.compile(r"CLIFFORD_2Q\(([^)]*)\)\s*(.*)")


@dataclass
class Layer:
    """One dynamics layer of an exported circuit."""

    gates: list[tuple[int, int, int]]  # (first qubit, second qubit, Clifford index)
    measured: list[int]


def parse_circuit(text: str) -> list[Layer]:
    """The dynamics layers of ``str(spec.circuit(u, seed=...))`` (everything after the first TICK).

    Layer ``t`` (``t = 1, 2, ...``) is the text between TICK ``t - 1`` and TICK ``t``: its
    ``CLIFFORD_2Q`` gates in order and its ``M`` targets (none when no site is measured).
    """
    layers: list[Layer] = []
    current = Layer([], [])
    seen_tick = False
    for line in text.splitlines():
        line = line.strip()
        if line == "TICK":
            if seen_tick:
                layers.append(current)
            seen_tick = True
            current = Layer([], [])
        elif not seen_tick:
            continue  # the Bell-pair setup
        elif line.startswith("CLIFFORD_2Q"):
            match = _CLIFFORD_LINE.fullmatch(line)
            assert match is not None, line
            indices = [int(x) for x in match.group(1).split(",")]
            targets = [int(x) for x in match.group(2).split()]
            assert len(targets) == 2 * len(indices), line
            current.gates.extend(
                (targets[2 * i], targets[2 * i + 1], indices[i]) for i in range(len(indices))
            )
        elif line.startswith("M "):
            current.measured.extend(int(x) for x in line[2:].split())
        elif line:
            raise AssertionError(f"unexpected line in a study circuit: {line!r}")
    return layers


@dataclass
class StudyDraws:
    """The raw draws of trajectory ``u``: per layer the edge and Clifford indices and the uniforms."""

    edge_index: np.ndarray  # (layers, h) values of next_below(|E|)
    clifford: np.ndarray  # (layers, h) values of next_below(11520)
    uniform: np.ndarray  # (layers, n) values of next_double


def study_draws(seed: Any, u: int, num_edges: int, n: int, layers: int) -> StudyDraws:
    """The draws of trajectory ``u`` for the study protocol, transcribed from v1.4 §7.3.

    Per layer: ``h = n // 2`` times (``next_below(num_edges)``, ``next_below(11520)``), then
    ``n`` times ``next_double``. The stream is ``stream_state(circuit_key(seed), u)``.
    """
    rng = ref.Xoshiro(ref.stream_state(circuit_key(seed), u))
    h = n // 2
    edge = np.zeros((layers, h), dtype=np.int64)
    cliff = np.zeros((layers, h), dtype=np.int64)
    uniform = np.zeros((layers, n), dtype=np.float64)
    for t in range(layers):
        for a in range(h):
            edge[t, a] = rng.next_below(num_edges)
            cliff[t, a] = rng.next_below(11520)
        for i in range(n):
            uniform[t, i] = rng.next_double()
    return StudyDraws(edge, cliff, uniform)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def _gamma_q(a: float, x: float) -> float:
    """The regularized upper incomplete gamma function Q(a, x) (series and continued fraction)."""
    if x <= 0:
        return 1.0
    log_prefix = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:  # series for P(a, x)
        term = total = 1.0 / a
        ap = a
        for _ in range(100_000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return max(0.0, 1.0 - total * math.exp(log_prefix))
    tiny = 1e-300  # modified Lentz continued fraction for Q(a, x)
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 100_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return math.exp(log_prefix) * h


def chi2_sf(x: float, dof: int) -> float:
    """$\\Pr(\\chi^2_{dof} > x)$."""
    if dof < 1:
        raise ValueError("dof must be at least 1")
    return _gamma_q(dof / 2, x / 2)


def chi2_p(counts: Any, expected: Any, *, min_expected: float = 5.0) -> tuple[float, int, float]:
    """Pearson's goodness of fit: ``(statistic, dof, p)`` of ``counts`` against ``expected``.

    Cells whose expected count is below ``min_expected`` are pooled into one cell (when the
    pool itself is below it, it joins the smallest kept cell), so the chi-square approximation
    holds. ``counts`` and ``expected`` must have equal totals.
    """
    c = np.asarray(counts, dtype=np.float64).ravel()
    e = np.asarray(expected, dtype=np.float64).ravel()
    if c.shape != e.shape or abs(c.sum() - e.sum()) > 1e-6 * max(1.0, e.sum()):
        raise ValueError("counts and expected must have the same shape and total")
    keep = e >= min_expected
    pooled_c, pooled_e = c[~keep].sum(), e[~keep].sum()
    c, e = c[keep], e[keep]
    if pooled_e > 0:
        if pooled_e >= min_expected or len(e) == 0:
            c, e = np.append(c, pooled_c), np.append(e, pooled_e)
        else:
            smallest = int(np.argmin(e))
            c[smallest] += pooled_c
            e[smallest] += pooled_e
    if len(e) < 2:
        raise ValueError("fewer than two cells after pooling")
    stat = float(np.sum((c - e) ** 2 / e))
    return stat, len(e) - 1, chi2_sf(stat, len(e) - 1)


def uniform_p(counts: Any) -> tuple[float, int, float]:
    """Chi-square test of equal cell probabilities: ``(statistic, dof, p)``."""
    c = np.asarray(counts, dtype=np.float64).ravel()
    return chi2_p(c, np.full(c.shape, c.sum() / c.size))


def binomial_pmf(m: int, p: float) -> np.ndarray:
    """The pmf of Binomial(m, p) on 0..m."""
    k = np.arange(m + 1)
    return np.array([math.comb(m, int(i)) for i in k], dtype=float) * p**k * (1 - p) ** (m - k)
