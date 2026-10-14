# General agent interface (MCP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 10-tool MCP server with a general, extensible agent interface to the whole SparseGF2 library: 31 flat-schema tools in 11 stacks, a call grammar, built-in, preset and custom tool stacks, and the library additions those tools need.

**Architecture:** A new `sparsegf2.agent` package holds:
* a tool registry (`ToolDef` plus `Param`, with JSON schemas generated from it);
* a call-grammar parser;
* stack resolution (built-in stacks, bundles, presets, custom JSON files);
* sessions for open states and background jobs;
* a renderer to compact text;
* a low-level MCP server (`mcp.server.lowlevel.Server`);
* one module of handlers per stack.

Handlers build on the existing library and `sparsegf2.api` (its validators and ensemble jobs) and never duplicate them. The library gains:
* `analysis.mean_first_failure` / `distance_bounds` / `hybrid_failure_by_size`;
* graph statistics and graph files;
* `StabilizerCode.contiguous_distance`;
* `Simulator.save/load`;
* `MonitoredCircuit.replay`;
* a `crosscheck` module for comparisons with Stim.

**Tech Stack:** Python >= 3.11, NumPy, Numba, MCP Python SDK 2.3 (`mcp>=2.2`), pydantic 2, Stim (tests and `crosscheck` only), matplotlib (optional `plot` extra), pytest, ruff.

**Spec:** [`docs/specs/2026-10-13-agent-tool-stacks-design.md`](../specs/2026-10-13-agent-tool-stacks-design.md) (revision 2). Read it before starting any task; section numbers below (e.g. "spec 5.3") refer to it.

## Global Constraints

- Python `>=3.11`; runtime dependencies stay `numpy>=1.26`, `numba>=0.59`. MCP extra: `mcp>=2.2`, `pydantic>=2`. New optional extra: `plot = ["matplotlib>=3.8"]`.
- Tool names are `<stack>_<verb>`; the 11 stacks are `help`, `state`, `circuit`, `ensemble`, `traj`, `code`, `graph`, `analysis`, `result`, `job`, `plot` (spec 6).
- Every input schema is flat: properties of type `string`, `integer`, `number`, `boolean`, string `enum`, or `array` of those; no nested objects, `$ref`, `$defs`, `oneOf`, `anyOf`, `allOf`; optional means "not in `required`" (spec 2.2).
- Each tool (name, description and schema, as JSON) is at most 5,000 characters, and the `core` bundle is at most 18,000 characters (spec 2.3).
- Results: plain text whose first line starts with the tool name, a `next:` line when there is a natural follow-up, capped at `--max-output-chars` (default 4000), and the data dict attached as `structured_content` with no NaN (spec 2.4, 5.5).
- Protocol parameters use the `MonitoredCircuit` field names exactly. Graph parameters are `graph_` plus the generator's parameter name. `reference_partner` -1 means random (spec 5.3).
- Every path argument resolves under `--root`, and `..` or absolute paths outside it are errors (spec 5.4).
- Every text file is opened with `encoding="utf-8"`.
- Stim is imported only in tests and lazily in `sparsegf2.crosscheck`; matplotlib only lazily in the plot tool.
- Commits: plain messages with no Co-Authored-By or other AI attribution trailers (Sam is the sole author).
- Tests: `python -m pytest` from the repo root (Windows: `.venv\Scripts\python -m pytest`); lint: `ruff check src tests && ruff format --check src tests`.
- Do not modify the existing JSON API's public functions' behaviour (`sparsegf2.api.run_circuit`, `run_ensemble`, ... and the CLI commands); only add.

## Review Focus

1. **Windows paths in tool arguments.** Backslashes, drive letters, a different letter case for the same root, and a relative path containing `..` that stays inside the root should all resolve correctly; only real escapes are rejected. The test goes in Task 10.
2. **Numbers sent as strings or as integral floats** (`"64"`, `64.0`, `"0.1"`, `"true"`). Local models do this. These should be coerced when unambiguous (`64.0` becomes 64), and anything else rejected with a message naming the parameter and the expected type. The test goes in Task 9.
3. **Malformed call strings**: unbalanced brackets, a trailing comma, empty strings, unicode letters, `@` with nothing after it, or a very long string. Each must give a `GrammarError` (a `ValueError`) with a column, never another exception type. The test (fuzz) goes in Task 8.
4. **NaN and infinity in results** (stopped trajectories, undefined statistics). The text shows `nan`, and `structured_content` carries `null`, never NaN, which is invalid JSON. The test goes in Task 10.
5. **The same request twice while a background job for it is still running.** The second call returns the running job's id rather than failing obscurely, and cancelling one job never corrupts another's `output_dir`. The test goes in Task 15.

## File Map

| File | Responsibility |
|---|---|
| `src/sparsegf2/analysis.py` (modify) | add `hybrid_failure_by_size`, `mean_first_failure`, `distance_bounds`, `junction_dip` |
| `src/sparsegf2/graphs.py` (modify) | add `stats`, `shortcut_fraction`, `save`, `load` |
| `src/sparsegf2/_kernel/graphstats.py` (create) | Numba BFS for mean shortest-path length |
| `src/sparsegf2/codes.py` (modify) | add `StabilizerCode.contiguous_distance` |
| `src/sparsegf2/simulator.py` (modify) | add `Simulator.save`, `Simulator.load` |
| `src/sparsegf2/mipt.py` (modify) | add `RANDOMNESS_CONTRACT`, `MonitoredCircuit.replay` |
| `src/sparsegf2/crosscheck.py` (create) | compare circuits with Stim (stabilizers per tick, measurement records) |
| `src/sparsegf2/api.py` (modify) | graph family `file` in specs |
| `src/sparsegf2/agent/__init__.py` | package marker, `INTERFACE_VERSION` |
| `src/sparsegf2/agent/grammar.py` | call grammar: tokenize, parse, bind, checkpoints, qubits, grids |
| `src/sparsegf2/agent/registry.py` | `Param`, `ToolDef`, `REGISTRY`, schemas, argument validation |
| `src/sparsegf2/agent/vocab.py` | vocabularies from the library: probes, observables, graph families, gates, fields |
| `src/sparsegf2/agent/params.py` | shared parameter groups and their conversion to API specs |
| `src/sparsegf2/agent/context.py` | `Context`: root confinement, default output dirs, limits |
| `src/sparsegf2/agent/render.py` | numbers, tables, `+-`, caps, `jsonable` |
| `src/sparsegf2/agent/sessions.py` | `StateStore`, `JobStore` (generic background jobs) |
| `src/sparsegf2/agent/stacks.py` | built-in stacks, bundles, presets, custom files, resolution, save/delete |
| `src/sparsegf2/agent/presets.json` | the four presets |
| `src/sparsegf2/agent/server.py` | the low-level MCP server, flags, `main` |
| `src/sparsegf2/agent/docs.py` | generate `docs/MCP_TOOLS.md` |
| `src/sparsegf2/agent/topics/*.md` | `help_topic` texts |
| `src/sparsegf2/agent/tools/__init__.py` | imports every tool module (registration) |
| `src/sparsegf2/agent/tools/{help,state,circuit,ensemble,traj,code,graph,analysis,result,job,plot}.py` | handlers + `ToolDef`s, one module per stack |
| `src/sparsegf2/mcp_server.py` (modify) | thin entry point delegating to `agent.server.main` |
| `src/sparsegf2/cli.py` (modify) | `stacks` and `warmup` commands |
| `tests/agent/...` (create) | tests per module (listed per task) |
| `scripts/eval_agent_tools.py`, `scripts/eval_requests.json` | local-model evaluation |
| docs: `docs/MCP_TOOLS.md` (generated), `docs/MCP_GETTING_STARTED.md`, `AGENTS.md`, `llms.txt`, `docs/LLM_INTERFACE.md`, `README.md`, `.claude/skills/sparsegf2/SKILL.md`, `.mcp.json`, `CHANGELOG.md`, `pyproject.toml` | documentation and packaging |

The work is in four phases. Each one ends with the full suite green, and is a natural place to stop and review:
1. Library additions (Tasks 1-7).
2. The agent framework, serving the help tools (Tasks 8-14).
3. The tool stacks (Tasks 15-27).
4. Coverage, documentation and release (Tasks 28-31).

---
## Phase 1: Library additions

These are useful on their own from Python, and the tools of Phase 3 build on them.

### Task 1: mu_D, distance bounds and the hybrid P_C(m) curve

**Files:**
- Modify: `src/sparsegf2/analysis.py` (add after `exact_failure_by_size`, and to `__all__`)
- Test: `tests/test_analysis_erasure_stats.py`

**Interfaces:**
- Consumes: the existing private helpers `_first_failure_array`, `_matching_counts`, and the public `failure_by_size`, `exact_failure_by_size`, `erasure_failure_probability` in `analysis.py`.
- Produces:
  - `hybrid_failure_by_size(first_failure, num_qubits, *, exact_counts=None) -> np.ndarray` with shape `(..., n + 1)`.
  - `mean_first_failure(first_failure, num_qubits, *, exact_counts=None) -> np.ndarray` with shape `(...)`.
  - `distance_bounds(first_failure, num_qubits, *, exact_counts=None) -> tuple[np.ndarray, np.ndarray]`, i.e. `(low, high)` as int64 with shape `(...)`.
  - `junction_dip(first_failure, num_qubits, exact_counts) -> np.ndarray` (bool, shape `(...)`).

- [ ] **Step 1: Write the failing tests**

```python
"""mu_D, distance bounds and the hybrid P_C(m) curve (analysis.py)."""

import numpy as np
import pytest

from sparsegf2 import StabilizerCode, analysis


def _code_data(code, samples=200, w=2):
    first = code.first_failure_sizes(samples, seed=0)
    counts = code.uncorrectable_counts(w)
    return first, counts


def test_422_code_has_distance_two_and_mean_first_failure_two():
    code = StabilizerCode(["XXXX", "ZZZZ"])  # [[4, 2, 2]]: every pair is uncorrectable
    first, counts = _code_data(code)
    assert np.all(first == 2)
    assert analysis.mean_first_failure(first, 4) == pytest.approx(2.0)
    assert analysis.mean_first_failure(first, 4, exact_counts=counts) == pytest.approx(2.0)
    low, high = analysis.distance_bounds(first, 4, exact_counts=counts)
    assert (int(low), int(high)) == (2, 2)


def test_code_without_logical_qubits_gives_n_plus_one():
    code = StabilizerCode(["XX", "ZZ"])  # k = 0
    first, counts = _code_data(code, w=1)
    assert np.all(first == 3)
    assert analysis.mean_first_failure(first, 2, exact_counts=counts) == pytest.approx(3.0)
    low, high = analysis.distance_bounds(first, 2, exact_counts=counts)
    assert (int(low), int(high)) == (3, 3)


def test_without_counts_mean_first_failure_is_the_sample_mean():
    rng = np.random.default_rng(1)
    first = rng.integers(1, 12, size=(5, 40))  # n = 10, D in [1, 11]
    np.testing.assert_allclose(analysis.mean_first_failure(first, 10), first.mean(axis=-1))


def test_hybrid_curve_uses_exact_counts_up_to_w_and_reproduces_phi():
    rng = np.random.default_rng(2)
    n, w = 12, 2
    first = rng.integers(3, 14, size=(4, 64))  # no failures at m <= 2 in the samples
    counts = np.array([[0, 3], [0, 0], [1, 5], [0, 1]], dtype=float)
    curve = analysis.hybrid_failure_by_size(first, n, exact_counts=counts)
    assert curve.shape == (4, n + 1)
    np.testing.assert_allclose(curve[:, 1 : w + 1], analysis.exact_failure_by_size(counts, n))
    np.testing.assert_allclose(curve[:, w + 1 :], analysis.failure_by_size(first, n)[:, w + 1 :])
    for e in (0.05, 0.2, 0.5):
        m = np.arange(n + 1)
        b = np.array([np.exp(analysis._log_binomial(n)[i]) for i in m]) * e**m * (1 - e) ** (n - m)
        np.testing.assert_allclose(
            curve @ b, analysis.erasure_failure_probability(first, n, e, exact_counts=counts)
        )


def test_distance_bounds_without_counts_and_with_zero_counts():
    first = np.array([[5, 7, 6], [4, 4, 9]])
    low, high = analysis.distance_bounds(first, 10)
    np.testing.assert_array_equal(low, [1, 1])
    np.testing.assert_array_equal(high, [5, 4])
    low, high = analysis.distance_bounds(first, 10, exact_counts=np.zeros((2, 3)))
    np.testing.assert_array_equal(low, [4, 4])  # w + 1 = 4
    np.testing.assert_array_equal(high, [5, 4])


def test_junction_dip_flags_a_sampled_value_below_the_exact_one():
    n = 6
    counts = np.array([[0, 15]], dtype=float)  # every pair fails: exact P(2) = 1
    first = np.array([[6, 6, 6, 6]])  # samples say P(3) = 0 < 1
    assert analysis.junction_dip(first, n, counts).tolist() == [True]
    first_ok = np.array([[2, 2, 2, 2]])
    assert analysis.junction_dip(first_ok, n, counts).tolist() == [False]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_analysis_erasure_stats.py -q`
Expected: FAIL with `AttributeError: module 'sparsegf2.analysis' has no attribute 'mean_first_failure'` (and the other new names)

- [ ] **Step 3: Implement**

Add to `src/sparsegf2/analysis.py`, after `exact_failure_by_size`. Add the four names to `__all__` (keep it sorted).

```python
def hybrid_failure_by_size(first_failure, num_qubits: int, *, exact_counts=None) -> np.ndarray:
    """P_C(m) for m = 0..n as the hybrid estimators see it, shape ``(..., n + 1)``.

    Without ``exact_counts`` this is :func:`failure_by_size`. With counts of
    the uncorrectable m-subsets for m = 1..w, the entries m <= w are the exact
    P_C(m) and the entries m > w the sampled fractions. This is the curve
    behind :func:`erasure_failure_probability`, whose hybrid estimate equals
    ``sum_m C(n, m) e^m (1 - e)^(n - m) P_C(m)`` of this curve.
    """
    n = int(num_qubits)
    curve = failure_by_size(first_failure, n)
    if exact_counts is None:
        return curve
    d = _first_failure_array(first_failure, n)
    exact = exact_failure_by_size(_matching_counts(exact_counts, d), n)
    out = curve.copy()
    out[..., 1 : exact.shape[-1] + 1] = exact
    return out


def mean_first_failure(first_failure, num_qubits: int, *, exact_counts=None) -> np.ndarray:
    """mu_D = E[D] = sum_{m=0}^{n} (1 - P_C(m)) per code, shape ``(...)``.

    Uses :func:`hybrid_failure_by_size`, so it agrees with
    :func:`erasure_failure_probability` given the same data. A code without
    logical qubits (every D = n + 1) gives n + 1; condition on k > 0 before
    averaging over an ensemble.
    """
    curve = hybrid_failure_by_size(first_failure, num_qubits, exact_counts=exact_counts)
    return (1.0 - curve).sum(axis=-1)


def distance_bounds(
    first_failure, num_qubits: int, *, exact_counts=None
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds ``(low, high)`` on each code's distance d, as int64 arrays of shape ``(...)``.

    Erasing fewer than d qubits is always correctable, so every sampled
    first-failure size D_j >= d and ``high = min_j D_j``. Without counts
    ``low = 1``. With exhaustive counts for m = 1..w, the smallest m with an
    uncorrectable m-subset is d exactly (``low = high = m``); if there is none,
    ``low = w + 1``. A code without logical qubits (min D = n + 1) gets
    ``(n + 1, n + 1)``: it has no distance, so reports exclude it.
    """
    n = int(num_qubits)
    d = _first_failure_array(first_failure, n)
    high = d.min(axis=-1)
    low = np.ones_like(high)
    if exact_counts is not None:
        c = _matching_counts(exact_counts, d)
        w = c.shape[-1]
        positive = c > 0
        found = positive.any(axis=-1)
        first = np.argmax(positive, axis=-1) + 1
        low = np.where(found, first, w + 1)
        high = np.where(found, first, high)
    no_code = d.min(axis=-1) == n + 1
    low = np.where(no_code, n + 1, low)
    high = np.where(no_code, n + 1, high)
    return low.astype(np.int64), high.astype(np.int64)


def junction_dip(first_failure, num_qubits: int, exact_counts) -> np.ndarray:
    """Whether the sampled P_C(w + 1) falls below the exact P_C(w), per code.

    The true P_C(m) is non-decreasing; a dip at the junction of the hybrid
    curve is sampling noise. The estimators stay unbiased (nothing is
    clipped); reports flag it and suggest more orderings.
    """
    n = int(num_qubits)
    curve = hybrid_failure_by_size(first_failure, n, exact_counts=exact_counts)
    w = np.asarray(exact_counts).shape[-1]
    if w >= n:
        return np.zeros(curve.shape[:-1], dtype=bool)
    return curve[..., w + 1] < curve[..., w]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_analysis_erasure_stats.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint and commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/analysis.py tests/test_analysis_erasure_stats.py
git commit -m "analysis: mean first-failure size, distance bounds, hybrid P_C(m) curve, junction-dip flag"
```

---

### Task 2: Graph statistics

**Files:**
- Create: `src/sparsegf2/_kernel/graphstats.py`
- Modify: `src/sparsegf2/graphs.py` (add `stats`, `shortcut_fraction`, `MAX_STATS_NODES`, and `__all__` entries)
- Test: `tests/test_graph_stats.py`

**Interfaces:**
- Consumes: `graphs.degrees(edges, n)`, `graphs.cycle`, `graphs.complete`, `graphs.watts_strogatz`, `graphs.path`.
- Produces:
  - `graphs.stats(edges, n) -> dict[str, float | int | bool]` with keys `lambda2`, `connected`, `mean_path_length`, `clustering`, `degree_min`, `degree_mean`, `degree_max`.
  - `graphs.shortcut_fraction(edges, n, k) -> float`.
  - `graphs.MAX_STATS_NODES = 4096`.

- [ ] **Step 1: Write the failing tests**

```python
"""graphs.stats and graphs.shortcut_fraction against closed forms."""

import math

import numpy as np
import pytest

from sparsegf2 import graphs


def test_cycle():
    n = 10
    s = graphs.stats(graphs.cycle(n), n)
    assert s["lambda2"] == pytest.approx(2 - 2 * math.cos(2 * math.pi / n))
    assert s["connected"] is True
    assert s["mean_path_length"] == pytest.approx(n * n / (4 * (n - 1)))  # even n
    assert s["clustering"] == 0.0
    assert (s["degree_min"], s["degree_mean"], s["degree_max"]) == (2, 2.0, 2)


def test_complete_graph():
    s = graphs.stats(graphs.complete(6), 6)
    assert s["lambda2"] == pytest.approx(6.0)
    assert s["clustering"] == pytest.approx(1.0)
    assert s["mean_path_length"] == pytest.approx(1.0)


def test_ring_lattice_clustering():
    k = 4
    s = graphs.stats(graphs.watts_strogatz(20, k, 0.0, seed=0), 20)
    assert s["clustering"] == pytest.approx(3 * (k - 2) / (4 * (k - 1)))


def test_disconnected_graph():
    edges = graphs.from_edges([(0, 1), (2, 3)], 4)
    s = graphs.stats(edges, 4)
    assert s["connected"] is False
    assert math.isnan(s["mean_path_length"])
    assert s["lambda2"] == pytest.approx(0.0, abs=1e-12)


def test_too_large_is_refused():
    with pytest.raises(ValueError, match="at most 4096"):
        graphs.stats(graphs.cycle(5000), 5000)


def test_shortcut_fraction():
    assert graphs.shortcut_fraction(graphs.watts_strogatz(64, 4, 0.0, seed=1), 64, 4) == 0.0
    f = graphs.shortcut_fraction(graphs.watts_strogatz(64, 4, 1.0, seed=1), 64, 4)
    assert 0.5 < f <= 1.0
    with pytest.raises(ValueError, match="even"):
        graphs.shortcut_fraction(graphs.cycle(8), 8, 3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_graph_stats.py -q`
Expected: FAIL with `AttributeError: module 'sparsegf2.graphs' has no attribute 'stats'`

- [ ] **Step 3: Implement the BFS kernel**

Create `src/sparsegf2/_kernel/graphstats.py`:

```python
"""Breadth-first search over a graph in CSR form, for graphs.stats."""

import numpy as np
from numba import njit


@njit(cache=True)
def path_length_sum(indptr, indices, n):
    """(sum of shortest-path lengths, number of reachable ordered pairs) over all sources."""
    dist = np.empty(n, np.int64)
    queue = np.empty(n, np.int64)
    total = 0
    pairs = 0
    for s in range(n):
        dist[:] = -1
        dist[s] = 0
        head = 0
        tail = 1
        queue[0] = s
        while head < tail:
            u = queue[head]
            head += 1
            for j in range(indptr[u], indptr[u + 1]):
                v = indices[j]
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    total += dist[v]
                    pairs += 1
                    queue[tail] = v
                    tail += 1
    return total, pairs
```

- [ ] **Step 4: Implement `stats` and `shortcut_fraction`**

Add to `src/sparsegf2/graphs.py` (and to `__all__`: `"MAX_STATS_NODES"`, `"shortcut_fraction"`, `"stats"`). The API's `_graph_families` excludes helpers by name, so add `"stats"`, `"shortcut_fraction"`, `"save"`, `"load"` to `api._GRAPH_HELPERS` in Task 3. Until then, add only `stats` and `shortcut_fraction` to that set in this task:

```python
MAX_STATS_NODES = 4096


def _adjacency(edges: np.ndarray, n: int) -> np.ndarray:
    adj = np.zeros((n, n))
    adj[edges[:, 0], edges[:, 1]] = 1.0
    adj[edges[:, 1], edges[:, 0]] = 1.0
    return adj


def _csr(edges: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.concatenate([edges[:, 0], edges[:, 1]])
    b = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(a, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(a, minlength=n))
    return indptr, b[order].astype(np.int64)


def stats(edges: np.ndarray, n: int) -> dict:
    """Spectral, distance and clustering statistics of one graph on ``n`` nodes.

    ``lambda2``: the algebraic connectivity, i.e. the second-smallest
    eigenvalue of the Laplacian (0 when disconnected). ``connected``.
    ``mean_path_length``: the mean shortest-path length over ordered pairs of
    distinct nodes (NaN when disconnected). ``clustering``: the average local
    clustering coefficient, with nodes of degree < 2 counting 0. Then
    ``degree_min``, ``degree_mean`` and ``degree_max``. Dense linear algebra,
    so ``n <= MAX_STATS_NODES``.
    """
    from sparsegf2._kernel.graphstats import path_length_sum

    n = int(n)
    if n > MAX_STATS_NODES:
        raise ValueError(f"graph statistics need n <= {MAX_STATS_NODES} (dense), got n = {n}")
    e = from_edges(np.asarray(edges, dtype=np.int64).reshape(-1, 2), n)
    deg = degrees(e, n)
    adj = _adjacency(e, n)
    lam = np.linalg.eigvalsh(np.diag(adj.sum(axis=1)) - adj)
    lambda2 = float(max(lam[1], 0.0)) if n > 1 else 0.0
    triangles = ((adj @ adj) * adj).sum(axis=1) / 2.0
    k = adj.sum(axis=1)
    possible = k * (k - 1) / 2.0
    local = np.divide(triangles, possible, out=np.zeros(n), where=possible > 0)
    indptr, indices = _csr(e, n)
    total, pairs = path_length_sum(indptr, indices, n)
    connected = bool(pairs == n * (n - 1))
    return {
        "lambda2": lambda2,
        "connected": connected,
        "mean_path_length": float(total / pairs) if connected and pairs else float("nan"),
        "clustering": float(local.mean()) if n else 0.0,
        "degree_min": int(deg.min()) if n else 0,
        "degree_mean": float(deg.mean()) if n else 0.0,
        "degree_max": int(deg.max()) if n else 0,
    }


def shortcut_fraction(edges: np.ndarray, n: int, k: int) -> float:
    """Fraction of edges whose ring distance exceeds k/2: the shortcuts of a ring-based graph.

    ``k`` is the ring lattice's total degree (``watts_strogatz`` /
    ``newman_watts``), so its own edges reach k/2 sites each way.
    """
    if k % 2:
        raise ValueError(f"k must be even (k/2 neighbours per side), got {k}")
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not len(e):
        return 0.0
    r = np.abs(e[:, 0] - e[:, 1])
    r = np.minimum(r, n - r)
    return float(np.mean(r > k // 2))
```

In `src/sparsegf2/api.py`, extend the helper set so the new functions are not mistaken for graph families:

```python
_GRAPH_HELPERS = {"from_edges", "from_networkx", "degrees", "edge_coloring", "stats", "shortcut_fraction"}
```

and exclude the constant from the families: in `_graph_families`, add `and callable(getattr(graphs, name))` to the comprehension's condition.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_graph_stats.py tests/test_api.py tests/test_graphs.py -q`
Expected: PASS (the API's capabilities list of graph families is unchanged)

- [ ] **Step 6: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/_kernel/graphstats.py src/sparsegf2/graphs.py src/sparsegf2/api.py tests/test_graph_stats.py
git commit -m "graphs: stats (lambda_2, mean path length, clustering, degrees) and shortcut_fraction"
```

---

### Task 3: Graph files, and the `file` family in specs

**Files:**
- Modify: `src/sparsegf2/graphs.py` (add `save`, `load`)
- Modify: `src/sparsegf2/api.py` (`_GRAPH_HELPERS`, `_build_graph`, `_build_spec`, capabilities text for graphs)
- Test: `tests/test_graph_files.py`

**Interfaces:**
- Consumes: `graphs.from_edges`.
- Produces:
  - `graphs.save(path, graphs_, num_nodes, weights=None) -> pathlib.Path`.
  - `graphs.load(path) -> tuple[int, list[np.ndarray], list[np.ndarray] | None]`. Edges come back normalised as by `from_edges` (pairs `a < b`, sorted), with weights permuted to match.
  - JSON API: `"graph": {"family": "file", "path": PATH}`. Weights stored in the file become the spec's `edge_weights`, and giving `edge_weights` as well is an error.

- [ ] **Step 1: Write the failing tests**

```python
"""graphs.save / graphs.load and the 'file' graph family of the JSON API."""

import json

import numpy as np
import pytest

from sparsegf2 import api, graphs


@pytest.mark.parametrize("suffix", [".json", ".npz"])
def test_round_trip_collection_with_weights(tmp_path, suffix):
    gs = [graphs.watts_strogatz(12, 4, 0.3, seed=(0, i)) for i in range(3)]
    ws = [np.linspace(1.0, 2.0, len(g)) for g in gs]
    path = graphs.save(tmp_path / f"c{suffix}", gs, 12, weights=ws)
    n, loaded, lw = graphs.load(path)
    assert n == 12 and len(loaded) == 3
    for a, b, wa, wb in zip(gs, loaded, ws, lw, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(wa, wb)


def test_json_single_graph_in_any_order_is_normalised(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(
        json.dumps({"num_nodes": 4, "edges": [[3, 2], [0, 1], [2, 1]], "weights": [3.0, 1.0, 2.0]}),
        encoding="utf-8",
    )
    n, gs, ws = graphs.load(path)
    np.testing.assert_array_equal(gs[0], [[0, 1], [1, 2], [2, 3]])
    np.testing.assert_allclose(ws[0], [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"num_nodes": 3, "edges": [[0, 3]]}, "node 3 outside"),
        ({"num_nodes": 3, "edges": [[1, 1]]}, "self-loop"),
        ({"num_nodes": 3, "edges": [[0, 1], [1, 0]]}, "repeated edge"),
        ({"edges": [[0, 1]]}, "num_nodes"),
        ({"num_nodes": 3}, "edges"),
    ],
)
def test_validation_errors_name_the_problem(tmp_path, payload, message):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        graphs.load(path)


def test_file_family_in_ensemble_spec(tmp_path):
    gs = [graphs.cycle(8), graphs.complete(8)]
    path = graphs.save(tmp_path / "two.json", gs, 8)
    spec = {"num_system": 8, "graph": {"family": "file", "path": str(path)}, "layers": 4,
            "measurement_rate": 0.2, "reference": "purification",
            "probes": [{"kind": "code_dimension"}]}
    out = api.run_ensemble(spec, seed=1)  # a collection: every graph once
    assert out["num_trajectories"] == 2


def test_file_family_size_mismatch_and_weights_conflict(tmp_path):
    path = graphs.save(tmp_path / "w.npz", [graphs.cycle(8)], 8, weights=[np.ones(8)])
    base = {"graph": {"family": "file", "path": str(path)}, "layers": 2, "measurement_rate": 0.1}
    with pytest.raises(ValueError, match="num_nodes 8"):
        api.estimate_ensemble({**base, "num_system": 10}, 4)
    with pytest.raises(ValueError, match="already holds weights"):
        api.estimate_ensemble(
            {**base, "num_system": 8, "schedule": "random_edges",
             "edge_weights": {"family": "power_law", "alpha": 2.0}}, 4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_graph_files.py -q`
Expected: FAIL with `AttributeError: module 'sparsegf2.graphs' has no attribute 'save'`

- [ ] **Step 3: Implement `save` / `load`**

Add to `src/sparsegf2/graphs.py` (top: `import json`, `from pathlib import Path`; `__all__`: `"load"`, `"save"`):

```python
def _checked(edges, n: int, where: str) -> np.ndarray:
    arr = np.asarray(edges, dtype=np.int64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{where}: edges must be [a, b] pairs")
    bad = arr[(arr < 0) | (arr >= n)]
    if bad.size:
        raise ValueError(f"{where}: node {int(bad[0])} outside [0, {n})")
    loops = np.flatnonzero(arr[:, 0] == arr[:, 1])
    if loops.size:
        raise ValueError(f"{where}: self-loop at node {int(arr[loops[0], 0])}")
    return arr


def _normalised(arr: np.ndarray, weights, where: str):
    oriented = np.sort(arr, axis=1)
    order = np.lexsort((oriented[:, 1], oriented[:, 0]))
    ranked = oriented[order]
    if len(ranked) > 1:
        same = np.flatnonzero(np.all(ranked[1:] == ranked[:-1], axis=1))
        if same.size:
            a, b = ranked[same[0]]
            raise ValueError(f"{where}: repeated edge ({int(a)}, {int(b)})")
    if weights is None:
        return ranked, None
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != (len(arr),):
        raise ValueError(f"{where}: {w.size} weights for {len(arr)} edges")
    return ranked, w[order]


def save(path, graphs_, num_nodes: int, weights=None) -> Path:
    """Save a graph collection to ``.json`` or ``.npz`` (the format follows the suffix).

    ``graphs_`` is one edge array or a list of them; ``weights`` (optional)
    is one array per graph, one weight per edge, in the order of its edges.
    """
    path = Path(path)
    gs = [graphs_] if isinstance(graphs_, np.ndarray) and np.ndim(graphs_) == 2 else list(graphs_)
    ws = None if weights is None else list(weights)
    n = int(num_nodes)
    pairs = [
        _normalised(_checked(g, n, f"graph {i}"), None if ws is None else ws[i], f"graph {i}")
        for i, g in enumerate(gs)
    ]
    if path.suffix == ".json":
        doc = {"num_nodes": n, "graphs": [e.tolist() for e, _ in pairs]}
        if ws is not None:
            doc["weights"] = [w.tolist() for _, w in pairs]
        path.write_text(json.dumps(doc), encoding="utf-8")
    elif path.suffix == ".npz":
        edges = np.concatenate([e for e, _ in pairs]) if pairs else np.zeros((0, 2), np.int64)
        ptr = np.cumsum([0] + [len(e) for e, _ in pairs]).astype(np.int64)
        arrays = {"num_nodes": np.int64(n), "edges": edges, "ptr": ptr}
        if ws is not None:
            arrays["weights"] = np.concatenate([w for _, w in pairs])
        with path.open("wb") as fh:
            np.savez(fh, **arrays)
    else:
        raise ValueError(f"graph files end in .json or .npz, got {path.name!r}")
    return path


def load(path) -> tuple[int, list[np.ndarray], list[np.ndarray] | None]:
    """Load a collection saved by :func:`save` (or written by hand in the same format).

    JSON: ``{"num_nodes": n, "graphs": [[[a, b], ...], ...], "weights": [[w, ...], ...]}``,
    or ``"edges"`` for a single graph. NPZ: ``num_nodes``, ``edges`` (E x 2), ``ptr``
    (G + 1 offsets; optional for one graph), optionally ``weights`` (E). Returns
    ``(num_nodes, graphs, weights or None)`` with edges normalised as by
    :func:`from_edges` and the weights permuted with them.
    """
    path = Path(path)
    if path.suffix == ".json":
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "num_nodes" not in doc:
            raise ValueError(f"{path.name}: missing num_nodes")
        raw = doc.get("graphs") or ([doc["edges"]] if "edges" in doc else None)
        if raw is None:
            raise ValueError(f"{path.name}: give 'graphs' (a list of edge lists) or 'edges'")
        weights = doc.get("weights")
        if weights is not None and "edges" in doc and "graphs" not in doc:
            weights = [weights]
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            if "num_nodes" not in z or "edges" not in z:
                raise ValueError(f"{path.name}: needs arrays num_nodes and edges")
            doc = {"num_nodes": int(z["num_nodes"])}
            edges = z["edges"]
            ptr = z["ptr"] if "ptr" in z else np.array([0, len(edges)])
            raw = [edges[ptr[i] : ptr[i + 1]] for i in range(len(ptr) - 1)]
            flat_w = z["weights"] if "weights" in z else None
            weights = None if flat_w is None else [flat_w[ptr[i] : ptr[i + 1]] for i in range(len(ptr) - 1)]
    else:
        raise ValueError(f"graph files end in .json or .npz, got {path.name!r}")
    n = int(doc["num_nodes"])
    out_g, out_w = [], []
    for i, g in enumerate(raw):
        where = f"{path.name}, graph {i}"
        e, w = _normalised(_checked(g, n, where), None if weights is None else weights[i], where)
        out_g.append(e)
        out_w.append(w)
    return n, out_g, (out_w if weights is not None else None)
```

- [ ] **Step 4: Wire the `file` family into the JSON API**

In `src/sparsegf2/api.py`:

```python
_GRAPH_HELPERS = {
    "from_edges", "from_networkx", "degrees", "edge_coloring", "stats", "shortcut_fraction",
    "save", "load",
}
```

Add a helper above `_build_graph`:

```python
def _file_graph(graph: dict, n: int) -> tuple[np.ndarray | list[np.ndarray], Any]:
    """Edges (one array, or a list for a collection) and weights of a ``{"family": "file"}`` graph."""
    extra = sorted(set(graph) - {"family", "path"})
    if extra:
        raise ValueError(f"graph (file): unknown key(s) {extra}; give only 'path'")
    path = graph.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError('graph (file): give "path", a .json or .npz graph file')
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"graph (file): no such file {str(p)!r}")
    num_nodes, gs, ws = graphs.load(p)
    if num_nodes != n:
        raise ValueError(
            f"graph (file): {p.name} has num_nodes {num_nodes}, but num_system is {n}"
        )
    if len(gs) == 1:
        return gs[0], (None if ws is None else ws[0])
    return gs, ws
```

In `_build_graph`, at the very top of the function body (before `families = ...`):

```python
    if isinstance(graph, dict) and graph.get("family") == "file":
        return _file_graph(graph, n)[0]
```

In `_build_spec`, replace the block

```python
    if "graph" in spec:
        edges = _build_graph(spec["graph"], n)
```

with

```python
    if "graph" in spec:
        graph = spec["graph"]
        if isinstance(graph, dict) and graph.get("family") == "file":
            edges, file_weights = _file_graph(graph, n)
            if file_weights is not None:
                if weights is not None:
                    raise ValueError(
                        "the graph file already holds weights; drop edge_weights"
                    )
                kwargs["edge_weights"] = file_weights
        else:
            edges = _build_graph(graph, n)
```

Nothing else changes. When the spec has no `edge_weights`, `weights` is `None`, so the later block (`if weights is not None: kwargs["edge_weights"] = ...`) is skipped and the file's weights stay in `kwargs`. When the spec has `edge_weights` and the file has none, the existing path builds them. When both have weights, the new check above raises.

In `capabilities()`'s graph-family section (search for `"graph_families"` in `api.py`), add the entry `"file": "a .json or .npz graph collection: {\"family\": \"file\", \"path\": ...} (any geometry; see graphs.save)"` next to the generated families.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_graph_files.py tests/test_api.py tests/test_graphs.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/graphs.py src/sparsegf2/api.py tests/test_graph_files.py
git commit -m "graphs: save/load graph collections (JSON, NPZ); 'file' graph family in specs"
```

---

### Task 4: `StabilizerCode.contiguous_distance` and `StabilizerCode.from_simulator`

**Files:**
- Modify: `src/sparsegf2/codes.py`
- Test: `tests/test_codes.py` (append)

**Interfaces:**
- Consumes: `StabilizerCode.lost_information`, `StabilizerCode.num_logical`, `StabilizerCode.num_qubits`, `Simulator.stabilizer_arrays()`.
- Produces:
  - `StabilizerCode.contiguous_distance(order=None, *, periodic=True) -> int`, which returns 0 when no window is uncorrectable.
  - `StabilizerCode.from_simulator(sim, reference) -> StabilizerCode` (classmethod): the code the non-reference qubits hold, with qubits renumbered `0..n_sys-1` in increasing order of the original indices.

- [ ] **Step 1: Write the failing tests (append to `tests/test_codes.py`)**

```python
def test_contiguous_distance_matches_the_engine_probe():
    import numpy as np

    from sparsegf2 import graphs, mipt

    n = 16
    spec = mipt.MonitoredCircuit(
        num_system=n, edges=graphs.cycle(n), layers=12, measurement_rate=0.1,
        reference="purification",
        probes=[mipt.code_dimension(), mipt.contiguous_distance(), mipt.code_snapshot()],
    )
    res = spec.run(40, seed=3)
    k = res.values["code_dimension"][:, -1]
    for t in range(res.num_trajectories):
        code = res.code(t, 0)
        if k[t] > 0:
            assert code.contiguous_distance() == res.values["contiguous_distance"][t, -1]
        else:
            assert code.contiguous_distance() == 0


def test_contiguous_distance_respects_order_and_periodic():
    from sparsegf2 import StabilizerCode

    code = StabilizerCode(["XXXX", "ZZZZ"])  # any pair is uncorrectable
    assert code.contiguous_distance() == 2
    assert code.contiguous_distance([3, 2, 1, 0], periodic=False) == 2
    import pytest

    with pytest.raises(ValueError, match="permutation"):
        code.contiguous_distance([0, 0, 1, 2])
```

Also append:

```python
def _rref(m):
    import numpy as np

    m = (np.asarray(m) % 2).astype(np.uint8)
    r = 0
    for c in range(m.shape[1]):
        nz = np.flatnonzero(m[r:, c])
        if not nz.size:
            continue
        p = r + nz[0]
        m[[r, p]] = m[[p, r]]
        others = np.flatnonzero(m[:, c])
        m[others[others != r]] ^= m[r]
        r += 1
        if r == m.shape[0]:
            break
    return m[:r]


def test_from_simulator_equals_the_engine_code_snapshot():
    import numpy as np

    import sparsegf2 as sg
    from sparsegf2 import StabilizerCode, graphs, mipt

    n = 12
    spec = mipt.MonitoredCircuit(
        num_system=n, edges=graphs.cycle(n), layers=10, measurement_rate=0.2,
        reference="purification", probes=[mipt.code_snapshot()],
    )
    res = spec.run(6, seed=5)
    for t in range(6):
        sim = sg.Simulator(spec.num_qubits, phases=False, seed=0)
        sim.run(spec.circuit(t, seed=5))
        mine = StabilizerCode.from_simulator(sim, reference=range(n, 2 * n))
        engine = res.code(t, 0)
        assert mine.num_qubits == n and mine.num_logical == engine.num_logical
        np.testing.assert_array_equal(
            _rref(mine.stabilizer_matrix()), _rref(engine.stabilizer_matrix())
        )


def test_from_simulator_fresh_purification_state_has_no_stabilizers():
    import sparsegf2 as sg
    from sparsegf2 import StabilizerCode

    sim = sg.Simulator(4, phases=False)
    for q in (0, 1):
        sim.h(q)
        sim.cx(q, q + 2)
    code = StabilizerCode.from_simulator(sim, reference=[2, 3])
    assert (code.num_qubits, code.num_logical, code.num_stabilizers) == (2, 2, 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_codes.py -q -k "contiguous or from_simulator"`
Expected: FAIL with `AttributeError: 'StabilizerCode' object has no attribute 'contiguous_distance'`

- [ ] **Step 3: Implement (add to `StabilizerCode`, after `uncorrectable_counts`)**

```python
    def contiguous_distance(self, order=None, *, periodic: bool = True) -> int:
        """Length of the shortest contiguous window of ``order`` whose erasure is uncorrectable.

        ``order`` is a permutation of the qubits (default ``0..n-1``); windows
        wrap around its end when ``periodic``. Returns 0 when no window is
        uncorrectable (in particular when the code has no logical qubits).
        Uncorrectability is monotone in the window length, so the length is
        found by bisection, with all windows of a length checked.
        """
        n = self.num_qubits
        seq = np.arange(n) if order is None else np.asarray(list(order), dtype=np.int64)
        if seq.shape != (n,) or not np.array_equal(np.sort(seq), np.arange(n)):
            raise ValueError(f"order must be a permutation of 0..{n - 1}")
        if self.num_logical == 0:
            return 0

        def some_window_fails(length: int) -> bool:
            starts = range(n) if periodic and length < n else range(n - length + 1)
            for s in starts:
                window = seq[np.arange(s, s + length) % n]
                if self.lost_information(window) > 0:
                    return True
            return False

        if not some_window_fails(n):
            return 0
        lo, hi = 1, n  # some window of length hi fails
        while lo < hi:
            mid = (lo + hi) // 2
            if some_window_fails(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
```

Then add the classmethod (after `from_packed`):

```python
    @classmethod
    def from_simulator(cls, sim, reference) -> StabilizerCode:
        """The code a simulator's system holds, the system being every qubit not in ``reference``.

        Gaussian elimination on the reference columns of the stabilizer
        generators leaves generators of the subgroup acting trivially on the
        reference; restricted to the system (renumbered 0..n_sys-1 in
        increasing order) they are the code's stabilizers, so ``num_logical``
        is S(reference). This is the same group the ensemble engine's
        ``code_snapshot`` saves.
        """
        n = sim.num_qubits
        ref = sorted({int(q) for q in reference})
        if not ref or ref[0] < 0 or ref[-1] >= n:
            raise ValueError(f"reference must be a non-empty set of qubits in [0, {n})")
        refset = set(ref)
        system = [q for q in range(n) if q not in refset]
        if not system:
            raise ValueError("the reference covers every qubit; there is no system")
        xs, zs, _ = sim.stabilizer_arrays()
        m = np.concatenate([xs[:, ref], zs[:, ref], xs[:, system], zs[:, system]], axis=1)
        m = (m % 2).astype(np.uint8)
        r, rows = 2 * len(ref), m.shape[0]
        pivot = 0
        for col in range(r):
            nz = np.flatnonzero(m[pivot:, col])
            if not nz.size:
                continue
            p = pivot + int(nz[0])
            m[[pivot, p]] = m[[p, pivot]]
            others = np.flatnonzero(m[:, col])
            m[others[others != pivot]] ^= m[pivot]
            pivot += 1
            if pivot == rows:
                break
        return cls(m[pivot:, r:], num_qubits=len(system))
```

If the constructor rejects an empty `(0, 2n)` matrix (the fresh-state test), make `_bit_rows` accept it: an empty generator list is a valid code with k = n.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_codes.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/codes.py tests/test_codes.py
git commit -m "codes: StabilizerCode.contiguous_distance and from_simulator, both checked against the engine"
```

---

### Task 5: `Simulator.save` / `Simulator.load`

**Files:**
- Modify: `src/sparsegf2/simulator.py`
- Test: `tests/test_simulator.py` (append)

**Interfaces:**
- Produces:
  - `Simulator.save(path) -> pathlib.Path`, which writes `.npz` arrays `tab`, `rng`, `meta` (JSON: `format`, `num_qubits`, `phases`, `seed_entropy` as a string, `seed_spawn_key`, `version`).
  - `Simulator.load(path) -> Simulator` (classmethod), which restores the exact tableau and the position of the random stream.

- [ ] **Step 1: Write the failing tests (append to `tests/test_simulator.py`)**

```python
def test_save_load_round_trip_continues_the_same_random_stream(tmp_path):
    import sparsegf2 as sg

    for phases in (True, False):
        sim = sg.Simulator(5, phases=phases, seed=11)
        sim.h(0)
        sim.cx(0, 1)
        sim.h(2)
        sim.measure(2)
        path = sim.save(tmp_path / f"s{phases}.state.npz")
        back = sg.Simulator.load(path)
        assert back.phases == phases and back.num_qubits == 5
        assert [str(p) for p in back.stabilizers()] == [str(p) for p in sim.stabilizers()]
        sim.h(3)
        back.h(3)
        assert sim.measure(3) == back.measure(3)  # same position in the stream
        assert back.seed.entropy == sim.seed.entropy


def test_load_rejects_foreign_files(tmp_path):
    import numpy as np
    import pytest

    import sparsegf2 as sg

    path = tmp_path / "x.npz"
    with path.open("wb") as fh:
        np.savez(fh, a=np.zeros(3))
    with pytest.raises(ValueError, match="not a saved Simulator"):
        sg.Simulator.load(path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_simulator.py -q -k "save_load or foreign"`
Expected: FAIL with `AttributeError: 'Simulator' object has no attribute 'save'`

- [ ] **Step 3: Implement (add after `copy`; add `import json`, `from pathlib import Path` and `from sparsegf2._version import __version__` at the top)**

```python
    _STATE_FORMAT = 1

    def save(self, path) -> Path:
        """Save the exact state (tableau, signs, random-stream position) to an ``.npz`` file.

        :meth:`load` restores it; the restored simulator continues the same
        random stream, so it behaves exactly as this one would.
        """
        path = Path(path)
        meta = {
            "format": self._STATE_FORMAT,
            "num_qubits": self._n,
            "phases": self._phases,
            "seed_entropy": str(self._seed.entropy),
            "seed_spawn_key": [int(k) for k in self._seed.spawn_key],
            "version": __version__,
        }
        with path.open("wb") as fh:
            np.savez(fh, tab=self._tab, rng=self._rng, meta=np.array(json.dumps(meta)))
        return path

    @classmethod
    def load(cls, path) -> Simulator:
        """A simulator restored from a file written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as z:
            if not {"tab", "rng", "meta"} <= set(z.files):
                raise ValueError(f"{Path(path).name} is not a saved Simulator (use Simulator.save)")
            meta = json.loads(str(z["meta"]))
            tab, rng = z["tab"], z["rng"]
        if meta.get("format") != cls._STATE_FORMAT:
            raise ValueError(f"unsupported state format {meta.get('format')!r}")
        n = int(meta["num_qubits"])
        phases = bool(meta["phases"])
        ws = (n + 63) // 64
        if tab.dtype != np.uint64 or tab.shape != (2 * n + 1, 2 * ws if phases else ws):
            raise ValueError("the saved tableau does not match its num_qubits / phases")
        sim = object.__new__(cls)
        sim._n, sim._phases, sim._ws = n, phases, ws
        sim._tab = tab.copy()
        sim._seed = np.random.SeedSequence(
            int(meta["seed_entropy"]), spawn_key=tuple(meta["seed_spawn_key"])
        )
        sim._rng = rng.astype(np.uint64).copy()
        sim._alloc_scratch()
        return sim
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_simulator.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/simulator.py tests/test_simulator.py
git commit -m "simulator: save/load the exact state, including the random-stream position"
```

---

### Task 6: Replay and the randomness contract

**Files:**
- Modify: `src/sparsegf2/mipt.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Produces:
  - `mipt.RANDOMNESS_CONTRACT: int = 1`.
  - `MonitoredCircuit.replay(trajectory: int, *, seed, probes=None, checkpoints=None) -> EnsembleResult`, a one-row result whose row is trajectory `trajectory` of `run(seed=seed)`.

- [ ] **Step 1: Write the failing tests**

```python
"""Replay: probes and checkpoints never change a trajectory (the randomness contract)."""

import numpy as np
import pytest

from sparsegf2 import graphs, mipt


def _spec(**kw):
    n = 16
    base = dict(
        num_system=n, edges=[graphs.watts_strogatz(n, 4, 0.2, seed=(0, g)) for g in range(3)],
        circuits_per_graph=2, layers=24, measurement_rate=0.15, schedule="random_edges",
        reference="purification", checkpoints="final", probes=[mipt.code_snapshot()],
    )
    base.update(kw)
    return mipt.MonitoredCircuit(**base)


def test_contract_constant():
    assert mipt.RANDOMNESS_CONTRACT == 1


@pytest.mark.parametrize("t", [0, 3, 5])
def test_replay_with_new_probes_reproduces_the_code_bit_for_bit(t):
    spec = _spec()
    full = spec.run(seed=9)
    again = spec.replay(
        t, seed=9, checkpoints=8,
        probes=[mipt.code_dimension(), mipt.erasure_first_failure(20, checkpoints=[8, 16]),
                mipt.code_snapshot(checkpoints="final")],
    )
    assert again.num_trajectories == 1 and int(again.trajectories[0]) == t
    np.testing.assert_array_equal(
        full.code(t, 0).stabilizer_matrix(), again.code(0, 0).stabilizer_matrix()
    )
    assert again.values["code_dimension"][0, -1] == full.code(t, 0).num_logical


def test_replay_validates_the_index():
    with pytest.raises(ValueError):
        _spec().replay(6, seed=1)  # 3 graphs x 2 circuits = trajectories 0..5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_replay.py -q`
Expected: FAIL with `AttributeError: module 'sparsegf2.mipt' has no attribute 'RANDOMNESS_CONTRACT'`

- [ ] **Step 3: Implement**

In `src/sparsegf2/mipt.py`, add `"RANDOMNESS_CONTRACT"` to `__all__` and, below `_H_INDEX`/`_CX_INDEX`:

```python
#: Version of the engine's random streams. Trajectory t of run(seed=s) is a
#: function of (protocol, s, t) only; bump this whenever a change to the
#: engine changes which circuit or which random numbers a trajectory gets, so
#: saved trajectory records refuse to replay across the change.
RANDOMNESS_CONTRACT = 1
```

Add to `MonitoredCircuit` (after `run`):

```python
    def replay(self, trajectory: int, *, seed, probes=None, checkpoints=None) -> EnsembleResult:
        """Trajectory ``trajectory`` of ``run(seed=seed)`` again, with other probes or checkpoints.

        Probes and checkpoints never change a trajectory's circuit or random
        numbers, so this is exactly that trajectory, observed differently: a
        one-row :class:`EnsembleResult`. This is how a trajectory is probed at
        new depths after the fact.
        """
        trajectory = _integer(trajectory, "trajectory")
        self._check_trajectories(trajectory, 1)
        spec = replace(
            self,
            probes=self.probes if probes is None else tuple(probes),
            checkpoints=self.checkpoints if checkpoints is None else checkpoints,
        )
        return spec.run(1, seed=seed, first_trajectory=trajectory)
```

(`_check_trajectories(first, count)` is the existing validator used by `circuit()`; it raises `ValueError` for an index beyond `num_trajectories`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_replay.py tests/test_mipt.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/mipt.py tests/test_replay.py
git commit -m "mipt: MonitoredCircuit.replay and RANDOMNESS_CONTRACT, pinned by a test"
```

---

### Task 7: `sparsegf2.crosscheck`, comparisons with Stim

**Files:**
- Create: `src/sparsegf2/crosscheck.py`
- Test: `tests/test_crosscheck.py`

**Interfaces:**
- Consumes: `Circuit.flattened()`, `Circuit.to_stim()`, `Circuit.num_qubits`, `Simulator.run(circuit, coins=..., observe=...)`, `Simulator.stabilizers(canonical=True)`, `RunResult.measurements/detectors/observables`.
- Produces:
  - `crosscheck.compare_stabilizers(circuit, *, num_qubits=None, phases=True, ticks=None, seed=0) -> dict`. Keys: `match` (bool), `ticks_compared` (int), `first_mismatch` (`None` or `{"tick", "sparsegf2", "stim"}`).
  - `crosscheck.compare_measurements(circuit, *, num_qubits=None, seed=0) -> dict`. Keys: `match`, `measurements`, `detectors`, `observables` (counts compared), `first_mismatch` (`None` or `{"kind", "index", "sparsegf2", "stim"}`).
  - Both raise `ValueError` for circuits with noise channels, and `ImportError` (message: `pip install stim`) when Stim is missing.

- [ ] **Step 1: Write the failing tests**

```python
"""crosscheck: SparseGF2 against Stim on one circuit."""

import pytest

import sparsegf2 as sg
from sparsegf2 import crosscheck

GHZ_TICKS = "H 0\nTICK\nCX 0 1\nTICK\nCX 1 2\nTICK\nM 0\nTICK\nH 1\nTICK"


@pytest.mark.parametrize("phases", [True, False])
def test_stabilizers_match_at_every_tick(phases):
    out = crosscheck.compare_stabilizers(sg.Circuit(GHZ_TICKS), phases=phases, seed=4)
    assert out == {"match": True, "ticks_compared": 5, "first_mismatch": None}


def test_selected_ticks():
    out = crosscheck.compare_stabilizers(sg.Circuit(GHZ_TICKS), ticks=[1, 3], seed=1)
    assert out["ticks_compared"] == 2 and out["match"]


def test_measurements_detectors_and_observables_match():
    text = "H 0\nCX 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\nMX 0"
    out = crosscheck.compare_measurements(sg.Circuit(text), seed=7)
    assert out["match"] and out["measurements"] == 3 and out["detectors"] == 1


def test_a_difference_is_caught(monkeypatch):
    real = crosscheck._sparsegf2_stabilizers

    def broken(circuit, num_qubits, phases, coins):
        rows = real(circuit, num_qubits, phases, coins)
        rows[-1] = ["+" + "X" * (len(rows[-1][0]) - 1)] + rows[-1][1:]
        return rows

    monkeypatch.setattr(crosscheck, "_sparsegf2_stabilizers", broken)
    out = crosscheck.compare_stabilizers(sg.Circuit(GHZ_TICKS), seed=4)
    assert not out["match"] and out["first_mismatch"]["tick"] == 4


def test_noise_is_refused():
    with pytest.raises(ValueError, match="noise-free"):
        crosscheck.compare_stabilizers(sg.Circuit("H 0\nX_ERROR(0.1) 0\nTICK"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_crosscheck.py -q`
Expected: FAIL with `ImportError: cannot import name 'crosscheck'`

- [ ] **Step 3: Implement `src/sparsegf2/crosscheck.py`**

```python
"""Compare SparseGF2 with Stim on one circuit.

Stim is imported lazily (it is a test dependency, not a runtime one). Random
measurement outcomes are taken from Stim and replayed in SparseGF2 with
``coins``, so with phases the comparison is sign for sign. Circuits with noise
channels are refused: the two simulators draw noise independently.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sparsegf2.circuit import Circuit
from sparsegf2.simulator import Simulator

__all__ = ["compare_measurements", "compare_stabilizers"]


def _stim():
    try:
        import stim
    except ImportError as exc:  # pragma: no cover - exercised only without Stim
        raise ImportError("the Stim cross-check needs Stim: pip install stim") from exc
    return stim


def _check(circuit: Circuit) -> None:
    for ins in circuit.flattened():
        if ins.info.kind == "noise":
            raise ValueError(
                f"the cross-check needs a noise-free circuit; {ins.name} draws noise, which "
                "Stim and SparseGF2 sample independently"
            )


def _stim_run(circuit: Circuit, n: int, seed: int) -> tuple[list[list[str]], np.ndarray]:
    """Stim's canonical stabilizers after every TICK, and its measurement record."""
    stim = _stim()
    sim = stim.TableauSimulator(seed=seed)
    sim.set_num_qubits(n)
    per_tick: list[list[str]] = []
    for ins in circuit.to_stim().flattened():
        if ins.name == "TICK":
            per_tick.append([str(p) for p in sim.canonical_stabilizers()])
        else:
            sim.do(ins)
    return per_tick, np.array(sim.current_measurement_record(), dtype=np.uint8)


def _sparsegf2_stabilizers(
    circuit: Circuit, num_qubits: int, phases: bool, coins: np.ndarray
) -> list[list[str]]:
    sim = Simulator(num_qubits, phases=phases, seed=0)
    result = sim.run(
        circuit,
        coins=coins,
        observe=lambda s, tick: [str(p) for p in s.stabilizers(canonical=True)],
    )
    return [value for _, value in result.observations]


def _unsigned(rows: list[str]) -> list[str]:
    return [r[1:] for r in rows]


def compare_stabilizers(
    circuit: Circuit, *, num_qubits: int | None = None, phases: bool = True, ticks=None, seed: int = 0
) -> dict[str, Any]:
    """Compare the canonical stabilizer groups after every ``TICK`` (or the given ticks).

    Signs are compared when ``phases`` (Stim's random outcomes are replayed);
    without phases only the Pauli parts are.
    """
    _check(circuit)
    n = max(circuit.num_qubits, num_qubits or 0, 1)
    stim_rows, record = _stim_run(circuit, n, seed)
    ours = _sparsegf2_stabilizers(circuit, n, phases, record)
    wanted = range(len(stim_rows)) if ticks is None else [int(t) for t in ticks]
    compared = 0
    for t in wanted:
        if not 0 <= t < len(stim_rows):
            raise ValueError(f"tick {t} does not exist; the circuit has {len(stim_rows)} ticks")
        a, b = ours[t], stim_rows[t]
        if not phases:
            a, b = _unsigned(a), _unsigned(b)
        compared += 1
        if a != b:
            return {
                "match": False,
                "ticks_compared": compared,
                "first_mismatch": {"tick": t, "sparsegf2": a, "stim": b},
            }
    return {"match": True, "ticks_compared": compared, "first_mismatch": None}


def compare_measurements(
    circuit: Circuit, *, num_qubits: int | None = None, seed: int = 0
) -> dict[str, Any]:
    """Replay Stim's random outcomes and compare every measurement, detector and observable."""
    _check(circuit)
    stim = _stim()
    n = max(circuit.num_qubits, num_qubits or 0, 1)
    _, record = _stim_run(circuit, n, seed)
    result = Simulator(n, phases=True, seed=0).run(circuit, coins=record)
    ours = {
        "measurements": np.asarray(result.measurements, dtype=np.uint8),
        "detectors": np.asarray(result.detectors, dtype=np.uint8),
        "observables": np.asarray(result.observables, dtype=np.uint8),
    }
    sc = circuit.to_stim()
    events = sc.compile_m2d_converter().convert(
        measurements=record.reshape(1, -1).astype(bool), append_observables=True
    )[0]
    nd = sc.num_detectors
    theirs = {
        "measurements": record,
        "detectors": events[:nd].astype(np.uint8),
        "observables": events[nd:].astype(np.uint8),
    }
    out: dict[str, Any] = {key: int(theirs[key].size) for key in theirs}
    for key in ("measurements", "detectors", "observables"):
        diff = np.flatnonzero(ours[key] != theirs[key])
        if diff.size:
            i = int(diff[0])
            return {
                **out,
                "match": False,
                "first_mismatch": {"kind": key, "index": i,
                                   "sparsegf2": int(ours[key][i]), "stim": int(theirs[key][i])},
            }
    return {**out, "match": True, "first_mismatch": None}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_crosscheck.py -q`
Expected: PASS. If `ins.info` does not exist on `Instruction` (check `Instruction.info` in `circuit.py`: it is a property returning `GateInfo`), use `gate(ins.name).kind` instead.

- [ ] **Step 5: Commit, and run the whole suite (end of Phase 1)**

```bash
ruff check src tests && ruff format src tests
python -m pytest -q
git add src/sparsegf2/crosscheck.py tests/test_crosscheck.py
git commit -m "crosscheck: compare stabilizers per tick and measurement records with Stim"
```

---
## Phase 2: The agent framework

The new server is built beside the old one. `sparsegf2-mcp` keeps serving the old 10 tools until Task 31 switches it over, so `main` stays shippable throughout. Tests build the new server with `sparsegf2.agent.server.build(...)` and talk to it through `mcp.Client(server)` in process.

### Task 8: The call grammar

**Files:**
- Create: `src/sparsegf2/agent/__init__.py`, `src/sparsegf2/agent/grammar.py`
- Test: `tests/agent/__init__.py` (empty), `tests/agent/test_grammar.py`

**Interfaces:**
- Produces (in `sparsegf2.agent.grammar`):
  - `GrammarError(ValueError)`.
  - `Range(start: int, stop: int, step: int = 1)` (frozen; `str()` gives `"a:b"` or `"a:b:s"`).
  - `Call(name: str, args: tuple, kwargs: tuple[tuple[str, Any], ...], at: Any, text: str)`.
  - `parse_call(text, *, allow_at=False, what="call") -> Call`.
  - `parse_calls(items, *, allow_at=False, what="calls") -> list[Call]`.
  - `bind(call, names, *, what, keyword_only=()) -> dict[str, Any]`.
  - `parse_checkpoints(value, *, what="checkpoints") -> str | int | list[int]`.
  - `parse_qubits(value, *, what) -> str | list[int]`.
  - `parse_grid(text) -> dict[str, list]`.
  - `plain(value)`: `Range` to `str` and lists recursively, for the JSON API.
- Produces (in `sparsegf2.agent`): `INTERFACE_VERSION = 1`.

- [ ] **Step 1: Write the failing tests**

`tests/agent/test_grammar.py`:

```python
"""The call grammar: every construct, every error with a column, and fuzzing."""

import random
import string

import pytest

from sparsegf2.agent.grammar import (
    Call, GrammarError, Range, bind, parse_call, parse_calls, parse_checkpoints, parse_grid,
    parse_qubits, plain,
)


def test_bare_name():
    c = parse_call("code_dimension")
    assert (c.name, c.args, c.kwargs, c.at) == ("code_dimension", (), (), None)


def test_positional_keyword_and_checkpoints():
    c = parse_call("mutual_information(0:4, b=[8, 9, 10])@every_layer", allow_at=True)
    assert c.args == (Range(0, 4),)
    assert c.kwargs == (("b", [8, 9, 10]),)
    assert c.at == "every_layer"


@pytest.mark.parametrize(
    "text, at",
    [("p@final", "final"), ("p@16", 16), ("p@[0, 64, 128]", [0, 64, 128])],
)
def test_checkpoint_forms(text, at):
    assert parse_call(text, allow_at=True).at == at


def test_values():
    c = parse_call('f(1, -2, 0.5, 1e-3, true, False, none, Z0*Z1, !X3, +XZ_, "a b", 0:16:2, [])')
    assert c.args == (1, -2, 0.5, 1e-3, True, False, None, "Z0*Z1", "!X3", "+XZ_", "a b",
                      Range(0, 16, 2), [])


def test_bind_maps_positionals_in_order_and_checks_names():
    c = parse_call("erasure_first_failure(512, name=ff)")
    assert bind(c, ["samples"], what="probes[0]", keyword_only=["name"]) == {
        "samples": 512, "name": "ff"}
    with pytest.raises(GrammarError, match="unknown argument 'regon'.*takes \\(region\\)"):
        bind(parse_call("entropy(regon=0:8)"), ["region"], what="probes[1]")
    with pytest.raises(GrammarError, match="at most 1 positional"):
        bind(parse_call("entropy(0:4, 4:8)"), ["region"], what="x")
    with pytest.raises(GrammarError, match="given twice"):
        bind(parse_call("entropy(0:4, region=4:8)"), ["region"], what="x")


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "expected a name"),
        ("entropy(0:8", "expected ',' or '\\)'"),
        ("entropy(0:8,)", "expected a value"),
        ("f(a=1, 2)", "positional argument after a keyword"),
        ("p@", "expected final, every_layer"),
        ("p@0.5", "expected final, every_layer"),
        ("p) x", "unexpected text"),
        ("f(0:4:0)", "step must not be 0"),
        ("β(1)", "unexpected character"),
        ("f([1, 2)", "expected ',' or '\\]'"),
    ],
)
def test_errors_carry_a_column(text, message):
    with pytest.raises(GrammarError, match=message) as info:
        parse_call(text, allow_at=True)
    assert "column" in str(info.value)


def test_at_is_refused_where_not_allowed():
    with pytest.raises(GrammarError, match="not allowed here"):
        parse_call("entropy(0:4)@final")


def test_parse_calls_names_the_item():
    with pytest.raises(GrammarError, match=r"probes\[1\]"):
        parse_calls(["code_dimension", "entropy("], allow_at=True, what="probes")
    with pytest.raises(GrammarError, match="list of strings"):
        parse_calls("code_dimension", what="probes")


@pytest.mark.parametrize(
    "value, expected",
    [("final", "final"), ("every_layer", "every_layer"), ("16", 16), (16, 16),
     ("[0, 8, 16]", [0, 8, 16]), ("0, 8, 16", [0, 8, 16])],
)
def test_parse_checkpoints(value, expected):
    assert parse_checkpoints(value) == expected


def test_parse_qubits():
    assert parse_qubits("0:8", what="q") == "0:8"
    assert parse_qubits("[3, 1]", what="q") == [3, 1]
    assert parse_qubits([3, 1], what="q") == [3, 1]
    assert parse_qubits(5, what="q") == [5]
    with pytest.raises(GrammarError):
        parse_qubits("a:b", what="q")


def test_parse_grid():
    assert parse_grid("measurement_rate=[0.05, 0.1]; num_system=[64,128];") == {
        "measurement_rate": [0.05, 0.1], "num_system": [64, 128]}
    assert parse_grid("graph_beta=0.3") == {"graph_beta": [0.3]}
    with pytest.raises(GrammarError, match="given twice"):
        parse_grid("a=[1]; a=[2]")
    with pytest.raises(GrammarError, match="empty"):
        parse_grid("a=[]")


def test_plain():
    assert plain([Range(0, 4), {"x": 1}, 2]) == ["0:4", {"x": 1}, 2]


def test_fuzz_only_grammar_errors():
    rng = random.Random(0)
    alphabet = string.printable + "βé€()[]@=,:;\"\\"
    for _ in range(3000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        try:
            parse_call(text, allow_at=True)
        except GrammarError:
            pass
    with pytest.raises(GrammarError, match="longer than"):
        parse_call("f(" + "1," * 3000 + "1)")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_grammar.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'sparsegf2.agent'`

- [ ] **Step 3: Implement**

`src/sparsegf2/agent/__init__.py`:

```python
"""The agent interface: the tool registry, stacks and MCP server behind ``sparsegf2-mcp``.

See docs/specs/2026-10-13-agent-tool-stacks-design.md. ``INTERFACE_VERSION``
is bumped whenever a tool or parameter is renamed or removed.
"""

INTERFACE_VERSION = 1
```

`src/sparsegf2/agent/grammar.py`:

```python
"""The call grammar of the agent tools.

Structured arguments are strings that read like Python calls, e.g.
``entropy(0:8)``, ``mutual_information(a=0:4, b=8:12)@every_layer`` or
``erasure_first_failure(samples=512)@[64,128]``. They are parsed here and
never evaluated. Ranges ``a:b[:s]`` use Python slice semantics.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Call", "GrammarError", "Range", "bind", "parse_call", "parse_calls", "parse_checkpoints",
    "parse_grid", "parse_qubits", "plain",
]

MAX_TEXT = 4000

_TOKENS = re.compile(
    r"""
      (?P<ws>\s+)
    | (?P<string>"(?:[^"\\]|\\.)*")
    | (?P<range>-?\d+:-?\d+(?::-?\d+)?(?![\w.:]))
    | (?P<number>-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w*!.:]))
    | (?P<word>[A-Za-z_!+\-][\w*!+\-]*)
    | (?P<punct>[()\[\],=@;])
    """,
    re.VERBOSE,
)
_NAME = re.compile(r"[A-Za-z_]\w*")


class GrammarError(ValueError):
    """A string that does not parse; the message says where and why."""


@dataclass(frozen=True)
class Range:
    start: int
    stop: int
    step: int = 1

    def __str__(self) -> str:
        return f"{self.start}:{self.stop}" + (f":{self.step}" if self.step != 1 else "")


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple = ()
    kwargs: tuple = ()
    at: Any = None
    text: str = ""


def _tokens(text: str, what: str) -> list[tuple[str, str, int]]:
    if len(text) > MAX_TEXT:
        raise GrammarError(f"{what}: longer than {MAX_TEXT} characters")
    out, pos = [], 0
    while pos < len(text):
        m = _TOKENS.match(text, pos)
        if m is None:
            raise GrammarError(
                f'{what} "{text}": unexpected character {text[pos]!r} at column {pos + 1}'
            )
        if m.lastgroup != "ws":
            out.append((m.lastgroup, m.group(), pos + 1))
        pos = m.end()
    out.append(("end", "", len(text) + 1))
    return out


class _Parser:
    def __init__(self, text: str, what: str):
        self.text, self.what = text, what
        self.toks = _tokens(text, what)
        self.i = 0

    def peek(self, ahead: int = 0) -> tuple[str, str, int]:
        return self.toks[min(self.i + ahead, len(self.toks) - 1)]

    def take(self) -> tuple[str, str, int]:
        tok = self.peek()
        self.i = min(self.i + 1, len(self.toks) - 1)
        return tok

    def fail(self, message: str, tok: tuple[str, str, int] | None = None) -> None:
        tok = tok or self.peek()
        found = f", found {tok[1]!r}" if tok[1] else ", found the end"
        raise GrammarError(f'{self.what} "{self.text}": {message}{found} at column {tok[2]}')

    def at_end(self) -> bool:
        return self.peek()[0] == "end"

    def value(self) -> Any:
        kind, text, _ = self.peek()
        if kind == "number":
            self.take()
            return float(text) if any(c in text for c in ".eE") else int(text)
        if kind == "range":
            parts = [int(p) for p in text.split(":")]
            if len(parts) == 3 and parts[2] == 0:
                self.fail("range step must not be 0")
            self.take()
            return Range(*parts)
        if kind == "string":
            self.take()
            return text[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        if kind == "word":
            self.take()
            low = text.lower()
            if low in ("true", "false"):
                return low == "true"
            if low in ("none", "null"):
                return None
            return text
        if (kind, text) == ("punct", "["):
            self.take()
            items: list[Any] = []
            if self.peek()[1] == "]":
                self.take()
                return items
            while True:
                items.append(self.value())
                sep = self.peek()
                if sep[1] == "]":
                    self.take()
                    return items
                if sep[1] != ",":
                    self.fail("expected ',' or ']'")
                self.take()
        self.fail("expected a value")
        return None  # unreachable

    def checkpoints(self) -> Any:
        kind, text, _ = self.peek()
        if kind == "word" and text in ("final", "every_layer"):
            self.take()
            return text
        if kind == "number" and text.lstrip("-").isdigit():
            self.take()
            return int(text)
        if (kind, text) == ("punct", "["):
            value = self.value()
            if all(isinstance(v, int) and not isinstance(v, bool) for v in value):
                return value
        self.fail("expected final, every_layer, a stride or a list of layers after '@'")
        return None  # unreachable

    def call(self, allow_at: bool) -> Call:
        kind, name, _ = self.peek()
        if kind != "word" or not _NAME.fullmatch(name):
            self.fail("expected a name")
        self.take()
        args: list[Any] = []
        kwargs: list[tuple[str, Any]] = []
        if self.peek()[1] == "(":
            self.take()
            if self.peek()[1] == ")":
                self.take()
            else:
                while True:
                    tok = self.peek()
                    if tok[0] == "word" and self.peek(1)[1] == "=":
                        self.take()
                        self.take()
                        if any(key == tok[1] for key, _ in kwargs):
                            self.fail(f"argument {tok[1]!r} given twice", tok)
                        kwargs.append((tok[1], self.value()))
                    else:
                        if kwargs:
                            self.fail("positional argument after a keyword argument")
                        args.append(self.value())
                    sep = self.peek()
                    if sep[1] == ")":
                        self.take()
                        break
                    if sep[1] != ",":
                        self.fail("expected ',' or ')'")
                    self.take()
        at = None
        if self.peek()[1] == "@":
            if not allow_at:
                self.fail("'@checkpoints' is not allowed here")
            self.take()
            at = self.checkpoints()
        return Call(name, tuple(args), tuple(kwargs), at, self.text)


def parse_call(text: Any, *, allow_at: bool = False, what: str = "call") -> Call:
    """Parse one ``name(args)[@checkpoints]`` string."""
    if not isinstance(text, str):
        raise GrammarError(f"{what}: expected a string like 'entropy(0:8)', got {text!r}")
    p = _Parser(text, what)
    call = p.call(allow_at)
    if not p.at_end():
        p.fail("unexpected text after the call")
    return call


def parse_calls(items: Any, *, allow_at: bool = False, what: str = "calls") -> list[Call]:
    """Parse a list of call strings; errors name the item, e.g. ``probes[2]``."""
    if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
        raise GrammarError(f"{what} must be a list of strings like ['code_dimension']")
    return [parse_call(s, allow_at=allow_at, what=f"{what}[{i}]") for i, s in enumerate(items)]


def bind(
    call: Call, names: Sequence[str], *, what: str, keyword_only: Iterable[str] = ()
) -> dict[str, Any]:
    """Map a call's arguments onto parameter ``names`` (positional ones in order)."""
    allowed = [*names, *keyword_only]
    if len(call.args) > len(names):
        raise GrammarError(
            f'{what} "{call.text}": {call.name} takes at most {len(names)} positional '
            f"argument(s) ({', '.join(names) or 'none'})"
        )
    out = dict(zip(names, call.args, strict=False))
    for key, value in call.kwargs:
        if key not in allowed:
            raise GrammarError(
                f'{what} "{call.text}": unknown argument {key!r}; {call.name} takes '
                f"({', '.join(allowed)})"
            )
        if key in out:
            raise GrammarError(f'{what} "{call.text}": argument {key!r} given twice')
        out[key] = value
    return out


def parse_checkpoints(value: Any, *, what: str = "checkpoints") -> str | int | list[int]:
    """``final``, ``every_layer``, a stride, or a list of layers (``[0,64]`` or ``0, 64``)."""
    if isinstance(value, bool):
        raise GrammarError(f"{what}: expected final, every_layer, a stride or a list, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        return value
    if not isinstance(value, str):
        raise GrammarError(f"{what}: expected final, every_layer, a stride or a list, got {value!r}")
    text = value.strip()
    if "," in text and not text.startswith("["):
        text = f"[{text}]"
    p = _Parser(text, what)
    out = p.checkpoints()
    if not p.at_end():
        p.fail("unexpected text after the checkpoints")
    return out


def parse_qubits(value: Any, *, what: str) -> str | list[int]:
    """A qubit list for the JSON API: a range string ``"a:b[:s]"`` or a list of ints."""
    if isinstance(value, Range):
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        return value
    if isinstance(value, str):
        p = _Parser(value.strip(), what)
        out = p.value()
        if not p.at_end():
            p.fail("unexpected text after the qubits")
        if isinstance(out, (Range, list, int)) and not isinstance(out, bool):
            return parse_qubits(out, what=what)
    raise GrammarError(f"{what}: expected qubits like 0:8 or [0, 3, 5], got {value!r}")


def parse_grid(text: Any) -> dict[str, list]:
    """``name=[v, ...]; name=v; ...``: a parameter grid (every combination is a point)."""
    if not isinstance(text, str):
        raise GrammarError(f"grid: expected a string like 'measurement_rate=[0.1, 0.2]', got {text!r}")
    p = _Parser(text, "grid")
    out: dict[str, list] = {}
    while True:
        tok = p.peek()
        if tok[0] != "word" or not _NAME.fullmatch(tok[1]):
            p.fail("expected a parameter name")
        p.take()
        if p.peek()[1] != "=":
            p.fail("expected '='")
        p.take()
        value = p.value()
        values = value if isinstance(value, list) else [value]
        if not values:
            p.fail(f"empty list of values for {tok[1]}", tok)
        if tok[1] in out:
            p.fail(f"{tok[1]} given twice", tok)
        out[tok[1]] = values
        if p.at_end():
            return out
        if p.peek()[1] != ";":
            p.fail("expected ';' between parameters")
        p.take()
        if p.at_end():
            return out


def plain(value: Any) -> Any:
    """Grammar values as plain JSON for the JSON API (``Range`` becomes ``"a:b[:s]"``)."""
    if isinstance(value, Range):
        return str(value)
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_grammar.py -q`
Expected: PASS. If the fuzz test finds an exception other than `GrammarError`, fix the parser path it exercises (the fix belongs in `_Parser`), never the test.

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent/__init__.py src/sparsegf2/agent/grammar.py tests/agent
git commit -m "agent: the call grammar (parse, bind, checkpoints, qubits, grids), fuzzed"
```

---

### Task 9: The tool registry

**Files:**
- Create: `src/sparsegf2/agent/registry.py`
- Test: `tests/agent/test_registry.py`

**Interfaces:**
- Produces (in `sparsegf2.agent.registry`):
  - `MISSING`, a sentinel.
  - `Param(name, type, description, default=MISSING, required=False, enum=None, minimum=None, maximum=None, items=None)`, with `.schema(default=MISSING) -> dict` and `.check(value, where) -> value`.
  - `BACKGROUND`, the `Param` added to every long-running tool.
  - `ToolDef(name, stack, summary, description, params, handler, render, examples=(), read_only=True, long_running=False, estimate=None, background=None, requires=())`, with:
    - `handler(ctx, args: dict) -> dict`;
    - `render(data: dict, ctx) -> str`;
    - `estimate(ctx, args) -> float` (seconds);
    - `background(ctx, args) -> dict` (job status data);
    - `.param(name)`;
    - `.available() -> str | None` (the reason when a required module is missing).
  - `Bound(tool, fixed={}, defaults={})`, with `.name`, `.input_schema()`, `.listing()`, `.size()` and `.validate(args) -> dict`.
  - `REGISTRY: dict[str, ToolDef]`, `register(tool) -> ToolDef`, `check_flat(schema) -> list[str]` (the violations).

- [ ] **Step 1: Write the failing tests**

```python
"""Param, ToolDef, Bound: schemas, validation, coercion, flatness."""

import json

import pytest

from sparsegf2.agent.registry import BACKGROUND, MISSING, Bound, Param, ToolDef, check_flat


def _tool(**kw):
    params = (
        Param("num_system", "integer", "System qubits.", required=True, minimum=2),
        Param("measurement_rate", "number", "Rate p.", default=0.1, minimum=0, maximum=1),
        Param("schedule", "string", "Gate placement.", default="matching",
              enum=("matching", "random_edges")),
        Param("probes", "array", "Probe calls.", items="string"),
        Param("phases", "boolean", "Track signs.", default=True),
    )
    base = dict(name="ensemble_run", stack="ensemble", summary="Run.", description="Run it.",
                params=params, handler=lambda ctx, a: a, render=lambda d, ctx: "ensemble_run")
    base.update(kw)
    return ToolDef(**base)


def test_schema_is_flat_and_marks_required():
    schema = Bound(_tool()).input_schema()
    assert schema["required"] == ["num_system"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["probes"] == {
        "type": "array", "description": "Probe calls.", "items": {"type": "string"}}
    assert schema["properties"]["schedule"]["enum"] == ["matching", "random_edges"]
    assert check_flat(schema) == []
    assert check_flat({"type": "object", "properties": {"x": {"anyOf": []}}}) != []


def test_long_running_tools_get_background():
    t = _tool(long_running=True)
    assert t.param("background") == BACKGROUND
    assert BACKGROUND.default == "auto"


@pytest.mark.parametrize(
    "given, expected",
    [({"num_system": "64"}, 64), ({"num_system": 64.0}, 64), ({"num_system": 64}, 64)],
)
def test_integers_are_coerced_when_unambiguous(given, expected):
    assert Bound(_tool()).validate(given)["num_system"] == expected


def test_numbers_and_booleans_from_text():
    out = Bound(_tool()).validate(
        {"num_system": 8, "measurement_rate": "0.25", "phases": "false"})
    assert out["measurement_rate"] == 0.25 and out["phases"] is False


@pytest.mark.parametrize(
    "given, message",
    [
        ({"num_system": "sixty"}, "num_system must be an integer"),
        ({"num_system": 64.5}, "num_system must be an integer"),
        ({"num_system": True}, "num_system must be a number"),
        ({"num_system": 1}, "num_system must be at least 2"),
        ({"num_system": 8, "measurement_rate": 2}, "at most 1"),
        ({"num_system": 8, "schedule": "brick"}, "must be one of"),
        ({"num_system": 8, "layrs": 3}, r"unknown argument\(s\) \['layrs'\]"),
        ({}, "missing required argument 'num_system'"),
    ],
)
def test_errors_name_the_parameter(given, message):
    with pytest.raises(ValueError, match=message):
        Bound(_tool()).validate(given)


def test_defaults_fill_and_none_means_omitted():
    out = Bound(_tool()).validate({"num_system": 8, "probes": None})
    assert out == {"num_system": 8, "measurement_rate": 0.1, "schedule": "matching",
                   "phases": True}


def test_single_value_for_an_array_becomes_a_list():
    assert Bound(_tool()).validate({"num_system": 8, "probes": "code_dimension"})["probes"] == [
        "code_dimension"]


def test_fixed_values_leave_the_schema_and_are_injected():
    b = Bound(_tool(), fixed={"num_system": 128}, defaults={"measurement_rate": 0.3})
    schema = b.input_schema()
    assert "num_system" not in schema["properties"] and "required" not in schema
    assert schema["properties"]["measurement_rate"]["default"] == 0.3
    assert b.validate({})["num_system"] == 128
    assert b.validate({"num_system": 128})["num_system"] == 128
    with pytest.raises(ValueError, match="fixed to 128 by this tool stack"):
        b.validate({"num_system": 64})


def test_size_counts_the_listing():
    b = Bound(_tool())
    assert b.size() == len(json.dumps(b.listing(), separators=(",", ":")))


def test_tool_names_carry_their_stack():
    with pytest.raises(ValueError, match="must start with 'ensemble_'"):
        _tool(name="run_ensemble")


def test_missing_sentinel_is_not_none():
    assert MISSING is not None and repr(MISSING) == "MISSING"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_registry.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'sparsegf2.agent.registry'`

- [ ] **Step 3: Implement `src/sparsegf2/agent/registry.py`**

```python
"""Tool definitions for the agent server: parameters, flat JSON schemas, validation.

Every tool is a :class:`ToolDef` registered in :data:`REGISTRY`. A
:class:`Bound` is a tool as a stack serves it, with some parameters fixed
(removed from the schema) and some defaults changed.
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["BACKGROUND", "MISSING", "REGISTRY", "Bound", "Param", "ToolDef", "check_flat",
           "register"]


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING: Any = _Missing()
TYPES = ("string", "integer", "number", "boolean", "array")
ITEM_TYPES = ("string", "integer", "number")


def _coerce(value: Any, kind: str, name: str) -> Any:
    if kind == "string":
        if isinstance(value, str):
            return value
        raise ValueError(f"{name} must be a string, got {value!r}")
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise ValueError(f"{name} must be true or false, got {value!r}")
    word = "an integer" if kind == "integer" else "a number"
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if isinstance(value, str):
        text = value.strip()
        try:
            value = float(text) if (kind == "number" or any(c in text for c in ".eE")) else int(text)
        except ValueError:
            raise ValueError(f"{name} must be {word}, got the text {text!r}") from None
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be {word}, got {value!r}")
    if kind == "integer":
        if float(value) != int(value):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(value)
    return float(value)


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    description: str
    default: Any = MISSING
    required: bool = False
    enum: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    items: str | None = None

    def __post_init__(self) -> None:
        if self.type not in TYPES:
            raise ValueError(f"param {self.name}: type must be one of {TYPES}")
        if (self.type == "array") != (self.items is not None):
            raise ValueError(f"param {self.name}: items is required for (and only for) arrays")
        if self.items is not None and self.items not in ITEM_TYPES:
            raise ValueError(f"param {self.name}: items must be one of {ITEM_TYPES}")
        if self.required and self.default is not MISSING:
            raise ValueError(f"param {self.name}: a required parameter has no default")

    def schema(self, default: Any = MISSING) -> dict[str, Any]:
        s: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.items is not None:
            s["items"] = {"type": self.items}
        if self.enum is not None:
            s["enum"] = list(self.enum)
        if self.minimum is not None:
            s["minimum"] = self.minimum
        if self.maximum is not None:
            s["maximum"] = self.maximum
        d = self.default if default is MISSING else default
        if d is not MISSING and d is not None:
            s["default"] = d
        return s

    def check(self, value: Any, where: str) -> Any:
        """``value`` converted to this parameter's type, or a ValueError naming the parameter."""
        name = f"{where}: {self.name}" if where else self.name
        if self.type == "array":
            if not isinstance(value, list):
                if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                    value = [value]
                else:
                    raise ValueError(f"{name} must be a list, got {value!r}")
            item = Param(self.name, self.items or "string", "", enum=self.enum,
                         minimum=self.minimum, maximum=self.maximum)
            return [item.check(v, where) for v in value]
        value = _coerce(value, self.type, name)
        if self.enum is not None and value not in self.enum:
            raise ValueError(f"{name} must be one of {list(self.enum)}, got {value!r}")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{name} must be at least {self.minimum:g}, got {value!r}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{name} must be at most {self.maximum:g}, got {value!r}")
        return value


BACKGROUND = Param(
    "background", "string",
    "auto: run in the background (returns a job_id) when the estimate exceeds the server's "
    "limit; always; never.",
    default="auto", enum=("auto", "always", "never"),
)


@dataclass(frozen=True)
class ToolDef:
    name: str
    stack: str
    summary: str
    description: str
    params: tuple[Param, ...]
    handler: Callable[[Any, dict], dict]
    render: Callable[[dict, Any], str]
    examples: tuple[dict, ...] = ()
    read_only: bool = True
    long_running: bool = False
    estimate: Callable[[Any, dict], float] | None = None
    background: Callable[[Any, dict], dict] | None = None
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.startswith(self.stack + "_"):
            raise ValueError(f"tool {self.name!r} must start with '{self.stack}_'")
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError(f"tool {self.name}: duplicate parameter names")
        if self.long_running and "background" not in names:
            object.__setattr__(self, "params", (*self.params, BACKGROUND))

    def param(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def available(self) -> str | None:
        """None when usable, else why not (a missing optional module)."""
        for module in self.requires:
            if importlib.util.find_spec(module) is None:
                return f"{self.name} needs the optional package {module!r}"
        return None


@dataclass(frozen=True)
class Bound:
    tool: ToolDef
    fixed: Mapping[str, Any] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.tool.name

    def input_schema(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        required = []
        for p in self.tool.params:
            if p.name in self.fixed:
                continue
            props[p.name] = p.schema(self.defaults.get(p.name, MISSING))
            if p.required and p.name not in self.defaults:
                required.append(p.name)
        schema: dict[str, Any] = {"type": "object", "properties": props,
                                  "additionalProperties": False}
        if required:
            schema["required"] = required
        return schema

    def listing(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.tool.description,
                "inputSchema": self.input_schema()}

    def size(self) -> int:
        return len(json.dumps(self.listing(), separators=(",", ":")))

    def validate(self, args: Mapping[str, Any] | None) -> dict[str, Any]:
        args = dict(args or {})
        where = self.name
        known = {p.name: p for p in self.tool.params}
        unknown = sorted(set(args) - set(known))
        if unknown:
            takes = sorted(set(known) - set(self.fixed))
            raise ValueError(f"{where}: unknown argument(s) {unknown}; it takes {takes}")
        out: dict[str, Any] = {}
        for name, p in known.items():
            given = args.get(name)
            if name in self.fixed:
                if given is not None and p.check(given, where) != self.fixed[name]:
                    raise ValueError(
                        f"{where}: {name} is fixed to {self.fixed[name]!r} by this tool stack"
                    )
                out[name] = self.fixed[name]
            elif given is not None:
                out[name] = p.check(given, where)
            elif name in self.defaults:
                out[name] = self.defaults[name]
            elif p.default is not MISSING and p.default is not None:
                out[name] = p.default
            elif p.required:
                raise ValueError(f"{where}: missing required argument {name!r} ({p.description})")
        return out


REGISTRY: dict[str, ToolDef] = {}


def register(tool: ToolDef) -> ToolDef:
    if tool.name in REGISTRY:
        raise ValueError(f"tool {tool.name!r} is already registered")
    REGISTRY[tool.name] = tool
    return tool


_FORBIDDEN = ("$ref", "$defs", "oneOf", "anyOf", "allOf", "not")


def check_flat(schema: Mapping[str, Any]) -> list[str]:
    """Violations of the flat-schema rule (spec 2.2); empty when the schema is flat."""
    problems = []
    for key in _FORBIDDEN:
        if key in schema:
            problems.append(f"top level uses {key}")
    for name, prop in schema.get("properties", {}).items():
        for key in _FORBIDDEN:
            if key in prop:
                problems.append(f"{name} uses {key}")
        kind = prop.get("type")
        if kind not in TYPES:
            problems.append(f"{name} has type {kind!r}")
        if kind == "array" and prop.get("items", {}).get("type") not in ITEM_TYPES:
            problems.append(f"{name} is an array of non-scalars")
    return problems
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_registry.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent/registry.py tests/agent/test_registry.py
git commit -m "agent: tool registry with flat schemas, lenient type coercion and stack overrides"
```

---

### Task 10: Context (root confinement) and rendering helpers

**Files:**
- Create: `src/sparsegf2/agent/context.py`, `src/sparsegf2/agent/render.py`
- Test: `tests/agent/test_context_render.py`

**Interfaces:**
- Consumes: `sparsegf2.api._jsonable`, which maps non-finite floats to `None`.
- Produces:
  - `Context(root, max_output_chars=4000, sync_seconds=60.0, max_states=16, quiet=False, stack="all", tool_names=())`, with:
    - attributes `.states` (a `StateStore`, Task 11) and `.jobs` (a `JobStore`, Task 11), created lazily;
    - `.path(value, *, what="path", must_exist=False) -> Path`;
    - `.show(path) -> str` (relative POSIX text);
    - `.output_dir(tool, args, *, key="output_dir") -> Path`.
  - In `render`:
    - `jsonable(obj)`;
    - `fmt(x, digits=4) -> str`;
    - `pm(mean, err) -> str` (`"m+-e"`);
    - `table(headers, rows) -> str`;
    - `cap(text, limit, *, full=None) -> str`;
    - `curve(values, *, start=0, max_items=17) -> str`;
    - `lines(*parts) -> str` (joins non-empty parts with newlines).

- [ ] **Step 1: Write the failing tests**

```python
"""Context paths (root confinement, Windows forms) and the rendering helpers."""

import json
import math
import os
import sys

import numpy as np
import pytest

from sparsegf2.agent.context import Context
from sparsegf2.agent.render import cap, curve, fmt, jsonable, pm, table


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "runs").mkdir()
    return Context(tmp_path)


def test_relative_and_absolute_paths_inside_the_root(ctx, tmp_path):
    assert ctx.path("runs/a") == (tmp_path / "runs" / "a").resolve()
    assert ctx.path(str(tmp_path / "runs")) == (tmp_path / "runs").resolve()
    assert ctx.path("runs/../runs/b") == (tmp_path / "runs" / "b").resolve()


def test_escapes_are_refused(ctx, tmp_path):
    with pytest.raises(ValueError, match="outside the root"):
        ctx.path("../elsewhere")
    with pytest.raises(ValueError, match="outside the root"):
        ctx.path(str(tmp_path.parent / "x"))
    with pytest.raises(ValueError, match="non-empty"):
        ctx.path("  ")


def test_must_exist(ctx):
    with pytest.raises(ValueError, match="does not exist"):
        ctx.path("nope.npz", must_exist=True)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path forms")
def test_windows_backslashes_and_drive_letter_case(ctx, tmp_path):
    assert ctx.path("runs\\c") == (tmp_path / "runs" / "c").resolve()
    other_case = str(tmp_path).upper() if str(tmp_path).islower() else str(tmp_path).lower()
    assert ctx.path(os.path.join(other_case, "runs")).name.lower() == "runs"


def test_show_is_relative_posix(ctx, tmp_path):
    assert ctx.show(tmp_path / "runs" / "x") == "runs/x"
    assert ctx.show(tmp_path) == "."


def test_default_output_dir_is_a_stable_hash(ctx, tmp_path):
    a = ctx.output_dir("ensemble_run", {"num_system": 8, "background": "auto"})
    b = ctx.output_dir("ensemble_run", {"num_system": 8, "background": "never"})
    c = ctx.output_dir("ensemble_run", {"num_system": 9})
    assert a == b != c and a.parent == (tmp_path / "runs").resolve()
    assert ctx.output_dir("x_y", {"output_dir": "mine"}) == (tmp_path / "mine").resolve()


def test_jsonable_never_emits_nan():
    data = jsonable({"a": np.array([1.0, np.nan, np.inf]), "b": math.nan, "c": (1, 2)})
    assert data == {"a": [1.0, None, None], "b": None, "c": [1, 2]}
    json.dumps(data, allow_nan=False)


def test_numbers():
    assert fmt(None) == "-" and fmt(math.nan) == "nan" and fmt(3) == "3"
    assert fmt(0.123456) == "0.1235" and fmt(True) == "true"
    assert pm(0.5, 0.01) == "0.5+-0.01" and pm(0.5, math.nan) == "0.5"


def test_table_aligns_columns():
    out = table(["layer", "k/n"], [[0, 1.0], [256, 0.0833]])
    assert out.splitlines() == ["layer     k/n", "    0       1", "  256  0.0833"]


def test_cap_cuts_at_a_line_and_says_where_the_rest_is():
    text = "\n".join(f"line {i}" for i in range(100))
    out = cap(text, 120, full="runs/x")
    assert len(out) <= 120 and out.endswith("(cut to 120 characters; full data in runs/x)")


def test_curve_subsamples():
    assert curve([0.0, 0.5, 1.0]) == "m=0..2: 0 0.5 1"
    long = curve(np.linspace(0, 1, 101))
    assert long.startswith("m=0..100 (every ") and len(long.split(": ")[1].split()) <= 17
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_context_render.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/sparsegf2/agent/render.py`**

```python
"""Compact text for tool results: numbers, +- errors, aligned tables, caps."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["cap", "curve", "fmt", "jsonable", "lines", "pm", "table"]


def jsonable(obj: Any) -> Any:
    """Plain JSON values (numpy converted, NaN and infinities as None)."""
    from sparsegf2.api import _jsonable

    return _jsonable(obj)


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "true" if x else "false"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return f"{x:.{digits}g}"
    return str(x)


def pm(mean: Any, err: Any) -> str:
    if err is None or (isinstance(err, float) and not math.isfinite(err)):
        return fmt(mean)
    return f"{fmt(mean)}+-{fmt(err, 2)}"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Aligned columns: the first header left-aligned, everything else right-aligned."""
    cells = [list(map(str, headers))] + [[fmt(v) for v in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    head = "  ".join(
        h.ljust(w) if i == 0 else h.rjust(w)
        for i, (h, w) in enumerate(zip(cells[0], widths, strict=True))
    )
    body = ["  ".join(c.rjust(w) for c, w in zip(r, widths, strict=True)) for r in cells[1:]]
    return "\n".join(line.rstrip() for line in [head, *body])


def cap(text: str, limit: int, *, full: str | None = None) -> str:
    if len(text) <= limit:
        return text
    where = full or "the saved files (pass output_dir, then result_get)"
    note = f"(cut to {limit} characters; full data in {where})"
    room = max(0, limit - len(note) - 1)
    cut = text.rfind("\n", 0, room)
    head = text[: cut if cut > 0 else room]
    return f"{head}\n{note}"


def curve(values: Sequence[float], *, start: int = 0, max_items: int = 17) -> str:
    v = np.asarray(values, dtype=float)
    n = v.size
    if n <= max_items:
        idx = np.arange(n)
        label = f"m={start}..{start + n - 1}"
    else:
        idx = np.unique(np.linspace(0, n - 1, max_items).round().astype(int))
        step = (n - 1) / (max_items - 1)
        label = f"m={start}..{start + n - 1} (every ~{step:.3g})"
    return f"{label}: " + " ".join(fmt(float(v[i]), 3) for i in idx)


def lines(*parts: str | None) -> str:
    return "\n".join(p for p in parts if p)
```

- [ ] **Step 4: Implement `src/sparsegf2/agent/context.py`**

```python
"""What every tool handler receives: the root folder, limits, open states and jobs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Context"]


@dataclass
class Context:
    root: Path
    max_output_chars: int = 4000
    sync_seconds: float = 60.0
    max_states: int = 16
    quiet: bool = False
    stack: str = "all"
    tool_names: tuple[str, ...] = ()
    _states: Any = field(default=None, repr=False)
    _jobs: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"root {str(root)!r} is not a directory")
        self.root = root

    @property
    def states(self):
        if self._states is None:
            from sparsegf2.agent.sessions import StateStore

            self._states = StateStore(self.max_states)
        return self._states

    @property
    def jobs(self):
        if self._jobs is None:
            from sparsegf2.agent.sessions import JobStore

            self._jobs = JobStore()
        return self._jobs

    def path(self, value: Any, *, what: str = "path", must_exist: bool = False) -> Path:
        """``value`` resolved under the root; anything outside it is refused."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{what} must be a non-empty path")
        p = Path(value.strip()).expanduser()
        p = (p if p.is_absolute() else self.root / p).resolve()
        if p != self.root and not p.is_relative_to(self.root):
            raise ValueError(
                f"{what} {value!r} is outside the root folder {str(self.root)!r}; give a path "
                "inside it (relative paths are relative to it)"
            )
        if must_exist and not p.exists():
            raise ValueError(f"{what} {value!r} does not exist (looked for {self.show(p)})")
        return p

    def show(self, path: Any) -> str:
        p = Path(path).resolve()
        try:
            rel = p.relative_to(self.root).as_posix()
        except ValueError:
            return str(p)
        return rel or "."

    def output_dir(self, tool: str, args: dict[str, Any], *, key: str = "output_dir") -> Path:
        """``args[key]`` under the root, or ``runs/<tool>-<hash of the arguments>``.

        The hash leaves out ``output_dir`` and ``background``, so the same request
        always maps to the same directory and running it again resumes it.
        """
        if args.get(key):
            return self.path(args[key], what=key)
        basis = {k: v for k, v in args.items() if k not in (key, "background")}
        text = json.dumps(basis, sort_keys=True, default=str)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
        return self.root / "runs" / f"{tool}-{digest}"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_context_render.py -q`
Expected: PASS (the Windows-only test runs on Windows CI and locally on Sam's machine)

- [ ] **Step 6: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent/context.py src/sparsegf2/agent/render.py tests/agent/test_context_render.py
git commit -m "agent: context with root confinement and stable output dirs; text rendering helpers"
```

---

### Task 11: Sessions: open states and generic background jobs

**Files:**
- Create: `src/sparsegf2/agent/sessions.py`
- Test: `tests/agent/test_sessions.py`

**Interfaces:**
- Consumes: `sparsegf2.api.job_status`, `api.cancel_job`, `api.list_jobs` (ensemble jobs), and `render.fmt`.
- Produces (in `sparsegf2.agent.sessions`):
  - `StateStore(max_states)`, with:
    - `.add(sim, origin) -> str` (ids `s1`, `s2`, ...);
    - `.get(state_id) -> Simulator`;
    - `.info(state_id) -> dict`;
    - `.count_measurements(state_id, k)`;
    - `.close(state_id) -> dict`;
    - `.list() -> list[dict]`.
  - `JobStore()`, with:
    - `.start(*, tool, key, run, render, output_dir=None) -> dict`, where `run(job) -> dict`, `render(data) -> str`, and the job's status dict comes back;
    - `.status(job_id, wait_seconds=0.0) -> dict`;
    - `.cancel(job_id) -> dict`;
    - `.list() -> list[dict]`.
  - `report(done, total, unit)` and `cancelled() -> bool`, to be called from inside a running job (thread-local current job).
  - `render_job(status) -> str`.
  - `from_api_status(s) -> dict`, which converts an ensemble job's status from `sparsegf2.api` into the format below.
  - The status dict has the keys `job_id`, `tool`, `status` (`queued|running|completed|cancelled|failed`), `progress` (`{"done", "total", "unit"}` or `None`), `elapsed_seconds`, `output_dir`, `error`, `result_text` (when completed), and `kind` (`"tool"` or `"ensemble"`).

- [ ] **Step 1: Write the failing tests**

```python
"""StateStore and JobStore."""

import threading
import time

import pytest

import sparsegf2 as sg
from sparsegf2.agent import sessions
from sparsegf2.agent.sessions import JobStore, StateStore, render_job


def test_state_store_ids_limits_and_listing():
    store = StateStore(max_states=2)
    a = store.add(sg.Simulator(2), "new")
    b = store.add(sg.Simulator(3, phases=False), "new")
    assert (a, b) == ("s1", "s2")
    with pytest.raises(ValueError, match="at most 2 open states"):
        store.add(sg.Simulator(1), "new")
    store.count_measurements(a, 3)
    assert store.info(a) == {"state_id": "s1", "num_qubits": 2, "phases": True,
                             "origin": "new", "measurements": 3}
    store.close(a)
    with pytest.raises(ValueError, match="unknown state_id 's1'.*open: \\['s2'\\]"):
        store.get(a)
    assert [s["state_id"] for s in store.list()] == ["s2"]


def test_job_runs_reports_progress_and_completes():
    jobs = JobStore()
    gate = threading.Event()

    def run(job):
        sessions.report(1, 2, "points")
        gate.wait(5)
        sessions.report(2, 2, "points")
        return {"answer": 42}

    status = jobs.start(tool="x_y", key="k", run=run, render=lambda d: f"x_y: {d['answer']}")
    assert status["status"] in ("queued", "running")
    time.sleep(0.05)
    assert jobs.status(status["job_id"])["progress"] == {"done": 1, "total": 2, "unit": "points"}
    gate.set()
    done = jobs.status(status["job_id"], wait_seconds=5)
    assert done["status"] == "completed" and done["result_text"] == "x_y: 42"


def test_same_key_while_running_returns_the_running_job():
    jobs = JobStore()
    gate = threading.Event()
    first = jobs.start(tool="x_y", key="same", run=lambda j: gate.wait(5) or {}, render=str)
    second = jobs.start(tool="x_y", key="same", run=lambda j: {}, render=str)
    assert second["job_id"] == first["job_id"] and "already running" in second["note"]
    gate.set()


def test_failure_and_cancel():
    jobs = JobStore()

    def boom(job):
        raise RuntimeError("bad")

    failed = jobs.status(jobs.start(tool="x_y", key="f", run=boom, render=str)["job_id"], 5)
    assert failed["status"] == "failed" and "RuntimeError: bad" in failed["error"]

    def loop(job):
        while not sessions.cancelled():
            time.sleep(0.01)
        return {"stopped": True}

    started = jobs.start(tool="x_y", key="c", run=loop, render=str)
    jobs.cancel(started["job_id"])
    assert jobs.status(started["job_id"], 5)["status"] == "cancelled"


def test_unknown_job_mentions_restarts():
    with pytest.raises(ValueError, match="unknown job_id"):
        JobStore().status("nope")


def test_render_job_has_next_step():
    text = render_job({"job_id": "abc", "tool": "x_y", "status": "running", "kind": "tool",
                       "progress": {"done": 1, "total": 4, "unit": "points"},
                       "elapsed_seconds": 2.0, "output_dir": "runs/a", "error": None})
    assert text.startswith("job abc (x_y): running, 1/4 points")
    assert "next: job_status(job_id=\"abc\", wait_seconds=30)" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_sessions.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/sparsegf2/agent/sessions.py`**

```python
"""Open stabilizer states and background jobs, held in the server process."""

from __future__ import annotations

import itertools
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["JobStore", "StateStore", "cancelled", "from_api_status", "render_job", "report"]


class StateStore:
    def __init__(self, max_states: int = 16):
        self.max_states = int(max_states)
        self._items: dict[str, dict[str, Any]] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def add(self, sim, origin: str) -> str:
        with self._lock:
            if len(self._items) >= self.max_states:
                raise ValueError(
                    f"at most {self.max_states} open states; close one with state_close "
                    f"(open: {sorted(self._items)})"
                )
            state_id = f"s{next(self._ids)}"
            self._items[state_id] = {"sim": sim, "origin": origin, "measurements": 0}
            return state_id

    def _entry(self, state_id: Any) -> dict[str, Any]:
        entry = self._items.get(state_id) if isinstance(state_id, str) else None
        if entry is None:
            raise ValueError(
                f"unknown state_id {state_id!r}; open: {sorted(self._items)} (states do not "
                "survive a server restart; save them with state_save)"
            )
        return entry

    def get(self, state_id: Any):
        return self._entry(state_id)["sim"]

    def count_measurements(self, state_id: str, k: int) -> None:
        self._entry(state_id)["measurements"] += int(k)

    def info(self, state_id: str) -> dict[str, Any]:
        e = self._entry(state_id)
        return {"state_id": state_id, "num_qubits": e["sim"].num_qubits,
                "phases": e["sim"].phases, "origin": e["origin"],
                "measurements": e["measurements"]}

    def close(self, state_id: str) -> dict[str, Any]:
        info = self.info(state_id)
        with self._lock:
            del self._items[state_id]
        return info

    def list(self) -> list[dict[str, Any]]:
        return [self.info(s) for s in sorted(self._items, key=lambda s: int(s[1:]))]


_local = threading.local()


def report(done: int, total: int, unit: str) -> None:
    """Progress of the job running in this thread (no-op outside a job)."""
    job = getattr(_local, "job", None)
    if job is not None:
        job.progress = {"done": int(done), "total": int(total), "unit": unit}


def cancelled() -> bool:
    """Whether the job running in this thread was asked to stop."""
    job = getattr(_local, "job", None)
    return bool(job is not None and job.cancel.is_set())


@dataclass
class _Job:
    job_id: str
    tool: str
    key: str
    output_dir: str | None
    status: str = "queued"
    started: float = field(default_factory=time.time)
    finished: float | None = None
    progress: dict[str, Any] | None = None
    error: str | None = None
    result_text: str | None = None
    data: dict[str, Any] | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id, "tool": self.tool, "kind": "tool", "status": self.status,
            "progress": self.progress,
            "elapsed_seconds": round((self.finished or time.time()) - self.started, 2),
            "output_dir": self.output_dir, "error": self.error,
            "result_text": self.result_text,
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def start(self, *, tool: str, key: str, run: Callable[[Any], dict],
              render: Callable[[dict], str], output_dir: str | None = None) -> dict[str, Any]:
        with self._lock:
            for job in self._jobs.values():
                if job.key == key and job.status in ("queued", "running"):
                    return {**job.snapshot(), "note": "already running; this is that job"}
            job = _Job(uuid.uuid4().hex[:12], tool, key, output_dir)
            self._jobs[job.job_id] = job
        job.thread = threading.Thread(
            target=self._run, args=(job, run, render), name=f"sparsegf2-{tool}-{job.job_id}",
            daemon=True,
        )
        job.thread.start()
        return job.snapshot()

    @staticmethod
    def _run(job: _Job, run: Callable[[Any], dict], render: Callable[[dict], str]) -> None:
        _local.job = job
        job.status = "running"
        try:
            data = run(job)
            job.data = data
            job.result_text = render(data)
            job.status = "cancelled" if job.cancel.is_set() else "completed"
        except Exception as exc:  # reported through job_status
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}\n" + "".join(
                traceback.format_exception(exc)[-3:])
        finally:
            job.finished = time.time()
            _local.job = None

    def _get(self, job_id: Any) -> _Job | None:
        with self._lock:
            return self._jobs.get(job_id) if isinstance(job_id, str) else None

    def status(self, job_id: Any, wait_seconds: float = 0.0) -> dict[str, Any]:
        job = self._get(job_id)
        if job is None:
            return self._ensemble("status", job_id, wait_seconds)
        if wait_seconds and job.thread is not None:
            job.thread.join(timeout=float(wait_seconds))
        return job.snapshot()

    def cancel(self, job_id: Any) -> dict[str, Any]:
        job = self._get(job_id)
        if job is None:
            return self._ensemble("cancel", job_id)
        job.cancel.set()
        return job.snapshot()

    def list(self) -> list[dict[str, Any]]:
        from sparsegf2 import api

        ours = [j.snapshot() for j in sorted(self._jobs.values(), key=lambda j: -j.started)]
        theirs = [{**row, "kind": "ensemble", "tool": "ensemble"} for row in api.list_jobs()["jobs"]]
        return ours + theirs

    def _ensemble(self, action: str, job_id: Any, wait_seconds: float = 0.0) -> dict[str, Any]:
        from sparsegf2 import api

        try:
            s = api.job_status(job_id, wait_seconds) if action == "status" else api.cancel_job(job_id)
        except ValueError:
            raise ValueError(
                f"unknown job_id {job_id!r}; jobs in this server: {sorted(self._jobs)} plus the "
                "ensemble jobs job_list shows (jobs do not survive a server restart; run the same "
                "request again to resume its saved chunks)"
            ) from None
        return from_api_status(s)


def from_api_status(s: dict[str, Any]) -> dict[str, Any]:
    """An ensemble job's status from ``sparsegf2.api`` in this module's status format."""
    return {
        "job_id": s["job_id"], "tool": "ensemble", "kind": "ensemble", "status": s["status"],
        "progress": {"done": s["trajectories_done"], "total": s["trajectories_total"],
                     "unit": "trajectories"},
        "elapsed_seconds": s.get("elapsed_seconds"), "eta_seconds": s.get("eta_seconds"),
        "output_dir": s.get("output_dir"), "error": s.get("error"), "result_text": None,
    }


def render_job(status: dict[str, Any]) -> str:
    from sparsegf2.agent.render import fmt

    p = status.get("progress")
    done = f", {p['done']}/{p['total']} {p['unit']}" if p else ""
    eta = f", eta {fmt(status['eta_seconds'])} s" if status.get("eta_seconds") else ""
    head = f"job {status['job_id']} ({status['tool']}): {status['status']}{done}{eta}"
    out = [head]
    if status.get("note"):
        out.append(status["note"])
    if status.get("output_dir"):
        out.append(f"output_dir: {status['output_dir']}")
    if status.get("error"):
        out.append(f"error: {status['error'].strip()}")
    if status["status"] in ("queued", "running"):
        out.append(f'next: job_status(job_id="{status["job_id"]}", wait_seconds=30)')
    elif status["status"] == "completed" and status.get("result_text"):
        out.append(status["result_text"])
    elif status["status"] == "completed" and status.get("kind") == "ensemble":
        out.append(f'next: analysis_summary(source="{status["output_dir"]}")')
    return "\n".join(out)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_sessions.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent/sessions.py tests/agent/test_sessions.py
git commit -m "agent: open-state store and generic background jobs with progress and dedup"
```

---

### Task 12: Vocabularies from the library, and the shared parameter groups

**Files:**
- Create: `src/sparsegf2/agent/vocab.py`, `src/sparsegf2/agent/params.py`
- Test: `tests/agent/test_params.py`

**Interfaces:**
- Consumes: `api._probe_kinds()`, `api._OBSERVABLES`, `api._graph_families()`, `api._weight_families()`, `api.SCHEDULES`, `api.REFERENCES`, `sparsegf2.all_gates()`, `dataclasses.fields(mipt.MonitoredCircuit)`, and from `grammar`: `parse_call`, `parse_calls`, `bind`, `parse_checkpoints`, `parse_qubits`, `plain`.
- Produces, in `vocab`:
  - `probe_signatures() -> dict[str, tuple[str, ...]]`;
  - `probe_descriptions() -> dict[str, str]`;
  - `observable_signatures() -> dict[str, tuple[str, ...]]`;
  - `observable_descriptions() -> dict[str, str]`;
  - `graph_families() -> dict[str, tuple[str, ...]]`, the parameters besides `n` and `seed`;
  - `random_graph_families() -> frozenset[str]`;
  - `weight_families() -> dict[str, tuple[str, ...]]`;
  - `protocol_fields() -> tuple[str, ...]`;
  - `gate_names() -> dict[str, list[str]]`, keyed by kind.
- Produces, in `params`:
  - `GRAPH_PARAMS`, `PROTOCOL_PARAMS`, `PROBES`, `CHECKPOINTS`, `SEED`, `OUTPUT_DIR`, `SOURCE` (tuples or single `Param`s);
  - `FIELD_MAP: dict[str, str]` (field to tool parameter, or `"graph"` / `"probes"` / `"checkpoints"`);
  - `graph_dict(args, n, ctx) -> dict`;
  - `ensemble_spec(args, ctx) -> dict` (the JSON API's nested spec);
  - `probe_dicts(items) -> list[dict]`;
  - `observable_dicts(items) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

```python
"""Vocabularies and the shared parameter groups (flat tool arguments -> JSON API specs)."""

import dataclasses

import pytest

from sparsegf2 import api, mipt
from sparsegf2.agent import params, vocab
from sparsegf2.agent.context import Context


@pytest.fixture
def ctx(tmp_path):
    return Context(tmp_path)


def test_vocab_matches_the_library():
    assert vocab.probe_signatures()["mutual_information"] == ("a", "b")
    assert vocab.probe_signatures()["erasure_first_failure"] == ("samples",)
    assert vocab.observable_signatures()["entropy"] == ("region",)
    assert vocab.graph_families()["watts_strogatz"] == ("k", "beta")
    assert "watts_strogatz" in vocab.random_graph_families()
    assert "cycle" not in vocab.random_graph_families()
    assert vocab.weight_families()["power_law"] == ("alpha", "periodic")
    assert "CX" in vocab.gate_names()["unitary"]


def test_every_monitored_circuit_field_is_mapped():
    fields = {f.name for f in dataclasses.fields(mipt.MonitoredCircuit) if not f.name.startswith("_")}
    assert set(params.FIELD_MAP) == fields


def test_watts_strogatz_collection_spec(ctx):
    args = {"num_system": 16, "layers": 8, "measurement_rate": 0.1, "graph": "watts_strogatz",
            "graph_count": 5, "graph_seed": 3, "circuits_per_graph": 2, "reference_partner": -1,
            "probes": ["code_dimension", "erasure_first_failure(512)@[4, 8]"],
            "checkpoints": "every_layer", "schedule": "random_edges", "reference": "purification"}
    spec = params.ensemble_spec(args, ctx)
    assert spec["graph"] == {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": 3,
                             "realizations": 5}
    assert spec["reference_partner"] == "random"
    assert spec["probes"] == [{"kind": "code_dimension"},
                              {"kind": "erasure_first_failure", "samples": 512,
                               "checkpoints": [4, 8]}]
    assert spec["checkpoints"] == "every_layer"
    mc = api._build_spec(spec)
    assert mc.num_trajectories == 10


def test_graph_parameters_must_fit_the_family(ctx):
    with pytest.raises(ValueError, match="graph_beta does not apply to graph=cycle"):
        params.graph_dict({"graph": "cycle", "graph_beta": 0.2, "graph_count": 1,
                           "graph_seed": 0}, 8, ctx)
    with pytest.raises(ValueError, match="graph_count applies to random families"):
        params.graph_dict({"graph": "cycle", "graph_count": 3, "graph_seed": 0}, 8, ctx)
    with pytest.raises(ValueError, match="graph=file needs graph_file"):
        params.graph_dict({"graph": "file", "graph_count": 1, "graph_seed": 0}, 8, ctx)


def test_probe_and_observable_strings(ctx):
    assert params.probe_dicts(["entropy(0:4, name=left)@final"]) == [
        {"kind": "entropy", "region": "0:4", "name": "left", "checkpoints": "final"}]
    with pytest.raises(ValueError, match="unknown probe 'entrpy'.*entropy"):
        params.probe_dicts(["entrpy(0:4)"])
    assert params.observable_dicts(["expectation(Z0*Z1)", "code_dimension(reference=4:8)"]) == [
        {"kind": "expectation", "pauli": "Z0*Z1"}, {"kind": "code_dimension", "reference": "4:8"}]


def test_edge_weights_gate_set_and_system_order(ctx):
    args = {"num_system": 8, "layers": 2, "measurement_rate": 0.1, "graph": "complete",
            "graph_count": 1, "graph_seed": 0, "schedule": "random_edges",
            "edge_weights": "power_law(alpha=2.0)", "gate_set": ["CX", "3"],
            "system_order": "[7,6,5,4,3,2,1,0]"}
    spec = params.ensemble_spec(args, ctx)
    assert spec["edge_weights"] == {"family": "power_law", "alpha": 2.0}
    assert spec["gate_set"] == ["CX", 3]
    assert spec["system_order"] == [7, 6, 5, 4, 3, 2, 1, 0]
    api._build_spec(spec)  # valid for the JSON API
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_params.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/sparsegf2/agent/vocab.py`**

```python
"""Names the tools accept, read from the library itself so that they never go stale."""

from __future__ import annotations

import dataclasses
import inspect
from functools import cache

__all__ = [
    "gate_names", "graph_families", "observable_descriptions", "observable_signatures",
    "probe_descriptions", "probe_signatures", "protocol_fields", "random_graph_families",
    "weight_families",
]

_PROBE_KEYWORDS = ("name", "checkpoints")


def _first_line(fn) -> str:
    return (inspect.getdoc(fn) or "").strip().splitlines()[0] if inspect.getdoc(fn) else ""


@cache
def probe_signatures() -> dict[str, tuple[str, ...]]:
    from sparsegf2 import api

    return {
        kind: tuple(p for p in inspect.signature(fn).parameters if p not in _PROBE_KEYWORDS)
        for kind, fn in api._probe_kinds().items()
    }


@cache
def probe_descriptions() -> dict[str, str]:
    from sparsegf2 import api

    return {kind: _first_line(fn) for kind, fn in api._probe_kinds().items()}


@cache
def observable_signatures() -> dict[str, tuple[str, ...]]:
    from sparsegf2 import api

    return {k: (*o.required, *o.optional) for k, o in api._OBSERVABLES.items()}


@cache
def observable_descriptions() -> dict[str, str]:
    from sparsegf2 import api

    return {k: o.description for k, o in api._OBSERVABLES.items()}


@cache
def graph_families() -> dict[str, tuple[str, ...]]:
    from sparsegf2 import api

    skip = {"n", "seed", "max_tries"}
    return {
        name: tuple(p for p in inspect.signature(fn).parameters if p not in skip)
        for name, fn in api._graph_families().items()
    }


@cache
def random_graph_families() -> frozenset[str]:
    from sparsegf2 import api

    return frozenset(
        name for name, fn in api._graph_families().items()
        if "seed" in inspect.signature(fn).parameters
    )


@cache
def weight_families() -> dict[str, tuple[str, ...]]:
    from sparsegf2 import api

    skip = {"edges", "n"}
    return {
        name: tuple(p for p in inspect.signature(fn).parameters if p not in skip)
        for name, fn in api._weight_families().items()
    }


@cache
def protocol_fields() -> tuple[str, ...]:
    from sparsegf2 import mipt

    return tuple(
        f.name for f in dataclasses.fields(mipt.MonitoredCircuit) if not f.name.startswith("_")
    )


@cache
def gate_names() -> dict[str, list[str]]:
    import sparsegf2 as sg

    out: dict[str, list[str]] = {}
    for g in sg.all_gates():
        out.setdefault(g.kind, []).append(g.name)
    return {k: sorted(v) for k, v in out.items()}
```

- [ ] **Step 4: Implement `src/sparsegf2/agent/params.py`**

```python
"""Parameter groups shared by several tools, and their translation to JSON API specs.

Protocol parameters carry the ``MonitoredCircuit`` field names exactly; graph
parameters are ``graph_`` plus the generator's parameter name (spec 5.3).
"""

from __future__ import annotations

import difflib
from typing import Any

from sparsegf2.agent import vocab
from sparsegf2.agent.grammar import (
    bind, parse_call, parse_calls, parse_checkpoints, parse_qubits, plain,
)
from sparsegf2.agent.registry import Param

_FAMILY_DEFAULTS = {"watts_strogatz": {"k": 4, "beta": 0.1}, "newman_watts": {"k": 4}}
_GRAPH_KEYS = ("graph_k", "graph_beta", "graph_p", "graph_d", "graph_rows", "graph_cols",
               "graph_periodic")


def _graph_enum() -> tuple[str, ...]:
    return (*vocab.graph_families(), "file")


GRAPH_PARAMS: tuple[Param, ...] = (
    Param("graph", "string", "Interaction graph family; file = any geometry from graph_file.",
          default="cycle", enum=_graph_enum()),
    Param("graph_k", "integer",
          "watts_strogatz/newman_watts: total degree K, even (K/2 neighbours per side). Default 4.",
          minimum=2),
    Param("graph_beta", "number", "watts_strogatz: rewiring probability. Default 0.1.",
          minimum=0, maximum=1),
    Param("graph_p", "number",
          "newman_watts: shortcut probability per edge; erdos_renyi: edge probability.",
          minimum=0, maximum=1),
    Param("graph_d", "integer", "random_regular: degree.", minimum=1),
    Param("graph_rows", "integer", "lattice_2d: rows (rows*cols = number of qubits).", minimum=1),
    Param("graph_cols", "integer", "lattice_2d: columns.", minimum=1),
    Param("graph_periodic", "boolean", "lattice_2d: periodic boundaries (default true)."),
    Param("graph_file", "string", "graph=file: a .json/.npz graph collection under the root."),
    Param("graph_count", "integer",
          "Random families: number of graphs G; graph i comes from seed (graph_seed, i).",
          default=1, minimum=1),
    Param("graph_seed", "integer",
          "Random families: seed of the graph collection (separate from the circuit seed).",
          default=0, minimum=0),
)

PROTOCOL_PARAMS: tuple[Param, ...] = (
    Param("num_system", "integer", "System qubits n (references follow at n..).",
          required=True, minimum=2),
    Param("layers", "integer", "Depth T: layers of gates then measurements (never rescaled).",
          required=True, minimum=0),
    Param("measurement_rate", "number",
          "Probability p that each system qubit is measured in Z after each layer.",
          required=True, minimum=0, maximum=1),
    Param("schedule", "string",
          "Gate placement: matching (edge colouring; brickwork on an even ring), random_edges "
          "(gates_per_layer edges with replacement), all_edges.",
          default="matching", enum=("matching", "random_edges", "all_edges")),
    Param("gates_per_layer", "integer", "random_edges: gates per layer (default n/2).", minimum=0),
    Param("circuits_per_graph", "integer",
          "Random circuits per graph; trajectory t runs on graph t // circuits_per_graph.",
          default=1, minimum=1),
    Param("reference", "string",
          "none: start in |0..0>; purification: each system qubit Bell-paired with its own "
          "reference (a code with n logical qubits); single: one reference qubit.",
          default="none", enum=("none", "purification", "single")),
    Param("reference_partner", "integer",
          "reference=single: the system qubit paired with it; -1 = random per trajectory.",
          default=-1, minimum=-1),
    Param("scramble_gates", "integer", "Random all-to-all gates before the dynamics.",
          default=0, minimum=0),
    Param("scramble_matchings", "integer",
          "Layers of gates on random perfect matchings before the dynamics.", default=0, minimum=0),
    Param("scramble_layers", "integer", "Gate-only layers of the schedule before the dynamics.",
          default=0, minimum=0),
    Param("gate_set", "array",
          "Two-qubit gates to draw from (names like CX, or CLIFFORD_2Q indices); default all 11520.",
          items="string"),
    Param("random_orientation", "boolean",
          "Randomly swap each gate's qubits (matters for a non-uniform gate_set).", default=False),
    Param("edge_weights", "string", "random_edges: edge weights as a call, e.g. power_law(alpha=2.0)."),
    Param("stop_when_purified", "boolean",
          "End a trajectory at the first checkpoint where the reference is pure.", default=False),
    Param("system_order", "string",
          "Qubit order for half-cut/profile/contiguous probes, e.g. 0:16 or [3,1,0,2]."),
    Param("periodic", "boolean", "Whether contiguous runs of system_order wrap around.",
          default=True),
)

PROBES = Param(
    "probes", "array",
    "Probe calls recorded at checkpoints, e.g. code_dimension, entropy(0:8), "
    "erasure_first_failure(samples=512)@[64,128] (help_list(what=probes)).",
    items="string",
)
CHECKPOINTS = Param(
    "checkpoints", "string",
    "Layers to record at: final, every_layer, a stride like 16, or a list like [0,64,128].",
    default="final",
)
SEED = Param("seed", "integer", "Circuit seed (omit for a fresh one; it is reported).", minimum=0)
OUTPUT_DIR = Param(
    "output_dir", "string",
    "Where results are saved, under the root (default runs/<tool>-<hash of the request>).",
)
SOURCE = Param("source", "string", "A saved run, record or file under the root (result_list).",
               required=True)

FIELD_MAP: dict[str, str] = {
    **{p.name: p.name for p in PROTOCOL_PARAMS},
    "edges": "graph", "probes": "probes", "checkpoints": "checkpoints",
}


def graph_dict(args: dict[str, Any], n: int, ctx) -> dict[str, Any]:
    """The JSON API's graph object for the graph parameters in ``args``."""
    family = args.get("graph", "cycle")
    families = vocab.graph_families()
    given = [k for k in _GRAPH_KEYS if k in args]
    if family == "file":
        if "graph_file" not in args:
            raise ValueError("graph=file needs graph_file (a .json or .npz graph collection)")
        if given:
            raise ValueError(f"{given[0]} does not apply to graph=file (the file fixes the graphs)")
        path = ctx.path(args["graph_file"], what="graph_file", must_exist=True)
        return {"family": "file", "path": str(path)}
    if "graph_file" in args:
        raise ValueError(f"graph_file needs graph=file (graph is {family})")
    takes = families[family]
    out: dict[str, Any] = {"family": family, **_FAMILY_DEFAULTS.get(family, {})}
    for key in given:
        name = key.removeprefix("graph_")
        if name not in takes:
            raise ValueError(
                f"{key} does not apply to graph={family}; {family} takes "
                f"{[f'graph_{t}' for t in takes] or 'no parameters'}"
            )
        out[name] = args[key]
    if family in vocab.random_graph_families():
        out["seed"] = args.get("graph_seed", 0)
        out["realizations"] = args.get("graph_count", 1)
    elif args.get("graph_count", 1) != 1:
        raise ValueError(f"graph_count applies to random families; {family} is deterministic")
    return out


def _suggest(name: str, options) -> str:
    close = difflib.get_close_matches(name, list(options), n=3)
    return f"; did you mean {', '.join(close)}?" if close else f"; choose from {sorted(options)}"


def probe_dicts(items: list[str]) -> list[dict[str, Any]]:
    sigs = vocab.probe_signatures()
    out = []
    for i, call in enumerate(parse_calls(items, allow_at=True, what="probes")):
        if call.name not in sigs:
            raise ValueError(f"probes[{i}]: unknown probe {call.name!r}{_suggest(call.name, sigs)}")
        bound = bind(call, sigs[call.name], what=f"probes[{i}]", keyword_only=["name"])
        d = {"kind": call.name, **plain(bound)}
        if call.at is not None:
            d["checkpoints"] = call.at
        out.append(d)
    return out


def observable_dicts(items: list[str]) -> list[dict[str, Any]]:
    sigs = vocab.observable_signatures()
    out = []
    for i, call in enumerate(parse_calls(items, what="observables")):
        if call.name not in sigs:
            raise ValueError(
                f"observables[{i}]: unknown observable {call.name!r}{_suggest(call.name, sigs)}"
            )
        bound = bind(call, sigs[call.name], what=f"observables[{i}]", keyword_only=["name"])
        out.append({"kind": call.name, **plain(bound)})
    return out


def ensemble_spec(args: dict[str, Any], ctx) -> dict[str, Any]:
    """The JSON API spec (``sparsegf2.api``) for flat ensemble/trajectory arguments."""
    special = {"edge_weights", "system_order", "reference_partner", "gate_set"}
    spec: dict[str, Any] = {
        p.name: args[p.name] for p in PROTOCOL_PARAMS if p.name in args and p.name not in special
    }
    spec["graph"] = graph_dict(args, args["num_system"], ctx)
    if "reference_partner" in args:
        rp = args["reference_partner"]
        spec["reference_partner"] = "random" if rp < 0 else rp
    if "edge_weights" in args:
        call = parse_call(args["edge_weights"], what="edge_weights")
        families = vocab.weight_families()
        if call.name not in families:
            raise ValueError(f"edge_weights: unknown family {call.name!r}{_suggest(call.name, families)}")
        spec["edge_weights"] = {"family": call.name,
                                **plain(bind(call, families[call.name], what="edge_weights"))}
    if "gate_set" in args:
        spec["gate_set"] = [int(g) if g.strip().isdigit() else g.strip() for g in args["gate_set"]]
    if "system_order" in args:
        spec["system_order"] = parse_qubits(args["system_order"], what="system_order")
    if "checkpoints" in args:
        spec["checkpoints"] = parse_checkpoints(args["checkpoints"])
    spec["probes"] = probe_dicts(args.get("probes") or [])
    return spec
```

Note: `test_edge_weights_gate_set_and_system_order` expects `system_order` as a list. `parse_qubits("[7,6,...]")` returns a list, and the API takes lists or ranges.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_params.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent/vocab.py src/sparsegf2/agent/params.py tests/agent/test_params.py
git commit -m "agent: vocabularies from the library; shared graph/protocol parameters -> JSON API specs"
```

---

### Task 13: Stacks: built-in, bundles, presets, custom files, and the `sparsegf2 stacks` CLI

**Files:**
- Create: `src/sparsegf2/agent/stacks.py`, `src/sparsegf2/agent/presets.json`
- Modify: `src/sparsegf2/cli.py` (the `stacks` command), `pyproject.toml` (package data `agent/*.json`, `agent/topics/*.md`)
- Test: `tests/agent/test_stacks.py`

**Interfaces:**
- Consumes: `REGISTRY`, `Bound`, `ToolDef.available()`, `Param.check`.
- Produces (in `sparsegf2.agent.stacks`):
  - `STACKS`, the 11 stack names in order.
  - `CORE`, a tuple of tool names.
  - `StackDef(name, description="", include=(), exclude=(), fixed={}, defaults={}, origin="")`.
  - `Resolution(name, bounds: list[Bound], notes: list[str])`.
  - `presets() -> dict[str, StackDef]`.
  - `user_file() -> Path`, `project_file(root) -> Path`, `stack_files(root, stacks_file=None) -> list[Path]`.
  - `load_file(path) -> dict[str, StackDef]`, `custom_stacks(root, stacks_file=None) -> dict[str, StackDef]`.
  - `resolve(name, *, root, stacks_file=None, registry=None) -> Resolution`.
  - `catalog(root, stacks_file=None) -> list[dict]` (name, kind, description, tools, size).
  - `save_stack(path, stack: StackDef) -> Path`, `delete_stack(path, name) -> bool`.
- CLI: `sparsegf2 stacks list|show NAME|save NAME ...|delete NAME [--project|--user] [--root DIR] [--stacks-file F] [--json]`.

- [ ] **Step 1: Write the failing tests**

The tests register a few throwaway tools in a private registry, so they do not depend on Phase 3.

```python
"""Stack resolution, custom stack files, and the stacks CLI."""

import json

import pytest

from sparsegf2.agent import stacks
from sparsegf2.agent.registry import Param, ToolDef


def _t(name, stack, long_running=False, params=(), requires=()):
    return ToolDef(name, stack, "s", "d", tuple(params), lambda c, a: {}, lambda d, c: name,
                   long_running=long_running, requires=requires)


@pytest.fixture
def reg():
    p = (Param("num_system", "integer", "n", required=True), Param("graph", "string", "g",
         default="cycle", enum=("cycle", "watts_strogatz")))
    tools = [
        _t("help_topic", "help"), _t("ensemble_run", "ensemble", True, p),
        _t("traj_run", "traj", True, p), _t("job_status", "job"), _t("job_cancel", "job"),
        _t("plot_make", "plot", requires=("module_that_does_not_exist_xyz",)),
    ]
    return {t.name: t for t in tools}


def test_builtin_stack_and_all_bundle(tmp_path, reg):
    r = stacks.resolve("job", root=tmp_path, registry=reg)
    assert [b.name for b in r.bounds] == ["job_status", "job_cancel"]
    r = stacks.resolve("all", root=tmp_path, registry=reg)
    assert "plot_make" not in [b.name for b in r.bounds]
    assert any("module_that_does_not_exist_xyz" in n for n in r.notes)


def test_long_running_tools_bring_job_status(tmp_path, reg):
    r = stacks.resolve("ensemble", root=tmp_path, registry=reg)
    assert [b.name for b in r.bounds] == ["ensemble_run", "job_status"]


def test_custom_stack_include_exclude_fixed_defaults(tmp_path, reg):
    f = tmp_path / ".sparsegf2" / "stacks.json"
    f.parent.mkdir()
    f.write_text(json.dumps({"version": 1, "stacks": {"mine": {
        "description": "x", "include": ["ensemble", "traj_run", "help"],
        "exclude": ["help_topic"],
        "fixed": {"*": {"num_system": 64}}, "defaults": {"ensemble_run": {"graph": "watts_strogatz"}},
    }}}), encoding="utf-8")
    r = stacks.resolve("mine", root=tmp_path, registry=reg)
    by = {b.name: b for b in r.bounds}
    assert set(by) == {"ensemble_run", "traj_run", "job_status"}
    assert by["traj_run"].fixed == {"num_system": 64}
    assert by["ensemble_run"].defaults == {"graph": "watts_strogatz"}
    assert "num_system" not in by["ensemble_run"].input_schema()["properties"]


@pytest.mark.parametrize(
    "stack, message",
    [
        ({"include": ["nope"]}, "unknown stack or tool 'nope'"),
        ({"include": ["ensemble"], "fixed": {"ensemble_run": {"layrs": 3}}}, "no parameter 'layrs'"),
        ({"include": ["ensemble"], "fixed": {"ensemble_run": {"graph": "ring"}}}, "must be one of"),
        ({"include": ["ensemble"], "fixed": {"traj_run": {"num_system": 3}}}, "not in this stack"),
        ({"include": ["mine"]}, "includes itself"),
    ],
)
def test_validation_errors_name_the_stack(tmp_path, reg, stack, message):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"version": 1, "stacks": {"mine": stack}}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        stacks.resolve("mine", root=tmp_path, stacks_file=f, registry=reg)


def test_custom_names_may_not_shadow_builtins(tmp_path, reg):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"version": 1, "stacks": {"core": {"include": ["job"]}}}),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="'core' is a built-in name"):
        stacks.resolve("job", root=tmp_path, stacks_file=f, registry=reg)


def test_unknown_stack_suggests(tmp_path, reg):
    with pytest.raises(ValueError, match="unknown stack 'jobs'.*job"):
        stacks.resolve("jobs", root=tmp_path, registry=reg)


def test_save_and_delete_are_atomic_round_trips(tmp_path):
    f = tmp_path / "s.json"
    stacks.save_stack(f, stacks.StackDef("a", include=("job",)))
    stacks.save_stack(f, stacks.StackDef("b", include=("help",), fixed={"*": {"x": 1}}))
    data = json.loads(f.read_text(encoding="utf-8"))
    assert set(data["stacks"]) == {"a", "b"} and data["version"] == 1
    assert stacks.delete_stack(f, "a") and not stacks.delete_stack(f, "a")


def test_presets_file_is_well_formed():
    p = stacks.presets()
    assert set(p) == {"erasure", "purification", "entanglement", "circuits"}
    assert p["erasure"].defaults["*"]["reference"] == "purification"
    assert p["erasure"].defaults["ensemble_run"]["probes"][1] == "erasure_first_failure(samples=512)"


def test_cli_save_show_delete(tmp_path, capsys):
    from sparsegf2 import cli

    f = str(tmp_path / "s.json")
    assert cli.main(["stacks", "save", "mine", "--include", "job,help", "--stacks-file", f,
                     "--root", str(tmp_path), "--json"]) == 0
    assert cli.main(["stacks", "show", "mine", "--stacks-file", f, "--root", str(tmp_path),
                     "--json"]) == 0
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert "job_status" in json.loads(out)["tools"]
    assert cli.main(["stacks", "delete", "mine", "--stacks-file", f, "--json"]) == 0
```

(`test_cli_save_show_delete` resolves `job` and `help` against the real registry, which has those stacks once Task 14 and Task 15 land. Mark it `@pytest.mark.skipif(not {"job_status", "help_topic"} <= set(__import__("sparsegf2.agent.registry", fromlist=["REGISTRY"]).REGISTRY), reason="tools land in Tasks 14-15")` until then, and remove the mark in Task 15.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_stacks.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `src/sparsegf2/agent/presets.json`**

```json
{
  "version": 1,
  "stacks": {
    "erasure": {
      "description": "Erasure protection of the codes monitored circuits generate (Watts-Strogatz ensembles by default).",
      "include": ["ensemble_run", "ensemble_sweep", "traj_run", "code_analyze", "analysis_erasure",
                  "analysis_compare", "graph_stats", "result_list", "job", "help_topic"],
      "defaults": {
        "*": {"reference": "purification", "schedule": "random_edges", "graph": "watts_strogatz",
              "graph_k": 4, "graph_count": 50, "circuits_per_graph": 4},
        "ensemble_run": {"probes": ["code_dimension", "erasure_first_failure(samples=512)",
                                    "uncorrectable_region_counts(2)"]},
        "ensemble_sweep": {"probes": ["code_dimension", "erasure_first_failure(samples=512)",
                                      "uncorrectable_region_counts(2)"]}
      }
    },
    "purification": {
      "description": "Purification of a single reference qubit: survival curves and transitions.",
      "include": ["ensemble_run", "ensemble_sweep", "analysis_purification", "analysis_compare",
                  "plot_make", "job", "help_topic"],
      "defaults": {
        "*": {"reference": "single", "reference_partner": -1, "stop_when_purified": true,
              "checkpoints": "every_layer"},
        "ensemble_run": {"probes": ["code_dimension"]},
        "ensemble_sweep": {"probes": ["code_dimension"]}
      }
    },
    "entanglement": {
      "description": "Entanglement dynamics and the measurement-induced transition.",
      "include": ["ensemble_run", "ensemble_sweep", "analysis_probe", "analysis_compare",
                  "plot_make", "job", "help_topic"],
      "defaults": {
        "ensemble_run": {"probes": ["half_cut_entropy", "tripartite_information"]},
        "ensemble_sweep": {"probes": ["half_cut_entropy", "tripartite_information"]}
      }
    },
    "circuits": {
      "description": "Stabilizer states gate by gate and Stim-language circuits.",
      "include": ["state", "circuit", "code_analyze", "help"]
    }
  }
}
```

Defaults under `"*"` apply to every tool in the stack that has the parameter (like `fixed`).

- [ ] **Step 4: Implement `src/sparsegf2/agent/stacks.py`**

```python
"""Tool stacks: built-in stacks, bundles, presets and custom stacks saved by the user.

A stack resolves to a list of :class:`~sparsegf2.agent.registry.Bound` tools.
Custom stacks live in JSON files (spec section 8.2); presets ship in
``presets.json`` in the same format.
"""

from __future__ import annotations

import difflib
import json
import os
import sys
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from sparsegf2.agent.registry import REGISTRY, Bound, ToolDef

__all__ = [
    "CORE", "STACKS", "Resolution", "StackDef", "catalog", "custom_stacks", "delete_stack",
    "load_file", "presets", "project_file", "resolve", "save_stack", "stack_files", "user_file",
]

STACKS = ("help", "state", "circuit", "ensemble", "traj", "code", "graph", "analysis", "result",
          "job", "plot")
CORE = ("help_topic", "help_list", "ensemble_run", "traj_run", "code_analyze",
        "analysis_summary", "analysis_probe", "analysis_compare", "result_list", "job_status",
        "job_cancel")
BUNDLES = ("all", "core")


@dataclass(frozen=True)
class StackDef:
    name: str
    description: str = ""
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    fixed: dict[str, dict[str, Any]] = field(default_factory=dict)
    defaults: dict[str, dict[str, Any]] = field(default_factory=dict)
    origin: str = ""

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"description": self.description, "include": list(self.include)}
        if self.exclude:
            out["exclude"] = list(self.exclude)
        if self.fixed:
            out["fixed"] = self.fixed
        if self.defaults:
            out["defaults"] = self.defaults
        return out


@dataclass
class Resolution:
    name: str
    bounds: list[Bound]
    notes: list[str]


def _parse(doc: Any, origin: str) -> dict[str, StackDef]:
    if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get("stacks"), dict):
        raise ValueError(f"{origin}: expected {{\"version\": 1, \"stacks\": {{...}}}}")
    out = {}
    for name, body in doc["stacks"].items():
        if not isinstance(body, dict):
            raise ValueError(f"{origin}: stack {name!r} must be an object")
        extra = set(body) - {"description", "include", "exclude", "fixed", "defaults"}
        if extra:
            raise ValueError(f"{origin}: stack {name!r} has unknown key(s) {sorted(extra)}")
        out[name] = StackDef(
            name, str(body.get("description", "")), tuple(body.get("include", ())),
            tuple(body.get("exclude", ())), dict(body.get("fixed", {})),
            dict(body.get("defaults", {})), origin,
        )
    return out


def presets() -> dict[str, StackDef]:
    text = resources.files("sparsegf2.agent").joinpath("presets.json").read_text(encoding="utf-8")
    return _parse(json.loads(text), "presets.json")


def user_file() -> Path:
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "sparsegf2" / "stacks.json"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "sparsegf2" / "stacks.json"


def project_file(root: Path) -> Path:
    return Path(root) / ".sparsegf2" / "stacks.json"


def stack_files(root: Path, stacks_file: Path | None = None) -> list[Path]:
    files = [Path(stacks_file)] if stacks_file else []
    return [*files, project_file(root), user_file()]


def load_file(path: Path) -> dict[str, StackDef]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from None
    return _parse(doc, str(path))


def custom_stacks(root: Path, stacks_file: Path | None = None) -> dict[str, StackDef]:
    reserved = {*STACKS, *BUNDLES, *presets()}
    out: dict[str, StackDef] = {}
    for path in stack_files(root, stacks_file):
        for name, stack in load_file(path).items():
            if name in reserved:
                raise ValueError(f"{path}: {name!r} is a built-in name; choose another")
            out.setdefault(name, stack)  # earlier files win
    return out


def _all_tools(registry: dict[str, ToolDef]) -> list[str]:
    order = {s: i for i, s in enumerate(STACKS)}
    return sorted(registry, key=lambda t: (order.get(registry[t].stack, 99), list(registry).index(t)))


def resolve(name: str, *, root: Path, stacks_file: Path | None = None,
            registry: dict[str, ToolDef] | None = None) -> Resolution:
    registry = REGISTRY if registry is None else registry
    named = {**presets(), **custom_stacks(root, stacks_file)}
    notes: list[str] = []

    def expand(item: str, seen: tuple[str, ...]) -> dict[str, tuple[dict, dict]]:
        if item in seen:
            raise ValueError(f"stack {seen[0]!r} includes itself through {' -> '.join((*seen, item))}")
        if item == "all":
            names = _all_tools(registry)
        elif item == "core":
            names = [t for t in CORE if t in registry]
        elif item in STACKS:
            names = [t for t in _all_tools(registry) if registry[t].stack == item]
        elif item in registry:
            names = [item]
        elif item in named:
            return expand_def(named[item], (*seen, item))
        else:
            options = [*STACKS, *BUNDLES, *named, *registry]
            close = difflib.get_close_matches(item, options, n=3)
            where = f"stack {seen[-1]!r}: " if seen else ""
            kind = "stack or tool" if seen else "stack"
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise ValueError(f"{where}unknown {kind} {item!r}{hint}")
        return {t: ({}, {}) for t in names}

    def apply(table: dict[str, dict[str, Any]], tools: dict[str, tuple[dict, dict]], which: int,
              stack: StackDef) -> None:
        for target, values in table.items():
            if target != "*" and target not in tools:
                label = "defaults" if which else "fixed"
                raise ValueError(f"stack {stack.name!r}: {label} names {target!r}, which is not "
                                 "in this stack")
            for tname in (tools if target == "*" else [target]):
                tool = registry[tname]
                names = {p.name: p for p in tool.params}
                for pname, value in values.items():
                    if pname not in names:
                        if target == "*":
                            continue
                        raise ValueError(f"stack {stack.name!r}: {tname} has no parameter {pname!r}")
                    checked = names[pname].check(value, f"stack {stack.name!r}, {tname}")
                    tools[tname][which][pname] = checked

    def expand_def(stack: StackDef, seen: tuple[str, ...]) -> dict[str, tuple[dict, dict]]:
        tools: dict[str, tuple[dict, dict]] = {}
        for item in stack.include:
            for t, (fx, df) in expand(item, seen).items():
                cur = tools.setdefault(t, ({}, {}))
                cur[0].update(fx)
                cur[1].update(df)
        for t in stack.exclude:
            tools.pop(t, None)
        apply(stack.fixed, tools, 0, stack)
        apply(stack.defaults, tools, 1, stack)
        return tools

    if name not in (*STACKS, *BUNDLES) and name not in named and name not in registry:
        expand(name, ())  # raises "unknown stack" with suggestions
    tools = expand(name, ())
    if any(registry[t].long_running for t in tools) and "job_status" in registry:
        tools.setdefault("job_status", ({}, {}))
    bounds = []
    for t, (fx, df) in tools.items():
        reason = registry[t].available()
        if reason:
            notes.append(f"{reason}; left out of stack {name!r}")
            continue
        bounds.append(Bound(registry[t], dict(fx), dict(df)))
    return Resolution(name, bounds, notes)


def catalog(root: Path, stacks_file: Path | None = None) -> list[dict[str, Any]]:
    rows = []
    custom = custom_stacks(root, stacks_file)
    entries = ([(s, "stack") for s in STACKS] + [(b, "bundle") for b in BUNDLES]
               + [(p, "preset") for p in presets()] + [(c, "custom") for c in custom])
    for name, kind in entries:
        try:
            r = resolve(name, root=root, stacks_file=stacks_file)
        except ValueError as exc:
            rows.append({"name": name, "kind": kind, "error": str(exc)})
            continue
        if not r.bounds and kind == "stack":
            continue
        desc = (presets().get(name) or custom.get(name) or StackDef(name)).description
        rows.append({"name": name, "kind": kind, "description": desc,
                     "tools": [b.name for b in r.bounds],
                     "size": sum(b.size() for b in r.bounds)})
    return rows


def save_stack(path: Path, stack: StackDef) -> Path:
    path = Path(path)
    doc = {"version": 1, "stacks": {}}
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
    doc["stacks"][stack.name] = stack.to_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def delete_stack(path: Path, name: str) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    doc = json.loads(path.read_text(encoding="utf-8"))
    if name not in doc.get("stacks", {}):
        return False
    del doc["stacks"][name]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return True
```

- [ ] **Step 5: Add the `stacks` command to `src/sparsegf2/cli.py`**

In `_build_parser`, before `add("mcp", ...)`:

```python
    p = add("stacks", "List, show, save or delete MCP tool stacks.", _STACKS_HELP)
    p.add_argument("action", choices=["list", "show", "save", "delete"])
    p.add_argument("name", nargs="?", help="stack name (show, save, delete)")
    p.add_argument("--root", default=".", help="project root (default: current directory)")
    p.add_argument("--stacks-file", default=None, help="a stacks.json to use first")
    p.add_argument("--from", dest="base", default=None, help="save: start from this stack")
    p.add_argument("--include", default="", help="save: comma-separated stacks and tools")
    p.add_argument("--exclude", default="", help="save: comma-separated tools to drop")
    p.add_argument("--fix", action="append", default=[], help="save: TOOL.PARAM=VALUE (TOOL may be *)")
    p.add_argument("--default", action="append", default=[], help="save: TOOL.PARAM=VALUE")
    p.add_argument("--description", default="", help="save: one line about the stack")
    where = p.add_mutually_exclusive_group()
    where.add_argument("--project", action="store_true", help="the project's .sparsegf2/stacks.json")
    where.add_argument("--user", action="store_true", help="the user stacks file (default)")
```

with the help text near the other `_*_HELP` strings:

```python
_STACKS_HELP = """\
examples:
  sparsegf2 stacks list
  sparsegf2 stacks show erasure
  sparsegf2 stacks save ws128 --from erasure --fix "*.num_system=128" --project
  sparsegf2 stacks delete ws128 --project
"""
```

and the command:

```python
def _cmd_stacks(args: argparse.Namespace) -> _Outcome:
    import sparsegf2.agent.tools  # noqa: F401  (registers the tools)
    from sparsegf2.agent import stacks

    root = Path(args.root).resolve()
    stacks_file = Path(args.stacks_file) if args.stacks_file else None
    target = (
        stacks_file if stacks_file else stacks.project_file(root) if args.project else stacks.user_file()
    )
    if args.action == "list":
        return {"stacks": stacks.catalog(root, stacks_file)}, EXIT_OK
    if not args.name:
        raise ValueError(f"stacks {args.action} needs a stack name")
    if args.action == "show":
        r = stacks.resolve(args.name, root=root, stacks_file=stacks_file)
        return {
            "name": args.name,
            "tools": [b.name for b in r.bounds],
            "fixed": {b.name: dict(b.fixed) for b in r.bounds if b.fixed},
            "defaults": {b.name: dict(b.defaults) for b in r.bounds if b.defaults},
            "size_chars": sum(b.size() for b in r.bounds),
            "notes": r.notes,
        }, EXIT_OK
    if args.action == "delete":
        removed = stacks.delete_stack(target, args.name)
        return {"deleted": removed, "file": str(target)}, EXIT_OK if removed else EXIT_USAGE

    def pairs(items: list[str], what: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for item in items:
            key, _, raw = item.partition("=")
            tool, _, param = key.partition(".")
            if not raw or not tool or not param:
                raise ValueError(f"--{what} expects TOOL.PARAM=VALUE, got {item!r}")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            out.setdefault(tool, {})[param] = value
        return out

    include = tuple(x for x in args.include.split(",") if x) or ((args.base,) if args.base else ())
    if args.base and args.base not in include:
        include = (args.base, *include)
    stack = stacks.StackDef(
        args.name, args.description, include,
        tuple(x for x in args.exclude.split(",") if x),
        pairs(args.fix, "fix"), pairs(args.default, "default"),
    )
    if not stack.include:
        raise ValueError("stacks save needs --include or --from")
    # validate by resolving against a temporary file holding just this stack
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        probe = stacks.save_stack(Path(tmp) / "stacks.json", stack)
        stacks.resolve(args.name, root=root, stacks_file=probe)
    path = stacks.save_stack(target, stack)
    return {"saved": args.name, "file": str(path)}, EXIT_OK
```

Register `"stacks": _cmd_stacks` in `_COMMANDS`, and add `from pathlib import Path` to the imports if it is missing. Check that `EXIT_OK` and `EXIT_USAGE` exist in `cli.py` (they are the exit-code constants `main` uses).

In `pyproject.toml`, extend the package data:

```toml
[tool.setuptools.package-data]
sparsegf2 = ["py.typed", "agent/*.json", "agent/topics/*.md"]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_stacks.py tests/test_cli.py -q`
Expected: PASS (the CLI test is skipped until Task 15)

- [ ] **Step 7: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent/stacks.py src/sparsegf2/agent/presets.json src/sparsegf2/cli.py pyproject.toml tests/agent/test_stacks.py
git commit -m "agent: stacks (built-in, bundles, presets, custom JSON) and the 'sparsegf2 stacks' CLI"
```

---

### Task 14: Dispatch, the low-level MCP server, and the help tools

**Files:**
- Create: `src/sparsegf2/agent/dispatch.py`, `src/sparsegf2/agent/server.py`, `src/sparsegf2/agent/tools/__init__.py`, `src/sparsegf2/agent/tools/help.py`, `src/sparsegf2/agent/topics/<topic>.md` (18 files)
- Test: `tests/agent/conftest.py`, `tests/agent/test_server.py`, `tests/agent/test_help_tools.py`

**Interfaces:**
- Consumes: everything from Tasks 8-13.
- Produces:
  - In `dispatch`: `Outcome(text, data, is_error)`, `call(ctx, bound, raw_args) -> Outcome`.
  - In `server`:
    - `build(stack="all", *, root=".", stacks_file=None, max_output_chars=4000, sync_seconds=60.0, max_states=16, quiet=False) -> tuple[Server, Context, Resolution]`;
    - `instructions(ctx) -> str`;
    - `main(argv=None) -> int`;
    - `python -m sparsegf2.agent.server` runs `main`.
  - The tools `help_topic`, `help_list`, with `TOPICS: tuple[str, ...]`.
  - In `tests/agent/conftest.py`, the fixtures `ctx`, `session(stack)` and `run_tool(name, **args)`, used by every later tool test.

- [ ] **Step 1: Write the shared test helpers `tests/agent/conftest.py`**

```python
"""Helpers for the agent tool tests: a context, direct dispatch and an MCP session."""

import asyncio

import pytest

import sparsegf2.agent.tools  # noqa: F401  (registers every tool)
from sparsegf2.agent import dispatch, server
from sparsegf2.agent.context import Context
from sparsegf2.agent.registry import REGISTRY, Bound


@pytest.fixture
def ctx(tmp_path):
    return Context(tmp_path, sync_seconds=60.0)


@pytest.fixture
def run_tool(ctx):
    """Call a tool through dispatch (validation, rendering, caps) and return the Outcome."""

    def call(name, **args):
        return dispatch.call(ctx, Bound(REGISTRY[name]), args)

    return call


@pytest.fixture
def session(tmp_path):
    """``session(stack, fn)`` runs ``await fn(client)`` against an in-process server."""
    import mcp

    def run(stack, fn):
        srv, _, _ = server.build(stack, root=tmp_path, quiet=True)

        async def go():
            async with mcp.Client(srv) as client:
                return await fn(client)

        return asyncio.run(go())

    return run
```

- [ ] **Step 2: Write the failing tests**

`tests/agent/test_help_tools.py`:

```python
"""help_topic and help_list."""

import pytest

from sparsegf2.agent.tools.help import TOPICS


@pytest.mark.parametrize("topic", TOPICS)
def test_every_topic_renders(run_tool, topic):
    out = run_tool("help_topic", topic=topic)
    assert not out.is_error and out.text.startswith(f"help_topic: {topic}")
    assert 200 < len(out.text) <= 4000


@pytest.mark.parametrize(
    "what, expect",
    [("probes", "erasure_first_failure(samples)"), ("observables", "entropy(region)"),
     ("graph_families", "watts_strogatz(graph_k, graph_beta)"), ("gates", "CX"),
     ("instructions", "MPP"), ("edge_weight_families", "power_law(alpha, periodic)"),
     ("protocol_parameters", "measurement_rate"), ("tools", "help_topic"), ("stacks", "core")],
)
def test_lists(run_tool, what, expect):
    out = run_tool("help_list", what=what)
    assert not out.is_error and out.text.startswith(f"help_list: {what}") and expect in out.text


def test_bad_topic_is_a_tool_error(run_tool):
    out = run_tool("help_topic", topic="nonsense")
    assert out.is_error and "must be one of" in out.text
```

`tests/agent/test_server.py`:

```python
"""The MCP server: listing, calling, errors, flags."""

import json

from sparsegf2.agent import server
from sparsegf2.agent.registry import check_flat


def test_list_tools_are_flat_and_named_by_stack(session):
    tools = session("help", lambda c: c.list_tools()).tools
    assert [t.name for t in tools] == ["help_topic", "help_list"]
    for t in tools:
        assert check_flat(t.input_schema) == []


def test_call_returns_text_and_structured_content(session):
    res = session("help", lambda c: c.call_tool("help_topic", {"topic": "conventions"}))
    assert not res.is_error and res.content[0].text.startswith("help_topic: conventions")
    json.dumps(res.structured_content, allow_nan=False)


def test_errors_are_tool_errors_not_protocol_errors(session):
    res = session("help", lambda c: c.call_tool("help_topic", {"topic": "x", "extra": 1}))
    assert res.is_error and "unknown argument(s) ['extra']" in res.content[0].text
    res = session("help", lambda c: c.call_tool("nope", {}))
    assert res.is_error and "unknown tool 'nope'" in res.content[0].text


def test_help_topics_are_resources(session):
    listed = session("help", lambda c: c.list_resources()).resources
    assert "docs://sparsegf2/conventions" in {str(r.uri) for r in listed}
    read = session("help", lambda c: c.read_resource("docs://sparsegf2/grammar"))
    assert "positional" in read.contents[0].text


def test_instructions_name_the_stack_and_version(tmp_path):
    srv, ctx, res = server.build("help", root=tmp_path, quiet=True)
    text = server.instructions(ctx)
    assert "stack: help" in text and "interface version 1" in text


def test_list_flag_prints_tools_and_sizes(tmp_path, capsys):
    assert server.main(["--stack", "help", "--root", str(tmp_path), "--list"]) == 0
    out = capsys.readouterr().out
    assert "help_topic" in out and "total" in out


def test_bad_stack_exits_2(tmp_path, capsys):
    assert server.main(["--stack", "nope", "--root", str(tmp_path)]) == 2
    assert "unknown stack 'nope'" in capsys.readouterr().err
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_server.py tests/agent/test_help_tools.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'sparsegf2.agent.tools'`

- [ ] **Step 4: Implement `src/sparsegf2/agent/dispatch.py`**

```python
"""Validate, run (here or as a background job), render, cap: one tool call."""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from typing import Any

from sparsegf2.agent.registry import Bound
from sparsegf2.agent.render import cap, jsonable
from sparsegf2.agent.sessions import render_job

__all__ = ["Outcome", "call"]


@dataclass(frozen=True)
class Outcome:
    text: str
    data: dict[str, Any] | None
    is_error: bool


def _start_generic(ctx, bound: Bound, args: dict[str, Any]) -> dict[str, Any]:
    tool = bound.tool
    key = tool.name + json.dumps(args, sort_keys=True, default=str)
    return ctx.jobs.start(
        tool=tool.name, key=key, run=lambda job: tool.handler(ctx, args),
        render=lambda data: tool.render(data, ctx),
        output_dir=args.get("output_dir"),
    )


def call(ctx, bound: Bound, raw_args: dict[str, Any] | None) -> Outcome:
    tool = bound.tool
    try:
        args = bound.validate(raw_args)
        mode = args.pop("background", "never") if tool.long_running else "never"
        background = mode == "always" or (
            mode == "auto" and tool.estimate is not None and tool.estimate(ctx, args) > ctx.sync_seconds
        )
        if background:
            data = tool.background(ctx, args) if tool.background else _start_generic(ctx, bound, args)
            text = render_job(data)
        else:
            data = tool.handler(ctx, args)
            text = tool.render(data, ctx)
        full = data.get("output_dir") if isinstance(data, dict) else None
        return Outcome(cap(text, ctx.max_output_chars, full=full), jsonable(data), False)
    except (ValueError, ImportError) as exc:
        return Outcome(f"{tool.name}: {exc}", None, True)
    except Exception as exc:  # a bug: report it plainly, never crash the server
        if not ctx.quiet:
            traceback.print_exc(file=sys.stderr)
        return Outcome(
            f"{tool.name}: internal error ({type(exc).__name__}): {exc}. This is a bug in "
            "SparseGF2; report it with the arguments used.", None, True,
        )
```

- [ ] **Step 5: Implement `src/sparsegf2/agent/server.py`**

```python
"""The SparseGF2 MCP server (low-level SDK server with registry-generated flat schemas)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sparsegf2._version import __version__
from sparsegf2.agent import INTERFACE_VERSION
from sparsegf2.agent.context import Context

__all__ = ["SERVER_NAME", "build", "instructions", "main"]

SERVER_NAME = "sparsegf2"


def instructions(ctx: Context) -> str:
    return (
        f"SparseGF2 {__version__}, interface version {INTERFACE_VERSION}, stack: {ctx.stack}. "
        "Fast stabilizer (Clifford) simulation: states gate by gate, Stim-language circuits, "
        "ensembles of monitored random circuits, single trajectories, codes and erasure, graphs, "
        "analysis of saved results. Structured arguments are call strings like entropy(0:8) or "
        "erasure_first_failure(samples=512)@[64,128]. Start with help_topic(topic=\"conventions\"); "
        f"help_list names everything. Paths are relative to {ctx.root}. Long runs return a "
        "job_id: poll job_status."
    )


def build(stack: str = "all", *, root: str | Path = ".", stacks_file: str | Path | None = None,
          max_output_chars: int = 4000, sync_seconds: float = 60.0, max_states: int = 16,
          quiet: bool = False):
    """The server for ``stack``, its context, and the stack resolution (tools and notes)."""
    from mcp import types
    from mcp.server.lowlevel import Server

    import anyio

    import sparsegf2.agent.tools  # noqa: F401  (registers every tool)
    from sparsegf2.agent import dispatch, stacks

    resolution = stacks.resolve(stack, root=Path(root), stacks_file=stacks_file)
    ctx = Context(Path(root), max_output_chars=max_output_chars, sync_seconds=sync_seconds,
                  max_states=max_states, quiet=quiet, stack=stack,
                  tool_names=tuple(b.name for b in resolution.bounds))
    by_name = {b.name: b for b in resolution.bounds}
    listing = [
        types.Tool(
            name=b.name, description=b.tool.description, input_schema=b.input_schema(),
            annotations=types.ToolAnnotations(read_only_hint=b.tool.read_only,
                                              destructive_hint=False, open_world_hint=False),
        )
        for b in resolution.bounds
    ]

    async def on_list_tools(_rctx, _params) -> Any:
        return types.ListToolsResult(tools=listing)

    async def on_call_tool(_rctx, params) -> Any:
        bound = by_name.get(params.name)
        if bound is None:
            text = f"unknown tool {params.name!r}; this server's tools: {sorted(by_name)}"
            return types.CallToolResult(content=[types.TextContent(type="text", text=text)],
                                        is_error=True)
        out = await anyio.to_thread.run_sync(dispatch.call, ctx, bound, params.arguments or {})
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=out.text)],
            structured_content=out.data, is_error=out.is_error,
        )

    from importlib import resources

    from sparsegf2.agent.tools.help import TOPICS

    def topic_text(topic: str) -> str:
        return resources.files("sparsegf2.agent").joinpath("topics", f"{topic}.md").read_text(
            encoding="utf-8")

    docs = [types.Resource(name=t, uri=f"docs://sparsegf2/{t}", mime_type="text/markdown",
                           description=f"SparseGF2 guidance: {t}") for t in TOPICS]

    async def on_list_resources(_rctx, _params) -> Any:
        return types.ListResourcesResult(resources=docs)

    async def on_read_resource(_rctx, params) -> Any:
        uri = str(params.uri)
        topic = uri.removeprefix("docs://sparsegf2/")
        if topic not in TOPICS:
            raise ValueError(f"unknown resource {uri!r}; resources: docs://sparsegf2/<topic>")
        return types.ReadResourceResult(contents=[types.TextResourceContents(
            uri=uri, mime_type="text/markdown", text=topic_text(topic))])

    srv = Server(SERVER_NAME, version=__version__, title="SparseGF2 stabilizer simulator",
                 instructions=instructions(ctx), on_list_tools=on_list_tools,
                 on_call_tool=on_call_tool, on_list_resources=on_list_resources,
                 on_read_resource=on_read_resource)
    return srv, ctx, resolution


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sparsegf2-mcp",
                                     description="Serve SparseGF2 over MCP (stdio).")
    parser.add_argument("--stack", default="all", help="stack, bundle, preset or custom stack")
    parser.add_argument("--root", default=".", help="folder all paths are confined to")
    parser.add_argument("--stacks-file", default=None, help="a stacks.json to read first")
    parser.add_argument("--sync-seconds", type=float, default=60.0)
    parser.add_argument("--max-output-chars", type=int, default=4000)
    parser.add_argument("--max-states", type=int, default=16)
    parser.add_argument("--quiet", action="store_true", help="no stderr output except fatal errors")
    parser.add_argument("--list", action="store_true", help="print the tools and sizes, then exit")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        srv, ctx, res = build(
            args.stack, root=args.root, stacks_file=args.stacks_file,
            max_output_chars=args.max_output_chars, sync_seconds=args.sync_seconds,
            max_states=args.max_states, quiet=args.quiet,
        )
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.list:
        total = 0
        for b in res.bounds:
            total += b.size()
            print(f"{b.name:24s} {b.size():6d} chars  {b.tool.summary}")
        print(f"{'total':24s} {total:6d} chars (~{total // 4} tokens), {len(res.bounds)} tools")
        for note in res.notes:
            print(f"note: {note}")
        return 0
    if not args.quiet:
        for note in res.notes:
            print(f"note: {note}", file=sys.stderr)
    import anyio
    from mcp.server.stdio import stdio_server

    async def serve() -> None:
        async with stdio_server() as (read, write):
            await srv.run(read, write, srv.create_initialization_options())

    anyio.run(serve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Implement `src/sparsegf2/agent/tools/__init__.py` and `tools/help.py`**

`tools/__init__.py` (later tasks add one import line each, in stack order):

```python
"""Importing this package registers every tool (one module per stack)."""

from sparsegf2.agent.tools import help  # noqa: F401
```

`tools/help.py`:

```python
"""help_topic and help_list: conventions and exact names, generated from the library."""

from __future__ import annotations

from importlib import resources
from typing import Any

from sparsegf2.agent import vocab
from sparsegf2.agent.registry import Param, ToolDef, register

TOPICS = (
    "overview", "conventions", "grammar", "states", "circuits", "ensembles", "probes",
    "trajectories", "codes", "erasure", "purification", "entanglement", "graphs", "analysis",
    "files", "jobs", "stacks", "recipes",
)
LISTS = (
    "tools", "gates", "instructions", "observables", "probes", "code_properties", "metrics",
    "graph_families", "edge_weight_families", "protocol_parameters", "stacks",
)


def _topic(ctx, args: dict[str, Any]) -> dict[str, Any]:
    text = resources.files("sparsegf2.agent").joinpath("topics", f"{args['topic']}.md")
    return {"topic": args["topic"], "text": text.read_text(encoding="utf-8").strip()}


def _list(ctx, args: dict[str, Any]) -> dict[str, Any]:
    what = args["what"]
    rows: list[str]
    if what == "tools":
        from sparsegf2.agent import INTERFACE_VERSION
        from sparsegf2.agent.registry import REGISTRY

        rows = [f"interface version {INTERFACE_VERSION}, stack {ctx.stack}"]
        rows += [f"{n}: {REGISTRY[n].summary}" for n in ctx.tool_names or REGISTRY]
    elif what == "gates":
        g = vocab.gate_names()
        rows = [f"{k}: {' '.join(v)}" for k, v in g.items() if k in ("unitary", "clifford_index")]
    elif what == "instructions":
        g = vocab.gate_names()
        rows = [f"{k}: {' '.join(v)}" for k, v in g.items() if k not in ("unitary", "clifford_index")]
    elif what == "observables":
        d = vocab.observable_descriptions()
        rows = [f"{k}({', '.join(s)}): {d[k]}" for k, s in vocab.observable_signatures().items()]
        rows.append("destabilizers(): the destabilizer generators (phases=true, up to 32 qubits)")
    elif what == "probes":
        d = vocab.probe_descriptions()
        rows = [f"{k}({', '.join(s)}): {d[k]}" for k, s in vocab.probe_signatures().items()]
    elif what == "code_properties":
        from sparsegf2.agent.tools.code import CODE_PROPERTIES

        rows = [f"{k}({', '.join(p)}): {desc}" for k, (p, desc, _) in CODE_PROPERTIES.items()]
    elif what == "metrics":
        from sparsegf2.agent.tools.analysis import METRICS

        rows = [f"{k}({', '.join(p)}): {desc}" for k, (p, desc, _) in METRICS.items()]
    elif what == "graph_families":
        rows = [f"{k}({', '.join('graph_' + p for p in s)})" for k, s in vocab.graph_families().items()]
        rows.append("file(graph_file): any geometry from a .json/.npz collection (graph_build saves one)")
    elif what == "edge_weight_families":
        rows = [f"{k}({', '.join(s)})" for k, s in vocab.weight_families().items()]
    elif what == "protocol_parameters":
        from sparsegf2.agent.params import PROTOCOL_PARAMS

        rows = [f"{p.name}: {p.description}" for p in PROTOCOL_PARAMS]
    else:  # stacks
        from sparsegf2.agent import stacks

        rows = [f"{r['name']} ({r['kind']}): {len(r.get('tools', []))} tools" + (
            f" - {r['description']}" if r.get("description") else "")
            for r in stacks.catalog(ctx.root)]
    return {"what": what, "rows": rows}


def _render_topic(data: dict[str, Any], ctx) -> str:
    return f"help_topic: {data['topic']}\n{data['text']}"


def _render_list(data: dict[str, Any], ctx) -> str:
    return f"help_list: {data['what']} ({len(data['rows'])})\n" + "\n".join(data["rows"])


register(ToolDef(
    name="help_topic", stack="help", summary="Guidance on one topic, with examples.",
    description=(
        "Read short guidance on a topic before using tools you have not used yet: the "
        "conventions (qubits, layers, seeds, units), the call grammar for probes and "
        "observables, and how each kind of study works. Example: {\"topic\": \"conventions\"}"
    ),
    params=(Param("topic", "string", "Which topic.", required=True, enum=TOPICS),),
    handler=_topic, render=_render_topic,
    examples=({"topic": "conventions"}, {"topic": "grammar"}),
))

register(ToolDef(
    name="help_list", stack="help", summary="Exact names: tools, gates, probes, observables, ...",
    description=(
        "List exact names with their arguments and one-line meanings, generated from the "
        "simulator itself: tools, gates, instructions, observables, probes, code_properties, "
        "metrics, graph_families, edge_weight_families, protocol_parameters, stacks. "
        "Example: {\"what\": \"probes\"}"
    ),
    params=(Param("what", "string", "Which list.", required=True, enum=LISTS),),
    handler=_list, render=_render_list,
    examples=({"what": "probes"}, {"what": "observables"}),
))
```

(`code_properties` and `metrics` import modules from Tasks 21 and 25. Until those exist, the `test_lists` parameter set above leaves them out. Task 21 and Task 25 add their own `help_list` test cases.)

- [ ] **Step 7: Write the 18 topic files `src/sparsegf2/agent/topics/<topic>.md`**

Each is 150-400 words of plain markdown that the model reads as is. Write them from the content below. Every fact is from `AGENTS.md`, `README.md` and the spec; copy numbers exactly.

- `overview.md`: what SparseGF2 is (fast stabilizer simulator for monitored random Clifford circuits; Stim-language circuits; verified against Stim). The stacks and what each is for, one line per stack (spec 6). How to start: `help_topic(conventions)`, `help_list(tools)`. Long work returns a job id.
- `conventions.md`:
  - Qubits: system qubits `0..n-1`, references at `n..`.
  - A layer is the gates followed by measurements, and is never rescaled; layer 0 is the state after setup and scramble.
  - Pictures: none / purification / single.
  - Seeds: circuit `seed` versus `graph_seed`; trajectory t depends only on (protocol, seed, t).
  - Phases: ensembles are phase-free; states default to `phases=true`.
  - Entropies are in bits, as exact integers.
  - Qubit lists `0:8` (slice semantics) or `[0,3,5]`.
  - Paths are under the root; default output dirs are `runs/<tool>-<hash>`, and running the same request again resumes.
  - Tau is -1 when censored; never average it.
- `grammar.md`: the grammar of spec 5.1 with the examples table: names from `help_list`, positional order, `@checkpoints` for probes, ranges, lists, Pauli words. Two common errors and their messages.
- `states.md`: the `state_new` / `state_apply` / `state_query` / `state_save` / `state_close` workflow. Circuit text applies to the state. Measurement outcomes require `phases=true`. Observables are call strings. Example: a Bell pair, then `entropy(0:1)`.
- `circuits.md`: Stim-language text, the `CLIFFORD_2Q(i) a b` extension, `circuit_inspect` before `circuit_run`, `shots`, `observe_ticks`, `circuit_convert` to Stim, and `circuit_crosscheck` (noise-free circuits only).
- `ensembles.md`: the protocol parameters (setup, scramble, layers, schedules). Graph collections: `graph_count` x `circuits_per_graph`, with trajectory t on graph `t // circuits_per_graph`. Checkpoints. `estimate_only`. Background jobs. Sweeps via `ensemble_sweep` with `grid`.
- `probes.md`: every probe with its meaning (from AGENTS.md "Probes"). Per-probe `@checkpoints`. Costs at n = 256 (first-failure with M = 100: 3-7 ms; counts with w = 2: about 1 ms, w = 3: 40-50 ms, w = 4: about 7 s). Values after `stop_when_purified`.
- `trajectories.md`: `traj_run` (stores the code by default; `store=state` for `state_new`; `compare_stim`; `save_circuit`). `traj_probe` replays the trajectory exactly with new probes and depths. The randomness contract.
- `codes.md`: `code_analyze` sources (`stabilizers`, saved results, open states) and its properties (`help_list(what=code_properties)`). l(A), correctability, distance bounds.
- `erasure.md`: the definitions (copy the "Erasure protection" block of the README):
  - l(A) = I(A:R);
  - the random ordering pi, the nested regions A_m and D_C = min{m : l(A_m) > 0};
  - P_C(m) = Pr[D_C <= m];
  - Phi_C(e) = E_D Pr[Bin(n,e) >= D], estimated by the mean over M orderings;
  - the hybrid estimator with exhaustive counts;
  - e_delta;
  - mu_D;
  - distance bounds.

  The workflow: `ensemble_run` with probes `code_dimension`, `erasure_first_failure(samples=512)` and `uncorrectable_region_counts(2)`, in the purification picture, then `analysis_erasure`.
- `purification.md`: the single reference; `stop_when_purified`; tau and censoring; survival, Kaplan-Meier, `tau_median_all`; `analysis_purification`; finding the transition via a sweep over `measurement_rate` and `analysis_compare(metrics=[survival(layer=...)])`.
- `entanglement.md`: `entropy`, `half_cut_entropy`, `mutual_information`, `tripartite_information` (-I3 crossing near p_c ~ 0.16 for brickwork on a ring), and `entropy_profile`.
- `graphs.md`:
  - The families with their `graph_*` parameters; `graph_k` is the total degree, even, with K/2 neighbours per side.
  - Collections are seeded `(graph_seed, i)`.
  - `graph=file` with `graph_file`. The JSON format: `{"num_nodes": n, "graphs": [[[a,b],...],...], "weights": [...]}`.
  - `graph_build`, and `graph_stats` (lambda_2, path length, clustering, shortcuts).
- `analysis.md`: `analysis_summary`, `analysis_probe` (statistics, `where`, `trajectories`, `graphs`), `analysis_erasure`, `analysis_purification`, and `analysis_compare` with the metrics list.
- `files.md`: the artifact kinds and their files (the spec 5.3 Source table), `result_list`, `result_get` (bounded raw values), `result_merge` (chunks from a cluster), the reading side in Python (`EnsembleResult.load`).
- `jobs.md`: `background=auto|always|never`; `job_status(wait_seconds<=60)`; a finished job's status contains the result; cancel; restarts lose job ids but not saved chunks, and running the same request again resumes.
- `stacks.md`: built-in stacks, the bundles `all` and `core`, the presets, custom stacks (`sparsegf2 stacks save ...`), and the fact that fixed parameters disappear from the schema. A server serves one stack, chosen at launch.
- `recipes.md`: three short recipes, as numbered tool-call sequences with arguments:
  1. A purification transition: a sweep over `measurement_rate`, then a survival comparison.
  2. An erasure study on a Watts-Strogatz collection: `ensemble_run`, then `analysis_erasure`, then `traj_run` on one interesting trajectory, then `code_analyze` with more orderings.
  3. Checking a circuit: `circuit_inspect`, then `circuit_run`, then `circuit_crosscheck`.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `python -m pytest tests/agent -q`
Expected: PASS (the 18 topics render; the help stack serves over MCP)

- [ ] **Step 9: Commit, and run the whole suite (end of Phase 2)**

```bash
ruff check src tests && ruff format src tests
python -m pytest -q
git add src/sparsegf2/agent tests/agent
git commit -m "agent: dispatch, low-level MCP server with flags, help tools and topic guides"
```

---
## Phase 3: The tool stacks

Every tool module follows the same pattern. It defines handlers `(ctx, args) -> dict` and renderers `(data, ctx) -> str`, then calls `register(ToolDef(...))` with at least one example, and adds one import line to `agent/tools/__init__.py` (in the order of `stacks.STACKS`). Tests go through the `run_tool` fixture (dispatch: validation, rendering, caps) and check the physics against the library called directly.

### Task 15: Job tools

**Files:**
- Create: `src/sparsegf2/agent/tools/job.py`
- Modify: `src/sparsegf2/agent/tools/__init__.py` (add `job`), `tests/agent/test_stacks.py` (remove the `skipif` on `test_cli_save_show_delete`)
- Test: `tests/agent/test_job_tools.py`

**Interfaces:**
- Consumes: `ctx.jobs` (`JobStore.status/cancel/list`), `sessions.render_job`, `dispatch.call`.
- Produces: the tools `job_status(job_id, wait_seconds=0)`, `job_cancel(job_id)` and `job_list()`.

- [ ] **Step 1: Write the failing tests**

```python
"""job_status, job_cancel, job_list, and generic background runs through dispatch."""

import threading

from sparsegf2.agent import dispatch, sessions
from sparsegf2.agent.registry import Bound, Param, ToolDef

GATE = threading.Event()


def _slow(ctx, args):
    while not GATE.wait(0.01):
        if sessions.cancelled():
            return {"value": -1}
    return {"value": args["x"] * 2}


SLOW = ToolDef(
    "plot_slow", "plot", "slow", "slow", (Param("x", "integer", "x", required=True),),
    _slow, lambda d, ctx: f"plot_slow: {d['value']}", long_running=True,
    estimate=lambda ctx, a: 1e9,
)


def test_background_job_then_status_returns_the_result(ctx, run_tool):
    GATE.clear()
    started = dispatch.call(ctx, Bound(SLOW), {"x": 21})
    assert not started.is_error and started.text.startswith("job ")
    job_id = started.data["job_id"]
    again = dispatch.call(ctx, Bound(SLOW), {"x": 21})  # same request while running
    assert again.data["job_id"] == job_id and "already running" in again.text
    GATE.set()
    done = run_tool("job_status", job_id=job_id, wait_seconds=10)
    assert "completed" in done.text and "plot_slow: 42" in done.text


def test_never_runs_in_place(ctx):
    GATE.set()
    out = dispatch.call(ctx, Bound(SLOW), {"x": 1, "background": "never"})
    assert out.text == "plot_slow: 2"


def test_cancel_and_list(ctx, run_tool):
    GATE.clear()
    job_id = dispatch.call(ctx, Bound(SLOW), {"x": 5}).data["job_id"]
    out = run_tool("job_cancel", job_id=job_id)
    assert not out.is_error
    assert "cancelled" in run_tool("job_status", job_id=job_id, wait_seconds=10).text
    listing = run_tool("job_list")
    assert job_id in listing.text
    GATE.set()


def test_unknown_job(run_tool):
    out = run_tool("job_status", job_id="zzz")
    assert out.is_error and "unknown job_id 'zzz'" in out.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_job_tools.py -q`
Expected: FAIL with `KeyError: 'job_status'`

- [ ] **Step 3: Implement `src/sparsegf2/agent/tools/job.py`**

```python
"""job_status, job_cancel, job_list: background work started by any long-running tool."""

from __future__ import annotations

from typing import Any

from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.render import fmt, table
from sparsegf2.agent.sessions import render_job

JOB_ID = Param("job_id", "string", "The job_id a tool returned.", required=True)


def _status(ctx, a: dict[str, Any]) -> dict[str, Any]:
    return ctx.jobs.status(a["job_id"], a.get("wait_seconds", 0.0))


def _cancel(ctx, a: dict[str, Any]) -> dict[str, Any]:
    return ctx.jobs.cancel(a["job_id"])


def _list(ctx, a: dict[str, Any]) -> dict[str, Any]:
    return {"jobs": ctx.jobs.list()}


def _render_list(data: dict[str, Any], ctx) -> str:
    rows = [
        [j["job_id"], j.get("tool", "ensemble"), j["status"],
         (f"{j['progress']['done']}/{j['progress']['total']} {j['progress']['unit']}"
          if j.get("progress") else
          f"{fmt(j.get('trajectories_done'))}/{fmt(j.get('trajectories_total'))} trajectories"),
         j.get("output_dir") or "-"]
        for j in data["jobs"]
    ]
    if not rows:
        return "job_list: no jobs in this server process"
    return f"job_list: {len(rows)} job(s)\n" + table(["job_id", "tool", "status", "progress", "output_dir"], rows)


register(ToolDef(
    name="job_status", stack="job", summary="Progress or result of a background job.",
    description=(
        "Progress of a background job (queued, running, completed, cancelled, failed). With "
        "wait_seconds (0-60) it waits for the job to finish first. A finished job's status "
        "contains the tool's result. Example: {\"job_id\": \"3f2a9c1b7d4e\", \"wait_seconds\": 30}"
    ),
    params=(JOB_ID, Param("wait_seconds", "number", "Wait up to this long for the job to end.",
                          default=0.0, minimum=0, maximum=60)),
    handler=_status, render=lambda d, ctx: render_job(d),
    examples=(),
))

register(ToolDef(
    name="job_cancel", stack="job", summary="Stop a background job (saved work is kept).",
    description=(
        "Stop a background job after its current chunk or point. Saved results are kept; "
        "running the same request again resumes. Example: {\"job_id\": \"3f2a9c1b7d4e\"}"
    ),
    params=(JOB_ID,), handler=_cancel, render=lambda d, ctx: render_job(d), read_only=False,
    examples=(),
))

register(ToolDef(
    name="job_list", stack="job", summary="Background jobs of this server process.",
    description="List the background jobs started in this server process. Example: {}",
    params=(), handler=_list, render=_render_list, examples=({},),
))
```

(`job_status` and `job_cancel` have no static examples, because they need a live job id. Task 28's example test skips tools whose examples tuple is empty but which are listed in `EXAMPLE_EXEMPT = {"job_status", "job_cancel", "state_apply", "state_query", "state_save", "state_close"}`, the tools that need live ids. Those tools are covered by the tests in their own tasks.)

In `tools/__init__.py`: `from sparsegf2.agent.tools import help, job  # noqa: F401` (later tasks insert their modules in stack order: `help, state, circuit, ensemble, traj, code, graph, analysis, result, job, plot`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent -q`
Expected: PASS (including the now-unskipped CLI stacks test)

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent/tools tests/agent
git commit -m "agent: job tools; generic background runs with dedup of identical requests"
```

---

### Task 16: Circuit tools

**Files:**
- Create: `src/sparsegf2/agent/tools/circuit.py`
- Modify: `src/sparsegf2/agent/render.py` (add `kv`), `src/sparsegf2/agent/tools/__init__.py`
- Test: `tests/agent/test_circuit_tools.py`

**Interfaces:**
- Consumes: `api.inspect_circuit`, `api.run_circuit`, `params.observable_dicts`, `crosscheck.compare_stabilizers/compare_measurements`, `Circuit`, `Circuit.to_stim`.
- Produces:
  - The tools `circuit_inspect`, `circuit_run`, `circuit_convert` and `circuit_crosscheck`.
  - `render.kv(data, keys=None) -> str`, which gives `key: value` lines for scalars and short lists.
  - `tools.circuit.circuit_text(ctx, args) -> str`, which reads exactly one of `circuit` / `circuit_file`.

- [ ] **Step 1: Write the failing tests**

```python
"""circuit_inspect, circuit_run, circuit_convert, circuit_crosscheck."""

GHZ = "H 0\nCX 0 1\nTICK\nCX 1 2\nTICK\nM 0 1 2\nDETECTOR rec[-1] rec[-2]"


def test_inspect_valid_and_invalid(run_tool):
    out = run_tool("circuit_inspect", circuit=GHZ)
    assert out.text.startswith("circuit_inspect: valid") and "num_qubits: 3" in out.text
    bad = run_tool("circuit_inspect", circuit="H 0\nFOO 1")
    assert "line 2: unknown gate 'FOO'" in bad.text and not bad.is_error


def test_run_single_shot_with_observables_per_tick(run_tool):
    out = run_tool("circuit_run", circuit=GHZ, seed=1,
                   observables=["entropy(0:1)", "stabilizers"], observe_ticks=True)
    assert not out.is_error, out.text
    assert out.text.startswith("circuit_run: 3 qubits, 3 measurements, 1 shot")
    assert "measurements: " in out.text and "entropy" in out.text


def test_run_many_shots(run_tool):
    out = run_tool("circuit_run", circuit=GHZ, shots=200, seed=2)
    assert "measurement_means" in out.text and "detector_rates" in out.text


def test_circuit_file_and_exactly_one_source(run_tool, ctx):
    (ctx.root / "c.stim").write_text(GHZ, encoding="utf-8")
    assert not run_tool("circuit_inspect", circuit_file="c.stim").is_error
    assert "exactly one of circuit or circuit_file" in run_tool("circuit_inspect").text
    assert "exactly one" in run_tool("circuit_inspect", circuit=GHZ, circuit_file="c.stim").text


def test_convert_expands_cliffords(run_tool, ctx):
    out = run_tool("circuit_convert", circuit="CLIFFORD_2Q(77) 0 1\nM 0", to="stim")
    assert "CLIFFORD_2Q" not in out.text.split("\n", 1)[1]
    run_tool("circuit_convert", circuit=GHZ, to="stim", output_file="out.stim")
    assert (ctx.root / "out.stim").read_text(encoding="utf-8").startswith("H 0")


def test_crosscheck(run_tool):
    out = run_tool("circuit_crosscheck", circuit=GHZ.replace("DETECTOR", "TICK\nDETECTOR"), seed=3)
    assert "stabilizers: match" in out.text and "measurements: match" in out.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_circuit_tools.py -q`
Expected: FAIL with `KeyError: 'circuit_inspect'`

- [ ] **Step 3: Add `kv` to `src/sparsegf2/agent/render.py` (and to `__all__`)**

```python
def kv(data: dict, keys: Sequence[str] | None = None, *, max_list: int = 16) -> str:
    """``key: value`` lines for scalars and short lists (longer lists are summarised)."""
    out = []
    for key in keys if keys is not None else data:
        if key not in data or data[key] is None:
            continue
        v = data[key]
        if isinstance(v, dict):
            continue
        if isinstance(v, (list, tuple, np.ndarray)):
            items = list(v)
            shown = " ".join(fmt(x) for x in items[:max_list])
            more = f" ... ({len(items)} values)" if len(items) > max_list else ""
            out.append(f"{key}: {shown}{more}")
        else:
            out.append(f"{key}: {fmt(v)}")
    return "\n".join(out)
```

- [ ] **Step 4: Implement `src/sparsegf2/agent/tools/circuit.py`**

```python
"""Stim-language circuits: inspect, run, convert, cross-check against Stim."""

from __future__ import annotations

from typing import Any

from sparsegf2 import api
from sparsegf2.agent.params import OUTPUT_DIR, SEED, observable_dicts
from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.render import kv, lines, table
from sparsegf2.circuit import Circuit

CIRCUIT = Param("circuit", "string",
                "Circuit text in Stim's language (CLIFFORD_2Q(i) a b is also accepted).")
CIRCUIT_FILE = Param("circuit_file", "string", "A circuit file under the root, instead of circuit.")


def circuit_text(ctx, a: dict[str, Any]) -> str:
    if ("circuit" in a) == ("circuit_file" in a):
        raise ValueError("give exactly one of circuit or circuit_file")
    if "circuit" in a:
        return a["circuit"]
    return ctx.path(a["circuit_file"], what="circuit_file", must_exist=True).read_text(
        encoding="utf-8")


def _inspect(ctx, a):
    return api.inspect_circuit(circuit_text(ctx, a))


def _render_inspect(d, ctx):
    if not d.get("valid"):
        errs = "\n".join(f"line {e['line']}: {e['message']}" for e in d.get("errors", []))
        return f"circuit_inspect: invalid\n{errs}"
    hist = sorted(d["instruction_histogram"].items(), key=lambda kv_: -kv_[1])[:12]
    return lines(
        "circuit_inspect: valid",
        kv(d, ["num_qubits", "num_measurements", "num_detectors", "num_observables",
               "num_ticks", "num_operations", "stim_compatible"]),
        "instructions: " + ", ".join(f"{k} x{v}" for k, v in hist),
        "next: circuit_run(circuit=...)",
    )


def _run(ctx, a):
    out_dir = str(ctx.path(a["output_dir"], what="output_dir")) if a.get("output_dir") else None
    return api.run_circuit(
        circuit_text(ctx, a), shots=a["shots"], phases=a["phases"], seed=a.get("seed"),
        observables=observable_dicts(a["observables"]) if a.get("observables") else None,
        observe_ticks=a["observe_ticks"], tick_stride=a.get("tick_stride"),
        num_qubits=a.get("num_qubits"), output_dir=out_dir, command="mcp:circuit_run",
    )


def _render_run(d, ctx):
    shots = d["shots"]
    head = (f"circuit_run: {d['num_qubits']} qubits, {d['num_measurements']} measurements, "
            f"{shots} shot{'s' if shots != 1 else ''}, seed {d['seed']}, phases {str(d['phases']).lower()}")
    body = kv(d, ["measurements", "random", "detectors", "observables_record", "random_fraction",
                  "measurement_mean", "measurement_means", "first_shots", "detector_rate_mean",
                  "detector_rate_max", "detector_rates", "any_detector_fraction",
                  "observable_rates"])
    obs = d.get("observables")
    obs_text = ""
    if isinstance(obs, dict) and obs:
        obs_text = "observables (final state):\n" + "\n".join(f"  {k}: {v}" for k, v in obs.items())
    ticks = d.get("tick_observables")
    tick_text = ""
    if ticks:
        names = list(ticks["values"])
        rows = [[t, *[ticks["values"][n][i] for n in names]] for i, t in enumerate(ticks["ticks"])]
        tick_text = "per tick:\n" + table(["tick", *names], rows[:40])
    files = f"saved: {d['files']}" if d.get("files") else ""
    return lines(head, body, obs_text, tick_text, files)


def _convert(ctx, a):
    circuit = Circuit(circuit_text(ctx, a))
    text = str(circuit.to_stim()) if a["to"] == "stim" else str(circuit)
    out = {"to": a["to"], "text": text}
    if a.get("output_file"):
        path = ctx.path(a["output_file"], what="output_file")
        path.write_text(text + "\n", encoding="utf-8")
        out["output_file"] = ctx.show(path)
    return out


def _render_convert(d, ctx):
    if d.get("output_file"):
        return f"circuit_convert: wrote {d['to']} text to {d['output_file']}"
    return f"circuit_convert: {d['to']} text\n{d['text']}"


def _crosscheck(ctx, a):
    from sparsegf2 import crosscheck

    circuit = Circuit(circuit_text(ctx, a))
    out: dict[str, Any] = {}
    if a["compare"] in ("stabilizers", "both"):
        out["stabilizers"] = crosscheck.compare_stabilizers(
            circuit, num_qubits=a.get("num_qubits"), phases=a["phases"],
            ticks=a.get("at_ticks"), seed=a["seed"])
    if a["compare"] in ("measurements", "both"):
        out["measurements"] = crosscheck.compare_measurements(
            circuit, num_qubits=a.get("num_qubits"), seed=a["seed"])
    return out


def _render_crosscheck(d, ctx):
    parts = ["circuit_crosscheck: SparseGF2 vs Stim"]
    if "stabilizers" in d:
        s = d["stabilizers"]
        parts.append(f"stabilizers: {'match' if s['match'] else 'MISMATCH'} "
                     f"({s['ticks_compared']} ticks compared)")
        if not s["match"]:
            m = s["first_mismatch"]
            parts.append(f"  first at tick {m['tick']}:\n  sparsegf2 {m['sparsegf2'][:8]}\n  stim      {m['stim'][:8]}")
    if "measurements" in d:
        m = d["measurements"]
        parts.append(f"measurements: {'match' if m['match'] else 'MISMATCH'} ({m['measurements']} "
                     f"measurements, {m['detectors']} detectors, {m['observables']} observables)")
        if not m["match"]:
            parts.append(f"  first: {m['first_mismatch']}")
    return "\n".join(parts)


register(ToolDef(
    name="circuit_inspect", stack="circuit", summary="Parse a circuit; counts or errors by line.",
    description=(
        "Check a Stim-language circuit without running it: qubit, measurement, detector and "
        "tick counts and an instruction histogram, or every bad line with its line number. "
        "Example: {\"circuit\": \"H 0\\nCX 0 1\\nM 0 1\"}"
    ),
    params=(CIRCUIT, CIRCUIT_FILE), handler=_inspect, render=_render_inspect,
    examples=({"circuit": "H 0\nCX 0 1\nM 0 1"},),
))

register(ToolDef(
    name="circuit_run", stack="circuit", summary="Run a circuit: record, rates, observables.",
    description=(
        "Run a Stim-language circuit from |0...0>. shots=1 gives the measurement record; more "
        "shots give per-measurement means and detector/observable rates. observables are call "
        "strings evaluated on the final state, and after every TICK with observe_ticks "
        "(help_list(what=observables)). Example: {\"circuit\": \"H 0\\nCX 0 1\\nTICK\\nM 0\", "
        "\"seed\": 1, \"observables\": [\"entropy(0:1)\"], \"observe_ticks\": true}"
    ),
    params=(
        CIRCUIT, CIRCUIT_FILE,
        Param("shots", "integer", "Number of shots.", default=1, minimum=1),
        Param("phases", "boolean", "Track signs (outcomes); false is faster, no outcomes.",
              default=True),
        SEED,
        Param("observables", "array", "Observable calls, e.g. entropy(0:4), expectation(Z0*Z1).",
              items="string"),
        Param("observe_ticks", "boolean", "Also evaluate observables after every TICK.",
              default=False),
        Param("tick_stride", "integer", "With observe_ticks: every k-th TICK only.", minimum=1),
        Param("num_qubits", "integer", "Simulate extra idle qubits up to this count.", minimum=1),
        OUTPUT_DIR,
    ),
    handler=_run, render=_render_run, long_running=True,
    examples=({"circuit": "H 0\nCX 0 1\nTICK\nM 0", "seed": 1, "observables": ["entropy(0:1)"],
               "observe_ticks": True},),
))

register(ToolDef(
    name="circuit_convert", stack="circuit", summary="Convert to Stim text (expands CLIFFORD_*).",
    description=(
        "Convert a circuit to Stim's language (to=stim: CLIFFORD_1Q/2Q expanded into Stim "
        "gates) or to SparseGF2's normalised text (to=canonical); optionally write a file. "
        "Example: {\"circuit\": \"CLIFFORD_2Q(77) 0 1\", \"to\": \"stim\"}"
    ),
    params=(CIRCUIT, CIRCUIT_FILE,
            Param("to", "string", "Target text.", default="stim", enum=("stim", "canonical")),
            Param("output_file", "string", "Write the text to this file under the root.")),
    handler=_convert, render=_render_convert, read_only=False,
    examples=({"circuit": "CLIFFORD_2Q(77) 0 1\nM 0", "to": "stim"},),
))

register(ToolDef(
    name="circuit_crosscheck", stack="circuit", summary="Compare with Stim (stabilizers, records).",
    description=(
        "Run a noise-free circuit in SparseGF2 and in Stim and compare the canonical "
        "stabilizer groups after every TICK (or at_ticks) and, replaying Stim's random "
        "outcomes, every measurement, detector and observable. Needs Stim installed. "
        "Example: {\"circuit\": \"H 0\\nTICK\\nCX 0 1\\nTICK\\nM 0 1\"}"
    ),
    params=(
        CIRCUIT, CIRCUIT_FILE,
        Param("compare", "string", "What to compare.", default="both",
              enum=("stabilizers", "measurements", "both")),
        Param("at_ticks", "array", "Ticks to compare (default every tick).", items="integer"),
        Param("phases", "boolean", "Compare signs too (stabilizers).", default=True),
        Param("seed", "integer", "Stim's seed.", default=0, minimum=0),
        Param("num_qubits", "integer", "Total qubits (default: those the circuit uses).",
              minimum=1),
    ),
    handler=_crosscheck, render=_render_crosscheck, long_running=True, requires=("stim",),
    examples=({"circuit": "H 0\nTICK\nCX 0 1\nTICK\nM 0 1"},),
))
```

Add `circuit` to `tools/__init__.py`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_circuit_tools.py -q`
Expected: PASS. If `circuit_run`'s output differs from the keys used above, check `api.run_circuit`'s return dict. The keys `measurements`, `random`, `detectors`, `measurement_means`, `detector_rates`, `observables` and `tick_observables` exist; `kv` skips any key that is absent.

- [ ] **Step 6: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent tests/agent/test_circuit_tools.py
git commit -m "agent: circuit tools (inspect, run with observable calls, convert, Stim cross-check)"
```

---

### Task 17: State tools (stabilizer states gate by gate)

**Files:**
- Create: `src/sparsegf2/agent/tools/state.py`
- Modify: `src/sparsegf2/agent/tools/__init__.py`
- Test: `tests/agent/test_state_tools.py`

**Interfaces:**
- Consumes: `ctx.states` (`StateStore`), `Simulator`, `Simulator.save/load` (Task 5), `Circuit`, `api._compile_observables`, `params.observable_dicts`.
- Produces: the tools `state_new`, `state_apply`, `state_query`, `state_save`, `state_close` and `state_list`.

- [ ] **Step 1: Write the failing tests**

```python
"""state_* tools: a Bell pair, queries, save/load, phase-free limits."""

import sparsegf2 as sg


def test_bell_pair_session(run_tool, ctx):
    new = run_tool("state_new", num_qubits=3, seed=4)
    assert new.text.startswith("state_new: s1, 3 qubits")
    applied = run_tool("state_apply", state_id="s1", circuit="H 0\nCX 0 1\nM 0")
    assert "1 measurement" in applied.text and "random: 1" in applied.text
    q = run_tool("state_query", state_id="s1",
                 observables=["entropy(0:1)", "expectation(Z0*Z1)", "stabilizers", "destabilizers"])
    assert "entropy(0:1): 0" in q.text and "expectation(Z0*Z1): 1" in q.text
    saved = run_tool("state_save", state_id="s1", path="bell.state.npz")
    assert (ctx.root / "bell.state.npz").exists() and not saved.is_error
    again = run_tool("state_new", source="bell.state.npz")
    assert again.text.startswith("state_new: s2")
    s1, s2 = ctx.states.get("s1"), ctx.states.get("s2")
    assert [str(p) for p in s1.stabilizers()] == [str(p) for p in s2.stabilizers()]
    assert "s1" in run_tool("state_list").text
    run_tool("state_close", state_id="s1")
    assert "unknown state_id 's1'" in run_tool("state_query", state_id="s1",
                                               observables=["entropy(0:1)"]).text


def test_phase_free_has_no_expectations(run_tool):
    run_tool("state_new", num_qubits=2, phases=False)
    out = run_tool("state_query", state_id="s1", observables=["expectation(Z0)"])
    assert out.is_error and "needs sign tracking" in out.text


def test_needs_exactly_one_source(run_tool):
    assert "exactly one of num_qubits or source" in run_tool("state_new").text


def test_save_suffix(run_tool):
    run_tool("state_new", num_qubits=1)
    assert ".state.npz" in run_tool("state_save", state_id="s1", path="x.npz").text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_state_tools.py -q`
Expected: FAIL with `KeyError: 'state_new'`

- [ ] **Step 3: Implement `src/sparsegf2/agent/tools/state.py`**

```python
"""Stabilizer states held by the server, acted on with circuit text and queried with observables."""

from __future__ import annotations

from typing import Any

import numpy as np

from sparsegf2 import api
from sparsegf2.agent.params import SEED, observable_dicts
from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.render import fmt, kv, lines, table
from sparsegf2.circuit import Circuit
from sparsegf2.simulator import Simulator

STATE_ID = Param("state_id", "string", "An open state's id (state_new returns it).", required=True)


def _new(ctx, a):
    if ("num_qubits" in a) == ("source" in a):
        raise ValueError("give exactly one of num_qubits or source (a .state.npz file)")
    if "source" in a:
        path = ctx.path(a["source"], what="source", must_exist=True)
        sim = Simulator.load(path)
        origin = f"loaded {ctx.show(path)}"
    else:
        sim = Simulator(a["num_qubits"], phases=a["phases"], seed=a.get("seed"))
        origin = "new"
    state_id = ctx.states.add(sim, origin)
    return {**ctx.states.info(state_id), "seed": str(sim.seed.entropy)}


def _render_new(d, ctx):
    return (f"state_new: {d['state_id']}, {d['num_qubits']} qubits, phases "
            f"{str(d['phases']).lower()} ({d['origin']})\n"
            f'next: state_apply(state_id="{d["state_id"]}", circuit="H 0\\nCX 0 1")')


def _apply(ctx, a):
    sim = ctx.states.get(a["state_id"])
    result = sim.run(Circuit(a["circuit"]))
    m = int(np.asarray(result.random).size)
    ctx.states.count_measurements(a["state_id"], m)
    out: dict[str, Any] = {"state_id": a["state_id"], "measurements": m,
                           "random": np.asarray(result.random).astype(int).tolist()}
    if sim.phases:
        out["outcomes"] = np.asarray(result.measurements).astype(int).tolist()
        if np.asarray(result.detectors).size:
            out["detectors"] = np.asarray(result.detectors).astype(int).tolist()
    return out


def _render_apply(d, ctx):
    head = f"state_apply: {d['state_id']}, {d['measurements']} measurement" + (
        "s" if d["measurements"] != 1 else "")
    return lines(head, kv(d, ["outcomes", "random", "detectors"], max_list=64))


def _destabilizers(sim) -> list[str]:
    if not sim.phases:
        raise ValueError("destabilizers exist only with phases=true")
    if sim.num_qubits > api.MAX_STABILIZER_QUBITS:
        raise ValueError(f"destabilizers are listed only up to {api.MAX_STABILIZER_QUBITS} qubits")
    return [str(p) for p in sim.destabilizers()]


def _query(ctx, a):
    sim = ctx.states.get(a["state_id"])
    items = list(a["observables"])
    extra = [i for i, s in enumerate(items) if s.strip() == "destabilizers"]
    rest = [s for i, s in enumerate(items) if i not in extra]
    compiled = api._compile_observables(observable_dicts(rest), sim.num_qubits, sim.phases, 0)
    values: list[tuple[str, Any]] = []
    it = iter(compiled)
    for i, text in enumerate(items):
        if i in extra:
            values.append(("destabilizers", _destabilizers(sim)))
            continue
        _name, fn, show = next(it)
        values.append((text.strip(), show(fn(sim))))
    return {"state_id": a["state_id"], "values": [{"observable": k, "value": v} for k, v in values]}


def _render_query(d, ctx):
    out = [f"state_query: {d['state_id']}"]
    for item in d["values"]:
        v = item["value"]
        if isinstance(v, list) and v and isinstance(v[0], str):
            out.append(f"{item['observable']}: " + " ".join(v))
        elif isinstance(v, dict):
            out.append(f"{item['observable']}: " + ", ".join(f"{k}={fmt(x) if not isinstance(x, (list, dict)) else x}" for k, x in v.items()))
        else:
            out.append(f"{item['observable']}: {fmt(v) if not isinstance(v, (list, dict)) else v}")
    return "\n".join(out)


def _save(ctx, a):
    path = ctx.path(a["path"], what="path")
    if not path.name.endswith(".state.npz"):
        raise ValueError("state files end in .state.npz (e.g. bell.state.npz)")
    path.parent.mkdir(parents=True, exist_ok=True)
    ctx.states.get(a["state_id"]).save(path)
    return {"state_id": a["state_id"], "path": ctx.show(path)}


def _close(ctx, a):
    return ctx.states.close(a["state_id"])


def _list(ctx, a):
    return {"states": ctx.states.list()}


def _render_list(d, ctx):
    if not d["states"]:
        return "state_list: no open states"
    rows = [[s["state_id"], s["num_qubits"], s["phases"], s["measurements"], s["origin"]]
            for s in d["states"]]
    return "state_list:\n" + table(["state_id", "qubits", "phases", "measurements", "origin"], rows)


register(ToolDef(
    name="state_new", stack="state", summary="Open a stabilizer state (|0..0> or from a file).",
    description=(
        "Open a stabilizer state held by the server: num_qubits qubits in |0...0>, or a state "
        "saved with state_save or by traj_run(store=state) (source). Returns a state_id for "
        "state_apply/state_query. Example: {\"num_qubits\": 4, \"seed\": 1}"
    ),
    params=(Param("num_qubits", "integer", "Qubits of a new |0...0> state.", minimum=1),
            Param("source", "string", "A .state.npz file under the root, instead."),
            Param("phases", "boolean", "Track signs (outcomes, expectations).", default=True),
            SEED),
    handler=_new, render=_render_new, read_only=False,
    examples=({"num_qubits": 4, "seed": 1},),
))

register(ToolDef(
    name="state_apply", stack="state", summary="Apply circuit text (gates, measurements, ...).",
    description=(
        "Apply Stim-language circuit text to an open state: gates, measurements, resets, MPP, "
        "noise, REPEAT. Returns the new measurement outcomes (with phases) and which were random. "
        "Example: {\"state_id\": \"s1\", \"circuit\": \"H 0\\nCX 0 1\\nM 0\"}"
    ),
    params=(STATE_ID, Param("circuit", "string", "Circuit text to apply.", required=True)),
    handler=_apply, render=_render_apply, read_only=False, long_running=True,
))

register(ToolDef(
    name="state_query", stack="state", summary="Observables of an open state.",
    description=(
        "Evaluate observables of an open state, as call strings: entropy(0:4), "
        "mutual_information(a=0:2, b=2:4), code_dimension(reference=4:8), expectation(Z0*Z1), "
        "stabilizers, destabilizers, ... (help_list(what=observables)). "
        "Example: {\"state_id\": \"s1\", \"observables\": [\"entropy(0:1)\", \"stabilizers\"]}"
    ),
    params=(STATE_ID, Param("observables", "array", "Observable calls.", required=True,
                            items="string")),
    handler=_query, render=_render_query, long_running=True,
))

register(ToolDef(
    name="state_save", stack="state", summary="Save an open state to a .state.npz file.",
    description=(
        "Save an open state exactly (tableau, signs, random-stream position) to a .state.npz "
        "file under the root; state_new(source=...) reopens it. "
        "Example: {\"state_id\": \"s1\", \"path\": \"states/bell.state.npz\"}"
    ),
    params=(STATE_ID, Param("path", "string", "Destination (.state.npz).", required=True)),
    handler=_save, render=lambda d, ctx: f"state_save: {d['state_id']} -> {d['path']}",
    read_only=False,
))

register(ToolDef(
    name="state_close", stack="state", summary="Close an open state.",
    description="Close an open state and free its memory. Example: {\"state_id\": \"s1\"}",
    params=(STATE_ID,), handler=_close,
    render=lambda d, ctx: f"state_close: closed {d['state_id']}", read_only=False,
))

register(ToolDef(
    name="state_list", stack="state", summary="Open states of this server.",
    description="List the open states (id, qubits, phases, measurements so far). Example: {}",
    params=(), handler=_list, render=_render_list, examples=({},),
))
```

Add `state` to `tools/__init__.py`. Note for Task 28: `state_apply`, `state_query`, `state_save` and `state_close` are in `EXAMPLE_EXEMPT`, because they need a live `state_id`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_state_tools.py -q`
Expected: PASS. `RunResult.random` and `RunResult.measurements` are arrays over the measurements; with `phases=False`, `measurements` raises `PhaseFreeError`, which is why the handler reads it only when `sim.phases`.

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent tests/agent/test_state_tools.py
git commit -m "agent: state tools (open, apply circuit text, query observables, save/load, close)"
```

---

### Task 18: `ensemble_run`

**Files:**
- Create: `src/sparsegf2/agent/tools/ensemble.py`
- Modify: `src/sparsegf2/agent/tools/__init__.py`
- Test: `tests/agent/test_ensemble_tools.py`

**Interfaces:**
- Consumes:
  - from `params`: `PROTOCOL_PARAMS`, `GRAPH_PARAMS`, `PROBES`, `CHECKPOINTS`, `SEED`, `OUTPUT_DIR`, `ensemble_spec`;
  - from `api`: `run_ensemble`, `estimate_ensemble`, `start_ensemble_job`, `load_results`, `job_status`, `_build_spec`, `_canonical`, `_JOBS`, `_JOBS_LOCK`;
  - from `sessions`: `from_api_status`.
- Produces:
  - The tool `ensemble_run`.
  - In `tools.ensemble`:
    - `ENSEMBLE_PARAMS: tuple[Param, ...]`;
    - `render_summary(title, summary, *, output_dir=None, extra=()) -> str`, which Task 23 reuses;
    - `prepare(ctx, args, tool) -> tuple[dict, MonitoredCircuit, Path]`;
    - `start_or_join(ctx, spec, args, out, command) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
"""ensemble_run: flat arguments, the same numbers as the library, reuse, jobs."""

import numpy as np

from sparsegf2 import graphs, mipt

ARGS = dict(num_system=8, layers=8, measurement_rate=0.15, graph="watts_strogatz",
            graph_count=3, graph_seed=2, circuits_per_graph=2, schedule="random_edges",
            reference="purification", checkpoints="4", seed=11,
            probes=["code_dimension", "erasure_first_failure(samples=16)"])


def test_matches_the_library(run_tool, ctx):
    out = run_tool("ensemble_run", **ARGS)
    assert not out.is_error, out.text
    assert out.text.startswith("ensemble_run: n=8, layers=8, p=0.15, 6 trajectories")
    saved = mipt.EnsembleResult.load(ctx.root / out.data["output_dir"] / "result.npz")
    direct = mipt.MonitoredCircuit(
        num_system=8, edges=[graphs.watts_strogatz(8, 4, 0.1, seed=(2, g)) for g in range(3)],
        circuits_per_graph=2, layers=8, measurement_rate=0.15, schedule="random_edges",
        reference="purification", checkpoints=4,
        probes=[mipt.code_dimension(), mipt.erasure_first_failure(16)],
    ).run(seed=11)
    np.testing.assert_array_equal(saved.values["code_dimension"], direct.values["code_dimension"])
    np.testing.assert_array_equal(saved.values["erasure_first_failure"],
                                  direct.values["erasure_first_failure"])


def test_estimate_only(run_tool):
    out = run_tool("ensemble_run", **ARGS, estimate_only=True)
    assert "estimated_seconds" in out.text and "nothing was run" in out.text


def test_same_request_again_reuses_the_saved_run(run_tool):
    first = run_tool("ensemble_run", **ARGS)
    second = run_tool("ensemble_run", **ARGS)
    assert second.data["output_dir"] == first.data["output_dir"] and "reused" in second.text


def test_deterministic_graph_needs_trajectories(run_tool):
    out = run_tool("ensemble_run", num_system=8, layers=4, measurement_rate=0.1)
    assert out.is_error and "give trajectories" in out.text
    assert not run_tool("ensemble_run", num_system=8, layers=4, measurement_rate=0.1,
                        trajectories=5).is_error


def test_background_always_then_job_status(run_tool):
    started = run_tool("ensemble_run", **{**ARGS, "seed": 12}, background="always")
    assert started.text.startswith("job ") and started.data["kind"] == "ensemble"
    done = run_tool("job_status", job_id=started.data["job_id"], wait_seconds=60)
    assert "completed" in done.text and "analysis_summary" in done.text


def test_second_request_joins_a_running_job(run_tool):
    big = dict(num_system=64, layers=400, measurement_rate=0.1, trajectories=4000, seed=1,
               probes=["code_dimension"], reference="single")
    a = run_tool("ensemble_run", **big, background="always")
    b = run_tool("ensemble_run", **big, background="always")
    assert b.data["job_id"] == a.data["job_id"] and "already running" in b.text
    run_tool("job_cancel", job_id=a.data["job_id"])
    run_tool("job_status", job_id=a.data["job_id"], wait_seconds=60)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_ensemble_tools.py -q`
Expected: FAIL with `KeyError: 'ensemble_run'`

- [ ] **Step 3: Implement `src/sparsegf2/agent/tools/ensemble.py`**

```python
"""ensemble_run (and, in Task 19, ensemble_sweep): monitored random circuits, flat arguments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2 import api
from sparsegf2.agent import vocab
from sparsegf2.agent.params import (
    CHECKPOINTS, GRAPH_PARAMS, OUTPUT_DIR, PROBES, PROTOCOL_PARAMS, SEED, ensemble_spec,
)
from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.render import curve, fmt, lines, pm, table
from sparsegf2.agent.sessions import from_api_status

TRAJECTORIES = Param(
    "trajectories", "integer",
    "Number of trajectories; default for a random graph family: graph_count x circuits_per_graph.",
    minimum=1,
)
ENSEMBLE_PARAMS = (*PROTOCOL_PARAMS, *GRAPH_PARAMS, PROBES, CHECKPOINTS, TRAJECTORIES, SEED,
                   OUTPUT_DIR)
MAX_ROWS = 12


def prepare(ctx, args: dict[str, Any], tool: str):
    spec = ensemble_spec(args, ctx)
    mc = api._build_spec(spec)
    if args.get("trajectories") is None and mc.num_trajectories is None:
        raise ValueError(
            "give trajectories (the number of random circuits); it may be left out only for a "
            "random graph family or a graph file with several graphs, which then run "
            "graph_count x circuits_per_graph circuits"
        )
    return spec, mc, ctx.output_dir(tool, args)


def _stored_spec(out: Path) -> dict | None:
    f = out / "spec.json"
    return json.loads(f.read_text(encoding="utf-8")).get("spec") if f.exists() else None


def _estimate(ctx, args):
    spec, mc, _ = prepare(ctx, args, "ensemble_run")
    return float(api.estimate_ensemble(spec, args.get("trajectories"))["estimated_seconds"] or 0.0)


def _run(ctx, args):
    spec, mc, out = prepare(ctx, args, "ensemble_run")
    if args.get("estimate_only"):
        return {"estimate": api.estimate_ensemble(spec, args.get("trajectories")),
                "output_dir": ctx.show(out)}
    if (out / "result.npz").exists() and not (out / "chunks").exists():
        if api._canonical(_stored_spec(out)) != api._canonical(spec):
            raise ValueError(f"{ctx.show(out)} holds a different run; give another output_dir")
        return {"summary": api.load_results(str(out)), "output_dir": ctx.show(out), "reused": True}
    if (out / "chunks").exists():
        raise ValueError(
            f"{ctx.show(out)} holds an unfinished background run of this request; run it again "
            "with background=always to resume it"
        )
    summary = api.run_ensemble(spec, args.get("trajectories"), seed=args.get("seed"),
                               output_dir=str(out), command="mcp:ensemble_run")
    return {"summary": summary, "output_dir": ctx.show(out)}


def start_or_join(ctx, spec: dict, args: dict, out: Path, command: str) -> dict[str, Any]:
    """Start (or resume) a chunked ensemble job, or return the job already writing to ``out``."""
    target = out.resolve()
    with api._JOBS_LOCK:
        running = [j for j in api._JOBS.values()
                   if j.output_dir == target and j.status in ("queued", "running")]
    if running:
        status = from_api_status(api.job_status(running[0].job_id))
        return {**status, "output_dir": ctx.show(out), "note": "already running; this is that job"}
    status = from_api_status(api.start_ensemble_job(
        spec, args.get("trajectories"), output_dir=str(out), seed=args.get("seed"),
        command=command))
    return {**status, "output_dir": ctx.show(out)}


def _background(ctx, args):
    spec, mc, out = prepare(ctx, args, "ensemble_run")
    return start_or_join(ctx, spec, args, out, "mcp:ensemble_run")


def _rows(n: int) -> np.ndarray:
    if n <= MAX_ROWS:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, MAX_ROWS).round().astype(int))


def render_summary(title: str, s: dict[str, Any], *, output_dir: str | None = None,
                   extra: tuple[str, ...] = ()) -> str:
    head = (f"{title}: n={s['num_system']}, layers={s['layers']}, p={fmt(s['measurement_rate'])}, "
            f"{s['num_trajectories']} trajectories, seed={s['seed']}, "
            f"{fmt(s['elapsed_seconds'], 3)} s")
    layers = s["checkpoints"]["layers"]
    scalar = {k: v for k, v in s["probes"].items() if not v.get("vector") and "layers" not in v}
    own = {k: v for k, v in s["probes"].items() if not v.get("vector") and "layers" in v}
    vector = {k: v for k, v in s["probes"].items() if v.get("vector")}
    parts = [head, *extra]
    if scalar:
        idx = _rows(len(layers))
        rows = [[layers[i], *[pm(v["mean"][i], v["stderr"][i]) for v in scalar.values()]]
                for i in idx]
        parts.append(table(["layer", *scalar], rows))
    for name, v in own.items():
        idx = _rows(len(v["layers"]))
        parts.append(f"{name}:\n" + table(["layer", name],
                                           [[v["layers"][i], pm(v["mean"][i], v["stderr"][i])] for i in idx]))
    for name, v in vector.items():
        if "mean" not in v:
            parts.append(f"{name}: no data")
            continue
        start = int(v["index"][0]) if "index" in v else 0
        parts.append(f"{name} at layer {v['at_layer']} ({v['what']}):\n  "
                     + curve(v["mean"], start=start))
    pur = s.get("purification")
    if pur:
        line = (f"purification: {pur['purified']} purified, {pur['censored']} censored "
                f"(horizon {pur['horizon_layer']})")
        if pur.get("tau_median_purified") is not None:
            line += f", tau median among purified {fmt(pur['tau_median_purified'])}"
        if pur.get("tau_median_all") is not None:
            line += f", tau median overall {pur['tau_median_all']}"
        parts.append(line)
    for note in (s.get("notes") or [])[:2]:
        parts.append(f"note: {note}")
    if output_dir:
        parts.append(f"saved: {output_dir}")
        nxt = [f'analysis_summary(source="{output_dir}")']
        kinds = " ".join(s["probes"])
        if "erasure_first_failure" in kinds:
            nxt.append(f'analysis_erasure(source="{output_dir}")')
        if pur:
            nxt.append(f'analysis_purification(source="{output_dir}")')
        parts.append("next: " + "; ".join(nxt))
    return lines(*parts)


def _render(d, ctx):
    if "estimate" in d:
        e = d["estimate"]
        return lines(
            f"ensemble_run: estimate only (nothing was run) for {e['trajectories']} trajectories",
            f"estimated_seconds: {fmt(e['estimated_seconds'])} on {e['threads']} threads "
            f"(pessimistic); result_megabytes: {fmt(e['result_megabytes'])}",
            f"output_dir would be: {d['output_dir']}",
        )
    extra = ("reused the saved run of this exact request",) if d.get("reused") else ()
    return render_summary("ensemble_run", d["summary"], output_dir=d["output_dir"], extra=extra)


register(ToolDef(
    name="ensemble_run", stack="ensemble", summary="Run an ensemble of monitored random circuits.",
    description=(
        "Run many trajectories of a monitored random Clifford circuit on a graph (or a "
        "collection of graphs): each layer applies random two-qubit Cliffords on graph edges, "
        "then measures each system qubit with probability measurement_rate. probes are call "
        "strings recorded at checkpoints (help_list(what=probes)). Returns means +- errors per "
        "checkpoint and purification statistics; everything is saved to output_dir. Long runs "
        "become background jobs. Example: {\"num_system\": 64, \"layers\": 256, "
        "\"measurement_rate\": 0.1, \"graph\": \"watts_strogatz\", \"graph_count\": 50, "
        "\"circuits_per_graph\": 4, \"schedule\": \"random_edges\", \"reference\": "
        "\"purification\", \"probes\": [\"code_dimension\", \"erasure_first_failure(samples=512)\"], "
        "\"seed\": 1}"
    ),
    params=(*ENSEMBLE_PARAMS,
            Param("estimate_only", "boolean", "Only estimate time and memory; run nothing.",
                  default=False)),
    handler=_run, render=_render, read_only=False, long_running=True, estimate=_estimate,
    background=_background,
    examples=({"num_system": 8, "layers": 8, "measurement_rate": 0.1, "graph": "watts_strogatz",
               "graph_count": 2, "reference": "purification", "probes": ["code_dimension"],
               "seed": 1},),
))
```

Add `ensemble` to `tools/__init__.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_ensemble_tools.py -q`
Expected: PASS. If the size test of Task 28 later finds `ensemble_run` above 5,000 characters, shorten the parameter descriptions in `params.py`, not the tool description's example.

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent tests/agent/test_ensemble_tools.py
git commit -m "agent: ensemble_run with flat parameters, reuse of saved runs, chunked background jobs"
```

---

### Task 19: `ensemble_sweep`

**Files:**
- Modify: `src/sparsegf2/agent/tools/ensemble.py`
- Test: `tests/agent/test_sweep_tool.py`

**Interfaces:**
- Consumes: `prepare`, `start_or_join`, `ENSEMBLE_PARAMS`, `grammar.parse_grid`, `sessions.report/cancelled`, `api._get_job`, `api._resolve_seed`.
- Produces:
  - The tool `ensemble_sweep`, whose parameters are `ENSEMBLE_PARAMS` with `num_system`, `layers` and `measurement_rate` optional, plus `grid` (required).
  - The sweep directory layout `sweep.json` (keys `version`, `grid`, `points`, `seed`, `base`), with each point at `<output_dir>/<k=v,...>/`.
  - `SWEEPABLE: frozenset[str]`.

- [ ] **Step 1: Write the failing tests**

```python
"""ensemble_sweep: a grid of ensembles sharing graph and circuit seeds."""

import json

import numpy as np

from sparsegf2 import mipt

BASE = dict(num_system=8, layers=6, graph="watts_strogatz", graph_count=2, reference="purification",
            probes=["code_dimension"], seed=3)


def test_grid_runs_every_point_and_shares_seeds(run_tool, ctx):
    out = run_tool("ensemble_sweep", **BASE, grid="measurement_rate=[0.05, 0.3]",
                   background="never")
    assert not out.is_error, out.text
    root = ctx.root / out.data["output_dir"]
    sweep = json.loads((root / "sweep.json").read_text(encoding="utf-8"))
    assert [p["params"] for p in sweep["points"]] == [{"measurement_rate": 0.05},
                                                     {"measurement_rate": 0.3}]
    lo, hi = (mipt.EnsembleResult.load(root / p["dir"] / "result.npz") for p in sweep["points"])
    assert np.all(lo.values["code_dimension"] >= hi.values["code_dimension"])  # nested measurements
    assert "next: analysis_compare" in out.text


def test_rerun_resumes_finished_points(run_tool):
    a = run_tool("ensemble_sweep", **BASE, grid="measurement_rate=[0.1]", background="never")
    b = run_tool("ensemble_sweep", **BASE, grid="measurement_rate=[0.1]", background="never")
    assert a.data["output_dir"] == b.data["output_dir"] and not b.is_error


def test_grid_validation(run_tool):
    assert "cannot be swept" in run_tool("ensemble_sweep", **BASE, measurement_rate=0.1,
                                         grid="reference=[1]", background="never").text
    assert "both in grid and as an argument" in run_tool(
        "ensemble_sweep", **BASE, measurement_rate=0.1, grid="measurement_rate=[0.2]",
        background="never").text
    no_n = {k: v for k, v in BASE.items() if k != "num_system"}
    assert "num_system" in run_tool("ensemble_sweep", **no_n, grid="measurement_rate=[0.2]",
                                    background="never").text


def test_default_is_a_background_job(run_tool):
    out = run_tool("ensemble_sweep", **BASE, grid="measurement_rate=[0.2, 0.25]")
    assert out.text.startswith("job ")
    done = run_tool("job_status", job_id=out.data["job_id"], wait_seconds=60)
    assert "completed" in done.text and "2/2 points" in done.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_sweep_tool.py -q`
Expected: FAIL with `KeyError: 'ensemble_sweep'`

- [ ] **Step 3: Implement (append to `tools/ensemble.py`)**

```python
import dataclasses
import itertools
import math

from sparsegf2.agent import sessions
from sparsegf2.agent.grammar import parse_grid

SWEEPABLE = frozenset({
    "num_system", "layers", "measurement_rate", "gates_per_layer", "circuits_per_graph",
    "scramble_gates", "scramble_matchings", "scramble_layers", "graph_k", "graph_beta", "graph_p",
    "graph_d", "graph_count", "trajectories",
})
_REQUIRED = ("num_system", "layers", "measurement_rate")
SWEEP_PARAMS = tuple(
    dataclasses.replace(p, required=False) if p.name in _REQUIRED else p for p in ENSEMBLE_PARAMS
) + (Param("grid", "string",
           "Parameters to sweep, every combination: e.g. measurement_rate=[0.05,0.1,0.15]; "
           "num_system=[64,128].", required=True),)


def _point_name(point: dict[str, Any]) -> str:
    return ",".join(f"{k}={fmt(v, 6)}" for k, v in point.items())


def _grid_points(args: dict[str, Any]) -> list[dict[str, Any]]:
    grid = parse_grid(args["grid"])
    by_name = {p.name: p for p in ENSEMBLE_PARAMS}
    for key, values in grid.items():
        if key not in SWEEPABLE:
            raise ValueError(f"grid: {key} cannot be swept; sweepable: {sorted(SWEEPABLE)}")
        if key in args:
            raise ValueError(f"{key} is both in grid and as an argument; give it in one place")
        grid[key] = [by_name[key].check(v, "grid") for v in values]
    missing = [k for k in _REQUIRED if k not in args and k not in grid]
    if missing:
        raise ValueError(f"give {', '.join(missing)} as arguments or in grid")
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]


def _sweep(ctx, args):
    points = _grid_points(args)
    root = ctx.output_dir("ensemble_sweep", args)
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / "sweep.json"
    if meta_path.exists():
        seed = json.loads(meta_path.read_text(encoding="utf-8"))["seed"]
    else:
        seed = args["seed"] if args.get("seed") is not None else api._resolve_seed(None)
    base = {k: v for k, v in args.items() if k not in ("grid", "output_dir")}
    meta = {"version": 1, "grid": args["grid"], "seed": seed, "base": base,
            "points": [{"params": p, "dir": _point_name(p)} for p in points]}
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    done: list[dict[str, Any]] = []
    for i, point in enumerate(points):
        if sessions.cancelled():
            break
        point_args = {**args, **point, "seed": seed}
        out = root / _point_name(point)
        spec, mc, _ = prepare(ctx, {**point_args, "output_dir": str(out)}, "ensemble_sweep")
        status = start_or_join(ctx, spec, point_args, out, "mcp:ensemble_sweep")
        job = api._get_job(status["job_id"])
        while job.thread is not None and job.thread.is_alive():
            job.thread.join(0.5)
            if sessions.cancelled():
                api.cancel_job(job.job_id)
        if job.status == "failed":
            raise RuntimeError(f"point {_point_name(point)} failed: {job.error}")
        done.append({"params": point, "dir": ctx.show(out), "status": job.status})
        sessions.report(i + 1, len(points), "points")
    return {"output_dir": ctx.show(root), "seed": seed, "points": done,
            "total_points": len(points), "cancelled": sessions.cancelled()}


def _render_sweep(d, ctx):
    rows = [[p["dir"].rsplit("/", 1)[-1], p["status"]] for p in d["points"]]
    return lines(
        f"ensemble_sweep: {len(d['points'])}/{d['total_points']} points, seed {d['seed']} "
        f"-> {d['output_dir']}" + (" (cancelled)" if d["cancelled"] else ""),
        table(["point", "status"], rows),
        f'next: analysis_compare(source="{d["output_dir"]}", metrics=["mean(code_dimension)"])',
    )


register(ToolDef(
    name="ensemble_sweep", stack="ensemble", summary="A grid of ensembles (e.g. a scan in p).",
    description=(
        "Run ensemble_run at every combination of the grid, e.g. grid=\"measurement_rate=[0.05, "
        "0.1, 0.15]; num_system=[64, 128]\". All points share graph seeds and the circuit seed, "
        "so lower rates measure a subset of the qubits higher ones do. A background job by "
        "default; each point is saved as it finishes and rerunning resumes. Compare the points "
        "with analysis_compare(source=output_dir). Example: {\"num_system\": 32, \"layers\": 128, "
        "\"grid\": \"measurement_rate=[0.05, 0.1, 0.2]\", \"reference\": \"purification\", "
        "\"probes\": [\"code_dimension\"], \"trajectories\": 200, \"seed\": 1}"
    ),
    params=SWEEP_PARAMS, handler=_sweep, render=_render_sweep, read_only=False,
    long_running=True, estimate=lambda ctx, a: math.inf,
    examples=({"num_system": 8, "layers": 4, "trajectories": 8, "reference": "purification",
               "grid": "measurement_rate=[0.1, 0.2]", "probes": ["code_dimension"],
               "background": "never", "seed": 1},),
))
```

Note: `start_or_join` passes `args.get("seed")`, and a sweep point always has the sweep's seed. `api.start_ensemble_job` on a finished point directory completes at once, which is what makes reruns resume.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_sweep_tool.py tests/agent/test_ensemble_tools.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent/tools/ensemble.py tests/agent/test_sweep_tool.py
git commit -m "agent: ensemble_sweep over any grid, shared seeds, resumable points"
```

---

### Task 20: Trajectory tools (`traj_run`, `traj_probe`)

**Files:**
- Create: `src/sparsegf2/agent/tools/traj.py`
- Modify: `src/sparsegf2/agent/tools/__init__.py`
- Test: `tests/agent/test_traj_tools.py`

**Interfaces:**
- Consumes:
  - from `params`: `ensemble_spec`, `probe_dicts`, `PROTOCOL_PARAMS`, `GRAPH_PARAMS`, `PROBES`, `CHECKPOINTS`, `SEED`, `OUTPUT_DIR`, `SOURCE`;
  - `api._build_spec`, `api._resolve_seed`;
  - from `mipt`: `MonitoredCircuit.replay`, `.circuit`, `.reference_qubits`, `RANDOMNESS_CONTRACT`, `code_snapshot`;
  - `Simulator.save`, `StabilizerCode.from_simulator`, `crosscheck.compare_stabilizers`;
  - `tools.ensemble.render_summary` is not used: trajectories render their own table.
- Produces:
  - The tools `traj_run` and `traj_probe`.
  - The trajectory record layout:
    - `trajectory.json`, with keys `version`, `spec`, `seed`, `trajectory`, `checkpoints`, `store`, `package_version`, `randomness_contract`;
    - `result.npz`, a one-row `EnsembleResult` whose code snapshot probe is named `trajectory_code`;
    - `state_L<layer>.state.npz` files;
    - `circuit.stim`.
  - `tools.traj.canonical(matrix) -> np.ndarray`, the reduced row echelon form over GF(2).

- [ ] **Step 1: Write the failing tests**

```python
"""traj_run and traj_probe: one trajectory, its code, depth replay, Stim cross-check."""

import json

import numpy as np

from sparsegf2 import StabilizerCode, graphs, mipt
from sparsegf2.agent.tools.traj import canonical

ARGS = dict(num_system=12, layers=10, measurement_rate=0.2, reference="purification",
            trajectory=3, seed=5, checkpoints="[5, 10]")


def test_default_stores_the_engine_code(run_tool, ctx):
    out = run_tool("traj_run", **ARGS)
    assert not out.is_error, out.text
    rec = ctx.root / out.data["output_dir"]
    meta = json.loads((rec / "trajectory.json").read_text(encoding="utf-8"))
    assert meta["trajectory"] == 3 and meta["randomness_contract"] == mipt.RANDOMNESS_CONTRACT
    ens = mipt.MonitoredCircuit(num_system=12, edges=graphs.cycle(12), layers=10,
                                measurement_rate=0.2, reference="purification",
                                checkpoints=[5, 10], probes=[mipt.code_snapshot()]).run(5, seed=5)
    mine = mipt.EnsembleResult.load(rec / "result.npz")
    for c in range(2):
        np.testing.assert_array_equal(mine.code(0, c).stabilizer_matrix(),
                                      ens.code(3, c).stabilizer_matrix())
    assert "layer" in out.text and "k" in out.text


def test_store_code_needs_a_reference(run_tool):
    out = run_tool("traj_run", num_system=8, layers=4, measurement_rate=0.1)
    assert out.is_error and "store=code needs a code" in out.text


def test_store_state_and_reopen(run_tool, ctx):
    out = run_tool("traj_run", **ARGS, store="state")
    rec = ctx.root / out.data["output_dir"]
    assert (rec / "state_L10.state.npz").exists()
    opened = run_tool("state_new", source=f"{out.data['output_dir']}/state_L10.state.npz")
    sim = ctx.states.get(opened.data["state_id"])
    code = StabilizerCode.from_simulator(sim, reference=range(12, 24))
    engine = run_tool("traj_run", **ARGS)  # same trajectory, code stored
    stored = mipt.EnsembleResult.load(ctx.root / engine.data["output_dir"] / "result.npz")
    np.testing.assert_array_equal(canonical(code.stabilizer_matrix()),
                                  canonical(stored.code(0, 1).stabilizer_matrix()))


def test_compare_stim_and_save_circuit(run_tool, ctx):
    import stim

    out = run_tool("traj_run", **ARGS, compare_stim=True, save_circuit=True)
    assert "stim: match" in out.text and "engine vs simulator: match" in out.text
    text = (ctx.root / out.data["output_dir"] / "circuit.stim").read_text(encoding="utf-8")
    stim.Circuit(text)


def test_probe_replays_at_new_depths(run_tool, ctx):
    first = run_tool("traj_run", **ARGS)
    again = run_tool("traj_probe", source=first.data["output_dir"],
                     probes=["code_dimension", "half_cut_entropy"], checkpoints="[2, 5, 10]")
    assert not again.is_error, again.text
    a = mipt.EnsembleResult.load(ctx.root / first.data["output_dir"] / "result.npz")
    b = mipt.EnsembleResult.load(ctx.root / again.data["output_dir"] / "result.npz")
    np.testing.assert_array_equal(a.code(0, 1).stabilizer_matrix(),
                                  b.code(0, 2).stabilizer_matrix())


def test_probe_refuses_another_randomness_contract(run_tool, ctx):
    first = run_tool("traj_run", **ARGS)
    f = ctx.root / first.data["output_dir"] / "trajectory.json"
    meta = json.loads(f.read_text(encoding="utf-8"))
    meta["randomness_contract"] = 0
    f.write_text(json.dumps(meta), encoding="utf-8")
    out = run_tool("traj_probe", source=first.data["output_dir"], probes=["code_dimension"])
    assert out.is_error and "randomness contract" in out.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_traj_tools.py -q`
Expected: FAIL with `KeyError: 'traj_run'`

- [ ] **Step 3: Implement `src/sparsegf2/agent/tools/traj.py`**

```python
"""traj_run and traj_probe: one trajectory of an ensemble, recorded and replayable."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2 import api, mipt
from sparsegf2._version import __version__
from sparsegf2.agent.grammar import parse_checkpoints
from sparsegf2.agent.params import (
    CHECKPOINTS, GRAPH_PARAMS, OUTPUT_DIR, PROBES, PROTOCOL_PARAMS, SEED, SOURCE, ensemble_spec,
    probe_dicts,
)
from sparsegf2.agent.registry import MISSING, Param, ToolDef, register
from sparsegf2.agent.render import fmt, lines, pm, table
from sparsegf2.codes import StabilizerCode
from sparsegf2.simulator import Simulator

SNAPSHOT = "trajectory_code"
STORE = Param("store", "string",
              "code: save the code at each checkpoint (needs a reference); state: save the full "
              "state (reopen with state_new); none.", default="code", enum=("code", "state", "none"))
COMPARE = Param("compare_stim", "boolean",
                "Check the trajectory against Stim and the engine's code against the gate-by-gate "
                "simulator at every checkpoint.", default=False)
SAVE_CIRCUIT = Param("save_circuit", "boolean", "Write the exact circuit as circuit.stim.",
                     default=False)


def canonical(matrix: np.ndarray) -> np.ndarray:
    """Reduced row echelon form over GF(2): equal for two generator sets of the same group."""
    m = (np.asarray(matrix) % 2).astype(np.uint8)
    r = 0
    for c in range(m.shape[1]):
        if r == m.shape[0]:
            break
        nz = np.flatnonzero(m[r:, c])
        if not nz.size:
            continue
        p = r + int(nz[0])
        m[[r, p]] = m[[p, r]]
        others = np.flatnonzero(m[:, c])
        m[others[others != r]] ^= m[r]
        r += 1
    return m[:r]


def _record(ctx, spec: dict, seed: int, t: int, store: str, compare: bool, save_circuit: bool,
            out: Path) -> dict[str, Any]:
    mc = api._build_spec(spec)
    if store == "code" and mc.reference == "none":
        raise ValueError("store=code needs a code: use reference=purification or single, or "
                         "store=state")
    probes = list(mc.probes)
    if store == "code":
        if any(p.kind == "code_snapshot" for p in probes):
            raise ValueError("store=code already saves the code; drop code_snapshot from probes")
        probes.append(mipt.code_snapshot(name=SNAPSHOT))
    res = mc.replay(t, seed=seed, probes=probes)
    out.mkdir(parents=True, exist_ok=True)
    res.save(out / "result.npz")
    layers = [int(x) for x in res.checkpoint_layers]
    data: dict[str, Any] = {"output_dir": ctx.show(out), "trajectory": t, "seed": seed,
                            "num_system": mc.num_system, "reference": mc.reference,
                            "layers": layers, "files": ["result.npz", "trajectory.json"]}
    if store == "code":
        data["k"] = []
        for c in range(len(layers)):
            try:
                data["k"].append(res.code(0, c, SNAPSHOT).num_logical)
            except ValueError:  # stopped: the reference is pure
                data["k"].append(0)
    data["probes"] = {}
    for name, vals in res.values.items():
        arr = np.asarray(vals, dtype=float)
        if arr.ndim == 2:  # a scalar probe: (1 trajectory, evaluations)
            data["probes"][name] = [None if np.isnan(v) else v for v in arr[0].tolist()]
    data["probe_layers"] = {name: [int(x) for x in res.layers(name)] for name in data["probes"]}
    circuit = mc.circuit(t, seed=seed) if (store == "state" or compare or save_circuit) else None
    if store == "state":
        wanted = set(layers)

        def observe(sim, tick):
            if tick in wanted:
                sim.save(out / f"state_L{tick}.state.npz")

        Simulator(mc.num_qubits, phases=False, seed=0).run(circuit, observe=observe)
        data["files"] += [f"state_L{x}.state.npz" for x in layers]
    if compare:
        from sparsegf2 import crosscheck

        data["stim"] = crosscheck.compare_stabilizers(circuit, num_qubits=mc.num_qubits,
                                                      phases=False, ticks=layers)
        if store == "code":
            codes: dict[int, np.ndarray] = {}

            def grab(sim, tick):
                if tick in set(layers):
                    codes[tick] = canonical(StabilizerCode.from_simulator(
                        sim, mc.reference_qubits).stabilizer_matrix())

            Simulator(mc.num_qubits, phases=False, seed=0).run(circuit, observe=grab)
            match = True
            for c, layer in enumerate(layers):
                try:
                    engine = canonical(res.code(0, c, SNAPSHOT).stabilizer_matrix())
                except ValueError:
                    continue
                match &= engine.shape == codes[layer].shape and np.array_equal(engine, codes[layer])
            data["engine_vs_simulator"] = bool(match)
    if save_circuit:
        (out / "circuit.stim").write_text(str(circuit.to_stim()) + "\n", encoding="utf-8")
        data["files"].append("circuit.stim")
    meta = {"version": 1, "spec": spec, "seed": seed, "trajectory": t, "checkpoints": layers,
            "store": store, "package_version": __version__,
            "randomness_contract": mipt.RANDOMNESS_CONTRACT}
    (out / "trajectory.json").write_text(json.dumps(api._jsonable(meta, exact=True), indent=1),
                                         encoding="utf-8")
    return data


def _run(ctx, a):
    spec = ensemble_spec(a, ctx)
    seed = api._resolve_seed(a.get("seed"))
    out = ctx.output_dir("traj_run", {**a, "seed": seed})
    return _record(ctx, spec, seed, a["trajectory"], a["store"], a["compare_stim"],
                   a["save_circuit"], out)


def _probe(ctx, a):
    src = ctx.path(a["source"], what="source", must_exist=True)
    meta_path = src / "trajectory.json"
    if not meta_path.exists():
        raise ValueError(f"{ctx.show(src)} is not a trajectory record (no trajectory.json); "
                         "traj_run makes one")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("randomness_contract") != mipt.RANDOMNESS_CONTRACT:
        raise ValueError(
            f"this record was made under randomness contract {meta.get('randomness_contract')} "
            f"(package {meta.get('package_version')}); this version uses contract "
            f"{mipt.RANDOMNESS_CONTRACT}, so replaying would give a different trajectory. Run "
            "traj_run again with this version."
        )
    spec = dict(meta["spec"])
    spec["probes"] = probe_dicts(a.get("probes") or [])
    if a.get("checkpoints") is not None:
        spec["checkpoints"] = parse_checkpoints(a["checkpoints"])
    out = (ctx.path(a["output_dir"], what="output_dir") if a.get("output_dir")
           else ctx.output_dir("traj_probe", {**a, "source": ctx.show(src)}))
    return _record(ctx, spec, meta["seed"], meta["trajectory"], a["store"], a["compare_stim"],
                   False, out)


def _render(d, ctx, title):
    head = (f"{title}: trajectory {d['trajectory']} (seed {d['seed']}), n={d['num_system']}, "
            f"reference={d['reference']} -> {d['output_dir']}")
    cols = ["layer"] + (["k"] if "k" in d else [])
    rows = [[layer] + ([d["k"][i]] if "k" in d else []) for i, layer in enumerate(d["layers"])]
    for name, values in d["probes"].items():
        if d["probe_layers"][name] == d["layers"] and name != SNAPSHOT:
            cols.append(name)
            for i, v in enumerate(values):
                rows[i].append(v)
    parts = [head, table(cols, rows)]
    if "stim" in d:
        s = d["stim"]
        parts.append(f"stim: {'match' if s['match'] else 'MISMATCH at tick ' + str(s['first_mismatch']['tick'])}"
                     f" ({s['ticks_compared']} checkpoints)")
    if "engine_vs_simulator" in d:
        parts.append(f"engine vs simulator: {'match' if d['engine_vs_simulator'] else 'MISMATCH'}")
    parts.append("files: " + ", ".join(d["files"]))
    parts.append(f'next: code_analyze(source="{d["output_dir"]}", properties=[...]); '
                 f'traj_probe(source="{d["output_dir"]}", probes=[...], checkpoints=...)')
    return lines(*parts)


TRAJ_PARAMS = (*PROTOCOL_PARAMS, *GRAPH_PARAMS, PROBES, CHECKPOINTS, SEED, OUTPUT_DIR,
    Param("trajectory", "integer",
          "Which trajectory of the ensemble with this seed (the same numbering as ensemble_run).",
          default=0, minimum=0),
    STORE, COMPARE, SAVE_CIRCUIT)

register(ToolDef(
    name="traj_run", stack="traj", summary="One trajectory: store its code, probe, cross-check.",
    description=(
        "Simulate one trajectory of a monitored random circuit (the same trajectory ensemble_run "
        "would run as number `trajectory` with this seed) and store its code at each checkpoint "
        "(store=code, the default), or its full state (store=state). probes are optional call "
        "strings; compare_stim checks it against Stim; traj_probe replays it later at other "
        "depths. Example: {\"num_system\": 64, \"layers\": 256, \"measurement_rate\": 0.1, "
        "\"graph\": \"watts_strogatz\", \"graph_count\": 50, \"reference\": \"purification\", "
        "\"trajectory\": 17, \"seed\": 1, \"checkpoints\": \"[64, 128, 256]\"}"
    ),
    params=TRAJ_PARAMS, handler=_run, render=lambda d, ctx: _render(d, ctx, "traj_run"),
    read_only=False, long_running=True,
    examples=({"num_system": 8, "layers": 6, "measurement_rate": 0.2,
               "reference": "purification", "seed": 1, "checkpoints": "[3, 6]"},),
))

register(ToolDef(
    name="traj_probe", stack="traj", summary="Replay a recorded trajectory with new probes/depths.",
    description=(
        "Replay a trajectory recorded by traj_run, exactly, with new probes at new checkpoints "
        "(and store its code there): probing a trajectory as a function of depth after the "
        "fact. Example: {\"source\": \"runs/traj_run-1a2b3c\", \"probes\": [\"code_dimension\", "
        "\"erasure_first_failure(samples=512)\"], \"checkpoints\": \"16\"}"
    ),
    params=(SOURCE, PROBES, dataclasses.replace(CHECKPOINTS, default=MISSING),
            STORE, COMPARE,
            Param("output_dir", "string", "Where to save (default: a folder per request).")),
    handler=_probe, render=lambda d, ctx: _render(d, ctx, "traj_probe"),
    read_only=False, long_running=True,
))
```

`traj_probe` keeps the recorded checkpoints when `checkpoints` is left out (its `CHECKPOINTS` copy has no default).

Add `traj` to `tools/__init__.py`. `traj_probe` has no static example, because it needs a record, so add it to `EXAMPLE_EXEMPT` in Task 28.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_traj_tools.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent tests/agent/test_traj_tools.py
git commit -m "agent: traj_run/traj_probe (record, replay at any depth, state export, Stim check)"
```

---
### Task 21: `code_analyze`

**Files:**
- Create: `src/sparsegf2/agent/tools/code.py`
- Modify: `src/sparsegf2/agent/tools/__init__.py`, `tests/agent/test_help_tools.py` (add the `code_properties` case)
- Test: `tests/agent/test_code_tool.py`

**Interfaces:**
- Consumes:
  - `StabilizerCode` (`first_failure_sizes`, `uncorrectable_counts`, `lost_information`, `is_correctable`, `contiguous_distance`, `from_simulator`, `stabilizer_matrix`);
  - from `analysis`: `failure_by_size`, `exact_failure_by_size`, `hybrid_failure_by_size`, `mean_first_failure`, `distance_bounds`, `erasure_failure_probability`, `erasure_tolerance`;
  - from `api`: `_load_directory`, `_load_result_file`, `_qubit_list`;
  - from `grammar`: `parse_calls`, `bind`, `plain`.
- Produces:
  - The tool `code_analyze`.
  - `CODE_PROPERTIES: dict[str, tuple[tuple[str, ...], str, Callable]]`, mapping name to (parameters, description, fn), where `fn(code, cache, **kw) -> Any`.
  - `load_code(ctx, args) -> tuple[StabilizerCode, str]`.
  - The `*.code.npz` format: arrays `stabilizers` (uint8, `(s, 2n)`) and `num_qubits`.

- [ ] **Step 1: Write the failing tests**

```python
"""code_analyze on the [[4,2,2]] code, a saved trajectory code, an open state, a code file."""

import numpy as np

from sparsegf2 import analysis, StabilizerCode

C422 = ["XXXX", "ZZZZ"]


def test_properties_of_the_422_code(run_tool):
    out = run_tool("code_analyze", stabilizers=C422, properties=[
        "summary", "first_failure(samples=50)", "exhaustive_counts(2)", "distance_bounds",
        "mean_first_failure", "failure_probability(rates=[0.1], delta=0.01)", "stabilizers",
        "stabilizer_weights", "contiguous_distance", "lost_information(0:2)",
        "is_correctable([0])"])
    assert not out.is_error, out.text
    t = out.text
    assert "summary: n=4, k=2, generators=2" in t
    assert "distance_bounds: d = 2" in t and "mean_first_failure: 2" in t
    assert "stabilizers: XXXX ZZZZ" in t and "is_correctable([0]): true" in t
    assert "lost_information(0:2): 2" in t and "contiguous_distance: 2" in t
    phi = analysis.erasure_failure_probability(
        StabilizerCode(C422).first_failure_sizes(512, seed=0), 4, 0.1,
        exact_counts=StabilizerCode(C422).uncorrectable_counts(2))
    assert f"{phi:.4g}" in t


def test_code_from_a_trajectory_record_and_a_saved_file(run_tool, ctx):
    rec = run_tool("traj_run", num_system=10, layers=8, measurement_rate=0.1,
                   reference="purification", seed=2).data["output_dir"]
    a = run_tool("code_analyze", source=rec, properties=["summary"], save_to="c.code.npz")
    b = run_tool("code_analyze", source="c.code.npz", properties=["summary"])
    assert a.text.split("\n")[1] == b.text.split("\n")[1]


def test_code_of_an_open_state(run_tool):
    run_tool("state_new", num_qubits=4, phases=False)
    run_tool("state_apply", state_id="s1", circuit="H 0\nCX 0 2\nH 1\nCX 1 3")
    out = run_tool("code_analyze", state_id="s1", reference_qubits="2:4", properties=["summary"])
    assert "n=2, k=2, generators=0" in out.text


def test_errors(run_tool):
    assert "exactly one of stabilizers, source or state_id" in run_tool(
        "code_analyze", properties=["summary"]).text
    out = run_tool("code_analyze", stabilizers=C422, properties=["distanse"])
    assert out.is_error and "did you mean distance_bounds" in out.text
```

Add to the parameter list of `test_lists` in `tests/agent/test_help_tools.py`: `("code_properties", "first_failure(samples, seed)")`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_code_tool.py -q`
Expected: FAIL with `KeyError: 'code_analyze'`

- [ ] **Step 3: Implement `src/sparsegf2/agent/tools/code.py`**

```python
"""code_analyze: every erasure and structure property of a stabilizer code, on request."""

from __future__ import annotations

import difflib
from collections.abc import Callable
from typing import Any

import numpy as np

from sparsegf2 import analysis, api
from sparsegf2.agent.grammar import bind, parse_calls, plain
from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.render import curve, fmt, lines
from sparsegf2.codes import StabilizerCode

_PAULI = np.array(["_", "X", "Z", "Y"])


def _qubits(code: StabilizerCode, value: Any, what: str) -> list[int]:
    return api._qubit_list(plain(value), what, code.num_qubits)


def _ff(code, cache, samples=512, seed=0):
    key = ("ff", int(samples), int(seed))
    if key not in cache:
        cache[key] = code.first_failure_sizes(int(samples), seed=int(seed))
    return cache[key]


def _counts(code, cache, max_size=2):
    key = ("counts", int(max_size))
    if key not in cache:
        cache[key] = code.uncorrectable_counts(int(max_size))
    return cache[key]


def _summary(code, cache):
    return {"n": code.num_qubits, "k": code.num_logical, "generators": code.num_stabilizers}


def _first_failure(code, cache, samples=512, seed=0):
    d = _ff(code, cache, samples, seed)
    return {"samples": int(d.size), "mean": float(d.mean()), "min": int(d.min()),
            "max": int(d.max()), "p_c": analysis.failure_by_size(d, code.num_qubits)}


def _failure_by_size(code, cache, samples=512, seed=0):
    return analysis.failure_by_size(_ff(code, cache, samples, seed), code.num_qubits)


def _exhaustive(code, cache, max_size=2):
    c = _counts(code, cache, max_size)
    return {"counts": c, "p_c_exact": analysis.exact_failure_by_size(c, code.num_qubits)}


def _bounds(code, cache, samples=512, max_size=2, seed=0):
    low, high = analysis.distance_bounds(_ff(code, cache, samples, seed), code.num_qubits,
                                         exact_counts=_counts(code, cache, max_size))
    return {"low": int(low), "high": int(high)}


def _mu(code, cache, samples=512, max_size=2, seed=0):
    return float(analysis.mean_first_failure(_ff(code, cache, samples, seed), code.num_qubits,
                                             exact_counts=_counts(code, cache, max_size)))


def _phi(code, cache, rates=(0.01, 0.02, 0.05, 0.1), delta=0.01, samples=512, max_size=2, seed=0):
    d = _ff(code, cache, samples, seed)
    counts = _counts(code, cache, max_size)
    rates = [float(r) for r in (rates if isinstance(rates, list | tuple) else [rates])]
    phi = analysis.erasure_failure_probability(d, code.num_qubits, rates, exact_counts=counts)
    e_d = analysis.erasure_tolerance(d[None, :], code.num_qubits, float(delta),
                                     exact_counts=counts[None, :])
    return {"rates": rates, "phi": np.atleast_1d(phi).tolist(), "delta": float(delta),
            "e_delta": e_d}


def _weights(code, cache):
    m = code.stabilizer_matrix()
    n = code.num_qubits
    w = ((m[:, :n] | m[:, n:]) > 0).sum(axis=1) if len(m) else np.zeros(0, int)
    return {"mean": float(w.mean()) if w.size else 0.0, "min": int(w.min()) if w.size else 0,
            "max": int(w.max()) if w.size else 0}


def _stabilizers(code, cache):
    n = code.num_qubits
    if n > api.MAX_STABILIZER_QUBITS:
        raise ValueError(f"stabilizers are listed only up to {api.MAX_STABILIZER_QUBITS} qubits; "
                         "use stabilizer_weights")
    m = code.stabilizer_matrix().astype(int)
    return ["".join(_PAULI[m[i, :n] + 2 * m[i, n:]]) for i in range(len(m))]


def _contiguous(code, cache, order=None, periodic=True):
    seq = None if order is None else _qubits(code, order, "contiguous_distance.order")
    return code.contiguous_distance(seq, periodic=bool(periodic))


def _lost(code, cache, region):
    return code.lost_information(_qubits(code, region, "lost_information.region"))


def _correctable(code, cache, region):
    return code.is_correctable(_qubits(code, region, "is_correctable.region"))


CODE_PROPERTIES: dict[str, tuple[tuple[str, ...], str, Callable]] = {
    "summary": ((), "n, k (logical qubits) and the number of generators", _summary),
    "lost_information": (("region",), "bits of logical information lost by erasing region", _lost),
    "is_correctable": (("region",), "whether erasing region is correctable", _correctable),
    "first_failure": (("samples", "seed"), "first-failure sizes D over random orderings: "
                      "mean, range and P_C(m)", _first_failure),
    "failure_by_size": (("samples", "seed"), "P_C(m), m = 0..n, from orderings", _failure_by_size),
    "exhaustive_counts": (("max_size",), "uncorrectable m-qubit erasures for m <= max_size, "
                          "and the exact P_C(m)", _exhaustive),
    "distance_bounds": (("samples", "max_size", "seed"), "bounds on the distance d (exact when "
                        "d <= max_size)", _bounds),
    "mean_first_failure": (("samples", "max_size", "seed"), "mu_D = E[D] (hybrid estimator)", _mu),
    "failure_probability": (("rates", "delta", "samples", "max_size", "seed"),
                            "Phi(e) at each erasure rate and e_delta", _phi),
    "stabilizer_weights": ((), "generator weights: mean, min, max (basis dependent)", _weights),
    "stabilizers": ((), "the generators as Pauli strings (up to 32 qubits)", _stabilizers),
    "contiguous_distance": (("order", "periodic"), "shortest contiguous window (in order) "
                            "whose erasure is uncorrectable", _contiguous),
}


def load_code(ctx, a: dict[str, Any]) -> tuple[StabilizerCode, str]:
    given = [k for k in ("stabilizers", "source", "state_id") if a.get(k) is not None]
    if len(given) != 1:
        raise ValueError("give exactly one of stabilizers, source or state_id")
    if given == ["stabilizers"]:
        return StabilizerCode(list(a["stabilizers"])), "the given stabilizers"
    if given == ["state_id"]:
        if not a.get("reference_qubits"):
            raise ValueError("state_id needs reference_qubits (e.g. 8:16)")
        sim = ctx.states.get(a["state_id"])
        ref = api._qubit_list(a["reference_qubits"], "reference_qubits", sim.num_qubits)
        return StabilizerCode.from_simulator(sim, ref), f"state {a['state_id']}"
    path = ctx.path(a["source"], what="source", must_exist=True)
    if path.name.endswith(".code.npz"):
        with np.load(path, allow_pickle=False) as z:
            return StabilizerCode(z["stabilizers"], num_qubits=int(z["num_qubits"])), ctx.show(path)
    if path.name.endswith(".state.npz"):
        if not a.get("reference_qubits"):
            raise ValueError("a state file needs reference_qubits (e.g. 8:16)")
        from sparsegf2.simulator import Simulator

        sim = Simulator.load(path)
        ref = api._qubit_list(a["reference_qubits"], "reference_qubits", sim.num_qubits)
        return StabilizerCode.from_simulator(sim, ref), ctx.show(path)
    res = api._load_directory(path)[0] if path.is_dir() else api._load_result_file(path)
    if res is None:
        raise ValueError(f"no saved result in {ctx.show(path)}")
    evaluations = len(res.layers(next(iter(res.snapshots))) if res.snapshots else [])
    evaluation = a.get("evaluation", -1)
    if evaluation < 0:
        evaluation += evaluations
    t = a.get("trajectory")
    numbers = np.asarray(res.trajectories)
    if t is None:
        row = 0
    else:
        hits = np.flatnonzero(numbers == t)
        if not hits.size:
            raise ValueError(f"trajectory {t} is not in {ctx.show(path)} (it holds "
                             f"{numbers.min()}..{numbers.max()})")
        row = int(hits[0])
    code = res.code(row, evaluation)
    return code, f"{ctx.show(path)}, trajectory {int(numbers[row])}, evaluation {evaluation}"


def _analyze(ctx, a):
    code, origin = load_code(ctx, a)
    cache: dict = {}
    results = []
    for i, call in enumerate(parse_calls(a.get("properties") or ["summary"], what="properties")):
        if call.name not in CODE_PROPERTIES:
            close = difflib.get_close_matches(call.name, list(CODE_PROPERTIES), n=2)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise ValueError(f"properties[{i}]: unknown property {call.name!r}{hint} "
                             "(help_list(what=code_properties))")
        names, _, fn = CODE_PROPERTIES[call.name]
        kwargs = plain(bind(call, names, what=f"properties[{i}]"))
        results.append({"property": call.text.strip(), "name": call.name,
                        "value": fn(code, cache, **kwargs)})
    out = {"origin": origin, "n": code.num_qubits, "k": code.num_logical, "results": results}
    if a.get("save_to"):
        path = ctx.path(a["save_to"], what="save_to")
        if not path.name.endswith(".code.npz"):
            raise ValueError("code files end in .code.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            np.savez(fh, stabilizers=code.stabilizer_matrix().astype(np.uint8),
                     num_qubits=np.int64(code.num_qubits))
        out["saved"] = ctx.show(path)
    return out


def _show(name: str, label: str, v: Any) -> str:
    if name == "summary":
        return f"{label}: n={v['n']}, k={v['k']}, generators={v['generators']}"
    if name == "first_failure":
        return (f"{label}: mean D={fmt(v['mean'])} over {v['samples']} orderings, "
                f"range {v['min']}..{v['max']}; P_C(m) " + curve(v["p_c"]))
    if name == "failure_by_size":
        return f"{label}: P_C(m) " + curve(v)
    if name == "exhaustive_counts":
        return (f"{label}: counts {' '.join(fmt(x) for x in v['counts'])}; exact P_C(m), m=1..: "
                + " ".join(fmt(x, 3) for x in v["p_c_exact"]))
    if name == "distance_bounds":
        if v["low"] == v["high"]:
            return f"{label}: d = {v['low']}"
        return f"{label}: {v['low']} <= d <= {v['high']}"
    if name == "failure_probability":
        pairs = ", ".join(f"Phi({fmt(e)})={fmt(p)}" for e, p in zip(v["rates"], v["phi"], strict=True))
        return f"{label}: {pairs}; e_{fmt(v['delta'])}={fmt(v['e_delta'])}"
    if name == "stabilizer_weights":
        return f"{label}: mean {fmt(v['mean'])}, min {v['min']}, max {v['max']}"
    if name == "stabilizers":
        return f"{label}: " + " ".join(v)
    return f"{label}: {fmt(v)}"


def _render(d, ctx):
    head = f"code_analyze: [[{d['n']}, {d['k']}]] code from {d['origin']}"
    body = [_show(r["name"], r["property"], r["value"]) for r in d["results"]]
    tail = f"saved: {d['saved']}" if d.get("saved") else ""
    return lines(head, *body, tail)


register(ToolDef(
    name="code_analyze", stack="code", summary="Any property of a stabilizer code, on request.",
    description=(
        "Analyse a stabilizer code: given as Pauli strings (stabilizers), saved (source: a "
        "traj_run record, an ensemble run with code snapshots, a .code.npz or a .state.npz with "
        "reference_qubits), or an open state (state_id with reference_qubits). properties are "
        "call strings: "
        "summary, first_failure(samples=512), exhaustive_counts(3), distance_bounds, "
        "mean_first_failure, failure_probability(rates=[0.01,0.05], delta=0.01), "
        "lost_information(0:4), is_correctable([0,1]), stabilizer_weights, stabilizers, "
        "contiguous_distance (help_list(what=code_properties)). Example: {\"stabilizers\": "
        "[\"XXXX\", \"ZZZZ\"], \"properties\": [\"summary\", \"distance_bounds\"]}"
    ),
    params=(
        Param("stabilizers", "array", "Generators as Pauli strings, e.g. [\"XXXX\", \"ZZZZ\"].",
              items="string"),
        Param("source", "string", "A saved code, trajectory record, ensemble run or state file."),
        Param("state_id", "string", "An open state (with reference)."),
        Param("reference_qubits", "string", "The reference qubits of a state, e.g. 8:16."),
        Param("trajectory", "integer", "Trajectory number in a saved ensemble (default: its first).",
              minimum=0),
        Param("evaluation", "integer", "Snapshot index (-1 = the last).", default=-1),
        Param("properties", "array", "Property calls (default summary).", items="string"),
        Param("save_to", "string", "Also save the code as a .code.npz file."),
    ),
    handler=_analyze, render=_render, long_running=True,
    examples=({"stabilizers": ["XXXX", "ZZZZ"], "properties": ["summary", "distance_bounds"]},),
))
```

Add `code` to `tools/__init__.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_code_tool.py tests/agent/test_help_tools.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent tests/agent
git commit -m "agent: code_analyze (every StabilizerCode property on request; codes from anywhere)"
```

---

### Task 22: Graph tools

**Files:**
- Create: `src/sparsegf2/agent/tools/graph.py`
- Modify: `src/sparsegf2/agent/tools/__init__.py`
- Test: `tests/agent/test_graph_tools.py`

**Interfaces:**
- Consumes: `params.GRAPH_PARAMS`, `params.graph_dict`, `api._build_graph`, `graphs.save/load/stats/shortcut_fraction/degrees`, `MAX_STATS_NODES`.
- Produces: the tools `graph_build` and `graph_stats`, and `collection(ctx, args, n) -> list[np.ndarray]`.

- [ ] **Step 1: Write the failing tests**

```python
"""graph_build and graph_stats."""

import math

from sparsegf2 import graphs


def test_build_then_stats_from_the_file(run_tool, ctx):
    out = run_tool("graph_build", num_nodes=16, graph="watts_strogatz", graph_count=3,
                   graph_seed=1, output_file="ws.graphs.json")
    assert not out.is_error and "3 graphs" in out.text
    n, gs, _ = graphs.load(ctx.root / "ws.graphs.json")
    assert n == 16 and len(gs) == 3
    stats = run_tool("graph_stats", num_nodes=16, graph="file", graph_file="ws.graphs.json")
    assert "lambda2" in stats.text and "3 graphs" in stats.text


def test_cycle_stats_match_the_closed_form(run_tool):
    out = run_tool("graph_stats", num_nodes=10, graph="cycle")
    assert f"{2 - 2 * math.cos(2 * math.pi / 10):.4g}" in out.text


def test_shortcut_fraction_for_watts_strogatz(run_tool):
    out = run_tool("graph_stats", num_nodes=32, graph="watts_strogatz", graph_beta=0.0,
                   graph_count=2)
    assert "shortcut_fraction" in out.text and "0+-0" in out.text.replace(" ", "")


def test_show_edges_small(run_tool):
    out = run_tool("graph_build", num_nodes=4, graph="cycle", output_file="c.graphs.npz",
                   show_edges=True)
    assert "[0,1]" in out.text.replace(" ", "")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_graph_tools.py -q`
Expected: FAIL with `KeyError: 'graph_build'`

- [ ] **Step 3: Implement `src/sparsegf2/agent/tools/graph.py`**

```python
"""graph_build and graph_stats: interaction graphs of any family or file."""

from __future__ import annotations

from typing import Any

import numpy as np

from sparsegf2 import analysis, api, graphs
from sparsegf2.agent.params import GRAPH_PARAMS, graph_dict
from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.render import lines, pm, table

NUM_NODES = Param("num_nodes", "integer", "Number of nodes (qubits).", required=True, minimum=2)
MAX_GRAPHS = 1000


def collection(ctx, a: dict[str, Any], n: int) -> list[np.ndarray]:
    g = graph_dict(a, n, ctx)
    if g["family"] == "file":
        num, gs, _ = graphs.load(g["path"])
        if num != n:
            raise ValueError(f"graph_file has num_nodes {num}, not {n}")
        return gs
    built = api._build_graph(g, n)
    return built if isinstance(built, list) else [built]


def _build(ctx, a):
    n = a["num_nodes"]
    gs = collection(ctx, a, n)
    path = ctx.path(a["output_file"], what="output_file")
    path.parent.mkdir(parents=True, exist_ok=True)
    graphs.save(path, gs, n)
    deg = np.concatenate([graphs.degrees(g, n) for g in gs])
    out = {"output_file": ctx.show(path), "num_nodes": n, "graphs": len(gs),
           "edges_mean": float(np.mean([len(g) for g in gs])),
           "degree_min": int(deg.min()), "degree_mean": float(deg.mean()),
           "degree_max": int(deg.max())}
    if a.get("show_edges"):
        if n > 64:
            raise ValueError("show_edges lists edges only for graphs of up to 64 nodes")
        out["edges_of_graph_0"] = gs[0].tolist()
    return out


def _render_build(d, ctx):
    head = (f"graph_build: {d['graphs']} graphs on {d['num_nodes']} nodes -> {d['output_file']} "
            f"(edges {d['edges_mean']:.4g} per graph, degree {d['degree_min']}..{d['degree_max']}, "
            f"mean {d['degree_mean']:.4g})")
    edges = f"edges of graph 0: {d['edges_of_graph_0']}" if "edges_of_graph_0" in d else ""
    return lines(head, edges, f'next: graph_stats(num_nodes={d["num_nodes"]}, graph="file", '
                              f'graph_file="{d["output_file"]}")')


def _stats(ctx, a):
    n = a["num_nodes"]
    gs = collection(ctx, a, n)
    if len(gs) > MAX_GRAPHS:
        gs = gs[:MAX_GRAPHS]
    rows = [graphs.stats(g, n) for g in gs]
    keys = ("lambda2", "mean_path_length", "clustering", "degree_mean")
    out: dict[str, Any] = {"num_nodes": n, "graphs": len(gs),
                           "connected_fraction": float(np.mean([r["connected"] for r in rows]))}
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            m, e = analysis.mean_and_error(vals)
            out[key] = (float(m), float(e))
    family = a.get("graph", "cycle")
    if family in ("watts_strogatz", "newman_watts"):
        k = graph_dict(a, n, ctx)["k"]
        sf = np.array([graphs.shortcut_fraction(g, n, k) for g in gs])
        m, e = analysis.mean_and_error(sf)
        out["shortcut_fraction"] = (float(m), float(e))
    return out


def _render_stats(d, ctx):
    rows = [[k, pm(*d[k])] for k in ("lambda2", "mean_path_length", "clustering", "degree_mean",
                                     "shortcut_fraction") if k in d]
    return lines(f"graph_stats: {d['graphs']} graphs on {d['num_nodes']} nodes, "
                 f"{d['connected_fraction']:.3g} connected (mean +- stderr over graphs)",
                 table(["statistic", "value"], rows))


register(ToolDef(
    name="graph_build", stack="graph", summary="Build a graph collection and save it to a file.",
    description=(
        "Build a collection of interaction graphs (any family; graph i from seed "
        "(graph_seed, i)) and save it as .json or .npz, to inspect or to reuse with graph=file. "
        "Example: {\"num_nodes\": 64, \"graph\": \"watts_strogatz\", \"graph_k\": 4, "
        "\"graph_beta\": 0.1, \"graph_count\": 50, \"output_file\": \"graphs/ws64.graphs.json\"}"
    ),
    params=(NUM_NODES, *GRAPH_PARAMS,
            Param("output_file", "string", "Destination (.json or .npz).", required=True),
            Param("show_edges", "boolean", "Also list graph 0's edges (up to 64 nodes).",
                  default=False)),
    handler=_build, render=_render_build, read_only=False,
    examples=({"num_nodes": 8, "graph": "cycle", "output_file": "example.graphs.json"},),
))

register(ToolDef(
    name="graph_stats", stack="graph", summary="lambda_2, path length, clustering, shortcuts.",
    description=(
        "Statistics of a graph collection, mean +- standard error over its graphs: algebraic "
        "connectivity lambda_2, fraction connected, mean shortest-path length, clustering, "
        "degrees, and for ring-based families the fraction of shortcut edges (n <= 4096). "
        "Example: {\"num_nodes\": 64, \"graph\": \"watts_strogatz\", \"graph_beta\": 0.1, "
        "\"graph_count\": 20}"
    ),
    params=(NUM_NODES, *GRAPH_PARAMS),
    handler=_stats, render=_render_stats, long_running=True,
    examples=({"num_nodes": 16, "graph": "watts_strogatz", "graph_count": 3},),
))
```

Add `graph` to `tools/__init__.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_graph_tools.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent tests/agent/test_graph_tools.py
git commit -m "agent: graph_build and graph_stats (any family or file)"
```

---

### Task 23: `analysis_summary` and `analysis_probe`, with result loading and selections

**Files:**
- Create: `src/sparsegf2/agent/tools/analysis.py`
- Modify: `src/sparsegf2/agent/tools/__init__.py`
- Test: `tests/agent/test_analysis_tools.py`

**Interfaces:**
- Consumes: `api._load_directory`, `api._load_result_file`, `api.load_results`, `tools.ensemble.render_summary`, `analysis.mean_and_error`, `grammar.parse_qubits`, `grammar.parse_checkpoints`.
- Produces (in `tools.analysis`):
  - The tools `analysis_summary` and `analysis_probe`.
  - `load_result(ctx, source) -> tuple[EnsembleResult, dict | None, Path]`.
  - `indices(value, what) -> np.ndarray`.
  - `select(res, *, layer, where=None, trajectories=None, graphs=None) -> np.ndarray` (a bool mask over rows).
  - `probe_by_kind(res, kind) -> str | None`.
  - `at_layer(res, name, layer) -> int | None` (the evaluation index).
  - `SELECTION_PARAMS`.

- [ ] **Step 1: Write the failing tests**

```python
"""analysis_summary and analysis_probe against direct numpy on the saved arrays."""

import numpy as np
import pytest

from sparsegf2 import api, mipt

SPEC = {"num_system": 8, "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 1,
                                    "realizations": 4},
        "circuits_per_graph": 3, "layers": 12, "measurement_rate": 0.15,
        "reference": "purification", "checkpoints": 4,
        "probes": [{"kind": "code_dimension"}, {"kind": "half_cut_entropy"},
                   {"kind": "entropy_profile"}]}


@pytest.fixture
def run_dir(ctx):
    api.run_ensemble(SPEC, seed=3, output_dir=str(ctx.root / "r"))
    return "r"


def test_summary(run_tool, run_dir):
    out = run_tool("analysis_summary", source=run_dir)
    assert out.text.startswith("analysis_summary: n=8, layers=12") and "code_dimension" in out.text


def test_probe_mean_matches_numpy(run_tool, ctx, run_dir):
    res = mipt.EnsembleResult.load(ctx.root / "r" / "result.npz")
    out = run_tool("analysis_probe", source=run_dir, probe="code_dimension", statistic="mean")
    k = res.values["code_dimension"]
    assert f"{k[:, -1].mean():.4g}" in out.text


def test_where_trajectories_and_graphs(run_tool, ctx, run_dir):
    res = mipt.EnsembleResult.load(ctx.root / "r" / "result.npz")
    k = res.values["code_dimension"][:, -1]
    out = run_tool("analysis_probe", source=run_dir, probe="half_cut_entropy", where="code",
                   depths="[12]")
    s = res.values["half_cut_entropy"][k > 0, -1]
    assert f"{s.mean():.4g}" in out.text
    g = run_tool("analysis_probe", source=run_dir, probe="code_dimension", graph_range="0:2",
                 depths="[12]")
    assert f"{k[res.graph_index < 2].mean():.4g}" in g.text
    bad = run_tool("analysis_probe", source=run_dir, probe="code_dimension", where="k >> 0")
    assert bad.is_error and "where" in bad.text


def test_vector_probe_and_statistics(run_tool, run_dir):
    out = run_tool("analysis_probe", source=run_dir, probe="entropy_profile")
    assert "m=0..8" in out.text
    q = run_tool("analysis_probe", source=run_dir, probe="code_dimension", statistic="quantiles")
    assert "q50" in q.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_analysis_tools.py -q`
Expected: FAIL with `KeyError: 'analysis_summary'`

- [ ] **Step 3: Implement `src/sparsegf2/agent/tools/analysis.py` (Part 1)**

```python
"""Analysis of saved ensemble results: summaries, any probe, erasure, purification, comparisons."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2 import analysis, api
from sparsegf2.agent.grammar import parse_checkpoints, parse_qubits
from sparsegf2.agent.params import SOURCE
from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.render import curve, fmt, lines, pm, table

_WHERE = re.compile(r"^\s*([A-Za-z_][\w\[\]#.]*)\s*(<=|>=|==|!=|<|>)\s*(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*$")
_OPS = {"<": np.less, ">": np.greater, "<=": np.less_equal, ">=": np.greater_equal,
        "==": np.equal, "!=": np.not_equal}

SELECTION_PARAMS = (
    Param("where", "string",
          "Select trajectories: code (k>0), purified, unpurified, or '<scalar probe> <op> <number>' "
          "e.g. 'half_cut_entropy > 2'; evaluated at each depth."),
    Param("trajectory_range", "string", "Trajectory numbers to use, e.g. 0:100 or [3,7]."),
    Param("graph_range", "string", "Graph indices to use, e.g. 0:10."),
)


def load_result(ctx, source: str):
    path = ctx.path(source, what="source", must_exist=True)
    if path.is_dir():
        if (path / "sweep.json").exists():
            raise ValueError(f"{ctx.show(path)} is a sweep: use analysis_compare(source=...) or "
                             "point source at one of its point folders")
        res, meta = api._load_directory(path)
        if res is None:
            raise ValueError(f"no saved results in {ctx.show(path)}")
        return res, meta, path
    return api._load_result_file(path), None, path


def indices(value: Any, what: str) -> np.ndarray:
    q = parse_qubits(value, what=what)
    if isinstance(q, str):
        parts = [int(x) for x in q.split(":")]
        return np.arange(*parts)
    return np.asarray(q, dtype=np.int64)


def probe_by_kind(res, kind: str) -> str | None:
    for p in res.spec.probes:
        if p.kind == kind:
            return p.name
    return None


def at_layer(res, name: str, layer: int) -> int | None:
    hits = np.flatnonzero(np.asarray(res.layers(name)) == layer)
    return int(hits[0]) if hits.size else None


def select(res, *, layer: int, where: str | None = None, trajectories: Any = None,
           graphs: Any = None) -> np.ndarray:
    mask = np.ones(res.num_trajectories, dtype=bool)
    if trajectories is not None:
        mask &= np.isin(np.asarray(res.trajectories), indices(trajectories, "trajectories"))
    if graphs is not None:
        mask &= np.isin(np.asarray(res.graph_index), indices(graphs, "graphs"))
    if not where or where == "all":
        return mask
    w = where.strip()
    if w in ("purified", "unpurified"):
        tau = res.purification_layer
        if tau is None:
            raise ValueError("where=purified needs a reference (purification data)")
        purified = (tau >= 0) & (tau <= layer)
        return mask & (purified if w == "purified" else ~purified)
    if w == "code":
        name = probe_by_kind(res, "code_dimension")
        if name is None:
            raise ValueError("where=code needs the code_dimension probe in this run")
        c = at_layer(res, name, layer)
        if c is None:
            raise ValueError(f"where=code: code_dimension was not recorded at layer {layer}")
        return mask & (res.values[name][:, c] > 0)
    m = _WHERE.match(w)
    if m is None:
        raise ValueError(f"where={where!r}: use code, purified, unpurified or "
                         "'<scalar probe> <op> <number>' with op one of < > <= >= == !=")
    name, op, number = m.groups()
    if name not in res.values or np.ndim(res.values[name]) != 2:
        raise ValueError(f"where: {name!r} is not a scalar probe of this run "
                         f"(scalar probes: {[k for k, v in res.values.items() if np.ndim(v) == 2]})")
    c = at_layer(res, name, layer)
    if c is None:
        raise ValueError(f"where: {name} was not recorded at layer {layer}")
    return mask & _OPS[op](res.values[name][:, c], float(number))


def _summary(ctx, a):
    path = ctx.path(a["source"], what="source", must_exist=True)
    return {"summary": api.load_results(str(path)), "source": ctx.show(path)}


def _render_summary(d, ctx):
    from sparsegf2.agent.tools.ensemble import render_summary

    return render_summary("analysis_summary", d["summary"], output_dir=d["source"])


def _wanted_layers(available: np.ndarray, depths: Any) -> list[int]:
    if depths is None:
        if len(available) <= 12:
            return [int(x) for x in available]
        idx = np.unique(np.linspace(0, len(available) - 1, 12).round().astype(int))
        return [int(available[i]) for i in idx]
    asked = parse_checkpoints(depths, what="depths")
    if not isinstance(asked, list):
        raise ValueError("depths must be a list of layers, e.g. [64, 128]")
    missing = sorted(set(asked) - {int(x) for x in available})
    if missing:
        raise ValueError(f"depths {missing} were not recorded; recorded: {available.tolist()}")
    return asked


def _probe(ctx, a):
    res, _, path = load_result(ctx, a["source"])
    name = a["probe"]
    if name not in res.values:
        raise ValueError(f"probe {name!r} is not in this run; probes: {sorted(res.values)}")
    values = np.asarray(res.values[name], dtype=float)
    layers = np.asarray(res.layers(name))
    stat = a["statistic"]
    rows = []
    for layer in _wanted_layers(layers, a.get("depths")):
        c = int(np.flatnonzero(layers == layer)[0])
        mask = select(res, layer=layer, where=a.get("where"), trajectories=a.get("trajectory_range"),
                      graphs=a.get("graph_range"))
        v = values[mask, c]
        if values.ndim == 3:  # vector probe: per index
            v = v[~np.isnan(v).any(axis=-1)] if v.size else v
            mean = v.mean(axis=0) if len(v) else np.full(values.shape[-1], np.nan)
            rows.append({"layer": layer, "count": int(len(v)), "curve": mean.tolist()})
            continue
        v = v[~np.isnan(v)]
        row: dict[str, Any] = {"layer": layer, "count": int(v.size)}
        if not v.size:
            rows.append(row)
            continue
        if stat == "mean":
            m, e = analysis.mean_and_error(v)
            row["mean"], row["stderr"] = float(m), float(e)
        elif stat == "median":
            row["median"] = float(np.median(v))
        elif stat == "quantiles":
            for q in (10, 25, 50, 75, 90):
                row[f"q{q}"] = float(np.percentile(v, q))
        elif stat == "fraction_positive":
            row["fraction_positive"] = float(np.mean(v > 0))
        else:  # histogram of the most common values
            vals, counts = np.unique(v, return_counts=True)
            top = np.argsort(-counts)[:8]
            row["histogram"] = {fmt(vals[i]): int(counts[i]) for i in sorted(top)}
        rows.append(row)
    return {"source": ctx.show(path), "probe": name, "statistic": stat, "vector": values.ndim == 3,
            "rows": rows}


def _render_probe(d, ctx):
    head = f"analysis_probe: {d['probe']} ({d['statistic']}) in {d['source']}"
    if d["vector"]:
        body = [f"layer {r['layer']} ({r['count']} trajectories): " + curve(r["curve"])
                for r in d["rows"]]
        return lines(head, *body)
    keys = [k for k in d["rows"][0] if k not in ("layer", "count")] if d["rows"] else []
    if d["statistic"] == "mean":
        rows = [[r["layer"], r["count"], pm(r.get("mean"), r.get("stderr"))] for r in d["rows"]]
        return lines(head, table(["layer", "count", "mean"], rows))
    if d["statistic"] == "histogram":
        return lines(head, *[f"layer {r['layer']}: {r.get('histogram')}" for r in d["rows"]])
    rows = [[r["layer"], r["count"], *[r.get(k) for k in keys]] for r in d["rows"]]
    return lines(head, table(["layer", "count", *keys], rows))


register(ToolDef(
    name="analysis_summary", stack="analysis", summary="Summary of a saved ensemble run.",
    description=(
        "Summarise a saved ensemble run (an output_dir or a result .npz, including the saved "
        "chunks of a running job): every probe vs depth, purification statistics, the spec. "
        "Example: {\"source\": \"runs/ensemble_run-3f2a9c1b7d\"}"
    ),
    params=(SOURCE,), handler=_summary, render=_render_summary,
))

register(ToolDef(
    name="analysis_probe", stack="analysis", summary="Any statistic of any probe vs depth.",
    description=(
        "A statistic of one probe of a saved run at each depth (mean +- stderr, median, "
        "quantiles, histogram, fraction_positive), over the trajectories you select (where, "
        "trajectory_range, graph_range). A vector probe gives its mean curve. Example: {\"source\": "
        "\"runs/ensemble_run-3f2a9c1b7d\", \"probe\": \"half_cut_entropy\", \"where\": \"code\"}"
    ),
    params=(SOURCE,
            Param("probe", "string", "Probe name as saved (analysis_summary lists them).",
                  required=True),
            Param("statistic", "string", "Which statistic.", default="mean",
                  enum=("mean", "median", "quantiles", "histogram", "fraction_positive")),
            Param("depths", "string", "Layers to show, e.g. [64,128] (default: up to 12)."),
            *SELECTION_PARAMS),
    handler=_probe, render=_render_probe,
))
```

Add `analysis` to `tools/__init__.py`. Both tools need a saved run, so add them to `EXAMPLE_EXEMPT` in Task 28 (they are covered here).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_analysis_tools.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent tests/agent/test_analysis_tools.py
git commit -m "agent: analysis_summary and analysis_probe with where/trajectories/graphs selections"
```

---

### Task 24: `analysis_erasure` and `analysis_purification`

**Files:**
- Modify: `src/sparsegf2/agent/tools/analysis.py`
- Test: `tests/agent/test_analysis_erasure.py`

**Interfaces:**
- Consumes: `load_result`, `select`, `probe_by_kind`, `at_layer`, and from `analysis`: `erasure_failure_probability`, `erasure_tolerance`, `mean_first_failure`, `distance_bounds`, `hybrid_failure_by_size`, `junction_dip`, `kaplan_meier`, `mean_and_error`.
- Produces:
  - The tools `analysis_erasure` and `analysis_purification`.
  - `erasure_at(res, layer, *, rates, delta, mask=None) -> dict` (keys `layer`, `count`, `k_over_n`, `q_plus`, `mu_D`, `distance`, `phi_all`, `phi_code`, `e_delta`, `p_c_code`, `dips`).
  - `km_curve(res) -> dict` (Kaplan-Meier `times`, `survival`, `stderr`), and `tau_median_all(km) -> float | None`.

- [ ] **Step 1: Write the failing tests**

```python
"""analysis_erasure and analysis_purification against the analysis module called directly."""

import numpy as np
import pytest

from sparsegf2 import analysis, api, mipt

SPEC = {"num_system": 12, "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": 0,
                                     "realizations": 6},
        "circuits_per_graph": 4, "layers": 24, "measurement_rate": 0.05,
        "schedule": "random_edges", "reference": "purification", "checkpoints": 12,
        "probes": [{"kind": "code_dimension"},
                   {"kind": "erasure_first_failure", "samples": 64},
                   {"kind": "uncorrectable_region_counts", "max_size": 2}]}


@pytest.fixture
def erasure_dir(ctx):
    api.run_ensemble(SPEC, seed=1, output_dir=str(ctx.root / "e"))
    return mipt.EnsembleResult.load(ctx.root / "e" / "result.npz")


def test_phi_e_delta_and_mu_match_the_library(run_tool, erasure_dir):
    res = erasure_dir
    out = run_tool("analysis_erasure", source="e", rates=[0.05, 0.1], delta=0.01)
    assert not out.is_error, out.text
    n = 12
    first = res.values["erasure_first_failure"][:, -1]
    counts = res.values["uncorrectable_region_counts[2]"][:, -1]
    k = res.values["code_dimension"][:, -1]
    phi = analysis.erasure_failure_probability(first, n, [0.05, 0.1], exact_counts=counts)
    code = k > 0
    assert f"{phi[:, 1].mean():.4g}" in out.text
    if code.any():
        e_d = analysis.erasure_tolerance(first[code], n, 0.01, exact_counts=counts[code])
        assert f"{e_d:.4g}" in out.text
        mu = analysis.mean_first_failure(first[code], n, exact_counts=counts[code]).mean()
        assert f"{mu:.4g}" in out.text


def test_requires_first_failure(run_tool, ctx):
    api.run_ensemble({**SPEC, "probes": [{"kind": "code_dimension"}]}, seed=1,
                     output_dir=str(ctx.root / "k"))
    out = run_tool("analysis_erasure", source="k")
    assert out.is_error and "erasure_first_failure" in out.text


def test_purification(run_tool, ctx):
    spec = {"num_system": 8, "graph": {"family": "cycle"}, "layers": 64, "measurement_rate": 0.3,
            "reference": "single", "checkpoints": "every_layer", "stop_when_purified": True,
            "probes": [{"kind": "code_dimension"}]}
    api.run_ensemble(spec, 200, seed=2, output_dir=str(ctx.root / "p"))
    res = mipt.EnsembleResult.load(ctx.root / "p" / "result.npz")
    tau = res.purification_layer
    out = run_tool("analysis_purification", source="p", at_layers=[8, 32])
    assert f"censored {np.mean(tau < 0):.3g}" in out.text
    assert "survival at layer 8" in out.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_analysis_erasure.py -q`
Expected: FAIL with `KeyError: 'analysis_erasure'`

- [ ] **Step 3: Implement (append to `tools/analysis.py`)**

```python
DEFAULT_RATES = [0.01, 0.02, 0.05, 0.1]


def erasure_at(res, layer: int, *, rates, delta: float, mask=None) -> dict[str, Any]:
    n = res.spec.num_system
    ff = probe_by_kind(res, "erasure_first_failure")
    if ff is None:
        raise ValueError("this run has no erasure_first_failure probe; add "
                         "erasure_first_failure(samples=512) to its probes")
    c = at_layer(res, ff, layer)
    rows = np.ones(res.num_trajectories, bool) if mask is None else mask
    first = np.asarray(res.values[ff])[rows, c]
    cn = probe_by_kind(res, "uncorrectable_counts")
    cc = at_layer(res, cn, layer) if cn else None
    counts = np.asarray(res.values[cn])[rows, cc] if cc is not None else None
    kn = probe_by_kind(res, "code_dimension")
    kc = at_layer(res, kn, layer) if kn else None
    k = np.asarray(res.values[kn])[rows, kc] if kc is not None else None
    code = (k > 0) if k is not None else (first.min(axis=-1) <= n)
    out: dict[str, Any] = {"layer": int(layer), "count": int(rows.sum()),
                           "q_plus": float(code.mean()) if code.size else None,
                           "k_over_n": analysis.mean_and_error(k / n) if k is not None and k.size else None}
    phi = analysis.erasure_failure_probability(first, n, rates, exact_counts=counts)
    out["phi_all"] = [analysis.mean_and_error(phi[:, i]) for i in range(len(rates))]
    if code.any():
        cnt = counts[code] if counts is not None else None
        out["phi_code"] = [analysis.mean_and_error(phi[code, i]) for i in range(len(rates))]
        out["e_delta"] = analysis.erasure_tolerance(first[code], n, delta, exact_counts=cnt)
        out["mu_D"] = analysis.mean_and_error(
            analysis.mean_first_failure(first[code], n, exact_counts=cnt))
        low, high = analysis.distance_bounds(first[code], n, exact_counts=cnt)
        exact = low == high
        dist: dict[str, Any] = {"exact_fraction": float(exact.mean()),
                                "upper_median": float(np.median(high))}
        if counts is not None:
            w = counts.shape[-1]
            dist["w"] = int(w)
            dist["d_values"] = {int(d): int((low[exact] == d).sum()) for d in np.unique(low[exact])}
        out["distance"] = dist
        out["p_c_code"] = analysis.hybrid_failure_by_size(first[code], n, exact_counts=cnt).mean(axis=0)
        out["dips"] = int(analysis.junction_dip(first[code], n, cnt).sum()) if cnt is not None else 0
    return out


def _erasure(ctx, a):
    res, _, path = load_result(ctx, a["source"])
    ff = probe_by_kind(res, "erasure_first_failure")
    if ff is None:
        erasure_at(res, 0, rates=[0.1], delta=0.01)  # raises the explanatory error
    rates = a.get("rates") or DEFAULT_RATES
    layers = np.asarray(res.layers(ff))
    rows = []
    for layer in _wanted_layers(layers, a.get("depths")):
        mask = select(res, layer=layer, where=a.get("where"), trajectories=a.get("trajectory_range"),
                      graphs=a.get("graph_range"))
        rows.append(erasure_at(res, layer, rates=rates, delta=a["delta"], mask=mask))
    return {"source": ctx.show(path), "n": res.spec.num_system, "rates": rates,
            "delta": a["delta"], "rows": rows}


def _render_erasure(d, ctx):
    rates, n = d["rates"], d["n"]
    cols = ["layer", "count", "k/n", "q+", "mu_D(k>0)"] + [f"Phi({fmt(e)})" for e in rates] + [
        f"Phi({fmt(rates[-1])})|k>0", f"e_{fmt(d['delta'])}|k>0"]
    rows = []
    for r in d["rows"]:
        kn = pm(*r["k_over_n"]) if r.get("k_over_n") else "-"
        rows.append([r["layer"], r["count"], kn, r.get("q_plus"),
                     pm(*r["mu_D"]) if r.get("mu_D") else "-",
                     *[pm(*p) for p in r["phi_all"]],
                     pm(*r["phi_code"][-1]) if r.get("phi_code") else "-",
                     r.get("e_delta")])
    parts = [f"analysis_erasure: n={n}, {d['source']}", table(cols, rows)]
    last = d["rows"][-1] if d["rows"] else None
    if last and last.get("distance"):
        dist = last["distance"]
        line = (f"distance at layer {last['layer']} (k>0): exact for {dist['exact_fraction']:.3g} of "
                f"codes")
        if "d_values" in dist:
            line += f" (d <= {dist['w']}: {dist['d_values']}); others d > {dist['w']}"
        line += f"; median upper bound min_j D_j = {fmt(dist['upper_median'])}"
        parts.append(line)
        parts.append(f"P_C(m) at layer {last['layer']} (k>0, hybrid): " + curve(last["p_c_code"]))
        if last.get("dips"):
            parts.append(f"note: {last['dips']} code(s) have sampled P_C(w+1) < exact P_C(w) "
                         "(sampling noise): use more orderings (erasure_first_failure(samples=...))")
    return lines(*parts)


def km_curve(res) -> dict[str, np.ndarray]:
    tau = res.purification_layer
    if tau is None:
        raise ValueError("this run has no reference, so no purification times")
    tau = np.asarray(tau)
    final = np.asarray(res.final_layer)
    return analysis.kaplan_meier(np.where(tau >= 0, tau, final), tau >= 0)


def tau_median_all(km: dict[str, np.ndarray]) -> float | None:
    below = np.flatnonzero(km["survival"] <= 0.5)
    return float(km["times"][below[0]]) if below.size else None


def _purification(ctx, a):
    res, _, path = load_result(ctx, a["source"])
    km = km_curve(res)
    tau = np.asarray(res.purification_layer)
    purified = tau[tau >= 0]
    out: dict[str, Any] = {"source": ctx.show(path), "trajectories": int(tau.size),
                           "censored_fraction": float(np.mean(tau < 0)),
                           "purified": int(purified.size), "tau_median_all": tau_median_all(km)}
    if purified.size:
        out["tau_mean_purified"] = analysis.mean_and_error(purified)
        out["tau_quantiles_purified"] = {f"q{int(q * 100)}": float(np.quantile(purified, q))
                                         for q in a["quantiles"]}
    surv = {}
    for layer in a.get("at_layers") or []:
        i = np.searchsorted(km["times"], layer, side="right") - 1
        surv[int(layer)] = float(km["survival"][i]) if i >= 0 else 1.0
    out["survival_at"] = surv
    out["km_times"], out["km_survival"] = km["times"], km["survival"]
    return out


def _render_purification(d, ctx):
    parts = [f"analysis_purification: {d['trajectories']} trajectories in {d['source']}",
             f"censored {d['censored_fraction']:.3g}, purified {d['purified']}, tau median overall "
             f"{fmt(d['tau_median_all'])} (Kaplan-Meier; '-' if under half purified)"]
    if "tau_mean_purified" in d:
        parts.append("among purified: mean " + pm(*d["tau_mean_purified"]) + ", quantiles "
                     + ", ".join(f"{k}={fmt(v)}" for k, v in d["tau_quantiles_purified"].items()))
    for layer, s in d["survival_at"].items():
        parts.append(f"survival at layer {layer}: {fmt(s)}")
    if len(d["km_times"]):
        idx = np.unique(np.linspace(0, len(d["km_times"]) - 1, min(9, len(d["km_times"]))).round().astype(int))
        parts.append("survival curve (layer: fraction unpurified): " + ", ".join(
            f"{fmt(d['km_times'][i])}: {fmt(d['km_survival'][i], 3)}" for i in idx))
    return lines(*parts)


register(ToolDef(
    name="analysis_erasure", stack="analysis", summary="Phi(e), e_delta, mu_D, distance, P_C(m).",
    description=(
        "Erasure protection of the codes in a saved run with erasure_first_failure (and "
        "optionally uncorrectable_region_counts for exact small sizes): per depth, the code "
        "rate k/n, q+ (fraction with k>0), mu_D, Phi(e) = probability that erasing each qubit "
        "with probability e is uncorrectable (all circuits, and k>0), e_delta = sup{e : Phi(e) "
        "<= delta} for k>0, distance bounds and P_C(m). Example: {\"source\": "
        "\"runs/ensemble_run-3f2a9c1b7d\", \"rates\": [0.01, 0.05, 0.1], \"delta\": 0.01}"
    ),
    params=(SOURCE,
            Param("rates", "array", "Erasure rates e (default 0.01, 0.02, 0.05, 0.1).",
                  items="number", minimum=0, maximum=1),
            Param("delta", "number", "Failure probability for e_delta.", default=0.01,
                  minimum=0, maximum=1),
            Param("depths", "string", "Layers, e.g. [128,256] (default: up to 12)."),
            *SELECTION_PARAMS),
    handler=_erasure, render=_render_erasure, long_running=True,
))

register(ToolDef(
    name="analysis_purification", stack="analysis", summary="Purification times with censoring.",
    description=(
        "Purification of the reference in a saved run: censored fraction, tau among purified "
        "trajectories, the median over all (Kaplan-Meier, defined when over half purified), "
        "survival at chosen layers and the survival curve. Never averages censored times. "
        "Example: {\"source\": \"runs/ensemble_run-3f2a9c1b7d\", \"at_layers\": [64, 256]}"
    ),
    params=(SOURCE,
            Param("quantiles", "array", "Quantiles of tau among purified.", items="number",
                  default=[0.1, 0.25, 0.5, 0.75, 0.9], minimum=0, maximum=1),
            Param("at_layers", "array", "Layers at which to report survival.", items="integer")),
    handler=_purification, render=_render_purification,
))
```

Both tools need a saved run, so they go in `EXAMPLE_EXEMPT` in Task 28.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_analysis_erasure.py tests/agent/test_analysis_tools.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent/tools/analysis.py tests/agent/test_analysis_erasure.py
git commit -m "agent: analysis_erasure (Phi, e_delta, mu_D, distance, P_C) and analysis_purification"
```

---

### Task 25: `analysis_compare` and the metrics

**Files:**
- Modify: `src/sparsegf2/agent/tools/analysis.py`, `tests/agent/test_help_tools.py` (add the `metrics` case)
- Test: `tests/agent/test_analysis_compare.py`

**Interfaces:**
- Consumes: `load_result`, `erasure_at`, `km_curve`, `tau_median_all`, `select`, `grammar.parse_calls/bind/plain`.
- Produces:
  - The tool `analysis_compare`.
  - `METRICS: dict[str, tuple[tuple[str, ...], str, Callable]]`, mapping name to (parameters, description, `fn(res, layer, **kw) -> (value, err or None)`).
  - `metric_value(res, call, layer) -> tuple[float | None, float | None]`.
  - `sources_of(ctx, a) -> list[Path]` (a list, or the points of a sweep).
  - `differing(specs) -> list[str]` (the flattened spec keys that differ between runs).

- [ ] **Step 1: Write the failing tests**

```python
"""analysis_compare over runs and over a sweep."""

import numpy as np

from sparsegf2 import api, mipt

BASE = {"num_system": 8, "graph": {"family": "cycle"}, "layers": 8, "reference": "purification",
        "checkpoints": 4, "probes": [{"kind": "code_dimension"},
                                     {"kind": "erasure_first_failure", "samples": 16}]}


def test_two_runs(run_tool, ctx):
    for p, name in ((0.05, "a"), (0.3, "b")):
        api.run_ensemble({**BASE, "measurement_rate": p}, 30, seed=1, output_dir=str(ctx.root / name))
    out = run_tool("analysis_compare", sources=["a", "b"],
                   metrics=["mean(code_dimension)", "phi(e=0.1)", "q_plus", "mu_D"])
    assert not out.is_error, out.text
    assert "measurement_rate" in out.text.splitlines()[1]
    res = mipt.EnsembleResult.load(ctx.root / "a" / "result.npz")
    assert f"{res.values['code_dimension'][:, -1].mean():.4g}" in out.text


def test_sweep_source(run_tool):
    sweep = run_tool("ensemble_sweep", num_system=8, layers=4, trajectories=12,
                     reference="purification", probes=["code_dimension"], seed=2,
                     grid="measurement_rate=[0.1, 0.3]", background="never")
    out = run_tool("analysis_compare", source=sweep.data["output_dir"],
                   metrics=["mean(code_dimension)", "fraction(code_dimension, above=0)"])
    assert out.text.count("\n") >= 3 and "0.1" in out.text and "0.3" in out.text


def test_unknown_metric_suggests(run_tool, ctx):
    api.run_ensemble({**BASE, "measurement_rate": 0.1}, 4, seed=1, output_dir=str(ctx.root / "c"))
    out = run_tool("analysis_compare", sources=["c"], metrics=["mena(code_dimension)"])
    assert out.is_error and "did you mean mean" in out.text
```

Add `("metrics", "phi(e, where)")` to `test_lists` in `tests/agent/test_help_tools.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_analysis_compare.py -q`
Expected: FAIL with `KeyError: 'analysis_compare'`

- [ ] **Step 3: Implement (append to `tools/analysis.py`)**

```python
import difflib
import json
from collections.abc import Callable

from sparsegf2.agent.grammar import bind, parse_calls, plain


def _scalar(res, name: str, layer: int) -> np.ndarray:
    if name not in res.values or np.ndim(res.values[name]) != 2:
        raise ValueError(f"{name!r} is not a scalar probe of this run")
    c = at_layer(res, name, layer)
    if c is None:
        raise ValueError(f"{name} was not recorded at layer {layer}")
    v = np.asarray(res.values[name], dtype=float)[:, c]
    return v[~np.isnan(v)]


def _m_mean(res, layer, probe):
    m, e = analysis.mean_and_error(_scalar(res, probe, layer))
    return float(m), float(e)


def _m_stderr(res, layer, probe):
    return float(analysis.mean_and_error(_scalar(res, probe, layer))[1]), None


def _m_fraction(res, layer, probe, above=0):
    v = _scalar(res, probe, layer) > float(above)
    m, e = analysis.mean_and_error(v.astype(float))
    return float(m), float(e)


def _m_phi(res, layer, e, where="all"):
    r = erasure_at(res, layer, rates=[float(e)], delta=0.01)
    key = "phi_code" if where == "code" else "phi_all"
    if key not in r:
        return None, None
    m, err = r[key][0]
    return float(m), float(err)


def _m_e_delta(res, layer, delta=0.01):
    return erasure_at(res, layer, rates=[0.1], delta=float(delta)).get("e_delta"), None


def _m_mu(res, layer):
    mu = erasure_at(res, layer, rates=[0.1], delta=0.01).get("mu_D")
    return (float(mu[0]), float(mu[1])) if mu is not None else (None, None)


def _m_q_plus(res, layer):
    return erasure_at(res, layer, rates=[0.1], delta=0.01)["q_plus"], None


def _m_tau_median(res, layer):
    return tau_median_all(km_curve(res)), None


def _m_censored(res, layer):
    tau = res.purification_layer
    if tau is None:
        raise ValueError("this run has no reference, so no purification times")
    return float(np.mean(np.asarray(tau) < 0)), None


def _m_survival(res, layer, at):
    km = km_curve(res)
    i = np.searchsorted(km["times"], int(at), side="right") - 1
    return (float(km["survival"][i]) if i >= 0 else 1.0), None


METRICS: dict[str, tuple[tuple[str, ...], str, Callable]] = {
    "mean": (("probe",), "mean of a scalar probe at the depth (+- stderr)", _m_mean),
    "stderr": (("probe",), "standard error of a scalar probe's mean", _m_stderr),
    "fraction": (("probe", "above"), "fraction of trajectories with probe > above", _m_fraction),
    "phi": (("e", "where"), "erasure failure probability Phi(e); where=all or code (k>0)", _m_phi),
    "e_delta": (("delta",), "tolerable erasure rate: sup{e : Phi(e) <= delta}, codes with k>0",
                _m_e_delta),
    "mu_D": ((), "mean first-failure size of codes with k>0", _m_mu),
    "q_plus": ((), "fraction of trajectories that hold a code (k>0)", _m_q_plus),
    "tau_median": ((), "median purification time over all trajectories (Kaplan-Meier)",
                   _m_tau_median),
    "censored_fraction": ((), "fraction never purified within the horizon", _m_censored),
    "survival": (("at",), "fraction not purified after layer `at` (Kaplan-Meier)", _m_survival),
}


def metric_value(res, call, layer: int):
    names, _, fn = METRICS[call.name]
    return fn(res, layer, **plain(bind(call, names, what="metrics")))


def sources_of(ctx, a) -> list[Path]:
    if ("sources" in a) == ("source" in a):
        raise ValueError("give sources (a list of runs) or source (a sweep folder), not both")
    if "source" in a:
        root = ctx.path(a["source"], what="source", must_exist=True)
        meta = root / "sweep.json"
        if not meta.exists():
            return [root]
        points = json.loads(meta.read_text(encoding="utf-8"))["points"]
        return [root / p["dir"] for p in points if (root / p["dir"] / "result.npz").exists()]
    return [ctx.path(s, what="sources", must_exist=True) for s in a["sources"]]


def _flat(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict):
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k == "probes":
                continue
            out.update(_flat(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): json.dumps(d, sort_keys=True)}


def differing(specs: list[dict]) -> list[str]:
    flats = [_flat(s or {}) for s in specs]
    keys = sorted(set().union(*flats))
    return [k for k in keys if len({f.get(k) for f in flats}) > 1]


def _compare(ctx, a):
    calls = parse_calls(a["metrics"], what="metrics")
    for i, call in enumerate(calls):
        if call.name not in METRICS:
            close = difflib.get_close_matches(call.name, list(METRICS), n=2)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise ValueError(f"metrics[{i}]: unknown metric {call.name!r}{hint} "
                             "(help_list(what=metrics))")
    paths = sources_of(ctx, a)
    if not paths:
        raise ValueError("no finished runs to compare")
    loaded = [load_result(ctx, str(p)) for p in paths]
    specs = [meta.get("spec") if meta else None for _, meta, _ in loaded]
    diff = differing(specs)
    rows = []
    for (res, meta, path), spec in zip(loaded, specs, strict=True):
        layer = int(a["depth"]) if a.get("depth") is not None else int(res.checkpoint_layers[-1])
        flat = _flat(spec or {})
        row = {"run": ctx.show(path), "params": {k: json.loads(flat[k]) if k in flat else None
                                                  for k in diff}, "layer": layer, "values": []}
        for call in calls:
            try:
                row["values"].append(metric_value(res, call, layer))
            except ValueError as exc:
                row["values"].append((None, None))
                row.setdefault("errors", []).append(f"{call.text}: {exc}")
        rows.append(row)
    return {"metrics": [c.text for c in calls], "differing": diff, "rows": rows}


def _render_compare(d, ctx):
    cols = (d["differing"] or ["run"]) + ["layer"] + d["metrics"]
    rows = []
    for r in d["rows"]:
        first = [r["params"][k] for k in d["differing"]] if d["differing"] else [r["run"]]
        rows.append([*[fmt(v) for v in first], r["layer"], *[pm(m, e) for m, e in r["values"]]])
    errors = [e for r in d["rows"] for e in r.get("errors", [])][:3]
    return lines(f"analysis_compare: {len(d['rows'])} runs", table(cols, rows),
                 *[f"note: {e}" for e in errors])


register(ToolDef(
    name="analysis_compare", stack="analysis", summary="Compare runs or a sweep: a metrics table.",
    description=(
        "One row per run (a list of runs, or the points of an ensemble_sweep folder), with the "
        "parameters that differ between runs as columns, then metrics as call strings: "
        "mean(probe), stderr(probe), fraction(probe, above=0), phi(e=0.05, where=code), "
        "e_delta(0.01), mu_D, q_plus, tau_median, censored_fraction, survival(at=100) "
        "(help_list(what=metrics)). Example: {\"source\": \"runs/ensemble_sweep-1a2b\", "
        "\"metrics\": [\"mean(code_dimension)\", \"phi(e=0.05)\"]}"
    ),
    params=(Param("sources", "array", "Runs to compare.", items="string"),
            Param("source", "string", "A sweep folder (or one run)."),
            Param("metrics", "array", "Metric calls.", required=True, items="string"),
            Param("depth", "integer", "Layer to evaluate at (default each run's last).",
                  minimum=0)),
    handler=_compare, render=_render_compare, long_running=True,
))
```

Hoist the new imports (`difflib`, `json`, `Callable`, `bind`, `parse_calls`, `plain`) to the top of the module. `analysis_compare` goes in `EXAMPLE_EXEMPT`, because it needs saved runs.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_analysis_compare.py tests/agent/test_help_tools.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent tests/agent
git commit -m "agent: analysis_compare with metric calls over runs and sweeps"
```

---

### Task 26: Result tools

**Files:**
- Create: `src/sparsegf2/agent/tools/result.py`
- Modify: `src/sparsegf2/agent/tools/__init__.py`
- Test: `tests/agent/test_result_tools.py`

**Interfaces:**
- Consumes: `load_result` (Task 23), `EnsembleResult.concatenate/save`, `api._write_run_metadata`, `api._atomic_save_result`.
- Produces: the tools `result_list`, `result_get` and `result_merge`, and `classify(path) -> str | None` (`ensemble`, `sweep`, `trajectory`, `state`, `code`, `graphs`, or `None`).

- [ ] **Step 1: Write the failing tests**

```python
"""result_list, result_get, result_merge."""

import numpy as np

from sparsegf2 import api, mipt

SPEC = {"num_system": 8, "graph": {"family": "cycle"}, "layers": 6, "measurement_rate": 0.2,
        "reference": "purification", "probes": [{"kind": "code_dimension"}]}


def test_list_classifies_artifacts(run_tool, ctx):
    api.run_ensemble(SPEC, 10, seed=1, output_dir=str(ctx.root / "runs" / "e"))
    run_tool("traj_run", num_system=8, layers=4, measurement_rate=0.1, reference="purification",
             seed=1, output_dir="runs/t")
    run_tool("graph_build", num_nodes=8, graph="cycle", output_file="g.graphs.json")
    out = run_tool("result_list", recursive=True)
    t = out.text
    assert "runs/e  ensemble" in t and "runs/t  trajectory" in t and "g.graphs.json  graphs" in t


def test_get_slices_and_bounds(run_tool, ctx):
    api.run_ensemble(SPEC, 10, seed=1, output_dir=str(ctx.root / "e"))
    res = mipt.EnsembleResult.load(ctx.root / "e" / "result.npz")
    out = run_tool("result_get", source="e", name="code_dimension", trajectory_range="0:3")
    assert str(res.values["code_dimension"][:3].astype(int).tolist()[0][0]) in out.text
    assert "shape (3, 1)" in out.text
    cut = run_tool("result_get", source="e", name="code_dimension", max_values=4)
    assert "showing 4 of 10" in cut.text


def test_merge_equals_one_run(run_tool, ctx):
    from sparsegf2.agent.params import ensemble_spec  # noqa: F401  (spec dicts below are API specs)

    mc = api._build_spec(SPEC)
    a = mc.run(5, seed=4)
    b = mc.run(5, seed=4, first_trajectory=5)
    for res, name in ((a, "a"), (b, "b")):
        (ctx.root / name).mkdir()
        res.save(ctx.root / name / "result.npz")
    out = run_tool("result_merge", sources=["a", "b"], output_dir="m")
    assert not out.is_error, out.text
    merged = mipt.EnsembleResult.load(ctx.root / "m" / "result.npz")
    whole = mc.run(10, seed=4)
    np.testing.assert_array_equal(merged.values["code_dimension"], whole.values["code_dimension"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_result_tools.py -q`
Expected: FAIL with `KeyError: 'result_list'`

- [ ] **Step 3: Implement `src/sparsegf2/agent/tools/result.py`**

```python
"""result_list, result_get, result_merge: the saved artifacts under the root."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2 import api, mipt
from sparsegf2.agent.params import SOURCE
from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.render import fmt, lines, table
from sparsegf2.agent.tools.analysis import indices, load_result

MAX_LISTED = 200


def classify(path: Path) -> str | None:
    if path.is_dir():
        if (path / "sweep.json").exists():
            return "sweep"
        if (path / "trajectory.json").exists():
            return "trajectory"
        if (path / "result.npz").exists() or (path / "chunks").is_dir():
            return "ensemble"
        return None
    name = path.name
    for suffix, kind in ((".state.npz", "state"), (".code.npz", "code"),
                         (".graphs.json", "graphs"), (".graphs.npz", "graphs"),
                         ("result.npz", "ensemble")):
        if name.endswith(suffix):
            return kind
    return None


def _describe(path: Path, kind: str) -> str:
    try:
        if kind == "ensemble" and (path / "spec.json").exists():
            s = json.loads((path / "spec.json").read_text(encoding="utf-8"))
            spec = s.get("spec", {})
            return (f"n={spec.get('num_system')}, layers={spec.get('layers')}, "
                    f"p={spec.get('measurement_rate')}, {s.get('trajectories')} trajectories")
        if kind == "sweep":
            s = json.loads((path / "sweep.json").read_text(encoding="utf-8"))
            return f"{len(s['points'])} points: {s['grid']}"
        if kind == "trajectory":
            s = json.loads((path / "trajectory.json").read_text(encoding="utf-8"))
            return (f"trajectory {s['trajectory']}, n={s['spec']['num_system']}, store={s['store']}, "
                    f"checkpoints {s['checkpoints']}")
    except (OSError, KeyError, json.JSONDecodeError):
        return "unreadable metadata"
    return ""


def _list(ctx, a):
    base = ctx.path(a.get("path", "."), what="path", must_exist=True)
    found = []
    candidates = base.rglob("*") if a["recursive"] else base.iterdir()
    for p in sorted(candidates):
        kind = classify(p)
        if kind is None or (kind == "ensemble" and p.name == "result.npz" and classify(p.parent)):
            continue
        if any(classify(parent) in ("ensemble", "trajectory", "sweep") for parent in p.parents
               if parent != base and parent.is_relative_to(base)):
            continue
        found.append({"path": ctx.show(p), "kind": kind, "about": _describe(p, kind)})
        if len(found) >= MAX_LISTED:
            break
    return {"base": ctx.show(base), "items": found}


def _render_list(d, ctx):
    if not d["items"]:
        return f"result_list: nothing saved under {d['base']}"
    width = max(len(i["path"]) for i in d["items"])
    rows = [f"{i['path'].ljust(width)}  {i['kind']}  {i['about']}".rstrip() for i in d["items"]]
    return lines(f"result_list: {len(rows)} item(s) under {d['base']}", *rows)


def _array(ctx, a) -> tuple[np.ndarray, str]:
    path = ctx.path(a["source"], what="source", must_exist=True)
    name = a["name"]
    if path.is_file() and not path.name.endswith("result.npz"):
        with np.load(path, allow_pickle=False) as z:
            if name not in z.files:
                raise ValueError(f"{name!r} is not in {path.name}; arrays: {z.files}")
            return np.asarray(z[name]), ctx.show(path)
    res, _, _ = load_result(ctx, str(path))
    fields = {"purification_layer": res.purification_layer, "final_layer": res.final_layer,
              "graph_index": res.graph_index, "checkpoint_layers": res.checkpoint_layers,
              "num_measurements": res.num_measurements, "trajectories": res.trajectories}
    if name in res.values:
        return np.asarray(res.values[name]), ctx.show(path)
    if name in fields and fields[name] is not None:
        return np.asarray(fields[name]), ctx.show(path)
    raise ValueError(f"{name!r} is not in this result; probes: {sorted(res.values)}; "
                     f"fields: {sorted(k for k, v in fields.items() if v is not None)}")


def _get(ctx, a):
    arr, where = _array(ctx, a)
    if a.get("trajectory_range") is not None and arr.ndim >= 1:
        arr = arr[indices(a["trajectory_range"], "trajectory_range")]
    if a.get("evaluation_range") is not None and arr.ndim >= 2:
        arr = arr[:, indices(a["evaluation_range"], "evaluation_range")]
    flat = arr.ravel()
    shown = flat[: a["max_values"]]
    return {"source": where, "name": a["name"], "shape": list(arr.shape), "dtype": str(arr.dtype),
            "values": shown.tolist(), "total": int(flat.size)}


def _render_get(d, ctx):
    head = (f"result_get: {d['name']} from {d['source']}, shape ({', '.join(map(str, d['shape']))}"
            f"{',' if len(d['shape']) == 1 else ''}), {d['dtype']}")
    note = (f"showing {len(d['values'])} of {d['total']} values (row-major); raise max_values or "
            "slice with trajectory_range/evaluation_range" if len(d["values"]) < d["total"] else "")
    return lines(head, note, " ".join(fmt(v) for v in d["values"]))


def _merge(ctx, a):
    parts = [load_result(ctx, s)[0] for s in a["sources"]]
    merged = mipt.EnsembleResult.concatenate(parts)
    out = ctx.path(a["output_dir"], what="output_dir")
    if (out / "result.npz").exists():
        raise ValueError(f"{ctx.show(out)} already holds a result; choose a new output_dir")
    out.mkdir(parents=True, exist_ok=True)
    api._atomic_save_result(merged, out / "result.npz")
    first_spec = Path(ctx.path(a["sources"][0], what="sources")) / "spec.json"
    if first_spec.exists():
        (out / "spec.json").write_text(first_spec.read_text(encoding="utf-8"), encoding="utf-8")
    return {"output_dir": ctx.show(out), "trajectories": merged.num_trajectories,
            "sources": len(parts)}


register(ToolDef(
    name="result_list", stack="result", summary="Saved runs, records, states, codes, graphs.",
    description=(
        "List saved artifacts under a folder: ensemble runs, sweeps, trajectory records, state "
        "files, code files and graph collections, each with a one-line description. "
        "Example: {\"path\": \"runs\", \"recursive\": true}"
    ),
    params=(Param("path", "string", "Folder under the root.", default="."),
            Param("recursive", "boolean", "Look in subfolders too.", default=False)),
    handler=_list, render=_render_list, examples=({"recursive": True},),
))

register(ToolDef(
    name="result_get", stack="result", summary="Raw values of a saved array (bounded).",
    description=(
        "Raw values of one array of a saved result: a probe (by name), purification_layer, "
        "final_layer, graph_index, checkpoint_layers, num_measurements, trajectories, or any "
        "array of a .npz file; sliced by trajectory_range/evaluation_range and capped at "
        "max_values. Example: {\"source\": \"runs/ensemble_run-3f2a\", \"name\": "
        "\"code_dimension\", \"trajectory_range\": \"0:5\"}"
    ),
    params=(SOURCE, Param("name", "string", "Array name.", required=True),
            Param("trajectory_range", "string", "Rows, e.g. 0:5."),
            Param("evaluation_range", "string", "Checkpoint indices, e.g. 0:3 or [0,2]."),
            Param("max_values", "integer", "At most this many values.", default=200, minimum=1,
                  maximum=5000)),
    handler=_get, render=_render_get,
))

register(ToolDef(
    name="result_merge", stack="result", summary="Merge chunks of one ensemble into one result.",
    description=(
        "Merge saved ensemble results of the same spec and seed (e.g. chunks run on a cluster "
        "with first_trajectory) into one result, identical to a single run. "
        "Example: {\"sources\": [\"runs/chunk0\", \"runs/chunk1\"], \"output_dir\": \"runs/all\"}"
    ),
    params=(Param("sources", "array", "The parts, in any order.", required=True, items="string"),
            Param("output_dir", "string", "Where the merged result goes.", required=True)),
    handler=_merge, render=lambda d, ctx: (f"result_merge: {d['sources']} parts, "
                                           f"{d['trajectories']} trajectories -> {d['output_dir']}"),
    read_only=False,
))
```

Add `result` to `tools/__init__.py`. Put `result_get` and `result_merge` in `EXAMPLE_EXEMPT`, because they need saved runs.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_result_tools.py -q`
Expected: PASS. `EnsembleResult.concatenate` orders the parts by trajectory and refuses mismatched specs or seeds; its error message passes through unchanged.

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent tests/agent/test_result_tools.py
git commit -m "agent: result_list, result_get, result_merge"
```

---

### Task 27: `plot_make`

**Files:**
- Create: `src/sparsegf2/agent/tools/plot.py`
- Modify: `src/sparsegf2/agent/tools/__init__.py`, `pyproject.toml` (the `plot` extra; add `matplotlib` to `dev` through it)
- Test: `tests/agent/test_plot_tool.py`

**Interfaces:**
- Consumes: `load_result`, `select`, `sources_of`, `differing`, `metric_value`, `km_curve`, `analysis.mean_and_error`, `grammar.parse_call`, matplotlib (`Agg` backend).
- Produces: the tool `plot_make(kind, sources|source, quantity, depth, where, output_file, title, logx, logy)` with `requires=("matplotlib",)`.

- [ ] **Step 1: Write the failing tests**

```python
"""plot_make writes figures (skipped without matplotlib)."""

import pytest

pytest.importorskip("matplotlib")

from sparsegf2 import api  # noqa: E402

SPEC = {"num_system": 8, "graph": {"family": "cycle"}, "layers": 16, "measurement_rate": 0.2,
        "reference": "single", "checkpoints": "every_layer", "stop_when_purified": True,
        "probes": [{"kind": "code_dimension"}, {"kind": "entropy_profile", "checkpoints": "final"}]}


@pytest.fixture
def runs(ctx):
    for p, name in ((0.1, "a"), (0.3, "b")):
        api.run_ensemble({**SPEC, "measurement_rate": p}, 40, seed=1, output_dir=str(ctx.root / name))


@pytest.mark.parametrize(
    "kind, quantity",
    [("vs_depth", "code_dimension"), ("survival", None), ("histogram", "code_dimension"),
     ("vs_size", "entropy_profile"), ("vs_parameter", "censored_fraction")],
)
def test_each_kind_writes_a_png(run_tool, ctx, runs, kind, quantity):
    args = {"kind": kind, "sources": ["a", "b"], "output_file": f"{kind}.png"}
    if quantity:
        args["quantity"] = quantity
    out = run_tool("plot_make", **args)
    assert not out.is_error, out.text
    assert (ctx.root / f"{kind}.png").stat().st_size > 1000
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/agent/test_plot_tool.py -q`
Expected: FAIL with `KeyError: 'plot_make'`

- [ ] **Step 3: Implement `src/sparsegf2/agent/tools/plot.py`**

```python
"""plot_make: figures of saved runs (matplotlib, optional extra 'plot')."""

from __future__ import annotations

from typing import Any

import numpy as np

from sparsegf2 import analysis
from sparsegf2.agent.grammar import parse_call
from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.tools.analysis import (
    METRICS, differing, km_curve, load_result, metric_value, select, sources_of,
)


def _label(ctx, path, params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items()) or ctx.show(path)


def _plot(ctx, a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kind = a["kind"]
    paths = sources_of(ctx, a)
    loaded = [load_result(ctx, str(p)) for p in paths]
    specs = [meta.get("spec") if meta else None for _, meta, _ in loaded]
    diff = differing(specs)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    quantity = a.get("quantity")
    for (res, meta, path), spec in zip(loaded, specs, strict=True):
        flat_params = {k: (spec or {}).get(k.split(".")[0]) for k in diff}
        label = _label(ctx, path, flat_params)
        if kind == "vs_depth":
            if quantity not in res.values or np.ndim(res.values[quantity]) != 2:
                raise ValueError(f"vs_depth needs a scalar probe; {quantity!r} is not one here")
            layers = np.asarray(res.layers(quantity))
            v = np.asarray(res.values[quantity], dtype=float)
            means, errs = [], []
            for c, layer in enumerate(layers):
                mask = select(res, layer=int(layer), where=a.get("where"))
                col = v[mask, c]
                m, e = analysis.mean_and_error(col[~np.isnan(col)]) if col.size else (np.nan, np.nan)
                means.append(m)
                errs.append(e)
            ax.errorbar(layers, means, yerr=errs, label=label, capsize=2)
            ax.set_xlabel("layer")
            ax.set_ylabel(quantity)
        elif kind == "vs_size":
            if quantity not in res.values or np.ndim(res.values[quantity]) != 3:
                raise ValueError(f"vs_size needs a vector probe; {quantity!r} is not one here")
            layers = np.asarray(res.layers(quantity))
            layer = int(a["depth"]) if a.get("depth") is not None else int(layers[-1])
            c = int(np.flatnonzero(layers == layer)[0])
            v = np.asarray(res.values[quantity], dtype=float)[:, c]
            v = v[~np.isnan(v).any(axis=-1)]
            ax.plot(np.arange(v.shape[-1]), v.mean(axis=0), label=f"{label} (layer {layer})")
            ax.set_xlabel("m")
            ax.set_ylabel(quantity)
        elif kind == "survival":
            km = km_curve(res)
            ax.step(km["times"], km["survival"], where="post", label=label)
            ax.set_xlabel("layer")
            ax.set_ylabel("fraction unpurified")
        elif kind == "histogram":
            layers = np.asarray(res.layers(quantity))
            layer = int(a["depth"]) if a.get("depth") is not None else int(layers[-1])
            c = int(np.flatnonzero(layers == layer)[0])
            v = np.asarray(res.values[quantity], dtype=float)[:, c]
            ax.hist(v[~np.isnan(v)], bins=30, alpha=0.6, label=label)
            ax.set_xlabel(f"{quantity} at layer {layer}")
            ax.set_ylabel("trajectories")
    if kind == "vs_parameter":
        if len(diff) != 1:
            raise ValueError(f"vs_parameter needs runs that differ in exactly one parameter; they "
                             f"differ in {diff or 'none'}")
        call = parse_call(quantity or "", what="quantity")
        if call.name not in METRICS:
            raise ValueError(f"quantity must be a metric call (help_list(what=metrics)), got {quantity!r}")
        xs, ys, es = [], [], []
        for (res, meta, path), spec in zip(loaded, specs, strict=True):
            layer = int(a["depth"]) if a.get("depth") is not None else int(res.checkpoint_layers[-1])
            m, e = metric_value(res, call, layer)
            key = diff[0]
            node: Any = spec or {}
            for part in key.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            xs.append(node)
            ys.append(np.nan if m is None else m)
            es.append(0.0 if e is None else e)
        order = np.argsort(xs)
        ax.errorbar(np.asarray(xs)[order], np.asarray(ys)[order], yerr=np.asarray(es)[order],
                    marker="o", capsize=2)
        ax.set_xlabel(diff[0])
        ax.set_ylabel(quantity)
    if a.get("logx"):
        ax.set_xscale("log")
    if a.get("logy"):
        ax.set_yscale("log")
    if a.get("title"):
        ax.set_title(a["title"])
    if kind != "vs_parameter" and len(loaded) > 1:
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = ctx.path(a["output_file"], what="output_file")
    if out.suffix.lower() != ".png":
        raise ValueError("output_file must end in .png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return {"output_file": ctx.show(out), "kind": kind, "runs": len(loaded)}


register(ToolDef(
    name="plot_make", stack="plot", summary="A PNG figure of saved runs.",
    description=(
        "Draw saved runs to a PNG: vs_depth (a scalar probe's mean +- stderr vs layer), "
        "vs_size (a vector probe's curve over m at a depth), survival (Kaplan-Meier), histogram "
        "(a scalar probe at a depth), vs_parameter (a metric call vs the one parameter the runs "
        "differ in, e.g. a sweep). Example: {\"kind\": \"vs_depth\", \"source\": "
        "\"runs/ensemble_sweep-1a2b\", \"quantity\": \"code_dimension\", \"output_file\": "
        "\"figs/k.png\"}"
    ),
    params=(
        Param("kind", "string", "Figure type.", required=True,
              enum=("vs_depth", "vs_size", "survival", "histogram", "vs_parameter")),
        Param("sources", "array", "Runs to draw.", items="string"),
        Param("source", "string", "A sweep folder (or one run)."),
        Param("quantity", "string", "A probe name, or a metric call for vs_parameter."),
        Param("depth", "integer", "Layer for vs_size / histogram / vs_parameter.", minimum=0),
        Param("where", "string", "Trajectory selection for vs_depth (as in analysis_probe)."),
        Param("output_file", "string", "Destination .png under the root.", required=True),
        Param("title", "string", "Figure title."),
        Param("logx", "boolean", "Log x axis.", default=False),
        Param("logy", "boolean", "Log y axis.", default=False),
    ),
    handler=_plot,
    render=lambda d, ctx: f"plot_make: {d['kind']} of {d['runs']} run(s) -> {d['output_file']}",
    read_only=False, long_running=True, requires=("matplotlib",),
))
```

In `pyproject.toml`: add `plot = ["matplotlib>=3.8"]` under `[project.optional-dependencies]`, and change `dev` to `["sparsegf2[test,lint,mcp,plot]"]`. Add `plot` to `tools/__init__.py`. `plot_make` goes in `EXAMPLE_EXEMPT`, because it needs saved runs.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_plot_tool.py -q`
Expected: PASS (or SKIP without matplotlib)

- [ ] **Step 5: Commit, and run the whole suite (end of Phase 3)**

```bash
ruff check src tests && ruff format src tests
python -m pytest -q
git add src/sparsegf2/agent pyproject.toml tests/agent/test_plot_tool.py
git commit -m "agent: plot_make (vs depth, vs size, survival, histogram, vs parameter)"
```

---
## Phase 4: Coverage, documentation, evaluation and switch-over

### Task 28: The coverage map and registry-wide tests

**Files:**
- Create: `src/sparsegf2/agent/coverage.py`
- Test: `tests/agent/test_coverage.py`, `tests/agent/test_registry_all.py`

**Interfaces:**
- Consumes: `REGISTRY`, `stacks.resolve/presets/STACKS/CORE`, `vocab.*`, `registry.check_flat`, `dispatch.call`, and the `session` fixture.
- Produces:
  - `coverage.COVERAGE: dict[str, dict[str, str]]`, keyed by library object (`"Simulator"`, `"StabilizerCode"`, `"analysis"`, `"graphs"`, `"EnsembleResult"`), mapping each public name to how a tool reaches it, or to `"NOT_EXPOSED: <reason>"`.
  - `coverage.public_names(obj) -> set[str]`.
  - `EXAMPLE_EXEMPT: frozenset[str]`, the tools whose examples need live ids or saved runs (in `coverage.py`).

- [ ] **Step 1: Write `src/sparsegf2/agent/coverage.py`**

```python
"""How every public capability of the library is reached through the tools (spec section 9).

tests/agent/test_coverage.py fails when the library gains a public name that is
missing here, so the agent interface keeps up with the simulator. Map a new name
to the tool that reaches it, or to "NOT_EXPOSED: <reason>".
"""

from __future__ import annotations

import inspect

NOT_EXPOSED = "NOT_EXPOSED"

EXAMPLE_EXEMPT = frozenset({
    "job_status", "job_cancel", "state_apply", "state_query", "state_save", "state_close",
    "traj_probe", "analysis_summary", "analysis_probe", "analysis_erasure",
    "analysis_purification", "analysis_compare", "result_get", "result_merge", "plot_make",
})

_GATE = "circuit text: state_apply, circuit_run"
COVERAGE: dict[str, dict[str, str]] = {
    "Simulator": {
        "from_tableau": "NOT_EXPOSED: Clifford tableaux are prepared with circuit text",
        "num_qubits": "state_list", "phases": "state_list", "seed": "state_new",
        "copy": "NOT_EXPOSED: save and reopen a state (state_save, state_new)",
        "do": _GATE, "clifford_1q": _GATE, "clifford_2q": _GATE, "apply": _GATE,
        "measure": _GATE, "measure_x": _GATE, "measure_y": _GATE, "measure_many": _GATE,
        "measure_pauli": "circuit text MPP: state_apply", "reset": _GATE, "reset_x": _GATE,
        "reset_y": _GATE, "peek_z": "state_query expectation", "peek_x": "state_query expectation",
        "peek_y": "state_query expectation", "peek_pauli": "state_query expectation",
        "is_deterministic": "state_query is_deterministic",
        "stabilizer_arrays": "state_query stabilizers", "stabilizers": "state_query stabilizers",
        "destabilizers": "state_query destabilizers", "run": "state_apply, circuit_run",
        "entropy": "state_query entropy", "save": "state_save", "load": "state_new(source)",
        "*gates": _GATE,
    },
    "StabilizerCode": {
        "from_packed": "NOT_EXPOSED: internal constructor (codes load from files)",
        "from_simulator": "code_analyze(state_id, reference)",
        "num_qubits": "code_analyze summary", "num_stabilizers": "code_analyze summary",
        "num_logical": "code_analyze summary", "stabilizer_matrix": "code_analyze stabilizers, save_to",
        "lost_information": "code_analyze lost_information", "is_correctable": "code_analyze is_correctable",
        "first_failure_sizes": "code_analyze first_failure",
        "uncorrectable_counts": "code_analyze exhaustive_counts",
        "contiguous_distance": "code_analyze contiguous_distance",
    },
    "analysis": {
        "mean_and_error": "every +- in the analysis tools", "kaplan_meier": "analysis_purification",
        "bernoulli_erasure_curve": "NOT_EXPOSED: Phi(e) comes from the first-failure estimators",
        "failure_by_size": "code_analyze failure_by_size, analysis_erasure",
        "exact_failure_by_size": "code_analyze exhaustive_counts",
        "hybrid_failure_by_size": "analysis_erasure P_C(m)",
        "erasure_failure_probability": "analysis_erasure, analysis_compare phi",
        "erasure_tolerance": "analysis_erasure, analysis_compare e_delta",
        "mean_first_failure": "analysis_erasure, code_analyze mean_first_failure",
        "distance_bounds": "analysis_erasure, code_analyze distance_bounds",
        "junction_dip": "analysis_erasure note",
    },
    "graphs": {
        "from_edges": "graph files (graph=file)", "from_networkx": "NOT_EXPOSED: save networkx "
        "graphs to a JSON graph file and use graph=file", "degrees": "graph_build, graph_stats",
        "edge_coloring": "schedule=matching", "stats": "graph_stats",
        "shortcut_fraction": "graph_stats", "save": "graph_build", "load": "graph=file",
        "MAX_STATS_NODES": "graph_stats limit", "*families": "graph parameter (every family)",
        "*weights": "edge_weights parameter",
    },
    "EnsembleResult": {
        "num_trajectories": "analysis_summary", "seed_sequence": "NOT_EXPOSED: seeds are reported",
        "graph_index": "graphs selections, result_get", "layers": "analysis_probe",
        "code": "code_analyze(source)", "censored": "analysis_purification",
        "trajectories_per_second": "analysis_summary", "mean": "analysis_probe",
        "survival": "analysis_purification", "save": "every run's output_dir",
        "load": "every analysis tool", "concatenate": "result_merge",
        "elapsed_seconds": "analysis_summary", "version": "analysis_summary (provenance)",
        "seed_pool_size": "NOT_EXPOSED: internal to the seed contract",
        "seed_spawn_key": "NOT_EXPOSED: internal to the seed contract",
    },
}


def public_names(obj) -> set[str]:
    """A module's ``__all__``, or a class's public attributes (methods, properties, fields
    with defaults)."""
    if inspect.isclass(obj):
        return {n for n in dir(obj) if not n.startswith("_")}
    return set(getattr(obj, "__all__", ()))


def is_gate_method(name: str) -> bool:
    """Whether a Simulator method is a gate (a Stim gate name or alias, lowercased)."""
    from sparsegf2.gates import gate

    try:
        gate(name.upper())
    except KeyError:
        return False
    return True
```

- [ ] **Step 2: Write `tests/agent/test_coverage.py`**

```python
"""Every public capability of the library is reachable through some tool (spec section 9)."""

import dataclasses

import pytest

import sparsegf2 as sg
from sparsegf2 import analysis, api, graphs, mipt
from sparsegf2.agent import vocab
from sparsegf2.agent.coverage import COVERAGE, is_gate_method, public_names
from sparsegf2.agent.params import FIELD_MAP, GRAPH_PARAMS

PROBE_EXAMPLES = {
    "code_dimension": "code_dimension", "code_snapshot": "code_snapshot@final",
    "contiguous_distance": "contiguous_distance", "entropy": "entropy(0:2)",
    "entropy_profile": "entropy_profile", "erasure_first_failure": "erasure_first_failure(8)",
    "erasure_lost_profile": "erasure_lost_profile(2)", "half_cut_entropy": "half_cut_entropy",
    "lost_information": "lost_information(0:2)", "mutual_information": "mutual_information(0:2, 4:6)",
    "tripartite_information": "tripartite_information",
    "uncorrectable_region_counts": "uncorrectable_region_counts(1)",
}
OBSERVABLE_EXAMPLES = {
    "entropy": "entropy(0:2)", "mutual_information": "mutual_information(0:1, 2:3)",
    "tripartite_information": "tripartite_information(0:1, 1:2, 2:3)",
    "entropy_profile": "entropy_profile", "code_dimension": "code_dimension(reference=4:8)",
    "lost_logical_information": "lost_logical_information(erased=[0], reference=4:8)",
    "erasure_is_correctable": "erasure_is_correctable(erased=[0], reference=4:8)",
    "recovery_probability": "recovery_probability(erased=[0], reference=4:8)",
    "random_erasure_statistics": "random_erasure_statistics(system=0:4, reference=4:8, sizes=[1, 2], samples=4)",
    "erasure_profile": "erasure_profile(system=0:4, reference=4:8, samples=2)",
    "contiguous_code_distance": "contiguous_code_distance(system=0:4, reference=4:8)",
    "stabilizer_weights": "stabilizer_weights",
    "stabilizer_length_distribution": "stabilizer_length_distribution",
    "stabilizers": "stabilizers", "expectation": "expectation(Z0*Z4)",
    "is_deterministic": "is_deterministic(Z0*Z4)",
}
GRAPH_EXAMPLES = {
    "cycle": {}, "path": {}, "complete": {}, "lattice_2d": {"graph_rows": 2, "graph_cols": 4},
    "watts_strogatz": {"graph_k": 4}, "newman_watts": {"graph_p": 0.2},
    "erdos_renyi": {"graph_p": 0.5}, "random_regular": {"graph_d": 3},
}


def test_every_probe_kind_runs_through_ensemble_run(run_tool):
    assert set(PROBE_EXAMPLES) == set(vocab.probe_signatures()), "map the new probe here"
    out = run_tool("ensemble_run", num_system=8, layers=4, measurement_rate=0.2,
                   reference="purification", trajectories=4, seed=1,
                   probes=list(PROBE_EXAMPLES.values()))
    assert not out.is_error, out.text


def test_every_observable_runs_through_state_query(run_tool):
    assert set(OBSERVABLE_EXAMPLES) == set(vocab.observable_signatures()), "map the new observable"
    run_tool("state_new", num_qubits=8)
    run_tool("state_apply", state_id="s1",
             circuit="H 0 1 2 3\nCX 0 4 1 5 2 6 3 7\nCX 0 1 1 2 2 3")
    out = run_tool("state_query", state_id="s1", observables=list(OBSERVABLE_EXAMPLES.values()))
    assert not out.is_error, out.text


def test_every_graph_family_builds(run_tool):
    assert set(GRAPH_EXAMPLES) == set(vocab.graph_families()), "map the new graph family"
    assert set(GRAPH_PARAMS[0].enum) == {*vocab.graph_families(), "file"}
    for family, extra in GRAPH_EXAMPLES.items():
        out = run_tool("graph_build", num_nodes=8, graph=family, output_file=f"{family}.graphs.json",
                       **extra)
        assert not out.is_error, (family, out.text)


def test_every_protocol_field_is_a_parameter():
    fields = {f.name for f in dataclasses.fields(mipt.MonitoredCircuit) if not f.name.startswith("_")}
    assert set(FIELD_MAP) == fields


def test_every_gate_is_accepted_in_circuit_text(run_tool):
    for name in vocab.gate_names()["unitary"]:
        info = sg.gate(name)
        targets = "0" if info.num_qubits == 1 else "0 1"
        out = run_tool("circuit_inspect", circuit=f"{name} {targets}")
        assert "circuit_inspect: valid" in out.text, name


@pytest.mark.parametrize(
    "key, obj",
    [("Simulator", sg.Simulator), ("StabilizerCode", sg.StabilizerCode),
     ("analysis", analysis), ("graphs", graphs), ("EnsembleResult", mipt.EnsembleResult)],
)
def test_coverage_map_is_complete(key, obj):
    mapped = set(COVERAGE[key])
    names = public_names(obj)
    if key == "Simulator":  # every Stim gate (and alias) is a method; circuit text covers them
        names = {n for n in names if not is_gate_method(n)} | {"*gates"}
    if key == "graphs":
        families = set(api._graph_families()) | {f"{w}_weights" for w in api._weight_families()}
        names = (names - families) | {"*families", "*weights"}
    missing = sorted(names - mapped)
    assert not missing, f"map these public names in agent/coverage.py: {missing}"
```

- [ ] **Step 3: Write `tests/agent/test_registry_all.py`**

```python
"""Properties every registered tool must have: flat, small, documented, tested by example."""

import pytest

from sparsegf2.agent import dispatch, stacks
from sparsegf2.agent.coverage import EXAMPLE_EXEMPT
from sparsegf2.agent.registry import REGISTRY, Bound, check_flat

TOOLS = sorted(REGISTRY)


@pytest.mark.parametrize("name", TOOLS)
def test_tool_contract(name):
    tool = REGISTRY[name]
    b = Bound(tool)
    assert tool.stack in stacks.STACKS
    assert check_flat(b.input_schema()) == []
    assert b.size() <= 5000, f"{name} is {b.size()} characters"
    assert "Example" in tool.description and len(tool.summary) <= 80


@pytest.mark.parametrize("name", [n for n in TOOLS if n not in EXAMPLE_EXEMPT])
def test_examples_run(name, ctx):
    tool = REGISTRY[name]
    if tool.available():
        pytest.skip(tool.available())
    assert tool.examples, f"{name} needs an example (or list it in EXAMPLE_EXEMPT)"
    for example in tool.examples:
        out = dispatch.call(ctx, Bound(tool), {**example, **({"background": "never"} if tool.long_running else {})})
        assert not out.is_error, out.text
        assert out.text.startswith(name), out.text[:80]


def test_the_full_server_builds_quickly(tmp_path):
    import time

    from sparsegf2.agent import server

    t0 = time.perf_counter()
    server.build("all", root=tmp_path, quiet=True)
    assert time.perf_counter() - t0 < 5.0  # Clank's check allows 30 s; no kernel compiles here


def test_core_bundle_size(tmp_path):
    r = stacks.resolve("core", root=tmp_path)
    assert [b.name for b in r.bounds] == list(stacks.CORE)
    assert sum(b.size() for b in r.bounds) <= 18000


def test_every_stack_has_tools(tmp_path):
    for s in stacks.STACKS:
        assert stacks.resolve(s, root=tmp_path).bounds or s == "plot"


@pytest.mark.parametrize("preset", ["erasure", "purification", "entanglement", "circuits"])
def test_presets_resolve_with_their_defaults(tmp_path, preset):
    r = stacks.resolve(preset, root=tmp_path)
    assert r.bounds
    if preset == "erasure":
        run = next(b for b in r.bounds if b.name == "ensemble_run")
        assert run.input_schema()["properties"]["reference"]["default"] == "purification"


@pytest.mark.parametrize("stack", [*stacks.STACKS, "core", "all", "erasure", "circuits"])
def test_each_stack_serves_over_mcp(session, tmp_path, stack):
    expected = [b.name for b in stacks.resolve(stack, root=tmp_path).bounds]
    tools = session(stack, lambda c: c.list_tools()).tools
    assert [t.name for t in tools] == expected


def test_shared_groups_are_identical_wherever_they_appear():
    from sparsegf2.agent.params import (
        CHECKPOINTS, GRAPH_PARAMS, OUTPUT_DIR, PROBES, PROTOCOL_PARAMS, SEED,
    )

    shared = {p.name: p for p in (*GRAPH_PARAMS, *PROTOCOL_PARAMS, PROBES, CHECKPOINTS, SEED,
                                  OUTPUT_DIR)}
    # documented exceptions: (tool, parameter) -> why it differs
    allowed = {
        ("ensemble_sweep", "num_system"): "optional: it may be in grid",
        ("ensemble_sweep", "layers"): "optional: it may be in grid",
        ("ensemble_sweep", "measurement_rate"): "optional: it may be in grid",
        ("traj_probe", "checkpoints"): "no default: keeps the recorded checkpoints",
        ("traj_probe", "output_dir"): "defaults to a folder per request",
        ("circuit_crosscheck", "seed"): "Stim's seed, default 0",
    }
    for tool in REGISTRY.values():
        for p in tool.params:
            if p.name in shared and (tool.name, p.name) not in allowed:
                assert p == shared[p.name], f"{tool.name}.{p.name} differs from the shared group"


def test_no_parameter_name_has_two_types():
    kinds: dict[str, set[tuple[str, str | None]]] = {}
    for tool in REGISTRY.values():
        for p in tool.params:
            kinds.setdefault(p.name, set()).add((p.type, p.items))
    clashes = {name: k for name, k in kinds.items() if len(k) > 1}
    assert not clashes, f"rename one meaning of each: {clashes}"
```

- [ ] **Step 4: Run them, and fix what they find**

Run: `python -m pytest tests/agent/test_coverage.py tests/agent/test_registry_all.py -q`
Expected: PASS after fixing what they report. The likely fixes are descriptions above the size budget (shorten the shared parameter descriptions), a missing `Example` in a description, and new public names to map. Each fix belongs in the module the failure names.

- [ ] **Step 5: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent/coverage.py tests/agent
git commit -m "agent: coverage map of the library and registry-wide contract tests"
```

---

### Task 29: Documentation (generated tool reference and guides)

**Files:**
- Create: `src/sparsegf2/agent/docs.py`, `docs/MCP_TOOLS.md` (generated)
- Modify: `docs/MCP_GETTING_STARTED.md`, `AGENTS.md`, `llms.txt`, `docs/LLM_INTERFACE.md`, `README.md` (the "For AI agents" section), `.claude/skills/sparsegf2/SKILL.md`, `docs/specs/2026-10-13-agent-tool-stacks-design.md` (status: implemented)
- Test: `tests/agent/test_docs_reference.py`

**Interfaces:**
- Produces: `agent.docs.render_reference() -> str` (markdown), and `python -m sparsegf2.agent.docs` prints it.

- [ ] **Step 1: Write the failing test**

```python
"""docs/MCP_TOOLS.md is generated from the registry and must be current."""

from pathlib import Path

from sparsegf2.agent.docs import render_reference

ROOT = Path(__file__).resolve().parents[2]


def test_reference_is_current():
    committed = (ROOT / "docs" / "MCP_TOOLS.md").read_text(encoding="utf-8")
    assert committed == render_reference(), (
        "regenerate: python -m sparsegf2.agent.docs > docs/MCP_TOOLS.md")


def test_reference_lists_every_tool():
    import sparsegf2.agent.tools  # noqa: F401
    from sparsegf2.agent.registry import REGISTRY

    text = render_reference()
    for name in REGISTRY:
        assert f"### `{name}`" in text
```

- [ ] **Step 2: Implement `src/sparsegf2/agent/docs.py`**

```python
"""Generate docs/MCP_TOOLS.md from the tool registry: python -m sparsegf2.agent.docs > docs/MCP_TOOLS.md"""

from __future__ import annotations

import sys

from sparsegf2.agent import INTERFACE_VERSION
from sparsegf2.agent.registry import MISSING, REGISTRY, Bound


def render_reference() -> str:
    import sparsegf2.agent.tools  # noqa: F401
    from sparsegf2.agent import stacks

    out = [
        "# SparseGF2 MCP tools",
        "",
        f"Generated from the tool registry (interface version {INTERFACE_VERSION}); do not edit by "
        "hand. Regenerate with `python -m sparsegf2.agent.docs > docs/MCP_TOOLS.md`. How to "
        "set up the server and choose stacks: [`MCP_GETTING_STARTED.md`](MCP_GETTING_STARTED.md).",
        "",
        "| stack | tools |",
        "|---|---|",
    ]
    for s in stacks.STACKS:
        names = [n for n, t in REGISTRY.items() if t.stack == s]
        out.append(f"| `{s}` | {', '.join(f'`{n}`' for n in names)} |")
    out += ["", f"Bundles: `all` (every stack), `core` ({', '.join(f'`{n}`' for n in stacks.CORE)}). "
            "Presets: " + ", ".join(f"`{p}`" for p in stacks.presets()) + ".", ""]
    for s in stacks.STACKS:
        out += [f"## `{s}`", ""]
        for name, tool in REGISTRY.items():
            if tool.stack != s:
                continue
            b = Bound(tool)
            out += [f"### `{name}`", "", tool.description, ""]
            flags = [x for x, on in (("long-running (background=auto|always|never)", tool.long_running),
                                     ("writes files", not tool.read_only),
                                     (f"needs {', '.join(tool.requires)}", bool(tool.requires))) if on]
            if flags:
                out += ["*" + "; ".join(flags) + "*", ""]
            out += ["| parameter | type | default | description |", "|---|---|---|---|"]
            for p in tool.params:
                kind = p.type + (f" of {p.items}" if p.items else "")
                if p.enum:
                    kind += ": " + " / ".join(p.enum)
                default = "required" if p.required else ("" if p.default is MISSING or p.default is None else f"`{p.default}`")
                out.append(f"| `{p.name}` | {kind} | {default} | {p.description} |")
            out += ["", f"Schema size: {b.size()} characters.", ""]
    return "\n".join(out).rstrip() + "\n"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(render_reference())
```

Generate the file: `python -m sparsegf2.agent.docs > docs/MCP_TOOLS.md`. On Windows, use `.venv\Scripts\python -m sparsegf2.agent.docs | Out-File -Encoding utf8NoBOM docs\MCP_TOOLS.md` in PowerShell 7, or the Bash redirection in Git Bash, so the file is UTF-8 without a BOM.

- [ ] **Step 3: Rewrite `docs/MCP_GETTING_STARTED.md`**

Keep the existing install, connect and troubleshooting structure, but change the content to match the new server:
1. **What the server is.** The 11 stacks (the table from spec 6), the call grammar in one paragraph with three examples, and results as text with `next:` lines.
2. **Install.** As today, plus the `plot` extra, plus `sparsegf2 warmup` (Task 31) to compile the kernels once.
3. **Choosing a stack.** Built-in stacks, `core` and `all`, the four presets, and `sparsegf2-mcp --list --stack NAME` to see sizes. Recommend `core` or a custom stack for local models and `all` for Claude Code.
4. **Custom stacks.** The JSON format and the `sparsegf2 stacks save` examples from spec 8.2. One worked example: `ws-erasure-128` fixing `num_system`, `graph` and `graph_k`, then `sparsegf2 stacks show ws-erasure-128` printing the smaller schema size.
5. **Claude Code.** The `.mcp.json` now passes `--stack all`. If Claude Code starts outside the virtual environment, the server fails to connect, because `sparsegf2-mcp` is not on PATH. Fix it by activating `.venv` first, or with a local-scope registration pointing at the venv executable: `claude mcp add --scope local sparsegf2 -- C:\Projects\SparseGF2\.venv\Scripts\sparsegf2-mcp.exe --stack all`.
6. **Clank.**
   - Install SparseGF2 into the workspace image, or into a venv inside the workspace. The sandbox has no network.
   - Set `NUMBA_CACHE_DIR=/workspace/.numba-cache` and run `sparsegf2 warmup` once.
   - Register the server with `clank mcp add sparsegf2 -- sparsegf2-mcp --stack core --root /workspace --quiet`.
   - Results are text, so nothing depends on `structuredContent`.
   - Long work returns a job id: tell the model to poll `job_status` with `wait_seconds=30`.
7. **A first session across stacks.** Six short exchanges, each showing the tool call and the start of its result:
   1. `help_topic(conventions)`;
   2. `state_new` / `state_apply` / `state_query` (a Bell pair);
   3. `circuit_run` with observables per tick;
   4. `ensemble_run` on a Watts-Strogatz collection with the erasure probes;
   5. `analysis_erasure` on its output;
   6. `traj_run` on one trajectory, then `code_analyze` with `failure_probability`.

   Produce the result texts by actually running the calls through `dispatch.call` (paste real output, trimmed).
8. **Adding a tool.** The checklist from spec 9.3:
   - a handler and renderer, and a `ToolDef` with an example, in `agent/tools/<stack>.py`;
   - the import in `tools/__init__.py`;
   - `coverage.py` updated if it exposes a new library name;
   - regenerate `docs/MCP_TOOLS.md`;
   - run the agent tests;
   - bump `INTERFACE_VERSION` if a tool or parameter was renamed or removed.

- [ ] **Step 4: Update the other agent documents**

- `AGENTS.md`:
  - Replace the "Five-minute tour" with the new tool names, in a tour across stacks (the same 6 steps as above, as JSON argument examples).
  - Replace "The protocol behind `run_ensemble`" with the flat parameter names, and point to `docs/MCP_TOOLS.md` for every parameter.
  - Keep "Conventions", "Pitfalls" and "Erasure analysis in Python" (still valid), changing tool names where they appear: `load_results` becomes `analysis_summary`, `start_ensemble_job` becomes `ensemble_run(background="always")`, `estimate_ensemble` becomes `ensemble_run(estimate_only=true)`.
- `llms.txt`: add `docs/MCP_TOOLS.md` and the stack list.
- `docs/LLM_INTERFACE.md`: replace the MCP section's tool list with a pointer to `MCP_TOOLS.md` and a short description of stacks and the grammar. Keep the CLI and JSON API sections (unchanged).
- `README.md` "For AI agents": the stacks in two sentences, the `--stack` flag, Clank and Claude Code in one line each, and links to `MCP_GETTING_STARTED.md` and `MCP_TOOLS.md`.
- `.claude/skills/sparsegf2/SKILL.md`: the new tool names; "call `help_topic(topic=conventions)` first"; `help_list` for exact names.
- The spec: set the status line to `implemented (revision 2)`. In 6.2, list `destabilizers` as the extra observable (`peek` is `expectation`).

- [ ] **Step 5: Run the doc tests and the guide's examples**

Run: `python -m pytest tests/agent/test_docs_reference.py tests/test_docs.py -q`
Expected: PASS. `tests/test_docs.py` executes the README and the getting-started guide; adjust any code blocks that mention old MCP tool names (they are prose, not executed Python, but check).

- [ ] **Step 6: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/sparsegf2/agent/docs.py docs AGENTS.md llms.txt README.md .claude/skills/sparsegf2/SKILL.md tests/agent/test_docs_reference.py
git commit -m "docs: generated MCP tool reference; guides for stacks, custom stacks, Clank, Claude Code"
```

---

### Task 30: The local-model evaluation script

**Files:**
- Create: `scripts/eval_agent_tools.py`, `scripts/eval_requests.json`
- Test: `tests/agent/test_eval_script.py`

**Interfaces:**
- Consumes: `server.build`, `Bound.validate`, `dispatch.call`, and an OpenAI-compatible `/v1/chat/completions` endpoint (stdlib `urllib`; no new dependency).
- Produces:
  - `python scripts/eval_agent_tools.py --base-url URL --model NAME [--stack core] [--api-key-env VAR] [--execute] [--only ID ...] [--out report.json]`.
  - Per request: the tool chosen, whether it is expected, whether the arguments are valid, and the constraint checks. It prints a summary table and the overall accuracy.
  - `score(request, tool_calls, bounds) -> dict` (pure, unit-tested without a model).

- [ ] **Step 1: Write `scripts/eval_requests.json`**

30 requests spanning every stack. Each is `{"id", "stack", "prompt", "expect_tool": [names], "expect_args": {param: value}}`, where `expect_args` is a subset that must match exactly. Write them all; here are the first six, showing the shape:

```json
[
  {"id": "help-conventions", "stack": "core", "prompt": "Before we start, what conventions does the simulator use for qubits and layers?", "expect_tool": ["help_topic"], "expect_args": {"topic": "conventions"}},
  {"id": "erasure-ws", "stack": "core", "prompt": "Run 50 Watts-Strogatz graphs (n=64, K=4, beta=0.1) with 4 circuits each in the purification picture, 256 layers, measurement rate 0.05, random edges, recording the code dimension and first-failure sizes with 512 orderings. Seed 1.", "expect_tool": ["ensemble_run"], "expect_args": {"num_system": 64, "graph": "watts_strogatz", "graph_count": 50, "circuits_per_graph": 4, "reference": "purification", "layers": 256, "measurement_rate": 0.05, "schedule": "random_edges"}},
  {"id": "phi-new-rate", "stack": "erasure", "prompt": "For the run in runs/ensemble_run-abc123, what is the erasure failure probability at e = 0.03 and the tolerable rate for 1% failure?", "expect_tool": ["analysis_erasure"], "expect_args": {"source": "runs/ensemble_run-abc123", "delta": 0.01}},
  {"id": "bell-state", "stack": "circuits", "prompt": "Make a 2-qubit Bell pair and tell me the entanglement entropy of qubit 0.", "expect_tool": ["state_new", "circuit_run"], "expect_args": {}},
  {"id": "code-422", "stack": "all", "prompt": "What is the distance of the code with stabilizers XXXX and ZZZZ?", "expect_tool": ["code_analyze"], "expect_args": {"stabilizers": ["XXXX", "ZZZZ"]}},
  {"id": "sweep-p", "stack": "all", "prompt": "Scan the measurement rate over 0.05, 0.10, 0.15 and 0.20 for n=32 brickwork circuits on a ring, 128 layers, single reference qubit, 500 trajectories each.", "expect_tool": ["ensemble_sweep"], "expect_args": {"num_system": 32, "layers": 128, "reference": "single", "schedule": "matching"}}
]
```

The remaining 24 must cover:
- `circuit_inspect`, `circuit_convert`, `circuit_crosscheck`;
- `state_query` with an observable, `state_save`;
- `traj_run` (`store=state`, `compare_stim`), `traj_probe`;
- `graph_build`, `graph_stats` for a WS collection, a `graph=file` run;
- `analysis_summary`, `analysis_probe` with `where=code`, `analysis_purification`, `analysis_compare` over a sweep;
- `result_list`, `result_get`, `result_merge`;
- `job_status` with a job id, `job_cancel`;
- `plot_make` (`vs_parameter`);
- `help_list(probes)`;
- an `edge_weights` power-law run;
- a `lattice_2d` run;
- a `stop_when_purified` purification run with `checkpoints=every_layer`.

- [ ] **Step 2: Write the failing test of the scorer**

```python
"""The evaluation script's scorer, without a model."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("eval_agent_tools", ROOT / "scripts" / "eval_agent_tools.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_requests_file_is_valid_and_covers_every_stack():
    reqs = json.loads((ROOT / "scripts" / "eval_requests.json").read_text(encoding="utf-8"))
    assert len(reqs) >= 30 and len({r["id"] for r in reqs}) == len(reqs)
    import sparsegf2.agent.tools  # noqa: F401
    from sparsegf2.agent.registry import REGISTRY

    stacks = {REGISTRY[t].stack for r in reqs for t in r["expect_tool"]}
    assert stacks == {"help", "state", "circuit", "ensemble", "traj", "code", "graph",
                      "analysis", "result", "job", "plot"}


def test_score(tmp_path):
    from sparsegf2.agent import stacks

    bounds = {b.name: b for b in stacks.resolve("all", root=tmp_path).bounds}
    req = {"id": "x", "expect_tool": ["code_analyze"], "expect_args": {"stabilizers": ["XXXX", "ZZZZ"]}}
    good = [{"name": "code_analyze", "arguments": json.dumps({"stabilizers": ["XXXX", "ZZZZ"]})}]
    s = mod.score(req, good, bounds)
    assert s == {"id": "x", "tool": "code_analyze", "tool_ok": True, "args_valid": True,
                 "args_match": True, "error": None}
    bad = [{"name": "code_analyze", "arguments": json.dumps({"stabilisers": ["XX"]})}]
    s = mod.score(req, bad, bounds)
    assert s["tool_ok"] and not s["args_valid"] and "unknown argument" in s["error"]
    assert mod.score(req, [], bounds)["tool"] is None
```

- [ ] **Step 3: Implement `scripts/eval_agent_tools.py`**

```python
"""Score a model's use of the SparseGF2 tools against a fixed set of requests.

    python scripts/eval_agent_tools.py --base-url http://localhost:8080/v1 --model qwen3.8-27b
    python scripts/eval_agent_tools.py --base-url ... --model ... --stack core --execute

The endpoint must speak OpenAI chat completions with tools (Clank's gateway, llama.cpp
server, Ollama, vLLM, ...). Each request is sent once with the stack's tools; the first tool
call is scored: right tool, arguments valid for the schema, expected arguments matched.
--execute also runs the call through the real tool (in a temporary root) and records errors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def score(request: dict[str, Any], tool_calls: list[dict[str, Any]], bounds: dict) -> dict[str, Any]:
    out = {"id": request["id"], "tool": None, "tool_ok": False, "args_valid": False,
           "args_match": False, "error": None}
    if not tool_calls:
        out["error"] = "no tool call"
        return out
    call = tool_calls[0]
    out["tool"] = call["name"]
    out["tool_ok"] = call["name"] in request["expect_tool"]
    bound = bounds.get(call["name"])
    if bound is None:
        out["error"] = f"tool {call['name']!r} is not in the stack"
        return out
    try:
        args = json.loads(call["arguments"]) if isinstance(call["arguments"], str) else call["arguments"]
        validated = bound.validate(args)
        out["args_valid"] = True
    except (ValueError, json.JSONDecodeError) as exc:
        out["error"] = str(exc)
        return out
    out["args_match"] = all(validated.get(k) == v for k, v in request.get("expect_args", {}).items())
    return out


def _chat(base_url: str, model: str, api_key: str | None, prompt: str, tools: list[dict]) -> list[dict]:
    body = {"model": model, "messages": [
        {"role": "system", "content": "You are a research assistant with SparseGF2 simulator tools. "
                                      "Answer by calling the right tool."},
        {"role": "user", "content": prompt}], "tools": tools, "tool_choice": "auto"}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
    with urllib.request.urlopen(req, timeout=600) as resp:
        msg = json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]
    return [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
            for c in msg.get("tool_calls") or []]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stack", default=None, help="override every request's stack")
    ap.add_argument("--api-key-env", default=None, help="environment variable holding an API key")
    ap.add_argument("--requests", default=str(HERE / "eval_requests.json"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sys.path.insert(0, str(HERE.parent / "src"))
    from sparsegf2.agent import dispatch, server

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    requests = json.loads(Path(args.requests).read_text(encoding="utf-8"))
    if args.only:
        requests = [r for r in requests if r["id"] in args.only]
    results = []
    with tempfile.TemporaryDirectory() as root:
        built: dict[str, Any] = {}
        for r in requests:
            stack = args.stack or r["stack"]
            if stack not in built:
                _, ctx, res = server.build(stack, root=root, quiet=True)
                built[stack] = (ctx, {b.name: b for b in res.bounds})
            ctx, bounds = built[stack]
            tools = [{"type": "function", "function": {"name": b.name, "description": b.tool.description,
                                                       "parameters": b.input_schema()}}
                     for b in bounds.values()]
            try:
                calls = _chat(args.base_url, args.model, api_key, r["prompt"], tools)
            except OSError as exc:
                calls, err = [], f"request failed: {exc}"
            else:
                err = None
            s = score(r, calls, bounds)
            if err:
                s["error"] = err
            if args.execute and s["args_valid"]:
                out = dispatch.call(ctx, bounds[s["tool"]], json.loads(calls[0]["arguments"]))
                s["executed_ok"] = not out.is_error
                s["result_head"] = out.text[:200]
            results.append(s)
            print(f"{s['id']:24s} tool={str(s['tool']):22s} ok={s['tool_ok']!s:5s} "
                  f"valid={s['args_valid']!s:5s} match={s['args_match']!s:5s} {s['error'] or ''}")
    n = len(results) or 1
    summary = {k: sum(r[k] for r in results) / n for k in ("tool_ok", "args_valid", "args_match")}
    print(f"tool choice {summary['tool_ok']:.0%}, valid arguments {summary['args_valid']:.0%}, "
          f"expected arguments {summary['args_match']:.0%} over {len(results)} requests")
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "results": results}, indent=1),
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/agent/test_eval_script.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check src tests scripts && ruff format src tests scripts
git add scripts/eval_agent_tools.py scripts/eval_requests.json tests/agent/test_eval_script.py
git commit -m "scripts: evaluate a local model's tool use against 30 requests over every stack"
```

---

### Task 31: Switch `sparsegf2-mcp` to the new server; `warmup`; release notes

**Files:**
- Modify: `src/sparsegf2/mcp_server.py`, `src/sparsegf2/cli.py`, `.mcp.json`, `CHANGELOG.md`, `src/sparsegf2/_version.py`, `docs/DESIGN.md` (agent interface section and roadmap)
- Replace: `tests/test_mcp_server.py` (old tool names) with `tests/test_mcp_entrypoint.py`
- Test: `tests/test_mcp_entrypoint.py`, `tests/test_cli.py` (the `warmup` and `mcp` pass-through)

**Interfaces:**
- Produces:
  - `sparsegf2-mcp` runs `sparsegf2.agent.server.main`. `mcp_server.build_server(stack="all", **kw)` returns the low-level server (kept for in-process clients).
  - `sparsegf2 mcp [server flags...]` passes the flags through.
  - `sparsegf2 warmup [--json]` compiles every Numba kernel.
  - Version `3.0.0.dev11`.

- [ ] **Step 1: Write the failing tests `tests/test_mcp_entrypoint.py`**

```python
"""The sparsegf2-mcp entry point serves the new agent interface."""

import asyncio
import json
import subprocess
import sys

import pytest

mcp = pytest.importorskip("mcp")

from sparsegf2 import cli, mcp_server  # noqa: E402


def test_build_server_is_the_agent_server(tmp_path):
    srv = mcp_server.build_server("help", root=tmp_path)

    async def go():
        async with mcp.Client(srv) as c:
            return [t.name for t in (await c.list_tools()).tools]

    assert asyncio.run(go()) == ["help_topic", "help_list"]


def test_console_script_list(tmp_path):
    out = subprocess.run([sys.executable, "-m", "sparsegf2.mcp_server", "--stack", "core",
                          "--root", str(tmp_path), "--list"], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    assert "ensemble_run" in out.stdout and "total" in out.stdout


def test_cli_mcp_passes_flags(tmp_path, capsys):
    assert cli.main(["mcp", "--stack", "help", "--root", str(tmp_path), "--list"]) == 0
    assert "help_topic" in capsys.readouterr().out


def test_warmup(capsys):
    assert cli.main(["warmup", "--json"]) == 0
    data = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert data["compiled"] and data["seconds"] >= 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_mcp_entrypoint.py -q`
Expected: FAIL (`build_server` still builds the old server, and `warmup` does not exist)

- [ ] **Step 3: Replace `src/sparsegf2/mcp_server.py`**

```python
"""``sparsegf2-mcp``: the SparseGF2 MCP server (see sparsegf2.agent and docs/MCP_TOOLS.md).

Run ``sparsegf2-mcp --stack core`` (or any stack, bundle, preset or custom stack);
``sparsegf2-mcp --list`` prints the tools and their sizes. The ``mcp`` package
(``pip install 'sparsegf2[mcp]'``) is imported only here and in sparsegf2.agent.server.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["SERVER_NAME", "build_server", "main", "require_mcp"]

SERVER_NAME = "sparsegf2"
_INSTALL_HINT = "The SparseGF2 MCP server needs the optional 'mcp' extra: pip install 'sparsegf2[mcp]'"


def require_mcp() -> None:
    try:
        from mcp.server.lowlevel import Server  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def build_server(stack: str = "all", **kwargs: Any) -> Any:
    """The low-level MCP server for ``stack`` (keyword arguments as sparsegf2.agent.server.build)."""
    require_mcp()
    from sparsegf2.agent import server

    return server.build(stack, **kwargs)[0]


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    try:
        require_mcp()
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    from sparsegf2.agent import server

    return server.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
```

Delete `tests/test_mcp_server.py`. It tests the removed tools; every tool is now covered under `tests/agent/`. `git rm tests/test_mcp_server.py`.

- [ ] **Step 4: Pass flags through `sparsegf2 mcp`, and add `warmup`, in `src/sparsegf2/cli.py`**

In `_build_parser`, replace `add("mcp", "Start the MCP server over stdio.", _MCP_HELP)` with:

```python
    p = add("mcp", "Start the MCP server over stdio (flags as sparsegf2-mcp).", _MCP_HELP)
    p.add_argument("server_args", nargs=argparse.REMAINDER, help="flags for sparsegf2-mcp")
    add("warmup", "Compile every kernel once into the Numba cache (first use is then fast).",
        _WARMUP_HELP)
```

with the help text:

```python
_WARMUP_HELP = """\
examples:
  sparsegf2 warmup
  NUMBA_CACHE_DIR=/workspace/.numba-cache sparsegf2 warmup   # Clank sandbox
"""
```

and update `_MCP_HELP`'s examples to `sparsegf2 mcp --stack core --list`.

In `main`, replace the `if args.command == "mcp":` block with:

```python
    if args.command == "mcp":
        from sparsegf2 import mcp_server

        rest = [a for a in args.server_args if a != "--json"]
        return mcp_server.main(rest)
```

Because the `mcp` subparser gets `--json` from `add`, `argparse.REMAINDER` keeps everything after `mcp`. Check that `["mcp", "--stack", "help", "--list"]` arrives as `server_args == ["--stack", "help", "--list"]`. If argparse consumes `--json`-like flags first, drop the `--json` option for `mcp`: build that subparser with `sub.add_parser` directly instead of `add`.

Add the command:

```python
def _cmd_warmup(args: argparse.Namespace) -> _Outcome:
    """Run a tiny instance of every compiled kernel so later calls start immediately."""
    import sparsegf2 as sg
    from sparsegf2 import graphs, mipt

    t0 = time.perf_counter()
    for phases in (True, False):
        sim = sg.Simulator(4, phases=phases, seed=0)
        sim.run(sg.Circuit("H 0\nCX 0 1\nCLIFFORD_2Q(5) 1 2\nM 0\nMX 1\nMPP X2*Z3\nR 3\nTICK"))
        sim.entropy([0, 1])
    sg.sample(sg.Circuit("H 0\nCX 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]"), 4, seed=0)
    probes = [mipt.code_dimension(), mipt.half_cut_entropy(), mipt.entropy([0, 1]),
              mipt.mutual_information([0, 1], [4, 5]), mipt.tripartite_information(),
              mipt.entropy_profile(), mipt.contiguous_distance(), mipt.lost_information([0, 1]),
              mipt.erasure_lost_profile(2), mipt.erasure_first_failure(4),
              mipt.uncorrectable_region_counts(2), mipt.code_snapshot()]
    for schedule in ("matching", "random_edges", "all_edges"):
        for reference in ("purification", "single"):
            mipt.MonitoredCircuit(
                num_system=8, edges=graphs.watts_strogatz(8, 4, 0.2, seed=0), layers=3,
                measurement_rate=0.2, schedule=schedule, reference=reference,
                scramble_gates=2, scramble_matchings=1, probes=probes,
            ).run(2, seed=0)
    code = sg.StabilizerCode(["XXXX", "ZZZZ"])
    code.first_failure_sizes(4, seed=0)
    code.uncorrectable_counts(2)
    graphs.stats(graphs.cycle(8), 8)
    return {"compiled": True, "seconds": round(time.perf_counter() - t0, 2),
            "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR")}, EXIT_OK
```

Register `"warmup": _cmd_warmup` in `_COMMANDS`, and add `import os` if it is missing.

- [ ] **Step 5: `.mcp.json`, the version and the changelog**

`.mcp.json`:

```json
{
  "mcpServers": {
    "sparsegf2": {
      "command": "sparsegf2-mcp",
      "args": ["--stack", "all"]
    }
  }
}
```

`src/sparsegf2/_version.py`: `__version__ = "3.0.0.dev11"`.

`CHANGELOG.md`, a new top entry `## 3.0.0.dev11: a general agent interface (MCP)`, with these bullets:
- The new server: 31 tools in 11 stacks covering the whole library, flat schemas, the call grammar, text results with `next:` lines, background jobs for any long tool, and root confinement.
- Stacks: built-in stacks, the `core` and `all` bundles, the presets, custom stacks, `sparsegf2 stacks`, and `sparsegf2 mcp` flags.
- Library additions:
  - `analysis.mean_first_failure`, `distance_bounds`, `hybrid_failure_by_size`, `junction_dip`;
  - `graphs.stats`, `shortcut_fraction`, `save` and `load`, plus the `file` graph family in specs;
  - `StabilizerCode.contiguous_distance` and `from_simulator`;
  - `Simulator.save` / `load`;
  - `MonitoredCircuit.replay` and `RANDOMNESS_CONTRACT`;
  - `sparsegf2.crosscheck`.
- `sparsegf2 warmup`, `scripts/eval_agent_tools.py`, and `docs/MCP_TOOLS.md` (generated).
- An "Agent interface" subsection: interface version 1, and the breaking changes from spec 14 (renamed tools, flat ensemble parameters instead of `spec`, observables and probes as call strings, prompts moved into `help_topic(recipes)`).

`docs/DESIGN.md`: in the agent-interface section, replace the description of the 10 tools with a paragraph on the registry, stacks and the grammar, pointing to the spec and to `MCP_TOOLS.md`. In section 14, mark the agent-interface redesign as done.

- [ ] **Step 6: Run everything**

```bash
ruff check src tests scripts && ruff format --check src tests scripts
python -m pytest -q
python -m sparsegf2.agent.docs > docs/MCP_TOOLS.md && git diff --stat docs/MCP_TOOLS.md
sparsegf2-mcp --stack core --list
sparsegf2 warmup
```

Expected:
- the suite passes;
- `MCP_TOOLS.md` is unchanged (or is regenerated and committed);
- `core` lists 11 tools at 18,000 characters or fewer;
- `warmup` reports `compiled: true`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Switch sparsegf2-mcp to the general agent interface; warmup command; 3.0.0.dev11"
```

---
