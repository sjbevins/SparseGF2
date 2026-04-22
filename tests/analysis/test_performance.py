"""Performance tests for sparsegf2.analysis functions.

Verifies that analysis functions meet wall-clock timing budgets on the
initial Bell-pair state (abar=2.0), and that Numba JIT warmup cost is
amortized correctly on subsequent calls.
"""
import time
import subprocess
import sys
import pytest
from sparsegf2.core.sparse_tableau import SparseGF2
from sparsegf2.analysis import (
    compute_weight_stats,
    build_tanner_graph,
    verify_weight_mass_identity,
)


# ── D1: Weight stats cost ────────────────────────────────────────

def test_weight_stats_cost_n512():
    """compute_weight_stats on n=512 (abar=2.0) must average < 50 ms."""
    sim = SparseGF2(512)

    # Warmup: trigger Numba JIT compilation
    compute_weight_stats(sim)

    n_calls = 10
    t0 = time.perf_counter()
    for _ in range(n_calls):
        compute_weight_stats(sim)
    elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / n_calls) * 1000

    print(f"\n  [D1] compute_weight_stats n=512: {avg_ms:.2f} ms avg "
          f"({elapsed*1000:.1f} ms total / {n_calls} calls)")

    assert avg_ms < 50, (
        f"compute_weight_stats too slow: {avg_ms:.2f} ms avg (budget: 50 ms)")


# ── D2: Tanner graph construction cost ───────────────────────────

def test_tanner_graph_cost_n128():
    """build_tanner_graph on n=128 must average < 500 ms."""
    sim = SparseGF2(128)

    # Warmup
    build_tanner_graph(sim)

    n_calls = 5
    t0 = time.perf_counter()
    for _ in range(n_calls):
        build_tanner_graph(sim)
    elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / n_calls) * 1000

    print(f"\n  [D2a] build_tanner_graph n=128: {avg_ms:.2f} ms avg "
          f"({elapsed*1000:.1f} ms total / {n_calls} calls)")

    assert avg_ms < 500, (
        f"build_tanner_graph n=128 too slow: {avg_ms:.2f} ms avg (budget: 500 ms)")


def test_tanner_graph_cost_n256():
    """build_tanner_graph on n=256 must average < 2000 ms."""
    sim = SparseGF2(256)

    # Warmup
    build_tanner_graph(sim)

    n_calls = 5
    t0 = time.perf_counter()
    for _ in range(n_calls):
        build_tanner_graph(sim)
    elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / n_calls) * 1000

    print(f"\n  [D2b] build_tanner_graph n=256: {avg_ms:.2f} ms avg "
          f"({elapsed*1000:.1f} ms total / {n_calls} calls)")

    assert avg_ms < 2000, (
        f"build_tanner_graph n=256 too slow: {avg_ms:.2f} ms avg (budget: 2000 ms)")


# ── D3: Weight mass identity verification cost ───────────────────

def test_verify_weight_mass_identity_cost_n512():
    """verify_weight_mass_identity on n=512 must average < 10 ms."""
    sim = SparseGF2(512)

    # Warmup
    verify_weight_mass_identity(sim)

    n_calls = 10
    t0 = time.perf_counter()
    for _ in range(n_calls):
        verify_weight_mass_identity(sim)
    elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / n_calls) * 1000

    print(f"\n  [D3] verify_weight_mass_identity n=512: {avg_ms:.2f} ms avg "
          f"({elapsed*1000:.1f} ms total / {n_calls} calls)")

    assert avg_ms < 10, (
        f"verify_weight_mass_identity too slow: {avg_ms:.2f} ms avg (budget: 10 ms)")


# ── D4: Numba JIT warmup amortization ────────────────────────────

def _build_jit_warmup_script(pkg_dir):
    """Build a script that clears the Numba cache then times cold vs warm calls."""
    return (
        "import pathlib, time\n"
        f"cache_dir = pathlib.Path(r'{pkg_dir}') / 'analysis' / '__pycache__'\n"
        "for p in cache_dir.glob('_numba_kernels*.nbi'):\n"
        "    p.unlink(missing_ok=True)\n"
        "for p in cache_dir.glob('_numba_kernels*.nbc'):\n"
        "    p.unlink(missing_ok=True)\n"
        "from sparsegf2.core.sparse_tableau import SparseGF2\n"
        "from sparsegf2.analysis import compute_weight_stats\n"
        "sim = SparseGF2(256)\n"
        "t0 = time.perf_counter()\n"
        "compute_weight_stats(sim)\n"
        "first_ms = (time.perf_counter() - t0) * 1000\n"
        "n = 5\n"
        "t0 = time.perf_counter()\n"
        "for _ in range(n):\n"
        "    compute_weight_stats(sim)\n"
        "subsequent_ms = (time.perf_counter() - t0) * 1000 / n\n"
        "print(f'{first_ms:.4f} {subsequent_ms:.4f}')\n"
    )


def test_numba_jit_warmup_amortization():
    """First call to compute_weight_stats (cold JIT) must be >= 5x slower
    than subsequent calls. Uses a subprocess with cleared Numba cache."""
    import sparsegf2
    pkg_dir = str(sparsegf2.__path__[0]).replace("\\", "/")
    script = _build_jit_warmup_script(pkg_dir)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    # Parse "first_ms subsequent_ms" from last line of stdout
    output_line = result.stdout.strip().split("\n")[-1]
    first_ms, subsequent_ms = map(float, output_line.split())
    speedup = first_ms / subsequent_ms if subsequent_ms > 0 else float("inf")

    print(f"\n  [D4] JIT warmup (subprocess): first call = {first_ms:.1f} ms, "
          f"subsequent avg = {subsequent_ms:.2f} ms, "
          f"speedup = {speedup:.1f}x")

    assert speedup >= 5, (
        f"JIT warmup not amortized: first call {first_ms:.1f} ms, "
        f"subsequent avg {subsequent_ms:.2f} ms, "
        f"speedup {speedup:.1f}x (need >= 5x)")
