"""
Validation agent for the single_ref MIPT-probe picture.

Runs a minimal all-to-all (complete-graph) sweep at n=4, 6, 8 with p=0 and
p=1, and asserts the invariants that a correct single_ref implementation
must satisfy:

1. The reference qubit at index n is never involved in any gate or
   measurement during the circuit schedule (verified by walking the
   :class:`CircuitBuilder` output directly).
2. The per-sample reference entropy ``k`` returned by the runner is
   always an integer in ``{0, 1}`` (a single qubit in a stabilizer state
   has integer von Neumann entropy).
3. Known analytic limits:
     - at ``p == 0.0`` no measurements are applied, so the initial Bell
       pair between system qubit 0 and reference qubit n survives any
       Clifford on qubits 0..n-1 and S(qubit n) = 1.
     - at ``p == 1.0`` every qubit is measured in Z at every layer, so
       the Bell-pair correlation is fully collapsed by the first
       measurement of qubit 0 and S(qubit n) = 0.

Runnable both under pytest and as ``python tests/test_single_ref.py``.
"""
from __future__ import annotations

import sys

import pytest

from sparsegf2.circuits import CircuitConfig, SimulationRunner
from sparsegf2.circuits.builder import CircuitBuilder


SIZES = (4, 6, 8)
RATES = (0.0, 1.0)


def _cfg(n: int, p: float, *, record_time_series: bool = False) -> CircuitConfig:
    return CircuitConfig(
        graph_spec="complete",
        n=n,
        picture="single_ref",
        matching_mode="round_robin",
        p=p,
        depth_factor=4,
        record_time_series=record_time_series,
    )


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("p", RATES)
def test_reference_qubit_never_touched(n: int, p: float):
    """Assertion 1: ref qubit at index n is never gated or measured."""
    cfg = _cfg(n, p)
    builder = CircuitBuilder(cfg, sample_seed=0)
    inspected = 0
    for layer in builder.layers():
        for qi, qj in layer.gate_pairs:
            assert qi < n and qj < n, (
                f"n={n} p={p}: gate involves ref qubit ({qi},{qj})"
            )
            inspected += 1
        for q in layer.meas_qubits:
            assert q < n, f"n={n} p={p}: measurement on ref qubit q={q}"
            inspected += 1
    assert inspected > 0, f"n={n} p={p}: builder produced no ops"


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("p", RATES)
def test_k_is_integer_0_or_1(n: int, p: float):
    """Assertion 2: a single sample's k is an integer in {0, 1}."""
    rec = SimulationRunner(_cfg(n, p), warmup_jit=False).run(sample_seed=0)
    assert isinstance(rec.k, int), (
        f"n={n} p={p}: k is {type(rec.k).__name__}, expected int"
    )
    assert rec.k in (0, 1), (
        f"n={n} p={p}: k={rec.k} not in {{0, 1}}"
    )


@pytest.mark.parametrize("n", SIZES)
def test_entropy_limit_at_p_zero(n: int):
    """Assertion 3a: at p=0 the initial Bell pair survives, S(ref)=1."""
    rec = SimulationRunner(_cfg(n, 0.0), warmup_jit=False).run(sample_seed=0)
    assert rec.k == 1, (
        f"n={n} p=0.0: k={rec.k}, expected 1 (unitary-only evolution on "
        "system cannot change S(qubit n))"
    )


@pytest.mark.parametrize("n", SIZES)
def test_entropy_limit_at_p_one(n: int):
    """Assertion 3b: at p=1 the Bell pair is destroyed, S(ref)=0."""
    rec = SimulationRunner(_cfg(n, 1.0), warmup_jit=False).run(sample_seed=0)
    assert rec.k == 0, (
        f"n={n} p=1.0: k={rec.k}, expected 0 (full Z-measurement of "
        "qubit 0 collapses the Bell-pair correlation)"
    )


def test_timeseries_has_expected_length_and_endpoints():
    """Smoke test: record_time_series=True emits a (total_layers+1,) array
    whose first entry is 1 (initial Bell pair) and whose last entry equals
    the endpoint ``k`` field of SampleRecord."""
    for n in SIZES:
        for p in RATES:
            cfg = _cfg(n, p, record_time_series=True)
            rec = SimulationRunner(cfg, warmup_jit=False).run(sample_seed=0)
            ts = rec.ref_entropy_timeseries
            assert ts is not None, (
                f"n={n} p={p}: record_time_series=True but ref_entropy_timeseries is None"
            )
            assert ts.shape == (cfg.total_layers() + 1,), (
                f"n={n} p={p}: unexpected timeseries shape {ts.shape}, "
                f"expected ({cfg.total_layers() + 1},)"
            )
            assert int(ts[0]) == 1, (
                f"n={n} p={p}: initial timeseries entry {ts[0]}, expected 1"
            )
            assert int(ts[-1]) == rec.k, (
                f"n={n} p={p}: final timeseries entry {ts[-1]} != k={rec.k}"
            )


def _run_standalone() -> int:
    ok = True
    for n in SIZES:
        for p in RATES:
            try:
                cfg = _cfg(n, p, record_time_series=True)
                test_reference_qubit_never_touched(n, p)
                rec = SimulationRunner(cfg, warmup_jit=False).run(sample_seed=0)
                assert isinstance(rec.k, int) and rec.k in (0, 1)
                if p == 0.0 and rec.k != 1:
                    raise AssertionError(
                        f"n={n} p=0.0: k={rec.k}, expected 1"
                    )
                if p == 1.0 and rec.k != 0:
                    raise AssertionError(
                        f"n={n} p=1.0: k={rec.k}, expected 0"
                    )
                ts = rec.ref_entropy_timeseries
                assert ts.shape == (cfg.total_layers() + 1,)
                assert int(ts[0]) == 1
                assert int(ts[-1]) == rec.k
                print(
                    f"PASS  all-to-all  n={n:2d}  p={p:.1f}  "
                    f"k={rec.k}  timeseries_len={ts.shape[0]}"
                )
            except AssertionError as exc:
                ok = False
                print(f"FAIL  all-to-all  n={n} p={p}: {exc}")
    if ok:
        print("\nAll single_ref assertions PASSED.")
        return 0
    print("\nsingle_ref test FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(_run_standalone())
