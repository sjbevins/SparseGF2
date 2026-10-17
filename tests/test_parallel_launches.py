"""Parallel kernels launched from several Python threads: they overlap only where Numba can.

Numba releases the GIL while a parallel region runs, whatever the kernel's ``nogil`` says, so
holding the GIL cannot make two launches take turns. Under the ``workqueue`` threading layer a
second region that starts while one runs terminates the process. ``sparsegf2._parallel.threads``,
around every parallel launch of the package, holds a process-wide lock under every layer but
``omp`` and ``tbb``.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numba
import pytest

from sparsegf2 import _parallel

ROOT = Path(__file__).resolve().parents[1]


def _lock_is_held_elsewhere() -> bool:
    """True when another thread holds the launch lock (probed without blocking)."""
    if _parallel._LAUNCH_LOCK.acquire(blocking=False):
        _parallel._LAUNCH_LOCK.release()
        return False
    return True


@pytest.mark.parametrize(
    "layer, take_turns",
    [("workqueue", True), ("omp", False), ("tbb", False), ("a_future_layer", True)],
)
@pytest.mark.parametrize("num_threads", [None, 1])
def test_blocks_take_turns_unless_the_layer_runs_launches_at_once(
    layer, take_turns, num_threads, monkeypatch
):
    """While one thread is inside a ``threads`` block, another finds the launch lock held
    exactly when the layer cannot run two regions at once; it is free again afterwards."""
    monkeypatch.setattr(numba, "threading_layer", lambda: layer)
    assert _parallel.concurrent_launches() is not take_turns
    inside, leave = threading.Event(), threading.Event()
    counts = []  # the holder's thread count before, inside and after its block

    def hold():
        counts.append(numba.get_num_threads())
        with _parallel.threads(num_threads):
            counts.append(numba.get_num_threads())
            inside.set()
            leave.wait()
        counts.append(numba.get_num_threads())

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert inside.wait(timeout=60)
        assert _lock_is_held_elsewhere() is take_turns
    finally:
        leave.set()
        holder.join()
    assert not _lock_is_held_elsewhere()
    before = counts[0]
    assert counts == [before, before if num_threads is None else 1, before]


def test_a_block_may_nest_and_an_error_releases_the_lock(monkeypatch):
    monkeypatch.setattr(numba, "threading_layer", lambda: "workqueue")
    before = numba.get_num_threads()
    # a block inside a block of the same thread does not wait for itself
    with pytest.raises(RuntimeError, match="inside"), _parallel.threads(1), _parallel.threads(None):
        assert numba.get_num_threads() == 1
        raise RuntimeError("inside")
    assert numba.get_num_threads() == before

    def probe(out):
        out.append(_lock_is_held_elsewhere())

    out: list[bool] = []
    t = threading.Thread(target=probe, args=(out,))
    t.start()
    t.join()
    assert out == [False]
    with pytest.raises(ValueError, match="num_threads must be at least 1"), _parallel.threads(0):
        pass


_CONCURRENT_SCRIPT = r"""
import json
import threading

import numba

from sparsegf2 import graphs, mipt
from sparsegf2.tableau import PhaseFreeTableau, physical_tableaux, to_bytes_batch

n = 24
spec = mipt.MonitoredCircuit(
    num_system=n, edges=graphs.cycle(n), layers=2 * n, measurement_rate=0.1,
    reference="purification", probes=[mipt.code_dimension(), mipt.tableau_snapshot()],
)
res = spec.run(64, seed=3)
generators, counts = res.snapshots["tableau_snapshot"]
N = spec.num_qubits
one = [res.tableau(t, 0) for t in range(64)]
expected = ([t.to_bytes() for t in one], [t.physical_tableau(n).to_bytes() for t in one])
engine = spec.run(16, seed=11).values["code_dimension"].tolist()

start = threading.Barrier(4)
failures = []

def batches():
    start.wait()
    for _ in range(30):
        joint = PhaseFreeTableau.from_words_batch(generators[:, 0], counts[:, 0], N)
        physical = physical_tableaux(joint, n)
        if (to_bytes_batch(joint), to_bytes_batch(physical)) != expected:
            failures.append("batch")

def simulations():
    start.wait()
    for _ in range(30):
        if spec.run(16, seed=11).values["code_dimension"].tolist() != engine:
            failures.append("engine")

workers = [threading.Thread(target=batches) for _ in range(3)]
workers.append(threading.Thread(target=simulations))
for w in workers:
    w.start()
for w in workers:
    w.join()
print(json.dumps({"layer": numba.threading_layer(), "failures": failures}))
"""


@pytest.mark.skipif(
    os.environ.get("NUMBA_DISABLE_JIT") == "1", reason="no parallel regions without the JIT"
)
def test_concurrent_batches_and_runs_under_the_workqueue_layer():
    """Three Python threads canonicalizing snapshots in batches and one running the engine,
    30 times each, under Numba's workqueue layer: without the launch lock the process
    terminates ("Concurrent access has been detected", exit code 3); with it every call
    returns the results of the one-at-a-time methods and of a run on its own."""
    env = {
        **os.environ,
        "NUMBA_THREADING_LAYER": "workqueue",
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _CONCURRENT_SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=ROOT,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report == {"layer": "workqueue", "failures": []}
