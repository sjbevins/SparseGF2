"""Headless / HPC runtime helpers: core allocation + the progress kill-switch."""

from __future__ import annotations

import pytest

from sparsegf2.analysis import available_cores
from sparsegf2.analysis._progress import _tqdm
from sparsegf2.analysis._runtime import progress_force_off, resolve_n_jobs


def test_available_cores_at_least_one():
    assert available_cores() >= 1


def test_available_cores_honors_scheduler_allocation(monkeypatch):
    """A scheduler core allocation takes priority over the node's cpu_count."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert available_cores() == 8
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "garbage")  # ignored, fall through
    assert available_cores() >= 1
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setenv("PBS_NCPUS", "5")
    assert available_cores() == 5


def test_resolve_n_jobs_negative_uses_allocation(monkeypatch):
    """Negative n_jobs counts the allocation, not the whole node; non-negative
    passes through (joblib convention, but allocation-aware)."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    assert resolve_n_jobs(-1) == 16  # all allocated cores
    assert resolve_n_jobs(-2) == 15  # all but one
    assert resolve_n_jobs(-16) == 1  # clamped to >= 1
    assert resolve_n_jobs(-100) == 1
    assert resolve_n_jobs(4) == 4
    assert resolve_n_jobs(1) == 1
    # 0 has no meaning as a worker count: reject it with a clear error rather
    # than letting joblib raise a cryptic one downstream.
    from sparsegf2.errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError):
        resolve_n_jobs(0)


def test_progress_kill_switch(monkeypatch):
    """SPARSEGF2_PROGRESS=falsey disables every bar regardless of progress=True."""
    monkeypatch.delenv("SPARSEGF2_PROGRESS", raising=False)
    assert not progress_force_off()
    for off in ("0", "false", "no", "off", "OFF", "False"):
        monkeypatch.setenv("SPARSEGF2_PROGRESS", off)
        assert progress_force_off()
        assert _tqdm(True) is None  # forced off even when progress is requested
    monkeypatch.setenv("SPARSEGF2_PROGRESS", "1")
    assert not progress_force_off()
