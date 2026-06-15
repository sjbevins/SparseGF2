"""Runtime helpers for headless and HPC execution.

Two concerns a cluster run cares about, kept in one small place:

* :func:`available_cores` reports how many CPU cores this process may actually
  use, honoring a scheduler allocation rather than the full node, so a parallel
  sweep with ``n_jobs=-1`` does not oversubscribe a partial-node job.
* :func:`progress_force_off` is a global kill-switch for progress bars, driven by
  the ``SPARSEGF2_PROGRESS`` environment variable. Set ``SPARSEGF2_PROGRESS=0``
  once in a batch job script and every sweep and study in that job runs silently,
  with no edits to notebook or script code.
"""

from __future__ import annotations

import os

from sparsegf2.errors import InvalidArgumentError

# Scheduler variables that report the per-task core allocation, most specific
# first. SLURM exports CPUS_PER_TASK for an srun step; PBS/Torque uses NCPUS.
_ALLOC_ENV_VARS = ("SLURM_CPUS_PER_TASK", "PBS_NCPUS", "NSLOTS")


def available_cores() -> int:
    """Number of CPU cores this process may use, honoring an HPC allocation.

    The resolution order is, from most to least specific:

    1. a batch-scheduler allocation variable (``SLURM_CPUS_PER_TASK``,
       ``PBS_NCPUS``, ``NSLOTS``),
    2. the Linux scheduler affinity mask (:func:`os.sched_getaffinity`, which
       reflects what a cgroup or cpuset actually grants),
    3. :func:`os.cpu_count` (the whole node).

    Always returns at least 1. Pass the result as ``n_jobs`` on a cluster so a
    partial-node allocation is filled but not oversubscribed.
    """
    for var in _ALLOC_ENV_VARS:
        val = os.environ.get(var, "").strip()
        if val.isdigit() and int(val) > 0:
            return int(val)
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            n = len(getaffinity(0))
            if n > 0:
                return n
        except OSError:  # pragma: no cover - platform dependent
            pass
    return os.cpu_count() or 1


def resolve_n_jobs(n_jobs: int) -> int:
    """Turn a possibly-negative ``n_jobs`` into a concrete worker count.

    Mirrors the joblib convention (``-1`` means all cores, ``-2`` all but one)
    but counts with :func:`available_cores`, so a negative request fills the
    *allocation* rather than the whole node. Positive values pass through; ``0``
    is rejected with a clear error (it has no meaning as a worker count, and
    joblib would otherwise raise a cryptic one).
    """
    if n_jobs == 0:
        raise InvalidArgumentError(
            "n_jobs must be a positive worker count or a negative joblib-style "
            "count (-1 = all allocated cores); got 0"
        )
    if n_jobs > 0:
        return n_jobs
    return max(1, available_cores() + 1 + n_jobs)


_FALSEY = {"0", "false", "no", "off"}


def progress_force_off() -> bool:
    """``True`` when ``SPARSEGF2_PROGRESS`` is set to a falsey value.

    The batch / HPC kill-switch: ``export SPARSEGF2_PROGRESS=0`` silences every
    progress bar for the whole job regardless of the ``progress=`` argument, so
    log files do not fill with carriage-return bar redraws.
    """
    return os.environ.get("SPARSEGF2_PROGRESS", "").strip().lower() in _FALSEY


__all__ = ["available_cores", "resolve_n_jobs", "progress_force_off"]
