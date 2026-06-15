"""Crash-only run logging.

A long :meth:`Study.run` / :func:`sweep` / :meth:`Study.augment` emits a detailed
trace through the ``sparsegf2`` logger. :func:`crash_log` streams that trace to a
file *while the run is in progress* and **keeps the file only if the run raises**; a clean run deletes it on the way
out. So you ignore logs entirely until
something goes wrong, and then the full descriptive trace of exactly what
happened (down to the cell and phase that failed, plus the traceback) is waiting
on disk.

Two detail levels:

* ``"info"`` (default): every cell, every phase (setup / scramble / warmup /
  measured loop), every observable, and every save step.
* ``"trace"``: additionally every individual layer (gates, measurements, the
  order parameter), verbose, for deep debugging.

While the context is active the ``sparsegf2`` logger is detached from the root
logger, so the verbose trace goes to the file only and never floods the console.
"""

from __future__ import annotations

import contextlib
import logging
import traceback
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from sparsegf2.errors import InvalidArgumentError

#: The package-wide logger every layer emits through. Quiet by default (no
#: handler beyond the import-time NullHandler); :func:`crash_log` is what gives
#: it a destination and a level.
logger = logging.getLogger("sparsegf2")

_LEVELS = {"info": logging.INFO, "trace": logging.DEBUG}

_FORMAT = "%(asctime)s %(levelname)-5s %(name)s | %(message)s"


@contextlib.contextmanager
def crash_log(
    log_dir: str | Path,
    *,
    label: str = "run",
    detail: str = "info",
    enabled: bool = True,
) -> Iterator[Path | None]:
    """Capture a verbose trace, kept on disk only if the wrapped block raises.

    Parameters
    ----------
    log_dir
        Directory the log is written under (created if needed).
    label
        Prefix for the log filename (e.g. ``"study"``, ``"augment"``).
    detail
        ``"info"`` (default) or ``"trace"`` (adds per-layer records).
    enabled
        When false, a no-op (yields ``None``), for the headless path that wants no
        logging at all.

    Yields the in-progress log path (or ``None`` when disabled). On a clean exit
    the file is removed; on any exception it is renamed to
    ``<label>-FAILED-<timestamp>.log`` and its location is printed.
    """
    if not enabled:
        yield None
        return
    if detail not in _LEVELS:
        raise InvalidArgumentError(f"detail must be one of {sorted(_LEVELS)}, got {detail!r}")

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    in_progress = log_dir / f".{label}.{stamp}.inprogress.log"

    handler = logging.FileHandler(in_progress, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.setLevel(_LEVELS[detail])

    prev_level, prev_propagate = logger.level, logger.propagate
    logger.setLevel(_LEVELS[detail])
    logger.propagate = False  # trace goes to the file only, not the console
    logger.addHandler(handler)
    logger.info("===== run start [%s] detail=%s =====", label, detail)
    try:
        yield in_progress
    except BaseException as exc:
        logger.error("run did NOT complete normally: %s\n%s", exc, traceback.format_exc())
        _detach(handler, prev_level, prev_propagate)
        kept = log_dir / f"{label}-FAILED-{stamp}.log"
        in_progress.replace(kept)
        print(f"[sparsegf2] run failed before completing; full trace saved to: {kept}")
        raise
    else:
        logger.info("===== run complete [%s] =====", label)
        _detach(handler, prev_level, prev_propagate)
        in_progress.unlink(missing_ok=True)  # clean run -> no log kept


def _detach(handler: logging.Handler, prev_level: int, prev_propagate: bool) -> None:
    handler.close()
    logger.removeHandler(handler)
    logger.setLevel(prev_level)
    logger.propagate = prev_propagate
