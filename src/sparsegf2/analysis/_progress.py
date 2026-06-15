"""Progress-bar helpers shared by the sweep, the study save, and augment.

``tqdm`` is the optional ``progress`` extra. When it is absent, or when
``progress`` is false (the headless / HPC path), every helper here degrades to a
plain pass-through, so importing this module or running without the extra never
fails. ``tqdm.auto`` is used so the same call renders a widget bar in Jupyter and
a text bar in a terminal.
"""

from __future__ import annotations

import warnings


def _tqdm(progress: bool):
    """Return the ``tqdm.auto.tqdm`` class, or ``None`` to disable the bar.

    The bar is disabled when ``progress`` is false or when the
    ``SPARSEGF2_PROGRESS`` environment kill-switch is set falsey (the headless /
    HPC path: silence every bar for the whole job without touching call sites).
    """
    from sparsegf2.analysis._runtime import progress_force_off

    if not progress or progress_force_off():
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        warnings.warn(
            "progress=True needs tqdm (pip install sparsegf2[progress]); "
            "running without a progress bar",
            stacklevel=3,
        )
        return None
    return tqdm


def make_bar(total: int | None, *, desc: str, unit: str, progress: bool):
    """A live tqdm bar, or ``None`` when progress is off / tqdm is missing.

    The caller drives it (``bar.set_postfix(...)`` / ``bar.update(1)`` /
    ``bar.close()``), so the bar can name the exact cell currently running.
    """
    tqdm = _tqdm(progress)
    return tqdm(total=total, desc=desc, unit=unit) if tqdm is not None else None


def resolve_bar(progress, *, total: int, desc: str, unit: str):
    """Make a bar, or reuse an existing one, for one phase of a longer job.

    ``progress`` is either a bool (``True`` makes and *owns* a new bar; ``False``
    means no bar) or an already-open tqdm bar to **share** across phases. Returns
    ``(bar, owned)``: when sharing, ``owned`` is ``False`` and the caller must
    update but not close it, so a single bar can span e.g. the simulation phase
    and the tableau-write phase as one whole-study progress meter.
    """
    if progress is True:
        return make_bar(total, desc=desc, unit=unit, progress=True), True
    if progress is False or progress is None:
        return None, True
    # an existing bar handed down by Study.run: relabel the phase, keep its total
    set_desc = getattr(progress, "set_description", None)
    if set_desc is None:
        return None, True
    set_desc(desc)
    return progress, False
