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
