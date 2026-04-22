"""
Built-in matplotlib style presets for plotting primitives.

MVP: one preset ``"research"`` with sensible defaults for publication-style
figures. Users can pass ``style=None`` to inherit the global matplotlib rc.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Optional

import matplotlib.pyplot as plt


STYLE_PRESETS: Dict[str, Dict[str, object]] = {
    "research": {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
    },
}


@contextmanager
def rc_preset(name: Optional[str]):
    if name is None:
        yield
        return
    if name not in STYLE_PRESETS:
        raise ValueError(f"unknown style {name!r}; valid: {list(STYLE_PRESETS)}")
    with plt.rc_context(STYLE_PRESETS[name]):
        yield


__all__ = ["STYLE_PRESETS", "rc_preset"]
