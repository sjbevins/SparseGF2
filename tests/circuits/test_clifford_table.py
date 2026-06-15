"""Tests for the Stim-free Sp(4, F_2) Clifford table."""

from __future__ import annotations

import numpy as np

from sparsegf2.circuits._clifford_table import SP4_SIZE, sp4_table
from sparsegf2.core.symplectic import is_symplectic


def test_table_shape_and_dtype():
    t = sp4_table()
    assert t.shape == (720, 4, 4)
    assert t.dtype == np.uint8
    assert SP4_SIZE == 720


def test_all_720_are_distinct_symplectic_matrices():
    t = sp4_table()
    # every matrix is in Sp(4, F_2)
    assert all(is_symplectic(t[k]) for k in range(len(t)))
    # all 720 are distinct
    seen = {t[k].tobytes() for k in range(len(t))}
    assert len(seen) == 720


def test_table_is_cached_same_object():
    # The core caches it; we return that cached view, so identity holds.
    assert sp4_table() is sp4_table()


def test_table_is_read_only():
    t = sp4_table()
    assert not t.flags.writeable


def test_no_runtime_stim_import():
    """The circuits package must never import ``stim`` at runtime.

    Source-grep for ``import stim`` / ``from stim`` under the circuits
    package. We do not assert ``"stim" not in sys.modules`` because the
    core's parity tests legitimately import stim into the shared pytest
    process; see ``tests/test_core_public_api.py``.
    """
    import pathlib
    import re

    pkg_root = pathlib.Path(__file__).resolve().parents[2] / "src" / "sparsegf2" / "circuits"
    pattern = re.compile(r"^\s*(?:import\s+stim\b|from\s+stim\b)", re.MULTILINE)
    offenders = [
        str(py) for py in pkg_root.rglob("*.py") if pattern.search(py.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"runtime Stim import found in circuits package: {offenders}"
