"""Context paths (root confinement, Windows forms) and the rendering helpers."""

import json
import math
import os
import sys

import numpy as np
import pytest

from sparsegf2.agent.context import Context
from sparsegf2.agent.render import cap, curve, fmt, jsonable, pm, table


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "runs").mkdir()
    return Context(tmp_path)


def test_relative_and_absolute_paths_inside_the_root(ctx, tmp_path):
    assert ctx.path("runs/a") == (tmp_path / "runs" / "a").resolve()
    assert ctx.path(str(tmp_path / "runs")) == (tmp_path / "runs").resolve()
    assert ctx.path("runs/../runs/b") == (tmp_path / "runs" / "b").resolve()


def test_escapes_are_refused(ctx, tmp_path):
    with pytest.raises(ValueError, match="outside the root"):
        ctx.path("../elsewhere")
    with pytest.raises(ValueError, match="outside the root"):
        ctx.path(str(tmp_path.parent / "x"))
    with pytest.raises(ValueError, match="non-empty"):
        ctx.path("  ")


def test_must_exist(ctx):
    with pytest.raises(ValueError, match="does not exist"):
        ctx.path("nope.npz", must_exist=True)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path forms")
def test_windows_backslashes_and_drive_letter_case(ctx, tmp_path):
    assert ctx.path("runs\\c") == (tmp_path / "runs" / "c").resolve()
    other_case = str(tmp_path).upper() if str(tmp_path).islower() else str(tmp_path).lower()
    assert ctx.path(os.path.join(other_case, "runs")).name.lower() == "runs"


def test_show_is_relative_posix(ctx, tmp_path):
    assert ctx.show(tmp_path / "runs" / "x") == "runs/x"
    assert ctx.show(tmp_path) == "."


def test_default_output_dir_is_a_stable_hash(ctx, tmp_path):
    a = ctx.output_dir("ensemble_run", {"num_system": 8, "background": "auto"})
    b = ctx.output_dir("ensemble_run", {"num_system": 8, "background": "never"})
    c = ctx.output_dir("ensemble_run", {"num_system": 9})
    assert a == b != c and a.parent == (tmp_path / "runs").resolve()
    assert ctx.output_dir("x_y", {"output_dir": "mine"}) == (tmp_path / "mine").resolve()


def test_jsonable_never_emits_nan():
    data = jsonable({"a": np.array([1.0, np.nan, np.inf]), "b": math.nan, "c": (1, 2)})
    assert data == {"a": [1.0, None, None], "b": None, "c": [1, 2]}
    json.dumps(data, allow_nan=False)


def test_numbers():
    assert fmt(None) == "-" and fmt(math.nan) == "nan" and fmt(3) == "3"
    assert fmt(0.123456) == "0.1235" and fmt(True) == "true"
    assert pm(0.5, 0.01) == "0.5+-0.01" and pm(0.5, math.nan) == "0.5"


def test_table_aligns_columns():
    out = table(["layer", "k/n"], [[0, 1.0], [256, 0.0833]])
    assert out.splitlines() == ["layer     k/n", "    0       1", "  256  0.0833"]


def test_cap_cuts_at_a_line_and_says_where_the_rest_is():
    text = "\n".join(f"line {i}" for i in range(100))
    out = cap(text, 120, full="runs/x")
    assert len(out) <= 120 and out.endswith("(cut to 120 characters; full data in runs/x)")


def test_curve_subsamples():
    assert curve([0.0, 0.5, 1.0]) == "m=0..2: 0 0.5 1"
    long = curve(np.linspace(0, 1, 101))
    assert long.startswith("m=0..100 (every ") and len(long.split(": ")[1].split()) <= 17


def test_cap_never_exceeds_a_tiny_limit():
    for limit in (20, 3, 0, -1):
        out = cap("x" * 500, limit, full="runs/x")
        assert len(out) <= max(limit, 0)
    assert cap("x" * 500, 20) == "x" * 20
    assert cap("x" * 500, 0) == ""
