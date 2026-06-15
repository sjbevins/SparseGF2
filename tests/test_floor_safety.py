"""Floor-safety: the runtime floor is numpy + numba; optional deps are lazy.

Importing any of the packages must not pull in networkx / pyarrow / h5py /
joblib / pandas / matplotlib / stim. We check this in a *fresh subprocess* so
the result is not contaminated by optional deps that the test session itself
already imported. We also check that every optional-feature code path raises an
actionable ``InvalidArgumentError`` naming the extra when its dependency is
absent.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from sparsegf2.errors import InvalidArgumentError

# scipy is intentionally NOT banned; numba imports it transitively.
_BANNED = ["networkx", "pyarrow", "h5py", "joblib", "pandas", "matplotlib", "stim"]


@pytest.mark.parametrize("pkg", ["sparsegf2", "sparsegf2.circuits", "sparsegf2.analysis"])
def test_import_floor_safe_subprocess(pkg):
    code = (
        f"import sys, {pkg}\n"
        f"banned = {_BANNED!r}\n"
        "leaked = sorted({b for b in banned for k in sys.modules if k == b or k.startswith(b + '.')})\n"
        "assert not leaked, 'leaked optional deps at import: ' + repr(leaked)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_sweep_njobs_missing_joblib(monkeypatch):
    from sparsegf2.analysis import sweep
    from sparsegf2.circuits import CircuitConfig

    monkeypatch.setitem(sys.modules, "joblib", None)
    cfg = CircuitConfig(graph_spec="cycle", n=8, p=0.2, depth_factor=1)
    with pytest.raises(InvalidArgumentError, match=r"parallel"):
        sweep(cfg, seeds=1, n_jobs=2)


def test_write_parquet_missing_pyarrow(monkeypatch, tmp_path):
    from sparsegf2.analysis import io

    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(InvalidArgumentError, match=r"data"):
        io.write_parquet([{"a": 1}], tmp_path / "x.parquet")


def test_write_hdf5_missing_h5py(monkeypatch, tmp_path):
    import numpy as np

    from sparsegf2.analysis import io

    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(InvalidArgumentError, match=r"data"):
        io.write_tableaux_hdf5({"a": np.eye(2, dtype=np.uint8)}, tmp_path / "x.h5")


def test_from_networkx_missing_networkx(monkeypatch):
    from sparsegf2.circuits import from_networkx

    monkeypatch.setitem(sys.modules, "networkx", None)
    with pytest.raises(InvalidArgumentError, match=r"graph"):
        from_networkx(object())


def test_draw_missing_latex(monkeypatch, tmp_path):
    # circuit drawing is LaTeX/quantikz-based: the .tex source / string generate
    # with no compiler, but compiling to pdf/png without pdflatex raises a clear,
    # actionable error rather than failing obscurely.
    import sparsegf2.circuits.draw as d
    from sparsegf2.circuits import CircuitConfig

    monkeypatch.setattr(d, "_which", lambda name: None)
    cfg = CircuitConfig(graph_spec="cycle", n=4, p=0.2, depth_factor=2)
    assert r"\begin{quantikz}" in d.circuit_to_quantikz(cfg, max_layers=2)
    d.save_circuit(cfg, tmp_path / "c.tex", max_layers=2)  # .tex needs no compiler
    assert (tmp_path / "c.tex").exists()
    with pytest.raises(InvalidArgumentError, match=r"LaTeX|pdflatex"):
        d.save_circuit(cfg, tmp_path / "c.pdf", max_layers=2)
