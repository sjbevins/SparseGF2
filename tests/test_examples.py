"""The example studies run end to end (tiny sizes) and save loadable results."""

import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from sparsegf2 import api, mipt

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _run(script: str, args: list[str], monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [script, *args])
    runpy.run_path(str(EXAMPLES / script), run_name="__main__")


def test_purification_time(tmp_path, monkeypatch):
    out = tmp_path / "pt.npz"
    _run(
        "purification_time.py",
        ["--n", "12", "--graphs", "10", "--circuits-per-graph", "2", "--out", str(out)],
        monkeypatch,
    )
    res = mipt.EnsembleResult.load(out)
    assert res.spec.num_graphs == 10 and res.spec.scramble_matchings == 32
    assert res.spec.reference_partner == 11 and res.purification_layer.shape == (20,)
    assert res.graph_index.tolist() == [g for g in range(10) for _ in range(2)]


def test_purification_time_mostly_censored_point(tmp_path, monkeypatch, capsys):
    # In the mixed phase fewer than half the trajectories purify within T_max:
    # the Kaplan-Meier median does not exist, and the run must still be saved.
    out = tmp_path / "mixed.npz"
    _run(
        "purification_time.py",
        ["--n", "12", "--p", "0.06", "--graphs", "40", "--out", str(out)],
        monkeypatch,
    )
    printed = capsys.readouterr().out
    res = mipt.EnsembleResult.load(out)
    purified = int((res.purification_layer >= 0).sum())
    assert 0 < purified < 20
    assert f"observed purifications: {purified} / 40" in printed
    assert "> last checkpoint (96) (more than half censored)" in printed


def test_purification_time_median_needs_a_strict_majority():
    median_tau = runpy.run_path(str(EXAMPLES / "purification_time.py"))["median_tau"]
    spec = {"num_system": 8, "graph": {"family": "cycle"}, "layers": 12, "measurement_rate": 0.2,
            "reference": "single", "checkpoints": "every_layer"}  # fmt: skip
    half = api._build_spec(spec).run(4, seed=1)
    assert half.purification_layer.tolist() == [10, -1, 4, -1]  # exactly half purified
    assert median_tau(half) == ">= 10 layers (not determined: only half purified)"
    # 13 of 20 purified; survival is exactly 1/2 from layer 11 to 21, then below
    spec.update(num_system=12, layers=40, measurement_rate=0.15, stop_when_purified=True)
    most = api._build_spec(spec).run(20, seed=1)
    tau = np.sort(np.where(most.purification_layer >= 0, most.purification_layer, np.inf))
    assert tau[9] == 11 and tau[10] == 22
    assert median_tau(most) == "11 layers"


def test_transition_scan(tmp_path, monkeypatch):
    out = tmp_path / "scan.json"
    args = [
        "--sizes",
        "8",
        "--rates",
        "0.1",
        "0.2",
        "0.1",
        "--trajectories",
        "5",
        "--out",
        str(out),
    ]
    _run("transition_scan.py", args, monkeypatch)
    assert out.exists()


def test_correlated_erasure(tmp_path, monkeypatch, capsys):
    args = ["--n", "12", "--trajectories", "8", "--samples", "40", "--couplings", "0", "2"]
    _run("correlated_erasure.py", [*args, "--out-dir", str(tmp_path)], monkeypatch)
    printed = capsys.readouterr().out
    for name, method in (("ring", "exact"), ("small_world", "chain")):
        res = mipt.EnsembleResult.load(tmp_path / f"{name}.npz")
        couplings, sizes, values = res.correlated()
        assert couplings.tolist() == [0.0, 2.0] and values.shape == (8, 1, 2, sizes.size)
        assert res.spec.correlated_sampler("correlated_erasure")["method"] == method
        assert np.all((values >= 0) & (values <= 1))
        assert f"== {name}: n = 12" in printed
    assert printed.count("inf (exact)") == 2  # the J -> inf limit on both graphs
    assert "mixing check, m = " in printed and "exact P_J(m), m <= 3" in printed


@pytest.mark.parametrize("p", [0.02, 0.4])  # codes survive / mostly purify
def test_code_and_erasure(tmp_path, monkeypatch, capsys, p):
    out = tmp_path / "code.npz"
    args = ["--n", "12", "--p", str(p), "--layers", "8", "--every", "4", "--graphs", "3"]
    args += ["--circuits-per-graph", "2", "--samples", "10"]
    _run("code_and_erasure.py", [*args, "--out", str(out)], monkeypatch)
    res = mipt.EnsembleResult.load(out)
    assert list(res.checkpoint_layers) == [0, 4, 8] and res.num_trajectories == 6
    assert res.values["erasure_first_failure"].shape == (6, 3, 10)
    assert res.layers("code_snapshot").tolist() == [8]
    assert "Phi(0.1)|k>0   e_0.1|k>0" in capsys.readouterr().out
