"""The example studies run end to end (tiny sizes) and save loadable results."""

import runpy
import sys
from pathlib import Path

import pytest

from sparsegf2_v3 import mipt

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _run(script: str, args: list[str], monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [script, *args])
    runpy.run_path(str(EXAMPLES / script), run_name="__main__")


def test_purification_time(tmp_path, monkeypatch):
    out = tmp_path / "pt.npz"
    _run(
        "purification_time.py",
        ["--n", "12", "--trajectories", "20", "--out", str(out)],
        monkeypatch,
    )
    res = mipt.EnsembleResult.load(out)
    assert res.spec.num_graphs == 20 and res.spec.scramble_matchings == 32
    assert res.spec.reference_partner == 11 and res.purification_layer.shape == (20,)


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


@pytest.mark.parametrize("graph", ["ring", "ws"])
def test_code_and_erasure(tmp_path, monkeypatch, graph):
    out = tmp_path / "code.npz"
    args = ["--n", "12", "--layers", "8", "--every", "4", "--trajectories", "5", "--graph", graph]
    _run("code_and_erasure.py", [*args, "--out", str(out)], monkeypatch)
    res = mipt.EnsembleResult.load(out)
    assert list(res.checkpoint_layers) == [0, 4, 8]
