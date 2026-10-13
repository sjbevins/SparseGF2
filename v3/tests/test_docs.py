"""The Python examples in README.md run, in order, as written."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(), flags=re.S)


@pytest.mark.parametrize("doc", ["README.md"])
def test_readme_examples_run(doc):
    blocks = _blocks(ROOT / doc)
    assert blocks, f"no python examples found in {doc}"
    namespace: dict = {}
    for i, code in enumerate(blocks):
        try:
            exec(compile(code, f"{doc} block {i + 1}", "exec"), namespace)
        except Exception as exc:  # pragma: no cover - the message is the point
            raise AssertionError(f"{doc}, python block {i + 1} failed:\n{code}") from exc


# -- statements corrected in the review stay true -------------------------------------------


def _flat(path: Path) -> str:
    return " ".join(path.read_text().split())


def test_readme_speedup_claims_match_the_benchmark_data():
    sim = json.loads((ROOT / "benchmarks" / "results" / "simulators.json").read_text())
    seconds = {(r["simulator"], r["n"], r["p"]): r["seconds"] for r in sim["runs"]}
    big = [
        seconds["stim", n, p] / seconds["sparsegf2_v3_phase_free", n, p]
        for name, n, p in seconds
        if name == "stim" and n in (512, 1024)
    ]
    readme = _flat(ROOT / "README.md")
    assert f"{round(min(big))}–{round(max(big))}× at n = 512–1024" in readme
    assert "Simulator.run (phase-free) on one thread" in readme  # the throughput figure


def test_design_and_literature_state_the_real_limits():
    design = _flat(ROOT / "docs" / "DESIGN.md")
    assert "costs milliseconds" not in design
    assert "a Python list of immutable `Instruction` objects" in design
    assert "`coins` covers measurement results only" in design
    assert "ascending qubit order" in design  # reference sample vs Stim's
    literature = _flat(ROOT / "docs" / "LITERATURE.md")
    assert "τ grows as O(log L)" in literature


def test_make_figures_outdir_leaves_the_committed_tables_alone(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import runpy
    import sys

    committed = ROOT / "benchmarks" / "results" / "tables.md"
    before = committed.read_text()
    sim = tmp_path / "sim.json"
    runs = [{"simulator": name, "n": n, "p": 0.16, "depth": 2 * n, "seconds": seconds * n}
            for n in (8, 16)
            for name, seconds in (("stim", 4e-6), ("sparsegf2_v3_phase_free", 1e-6),
                                  ("sparsegf2_v3_phases", 2e-6))]  # fmt: skip
    ensemble = [{"n": n, "p": 0.16, "depth": 2 * n, "threads": 4, "trajectories": 64,
                 "circuits_per_second": 1e5 / n} for n in (8, 16)]  # fmt: skip
    environment = {"cpu_model": "test CPU", "stim": "1.0"}
    sim.write_text(json.dumps({"environment": environment, "runs": runs, "ensemble": ensemble}))
    missing = str(tmp_path / "missing.json")
    argv = ["make_figures.py", "--simulators", str(sim), "--physics", missing,
            "--outdir", str(tmp_path / "figs")]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(ROOT / "benchmarks" / "make_figures.py"), run_name="__main__")
    assert committed.read_text() == before
    tables = (tmp_path / "figs" / "tables.md").read_text()
    assert "| 16 | 0.032 (2.0×) | 0.016 (4.0×) | 0.064 |" in tables
    assert "| 0.16 | 12,500 | 6,250 |" in tables
    for name in ("runtime", "speedup", "throughput"):
        assert (tmp_path / "figs" / f"{name}-dark.png").exists(), name
