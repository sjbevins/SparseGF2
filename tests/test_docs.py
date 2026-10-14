"""The Python examples in README.md and docs/GETTING_STARTED.md run, in order, as written.

A python block immediately preceded by the line ``<!-- not run -->`` (blank
lines between them allowed) is shown to the reader but not executed: the
guides use that marker for full-scale code that would take hours. Every other
python block of a document runs in one shared namespace, in order, with the
working directory and ``tempfile``'s default directory moved to pytest's
temporary directory, so whatever the examples write lands there.
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOT_RUN = "<!-- not run -->"
_FENCE = re.compile(r"```python\n(.*?)```", flags=re.S)


def _blocks(path: Path) -> list[str]:
    """The python blocks of ``path`` that are meant to run (not marked ``<!-- not run -->``)."""
    text = path.read_text(encoding="utf-8")
    blocks = []
    for match in _FENCE.finditer(text):
        before = text[: match.start()].rstrip().splitlines()
        if before and before[-1].strip() == NOT_RUN:
            continue
        blocks.append(match.group(1))
    return blocks


@pytest.mark.parametrize("doc", ["README.md", "docs/GETTING_STARTED.md", "docs/HOW_IT_WORKS.md"])
def test_doc_examples_run(doc, tmp_path, monkeypatch):
    blocks = _blocks(ROOT / doc)
    assert blocks, f"no python examples found in {doc}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace: dict = {}
    for i, code in enumerate(blocks):
        try:
            exec(compile(code, f"{doc} block {i + 1}", "exec"), namespace)
        except Exception as exc:  # pragma: no cover - the message is the point
            raise AssertionError(f"{doc}, python block {i + 1} failed:\n{code}") from exc


def test_not_run_marker_is_honoured(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text(
        "```python\nx = 1\n```\n\n<!-- not run -->\n\n```python\nraise SystemExit\n```\n"
        "text\n```python\ny = 2\n```\n",
        encoding="utf-8",
    )
    assert _blocks(doc) == ["x = 1\n", "y = 2\n"]


def test_getting_started_marks_only_full_scale_code_as_not_run():
    text = (ROOT / "docs" / "GETTING_STARTED.md").read_text(encoding="utf-8")
    skipped = [m for m in _FENCE.finditer(text) if text[: m.start()].rstrip().endswith(NOT_RUN)]
    assert 1 <= len(skipped) <= 3
    for match in skipped:
        assert "1000" in match.group(1) or "256" in match.group(1), match.group(1)


def _working_bash() -> bool:
    # On Windows, System32's bash.exe is the WSL launcher, which fails without
    # an installed distribution, so being on PATH is not enough.
    if shutil.which("bash") is None:
        return False
    probe = subprocess.run(["bash", "-c", "exit 0"], capture_output=True)
    return probe.returncode == 0


def test_getting_started_cluster_script_is_valid_bash():
    if not _working_bash():  # pragma: no cover
        pytest.skip("no working bash")
    text = (ROOT / "docs" / "GETTING_STARTED.md").read_text(encoding="utf-8")
    scripts = [b for b in re.findall(r"```bash\n(.*?)```", text, flags=re.S) if "#SBATCH" in b]
    assert scripts, "the SLURM array example is missing"
    for script in scripts:
        proc = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


def test_getting_started_links_resolve():
    path = ROOT / "docs" / "GETTING_STARTED.md"
    targets = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", path.read_text(encoding="utf-8"))
    local = [t for t in targets if not t.startswith(("http://", "https://", "mailto:"))]
    assert local
    missing = [t for t in local if not (path.parent / t).exists()]
    assert not missing, f"broken links in GETTING_STARTED.md: {missing}"


# -- statements corrected in the review stay true -------------------------------------------


def _flat(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_readme_speedup_claims_match_the_benchmark_data():
    sim = json.loads(
        (ROOT / "benchmarks" / "results" / "simulators.json").read_text(encoding="utf-8")
    )
    seconds = {(r["simulator"], r["n"], r["p"]): r["seconds"] for r in sim["runs"]}
    big = [
        seconds["stim", n, p] / seconds["sparsegf2_phase_free", n, p]
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
    before = committed.read_text(encoding="utf-8")
    sim = tmp_path / "sim.json"
    runs = [{"simulator": name, "n": n, "p": 0.16, "depth": 2 * n, "seconds": seconds * n}
            for n in (8, 16)
            for name, seconds in (("stim", 4e-6), ("sparsegf2_phase_free", 1e-6),
                                  ("sparsegf2_phases", 2e-6))]  # fmt: skip
    ensemble = [{"n": n, "p": 0.16, "depth": 2 * n, "threads": 4, "trajectories": 64,
                 "circuits_per_second": 1e5 / n} for n in (8, 16)]  # fmt: skip
    environment = {"cpu_model": "test CPU", "stim": "1.0"}
    sim.write_text(
        json.dumps({"environment": environment, "runs": runs, "ensemble": ensemble}),
        encoding="utf-8",
    )
    missing = str(tmp_path / "missing.json")
    argv = ["make_figures.py", "--simulators", str(sim), "--physics", missing,
            "--outdir", str(tmp_path / "figs")]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(ROOT / "benchmarks" / "make_figures.py"), run_name="__main__")
    assert committed.read_text(encoding="utf-8") == before
    tables = (tmp_path / "figs" / "tables.md").read_text(encoding="utf-8")
    assert "| 16 | 0.032 (2.0×) | 0.016 (4.0×) | 0.064 |" in tables
    assert "| 0.16 | 12,500 | 6,250 |" in tables
    for name in ("runtime", "speedup", "throughput"):
        assert (tmp_path / "figs" / f"{name}-dark.png").exists(), name
