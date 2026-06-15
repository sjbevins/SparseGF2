"""Crash-only run logging + the error-bar plotting aggregation."""

from __future__ import annotations

import pytest

from sparsegf2.analysis import Study
from sparsegf2.circuits import CircuitConfig
from sparsegf2.errors import InvalidArgumentError

pytest.importorskip("pyarrow")
pytest.importorskip("h5py")


def _base():
    return CircuitConfig(graph_spec="cycle", n=8, picture="purification", p=0.1, depth_factor=4)


def _boom(sim, spec):
    raise RuntimeError("synthetic-failure-xyz")


def test_clean_run_discards_log(tmp_path):
    Study.run(
        tmp_path / "ok",
        _base(),
        grid={"n": [8]},
        seeds=2,
        analyses=["code_dimension"],
        save_tableaux=False,
        log=True,
    )
    # a run that finishes normally keeps no log at all
    assert list((tmp_path / "ok" / "logs").glob("*")) == []


def test_failed_run_keeps_descriptive_log(tmp_path):
    with pytest.raises(RuntimeError, match="synthetic-failure-xyz"):
        Study.run(
            tmp_path / "bad",
            _base(),
            grid={"n": [8], "p": [0.1]},
            seeds=2,
            analyses={"boom": _boom},
            save_tableaux=False,
            log=True,
        )
    failed = list((tmp_path / "bad" / "logs").glob("*FAILED*.log"))
    assert len(failed) == 1
    text = failed[0].read_text(encoding="utf-8")
    assert "cell 1/2" in text  # which cell was running when it died
    assert "synthetic-failure-xyz" in text  # the error
    assert "Traceback" in text  # the full traceback for the postmortem


def test_log_disabled_writes_nothing(tmp_path):
    Study.run(
        tmp_path / "nolog",
        _base(),
        grid={"n": [8]},
        seeds=1,
        analyses=["code_dimension"],
        save_tableaux=False,
        log=False,
    )
    assert not (tmp_path / "nolog" / "logs").exists()


def test_trace_detail_records_every_layer(tmp_path):
    with pytest.raises(RuntimeError):
        Study.run(
            tmp_path / "tr",
            _base(),
            grid={"n": [8]},
            seeds=1,
            analyses={"boom": _boom},
            save_tableaux=False,
            log_detail="trace",
        )
    text = next((tmp_path / "tr" / "logs").glob("*FAILED*.log")).read_text(encoding="utf-8")
    assert "measured layer 1:" in text  # per-layer records only appear at detail="trace"


def test_info_detail_omits_per_layer(tmp_path):
    with pytest.raises(RuntimeError):
        Study.run(
            tmp_path / "info",
            _base(),
            grid={"n": [8]},
            seeds=1,
            analyses={"boom": _boom},
            save_tableaux=False,
            log_detail="info",
        )
    text = next((tmp_path / "info" / "logs").glob("*FAILED*.log")).read_text(encoding="utf-8")
    assert "measured layer" not in text  # the cheap default stays at cell/phase level
    assert "simulate seed=0" in text  # but still names the simulation that failed


def test_invalid_detail_rejected(tmp_path):
    with pytest.raises(InvalidArgumentError):
        Study.run(
            tmp_path / "z",
            _base(),
            grid={"n": [8]},
            seeds=1,
            analyses=["code_dimension"],
            save_tableaux=False,
            log_detail="nope",
        )


def test_aggregate_and_error_styles(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    import matplotlib

    matplotlib.use("Agg")
    from sparsegf2.analysis import aggregate, plot_study

    rows = [
        {"n": n, "p": p, "sample_seed": s, "a.code_dimension": float(n) - 4 * p * s}
        for n in (8, 10)
        for p in (0.1, 0.2)
        for s in range(4)
    ]
    agg = aggregate(rows, "a.code_dimension", xcol="p", group="n", err="sem")
    assert {"n", "p", "mean", "err", "count"}.issubset(agg.columns)
    assert (agg["count"] == 4).all()
    assert (agg["err"] >= 0).all()
    # both uncertainty styles render to a file without error
    plot_study(rows, errorstyle="bars", out=tmp_path / "bars.png")
    plot_study(rows, errorstyle="band", err="std", out=tmp_path / "band.png")
    assert (tmp_path / "bars.png").exists() and (tmp_path / "band.png").exists()
    with pytest.raises(InvalidArgumentError):
        plot_study(rows, errorstyle="nope")
