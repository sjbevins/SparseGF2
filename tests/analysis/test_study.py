"""The persisted, augmentable Study database + the augment->plot workflow."""

from __future__ import annotations

import pytest

from sparsegf2.analysis import Study, analyze_tableaux, io
from sparsegf2.analysis.sweep import _task_key
from sparsegf2.circuits import CircuitConfig
from sparsegf2.circuits.picture import setup_picture
from sparsegf2.errors import InvalidArgumentError

pytest.importorskip("pyarrow")
pytest.importorskip("h5py")

BASE = CircuitConfig(graph_spec="cycle", n=12, picture="purification", p=0.2, depth_factor=6)


def test_run_persists_files(tmp_path):
    st = Study.run(tmp_path / "s", BASE, {"p": [0.1, 0.3]}, seeds=3, analyses=["code_dimension"])
    assert (tmp_path / "s" / "rows.parquet").exists()
    assert (tmp_path / "s" / "manifest.json").exists()
    assert (tmp_path / "s" / "tableaux.h5").exists()
    assert len(st) == 6
    assert st.has_tableaux


def test_augment_adds_columns_keyed(tmp_path):
    st = Study.run(
        tmp_path / "s", BASE, {"n": [8, 12], "p": [0.05, 0.3]}, seeds=3, analyses=["code_dimension"]
    )
    assert st.analysis_columns() == ["a.code_dimension"]
    st.augment(["code_rate", "contiguous_distance"])
    assert "a.code_rate" in st.analysis_columns()
    assert "a.contiguous_distance" in st.analysis_columns()

    # augmented values match a direct offline recompute on the saved tableaux
    tabs = io.read_tableaux_hdf5(tmp_path / "s" / "tableaux.h5")
    spec_by_n = {n: setup_picture("purification", n)[1] for n in (8, 12)}
    for r in st.rows:
        key = _task_key({"n": r["n"], "p": r["p"]}, int(r["sample_seed"]))
        direct = analyze_tableaux([tabs[key]], spec_by_n[int(r["n"])], ["code_rate"])[0][
            "code_rate"
        ]
        assert r["a.code_rate"] == direct


def test_augment_idempotent(tmp_path):
    st = Study.run(tmp_path / "s", BASE, {"p": [0.1, 0.3]}, seeds=2, analyses=["code_dimension"])
    st.augment(["code_rate"])
    snapshot = [dict(r) for r in st.rows]
    st.augment(["code_rate"])  # already present -> no recompute
    assert [dict(r) for r in st.rows] == snapshot


def test_augment_force_recomputes(tmp_path):
    st = Study.run(tmp_path / "s", BASE, {"p": [0.2]}, seeds=2, analyses=["code_dimension"])
    st.augment(["code_rate"])
    # corrupt the stored value; force should overwrite it from the tableau
    st.rows[0]["a.code_rate"] = -999.0
    st.augment(["code_rate"], force=True)
    assert st.rows[0]["a.code_rate"] != -999.0


def test_reopen_preserves_augmented_columns(tmp_path):
    Study.run(
        tmp_path / "s", BASE, {"p": [0.1, 0.3]}, seeds=2, analyses=["code_dimension"]
    ).augment(["code_rate"])
    reopened = Study.open(tmp_path / "s")
    assert "a.code_rate" in reopened.analysis_columns()
    assert reopened.manifest["analyses"] == ["code_dimension", "code_rate"]


def test_augment_without_tableaux_raises(tmp_path):
    st = Study.run(
        tmp_path / "s",
        BASE,
        {"p": [0.2]},
        seeds=2,
        analyses=["code_dimension"],
        save_tableaux=False,
    )
    assert not st.has_tableaux
    with pytest.raises(InvalidArgumentError):
        st.augment(["code_rate"])


def test_open_nonstudy_raises(tmp_path):
    with pytest.raises(InvalidArgumentError):
        Study.open(tmp_path)


def test_array_analysis_goes_to_sidecar(tmp_path):
    st = Study.run(tmp_path / "s", BASE, {"p": [0.2]}, seeds=2, analyses=["code_dimension"])
    st.augment(["entropy_profile"])  # array-valued
    assert (tmp_path / "s" / "arrays.npz").exists()
    # no array column leaked into the row table
    assert all("a.entropy_profile" not in r for r in st.rows)


def test_plot_study_auto_detects_columns(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    import matplotlib

    matplotlib.use("Agg")
    from sparsegf2.analysis import plot_study

    st = Study.run(
        tmp_path / "s", BASE, {"p": [0.05, 0.15, 0.3]}, seeds=4, analyses=["code_dimension"]
    )
    st.augment(["code_rate", "contiguous_distance"])
    fig = plot_study(st, out=tmp_path / "p.png")
    titles = {ax.get_title() for ax in fig.axes if ax.get_title()}
    assert "a.code_rate" in titles and "a.contiguous_distance" in titles
    assert (tmp_path / "p.png").exists()


def test_study_run_with_object_grid_values(tmp_path):
    """Grid values that are rich objects (e.g. GraphTopology) must not break the
    manifest write; they are stored JSON-safe, exactly as the rows store them."""
    import json

    from sparsegf2.circuits import from_spec

    g = from_spec("cycle", 8)
    base = CircuitConfig(graph_spec=g, n=8, p=0.3, gating_mode="random_edge", depth_factor=2)
    study = Study.run(tmp_path / "objgrid", base, grid={"graph_spec": [g]}, seeds=2)
    manifest = json.loads((tmp_path / "objgrid" / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(manifest["grid"]["graph_spec"][0], str)
    assert len(study.rows) == 2
