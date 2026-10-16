"""``load_results`` reports the disk use of a saved run: each file and each array, compressed."""

import json
import zipfile

import numpy as np
import pytest

from sparsegf2 import api

SPEC = {
    "num_system": 8,
    "graph": {"family": "cycle"},
    "layers": 12,
    "measurement_rate": 0.2,
    "reference": "single",
    "checkpoints": 4,
    "probes": [
        {"kind": "code_dimension"},
        {"kind": "entropy_profile", "checkpoints": "final"},
        {"kind": "code_snapshot", "checkpoints": "final"},
    ],
}


def members(path) -> list[str]:
    with np.load(path) as z:
        return list(z.files)


def test_a_saved_run_reports_its_files_and_arrays(tmp_path):
    out = tmp_path / "run"
    api.run_ensemble(SPEC, 10, seed=3, output_dir=str(out))
    storage = api.load_results(str(out))["storage"]
    size = (out / "result.npz").stat().st_size
    assert [f["path"] for f in storage["files"]] == ["result.npz"]
    assert storage["files"][0]["megabytes"] == pytest.approx(size / 1e6, rel=1e-5)
    assert storage["total_megabytes"] == pytest.approx(size / 1e6, rel=1e-5)
    names = [a["name"] for a in storage["arrays"]]
    assert names == members(out / "result.npz")
    assert {a["file"] for a in storage["arrays"]} == {"result.npz"}
    # each array is its compressed member of the zip file
    with zipfile.ZipFile(out / "result.npz") as z:
        sizes = {i.filename.removesuffix(".npy"): i.compress_size for i in z.infolist()}
    for a in storage["arrays"]:
        assert a["megabytes"] == pytest.approx(sizes[a["name"]] / 1e6, rel=1e-5, abs=1e-6)
    json.dumps(storage)


def test_a_lone_npz_file(tmp_path):
    out = tmp_path / "run"
    api.run_ensemble(SPEC, 6, seed=3, output_dir=str(out))
    storage = api.load_results(str(out / "result.npz"))["storage"]
    assert [f["path"] for f in storage["files"]] == ["result.npz"]
    assert storage["total_megabytes"] == pytest.approx(
        (out / "result.npz").stat().st_size / 1e6, rel=1e-5
    )


def test_many_chunks_are_summed_by_array_name_and_the_summary_stays_bounded(tmp_path):
    out = tmp_path / "run"
    (out / "chunks").mkdir(parents=True)
    mc = api._build_spec(SPEC)
    for first in range(0, 60, 5):
        result = mc.run(5, seed=3, first_trajectory=first)
        result.save(out / "chunks" / f"traj_{first:012d}.npz")
    files = sorted((out / "chunks").iterdir())
    summary = api.load_results(str(out))
    storage = summary["storage"]
    assert len(json.dumps(summary)) <= api.MAX_OUTPUT_CHARS
    total = sum(f.stat().st_size for f in files)
    assert storage["total_megabytes"] == pytest.approx(total / 1e6, rel=1e-5)
    assert storage["arrays_summed_over_files"] == len(files)
    assert len(storage["files"]) + storage["files_not_shown"] == len(files)
    assert [f["path"] for f in storage["files"]][0] == "chunks/traj_000000000000.npz"
    # one entry per array name, summed over the chunks
    assert [a["name"] for a in storage["arrays"]] == members(files[0])
    assert all("file" not in a for a in storage["arrays"])
    expected = 0.0
    for f in files:
        with zipfile.ZipFile(f) as z:
            expected += z.getinfo("value::code_dimension.npy").compress_size
    (row,) = [a for a in storage["arrays"] if a["name"] == "value::code_dimension"]
    assert row["megabytes"] == pytest.approx(expected / 1e6, rel=1e-5, abs=1e-6)


def test_the_arrays_listed_are_bounded(tmp_path):
    probes = [{"kind": "code_dimension", "name": f"k{i}"} for i in range(40)]
    out = tmp_path / "run"
    api.run_ensemble(dict(SPEC, probes=probes), 4, seed=1, output_dir=str(out))
    storage = api.load_results(str(out))["storage"]
    size = (out / "result.npz").stat().st_size
    assert len(storage["arrays"]) <= api.MAX_STORAGE_ARRAYS
    assert storage["arrays"][-1]["name"] == "(other arrays)"
    assert storage["total_megabytes"] == pytest.approx(size / 1e6, rel=1e-5)
    assert len(json.dumps(storage)) < api.MAX_OUTPUT_CHARS // 4
