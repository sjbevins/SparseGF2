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
    # every regular file counts, the largest first
    everything = [f for f in out.rglob("*") if f.is_file()]
    assert {f["path"] for f in storage["files"]} == {
        f.relative_to(out).as_posix() for f in everything
    }
    assert {"result.npz", "spec.json", "provenance.json"} <= {f["path"] for f in storage["files"]}
    assert storage["files"][0]["path"] == "result.npz"
    assert storage["files"][0]["megabytes"] == pytest.approx(size / 1e6, rel=1e-5)
    assert storage["total_megabytes"] == pytest.approx(
        sum(f.stat().st_size for f in everything) / 1e6, rel=1e-5
    )
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
    everything = [f for f in out.rglob("*") if f.is_file()]
    total = sum(f.stat().st_size for f in everything)
    assert storage["total_megabytes"] == pytest.approx(total / 1e6, rel=1e-5)
    assert storage["arrays_summed_over_files"] == len(files)
    assert len(storage["files"]) + storage["files_not_shown"] == len(everything)
    sizes = [f["megabytes"] for f in storage["files"]]
    assert sizes == sorted(sizes, reverse=True)  # the largest are listed
    # one entry per array name, summed over the chunks
    assert [a["name"] for a in storage["arrays"]] == members(files[0])
    assert all("file" not in a for a in storage["arrays"])
    expected = 0.0
    for f in files:
        with zipfile.ZipFile(f) as z:
            expected += z.getinfo("value::code_dimension.npy").compress_size
    (row,) = [a for a in storage["arrays"] if a["name"] == "value::code_dimension"]
    assert row["megabytes"] == pytest.approx(expected / 1e6, rel=1e-5, abs=1e-6)


def test_long_probe_names_are_abbreviated_and_the_summary_is_not_cut(tmp_path):
    probes = [{"kind": "entropy", "region": list(range(i, i + 40))} for i in range(0, 20)]
    spec = dict(SPEC, num_system=64, probes=probes, checkpoints="final")
    out = tmp_path / "run"
    api.run_ensemble(spec, 2, seed=1, output_dir=str(out))
    summary = api.load_results(str(out))
    storage = summary["storage"]
    assert len(json.dumps(summary)) <= api.MAX_OUTPUT_CHARS
    # the stored spec of 20 long probes is itself too long to show (the summary's own rule);
    # nothing else is cut, and storage is shown whole
    assert [n for n in summary["notes"] if n.startswith("not shown")] == [
        "not shown: spec (too long)"
    ]
    assert len(storage["arrays"]) == api.MAX_STORAGE_ARRAYS
    names = [a["name"] for a in storage["arrays"]]
    assert len(set(names)) == len(names)
    assert all(len(n) <= api.MAX_NAME_CHARS + len("value::") for n in names)


def test_the_arrays_listed_are_bounded(tmp_path):
    probes = [{"kind": "code_dimension", "name": f"k{i}"} for i in range(40)]
    out = tmp_path / "run"
    api.run_ensemble(dict(SPEC, probes=probes), 4, seed=1, output_dir=str(out))
    storage = api.load_results(str(out))["storage"]
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    assert len(storage["arrays"]) <= api.MAX_STORAGE_ARRAYS
    assert storage["arrays"][-1]["name"] == "(other arrays)"
    assert storage["total_megabytes"] == pytest.approx(size / 1e6, rel=1e-5)
    assert len(json.dumps(storage)) < api.MAX_OUTPUT_CHARS // 4


def test_rebuildable_arrays_are_not_stored_and_load_bitwise(tmp_path):
    """A symbolic checkpoint rule ("every_layer", "final", a stride) and the default gate set are
    rebuilt by load, so a file stores neither (every layer of T = 10^5 would be 800 kB per
    file); the loaded result is the same, bitwise. Explicit lists and other gate sets are
    stored, and a file that holds the arrays (written before) loads as before."""
    from sparsegf2 import graphs, mipt

    def spec(checkpoints, gate_set=None):
        return mipt.MonitoredCircuit(
            num_system=6, edges=graphs.cycle(6), layers=100_000, measurement_rate=1e-4,
            reference="purification", checkpoints=checkpoints, gate_set=gate_set,
            probes=[mipt.code_dimension(checkpoints=[0, 10, 100_000])],
        )  # fmt: skip

    def same(a: mipt.EnsembleResult, b: mipt.EnsembleResult) -> None:
        assert mipt._protocol_difference(a.spec, b.spec) is None
        for name in ("checkpoint_layers", "trajectories", "final_layer", "first_passage"):
            x, y = getattr(a, name), getattr(b, name)
            assert x.dtype == y.dtype and np.array_equal(x, y), name
        assert np.array_equal(a.spec.gate_set, b.spec.gate_set)
        assert np.array_equal(a.values["code_dimension"], b.values["code_dimension"])

    for checkpoints, stored in (("every_layer", False), (1000, False), ("final", False),
                                ([0, 5, 100_000], True)):  # fmt: skip
        res = spec(checkpoints).run(3, seed=1)
        path = tmp_path / f"{checkpoints}.npz"
        res.save(path, compression="fast")
        names = members(path)
        assert "gate_set.npy" not in names and "gate_set" not in names
        assert ("checkpoint_layers" in names) == stored
        back = mipt.EnsembleResult.load(path)
        same(back, res)
        assert back.checkpoint_layers.flags.writeable  # as an array read from a file
        if checkpoints == "every_layer":
            assert path.stat().st_size < 50_000  # 100,001 layers are not in the file
            # a file written before, with both arrays, loads to the same result
            old = tmp_path / "old.npz"
            with zipfile.ZipFile(path) as z, zipfile.ZipFile(old, "w") as w:
                for info in z.infolist():
                    w.writestr(info, z.read(info))
                for name, value in (("gate_set", res.spec.gate_set),
                                    ("checkpoint_layers", res.checkpoint_layers)):  # fmt: skip
                    with w.open(f"{name}.npy", "w") as fh:
                        np.lib.format.write_array(fh, np.asarray(value))
            same(mipt.EnsembleResult.load(old), res)
    # another gate set is stored and read back
    res = spec("final", gate_set=[0, 17, 400]).run(2, seed=1)
    res.save(tmp_path / "gates.npz")
    assert "gate_set" in members(tmp_path / "gates.npz")
    same(mipt.EnsembleResult.load(tmp_path / "gates.npz"), res)
