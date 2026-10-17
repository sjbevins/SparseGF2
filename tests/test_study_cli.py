"""``sparsegf2 study``: every command prints what the service layer returns, with the CLI's
exit codes (0 ok, 2 invalid input, 1 a failed verification or an internal error).

Parity: the ``--json`` output of a command equals ``sparsegf2.study.api``'s result for the same
arguments (fields that measure time, and paths of directories made separately, aside); an
invalid input exits with 2 and the API's own message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _study_api_support import MANIFEST, build, without_timing
from sparsegf2 import cli
from sparsegf2.study import api as sa


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build(tmp_path_factory.mktemp("study_cli") / "store")


def run_cli(capsys, *argv: str) -> tuple[int, dict]:
    capsys.readouterr()
    code = cli.main(["study", *argv, "--json"])
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert len(lines) == 1, out  # exactly one JSON object on stdout
    return code, json.loads(lines[0])


def same(a: dict, b: dict, *drop: str) -> None:
    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if k not in drop}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    assert strip(without_timing(a)) == strip(without_timing(json.loads(json.dumps(b))))


def test_reading_commands_print_the_api_result(store, capsys):
    cases = [
        (["catalog"], lambda: sa.catalog()),
        (["catalog", "--observable", "first_passage"], lambda: sa.catalog("first_passage")),
        (["status", store], lambda: sa.status(store)),
        (["plan", store, "--threads", "1"], lambda: sa.plan(store, 1)),
        (["tableau", store, "--n", "6", "--beta", "0.2", "--p", "0.1", "--g", "1", "--c", "0",
          "--t", "5"], lambda: sa.tableau(store, n=6, beta="0.2", p="0.1", g=1, c=0, t=5)),
        (["aggregate", store, "k", "--where", '{"n": 6}', "--paired-rates", "0.1", "0.3"],
         lambda: sa.aggregate(store, "k", where={"n": 6}, paired_rates=["0.1", "0.3"])),
        (["aggregate", store, "first_passage", "--ell", "0", "1", "--where", '{"p": "0.2"}'],
         lambda: sa.aggregate(store, "first_passage", where={"p": "0.2"}, ell=[0, 1])),
        (["aggregate", store, "peak_d_cyc", "--horizon-policy", "include", "--params",
          '{"rule": "exact"}'], lambda: sa.aggregate(store, "peak_d_cyc", {"rule": "exact"},
                                                      horizon_policy="include")),
        (["aggregate", store, "d_cyc", "--field", "zero_extended", "--where", '{"t": [1, 5]}'],
         lambda: sa.aggregate(store, "d_cyc", where={"t": [1, 5]}, field="zero_extended")),
        (["verify", store, "--sample", "2", "--seed", "1", "--threads", "1"],
         lambda: sa.verify(store, 2, 1, num_threads=1)),
    ]  # fmt: skip
    for argv, api_call in cases:
        code, out = run_cli(capsys, *argv)
        assert code == 0, argv
        same(out, api_call())


def test_fss_and_plot_commands_print_the_api_result(store, capsys, tmp_path):
    args = ["--beta", "0.2", "--observable", "peak_d_cyc", "--horizon-policy", "include",
            "--max-horizon-fraction", "1", "--starts", "1", "--bootstrap", "2"]  # fmt: skip
    code, out = run_cli(capsys, "fss", store, *args, "--out", str(tmp_path / "cli"))
    assert code == 0
    expected = sa.fss(store, "peak_d_cyc", "0.2", horizon_policy="include",
                      max_horizon_fraction=1.0, starts=1, bootstrap=2,
                      out=str(tmp_path / "api"))  # fmt: skip
    same(out, expected, "files")
    name = Path(out["files"]["report"]).name
    assert (tmp_path / "cli" / name).read_bytes() == (tmp_path / "api" / name).read_bytes()
    shared = ["--beta", "0.2", "--observables", '[{"id": "peak_d_cyc"}, {"id": "peak_D"}]',
              "--horizon-policy", "include", "--max-horizon-fraction", "1", "--starts", "1",
              "--bootstrap", "2", "--no-leave-one-out"]  # fmt: skip
    code, out = run_cli(capsys, "fss", store, *shared)
    assert code == 0 and out["shared"]["verdict"]
    code, out = run_cli(capsys, "plot", "survival", "--store", store, "--out",
                        str(tmp_path / "figs"), "--options", '{"ell": 1, "where": {"n": 6}}')  # fmt: skip
    assert code == 0 and Path(out["figures"]["svg"]).is_file()
    expected = sa.plot(store, "survival", str(tmp_path / "figs2"), {"ell": 1, "where": {"n": 6}})
    same(out, expected, "figures", "data")
    code, out = run_cli(capsys, "plot", "collapse", "--report",
                        str(tmp_path / "cli" / name), "--out", str(tmp_path / "figs"))  # fmt: skip
    assert code == 0 and out["kind"] == "collapse"


def test_writing_commands_match_the_api_on_twin_stores(tmp_path, capsys):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    manifest_file = tmp_path / "m.json"
    manifest_file.write_text(json.dumps(MANIFEST), encoding="utf-8")
    code, out = run_cli(capsys, "init", str(manifest_file), a)
    assert code == 0
    same(out, sa.init(str(manifest_file), b), "store")
    code, out = run_cli(capsys, "run", a, "--max-units", "3", "--threads", "1")
    assert code == 0 and out["stopped"] == "max_units"
    same(out, sa.run(b, max_units=3, num_threads=1), "store")
    code, out = run_cli(capsys, "run", a, "--threads", "1")
    same(out, sa.run(b, num_threads=1), "store")
    wider = dict(MANIFEST, conditions=[{"n": [6, 8], "beta": ["0.2"],
                                        "p": ["0.1", "0.2", "0.3", "0.4"]}])  # fmt: skip
    (tmp_path / "w.json").write_text(json.dumps(wider), encoding="utf-8")
    code, out = run_cli(capsys, "extend", a, str(tmp_path / "w.json"))
    assert code == 0 and out["revision"] == 2
    same(out, sa.extend(b, str(tmp_path / "w.json")), "store")
    code, out = run_cli(capsys, "compute", a, "half_cut_entropy", "--n", "6", "--p", "0.1")
    assert code == 0
    same(out, sa.compute(b, "half_cut_entropy", n=6, p="0.1"))
    code, out = run_cli(capsys, "aggregate", a, "k", "--where", '{"n": 8, "p": "0.3"}',
                        "--record")  # fmt: skip
    assert code == 0 and out["recorded"] > 0
    same(out, sa.aggregate(b, "k", where={"n": 8, "p": "0.3"}, record=True))
    code, out = run_cli(capsys, "export", a, str(tmp_path / "xa"))
    assert code == 0
    same(out, sa.export(b, str(tmp_path / "xb")), "directory")


@pytest.mark.parametrize(
    "argv, call",
    [
        (["status", "{store}/missing"], lambda s: sa.status(f"{s}/missing")),
        (["aggregate", "{store}", "k", "--where", '{"m": 1}'],
         lambda s: sa.aggregate(s, "k", where={"m": 1})),
        (["aggregate", "{store}", "nope"], lambda s: sa.aggregate(s, "nope")),
        (["aggregate", "{store}", "d_cyc"], lambda s: sa.aggregate(s, "d_cyc")),
        (["tableau", "{store}", "--n", "6", "--beta", "0.2", "--p", "0.1", "--g", "0", "--c", "0",
          "--t", "3"], lambda s: sa.tableau(s, n=6, beta="0.2", p="0.1", g=0, c=0, t=3)),
        (["fss", "{store}", "--beta", "0.2", "--observable", "first_passage"],
         lambda s: sa.fss(s, "first_passage", "0.2")),
        (["catalog", "--observable", "nope"], lambda s: sa.catalog("nope")),
    ],
)  # fmt: skip
def test_invalid_input_exits_2_with_the_api_message(store, capsys, argv, call):
    argv = [a.replace("{store}", store) for a in argv]
    with pytest.raises(ValueError) as info:
        call(store)
    code, out = run_cli(capsys, *argv)
    assert code == cli.EXIT_USAGE
    assert out["error"] == {"type": "ValidationError", "message": str(info.value)}


def test_usage_errors_exit_2(capsys):
    assert cli.main(["study", "tableau", "somewhere", "--json"]) == cli.EXIT_USAGE
    assert "required" in json.loads(capsys.readouterr().out)["error"]["message"]
    assert cli.main(["study", "fss", "somewhere", "--beta", "0.2", "--json"]) == cli.EXIT_USAGE
    assert "exactly one of --observable" in json.loads(capsys.readouterr().out)["error"]["message"]
    assert cli.main(["study", "--help"]) == 0
    assert "sparsegf2 study init" in capsys.readouterr().out


def test_a_failed_verification_exits_1(store, capsys, monkeypatch):
    monkeypatch.setattr(
        sa, "verify", lambda *a, **k: {"ok": False, "trajectories": 1, "failures": ["x: differs"]}
    )
    code, out = run_cli(capsys, "verify", store)
    assert code == cli.EXIT_INTERNAL and out["failures"] == ["x: differs"]


def test_human_output_and_progress(store, capsys, tmp_path):
    assert cli.main(["study", "status", store]) == 0
    captured = capsys.readouterr()
    assert "experiment_id: ws-api-test" in captured.out and "done in" in captured.err
    path = build(tmp_path / "s", run=False)
    assert cli.main(["study", "run", path, "--max-units", "2", "--threads", "1"]) == 0
    captured = capsys.readouterr()
    assert captured.err.count("trajectories in") == 2  # one progress line per unit, on stderr
    assert "stopped: max_units" in captured.out
