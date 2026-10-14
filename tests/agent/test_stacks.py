"""Stack resolution, custom stack files, and the stacks CLI."""

import json

import pytest

from sparsegf2.agent import stacks
from sparsegf2.agent.registry import Param, ToolDef


def _t(name, stack, long_running=False, params=(), requires=()):
    return ToolDef(
        name,
        stack,
        "s",
        "d",
        tuple(params),
        lambda c, a: {},
        lambda d, c: name,
        long_running=long_running,
        requires=requires,
    )


@pytest.fixture
def reg():
    p = (
        Param("num_system", "integer", "n", required=True),
        Param("graph", "string", "g", default="cycle", enum=("cycle", "watts_strogatz")),
    )
    tools = [
        _t("help_topic", "help"),
        _t("ensemble_run", "ensemble", True, p),
        _t("traj_run", "traj", True, p),
        _t("job_status", "job"),
        _t("job_cancel", "job"),
        _t("plot_make", "plot", requires=("module_that_does_not_exist_xyz",)),
    ]
    return {t.name: t for t in tools}


def test_builtin_stack_and_all_bundle(tmp_path, reg):
    r = stacks.resolve("job", root=tmp_path, registry=reg)
    assert [b.name for b in r.bounds] == ["job_status", "job_cancel"]
    r = stacks.resolve("all", root=tmp_path, registry=reg)
    assert "plot_make" not in [b.name for b in r.bounds]
    assert any("module_that_does_not_exist_xyz" in n for n in r.notes)


def test_long_running_tools_bring_job_status(tmp_path, reg):
    r = stacks.resolve("ensemble", root=tmp_path, registry=reg)
    assert [b.name for b in r.bounds] == ["ensemble_run", "job_status"]


def test_custom_stack_include_exclude_fixed_defaults(tmp_path, reg):
    f = tmp_path / ".sparsegf2" / "stacks.json"
    f.parent.mkdir()
    f.write_text(
        json.dumps(
            {
                "version": 1,
                "stacks": {
                    "mine": {
                        "description": "x",
                        "include": ["ensemble", "traj_run", "help"],
                        "exclude": ["help_topic"],
                        "fixed": {"*": {"num_system": 64}},
                        "defaults": {"ensemble_run": {"graph": "watts_strogatz"}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    r = stacks.resolve("mine", root=tmp_path, registry=reg)
    by = {b.name: b for b in r.bounds}
    assert set(by) == {"ensemble_run", "traj_run", "job_status"}
    assert by["traj_run"].fixed == {"num_system": 64}
    assert by["ensemble_run"].defaults == {"graph": "watts_strogatz"}
    assert "num_system" not in by["ensemble_run"].input_schema()["properties"]


@pytest.mark.parametrize(
    "stack, message",
    [
        ({"include": ["nope"]}, "unknown stack or tool 'nope'"),
        (
            {"include": ["ensemble"], "fixed": {"ensemble_run": {"layrs": 3}}},
            "no parameter 'layrs'",
        ),
        (
            {"include": ["ensemble"], "fixed": {"ensemble_run": {"graph": "ring"}}},
            "must be one of",
        ),
        (
            {"include": ["ensemble"], "fixed": {"traj_run": {"num_system": 3}}},
            "not in this stack",
        ),
        ({"include": ["mine"]}, "includes itself"),
    ],
)
def test_validation_errors_name_the_stack(tmp_path, reg, stack, message):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"version": 1, "stacks": {"mine": stack}}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        stacks.resolve("mine", root=tmp_path, stacks_file=f, registry=reg)


def test_custom_names_may_not_shadow_builtins(tmp_path, reg):
    f = tmp_path / "s.json"
    f.write_text(
        json.dumps({"version": 1, "stacks": {"core": {"include": ["job"]}}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="'core' is a built-in name"):
        stacks.resolve("job", root=tmp_path, stacks_file=f, registry=reg)


def test_unknown_stack_suggests(tmp_path, reg):
    with pytest.raises(ValueError, match=r"unknown stack 'jobs'.*job"):
        stacks.resolve("jobs", root=tmp_path, registry=reg)


def test_save_and_delete_are_atomic_round_trips(tmp_path):
    f = tmp_path / "s.json"
    stacks.save_stack(f, stacks.StackDef("a", include=("job",)))
    stacks.save_stack(f, stacks.StackDef("b", include=("help",), fixed={"*": {"x": 1}}))
    data = json.loads(f.read_text(encoding="utf-8"))
    assert set(data["stacks"]) == {"a", "b"} and data["version"] == 1
    assert stacks.delete_stack(f, "a") and not stacks.delete_stack(f, "a")


def test_presets_file_is_well_formed():
    p = stacks.presets()
    assert set(p) == {"erasure", "purification", "entanglement", "circuits"}
    assert p["erasure"].defaults["*"]["reference"] == "purification"
    assert (
        p["erasure"].defaults["ensemble_run"]["probes"][1] == "erasure_first_failure(samples=512)"
    )


def test_cli_save_show_delete(tmp_path, capsys):
    from sparsegf2 import cli

    f = str(tmp_path / "s.json")
    assert (
        cli.main(
            [
                "stacks",
                "save",
                "mine",
                "--include",
                "job,help",
                "--stacks-file",
                f,
                "--root",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    assert (
        cli.main(["stacks", "show", "mine", "--stacks-file", f, "--root", str(tmp_path), "--json"])
        == 0
    )
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert "job_status" in json.loads(out)["tools"]
    assert cli.main(["stacks", "delete", "mine", "--stacks-file", f, "--json"]) == 0


@pytest.mark.parametrize(
    "stack, message",
    [
        ({"include": ["ensemble"], "exclude": ["nope_tool"]}, r"exclude names 'nope_tool'"),
        ({"include": "job"}, r"stack 'mine': include must be a list of names"),
        ({"include": ["job"], "fixed": {"ensemble_run": 3}}, r"fixed must be an object"),
    ],
)
def test_field_type_and_exclude_validation(tmp_path, reg, stack, message):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"version": 1, "stacks": {"mine": stack}}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        stacks.resolve("mine", root=tmp_path, stacks_file=f, registry=reg)
