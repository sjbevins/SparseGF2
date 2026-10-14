"""Param, ToolDef, Bound: schemas, validation, coercion, flatness."""

import json

import pytest

from sparsegf2.agent.registry import BACKGROUND, MISSING, Bound, Param, ToolDef, check_flat


def _tool(**kw):
    params = (
        Param("num_system", "integer", "System qubits.", required=True, minimum=2),
        Param("measurement_rate", "number", "Rate p.", default=0.1, minimum=0, maximum=1),
        Param(
            "schedule",
            "string",
            "Gate placement.",
            default="matching",
            enum=("matching", "random_edges"),
        ),
        Param("probes", "array", "Probe calls.", items="string"),
        Param("phases", "boolean", "Track signs.", default=True),
    )
    base = dict(
        name="ensemble_run",
        stack="ensemble",
        summary="Run.",
        description="Run it.",
        params=params,
        handler=lambda ctx, a: a,
        render=lambda d, ctx: "ensemble_run",
    )
    base.update(kw)
    return ToolDef(**base)


def test_schema_is_flat_and_marks_required():
    schema = Bound(_tool()).input_schema()
    assert schema["required"] == ["num_system"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["probes"] == {
        "type": "array",
        "description": "Probe calls.",
        "items": {"type": "string"},
    }
    assert schema["properties"]["schedule"]["enum"] == ["matching", "random_edges"]
    assert check_flat(schema) == []
    assert check_flat({"type": "object", "properties": {"x": {"anyOf": []}}}) != []


def test_long_running_tools_get_background():
    t = _tool(long_running=True)
    assert t.param("background") == BACKGROUND
    assert BACKGROUND.default == "auto"


@pytest.mark.parametrize(
    "given, expected",
    [({"num_system": "64"}, 64), ({"num_system": 64.0}, 64), ({"num_system": 64}, 64)],
)
def test_integers_are_coerced_when_unambiguous(given, expected):
    assert Bound(_tool()).validate(given)["num_system"] == expected


def test_numbers_and_booleans_from_text():
    out = Bound(_tool()).validate({"num_system": 8, "measurement_rate": "0.25", "phases": "false"})
    assert out["measurement_rate"] == 0.25 and out["phases"] is False


@pytest.mark.parametrize(
    "given, message",
    [
        ({"num_system": "sixty"}, "num_system must be an integer"),
        ({"num_system": 64.5}, "num_system must be an integer"),
        ({"num_system": True}, "num_system must be a number"),
        ({"num_system": 1}, "num_system must be at least 2"),
        ({"num_system": 8, "measurement_rate": 2}, "at most 1"),
        ({"num_system": 8, "schedule": "brick"}, "must be one of"),
        ({"num_system": 8, "layrs": 3}, r"unknown argument\(s\) \['layrs'\]"),
        ({}, "missing required argument 'num_system'"),
    ],
)
def test_errors_name_the_parameter(given, message):
    with pytest.raises(ValueError, match=message):
        Bound(_tool()).validate(given)


def test_defaults_fill_and_none_means_omitted():
    out = Bound(_tool()).validate({"num_system": 8, "probes": None})
    assert out == {"num_system": 8, "measurement_rate": 0.1, "schedule": "matching", "phases": True}


def test_single_value_for_an_array_becomes_a_list():
    assert Bound(_tool()).validate({"num_system": 8, "probes": "code_dimension"})["probes"] == [
        "code_dimension"
    ]


def test_fixed_values_leave_the_schema_and_are_injected():
    b = Bound(_tool(), fixed={"num_system": 128}, defaults={"measurement_rate": 0.3})
    schema = b.input_schema()
    assert "num_system" not in schema["properties"] and "required" not in schema
    assert schema["properties"]["measurement_rate"]["default"] == 0.3
    assert b.validate({})["num_system"] == 128
    assert b.validate({"num_system": 128})["num_system"] == 128
    with pytest.raises(ValueError, match="fixed to 128 by this tool stack"):
        b.validate({"num_system": 64})


def test_size_counts_the_listing():
    b = Bound(_tool())
    assert b.size() == len(json.dumps(b.listing(), separators=(",", ":")))


def test_tool_names_carry_their_stack():
    with pytest.raises(ValueError, match="must start with 'ensemble_'"):
        _tool(name="run_ensemble")


def test_missing_sentinel_is_not_none():
    assert MISSING is not None and repr(MISSING) == "MISSING"
