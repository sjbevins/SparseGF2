"""The call grammar: every construct, every error with a column, and fuzzing."""

import contextlib
import random
import string

import pytest

from sparsegf2.agent.grammar import (
    GrammarError,
    Range,
    bind,
    parse_call,
    parse_calls,
    parse_checkpoints,
    parse_grid,
    parse_qubits,
    plain,
)


def test_bare_name():
    c = parse_call("code_dimension")
    assert (c.name, c.args, c.kwargs, c.at) == ("code_dimension", (), (), None)


def test_positional_keyword_and_checkpoints():
    c = parse_call("mutual_information(0:4, b=[8, 9, 10])@every_layer", allow_at=True)
    assert c.args == (Range(0, 4),)
    assert c.kwargs == (("b", [8, 9, 10]),)
    assert c.at == "every_layer"


@pytest.mark.parametrize(
    "text, at",
    [("p@final", "final"), ("p@16", 16), ("p@[0, 64, 128]", [0, 64, 128])],
)
def test_checkpoint_forms(text, at):
    assert parse_call(text, allow_at=True).at == at


def test_values():
    c = parse_call('f(1, -2, 0.5, 1e-3, true, False, none, Z0*Z1, !X3, +XZ_, "a b", 0:16:2, [])')
    assert c.args == (
        1,
        -2,
        0.5,
        1e-3,
        True,
        False,
        None,
        "Z0*Z1",
        "!X3",
        "+XZ_",
        "a b",
        Range(0, 16, 2),
        [],
    )


def test_bind_maps_positionals_in_order_and_checks_names():
    c = parse_call("erasure_first_failure(512, name=ff)")
    assert bind(c, ["samples"], what="probes[0]", keyword_only=["name"]) == {
        "samples": 512,
        "name": "ff",
    }
    with pytest.raises(GrammarError, match=r"unknown argument 'regon'.*takes \(region\)"):
        bind(parse_call("entropy(regon=0:8)"), ["region"], what="probes[1]")
    with pytest.raises(GrammarError, match="at most 1 positional"):
        bind(parse_call("entropy(0:4, 4:8)"), ["region"], what="x")
    with pytest.raises(GrammarError, match="given twice"):
        bind(parse_call("entropy(0:4, region=4:8)"), ["region"], what="x")


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "expected a name"),
        ("entropy(0:8", r"expected ',' or '\)'"),
        ("entropy(0:8,)", "expected a value"),
        ("f(a=1, 2)", "positional argument after a keyword"),
        ("p@", "expected final, every_layer"),
        ("p@0.5", "expected final, every_layer"),
        ("p) x", "unexpected text"),
        ("f(0:4:0)", "step must not be 0"),
        ("β(1)", "unexpected character"),
        ("f([1, 2)", r"expected ',' or '\]'"),
    ],
)
def test_errors_carry_a_column(text, message):
    with pytest.raises(GrammarError, match=message) as info:
        parse_call(text, allow_at=True)
    assert "column" in str(info.value)


def test_at_is_refused_where_not_allowed():
    with pytest.raises(GrammarError, match="not allowed here"):
        parse_call("entropy(0:4)@final")


def test_parse_calls_names_the_item():
    with pytest.raises(GrammarError, match=r"probes\[1\]"):
        parse_calls(["code_dimension", "entropy("], allow_at=True, what="probes")
    with pytest.raises(GrammarError, match="list of strings"):
        parse_calls("code_dimension", what="probes")


@pytest.mark.parametrize(
    "value, expected",
    [
        ("final", "final"),
        ("every_layer", "every_layer"),
        ("16", 16),
        (16, 16),
        ("[0, 8, 16]", [0, 8, 16]),
        ("0, 8, 16", [0, 8, 16]),
    ],
)
def test_parse_checkpoints(value, expected):
    assert parse_checkpoints(value) == expected


def test_parse_qubits():
    assert parse_qubits("0:8", what="q") == "0:8"
    assert parse_qubits("[3, 1]", what="q") == [3, 1]
    assert parse_qubits([3, 1], what="q") == [3, 1]
    assert parse_qubits(5, what="q") == [5]
    with pytest.raises(GrammarError):
        parse_qubits("a:b", what="q")


def test_parse_grid():
    assert parse_grid("measurement_rate=[0.05, 0.1]; num_system=[64,128];") == {
        "measurement_rate": [0.05, 0.1],
        "num_system": [64, 128],
    }
    assert parse_grid("graph_beta=0.3") == {"graph_beta": [0.3]}
    with pytest.raises(GrammarError, match="given twice"):
        parse_grid("a=[1]; a=[2]")
    with pytest.raises(GrammarError, match="empty"):
        parse_grid("a=[]")


def test_plain():
    assert plain([Range(0, 4), {"x": 1}, 2]) == ["0:4", {"x": 1}, 2]


def test_deep_nesting_is_a_grammar_error():
    with pytest.raises(GrammarError, match="nested too deeply"):
        parse_call("f(" + "[" * 40 + "]" * 40 + ")")
    with pytest.raises(GrammarError, match="nested too deeply"):
        parse_qubits("[" * 3000, what="q")
    assert parse_call("f(" + "[" * 32 + "]" * 32 + ")").name == "f"


def test_fuzz_only_grammar_errors():
    rng = random.Random(0)
    alphabet = string.printable + 'βé€()[]@=,:;"\\'
    for _ in range(3000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        with contextlib.suppress(GrammarError):
            parse_call(text, allow_at=True)
    with pytest.raises(GrammarError, match="longer than"):
        parse_call("f(" + "1," * 3000 + "1)")
