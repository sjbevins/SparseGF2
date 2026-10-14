"""The call grammar of the agent tools.

Structured arguments are strings that read like Python calls, e.g.
``entropy(0:8)``, ``mutual_information(a=0:4, b=8:12)@every_layer`` or
``erasure_first_failure(samples=512)@[64,128]``. They are parsed here and
never evaluated. Ranges ``a:b[:s]`` use Python slice semantics.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Call",
    "GrammarError",
    "Range",
    "bind",
    "parse_call",
    "parse_calls",
    "parse_checkpoints",
    "parse_grid",
    "parse_qubits",
    "plain",
]

MAX_TEXT = 4000
MAX_DEPTH = 32

_TOKENS = re.compile(
    r"""
      (?P<ws>\s+)
    | (?P<string>"(?:[^"\\]|\\.)*")
    | (?P<range>-?\d+:-?\d+(?::-?\d+)?(?![\w.:]))
    | (?P<number>-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w*!.:]))
    | (?P<word>[A-Za-z_!+\-][\w*!+\-]*)
    | (?P<punct>[()\[\],=@;])
    """,
    re.VERBOSE,
)
_NAME = re.compile(r"[A-Za-z_]\w*")


class GrammarError(ValueError):
    """A string that does not parse; the message says where and why."""


@dataclass(frozen=True)
class Range:
    start: int
    stop: int
    step: int = 1

    def __str__(self) -> str:
        return f"{self.start}:{self.stop}" + (f":{self.step}" if self.step != 1 else "")


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple = ()
    kwargs: tuple = ()
    at: Any = None
    text: str = ""


def _tokens(text: str, what: str) -> list[tuple[str, str, int]]:
    if len(text) > MAX_TEXT:
        raise GrammarError(f"{what}: longer than {MAX_TEXT} characters")
    out, pos = [], 0
    while pos < len(text):
        m = _TOKENS.match(text, pos)
        if m is None:
            raise GrammarError(
                f'{what} "{text}": unexpected character {text[pos]!r} at column {pos + 1}'
            )
        if m.lastgroup != "ws":
            out.append((m.lastgroup, m.group(), pos + 1))
        pos = m.end()
    out.append(("end", "", len(text) + 1))
    return out


class _Parser:
    def __init__(self, text: str, what: str):
        self.text, self.what = text, what
        self.toks = _tokens(text, what)
        self.i = 0
        self.depth = 0

    def peek(self, ahead: int = 0) -> tuple[str, str, int]:
        return self.toks[min(self.i + ahead, len(self.toks) - 1)]

    def take(self) -> tuple[str, str, int]:
        tok = self.peek()
        self.i = min(self.i + 1, len(self.toks) - 1)
        return tok

    def fail(self, message: str, tok: tuple[str, str, int] | None = None) -> None:
        tok = tok or self.peek()
        found = f", found {tok[1]!r}" if tok[1] else ", found the end"
        raise GrammarError(f'{self.what} "{self.text}": {message}{found} at column {tok[2]}')

    def at_end(self) -> bool:
        return self.peek()[0] == "end"

    def value(self) -> Any:
        kind, text, _ = self.peek()
        if kind == "number":
            self.take()
            return float(text) if any(c in text for c in ".eE") else int(text)
        if kind == "range":
            parts = [int(p) for p in text.split(":")]
            if len(parts) == 3 and parts[2] == 0:
                self.fail("range step must not be 0")
            self.take()
            return Range(*parts)
        if kind == "string":
            self.take()
            return text[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        if kind == "word":
            self.take()
            low = text.lower()
            if low in ("true", "false"):
                return low == "true"
            if low in ("none", "null"):
                return None
            return text
        if (kind, text) == ("punct", "["):
            if self.depth >= MAX_DEPTH:
                self.fail("lists nested too deeply")
            self.take()
            self.depth += 1
            items: list[Any] = []
            if self.peek()[1] == "]":
                self.take()
                self.depth -= 1
                return items
            while True:
                items.append(self.value())
                sep = self.peek()
                if sep[1] == "]":
                    self.take()
                    self.depth -= 1
                    return items
                if sep[1] != ",":
                    self.fail("expected ',' or ']'")
                self.take()
        self.fail("expected a value")
        return None  # unreachable

    def checkpoints(self) -> Any:
        kind, text, _ = self.peek()
        if kind == "word" and text in ("final", "every_layer"):
            self.take()
            return text
        if kind == "number" and text.lstrip("-").isdigit():
            self.take()
            return int(text)
        if (kind, text) == ("punct", "["):
            value = self.value()
            if all(isinstance(v, int) and not isinstance(v, bool) for v in value):
                return value
        self.fail("expected final, every_layer, a stride or a list of layers after '@'")
        return None  # unreachable

    def call(self, allow_at: bool) -> Call:
        kind, name, _ = self.peek()
        if kind != "word" or not _NAME.fullmatch(name):
            self.fail("expected a name")
        self.take()
        args: list[Any] = []
        kwargs: list[tuple[str, Any]] = []
        if self.peek()[1] == "(":
            self.take()
            if self.peek()[1] == ")":
                self.take()
            else:
                while True:
                    tok = self.peek()
                    if tok[0] == "word" and self.peek(1)[1] == "=":
                        self.take()
                        self.take()
                        if any(key == tok[1] for key, _ in kwargs):
                            self.fail(f"argument {tok[1]!r} given twice", tok)
                        kwargs.append((tok[1], self.value()))
                    else:
                        if kwargs:
                            self.fail("positional argument after a keyword argument")
                        args.append(self.value())
                    sep = self.peek()
                    if sep[1] == ")":
                        self.take()
                        break
                    if sep[1] != ",":
                        self.fail("expected ',' or ')'")
                    self.take()
        at = None
        if self.peek()[1] == "@":
            if not allow_at:
                self.fail("'@checkpoints' is not allowed here")
            self.take()
            at = self.checkpoints()
        return Call(name, tuple(args), tuple(kwargs), at, self.text)


def parse_call(text: Any, *, allow_at: bool = False, what: str = "call") -> Call:
    """Parse one ``name(args)[@checkpoints]`` string."""
    if not isinstance(text, str):
        raise GrammarError(f"{what}: expected a string like 'entropy(0:8)', got {text!r}")
    p = _Parser(text, what)
    call = p.call(allow_at)
    if not p.at_end():
        p.fail("unexpected text after the call")
    return call


def parse_calls(items: Any, *, allow_at: bool = False, what: str = "calls") -> list[Call]:
    """Parse a list of call strings; errors name the item, e.g. ``probes[2]``."""
    if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
        raise GrammarError(f"{what} must be a list of strings like ['code_dimension']")
    return [parse_call(s, allow_at=allow_at, what=f"{what}[{i}]") for i, s in enumerate(items)]


def bind(
    call: Call, names: Sequence[str], *, what: str, keyword_only: Iterable[str] = ()
) -> dict[str, Any]:
    """Map a call's arguments onto parameter ``names`` (positional ones in order)."""
    allowed = [*names, *keyword_only]
    if len(call.args) > len(names):
        raise GrammarError(
            f'{what} "{call.text}": {call.name} takes at most {len(names)} positional '
            f"argument(s) ({', '.join(names) or 'none'})"
        )
    out = dict(zip(names, call.args, strict=False))
    for key, value in call.kwargs:
        if key not in allowed:
            raise GrammarError(
                f'{what} "{call.text}": unknown argument {key!r}; {call.name} takes '
                f"({', '.join(allowed)})"
            )
        if key in out:
            raise GrammarError(f'{what} "{call.text}": argument {key!r} given twice')
        out[key] = value
    return out


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def parse_checkpoints(value: Any, *, what: str = "checkpoints") -> str | int | list[int]:
    """``final``, ``every_layer``, a stride, or a list of layers (``[0,64]`` or ``0, 64``)."""
    expected = f"{what}: expected final, every_layer, a stride or a list, got {value!r}"
    if isinstance(value, bool):
        raise GrammarError(expected)
    if isinstance(value, int):
        return value
    if isinstance(value, list) and all(_is_int(v) for v in value):
        return value
    if not isinstance(value, str):
        raise GrammarError(expected)
    text = value.strip()
    if "," in text and not text.startswith("["):
        text = f"[{text}]"
    p = _Parser(text, what)
    out = p.checkpoints()
    if not p.at_end():
        p.fail("unexpected text after the checkpoints")
    return out


def parse_qubits(value: Any, *, what: str) -> str | list[int]:
    """A qubit list for the JSON API: a range string ``"a:b[:s]"`` or a list of ints."""
    if isinstance(value, Range):
        return str(value)
    if _is_int(value):
        return [value]
    if isinstance(value, list) and all(_is_int(v) for v in value):
        return value
    if isinstance(value, str):
        p = _Parser(value.strip(), what)
        out = p.value()
        if not p.at_end():
            p.fail("unexpected text after the qubits")
        if isinstance(out, (Range, list, int)) and not isinstance(out, bool):
            return parse_qubits(out, what=what)
    raise GrammarError(f"{what}: expected qubits like 0:8 or [0, 3, 5], got {value!r}")


def parse_grid(text: Any) -> dict[str, list]:
    """``name=[v, ...]; name=v; ...``: a parameter grid (every combination is a point)."""
    if not isinstance(text, str):
        raise GrammarError(
            f"grid: expected a string like 'measurement_rate=[0.1, 0.2]', got {text!r}"
        )
    p = _Parser(text, "grid")
    out: dict[str, list] = {}
    while True:
        tok = p.peek()
        if tok[0] != "word" or not _NAME.fullmatch(tok[1]):
            p.fail("expected a parameter name")
        p.take()
        if p.peek()[1] != "=":
            p.fail("expected '='")
        p.take()
        value = p.value()
        values = value if isinstance(value, list) else [value]
        if not values:
            p.fail(f"empty list of values for {tok[1]}", tok)
        if tok[1] in out:
            p.fail(f"{tok[1]} given twice", tok)
        out[tok[1]] = values
        if p.at_end():
            return out
        if p.peek()[1] != ";":
            p.fail("expected ';' between parameters")
        p.take()
        if p.at_end():
            return out


def plain(value: Any) -> Any:
    """Grammar values as plain JSON for the JSON API (``Range`` becomes ``"a:b[:s]"``)."""
    if isinstance(value, Range):
        return str(value)
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value
