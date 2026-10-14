"""Tool definitions for the agent server: parameters, flat JSON schemas, validation.

Every tool is a :class:`ToolDef` registered in :data:`REGISTRY`. A
:class:`Bound` is a tool as a stack serves it, with some parameters fixed
(removed from the schema) and some defaults changed.
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BACKGROUND",
    "MISSING",
    "REGISTRY",
    "Bound",
    "Param",
    "ToolDef",
    "check_flat",
    "register",
]


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING: Any = _Missing()
TYPES = ("string", "integer", "number", "boolean", "array")
ITEM_TYPES = ("string", "integer", "number")


def _coerce(value: Any, kind: str, name: str) -> Any:
    if kind == "string":
        if isinstance(value, str):
            return value
        raise ValueError(f"{name} must be a string, got {value!r}")
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise ValueError(f"{name} must be true or false, got {value!r}")
    word = "an integer" if kind == "integer" else "a number"
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if isinstance(value, str):
        text = value.strip()
        try:
            value = (
                float(text) if (kind == "number" or any(c in text for c in ".eE")) else int(text)
            )
        except ValueError:
            raise ValueError(f"{name} must be {word}, got the text {text!r}") from None
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be {word}, got {value!r}")
    if kind == "integer":
        if float(value) != int(value):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(value)
    return float(value)


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    description: str
    default: Any = MISSING
    required: bool = False
    enum: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    items: str | None = None

    def __post_init__(self) -> None:
        if self.type not in TYPES:
            raise ValueError(f"param {self.name}: type must be one of {TYPES}")
        if (self.type == "array") != (self.items is not None):
            raise ValueError(f"param {self.name}: items is required for (and only for) arrays")
        if self.items is not None and self.items not in ITEM_TYPES:
            raise ValueError(f"param {self.name}: items must be one of {ITEM_TYPES}")
        if self.required and self.default is not MISSING:
            raise ValueError(f"param {self.name}: a required parameter has no default")

    def schema(self, default: Any = MISSING) -> dict[str, Any]:
        s: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.items is not None:
            s["items"] = {"type": self.items}
        if self.enum is not None:
            s["enum"] = list(self.enum)
        if self.minimum is not None:
            s["minimum"] = self.minimum
        if self.maximum is not None:
            s["maximum"] = self.maximum
        d = self.default if default is MISSING else default
        if d is not MISSING and d is not None:
            s["default"] = d
        return s

    def check(self, value: Any, where: str) -> Any:
        """``value`` converted to this parameter's type, or a ValueError naming the parameter."""
        name = f"{where}: {self.name}" if where else self.name
        if self.type == "array":
            if not isinstance(value, list):
                if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                    value = [value]
                else:
                    raise ValueError(f"{name} must be a list, got {value!r}")
            item = Param(
                self.name,
                self.items or "string",
                "",
                enum=self.enum,
                minimum=self.minimum,
                maximum=self.maximum,
            )
            return [item.check(v, where) for v in value]
        value = _coerce(value, self.type, name)
        if self.enum is not None and value not in self.enum:
            raise ValueError(f"{name} must be one of {list(self.enum)}, got {value!r}")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{name} must be at least {self.minimum:g}, got {value!r}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{name} must be at most {self.maximum:g}, got {value!r}")
        return value


BACKGROUND = Param(
    "background",
    "string",
    "auto: run in the background (returns a job_id) when the estimate exceeds the server's "
    "limit; always; never.",
    default="auto",
    enum=("auto", "always", "never"),
)


@dataclass(frozen=True)
class ToolDef:
    name: str
    stack: str
    summary: str
    description: str
    params: tuple[Param, ...]
    handler: Callable[[Any, dict], dict]
    render: Callable[[dict, Any], str]
    examples: tuple[dict, ...] = ()
    read_only: bool = True
    long_running: bool = False
    estimate: Callable[[Any, dict], float] | None = None
    background: Callable[[Any, dict], dict] | None = None
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.startswith(self.stack + "_"):
            raise ValueError(f"tool {self.name!r} must start with '{self.stack}_'")
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError(f"tool {self.name}: duplicate parameter names")
        if self.long_running and "background" not in names:
            object.__setattr__(self, "params", (*self.params, BACKGROUND))

    def param(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def available(self) -> str | None:
        """None when usable, else why not (a missing optional module)."""
        for module in self.requires:
            if importlib.util.find_spec(module) is None:
                return f"{self.name} needs the optional package {module!r}"
        return None


@dataclass(frozen=True)
class Bound:
    tool: ToolDef
    fixed: Mapping[str, Any] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.tool.name

    def input_schema(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        required = []
        for p in self.tool.params:
            if p.name in self.fixed:
                continue
            props[p.name] = p.schema(self.defaults.get(p.name, MISSING))
            if p.required and p.name not in self.defaults:
                required.append(p.name)
        schema: dict[str, Any] = {
            "type": "object",
            "properties": props,
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required
        return schema

    def listing(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.tool.description,
            "inputSchema": self.input_schema(),
        }

    def size(self) -> int:
        return len(json.dumps(self.listing(), separators=(",", ":")))

    def validate(self, args: Mapping[str, Any] | None) -> dict[str, Any]:
        args = dict(args or {})
        where = self.name
        known = {p.name: p for p in self.tool.params}
        unknown = sorted(set(args) - set(known))
        if unknown:
            takes = sorted(set(known) - set(self.fixed))
            raise ValueError(f"{where}: unknown argument(s) {unknown}; it takes {takes}")
        out: dict[str, Any] = {}
        for name, p in known.items():
            given = args.get(name)
            if name in self.fixed:
                if given is not None and p.check(given, where) != self.fixed[name]:
                    raise ValueError(
                        f"{where}: {name} is fixed to {self.fixed[name]!r} by this tool stack"
                    )
                out[name] = self.fixed[name]
            elif given is not None:
                out[name] = p.check(given, where)
            elif name in self.defaults:
                out[name] = self.defaults[name]
            elif p.default is not MISSING and p.default is not None:
                out[name] = p.default
            elif p.required:
                raise ValueError(f"{where}: missing required argument {name!r} ({p.description})")
        return out


REGISTRY: dict[str, ToolDef] = {}


def register(tool: ToolDef) -> ToolDef:
    if tool.name in REGISTRY:
        raise ValueError(f"tool {tool.name!r} is already registered")
    REGISTRY[tool.name] = tool
    return tool


_FORBIDDEN = ("$ref", "$defs", "oneOf", "anyOf", "allOf", "not")


def check_flat(schema: Mapping[str, Any]) -> list[str]:
    """Violations of the flat-schema rule (spec 2.2); empty when the schema is flat."""
    problems = []
    for key in _FORBIDDEN:
        if key in schema:
            problems.append(f"top level uses {key}")
    for name, prop in schema.get("properties", {}).items():
        for key in _FORBIDDEN:
            if key in prop:
                problems.append(f"{name} uses {key}")
        kind = prop.get("type")
        if kind not in TYPES:
            problems.append(f"{name} has type {kind!r}")
        if kind == "array" and prop.get("items", {}).get("type") not in ITEM_TYPES:
            problems.append(f"{name} is an array of non-scalars")
    return problems
