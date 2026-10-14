"""Tool stacks: built-in stacks, bundles, presets and custom stacks saved by the user.

A stack resolves to a list of :class:`~sparsegf2.agent.registry.Bound` tools.
Custom stacks live in JSON files (spec section 8.2); presets ship in
``presets.json`` in the same format.
"""

from __future__ import annotations

import difflib
import json
import os
import sys
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from sparsegf2.agent.registry import REGISTRY, Bound, ToolDef

__all__ = [
    "CORE",
    "STACKS",
    "Resolution",
    "StackDef",
    "catalog",
    "custom_stacks",
    "delete_stack",
    "load_file",
    "presets",
    "project_file",
    "resolve",
    "save_stack",
    "stack_files",
    "user_file",
]

STACKS = (
    "help", "state", "circuit", "ensemble", "traj", "code", "graph", "analysis", "result",
    "job", "plot",
)  # fmt: skip
CORE = (
    "help_topic", "help_list", "ensemble_run", "traj_run", "code_analyze", "analysis_summary",
    "analysis_probe", "analysis_compare", "result_list", "job_status", "job_cancel",
)  # fmt: skip
BUNDLES = ("all", "core")


@dataclass(frozen=True)
class StackDef:
    name: str
    description: str = ""
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    fixed: dict[str, dict[str, Any]] = field(default_factory=dict)
    defaults: dict[str, dict[str, Any]] = field(default_factory=dict)
    origin: str = ""

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"description": self.description, "include": list(self.include)}
        if self.exclude:
            out["exclude"] = list(self.exclude)
        if self.fixed:
            out["fixed"] = self.fixed
        if self.defaults:
            out["defaults"] = self.defaults
        return out


@dataclass
class Resolution:
    name: str
    bounds: list[Bound]
    notes: list[str]


def _parse(doc: Any, origin: str) -> dict[str, StackDef]:
    if (
        not isinstance(doc, dict)
        or doc.get("version") != 1
        or not isinstance(doc.get("stacks"), dict)
    ):
        raise ValueError(f'{origin}: expected {{"version": 1, "stacks": {{...}}}}')
    out = {}
    for name, body in doc["stacks"].items():
        if not isinstance(body, dict):
            raise ValueError(f"{origin}: stack {name!r} must be an object")
        extra = set(body) - {"description", "include", "exclude", "fixed", "defaults"}
        if extra:
            raise ValueError(f"{origin}: stack {name!r} has unknown key(s) {sorted(extra)}")
        here = f"{origin}: stack {name!r}"
        if not isinstance(body.get("description", ""), str):
            raise ValueError(f"{here}: description must be a string")
        for key in ("include", "exclude"):
            value = body.get(key, [])
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise ValueError(f"{here}: {key} must be a list of names")
        for key in ("fixed", "defaults"):
            value = body.get(key, {})
            if not isinstance(value, dict) or not all(isinstance(v, dict) for v in value.values()):
                raise ValueError(f"{here}: {key} must be an object of objects (tool -> parameters)")
        out[name] = StackDef(
            name,
            body.get("description", ""),
            tuple(body.get("include", ())),
            tuple(body.get("exclude", ())),
            dict(body.get("fixed", {})),
            dict(body.get("defaults", {})),
            origin,
        )
    return out


def presets() -> dict[str, StackDef]:
    text = resources.files("sparsegf2.agent").joinpath("presets.json").read_text(encoding="utf-8")
    return _parse(json.loads(text), "presets.json")


def user_file() -> Path:
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "sparsegf2" / "stacks.json"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "sparsegf2" / "stacks.json"


def project_file(root: Path) -> Path:
    return Path(root) / ".sparsegf2" / "stacks.json"


def stack_files(root: Path, stacks_file: Path | None = None) -> list[Path]:
    files = [Path(stacks_file)] if stacks_file else []
    return [*files, project_file(root), user_file()]


def load_file(path: Path) -> dict[str, StackDef]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from None
    return _parse(doc, str(path))


def custom_stacks(root: Path, stacks_file: Path | None = None) -> dict[str, StackDef]:
    reserved = {*STACKS, *BUNDLES, *presets()}
    out: dict[str, StackDef] = {}
    for path in stack_files(root, stacks_file):
        for name, stack in load_file(path).items():
            if name in reserved:
                raise ValueError(f"{path}: {name!r} is a built-in name; choose another")
            out.setdefault(name, stack)  # earlier files win
    return out


def _all_tools(registry: dict[str, ToolDef]) -> list[str]:
    order = {s: i for i, s in enumerate(STACKS)}
    position = {t: i for i, t in enumerate(registry)}
    return sorted(registry, key=lambda t: (order.get(registry[t].stack, 99), position[t]))


def resolve(
    name: str,
    *,
    root: Path,
    stacks_file: Path | None = None,
    registry: dict[str, ToolDef] | None = None,
) -> Resolution:
    registry = REGISTRY if registry is None else registry
    named = {**presets(), **custom_stacks(root, stacks_file)}
    notes: list[str] = []

    def expand(item: str, seen: tuple[str, ...]) -> dict[str, tuple[dict, dict]]:
        if item in seen:
            raise ValueError(
                f"stack {seen[0]!r} includes itself through {' -> '.join((*seen, item))}"
            )
        if item == "all":
            names = _all_tools(registry)
        elif item == "core":
            names = [t for t in CORE if t in registry]
        elif item in STACKS:
            names = [t for t in _all_tools(registry) if registry[t].stack == item]
        elif item in registry:
            names = [item]
        elif item in named:
            return expand_def(named[item], (*seen, item))
        else:
            options = [*STACKS, *BUNDLES, *named, *registry]
            close = difflib.get_close_matches(item, options, n=3)
            where = f"stack {seen[-1]!r}: " if seen else ""
            kind = "stack or tool" if seen else "stack"
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise ValueError(f"{where}unknown {kind} {item!r}{hint}")
        return {t: ({}, {}) for t in names}

    def apply(
        table: dict[str, dict[str, Any]],
        tools: dict[str, tuple[dict, dict]],
        which: int,
        stack: StackDef,
    ) -> None:
        for target, values in table.items():
            if target != "*" and target not in tools:
                label = "defaults" if which else "fixed"
                raise ValueError(
                    f"stack {stack.name!r}: {label} names {target!r}, which is not in this stack"
                )
            for tname in tools if target == "*" else [target]:
                tool = registry[tname]
                names = {p.name: p for p in tool.params}
                for pname, value in values.items():
                    if pname not in names:
                        if target == "*":
                            continue
                        raise ValueError(
                            f"stack {stack.name!r}: {tname} has no parameter {pname!r}"
                        )
                    checked = names[pname].check(value, f"stack {stack.name!r}, {tname}")
                    tools[tname][which][pname] = checked

    def expand_def(stack: StackDef, seen: tuple[str, ...]) -> dict[str, tuple[dict, dict]]:
        tools: dict[str, tuple[dict, dict]] = {}
        for item in stack.include:
            for t, (fx, df) in expand(item, seen).items():
                cur = tools.setdefault(t, ({}, {}))
                cur[0].update(fx)
                cur[1].update(df)
        for t in stack.exclude:
            if t not in registry:
                close = difflib.get_close_matches(t, list(registry), n=3)
                hint = f"; did you mean {', '.join(close)}?" if close else ""
                raise ValueError(
                    f"stack {stack.name!r}: exclude names {t!r}, which is not a tool{hint}"
                )
            tools.pop(t, None)
        apply(stack.fixed, tools, 0, stack)
        apply(stack.defaults, tools, 1, stack)
        return tools

    tools = expand(name, ())
    if any(registry[t].long_running for t in tools) and "job_status" in registry:
        tools.setdefault("job_status", ({}, {}))
    bounds = []
    for t, (fx, df) in tools.items():
        reason = registry[t].available()
        if reason:
            notes.append(f"{reason}; left out of stack {name!r}")
            continue
        bounds.append(Bound(registry[t], dict(fx), dict(df)))
    return Resolution(name, bounds, notes)


def catalog(root: Path, stacks_file: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    custom = custom_stacks(root, stacks_file)
    entries = (
        [(s, "stack") for s in STACKS]
        + [(b, "bundle") for b in BUNDLES]
        + [(p, "preset") for p in presets()]
        + [(c, "custom") for c in custom]
    )
    for name, kind in entries:
        try:
            r = resolve(name, root=root, stacks_file=stacks_file)
        except ValueError as exc:
            rows.append({"name": name, "kind": kind, "error": str(exc)})
            continue
        if not r.bounds and kind == "stack":
            continue
        desc = (presets().get(name) or custom.get(name) or StackDef(name)).description
        rows.append(
            {
                "name": name,
                "kind": kind,
                "description": desc,
                "tools": [b.name for b in r.bounds],
                "size": sum(b.size() for b in r.bounds),
            }
        )
    return rows


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def save_stack(path: Path, stack: StackDef) -> Path:
    path = Path(path)
    doc: dict[str, Any] = {"version": 1, "stacks": {}}
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
    doc["stacks"][stack.name] = stack.to_json()
    _write(path, doc)
    return path


def delete_stack(path: Path, name: str) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    doc = json.loads(path.read_text(encoding="utf-8"))
    if name not in doc.get("stacks", {}):
        return False
    del doc["stacks"][name]
    _write(path, doc)
    return True
