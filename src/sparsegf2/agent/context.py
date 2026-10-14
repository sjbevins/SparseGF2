"""What every tool handler receives: the root folder, limits, open states and jobs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Context"]


@dataclass
class Context:
    root: Path
    max_output_chars: int = 4000
    sync_seconds: float = 60.0
    max_states: int = 16
    quiet: bool = False
    stack: str = "all"
    tool_names: tuple[str, ...] = ()
    _states: Any = field(default=None, repr=False)
    _jobs: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"root {str(root)!r} is not a directory")
        self.root = root

    @property
    def states(self):
        if self._states is None:
            from sparsegf2.agent.sessions import StateStore

            self._states = StateStore(self.max_states)
        return self._states

    @property
    def jobs(self):
        if self._jobs is None:
            from sparsegf2.agent.sessions import JobStore

            self._jobs = JobStore()
        return self._jobs

    def path(self, value: Any, *, what: str = "path", must_exist: bool = False) -> Path:
        """``value`` resolved under the root; anything outside it is refused."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{what} must be a non-empty path")
        p = Path(value.strip()).expanduser()
        p = (p if p.is_absolute() else self.root / p).resolve()
        if p != self.root and not p.is_relative_to(self.root):
            raise ValueError(
                f"{what} {value!r} is outside the root folder {str(self.root)!r}; give a path "
                "inside it (relative paths are relative to it)"
            )
        if must_exist and not p.exists():
            raise ValueError(f"{what} {value!r} does not exist (looked for {self.show(p)})")
        return p

    def show(self, path: Any) -> str:
        p = Path(path).resolve()
        try:
            rel = p.relative_to(self.root).as_posix()
        except ValueError:
            return str(p)
        return rel or "."

    def output_dir(self, tool: str, args: dict[str, Any], *, key: str = "output_dir") -> Path:
        """``args[key]`` under the root, or ``runs/<tool>-<hash of the arguments>``.

        The hash leaves out ``output_dir`` and ``background``, so the same request
        always maps to the same directory and running it again resumes it.
        """
        if args.get(key):
            return self.path(args[key], what=key)
        basis = {k: v for k, v in args.items() if k not in (key, "background")}
        text = json.dumps(basis, sort_keys=True, default=str)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
        return self.root / "runs" / f"{tool}-{digest}"
