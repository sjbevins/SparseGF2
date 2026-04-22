"""Read/write the per-cell ``analysis/_registry.json`` manifest."""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional


SCHEMA_VERSION = "1.0.0"


@dataclass
class RegistryEntry:
    """Metadata for one analysis in one cell's ``_registry.json``."""

    computed_utc: str
    package_version: str
    git_hash: Optional[str]
    git_dirty: Optional[bool]
    params: dict
    runtime_s: float
    n_samples: int


def _utcnow() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_hash(repo_root: Path) -> tuple[Optional[str], Optional[bool]]:
    """Best-effort git hash + dirty state; both ``None`` on failure."""
    try:
        h = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=2.0
        ).strip()
    except Exception:
        return None, None
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True, timeout=2.0
        )
        return h, bool(out.strip())
    except Exception:
        return h, None


def make_entry(
    *, package_version: str, params: dict, runtime_s: float, n_samples: int,
    repo_root: Optional[Path] = None,
) -> RegistryEntry:
    git_hash, git_dirty = _git_hash(repo_root or Path.cwd())
    return RegistryEntry(
        computed_utc=_utcnow(),
        package_version=package_version,
        git_hash=git_hash,
        git_dirty=git_dirty,
        params=dict(params),
        runtime_s=float(runtime_s),
        n_samples=int(n_samples),
    )


# ══════════════════════════════════════════════════════════════
# Read/write
# ══════════════════════════════════════════════════════════════

def read_cell_registry(cell_analysis_dir: Path) -> Dict[str, dict]:
    """Return the ``entries`` mapping of a cell registry, empty dict if absent."""
    path = Path(cell_analysis_dir) / "_registry.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data.get("entries", {}))


def write_cell_registry(cell_analysis_dir: Path, entries: Dict[str, dict]) -> None:
    """Write the registry file, creating the analysis dir if needed."""
    d = Path(cell_analysis_dir)
    d.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "entries": entries}
    (d / "_registry.json").write_text(json.dumps(payload, indent=2, sort_keys=False))


def upsert_entry(
    cell_analysis_dir: Path, name: str, entry: RegistryEntry
) -> None:
    """Insert or update one analysis entry in the cell registry."""
    entries = read_cell_registry(cell_analysis_dir)
    entries[name] = asdict(entry)
    write_cell_registry(cell_analysis_dir, entries)


__all__ = [
    "SCHEMA_VERSION",
    "RegistryEntry",
    "make_entry",
    "read_cell_registry",
    "write_cell_registry",
    "upsert_entry",
]
