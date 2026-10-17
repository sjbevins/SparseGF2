"""The JSON service layer of the Watts–Strogatz study: one implementation for every front end.

The ``sparsegf2 study`` commands of the CLI, the ``study_*`` tools of the MCP server and Python
callers all call these functions, so their arguments, results and error messages are the same
everywhere (design §8, gap G-CLI-1), as for the rest of SparseGF2 (:mod:`sparsegf2.api`). Every
public function takes plain JSON values and returns a JSON-serializable dict that is **small**:
under :data:`sparsegf2.api.MAX_OUTPUT_CHARS` characters when encoded, so that an agent with a
short context reads it whole; long tables are written to files when an ``out`` path is given.
Invalid input raises :class:`ValueError` with a message that says what to change (a missing
trajectory, graph, snapshot or observable included).

A study store is named by its directory (a path string). Reading functions open it without the
writer lock, so they work while a run is writing; writing functions (:func:`init`, :func:`run`,
:func:`start_run_job`, :func:`extend`, :func:`compute`, ``aggregate(record=True)``) take the
store's lock and are refused, with the holder named, while another writer has it.

Functions
---------
:func:`catalog`
    The observable catalog: ids, versions, levels, inputs, parameters, definitions.
:func:`init`
    Create a store from a manifest (revision 1) and report the planned work.
:func:`status`
    Revisions, conditions, units and trajectories done and remaining, snapshots, storage, the
    stored observables.
:func:`plan`
    The remaining work units with estimated runtimes (``sparsegf2.api.estimate_ensemble`` on
    each unit's engine spec, scaled by a factor measured on this machine), storage and memory.
:func:`run`, :func:`start_run_job`
    Run the plan synchronously, or as a background job that ``sparsegf2.api.job_status``,
    ``cancel_job`` and ``list_jobs`` follow; both stop cleanly between units.
:func:`extend`
    Add a manifest revision (new conditions, rates, graphs, circuits, depths, observables).
:func:`tableau`
    One stored (or replayed) phase-free tableau: its size, rank, ``k``, digests, generators.
:func:`compute`
    A retrospective observable over the stored snapshots.
:func:`aggregate`, :func:`aggregate_arrays`
    Nested equal-graph-weight means with graph-cluster errors per condition and depth;
    censoring-aware first-passage summaries; peak-time summaries; paired rate differences.
:func:`fss`
    A finite-size-scaling fit report at one ``beta``.
:func:`export`, :func:`verify`, :func:`plot`
    Open tables, a replay check of a sample, diagnostic figures.

``docs/study/09-cli-api-and-plots.md`` describes every function, its result and the
mathematics of what it reports.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import json
import math
import os
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from sparsegf2 import api as _api
from sparsegf2.study import fss as _fss
from sparsegf2.study import observables as _ob
from sparsegf2.study import runner as _runner
from sparsegf2.study import stats as _stats
from sparsegf2.study.identity import Token, build_graph, canonical_json, check_protocol, sha256_hex
from sparsegf2.study.manifest import CorrelatedErasure, ExperimentManifest
from sparsegf2.study.store import (
    ESTIMATOR_VERSION,
    EXPORT_FORMAT,
    QUBIT_ORDER,
    SNAPSHOT_ABSENT,
    STORE_FILE,
    TableauStore,
)
from sparsegf2.tableau import PhaseFreeTableau

__all__ = [
    "AGGREGATE_KINDS",
    "FSS_REPORT_SCHEMA",
    "JOB_KIND",
    "aggregate",
    "aggregate_arrays",
    "catalog",
    "compute",
    "export",
    "extend",
    "fss",
    "fss_report",
    "init",
    "plan",
    "plot",
    "run",
    "safe_name",
    "start_run_job",
    "status",
    "tableau",
    "verify",
]

#: The schema of the report :func:`fss` returns in full and writes to ``out``.
FSS_REPORT_SCHEMA = "sparsegf2.study.fss/1"

#: The ``kind`` of a study run in ``sparsegf2.api.job_status`` and ``list_jobs``.
JOB_KIND = "study_run"

#: How :func:`aggregate` treats an observable: by its catalog level and id.
AGGREGATE_KINDS = ("snapshot", "first_passage", "peak", "k_trace")

#: Conditions and units shown in one result (the rest are counted; ``out`` writes them all).
MAX_SHOWN_CONDITIONS = 24
MAX_SHOWN_UNITS = 12
MAX_SHOWN_FAILURES = 10

#: Largest estimated engine time (seconds) of the trial unit of ``plan(benchmark=True)``.
TRIAL_ENGINE_SECONDS = 5.0

#: The ``where`` filters: protocol, size, rewiring probability, rate, depth.
WHERE_KEYS = ("protocol", "n", "beta", "p", "t")

_PEAK_TRACES: dict[str, tuple[str, str | None]] = {
    "peak_d_cyc": ("d_cyc", "zero_extended"),
    "peak_D": ("D", None),
    "peak_D_J": ("D_J", "robustness"),
    "peak_d_conn": ("d_conn", None),
}
_THETA = {
    "first_passage": "tau",
    "peak_d_cyc": "T*_cyc",
    "peak_D": "T*_D",
    "peak_D_J": "T*_J",
    "peak_d_conn": "T*_conn",
}
_METHODS = {"hh": _fss.HOUDAYER_HARTMANN, "houdayer_hartmann": _fss.HOUDAYER_HARTMANN,
            "polynomial": _fss.POLYNOMIAL}  # fmt: skip
_DEFAULT_BOUNDS = {"z": (0.0, 3.0), "nu": (0.5, 5.0)}  # p_c: the range of the rates used


# ---------------------------------------------------------------------------------------------
# small helpers: errors, input checks, opening stores
# ---------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _lookups() -> Iterator[None]:
    """The store and the catalog raise ``KeyError`` for a record that is not there (a
    trajectory, graph, snapshot, observable or version); to a caller that is an input error."""
    try:
        yield
    except KeyError as exc:
        message = exc.args[0] if exc.args and isinstance(exc.args[0], str) else str(exc)
        raise ValueError(message) from None


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value.strip()


def _int(value: Any, what: str, minimum: int | None = 0) -> int:
    return _api._require_int(value, what, minimum=minimum)


def _optional_int(value: Any, what: str, minimum: int | None = 0) -> int | None:
    return None if value is None else _int(value, what, minimum)


def _positive_seconds(value: Any, what: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a positive number of seconds, got {value!r}")
    if not float(value) > 0 or not math.isfinite(float(value)):
        raise ValueError(f"{what} must be a positive number of seconds, got {value!r}")
    return float(value)


def _flag(value: Any, what: str) -> bool:
    return _api._require_bool(value, what)


def _choice(value: Any, what: str, choices: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{what} must be one of {list(choices)}, got {value!r}")
    return value


def _store_dir(store: Any) -> Path:
    if not isinstance(store, (str, os.PathLike)) or not str(store).strip():
        raise ValueError(f"store must be the path of a study store directory, got {store!r}")
    return Path(store).expanduser().resolve()


@contextlib.contextmanager
def _opened(store: Any, *, write: bool = False) -> Iterator[TableauStore]:
    """The store ``store`` (a directory, or an open :class:`TableauStore`, kept open), closed
    (and its writer lock released) at the end."""
    if isinstance(store, TableauStore):
        if write:
            store._require_writer()
        yield store
        return
    s = TableauStore.open(_store_dir(store), write=write)
    try:
        yield s
    finally:
        s.close()


def _manifest(value: Any) -> ExperimentManifest:
    """A manifest from a JSON object, JSON text, or the path of a ``.json`` file."""
    if isinstance(value, ExperimentManifest):
        return value
    try:
        if isinstance(value, Mapping):
            return ExperimentManifest.from_json(dict(value))
        if isinstance(value, (str, os.PathLike)):
            text = str(value)
            if text.lstrip().startswith("{"):
                return ExperimentManifest.from_json(text)
            path = Path(text).expanduser()
            if not path.is_file():
                raise ValueError(
                    f"manifest: no such file {text!r} (give the manifest as a JSON object, as "
                    "JSON text, or as the path of a .json file)"
                )
            return ExperimentManifest.from_json(path)
    except TypeError as exc:
        raise ValueError(f"manifest: {exc}") from None
    raise ValueError(
        "manifest must be a JSON object, JSON text or the path of a .json file, got "
        f"{type(value).__name__}"
    )


def _finish(out: dict[str, Any]) -> dict[str, Any]:
    """The result as a small JSON dict (floats rounded for display, long lists cut, with
    ``truncated`` and a note when anything was cut): :func:`sparsegf2.api._bounded`."""
    return _api._bounded(out)


def _exact(obj: Any) -> Any:
    """Plain JSON values with floats kept exactly (non-finite as None), for files."""
    return _api._jsonable(obj, exact=True)


def _write_json(path: Path, obj: Any) -> None:
    """Canonical text (sorted keys, two-space indent, a final newline): the same content gives
    the same bytes."""
    text = json.dumps(_exact(obj), sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if v is None else _cell(v) for v in row])


def _cell(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        x = float(value)
        return "" if not math.isfinite(x) else repr(x)
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _out_dir(path: Any, what: str = "out") -> Path:
    if not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise ValueError(f"{what} must be a directory path, got {path!r}")
    out = Path(path).expanduser().resolve()
    if out.exists() and not out.is_dir():
        raise ValueError(f"{what} {str(out)!r} exists and is not a directory")
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"{what} {str(out)!r} cannot be created: {exc}") from None
    return out


def safe_name(text: str) -> str:
    """``text`` as a file-name part: ``*`` as ``star``, ``/`` as ``over``, other characters
    that are not letters, digits or ``._=+-`` as ``_`` (``T*_J=1`` is ``Tstar_J=1``)."""
    text = str(text).replace("*", "star").replace("/", "over")
    return "".join(ch if ch.isalnum() or ch in "._=+-" else "_" for ch in text)


def _peak_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest's ``analysis.peak`` with its width named ``plateau_z``: the plateau rule's
    ``z`` (v1.4 §5.8) is not the dynamical exponent ``z`` of the scaling fits (v1.4 §5.9)."""
    return {
        "rule": rule.get("rule"),
        "plateau_z": rule.get("z"),
        "sensitivity_plateau_z": list(rule.get("sensitivity_z", [])),
        "bootstrap": rule.get("bootstrap"),
    }


# ---------------------------------------------------------------------------------------------
# conditions and the `where` filter
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Cond:
    """One condition ``(protocol, n, beta, p)``, tokens as their canonical strings."""

    protocol: str
    n: int
    beta: str
    p: str

    def fields(self) -> dict[str, Any]:
        return {"protocol": self.protocol, "n": self.n, "beta": self.beta, "p": self.p}

    def label(self) -> str:
        return f"{self.protocol}:n={self.n}:beta={self.beta}:p={self.p}"


def _where(where: Any) -> dict[str, set[Any]]:
    """``where`` as sets of canonical values per key (an absent key matches everything)."""
    if where is None:
        return {}
    if not isinstance(where, Mapping):
        raise ValueError(
            'where must be an object such as {"n": [16, 32], "beta": "0.1", "p": ["0.1", '
            f'"0.2"], "t": 8}}, got {where!r}'
        )
    _api._check_keys(dict(where), WHERE_KEYS, "where")
    out: dict[str, set[Any]] = {}
    for key, value in where.items():
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        if not values:
            raise ValueError(f"where.{key} is an empty list; leave the key out to keep every value")
        if key == "protocol":
            out[key] = {check_protocol(_text(v, "where.protocol")) for v in values}
        elif key in ("n", "t"):
            out[key] = {_int(v, f"where.{key}") for v in values}
        else:
            out[key] = {Token.parse(v, what=f"where.{key}").token for v in values}
    return out


def _conditions(manifest: ExperimentManifest, where: Mapping[str, set[Any]]) -> list[_Cond]:
    """The conditions of the latest manifest that ``where`` selects, in the manifest's order."""
    out: list[_Cond] = []
    for protocol in manifest.protocol_names:
        if "protocol" in where and protocol not in where["protocol"]:
            continue
        for cond in manifest.conditions():
            if "n" in where and cond.n not in where["n"]:
                continue
            if "beta" in where and cond.beta.token not in where["beta"]:
                continue
            for p in cond.rates:
                if "p" in where and p.token not in where["p"]:
                    continue
                out.append(_Cond(protocol, cond.n, cond.beta.token, p.token))
    if not out:
        conds = manifest.conditions()
        raise ValueError(
            f"where { ({k: sorted(v, key=str) for k, v in where.items()}) } selects no condition "
            f"of experiment {manifest.experiment_id!r}: it has protocols "
            f"{list(manifest.protocol_names)}, n {sorted({c.n for c in conds})}, beta "
            f"{sorted({c.beta.token for c in conds})} and p "
            f"{sorted({p.token for c in conds for p in c.rates})}"
        )
    return out


def _stored_keys(s: TableauStore) -> list[tuple[str, str, str]]:
    """The stored observable keys ``(observable, version, params)`` (the ``observable_defs``
    table of the store schema, docs/study/05 §3.3)."""
    rows = s._db().execute(
        "SELECT observable, version, params FROM observable_defs ORDER BY observable, version, "
        "params"
    )
    return [(str(o), str(v), str(p)) for o, v, p in rows.fetchall()]


def _same(given: Any, stored: Any) -> bool:
    if isinstance(given, (list, tuple)) and isinstance(stored, (list, tuple)):
        return len(given) == len(stored) and all(
            _same(a, b) for a, b in zip(given, stored, strict=True)
        )
    if isinstance(given, bool) or isinstance(stored, bool):
        return given is stored
    if isinstance(given, (int, float)) and isinstance(stored, (int, float)):
        return float(given) == float(stored)
    return bool(given == stored)


def _matches(stored: Mapping[str, Any], wanted: Mapping[str, Any]) -> bool:
    """Whether every entry of ``wanted`` is in ``stored`` with an equal value."""
    return all(k in stored and _same(v, stored[k]) for k, v in wanted.items())


def _select(s: TableauStore, observable: str, version: str | None, params: Any) -> tuple[str, str]:
    """The stored ``(version, params)`` of ``observable`` that ``version`` and ``params``
    select. ``params`` may be the stored canonical text, or an object whose entries must all
    match a stored parameter set (so ``{"J": 1}`` picks the coupling of ``peak_D_J`` without
    the peak rule's other settings); with one stored set it may be left out."""
    with _lookups():
        _ob.get(observable, version)
    choices = [(v, p) for o, v, p in _stored_keys(s) if o == observable]
    if version is not None:
        choices = [c for c in choices if c[0] == version]
    if not choices:
        stored = sorted({o for o, _, _ in _stored_keys(s)})
        raise ValueError(
            f"this store holds no values of {observable!r}"
            + (f" version {version!r}" if version is not None else "")
            + f"; it holds {stored} (a retrospective observable is added with compute)"
        )
    if isinstance(params, str):
        picked = [c for c in choices if c[1] == params]
    elif isinstance(params, Mapping):
        picked = [c for c in choices if _matches(json.loads(c[1]), params)]
    elif params is None:
        picked = choices
    else:
        raise ValueError(
            f"params must be an object (the parameters to match) or the stored text, got {params!r}"
        )
    shown = [{"version": v, "params": json.loads(p)} for v, p in choices]
    if not picked:
        raise ValueError(
            f"no stored values of {observable!r} match params {params!r}; stored: {shown}"
        )
    if len(picked) > 1:
        raise ValueError(
            f"{observable!r} is stored under several versions or parameter sets; choose one with "
            f"params (and version): {[{'version': v, 'params': json.loads(p)} for v, p in picked]}"
        )
    return picked[0]


# ---------------------------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------------------------


def catalog(observable: str | None = None) -> dict[str, Any]:
    """The observable catalog (design §5, ``study/observables.py``): every observable's id,
    version, level, inputs, default parameters and definition.

    With ``observable`` one entry in full (every version, the parameter kinds and the function
    that computes it). Retrospective observables (``retrospective: true``) are computed later
    from stored snapshots with :func:`compute`; the others are recorded by the run.
    """
    rows = list(_ob.catalog())
    notes = [
        "retrospective observables are computed from stored snapshots with compute(store, id, "
        "params); the others are recorded when a trajectory runs",
        "status(store).observables lists the keys a store holds; aggregate(store, id) summarizes "
        "them",
    ]
    if observable is not None:
        with _lookups():
            entry = _ob.get(_text(observable, "observable"))
        return _finish(
            {
                "observable": entry.id,
                "versions": [r for r in rows if r["id"] == entry.id],
                "notes": notes,
            }  # fmt: skip
        )
    keys = ("id", "version", "level", "retrospective", "inputs", "params", "definition")
    compact = [{k: r[k] for k in keys} for r in rows]
    out: dict[str, Any] = {
        "count": len(rows),
        "levels": list(_ob.LEVELS),
        "observables": compact,
        "notes": notes,
    }
    if len(json.dumps(_api._jsonable(out))) > _api.MAX_OUTPUT_CHARS:
        for row in compact:
            text = str(row["definition"])
            row["definition"] = text if len(text) <= 110 else text[:107] + "..."
        notes.append("definitions shortened; catalog(observable=ID) gives one entry in full")
    return _finish(out)


# ---------------------------------------------------------------------------------------------
# init, status
# ---------------------------------------------------------------------------------------------


def _storage(directory: Path, snapshots: int) -> dict[str, Any]:
    files = {
        p.name: p.stat().st_size
        for p in sorted(directory.iterdir())
        if p.is_file() and p.name.startswith(STORE_FILE)
    }
    total = sum(files.values())
    return {
        "megabytes": total / 1e6,
        "files": {k: v / 1e6 for k, v in files.items()},
        "bytes_per_snapshot": total / snapshots if snapshots else None,
    }


def _plan_counts(units: Sequence[_runner.WorkUnit]) -> dict[str, int]:
    return {
        "units": len(units),
        "trajectories": sum(u.trajectories for u in units),
        "rerun_units": sum(u.rerun for u in units),
        "rerun_trajectories": sum(u.trajectories for u in units if u.rerun),
    }


def init(manifest: Any, store: Any) -> dict[str, Any]:
    """Create a study store from ``manifest`` (its revision 1) in the new directory ``store``.

    ``manifest`` is a JSON object, JSON text or the path of a ``.json`` file
    (``docs/study/02``, the manifest field by field). Nothing runs: the result says what the
    plan holds; :func:`plan` estimates it and :func:`run` or :func:`start_run_job` runs it.
    """
    m = _manifest(manifest)
    directory = _store_dir(store)
    s = TableauStore.create(directory, m)
    try:
        counts = _plan_counts(_runner.plan(s))
        conds = m.conditions()
        out = {
            "store": str(directory),
            "experiment_id": m.experiment_id,
            "label": m.label,
            "revision": s.revision,
            "manifest_digest": m.digest(),
            "protocols": list(m.protocol_names),
            "conditions": len(conds),
            "rates": sum(len(c.rates) for c in conds),
            "trajectories_per_condition": m.trajectories_per_condition,
            "planned_units": counts["units"],
            "planned_trajectories": counts["trajectories"],
            "warnings": list(m.warnings),
            "next_step": "plan(store) for the runtime estimate, then run(store) or "
            "start_run_job(store)",
        }
    finally:
        s.close()
    return _finish(out)


def status(store: Any) -> dict[str, Any]:
    """Where a study stands: revisions, conditions, units and trajectories done and remaining,
    snapshots, storage and the stored observable keys.

    A trajectory is ``complete`` when it holds every record the latest revision asks for; a
    ``rerun`` is a stored trajectory that a later revision asked more of (new depths, a deeper
    horizon, a new observable): :func:`run` reruns it from layer 0 and checks every stored
    record before adding the new ones (design D3).
    """
    with _opened(store) as s:
        revisions = s.revisions()
        latest = revisions[-1]
        units = _runner.plan(s)
        remaining: Counter[_Cond] = Counter()
        reruns: Counter[_Cond] = Counter()
        for u in units:
            cond = _Cond(u.protocol, u.n, u.beta.token, u.p.token)
            remaining[cond] += u.trajectories
            if u.rerun:
                reruns[cond] += u.trajectories
        expected = latest.trajectories_per_condition
        rows = []
        for cond in _conditions(latest, {}):
            stored = len(s.stored_pairs(cond.protocol, cond.n, cond.beta, cond.p))
            rows.append(
                {
                    **cond.fields(),
                    "expected": expected,
                    "stored": stored,
                    "complete": expected - remaining[cond],
                    "reruns": reruns[cond],
                }  # fmt: skip
            )
        recorded = s.units()
        done = [u for u in recorded if u["status"] == "done"]
        failed = [u for u in recorded if u["status"] == "failed"]
        snap = dict(s._db().execute("SELECT status, count(*) FROM snapshots GROUP BY status"))
        total_snapshots = sum(int(v) for v in snap.values())
        seconds = sum(float(u["elapsed_seconds"] or 0.0) for u in done)
        done_trajectories = sum(int(u["trajectories"]) for u in done)
        stored_keys = _stored_keys(s)
        storage = _storage(s.path, total_snapshots - int(snap.get(SNAPSHOT_ABSENT, 0)))
        versions = s.meta()["versions"]
        directory = s.path
    rows.sort(key=lambda r: r["complete"] == r["expected"])  # incomplete first, stable
    shown = rows[:MAX_SHOWN_CONDITIONS]
    total_expected = expected * len(rows)
    left = sum(remaining.values())
    out: dict[str, Any] = {
        "store": str(directory),
        "experiment_id": latest.experiment_id,
        "label": latest.label,
        "revision": len(revisions),
        "manifest_digest": latest.digest(),
        "complete": not units,
        "trajectories": {
            "expected": total_expected,
            "stored": sum(r["stored"] for r in rows),
            "complete": total_expected - left,
            "remaining": left,
            "reruns": sum(reruns.values()),
        },
        "units": {
            "done": len(done),
            "failed": len(failed),
            "remaining": len(units),
            "seconds": seconds,
            "trajectories_per_second": done_trajectories / seconds if seconds > 0 else None,
        },
        "snapshots": {"total": total_snapshots, **{str(k): int(v) for k, v in snap.items()}},
        "storage": storage,
        "observables": [
            {"id": o, "version": v, "params": json.loads(p)} for o, v, p in stored_keys
        ],
        "peak_rule": _peak_rule(latest.analysis.peak.to_dict()),
        "versions": {
            k: versions.get(k) for k in ("randomness_contract", "store_schema", "sparsegf2")
        },
        "conditions": shown,
        "conditions_total": len(rows),
        "conditions_complete": sum(r["complete"] == r["expected"] for r in rows),
    }
    if failed:
        out["failed_units"] = [
            {
                "unit_id": u["unit_id"],
                "condition": f"{u['protocol']}:n={u['n']}:beta={u['beta']}:p={u['p']}",
                "error": str(u["error"])[:300],
            }  # fmt: skip
            for u in failed[-3:]
        ]
    notes = []
    if len(rows) > len(shown):
        notes.append(
            f"{len(shown)} of {len(rows)} conditions shown (incomplete ones first); "
            "aggregate(..., where=...) reads any of them"
        )
    if failed:
        notes.append(
            "failed units wrote no data (each unit is one transaction); run again to retry them"
        )
    if notes:
        out["notes"] = notes
    out["next_step"] = (
        "aggregate(store, observable, where=...) and fss(store, ...)"
        if not units
        else "run(store) or start_run_job(store) runs the remaining units (plan(store) first "
        "for the estimate)"
    )
    return _finish(out)


# ---------------------------------------------------------------------------------------------
# plan: the remaining units with estimated runtimes, memory and storage
# ---------------------------------------------------------------------------------------------


def _probes_json(req: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The JSON probes of :func:`sparsegf2.api.estimate_ensemble` for the requirements ``req``:
    the same probes as :func:`sparsegf2.study.runner.engine_probes` (a test checks that both
    build equal engine specs)."""
    names = _runner.PROBE_NAMES
    cps = [int(t) for t in req["checkpoints"]]
    if req["protocol"] == "P":
        return [{"kind": "tableau_snapshot", "name": names["tableau_snapshot"], "checkpoints": cps}]
    out: list[dict[str, Any]] = [{"kind": "code_dimension"}]
    snapshot = str(req["snapshot"])
    out.append({"kind": snapshot, "name": names[snapshot], "checkpoints": cps})
    if req["contiguous_distance"]:
        out.append({"kind": "contiguous_distance", "checkpoints": cps})
    uniform = req["uniform_erasure"]
    if uniform is not None:
        out.append(
            {
                "kind": "erasure_first_failure",
                "samples": int(uniform["samples"]),
                "name": names["erasure_first_failure"],
                "checkpoints": cps,
            }  # fmt: skip
        )
        if uniform["exact_budget"]:
            out.append(
                {
                    "kind": "uncorrectable_region_counts",
                    "budget": int(uniform["exact_budget"]),
                    "name": names["uncorrectable_region_counts"],
                    "checkpoints": cps,
                }  # fmt: skip
            )
    for entry in req["correlated_erasure"]:
        settings = CorrelatedErasure.from_dict(
            {k: v for k, v in entry.items() if k != "name"}, "correlated_erasure"
        ).to_dict()
        settings.pop("name", None)
        out.append(
            {"kind": "correlated_erasure", **settings, "name": entry["name"], "checkpoints": cps}
        )
    return out


def _unit_spec(
    manifest: ExperimentManifest, req: Mapping[str, Any], p: Any, edges: np.ndarray
) -> dict[str, Any]:
    """The JSON engine spec of one trajectory of the condition (``sparsegf2.api`` format):
    the manifest's keywords (v1.4 §8.1) with the requirements' horizon and probes, on one of
    the condition's graphs (every Watts–Strogatz graph of a size has ``2n`` edges)."""
    kwargs = manifest.monitored_circuit_kwargs(req["n"], req["beta"], p, req["protocol"])
    kwargs.pop("circuits_per_graph", None)
    kwargs["layers"] = int(req["layers"])
    if req["protocol"] == "F":
        kwargs["stop_when_purified"] = bool(req["stop_when_purified"])
    return {**kwargs, "edges": np.asarray(edges).tolist(), "probes": _probes_json(req)}


class _Estimator:
    """Engine estimates (``sparsegf2.api.estimate_ensemble``) of unit specs, cached by
    condition, horizon and trajectory count."""

    def __init__(self, s: TableauStore, manifest: ExperimentManifest):
        self.s = s
        self.manifest = manifest
        self.graphs: dict[tuple[int, str], np.ndarray] = {}
        self.cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def edges(self, n: int, beta: str) -> np.ndarray:
        key = (n, beta)
        if key not in self.graphs:
            if self.s.has_graph(n, beta, 0):
                self.graphs[key] = self.s.graph(n, beta, 0).edges
            else:
                seed = self.manifest.seeds.master_graph_seed
                self.graphs[key] = build_graph(n, beta, 0, seed).edges
        return self.graphs[key]

    def estimate(
        self, req: Mapping[str, Any], p: str, trajectories: int, threads: int
    ) -> dict[str, Any]:
        digest = _runner.requirements_digest(req)
        key = (digest, p, trajectories, threads)
        if key not in self.cache:
            spec = _unit_spec(self.manifest, req, p, self.edges(int(req["n"]), str(req["beta"])))
            est = _api.estimate_ensemble(spec, trajectories, threads=[threads])
            snapshots_mb = sum(
                float(a["megabytes"] or 0.0)
                for a in est["layout"]["arrays"]
                if str(a["name"]).startswith("snapshot::")
            )
            self.cache[key] = {
                "engine_seconds": float(est["by_threads"][0]["estimated_seconds"]),
                # the engine's result, and the canonical copies of its snapshots that the
                # ingestion holds beside it (one per snapshot): a model, not a measurement
                "memory_megabytes": float(est["result_megabytes"]) + snapshots_mb,
            }
        return self.cache[key]


def _snapshot_count(s: TableauStore, unit: _runner.WorkUnit, req: Mapping[str, Any]) -> int:
    """The snapshots a unit adds: every checkpoint of a new trajectory; for a rerun the depths
    its trajectories do not hold yet."""
    depths = len(req["checkpoints"])
    if not unit.rerun:
        return unit.trajectories * depths
    key = s.key(n=unit.n, beta=unit.beta, p=unit.p, g=unit.graphs[0], c=unit.circuits[0],
                protocol=unit.protocol)  # fmt: skip
    return unit.trajectories * max(0, depths - len(s.depths(key)))


def _history_factor(
    s: TableauStore, estimator: _Estimator, revisions: Sequence[ExperimentManifest]
) -> dict[str, Any] | None:
    """Measured over estimated seconds of the units this store has run (at most the last
    200): the elapsed time of a unit includes its ingestion (canonical forms, observables, the
    transaction), which ``estimate_ensemble`` does not price."""
    done = [u for u in s.units(status="done") if u["elapsed_seconds"]][-200:]
    if not done:
        return None
    measured = estimated = 0.0
    for u in done:
        try:
            req = s.requirements(u["requirements"])
        except KeyError:
            continue
        est = estimator.estimate(req, u["p"], int(u["trajectories"]), int(u["threads"] or 1))
        measured += float(u["elapsed_seconds"])
        estimated += est["engine_seconds"]
    if estimated <= 0:
        return None
    return {"value": measured / estimated, "source": "store", "units_measured": len(done)}


def _trial(manifest: ExperimentManifest, threads: int, num_threads: int | None) -> dict[str, Any]:
    """Run one small unit twice in scratch stores (the first warms the kernels and caches), time
    the second, and measure its storage per snapshot. Nothing is written to the study."""
    root = Path(tempfile.mkdtemp(prefix="sparsegf2-study-trial-"))

    def size(directory: Path) -> int:
        return sum(p.stat().st_size for p in directory.iterdir() if p.is_file())

    try:
        first = TableauStore.create(root / "warm", manifest)
        try:
            estimator = _Estimator(first, manifest)
            revisions = first.revisions()
            k = max(1, min(manifest.trials.circuits_per_graph, threads))
            # one rectangle (one graph, k circuits) per (protocol, n, beta): the costliest one
            # whose engine estimate is within TRIAL_ENGINE_SECONDS, else the cheapest
            candidates: list[tuple[float, _runner.WorkUnit]] = []
            seen: set[tuple[str, int, str]] = set()
            for unit in _runner.plan(first):
                ident = (unit.protocol, unit.n, unit.beta.token)
                if ident in seen:
                    continue
                seen.add(ident)
                req = _runner.requirements(revisions, unit.protocol, unit.n, unit.beta)
                rect = dataclasses.replace(unit, graphs=(0, 1), circuits=(0, k))
                est = estimator.estimate(req, unit.p.token, k, threads)
                candidates.append((est["engine_seconds"], rect))
            within = [c for c in candidates if c[0] <= TRIAL_ENGINE_SECONDS]
            engine, rect = (
                max(within, key=lambda c: c[0]) if within else min(candidates, key=lambda c: c[0])
            )
            _runner.run(first, units=[rect], num_threads=num_threads)  # warms kernels and caches
        finally:
            first.close()
        TableauStore.create(root / "measured", manifest).close()
        empty = size(root / "measured")
        second = TableauStore.open(root / "measured", write=True)
        try:
            start = time.perf_counter()
            _runner.run(second, units=[rect], num_threads=num_threads)
            measured = time.perf_counter() - start
            snapshots = second.counts()["snapshots"]
            # the stored bytes themselves: a small trial can fit in pages SQLite already has
            payload = sum(
                int(second._db().execute(sql).fetchone()[0] or 0)
                for sql in (
                    "SELECT sum(length(joint)) + sum(length(phys)) FROM snapshots",
                    "SELECT sum(length(data)) FROM observations",
                    "SELECT sum(length(k_points)) + sum(length(tau)) FROM traces",
                )
            )
        finally:
            second.close()  # a closed store holds everything in its one file
        full = size(root / "measured")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return {
        "unit": str(rect),
        "trajectories": rect.trajectories,
        "measured_seconds": measured,
        "estimated_engine_seconds": engine,
        "factor": measured / engine if engine > 0 else None,
        "bytes_per_snapshot": max(full - empty, payload) / snapshots if snapshots else None,
    }


def plan(store: Any, threads: int | None = None, benchmark: bool = False) -> dict[str, Any]:
    """The work still to do, with estimated runtime, memory and storage (nothing is run).

    The units are :func:`sparsegf2.study.runner.plan`'s (reruns first, then new trajectories,
    in a fixed order). Each unit's engine time is ``sparsegf2.api.estimate_ensemble`` of its
    engine spec on ``threads`` threads (default: the manifest's ``resources.num_threads``, else
    every thread). A unit also ingests its results (canonical forms, observables, one
    transaction), which the engine model leaves out, so the engine estimate is scaled by a
    **measured factor**: measured over estimated seconds of the units this store has already
    run, or, with ``benchmark=True``, of a trial unit run twice in a scratch store (a few
    seconds; the study is not touched). Storage is the remaining snapshots times the bytes per
    snapshot measured in the store (or in the trial). Memory per unit is the engine's result
    plus one canonical copy of its snapshots (a model).
    """
    import numba

    if threads is not None:
        threads = _int(threads, "threads", 1)
    benchmark = _flag(benchmark, "benchmark")
    with _opened(store) as s:
        revisions = s.revisions()
        manifest = revisions[-1]
        start = time.perf_counter()
        units = _runner.plan(s)
        plan_seconds = time.perf_counter() - start
        running = threads or manifest.resources.num_threads or numba.get_num_threads()
        running = min(int(running), int(numba.config.NUMBA_NUM_THREADS))
        estimator = _Estimator(s, manifest)
        rows = []
        by_condition: dict[_Cond, dict[str, Any]] = {}
        new_snapshots = 0
        reqs: dict[tuple[str, int, str], dict[str, Any]] = {}
        for unit in units:
            ident = (unit.protocol, unit.n, unit.beta.token)
            if ident not in reqs:
                reqs[ident] = _runner.requirements(revisions, unit.protocol, unit.n, unit.beta)
            req = reqs[ident]
            est = estimator.estimate(req, unit.p.token, unit.trajectories, running)
            snaps = _snapshot_count(s, unit, req)
            new_snapshots += snaps
            rows.append((unit, est, snaps))
            cond = _Cond(unit.protocol, unit.n, unit.beta.token, unit.p.token)
            agg = by_condition.setdefault(cond, {"units": 0, "trajectories": 0, "engine": 0.0})
            agg["units"] += 1
            agg["trajectories"] += unit.trajectories
            agg["engine"] += est["engine_seconds"]
        history = _history_factor(s, estimator, revisions)
        done_snapshots = s.counts()["snapshots"]
        storage = _storage(s.path, done_snapshots)
        directory = s.path
    trial = _trial(manifest, running, threads) if benchmark else None
    factor = history
    if trial is not None and trial["factor"] is not None:
        factor = {"value": trial["factor"], "source": "trial", "units_measured": 1}
    engine_total = sum(est["engine_seconds"] for _, est, _ in rows)
    scale = None if factor is None else float(factor["value"])
    per_snapshot = storage["bytes_per_snapshot"]
    snapshot_source = "store" if per_snapshot is not None else None
    if per_snapshot is None and trial is not None and trial["bytes_per_snapshot"] is not None:
        per_snapshot, snapshot_source = trial["bytes_per_snapshot"], "trial"
    largest = max(rows, key=lambda r: r[1]["memory_megabytes"], default=None)
    out: dict[str, Any] = {
        "store": str(directory),
        "experiment_id": manifest.experiment_id,
        "revision": len(revisions),
        **{f"{k}": v for k, v in _plan_counts(units).items()},
        "threads": running,
        "estimated_engine_seconds": engine_total,
        "factor": factor,
        "estimated_seconds": None if scale is None else engine_total * scale,
        "storage": {
            "bytes_per_snapshot": per_snapshot,
            "source": snapshot_source,
            "new_snapshots": new_snapshots,
            "estimated_megabytes": None
            if per_snapshot is None
            else new_snapshots * per_snapshot / 1e6,
            "store_megabytes": storage["megabytes"],
        },
        "peak_memory_megabytes": None if largest is None else largest[1]["memory_megabytes"],
        "largest_unit": None if largest is None else str(largest[0]),
        "units_shown": [
            {
                "unit": str(u),
                "trajectories": u.trajectories,
                "rerun": u.rerun,
                "engine_seconds": est["engine_seconds"],
                "estimated_seconds": None if scale is None else est["engine_seconds"] * scale,
                "memory_megabytes": est["memory_megabytes"],
                "new_snapshots": snaps,
            }  # fmt: skip
            for u, est, snaps in rows[:MAX_SHOWN_UNITS]
        ],
        "by_condition": [
            {
                **c.fields(),
                "units": a["units"],
                "trajectories": a["trajectories"],
                "estimated_seconds": a["engine"] if scale is None else a["engine"] * scale,
            }  # fmt: skip
            for c, a in list(by_condition.items())[:MAX_SHOWN_CONDITIONS]
        ],
        "plan_seconds": plan_seconds,
    }
    if trial is not None:
        out["trial"] = trial
    notes = []
    if scale is None:
        notes.append(
            "estimated_seconds is null: the engine estimate leaves out ingestion (canonical "
            "forms, observables, the transaction), which can take longer than the engine; pass "
            "benchmark=true, or run a few units (max_units) and plan again, for a measured factor"
        )
    if len(rows) > MAX_SHOWN_UNITS:
        notes.append(f"{MAX_SHOWN_UNITS} of {len(rows)} units shown, in the order they run")
    if len(by_condition) > MAX_SHOWN_CONDITIONS:
        notes.append(f"{MAX_SHOWN_CONDITIONS} of {len(by_condition)} conditions shown")
    if notes:
        out["notes"] = notes
    out["next_step"] = (
        "nothing to run"
        if not units
        else "run(store) (synchronous) or start_run_job(store) (background; job_status, "
        "cancel_job); max_units and max_seconds stop cleanly between units"
    )
    return _finish(out)


# ---------------------------------------------------------------------------------------------
# running: synchronous, and as a background job of sparsegf2.api's job machinery
# ---------------------------------------------------------------------------------------------


class _Stop(Exception):
    """Raised from the runner's progress callback (after a unit's commit) to stop between units."""


@dataclasses.dataclass
class _Progress:
    units_done: int = 0
    trajectories_done: int = 0
    snapshots: int = 0
    stopped: str = "complete"


def _execute(
    s: TableauStore,
    todo: Sequence[_runner.WorkUnit],
    plan_seconds: float,
    *,
    num_threads: int | None,
    deadline: float | None,
    cancel: threading.Event | None,
    state: _Progress,
    on_unit: Callable[[_runner.WorkUnit, Mapping[str, Any]], None] | None = None,
) -> None:
    """Run ``todo`` unit by unit, stopping between units when ``cancel`` is set or ``deadline``
    (``time.monotonic``) has passed.

    Units run in batches of :func:`sparsegf2.study.runner.run` calls holding
    ``sparsegf2.api._COMPUTE_LOCK`` (one parallel region at a time in this process, as for the
    ensemble jobs). Each call plans once more at its end, so a batch holds about
    ``20 x plan_seconds`` of work (at least one unit): the planning stays near 5% of the time.
    """
    i = 0
    batch = 1

    def progress(unit: _runner.WorkUnit, index: int, total: int, info: Mapping[str, Any]) -> None:
        state.units_done += 1
        state.trajectories_done += unit.trajectories
        state.snapshots += int(info["snapshots"])
        if on_unit is not None:
            on_unit(unit, info)
        if cancel is not None and cancel.is_set():
            raise _Stop("cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            raise _Stop("max_seconds")

    while i < len(todo):
        if cancel is not None and cancel.is_set():
            state.stopped = "cancelled"
            return
        if deadline is not None and time.monotonic() >= deadline:
            state.stopped = "max_seconds"
            return
        chunk = list(todo[i : i + batch])
        start = time.perf_counter()
        with _api._COMPUTE_LOCK:
            try:
                _runner.run(s, num_threads=num_threads, units=chunk, progress=progress)
            except _Stop as stop:
                state.stopped = str(stop.args[0])
                return
        i += len(chunk)
        per_unit = (time.perf_counter() - start) / len(chunk)
        batch = int(min(256, max(1, math.ceil(20.0 * plan_seconds / max(per_unit, 1e-6)))))


def _run_args(max_units: Any, max_seconds: Any, num_threads: Any) -> tuple[Any, Any, Any]:
    return (
        _optional_int(max_units, "max_units", 0),
        _positive_seconds(max_seconds, "max_seconds"),
        _optional_int(num_threads, "num_threads", 1),
    )


def run(
    store: Any,
    *,
    max_units: int | None = None,
    max_seconds: float | None = None,
    num_threads: int | None = None,
    progress: Callable[[_runner.WorkUnit, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the store's plan in this process and return when it is done or a budget is spent.

    Every unit runs in one engine call and is stored in one transaction (design §4): an
    interruption leaves the store as it was after the last finished unit, and running again
    resumes. ``max_units`` and ``max_seconds`` stop **between** units (a unit that started
    finishes); ``stopped`` says why the call returned (``complete``, ``max_units``,
    ``max_seconds``). ``num_threads`` (default: the manifest's, else every thread) never
    changes a result. ``progress(unit, info)`` (Python only) is called after each unit.
    """
    max_units, max_seconds, num_threads = _run_args(max_units, max_seconds, num_threads)
    with _opened(store, write=True) as s:
        start = time.perf_counter()
        todo = _runner.plan(s)
        plan_seconds = time.perf_counter() - start
        total = len(todo)
        if max_units is not None:
            todo = todo[:max_units]
        state = _Progress()
        deadline = None if max_seconds is None else time.monotonic() + max_seconds
        _execute(s, todo, plan_seconds, num_threads=num_threads, deadline=deadline, cancel=None,
                 state=state, on_unit=progress)  # fmt: skip
        if state.stopped == "complete" and len(todo) < total:
            state.stopped = "max_units"
        left = _runner.plan(s)
        directory = s.path
    elapsed = time.perf_counter() - start
    counts = _plan_counts(left)
    return _finish(
        {
            "store": str(directory),
            "stopped": state.stopped,
            "complete": not left,
            "units_run": state.units_done,
            "trajectories_run": state.trajectories_done,
            "snapshots_written": state.snapshots,
            "elapsed_seconds": elapsed,
            "units_remaining": counts["units"],
            "trajectories_remaining": counts["trajectories"],
            "next_step": "aggregate(store, observable) and fss(store, ...)"
            if not left
            else "run(store) again resumes; status(store) shows what is left",
        }
    )


@dataclasses.dataclass
class _StudyJob(_api._Job):
    """A study run in a background thread, kept with the ensemble jobs of :mod:`sparsegf2.api`
    so ``job_status``, ``cancel_job`` and ``list_jobs`` follow it. ``trajectories`` and
    ``done`` count trajectories; the store is ``output_dir``."""

    units_total: int = 0
    units_done: int = 0
    max_units: int | None = None
    max_seconds: float | None = None
    stopped: str | None = None
    units_remaining: int | None = None
    trajectories_remaining: int | None = None
    opened: threading.Event = dataclasses.field(default_factory=threading.Event)
    open_error: BaseException | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            elapsed = ((self.finished or now) - self.started) if self.started else 0.0
            rate = self.done / self.compute_seconds if self.compute_seconds > 0 else None
            left = self.trajectories - self.done
            eta = left / rate if rate and self.status in ("queued", "running") else None
            out: dict[str, Any] = {
                "job_id": self.job_id,
                "kind": JOB_KIND,
                "status": self.status,
                "store": str(self.output_dir),
                "output_dir": str(self.output_dir),
                "units_done": self.units_done,
                "units_total": self.units_total,
                "trajectories_done": self.done,
                "trajectories_total": self.trajectories,
                "progress": self.done / self.trajectories if self.trajectories else 1.0,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "trajectories_per_second": rate,
                "max_units": self.max_units,
                "max_seconds": self.max_seconds,
            }
            if self.stopped is not None:
                out["stopped"] = self.stopped
                out["units_remaining"] = self.units_remaining
                out["trajectories_remaining"] = self.trajectories_remaining
            if self.error:
                out["error"] = self.error
            if self.notes:
                out["notes"] = list(self.notes)
            if self.status in ("completed", "stopped", "cancelled", "failed"):
                out["next_step"] = (
                    f"status({str(self.output_dir)!r}); start_run_job resumes what is left"
                    if self.status != "completed"
                    else f"aggregate({str(self.output_dir)!r}, observable) and fss(...)"
                )
            return out


def _study_job(job: _StudyJob) -> None:
    """The job's thread: open the store as its writer, run, close (which releases the lock)."""
    try:
        s = TableauStore.open(job.output_dir, write=True)
    except BaseException as exc:  # reported to start_run_job, which raises it
        job.open_error = exc
        with job.lock:
            job.status, job.finished = "failed", time.time()
            job.error = f"{type(exc).__name__}: {exc}"
        job.opened.set()
        return
    status, error = "failed", None
    state = _Progress()
    try:
        start = time.perf_counter()
        todo = _runner.plan(s)
        plan_seconds = time.perf_counter() - start
        total = len(todo)
        if job.max_units is not None:
            todo = todo[: job.max_units]
        with job.lock:
            job.units_total = len(todo)
            job.trajectories = sum(u.trajectories for u in todo)
            job.started = time.time()
            job.status = "running"
        job.opened.set()

        def on_unit(unit: _runner.WorkUnit, info: Mapping[str, Any]) -> None:
            with job.lock:
                job.units_done += 1
                job.done += unit.trajectories
                job.compute_seconds += float(info["elapsed_seconds"])

        deadline = None if job.max_seconds is None else time.monotonic() + job.max_seconds
        _execute(s, todo, plan_seconds, num_threads=job.num_threads, deadline=deadline,
                 cancel=job.cancel, state=state, on_unit=on_unit)  # fmt: skip
        if state.stopped == "complete" and len(todo) < total:
            state.stopped = "max_units"
        left = _runner.plan(s)
        counts = _plan_counts(left)
        with job.lock:
            job.units_remaining = counts["units"]
            job.trajectories_remaining = counts["trajectories"]
        if not left:
            status = "completed"
        elif state.stopped == "cancelled":
            status = "cancelled"
        else:
            status = "stopped"
    except Exception as exc:  # reported through job_status
        error = f"{type(exc).__name__}: {exc}\n" + "".join(traceback.format_exception(exc)[-3:])
    finally:
        s.close()  # releases the lock first: the next writer may start as soon as we report
        with job.lock:
            job.status, job.error, job.finished = status, error, time.time()
            job.stopped = state.stopped if status != "failed" else None
        job.opened.set()


def start_run_job(
    store: Any,
    *,
    max_units: int | None = None,
    max_seconds: float | None = None,
    num_threads: int | None = None,
) -> dict[str, Any]:
    """Run the store's plan in a background thread; returns at once with a ``job_id``.

    ``sparsegf2.api.job_status(job_id, wait_seconds)`` reports progress (units and trajectories
    done, rate, ETA), ``cancel_job(job_id)`` stops it after the unit that is running, and
    ``list_jobs()`` lists it with the ensemble jobs. Its ``status`` ends ``completed`` (no work
    left), ``stopped`` (``max_units`` or ``max_seconds`` spent; ``stopped`` says which),
    ``cancelled`` or ``failed`` (with ``error``). The job holds the store's writer lock until it
    ends; a store another writer holds is refused here, with the holder named. Jobs live in this
    process; after a restart, call it again: every finished unit is in the store.
    """
    max_units, max_seconds, num_threads = _run_args(max_units, max_seconds, num_threads)
    directory = _store_dir(store)
    if not (directory / STORE_FILE).is_file():
        raise ValueError(f"{str(directory)!r} is not a study store (no {STORE_FILE} in it)")
    _api._refuse_a_running_job(directory)
    job = _StudyJob(
        job_id=uuid.uuid4().hex[:12],
        spec={},
        trajectories=0,
        seed=0,
        output_dir=directory,
        chunk_size=0,
        num_threads=num_threads,
        created=time.time(),
        max_units=max_units,
        max_seconds=max_seconds,
    )
    job.thread = threading.Thread(
        target=_study_job, args=(job,), name=f"sparsegf2-study-{job.job_id}", daemon=True
    )
    job.thread.start()
    job.opened.wait()  # the thread holds the store's lock, or could not take it
    if job.open_error is not None:  # never listed: the refusal names the real holder
        error = job.open_error
        if isinstance(error, ValueError):
            raise ValueError(str(error)) from None
        raise error
    with _api._JOBS_LOCK:
        _api._JOBS[job.job_id] = job
    return _finish(job.snapshot())


# ---------------------------------------------------------------------------------------------
# extend, compute, export, verify
# ---------------------------------------------------------------------------------------------


def extend(store: Any, manifest: Any) -> dict[str, Any]:
    """Add ``manifest`` as the store's next revision (design §3: it may add conditions, rates,
    graphs, circuits up to the stride, raise ``T_max``, change the checkpoint rule, add
    observables; it may not change seeds, the gate mode, the stride or a recorded observable's
    settings, and the first forbidden change is named). The latest manifest again changes
    nothing. Nothing runs: the result is the new plan's size; :func:`run` executes it, rerunning
    stored trajectories from layer 0 where a revision asks more of them (design D3)."""
    m = _manifest(manifest)
    with _opened(store, write=True) as s:
        before = s.revision
        units = _runner.extend(s, m)
        after = s.revision
        directory = s.path
    counts = _plan_counts(units)
    return _finish(
        {
            "store": str(directory),
            "revision": after,
            "changed": after != before,
            "manifest_digest": m.digest(),
            "units_remaining": counts["units"],
            "trajectories_remaining": counts["trajectories"],
            "rerun_units": counts["rerun_units"],
            "rerun_trajectories": counts["rerun_trajectories"],
            "warnings": list(m.warnings),
            "next_step": "plan(store), then run(store) or start_run_job(store)"
            if units
            else "nothing to run",
        }
    )


def compute(
    store: Any,
    observable: str,
    params: Mapping[str, Any] | None = None,
    *,
    protocol: str = "F",
    n: int | None = None,
    beta: Any = None,
    p: Any = None,
    t: int | None = None,
) -> dict[str, Any]:
    """Compute a retrospective observable of the catalog over the stored snapshots (design
    §4), without simulating: ``k``, ``code_rate``, ``stabilizer_rank``, ``d_cyc``, ``d_conn``,
    ``half_cut_entropy``, ``mean_leakage``. ``params`` are the observable's (defaults from the
    catalog); ``n``, ``beta``, ``p``, ``t`` restrict the snapshots. Values already stored under
    the same key are compared, never replaced, so a second call is a cross-check that inserts
    nothing."""
    observable = _text(observable, "observable")
    if params is not None and not isinstance(params, Mapping):
        raise ValueError(f"params must be an object of the observable's parameters, got {params!r}")
    start = time.perf_counter()
    with _opened(store, write=True) as s, _lookups():
        summary = _runner.compute(
            s, observable, params, protocol=check_protocol(_text(protocol, "protocol")),
            n=_optional_int(n, "n", 1), beta=beta, p=p, t=_optional_int(t, "t", 0),
        )  # fmt: skip
    return _finish(
        {
            "observable": summary.observable,
            "version": summary.version,
            "params": json.loads(summary.params),
            "inserted": summary.inserted,
            "existing": summary.existing,
            "snapshots": summary.snapshots,
            "elapsed_seconds": time.perf_counter() - start,
            "next_step": f"aggregate(store, {summary.observable!r}, params=...)",
        }
    )


def export(store: Any, directory: Any) -> dict[str, Any]:
    """Write the store's data as open tables (CSV and NPZ, format
    ``sparsegf2.study.store.EXPORT_FORMAT``) into the new directory ``directory``:
    :meth:`TableauStore.export`. An interrupted export leaves nothing."""
    if not isinstance(directory, (str, os.PathLike)) or not str(directory).strip():
        raise ValueError(f"directory must be a path, got {directory!r}")
    target = Path(directory).expanduser().resolve()
    with _opened(store) as s:
        path = s.export(target)
    files = sorted(p for p in path.rglob("*") if p.is_file())
    return _finish(
        {
            "directory": str(path),
            "format": EXPORT_FORMAT,
            "files": len(files),
            "megabytes": sum(p.stat().st_size for p in files) / 1e6,
            "conditions": sum(1 for p in path.iterdir() if p.is_dir()),
            "top_level": sorted(p.name for p in path.iterdir() if p.is_file()),
        }
    )


def verify(
    store: Any, sample: int | None = 8, seed: int = 0, *, num_threads: int | None = None
) -> dict[str, Any]:
    """Replay a sample of stored trajectories and compare everything (v1.4 §7.4, validation 5):
    :func:`sparsegf2.study.runner.verify`. ``sample`` trajectories (``None``: all) are drawn
    with ``seed``; each one's graph is regenerated, every stored snapshot decoded against its
    digest, the trajectory rerun (k trace and every stored digest must be reproduced) and, for
    protocol F, the engine's own ``code_snapshot`` compared with the stored physical code.
    ``ok`` is false when anything differs; ``failures`` says what."""
    sample = _optional_int(sample, "sample", 1)
    seed = _int(seed, "seed", 0)
    num_threads = _optional_int(num_threads, "num_threads", 1)
    start = time.perf_counter()
    with _opened(store) as s, _api._COMPUTE_LOCK:
        result = _runner.verify(s, sample, seed, num_threads=num_threads)
    failures = list(result.failures)
    return _finish(
        {
            "ok": result.ok,
            "trajectories": result.trajectories,
            "snapshots": result.snapshots,
            "failures": failures[:MAX_SHOWN_FAILURES],
            "failures_total": len(failures),
            "sample": sample,
            "seed": seed,
            "elapsed_seconds": time.perf_counter() - start,
        }
    )


# ---------------------------------------------------------------------------------------------
# tableau
# ---------------------------------------------------------------------------------------------

_PAULI = np.array(["_", "X", "Z", "Y"])


def _pauli_strings(tab: PhaseFreeTableau) -> list[str]:
    """The canonical generators as Pauli strings (``_`` for the identity), qubit 0 first."""
    rows = tab.canonical()
    n = tab.num_qubits
    if rows.shape[0] == 0:
        return []
    letters = _PAULI[rows[:, :n].astype(np.int64) + 2 * rows[:, n:].astype(np.int64)]
    return ["".join(row) for row in letters]


def _tableau_file(path: Path, tab: PhaseFreeTableau, meta: Mapping[str, Any]) -> None:
    if path.suffix.lower() == ".npz":
        np.savez_compressed(
            path, canonical=tab.canonical(), words=np.asarray(tab.words),
            num_qubits=np.int64(tab.num_qubits), metadata=np.array(json.dumps(_exact(meta))),
        )  # fmt: skip
    else:
        path.write_bytes(tab.to_bytes())


def tableau(
    store: Any,
    *,
    n: int,
    beta: Any,
    p: Any,
    g: int,
    c: int,
    t: int,
    protocol: str = "F",
    replay: bool = False,
    out: Any = None,
) -> dict[str, Any]:
    """One trajectory's phase-free tableau at depth ``t`` (v1.4 §5.3), by its parameter tuple.

    By default the stored snapshot (no simulation): the joint row space on ``N = 2n`` qubits
    (system ``0..n-1``, then reference ``n + i`` paired with system ``i``; ``N = n`` for
    protocol P) in canonical form, its rank, ``k`` and SHA-256 digest of the canonical bytes, and
    the physical code (rank ``n - k``). ``replay=True`` reruns the trajectory from layer 0 to
    any depth (stored or not) and checks the result against every stored digest it reaches
    (``verified``). The generators are listed as Pauli strings for ``N <= 32``. ``out`` writes
    the joint tableau (or, for a store that kept physical codes only and no replay, the
    physical code): a ``.npz`` file (canonical 0/1 rows, packed words, metadata) or, for any
    other name, the canonical bytes (format ``SGF2RS1``, ``docs/study/03``).
    """
    n = _int(n, "n", 1)
    g, c, t = _int(g, "g"), _int(c, "c"), _int(t, "t")
    replay = _flag(replay, "replay")
    protocol = check_protocol(_text(protocol, "protocol"))
    with _opened(store) as s, _lookups():
        key = s.key(n=n, beta=beta, p=p, g=g, c=c, protocol=protocol)
        s.trajectory(key)
        stored_depths = list(s.depths(key))
        snap = s.snapshot(key, t) if t in stored_depths else None
        verified: dict[str, Any] | None = None
        joint: PhaseFreeTableau | None
        phys: PhaseFreeTableau | None = None
        if replay:
            with _api._COMPUTE_LOCK:
                # one trajectory: more threads cannot help it
                rep = _runner.replay(s, key, [t], verify=True, num_threads=1)  # a mismatch raises
            joint = rep.tableaux[t]
            source = "replay"
            verified = {
                # True: the replayed tableau has the stored digest at t; None: t is not stored
                "digest_matches_store": True if t in rep.verified_depths else None,
                "k_trace_through_layer": rep.verified_layers if rep.k is not None else None,
            }
            if protocol == "F":
                phys = joint.physical_tableau(n)
        else:
            if snap is None:
                raise ValueError(
                    f"trajectory {key} has no snapshot at t = {t}; it holds the depths "
                    f"{stored_depths}: pass replay=true to rerun it to any depth"
                )
            if snap.status == SNAPSHOT_ABSENT:
                raise ValueError(
                    f"trajectory {key} stopped when its reference purified, so nothing is stored "
                    f"at t = {t} (k = 0 there): pass replay=true for the continued state"
                )
            joint = snap.tableau() if snap.joint_blob is not None else None
            phys = snap.physical_tableau() if protocol == "F" else None
            source = "store"
        status_ = None if snap is None else snap.status
    summary: dict[str, Any] = {
        "key": str(key),
        "t": t,
        "source": source,
        "snapshot_status": status_,
        "qubit_order": QUBIT_ORDER,
    }
    if joint is not None:
        summary["joint"] = {
            "num_qubits": joint.num_qubits,
            "rank": joint.rank,
            "pure": joint.is_pure(),
            "digest": joint.digest(),
        }
    else:
        summary["joint"] = None
    if phys is not None:
        summary["k"] = n - phys.rank
        summary["physical"] = {"num_qubits": n, "rank": phys.rank, "digest": phys.digest()}
    if verified is not None:
        summary["verified"] = verified
    shown = joint if joint is not None else phys
    notes = []
    if joint is None:
        notes.append(
            "this store kept physical codes only (observables.tableau_snapshots = false): "
            "replay=true gives the joint state"
        )
    if shown is not None and shown.num_qubits <= _api.MAX_STABILIZER_QUBITS:
        summary["generators"] = _pauli_strings(shown)
        if phys is not None and joint is not None:
            summary["physical_generators"] = _pauli_strings(phys)
    elif shown is not None:
        notes.append(
            f"generators are listed for at most {_api.MAX_STABILIZER_QUBITS} qubits; pass out "
            "to write the tableau to a file"
        )
    if out is not None and shown is not None:
        if not isinstance(out, (str, os.PathLike)) or not str(out).strip():
            raise ValueError(f"out must be a file path, got {out!r}")
        path = Path(out).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {k: v for k, v in summary.items() if k != "generators"}
        meta["written"] = "joint" if joint is not None else "physical"
        _tableau_file(path, shown, meta)
        summary["file"] = str(path)
        summary["file_format"] = "npz" if path.suffix.lower() == ".npz" else "SGF2RS1"
    if notes:
        summary["notes"] = notes
    return _finish(summary)


# ---------------------------------------------------------------------------------------------
# aggregate: the internal computation (full arrays) and its JSON rendering
# ---------------------------------------------------------------------------------------------


def _aggregate_kind(entry: _ob.ObservableDef) -> str:
    if entry.id == "k_trace":
        return "k_trace"
    if entry.id == "first_passage":
        return "first_passage"
    if entry.id.startswith("peak_"):
        return "peak"
    if entry.level == "snapshot":
        return "snapshot"
    raise ValueError(
        f"observable {entry.id!r} (level {entry.level}) is not aggregated per condition; "
        "aggregate snapshot observables, first_passage, k_trace or a peak_* observable"
    )


def _ells(ell: Any) -> list[int]:
    if ell is None:
        return [0, 1]
    values = list(ell) if isinstance(ell, (list, tuple)) else [ell]
    if not values:
        raise ValueError("ell must be a threshold or a non-empty list of thresholds")
    return sorted({_int(v, "ell", 0) for v in values})


def _no_arg(name: str, value: Any, kind: str, default: Any = None) -> None:
    if value != default:
        raise ValueError(f"{name} does not apply to {kind} observables; leave it out")


def _snapshot_kind(
    s: TableauStore,
    observable: str,
    version: str,
    params: str,
    conds: Sequence[_Cond],
    depths: set[int] | None,
    field: str | None,
    index: int | None,
    nan_policy: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    out: list[dict[str, Any]] = []
    empty: list[str] = []
    for cond in conds:
        rows = s.observations(observable, version=version, params=params, protocol=cond.protocol,
                              n=cond.n, beta=cond.beta, p=cond.p, field=field)  # fmt: skip
        if rows["g"].size == 0:
            empty.append(cond.label())
            continue
        values = rows[field] if field is not None else rows["value"]
        if values.ndim > 1:
            if index is None:
                raise ValueError(
                    f"field {field!r} of {observable!r} is a list of {values.shape[1]} entries "
                    f"per value: give index (0..{values.shape[1] - 1})"
                )
            if not 0 <= index < values.shape[1]:
                raise ValueError(f"index must be in 0..{values.shape[1] - 1}, got {index}")
            values = values[:, index]
        elif index is not None:
            raise ValueError(f"index applies to a list field only; {field or 'value'} is a number")
        status = rows["status"]
        bad = ~np.isfinite(values)
        if field is None:
            bad |= status != "ok"
        ts = sorted({int(t) for t in rows["t"].tolist()} & (depths or set(rows["t"].tolist())))
        if field is None and (status == "vector").all():
            data = s.observation(None, int(rows["t"][0]), observable, version=version,
                                 params=params, n=cond.n, beta=cond.beta, p=cond.p,
                                 g=int(rows["g"][0]), c=int(rows["c"][0]),
                                 protocol=cond.protocol)["data"]  # fmt: skip
            raise ValueError(
                f"{observable!r} stores a vector per value (status 'vector'): choose field (one "
                f"of {sorted(data or {})}) and, for a list, index"
            )
        means, ses, graphs_used, traj_used, ests = [], [], [], [], []
        for t in ts:
            sel = rows["t"] == t
            if bad[sel].any() and nan_policy == "raise":
                kinds = Counter(status[sel & bad].tolist())
                hint = (
                    "; for d_cyc, field 'zero_extended' is the unconditional trace (0 without a "
                    "code, v1.4 §5.5)"
                    if observable == "d_cyc"
                    else ""
                )
                raise ValueError(
                    f"{int((sel & bad).sum())} values of {observable!r} at {cond.label()}, t = {t} "
                    f"are not finite numbers (status {dict(kinds)}): nan_policy='omit' leaves them "
                    "out (the estimand then changes: an equal-graph mean over the graphs with a "
                    f"value of each graph's mean over its defined values){hint}"
                )
            est = _stats.nested_mean(
                rows["g"][sel],
                rows["c"][sel],
                rows["r"][sel],
                np.where(bad[sel], np.nan, values[sel]),
                nan_policy=cast(Any, nan_policy),
            )
            ests.append(est)
            means.append(float(est.mean))
            ses.append(float(est.se))
            graphs_used.append(int(est.contributing_graphs))
            traj_used.append(int(est.contributing_trajectories))
        out.append(
            {
                **cond.fields(),
                "depths": np.array(ts, dtype=np.int64),
                "mean": np.array(means),
                "se": np.array(ses),
                "graphs": int(len(set(rows["g"].tolist()))),
                "trajectories": int(
                    len(
                        set(
                            zip(
                                rows["g"].tolist(),
                                rows["c"].tolist(),
                                rows["r"].tolist(),
                                strict=True,
                            )
                        )
                    )
                ),  # fmt: skip
                "contributing_graphs": np.array(graphs_used, dtype=np.int64),
                "contributing_trajectories": np.array(traj_used, dtype=np.int64),
                "status_counts": dict(Counter(status.tolist())),
                "_estimates": ests,
            }
        )
    return out, empty


def _first_passage_rows(
    s: TableauStore, cond: _Cond
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray] | None:
    """``rows``, ``tau`` ``(N, n+1)`` (NaN where censored), ``observed`` (bool) and the
    censoring depth per trajectory, from the stored first-passage records."""
    args: dict[str, Any] = dict(protocol=cond.protocol, n=cond.n, beta=cond.beta, p=cond.p)
    rows = s.observations("first_passage", field="tau", **args)
    if rows["g"].size == 0:
        return None
    observed = s.observations("first_passage", field="observed", **args)
    depth = s.observations("first_passage", field="censor_depth", **args)
    for other in (observed, depth):
        if not (np.array_equal(other["g"], rows["g"]) and np.array_equal(other["c"], rows["c"])):
            raise RuntimeError("first-passage fields came back in different orders")
    return rows, rows["tau"], observed["observed"].astype(bool), depth["censor_depth"]


def _survival(
    rows: Mapping[str, np.ndarray],
    tau: np.ndarray,
    observed: np.ndarray,
    depth: np.ndarray,
    ell: int,
) -> tuple[_stats.SurvivalCurve, _stats.CensoringSummary, np.ndarray, np.ndarray]:
    seen = observed[:, ell]
    times = np.where(seen, tau[:, ell], depth)
    curve = _stats.nested_kaplan_meier(rows["g"], rows["c"], rows["r"], times, seen)
    cens = _stats.censoring_summary(rows["g"], rows["c"], rows["r"], times, seen)
    return curve, cens, times, seen


def _first_passage_kind(
    s: TableauStore, conds: Sequence[_Cond], ells: Sequence[int]
) -> tuple[list[dict[str, Any]], list[str]]:
    out: list[dict[str, Any]] = []
    empty: list[str] = []
    for cond in conds:
        got = _first_passage_rows(s, cond)
        if got is None:
            empty.append(cond.label())
            continue
        rows, tau, observed, depth = got
        entries = []
        for ell in ells:
            if ell > cond.n:
                continue
            curve, cens, times, seen = _survival(rows, tau, observed, depth, ell)
            entries.append(
                {
                    "ell": ell,
                    "trajectories": cens.trajectories,
                    "graphs": cens.graphs,
                    "censored": cens.censored,
                    "censored_fraction": cens.censored_fraction,
                    "censored_fraction_se": cens.censored_fraction_se,
                    "horizon": float(np.max(depth)),
                    "median": curve.median,
                    "median_upper": curve.median_upper,
                    "restricted_mean": curve.restricted_mean,
                    "restricted_mean_se": curve.restricted_mean_se,
                    "restricted_horizon": curve.restricted_horizon,
                    "mean": curve.mean,
                    "mean_se": curve.mean_se,
                    "_curve": curve,
                    "_times": times,
                    "_observed": seen,
                }
            )
        out.append({**cond.fields(), "ells": entries, "_rows": rows})
    return out, empty


def _trace_matrix(
    trace_rows: Mapping[str, np.ndarray],
    values: np.ndarray,
    g: np.ndarray,
    c: np.ndarray,
    r: np.ndarray,
    checkpoints: np.ndarray,
) -> np.ndarray:
    """The trace ``d(t)`` of each trajectory ``(g, c, r)`` at ``checkpoints`` (NaN where not
    stored)."""
    lookup: dict[tuple[int, int, int, int], float] = {}
    for gg, cc, rr, tt, v in zip(trace_rows["g"].tolist(), trace_rows["c"].tolist(),
                                 trace_rows["r"].tolist(), trace_rows["t"].tolist(),
                                 values.tolist(), strict=True):  # fmt: skip
        lookup[(gg, cc, rr, tt)] = v
    return np.array(
        [
            [lookup.get((int(a), int(b), int(q), int(t)), math.nan) for t in checkpoints.tolist()]
            for a, b, q in zip(g.tolist(), c.tolist(), r.tolist(), strict=True)
        ],  # fmt: skip
        dtype=np.float64,
    ).reshape(g.size, checkpoints.size)


def _peak_kind(
    s: TableauStore,
    observable: str,
    version: str,
    params: str,
    conds: Sequence[_Cond],
    horizon_policy: str,
    probe: str | None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    out: list[dict[str, Any]] = []
    empty: list[str] = []
    notes: list[str] = []
    trace_id, trace_field = _PEAK_TRACES.get(observable, ("", None))
    for cond in conds:
        args: dict[str, Any] = dict(version=version, params=params, protocol=cond.protocol, n=cond.n,
                    beta=cond.beta, p=cond.p)  # fmt: skip
        rows = s.observations(observable, **args)
        if rows["g"].size == 0:
            empty.append(cond.label())
            continue
        at_h = s.observations(observable, field="at_horizon", **args)["at_horizon"]
        domains = s.observations(observable, field="domain", **args)["domain"]
        keep = np.ones(rows["g"].size, dtype=bool)
        names = np.array(
            [str(x).split("/", 1)[1] if "/" in str(x) else "" for x in rows["context"]]
        )
        if len(set(names.tolist())) > 1:
            if probe is None:
                raise ValueError(
                    f"{observable!r} is stored for several correlated probes {sorted(set(names))}: "
                    "give probe"
                )
            keep = names == probe
            if not keep.any():
                raise ValueError(f"probe must be one of {sorted(set(names))}, got {probe!r}")
        g, c, r = rows["g"][keep], rows["c"][keep], rows["r"][keep]
        tstar = rows["value"][keep]
        horizon = at_h[keep].astype(bool)
        domain = domains[keep]
        if not (domain == domain[0]).all():
            raise RuntimeError("the peaks of one condition were taken over different domains")
        checkpoints = domain[0].astype(np.int64)
        traces = None
        if trace_id:
            traces = _peak_traces(s, observable, trace_id, trace_field, cond, params, names[keep],
                                  g, c, r, checkpoints)  # fmt: skip
            if traces is None and cond == conds[0]:
                notes.append(f"no stored {trace_id} trace: argmax_t E[d(t)] is not computed")
        summary = _stats.peak_time_summaries(
            g,
            c,
            r,
            tstar,
            at_horizon=horizon,
            traces=traces,
            checkpoints=None if traces is None else checkpoints,
            horizon_policy=cast(Any, horizon_policy),
            trace_nan_policy="omit",
        )
        arith = summary.arithmetic
        out.append(
            {
                **cond.fields(),
                "trajectories": int(g.size),
                "missing_fraction": summary.missing_fraction,
                "horizon_fraction": summary.horizon_fraction,
                "horizon_policy": horizon_policy,
                "arithmetic_mean": None if arith is None else float(arith.mean),
                "arithmetic_se": None if arith is None else float(arith.se),
                "contributing_graphs": None if arith is None else int(arith.contributing_graphs),
                "geometric_mean": summary.geometric,
                "geometric_se": summary.geometric_se,
                "argmax": summary.argmax,
                "argmax_set": None if summary.argmax_set is None else summary.argmax_set,
                "argmax_at_horizon": summary.argmax_at_horizon,
                "domain": checkpoints,
                "note": summary.note,
                "_summary": summary,
                "_g": g,
                "_c": c,
                "_r": r,
                "_t_star": tstar,
                "_at_horizon": horizon,
            }
        )
    return out, empty, notes


def _peak_traces(
    s: TableauStore,
    observable: str,
    trace_id: str,
    trace_field: str | None,
    cond: _Cond,
    peak_params: str,
    names: np.ndarray,
    g: np.ndarray,
    c: np.ndarray,
    r: np.ndarray,
    checkpoints: np.ndarray,
) -> np.ndarray | None:
    """The traces ``d(t)`` behind a peak observable: ``d_cyc`` zero-extended, ``D``, the
    coupling's column of ``D_J``'s robustness, ``d_conn``."""
    choices = [(v, p) for o, v, p in _stored_keys(s) if o == trace_id]
    if trace_id == "D_J":
        name = str(names[0]) if names.size else ""
        choices = [(v, p) for v, p in choices if json.loads(p).get("name") == name]
    if len(choices) != 1:
        return None
    version, params = choices[0]
    args: dict[str, Any] = dict(version=version, params=params, protocol=cond.protocol, n=cond.n,
                beta=cond.beta, p=cond.p)  # fmt: skip
    rows = s.observations(trace_id, field=trace_field, **args)
    if rows["g"].size == 0:
        return None
    values = rows[trace_field] if trace_field is not None else rows["value"]
    if trace_id == "D_J":
        coupling = float(json.loads(peak_params)["J"])
        js = s.observations(trace_id, field="couplings", **args)["couplings"][0]
        where = np.flatnonzero(js == coupling)
        if where.size != 1:
            return None
        values = values[:, int(where[0])]
    return _trace_matrix(rows, values, g, c, r, checkpoints)


def _k_trace_kind(
    s: TableauStore, conds: Sequence[_Cond], normalized: bool
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    out: list[dict[str, Any]] = []
    empty: list[str] = []
    notes: list[str] = []
    for cond in conds:
        if cond.protocol != "F":
            continue
        keys = s.trajectory_keys(protocol=cond.protocol, n=cond.n, beta=cond.beta, p=cond.p)
        if not keys:
            empty.append(cond.label())
            continue
        traces = [s.k_trace(key) for key in keys]
        length = min(len(k) for k in traces)
        if any(len(k) != length for k in traces):
            notes.append(
                f"{cond.label()}: trajectories hold traces of different horizons (a revision "
                f"raised T_max and not all were rerun): the common prefix 0..{length - 1} is used"
            )
        matrix = np.array([k[:length] for k in traces], dtype=np.float64)
        if normalized:
            matrix = matrix / cond.n
        est = _stats.nested_mean([k.g for k in keys], [k.c for k in keys], [k.r for k in keys],
                                 matrix)  # fmt: skip
        out.append(
            {
                **cond.fields(),
                "depths": np.arange(length, dtype=np.int64),
                "mean": np.asarray(est.mean, dtype=np.float64),
                "se": np.asarray(est.se, dtype=np.float64),
                "graphs": est.num_graphs,
                "trajectories": est.num_trajectories,
                "_estimate": est,
            }
        )
    return out, empty, notes


def _paired(results: Sequence[Mapping[str, Any]], rates: Sequence[str]) -> list[dict[str, Any]]:
    """Within-graph differences ``value(p2) - value(p1)`` per ``(protocol, n, beta)`` and depth,
    with the graph-cluster error of the differences (v1.4 §6.4)."""
    p1, p2 = rates
    by: dict[tuple[str, int, str], dict[str, Mapping[str, Any]]] = {}
    for res in results:
        by.setdefault((res["protocol"], res["n"], res["beta"]), {})[res["p"]] = res
    out = []
    for (protocol, n, beta), pair in by.items():
        if p1 not in pair or p2 not in pair:
            continue
        a, b = pair[p1], pair[p2]
        common = sorted(set(a["depths"].tolist()) & set(b["depths"].tolist()))
        ia = {int(t): i for i, t in enumerate(a["depths"].tolist())}
        ib = {int(t): i for i, t in enumerate(b["depths"].tolist())}
        means, ses, graphs = [], [], []
        for t in common:
            diff = _stats.paired_difference(a["_estimates"][ia[t]], b["_estimates"][ib[t]])
            means.append(float(diff.mean))
            ses.append(float(diff.se))
            graphs.append(int(diff.contributing_graphs))
        out.append(
            {
                "protocol": protocol,
                "n": n,
                "beta": beta,
                "p": [p1, p2],
                "depths": np.array(common, dtype=np.int64),
                "difference": np.array(means),
                "se": np.array(ses),
                "graphs": np.array(graphs, dtype=np.int64),
            }  # fmt: skip
        )
    return out


def _aggregate(
    s: TableauStore,
    observable: str,
    params: Any = None,
    where: Any = None,
    *,
    version: str | None = None,
    field: str | None = None,
    index: int | None = None,
    nan_policy: str = "raise",
    ell: Any = None,
    horizon_policy: str = "refuse",
    paired_rates: Any = None,
    normalized: bool = False,
    probe: str | None = None,
) -> dict[str, Any]:
    observable = _text(observable, "observable")
    with _lookups():
        entry = _ob.get(observable, version)
    kind = _aggregate_kind(entry)
    filters = _where(where)
    depths = filters.pop("t", None)
    manifest = s.manifest
    conds = _conditions(manifest, filters)
    index = _optional_int(index, "index", 0)
    if field is not None:
        field = _text(field, "field")
    nan_policy = _choice(nan_policy, "nan_policy", ("raise", "omit"))
    horizon_policy = _choice(horizon_policy, "horizon_policy", ("refuse", "exclude", "include"))
    normalized = _flag(normalized, "normalized")
    result: dict[str, Any] = {"observable": entry.id, "kind": kind}
    notes: list[str] = []
    if kind != "snapshot":
        for name, value in (("field", field), ("index", index), ("paired_rates", paired_rates)):
            _no_arg(name, value, kind)
        _no_arg("nan_policy", nan_policy, kind, "raise")
    if kind != "first_passage":
        _no_arg("ell", ell, kind)
    if kind != "peak":
        _no_arg("horizon_policy", horizon_policy, kind, "refuse")
        _no_arg("probe", probe, kind)
    if kind != "k_trace":
        _no_arg("normalized", normalized, kind, False)
    if depths is not None and kind != "snapshot":
        raise ValueError("where.t selects depths of snapshot observables only")
    if kind in ("first_passage", "k_trace") and params is not None:
        raise ValueError(f"{entry.id!r} has no parameters; leave params out")
    if kind == "snapshot":
        chosen_version, chosen = _select(s, entry.id, version, params)
        rows, empty = _snapshot_kind(s, entry.id, chosen_version, chosen, conds, depths, field,
                                     index, nan_policy)  # fmt: skip
        result.update(
            version=chosen_version, params=json.loads(chosen), _params_text=chosen, field=field,
            index=index, nan_policy=nan_policy, conditions=rows,
            estimand=(
                f"E[{entry.id}{'.' + field if field else ''}"
                f"{f'[{index}]' if index is not None else ''}] per depth"
                + (" (defined values only: an equal-graph mean of each graph's mean over them)"
                   if nan_policy == "omit" else "")
            ),
            estimator="nested equal-graph-weight mean (v1.4 §6.3) with graph-cluster standard "
            f"error (§6.4); {ESTIMATOR_VERSION}",
        )  # fmt: skip
        if paired_rates is not None:
            if not isinstance(paired_rates, (list, tuple)) or len(paired_rates) != 2:
                raise ValueError("paired_rates must be two rates [p1, p2]")
            pair = [Token.parse(v, what="paired_rates").token for v in paired_rates]
            if pair[0] == pair[1]:
                raise ValueError("paired_rates must be two different rates")
            result["paired"] = _paired(rows, pair)
            result["paired_estimand"] = (
                f"E[{entry.id}](p = {pair[1]}) - E[{entry.id}](p = {pair[0]}) on the same graphs: "
                "within-graph differences, then the graph-cluster error (v1.4 §6.4)"
            )
            if not result["paired"]:
                notes.append("no (protocol, n, beta) holds both paired rates in this selection")
    elif kind == "first_passage":
        ells = _ells(ell)
        rows, empty = _first_passage_kind(s, conds, ells)
        result.update(
            ells=ells, conditions=rows,
            estimand="tau_ell = min{t : k_t <= ell} (exact, every layer; v1.4 §5.2), right-"
            "censored at T_max",
            estimator="graph-equal-weight Kaplan-Meier (stats.nested_kaplan_meier): median, "
            "restricted mean E[min(tau, T_max)]; the mean only when nothing is censored",
        )  # fmt: skip
        notes.append(
            "a censored tau is not observed: mean is null when any trajectory is censored, and "
            "restricted_mean (a lower bound of E[tau]) is the mean of min(tau, T_max)"
        )
    elif kind == "peak":
        chosen_version, chosen = _select(s, entry.id, version, params)
        rows, empty, extra = _peak_kind(s, entry.id, chosen_version, chosen, conds,
                                        horizon_policy, probe)  # fmt: skip
        notes.extend(extra)
        result.update(
            version=chosen_version, params=json.loads(chosen), horizon_policy=horizon_policy,
            conditions=rows,
            estimand="T* = exp(mean of ln t over the maximizing set) per trajectory (v1.4 §5.8): "
            "E[T*] (arithmetic), exp(E[ln T*]) (geometric) and argmax_t E[d(t)] are different "
            "statistics",
            estimator="nested equal-graph means over trajectories with a peak (missing_fraction "
            "have none); horizon_policy decides about peaks at T_max (lower bounds)",
        )  # fmt: skip
    else:
        rows, empty, extra = _k_trace_kind(s, conds, normalized)
        notes.extend(extra)
        result.update(
            normalized=normalized, conditions=rows,
            estimand=("k_t / n" if normalized else "k_t") + " at every layer (v1.4 §5.2)",
            estimator="nested equal-graph-weight mean (v1.4 §6.3) with graph-cluster error",
        )  # fmt: skip
    if empty:
        notes.append(f"no stored values at {len(empty)} selected conditions (e.g. {empty[0]})")
    result["notes"] = notes
    return result


def aggregate_arrays(
    store: Any,
    observable: str,
    params: Any = None,
    where: Any = None,
    **options: Any,
) -> dict[str, Any]:
    """:func:`aggregate` uncapped: the same result with numpy arrays at full precision and
    the estimator objects (keys starting with ``_``: ``stats.NestedMean``,
    ``stats.SurvivalCurve``, ``stats.PeakTimeSummary`` and the per-trajectory arrays behind
    them), for scripts and :mod:`sparsegf2.study.plots`. ``store`` may be an open
    :class:`TableauStore`."""
    with _opened(store) as s:
        return _aggregate(s, observable, params, where, **options)


_SUBSAMPLED = ("depths", "mean", "se", "contributing_graphs", "contributing_trajectories")


def _render(result: Mapping[str, Any]) -> dict[str, Any]:
    """The JSON form of :func:`_aggregate`'s result: private keys dropped, at most
    :data:`MAX_SHOWN_CONDITIONS` conditions and ``sparsegf2.api.MAX_CHECKPOINTS_SHOWN`` depths
    per condition (evenly spaced, both ends kept)."""
    limit = _api.MAX_CHECKPOINTS_SHOWN
    notes = list(result.get("notes", []))

    def clean(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            out = {k: clean(v) for k, v in obj.items() if not str(k).startswith("_")}
            if "depths" in out and isinstance(obj.get("depths"), np.ndarray):
                count = len(obj["depths"])
                if count > limit:
                    keep = _api._subsample(count, limit)
                    for key in _SUBSAMPLED:
                        if key in obj and isinstance(obj[key], np.ndarray):
                            out[key] = clean(obj[key][keep])
                    out["depths_total"] = count
            return out
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        return obj

    rendered = clean({k: v for k, v in result.items() if k != "notes"})
    conds = rendered.get("conditions", [])
    if any("depths_total" in c for c in conds):
        notes.append(
            f"depths subsampled to {limit} per condition (evenly spaced, both ends kept); pass "
            "out to write every depth"
        )
    if len(conds) > MAX_SHOWN_CONDITIONS:
        rendered["conditions"] = conds[:MAX_SHOWN_CONDITIONS]
        rendered["conditions_total"] = len(conds)
        notes.append(
            f"{MAX_SHOWN_CONDITIONS} of {len(conds)} conditions shown: narrow where, or pass out"
        )
    rendered["notes"] = notes
    return rendered


def _aggregate_files(result: Mapping[str, Any], directory: Path) -> dict[str, str]:
    """``aggregate_<observable>.json`` (every number, exact) and ``.csv`` (one row per
    condition and depth, threshold, or condition)."""
    stem = f"aggregate_{result['observable']}"
    full = {k: v for k, v in result.items() if not str(k).startswith("_")}

    def strip(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return {k: strip(v) for k, v in obj.items() if not str(k).startswith("_")}
        if isinstance(obj, (list, tuple)):
            return [strip(v) for v in obj]
        return obj

    json_path = directory / f"{stem}.json"
    _write_json(json_path, strip(full))
    csv_path = directory / f"{stem}.csv"
    kind = result["kind"]
    rows: list[list[Any]] = []
    if kind in ("snapshot", "k_trace"):
        header = ["protocol", "n", "beta", "p", "t", "mean", "se"]
        for c in result["conditions"]:
            for i, t in enumerate(c["depths"].tolist()):
                rows.append([c["protocol"], c["n"], c["beta"], c["p"], t, c["mean"][i], c["se"][i]])
    elif kind == "first_passage":
        header = ["protocol", "n", "beta", "p", "ell", "trajectories", "censored",
                  "censored_fraction", "median", "restricted_mean", "restricted_mean_se",
                  "restricted_horizon", "mean", "mean_se"]  # fmt: skip
        for c in result["conditions"]:
            for e in c["ells"]:
                rows.append([c["protocol"], c["n"], c["beta"], c["p"], e["ell"],
                             e["trajectories"], e["censored"], e["censored_fraction"],
                             e["median"], e["restricted_mean"], e["restricted_mean_se"],
                             e["restricted_horizon"], e["mean"], e["mean_se"]])  # fmt: skip
    else:
        header = ["protocol", "n", "beta", "p", "trajectories", "missing_fraction",
                  "horizon_fraction", "arithmetic_mean", "arithmetic_se", "geometric_mean",
                  "geometric_se", "argmax"]  # fmt: skip
        for c in result["conditions"]:
            rows.append([c["protocol"], c["n"], c["beta"], c["p"], c["trajectories"],
                         c["missing_fraction"], c["horizon_fraction"], c["arithmetic_mean"],
                         c["arithmetic_se"], c["geometric_mean"], c["geometric_se"],
                         c["argmax"]])  # fmt: skip
    _write_csv(csv_path, header, rows)
    return {"json": str(json_path), "csv": str(csv_path)}


def aggregate(
    store: Any,
    observable: str,
    params: Any = None,
    where: Any = None,
    *,
    version: str | None = None,
    field: str | None = None,
    index: int | None = None,
    nan_policy: str = "raise",
    ell: Any = None,
    horizon_policy: str = "refuse",
    paired_rates: Any = None,
    normalized: bool = False,
    probe: str | None = None,
    record: bool = False,
    out: Any = None,
) -> dict[str, Any]:
    """Ensemble summaries of one observable per condition (v1.4 §6), never a plain mean of
    censored times.

    ``where`` selects conditions and depths (``{"protocol", "n", "beta", "p", "t"}``, each a
    value or a list; default every condition of the latest manifest). By the observable's kind:

    * **snapshot** observables (``k``, ``d_cyc``, ``D``, ``half_cut_entropy``, ...): at every
      stored depth, the nested equal-graph-weight mean (v1.4 §6.3) and its graph-cluster error
      (§6.4), with the graphs and trajectories that contributed. ``params`` picks a stored
      parameter set (an object whose entries must match, or left out when one is stored);
      ``field`` and ``index`` pick a data entry (``d_cyc`` ``field="zero_extended"``: the
      unconditional trace, 0 without a code; ``D_J`` ``field="robustness"``, ``index`` the
      coupling). A value that is not a finite number (``d_cyc`` without a code, an absent
      state) is refused unless ``nan_policy="omit"``, which changes the estimand and says so.
      ``paired_rates=[p1, p2]`` adds within-graph differences across two rates with their
      cluster error (v1.4 §6.4: never two independent errors).
    * **first_passage**: for each threshold ``ell`` (default 0 and 1), the graph-weighted
      Kaplan-Meier median, the restricted mean ``E[min(tau, T_max)]`` with its error, the
      censored fraction with its error, and the mean only when nothing is censored.
    * **peak_*** (``peak_d_cyc``, ``peak_D``, ``peak_D_J`` with ``params={"J": ...}``): the
      arithmetic ``E[T*]``, the geometric ``exp(E[ln T*])`` and ``argmax_t E[d(t)]``, the
      fractions of trajectories without a peak (``missing_fraction``) and with the peak at the
      horizon (``horizon_fraction``); ``horizon_policy`` (``refuse``: means withheld when a
      peak is at ``T_max``; ``exclude``; ``include``: lower bounds) is stated in the result.
    * **k_trace**: ``k_t`` (``normalized``: ``k_t / n``) at every layer, from the stored traces.

    ``record=True`` (writer) also stores each snapshot aggregate in the store's ``aggregates``
    table, keyed by its sample (design G-STO-4). ``out`` (a directory) writes every number
    (``aggregate_<id>.json``, exact, and ``.csv``).
    """
    record = _flag(record, "record")
    with _opened(store, write=record) as s, _lookups():
        result = _aggregate(s, observable, params, where, version=version, field=field,
                            index=index, nan_policy=nan_policy, ell=ell,
                            horizon_policy=horizon_policy, paired_rates=paired_rates,
                            normalized=normalized, probe=probe)  # fmt: skip
        if record:
            if result["kind"] != "snapshot":
                raise ValueError("record stores snapshot aggregates only")
            written = 0
            for cond in result["conditions"]:
                for t in cond["depths"].tolist():
                    s.aggregate(result["observable"], t=int(t), n=cond["n"], beta=cond["beta"],
                                p=cond["p"], protocol=cond["protocol"],
                                version=result["version"], params=result["_params_text"],
                                field=field, index=result["index"], nan_policy=nan_policy,
                                record=True)  # fmt: skip
                    written += 1
            result["recorded"] = written
    rendered = _render(result)
    if out is not None:
        rendered["files"] = _aggregate_files(result, _out_dir(out))
    return _finish(rendered)


# ---------------------------------------------------------------------------------------------
# finite-size scaling
# ---------------------------------------------------------------------------------------------


def _theta(observable: Any, ell: Any, ell_fraction: Any, coupling: Any) -> dict[str, Any]:
    """One scaling observable ``theta`` of v1.4 §5.9: ``tau_ell`` (``first_passage`` with
    ``ell``, or ``ell_fraction`` for ``ell = floor(fraction * n)``), or a peak time."""
    if isinstance(observable, Mapping):
        spec = dict(observable)
        _api._check_keys(spec, ("id", "ell", "ell_fraction", "J"), "observable")
        if "id" not in spec:
            raise ValueError('observable needs "id", e.g. {"id": "first_passage", "ell": 1}')
        if any(v is not None for v in (ell, ell_fraction, coupling)):
            raise ValueError("give ell, ell_fraction and J inside the observable object")
        observable = spec["id"]
        ell, ell_fraction, coupling = spec.get("ell"), spec.get("ell_fraction"), spec.get("J")
    oid = _text(observable, "observable")
    if oid not in _THETA:
        raise ValueError(
            f"fss fits first_passage (tau_ell) or a peak time ({sorted(k for k in _THETA if k != 'first_passage')}), "
            f"not {oid!r} (v1.4 §5.9)"
        )
    out: dict[str, Any] = {"id": oid}
    if oid == "first_passage":
        if (ell is None) == (ell_fraction is None):
            raise ValueError(
                "first_passage needs exactly one of ell (a threshold) and ell_fraction"
            )
        if ell is not None:
            out["ell"] = _int(ell, "ell", 0)
            out["theta"] = f"tau_{out['ell']}"
        else:
            frac = float(ell_fraction)
            if not 0 <= frac <= 1:
                raise ValueError(f"ell_fraction must be in [0, 1], got {ell_fraction!r}")
            out["ell_fraction"] = frac
            out["theta"] = f"tau_floor({frac:g}n)"
    else:
        if ell is not None or ell_fraction is not None:
            raise ValueError("ell and ell_fraction apply to first_passage only")
        if oid == "peak_D_J":
            if coupling is None:
                raise ValueError("peak_D_J needs J, the coupling of its D(J) trace")
            out["J"] = float(coupling)
            out["theta"] = f"T*_J={out['J']:g}"
        else:
            if coupling is not None:
                raise ValueError("J applies to peak_D_J only")
            out["theta"] = _THETA[oid]
    return out


def _bounds(bounds: Any, rates: Sequence[float]) -> dict[str, tuple[float, float]]:
    out = {"p_c": (min(rates), max(rates)), **_DEFAULT_BOUNDS}
    if bounds is None:
        return out
    if not isinstance(bounds, Mapping):
        raise ValueError('bounds must be an object such as {"p_c": [0.1, 0.3], "nu": [0.5, 3]}')
    _api._check_keys(dict(bounds), _fss.PARAMETERS, "bounds")
    for name, pair in bounds.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"bounds.{name} must be [low, high], got {pair!r}")
        lo, hi = (float(_api._require_number(v, f"bounds.{name}", -1e6, 1e6)) for v in pair)
        if lo > hi:
            raise ValueError(f"bounds.{name}: low {lo:g} above high {hi:g}")
        out[name] = (lo, hi)
    if out["nu"][0] <= 0:
        raise ValueError("bounds.nu must be positive (nu > 0)")
    return out


@dataclasses.dataclass
class _Point:
    n: int
    p: str
    p_float: float
    estimate: _stats.GraphMean | None
    fractions: dict[str, float]
    counts: dict[str, int]
    reason: str | None = None


def _fss_points(
    s: TableauStore,
    theta: Mapping[str, Any],
    beta: str,
    protocol: str,
    sizes: Sequence[int],
    p_range: tuple[float, float] | None,
    incomplete: Mapping[_Cond, int],
    settings: Mapping[str, Any],
) -> list[_Point]:
    manifest = s.manifest
    expected = manifest.trajectories_per_condition
    points: list[_Point] = []
    peak_choice: tuple[str, str] | None = None
    if theta["id"] != "first_passage":
        params = {"J": theta["J"]} if theta["id"] == "peak_D_J" else None
        peak_choice = _select(s, theta["id"], None, params)
    for n in sizes:
        for p in manifest.rates(n, beta):
            cond = _Cond(protocol, n, beta, p.token)
            x = p.float64
            if p_range is not None and not p_range[0] <= x <= p_range[1]:
                continue
            point = _Point(n, p.token, x, None, {}, {})
            points.append(point)
            if incomplete.get(cond, 0):
                point.reason = (
                    f"incomplete: {expected - incomplete[cond]} of {expected} trajectories "
                    "complete (run the plan first)"
                )
                continue
            if theta["id"] == "first_passage":
                ell = int(theta["ell"]) if "ell" in theta else math.floor(theta["ell_fraction"] * n)
                got = _first_passage_rows(s, cond)
                if got is None:
                    point.reason = "no stored first passages"
                    continue
                rows, tau, observed, depth = got
                curve, cens, times, seen = _survival(rows, tau, observed, depth, ell)
                frac = cens.censored_fraction
                point.fractions["censored_fraction"] = frac
                point.counts.update(trajectories=cens.trajectories, censored=cens.censored,
                                    graphs=cens.graphs, ell=ell)  # fmt: skip
                if cens.censored == 0:
                    point.estimate = _stats.nested_mean(rows["g"], rows["c"], rows["r"], times)
                elif (
                    settings["censoring"] == "restricted_mean"
                    and frac <= settings["max_censored_fraction"]
                ):
                    point.estimate = _stats.nested_mean(rows["g"], rows["c"], rows["r"], times)
                    point.counts["restricted"] = 1
                else:
                    point.reason = (
                        f"censored: {cens.censored} of {cens.trajectories} trajectories "
                        f"(graph-weighted fraction {frac:.3g}) did not reach k <= {ell} by "
                        f"T_max = {cens.horizon:g}"
                    )
            else:
                assert peak_choice is not None
                rows_, _, _ = _peak_kind(s, theta["id"], peak_choice[0], peak_choice[1], [cond],
                                         settings["horizon_policy"], None)  # fmt: skip
                if not rows_:
                    point.reason = "no stored peaks"
                    continue
                res = rows_[0]
                summary: _stats.PeakTimeSummary = res["_summary"]
                point.fractions.update(missing_fraction=summary.missing_fraction,
                                       horizon_fraction=summary.horizon_fraction)  # fmt: skip
                point.counts["trajectories"] = res["trajectories"]
                if summary.missing_fraction > settings["max_missing_fraction"]:
                    point.reason = (
                        f"missing peaks: fraction {summary.missing_fraction:.3g} above "
                        f"max_missing_fraction {settings['max_missing_fraction']:g}"
                    )
                elif summary.horizon_fraction > settings["max_horizon_fraction"]:
                    point.reason = (
                        f"peaks at the horizon: fraction {summary.horizon_fraction:.3g} above "
                        f"max_horizon_fraction {settings['max_horizon_fraction']:g} (T* there is "
                        "a lower bound)"
                    )
                elif summary.arithmetic is None:
                    point.reason = summary.note or "no peak time kept"
                else:
                    point.estimate = summary.arithmetic
            if point.estimate is not None and point.reason is None:
                est = point.estimate
                if not (np.isfinite(est.mean) and np.isfinite(est.se)):
                    point.reason = "fewer than two graphs with a value (no graph-cluster error)"
                    point.estimate = None
                elif not float(est.se) > 0:
                    point.reason = "zero standard error (every graph gave the same value)"
                    point.estimate = None
                else:
                    point.counts["graphs"] = int(est.contributing_graphs)
    return points


def _dataset(
    points: Sequence[_Point], theta: Mapping[str, Any], beta: str, label: str, estimator: str
) -> _fss.ScalingData | None:
    used = [pt for pt in points if pt.estimate is not None]
    if not used:
        return None
    return _fss.ScalingData(
        n=np.array([pt.n for pt in used], dtype=np.int64),
        p=np.array([pt.p_float for pt in used]),
        y=np.array([float(pt.estimate.mean) for pt in used]),  # type: ignore[union-attr]
        se=np.array([float(pt.estimate.se) for pt in used]),  # type: ignore[union-attr]
        observable=str(theta["theta"]),
        beta=beta,
        label=label,
        estimator=estimator,
        censored_fraction=np.array([pt.fractions.get("censored_fraction", 0.0) for pt in used]),
        horizon_fraction=np.array([pt.fractions.get("horizon_fraction", 0.0) for pt in used]),
        graphs=tuple(pt.estimate.graphs for pt in used),  # type: ignore[union-attr]
        graph_means=tuple(pt.estimate.graph_means for pt in used),  # type: ignore[union-attr]
    )


def _estimator_text(theta: Mapping[str, Any], settings: Mapping[str, Any]) -> str:
    if theta["id"] == "first_passage":
        if settings["censoring"] == "restricted_mean":
            return (
                "nested mean of min(tau, T_max): E[tau] where nothing is censored, the restricted "
                f"mean (a lower bound) where at most {settings['max_censored_fraction']:g} is "
                "censored; more censored points excluded"
            )
        return "nested mean of tau over points without censoring (censored points excluded)"
    return (
        f"nested mean of T* over trajectories with a peak (horizon_policy "
        f"{settings['horizon_policy']}); points with missing fraction above "
        f"{settings['max_missing_fraction']:g} or horizon fraction above "
        f"{settings['max_horizon_fraction']:g} excluded"
    )


def _fit_summary(fit: _fss.CollapseFit) -> dict[str, Any]:
    keys = ("p_c", "z", "nu", "cost", "points_used", "total_points", "status", "message",
            "at_bound", "sizes", "method", "degree", "window", "chi2", "dof", "window_range",
            "window_stable", "starts", "evaluations")  # fmt: skip
    out = {k: getattr(fit, k) for k in keys}
    if fit.coefficients is not None:
        out["coefficients"] = fit.coefficients
    if fit.selected is not None:
        out["selected"] = fit.selected
    return out


def _bootstrap_summary(boot: _fss.BootstrapFit) -> dict[str, Any]:
    return {
        "replicates": int(boot.replicates.shape[0]),
        "level": boot.level,
        "se": dict(zip(_fss.PARAMETERS, boot.se.tolist(), strict=True)),
        "interval": {
            name: [lo, hi]
            for name, lo, hi in zip(
                _fss.PARAMETERS, boot.lower.tolist(), boot.upper.tolist(), strict=True
            )  # fmt: skip
        },
        "failed": boot.failed,
        "at_bound": boot.at_bound,
        "seed": boot.seed,
        "reselect_window": boot.reselect_window,
    }


def _loso_summary(loso: _fss.LeaveOneSizeOut) -> dict[str, Any]:
    return {
        "sizes": list(loso.sizes),
        "params": loso.params,
        "statuses": [f.status for f in loso.fits],
        "held_out_cost": loso.held_out_cost,
        "held_out_points": loso.held_out_points,
        "spread": dict(zip(_fss.PARAMETERS, loso.spread.tolist(), strict=True)),
    }


def _diagnostics_summary(diag: _fss.CollapseDiagnostics) -> dict[str, Any]:
    return {
        "coverage": diag.coverage,
        "coverage_by_size": {str(k): v for k, v in diag.coverage_by_size.items()},
        "drift": None
        if diag.drift is None
        else dict(zip(_fss.PARAMETERS, diag.drift.tolist(), strict=True)),
        "drift_in_se": None
        if diag.drift_in_se is None
        else dict(zip(_fss.PARAMETERS, diag.drift_in_se.tolist(), strict=True)),
        "max_censored_fraction": diag.max_censored_fraction,
        "max_horizon_fraction": diag.max_horizon_fraction,
        "notes": list(diag.notes),
    }


def _schedules(s: TableauStore, beta: str, protocol: str, sizes: Sequence[int]) -> dict[str, Any]:
    """The depth schedule each size's trajectories hold (R322: the checkpoint resolution)."""
    manifest = s.manifest
    out: dict[str, Any] = {}
    for n in sizes:
        entry: dict[str, Any] = {
            "layers": manifest.layers(n, beta),
            "checkpoints": list(manifest.checkpoints(n, beta, protocol)),
        }
        rate = manifest.rates(n, beta)[0]
        try:
            key = s.key(n=n, beta=beta, p=rate, g=0, c=0, protocol=protocol)
            held = s.schedule(key)
            entry["checkpoints"] = list(held["checkpoints"])
            entry["peak_domain"] = list(held["domain"])
            entry["layers"] = held["layers"]
        except KeyError:
            pass
        out[str(n)] = entry
    return out


def _input_digest(data: _fss.ScalingData | None) -> str | None:
    if data is None:
        return None
    payload = {
        "n": data.n.tolist(),
        "p": [float(x).hex() for x in data.p],
        "y": [float(x).hex() for x in data.y],
        "se": [float(x).hex() for x in data.se],
        "graphs": [g.tolist() for g in (data.graphs or ())],
        "graph_means": [[float(x).hex() for x in m] for m in (data.graph_means or ())],
    }
    return sha256_hex(canonical_json(payload).encode("utf-8"))


def fss_report(
    store: Any,
    observable: Any,
    beta: Any,
    *,
    ell: Any = None,
    ell_fraction: Any = None,
    J: Any = None,
    protocol: str = "F",
    method: str = "houdayer_hartmann",
    degree: int = _fss.DEFAULT_DEGREE,
    window: Any = "overlap",
    p_range: Any = None,
    n_min: int | None = None,
    sizes: Any = None,
    bounds: Any = None,
    starts: int = 8,
    seed: int = 0,
    bootstrap: int = 200,
    level: float = 0.95,
    censoring: str = "mean",
    max_censored_fraction: float = 0.0,
    horizon_policy: str = "exclude",
    max_horizon_fraction: float = 0.0,
    max_missing_fraction: float = 0.0,
    leave_one_out: bool = True,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """The full finite-size-scaling report of :func:`fss` (exact floats, numpy arrays), for
    scripts and :func:`sparsegf2.study.plots.collapse`. The arguments are :func:`fss`'s."""
    beta_t = Token.parse(beta, what="beta").token
    protocol = check_protocol(_text(protocol, "protocol"))
    method_ = _METHODS.get(str(method))
    if method_ is None:
        raise ValueError(f"method must be one of {sorted(_METHODS)}, got {method!r}")
    if isinstance(observable, (list, tuple)):
        if any(v is not None for v in (ell, ell_fraction, J)):
            raise ValueError(
                "with a list of observables give ell, ell_fraction and J inside each object, "
                'e.g. [{"id": "first_passage", "ell": 1}, {"id": "peak_d_cyc"}]'
            )
        thetas = [_theta(o, None, None, None) for o in observable]
        if len(thetas) < 2:
            raise ValueError("a list of observables (a shared-exponent test) needs at least two")
    else:
        thetas = [_theta(observable, ell, ell_fraction, J)]
    settings: dict[str, Any] = {
        "method": method_,
        "degree": _int(degree, "degree", 1),
        "window": window
        if isinstance(window, str)
        else _api._require_number(window, "window", 1e-12, 1e12),  # fmt: skip
        "starts": _int(starts, "starts", 0),
        "seed": _int(seed, "seed", 0),
        "bootstrap": _int(bootstrap, "bootstrap", 0),
        "level": _api._require_number(level, "level", 0.5, 0.999),
        "censoring": _choice(censoring, "censoring", ("mean", "restricted_mean")),
        "max_censored_fraction": _api._require_number(
            max_censored_fraction, "max_censored_fraction", 0, 1
        ),  # fmt: skip
        "horizon_policy": _choice(horizon_policy, "horizon_policy", ("exclude", "include")),
        "max_horizon_fraction": _api._require_number(
            max_horizon_fraction, "max_horizon_fraction", 0, 1
        ),  # fmt: skip
        "max_missing_fraction": _api._require_number(
            max_missing_fraction, "max_missing_fraction", 0, 1
        ),  # fmt: skip
        "leave_one_out": _flag(leave_one_out, "leave_one_out"),
        "protocol": protocol,
    }
    if isinstance(window, str) and window != "overlap":
        raise ValueError(f"window must be 'overlap' or a positive number x_max, got {window!r}")
    if settings["censoring"] == "mean" and settings["max_censored_fraction"] != 0:
        raise ValueError(
            "max_censored_fraction applies with censoring='restricted_mean' only: the mean of tau "
            "is not identified where any trajectory is censored"
        )
    max_seconds = _positive_seconds(max_seconds, "max_seconds")
    p_window: tuple[float, float] | None = None
    if p_range is not None:
        if not isinstance(p_range, (list, tuple)) or len(p_range) != 2:
            raise ValueError("p_range must be [low, high]")
        p_window = (Token.parse(p_range[0], what="p_range").float64,
                    Token.parse(p_range[1], what="p_range").float64)  # fmt: skip
    start = time.perf_counter()
    with _opened(store) as s, _lookups():
        manifest = s.manifest
        all_sizes = sorted({c.n for c in manifest.conditions() if c.beta.token == beta_t})
        if not all_sizes:
            raise ValueError(
                f"beta = {beta_t} is not a condition of this study; it has beta "
                f"{sorted({c.beta.token for c in manifest.conditions()})}"
            )
        chosen = all_sizes
        if sizes is not None:
            wanted = {
                _int(v, "sizes", 1)
                for v in (sizes if isinstance(sizes, (list, tuple)) else [sizes])
            }
            missing = sorted(wanted - set(all_sizes))
            if missing:
                raise ValueError(
                    f"sizes {missing} are not run at beta = {beta_t}; it has {all_sizes}"
                )
            chosen = sorted(wanted)
        if n_min is not None:
            chosen = [n for n in chosen if n >= _int(n_min, "n_min", 1)]
        incomplete: Counter[_Cond] = Counter()
        for unit in _runner.plan(s):
            incomplete[_Cond(unit.protocol, unit.n, unit.beta.token, unit.p.token)] += (
                unit.trajectories
            )
        reports = []
        datasets = []
        for theta in thetas:
            points = _fss_points(s, theta, beta_t, protocol, chosen, p_window, incomplete, settings)
            data = _dataset(points, theta, beta_t, manifest.label, _estimator_text(theta, settings))
            reports.append((theta, points, data))
            datasets.append(data)
        schedules = _schedules(s, beta_t, protocol, chosen)
        stride = manifest.stride
        hierarchy = {
            "graphs": manifest.trials.graphs,
            "circuits_per_graph": manifest.trials.circuits_per_graph,
            "measurement_replicas": 1,
            "circuit_index_stride": stride,
        }
        context = {
            "experiment_id": manifest.experiment_id,
            "label": manifest.label,
            "store_revision": s.revision,
            "manifest_digest": manifest.digest(),
        }
    used_rates = [
        pt.p_float for _, points, _ in reports for pt in points if pt.estimate is not None
    ]
    box = _bounds(bounds, used_rates or [0.0, 1.0])
    settings["bounds"] = {k: list(v) for k, v in box.items()}
    fit_kwargs: dict[str, Any] = dict(bounds=box, method=method_, degree=settings["degree"],
                      window=settings["window"], starts=settings["starts"],
                      seed=settings["seed"])  # fmt: skip
    results = []
    for theta, points, data in reports:
        entry: dict[str, Any] = {
            "observable": theta,
            "estimand": f"E[{theta['theta']} | n, beta, p]",
            "estimator": None if data is None else data.estimator,
            "points": [
                {
                    "n": pt.n,
                    "p": pt.p,
                    "y": None if pt.estimate is None else float(pt.estimate.mean),
                    "se": None if pt.estimate is None else float(pt.estimate.se),
                    "used": pt.estimate is not None,
                    "reason": pt.reason,
                    **pt.fractions,
                    **pt.counts,
                }  # fmt: skip
                for pt in points
            ],
            "input_digest": _input_digest(data),
        }
        if data is None:
            entry.update(status="too_few_points", fit=None, bootstrap=None,
                         leave_one_size_out=None, diagnostics=None)  # fmt: skip
            results.append(entry)
            continue
        fit_start = time.perf_counter()
        fit = _fss.fit_collapse(data, **fit_kwargs)
        fit_seconds = time.perf_counter() - fit_start
        later = settings["bootstrap"] + (len(data.sizes) + 1) * settings["leave_one_out"] + 1
        if max_seconds is not None and fit_seconds * later > max_seconds:
            raise ValueError(
                f"one fit took {fit_seconds:.2f} s, so this report (about {later} fits: the "
                f"bootstrap, leave-one-size-out and drift refits) would take about "
                f"{fit_seconds * later:.0f} s, above max_seconds = {max_seconds:g}: lower bootstrap, "
                "or run `sparsegf2 study fss` in a shell"
            )
        boot = None
        if settings["bootstrap"] and fit.usable:
            boot = _fss.bootstrap_fit(data, replicates=settings["bootstrap"],
                                      level=settings["level"], **fit_kwargs)  # fmt: skip
        loso = None
        if settings["leave_one_out"] and fit.usable:
            loso = _fss.leave_one_size_out(data, **fit_kwargs)
        diag = _fss.collapse_diagnostics(data, fit, bootstrap=boot)
        status_ = fit.status
        notes = []
        if len(data.sizes) < 3:
            notes.append(
                f"only {len(data.sizes)} sizes: (p_c, z, nu) are not constrained (at least three "
                "are needed for leave-one-size-out checks)"
            )
        entry.update(
            status=status_,
            fit=_fit_summary(fit),
            bootstrap=None if boot is None else _bootstrap_summary(boot),
            leave_one_size_out=None if loso is None else _loso_summary(loso),
            diagnostics=_diagnostics_summary(diag),
            notes=notes,
            _data=data,
            _fit=fit,
        )
        results.append(entry)
    shared = None
    if len(thetas) > 1:
        usable = [d for d in datasets if d is not None]
        if len(usable) == len(datasets):
            comparison = _fss.compare_shared_independent(
                usable, replicates=max(settings["bootstrap"], 19), alpha=0.05,
                **fit_kwargs,
            )  # fmt: skip
            sh = comparison.shared
            verdict = (
                "undetermined: " + comparison.message
                if comparison.prefers == "undetermined"
                else (
                    f"shared (p_c, z, nu) rejected at alpha = {comparison.alpha:g} "
                    f"(p = {comparison.p_value:.3g})"
                    if comparison.prefers == "independent"
                    else f"consistent with shared (p_c, z, nu) at alpha = {comparison.alpha:g} "
                    f"(p = {comparison.p_value:.3g}); not evidence that they are equal"
                )
            )
            shared = {
                "fit": {
                    "p_c": sh.p_c,
                    "z": sh.z,
                    "nu": sh.nu,
                    "cost": sh.cost,
                    "status": sh.status,
                    "observables": list(sh.observables),
                },  # fmt: skip
                "independent_cost": comparison.independent_cost,
                "delta": comparison.delta,
                "p_value": comparison.p_value,
                "alpha": comparison.alpha,
                "prefers": comparison.prefers,
                "failed": comparison.failed,
                "verdict": verdict,
            }
        else:
            shared = {"verdict": "undetermined: an observable has no usable points"}
    statuses = [r["status"] for r in results]
    report = {
        "schema": FSS_REPORT_SCHEMA,
        "hypothesis": "unverified ansatz (v1.4 §5.9): E[theta | n, beta, p] ~ n^z F((p - p_c) "
        "n^(1/nu)), fitted at fixed beta with one scaling function F per observable",
        "beta": beta_t,
        **context,
        "settings": settings,
        "sample_hierarchy": hierarchy,
        "resampling": "graph-cluster bootstrap: graphs resampled once per (n, beta) with seed "
        "[seed, n] and applied to every rate (paired across p)",
        "schedules": schedules,
        "sizes": chosen,
        "results": results,
        "shared": shared,
        "status": statuses[0] if len(statuses) == 1 else statuses,
        "elapsed_seconds": time.perf_counter() - start,
    }
    return report


def _report_file(report: Mapping[str, Any]) -> dict[str, Any]:
    """The report without timing and private objects: rerunning on the same store gives the
    same file, byte for byte."""

    def strip(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return {k: strip(v) for k, v in obj.items()
                    if not str(k).startswith("_") and k != "elapsed_seconds"}  # fmt: skip
        if isinstance(obj, (list, tuple)):
            return [strip(v) for v in obj]
        return obj

    return dict(strip(report))


def fss(
    store: Any,
    observable: Any,
    beta: Any,
    *,
    out: Any = None,
    **options: Any,
) -> dict[str, Any]:
    """Fit the dynamic scaling ansatz at one ``beta`` (v1.4 §5.9; ``study/fss.py``) and report
    the fit, its uncertainty and its diagnostics.

    ``observable``: ``"first_passage"`` with ``ell`` (or ``ell_fraction``: ``ell = floor(f n)``)
    for ``tau_ell``, ``"peak_d_cyc"``, ``"peak_D"`` or ``"peak_D_J"`` with ``J`` for a peak time
    ``T*``; or a list of such objects (``{"id": "first_passage", "ell": 1}``) to test shared
    exponents against independent ones. Each point ``(n, p)`` is a nested equal-graph mean with
    its graph-cluster error, from complete conditions only. **Censoring**: by default a point
    with any censored trajectory is excluded and listed (the mean of ``tau`` is not identified
    there); ``censoring="restricted_mean"`` with ``max_censored_fraction`` admits points whose
    graph-weighted censored fraction is at most that, as ``E[min(tau, T_max)]``, a lower bound.
    **Peaks**: points whose fraction of missing peaks or of peaks at the horizon exceeds
    ``max_missing_fraction`` / ``max_horizon_fraction`` (default 0) are excluded; kept peaks at
    the horizon are excluded or included per ``horizon_policy``.

    The fit (``method`` ``houdayer_hartmann`` (``hh``) or ``polynomial`` with ``degree`` and
    ``window``; ``bounds`` default ``p_c`` within the rates used, ``z`` in [0, 3], ``nu`` in
    [0.5, 5]; ``starts`` and ``seed`` fix the search) is followed by a graph-cluster bootstrap
    (``bootstrap`` replicates, paired across ``p``), leave-one-size-out refits and the
    diagnostics of ``study/fss.py``. ``out`` (a directory) writes the full report as
    ``fss_<theta>_beta<b>.json`` (identical bytes when rerun on the same data) and the points
    as ``.csv``. ``max_seconds`` refuses a report whose refits would take longer, from the
    first fit's time. The plateau width of the peak rule is ``plateau_z``; ``z`` here is the
    dynamical exponent.
    """
    report = fss_report(store, observable, beta, **options)
    summary: dict[str, Any] = {
        "schema": report["schema"],
        "hypothesis": report["hypothesis"],
        "beta": report["beta"],
        "label": report["label"],
        "status": report["status"],
        "settings": report["settings"],
        "sizes": report["sizes"],
        "elapsed_seconds": report["elapsed_seconds"],
        "results": [],
    }
    for res in report["results"]:
        used = [pt for pt in res["points"] if pt["used"]]
        excluded = [pt for pt in res["points"] if not pt["used"]]
        item: dict[str, Any] = {
            "observable": res["observable"]["theta"],
            "status": res["status"],
            "points_used": len(used),
            "points_excluded": len(excluded),
            "excluded": [
                {"n": pt["n"], "p": pt["p"], "reason": pt["reason"]} for pt in excluded[:8]
            ],  # fmt: skip
            "input_digest": res["input_digest"],
        }
        if res["fit"] is not None:
            fit = res["fit"]
            item["fit"] = {k: fit[k] for k in ("p_c", "z", "nu", "cost", "points_used",
                                               "total_points", "status", "message", "at_bound")}  # fmt: skip
            if res["bootstrap"] is not None:
                item["bootstrap"] = {k: res["bootstrap"][k]
                                     for k in ("replicates", "level", "se", "interval", "failed")}  # fmt: skip
            if res["leave_one_size_out"] is not None:
                item["leave_one_size_out"] = {k: res["leave_one_size_out"][k]
                                              for k in ("sizes", "spread", "held_out_cost")}  # fmt: skip
            item["diagnostics"] = res["diagnostics"]
            if res.get("notes"):
                item["notes"] = res["notes"]
        summary["results"].append(item)
    if report["shared"] is not None:
        summary["shared"] = report["shared"]
    if out is not None:
        directory = _out_dir(out)
        names = "+".join(r["observable"]["theta"] for r in report["results"])
        stem = f"fss_{safe_name(names)}_beta{safe_name(report['beta'])}"
        json_path = directory / f"{stem}.json"
        _write_json(json_path, _report_file(report))
        csv_path = directory / f"{stem}.csv"
        rows = [
            [
                r["observable"]["theta"],
                pt["n"],
                pt["p"],
                pt["y"],
                pt["se"],
                pt["used"],
                pt.get("censored_fraction"),
                pt.get("missing_fraction"),
                pt.get("horizon_fraction"),
                pt["reason"],
            ]  # fmt: skip
            for r in report["results"]
            for pt in r["points"]
        ]
        _write_csv(csv_path, ["observable", "n", "p", "y", "se", "used", "censored_fraction",
                              "missing_fraction", "horizon_fraction", "reason"], rows)  # fmt: skip
        summary["files"] = {"report": str(json_path), "points": str(csv_path)}
    return _finish(summary)


# ---------------------------------------------------------------------------------------------
# plots (matplotlib is imported by sparsegf2.study.plots only when a figure is drawn)
# ---------------------------------------------------------------------------------------------


def plot(
    store: Any, kind: str, out: Any, options: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Draw one diagnostic figure into the directory ``out`` and write the numbers it shows.

    ``kind``: ``k`` (mean ``k_t`` with bands), ``survival`` (Kaplan-Meier of ``tau_ell``,
    censoring marked), ``d_cyc`` (zero-extended mean ``d_cyc`` versus depth), ``peak_distribution``
    (the distribution of ``T*`` with missing and horizon fractions), ``peak_vs_tau1`` (``T*``
    against ``tau_1`` per trajectory), ``vs_p`` (an observable versus ``p`` for each ``n``),
    ``graph_measures`` (graph measurements versus ``beta``) or ``collapse`` (a scaling collapse
    from an :func:`fss` report file: ``options={"report": path}``; ``store`` may then be
    null). ``options`` are the plot function's keywords (``sparsegf2.study.plots``). Each figure
    is written as ``.svg`` and ``.png`` with a ``.csv`` and ``.json`` of its numbers; the result
    has the paths and the caption.
    """
    from sparsegf2.study import plots as _plots

    kind = _text(kind, "kind")
    functions = _plots.PLOTS
    if kind not in functions:
        raise ValueError(f"kind must be one of {sorted(functions)}, got {kind!r}")
    options = dict(options or {})
    directory = _out_dir(out)
    if kind == "collapse":
        result = functions[kind](options.pop("report", None), directory, **options)
    else:
        with _opened(store) as s, _lookups():
            result = functions[kind](s, directory, **options)
    return _finish(result)
