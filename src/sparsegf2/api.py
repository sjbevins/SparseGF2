"""A JSON-in, JSON-out service layer: the one interface the CLI and the MCP server share.

Every public function here takes plain JSON values (str, int, float, bool,
None, lists and dicts) and returns a JSON-serializable dict that is **small**:
by default under :data:`MAX_OUTPUT_CHARS` characters when encoded, so an LLM
agent can read it whole. Large arrays are summarized (means, standard errors,
subsampled checkpoints) and, when an ``output_dir`` is given, written to files
whose absolute paths are returned. Invalid input raises :class:`ValueError`
with a message that says what to change.

Functions
---------
:func:`capabilities`
    What the simulator supports: instructions, observables, probes, graph
    families, limits and worked examples.
:func:`inspect_circuit`
    Parse a Stim-language circuit; counts, an instruction histogram, and
    validation errors with line numbers.
:func:`run_circuit`
    Run a circuit for one or many shots; measurement statistics, and
    entanglement / code / erasure observables of the final state (optionally
    after every ``TICK`` too).
:func:`run_ensemble`
    Run many trajectories of a random monitored circuit
    (:class:`sparsegf2.mipt.MonitoredCircuit`) and summarize the probes and
    the purification statistics.
:func:`estimate_ensemble`
    A rough runtime estimate for :func:`run_ensemble`, with the layout of the
    saved result.
:func:`benchmark_ensemble`
    Measured speed and storage of a run on this machine, projected to other
    thread counts; writes nothing.
:func:`start_ensemble_job`, :func:`job_status`, :func:`cancel_job`, :func:`list_jobs`
    The same ensembles as resumable background jobs, saved chunk by chunk.
:func:`load_results`
    Summarize a saved ensemble (a ``.npz`` file or an output directory).
:func:`sample_erasure_regions`
    Geometry-correlated erasure regions on a graph (no circuit): the boundary's mean
    against its exact value, chain diagnostics and notes on mixing; read-only.
:func:`erasure_curves`
    The failure probability and tolerance of the independent and correlated erasure
    channels of a saved run, per coupling; read-only.

Conventions shared by every function: seeds are non-negative ints (``None``
draws a fresh one, which is reported back so the run can be repeated); qubit
lists are JSON lists of ints or a range string ``"start:stop[:step]"`` with
Python slice semantics (``"0:8"`` is qubits 0..7).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import inspect
import json
import math
import os
import platform
import re
import secrets
import socket
import statistics
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import uuid
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from sparsegf2 import __version__, analysis, erasures, graphs, mipt, observables
from sparsegf2._compile import DEFAULT_MAX_OPS
from sparsegf2._kernel.entropy import capped_by_runs
from sparsegf2.circuit import Circuit, Instruction, RepeatBlock, _strip_comment
from sparsegf2.errors import PhaseFreeError
from sparsegf2.gates import KINDS, all_gates, gate
from sparsegf2.sampling import sample
from sparsegf2.simulator import Simulator

__all__ = [
    "MAX_BENCHMARK_SECONDS",
    "MAX_OUTPUT_CHARS",
    "SYNC_ENSEMBLE_MAX_SECONDS",
    "benchmark_ensemble",
    "cancel_job",
    "capabilities",
    "erasure_curves",
    "estimate_ensemble",
    "inspect_circuit",
    "job_status",
    "list_jobs",
    "load_results",
    "quickstart",
    "run_circuit",
    "run_ensemble",
    "sample_erasure_regions",
    "start_ensemble_job",
]

# ---------------------------------------------------------------------------
# limits
# ---------------------------------------------------------------------------

MAX_OUTPUT_CHARS = 8000
"""Target upper bound on the JSON size of every result."""

SYNC_ENSEMBLE_MAX_SECONDS = 60.0
"""Estimated runtime above which interactive callers should use a background job."""

MAX_GRAPH_REALIZATIONS = 100_000
MAX_RESULT_GB = 8.0
"""Largest ensemble result (probe values and code snapshots) one run or job may hold in memory."""

MAX_GRAPH_GB = 2.0
"""Largest memory a spec's graphs may take, checked from the family's parameters before any
graph is generated: the edges a collection keeps (16 bytes per edge), and the peak while one
graph is generated (48 bytes per pair for ``complete`` and ``erdos_renyi``, up to 512 per
edge for the ring families). The engine holds the edges about three times over (the graphs,
their concatenation and an index per edge), so a collection at this limit needs about as
much memory as the largest result (:data:`MAX_RESULT_GB`)."""

MAX_CHECKPOINTS_SHOWN = 50
MAX_BITS_SHOWN = 256
MAX_LIST_SHOWN = 64
MAX_PROFILE_SHOWN = 65
MAX_TEXT_SHOWN = 3000
MAX_ERRORS_SHOWN = 20
MAX_STABILIZER_QUBITS = 32
MAX_NAME_CHARS = 64
MAX_VECTOR_DEPTHS = 8
MAX_GRID_SIZES = 17
MAX_OBSERVED_TICKS = 1_000_000
MAX_TICK_ROWS = 10_000
MAX_QUBITS = 65_536
MAX_SHOT_RESULTS = 200_000_000
MAX_TRAJECTORIES = 100_000_000
MAX_INT = 2**62
"""Largest integer accepted for counts, sizes and indices (seeds excepted): nothing larger fits."""

SCHEDULES = ("matching", "random_edges", "all_edges")
REFERENCES = ("none", "purification", "single")

# Compiled kernels run in parallel over all cores; serialize them so that a
# background job and an interactive call never run two parallel regions at
# once (Numba's threading layers are not all re-entrant).
_COMPUTE_LOCK = threading.Lock()

_RESULT_FILE = "result.npz"
_CHUNK_DIR = "chunks"
_SPEC_FILE = "spec.json"
_PROVENANCE_FILE = "provenance.json"
_LOCK_FILE = ".lock"  # held by the one process writing to an output directory
_TAKEOVER_FILE = ".lock.takeover"  # the turn of a process removing a stale lock

# Windows refuses to replace or delete a file that another process holds open (a
# notebook reading result.npz, say), and to open one that is being replaced; that
# passes within moments, so these operations are retried with this backoff (seconds,
# about 5 s in total) before they fail.
_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 1.6)

_MERGE_PENDING = (
    "merge pending: result.npz is in use; the chunk files hold the data, and the next "
    "load_results or start_ensemble_job merges them"
)


# ---------------------------------------------------------------------------
# small helpers: JSON cleaning, bounding, validation
# ---------------------------------------------------------------------------


def _num(x: float | np.floating[Any]) -> float | None:
    """A float rounded to 6 significant digits; NaN and infinities become None."""
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f"{x:.6g}")


def _jsonable(obj: Any, *, exact: bool = False) -> Any:
    """Recursively convert numpy values, tuples and paths to plain JSON values.

    Floats are rounded to 6 significant digits for display; ``exact=True``
    keeps them as they are (for files and stored specs; non-finite values
    still become None).
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, exact=exact) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, exact=exact) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist(), exact=exact)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if exact:
            return float(obj) if math.isfinite(obj) else None
        return _num(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _short_key(key: str, limit: int = MAX_NAME_CHARS) -> str:
    """``key`` itself if short, else a prefix plus a hash that keeps keys distinct."""
    if len(key) <= limit:
        return key
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    return f"{key[: limit - 12]}...#{digest}"


def _shrink(obj: Any, max_items: int, max_text: int, depth: int = 0) -> Any:
    """Cut lists (and nested dicts) to ``max_items`` entries and strings and keys to ``max_text``."""
    if isinstance(obj, dict):
        items = list(obj.items())
        if depth > 0:
            items = items[: max(max_items, 1)]
        return {
            _short_key(k, max(max_text, 32)): _shrink(v, max_items, max_text, depth + 1)
            for k, v in items
        }
    if isinstance(obj, list):
        return [_shrink(v, max_items, max_text, depth + 1) for v in obj[:max_items]]
    if isinstance(obj, str) and len(obj) > max_text:
        return obj[:max_text] + "...[truncated]"
    return obj


def _mark_truncated(out: dict[str, Any], note: str) -> dict[str, Any]:
    """Set ``truncated`` and append ``note`` (once) after every note ``out`` already has."""
    notes = list(out["notes"]) if isinstance(out.get("notes"), list) else []
    if note not in notes:
        notes.append(note)
    out["notes"] = notes
    out["truncated"] = True
    return out


def _clean(out: dict[str, Any], exact: Sequence[str] = ()) -> dict[str, Any]:
    """``_jsonable(out)``, except that the top-level fields named in ``exact`` keep their floats."""
    return {**_jsonable(out), **{k: _jsonable(out[k], exact=True) for k in exact if k in out}}


def _bounded(
    out: dict[str, Any], limit: int = MAX_OUTPUT_CHARS, *, exact: Sequence[str] = ()
) -> dict[str, Any]:
    """Make ``out`` JSON-clean and, if still too long, cut long lists, strings and keys.

    The summaries below are designed to fit already; this is the safety net.
    When it acts, ``truncated`` is set and a note says how to get everything.
    The notes say how to read the result, so they are kept whole (unless
    they alone exceed the limit). As a last resort only the top-level scalar
    fields (seed, counts, ...), ``files`` and the notes are kept, so a caller
    never loses the run's identity or paths. The top-level fields named in
    ``exact`` (a stored spec, which a caller passes back) are neither rounded
    nor cut: they are shown as they are, or (in the last resort) left out.
    """
    out = _clean(out, exact)
    if len(json.dumps(out)) <= limit:
        return out
    note = (
        "output cut to fit the size limit; pass output_dir (or read the saved files) for all data"
    )
    notes = out.get("notes")
    kept = {k: out[k] for k in exact if k in out}
    for max_items in (32, 16, 8, 4, 2, 1, 0):
        shrunk = {**_shrink(out, max_items, max(200, 80 * max_items)), **kept}
        if isinstance(notes, list):
            shrunk["notes"] = notes
        shrunk = _mark_truncated(shrunk, note)
        if len(json.dumps(shrunk)) <= limit:
            return shrunk
    essentials = {
        k: v
        for k, v in out.items()
        if not isinstance(v, (dict, list)) and not (isinstance(v, str) and len(v) > 200)
    }
    if isinstance(out.get("files"), dict):
        essentials["files"] = out["files"]
    if isinstance(notes, list):
        essentials["notes"] = notes
    essentials = _mark_truncated(
        essentials, "result too large to show; only scalar fields, file paths and notes are kept"
    )
    if len(json.dumps(essentials)) <= limit:
        return essentials
    return {"truncated": True, "notes": ["result too large to show; pass output_dir to save it"]}


def _require_int(
    value: Any, what: str, *, minimum: int | None = None, maximum: int | None = MAX_INT
) -> int:
    """``value`` as an int; ``maximum=None`` lifts the size bound (for seeds)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        else:
            raise ValueError(f"{what} must be an integer, got {value!r}")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{what} must be at least {minimum}, got {value}")
    if maximum is not None and abs(value) > maximum:
        raise ValueError(f"{what} is too large (at most {maximum:,}), got {value}")
    return value


def _require_bool(value: Any, what: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{what} must be true or false, got {value!r}")
    return bool(value)


def _require_number(value: Any, what: str, low: float, high: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{what} must be a number in [{low:g}, {high:g}], got {value!r}")
    value = float(value)
    if not low <= value <= high:
        raise ValueError(f"{what} must be in [{low:g}, {high:g}], got {value:g}")
    return value


_SIMPLE_TYPES = {
    "int": "an integer",
    "float": "a number",
    "bool": "true or false",
    "str": "a string",
    "None": "null",
}


def _check_type(value: Any, annotation: Any, what: str) -> Any:
    """Check ``value`` against a simple type annotation (``int``, ``float | None``, ...).

    Annotations may be strings (``from __future__ import annotations``). Values
    are converted only where JSON is ambiguous: a whole-number float is
    accepted for ``int``, an int for ``float``. Annotations involving other
    types (arrays, sequences, probes) are left to the library to validate.
    """
    if annotation is inspect.Parameter.empty:
        return value
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    parts = [p.strip() for p in str(text).split("|")]
    if not parts or any(p not in _SIMPLE_TYPES for p in parts):
        return value
    if value is None:
        if "None" in parts:
            return None
    elif isinstance(value, (bool, np.bool_)):
        if "bool" in parts:
            return bool(value)
    elif isinstance(value, (int, np.integer)):
        if "int" in parts:
            return _require_int(value, what)
        if "float" in parts:
            return float(value)
    elif isinstance(value, (float, np.floating)):
        if "float" in parts:
            return float(value)
        if "int" in parts and float(value).is_integer():
            return _require_int(value, what)
    elif isinstance(value, str) and "str" in parts:
        return value
    expected = " or ".join(_SIMPLE_TYPES[p] for p in parts)
    raise ValueError(f"{what} must be {expected}, got {value!r}")


def _resolve_seed(seed: Any) -> int:
    """A concrete non-negative int seed; ``None`` draws a fresh one below 2**53.

    53 bits, so that the reported seed survives a client that reads JSON numbers as
    doubles (JavaScript, Claude Code among them): a larger one would come back rounded
    and repeat another run. An explicit seed of any size is used, and reported, exactly.
    """
    if seed is None:
        return secrets.randbits(53)
    return _require_int(seed, "seed", minimum=0, maximum=None)


_BIG_SEED_NOTE = (
    "seed >= 2**53: a client that reads JSON numbers as doubles (JavaScript, Claude Code) "
    "may round it; to repeat runs from such a client, use seeds below 2**53"
)


def _seed_notes(seed: Any) -> list[str]:
    """``[_BIG_SEED_NOTE]`` for a seed that such a client cannot hold exactly, else ``[]``.

    The seed itself is used and reported exactly; the client may already have rounded
    the one it sent, and cannot read this one back unrounded.
    """
    return [_BIG_SEED_NOTE] if isinstance(seed, int) and seed >= 2**53 else []


_RANGE_RE = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*(?::\s*(\d+)\s*)?$")


def _qubit_list(value: Any, what: str, limit: int | None = None) -> list[int]:
    """Parse a qubit list: a JSON list of ints or a range string ``"a:b[:step]"``."""
    if isinstance(value, str):
        m = _RANGE_RE.match(value)
        if not m:
            raise ValueError(
                f"{what}: expected a list of qubit indices or a range like '0:8', got {value!r}"
            )
        start, stop, step = int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)
        if step < 1:
            raise ValueError(f"{what}: range step must be positive")
        qubits = list(range(start, stop, step))
    elif isinstance(value, (list, tuple)):
        qubits = [_require_int(q, f"{what} entries", minimum=0) for q in value]
    else:
        raise ValueError(
            f"{what}: expected a list of qubit indices or a range like '0:8', got {value!r}"
        )
    if limit is not None and any(q >= limit for q in qubits):
        raise ValueError(f"{what}: qubit {max(qubits)} out of range (valid: 0..{limit - 1})")
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"{what}: contains duplicate qubits")
    return qubits


def _check_keys(given: dict, allowed: Sequence[str], what: str) -> None:
    unknown = sorted(set(given) - set(allowed))
    if unknown:
        raise ValueError(f"{what}: unknown key(s) {unknown}; allowed keys are {sorted(allowed)}")


def _require_name(value: Any, what: str, choices: Iterable[str]) -> str:
    """``value``, a user-given key into a table (a family, a kind, a section), if it is a
    string: checked before the lookup, which a list or a dict would fail with a TypeError."""
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a string (one of {sorted(choices)}), got {value!r}")
    return value


def _subsample(count: int, limit: int) -> np.ndarray:
    """At most ``limit`` evenly spaced indices into ``range(count)``, always keeping both ends."""
    if count <= limit:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, limit).round().astype(np.int64))


def _spread_by_value(values: np.ndarray, limit: int) -> np.ndarray:
    """Indices of at most ``limit`` of ``values`` for display: the given order if they all fit,
    else evenly spaced *in increasing order of the value*, always including the smallest and
    the largest (couplings come in any order, so positions alone could drop an extreme)."""
    if values.size <= limit:
        return np.arange(values.size)
    order = np.argsort(values, kind="stable")
    return order[_subsample(values.size, limit)]


def _bits(arr: np.ndarray) -> str:
    return "".join("1" if b else "0" for b in np.asarray(arr).ravel())


def _parse_circuit(text: Any) -> Circuit:
    """Parse and fully validate circuit text; errors name the offending line."""
    if not isinstance(text, str):
        raise ValueError(f"circuit must be Stim circuit text (a string), got {type(text).__name__}")
    try:
        parsed = Circuit(text)
    except (ValueError, KeyError) as exc:
        msg = exc.args[0] if exc.args else str(exc)
        raise ValueError(f"invalid circuit: {msg}") from None
    errors = _semantic_errors(parsed, text)
    if errors:
        e = errors[0]
        where = f"line {e['line']}: " if e["line"] is not None else ""
        raise ValueError(f"invalid circuit: {where}{e['message']}")
    return parsed


def _first_sentence(fn: Callable) -> str:
    """The first sentence of a docstring, on one line."""
    para = (inspect.getdoc(fn) or "").split("\n\n")[0].replace("\n", " ").strip()
    end = para.find(". ")
    para = para if end < 0 else para[: end + 1]
    return re.sub(r":\w+:`([^`]*)`", r"\1", para).replace("``", "")


# ---------------------------------------------------------------------------
# observables of a single state (run_circuit)
# ---------------------------------------------------------------------------

_QUBIT_ARGS = {"region", "a", "b", "c", "reference", "erased", "system", "order"}


@dataclasses.dataclass(frozen=True)
class _Observable:
    compute: Callable[[Simulator, dict], Any]  # the full value (saved with output_dir)
    required: tuple[str, ...]
    optional: tuple[str, ...]
    description: str
    needs_phases: bool = False
    show: Callable[[Any], Any] | None = None  # the full value -> its summary in results


def _profile_summary(values: np.ndarray) -> dict[str, Any]:
    idx = _subsample(len(values), MAX_PROFILE_SHOWN)
    out: dict[str, Any] = {"values": values[idx]}
    if len(idx) < len(values):
        out["index"] = idx
        out["subsampled"] = True
    return out


def _weights_summary(w: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(w.mean()) if w.size else 0.0,
        "min": int(w.min()) if w.size else 0,
        "max": int(w.max()) if w.size else 0,
    }


def _stabilizer_strings(sim: Simulator, a: dict) -> list[str]:
    if sim.num_qubits > MAX_STABILIZER_QUBITS:
        raise ValueError(
            f"'stabilizers' lists generators only for up to {MAX_STABILIZER_QUBITS} qubits "
            f"(this state has {sim.num_qubits}); use 'stabilizer_weights' or "
            "'stabilizer_length_distribution' for large states"
        )
    strings = [str(p) for p in sim.stabilizers()]
    # Without sign tracking every sign would read "+": report the Paulis unsigned.
    return strings if sim.phases else [text[1:] for text in strings]


def _erasure_stats(sim: Simulator, a: dict) -> dict[str, Any]:
    s = observables.random_erasure_statistics(
        sim, a["system"], a["reference"], a["sizes"], a["samples"], seed=a["seed"]
    )
    return {
        "sizes": s["sizes"],
        "correctable_fraction": s["correctable_fraction"],
        "mean_lost": s["lost"].mean(axis=1),
        "mean_recovery_probability": s["mean_recovery_probability"],
    }


def _erasure_profile(sim: Simulator, a: dict) -> dict[str, Any]:
    s = observables.erasure_profile(sim, a["system"], a["reference"], a["samples"], seed=a["seed"])
    return {"correctable_fraction": s["correctable_fraction"], "mean_lost": s["mean_lost"]}


def _profiles_summary(value: dict[str, np.ndarray]) -> dict[str, Any]:
    return {key: _profile_summary(v) for key, v in value.items()}


def _subsampled(value: Any) -> bool:
    """Whether a shown observable value had a vector subsampled (see :func:`_profile_summary`)."""
    if isinstance(value, dict):
        return bool(value.get("subsampled")) or any(_subsampled(v) for v in value.values())
    return False


_OBSERVABLES: dict[str, _Observable] = {
    "entropy": _Observable(
        lambda sim, a: observables.entropy(sim, a["region"]),
        ("region",),
        (),
        "Entanglement entropy S(A) of a region, in bits.",
    ),
    "mutual_information": _Observable(
        lambda sim, a: observables.mutual_information(sim, a["a"], a["b"]),
        ("a", "b"),
        (),
        "I(A:B) = S(A) + S(B) - S(AB) for disjoint regions.",
    ),
    "tripartite_information": _Observable(
        lambda sim, a: observables.tripartite_information(sim, a["a"], a["b"], a["c"]),
        ("a", "b", "c"),
        (),
        "I3(A:B:C) for disjoint regions (negative at an MIPT).",
    ),
    "entropy_profile": _Observable(
        lambda sim, a: observables.entropy_profile(sim, a.get("order")),
        (),
        ("order",),
        "S(order[:k]) for k = 0..len(order) (default order 0..n-1).",
        show=_profile_summary,
    ),
    "code_dimension": _Observable(
        lambda sim, a: observables.code_dimension(sim, a["reference"]),
        ("reference",),
        (),
        "Logical qubits held by the system: k = S(reference).",
    ),
    "lost_logical_information": _Observable(
        lambda sim, a: observables.lost_logical_information(sim, a["erased"], a["reference"]),
        ("erased", "reference"),
        (),
        "I(R:E): logical bits destroyed by erasing E (0 = correctable).",
    ),
    "erasure_is_correctable": _Observable(
        lambda sim, a: observables.erasure_is_correctable(sim, a["erased"], a["reference"]),
        ("erased", "reference"),
        (),
        "True iff erasing E loses no logical information.",
    ),
    "recovery_probability": _Observable(
        lambda sim, a: observables.recovery_probability(sim, a["erased"], a["reference"]),
        ("erased", "reference"),
        (),
        "2**-I(R:E): optimal recovery probability after erasing E.",
    ),
    "random_erasure_statistics": _Observable(
        _erasure_stats,
        ("system", "reference", "sizes"),
        ("samples", "seed"),
        "Random erasures of each size: correctable fraction, mean lost bits, recovery probability.",
    ),
    "erasure_profile": _Observable(
        _erasure_profile,
        ("system", "reference"),
        ("samples", "seed"),
        "Correctable fraction and mean lost bits for random erasures of every size 0..|system|.",
        show=_profiles_summary,
    ),
    "contiguous_code_distance": _Observable(
        lambda sim, a: observables.contiguous_code_distance(
            sim, a["system"], a["reference"], periodic=a.get("periodic", True)
        ),
        ("system", "reference"),
        ("periodic",),
        "Shortest contiguous run of system qubits whose erasure is not correctable.",
    ),
    "stabilizer_weights": _Observable(
        lambda sim, a: _weights_summary(observables.stabilizer_weights(sim)),
        (),
        (),
        "Mean/min/max generator weight (basis dependent; a sparsity diagnostic).",
    ),
    "stabilizer_length_distribution": _Observable(
        lambda sim, a: observables.stabilizer_length_distribution(sim, a.get("order")),
        (),
        ("order",),
        "Histogram of clipped-gauge stabilizer lengths (index = length).",
        show=_profile_summary,
    ),
    "stabilizers": _Observable(
        _stabilizer_strings,
        (),
        (),
        f"Canonical stabilizer generators as Pauli strings (up to {MAX_STABILIZER_QUBITS} "
        "qubits): signed with phases=true, unsigned ('XZ_') with phases=false.",
    ),
    "expectation": _Observable(
        lambda sim, a: sim.peek_pauli(a["pauli"]),
        ("pauli",),
        (),
        "<P> of a Pauli product like 'Z0*Z1' or '!X3': +1, -1, or 0 if random.",
        needs_phases=True,
    ),
    "is_deterministic": _Observable(
        lambda sim, a: sim.is_deterministic(a["pauli"]),
        ("pauli",),
        (),
        "Whether measuring a Pauli product (e.g. 'X0*X1') has a fixed outcome.",
    ),
}


def _compile_observables(
    specs: Any, num_qubits: int, phases: bool, seed: int
) -> list[tuple[str, Callable[[Simulator], Any], Callable[[Any], Any]]]:
    """Validate observable specs up front: ``(name, fn(sim) -> full value, show(value))``."""
    if specs is None:
        return []
    if not isinstance(specs, list):
        raise ValueError(
            'observables must be a list of objects like {"kind": "entropy", "region": [0, 1]}'
        )
    compiled: list[tuple[str, Callable[[Simulator], Any], Callable[[Any], Any]]] = []
    # Every result key is distinct: a name given explicitly is kept (a repeat gets a
    # suffix), and the automatic ones (the kind, then kind#2, kind#3, ...) skip every
    # name in use or given explicitly.
    explicit = {s["name"] for s in specs if isinstance(s, dict) and isinstance(s.get("name"), str)}
    used: set[str] = set()
    for i, spec in enumerate(specs):
        where = f"observables[{i}]"
        if not isinstance(spec, dict) or "kind" not in spec:
            raise ValueError(f'{where}: expected an object with a "kind", got {spec!r}')
        kind = _require_name(spec["kind"], f"{where}.kind", _OBSERVABLES)
        if kind not in _OBSERVABLES:
            raise ValueError(
                f"{where}: unknown observable kind {kind!r}; choose from {sorted(_OBSERVABLES)}"
            )
        obs = _OBSERVABLES[kind]
        _check_keys(spec, ("kind", "name", *obs.required, *obs.optional), f"{where} ({kind})")
        missing = [k for k in obs.required if k not in spec]
        if missing:
            raise ValueError(f"{where} ({kind}): missing required key(s) {missing}")
        if obs.needs_phases and not phases:
            raise ValueError(
                f"{where} ({kind}) needs sign tracking; run with phases=true (the default)"
            )
        args: dict[str, Any] = {}
        for key, value in spec.items():
            if key in ("kind", "name"):
                continue
            if key in _QUBIT_ARGS:
                args[key] = _qubit_list(value, f"{where}.{key}", num_qubits)
            elif key == "sizes":
                if not isinstance(value, list):
                    raise ValueError(f"{where}.sizes must be a list of erasure sizes")
                args[key] = [_require_int(v, f"{where}.sizes entries", minimum=0) for v in value]
            elif key in ("samples", "seed"):
                args[key] = _require_int(
                    value,
                    f"{where}.{key}",
                    minimum=1 if key == "samples" else 0,
                    maximum=MAX_INT if key == "samples" else None,
                )
            elif key == "periodic":
                args[key] = _require_bool(value, f"{where}.periodic")
            elif key == "pauli":
                if not isinstance(value, str):
                    raise ValueError(f"{where}.pauli must be a string like 'Z0*Z1'")
                args[key] = value
        if kind in ("random_erasure_statistics", "erasure_profile"):
            args.setdefault("samples", 100 if kind == "random_erasure_statistics" else 10)
            args.setdefault("seed", seed)
        given = spec.get("name")
        name = given or kind
        if not isinstance(name, str):
            raise ValueError(f"{where}.name must be a string")
        base, copy = name, 2
        while name in used or (name in explicit and name != given):
            name, copy = f"{base}#{copy}", copy + 1
        used.add(name)

        def fn(sim: Simulator, _obs=obs, _args=args, _where=where, _kind=kind) -> Any:
            try:
                return _obs.compute(sim, _args)
            except (ValueError, IndexError, PhaseFreeError) as exc:
                raise ValueError(f"{_where} ({_kind}): {exc}") from None

        compiled.append((name, fn, obs.show or (lambda value: value)))
    return compiled


# ---------------------------------------------------------------------------
# circuits
# ---------------------------------------------------------------------------


def _applications(ins: Instruction) -> int:
    info = gate(ins.name)
    if info.kind == "annotation":
        return 1
    if info.kind == "measure_product":
        return sum(1 for t in ins.targets if t.kind == "pauli") - sum(
            1 for t in ins.targets if t.kind == "combiner"
        )
    return len(ins.targets) // max(info.num_qubits, 1)


def _histogram(circuit: Circuit, multiplier: int, hist: Counter[str]) -> int:
    """Fill ``hist`` with applications per instruction name; returns the REPEAT block count."""
    blocks = 0
    for item in circuit:
        if isinstance(item, RepeatBlock):
            blocks += 1 + _histogram(item.body, multiplier * item.count, hist)
        else:
            hist[item.name] += multiplier * _applications(item)
    return blocks


_MEASURE_KINDS = {"measure", "measure_reset", "measure_pair", "measure_product"}
_NO_OP_ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}


def _unrolled_ops(hist: Counter[str]) -> int:
    """Kernel operations the compiled circuit needs (annotations other than TICK are free)."""
    return sum(count for name, count in hist.items() if name not in _NO_OP_ANNOTATIONS)


def _num_observables(circuit: Circuit) -> int:
    """Number of logical observables (largest OBSERVABLE_INCLUDE index + 1), without unrolling."""
    top = -1
    for item in circuit:
        if isinstance(item, RepeatBlock):
            top = max(top, _num_observables(item.body) - 1)
        elif item.name == "OBSERVABLE_INCLUDE":
            top = max(top, int(item.args[0]))
    return top + 1


def _semantic_errors(circuit: Circuit, text: str) -> list[dict[str, Any]]:
    """Errors the parser cannot see: record lookbacks before the start, and size.

    ``circuit`` must have been parsed from ``text``: each non-blank line of the
    text is then one item (instruction, ``REPEAT ... {`` or ``}``) in
    traversal order, which maps items back to line numbers. A lookback is
    checked in the first iteration of a REPEAT body, where the fewest
    measurements precede it; REPEAT blocks are never unrolled.
    """
    lines = iter(
        lineno for lineno, raw in enumerate(text.splitlines(), start=1) if _strip_comment(raw)
    )
    errors: list[dict[str, Any]] = []

    def walk(c: Circuit, measured: int) -> int:
        for item in c:
            if isinstance(item, RepeatBlock):
                next(lines, None)  # the REPEAT header
                walk(item.body, measured)
                next(lines, None)  # the closing brace
                measured += item.count * item.body.num_measurements
                continue
            line = next(lines, None)
            for t in item.targets:
                if t.kind == "rec" and measured + t.value < 0 and len(errors) < MAX_ERRORS_SHOWN:
                    errors.append(
                        {
                            "line": line,
                            "message": f"rec[{t.value}] refers to a measurement before the start "
                            f"of the circuit (only {measured} recorded before this line)",
                        }
                    )
                    break
            if gate(item.name).kind in _MEASURE_KINDS:
                measured += _applications(item)
        return measured

    walk(circuit, 0)
    hist: Counter[str] = Counter()
    _histogram(circuit, 1, hist)
    ops = _unrolled_ops(hist)
    if ops > DEFAULT_MAX_OPS:
        errors.append(
            {
                "line": None,
                "message": f"circuit unrolls to {ops:,} operations, more than the limit of "
                f"{DEFAULT_MAX_OPS:,}; reduce REPEAT counts or split the circuit",
            }
        )
    return errors


_LINE_ERROR_RE = re.compile(r"^line (\d+): (.*)$", re.DOTALL)


def _circuit_errors(text: str, first_error: str) -> list[dict[str, Any]]:
    """Every line that fails on its own, plus the whole-text error if it is structural."""
    errors: dict[int | None, str] = {}
    m = _LINE_ERROR_RE.match(first_error)
    if m:
        errors[int(m.group(1))] = m.group(2)
    else:
        errors[None] = first_error
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw)
        if not line or line == "}" or line.upper().startswith("REPEAT"):
            continue
        try:
            Circuit(line)
        except (ValueError, KeyError) as exc:
            msg = exc.args[0] if exc.args else str(exc)
            errors.setdefault(lineno, _LINE_ERROR_RE.sub(r"\2", msg))
        if len(errors) >= MAX_ERRORS_SHOWN:
            break
    ordered = sorted(errors.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
    return [{"line": line, "message": msg} for line, msg in ordered]


def inspect_circuit(circuit: str) -> dict[str, Any]:
    """Parse a Stim-language circuit and describe it, without running it.

    Returns ``valid`` and, for a valid circuit, ``num_qubits``,
    ``num_measurements``, ``num_detectors``, ``num_observables``,
    ``num_ticks``, ``repeat_blocks``, an ``instruction_histogram`` (number of
    applications of each instruction, REPEAT blocks unrolled), the
    ``extensions`` used (``CLIFFORD_1Q``/``CLIFFORD_2Q``, which Stim lacks) and
    the normalized circuit text (``text``, truncated if long). For an invalid
    circuit, ``errors`` lists ``{"line", "message"}`` for every bad line,
    including record lookbacks before the first measurement; ``line`` is
    ``None`` for whole-circuit problems (an unclosed REPEAT, or more than
    ``DEFAULT_MAX_OPS`` operations once unrolled). Nothing is unrolled here,
    so huge REPEAT counts are cheap to inspect.
    """
    if not isinstance(circuit, str):
        raise ValueError(
            f"circuit must be Stim circuit text (a string), got {type(circuit).__name__}"
        )
    try:
        parsed = Circuit(circuit)
    except (ValueError, KeyError) as exc:
        msg = exc.args[0] if exc.args else str(exc)
        return _bounded({"valid": False, "errors": _circuit_errors(circuit, msg)})
    errors = _semantic_errors(parsed, circuit)
    if errors:
        return _bounded({"valid": False, "errors": errors})
    hist: Counter[str] = Counter()
    blocks = _histogram(parsed, 1, hist)
    kinds: Counter[str] = Counter()
    for name, count in hist.items():
        kinds[gate(name).kind] += count
    text = str(parsed)
    extensions = sorted(n for n in hist if gate(n).kind == "clifford_index")
    out: dict[str, Any] = {
        "valid": True,
        "num_qubits": parsed.num_qubits,
        "num_measurements": parsed.num_measurements,
        "num_detectors": parsed.num_detectors,
        "num_observables": _num_observables(parsed),
        "num_ticks": parsed.num_ticks,
        "num_operations": _unrolled_ops(hist),
        "repeat_blocks": blocks,
        "instruction_histogram": dict(sorted(hist.items(), key=lambda kv: -kv[1])[:MAX_LIST_SHOWN]),
        "kind_totals": dict(kinds),
        "extensions": extensions,
        "stim_compatible": not extensions,
        "text": text[:MAX_TEXT_SHOWN],
        "text_truncated": len(text) > MAX_TEXT_SHOWN,
    }
    notes = []
    if len(hist) > MAX_LIST_SHOWN:
        out["truncated"] = True
        notes.append(
            f"instruction_histogram shows the {MAX_LIST_SHOWN} most frequent of {len(hist)} "
            "instructions; kind_totals covers all of them"
        )
    if extensions:
        notes.append(
            "CLIFFORD_1Q/CLIFFORD_2Q are SparseGF2 extensions; Circuit.to_stim() expands them "
            "into Stim gates"
        )
    if notes:
        out["notes"] = notes
    return _bounded(out)


def run_circuit(
    circuit: str,
    shots: int = 1,
    phases: bool = True,
    seed: int | None = None,
    observables: list[dict[str, Any]] | None = None,
    observe_ticks: bool = False,
    tick_stride: int | None = None,
    num_qubits: int | None = None,
    output_dir: str | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Run a Stim-language circuit from |0...0> and summarize what happened.

    Parameters
    ----------
    circuit
        Stim circuit text; SparseGF2's ``CLIFFORD_1Q(i)``/``CLIFFORD_2Q(i)``
        extensions are accepted.
    shots
        With ``shots=1`` the circuit runs once on
        ``Simulator(num_qubits, phases=phases, seed=seed)`` and the full
        record is returned when small. With ``shots > 1`` it is run with
        ``sparsegf2.sample(circuit, shots, seed=seed, phases=phases)`` (in
        parallel) and per-measurement / per-detector rates are returned.
    phases
        Track signs (default). ``False`` is about twice as fast but measurement
        outcomes, detectors and expectation values do not exist; only which
        measurements were random is reported.
    seed
        Non-negative int, or ``None`` for a fresh seed (reported back).
    observables
        Quantities to evaluate on the final state, as
        ``[{"kind": "entropy", "region": [0, 1]}, ...]``; see
        ``capabilities()["observables"]``. They are evaluated on the state of
        the single-shot run ``Simulator(..., seed=seed).run(circuit)``
        whatever ``shots`` is.
    observe_ticks
        Also evaluate the observables after ``TICK`` instructions (time series
        in ``tick_observables``: at most 50 ticks shown, all evaluated ones
        saved with ``output_dir``). Circuits with up to
        :data:`MAX_OBSERVED_TICKS` ticks are supported.
    tick_stride
        With ``observe_ticks``, evaluate after ticks ``0, k, 2k, ...`` and the
        last one. Default: every tick, or the smallest stride that keeps at
        most :data:`MAX_TICK_ROWS` evaluations, the last tick included.
    num_qubits
        Simulate more qubits than the circuit touches (idle qubits).
    output_dir
        If given, write the full arrays to ``output_dir/circuit_run.npz``
        together with ``circuit.stim`` and ``provenance.json``.
    command
        Free text recorded in ``provenance.json`` (e.g. the CLI command line).
    """
    parsed = _parse_circuit(circuit)
    shots = _require_int(shots, "shots", minimum=1)
    phases = _require_bool(phases, "phases")
    observe_ticks = _require_bool(observe_ticks, "observe_ticks")
    seed = _resolve_seed(seed)
    n = max(parsed.num_qubits, 1)
    if num_qubits is not None:
        requested = _require_int(num_qubits, "num_qubits", minimum=1)
        if requested < parsed.num_qubits:
            raise ValueError(
                f"num_qubits={requested} is smaller than the {parsed.num_qubits} qubits the "
                "circuit uses"
            )
        n = requested
    if n > MAX_QUBITS:
        raise ValueError(f"at most {MAX_QUBITS} qubits are supported here, got {n}")
    m = parsed.num_measurements
    if shots * m > MAX_SHOT_RESULTS:
        raise ValueError(
            f"shots x measurements = {shots * m:,} exceeds {MAX_SHOT_RESULTS:,}; "
            "use fewer shots per call and combine the results"
        )
    compiled = _compile_observables(observables, n, phases, seed)
    num_ticks = parsed.num_ticks
    stride = _tick_stride(observe_ticks, tick_stride, num_ticks, bool(compiled))

    out: dict[str, Any] = {
        "num_qubits": n,
        "num_measurements": m,
        "num_detectors": parsed.num_detectors,
        "num_observables": _num_observables(parsed),
        "num_ticks": num_ticks,
        "shots": shots,
        "phases": phases,
        "seed": seed,
    }
    notes: list[str] = _seed_notes(seed)
    arrays: dict[str, np.ndarray] = {}
    tick_rows: list[tuple[int, dict[str, Any]]] = []
    final_values: dict[str, Any] = {}
    start = time.perf_counter()

    def evaluate(sim: Simulator) -> dict[str, Any]:
        return {name: fn(sim) for name, fn, _ in compiled}

    def shown(values: dict[str, Any]) -> dict[str, Any]:
        return {name: show(values[name]) for name, _, show in compiled}

    def observe(sim: Simulator, tick: int) -> dict[str, Any] | None:
        if tick % stride == 0 or tick == num_ticks - 1:
            return evaluate(sim)
        return None

    with _COMPUTE_LOCK:
        if shots == 1 or compiled:
            sim = Simulator(n, phases=phases, seed=seed)
            single = sim.run(parsed, observe=observe if observe_ticks else None)
            if compiled:
                final_values = evaluate(sim)
            tick_rows = [(t, v) for t, v in single.observations if v is not None]
        if shots == 1:
            random_flags = single.random[None, :]
            records = single.measurements[None, :] if phases else None
            detectors = single.detectors[None, :] if phases else None
            obs_flips = single.observables[None, :] if phases else None
        else:
            many = sample(parsed, shots, seed=seed, phases=phases)
            random_flags = many.random
            records = many.measurements if phases else None
            detectors = many.detectors if phases else None
            obs_flips = many.observables if phases else None
    out["elapsed_seconds"] = time.perf_counter() - start

    out["random_fraction"] = float(random_flags.mean()) if random_flags.size else 0.0
    arrays["random"] = random_flags
    if not phases:
        note = (
            "phases=false: measurement outcomes, detectors and observable flips do not exist; "
            "only which measurements were random is reported"
        )
        if any(isinstance(o, dict) and o.get("kind") == "stabilizers" for o in observables or []):
            note += "; stabilizers are listed without signs (they are not tracked)"
        notes.append(note)
    cut: list[str] = []
    if shots == 1:
        if m <= MAX_BITS_SHOWN:
            out["random"] = _bits(random_flags)
            if records is not None:
                out["measurements"] = _bits(records)
        elif m:
            out["num_random"] = int(random_flags.sum())
            if records is not None:
                out["num_ones"] = int(records.sum())
                out["measurement_mean"] = float(records.mean())
            cut.append(f"the {m}-bit measurement record")
        if detectors is not None and detectors.shape[1]:
            if detectors.shape[1] <= MAX_BITS_SHOWN:
                out["detectors"] = _bits(detectors)
            else:
                cut.append(f"the {detectors.shape[1]} detector bits")
            out["detector_fraction"] = float(detectors.mean())
        if obs_flips is not None and obs_flips.shape[1]:
            out["observable_flips"] = _bits(obs_flips[:, :MAX_BITS_SHOWN])
            if obs_flips.shape[1] > MAX_BITS_SHOWN:
                cut.append(f"observable_flips beyond the first {MAX_BITS_SHOWN}")
    else:
        if records is not None and m:
            out["measurement_mean"] = float(records.mean())
            if m <= MAX_LIST_SHOWN:
                out["measurement_means"] = records.mean(axis=0)
                out["first_shots"] = [_bits(r) for r in records[:4]]
            else:
                cut.append(f"per-measurement means of {m} measurements")
        if detectors is not None and detectors.shape[1]:
            rates = detectors.mean(axis=0)
            out["detector_rate_mean"] = float(rates.mean())
            out["detector_rate_max"] = float(np.max(rates))
            out["any_detector_fraction"] = float(detectors.any(axis=1).mean())
            if rates.size <= MAX_LIST_SHOWN:
                out["detector_rates"] = rates
            else:
                cut.append(f"per-detector rates of {rates.size} detectors")
        if obs_flips is not None and obs_flips.shape[1]:
            out["observable_flip_rates"] = obs_flips.mean(axis=0)[:MAX_LIST_SHOWN]
            if obs_flips.shape[1] > MAX_LIST_SHOWN:
                cut.append(f"observable_flip_rates beyond the first {MAX_LIST_SHOWN}")
    if records is not None and detectors is not None and obs_flips is not None:
        arrays["measurements"] = records
        arrays["detectors"] = detectors
        arrays["observable_flips"] = obs_flips

    if compiled:
        out["observables"] = shown(final_values)
        if _subsampled(out["observables"]):
            cut.append(f"profile entries beyond {MAX_PROFILE_SHOWN} per observable")
        if shots > 1:
            notes.append("observables are evaluated on the final state of the shots=1 run")
    if observe_ticks:
        idx = _subsample(len(tick_rows), MAX_CHECKPOINTS_SHOWN)
        rows = [shown(tick_rows[i][1]) for i in idx]
        out["tick_observables"] = {
            "ticks": [tick_rows[i][0] for i in idx],
            "values": {name: [row[name] for row in rows] for name, _, _ in compiled},
            "num_ticks": num_ticks,
            "num_evaluated": len(tick_rows),
            "tick_stride": stride,
            "subsampled": len(idx) < len(tick_rows),
        }
        if any(_subsampled(row) for row in rows):
            cut.append("profile entries in tick_observables")
    if output_dir is not None:
        out["files"] = _save_circuit_run(
            output_dir, parsed, arrays, tick_rows, final_values, seed, command
        )
    if cut:
        out["truncated"] = True
        where = (
            "every value is in the saved files listed under 'files'"
            if output_dir is not None
            else "pass output_dir to save every value"
        )
        notes.append(f"not shown: {'; '.join(cut)}; {where}")
    if notes:
        out["notes"] = notes
    return _bounded(out)


def _tick_stride(observe_ticks: bool, tick_stride: Any, num_ticks: int, has_obs: bool) -> int:
    """Validate the observe_ticks options and pick the stride between evaluated ticks."""
    if not observe_ticks:
        if tick_stride is not None:
            raise ValueError("tick_stride only applies with observe_ticks=true")
        return 1
    if not has_obs:
        raise ValueError("observe_ticks needs at least one entry in observables")
    if num_ticks > MAX_OBSERVED_TICKS:
        raise ValueError(
            f"observe_ticks supports circuits with at most {MAX_OBSERVED_TICKS:,} TICKs "
            f"(this one has {num_ticks:,}); split the circuit into segments"
        )
    # Stride s evaluates ticks 0, s, 2s, ... and the last one, ceil((num_ticks - 1) / s) + 1
    # in all; the smallest s that keeps this within MAX_TICK_ROWS:
    needed = max(1, -(-(num_ticks - 1) // (MAX_TICK_ROWS - 1)))
    if tick_stride is None:
        return needed
    stride = _require_int(tick_stride, "tick_stride", minimum=1)
    if stride < needed:
        evaluated = -(-(num_ticks - 1) // stride) + 1
        raise ValueError(
            f"tick_stride={stride} would evaluate {evaluated:,} of {num_ticks:,} ticks, more "
            f"than {MAX_TICK_ROWS:,}; use tick_stride >= {needed}"
        )
    return stride


def _save_circuit_run(
    output_dir: str,
    circuit: Circuit,
    arrays: dict[str, np.ndarray],
    tick_rows: list,
    final_values: dict,
    seed: int,
    command: str | None,
) -> dict[str, str]:
    out = _output_dir(output_dir)
    with _writing_to(out):
        npz = out / "circuit_run.npz"
        _atomic_savez(npz, {k: v for k, v in arrays.items() if v is not None})
        stim_path = out / "circuit.stim"
        stim_path.write_text(str(circuit) + "\n", encoding="utf-8")
        files = {"arrays": str(npz), "circuit": str(stim_path)}
        if final_values or tick_rows:
            obs_path = out / "observables.json"
            values = {"final": final_values, "ticks": [{"tick": t, **v} for t, v in tick_rows]}
            obs_path.write_text(
                json.dumps(_jsonable(values, exact=True), indent=1), encoding="utf-8"
            )
            files["observables"] = str(obs_path)
        prov = out / _PROVENANCE_FILE
        prov.write_text(
            json.dumps(_provenance(seed, command or "run_circuit"), indent=2), encoding="utf-8"
        )
        files["provenance"] = str(prov)
    return files


# ---------------------------------------------------------------------------
# ensembles: spec translation
# ---------------------------------------------------------------------------

_GRAPH_HELPERS = {
    "from_edges",
    "from_networkx",
    "degrees",
    "edge_coloring",
    "stats",
    "shortcut_fraction",
    "save",
    "load",
}


def _graph_families() -> dict[str, Callable]:
    return {
        name: getattr(graphs, name)
        for name in graphs.__all__
        if name not in _GRAPH_HELPERS
        and not name.endswith("_weights")
        and callable(getattr(graphs, name))
    }


def _weight_families() -> dict[str, Callable]:
    return {
        name.removesuffix("_weights"): getattr(graphs, name)
        for name in graphs.__all__
        if name.endswith("_weights")
    }


def _probe_kinds() -> dict[str, Callable]:
    """Probe constructors of :mod:`sparsegf2.mipt`: the functions that return a Probe."""
    kinds: dict[str, Callable] = {}
    for name in mipt.__all__:
        fn = getattr(mipt, name)
        if inspect.isfunction(fn) and inspect.signature(fn).return_annotation in (
            "Probe",
            mipt.Probe,
        ):
            kinds[name] = fn
    return kinds


def _spec_fields() -> dict[str, Any]:
    """MonitoredCircuit fields and their defaults (``...`` marks a required field)."""
    out = {}
    for f in dataclasses.fields(mipt.MonitoredCircuit):
        if f.name.startswith("_"):
            continue
        out[f.name] = f.default if f.default is not dataclasses.MISSING else ...
    return out


def _params(fn: Callable) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(fn).parameters)


def _call_with(fn: Callable, given: dict, fill: dict, what: str) -> Any:
    """Call ``fn`` with keys from ``given`` plus automatic ``fill`` values, validating names."""
    params = _params(fn)
    settable = [p for p in params if p not in fill]
    _check_keys(given, settable, what)
    kwargs = {k: v for k, v in fill.items() if k in params}
    for key, value in given.items():
        if key == "seed" and params[key].annotation is inspect.Parameter.empty:
            kwargs[key] = (
                [_require_int(v, f"{what}: seed entries", minimum=0, maximum=None) for v in value]
                if isinstance(value, list) and value
                else _require_int(value, f"{what}: seed", minimum=0, maximum=None)
            )
        else:
            kwargs[key] = _check_type(value, params[key].annotation, f"{what}: {key}")
    missing = [
        p.name
        for p in params.values()
        if p.default is inspect.Parameter.empty and p.name not in kwargs
    ]
    if missing:
        raise ValueError(f"{what}: missing required parameter(s) {missing}")
    try:
        return fn(**kwargs)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{what}: {exc}") from None


def _read_graph_file(graph: dict) -> tuple[int, list[np.ndarray], list[np.ndarray] | None]:
    """``(num_nodes, graphs, weights)`` of a ``{"family": "file", "path": ...}`` graph."""
    extra = sorted(set(graph) - {"family", "path"})
    if extra:
        raise ValueError(f"graph (file): unknown key(s) {extra}; give only 'path'")
    path = graph.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError('graph (file): give "path", a .json or .npz graph file')
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"graph (file): no such file {str(p)!r}")
    try:
        return graphs.load(p)
    except OSError as exc:
        raise _unreadable_graph_file(p, exc) from None


def _file_graph(graph: dict, n: int) -> tuple[np.ndarray | list[np.ndarray], Any]:
    """Edges (one array, or a list for a collection) and weights of a ``{"family": "file"}`` graph."""
    num_nodes, gs, ws = _read_graph_file(graph)
    p = Path(graph["path"]).expanduser().resolve()
    if num_nodes != n:
        raise ValueError(f"graph (file): {p.name} has num_nodes {num_nodes}, but num_system is {n}")
    if len(gs) == 1:
        return gs[0], (None if ws is None else ws[0])
    return gs, ws


def _graph_file_record(spec: dict) -> dict[str, str] | None:
    """The resolved path and the sha256 of the bytes of a ``{"family": "file"}`` graph.

    Recorded in ``spec.json`` beside the spec, so that a resume can tell
    whether the file still holds the graphs the run started with. ``spec`` has
    been built already, so the path names a readable file.
    """
    graph = spec.get("graph")
    if not (isinstance(graph, dict) and graph.get("family") == "file"):
        return None
    path = Path(graph["path"]).expanduser().resolve()
    try:
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
    except OSError as exc:
        raise _unreadable_graph_file(path, exc) from None
    return {"path": str(path), "sha256": digest}


def _unreadable_graph_file(path: Path, exc: OSError) -> ValueError:
    """An OSError reading a graph file, as the input error it is."""
    return ValueError(
        f"graph (file): could not read {str(path)!r}: {exc}; close any program that holds "
        "it open, or check its permissions"
    )


def _build_graph(graph: Any, n: int) -> np.ndarray | list[np.ndarray]:
    """One edge array, or a collection of graphs (a list) when ``realizations`` is given.

    With ``"realizations": G`` a random family yields ``G`` independent graphs,
    graph ``i`` generated from seed ``(seed, i)``. The ensemble then runs
    ``circuits_per_graph`` circuits (default 1) on each graph: trajectory ``t``
    uses graph ``t // circuits_per_graph``, and all ``G * circuits_per_graph``
    trajectories run unless fewer are asked for.
    """
    if isinstance(graph, dict) and graph.get("family") == "file":
        return _file_graph(graph, n)[0]
    families = _graph_families()
    if not isinstance(graph, dict) or "family" not in graph:
        raise ValueError(
            'graph must be an object like {"family": "watts_strogatz", "k": 4, "beta": 0.1, '
            f'"seed": 1}}; families: {sorted(families)}'
        )
    params = dict(graph)
    realizations = params.pop("realizations", None)
    if realizations is not None:
        realizations = _require_int(realizations, "graph.realizations", minimum=1)
        if realizations > MAX_GRAPH_REALIZATIONS:
            raise ValueError(
                f"graph.realizations must be at most {MAX_GRAPH_REALIZATIONS}; run more "
                "circuits per graph (circuits_per_graph) or pass explicit edges in Python"
            )
        if "seed" not in params:
            raise ValueError("graph.realizations needs a random family with its own 'seed'")
        family = params.get("family")
        if isinstance(family, str):  # any other family value is reported by the first graph
            _check_graph_memory(family, params, n, realizations)  # before any graph exists
        seed = params.pop("seed")
        base = [_require_int(seed, "graph.seed", minimum=0)] if not isinstance(seed, list) else seed
        bank: list[np.ndarray] = []
        for i in range(realizations):
            one = _build_graph({**params, "seed": [*base, i]}, n)
            assert isinstance(one, np.ndarray), "without realizations a family yields one graph"
            bank.append(one)
        return bank
    family = _require_name(params.pop("family"), "graph.family", families)
    if family not in families:
        raise ValueError(f"unknown graph family {family!r}; choose from {sorted(families)}")
    fn = families[family]
    if "seed" in _params(fn) and "seed" not in params:
        raise ValueError(
            f"graph family {family!r} is random: give it its own 'seed' (graph disorder is "
            "kept separate from the circuit seed)"
        )
    if family == "lattice_2d":
        rows, cols = params.get("rows"), params.get("cols")
        if rows is None or cols is None:
            raise ValueError("graph (lattice_2d): give rows and cols (rows * cols = num_system)")
        rows = _require_int(rows, "graph (lattice_2d): rows", minimum=1)
        cols = _require_int(cols, "graph (lattice_2d): cols", minimum=1)
        if rows * cols != n:
            raise ValueError(f"graph (lattice_2d): rows * cols must equal num_system = {n}")
    _check_graph_memory(family, params, n, None)
    return _call_with(fn, params, {"n": n}, f"graph ({family})")


_EDGE_BYTES = 16  # one edge of an (m, 2) int64 edge array, as a collection keeps it

# Peak bytes allocated while one graph is generated, per pair drawn (``complete`` and
# ``erdos_renyi`` build every pair: triu_indices, the stacked pairs and their int64 copy are
# 48 bytes per pair) or per edge of the bound (the Python tuples and sets of the ring
# families, about 390 bytes per edge). Measured with tracemalloc at n = 2048 to 8192 and
# rounded up; plus a fixed allowance per graph for the interpreter's own objects (up to
# about 8 KB at small n).
_GENERATION_BYTES = {
    "cycle": 128,
    "path": 64,
    "complete": 48,
    "erdos_renyi": 48,
    "lattice_2d": 256,
    "watts_strogatz": 512,
    "newman_watts": 512,
    "random_regular": 128,
}
_GENERATION_OVERHEAD = 64 * 1024


def _graph_edges(family: str, params: dict, n: int) -> tuple[int, int] | None:
    """At most how many edges one graph of ``family`` on ``n`` nodes keeps, and how many
    pairs or edges generating it materializes, from its parameters alone (before it exists).

    None for a family not listed here, or when the family rejects its parameters at this n
    (a degree that is not an integer in its range, an odd ``n * d``): the family itself then
    reports the parameter.
    """
    pairs = n * (n - 1) // 2

    def number(key: str) -> float | None:
        v = params.get(key)
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)):
            return None
        try:
            x = float(v)
        except OverflowError:  # an int beyond any float: out of every family's range
            return None
        return x if math.isfinite(x) else None

    if family == "cycle":
        return n, n
    if family == "path":
        return n - 1, n - 1
    if family == "complete":
        return pairs, pairs
    if family == "lattice_2d":
        return 2 * n, 2 * n
    if family == "erdos_renyi":
        # every pair is drawn (so materialized); the edges kept are binomial: mean plus
        # six standard deviations
        p = number("p")
        if p is None or not 0.0 <= p <= 1.0:
            return None
        mean = p * pairs
        return min(pairs, math.ceil(mean + 6.0 * math.sqrt(mean * (1.0 - p)) + 1.0)), pairs
    if family in ("watts_strogatz", "newman_watts"):
        k = number("k")  # the ring lattice needs an even k with 2 <= k < n
        if k is None or k != int(k) or int(k) % 2 or not 2 <= k < n:
            return None
        ring = n * int(k) // 2
        # Newman-Watts keeps the ring and adds at most one shortcut per ring edge
        edges = min(pairs, 2 * ring if family == "newman_watts" else ring)
        return edges, edges
    if family == "random_regular":
        d = number("d")  # an integer degree 1 <= d < n, with n * d even
        if d is None or d != int(d) or not 1 <= d < n or (n * int(d)) % 2:
            return None
        return n * int(d) // 2, n * int(d) // 2
    return None


def _generation_bytes(family: str, params: dict, n: int) -> int | None:
    """Peak bytes allocated while one graph of ``family`` on ``n`` nodes is generated."""
    counts = _graph_edges(family, params, n)
    if counts is None:
        return None
    return _GENERATION_BYTES[family] * counts[1] + _GENERATION_OVERHEAD


def _check_graph_memory(family: str, params: dict, n: int, realizations: int | None) -> None:
    """Refuse graphs that would take more than :data:`MAX_GRAPH_GB`, before generating any.

    Checks generating one graph (what generation allocates at its peak), and keeping
    ``realizations`` of them (16 bytes per edge; nothing when ``realizations`` is None).
    """
    counts = _graph_edges(family, params, n)
    if counts is None:
        return
    kept, drawn = counts
    limit = MAX_GRAPH_GB * 1e9
    per = _GENERATION_BYTES[family]
    generating = per * drawn + _GENERATION_OVERHEAD
    if generating > limit:
        what = "pairs drawn" if family in ("complete", "erdos_renyi") else "edges"
        largest = _largest_fitting_n(family, params, n, limit)
        fits = (
            f"at most num_system = {largest:,} fits for this family and these parameters; "
            "choose a smaller num_system or a sparser graph"
            if largest is not None
            else "no num_system fits these parameters; choose a sparser graph (a smaller k or d)"
        )
        raise ValueError(
            f"graph ({family}): one graph on num_system = {n:,} qubits takes about "
            f"{generating / 1e9:.1f} GB while it is generated ({drawn:,} {what} at about "
            f"{per} bytes each; limit {MAX_GRAPH_GB:g} GB): {fits}"
        )
    if realizations is not None and _EDGE_BYTES * kept * realizations > limit:
        raise ValueError(
            f"graph.realizations = {realizations:,} graphs on num_system = {n:,} qubits would "
            f"take about {_EDGE_BYTES * kept * realizations / 1e9:.1f} GB of edges ({kept:,} "
            f"edges each at {_EDGE_BYTES} bytes; limit {MAX_GRAPH_GB:g} GB): at most "
            f"{int(limit // (_EDGE_BYTES * kept)):,} realizations fit at num_system = {n:,}; "
            "use fewer graphs with more circuits_per_graph each, or a smaller num_system"
        )


def _largest_fitting_n(family: str, params: dict, n: int, limit: float) -> int | None:
    """The largest num_system below ``n`` that the family accepts with these parameters and
    at which one graph fits in ``limit``; None if there is none."""

    def accepted(m: int) -> bool:
        return _graph_edges(family, params, m) is not None

    def small(m: int) -> bool:
        # A size the family rejects is priced as its neighbor below (random_regular rejects
        # an odd n * d) or, when that is rejected too, counts as small: below the range.
        size = _generation_bytes(family, params, m)
        if size is None:
            size = _generation_bytes(family, params, m - 1)
        return size is None or size <= limit

    # Every family's bound grows with n and its range is n above some minimum, so small()
    # holds up to one size and fails above it.
    low, high = 2, n
    while high - low > 1:
        mid = (low + high) // 2
        if small(mid):
            low = mid
        else:
            high = mid
    for m in (low, low - 1):  # an odd n * d is next to an even one
        if m >= 2 and accepted(m) and small(m):
            return m
    return None


def _build_weights(weights: Any, edges: np.ndarray, n: int) -> np.ndarray:
    if isinstance(weights, list):
        if not all(isinstance(w, (int, float)) and not isinstance(w, bool) for w in weights):
            raise ValueError("edge_weights as a list must hold one number per edge")
        return np.asarray(weights, dtype=np.float64)
    families = _weight_families()
    if not isinstance(weights, dict) or "family" not in weights:
        raise ValueError(
            "edge_weights must be a list (one weight per edge) or an object like "
            f'{{"family": "power_law", "alpha": 2.0}}; families: {sorted(families)}'
        )
    params = dict(weights)
    family = _require_name(params.pop("family"), "edge_weights.family", families)
    if family not in families:
        raise ValueError(f"unknown edge_weights family {family!r}; choose from {sorted(families)}")
    return _call_with(
        families[family], params, {"edges": edges, "n": n}, f"edge_weights ({family})"
    )


# Probe arguments that may map the system size to a value; JSON object keys are strings,
# so {"64": 3} means n = 64.
_SIZE_MAP_ARGS = {"max_size", "budget"}


def _size_map(value: dict, what: str) -> dict:
    """``{"64": 3, ...}`` keyed by num_system, as ``{64: 3, ...}``; the probe checks the values."""
    out: dict = {}
    for key, v in value.items():
        if isinstance(key, str) and re.fullmatch(r"[0-9]+", key):
            n = int(key)
        elif isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            n = int(key)  # from Python callers
        else:
            raise ValueError(
                f'{what}: the keys are system sizes, written as positive integers like "64", '
                f"got {key!r}"
            )
        if n in out:
            raise ValueError(f"{what} gives the system size {n} twice")
        out[n] = v
    return out


def _build_probe(spec: Any, index: int, num_system: int) -> mipt.Probe:
    """One JSON probe as a :class:`~sparsegf2.mipt.Probe`, with its exhaustive size resolved
    for ``num_system`` so that every error names the probe as ``probes[i]``."""
    kinds = _probe_kinds()
    where = f"probes[{index}]"
    if not isinstance(spec, dict) or "kind" not in spec:
        raise ValueError(
            f'{where}: expected an object like {{"kind": "code_dimension"}}; kinds: {sorted(kinds)}'
        )
    params = dict(spec)
    kind = _require_name(params.pop("kind"), f"{where}.kind", kinds)
    if kind not in kinds:
        raise ValueError(f"{where}: unknown probe kind {kind!r}; choose from {sorted(kinds)}")
    name = params.pop("name", None)
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError(f"{where}.name must be a non-empty string, got {name!r}")
    what = f"{where} ({kind})"
    # unknown keys first: their values may not be what the conversions below expect
    _check_keys(params, [k for k in _params(kinds[kind]) if k != "name"], what)
    for key, value in list(params.items()):
        if key in _QUBIT_ARGS and value is not None:
            params[key] = _qubit_list(value, f"{where}.{key}")
        if key in _SIZE_MAP_ARGS and isinstance(value, dict):
            params[key] = _size_map(value, f"{where}.{key}")
    if params.get("checkpoints") is not None:
        params["checkpoints"] = _checkpoints(params["checkpoints"], f"{where}.checkpoints")
    probe = _call_with(kinds[kind], params, {"name": None}, what)
    if name is not None:
        probe = dataclasses.replace(probe, name=name)
    if probe.kind == "uncorrectable_counts":
        # Resolved here so the errors say probes[i]; the spec's own resolution is then a no-op.
        try:
            probe = mipt._counts_size(probe, num_system)
        except ValueError as exc:
            raise ValueError(f"{what}: {exc}") from None
    if probe.kind == "correlated_erasure":  # checked here too, so the errors say probes[i]
        try:
            mipt._correlated_resolved(probe, num_system)
        except ValueError as exc:
            raise ValueError(f"{what}: {exc}") from None
    return probe


def _gate_set(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(
            "gate_set must be a non-empty list of two-qubit gate names (e.g. ['CX', 'SWAP']) "
            "or canonical CLIFFORD_2Q indices in [0, 11520)"
        )
    out = []
    for g in value:
        if isinstance(g, str):
            try:
                info = gate(g)
            except KeyError:
                raise ValueError(f"gate_set: unknown gate {g!r}") from None
            if info.kind != "unitary" or info.num_qubits != 2:
                raise ValueError(f"gate_set: {g!r} is not a two-qubit unitary gate")
            assert info.clifford is not None, "unitary gates carry their Clifford"
            out.append(info.clifford.index())
        else:
            out.append(_require_int(g, "gate_set entries", minimum=0))
    return np.asarray(out, dtype=np.int64)


def _build_spec(spec: Any) -> mipt.MonitoredCircuit:
    """Translate a JSON ensemble spec into a :class:`~sparsegf2.mipt.MonitoredCircuit`."""
    if not isinstance(spec, dict):
        raise ValueError("spec must be an object (dict) of MonitoredCircuit parameters")
    fields = _spec_fields()
    _check_keys(spec, [*fields, "graph"], "spec")
    if ("graph" in spec) == ("edges" in spec):
        raise ValueError(
            'spec needs exactly one of "graph" (e.g. {"family": "cycle"}) or "edges" '
            "(a list of [a, b] pairs)"
        )
    missing = [k for k, d in fields.items() if d is ... and k not in spec and k != "edges"]
    if missing:
        raise ValueError(f"spec: missing required key(s) {missing}")
    annotations = {f.name: f.type for f in dataclasses.fields(mipt.MonitoredCircuit)}
    kwargs = {
        k: v if k in _SPECIAL_SPEC_KEYS else _check_type(v, annotations[k], k)
        for k, v in spec.items()
        if k != "graph"
    }
    # bounded before any graph is generated: 10**12 qubits would exhaust the memory
    n = _require_int(spec["num_system"], "num_system", minimum=2, maximum=MAX_QUBITS)
    kwargs["num_system"] = n
    kwargs["layers"] = _require_int(spec["layers"], "layers", minimum=0)
    kwargs["measurement_rate"] = _require_number(
        spec["measurement_rate"], "measurement_rate", 0.0, 1.0
    )
    if "reference_partner" in spec:
        kwargs["reference_partner"] = _reference_partner(spec["reference_partner"])
    if "checkpoints" in spec:
        kwargs["checkpoints"] = _checkpoints(spec["checkpoints"])
    weights = spec.get("edge_weights")
    if "graph" in spec:
        graph = spec["graph"]
        if isinstance(graph, dict) and graph.get("family") == "file":
            edges, file_weights = _file_graph(graph, n)
            if file_weights is not None:
                if weights is not None:
                    raise ValueError("the graph file already holds weights; drop edge_weights")
                kwargs["edge_weights"] = file_weights
        else:
            edges = _build_graph(graph, n)
        if isinstance(weights, list) and (isinstance(edges, list) or _random_family(spec["graph"])):
            raise ValueError(
                "edge_weights as a list needs a known edge order, which a random graph family "
                "(or graph.realizations) does not have: give explicit 'edges' with one weight "
                'each, or a weight family such as {"family": "power_law", "alpha": 2.0}'
            )
    else:
        raw_edges = _edge_list(spec["edges"])
        edges = graphs.from_edges(raw_edges, n)
        if isinstance(weights, list):
            weights = _weights_in_normalized_order(raw_edges, weights)
    kwargs["edges"] = edges
    if weights is not None:
        kwargs["edge_weights"] = (
            [_build_weights(weights, g, n) for g in edges]
            if isinstance(edges, list)
            else _build_weights(weights, edges, n)
        )
    if "gate_set" in spec:
        kwargs["gate_set"] = _gate_set(spec["gate_set"])
    if spec.get("system_order") is not None:
        kwargs["system_order"] = _qubit_list(spec["system_order"], "system_order", n)
    if "probes" in spec:
        if not isinstance(spec["probes"], list):
            raise ValueError('probes must be a list like [{"kind": "code_dimension"}]')
        kwargs["probes"] = [_build_probe(p, i, n) for i, p in enumerate(spec["probes"])]
    try:
        return mipt.MonitoredCircuit(**kwargs)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid spec: {exc}") from None


def _random_family(graph: Any) -> bool:
    family: Any = graph.get("family") if isinstance(graph, dict) else None
    fn = _graph_families().get(family) if isinstance(family, str) else None
    return fn is not None and "seed" in _params(fn)


def _weights_in_normalized_order(raw_edges: np.ndarray, weights: list) -> list:
    """Reorder per-edge weights given for ``raw_edges`` to match the normalized edge list.

    The engine stores edges as ``graphs.from_edges`` returns them (each pair
    oriented ``a < b``, the pairs sorted), so a weight list in the caller's
    order must move with its edges or it would weight other edges.
    """
    if len(weights) != len(raw_edges):
        raise ValueError(
            f"edge_weights has {len(weights)} entries for {len(raw_edges)} edges; give one "
            "weight per edge, in the order of 'edges'"
        )
    oriented = np.sort(raw_edges, axis=1)
    order = np.lexsort((oriented[:, 1], oriented[:, 0]))
    ranked = oriented[order]
    if len(ranked) > 1 and np.any(np.all(ranked[1:] == ranked[:-1], axis=1)):
        raise ValueError(
            "edges lists a pair more than once; with a list of edge_weights every edge must "
            "appear exactly once"
        )
    return [weights[i] for i in order]


# Spec keys converted by dedicated code in _build_spec rather than by annotation.
_SPECIAL_SPEC_KEYS = {
    "num_system",
    "edges",
    "layers",
    "measurement_rate",
    "edge_weights",
    "reference_partner",
    "gate_set",
    "checkpoints",
    "probes",
    "system_order",
}


def _reference_partner(value: Any) -> int | str:
    if value == "random":
        return "random"
    if isinstance(value, str):
        raise ValueError(
            f'reference_partner must be "random" or a system qubit index, got {value!r}'
        )
    return _require_int(value, "reference_partner", minimum=0)


def _checkpoints(value: Any, what: str = "checkpoints") -> str | int | list[int]:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [_require_int(c, f"{what} entries", minimum=0) for c in value]
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return _require_int(value, f"{what} (a stride)", minimum=1)
    raise ValueError(
        f'{what} must be "final", "every_layer", a positive integer stride or a list of '
        f"layers, got {value!r}"
    )


def _edge_list(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("edges must be a list of [a, b] integer pairs (qubit indices)") from None
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("edges must be a list of [a, b] integer pairs")
    return arr


# The probe terms of the cost model: one-thread microseconds per evaluation, deliberately
# pessimistic. Fitted to benchmarks/results/probe_cost.md (AMD Ryzen 9 9950X3D, one thread,
# 2026-10-15, n = 64..2048, in the states that make each probe expensive) at about 3 times the
# measured cost or more at every measured point, which leaves room for a machine about twice as
# slow per thread. How much slower each thread runs while many run at once is priced separately
# (_parallel_threads). W is the tableau's words per column, ceil(num_qubits / 64);
# Wc = ceil(2 n / 64), the words of a code column.


def _elimination_us(m: int, words: int) -> float:
    """A Gaussian elimination over the 2m stabilizer columns of m qubits: O(m**2 W)."""
    return 2.0 + 1e-3 * m * m * (32.0 + 1.1 * words)


def _entropy_us(sizes: Sequence[int], num_qubits: int, words: int) -> float:
    """Entropies of regions of the given sizes: each marks the region among all the qubits
    and eliminates over the smaller side."""
    return sum(
        2.0 + 0.01 * num_qubits + _elimination_us(min(size, num_qubits - size), words)
        for size in sizes
    )


def _code_columns_us(n: int, words: int) -> float:
    """``code_columns``, which every evaluation of an erasure probe starts with: a bit test
    per tableau row (64 W of them) and code column (2n), and the row reduction behind it."""
    code_words = (2 * n + 63) // 64
    return 1e-3 * (64 * words) * (2 * n) * (24.0 + 0.15 * code_words)


def _probe_us(p: mipt.Probe, mc: mipt.MonitoredCircuit, single: bool | None = None) -> float:
    """Microseconds (one thread) to evaluate one probe at one checkpoint, pessimistically.

    ``single`` prices the erasure scans as in the single-reference picture (True) or the
    purification picture (False); by default, as in the picture of ``mc``.

    The erasure scans that stop at the code's first failure (``contiguous_distance``,
    ``erasure_first_failure``) cost what the code makes them: they are priced for the worst
    case of the single-reference picture (k = 1 and erasures that fail only at about n / 2
    qubits, as in the volume-law phase) and for the codes of monitored dynamics measured in
    the purification picture (benchmarks/results/probe_cost.md and erasure_cost.md). A
    purification code of rate near 0 whose erasures fail only near n / 2 would cost up to
    the single-reference price.
    """
    n, num_qubits = mc.num_system, mc.num_qubits
    words = (num_qubits + 63) // 64
    code_words = (2 * n + 63) // 64
    sizes = [len(region) for region in p.regions]
    if single is None:
        single = mc.reference == "single"
    if p.kind == "code_dimension":
        return 1.0 + 0.06 * n
    if p.kind in ("entropy", "half_cut"):
        return _entropy_us(sizes, num_qubits, words)
    if p.kind == "lost_information":  # two eliminations over the erased qubits themselves
        return sum(2 * _elimination_us(size, words) for size in sizes)
    if p.kind == "mutual_information":
        return _entropy_us([*sizes, sum(sizes)], num_qubits, words)
    if p.kind == "tripartite":
        a, b, c = sizes
        return _entropy_us([a, b, c, a + b, a + c, b + c, a + b + c], num_qubits, words)
    if p.kind == "contiguous_distance":
        if _capped_by_runs(p, mc):  # n single-qubit checks, then runs of 2 .. cap - 1 qubits
            m = p.size - 1 if p.size > 2 else 0  # from each of the n starts
            return 12.0 + 1.5e-3 * n * words + n * m * (0.15 + 0.05 * m + 5e-4 * words * m)
        # without a cap, or with one the kernel settles by the full distance
        if single:  # n starts, each grown to about n / 2 qubits: O(n**3 W)
            return 20.0 + 1e-3 * n**3 * (7.8 + 0.6 * words)
        return 50.0 + 0.03 * n**2.25
    if p.kind in ("logical_sites", "site_information"):
        # two columns of W words per qubit, no elimination: the n single-qubit checks of the
        # cap-2 contiguous distance, priced as that probe (plus n stored numbers per vector)
        stored = 0.01 * n if p.kind == "site_information" else 0.0
        return 12.0 + 1.5e-3 * n * words + stored
    if p.kind == "entropy_profile":  # one incremental pass over the n qubits: O(n**2 W)
        return 5.0 + 1e-3 * n * n * (18.0 + 1.5 * words)
    if p.kind == "erasure_lost_profile":  # two such passes per sample
        return 5.0 + p.samples * (2.0 + 1e-3 * n * n * (30.0 + 1.8 * words))
    if p.kind == "erasure_first_failure":
        # with a single reference every ordering is scanned to about n / 2: O(n**2 Wc)
        ordering = 1e-3 * n * n * (7.8 + 0.3 * code_words) if single else 0.03 * n**1.5
        return 5.0 + _code_columns_us(n, words) + p.samples * ordering
    if p.kind == "uncorrectable_counts":
        # C(n, w - 1) prefixes, each walking the 4**(w - 1) elements of its span with a hash
        # lookup of Wc words; in logs, and capped, so that an absurd max_size gives a huge
        # estimate rather than an overflow, even times many evaluations and trajectories
        w = p.size
        log_prefixes = math.lgamma(n + 1) - math.lgamma(w) - math.lgamma(n - w + 2)
        log_us = log_prefixes + (w - 1) * math.log(4) + math.log(1e-3 * (56.0 + 14.0 * code_words))
        return 5.0 + _code_columns_us(n, words) + math.exp(min(log_us, 400.0))
    if p.kind == "code_snapshot":  # a bit test per generator (at most n) and system qubit
        return 2.0 + 0.022 * n * n + 1e-3 * num_qubits
    if p.kind == "tableau_snapshot":
        # all N rows over all N qubits, a transpose of the tableau: O(N^2 / 32 + set bits).
        # Measured on one thread of an AMD Ryzen 9 9950X3D (2026-10-17, scrambled states,
        # tests/test_estimates.py's method): 0.0005 us per N^2 in the purification picture
        # (its reference columns stay sparse) and 0.001 in the single and no-reference
        # pictures (dense: twice the set bits) for N up to 3072; above N = 2048 the scattered
        # writes of a power-of-two N alias in the cache: 0.0027 at N = 4096, 0.0047 at 8192
        # (0.0013 at 6144). Priced at 4 times the dense cost up to N = 2048, growing as
        # (N / 2048)^1.5 above it: 3 times or more at every measured point
        growth = max(1.0, num_qubits / 2048) ** 1.5
        return 2.0 + 0.004 * num_qubits * num_qubits * growth
    if p.kind == "correlated_erasure":
        return _correlated_us(p, mc, words)
    # any newer probe: priced as a pass over every qubit
    return 5.0 + 1e-3 * n * n * (18.0 + 1.5 * words)


def _region_failure_us(n: int, columns: int, single: bool) -> float:
    """One region's failure (``region_failures``): ``I(A:R)`` over ``columns = min(m, n - m)``
    qubits, priced at 3 times a fit to measurements.

    Measured on one thread of an AMD Ryzen 9 9950X3D (2026-10-16, n = 64..512, m = n/8, n/4,
    n/2): about ``m'^2 (12 + 0.024 n)`` ns in the
    single-reference picture (a scrambled state, k = 1: the costliest), and
    ``m'^2 (9.5 + 0.0049 n)`` ns in the purification picture at its costliest (the scrambled
    state before any measurement, k = n; codes of monitored dynamics on a Watts-Strogatz graph
    or the brickwork ring cost 0.6 to 0.8 of that), ``m' = columns``.
    """
    slope = 12.0 + 0.024 * n if single else 9.5 + 0.0049 * n
    return 3.0 + 3e-3 * columns * columns * slope


def _correlated_us(p: mipt.Probe, mc: mipt.MonitoredCircuit, words: int) -> float:
    """One evaluation of a correlated_erasure probe (with a code; k = 0 draws nothing).

    Per size m: the regions, ``rungs x (burn_in + (samples - 1) thin) x n`` swap
    proposals of a chain (measured 19 ns at mean degree 4 and 30 ns at degree 8 on one
    thread of an AMD Ryzen 9 9950X3D, n = 64..1024, 2026-10-16; priced at about 3 times
    that, 30 + 7.5 x the mean degree ns), averaged over the erasure graphs, or
    ``c x samples`` exact samples (measured 0.3 + 0.0137 n us each, n = 64..4096, ring and
    path alike; priced 1 + 0.042 n us); then ``c x samples`` failures
    (:func:`_region_failure_us`, for the costliest state of the spec's picture) and the
    kernel's check of k, one evaluation over all ``n`` system qubits per size (priced at twice
    a region of ``n`` columns: the fit's range ends at ``n / 2``). Measured on
    the ring with exact samples (n = 64..512, purification and single pictures, m = n/8 to
    n/2): the whole evaluation came out at about 3 to 5 times below its estimate in the
    costliest states and 4 to 7 times below on monitored codes.
    """
    n = mc.num_system
    single = mc.reference == "single"
    plan = mc._plans[p.name]
    c, samples = len(p.couplings), p.samples
    sweeps = plan.burn_in + (samples - 1) * p.thin
    weights = np.bincount(plan.graph_class, minlength=len(plan.classes)) / plan.graph_class.size
    total = 5.0
    for i, m in enumerate(plan.sizes.tolist()):
        for cls, key in enumerate(plan.classes):
            if not weights[cls]:
                continue
            if key[0] == "exact":
                regions = c * samples * (1.0 + 0.042 * n)
            else:
                degree = 2.0 * key[1] / n
                rungs = plan.ladders[int(plan.ladder_id[cls, i])].size
                regions = rungs * sweeps * n * 1e-3 * (30.0 + 7.5 * degree)
            total += weights[cls] * regions
        failures = c * samples * _region_failure_us(n, min(m, n - m), single)
        total += failures + 2 * _region_failure_us(n, n, single)  # the check of k
    return total


def _trajectory_us(mc: mipt.MonitoredCircuit) -> float:
    """Pessimistic microseconds (one thread) for one trajectory of ``mc``.

    The dynamics and setup terms were fitted to measured runs (4-core x86-64,
    2026, n = 32..512, both reference pictures) and rounded up; real runs
    were 1x to 3x faster. On an AMD Ryzen 9 9950X3D (one thread, n = 64 to
    2048) they are 3.3 to 33 times the measured cost
    (benchmarks/results/probe_cost.md). The probe terms are
    :func:`_probe_us`. Trajectories that stop early (``stop_when_purified``,
    ``stop_when_distance``) make real runs faster still; the estimate prices
    every trajectory to ``layers``.
    """
    n, words = mc.num_system, (mc.num_qubits + 63) // 64
    gate_us = 0.2 + 0.04 * words
    if mc.schedule == "random_edges":
        gates_per_layer = mc.gates_per_layer or 0
    elif mc.schedule == "all_edges":
        gates_per_layer = max(len(g) for g in mc.graphs)
    else:
        gates_per_layer = max(1, n // 2)
    layer = 1.0 + gates_per_layer * gate_us + mc.measurement_rate * n * 0.15 * words
    if mc.record_first_touch:  # two tests per gate and one per measured qubit
        layer += 0.01 * (gates_per_layer + mc.measurement_rate * n)
    scramble_gates = (
        mc.scramble_gates + mc.scramble_matchings * (n // 2) + mc.scramble_layers * gates_per_layer
    )
    dynamics = mc.layers * layer + scramble_gates * gate_us
    probes = sum(len(mc.probe_layers(p.name)) * _probe_us(p, mc) for p in mc.probes)
    detection = len(mc.checkpoint_layers) * (1.0 + 0.06 * n) if mc.reference != "none" else 0.0
    setup = 5.0 + 0.01 * mc.num_qubits * words
    return setup + dynamics + probes + detection


_CALL_OVERHEAD_SECONDS = 0.05


def _setup_us(spec: dict, mc: mipt.MonitoredCircuit) -> float:
    """Pessimistic one-thread microseconds to build ``mc`` from ``spec`` and preprocess its graphs.

    Graph generation, validation and (for the matching schedule) edge
    coloring run in Python, once per graph, so a large graph bank
    (``realizations``) costs seconds before the first trajectory starts.
    Fitted to measured runs (n = 64..16384, every random family) and rounded
    up: at least 2 times the cost measured on an AMD Ryzen 9 9950X3D.
    """
    edges = sum(len(g) for g in mc.graphs)
    us = 50.0 * mc.num_graphs + 0.5 * edges  # normalizing and validating every graph
    if "graph" in spec:  # generating the graphs
        us += mc.num_graphs * _generation_us(spec["graph"], mc.num_system) + 2.5 * edges
    if mc.schedule == "matching":
        us += 2.5 * edges  # edge coloring
    return us


def _generation_us(graph: Any, n: int) -> float:
    """One-thread microseconds to generate one graph of ``graph`` beyond its edges.

    ``erdos_renyi`` draws a random number for every pair, and ``random_regular``
    redraws its pairing until the graph is simple, about exp((d**2 - 1) / 4)
    times for degree d, at most ``max_tries`` (default 1000) times.
    """
    family = graph.get("family") if isinstance(graph, dict) else None
    us = 300.0
    if family == "erdos_renyi":
        us += 0.6 * (n * (n - 1) // 2)
    elif family == "random_regular":
        d = graph.get("d")
        cap = graph.get("max_tries", 1000)
        if not isinstance(cap, (int, float)) or isinstance(cap, bool) or not cap >= 1:
            cap = 1000  # not a count: the family reports it
        if isinstance(d, (int, float)) and not isinstance(d, bool) and 0 < d < n:
            exponent = (float(d) ** 2 - 1.0) / 4.0
            # math.log takes ints of any size; an infinite float cap stays infinite
            log_cap = math.inf if isinstance(cap, float) and cap == math.inf else math.log(cap)
            # in logs, and capped, so that absurd degrees and caps stay finite
            us += 2.0 * math.exp(min(exponent, log_cap, 400.0)) * n * d
    return us


def _parallel_threads(threads: int) -> float:
    """What ``threads`` busy threads are worth in one-thread speed, pessimistically.

    A trajectory runs slower while others run beside it: all-core clocks are
    lower than one core's, SMT siblings share a core, and the threads share the
    memory bandwidth. On an AMD Ryzen 9 9950X3D (16 cores, 32 logical
    processors) each of 32 busy threads ran probe-heavy trajectories at 0.35 to
    0.52 of one thread's speed, and dynamics-heavy ones at 0.54 to 0.66
    (benchmarks/results/probe_cost.md, thread_scaling.md); this model gives
    0.45 (0.89 at 4 threads), and the margin of the one-thread cost model
    covers the rest.
    """
    return threads / (1.0 + 0.04 * (threads - 1))


def _running_threads(num_threads: int | None) -> int | None:
    """The threads a run asked for ``num_threads`` runs on: at most every thread this process
    may use (``NUMBA_NUM_THREADS``, as ``sparsegf2._parallel.threads`` clamps it); ``None``
    stays ``None`` (Numba's current count)."""
    import numba

    return None if num_threads is None else min(num_threads, numba.config.NUMBA_NUM_THREADS)


def _estimate_seconds(
    spec: dict, mc: mipt.MonitoredCircuit, trajectories: int, threads: int | None = None
) -> float:
    """The estimated seconds of a run on ``threads`` threads (default: Numba's current count)."""
    import numba

    # Trajectories are the unit of parallelism: fewer trajectories than threads
    # leave threads idle.
    parallel = min(numba.get_num_threads() if threads is None else threads, trajectories)
    kernel = trajectories * _trajectory_us(mc) * 1e-6 / _parallel_threads(parallel)
    return _CALL_OVERHEAD_SECONDS + _setup_us(spec, mc) * 1e-6 + kernel


def _result_bytes(mc: mipt.MonitoredCircuit, trajectories: int) -> int:
    """Memory of the run's result: every probe value (float64) plus the saved codes or tableaux
    (uint64 words: ``n x 2 ceil(n / 64)`` per code, ``N x 2 ceil(N / 64)`` per joint tableau)
    and, with a reference, the ``n + 1`` first passages (int64), and with
    ``record_first_touch`` the ``n`` first touches (int64)."""
    n = mc.num_system
    values = sum(len(mc.probe_layers(p.name)) * mipt._probe_length(p, n) for p in mc.probes)
    snapshots = 0
    for p in mc.probes:
        if p.kind in mipt._SNAPSHOT_KINDS:
            _, evaluations, rows, words = mipt._snapshot_shape(mc, p, 1)
            snapshots += evaluations * rows * words
    first = n + 1 if mc.reference != "none" else 0
    touches = n if mc.record_first_touch else 0
    return trajectories * 8 * (values + snapshots + first + touches + 4)


def _trajectory_count(mc: mipt.MonitoredCircuit, trajectories: Any) -> int:
    """The number of trajectories to run: as given, or every circuit of a graph collection."""
    if trajectories is None:
        if mc.num_trajectories is None:
            raise ValueError(
                "give trajectories (the number of random circuits; --trajectories in the CLI); "
                "it may be omitted only for a collection of graphs (graph.realizations, or a "
                "graph file holding several graphs), which runs circuits_per_graph circuits on "
                "each graph, or to resume a job in its output_dir, which runs to the target "
                "stored there"
            )
        trajectories = mc.num_trajectories
    trajectories = _require_int(trajectories, "trajectories", minimum=1)
    if trajectories > MAX_TRAJECTORIES:
        raise ValueError(f"at most {MAX_TRAJECTORIES:,} trajectories per call")
    if mc.num_trajectories is not None and trajectories > mc.num_trajectories:
        raise ValueError(
            f"{trajectories} trajectories requested, but {mc.num_graphs} graphs x "
            f"circuits_per_graph={mc.circuits_per_graph} give {mc.num_trajectories}: raise "
            "circuits_per_graph (or graph.realizations), or omit trajectories to run them all"
        )
    gigabytes = _result_bytes(mc, trajectories) / 1e9
    if gigabytes > MAX_RESULT_GB:
        raise ValueError(
            f"the result would take about {gigabytes:.1f} GB of memory (limit {MAX_RESULT_GB:g} "
            "GB): give the large probes (erasure_first_failure samples, code_snapshot, "
            "tableau_snapshot, vector probes) fewer checkpoints of their own, or split the "
            "trajectories into several output directories"
        )
    return trajectories


def _require_thread_counts(threads: Any) -> list[int]:
    """``threads`` as a non-empty list of positive integers, or an error that names it."""
    if not isinstance(threads, (list, tuple)) or not threads:
        raise ValueError(
            f"threads must be a non-empty list of positive integers such as [1, 32], got {threads!r}"
        )
    return [_require_int(t, "threads", minimum=1) for t in threads]


def _array_row(name: str, shape: Sequence[int], dtype: Any) -> dict[str, Any]:
    """One array of the saved file: its shape, dtype and uncompressed megabytes."""
    size = int(np.prod(shape, dtype=object)) * np.dtype(dtype).itemsize
    return {"name": name, "shape": [int(s) for s in shape], "dtype": str(np.dtype(dtype)),
            "megabytes": _num(size / 1e6)}  # fmt: skip


def _result_layout(mc: mipt.MonitoredCircuit, trajectories: int) -> dict[str, Any]:
    """The arrays and files that ``run_ensemble(..., output_dir=...)`` saves for ``mc``.

    Mirrors :meth:`sparsegf2.mipt.EnsembleResult.save`; sizes are uncompressed.
    """
    n = mc.num_system
    edges = sum(len(g) for g in mc.graphs)
    gate_set, order = np.asarray(mc.gate_set), np.asarray(mc.system_order)  # resolved by the spec
    rows = [
        _array_row("edges", (edges, 2), np.concatenate(mc.graphs).dtype),
        _array_row("graph_ptr", (len(mc.graphs) + 1,), np.int64),
        _array_row("system_order", order.shape, order.dtype),
        _array_row("trajectories", (trajectories,), np.int64),
        _array_row("final_layer", (trajectories,), np.int64),
        _array_row("num_measurements", (trajectories,), np.int64),
    ]
    # save stores neither the default gate set nor the layers of a symbolic checkpoint rule
    if not mipt._every_clifford(gate_set):
        rows.append(_array_row("gate_set", gate_set.shape, gate_set.dtype))
    if not isinstance(mc.checkpoints, (str, int)):
        rows.append(_array_row("checkpoint_layers", (len(mc.checkpoint_layers),), np.int64))
    if mc.edge_weights is not None:
        rows.append(_array_row("edge_weights", (edges,), np.float64))
    snapshots = tableaux = False
    for p in mc.probes:
        evaluations = len(mc.probe_layers(p.name))
        shape = [trajectories, evaluations]
        if p.is_vector:
            shape.append(mipt._probe_length(p, n))
        rows.append(_array_row(f"value::{p.name}", shape, np.float64))
        if p.kind in mipt._SNAPSHOT_KINDS:
            snapshots = snapshots or p.kind == "code_snapshot"
            tableaux = tableaux or p.kind == "tableau_snapshot"
            rows.append(
                _array_row(
                    f"snapshot::{p.name}", mipt._snapshot_shape(mc, p, trajectories), np.uint64
                )
            )
            rows.append(
                _array_row(f"snapshot_counts::{p.name}", (trajectories, evaluations), np.int64)
            )
    if mc.reference != "none":
        rows.append(_array_row("purification_layer", (trajectories,), np.int64))
        rows.append(_array_row("first_passage", (trajectories, n + 1), np.int64))
    if mc.record_first_touch:
        rows.append(_array_row("first_touch", (trajectories, n), np.int64))
    note = (
        "Sizes are uncompressed. The file is saved with np.savez_compressed, so the disk use "
        "is lower (benchmark_ensemble measures it), and a small __meta__ record holds the "
        "spec; spec.json and provenance.json are small text files."
    )
    if snapshots:
        note += (
            " A snapshot:: array is stored at its full shape, an upper bound: each code "
            "fills only its first generator rows (snapshot_counts:: says how many), the rest "
            "is zero."
        )
    if tableaux:
        note += (
            " A tableau_snapshot's snapshot:: array holds every generator of the joint state "
            "over all system and reference qubits at each evaluation (snapshot_counts:: is "
            "their number, or -1 where a stopped trajectory saved nothing)."
        )
    return {
        "arrays": rows,
        "files": [_RESULT_FILE, _SPEC_FILE, _PROVENANCE_FILE],
        "note": note,
    }


def estimate_ensemble(
    spec: dict[str, Any], trajectories: int | None = None, threads: list[int] | None = None
) -> dict[str, Any]:
    """A rough (deliberately pessimistic) runtime estimate for :func:`run_ensemble`.

    Returns ``estimated_seconds`` (building the spec, generating and
    preprocessing its graphs, and the trajectories), the thread count it
    assumes, whether a synchronous run fits within
    :data:`SYNC_ENSEMBLE_MAX_SECONDS`, the trajectory count (every circuit of a
    graph collection when ``trajectories`` is omitted) and the memory the
    result needs. Trajectories that stop early (``stop_when_purified``,
    ``stop_when_distance``) make real runs faster. ``notes`` (when present) says
    where the estimate can be too low.

    ``threads`` (a list of positive integers) adds ``by_threads``, the
    estimate for each count in the order given (threads beyond the number of
    trajectories stay idle). ``layout`` lists the arrays and files that
    ``run_ensemble(..., output_dir=...)`` saves, with their shapes, dtypes and
    uncompressed megabytes (disk use is lower; see :func:`benchmark_ensemble`).
    """
    import numba

    counts = None if threads is None else _require_thread_counts(threads)
    mc = _build_spec(spec)
    trajectories = _trajectory_count(mc, trajectories)
    seconds = _estimate_seconds(spec, mc, trajectories)
    out: dict[str, Any] = {
        "estimated_seconds": _num(seconds),
        "trajectories": trajectories,
        "result_megabytes": _num(_result_bytes(mc, trajectories) / 1e6),
        "threads": numba.get_num_threads(),
        "fits_sync_limit": seconds <= SYNC_ENSEMBLE_MAX_SECONDS,
        "sync_limit_seconds": SYNC_ENSEMBLE_MAX_SECONDS,
    }
    if counts is not None:
        out["by_threads"] = [
            {
                "threads": t,
                "estimated_seconds": _num(_estimate_seconds(spec, mc, trajectories, threads=t)),
            }
            for t in counts
        ]
    out["layout"] = _result_layout(mc, trajectories)
    notes = _estimate_notes(mc)
    per_thread = mc.correlated_bytes_per_thread()
    if per_thread:
        running = min(numba.get_num_threads(), trajectories)
        out["working_megabytes"] = _num(running * per_thread / 1e6)
        if counts is not None:
            for row, t in zip(out["by_threads"], counts, strict=True):
                row["working_megabytes"] = _num(min(t, trajectories) * per_thread / 1e6)
        notes.append(_working_memory_note(per_thread, running))
    if notes:
        out["notes"] = notes
    return out


def _working_memory_note(per_thread: int, running: int) -> str:
    """The note on the regions the correlated_erasure probes hold in every running thread."""
    total = running * per_thread
    limit = mipt.MAX_WORKING_BYTES
    note = (
        f"correlated_erasure holds up to {per_thread / 1e6:.3g} MB of regions (8 x couplings x "
        f"samples x (largest size + 1) bytes) in each running thread: {total / 1e6:.3g} MB on "
        f"{running} thread{'s' if running != 1 else ''} (working_megabytes); a run needing "
        f"more than {limit / 1e9:.3g} GB in all is refused"
    )
    if total > limit:
        fits = max(limit // per_thread, 0)
        advice = f"run on at most {fits} threads (num_threads)" if fits else "record fewer samples"
        note += f": this one would be; {advice}, or fewer samples, smaller sizes or couplings"
    return note


MAX_BENCHMARK_SECONDS = 600.0
"""Largest ``seconds`` that :func:`benchmark_ensemble` accepts."""


def _projected_rate(r1: float, rp: float | None, p: int, busy: int) -> float:
    """Trajectories per second with ``busy`` busy threads, from measured rates.

    ``r1`` is the measured rate on one thread; ``rp`` the measured rate on ``p``
    threads (``None`` when only one thread was measured). The simulator's thread
    model (:func:`_parallel_threads`) gives the shape, and a constant calibrates
    it so that it reproduces the measured ``p``-thread rate; one busy thread is
    the measured one-thread rate itself.
    """
    f = _parallel_threads
    if busy == 1:
        return r1  # measured
    scale = 1.0 if rp is None or p == 1 else rp / (r1 * f(p) / f(1))
    return r1 * f(busy) / f(1) * scale


def _benchmark_minimum_seconds(mc: mipt.MonitoredCircuit, trajectories: int, p: int) -> float:
    """The estimated seconds of the least :func:`benchmark_ensemble` can do: a one-thread
    warm-up and measurement of one trajectory each, and, for ``p`` > 1 threads, a warm-up and
    a measurement of ``p`` trajectories (one trajectory-time on ``p`` threads). Pessimistic,
    like :func:`_estimate_seconds`, and without the setup."""
    one = _trajectory_us(mc) * 1e-6
    on_p = one * p / _parallel_threads(p)
    return 2 * one + (2 * on_p if p > 1 else 0.0)


def _measure(
    mc: mipt.MonitoredCircuit, threads: int, first: int, limit: int, seconds: float, seed: int
) -> tuple[mipt.EnsembleResult, float]:
    """Run the run's first ``k`` trajectories on ``threads`` threads, doubling ``k`` from
    ``first`` until a run takes ``seconds / 4`` or ``k`` reaches ``limit``; the last run and its
    wall-clock seconds."""
    k = first
    while True:
        start = time.perf_counter()
        with _COMPUTE_LOCK:
            result = mc.run(k, seed=seed, num_threads=threads)
        elapsed = time.perf_counter() - start
        if elapsed >= seconds / 4 or k >= limit:
            return result, elapsed
        k = min(limit, 2 * k)


def _file_bytes(result: mipt.EnsembleResult, directory: Path) -> int:
    """The size of ``result`` saved as ``directory/result.npz``, the file a run keeps."""
    path = directory / _RESULT_FILE
    result.save(path)
    return path.stat().st_size


def benchmark_ensemble(
    spec: dict[str, Any],
    trajectories: int | None = None,
    threads: list[int] | None = None,
    seconds: float = 30.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure the speed and storage of a run on this machine; write nothing.

    After a warm-up (one trajectory, which compiles the kernels), the first
    trajectories of the run (seed ``seed``) are timed on one thread and on
    ``P = min(max(threads), available, trajectories)`` threads, where
    ``available`` is the most threads Numba can use here. Each measurement doubles
    its trajectory count until a run takes ``seconds / 4``, so the call takes up to
    about ``2 * seconds`` plus the warm-up. ``threads`` defaults to
    ``[1, available]``.

    ``measured`` lists the timings. ``projected`` gives, for each count in
    ``threads``, the seconds the whole run would take (the trajectories only, not
    building the spec and its graphs), calibrated to the measured rates; a count
    beyond the ``P`` measured is projected with the simulator's thread model
    (``basis``: ``"projected"``), and a count beyond the threads of this process
    gets a note. ``estimate_ratio`` is the measured seconds per trajectory at
    ``P`` threads over :func:`estimate_ensemble`'s, to scale cheap estimates.
    ``storage`` gives the megabytes of ``result.npz`` (compressed, measured on
    the sample and scaled to all trajectories) and of the result in memory. The
    sample is saved in a temporary folder that is removed.
    """
    import tempfile

    import numba

    mc = _build_spec(spec)
    trajectories = _trajectory_count(mc, trajectories)
    seconds = _require_number(seconds, "seconds", 0.0, MAX_BENCHMARK_SECONDS)
    if seconds <= 0:
        raise ValueError(f"seconds must be positive (at most {MAX_BENCHMARK_SECONDS:g}), got 0")
    seed = _require_int(seed, "seed", minimum=0, maximum=None)
    max_threads = int(numba.config.NUMBA_NUM_THREADS)
    counts = sorted({1, max_threads}) if threads is None else _require_thread_counts(threads)
    p = min(max(counts), max_threads, trajectories)
    # The least work is the warm-up and the first measurement, whatever `seconds` is.
    minimum = _benchmark_minimum_seconds(mc, trajectories, p)
    if minimum > 2 * seconds:
        raise ValueError(
            f"even the least measurement is estimated at about {minimum:.0f} s, above 2 x "
            f"seconds ({2 * seconds:g} s): measure a smaller run (fewer layers or a smaller "
            "num_system) and scale the figures, or use 'sparsegf2 benchmark' in a shell with "
            "a larger --seconds"
        )
    notes: list[str] = []
    if any(job.status == "running" for job in _JOBS.values()):
        notes.append("a background job was running: the timings are lower than an idle machine's")

    start = time.perf_counter()
    with _COMPUTE_LOCK:
        first = mc.run(1, seed=seed, num_threads=1)
        if p > 1:  # the first parallel region starts the thread pool
            mc.run(p, seed=seed, num_threads=p)
    warmup = time.perf_counter() - start

    one, one_seconds = _measure(mc, 1, 1, trajectories, seconds, seed)
    measured = [(1, one.num_trajectories, one_seconds)]
    sample, sample_seconds = one, one_seconds
    if p > 1:
        sample, sample_seconds = _measure(mc, p, p, trajectories, seconds, seed)
        measured.append((p, sample.num_trajectories, sample_seconds))
    r1 = one.num_trajectories / one_seconds
    rp = None if p == 1 else sample.num_trajectories / sample_seconds

    projected = []
    for t in counts:
        busy = min(t, trajectories)
        rate = _projected_rate(r1, rp, p, busy)
        projected.append(
            {
                "threads": t,
                "seconds": _num(trajectories / rate),
                "basis": "measured" if busy in (1, p) else "projected",
            }
        )
    if max(counts) > max_threads:
        notes.append(
            f"counts above {max_threads} are projected beyond the {max_threads} threads this "
            "process has, with the simulator's thread model"
        )
    notes.append("seconds are for the trajectories; building the spec and its graphs is extra")

    rate_p = r1 if rp is None else rp
    modeled = _trajectory_us(mc) * 1e-6 / _parallel_threads(p)  # seconds per trajectory
    with tempfile.TemporaryDirectory(prefix="sparsegf2-benchmark-") as folder:
        one_bytes = _file_bytes(first, Path(folder))
        big_bytes = _file_bytes(sample, Path(folder))
    k = sample.num_trajectories
    if k > 1 and big_bytes > one_bytes:
        # the file has a part that does not grow with the trajectories (the graphs, the
        # gate set): the growth per trajectory is what two sample sizes show
        disk = one_bytes + (big_bytes - one_bytes) / (k - 1) * (trajectories - 1)
    elif k > 1:
        disk = float(one_bytes)  # no growth showed: the one-trajectory file is the floor
        notes.append(
            "disk_megabytes is rough: the sample file was not larger than the one-trajectory "
            "file, so the one-trajectory size is used as a floor"
        )
    else:
        disk = max(one_bytes, big_bytes) * trajectories / k
        if trajectories > 1:
            notes.append(
                "disk_megabytes is an upper bound: one trajectory took longer than seconds / 4, "
                "so its fixed part (graphs, spec) was scaled with the rest"
            )
    if 1 < k < 8:
        notes.append(f"disk_megabytes is rough: it comes from a sample of only {k} trajectories")
    notes.append(
        "a background job writes chunk files before merging them, so its peak disk use is "
        "about twice the final result"
    )
    return {
        "trajectories": trajectories,
        "seed": seed,
        "threads_available": max_threads,
        "cpus": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
        "estimated_seconds": _num(_estimate_seconds(spec, mc, trajectories)),
        "warmup_seconds": _num(warmup),
        "measured": [
            {
                "threads": t,
                "trajectories": count,
                "seconds": _num(elapsed),
                "trajectories_per_second": _num(count / elapsed),
            }
            for t, count, elapsed in measured
        ],
        "projected": projected,
        "estimate_ratio": _num(1.0 / rate_p / modeled),
        "storage": {
            "sample_trajectories": k,
            "disk_megabytes": _num(disk / 1e6),
            "memory_megabytes": _num(_result_bytes(mc, trajectories) / 1e6),
        },
        "notes": notes,
    }


_SCAN_KINDS = ("contiguous_distance", "erasure_first_failure")  # stop at the first failure


def _capped_by_runs(p: mipt.Probe, mc: mipt.MonitoredCircuit) -> bool:
    """A contiguous_distance probe whose cap the kernel settles by short runs alone
    (``_kernel.entropy.capped_by_runs``), priced for the codes where no shorter run fails."""
    return p.kind == "contiguous_distance" and p.size > 0 and capped_by_runs(mc.num_system, p.size)


def _estimate_notes(mc: mipt.MonitoredCircuit) -> list[str]:
    """Where the estimate of ``mc`` can be too low, one sentence each.

    The scans that stop at a code's first failure are priced for the worst case only with a
    single reference (:func:`_probe_us`); in the purification picture, a code whose first
    failures are near n / 2 makes them cost up to their single-reference price.
    """
    scans = [p for p in mc.probes if p.kind in _SCAN_KINDS and not _capped_by_runs(p, mc)]
    if not scans or mc.reference == "single":
        return []
    priced = worst = 0.0
    for p in scans:
        evaluations = len(mc.probe_layers(p.name))
        priced += evaluations * _probe_us(p, mc)
        worst += evaluations * _probe_us(p, mc, single=True)
    factor = worst / priced
    shown = f"{factor:.0f}" if factor >= 10 else f"{factor:.1f}"
    kinds = sorted({p.kind for p in scans})
    return [
        f"{' and '.join(kinds)} {'is' if len(kinds) == 1 else 'are'} priced for codes whose "
        "erasures fail well before n / 2 qubits; a code "
        f"whose first failures are near n / 2 (low rate, large distance) can make "
        f"{'it' if len(kinds) == 1 else 'them'} take up to about {shown} times "
        f"{'its' if len(kinds) == 1 else 'their'} estimate"
    ]


# ---------------------------------------------------------------------------
# ensembles: summaries
# ---------------------------------------------------------------------------


_RANGE_LIST_RE = re.compile(r"\[(\d+(?:, \d+)+)\]")


def _compress_ints(values: list[int]) -> str:
    """``[0, 1, 2, 3, 9, 11, 13]`` -> ``"0:4, 9:15:2"``: arithmetic runs of 3+ become slices.

    The slices use the same ``start:stop[:step]`` convention as qubit-list inputs.
    """
    parts: list[str] = []
    i = 0
    while i < len(values):
        j = i
        step = values[i + 1] - values[i] if i + 1 < len(values) else 0
        if step > 0:
            while j + 1 < len(values) and values[j + 1] - values[j] == step:
                j += 1
        if j - i >= 2:
            stop = values[j] + 1 if step == 1 else values[j] + step
            parts.append(f"{values[i]}:{stop}" + (f":{step}" if step != 1 else ""))
            i = j + 1
        else:
            parts.append(str(values[i]))
            i += 1
    return ", ".join(parts)


def _summary_names(names: Sequence[str]) -> dict[str, str]:
    """Display keys for probe names: unchanged if short, else regions written as ranges.

    Default probe names spell out their regions (``entropy[0, 1, ..., 1023]``);
    those become ``entropy[0:1024]``, and anything still longer than
    :data:`MAX_NAME_CHARS` is cut and suffixed with a hash. Keys stay distinct.
    """
    out: dict[str, str] = {}
    used: set[str] = set()
    for name in names:
        key = name
        if len(key) > MAX_NAME_CHARS:
            key = _RANGE_LIST_RE.sub(
                lambda m: "[" + _compress_ints([int(x) for x in m.group(1).split(", ")]) + "]",
                key,
            )
            key = _short_key(key)
        base, copy = key, 2
        while key in used:
            key, copy = f"{base}#{copy}", copy + 1
        used.add(key)
        out[name] = key
    return out


def _vector_summary(
    stats: Callable[[int], tuple[np.ndarray, np.ndarray, int | None]],
    alive: np.ndarray,
    layers: np.ndarray,
    sizes_shown: int,
    depths_shown: int,
) -> tuple[dict[str, Any], bool]:
    """A vector probe: one curve over erasure or region size m = 0, 1, ... per evaluation.

    ``stats(c)`` returns the mean and standard error over trajectories of the
    curve at evaluation ``c``, and for a curve conditioned on a code the
    number of trajectories that hold one there (else None); ``alive``
    (trajectories x evaluations) marks the values that exist (with
    ``stop_when_purified``, entanglement profiles of stopped trajectories do
    not). ``mean``/``stderr`` are at the last evaluation that has data (for a
    curve conditioned on a code, the last one at which some trajectory holds
    one, if any does), and ``vs_depth`` gives the mean on a coarse grid of
    evaluations x sizes, so the change with circuit depth stays visible.
    ``index`` lists the sizes m shown when not all are. Returns the entry and
    whether anything was left out (sizes of the curve, or evaluations or
    sizes of the grid).
    """
    count = alive.sum(axis=0)
    with_data = np.flatnonzero(count > 0)
    if not with_data.size:
        empty = {"vector": True, "count": 0, "notes": ["no trajectory reached any checkpoint"]}
        return empty, False
    last = int(with_data[-1])
    if stats(last)[2] == 0:  # no code left there: show the last evaluation that has one
        last = next((int(c) for c in with_data[::-1] if stats(int(c))[2]), last)
    mean, err, with_code = stats(last)
    sizes = _subsample(mean.size, sizes_shown)
    entry: dict[str, Any] = {
        "vector": True,
        "at_layer": int(layers[last]),
        "count": int(count[last]),
    }
    if with_code is not None:
        entry["count_with_code"] = with_code
    entry["mean"] = mean[sizes]
    entry["stderr"] = err[sizes]
    cut = len(sizes) < mean.size
    if cut:
        entry["index"] = sizes
    if len(with_data) > 1:
        depth_idx = with_data[_subsample(len(with_data), depths_shown)]
        grid_sizes = _subsample(mean.size, MAX_GRID_SIZES)
        rows = [stats(int(c)) for c in depth_idx]
        grid: dict[str, Any] = {"layers": layers[depth_idx], "count": count[depth_idx]}
        if with_code is not None:
            grid["count_with_code"] = [row[2] for row in rows]
        grid["index"] = grid_sizes
        grid["mean"] = np.array([row[0][grid_sizes] for row in rows])
        entry["vs_depth"] = grid
        cut = cut or len(grid_sizes) < mean.size or len(depth_idx) < len(with_data)
    return entry, cut


def _vector_stats(
    probe: mipt.Probe,
    values: np.ndarray,
    n: int,
    cache: dict,
    has_code: np.ndarray | None = None,
) -> Callable[[int], tuple[np.ndarray, np.ndarray, int | None]]:
    """Mean and standard error of a vector probe's curve at one evaluation, memoized in ``cache``.

    Profiles are averaged as stored. The erasure probes are averaged over the
    trajectories that hold a code (k > 0) at the evaluation, whose number is
    returned as well (None for the curves that are not conditioned):
    first-failure samples show it themselves (some sample <= n; with k = 0
    every sample is n + 1), exhaustive counts need ``has_code`` (trajectories
    x the probe's evaluations, from :func:`_code_holders_at`), and without it
    average every trajectory. First-failure samples become per-trajectory
    P_C(m), m = 0..n (fraction of samples with D <= m), in chunks of
    trajectories so the memory stays bounded whatever the ensemble size;
    exhaustive counts become the exact P_C(m), m = 0..w (P_C(0) = 0).
    """

    def holders(c: int) -> np.ndarray | None:
        """The trajectories that hold a code at evaluation ``c``, or None if not conditioned."""
        if probe.kind == "erasure_first_failure":
            return np.flatnonzero(values[:, c, :].min(axis=1) <= n)
        if probe.kind == "uncorrectable_counts" and has_code is not None:
            return np.flatnonzero(has_code[:, c])
        return None

    def compute(c: int, rows: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        if probe.kind == "uncorrectable_counts":
            # rows None: every trajectory that reached the evaluation (NaN after a
            # stop_when_distance stop)
            counts = (
                values[~np.isnan(values[:, c, 0]), c, :] if rows is None else values[rows, c, :]
            )
            exact = analysis.exact_failure_by_size(counts, n)
            curves = np.concatenate([np.zeros((exact.shape[0], 1)), exact], axis=1)
            return analysis.mean_and_error(curves, axis=0)
        if rows is None:  # a profile
            return analysis.mean_and_error(values[:, c, :], axis=0)
        total = np.zeros(n + 1)
        squares = np.zeros(n + 1)
        chunk = max(1, (1 << 22) // (n + 2))
        for start in range(0, rows.size, chunk):
            curves = analysis.failure_by_size(values[rows[start : start + chunk], c, :], n)
            total += curves.sum(axis=0)
            squares += (curves**2).sum(axis=0)
        t = rows.size
        with np.errstate(invalid="ignore", divide="ignore"):  # no code (t = 0): NaN, shown as null
            mean = total / t
            var = np.maximum(squares - t * mean**2, 0.0) / (t - 1)
            err = np.sqrt(var / t)
        return mean, err

    def stats(c: int) -> tuple[np.ndarray, np.ndarray, int | None]:
        key = (probe.name, c)
        if key not in cache:
            rows = holders(c)
            cache[key] = (*compute(c, rows), None if rows is None else int(rows.size))
        return cache[key]

    return stats


def _code_holders_at(result: mipt.EnsembleResult, layers: np.ndarray) -> np.ndarray | None:
    """Whether each trajectory holds a code (k > 0) at each of ``layers``, or None.

    The answer (trajectories x layers) comes from the first ``code_dimension``
    probe evaluated at every one of these layers (on the same schedule or a
    denser one), or else from the first such ``erasure_first_failure`` probe
    (a code is held iff some sample is <= n). Probes are found by layer, not by
    name; None when neither kind covers all of ``layers``.
    """
    n = result.spec.num_system
    for kind in ("code_dimension", "erasure_first_failure"):
        for p in result.spec.probes:
            if p.kind != kind:
                continue
            own = np.asarray(result.layers(p.name))
            if not np.isin(layers, own).all():
                continue
            values = np.asarray(result.values[p.name])
            cols = np.searchsorted(own, layers)
            if kind == "code_dimension":
                return values[:, cols] > 0
            return np.stack([values[:, c, :].min(axis=1) <= n for c in cols], axis=1)
    return None


_HOLDS_A_CODE = (
    ", among trajectories that hold a code (k > 0); count_with_code says how many; at_layer "
    "is the last evaluation with count_with_code > 0, if there is one"
)
_LOSES_NOTHING = (
    "averaged over all trajectories; trajectories without a code (k = 0) count as 0 bits lost"
)
_OVER_ALL = "averaged over all trajectories"
_OVER_RUNNING = (
    "averaged over the trajectories still running (a stop_when_distance stop leaves a "
    "trajectory unknown after its stop; see 'count')"
)


def _loses_nothing_note(key: str) -> str:
    """The note on a lost-information probe, which is not conditioned on a code."""
    return (
        f"{key}: {_LOSES_NOTHING}, so the mean falls as codes are lost; select k > 0 in the "
        "saved arrays (output_dir) to average over the codes only"
    )


def _logical_sites_entry(
    result: mipt.EnsembleResult, values: np.ndarray, layers: np.ndarray, shown: np.ndarray
) -> dict[str, Any]:
    """What the summary adds to a ``logical_sites`` probe when a code_dimension (or
    erasure_first_failure) probe covers its layers: the mean over the trajectories that hold a
    code (k > 0), their number, and the fraction of all trajectories with k > 0 and no
    weight-one logical operator (contiguous distance at least 2), over those whose values are
    known (a ``stop_when_distance`` stop leaves NaN)."""
    holds = _code_holders_at(result, layers)
    if holds is None:
        return {}
    known = ~np.isnan(values)
    mean, err = analysis.mean_and_error(np.where(holds & known, values, np.nan), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):  # no known value: NaN, shown as null
        protected = (holds & known & (values == 0)).sum(axis=0) / known.sum(axis=0)
    return {
        "mean_with_code": mean[shown],
        "stderr_with_code": err[shown],
        "count_with_code": (holds & known).sum(axis=0)[shown],
        "distance_two_fraction": protected[shown],
    }


def _logical_sites_note(key: str, conditioned: bool) -> str:
    """The note on a ``logical_sites`` probe: what its means average over."""
    note = (
        f"{key}: N_log, the number of system qubits with I(i:R) > 0 (a weight-one logical "
        "operator); 'mean' counts trajectories without a code (k = 0) as 0"
    )
    if conditioned:
        return note + (
            "; mean_with_code averages the trajectories with k > 0 (count_with_code of them), "
            "and distance_two_fraction is the fraction of all trajectories with k > 0 and "
            "N_log = 0 (contiguous distance >= 2)"
        )
    return note + (
        "; add a code_dimension probe evaluated at the same layers for the mean over the codes "
        "(k > 0) and the fraction with k > 0 and N_log = 0"
    )


_VECTOR_MEANING = {
    "entropy_profile": "mean S(first m qubits of system_order) vs m (index)",
    "erasure_lost_profile": (
        "mean bits of logical information lost vs erasure size m (index), " + _LOSES_NOTHING
    ),
    "erasure_first_failure": (
        "P_C(m): fraction of random m-qubit erasures that are uncorrectable vs m (index), "
        "estimated from the first-failure samples" + _HOLDS_A_CODE
    ),
    "uncorrectable_counts": (
        "exact P_C(m): fraction of all m-qubit erasures that are uncorrectable vs m (index)"
    ),
    "site_information": (
        "mean I(i:R), the bits lost by erasing system qubit i alone (0, 1 or 2), vs qubit i "
        "(index), " + _LOSES_NOTHING + "; I(i:R) > 0 marks a weight-one logical operator on i"
    ),
}


_CORRELATED_GRID_SIZES = 9  # sizes per coupling in a correlated_erasure vs_depth grid
_CORRELATED_COUPLINGS = 12  # couplings shown in a correlated_erasure summary (ends kept)

_CORRELATED_MEANING = (
    "P_{C,G,J}(m): the fraction of m-qubit erasure regions A, drawn with probability "
    "proportional to exp(-J b_G(A)) (b_G(A): edges of the erasure graph G leaving A; J > 0 "
    "favors compact regions, J = 0 is uniform), that lose logical information; mean[i][j] "
    "is couplings[i] at sizes[j], and vs_depth.mean[i][d][j] is couplings[i] at "
    "vs_depth.layers[d] and vs_depth.sizes[j]"
)


def _sampler_text(sampler: dict[str, Any]) -> str:
    """One line on how a correlated_erasure probe draws its regions (``correlated_sampler``)."""
    exact = f"exact independent samples ({' and '.join(sampler.get('geometries', []))})"
    if sampler["method"] == "exact":
        return exact
    low, high = sampler["rungs"]
    rungs = f"{low} rungs" if low == high else f"{low}-{high} rungs"
    chain = (
        f"chain with replica exchange over {rungs}"
        if sampler["tempering"]
        else "one swap chain per coupling, without exchanges"
    )
    chain += f", burn_in {sampler['burn_in']} sweeps, thin {sampler['thin']}"
    if sampler["method"] == "chain":
        return chain
    return (
        f"{exact} on {sampler['exact_graphs']} of {sampler['graphs']} graphs, a {chain} on the rest"
    )


def _correlated_shown(shown: int, sizes_shown: int) -> tuple[int, int, int]:
    """How many couplings, sizes and vs_depth sizes a correlated_erasure summary shows.

    At the full checkpoint budget (``shown`` = :data:`MAX_CHECKPOINTS_SHOWN`) at most
    :data:`_CORRELATED_COUPLINGS` couplings by :data:`MAX_GRID_SIZES` sizes, and a
    vs_depth grid of :data:`_CORRELATED_GRID_SIZES` sizes; as :func:`_bounded_summary`
    shows fewer checkpoints to fit the size limit, fewer couplings (``shown / 2``, at least
    2) and sizes (``sizes_shown``, the other vector probes' budget) too.
    """
    couplings = _CORRELATED_COUPLINGS
    if shown < MAX_CHECKPOINTS_SHOWN:
        couplings = max(2, min(couplings, shown // 2))
    sizes = min(MAX_GRID_SIZES, sizes_shown)
    return couplings, sizes, min(_CORRELATED_GRID_SIZES, sizes)


def _correlated_summary(
    result: mipt.EnsembleResult,
    probe: mipt.Probe,
    own: np.ndarray,
    has_code: np.ndarray | None,
    depths_shown: int,
    shown: int = MAX_CHECKPOINTS_SHOWN,
    sizes_shown: int = MAX_GRID_SIZES,
) -> tuple[dict[str, Any], str | None]:
    """A correlated_erasure probe: P_{C,G,J}(m) per coupling over the sizes, and vs depth.

    The curves are averaged over the trajectories that hold a code (k > 0) when
    ``has_code`` (trajectories x the probe's evaluations, :func:`_code_holders_at`) is
    given, else over every trajectory (k = 0 counts as 0). ``mean``/``stderr`` (one row
    per coupling shown) are at the last evaluation, or the last one with a code.
    ``vs_depth`` gives the mean per coupling on a grid of at most ``depths_shown``
    evaluations by fewer sizes, indexed ``[coupling][depth][size]``. Couplings and sizes
    are subsampled evenly to the counts of :func:`_correlated_shown` (sizes in their order,
    both ends kept; couplings in increasing order of J when some are left out, the smallest
    and the largest J kept)
    (``shown`` and ``sizes_shown`` are the summary's budgets). Returns the entry and, when
    couplings, sizes or evaluations were left out, the sentence that says so (else None).
    """
    couplings, sizes, values = result.correlated(probe.name)
    trajectories, evaluations = values.shape[:2]
    what = _CORRELATED_MEANING + (
        _HOLDS_A_CODE
        if has_code is not None
        else ", averaged over all trajectories; trajectories without a code (k = 0) count as 0"
    )
    sampler = _sampler_text(result.spec.correlated_sampler(probe.name))
    if not trajectories or not evaluations:
        return {"what": what, "vector": True, "count": 0, "sampler": sampler}, None

    def rows(e: int) -> np.ndarray:
        return np.arange(trajectories) if has_code is None else np.flatnonzero(has_code[:, e])

    last = evaluations - 1
    if has_code is not None and not has_code[:, last].any():
        last = next((e for e in range(evaluations - 1, -1, -1) if has_code[:, e].any()), last)
    chosen = rows(last)
    mean, err = analysis.mean_and_error(values[chosen, last], axis=0)  # (couplings, sizes)
    most_couplings, most_sizes, grid_most = _correlated_shown(shown, sizes_shown)
    picked = _spread_by_value(couplings, most_couplings)
    columns = _subsample(sizes.size, most_sizes)
    # the rows known at that evaluation: a distance stop leaves every later value NaN
    known = ~np.isnan(values[:, last].reshape(trajectories, -1)).all(axis=1)
    entry: dict[str, Any] = {
        "what": what,
        "vector": True,
        "at_layer": int(own[last]),
        "count": int(known.sum()),
    }
    if has_code is not None:
        entry["count_with_code"] = int(chosen.size)
    entry.update(
        couplings=couplings[picked],
        sizes=sizes[columns],
        mean=mean[np.ix_(picked, columns)],
        stderr=err[np.ix_(picked, columns)],
        sampler=sampler,
    )
    left_out = []
    if picked.size < couplings.size:
        left_out.append(f"{picked.size} of the {couplings.size} couplings")
    if columns.size < sizes.size:
        left_out.append(f"{columns.size} of the {sizes.size} sizes")
    parts = []
    if left_out:
        parts.append(
            f"{' and '.join(left_out)} are shown (evenly spaced, both ends kept; the couplings "
            "in increasing order of J, the smallest and the largest included; listed in "
            "'couplings' and 'sizes')"
        )
    if evaluations > 1:
        depth_idx = _subsample(evaluations, depths_shown)
        grid_sizes = _subsample(sizes.size, grid_most)
        grid: dict[str, Any] = {"layers": own[depth_idx]}
        if has_code is not None:
            grid["count_with_code"] = [int(rows(int(e)).size) for e in depth_idx]
        grid["sizes"] = sizes[grid_sizes]
        means = [analysis.mean_and_error(values[rows(int(e)), e], axis=0)[0] for e in depth_idx]
        # indexed [coupling][depth][size], the couplings and sizes as listed
        grid["mean"] = np.stack([m[np.ix_(picked, grid_sizes)] for m in means], axis=1)
        entry["vs_depth"] = grid
        if grid_sizes.size < sizes.size or depth_idx.size < evaluations:
            depths = (
                f"{depth_idx.size} of the {evaluations} evaluations"
                if depth_idx.size < evaluations
                else "every evaluation"
            )
            parts.append(f"vs_depth shows {depths} at {grid_sizes.size} of the {sizes.size} sizes")
    if not parts:
        return entry, None
    return entry, (
        "; ".join(parts) + "; EnsembleResult.correlated() on the saved result (output_dir) "
        "has every value"
    )


def _bounded_summary(result: mipt.EnsembleResult, extra: dict[str, Any]) -> dict[str, Any]:
    """The summary of ``result`` plus ``extra``, within the size limit.

    When it is too long, fewer checkpoints are shown (evenly spaced, always
    the first and last, the same ones for every probe and the survival curve)
    before the generic cut of :func:`_bounded` is used as a last resort. A
    stored ``spec`` in ``extra`` is shown exactly (never rounded or cut), so
    it can be passed back to resume or extend the run.
    """
    total = max(
        [len(result.checkpoint_layers)] + [len(result.layers(name)) for name in result.values]
    )
    exact = ("spec",)
    long_extras = [
        key
        for key, value in extra.items()
        if key not in ("files", "path", "storage")  # storage is bounded by construction
        and len(json.dumps(_jsonable(value, exact=True))) > MAX_OUTPUT_CHARS // 4
    ]
    if long_extras:  # e.g. a stored spec with a long explicit edge list; it is on disk
        extra = {**extra, **{key: "too long to show; see the saved files" for key in long_extras}}
    out: dict[str, Any] = {}
    cache: dict = {}
    for shown in (MAX_CHECKPOINTS_SHOWN, 32, 16, 8, 4, 2):
        out = {**_summarize_ensemble(result, shown, cache), **extra}
        if long_extras:
            _mark_truncated(out, f"not shown: {', '.join(long_extras)} (too long)")
        if shown < min(MAX_CHECKPOINTS_SHOWN, total):
            _mark_truncated(
                out,
                f"only {shown} of up to {total} evaluations per probe are shown (evenly "
                "spaced, first and last included) to fit the size limit; pass output_dir or "
                "read the saved files for all of them",
            )
        if len(json.dumps(_clean(out, exact))) <= MAX_OUTPUT_CHARS:
            return _bounded(out, exact=exact)
    return _bounded(out, exact=exact)


def _summarize_ensemble(
    result: mipt.EnsembleResult, shown: int = MAX_CHECKPOINTS_SHOWN, cache: dict | None = None
) -> dict[str, Any]:
    cache = {} if cache is None else cache  # vector-probe statistics, reused across sizes
    spec = result.spec
    layers = np.asarray(result.checkpoint_layers)
    idx = _subsample(len(layers), shown)
    out: dict[str, Any] = {
        "num_trajectories": result.num_trajectories,
        "num_system": spec.num_system,
        "num_qubits": spec.num_qubits,
        "layers": spec.layers,
        "measurement_rate": spec.measurement_rate,
        "seed": result.seed_entropy,
        "randomness_contract": result.randomness_contract,
        "elapsed_seconds": result.elapsed_seconds,
        "trajectories_per_second": result.trajectories_per_second
        if result.elapsed_seconds > 0
        else None,
        "checkpoints": {
            "layers": layers[idx],
            "num_total": len(layers),
            "subsampled": len(idx) < len(layers),
        },
        "probes": {},
    }
    notes = _seed_notes(result.seed_entropy)
    keys = _summary_names(list(result.values))
    if any(k != name for name, k in keys.items()):
        notes.append(
            "long probe names are abbreviated (regions written as start:stop ranges); give "
            "probes a 'name' to choose the key"
        )
    num_vectors = sum(np.ndim(v) == 3 for v in result.values.values())
    sizes_shown = MAX_PROFILE_SHOWN if num_vectors <= 2 else 33
    if shown < MAX_CHECKPOINTS_SHOWN:  # a tighter budget: coarser profiles too
        sizes_shown = min(sizes_shown, max(shown, 4) + 1)
    probes = {p.name: p for p in spec.probes}
    n = spec.num_system
    depths_shown = min(MAX_VECTOR_DEPTHS, shown)
    vectors_cut = correlated_cut = False
    for name, values in result.values.items():
        values = np.asarray(values, dtype=np.float64)
        probe = probes[name]
        own = np.asarray(result.layers(name))
        if values.ndim == 2:
            pidx = _subsample(len(own), shown)
            mean, err = analysis.mean_and_error(values, axis=0)
            entry: dict[str, Any] = {"mean": mean[pidx], "stderr": err[pidx]}
            if not np.array_equal(own, layers):
                entry["layers"] = own[pidx]
            if np.isnan(values).any():
                entry["count"] = np.sum(~np.isnan(values), axis=0)[pidx]
                if probe.kind == "contiguous_distance":
                    notes.append(
                        f"{keys[name]}: the distance is undefined without a code, so it averages "
                        "only trajectories whose code dimension is above 0 (see 'count')"
                    )
            if probe.kind == "contiguous_distance" and 0 < probe.size < result.spec.num_system:
                notes.append(
                    f"{keys[name]}: min(d, {probe.size}), the contiguous distance capped at "
                    f"{probe.size}: a mean of capped values, not of distances"
                )
            if probe.kind == "code_snapshot":
                notes.append(
                    f"{keys[name]}: the mean number of stabilizer generators (n - k); the codes "
                    "themselves are saved with output_dir (EnsembleResult.code(t, c) in Python)"
                )
            if probe.kind == "tableau_snapshot":
                notes.append(
                    f"{keys[name]}: the number of generators of the joint state (system plus "
                    "reference qubits); the tableaux themselves are saved with output_dir "
                    "(EnsembleResult.tableau(t, c) in Python: a PhaseFreeTableau)"
                )
            if probe.kind == "lost_information":
                notes.append(_loses_nothing_note(keys[name]))
            if probe.kind == "logical_sites":
                entry.update(_logical_sites_entry(result, values, own, pidx))
                notes.append(_logical_sites_note(keys[name], "count_with_code" in entry))
        elif probe.kind == "correlated_erasure":
            has_code = _code_holders_at(result, own)
            summary, cut_note = _correlated_summary(
                result, probe, own, has_code, depths_shown, shown, sizes_shown
            )
            correlated_cut = correlated_cut or cut_note is not None
            if cut_note is not None:
                notes.append(f"{keys[name]}: {cut_note}")
            if has_code is None:
                notes.append(
                    f"{keys[name]}: trajectories without a code (k = 0) count as P = 0 in these "
                    "curves; add a code_dimension or erasure_first_failure probe evaluated at "
                    "the same layers to average over the codes (k > 0) only"
                )
            sampler = result.spec.correlated_sampler(name)
            if sampler["method"] != "exact":
                notes.append(
                    f"{keys[name]}: its regions come from a swap chain ({summary['sampler']}), "
                    "whose mixing depends on the graph, the sizes, the couplings and these "
                    "settings, never on the code: check it with sample_erasure_regions "
                    "(erasures.sample_regions in Python: the same kernel and ladder) before a "
                    "large run"
                )
            entry = summary
        else:
            alive = (
                ~np.isnan(values[:, :, 0])
                if mipt._KINDS[probe.kind].after_purification == "nan"
                or spec.stop_when_distance is not None  # every probe is NaN after such a stop
                else np.ones(values.shape[:2], dtype=bool)
            )
            what = _VECTOR_MEANING.get(probe.kind, "a curve over size m (index)")
            has_code = None
            if probe.kind == "uncorrectable_counts":
                # The counts alone cannot tell k = 0 from a distance above w.
                has_code = _code_holders_at(result, own)
                if has_code is None:
                    what += (
                        ", averaged over all trajectories; trajectories without a code (k = 0) "
                        "count as P_C = 0"
                    )
                    notes.append(
                        f"{keys[name]}: trajectories without a code (k = 0) count as P_C = 0 in "
                        "this curve, since the counts alone cannot tell k = 0 from a large "
                        "distance; add a code_dimension or erasure_first_failure probe evaluated "
                        "at the same layers to average over the codes (k > 0) only"
                    )
                else:
                    what += _HOLDS_A_CODE
            if probe.kind == "erasure_lost_profile":
                notes.append(_loses_nothing_note(keys[name]))
            stats = _vector_stats(probe, values, n, cache, has_code)
            summary, cut = _vector_summary(stats, alive, own, sizes_shown, depths_shown)
            vectors_cut = vectors_cut or cut
            entry = {"what": what, **summary}
        out["probes"][keys[name]] = entry
    if result.purification_layer is not None:
        out["purification"] = _purification_summary(
            result.purification_layer, layers, shown, result.final_layer
        )
    if result.first_touch is not None:
        out["first_touch"] = _first_touch_summary(result, layers, idx)
        notes.append(
            "first_touch: bare_mean is the mean N_bare(T), the number of system qubits that no "
            "gate or measurement had reached by layer T (scramble gates count as layer 0); "
            "fraction_with_bare is P(N_bare(T) > 0); tau_bare_median_all is the median of "
            "tau_bare (the first layer with N_bare = 0), exact to the layer whatever the "
            "checkpoints, when more than half of the trajectories ran out of bare qubits by the "
            "earliest layer at which a trajectory was censored (null otherwise)"
        )
    out["final_layer_mean"] = (
        float(np.mean(result.final_layer)) if result.num_trajectories else None
    )
    out["measurements_per_trajectory"] = (
        float(np.mean(result.num_measurements)) if result.num_trajectories else None
    )
    if spec.stop_when_purified:
        notes.append(
            "stop_when_purified: code probes take their exact values for a purified trajectory "
            "(k = 0, nothing lost); entanglement probes average only trajectories still running "
            "(see 'count')"
        )
    if spec.stop_when_distance is not None:
        stopped = result.stopped_on_distance
        stop_layers = np.asarray(result.final_layer)[stopped]
        out["stop_when_distance"] = {
            "at_least": spec.stop_when_distance,
            "not_before": spec.stop_not_before,
            "probe": keys[spec.stop_probe.name] if spec.stop_probe is not None else None,
            "stopped": int(stopped.sum()),
            "stop_layer_mean": float(stop_layers.mean()) if stop_layers.size else None,
            "stop_layer_max": int(stop_layers.max()) if stop_layers.size else None,
        }
        notes.append(
            f"stop_when_distance: {int(stopped.sum())} of {result.num_trajectories} "
            f"trajectories stopped at the first layer >= {spec.stop_not_before} by which their "
            f"contiguous distance had reached {spec.stop_when_distance}; every probe is NaN "
            "(unknown) after such a stop, so later means average only the trajectories still "
            "running (see 'count'), and the purification time is censored at the stop"
        )
        # The texts of the probes that are not conditioned on a code say "all trajectories";
        # under the rule a stopped trajectory is unknown after its stop and left out.
        for entry in out["probes"].values():
            if isinstance(entry, dict) and isinstance(entry.get("what"), str):
                entry["what"] = entry["what"].replace(_OVER_ALL, _OVER_RUNNING)
        notes[:] = [note.replace(_OVER_ALL, _OVER_RUNNING) for note in notes]
    if correlated_cut:
        out["truncated"] = True
    if vectors_cut:
        out["truncated"] = True
        notes.append(
            f"vector probes are subsampled: curves to at most {sizes_shown} sizes m (listed in "
            f"'index'), vs_depth grids to at most {depths_shown} evaluations by "
            f"{MAX_GRID_SIZES} sizes; pass output_dir or read the saved files for every value"
        )
    if notes:
        out["notes"] = notes
    return out


def _first_touch_summary(
    result: mipt.EnsembleResult, layers: np.ndarray, shown: np.ndarray
) -> dict[str, Any]:
    """The untouched ("bare") qubits of a run with ``record_first_touch``, at the checkpoints.

    ``bare_mean`` (and ``count``, the trajectories it averages) is the mean N_bare at each
    shown checkpoint over the trajectories whose N_bare is known there
    (:meth:`~sparsegf2.mipt.EnsembleResult.bare_counts`: NaN only past the stop of a
    trajectory that still had bare qubits). ``fraction_with_bare`` is P(N_bare > 0) = P(tau_bare
    > layer), NaN (null) where it is not known for every trajectory; ``all_touched`` counts the
    trajectories whose every qubit was touched by their final layer (tau_bare observed), and
    ``tau_bare_median_all`` is the median of tau_bare over all trajectories, exact to the layer
    whatever the checkpoints (first touches are recorded at every layer): the first layer T
    with P(tau_bare > T) <= 1/2, defined (as ``tau_median_all`` for purification) only when
    strictly more than half of the trajectories ran out of bare qubits by the earliest
    censoring depth, below which P(tau_bare > T) is known for every trajectory; else None.
    """
    at, counts = result.bare_counts(layers)
    known = ~np.isnan(counts)
    mean, err = analysis.mean_and_error(counts, axis=0)
    with np.errstate(invalid="ignore"):
        some = np.where(known.all(axis=0), (counts > 0).mean(axis=0), np.nan)
    time, observed = result.bare_times()
    # tau_bare is exact where observed and known to exceed final_layer (its time here) where
    # censored, so P(tau_bare > T) is known for every T up to the earliest censoring depth
    known_to = int(time[~observed].min()) if not observed.all() else int(result.spec.layers)
    done = np.sort(time[observed & (time <= known_to)])
    total = int(time.size)
    return {
        "layers": at[shown],
        "bare_mean": mean[shown],
        "stderr": err[shown],
        "count": known.sum(axis=0)[shown],
        "fraction_with_bare": some[shown],
        "all_touched": int(observed.sum()),
        "censored": int(observed.size - observed.sum()),
        "tau_bare_median_all": int(done[(total + 1) // 2 - 1]) if 2 * done.size > total else None,
    }


def _purification_summary(
    tau: np.ndarray,
    layers: np.ndarray,
    shown: int = MAX_CHECKPOINTS_SHOWN,
    final: np.ndarray | None = None,
) -> dict[str, Any]:
    """Purification statistics of the times ``tau`` (-1: censored) at the checkpoint ``layers``.

    ``final`` (each trajectory's last simulated layer) marks what is known: a layer
    beyond the stop of an unpurified trajectory (``stop_when_distance``) is not
    known for every trajectory, so the survival there is NaN (shown as null) and
    ``tau_median_all`` uses only the layers known for all.
    """
    tau = np.asarray(tau)
    total = tau.size
    purified = tau[tau >= 0]
    out: dict[str, Any] = {
        "trajectories": total,
        "purified": int(purified.size),
        "censored": int(total - purified.size),
        "censored_fraction": (total - purified.size) / total if total else None,
        "horizon_layer": int(layers[-1]) if layers.size else None,
    }
    if purified.size:
        q = np.percentile(purified, [10, 25, 50, 75, 90])
        out.update(
            tau_mean_purified=float(purified.mean()),
            tau_median_purified=float(q[2]),
            tau_min=int(purified.min()),
            tau_max=int(purified.max()),
            tau_quantiles_purified={"p10": q[0], "p25": q[1], "p75": q[3], "p90": q[4]},
        )
    if total and layers.size:
        idx = _subsample(len(layers), shown)
        alive = (tau[None, :] < 0) | (tau[None, :] > layers[:, None])
        surviving = alive.mean(axis=1)
        if final is not None:  # unknown where an unpurified trajectory had stopped earlier
            known = (tau[None, :] >= 0) | (np.asarray(final)[None, :] >= layers[:, None])
            surviving = np.where(known.all(axis=1), surviving, np.nan)
        # The median of tau over all trajectories is well defined despite
        # censoring only when strictly more than half purified within the
        # horizon (with exactly half it lies anywhere from the last purification
        # to beyond the horizon); it is then the first layer with S <= 1/2.
        # S is non-increasing, and NaN only after the last layer known for all.
        below = np.flatnonzero(surviving <= 0.5)
        out["tau_median_all"] = int(layers[below[0]]) if (surviving < 0.5).any() else None
        out["survival"] = {"layers": layers[idx], "fraction_unpurified": surviving[idx]}
    return out


# ---------------------------------------------------------------------------
# ensembles: files
# ---------------------------------------------------------------------------


def _output_dir(path: Any) -> Path:
    if not isinstance(path, (str, os.PathLike)) or not str(path):
        raise ValueError("output_dir must be a non-empty path string")
    out = Path(path).expanduser().resolve()
    if out.exists() and not out.is_dir():
        raise ValueError(f"output_dir {str(out)!r} exists and is not a directory")
    parent = next((p for p in out.parents if p.exists()), None)  # the nearest that exists
    if parent is not None and not parent.is_dir():
        raise ValueError(
            f"output_dir {str(out)!r} cannot be created: {str(parent)!r} is a file, not a "
            "directory; choose another output_dir"
        )
    with _writing_to(out):
        out.mkdir(parents=True, exist_ok=True)
    return out


def _not_writable(out: Path, exc: OSError) -> ValueError:
    """An OSError creating or writing to an output directory, as the input error it is."""
    return ValueError(
        f"output_dir {str(out)!r} is not writable: {exc}; free space or fix its permissions, "
        "or choose another output_dir"
    )


@contextlib.contextmanager
def _writing_to(out: Path) -> Iterator[None]:
    """Report an OSError raised in the block (no permission, a full drive, a path too
    long) as :func:`_not_writable`."""
    try:
        yield
    except OSError as exc:
        raise _not_writable(out, exc) from None


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    # numpy's stubs type the **arrays of savez_compressed as its allow_pickle flag
    np.savez_compressed(tmp, **arrays)  # type: ignore[arg-type, unused-ignore]
    os.replace(tmp, path)


_T = TypeVar("_T")


def _retrying(
    action: Callable[[], _T], errors: tuple[type[OSError], ...] = (PermissionError,)
) -> _T:
    """``action()``, retried with the :data:`_RETRY_DELAYS` backoff while it raises ``errors``."""
    for delay in _RETRY_DELAYS:
        try:
            return action()
        except errors:
            time.sleep(delay)
    return action()


def _temporary(path: Path) -> Path:
    """A temporary file beside ``path`` whose name is this writer's own.

    Two writers never write into one temporary file (their bytes would
    interleave). Loads never read temporaries (:func:`_result_parts`), and a
    merge deletes those a killed writer left (:func:`_remove_temporaries`).
    """
    return path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp.npz")


def _save_temporary(result: mipt.EnsembleResult, path: Path) -> Path:
    """Save ``result`` to a :func:`_temporary` file for ``path``; returns its path."""
    tmp = _temporary(path)
    try:
        result.save(tmp)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    return tmp


def _replace(tmp: Path, path: Path) -> None:
    """Move ``tmp`` onto ``path``, retried while a reader holds ``path``; ``tmp`` is
    deleted if that still fails."""
    try:
        _retrying(functools.partial(os.replace, tmp, path))
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _atomic_save_result(result: mipt.EnsembleResult, path: Path) -> None:
    """Save to a temporary file, then replace ``path`` (retried while a reader holds it)."""
    _replace(_save_temporary(result, path), path)


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _provenance(seed: Any, command: str) -> dict[str, Any]:
    import numba

    return {
        "sparsegf2": __version__,
        "randomness_contract": mipt.RANDOMNESS_CONTRACT,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "seed": seed,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": command,
        "argv": sys.argv,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    """Write a temporary file, then replace ``path``, retried while a reader holds it
    (Windows; a ``load_results`` reading ``spec.json``, say) as :func:`_replace` does."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _replace(tmp, path)


def _write_run_metadata(
    out: Path,
    spec: dict,
    trajectories: int,
    seed: int,
    command: str,
    *,
    resumed_from: int | None = None,
    graph_file: dict[str, str] | None = None,
) -> dict[str, str]:
    """Write ``spec.json`` (exact: replaying it repeats the run) and ``provenance.json``.

    Replaying ``spec.json`` repeats the run only under the randomness contract
    that ``provenance.json`` records: this version cannot reproduce a run made
    under another one (``mipt.RANDOMNESS_CONTRACT``). For a graph file,
    ``spec.json`` also records its resolved path and the sha256 of its bytes
    (``graph_file``), with which a resume checks that the file is unchanged.

    For a resumed job (``resumed_from`` set) the original provenance is kept and
    this process's record is appended under ``"resumes"``.
    """
    record_file = {} if graph_file is None else {"graph_file": graph_file}
    try:
        spec_text = json.dumps(
            {
                "spec": _jsonable(spec, exact=True),
                "trajectories": trajectories,
                "seed": seed,
                **record_file,
            },
            indent=2,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"spec must be JSON-serializable to be saved: {exc}") from None
    spec_path = out / _SPEC_FILE
    _atomic_write_text(spec_path, spec_text)
    prov_path = out / _PROVENANCE_FILE
    record = _provenance(seed, command)
    stored = _read_provenance(out) if resumed_from is not None else None
    if stored is not None:
        resume = {**record, "resumed_from": resumed_from}
        earlier = stored.get("resumes")  # not a list: ignored (_provenance_notes)
        record = {**stored, "resumes": [*(earlier if isinstance(earlier, list) else []), resume]}
    _atomic_write_text(prov_path, json.dumps(record, indent=2))
    return {"spec": str(spec_path), "provenance": str(prov_path)}


def _provenance_notes(stored: dict | None) -> list[str]:
    """Notes on fields of a stored ``provenance.json`` that a resume reads but that hold
    another type (a hand edit, say). They are ignored, as an unreadable provenance.json
    is (the saved results carry the version and contract that a resume checks), never an
    error: ``resumes`` restarts with this resume, and ``git_commit`` is not compared."""
    notes = []
    for key, kind, then in (
        ("resumes", list, "a list; it restarts with this resume"),
        ("git_commit", str, "a string; it is not compared with this resume's commit"),
    ):
        value = None if stored is None else stored.get(key)
        if value is not None and not isinstance(value, kind):
            notes.append(f"provenance.json: {key} is {_shown(value)}, not {then}")
    return notes


def _read_provenance(out: Path) -> dict | None:
    try:
        stored = json.loads((out / _PROVENANCE_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return stored if isinstance(stored, dict) else None


def _check_contract(out: Path, stored: dict | None, existing: mipt.EnsembleResult | None) -> None:
    """Refuse to resume or extend a run made under another randomness contract.

    The contract is read from ``provenance.json`` (the original record) and
    from the saved results. A record without one predates contract 2, so it
    counts as contract 1, as ``EnsembleResult.load`` does.
    """
    made = [] if stored is None else [stored.get("randomness_contract", 1)]
    if existing is not None:
        made.append(existing.randomness_contract)
    for contract in made:
        if contract != mipt.RANDOMNESS_CONTRACT:
            raise ValueError(
                f"output_dir {str(out)!r}: this run was made under randomness contract "
                f"{contract}; this version uses {mipt.RANDOMNESS_CONTRACT}; start a new "
                "output_dir"
            )


def _check_version(out: Path, stored: dict | None, existing: mipt.EnsembleResult | None) -> None:
    """Refuse to resume or extend a run started with another ``sparsegf2`` version.

    The version is read from ``provenance.json`` (the original record) and
    from the saved results, which a merge requires to match, so a missing or
    unreadable ``provenance.json`` cannot let another version's results through.
    """
    made = [] if stored is None or stored.get("sparsegf2") is None else [stored["sparsegf2"]]
    if existing is not None:
        made.append(existing.version)
    for made_with in made:
        if made_with != __version__:
            shown = f"sparsegf2 {made_with}" if made_with else "an unrecorded sparsegf2 version"
            raise ValueError(
                f"output_dir {str(out)!r} holds a run started with {shown}, "
                f"but this is {__version__}; trajectories from different versions need not "
                "agree, so finish the run with that version or start a new directory"
            )


def _check_saved_seed(out: Path, seed: Any, existing: mipt.EnsembleResult | None) -> None:
    """Refuse a ``spec.json`` seed that is not the one the saved results were made with.

    New chunks run with ``seed`` from ``spec.json``; a merge requires the seed
    record of every part to match, so a mismatch is refused before anything runs.
    """
    if existing is None:
        return
    ss = np.random.SeedSequence(_resolve_seed(seed))
    if mipt._seed_record(existing) != (ss.entropy, tuple(ss.spawn_key), ss.pool_size):
        raise ValueError(
            f"output_dir {str(out)!r}: its spec.json gives seed {seed}, but the saved results "
            f"were made with seed {existing.seed_entropy}; restore the seed in spec.json, or "
            "use a new directory"
        )


_CHUNK_NAME = re.compile(r"traj_\d+\.npz")


def _result_parts(out: Path) -> list[Path]:
    """``result.npz`` (if any) followed by the saved chunk files in trajectory order.

    Only these exact names are parts: a temporary file (``*.tmp.npz``), also one
    a killed writer left, is never read as one. A ``chunks/`` that is missing, or
    that a merge removes during the listing, holds no chunks: a merge replaces
    result.npz before it deletes them, which is why the chunks are listed first.
    (On Windows a directory being deleted cannot be listed for a moment either;
    that ``PermissionError`` is raised, and readers retry it.)
    """
    try:
        chunks = sorted(p for p in (out / _CHUNK_DIR).iterdir() if _CHUNK_NAME.fullmatch(p.name))
    except (FileNotFoundError, NotADirectoryError):
        chunks = []
    result = [out / _RESULT_FILE] if (out / _RESULT_FILE).exists() else []
    return [*result, *chunks]


def _load_result_file(path: Path) -> mipt.EnsembleResult:
    """Load one ``EnsembleResult.save`` file; anything else is reported as not a result.

    ``FileNotFoundError`` and ``PermissionError`` pass through: a merge deleting
    or replacing the file meanwhile raises them, and the caller retries.
    """
    try:
        return mipt.EnsembleResult.load(path)
    except (FileNotFoundError, PermissionError):
        raise
    except (
        OSError,
        ValueError,
        KeyError,
        EOFError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            f"{str(path)!r} is not a saved SparseGF2 ensemble result (expected a .npz written "
            f"by EnsembleResult.save, or an output directory): {type(exc).__name__}: {exc}"
        ) from None


def _load_parts_once(parts: list[Path]) -> mipt.EnsembleResult:
    """Merge result.npz and chunks, keeping each trajectory once.

    Trajectory ``t`` depends only on ``(spec, seed, t)``, so a part's
    trajectories that an earlier part already holds are redundant, not a
    conflict, and are dropped: a merge writes result.npz before deleting the
    chunks it absorbed (a crash, or a reader holding a chunk, in between
    leaves chunks that overlap result.npz), and jobs run with different chunk
    sizes save overlapping ranges. Only a part of the same run is trimmed; a
    part of another run is refused, even when every trajectory it holds is
    covered.
    """
    results: list[mipt.EnsembleResult] = []
    paths: list[Path] = []
    covered = 0
    for path in parts:
        part = _load_result_file(path)
        if results and _same_run_as(results[0], part, overlap=True):
            keep = np.asarray(part.trajectories) >= covered
            if not keep.any():
                continue
            if not keep.all():
                part = _rows(part, keep)
        results.append(part)
        paths.append(path)
        if part.num_trajectories:
            covered = max(covered, int(np.max(part.trajectories)) + 1)
    try:
        merged = results[0] if len(results) == 1 else mipt.EnsembleResult.concatenate(results)
    except ValueError as exc:
        raise ValueError(
            f"saved parts do not belong to the same run: {exc}{_strays(results, paths)}"
        ) from None
    expected = np.arange(merged.num_trajectories)
    if not np.array_equal(np.asarray(merged.trajectories), expected):
        raise ValueError(
            "saved trajectories are not contiguous from 0; the directory mixes different runs"
        )
    return merged


_STRAYS_SHOWN = 20


def _strays(results: list[mipt.EnsembleResult], paths: list[Path]) -> str:
    """The saved parts that cannot be merged with the earliest one, and how to recover.

    The earliest part (``result.npz``, else the first chunk) holds the earlier
    data; a part of another run (another protocol, seed, version or randomness
    contract) is named so that deleting it makes the directory readable again.
    """
    first = results[0]
    strays = [
        path
        for path, part in zip(paths[1:], results[1:], strict=True)
        if not _same_run_as(first, part)
    ]
    if not strays:
        return ""
    names = [p.name for p in strays[:_STRAYS_SHOWN]]
    if len(strays) > _STRAYS_SHOWN:
        names.append(f"and {len(strays) - _STRAYS_SHOWN} more")
    return (
        f". These files in {str(strays[0].parent)!r} do not belong to the run in "
        f"{paths[0].name}: {', '.join(names)}; delete them to recover the trajectories saved "
        "before (load_results then reads them, and start_ensemble_job resumes the run)"
    )


def _same_run_as(
    first: mipt.EnsembleResult, part: mipt.EnsembleResult, *, overlap: bool = False
) -> bool:
    """Whether ``part`` could be merged with ``first``: what ``EnsembleResult.concatenate``
    checks, without copying any data (contract, seed, protocol, probes, layers, version,
    and, unless ``overlap``, no trajectory saved twice)."""
    return (
        part.randomness_contract == first.randomness_contract
        and mipt._seed_record(part) == mipt._seed_record(first)
        and mipt._protocol_difference(first.spec, part.spec) is None
        and list(part.values) == list(first.values)
        and np.array_equal(part.checkpoint_layers, first.checkpoint_layers)
        and part.version == first.version
        and (overlap or not np.intersect1d(first.trajectories, part.trajectories).size)
    )


def _rows(result: mipt.EnsembleResult, keep: np.ndarray) -> mipt.EnsembleResult:
    """The trajectories of ``result`` that the boolean mask ``keep`` selects."""
    share = float(np.count_nonzero(keep)) / result.num_trajectories
    tau = result.purification_layer
    return dataclasses.replace(
        result,
        trajectories=np.asarray(result.trajectories)[keep],
        values={name: v[keep] for name, v in result.values.items()},
        purification_layer=None if tau is None else tau[keep],
        final_layer=result.final_layer[keep],
        num_measurements=result.num_measurements[keep],
        elapsed_seconds=result.elapsed_seconds * share,
        snapshots={name: (g[keep], c[keep]) for name, (g, c) in result.snapshots.items()},
        first_passage=None if result.first_passage is None else result.first_passage[keep],
        first_touch=None if result.first_touch is None else result.first_touch[keep],
    )


def _load_directory(
    out: Path, files_read: list[Path] | None = None
) -> tuple[mipt.EnsembleResult | None, dict | None]:
    """Read an output directory, retrying while a running job merges chunks meanwhile.

    A merge deletes the chunks it absorbed (``FileNotFoundError`` for a reader
    that listed them) and, on Windows, a file being replaced cannot be opened
    for a moment (``PermissionError``). A file that still cannot be read after
    the retries (another program holds it, or this process may not read it) is
    an input error that names it. ``files_read``, if given, is filled with the
    files that were listed.
    """

    def read() -> tuple[mipt.EnsembleResult | None, dict | None]:
        meta = _read_meta(out)
        parts = _result_parts(out)
        if files_read is not None:
            files_read[:] = parts
        return (_load_parts_once(parts) if parts else None), meta

    try:
        return _retrying(read, (FileNotFoundError, PermissionError))
    except OSError as exc:
        raise _unreadable(out, exc) from None


def _unreadable(where: Path, exc: OSError) -> ValueError:
    """An OSError reading the saved run in ``where`` (``exc`` names the file), as the input
    error it is."""
    return ValueError(
        f"could not read results from {str(where)!r}: {exc}; close any program that holds "
        "the file open, or check its permissions"
    )


def _read_meta(out: Path) -> dict | None:
    """``out/spec.json``, the run's record (None if there is none); a file that holds
    none is an input error that names it."""
    path = out / _SPEC_FILE
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        return None
    except ValueError as exc:  # not JSON, or not UTF-8
        raise ValueError(
            f"{str(path)!r} is not valid JSON ({exc}); restore it, or use a new output_dir"
        ) from None
    if not isinstance(meta, dict):
        raise ValueError(
            f"{str(path)!r} holds no run's record (an object with spec, trajectories and "
            "seed); restore it, or use a new output_dir"
        )
    return meta


def _merge_if_pending(out: Path, result: mipt.EnsembleResult, meta: dict | None) -> None:
    """Merge the chunks of a finished run that its job could not merge (a reader held
    result.npz), unless a writer holds the directory: a running job merges them itself."""
    target = None if meta is None else meta.get("trajectories")
    if not isinstance(target, int) or result.num_trajectories < target:
        return
    try:
        pending = any(p.parent.name == _CHUNK_DIR for p in _result_parts(out))
    except OSError:  # chunks/ is being deleted (Windows): a merge is under way
        return
    if not pending:
        return
    try:
        _refuse_a_running_job(out)
        lock = _lock_output_dir(out)
    except (OSError, ValueError):  # another writer, or a directory this process cannot write
        return
    try:
        with contextlib.suppress(OSError):  # still pending: the next call tries again
            _merge_chunks(out)
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# ensembles: one writer per output directory
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _DirLock:
    """``output_dir/.lock``, held by the one process that writes to the directory."""

    path: Path
    data: bytes  # this holder's record, as written

    def release(self) -> None:
        """Delete the lock, if it is still this one (never another writer's)."""

        def remove() -> None:
            if self.path.read_bytes() == self.data:
                self.path.unlink()

        with contextlib.suppress(OSError):  # already gone, or still in use after the retries
            _retrying(remove)


@functools.cache
def _os_identity() -> str:
    """The operating system whose process ids a lock's pid belongs to.

    The hostname alone does not say it: WSL distributions report the Windows
    computer name, and containers may share the host's name, but a pid of one
    means nothing in another. So a lock records this too: the system's name
    and, on Linux, its pid namespace (``/proc/self/ns/pid``), which tells WSL
    distributions and containers apart. Unlike the kernel release, it stays
    the same across reboots and upgrades of a machine whose processes run in
    the initial pid namespace (Windows, macOS, Linux on bare metal), so a lock
    that a crash leaves can still be judged after them. A container, and a
    WSL2 distribution, run in a pid namespace of their own, which may be a new
    one each time they start (WSL after ``wsl --shutdown``, a reboot or its
    idle shutdown): a lock that a crashed writer left there then counts as
    another machine's, so the next writer is refused, and it must be deleted
    by hand. That fails safe: two writers never share a directory.
    """
    identity = platform.system()
    with contextlib.suppress(OSError):  # no /proc (Windows, macOS)
        identity += f" pidns:{os.stat('/proc/self/ns/pid').st_ino}"
    return identity


def _process_state(pid: int) -> tuple[bool, int | None]:
    """Whether process ``pid`` runs on this machine, and its creation time where the
    system gives it exactly (Windows: the ``FILETIME``, in 100 ns ticks; else None)."""
    if pid <= 0:  # os.kill(0) and os.kill(-1) would address process groups
        return False, None
    if sys.platform == "win32":
        return _windows_process(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, None
    except PermissionError:  # it runs, under another user
        return True, None
    return True, None


def _windows_process(pid: int) -> tuple[bool, int | None]:
    """Whether process ``pid`` runs, and its creation time, from kernel32 (Windows only)."""
    if sys.platform != "win32":
        raise OSError("kernel32 exists on Windows only")
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetProcessTimes.argtypes = (wintypes.HANDLE, *[ctypes.POINTER(wintypes.FILETIME)] * 4)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        # ERROR_ACCESS_DENIED: it runs, under an account this one may not query
        return ctypes.get_last_error() == 5, None
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True, None  # unknown: never take over a lock that may be live
        times = [wintypes.FILETIME() for _ in range(4)]  # creation, exit, kernel, user
        created = None
        if kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            created = times[0].dwHighDateTime << 32 | times[0].dwLowDateTime
        return code.value == 259, created  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _holder_alive(holder: dict) -> bool:
    """Whether the process of this machine that took a lock still runs.

    Its pid must name a running process, and, where the lock records its
    holder's creation time, the very process: Windows soon gives the pid of
    a process that ended to another one, whose creation time differs. Both
    times are the kernel's, so a clock step cannot make them differ.
    """
    alive, created = _process_state(holder["pid"])
    recorded = holder.get("created")
    return alive and (recorded is None or created is None or created == recorded)


def _lock_record(data: bytes) -> dict | None:
    """The holder written in a lock file, or None if the file holds no complete record.

    A ``created`` that is not an int (a float, a string, a bool) counts as
    absent, never as another process's creation time: a malformed optional
    field must not make a live holder's lock look stale.
    """
    try:
        record = json.loads(data.decode("utf-8"))
    except ValueError:
        return None
    if not isinstance(record, dict) or not {"pid", "host", "started"} <= set(record):
        return None
    pid = record["pid"]
    if isinstance(pid, bool) or not isinstance(pid, int) or not 0 < pid < 2**31:
        return None
    created = record.get("created")
    if "created" in record and (isinstance(created, bool) or not isinstance(created, int)):
        record = {key: value for key, value in record.items() if key != "created"}
    return record if isinstance(record["host"], str) else None


def _this_machine(holder: dict) -> bool:
    """Whether a lock was taken on this machine: the same hostname and, where the lock
    records it (locks written before it was recorded do not), the same
    :func:`_os_identity`. Another machine's lock never expires."""
    return holder["host"] == socket.gethostname() and holder.get("os", _os_identity()) == (
        _os_identity()
    )


def _lock_holder(path: Path) -> tuple[bytes, dict | None] | None:
    """The bytes of the lock file and the holder they record (None if they record
    none), or None if there is no lock."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except PermissionError:  # Windows: being deleted
        data = b""
    return data, _lock_record(data)


def _age(path: Path) -> float:
    """Seconds since ``path`` was last written (0 if that cannot be read)."""
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return 0.0


# A lock without a record, or a takeover turn, this old was left by a process that died
# while writing it.
_TAKEOVER_ABANDONED_SECONDS = 10.0


def _remove_stale_lock(path: Path, seen: bytes) -> None:
    """Delete a stale lock, but only if it still holds ``seen``, the bytes judged stale.

    Processes that find a stale lock take turns, each creating
    ``.lock.takeover`` with ``O_CREAT | O_EXCL``, and delete the lock only if
    it still holds ``seen``. Between that check and the deletion the lock
    cannot change: its holder is gone, every other process that would delete
    it waits for its turn, and a new lock can be created only once it is
    deleted. So a lock that another process has just taken over is never
    deleted, and of the processes racing for a stale lock exactly one creates
    the new one. A turn left behind by a process that died in it is freed
    after :data:`_TAKEOVER_ABANDONED_SECONDS`.
    """
    guard = path.with_name(_TAKEOVER_FILE)
    try:
        os.close(os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        if _age(guard) > _TAKEOVER_ABANDONED_SECONDS:
            with contextlib.suppress(OSError):
                guard.unlink()
        return
    except PermissionError:  # Windows: the previous turn's file is being deleted
        return
    try:
        if path.read_bytes() == seen:
            path.unlink()
    except (FileNotFoundError, PermissionError):  # gone, or in use: try again later
        pass
    finally:
        with contextlib.suppress(OSError):
            guard.unlink()


def _running_job(out: Path) -> _Job | None:
    """A job of this process that is writing to ``out``, if any."""
    with _JOBS_LOCK:
        for job in _JOBS.values():
            if job.output_dir == out and job.status in ("queued", "running"):
                return job
    return None


def _refuse_a_running_job(out: Path) -> None:
    """Refuse, naming it, a job of this process that writes to ``out``: the check that
    holds in this process even if ``.lock`` was deleted by hand."""
    job = _running_job(out)
    if job is not None:
        raise ValueError(
            f"job {job.job_id} is already writing to {str(out)!r}; poll it or cancel it"
        )


def _held_message(out: Path, holder: dict) -> str:
    """The refusal for a writer while ``holder`` holds the lock of ``out``."""
    if holder["pid"] == os.getpid() and _this_machine(holder):
        job = _running_job(out)
        if job is not None:  # a job of this process: name it, so it can be polled or canceled
            return f"job {job.job_id} is already writing to {str(out)!r}; poll it or cancel it"
        who = "this process"
    else:
        who = "another process"
    # the lock's system, when it is another one: WSL and containers share the host name,
    # and their pids name nothing here
    system = holder.get("os")
    shown = f" ({system})" if system is not None and system != _os_identity() else ""
    return (
        f"{who} (pid {holder['pid']} on host {holder['host']}{shown}, started "
        f"{holder['started']}) is writing to {str(out)!r}; wait for it to finish or cancel it"
    )


def _create_lock(path: Path, data: bytes) -> bool:
    """Create ``path`` holding ``data``, unless it exists (then return False).

    The record is written to a file of this writer's own first and then
    hard-linked to ``path``: ``os.link`` fails if ``path`` exists, and the
    lock appears with its whole record, so no other process ever reads it
    half written. Where hard links are not supported (FAT drives, some
    network shares) it is created with ``O_CREAT | O_EXCL`` and written
    after; a lock found without a record is then judged by its age
    (:data:`_TAKEOVER_ABANDONED_SECONDS`).
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with open(tmp, "xb") as f:
            f.write(data)
        try:
            os.link(tmp, path)
            return True
        except FileExistsError:
            return False
        except OSError:  # no hard links here
            pass
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0))
        except FileExistsError:
            return False
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        except BaseException:
            with contextlib.suppress(OSError):
                path.unlink()
            raise
        return True
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def _lock_output_dir(out: Path) -> _DirLock:
    """Take ``out/.lock`` for this process, or refuse because another writer holds it.

    The lock is created exclusively (:func:`_create_lock`), so of several
    writers exactly one gets it. It records the holder's pid, host, start
    time, :func:`_os_identity` and, on Windows, its creation time. A lock of
    another machine always holds (its process cannot be checked from here).
    A lock of this machine whose process is gone (:func:`_holder_alive`), or
    one without a record that is older than
    :data:`_TAKEOVER_ABANDONED_SECONDS`, is stale and is taken over
    (:func:`_remove_stale_lock`). The holder releases it with
    :meth:`_DirLock.release`, in a ``finally``. A lock that cannot be created
    or read (a read-only directory or drive, a full one) is an input error.
    """
    path = out / _LOCK_FILE
    record: dict[str, Any] = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "os": _os_identity(),
    }
    created = _process_state(os.getpid())[1]
    if created is not None:
        record["created"] = created
    data = json.dumps(record).encode("utf-8")
    failure: OSError | None = None
    for delay in (*_RETRY_DELAYS, None):
        try:
            if _create_lock(path, data):
                return _DirLock(path, data)
            failure = None
            held = _lock_holder(path)
            if held is not None:
                seen, holder = held
                if holder is None:  # no record: a writer died writing it, or is writing it
                    if _age(path) > _TAKEOVER_ABANDONED_SECONDS:
                        _remove_stale_lock(path, seen)
                elif not _this_machine(holder) or _holder_alive(holder):
                    raise ValueError(_held_message(out, holder))
                else:
                    _remove_stale_lock(path, seen)
        except PermissionError as exc:  # Windows: the previous lock is being deleted
            failure = exc
        except OSError as exc:  # a read-only or full drive, say: retrying cannot help
            raise _not_writable(out, exc) from None
        if delay is None:
            break
        time.sleep(delay)
    if failure is not None:  # no permission to write here
        raise _not_writable(out, failure) from None
    raise ValueError(
        f"could not take the lock {str(path)!r}: other processes keep changing it, or it "
        "cannot be read (check its permissions); retry, or delete it if no process is "
        f"writing to {str(out)!r}"
    )


# ---------------------------------------------------------------------------
# ensembles: synchronous runs
# ---------------------------------------------------------------------------


def run_ensemble(
    spec: dict[str, Any],
    trajectories: int | None = None,
    seed: int | None = None,
    output_dir: str | None = None,
    num_threads: int | None = None,
    command: str | None = None,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """Run ``trajectories`` random monitored circuits and summarize them.

    For a collection of graphs (``graph.realizations``) ``trajectories`` may
    be omitted: every graph then gets ``circuits_per_graph`` circuits.

    ``spec`` holds :class:`sparsegf2.mipt.MonitoredCircuit` parameters as
    JSON (see ``capabilities()["ensemble_spec"]``), with the graph given as
    ``{"family": ..., params}`` or as explicit ``"edges"``, and probes as
    ``[{"kind": "code_dimension"}, ...]``. The result is exactly
    ``MonitoredCircuit(...).run(trajectories, seed=seed)``, summarized: probe
    means and standard errors at (at most 50) checkpoints, purification
    statistics with censoring, and throughput. ``elapsed_seconds`` and
    ``trajectories_per_second`` are wall-clock figures for the whole call,
    building the spec and its graphs included. With ``output_dir`` the full
    :class:`~sparsegf2.mipt.EnsembleResult` is saved to
    ``output_dir/result.npz`` with ``spec.json`` and ``provenance.json``,
    under ``output_dir/.lock`` (a directory another process is writing to is
    refused). With ``max_seconds``, a run whose :func:`estimate_ensemble` exceeds it is
    refused before anything runs (the MCP tool's guard).
    """
    start = time.perf_counter()
    mc = _build_spec(spec)
    graph_file = _graph_file_record(spec) if output_dir is not None else None
    trajectories = _trajectory_count(mc, trajectories)
    if num_threads is not None:
        num_threads = _require_int(num_threads, "num_threads", minimum=1)
    if max_seconds is not None:
        max_seconds = _require_number(max_seconds, "max_seconds", 0.0, math.inf)
        # priced on the threads it will run on: fewer threads take longer
        estimate = _estimate_seconds(spec, mc, trajectories, _running_threads(num_threads))
        if estimate > max_seconds:
            raise ValueError(
                f"this run is estimated at about {estimate:.0f} s, above "
                f"max_seconds={max_seconds:g}. Use start_ensemble_job with an output_dir and "
                "poll job_status, or reduce trajectories/layers/num_system (or raise "
                "max_seconds if you are willing to wait)."
            )
    seed = _resolve_seed(seed)
    out_path = _output_dir(output_dir) if output_dir is not None else None
    lock = None
    if out_path is not None:  # one writer per directory
        _refuse_a_running_job(out_path)
        lock = _lock_output_dir(out_path)
    try:
        if out_path is not None:
            try:
                held = bool(_result_parts(out_path)) or (out_path / _SPEC_FILE).exists()
            except OSError as exc:  # a chunks/ this process may not list
                raise _unreadable(out_path, exc) from None
            if held:
                raise ValueError(
                    f"output_dir {str(out_path)!r} already holds a run; choose a new directory "
                    "(or use start_ensemble_job to resume or extend it)"
                )
        setup = time.perf_counter() - start
        with _COMPUTE_LOCK:
            result = mc.run(trajectories, seed=seed, num_threads=num_threads)
        # the run's own timer covers graph preprocessing and the kernel; add the
        # spec construction (graph generation, validation) before it
        result.elapsed_seconds += setup
        extra: dict[str, Any] = {}
        if out_path is not None:
            with _writing_to(out_path):
                _atomic_save_result(result, out_path / _RESULT_FILE)
                files = {"result": str(out_path / _RESULT_FILE)}
                files.update(
                    _write_run_metadata(
                        out_path,
                        spec,
                        trajectories,
                        seed,
                        command or "run_ensemble",
                        graph_file=graph_file,
                    )
                )
            extra["files"] = files
    finally:
        if lock is not None:
            lock.release()
    return _bounded_summary(result, extra)


def load_results(path: str) -> dict[str, Any]:
    """Summarize a saved ensemble: a ``.npz`` written by ``EnsembleResult.save`` or an output directory.

    A sweep's directory (``sweep.json``, ``docs/SWEEPS.md``) gives its index: every
    condition's saved and target trajectories, the versions and the spec. One of its condition
    folders gives the summary of all its saved parts together (``sparsegf2.study.sweep.load``);
    one above the memory limit is refused with a message naming ``sweep.parts``.

    For a directory (from :func:`run_ensemble` or a job), every saved chunk is
    included, so this also reads partial results of a running or canceled
    job. The chunks of a finished run that its job could not merge (a reader
    held ``result.npz``) are merged now, unless another writer holds the
    directory. The summary has the same form as :func:`run_ensemble`'s, plus
    the stored ``spec`` when ``spec.json`` exists: exactly as stored (its
    floats are not rounded for display), so passing it back to
    :func:`start_ensemble_job` resumes or extends the run. ``storage`` gives
    the disk use of the saved ``.npz`` files: ``total_megabytes``, each file
    (``files``) and each array (``arrays``, compressed sizes; with several
    chunk files, summed by name).
    """
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError("path must be a string")
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ValueError(f"no such file or directory: {str(p)!r}")
    if p.is_dir() and (p / "sweep.json").is_file():
        return _sweep_index_summary(p)
    if p.is_dir() and (p.parent / "sweep.json").is_file():
        return _sweep_condition_summary(p)
    parts: list[Path] = [p]
    try:
        if p.is_dir():
            result, meta = _load_directory(p, parts)
            if result is None:
                raise ValueError(f"no saved results in {str(p)!r} (expected result.npz or chunks/)")
            _merge_if_pending(p, result, meta)
            # a merge replaced the chunks by result.npz
            parts = [q for q in parts if q.exists()]
            if (p / _RESULT_FILE).exists() and (p / _RESULT_FILE) not in parts:
                parts.insert(0, p / _RESULT_FILE)
        else:
            result, meta = _retrying(functools.partial(_load_result_file, p)), None
    except OSError as exc:  # still in use after the retries, or not ours to read
        raise _unreadable(p, exc) from None
    extra: dict[str, Any] = {"path": str(p)}
    if meta is not None:
        extra["spec"] = meta.get("spec")
        extra["trajectories_requested"] = meta.get("trajectories")
    # a file replaced meanwhile: the data was read, only the sizes are missing
    with contextlib.suppress(OSError, zipfile.BadZipFile):
        root = p if p.is_dir() else p.parent
        everything = [q for q in p.rglob("*") if q.is_file()] if p.is_dir() else [p]
        extra["storage"] = _storage_summary(parts, root, everything)
    return _bounded_summary(result, extra)


def _sweep_index_summary(root: Path) -> dict[str, Any]:
    """``load_results`` of a sweep's directory: its index and every condition's counts."""
    from sparsegf2.study import sweep

    index = sweep._read_index(root)
    assert index is not None, "the caller found sweep.json"
    status = sweep.status(root)
    out = {
        "path": str(root),
        "kind": "sweep",
        "digest": index["digest"],
        "complete": status["complete"],
        "trajectories": status["trajectories"],
        "saved": status["saved"],
        "versions": index["versions"],
        "conditions": status["conditions"],
        "spec": index["spec"],
        "notes": [
            "a sweep (docs/SWEEPS.md): load_results(<this path>/<conditions[i].path>) summarizes "
            "one condition; in Python sparsegf2.study.sweep.load(path, n, beta, p)"
        ],
    }
    return _bounded(out, exact=("spec",))


def _sweep_condition_summary(folder: Path) -> dict[str, Any]:
    """``load_results`` of one condition of a sweep: every saved part, as one result."""
    from sparsegf2.study import sweep

    root = folder.parent
    index = sweep._read_index(root)
    assert index is not None, "the caller found sweep.json"
    entry = next((c for c in index["conditions"] if c["path"] == folder.name), None)
    if entry is None:
        raise ValueError(f"{str(folder)!r} is not a condition of the sweep in {str(root)!r}")
    files = sweep._part_paths(folder)
    if not files:
        raise ValueError(f"nothing saved yet in {str(folder)!r}")
    size = 0
    for path in files:
        try:
            with zipfile.ZipFile(path) as z:
                size += sum(info.file_size for info in z.infolist())
        except (zipfile.BadZipFile, OSError) as exc:
            raise sweep.unreadable_part(path, exc) from None
    n, beta, p = entry["n"], entry["beta"], entry["p"]
    if size > MAX_RESULT_GB * 2**30:
        raise ValueError(
            f"condition {folder.name} holds {size / 2**30:.1f} GB of results, above the "
            f"{MAX_RESULT_GB:g} GB memory limit: read its parts one at a time in Python with "
            f"sparsegf2.study.sweep.parts({str(root)!r}, {n}, {beta!r}, {p!r})"
        )
    result = sweep.load(root, n, beta, p)
    everything = [q for q in folder.iterdir() if q.is_file()]
    extra = {
        "path": str(folder),
        "sweep": {"root": str(root), "n": n, "beta": beta, "p": p},
        "trajectories_requested": entry["trajectories"],
        "storage": _storage_summary(files, folder, everything),
    }
    return _bounded_summary(result, extra)


MAX_STORAGE_FILES = 8
MAX_STORAGE_ARRAYS = 20


def _storage_summary(parts: list[Path], root: Path, everything: list[Path]) -> dict[str, Any]:
    """The disk use of a saved run, in megabytes: every file, and the arrays of its results.

    ``everything`` is every regular file of the run (``result.npz``, chunks, ``spec.json``,
    ``provenance.json``, ...); ``total_megabytes`` counts them all and ``files`` lists the
    largest :data:`MAX_STORAGE_FILES` (``files_not_shown`` counts the rest). ``arrays`` are
    the members of the ``.npz`` files ``parts``, by compressed size, with the names
    abbreviated as in the run summary. With several result files (chunks of a run in
    progress) they are summed by name; at most :data:`MAX_STORAGE_ARRAYS` are listed, the
    smaller ones summed as one entry.
    """
    sizes = sorted(((q.stat().st_size, q) for q in everything), key=lambda s: (-s[0], str(s[1])))
    files = [
        {"path": q.relative_to(root).as_posix(), "megabytes": _num(size / 1e6)}
        for size, q in sizes[:MAX_STORAGE_FILES]
    ]
    sums: dict[str, int] = {}
    for path in parts:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                name = info.filename.removesuffix(".npy")
                sums[name] = sums.get(name, 0) + info.compress_size
    rows = list(sums.items())
    if len(rows) > MAX_STORAGE_ARRAYS:
        keep = {name for name, _ in sorted(rows, key=lambda r: -r[1])[: MAX_STORAGE_ARRAYS - 1]}
        rest = sum(size for name, size in rows if name not in keep)
        rows = [r for r in rows if r[0] in keep] + [("(other arrays)", rest)]
    shown = _summary_names([name for name, _ in rows])
    single = len(parts) == 1
    arrays = [
        {
            **({"file": parts[0].relative_to(root).as_posix()} if single else {}),
            "name": shown[name],
            "megabytes": _num(b / 1e6),
        }
        for name, b in rows
    ]
    out: dict[str, Any] = {
        "total_megabytes": _num(sum(size for size, _ in sizes) / 1e6),
        "files": files,
        "arrays": arrays,
    }
    if len(sizes) > MAX_STORAGE_FILES:
        out["files_not_shown"] = len(sizes) - MAX_STORAGE_FILES
    if not single:
        out["arrays_summed_over_files"] = len(parts)
    return out


# ---------------------------------------------------------------------------
# ensembles: background jobs
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Job:
    job_id: str
    spec: dict
    trajectories: int
    seed: int
    output_dir: Path
    chunk_size: int
    num_threads: int | None
    created: float
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    cancel: threading.Event = dataclasses.field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    status: str = "queued"
    done: int = 0
    resumed_from: int = 0
    compute_seconds: float = 0.0
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    notes: list[str] = dataclasses.field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            elapsed = ((self.finished or now) - self.started) if self.started else 0.0
            new = self.done - self.resumed_from
            rate = new / self.compute_seconds if self.compute_seconds > 0 else None
            remaining = self.trajectories - self.done
            eta = remaining / rate if rate and self.status in ("queued", "running") else None
            out = {
                "job_id": self.job_id,
                "status": self.status,
                "trajectories_done": self.done,
                "trajectories_total": self.trajectories,
                "progress": self.done / self.trajectories,
                "resumed_from": self.resumed_from,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "trajectories_per_second": rate,
                "chunk_size": self.chunk_size,
                "seed": self.seed,
                "output_dir": str(self.output_dir),
            }
            if self.error:
                out["error"] = self.error
            if self.notes:
                out["notes"] = list(self.notes)
            if self.status in ("completed", "cancelled", "failed") and self.done:
                out["next_step"] = f"load_results({str(self.output_dir)!r}) for the summary"
            return out


_JOBS: dict[str, _Job] = {}
_JOBS_LOCK = threading.Lock()


def _default_chunk_size(
    mc: mipt.MonitoredCircuit, trajectories: int, threads: int | None = None
) -> int:
    """Chunks of roughly 10 estimated seconds on ``threads`` threads (default: Numba's
    current count), at least a few per thread."""
    import numba

    threads = numba.get_num_threads() if threads is None else threads
    per_traj = max(_trajectory_us(mc) * 1e-6 / _parallel_threads(threads), 1e-9)
    size = int(10.0 / per_traj)
    return int(min(trajectories, max(size, 4 * threads)))


def start_ensemble_job(
    spec: dict[str, Any],
    trajectories: int | None = None,
    output_dir: str | None = None,
    seed: int | None = None,
    chunk_size: int | None = None,
    num_threads: int | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Start (or resume) an ensemble in a background thread; returns the job status.

    The run proceeds in chunks of ``chunk_size`` trajectories (default: about
    10 s each). Each chunk is saved to ``output_dir/chunks/`` as soon as it
    finishes; on completion everything is merged into
    ``output_dir/result.npz``. Results are identical to one
    :func:`run_ensemble` call with the same spec and seed, because trajectory
    ``t`` depends only on ``(spec, seed, t)``.

    Resuming: call again with the same ``output_dir`` and spec (``seed`` may
    be omitted; the stored one is used). Finished trajectories are kept, and a
    larger ``trajectories`` extends a finished run. Without ``trajectories``
    a resume runs to the target stored in ``spec.json``, for a collection of
    graphs too: a new collection run without it stores all of its circuits
    (every graph gets ``circuits_per_graph``), and one started with fewer
    is extended only by an explicit count. Before anything runs, a directory
    made under another randomness contract or ``sparsegf2`` version, or with
    another seed, is refused (the saved results are checked as well as
    ``provenance.json`` and ``spec.json``), and so is a spec that builds
    another protocol than the saved run (the error names a differing key
    and both values) or a graph file whose contents changed since the run
    started.

    One writer per directory: the job holds ``output_dir/.lock`` until it
    ends, and a directory that another job or process is writing to is
    refused. If a reader keeps ``result.npz`` open through the final merge
    (Windows), the job still ends ``completed``, with a note that the merge
    is pending; the next :func:`load_results` or ``start_ensemble_job``
    merges the chunks.
    """
    mc = _build_spec(spec)
    graph_file = _graph_file_record(spec)
    # Without a count, a resume runs to the target in spec.json, read once the directory is
    # known to hold this run; otherwise the count is checked before the directory is made.
    resume_to_target = trajectories is None and _stored_target(output_dir) is not None
    count = 0 if resume_to_target else _trajectory_count(mc, trajectories)
    if num_threads is not None:
        num_threads = _require_int(num_threads, "num_threads", minimum=1)
    if output_dir is None:
        raise ValueError(
            "background jobs need an output_dir (results are saved there chunk by chunk)"
        )
    out = _output_dir(output_dir)
    # One writer per directory: a job of this process is refused by name, and the lock
    # refuses any other writer, in this process or another. The job releases it when it ends.
    _refuse_a_running_job(out)
    lock = _lock_output_dir(out)
    try:
        existing, meta = _load_directory(out)
        notes: list[str] = []
        if meta is not None:
            # The contract and version first: nothing else can be resumed under another one.
            stored = _read_provenance(out)
            _check_contract(out, stored, existing)
            _check_version(out, stored, existing)
            notes.extend(_check_same_run(out, spec, mc, graph_file, meta, existing))
            if "seed" not in meta:
                raise ValueError(
                    f"output_dir {str(out)!r}: its spec.json records no seed; restore it, or "
                    "use a new directory"
                )
            if seed is not None and _resolve_seed(seed) != meta.get("seed"):
                raise ValueError(
                    f"output_dir {str(out)!r} holds a run with seed {meta.get('seed')}; omit "
                    "seed to resume it, or use a new directory"
                )
            seed = meta["seed"]
            _check_saved_seed(out, seed, existing)
            notes.extend(_provenance_notes(stored))
            started_at, now_at = (stored or {}).get("git_commit"), _git_commit()
            if isinstance(started_at, str) and started_at and now_at and started_at != now_at:
                notes.append(
                    f"this resume runs git commit {now_at[:12]}, the run started at "
                    f"{started_at[:12]}; both are recorded in provenance.json"
                )
        elif existing is not None:
            raise ValueError(
                f"output_dir {str(out)!r} holds results without spec.json; use a new directory"
            )
        if resume_to_target:  # the directory holds this run: run to its stored target
            count = _trajectory_count(mc, None if meta is None else meta.get("trajectories"))
        seed = _resolve_seed(seed)
        notes.extend(_seed_notes(seed))
        done = existing.num_trajectories if existing is not None else 0
        job = _Job(
            job_id=uuid.uuid4().hex[:12],
            spec=spec,
            trajectories=max(count, done),
            seed=seed,
            output_dir=out,
            chunk_size=(
                _require_int(chunk_size, "chunk_size", minimum=1)
                if chunk_size is not None
                else _default_chunk_size(mc, count, _running_threads(num_threads))
            ),
            num_threads=num_threads,
            created=time.time(),
            done=done,
            resumed_from=done,
        )
        with _JOBS_LOCK:
            _JOBS[job.job_id] = job
        try:
            with _writing_to(out):
                _write_run_metadata(
                    out,
                    spec,
                    job.trajectories,
                    seed,
                    command or "start_ensemble_job",
                    resumed_from=done if meta is not None else None,
                    graph_file=graph_file,
                )
        except BaseException:
            _forget_job(job)
            raise
    except BaseException:
        lock.release()  # not handed over to a job
        raise
    # The hand-over: from here on the job owns the lock, so an interrupt arriving now (Ctrl-C
    # in the CLI) must not release it under a running job.
    job.thread = threading.Thread(
        target=_run_job, args=(job, mc, lock), name=f"sparsegf2-job-{job.job_id}", daemon=True
    )
    try:
        job.thread.start()
    except RuntimeError:  # no thread could be started, so nothing owns the lock
        _forget_job(job)
        lock.release()
        raise
    status = job.snapshot()
    if notes:
        status["notes"] = [*notes, *status.get("notes", [])]
    return _bounded(status)


def _forget_job(job: _Job) -> None:
    """Remove a job that never started, so it cannot block its output_dir."""
    with _JOBS_LOCK:
        _JOBS.pop(job.job_id, None)


def _canonical(obj: Any) -> str:
    """Exact JSON text of a spec, for comparing a stored run with a new request."""
    return json.dumps(_jsonable(obj, exact=True), sort_keys=True)


_UNREADABLE = object()


def _stored_target(output_dir: Any) -> Any:
    """The trajectory target in ``output_dir/spec.json``: None if there is none (nothing
    is created), or :data:`_UNREADABLE` if the file cannot be read, which the read under
    the directory's lock then reports."""
    if not isinstance(output_dir, (str, os.PathLike)) or not str(output_dir):
        return None
    try:
        meta = _read_meta(Path(output_dir).expanduser().resolve())
    except (OSError, ValueError):
        return _UNREADABLE
    return None if meta is None else meta.get("trajectories")


_ABSENT = object()


def _shown(value: Any) -> str:
    """A spec value as JSON for a message (``absent`` for a missing key), shortened."""
    if value is _ABSENT:
        return "absent"
    text = json.dumps(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _spec_differences(saved: Any, given: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Every key path where two JSON specs differ, as ``(top-level key, message)``.

    Values compare as exact JSON (``1`` and ``1.0`` differ). The keys of
    ``saved`` come first, in their order, then the keys only ``given`` has.
    """
    if isinstance(saved, dict) and isinstance(given, dict):
        for key in [*saved, *(k for k in given if k not in saved)]:
            where = f"{path}.{key}" if path else str(key)
            a, b = saved.get(key, _ABSENT), given.get(key, _ABSENT)
            if a is _ABSENT or b is _ABSENT:
                yield _difference(where, a, b)
            else:
                yield from _spec_differences(a, b, where)
    elif isinstance(saved, list) and isinstance(given, list):
        for i in range(max(len(saved), len(given))):
            a = saved[i] if i < len(saved) else _ABSENT
            b = given[i] if i < len(given) else _ABSENT
            if a is _ABSENT or b is _ABSENT:
                yield _difference(f"{path}[{i}]", a, b)
            else:
                yield from _spec_differences(a, b, f"{path}[{i}]")
    elif _canonical(saved) != _canonical(given):
        yield _difference(path, saved, given)


def _difference(path: str, saved: Any, given: Any) -> tuple[str, str]:
    """``(top-level key of path, message naming the path and both values)``."""
    top = re.split(r"[.\[]", path, maxsplit=1)[0]
    where = path or "the spec"
    return top, f"{where} is {_shown(saved)} in its spec.json and {_shown(given)} here"


# The spec keys that set each field compared by mipt._protocol_difference, where they
# are not the field itself.
_PROTOCOL_KEYS = {
    "num_system": ("num_system", "graph", "edges"),
    "edges": ("graph", "edges", "num_system"),
    "edge_weights": ("edge_weights", "graph", "edges"),
    "checkpoint_layers": ("checkpoints", "layers"),
}


def _spec_difference(saved: Any, given: Any, protocol: str | None = None) -> str | None:
    """A key path where two JSON specs differ, with both values, or None.

    With ``protocol`` (what :func:`sparsegf2.mipt._protocol_difference` says
    differs) the first difference among the keys that set that field is
    named, so changing it back resolves the refusal; otherwise, and if no
    such key differs, the first difference.
    """
    field = None if protocol is None else protocol.split(" (", 1)[0]
    keys = _PROTOCOL_KEYS.get(field or "", (field,))
    first = None
    for top, message in _spec_differences(saved, given):
        if top in keys:
            return message
        first = first or message
        if field is None:
            break
    return first


def _check_same_run(
    out: Path,
    spec: dict,
    mc: mipt.MonitoredCircuit,
    graph_file: dict[str, str] | None,
    meta: dict,
    existing: mipt.EnsembleResult | None,
) -> list[str]:
    """Refuse, before anything runs, a resume or extension that would not continue the saved run.

    A run is identified by its protocol (what the spec builds, compared by
    :func:`sparsegf2.mipt._protocol_difference`, which every merge of saved
    parts also requires), not by the text of its spec. So a spec written
    differently that builds the same protocol continues the run: a cutoff
    given as a budget or per-size mapping that resolves to the same size, a
    range string for a qubit list, a default written out, another path to
    the same graphs. The saved protocol is that of the saved results, or,
    when none are saved yet, the one ``spec.json`` builds.

    A graph file must still hold the bytes recorded in ``spec.json``. If its
    bytes changed, the resume goes on only when saved results show that it
    holds the same graphs; with nothing saved there is nothing to compare
    with, and it is refused. A ``spec.json`` written before the hash was
    recorded is checked against the saved results alone.

    Returns notes for the job status; raises ``ValueError`` naming what differs.
    """
    where = f"output_dir {str(out)!r}"
    stored_spec, given = meta.get("spec"), _jsonable(spec, exact=True)
    same_text = _canonical(stored_spec) == _canonical(given)
    saved: mipt.MonitoredCircuit | None = None
    if existing is not None:
        saved = existing.spec
    elif not same_text:
        with contextlib.suppress(ValueError):  # e.g. its graph file is gone
            saved = _build_spec(stored_spec)
    difference = None if saved is None else mipt._protocol_difference(saved, mc)
    text = None if same_text else _spec_difference(stored_spec, given, difference)
    notes = []
    recorded = meta.get("graph_file")
    if (
        graph_file is not None
        and isinstance(recorded, dict)
        and recorded.get("sha256") not in (None, graph_file["sha256"])
    ):
        if existing is None or difference is not None:
            then = recorded.get("path")
            started = f" (then {then!r})" if then and then != graph_file["path"] else ""
            graphs_differ = (
                f", and the saved results differ from what the spec builds with it now "
                f"({difference} differs)"
                if existing is not None
                else "; with no results saved yet, its graphs cannot be compared"
            )
            raise ValueError(
                f"{where}: the graph file {graph_file['path']!r} changed since the run "
                f"started{started} (sha256 {recorded['sha256'][:16]}... in spec.json, "
                f"{graph_file['sha256'][:16]}... now){graphs_differ}; restore the file the run "
                "used, or use a new directory"
            )
        notes.append(
            f"the graph file {graph_file['path']!r} has other bytes than when the run started, "
            "but holds the graphs the saved results used; spec.json now records its new sha256"
        )
    if text is not None and (saved is None or difference is not None):
        protocol = "" if difference is None else f" (the protocol differs in {difference})"
        raise ValueError(
            f"{where} holds a run with a different spec: {text}{protocol}; resume with the spec "
            "in its spec.json (load_results returns it), or use a new directory"
        )
    if difference is not None:  # the same spec text, but it builds another protocol now
        if graph_file is not None:
            raise ValueError(
                f"{where}: the graph file {graph_file['path']!r} no longer holds the graphs the "
                f"saved results used ({difference} differs); restore the file the run used, or "
                "use a new directory"
            )
        raise ValueError(
            f"{where}: its spec now builds another protocol than the saved results "
            f"({difference} differs); use a new directory"
        )
    if text is not None:
        notes.append(
            f"this spec is written differently from the one in spec.json ({text}) but builds "
            "the same protocol, so it continues the run; spec.json now stores this form"
        )
    return notes


def _run_job(job: _Job, mc: mipt.MonitoredCircuit, lock: _DirLock) -> None:
    chunk_dir = job.output_dir / _CHUNK_DIR
    with job.lock:
        job.started = time.time()
        job.status = "running"
    status, error, notes = "failed", None, []  # what the job reports when it ends
    try:
        while True:
            with job.lock:
                first, total = job.done, job.trajectories
            if first >= total or job.cancel.is_set():
                break
            count = min(job.chunk_size, total - first)
            with _COMPUTE_LOCK:
                t0 = time.perf_counter()
                chunk = mc.run(
                    count, seed=job.seed, first_trajectory=first, num_threads=job.num_threads
                )
                seconds = time.perf_counter() - t0
            chunk_dir.mkdir(exist_ok=True)
            _atomic_save_result(chunk, chunk_dir / f"traj_{first:012d}.npz")
            with job.lock:
                job.done = first + count
                job.compute_seconds += seconds
        finished = job.done >= job.trajectories
        if finished and not _merge_chunks(job.output_dir):
            notes.append(_MERGE_PENDING)
        status = "completed" if finished else "cancelled"
    except Exception as exc:  # reported through job_status
        error = f"{type(exc).__name__}: {exc}\n" + "".join(traceback.format_exception(exc)[-3:])
    finally:
        # The lock first: once the job reports its end, the next writer may start at once.
        lock.release()
        with job.lock:
            job.status, job.error, job.finished = status, error, time.time()
            job.notes.extend(notes)


def _merge_chunks(out: Path) -> bool:
    """Fold the chunk files into result.npz, then delete them (safe to interrupt anywhere).

    Run only by the holder of the directory's lock. If every chunk repeats
    trajectories of result.npz (chunks a held delete left behind), they are
    only deleted: result.npz is not written again. Replacing result.npz is
    retried while a reader holds it (Windows); if it stays in use, the chunks
    are kept and False is returned: the merge is pending, and the next one
    folds them in. An error writing the merged file itself is raised. The
    chunks are then deleted (:func:`_delete_files`); one that stays in use is
    left behind, a repeat of result.npz that a load drops and the next merge
    deletes. Temporary files that killed writers left are deleted too.
    """
    parts = _result_parts(out)
    chunk_parts = [p for p in parts if p.parent.name == _CHUNK_DIR]
    if chunk_parts:
        result_path = out / _RESULT_FILE
        merged = _load_parts_once(parts)
        if merged.num_trajectories > _saved_trajectories(result_path):
            tmp = _save_temporary(merged, result_path)
            try:
                _replace(tmp, result_path)
            except PermissionError:
                return False
        _delete_files(chunk_parts)
    _remove_temporaries(out)
    with contextlib.suppress(OSError):  # a chunk still in use
        (out / _CHUNK_DIR).rmdir()
    return True


def _saved_trajectories(path: Path) -> int:
    """How many trajectories ``path`` (a result.npz) holds, 0 if there is none; reads only
    that array."""
    if not path.exists():
        return 0
    with np.load(path, allow_pickle=False) as z:
        return int(z["trajectories"].size)


def _delete_files(paths: Sequence[Path]) -> None:
    """Delete ``paths``, retrying the ones a reader holds (Windows) with one backoff of
    about 5 s for all of them; the ones still in use after it are left."""
    left = list(paths)
    for delay in (*_RETRY_DELAYS, None):
        in_use = []
        for p in left:
            try:
                p.unlink(missing_ok=True)
            except PermissionError:
                in_use.append(p)
        left = in_use
        if not left or delay is None:
            return
        time.sleep(delay)


def _remove_temporaries(out: Path) -> None:
    """Delete the temporary files of writers that were killed while saving (run by the
    lock holder, so no other writer's file is among them)."""
    stale = [*out.glob(f"{_RESULT_FILE}*.tmp.npz"), *(out / _CHUNK_DIR).glob("*.tmp.npz")]
    for p in stale:
        with contextlib.suppress(OSError):
            p.unlink()


def _get_job(job_id: Any) -> _Job:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id) if isinstance(job_id, str) else None
        known = sorted(_JOBS)
    if job is None:
        raise ValueError(
            f"unknown job_id {job_id!r}; jobs in this process: {known or 'none'} "
            "(jobs do not survive a restart; resume by calling start_ensemble_job with the "
            "same output_dir)"
        )
    return job


def job_status(job_id: str, wait_seconds: float = 0.0) -> dict[str, Any]:
    """Progress of a background job: status, trajectories done, throughput and ETA.

    ``status`` is ``queued``, ``running``, ``completed``, ``cancelled`` or
    ``failed`` (with ``error``). ``wait_seconds`` (at most 60) blocks until the
    job finishes or that time passes, which saves polling round trips.
    """
    job = _get_job(job_id)
    if isinstance(wait_seconds, bool) or not isinstance(wait_seconds, (int, float)):
        raise ValueError("wait_seconds must be a number")
    if wait_seconds < 0 or wait_seconds > 60:
        raise ValueError("wait_seconds must be between 0 and 60")
    if wait_seconds and job.thread is not None:
        job.thread.join(timeout=float(wait_seconds))
    return _bounded(job.snapshot())


def cancel_job(job_id: str) -> dict[str, Any]:
    """Ask a job to stop after its current chunk; saved chunks are kept and the job can be resumed."""
    job = _get_job(job_id)
    job.cancel.set()
    return _bounded(job.snapshot())


def list_jobs() -> dict[str, Any]:
    """All background jobs started in this process, newest first."""
    with _JOBS_LOCK:
        jobs = sorted(_JOBS.values(), key=lambda j: -j.created)
    rows = []
    for job in jobs[:MAX_LIST_SHOWN]:
        s = job.snapshot()
        rows.append(
            {
                k: s[k]
                for k in (
                    "job_id",
                    "status",
                    "trajectories_done",
                    "trajectories_total",
                    "output_dir",
                )
            }
        )
    return _bounded({"jobs": rows, "count": len(jobs)})


# ---------------------------------------------------------------------------
# erasure tools: regions on a graph (sample_erasure_regions), failure curves of a
# saved run (erasure_curves)
# ---------------------------------------------------------------------------

MAX_REGION_COUPLINGS = 12
"""Most couplings one :func:`sample_erasure_regions` call takes (one row each in the result)."""

MAX_REGION_CHAINS = 32
MAX_REGION_BYTES = 2**29
"""Largest memory of the regions :func:`sample_erasure_regions` keeps for the position
diagnostics (all chains and couplings): 512 MiB of int64 node numbers."""

MAX_EXACT_REGIONS = 100_000_000
"""Most regions :func:`sample_erasure_regions` enumerates to give the exact mean and variance of
the boundary (about 8 ns per region compiled)."""

RHAT_WARNING = 1.01
"""The least R-hat limit (Vehtari et al. 2021): see :func:`_rhat_limit`, which raises it for the
number of series tested and for the chains' own autocorrelation."""

TEST_ALPHA = 0.01
"""The chance of a false note from the R-hat series or from Proposition 2's steps in one call
of :func:`sample_erasure_regions` with a correct sampler (Bonferroni over everything tested)."""

ESS_WARNING_PER_CHAIN = 100
"""An effective sample size below this *per chain* gets a note: 100 per chain is the least
Vehtari et al. (2021) ask of an effective sample size, 400 for their four chains, so the
threshold of a call is this times its ``chains``."""

EXCHANGE_MIXING_WARNING = 0.05
"""A replica-exchange acceptance below this between two rungs gets a note."""

DEVIATION_WARNING = 5.0
"""A mean boundary this many standard errors from its exact value gets a note."""

R_HAT_INFINITE = 1000.0
"""R-hat is infinite when every half-chain is constant but they differ (the chains are stuck
apart); JSON has no infinity, so :func:`sample_erasure_regions` reports this number instead."""

MAX_WARNINGS = 6
MAX_CURVE_ERASURES = 16
MAX_CURVE_COUPLINGS = 12
MAX_TOLERANCE_CODES = 4000
"""Most codes whose own tolerance :func:`erasure_curves` computes (see also
:data:`MAX_TOLERANCE_WORK`, which lowers it for large systems)."""

MAX_TOLERANCE_WORK = 1e10
"""Multiply-adds :func:`erasure_curves` spends on the per-code tolerances: the number of codes
times couplings times (n + 1) times the 2,049 grid points of ``tolerance_from_curve``; about 3
seconds here (3.5e9 per second measured, n = 64..1024). Past it fewer codes (at least 10) are used."""

MAX_CURVE_VALUES = 1e8
"""Curve values (codes x couplings x (n + 1)) :func:`erasure_curves` builds and averages: about 4
to 6 seconds here (18 ns per value at n = 256, 38 ns at n = 1024). Past it, evenly spaced codes (at
least 100) are used, for Phi as well, and a note says so."""

INTERPOLATION_MASS_WARNING = 1e-3
"""Interpolated sizes that carry more binomial probability than this make a curve incomplete."""

_ROW_CHUNK = 2000  # curves whose tolerance is computed at once (2049 grid points each)
_REGION_TEXT_CHARS = 240


def _region_graph(graph: Any, num_nodes: Any) -> tuple[np.ndarray, int, str, list[str]]:
    """The one graph of :func:`sample_erasure_regions`: ``(edges, n, family, notes)``.

    A family (``{"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": 1}``, with
    ``num_nodes``), a graph file (its own ``num_nodes``; the first graph of a collection) or
    ``{"edges": [[a, b], ...], "num_nodes": n}``. A collection built with ``realizations`` is
    refused: this tool samples one graph.
    """
    form = (
        'graph must be an object like {"family": "watts_strogatz", "k": 4, "beta": 0.1, '
        '"seed": 1} (with num_nodes), {"family": "cycle"}, {"family": "file", "path": ...} or '
        '{"edges": [[0, 1], ...], "num_nodes": n}'
    )
    if not isinstance(graph, dict) or not {"family", "edges"} & set(graph):
        raise ValueError(f"{form}; got {_shown(graph)}")
    given = (
        None
        if num_nodes is None
        else _require_int(num_nodes, "num_nodes", minimum=1, maximum=MAX_QUBITS)
    )
    inner = graph.get("num_nodes")  # accepted inside the graph object too, beside the family
    if inner is not None:
        inner = _require_int(inner, "graph.num_nodes", minimum=1, maximum=MAX_QUBITS)
    if inner is not None and given is not None and inner != given:
        raise ValueError(f"num_nodes is {given} but graph.num_nodes is {inner}: give one of them")
    given = inner if inner is not None else given
    notes: list[str] = []
    if "edges" in graph:
        _check_keys(graph, ("edges", "num_nodes"), "graph (edges)")
        if given is None:
            raise ValueError(
                "give num_nodes, the number of qubits n (also those without an edge), with a "
                "graph of edges"
            )
        if isinstance(graph["edges"], (list, tuple)) and not graph["edges"]:
            raise ValueError(
                "graph (edges): the edge list is empty; a graph without edges has no boundary"
            )
        try:
            raw = _edge_list(graph["edges"])
            edges = graphs.from_edges(raw, given)
        except ValueError as exc:
            raise ValueError(f"graph (edges): {exc}") from None
        if len(edges) < len(raw):
            notes.append(
                f"{len(raw) - len(edges)} repeated edge(s) were merged: the boundary counts each "
                "edge once (the graph has no weights)"
            )
        return edges, given, "edges", notes
    if graph.get("family") == "file":
        n_file, bank, weights = _read_graph_file(
            {k: v for k, v in graph.items() if k != "num_nodes"}
        )
        if given is not None and given != n_file:
            raise ValueError(
                f"graph (file): the file has num_nodes {n_file}, but num_nodes is {given}: "
                "drop num_nodes (the file carries its own)"
            )
        if len(bank) > 1:
            notes.append(
                f"the graph file holds {len(bank)} graphs; only the first of {len(bank)} graphs "
                "is sampled (save another one alone with graphs.save to check it)"
            )
        if weights is not None:
            notes.append("the file's edge weights are ignored: the boundary counts edges")
        return bank[0], n_file, "file", notes
    if "realizations" in graph:
        raise ValueError(
            "graph.realizations builds a collection of graphs, and this tool samples one: drop "
            "'realizations' (graph i of a collection is built from the seed [seed, i]: give "
            "that as 'seed' to check graph i)"
        )
    if given is None:
        raise ValueError(
            "give num_nodes, the number of qubits n, with a graph family (a graph file carries "
            "its own)"
        )
    built = _build_graph({k: v for k, v in graph.items() if k != "num_nodes"}, given)
    assert isinstance(built, np.ndarray), "without realizations a family yields one graph"
    return built, given, str(graph["family"]), notes


def _region_couplings(couplings: Any) -> np.ndarray:
    """The couplings of :func:`sample_erasure_regions` as given: distinct, |J| <= 1e6, a few."""
    if isinstance(couplings, (int, float, np.number)) and not isinstance(couplings, bool):
        couplings = [couplings]
    if not isinstance(couplings, (list, tuple)):
        raise ValueError(
            f"couplings must be a number or a list of numbers such as [0, 1, 2], got {couplings!r}"
        )
    if any(isinstance(x, (bool, np.bool_, str, bytes)) for x in couplings):
        raise ValueError(f"couplings must be numbers, got {list(couplings)!r}")
    given = erasures._couplings(list(couplings), distinct=True)
    if given.size > MAX_REGION_COUPLINGS:
        raise ValueError(
            f"give at most {MAX_REGION_COUPLINGS} couplings per call (each adds rungs to the "
            f"chain and a row to the result), got {given.size}: split the list over several "
            "calls"
        )
    return given


def _proposition_two(
    couplings: Sequence[float], means: Sequence[float], errors: Sequence[float]
) -> dict[str, Any] | None:
    """Proposition 2 on the estimates: the mean boundary does not increase with the coupling.

    Along the couplings in increasing order, each step from one coupling to the next is a
    one-sided test of "the mean does not rise": ``z = (later mean - earlier mean) /
    hypot(stderr, stderr)``. The ``c - 1`` consecutive steps share the chance
    :data:`TEST_ALPHA` of a false alarm (Bonferroni: a step fails above the normal quantile
    ``z_critical = Phi^-1(1 - alpha / (c - 1))``, 3.1 for 12 couplings, 2.33 for two). A
    step's two means come from chains that exchange states, so their errors are positively
    correlated and the test is a little conservative. Returns None with fewer than two
    couplings, else a dict: ``holds`` (no step fails), ``z_critical``, ``steps`` (c - 1),
    ``worst`` (the step with the largest z) and ``failed`` (the steps above ``z_critical``,
    at most four), each ``{"J": [from, to], "increase": later - earlier, "z": z}`` (a z of 1000
    stands for a rise with no variation at all). A failed step means the estimates are off (too few samples, or a chain that has not
    equilibrated), not that the proposition is false.
    """
    if len(couplings) < 2:
        return None
    order = np.argsort(np.asarray(couplings, dtype=np.float64), kind="stable")
    js = np.asarray(couplings, dtype=np.float64)[order]
    mu = np.asarray(means, dtype=np.float64)[order]
    se = np.asarray(errors, dtype=np.float64)[order]
    steps = len(js) - 1
    z_critical = statistics.NormalDist().inv_cdf(1.0 - TEST_ALPHA / steps)
    found: list[dict[str, Any]] = []
    for i in range(steps):
        rise = float(mu[i + 1] - mu[i])
        error = math.hypot(se[i], se[i + 1])
        # no variation in either: a real rise (beyond rounding) is as significant as can be
        no_variation = R_HAT_INFINITE if rise > 1e-9 else 0.0
        z = rise / error if error > 0 else no_variation
        found.append({"J": [float(js[i]), float(js[i + 1])], "increase": rise, "z": z})
    failed = [s for s in found if s["z"] > z_critical]
    worst = max(found, key=lambda s: s["z"])
    return {
        "holds": not failed,
        "z_critical": z_critical,
        "steps": steps,
        "worst": worst,
        "failed": failed[:4],
    }


def _betacf(a: float, b: float, x: float) -> float:
    """The continued fraction of the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (tiny if abs(d) < tiny else d)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        for aa in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + aa * d
            d = 1.0 / (tiny if abs(d) < tiny else d)
            c = 1.0 + aa / c
            c = tiny if abs(c) < tiny else c
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def _f_upper_tail(f: float, d1: float, d2: float) -> float:
    """``P(F > f)`` for the F distribution with ``d1`` and ``d2`` degrees of freedom.

    The regularized incomplete beta function ``I_x(d2 / 2, d1 / 2)`` at ``x = d2 / (d2 + d1 f)``
    (numerical recipes' continued fraction; no scipy).
    """
    a, b, x = d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f)
    if x <= 0.0:
        return 0.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


@functools.lru_cache(maxsize=512)
def _f_quantile(tail: float, d1: int, d2: int) -> float:
    """The ``f`` with ``P(F > f) = tail`` for the F distribution (bisection on the tail)."""
    lo, hi = 0.0, 1.0
    while _f_upper_tail(hi, d1, d2) > tail:
        lo, hi = hi, hi * 2.0
        if hi > 1e12:
            return math.inf
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _f_upper_tail(mid, d1, d2) > tail:
            lo = mid
        else:
            hi = mid
    return hi


def _rhat_limit(chains: int, samples: int, series: int, tau: float = 1.0) -> float:
    """The R-hat above which a series of draws differs from one stationary distribution.

    The split R-hat of ``chains`` chains of ``samples`` draws (each cut in two halves of
    ``L = samples // 2``, so ``M = 2 x chains`` of them) is, for stationary draws with
    integrated autocorrelation time ``tau``, about ``sqrt(1 + (tau F - 1) / L)`` with ``F``
    following the F distribution with ``M - 1`` and ``M (L - 1)`` degrees of freedom (the
    between-chain over the within-chain variance). The limit is that at the
    ``1 - alpha / series`` quantile, ``alpha = TEST_ALPHA``, so that ``series`` series tested
    together raise a false alarm with chance ``alpha``, and never below
    :data:`RHAT_WARNING`. With ``tau = 1`` (independent draws) and 4 chains of 200 draws it is
    1.0136 for one coupling (18 series), 1.0172 for 8 and 1.0179 for 12 (the most a call
    takes); a chain's own
    ``tau`` (at least 1) raises it, so that autocorrelation alone, which the effective sample
    size reports, does not raise it. The rank-normalized R-hat is approximated by the plain one.
    """
    length = samples // 2
    m = 2 * chains
    if length < 2 or m < 2:
        return math.inf
    q = _f_quantile(TEST_ALPHA / series, m - 1, m * (length - 1))
    return max(RHAT_WARNING, math.sqrt(1.0 + (max(tau, 1.0) * q - 1.0) / length))


def _exact_boundary(
    edges: np.ndarray, n: int, m: int, geometry: str, couplings: np.ndarray
) -> tuple[str, list[dict[str, float] | None]]:
    """The exact mean and variance of the boundary per coupling, and where they come from.

    The ring and the open path have closed forms (whatever their labeling); other graphs
    are enumerated when ``C(n, m)`` is within :data:`MAX_EXACT_REGIONS`; otherwise only
    ``J = 0`` is known (closed form of the uniform law).
    """
    js = couplings.tolist()
    rows: list[dict[str, float] | None] = []
    if geometry in ("cycle", "path"):
        ring = geometry == "cycle"
        law = erasures.ring_boundary_distribution if ring else erasures.path_boundary_distribution
        for j in js:
            values, p = law(n, m, j)
            mean = float(np.dot(values, p))
            rows.append({"mean_boundary": mean, "variance": float(np.dot(p, (values - mean) ** 2))})
        return f"{'ring' if ring else 'path'} closed form (the graph is a {geometry})", rows
    total = math.comb(n, m)
    if total <= MAX_EXACT_REGIONS:
        counts = erasures.boundary_counts(edges, n, m, budget=MAX_EXACT_REGIONS)
        moments = erasures.boundary_moments(counts, js)
        rows = [
            {"mean_boundary": float(a), "variance": float(b)}
            for a, b in zip(moments["mean"], moments["variance"], strict=True)
        ]
        return f"enumeration of all C({n}, {m}) = {total} regions", rows
    mean0, var0 = erasures.uniform_boundary_moments(edges, n, m)
    rows = [{"mean_boundary": mean0, "variance": var0} if j == 0 else None for j in js]
    return (
        f"uniform closed form at J = 0 only (C({n}, {m}) has {len(str(total))} digits: too many "
        "regions to enumerate, and the graph is neither a cycle nor a path)"
    ), rows


def _regions_seconds(
    *,
    n: int,
    m: int,
    num_edges: int,
    exact: bool,
    rungs: int,
    couplings: int,
    burn_in: int,
    samples: int,
    thin: int,
    chains: int,
    enumerated: int,
) -> float:
    """A pessimistic estimate of the seconds :func:`sample_erasure_regions` takes.

    The chain costs ``rungs x (burn_in + (samples - 1) thin) x n`` swap proposals per chain,
    each ``O(degree)``: measured at 19 to 32 ns at mean degree 4 and 30 ns at 8 on one thread
    of an AMD Ryzen 9 9950X3D (n = 64..1024, 2026-10-16), priced at ``30 + 7.5 x degree`` ns
    (about 3 times that, as :func:`_correlated_us`). The exact samplers cost up to 12 us each
    at n = 1024 (priced ``3 + 0.04 n`` us) and need no diagnostics. A chain's diagnostics cost
    about 17 ns per node of a region (the 8 followed nodes) and 100 ns per sample, run twice
    (``sample_regions`` checks its own chain too; priced at 40 and 100 ns), and enumerating
    costs 8 ns per region (priced at 32 ns).
    """
    if exact:
        sampling = chains * couplings * samples * (3.0 + 0.04 * n) * 1e-6
        diagnostics = 0.0
    else:
        degree = 2.0 * num_edges / n
        sweeps = burn_in + (samples - 1) * thin
        sampling = chains * rungs * sweeps * n * (30.0 + 7.5 * degree) * 1e-9
        diagnostics = 2 * chains * couplings * samples * (40e-9 * m + 100e-9)
    return 0.1 + sampling + diagnostics + 32e-9 * enumerated


def _sig(x: float | None, digits: int = 5) -> float | None:
    """``x`` to ``digits`` significant digits; None for None, NaN and infinities."""
    if x is None or not math.isfinite(x):
        return None
    return float(f"{x:.{digits}g}")


def _where(node: int | None) -> str:
    """What a diagnostic followed: the boundary, or the membership of one node."""
    return "boundary" if node is None else f"node {node}"


def _listed(items: list[str], limit: int = 8) -> str:
    """Comma-separated entries, the first ``limit`` of them and a count of the rest."""
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" and {len(items) - limit} more"


def _regions_mixing_notes(
    rows: list[dict[str, Any]],
    details: list[dict[str, Any]],
    *,
    method: str,
    chains: int,
    samples: int,
    burn_in: int,
    pair_acceptance: np.ndarray,
    ladder: np.ndarray,
    proposition: dict[str, Any] | None = None,
) -> list[str]:
    """The notes on how well the chains mixed: one per kind of problem, with what to change.

    R-hat above its limit (:func:`_rhat_limit`; chains only: exact samples are independent),
    an effective sample size below :data:`ESS_WARNING_PER_CHAIN` per chain (the threshold is
    that times ``chains``: 400 for the default four), a mean boundary more than
    :data:`DEVIATION_WARNING` standard errors from its exact value, a replica-exchange
    acceptance below :data:`EXCHANGE_MIXING_WARNING` between two rungs, and a rise of the mean
    boundary with the coupling beyond chance (``proposition``, from :func:`_proposition_two`).
    The entries name the coupling, the value and what was followed (the boundary, or the
    membership of a node).
    """
    chain = method == "chain"
    rhat, ess, off, exchange = [], [], [], []
    for row, info in zip(rows, details, strict=True):
        j = f"J = {row['coupling']:g}"
        if chain and info["r_hat"] is not None and info["r_hat"] > info["r_hat_limit"]:
            shown = "infinite" if info["r_hat"] >= R_HAT_INFINITE else f"{info['r_hat']:.3f}"
            rhat.append(f"{j} ({shown} > {info['r_hat_limit']:.3f}, {_where(info['r_hat_node'])})")
        if row["ess"] is not None and row["ess"] < ESS_WARNING_PER_CHAIN * chains:
            ess.append(f"{j} ({row['ess']:.0f}, {_where(info['slowest'])})")
        dev = row["deviation_se"]
        if dev is not None and abs(dev) > DEVIATION_WARNING:
            off.append(f"{j} ({dev:+.1f})")
    notes = []
    if rhat:
        notes.append(
            f"R-hat over {chains} chains above the limit that one stationary distribution allows "
            f"(given the chains' length and autocorrelation) at {_listed(rhat)}: the chains "
            "differ more than that, so they are not in equilibrium yet, or sit in different "
            f"regions, or are too short for their autocorrelation; raise burn_in (now {burn_in}), "
            f"then thin or samples (now {samples}) or chains, add couplings between so that "
            "replica exchange helps, and compare another seed"
        )
    if ess:
        notes.append(
            f"effective sample size below {ESS_WARNING_PER_CHAIN * chains} "
            f"({ESS_WARNING_PER_CHAIN} per chain) at {_listed(ess)}: raise samples (now "
            f"{samples}) or chains (now {chains}), or thin more"
        )
    if off:
        notes.append(
            f"mean boundary more than {DEVIATION_WARNING:g} standard errors from its exact value "
            f"at {_listed(off)}: "
            + (
                "the chains are biased (too short a burn_in leaves the starting states' "
                "boundaries behind) or the error is underestimated; raise burn_in and compare "
                "another seed"
                if chain
                else "the sampler or the exact value is wrong: report it"
            )
        )
    if chain:
        for k, rate in enumerate(pair_acceptance.tolist()):
            if math.isfinite(rate) and rate < EXCHANGE_MIXING_WARNING:
                exchange.append(f"J = {ladder[k]:g} and {ladder[k + 1]:g} ({100 * rate:.1f}%)")
    if exchange:
        notes.append(
            f"replica exchange accepted in under {100 * EXCHANGE_MIXING_WARNING:.0f}% of attempts "
            f"between {_listed(exchange, 6)}: add couplings between them (or pass a smaller "
            "max_step); the colder chains hardly move"
        )
    if proposition is not None and not proposition["holds"]:
        rises = [
            f"J = {s['J'][0]:g} to {s['J'][1]:g} ({s['increase']:+.3g}, z = {s['z']:.1f})"
            for s in proposition["failed"]
        ]
        notes.append(
            f"the mean boundary rises with the coupling at {_listed(rises)}, above z = "
            f"{proposition['z_critical']:.2f} ({proposition['steps']} steps, {100 * TEST_ALPHA:g}% "
            "false-alarm budget); Proposition 2 forbids that in expectation, so the estimates "
            f"are off, not the proposition: raise samples (now {samples}) or chains (now "
            f"{chains}), then burn_in (now {burn_in}), and repeat"
        )
    return notes


def sample_erasure_regions(
    graph: dict[str, Any],
    size: int,
    couplings: float | list[float],
    samples: int = 200,
    *,
    num_nodes: int | None = None,
    burn_in: int | None = None,
    thin: int = 1,
    tempering: bool = True,
    chains: int = 4,
    seed: int | None = None,
    method: str = "auto",
    ladder: str | list[float] = "auto",
    max_step: float | None = None,
    max_seconds: float = SYNC_ENSEMBLE_MAX_SECONDS,
) -> dict[str, Any]:
    """Sample geometry-correlated erasure regions on a graph and check how well they mix.

    A region ``A`` of ``size`` nodes is drawn with probability ``exp(-J b(A)) / Z``, ``b(A)``
    the edges of the graph leaving ``A`` (``J > 0`` favors compact regions, ``J = 0`` is
    uniform). This is the graph side of the ``correlated_erasure`` probe, with no circuit:
    ``chains`` independent runs of :func:`sparsegf2.erasures.sample_regions` (chain ``i``
    seeded ``[seed, i]``; exact independent samples when the graph is a cycle or a path,
    otherwise a swap chain with replica exchange), so the settings can be checked before a
    long run uses them. ``graph`` is ``{"family": ..., params}`` with ``num_nodes``,
    ``{"family": "file", "path": ...}`` (the first graph of a collection) or
    ``{"edges": [[a, b], ...], "num_nodes": n}``; ``burn_in=None`` is
    ``max(200, 8 n)`` sweeps. ``couplings``: at most :data:`MAX_REGION_COUPLINGS` distinct
    numbers, each with ``|J| <= erasures.MAX_ABS_COUPLING`` (10^6; beyond it the exact
    ``J -> inf`` routes of the code side apply).

    Per coupling (in the order given):

    * ``mean_boundary`` with ``stderr = sqrt(variance / ess_boundary)``, ``ess_boundary`` the
      effective sample size of the boundary itself;
      ``ess_boundary`` and ``ess`` are capped at ``chains x samples``: the integrated
      autocorrelation time of a short series can be estimated below 1 from noise, which would
      make the error bar smaller than that of independent draws (``tau_int`` is as estimated);
    * ``acceptance`` of the swaps and ``exchange_acceptance`` of the replica exchanges next to
      the coupling (null for exact samples);
    * ``tau_int`` and ``ess``: the integrated autocorrelation time (largest) and effective sample
      size (smallest) over the boundary and the membership of 8 spread nodes (``slowest`` says
      which): the boundary alone looks mixed on a ring long before the regions have moved;
    * ``r_hat``, ``r_hat_limit`` and ``r_hat_at``: the R-hat over the chains of the series (the
      boundary or a node) that is furthest above its limit, its limit and which series it is.
      The limit is what stationary draws of that series' own autocorrelation allow
      (:func:`_rhat_limit`): the 99% point of the F distribution of the split R-hat, shared
      over the ``2 x 9 x couplings`` series tested, so that a correct sampler raises it with
      chance :data:`TEST_ALPHA` (1%) per call; it is never below 1.01. Measured on correct swap
      chains (4 chains, default burn-in, J = 0 to 2): on a ring of 32 nodes (size 8, the chain
      forced) and a Watts-Strogatz graph of 14 nodes (size 5) the R-hat note fired in 2.7% and
      2.3% of 300 calls at 200 samples and in none of 100 at 800, while the limit for
      independent draws alone was exceeded in 100% and 87% of 60 calls at 200 samples (and 1.01
      in 100% and 97%): that R-hat sees autocorrelation, which the effective sample size
      reports. Where a record is too short for its autocorrelation the note fires on a correct
      chain too (the same 14-node graph with 12 couplings up to J = 2.2: 22% of 60 calls at 200
      samples, none at thin = 5, and the effective sample size note in 95%); a graph whose
      regions really move slowly (a Watts-Strogatz graph of 64 nodes, size 16, J up to 2) is
      flagged in 90% of 40 calls at 200 samples and in none of 20 at thin = 20. An ``r_hat``
      of 1000 stands for infinity (every half-chain constant, and different: the chains are
      stuck apart); null means that nothing varied at all (and, for exact samples, that R-hat
      does not apply);
    * ``min_boundary_seen``, and ``exact`` / ``deviation_se``: the exact mean and variance where
      they exist (``exact_source`` says which: the ring or path closed form, the enumeration of
      every region, or the uniform law at ``J = 0``) and the mean's distance from it in
      standard errors.

    For exact samples (a cycle or a path) the draws are independent: ``tau_int`` is 1, ``ess``
    is ``chains x samples``, and R-hat, ``r_hat_limit`` and the acceptances are null.

    ``nonincreasing_in_J`` tests Proposition 2 on the estimates (:func:`_proposition_two`):
    each step to the next coupling must not rise by more than the normal quantile that
    shares a 1% false-alarm chance over the ``c - 1`` steps (3.1 for 12 couplings, measured
    at 1.5% false alarms over 400 calls with 12 couplings 0.001 apart on a ring); ``proposition2``
    gives that ``z_critical``, the ``worst`` step and the ``failed`` steps with their z. A
    failure means the estimates are off. ``mixing`` lists what looks wrong (R-hat above its
    limit, effective sample size below 100 per chain, exchange acceptance below 5%, a mean far from its
    exact value, a failed Proposition-2 step) and what to change; ``warnings`` passes on the
    sampler's own. No note is no proof of equilibrium: repeat with another seed and a longer
    burn-in. Three example regions at the largest coupling are written as node runs
    ``start:stop[:step]``.

    The call is refused when its estimated time (:func:`_regions_seconds`, pessimistic)
    exceeds ``max_seconds``; the message says how to shrink it. Writes nothing.
    """
    start = time.perf_counter()
    samples = _require_int(samples, "samples", minimum=8)
    chains = _require_int(chains, "chains", minimum=1)
    if chains > MAX_REGION_CHAINS:
        raise ValueError(f"chains must be at most {MAX_REGION_CHAINS}, got {chains}")
    thin = _require_int(thin, "thin", minimum=1)
    burn = None if burn_in is None else _require_int(burn_in, "burn_in", minimum=0)
    tempering = _require_bool(tempering, "tempering")
    method = _require_name(method, "method", erasures.METHODS)
    if method not in erasures.METHODS:
        raise ValueError(f"method must be one of {list(erasures.METHODS)}, got {method!r}")
    extra = erasures._ladder_extra(ladder)
    step = None if max_step is None else erasures._max_step(max_step)
    max_seconds = _require_number(max_seconds, "max_seconds", 0.0, math.inf)
    seed = _resolve_seed(seed)
    given = _region_couplings(couplings)
    edges, n, family, notes = _region_graph(graph, num_nodes)
    m = _require_int(size, "size")
    if not 1 <= m <= n - 1:
        raise ValueError(
            f"size must be between 1 and {n - 1} (the number of erased nodes: 0 and {n} erase "
            f"nothing or everything), got {m}"
        )
    if burn is None:
        burn = erasures.default_burn_in(n)
    geometry = _graph_geometry(edges, n)
    if method == "exact" and geometry == "other":
        erasures.sample_regions(edges, n, m, [0.0], 8, method="exact", seed=0)  # refuses
    exact = method != "chain" and geometry != "other"
    c = given.size
    rungs = c
    if not exact:
        rungs = len(
            erasures._chain_ladder(
                np.sort(given),
                ladder,
                extra,
                tempering,
                step,
                lambda: erasures.uniform_boundary_moments(edges, n, m)[1],
            )[0]
        )
    if 8 * chains * c * samples * (m + 1) > MAX_REGION_BYTES:
        raise ValueError(
            f"the regions of {chains} chains x {c} couplings x {samples:,} samples of size {m} "
            f"would take {8 * chains * c * samples * (m + 1) / 2**20:,.0f} MiB (limit "
            f"{MAX_REGION_BYTES // 2**20} MiB): use fewer samples (thin more instead), chains "
            "or couplings, or a smaller size"
        )
    enumerated = 0
    if geometry == "other" and math.comb(n, m) <= MAX_EXACT_REGIONS:
        enumerated = math.comb(n, m)
    estimate = _regions_seconds(
        n=n,
        m=m,
        num_edges=len(edges),
        exact=exact,
        rungs=rungs,
        couplings=c,
        burn_in=burn,
        samples=samples,
        thin=thin,
        chains=chains,
        enumerated=enumerated,
    )
    if estimate > max_seconds:
        raise ValueError(
            f"this call is estimated at about {estimate:.0f} s, above max_seconds="
            f"{max_seconds:g}: shrink it with fewer chains (now {chains}), fewer samples (now "
            f"{samples}) or a smaller burn_in (now {burn}) or thin (now {thin}), fewer "
            "couplings (each adds rungs), or a smaller graph; or raise max_seconds (the CLI "
            "option --max-seconds) if you are willing to wait"
        )

    regions = np.empty((chains, c, samples, m), dtype=np.int64)
    bounds = np.empty((chains, c, samples), dtype=np.int64)
    swap = np.empty((chains, c))
    pairs: list[np.ndarray] = []
    seen: dict[str, int] = {}
    first: erasures.RegionSample | None = None
    for i in range(chains):
        drawn = erasures.sample_regions(
            edges,
            n,
            m,
            given,
            samples,
            burn_in=burn,
            thin=thin,
            tempering=tempering,
            seed=[seed, i],
            method=method,
            ladder=ladder,
            max_step=max_step,
        )
        first = drawn if first is None else first
        regions[i], bounds[i], swap[i] = drawn.regions, drawn.boundaries, drawn.acceptance
        pairs.append(drawn.exchange_acceptance)
        for text in drawn.warnings:
            seen[text] = seen.get(text, 0) + 1
    assert first is not None
    sample = first
    used = sample.method
    rungs_used = sample.ladder
    pair_acceptance = analysis.mean_and_error(np.stack(pairs), axis=0)[0]
    swap_mean = analysis.mean_and_error(swap, axis=0)[0]

    source, truth = _exact_boundary(edges, n, m, sample.geometry, given)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    draws = chains * samples
    followed = len({v * n // erasures.REGION_PROBES for v in range(erasures.REGION_PROBES)})
    series = 2 * (1 + followed) * c  # tested for R-hat: bulk and folded, boundary and nodes
    raw: list[tuple[float, float]] = []  # each coupling's mean boundary and its standard error
    for k in range(c):
        j = float(given[k])
        exact_row = truth[k]
        if used == "exact":
            # independent draws: no autocorrelation to estimate, no chains to compare
            mean = float(np.mean(bounds[:, k]))
            variance = float(np.var(bounds[:, k]))
            diag = {"mean": mean, "variance": variance, "std_error": math.sqrt(variance / draws)}
            tau, ess, ess_boundary = 1.0, float(draws), float(draws)
            slowest = r_hat = r_node = r_limit = None
        else:
            diag = erasures.chain_diagnostics(bounds[:, k])
            where = erasures.region_diagnostics(regions[:, k], n)
            tau_b, tau_p = diag["tau_int"], where["tau_int_max"]
            slowest = None if tau_b >= tau_p else where["slowest"]
            # an effective size feeds an error bar, so it is capped at the draws: the
            # estimated tau of a short series can fall below 1 from noise (tau_int itself
            # is reported as estimated)
            tau, ess, ess_boundary = (
                max(tau_b, tau_p),
                min(diag["ess"], where["ess_min"], float(draws)),
                min(diag["ess"], float(draws)),
            )
            diag = {**diag, "std_error": math.sqrt(diag["variance"] / ess_boundary)}
            # R-hat of the boundary and of each followed node against its own limit; the row
            # shows the series furthest above (or nearest to) its limit
            candidates = [(diag["rhat"], tau_b, None)] + [
                (r, t, v)
                for r, t, v in zip(
                    where["rhat"].tolist(),
                    where["tau_int"].tolist(),
                    where["vertices"],
                    strict=True,
                )
            ]
            scored = [
                (min(r, R_HAT_INFINITE), _rhat_limit(chains, samples, series, t), v)
                for r, t, v in candidates
                if not math.isnan(r)
            ]
            r_hat, r_limit, r_node = (
                max(scored, key=lambda s: s[0] / s[1]) if scored else (None,) * 3
            )
        position = int(np.searchsorted(rungs_used, j)) if used == "chain" else -1
        near = [
            pair_acceptance[p]
            for p in (position - 1, position)
            if 0 <= p < len(pair_acceptance) and math.isfinite(pair_acceptance[p])
        ]
        raw.append((diag["mean"], diag["std_error"]))
        deviation = None
        if exact_row is not None and diag["std_error"] > 0:
            deviation = (diag["mean"] - exact_row["mean_boundary"]) / diag["std_error"]
        rows.append(
            {
                "coupling": j,
                "mean_boundary": _sig(diag["mean"]),
                "stderr": _sig(diag["std_error"]),
                "variance": _sig(diag["variance"]),
                "acceptance": _sig(swap_mean[k]),
                "exchange_acceptance": _sig(min(near)) if near else None,
                "tau_int": _sig(tau),
                "ess": _sig(ess),
                "ess_boundary": _sig(ess_boundary),
                "slowest": None if used == "exact" else _where(slowest),
                "r_hat": _sig(r_hat),
                "r_hat_limit": _sig(r_limit),
                "r_hat_at": None if r_hat is None else _where(r_node),
                "min_boundary_seen": int(np.min(bounds[:, k])),
                "exact": exact_row,
                "deviation_se": _sig(deviation),
            }
        )
        details.append(
            {
                "r_hat": r_hat,
                "r_hat_limit": r_limit,
                "r_hat_node": r_node,
                "slowest": slowest,
            }
        )
    proposition = _proposition_two(
        [r["coupling"] for r in rows], [a for a, _ in raw], [b for _, b in raw]
    )

    top = int(np.argmax(given))
    examples = []
    cut_examples = False
    for k in range(3):
        chain_i, record = k % chains, samples - 1 - (k // chains) * (samples // 3)
        text = _compress_ints(regions[chain_i, top, record].tolist())
        if len(text) > _REGION_TEXT_CHARS:
            text, cut_examples = text[: _REGION_TEXT_CHARS - 3] + "...", True
        examples.append({"boundary": int(bounds[chain_i, top, record]), "nodes": text})

    mixing = _regions_mixing_notes(
        rows,
        details,
        method=used,
        chains=chains,
        samples=samples,
        burn_in=burn,
        pair_acceptance=pair_acceptance,
        ladder=rungs_used,
        proposition=proposition,
    )
    warnings_out = [f"{text} [in {k} of {chains} chains]" for text, k in seen.items()]
    if len(warnings_out) > MAX_WARNINGS:
        more = len(warnings_out) - MAX_WARNINGS
        warnings_out = [*warnings_out[:MAX_WARNINGS], f"and {more} more"]
    notes += _seed_notes(seed)
    notes.append(
        f"chain i runs erasures.sample_regions with seed [{seed}, i], i = 0..{chains - 1}; "
        "boundary means and errors pool the chains; stderr = sqrt(variance / ess_boundary), "
        "while ess and tau_int are those of the slowest of the boundary and the followed nodes; "
        "ess and ess_boundary are capped at chains x samples (tau_int is as estimated)"
    )
    if used == "exact":
        notes.append(
            "the samples are exact and independent (the graph is a "
            f"{sample.geometry}): tau_int is 1, ess = chains x samples, and burn_in, thin, "
            "tempering, ladder, R-hat and the acceptances do not apply (null)"
        )
    notes.append(
        "an empty mixing list is no proof of equilibrium (a bias left by too short a burn_in "
        "can pass every check): repeat with another seed and a longer burn_in and compare"
    )
    out: dict[str, Any] = {
        "graph": {
            "family": family,
            "num_nodes": n,
            "num_edges": len(edges),
            "geometry": sample.geometry,
        },
        "size": m,
        "samples": samples,
        "chains": chains,
        "method": used,
    }
    if used == "chain":
        out.update(
            burn_in=sample.burn_in,
            thin=sample.thin,
            tempering=sample.tempering,
            ladder={
                "rungs": len(rungs_used),
                "lowest": rungs_used[0],
                "highest": rungs_used[-1],
                "max_step": sample.max_step,
            },
        )
    out.update(
        seed=seed,
        couplings=rows,
        exact_source=source,
        nonincreasing_in_J=None if proposition is None else proposition["holds"],
        proposition2=None if proposition is None else _proposition_summary(proposition),
        mixing=mixing,
        warnings=warnings_out,
        examples={"coupling": float(given[top]), "regions": examples},
        estimated_seconds=estimate,
        elapsed_seconds=time.perf_counter() - start,
        notes=notes,
    )
    if cut_examples:
        _mark_truncated(
            out,
            f"an example region is cut to {_REGION_TEXT_CHARS} characters (its node runs are "
            "listed in full by erasures.sample_regions in Python)",
        )
    return _bounded(out)


def _proposition_summary(proposition: dict[str, Any]) -> dict[str, Any]:
    """The Proposition-2 test for the result: its critical z, the worst step and the failed ones."""

    def step(s: dict[str, Any]) -> dict[str, Any]:
        return {"J": s["J"], "increase": _sig(s["increase"]), "z": _sig(s["z"], 3)}

    return {
        "z_critical": _sig(proposition["z_critical"], 3),
        "steps": proposition["steps"],
        "worst": step(proposition["worst"]),
        "failed": [step(s) for s in proposition["failed"]],
    }


def _graph_geometry(edges: np.ndarray, n: int) -> str:
    """``"cycle"`` or ``"path"`` when the graph is one cycle or one path through all its nodes
    (in any labeling), else ``"other"``: the graphs whose regions are sampled exactly."""
    indptr, indices = erasures._csr(n, np.ascontiguousarray(edges, dtype=np.int64))
    _, kind = erasures._chain.geometry_order(indptr, indices, n)
    return erasures._GEOMETRIES[int(kind)]


def _read_saved_result(path: Any) -> tuple[mipt.EnsembleResult, Path]:
    """A saved run (an output directory, its chunks included, or a ``.npz``), read-only."""
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError("path must be a string")
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ValueError(f"no such file or directory: {str(p)!r}")
    try:
        if p.is_dir():
            result, _ = _load_directory(p)
            if result is None:
                raise ValueError(f"no saved results in {str(p)!r} (expected result.npz or chunks/)")
        else:
            result = _retrying(functools.partial(_load_result_file, p))
    except OSError as exc:  # still in use after the retries, or not ours to read
        raise _unreadable(p, exc) from None
    return result, p


def _curve_erasures(e: Any) -> list[float] | None:
    """The erasure probabilities asked for (a number or a list), or None for the default grid."""
    if e is None:
        return None
    values = [e] if isinstance(e, (int, float, np.number)) and not isinstance(e, bool) else e
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"e must be a number or a list of numbers in [0, 1], got {e!r}")
    if not values:
        raise ValueError("give at least one value of e")
    if len(values) > MAX_CURVE_ERASURES:
        raise ValueError(
            f"e has {len(values)} values; give at most {MAX_CURVE_ERASURES} values (all are "
            "shown): analysis.failure_probability_from_curve takes any number in Python"
        )
    out = []
    for x in values:
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.number)):
            raise ValueError(f"e must be numbers in [0, 1], got {x!r}")
        if not 0.0 <= float(x) <= 1.0:
            raise ValueError(f"e must be numbers in [0, 1], got {float(x):g}")
        out.append(float(x))
    return out


def _default_erasures(n: int) -> list[float]:
    """Eight probabilities log-spaced from ``1 / n`` (about one erased qubit) to 1/2, 2 digits."""
    low = 1.0 / n
    if low >= 0.5:
        return [0.5]
    grid = {float(f"{x:.2g}") for x in np.geomspace(low, 0.5, 8)}
    return sorted(grid | {0.5})


def _binomial_mass(n: int, es: Sequence[float], sizes: np.ndarray) -> np.ndarray:
    """``sum over m in sizes of C(n, m) e^m (1 - e)^(n - m)`` for each ``e``: the probability
    that a binomial erasure has one of these sizes."""
    out = np.zeros(len(es))
    if not sizes.size:
        return out
    m = sizes.astype(np.float64)
    log_comb = np.array(
        [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) for k in sizes.tolist()]
    )
    for i, x in enumerate(es):
        if x == 0.0:
            out[i] = float(np.any(sizes == 0))
        elif x == 1.0:
            out[i] = float(np.any(sizes == n))
        else:
            out[i] = float(np.exp(log_comb + m * math.log(x) + (n - m) * math.log1p(-x)).sum())
    return np.minimum(out, 1.0)


def _layers_text(layers: np.ndarray) -> str:
    """A probe's evaluation layers for a message: all of them if few, else a range."""
    values = layers.tolist()
    if len(values) <= 10:
        return str(values)
    return f"{len(values)} layers from {values[0]} to {values[-1]}"


def _chunk_rows(n: int, c: int) -> int:
    """Codes per chunk: at most :data:`_ROW_CHUNK` curves and about 4 million curve values."""
    c = max(c, 1)
    return max(1, min(_ROW_CHUNK // c, (1 << 22) // (c * (n + 1))))


MIN_CURVE_CODES = 100
MIN_TOLERANCE_CODES = 10


def _curve_plan(n: int, c: int, codes: int) -> tuple[int, int]:
    """How many of ``codes`` codes to use: ``(for Phi and the mean curve, for the per-code tolerance)``.

    The curves of ``c`` couplings are ``codes x c x (n + 1)`` values, at most
    :data:`MAX_CURVE_VALUES` of them (but at least :data:`MIN_CURVE_CODES` codes); the
    per-code tolerance costs ``codes x c x (n + 1) x 2049`` multiply-adds, at most
    :data:`MAX_TOLERANCE_WORK` (but at least :data:`MIN_TOLERANCE_CODES` codes) and
    :data:`MAX_TOLERANCE_CODES` codes. A smaller number means evenly spaced codes.
    """
    per_curve = max(c, 1) * (n + 1)
    used = min(codes, max(MIN_CURVE_CODES, int(MAX_CURVE_VALUES // per_curve)))
    work = int(MAX_TOLERANCE_WORK // (per_curve * 2049))
    return used, min(used, MAX_TOLERANCE_CODES, max(MIN_TOLERANCE_CODES, work))


def _curve_statistics(
    chunks: Iterable[np.ndarray],
    n: int,
    es: list[float],
    delta: float,
    codes: int,
    tolerance_codes: int,
) -> dict[str, Any]:
    """Ensemble statistics of failure curves, one chunk of codes at a time.

    Each chunk is ``(rows, c, n + 1)``: the curve P(m), m = 0..n, of each code at each of
    ``c`` couplings (``c = 1`` for the uniform channel); together they hold ``codes`` codes.
    Returns, per coupling: the mean and standard error of ``Phi(e)`` over all ``codes``
    (``phi_mean``, ``phi_stderr``), the tolerance ``e_delta`` of each code's own curve
    summarized by its mean, median and standard error over ``tolerance_codes`` codes spaced
    evenly (``tol_mean``, ``tol_median``, ``tol_stderr``), the tolerance of the mean curve
    (``tol_of_mean``: the ensemble's, what ``analysis.erasure_tolerance`` gives) and of the
    mean curve moved by plus and minus its standard error at every size
    (``tol_range``: the tolerance of the higher curve, then of the lower one), whether the
    mean curve is nondecreasing in m (``tol_of_mean_monotone``: where it is not, the
    tolerance is the last crossing of delta, not the only one), and the fraction of the
    ``tolerance_codes`` codes whose curve is nondecreasing in m (``monotone``).
    """
    keep = tolerance_codes
    # running mean and sum of squared deviations (Chan et al.) of Phi and of the curve itself
    mean = squares = curve_mean = curve_squares = np.zeros(0)
    tolerances: list[np.ndarray] = []
    flags: list[np.ndarray] = []
    done = 0
    for chunk in chunks:
        rows = chunk.shape[0]
        phi = analysis.failure_probability_from_curve(chunk, n, es)  # (rows, c, len(es))
        index = np.arange(done, done + rows)
        picked = (index + 1) * keep // codes > index * keep // codes
        if picked.any():
            tol, mono = analysis.tolerance_from_curve(chunk[picked], n, delta, return_monotone=True)
            tolerances.append(tol)
            flags.append(mono)
        here, here_curve = phi.mean(axis=0), chunk.mean(axis=0)
        around = ((phi - here) ** 2).sum(axis=0)
        around_curve = ((chunk - here_curve) ** 2).sum(axis=0)
        if not done:
            mean, squares, curve_mean, curve_squares = here, around, here_curve, around_curve
        else:
            share = done * rows / (done + rows)
            squares = squares + around + (here - mean) ** 2 * share
            curve_squares = curve_squares + around_curve + (here_curve - curve_mean) ** 2 * share
            mean = mean + (here - mean) * rows / (done + rows)
            curve_mean = curve_mean + (here_curve - curve_mean) * rows / (done + rows)
        done += rows
    assert done == codes, "the chunks must hold every code"
    with np.errstate(invalid="ignore", divide="ignore"):
        stderr = np.sqrt(squares / (codes - 1) / codes)
        curve_stderr = np.sqrt(curve_squares / (codes - 1) / codes)
    each = np.concatenate(tolerances, axis=0)  # (kept, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        tol_err = np.std(each, axis=0, ddof=1) / math.sqrt(each.shape[0])
    higher = np.clip(curve_mean + np.nan_to_num(curve_stderr), 0.0, 1.0)
    lower = np.clip(curve_mean - np.nan_to_num(curve_stderr), 0.0, 1.0)
    tol_range = np.stack(
        [
            np.atleast_1d(analysis.tolerance_from_curve(higher, n, delta)),
            np.atleast_1d(analysis.tolerance_from_curve(lower, n, delta)),
        ],
        axis=-1,
    )
    of_mean, of_mean_monotone = analysis.tolerance_from_curve(
        curve_mean, n, delta, return_monotone=True
    )
    return {
        "phi_mean": mean,
        "phi_stderr": stderr,
        "phi_codes": codes,
        "tol_mean": each.mean(axis=0),
        "tol_median": np.median(each, axis=0),
        "tol_stderr": tol_err,
        "tol_of_mean": np.atleast_1d(of_mean),
        "tol_of_mean_monotone": np.atleast_1d(of_mean_monotone),
        "tol_range": tol_range,
        "monotone": np.concatenate(flags, axis=0).mean(axis=0),
        "tol_codes": each.shape[0],
    }


def _tolerance_entry(stats: dict[str, Any], pick: int | slice) -> dict[str, Any]:
    """The ``e_delta`` fields of one coupling (``pick`` an index) or of the couplings shown."""
    return {
        "of_mean_phi": stats["tol_of_mean"][pick],
        "of_mean_phi_range": stats["tol_range"][pick],
        "of_mean_phi_monotone": stats["tol_of_mean_monotone"][pick],
        "per_code_mean": stats["tol_mean"][pick],
        "per_code_median": stats["tol_median"][pick],
        "per_code_stderr": stats["tol_stderr"][pick],
        "codes": stats["tol_codes"],
    }


def _probe_column(
    result: mipt.EnsembleResult, probe: mipt.Probe, layer: int | None, *, exactly: bool = False
) -> int | None:
    """The evaluation of ``probe`` at ``layer`` (``None``: its last, unless ``exactly``), or
    None if there is none."""
    own = np.asarray(result.layers(probe.name))
    if layer is None and not exactly:
        return int(own.size - 1) if own.size else None
    hit = np.flatnonzero(own == layer)
    return int(hit[0]) if hit.size else None


def _subset_note(name: str, stats: dict[str, Any], holding: int) -> str | None:
    """The note on codes left out to bound the work, if any were."""
    used, tolerance = stats["phi_codes"], stats["tol_codes"]
    if used < holding:
        return (
            f"{name}: to bound the work, Phi, the tolerances and their standard errors use "
            f"{used} of the {holding} codes that hold a code (evenly spaced); the per-code "
            f"e_delta uses {tolerance} of them"
        )
    if tolerance < used:
        return (
            f"{name}: the per-code e_delta uses {tolerance} of the {used} codes (evenly spaced) "
            "to bound the work; Phi and of_mean_phi use all of them"
        )
    return None


def _spaced(rows: np.ndarray, used: int) -> np.ndarray:
    """At most ``used`` of the ``rows``, evenly spaced (ends kept)."""
    return rows if rows.size <= used else rows[_subsample(rows.size, used)]


def _uniform_curves(
    result: mipt.EnsembleResult,
    probe: mipt.Probe,
    column: int,
    es: list[float],
    delta: float,
    notes: list[str],
) -> dict[str, Any]:
    """The independent-erasure channel of an ``erasure_first_failure`` probe at one layer.

    The curve of each code is the hybrid P_C(m): exact up to w where an
    ``uncorrectable_region_counts`` probe was evaluated at the same layer, sampled above.
    """
    n = result.spec.num_system
    layer = int(result.layers(probe.name)[column])
    first = np.asarray(result.values[probe.name][:, column, :])
    code = first.min(axis=1) <= n  # some ordering fails by n qubits: k > 0
    counts: np.ndarray | None = None
    exact_info = None
    found = [
        (q, col)
        for q in result.spec.probes
        if q.kind == "uncorrectable_counts"
        and (col := _probe_column(result, q, layer, exactly=True)) is not None
    ]
    if found:  # the largest exhaustive size w evaluated at this layer
        best, col = max(found, key=lambda t: np.asarray(result.values[t[0].name]).shape[-1])
        counts = np.asarray(result.values[best.name][:, col, :])
        exact_info = {"probe": best.name, "max_size": int(counts.shape[-1])}
    # the rows that reached the layer (NaN: stopped earlier by stop_when_distance)
    total = int(np.count_nonzero(~np.isnan(first[:, 0]))) if first.shape[1] else 0
    k = int(code.sum())
    entry: dict[str, Any] = {
        "layer": layer,
        "count": total,
        "count_with_code": k,
        "q_plus": k / total if total else None,
        "exact_counts": exact_info,
        "what": (
            "independent erasure, over the trajectories that hold a code (k > 0): Phi(e) = "
            "sum_m C(n, m) e^m (1 - e)^(n - m) P_C(m), P_C(m) from the first-failure samples "
            "(exact for m <= max_size where counts are given)"
        ),
        "phi": None,
        "e_delta": None,
    }
    if k == 0:
        return entry
    used, tolerance = _curve_plan(n, 1, k)
    rows = _spaced(np.flatnonzero(code), used)
    step = _chunk_rows(n, 1)

    def chunks() -> Iterator[np.ndarray]:
        for a in range(0, rows.size, step):
            sel = rows[a : a + step]
            curve = analysis.hybrid_failure_by_size(
                first[sel], n, exact_counts=None if counts is None else counts[sel]
            )
            yield curve[:, None, :]

    stats = _curve_statistics(chunks(), n, es, delta, rows.size, tolerance)
    entry["phi"] = {
        "mean": stats["phi_mean"][0],
        "stderr": stats["phi_stderr"][0],
        "codes": stats["phi_codes"],
    }
    entry["e_delta"] = {
        key: (v[0] if isinstance(v, np.ndarray) else v)
        for key, v in _tolerance_entry(stats, slice(None)).items()
    }
    note = _subset_note(probe.name, stats, k)
    if note:
        notes.append(note)
    return entry


def _correlated_curves(
    result: mipt.EnsembleResult,
    probe: mipt.Probe,
    column: int,
    es: list[float],
    delta: float,
    notes: list[str],
) -> dict[str, Any]:
    """The correlated channel of a ``correlated_erasure`` probe at one layer, per coupling.

    At most :data:`MAX_CURVE_COUPLINGS` couplings are computed: all, in the probe's order,
    or else evenly spaced in increasing order of J, the smallest and the largest included.
    """
    n = result.spec.num_system
    layer = int(result.layers(probe.name)[column])
    couplings, sizes, values = result.correlated(probe.name)
    sampled = np.asarray(values[:, column])  # (T, c, s)
    # the rows that reached the layer (NaN: stopped earlier by stop_when_distance)
    reached = ~np.isnan(sampled.reshape(sampled.shape[0], -1)).any(axis=1)
    total = int(reached.sum())
    holders = _code_holders_at(result, np.array([layer]))
    code = reached if holders is None else holders[:, 0] & reached
    k = int(code.sum())
    unknown = np.ones(n + 1, dtype=bool)  # the sizes the probe did not sample
    unknown[sizes] = False
    unknown[[0, n]] = False  # P(0) = 0 and P(n) = 1[k > 0] are known
    interpolated = np.flatnonzero(unknown)
    mass = _binomial_mass(n, es, interpolated)
    shown = _spread_by_value(couplings, MAX_CURVE_COUPLINGS)
    sampler = result.spec.correlated_sampler(probe.name)
    entry: dict[str, Any] = {
        "layer": layer,
        "count": total,
        "count_with_code": k if holders is not None else None,
        "q_plus": (k / total if total else None) if holders is not None else None,
        "num_couplings": int(couplings.size),
        "couplings": couplings[shown],
        "sampled_sizes": _compress_ints(np.sort(sizes).tolist()),
        "num_interpolated": int(interpolated.size),
        "interpolated_sizes": _compress_ints(interpolated.tolist()) if interpolated.size else None,
        "interpolated_mass": mass,
        "complete": bool(mass.max(initial=0.0) <= INTERPOLATION_MASS_WARNING),
        "e_delta_complete": None,
        "binomial_window": list(analysis.binomial_window(n, es, 1e-6)),
        "sampler": _sampler_text(sampler),
        "what": (
            "correlated erasure, over the trajectories that hold a code (k > 0): the region of "
            "M ~ Bin(n, e) erased qubits is drawn with probability proportional to exp(-J b(A)); "
            "Phi_J(e) = sum_m C(n, m) e^m (1 - e)^(n - m) P_J(m), P_J(m) the sampled failure "
            "fraction, linearly interpolated between sizes the probe did not sample"
        ),
        "phi": None,
        "e_delta": None,
    }
    if holders is None:
        entry.pop("count_with_code")
        entry["what"] += "; here every trajectory counts, k = 0 as P = 0"
        notes.append(
            f"{probe.name}: no code_dimension or erasure_first_failure probe covers layer "
            f"{layer}, so trajectories without a code (k = 0) count as P = 0 in these curves "
            "(a trajectory whose stored fractions are all 0 is taken as one); add one evaluated "
            "at the same layers to average over the codes (k > 0) only"
        )
    if k == 0:
        return entry
    used, tolerance = _curve_plan(n, int(shown.size), k)
    rows = _spaced(np.flatnonzero(code), used)
    step = _chunk_rows(n, int(shown.size))

    def chunks() -> Iterator[np.ndarray]:
        for a in range(0, rows.size, step):
            part = sampled[rows[a : a + step]][:, shown, :]
            # without a code probe a trajectory with all fractions 0 loses nothing: P(n) = 0
            holds = None if holders is not None else part.max(axis=(1, 2)) > 0
            yield analysis.correlated_failure_curve(part, sizes, n, holds_code=holds)[0]

    stats = _curve_statistics(chunks(), n, es, delta, rows.size, tolerance)
    entry["phi"] = {
        "mean": stats["phi_mean"],
        "stderr": stats["phi_stderr"],
        "codes": stats["phi_codes"],
    }
    # the tolerance searches every e, not only the e asked for: the binomial mass of the
    # interpolated sizes at e_delta itself (of_mean_phi, then the ends of its range)
    tolerances = np.column_stack([stats["tol_of_mean"], stats["tol_range"]])  # (couplings, 3)
    at_tolerance = np.stack([_binomial_mass(n, row.tolist(), interpolated) for row in tolerances])
    entry["e_delta"] = _tolerance_entry(stats, slice(None)) | {
        "monotone_fraction": stats["monotone"],
        "interpolated_mass_at_e_delta": at_tolerance,
    }
    entry["e_delta_complete"] = bool(np.all(at_tolerance <= INTERPOLATION_MASS_WARNING))
    note = _subset_note(probe.name, stats, k)
    if note:
        notes.append(note)
    if interpolated.size:
        notes.append(
            f"{probe.name}: sizes {entry['interpolated_sizes']} were not sampled and are "
            "linearly interpolated; interpolated_mass is the binomial probability they carry "
            "at each e"
            + (
                ""
                if entry["complete"]
                else f" (above {100 * INTERPOLATION_MASS_WARNING:g}% somewhere: Phi there "
                "leans on the interpolation; sample more sizes)"
            )
        )
    if not entry["e_delta_complete"]:
        low, high = analysis.binomial_window(
            n, np.unique(tolerances[np.isfinite(tolerances)]), 1e-6
        )
        notes.append(
            f"{probe.name}: e_delta_complete is false: the interpolated sizes carry up to "
            f"{100 * float(np.nanmax(at_tolerance)):.2g}% of the binomial probability at "
            "e_delta itself (interpolated_mass_at_e_delta: at of_mean_phi, then at both ends "
            "of of_mean_phi_range), so the tolerance leans on the interpolation even where "
            f"complete is {str(entry['complete']).lower()} (that is at the e asked for only); "
            f"sample the sizes {max(low, 1)} to {min(high, n - 1)} that e_delta needs"
        )
    if sampler["method"] != "exact":
        notes.append(
            f"{probe.name}: its regions come from a swap chain ({entry['sampler']}) whose "
            "mixing is not checked here: check it with sample_erasure_regions on the same "
            "graph, size, couplings and settings"
        )
    return entry


def erasure_curves(
    path: str,
    e: float | list[float] | None = None,
    delta: float = 0.01,
    layer: int | None = None,
) -> dict[str, Any]:
    """Failure probability and tolerance of the erasure channels of a saved run, per coupling.

    ``path`` is an output directory (chunks included) or a ``result.npz``. For the run's
    ``erasure_first_failure`` probes (the independent channel, with the exact
    ``uncorrectable_region_counts`` for ``m <= w`` where one was evaluated at the same layer)
    and ``correlated_erasure`` probes (one curve per coupling), at ``layer`` (default: each
    probe's last evaluation), over the trajectories that hold a code (k > 0): ``q_plus``
    (the fraction of trajectories that do) and ``count_with_code``, out of ``count``, the
    trajectories that reached the layer (a ``stop_when_distance`` stop ends some earlier);
    ``phi``, ``Phi(e)`` with
    its standard error at each ``e`` (default eight values log-spaced from 1/n to 1/2); and
    ``e_delta``, the tolerance ``sup{e : Phi(e) <= delta}``. ``Phi(e)`` erases each qubit
    independently with probability ``e`` only for the ``uniform`` rows and for ``J = 0``; for a
    correlated row with ``J != 0`` it erases ``M ~ Bin(n, e)`` qubits as one region drawn from
    ``Q_J``.

    * ``of_mean_phi`` is the headline: the tolerance of the ensemble-mean ``Phi`` (what
      ``analysis.erasure_tolerance`` returns for the whole ensemble), with
      ``of_mean_phi_range`` its value when the mean curve moves by plus and minus its standard
      error at every size (the tolerance of the higher curve, then of the lower: a rough
      one-standard-error band, since errors at different sizes are positively correlated), and
      ``of_mean_phi_monotone``, false where the mean curve steps down in m (``Phi`` may then
      cross ``delta`` more than once, and ``of_mean_phi`` is the last crossing);
    * ``per_code_mean``, ``per_code_median`` and ``per_code_stderr`` summarize the tolerance
      of each code's own curve: they describe the spread across codes and are biased upward
      for noisy curves, because the tolerance is not linear in ``Phi``.

    For a correlated channel, ``interpolated_sizes`` are the sizes the probe did not sample,
    whose curve is interpolated, ``interpolated_mass`` the binomial probability they carry at
    each ``e`` given, ``complete`` whether that stays below 0.1% at those ``e`` and
    ``binomial_window`` the sizes (from, to) that carry all but a millionth of the probability
    at the ``e`` given (the sizes worth sampling). The tolerance searches every ``e``, not
    only those given: ``e_delta.interpolated_mass_at_e_delta`` is, per coupling, the mass at
    ``of_mean_phi`` and at the two ends of ``of_mean_phi_range`` (in that order), and
    ``e_delta_complete`` is false, with a note, when any of them exceeds 0.1%: the tolerance
    then leans on the interpolation even where ``complete`` is true (it is null when no
    trajectory holds a code, since there is no tolerance). (The ``uniform`` rows
    have none of these fields: their curve is sampled or exact at every size.) A curve that
    is exact for small sizes at every coupling needs the
    saved codes (``StabilizerCode.boundary_failure_counts``). The standard errors treat
    trajectories as independent; circuits on one graph are not (a note says so for a graph
    collection).

    The work is bounded, not the codes alone: the curves of at most :data:`MAX_CURVE_VALUES`
    values (codes x couplings x (n + 1)) and the per-code tolerances of at most
    :data:`MAX_TOLERANCE_WORK` multiply-adds (and :data:`MAX_TOLERANCE_CODES` codes) are
    computed, with evenly spaced codes beyond that and a note; about 10 seconds at the most
    on the machine measured (``elapsed_seconds`` says). The output is bounded too: at most
    :data:`MAX_CURVE_COUPLINGS` couplings are shown (all of them in the probe's order, or
    evenly spaced in increasing order of J with the smallest and the largest included;
    ``truncated`` is then set). Writes nothing.
    """
    start = time.perf_counter()
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError("path must be a string")
    given = _curve_erasures(e)
    level = _require_number(delta, "delta", 0.0, 1.0)
    at = None if layer is None else _require_int(layer, "layer", minimum=0)
    result, where = _read_saved_result(path)
    n = result.spec.num_system
    probes = result.spec.probes
    firsts = [p for p in probes if p.kind == "erasure_first_failure"]
    correlated = [p for p in probes if p.kind == "correlated_erasure"]
    if not firsts and not correlated:
        raise ValueError(
            f"the run in {str(where)!r} has no erasure_first_failure or correlated_erasure "
            f"probe (its probes: {[p.name for p in probes]}): add one to the spec and rerun, "
            'for example {"kind": "erasure_first_failure", "samples": 100} or {"kind": '
            '"correlated_erasure", "couplings": [0, 1, 2]}; uncorrectable_region_counts alone '
            "covers only the sizes up to its w"
        )
    es = given if given is not None else _default_erasures(n)
    columns = {p.name: _probe_column(result, p, at) for p in [*firsts, *correlated]}
    if all(col is None for col in columns.values()):
        evaluated = "; ".join(
            f"{p.name} was evaluated at {_layers_text(np.asarray(result.layers(p.name)))}"
            for p in [*firsts, *correlated]
        )
        raise ValueError(f"layer {at} was not evaluated by any erasure probe: {evaluated}")
    notes: list[str] = []
    uniform = {
        p.name: _uniform_curves(result, p, col, es, level, notes)
        for p in firsts
        if (col := columns[p.name]) is not None
    }
    curves = {
        p.name: _correlated_curves(result, p, col, es, level, notes)
        for p in correlated
        if (col := columns[p.name]) is not None
    }
    skipped = [name for name, col in columns.items() if col is None]
    if skipped:
        notes.append(f"layer {at} was not evaluated by {skipped}: they are left out")
    for name, entry in {**uniform, **curves}.items():
        if entry.get("count_with_code") == 0:
            notes.append(
                f"{name}: no trajectory holds a code (k > 0) at layer {entry['layer']}, so "
                "there is no curve to average"
            )
    notes.append(
        "e_delta.of_mean_phi is the tolerance of the ensemble-mean Phi and the headline "
        "(of_mean_phi_range: its tolerance when the mean curve moves by one standard error); "
        "per_code_* summarize each code's own tolerance, which spreads across codes and is "
        "biased upward for noisy curves (the tolerance is not linear in Phi)"
    )
    graph_of = np.asarray(result.graph_index)
    if isinstance(result.spec.edges, tuple) and np.unique(graph_of).size < graph_of.size:
        notes.append(
            f"the {result.num_trajectories} trajectories ran on {np.unique(graph_of).size} "
            "graphs: circuits on one graph are correlated, but the standard errors treat every "
            "trajectory as independent and can be too small (by about 15% in a test with "
            "strong disorder); average per graph first for errors that allow for it"
        )
    out: dict[str, Any] = {
        "path": str(where),
        "num_trajectories": result.num_trajectories,
        "num_system": n,
        "layer": at,
        "delta": level,
        "e": es,
    }
    if uniform:
        out["uniform"] = uniform
    if curves:
        out["correlated"] = curves
    out["notes"] = notes
    out["elapsed_seconds"] = time.perf_counter() - start
    return _bounded(_fit_curves(out))


def _fit_curves(out: dict[str, Any]) -> dict[str, Any]:
    """Show fewer couplings of the correlated entries until the result fits the output limit.

    ``truncated`` is set (with a note) when a run has more couplings than are shown.
    """
    entries = out.get("correlated")
    if not entries:
        return out
    cut_note = (
        "only some couplings are shown (evenly spaced in increasing order of J, the smallest "
        "and the largest included) to fit the size limit; "
        "result.npz holds every value (EnsembleResult.correlated(), and "
        "analysis.correlated_failure_curve computes any curve)"
    )
    candidate = out
    for keep in (MAX_CURVE_COUPLINGS, 8, 6, 4, 3, 2, 1):
        shown: dict[str, Any] = {}
        for name, entry in entries.items():
            pick = _spread_by_value(np.asarray(entry["couplings"]), keep)
            row = dict(entry)
            row["couplings"] = entry["couplings"][pick]
            if entry["phi"] is not None:
                for field in ("phi", "e_delta"):
                    row[field] = {
                        key: (v[pick] if isinstance(v, np.ndarray) else v)
                        for key, v in entry[field].items()
                    }
            shown[name] = row
        candidate = {**out, "correlated": shown}
        if any(len(row["couplings"]) < row["num_couplings"] for row in shown.values()):
            candidate["truncated"] = True
            candidate["notes"] = [*out["notes"], cut_note]
        if len(json.dumps(_clean(candidate))) <= MAX_OUTPUT_CHARS:
            return candidate
    return candidate


# ---------------------------------------------------------------------------
# self-description
# ---------------------------------------------------------------------------


def _signature_summary(fn: Callable, skip: Sequence[str] = ()) -> dict[str, Any]:
    out = {}
    for name, p in _params(fn).items():
        if name in skip:
            continue
        out[name] = "required" if p.default is inspect.Parameter.empty else _jsonable(p.default)
    return out


_DETAIL_SECTIONS = ("probes", "observables", "instructions", "ensemble_spec", "graph_families")

# What a probe's signature does not show: defaults chosen inside the function, and the
# argument forms beyond plain numbers.
_PROBE_EXTRAS: dict[str, dict[str, Any]] = {
    "contiguous_distance": {
        "forms": [{"cap": 2}, {"cap": 3, "name": "d3"}],
        "note": (
            "cap c (at least 2) records min(d, c): exactly d below c and c for every distance at "
            "or above it; cap 2 asks whether no single qubit carries logical information. A cap "
            "with (c - 1)**2 <= 2 num_system checks only the runs shorter than c, about n c**2 "
            "row operations, and skips the full distance's gauge eliminations, whose cost grows "
            "as n**3 and dominates a trajectory at large n; a larger cap takes the full "
            "distance and caps it. The key is contiguous_distance, or "
            "contiguous_distance[cap=c] with a cap, unless 'name' is given"
        ),
    },
    "logical_sites": {
        "note": (
            "N_log = #{i : I(i:R) > 0}: the number of system qubits carrying a weight-one logical "
            "operator (not the sum of I(i:R), which is 2 at a qubit still paired with its "
            "reference). d_cont >= 2 exactly when k > 0 and N_log = 0. One pass over the qubits, "
            "O(n W) words, as cap-2 contiguous_distance on a code without such a qubit; 0 where "
            "k = 0. The summary adds mean_with_code and distance_two_fraction when a "
            "code_dimension probe covers its layers"
        ),
    },
    "site_information": {
        "note": (
            "a vector of n values per evaluation, I(i:R) in {0, 1, 2} for qubit i (label "
            "order); > 0 is the sitewise defect bitset. Same cost as logical_sites plus n "
            "stored numbers: give it its own few checkpoints"
        ),
    },
    "uncorrectable_region_counts": {
        "params": {"max_size": 2},
        "forms": [
            {"max_size": 3},
            {"budget": 50000},
            {"max_size": {"64": 3, "128": 2}},
            {"budget": {"64": 20000, "256": 50000}},
        ],
        "note": (
            "w = m*(n), the largest exhaustive size: give max_size (w itself, default 2) or "
            "budget B (w = max{w : sum_{m=1}^{w} C(n, m) <= B}; B counts regions, not "
            "seconds), not both; an object maps num_system to a value, so one probe serves a "
            "sweep over n; the key is uncorrectable_region_counts[w] unless 'name' is given"
        ),
    },
    "correlated_erasure": {
        "forms": [
            {"couplings": [0, 1, 2], "sizes": "4:29:4", "samples": 100},
            {"couplings": [0, 2], "sizes": [8, 16], "geometry": "cycle", "checkpoints": "final"},
        ],
        "note": (
            "P_{C,G,J}(m): fraction of m-qubit erasure regions, drawn with probability "
            "proportional to exp(-J b_G(A)) (b_G: edges of the erasure graph leaving A), that "
            "lose logical information; needs a reference; k = 0 gives 0. Values are "
            "coupling-major, len(couplings) x len(sizes) per evaluation "
            "(EnsembleResult.correlated() reshapes them). couplings: distinct, |J| <= 1e6 (J -> "
            "inf is a limit, not a coupling: StabilizerCode.interval_failure_by_size gives the "
            "ring's); sizes: a list, a range 'a:b[:step]' or null for every m in 1..n-1 (the "
            "forms need n >= 29 and n >= 17). geometry: the trajectory's interaction graph "
            "(default), or 'cycle' or 'path' over the qubit labels. method: auto (exact "
            "samples on a cycle or path, else the chain), chain or exact. The chain: a swap "
            "chain per rung of a replica-exchange ladder (tempering: exchanges between rungs); "
            "ladder auto (the couplings, 0 and rungs at most max_step apart, default "
            "1/sqrt(Var_0 b)), given (the couplings only) or a list of extra rungs; burn_in "
            "sweeps before the first record (default max(200, 8 n), per size and evaluation), "
            "thin sweeps between records; as in sample_erasure_regions, which checks the "
            "mixing. Cost per evaluation and size about rungs x (burn_in + samples x thin) x n "
            "proposals plus couplings x samples region evaluations: keep sizes narrow "
            "(analysis.binomial_window), give it checkpoints of its own, see "
            "estimate_ensemble. Memory: 8 x couplings x samples x (largest size + 1) bytes in "
            "each running thread"
        ),
    },
}


def _capability_sections() -> dict[str, Any]:
    fields = _spec_fields()
    spec_keys = {
        k: ("required" if d is ... else None if isinstance(d, np.ndarray) else _jsonable(d))
        for k, d in fields.items()
    }
    spec_keys["graph"] = "required unless edges is given"
    spec_keys["edges"] = "list of [a, b] pairs (alternative to graph)"
    return {
        "package": "sparsegf2",
        "version": __version__,
        "description": (
            "Fast stabilizer (Clifford) simulator for monitored random circuits: "
            "measurement-induced transitions, purification, dynamically generated codes, "
            "erasure protection. Stim circuit language; results verified against Stim."
        ),
        "functions": [
            "capabilities",
            "inspect_circuit",
            "run_circuit",
            "run_ensemble",
            "estimate_ensemble",
            "benchmark_ensemble",
            "start_ensemble_job",
            "job_status",
            "cancel_job",
            "list_jobs",
            "load_results",
            "sample_erasure_regions",
            "erasure_curves",
        ],
        "instructions": {
            kind: [g.name for g in all_gates(kind)] for kind in KINDS if all_gates(kind)
        },
        "aliases": "Stim aliases are accepted (CNOT, ZCX, MZ, RZ, SQRT_Z, ...)",
        "observables": {
            kind: {
                "args": list(o.required),
                **({"optional": list(o.optional)} if o.optional else {}),
                "description": o.description,
                **({"needs_phases": True} if o.needs_phases else {}),
            }
            for kind, o in _OBSERVABLES.items()
        },
        "ensemble_spec": spec_keys,
        "probes": {
            name: {
                "params": {
                    **_signature_summary(fn, ("name", "checkpoints")),
                    **_PROBE_EXTRAS.get(name, {}).get("params", {}),
                    "name": None,
                },
                "description": _first_sentence(fn),
                **{k: v for k, v in _PROBE_EXTRAS.get(name, {}).items() if k != "params"},
            }
            for name, fn in _probe_kinds().items()
        },
        "graph_families": {
            **{name: _signature_summary(fn, skip=("n",)) for name, fn in _graph_families().items()},
            "file": (
                'a .json or .npz graph collection: {"family": "file", "path": ...} '
                "(any geometry; see graphs.save)"
            ),
        },
        "edge_weight_families": {
            name: _signature_summary(fn, skip=("edges", "n"))
            for name, fn in _weight_families().items()
        },
        "schedules": list(SCHEDULES),
        "references": list(REFERENCES),
        "limits": {
            "max_output_chars": MAX_OUTPUT_CHARS,
            "sync_ensemble_max_seconds": SYNC_ENSEMBLE_MAX_SECONDS,
            "max_qubits_run_circuit": MAX_QUBITS,
            "max_num_system": MAX_QUBITS,
            "max_shot_results": MAX_SHOT_RESULTS,
            "checkpoints_shown": MAX_CHECKPOINTS_SHOWN,
        },
        "conventions": [
            "qubits 0..num_system-1 are the system, references follow (n..n+R-1)",
            "qubit lists: JSON list of ints or a range string 'a:b[:step]'",
            "seed=None draws a fresh seed, reported back as 'seed'",
            "phases=True (default) tracks signs; ensembles always run phase-free",
            "entropies are in bits; purification layer -1 means censored",
            "every probe and observable takes an optional 'name': its key in the results",
            "every probe takes optional 'checkpoints' (its own layers; default: the spec's)",
            "stop_when_distance: c (>= 2) with stop_not_before: H ends a trajectory at the first "
            "layer >= H by which its contiguous distance reached c (needs a reference and one "
            "contiguous_distance probe at every layer, cap none or >= c); later evaluations are "
            "NaN and the purification time is censored at the stop",
            "logical_sites: N_log = #{i : I(i:R) > 0}, the qubits with a weight-one logical "
            "operator (O(n W) per evaluation, as cap-2 contiguous_distance without a defect); "
            "site_information: I(i:R) per qubit (the defect bitsets: > 0); both need a reference",
            "record_first_touch: true saves first_touch (trajectories x n, int64): the first "
            "layer each system qubit was in a gate or measured (scramble: 0; -1: untouched "
            "through final_layer); the summary's first_touch block gives N_bare(T)",
        ],
        "examples": {
            "run_circuit": {
                "circuit": "H 0\nCX 0 1\nM 0 1",
                "shots": 1000,
                "seed": 1,
                "observables": [{"kind": "entropy", "region": [0]}],
            },
            "run_ensemble": {
                "spec": {
                    "num_system": 32,
                    "graph": {"family": "cycle"},
                    "layers": 128,
                    "measurement_rate": 0.1,
                    "reference": "single",
                    "checkpoints": 8,
                    "probes": [{"kind": "code_dimension"}],
                    "stop_when_purified": True,
                },
                "trajectories": 200,
                "seed": 1,
            },
            "sample_erasure_regions": {
                "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": 1},
                "num_nodes": 64,
                "size": 16,
                "couplings": [0, 0.5, 1, 2],
                "samples": 200,
                "seed": 7,
            },
            "erasure_curves": {
                "path": "runs/erasure",
                "e": [0.02, 0.05, 0.1],
                "delta": 0.01,
            },
        },
        "docs": "quickstart() / MCP resource docs://sparsegf2/quickstart",
    }


def capabilities(section: str | None = None) -> dict[str, Any]:
    """What the simulator supports: instructions, observables, probes, graph families, limits, examples.

    With ``section`` (e.g. ``"probes"``), only that part is returned, in full.
    If the complete overview would exceed the output limit, the largest
    detail sections are replaced by their names and a pointer to
    ``capabilities(section=...)``.
    """
    sections = _capability_sections()
    if section is not None:
        if _require_name(section, "section", sections) not in sections:
            raise ValueError(f"unknown section {section!r}; choose from {sorted(sections)}")
        return _bounded({section: sections[section]})
    out = _jsonable(sections)
    for name in _DETAIL_SECTIONS:
        if len(json.dumps(out)) <= MAX_OUTPUT_CHARS:
            break
        value = out[name]
        out[name] = {
            "names": list(value) if isinstance(value, dict) else value,
            "details": f"capabilities(section={name!r})",
        }
    return _bounded(out)


_QUICKSTART = """\
# SparseGF2 quickstart for agents

SparseGF2 is a fast stabilizer-circuit simulator for monitored random Clifford
circuits. Every call takes JSON and returns a compact JSON summary.

## 1. Single circuits (Stim language)

run_circuit(circuit="H 0\\nCX 0 1\\nM 0 1", shots=1000, seed=1)
  -> measurement_means, random_fraction, detector rates, ...

Observables of the final state (entropy, mutual_information,
tripartite_information, code_dimension, lost_logical_information,
random_erasure_statistics, erasure_profile, contiguous_code_distance,
stabilizer_weights, stabilizers, expectation, ...):

run_circuit(circuit=..., observables=[{"kind": "entropy", "region": "0:4"}],
            observe_ticks=true)   # also after every TICK: dynamics vs depth

CLIFFORD_2Q(i) a b applies the two-qubit Clifford with canonical index i in
[0, 11520) (a uniformly random i is a uniformly random Clifford). Use
inspect_circuit first to validate text: errors come back with line numbers.

## 2. Ensembles of random monitored circuits

run_ensemble(spec={
    "num_system": 64, "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": 1},
    "layers": 512, "measurement_rate": 0.1, "schedule": "random_edges",
    "reference": "single", "reference_partner": "random", "scramble_gates": 128,
    "checkpoints": "every_layer", "probes": [{"kind": "code_dimension"}],
    "stop_when_purified": true}, trajectories=1000, seed=7)
  -> purification: purified, censored_fraction, tau_median_purified, survival curve
  -> probes: mean and stderr per checkpoint (at most 50 shown)

Pictures: reference="purification" (system maximally mixed with n references),
"single" (one reference qubit Bell-paired with a system qubit), "none".
Probes: <PROBES>.
Erasure probes need a reference; any probe may take its own "checkpoints". A
"layer" is one round of gates followed by measuring each system qubit with
probability measurement_rate.

Graph disorder: "graph": {..., "seed": 1, "realizations": G} plus
"circuits_per_graph": c runs c circuits on each of G graphs (trajectory t on
graph t // c); omit trajectories to run all G * c.

Geometry-correlated erasures: the probe correlated_erasure measures P_{C,G,J}(m), the
failure fraction of m-qubit regions drawn with weight exp(-J b_G(A)) (b_G: edges of the
graph leaving A). Before a large run check the sampler on the graph with
sample_erasure_regions(graph, size, couplings) (no circuit; mixing notes, exact checks);
after it, erasure_curves(output_dir) gives Phi(e) and the tolerance e_delta per coupling.

## 3. Long runs

estimate_ensemble(spec, trajectories, threads=[1, 32]) gives a rough runtime
(per thread count), the result's memory and the layout of the saved arrays.
benchmark_ensemble(spec, trajectories, threads=[1, 32]) measures the run's
first trajectories on this machine and the size of result.npz; it writes
nothing. Above ~60 s use
start_ensemble_job(spec, output_dir=..., trajectories=...), then
job_status(job_id, wait_seconds=30) until "completed", then
load_results(output_dir). Jobs save every chunk; call start_ensemble_job
again with the same output_dir to resume or extend.

## Conventions and pitfalls

* Seeds: give an int to reproduce; None draws one and reports it. Graph seeds
  are separate from circuit seeds (graph disorder vs circuit disorder).
* Qubit lists: [0, 1, 2] or "0:3". System qubits are 0..n-1, references n...
* Phase-free mode (phases=false) has no measurement outcomes: only which
  measurements were random. Ensembles always run phase-free (entanglement
  does not depend on outcomes).
* tau = -1 means the reference never purified within the horizon (censored);
  never average it as a number. Use censored_fraction and the survival curve.
* Outputs are summaries. Pass output_dir to keep every array (npz) plus
  spec.json and provenance.json.
* Errors say what to change: fix the argument and retry. A message ending
  "(internal error: please report it)" is a bug in SparseGF2: report it, do
  not retry with other arguments.
"""


def quickstart() -> str:
    """A concise usage guide for agents (Markdown); also served as an MCP resource."""
    probes = textwrap.fill(", ".join(_probe_kinds()), width=78, initial_indent=" " * 8)
    return _QUICKSTART.replace("<PROBES>", probes.strip())
