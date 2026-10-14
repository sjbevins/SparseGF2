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
    A rough runtime estimate for :func:`run_ensemble`.
:func:`start_ensemble_job`, :func:`job_status`, :func:`cancel_job`, :func:`list_jobs`
    The same ensembles as resumable background jobs, saved chunk by chunk.
:func:`load_results`
    Summarize a saved ensemble (a ``.npz`` file or an output directory).

Conventions shared by every function: seeds are non-negative ints (``None``
draws a fresh one, which is reported back so the run can be repeated); qubit
lists are JSON lists of ints or a range string ``"start:stop[:step]"`` with
Python slice semantics (``"0:8"`` is qubits 0..7).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import json
import math
import os
import platform
import re
import secrets
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import uuid
import zipfile
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2 import __version__, analysis, graphs, mipt, observables
from sparsegf2._compile import DEFAULT_MAX_OPS
from sparsegf2.circuit import Circuit, Instruction, RepeatBlock, _strip_comment
from sparsegf2.errors import PhaseFreeError
from sparsegf2.gates import KINDS, all_gates, gate
from sparsegf2.sampling import sample
from sparsegf2.simulator import Simulator

__all__ = [
    "MAX_OUTPUT_CHARS",
    "SYNC_ENSEMBLE_MAX_SECONDS",
    "cancel_job",
    "capabilities",
    "estimate_ensemble",
    "inspect_circuit",
    "job_status",
    "list_jobs",
    "load_results",
    "quickstart",
    "run_circuit",
    "run_ensemble",
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

MAX_CHECKPOINTS_SHOWN = 50
MAX_BITS_SHOWN = 256
MAX_LIST_SHOWN = 64
MAX_PROFILE_SHOWN = 65
MAX_TEXT_SHOWN = 3000
MAX_ERRORS_SHOWN = 20
MAX_STABILIZER_QUBITS = 32
MAX_NAME_CHARS = 64
MAX_VECTOR_DEPTHS = 8
MAX_OBSERVED_TICKS = 1_000_000
MAX_TICK_ROWS = 10_000
MAX_QUBITS = 65_536
MAX_SHOT_RESULTS = 200_000_000
MAX_TRAJECTORIES = 100_000_000
MAX_INT = 2**62
"""Largest integer accepted for counts, sizes and indices (seeds excepted): nothing larger fits."""

SCHEDULES = ("matching", "random_edges", "all_edges")
REFERENCES = ("none", "purification", "single")

# Compiled kernels run in parallel over all cores; serialise them so that a
# background job and an interactive call never run two parallel regions at
# once (Numba's threading layers are not all re-entrant).
_COMPUTE_LOCK = threading.Lock()

_RESULT_FILE = "result.npz"
_CHUNK_DIR = "chunks"
_SPEC_FILE = "spec.json"
_PROVENANCE_FILE = "provenance.json"


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
    notes = out.get("notes")
    out["notes"] = [*(notes[:2] if isinstance(notes, list) else []), note]
    out["truncated"] = True
    return out


def _bounded(out: dict[str, Any], limit: int = MAX_OUTPUT_CHARS) -> dict[str, Any]:
    """Make ``out`` JSON-clean and, if still too long, cut long lists, strings and keys.

    The summaries below are designed to fit already; this is the safety net.
    When it acts, ``truncated`` is set and a note says how to get everything.
    As a last resort only the top-level scalar fields (seed, counts, ...) and
    ``files`` are kept, so a caller never loses the run's identity or paths.
    """
    out = _jsonable(out)
    if len(json.dumps(out)) <= limit:
        return out
    note = (
        "output cut to fit the size limit; pass output_dir (or read the saved files) for all data"
    )
    for max_items in (32, 16, 8, 4, 2, 1, 0):
        shrunk = _mark_truncated(_shrink(out, max_items, max(200, 80 * max_items)), note)
        if len(json.dumps(shrunk)) <= limit:
            return shrunk
    essentials = {
        k: v
        for k, v in out.items()
        if not isinstance(v, (dict, list)) and not (isinstance(v, str) and len(v) > 200)
    }
    if isinstance(out.get("files"), dict):
        essentials["files"] = out["files"]
    essentials = _mark_truncated(
        essentials, "result too large to show; only scalar fields and file paths are kept"
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
    """A concrete non-negative int seed; ``None`` draws a fresh 63-bit one."""
    if seed is None:
        return secrets.randbits(63)
    return _require_int(seed, "seed", minimum=0, maximum=None)


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


def _subsample(count: int, limit: int) -> np.ndarray:
    """At most ``limit`` evenly spaced indices into ``range(count)``, always keeping both ends."""
    if count <= limit:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, limit).round().astype(np.int64))


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
    names: Counter[str] = Counter()
    for i, spec in enumerate(specs):
        where = f"observables[{i}]"
        if not isinstance(spec, dict) or "kind" not in spec:
            raise ValueError(f'{where}: expected an object with a "kind", got {spec!r}')
        kind = spec["kind"]
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
        name = spec.get("name") or kind
        if not isinstance(name, str):
            raise ValueError(f"{where}.name must be a string")
        names[name] += 1
        if names[name] > 1:
            name = f"{name}#{names[name]}"

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
    the normalised circuit text (``text``, truncated if long). For an invalid
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
        most :data:`MAX_TICK_ROWS` evaluations.
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
    notes: list[str] = []
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
    needed = max(1, math.ceil(num_ticks / MAX_TICK_ROWS))
    if tick_stride is None:
        return needed
    stride = _require_int(tick_stride, "tick_stride", minimum=1)
    if stride < needed:
        raise ValueError(
            f"tick_stride={stride} would evaluate more than {MAX_TICK_ROWS:,} ticks of "
            f"{num_ticks:,}; use tick_stride >= {needed}"
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
    npz = out / "circuit_run.npz"
    _atomic_savez(npz, {k: v for k, v in arrays.items() if v is not None})
    stim_path = out / "circuit.stim"
    stim_path.write_text(str(circuit) + "\n", encoding="utf-8")
    files = {"arrays": str(npz), "circuit": str(stim_path)}
    if final_values or tick_rows:
        obs_path = out / "observables.json"
        obs_path.write_text(
            json.dumps(
                _jsonable(
                    {"final": final_values, "ticks": [{"tick": t, **v} for t, v in tick_rows]},
                    exact=True,
                ),
                indent=1,
            ),
            encoding="utf-8",
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


def _file_graph(graph: dict, n: int) -> tuple[np.ndarray | list[np.ndarray], Any]:
    """Edges (one array, or a list for a collection) and weights of a ``{"family": "file"}`` graph."""
    extra = sorted(set(graph) - {"family", "path"})
    if extra:
        raise ValueError(f"graph (file): unknown key(s) {extra}; give only 'path'")
    path = graph.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError('graph (file): give "path", a .json or .npz graph file')
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"graph (file): no such file {str(p)!r}")
    num_nodes, gs, ws = graphs.load(p)
    if num_nodes != n:
        raise ValueError(f"graph (file): {p.name} has num_nodes {num_nodes}, but num_system is {n}")
    if len(gs) == 1:
        return gs[0], (None if ws is None else ws[0])
    return gs, ws


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
        seed = params.pop("seed")
        base = [_require_int(seed, "graph.seed", minimum=0)] if not isinstance(seed, list) else seed
        bank: list[np.ndarray] = []
        for i in range(realizations):
            one = _build_graph({**params, "seed": [*base, i]}, n)
            assert isinstance(one, np.ndarray), "without realizations a family yields one graph"
            bank.append(one)
        return bank
    family = params.pop("family")
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
    return _call_with(fn, params, {"n": n}, f"graph ({family})")


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
    family = params.pop("family")
    if family not in families:
        raise ValueError(f"unknown edge_weights family {family!r}; choose from {sorted(families)}")
    return _call_with(
        families[family], params, {"edges": edges, "n": n}, f"edge_weights ({family})"
    )


def _build_probe(spec: Any, index: int) -> mipt.Probe:
    kinds = _probe_kinds()
    where = f"probes[{index}]"
    if not isinstance(spec, dict) or "kind" not in spec:
        raise ValueError(
            f'{where}: expected an object like {{"kind": "code_dimension"}}; kinds: {sorted(kinds)}'
        )
    params = dict(spec)
    kind = params.pop("kind")
    if kind not in kinds:
        raise ValueError(f"{where}: unknown probe kind {kind!r}; choose from {sorted(kinds)}")
    name = params.pop("name", None)
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError(f"{where}.name must be a non-empty string, got {name!r}")
    for key, value in list(params.items()):
        if key in _QUBIT_ARGS and value is not None:
            params[key] = _qubit_list(value, f"{where}.{key}")
    if params.get("checkpoints") is not None:
        params["checkpoints"] = _checkpoints(params["checkpoints"], f"{where}.checkpoints")
    probe = _call_with(kinds[kind], params, {"name": None}, f"{where} ({kind})")
    return dataclasses.replace(probe, name=name) if name is not None else probe


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
    n = _require_int(spec["num_system"], "num_system", minimum=2)
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
            weights = _weights_in_normalised_order(raw_edges, weights)
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
        kwargs["probes"] = [_build_probe(p, i) for i, p in enumerate(spec["probes"])]
    try:
        return mipt.MonitoredCircuit(**kwargs)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid spec: {exc}") from None


def _random_family(graph: Any) -> bool:
    family: Any = graph.get("family") if isinstance(graph, dict) else None
    fn = _graph_families().get(family) if isinstance(graph, dict) else None
    return fn is not None and "seed" in _params(fn)


def _weights_in_normalised_order(raw_edges: np.ndarray, weights: list) -> list:
    """Reorder per-edge weights given for ``raw_edges`` to match the normalised edge list.

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


def _entropy_us(sizes: Sequence[int], words: int) -> float:
    """Cost of entropies of regions with the given sizes on a ``words``-word tableau."""
    return sum(4.0 + 0.1 * size * words for size in sizes)


def _probe_us(p: mipt.Probe, n: int, words: int) -> float:
    """Microseconds (one thread) to evaluate one probe at one checkpoint, pessimistically."""
    r = [len(region) for region in p.regions]
    if p.kind == "code_dimension":
        return 1.0 + 0.06 * n
    if p.kind in ("entropy", "half_cut"):
        return _entropy_us(r, words)
    if p.kind == "lost_information":
        return _entropy_us([2 * size for size in r], words)
    if p.kind == "mutual_information":
        return _entropy_us([*r, sum(r)], words)
    if p.kind == "tripartite":
        a, b, c = r
        return _entropy_us([a, b, c, a + b, a + c, b + c, a + b + c], words)
    if p.kind == "contiguous_distance":
        return 50.0 + 0.03 * n**2.25
    if p.kind == "erasure_lost_profile":
        return p.samples * (0.8 * n + 0.05 * n * words)
    # The erasure kernels first row-reduce the code's columns (2n columns of at
    # most 2n bits); fitted to n = 256..2048 and rounded up.
    code_words = (2 * n + 63) // 64
    columns = 0.004 * n**2 * math.sqrt(code_words)
    if p.kind == "erasure_first_failure":
        return columns + p.samples * 0.03 * n**1.5
    if p.kind == "uncorrectable_counts":
        # C(n, w - 1) prefixes, each with up to 4**(w - 1) hash lookups; in logs,
        # so an absurd max_size gives a huge estimate rather than an overflow
        w = p.size
        log_prefixes = math.lgamma(n + 1) - math.lgamma(w) - math.lgamma(n - w + 2)
        log_us = log_prefixes + (w - 1) * math.log(4) + math.log(0.15 + 0.015 * code_words)
        return columns + math.exp(min(log_us, 700.0))
    if p.kind == "code_snapshot":
        return 1.0 + 0.01 * n**2
    return 10.0 + 0.5 * n + 0.05 * n * words  # entropy_profile, and any newer probe


def _trajectory_us(mc: mipt.MonitoredCircuit) -> float:
    """Pessimistic microseconds (one thread) for one trajectory of ``mc``.

    Fitted to measured runs (4-core x86-64, 2026, n = 32..512, both reference
    pictures) and rounded up; real runs were 1x to 3x faster. Trajectories
    that stop early (``stop_when_purified``) make real runs faster still.
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
    scramble_gates = (
        mc.scramble_gates + mc.scramble_matchings * (n // 2) + mc.scramble_layers * gates_per_layer
    )
    dynamics = mc.layers * layer + scramble_gates * gate_us
    probes = sum(len(mc.probe_layers(p.name)) * _probe_us(p, n, words) for p in mc.probes)
    detection = len(mc.checkpoint_layers) * (1.0 + 0.06 * n) if mc.reference != "none" else 0.0
    setup = 5.0 + 0.01 * mc.num_qubits * words
    return setup + dynamics + probes + detection


_CALL_OVERHEAD_SECONDS = 0.05


def _setup_us(spec: dict, mc: mipt.MonitoredCircuit) -> float:
    """Pessimistic one-thread microseconds to build ``mc`` from ``spec`` and preprocess its graphs.

    Graph generation, validation and (for the matching schedule) edge
    colouring run in Python, once per graph, so a large graph bank
    (``realizations``) costs seconds before the first trajectory starts.
    Fitted to measured runs (WS graphs, n = 32..128) and rounded up.
    """
    edges = sum(len(g) for g in mc.graphs)
    us = 50.0 * mc.num_graphs + 0.5 * edges  # normalising and validating every graph
    if "graph" in spec:
        us += 300.0 * mc.num_graphs + 1.0 * edges  # generating the graphs
    if mc.schedule == "matching":
        us += 2.5 * edges  # edge colouring
    return us


def _estimate_seconds(spec: dict, mc: mipt.MonitoredCircuit, trajectories: int) -> float:
    import numba

    # Trajectories are the unit of parallelism: fewer trajectories than threads
    # leave threads idle.
    parallel = min(numba.get_num_threads(), trajectories)
    kernel = trajectories * _trajectory_us(mc) * 1e-6 / parallel
    return _CALL_OVERHEAD_SECONDS + _setup_us(spec, mc) * 1e-6 + kernel


def _result_bytes(mc: mipt.MonitoredCircuit, trajectories: int) -> int:
    """Memory of the run's result: every probe value (float64) plus the saved codes."""
    n = mc.num_system
    values = sum(len(mc.probe_layers(p.name)) * mipt._probe_length(p, n) for p in mc.probes)
    codes = sum(
        len(mc.probe_layers(p.name)) * n * 2 * ((n + 63) // 64)
        for p in mc.probes
        if p.kind == "code_snapshot"
    )
    return trajectories * 8 * (values + codes + 4)


def _trajectory_count(mc: mipt.MonitoredCircuit, trajectories: Any) -> int:
    """The number of trajectories to run: as given, or every circuit of a graph collection."""
    if trajectories is None:
        if mc.num_trajectories is None:
            raise ValueError(
                "give trajectories (the number of random circuits); it may be omitted only for "
                "a collection of graphs (graph.realizations), which runs circuits_per_graph "
                "circuits on each graph"
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
            "GB): give the large probes (erasure_first_failure samples, code_snapshot, vector "
            "probes) fewer checkpoints of their own, or split the trajectories into several "
            "output directories"
        )
    return trajectories


def estimate_ensemble(spec: dict[str, Any], trajectories: int | None = None) -> dict[str, Any]:
    """A rough (deliberately pessimistic) runtime estimate for :func:`run_ensemble`.

    Returns ``estimated_seconds`` (building the spec, generating and
    preprocessing its graphs, and the trajectories), the thread count it
    assumes, whether a synchronous run fits within
    :data:`SYNC_ENSEMBLE_MAX_SECONDS`, the trajectory count (every circuit of a
    graph collection when ``trajectories`` is omitted) and the memory the
    result needs. Trajectories that stop early (``stop_when_purified``) make
    real runs faster.
    """
    import numba

    mc = _build_spec(spec)
    trajectories = _trajectory_count(mc, trajectories)
    seconds = _estimate_seconds(spec, mc, trajectories)
    return {
        "estimated_seconds": _num(seconds),
        "trajectories": trajectories,
        "result_megabytes": _num(_result_bytes(mc, trajectories) / 1e6),
        "threads": numba.get_num_threads(),
        "fits_sync_limit": seconds <= SYNC_ENSEMBLE_MAX_SECONDS,
        "sync_limit_seconds": SYNC_ENSEMBLE_MAX_SECONDS,
    }


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
    stats: Callable[[int], tuple[np.ndarray, np.ndarray]],
    alive: np.ndarray,
    layers: np.ndarray,
    sizes_shown: int,
    depths_shown: int,
) -> dict[str, Any]:
    """A vector probe: one curve over erasure or region size m = 0, 1, ... per evaluation.

    ``stats(c)`` returns the mean and standard error over trajectories of the
    curve at evaluation ``c``; ``alive`` (trajectories x evaluations) marks
    the values that exist (with ``stop_when_purified``, entanglement profiles
    of stopped trajectories do not). ``mean``/``stderr`` are at the last
    evaluation that has data, and ``vs_depth`` gives the mean on a coarse
    grid of evaluations x sizes, so the change with circuit depth stays
    visible. ``index`` lists the sizes m shown when not all are.
    """
    count = alive.sum(axis=0)
    with_data = np.flatnonzero(count > 0)
    if not with_data.size:
        return {"vector": True, "count": 0, "notes": ["no trajectory reached any checkpoint"]}
    last = int(with_data[-1])
    mean, err = stats(last)
    sizes = _subsample(mean.size, sizes_shown)
    entry: dict[str, Any] = {
        "vector": True,
        "at_layer": int(layers[last]),
        "count": int(count[last]),
        "mean": mean[sizes],
        "stderr": err[sizes],
    }
    if len(sizes) < mean.size:
        entry["index"] = sizes
    if len(with_data) > 1:
        depth_idx = with_data[_subsample(len(with_data), depths_shown)]
        grid_sizes = _subsample(mean.size, 17)
        entry["vs_depth"] = {
            "layers": layers[depth_idx],
            "count": count[depth_idx],
            "index": grid_sizes,
            "mean": np.array([stats(int(c))[0][grid_sizes] for c in depth_idx]),
        }
    return entry


def _vector_stats(
    probe: mipt.Probe, values: np.ndarray, n: int, cache: dict
) -> Callable[[int], tuple[np.ndarray, np.ndarray]]:
    """Mean and standard error of a vector probe's curve at one evaluation, memoized in ``cache``.

    Profiles are averaged as stored. First-failure samples become per-trajectory
    P_C(m), m = 0..n (fraction of samples with D <= m), in chunks of
    trajectories so the memory stays bounded whatever the ensemble size;
    exhaustive counts become the exact P_C(m), m = 0..w (P_C(0) = 0).
    """

    def compute(c: int) -> tuple[np.ndarray, np.ndarray]:
        if probe.kind == "uncorrectable_counts":
            exact = analysis.exact_failure_by_size(values[:, c, :], n)
            curves = np.concatenate([np.zeros((exact.shape[0], 1)), exact], axis=1)
            return analysis.mean_and_error(curves, axis=0)
        if probe.kind != "erasure_first_failure":
            return analysis.mean_and_error(values[:, c, :], axis=0)
        total = np.zeros(n + 1)
        squares = np.zeros(n + 1)
        chunk = max(1, (1 << 22) // (n + 2))
        for start in range(0, values.shape[0], chunk):
            curves = analysis.failure_by_size(values[start : start + chunk, c, :], n)
            total += curves.sum(axis=0)
            squares += (curves**2).sum(axis=0)
        t = values.shape[0]
        mean = total / t
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.maximum(squares - t * mean**2, 0.0) / (t - 1)
            err = np.sqrt(var / t)
        return mean, err

    def stats(c: int) -> tuple[np.ndarray, np.ndarray]:
        key = (probe.name, c)
        if key not in cache:
            cache[key] = compute(c)
        return cache[key]

    return stats


_VECTOR_MEANING = {
    "entropy_profile": "mean S(first m qubits of system_order) vs m (index)",
    "erasure_lost_profile": "mean bits of logical information lost vs erasure size m (index)",
    "erasure_first_failure": (
        "P_C(m): fraction of random m-qubit erasures that are uncorrectable vs m (index), "
        "estimated from the first-failure samples"
    ),
    "uncorrectable_counts": (
        "exact P_C(m): fraction of all m-qubit erasures that are uncorrectable vs m (index)"
    ),
}


def _bounded_summary(result: mipt.EnsembleResult, extra: dict[str, Any]) -> dict[str, Any]:
    """The summary of ``result`` plus ``extra``, within the size limit.

    When it is too long, fewer checkpoints are shown (evenly spaced, always
    the first and last, the same ones for every probe and the survival curve)
    before the generic cut of :func:`_bounded` is used as a last resort.
    """
    total = max(
        [len(result.checkpoint_layers)] + [len(result.layers(name)) for name in result.values]
    )
    long_extras = [
        key
        for key, value in extra.items()
        if key not in ("files", "path")
        and len(json.dumps(_jsonable(value))) > MAX_OUTPUT_CHARS // 4
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
        if len(json.dumps(_jsonable(out))) <= MAX_OUTPUT_CHARS:
            return _bounded(out)
    return _bounded(out)


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
    notes = []
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
            if probe.kind == "code_snapshot":
                notes.append(
                    f"{keys[name]}: the mean number of stabilizer generators (n - k); the codes "
                    "themselves are saved with output_dir (EnsembleResult.code(t, c) in Python)"
                )
        else:
            alive = (
                ~np.isnan(values[:, :, 0])
                if mipt._KINDS[probe.kind].after_purification == "nan"
                else np.ones(values.shape[:2], dtype=bool)
            )
            stats = _vector_stats(probe, values, n, cache)
            entry = {
                "what": _VECTOR_MEANING.get(probe.kind, "a curve over size m (index)"),
                **_vector_summary(stats, alive, own, sizes_shown, min(MAX_VECTOR_DEPTHS, shown)),
            }
        out["probes"][keys[name]] = entry
    if result.purification_layer is not None:
        out["purification"] = _purification_summary(result.purification_layer, layers, shown)
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
    if notes:
        out["notes"] = notes
    return out


def _purification_summary(
    tau: np.ndarray, layers: np.ndarray, shown: int = MAX_CHECKPOINTS_SHOWN
) -> dict[str, Any]:
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
        # The median of tau over all trajectories is well defined despite
        # censoring whenever more than half purified within the horizon.
        below = np.flatnonzero(surviving <= 0.5)
        out["tau_median_all"] = int(layers[below[0]]) if below.size else None
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
    out.mkdir(parents=True, exist_ok=True)
    return out


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    # numpy's stubs type the **arrays of savez_compressed as its allow_pickle flag
    np.savez_compressed(tmp, **arrays)  # type: ignore[arg-type, unused-ignore]
    os.replace(tmp, path)


def _atomic_save_result(result: mipt.EnsembleResult, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    result.save(tmp)
    os.replace(tmp, path)


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
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_run_metadata(
    out: Path,
    spec: dict,
    trajectories: int,
    seed: int,
    command: str,
    *,
    resumed_from: int | None = None,
) -> dict[str, str]:
    """Write ``spec.json`` (exact: replaying it repeats the run) and ``provenance.json``.

    For a resumed job (``resumed_from`` set) the original provenance is kept and
    this process's record is appended under ``"resumes"``.
    """
    try:
        spec_text = json.dumps(
            {"spec": _jsonable(spec, exact=True), "trajectories": trajectories, "seed": seed},
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
        record = {**stored, "resumes": [*stored.get("resumes", []), resume]}
    _atomic_write_text(prov_path, json.dumps(record, indent=2))
    return {"spec": str(spec_path), "provenance": str(prov_path)}


def _read_provenance(out: Path) -> dict | None:
    try:
        stored = json.loads((out / _PROVENANCE_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return stored if isinstance(stored, dict) else None


def _result_parts(out: Path) -> list[Path]:
    """``result.npz`` (if any) followed by the saved chunk files in trajectory order."""
    parts = []
    if (out / _RESULT_FILE).exists():
        parts.append(out / _RESULT_FILE)
    chunk_dir = out / _CHUNK_DIR
    if chunk_dir.is_dir():
        parts.extend(sorted(p for p in chunk_dir.glob("traj_*.npz") if ".tmp" not in p.name))
    return parts


def _load_result_file(path: Path) -> mipt.EnsembleResult:
    """Load one ``EnsembleResult.save`` file; anything else is reported as not a result."""
    try:
        return mipt.EnsembleResult.load(path)
    except FileNotFoundError:
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
    """Merge result.npz and chunks, skipping chunks that result.npz already contains.

    A merge writes result.npz before deleting the chunks it absorbed, so a
    crash (or a concurrent read) in between leaves chunks that overlap
    result.npz; they are redundant, not a conflict.
    """
    results: list[mipt.EnsembleResult] = []
    covered = 0
    for path in parts:
        part = _load_result_file(path)
        if path.name != _RESULT_FILE and part.num_trajectories and part.trajectories[-1] < covered:
            continue
        results.append(part)
        if part.num_trajectories:
            covered = max(covered, int(part.trajectories[-1]) + 1)
    try:
        merged = results[0] if len(results) == 1 else mipt.EnsembleResult.concatenate(results)
    except ValueError as exc:
        raise ValueError(f"saved parts do not belong to the same run: {exc}") from None
    expected = np.arange(merged.num_trajectories)
    if not np.array_equal(np.asarray(merged.trajectories), expected):
        raise ValueError(
            "saved trajectories are not contiguous from 0; the directory mixes different runs"
        )
    return merged


def _load_directory(out: Path) -> tuple[mipt.EnsembleResult | None, dict | None]:
    """Read an output directory, retrying if a running job merges chunks meanwhile."""
    for attempt in range(5):
        try:
            meta = None
            if (out / _SPEC_FILE).exists():
                meta = json.loads((out / _SPEC_FILE).read_text(encoding="utf-8"))
            parts = _result_parts(out)
            return (_load_parts_once(parts) if parts else None), meta
        except FileNotFoundError:
            if attempt == 4:
                raise
            time.sleep(0.1)
    raise AssertionError("unreachable")


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
    ``output_dir/result.npz`` with ``spec.json`` and ``provenance.json``.
    With ``max_seconds``, a run whose :func:`estimate_ensemble` exceeds it is
    refused before anything runs (the MCP tool's guard).
    """
    start = time.perf_counter()
    mc = _build_spec(spec)
    trajectories = _trajectory_count(mc, trajectories)
    if max_seconds is not None:
        max_seconds = _require_number(max_seconds, "max_seconds", 0.0, math.inf)
        estimate = _estimate_seconds(spec, mc, trajectories)
        if estimate > max_seconds:
            raise ValueError(
                f"this run is estimated at about {estimate:.0f} s, above "
                f"max_seconds={max_seconds:g}. Use start_ensemble_job with an output_dir and "
                "poll job_status, or reduce trajectories/layers/num_system (or raise "
                "max_seconds if you are willing to wait)."
            )
    seed = _resolve_seed(seed)
    if num_threads is not None:
        num_threads = _require_int(num_threads, "num_threads", minimum=1)
    out_path = _output_dir(output_dir) if output_dir is not None else None
    if out_path is not None and (_result_parts(out_path) or (out_path / _SPEC_FILE).exists()):
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
        _atomic_save_result(result, out_path / _RESULT_FILE)
        files = {"result": str(out_path / _RESULT_FILE)}
        files.update(
            _write_run_metadata(out_path, spec, trajectories, seed, command or "run_ensemble")
        )
        extra["files"] = files
    return _bounded_summary(result, extra)


def load_results(path: str) -> dict[str, Any]:
    """Summarize a saved ensemble: a ``.npz`` written by ``EnsembleResult.save`` or an output directory.

    For a directory (from :func:`run_ensemble` or a job), every saved chunk is
    included, so this also reads partial results of a running or cancelled
    job. The summary has the same form as :func:`run_ensemble`'s, plus the
    stored ``spec`` when ``spec.json`` exists.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError("path must be a string")
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ValueError(f"no such file or directory: {str(p)!r}")
    try:
        if p.is_dir():
            result, meta = _load_directory(p)
            if result is None:
                raise ValueError(f"no saved results in {str(p)!r} (expected result.npz or chunks/)")
        else:
            result, meta = _load_result_file(p), None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read results from {str(p)!r}: {exc}") from None
    extra: dict[str, Any] = {"path": str(p)}
    if meta is not None:
        extra["spec"] = meta.get("spec")
        extra["trajectories_requested"] = meta.get("trajectories")
    return _bounded_summary(result, extra)


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
            if self.status in ("completed", "cancelled", "failed") and self.done:
                out["next_step"] = f"load_results({str(self.output_dir)!r}) for the summary"
            return out


_JOBS: dict[str, _Job] = {}
_JOBS_LOCK = threading.Lock()


def _default_chunk_size(mc: mipt.MonitoredCircuit, trajectories: int) -> int:
    """Chunks of roughly 10 estimated seconds, at least a few per thread."""
    import numba

    threads = numba.get_num_threads()
    per_traj = max(_trajectory_us(mc) * 1e-6 / threads, 1e-9)
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
    larger ``trajectories`` extends a finished run. A directory holding a
    different spec or seed is refused. For a collection of graphs
    ``trajectories`` may be omitted (every graph gets ``circuits_per_graph``
    circuits).
    """
    mc = _build_spec(spec)
    trajectories = _trajectory_count(mc, trajectories)
    if num_threads is not None:
        num_threads = _require_int(num_threads, "num_threads", minimum=1)
    if output_dir is None:
        raise ValueError(
            "background jobs need an output_dir (results are saved there chunk by chunk)"
        )
    out = _output_dir(output_dir)
    existing, meta = _load_directory(out)
    notes: list[str] = []
    if meta is not None:
        if _canonical(meta.get("spec")) != _canonical(spec):
            raise ValueError(
                f"output_dir {str(out)!r} holds a run with a different spec; use a new directory"
            )
        if seed is not None and _resolve_seed(seed) != meta.get("seed"):
            raise ValueError(
                f"output_dir {str(out)!r} holds a run with seed {meta.get('seed')}; omit seed to "
                "resume it, or use a new directory"
            )
        seed = meta["seed"]
        stored = _read_provenance(out)
        made_with = None if stored is None else stored.get("sparsegf2")
        if made_with is not None and made_with != __version__:
            raise ValueError(
                f"output_dir {str(out)!r} holds a run started with sparsegf2 {made_with}, "
                f"but this is {__version__}; trajectories from different versions need not "
                "agree, so finish the run with that version or start a new directory"
            )
        started_at, now_at = (stored or {}).get("git_commit"), _git_commit()
        if started_at and now_at and started_at != now_at:
            notes.append(
                f"this resume runs git commit {now_at[:12]}, the run started at {started_at[:12]}; "
                "both are recorded in provenance.json"
            )
    elif existing is not None:
        raise ValueError(
            f"output_dir {str(out)!r} holds results without spec.json; use a new directory"
        )
    seed = _resolve_seed(seed)
    with _JOBS_LOCK:
        for other in _JOBS.values():
            if other.output_dir == out and other.status in ("queued", "running"):
                raise ValueError(
                    f"job {other.job_id} is already writing to {str(out)!r}; poll it or cancel it"
                )
        done = existing.num_trajectories if existing is not None else 0
        job = _Job(
            job_id=uuid.uuid4().hex[:12],
            spec=spec,
            trajectories=max(trajectories, done),
            seed=seed,
            output_dir=out,
            chunk_size=(
                _require_int(chunk_size, "chunk_size", minimum=1)
                if chunk_size is not None
                else _default_chunk_size(mc, trajectories)
            ),
            num_threads=num_threads,
            created=time.time(),
            done=done,
            resumed_from=done,
        )
        _JOBS[job.job_id] = job
    try:
        _write_run_metadata(
            out,
            spec,
            job.trajectories,
            seed,
            command or "start_ensemble_job",
            resumed_from=done if meta is not None else None,
        )
    except BaseException:
        with _JOBS_LOCK:  # never leave a job that would block this output_dir
            del _JOBS[job.job_id]
        raise
    job.thread = threading.Thread(
        target=_run_job, args=(job, mc), name=f"sparsegf2-job-{job.job_id}", daemon=True
    )
    job.thread.start()
    status = job.snapshot()
    if notes:
        status["notes"] = notes
    return _bounded(status)


def _canonical(obj: Any) -> str:
    """Exact JSON text of a spec, for comparing a stored run with a new request."""
    return json.dumps(_jsonable(obj, exact=True), sort_keys=True)


def _run_job(job: _Job, mc: mipt.MonitoredCircuit) -> None:
    chunk_dir = job.output_dir / _CHUNK_DIR
    with job.lock:
        job.started = time.time()
        job.status = "running"
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
        if finished:
            _merge_chunks(job.output_dir)
        with job.lock:
            job.status = "completed" if finished else "cancelled"
            job.finished = time.time()
    except Exception as exc:  # reported through job_status
        with job.lock:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}\n" + "".join(
                traceback.format_exception(exc)[-3:]
            )
            job.finished = time.time()


def _merge_chunks(out: Path) -> None:
    """Fold the chunk files into result.npz, then delete them (safe to interrupt anywhere)."""
    parts = _result_parts(out)
    chunk_parts = [p for p in parts if p.parent.name == _CHUNK_DIR]
    if not chunk_parts:
        return
    merged = _load_parts_once(parts)
    _atomic_save_result(merged, out / _RESULT_FILE)
    for p in chunk_parts:
        p.unlink(missing_ok=True)
    with contextlib.suppress(OSError):  # leftover temporary files of an interrupted save
        (out / _CHUNK_DIR).rmdir()


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
            "start_ensemble_job",
            "job_status",
            "cancel_job",
            "list_jobs",
            "load_results",
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
                "params": {**_signature_summary(fn, ("name", "checkpoints")), "name": None},
                "description": _first_sentence(fn),
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
        if section not in sections:
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

## 3. Long runs

estimate_ensemble(spec, trajectories) gives a rough runtime and the result's
memory. Above ~60 s use
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
"""


def quickstart() -> str:
    """A concise usage guide for agents (Markdown); also served as an MCP resource."""
    probes = textwrap.fill(", ".join(_probe_kinds()), width=78, initial_indent=" " * 8)
    return _QUICKSTART.replace("<PROBES>", probes.strip())
