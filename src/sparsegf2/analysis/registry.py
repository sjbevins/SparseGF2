"""The named-analysis registry.

An *analysis* is a callable ``fn(sim, spec) -> value`` that reads a final
:class:`~sparsegf2.SparseGF2` state and its :class:`~sparsegf2.circuits.PictureSpec`
and returns a scalar or ``ndarray``. The same callable is used both **online**
(the live ``sim`` at the end of a circuit) and **offline** (a ``sim`` rebuilt
from a saved tableau via :meth:`SparseGF2.from_symplectic`). Because every
analysis here is a phase-free function of the stabilizer subspace, the two paths
return identical values.

The built-ins bind their subsystem from the :class:`PictureSpec` (the system
register is qubits ``0 .. n_system-1``; the reference, when present, follows it),
so a single registered name behaves correctly across every picture.

Pass analyses to :func:`sparsegf2.circuits.simulate` (or
:func:`sparsegf2.analysis.sweep`) as a list mixing names and callables, e.g.
``analyses=["half_cut_entropy", "tripartite_mutual_info", my_fn]``; or as a
``{name: fn}`` mapping to name custom callables explicitly. Results land in
``SampleRecord.analyses``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from sparsegf2.core.observables import (
    average_stabilizer_weight,
    code_dimension,
    code_rate,
    contiguous_distance,
    entanglement_entropy,
    mutual_information,
    stabilizer_weight_spectrum,
    tripartite_mutual_info,
)
from sparsegf2.errors import InvalidArgumentError

# An analysis maps (final sim, picture layout) -> a scalar or ndarray.
Analysis = Callable[[Any, Any], Any]

ANALYSES: dict[str, Analysis] = {}


def register_analysis(name: str, fn: Analysis) -> None:
    """Register a named analysis. Raises if ``name`` is already taken."""
    if name in ANALYSES:
        raise InvalidArgumentError(f"analysis {name!r} is already registered")
    ANALYSES[name] = fn


def resolve_analyses(
    analyses: Sequence[str | Analysis] | Mapping[str, Analysis] | None,
) -> dict[str, Analysis]:
    """Normalize the user's ``analyses`` argument to an ordered ``{name: fn}``.

    Accepts ``None`` (→ empty), a single registered name (``str``), a mapping
    ``{name: fn}``, or a sequence mixing registered names and bare callables
    (keyed by ``fn.__name__``). Two callables that resolve to the same name
    (e.g. two ``lambda``s, both ``"<lambda>"``) raise; pass a ``{name: fn}``
    mapping to name them explicitly rather than silently dropping one.
    """
    if analyses is None:
        return {}
    if isinstance(analyses, str):
        analyses = [analyses]  # a bare name is a one-element list, not chars
    if isinstance(analyses, Mapping):
        out: dict[str, Analysis] = {}
        for name, fn in analyses.items():
            if not callable(fn):
                raise InvalidArgumentError(
                    f"analysis {name!r} must map to a callable; got {type(fn).__name__}"
                )
            out[name] = fn
        return out
    out = {}
    for item in analyses:
        if isinstance(item, str):
            if item not in ANALYSES:
                raise InvalidArgumentError(f"unknown analysis {item!r}; known: {sorted(ANALYSES)}")
            out[item] = ANALYSES[item]
        elif callable(item):
            key = getattr(item, "__name__", repr(item))
            if key in out:
                raise InvalidArgumentError(
                    f"duplicate analysis name {key!r}; two analyses resolve to the same "
                    "name: pass a {name: fn} mapping to name them explicitly"
                )
            out[key] = item
        else:
            raise InvalidArgumentError(
                f"analyses items must be a name (str) or a callable, got {type(item)}"
            )
    return out


# ----------------------------------------------------------------------
# Built-in analyses (subsystem bound from the PictureSpec)
# ----------------------------------------------------------------------
#
# The system register is qubits ``0 .. spec.n_system - 1`` for every picture
# (pure_state: the whole state; purification / single_ref: the system half,
# with the reference following). Contiguous ranges over it therefore match the
# runner's fixed observables and are picture-correct.


def half_cut_entropy(sim, spec) -> int:
    """``S(0 .. n_system/2 - 1)``, the half-cut of the system register.

    Reproduces ``SampleRecord.entropy_half_cut`` exactly.
    """
    if spec.n_system < 2:
        return 0
    return int(entanglement_entropy(sim, range(spec.n_system // 2)))


def code_dimension_analysis(sim, spec) -> int | None:
    """``k = S(system)`` in the purification picture; ``None`` otherwise."""
    if spec.order_parameter != "code_dimension":
        return None
    return int(code_dimension(sim, spec.n_system))


def ref_entropy(sim, spec) -> int | None:
    """``S(reference)`` in the single_ref picture; ``None`` otherwise."""
    if spec.order_parameter != "ref_entropy" or spec.reference_qubits.size == 0:
        return None
    return int(entanglement_entropy(sim, spec.reference_qubits))


def bipartite_mutual_info(sim, spec) -> int | None:
    """``I(left half : right half)`` of the system register."""
    n = spec.n_system
    if n < 2:
        return None
    half = n // 2
    return int(mutual_information(sim, range(half), range(half, n)))


def tripartite_mutual_info_quarters(sim, spec) -> int | None:
    """Canonical ``I_3(A:B:C)`` over the first three quarters of the system."""
    n = spec.n_system
    if n < 4:
        return None
    q = n // 4
    return int(tripartite_mutual_info(sim, range(q), range(q, 2 * q), range(2 * q, 3 * q)))


def code_rate_analysis(sim, spec) -> float | None:
    """``k / n_system``, the encoding rate; purification picture only."""
    if spec.order_parameter != "code_dimension":
        return None
    return code_rate(sim, spec.n_system)


def contiguous_distance_analysis(sim, spec) -> int | None:
    """Smallest contiguous system region carrying logical info (purification /
    single_ref pictures); ``None`` for pure_state or a fully-purified system."""
    if spec.reference_qubits.size == 0:
        return None
    return contiguous_distance(sim, spec.n_system, reference=spec.reference_qubits)


def average_stabilizer_weight_analysis(sim, spec) -> float:
    """Mean number of non-identity Paulis per stabilizer generator."""
    return float(average_stabilizer_weight(sim))


def stabilizer_weight_spectrum_analysis(sim, spec) -> np.ndarray:
    """Histogram of stabilizer weights, shape ``(n + 1,)``."""
    return stabilizer_weight_spectrum(sim)


def entropy_profile(sim, spec) -> np.ndarray:
    """``S(0..j)`` over contiguous system cuts ``j = 1 .. n_system - 1``."""
    n = spec.n_system
    if n < 2:
        return np.zeros((0,), dtype=np.int64)
    return np.array([entanglement_entropy(sim, range(j)) for j in range(1, n)], dtype=np.int64)


register_analysis("half_cut_entropy", half_cut_entropy)
register_analysis("code_dimension", code_dimension_analysis)
register_analysis("code_rate", code_rate_analysis)
register_analysis("contiguous_distance", contiguous_distance_analysis)
register_analysis("ref_entropy", ref_entropy)
register_analysis("bipartite_mutual_info", bipartite_mutual_info)
register_analysis("tripartite_mutual_info", tripartite_mutual_info_quarters)
register_analysis("average_stabilizer_weight", average_stabilizer_weight_analysis)
register_analysis("stabilizer_weight_spectrum", stabilizer_weight_spectrum_analysis)
register_analysis("entropy_profile", entropy_profile)


__all__ = ["ANALYSES", "Analysis", "register_analysis", "resolve_analyses"]
