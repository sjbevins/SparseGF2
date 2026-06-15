"""Circuit operation primitives for the inspector / trace layer.

An :class:`Op` is the atomic, **render-agnostic** unit of a circuit trace:
one gate, one measurement, etc. It is deliberately a single flexible
dataclass (a ``kind`` tag plus a ``detail`` dict) so that new operation kinds
and new constructions extend it *without a schema change*: a renderer
switches on ``kind`` and falls back gracefully for anything it doesn't
recognise, and a construction can stash arbitrary metadata in ``detail``.

This is the seam that makes the inspector extensible. The
:func:`apply_named_gate` helper is the inverse direction: it *executes* a
deterministic named gate on a simulator, so a picture can declare its setup
as a list of :class:`Op` (the single source of truth) and ``setup_picture``
simply applies them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sparsegf2.errors import InvalidArgumentError

# Named deterministic gates -> the SparseGF2 method that applies them. The
# simulator is duck-typed (no core import here), so ``ops`` stays a leaf
# module that both ``picture`` and the inspector can import without cycles.
_ONE_Q = {"H": "apply_h", "S": "apply_s", "SQRT_X": "apply_sqrt_x"}
_TWO_Q = {"CX": "apply_cx", "CZ": "apply_cz", "SWAP": "apply_swap"}

#: Gate names :func:`apply_named_gate` understands. Two-qubit names in
#: :data:`DIRECTED_GATES` have a control→target orientation.
NAMED_GATES: tuple[str, ...] = (*_ONE_Q, *_TWO_Q)
DIRECTED_GATES: frozenset[str] = frozenset({"CX"})


@dataclass
class Op:
    """One operation in a circuit trace.

    Attributes
    ----------
    kind : str
        Operation category: ``"gate"`` or ``"measure"`` today; new kinds
        (e.g. ``"reset"``) just add a string and a renderer branch.
    qubits : tuple of int
        The qubits the op acts on (1 for a single-qubit gate / measurement,
        2 for a two-qubit gate, …).
    label : str
        A short human label: a gate name (``"H"``, ``"CX"``), a random
        Clifford tag (``"C[417]"``), or a measurement (``"MZ"``).
    random : bool
        ``True`` when the op, or its placement, was *randomly chosen*
        (the specific Clifford, or which qubits got measured). ``False`` for
        deterministic construction gates (Bell pairs, etc.).
    detail : dict
        Free-form, extensible metadata: ``clifford_index``, ``basis``,
        ``note``, … A renderer may use it or ignore it.
    """

    kind: str
    qubits: tuple[int, ...]
    label: str
    random: bool = False
    detail: dict[str, Any] = field(default_factory=dict)


def apply_named_gate(sim, name: str, qubits: tuple[int, ...]) -> None:
    """Apply the deterministic named gate ``name`` on ``qubits`` to ``sim``.

    Dispatches by name to the simulator's ``apply_*`` methods. Raises
    :class:`~sparsegf2.errors.InvalidArgumentError` for an unknown gate name
    or a qubit-count mismatch.
    """
    if name in _ONE_Q:
        if len(qubits) != 1:
            raise InvalidArgumentError(f"{name} is a 1-qubit gate; got qubits={qubits}")
        getattr(sim, _ONE_Q[name])(qubits[0])
    elif name in _TWO_Q:
        if len(qubits) != 2:
            raise InvalidArgumentError(f"{name} is a 2-qubit gate; got qubits={qubits}")
        getattr(sim, _TWO_Q[name])(qubits[0], qubits[1])
    else:
        raise InvalidArgumentError(f"unknown named gate {name!r}; known: {NAMED_GATES}")


__all__ = ["Op", "NAMED_GATES", "DIRECTED_GATES", "apply_named_gate"]
