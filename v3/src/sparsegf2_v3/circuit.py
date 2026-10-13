"""Circuits written in Stim's circuit language.

A :class:`Circuit` is a list of instructions (and ``REPEAT`` blocks), exactly as
in Stim, and its text form *is* Stim's text form::

    circuit = Circuit('''
        R 0 1
        H 0
        CX 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
    ''')
    print(circuit)            # the same text back, normalised
    circuit.to_stim()         # a stim.Circuit (needs Stim installed)
    Circuit.from_stim(stim_circuit)

Supported instructions are listed by :func:`sparsegf2_v3.all_gates`: every
Clifford gate, ``M MX MY MR MRX MRY MXX MYY MZZ MPP R RX RY``, the Pauli noise
channels, ``TICK DETECTOR OBSERVABLE_INCLUDE QUBIT_COORDS SHIFT_COORDS``, and
``REPEAT`` blocks. Classically controlled Paulis use Stim's syntax
(``CX rec[-1] 5``, ``CZ rec[-2] 3``). Anything else raises an error naming the
instruction.

SparseGF2 adds two instructions, ``CLIFFORD_1Q`` and ``CLIFFORD_2Q``, that apply
a Clifford by its canonical index (:meth:`sparsegf2_v3.Clifford.from_index`),
with one argument per application::

    CLIFFORD_2Q(17, 9001) 0 1 2 3     # Clifford #17 on (0,1), #9001 on (2,3)

They are what random-circuit generators emit. ``to_stim()`` expands them into
Stim gates, so every circuit can still be handed to Stim.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from sparsegf2_v3.clifford import Clifford
from sparsegf2_v3.gates import GateInfo, gate

__all__ = [
    "Circuit",
    "Instruction",
    "RepeatBlock",
    "Target",
    "target_combiner",
    "target_inv",
    "target_rec",
    "target_x",
    "target_y",
    "target_z",
]


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """One target of an instruction, as in Stim.

    ``kind`` is ``"qubit"`` (``5``, or ``!5`` when inverted), ``"rec"`` (a
    measurement-record lookback ``rec[-k]``, with ``value = -k``), ``"pauli"``
    (``X5``/``Y5``/``Z5`` in MPP, possibly inverted) or ``"combiner"`` (the
    ``*`` joining MPP factors).
    """

    kind: str
    value: int = 0
    pauli: str = ""
    inverted: bool = False

    def __str__(self) -> str:
        bang = "!" if self.inverted else ""
        if self.kind == "qubit":
            return f"{bang}{self.value}"
        if self.kind == "rec":
            return f"rec[{self.value}]"
        if self.kind == "pauli":
            return f"{bang}{self.pauli}{self.value}"
        return "*"


def target_rec(lookback: int) -> Target:
    """``rec[lookback]``, a negative index into the measurement record."""
    if lookback >= 0:
        raise ValueError(f"record lookbacks are negative (rec[-1] is the latest), got {lookback}")
    return Target("rec", lookback)


def target_inv(qubit: int) -> Target:
    """``!qubit``: a measurement target whose recorded result is inverted."""
    return Target("qubit", _nonnegative(qubit), inverted=True)


def target_x(qubit: int, inverted: bool = False) -> Target:
    """``X<qubit>``, an MPP factor."""
    return Target("pauli", _nonnegative(qubit), "X", inverted)


def target_y(qubit: int, inverted: bool = False) -> Target:
    """``Y<qubit>``, an MPP factor."""
    return Target("pauli", _nonnegative(qubit), "Y", inverted)


def target_z(qubit: int, inverted: bool = False) -> Target:
    """``Z<qubit>``, an MPP factor."""
    return Target("pauli", _nonnegative(qubit), "Z", inverted)


def target_combiner() -> Target:
    """``*``, joining MPP factors into one product."""
    return Target("combiner")


def _nonnegative(q) -> int:
    q = int(q)
    if q < 0:
        raise ValueError(f"qubit indices must be non-negative, got {q}")
    return q


def _as_target(t) -> Target:
    if isinstance(t, Target):
        return t
    if isinstance(t, bool):
        raise TypeError("targets must be ints or Target objects, not bool")
    try:
        return Target("qubit", _nonnegative(int(t)))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"targets must be ints or Target objects, got {t!r}") from exc


_TARGET_RE = re.compile(r"^(!?)(?:(\d+)|([XYZ])(\d+)|rec\[(-\d+)\]|(\*))$", re.IGNORECASE)


def _parse_target(token: str) -> Target:
    m = _TARGET_RE.match(token)
    if not m:
        if token.lower().startswith("sweep["):
            raise ValueError("sweep[] targets are not supported")
        raise ValueError(f"cannot parse target {token!r}")
    bang, qubit, pauli, pauli_qubit, rec, star = m.groups()
    inverted = bang == "!"
    if qubit is not None:
        return Target("qubit", int(qubit), inverted=inverted)
    if pauli is not None:
        return Target("pauli", int(pauli_qubit), pauli.upper(), inverted)
    if rec is not None:
        if inverted:
            raise ValueError(f"record targets cannot be inverted: {token!r}")
        return target_rec(int(rec))
    if inverted:
        raise ValueError("the combiner '*' cannot be inverted")
    return target_combiner()


# ---------------------------------------------------------------------------
# instructions
# ---------------------------------------------------------------------------


def _format_arg(x: float) -> str:
    return str(int(x)) if float(x).is_integer() and abs(x) < 2**53 else repr(float(x))


@dataclass(frozen=True)
class Instruction:
    """One line of a circuit: a gate name, its targets, and parenthesized arguments."""

    name: str
    targets: tuple[Target, ...] = ()
    args: tuple[float, ...] = ()
    tag: str = ""

    @property
    def info(self) -> GateInfo:
        return gate(self.name)

    def __str__(self) -> str:
        text = self.name
        if self.tag:
            text += f"[{self.tag}]"
        if self.args:
            text += "(" + ", ".join(_format_arg(a) for a in self.args) + ")"
        if self.targets:
            parts: list[str] = []
            for t in self.targets:
                if t.kind == "combiner" or (parts and parts[-1].endswith("*")):
                    parts[-1] += str(t)
                else:
                    parts.append(str(t))
            text += " " + " ".join(parts)
        return text


@dataclass(frozen=True)
class RepeatBlock:
    """``REPEAT count { body }``."""

    count: int
    body: Circuit = field(compare=True)

    def __str__(self) -> str:
        inner = "\n".join("    " + line for line in str(self.body).splitlines())
        return f"REPEAT {self.count} {{\n{inner}\n}}"


def _validate(name: str, targets: tuple[Target, ...], args: tuple[float, ...]) -> GateInfo:
    """Check one instruction and return its gate info (raises ValueError)."""
    info = gate(name)
    kind = info.kind
    if info.num_args is not None and len(args) not in info.num_args:
        raise ValueError(f"{info.name} takes {info.num_args} arguments, got {len(args)}")
    kinds = {t.kind for t in targets}

    if kind == "annotation":
        if info.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            if kinds - {"rec"}:
                raise ValueError(f"{info.name} targets must be rec[-k] lookbacks")
            if info.name == "OBSERVABLE_INCLUDE" and (
                args[0] < 0 or not float(args[0]).is_integer()
            ):
                raise ValueError("OBSERVABLE_INCLUDE needs a non-negative integer index")
        elif info.name == "QUBIT_COORDS":
            if kinds - {"qubit"} or any(t.inverted for t in targets):
                raise ValueError("QUBIT_COORDS targets must be qubits")
        elif targets:
            raise ValueError(f"{info.name} takes no targets")
        return info

    if kind == "measure_product":
        if kinds - {"pauli", "combiner"}:
            raise ValueError("MPP targets must be Pauli factors like X0, Z3 joined by '*'")
        # Products are factors joined by '*'; consecutive factors without '*'
        # start a new product. A '*' must sit between two factors.
        prev = "combiner"
        for t in targets:
            if t.kind == "combiner" and prev == "combiner":
                raise ValueError("misplaced '*' in MPP targets (expected e.g. X0*Z1 Y2)")
            prev = t.kind
        if targets and prev == "combiner":
            raise ValueError("MPP target list ends with '*'")
        if any(a < 0 or a > 1 for a in args):
            raise ValueError("measurement flip probabilities must be in [0, 1]")
        return info

    # Remaining kinds take plain qubit targets (plus rec controls for feedback).
    if "pauli" in kinds or "combiner" in kinds:
        raise ValueError(f"{info.name} does not take Pauli or '*' targets")
    if any(t.inverted for t in targets) and kind not in (
        "measure",
        "measure_reset",
        "measure_pair",
    ):
        raise ValueError(f"only measurement targets can be inverted (got one in {info.name})")
    if "rec" in kinds and info.name not in ("CX", "CY", "CZ"):
        raise ValueError(
            f"{info.name} cannot be classically controlled; only CX, CY and CZ accept rec[] targets"
        )
    k = info.num_qubits
    if k == 2:
        if len(targets) % 2:
            raise ValueError(f"{info.name} needs an even number of targets, got {len(targets)}")
        for a, b in zip(targets[::2], targets[1::2], strict=True):
            if a.kind == "rec" and b.kind == "rec":
                raise ValueError(f"{info.name} cannot have two rec[] targets in one pair")
            if a.kind == "rec" or b.kind == "rec":
                if info.name in ("CX", "CY") and b.kind == "rec":
                    raise ValueError(
                        f"{info.name} takes the rec[] control first: {info.name} rec[-1] q"
                    )
            elif a.value == b.value:
                raise ValueError(
                    f"{info.name} targets in a pair must differ, got ({a.value}, {b.value})"
                )
    if kind == "noise":
        total = sum(args)
        if any(a < 0 for a in args) or total > 1 + 1e-12:
            raise ValueError(f"{info.name} probabilities must be non-negative and sum to at most 1")
    if kind in ("measure", "measure_reset", "measure_pair") and any(a < 0 or a > 1 for a in args):
        raise ValueError("measurement flip probabilities must be in [0, 1]")
    if kind == "clifford_index":
        count = len(targets) // k
        if len(args) != count:
            raise ValueError(
                f"{info.name} needs one index per application ({count}), got {len(args)}"
            )
        limit = 24 if k == 1 else 11520
        for a in args:
            if not float(a).is_integer() or not 0 <= a < limit:
                raise ValueError(f"{info.name} indices must be integers in [0, {limit}), got {a}")
    return info


# ---------------------------------------------------------------------------
# the circuit
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\[(?P<tag>[^\]]*)\])?"
    r"(?:\((?P<args>[^)]*)\))?"
    r"(?P<rest>.*)$"
)


class Circuit:
    """A Stim-language circuit. Build it from text, by appending, or from Stim.

    Circuits are mutable lists of :class:`Instruction` and :class:`RepeatBlock`
    items; iterate to read them. ``str(circuit)`` is valid Stim text (apart from
    the ``CLIFFORD_1Q``/``CLIFFORD_2Q`` extensions, which :meth:`to_stim`
    expands).
    """

    def __init__(self, text: str = ""):
        self._items: list[Instruction | RepeatBlock] = []
        self._version = 0
        if text:
            self._items = _parse(text)

    # -- building ---------------------------------------------------------------------

    def append(self, name: str, targets: Iterable = (), args: Iterable[float] = (), tag: str = ""):
        """Append one instruction, e.g. ``c.append("CX", [0, 1, 2, 3])``.

        ``targets`` are ints (qubits) or :class:`Target` objects (see
        :func:`target_rec`, :func:`target_x`, ...). ``args`` are the
        parenthesized arguments (noise probabilities, Clifford indices, ...).
        """
        tgts = tuple(_as_target(t) for t in targets)
        arg_tuple = tuple(float(a) for a in args)
        info = _validate(name, tgts, arg_tuple)
        self._items.append(Instruction(info.name, tgts, arg_tuple, tag))
        self._version += 1

    def append_text(self, text: str) -> None:
        """Append instructions written in Stim's text format."""
        self._items.extend(_parse(text))
        self._version += 1

    def append_repeat(self, count: int, body: Circuit) -> None:
        """Append ``REPEAT count { body }`` (``body`` is copied)."""
        if int(count) < 1:
            raise ValueError(f"REPEAT count must be positive, got {count}")
        self._items.append(RepeatBlock(int(count), body.copy()))
        self._version += 1

    def __iadd__(self, other: Circuit) -> Circuit:
        if not isinstance(other, Circuit):
            return NotImplemented
        self._items.extend(other._items)
        self._version += 1
        return self

    def __add__(self, other: Circuit) -> Circuit:
        if not isinstance(other, Circuit):
            return NotImplemented
        out = self.copy()
        out += other
        return out

    def __mul__(self, repetitions: int) -> Circuit:
        """``circuit * k`` wraps the circuit in ``REPEAT k { ... }``."""
        out = Circuit()
        if repetitions == 1:
            out += self
        elif repetitions > 1:
            out.append_repeat(repetitions, self)
        return out

    def copy(self) -> Circuit:
        out = Circuit()
        out._items = list(self._items)
        return out

    # -- reading --------------------------------------------------------------------------

    def __iter__(self) -> Iterator[Instruction | RepeatBlock]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Instruction | RepeatBlock:
        return self._items[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Circuit):
            return NotImplemented
        return self._items == other._items

    __hash__ = None  # mutable

    def __str__(self) -> str:
        return "\n".join(str(item) for item in self._items)

    def __repr__(self) -> str:
        return f"Circuit('''\n{self}\n''')"

    def flattened(self) -> Iterator[Instruction]:
        """Every instruction in execution order, with REPEAT blocks unrolled."""
        for item in self._items:
            if isinstance(item, RepeatBlock):
                for _ in range(item.count):
                    yield from item.body.flattened()
            else:
                yield item

    @property
    def num_qubits(self) -> int:
        """One more than the largest qubit index used (0 for an empty circuit)."""
        top = -1
        for item in self._items:
            if isinstance(item, RepeatBlock):
                top = max(top, item.body.num_qubits - 1)
            else:
                for t in item.targets:
                    if t.kind in ("qubit", "pauli"):
                        top = max(top, t.value)
        return top + 1

    def _count(self, fn) -> int:
        total = 0
        for item in self._items:
            if isinstance(item, RepeatBlock):
                total += item.count * item.body._count(fn)
            else:
                total += fn(item)
        return total

    @property
    def num_measurements(self) -> int:
        """Number of measurement results the circuit records."""
        return self._count(_measurements_of)

    @property
    def num_detectors(self) -> int:
        return self._count(lambda ins: 1 if ins.name == "DETECTOR" else 0)

    @property
    def num_observables(self) -> int:
        top = -1
        for ins in self.flattened():
            if ins.name == "OBSERVABLE_INCLUDE":
                top = max(top, int(ins.args[0]))
        return top + 1

    @property
    def num_ticks(self) -> int:
        return self._count(lambda ins: 1 if ins.name == "TICK" else 0)

    # -- Stim interop ---------------------------------------------------------------------

    def to_stim(self):
        """Convert to a ``stim.Circuit`` (requires Stim).

        ``CLIFFORD_1Q``/``CLIFFORD_2Q`` instructions are expanded into an
        equivalent sequence of Stim gates (Stim's ``Tableau.to_circuit``).
        """
        import stim

        return stim.Circuit(self._stim_text())

    def _stim_text(self) -> str:
        lines = []
        for item in self._items:
            if isinstance(item, RepeatBlock):
                inner = "\n".join("    " + ln for ln in item.body._stim_text().splitlines())
                lines.append(f"REPEAT {item.count} {{\n{inner}\n}}")
            elif item.name in ("CLIFFORD_1Q", "CLIFFORD_2Q"):
                lines.extend(_expand_clifford_index(item))
            else:
                lines.append(str(item))
        return "\n".join(lines)

    @classmethod
    def from_stim(cls, stim_circuit) -> Circuit:
        """Convert from a ``stim.Circuit``. Unsupported instructions raise ``ValueError``."""
        return cls(str(stim_circuit))


def _measurements_of(ins: Instruction) -> int:
    kind = ins.info.kind
    if kind in ("measure", "measure_reset"):
        return len(ins.targets)
    if kind == "measure_pair":
        return len(ins.targets) // 2
    if kind == "measure_product":
        return sum(1 for t in ins.targets if t.kind == "pauli") - sum(
            1 for t in ins.targets if t.kind == "combiner"
        )
    return 0


def _expand_clifford_index(ins: Instruction) -> list[str]:
    k = ins.info.num_qubits
    lines = []
    for j, index in enumerate(ins.args):
        qubits = [t.value for t in ins.targets[k * j : k * (j + 1)]]
        sub = Clifford.from_index(k, int(index)).to_stim().to_circuit("elimination")
        for op in sub:
            remapped = " ".join(str(qubits[t.value]) for t in op.targets_copy())
            lines.append(f"{op.name} {remapped}")
    return lines


def _parse(text: str) -> list[Instruction | RepeatBlock]:
    """Parse Stim circuit text into items (raises ValueError with a line number)."""
    stack: list[tuple[list, int]] = [([], 0)]
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            if line == "}":
                if len(stack) == 1:
                    raise ValueError("unmatched '}'")
                items, count = stack.pop()
                body = Circuit()
                body._items = items
                stack[-1][0].append(RepeatBlock(count, body))
                continue
            m = _LINE_RE.match(line)
            if not m:
                raise ValueError(f"cannot parse {line!r}")
            name = m.group("name").upper()
            rest = m.group("rest").strip()
            if name == "REPEAT":
                if not rest.endswith("{"):
                    raise ValueError("REPEAT needs '{' on the same line")
                count = int(rest[:-1].strip())
                if count < 1:
                    raise ValueError(f"REPEAT count must be positive, got {count}")
                stack.append(([], count))
                continue
            args_text = m.group("args")
            args = (
                tuple(float(a) for a in args_text.split(","))
                if args_text and args_text.strip()
                else ()
            )
            tokens = rest.replace("*", " * ").split()
            targets = tuple(_parse_target(tok) for tok in tokens)
            info = _validate(name, targets, args)
            stack[-1][0].append(Instruction(info.name, targets, args, m.group("tag") or ""))
        except (ValueError, KeyError) as exc:
            msg = exc.args[0] if exc.args else str(exc)
            raise ValueError(f"line {lineno}: {msg}") from None
    if len(stack) != 1:
        raise ValueError("unterminated REPEAT block (missing '}')")
    return stack[0][0]
