"""Identities, seeds and graph records of the Watts–Strogatz study (v1.4 §1–§3, §7.2, §8).

Everything here is a pure function of its arguments, so any record of the study can be
rebuilt from the manifest alone. Four things are defined:

**Tokens** (:class:`Token`). The measurement rate $p$ and the rewiring probability $\\beta$
are identified by an exact rational number, never by a float. A token holds the
:class:`~fractions.Fraction` $x \\in [0, 1]$, its canonical string (the decimal expansion
without exponent or trailing zeros when $x$ has one, else ``"num/den"`` in lowest terms),
the correctly rounded ``float64`` handed to the simulator, and ``float.hex()`` of that float.
The ideal Bernoulli($x$) law and the law the simulator realizes differ by less than
$2^{-52}$ (v1.4 §7.2, §9; ``docs/study/01-ensemble-and-sampling.md``).

**Seeds** (:func:`graph_seed`, :func:`run_seed`, :func:`stream_index`; ruling R-SEED). Every
seed is a fixed-length tuple of 32-bit words that begins with a domain tag and names the
condition: graph $g$ of $(n, \\beta)$ is built from ``(GRAPH_SEED_TAG, master_graph_seed, n,
*token_words(beta), g)``, so the graph collection does not depend on the measurement rate
(one collection serves every $p$) and sizes and rewiring probabilities get independent
collections. The run seed is ``(CIRCUIT_SEED_TAG, master_circuit_seed, n,
*token_words(beta))`` in ``gate_mode="shared"``, the same for every $p$ (the native coupling:
gates and mask uniforms are common to all rates, masks are nested), and that tuple followed
by ``token_words(p)`` in ``"independent"``. The two tags make graph seeds and run seeds
disjoint whatever the master seeds are. Trajectory $(g, c)$ has the native stream index
$u = g \\cdot s + c$, with the stride $s$ frozen in the manifest, so adding graphs or
circuits never renumbers an existing trajectory.

**Graph records** (:class:`GraphRecord`, :func:`build_graph`). A record holds the
Watts–Strogatz graph $G_g \\sim \\mu_{n,\\beta}$ of v1.4 §2 with the seed and generator version
it came from, its canonical sorted edge list and the SHA-256 digest of that list.

**Canonical bytes** (:func:`canonical_json`, :func:`sha256_hex`, :func:`json_digest`). The
manifest digest and every later record digest are SHA-256 of canonical JSON (sorted keys, no
spaces, tokens as strings, finite numbers only).

The version constants name the exact algorithms a record was produced under. Changing any
algorithm they name needs a new constant: results saved under different values must not be
mixed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from sparsegf2 import graphs, mipt
from sparsegf2._version import __version__

__all__ = [
    "CIRCUIT_SEED_TAG",
    "CLIFFORD_ENUMERATION_VERSION",
    "DYNAMICS_VERSION",
    "EDGE_DIGEST_MAGIC",
    "GATE_MODES",
    "GRAPH_GENERATOR_VERSION",
    "GRAPH_SEED_TAG",
    "ORIGINAL_EDGE_ORDER",
    "PRODUCT_DYNAMICS_VERSION",
    "PROTOCOLS",
    "ROWSPACE_FORMAT_VERSION",
    "SEED_LIMIT",
    "STORE_SCHEMA_VERSION",
    "GraphRecord",
    "Token",
    "TrajectoryKey",
    "build_graph",
    "canonical_json",
    "check_experiment_id",
    "check_protocol",
    "circulant_edges",
    "dynamics_version",
    "edge_digest",
    "graph_seed",
    "json_digest",
    "original_edge_list",
    "rate_hash",
    "run_seed",
    "sha256_hex",
    "split_stream_index",
    "stream_index",
    "token_words",
    "validate_graph",
    "version_record",
]

# ---------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------

#: The graph law and its realization: v1.4 §2.2 (Bernoulli(beta) per original edge in the
#: offset-major order, eligible targets taken before the deletion, retained endpoint ``a``),
#: with ``next_double() < beta`` for the Bernoulli draw and rejection with ``next_below(n)``
#: for the uniform target, from the xoshiro256** state ``SeedSequence(seed).generate_state(4)``.
GRAPH_GENERATOR_VERSION = "ws-offset-major-reject-1"

#: The native two-qubit Clifford numbering, ``j = 16 s + b`` (v1.4 §9.1): ``s`` the index of the
#: symplectic matrix in the sorted list of the 720 elements of Sp(4, F_2) (row-major bits, first
#: entry most significant), ``b`` the four image-sign bits in the order X_0, X_1, Z_0, Z_1.
CLIFFORD_ENUMERATION_VERSION = "sp4-sorted-rowmajor-x16-signs-1"

#: The dynamics of one trajectory: purification picture (n Bell pairs), ``n // 2`` gates per
#: layer on uniformly drawn edges with uniformly drawn Cliffords, then Z measurements with
#: probability p per site, phase-free (v1.4 §3, §4, §9).
DYNAMICS_VERSION = "monitored-random-edges-purification-phase-free-1"

#: The same gates and measurements started from the pure product state |0...0> of the n system
#: qubits, without a reference (protocol ``"P"`` of design §12; the queued study v0.3 §2).
PRODUCT_DYNAMICS_VERSION = "monitored-random-edges-product-state-phase-free-1"

#: The two initial conditions of a trajectory (design §12): ``"F"``, the full reference of v1.4
#: (n Bell pairs between the system and n reference qubits, ``reference="purification"``), and
#: ``"P"``, the pure product state |0...0> of the system alone (``reference="none"``). Both draw
#: the same gates and measurement masks from the same stream for the same graph, run seed and
#: stream index, so a P trajectory replays its F partner's gate and mask history.
PROTOCOLS = ("F", "P")

#: The canonical phase-free row-space serialization (header ``b"SGF2RS1\0"``, design §2).
ROWSPACE_FORMAT_VERSION = "SGF2RS1"

#: The layout of the SQLite store (design §4).
STORE_SCHEMA_VERSION = "sparsegf2.study.store/1"

#: The ordered original-edge convention of v1.4 §2.2 recorded with every graph.
ORIGINAL_EDGE_ORDER = "offset-major"

#: Prefix of the bytes whose SHA-256 is an edge digest.
EDGE_DIGEST_MAGIC = b"SGF2EDGES1\x00"

#: Master seeds, sizes and graph labels are unsigned 32-bit integers: each is exactly one word of
#: the seed tuples below (a larger value would be split into several words by ``SeedSequence``).
SEED_LIMIT = 2**32

#: The first word of every graph seed, ``b"WSG1"`` read as a big-endian integer (design §3,
#: ruling R-SEED). Together with :data:`CIRCUIT_SEED_TAG` it makes the two seed domains disjoint
#: whatever the master seeds are. A different derivation of the seeds needs different tags.
GRAPH_SEED_TAG = 0x57534731

#: The first word of every run (circuit) seed, ``b"WSC1"`` read as a big-endian integer.
CIRCUIT_SEED_TAG = 0x57534331

_WORD_MASK = SEED_LIMIT - 1

#: The two ways the circuit seed depends on the measurement rate (design §3).
GATE_MODES = ("shared", "independent")


def version_record() -> dict[str, Any]:
    """Every version a record of the study depends on, as a plain dictionary.

    The keys are the lower-case names of the constants in this module, the native
    ``mipt.RANDOMNESS_CONTRACT`` and the package version.
    """
    return {
        "graph_generator": GRAPH_GENERATOR_VERSION,
        "clifford_enumeration": CLIFFORD_ENUMERATION_VERSION,
        "dynamics": DYNAMICS_VERSION,
        "rowspace_format": ROWSPACE_FORMAT_VERSION,
        "store_schema": STORE_SCHEMA_VERSION,
        "graph_seed_tag": GRAPH_SEED_TAG,
        "circuit_seed_tag": CIRCUIT_SEED_TAG,
        "randomness_contract": int(mipt.RANDOMNESS_CONTRACT),
        "sparsegf2": __version__,
    }


# ---------------------------------------------------------------------------
# small validators
# ---------------------------------------------------------------------------


def _integer(
    value: Any, what: str, *, minimum: int | None = None, maximum: int | None = None
) -> int:
    """``value`` as a Python int: ints and numpy integers only (never bools, floats or strings)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    v = int(value)
    if minimum is not None and v < minimum:
        raise ValueError(f"{what} must be at least {minimum}, got {v}")
    if maximum is not None and v > maximum:
        raise ValueError(f"{what} must be at most {maximum}, got {v}")
    return v


def _seed_word(value: Any, what: str) -> int:
    """A seed or label in ``[0, SEED_LIMIT)`` (one 32-bit word of the seed entropy)."""
    v = _integer(value, what, minimum=0)
    if v >= SEED_LIMIT:
        raise ValueError(
            f"{what} must be below 2**32 = {SEED_LIMIT} (one 32-bit word of a seed tuple; a larger "
            f"value would be split into several words), got {v}"
        )
    return v


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------

_DECIMAL = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]{1,4})?")
_RATIONAL = re.compile(r"[+-]?[0-9]+/[0-9]+")


def _canonical_token(value: Fraction) -> str:
    """The canonical string of a rational in [0, 1] (see :class:`Token`)."""
    num, den = value.numerator, value.denominator
    rest, twos, fives = den, 0, 0
    while rest % 2 == 0:
        rest //= 2
        twos += 1
    while rest % 5 == 0:
        rest //= 5
        fives += 1
    if rest != 1:  # not a terminating decimal: lowest terms
        return f"{num}/{den}"
    digits = max(twos, fives)
    scaled = num * 10**digits // den  # exact: den divides 10**digits
    text = str(scaled).rjust(digits + 1, "0")
    if digits == 0:
        return text
    whole, fraction = text[:-digits], text[-digits:].rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


@dataclass(frozen=True, order=True)
class Token:
    """An exact probability in $[0, 1]$: the identifier of a rate $p$ or a rewiring probability $\\beta$.

    A token is the rational number $x$ the experimenter meant, held exactly
    (``value``, a :class:`~fractions.Fraction`). It is built by :meth:`parse` from

    * a decimal string (``"0.1"``, ``"1e-2"``, ``".25"``),
    * a rational string (``"1/7"``; ``"1/10"`` and ``"0.1"`` are the same token),
    * a JSON number: a float is read through its shortest round-trip decimal ``repr``
      (``0.1`` is $1/10$, not the binary neighbor $0.1000000000000000055\\ldots$), an int as is,
    * another token or a ``Fraction``.

    Properties:

    * ``token``: the canonical string. If the denominator of $x$ (in lowest terms) has only the
      prime factors 2 and 5, the decimal expansion without exponent and without trailing zeros
      (``"0"``, ``"1"``, ``"0.05"``); else ``"num/den"`` (``"1/3"``). Two tokens are equal
      exactly when their canonical strings are.
    * ``float64``: ``float(x)``, the correctly rounded IEEE-754 double (``numerator /
      denominator`` in Python is correctly rounded). This is the number the simulator receives.
    * ``hex``: ``float.hex()`` of ``float64``, the bit-exact record of what ran.
    * ``is_exact``: whether ``float64`` equals $x$ (true for dyadic rationals such as ``"0.5"``).
    """

    value: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.value, Fraction) or not 0 <= self.value <= 1:
            raise ValueError(f"a token holds a Fraction in [0, 1], got {self.value!r}")

    @classmethod
    def parse(cls, source: Any, *, what: str = "value") -> Token:
        """The token of ``source`` (a string, number, ``Fraction`` or token), in $[0, 1]$.

        ``what`` names the field in error messages.
        """
        if isinstance(source, Token):
            return source
        if isinstance(source, Fraction):
            value = source
        elif isinstance(source, bool):
            raise ValueError(f"{what} must be a probability, got the boolean {source!r}")
        elif isinstance(source, (int, np.integer)):
            value = Fraction(int(source))
        elif isinstance(source, (float, np.floating)):
            x = float(source)
            if not math.isfinite(x):
                raise ValueError(f"{what} must be a finite probability, got {x}")
            value = Fraction(repr(x))  # the shortest decimal that round-trips
        elif isinstance(source, str):
            text = source.strip()
            if _RATIONAL.fullmatch(text):
                num, den = text.split("/")
                if int(den) == 0:
                    raise ValueError(f"{what} {source!r} has denominator 0")
                value = Fraction(int(num), int(den))
            elif _DECIMAL.fullmatch(text):
                value = Fraction(text)
            else:
                raise ValueError(
                    f"{what} must be a decimal such as '0.1' or a rational such as '1/7' in "
                    f"[0, 1], got {source!r}"
                )
        else:
            raise ValueError(
                f"{what} must be a decimal string, a rational string or a number in [0, 1], "
                f"got {source!r}"
            )
        if not 0 <= value <= 1:
            raise ValueError(f"{what} must lie in [0, 1], got {source!r}")
        return cls(value)

    @property
    def token(self) -> str:
        return _canonical_token(self.value)

    @property
    def float64(self) -> float:
        return float(self.value)

    @property
    def hex(self) -> str:
        return self.float64.hex()

    @property
    def is_exact(self) -> bool:
        return Fraction(self.float64) == self.value

    def __str__(self) -> str:
        return self.token

    def __repr__(self) -> str:
        return f"Token({self.token!r})"


def rate_hash(token: Token | str | float) -> int:
    """The 64-bit hash that names a rate, or a rewiring probability, in a seed (design §3).

    The first 8 bytes of ``blake2b(token.encode("ascii"), digest_size=8)`` of the canonical
    token string, read little-endian: the same convention as the probe-name hash of
    ``sparsegf2.mipt`` (``digest_size=8``, not a truncated 64-byte digest). Equal tokens give
    equal hashes whatever their spelling; :func:`token_words` splits the hash into the two
    32-bit words that enter the seeds.
    """
    text = Token.parse(token, what="rate").token
    return int.from_bytes(hashlib.blake2b(text.encode("ascii"), digest_size=8).digest(), "little")


def token_words(token: Token | str | float) -> tuple[int, int]:
    r"""The two 32-bit words ``(lo, hi)`` of :func:`rate_hash` of a token, low word first.

    ``lo = h mod 2**32`` and ``hi = h // 2**32`` for ``h = rate_hash(token)``. They name the
    rewiring probability $\beta$ (and, in ``gate_mode="independent"``, the rate $p$) inside a
    seed tuple. Two different tokens have different words unless their 64-bit hashes collide
    (probability $2^{-64}$ for a given pair); the manifest refuses a collision among its own
    tokens.
    """
    value = rate_hash(token)
    return value & _WORD_MASK, value >> 32


# ---------------------------------------------------------------------------
# trajectory identity and seeds
# ---------------------------------------------------------------------------

_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def check_experiment_id(value: Any, what: str = "experiment_id") -> str:
    """``value`` if it is a valid experiment identifier, else ``ValueError``.

    One to 64 characters from ``A-Z a-z 0-9 . _ -``, starting with a letter or digit, so that
    it is safe in file names, SQL and the colon-separated form of :class:`TrajectoryKey`.
    """
    if not isinstance(value, str) or not _EXPERIMENT_ID.fullmatch(value):
        raise ValueError(
            f"{what} must be 1 to 64 characters from A-Z a-z 0-9 . _ - starting with a letter "
            f"or digit, got {value!r}"
        )
    return value


def check_protocol(value: Any, what: str = "protocol") -> str:
    """``value`` if it names a protocol of :data:`PROTOCOLS` (``"F"`` or ``"P"``), else ``ValueError``."""
    if not isinstance(value, str) or value not in PROTOCOLS:
        raise ValueError(
            f"{what} must be one of {list(PROTOCOLS)} ('F': full reference, the v1.4 protocol; "
            f"'P': the pure product state without a reference), got {value!r}"
        )
    return value


def dynamics_version(protocol: str = "F") -> str:
    """The dynamics version of a protocol: :data:`DYNAMICS_VERSION` for ``"F"``,
    :data:`PRODUCT_DYNAMICS_VERSION` for ``"P"``."""
    return DYNAMICS_VERSION if check_protocol(protocol) == "F" else PRODUCT_DYNAMICS_VERSION


@dataclass(frozen=True, order=True)
class TrajectoryKey:
    """The identity of one monitored trajectory, $(\\text{experiment}, n, \\beta, p, g, c, r)$ (v1.4 §8).

    ``beta`` and ``p`` are tokens (anything :meth:`Token.parse` accepts is converted).
    ``r`` is the measurement-replica label; the native engine draws gates and masks from one
    stream, so the study uses $r = 0$ only (design D1) and the field keeps its place for a
    future versioned extension. ``protocol`` is the initial condition (:data:`PROTOCOLS`, design
    §12): ``"F"`` (the default, the v1.4 protocol with its full reference) or ``"P"`` (the pure
    product state); it is the last field so that the seven-field form of v1.4 keeps meaning the
    F trajectory. The key is frozen, hashable and has the canonical string form
    ``"<id>:n=16:beta=0.1:p=1/3:g=3:c=1:r=0"`` for protocol F and
    ``"<id>:protocol=P:n=16:beta=0.1:p=1/3:g=3:c=1:r=0"`` otherwise (:meth:`parse` inverts it).
    """

    experiment_id: str
    n: int
    beta: Token
    p: Token
    g: int
    c: int
    r: int = 0
    protocol: str = "F"

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_id", check_experiment_id(self.experiment_id))
        object.__setattr__(self, "n", _integer(self.n, "n", minimum=1))
        object.__setattr__(self, "beta", Token.parse(self.beta, what="beta"))
        object.__setattr__(self, "p", Token.parse(self.p, what="p"))
        for name in ("g", "c", "r"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=0))
        check_protocol(self.protocol)

    def __str__(self) -> str:
        head = self.experiment_id
        if self.protocol != "F":
            head += f":protocol={self.protocol}"
        return f"{head}:n={self.n}:beta={self.beta}:p={self.p}:g={self.g}:c={self.c}:r={self.r}"

    @classmethod
    def parse(cls, text: str) -> TrajectoryKey:
        """The key whose canonical string is ``text``."""
        parts = text.split(":") if isinstance(text, str) else []
        names = ("n", "beta", "p", "g", "c", "r")
        protocol = "F"
        if len(parts) == 8 and parts[1].startswith("protocol="):
            protocol = parts.pop(1).partition("=")[2]
            if protocol == "F" or protocol not in PROTOCOLS:
                raise ValueError(
                    f"not a trajectory key: the protocol part names {list(PROTOCOLS[1:])} (an F "
                    f"key has no protocol part), got {text!r}"
                )
        if len(parts) != 7:
            raise ValueError(
                f"not a trajectory key (expected '<id>:n=..:beta=..:p=..:g=..:c=..:r=..', or "
                f"'<id>:protocol=P:n=..:...'): {text!r}"
            )
        fields: dict[str, Any] = {"protocol": protocol}
        for name, part in zip(names, parts[1:], strict=True):
            key, _, value = part.partition("=")
            if key != name or not value:
                raise ValueError(
                    f"not a trajectory key: expected '{name}=...' but got {part!r} in {text!r}"
                )
            fields[name] = value
        for name in ("n", "g", "c", "r"):
            if not re.fullmatch(r"[0-9]+", fields[name]):
                raise ValueError(
                    f"not a trajectory key: {name} must be a non-negative integer in {text!r}"
                )
            fields[name] = int(fields[name])
        return cls(parts[0], **fields)

    def stream_index(self, stride: int) -> int:
        """The native trajectory index $u = g s + c$ of this key for the circuit index stride $s$."""
        return stream_index(self.g, self.c, stride)


def graph_seed(
    master_graph_seed: int, n: int, beta_token: Token | str | float, g: int
) -> tuple[int, int, int, int, int, int]:
    r"""The seed of graph $g$ of the condition $(n, \beta)$ (v1.4 §7.2, design §3, ruling R-SEED).

    The six 32-bit words ``(GRAPH_SEED_TAG, master_graph_seed, n, lo, hi, g)`` with
    ``(lo, hi) = token_words(beta_token)``. It does not depend on the measurement rate $p$, so
    one collection per $(n, \beta)$ serves every rate (v1.4 couples conditions across $p$
    only); it does depend on $n$ and $\beta$, so the collections of different sizes and of
    different rewiring probabilities are independent samples, as finite-size fits assume. The
    tuple always has six words, so the zero padding of ``SeedSequence`` cannot identify two
    of them, and the tag keeps it away from every :func:`run_seed`.
    """
    master = _seed_word(master_graph_seed, "master_graph_seed")
    size = _seed_word(n, "n")
    beta = Token.parse(beta_token, what="beta")
    label = _seed_word(g, "g")
    lo, hi = token_words(beta)
    return (GRAPH_SEED_TAG, master, size, lo, hi, label)


def run_seed(
    master_circuit_seed: int,
    n: int,
    beta_token: Token | str | float,
    gate_mode: str,
    p_token: Token | str | float | None = None,
) -> tuple[int, ...]:
    r"""The seed passed to ``MonitoredCircuit.run`` for the condition $(n, \beta, p)$ (design §3).

    * ``"shared"``: ``(CIRCUIT_SEED_TAG, master_circuit_seed, n, lo, hi)`` with
      ``(lo, hi) = token_words(beta_token)``: five words, the same for every rate, so trajectory
      $u$ draws the same gate sequence and the same mask uniforms at every $p$ and the masks are
      nested in $p$ (v1.4 §7.3). ``p_token`` is checked when given and otherwise ignored.
    * ``"independent"``: the same five words followed by ``token_words(p_token)``: seven words,
      different for every rate, so gates and masks are independent across rates and nothing is
      nested. ``p_token`` is required.

    The seeds of different $(n, \beta)$ differ (the conditions are independent samples), and
    the tag keeps every run seed away from every :func:`graph_seed`, so equal master seeds are
    harmless. Within a gate mode the length is fixed (5 or 7, above the pool size 4 of
    ``SeedSequence``), so zero padding cannot identify two seeds.
    """
    master = _seed_word(master_circuit_seed, "master_circuit_seed")
    size = _seed_word(n, "n")
    beta = Token.parse(beta_token, what="beta")
    if gate_mode not in GATE_MODES:
        raise ValueError(f"gate_mode must be one of {list(GATE_MODES)}, got {gate_mode!r}")
    rate = None if p_token is None else Token.parse(p_token, what="p")  # checked in every mode
    base = (CIRCUIT_SEED_TAG, master, size, *token_words(beta))
    if gate_mode == "shared":
        return base
    if rate is None:
        raise ValueError(
            "run_seed with gate_mode='independent' needs p_token: the seed names the rate"
        )
    return (*base, *token_words(rate))


_MAX_INDEX = 2**63 - 1  # the engine takes int64 trajectory indices


def stream_index(g: int, c: int, stride: int) -> int:
    """The native trajectory index $u = g\\, s + c$ of circuit $c$ on graph $g$ (v1.4 §7.2, design D2).

    ``stride`` $s$ is frozen in the manifest and $c < s$. The engine runs with
    ``circuits_per_graph = s``, so trajectory $u$ uses graph $u // s = g$ and the circuit
    stream $u$. $u$ depends on $(g, c, s)$ only: adding graphs or circuits (up to $s$) never
    changes the index, hence the circuit, of an existing trajectory.
    """
    g = _integer(g, "g", minimum=0)
    stride = _integer(stride, "circuit index stride", minimum=1)
    c = _integer(c, "c", minimum=0)
    if c >= stride:
        raise ValueError(
            f"circuit label c = {c} must be below the circuit index stride {stride}: the stride "
            "is frozen by the first manifest revision (it fixes every trajectory index), so "
            "choose it at least as large as the most circuits per graph the study may ever use"
        )
    u = g * stride + c
    if u > _MAX_INDEX:
        raise ValueError(f"trajectory index {u} does not fit the engine's int64 indices")
    return u


def split_stream_index(u: int, stride: int) -> tuple[int, int]:
    """The inverse of :func:`stream_index`: $(g, c) = (u // s, u \\bmod s)$."""
    u = _integer(u, "u", minimum=0)
    stride = _integer(stride, "circuit index stride", minimum=1)
    g, c = divmod(u, stride)
    return g, c


# ---------------------------------------------------------------------------
# canonical bytes
# ---------------------------------------------------------------------------


def _canonical(obj: Any, path: str) -> Any:
    if isinstance(obj, Token):
        return obj.token
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            raise ValueError(f"{path}: JSON has no {x}; use a finite number")
        return 0.0 if x == 0 else x  # no negative zero
    if isinstance(obj, Fraction):
        raise ValueError(f"{path}: a Fraction is not JSON; use a Token (its canonical string)")
    if isinstance(obj, Mapping):
        out = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: object keys must be strings, got {key!r}")
            out[key] = _canonical(value, f"{path}.{key}")
        return out
    if isinstance(obj, np.ndarray):
        return _canonical(obj.tolist(), path)
    if isinstance(obj, (list, tuple)):
        return [_canonical(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    raise ValueError(f"{path}: cannot write a {type(obj).__name__} as JSON")


def canonical_json(obj: Any) -> str:
    """The canonical JSON text of ``obj``: sorted keys, no spaces, ASCII, finite numbers.

    Tokens are written as their canonical strings (never as floats); numpy scalars and arrays
    as plain JSON numbers and lists; tuples as lists; ``-0.0`` as ``0.0``. NaN, infinities,
    fractions, bytes and non-string keys are refused. The text of equal data is equal
    whatever the key order or whitespace of the source, so its SHA-256 identifies the data.
    """
    return json.dumps(
        _canonical(obj, "$"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_hex(data: bytes) -> str:
    """The SHA-256 digest of ``data`` as 64 lower-case hexadecimal digits."""
    return hashlib.sha256(data).hexdigest()


def json_digest(obj: Any) -> str:
    """``sha256_hex`` of the UTF-8 bytes of ``canonical_json(obj)``."""
    return sha256_hex(canonical_json(obj).encode("utf-8"))


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------


def circulant_edges(n: int) -> np.ndarray:
    """The edges of $C(n, 2)$ (v1.4 §2.1), sorted, ``a < b``, as an ``(m, 2)`` int64 array.

    $E_0(n) = \\{\\{i, (i + d) \\bmod n\\} : i \\in V,\\ d \\in \\{1, 2\\},\\ (i + d) \\bmod n \\ne i\\}$
    with duplicates identified: $\\min(4, n - 1) n / 2$ edges, which is $K_n$ for $n \\le 5$ and
    the degree-four ring with $2n$ edges for $n \\ge 5$.
    """
    n = _integer(n, "n", minimum=1)
    pairs = {
        (min(i, (i + d) % n), max(i, (i + d) % n))
        for i in range(n)
        for d in (1, 2)
        if (i + d) % n != i
    }
    return np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def original_edge_list(n: int) -> list[tuple[int, int]]:
    """The oriented original edges $L_n$ of v1.4 §2.2 in the native offset-major order.

    $d = 1$ then $d = 2$ (outer loop), $i = 0, \\ldots, n - 1$ (inner), appending
    $(a, b) = (i, (i + d) \\bmod n)$; the retained endpoint of a rewired edge is $a$. For
    $n \\ge 5$ the $2n$ pairs are distinct as unordered pairs (the invariant that makes the
    "absent original edge" abort of §2.2 unreachable).
    """
    n = _integer(n, "n", minimum=5)
    return [(i, (i + d) % n) for d in (1, 2) for i in range(n)]


def edge_digest(edges: np.ndarray, n: int) -> str:
    """The SHA-256 digest of a canonical edge list (design §3).

    The bytes are ``b"SGF2EDGES1\\0"``, then $n$ as an unsigned 32-bit little-endian integer,
    then the edge array ``(m, 2)`` as signed 32-bit little-endian integers in row-major order.
    ``edges`` must be canonical: ``a < b`` in every row and the rows strictly increasing in
    lexicographic order (what ``graphs.from_edges`` returns).
    """
    n = _integer(n, "n", minimum=1)
    arr = np.asarray(edges)
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"edges must be an (m, 2) integer array, got shape {arr.shape} and dtype {arr.dtype}"
        )
    if arr.size:
        if arr.min() < 0 or arr.max() >= min(n, 2**31):
            raise ValueError(f"edge endpoints must lie in [0, {n})")
        if not np.all(arr[:, 0] < arr[:, 1]):
            raise ValueError("edges must be oriented a < b")
        a, b = arr[:, 0], arr[:, 1]
        if not np.all((a[1:] > a[:-1]) | ((a[1:] == a[:-1]) & (b[1:] > b[:-1]))):
            raise ValueError("edges must be sorted in lexicographic order without repeats")
    payload = EDGE_DIGEST_MAGIC + struct.pack("<I", n) + arr.astype("<i4").tobytes()
    return sha256_hex(payload)


def validate_graph(edges: np.ndarray, n: int, beta: Token | str | float) -> None:
    """Raise ``ValueError`` unless ``edges`` is a possible $G_g \\sim \\mu_{n,\\beta}$ (v1.4 §2.2, §7.4).

    Checks, in this order (each is a property of every graph the rewiring law produces for
    $n \\ge 5$; the proofs are in ``docs/study/01-ensemble-and-sampling.md``):

    1. canonical form: an ``(m, 2)`` integer array, ``0 <= a < b < n``, rows strictly
       increasing (so no loop and no repeated edge: the graph is simple);
    2. $m = 2n$ (the edge count of $C(n, 2)$ is preserved);
    3. every vertex has degree at least 2 (it keeps the two edges it retains as endpoint $a$);
    4. $\\beta = 0$: the edge set is exactly $C(n, 2)$.

    For $n = 5$ the checks leave only $K_5$ (ten distinct pairs of five vertices), the only
    graph the law can produce there at any $\\beta$: no rewiring target exists.
    """
    n = _integer(n, "n", minimum=5)
    token = Token.parse(beta, what="beta")
    arr = np.asarray(edges)
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("graph: edges must be an (m, 2) integer array")
    if arr.size and (arr.min() < 0 or arr.max() >= n):
        raise ValueError(f"graph: an edge endpoint lies outside [0, {n})")
    if arr.size and np.any(arr[:, 0] == arr[:, 1]):
        raise ValueError("graph: a self-loop is present")
    if arr.size and np.any(arr[:, 0] > arr[:, 1]):
        raise ValueError("graph: edges must be oriented a < b")
    if len(arr) > 1:
        a, b = arr[:, 0], arr[:, 1]
        if not np.all((a[1:] > a[:-1]) | ((a[1:] == a[:-1]) & (b[1:] >= b[:-1]))):
            raise ValueError("graph: edges must be sorted in lexicographic order")
        if np.any((a[1:] == a[:-1]) & (b[1:] == b[:-1])):
            raise ValueError("graph: a repeated (parallel) edge is present")
    if len(arr) != 2 * n:
        raise ValueError(f"graph: expected 2n = {2 * n} edges, got {len(arr)}")
    if int(graphs.degrees(arr, n).min()) < 2:
        raise ValueError("graph: a vertex has degree below 2 (each keeps its two retained edges)")
    if token.value == 0 and not np.array_equal(arr, circulant_edges(n)):
        raise ValueError("graph: beta = 0 must give the circulant graph C(n, 2) exactly")


@dataclass(frozen=True, eq=False)
class GraphRecord:
    """The record of graph $G_g$ for $(n, \\beta, g)$ (v1.4 §2.2 "Tracking", §8; design §3).

    * ``n``, ``beta`` (a :class:`Token`), ``g``: the identity;
    * ``seed``: ``(GRAPH_SEED_TAG, master_graph_seed, n, lo, hi, g)`` (:func:`graph_seed`), the
      seed ``graphs.watts_strogatz`` was called with;
    * ``generator``: :data:`GRAPH_GENERATOR_VERSION`; ``edge_order``: :data:`ORIGINAL_EDGE_ORDER`;
    * ``edges``: the canonical sorted edge list (read-only ``(2n, 2)`` int64);
    * ``edge_digest``: :func:`edge_digest` of ``edges``.

    Two records are equal when their identity, seed, generator and digest are. :meth:`verify`
    re-checks the digest and the graph invariants; :meth:`regenerate` rebuilds the graph from
    the seed and requires the same digest (bitwise replay of the graph).
    """

    n: int
    beta: Token
    g: int
    seed: tuple[int, ...]
    generator: str
    edge_order: str
    edges: np.ndarray
    edge_digest: str

    def _key(self) -> tuple[Any, ...]:
        return (
            self.n,
            self.beta,
            self.g,
            self.seed,
            self.generator,
            self.edge_order,
            self.edge_digest,
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GraphRecord) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    @property
    def num_edges(self) -> int:
        return int(len(self.edges))

    def verify(self) -> None:
        """Raise ``ValueError`` if the digest does not match the edges or the graph is invalid."""
        validate_graph(self.edges, self.n, self.beta)
        if edge_digest(self.edges, self.n) != self.edge_digest:
            raise ValueError(
                f"graph record (n={self.n}, beta={self.beta}, g={self.g}): the edge list does not "
                "match its digest"
            )

    def regenerate(self) -> GraphRecord:
        """Rebuild the graph from its seed and return the new record, which must equal this one.

        The master graph seed is the second word of ``seed``; everything else is recomputed
        from ``(n, beta, g)``, so a record whose seed tuple, generator or digest was altered
        does not reproduce.
        """
        if len(self.seed) != 6:
            raise ValueError(
                f"graph record (n={self.n}, beta={self.beta}, g={self.g}) does not reproduce: its "
                f"seed must be the six words of graph_seed(...), got {self.seed!r}"
            )
        again = build_graph(self.n, self.beta, self.g, self.seed[1])
        if again != self:
            raise ValueError(
                f"graph record (n={self.n}, beta={self.beta}, g={self.g}) does not reproduce: "
                f"digest {self.edge_digest} but regenerating gives {again.edge_digest} "
                f"(generator {self.generator!r} now {again.generator!r})"
            )
        return again

    def to_dict(self, *, include_edges: bool = False) -> dict[str, Any]:
        """The record as plain JSON data (the edge list only when asked)."""
        out: dict[str, Any] = {
            "n": self.n,
            "beta": self.beta.token,
            "g": self.g,
            "seed": list(self.seed),
            "generator": self.generator,
            "edge_order": self.edge_order,
            "num_edges": self.num_edges,
            "edge_digest": self.edge_digest,
        }
        if include_edges:
            out["edges"] = self.edges.tolist()
        return out


def build_graph(
    n: int, beta_token: Token | str | float, g: int, master_graph_seed: int
) -> GraphRecord:
    """Graph $G_g \\sim \\mu_{n,\\beta}$ of $(n, \\beta, g)$ as a validated :class:`GraphRecord`.

    The graph is ``graphs.watts_strogatz(n, 4, float(beta), seed=graph_seed(master_graph_seed,
    n, beta, g))``: the sequential rewiring of $C(n, 2)$ of v1.4 §2.2, each of the $2n$ original
    edges, in the offset-major order, rewired with probability $\\beta$ to a target drawn
    uniformly from the vertices that are neither the retained endpoint nor its neighbors. The
    result is then checked by :func:`validate_graph` (simple, $2n$ edges, $\\beta = 0$ gives
    $C(n, 2)$, $n = 5$ gives $K_5$) and recorded with the digest of its canonical edge list.
    $n \\ge 5$ (the native generator needs $k = 4 < n$). Nothing here depends on the rate $p$.
    """
    n = _integer(n, "n", minimum=5)
    beta = Token.parse(beta_token, what="beta")
    label = _seed_word(g, "g")
    seed = graph_seed(master_graph_seed, n, beta, label)
    edges = graphs.watts_strogatz(n, 4, beta.float64, seed=seed)
    edges = np.ascontiguousarray(edges, dtype=np.int64)
    validate_graph(edges, n, beta)
    digest = edge_digest(edges, n)
    edges.flags.writeable = False
    return GraphRecord(
        n=n,
        beta=beta,
        g=label,
        seed=seed,
        generator=GRAPH_GENERATOR_VERSION,
        edge_order=ORIGINAL_EDGE_ORDER,
        edges=edges,
        edge_digest=digest,
    )
