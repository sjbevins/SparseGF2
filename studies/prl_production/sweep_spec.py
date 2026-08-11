"""Immutable, graph-family-agnostic specifications for production sweeps.

This module defines identities and work decomposition only.  It deliberately
does not construct graphs, write the graph-registry database, or run a
simulation.  The separation lets a proposed production run be reviewed and
fingerprinted before any expensive or stateful work begins.

Graph cells are the Cartesian product of ``sizes`` and named parameter axes.
Every identity is based on tagged, canonical JSON, so parameter-map insertion
order and axis/value declaration order cannot change a collection.  Integer
``1`` and floating-point ``1.0`` remain distinct values, and non-finite floats
are rejected.

The raw single-reference work unit is one ``(graph cell, p)`` point containing
an array indexed as ``[graph_index, circuit_index]``.  That layout preserves
per-graph statistics while keeping the number of external artifacts bounded.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from pathlib import PurePosixPath

SPECIFICATION_VERSION = 1
GRAPH_SEED_DERIVATION = "sha256_graph_cell_v1"
TRAJECTORY_SEED_DERIVATION = "sha256_single_ref_stream_v2"
RAW_TAU_SCHEMA_VERSION = 1
TAU_CENSORED = -1
TAU_INCOMPLETE = -2

_MAX_SEED = (1 << 63) - 1
_MAX_INT32 = (1 << 31) - 1
_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PARAMETER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")

type ParameterValue = bool | int | float | str | None
type DecimalInput = Decimal | int | float | str


def _strict_integer(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = "" if maximum is None else f" and <= {maximum}"
        raise ValueError(f"{name} must be >= {minimum}{upper}; got {value}")
    return value


def _key(value: object, name: str) -> str:
    if not isinstance(value, str) or _KEY_RE.fullmatch(value) is None:
        raise ValueError(
            f"{name} must contain 1-128 ASCII letters, digits, '.', '_', ':', or '-', "
            "and must start with a letter"
        )
    return value


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be nonempty text without surrounding whitespace")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain NUL characters")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _normalize_parameter(value: object, name: str) -> ParameterValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        real_value = float(value)
        if not math.isfinite(real_value):
            raise ValueError(f"{name} must not contain NaN or infinity")
        return 0.0 if real_value == 0.0 else real_value
    raise TypeError(f"{name} values must be JSON scalar values")


def _tagged_parameter(value: ParameterValue) -> dict[str, object]:
    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, bool):
        return {"type": "boolean", "value": value}
    if isinstance(value, int):
        return {"type": "integer", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float64", "value": value.hex()}
    return {"type": "string", "value": value}


def _parameter_token(value: ParameterValue) -> str:
    return _canonical_json(_tagged_parameter(value))


def _decimal(value: DecimalInput, name: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a decimal number")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a decimal number") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be finite")
    return Decimal(0) if parsed == 0 else parsed.normalize()


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


@dataclass(frozen=True, slots=True)
class ParameterAxis:
    """One named graph-parameter axis with canonical, unique values."""

    name: str
    values: tuple[ParameterValue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _PARAMETER_RE.fullmatch(self.name) is None:
            raise ValueError(
                "parameter names must start with a letter and contain only letters, "
                "digits, and underscores"
            )
        normalized = tuple(
            _normalize_parameter(value, f"parameter axis {self.name!r}") for value in self.values
        )
        if not normalized:
            raise ValueError(f"parameter axis {self.name!r} must contain at least one value")
        tokens = tuple(_parameter_token(value) for value in normalized)
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"parameter axis {self.name!r} contains duplicate values")
        ordered = tuple(value for _, value in sorted(zip(tokens, normalized, strict=True)))
        object.__setattr__(self, "values", ordered)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "values": [_tagged_parameter(value) for value in self.values],
        }


@dataclass(frozen=True, slots=True)
class ParameterPoint:
    """One immutable point in a named graph-parameter grid."""

    items: tuple[tuple[str, ParameterValue], ...]

    def __post_init__(self) -> None:
        normalized: list[tuple[str, ParameterValue]] = []
        for name, value in self.items:
            if not isinstance(name, str) or _PARAMETER_RE.fullmatch(name) is None:
                raise ValueError(f"invalid graph-parameter name {name!r}")
            normalized.append((name, _normalize_parameter(value, f"parameter {name!r}")))
        normalized.sort(key=lambda item: item[0])
        names = tuple(name for name, _ in normalized)
        if len(set(names)) != len(names):
            raise ValueError("a parameter point must not repeat a parameter name")
        object.__setattr__(self, "items", tuple(normalized))

    @property
    def values(self) -> dict[str, ParameterValue]:
        return dict(self.items)

    def canonical_payload(self) -> dict[str, object]:
        return {name: _tagged_parameter(value) for name, value in self.items}

    @property
    def specification_sha256(self) -> str:
        return _payload_sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class GraphParameterGrid:
    """A Cartesian product of named graph-parameter axes."""

    axes: tuple[ParameterAxis, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(axis, ParameterAxis) for axis in self.axes):
            raise TypeError("axes must contain ParameterAxis values")
        ordered = tuple(sorted(self.axes, key=lambda axis: axis.name))
        names = tuple(axis.name for axis in ordered)
        if len(set(names)) != len(names):
            raise ValueError("graph-parameter axis names must be unique")
        object.__setattr__(self, "axes", ordered)

    @property
    def n_points(self) -> int:
        return math.prod(len(axis.values) for axis in self.axes)

    def points(self) -> Iterator[ParameterPoint]:
        names = tuple(axis.name for axis in self.axes)
        products = itertools.product(*(axis.values for axis in self.axes))
        for values in products:
            yield ParameterPoint(tuple(zip(names, values, strict=True)))

    def resolve(self, parameters: Mapping[str, object]) -> ParameterPoint:
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        expected = {axis.name for axis in self.axes}
        if set(parameters) != expected:
            raise ValueError(
                f"parameters must have exactly the keys {sorted(expected)}; "
                f"got {sorted(parameters)}"
            )
        point = ParameterPoint(tuple((name, parameters[name]) for name in sorted(parameters)))
        allowed = {
            axis.name: {_parameter_token(value) for value in axis.values} for axis in self.axes
        }
        for name, value in point.items:
            if _parameter_token(value) not in allowed[name]:
                raise ValueError(f"parameter {name!r} value {value!r} is not in the grid")
        return point

    def canonical_payload(self) -> list[dict[str, object]]:
        return [axis.canonical_payload() for axis in self.axes]


@dataclass(frozen=True, slots=True)
class GraphCellSpec:
    """One ``(n, graph_parameters)`` cell in canonical collection order."""

    collection_sha256: str
    cell_index: int
    n: int
    parameters: ParameterPoint

    def __post_init__(self) -> None:
        _sha256(self.collection_sha256, "collection_sha256")
        _strict_integer(self.cell_index, "cell_index")
        _strict_integer(self.n, "n", minimum=2, maximum=_MAX_INT32)
        if not isinstance(self.parameters, ParameterPoint):
            raise TypeError("parameters must be a ParameterPoint")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "collection_sha256": self.collection_sha256,
            "n": self.n,
            "parameters": self.parameters.canonical_payload(),
        }

    @property
    def cell_sha256(self) -> str:
        return _payload_sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class GraphSeedAssignment:
    """One indexed graph realization linked to a canonical graph cell."""

    cell: GraphCellSpec
    graph_index: int
    graph_seed: int


@dataclass(frozen=True, slots=True)
class GraphCollectionGridSpec:
    """Complete immutable recipe for a family-agnostic graph collection."""

    name: str
    graph_family: str
    generator_name: str
    generator_version: str
    sizes: tuple[int, ...]
    parameter_grid: GraphParameterGrid
    graphs_per_cell: int
    master_seed: int
    seed_derivation: str = GRAPH_SEED_DERIVATION
    specification_version: int = SPECIFICATION_VERSION

    def __post_init__(self) -> None:
        _key(self.name, "name")
        _key(self.graph_family, "graph_family")
        _nonempty_text(self.generator_name, "generator_name")
        _nonempty_text(self.generator_version, "generator_version")
        sizes = tuple(_strict_integer(n, "n", minimum=2, maximum=_MAX_INT32) for n in self.sizes)
        if not sizes or len(set(sizes)) != len(sizes):
            raise ValueError("sizes must be nonempty and unique")
        object.__setattr__(self, "sizes", tuple(sorted(sizes)))
        if not isinstance(self.parameter_grid, GraphParameterGrid):
            raise TypeError("parameter_grid must be a GraphParameterGrid")
        _strict_integer(
            self.graphs_per_cell,
            "graphs_per_cell",
            minimum=1,
            maximum=_MAX_INT32,
        )
        _strict_integer(self.master_seed, "master_seed", maximum=_MAX_SEED)
        _nonempty_text(self.seed_derivation, "seed_derivation")
        _strict_integer(self.specification_version, "specification_version", minimum=1)

    @property
    def n_cells(self) -> int:
        return len(self.sizes) * self.parameter_grid.n_points

    @property
    def n_graphs(self) -> int:
        return self.n_cells * self.graphs_per_cell

    def canonical_payload(self) -> dict[str, object]:
        return {
            "generator_name": self.generator_name,
            "generator_version": self.generator_version,
            "graph_family": self.graph_family,
            "graphs_per_cell": self.graphs_per_cell,
            "master_seed": self.master_seed,
            "name": self.name,
            "parameter_axes": self.parameter_grid.canonical_payload(),
            "seed_derivation": self.seed_derivation,
            "sizes": list(self.sizes),
            "specification_version": self.specification_version,
        }

    @property
    def specification_sha256(self) -> str:
        return _payload_sha256(self.canonical_payload())

    @property
    def collection_id(self) -> str:
        return f"{self.name}_{self.specification_sha256[:16]}"

    def cells(self) -> Iterator[GraphCellSpec]:
        collection_sha256 = self.specification_sha256
        cell_index = 0
        for n in self.sizes:
            for parameters in self.parameter_grid.points():
                yield GraphCellSpec(
                    collection_sha256=collection_sha256,
                    cell_index=cell_index,
                    n=n,
                    parameters=parameters,
                )
                cell_index += 1

    def cell(self, n: int, parameters: Mapping[str, object]) -> GraphCellSpec:
        size = _strict_integer(n, "n", minimum=2)
        if size not in self.sizes:
            raise ValueError(f"n={size} is not in this collection")
        point = self.parameter_grid.resolve(parameters)
        for cell in self.cells():
            if cell.n == size and cell.parameters == point:
                return cell
        raise AssertionError("validated graph cell was not generated")

    def graph_seed(self, cell: GraphCellSpec, graph_index: int) -> int:
        if not isinstance(cell, GraphCellSpec):
            raise TypeError("cell must be a GraphCellSpec")
        if cell.collection_sha256 != self.specification_sha256:
            raise ValueError("cell belongs to a different graph collection")
        points = tuple(self.parameter_grid.points())
        size_index, parameter_index = divmod(cell.cell_index, len(points))
        expected = (
            None
            if size_index >= len(self.sizes)
            else GraphCellSpec(
                collection_sha256=cell.collection_sha256,
                cell_index=cell.cell_index,
                n=self.sizes[size_index],
                parameters=points[parameter_index],
            )
        )
        if expected != cell:
            raise ValueError("cell is not the canonical cell at its declared cell_index")
        index = _strict_integer(graph_index, "graph_index")
        if index >= self.graphs_per_cell:
            raise ValueError(f"graph_index must be < {self.graphs_per_cell}; got {index}")
        return self._derive_graph_seed(cell, index)

    def _derive_graph_seed(self, cell: GraphCellSpec, graph_index: int) -> int:
        payload = (
            f"{self.seed_derivation}\0{self.master_seed}\0{cell.collection_sha256}\0"
            f"{cell.cell_sha256}\0{graph_index}"
        ).encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _MAX_SEED

    def records(self) -> Iterator[GraphSeedAssignment]:
        for cell in self.cells():
            for graph_index in range(self.graphs_per_cell):
                yield GraphSeedAssignment(
                    cell=cell,
                    graph_index=graph_index,
                    graph_seed=self._derive_graph_seed(cell, graph_index),
                )


@dataclass(frozen=True, slots=True)
class ProbabilityGrid:
    """An exact inclusive, uniformly spaced measurement-probability grid."""

    p_min: DecimalInput
    p_max: DecimalInput
    delta_p: DecimalInput

    def __post_init__(self) -> None:
        p_min = _decimal(self.p_min, "p_min")
        p_max = _decimal(self.p_max, "p_max")
        delta = _decimal(self.delta_p, "delta_p")
        if not Decimal(0) <= p_min <= p_max <= Decimal(1):
            raise ValueError("p_min and p_max must satisfy 0 <= p_min <= p_max <= 1")
        if delta <= 0:
            raise ValueError("delta_p must be positive")
        span = p_max - p_min
        if span and span % delta:
            raise ValueError("p_max - p_min must be an integer multiple of delta_p")
        n_points = int(span / delta) + 1
        if n_points > 1_000_000:
            raise ValueError("the p grid is unexpectedly large (>1,000,000 points)")
        decimal_values = tuple(p_min + index * delta for index in range(n_points))
        binary_values = tuple(float(value) for value in decimal_values)
        if any(right <= left for left, right in itertools.pairwise(binary_values)):
            raise ValueError(
                "adjacent p-grid values must remain distinct after binary64 conversion"
            )
        object.__setattr__(self, "p_min", p_min)
        object.__setattr__(self, "p_max", p_max)
        object.__setattr__(self, "delta_p", delta)

    @property
    def decimal_values(self) -> tuple[Decimal, ...]:
        count = int((self.p_max - self.p_min) / self.delta_p) + 1
        return tuple(self.p_min + index * self.delta_p for index in range(count))

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.decimal_values)

    @property
    def canonical_values(self) -> tuple[str, ...]:
        return tuple(_decimal_text(value) for value in self.decimal_values)

    def canonical_payload(self) -> dict[str, str]:
        return {
            "delta_p": _decimal_text(self.delta_p),
            "p_max": _decimal_text(self.p_max),
            "p_min": _decimal_text(self.p_min),
        }


@dataclass(frozen=True, slots=True)
class ScientificEnvironmentContract:
    """Dependency versions that must not change within one raw-data experiment."""

    python: str
    numpy: str
    numba: str
    bit_generator: str = "PCG64"

    def __post_init__(self) -> None:
        for name in ("python", "numpy", "numba", "bit_generator"):
            _nonempty_text(getattr(self, name), name)

    def canonical_payload(self) -> dict[str, str]:
        return {
            "bit_generator": self.bit_generator,
            "numba": self.numba,
            "numpy": self.numpy,
            "python": self.python,
        }

    @property
    def specification_sha256(self) -> str:
        return _payload_sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class SingleReferenceProtocolSpec:
    """Immutable physical protocol for raw single-reference first-passage data.

    ``p_randomness_policy="independent"`` gives every measurement rate its own
    streams.  ``"common_circuit_disorder"`` reserves paired cross-``p``
    streams; an engine must not accept that policy until measurement masks and
    outcomes are indexed independently of control flow.
    """

    n_circuits: int
    q_scramble: int
    q_max: int
    p_grid: ProbabilityGrid
    master_seed: int
    reference_system_qubit_policy: str = "fixed_last"
    p_randomness_policy: str = "independent"
    protocol_version: str = "single_ref_tau_exact_layer_v2"

    def __post_init__(self) -> None:
        _strict_integer(
            self.n_circuits,
            "n_circuits",
            minimum=1,
            maximum=_MAX_INT32,
        )
        _strict_integer(self.q_scramble, "q_scramble", maximum=_MAX_INT32)
        _strict_integer(self.q_max, "q_max", minimum=1, maximum=_MAX_INT32)
        if not isinstance(self.p_grid, ProbabilityGrid):
            raise TypeError("p_grid must be a ProbabilityGrid")
        _strict_integer(self.master_seed, "master_seed", maximum=_MAX_SEED)
        allowed = {"fixed_last", "uniform_system_qubit_per_circuit"}
        if self.reference_system_qubit_policy not in allowed:
            raise ValueError(f"reference_system_qubit_policy must be one of {sorted(allowed)}")
        randomness_policies = {"common_circuit_disorder", "independent"}
        if self.p_randomness_policy not in randomness_policies:
            raise ValueError(f"p_randomness_policy must be one of {sorted(randomness_policies)}")
        _nonempty_text(self.protocol_version, "protocol_version")

    @property
    def t_max_factor(self) -> int:
        """Backward-readable alias for the depth multiplier ``q_max``."""

        return self.q_max

    def t_max(self, n: int) -> int:
        value = self.q_max * _strict_integer(n, "n", minimum=2, maximum=_MAX_INT32)
        if value > _MAX_INT32:
            raise OverflowError("q_max*n exceeds the int32 raw-data representation")
        return value

    def scramble_gate_count(self, n: int) -> int:
        value = self.q_scramble * _strict_integer(
            n,
            "n",
            minimum=2,
            maximum=_MAX_INT32,
        )
        if value > _MAX_INT32:
            raise OverflowError("q_scramble*n exceeds the int32 raw-data representation")
        return value

    def canonical_payload(self) -> dict[str, object]:
        return {
            "clifford_sampling": "uniform_phase_free_sp4_with_replacement",
            "clifford_table_size": 720,
            "dynamic_edge_sampling": "uniform_edges_with_replacement",
            "entropy_check": "after_every_complete_layer",
            "gates_per_layer": "floor(n/2)",
            "master_seed": self.master_seed,
            "measurement": "independent_bernoulli_z_on_every_system_qubit",
            "n_circuits_per_graph_and_p": self.n_circuits,
            "p_grid": self.p_grid.canonical_payload(),
            "p_randomness_policy": self.p_randomness_policy,
            "picture": "single_ref",
            "protocol_version": self.protocol_version,
            "q_max": self.q_max,
            "q_scramble": self.q_scramble,
            "reference_system_qubit_policy": self.reference_system_qubit_policy,
            "scramble_gate_count": "q_scramble*n",
            "scramble_pair_sampling": "uniform_distinct_system_pairs_with_replacement",
            "t_max": "q_max*n",
            "tau_definition": "first_integer_layer_t_at_which_S_reference_equals_0",
            "tau_first_layer": 1,
            "trajectory_seed_derivation": TRAJECTORY_SEED_DERIVATION,
        }

    @property
    def specification_sha256(self) -> str:
        return _payload_sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class TauWorkUnit:
    """Exclusive-writer unit for one graph cell and one measurement rate."""

    experiment_sha256: str
    protocol: SingleReferenceProtocolSpec
    cell: GraphCellSpec
    graphs_per_cell: int
    p_index: int
    p_decimal: str

    def __post_init__(self) -> None:
        _sha256(self.experiment_sha256, "experiment_sha256")
        if not isinstance(self.protocol, SingleReferenceProtocolSpec):
            raise TypeError("protocol must be a SingleReferenceProtocolSpec")
        if not isinstance(self.cell, GraphCellSpec):
            raise TypeError("cell must be a GraphCellSpec")
        _strict_integer(self.graphs_per_cell, "graphs_per_cell", minimum=1)
        index = _strict_integer(self.p_index, "p_index")
        if index >= len(self.protocol.p_grid.canonical_values):
            raise ValueError("p_index is outside the protocol p grid")
        expected = self.protocol.p_grid.canonical_values[index]
        if self.p_decimal != expected:
            raise ValueError(f"p_decimal={self.p_decimal!r}, expected {expected!r}")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "cell_sha256": self.cell.cell_sha256,
            "experiment_sha256": self.experiment_sha256,
            "p_decimal": self.p_decimal,
            "p_index": self.p_index,
        }

    @property
    def work_sha256(self) -> str:
        return _payload_sha256(self.canonical_payload())

    @property
    def raw_shape(self) -> tuple[int, int]:
        return (self.graphs_per_cell, self.protocol.n_circuits)

    @property
    def artifact_relative_path(self) -> PurePosixPath:
        return PurePosixPath(
            "single_ref",
            "raw_tau",
            self.experiment_sha256,
            self.cell.cell_sha256,
            f"p{self.p_index:06d}_{self.work_sha256[:16]}.npz",
        )

    def trajectory_seed(self, graph_index: int, circuit_index: int, stream: str) -> int:
        """Return a stream seed stable under campaign renaming and grid refinement.

        The immutable graph-cell identity, exact measurement rate, circuit
        index, protocol master seed, and stream role determine the seed.  The
        enclosing experiment hash is deliberately excluded: it contains the
        full p grid, source fingerprint, and human-readable experiment name,
        none of which should change an already-defined trajectory when a grid
        is extended or a campaign is renamed.
        """
        graph = _strict_integer(graph_index, "graph_index")
        circuit = _strict_integer(circuit_index, "circuit_index")
        if graph >= self.graphs_per_cell:
            raise ValueError(f"graph_index must be < {self.graphs_per_cell}; got {graph}")
        if circuit >= self.protocol.n_circuits:
            raise ValueError(f"circuit_index must be < {self.protocol.n_circuits}; got {circuit}")
        allowed_streams = {
            "dynamic_cliffords",
            "dynamic_edges",
            "measurement_mask",
            "measurement_outcomes",
            "reference_placement",
            "scramble_cliffords",
            "scramble_pairs",
        }
        if stream not in allowed_streams:
            raise ValueError(f"stream must be one of {sorted(allowed_streams)}")
        p_component = (
            self.p_decimal
            if self.protocol.p_randomness_policy == "independent"
            else "shared_across_p"
        )
        payload = (
            f"{TRAJECTORY_SEED_DERIVATION}\0{self.protocol.master_seed}\0"
            f"{self.cell.cell_sha256}\0{p_component}\0"
            f"{graph}\0{circuit}\0{stream}"
        ).encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _MAX_SEED


@dataclass(frozen=True, slots=True)
class SingleReferenceSweepSpec:
    """Link a graph collection, code fingerprint, and single-reference protocol."""

    name: str
    graph_collection_sha256: str
    source_fingerprint_sha256: str
    environment_contract: ScientificEnvironmentContract
    protocol: SingleReferenceProtocolSpec
    specification_version: int = SPECIFICATION_VERSION

    def __post_init__(self) -> None:
        _key(self.name, "name")
        _sha256(self.graph_collection_sha256, "graph_collection_sha256")
        _sha256(self.source_fingerprint_sha256, "source_fingerprint_sha256")
        if not isinstance(self.environment_contract, ScientificEnvironmentContract):
            raise TypeError("environment_contract must be a ScientificEnvironmentContract")
        if not isinstance(self.protocol, SingleReferenceProtocolSpec):
            raise TypeError("protocol must be a SingleReferenceProtocolSpec")
        _strict_integer(self.specification_version, "specification_version", minimum=1)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "graph_collection_sha256": self.graph_collection_sha256,
            "environment_contract": self.environment_contract.canonical_payload(),
            "name": self.name,
            "protocol": self.protocol.canonical_payload(),
            "source_fingerprint_sha256": self.source_fingerprint_sha256,
            "specification_version": self.specification_version,
        }

    @property
    def specification_sha256(self) -> str:
        return _payload_sha256(self.canonical_payload())

    @property
    def experiment_id(self) -> str:
        return f"{self.name}_{self.specification_sha256[:16]}"

    def work_units(self, graphs: GraphCollectionGridSpec) -> Iterator[TauWorkUnit]:
        if not isinstance(graphs, GraphCollectionGridSpec):
            raise TypeError("graphs must be a GraphCollectionGridSpec")
        if graphs.specification_sha256 != self.graph_collection_sha256:
            raise ValueError("graph collection does not match this experiment")
        p_values = self.protocol.p_grid.canonical_values
        for cell in graphs.cells():
            for p_index, p_decimal in enumerate(p_values):
                yield TauWorkUnit(
                    experiment_sha256=self.specification_sha256,
                    protocol=self.protocol,
                    cell=cell,
                    graphs_per_cell=graphs.graphs_per_cell,
                    p_index=p_index,
                    p_decimal=p_decimal,
                )


__all__ = [
    "DecimalInput",
    "GRAPH_SEED_DERIVATION",
    "RAW_TAU_SCHEMA_VERSION",
    "SPECIFICATION_VERSION",
    "TAU_CENSORED",
    "TAU_INCOMPLETE",
    "TRAJECTORY_SEED_DERIVATION",
    "GraphCellSpec",
    "GraphCollectionGridSpec",
    "GraphParameterGrid",
    "GraphSeedAssignment",
    "ParameterAxis",
    "ParameterPoint",
    "ParameterValue",
    "ProbabilityGrid",
    "ScientificEnvironmentContract",
    "SingleReferenceProtocolSpec",
    "SingleReferenceSweepSpec",
    "TauWorkUnit",
]
