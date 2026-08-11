"""Graph-family-agnostic providers and immutable edge-bank materialization."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import marshal
import math
import platform
import re
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from studies.prl_production.graph_registry.collection import GraphCollection
from studies.prl_production.single_ref.shared_io import load_npz_snapshot, read_shared_bytes
from studies.prl_production.sweep_spec import (
    RAW_TAU_SCHEMA_VERSION,
    GraphCellSpec,
    GraphCollectionGridSpec,
    ParameterPoint,
    ParameterValue,
)

from sparsegf2.circuits import graphs as graph_module
from sparsegf2.circuits.graphs import (
    _ws_rewire_edges,
    complete_graph,
    cycle_graph,
    lattice_2d,
    newman_watts,
    path_graph,
)

from .io import array_sha256, file_sha256, write_deterministic_npz

REPO_ROOT = Path(__file__).resolve().parents[4]
EDGE_BANK_VERSION = "generic_csr_edge_bank_v1"
EDGE_BANK_VALIDATION_VERSION = "edge_bank_validation_receipt_v1"
SEALED_WS_GRAPH_FAMILY = "watts_strogatz_rewired_circulant"
_COLLECTION_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")

EdgeFactory = Callable[[int, Mapping[str, ParameterValue], int], Iterable[tuple[int, int]]]


def _exact_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]; got {parsed}")
    return parsed


def _graph_rng_environment() -> dict[str, str]:
    generator = np.random.default_rng(0)
    return {
        "bit_generator": type(generator.bit_generator).__name__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def validate_builtin_parameters(
    graph_family: str,
    n: int,
    parameters: Mapping[str, ParameterValue],
) -> None:
    """Validate every built-in graph cell without constructing a graph."""

    params = dict(parameters)
    if graph_family == "watts_strogatz":
        allowed = {"k", "beta"}
        if set(params) != allowed:
            raise ValueError(f"watts_strogatz parameters must be {sorted(allowed)}")
        k = _exact_integer(params["k"], "k")
        if k < 1 or 2 * k >= n:
            raise ValueError(f"watts_strogatz k must satisfy 1 <= k < n/2; got n={n}, k={k}")
        _probability(params["beta"], "beta")
    elif graph_family == "newman_watts":
        allowed = {"k", "p"}
        if set(params) != allowed:
            raise ValueError(f"newman_watts parameters must be {sorted(allowed)}")
        k = _exact_integer(params["k"], "k")
        if n < 3 or k < 2 or k >= n or k % 2:
            raise ValueError(f"newman_watts requires n>=3 and even 2 <= k < n; got n={n}, k={k}")
        _probability(params["p"], "p")
    elif graph_family == "cycle":
        if params:
            raise ValueError("cycle has no graph parameters")
        if n < 3:
            raise ValueError(f"cycle requires n >= 3; got n={n}")
    elif graph_family in {"path", "complete"}:
        if params:
            raise ValueError(f"{graph_family} has no graph parameters")
        if n < 2:
            raise ValueError(f"{graph_family} requires n >= 2; got n={n}")
    elif graph_family == "lattice_2d":
        allowed = {"rows", "cols"}
        if set(params) != allowed:
            raise ValueError(f"lattice_2d parameters must be {sorted(allowed)}")
        rows = _exact_integer(params["rows"], "rows")
        cols = _exact_integer(params["cols"], "cols")
        if rows < 1 or cols < 1 or rows * cols != n:
            raise ValueError(
                f"lattice_2d requires positive rows*cols equal n={n}; got {rows}*{cols}"
            )
    else:
        raise ValueError(f"unknown built-in graph family {graph_family!r}")


@dataclass(frozen=True, slots=True)
class ProviderCell:
    """One graph cell plus the information required to reconstruct its draws."""

    collection_id: str
    graph_family: str
    generator_name: str
    generator_version: str
    generator_contract_sha256: str
    spec: GraphCellSpec
    graphs_per_cell: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.collection_id, str)
            or _COLLECTION_ID_RE.fullmatch(self.collection_id) is None
        ):
            raise ValueError("collection_id must be a path-safe canonical key")
        for name in ("graph_family", "generator_name", "generator_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty text")
        if len(self.generator_contract_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.generator_contract_sha256
        ):
            raise ValueError("generator_contract_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.spec, GraphCellSpec):
            raise TypeError("spec must be a GraphCellSpec")
        if (
            isinstance(self.graphs_per_cell, bool)
            or not isinstance(self.graphs_per_cell, int)
            or self.graphs_per_cell < 1
        ):
            raise ValueError("graphs_per_cell must be a positive integer")

    @property
    def n(self) -> int:
        return self.spec.n

    @property
    def cell_sha256(self) -> str:
        return self.spec.cell_sha256

    @property
    def parameters(self) -> dict[str, ParameterValue]:
        return self.spec.parameters.values


@runtime_checkable
class GraphProvider(Protocol):
    """Minimal provider contract consumed by the raw-tau workflow."""

    @property
    def collection_id(self) -> str: ...

    @property
    def collection_sha256(self) -> str: ...

    def cells(self) -> tuple[ProviderCell, ...]: ...

    def graph_seeds(self, cell: ProviderCell) -> tuple[int, ...]: ...

    def build_edges(self, cell: ProviderCell, graph_seed: int) -> Iterable[tuple[int, int]]: ...


def _topology_edges(topology) -> tuple[tuple[int, int], ...]:
    return tuple((int(u), int(v)) for u, v in topology.edges)


def builtin_graph_factory(
    graph_family: str,
) -> EdgeFactory:
    """Return a strict adapter for a built-in SparseGF2 graph family."""

    def factory(n: int, parameters: Mapping[str, ParameterValue], seed: int):
        params = dict(parameters)
        validate_builtin_parameters(graph_family, n, params)
        graph_seed = _exact_integer(seed, "graph seed")
        if graph_seed < 0:
            raise ValueError(f"graph seed must be nonnegative; got {graph_seed}")
        if graph_family == "watts_strogatz":
            k = _exact_integer(params["k"], "k")
            return tuple(
                _ws_rewire_edges(
                    n,
                    k,
                    _probability(params["beta"], "beta"),
                    graph_seed,
                )
            )
        elif graph_family == "newman_watts":
            topology = newman_watts(
                n,
                k=_exact_integer(params["k"], "k"),
                p=_probability(params["p"], "p"),
                seed=graph_seed,
            )
        elif graph_family == "cycle":
            topology = cycle_graph(n)
        elif graph_family == "path":
            topology = path_graph(n)
        elif graph_family == "complete":
            topology = complete_graph(n)
        elif graph_family == "lattice_2d":
            rows = _exact_integer(params["rows"], "rows")
            cols = _exact_integer(params["cols"], "cols")
            topology = lattice_2d(rows, cols)
        else:
            raise ValueError(
                f"unknown built-in graph family {graph_family!r}; pass an explicit factory"
            )
        return _topology_edges(topology)

    # Validate eagerly rather than failing after an expensive campaign is planned.
    if graph_family not in {
        "watts_strogatz",
        "newman_watts",
        "cycle",
        "path",
        "complete",
        "lattice_2d",
    }:
        raise ValueError(f"unknown built-in graph family {graph_family!r}")
    return factory


def _builtin_generator_contract_sha256(graph_family: str) -> str:
    """Bind cached edge banks to the exact built-in adapter and graph code."""

    digest = hashlib.sha256(b"raw_tau_builtin_graph_contract_v1\0")
    digest.update(graph_family.encode("ascii"))
    digest.update(
        json.dumps(
            _graph_rng_environment(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for path in (Path(__file__).resolve(), Path(graph_module.__file__).resolve()):
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _custom_factory_contract_sha256(factory: EdgeFactory) -> str:
    """Fingerprint the explicit test/extension hook without claiming plugin stability."""

    code = getattr(factory, "__code__", None)
    if code is None or not inspect.isfunction(factory):
        raise TypeError("an explicit graph factory must be a Python function")
    if factory.__closure__:
        raise ValueError("explicit graph factories must not close over mutable state")
    digest = hashlib.sha256(b"raw_tau_custom_graph_factory_contract_v1\0")
    digest.update(str(factory.__module__).encode("utf-8"))
    digest.update(str(factory.__qualname__).encode("utf-8"))
    digest.update(marshal.dumps(code))
    digest.update(repr(factory.__defaults__).encode("utf-8"))
    digest.update(repr(factory.__kwdefaults__).encode("utf-8"))
    return digest.hexdigest()


class GridGraphProvider:
    """Provider for a new Cartesian graph collection specification."""

    def __init__(self, spec: GraphCollectionGridSpec, factory: EdgeFactory | None = None) -> None:
        if not isinstance(spec, GraphCollectionGridSpec):
            raise TypeError("spec must be a GraphCollectionGridSpec")
        self.spec = spec
        if factory is None:
            for cell in spec.cells():
                validate_builtin_parameters(spec.graph_family, cell.n, cell.parameters.values)
            self._factory = builtin_graph_factory(spec.graph_family)
            self._generator_contract_sha256 = _builtin_generator_contract_sha256(spec.graph_family)
        else:
            self._factory = factory
            self._generator_contract_sha256 = _custom_factory_contract_sha256(factory)

    @property
    def collection_id(self) -> str:
        return self.spec.collection_id

    @property
    def collection_sha256(self) -> str:
        return self.spec.specification_sha256

    def cells(self) -> tuple[ProviderCell, ...]:
        return tuple(
            ProviderCell(
                collection_id=self.collection_id,
                graph_family=self.spec.graph_family,
                generator_name=self.spec.generator_name,
                generator_version=self.spec.generator_version,
                generator_contract_sha256=self._generator_contract_sha256,
                spec=cell,
                graphs_per_cell=self.spec.graphs_per_cell,
            )
            for cell in self.spec.cells()
        )

    def graph_seeds(self, cell: ProviderCell) -> tuple[int, ...]:
        self._validate_cell(cell)
        return tuple(
            self.spec.graph_seed(cell.spec, graph_index)
            for graph_index in range(self.spec.graphs_per_cell)
        )

    def build_edges(self, cell: ProviderCell, graph_seed: int) -> Iterable[tuple[int, int]]:
        self._validate_cell(cell)
        seed = _exact_integer(graph_seed, "graph seed")
        if seed < 0:
            raise ValueError(f"graph seed must be nonnegative; got {seed}")
        return self._factory(cell.n, cell.parameters, seed)

    def _validate_cell(self, cell: ProviderCell) -> None:
        if not isinstance(cell, ProviderCell):
            raise TypeError("cell must be a ProviderCell")
        if (
            cell.collection_id != self.collection_id
            or cell.spec.collection_sha256 != self.collection_sha256
        ):
            raise ValueError("cell belongs to a different graph collection")
        canonical = self.cells()
        if cell.spec.cell_index >= len(canonical) or cell != canonical[cell.spec.cell_index]:
            raise ValueError("cell is not a canonical member of this graph collection")


class WattsStrogatzRegistryProvider:
    """Adapter for the sealed 350,000-draw Watts-Strogatz SQLite registry."""

    def __init__(self, manifest_path: str | Path) -> None:
        path = Path(manifest_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "collection_id",
            "specification_sha256",
            "database",
            "database_sha256",
            "generator_source_sha256",
            "graph_k",
            "graphs_per_cell",
            "master_seed",
            "mean_degree",
            "schema_version",
            "seed_derivation",
            "sizes",
            "betas",
            "beta_keys",
            "environment",
            "validation",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"{path}: manifest is missing {sorted(missing)}")
        recorded_environment = payload["environment"]
        if not isinstance(recorded_environment, dict):
            raise ValueError(f"{path}: environment must be a JSON object")
        current_graph_environment = _graph_rng_environment()
        if current_graph_environment["bit_generator"] != "PCG64":
            raise ValueError("Watts-Strogatz reconstruction requires NumPy PCG64")
        for key in ("python", "numpy"):
            if recorded_environment.get(key) != current_graph_environment[key]:
                raise ValueError(
                    f"{path}: recorded graph {key}={recorded_environment.get(key)!r}, "
                    f"current {key}={current_graph_environment[key]!r}; reconstruct "
                    "the sealed collection only in its recorded RNG environment"
                )
        database = Path(payload["database"])
        if not database.is_absolute():
            database = REPO_ROOT / database
        if not database.is_file():
            raise FileNotFoundError(database)
        generator_path = Path(graph_module.__file__).resolve()
        if file_sha256(generator_path) != payload["generator_source_sha256"]:
            raise ValueError(
                "the graph-generator source differs from the sealed collection; "
                "use the recorded source revision before reconstructing graphs"
            )
        beta_keys = tuple(_exact_integer(value, "beta_key") for value in payload["beta_keys"])
        supplied_betas = tuple(float(value) for value in payload["betas"])
        if len(supplied_betas) != len(beta_keys) or not supplied_betas:
            raise ValueError(f"{path}: beta and beta-key grids are inconsistent")
        betas = tuple(key / 1_000_000_000 for key in beta_keys)
        if supplied_betas != betas:
            raise ValueError(f"{path}: beta values must equal beta_key / 1e9 exactly")
        sizes = tuple(_exact_integer(n, "size") for n in payload["sizes"])
        if not sizes or len(set(sizes)) != len(sizes):
            raise ValueError(f"{path}: size grid must be nonempty and unique")

        self.manifest_path = path
        self._payload = payload
        self._collection = GraphCollection(database, str(payload["collection_id"]))
        with self._collection.registry.read_connection() as connection:
            ensemble = connection.execute(
                "SELECT graph_family, metadata_json FROM ensembles WHERE ensemble_key = ?",
                (str(payload["collection_id"]),),
            ).fetchone()
        if ensemble is None or str(ensemble["graph_family"]) != SEALED_WS_GRAPH_FAMILY:
            actual = None if ensemble is None else str(ensemble["graph_family"])
            raise ValueError(
                f"{database}: graph_family={actual!r}, expected the sealed "
                f"{SEALED_WS_GRAPH_FAMILY!r} ensemble"
            )
        ensemble_metadata = json.loads(str(ensemble["metadata_json"]))
        sealed_specification = ensemble_metadata.get("specification")
        if not isinstance(sealed_specification, dict):
            raise ValueError(f"{database}: ensemble metadata lacks its specification")
        graph_k = _exact_integer(payload["graph_k"], "graph_k")
        graphs_per_cell = _exact_integer(payload["graphs_per_cell"], "graphs_per_cell")
        manifest_projection = {
            "beta_keys": list(beta_keys),
            "graph_k": graph_k,
            "graphs_per_cell": graphs_per_cell,
            "master_seed": _exact_integer(payload["master_seed"], "master_seed"),
            "schema_version": _exact_integer(payload["schema_version"], "schema_version"),
            "seed_derivation": payload["seed_derivation"],
            "sizes": list(sizes),
        }
        sealed_projection = {key: sealed_specification.get(key) for key in manifest_projection}
        if manifest_projection != sealed_projection:
            raise ValueError(
                f"{path}: reconstruction fields conflict with the sealed ensemble specification"
            )
        if ensemble_metadata.get("generator_source_sha256") != payload["generator_source_sha256"]:
            raise ValueError(f"{path}: generator source conflicts with sealed ensemble metadata")
        if ensemble_metadata.get("graph_k") != graph_k:
            raise ValueError(f"{path}: graph_k conflicts with sealed ensemble metadata")
        expected_mean_degree = 2 * graph_k
        if (
            _exact_integer(payload["mean_degree"], "mean_degree") != expected_mean_degree
            or ensemble_metadata.get("mean_degree") != expected_mean_degree
        ):
            raise ValueError(f"{path}: mean_degree must equal 2*graph_k")
        if (
            ensemble_metadata.get("beta_scale") != 1_000_000_000
            or sealed_specification.get("beta_scale") != 1_000_000_000
        ):
            raise ValueError(f"{database}: sealed beta_scale must equal 1e9")
        validation = payload["validation"]
        if not isinstance(validation, dict):
            raise ValueError(f"{path}: validation must be a JSON object")
        generator_version = ensemble_metadata.get("generator_version")
        if not isinstance(generator_version, str) or not generator_version:
            raise ValueError(f"{database}: ensemble metadata lacks generator_version")
        if ensemble_metadata.get("specification_sha256") != payload["specification_sha256"]:
            raise ValueError(f"{database}: ensemble specification does not match the manifest")
        expected_seed_digest = ensemble_metadata.get("seed_content_sha256")
        if not isinstance(expected_seed_digest, str) or len(expected_seed_digest) != 64:
            raise ValueError(f"{database}: ensemble metadata lacks seed_content_sha256")
        if validation.get("seed_content_sha256") != expected_seed_digest:
            raise ValueError(f"{path}: validation seed digest conflicts with the sealed ensemble")
        seed_digest = hashlib.sha256()
        count = 0
        seeds_by_cell: dict[tuple[int, int], list[int]] = {}
        with self._collection.registry.read_connection() as connection:
            rows = connection.execute(
                """
                SELECT n, beta_key, graph_index, graph_seed
                FROM graphs WHERE ensemble_id = ?
                ORDER BY n, beta_key, graph_index
                """,
                (self._collection.ensemble_id,),
            )
            for row in rows:
                n_value = int(row["n"])
                beta_key_value = int(row["beta_key"])
                graph_index = int(row["graph_index"])
                graph_seed = int(row["graph_seed"])
                seed_digest.update(
                    struct.pack(
                        ">IQIQ",
                        n_value,
                        beta_key_value,
                        graph_index,
                        graph_seed,
                    )
                )
                cell_seeds = seeds_by_cell.setdefault((n_value, beta_key_value), [])
                if graph_index != len(cell_seeds):
                    raise ValueError(
                        f"{database}: graph indices are not canonical in cell "
                        f"(n={n_value}, beta_key={beta_key_value})"
                    )
                cell_seeds.append(graph_seed)
                count += 1
        expected_count = len(sizes) * len(betas) * graphs_per_cell
        if count != expected_count or seed_digest.hexdigest() != expected_seed_digest:
            raise ValueError(f"{database}: canonical graph-seed table failed validation")
        expected_keys = {(n, beta_key) for n in sizes for beta_key in beta_keys}
        if set(seeds_by_cell) != expected_keys or any(
            len(seeds) != graphs_per_cell for seeds in seeds_by_cell.values()
        ):
            raise ValueError(f"{database}: graph-seed cells are incomplete")
        self._seeds_by_cell = {key: tuple(seeds) for key, seeds in seeds_by_cell.items()}
        self._generator_version = generator_version
        generator_contract = hashlib.sha256(b"sealed_ws_generator_contract_v1\0")
        generator_contract.update(str(payload["generator_source_sha256"]).encode("ascii"))
        generator_contract.update(
            json.dumps(
                current_graph_environment,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
        generator_contract.update(hashlib.sha256(Path(__file__).read_bytes()).digest())
        self._generator_contract_sha256 = generator_contract.hexdigest()
        self._graph_k = graph_k
        self._graphs_per_cell = graphs_per_cell
        self._betas = betas
        self._beta_keys = beta_keys
        self._sizes = sizes

    @property
    def collection_id(self) -> str:
        return str(self._payload["collection_id"])

    @property
    def collection_sha256(self) -> str:
        return str(self._payload["specification_sha256"])

    def cells(self) -> tuple[ProviderCell, ...]:
        cells: list[ProviderCell] = []
        cell_index = 0
        k = self._graph_k
        for n in self._sizes:
            for beta in self._betas:
                spec = GraphCellSpec(
                    collection_sha256=self.collection_sha256,
                    cell_index=cell_index,
                    n=n,
                    parameters=ParameterPoint((("beta", beta), ("k", k))),
                )
                cells.append(
                    ProviderCell(
                        collection_id=self.collection_id,
                        graph_family="watts_strogatz",
                        generator_name="sparsegf2.circuits.graphs.watts_strogatz",
                        generator_version=self._generator_version,
                        generator_contract_sha256=self._generator_contract_sha256,
                        spec=spec,
                        graphs_per_cell=self._graphs_per_cell,
                    )
                )
                cell_index += 1
        return tuple(cells)

    def graph_seeds(self, cell: ProviderCell) -> tuple[int, ...]:
        self._validate_cell(cell)
        beta_key = int(round(float(cell.parameters["beta"]) * 1_000_000_000))
        return self._seeds_by_cell[(cell.n, beta_key)]

    def build_edges(self, cell: ProviderCell, graph_seed: int) -> Iterable[tuple[int, int]]:
        self._validate_cell(cell)
        seed = _exact_integer(graph_seed, "graph seed")
        if seed < 0:
            raise ValueError(f"graph seed must be nonnegative; got {seed}")
        return _ws_rewire_edges(
            cell.n,
            _exact_integer(cell.parameters["k"], "k"),
            _probability(cell.parameters["beta"], "beta"),
            seed,
        )

    def _validate_cell(self, cell: ProviderCell) -> None:
        if not isinstance(cell, ProviderCell):
            raise TypeError("cell must be a ProviderCell")
        if (
            cell.collection_id != self.collection_id
            or cell.spec.collection_sha256 != self.collection_sha256
        ):
            raise ValueError("cell belongs to a different graph collection")
        canonical = self.cells()
        if cell.spec.cell_index >= len(canonical) or cell != canonical[cell.spec.cell_index]:
            raise ValueError("cell is not a canonical member of this graph collection")
        if cell.n not in self._sizes:
            raise ValueError(f"n={cell.n} is not in the sealed collection")
        beta = float(cell.parameters.get("beta", math.nan))
        if int(round(beta * 1_000_000_000)) not in self._beta_keys:
            raise ValueError(f"beta={beta!r} is not in the sealed collection")


def _canonical_edges(n: int, edges: Iterable[tuple[int, int]]) -> NDArray[np.int32]:
    array = np.asarray(list(edges))
    if array.ndim != 2 or array.shape[1:] != (2,):
        raise ValueError(f"graph edges must have shape (m, 2); got {array.shape}")
    if array.shape[0] < 1:
        raise ValueError("graph edge set must be nonempty")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("graph endpoints must be integers")
    array = array.astype(np.int64, copy=False)
    if array.min() < 0 or array.max() >= n:
        raise ValueError(f"graph endpoints must lie in [0, n={n})")
    if np.any(array[:, 0] == array[:, 1]):
        raise ValueError("graph edges must not contain self-loops")
    canonical = np.sort(array, axis=1)
    order = np.lexsort((canonical[:, 1], canonical[:, 0]))
    canonical = canonical[order]
    if canonical.shape[0] > 1 and np.any(np.all(canonical[1:] == canonical[:-1], axis=1)):
        raise ValueError("graph edges must not contain duplicates")
    return np.ascontiguousarray(canonical, dtype=np.int32)


@dataclass(frozen=True, slots=True)
class CellEdgeBank:
    """Validated CSR-like edge storage for a complete graph cell."""

    path: Path
    artifact_sha256: str
    graph_seed: NDArray[np.int64]
    edge_offsets: NDArray[np.int64]
    edges: NDArray[np.int32]

    def __post_init__(self) -> None:
        if len(self.artifact_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.artifact_sha256
        ):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")

    @property
    def n_graphs(self) -> int:
        return int(self.graph_seed.shape[0])

    def graph_edges(self, graph_index: int) -> NDArray[np.int32]:
        if isinstance(graph_index, bool) or not isinstance(graph_index, (int, np.integer)):
            raise TypeError("graph_index must be an integer")
        index = int(graph_index)
        if not 0 <= index < self.n_graphs:
            raise IndexError(f"graph_index must lie in [0, {self.n_graphs})")
        start, stop = int(self.edge_offsets[index]), int(self.edge_offsets[index + 1])
        return self.edges[start:stop]


def edge_bank_path(data_root: str | Path, cell: ProviderCell) -> Path:
    edge_root = (Path(data_root) / "single_ref" / "raw_tau" / "edge_banks").resolve()
    candidate = (
        edge_root
        / cell.collection_id
        / f"contract_{cell.generator_contract_sha256[:16]}"
        / f"n{cell.n}"
        / f"cell_{cell.spec.cell_index:06d}_{cell.cell_sha256[:16]}.npz"
    ).resolve()
    try:
        candidate.relative_to(edge_root)
    except ValueError as exc:  # defense in depth beyond ProviderCell key validation
        raise ValueError("edge-bank path escaped the configured data root") from exc
    return candidate


def _validation_receipt_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.validated.npz")


def _publish_validation_receipt(bank: CellEdgeBank, cell: ProviderCell) -> None:
    """Atomically seal a coordinator-validated bank for cheap worker loading."""
    arrays: dict[str, object] = {
        "validation_version": np.str_(EDGE_BANK_VALIDATION_VERSION),
        "edge_bank_sha256": np.str_(bank.artifact_sha256),
        "edge_bank_size": np.int64(bank.path.stat().st_size),
        "collection_id": np.str_(cell.collection_id),
        "collection_sha256": np.str_(cell.spec.collection_sha256),
        "cell_sha256": np.str_(cell.cell_sha256),
        "generator_contract_sha256": np.str_(cell.generator_contract_sha256),
    }
    write_deterministic_npz(_validation_receipt_path(bank.path), arrays)


def _load_validation_receipt(path: Path, cell: ProviderCell) -> tuple[str, int] | None:
    receipt_path = _validation_receipt_path(path)
    if not receipt_path.is_file():
        return None
    with load_npz_snapshot(receipt_path) as data:
        expected = {
            "validation_version": EDGE_BANK_VALIDATION_VERSION,
            "collection_id": cell.collection_id,
            "collection_sha256": cell.spec.collection_sha256,
            "cell_sha256": cell.cell_sha256,
            "generator_contract_sha256": cell.generator_contract_sha256,
        }
        for key, value in expected.items():
            if key not in data.files:
                raise ValueError(f"{receipt_path}: missing {key}")
            actual = _scalar(data, key)
            if actual != value:
                raise ValueError(f"{receipt_path}: {key}={actual!r}, expected {value!r}")
        for key in ("edge_bank_sha256", "edge_bank_size"):
            if key not in data.files:
                raise ValueError(f"{receipt_path}: missing {key}")
        digest = str(_scalar(data, "edge_bank_sha256"))
        size = int(_scalar(data, "edge_bank_size"))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{receipt_path}: invalid edge-bank SHA-256 digest")
    if size < 1:
        raise ValueError(f"{receipt_path}: edge-bank size must be positive")
    return digest, size


def prepare_edge_bank(
    data_root: str | Path,
    provider: GraphProvider,
    cell: ProviderCell,
) -> str:
    """Create or validate one variable-edge-count cell bank exactly once."""
    path = edge_bank_path(data_root, cell)
    seeds = provider.graph_seeds(cell)
    if len(seeds) != cell.graphs_per_cell:
        raise ValueError(f"provider returned {len(seeds)} seeds, expected {cell.graphs_per_cell}")
    if path.exists():
        bank = load_edge_bank(path, cell, seeds, validation="full")
        _publish_validation_receipt(bank, cell)
        return str(path)

    blocks: list[NDArray[np.int32]] = []
    offsets = np.zeros(cell.graphs_per_cell + 1, dtype=np.int64)
    for graph_index, seed in enumerate(seeds):
        block = _canonical_edges(cell.n, provider.build_edges(cell, seed))
        blocks.append(block)
        offsets[graph_index + 1] = offsets[graph_index] + block.shape[0]
    all_edges = np.concatenate(blocks, axis=0)
    seed_array = np.asarray(seeds, dtype=np.int64)
    arrays: dict[str, object] = {
        "schema_version": np.int32(RAW_TAU_SCHEMA_VERSION),
        "edge_bank_version": np.str_(EDGE_BANK_VERSION),
        "collection_id": np.str_(cell.collection_id),
        "collection_sha256": np.str_(cell.spec.collection_sha256),
        "cell_sha256": np.str_(cell.cell_sha256),
        "cell_index": np.int32(cell.spec.cell_index),
        "graph_family": np.str_(cell.graph_family),
        "generator_name": np.str_(cell.generator_name),
        "generator_version": np.str_(cell.generator_version),
        "generator_contract_sha256": np.str_(cell.generator_contract_sha256),
        "n": np.int32(cell.n),
        "parameters_json": np.str_(
            json.dumps(cell.parameters, allow_nan=False, sort_keys=True, separators=(",", ":"))
        ),
        "graphs_per_cell": np.int32(cell.graphs_per_cell),
        "graph_index": np.arange(cell.graphs_per_cell, dtype=np.int32),
        "graph_seed": seed_array,
        "edge_offsets": offsets,
        "edges": all_edges,
        "seed_sha256": np.str_(array_sha256(seed_array)),
        "edges_sha256": np.str_(array_sha256(all_edges)),
    }
    write_deterministic_npz(path, arrays)
    bank = load_edge_bank(path, cell, seeds, validation="full")
    _publish_validation_receipt(bank, cell)
    return str(path)


def _scalar(data: np.lib.npyio.NpzFile, key: str):
    value = data[key]
    if value.shape != ():
        raise ValueError(f"{key} must be scalar; got {value.shape}")
    return value.item()


def _stored_integer_array(
    data: np.lib.npyio.NpzFile,
    key: str,
    *,
    kind: str,
    itemsize: int,
) -> np.ndarray:
    array = np.asarray(data[key])
    if array.dtype.kind != kind or array.dtype.itemsize != itemsize:
        raise ValueError(f"{key} must have {kind}{itemsize} integer storage; got {array.dtype}")
    return array


def load_edge_bank(
    path: str | Path,
    cell: ProviderCell,
    expected_seeds: Iterable[int] | None = None,
    *,
    validation: Literal["auto", "full"] = "auto",
) -> CellEdgeBank:
    """Load an immutable cell edge bank with full or receipt-backed validation.

    ``prepare_edge_bank`` performs the expensive per-graph canonical validation
    in the coordinator and atomically publishes a receipt containing the bank's
    whole-file digest.  The default worker path verifies that digest against one
    closed-handle byte snapshot and can then skip re-canonicalizing every graph.
    ``validation="full"`` deliberately ignores the receipt and repeats all
    graph-level checks.
    """
    if validation not in {"auto", "full"}:
        raise ValueError("validation must be 'auto' or 'full'")
    resolved = Path(path)
    receipt = _load_validation_receipt(resolved, cell) if validation == "auto" else None
    payload = read_shared_bytes(resolved)
    artifact_sha256 = hashlib.sha256(payload).hexdigest()
    if receipt is not None:
        expected_digest, expected_size = receipt
        if len(payload) != expected_size or artifact_sha256 != expected_digest:
            raise ValueError(
                f"{resolved}: immutable edge-bank digest does not match its validation receipt"
            )
    buffer = io.BytesIO(payload)
    try:
        data_context = np.load(buffer, allow_pickle=False)
    except Exception:
        buffer.close()
        raise
    try:
        data = data_context
        expected = {
            "schema_version": RAW_TAU_SCHEMA_VERSION,
            "edge_bank_version": EDGE_BANK_VERSION,
            "collection_id": cell.collection_id,
            "collection_sha256": cell.spec.collection_sha256,
            "cell_sha256": cell.cell_sha256,
            "cell_index": cell.spec.cell_index,
            "graph_family": cell.graph_family,
            "generator_name": cell.generator_name,
            "generator_version": cell.generator_version,
            "generator_contract_sha256": cell.generator_contract_sha256,
            "n": cell.n,
            "graphs_per_cell": cell.graphs_per_cell,
        }
        for key, value in expected.items():
            actual = _scalar(data, key)
            if actual != value:
                raise ValueError(f"{resolved}: {key}={actual!r}, expected {value!r}")
        expected_parameters = json.dumps(
            cell.parameters, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        if _scalar(data, "parameters_json") != expected_parameters:
            raise ValueError(f"{resolved}: graph-parameter metadata mismatch")
        graph_index = _stored_integer_array(data, "graph_index", kind="i", itemsize=4)
        seeds = _stored_integer_array(data, "graph_seed", kind="i", itemsize=8)
        offsets = _stored_integer_array(data, "edge_offsets", kind="i", itemsize=8)
        edges = _stored_integer_array(data, "edges", kind="i", itemsize=4)
        seed_hash = str(_scalar(data, "seed_sha256"))
        edge_hash = str(_scalar(data, "edges_sha256"))
    finally:
        data_context.close()
        buffer.close()
    if not np.array_equal(graph_index, np.arange(cell.graphs_per_cell, dtype=np.int32)):
        raise ValueError(f"{resolved}: graph_index is not canonical")
    if seeds.shape != (cell.graphs_per_cell,):
        raise ValueError(f"{resolved}: invalid graph_seed shape {seeds.shape}")
    if expected_seeds is not None and not np.array_equal(
        seeds, np.asarray(tuple(expected_seeds), dtype=np.int64)
    ):
        raise ValueError(f"{resolved}: graph seeds do not match the provider")
    if offsets.shape != (cell.graphs_per_cell + 1,) or offsets[0] != 0:
        raise ValueError(f"{resolved}: invalid edge_offsets")
    if np.any(np.diff(offsets) <= 0) or offsets[-1] != edges.shape[0]:
        raise ValueError(f"{resolved}: edge_offsets do not delimit nonempty graphs")
    if edges.ndim != 2 or edges.shape[1:] != (2,):
        raise ValueError(f"{resolved}: invalid edges shape {edges.shape}")
    if receipt is None and (array_sha256(seeds) != seed_hash or array_sha256(edges) != edge_hash):
        raise ValueError(f"{resolved}: edge-bank content digest mismatch")
    if receipt is None:
        for graph_index_value in range(cell.graphs_per_cell):
            start, stop = int(offsets[graph_index_value]), int(offsets[graph_index_value + 1])
            canonical = _canonical_edges(cell.n, edges[start:stop])
            if not np.array_equal(canonical, edges[start:stop]):
                raise ValueError(f"{resolved}: graph {graph_index_value} edges are not canonical")
    seeds.flags.writeable = False
    offsets.flags.writeable = False
    edges.flags.writeable = False
    return CellEdgeBank(
        path=resolved,
        artifact_sha256=artifact_sha256,
        graph_seed=seeds,
        edge_offsets=offsets,
        edges=edges,
    )


__all__ = [
    "EDGE_BANK_VERSION",
    "EDGE_BANK_VALIDATION_VERSION",
    "SEALED_WS_GRAPH_FAMILY",
    "CellEdgeBank",
    "EdgeFactory",
    "GraphProvider",
    "GridGraphProvider",
    "ProviderCell",
    "WattsStrogatzRegistryProvider",
    "builtin_graph_factory",
    "edge_bank_path",
    "load_edge_bank",
    "prepare_edge_bank",
    "validate_builtin_parameters",
]
