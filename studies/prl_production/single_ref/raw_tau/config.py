"""Strict JSON configuration and source fingerprints for raw-tau sweeps."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np
from studies.prl_production.sweep_spec import (
    GraphCollectionGridSpec,
    GraphParameterGrid,
    ParameterAxis,
    ProbabilityGrid,
    ScientificEnvironmentContract,
    SingleReferenceProtocolSpec,
    SingleReferenceSweepSpec,
)

from .providers import GraphProvider, GridGraphProvider, WattsStrogatzRegistryProvider

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_SCHEMA_VERSION = 1
SOURCE_FINGERPRINT_PATHS = (
    "studies/prl_production/sweep_spec.py",
    "studies/prl_production/single_ref/raw_tau/catalog.py",
    "studies/prl_production/single_ref/raw_tau/config.py",
    "studies/prl_production/single_ref/raw_tau/engine.py",
    "studies/prl_production/single_ref/raw_tau/io.py",
    "studies/prl_production/single_ref/raw_tau/providers.py",
    "studies/prl_production/single_ref/raw_tau/run.py",
    "studies/prl_production/single_ref/raw_tau/storage.py",
    "studies/prl_production/single_ref/shared_io.py",
    "studies/prl_production/graph_registry/collection.py",
    "studies/prl_production/graph_registry/database.py",
    "src/sparsegf2/__init__.py",
    "src/sparsegf2/errors.py",
    "src/sparsegf2/core/__init__.py",
    "src/sparsegf2/core/_protocol.py",
    "src/sparsegf2/core/linalg_gf2.py",
    "src/sparsegf2/core/numba_kernels.py",
    "src/sparsegf2/core/observables.py",
    "src/sparsegf2/core/sparse_tableau.py",
    "src/sparsegf2/core/symplectic.py",
    "src/sparsegf2/circuits/_clifford_table.py",
    "src/sparsegf2/circuits/graphs.py",
)


def current_scientific_environment_contract() -> ScientificEnvironmentContract:
    """Return the exact dependency contract pinned by a production experiment."""
    return ScientificEnvironmentContract(
        python=platform.python_version(),
        numpy=np.__version__,
        numba=numba.__version__,
        bit_generator=np.random.PCG64.__name__,
    )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Execution-only settings excluded from the scientific fingerprint."""

    data_root: Path
    workers: int
    checkpoint_every: int
    max_in_flight: int

    def __post_init__(self) -> None:
        if not isinstance(self.data_root, Path) or not self.data_root.is_absolute():
            raise ValueError("data_root must be an absolute pathlib.Path")
        if self.data_root.exists() and not self.data_root.is_dir():
            raise ValueError(f"data_root exists but is not a directory: {self.data_root}")
        for name in ("workers", "checkpoint_every", "max_in_flight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_in_flight < self.workers:
            raise ValueError("max_in_flight must be at least workers")


@dataclass(frozen=True, slots=True)
class ResolvedRawTauConfig:
    """One validated provider, immutable experiment, and runtime policy."""

    provider: GraphProvider
    sweep: SingleReferenceSweepSpec
    runtime: RuntimeConfig
    config_path: Path


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _exact_keys(
    value: dict[str, object],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    name: str,
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing or unknown:
        pieces = []
        if missing:
            pieces.append(f"missing {sorted(missing)}")
        if unknown:
            pieces.append(f"unknown {sorted(unknown)}")
        raise ValueError(f"{name}: " + "; ".join(pieces))


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be nonempty text without surrounding whitespace")
    return value


def _resolve_path(value: object, config_path: Path, name: str) -> Path:
    path = Path(_text(value, name))
    if not path.is_absolute():
        candidate = (config_path.parent / path).resolve()
        if candidate.exists() or not (REPO_ROOT / path).exists():
            return candidate
        return (REPO_ROOT / path).resolve()
    return path.resolve()


def source_fingerprint_sha256() -> str:
    """Fingerprint every source file that fixes raw trajectory semantics."""
    digest = hashlib.sha256()
    for relative in SOURCE_FINGERPRINT_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        with path.open("rb") as handle:
            digest.update(hashlib.file_digest(handle, "sha256").digest())
    return digest.hexdigest()


def _load_provider(
    payload: dict[str, object],
    config_path: Path,
) -> GraphProvider:
    kind = _text(payload.get("kind"), "graph_source.kind")
    if kind == "ws_registry":
        _exact_keys(payload, required={"kind", "manifest"}, name="graph_source")
        manifest = _resolve_path(payload["manifest"], config_path, "graph_source.manifest")
        return WattsStrogatzRegistryProvider(manifest)
    if kind != "cartesian_builtin":
        raise ValueError("graph_source.kind must be 'ws_registry' or 'cartesian_builtin'")

    _exact_keys(
        payload,
        required={
            "kind",
            "name",
            "graph_family",
            "generator_name",
            "generator_version",
            "sizes",
            "parameter_axes",
            "graphs_per_cell",
            "master_seed",
        },
        name="graph_source",
    )
    sizes_raw = payload["sizes"]
    if not isinstance(sizes_raw, list):
        raise TypeError("graph_source.sizes must be a JSON array")
    axes_raw = _mapping(payload["parameter_axes"], "graph_source.parameter_axes")
    axes = tuple(
        ParameterAxis(name, tuple(values))
        for name, values in axes_raw.items()
        if isinstance(values, list)
    )
    if len(axes) != len(axes_raw):
        raise TypeError("every graph parameter axis must be a JSON array")
    spec = GraphCollectionGridSpec(
        name=_text(payload["name"], "graph_source.name"),
        graph_family=_text(payload["graph_family"], "graph_source.graph_family"),
        generator_name=_text(payload["generator_name"], "graph_source.generator_name"),
        generator_version=_text(payload["generator_version"], "graph_source.generator_version"),
        sizes=tuple(_integer(value, "graph_source size", minimum=2) for value in sizes_raw),
        parameter_grid=GraphParameterGrid(axes),
        graphs_per_cell=_integer(
            payload["graphs_per_cell"], "graph_source.graphs_per_cell", minimum=1
        ),
        master_seed=_integer(payload["master_seed"], "graph_source.master_seed"),
    )
    return GridGraphProvider(spec)


def load_config(path: str | os.PathLike[str]) -> ResolvedRawTauConfig:
    """Load a strict, fully resolved raw-tau JSON configuration."""
    config_path = Path(path).resolve()
    payload = _mapping(json.loads(config_path.read_text(encoding="utf-8")), "configuration")
    _exact_keys(
        payload,
        required={"schema_version", "name", "graph_source", "protocol", "runtime"},
        name="configuration",
    )
    if payload["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"configuration.schema_version must be {CONFIG_SCHEMA_VERSION}; "
            f"got {payload['schema_version']!r}"
        )
    provider = _load_provider(_mapping(payload["graph_source"], "graph_source"), config_path)

    protocol_payload = _mapping(payload["protocol"], "protocol")
    _exact_keys(
        protocol_payload,
        required={
            "n_circuits",
            "q_scramble",
            "q_max",
            "p_min",
            "p_max",
            "delta_p",
            "master_seed",
        },
        optional={"reference_system_qubit_policy", "p_randomness_policy"},
        name="protocol",
    )
    protocol = SingleReferenceProtocolSpec(
        n_circuits=_integer(protocol_payload["n_circuits"], "protocol.n_circuits", minimum=1),
        q_scramble=_integer(protocol_payload["q_scramble"], "protocol.q_scramble"),
        q_max=_integer(protocol_payload["q_max"], "protocol.q_max", minimum=1),
        p_grid=ProbabilityGrid(
            protocol_payload["p_min"],
            protocol_payload["p_max"],
            protocol_payload["delta_p"],
        ),
        master_seed=_integer(protocol_payload["master_seed"], "protocol.master_seed"),
        reference_system_qubit_policy=_text(
            protocol_payload.get("reference_system_qubit_policy", "fixed_last"),
            "protocol.reference_system_qubit_policy",
        ),
        p_randomness_policy=_text(
            protocol_payload.get("p_randomness_policy", "independent"),
            "protocol.p_randomness_policy",
        ),
    )
    sweep = SingleReferenceSweepSpec(
        name=_text(payload["name"], "name"),
        graph_collection_sha256=provider.collection_sha256,
        source_fingerprint_sha256=source_fingerprint_sha256(),
        environment_contract=current_scientific_environment_contract(),
        protocol=protocol,
    )

    runtime_payload = _mapping(payload["runtime"], "runtime")
    _exact_keys(
        runtime_payload,
        required={"data_root", "workers", "checkpoint_every"},
        optional={"max_in_flight"},
        name="runtime",
    )
    workers = _integer(runtime_payload["workers"], "runtime.workers", minimum=1)
    runtime = RuntimeConfig(
        data_root=_resolve_path(runtime_payload["data_root"], config_path, "runtime.data_root"),
        workers=workers,
        checkpoint_every=_integer(
            runtime_payload["checkpoint_every"], "runtime.checkpoint_every", minimum=1
        ),
        max_in_flight=_integer(
            runtime_payload.get("max_in_flight", 2 * workers),
            "runtime.max_in_flight",
            minimum=1,
        ),
    )
    return ResolvedRawTauConfig(
        provider=provider,
        sweep=sweep,
        runtime=runtime,
        config_path=config_path,
    )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "ResolvedRawTauConfig",
    "RuntimeConfig",
    "current_scientific_environment_contract",
    "SOURCE_FINGERPRINT_PATHS",
    "load_config",
    "source_fingerprint_sha256",
]
