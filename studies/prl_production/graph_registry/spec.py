"""Immutable specification for the paper's Watts--Strogatz graph collection."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Real

import numpy as np

BETA_SCALE = 1_000_000_000
GRAPH_K = 2
MEAN_DEGREE = 2 * GRAPH_K
POSITIVE_BETA_COUNT = 49
GRAPHS_PER_CELL = 1_000
GRAPH_SEED_MASTER = 3_700_000_001
SEED_DERIVATION = "sha256_tuple_v1"
GENERATOR_NAME = "sparsegf2.circuits.graphs.watts_strogatz"
GENERATOR_VERSION = "ws_rewire_v1"
SCHEMA_VERSION = 1

PRODUCTION_SIZES = (64, 96, 128, 160, 192, 224, 256)


def _strict_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def canonical_beta_key(beta: float) -> int:
    """Return the integer identity used by the WS generator for one beta."""
    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise TypeError("beta must be a real number")
    value = float(beta)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"beta must be finite and in [0, 1]; got {beta!r}")
    return int(round(value * BETA_SCALE))


def beta_from_key(beta_key: int) -> float:
    """Return the canonical simulation beta represented by ``beta_key``."""
    key = _strict_integer(beta_key, "beta_key")
    if key > BETA_SCALE:
        raise ValueError(f"beta_key must be <= {BETA_SCALE}; got {key}")
    return key / BETA_SCALE


def production_beta_keys() -> tuple[int, ...]:
    """Return zero plus 49 log-spaced positive beta identities."""
    positive = tuple(
        canonical_beta_key(value) for value in np.geomspace(0.005, 1.0, POSITIVE_BETA_COUNT)
    )
    keys = (0, *positive)
    if len(keys) != 50 or len(set(keys)) != len(keys):
        raise RuntimeError("the canonical production beta grid must contain 50 unique values")
    if keys[1] != 5_000_000 or keys[-1] != BETA_SCALE:
        raise RuntimeError("the production beta grid endpoints changed unexpectedly")
    if tuple(sorted(keys)) != keys:
        raise RuntimeError("the production beta grid must be strictly increasing")
    return keys


@dataclass(frozen=True, slots=True)
class SeedAssignment:
    """One independently indexed graph draw in a registry collection."""

    n: int
    beta_key: int
    graph_index: int
    graph_seed: int

    @property
    def beta(self) -> float:
        return beta_from_key(self.beta_key)


@dataclass(frozen=True, slots=True)
class GraphCollectionSpec:
    """Complete immutable recipe for a graph-seed collection."""

    name: str
    sizes: tuple[int, ...]
    beta_keys: tuple[int, ...]
    graphs_per_cell: int
    graph_k: int = GRAPH_K
    master_seed: int = GRAPH_SEED_MASTER
    seed_derivation: str = SEED_DERIVATION
    generator_name: str = GENERATOR_NAME
    generator_version: str = GENERATOR_VERSION
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("name must be a nonempty string without surrounding whitespace")
        sizes = tuple(_strict_integer(n, "n", minimum=3) for n in self.sizes)
        if not sizes or tuple(sorted(set(sizes))) != sizes:
            raise ValueError("sizes must be nonempty, unique, and strictly increasing")
        graph_k = _strict_integer(self.graph_k, "graph_k", minimum=1)
        if any(2 * graph_k >= n for n in sizes):
            raise ValueError("every size must satisfy 2 * graph_k < n")
        keys = tuple(_strict_integer(key, "beta_key") for key in self.beta_keys)
        if not keys or tuple(sorted(set(keys))) != keys:
            raise ValueError("beta_keys must be nonempty, unique, and strictly increasing")
        if keys[-1] > BETA_SCALE:
            raise ValueError(f"beta_keys must not exceed {BETA_SCALE}")
        _strict_integer(self.graphs_per_cell, "graphs_per_cell", minimum=1)
        _strict_integer(self.master_seed, "master_seed")
        _strict_integer(self.schema_version, "schema_version", minimum=1)
        if not self.seed_derivation:
            raise ValueError("seed_derivation must be nonempty")
        if not self.generator_name or not self.generator_version:
            raise ValueError("generator identity must be nonempty")

    @property
    def betas(self) -> tuple[float, ...]:
        return tuple(beta_from_key(key) for key in self.beta_keys)

    @property
    def n_cells(self) -> int:
        return len(self.sizes) * len(self.beta_keys)

    @property
    def n_graphs(self) -> int:
        return self.n_cells * self.graphs_per_cell

    def canonical_payload(self) -> dict[str, object]:
        """Return the JSON-safe content that fixes the collection identity."""
        return {
            "beta_keys": list(self.beta_keys),
            "beta_scale": BETA_SCALE,
            "generator_name": self.generator_name,
            "generator_version": self.generator_version,
            "graph_k": self.graph_k,
            "graphs_per_cell": self.graphs_per_cell,
            "master_seed": self.master_seed,
            "name": self.name,
            "schema_version": self.schema_version,
            "seed_derivation": self.seed_derivation,
            "sizes": list(self.sizes),
        }

    @property
    def specification_sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def collection_id(self) -> str:
        return f"{self.name}_{self.specification_sha256[:16]}"

    def graph_seed(self, n: int, beta_key: int, graph_index: int) -> int:
        """Derive one stable nonnegative signed-64-bit seed from its full identity."""
        size = _strict_integer(n, "n", minimum=3)
        key = _strict_integer(beta_key, "beta_key")
        index = _strict_integer(graph_index, "graph_index")
        if size not in self.sizes:
            raise ValueError(f"n={size} is not in this collection")
        if key not in self.beta_keys:
            raise ValueError(f"beta_key={key} is not in this collection")
        if index >= self.graphs_per_cell:
            raise ValueError(f"graph_index must be < {self.graphs_per_cell}; got {graph_index}")
        payload = (f"{self.seed_derivation}\0{self.master_seed}\0{size}\0{key}\0{index}").encode(
            "ascii"
        )
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)

    def records(self) -> Iterator[SeedAssignment]:
        """Yield every record in canonical cell and graph-index order."""
        for n in self.sizes:
            for beta_key in self.beta_keys:
                for graph_index in range(self.graphs_per_cell):
                    yield SeedAssignment(
                        n=n,
                        beta_key=beta_key,
                        graph_index=graph_index,
                        graph_seed=self.graph_seed(n, beta_key, graph_index),
                    )

    def seed_content_sha256(self) -> str:
        """Hash the canonical graph identity and seed table independently of SQLite bytes."""
        digest = hashlib.sha256()
        for record in self.records():
            digest.update(
                struct.pack(
                    ">IQIQ",
                    record.n,
                    record.beta_key,
                    record.graph_index,
                    record.graph_seed,
                )
            )
        return digest.hexdigest()


def production_spec() -> GraphCollectionSpec:
    """Return the 350,000-draw graph collection requested for the paper."""
    return GraphCollectionSpec(
        name="ws_c2_paper_v1",
        sizes=PRODUCTION_SIZES,
        beta_keys=production_beta_keys(),
        graphs_per_cell=GRAPHS_PER_CELL,
    )


def smoke_spec() -> GraphCollectionSpec:
    """Return a tiny collection for end-to-end validation."""
    return GraphCollectionSpec(
        name="ws_c2_smoke_v1",
        sizes=(8, 12),
        beta_keys=(0, 5_000_000, BETA_SCALE),
        graphs_per_cell=4,
    )


__all__ = [
    "BETA_SCALE",
    "GENERATOR_NAME",
    "GENERATOR_VERSION",
    "GRAPHS_PER_CELL",
    "GRAPH_K",
    "GRAPH_SEED_MASTER",
    "GraphCollectionSpec",
    "SeedAssignment",
    "MEAN_DEGREE",
    "POSITIVE_BETA_COUNT",
    "PRODUCTION_SIZES",
    "SCHEMA_VERSION",
    "SEED_DERIVATION",
    "beta_from_key",
    "canonical_beta_key",
    "production_beta_keys",
    "production_spec",
    "smoke_spec",
]
