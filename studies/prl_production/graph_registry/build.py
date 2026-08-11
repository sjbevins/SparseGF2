"""Build and validate the paper's persistent Watts--Strogatz graph registry."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sqlite3
import struct
import sys
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sparsegf2.circuits import graphs as graph_module
from sparsegf2.circuits.graphs import watts_strogatz

from .database import GraphRegistryDatabase, GraphSeedRecord
from .spec import (
    GraphCollectionSpec,
    SeedAssignment,
    beta_from_key,
    production_spec,
    smoke_spec,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "graph_registry"
PUBLIC_STATUS = Path(__file__).resolve().parent / "STATUS.md"


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Evidence from a complete registry validation pass."""

    graph_count: int
    cell_count: int
    distinct_seed_count: int
    seed_content_sha256: str
    reconstructed_graphs: int
    sqlite_integrity: str
    foreign_key_violations: int


@dataclass(frozen=True, slots=True)
class BuildSummary:
    """Paths and immutable identity of one built collection."""

    collection_id: str
    ensemble_id: int
    database_path: Path
    database_sha256: str
    specification_sha256: str
    generator_source_sha256: str
    validation: ValidationSummary


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def default_database_path(spec: GraphCollectionSpec) -> Path:
    return DATA_ROOT / spec.collection_id / "graph_registry.sqlite3"


def _display_path(path: Path) -> str:
    try:
        displayed = path.resolve().relative_to(ROOT.parent.parent)
    except ValueError:
        displayed = path.resolve()
    return str(displayed).replace("\\", "/")


def _generator_source_sha256() -> str:
    source_path = Path(graph_module.__file__).resolve()
    return _file_sha256(source_path)


def _ensemble_metadata(spec: GraphCollectionSpec) -> dict[str, object]:
    return {
        "beta_scale": 1_000_000_000,
        "generator_source_sha256": _generator_source_sha256(),
        "generator_version": spec.generator_version,
        "graph_k": spec.graph_k,
        "mean_degree": 2 * spec.graph_k,
        "seed_content_sha256": spec.seed_content_sha256(),
        "seed_derivation": spec.seed_derivation,
        "specification": spec.canonical_payload(),
        "specification_sha256": spec.specification_sha256,
    }


def _database_records(
    records: Iterator[SeedAssignment], batch_size: int
) -> Iterator[list[GraphSeedRecord]]:
    batch: list[GraphSeedRecord] = []
    for value in records:
        n = value.n
        beta_key = value.beta_key
        batch.append(
            GraphSeedRecord(
                n=n,
                beta=beta_from_key(beta_key),
                beta_key=beta_key,
                graph_index=value.graph_index,
                graph_seed=value.graph_seed,
            )
        )
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _selected_reconstructions(spec: GraphCollectionSpec) -> tuple[tuple[int, int, int], ...]:
    beta_positions = sorted({0, 1, len(spec.beta_keys) // 2, len(spec.beta_keys) - 1})
    selected: list[tuple[int, int, int]] = []
    for n in spec.sizes:
        for position in beta_positions:
            graph_index = 0 if position in (0, 1) else spec.graphs_per_cell - 1
            selected.append((n, spec.beta_keys[position], graph_index))
        if spec.graphs_per_cell > 1:
            selected.append((n, 0, spec.graphs_per_cell - 1))
    return tuple(selected)


def _validate_graph_realizations(
    registry: GraphRegistryDatabase,
    spec: GraphCollectionSpec,
    ensemble_id: int,
) -> int:
    reconstructed = 0
    beta_zero_edges: dict[int, tuple[tuple[int, int], ...]] = {}
    for n, beta_key, graph_index in _selected_reconstructions(spec):
        cell = registry.graphs_for_cell(ensemble_id, n, beta_key)
        if len(cell) != spec.graphs_per_cell:
            raise RuntimeError(f"cell (n={n}, beta_key={beta_key}) is incomplete")
        record = cell[graph_index]
        if record.graph_index != graph_index:
            raise RuntimeError("cell ordering no longer maps directly to graph_index")
        topology = watts_strogatz(
            n,
            k=spec.graph_k,
            beta=beta_from_key(beta_key),
            seed=record.graph_seed,
        )
        edges = tuple(topology.edges)
        expected_edges = n * spec.graph_k
        if len(edges) != expected_edges or len(set(edges)) != expected_edges:
            raise RuntimeError("a reconstructed graph has an invalid edge count or duplicate edge")
        if edges != tuple(sorted(edges)):
            raise RuntimeError("a reconstructed edge list is not in canonical order")
        if any(not (0 <= u < v < n) for u, v in edges):
            raise RuntimeError("a reconstructed graph has an invalid vertex or self-loop")
        if beta_key == 0:
            previous = beta_zero_edges.setdefault(n, edges)
            if previous != edges:
                raise RuntimeError("beta=0 must reconstruct the same C(n,2) for every seed")
        reconstructed += 1
    return reconstructed


def _stored_seed_digest(
    registry: GraphRegistryDatabase,
    spec: GraphCollectionSpec,
    ensemble_id: int,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    expected = spec.records()
    count = 0
    with registry.read_connection() as connection:
        cursor = connection.execute(
            """
            SELECT n, beta_key, graph_index, graph_seed
            FROM graphs
            WHERE ensemble_id = ?
            ORDER BY n, beta_key, graph_index
            """,
            (ensemble_id,),
        )
        for row in cursor:
            try:
                planned = next(expected)
            except StopIteration as exc:
                raise RuntimeError(
                    "registry contains more graph rows than the specification"
                ) from exc
            actual = (
                int(row["n"]),
                int(row["beta_key"]),
                int(row["graph_index"]),
                int(row["graph_seed"]),
            )
            planned_values = (
                planned.n,
                planned.beta_key,
                planned.graph_index,
                planned.graph_seed,
            )
            if actual != planned_values:
                raise RuntimeError(
                    f"stored graph assignment {actual!r} does not match {planned_values!r}"
                )
            digest.update(struct.pack(">IQIQ", *actual))
            count += 1
    try:
        extra = next(expected)
    except StopIteration:
        pass
    else:
        raise RuntimeError(f"registry is missing graph assignment {extra!r}")
    return digest.hexdigest(), count


def validate_collection(
    registry: GraphRegistryDatabase,
    spec: GraphCollectionSpec,
    ensemble_id: int,
    *,
    reconstruct: bool = True,
) -> ValidationSummary:
    """Perform complete row-level validation plus bounded graph reconstruction checks."""
    stored_digest, graph_count = _stored_seed_digest(registry, spec, ensemble_id)
    expected_digest = spec.seed_content_sha256()
    if stored_digest != expected_digest:
        raise RuntimeError(
            f"seed-content SHA-256 mismatch: stored={stored_digest}, expected={expected_digest}"
        )

    expected_cells = {(n, beta_key) for n in spec.sizes for beta_key in spec.beta_keys}
    with registry.read_connection() as connection:
        cell_rows = connection.execute(
            """
            SELECT n, beta_key, count(*) AS count_rows,
                   min(graph_index) AS first_index, max(graph_index) AS last_index,
                   count(DISTINCT graph_index) AS distinct_indices
            FROM graphs WHERE ensemble_id = ? GROUP BY n, beta_key
            """,
            (ensemble_id,),
        ).fetchall()
        distinct_seed_count = int(
            connection.execute(
                "SELECT count(DISTINCT graph_seed) FROM graphs WHERE ensemble_id = ?",
                (ensemble_id,),
            ).fetchone()[0]
        )
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_key_violations = len(connection.execute("PRAGMA foreign_key_check").fetchall())

    actual_cells = {(int(row["n"]), int(row["beta_key"])) for row in cell_rows}
    if actual_cells != expected_cells:
        raise RuntimeError("stored (n, beta) cells do not match the collection specification")
    for row in cell_rows:
        observed = (
            int(row["count_rows"]),
            int(row["first_index"]),
            int(row["last_index"]),
            int(row["distinct_indices"]),
        )
        expected = (spec.graphs_per_cell, 0, spec.graphs_per_cell - 1, spec.graphs_per_cell)
        if observed != expected:
            raise RuntimeError(f"cell has invalid graph-index coverage: {observed!r}")
    if graph_count != spec.n_graphs:
        raise RuntimeError(f"registry contains {graph_count} graphs; expected {spec.n_graphs}")
    if integrity != "ok" or foreign_key_violations:
        raise RuntimeError(
            f"SQLite validation failed: integrity={integrity!r}, "
            f"foreign_key_violations={foreign_key_violations}"
        )

    reconstructed = _validate_graph_realizations(registry, spec, ensemble_id) if reconstruct else 0
    return ValidationSummary(
        graph_count=graph_count,
        cell_count=len(cell_rows),
        distinct_seed_count=distinct_seed_count,
        seed_content_sha256=stored_digest,
        reconstructed_graphs=reconstructed,
        sqlite_integrity=integrity,
        foreign_key_violations=foreign_key_violations,
    )


def _find_ensemble_id(registry: GraphRegistryDatabase, collection_id: str) -> int:
    with registry.read_connection() as connection:
        row = connection.execute(
            "SELECT ensemble_id FROM ensembles WHERE ensemble_key = ?", (collection_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"database does not contain collection {collection_id!r}")
    return int(row["ensemble_id"])


def _checkpoint_wal(path: Path) -> None:
    with sqlite3.connect(path, timeout=30.0) as connection:
        result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if result is None or int(result[0]) != 0:
        raise RuntimeError(f"could not checkpoint the SQLite WAL: {result!r}")


def build_collection(
    spec: GraphCollectionSpec,
    database_path: Path,
    *,
    batch_size: int = 1_000,
    progress: Callable[[int, int], None] | None = None,
) -> BuildSummary:
    """Create or exactly resume one collection, then seal it after validation."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    registry = GraphRegistryDatabase(database_path)
    metadata = _ensemble_metadata(spec)
    ensemble_id = registry.register_ensemble(
        spec.collection_id,
        graph_family="watts_strogatz_rewired_circulant",
        expected_graphs_per_cell=spec.graphs_per_cell,
        description=("Paper graph collection: independently indexed Watts-Strogatz C(n,2) draws"),
        metadata=metadata,
        status="active",
    )
    registry.set_ensemble_status(ensemble_id, "active")

    completed = 0
    for batch in _database_records(spec.records(), batch_size):
        registry.register_graphs(ensemble_id, batch)
        completed += len(batch)
        if progress is not None:
            progress(completed, spec.n_graphs)

    validation = validate_collection(registry, spec, ensemble_id)
    registry.set_ensemble_status(ensemble_id, "complete")
    _checkpoint_wal(database_path)
    return BuildSummary(
        collection_id=spec.collection_id,
        ensemble_id=ensemble_id,
        database_path=database_path.resolve(),
        database_sha256=_file_sha256(database_path),
        specification_sha256=spec.specification_sha256,
        generator_source_sha256=str(metadata["generator_source_sha256"]),
        validation=validation,
    )


def validate_existing_collection(spec: GraphCollectionSpec, database_path: Path) -> BuildSummary:
    """Validate an existing collection without changing registry rows or statuses."""
    if not database_path.is_file():
        raise FileNotFoundError(f"registry does not exist: {database_path}")
    registry = GraphRegistryDatabase(database_path)
    ensemble_id = _find_ensemble_id(registry, spec.collection_id)
    validation = validate_collection(registry, spec, ensemble_id)
    _checkpoint_wal(database_path)
    return BuildSummary(
        collection_id=spec.collection_id,
        ensemble_id=ensemble_id,
        database_path=database_path.resolve(),
        database_sha256=_file_sha256(database_path),
        specification_sha256=spec.specification_sha256,
        generator_source_sha256=_generator_source_sha256(),
        validation=validation,
    )


def manifest_payload(spec: GraphCollectionSpec, summary: BuildSummary) -> dict[str, object]:
    """Return the complete machine-readable provenance record."""
    return {
        "beta_keys": list(spec.beta_keys),
        "betas": list(spec.betas),
        "collection_id": summary.collection_id,
        "created_or_validated_utc": _utc_now(),
        "database": _display_path(summary.database_path),
        "database_sha256": summary.database_sha256,
        "environment": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
        },
        "generator_source_sha256": summary.generator_source_sha256,
        "graph_k": spec.graph_k,
        "graphs_per_cell": spec.graphs_per_cell,
        "master_seed": spec.master_seed,
        "mean_degree": 2 * spec.graph_k,
        "schema_version": spec.schema_version,
        "seed_derivation": spec.seed_derivation,
        "sizes": list(spec.sizes),
        "specification_sha256": summary.specification_sha256,
        "validation": asdict(summary.validation),
    }


def status_markdown(spec: GraphCollectionSpec, summary: BuildSummary) -> str:
    validation = summary.validation
    return f"""# Graph registry status

Status: **seed collection complete and validated**

- Collection: `{summary.collection_id}`
- Sizes: `{", ".join(str(n) for n in spec.sizes)}`
- Beta grid: `{len(spec.beta_keys)}` values (`0` plus `{len(spec.beta_keys) - 1}` log-spaced
  values from `0.005` through `1`)
- Cells: `{validation.cell_count}`
- Indexed graph draws per cell: `{spec.graphs_per_cell}`
- Total indexed graph draws: `{validation.graph_count:,}`
- Graph construction: Watts-Strogatz rewiring of `C(n,2)`, mean degree `4`
- Seed-table SHA-256: `{validation.seed_content_sha256}`
- SQLite integrity: `{validation.sqlite_integrity}`; foreign-key violations:
  `{validation.foreign_key_violations}`
- Representative graph reconstructions checked: `{validation.reconstructed_graphs}`

The SQLite catalog is at `{_display_path(summary.database_path)}`.  Its graph rows are the
canonical collection.  Edge banks, invariants, and circuit results are separate,
versioned additions linked to these immutable graph identities.  Equal seeds or
equal topologies never merge records.  At `beta=0`, all indexed draws correctly
reconstruct the same unrewired `C(n,2)` geometry.
"""


def write_reports(
    spec: GraphCollectionSpec,
    summary: BuildSummary,
    *,
    publish_status: bool,
) -> tuple[Path, Path]:
    """Atomically write the local manifest and human-readable status report."""
    output = summary.database_path.parent
    manifest_path = output / "manifest.json"
    status_path = output / "STATUS.md"
    payload = manifest_payload(spec, summary)
    _atomic_text(manifest_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rendered = status_markdown(spec, summary)
    _atomic_text(status_path, rendered)
    if publish_status:
        _atomic_text(PUBLIC_STATUS, rendered)
    return manifest_path, status_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "production"), default="smoke")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--confirm-production", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = production_spec() if args.profile == "production" else smoke_spec()
    database_path = (args.database or default_database_path(spec)).resolve()
    if (
        args.profile == "production"
        and not args.confirm_production
        and not args.validate_only
        and not args.dry_run
    ):
        raise SystemExit("production creation requires --confirm-production")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "collection_id": spec.collection_id,
                    "database": str(database_path),
                    "n_beta": len(spec.beta_keys),
                    "n_cells": spec.n_cells,
                    "n_graphs": spec.n_graphs,
                    "seed_content_sha256": spec.seed_content_sha256(),
                    "sizes": spec.sizes,
                },
                indent=2,
            )
        )
        return 0

    last_reported = 0

    def report(completed: int, total: int) -> None:
        nonlocal last_reported
        percent = completed * 100 // total
        if percent >= last_reported + 5 or completed == total:
            last_reported = percent
            print(f"registered {completed:,}/{total:,} graph draws ({percent}%)", flush=True)

    if args.validate_only:
        if not database_path.is_file():
            raise SystemExit(f"registry does not exist: {database_path}")
        summary = validate_existing_collection(spec, database_path)
    else:
        summary = build_collection(
            spec,
            database_path,
            batch_size=args.batch_size,
            progress=report,
        )
    manifest_path, status_path = write_reports(
        spec,
        summary,
        publish_status=args.profile == "production",
    )
    print(f"validated {summary.validation.graph_count:,} graph draws")
    print(f"seed content SHA-256: {summary.validation.seed_content_sha256}")
    print(f"database: {summary.database_path}")
    print(f"manifest: {manifest_path}")
    print(f"status: {status_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
