"""The indexed, crash-safe tableau store of the Watts–Strogatz study (v1.4 §1, §5.3, §8; design §4).

One store holds one experiment: every manifest revision, the graph records, every trajectory with
its per-layer ``k`` trace and exact first passages, the checkpoint tableau dictionary
$\\mathcal D_\\xi = \\{t \\mapsto \\widetilde{\\mathsf S}_t\\}$ of every trajectory (the canonical
joint phase-free row space and the physical code at each checkpoint), and every observable value,
each keyed by its full identity. ``docs/study/05-store-and-reproducibility.md`` has the record
hierarchy, the schema with every column, the transaction and lock protocol and the proofs; this
docstring fixes the conventions.

**Layout.** A store is a directory: ``study.sqlite`` (SQLite in WAL mode, the standard library's
``sqlite3``; :data:`STORE_FILE`) and, while a writer has it open, ``.lock`` (the lock file of
:mod:`sparsegf2.api`, naming the writing process; a lock left by a process that has ended on this
machine is taken over). Readers need no lock and may read while a writer writes.

**Identities.** A trajectory is ``(protocol, n, beta, p, g, c, r)`` (the
:class:`~sparsegf2.study.identity.TrajectoryKey` of the store's experiment, tokens as their
canonical strings); a snapshot adds the depth ``t``; an observation adds ``(observable, version,
params, context)`` (the context is empty for a snapshot value, and for a trajectory-level value
the digest of the horizon and the peak domain it was derived over); a graph is ``(n, beta, g)``. Each identity is stored once and never
overwritten: meeting it again compares the content (digests, decoded traces, values) and raises
``ValueError`` naming the identity on any difference. The single exception fills a missing joint
tableau into a snapshot that held only the physical code (a revision switched
``tableau_snapshots`` on), after checking that its physical code is the stored one.

**Encodings** (all blobs are zlib streams; the compression level never changes a decoded value):

* ``graphs.edges``: the canonical edge list as int32 little-endian ``(m, 2)``, row-major;
* ``traces.k_points``: the change points of :func:`~sparsegf2.study.observables.k_change_points`
  as int32 little-endian ``(J, 2)``; ``traces.tau``: the ``n + 1`` exact first-passage depths as
  int32 little-endian, ``-1`` for a censored threshold;
* ``snapshots.joint`` / ``snapshots.phys``: the canonical bytes of
  :meth:`~sparsegf2.tableau.PhaseFreeTableau.to_bytes` (format ``SGF2RS1``); ``joint_digest`` /
  ``phys_digest`` are the SHA-256 of the uncompressed bytes, checked on every read;
* ``observations.data``: canonical JSON (:func:`~sparsegf2.study.identity.canonical_json`) of
  the value's details, NaN and infinities written as ``null``; ``observations.value`` is a
  float, SQL ``NULL`` standing for NaN (SQLite has no NaN; infinities are kept).

**Transactions.** Every write runs in one ``BEGIN IMMEDIATE`` ... ``COMMIT`` transaction
(:meth:`TableauStore.transaction`); an exception rolls it back, a killed process leaves it
uncommitted (WAL recovery discards it), so a reader sees a unit completely or not at all. A
read of many tables (export, the logical digest, ``check``) runs in one read transaction
(:meth:`TableauStore.reading`), so it sees one committed state while a writer commits.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2 import graphs, mipt
from sparsegf2._version import __version__
from sparsegf2.codes import StabilizerCode
from sparsegf2.study import observables as ob
from sparsegf2.study import stats
from sparsegf2.study.identity import (
    STORE_SCHEMA_VERSION,
    GraphRecord,
    Token,
    TrajectoryKey,
    canonical_json,
    check_protocol,
    circulant_edges,
    sha256_hex,
    version_record,
)
from sparsegf2.study.manifest import ExperimentManifest, check_extension
from sparsegf2.tableau import PhaseFreeTableau

__all__ = [
    "EXPORT_FORMAT",
    "GRAPH_MEASURES_VERSION",
    "SNAPSHOT_ABSENT",
    "SNAPSHOT_STATUSES",
    "SNAPSHOT_STORED",
    "STORE_FILE",
    "SnapshotRecord",
    "TableauStore",
    "TraceRecord",
    "TrajectoryRecord",
    "graph_measures",
]

#: The SQLite file inside a store directory.
STORE_FILE = "study.sqlite"

#: The version of :func:`graph_measures` (stored with every graph record's measurements).
GRAPH_MEASURES_VERSION = "ws-graph-measures-1"

#: The layout written by :meth:`TableauStore.export`.
EXPORT_FORMAT = "sparsegf2.study.export/2"

#: The qubit order of every stored row space (design §2.1; ``identity.ROWSPACE_FORMAT_VERSION``
#: says how the rows are written, this says what the columns are).
QUBIT_ORDER = (
    "F: system 0..n-1, then reference n..2n-1, reference n+i Bell-paired with system i at t = 0; "
    "P: system 0..n-1; physical codes: system 0..n-1"
)

#: The canonical text of a probability token (``identity.Token.token``) hashed into every seed:
#: the exact decimal without exponent or trailing zeros when it terminates, else ``num/den`` in
#: lowest terms. A change of this rule changes every seed and needs new seed tags.
TOKEN_FORMAT = "decimal-or-lowest-terms-rational-1"

#: The first word of every outcome seed of the outcome-aware replay (``b"WSO1"``, big-endian):
#: a domain of its own, disjoint from the graph (WSG1), run (WSC1) and region (WSRO) seeds.
OUTCOME_SEED_TAG = 0x57534F31

#: The first word of the seeds of the sampler diagnostics (``b"WSSD"``, big-endian).
DIAGNOSTICS_SEED_TAG = 0x57535344

#: The canonical bytes of a signed stabilizer group recorded by the outcome-aware replay.
SIGNED_FORMAT = "SGF2SG1"

#: The estimator of :meth:`TableauStore.aggregate` (v1.4 §6.3, §6.4).
ESTIMATOR_VERSION = "stats.nested_mean-1"

#: A snapshot row whose joint row space and physical code are stored.
SNAPSHOT_STORED = "stored"
#: A snapshot row of a store that keeps physical codes only (``tableau_snapshots = false``).
SNAPSHOT_PHYS_ONLY = "phys_only"
#: A checkpoint after a ``stop_when_purified`` stop: the state was not evolved, nothing is stored.
SNAPSHOT_ABSENT = "absent_after_purification"
SNAPSHOT_STATUSES = (SNAPSHOT_STORED, SNAPSHOT_PHYS_ONLY, SNAPSHOT_ABSENT)

#: The status of an observation (v1.4 §5.2, §5.5, §5.6, §5.8; SQLite has no NaN, so ``value``
#: alone cannot say why it is missing): ``ok`` (a finite value), ``infinite`` (``+inf``, e.g.
#: ``d_conn`` without a connected failing set), ``undefined_no_code`` (a quantity without
#: meaning at ``k = 0``, the raw ``d_cyc``; the zero extension is in the data), ``missing`` (a
#: peak with ``d* = 0`` or an invalid trace), ``censored`` (a first passage not reached by the
#: horizon), ``vector`` (the value is in the data), ``absent_after_purification`` (the state
#: was not evolved after a stop), ``undefined`` (any other NaN).
OBSERVATION_STATUSES = (
    "ok",
    "infinite",
    "undefined_no_code",
    "missing",
    "censored",
    "vector",
    "absent_after_purification",
    "undefined",
)

_SCHEMA = """
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE manifests (
    revision INTEGER PRIMARY KEY,
    digest TEXT NOT NULL UNIQUE,
    json TEXT NOT NULL,
    created_utc TEXT NOT NULL,
    sparsegf2_version TEXT NOT NULL,
    git_commit TEXT
);
CREATE TABLE graphs (
    n INTEGER NOT NULL,
    beta TEXT NOT NULL,
    g INTEGER NOT NULL,
    seed TEXT NOT NULL,
    generator TEXT NOT NULL,
    edge_order TEXT NOT NULL,
    num_edges INTEGER NOT NULL,
    edges BLOB NOT NULL,
    digest TEXT NOT NULL,
    measures TEXT NOT NULL,
    measures_version TEXT NOT NULL,
    PRIMARY KEY (n, beta, g)
);
CREATE TABLE requirements (
    digest TEXT PRIMARY KEY,
    json TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE units (
    unit_id INTEGER PRIMARY KEY,
    protocol TEXT NOT NULL CHECK (protocol IN ('F', 'P')),
    n INTEGER NOT NULL,
    beta TEXT NOT NULL,
    p TEXT NOT NULL,
    graphs TEXT NOT NULL,
    circuits TEXT NOT NULL,
    layers INTEGER NOT NULL,
    requirements TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('done', 'failed')),
    revision INTEGER NOT NULL REFERENCES manifests (revision),
    started_utc TEXT,
    finished_utc TEXT,
    elapsed_seconds REAL,
    threads INTEGER,
    trajectories INTEGER NOT NULL,
    error TEXT,
    sparsegf2_version TEXT NOT NULL,
    git_commit TEXT,
    git_dirty INTEGER,
    randomness_contract INTEGER NOT NULL,
    environment TEXT NOT NULL,
    environment_digest TEXT NOT NULL
);
CREATE TABLE trajectories (
    tid INTEGER PRIMARY KEY,
    protocol TEXT NOT NULL CHECK (protocol IN ('F', 'P')),
    n INTEGER NOT NULL,
    beta TEXT NOT NULL,
    p TEXT NOT NULL,
    g INTEGER NOT NULL,
    c INTEGER NOT NULL,
    r INTEGER NOT NULL,
    u INTEGER NOT NULL,
    p_float_hex TEXT NOT NULL,
    run_seed TEXT NOT NULL,
    graph_digest TEXT NOT NULL,
    dynamics_version TEXT NOT NULL,
    unit_id INTEGER NOT NULL REFERENCES units (unit_id),
    revision INTEGER NOT NULL REFERENCES manifests (revision),
    UNIQUE (protocol, n, beta, p, g, c, r),
    FOREIGN KEY (n, beta, g) REFERENCES graphs (n, beta, g)
);
CREATE TABLE traces (
    tid INTEGER NOT NULL REFERENCES trajectories (tid),
    layers INTEGER NOT NULL,
    final_layer INTEGER NOT NULL,
    k_points BLOB,
    tau BLOB,
    tau0 INTEGER,
    tau1 INTEGER,
    censored0 INTEGER,
    status TEXT NOT NULL
        CHECK (status IN ('purified', 'stopped_at_purification', 'censored', 'no_reference')),
    num_measurements INTEGER NOT NULL,
    unit_id INTEGER NOT NULL REFERENCES units (unit_id),
    revision INTEGER NOT NULL REFERENCES manifests (revision),
    UNIQUE (tid, layers)
);
CREATE TABLE completions (
    tid INTEGER NOT NULL REFERENCES trajectories (tid),
    requirements TEXT NOT NULL REFERENCES requirements (digest),
    unit_id INTEGER NOT NULL REFERENCES units (unit_id),
    revision INTEGER NOT NULL REFERENCES manifests (revision),
    PRIMARY KEY (tid, requirements)
) WITHOUT ROWID;
CREATE TABLE derivations (
    tid INTEGER NOT NULL REFERENCES trajectories (tid),
    derivation TEXT NOT NULL REFERENCES requirements (digest),
    unit_id INTEGER REFERENCES units (unit_id),
    revision INTEGER NOT NULL REFERENCES manifests (revision),
    PRIMARY KEY (tid, derivation)
) WITHOUT ROWID;
CREATE TABLE snapshots (
    tid INTEGER NOT NULL REFERENCES trajectories (tid),
    t INTEGER NOT NULL,
    num_qubits INTEGER NOT NULL,
    rank INTEGER,
    k INTEGER,
    joint BLOB,
    joint_digest TEXT,
    phys BLOB,
    phys_digest TEXT,
    status TEXT NOT NULL
        CHECK (status IN ('stored', 'phys_only', 'absent_after_purification')),
    unit_id INTEGER NOT NULL REFERENCES units (unit_id),
    UNIQUE (tid, t)
);
CREATE TABLE observable_defs (
    observable TEXT NOT NULL,
    version TEXT NOT NULL,
    params TEXT NOT NULL,
    params_json TEXT NOT NULL,
    level TEXT NOT NULL,
    inputs TEXT NOT NULL,
    definition TEXT NOT NULL,
    created_utc TEXT NOT NULL,
    PRIMARY KEY (observable, version, params)
) WITHOUT ROWID;
CREATE TABLE observations (
    tid INTEGER NOT NULL REFERENCES trajectories (tid),
    t INTEGER NOT NULL,
    observable TEXT NOT NULL,
    version TEXT NOT NULL,
    params TEXT NOT NULL,
    context TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'infinite', 'undefined_no_code', 'missing',
        'censored', 'vector', 'absent_after_purification', 'undefined')),
    value REAL,
    se REAL,
    data BLOB,
    method TEXT NOT NULL,
    unit_id INTEGER REFERENCES units (unit_id),
    UNIQUE (tid, t, observable, version, params, context),
    FOREIGN KEY (observable, version, params) REFERENCES observable_defs (observable, version, params)
);
CREATE TABLE aggregates (
    protocol TEXT NOT NULL,
    n INTEGER NOT NULL,
    beta TEXT NOT NULL,
    p TEXT NOT NULL,
    observable TEXT NOT NULL,
    version TEXT NOT NULL,
    params TEXT NOT NULL,
    selection TEXT NOT NULL,
    t INTEGER NOT NULL,
    context TEXT NOT NULL,
    estimator TEXT NOT NULL,
    policy TEXT NOT NULL,
    sample_set TEXT NOT NULL,
    mean REAL,
    se REAL,
    graphs INTEGER NOT NULL,
    contributing_graphs INTEGER NOT NULL,
    trajectories INTEGER NOT NULL,
    contributing_trajectories INTEGER NOT NULL,
    data BLOB,
    created_utc TEXT NOT NULL,
    PRIMARY KEY (protocol, n, beta, p, observable, version, params, selection, t, context,
        estimator, policy, sample_set)
) WITHOUT ROWID;
CREATE TABLE signed_runs (
    tid INTEGER NOT NULL REFERENCES trajectories (tid),
    outcome_seed TEXT NOT NULL,
    layers INTEGER NOT NULL,
    depths TEXT NOT NULL,
    outcomes BLOB NOT NULL,
    random BLOB NOT NULL,
    signed_digests TEXT NOT NULL,
    format TEXT NOT NULL,
    created_utc TEXT NOT NULL,
    PRIMARY KEY (tid, outcome_seed, layers, depths)
);
CREATE TABLE sampler_diagnostics (
    n INTEGER NOT NULL,
    beta TEXT NOT NULL,
    g INTEGER NOT NULL,
    probe TEXT NOT NULL,
    config TEXT NOT NULL,
    m INTEGER NOT NULL,
    coupling REAL NOT NULL,
    method TEXT NOT NULL,
    chains INTEGER NOT NULL,
    samples INTEGER NOT NULL,
    seed TEXT NOT NULL,
    ess REAL,
    tau_int REAL,
    acceptance REAL,
    exchange_acceptance REAL,
    r_hat REAL,
    r_hat_limit REAL,
    notes TEXT NOT NULL,
    data BLOB,
    created_utc TEXT NOT NULL,
    PRIMARY KEY (n, beta, g, probe, config, m, coupling)
);
CREATE INDEX observations_by_observable ON observations (observable, version, params, tid, t);
CREATE INDEX traces_by_tid ON traces (tid, layers);
"""

# Append-only records (G-RUN-1): no UPDATE and no DELETE, except the one fill-in of a snapshot
# that kept its physical code only (its joint state, status and rank; nothing else changes).
_KEPT_TABLES = (
    "manifests",
    "graphs",
    "requirements",
    "trajectories",
    "traces",
    "completions",
    "derivations",
    "observable_defs",
    "observations",
    "aggregates",
    "signed_runs",
    "sampler_diagnostics",
)
_TRIGGERS = [
    *(
        f"CREATE TRIGGER {table}_never_updated BEFORE UPDATE ON {table} BEGIN "
        f"SELECT RAISE(ABORT, 'stored {table} rows are never updated'); END"
        for table in _KEPT_TABLES
    ),
    *(
        f"CREATE TRIGGER {table}_never_deleted BEFORE DELETE ON {table} BEGIN "
        f"SELECT RAISE(ABORT, 'stored {table} rows are never deleted'); END"
        for table in (*_KEPT_TABLES, "snapshots", "units")
    ),
    "CREATE TRIGGER snapshots_fill_in_only BEFORE UPDATE ON snapshots WHEN "
    "OLD.status != 'phys_only' OR NEW.status != 'stored' OR OLD.joint IS NOT NULL "
    "OR NEW.tid != OLD.tid OR NEW.t != OLD.t OR NEW.num_qubits != OLD.num_qubits "
    "OR NEW.k IS NOT OLD.k OR NEW.phys IS NOT OLD.phys OR NEW.phys_digest IS NOT OLD.phys_digest "
    "BEGIN SELECT RAISE(ABORT, 'a stored snapshot is never updated (only the joint state of a "
    "phys_only row is filled in)'); END",
]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


_COMMIT: str | None = None
_COMMIT_READ = False
_DIRTY: bool | None = None


def _git_commit() -> str | None:
    """The commit of the code this process loaded (read once per process, at first use)."""
    global _COMMIT, _COMMIT_READ, _DIRTY
    if not _COMMIT_READ:
        from sparsegf2 import api

        _COMMIT = api._git_commit()
        _DIRTY = _git_dirty() if _COMMIT is not None else None
        _COMMIT_READ = True
    return _COMMIT


def _git_dirty() -> bool | None:
    """Whether the checkout of the package has uncommitted changes to tracked files
    (``git status --porcelain --untracked-files=no``); None if git cannot tell."""
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return None if proc.returncode != 0 else bool(proc.stdout.strip())


def git_provenance() -> dict[str, Any]:
    """The commit of the running code and whether its checkout had uncommitted changes."""
    commit = _git_commit()
    return {"git_commit": commit, "git_dirty": _DIRTY}


def environment() -> dict[str, Any]:
    """The software of the running process (G-RUN-3): Python, numpy, numba, llvmlite, the
    package and the platform. Each unit records it and its digest (:func:`environment_digest`)."""
    import platform

    import llvmlite
    import numba

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "llvmlite": llvmlite.__version__,
        "sparsegf2": __version__,
        "platform": platform.platform(),
    }


def environment_digest(env: Mapping[str, Any]) -> str:
    """The SHA-256 of the canonical JSON of an :func:`environment` record."""
    return sha256_hex(canonical_json(dict(env)).encode("utf-8"))


def store_versions() -> dict[str, Any]:
    """Every version a stored record depends on (G-ID-1): ``identity.version_record()`` (the
    graph generator, the Clifford numbering, the dynamics, the row-space format, the store
    schema, the graph and run seed tags, ``RANDOMNESS_CONTRACT``, the package) and the region
    and peak-bootstrap seeds' tags and lengths, the outcome and diagnostics seed tags, the token
    format, the qubit order, the graph measurements, the signed-group format and the estimator
    of aggregates."""
    out = dict(version_record())
    out.update(
        region_seed_tag=ob.REGION_SEED_TAG,
        region_seed_words=ob.REGION_SEED_WORDS,
        bootstrap_seed_tag=ob.BOOTSTRAP_SEED_TAG,
        bootstrap_seed_words=ob.BOOTSTRAP_SEED_WORDS,
        outcome_seed_tag=OUTCOME_SEED_TAG,
        diagnostics_seed_tag=DIAGNOSTICS_SEED_TAG,
        token_format=TOKEN_FORMAT,
        qubit_order=QUBIT_ORDER,
        graph_measures=GRAPH_MEASURES_VERSION,
        signed_format=SIGNED_FORMAT,
        aggregate_estimator=ESTIMATOR_VERSION,
    )
    return out


# ---------------------------------------------------------------------------------------------
# encodings
# ---------------------------------------------------------------------------------------------


def _int32_blob(values: Any, level: int) -> bytes:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    if arr.size and (arr.min() < -(2**31) or arr.max() >= 2**31):
        raise ValueError("a stored integer array must fit int32")
    return zlib.compress(arr.astype("<i4").tobytes(), level)


def _int32_array(blob: bytes, what: str, columns: int | None = None) -> np.ndarray:
    try:
        raw = zlib.decompress(blob)
    except zlib.error:
        raise ValueError(f"{what}: the stored blob is corrupt (not a zlib stream)") from None
    if len(raw) % 4:
        raise ValueError(f"{what}: the stored blob is corrupt ({len(raw)} bytes)")
    arr: np.ndarray = np.frombuffer(raw, dtype="<i4").astype(np.int64)
    if columns is not None:
        if arr.size % columns:
            raise ValueError(f"{what}: the stored blob is corrupt (shape)")
        arr = arr.reshape(-1, columns)
    return arr


def _plain_array(arr: np.ndarray) -> Any:
    """A numpy array as nested lists of Python numbers, NaN and infinities as None, ``-0.0`` as
    ``0.0`` (the conventions of canonical JSON), without a Python call per entry when the
    array is finite."""
    if arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.integer):
        return arr.tolist()
    if np.issubdtype(arr.dtype, np.floating):
        values = np.asarray(arr, dtype=np.float64) + 0.0  # -0.0 + 0.0 is 0.0
        if np.isfinite(values).all():
            return values.tolist()
        return np.where(np.isfinite(values), values, None).tolist()  # an object array
    return _plain(arr.tolist())


def _plain(obj: Any) -> Any:
    """Plain JSON data: numpy as Python, tuples as lists, NaN and infinities as None."""
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return _plain_array(obj)
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return (x + 0.0) if math.isfinite(x) else None
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, Token):
        return obj.token
    raise ValueError(f"cannot store a {type(obj).__name__} in an observation's data")


def _data_blob(data: Any, level: int) -> bytes | None:
    """``zlib`` of the canonical JSON of ``data`` (sorted keys, no spaces, ASCII), NaN as null.

    The same text as :func:`~sparsegf2.study.identity.canonical_json` of the plain data, made
    with ``json.dumps`` directly: :func:`_plain` has already reduced everything to dicts, lists,
    strings, finite floats, ints, booleans and None.
    """
    if data is None:
        return None
    text = json.dumps(
        _plain(data), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return zlib.compress(text.encode("utf-8"), level)


def _data_text(blob: bytes | None, what: str) -> str | None:
    if blob is None:
        return None
    try:
        return zlib.decompress(blob).decode("utf-8")
    except (zlib.error, UnicodeDecodeError):
        raise ValueError(f"{what}: the stored data blob is corrupt") from None


def _sql_float(x: Any) -> float | None:
    """A float for SQLite: NaN (and None) as NULL; infinities are kept."""
    if x is None:
        return None
    v = float(x)
    return None if math.isnan(v) else v


def _same_float(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a == b or (math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0))


def _rowspace_blob(tab: PhaseFreeTableau, level: int) -> tuple[bytes, str]:
    raw = tab.to_bytes()
    return zlib.compress(raw, level), sha256_hex(raw)


def _decode_rowspace(blob: bytes | None, digest: str | None, what: str) -> PhaseFreeTableau:
    """The row space of a stored blob, after checking its digest (the integrity check on read)."""
    if blob is None or digest is None:
        raise ValueError(f"{what}: nothing is stored")
    try:
        raw = zlib.decompress(blob)
    except zlib.error:
        raise ValueError(f"{what}: the stored blob is corrupt (not a zlib stream)") from None
    if sha256_hex(raw) != digest:
        raise ValueError(
            f"{what}: the stored bytes do not match their digest {digest[:16]}...: the store was "
            "altered or damaged"
        )
    try:
        return PhaseFreeTableau.from_bytes(raw)
    except ValueError as exc:
        raise ValueError(f"{what}: {exc}") from None


# ---------------------------------------------------------------------------------------------
# rows ready to store (validated and encoded outside a transaction)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SnapshotRow:
    """One snapshot ready to be written: its depth, status, ``N``, rank, ``k`` and the
    compressed canonical bytes with their digests (:func:`snapshot_row`)."""

    t: int
    status: str
    num_qubits: int
    rank: int | None
    k: int | None
    joint: bytes | None
    joint_digest: str | None
    phys: bytes | None
    phys_digest: str | None


def snapshot_row(
    key: TrajectoryKey,
    t: int,
    *,
    status: str,
    num_qubits: int,
    k: int | None,
    joint: PhaseFreeTableau | None,
    phys: PhaseFreeTableau | None,
    level: int,
    check_physical: bool = True,
) -> SnapshotRow:
    """Validate one snapshot and encode it (zlib at ``level``, SHA-256 of the canonical bytes).

    Checked: a stored joint state is pure (rank ``N``, isotropic); a protocol-F physical code
    has rank ``n - k`` and commutes; with ``check_physical`` the physical code is recomputed from
    the joint state and must equal it (the runner derives one from the other and skips it). A
    pure function of its arguments that touches no database: the runner prepares the rows of a
    unit on several threads (the isotropy kernel, zlib and SHA-256 release the GIL).
    """
    what = f"snapshot of {key} at t = {t}"
    if status not in SNAPSHOT_STATUSES:
        raise ValueError(f"{what}: unknown status {status!r}")
    jb = jd = pb = pd = None
    rank = None
    if status == SNAPSHOT_PHYS_ONLY and (joint is not None or key.protocol != "F"):
        raise ValueError(f"{what}: a phys_only snapshot holds a protocol-F physical code only")
    if status == SNAPSHOT_STORED and key.protocol == "F" and joint is None:
        raise ValueError(f"{what}: a stored protocol-F snapshot holds its joint state")
    if status in (SNAPSHOT_STORED, SNAPSHOT_PHYS_ONLY):
        if joint is not None:
            if joint.num_qubits != num_qubits or not joint.is_pure():
                raise ValueError(
                    f"{what}: the joint state must be pure on {num_qubits} qubits (rank N and "
                    f"isotropic), got rank {joint.rank} on {joint.num_qubits}"
                )
            jb, jd = _rowspace_blob(joint, level)
            rank = joint.rank
        if key.protocol == "F":
            if phys is None or k is None:
                raise ValueError(f"{what}: a protocol-F snapshot needs its physical code and k")
            if phys.rank != key.n - k or phys.num_qubits != key.n or not phys.is_isotropic():
                raise ValueError(
                    f"{what}: the physical code must have rank n - k = {key.n - k} on "
                    f"{key.n} qubits and commute, got rank {phys.rank}"
                )
            if (
                check_physical
                and joint is not None
                and not joint.physical_tableau(key.n).same_space(phys)
            ):
                raise ValueError(f"{what}: the physical code is not that of the joint state")
            pb, pd = _rowspace_blob(phys, level)
        elif joint is None:
            raise ValueError(f"{what}: a protocol-{key.protocol} snapshot needs its joint state")
    elif joint is not None or phys is not None:
        raise ValueError(f"{what}: an absent snapshot stores nothing")
    return SnapshotRow(
        t=int(t),
        status=status,
        num_qubits=int(num_qubits),
        rank=rank,
        k=None if k is None else int(k),
        joint=jb,
        joint_digest=jd,
        phys=pb,
        phys_digest=pd,
    )


def _compare_snapshot(what: str, stored: Sequence[Any], row: SnapshotRow) -> None:
    """Raise unless the stored ``(status, num_qubits, rank, k, joint_digest, phys_digest)``
    holds the content of ``row`` (a code-only row meets a stored joint row by its code)."""
    status = row.status
    if stored[0] == SNAPSHOT_STORED and status == SNAPSHOT_PHYS_ONLY:
        status = SNAPSHOT_STORED  # a code-only rerun of a stored row: compare the code
    new = (status, row.num_qubits, row.k, row.phys_digest)
    old = (stored[0], stored[1], stored[3], stored[5])
    for name, a, b in zip(("status", "num_qubits", "k", "phys_digest"), old, new, strict=True):
        if a != b:
            raise _differs(what, name, a, b)
    if stored[4] is not None and row.joint_digest is not None and stored[4] != row.joint_digest:
        raise _differs(what, "joint_digest", stored[4], row.joint_digest)


@dataclass(frozen=True, eq=False)
class ObservationRow:
    """One observation ready to be written (:func:`observation_row`): its key part
    ``(t, observable, version, params, context)``, status, value and error in their SQL form
    (NaN as None), the compressed canonical JSON data and the method."""

    t: int
    observable: str
    version: str
    params: str
    context: str
    status: str
    value: float | None
    se: float | None
    data: bytes | None
    method: str


def observation_row(
    t: int,
    observable: str,
    params: Mapping[str, Any] | str | None,
    *,
    value: Any,
    se: Any,
    data: Any,
    method: str,
    level: int,
    context: str = "",
    version: str | None = None,
    status: str | None = None,
) -> ObservationRow:
    """One value under its key, encoded for the store (:meth:`_Transaction.put_observation`).

    ``params`` is a dict (resolved with the catalog's defaults, :func:`observables.params_key`)
    or the canonical text :func:`observables.params_key` returned. ``status``
    (:data:`OBSERVATION_STATUSES`) says what the value is; by default ``ok`` for a finite value,
    ``infinite`` for an infinite one and ``undefined`` for NaN. Touches no database.
    """
    entry = ob.get(observable, version)
    text = (
        params
        if isinstance(params, str)
        else ob.params_key(observable, params, version=entry.version)
    )
    v, s = _sql_float(value), _sql_float(se)
    if status is None:
        status = "undefined" if v is None else ("infinite" if math.isinf(v) else "ok")
    if status not in OBSERVATION_STATUSES:
        raise ValueError(f"status must be one of {list(OBSERVATION_STATUSES)}, got {status!r}")
    if status == "ok" and (v is None or math.isinf(v)):
        raise ValueError(f"an observation with status 'ok' has a finite value, got {value!r}")
    return ObservationRow(
        t=int(t),
        observable=observable,
        version=entry.version,
        params=text,
        context=context,
        status=status,
        value=v,
        se=s,
        data=_data_blob(data, level),
        method=method,
    )


def _compare_observation(
    what: str, stored: Sequence[Any], row: ObservationRow, compare: str = "all"
) -> None:
    """Raise unless the stored ``(value, se, method, data, status)`` equals ``row``: the value and
    error bit for bit (NaN equal to NaN) and the status, and with ``compare="all"`` also the
    method and the decoded data."""
    if not _same_float(stored[0], row.value):
        raise _differs(what, "value", stored[0], row.value)
    if not _same_float(stored[1], row.se):
        raise _differs(what, "standard error", stored[1], row.se)
    if stored[4] != row.status:
        raise _differs(what, "status", stored[4], row.status)
    if compare == "value":
        return
    if stored[2] != row.method:
        raise _differs(what, "method", stored[2], row.method)
    old_text = _data_text(stored[3], what)
    new_text = _data_text(row.data, what)
    if old_text != new_text:
        raise _differs(what, "data", old_text, new_text)


# ---------------------------------------------------------------------------------------------
# graph measurements (design §12; queued study v0.3 §4)
# ---------------------------------------------------------------------------------------------


def graph_measures(edges: Any, n: int) -> dict[str, Any]:
    r"""The four graph measurements of design §12, from the edge list alone.

    For a simple graph $G$ on $V = \{0, \ldots, n-1\}$ with edge set $E(G)$:

    * ``q`` $= q_G = |E(G) \setminus E(C(n,2))| / (2n)$, the fraction of edges that are not edges
      of the circulant ring $C(n, 2)$ (``rewired_edges`` $= |E(G) \setminus E(C(n, 2))|$);
    * ``mean_distance`` $= \bar d_G$, the mean shortest-path distance over the
      ``reachable_pairs`` unordered pairs of distinct vertices joined by a path (NaN when there is
      no such pair), and ``components``, the number of connected components;
    * ``lambda2`` $= \lambda_2(D - A)$, the second-smallest eigenvalue of the unnormalized
      Laplacian, set to exactly 0 when $G$ is disconnected (a LAPACK eigenvalue otherwise:
      its last bits may differ between platforms);
    * ``degree_variance`` $= \frac1n \sum_v \deg(v)^2 - \big(\frac1n \sum_v \deg(v)\big)^2$, the
      population variance of the degrees (from the exact integer sums).
    """
    n = int(n)
    e = graphs.from_edges(np.asarray(edges, dtype=np.int64).reshape(-1, 2), n)
    ring = {(int(a), int(b)) for a, b in circulant_edges(n)}
    rewired = sum(1 for a, b in e.tolist() if (a, b) not in ring)
    deg = graphs.degrees(e, n).astype(np.int64)
    parent = list(range(n))

    def root(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b in e.tolist():
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    components = len({root(v) for v in range(n)})
    from sparsegf2._kernel.graphstats import path_length_sum

    indptr, indices = graphs._csr(e, n)
    total, ordered = path_length_sum(indptr, indices, n)
    pairs = int(ordered) // 2
    if components > 1 or n < 2:
        lambda2 = 0.0
    else:
        adj = np.zeros((n, n))
        adj[e[:, 0], e[:, 1]] = 1.0
        adj[e[:, 1], e[:, 0]] = 1.0
        lam = np.linalg.eigvalsh(np.diag(adj.sum(axis=1)) - adj)
        lambda2 = float(max(lam[1], 0.0))
    s1, s2 = int(deg.sum()), int((deg * deg).sum())
    return {
        "n": n,
        "num_edges": int(len(e)),
        "rewired_edges": int(rewired),
        "q": rewired / (2 * n),
        "mean_distance": float(total) / float(ordered) if ordered else math.nan,
        "reachable_pairs": pairs,
        "components": int(components),
        "lambda2": lambda2,
        "degree_variance": s2 / n - (s1 / n) ** 2,
    }


# ---------------------------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryRecord:
    """The identity and provenance of one stored trajectory (v1.4 §7.2, §8 ``trajectory``)."""

    key: TrajectoryKey
    tid: int
    u: int
    p_float_hex: str
    run_seed: tuple[int, ...]
    graph_digest: str
    dynamics_version: str
    unit_id: int
    revision: int


@dataclass(frozen=True, eq=False)
class TraceRecord:
    """The per-layer record of one trajectory at one horizon (v1.4 §5.2, §5.3, §8 ``purification_trace``).

    ``k``: ``k_0, ..., k_T`` (int64, ``T = layers``; ``None`` for protocol P, which has no
    reference); ``tau``: the ``n + 1`` exact first passages, ``-1`` where censored at ``T``;
    ``tau0``, ``tau1``: ``tau[0]``, ``tau[1]``; ``censored0``: whether ``tau_0`` is censored;
    ``final_layer``: the last simulated layer (``tau_0`` when ``stop_when_purified`` stopped
    the trajectory); ``status``: ``"purified"``, ``"stopped_at_purification"``, ``"censored"``
    or ``"no_reference"`` (P); ``num_measurements`` through ``final_layer``.
    """

    key: TrajectoryKey
    layers: int
    final_layer: int
    k: np.ndarray | None
    tau: np.ndarray | None
    tau0: int | None
    tau1: int | None
    censored0: bool | None
    status: str
    num_measurements: int
    unit_id: int
    revision: int


@dataclass(frozen=True, eq=False)
class SnapshotRecord:
    """One entry ``t -> S_t`` of a trajectory's tableau dictionary (v1.4 §5.3, §8 ``snapshot``).

    ``status`` is :data:`SNAPSHOT_STORED` or :data:`SNAPSHOT_ABSENT` (after a
    ``stop_when_purified`` stop: nothing stored, ``k = 0``). ``num_qubits`` is ``N`` (``2n``
    for protocol F, ``n`` for P), ``rank`` the joint rank (``N``), ``k`` the logical qubits
    (``None`` for P). :meth:`tableau` and :meth:`physical_tableau` decode the compressed blobs and
    check them against their digests; they raise ``ValueError`` for an absent snapshot, and
    :meth:`tableau` for a store that kept physical codes only.
    """

    key: TrajectoryKey
    t: int
    status: str
    num_qubits: int
    rank: int | None
    k: int | None
    joint_digest: str | None
    phys_digest: str | None
    joint_blob: bytes | None
    phys_blob: bytes | None

    def _where(self) -> str:
        return f"snapshot of {self.key} at t = {self.t}"

    def _require_stored(self) -> None:
        if self.status == SNAPSHOT_ABSENT:
            raise ValueError(
                f"{self._where()} is {self.status}: the trajectory stopped when its reference "
                "purified, its state was not evolved further and nothing was saved at this depth "
                "(k = 0 there exactly); replay it with runner.replay for the continued state"
            )

    def tableau(self) -> PhaseFreeTableau:
        """The joint phase-free row space (all ``N`` qubits: system, then references)."""
        self._require_stored()
        if self.status == SNAPSHOT_PHYS_ONLY or self.joint_blob is None:
            raise ValueError(
                f"{self._where()}: this store kept the physical code only "
                "(observables.tableau_snapshots = false); use code(), or replay the joint state "
                "with runner.replay"
            )
        return _decode_rowspace(self.joint_blob, self.joint_digest, self._where())

    def physical_tableau(self) -> PhaseFreeTableau:
        """The physical stabilizer group $S_t^{phys}$ as a row space of the ``n`` system qubits."""
        self._require_stored()
        if self.key.protocol != "F":
            raise ValueError(
                f"{self._where()}: protocol {self.key.protocol} has no reference, hence no "
                "physical code; its tableau() is the pure state of the system"
            )
        return _decode_rowspace(self.phys_blob, self.phys_digest, self._where() + " (physical)")

    def code(self) -> StabilizerCode:
        """The physical code as a :class:`~sparsegf2.codes.StabilizerCode` (``n - k`` generators)."""
        phys = self.physical_tableau()
        return StabilizerCode.from_packed(np.array(phys.words), phys.num_qubits)


_SNAPSHOT_COLUMNS = (
    "s.t, s.status, s.num_qubits, s.rank, s.k, s.joint_digest, s.phys_digest, s.joint, s.phys"
)
_KEY_COLUMNS = "tr.protocol, tr.n, tr.beta, tr.p, tr.g, tr.c, tr.r"


# ---------------------------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------------------------


class TableauStore:
    """The SQLite store of one experiment (see the module docstring).

    Build it with :meth:`create` (a new directory, from the first manifest revision) or
    :meth:`open` (``write=True`` for a writer, which takes the directory's lock; readers take
    none). Close it with :meth:`close` or use it as a context manager. Retrieval by the
    parameter tuple: :meth:`tableau`, :meth:`code`, :meth:`k_trace`, :meth:`first_passage`,
    :meth:`snapshot`, :meth:`observations`, :meth:`iter_snapshots`. The runner
    (:mod:`sparsegf2.study.runner`) fills it.
    """

    def __init__(self, directory: Path, conn: sqlite3.Connection, lock: Any, writable: bool):
        self._dir = directory
        self._conn = conn
        self._lock = lock
        self._writable = writable
        self._in_transaction = False
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        self._meta = meta
        self.experiment_id: str = meta["experiment_id"]

    # -- opening and closing -------------------------------------------------------------------

    @staticmethod
    def _connect(db: Path, *, writable: bool) -> sqlite3.Connection:
        conn = sqlite3.connect(db, isolation_level=None, timeout=60.0)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 60000")
        if writable:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = FULL")
        else:
            conn.execute("PRAGMA query_only = ON")
        return conn

    @staticmethod
    def _take_lock(directory: Path) -> Any:
        from sparsegf2 import api

        return api._lock_output_dir(directory)

    @classmethod
    def create(cls, path: str | os.PathLike[str], manifest: ExperimentManifest) -> TableauStore:
        """A new store in the directory ``path`` (created; it must not exist or be empty),
        holding ``manifest`` as revision 1, opened for writing."""
        if not isinstance(manifest, ExperimentManifest):
            raise TypeError(
                f"manifest must be an ExperimentManifest, got {type(manifest).__name__}"
            )
        directory = Path(path)
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError(
                f"{str(directory)!r} exists and is not an empty directory: a store is created in "
                "a new directory (open an existing store with TableauStore.open)"
            )
        directory.mkdir(parents=True, exist_ok=True)
        lock = cls._take_lock(directory)
        try:
            db = directory / STORE_FILE
            conn = cls._connect(db, writable=True)
            try:
                conn.execute("BEGIN IMMEDIATE")
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        conn.execute(statement)
                for trigger in _TRIGGERS:
                    conn.execute(trigger)
                meta = {
                    "store_schema": STORE_SCHEMA_VERSION,
                    "experiment_id": manifest.experiment_id,
                    "created_utc": _now(),
                    "versions": canonical_json(store_versions()),
                    "graph_measures_version": GRAPH_MEASURES_VERSION,
                    "qubit_order": QUBIT_ORDER,
                    "environment": canonical_json(environment()),
                }
                conn.executemany(
                    "INSERT INTO meta (key, value) VALUES (?, ?)", sorted(meta.items())
                )
                conn.execute(
                    "INSERT INTO manifests (revision, digest, json, created_utc, sparsegf2_version, "
                    "git_commit) VALUES (1, ?, ?, ?, ?, ?)",
                    (manifest.digest(), manifest.to_json(), _now(), __version__, _git_commit()),
                )
                conn.execute("COMMIT")
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
                conn.close()
                for suffix in ("", "-wal", "-shm", "-journal"):  # nothing was committed
                    with contextlib.suppress(OSError):
                        (directory / (STORE_FILE + suffix)).unlink()
                raise
        except BaseException:
            lock.release()
            raise
        return cls(directory, conn, lock, True)

    @classmethod
    def open(cls, path: str | os.PathLike[str], *, write: bool = False) -> TableauStore:
        """Open the store in the directory ``path``; ``write=True`` takes its writer lock.

        A second writer, in this or another process, is refused with a message naming the
        process that holds the lock. Writing needs every algorithm version the store was created
        under (:func:`store_versions`: the randomness contract, generator, Clifford numbering,
        dynamics, row-space format, seed tags and layouts, token format, qubit order); reading
        needs only the schema (a replay also needs the contract, :meth:`check_contract`).
        """
        directory = Path(path)
        db = directory / STORE_FILE
        if not db.is_file():
            raise ValueError(f"{str(directory)!r} is not a study store (no {STORE_FILE} in it)")
        lock = cls._take_lock(directory) if write else None
        try:
            conn = cls._connect(db, writable=write)
            try:
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
                if "meta" not in tables:
                    raise ValueError(
                        f"{str(directory)!r} holds a {STORE_FILE} without the store's tables: its "
                        "creation never completed (the process ended inside TableauStore.create). "
                        "Nothing was stored in it: remove the directory and create the store again"
                    )
                store = cls(directory, conn, lock, write)
                store._check_versions(write)
            except BaseException:
                conn.close()
                raise
        except BaseException:
            if lock is not None:
                lock.release()
            raise
        return store

    def _check_versions(self, write: bool) -> None:
        schema = self._meta.get("store_schema")
        if schema != STORE_SCHEMA_VERSION:
            raise ValueError(
                f"this store has schema {schema!r}; this version of sparsegf2 reads "
                f"{STORE_SCHEMA_VERSION!r}"
            )
        if not write:
            return
        stored = json.loads(self._meta["versions"])
        now = store_versions()
        for name in sorted(set(now) | set(stored)):
            if name == "sparsegf2":  # the package version may move; the algorithms may not
                continue
            if stored.get(name) != now.get(name):
                raise ValueError(
                    f"this store was made with {name} = {stored.get(name)!r} and this code has "
                    f"{now.get(name)!r}: its trajectories cannot be continued or extended here "
                    "(open it for reading, or use the code it was made with)"
                )

    def check_contract(self) -> None:
        """Raise ``ValueError`` unless this code's ``mipt.RANDOMNESS_CONTRACT`` is the one the
        store was written under: under another contract the same seeds give other circuits, so
        nothing could be replayed, verified or extended (G-ID-1)."""
        stored = json.loads(self._meta["versions"]).get("randomness_contract")
        if stored != mipt.RANDOMNESS_CONTRACT:
            raise ValueError(
                f"this store was written under randomness contract {stored!r}; this code uses "
                f"{mipt.RANDOMNESS_CONTRACT}, under which its seeds give other circuits: its "
                "trajectories cannot be replayed or extended with this code"
            )

    def close(self) -> None:
        """Close the connection and release the writer lock (if held)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    def __enter__(self) -> TableauStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        mode = "writer" if self._writable else "reader"
        return f"TableauStore({str(self._dir)!r}, experiment={self.experiment_id!r}, {mode})"

    @property
    def path(self) -> Path:
        """The store directory."""
        return self._dir

    @property
    def writable(self) -> bool:
        return self._writable

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise ValueError("this store is closed")
        return self._conn

    def _require_writer(self) -> None:
        if not self._writable:
            raise ValueError(
                "this store was opened for reading: open it with TableauStore.open(path, "
                "write=True) to write"
            )

    # -- transactions --------------------------------------------------------------------------

    @contextlib.contextmanager
    def transaction(self) -> Iterator[_Transaction]:
        """One write transaction: ``BEGIN IMMEDIATE``, the writes, ``COMMIT``; any exception
        rolls everything back. Nested use is refused."""
        self._require_writer()
        if self._in_transaction:
            raise RuntimeError("a transaction is already open on this store")
        conn = self._db()
        conn.execute("BEGIN IMMEDIATE")
        self._in_transaction = True
        try:
            yield _Transaction(self, conn)
            conn.execute("COMMIT")
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise
        finally:
            self._in_transaction = False

    @contextlib.contextmanager
    def reading(self) -> Iterator[sqlite3.Connection]:
        """One read transaction: every statement inside it sees the same committed state.

        ``BEGIN`` (deferred: the snapshot is taken at the first read) ... ``COMMIT``. In WAL mode
        a writer may commit meanwhile (a reader is never blocked) without changing what the
        block reads. Used by :meth:`export`, :meth:`logical_digest` and :meth:`check`, which
        read many tables; inside an open transaction of this connection that one is used.
        """
        conn = self._db()
        if conn.in_transaction:
            yield conn
            return
        conn.execute("BEGIN")
        try:
            yield conn
        finally:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("COMMIT")

    # -- manifests -----------------------------------------------------------------------------

    def revisions(self) -> list[ExperimentManifest]:
        """Every manifest revision, oldest first (each re-read and its digest checked)."""
        rows = self._db().execute("SELECT revision, digest, json FROM manifests ORDER BY revision")
        out = []
        for revision, digest, text in rows.fetchall():
            manifest = ExperimentManifest.from_json(text)
            if manifest.digest() != digest:
                raise ValueError(
                    f"manifest revision {revision}: its JSON does not match its digest"
                )
            out.append(manifest)
        return out

    @property
    def manifest(self) -> ExperimentManifest:
        """The latest manifest revision."""
        return self.revisions()[-1]

    @property
    def revision(self) -> int:
        """The number of the latest revision (1 for a new store)."""
        return int(self._db().execute("SELECT max(revision) FROM manifests").fetchone()[0])

    def add_revision(self, new: ExperimentManifest) -> int:
        """Add ``new`` as the next revision if it extends the latest one
        (:func:`~sparsegf2.study.manifest.check_extension`); return its number.

        Adding the latest manifest again changes nothing (its number is returned); a manifest
        that changes what is recorded is refused with the first forbidden change named.
        """
        if not isinstance(new, ExperimentManifest):
            raise TypeError(f"expected an ExperimentManifest, got {type(new).__name__}")
        self._require_writer()
        latest = self.manifest
        if new.digest() == latest.digest():
            return self.revision
        earlier = (
            self._db()
            .execute("SELECT revision FROM manifests WHERE digest = ?", (new.digest(),))
            .fetchone()
        )
        if earlier is not None:
            raise ValueError(
                f"the new manifest is revision {int(earlier[0])} again (digest "
                f"{new.digest()[:16]}...): a revision differs from every earlier one, since each "
                "manifest is stored once under its digest. The settings it returns to are in "
                "effect again through a revision that also records the return, for example a "
                "changed storage.graphs_per_unit or resources field (they change no stored value)"
            )
        message = check_extension(latest, new)
        if message is not None:
            raise ValueError(
                f"the new manifest does not extend revision {self.revision}: {message}"
            )
        with self.transaction() as tx:
            number = self.revision + 1
            tx._conn.execute(
                "INSERT INTO manifests (revision, digest, json, created_utc, sparsegf2_version, "
                "git_commit) VALUES (?, ?, ?, ?, ?, ?)",
                (number, new.digest(), new.to_json(), _now(), __version__, _git_commit()),
            )
        return number

    def meta(self) -> dict[str, Any]:
        """The store's metadata: schema, experiment, creation time, the versions it was made with."""
        out: dict[str, Any] = dict(self._meta)
        out["versions"] = json.loads(out["versions"])
        return out

    # -- keys ----------------------------------------------------------------------------------

    def key(
        self,
        key: TrajectoryKey | str | None = None,
        *,
        n: int | None = None,
        beta: Any = None,
        p: Any = None,
        g: int | None = None,
        c: int | None = None,
        r: int = 0,
        protocol: str = "F",
    ) -> TrajectoryKey:
        """The trajectory key of this store's experiment from a key, its string, or its parts.

        ``beta`` and ``p`` may be given in any form a token accepts (``0.1``, ``"0.1"``,
        ``"1/10"``): they resolve to the same key.
        """
        if key is not None:
            k = TrajectoryKey.parse(key) if isinstance(key, str) else key
            if not isinstance(k, TrajectoryKey):
                raise TypeError(
                    f"key must be a TrajectoryKey or its string, got {type(key).__name__}"
                )
            if k.experiment_id != self.experiment_id:
                raise ValueError(
                    f"key {k} belongs to experiment {k.experiment_id!r}, this store holds "
                    f"{self.experiment_id!r}"
                )
            return k
        missing = [
            name
            for name, v in (("n", n), ("beta", beta), ("p", p), ("g", g), ("c", c))
            if v is None
        ]
        if missing:
            raise ValueError(f"give a key, or n, beta, p, g and c (missing {missing})")
        assert n is not None and g is not None and c is not None
        return TrajectoryKey(self.experiment_id, n, beta, p, g, c, r, protocol)

    @staticmethod
    def _key_params(key: TrajectoryKey) -> tuple[Any, ...]:
        return (key.protocol, key.n, key.beta.token, key.p.token, key.g, key.c, key.r)

    @staticmethod
    def _row_key(experiment_id: str, row: Sequence[Any]) -> TrajectoryKey:
        protocol, n, beta, p, g, c, r = row
        return TrajectoryKey(experiment_id, int(n), beta, p, int(g), int(c), int(r), protocol)

    def _tid(self, key: TrajectoryKey) -> int | None:
        row = (
            self._db()
            .execute(
                "SELECT tid FROM trajectories WHERE protocol = ? AND n = ? AND beta = ? AND p = ? "
                "AND g = ? AND c = ? AND r = ?",
                self._key_params(key),
            )
            .fetchone()
        )
        return None if row is None else int(row[0])

    def _require_tid(self, key: TrajectoryKey) -> int:
        tid = self._tid(key)
        if tid is None:
            raise KeyError(f"no trajectory {key} in this store")
        return tid

    # -- graphs --------------------------------------------------------------------------------

    def graph(self, n: int, beta: Any, g: int) -> GraphRecord:
        """The graph record ``(n, beta, g)``, its digest and invariants checked."""
        b = Token.parse(beta, what="beta")
        row = (
            self._db()
            .execute(
                "SELECT seed, generator, edge_order, num_edges, edges, digest FROM graphs "
                "WHERE n = ? AND beta = ? AND g = ?",
                (int(n), b.token, int(g)),
            )
            .fetchone()
        )
        if row is None:
            raise KeyError(f"no graph (n = {n}, beta = {b}, g = {g}) in this store")
        seed, generator, order, count, blob, digest = row
        edges = _int32_array(blob, f"graph (n = {n}, beta = {b}, g = {g})", 2)
        edges.flags.writeable = False
        record = GraphRecord(
            n=int(n),
            beta=b,
            g=int(g),
            seed=tuple(json.loads(seed)),
            generator=generator,
            edge_order=order,
            edges=edges,
            edge_digest=digest,
        )
        if record.num_edges != count:
            raise ValueError(f"graph (n = {n}, beta = {b}, g = {g}): the edge count does not match")
        try:
            record.verify()
        except ValueError as exc:
            what = f"graph (n = {n}, beta = {b}, g = {g})"
            text = str(exc)
            raise ValueError(text if text.startswith(what) else f"{what}: {text}") from None
        return record

    def graph_measures(self, n: int, beta: Any, g: int) -> dict[str, Any]:
        """The stored measurements of graph ``(n, beta, g)`` (:func:`graph_measures`) and their version."""
        b = Token.parse(beta, what="beta")
        row = (
            self._db()
            .execute(
                "SELECT measures, measures_version FROM graphs WHERE n = ? AND beta = ? AND g = ?",
                (int(n), b.token, int(g)),
            )
            .fetchone()
        )
        if row is None:
            raise KeyError(f"no graph (n = {n}, beta = {b}, g = {g}) in this store")
        out = json.loads(row[0])
        out["version"] = row[1]
        if out.get("mean_distance") is None:
            out["mean_distance"] = math.nan
        return dict(out)

    def has_graph(self, n: int, beta: Any, g: int) -> bool:
        b = Token.parse(beta, what="beta")
        return (
            self._db()
            .execute(
                "SELECT 1 FROM graphs WHERE n = ? AND beta = ? AND g = ?", (int(n), b.token, int(g))
            )
            .fetchone()
            is not None
        )

    # -- trajectories and traces ---------------------------------------------------------------

    def trajectory(self, key: TrajectoryKey | str | None = None, **parts: Any) -> TrajectoryRecord:
        """The identity and provenance of a stored trajectory (``KeyError`` if absent)."""
        k = self.key(key, **parts)
        row = (
            self._db()
            .execute(
                "SELECT tid, u, p_float_hex, run_seed, graph_digest, dynamics_version, unit_id, revision "
                "FROM trajectories WHERE protocol = ? AND n = ? AND beta = ? AND p = ? AND g = ? AND "
                "c = ? AND r = ?",
                self._key_params(k),
            )
            .fetchone()
        )
        if row is None:
            raise KeyError(f"no trajectory {k} in this store")
        tid, u, hex_, seed, digest, dynamics, unit_id, revision = row
        return TrajectoryRecord(
            key=k,
            tid=int(tid),
            u=int(u),
            p_float_hex=hex_,
            run_seed=tuple(json.loads(seed)),
            graph_digest=digest,
            dynamics_version=dynamics,
            unit_id=int(unit_id),
            revision=int(revision),
        )

    def trajectory_keys(
        self,
        *,
        protocol: str | None = None,
        n: int | None = None,
        beta: Any = None,
        p: Any = None,
    ) -> list[TrajectoryKey]:
        """The keys of the stored trajectories matching the filters, in natural-key order."""
        where, args = self._filters(protocol=protocol, n=n, beta=beta, p=p)
        rows = self._db().execute(
            f"SELECT {_KEY_COLUMNS} FROM trajectories tr WHERE {where} "
            "ORDER BY tr.protocol, tr.n, tr.beta, tr.p, tr.g, tr.c, tr.r",
            args,
        )
        return [self._row_key(self.experiment_id, row) for row in rows.fetchall()]

    @staticmethod
    def _filters(
        *,
        protocol: str | None = None,
        n: int | None = None,
        beta: Any = None,
        p: Any = None,
        g: int | None = None,
        c: int | None = None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = ["1"]
        args: list[Any] = []
        if protocol is not None:
            clauses.append("tr.protocol = ?")
            args.append(check_protocol(protocol))
        if n is not None:
            clauses.append("tr.n = ?")
            args.append(int(n))
        if beta is not None:
            clauses.append("tr.beta = ?")
            args.append(Token.parse(beta, what="beta").token)
        if p is not None:
            clauses.append("tr.p = ?")
            args.append(Token.parse(p, what="p").token)
        if g is not None:
            clauses.append("tr.g = ?")
            args.append(int(g))
        if c is not None:
            clauses.append("tr.c = ?")
            args.append(int(c))
        return " AND ".join(clauses), args

    def trace(
        self, key: TrajectoryKey | str | None = None, *, layers: int | None = None, **parts: Any
    ) -> TraceRecord:
        """The trace of a trajectory at horizon ``layers`` (default: its largest stored horizon)."""
        k = self.key(key, **parts)
        tid = self._require_tid(k)
        rows = (
            self._db()
            .execute(
                "SELECT layers, final_layer, k_points, tau, tau0, tau1, censored0, status, "
                "num_measurements, unit_id, revision FROM traces WHERE tid = ? ORDER BY layers",
                (tid,),
            )
            .fetchall()
        )
        if not rows:
            raise KeyError(f"trajectory {k} has no trace")
        if layers is None:
            row = rows[-1]
        else:
            found = [r for r in rows if r[0] == int(layers)]
            if not found:
                raise KeyError(
                    f"trajectory {k} has no trace at horizon {layers}; it has {[r[0] for r in rows]}"
                )
            row = found[0]
        return self._trace_record(k, row)

    @staticmethod
    def _trace_record(key: TrajectoryKey, row: Sequence[Any]) -> TraceRecord:
        layers, final, kp, tau, tau0, tau1, cens, status, meas, unit_id, revision = row
        what = f"trace of {key} at horizon {layers}"
        k = None if kp is None else ob.k_from_change_points(_int32_array(kp, what, 2))
        if k is not None and k.size != layers + 1:
            raise ValueError(f"{what}: the stored change points do not end at the horizon")
        return TraceRecord(
            key=key,
            layers=int(layers),
            final_layer=int(final),
            k=k,
            tau=None if tau is None else _int32_array(tau, what),
            tau0=None if tau0 is None else int(tau0),
            tau1=None if tau1 is None else int(tau1),
            censored0=None if cens is None else bool(cens),
            status=status,
            num_measurements=int(meas),
            unit_id=int(unit_id),
            revision=int(revision),
        )

    def k_trace(
        self, key: TrajectoryKey | str | None = None, *, layers: int | None = None, **parts: Any
    ) -> np.ndarray:
        """``k_0, ..., k_T`` of a protocol-F trajectory, decoded losslessly from its change points."""
        rec = self.trace(key, layers=layers, **parts)
        if rec.k is None:
            raise ValueError(f"trajectory {rec.key} (protocol {rec.key.protocol}) has no k trace")
        return rec.k

    def first_passage(
        self,
        key: TrajectoryKey | str | None = None,
        *,
        layers: int | None = None,
        checkpoints: Any = "current",
        **parts: Any,
    ) -> ob.FirstPassage:
        """The exact first passages ``tau_ell``, ``ell = 0..n``, with censoring (v1.4 §5.2).

        Rebuilt from the stored trace by :func:`~sparsegf2.study.observables.first_passage` and
        checked against the stored ``tau``. The checkpoint bounds ``tau_chk`` are taken over
        ``checkpoints``: ``"current"`` (default) the latest revision's own schedule $S_r$ of the
        condition up to the horizon, the domain of the stored ``first_passage`` value (G-OBS-2,
        Proposition 15 of doc 05: the same in a revised store and in one made directly under the
        latest manifest); ``"stored"`` every stored snapshot depth up to the horizon (the union
        $E_r$ of every revision's schedule, which depends on the revision history); ``None``
        leaves them out; a list uses those depths.
        """
        rec = self.trace(key, layers=layers, **parts)
        if rec.k is None:
            raise ValueError(f"trajectory {rec.key} (protocol {rec.key.protocol}) has no k trace")
        if isinstance(checkpoints, str) and checkpoints == "stored":
            cps = [t for t in self.depths(rec.key) if t <= rec.layers] or None
        elif isinstance(checkpoints, str) and checkpoints == "current":
            k = rec.key
            domain = self.manifest.checkpoints(k.n, k.beta, k.protocol)
            cps = [int(t) for t in domain if int(t) <= rec.layers] or None
        else:
            cps = checkpoints
        fp = ob.first_passage(rec.k, cps, num_system=rec.key.n)
        if rec.tau is not None and not np.array_equal(fp.tau, rec.tau):
            raise ValueError(f"trajectory {rec.key}: the stored tau does not match its k trace")
        return fp

    # -- snapshots -----------------------------------------------------------------------------

    def depths(self, key: TrajectoryKey | str | None = None, **parts: Any) -> tuple[int, ...]:
        """The depths of the trajectory's tableau dictionary (stored or absent), increasing."""
        k = self.key(key, **parts)
        tid = self._require_tid(k)
        rows = self._db().execute("SELECT t FROM snapshots WHERE tid = ? ORDER BY t", (tid,))
        return tuple(int(r[0]) for r in rows.fetchall())

    def snapshot(
        self, key: TrajectoryKey | str | None = None, t: int | None = None, **parts: Any
    ) -> SnapshotRecord:
        """The entry ``t`` of a trajectory's tableau dictionary with its explicit status.

        ``KeyError`` if the trajectory, or the depth, is not in the store (an unsaved depth
        never returns a neighboring one).
        """
        if t is None:
            raise ValueError("give the depth t")
        k = self.key(key, **parts)
        tid = self._require_tid(k)
        row = (
            self._db()
            .execute(
                f"SELECT {_SNAPSHOT_COLUMNS} FROM snapshots s WHERE s.tid = ? AND s.t = ?",
                (tid, int(t)),
            )
            .fetchone()
        )
        if row is None:
            stored = self.depths(k)
            raise KeyError(
                f"trajectory {k} has no snapshot at t = {t}; its dictionary holds the depths "
                f"{list(stored)} (runner.replay gives the state at any depth)"
            )
        return self._snapshot_record(k, row)

    @staticmethod
    def _snapshot_record(key: TrajectoryKey, row: Sequence[Any]) -> SnapshotRecord:
        t, status, nq, rank, kk, jd, pd, jb, pb = row
        return SnapshotRecord(
            key=key,
            t=int(t),
            status=status,
            num_qubits=int(nq),
            rank=None if rank is None else int(rank),
            k=None if kk is None else int(kk),
            joint_digest=jd,
            phys_digest=pd,
            joint_blob=jb,
            phys_blob=pb,
        )

    def tableau(
        self, key: TrajectoryKey | str | None = None, t: int | None = None, **parts: Any
    ) -> PhaseFreeTableau:
        """The stored joint phase-free tableau of a trajectory at depth ``t``.

        ``store.tableau(n=16, beta="0.1", p=0.15, g=3, c=1, t=8)`` (or a key and ``t``):
        decoded, checked against its digest, in canonical form; for protocol P
        (``protocol="P"``) the pure ``n``-qubit state of the system. Nothing is simulated.
        """
        return self.snapshot(key, t, **parts).tableau()

    def code(
        self, key: TrajectoryKey | str | None = None, t: int | None = None, **parts: Any
    ) -> StabilizerCode:
        """The stored physical code (v1.4 §5, §8.1) of a protocol-F trajectory at depth ``t``."""
        return self.snapshot(key, t, **parts).code()

    def iter_snapshots(
        self,
        *,
        protocol: str | None = None,
        n: int | None = None,
        beta: Any = None,
        p: Any = None,
        g: int | None = None,
        c: int | None = None,
        t: int | None = None,
        status: str | Sequence[str] | None = (SNAPSHOT_STORED, SNAPSHOT_PHYS_ONLY),
        batch_size: int = 256,
    ) -> Iterator[SnapshotRecord]:
        """Every snapshot matching the filters, in natural order, ``batch_size`` rows at a time.

        A keyset cursor: each batch is one query that resumes after the last ``(tid, t)`` read,
        walking the trajectories in row-id order from the last one (the integer primary key)
        and each trajectory's snapshots through the unique index ``(tid, t)``, so no batch
        sorts or revisits earlier rows (the plan has no ``TEMP B-TREE``; the tests check the
        store's own query) and a whole iteration costs time linear in the store. Memory holds
        at most ``batch_size`` compressed snapshots, and nothing is decoded until
        :meth:`SnapshotRecord.tableau` or ``code`` is called. ``status`` is a status of
        :data:`SNAPSHOT_STATUSES` or a list of them (default: the snapshots that hold a row
        space, ``stored`` and ``phys_only``); ``None`` includes absent snapshots.
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        where, args = self._filters(protocol=protocol, n=n, beta=beta, p=p, g=g, c=c)
        extra: list[str] = []
        extra_args: list[Any] = []
        if t is not None:
            extra.append("s.t = ?")
            extra_args.append(int(t))
        if status is not None:
            wanted = [status] if isinstance(status, str) else list(status)
            if not wanted or any(w not in SNAPSHOT_STATUSES for w in wanted):
                raise ValueError(
                    f"status must be one of {list(SNAPSHOT_STATUSES)}, a list of them, or None"
                )
            extra.append(f"s.status IN ({', '.join('?' * len(wanted))})")
            extra_args.extend(wanted)
        more = "".join(f" AND {e}" for e in extra)
        sql = (
            f"SELECT s.tid, {_KEY_COLUMNS}, {_SNAPSHOT_COLUMNS} FROM trajectories tr "
            "CROSS JOIN snapshots s ON s.tid = tr.tid "
            f"WHERE tr.tid >= ? AND (tr.tid > ? OR s.t > ?) AND {where}{more} "
            "ORDER BY tr.tid, s.t LIMIT ?"
        )
        last = (-1, -1)
        while True:
            rows = (
                self._db()
                .execute(sql, (last[0], last[0], last[1], *args, *extra_args, batch_size))
                .fetchall()
            )
            if not rows:
                return
            for row in rows:
                key = self._row_key(self.experiment_id, row[1:8])
                yield self._snapshot_record(key, row[8:])
            last = (int(rows[-1][0]), int(rows[-1][8]))
            if len(rows) < batch_size:
                return

    # -- observations --------------------------------------------------------------------------

    def _current(self, protocol: str, n: int, beta: Any, revisions: Any = None) -> Any:
        """What the latest revision asks of the trajectories of ``(protocol, n, beta)``:
        :func:`sparsegf2.study.runner.current` (its requirements, derivation and the context of
        its trajectory-level values)."""
        from sparsegf2.study import runner as _runner

        chain = self.revisions() if revisions is None else revisions
        return _runner.current(chain, protocol, n, beta)

    def _currents(self) -> Any:
        """A per-call cache of :meth:`_current`, keyed by ``(protocol, n, beta token)``."""
        revisions = self.revisions()
        cache: dict[tuple[str, int, str], Any] = {}

        def get(protocol: str, n: int, beta: Any) -> Any:
            ck = (protocol, int(n), Token.parse(beta, what="beta").token)
            if ck not in cache:
                cache[ck] = self._current(protocol, n, beta, revisions)
            return cache[ck]

        return get

    def _select(
        self,
        observable: str,
        version: str | None,
        params: Mapping[str, Any] | str | None,
        *,
        prefer_current: bool = False,
    ) -> tuple[str, str]:
        """The stored ``(version, params)`` of ``observable`` that ``version`` and ``params`` name.

        The choices are the stored definitions (``observable_defs``: one row per key of a stored
        value). ``params`` is a dict (resolved with the registry's defaults) or the canonical
        text; left out, the store must hold one choice, or, with ``prefer_current``, one among
        the trajectory-level values the latest revision derives.
        """
        choices = [
            (str(v), str(p))
            for v, p in self._db().execute(
                "SELECT version, params FROM observable_defs WHERE observable = ? "
                "ORDER BY version, params",
                (observable,),
            )
        ]
        if version is not None:
            choices = [ch for ch in choices if ch[0] == version]
        if params is not None:
            text = (
                params
                if isinstance(params, str)
                else ob.params_key(observable, params, version=version)
            )
            choices = [ch for ch in choices if ch[1] == text]
        if not choices:
            raise KeyError(f"no stored values of observable {observable!r} with these settings")
        if len(choices) > 1 and prefer_current:
            latest = self.manifest
            wanted: set[tuple[str, str]] = set()
            for protocol in latest.protocol_names:
                for cond in latest.conditions():
                    cur = self._current(protocol, cond.n, cond.beta)
                    wanted |= {(v, p) for o, v, p in cur.derived if o == observable}
            if wanted:
                choices = [ch for ch in choices if ch in wanted] or choices
        if len(choices) > 1:
            raise ValueError(
                f"observable {observable!r} is stored under several versions or parameter sets; "
                f"choose one with version= and params=: {sorted(choices)}"
            )
        return choices[0]

    def observations(
        self,
        observable: str,
        *,
        version: str | None = None,
        params: Mapping[str, Any] | str | None = None,
        context: str | None = "current",
        protocol: str = "F",
        n: int | None = None,
        beta: Any = None,
        p: Any = None,
        t: int | None = None,
        field: str | None = None,
    ) -> dict[str, np.ndarray]:
        """The stored values of one observable as arrays with ``g, c, r`` columns (design §4).

        One entry per stored row, in natural-key order: ``n``, ``beta``, ``p`` (tokens),
        ``g``, ``c``, ``r``, ``t`` (``-1`` for trajectory-level values), ``value`` (float64,
        NaN where undefined), ``se``, ``method``, ``context``, ``status`` (one of
        :data:`OBSERVATION_STATUSES`: why a value is NaN, infinite or a vector), and, with
        ``field``, that entry of each row's data (a float64 array, stacked when it is a list;
        the data are read only then). ``version`` and ``params`` may be left out when the store
        holds one choice of them (``params`` is a dict, resolved with the registry's defaults,
        or the stored canonical text); for a trajectory-level observable stored under several
        parameter sets (a revision changed the peak rule) the one the latest revision derives
        is taken. ``context="current"`` keeps, for trajectory-level values, those of the latest
        revision's horizon and peak domain of each condition (:func:`runner.derivation`: the
        same rows a store made directly under the latest manifest holds); a trajectory not yet
        run or derived under it has none. ``None`` keeps every context; any other string keeps
        that one.
        """
        db = self._db()
        version_, params_ = self._select(
            observable, version, params, prefer_current=context == "current"
        )
        where, args = self._filters(protocol=protocol, n=n, beta=beta, p=p)
        blob = "o.data" if field is not None else "NULL"
        sql = (
            f"SELECT tr.tid, {_KEY_COLUMNS}, o.t, o.value, o.se, o.method, o.context, {blob}, "
            "o.status FROM observations o JOIN trajectories tr ON tr.tid = o.tid "
            f"WHERE o.observable = ? AND o.version = ? AND o.params = ? AND {where}"
        )
        sql_args: list[Any] = [observable, version_, params_, *args]
        if t is not None:
            sql += " AND o.t = ?"
            sql_args.append(int(t))
        sql += " ORDER BY tr.protocol, tr.n, tr.beta, tr.p, tr.g, tr.c, tr.r, o.t, o.context"
        rows = db.execute(sql, sql_args).fetchall()
        if context == "current":
            currents = self._currents()
            rows = [r for r in rows if r[8] != -1 or r[12] == currents(r[1], r[2], r[3]).context]
        elif context is not None:
            rows = [r for r in rows if r[12] == context]
        out: dict[str, np.ndarray] = {
            "n": np.array([r[2] for r in rows], dtype=np.int64),
            "beta": np.array([r[3] for r in rows], dtype=object),
            "p": np.array([r[4] for r in rows], dtype=object),
            "g": np.array([r[5] for r in rows], dtype=np.int64),
            "c": np.array([r[6] for r in rows], dtype=np.int64),
            "r": np.array([r[7] for r in rows], dtype=np.int64),
            "t": np.array([r[8] for r in rows], dtype=np.int64),
            "value": np.array([np.nan if r[9] is None else r[9] for r in rows], dtype=np.float64),
            "se": np.array([np.nan if r[10] is None else r[10] for r in rows], dtype=np.float64),
            "method": np.array([r[11] for r in rows], dtype=object),
            "context": np.array([r[12] for r in rows], dtype=object),
            "status": np.array([r[14] for r in rows], dtype=object),
        }
        out["version"] = np.array([version_] * len(rows), dtype=object)
        out["params"] = np.array([params_] * len(rows), dtype=object)
        if field is not None:
            values = []
            for r in rows:
                body = _data_text(r[13], f"observation {observable!r}")
                data = {} if body is None else json.loads(body)
                if field not in data:
                    raise KeyError(f"observation {observable!r} has no data field {field!r}")
                values.append(data[field])
            out[field] = _float_array(values)
        return out

    def observation(
        self,
        key: TrajectoryKey | str | None,
        t: int,
        observable: str,
        *,
        version: str | None = None,
        params: Mapping[str, Any] | str | None = None,
        context: str = "",
        **parts: Any,
    ) -> dict[str, Any]:
        """One stored value: ``{"value", "se", "status", "method", "data"}`` (``KeyError`` if absent).

        A trajectory-level value (``t = -1``) is taken under the current context of the
        trajectory's condition (the latest revision's horizon and peak domain) unless
        ``context`` names another one.
        """
        k = self.key(key, **parts)
        tid = self._require_tid(k)
        entry = ob.get(observable, version)
        text = (
            params
            if isinstance(params, str)
            else ob.params_key(observable, params, version=entry.version)
        )
        if t == -1 and context == "":
            context = self._current(k.protocol, k.n, k.beta).context
        row = (
            self._db()
            .execute(
                "SELECT value, se, method, data, context, status FROM observations WHERE tid = ? "
                "AND t = ? AND observable = ? AND version = ? AND params = ? AND context = ?",
                (tid, int(t), observable, entry.version, text, context),
            )
            .fetchone()
        )
        if row is None:
            more = (
                " under the current context (the latest revision's horizon and peak domain: run "
                "runner.run or runner.derive, or pass the context of an earlier one)"
                if t == -1
                else ""
            )
            raise KeyError(
                f"no value of {observable!r} for {k} at t = {t} with these settings{more}"
            )
        value, se, method, data, _, status = row
        body = _data_text(data, f"observation {observable!r} of {k}")
        return {
            "value": math.nan if value is None else float(value),
            "se": math.nan if se is None else float(se),
            "status": status,
            "method": method,
            "data": None if body is None else json.loads(body),
        }

    # -- planning helpers ----------------------------------------------------------------------

    def completed(
        self, protocol: str, n: int, beta: Any, p: Any, requirements: str
    ) -> set[tuple[int, int]]:
        """The ``(g, c)`` of the trajectories of a condition complete under ``requirements``."""
        rows = self._db().execute(
            "SELECT tr.g, tr.c FROM trajectories tr JOIN completions co ON co.tid = tr.tid "
            "WHERE tr.protocol = ? AND tr.n = ? AND tr.beta = ? AND tr.p = ? AND tr.r = 0 "
            "AND co.requirements = ?",
            (
                check_protocol(protocol),
                int(n),
                Token.parse(beta).token,
                Token.parse(p).token,
                requirements,
            ),
        )
        return {(int(g), int(c)) for g, c in rows.fetchall()}

    def derived_pairs(
        self, protocol: str, n: int, beta: Any, p: Any, derivation: str
    ) -> set[tuple[int, int]]:
        """The ``(g, c)`` of the trajectories of a condition derived under ``derivation`` (every
        trajectory-level value of :func:`runner.derivation` stored)."""
        rows = self._db().execute(
            "SELECT tr.g, tr.c FROM trajectories tr JOIN derivations d ON d.tid = tr.tid "
            "WHERE tr.protocol = ? AND tr.n = ? AND tr.beta = ? AND tr.p = ? AND tr.r = 0 "
            "AND d.derivation = ?",
            (
                check_protocol(protocol),
                int(n),
                Token.parse(beta).token,
                Token.parse(p).token,
                derivation,
            ),
        )
        return {(int(g), int(c)) for g, c in rows.fetchall()}

    def _has_observation(
        self, tid: int, t: int, observable: str, version: str, params: str, context: str = ""
    ) -> bool:
        return (
            self._db()
            .execute(
                "SELECT 1 FROM observations WHERE tid = ? AND t = ? AND observable = ? AND "
                "version = ? AND params = ? AND context = ?",
                (tid, int(t), observable, version, params, context),
            )
            .fetchone()
            is not None
        )

    def stored_pairs(self, protocol: str, n: int, beta: Any, p: Any) -> set[tuple[int, int]]:
        """The ``(g, c)`` of every stored trajectory of a condition (complete or not)."""
        rows = self._db().execute(
            "SELECT g, c FROM trajectories WHERE protocol = ? AND n = ? AND beta = ? AND p = ? "
            "AND r = 0",
            (check_protocol(protocol), int(n), Token.parse(beta).token, Token.parse(p).token),
        )
        return {(int(g), int(c)) for g, c in rows.fetchall()}

    def schedule(self, key: TrajectoryKey | str | None = None, **parts: Any) -> dict[str, Any]:
        """The schedule a trajectory holds under the latest revision (G-RUN-2): the current
        requirements' horizon (``layers``) and checkpoints (``checkpoints``, the depths of its
        dictionary and of its snapshot values), the peak domain (``domain``, the latest
        revision's own schedule; protocol P: the checkpoints), the requirements digest and the
        context of its trajectory-level values. ``KeyError`` when the trajectory is not
        complete under the current requirements (a rerun is pending: ``runner.run``)."""
        k = self.key(key, **parts)
        tid = self._require_tid(k)
        cur = self._current(k.protocol, k.n, k.beta)
        row = (
            self._db()
            .execute(
                "SELECT 1 FROM completions WHERE tid = ? AND requirements = ?", (tid, cur.digest)
            )
            .fetchone()
        )
        if row is None:
            raise KeyError(
                f"trajectory {k} is not complete under the latest revision's requirements (a "
                "rerun is pending: runner.run)"
            )
        req = cur.requirements
        domain = req["checkpoints"] if cur.derivation is None else cur.derivation["domain"]
        return {
            "requirements": cur.digest,
            "layers": req["layers"],
            "checkpoints": list(req["checkpoints"]),
            "domain": list(domain),
            "context": cur.context,
        }

    def signed_runs(
        self, key: TrajectoryKey | str | None = None, **parts: Any
    ) -> list[dict[str, Any]]:
        """The recorded outcome-aware replays of a trajectory (G-TAB-2)."""
        k = self.key(key, **parts)
        tid = self._require_tid(k)
        out = []
        for seed, layers, depths, outcomes, flags, digests, fmt in self._db().execute(
            "SELECT outcome_seed, layers, depths, outcomes, random, signed_digests, format FROM "
            "signed_runs WHERE tid = ? ORDER BY outcome_seed, layers, depths",
            (tid,),
        ):
            out.append(
                {
                    "outcome_seed": json.loads(seed),
                    "layers": int(layers),
                    "depths": json.loads(depths),
                    "outcomes": np.frombuffer(zlib.decompress(outcomes), dtype=np.uint8).copy(),
                    "random": np.frombuffer(zlib.decompress(flags), dtype=np.uint8).astype(bool),
                    "signed_digests": {int(t): d for t, d in json.loads(digests).items()},
                    "format": fmt,
                }
            )
        return out

    def sampler_diagnostics(self, **filters: Any) -> list[dict[str, Any]]:
        """The stored graph-side diagnostics of the correlated samplers (G-OBS-4), filtered by
        any of ``n``, ``beta``, ``g``, ``probe``, ``m``."""
        cols = [
            "n", "beta", "g", "probe", "config", "m", "coupling", "method", "chains", "samples",
            "seed", "ess", "tau_int", "acceptance", "exchange_acceptance", "r_hat",
            "r_hat_limit", "notes",
        ]  # fmt: skip
        clauses, args = ["1"], []
        for name, value in filters.items():
            if name not in ("n", "beta", "g", "probe", "m"):
                raise ValueError(f"unknown filter {name!r}")
            clauses.append(f"{name} = ?")
            args.append(Token.parse(value).token if name == "beta" else value)
        rows = self._db().execute(
            f"SELECT {', '.join(cols)} FROM sampler_diagnostics WHERE {' AND '.join(clauses)} "
            "ORDER BY n, beta, g, probe, config, m, coupling",
            args,
        )
        out = []
        for row in rows.fetchall():
            d = dict(zip(cols, row, strict=True))
            d["notes"] = json.loads(d["notes"])
            d["seed"] = json.loads(d["seed"])
            out.append(d)
        return out

    def aggregate(
        self,
        observable: str,
        *,
        t: int,
        n: int,
        beta: Any,
        p: Any,
        protocol: str = "F",
        version: str | None = None,
        params: Mapping[str, Any] | str | None = None,
        field: str | None = None,
        index: int | None = None,
        nan_policy: str = "raise",
        context: str = "current",
        partial: bool = False,
        record: bool = True,
    ) -> dict[str, Any]:
        """The nested equal-graph-weight mean of one observable at one depth, with its
        graph-cluster error (v1.4 §6.3, §6.4; ``stats.nested_mean``), recorded (G-STO-4).

        The rows are the stored values at depth ``t`` of the condition (for ``t = -1`` the
        trajectory-level values of ``context``: by default the current one, the latest
        revision's horizon and peak domain), the ``value`` column or the data ``field`` (entry
        ``index`` of it when it is a list). Every stored trajectory of the condition must have a
        row, unless ``partial`` (a rerun or a derivation pending after a revision would
        otherwise give a mean over a subset, or over two horizons). Rows whose status is not
        ``ok`` (NaN, missing, censored, ...) are refused under ``nan_policy="raise"`` and left
        out under ``"omit"`` (the estimand then changes: ``stats.NestedMean``; the record's data
        count the rows left out per status). The record is keyed by the condition, the
        observable's stored key (the ``(version, params)`` the rows were read under), the
        selection (field and index), the depth, the context of the rows (empty for snapshot
        values), the estimator version (:data:`ESTIMATOR_VERSION`), the policy and the
        ``sample_set``: the SHA-256 of the sorted ``(g, c, r)`` of the rows used, so a new sample
        (more graphs) or another horizon is a new record and no record is replaced. With
        ``record`` (a writer store) it is inserted, or compared with the stored one.
        """
        if context is None:
            raise ValueError("an aggregate reads one context: give 'current' or a context")
        version_, params_ = self._select(
            observable, version, params, prefer_current=context == "current"
        )
        rows = self.observations(
            observable, version=version_, params=params_, protocol=protocol, n=n, beta=beta,
            p=p, t=t, field=field, context=context,
        )  # fmt: skip
        if rows["g"].size == 0:
            raise KeyError(f"no stored values of {observable!r} at t = {t} for this condition")
        used = "" if int(t) >= 0 else str(rows["context"][0])
        stored = len(self.trajectory_keys(protocol=protocol, n=n, beta=beta, p=p))
        have = len(set(zip(rows["g"].tolist(), rows["c"].tolist(), rows["r"].tolist(),
                           strict=True)))  # fmt: skip
        if have < stored and not partial:
            raise ValueError(
                f"{have} of the {stored} stored trajectories of this condition have a value of "
                f"{observable!r} at t = {t}"
                + (" under the current context" if t == -1 and context == "current" else "")
                + ": a rerun or a derivation is pending (runner.run), or the observable was "
                "computed for part of them; partial=True aggregates the ones that have it"
            )
        values = rows[field] if field is not None else rows["value"]
        if values.ndim > 1:
            if index is None:
                raise ValueError(f"data field {field!r} is a list: give index")
            values = values[:, int(index)]
        bad = ~np.isfinite(values)
        if field is None:
            bad |= rows["status"] != "ok"
        if bad.any() and nan_policy == "raise":
            first = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"{int(bad.sum())} rows of {observable!r} at t = {t} are not finite values (the "
                f"first has status {rows['status'][first]!r}): choose nan_policy='omit' (it changes "
                "the estimand) or an estimator for censored or missing values"
            )
        clean = np.where(bad, np.nan, values)
        est = stats.nested_mean(rows["g"], rows["c"], rows["r"], clean, nan_policy=nan_policy)  # type: ignore[arg-type]
        keys = sorted(zip(rows["g"].tolist(), rows["c"].tolist(), rows["r"].tolist(), strict=True))
        sample_set = sha256_hex(canonical_json(keys).encode("utf-8"))
        omitted: dict[str, int] = {}
        for status, flag in zip(rows["status"].tolist(), bad.tolist(), strict=True):
            if flag:
                name = "nonfinite" if status == "ok" else str(status)
                omitted[name] = omitted.get(name, 0) + 1
        selection = canonical_json({"field": field, "index": index})
        out = {
            "protocol": check_protocol(protocol),
            "n": int(n),
            "beta": Token.parse(beta).token,
            "p": Token.parse(p).token,
            "observable": observable,
            "version": version_,
            "params": params_,
            "selection": selection,
            "t": int(t),
            "context": used,
            "estimator": ESTIMATOR_VERSION,
            "policy": nan_policy,
            "sample_set": sample_set,
            "mean": float(est.mean),
            "se": float(est.se),
            "graphs": int(est.num_graphs),
            "contributing_graphs": int(est.contributing_graphs),
            "trajectories": int(est.num_trajectories),
            "contributing_trajectories": int(est.contributing_trajectories),
            "omitted": dict(sorted(omitted.items())),
        }
        if record:
            self._require_writer()
            with self.transaction() as tx:
                tx.put_aggregate(
                    out,
                    {
                        "graph_means": est.graph_means,
                        "graphs": est.graphs,
                        "omitted": out["omitted"],
                    },
                )
        return out

    def aggregates(self, observable: str | None = None) -> list[dict[str, Any]]:
        """The recorded aggregates (of one observable, or all)."""
        cols = [
            "protocol", "n", "beta", "p", "observable", "version", "params", "selection", "t",
            "context", "estimator", "policy", "sample_set", "mean", "se", "graphs",
            "contributing_graphs", "trajectories", "contributing_trajectories",
        ]  # fmt: skip
        sql = f"SELECT {', '.join(cols)} FROM aggregates"
        args: list[Any] = []
        if observable is not None:
            sql += " WHERE observable = ?"
            args.append(observable)
        rows = self._db().execute(sql + f" ORDER BY {', '.join(cols[:13])}", args).fetchall()
        return [
            {**dict(zip(cols, row, strict=True)), "mean": _nan(row[13]), "se": _nan(row[14])}
            for row in rows
        ]

    def units(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """The work units recorded so far (done units; and failed ones, which wrote no data)."""
        sql = (
            "SELECT unit_id, protocol, n, beta, p, graphs, circuits, layers, requirements, status, "
            "revision, started_utc, finished_utc, elapsed_seconds, threads, trajectories, error, "
            "sparsegf2_version, git_commit FROM units"
        )
        args: list[Any] = []
        if status is not None:
            sql += " WHERE status = ?"
            args.append(status)
        cols = [
            "unit_id",
            "protocol",
            "n",
            "beta",
            "p",
            "graphs",
            "circuits",
            "layers",
            "requirements",
            "status",
            "revision",
            "started_utc",
            "finished_utc",
            "elapsed_seconds",
            "threads",
            "trajectories",
            "error",
            "sparsegf2_version",
            "git_commit",
        ]
        out = []
        for row in self._db().execute(sql + " ORDER BY unit_id", args).fetchall():
            d = dict(zip(cols, row, strict=True))
            d["graphs"] = json.loads(d["graphs"])
            d["circuits"] = json.loads(d["circuits"])
            out.append(d)
        return out

    def requirements(self, digest: str) -> dict[str, Any]:
        """The requirements (what a complete trajectory records) stored under ``digest``."""
        row = (
            self._db()
            .execute("SELECT json FROM requirements WHERE digest = ?", (digest,))
            .fetchone()
        )
        if row is None:
            raise KeyError(f"no requirements {digest} in this store")
        return dict(json.loads(row[0]))

    def counts(self) -> dict[str, int]:
        """Row counts of the main tables."""
        out = {}
        for table in (
            "manifests",
            "graphs",
            "units",
            "trajectories",
            "traces",
            "completions",
            "derivations",
            "snapshots",
            "observations",
            "observable_defs",
        ):
            out[table] = int(self._db().execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        return out

    # -- integrity -----------------------------------------------------------------------------

    def check(self, *, blobs: bool = True, batch_size: int = 256) -> list[str]:
        """Check the whole store; return the problems found (an empty list when it is sound).

        SQLite's ``integrity_check`` and ``foreign_key_check``; every manifest's digest; every
        graph's digest and invariants; every trajectory's graph digest against its graph record;
        with ``blobs``, every trace (read one row at a time) and every stored row space (in
        batches of ``batch_size``) decoded and checked against its digest, and every snapshot's
        rank and ``k``. One read snapshot (:meth:`reading`): a writer committing meanwhile does
        not change what is checked. The stored values are not recomputed here
        (:func:`runner.verify` runs trajectories again and compares every record).
        """
        with self.reading():
            return self._check(blobs=blobs, batch_size=batch_size)

    def _check(self, *, blobs: bool, batch_size: int) -> list[str]:
        db = self._db()
        problems: list[str] = []
        result = db.execute("PRAGMA integrity_check").fetchall()
        if result != [("ok",)]:
            problems.append(f"sqlite integrity_check: {result[:5]}")
        orphans = db.execute("PRAGMA foreign_key_check").fetchall()
        if orphans:
            problems.append(f"foreign_key_check: {orphans[:5]}")
        try:
            self.revisions()
        except ValueError as exc:
            problems.append(str(exc))
        for n, beta, g in db.execute(
            "SELECT n, beta, g FROM graphs ORDER BY n, beta, g"
        ).fetchall():
            try:
                self.graph(n, beta, g)
            except ValueError as exc:
                problems.append(str(exc))
        bad = db.execute(
            "SELECT tr.protocol, tr.n, tr.beta, tr.p, tr.g, tr.c, tr.r FROM trajectories tr "
            "JOIN graphs gr ON gr.n = tr.n AND gr.beta = tr.beta AND gr.g = tr.g "
            "WHERE gr.digest != tr.graph_digest"
        ).fetchall()
        for row in bad:
            problems.append(
                f"trajectory {self._row_key(self.experiment_id, row)}: graph digest differs"
            )
        if not blobs:
            return problems
        for row in db.execute(
            f"SELECT {_KEY_COLUMNS}, t.layers, t.final_layer, t.k_points, t.tau, t.tau0, t.tau1, "
            "t.censored0, t.status, t.num_measurements, t.unit_id, t.revision FROM traces t "
            "JOIN trajectories tr ON tr.tid = t.tid"
        ):
            key = self._row_key(self.experiment_id, row[:7])
            try:
                rec = self._trace_record(key, row[7:])
                if rec.k is not None:
                    fp = ob.first_passage(rec.k, num_system=key.n)
                    if rec.tau is None or not np.array_equal(fp.tau, rec.tau):
                        problems.append(f"trace of {key}: tau does not match the k trace")
            except ValueError as exc:
                problems.append(str(exc))
        for snap in self.iter_snapshots(batch_size=batch_size):
            try:
                if snap.joint_blob is not None:
                    joint = snap.tableau()
                    if not joint.is_pure() or joint.num_qubits != snap.num_qubits:
                        problems.append(f"{snap._where()}: the joint state is not pure")
                if snap.key.protocol == "F":
                    phys = snap.physical_tableau()
                    if snap.k is None or phys.rank != snap.key.n - snap.k:
                        problems.append(f"{snap._where()}: rank(phys) != n - k")
            except ValueError as exc:
                problems.append(str(exc))
        return problems

    # -- the logical digest --------------------------------------------------------------------

    def logical_digest(self, *, history: bool = True, manifests: bool = True) -> str:
        """The SHA-256 of the store's logical content: equal content, equal digest.

        Every record is written as one line, a tag and the canonical JSON of its logical
        fields, in natural-key order, and the lines are hashed (one read snapshot,
        :meth:`reading`): the manifest digests (``M``), the graph records (``G``: identity,
        seed, generator, edge order and count, edge digest and the measurements' version; the
        measurements themselves are a function of the edges, and ``lambda2`` is a LAPACK
        eigenvalue whose last bits may differ between machines, so they are left out), the
        trajectories (``T``: key, ``u``, rate hex, run seed, graph digest, dynamics), the traces
        (``R``: the decoded ``k`` and ``tau``, final layer, status, measurement count), the
        completions (``C``) and derivations (``Z``), the snapshots (``S``: status, ``N``,
        rank, ``k`` and both row-space digests), the observations (``O``: value and error as
        ``float.hex``, method, decoded data and status), the definitions of the observables
        (``D``) and the recorded outcome-aware replays (``X``). Row ids, unit numbers, revisions
        of creation, timestamps, threads, elapsed times, code versions and the compression of
        blobs are left out: they describe how the content was produced, not the content, so
        stores made with other unit sizes, thread counts, unit orders or after an interruption
        have the same digest. Aggregate records and sampler diagnostics (summaries computed on
        request) are not part of it.

        ``history=False`` digests the **current state**: the records a store made directly
        under the latest revision by :func:`runner.run` holds. For each trajectory its trace at
        the latest horizon, its snapshots and snapshot-level values at the depths of the latest
        revision's own schedule $S_r$, and its trajectory-level values of the current context
        and parameters (:func:`runner.derivation`), the definitions those values use and the
        latest manifest digest; no completions, derivations or replays. So a store extended to
        the latest revision and run to the end has the digest of a store made directly under it
        (doc 05 Proposition 6(b)), whatever the earlier schedules, horizons and peak rules.
        With ``history=True`` (default) everything is included. ``manifests=False`` leaves the
        manifest digests out: they also change with settings that never change a recorded value
        (``storage``, ``resources``), so stores made with other unit sizes or compression levels
        compare equal without them.
        """
        h = hashlib.sha256()

        def put(tag: str, row: Any) -> None:
            h.update(tag.encode("ascii"))
            h.update(canonical_json(_plain(row)).encode("utf-8"))
            h.update(b"\n")

        order = "tr.protocol, tr.n, tr.beta, tr.p, tr.g, tr.c, tr.r"
        with self.reading() as db:
            # the current state needs the latest revision; the whole history reads no manifest
            currents = None if history else self._currents()
            latest = None if history else self.manifest
            schedules: dict[tuple[str, int, str], set[int]] = {}

            def schedule(protocol: str, n: int, beta: str) -> set[int]:
                ck = (protocol, int(n), beta)
                if ck not in schedules:
                    assert latest is not None
                    schedules[ck] = {int(t) for t in latest.checkpoints(n, beta, protocol)}
                return schedules[ck]

            revisions = db.execute(
                "SELECT revision, digest FROM manifests ORDER BY revision"
            ).fetchall()
            for revision, digest in (revisions if history else revisions[-1:]) if manifests else []:
                put("M", [revision if history else 0, digest])
            for row in db.execute(
                "SELECT n, beta, g, seed, generator, edge_order, num_edges, digest, "
                "measures_version FROM graphs ORDER BY n, beta, g"
            ):
                put("G", row)
            for row in db.execute(
                f"SELECT {_KEY_COLUMNS}, tr.u, tr.p_float_hex, tr.run_seed, tr.graph_digest, "
                f"tr.dynamics_version FROM trajectories tr ORDER BY {order}"
            ):
                put("T", row)
            for row in db.execute(
                f"SELECT tr.tid, {_KEY_COLUMNS}, t.layers, t.final_layer, t.k_points, t.tau, "
                f"t.tau0, t.tau1, t.censored0, t.status, t.num_measurements FROM traces t JOIN "
                f"trajectories tr ON tr.tid = t.tid ORDER BY {order}, t.layers"
            ):
                if currents is not None:
                    layers = currents(row[1], row[2], row[3]).requirements["layers"]
                    if int(row[8]) != int(layers):
                        continue
                what = f"trace {row[1:8]}"
                kp = None if row[10] is None else _int32_array(row[10], what).tolist()
                tau = None if row[11] is None else _int32_array(row[11], what).tolist()
                put("R", [*row[1:10], kp, tau, *row[12:]])
            if history:
                for row in db.execute(
                    f"SELECT {_KEY_COLUMNS}, co.requirements FROM completions co JOIN "
                    f"trajectories tr ON tr.tid = co.tid ORDER BY {order}, co.requirements"
                ):
                    put("C", row)
                for row in db.execute(
                    f"SELECT {_KEY_COLUMNS}, d.derivation FROM derivations d JOIN trajectories "
                    f"tr ON tr.tid = d.tid ORDER BY {order}, d.derivation"
                ):
                    put("Z", row)
            for row in db.execute(
                f"SELECT {_KEY_COLUMNS}, s.t, s.status, s.num_qubits, s.rank, s.k, "
                f"s.joint_digest, s.phys_digest FROM snapshots s JOIN trajectories tr ON "
                f"tr.tid = s.tid ORDER BY {order}, s.t"
            ):
                if not history and int(row[7]) not in schedule(row[0], row[1], row[2]):
                    continue
                put("S", row)
            used: set[tuple[str, str, str]] = set()
            for row in db.execute(
                f"SELECT tr.tid, {_KEY_COLUMNS}, o.t, o.observable, o.version, o.params, "
                f"o.context, o.value, o.se, o.method, o.data, o.status FROM observations o JOIN "
                f"trajectories tr ON tr.tid = o.tid ORDER BY {order}, o.t, o.observable, "
                f"o.version, o.params, o.context"
            ):
                ident = (row[9], row[10], row[11])
                if currents is not None:
                    if int(row[8]) == -1:
                        cur = currents(row[1], row[2], row[3])
                        if row[12] != cur.context or ident not in cur.derived:
                            continue
                    elif int(row[8]) not in schedule(row[1], row[2], row[3]):
                        continue
                used.add(ident)
                text = _data_text(row[16], "observation")
                put(
                    "O",
                    [*row[1:13], _float_text(row[13]), _float_text(row[14]), row[15], text,
                     row[17]],
                )  # fmt: skip
            for row in db.execute(
                "SELECT observable, version, params, params_json, level, inputs, definition "
                "FROM observable_defs ORDER BY observable, version, params"
            ):
                if history or (row[0], row[1], row[2]) in used:
                    put("D", row)
            if history:
                for row in db.execute(
                    f"SELECT {_KEY_COLUMNS}, sr.outcome_seed, sr.layers, sr.depths, sr.outcomes, "
                    f"sr.random, sr.signed_digests, sr.format FROM signed_runs sr JOIN "
                    f"trajectories tr ON tr.tid = sr.tid ORDER BY {order}, sr.outcome_seed, "
                    f"sr.layers, sr.depths"
                ):
                    outcomes = zlib.decompress(row[10]).hex()
                    flags = zlib.decompress(row[11]).hex()
                    put("X", [*row[:10], outcomes, flags, *row[12:]])
        return h.hexdigest()

    # -- status and export ---------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """A summary: the experiment, revisions, row counts, units done and failed, and per
        condition the trajectories stored and complete under the latest requirements."""
        from sparsegf2.study.runner import plan as _plan

        counts = self.counts()
        units = self.units()
        plan = _plan(self)
        pending = 0
        latest = self.manifest
        if "F" in latest.protocol_names:
            for cond in latest.conditions():
                cur = self._current("F", cond.n, cond.beta)
                for p in cond.rates:
                    complete = self.completed("F", cond.n, cond.beta, p, cur.digest)
                    derived = self.derived_pairs("F", cond.n, cond.beta, p, cur.derivation_digest)
                    pending += len(complete - derived)
        return {
            "experiment_id": self.experiment_id,
            "revision": self.revision,
            "counts": counts,
            "units_done": sum(u["status"] == "done" for u in units),
            "units_failed": sum(u["status"] == "failed" for u in units),
            "units_remaining": len(plan),
            "trajectories_remaining": sum(u.trajectories for u in plan),
            "derivations_remaining": pending,
        }

    def export(self, directory: str | os.PathLike[str]) -> Path:
        """Write the store's data as open tables (CSV and NPZ) under ``directory`` (design §4).

        Layout (format :data:`EXPORT_FORMAT`; doc 05 §3.9): ``manifest.json`` (the latest
        revision), ``revisions.json``, ``versions.json`` (the versions and both logical
        digests), ``requirements.json`` (every requirements, derivation and context record by
        its digest), ``graphs.csv`` and ``graph_aggregates.csv``, ``conditions.csv``, and one
        directory per condition ``<protocol>_n<n>_beta<b>_p<p>`` (a ``/`` of a rational token
        written ``over``) with ``trajectories.csv`` (the latest trace of each trajectory, the
        current requirements and derivation it is complete under), ``first_passage.npz``,
        ``censoring.csv``, ``observations.csv`` and ``observations.npz`` (every snapshot value
        and every current trajectory-level value, with its ``status``; the CSV also has the
        data as canonical JSON), ``history.csv`` (every trajectory-level value with a
        ``current`` flag), ``peaks.csv`` (each trajectory's current peaks: ``T*``, ``d*``, the
        horizon flag, the maximizing set and ``k`` on it), ``peaks.json`` (the peak-time
        statistics) and ``aggregates.csv`` (the nested equal-graph-weight mean and
        graph-cluster error of every scalar snapshot observable at every depth, with the rows
        left out per status). Everything is read in one committed state (:meth:`reading`) and
        the observations are streamed. The files are written in a temporary directory that is
        renamed to ``directory`` when complete, so an interrupted export leaves no partial
        result; ``directory`` must not exist.
        """
        return _export(self, Path(directory))


def _float_text(x: Any) -> str | None:
    return None if x is None else float(x).hex()


def _float_array(values: list[Any]) -> np.ndarray:
    def conv(v: Any) -> Any:
        if v is None:
            return np.nan
        if isinstance(v, list):
            return [conv(x) for x in v]
        if isinstance(v, bool):
            return float(v)
        return v

    return np.array([conv(v) for v in values], dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# writes
# ---------------------------------------------------------------------------------------------


def _differs(what: str, field: str, stored: Any, new: Any) -> ValueError:
    return ValueError(
        f"{what} is already stored with another {field} (stored {stored!r}, new {new!r}): "
        "stored data is never overwritten; this means the store, the code or the inputs changed "
        "(run study verify, and check the versions the store was made with)"
    )


class _Transaction:
    """The writes of one transaction (:meth:`TableauStore.transaction`): insert, or compare
    with what is stored under the same identity and raise on a difference."""

    def __init__(self, store: TableauStore, conn: sqlite3.Connection):
        self.store = store
        self._conn = conn
        self.level = int(store.manifest.storage.compression_level)
        self._keys: dict[tuple[str, str | None, str], tuple[str, str]] = {}
        self._defs: set[tuple[str, str, str]] = set()
        #: the trajectories inserted in this transaction: nothing of them can be stored before
        self.inserted: set[int] = set()

    # graphs
    def put_graph(self, record: GraphRecord, measures: Mapping[str, Any] | None = None) -> None:
        what = f"graph (n = {record.n}, beta = {record.beta}, g = {record.g})"
        row = self._conn.execute(
            "SELECT seed, generator, edge_order, num_edges, digest FROM graphs WHERE n = ? AND "
            "beta = ? AND g = ?",
            (record.n, record.beta.token, record.g),
        ).fetchone()
        if row is not None:
            new = (json.dumps(list(record.seed)), record.generator, record.edge_order,
                   record.num_edges, record.edge_digest)  # fmt: skip
            names = ("seed", "generator", "edge order", "edge count", "edge digest")
            for name, a, b in zip(names, row, new, strict=True):
                if a != b:
                    raise _differs(what, name, a, b)
            return
        record.verify()
        m = graph_measures(record.edges, record.n) if measures is None else dict(measures)
        self._conn.execute(
            "INSERT INTO graphs (n, beta, g, seed, generator, edge_order, num_edges, edges, digest, "
            "measures, measures_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.n,
                record.beta.token,
                record.g,
                json.dumps(list(record.seed)),
                record.generator,
                record.edge_order,
                record.num_edges,
                _int32_blob(record.edges, self.level),
                record.edge_digest,
                canonical_json(_plain(m)),
                GRAPH_MEASURES_VERSION,
            ),
        )

    # bookkeeping
    def put_requirements(self, digest: str, text: str) -> None:
        if sha256_hex(text.encode("utf-8")) != digest:
            raise ValueError("requirements: the digest does not match the text")
        row = self._conn.execute(
            "SELECT json FROM requirements WHERE digest = ?", (digest,)
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO requirements (digest, json) VALUES (?, ?)", (digest, text)
            )
        elif row[0] != text:
            raise _differs(f"requirements {digest}", "text", row[0], text)

    def add_unit(self, **fields: Any) -> int:
        cols = (
            "protocol", "n", "beta", "p", "graphs", "circuits", "layers", "requirements", "status",
            "revision", "started_utc", "finished_utc", "elapsed_seconds", "threads",
            "trajectories", "error", "sparsegf2_version", "git_commit", "git_dirty",
            "randomness_contract", "environment", "environment_digest",
        )  # fmt: skip
        values = [fields.get(c) for c in cols]
        cur = self._conn.execute(
            f"INSERT INTO units ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", values
        )
        assert cur.lastrowid is not None
        return int(cur.lastrowid)

    def finish_unit(self, unit_id: int, *, finished_utc: str, elapsed_seconds: float) -> None:
        """Set the finish time of the unit this transaction created (bookkeeping, not data)."""
        self._conn.execute(
            "UPDATE units SET finished_utc = ?, elapsed_seconds = ? WHERE unit_id = ?",
            (finished_utc, float(elapsed_seconds), int(unit_id)),
        )

    def put_observable_def(self, observable: str, version: str, params: str) -> None:
        if (observable, version, params) in self._defs:
            return
        self._defs.add((observable, version, params))
        row = self._conn.execute(
            "SELECT 1 FROM observable_defs WHERE observable = ? AND version = ? AND params = ?",
            (observable, version, params),
        ).fetchone()
        if row is not None:
            return
        entry = ob.get(observable, version).row()
        self._conn.execute(
            "INSERT INTO observable_defs (observable, version, params, params_json, level, inputs, "
            "definition, created_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                observable,
                version,
                params,
                params,
                entry["level"],
                canonical_json(entry["inputs"]),
                entry["definition"],
                _now(),
            ),
        )

    # trajectories
    def put_trajectory(
        self,
        key: TrajectoryKey,
        *,
        u: int,
        p_float_hex: str,
        run_seed: Sequence[int],
        graph_digest: str,
        dynamics_version: str,
        unit_id: int,
        revision: int,
    ) -> int:
        row = self._conn.execute(
            "SELECT tid, u, p_float_hex, run_seed, graph_digest, dynamics_version FROM trajectories "
            "WHERE protocol = ? AND n = ? AND beta = ? AND p = ? AND g = ? AND c = ? AND r = ?",
            TableauStore._key_params(key),
        ).fetchone()
        seed = json.dumps([int(w) for w in run_seed])
        if row is not None:
            new = (int(u), p_float_hex, seed, graph_digest, dynamics_version)
            names = ("u", "p_float_hex", "run_seed", "graph_digest", "dynamics_version")
            for name, a, b in zip(names, row[1:], new, strict=True):
                if a != b:
                    raise _differs(f"trajectory {key}", name, a, b)
            return int(row[0])
        cur = self._conn.execute(
            "INSERT INTO trajectories (protocol, n, beta, p, g, c, r, u, p_float_hex, run_seed, "
            "graph_digest, dynamics_version, unit_id, revision) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                *TableauStore._key_params(key),
                int(u),
                p_float_hex,
                seed,
                graph_digest,
                dynamics_version,
                int(unit_id),
                int(revision),
            ),  # fmt: skip
        )
        assert cur.lastrowid is not None
        self.inserted.add(int(cur.lastrowid))
        return int(cur.lastrowid)

    def put_trace(
        self,
        tid: int,
        key: TrajectoryKey,
        *,
        layers: int,
        final_layer: int,
        k: np.ndarray | None,
        tau: np.ndarray | None,
        status: str,
        num_measurements: int,
        unit_id: int,
        revision: int,
    ) -> None:
        what = f"trace of {key}"
        if k is not None:  # validate before comparing: k nonincreasing, k_0 <= n, tau exact
            k = np.asarray(k, dtype=np.int64)
            if k.shape != (layers + 1,) or k[0] > key.n:
                raise ValueError(f"{what}: k must hold k_0 <= n, ..., k_T for T = {layers}")
            fp = ob.first_passage(k, num_system=key.n)  # refuses an increasing trace
            if tau is None or not np.array_equal(np.asarray(tau), fp.tau):
                raise ValueError(f"{what}: tau must be the exact first passages of the k trace")
        elif tau is not None:
            raise ValueError(f"{what}: tau without a k trace")
        rows = self._conn.execute(
            "SELECT layers, final_layer, k_points, tau, tau0, tau1, censored0, status, "
            "num_measurements, unit_id, revision FROM traces WHERE tid = ? ORDER BY layers",
            (tid,),
        ).fetchall()
        for row in rows:
            old = TableauStore._trace_record(key, row)
            if old.layers == layers:  # the same horizon: everything must be equal
                same = (
                    old.final_layer == final_layer
                    and old.status == status
                    and old.num_measurements == num_measurements
                    and _same_ints(old.k, k)
                    and _same_ints(old.tau, tau)
                )
                if not same:
                    raise ValueError(
                        f"{what} at horizon {layers} is already stored and differs from the new "
                        "one (k trace, tau, final layer or status): stored data is never "
                        "overwritten; the trajectory did not reproduce"
                    )
                return
            short, long = (old, None) if old.layers < layers else (None, old)
            # the shorter of the two traces must be a prefix of the longer (v1.4 §7.3)
            k_short = old.k if short is not None else k
            k_long = k if short is not None else old.k
            depth = min(old.layers, layers)
            if (k_short is None) != (k_long is None) or (
                k_short is not None
                and k_long is not None
                and not np.array_equal(k_long[: depth + 1], k_short)
            ):
                raise ValueError(
                    f"{what}: the k traces at horizons {old.layers} and {layers} disagree on "
                    f"layers 0..{depth} (v1.4 §7.3 prefix invariance): the trajectory did not "
                    "reproduce"
                )
            if (
                short is not None
                and short.final_layer < short.layers
                and final_layer != short.final_layer
            ):
                raise ValueError(
                    f"{what}: it stopped at layer {short.final_layer} at horizon "
                    f"{short.layers} but at layer {final_layer} at horizon {layers}"
                )
            if long is not None and final_layer < layers and long.final_layer != final_layer:
                raise ValueError(
                    f"{what}: it stopped at layer {final_layer} at horizon {layers} but the "
                    f"stored horizon {long.layers} has final layer {long.final_layer}"
                )
        tau0 = tau1 = cens = None
        if tau is not None:
            tau0, tau1 = int(tau[0]), int(tau[1])
            cens = int(tau[0] < 0)
        self._conn.execute(
            "INSERT INTO traces (tid, layers, final_layer, k_points, tau, tau0, tau1, censored0, "
            "status, num_measurements, unit_id, revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                tid,
                int(layers),
                int(final_layer),
                None if k is None else _int32_blob(ob.k_change_points(k), self.level),
                None if tau is None else _int32_blob(tau, self.level),
                tau0,
                tau1,
                cens,
                status,
                int(num_measurements),
                int(unit_id),
                int(revision),
            ),
        )

    def put_completion(self, tid: int, requirements: str, unit_id: int, revision: int) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO completions (tid, requirements, unit_id, revision) "
            "VALUES (?, ?, ?, ?)",
            (tid, requirements, int(unit_id), int(revision)),
        )

    # snapshots
    def put_snapshot(
        self,
        tid: int,
        key: TrajectoryKey,
        t: int,
        *,
        status: str,
        num_qubits: int,
        k: int | None,
        joint: PhaseFreeTableau | None,
        phys: PhaseFreeTableau | None,
        unit_id: int,
        check_physical: bool = True,
    ) -> None:
        """Insert one snapshot, or compare it with the stored one (see the module docstring).

        Validated and encoded by :func:`snapshot_row` before anything is written (purity,
        rank ``n - k``, isotropy and, with ``check_physical``, the physical code of the joint
        state), then written by :meth:`put_snapshot_row`.
        """
        row = snapshot_row(
            key, t, status=status, num_qubits=num_qubits, k=k, joint=joint, phys=phys,
            level=self.level, check_physical=check_physical,
        )  # fmt: skip
        self.put_snapshot_row(tid, key, row, unit_id)

    def put_snapshot_row(
        self, tid: int, key: TrajectoryKey, row: SnapshotRow, unit_id: int, *, fresh: bool = False
    ) -> None:
        """Write a validated snapshot row: insert it, or compare it with the stored one (the one
        fill-in: the joint state of a row that kept its physical code only). ``fresh`` (the
        trajectory was inserted in this transaction, so nothing of it can be stored) skips the
        lookup."""
        what = f"snapshot of {key} at t = {row.t}"
        if not fresh:
            stored = self._conn.execute(
                "SELECT status, num_qubits, rank, k, joint_digest, phys_digest FROM snapshots "
                "WHERE tid = ? AND t = ?",
                (tid, row.t),
            ).fetchone()
            if stored is not None:
                if stored[0] == SNAPSHOT_PHYS_ONLY and row.status == SNAPSHOT_STORED:
                    # the one fill-in: the joint state of a row that kept the physical code only
                    new = (row.num_qubits, row.k, row.phys_digest)
                    if (stored[1], stored[3], stored[5]) != new:
                        raise _differs(
                            what, "physical code", (stored[1], stored[3], stored[5]), new
                        )
                    self._conn.execute(
                        "UPDATE snapshots SET joint = ?, joint_digest = ?, rank = ?, status = ? "
                        "WHERE tid = ? AND t = ? AND joint IS NULL AND status = ?",
                        (row.joint, row.joint_digest, row.rank, SNAPSHOT_STORED, tid, row.t,
                         SNAPSHOT_PHYS_ONLY),
                    )  # fmt: skip
                    return
                _compare_snapshot(what, stored, row)
                return
        self._conn.execute(
            "INSERT INTO snapshots (tid, t, num_qubits, rank, k, joint, joint_digest, phys, "
            "phys_digest, status, unit_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (tid, row.t, row.num_qubits, row.rank, row.k, row.joint, row.joint_digest, row.phys,
             row.phys_digest, row.status, int(unit_id)),
        )  # fmt: skip

    # observations
    def put_observation(
        self,
        tid: int,
        key: TrajectoryKey,
        t: int,
        observable: str,
        params: Mapping[str, Any] | None,
        *,
        value: Any,
        se: Any,
        data: Any,
        method: str,
        context: str = "",
        unit_id: int | None = None,
        version: str | None = None,
        compare: str = "all",
        status: str | None = None,
    ) -> bool:
        """Insert one value (``True``), or check it against the stored one (``False``).

        ``status`` (:data:`OBSERVATION_STATUSES`) says what the value is; by default ``ok`` for
        a finite value, ``infinite`` for an infinite one and ``undefined`` for NaN (the runner
        names ``undefined_no_code``, ``missing``, ``censored``, ``vector`` itself).

        ``compare="all"`` requires an equal value, error, status, method and data; ``"value"``
        (a retrospective recomputation of a value the run stored, possibly by another method)
        requires an equal value, error and status. A difference raises ``ValueError``.
        """
        cache_key = (observable, version, canonical_json(_plain(params or {})))
        if cache_key not in self._keys:
            entry = ob.get(observable, version)
            text = ob.params_key(observable, params, version=entry.version)
            self._keys[cache_key] = (entry.version, text)
        resolved_version, text = self._keys[cache_key]
        row = observation_row(
            t, observable, text, value=value, se=se, data=data, method=method, level=self.level,
            context=context, version=resolved_version, status=status,
        )  # fmt: skip
        return self.put_observation_row(tid, key, row, unit_id, compare=compare)

    def put_observation_row(
        self,
        tid: int,
        key: TrajectoryKey,
        row: ObservationRow,
        unit_id: int | None,
        *,
        compare: str = "all",
        fresh: bool = False,
    ) -> bool:
        """Write an encoded observation row: insert it (``True``) or compare it with the stored one
        (``False``; :func:`_compare_observation`). ``fresh`` skips the lookup (the trajectory was
        inserted in this transaction)."""
        if compare not in ("all", "value"):
            raise ValueError(f"compare must be 'all' or 'value', got {compare!r}")
        self.put_observable_def(row.observable, row.version, row.params)
        if not fresh:
            stored = self._conn.execute(
                "SELECT value, se, method, data, status FROM observations WHERE tid = ? AND t = ? "
                "AND observable = ? AND version = ? AND params = ? AND context = ?",
                (tid, row.t, row.observable, row.version, row.params, row.context),
            ).fetchone()
            if stored is not None:
                what = (
                    f"observation {row.observable!r} (version {row.version}) of {key} at "
                    f"t = {row.t}"
                )
                _compare_observation(what, stored, row, compare)
                return False
        self._conn.execute(
            "INSERT INTO observations (tid, t, observable, version, params, context, status, "
            "value, se, data, method, unit_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _observation_values(tid, row, unit_id),
        )
        return True

    def put_fresh_observations(
        self, tid: int, rows: Sequence[ObservationRow], unit_id: int | None
    ) -> None:
        """Insert the observation rows of a trajectory inserted in this transaction (nothing of
        it can be stored yet, so nothing is looked up or compared); one statement for all."""
        for row in rows:
            self.put_observable_def(row.observable, row.version, row.params)
        self._conn.executemany(
            "INSERT INTO observations (tid, t, observable, version, params, context, status, "
            "value, se, data, method, unit_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [_observation_values(tid, row, unit_id) for row in rows],
        )

    def put_derivation(self, tid: int, derivation: str, unit_id: int | None, revision: int) -> None:
        """Mark trajectory ``tid`` derived under ``derivation`` (every trajectory-level value it
        asks for is stored), unless it is marked already."""
        self._conn.execute(
            "INSERT OR IGNORE INTO derivations (tid, derivation, unit_id, revision) "
            "VALUES (?, ?, ?, ?)",
            (tid, derivation, unit_id, int(revision)),
        )

    # aggregates (G-STO-4)
    def put_aggregate(self, record: Mapping[str, Any], data: Any = None) -> bool:
        """Insert one aggregate record (``True``), or compare it with the stored one."""
        keys = ("protocol", "n", "beta", "p", "observable", "version", "params", "selection", "t",
                "context", "estimator", "policy", "sample_set")  # fmt: skip
        values = ("mean", "se", "graphs", "contributing_graphs", "trajectories",
                  "contributing_trajectories")  # fmt: skip
        old = self._conn.execute(
            f"SELECT {', '.join(values)} FROM aggregates WHERE "
            + " AND ".join(f"{k} = ?" for k in keys),
            [record[k] for k in keys],
        ).fetchone()
        new = [_sql_float(record[v]) if v in ("mean", "se") else record[v] for v in values]
        if old is not None:
            if any(
                not _same_float(a, b) if i < 2 else a != b
                for i, (a, b) in enumerate(zip(old, new, strict=True))
            ):
                raise _differs(f"aggregate {[record[k] for k in keys]}", "values", old, new)
            return False
        self._conn.execute(
            f"INSERT INTO aggregates ({', '.join((*keys, *values))}, data, created_utc) VALUES "
            f"({', '.join('?' * (len(keys) + len(values) + 2))})",
            [*(record[k] for k in keys), *new, _data_blob(data, self.level), _now()],
        )
        return True

    # outcome-aware replays (G-TAB-2)
    def put_signed_run(
        self,
        tid: int,
        key: TrajectoryKey,
        *,
        outcome_seed: Sequence[int] | int,
        layers: int,
        depths: Sequence[int],
        outcomes: np.ndarray,
        random: np.ndarray,
        signed_digests: Mapping[int, str],
    ) -> bool:
        """Record an outcome-aware replay (``True``), or check it against the stored one."""
        seed_text = canonical_json(_plain(outcome_seed))
        depth_text = canonical_json([int(t) for t in depths])
        out = zlib.compress(np.asarray(outcomes, dtype=np.uint8).tobytes(), self.level)
        flags = zlib.compress(np.asarray(random, dtype=np.uint8).tobytes(), self.level)
        digests = canonical_json({str(int(t)): d for t, d in signed_digests.items()})
        row = self._conn.execute(
            "SELECT outcomes, random, signed_digests, format FROM signed_runs WHERE tid = ? AND "
            "outcome_seed = ? AND layers = ? AND depths = ?",
            (tid, seed_text, int(layers), depth_text),
        ).fetchone()
        what = f"signed replay of {key} with outcome seed {seed_text}"
        if row is not None:
            same = (
                zlib.decompress(row[0]) == zlib.decompress(out)
                and zlib.decompress(row[1]) == zlib.decompress(flags)
                and row[2] == digests
                and row[3] == SIGNED_FORMAT
            )
            if not same:
                raise _differs(what, "outcomes or signed groups", "stored", "new")
            return False
        self._conn.execute(
            "INSERT INTO signed_runs (tid, outcome_seed, layers, depths, outcomes, random, "
            "signed_digests, format, created_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (tid, seed_text, int(layers), depth_text, out, flags, digests, SIGNED_FORMAT, _now()),
        )
        return True

    # sampler diagnostics (G-OBS-4)
    def put_sampler_diagnostic(self, **row: Any) -> bool:
        """Record the graph-side diagnostics of one correlated sampler (``True``), or compare."""
        cols = (
            "n", "beta", "g", "probe", "config", "m", "coupling", "method", "chains", "samples",
            "seed", "ess", "tau_int", "acceptance", "exchange_acceptance", "r_hat",
            "r_hat_limit", "notes",
        )  # fmt: skip
        values = [row[c] for c in cols]
        data = _data_blob(row.get("data"), self.level)
        old = self._conn.execute(
            f"SELECT {', '.join(cols[7:])} FROM sampler_diagnostics WHERE n = ? AND beta = ? AND "
            "g = ? AND probe = ? AND config = ? AND m = ? AND coupling = ?",
            values[:7],
        ).fetchone()
        if old is not None:
            if [_plain(v) for v in old] != [_plain(v) for v in values[7:]]:
                raise _differs(f"sampler diagnostics {values[:7]}", "values", old, values[7:])
            return False
        self._conn.execute(
            f"INSERT INTO sampler_diagnostics ({', '.join(cols)}, data, created_utc) VALUES "
            f"({', '.join('?' * (len(cols) + 2))})",
            (*values, data, _now()),
        )
        return True


def _observation_values(tid: int, row: ObservationRow, unit_id: int | None) -> tuple[Any, ...]:
    """The values of an ``INSERT INTO observations`` of ``row`` (the column order of the store)."""
    return (tid, row.t, row.observable, row.version, row.params, row.context, row.status,
            row.value, row.se, row.data, row.method, unit_id)  # fmt: skip


def _same_ints(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.array_equal(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)))


# ---------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------


def _condition_dir(protocol: str, n: int, beta: str, p: str) -> str:
    """The directory name of a condition: ``F_n16_beta0.1_p1over7`` (``/`` written ``over``)."""
    return f"{protocol}_n{n}_beta{beta}_p{p}".replace("/", "over")


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([_csv_cell(v) for v in row])


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_plain(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _export(store: TableauStore, directory: Path) -> Path:
    if directory.exists():
        raise ValueError(
            f"{str(directory)!r} exists: an export is written to a new directory (remove the old "
            "export or choose another name)"
        )
    directory.parent.mkdir(parents=True, exist_ok=True)
    tmp = directory.parent / f".{directory.name}.partial-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        _export_into(store, tmp)
        os.rename(tmp, directory)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return directory


_MEASURES = (
    "q",
    "rewired_edges",
    "mean_distance",
    "reachable_pairs",
    "components",
    "lambda2",
    "degree_variance",
)


def _export_into(store: TableauStore, out: Path) -> None:
    with store.reading() as db:
        revisions = store.revisions()
        (out / "manifest.json").write_text(revisions[-1].to_json(indent=2) + "\n", encoding="utf-8")
        _write_json(
            out / "revisions.json",
            [
                {"revision": i + 1, "digest": m.digest(), "manifest": m.to_dict()}
                for i, m in enumerate(revisions)
            ],
        )
        _write_json(
            out / "versions.json",
            {
                "export_format": EXPORT_FORMAT,
                "store": store.meta(),
                "randomness_contract": mipt.RANDOMNESS_CONTRACT,
                "logical_digest": store.logical_digest(),
                "logical_digest_current": store.logical_digest(history=False),
            },
        )
        _write_json(
            out / "requirements.json",
            {
                str(digest): json.loads(text)
                for digest, text in db.execute(
                    "SELECT digest, json FROM requirements ORDER BY digest"
                )
            },
        )
        graph_rows = []
        for n, beta, g, seed, generator, order, count, digest, measures, mversion in db.execute(
            "SELECT n, beta, g, seed, generator, edge_order, num_edges, digest, measures, "
            "measures_version FROM graphs ORDER BY n, beta, g"
        ):
            m = json.loads(measures)
            graph_rows.append(
                [n, beta, g, seed, generator, order, count, digest, mversion]
                + [m.get(name) for name in _MEASURES]
            )
        _write_csv(
            out / "graphs.csv",
            ["n", "beta", "g", "seed", "generator", "edge_order", "num_edges", "edge_digest"]
            + ["measures_version", *_MEASURES],
            graph_rows,
        )
        _write_csv(
            out / "graph_aggregates.csv",
            ["n", "beta", "measure", "graphs", "defined", "mean", "se", "connected_fraction"],
            _graph_aggregates(graph_rows),
        )
        conditions = db.execute(
            "SELECT DISTINCT protocol, n, beta, p FROM trajectories ORDER BY protocol, n, beta, p"
        ).fetchall()
        currents = store._currents()
        condition_rows = []
        for protocol, n, beta, p in conditions:
            name = _condition_dir(protocol, n, beta, p)
            cur = currents(protocol, int(n), beta)
            count = _export_condition(store, out / name, protocol, int(n), beta, p, cur)
            condition_rows.append([protocol, n, beta, p, name, count])
        _write_csv(
            out / "conditions.csv",
            ["protocol", "n", "beta", "p", "directory", "trajectories"],
            condition_rows,
        )


def _graph_aggregates(graph_rows: Sequence[Sequence[Any]]) -> list[list[Any]]:
    """Per ``(n, beta)`` and graph measurement: the mean over the stored graphs (each graph one
    independent draw of v1.4 §2.2, equal weight), its standard error ``s / sqrt(G)``, and the
    fraction of connected graphs. ``mean_distance`` is the reachable-pair mean of each graph
    (design §12): on a disconnected graph it averages within components, so its mean is read
    with ``connected_fraction`` beside it; graphs without a reachable pair are left out
    (``defined``)."""
    groups: dict[tuple[int, str], list[Sequence[Any]]] = {}
    for row in graph_rows:
        groups.setdefault((int(row[0]), str(row[1])), []).append(row)
    out: list[list[Any]] = []
    offset = 9  # the measurements follow the nine record columns
    for (n, beta), rows in sorted(groups.items()):
        components = np.array([r[offset + _MEASURES.index("components")] for r in rows], float)
        connected = float(np.mean(components == 1))
        for i, name in enumerate(_MEASURES):
            values = np.array([_nan(r[offset + i]) for r in rows], dtype=np.float64)
            kept = values[np.isfinite(values)]
            mean = float(kept.mean()) if kept.size else math.nan
            se = float(kept.std(ddof=1) / math.sqrt(kept.size)) if kept.size > 1 else math.nan
            out.append([n, beta, name, len(rows), int(kept.size), mean, se, connected])
    return out


def _data_dict(blob: bytes | None, what: str) -> dict[str, Any]:
    text = _data_text(blob, what)
    return {} if text is None else dict(json.loads(text))


_OBSERVATION_COLUMNS = (
    "g", "c", "r", "t", "observable", "version", "params", "context", "status", "value", "se",
    "method",
)  # fmt: skip

_PEAK_COLUMNS = (
    "observable", "version", "params", "context", "g", "c", "r", "status", "t_star", "d_star",
    "at_horizon", "horizon", "set_size", "set_min", "set_max", "maximizing_set", "k_on_set",
    "se_exact",
)  # fmt: skip


def _export_condition(
    store: TableauStore, out: Path, protocol: str, n: int, beta: str, p: str, cur: Any
) -> int:
    """The tables of one condition (see :meth:`TableauStore.export`), read in the export's
    snapshot. The observations are streamed in key order: each row is written as it is read
    (its data decoded only where a table needs them), and the aggregates of one key are made
    when its rows end, so memory holds one group and the columns of ``observations.npz``."""
    out.mkdir()
    db = store._db()
    args = (protocol, n, beta, p)
    complete = {
        int(r[0])
        for r in db.execute(
            "SELECT co.tid FROM completions co JOIN trajectories tr ON tr.tid = co.tid WHERE "
            "tr.protocol = ? AND tr.n = ? AND tr.beta = ? AND tr.p = ? AND co.requirements = ?",
            (*args, cur.digest),
        )
    }
    derived = {
        int(r[0])
        for r in db.execute(
            "SELECT d.tid FROM derivations d JOIN trajectories tr ON tr.tid = d.tid WHERE "
            "tr.protocol = ? AND tr.n = ? AND tr.beta = ? AND tr.p = ? AND d.derivation = ?",
            (*args, cur.derivation_digest),
        )
    }
    rows = db.execute(
        "SELECT tr.tid, tr.g, tr.c, tr.r, tr.u, t.layers, t.final_layer, t.status, t.tau0, "
        "t.tau1, t.censored0, t.num_measurements, t.tau FROM trajectories tr JOIN traces t "
        "ON t.tid = tr.tid WHERE tr.protocol = ? AND tr.n = ? AND tr.beta = ? AND tr.p = ? "
        "AND t.layers = (SELECT max(layers) FROM traces x WHERE x.tid = tr.tid) "
        "ORDER BY tr.g, tr.c, tr.r",
        args,
    ).fetchall()
    _write_csv(
        out / "trajectories.csv",
        ["g", "c", "r", "u", "layers", "final_layer", "status", "tau0", "tau1", "censored0"]
        + ["num_measurements", "requirements", "derivation", "context"],
        [
            list(row[1:12])
            + [cur.digest if int(row[0]) in complete else ""]
            + [cur.derivation_digest if int(row[0]) in derived else ""]
            + [cur.context if int(row[0]) in derived else ""]
            for row in rows
        ],
    )
    g = np.array([row[1] for row in rows], dtype=np.int64)
    c = np.array([row[2] for row in rows], dtype=np.int64)
    r = np.array([row[3] for row in rows], dtype=np.int64)
    if protocol == "F" and rows:
        tau = np.stack([_int32_array(row[12], "tau") for row in rows])
        layers = np.array([row[5] for row in rows], dtype=np.int64)
        np.savez_compressed(
            out / "first_passage.npz", g=g, c=c, r=r, tau=tau, observed=tau >= 0, horizon=layers
        )
        censoring = []
        for ell in range(n + 1):
            observed = tau[:, ell] >= 0
            times = np.where(observed, tau[:, ell], layers)
            summary = stats.censoring_summary(g, c, r, times, observed)
            curve = stats.nested_kaplan_meier(g, c, r, times, observed)
            censoring.append(
                [ell, summary.trajectories, summary.observed, summary.censored]
                + [summary.censored_fraction, summary.censored_fraction_se, summary.horizon]
                + [curve.median, curve.median_upper, curve.restricted_horizon]
                + [curve.restricted_mean, curve.restricted_mean_se]
            )
        _write_csv(
            out / "censoring.csv",
            ["ell", "trajectories", "observed", "censored", "censored_fraction"]
            + ["censored_fraction_se", "horizon", "km_median", "km_median_upper"]
            + ["restricted_horizon", "restricted_mean", "restricted_mean_se"],
            censoring,
        )
    current_keys = set(cur.derived)
    columns: dict[str, list[Any]] = {name: [] for name in _OBSERVATION_COLUMNS}
    aggregates: list[list[Any]] = []
    peaks: dict[tuple[str, str, str, str], list[tuple[Any, ...]]] = {}
    group: list[tuple[Any, ...]] = []
    group_key: tuple[Any, ...] | None = None
    with (
        open(out / "observations.csv", "w", newline="", encoding="utf-8") as obs_file,
        open(out / "history.csv", "w", newline="", encoding="utf-8") as history_file,
        open(out / "peaks.csv", "w", newline="", encoding="utf-8") as peak_file,
    ):
        obs_csv = csv.writer(obs_file, lineterminator="\n")
        obs_csv.writerow([*_OBSERVATION_COLUMNS, "data"])
        history_csv = csv.writer(history_file, lineterminator="\n")
        history_csv.writerow([*_OBSERVATION_COLUMNS, "current", "data"])
        peak_csv = csv.writer(peak_file, lineterminator="\n")
        peak_csv.writerow(_PEAK_COLUMNS)
        cursor = db.execute(
            "SELECT tr.g, tr.c, tr.r, o.t, o.observable, o.version, o.params, o.context, "
            "o.status, o.value, o.se, o.method, o.data FROM observations o JOIN trajectories tr "
            "ON tr.tid = o.tid WHERE tr.protocol = ? AND tr.n = ? AND tr.beta = ? AND tr.p = ? "
            "ORDER BY o.observable, o.version, o.params, o.context, o.t, tr.g, tr.c, tr.r",
            args,
        )
        for o in cursor:
            t = int(o[3])
            current = t >= 0 or (o[7] == cur.context and (o[4], o[5], o[6]) in current_keys)
            cells = [_csv_cell(v) for v in o[:12]]
            text = _data_text(o[12], f"observation {o[4]!r}")
            if t == -1:
                history_csv.writerow([*cells, "1" if current else "0", text or ""])
            if not current:
                continue
            obs_csv.writerow([*cells, text or ""])
            for name, value in zip(_OBSERVATION_COLUMNS, o[:12], strict=True):
                columns[name].append(value)
            if t >= 0:
                ident = (o[4], o[5], o[6], t)
                if ident != group_key:
                    if group_key is not None:
                        aggregates.extend(_aggregates(group_key, group))
                    group_key, group = ident, []
                group.append(o)
            elif str(o[4]).startswith("peak_"):
                data = {} if text is None else json.loads(text)
                peaks.setdefault((o[4], o[5], o[6], o[7]), []).append(
                    (int(o[0]), int(o[1]), int(o[2]), _nan(o[9]), bool(data.get("at_horizon")))
                )
                peak_csv.writerow([_csv_cell(v) for v in _peak_row(o, data)])
        if group_key is not None:
            aggregates.extend(_aggregates(group_key, group))
    np.savez_compressed(
        out / "observations.npz",
        g=np.array(columns["g"], dtype=np.int64),
        c=np.array(columns["c"], dtype=np.int64),
        r=np.array(columns["r"], dtype=np.int64),
        t=np.array(columns["t"], dtype=np.int64),
        observable=np.array(columns["observable"], dtype=str),
        version=np.array(columns["version"], dtype=str),
        params=np.array(columns["params"], dtype=str),
        context=np.array(columns["context"], dtype=str),
        status=np.array(columns["status"], dtype=str),
        value=np.array([_nan(v) for v in columns["value"]], dtype=np.float64),
        se=np.array([_nan(v) for v in columns["se"]], dtype=np.float64),
        method=np.array(columns["method"], dtype=str),
    )
    _write_csv(
        out / "aggregates.csv",
        ["observable", "version", "params", "t", "mean", "se", "graphs", "contributing_graphs"]
        + ["trajectories", "contributing_trajectories", "nan_policy", "omitted"],
        aggregates,
    )
    summaries = _peak_summaries(peaks)
    if summaries:
        _write_json(out / "peaks.json", summaries)
    return len(rows)


def _peak_row(o: Sequence[Any], data: Mapping[str, Any]) -> list[Any]:
    """One row of ``peaks.csv``: the peak record of one trajectory (v1.4 §5.8): ``T*``,
    ``d*``, the horizon flag, the maximizing set (its size, ends and members) and ``k`` on it."""
    members = list(data.get("maximizing_set") or [])
    k_on = data.get("k_on_set")
    return [
        o[4], o[5], o[6], o[7], o[0], o[1], o[2], o[8], _nan(o[9]),
        _nan(data.get("d_star")), bool(data.get("at_horizon")), data.get("horizon"),
        len(members), min(members) if members else None, max(members) if members else None,
        json.dumps(members), None if k_on is None else json.dumps(k_on), data.get("se_exact"),
    ]  # fmt: skip


def _aggregates(ident: tuple[Any, ...], items: Sequence[Sequence[Any]]) -> list[list[Any]]:
    """The nested mean (v1.4 §6.3) and graph-cluster error (§6.4) of one scalar snapshot
    observable at one depth: the stored value, plus ``d_cyc``'s zero extension and ``D(J)`` at
    each coupling from the stored data. A row whose status is not ``ok`` (undefined, infinite,
    absent after a stop, ...) is left out; the row then says ``nan_policy = omit`` and counts
    the rows left out per status in ``omitted`` (the mean is over the rows with a value)."""
    name, version, params, t = ident
    out: list[list[Any]] = []

    def add(label: str, values: list[tuple[int, int, int, float, str]]) -> None:
        gg = np.array([v[0] for v in values], dtype=np.int64)
        cc = np.array([v[1] for v in values], dtype=np.int64)
        rr = np.array([v[2] for v in values], dtype=np.int64)
        kept = np.array([v[4] == "ok" and math.isfinite(v[3]) for v in values], dtype=bool)
        if not kept.any():
            return
        omitted: dict[str, int] = {}
        for v, keep in zip(values, kept.tolist(), strict=True):
            if not keep:
                omitted[v[4]] = omitted.get(v[4], 0) + 1
        vv = np.where(kept, np.array([v[3] for v in values], dtype=np.float64), np.nan)
        policy = "omit" if omitted else "raise"
        est = stats.nested_mean(gg, cc, rr, vv, nan_policy=policy)  # type: ignore[arg-type]
        out.append(
            [label, version, params, t, float(est.mean), float(est.se), est.num_graphs]
            + [int(est.contributing_graphs), est.num_trajectories]
            + [int(est.contributing_trajectories), policy]
            + [json.dumps(dict(sorted(omitted.items())), sort_keys=True)]
        )

    if any(o[9] is not None or o[8] not in ("vector",) for o in items):
        add(name, [(int(o[0]), int(o[1]), int(o[2]), _nan(o[9]), str(o[8])) for o in items])
    if name == "d_cyc":
        zero = [
            (int(o[0]), int(o[1]), int(o[2]), float(_data_dict(o[12], name)["zero_extended"]),
             "ok")
            for o in items
        ]  # fmt: skip
        add("d_cyc:zero_extended", zero)
    if name == "D_J":
        datas = [_data_dict(o[12], name) for o in items]
        for j, coupling in enumerate(datas[0]["couplings"]):
            vals = []
            for o, d in zip(items, datas, strict=True):
                x = _nan(d["robustness"][j])
                vals.append((int(o[0]), int(o[1]), int(o[2]), x,
                             "ok" if math.isfinite(x) else "undefined"))  # fmt: skip
            add(f"D_J:J={coupling!r}", vals)
    return out


def _nan(x: Any) -> float:
    return math.nan if x is None else float(x)


def _peak_summaries(
    peaks: Mapping[tuple[str, str, str, str], Sequence[tuple[Any, ...]]],
) -> dict[str, list[dict[str, Any]]]:
    """The three peak-time statistics of v1.4 §5.8 (:func:`~sparsegf2.study.stats.peak_time_summaries`)
    of every current peak, under both explicit horizon policies (``exclude``, ``include``)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for (name, version, params, context), items in sorted(peaks.items()):
        gg = np.array([i[0] for i in items], dtype=np.int64)
        cc = np.array([i[1] for i in items], dtype=np.int64)
        rr = np.array([i[2] for i in items], dtype=np.int64)
        tstar = np.array([i[3] for i in items], dtype=np.float64)
        at_horizon = np.array([i[4] for i in items], dtype=bool)
        summaries = {
            policy: stats.peak_time_summaries(
                gg, cc, rr, tstar, at_horizon=at_horizon, horizon_policy=policy
            ).to_dict()
            for policy in ("exclude", "include")
        }
        out.setdefault(name, []).append(
            {"version": version, "params": params, "context": context, "summaries": summaries}
        )
    return out
