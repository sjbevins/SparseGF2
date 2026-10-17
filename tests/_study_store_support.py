"""Shared helpers of the store, runner and reproducibility tests (test-only).

* :func:`manifest` builds small study manifests from :data:`BASE` with dotted-path changes.
* :func:`build` creates a store in a temporary directory and runs it.
* :func:`canonical_bytes` writes the ``SGF2RS1`` canonical row-space bytes from 0/1 rows with
  an elimination of its own (``_rowspace_oracle.rref`` on Python ints), never through
  ``sparsegf2.tableau``; :func:`stim_states` gives the unsigned stabilizers of a circuit after
  every ``TICK`` from Stim.
* :func:`records` dumps every logical record of a store by its natural key (decoded), so a test
  can check that a later operation left every earlier record unchanged.
* :func:`run_python` runs a script in a fresh interpreter (the same virtual environment).
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from _rowspace_oracle import int_rows, rref, stim_tick_states
from sparsegf2.study import runner
from sparsegf2.study.manifest import ExperimentManifest
from sparsegf2.study.store import STORE_FILE, TableauStore

#: A small manifest: two rates, three graphs of two circuits (stride 4), n = 6, 10 layers.
BASE: dict[str, Any] = {
    "experiment_id": "ws-store-test",
    "conditions": [{"n": [6], "beta": ["0.25"], "p": ["0.1", "0.3"]}],
    "depth": {"layers": {"fixed": 10}, "checkpoints": {"list": [2, 5]}},
    "trials": {"graphs": 3, "circuits_per_graph": 2, "circuit_index_stride": 4},
    "seeds": {"master_graph_seed": 101, "master_circuit_seed": 202},
    "observables": {
        "tableau_snapshots": True,
        "contiguous_distance": True,
        "uniform_erasure": {"samples": 12, "exact_budget": 6},
        "correlated_erasure": [
            {"couplings": [0, 1.5], "sizes": [1, 3], "samples": 8, "name": "ring"}
        ],
    },
    "storage": {"graphs_per_unit": 2},
    "analysis": {"peak": {"bootstrap": 5}},
}


def manifest(base: dict[str, Any] | None = None, **changes: Any) -> ExperimentManifest:
    """``BASE`` (or ``base``) with changes given as ``path__to__key=value`` (``__`` separates)."""
    data = copy.deepcopy(BASE if base is None else base)
    for path, value in changes.items():
        keys = path.split("__")
        node = data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        if value is DELETE:
            node.pop(keys[-1], None)
        else:
            node[keys[-1]] = value
    return ExperimentManifest.from_json(data)


DELETE = object()


def build(
    directory: Path, m: ExperimentManifest | None = None, *, run: bool = True, **kwargs: Any
) -> TableauStore:
    """A new store in ``directory`` (a writer), run to completion unless ``run`` is False."""
    store = TableauStore.create(directory, manifest() if m is None else m)
    if run:
        runner.run(store, **kwargs)
    return store


# ---------------------------------------------------------------------------------------------
# independent canonical forms
# ---------------------------------------------------------------------------------------------


def canonical_rows(dense: Any) -> list[int]:
    """The canonical (reduced row echelon) rows of 0/1 rows ``(s, 2N)``, columns
    ``x_0..x_{N-1}, z_0..z_{N-1}``, as ints (bit j = column j)."""
    arr = np.asarray(dense, dtype=np.uint8)
    return rref(int_rows(arr), range(arr.shape[1]))


def canonical_bytes(dense: Any, num_qubits: int) -> bytes:
    """The ``SGF2RS1`` bytes written from scratch: ``b"SGF2RS1\\0"``, N and the rank as
    little-endian uint32, then each canonical row's 2N bits, column 0 first, most significant
    bit first in each byte, zero padding at the end."""
    width = 2 * num_qubits
    rows = canonical_rows(np.asarray(dense, dtype=np.uint8).reshape(-1, width))
    out = bytearray(b"SGF2RS1\x00" + struct.pack("<II", num_qubits, len(rows)))
    for r in rows:
        nbytes = (width + 7) // 8
        row = bytearray(nbytes)
        for j in range(width):
            if (r >> j) & 1:
                row[j // 8] |= 0x80 >> (j % 8)
        out += row
    return bytes(out)


def canonical_digest(dense: Any, num_qubits: int) -> str:
    return hashlib.sha256(canonical_bytes(dense, num_qubits)).hexdigest()


def stim_states(circuit: Any, num_qubits: int, seed: int = 0) -> list[np.ndarray]:
    """The unsigned stabilizers ``(N, 2N)`` after every TICK of a sparsegf2 circuit, from Stim."""
    return stim_tick_states(circuit.to_stim(), num_qubits, seed)


# ---------------------------------------------------------------------------------------------
# store dumps
# ---------------------------------------------------------------------------------------------


def _key(row: Any) -> tuple[Any, ...]:
    return tuple(row)


def records(store: TableauStore) -> dict[tuple[Any, ...], Any]:
    """Every logical record of the store by natural key, blobs decoded (no row ids, unit
    numbers or timestamps)."""
    db = store._db()
    out: dict[tuple[Any, ...], Any] = {}
    key_cols = "tr.protocol, tr.n, tr.beta, tr.p, tr.g, tr.c, tr.r"
    for rev, digest in db.execute("SELECT revision, digest FROM manifests"):
        out[("manifest", rev)] = digest
    for row in db.execute(
        "SELECT n, beta, g, seed, generator, edge_order, num_edges, edges, digest, measures "
        "FROM graphs"
    ):
        out[("graph", *row[:3])] = (*row[3:7], zlib.decompress(row[7]), row[8], row[9])
    for row in db.execute(
        f"SELECT {key_cols}, u, p_float_hex, run_seed, graph_digest, dynamics_version "
        "FROM trajectories tr"
    ):
        out[("trajectory", *row[:7])] = row[7:]
    for row in db.execute(
        f"SELECT {key_cols}, t.layers, t.final_layer, t.k_points, t.tau, t.status, "
        "t.num_measurements FROM traces t JOIN trajectories tr ON tr.tid = t.tid"
    ):
        kp = None if row[9] is None else zlib.decompress(row[9])
        tau = None if row[10] is None else zlib.decompress(row[10])
        out[("trace", *row[:8])] = (row[8], kp, tau, row[11], row[12])
    for row in db.execute(
        f"SELECT {key_cols}, s.t, s.status, s.num_qubits, s.rank, s.k, s.joint, s.joint_digest, "
        "s.phys, s.phys_digest FROM snapshots s JOIN trajectories tr ON tr.tid = s.tid"
    ):
        joint = None if row[12] is None else zlib.decompress(row[12])
        phys = None if row[14] is None else zlib.decompress(row[14])
        out[("snapshot", *row[:8])] = (*row[8:12], joint, row[13], phys, row[15])
    for row in db.execute(
        f"SELECT {key_cols}, o.t, o.observable, o.version, o.params, o.context, o.value, o.se, "
        "o.method, o.data FROM observations o JOIN trajectories tr ON tr.tid = o.tid"
    ):
        data = None if row[15] is None else zlib.decompress(row[15])
        value = None if row[12] is None else float(row[12]).hex()
        se = None if row[13] is None else float(row[13]).hex()
        out[("observation", *row[:12])] = (value, se, row[14], data)
    for row in db.execute(
        f"SELECT {key_cols}, co.requirements FROM completions co JOIN trajectories tr "
        "ON tr.tid = co.tid"
    ):
        out[("completion", *row)] = True
    return out


def assert_unchanged(before: dict[tuple[Any, ...], Any], after: dict[tuple[Any, ...], Any]) -> None:
    """Every record of ``before`` is in ``after`` with the same content."""
    missing = [k for k in before if k not in after]
    assert not missing, f"records disappeared: {missing[:3]}"
    changed = [k for k in before if before[k] != after[k]]
    assert not changed, f"records changed: {changed[:3]}"


def open_for_tampering(store: TableauStore) -> sqlite3.Connection:
    """A raw connection to the store's database with its append-only triggers dropped: what
    damage from outside SQLite (a disk, an editor, another program) can do. Tests use it to
    show that such changes are detected; the store itself never takes this route."""
    con = sqlite3.connect(store.path / STORE_FILE)
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")]
    for name in names:
        con.execute(f"DROP TRIGGER {name}")
    con.commit()
    return con


# ---------------------------------------------------------------------------------------------
# subprocesses
# ---------------------------------------------------------------------------------------------

SRC = Path(__file__).resolve().parents[1] / "src"
TESTS = Path(__file__).resolve().parent


def run_python(code: str, *, timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter of this environment (``src`` and ``tests`` on the path)."""
    prelude = f"import sys\nsys.path[:0] = [{str(SRC)!r}, {str(TESTS)!r}]\n"
    return subprocess.run(
        [sys.executable, "-c", prelude + code],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True)
