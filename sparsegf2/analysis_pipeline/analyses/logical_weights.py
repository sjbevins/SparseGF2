"""
``logical_weights`` analysis: minimum-weight logical operator per Pauli type.

For each sample, compute the minimum-weight logical operator of pure Pauli
type X, Y, or Z acting on the SYSTEM qubits of the emergent code produced
by the purification-picture simulation.

Definition (Aaronson-Gottesman 2004 Sec. IV; Fattal-Cubitt-Yamamoto-
Bravyi-Chuang 2004). For the 2n-qubit pure state (system + reference)
stabilized by 2n generators, the emergent code on system has:
  - code stabilizer group = {v on system : v (x) I_ref is in the full
    stabilizer group}. Equivalently, the symplectic orthogonal complement
    of the row-span of system parts of the 2n generators (see derivation
    below).
  - logical group = normalizer of code stabilizer / code stabilizer.

A system Pauli v is a LOGICAL operator iff:
  (a) v lies in the row-span of the 2n system-parts (i.e. v equals the
      system-part of some full-state stabilizer), AND
  (b) v does NOT commute with every one of the 2n full-state stabilizers
      (i.e. v is not itself a code stabilizer).

Reason: in the purification picture, v is a logical iff there exists a
non-identity reference Pauli P_R such that v (x) P_R is a full-state
stabilizer. Projecting to the system gives condition (a). The presence
of a non-trivial P_R is detected via condition (b): v (x) I_ref commutes
with every stabilizer iff v is in the code-stabilizer subgroup, so v
being a logical requires (b) to fail.

Algorithm: exhaustive subset enumeration up to
``max_exhaustive = min(max_exhaustive_param, d_cont_bound)``. For each |A|,
enumerate C(n, |A|) subsets and test conditions (a) and (b). Columns whose
search exhausts without finding a logical are filled with sentinel ``n + 1``.

EXPENSIVE: compute cost is C(n, d) checks, which grows combinatorially.
Default ``max_exhaustive_param = 4`` keeps cost tractable for n <= 128.
"""
from __future__ import annotations

import itertools
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sparsegf2 import __version__ as _SGF2_VERSION
from sparsegf2.analysis_pipeline.analyses._common import CellContext, CellRunResult
from sparsegf2.analysis_pipeline.rehydrate import rehydrate_sim
from sparsegf2.analysis_pipeline.registry import (
    make_entry, read_cell_registry, write_cell_registry,
)


NAME = "logical_weights"
OUTPUT_FILENAME = "logical_weights.parquet"
OUTPUT_KIND = "parquet"
CELL_SCOPE = True
EXPENSIVE = True
DEFAULT_PARAMS: Dict = {
    "max_exhaustive": 4,
    "use_d_cont_bound": True,   # cap search at d_cont when available
}


_SCHEMA = pa.schema([
    ("sample_seed", pa.int64()),
    ("d_logical_x", pa.int32()),
    ("d_logical_y", pa.int32()),
    ("d_logical_z", pa.int32()),
    ("d_logical_min", pa.int32()),              # min across X/Y/Z; equals d_min for pure-type codes
    ("search_depth", pa.int32()),               # |A| actually enumerated
    ("method", pa.string()),
    ("runtime_s", pa.float64()),
])


# Symplectic helpers

def _extract_symplectic(sim, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_bits, z_bits) of shape (2n, n) for the system qubits.

    Extracted from the PLT: bit layout is ``plt[r, q] = 2*X_bit + Z_bit``.
    """
    N = 2 * n
    plt = sim.plt  # uint8 [N, n]
    x = ((plt >> 1) & 1).astype(np.uint8)
    z = (plt & 1).astype(np.uint8)
    return x, z


def _sympl_inner_row(row_x: np.ndarray, row_z: np.ndarray,
                     v_x: np.ndarray, v_z: np.ndarray) -> int:
    """Symplectic inner product of two Paulis mod 2.

    ``<(row_x, row_z), (v_x, v_z)> = row_x @ v_z + row_z @ v_x  (mod 2)``.
    Returns 0 (commute) or 1 (anticommute).
    """
    return int((np.dot(row_x, v_z) + np.dot(row_z, v_x)) & 1)


def _pauli_vec_for_type_and_A(pauli: str, A: Sequence[int], n: int
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Build the symplectic (x, z) of a pure Pauli of given type on subset A."""
    x = np.zeros(n, dtype=np.uint8)
    z = np.zeros(n, dtype=np.uint8)
    if pauli == "X":
        x[list(A)] = 1
    elif pauli == "Z":
        z[list(A)] = 1
    elif pauli == "Y":
        x[list(A)] = 1
        z[list(A)] = 1
    else:
        raise ValueError(f"pauli must be 'X','Y','Z'; got {pauli!r}")
    return x, z


# Commutation + stabilizer-containment tests

def _commutes_with_all(stab_x: np.ndarray, stab_z: np.ndarray,
                       v_x: np.ndarray, v_z: np.ndarray) -> bool:
    """True iff v (x) I_ref commutes with every full-state stabilizer.

    stab_x, stab_z have shape (2n, n); each row is the system-part of one
    full-state stabilizer generator. For a pure-system Pauli v (no reference
    support), its commutator with a full stabilizer S = S_sys (x) S_ref is
    controlled solely by the system part: <v (x) I_ref, S> = <v, S_sys>. So
    this is computed with system-part inner products alone.

    Passing this test means v is in the CODE STABILIZER group (it is the
    system part of a stabilizer whose reference part is I_ref). FAILING this
    test is therefore the test-for-non-stabilizer-ness used in the logical
    operator check.
    """
    inner = (stab_x @ v_z + stab_z @ v_x) & 1
    return bool(np.all(inner == 0))


def _is_in_row_span(stab_x: np.ndarray, stab_z: np.ndarray,
                    v_x: np.ndarray, v_z: np.ndarray) -> bool:
    """True iff (v_x, v_z) is a GF(2) linear combination of the rows.

    Uses Gaussian elimination over GF(2) on the augmented matrix.
    """
    N = stab_x.shape[0]
    n = stab_x.shape[1]
    # Augmented matrix: rows are [x_i | z_i]; we ask if [v_x | v_z] is in the span.
    mat = np.concatenate([stab_x, stab_z], axis=1).astype(np.uint8)
    target = np.concatenate([v_x, v_z]).astype(np.uint8)
    # Work on a copy, reduce to RREF, check residual.
    M = mat.copy()
    cols = M.shape[1]
    rank = 0
    for col in range(cols):
        pivot = -1
        for r in range(rank, N):
            if M[r, col]:
                pivot = r
                break
        if pivot == -1:
            continue
        if pivot != rank:
            M[[rank, pivot]] = M[[pivot, rank]]
        for r in range(N):
            if r != rank and M[r, col]:
                M[r] ^= M[rank]
        # Apply same column-elimination to the target: if target[col] == 1 and
        # rank-th row has a 1 in column col, XOR rank-th row into target (after
        # we have reduced it). Actually we need a different reduction: reduce
        # target relative to the pivot rows.
        rank += 1
    # Now reduce target against the RREF rows.
    for r in range(rank):
        # Find leading 1 in row r
        leading = -1
        for c in range(cols):
            if M[r, c]:
                leading = c
                break
        if leading == -1:
            continue
        if target[leading]:
            target ^= M[r]
    return bool(np.all(target == 0))


# Min-weight type-restricted logical search

def _min_weight_logical_type(
    stab_x: np.ndarray, stab_z: np.ndarray,
    n: int, pauli: str, max_depth: int,
) -> int:
    """Return min |A| such that the pure-Pauli on A is a logical, or n+1 if none.

    A pure-Pauli v on subset A is a LOGICAL of the emergent code iff:
      (a) v is in the row-span of (stab_x | stab_z) -- the system-parts of the
          2n full-state stabilizer generators -- AND
      (b) v does NOT commute with every full-state stabilizer (so v (x) I_ref
          is not itself a code stabilizer).

    See module docstring for the derivation.
    """
    for d in range(1, max_depth + 1):
        for A in itertools.combinations(range(n), d):
            v_x, v_z = _pauli_vec_for_type_and_A(pauli, A, n)
            # (a) v must be the system-part of some full-state stabilizer.
            if not _is_in_row_span(stab_x, stab_z, v_x, v_z):
                continue
            # (b) v must not itself be a code stabilizer (i.e. v must fail
            # commutation with at least one full stabilizer's system-part).
            if _commutes_with_all(stab_x, stab_z, v_x, v_z):
                continue
            return d
    return n + 1


# Cell entry point

def run_cell(ctx: CellContext, params: dict, force: bool = False) -> CellRunResult:
    output = ctx.analysis_dir / OUTPUT_FILENAME
    if output.exists() and not force:
        return CellRunResult(
            name=NAME, status="existing", output_path=output,
            n_samples=ctx.n_samples, params=dict(params),
            message=f"{output.name} already exists (use --force {NAME})",
        )

    merged = {**DEFAULT_PARAMS, **params}
    max_exh = int(merged["max_exhaustive"])
    use_bound = bool(merged["use_d_cont_bound"])

    ctx.analysis_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # Optional d_cont bound: load distances.parquet if already present.
    # Use pyarrow-native column access (pyarrow is already a required dep
    # under [circuits]); avoid .to_pandas() so we don't impose pandas as a
    # transitive requirement just for this lookup.
    d_cont_by_seed: Dict[int, int] = {}
    if use_bound:
        dist_path = ctx.analysis_dir / "distances.parquet"
        if dist_path.exists():
            tbl = pq.read_table(dist_path, columns=["sample_seed", "d_cont"])
            for seed_val, d_val in zip(
                tbl.column("sample_seed").to_pylist(),
                tbl.column("d_cont").to_pylist(),
            ):
                d_cont_by_seed[int(seed_val)] = int(d_val)

    n = ctx.n
    data: Dict[str, list] = {name: [] for name in _SCHEMA.names}

    for i in range(ctx.n_samples):
        t_s = time.perf_counter()
        seed = int(ctx.seeds[i])
        sim = rehydrate_sim(n, ctx.x_stack[i], ctx.z_stack[i])
        k = int(sim.compute_k())

        if k == 0:
            data["sample_seed"].append(seed)
            data["d_logical_x"].append(0)
            data["d_logical_y"].append(0)
            data["d_logical_z"].append(0)
            data["d_logical_min"].append(0)
            data["search_depth"].append(0)
            data["method"].append("trivial (k=0)")
            data["runtime_s"].append(float(time.perf_counter() - t_s))
            continue

        depth_cap = max_exh
        if use_bound and seed in d_cont_by_seed:
            depth_cap = min(max_exh, d_cont_by_seed[seed])
        depth_cap = max(1, depth_cap)

        stab_x, stab_z = _extract_symplectic(sim, n)
        dx = _min_weight_logical_type(stab_x, stab_z, n, "X", depth_cap)
        dz = _min_weight_logical_type(stab_x, stab_z, n, "Z", depth_cap)
        dy = _min_weight_logical_type(stab_x, stab_z, n, "Y", depth_cap)

        data["sample_seed"].append(seed)
        data["d_logical_x"].append(int(dx))
        data["d_logical_y"].append(int(dy))
        data["d_logical_z"].append(int(dz))
        data["d_logical_min"].append(int(min(dx, dy, dz)))
        data["search_depth"].append(int(depth_cap))
        data["method"].append(f"exhaustive_leq_{depth_cap}")
        data["runtime_s"].append(float(time.perf_counter() - t_s))

    table = pa.Table.from_pydict(data, schema=_SCHEMA)
    pq.write_table(table, output)
    runtime = time.perf_counter() - t0

    entries = read_cell_registry(ctx.analysis_dir)
    entries[NAME] = make_entry(
        package_version=_SGF2_VERSION, params=merged,
        runtime_s=runtime, n_samples=ctx.n_samples,
    ).__dict__
    write_cell_registry(ctx.analysis_dir, entries)

    return CellRunResult(
        name=NAME, status="computed", output_path=output,
        runtime_s=runtime, n_samples=ctx.n_samples, params=merged,
    )
