# Immutable graph-collection protocol

## Grid

The positive grid is

```text
geomspace(0.005, 1, 49), inclusive,
```

with `beta=0` prepended.  Every value is canonicalized once as

```text
beta_key = round(beta * 10^9),    beta_used = beta_key / 10^9.
```

The 50 integer keys are unique and strictly increasing.  Combined with seven
sizes and 1,000 indexed draws per cell, the exact count is

```text
7 * 50 * 1000 = 350,000.
```

## Seed derivation

Version `sha256_tuple_v1` derives a nonnegative signed-64-bit seed from the
ASCII tuple

```text
(derivation_version, master_seed, n, beta_key, graph_index).
```

The first eight bytes of the SHA-256 digest, masked to 63 bits, are the stored
seed.  This is a deterministic, tuple-keyed sampling rule: adding or reordering
other grid points cannot alter an existing draw.  The schema deliberately has
no uniqueness constraint on `graph_seed`; any collision is retained as a
separately indexed draw.

Graph and circuit randomness are separate.  Future circuit jobs must key their
random streams on immutable graph identity and the experiment protocol, not
reuse `graph_seed` as circuit randomness.

## Graph construction

Each row reconstructs through the public SparseGF2 Watts-Strogatz constructor
with `k=2`.  It starts from `C(n,2)` and independently rewires each lattice edge
with probability `beta_used`.  Rewiring moves rather than adds an edge, so all
realizations contain `2n` edges and have mean degree four.  Independent draws
need not be distinct topologies.  At `beta=0`, every seed necessarily produces
the same graph.

## Integrity requirements

A collection is marked complete only after all of the following pass:

1. all 350 cells are present;
2. every cell has graph indices `0` through `999` exactly once;
3. all 350,000 stored tuples exactly match the immutable derivation recipe;
4. the streamed seed-table SHA-256 matches the specification hash;
5. SQLite integrity and foreign-key checks pass;
6. representative realizations across every size reconstruct with sorted,
   unique, in-range edges and exactly `2n` edges;
7. multiple `beta=0` seeds reconstruct identical `C(n,2)` edge lists.

The seed-table content hash excludes SQLite row IDs, timestamps, and page
layout, so it is stable across an interrupted and resumed build.  The manifest
also records a database-file hash, generator-source hash, Python version, NumPy
version, and SQLite version.

The seed registry is only the first stage.  Edge banks and topology hashes are
materialized separately, then graph invariants and circuit experiments are
registered with their own versioned definitions and provenance.
