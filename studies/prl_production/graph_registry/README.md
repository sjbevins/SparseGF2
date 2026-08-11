# Paper graph registry

This directory defines the reusable graph collection for the paper.  The
collection contains 1,000 independently indexed Watts-Strogatz draws for every
pair in

```text
n = 64, 96, 128, 160, 192, 224, 256
beta = 0 plus 49 log-spaced values from 0.005 through 1
```

There are 350 `(n, beta)` cells and 350,000 graph records.  Every graph is a
rewiring of `C(n,2)`, so every realization has exactly `2n` edges and mean
degree four.

## Why one normalized database

The registry is one SQLite catalog, not 350,000 small database files.  A graph
draw has the immutable identity

```text
(collection_id, n, beta_key, graph_index)
```

and stores an explicit `graph_seed`.  `graph_seed` is not unique.  Equal seeds
and equal realized topologies remain separate indexed draws.  In particular,
all 1,000 records at `beta=0` reconstruct the same unrewired `C(n,2)` geometry,
as they must, while remaining separate observations for circuit randomness.

The ordered query `graphs_for_cell(...)` returns the requested vector

```text
(G_0, G_1, ..., G_999).
```

For one returned graph, `graph_seed` is the first part of
`G_i = (graph_seed, DATABASE)`.  The expandable `DATABASE` part is represented
by child rows and immutable artifact references.  `graph_snapshot(graph_id)`
collects that logical per-graph view without duplicating a mutable JSON blob.

```python
from studies.prl_production.graph_registry import GraphCollection

collection = GraphCollection(database_path, collection_id)
graphs = collection.cell(n=128, beta=0.01)
G_i = graphs[i]
graph_seed = G_i.graph_seed
DATABASE = collection.snapshot(G_i)
```

The schema currently provides:

- versioned graph-invariant definitions and graph-invariant results;
- versioned experiment definitions and graph-scoped results;
- content-addressed references to large arrays and traces;
- typed statuses, foreign keys, conflict detection, and indexed cell queries;
- an idempotent resume path that inserts missing rows but rejects altered rows.

Scalar invariants and compact scalar results belong in SQLite.  Edge banks,
spectra, full depth traces, and other arrays belong in batched artifacts linked
by relative path and SHA-256.  This keeps the catalog queryable while avoiding
one high-overhead file per graph.

## Build and validate

Use only the repository interpreter.  A production build must be confirmed
explicitly:

```powershell
.venv\Scripts\python.exe -m studies.prl_production.graph_registry.build `
  --profile production --confirm-production
```

The command is single-process, transactional, and resume-safe.  It does not
materialize graph edges or compute expensive invariants; it creates and fully
validates the immutable seed collection.  Generated output is stored below
`studies/prl_production/data/graph_registry/`, which is ignored by Git.  The
collection-local `manifest.json` records exact grids, hashes, source identity,
and environment versions.  [STATUS.md](STATUS.md) mirrors the latest validated
production status.

Useful bounded checks are:

```powershell
.venv\Scripts\python.exe -m studies.prl_production.graph_registry.build `
  --profile production --dry-run
.venv\Scripts\python.exe -m studies.prl_production.graph_registry.build `
  --profile production --validate-only
```

The database uses WAL, full synchronous durability, foreign keys, strict
tables, and a bounded busy timeout.  One coordinator owns writes.  Simulation
workers should return result batches to that coordinator or publish immutable
artifacts; they should not write SQLite independently.

## Reconstruct a graph

The public reconstruction contract is

```python
from sparsegf2.circuits.graphs import watts_strogatz

topology = watts_strogatz(
    graph.n,
    k=2,
    beta=graph.beta_key / 1_000_000_000,
    seed=graph.graph_seed,
)
```

The canonical integer `beta_key` is the identity; display-rounded floating
point strings are never used for joins.  Before simulations or invariant jobs
consume a cell, the next production stage should materialize and verify one
batched edge artifact for that cell, retaining both seeds and topology hashes.

Exact-distance and MILP tooling is intentionally outside this registry work.

## Realized rewired-edge invariant

The first complete graph invariant is documented in
[REWIRED_EDGES.md](REWIRED_EDGES.md).  It measures the final number of
off-lattice edges,

```text
N_rew(G) = |E(G) - E(C(n,2))|,
```

for all 350,000 indexed draws.  This is distinct from the number of accepted
construction operations because a later rewiring can restore a previously
removed lattice edge.  Reproduce or exactly resume the calculation with two
BelowNormal workers using

```powershell
.venv\Scripts\python.exe -m studies.prl_production.graph_registry.analyze_rewiring `
  --profile production --workers 2 --confirm-production
```

The calculation checkpoints one small file per `(n, beta)` cell, writes the
per-graph invariant into SQLite through one coordinator, publishes a 350-row
mean-and-SEM summary, and creates both the mean curves and conditional
histograms of the complete per-graph distributions. The histogram overview and
seven detailed size-resolved plots use exact integer edge-count bins on the
normalized ordinate `N_rew/(2n)`. Positive `beta` is logarithmic and `beta=0`
is joined through conventional diagonal axis-break marks. Every size has a
one-edge zoom showing the empirical mean, its raw value at `beta_min`, the
nominal `f=beta` theory, and the level `N_rew=1`.

## Algebraic connectivity and cumulative normalized gain

The combinatorial-Laplacian algebraic connectivity is stored for every one of
the 350,000 registered graphs. Reproduce or exactly resume it with

```powershell
.venv\Scripts\python.exe -m studies.prl_production.graph_registry.analyze_connectivity `
  --profile production --workers 8 --confirm-production
```

The analysis first averages `lambda_2` over the 1,000 graphs in each cell, then
normalizes by the exact unrewired `C(n,2)` gap and takes the geometric mean over
the requested cumulative sets `[64]`, `[64,128]`, `[64,128,192]`, and
`[64,128,192,256]`. [ALGEBRAIC_CONNECTIVITY.md](ALGEBRAIC_CONNECTIVITY.md)
contains the complete estimator definition, convergence table, figure, and
artifact paths. The current four-set data show decreasing relative increments
but do not converge to a size-independent unscaled master curve.
