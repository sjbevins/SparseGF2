# Generalized raw single-reference purification-time protocol

This document fixes the meaning, indexing, storage, and execution rules for the
raw single-reference campaign. It does not choose the production parameter
values and does not authorize a launch. The choices listed in
[Prelaunch decisions](#prelaunch-decisions) must be resolved and fingerprinted
before production data are generated.

The companion [FLOWCHART.md](FLOWCHART.md) is the one-page process map. The
immutable identity objects are implemented in
[`sweep_spec.py`](../../sweep_spec.py); the graph adapters, trajectory kernel,
resumable storage, strict configuration loader, and guarded process coordinator
are in this directory.

## Work hierarchy and generalized graph grid

Let the system-size set be \(\mathcal N\). A graph family has zero or more named
parameter axes

\[
\mathcal A_1,\mathcal A_2,\ldots,\mathcal A_r .
\]

The graph cells are the complete Cartesian product

\[
\mathcal C = \mathcal N \times \mathcal A_1 \times \cdots \times \mathcal A_r.
\]

Thus every size is paired with every possible combination of the supplied
graph-parameter values. A family with no parameter axes has one cell per size.
Each cell contains `N_graphs = graphs_per_cell` indexed graph realizations. The
ordering of sizes, parameter axes, or values in the input does not alter the
canonical collection identity. Parameter types remain distinct, so the integer
`1` and floating-point `1.0` are not silently identified. Axis entries are
JSON scalar values (`null`, Boolean, integer, finite float, or string);
structured graph choices must be represented by separate named axes.

Two graph-source paths implement this abstraction:

- `WattsStrogatzRegistryProvider` adapts the sealed 350,000-draw
  Watts-Strogatz registry. Its stored graph seeds remain authoritative. Before
  reconstructing edges, the adapter validates the manifest, generator source,
  collection identity, canonical seed-table digest, exact cell keys, canonical
  graph indices, size grid, beta grid, and graph count in one ordered database
  scan. That scan builds a process-local immutable seed tuple for every
  `(n, beta_key)` cell; later `graph_seeds()` calls use this sealed snapshot and
  never re-query mutable SQLite. This is not a permanent database lock: a new
  provider instance must repeat and pass the complete validation.
- `GridGraphProvider` creates a new Cartesian collection. It derives a stable
  graph seed for every `(cell, graph_index)` and passes `(n, parameters,
  graph_seed)` to a graph factory. The strict JSON configuration supports the
  built-in `watts_strogatz`, `newman_watts`, `cycle`, `path`, `complete`, and
  `lattice_2d` factories, whose implementation source is included in the sweep
  fingerprint. The optional `GridGraphProvider(factory=...)` interface is an
  extension and test hook, not a production path. A custom factory must not be
  used for production until its exact source digest and generator identity are
  bound into the collection and experiment fingerprints.

Graph seeds label separately constructed graph records; neither the seed values
nor the realized graphs are required to be pairwise distinct. Graphs may have
different numbers of edges. For each cell, the workflow materializes a
validated, CSR-like edge bank containing `graph_seed`, `edge_offsets`, and one
canonical edge array. Self-loops, duplicate undirected edges, out-of-range
endpoints, and empty edge sets are rejected.

## Scientific parameters

| Parameter | Definition and constraint |
|---|---|
| `N_circuits` (`n_circuits`) | Number of independent circuit trajectories for every `(graph, p)` pair; positive integer. |
| `q_scramble` | Scramble-gate multiplier. The initialization applies exactly `q_scramble*n` two-qubit gates; nonnegative integer. |
| `q_max` | Depth-cap multiplier. The monitored evolution ends at `T_max = q_max*n`; positive integer. |
| `p_min`, `p_max`, `delta_p` | Exact inclusive measurement-rate grid with `0 <= p_min <= p_max <= 1`, `delta_p > 0`, and `(p_max-p_min)/delta_p` an integer. |
| protocol `master_seed` | Root of all named trajectory random streams; nonnegative integer. |
| `reference_system_qubit_policy` | `fixed_last` uses system qubit `n-1`; `uniform_system_qubit_per_circuit` samples one system qubit uniformly for each trajectory. |
| `p_randomness_policy` | Production currently supports `independent`: every `p` has independent circuit and measurement streams. `common_circuit_disorder` is reserved and the engine rejects it until layer-indexed common randomness is implemented. |

The number of measurement rates is exact:

\[
N_p = 1 + \frac{p_{\max}-p_{\min}}{\Delta p},
\]

and the complete planned trajectory count is

\[
N_{\mathrm{traj}} = |\mathcal C|\,N_{\mathrm{graphs}}\,N_p\,N_{\mathrm{circuits}}.
\]

`ProbabilityGrid` constructs the grid with decimal arithmetic and stores a
canonical decimal string for every point. Binary floating-point accumulation
therefore cannot omit an endpoint or alter experiment identity.

## Scientific environment contract

The resolved sweep pins a canonical `environment_contract` containing the
exact `python`, `numpy`, and `numba` versions and `bit_generator = "PCG64"`.
Its SHA-256 is part of the sweep and experiment identity. The scientific
manifest stores the payload and digest; every raw NPZ and checkpoint journal
stores the same canonical JSON and digest. Each spawned worker reconstructs the
current contract and refuses work if it differs from the experiment contract.
This prevents an interpreter, dependency, or NumPy random-generator change
from silently resuming the same raw-data experiment.

Operating system and CPU details, worker count, checkpoint cadence, and
in-flight task limit do not define the physics. They remain runtime audit data
and may change on a reviewed resume without changing the experiment identity or
trajectory seeds.

## One trajectory

One trajectory is identified physically by `(protocol master seed, graph cell,
p, graph_index, circuit_index)`. Its random choices come from separate
deterministic streams for reference placement, scramble pairs, scramble
Cliffords, dynamic edges, dynamic Cliffords, measurement masks, and measurement
outcomes. Seeds do not depend on the experiment's display name, worker number,
scheduling order, checkpoint boundaries, `p_index`, or unrelated points in the
complete `p` grid. Renaming an experiment or adding new `p` points therefore
changes the experiment artifact identity and output path but leaves every
preexisting physical trajectory stream unchanged.

### Initialization and scrambling at `t = 0`

The simulator contains system qubits `0,...,n-1` and one reference qubit `n`.
It prepares a Bell pair between the reference and the system qubit selected by
`reference_system_qubit_policy`; all other system qubits begin unentangled. The
reference has `S(R)=1` and is never acted on again.

The system-only global scramble is the composite of exactly `q_scramble*n`
two-qubit Clifford gates. For every gate, an ordered pair of distinct system
qubits is sampled uniformly; pairs are sampled independently with replacement.
The gate is sampled uniformly and independently with replacement from the 720
phase-free representatives of `Sp(4,F_2)`. The kernel verifies that system-only
scrambling leaves `S(R)=1`.

### One monitored layer

For integer layers `t = 1,...,T_max`, where `T_max = q_max*n`:

1. Sample `m = floor(n/2)` edges independently and uniformly with replacement
   from the fixed edge set of the indexed graph.
2. For each sampled edge, independently sample with replacement one of the 720
   phase-free two-qubit Cliffords and apply it to that edge's endpoints. A qubit
   may participate in more than one gate in a layer because edge sampling is
   with replacement and does not form a matching.
3. For each of the `n` system qubits, independently perform a `Z` measurement
   with probability `p`. If `M_t` is the number measured in that layer, then

   \[
   M_t\sim\operatorname{Binomial}(n,p),\qquad
   \mathbb E[M_t]=np,\qquad
   \Pr(M_t\ge 1)=1-(1-p)^n.
   \]

   The expected gate-to-measurement count is
   `floor(n/2) : n*p`; for even `n` this is `1 : 2p` after normalizing by the
   number of gates.
4. After the entire gate and measurement layer, compute the single-qubit
   entropy `S(R)`, which is exactly zero or one for this stabilizer state.

### First passage and censoring

The stored purification time is

\[
\tau_p=\min\{t\in\{1,\ldots,T_{\max}\}:S(R;t)=0\}.
\]

If the event occurs, the trajectory stops immediately and stores
`tau_p = stop_layer = t` with `event_observed = 1`. If `S(R)=1` through the
complete layer `T_max`, the trajectory is right-censored: it stores
`tau_p = -1`, `stop_layer = T_max`, and `event_observed = 0`. The value `-2` is
reserved for an incomplete trajectory and is never treated as a physical
result. Raw production output stores first-passage and censoring fields, not a
full entropy trace.

## Identity, shards, and resume behavior

Canonical JSON and SHA-256 fingerprints bind the graph collection, graph cell,
protocol, source files that determine trajectory semantics, experiment, and
each `(cell, p)` work unit. Any change to a scientific parameter or fingerprint
creates a different experiment path rather than silently resuming incompatible
data. Runtime-only worker and checkpoint settings are deliberately excluded
from the scientific fingerprint.

Each graph cell has one immutable edge-bank artifact:

```text
DATA_ROOT/single_ref/raw_tau/edge_banks/
  COLLECTION_ID/contract_GENERATORHASH/nN/cell_CELLINDEX_CELLHASH.npz
  COLLECTION_ID/contract_GENERATORHASH/nN/cell_CELLINDEX_CELLHASH.npz.validated.npz
```

The coordinator performs full identity, seed, offset, digest, and canonical-edge
validation, and binds the path and metadata to the exact generator-code
contract. It then atomically publishes the adjacent validation receipt with
the bank's whole-file SHA-256 and byte size. Workers verify the immutable bank
against that receipt when loading it. The receipt avoids repeating the full
per-graph canonical-edge audit at every raw-shard checkpoint without weakening
the file-integrity check.

Each `(graph cell, p)` work unit has one raw result shard:

```text
DATA_ROOT/single_ref/raw_tau/
  EXPERIMENT_SHA256/CELL_SHA256/pPINDEX_WORKHASH.npz
  EXPERIMENT_SHA256/CELL_SHA256/pPINDEX_WORKHASH.npz.checkpoint.sqlite3
```

The science arrays in a raw shard have shape
`[N_graphs, N_circuits]`. They include `tau_p`, `stop_layer`,
`event_observed`, `complete`, and `reference_system_qubit`; indexed graph seeds
and all identity metadata are stored beside them. This preserves per-graph and
per-circuit statistics while avoiding one file per trajectory.

A work unit has exactly one writer. Until completion, it owns the adjacent
SQLite checkpoint journal rather than repeatedly rewriting a growing NPZ. The
journal's canonical metadata pins its journal and raw schema versions, engine,
source, experiment, protocol, collection, cell, work unit, graph parameters,
`n`, `p`, `q` values, raw shape, randomness policies, and validated edge-bank
SHA-256. Opening a journal runs SQLite `quick_check` and requires an exact
metadata match.

Committed result rows have primary key `(graph_index, circuit_index)` and store
`tau_p`, `stop_layer`, `event_observed`, and `reference_system_qubit`. Resume
replays them in canonical order, validates their bounds and event/censoring
semantics, reconstructs the full in-memory arrays, and then performs the same
whole-shard semantic validation used for final output. Only absent rows are
simulated. For each finished trajectory, all in-memory result fields are
written before `complete` is set to one.

The journal uses SQLite `DELETE` journal mode and `synchronous=FULL`.
Transactions commit every `checkpoint_every` newly simulated trajectories and
commit the remaining rows on normal shutdown. An interruption can therefore
lose only the current uncommitted batch, whose deterministic trajectories are
rerun on resume. No partial NPZ is published.

Once every row is present, the writer closes SQLite, writes the deterministic
NPZ through a temporary sibling and atomic replacement, reloads and
semantic-validates it, and compares every published field with the in-memory
arrays. Only then is the checkpoint journal removed, using bounded Windows
retries. If interruption occurs after NPZ publication but before cleanup, the
next run accepts removal only when the remaining journal is complete and every
replayed field exactly matches the final shard. Interrupted and uninterrupted
execution must therefore produce byte-identical completed NPZ artifacts.

The workflow also computes a canonical logical-result SHA-256 over the exact
cell and `p` identity, graph and circuit indices, graph seeds, first-passage and
censoring arrays, completion flags, and reference sites. Integer arrays are
hashed in explicit little-endian form. This digest compares scientific content
independently of ZIP compression or host byte order; the separate whole-file
SHA-256 remains the provenance check for the physical NPZ container.

The coordinator also writes run-level records under

```text
DATA_ROOT/single_ref/raw_tau/runs/EXPERIMENT_ID/
  manifest.json
  status.json
  STATUS.md
  runtime_history/TIMESTAMP_EVENT_INVOCATION.json
```

The immutable manifest records only the resolved scientific identities,
environment contract, protocol, and bounded work plan. Every invocation
appends separate `started` and terminal (`complete`, `failed`, or `interrupted`)
audit records containing the worker count, checkpoint cadence, in-flight bound,
configuration path, process ID, Python version, platform, timestamp, and any
error. Execution records also include the NumPy, Numba, and SparseGF2 versions.
Execution settings can therefore change without pretending that the physics
changed or silently erasing prior settings. Status files are atomically
refreshed after every completed `(cell, p)` shard. The coordinator holds a
nonblocking operating system lock on
`single_ref/raw_tau/runtime/EXPERIMENT_ID.lock` for the entire run, rejecting a
second runner for the same experiment. The file also records the lock ID,
process ID, acquisition and release times, and `locked` or `released` state.
Normal exit marks it released; after a crash the operating system releases
ownership automatically, so a later invocation can safely overwrite stale
audit content without deleting another process's live lock.

Before workers start, the coordinator transactionally registers the complete
experiment, canonical graph cells, and every planned `(cell, p)` work unit in

```text
DATA_ROOT/single_ref/raw_tau/catalog.sqlite3
```

This strict, versioned SQLite catalog records the array layout joining axis 0
to `graph_index` and axis 1 to `circuit_index`. After a worker returns a
complete shard, the coordinator independently verifies its canonical path,
container SHA-256, and logical-result SHA-256 before marking that work unit
complete. The coordinator is the catalog's only writer; workers never open it.
Registration is idempotent on resume and immutable conflicts are errors.

Completed raw NPZ shards remain the authoritative trajectory data; the catalog
is their searchable status and provenance index, and checkpoint journals are
transient exclusive-resume state. Writing results into the separate shared
graph-registry SQLite database is intentionally deferred. A future generic
graph-registry v2 can reference these artifacts without changing raw
trajectory semantics; the sealed Watts-Strogatz registry remains read-only.

## Multiprocessing and benchmark requirements

Trajectories are CPU-bound, so the implemented coordinator uses a spawned
process pool rather than a Python thread pool. One task exclusively owns one
complete `(cell, p)` shard and advances its pending trajectories sequentially;
different `p` shards run in parallel. The coordinator alternates low- and
high-`p` tasks to reduce a final tail of long capped trajectories, bounds queued
tasks by `max_in_flight`, and completes every `p` shard in a cell before moving
to the next graph cell. Workers read immutable edge banks and registry metadata
but never write the shared graph registry.

Every spawned process must import NumPy and Numba under these variables set to
`1`, preventing nested numerical-library threads from multiplying the requested
worker count:

```text
NUMBA_NUM_THREADS
OMP_NUM_THREADS
OPENBLAS_NUM_THREADS
MKL_NUM_THREADS
VECLIB_MAXIMUM_THREADS
NUMEXPR_NUM_THREADS
```

The bounded benchmark launcher sets them before Python starts. The production
coordinator sets them before creating its Windows `spawn` pool, and each worker
verifies the inherited values, calls `numba.set_num_threads(1)`, and initializes
the immutable 720-gate lookup table. The benchmark additionally runs untimed
warm-up trajectories before steady measurements. The hot path uses batched
two-qubit and measurement kernels with the hybrid tableau enabled within each
single-threaded worker. Scalar and sparse-only paths remain equivalence and
benchmark controls rather than production configuration choices.

The bounded benchmark in [`../BENCHMARKS.md`](../BENCHMARKS.md) measures stage
costs and strong scaling. It must be rerun with representative smallest,
middle, and largest production sizes; local, crossover, and rewired graph
cells; and `p=0`, a transition-region rate, and `p=1`. Test worker counts
`1, 2, 4, 8, 12, 16`, report cold startup separately, and choose the smallest
worker count on the steady-throughput plateau. Do not assume that 32 logical
processors outperform the 16 physical cores. Keep no more than `max_in_flight`
tasks queued, with `max_in_flight >= workers`, and preserve deterministic output
when worker count or scheduling order changes.

## Planning, launch, interruption, and resume

The strict JSON configuration has five top-level keys:
`schema_version`, `name`, `graph_source`, `protocol`, and `runtime`.
`graph_source.kind` is either `ws_registry`, with a manifest path, or
`cartesian_builtin`, with sizes, named parameter arrays, graph count, generator
identity, and graph master seed. `protocol` contains the scientific parameters
defined above. `runtime` contains `data_root`, `workers`, `checkpoint_every`,
and optional `max_in_flight`.

The command-line default is a write-free exact plan:

```powershell
studies\prl_production\run_raw_tau.ps1 `
  -Config PATH_TO_REVIEWED_CONFIG.json
```

The wrapper installs the numerical thread limits before starting the repository
interpreter. It prints the resolved experiment ID and exact counts of cells,
`p` values, work units, trajectories, maximum layers, dynamic and scramble
gates, measurement trials, and raw `tau_p` bytes. The same plan envelope shows
every exact decimal `p`, the canonical protocol and its digest, collection and
source digests, and the readable scientific-environment contract plus digest.
Review this output and the benchmark before allowing writes. The byte count is
the uncompressed four-byte `tau_p` array only; auxiliary arrays, metadata,
checksums, edge banks, catalog, and NPZ container overhead must be added to a
disk-capacity estimate. Execution requires both `-Run` and a confirmation that
exactly matches the resolved experiment ID:

```powershell
studies\prl_production\run_raw_tau.ps1 `
  -Config PATH_TO_REVIEWED_CONFIG.json -Run `
  -ConfirmExperimentId EXACT_ID_FROM_PLAN
```

The runner creates or validates each cell edge bank, runs all of that cell's
`p` shards, and then advances to the next canonical cell. On a worker failure it
terminates the remaining pool and marks the run `failed`; `Ctrl+C` marks it
`interrupted`. Committed journal checkpoints and finalized NPZ shards remain
valid. Repeating the same confirmed experiment validates them and computes only
rows still incomplete. A different worker count or checkpoint cadence is an
audited runtime attempt, not a new scientific experiment. A conflicting
scientific run manifest or active same-experiment lock is a hard error rather
than an implicit overwrite.

## Prelaunch decisions

No production simulation should start until all of the following are recorded
in a reviewed strict JSON configuration and its resolved fingerprint:

- graph source: the sealed Watts-Strogatz registry or a new Cartesian
  collection using one of the six fingerprinted built-in factories;
- sizes, every graph-parameter axis and value, and `N_graphs` for a new
  collection;
- `N_circuits`, `q_scramble`, and `q_max`;
- exact `p_min`, `p_max`, and `delta_p` strings;
- graph-collection and protocol master seeds where a new value is required;
- fixed-last or uniformly sampled reference-system-qubit placement;
- confirmation that independent randomness across `p` is the intended current
  policy;
- review of the pinned Python, NumPy, Numba, and PCG64 scientific environment
  contract printed by the resolved configuration;
- output `data_root`, `checkpoint_every`, and `max_in_flight`;
- batch/hybrid production execution after scalar, batch, sparse, and hybrid
  equivalence tests pass; and
- worker count selected from bounded production-representative scaling
  benchmarks, plus the amount of CPU capacity to leave available for other
  work; and
- verification that runtime-attempt auditing permits a deliberate worker or
  checkpoint change on resume without changing any trajectory seed.

Before launch, validate the complete graph source, run the focused correctness
and resume tests, perform the write-free work/storage plan, and preserve the
reviewed configuration and benchmark JSON. During execution the coordinator
fully validates and seals each cell's edge bank before submitting that cell's
trajectory shards.
