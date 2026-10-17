# The tableau store and the experiment runner: records, transactions, retrieval and reproducibility

How the Watts–Strogatz study keeps every trajectory it simulates: the record hierarchy of v1.4 §8
as SQLite tables, every saved phase-free stabilizer tableau retrievable by its parameter tuple,
writes that an interruption cannot leave half done, campaigns that resume and extend without
changing a stored value, the replay that reproduces any stored or unsaved depth, and the
observables computed later from the stored snapshots without rerunning the study.

This document is written for a reader fluent in the stabilizer formalism and for language
models. Every symbol is defined where it first appears, every formula is the one the code
computes, and every statement about the code names the function it describes as
`path::function` (paths relative to `src/sparsegf2/`). The Python blocks run as written, in
order, in one namespace, in a few seconds once the kernels are compiled.

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Notation](#2-notation)
3. [Definitions](#3-definitions)
4. [Propositions and proofs](#4-propositions-and-proofs)
5. [Algorithms](#5-algorithms)
6. [From the mathematics to the code](#6-from-the-mathematics-to-the-code)
7. [Worked examples](#7-worked-examples)
8. [Validation map](#8-validation-map)
9. [Limitations and departures](#9-limitations-and-departures)

---

## 1. Purpose and scope

The owner's requirement: *"I need to be able to retrieve every saved phase-free stabilizer
tableau using its simulation-parameter tuple, and then compute additional observables later
without rerunning the entire study."* With it: writes that are safe against interruption,
datasets that are never loaded whole into memory, and campaigns that resume and extend.

This part implements:

* `study/store.py::TableauStore`: one SQLite store per experiment holding every manifest
  revision, the graph records with four graph measurements, every trajectory with its per-layer
  $k_t$ trace and exact first passages, the **checkpoint tableau dictionary**
  $\mathcal D_\xi = \lbrace t \mapsto \widetilde{\mathsf S}_t : t \in \mathcal T \rbrace$ of every
  trajectory $\xi$ (the canonical joint row space and the physical code at each checkpoint),
  and every observable value, each under its full identity; retrieval by the parameter tuple,
  integrity checks, a logical digest and an export;
* `study/runner.py`: the work units still to do (`plan`), their execution through the native
  engine with every record of a unit written in one transaction (`run`), revisions (`extend`),
  the replay of one trajectory (`replay`, `replay_signed` with its recorded outcome seed,
  `circuit`), retrospective observables (`compute`), verification (`verify`), aggregate records
  (`TableauStore.aggregate`) and the graph-side diagnostics of the correlated samplers
  (`diagnose_samplers`);
* the protocol dimension of design §12 (protocol F, the full reference of v1.4; protocol P, the
  pure product state of the queued MIPT study v0.3) in the trajectory key, the manifest
  (`study/manifest.py::ProtocolSettings`), the runner and the retrieval.

It implements v1.4 §1 (the record hierarchy), §5.1–§5.3 (the phase-free tableau dictionary
keyed by the trajectory identity, the $k_t$ trace at every layer, the first passages with their
censoring, observables recomputed from stored tableaux), §7.2 (the identity stored with every
trajectory), §7.3 (prefix, checkpoint and probe-stream invariance, used and tested), §7.4 (the
deterministic acceptance checks for replay, chunking and threads), §8 (every record type) and
§8.1 (the joint snapshot extension in production), the §9 rows "Storage", "Purification
stopping" and "Checkpoints", and §9.2 (c), (d), (g), (h). It changes no probability law, no
random stream and no observable definition: every trajectory is simulated by
`mipt.MonitoredCircuit.run`, and every value by the functions of `study/observables.py`.

## 2. Notation

| Symbol | Meaning | Type | Code |
|---|---|---|---|
| $\pi$ | protocol: F (full reference, v1.4) or P (pure product state, design §12) | `"F"` or `"P"` | `protocol` |
| $n$ | system qubits; $N = 2n$ (F) or $N = n$ (P) qubits in a snapshot | int | `n`, `num_qubits` |
| $\beta$, $p$ | rewiring probability, measurement rate, as exact tokens | `Token` | `beta`, `p` (canonical strings) |
| $g$, $c$, $r$ | graph label, circuit label, replica label ($r = 0$, design D1) | int | `g`, `c`, `r` |
| $\xi$ | trajectory key $(\text{experiment}, \pi, n, \beta, p, g, c, r)$ | `TrajectoryKey` | `key` |
| $s$ | circuit index stride (frozen); $u = g s + c$ the engine's trajectory index | int | `stride`, `u` |
| $G_*$, $C$ | graphs per condition, circuits per graph | int | `trials.graphs`, `trials.circuits_per_graph` |
| $T$ | horizon $T_{\max}(n)$ of a revision | int | `layers` |
| $\mathcal T$, $E_r$ | checkpoint schedule; the effective schedule of revision $r$ (ruling R-CK) | sorted ints | `checkpoints`, `effective_checkpoints` |
| $\widetilde{\mathsf S}_t$ | joint phase-free stabilizer row space after $t$ layers | `PhaseFreeTableau` | `joint` |
| $W_t^{\mathrm{phys}}$ | physical stabilizer row space (no reference support), rank $r_t$ | `PhaseFreeTableau` | `phys` |
| $k_t$ | logical qubits, $k_t = n - r_t$ | int | `k` |
| $\tau_\ell$ | exact first passage $\min\lbrace t : k_t \le \ell\rbrace$, $-1$ when censored | int | `tau` |
| $\mathcal D_\xi$ | the tableau dictionary of $\xi$ | rows of `snapshots` | `store.depths`, `store.snapshot` |
| $R$, $\rho(R)$ | requirements of a condition and their SHA-256 | dict, hex | `runner.requirements`, `requirements_digest` |
| $U$ | work unit: $\pi, n, \beta, p$, graphs $[g_0, g_1)$, circuits $[c_0, c_1)$, $T$ | `WorkUnit` | `runner.WorkUnit` |
| $H(\cdot)$ | SHA-256, as 64 hexadecimal digits | str | `identity.sha256_hex` |
| $z(\cdot)$ | zlib compression (level `storage.compression_level`) | bytes | `zlib.compress` |
| $\Lambda$ | logical digest of a store | hex | `TableauStore.logical_digest` |
| $q_G$, $\bar d_G$, $\lambda_2$, $\mathrm{Var}(\deg)$ | graph measurements (design §12) | float | `store.graph_measures` |

## 3. Definitions

### 3.1 The record hierarchy of v1.4 §8 as tables

| v1.4 §8 record | Primary key in v1.4 | Where it lives | Natural key in the store |
|---|---|---|---|
| `experiment` | `experiment_id` | `meta` (versions) and `manifests` (every revision's canonical JSON and digest) | `meta.experiment_id`; `manifests.revision`, `manifests.digest` |
| `graph` | $(n, \beta, g)$ | `graphs` | `(n, beta, g)` |
| `gate_sequence` | $(n, \beta, g, c[, p])$ | `trajectories` (run seed, $u$, graph digest); the gates themselves on demand (`runner.circuit`) | `(protocol, n, beta, p, g, c, r)` |
| `trajectory` | $(n, \beta, g, c, p, r)$ | `trajectories` | `(protocol, n, beta, p, g, c, r)` |
| `layer` | $(\ldots, t)$ | not stored: `runner.circuit(store, key)` exports the ordered gates and measured sites of every layer (§3.9) | — |
| `snapshot` | $(\ldots, t)$ | `snapshots` | `(tid, t)` |
| `purification_trace` | (trajectory key, dynamics version) | `traces`, one row per horizon | `(tid, layers)` |
| `observable_snapshot` | (snapshot key, id, version) | `observations` with $t \ge 0$ | `(tid, t, observable, version, params, context)` |
| `observable_trajectory` | (trajectory key, id, version, params) | `observations` with $t = -1$ | same, `context` = requirements digest |
| `region_sample` | (trajectory key, $t$, $m$, trial) | not stored (§9); the samplers' graph-side diagnostics are in `sampler_diagnostics` | `(n, beta, g, probe, config, m, coupling)` |
| `geometry_erasure_profile` | (trajectory key, $t$, $J$, version) | `observations` `correlated_failure` and `D_J` | as `observable_snapshot` |
| `aggregate` | ($n, \beta, p$, id, ...) | `aggregates` (`TableauStore.aggregate`), and `aggregates.csv` of the export | `(protocol, n, beta, p, observable, version, params, selection, t, estimator, policy, sample_set)` |
| (outcome-aware replay) | — | `signed_runs`: outcomes, random flags and signed digests of `runner.replay_signed(..., record=True)` | `(tid, outcome_seed, layers, depths)` |
| `rng_validation` | (experiment, protocol, version) | not this part (the demonstration task) | — |

`tid` is the row id of a trajectory; every child row refers to it, and the natural key of the
trajectory is unique, so `(tid, t)` and the natural key `(protocol, n, beta, p, g, c, r, t)`
name the same snapshot.

### 3.2 Identities and the never-overwrite rule

Every identity of §3.1 is stored at most once. Meeting it again (a resumed unit, a rerun for a
deeper horizon, a recomputation) **compares** the content with what is stored:

* a trajectory: $u$, the rate's `float.hex`, the run seed, the graph digest, the dynamics
  version;
* a trace at the same horizon: the decoded $k$ trace, $\tau$, the final layer, the status and
  the measurement count; at another horizon: the shorter trace must be a prefix of the longer
  one, and a stop at purification must happen at the same layer;
* a snapshot: status, $N$, $k$, the physical digest and the joint digest;
* an observation: the value and its error bit for bit (NaN equal to NaN), the method and the
  decoded data; for a retrospective recomputation (`runner.compute`) the value and error only.

Equal content is a no-op; any difference raises `ValueError` naming the identity, and the
transaction rolls back. The only `UPDATE` statements of the store are the finish time of the
unit being written and one fill-in: a snapshot stored with its physical code only (status
`phys_only`: a revision had `tableau_snapshots = false`) receives its joint row space and the
status `stored` when a later revision switches the joint snapshots on, after the physical code
of the new joint state is found equal to the stored one. SQLite triggers (`study/store.py::_TRIGGERS`)
enforce this inside the database: every `UPDATE` and `DELETE` of `manifests`, `graphs`,
`requirements`, `trajectories`, `traces`, `completions`, `observable_defs`, `observations`,
`aggregates`, `signed_runs` and `sampler_diagnostics`, every `DELETE` of `snapshots` and
`units`, and every `UPDATE` of a snapshot other than that fill-in abort with "never updated" or
"never deleted".

### 3.3 The schema

`study/store.py::_SCHEMA`; `PRAGMA foreign_keys = ON`, `journal_mode = WAL`, `synchronous = FULL` for a
writer, `query_only = ON` for a reader. Tokens are canonical strings (`identity.Token.token`).

**`meta`** (`key`, `value`): `store_schema` (`identity.STORE_SCHEMA_VERSION`,
`"sparsegf2.study.store/1"`), `experiment_id`, `created_utc`, `versions` (canonical JSON of
`study/store.py::store_versions()`: `identity.version_record()` with the graph generator, Clifford
numbering, dynamics, row-space format, store schema, the graph and run seed tags,
`RANDOMNESS_CONTRACT` and the package version, and in addition the region seed's tag and length
(`0x5753524F`, 19 words), the peak-bootstrap seed's (`0x57535242`, WSRB, 15 words; the seed of
every stored `peak_D` bootstrap, so a store whose bootstraps were seeded otherwise is not
continued), the outcome seed tag (`0x57534F31`, WSO1), the diagnostics seed tag
(`0x57535344`, WSSD), the token format, the qubit order, the graph-measures version, the
signed-group format and the aggregate estimator), `graph_measures_version`, `qubit_order`
(`store.QUBIT_ORDER`: system $0..n-1$, then reference $n..2n-1$, reference $n+i$ Bell-paired with
system $i$ at $t = 0$; protocol P: system only) and `environment` (of the creating process).
Writing needs every algorithm version of `versions` to equal the code's (the package version may
differ); reading needs only the schema; replay, verification and runs need the randomness
contract (`TableauStore.check_contract`).

**`manifests`**: `revision` (1, 2, ...), `digest` ($H$ of the canonical JSON, unique), `json`
(the canonical JSON), `created_utc`, `sparsegf2_version`, `git_commit` (of the writing process,
or NULL).

**`graphs`** (key `(n, beta, g)`): `seed` (JSON list of the six words of
`identity.graph_seed`), `generator` (`GRAPH_GENERATOR_VERSION`), `edge_order`
(`"offset-major"`), `num_edges` ($2n$), `edges` (blob, §3.4), `digest`
(`identity.edge_digest`), `measures` (canonical JSON of §3.8), `measures_version`.

**`requirements`** (`digest`, `json`): every requirements dictionary (§3.5) used by a unit.

**`units`**: `unit_id`, `protocol`, `n`, `beta`, `p`, `graphs` (JSON $[g_0, g_1)$), `circuits`
(JSON $[c_0, c_1)$), `layers`, `requirements` (digest), `status` (`"done"` or `"failed"`),
`revision`, `started_utc`, `finished_utc`, `elapsed_seconds`, `threads` (Numba threads of the
engine call), `trajectories`, `error` (the exception of a failed unit), and its provenance
(G-RUN-3): `sparsegf2_version`, `git_commit`, `git_dirty` (uncommitted changes to tracked files
of the checkout, NULL if unknown), `randomness_contract`, `environment` (canonical JSON of
`store.environment()`: Python, numpy, numba, llvmlite, the package, the platform) and
`environment_digest`. A unit row is bookkeeping: it is written with the unit's data, or alone
with `status = "failed"` when the unit raised (then no data of it exists). Under the label
`confirmatory` a unit whose environment differs from the units already done is refused (the
message names the differing packages); an exploratory study records it.

**`trajectories`** (natural key `(protocol, n, beta, p, g, c, r)`): `tid`, `u`, `p_float_hex`
(`float.hex` of the double the engine received), `run_seed` (JSON words of `identity.run_seed`),
`graph_digest`, `dynamics_version` (`DYNAMICS_VERSION` for F, `PRODUCT_DYNAMICS_VERSION` for P),
`unit_id` and `revision` of creation. Foreign key `(n, beta, g)` into `graphs`.

**`traces`** (key `(tid, layers)`): `layers` ($T$ of that horizon), `final_layer` (the last
simulated layer: $\tau_0$ when `stop_when_purified` stopped the trajectory, else $T$),
`k_points` and `tau` (blobs, §3.4; NULL for P), `tau0`, `tau1` ($\tau_0$, $\tau_1$, $-1$ when
censored), `censored0` (0/1), `status` (`"purified"`, `"stopped_at_purification"`,
`"censored"` or `"no_reference"` for P), `num_measurements` (measurements through
`final_layer`), `unit_id`, `revision`.

**`completions`** (key `(tid, requirements)`): trajectory `tid` holds every record that
requirements `requirements` ask for; `unit_id`, `revision`.

**`snapshots`** (key `(tid, t)`): `num_qubits` ($N$), `rank` (of the joint space, $N$ when
stored, NULL otherwise), `k` ($k_t$; NULL for P; 0 for an absent snapshot), `joint`,
`joint_digest`, `phys`, `phys_digest` (§3.4), `status` (`"stored"`: the joint state and, for F,
its physical code; `"phys_only"`: the physical code from the engine's native `code_snapshot`,
when `tableau_snapshots` is false, at every probe depth; `"absent_after_purification"`),
`unit_id`.

**`observable_defs`** (key `(observable, version, params)`): `params_json` (the same canonical
text), `level`, `inputs` (JSON), `definition` (the catalog's formula, `observables.get(id).row()`),
`created_utc`.

**`observations`** (key `(tid, t, observable, version, params, context)`): `t` (the depth, or
$-1$ for a trajectory-level value), `params` (`observables.params_key`: canonical JSON of the
resolved parameters), `context` (§3.6), `status` (`store.OBSERVATION_STATUSES`: `ok` a finite
value; `infinite`; `undefined_no_code` a quantity without meaning at $k = 0$, such as the raw
$d_{\mathrm{cyc}}$; `missing` a peak with $d^* = 0$ or an invalid trace; `censored` a first
passage beyond the horizon; `vector` the value is in the data; `absent_after_purification`
no state after a stop; `undefined` any other NaN), `value` (float; SQL NULL stands for NaN,
infinities are kept), `se` (its standard error), `data` (blob, §3.4), `method`, `unit_id` (NULL
for a retrospective value). Index `observations_by_observable (observable, version, params,
tid, t)`.

**`aggregates`** (G-STO-4; key `(protocol, n, beta, p, observable, version, params, selection, t,
estimator, policy, sample_set)`): `selection` (canonical JSON of the data field and index read,
or nulls for `value`), `estimator` (`store.ESTIMATOR_VERSION`, `"stats.nested_mean-1"`),
`policy` (`raise` or `omit`), `sample_set` ($H$ of the canonical JSON of the sorted $(g, c, r)$ of
the rows used), `mean`, `se`, `graphs`, `contributing_graphs`, `trajectories`,
`contributing_trajectories`, `data` (the graph means), `created_utc`. A new sample (more graphs)
is a new record; none is replaced.

**`signed_runs`** (G-TAB-2; key `(tid, outcome_seed, layers, depths)`): the outcome seed
(canonical JSON), the horizon of the replay, the depths, `outcomes` (zlib of one byte 0/1 per
measured site, in circuit order), `random` (zlib of one byte per measurement: 1 when the outcome
was random), `signed_digests` (canonical JSON $t \mapsto$ `runner.signed_digest` of the signed
group at $t$), `format` (`"SGF2SG1"`). Never mixed into the phase-free tables.

**`sampler_diagnostics`** (G-OBS-4; key `(n, beta, g, probe, config, m, coupling)`; `g = -1` for
the cycle and path geometries): `config` ($H$ of the probe's settings), `method`, `chains`,
`samples`, `seed`, `ess`, `tau_int`, `acceptance`, `exchange_acceptance`, `r_hat`,
`r_hat_limit`, `notes` (the `mixing` list of `api.sample_erasure_regions`), `data`.

### 3.4 Byte layouts of the blobs

Every blob is a zlib stream of the bytes below; the compression level
(`storage.compression_level`, 0 to 9) changes no decoded byte.

* `graphs.edges`: the canonical edge list ($a < b$, rows increasing) as signed 32-bit
  little-endian integers, row-major, $2 \times$ `num_edges` of them.
* `traces.k_points`: the change points of `observables.k_change_points`: rows $(t_j, k_{t_j})$
  for $t_0 = 0$, every layer where $k$ drops, and the horizon $T$ as the last row; int32
  little-endian, row-major. `traces.tau`: $\tau_0, \ldots, \tau_n$ as int32 little-endian, $-1$
  marking a censored threshold.
* `snapshots.joint`, `snapshots.phys`: the canonical bytes of
  `tableau.py::PhaseFreeTableau.to_bytes` (format `SGF2RS1`: the 8-byte header
  `b"SGF2RS1\0"`, $N$ and the rank as uint32 little-endian, then the rows of the reduced row
  echelon form over the columns $x_0..x_{N-1}, z_0..z_{N-1}$, each packed most significant bit
  first in $\lceil 2N/8 \rceil$ bytes; doc 03 §3.5). `joint_digest` and `phys_digest` are $H$ of
  these uncompressed bytes.
* `observations.data`: canonical JSON (sorted keys, no spaces, ASCII) of the value's details,
  NaN and infinities written as `null`, `-0.0` as `0.0`.
* A signed group (`runner.signed_digest`): `b"SGF2SG1\0"`, $N$ as uint32 little-endian, then
  per generator of Stim's canonical form its $2N$ bits (X then Z) packed most significant bit
  first and one sign byte (1 for $-$); the digest is $H$ of these bytes.

### 3.5 Requirements, completions and work units

For the latest revision $M_r$ of a chain $M_1, \ldots, M_r$, a protocol $\pi$ and a condition
$(n, \beta)$, the **requirements** $R(\pi, n, \beta)$ (`runner.requirements`) are the dictionary

* `protocol`, `n`, `beta`, `layers` $= T_{\max,r}(n)$, `checkpoints` $= E_r(n, \beta)$ (for
  protocol $\pi$), `dynamics` (the dynamics version);
* for F: `stop_when_purified`, `snapshot` (`"tableau_snapshot"`, the joint state and its
  physical code, or `"code_snapshot"`, the physical code only), `contiguous_distance`,
  `uniform_erasure` (samples and exact budget, or null), `correlated_erasure` (each entry's
  settings with its probe name), `peak` (`analysis.peak`, the predeclared peak rule), `domain`
  (the last revision's own schedule $S_r(n, \beta)$, over which peaks and the checkpoint bounds
  of the first passages are taken; $S_r \subseteq E_r$);
* for P: `snapshot = "tableau_snapshot"`.

The rate is not part of it: every rate of a condition records the same things. Its digest is
$\rho(R) = H(\text{canonical JSON of } R)$. A trajectory is **complete** under $R$ when the
`completions` table holds $(\text{tid}, \rho(R))$; that row is written in the same transaction as
every record $R$ asks for. The **target set** of the latest revision is every
$(\pi, n, \beta, p, g, c)$ with $\pi$ a protocol, $(n, \beta, p)$ a condition, $g < G_*$, $c < C$.
A **work unit** is a rectangle $[g_0, g_1) \times [c_0, c_1)$ of target trajectories of one
$(\pi, n, \beta, p)$ that are not complete, either all stored already (a *rerun*, for what a
revision added: a deeper horizon, new depths, a new observable) or all new; reruns are planned
first, so the stored trajectories receive the new depths before new trajectories are added and
every trajectory of a condition ends with the same schedule (`TableauStore.schedule`). A unit
runs in one engine call on the indices
$u = g s + c$ with the graph collection $0, \ldots, g_1 - 1$ (the engine reads trajectory $u$'s
graph as `edges[u // s]`; design §3).

The engine probes of $R$ have fixed names (`runner.PROBE_NAMES`, the manifest's
`RESERVED_PROBE_NAMES`; a correlated entry is named by `Observables.probe_names`): the sampled
probes' streams are keyed by the name, so the name is part of every sampled value. F runs
`code_dimension` at every layer (the global schedule is every layer), the snapshot probe, and
`contiguous_distance`, `erasure_first_failure`, `uncorrectable_region_counts` and each
`correlated_erasure` entry at $E_r$; P runs `tableau_snapshot` at its own $E_r$.

### 3.6 What the runner records for a trajectory

For protocol F, from the engine's arrays of row $i$ (all functions of `study/observables.py`):

* the trace: $k_0..k_T$ from `code_dimension` (required to start at $k_0 = n$, to be whole and
  nonincreasing), $\tau$ = `first_passage(k)`, checked against the engine's
  `purification_layer` ($\tau_0$ with every layer a checkpoint), and the final layer
  ($\tau_0$ if the trajectory stopped, else $T$);
* at every $t \in E_r$: the snapshot (joint canonical row space and physical code, or the
  physical code only as `phys_only`, or `absent_after_purification` with $k = 0$ after a stop)
  and the snapshot-level observations, `context = ""`, each with its status:

| observable | params | value, se | data | method |
|---|---|---|---|---|
| `k` | `{}` | $k_t$, 0 | — | `k-trace` |
| `d_cyc` | `{}` | raw $d_{\mathrm{cyc}}$, 0; at $k = 0$ NaN with status `undefined_no_code` (SI-08: the engine's NaN kept raw, and from a stored code the same) | `has_code`, `zero_extended` (0 at $k = 0$) | `engine-probe` |
| `uniform_failure` | `samples`, `exact_budget` | — | `first_failure` (the $S$ sizes), `exact_counts` ($m \le w$), `curve` $P_C(m)$, `se`, `max_exact`, `samples` | `hybrid` or `orderings` |
| `D` | as above | $D(0)$, its error; 0 at $k = 0$ (SI-07: the engine records $n + 1$ for every ordering of a code without logical qubits; the raw sizes stay in `uniform_failure`, and $D$ takes the defined 0) | `has_code`, `samples`, `max_exact` | as above |
| `correlated_failure` | the entry's settings and name | — | `couplings`, `sizes`, `failure` $(c \times s)$, `se`, `interpolated_sizes`, `sampler` | the sampler's method |
| `D_J` | as above | — | `couplings`, `robustness` $D(J)$ per coupling, `robustness_se`, `complete`, `has_code` | as above |

* the trajectory-level observations, $t = -1$, `context` $= \rho(R)$ (for `peak_D_J`,
  $\rho(R)$`/`probe name), because they depend on the horizon and the schedule:

| observable | params | value | data |
|---|---|---|---|
| `first_passage` | `{}` | $\tau_0$ (status `censored` and NaN if not reached) | `FirstPassage.to_dict()` with the checkpoint bounds $\widehat\tau^{\mathrm{chk}}_\ell$ over the domain $S_r$, the domain and its digest |
| `peak_d_cyc` | `rule = "exact"` | $T^*$ of the zero-extended $d_{\mathrm{cyc}}$ over the domain (status `missing` when $d^* = 0$) | `Peak.to_dict()` (maximizing set, $k_t$ on it, horizon flag), the domain and its digest |
| `peak_D` | `analysis.peak` | $T^*$ of $D(0)$ by the plateau rule | the peak, one peak per `sensitivity_plateau_z` (each with `se_exact` True: the orderings give the true errors), the bootstrap of $T^*$ (`bootstrap` replicates; seed `observables.bootstrap_seed(key, "peak_D", version)`, the catalog's: neither the horizon nor the checkpoints enter it) |
| `peak_D_J` | `analysis.peak` and `J` | $T^*$ of $D(J)$ | the peak, the sensitivity (each with `se_exact` from the probe's sampler: False for the chain, whose binomial errors are lower bounds, True for exact samples; doc 04 §4.6), the probe name |

For protocol P: the trace row (no $k$, status `no_reference`, final layer $T$) and the snapshot
of the pure $n$-qubit state at every $t \in E_r$ of P. A retrospective value
(`runner.compute`) is a snapshot-level row with `context = ""` and `unit_id` NULL; at a depth
after a stop it is a row of status `absent_after_purification` with no value (SI-05: the
system state there was not evolved, so an entanglement value is missing, never zero), unless
the run stored a value there already.

**Estimator randomness (G-ID-2).** The engine's sampled probes draw from streams keyed by the
run seed's probe key, the trajectory index $u$, the probe name and the layer (and, for
correlated erasures, the size): neither the rate nor the protocol enters. In `gate_mode =
"shared"` the run seed is the same at every rate, so trajectory $(g, c)$ draws **the same**
first-failure orderings and correlated regions at every rate of its condition: the sampling
errors of a rate difference are correlated (the graph-level paired error of `stats` carries
them). The retrospective samplers (`observables.region_seed`) put $p$ into their seed, so their
regions differ between rates.

### 3.7 The logical digest

$\Lambda$ (`TableauStore.logical_digest`) is the SHA-256 of the concatenation, in natural-key
order, of one line per record: a one-letter tag, the canonical JSON of the record's logical
fields, a newline. Logical fields: the manifest digests (`M`), the graph records with their
measurements (`G`), the trajectory identities and provenance used to replay them (`T`), the
traces with the decoded $k$ change points and $\tau$ (`R`), the completions (`C`), the
snapshots' status, $N$, rank, $k$ and digests (`S`), the observable definitions (`D`), and the
observations with `float.hex` of the value and error, the method and the decoded data (`O`).
Left out: row ids, unit ids, revisions of creation, timestamps, elapsed times, thread counts,
code versions and the compressed bytes. `history=False` keeps the current state only: the latest
manifest digest, for each trajectory the trace of its largest horizon, its latest completion and
the trajectory-level observations of its latest requirements. `manifests=False` leaves the
manifest digests out (they also change with `storage` and `resources`, which never change a
value).

### 3.8 Protocol P and the graph measurements (design §12)

**Protocol P** runs `MonitoredCircuit` with `reference = "none"` (the system starts in
$\lvert 0\rangle^{\otimes n}$) and without `stop_when_purified`, with the graphs, run seed,
indices and horizon of F. Its key is $\xi$ with $\pi = $ P (string `"<id>:protocol=P:n=.."`); its
snapshots are the pure $n$-qubit state (`store.tableau(..., protocol="P")`); it has no $k$
trace and no physical code.

**Graph measurements** (`study/store.py::graph_measures`, version `GRAPH_MEASURES_VERSION`), computed
once from the edge list when a graph is first stored:
$q_G = \lvert E(G) \setminus E(C(n,2))\rvert / (2n)$ (`q`, with the count `rewired_edges`);
$\bar d_G$ = the mean shortest-path distance over the `reachable_pairs` unordered pairs of
distinct vertices joined by a path (NaN if none), and `components`; $\lambda_2$ = the
second-smallest eigenvalue of the unnormalized Laplacian $L = D - A$, set to 0 exactly when $G$
is disconnected; $\mathrm{Var}(\deg) = \frac1n\sum_v \deg(v)^2 - (\frac1n\sum_v \deg(v))^2$.

### 3.9 Export

`TableauStore.export(directory)` (format `EXPORT_FORMAT = "sparsegf2.study.export/1"`) writes
`manifest.json` (latest revision), `revisions.json`, `versions.json` (the store's versions and
both logical digests), `graphs.csv` (records and measurements), `conditions.csv`, and per
condition a directory `<protocol>_n<n>_beta<b>_p<p>` (a `/` of a rational token written `over`)
with:

* `trajectories.csv`: per trajectory ($g, c, r, u$) the latest trace's horizon, final layer,
  status, $\tau_0$, $\tau_1$, censoring and measurement count;
* `first_passage.npz` (F): `g, c, r, tau (rows, n + 1), observed, horizon`;
* `observations.csv` and `observations.npz`: every stored value with
  `g, c, r, t, observable, version, params, context, value, se, method` (NaN as `nan`; the data
  stay in the store);
* `aggregates.csv`: for every scalar snapshot observable at every depth (and `d_cyc`'s zero
  extension and $D(J)$ at each coupling) the nested equal-graph-weight mean of v1.4 §6.3 with the
  graph-cluster error of §6.4 (`stats.nested_mean`; undefined values left out with
  `nan_policy = "omit"`, said in the row);
* `censoring.csv` (F): per threshold $\ell$, `stats.censoring_summary` and the graph-weighted
  Kaplan–Meier median and restricted mean of $\tau_\ell$ (`stats.nested_kaplan_meier`);
* `peaks.json`: `stats.peak_time_summaries` of every stored peak under both explicit horizon
  policies.

The files are written in a temporary sibling directory renamed to `directory` when complete;
`directory` must not exist. `graph_aggregates.csv` gives, per $(n, \beta)$ and measurement, the
equal-weight mean over the stored graphs with its standard error and the connected fraction
(the reachable-pair mean distance of a disconnected graph averages within its components, so
it is read beside the connected fraction).

### 3.10 The outcome-aware replay and its seed (v1.4 §4, validation 5; G-TAB-2, SI-24)

`runner.replay_signed(store, key, depths)` runs the trajectory's exported circuit on a
phase-tracking `Simulator`. Its Born outcomes come from the simulator's stream, seeded by
default with

$$\mathrm{outcome\_seed}(\xi) = (\texttt{0x57534F31}, w(\text{experiment}), \mathrm{index}(\pi), n,
w(\beta), w(p), g, c, r),$$

twelve 32-bit words (`runner.outcome_seed_of`), $w(x)$ the two words of the 8-byte BLAKE2b of
the text read little-endian (`identity.token_words` for the tokens), $\mathrm{index}(\pi)$ 0
for F and 1 for P. The seed is a function of the stored identity alone, so the replay is
reproducible from the store; `record=True` keeps the outcomes, the random flags and the
signed digests in `signed_runs`.

## 4. Propositions and proofs

**Proposition 1 (a unit is all or nothing).** After any interruption of `runner.run` (an
exception, or the process killed at any instant), for every work unit either every record of it
(its graphs, trajectories, traces, snapshots, observations, completions and its `done` unit row)
is in the store, or none is.

*Proof.* `study/runner.py::_run_unit` writes all of a unit's records between one `BEGIN IMMEDIATE` and
one `COMMIT` (`TableauStore.transaction`). SQLite in WAL mode appends a transaction's pages to the
write-ahead log and makes them visible only through the commit frame written last; a reader, and
the recovery that runs when the database is next opened, use the log only up to the last commit
frame. So pages of a transaction without its commit frame are never read. An exception inside the
block executes `ROLLBACK`, which discards the transaction. A killed process writes no commit
frame for an unfinished transaction. Hence a unit's records are either all committed, by its one
commit, or none are. (Only the failure row of `_record_failure`, in a transaction of its own,
follows an exception; it carries no data.) ∎

**Proposition 2 (one trajectory's records do not depend on its unit).** The records the runner
writes for a trajectory $\xi$ depend only on $\xi$, the requirements $R$ and the versions of
`meta.versions`: not on the other trajectories of its unit, the unit's size, the order of units,
the thread count, the compression level or the moment of a resume.

*Proof.* (i) The engine: row $i$ of `run(indices=...)` is trajectory $u_i$, exactly as a range
run gives it: its circuit stream is `stream_state(key_struct, u_i)`, its sampled probes draw
from `mix(u_i, name hash, layer)` (and the region size for correlated erasures), and its graph is
`edges[u_i // s]` (doc 03 Proposition 13; design §2.2). The collection passed is the stored
graphs $0..g_1-1$, so `edges[g]` is graph $g$'s record whatever $g_1$ is. Each trajectory is
simulated by one thread, sequentially, so the thread count changes no arithmetic. The seeds are
the manifest's (`run_seed`), which depend on $(n, \beta)$ (and $p$ in independent mode) only.
(ii) The ingestion (`_ingest`) reads only row $i$ of the result arrays, the graph of $\xi$ and,
for the correlated sampler's description, a spec holding that graph alone
(`study/runner.py::_graph_sampler`), and applies the pure functions of `study/observables.py`.
(iii) The stored row spaces are canonical: the reduced row echelon form is unique per row space
(doc 03 Proposition 5), so the bytes and digests do not depend on the generators the engine
produced; the batch canonicalization gives exactly the one-at-a-time result (doc 03 §5.9).
(iv) Compression changes stored bytes but no decoded value, and the comparisons of §3.2 and the
logical digest use decoded values. ∎

**Proposition 3 (resume).** Let a run of a revision be interrupted, and `runner.run` called
again until `plan` is empty. The store then holds exactly the records of an uninterrupted run;
in particular $\Lambda$ is the same.

*Proof.* By Proposition 1 the interrupted store is the uninterrupted store restricted to the
committed units. `plan` returns the target trajectories without a completion under the current
$R$, that is, exactly those of the uncommitted units (possibly split differently). By
Proposition 2 their records do not depend on the split. Records shared by two units (a graph
inserted by both) are compared and equal (§3.2), so the union of the units' records is the
uninterrupted store's record set. $\Lambda$ is a function of that set (§3.7). ∎

**Proposition 4 (prefix and horizon invariance; v1.4 §7.3).** For one trajectory and horizons
$T_1 \le T_2$, the runs to $T_1$ and to $T_2$ have the same $k_t$ for $t \le T_1$, the same
snapshot at every depth $t \le T_1$ they both evaluate, the same probe values there, and, with
`stop_when_purified`, the same final layer if it is below $T_1$.

*Proof.* The engine draws layer $t$'s gates and mask uniforms, in order, from the trajectory's
stream after the draws of layers $1..t-1$ and of nothing that depends on $T$ (doc 01 Proposition
13; doc 03 Proposition 13), so the state after layer $t \le T_1$ is the same in both runs. The
probe evaluations at a depth draw from their own keyed streams, independent of $T$. A stop at
purification happens at the first layer where $k = 0$, which is the same layer in both runs when
it is $\le T_1$; after it $k = 0$ in both traces (the zero fill is exact, doc 04 Proposition 4).
∎

**Proposition 5 (probe-stream separation; v1.4 §7.3).** Adding a probe, a checkpoint or a
correlated size changes no circuit, no $k_t$, no stored snapshot and no other probe's value at a
depth it evaluates.

*Proof.* The circuit stream of $u$ is consumed by the gates and masks only (the probes draw
nothing from it); a probe's evaluation at layer $L$ draws from the stream
`mix(u, hash(name), L)` (and, for a correlated size $m$, from a stream keyed by $m$ too), which no
other probe, depth or size uses. Snapshots draw nothing. ∎

**Proposition 6 (extension).** Let $M_1, \ldots, M_r$ be a valid revision chain and a store be
complete under $M_{r-1}$. After `runner.extend(store, M_r)` and `runner.run` until the plan is
empty: (a) every record stored before is unchanged; (b) if the schedule of $M_r$ alone contains
every earlier depth ($S_r(n, \beta) = E_r(n, \beta)$ for every condition and protocol), the
store's current state equals that of a store made directly under $M_r$:
$\Lambda_{\text{history=False}}$ is the same.

*Proof.* (a) Nothing is updated (§3.2): every identity met again is compared, and by
Propositions 4 and 5 the rerun of a stored trajectory under $R_r$ reproduces every stored record
at the depths and horizons where it exists: the stored trace is a prefix of the new one, each
stored snapshot and value is met with equal content. New identities are inserted. (b) The target
set of $M_r$ is the same for both stores. Every trajectory of either store is complete under
$R_r$ and, by Proposition 2, has the same records under $R_r$ in both: the trace at $T_r$, the
snapshots and snapshot-level values at every $t \in E_r$ (in the extended store the depths of
earlier revisions are a subset of $E_r$, and they are equal by (a)), and the trajectory-level
values of context $\rho(R_r)$. $\Lambda_{\text{history=False}}$ reads exactly these, the graphs,
the observable definitions and the latest manifest digest. ∎

**Proposition 7 (a retrospective observable draws nothing from the circuits).** `runner.compute`
calls no engine function, and the values it writes do not depend on the order of the snapshots,
the batch size or other observables computed before.

*Proof.* `compute` reads stored snapshots and graphs, and calls `observables.compute`. The only
random draws are those of the sampled `mean_leakage`, from `erasures.sample_regions` with the seed
`observables.region_seed(key, t, m, J, id, version)`: a function of the trajectory key and the
arguments only, in the domain tagged `WSRO`, disjoint from the graph (`WSG1`) and run (`WSC1`)
seeds (doc 04 Proposition 18). So each value is a function of the stored snapshot, the graph,
the key and the parameters. ∎

**Proposition 8 (bounded memory and indexed retrieval).** (a) `iter_snapshots(batch_size=B)`
holds at most $B$ snapshot rows (compressed) at a time, whatever the size of the store. (b) A
lookup by the parameter tuple reads $O(\log M)$ pages of $M$ stored rows and decompresses only
the blobs of the requested snapshot.

*Proof.* (a) Each query of the keyset cursor has `LIMIT B` and resumes after the last
`(tid, t)` read, using the order of the unique index `(tid, t)`; the rows of one query are
yielded and released before the next query. (b) The trajectory is found through the unique
index on `(protocol, n, beta, p, g, c, r)` and the snapshot through the unique index on
`(tid, t)` (B-trees); the blobs of other rows are not read. The query plans are checked by the
tests (no `SCAN` of a table). ∎

**Proposition 9 (integrity on read).** A stored row space whose blob was altered is detected
when it is read, unless the alteration produces another zlib stream whose decompressed bytes
have the same SHA-256.

*Proof.* `study/store.py::_decode_rowspace` decompresses the blob (an invalid stream raises), compares
$H$ of the bytes with the stored digest (a different digest raises), and parses them with
`PhaseFreeTableau.from_bytes`, which accepts only canonical bytes. Altered bytes with an
unchanged SHA-256 would be a collision. An altered digest column is detected by the replay
(`runner.verify`), which recomputes the state and compares digests. ∎

**Proposition 10 (one writer).** At most one process writes a store at a time; a lock left by a
process that has ended on the same machine is taken over.

*Proof.* A writer holds `directory/.lock`, created by `api.py::_lock_output_dir`: the record is
written to a private file and hard-linked to the lock path (or created with `O_CREAT | O_EXCL`),
operations that fail when the path exists, so of concurrent writers exactly one creates it. A
second writer reads the holder's record and is refused with a message naming the holder's pid,
host and start time, unless the holder's process has ended on this machine (`_holder_alive`,
which on Windows also compares the process creation time, since pids are reused); then the stale
lock is removed by `_remove_stale_lock`, which deletes it only if it still holds the bytes judged
stale, one remover at a time. ∎

**Proposition 11 (the logical digest separates contents).** Two stores have equal $\Lambda$ if
their logical records (§3.7) are equal; if they differ, their $\Lambda$ differ unless SHA-256
collides.

*Proof.* The digested text is a function of the records in natural-key order. It is also
injective: each line is a tag followed by the canonical JSON of a list with a fixed number of
fields for that tag, ended by a newline (JSON strings escape newlines), so the text parses back
into the sequence of records. ∎

**Proposition 12 (P replays F's gates and masks; design §12).** For the same graph collection,
run seed and index $u$, protocols F and P apply the same gates, in the same order, and measure
the same sites in every layer.

*Proof.* The engine's draw order per layer is: for each of the $\lfloor n/2\rfloor$ gates an edge
and a Clifford, then one uniform per site (doc 01 §4; `tests/_ws_reference.py::study_draws`).
The reference enters only through the setup and the probes: for `"purification"` and `"none"`
no number is drawn for the reference (a partner is drawn only for `"single"`). So the streams
of F and P are consumed identically. The tests compare the exported circuits layer by layer with
each other and with the independent transcription of the draws. ∎

**Proposition 13 ($\lambda_2 = 0$ exactly when $G$ is disconnected).** For $L = D - A$,
$\dim\ker L$ is the number of connected components.

*Proof (derived here).* For $x \in \mathbb R^n$, $x^\top L x = \sum_{\lbrace a,b\rbrace \in E}
(x_a - x_b)^2 \ge 0$, so $L$ is positive semidefinite and $Lx = 0$ iff $x^\top L x = 0$ iff
$x_a = x_b$ on every edge, that is, $x$ is constant on each component. The indicator vectors of
the components are a basis of these $x$. So the eigenvalue 0 has multiplicity equal to the
number of components, and $\lambda_2 = 0$ iff there are two or more. The code sets it to 0
exactly in that case instead of using the floating-point eigenvalue, which is only near 0. ∎

**Proposition 14 (the outcome seed).** (a) Distinct trajectory keys have distinct outcome
seeds unless two texts of one field share a 64-bit BLAKE2b digest. (b) No outcome seed equals a
graph, run, region or peak-bootstrap seed as a `SeedSequence` entropy. (c) The same seed gives
the same outcomes and signed groups.

*Proof (derived here).* (a) Every field of the key sits at a fixed position of a fixed-length
tuple, the integers one word each (refused at $2^{32}$) and the texts two words each, so the map
is injective up to a collision of the text hashes. (b) The first word is `0x57534F31`; graph,
run, region and peak-bootstrap seeds start with `0x57534731`, `0x57534331`, `0x5753524F` and
`0x57535242`, and every tuple has more than four words, so `SeedSequence`'s zero padding never
applies. (c) The simulator's random outcomes are drawn from its stream in circuit order, and the
circuit is a function of
the stored identity (Proposition 2). ∎

**Proposition 15 (peaks do not depend on the revision history).** The peaks and checkpoint
bounds of a trajectory complete under the requirements of revision $r$ are the same in a store
built through revisions $1..r$ and in a store made directly under $M_r$.

*Proof.* They are computed over the domain $S_r(n, \beta)$, the last revision's own schedule,
which is part of $R_r$ in both stores and contained in the stored depths $E_r$; the values of
the traces at those depths are equal in both stores (Propositions 4 and 5). ∎

The Born law of the outcome-aware replay (v1.4 §4: a measured $Z_i$ whose value the state
determines gives that value; otherwise each outcome has probability 1/2) is checked by the tests
against Stim, measurement by measurement, rather than proved here: the simulator's measurement
update is the one of doc 03 Proposition 4.

## 5. Algorithms

### 5.1 `runner.plan(store)`

1. Read the revision chain; take the latest manifest $M_r$.
2. For each protocol (F, then P), condition $(n, \beta)$ and rate $p$: compute $R$ and
   $\rho(R)$; read the set $D$ of $(g, c)$ complete under $\rho(R)$ (one indexed query).
3. For each $g < G_*$, the missing circuits $\lbrace c < C : (g, c) \notin D\rbrace$ as maximal
   runs $[c_0, c_1)$; split each run into pieces of at most `cap` circuits, where `cap` $=
   \max(1, \min(\text{max\_unit\_trajectories}, \lfloor \text{max\_unit\_bytes} / b\rfloor))$
   and $b$ is the engine memory of one trajectory (`_bytes_per_trajectory`: $8 E N
   \cdot 2\lceil N/64\rceil$ bytes of snapshot rows plus 8 bytes per probe value).
4. Group the graphs with the same piece into runs of consecutive labels, cut into units of at
   most $\min(\text{graphs\_per\_unit}, \lfloor\text{cap}/(c_1 - c_0)\rfloor)$ graphs.
   Missing trajectories that are stored already (reruns) and new ones never share a unit.
5. Sort the units: reruns first, then by (protocol, $n$, $\beta$, $p$, $g_0$, $c_0$).

Cost: one query per condition and rate, $O(G_* C)$ work. A complete store gives an empty plan.

### 5.2 `runner.run` and one unit

1. Check that the unit's requirements digest and revision are current (else plan again).
2. Graphs $0..g_1-1$: from the store (decoded, digest and invariants checked, seed and generator
   compared with the manifest) or built by `identity.build_graph` (kept for insertion).
3. `MonitoredCircuit(edges, probes, **manifest.monitored_circuit_kwargs(n, beta, p, protocol))`
   and `run(indices=[g s + c ...], seed=list(run_seed))`.
4. Canonicalize every snapshot of the unit, one evaluation at a time, in parallel
   (`_canonical_snapshots`: `from_words_batch`, `physical_tableaux`, `to_bytes_batch`).
5. One transaction: insert the new graphs with their measurements, the requirements, the unit
   row; for each trajectory (§3.6) insert-or-compare the trajectory, the trace, the snapshots,
   the observations and the completion; set the unit's finish time; commit.
6. On an exception: roll back, write the failure row, raise.

### 5.3 `runner.replay(store, key, depths)`

The stored run seed, $u$ and graphs $0..g$ are checked against the manifest; a spec with
`layers = max(depths)`, `stop_when_purified = False`, a `tableau_snapshot` at `depths` (and
`code_dimension` for F) runs `indices = [u]`. With `verify`, each requested depth with a stored
snapshot must give its joint digest (or, for a code-only store, its physical digest), and $k$
must equal the stored trace on the common layers.

### 5.4 `runner.replay_signed(store, key, depths, outcome_seed=...)`

The trajectory's exported circuit (`runner.circuit`) runs on `Simulator(N, phases=True,
seed=outcome_seed)`; at each requested tick the signed generators (Stim's canonical form) and
the phase-free row space (`PhaseFreeTableau.from_simulator`, signs dropped) are kept; with
`verify` the latter must equal the stored row space. Same seed, same outcomes.

### 5.5 `runner.compute(store, id, params)`

Iterate the stored snapshots of the protocol and filters in batches (`iter_snapshots`); for
each, build `SnapshotInput(num_system, t, key, code, tableau, edges)` (the physical code from
`phys`, the joint state only if the observable needs it or for P, the graph if needed); compute;
write each batch in one transaction (`put_observation(..., compare="value")`).

### 5.6 `runner.verify(store, sample, seed)`

Draw `sample` trajectories without replacement with `numpy.random.Generator(PCG64(seed))`
(all for `None`); for each, regenerate its graph from the seed, decode every stored snapshot
(digest check), and replay it at its stored depths with verification. Failures are collected,
not raised.

### 5.7 The lock and transaction protocol

Writer: take the lock (Proposition 10), open the database with `journal_mode = WAL`,
`synchronous = FULL`, `foreign_keys = ON`; every write in `BEGIN IMMEDIATE ... COMMIT`;
`close()` releases the lock. Reader: no lock, `query_only = ON`; it sees the last committed
state and is never blocked by the writer (WAL).

### 5.8 Measured costs

One machine (AMD Ryzen 9 9950X3D, 32 threads, Windows 11; other processes were running, so the
numbers are indicative), 64 trajectories of one condition ($\beta = 0.1$, $p = 0.1$,
$T = 2n$, the geometric rule with 12 points, `contiguous_distance`, `uniform_erasure` with 100
orderings and an exact budget $n$, no correlated probe):

| $n$ | engine (s) | ingestion (s), no $T^*$ bootstrap | trajectories/s | snapshots/s | ingestion with a 200-replicate bootstrap (s) | joint bytes per snapshot (compressed) | physical bytes | file bytes per snapshot |
|---|---|---|---|---|---|---|---|---|
| 32 | 0.005 | 0.27 | 239 | 2864 | 1.55 | 386 | 93 | 1925 |
| 64 | 0.012 | 0.28 | 226 | 2708 | 1.48 | 1057 | 200 | 3115 |
| 128 | 0.032 | 0.52 | 123 | 1600 | 1.74 | 3133 | 485 | 5785 |

(scratchpad script `t3_bench.py`; "file bytes" include the observations, about 120 values per
trajectory.) The per-trajectory bootstrap of $T^*$ (`observables.bootstrap_peak`, 200
replicates) costs about 20 ms per trajectory and dominates ingestion; set `analysis.peak.bootstrap`
to 0 for large campaigns.

Lookup by key (scratchpad script `t3_lookup.py`; $n = 6$, 2000 random keys and depths per size,
the same machine):

| snapshots stored | trajectories | `store.tableau` (decoded, digest checked) | `store.snapshot` (record only) |
|---|---|---|---|
| 1,000 | 200 | 40 µs | 18 µs |
| 10,000 | 2,000 | 74 µs | 32 µs |
| 100,000 | 20,000 | 69 µs | 34 µs |

The latency does not grow with the store beyond the first cache effects (Proposition 8).

## 6. From the mathematics to the code

| Definition or algorithm | Code |
|---|---|
| store creation, opening, versions, lock | `study/store.py::TableauStore.create`, `TableauStore.open`, `_check_versions`, `api.py::_lock_output_dir` |
| one transaction | `TableauStore.transaction`, `study/store.py::_Transaction` |
| insert or compare (§3.2) | `_Transaction.put_graph`, `put_trajectory`, `put_trace`, `put_snapshot`, `put_observation`, `put_completion`, `put_requirements` |
| retrieval by the tuple | `TableauStore.key`, `trajectory`, `trace`, `k_trace`, `first_passage`, `depths`, `snapshot`, `tableau`, `code`, `observation`, `observations`, `iter_snapshots`, `graph`, `graph_measures` |
| decoding with integrity checks | `study/store.py::_decode_rowspace`, `_int32_array`, `SnapshotRecord.tableau`, `SnapshotRecord.code` |
| check, logical digest, export | `TableauStore.check`, `TableauStore.logical_digest`, `TableauStore.export`, `study/store.py::_export_condition`, `_aggregates`, `_peak_summaries` |
| graph measurements | `study/store.py::graph_measures` |
| requirements, digest, probes | `study/runner.py::requirements`, `requirements_digest`, `_probes`, `PROBE_NAMES` |
| the plan | `study/runner.py::plan`, `WorkUnit`, `_runs`, `_bytes_per_trajectory` |
| running a unit | `study/runner.py::run`, `_run_unit`, `_canonical_snapshots`, `_ingest`, `_graph_sampler`, `_record_failure` |
| revisions | `study/runner.py::extend`, `TableauStore.add_revision`, `study/manifest.py::check_extension`, `effective_checkpoints` |
| replay, signed replay, circuit | `study/runner.py::replay`, `replay_signed`, `circuit` |
| retrospective observables | `study/runner.py::compute`, `study/observables.py::compute` |
| verification | `study/runner.py::verify`, `study/runner.py::_native_codes` |
| outcome seed, signed digest | `study/runner.py::outcome_seed_of`, `study/runner.py::signed_digest`, `study/store.py::_Transaction.put_signed_run`, `TableauStore.signed_runs` |
| aggregates | `TableauStore.aggregate`, `TableauStore.aggregates`, `study/store.py::_Transaction.put_aggregate` |
| sampler diagnostics | `study/runner.py::diagnose_samplers`, `TableauStore.sampler_diagnostics` |
| provenance, versions | `study/store.py::store_versions`, `study/store.py::environment`, `study/runner.py::_provenance`, `study/runner.py::_check_environment`, `TableauStore.check_contract` |
| append-only | `study/store.py::_TRIGGERS` |
| protocols | `study/identity.py::PROTOCOLS`, `TrajectoryKey.protocol`, `dynamics_version`; `study/manifest.py::ProtocolSettings`, `ExperimentManifest.protocol_settings`, `monitored_circuit_kwargs(..., protocol)` |

| Stored field | Formula |
|---|---|
| `traces.k_points` | $(t_j, k_{t_j})$: $t_0 = 0$, every $t$ with $k_t < k_{t-1}$, and $T$ |
| `traces.tau` | $\tau_\ell = \min\lbrace t \le T : k_t \le \ell\rbrace$, $-1$ if none, $\ell = 0..n$ |
| `snapshots.k` | $k_t = n - \operatorname{rank} W_t^{\mathrm{phys}}$ ($= $ the engine's `code_dimension`) |
| `snapshots.joint_digest` | $H(\text{SGF2RS1 bytes of } \widetilde{\mathsf S}_t)$ |
| `snapshots.phys_digest` | $H(\text{SGF2RS1 bytes of } W_t^{\mathrm{phys}})$ |
| `observations` `D` | $D(0) = \sum_{m<n} (1 - P_C(m))$ (doc 04 Proposition 13) |
| `graphs.measures.q` | $\lvert E(G) \setminus E(C(n,2))\rvert / (2n)$ |
| `graphs.measures.lambda2` | second-smallest eigenvalue of $D - A$; 0 if disconnected |

## 7. Worked examples

### (a) A store, a run, and a tableau by its parameter tuple

```python
import math
import tempfile
from pathlib import Path

import numpy as np

from sparsegf2.study import runner
from sparsegf2.study.manifest import ExperimentManifest
from sparsegf2.study.store import TableauStore

data = {
    "experiment_id": "doc05",
    "conditions": [{"n": [6], "beta": ["0.25"], "p": ["0.1", "0.3"]}],
    "depth": {"layers": {"fixed": 10}, "checkpoints": {"list": [2, 5]}},
    "trials": {"graphs": 3, "circuits_per_graph": 2, "circuit_index_stride": 4},
    "seeds": {"master_graph_seed": 101, "master_circuit_seed": 202},
    "observables": {"uniform_erasure": {"samples": 12, "exact_budget": 6}},
    "storage": {"graphs_per_unit": 2},
    "analysis": {"peak": {"bootstrap": 5}},
}
manifest = ExperimentManifest.from_json(data)
root = Path(tempfile.mkdtemp())
store = TableauStore.create(root / "store", manifest)
units = runner.plan(store)
print([str(u) for u in units][:2], len(units))
summary = runner.run(store)
assert summary.trajectories == 12 and runner.plan(store) == []

# a snapshot by its tuple; beta and p in any spelling of the same token
tab = store.tableau(n=6, beta="1/4", p=0.3, g=2, c=1, t=5)
code = store.code(n=6, beta=0.25, p="0.3", g=2, c=1, t=5)
k = store.k_trace(n=6, beta="0.25", p="0.3", g=2, c=1)
assert tab.num_qubits == 12 and tab.is_pure()
assert code.num_logical == k[5] == 6 - tab.physical_tableau(6).rank
print(store.depths(n=6, beta="0.25", p="0.3", g=2, c=1), k.tolist())
```

### (b) The trace: change points, exact first passages, censoring

```python
key = store.trajectory_keys(p="0.1")[0]
trace = store.trace(key)
fp = store.first_passage(key)
for ell in range(7):  # the definition tau_ell = min{t : k_t <= ell}, -1 if never
    hits = [t for t in range(trace.layers + 1) if trace.k[t] <= ell]
    assert fp.tau[ell] == (hits[0] if hits else -1)
print(trace.status, fp.to_dict()["tau"], fp.to_dict()["tau_chk"])
```

### (c) An interruption before a commit changes nothing; the resumed store is the same

```python
reference = store.logical_digest()
other = TableauStore.create(root / "interrupted", manifest)
seen = []


def die(unit):
    seen.append(unit)
    if len(seen) == 2:
        raise RuntimeError("killed after the inserts, before the commit")


runner._BEFORE_COMMIT = die
try:
    runner.run(other)
except RuntimeError:
    pass
runner._BEFORE_COMMIT = None
print([u["status"] for u in other.units()], other.counts()["trajectories"])
runner.run(other)  # resume: the plan holds the uncommitted units
assert other.logical_digest() == reference
other.close()
```

### (d) A deeper horizon: every stored record kept, the prefix verified

```python
deeper = dict(data, depth={"layers": {"fixed": 14}, "checkpoints": {"list": [2, 5, 10, 12]}})
before = store.trace(key)
runner.extend(store, ExperimentManifest.from_json(deeper))
runner.run(store)
after = store.trace(key)
assert (before.layers, after.layers) == (10, 14)
assert np.array_equal(after.k[:11], before.k)  # the stored prefix, reproduced exactly
assert store.trace(key, layers=10).k.tolist() == before.k.tolist()  # and still stored
fresh = TableauStore.create(root / "fresh", ExperimentManifest.from_json(deeper))
runner.run(fresh)
assert fresh.logical_digest(history=False) == store.logical_digest(history=False)
fresh.close()
```

### (e) Replay at unsaved depths, and the outcome-aware replay

```python
replayed = runner.replay(store, key, [3, 5, 20])  # 3 and 20 were never saved
assert replayed.verified_depths == (5,)
assert replayed.tableaux[5] == store.tableau(key, 5)
signed = runner.replay_signed(store, key, [5, 14], outcome_seed=1)
assert signed.tableaux[5] == store.tableau(key, 5)  # signs dropped: the stored row space
xs, zs, signs = signed.signed[5]
print(len(signed.measurements), int(signs.sum()))
```

### (f) An observable computed later, from the stored snapshots only

```python
from sparsegf2 import mipt

real_run = mipt.MonitoredCircuit.run
mipt.MonitoredCircuit.run = None  # nothing may simulate
try:
    first = runner.compute(store, "half_cut_entropy")
    again = runner.compute(store, "half_cut_entropy")  # idempotent
finally:
    mipt.MonitoredCircuit.run = real_run
assert again.inserted == 0 and again.existing == first.inserted
values = store.observations("half_cut_entropy", p="0.3")
assert values["value"][0] == store.tableau(store.trajectory_keys(p="0.3")[0], 0).entropy([0, 1, 2])
print(first, values["g"][:5], values["t"][:5])
```

### (g) The graph measurements, by hand

```python
from sparsegf2.study.store import graph_measures

edges = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4], [3, 5], [4, 5], [0, 3], [1, 5]])
m = graph_measures(edges, 6)
ring = {tuple(sorted((i, (i + d) % 6))) for i in range(6) for d in (1, 2)}
assert m["rewired_edges"] == sum(1 for a, b in edges.tolist() if (a, b) not in ring)
A = np.zeros((6, 6))
A[edges[:, 0], edges[:, 1]] = A[edges[:, 1], edges[:, 0]] = 1
L = np.diag(A.sum(axis=1)) - A
assert math.isclose(m["lambda2"], np.linalg.eigvalsh(L)[1], rel_tol=1e-12)
deg = A.sum(axis=1)
assert math.isclose(m["degree_variance"], deg.var(), rel_tol=1e-12) and m["components"] == 1
print({k_: round(v, 4) if isinstance(v, float) else v for k_, v in m.items()})
```

### (i) Statuses, an aggregate record, a recorded signed replay, append-only rows

```python
import sqlite3

from sparsegf2.study.store import STORE_FILE

d = store.observations("d_cyc", p="0.3")
print(sorted(set(d["status"])))  # "ok", and "undefined_no_code" where k = 0
record = store.aggregate("D", t=5, n=6, beta="0.25", p="0.3",
                         params={"samples": 12, "exact_budget": 6})
print(round(record["mean"], 4), round(record["se"], 4), record["graphs"], record["sample_set"][:12])
signed = runner.replay_signed(store, key, [5], record=True)  # the seed of the key itself
assert signed.outcome_seed == runner.outcome_seed_of(key)
assert store.signed_runs(key)[0]["outcome_seed"] == list(signed.outcome_seed)
raw = sqlite3.connect(store.path / STORE_FILE)
try:
    raw.execute("UPDATE snapshots SET k = 0")
except sqlite3.IntegrityError as exc:
    print("refused:", exc)
raw.close()
```

### (h) Export, and an aggregate recomputed from the exported rows

```python
import csv

from sparsegf2.study import stats

out = store.export(root / "export")
folder = out / "F_n6_beta0.25_p0.1"
with np.load(folder / "observations.npz") as obs:  # closed at once: no file left open
    sel = (obs["observable"] == "D") & (obs["t"] == 5)
    estimate = stats.nested_mean(obs["g"][sel], obs["c"][sel], obs["r"][sel], obs["value"][sel])
with open(folder / "aggregates.csv", newline="") as f:
    row = next(r for r in csv.DictReader(f) if r["observable"] == "D" and r["t"] == "5")
assert float(row["mean"]) == float(estimate.mean)
print(sorted(p.name for p in folder.iterdir()))
store.close()
```

## 8. Validation map

`tests/test_study_store.py` (S), `tests/test_study_runner.py` (R),
`tests/test_study_reproducibility.py` (P), `tests/test_study_records.py` (G),
`tests/test_study_runner_peaks.py` (K); every check is
exact (equality of bytes, digests, arrays or decoded records) unless it says otherwise.

| Property | Test | What is checked |
|---|---|---|
| schema, versions, natural keys | S::test_create_writes_the_schema_the_versions_and_revision_one, S::test_every_table_has_a_unique_natural_key, S::test_a_store_of_another_schema_is_refused, S::test_writing_needs_the_versions_the_store_was_made_with | meta, the revision, unique keys, foreign keys, refusals |
| round trip against the engine (§9.2 (h)) | S::test_every_stored_tableau_equals_a_fresh_engine_run | every stored joint tableau equals a run built by hand (reference graphs `_ws_reference.rewire`, reference seeds, the v1.4 §8.1 keywords written out); bytes equal the SGF2RS1 bytes written by an independent elimination |
| physical code (§9.2 (g)) | S::test_the_stored_physical_code_equals_the_native_code_snapshot | equal row space to the native `code_snapshot`, and to the independent physical rows of the stored joint state |
| Stim | S::test_stored_tableaux_equal_stim_on_the_exported_circuit, R::test_p_snapshots_are_the_pure_system_state | Stim on the exported circuit at every depth, independent canonical digest |
| observations from stored codes | S::test_stored_observations_reproduce_from_the_stored_codes, R::test_recomputing_the_run_s_values_from_stored_codes_cross_checks_them | $k$, $d_{\mathrm{cyc}}$, exact counts from the stored code; recomputation equals the run's values, a forged value is caught |
| trace and first passage | S::test_k_trace_and_first_passage_by_the_parameter_tuple | $k$ equals a fresh engine run; $\tau$ by the definition; $\tau_0$ = engine; token spellings |
| explicit absence | S::test_unsaved_depths_and_absent_snapshots_are_explicit | unsaved depth `KeyError`; absent after a stop, `ValueError`, $k = 0$, no blob |
| indexes, memory (Prop. 8) | S::test_lookups_by_key_use_the_indexes, S::test_iterating_snapshots_holds_one_batch_at_a_time | query plans without `SCAN`; tracemalloc peak over 1200 snapshots within 1.5 times that over 20 plus 20 KB, and below a third of a one-batch read |
| never overwrite, validation (§3.2) | S::test_meeting_an_identity_again_compares_instead_of_overwriting, S::test_invalid_records_are_refused_before_commit | equal content a no-op; any difference refused, store unchanged; rank, isotropy, $n - k$, monotonicity, $\tau$ refused when wrong |
| integrity (Prop. 9) | S::test_altered_blobs_and_digests_are_detected_on_read, R::test_verify_passes_on_a_clean_store_and_fails_on_a_tampered_one, R::test_replay_fails_loudly_on_a_store_that_does_not_reproduce | a flipped bit, a corrupt stream, an altered graph, an altered digest |
| NaN and infinity | S::test_nan_and_infinity_values_round_trip | NULL as NaN, inf kept, NaN equal to NaN on comparison |
| graph measurements (Prop. 13) | S::test_graph_measures_against_networkx, S::test_graph_records_are_stored_once_with_their_measurements | networkx shortest paths, components, the Laplacian, `statistics.pvariance`; graphs equal the reference rewiring |
| keys, directories, revisions | S::test_a_trajectory_key_of_another_experiment_is_refused, S::test_create_refuses_a_used_directory_and_open_refuses_a_non_store, S::test_revisions_must_extend_the_latest, R::test_run_summaries_and_progress | refusals with the field named; revisions through `check_extension`; summaries, progress calls, unit rows, status |
| one writer (Prop. 10) | S::test_a_second_writer_is_refused_with_a_message_naming_the_first, S::test_another_process_holding_the_lock_is_named_and_its_stale_lock_taken_over | same process, another process named by pid, takeover after it died |
| export | S::test_export_writes_open_tables_whose_aggregates_reproduce_from_the_rows, S::test_an_interrupted_export_leaves_nothing | aggregates recomputed from the exported rows and by hand; two exports identical file by file; no partial directory |
| protocol dimension | R::test_the_trajectory_key_carries_the_protocol, R::test_the_manifest_protocols_and_their_canonical_form, R::test_protocol_p_runs_the_same_circuits_without_the_reference, R::test_effective_checkpoints_and_revisions_per_protocol | strings, parsing, digests unchanged for F-only manifests, refusals, kwargs, R-CK per protocol |
| requirements and plan | R::test_requirements_name_what_a_trajectory_records, R::test_the_plan_covers_every_trajectory_once_within_the_unit_limits | every target trajectory in exactly one unit under four sizings; order; partial runs |
| one engine call per unit (v1.4 §8.1) | R::test_one_engine_call_per_unit_on_its_indices_with_the_v14_protocol | a spy: indices $u = g s + c$, the reference seed, every keyword, the collection, the probe schedules |
| failed units (Prop. 1) | R::test_a_failed_unit_writes_no_data_and_is_recorded | nothing of the unit, the failure recorded, the rerun completes |
| code-only stores | R::test_a_code_only_store_and_a_revision_that_adds_the_joint_tableaux | joint refused, then filled in; physical code unchanged |
| P and F (Prop. 12) | R::test_p_and_f_replay_the_same_gate_and_mask_history | exported circuits equal gate by gate and mask by mask, and equal the independent draw transcription |
| replay (validation 5) | R::test_replay_reproduces_the_stored_digests_and_gives_unsaved_depths, R::test_the_outcome_aware_replay_drops_its_signs_to_the_stored_row_space | stored digests verified; unsaved and beyond-horizon depths equal Stim; same outcome seed, same outcomes and signed groups; signs dropped give the stored space |
| circuit export (v1.4 §4 Tracking) | R::test_the_exported_circuit_of_a_stored_trajectory | every layer's gates, Clifford indices and measured sites against the reference draws; measurement count |
| retrospective observables (Prop. 7) | R::test_compute_reads_stored_snapshots_only_and_is_idempotent, P::test_recomputing_an_observable_draws_no_circuit_randomness | no simulation (patched engine), idempotent, values from stored tableaux; the sampled `mean_leakage` equal in another store and order |
| unit split, threads, order, resume (Props. 2, 3) | P::test_how_the_work_is_split_and_run_never_changes_the_content | one graph per unit and one thread; one trajectory per unit reversed on two threads; four resumed parts: equal $\Lambda$ and equal records |
| the digest (Prop. 11) | P::test_the_logical_digest_sees_every_change_of_content, S::test_the_logical_digest_ignores_compression_and_provenance | five kinds of changes each give a new digest; compression level and unit sizes do not |
| interruption (Props. 1, 3) | P::test_an_exception_before_a_commit_leaves_the_store_as_it_was, P::test_a_process_killed_inside_a_unit_transaction_leaves_a_consistent_store | exception after the inserts; `os._exit` inside the transaction in a subprocess; integrity, stale lock, resume, equal $\Lambda$ |
| extension (Prop. 6) | P::test_an_extension_leaves_every_record_unchanged_and_equals_a_fresh_run, P::test_a_deeper_horizon_fails_loudly_on_a_stored_prefix_that_does_not_reproduce | a rate, a size, a beta, graphs, circuits, protocol P, a deeper horizon and a depth: every record kept, current state equals a fresh store; a forged prefix refused |
| schedules (Props. 4, 5) | P::test_a_checkpoint_rule_change_cannot_change_a_circuit_or_a_stored_value, P::test_engine_probes_and_schedules_never_change_the_trajectory | exported circuits and stored values unchanged by a new rule; replays at other depth lists agree |

| graphs once, checked (G-GR-1) | G::test_graphs_are_generated_once_and_checked_in_every_unit | one generator call per $(n, \beta, g)$ over a multi-unit, multi-rate run reopened between units; a forged graph refused by the next unit, nothing changed |
| graph measurements (G-GR-2) | G::test_graph_measures_conventions_on_hand_cases_and_by_bfs | two $K_5$: 2 components, $\lambda_2 = 0$, distance 1 over 20 pairs; an isolated vertex; reachable pairs, components and mean distance against a BFS written in the test on 12 random graphs |
| the engine call (G-ENG-1) | G::test_every_unit_runs_exactly_the_v14_engine_call | every keyword of v1.4 §8.1, the gate set, the order and every probe's kind, name and schedule, F and P |
| versions, qubit order (G-TAB-1, G-ID-1) | G::test_the_versions_the_qubit_order_and_the_bell_pairs_are_recorded, G::test_a_store_of_another_randomness_contract_is_refused | the recorded tags and formats; $t = 0$ equals the golden digest of $\mathrm{span}\lbrace X_iX_{n+i}, Z_iZ_{n+i}\rbrace$ written out; another contract refused to write, replay and verify |
| outcome seed, Born law (G-TAB-2, SI-24; Prop. 14) | G::test_the_outcome_seed_is_a_tagged_function_of_the_whole_key, G::test_the_signed_replay_is_recorded_and_reproduced_from_the_identity, G::test_the_signed_replay_follows_the_born_law | the seed against hashlib and struct, distinct over every key field; recorded and compared; Stim follows the outcomes: determined ones equal the state's eigenvalue, random ones are fair (binomial $\lvert z\rvert < 4.5$, false alarm $6.8 \times 10^{-6}$) |
| statuses and $k = 0$ (G-STO-2, SI-05, SI-07, SI-08) | G::test_statuses_and_the_conventions_at_k_zero, G::test_the_store_says_why_a_value_is_missing | raw $d_{\mathrm{cyc}}$ undefined with zero extension, $n + 1$ kept raw and $D = 0$, censored and missing statuses, absent retrospective rows; refusals |
| append-only (G-RUN-1) | G::test_stored_records_cannot_be_updated_or_deleted | twelve `UPDATE` and `DELETE` statements through SQLite refused by the triggers |
| one schedule, reruns first (G-RUN-2) | G::test_reruns_come_first_and_every_trajectory_holds_one_schedule | a rate and depths added: reruns planned first, identical depth sets and schedules for old and new rates |
| provenance (G-RUN-3) | G::test_units_record_their_provenance, G::test_a_confirmatory_study_keeps_its_environment | contract, environment and its digest per unit; a changed numpy refused under `confirmatory`, recorded under `exploratory` |
| peak domain (G-OBS-2; Prop. 15) | G::test_peaks_do_not_depend_on_the_revision_history | peaks and first passages equal in a revised and a fresh store |
| peaks as the catalog says (doc 04 §4.6, Props. 18(d), 20) | K::test_stored_peaks_record_their_errors_and_the_catalog_bootstrap_seed, K::test_the_version_record_carries_the_bootstrap_seed | `peak_D` and its sensitivity `se_exact` True, its bootstrap seed `bootstrap_seed(key, "peak_D", version)`, and the bootstrap rebuilt from the stored orderings, counts and $k$ with it equals the stored one (the earlier `region_seed` derivation does not: all 12 trajectories differ, the test asks for 6); `peak_D_J` `se_exact` False for the chain on the interaction graph, True for exact samples on a cycle; the bootstrap seed's tag and length recorded, a store without them read but not continued |
| aggregates (G-STO-4) | G::test_aggregates_are_recorded_with_their_sample_set | the record against a direct computation; recompared; more graphs give a new record; refusal and `omit` on undefined values |
| sampler diagnostics (G-OBS-4) | G::test_the_correlated_samplers_diagnostics_are_archived | one row per graph, size and coupling; idempotent; equal to the graph-side tool with the recorded seed |
| probe streams across rates (G-ID-2) | G::test_probe_streams_are_shared_across_rates_unlike_the_retrospective_regions | the oracle's transcription of the probe stream reproduces both rates' first-failure samples from their stored codes; region seeds differ |

A mutation check (scratchpad `t3_mutate.py`): 11 single faults in `store.py` and `runner.py`
(prefix not compared, $g$ and $c$ swapped, values left out of the digest, snapshot identity not
compared, replay not verifying, a circuit dropped from the plan, unbounded reads, a value not
compared, another run seed, no digest check on read, completions under the wrong requirements),
each made a test fail.

## 9. Limitations and departures

Refinements of the design §4 schema (its interfaces are kept; the tables are internal):

* **S1. Traces per horizon.** The per-layer record (`k_points`, `tau`, `final_layer`, status)
  lives in `traces`, keyed by `(tid, layers)`, instead of columns of `trajectories`: raising
  $T_{\max}$ adds a trace and keeps the old one (v1.4 §5.3 "instead of overwriting"; the
  requirements matrix's "re-derived without silently overwriting the originals"). The checkpoint
  bounds $\widehat\tau^{\mathrm{chk}}$ depend on the schedule, so they are in the `first_passage`
  observation of each requirements context, and `store.first_passage` recomputes them from the
  stored trace on demand.
* **S2. `observations.context`.** Trajectory-level values depend on the horizon and the
  schedule; the context (the requirements digest) keeps the values of each stage apart, so none
  is overwritten.
* **S3. A store is a directory** (`study.sqlite` and the lock), so the lock of `sparsegf2.api`
  is reused unchanged.
* **S4. `requirements` and `completions`** (not in the design) make "done" a property of a
  trajectory under explicit requirements, which is what a revision changes.
* **S5. The joint fill-in** (§3.2) is the one update of a data column.
* **S6. Recomputing a value the run stored** compares the value, its error and its status only
  (the run's method and details may differ).
* **S7. Append-only by triggers** (G-RUN-1). Changes from outside SQLite are not prevented, only
  detected (digests on read, `check`, `verify`, the logical digest).
* **S8. Observation statuses** (G-STO-2) are a column the design did not have.
* **S9. The peak domain** (G-OBS-2): v1.4 §5.8 takes peaks over $\mathcal T(n, \beta)$; under the
  R-CK unions the store holds $E_r$, and peaks and checkpoint bounds are taken over the last
  revision's own schedule $S_r$, recorded with its digest. A different domain (the union) would
  be another parameter, not computed here.
* **S10. A new seed domain**, WSO1 (`0x57534F31`), for the outcome-aware replay (Proposition 14),
  and WSSD (`0x57535344`) for the sampler diagnostics; both recorded in `meta.versions`.
* **S11. Graph measurements** are a versioned JSON record per graph, not one column each.

Not done here (the requirements review proposed them; they belong to other parts or need a
decision):

* the environment of a confirmatory study is compared with the units already run, not with a
  lock pinned in the manifest beforehand (G-RUN-3), and there is no script that clones the
  recorded commit into a fresh environment;
* the sampler diagnostics are archived on request (`diagnose_samplers`), not before every unit,
  and a unit is not flagged when their notes fire (G-OBS-4: their cost grows with graphs times
  sizes); the error of $D(J > 0)$ remains the binomial one of doc 04 (an underestimate for the
  chain);
* golden seed tuples in a data file and the protocol arguments of `stats` and `fss` (identity,
  statistics); the CLI forms of export (`--tableaux`) and the reproduction script;
* comparison of the graph collections' statistics with the exact law and networkx samples
  (graph conformance, doc 01).

Not stored, or not this part:

* **Region samples** (v1.4 §8 `region_sample`): the engine returns failing fractions, not the
  regions; they are reproducible from the keyed probe streams. Per-chain diagnostics of the
  correlated sampler are not available in-run (doc 04 §9); the `sampler` data describe the
  sampler of the trajectory's own graph.
* **Per-layer records** (`layer`): on demand from `runner.circuit`, exactly the gates and
  measured sites (v1.4 §8.1 "export the exact circuit for audit").
* **Aggregates** are written by the export, not stored; `rng_validation` belongs to the
  demonstration task.
* **Measurement replicas** $r > 0$ (design D1): the key keeps $r$; the runner writes $r = 0$.

Costs and conventions:

* The per-trajectory bootstrap of $T^*$ dominates ingestion (§5.8); `analysis.peak.bootstrap = 0`
  removes it. The peak rule is part of the requirements, so changing `analysis.peak` in a
  revision reruns every trajectory (every stored value is verified, new peak rows are added).
* Ingestion is single-threaded Python after the parallel engine call and canonicalization; at
  $n \le 128$ it costs about 10 to 30 times the engine call (§5.8).
* $\lambda_2$ of a connected graph is a LAPACK eigenvalue: its last bits may differ between
  platforms. Graph records are compared by seed, generator and edge digest, never by their
  measurements.
* A store opened for writing must have been made with the code's versions (randomness contract,
  generator, Clifford numbering, dynamics, row-space format, seed tags); a store of another
  dynamics version is refused rather than mixed (v1.4 §8's `(trajectory key, dynamics_version)`
  is met by this policy).
* SQLite's WAL needs shared memory between processes on one machine: a store on a network share
  should be written and read from one machine.
* The retrospective region seeds (`observables.region_seed`) do not contain the protocol; a
  protocol-P state has no code ($k = 0$), so its sampled code observables draw nothing.
* `DEFAULT_UNIT_BYTES` (1 GiB) bounds the engine's buffers as estimated by
  `_bytes_per_trajectory`; the canonical forms of a unit's snapshots add about as much again
  while it is ingested.
