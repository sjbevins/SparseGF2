# Design: the Watts–Strogatz monitored-Clifford study pipeline (spec v1.4)

The authoritative scientific specification is `docs/misc/watts_strogatz_study_spec_v1_4.md`
(cited below as "v1.4 §x"). This document fixes how SparseGF2 implements it: the architecture,
the interfaces between parts, the storage schema, and every place where the implementation
had to choose or to depart. A departure from v1.4 is listed in §9 with its reason; nothing
else may change a probability law, an observable's definition or the native RNG.

## 0. Audit of the current code (what v1.4 relies on)

| v1.4 item | Current code (verified by reading) | Consequence |
|---|---|---|
| WS law, §2.2 | `graphs.watts_strogatz`: edges `_ring_lattice(n, 4)` offset-major (`d = 1, 2` outer, `i` inner), retained endpoint `a = i`; `next_double < beta` decides; target by rejection `next_below(n)` until `w != a` and `w` not adjacent to `a` (eligible set taken before deleting `{a, b}`; `b` is adjacent so excluded); skips when `deg(a) >= n - 1`. `from_edges` gives the canonical sorted list (`a < b`). | Matches §2.2 exactly (uniform target by rejection). It does not check that the visited edge is present; that edge is provably present for `n >= 5` (an original edge is removed only at its own visit, and an added edge never duplicates a present one), so the "abort" of §2.2 lives in the independent reference rewiring of the tests and in the study's graph validation, not in `graphs.py`. `n >= 5` only (`k < n`); `n = 5` is `K_5`, never rewired. |
| Gates, §3 | `_kernel/ensemble.py::gen_layer`: `schedule="random_edges"`, `gates_per_layer` default `n // 2`, per gate `next_below(|E|)` edge, optional `next_bit` orientation (off by default), `pool[next_below(len(pool))]` Clifford (pool = all 11,520 by default); then one `next_double < p` per site in ascending order. | Matches §3, §4, §7.3 with `random_orientation=False`, unweighted edges, no scramble. |
| Streams, §7 | Trajectory `t` of a run draws its circuit from stream `stream_state(key_struct, t)`; probe evaluations from `stream_state(key_probe, mix(t, phash, L))` (`RANDOMNESS_CONTRACT = 2`). Graph of trajectory `t` in a collection: `t // circuits_per_graph`. Run seed: `np.random.SeedSequence(seed)` (int or int sequence). | Prefix, checkpoint, thread and chunk invariance already hold and are tested; `u = g C + c`. |
| Tableau | Engine tableau `tab[2q]` = X bits of qubit `q`, `tab[2q + 1]` = Z bits, over rows packed in 64-bit words; `ref_rows` marks rows with reference support. `code_snapshot` writes the rows outside `ref_rows` (the physical code). | `code_snapshot` is the physical code only (v1.4 §8.1). A joint snapshot needs a new probe that writes all rows over all `2n` qubits. |
| Oracles | `tests/_reference_rng.py` (pure-Python xoshiro256**, splitmix64, Lemire), `tests/_protocol_oracle.py` (independent circuit draws and text). | Reused for the independent graph and circuit references. |

## 1. Architecture

```
sparsegf2/
  _kernel/ensemble.py      + PROBE_TABLEAU_SNAPSHOT (all rows, all qubits); trajectory indices as an array
  _kernel/rowspace.py      NEW  GF(2) row-space kernels: canonical RREF of packed rows, ranks of column subsets
  mipt.py                  + tableau_snapshot() probe; run(indices=...); EnsembleResult.tableau()
  tableau.py               NEW  PhaseFreeTableau: canonical row space, digest, serialization, entropies, I(A:R),
                                physical code extraction, signed-tableau comparison helpers
  study/
    __init__.py            public names
    identity.py            rate/beta tokens, trajectory keys, seed hierarchy, graph records and digests, versions
    manifest.py            ExperimentManifest: strict validation, canonical JSON, digest, checkpoint rule (v1.4 §9),
                           revision compatibility (what an extension may change)
    observables.py         observable catalog (registry with ids, versions, levels, inputs, definitions) and the
                           pure functions: k-trace compression, first passage tau_ell, d_cyc, D(J), peaks, ...
    store.py               TableauStore: SQLite index + compressed blobs (schema §4)
    runner.py              work units, execution through MonitoredCircuit, ingestion, resume, extension, replay,
                           verification, retrospective observables
    stats.py               nested equal-graph-weight estimators, graph-cluster errors, paired rate differences,
                           censoring (weighted Kaplan-Meier, restricted mean), bootstrap over graphs
    fss.py                 dynamic finite-size scaling: collapse cost, bounded Nelder-Mead, fits per beta, shared vs
                           independent exponents, leave-one-size-out, graph-cluster bootstrap paired across p
    plots.py               diagnostic plots (matplotlib imported lazily)
    cli.py                 `sparsegf2 study ...` subcommands
    demo.py                the demonstration manifest and its end-to-end run and determinism comparison
```

Compiled code stays in `_kernel/`. The study package never simulates by itself: every
trajectory runs through `MonitoredCircuit.run` (phase-free engine).

## 2. Engine extensions (task 1)

1. **`mipt.tableau_snapshot(name=None, *, checkpoints=None)`**, kind `"tableau_snapshot"`: at
   each of its evaluations, the full phase-free stabilizer tableau of the joint state: every
   generator row over every qubit (system `0..n-1`, then reference `n..n+n_ref-1`). Value: the
   number of rows. It is a snapshot probe: at most one snapshot probe (`code_snapshot` or
   `tableau_snapshot`) per spec, so the kernel keeps its single snapshot buffer, sized
   `(trajectories, evaluations, N, 2 wq)` with `N` the total qubits and `wq = ceil(N / 64)`,
   row layout: X bits of qubits `0..N-1` in words `[0, wq)`, Z bits in `[wq, 2 wq)` (the
   `code_snapshot` layout, over all qubits). Kernel: `PROBE_TABLEAU_SNAPSHOT = 12` and a
   `joint_snapshot_rows(tab, n_qubits, n_rows, out)` kernel. It draws no random numbers, so no
   stream changes (pinned digests unchanged). After `stop_when_purified` stops a trajectory,
   later evaluations are marked `-1` (not evolved), as for `code_snapshot`.
2. **`MonitoredCircuit.run(..., indices=None)`**: an explicit array of distinct non-negative
   trajectory indices (any order) instead of `first_trajectory`/`trajectories`. Trajectory
   `u` is exactly the trajectory a range run would give it (graph `u // circuits_per_graph`,
   circuit stream `u`, probe streams keyed by `u`). The kernel takes an int64 index array in
   place of `first`. `result.trajectories` holds the indices in the given order.
3. **`EnsembleResult.tableau(trajectory, evaluation, name=None) -> PhaseFreeTableau`**, save,
   load, concatenate and replay for the new probe; `EnsembleResult.code(...)` on a tableau
   snapshot returns its physical code.
4. **`sparsegf2.tableau.PhaseFreeTableau`** (pure Python + `_kernel/rowspace.py`):
   * built from engine words (`from_words(words, num_qubits)`), a dense 0/1 matrix
     `(rows, 2N)` (X bits then Z bits), or Pauli strings;
   * `canonical()`: the reduced row echelon form over GF(2) with columns in the order
     `x_0..x_{N-1}, z_0..z_{N-1}`, pivots leftmost, zero rows dropped. Unique per row space;
   * `to_bytes()` / `from_bytes()`: header `b"SGF2RS1\0"`, `N` (uint32 LE), rank (uint32 LE),
     then the canonical rows, each `np.packbits(row, bitorder="big")` over its `2N` columns;
   * `digest()`: SHA-256 hex of `to_bytes()` (the canonical phase-free row-space digest of
     v1.4 §5, independent of the generator basis and of the backend);
   * `rank`, `num_qubits`, `is_isotropic()` (all rows commute), `same_space(other)`;
   * `entropy(region)` = `rank(S|_region) - |region|` for a pure state (bits), `mutual_information`;
   * `physical_code(num_system)`: the rows of the row space with no support on qubits
     `>= num_system` (elimination with reference columns first), as a `StabilizerCode`;
     its rank is `r_t` and `k_t = num_system - r_t` for the purification picture.

## 3. Identity, seeds and the manifest (task 2)

* **Tokens.** `beta` and `p` are given as decimal strings, rational strings (`"1/7"`) or JSON
  numbers. A number is read through its shortest round-trip decimal (`repr`). The value is
  held as an exact `fractions.Fraction`; its token is the canonical decimal (no exponent,
  trailing zeros stripped, `"0"` and `"1"` plain) when the denominator divides a power of 10,
  else `"num/den"`. The simulator gets `float(Fraction)` (correctly rounded); records keep the
  token, the float64 and `float.hex()`.
* **Trajectory key** `(experiment_id, n, beta_token, p_token, g, c, r)`, `r = 0` only (§9 D1).
* **Seed domains (ruling R-SEED, supersedes the first draft's shared `(master, g)`).** v1.4 couples
  conditions only across rates `p` (§1, §6.2, §7.2: "for a fixed `(n, beta)` choose `graph_seed(g)`
  by appending `g` to a versioned graph-seed namespace"). Sharing seeds across `n` or `beta` would
  correlate data that finite-size fits treat as independent (the same raw draws feed
  `next_below(|E|)` at different `|E|`, and the first rewiring decisions coincide). So every seed is
  a fixed-length tuple of 32-bit words with a domain tag and the condition:
  `w(x)` = the two 32-bit words (low, high) of `int.from_bytes(blake2b(x.encode(),
  digest_size=8).digest(), "little")` (the repository's probe-name convention);
  `graph_seed = (GRAPH_SEED_TAG, master_graph_seed, n, *w(beta_token), g)`,
  `run_seed = (CIRCUIT_SEED_TAG, master_circuit_seed, n, *w(beta_token))` for `gate_mode="shared"`
  (one run seed per `(n, beta)`, the same at every `p`: the native coupling), and
  `(CIRCUIT_SEED_TAG, master_circuit_seed, n, *w(beta_token), *w(p_token))` for
  `gate_mode="independent"`. The tags (`GRAPH_SEED_TAG = 0x57534731`, "WSG1";
  `CIRCUIT_SEED_TAG = 0x57534331`, "WSC1") make the two domains disjoint whatever the master seeds,
  so equal master seeds are allowed; within a domain the tuples have one fixed length, so the zero
  padding of `SeedSequence` cannot make two of them equal. Different `(n, beta, g)` give different
  seeds. Graph record: `n, beta_token, g`, the seed tuple, `GRAPH_GENERATOR_VERSION`, canonical
  sorted edges, `edge_digest` = SHA-256 of `b"SGF2EDGES1\0" + n (uint32) + edges (int32 LE,
  row-major)`.
* **Stream index** `u(g, c) = g * stride + c` with `stride = circuit_index_stride` frozen in the
  manifest (default: `circuits_per_graph`); the engine runs with `circuits_per_graph = stride`
  and `indices = [u(g, c) for the requested (g, c)]`, so graph `u // stride = g`. With
  `stride > C`, circuits `C..stride-1` can be added later without renumbering (v1.4 §7.2's
  `u = gC + c` with `C = stride`). Adding graphs never renumbers.
* **Run seed** (R-SEED above): shared mode, one per `(n, beta)` at every `p` (gates and mask
  uniforms coupled across rates, masks nested); independent mode, one per `(n, beta, p)` (gates and
  masks independent across rates). Master seeds are below `2**32` (one word each).
* **Versions** (strings, recorded per experiment and per record): `GRAPH_GENERATOR_VERSION`,
  `CLIFFORD_ENUMERATION_VERSION` (`j = 16 s + b`, v1.4 §9.1), `DYNAMICS_VERSION`,
  `ROWSPACE_FORMAT_VERSION`, `STORE_SCHEMA_VERSION`, one per observable.
* **Manifest** (`ExperimentManifest`, JSON in, frozen dataclasses, unknown keys refused with
  the allowed ones listed):

  ```json
  {"schema": "sparsegf2.study.manifest/1", "experiment_id": "ws-pilot-001",
   "label": "exploratory",
   "conditions": [{"n": [16, 24], "beta": ["0", "0.1"], "p": ["0.1", "0.15", "0.2"]}],
   "depth": {"layers": {"per_n": 8}, "checkpoints": {"count": 12}},
   "trials": {"graphs": 20, "circuits_per_graph": 2, "circuit_index_stride": 8,
              "measurement_replicas": 1},
   "seeds": {"master_graph_seed": 17, "master_circuit_seed": 29, "gate_mode": "shared"},
   "dynamics": {"stop_when_purified": false},
   "observables": {"tableau_snapshots": true, "contiguous_distance": true,
                   "uniform_erasure": {"samples": 200, "exact_budget": 5000},
                   "correlated_erasure": [{"couplings": [0, 1, 2], "sizes": null, "samples": 100,
                                           "thin": 20, "geometry": "interaction"}]},
   "storage": {"graphs_per_unit": 8, "compression_level": 6},
   "resources": {"num_threads": null, "max_unit_trajectories": 4096},
   "analysis": {"peak": {"rule": "plateau", "z": 1.0, "sensitivity_z": [0, 1, 2],
                         "bootstrap": 200},
                "bootstrap": 1000, "censoring": "survival"}}
  ```

  `conditions` is a list of blocks; each block is the product of its lists, so
  `P(n, beta)` may differ between blocks. `layers`: `{"per_n": a}` (`T_max = a n`), `{"fixed": T}`
  or `{"table": {"16": 128, ...}}`. `checkpoints`: `{"count": N_T}` (the v1.4 §9 geometric rule:
  `a_j = exp(j ln T_max / (N_T - 1))`, `j = 0..N_T-1`, rounded to the nearest integer with ties
  upward, clamped to `[0, T_max]`, deduplicated, sorted, with `0, 1, T_max` included; for
  `N_T = 1` just `{0, 1, T_max}`), `{"list": [...]}` (one list for every size: the entries above a
  size's `T_max` are dropped for that size, and `0, 1, T_max` are always included) or
  `{"table": {"16": [...], ...}}` (one list per size, every size covered). Validation: `5 <= n` (warning at
  `n = 5`), rates and `beta` in `[0, 1]`, `measurement_replicas == 1` (§9 D1), seeds
  non-negative ints, stride `>= circuits_per_graph`, `tableau_snapshots` with
  `stop_when_purified = true` allowed (later snapshots then recorded as absent, §4) but warned.
  `digest()` = SHA-256 of the canonical JSON (sorted keys, no spaces, tokens not floats).
* **Revisions.** A store keeps every manifest revision. A new revision may add conditions,
  rates, graphs, circuits (up to the stride), raise `T_max` (never lower it), change the
  checkpoint rule, add observables; it may not change seeds, `gate_mode`, the stride, the
  protocol, or the settings of an observable already recorded (a changed setting is a new
  observable version or name). `manifest.check_extension(old, new)` names the first forbidden
  change.
* **Checkpoints across revisions (ruling R-CK).** Revision `r` has its own schedule
  `S_r(n, beta)` from its own rule and `T_max`. The schedule used for work under revision `r` is
  the union `E_r(n, beta) = (S_1 ∪ ... ∪ S_r)(n, beta) ∩ [0, T_max_r(n, beta)]`; since `T_max`
  never decreases, `E_r ⊇ E_{r-1}`: a stored depth is never dropped, and raising `T_max` under the
  geometric rule adds the new rule's depths to the old ones.
  `manifest.effective_checkpoints(revisions, n, beta)` computes it.
* **Graph lookup (engine constraint).** The engine finds trajectory `u`'s graph as
  `edges[u // stride]`, so a unit for graphs `g0..g1-1` passes the collection `0..g1-1` (graphs
  below `g0` are never run in that unit but keep the indexing).

## 4. Store (task 3): `TableauStore`, one SQLite file per experiment

SQLite (standard library) in WAL mode gives transactions, indexes, bounded-memory queries and
crash safety; arrays are compressed blobs (zlib) in it. One writer at a time (a lock file
naming its process, taken over when that process is gone); readers any time.

```
meta(key PRIMARY KEY, value)                    schema version, experiment_id, created, versions
manifests(revision PK, digest UNIQUE, json, created_utc, sparsegf2_version, git_commit)
graphs(n, beta, g, seed, generator, num_edges, edges BLOB, digest, PK(n, beta, g))
units(unit_id PK, n, beta, p, graphs TEXT, circuits TEXT, layers, status, revision, started_utc,
      finished_utc, elapsed_seconds, threads, trajectories, error)
trajectories(tid INTEGER PK, n, beta, p, g, c, r, u, p_float_hex, run_seed TEXT, graph_digest,
      layers, final_layer, k_points BLOB, tau BLOB, tau_chk BLOB, tau0, tau1, censored0, unit_id,
      revision, dynamics_version, UNIQUE(n, beta, p, g, c, r))
snapshots(tid, t, num_qubits, rank, k, joint BLOB, joint_digest, phys BLOB, phys_digest,
      status, PK(tid, t))                       status: "stored" | "absent_after_purification"
observations(tid, t, observable, version, params, value REAL, se REAL, data BLOB, method,
      PK(tid, t, observable, version, params))  t = -1 for trajectory-level observables
observable_defs(observable, version, params, params_json, level, inputs, definition,
      created_utc, PK(observable, version, params))
```

* A **work unit** is `(n, beta, p, graphs g0..g1-1, circuits 0..C-1, T_max)`; it runs in one
  engine call (`indices`), then is ingested in **one transaction** that also marks the unit
  done. An interrupted unit leaves nothing (rolled back) and is rerun on resume. Rows are
  never overwritten: an existing identity is compared (digests, k trace) instead.
* **k trace** stored as change points `(t_j, k_j)` (int32 pairs, zlib), lossless (`k` is
  nonincreasing); `tau` the `n + 1` exact first-passage depths (`-1` when censored) and the
  censor depth; `tau_chk` the checkpoint upper bounds.
* **Snapshots**: the canonical joint row space (`PhaseFreeTableau.to_bytes`, zlib) and its
  digest, and the canonical physical code (`physical_code` of the joint) and its digest.
* **Retrieval**: `store.tableau(n=, beta=, p=, g=, c=, t=)`, `store.code(...)`,
  `store.k_trace(...)`, `store.first_passage(...)`, `store.iter_snapshots(where...)` (cursor,
  bounded memory), `store.observations(observable, ...)` as arrays with `(g, c, r)` columns.
* **Replay**: `runner.replay(store, key, depths)` reruns one trajectory through
  `MonitoredCircuit.run(indices=[u])` with a `tableau_snapshot` at `depths` (and
  `stop_when_purified=False`), returns the tableaux, and verifies them against stored digests
  where they exist. `runner.replay_signed(...)` reruns the exported `spec.circuit(u)` on
  `Simulator(2n, phases=True)` (outcome-aware mode) and checks that dropping signs gives the
  stored row space.
* **Extension**: raising `T_max` reruns the affected trajectories from layer 0 to the new
  horizon (the native stream has no random-access layer state), checks that the stored prefix
  (k trace, snapshot digests) is reproduced exactly, then adds the new depths.
* **Retrospective observables**: `runner.compute(store, observable, params)` reads snapshots
  in batches, computes a registered snapshot observable (§5), and inserts rows under its own
  `(observable, version, params)`; existing rows are never changed.
* **Export**: `store.export(directory)` writes per-condition tables (CSV and NPZ) of the
  observations and the aggregates, with the manifest and versions.

## 5. Observables (task 4), `study/observables.py`

Registry entries: `id`, `version`, `level` (`snapshot` | `trajectory` | `aggregate` |
`scaling`), `inputs` (`k_trace` | `joint_tableau` | `physical_code` | `graph` | engine probe),
`params`, `definition` (the formula, v1.4 section). Built-ins:

| id | level | definition (v1.4) | computed from |
|---|---|---|---|
| `k` / `code_rate` / `stabilizer_rank` | snapshot | §5.2 `k_t`, `k_t/n`, `r_t = n - k_t` | k trace (every layer) |
| `first_passage` | trajectory | §5.2 `tau_ell = min{t : k_t <= ell}`, `ell = 0..n`, censored at `T_max`; `tau_chk` | k trace |
| `d_cyc` | snapshot | §5.5 first failing cyclic interval; raw `NaN` at `k = 0`, `has_code`, zero-extended | engine `contiguous_distance` (and from a stored code) |
| `uniform_failure` | snapshot | §5.7 `P_C(m)` (hybrid: exact counts `m <= w`, first-failure orderings above) | engine `erasure_first_failure` + `uncorrectable_region_counts` |
| `D` (J = 0) | snapshot | §5.7 `D(0) = sum_{m<n} (1 - P(m)) = E_pi[M_pi]`, with standard error | first-failure samples |
| `correlated_failure` / `D_J` | snapshot | §5.7 `P_{C,G,J}(m)`, `D(J)`; interpolated sizes flagged | engine `correlated_erasure` |
| `mean_leakage` | snapshot (retrospective) | §5.7 `l(m, J) = E_Q[I(A:R)]` | stored code + graph, keyed region sampler |
| `d_conn` | snapshot (retrospective) | §5.6, exact enumeration, guarded `n <= 20` | stored code + graph |
| `half_cut_entropy` | snapshot (retrospective) | `S(A)` of system qubits `0..n/2-1` from the joint state | stored joint tableau |
| `peak_<trace>` | trajectory | §5.8 `d*`, maximizing set `A`, `T* = exp(mean_{A} ln t)`, horizon flag, `k_t` on `A` | stored traces |

Peaks: exact traces (`d_cyc`, integers) use exact ties. Monte Carlo traces (`D` from samples)
use the predeclared plateau rule of the manifest: `A = {t : d(t) >= d* - z sqrt(se(t)^2 +
se(t*)^2)}`, reported at every `z` of `sensitivity_z`, plus a bootstrap of `T*` that resamples
each checkpoint's orderings. `T*` missing when `d* = 0`; `at_horizon` when `T_max` is in `A`.

## 6. Statistics (task 5), `study/stats.py`

Inputs: arrays `g, c, r, value` (and `observed` for censored times). `nested_mean` (v1.4 §6.3,
equal graph weight whatever the allocation), `graph_cluster_se` (§6.4), `paired_difference`
(within-graph differences across rates, then the cluster error), `weighted_kaplan_meier`
(graph-equal weights) with median and restricted mean to the horizon, `censoring_summary`,
`graph_bootstrap` (resamples graph indices once and applies them to every rate and observable,
so pairing survives). Never reports a plain observed mean of a censored time as the ensemble
mean.

## 7. Finite-size scaling (task 6), `study/fss.py`

Model (v1.4 §5.9): `E[theta | n, beta, p] ~ n^z F((p - p_c) n^(1/nu))`. Cost: the collapse
quality of Houdayer and Hartmann (PRB 70, 014418, 2004): each point is compared with a local
linear interpolation of the other sizes' points at the same scaled abscissa, only where they
bracket it (no extrapolation), weighted by the combined errors; the cost is the mean squared
normalized deviation (about 1 for a perfect collapse within errors). Optimizer: Nelder–Mead in
an unbounded reparametrization of box bounds, with multi-start; written here and tested (no
SciPy). Fits per fixed `beta`: independent `(p_c, z, nu)` per observable, or shared across
selected observables; comparison by the cost difference against its bootstrap distribution;
leave-one-size-out refits; graph-cluster bootstrap paired across `p`; diagnostics: points
used, bracketing coverage, censoring and horizon-peak fractions per point, drift of fits with
the smallest size removed. Exploratory and confirmatory runs are kept apart by the manifest
`label`.

## 8. CLI and demonstration (tasks 7, 8)

`sparsegf2 study init MANIFEST STORE`, `run STORE [--threads N] [--max-units K]`, `status`,
`extend STORE MANIFEST`, `tableau STORE --n --beta --p --g --c --t [--replay] [--out FILE]`,
`compute STORE OBSERVABLE [--params JSON]`, `aggregate STORE [--out DIR]`, `fss STORE ...`,
`export STORE DIR`, `verify STORE [--sample K]` (replay and compare digests), `demo DIR`.
`--json` everywhere, exit codes as the rest of the CLI.

## 9. Departures and exceptions (each also in the final report)

* **D1. Measurement replicas `r > 0`** are not implemented: the native engine draws gates and
  masks from one stream (v1.4 §8.1 marks it "extension required if r > 0 is used"; the
  baseline is `r = 0`). The manifest refuses `measurement_replicas != 1` with this reason. The
  key keeps `r` so a future versioned extension fits.
* **D2. Circuit index stride.** `u = g * stride + c` with a frozen `stride >= C` (default `C`),
  so adding circuits later never renumbers (v1.4 §7.2 with `C = stride`).
* **D3. Depth extension** reruns from layer 0 (no serialized mid-stream state exists in the
  native engine); prefixes are verified, not assumed.
* **D4. `gate_mode="independent"`** derives the run seed from the rate token, which also makes
  masks independent across rates (the native stream couples gates and masks).
* **D5.** The "abort on an absent original edge" check (v1.4 §2.2) is in the reference
  rewiring and the graph validation, since it is unreachable in the native generator (§0).

## 10. Gate sampling over the 720 symplectic classes, and architecture-aware performance

**Uniformity over the 720 classes (owner's request, 2026-10-17).** The native sampler draws the projective
Clifford index `j = 16 s + b` uniformly from `[0, 11520)` with Lemire's unbiased `next_below(11520)`. Since
`11520 = 16 * 720` and each symplectic class `s = j >> 4` has exactly 16 indices, `Pr(s) = 16 / 11520 = 1 / 720`
exactly (not only in the ideal law: the integer draw itself is exactly uniform). In phase-free mode the kernel
applies a gate only through `LIN2[j >> 4]` (`_kernel/tableau.py::apply_2q`; the sign bits `j & 15` enter only the
sign row, which is not tracked), so every phase-free quantity depends on the gate only through its class: the
phase-free ensemble IS uniform sampling of the 720 elements of `Sp(4, F_2)`. Drawing `s` directly with
`next_below(720)` would give the same law but new realizations for the same seeds (a new randomness contract,
forbidden by v1.4 §9.1 without a new protocol version) and no speedup (one bounded draw either way). Tests: the
class of every index is its symplectic matrix (`clifford.py`); replacing every `j` by `16 (j >> 4)` leaves every
phase-free tableau bit-identical; the classes of sampled gates are uniform on 720 (statistical, exports).

**Performance (task T9).** Speed comes from measured bottlenecks, not assumptions. On the owner's Ryzen 9 9950X3D
(16 cores, 32 threads; one CCD with 96 MB of 3D V-Cache L3, one with 32 MB; 1 MB L2 and 48 KB L1d per core) the
per-trajectory working set is the tableau, `4N` columns of `ceil(N/64)` words for `N = 2n` qubits: `n^2 / 2`
bytes (32 KB at n = 256, 512 KB at n = 1024, 2 MB at n = 2048). Below about n = 1400 it fits in L2 and the V-Cache
cannot help a single trajectory; above, threads on the 32 MB CCD spill to memory first. T9 measures the cost
breakdown (gates, measurements, probes, snapshots), thread count and placement (all 32, one per core, the V-Cache
CCD only, the other CCD only), scheduling of uneven trajectories (`stop_when_purified`), and kernel
micro-optimizations (a compact 720-entry gate table that stays in L1), then implements what measurably pays
behind a `performance` setting (CPU topology detection, affinity presets, scheduling), with every setting proven
and tested to leave every result bit-identical.

## 11. Later rulings (2026-10-17)

* **D6 / R-SEED.** Graph and run seeds are domain-separated and depend on `(n, beta)` (design §3),
  instead of the native collection convention `[seed, i]` shared across conditions: v1.4 couples
  only across `p`, and finite-size fits need sizes (and `beta` values) to be independent samples.
  The graphs are still generated by the native `graphs.watts_strogatz` (only the seed tuple is the
  study's), and within one `(n, beta)` the native collection mapping `u = g * stride + c` holds.
* **R-CK.** Checkpoint schedules across revisions are unions (design §3).
* **R-FSS.** Besides the Houdayer-Hartmann collapse quality (continuous in the parameters between
  changes of bracketing, discontinuous where a point's bracketing pair changes), `fss` offers a
  continuous objective: a weighted least-squares fit of a polynomial master curve `F` of chosen
  degree jointly with `(p_c, z, nu)` on the points whose scaled abscissa lies in the fitted window;
  the two methods' agreement is a reported diagnostic. Its bootstrap re-selects the window inside
  every replicate by default, so the window's own variability enters the error (ruling, after the
  measured 0.68 coverage ratio for `nu` with a fixed window).

## 12. Forward compatibility with the queued MIPT study (v0.3)

The owner queued `docs/misc/watts_strogatz_mipt_focused_program_v0_3.md`, to start after this
implementation. Its infrastructure needs are built now so the store needs no migration later; its
observables and campaign are the queued task.

* **Two protocols** (v0.3 §2). `F` (full reference: `reference="purification"`, the v1.4 protocol)
  and `P` (pure product `|0...0>` on the system only: `reference="none"`). The trajectory key is
  `(experiment_id, protocol, n, beta_token, p_token, g, c, r)`; the store's tables carry a
  `protocol` column (default `"F"`). P and F with the same graph, run seed and stream index `u`
  have the same ordered gate and mask history (the engine draws nothing reference-specific for
  these two modes; a test proves it on exported circuits). P snapshots are the `n`-qubit pure state
  (`tableau_snapshot` works in any reference mode, `N = n`); `stop_when_purified` does not apply.
  The manifest gains `protocols` (default `["F"]`) and per-protocol checkpoint and observable
  settings.
* **Graph measurements** (v0.3 §4), computed once per graph record from the edge list alone and
  versioned: `q_G = |E(G) \ E(C(n,2))| / (2n)`; the mean shortest-path distance over reachable
  distinct vertex pairs and the component count; `lambda_2` of the unnormalized Laplacian `D - A`
  (0 when disconnected); the population variance of the degrees.
