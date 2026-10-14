# Audit of SparseGF2 v2 (as of commit `2d92a76`)

This is the baseline for the v3 rebuild. It records what the existing code does,
what is good in it, and what is wrong with it, with file:line references into
`src/sparsegf2/` and `studies/`. Every bug listed under "Confirmed defects" was
reproduced or read directly in the code, not inferred.

**Short version.** The physics engine at the core of v2 is correct, and it was
checked carefully against Stim, but only for the sign-free (phase-free) part of
the state. Around it the package grew three layers (circuits, analysis, studies)
that are hard to use and mostly bypassed: your actual production campaign
re-implements the simulation loop in ~150 lines on top of *private* kernel
methods. The headline "23x faster than Stim" benchmark compares against Stim
driven in its slowest mode. Driven properly, Stim was faster than v2 at every
size tested (up to n = 512). Prototypes built during this audit show a different data layout
beats both by a wide margin (see `DESIGN.md`).

---

## 1. Size and shape

| area | lines | used by your research? |
|---|---:|---|
| `core/` (tableau, Numba kernels, symplectic group, observables, GF(2) linear algebra) | 5,850 | yes, but through private methods |
| `circuits/` (config, runner, graphs, scheduler, pictures, drawing) | 3,930 | **no** (production bypasses it) |
| `analysis/` (registry, sweep, Study, plotting, FSS collapse) | 2,370 | notebooks only |
| `expurgation/` (Gullans et al. code surgery) | 1,120 | tests only |
| `studies/prl_production/` (campaign drivers, provenance, analysis) | 17,600 | yes |
| tests | ~10,000 | |

The research campaign's physics fits in about 150 lines
(`studies/prl_production/single_ref/raw_tau/engine.py:128-270`). About 90% of
the studies code is provenance, locking, journalling and validation.

## 2. The core simulator (`core/sparse_tableau.py`, `core/numba_kernels.py`)

### What it is

An Aaronson-Gottesman tableau with `2n` rows (n destabilizers + n stabilizers),
**without sign bits**. The same bits are stored four ways at once
(`sparse_tableau.py:7-19`):

1. a dense `(2n, n)` byte table `plt` of 2-bit Pauli codes,
2. per-row support lists `supp_q` with back-pointers `supp_pos`,
3. per-qubit lists of rows touching that qubit, `inv` / `inv_pos`,
4. per-qubit lists of rows with an X bit there, `inv_x` / `inv_x_pos`.

Every gate and measurement must keep all four in sync (`_apply_2q_kernel`,
`sparse_tableau.py:554-645`; `_sparse_xor_rows`, `:653-718`). A gate costs time
proportional to the number of rows touching its qubits, and a measurement costs
(anticommuting rows) x (pivot weight). There is an optional `hybrid=True` mode
that mirrors everything into bit-packed arrays when the state gets dense
(`:1389-1424`).

### What is good

* The measurement update and gate kernels are correct. Stim agreed on the
  sign-free stabilizer group at every step of every test.
* Gates as 16-entry lookup tables indexed by the packed 2-qubit Pauli is a
  clean idea and carries over to v3.
* The canonical-form comparison (GF(2) row reduction of the stabilizer block)
  is the right sign-free equality test.
* Free-function observables over a small protocol (`observables.py`), and a
  fast single-qubit entropy (`observables.py:306`) for the purification-time
  workload.
* Careful input validation with actionable error messages.

### What is wrong

* **Memory.** The four representations cost about 27·n² bytes (n = 1024: ~27 MB
  per state; n = 8192: ~1.8 GB). Stim needs about n²/2 bytes.
* **Destabilizers are carried but never needed.** Without signs, nothing reads
  the destabilizers. Measured on brickwork circuits, they are about twice as
  heavy as the stabilizers in the area-law phase (p = 0.2, n = 256: mean
  destabilizer weight 9.3 vs stabilizer weight 5.0), so a large share of the
  gate and measurement work goes into rows that no sign-free observable uses.
* **The measurement "outcome" is fake.** `measure_z` returns 0 for a
  deterministic measurement and an independent coin otherwise
  (`sparse_tableau.py:1945-1951`). The docstring claims correlations between
  qubits measured together are accurate (`:1879-1880`). They are not: on a Bell
  pair, Z0 xor Z1 should always be 0, and v2 gives 0.55 on average (reproduced).
  The circuits runner discards these values (`circuits/runner.py:365`).
* **Speed is limited by Python, not the kernels.** Each gate is a Python→Numba
  call. The runner batches calls (`_dispatch_2q_batch`, `_measure_z_batch`),
  but those are private (leading underscore), and the production code had to
  call them directly.
* **Two engines in one class.** The hybrid mode makes every method branch on
  `_dense_mode` (`:1328-1333`, invariant comment), and the dense path is
  "sticky": once it switches it rarely switches back (`:1197-1200`).
* **Single file of 2,889 lines** holding the pure-Python kernels, the dense
  kernels, mode switching, Pauli measurement, code-space projection,
  serialization and constructors.

## 3. The circuits layer (`circuits/`)

### What it is

`simulate(CircuitConfig(...), sample_seed)` builds and runs one family of
circuits: random Sp(4) two-qubit Cliffords on the edges of a graph, followed by
random single-qubit Z measurements. `CircuitConfig` has 20 fields
(`config.py:147-169`). There are 3 pictures, 4 gating modes, 3 matching modes,
4 measurement modes and 3 depth modes.

### What it cannot express

Arbitrary gate lists; single-qubit gates during the dynamics; X/Y or
multi-qubit measurements; measuring or gating the reference qubits; choosing
where the reference sits (hard-coded to qubit n-1, `picture.py:205-207`);
resets; noise; detectors; recorded measurement outcomes. There is no circuit
representation you can write by hand. The schedule exists only as a lazy random
stream (`scheduler.py:211`).

### Confirmed defects

1. **Brickwork + `matching_mode="fresh"` on `path` or `lattice_2d` silently
   reuses one fixed matching every layer** (`graphs.py:346-350, 395-405`), so the
   circuit factorizes into disjoint 2-qubit blocks. Reproduced: path, n = 8,
   p = 0, half-cut entropy 0 with `fresh` vs 2 with `round_robin`. Validation
   accepts the combination.
2. **Warmup restarts the layer index** (`scheduler.py:207, 217`). With
   `round_robin` and an odd number of warmup layers, the first measured layer
   repeats the previous matching.
3. **"Layer" means different things in different modes.** Random gating modes
   rescale the depth by n/(2m) (`config.py:425-465`), so `random_edge` with the
   default m = 1 runs 4n² layers (1024 at n = 16, vs 128 for brickwork). At fixed
   `p`, switching gating mode silently changes the gate-to-measurement ratio by
   n/(2m).
4. **`expected_gate_to_meas_ratio` assumes n/2 gates per brickwork layer**
   (`config.py:494-547`), which is wrong on path, lattice and networkx graphs.
5. **`n_cliffords=k` takes the first k rows of an enumeration whose order is
   documented as implementation-defined.** This is not a meaningful gate set.
6. **Seeds.** `base_seed` both seeds every circuit stream *and* chooses the
   graph realization (`config.py:339-343`), so you cannot vary one without the
   other. `simulate` defaults to `sample_seed=0`, so a loop of `simulate(cfg)`
   produces identical samples. `newman_watts` uses networkx's RNG, so graphs
   change across networkx versions (`graphs.py:594`).
7. **Mutable config with a cached graph.** Changing a field after construction
   leaves a stale `_graph` and skips validation; `to_dict()` does not
   round-trip for prebuilt graphs.

### Good ideas worth keeping

Graphs as pure objects; separating "generate a schedule" from "execute it";
named, independent random streams per purpose (the production engine already
does this properly); pictures as declarative setup ops; prebuilt gate lookup
tables; checkpoint callbacks.

## 4. The analysis layer (`analysis/`)

A registry of named observables, a grid x seeds `sweep`, a persisted `Study`
(parquet + HDF5 + JSON) with an "augment later" workflow, plotting, and
finite-size-scaling fitters.

### Confirmed defects

1. **`Study.augment` writes analysis arrays with the tableau writer, which
   casts to `uint8`** (`study.py:250-257` calls `io.write_tableaux_npz`, which
   casts at `io.py:183`). Entropy profiles above 255 and every float array are
   silently corrupted. The correct `write_arrays_npz` exists (`io.py:197`) but is
   not used.
2. **`Study.run` drops array-valued analyses.** `res.array_results` is never
   saved (`study.py:124-139`), but the names are recorded as computed, so a
   later `augment` skips them.
3. **Censored purification times are averaged as if complete.** `aggregate`
   takes the mean over the column (`plotting.py:103`); trajectories that never
   purified are NaN and dropped, which biases τ low. The analysis layer has no
   notion of censoring.
4. **Graph provenance is lost**: rows omit `graph6`, and grid values are keyed
   by `.name`, so two different graphs with the same name collide
   (`sweep.py:50-69, 141-156`).

Your studies code re-implements survival analysis and finite-size scaling
correctly (Kaplan-Meier, censoring-aware), so `fss_collapse.py` (561 lines) is
unused.

## 5. Expurgation (`expurgation/`)

Clean and self-contained, built on two kernel primitives
(`pauli_anticommuting_rows`, `project_pauli_into_pair`). It has no consumers
outside its tests. Its README and docstrings say it uses `measure_pauli`; it
actually uses `project_pauli_into_pair` (`roles.py:269`), which has no direct
test. Worth porting later as an optional module.

## 6. The production campaign (`studies/prl_production/`)

**This is the most informative part of the audit, because it shows what you
actually need.**

* Physics: single-reference purification time τ: n system qubits + 1 reference
  Bell-paired to a system qubit (fixed or random per circuit), a scramble of
  q·n random two-qubit gates on random pairs, then layers of ⌊n/2⌋ random edges
  (with replacement) of a Watts-Strogatz graph, each with a uniform Sp(4) gate,
  then Bernoulli(p) Z measurements; stop at the first layer where S(R) = 0, with
  right-censoring at 8n layers.
* Scale: n up to 256, 18.7 M trajectories planned, 4.27 M done when paused.
* Performance (from `BENCHMARKS.md:211-217`): n = 64, 64 layers took 3.82 ms. Of
  that, the per-layer entropy check took 26% and Python overhead per layer
  dominated.
* **It bypasses `circuits.simulate` entirely** (`raw_tau/engine.py:128-270`)
  and calls private methods: `_dispatch_2q_batch`, `_measure_z_batch`,
  `_build_2q_lut`, plus the private graph helper `_ws_rewire_edges`. The reasons
  are all things `simulate` cannot express: separate graph-disorder and
  circuit-disorder axes, named random streams per role, "N random pairs"
  scrambling, random reference placement, explicit edge lists.
* Bit-exact reproducibility across execution paths is a hard requirement.

**Conclusion for v3:** the kernel must expose a small, public, fast path for
exactly this workload (batched gates by table index, batched measurements,
cheap in-loop entropy probes), and circuits must be explicit data that a
50-line generator can produce.

## 7. Tests and Stim parity

* 607 core tests pass (about 2 minutes; 89 s of that is two tests that
  enumerate the 1.45 M-element group Sp(6)).
* **What was compared against Stim:** the sign-free stabilizer group (GF(2)
  RREF) after each operation, and whether each measurement was deterministic.
  Signs and outcomes were never compared, because v2 cannot produce them.
* **Without Stim installed, 300 of the 607 tests skip silently**, including many
  that do not need Stim. One CI leg runs exactly that way.
* **Not tested against Stim at all:** the circuit runner and batched kernels
  (only compared against each other), `project_pauli_into_pair`, `reset_z`,
  `apply_clifford` with 3+ qubits, mutual and tripartite information, code
  dimension, distance, everything in circuits/analysis/expurgation, and any
  system larger than n = 65.
* The entropy test checks v2 against Stim by applying the *same* rank formula
  to both tableaux (`test_observables.py:68-78`), so the formula itself is never
  checked independently.
* Several "parity" tests are self-referential (JIT vs Python, hybrid vs sparse,
  batch vs scalar). They catch regressions but cannot catch a shared mistake.

## 8. The benchmark against Stim

`benchmarks/benchmark_random_clifford.py` and `docs/figures/benchmark_runtime.py`
(the README's "23.4x at n = 512") drive Stim from a Python loop, one
`do_tableau` call per gate (about 7-9 µs of Python overhead each), and one
`measure(q)` call per qubit, which repeats Stim's O(n²) collapse step every
time. On the same circuits with Stim driven properly (one prebuilt
`stim.Circuit`, measurements batched per layer, `do_circuit`):

| n | depth | v2 | Stim, as in README | Stim, `do_circuit` | v2 vs fair Stim |
|---:|---:|---:|---:|---:|---:|
| 32 | 256 | 0.044 s | 0.067 s | 0.004 s | 0.1x |
| 128 | 1024 | 0.465 s | 1.214 s | 0.113 s | 0.2x |
| 256 | 512 | 0.454 s | 1.995 s | 0.160 s | 0.4x |
| 512 | 1024 | 2.12 s | (not run) | 1.46 s | 0.7x |

(brickwork, random Sp(4), p = 0.25; measured during this audit.) Driven fairly,
Stim was faster than v2 at every size tested, even though Stim also tracks
signs. The README's
claim that Stim pays O(n²) per gate is also wrong; Stim's gates cost O(n/64).

## 9. Verdict

| keep (as ideas or code) | redesign |
|---|---|
| LUT-driven gates; canonical-form comparison; free-function observables; fast single-qubit entropy; Sp(4) group machinery; graphs as data; named random streams; atomic I/O helpers; expurgation as an optional add-on | the data layout (memory, sync burden, no signs); fake outcomes; private hot paths; the hybrid mode; the 20-field config; implicit layer rescaling; seeding defaults; the Study/sweep stack; the README benchmark |

The v3 design that follows from this is in [`DESIGN.md`](DESIGN.md).
