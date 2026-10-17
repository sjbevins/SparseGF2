# Sweeps: phase-free monitored circuits on Watts–Strogatz graphs

A **sweep** is a grid of conditions `(n, beta, p)` (system size, rewiring probability,
measurement rate) on which the engine simulates random monitored Clifford circuits, phase-free,
with every seed derived from two master seeds. `sparsegf2.study.sweep` generates it into a
directory, resumes it, grows it and reads it back; `sparsegf2.study.identity` holds the seeds.
The science is specified in [`misc/watts_strogatz_study_spec_v1_4.md`](misc/watts_strogatz_study_spec_v1_4.md)
(cited "v1.4"). This document says exactly what the code does; each claim names the function
that does it, as `file:function` under `src/sparsegf2/`.

Contents: 1 a sweep in one minute · 2 the phase-free state · 3 the laws as generated ·
4 seeds and stream indices · 5 what is shared across rates · 6 checkpoints · 7 probes ·
8 snapshots and new observables · 9 files · 10 resume, grow, replay · 11 expectation values ·
12 performance options · 13 limits · 14 signed runs for decoding.

## 1. A sweep in one minute

```python
import json
from pathlib import Path

import numpy as np

from sparsegf2.study import identity, sweep
from sparsegf2.study.sweep import SweepSpec

spec = {
    "sizes": [8, 12],                 # n
    "betas": ["0", "0.2"],            # exact tokens (section 3)
    "rates": ["0.1", "0.3"],          # one list for every n, or {"8": [...], "12": [...]}
    "graphs": 4,                      # graphs per (n, beta); an int or a per-n table
    "circuits_per_graph": 2,          # circuits per graph; an int or a per-n table
    "stride": 8,                      # frozen: the most circuits per graph ever (default 1024)
    "layers": {"per_n": 2},           # T = 2 n; or a table {"8": 16, "12": 24}
    "checkpoints": {"count": 5},      # the geometric rule (section 6); or {"list": [...]}
    "master_graph_seed": 17,
    "master_circuit_seed": 29,
}
out = sweep.run(spec, "demo")          # also: sparsegf2 sweep run spec.json demo
assert out["complete"] and out["saved"] == 2 * 2 * 2 * 4 * 2 == 64
res = sweep.load("demo", 8, "0.2", "0.3")    # an EnsembleResult, rows sorted by u
assert res.values["code_dimension"].shape == (8, 17)   # k at layers 0..16
assert res.layers("tableau_snapshot").tolist() == [0, 1, 2, 4, 8, 16]
```

`SweepSpec.from_json` (`study/sweep.py:SweepSpec.from_json`) reads the spec strictly: unknown
or missing keys, malformed tokens, sizes below 5, a stride below `circuits_per_graph`, seeds
outside `[0, 2^32)` and per-n tables that miss a size are refused with a message that starts
`sweep spec:` and names the field. It then builds the engine spec of every size
(`SweepSpec.monitored_circuit`), so the engine's own validation of the protocol and of every
probe (`mipt.py:MonitoredCircuit.__post_init__`, `api.py:_build_probe`) runs before anything
is written. Optional keys and their defaults: `stride` 1024, `checkpoints` `{"count": 12}`,
`probes` `[{"kind": "code_dimension", "checkpoints": "every_layer"}, {"kind":
"tableau_snapshot"}]`, `picture` `"purification"` (or `"single"`: one reference qubit
Bell-paired with a system qubit drawn per trajectory; or `"none"`: the system alone, from
`|0...0>`, and then name the probes, since `code_dimension` needs a reference),
`stop_when_purified` `false`. `SweepSpec.to_json` is the canonical form (defaults written out,
per-size values resolved, tokens as canonical strings) and `SweepSpec.digest` its SHA-256
(`study/identity.py:json_digest`): two spellings of one sweep have one digest.

The command line: `sparsegf2 sweep run SPEC.json DIR [--threads N] [--chunk-size K]
[--compression fast|none|default] [--writers W]`,
`sparsegf2 sweep status DIR`, `sparsegf2 sweep replay DIR N BETA P U` (`cli.py:_cmd_sweep`;
`--json` for one JSON object, exit codes 0, 2 for bad input, 1 for an internal error or a
replay that differs, 130 for Ctrl-C). There are no MCP tools of the sweep's own: the existing
`load_results` (`api.py:load_results`) reads a sweep's directory (the index: every condition's
`path`, saved and target trajectories, the versions, the spec), one condition's folder (the
summary of all its parts, through `sweep.load`; above the memory limit it refuses and names
`sweep.parts`) and a single part file.

## 2. The phase-free state and its update rules

**Representation.** A trajectory on `N = n + n_ref` qubits (`n_ref = n` in the purification
picture, `1` for `"single"`, `0` for `"none"`) holds one `uint64` array `tab` of shape `(2N + 1, ceil(N/64))`:
`tab[2q]` is the X column of qubit `q` (bit `r` set iff stabilizer `r` has X or Y on `q`),
`tab[2q+1]` its Z column; the sign row is never read in phase-free mode, and there are no
destabilizers (`_kernel/tableau.py`, module docstring; `_kernel/ensemble.py:run_ensemble` calls
every kernel with `phases=False`). The state is the **phase-free stabilizer group**: the span
over GF(2) of the `N` rows `(x_0..x_{N-1}, z_0..z_{N-1})`, an isotropic subspace of dimension
`N` (the product of Paulis is the sum of their vectors, up to a phase).

**Start.** `_kernel/ensemble.py:_setup`: from `|0...0>` (`init_zero`), for the purification
picture `H` on system qubit `i` then `CX(i, n + i)` for `i = 0..n-1`: `n` Bell pairs, system
qubit `i` with reference `n + i`; for `"single"`, `H` on the partner `a` then `CX(a, n)`, with
`a = next_below(n)` the first draw of the trajectory's stream (`run_ensemble`), before any
gate, so it does not depend on `p` either.

**A gate is its symplectic matrix.** A two-qubit Clifford has the index `j = 16 s + b`
(`clifford.py`, module docstring): `s` is the position of its symplectic matrix `M` in the
sorted list of the 720 elements of `Sp(4, F_2)` (`clifford.py:symplectic_group`; rows are the
images of `X_a, X_b, Z_a, Z_b`), `b` its four sign bits. `_kernel/tableau.py:apply_2q` takes
`sym = j >> 4` and replaces, for every row at once (64 per word), the local vector
`v = (x_a, x_b, z_a, z_b)` by `v M (mod 2)`; the sign bits `b` enter only the sign row
(`track = phases and anf != 0`), which is off. So the phase-free action of gate `j` is exactly
`M_{j // 16}`: the 16 Cliffords of one class differ by a Pauli, which fixes every phase-free
group.

**A uniform index is a uniform class.** The gate index is `pool[next_below(rng, 11520)]` with
`pool = 0..11519` (`_kernel/ensemble.py:gen_layer`, `mipt.py:MonitoredCircuit.__post_init__`):
uniform on the 11,520 projective two-qubit Cliffords (`clifford.py:num_cliffords` is
`720 * 16`). Each class `s` holds exactly the 16 indices `16 s .. 16 s + 15`, so
`P(class = s) = 16 / 11520 = 1/720`: the phase-free gate is uniform on `Sp(4, F_2)`.
`tests/test_study_sampling.py` checks both laws (`test_clifford_indices_are_uniform_over_11520_and_their_classes_over_720`).

**A Z measurement acts on the row space; its outcome is not needed.**
`_kernel/tableau.py:measure_1q_phase_free_tracked` with `code = 2` (Z on `q`): the rows that
anticommute with `Z_q` are those with `x_q = 1` (`_anticommuting_1q`). If there are none,
`Z_q` (up to sign) is already in the group: the measurement is deterministic and the
phase-free group does not change. Otherwise one anticommuting row (the pivot) multiplies every
other anticommuting row (they now commute with `Z_q`) and is then replaced by `Z_q`
(`_collapse_phase_free`). The new group is `<commuting part of S, Z_q>`, and the outcome
`+-1` would only set the sign of `Z_q` and of nothing else in the phase-free group. So the
phase-free group after a measurement is the same function of the group before it for both
outcomes, whatever their Born probabilities: every observable of the phase-free group
(entropies, the code, its distances) has the same value on every outcome history, and the
engine draws no outcome. Signs matter only for decoding (section 14).

**The code dimension is a popcount.** With a reference, `ref_rows` marks the stabilizer rows
with support on the reference qubits `R`, under the invariant that their `R`-parts are linearly
independent (`_kernel/tableau.py:measure_1q_phase_free_tracked`, which keeps it by choosing a
pivot outside `ref_rows` when there is one). For a pure state `S(R) = rank(S|_R) - |R|`, and
`rank(S|_R)` is the number of marked rows, so `k = S(R) = popcount(ref_rows) - n_ref`
(`_kernel/ensemble.py:run_ensemble`, `s_ref`). The reference is pure exactly when `k = 0`.

**Canonical form of a row space** (`tableau.py`, module docstring). Order the `2N` columns
`x_0..x_{N-1}, z_0..z_{N-1}`. The canonical basis is the reduced row echelon form over GF(2):
the first nonzero column of each row (its pivot) is 1, pivots increase down the rows, every
pivot column is zero in every other row, zero rows dropped (`_kernel/rowspace.py:rref`).
*Every subspace `V` has exactly one such basis.* Existence: Gaussian elimination. Uniqueness:
the pivot columns are the set `P = {first nonzero column of v : v in V, v != 0}`, a property of
`V` alone (a nonzero combination of reduced rows has as its first nonzero column the smallest
pivot among the rows used, since the other rows vanish there); and for each pivot `c` the
reduced row with pivot `c` is the unique `v in V` with first nonzero column `c` and `v_{c'} = 0`
for every other `c' in P` (two such vectors would differ by an element of `V` that vanishes on
every pivot column, and a nonzero element of `V` has a pivot column as its first nonzero
column). So two generating sets of one group give the same canonical rows, bytes and digest.

**Entropies of a row space** (`tableau.py:PhaseFreeTableau.entropy`). For a stabilizer state
with phase-free group `S` of dimension `r` and a region `A`, `S(A) = |A| - dim S_A` with `S_A`
the elements supported in `A`; restriction to the complement `Abar` has kernel `S_A`, so
`dim S_A = r - rank(S|_Abar)` and `S(A) = |A| - r + rank(S|_Abar)`; for a pure state
(`r = N`) this equals `rank(S|_A) - |A|`.

**The physical code** (`tableau.py:PhaseFreeTableau.physical_tableau`). Reduce with the
reference columns first: the rows whose pivot is a system column are zero on every reference
column and span every element of `S` that is (an element zero on the reference is a
combination of reduced rows; the smallest pivot used would be a reference column if any row
with a reference pivot were used, and it is not). Restricted to the system they span `S_phys`
of dimension `r_t`, and `k_t = n - r_t` (`physical_code(n).num_logical`) equals `S(R)`.

**Checked against Stim.** `tests/test_sweep.py::test_saved_states_and_code_dimensions_equal_stim_on_the_exported_circuits`
runs every saved trajectory of two sweeps (purification and single pictures, `n = 7, 8`,
`beta = 0, 0.3`, two rates) in Stim from its exported circuit (section 14) and finds the same
canonical joint state, signs dropped, at every saved checkpoint and the same `k` at every
layer.

## 3. The laws as generated

**Tokens** (`study/identity.py:Token`). Every rate and rewiring probability is an exact
rational `x` in `[0, 1]`, written canonically: the decimal expansion without exponent or
trailing zeros when it terminates (`"0.05"`), else `"num/den"` in lowest terms (`"1/3"`);
`"0.1"`, `".1"`, `"1/10"` and the JSON number `0.1` are one token. The simulator receives
`fl(x) = float(x)`, the correctly rounded double (`Token.float64`). The kernels compare
`next_double()` with it, and `next_double` returns `k / 2^53` for `k` uniform in
`[0, 2^53)` (`_kernel/rng.py:next_double`), so the realized probability is
`x' = ceil(2^53 fl(x)) / 2^53`: `x' = x` when `x` is a multiple of `2^-53`, else
`|x' - x| < 2^-53 + 2^-54 < 2^-52`.

**Watts–Strogatz graphs** (`graphs.py:watts_strogatz` with `k = 4`, called by
`study/identity.py:build_graph`). Start from `C(n, 2)`: the edges `{i, (i + d) mod n}` for
`d = 1, 2`. Visit the original edges `(a, b) = (i, (i + d) mod n)` in offset-major order
(`d = 1` for `i = 0..n-1`, then `d = 2`; `graphs.py:_ring_lattice`). For each, draw
`next_double() < fl(beta)` (probability `beta'`, as above); if it holds and `a` is not already
adjacent to every other vertex, draw `w = next_below(n)` until `w != a` and `w` is not adjacent
to `a` (rejection: `w` uniform on the vertices that are neither `a` nor its neighbors, the
current neighbors including `b`), then delete `{a, b}` and add `{a, w}`. The draws come from
the xoshiro256** state `SeedSequence(graph_seed).generate_state(4)` (`_kernel/rng.py:seed_state`).
The result, as a sorted `(2n, 2)` array with `a < b` (`graphs.py:from_edges`), passes
`study/identity.py:validate_graph`: (i) simple, since `w` is never `a` nor a neighbor; (ii)
`2n` edges, since each rewiring deletes one edge and adds one absent edge and the `2n` original
pairs are distinct for `n >= 5`; (iii) every degree is at least 2, since vertex `v` is the
retained endpoint `a` of its two original edges `(v, v+1)`, `(v, v+2)` and keeps an edge at
each of their visits, and the only edge a visit deletes is its own original pair, which is
present when visited (a created edge `{a, w}` is never a pair still to be visited: that pair
would be present, so `w` adjacent); (iv) `beta = 0` gives `C(n, 2)`; and for `n = 5`,
`C(5, 2) = K_5`, every vertex is adjacent to all others, and no edge is ever rewired.
`tests/test_study_sampling.py` compares the generator with an independent transcription of
v1.4 §2.2 (`tests/_ws_reference.py:rewire`) trace for trace, and its exact law at `n = 6, 7`.

**Gates and masks, per layer** (`_kernel/ensemble.py:gen_layer`, protocol of
`study/sweep.py:SweepSpec.engine_spec`). Layer `t = 1..T` of trajectory `u` on graph `g`
(edge list `E_g`, sorted) draws, from the trajectory's stream (section 4):

1. `h = floor(n/2)` gates, each: an edge `e = next_below(|E_g|)` (uniform, with replacement;
   `next_below` is Lemire's unbiased method), then the gate index `next_below(11520)`; the gate
   acts on `(a, b) = E_g[e]`, low endpoint first (`random_orientation = False`);
2. `n` uniforms, one per system qubit `q = 0..n-1` in order: `q` is measured in Z iff
   `next_double() < fl(p)`. References are never measured.

The gates are applied in order, then the measurements (`run_ensemble`). No scramble
(`scramble_* = 0`). Every layer draws exactly `2h` integers (plus Lemire's rare retries, which
depend on the stream only) and `n` doubles, whatever `p`.

## 4. Seeds and stream indices

All seeds are fixed-length tuples of 32-bit words beginning with a tag
(`study/identity.py`; the words of a token `x` are `token_words(x) = (h mod 2^32, h div 2^32)`
with `h` the first 8 bytes, little-endian, of `blake2b(token, digest_size=8)`):

| seed | words | function | depends on |
|---|---|---|---|
| graph `g` of `(n, beta)` | `(0x57534731 "WSG1", master_graph_seed, n, w(beta), g)`, 6 words | `graph_seed` | not `p` |
| run seed of `(n, beta)` | `(0x57534331 "WSC1", master_circuit_seed, n, w(beta))`, 5 words | `run_seed(..., "shared")`, `SweepSpec.run_seed` | not `p`, `g`, `c` |
| outcomes of `(n, beta, p, g, c)` | `(0x57534F31 "WSO1", master_circuit_seed, n, w(beta), w(p), g, c)`, 9 words | `outcome_seed` | everything |

The tags keep the three domains disjoint whatever the master seeds; every length exceeds the
`SeedSequence` pool of 4 words, so zero padding cannot identify two seeds
(`tests/test_study_identity.py`).

**Stream index.** Trajectory `(g, c)` of a size has the index `u = g * stride + c` with
`c < circuits_per_graph <= stride` (`study/identity.py:stream_index`;
`SweepSpec.indices`). The engine runs with `circuits_per_graph = stride` (`engine_spec`), so
trajectory `u` uses graph `u // stride = g` (`_kernel/ensemble.py:run_ensemble`).

**From the run seed to the draws.** `SeedSequence(run_seed).generate_state(4, uint64)` gives
`key_struct` (words 0, 1) and `key_probe` (words 2, 3) (`mipt.py:_keys`). Trajectory `u` draws
its whole circuit from the xoshiro256** state `stream_state(key_struct, u)`:
`x = key0 XOR sm(u)`, `gamma = sm(key1)`, word `j` = `mix64(x + (j + 1) gamma)` with
splitmix64's `mix64` and `sm(z) = mix64(z + 0x9E3779B97F4A7C15)` (`_kernel/rng.py:stream_state`).
A sampled probe evaluated at layer `L` draws from `stream_state(key_probe, mix(u, hash(name), L))`
(`_kernel/ensemble.py`, module docstring). A trajectory depends only on `(run seed, u, its
graph, the protocol)`: not on the other trajectories, their order, the thread count or the
chunks (`mipt.py:MonitoredCircuit.run`, `indices=`).

A worked example: `n = 16`, `beta = 0.1`, `p = 0.15`, graph `g = 3`, circuit `c = 1`, stride
1024, master seeds 17 and 29.

```python
w_beta, w_p = identity.token_words("0.1"), identity.token_words("0.15")
assert w_beta == (2007713733, 3507754741) and w_p == (3479488031, 1099992840)
assert identity.stream_index(3, 1, 1024) == 3 * 1024 + 1 == 3073
assert identity.graph_seed(17, 16, "0.1", 3) == (0x57534731, 17, 16, *w_beta, 3)
assert identity.run_seed(29, 16, "0.1", "shared") == (0x57534331, 29, 16, *w_beta)
assert identity.outcome_seed(29, 16, "0.1", "0.15", 3, 1) == (
    0x57534F31, 29, 16, *w_beta, *w_p, 3, 1
)
from sparsegf2._kernel.rng import stream_state

state = np.random.SeedSequence(list(identity.run_seed(29, 16, "0.1", "shared")))
key_struct = state.generate_state(4, np.uint64)[:2]   # one key for every trajectory of (16, 0.1)
xoshiro = np.empty(4, np.uint64)
stream_state(key_struct, 3073, xoshiro)               # the state trajectory (3, 1) draws from
graph = identity.build_graph(16, "0.1", 3, 17)        # the same graph at every rate
assert graph.num_edges == 32 and graph.regenerate() == graph
```

## 5. What is shared across rates, and why it reduces the variance of differences

For one `(n, beta)` every rate uses the same graph collection (the graph seed has no `p`) and
the same run seed (no `p`), so trajectory `u` reads the same stream at every rate. Since a
layer consumes the same draws whatever `p` (section 3), the stream positions agree layer by
layer: the gates are identical across rates, and the mask of layer `t` at rate `p` is
`{q : U_{t,q} < fl(p)}` for uniforms `U_{t,q}` shared by all rates, so masks are **nested**:
`p1 < p2` implies `M_t(p1) ⊆ M_t(p2)` (`tests/test_sweep.py::test_gates_are_shared_and_masks_nested_across_rates`,
`tests/test_study_sampling.py::test_masks_are_uniform_below_the_float_threshold_and_nested_across_rates`).
Each rate's trajectories still have exactly their own law (the marginal of a coupling is
untouched). Different `(n, beta)` use different seeds and are independent samples.

For an observable `X_u(p)`, the paired estimate of `E X(p2) - E X(p1)` averages
`D_u = X_u(p2) - X_u(p1)`, with `Var D = Var X(p2) + Var X(p1) - 2 Cov(X(p1), X(p2))`;
independent samples would have the first two terms only. So pairing helps exactly when the
covariance is positive, which the shared graph, gates and nested masks make likely and which
the data shows (section 11 compares the two errors).

## 6. Checkpoints

The engine's global checkpoints are `"every_layer"` (`engine_spec`), so purification is tested
after every layer (a popcount, section 2) and `purification_layer` is the exact first layer
with `k = 0` (`-1`: not by `T`). The **sparse schedule** `S(n)` comes from `checkpoints`:

* `{"count": N}`: v1.4 §9, `a_j = T^(j/(N-1))` for `j = 0..N-1`, each rounded to the nearest
  integer with ties upward, plus `0, 1, T`, sorted and deduplicated
  (`study/sweep.py:geometric_checkpoints`). The rounding is exact integer arithmetic
  (`round_half_up_power`: `floor(s^(1/q) / 2 + 1/2)` for `s = 2^q T^j`), so no float can move a
  value across a boundary; `N = 1` gives `0, 1, T`;
* `{"list": [...]}`: the entries up to `T`, plus `0, 1, T` (`SweepSpec.checkpoints_at`).

`S(n)` depends on `T(n)` only, so every rate of a size has the same schedule. A probe without
its own `"checkpoints"` is evaluated on `S(n)`; one with `"checkpoints"` (any engine form:
`"every_layer"`, a stride, a list) keeps it. Checkpoints draw nothing: they never change a
trajectory.

## 7. Probes

The probes are the engine's (`mipt.py`, the probe functions; AGENTS.md "Probes"), given as their
JSON form and built by `api.py:_build_probe`. The defaults:

* `code_dimension` at every layer: `k_t = S(R)`, the number of logical qubits the system still
  encodes (the reference's entropy), `values["code_dimension"]` of shape
  `(trajectories, T + 1)` for layers `0..T`; `k_0 = n` and `k_t` never increases.
* `tableau_snapshot` on `S(n)`: the joint state itself, all `N` generators over all `N`
  qubits (system `0..n-1`, then reference `n + i` paired with system `i`), as words
  (section 8); its value is `N`. With `stop_when_purified`, evaluations after the stop save
  nothing (row count `-1`).

Each result also holds `purification_layer`, `final_layer` (the last layer simulated) and
`num_measurements` per trajectory. `stop_when_purified = true` ends a trajectory at its
purification layer: code probes then hold their exact values for a pure reference
(`k = 0`), entanglement probes `NaN`, and snapshots nothing (`mipt.py:EnsembleResult`).

## 8. Snapshots, and new observables from them

A `tableau_snapshot` evaluation is `generators[t, e]` of shape `(N, 2 wq)` (`wq = ceil(N/64)`):
row `r` holds the X bits of qubits `0..N-1` in words `0..wq-1` (bit `q & 63` of word `q >> 6`)
and the Z bits in words `wq..2wq-1`, in the engine's basis
(`_kernel/entropy.py:joint_snapshot_rows`). `EnsembleResult.tableau(t, e)` and, for a whole
evaluation at once in parallel, `EnsembleResult.tableaux(e)` return
`tableau.PhaseFreeTableau` objects in canonical form (section 2). `to_bytes()` is
`b"SGF2RS1\0"`, `N` and the rank as little-endian uint32, then each canonical row's `2N` bits
packed most significant bit first (`ceil(2N/8)` bytes); `digest()` is its SHA-256: equal for
equal groups whatever the basis or the simulator. Any phase-free observable of the joint state
at a saved depth is then a function of the snapshot, without simulating:

```python
n, layers = 8, res.layers("tableau_snapshot")
system, reference = range(n), range(n, 2 * n)
e = int(np.flatnonzero(layers == 4)[0])            # the evaluation at layer 4
for row, tab in enumerate(res.tableaux(e)):        # None where a stopped row saved nothing
    k = tab.entropy(reference)                     # S(R): the code dimension
    assert k == res.values["code_dimension"][row, 4] == tab.physical_code(n).num_logical
    half = tab.entropy(range(n // 2))              # half-cut entropy of the joint state
    lost = tab.mutual_information(range(2), reference)   # I(A:R) of erasing qubits 0, 1
    assert 0 <= lost <= 2 * k and tab.is_pure()
```

## 9. The files

```text
DIR/
  sweep.json                 the index (below)
  .lock                      held by the one writer (api.py:_lock_output_dir)
  graphs/n8_beta0.2.npz      graphs 0..G-1 of (n, beta): graphs.save format, read by
                             graphs.load and by the API's {"family": "file"} graphs
  graphs/n8_beta0.2.json     per graph: g, its seed words, generator, edge digest
  n8_beta0.2_p0.3/           one folder per condition ('/' of a rational token -> '-')
    result-000000000000.npz  saved parts: each an EnsembleResult file (mipt.py:EnsembleResult.save),
    result-000000000016.npz  named by its first stream index u, rows in increasing u
    spec.json                the condition's engine spec in the API's JSON form, its seed,
                             and under sweep the graphs, circuits per graph and stride
    provenance.json          versions, platform, git commit, command; one entry per resume;
                             parts: each saved part with the package version and compression
```

`sweep.json` (`study/sweep.py:_index`): `format`, `digest`, the canonical `spec`, `versions`
(`study/identity.py:version_record`: graph generator, Clifford numbering, dynamics, row-space
format, the three seed tags, `randomness_contract`, `sparsegf2`), `graphs` (per `(n, beta)`:
files and count) and `conditions` (per `(n, beta, p)`: the tokens, `p`'s double in hex, the
folder, the run seed words and the target number of trajectories). A part file holds the
spec with its graph collection, the seed (`seed_entropy` = the run seed words), `trajectories`
(the stream indices `u`), `value::<probe>`, `snapshot::<probe>` and
`snapshot_counts::<probe>`, `purification_layer`, `final_layer`, `num_measurements`.

`sweep.status(DIR)` counts the saved indices (it reads only the `trajectories` arrays).
`sweep.load(DIR, n, beta, p)` concatenates a condition's parts (`EnsembleResult.concatenate`)
sorted by `u`; `sweep.parts(...)` yields them one at a time for data too large to hold;
`sweep.rows(...)` gives, in `load`'s order, `n, beta, p, g, c, u` and the graph, run and
outcome seed words of every row.

## 10. Resume, grow, replay

`sweep.run(spec, DIR)` (`study/sweep.py:run`) takes the directory's lock, checks a stored
spec against the new one (`_check_growth`) and the stored versions against this code (below),
replays the first and the last saved trajectory of every condition it will add to
(`_replay_check`, `verify`: every stored value and snapshot word must be equal), writes
the graph files (`_write_graphs`: every graph rebuilt from its seed; a recorded edge digest that
no longer reproduces is refused), writes `sweep.json`, then runs each condition's missing
indices (`_run_condition`: target `SweepSpec.indices(n)` minus the saved ones) in chunks with
`MonitoredCircuit.run(indices=chunk, seed=run_seed)`, each saved atomically (temporary file,
then `fsync`, then rename; `study/sweep.py:_save_part`) on a writer thread while the next chunk
computes. An interrupted run waits for the saves in flight and loses at most the chunk being
computed; the next run deletes stray temporaries and resumes. A part that cannot be read
(damaged after it was saved) makes `status`, `run`, `load` and `load_results` refuse with a
message naming the file: delete it, and the next run recomputes its trajectories exactly.
Chunk sizes, compression and writers: section 12.

**Growing.** A new spec may add sizes, rewiring probabilities, rates, graphs and circuits per
graph. It may not change `stride`, the master seeds, `picture`, `stop_when_purified`,
`probes` or `checkpoints`, drop a size, rate or rewiring probability, lower a count, or change
`T` of a saved size: each is refused naming the field, before anything is written. Adding
leaves every saved trajectory bitwise unchanged because its stream index `u = g * stride + c`,
its graph (seeded by `g`, not by `G`) and its stream (`(run seed, u)`) are unchanged, and the
engine picks graph `u // stride` whatever the collection's length. A part saved before graphs
were added holds the shorter collection; `parts`/`load` check that it is the current spec on
those first graphs (`mipt.py:_protocol_difference`) and read it on the current spec
(`study/sweep.py:_rebased`). `tests/test_sweep.py` checks that a grown directory equals a fresh
run of the grown spec, bitwise.

**Across package versions.** A later `sparsegf2` may grow a directory when every other entry
of `versions` is equal and the replay check above passes; otherwise `run` refuses, saying that
the versions disagree, before anything is written. Each part keeps the version that wrote it
(in the part file and under `parts` in `provenance.json`), and `load` joins parts of different
versions (`EnsembleResult.concatenate(..., same_version=False)`, which only the sweep passes).

**Replay.** `sweep.replay(DIR, n, beta, p, u)` simulates trajectory `u` again from its seed (a
one-row result; other `probes` or `checkpoints` may be given, which observe the same
trajectory). `sweep.verify(...)` (`sparsegf2 sweep replay`) compares it with the saved row,
every value and snapshot word, and returns `identical` and the saved tableau digests.

```python
g8 = SweepSpec.from_json({**spec, "graphs": 6, "rates": ["0.1", "0.2", "0.3"]})
before = sweep.load("demo", 8, "0.2", "0.3")
sweep.run(g8, "demo")                               # grows: 2 graphs and 1 rate more
after = sweep.load("demo", 8, "0.2", "0.3")
old = np.isin(after.trajectories, before.trajectories)
assert np.array_equal(after.values["code_dimension"][old], before.values["code_dimension"])
check = sweep.verify("demo", 8, "0.2", "0.3", 17)  # u = 17: graph 2, circuit 1
assert check["identical"] and (check["g"], check["c"]) == (2, 1)
```

## 11. Expectation values

The ensemble of a condition is nested: graphs `G ~ WS(n, beta)`, circuits on a graph
(gates and masks), and nothing else (no outcomes, section 2). The target is
`E[X] = E_G[ E[X | G] ]`. Estimate it with **equal weight per graph**: the graph means
`m_g = mean_c X_{g,c}`, then `mean_g m_g`, with the standard error `sd_g(m_g) / sqrt(#graphs)`
(circuits on one graph are correlated, so they are not independent samples; v1.4 §6.3, §6.4).
With equal circuits per graph the point estimate equals the plain mean, but a partial run
(an interrupted condition) can hold unequal counts, and the error needs the graph means
either way. `res.graph_index` (or `rows(...)["g"]`) gives each row's graph.

```python
res = sweep.load("demo", 8, "0.2", "0.3")
k = res.values["code_dimension"]                       # (rows, T + 1)
graph = res.graph_index
labels = np.unique(graph)
m = np.array([k[graph == x].mean(axis=0) for x in labels])   # graph means, (G, T + 1)
mean, se = m.mean(axis=0), m.std(axis=0, ddof=1) / np.sqrt(len(labels))

# purification time: -1 marks a censored trajectory (not purified by T); never average it
tau = res.purification_layer
assert np.all((tau == -1) | (tau >= 1))
t = np.arange(k.shape[1])
alive = (tau[:, None] < 0) | (tau[:, None] > t[None, :])     # tau > t, exact at every layer
survival = np.array([alive[graph == x].mean(axis=0) for x in labels]).mean(axis=0)
assert np.allclose(survival, (k > 0).mean(axis=0))

# a paired difference across rates: same rows (u), same graphs
low = sweep.load("demo", 8, "0.2", "0.1")
assert np.array_equal(low.trajectories, res.trajectories)
d = k - low.values["code_dimension"]
dm = np.array([d[graph == x].mean(axis=0) for x in labels])
paired_se = dm.std(axis=0, ddof=1) / np.sqrt(len(labels))
```

`survival[t]` estimates `P(tau > t)`, which here equals the fraction with `k_t > 0` because
`k` never increases. Censoring: `tau = -1` means "not purified by `T`", a lower bound, never a
time; use the survival curve, the censored fraction or a Kaplan–Meier estimate
(`analysis.kaplan_meier(*res.purification_times())`); raise `T` (a new directory) when too
many are censored. `NaN` appears only after a `stop_when_purified` stop (entanglement probes,
undefined there) and in `contiguous_distance` without a code: average over the rows that hold
a value and say so, or record without stopping. `paired_se` against
`sqrt(se_low^2 + se_high^2)` measures what the coupling of section 5 gained.

## 12. Performance options

### Baseline

`benchmarks/sweep_throughput.py` times `sweep.run` on one condition at a time (results in
`benchmarks/results/sweep_throughput.md`): beta = 0.1, the purification picture, T = 4n,
`code_dimension` at every layer, without and with a `tableau_snapshot` on 12 geometric
checkpoints, default threads, chunks, compression and writers. On an AMD Ryzen 9 9950X3D (16
cores, 32 threads, Windows 11, Numba 0.68 `omp`), sparsegf2 3.0.0.dev12, before any
CPU-specific option:

| n | trajectories/s, k only (p = 0.05 / 0.2) | with snapshots (p = 0.05 / 0.2) | disk bytes per trajectory with snapshots (p = 0.05 / 0.2) |
|---:|---:|---:|---:|
| 64 | 13,580 / 10,980 | 7,200 / 8,540 | 25,483 / 14,949 |
| 128 | 5,020 / 3,790 | 2,670 / 2,860 | 79,071 / 37,443 |
| 256 | 1,130 / 661 | 802 / 566 | 280,250 / 109,399 |
| 512 | 170 / 80 | 127 / 69 | 993,800 / 331,521 |

**Saving parts.** A part is saved on one of `writers` threads (default 4) while the next one
computes (`study/sweep.py:_run_condition`, `_save_part`; the kernels and zlib release the GIL;
at most `writers` saves wait, so at most `writers + 1` parts, each at most `PART_BYTES` =
256 MB, are in memory). The compression of `EnsembleResult.save` is chosen per run
(`run(..., compression=...)`, `--compression`; recorded per part in `provenance.json`, never
part of the sweep's identity, and it changes no result). Saving one part with snapshots (the
largest a sweep makes, p = 0.05), `fsync` included, against computing it:

| n | trajectories | compute | `none` | `fast` (zlib level 1, default) | `default` (zlib's default level) |
|---:|---:|---:|---:|---:|---:|
| 128 | 1,235 | 0.15 s | 0.18 s, 269.8 MB | 0.74 s, 95.0 MB | 2.99 s, 77.0 MB |
| 256 | 312 | 0.27 s | 0.19 s, 269.2 MB | 0.67 s, 84.9 MB | 2.21 s, 72.6 MB |

`fast` is the default: it writes 2.8 to 3.2 times less than `none` and is 3 to 4 times faster
than `default`, and with four writers the cores wait little on it from n = 256 on (the engine
is 5.87 of 7.66 s at n = 256 and 11.18 of 12.11 s at n = 512, p = 0.05). Below that the data
outruns zlib (the engine is 1.50 of 6.82 s at n = 64): choose `none` when disk is plentiful.
Measured during the choice (n = 256, 9,216 trajectories with snapshots, 512 MB parts): 9.6 s with `none` and
one writer, 17.1, 11.9 and 11.2 s with `fast` and 2, 4 and 6 writers (engine 8.1 to 8.9 s).
Without snapshots the parts are small and the engine takes most of the wall time from
n = 256 on (25.9 of 27.1 s at n = 512, p = 0.05); every part stores the condition's graph
collection.

**Chunks.** Without `chunk_size`, the first chunk of a condition is sized by
`estimate_ensemble`'s cost model (about 10 estimated seconds, `api.py:_default_chunk_size`),
which overestimates the cost several times over (at n = 128 with snapshots it models 26.6 ms
of one thread per trajectory where 4.4 ms of thread time, 0.14 ms of wall time on 32 threads,
was measured); every later chunk holds about
`CHUNK_SECONDS` = 10 s at the rate the previous chunk measured, and at most `PART_BYTES` of
result. Chunk sizes never change a result.

### CPU-specific option

(Left for the performance task.)

## 13. Limits

* `n >= 5` (the generator needs `k = 4 < n`); `n`, `g`, the master seeds and `c` are 32-bit
  words (at most `2^32` graphs and circuits per graph); `stride` and the largest stream index
  `(graphs - 1) * stride + circuits_per_graph - 1` must be below `2^63`. `SweepSpec.from_json`
  refuses a spec that breaks one of these before anything is written. A part's name is its
  first `u`, zero-padded to at least 12 digits.
* `stride` is frozen: more circuits per graph than `stride` need a new directory.
* A directory grows only under the randomness contract, graph generator, Clifford numbering,
  dynamics, row-space format and seed tags recorded in `versions`, and reading it needs the
  same contract. The package version may differ (section 10).
* One snapshot probe per spec (`mipt.py:MonitoredCircuit.__post_init__`). A tableau snapshot
  costs `N * 2 ceil(N/64) * 8` bytes per evaluation and trajectory (64 KiB at `N = 512`): give
  it few checkpoints. `load` holds a whole condition in memory; `parts` holds one part.
* A condition's `spec.json` is a record, not a way to rerun it: its `trajectories` counts the
  indices `u = g * stride + c` (`g < graphs`, `c < circuits_per_graph`, written under `sweep`),
  but running its `spec` through the API (`run_ensemble`, `start_ensemble_job`) runs
  `0 .. trajectories - 1`, which are the sweep's trajectories only when `circuits_per_graph`
  equals `stride` (2 graphs of 2 circuits at stride 4 are `u = 0, 1, 4, 5`; the API would run
  `0, 1, 2, 3`). With one graph it does not even rebuild there (a graph file with one graph is
  a single graph, which refuses `circuits_per_graph`). Rerun trajectories with `sweep.replay`
  and `sweep.verify`; the sweep itself never reads `spec.json`.
* The realized rates and rewiring probabilities are `x' = ceil(2^53 fl(x)) / 2^53`
  (section 3).

## 14. Signed runs for decoding

A phase-free run draws no outcomes. A signed rerun of trajectory `(g, c)` of `(n, beta, p)`
replays its exact circuit (`MonitoredCircuit.circuit(u, seed=run_seed)`: setup, a `TICK`, then
per layer `CLIFFORD_2Q` gates, `M` and a `TICK`, so tick `t` follows layer `t`) on a
phase-tracking `Simulator` whose Born outcomes come from the outcome seed (section 4). Dropping
the signs gives back the saved phase-free states, and the same outcome seed gives the same
outcomes:

```python
from sparsegf2 import Simulator, graphs
from sparsegf2.tableau import PhaseFreeTableau

spec_ = SweepSpec.from_json(json.loads(Path("demo/sweep.json").read_text())["spec"])
_, collection, _ = graphs.load("demo/graphs/n8_beta0.2.npz")
mc = spec_.monitored_circuit(8, "0.2", "0.3", collection[: spec_.graphs[8]])
u = 17
g, c = divmod(u, spec_.stride)
circuit = mc.circuit(u, seed=list(spec_.run_seed(8, "0.2")))
outcomes = identity.outcome_seed(spec_.master_circuit_seed, 8, "0.2", "0.3", g, c)
sim = Simulator(16, phases=True, seed=np.random.SeedSequence(list(outcomes)))
wanted = res.layers("tableau_snapshot").tolist()
run = sim.run(circuit, observe=lambda s, tick: PhaseFreeTableau.from_simulator(s).digest()
              if tick in wanted else None)
signed = [digest for _, digest in run.observations if digest is not None]
assert signed == sweep.verify("demo", 8, "0.2", "0.3", u)["tableau_digests"]
```
