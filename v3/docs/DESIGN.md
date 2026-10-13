# SparseGF2 v3: design

Status: **living design document**, updated at 3.0.0.dev9. It was first
written at milestone 1 as a proposal; section 12 records the decisions you
made, and the text below describes what is built (with the reasons) and what
is planned (section 14). The audit this design responds to is in
[`AUDIT.md`](AUDIT.md) (an audit of the old package, v2, which still lives in
the repository root as `sparsegf2`); the literature review behind the feature
set is in [`LITERATURE.md`](LITERATURE.md).

---

## 1. Goals

From your brief:

1. **Easy to use.** You should be able to read a script and know exactly what
   circuit ran, with what randomness, and what was measured.
2. **Fast.** Faster than Stim on monitored (measurement-heavy) circuits,
   measured fairly, and fast enough that ensembles of many thousands of
   monitored trajectories, with measurements of the state along the way, are
   routine.
3. **Checked against Stim at every step.** Every operation the simulator can
   do has a test that compares it with Stim.
4. **More options.** Measurement records, optional phase (sign) tracking, and
   the rest of Stim's circuit language (resets, noise, detectors), so that the
   same tool covers monitored circuits and things like surface codes.
5. **Built in steps you can follow.** Small milestones, each one runnable and
   tested on your machine before the next begins.

Non-goals for now: replacing Stim's bulk detector sampler (it uses a different
algorithm, Pauli-frame simulation, which is the right tool for sampling
millions of QEC shots), GPU support, and non-Clifford gates.

## 2. Principles

1. **One circuit language.** Every circuit is a `Circuit`, written in Stim's
   syntax (plus a few extensions, section 6), and every random monitored
   circuit the ensemble engine simulates can be exported as one
   (`spec.circuit(t, seed=...)`), so what ran is always inspectable. The one
   exception to "circuit families are functions that return circuits" is the
   ensemble engine (section 8): for speed it generates each trajectory's
   circuit *inside* the compiled loop, so its protocol is a frozen, validated
   description (`mipt.MonitoredCircuit`) instead of a circuit. The tests check
   that the engine and the exported circuits agree exactly.
2. **One state object with explicit modes.** `Simulator(n, phases=...)`. What is
   tracked is visible at construction.
3. **No fake data.** If a quantity is not defined in a mode, asking for it
   raises an error that says why. A phase-free measurement has no outcome, so
   none is invented.
4. **Compiled execution.** A circuit is compiled to flat arrays and run in one
   Numba call. Python never runs once per gate.
5. **Stim is the oracle.** Tests compare against Stim: exact outcomes and signs
   when phases are tracked, exact stabilizer groups when they are not.
6. **Explicit randomness.** Circuit structure and measurement outcomes come from
   separate, named seeds. Nothing silently defaults to seed 0.
7. **Small public API.** If the fast path is needed for research, it is public
   and documented. No research code should ever need a leading-underscore
   method.
8. **Features for studies, not studies.** The library provides what studies
   measure (probes, saved codes, estimators); a study is a script on top. Only
   features a study needs now are built, and each kind of feature has one
   place to extend: a new probe is one table entry and one kernel branch
   (section 8.3).

## 3. What using it looks like

```python
import sparsegf2_v3 as sg

# Interactive use, mirroring stim.TableauSimulator method names.
sim = sg.Simulator(3, seed=7)          # phases=True (default): exact, sign-tracking, Stim-identical
sim.h(0)
sim.cx(0, 1)
sim.cx(1, 2)
sim.measure(0)                         # 0 or 1: a physical outcome
sim.peek_z(2)                          # +1 / -1 if deterministic, 0 if random (like Stim)
sim.stabilizers()                      # canonical list of PauliStrings, with signs
sim.entropy([0])                       # entanglement entropy in bits

# Phase-free mode: entanglement dynamics only; half the rows, no signs.
fast = sg.Simulator(1024, phases=False, seed=7)
fast.measure(5)                        # returns None: no outcome exists without phases

# Circuits, in Stim's syntax.
circuit = sg.Circuit("""
    H 0
    CX 0 1
    M 0 1
    DETECTOR rec[-1] rec[-2]
""")
result = sg.Simulator(2, seed=1).run(circuit)   # one compiled call
result.measurements                    # array([b0, b1]): outcome bits
result.random                          # array([True, False]): which outcomes were random
result.detectors                       # array([0])
shots = sg.sample(circuit, shots=10_000, seed=1)  # many shots in parallel

# Interop both ways.
circuit = sg.Circuit.from_stim(stim_circuit)
stim_circuit = circuit.to_stim()

# Ensembles of random monitored circuits: describe the protocol, run it.
from sparsegf2_v3 import graphs, mipt
spec = mipt.MonitoredCircuit(
    num_system=128,
    edges=[graphs.watts_strogatz(128, 4, 0.1, seed=(11, g)) for g in range(250)],
    circuits_per_graph=4,                          # 250 graphs x 4 circuits = 1000 trajectories
    layers=1024, measurement_rate=0.16, schedule="random_edges",
    reference="single", reference_partner=127, scramble_matchings=32,
    checkpoints="every_layer", stop_when_purified=True,
    probes=[mipt.code_dimension()],
)
result = spec.run(seed=12)                         # all 1000, compiled, parallel over cores
result.purification_layer                          # tau per trajectory, -1 if censored
result.graph_index                                 # the graph of each trajectory (t // 4)
spec.circuit(trajectory=3, seed=12)                # the exact Circuit trajectory 3 ran
```

## 4. Architecture

```
sparsegf2_v3/
  pauli.py        PauliString: Pauli operators with exact phases
  clifford.py     Clifford tableaux; canonical 1q/2q Clifford groups and indices
  gates.py        the gate library, Stim names and conventions
  _kernel/        Numba kernels
    bits.py, rng.py, tables.py   bit tricks, xoshiro256** streams, gate tables
    tableau.py    layout, gates, measurement (both modes), peeks, canonical form
    program.py    the compiled circuit interpreter (one call per run / per shot)
    entropy.py    ranks, entanglement profiles, distance, clipped gauge, and the
                  erasure kernels (adapted code columns, first failures, exhaustive counts)
    ensemble.py   the monitored-circuit engine (generate + simulate + probe)
  simulator.py    Simulator, the public state object
  circuit.py      Circuit: Stim-syntax IR, parse/print, from/to Stim
  _compile.py     Circuit -> flat instruction arrays (cached per circuit version)
  result.py       RunResult / SampleResult: record, random flags, detectors, observables
  sampling.py     sample(): many shots in parallel
  observables.py  entropy, mutual/tripartite information, code dimension, erasure, distance
  graphs.py       geometries as edge arrays (cycle, lattice, WS, NW, ER, regular, complete)
  mipt.py         MonitoredCircuit protocols, the probe registry, EnsembleResult
  codes.py        StabilizerCode: erasure analysis of a code after the run
  analysis.py     mean/error, Kaplan-Meier survival, P_C(m), Phi(e), e_delta
  api.py          JSON-in/JSON-out service layer shared by the CLI and MCP server
  cli.py          the `sparsegf2` command
  mcp_server.py   the `sparsegf2-mcp` Model Context Protocol server
```

Layers only depend downward: kernels know nothing about circuits; circuits know
nothing about MIPT; `api.py` only calls the public library.

## 5. The kernel

### 5.1 The data layout (decided by measurement, not taste)

During the audit I built candidate kernels and ran them on identical brickwork
circuits (random two-qubit Cliffords on a ring, Bernoulli-p Z measurements,
depth 4n), each driven from a single compiled loop so Python overhead is
excluded. Two layouts were in the running:

* **dual**: bit-packed rows *and* bit-packed columns, stabilizer rows only.
* **column**: bit-packed columns only, stabilizer rows only. For each qubit,
  store *which generators have an X there* and *which have a Z there*, each as a
  bitset over generators. This is the transpose of the usual picture.

Both produce identical stabilizer groups (checked against each other and
against Stim). Wall-clock seconds, phase-free, single runs (the n = 128
entries are a few milliseconds and therefore noisy):

| n | p | dual | **column** | Stim (`do_circuit`) | column vs Stim |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.05 | 0.100 | **0.002** | 0.032 | 16x |
| 128 | 0.16 | 0.030 | **0.005** | 0.039 | 8x |
| 128 | 0.30 | 0.015 | **0.005** | 0.052 | 10x |
| 256 | 0.05 | 0.710 | **0.012** | 0.138 | 12x |
| 256 | 0.16 | 0.171 | **0.022** | 0.240 | 11x |
| 256 | 0.30 | 0.058 | **0.024** | 0.349 | 15x |
| 512 | 0.05 | 5.396 | **0.086** | 1.034 | 12x |
| 512 | 0.10 | 3.040 | **0.137** | 1.537 | 11x |
| 512 | 0.16 | 0.700 | **0.133** | 2.078 | 16x |
| 512 | 0.20 | 0.347 | **0.151** | 2.342 | 16x |
| 512 | 0.30 | 0.270 | **0.219** | 3.117 | 14x |

And at large n, deep in the area-law phase (depth n, single run):

| n | p | dual | column |
|---:|---:|---:|---:|
| 2048 | 0.2 | **1.46** | 3.42 |
| 2048 | 0.3 | **1.23** | 4.50 |
| 2048 | 0.5 | **1.17** | 6.79 |

The scripts that produced every number here are in
[`benchmarks/prototypes/`](../benchmarks/prototypes/README.md).

**Caveats, stated plainly.** These prototypes track neither signs nor
destabilizers, and Stim tracks both, so the Stim column is not like-for-like.
Destabilizers double the number of rows, so they should roughly double the
gate cost; signs add a constant factor. The like-for-like comparison (v3 with
phases against Stim) was made in milestone 5; its results are in
[`benchmarks/`](../benchmarks/README.md) and the README.

**Why the column layout wins.**

* A gate on qubits (a, b) rewrites four bitsets (X and Z for a and b) with
  word-parallel boolean operations: about n/64 machine words, regardless of how
  entangled the state is. This is how Stim does gates. In a row-sparse layout
  (each generator stored as a list of the qubits it touches, plus per-qubit
  indices) a gate costs time proportional to the number of generators touching
  a and b, which is about n in the volume-law phase.
* A Z-measurement finds the anticommuting generators by reading one bitset
  (X-bits of the measured qubit), then multiplies the pivot into all of them at
  once. For each qubit in the pivot's support, it XORs a single mask into that
  qubit's bitsets. The cost is (pivot weight) x R/64 words, where R is the
  number of stored rows. Sparse stabilizers still make measurements cheap,
  without any index bookkeeping.
* Memory is two R-bit bitsets per qubit: n²/4 bytes phase-free (R = n rows),
  n²/2 bytes with destabilizers (R = 2n). At n = 1024 that is 256 KB or
  512 KB: a whole n = 1024 state fits in a CPU cache.
* There is **one representation for every regime**: no switching between
  sparse and dense storage, and nothing that has to branch on it.

The name stays SparseGF2, but v3's storage is dense bit-packed. What remains
"sparse" is that measurement cost scales with stabilizer weight.

### 5.2 Modes

| mode | rows stored | signs | `measure()` returns | use it for |
|---|---|---|---|---|
| `phases=True` (default) | n stabilizers + n destabilizers | yes | the physical outcome | anything involving outcomes, detectors, signs, QEC; exact agreement with Stim |
| `phases=False` | n stabilizers only | no | `None` | entanglement dynamics (entropies, code dimension, purification time) at maximum speed |

Destabilizers are only needed to compute deterministic outcomes efficiently,
and those only exist when signs are tracked. That is why the fast mode drops
both. An optional `destabilizers=True` for the phase-free mode (needed by
expurgation, which uses stabilizer/destabilizer pairs) is easy to add when
that module is ported.

### 5.3 Operations and their cost

R is the number of stored rows (n or 2n); "words" means 64-bit operations.

| operation | cost |
|---|---|
| 1- or 2-qubit Clifford gate | R/64 words |
| Pauli gate / Pauli noise (phases mode) | R/64 words (sign flips) |
| Z/X/Y measurement, deterministic, phase-free | R/64 words |
| Z/X/Y measurement, random | R/64 + n bit reads + wt(pivot) * R/64 words |
| deterministic outcome (phases mode) | product of the stabilizers flagged by the destabilizers |
| general Pauli measurement (MPP) | wt(P) * R/64 to find anticommuters, then as above |
| single-qubit entropy | R/64 words |
| entropy of a region A | GF(2) rank of 2·abs(A) bitsets: about abs(A)² * n/64 words |

Pivot rule: the anticommuting stabilizer with the lowest row index. The other
natural choice, the lightest one, costs an extra scan per candidate in this
layout, and in the benchmarks it was 2-11x slower overall (the "lightest"
variant is in the prototype scripts). The pivot choice changes which
generators you get, never the state or any physical observable.

The ensemble engine refines the rule when there is a reference R (qubits no
operation touches): it takes the first anticommuting row *without* support on
R if there is one. That keeps an invariant: the rows touching R (a bitset,
`ref_rows`) have linearly independent R-parts. Then the code dimension is
`popcount(ref_rows) - |R|`, the unmarked rows generate the code's stabilizer
group, and the information an erasure E leaks is
`I(R:E) = rank(E) - rank(E restricted to unmarked rows)`, so reference
columns never enter an elimination. This made k(t) at every layer free (41 to
13 us per layer at n = 128 in the purification picture) and erasure profiles
about 2x faster. The proof is in the docstring of
`measure_1q_phase_free_tracked`; the tests compare against the generic
formulas on random codes. The erasure kernels of section 8.4 build on the same
invariant.

**Known weak spot, and the planned fix.** The second table shows it: at
n = 2048 deep in the area-law phase, the column layout falls behind the dual
layout, because every random measurement scans all n qubits to find the
pivot's support. The fix is a **row-major cache** on top of the column layout.
The column bitsets stay authoritative. A gate that touches only a few rows
(the area-law case) also patches those rows in the cache. A gate that touches
many rows (the volume-law case) just marks them stale, at a cost of n/64
words. A measurement then reads the pivot's support from the cache when it is
fresh (cost ~ its weight) and rebuilds it from the columns otherwise. That
gives dual-layout speed in the area-law phase and column-layout speed in the
volume-law phase, with no mode switch and no change to the API. With fresh
rows, the lightest-pivot rule becomes cheap again and can be re-evaluated.
For n ≤ 512 the plain column layout is already the fastest by a wide margin,
so milestone 2 started with that layout, the simplest correct design.

### 5.4 Signs

Signs live in one more bitset over rows. Gate sign updates are word-parallel
boolean functions of the four input bitsets, derived from each gate's
conjugation table (`Clifford.conjugation_table()`, already in M1). Multiplying
the pivot into many rows at once tracks the factors of i with a 2-bit counter
per row, also word-parallel. This is the bit-sliced counting trick Stim uses
for products of Pauli strings, applied across rows instead of along one row.

## 6. Circuits

### 6.1 The language

`Circuit` reads and writes Stim's text format and converts losslessly to and
from `stim.Circuit` for every instruction it supports. The supported set grows
by milestone:

* M3: all Clifford gates; `M MX MY MPP MXX MYY MZZ R RX RY MR MRX MRY`; `TICK`; `REPEAT`;
  `DETECTOR`, `OBSERVABLE_INCLUDE`, `QUBIT_COORDS`, `SHIFT_COORDS`; noise
  channels `X_ERROR Y_ERROR Z_ERROR DEPOLARIZE1 DEPOLARIZE2 PAULI_CHANNEL_1
  PAULI_CHANNEL_2`, plus measurement flip noise `M(p)`.
* Later, as needed: `SPP`, `E/ELSE_CORRELATED_ERROR`, heralded channels, `MPAD`.

Anything unsupported raises an error that names the instruction.

Extensions Stim does not have, needed for monitored circuits:

* `CLIFFORD_1Q(i) q` and `CLIFFORD_2Q(i) a b`: the Clifford with canonical
  index `i` (`Clifford.from_index(k, i)`). A uniform random two-qubit gate is a
  uniform `i` in `[0, 11520)`. On export to Stim they are expanded into Stim
  gates (`Tableau.to_circuit("elimination")`).
* Observing the state during a circuit: `Simulator.run(circuit, observe=f)`
  calls `f(sim, tick)` at every `TICK`, so any observable can be recorded as a
  function of depth. For the purification-time workload, where checking the
  reference after every layer from Python would cost more than the layer,
  observation happens inside the compiled ensemble engine instead
  (checkpoints and probes, section 8). No observation instructions were added
  to the language.

Internally a circuit is a Python list of immutable `Instruction` objects (one
per line, each holding a tuple of `Target`s) and `RepeatBlock`s. Building one
therefore runs at Python speed: on the benchmark machine 300,000 random
two-qubit gates take about 3 s appended one per instruction, about 1.5 s with
one 32-gate `CLIFFORD_2Q` instruction per layer, and about 1.4 s to parse from
text. `run()` and `sample()` then compile the circuit once into flat arrays
(about 0.3 s for those 300,000 gates; cached until the circuit or a `REPEAT`
body in it changes), so simulation never runs Python per gate. For random
monitored circuits at scale, `mipt.MonitoredCircuit` generates each circuit
inside the compiled engine instead (section 8).

### 6.2 Randomness and the measurement record

* `Simulator(n, seed=...)` owns the random stream for measurement outcomes and
  noise. Circuit generators take their own seed for circuit structure (which
  gates, where, which qubits are measured). Changing one never changes the
  other.
* **Each measurement result in a circuit consumes exactly one random bit,
  whether or not it turns out to be random.** As a result, a run is reproducible
  from (circuit, seed). Circuits with the same measurement layout share random
  numbers, for example across a p-sweep if you want that. And you can supply
  the bits yourself:
  `sim.run(circuit, coins=bits)` forces every random outcome
  (`sim.measure(q, coin=b)` for a single measurement). This one
  mechanism serves post-selection studies, replaying a recorded experiment, and
  exact comparison with Stim (for a noiseless circuit, feed Stim's
  measurement record in and every outcome must come back identical).
  One restriction: `coins` covers measurement results only. A reset (`R`,
  `RX`, `RY`) collapses its qubit with a bit from the simulator's own stream,
  so a reset of a qubit that is not already in an eigenstate of the reset
  basis (one entangled with others, say) is an unrecorded random event, and
  later outcomes can differ from the recorded ones. Replay is exact when every
  reset acts on such an eigenstate: resets at the start of a circuit, or right
  after a measurement in the same basis, as in QEC circuits. `MR`, `MRX` and
  `MRY` are always fine (their result is recorded).
* `RunResult.random[k]` says whether measurement k was random. It is available
  in both modes; it is a genuine physical property of the trajectory.
  `RunResult.measurements` needs `phases=True` and otherwise raises.

## 7. Observables

`sparsegf2_v3.observables`, over any simulator state: `entropy`,
`mutual_information`, `tripartite_information`, `entropy_profile` (all
bipartitions of an ordering in one pass), `code_dimension`,
`lost_logical_information` I(R:E), `erasure_is_correctable`,
`recovery_probability` (2^-I(R:E)), `random_erasure_statistics`,
`erasure_profile` (I(R:E_m) for every size m along one random ordering),
`contiguous_code_distance`, `clipped_gauge` and
`stabilizer_length_distribution`, `stabilizer_weights`; plus
`sim.stabilizers()` / `sim.stabilizer_arrays()` / `sim.destabilizers()`. The
column layout stores, for each qubit, its X and Z bitsets over stabilizers, so
the entropy of a region is the GF(2) rank of the 2·|A| bitsets of its qubits,
with no transposition. Every observable is checked against a state-vector
calculation (SVD of the reduced state) on small systems or against brute
force. A code on its own, without a simulator state, is a
`codes.StabilizerCode` (section 8.5).

## 8. Monitored circuits

`mipt.MonitoredCircuit` describes a protocol: graph (one, or a collection of
graphs each run with `circuits_per_graph` circuits, section 8.1), schedule
(`matching` = brickwork on any graph through an edge colouring,
`random_edges` with optional edge weights for long-range circuits,
`all_edges`), measurement rate, picture (`none`, `purification`, `single`
reference with a fixed or random partner), scramble (random pairs, random
all-to-all perfect matchings, gate-only layers), gate set, checkpoints, probes
(each on the spec's checkpoints or its own, section 8.2) and early stopping.
`spec.run(trajectories, seed=...)` generates and simulates every trajectory
inside one compiled, parallel kernel and records every probe at its
checkpoints; `EnsembleResult` holds the arrays (`values`, `layers(name)`,
`survival()`, `graph_index`, `snapshots` and `code(t, c)`, `save`/`load`,
`concatenate`).
[`examples/purification_time.py`](../examples/purification_time.py) measures
the single-reference purification time on a collection of Watts-Strogatz
graphs, and [`examples/code_and_erasure.py`](../examples/code_and_erasure.py)
follows the generated code and its erasure protection with depth.

Why a protocol object here, against principle 1: generating circuits in Python
and handing them to the simulator costs more than simulating them at the sizes
the studies use, and checking the reference from Python after every layer
costs more than the layer. The engine therefore draws each layer from the
trajectory's own random stream and simulates it immediately. Two safeguards
keep it honest: `spec.circuit(t, seed)` exports exactly what trajectory `t`
ran (the generator code is shared), and the tests replay exported circuits
through the general `Simulator` and require every probe value to agree at
every checkpoint.

Requirements carried over from the audit, all met:

* graph disorder, circuit disorder and probe sampling use separate seeds, and
  trajectory `t` is a deterministic function of `(spec, seed, t)` whatever the
  thread count or chunking;
* graphs are plain edge arrays generated with SparseGF2's own PRNG;
* the reference site is configurable, including random per trajectory;
* "layer" always means what you wrote, with no implicit rescaling;
* censoring is explicit (`purification_layer == -1`), and `analysis` has
  Kaplan-Meier with Greenwood errors.

Validation: the engine equals the Simulator on exported circuits, probe by
probe at every checkpoint (exact). The sampled erasure probes are checked
against the codes the same run saved (section 8.5), and the exhaustive counts
against brute force over every subset.

### 8.1 Graph collections

`edges=[g_0, ..., g_{G-1}]` with `circuits_per_graph=c` (default 1) defines
G·c trajectories: trajectory `t` runs on graph `t // c`, so graph `g` carries
the block of trajectories `g·c .. g·c + c - 1` (`EnsembleResult.graph_index`).
`spec.num_trajectories` is G·c, and `spec.run(seed=...)` without a count runs
them all; `run(trajectories, seed=..., first_trajectory=...)` still runs any
contiguous chunk, and asking past G·c is an error. With a single graph,
`num_trajectories` is `None`, `run()` needs the count and `circuits_per_graph`
must be 1. A graph is never redrawn during a trajectory (quenched disorder);
`spec.with_edges(...)` gives the same protocol on other graphs.

The mapping is blocked (`t // c`) rather than cyclic (`t mod G`, the rule
before 3.0.0.dev9) for two reasons:

* **One graph's circuits are contiguous.** Chunks (`first_trajectory`, the
  background jobs' chunk files, `concatenate`) hold whole graphs except at
  their two ends, and averaging over circuits first and graphs second is a
  reshape of a full run's `values[name]` to `(G, c, ...)`.
* **Trajectory `t` depends on `t` and `c`, not on G.** Graph `i` is built from
  its own seed (`graphs.watts_strogatz(n, 4, beta, seed=(17, i))`), so adding
  graphs to a collection appends trajectories and leaves the earlier ones
  unchanged.

Graphs carry their own seeds and never draw from the circuit's streams, so
one collection serves every measurement rate. With the same circuit seed the
gates and random numbers are shared too: each layer draws its gates, then one
uniform number per system qubit for the measurement mask whatever the rate, so
the qubits measured at a lower rate are a subset of those measured at a
higher one (common random numbers across a rate scan). In the JSON interface,
`graph: {..., "realizations": G}` builds the collection (graph `i` from seed
`[seed, i]`) and the spec key `circuits_per_graph` sets c.

### 8.2 Probes, schedules and the output layout

A probe is a frozen `Probe` (kind, name, regions, size, samples, checkpoints)
made by a factory: `entropy`, `half_cut_entropy`, `mutual_information`,
`tripartite_information`, `entropy_profile`, `code_dimension`,
`contiguous_distance`, `lost_information(region)`,
`erasure_lost_profile(samples=10)`, `erasure_first_failure(samples=100)`,
`uncorrectable_region_counts(max_size=2)` and `code_snapshot()`. Every factory
takes `checkpoints=` in the same forms as the spec's (`"final"`,
`"every_layer"`, a positive stride, a list of layers); without it the probe
follows the spec's checkpoints. `spec.probe_layers(name)` and
`result.layers(name)` give the layers. So a cheap probe can run every layer
while an expensive one runs at a few depths.

How the engine handles them (`_kernel/ensemble.py`):

* **Flat rows.** Probe `j` writes `plen[j]` numbers per evaluation (1 for a
  scalar, n + 1 for a profile, `samples` or `size` for the erasure vectors)
  into its own slice of the trajectory's output row: evaluation `c` fills
  `out[t, poff[j] + c·plen[j] : ... + plen[j]]`. `values[name]` is that slice
  reshaped to `(trajectories, evaluations)` or `(trajectories, evaluations,
  length)`.
* **Schedules.** Each probe's layers are sorted (`pck[pck_ptr[j]:pck_ptr[j +
  1]]`). The layer loop stops at the union of all schedules and the spec's
  checkpoints (`gck`); at each such layer it reads k = S(R) once (a popcount
  of the reference rows) and evaluates exactly the probes whose next
  scheduled layer it is.
* **Purification** is tested only at the spec's own checkpoints (`detect`),
  so the purification layer does not depend on which probes are attached or
  on their schedules.
* **After a stop.** When `stop_when_purified` ends a trajectory, its remaining
  evaluations get `pfill[j]`, the probe's exact value for a pure reference:
  0 for `code_dimension`, `lost_information`, `erasure_lost_profile` and
  `uncorrectable_region_counts`; n + 1 for `erasure_first_failure`; n for
  `code_snapshot` (with no code stored, snapshot count -1). Entanglement
  probes (`entropy`, `half_cut_entropy`, `mutual_information`,
  `tripartite_information`, `entropy_profile`) are NaN there, because the
  system keeps evolving and the stopped trajectory no longer follows it; so is
  `contiguous_distance`, which is NaN wherever k = 0. Averages of code rates
  and erasure statistics over all trajectories therefore need no NaN
  handling.
* **Snapshots.** `code_snapshot` writes the code's generators (the rows outside
  `ref_rows`, packed as X words then Z words over the system qubits) into a
  separate buffer of shape `(trajectories, evaluations, n, 2⌈n/64⌉)` and their
  number into a count array; the flat row holds the number, n - k. There is
  at most one snapshot probe per spec, and without one the buffer is a single
  dummy slot. A code at n = 256 takes 16 KB per evaluation, which is why the
  snapshot usually gets its own `checkpoints="final"`.
* **Randomness.** The sampling probes (`erasure_lost_profile`,
  `erasure_first_failure`) draw from the trajectory's probe stream, a key
  separate from the circuit's, in probe order at each evaluation. The circuit
  never depends on the probes; the samples of one sampling probe do depend on
  which sampling probes come before it.

### 8.3 Adding a probe

The probe registry is the `_KINDS` table in `mipt.py`, one `_Kind` entry per
probe kind:

| field | meaning |
|---|---|
| `code` | the kernel code (`_kernel.ensemble.PROBE_*`) that `eval_probe` branches on |
| `reference` | whether the probe needs a reference (`"purification"` or `"single"`); checked when the spec is built |
| `length` | values per evaluation: `"one"`, `"profile"` (n + 1), `"samples"` (`Probe.samples`) or `"size"` (`Probe.size`); `_probe_length` applies it |
| `after_purification` | the value recorded after a stop: `"nan"`, `"zero"`, `"n"` or `"n+1"`; `_probe_fill` applies it |
| `sampled` | whether the probe draws random orderings (`Probe.samples` must then be at least 1) |

Offsets, schedules, post-stop values and the result arrays all derive from
this table, so adding a probe takes these steps and nothing else in the
engine:

1. **The computation**: a compiled routine, usually in `_kernel/entropy.py`
   beside the `*_rows` routines, which work on the tableau and the reference
   rows (k is a popcount, I(R:E) = rank(E) - rank(E over the unmarked rows),
   and `code_columns` gives the S and N(S) ranks of many sets cheaply,
   section 8.4).
2. **The kernel branch**: a new constant `PROBE_...` in `_kernel/ensemble.py`
   and a branch in `eval_probe` that fills `row`, the probe's slice for this
   evaluation (`row[0]` for a scalar). The branch receives the tableau,
   `ref_rows`, `s_ref` (k, read once per checkpoint), its regions (`idx` from
   `ptr[j]`, lengths `region_len[j]`), `size_param[j]`, `sample_param[j]`,
   `order` (the spec's `system_order`), `periodic`, the probe stream `prng` and
   work buffers.
3. **The registry entry**: one line in `_KINDS`; a new length or post-stop
   rule is one more key in `_probe_length` or `_probe_fill`.
4. **The factory**: a function returning `Probe(kind, default_name, regions,
   size, samples, _schedule(checkpoints))`, with a keyword `checkpoints=None`,
   listed in `mipt.__all__`. Parameter checks go in
   `MonitoredCircuit._resolve_probe`.
5. **The interfaces**: `api.py` discovers probe kinds from `mipt.__all__`
   (every function annotated to return `Probe`), so the CLI and the MCP
   server accept the new kind at once. Add a cost line to `api._probe_us`
   (otherwise a generic estimate is used) and, for a vector probe, a `what`
   line in `api._VECTOR_MEANING`.
6. **The tests**: add the probe to a spec in `tests/test_mipt.py` and its
   value on a `Simulator` state to `_replay`, which runs the exported circuits
   and requires the engine to agree at every checkpoint. A sampled probe goes
   in `RANDOM_KINDS` and gets a statistical or brute-force test instead.

### 8.4 Erasure kernels

With a reference R holding the logical information, erasing a set A of system
qubits loses `l(A) = I(A:R) = rank(N(S)|_A) - rank(S|_A)` bits, where S is
the code's stabilizer group (on the system, the rows outside `ref_rows`),
N(S) its normalizer (spanned, on the system, by all rows) and `G|_A` the
restriction of a group's generators to A. The erasure is correctable exactly
when `l(A) = 0`, and `l` never decreases when qubits are added (`I(A:R)` is
monotone in A). The erasure studied is independent erasure of every qubit
with probability e at the end of the circuit, so everything below acts on the
code at a checkpoint and never changes the dynamics.

**Adapted columns (`code_columns`).** The rows restricted to the system are
row-reduced, the rows outside `ref_rows` first: the first s = n - k echelon
rows span S and all r = n + k span N(S). The echelon is then transposed, so
qubit q has an X column and a Z column of r bits. Row operations leave the
ranks of column sets unchanged, so for any A, `rank(N(S)|_A)` is the rank of
A's 2|A| columns and `rank(S|_A)` the rank of the same columns cut to their
first s bits. One array of ⌈r/64⌉-word columns serves both ranks, and the
two routines below work only on it.

**First-failure sizes (`first_failure_samples`).** For each of M orderings (a
Fisher-Yates shuffle from the probe stream), qubits are added one at a time to
an incremental basis of their S-columns. While the S-rank equals 2m the
prefix is correctable, since `l = rank_N - rank_S` and `rank_N ≤ 2m`, and the
N(S) rank is never computed. At the first prefix whose S-rank falls short,
the N(S) basis is built once, from the whole prefix; S-rank deficiency then
persists, so from there both bases grow together until `rank_N > rank_S`: the
first-failure size D. The scan stops at D. With k = 0 every entry is n + 1
and no random numbers are drawn. The prefixes of one ordering are uniformly
random m-sets for every m, so the fraction of samples with D ≤ m estimates
P_C(m), the probability that a random m-qubit erasure is uncorrectable, for
all m at once, and the failure probability under independent erasure is
`Phi_C(e) = sum_m C(n,m) e^m (1-e)^(n-m) P_C(m) = mean_j Pr[Bin(n, e) >= D_j]`.

**Exhaustive counts (`uncorrectable_counts_rows`).** The number of m-subsets
with `l > 0`, for m = 1..w:

* *Depth-first walk with pruning.* Prefixes (sets in increasing index order)
  of up to w - 1 qubits are walked depth-first, each level keeping a small
  echelon basis of the prefix's S-columns. Monotonicity prunes the walk: when
  a prefix P with largest index i fails, every extension fails too, and the
  kernel adds C(n - 1 - i, j) to the count of size |P| + j for every j instead
  of visiting them.
* *Lazy N(S) basis.* A prefix whose S-columns have full rank 2|P| is
  correctable (both ranks are at most 2|P|), so its N(S) basis is not
  extended. It is rebuilt from the deepest level that has one only when a
  descendant's S-rank falls short or the last-qubit step needs it.
* *The last qubit by lookup.* For a correctable prefix P of w - 1 qubits, the
  last qubit c is not looped over. Adding c's columns x_c, z_c raises a rank
  by `2 - dim K`, where `K = {(a, b) : a x_c + b z_c ∈ span(P's columns)}`, so
  `l(P + c) = dim K_S - dim K_N` (K_S with the columns cut to s bits, K_N with
  full columns). A relation among full columns survives the cut, so
  `K_N ⊆ K_S`, and `l(P + c) > 0` exactly when `K_S != K_N`. A nonzero
  element of K means one of x_c, z_c, y_c = x_c + z_c lies in the span, so the
  kernel walks the at most 4^(w-1) elements of P's S-span (Gray-code order)
  and looks each one up in a hash table of every qubit's X, Z and Y columns
  (cut to s bits). That marks K_S for all later qubits at once. A qubit with
  K_S = {0} cannot fail; for the few others the same lookup over P's
  N(S)-span gives K_N, and c is counted where the two differ.

The walk visits about C(n, w - 1) prefixes with up to 4^(w-1) lookups each,
instead of C(n, w) rank computations.

Measured costs per evaluation on one thread, at n = 256, for codes generated
by random circuits (the engine runs trajectories in parallel over all cores):

| probe | cost per evaluation |
|---|---|
| `erasure_first_failure`, M = 100 samples | 3 to 7 ms |
| `uncorrectable_region_counts`, w = 2 | about 1 ms |
| `uncorrectable_region_counts`, w = 3 | 40 to 50 ms |
| `uncorrectable_region_counts`, w = 4 | about 7 s (C(256, 4) = 1.7·10^8 sets; practical only at a few depths) |
| `code_snapshot` | under 1 ms |

### 8.5 Codes after the run

`codes.StabilizerCode` (also `sparsegf2_v3.StabilizerCode`) is a code given by
its generators: an (s, 2n) 0/1 array (X bits, then Z bits) or Pauli strings,
commuting but not necessarily independent; `from_packed` takes a snapshot's
packed rows, which is how `EnsembleResult.code(t, c)` returns a saved code. It
computes a basis of N(S) (the symplectic complement of the generators),
stacks the S generators and the N(S) basis into one column-major tableau, and
marks the N(S) rows as reference rows. The engine's kernels then apply
unchanged: the unmarked rows span S and all rows span N(S), which contains S,
so `lost_information_rows` returns `rank(N(S)|_A) - rank(S|_A) = l(A)`, and
`first_failure_samples` and `uncorrectable_counts_rows` give the same
quantities as the probes (`code_columns` discards redundant generators). In
terms of the check matrix H, `l(A) = 2|A| - s - rank(H_A) + rank(H_Abar)`. A
saved code can therefore be analysed later with more orderings
(`first_failure_sizes(samples, seed=...)`) or a larger exhaustive search
(`uncorrectable_counts(w)`) than the run used.

`analysis` turns the arrays into the quantities a study reports:
`failure_by_size` (P_C(m) from first-failure samples), `exact_failure_by_size`
(counts / C(n, m)), `erasure_failure_probability` (Phi_C(e); with
`exact_counts`, the hybrid estimator
`sum_{m<=w} b_m P_exact(m) + mean_j Pr[Bin(n, e) >= max(D_j, w + 1)]`, with
b_m the binomial weights, exact for the small erasures that sampling resolves
worst) and `erasure_tolerance` (`e_delta = sup{e : Phi(e) <= delta}` for an
ensemble, by bisection, since Phi is non-decreasing in e). A code with k = 0
has D = n + 1 and Phi = 0; conditioning on k > 0 is a selection of
trajectories before the call.

## 9. QEC and other Clifford circuits (M7)

Any Stim circuit in the supported subset runs in v3: one trajectory per run,
noise sampled, detectors and observables computed from the record; `sample()`
runs many shots in parallel. Example:
`Circuit.from_stim(stim.Circuit.generated("surface_code:rotated_memory_z", ...))`.
Detection-event statistics are tested against Stim's own sampler. Detection
events follow Stim's convention (XOR with a noiseless reference sample, every
random collapse biased to +1). For deterministic detectors and observables
(the ones a noiseless circuit fixes) they equal Stim's exactly. The reference
record itself can differ from Stim's `reference_sample()` when one
instruction measures or resets several entangled qubits in the X or Y basis
(Stim collapses those in ascending qubit order, v3 in target order; in the Z
basis both use target order), so per-shot values of non-deterministic
detectors can differ from Stim's while having the same distribution.

What v3 adds over Stim here is access to the state *during* the circuit
(stabilizers, entanglement, code properties) and custom monitored dynamics. For
raw sampling speed on millions of shots, Stim's frame sampler remains the right
tool; a frame sampler in v3 could be a later milestone if you want one.

## 10. Testing strategy

1. **Stim is required to run the tests.** If it is missing the suite fails
   with an install hint; it never silently skips.
2. **Gate definitions** are checked entry-by-entry against `stim.gate_data()`,
   plus an independent check against textbook unitary matrices (M1, done).
3. **Lock-step harness (M2).** Random circuits run instruction by instruction
   in v3 and in `stim.TableauSimulator`. After every instruction we compare:
   * outcomes, with random outcomes forced to agree via Stim's `postselect_*`;
   * which outcomes were random (Stim: `peek_* == 0`);
   * the canonical stabilizer list from `canonical_stabilizers()`, with signs in
     phases mode and without in phase-free mode;
   * internal invariants (stabilizers commute; destabilizer pairing is
     symplectic).
4. **Whole-circuit replay (M3).** Stim's measurement record is fed to v3 as
   forced outcomes, and the records and final states must be identical.
5. **Fuzzing.** Random circuits over every supported instruction, sizes 1 to
   about 200 including the 63/64/65 and 127/128/129 word boundaries, starting from
   |0...0> and from random states (`stim.Tableau.random`).
6. **Independent of Stim.** State-vector simulation for n ≤ 10: entropies by
   SVD, outcome probabilities.
7. **Statistics** for noisy circuits against Stim's sampler (M3, M7).
8. **The ensemble engine** against the `Simulator` running its exported
   circuits, probe by probe at every checkpoint (M6).
9. **Erasure kernels** against brute force: `l(A)` and the exhaustive counts
   against the generic formula over every subset of small codes (including
   codes wider than one 64-bit word), the first-failure law against the exact
   P_C(m), and the engine's samples against the codes the same run saved
   (M9).
10. **CI** gets a v3 job with Stim required.

## 11. Milestones

| | milestone | status |
|---|---|---|
| **M1** | Foundations: Pauli strings, Cliffords, the gate library | done |
| **M2** | Kernel + `Simulator`, both modes | done (lock-step fuzzing vs Stim with exact signs and outcomes) |
| **M3** | `Circuit` + compiled `run()` / `sample()` | done (whole-circuit replay identical to Stim when resets act on unentangled qubits, section 6.2; noise statistics match) |
| **M4** | Observables incl. erasure protection | done (state-vector and brute-force checks) |
| **M5** | Performance | benchmarks published (`benchmarks/`, README); row cache planned (section 5.3) |
| **M6** | Monitored circuits | done: ensemble engine, graph collections with several circuits per graph, pictures, scrambles, probes on their own schedules, survival analysis |
| **M7** | QEC | circuits run and match Stim's detector statistics; a frame sampler is not planned |
| **M8** | Studies + docs | in progress: examples, agent interface (section 13) |
| **M9** | Codes and erasure | done (3.0.0.dev9): first-failure sizes, exhaustive counts, code snapshots, `StabilizerCode`, the Phi(e) and e_delta estimators (sections 8.4, 8.5) |

## 12. Decisions

Your answers to the milestone-1 questions and your later direction, and what
they mean in the code.

1. **Where the code lives:** `v3/`, import name `sparsegf2_v3`, beside v2.
   After M6 it takes over the `sparsegf2` name as version 3.0.
2. **Default mode:** `phases=True`. The ensemble engine always runs
   phase-free (entanglement does not depend on outcomes), so your studies get
   the speed without passing anything.
3. **No outcomes without phases:** phase-free `measure()` returns `None`,
   `RunResult.measurements` raises `PhaseFreeError`, and `RunResult.random`
   (which outcomes were random) works in both modes. Pauli gates and Pauli
   noise are exact no-ops on a phase-free state and are accepted.
4. **Pivot rule:** lowest-index anticommuting stabilizer (Simulator), with the
   reference-free preference in the engine (section 5.3). Generator-dependent
   diagnostics are not offered; the clipped gauge is the gauge-invariant way
   to look at stabilizer lengths.
5. **v2 extras:** expurgation is still to be ported (section 14); `Study` /
   `sweep` are replaced by `MonitoredCircuit` + `EnsembleResult` chunking.
6. **Focus on v3:** new simulation campaigns run on v3. The v3 package carries
   no v2 production protocol, cross-check against v2 or speed comparison with
   v2; the old package stays in the repository root.
7. **Python:** 3.11+.
8. **Graph collections:** a study starts from a collection of graphs and
   evolves each with random circuits at every measurement rate:
   `circuits_per_graph` circuits per graph (default 1), trajectory `t` on graph
   `t // circuits_per_graph` (section 8.1). A graph is never redrawn during a
   trajectory.
9. **Erasure:** the code at the end of the dynamics is what matters; its
   qubits are then erased independently with probability e. The engine records
   what that needs (first-failure sizes, exact counts for small erasures, the
   code itself) and `analysis` computes P_C(m), Phi(e) and e_delta. No
   erasure channel acts during the circuit.
10. **Bare-bones features:** the library implements the measurements studies
    need, not the studies. Superseded probes are removed rather than kept
    beside their replacements (`erasure_lost_information`,
    `erasure_correctable_fraction` and `erasure_correctable_profile` went in
    3.0.0.dev9), and new ones go through the probe registry (section 8.3).

## 13. Interfaces for AI agents

The simulator is meant to be driven by language-model agents as easily as by
people. One service layer, `sparsegf2_v3.api`, takes and returns JSON
(validated arguments, compact summaries with standard errors, errors that say
how to fix the input), and two thin front ends expose it: the `sparsegf2` CLI
(`--json` on every command) and the `sparsegf2-mcp` Model Context Protocol
server (typed tools, background jobs for long ensembles with resumable chunk
files). The JSON spec mirrors `MonitoredCircuit`, including
`circuits_per_graph`, `graph.realizations` for a collection (then
`trajectories` may be omitted) and per-probe `checkpoints`. Summaries show
each probe on its own layers and the erasure probes as P_C(m) curves;
`estimate_ensemble` reports the trajectory count and the result's memory, and
runs whose result would exceed 8 GB (`api.MAX_RESULT_GB`) are refused.
`AGENTS.md` and `llms.txt` describe the workflow for agents;
[`LLM_INTERFACE.md`](LLM_INTERFACE.md) is the reference. The repository's
`.mcp.json` registers the server for Claude Code.

## 14. Roadmap

From the audit, the literature review ([`LITERATURE.md`](LITERATURE.md)) and
the benchmarks, in rough priority order:

* **Row cache** for the area-law phase at large n (section 5.3).
* **Erasure and noise channels during the circuit**: append environment rows
  for erasure / dephasing, information retention I(R:Q) and coherent
  information under noise (the "first time I(R:sys) = 0" replaces tau, which
  noise does not define). Erasure at the end of the circuit, the case studied
  now, is covered by sections 8.4 and 8.5 and needs none of this.
* **Expurgation** (stabilizer/destabilizer pairs in phase-free mode).
* **Measurement schedules**: exactly floor(p n) measurements per layer,
  measurement right after each gate, space-time modulated rates.
* **Graph disorder redrawn every layer** (annealed disorder); quenched
  collections, with several circuits per graph, are done (section 8.1).
* **Multi-reference correlators** and participation entropy.
* **Uniform n-qubit Clifford sampler** for global scramblers and random-code
  baselines.
* **Further probes** as studies need them, through the probe registry
  (section 8.3).
