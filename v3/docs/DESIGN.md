# SparseGF2 v3: design

Status: **living design document**, updated at 3.0.0.dev8. It was first
written at milestone 1 as a proposal; section 12 records the decisions you
made, and the text below describes what is built (with the reasons) and what
is planned (section 14). The audit this design responds to is in
[`AUDIT.md`](AUDIT.md); the literature review behind the feature set is in
[`LITERATURE.md`](LITERATURE.md).

---

## 1. Goals

From your brief:

1. **Easy to use.** You should be able to read a script and know exactly what
   circuit ran, with what randomness, and what was measured.
2. **Fast.** Faster than v2 everywhere, and faster than Stim on monitored
   (measurement-heavy) circuits, measured fairly.
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
   raises an error that says why. v2's coin-flip "outcomes" are gone.
4. **Compiled execution.** A circuit is compiled to flat arrays and run in one
   Numba call. Python never runs once per gate.
5. **Stim is the oracle.** Tests compare against Stim: exact outcomes and signs
   when phases are tracked, exact stabilizer groups when they are not.
6. **Explicit randomness.** Circuit structure and measurement outcomes come from
   separate, named seeds. Nothing silently defaults to seed 0.
7. **Small public API.** If the fast path is needed for research, it is public
   and documented. No research code should ever need a leading-underscore
   method.

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
    edges=[graphs.watts_strogatz(128, 4, 0.1, seed=(11, i)) for i in range(1000)],
    layers=1024, measurement_rate=0.16, schedule="random_edges",
    reference="single", reference_partner=127, scramble_matchings=32,
    checkpoints="every_layer", stop_when_purified=True,
)
result = spec.run(trajectories=1000, seed=12)     # compiled, parallel over cores
result.purification_layer                          # tau per trajectory, -1 if censored
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
    entropy.py    ranks, entanglement profiles, erasure, distance, clipped gauge
    ensemble.py   the monitored-circuit engine (generate + simulate + probe)
  simulator.py    Simulator, the public state object
  circuit.py      Circuit: Stim-syntax IR, parse/print, from/to Stim
  _compile.py     Circuit -> flat instruction arrays (cached per circuit version)
  result.py       RunResult / SampleResult: record, random flags, detectors, observables
  sampling.py     sample(): many shots in parallel
  observables.py  entropy, mutual/tripartite information, code dimension, erasure, distance
  analysis.py     mean/error, Kaplan-Meier survival, Bernoulli erasure curves
  graphs.py       geometries as edge arrays (cycle, lattice, WS, NW, ER, regular, complete)
  mipt.py         MonitoredCircuit protocols, probes, EnsembleResult
  api.py          JSON-in/JSON-out service layer shared by the CLI and MCP server
  cli.py          the `sparsegf2` command
  mcp_server.py   the `sparsegf2-mcp` Model Context Protocol server
```

Layers only depend downward: kernels know nothing about circuits; circuits know
nothing about MIPT; `api.py` only calls the public library.

## 5. The kernel

### 5.1 The data layout (decided by measurement, not taste)

During the audit I built three kernels and ran them on identical brickwork
circuits (random two-qubit Cliffords on a ring, Bernoulli-p Z measurements,
depth 4n), each driven from a single compiled loop so Python overhead is
excluded:

* **v2**: the existing kernels (2n rows, byte table + inverted indices).
* **dual**: bit-packed rows *and* bit-packed columns, stabilizer rows only.
* **column**: bit-packed columns only, stabilizer rows only. For each qubit,
  store *which generators have an X there* and *which have a Z there*, each as a
  bitset over generators. This is the transpose of the usual picture.

All three produce identical stabilizer groups (checked against each other and
against Stim). Wall-clock seconds, phase-free, single runs (the n = 128
entries are a few milliseconds and therefore noisy):

| n | p | v2 kernel | dual | **column** | Stim (`do_circuit`) | column vs v2 | column vs Stim |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.05 | 0.518 | 0.100 | **0.002** | 0.032 | 259x | 16x |
| 128 | 0.16 | 0.198 | 0.030 | **0.005** | 0.039 | 40x | 8x |
| 128 | 0.30 | 0.091 | 0.015 | **0.005** | 0.052 | 18x | 10x |
| 256 | 0.05 | 4.052 | 0.710 | **0.012** | 0.138 | 338x | 12x |
| 256 | 0.16 | 1.045 | 0.171 | **0.022** | 0.240 | 48x | 11x |
| 256 | 0.30 | 0.355 | 0.058 | **0.024** | 0.349 | 15x | 15x |
| 512 | 0.05 | 32.98 | 5.396 | **0.086** | 1.034 | 383x | 12x |
| 512 | 0.10 | 19.02 | 3.040 | **0.137** | 1.537 | 139x | 11x |
| 512 | 0.16 | 4.859 | 0.700 | **0.133** | 2.078 | 37x | 16x |
| 512 | 0.20 | 2.305 | 0.347 | **0.151** | 2.342 | 15x | 16x |
| 512 | 0.30 | 1.632 | 0.270 | **0.219** | 3.117 | 7x | 14x |

And at large n, deep in the area-law phase (depth n, single run):

| n | p | v2 kernel | dual | column |
|---:|---:|---:|---:|---:|
| 2048 | 0.2 | 9.79 | **1.46** | 3.42 |
| 2048 | 0.3 | 6.96 | **1.23** | 4.50 |
| 2048 | 0.5 | 6.26 | **1.17** | 6.79 |

The scripts that produced every number here are in
[`benchmarks/prototypes/`](../benchmarks/prototypes/README.md).

**Caveats, stated plainly.** These prototypes track neither signs nor
destabilizers, and Stim tracks both, so the Stim column is not yet
like-for-like. Destabilizers double the number of rows, so they should
roughly double the gate cost; signs add a constant factor. The fair comparison (v3 with phases against Stim) is part of
milestone 5. The margin over Stim is large enough that I expect v3 with phases
to stay ahead on measurement-heavy circuits, but that is a prediction until
measured.

**Why the column layout wins.**

* A gate on qubits (a, b) rewrites four bitsets (X and Z for a and b) with
  word-parallel boolean operations: about n/64 machine words, regardless of how
  entangled the state is. This is how Stim does gates. In v2 a gate cost time
  proportional to the number of generators touching a and b, which is about n
  in the volume-law phase.
* A Z-measurement finds the anticommuting generators by reading one bitset
  (X-bits of the measured qubit), then multiplies the pivot into all of them at
  once. For each qubit in the pivot's support, it XORs a single mask into that
  qubit's bitsets. The cost is (pivot weight) x R/64 words, where R is the
  number of stored rows. Sparse stabilizers
  still make measurements cheap, as in v2, but without any index bookkeeping.
* Memory is two R-bit bitsets per qubit: n²/4 bytes phase-free (R = n rows),
  n²/2 bytes with destabilizers (R = 2n). At n = 1024 that is 256 KB or 512 KB,
  against about 27 MB for v2.
  A whole n = 1024 state fits in a CPU cache.
* There is **one representation for every regime**. v2's hybrid sparse/dense
  switching, and everything that has to branch on it, disappears.

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

| operation | v3 cost | v2 cost |
|---|---|---|
| 1- or 2-qubit Clifford gate | R/64 words | number of rows touching the qubits |
| Pauli gate / Pauli noise (phases mode) | R/64 words (sign flips) | not supported |
| Z/X/Y measurement, deterministic, phase-free | R/64 words | rows with X there |
| Z/X/Y measurement, random | R/64 + n bit reads + wt(pivot) * R/64 words | anticommuting rows * wt(pivot) * index upkeep |
| deterministic outcome (phases mode) | product of the stabilizers flagged by the destabilizers | not available |
| general Pauli measurement (MPP) | wt(P) * R/64 to find anticommuters, then as above | similar |
| single-qubit entropy | R/64 words | O(n) |
| entropy of a region A | GF(2) rank of 2·abs(A) bitsets: about abs(A)² * n/64 words | similar |

Pivot rule: the anticommuting stabilizer with the lowest row index. v2 used
the lightest one. In the new layout that costs an extra scan per candidate,
and in the benchmarks it was 2-11x slower overall (the "lightest" variant is
in the prototype scripts). The pivot choice changes
which generators you get, never the state or any physical observable.

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
formulas on random codes.

**Known weak spot, and the fix.** The second table shows it: at n = 2048
deep in the area-law phase, the column layout falls behind the dual layout,
because every random measurement scans all n qubits to find the pivot's
support. Milestone 5 adds a **row-major cache** on top of the column layout. The
column bitsets stay authoritative. A gate that touches only a few rows (the
area-law case) also patches those rows in the cache. A gate that touches many
rows (the volume-law case) just marks them stale, at a cost of n/64 words. A measurement then reads
the pivot's support from the cache when it is fresh (cost ~ its weight) and
rebuilds it from the columns otherwise. That gives dual-layout speed in the
area-law phase and column-layout speed in the volume-law phase, with no mode
switch and no change to the API. With fresh rows, the lightest-pivot rule
becomes cheap again and can be re-evaluated. For n ≤ 512, the sizes the
production campaign uses, the plain column layout is already the fastest
by a wide margin. Milestone 2 starts with that layout, the simplest correct
design.

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

* M3: all Clifford gates; `M MX MY MPP R RX RY MR MRX MRY`; `TICK`; `REPEAT`;
  `DETECTOR`, `OBSERVABLE_INCLUDE`, `QUBIT_COORDS`, `SHIFT_COORDS`; noise
  channels `X_ERROR Y_ERROR Z_ERROR DEPOLARIZE1 DEPOLARIZE2 PAULI_CHANNEL_1
  PAULI_CHANNEL_2`, plus measurement flip noise `M(p)`.
* Later, as needed: `MXX MYY MZZ`, `SPP`, `E/ELSE_CORRELATED_ERROR`, heralded
  channels, `MPAD`.

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
force.

## 8. Monitored circuits

`mipt.MonitoredCircuit` describes a protocol: graph (or a bank of graphs, one
per trajectory), schedule (`matching` = brickwork on any graph through an edge
colouring, `random_edges` with optional edge weights for long-range circuits,
`all_edges`), measurement rate, picture (`none`, `purification`, `single`
reference with a fixed or random partner), scramble (random pairs, random
all-to-all perfect matchings, gate-only layers), gate set, checkpoints, probes
and early stopping. `spec.run(trajectories, seed=...)` generates and simulates
every trajectory inside one compiled, parallel kernel and records probes at
the checkpoints; `EnsembleResult` holds the arrays (`survival()`,
`save`/`load`, `concatenate`). The PRL production protocol is
[`examples/purification_time.py`](../examples/purification_time.py).

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

Validation: the engine equals the Simulator on exported circuits (exact), and
on the PRL protocol its purification-time distribution agrees with the v2
production engine on identical graph banks
([`benchmarks/crosscheck_v2_production.py`](../benchmarks/crosscheck_v2_production.py)).

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
   with an install hint; it never silently skips (v2 lost 300 tests that way).
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
8. **v2 as a third reference** for MIPT observables on identical circuits
   (M6).
9. **CI** gets a v3 job with Stim required.

## 11. Milestones

| | milestone | status |
|---|---|---|
| **M1** | Foundations: Pauli strings, Cliffords, the gate library | done |
| **M2** | Kernel + `Simulator`, both modes | done (lock-step fuzzing vs Stim with exact signs and outcomes) |
| **M3** | `Circuit` + compiled `run()` / `sample()` | done (whole-circuit replay identical to Stim when resets act on unentangled qubits, section 6.2; noise statistics match) |
| **M4** | Observables incl. erasure protection | done (state-vector and brute-force checks) |
| **M5** | Performance | benchmarks published (`benchmarks/`, README); row cache planned (section 5.3) |
| **M6** | Monitored circuits | done: ensemble engine, graph banks, pictures, scrambles, probes, survival analysis |
| **M7** | QEC | circuits run and match Stim's detector statistics; a frame sampler is not planned |
| **M8** | Studies + docs | in progress: examples, agent interface (section 13), migration notes |

## 12. Decisions

Your answers to the milestone-1 questions, and what they mean in the code.

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
6. **Production campaign:** `studies/prl_production` (single-reference
   purification time on Watts-Strogatz graphs, 4.27M of 18.7M trajectories
   done) can continue on v2, or switch to v3 with
   `examples/purification_time.py`, which reproduces its statistics.
7. **Python:** 3.11+.

## 13. Interfaces for AI agents

The simulator is meant to be driven by language-model agents as easily as by
people. One service layer, `sparsegf2_v3.api`, takes and returns JSON
(validated arguments, compact summaries with standard errors, errors that say
how to fix the input), and two thin front ends expose it: the `sparsegf2` CLI
(`--json` on every command) and the `sparsegf2-mcp` Model Context Protocol
server (typed tools, background jobs for long ensembles with resumable chunk
files). `AGENTS.md` and `llms.txt` describe the workflow for agents;
[`LLM_INTERFACE.md`](LLM_INTERFACE.md) is the reference. The repository's
`.mcp.json` registers the server for Claude Code.

## 14. Roadmap

From the audit, the literature review ([`LITERATURE.md`](LITERATURE.md)) and
the benchmarks, in rough priority order:

* **Row cache** for the area-law phase at large n (section 5.3).
* **Erasure and noise channels in the engine**: append environment rows for
  erasure / dephasing, information retention I(R:Q) and coherent information
  under noise (the "first time I(R:sys) = 0" replaces tau, which noise does
  not define).
* **Expurgation** (stabilizer/destabilizer pairs in phase-free mode).
* **Measurement schedules**: exactly floor(p n) measurements per layer,
  measurement right after each gate, space-time modulated rates.
* **Graph disorder redrawn every layer** (annealed disorder); per-trajectory
  disorder (graph banks) is done.
* **Multi-reference correlators** and participation entropy.
* **Uniform n-qubit Clifford sampler** for global scramblers and random-code
  baselines.
* **Tableau snapshots at checkpoints** (the code's stabilizers, from the
  reference rows) for offline analysis.
