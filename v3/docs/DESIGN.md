# SparseGF2 v3: design

Status: **proposal for review**, written at milestone 1. Everything here can
still change; section 12 lists the decisions I need from you. The audit this
design responds to is in [`AUDIT.md`](AUDIT.md).

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

1. **One circuit language.** Everything that runs is a `Circuit`, written in
   Stim's syntax (plus a few extensions, section 6). Circuit *families* (brickwork
   MIPT, single-reference protocols, surface codes) are ordinary functions that
   *return* circuits. There is no 20-field config object.
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

This is the target API. Milestone 1 (now) delivers the foundations
(`PauliString`, `Clifford`, `gate`); the rest arrives milestone by milestone
(section 11).

```python
import sparsegf2_v3 as sg

# Interactive use, mirroring stim.TableauSimulator method names.
sim = sg.Simulator(3, seed=7)          # phases=True: exact, sign-tracking, Stim-identical
sim.h(0)
sim.cx(0, 1)
sim.cx(1, 2)
sim.measure(0)                         # 0 or 1: a physical outcome
sim.peek_z(2)                          # +1 / -1 if deterministic, 0 if random (like Stim)
sim.stabilizers()                      # canonical list of PauliStrings, with signs
sim.entropy([0])                       # entanglement entropy in bits

# Phase-free mode: entanglement dynamics only; half the rows, so roughly twice as fast.
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

# Interop both ways.
circuit = sg.Circuit.from_stim(stim_circuit)
stim_circuit = circuit.to_stim()

# Monitored circuits come from small, readable generator functions.
from sparsegf2_v3 import graphs, mipt
edges = graphs.watts_strogatz(n=128, k=2, beta=0.1, seed=11)       # plain (m, 2) array
circuit = mipt.random_edge_layers(edges, n=128, layers=1024, p=0.16, seed=12)
sim = sg.Simulator(129, phases=False, seed=13)
sim.run(circuit)
```

## 4. Architecture

```
sparsegf2_v3/
  pauli.py        PauliString: Pauli operators with exact phases                 M1 (done)
  clifford.py     Clifford tableaux; canonical 1q/2q Clifford groups               M1 (done)
  gates.py        the gate library, Stim names and conventions                    M1 (done; M3 adds M/R/noise)
  _kernel/        Numba kernels: tableau layout, gates, measurement, probes        M2
  simulator.py    Simulator, the public state object                               M2
  circuit.py      Circuit: Stim-syntax IR, parse/print, from/to Stim               M3
  _compile.py     Circuit -> flat instruction arrays                               M3
  result.py       RunResult: measurement record, random flags, detectors, probes   M3
  observables.py  entropy, mutual info, tripartite info, code dimension, distance  M4
  graphs.py       geometries as edge arrays (cycle, lattice, WS, NW, complete)     M6
  mipt.py         monitored-circuit generators, pictures, scrambles                M6
```

Layers only depend downward: kernels know nothing about circuits; circuits know
nothing about MIPT; MIPT generators are just functions that build circuits.

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
| `phases=True` (proposed default) | n stabilizers + n destabilizers | yes | the physical outcome | anything involving outcomes, detectors, signs, QEC; exact agreement with Stim |
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

* `CLIFFORD_2Q(i) a b`: the two-qubit Clifford with canonical index `i`
  (`Clifford.from_index(2, i)`, defined in M1). A uniform random gate is a
  uniform `i` in `[0, 11520)`. On export to Stim it is expanded into Stim gates.
* Observation instructions such as "record the entropy of qubit r here" and
  "stop the run when qubit r is pure". These are evaluated inside the compiled
  loop, which the purification-time workload needs, because checking entropy
  from Python after every layer would cost more than the layer itself. The
  exact syntax is a milestone-6 decision.

Internally a circuit is stored as arrays, not one Python object per gate, so
building a 300,000-gate random circuit costs milliseconds.

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
  `sim.run(circuit, outcomes=bits)` forces every random outcome. This one
  mechanism serves post-selection studies, replaying a recorded experiment, and
  exact comparison with Stim (for a noiseless circuit, feed Stim's
  measurement record in and every outcome must come back identical).
* `RunResult.random[k]` says whether measurement k was random. It is available
  in both modes; it is a genuine physical property of the trajectory.
  `RunResult.measurements` needs `phases=True` and otherwise raises.

## 7. Observables (M4)

Functions over a simulator state: `entropy(sim, A)`, `mutual_information`,
`tripartite_information`, `code_dimension`, `contiguous_distance`,
`stabilizer_weights`, plus `sim.stabilizers()` / `sim.tableau()` export. The
column layout stores, for each qubit, its X and Z bitsets over stabilizers, so
the entropy of a region is the GF(2) rank of the 2·|A| bitsets of its qubits. That
calculation is direct, with no transposition. Single-qubit entropy takes a
few word operations.

Every observable will be checked against a state-vector calculation (SVD of
the reduced state) on small systems, which needs no Stim and does not reuse
the rank formula it is testing.

## 8. Monitored circuits (M6)

The production workload, written in v3 terms:

```python
edges = graphs.watts_strogatz(n, k=2, beta=beta, seed=graph_seed)     # graph disorder
circuit = mipt.single_reference(                                     # circuit disorder
    n, edges, p=p, reference_partner="random", scramble_pairs=q * n,
    max_layers=8 * n, stop_when_reference_pure=True, seed=circuit_seed)
result = sg.Simulator(n + 1, phases=False, seed=outcome_seed).run(circuit)
result.stop_layer, result.purified                                   # tau and the censoring flag
```

Requirements carried over from the audit:

* graph disorder, circuit disorder and outcome randomness are separate seeds;
* graphs are plain edge arrays, generated with their own RNG and not networkx's;
* the reference site is configurable, including random per circuit;
* scrambles can be random pairs, brickwork layers, or a global Clifford;
* "layer" always means what you wrote, with no implicit rescaling;
* censoring is explicit in the result.

Validation: on identical circuits v3 must reproduce v2's entropies exactly
(both are exact simulators of the same physics), and the ported production
engine must reproduce existing τ statistics.

## 9. QEC and other Clifford circuits (M7)

Any Stim circuit in the supported subset runs in v3: one trajectory per run,
noise sampled, detectors and observables computed from the record. Example:
`Circuit.from_stim(stim.Circuit.generated("surface_code:rotated_memory_z", ...))`.
Detection-event statistics are tested against Stim's own sampler.

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

Each milestone is one or a few commits on this branch, ends with all tests
passing, and adds a "try it" section to `v3/README.md`. I stop at each
milestone boundary for your review.

| | milestone | you will be able to | acceptance |
|---|---|---|---|
| **M1** | Foundations *(this commit)* | build Pauli strings and Cliffords; look up every Stim gate; enumerate and index the 2-qubit Clifford group | 159 tests vs Stim and textbook matrices |
| **M2** | Kernel + `Simulator` | apply any gate, measure Z/X/Y/Pauli products, reset, peek, export stabilizers, in both modes | lock-step fuzzing vs Stim with exact signs and outcomes |
| **M3** | `Circuit` + compiled `run()` | write or import Stim circuits, run them in one call, get the measurement record, detectors, noise | whole-circuit replay identical to Stim; noise statistics match |
| **M4** | Observables | entropies, mutual/tripartite information, code dimension, distance | state-vector checks; v2 agreement |
| **M5** | Performance | see fair benchmark tables vs Stim and v2 | published, reproducible numbers |
| **M6** | Monitored circuits | graphs, layers, pictures, scrambles, purification time with censoring | v2 equivalence on identical circuits; production engine ported |
| **M7** | QEC | run surface/repetition codes, inspect states mid-circuit | detector statistics match Stim |
| **M8** | Studies + docs | thin resumable sweep runner, migration guide from v2, tutorials | your call |

## 12. Decisions I need from you

My recommendation is listed first in each.

1. **Where the code lives.** *Recommended:* in `v3/` in this repository, with
   import name `sparsegf2_v3`, beside v2, which stays untouched. Both can be
   installed together and cross-checked, and your production campaign keeps
   running on v2. When v3 reaches parity (after M6) it takes over the
   `sparsegf2` name as version 3.0 and v2 is retired. *Alternative:* a separate
   repository.
2. **Default mode.** *Recommended:* `phases=True` by default (safe, exact,
   Stim-identical), with MIPT generators and your scripts passing
   `phases=False` explicitly for speed. *Alternative:* default to
   `phases=False`, since most of your work is entanglement dynamics.
3. **No outcomes without phases.** *Recommended:* in phase-free mode
   `measure()` returns `None` and results only report which measurements were
   random. v2's coin-flip outcomes are not reproduced, even as an option.
4. **Pivot rule.** *Recommended:* lowest-index anticommuting stabilizer (fastest
   in the new layout). This changes generator-weight diagnostics compared with
   v2, though not any physical observable.
5. **Which v2 extras to port.** *Recommended:* port expurgation as an optional
   module after M4; replace `Study`/`sweep` with a thin, censoring-aware runner
   in M8; drop `fss_collapse.py` (unused); use Stim's circuit diagrams through
   `to_stim()` instead of porting the quantikz drawer (it could come back
   later if you rely on it for papers).
6. **Production campaign.** *Recommended:* finish or continue it on v2. Switch
   only after M6 shows v3 reproduces its statistics.
7. **Python version.** *Recommended:* 3.11+.
