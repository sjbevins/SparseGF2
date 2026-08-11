# Single-reference production benchmark protocol

This document defines the performance measurements required before launching a
single-reference purification-time production campaign. Benchmark output is a
diagnostic artifact, not simulation data, and must never share a run fingerprint
or resume directory with production results.

## Unit of work

Let `C` be the Cartesian product of the requested system sizes and graph-parameter
values. If a cell contains `N_graphs` graphs, the measurement grid contains
`N_p` rates, and each graph-rate pair has `N_circuits` trajectories, then

```text
N_trajectories = |C| N_graphs N_p N_circuits.
```

For one trajectory the scramble contains exactly `q_scramble n` two-qubit
Clifford gates. The measured evolution stops after `L <= q_max n` layers. It
contains `floor(n/2) L` two-qubit gates, `n L` Bernoulli measurement trials,
and `L` single-reference entropy checks. The number of measurements is random,
with expectation `p n L`. Thus a useful work report includes counts as well as
wall time; trajectories per second alone can be misleading when the stopping
layer changes with `p`.

## Required benchmark strata

Use the smallest, middle, and largest production sizes. For each size use a
local, crossover, and strongly rewired graph cell, and three measurement
regimes:

- `p=0`, which always reaches the cap and measures worst-case layer throughput;
- a rate near the transition, which represents the scientific workload;
- `p=1`, which exposes initialization, process, and scheduling overhead.

Once the measurement grid is fixed, replace the generic transition rate with
actual grid points. Benchmark the exact production protocol and graph records;
do not substitute the legacy Watts-Strogatz-only engine or fixed scramble/depth
constants.

Each benchmark has one untimed warm-up trajectory before at least three timed
repetitions. Report medians and dispersion, not the best time. Cold process/JIT
startup is reported separately from steady-state throughput.

## Stage timing

The implemented trajectory diagnostic records simulator/Bell initialization,
scramble gates, monitored graph-edge gates, measurements, and the per-layer
`S(R)` test. Per-layer calls to `perf_counter` perturb the hot loop, so stage
timing is a separate diagnostic mode. Production-style throughput is measured
without hot-loop timers. Both modes record the observed stopping layer, gate
count, Bernoulli trial count, realized measurement count, and event/censoring
flag.

The trajectory profile deliberately excludes graph-provider validation, edge-bank
materialization, process startup, transactional checkpoint commits, and final
deterministic NPZ publication. Measure those costs separately and include them
in the production estimate. In particular, report steady compute throughput,
journal-commit latency at the proposed `checkpoint_every`, full-shard
finalization time, and coordinator edge-bank preparation time.

## Process scaling on Windows

Trajectories are CPU-bound. Use a `spawn` process pool with top-level, pickleable
worker functions and an `if __name__ == "__main__"` guard. Do not use a thread
pool for trajectory execution. A worker initializer loads the immutable
two-qubit Clifford lookup table and sets Numba to one thread. Each production
task then loads its sealed cell edge bank read-only.

Before Python starts, the launcher sets each numerical runtime to one thread:

```text
NUMBA_NUM_THREADS=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

Setting these only in a process-pool initializer is too late because NumPy and
Numba are imported while a spawned worker starts. The initializer should still
assert the inherited values and call `numba.set_num_threads(1)` defensively.

Measure strong scaling at `1, 2, 4, 8, 12, 16` workers on the current 16-core,
32-thread workstation. Test logical-core counts only as a diagnostic; production
must not assume that simultaneous multithreading improves this kernel. For worker
count `w`, report

```text
speedup(w) = throughput(w) / throughput(1)
efficiency(w) = speedup(w) / w.
```

Also report aggregate worker CPU time divided by wall time, peak resident memory,
and the distribution of task durations. The production worker count is the
smallest value on the throughput plateau, not automatically the number of logical
processors.

## Benchmark tiles, production tasks, and I/O

The deterministic stored leaf coordinate is

```text
(experiment fingerprint, graph cell, exact p value, graph index, circuit index).
```

Random-stream seeds use the narrower physical coordinate
`(protocol master seed, graph cell, exact p value, graph index, circuit index,
stream role)`. They exclude the experiment name and fingerprint, full `p` grid,
`p_index`, worker number, task order, Python's randomized `hash()`, and chunk
boundaries. Therefore campaign renaming, adding unrelated `p` points, changing
the worker count, or changing benchmark tile size must leave every preexisting
trajectory stream unchanged.

The artifact identity separately pins the exact Python, NumPy, and Numba
versions and the NumPy `PCG64` bit-generator contract. Every benchmark report
must record that same environment contract, and production workers must reject
an environment mismatch before accepting tasks.

The scaling harness groups circuit indices into adaptive `0.5-2 s` tiles to
measure process-pool throughput with bounded Windows spawn and IPC overhead.
Those tiles are benchmark machinery only; they are not the production storage
or scheduling unit.

The implemented production task exclusively owns one complete `(graph cell,
p)` shard. Within that task it advances the canonical
`[graph_index, circuit_index]` rows and commits checkpoints to the shard's local
transactional journal. Different `p` shards from the current graph cell run in
parallel. The coordinator alternates low and high `p`, keeps at most
`max_in_flight` tasks queued, and completes every `p` value before advancing to
the next cell. The fine measurement-rate grid is therefore the primary source
of parallel tasks.

Production-representative benchmarks must use the proposed `N_graphs`,
`N_circuits`, and `p` grid, not only small circuit tiles. Report the number of
runnable `p` shards per cell, worker idle time, per-shard duration distribution,
and the low-`p` tail. The chosen granularity is accepted only if it keeps the
candidate worker count usefully occupied without making interruption or resume
latency excessive. If it does not, revise and revalidate the scheduler before
launch rather than assuming the circuit-tile benchmark predicts production
scaling.

Workers never write the shared graph-registry SQLite database. Each worker has
exclusive access to its own `(cell, p)` checkpoint journal; after all rows are
complete it validates and atomically publishes one deterministic NPZ shard.
The finalized shard stores `tau_p` plus censoring and audit fields, not complete
binary entropy traces. The coordinator alone registers the complete plan and
validated terminal artifacts in `single_ref/raw_tau/catalog.sqlite3`; write-back
into the separate shared graph-registry database remains deferred.

Benchmark compute-only, compute-plus-process-pool, checkpoint-journal, final
NPZ-publication, and coordinator-catalog costs separately. A dry-run must
report estimated raw bytes,

```text
4 |C| N_graphs N_p N_circuits
```

for an `int32` `tau_p` tensor, plus metadata and shard/checksum overhead.

## Acceptance checks

Performance measurements are accepted only after tests show:

- scalar, batched, sparse, and hybrid execution give the same `tau_p` and
  censoring decision;
- entropy is tested after every measured layer and stopping occurs at the first
  zero;
- `p=0` is censored at `q_max n`, while `p=1` exercises the first-layer path;
- changing worker count, production task order, or benchmark circuit-tile shape
  gives byte-identical logical results;
- interrupted journal checkpoints resume to the same deterministic final NPZ as
  an uninterrupted run, and a journal is removed only after validated publish;
- each task uses exactly `floor(n/2)` independently sampled graph edges with
  replacement per measured layer and exactly `q_scramble n` scramble gates;
- the inherited numerical-library thread limits equal one in every worker; and
- benchmark JSON records the protocol/source fingerprint, interpreter and
  dependency versions, CPU topology, thread limits, cases, raw repetitions, and
  summary metrics.

CI tests validate benchmark accounting and output schemas but do not enforce
wall-time thresholds. Performance regressions are evaluated from explicit,
bounded benchmark runs on the same machine and power configuration.

## Bounded benchmark command

The PowerShell launcher installs the thread limits before importing Python and
then invokes the v2 trajectory engine. Its defaults are intentionally small.
For example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  studies\prl_production\benchmark_single_ref.ps1 `
  -N 64 -MeasurementRate 0 -QMax 1 -QScramble 1 `
  -Circuits 16 -Workers 1,2 -Repetitions 3
```

Use `-Output path.json` to retain the raw stage, admission-pilot, cold-start,
trial, and scaling records. An optional `-EdgesNpy path.npy` replaces the
built-in `C(n,2)` edge array. Admission profiles fixed circuit indices at the
requested rate and at `p=0`, where every trajectory is verified to reach
`T_max`; the largest observed time receives a 1.25 safety factor before the
one-worker repetition estimate is compared with the cap. This prevents one
early-purifying trajectory from admitting a long repeat. The first timed
trajectory is the capped `p=0` case, and the driver aborts before any remaining
pilots if that first projection already exceeds the cap. Raise
`-MaxEstimatedSecondsPerRepeat` only for an intentional, reviewed benchmark.

On 2026-08-11, the command above completed in 3.1 seconds on the current
16-physical-core/32-logical-CPU Windows workstation. The steady medians were
275.9 trajectories/s at one worker and 510.3 trajectories/s at two workers, a
1.85x speedup and 92.5% two-worker efficiency. The profiled capped trajectory
executed 64 layers, 2,048 dynamic gates, and 64 scramble gates in 3.82 ms;
dynamic gates used 2.17 ms and the per-layer reference check used 1.01 ms.
Worker cold start and JIT warm-up were about 0.64 seconds and are excluded from
steady throughput. These numbers validate the harness only: they are not a
production cost estimate because the scientific `q` values, measurement grid,
and `N_circuits` have not yet been selected.
