# Reverse log: walking a trajectory backward from its final tableau

Status: **planned, deferred**, 2026-10-13. To be implemented once the
simulator is complete and fully tested. Depth probing today uses replay
([`2026-10-13-agent-tool-stacks-design.md`](2026-10-13-agent-tool-stacks-design.md),
section 6.4).

## 1. Idea

Keep the final tableau of a trajectory and a compact log of what the circuit
did. Undoing the logged operations in reverse order recovers the tableau,
and so the code, at any earlier depth. Nothing has to be simulated forward
again, and no snapshot is needed at every interesting depth.

## 2. Why it works

The state is a phase-free stabilizer tableau, stored column-major: rows are
generators, and each qubit has an X column and a Z column of bitsets over
the rows.

* **Gates are invertible.** A two-qubit Clifford g on (a, b) is undone by
  applying g^-1 on (a, b). `CLIFFORD_2Q(i)` has an inverse index, and
  single-qubit gates likewise. Nothing needs storing except the gate itself,
  and the gate can be regenerated from the seed.
* **A deterministic measurement changes nothing** in phase-free mode, since
  no stabilizer anticommutes with the measured Pauli. Nothing needs storing.
* **A random measurement is not invertible from the result alone**, but it
  becomes invertible with a little extra data. The forward update
  (`_kernel/tableau.py`: `_collapse_phase_free`, and
  `measure_1q_phase_free_tracked` for reference rows) works like this:
  1. A = the rows that anticommute with the measured Pauli M on qubit q;
  2. the pivot p is the first row of A outside the reference rows, if there
     is one, otherwise the first row of A;
  3. every r in A \ {p} becomes r · P, where P is the old pivot row;
  4. row p becomes M. If p was a reference row, its reference-row bit is
     cleared.

  The undo, given (p, A, P, was_reference):
  1. for every r in A \ {p}: r ← r · P (since (r · P) · P = r);
  2. row p ← P;
  3. if was_reference, set p's reference-row bit again.

  Both directions cost the same: one XOR of A into each of P's support
  columns. The undo needs no anticommutation scan.

The reconstruction is bit-exact: the recovered tableau equals the one the
forward run had at that depth, generator by generator, not just as a group.

## 3. What is logged

Per trajectory:

| stream | per entry | needed for |
|---|---|---|
| final tableau | 2N columns × ceil(rows/64) words (N qubits incl. references) | the starting point |
| gates | layer offset; (a, b, Clifford index) | walking (or regenerate from the seed) |
| measurements | layer offset; qubit; random flag | walking (or regenerate from the seed) |
| undo records | for each *random* measurement: p (int32), A (ceil(rows/64) words), P (2N bits, dense) | walking backward |

Size at n = 256 in the purification picture (N = 512 qubits, 512 rows): the
undo record is 4 + 64 + 128 bytes ≈ 200 B. With p = 0.1 and T = 256 there
are at most p·n·T ≈ 6,600 random measurements, so about 1.3 MB per
trajectory. The final tableau is 64 KB. Gates and measurements are about
0.4 MB if stored, and nothing if regenerated from the seed.

For comparison, a snapshot of the code at every layer costs 256 × 64 KB =
16 MB, and the replay record is (protocol, seed, index), essentially nothing.

## 4. Algorithms

* **Backward walk** from the final layer T down to the requested depths,
  evaluating probes at each one (the same probe kernels as the engine).
  It costs T − d layers of undo work to reach depth d.
* **Bidirectional with sparse snapshots** (as in checkpointing schemes such
  as `revolve`). Keep the tableau every K layers. A query at depth d starts
  from the nearest snapshot and walks forward (gates plus measurements; no
  undo data needed) or backward (undo data). That bounds any query to K/2
  layers. With snapshots, the undo stream is only needed for backward walks.

## 5. When it beats replay, and when it does not

Replay re-simulates the trajectory forward from layer 0. Measured at
n = 256, T = 256, p = 0.1 on one thread, the whole trajectory takes 4.5 ms,
while one first-failure evaluation with M = 512 takes 5.2 ms. Reaching depth
d costs d layers forward or T − d layers backward. To visit many depths you
cross the whole circuit either way, so the reverse log saves at most about
2× of the (cheap) simulation, at the price of the storage above.

It is worth building when one of these holds:

1. **A self-contained, version-independent archive.** The final tableau plus
   the log reproduces the whole history without the engine's random
   streams. Replay breaks if the randomness contract changes.
2. **Circuits that are not seed-generated:** circuits supplied from outside,
   adaptive circuits, or noise channels during the circuit (on the roadmap).
3. **Simulation becomes the bottleneck:** very large n deep in the area-law
   phase, or long circuits probed only near their end.
4. **Interactive exploration of one long trajectory**, stepping back and
   forth layer by layer (e.g. in a viewer).

## 6. Proposed API

```python
rec = spec.record(trajectory=3, seed=7, snapshots_every=64)  # TrajectoryRecord
rec.final_code()                   # StabilizerCode at layer T
rec.code_at(128)                   # StabilizerCode at layer 128 (walks from the nearest snapshot)
rec.evaluate([mipt.erasure_first_failure(512)], layers=[0, 64, 128, 256])  # probe values vs depth
rec.save("traj3.npz"); TrajectoryRecord.load("traj3.npz")
```

* Kernel: a recording variant of the phase-free measurement that copies P
  and A before modifying them (`measure_1q_phase_free_tracked_logged`),
  plus `undo_measurement` and `apply_2q_inverse`. The ensemble kernel gains
  an optional per-trajectory record buffer (worst case T·n undo slots,
  preallocated for single trajectories; for ensembles, only for a selected
  list of trajectories).
* File format: `.npz` with `final` (packed tableau), `gate_ptr`, `ga`, `gb`,
  `gg`, `meas_ptr`, `mq`, `random`, `undo_pivot`, `undo_anti`,
  `undo_row`, optional `snapshot_layers` and `snapshots`, and JSON metadata
  (protocol, seed, trajectory, package version, randomness contract).
* Agent interface: `traj_run(record="reverse_log")` and `traj_probe` reading
  it, with no new tools.

## 7. Acceptance criteria

* For random trajectories (n = 8 to 128, every picture and schedule, p from
  0 to 1), walking backward from the final tableau reproduces the forward
  run's tableau **bit for bit** at every layer, generators included, and so
  matches `code_snapshot` at every layer.
* `rec.evaluate` equals replay's probe values exactly.
* A record saved by one version loads and walks in a later one.
* Size and speed are measured and documented against replay and snapshots
  at n = 64 to 1024.

## 8. Risks

* The engine's measurement update may grow extra row operations (gauge
  fixing, a row cache for the area-law phase; DESIGN.md §5.3). Each needs
  its own undo record, so the logged variant must be kept in lock-step,
  pinned by the bit-exact acceptance test.
* The worst-case preallocation of T·n undo slots is large for big ensembles.
  Recording is meant for selected trajectories, not whole campaigns.
