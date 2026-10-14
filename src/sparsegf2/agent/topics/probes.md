# Probes

Probes are recorded inside the engine at the checkpoints, or at the probe's own `@checkpoints`, so a cheap probe can run every layer and an expensive one at a few depths. `help_list(what="probes")` has exact signatures.

- **Entanglement:** `half_cut_entropy`, `entropy`, `mutual_information`, `tripartite_information`, and the vector `entropy_profile` (S of the first m qubits of `system_order`, m = 0..n).
- **Code** (need a reference): `code_dimension` (k = S(R)), `contiguous_distance`, `lost_information` (one fixed region), and the vector `erasure_lost_profile` (mean bits lost by a random m-qubit erasure, m = 0..n).
- **Erasure** (need a reference): `erasure_first_failure(samples)` records, for each of `samples` random orderings, the first-failure size D: the smallest m such that erasing the first m qubits loses logical information (n + 1 when k = 0). `uncorrectable_region_counts(max_size)` counts uncorrectable m-qubit erasures for m = 1..w by exhaustion, so count / C(n, m) is the exact P_C(m); all counts are 0 exactly when the distance exceeds w.
- **`code_snapshot`:** saves the stabilizer generators; its value is their number, n - k. At most one per spec; give it its own `@final`.

Costs per evaluation at n = 256, one thread: `erasure_first_failure` with M = 100 samples takes 3-7 ms; `uncorrectable_region_counts` takes about 1 ms for w = 2, 40-50 ms for w = 3 and about 7 s for w = 4 (practical only at a few depths). `erasure_first_failure` stores `samples` numbers per evaluation and `code_snapshot` a whole code, so give them few checkpoints.

With `stop_when_purified`, code probes record exact values for a pure reference after a trajectory stops: `code_dimension`, `lost_information`, `erasure_lost_profile` and the counts are 0, `erasure_first_failure` is n + 1, and `code_snapshot` is n with no code saved. Entanglement probes are NaN there and average only trajectories still running.
