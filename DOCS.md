# SparseGF2 — extended documentation

This file complements `README.md` and `GETTING_STARTED.md` with reference
material for individual features. It is generated/updated as new
capabilities are added.

## The `single_ref` physics picture (v0.2.1+)

`sparsegf2.circuits` supports two physics pictures for MIPT sweeps:
`purification` (the historical default) and `single_ref` (new in
v0.2.1). Both are selected via `CircuitConfig.picture`.

### What it is

The `single_ref` picture runs a graph-defined random-Clifford +
measurement circuit on `n` system qubits while using **one** ancilla as
a reference probe, instead of the `n` ancillas used in the full
purification picture.

The initialization, produced by
`sparsegf2.circuits.pictures.init_picture("single_ref", n)`, is:

1. Allocate `n + 1` qubits in `|0...0>` (via
   `StabilizerTableau.from_zero_state(n + 1)`).
2. Apply `H` to qubit 0.
3. Apply `CNOT(0 -> n)`.

The resulting state is a product of a Bell pair `|Phi+>` on qubits
`(0, n)` and `|0>` on qubits `1..n-1`. The reference qubit is qubit
index `n`.

During the circuit, every gate and every measurement acts only on qubits
`0..n-1`. Qubit `n` is never addressed. At the end of the circuit, the
MIPT probe observable is

```
S_ref = S(qubit n) = entropy of the reduced density matrix on qubit n,
```

stored in the `obs.k` column of `samples.parquet`. Because a single
qubit in a stabilizer state has integer entropy, `S_ref` is always
exactly `0` or `1`.

### How it differs from `purification`

|                             | `purification`                     | `single_ref`                         |
|-----------------------------|------------------------------------|--------------------------------------|
| Total qubits                | `2n`                               | `n + 1`                              |
| Initial state               | `n` Bell pairs, one per system qubit | one Bell pair on `(0, n)`; rest in `|0>` |
| System qubits               | `0..n-1`                           | `0..n-1`                             |
| Reference qubits            | `n..2n-1`                          | `n` (single)                         |
| Simulator returned          | `SparseGF2` (sparse, Numba-JIT)    | `StabilizerTableau` (dense)          |
| `obs.k` interpretation      | emergent-code rate `k = S(R)`,     | `S(qubit n)`, integer 0 or 1:        |
|                             | integer in `[0, n]`                | a single-bit MIPT diagnostic         |
| `obs.bandwidth`, `obs.tmi`, | defined, computed from the         | set to `0` (not meaningful on an     |
| `obs.entropy_half_cut`      | 2n-qubit tableau                   | `n + 1` qubit state)                 |

The purification picture gives the full code structure — rate `k/n`
and distance `d_cont`, `d_min` all follow from `S(R)` and sub-block
entropies. The `single_ref` picture gives a *minimal* diagnostic that
is cheap to simulate for large `n` and is sufficient to locate the
MIPT critical point `p_c`: `<S_ref>` saturates at 1 for `p < p_c`
(volume-law phase, the Bell-pair correlation survives the scrambling),
decays to 0 for `p > p_c`, and the curves for different `n` cross at
`p_c`.

### Schema impact on `samples.parquet`

When `picture == "single_ref"` the writable columns are identical to the
purification layout, but their semantics shift:

| Column               | Purification meaning           | `single_ref` meaning                  |
|----------------------|--------------------------------|---------------------------------------|
| `obs.k`              | `k = S(R)`, in `[0, n]`        | `S(qubit n)`, in `{0, 1}`             |
| `obs.p_k_gt_0`       | 1 iff `k > 0`                  | 1 iff `S(qubit n) > 0`                |
| `obs.bandwidth`      | min `l` with `I(A_l; R) > 0`   | `0` (not meaningful)                  |
| `obs.tmi`            | tripartite MI of system       | `0.0` (not meaningful)                |
| `obs.entropy_half_cut` | `S(A)` on first `n/2` qubits | `S(A)` on first `n/2` qubits (same formula) |

All `diag.*` columns retain their purification meaning.

### Analysis

`sparsegf2.analysis.single_ref` provides a crossing-plot utility:

```python
from sparsegf2.analysis import (
    load_single_ref_samples,
    aggregate_single_ref_entropy,
    plot_single_ref_crossing,
)

df = load_single_ref_samples("runs/my_single_ref_sweep")
agg = aggregate_single_ref_entropy(df)  # group by (n, p) -> mean, SEM, count
fig, ax = plot_single_ref_crossing(
    agg,
    out_path="crossing.png",
    title="MIPT probe on cycle, single_ref",
)
```

The crossing of the `<S>` curves for different `n` locates `p_c`.

### Validation

A standalone QA agent is shipped at `tests/validate_single_ref.py`. It
runs a sweep test across
`{all-to-all, brickwork} x {n=4, 8, 12} x {p=0.0, 1.0}`
and asserts:

- the reference qubit at index `n` is never involved in any gate or
  measurement during the circuit schedule (verified by inspecting the
  `CircuitBuilder` output directly);
- the per-sample reference entropy is strictly an integer in `{0, 1}`;
- at `p == 0.0` the entropy is `1`, at `p == 1.0` the entropy is `0`.

Run it with `python tests/validate_single_ref.py` (exit code 0 on pass)
or under `pytest tests/validate_single_ref.py`.

### Example CLI

```bash
python -m sparsegf2.circuits \
    --graph cycle --matching-mode round_robin \
    --picture single_ref \
    --sizes 8 16 32 --p-min 0.0 --p-max 0.5 --n-p 21 \
    --samples 500 --workers 4 \
    --output runs/ --run-id cycle_single_ref_sweep
```

### References

- Gullans & Huse, "Dynamical Purification Phase Transition Induced by
  Quantum Measurements", Phys. Rev. X 10, 041020 (2020),
  arXiv:1905.05195.
- Skinner, Ruhman, Nahum, "Measurement-Induced Phase Transitions in the
  Dynamics of Entanglement", Phys. Rev. X 9, 031009 (2019),
  arXiv:1808.05953.
- Fattal, Cubitt, Yamamoto, Bravyi, Chuang 2004,
  arXiv:quant-ph/0406168 (the rank formula for stabilizer-state
  subsystem entropy, used to compute `S(qubit n)`).
