# SparseGF2 — extended documentation

This file complements `README.md` and `GETTING_STARTED.md` with reference
material for individual features. It is generated/updated as new
capabilities are added.

## The Single-Qubit Probe Protocol (`single_ref`)

### Overview

The single-qubit probe is a minimal-ancilla MIPT diagnostic: run a
graph-defined random-Clifford + measurement circuit on ``n`` system
qubits with **one** reference qubit attached (not ``n`` as in the full
purification picture), and watch the entanglement between the reference
and the system. In the volume-law phase the ancilla stays entangled
with the scrambled system and ``S(ref) = 1``; in the area-law phase
measurements destroy the correlation and ``S(ref) = 0``. Curves of
``<S(ref)>`` vs ``p`` for different ``n`` cross at the critical rate
``p_c``, and the time at which ``<S(t)>`` drops to ``0.5`` is the
characteristic purification time ``tau``.

### n+1 qubit layout

`init_picture("single_ref", n)` (in `sparsegf2.circuits.pictures`)
returns a `StabilizerTableau` of exactly ``n + 1`` physical qubits,
prepared as follows:

1. allocate ``n + 1`` qubits in ``|0...0>`` (`from_zero_state(n+1)`);
2. apply `H(0)`;
3. apply `CNOT(0 -> n)`.

The resulting state is
``|Phi+>_{0,n} (x) |0>_1 (x) |0>_2 (x) ... (x) |0>_{n-1}``:
a Bell pair on qubits ``(0, n)`` and a product of ``|0>`` on qubits
``1, 2, ..., n-1``. The reference qubit is always qubit index ``n``.

During the circuit execution the runner guarantees:

- every 2-qubit Clifford acts on a pair ``(qi, qj)`` with both indices
  strictly ``< n`` (i.e. inside the system block);
- every Z-basis measurement acts on an index ``< n``;
- qubit ``n`` is never addressed.

### `obs.k` interpretation

When ``picture == "single_ref"`` the ``obs.k`` column of
``samples.parquet`` contains ``S(qubit n)``, computed via the Fattal-
Cubitt-Yamamoto-Bravyi-Chuang rank formula
``S(A) = rank(M|_A) - |A|`` (arXiv:quant-ph/0406168 Thm. 1) applied to
the single-qubit subset ``A = {n}``. Because a single qubit in a
stabilizer state has integer von Neumann entropy,
``obs.k`` is always exactly ``0`` or ``1``. Do **not** interpret it as
the emergent-code rate (which is the purification-picture meaning of
``k``); for `single_ref` it is a single-bit survival indicator.

The companion column ``obs.p_k_gt_0`` equals ``1`` iff ``obs.k > 0``,
and in the single_ref picture is exactly the per-sample survival
indicator.

### Time-series recording (`record_time_series`)

`CircuitConfig.record_time_series` (bool, default False; only valid for
``single_ref``) turns on per-layer tracking of ``S(qubit n)``. When it
is ``True`` the runner records ``S(qubit n)`` at

- ``t = 0``: the initial Bell-pair state (value is always 1),
- ``t = 1, 2, ..., T``: after each of the ``T = total_layers`` circuit
  layers.

Each sample carries a ``uint8`` array of shape ``(T + 1,)`` through
`SampleRecord.ref_entropy_timeseries`. The writer collects these across
samples and persists them per cell as ``timeseries.h5``:

```
runs/<run_id>/data/n=XXXX/p=YYYY/timeseries.h5
  attrs: schema_version, encoding_version, n, n_samples, total_layers
  S_of_t       uint8  [n_samples, total_layers + 1]
  sample_seed  int64  [n_samples]
  t_axis       int32  [total_layers + 1]
```

### Analysis pipeline

`sparsegf2.analysis.analyze_single_ref(run_dir)` auto-detects the picture
from ``manifest.json`` and runs:

1. **Crossing analysis** (always). Reads every ``samples.parquet``
   under ``run_dir``, aggregates ``obs.k`` by ``(n, p)`` (mean, std,
   SEM, count via polars), and writes ``crossing_plot.png``.
2. **Purification-time analysis** (when any ``timeseries.h5`` is
   present). Reads the per-layer traces, computes
   ``P(t) = <S(t)>_{samples}``, and writes:
   - ``purification_decay.png``: ``P(t)`` vs ``t`` for every
     ``(n, p)`` cell, with a dashed line at ``P = 0.5``;
   - ``tau_scaling.png``: the characteristic time
     ``tau = min{ t : P(t) <= 0.5 }`` (with linear interpolation
     between adjacent time points) plotted against ``n``, one curve
     per ``p`` value.

Direct building blocks are exported from `sparsegf2.analysis` with a
single_ref prefix:

```python
from sparsegf2.analysis import (
    detect_picture,
    has_timeseries,
    load_single_ref_samples,
    aggregate_single_ref_entropy,
    plot_single_ref_crossing,
    load_single_ref_timeseries,
    compute_single_ref_tau,
    plot_single_ref_purification_decay,
    plot_single_ref_tau_scaling,
    analyze_single_ref,
)
```

### Example end-to-end invocation

```bash
# Sweep both endpoint and per-layer data:
python -m sparsegf2.circuits \
    --graph complete --matching-mode round_robin \
    --picture single_ref --record-time-series \
    --sizes 8 16 32 --p-min 0.0 --p-max 0.5 --n-p 21 \
    --samples 256 --workers 4 \
    --output runs/ --run-id allo_single_ref

# Auto-detecting analysis:
python -c "from sparsegf2.analysis import analyze_single_ref; \
analyze_single_ref('runs/allo_single_ref')"
# -> runs/allo_single_ref/crossing_plot.png
# -> runs/allo_single_ref/purification_decay.png
# -> runs/allo_single_ref/tau_scaling.png
```

### Validation

`tests/test_single_ref.py` is an integrated QA agent that sweeps
``n in {4, 6, 8}`` on the all-to-all (complete) graph at
``p in {0.0, 1.0}`` and asserts:

- the reference qubit at index ``n`` is never involved in any gate or
  measurement (verified by walking the `CircuitBuilder` schedule);
- ``obs.k`` is integer 0 or 1 in every sample;
- ``obs.k == 1`` at ``p = 0.0`` (no measurements, Bell pair preserved)
  and ``obs.k == 0`` at ``p = 1.0`` (full measurement destroys the
  correlation);
- with ``record_time_series=True``, the per-sample trace has the
  expected length ``total_layers + 1``, its first entry is 1, and its
  last entry matches the endpoint ``k`` field.

### The `single_ref` physics picture (v0.2.1+)

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
