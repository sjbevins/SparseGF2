# sparsegf2.expurgation

Expurgation of low-depth random-circuit codes, run natively on the SparseGF2
phase-free tableau. The algorithm is from Gullans, Krastanov, Huse, Jiang, and
Flammia, *Quantum Coding with Low-Depth Random Circuits*, PRX **11**, 031066
(2021), Sec. VI ([arXiv:2010.09775](https://arxiv.org/abs/2010.09775)).

## What it does

A random Clifford circuit applied to an all-zeros state with `k` data inputs
defines an `[[n, k]]` code. Some erasure patterns defeat that code: they support
error operators with zero syndrome and a nontrivial action on the logical
qubits, which no decoder can undo. Expurgation finds those operators and removes
them by projectively measuring them. Each measurement spends exactly one logical
qubit, and the code distance and the erasure recovery probability never decrease
(the paper's Propositions 1 and 2). For codes in two or more dimensions and in
all-to-all geometries, trading a little rate this way buys large recovery
improvements; in one dimension the loop runs the code down to `k = 0`, exactly
as the paper reports.

Every quantity the algorithm consumes is a function of the phase-free symplectic
tableau, so no signed simulator is needed at any step. The measurement outcome
sign, the one thing SparseGF2 does not track, is irrelevant: the paper itself
takes it to be random.

## Quick start

```python
import numpy as np
from sparsegf2.expurgation import ExpurgationConfig, expurgate, random_encoding

# depth-4 all-to-all random encoding of an [[64, 32]] code
code = random_encoding(64, range(32, 64), 4, geometry="all_to_all", rng=np.random.default_rng(0))

result = expurgate(
    code,
    ExpurgationConfig(
        erasure_count=8,  # each round erases 8 random sites
        k_target=16,  # stop at rate 1/4
        validation_patterns=60,  # frozen patterns for the recovery metric
        seed=1,
    ),
)
print(result.k_initial, "->", result.k_final, f"({result.stop_reason})")
print("mean recovery:", result.recovery_before, "->", result.recovery_after)
```

`result` also carries the per-round `k_trajectory` and the weight, pair index,
and round of every measured operator.

## How it works

One round of the loop:

1. Sample an erasure pattern from the configured model (`erasure_rate` or
   `erasure_count`, optionally restricted to a `sites` pool).
2. Assemble the uncorrectable-error matrix `M`: one row per local error
   (`Z_i`, then `X_i`) on each erased site, columns the commutation bits with
   the checks and then with the logical generators. Each entry is a single
   commutation query on the tableau, so assembly is a pass over the erased
   sites' inverted indices.
3. Row-reduce the augmented matrix `[M | I]` on the syndrome columns, then the
   logical columns. The logical-block pivot rows are the uncorrectable
   directions, `r_M` of them, and their identity-block witness bits reconstruct
   the offending Pauli operators, returned lightest first.
4. Re-validate each candidate against the current tableau (earlier measurements
   in the round may have made it detectable or trivial) and measure the
   survivors with `SparseGF2.measure_pauli`. The destabilizer/stabilizer pair
   that absorbs the operator leaves the logical set. Under the default `gauge`
   strategy it becomes a gauge pair and the original checks stay untouched;
   under the `stabilizer` strategy it becomes a new check.
5. Stop when `k` reaches `k_target`, the validation metric reaches
   `recovery_target`, `max_barren_rounds` consecutive rounds produce no valid
   candidate, `max_rounds` is hit, or `k` reaches zero (expurgation failed).

The exact optimal-decoding recovery probability for a pattern is `2**-r_M`, so
the validation metric costs two GF(2) ranks per pattern. Validation patterns
are frozen at the start, which makes the before and after numbers comparable
and the target check monotone.

## Works with any tableau

The package does not care how the tableau was produced. `StabilizerCode`
couples to a `SparseGF2` instance plus a role array over its
destabilizer/stabilizer pairs (check, logical, or gauge), and everything else
is computed through the simulator's public measurement and commutation API.
Four common entry points:

```python
from sparsegf2 import SparseGF2
from sparsegf2.expurgation import StabilizerCode, from_purification

# 1. An encoding circuit built from core gates: pairs on the data qubits
#    are logical, the rest are checks.
code = StabilizerCode.from_encoding(sim, data_qubits=[2, 3])

# 2. A monitored circuit in the purification picture (system + reference
#    qubits): extract the system code. Checks are the stabilizers
#    supported entirely on the system; k = S(system).
code = from_purification(sim)  # first half = system
code = from_purification(sim, system_qubits=range(n_sys))  # explicit

# 3. An externally produced tableau (any simulator, a file, a paper):
code = StabilizerCode.from_encoding(SparseGF2.from_symplectic(mat), [2, 3])

# 4. Full control, including gauge pairs:
code = StabilizerCode(sim, roles)
```

For a purification input, the extracted code lives on the system qubits (code
qubit `i` is `sorted(system_qubits)[i]`), and expurgation runs on the
extracted code. Measuring the same candidates on the original purification
tableau with `measure_pauli` (indices mapped through that list) is the
identical operation and lowers `code_dimension` by one per candidate, so the
two pictures stay in sync if you need both. The circuits and analysis layers
are never imported.

## Module map

| module | contents |
|---|---|
| `roles.py` | `StabilizerCode`, the role constants, and the measure-and-relabel move |
| `erasure.py` | `sample_erasure`, `uncorrectable_matrix`, `uncorrectable_rank`, `recovery_probability`, `expurgation_candidates` |
| `driver.py` | `ExpurgationConfig`, `expurgate`, `mean_recovery`, `ExpurgationResult` |
| `encoding.py` | `random_encoding` (brickwork or all-to-all) |
| `purification.py` | `from_purification` (system + reference tableau to code view) |

The two core primitives the package rides on live with the simulator:
`SparseGF2.measure_pauli` / `SparseGF2.pauli_anticommuting_rows` in
`sparsegf2.core.sparse_tableau`, and `gf2_eliminate_on_columns` in
`sparsegf2.core.linalg_gf2`.

## Testing

`tests/expurgation/` traces the two worked examples of the companion
implementation notes end to end: a 4-qubit chain where the uncorrectable-error
matrix, the extracted candidate, and the recovery improvement are checked
verbatim, and a depth-2 encoding where one measurement provably raises the
brute-force code distance from 1 to 2. Random codes cross-check `M` against
dense symplectic products and pin the loop's accounting and monotonicity.
`tests/test_measure_pauli.py` checks the new kernel against `measure_z` and
against Stim's `MPP` instruction (skipped when Stim is not installed).

## References

- Gullans, Krastanov, Huse, Jiang, Flammia,
  [PRX 11, 031066 (2021)](https://arxiv.org/abs/2010.09775): the algorithm
  (Sec. VI) and the erasure-decoding formalism (Eqs. 16-18).
- Aaronson, Gottesman,
  [PRA 70, 052328 (2004)](https://arxiv.org/abs/quant-ph/0406196): the
  measurement update `measure_pauli` generalizes.
- Poulin, [PRL 95, 230504 (2005)](https://arxiv.org/abs/quant-ph/0508131):
  stabilizer subsystem codes (the gauge strategy).
