# Sanity Checks: SparseGF2 vs Stim equivalence

This directory contains standalone verification scripts that prove the dense
reference simulator (`StabilizerTableau`, in `sparsegf2/core/tableau.py`)
produces the **exact same stabilizer group** as Google's Stim simulator for
arbitrary Clifford circuits.

> **Scope**: These scripts test `StabilizerTableau` (the scan-all-rows dense
> reference). RREF-level parity of the primary `SparseGF2` sparse simulator
> against Stim is covered by the pytest suite
> (`tests/test_stim_rref_verification.py`, 150+ parity tests).

## What is verified

Two stabilizer simulators produce the same stabilizer group if and only if their
symplectic tableau representations span the same GF(2) subspace. We check this by
computing the **reduced row echelon form (RREF)** of both tableaux over GF(2) and
verifying they are identical.

This is the physically meaningful equivalence: two sets of stabilizer generators
define the same stabilizer state iff they span the same subspace, regardless of
how individual generators are written.

## Test suite

`verify_vs_stim.py` contains 10 independent test classes:

| Test | Description | Scale |
|------|-------------|-------|
| 1. Single-qubit gates | H, S gates only on n=1,2,4,8 | 80 circuits |
| 2. Two-qubit gates | CNOT, CZ, H, S on n=2,4,8,16 | 80 circuits |
| 3. With measurements | Gates + Z-measurement/reset on n=4,8,16 | 60 circuits |
| 4. Purification picture | Bell pairs + system gates + measurements on n_sys=4,8,16 | 45 circuits |
| 5. All 11,520 Cliffords | Every 2-qubit Clifford gate individually | 11,520 gates |
| 6. Deep random circuits | 20n operations on n=4,8,16,32 | 20 circuits |
| 7. Named gate consistency | CNOT/CZ via `apply_clifford_2q` matches dedicated methods | 100 checks |
| 8. iSWAP vs Stim | iSWAP gate on n=2,4,8,16 | 80 circuits |
| 9. Input validation | Error handling, edge cases (n=0, n=1, out-of-range, duplicates) | 30+ checks |
| 10. Deep circuits + iSWAP | 15n ops including iSWAP on n=4,8,16 | 30 circuits |

**All tests passing.**

## How to run

```bash
cd SparseGF2
python sanity_checks/verify_vs_stim.py
```

Or with pytest:

```bash
pip install pytest stim
python -m pytest sanity_checks/verify_vs_stim.py -v
```

## Expected output

```
============================================================
SparseGF2 vs Stim: Stabilizer Group Equivalence Checks
============================================================

  [PASS] Test 1: Single-qubit gates (H, S)
  [PASS] Test 2: Two-qubit gates (CNOT, CZ, H, S)
  [PASS] Test 3: Circuits with measurements
  [PASS] Test 4: Purification picture (Bell pairs + system gates + measurements)
  [PASS] Test 5: All 11520 two-qubit Cliffords verified
  [PASS] Test 6: Deep random circuits (n=4,8,16,32)
  [PASS] Test 7: apply_clifford_2q matches named gates (CNOT, CZ)
  [PASS] Test 8: iSWAP vs Stim
  [PASS] Test 9: Input validation
  [PASS] Test 10: Deep circuits with iSWAP

============================================================
ALL CHECKS PASSED
============================================================
```

## How the comparison works

1. **Build identical circuits** for both simulators using the same RNG seed
2. **Run through SparseGF2** `StabilizerTableau` (our simulator)
3. **Run through Stim** `TableauSimulator` (the reference)
4. **Extract symplectic matrices** from both:
   - SparseGF2: `tab.to_symplectic()` returns `[X | Z]` shape `(n, 2n)`
   - Stim: `extract_stim_symplectic()` reads `tab.z_output(r)` for each generator
5. **Compute RREF** of both matrices over GF(2)
6. **Compare**: identical RREF = identical stabilizer group

## Symplectic representation

Each stabilizer generator is a Pauli string. We encode it as a row in the
symplectic matrix `[X | Z]` where:

```
x[r, q] = 1  if generator r has X or Y on qubit q
z[r, q] = 1  if generator r has Y or Z on qubit q

Pauli encoding:  I = (0,0),  X = (1,0),  Y = (1,1),  Z = (0,1)
```

A 2-qubit Clifford gate is described by a 4x4 GF(2) matrix `S` that maps the
input `(x_q0, x_q1, z_q0, z_q1)` to the output. This matrix is extracted from
Stim's `Tableau` object and applied to every generator row.
