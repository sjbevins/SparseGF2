# Changelog

All notable changes to the SparseGF2 package.

## [0.2.1] - 2026-04-22

### Fixed

- `sparsegf2/__init__.py` was missing, so the package installed as a
  namespace module with zero exports. Every `from sparsegf2 import ...`
  across the codebase failed. Added the top-level init with canonical
  exports (`SparseGF2`, `StabilizerTableau`, `warmup`,
  `symplectic_from_stim_tableau`, `__version__`).
- `SparseGF2.apply_s` and `SparseGF2.apply_sqrt_x` had their symplectic
  matrices swapped under the row-vector convention, so `apply_s` was
  implementing sqrt(X) and vice versa. Swapped `_S_SYMP` and
  `_SQRT_X_SYMP`, corrected the specialized `_apply_s_kernel` to
  implement the true S action (`x' = x`, `z' = x ^ z`, per Aaronson-
  Gottesman 2004 Table I), updated the batch-kernel dispatch pattern,
  and fixed the `apply_measurement_y` decomposition (now `S; H; M; H;
  S` rather than `H; S; M; S; H`). Removed the "Known issue" disclaimer.
- `analysis_pipeline/analyses/logical_weights.py` always returned the
  sentinel `n+1` because its two boolean conditions were mutually
  exclusive on a pure 2n-qubit system+reference state (commutation
  with all 2n stabilizers => element is itself a stabilizer, for any
  pure rank-2n stabilizer group). Corrected by swapping the conditions
  to test (in-row-span) AND (not-commutes-with-all), which is the
  proper definition of a logical of the emergent code.
- `CircuitConfig.expected_gate_to_meas_ratio()` now returns `1/(2p)`,
  matching what `runner.py` actually stores.
- `RunWriter.begin_run` writes one `graph_n{n:04d}.g6` per system size
  instead of a single `graph.g6` encoding only `sizes[0]`.
- `StabilizerTableau.apply_clifford_2q(track_inverse=True)` now computes
  the GF(2) inverse of the symplectic matrix for the inverse tableau
  update instead of reusing the forward matrix. The previous code only
  worked for self-inverse gates (CNOT, CZ, SWAP).
- `ValidationReport.format()` label inversion: passing sizes now print
  `[ok]` unconditionally.

### Removed

- `core/tableau_ref.py` (dead-code reference implementation, zero imports).
- `core/sparse_tableau_legacy.py` (deprecated; shim only called by a
  sanity check).
- Six 2-byte `fred` placeholder files from `sparsegf2/`, `tests/`,
  `tests/analysis/`, `examples/`, `benchmarks/`, `sanity_checks/`.
- Dead "Known issue" disclaimer on `apply_measurement_y`.

### Added

- Literature citations in every physics-relevant docstring:
  - Aaronson & Gottesman, Phys. Rev. A 70, 052328 (2004),
    arXiv:quant-ph/0406196 (stabilizer simulation, gate updates,
    measurement).
  - Fattal, Cubitt, Yamamoto, Bravyi, Chuang 2004,
    arXiv:quant-ph/0406168 (subsystem-entropy rank formula).
  - Gullans & Huse, Phys. Rev. X 10, 041020 (2020),
    arXiv:1905.05195 (purification picture, code rate = S(R)).
  - Skinner, Ruhman, Nahum, Phys. Rev. X 9, 031009 (2019),
    arXiv:1808.05953 (MIPT).
  - Hosur, Qi, Roberts, Yoshida, JHEP 2016:4 (2016),
    arXiv:1511.04021, Eq. 24 (tripartite mutual information).
  - Wilson 1927, JASA 22, 209-212 (Wilson score interval).
  - Brown, Cai, DasGupta 2001, Stat. Sci. 16, 101-133.
  - McKay graph6 spec.
  - Anderson 2001 for 1-factorization of K_n.
  - Vizing 1964 for the chromatic-index theorem.
- `tests/test_single_qubit_clifford_truth.py`: ground-truth Pauli-
  action tests for H, S, sqrt(X) that would have caught the S/sqrt(X)
  label-swap bug.
- `tests/test_logical_weights_correctness.py`: Bell-pair ground-truth
  tests and an end-to-end integration test for logical_weights.
- `.github/workflows/tests.yml`: pytest matrix CI on Python 3.11-3.13.
- `CHANGELOG.md` (this file).
- `.gitignore` covering build artefacts, Numba caches, editor junk.

## [0.2.0] - 2026-04-16

Initial public release (untracked — see git history for the content).
