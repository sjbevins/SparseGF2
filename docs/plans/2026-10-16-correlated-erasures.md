# Geometry-correlated erasures: implementation plan

**Spec:** [docs/specs/2026-10-16-correlated-erasures-design.md](../specs/2026-10-16-correlated-erasures-design.md)
(the mathematics, the rulings R1-R8, the units, what must hold).

**Goal:** the owner's study "Geometry-correlated erasures in monitored quantum circuits" runnable end to end: sample
erasure regions with probability `∝ exp(-J b_G(A))` at fixed size on any graph, evaluate `P_{C,G,J}(m)` and
`Φ_{C,G,J}(ε)` on monitored-circuit codes (in a run and on saved codes), exactly where enumeration allows, with the
nearest-neighbour brickwork ring as the baseline.

**Execution:** one implementer per task (tests first), one review per task, a push after each task to branch
`correlated-erasures` (pull request against `main`; merged by the owner).

## Global constraints

- Tests first; every check passes before a task ends: `python -m pytest`, `ruff check src tests`,
  `ruff format --check src tests`, `python -m mypy` (Success). Compiled kernels live only in `src/sparsegf2/_kernel/`
  (a test enforces it) and are covered by tests (the weekly JIT-free run requires 98% kernel coverage).
- The randomness contract stays 2: the existing trajectories and the existing probes' draws are unchanged; a new
  sampled probe draws only from its own per-evaluation stream `stream_state(key_probe, mix(first+t, phash, layer))`.
- No version bump (3.0.0.dev12). No new dependency.
- The owner (sjbevins) is the sole author: no `Co-Authored-By` or other trailer, no "Generated with" line.
- Every public function has a docstring that states its mathematics in words a language model can follow (the
  quantity, its formula, its units, its shape), and the docs say the same.

## Tasks

- [ ] **T1. The graph side.** `_kernel/erasure_chain.py` (CSR, boundary, the swap chain with replica exchange) and
  `sparsegf2/erasures.py` (boundary, exact `N_m(b)`, sampling, diagnostics, minimum-boundary sets, the ring's closed
  forms and its exact sampler). Spec §4 items 1-4.
- [ ] **T2. The code side.** Batch failure evaluation and the boundary-resolved enumeration kernels;
  `StabilizerCode.correlated_failure`, `boundary_failure_counts`, `interval_failure_by_size`; the `analysis` functions
  (`exact_correlated_failure`, full curves, `Φ`, `ε_δ`, `binomial_window`). Spec §4 item 5.
- [ ] **T3. The in-run probe** `correlated_erasure` (and, if it fits, `boundary_failure_counts` as a probe): the probe
  table, storage, save and load, replay and its oracle, conditioning on `k > 0`, summaries, capabilities, cost
  estimates. Spec §4 item 6.
- [ ] **T4. The interfaces.** `api.sample_erasure_regions` and `api.erasure_curves` with CLI commands and read-only MCP
  tools; descriptions within the MCP tests' limits.
- [ ] **T5. The documents.** `docs/observables/correlated-erasure.md` in complete mathematical detail (the study
  transcribed and corrected, the Ising form, every proof, the chain and replica exchange with their correctness
  proofs, the diagnostics, the estimators and their errors, exact boundary-resolved counts, the limits `J -> ±∞` on the
  ring, the path and small-world graphs, the literature), AGENTS.md, docs/LLM_INTERFACE.md, the observables index,
  CHANGELOG, `examples/correlated_erasure.py`; worked examples run by `tests/test_docs.py`.
- [ ] **T6. The brickwork baseline.** `tests/test_brickwork_correlated.py` and `benchmarks/correlated_erasure_baseline.py`
  with its report: every identity on real nearest-neighbour brickwork codes, `J -> ∞` against intervals, and the
  literature's expectations.
- [ ] **T7. Whole-branch review and fixes;** the owner's SmallWorldCodes driver (`scripts/ws_erasure.py`) gains the
  correlated erasures (left uncommitted for the owner).
