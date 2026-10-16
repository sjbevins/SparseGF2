# Hardening: a tested simulator that catches its own regressions

Status: **in progress**, 2026-10-14, branch `hardening`.

## 1. Goal

Before the research campaigns start, the simulator should be fully working and
fully tested. Two things are wanted:

* every number the research uses is checked against an **independent oracle**
  (Stim, a state vector, brute force or a closed form), never only against
  another path through the same code;
* a bug introduced later **fails CI at once**, instead of turning up months
  later in a figure.

Issue #2 (the reverse log) waits for this, as do further features.

## 2. What the review found

Five independent reviews ran on `main` at `7dfd80d`. Each confirmed finding
came with a reproduction script. The reports are kept outside the repository,
in `.superpowers/hardening/review/`.

* **Engine.** The tableau kernels, the compiled run loop, noise sampling and the
  parser are clean: 0 mismatches with Stim in fuzzing up to n = 257. The
  ensemble engine is also clean: 0 mismatches with its exported circuits over
  850 random specs and at the word boundaries. The erasure numbers are clean:
  k, I(A:R), the exhaustive counts, the first-failure law, Phi, mu, e_delta and
  Kaplan-Meier all agree with independent oracles up to n = 4096.
* **Bugs are in the code around the engine.** There are 32 confirmed findings.
  The ones that can give wrong research numbers are:
  * **N1:** the documented Kaplan-Meier recipe censors at `final_layer`, which
    is later than the last purification test;
  * **A1:** summary P_C(m) curves average in trajectories without a code;
  * **N4:** replay does not reproduce the sampled probes;
  * **N2:** `dataclasses.replace` keeps stale resolved defaults;
  * **N3/A8:** malformed edge arrays are silently reinterpreted.
* **Tests.** 7 of 15 single-line mutants survived the whole suite. Four of
  them would bias published numbers without any symptom:
  * the last qubit is never measured;
  * the first-failure mask is wrong when s is a multiple of 64;
  * e_delta is off at the junction of the exact counts and the samples;
  * the censored tau is averaged in.

  The ensemble protocol is checked only against itself, because the engine and
  its oracle share the circuit generator. The randomness contract has no
  known-answer values.

## 3. Rulings

Each ruling: what was decided, why, and what it costs if it is wrong.

* **R1: one branch, three phases.** Phase 1 is the safety net (gates and
  oracle tests). Phase 2 fixes the bugs, each test first. Phase 3 is the docs,
  the version (3.0.0.dev12), a whole-branch review and a PR. Phase 1 lands
  first, so every fix is made under the stricter checks. If this is wrong,
  some Phase 1 tests must be marked `xfail(strict=True)` until their fix in
  Phase 2. That is allowed, and the fix task removes the mark.
* **R2: N1 is fixed by giving the right censoring time, not by changing
  `final_layer`.** `EnsembleResult.censoring_layer` is the last spec
  checkpoint each trajectory was tested at. `EnsembleResult.purification_times()`
  returns `(time, observed)` ready for `analysis.kaplan_meier`. Every recipe
  (AGENTS.md, `analysis.kaplan_meier`, GETTING_STARTED, the purification
  example) uses it. `final_layer` keeps its meaning (last layer simulated).
  Cost if wrong: one more property on the result.
* **R3: N4 is fixed by making sampled probes a function of (seed, trajectory,
  probe name, layer).** Each evaluation of a sampled probe draws from its own
  stream, keyed by a stable 64-bit hash of the probe name and the layer, so a
  replay with other probes or checkpoints reproduces the samples exactly. This
  changes the randomness contract, so `RANDOMNESS_CONTRACT` becomes 2.
  * The same bump fixes kernel suspicion S1: `stream_state` folds its 128-bit
    key to 64 bits. Both key words must be mixed in.
  * The contract number is now saved with every `EnsembleResult` and API
    output, and is checked by `load`, `concatenate`, `replay`, and job
    resume/extend, which refuse to mix contracts.

  Cost if wrong: results saved under contract 1 cannot be extended. No
  research campaign has been run on v3 yet, so this is the cheapest moment.
* **R4: A1 is fixed by reporting P_C(m) conditioned on a code.** The summary
  curve of `erasure_first_failure` is averaged over the trajectories that hold
  a code (some sample <= n), with `count_with_code`. Its `what` line says so.
  * `uncorrectable_region_counts` is conditioned on a `code_dimension` probe at
    the same layer when the spec has one.
  * Otherwise its `what` line and a note say that trajectories without a code
    count as 0.

  This matches AGENTS.md's advice to condition on k > 0.
* **R5: the MCP server fixes that the current server needs now are made here**
  (A6: errors reach the client with their message; A13: strict integer and
  boolean arguments). The agent-interface branch (`agent-stacks`) replaces the
  server later and must carry both behaviors. The agent-interface ledger
  records them as inherited requirements.
* **R6: the JIT-free fallbacks for `popcount` and `ctz` are kept in the
  package.** With them, `NUMBA_DISABLE_JIT=1` runs the kernels as Python. That
  gives kernel coverage, and out-of-range indexing becomes an error. A full
  JIT-free suite takes hours, so CI runs a chosen kernel subset weekly. The
  bounds-checked JIT leg runs on every push (about 60 s).
* **R7: mypy (default strictness) is enforced on `src/`.** The 62 current
  errors are fixed by narrowing, not by `# type: ignore`, except where numpy's
  stubs are wrong. Those few get an ignore with the specific error code and a
  reason.

## 4. Gates added to CI

| gate | catches |
|---|---|
| pytest `filterwarnings = error`, `--strict-markers`, `xfail_strict` | silent NaN/RuntimeWarning, tests that stop running |
| networkx in `[test]` (no skipped comparison) | graph builders drifting from the reference |
| `NUMBA_BOUNDSCHECK=1` leg | out-of-range reads/writes in kernels (silent memory corruption under the JIT) |
| coverage with a floor on the Python modules | untested new code |
| mypy on `src/` | None and type slips in the service layer |
| hypothesis properties (CI profile; a deep profile weekly) | whole families of estimator bugs, with shrunk counterexamples |
| weekly scheduled workflow: deep fuzz, JIT-free kernel subset with coverage | rare and slow-to-find bugs |
| known-answer randomness tests tied to `RANDOMNESS_CONTRACT` | silent changes to the random streams |

## 5. Oracles added

* **An independent protocol oracle.** A Python transcription of the documented
  draw order (`_kernel/ensemble.py` docstring) must rebuild
  `spec.circuit(t)` exactly. Statistical tests check the laws of every draw:
  the measurement rate per qubit, the gate index, edges, the partner, the
  scramble and the orientation.
* **Engine vs exported circuits with references above 64 qubits** and at
  63/64/65/127/128/129.
* **One chunk/thread invariance test with every feature at once.** It covers
  a collection with different edge counts, several circuits per graph, every
  probe, sampled probes, snapshots and `stop_when_purified`.
* **Erasure fixtures spanning several words.** These are codes with logical
  qubits and distance >= 2, with s in {18, 63, 64, 65, 128}, checked by an
  independent GF(2) brute force. The first-failure law at n >= 64 is checked
  against closed forms. Every fixture asserts that it is not trivial.
* **Summary fields against numpy:** every `stderr`, `tau_mean_purified`,
  `tau_median_all` at the 1/2 boundary, and the quantiles.
* **Lock-step extensions:** destabilizer and symplectic invariants after every
  phased operation, random-state sizes 63/127/128/129/200, MR*/MXX/MYY/MZZ in
  the operation set, and a GF(2)-rank entropy oracle at the word boundaries.
* **Watts-Strogatz against the model:** rewired fraction, clustering and the
  degree histogram, compared with networkx.
* **The Python blocks of AGENTS.md and LLM_INTERFACE.md are executed** by
  `test_docs.py`.

## 6. Acceptance

* Every confirmed finding is fixed, or explicitly documented as intended with
  a test pinning the documented behavior.
* The seven surviving mutants of the audit are caught by the suite. The audit
  driver's mutants are re-run at the end.
* CI is green on all legs, and the weekly workflow passes once
  (`workflow_dispatch`).
* AGENTS.md, DESIGN.md section 10 and the CHANGELOG describe the new
  guarantees.
