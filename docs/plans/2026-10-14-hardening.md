# Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- Every research number is checked against an independent oracle.
- Every confirmed bug is fixed, test first.
- CI gates fail as soon as a regression is introduced.

**Architecture:**
- **Phase 1** adds gates and oracle tests. Its only behavior change is the JIT-free kernel fallbacks.
- **Phase 2** fixes the confirmed review findings (IDs such as N1 and A1 refer to the review reports), each starting with a failing test.
- **Phase 3** updates the docs and the version, and re-runs the audit's mutants.

**Tech Stack:**
- Python >= 3.11, numpy, numba 0.68 (kernels);
- pytest, with stim as the oracle;
- networkx and hypothesis (test only);
- mypy and ruff;
- GitHub Actions.

**Spec:** [`docs/specs/2026-10-14-hardening-design.md`](../specs/2026-10-14-hardening-design.md) (rulings R1-R7). The review reports and their repro scripts are not in git; they are in the worktree at `.superpowers/hardening/review/{kernel,erasure,ensemble,api,tests}.md` and `.superpowers/hardening/review/scripts/<area>/`. Each finding ID in this plan refers to those reports. Read the finding's section before implementing it.

## Global Constraints

- **Python 3.11 floor:** use no syntax or stdlib API newer than 3.11. CI runs 3.11.
- **Dependencies:** add no new runtime dependency; numpy and numba only. Test-only additions go in `[test]`: `networkx`, `hypothesis`, `pytest-cov`. Lint-only additions go in `[lint]`: `mypy`, `pre-commit`.
- **Commits:**
  - The repository owner is the only author. Never add `Co-Authored-By`, "Generated with" or any other AI or tool trailer.
  - Messages follow `git log` style: `<area>: <what changed>`, with a lowercase area (`tests:`, `ci:`, `mipt:`, `api:` ...).
  - Commit only your task's files.
- **Every task ends with these passing** (Windows paths, from the worktree root `C:\Projects\SparseGF2-hardening`):
  - `.venv\Scripts\python.exe -m pytest`
  - `.venv\Scripts\ruff.exe check src tests`
  - `.venv\Scripts\ruff.exe format --check src tests`
  - from Task 3 on, `.venv\Scripts\python.exe -m mypy`
- **Determinism:** tests use fixed seeds or the `rng` fixture.
  - Statistical tests use fixed seeds and acceptance bands of at least 4.5 sigma. Compute sigma in the test and state it in a comment.
  - A new test file should take at most about 10 s.
  - The whole suite must stay under 150 s on this machine (the baseline is 71 s). Report the new total.
- **Oracle independence:** an oracle never computes its expected value with the function under test, or with `sparsegf2._kernel`. The deliberate exceptions are the Python transcriptions under `tests/` written for that purpose: `_reference_rng.py`, `_protocol_oracle.py` and `_gf2_oracle.py`.
- **Text files** are read and written with `encoding="utf-8"`.
- **Behavior changes:**
  - Public behavior changes only in Phase 2.
  - Every Phase 2 task adds its bullets to an `## Unreleased` section at the top of `CHANGELOG.md` (create it if it is missing). Task 24 turns that section into the dev12 entry.
- **Protected files:**
  - Edit `.github/workflows/` only in Tasks 2, 3 and 10.
  - Never edit `docs/specs/2026-10-13-*` or `docs/plans/2026-10-13-*`.
- **Style:** match the surrounding code's naming, comment density and docstring style. Error messages say what to change, as the existing ones do.
- **Environment:** use the worktree's own interpreter, `.venv\Scripts\python.exe`. The main checkout `C:\Projects\SparseGF2` is never touched.

## Review Focus

1. **n at the 64-bit word boundaries** (63, 64, 65, 127, 128, 129) with references (purification and single). Results must agree exactly with the oracles; that they merely run is not enough. Tests: Tasks 5, 7 and 8.
2. **Ensembles in which some trajectories have k = 0.** Every code statistic must either condition on k > 0 or say plainly that it does not. Tests: Tasks 13 and 14.
3. **Checkpoint lists that end before `layers`, per-probe schedules, and `stop_when_purified`.** The censoring time, replay and the summaries must stay right. Tests: Tasks 6, 12 and 15.
4. **Results or output directories written by an older version or randomness contract.** They must be refused with a clear message, never silently mixed. Tests: Tasks 15 and 19.
5. **Inputs of the wrong type or shape** (fractional indices, `(2, m)` edge arrays, lists where a string is expected). They must raise a `ValueError` or `TypeError` that names the field, never be truncated, re-paired or crash with a traceback. Tests: Tasks 17 and 21.

## Shared test helpers (interfaces between tasks)

| Helper | Created in | Used in |
|---|---|---|
| `tests/_reference_rng.py`: pure-Python `splitmix64`, `stream_state(key, index)`, `Xoshiro` with `next_u64`, `next_double`, `next_below`, `next_bit`, and `RANDOMNESS_CONTRACT_MODEL` | Task 4 | Tasks 5 and 15 (contract 2 updates it) |
| `tests/_protocol_oracle.py`: `oracle_circuit_text(spec, trajectory, seed) -> str` | Task 5 | Task 15 |
| `tests/_gf2_oracle.py`: `rank`, `normalizer_dim_on`, `stabilizer_dim_on`, `lost_information`, `uncorrectable_counts`, `entropy` | Task 7 | Tasks 8 and 10 |
| hypothesis profiles `ci` and `deep`, selected by `HYPOTHESIS_PROFILE` | Task 10 | the weekly workflow |
| no-skip gate (`SPARSEGF2_NO_SKIPS=1`) | Task 1 | CI (Task 2) |

---

# Phase 1: the safety net

### Task 1: pytest gates and test dependencies

**Files:**
- Modify: `pyproject.toml` (`[tool.pytest.ini_options]`; `[project.optional-dependencies] test`)
- Modify: `tests/conftest.py` (no-skip gate)
- Modify: `tests/test_graphs.py` (`test_from_networkx` always runs)
- Modify: `tests/test_api.py` (the tolerance at about line 316)
- Create: `tests/test_meta.py`

**Interfaces:**
- Produces the `SPARSEGF2_NO_SKIPS` environment switch. Task 2 sets it in CI.

**Requirements:**
- In pytest: set `filterwarnings = ["error"]`, `xfail_strict = true`, and add `--strict-markers` to `addopts`. Keep `-q --doctest-modules`, `testpaths` and `pythonpath`.
- `[test]` becomes `["pytest>=8", "stim>=1.15", "matplotlib>=3.8", "networkx>=3.2", "hypothesis>=6.100", "pytest-cov>=5"]`, with a comment for each new entry in the style of the existing comments. Install them into the worktree venv: `uv pip install --python .venv\Scripts\python.exe -e ".[dev]"`.
- In `tests/test_graphs.py`, networkx is imported at module level, and `test_from_networkx` no longer uses `importorskip`.
- `tests/test_api.py` (about line 316): a value the API rounds to 6 significant digits is compared with `rel=1e-6`, which fails for fractions such as 11/60. Compare with `pytest.approx(expected, abs=1e-6)`.
- **No-skip gate.** When the environment variable `SPARSEGF2_NO_SKIPS=1` is set, a hook in `tests/conftest.py` turns any skipped test into a session failure (`session.exitstatus = 1`) and prints the skipped test ids and reasons. The one exception is a skip whose reason is exactly `"no working bash"` (the docs test on machines without bash). Implement it with `pytest_runtest_logreport` to collect the skips and `pytest_sessionfinish` to fail.
- If any warning now turns into an error, fix the cause. Add a narrow `filterwarnings` ignore only for third-party warnings you cannot fix, with a comment naming the package and the reason.

**Tests (`tests/test_meta.py`):**
- `test_warnings_are_errors`: `with pytest.raises(UserWarning): warnings.warn("x", UserWarning)`.
- `test_no_skip_gate_fails_a_skipping_session`: use `pytester` (`pytest_plugins = ["pytester"]` at the top of `tests/test_meta.py`).
  - Copy the gate hook into the inner session, either by `pytester.makeconftest` with the hook source, or by importing a helper module from `tests`.
  - Run a file with one `pytest.skip("x")` test with `SPARSEGF2_NO_SKIPS=1`, and assert the outcome is a failure (`result.ret != 0`).
  - Run it without the variable, and assert success.
  - Also check that the allowed reason does not fail.

**Steps:**
- [ ] Write the `test_meta.py` tests and run them. They fail: there is no gate yet, and warnings are not errors.
- [ ] Implement the pyproject, conftest, `test_graphs` and `test_api` changes. Install the extras.
- [ ] Run the full suite. Then run it again with `SPARSEGF2_NO_SKIPS=1` (PowerShell: `$env:SPARSEGF2_NO_SKIPS='1'`). Both must pass with zero skips (`test_from_networkx` now runs).
- [ ] Commit: `tests: warnings are errors, strict markers and xfails, no silent skips; networkx and hypothesis join the test extras`

### Task 2: JIT-free kernel fallbacks and kernel safety in CI

**Files:**
- Modify: `src/sparsegf2/_kernel/bits.py`
- Create: `tests/test_bits.py`
- Modify: `pyproject.toml` (`[tool.coverage.*]`)
- Modify: `.github/workflows/ci.yml`
- Create: `.github/workflows/weekly.yml`

**Requirements:**
- **Fallbacks in `bits.py` (R6).** If `numba.config.DISABLE_JIT` is true when the module is imported, `popcount` and `ctz` are plain Python functions:
  - `popcount(x)` returns `np.uint64(int(x).bit_count())`;
  - `ctz(x)` returns `np.uint64((x & -x).bit_length() - 1)` for `x != 0`, and raises `ValueError("ctz(0) is undefined")` for 0. The intrinsic is undefined there ("callers check"), so the JIT-free run turns a caller bug into an error.
  - The JIT path is unchanged. Comment why the fallback exists.
- **`tests/test_bits.py`:**
  - The compiled intrinsics are called through small `@njit` wrappers, since intrinsics cannot be called from Python. They must equal the Python reference (`int.bit_count`, lowest set bit) on 0 (popcount only), 1, 2**63, 2**64-1, 0x5555555555555555, 0xAAAAAAAAAAAAAAAA, and 1000 random words from a fixed seed.
  - A subprocess test (`sys.executable -c ...` with `NUMBA_DISABLE_JIT=1` in the environment) imports sparsegf2. In that subprocess:
    - check that the fallbacks give the same values;
    - check that `ctz(0)` raises `ValueError`;
    - run a Bell pair through `Simulator(2)` (`h`, `cx`) and check that its entropy is 1.
- **Coverage config:**
  - Set `[tool.coverage.run] source = ["sparsegf2"]`, `branch = true`, `omit = ["*/sparsegf2/_kernel/*"]`. Compiled code is not traced; the kernels are covered by the JIT-free weekly run.
  - Set `[tool.coverage.report] show_missing = true`, `skip_covered = true`, and `fail_under = F`. Here `F` is the measured total of the JIT run with that omit, rounded down to an integer, minus 1. Measure it with `python -m pytest --cov`, and report the measured value and `F`.
- **`ci.yml`:**
  - Set `SPARSEGF2_NO_SKIPS: "1"` in the env of every job.
  - Add job `boundscheck`: ubuntu-latest, Python 3.12, install `.[test,mcp]`, with env `NUMBA_BOUNDSCHECK: "1"` and `NUMBA_CACHE_DIR: ${{ runner.temp }}/numba-boundscheck`, running `python -m pytest`.
  - Add job `coverage`: ubuntu-latest, Python 3.12, install `.[test,mcp]`, running `python -m pytest --cov`. `fail_under` comes from the pyproject.
  - Keep `persist-credentials: false` and the existing style.
- **`weekly.yml`:**
  - Triggers: `schedule: cron "0 6 * * 1"` and `workflow_dispatch`.
  - Job `jit-free-kernels`: ubuntu-latest, Python 3.12, with env `NUMBA_DISABLE_JIT: "1"` and `SPARSEGF2_NO_SKIPS: "1"`, and `timeout-minutes: 150`. It runs a kernel-heavy subset with `--cov=sparsegf2._kernel --cov-branch --cov-report=term-missing`. Override `omit` there with `--cov-config` pointing to a small `.github/coverage-kernels.toml`, or with an inline `[tool.coverage]` alternative; choose and document.
  - Choose the subset by measuring locally so that it finishes in under about 60 minutes on a CI runner (about 2x slower than this machine). Start from `tests/test_bits.py`, `tests/test_simulator.py` restricted to sizes at most 65, `tests/test_observables.py`, `tests/test_codes.py` and `tests/test_reference_rows.py`. Report the measured local time and the kernel coverage.
  - Hypothesis's deep profile is added to this file in Task 10.

**Steps:**
- [ ] Write `tests/test_bits.py` and confirm the subprocess test fails, because `popcount` is not usable in pure Python.
- [ ] Implement the fallbacks. Run the new tests and the full suite.
- [ ] Add the coverage config, measure, and set `fail_under`.
- [ ] Locally, run the full suite once with `NUMBA_BOUNDSCHECK=1` and a separate `NUMBA_CACHE_DIR`, and run the chosen JIT-free subset. Report both times.
- [ ] Edit `ci.yml` and add `weekly.yml`. Validate the YAML by loading it with Python (`yaml` if available; otherwise check it by eye against the existing file).
- [ ] Commit: `kernel: pure-Python popcount and ctz when the JIT is disabled; ci: bounds-checked and coverage legs, weekly JIT-free kernel run`

### Task 3: mypy runs clean, and pre-commit hooks

**Files:**
- Modify: `pyproject.toml` (`[tool.mypy]`; the `lint` extra)
- Modify: whichever `src/sparsegf2/*.py` files carry the errors (about 62, in default mode: `api.py`, `mipt.py`, `_compile.py`, `simulator.py`, `circuit.py`, `graphs.py`, `_parallel.py`)
- Create: `.pre-commit-config.yaml`
- Modify: `.github/workflows/ci.yml`
- Modify: `AGENTS.md` ("Testing changes")

**Requirements (R7):**
- `[tool.mypy]`: `files = ["src/sparsegf2"]`, `python_version = "3.11"`, `warn_unused_ignores = true`, `warn_redundant_casts = true`. Add `[[tool.mypy.overrides]]` with `ignore_missing_imports = true` for `numba.*`, `llvmlite.*`, `stim`, `mcp.*` and `pydantic.*` only if needed. Use default strictness otherwise.
- Fix every error by narrowing: `if x is None: raise ...`, an `assert x is not None, "why"` where an invariant guarantees it, local variables, or correct annotations.
  - `# type: ignore[<code>]  # <reason>` is allowed only where numpy's stubs are wrong (for example `np.savez(path, **arrays)`). It must name the error code and give a reason.
  - If an error reveals a real bug, fix it with a regression test, and say so in your report.
- The `lint` extra becomes `["ruff>=0.5", "mypy>=1.10", "pre-commit>=3.7"]`.
- `.pre-commit-config.yaml`:
  - `astral-sh/ruff-pre-commit`, pinned to the installed ruff version (`ruff --version`), with hooks `ruff` (args `[--fix]`) and `ruff-format`.
  - `pre-commit/pre-commit-hooks` with `end-of-file-fixer`, `trailing-whitespace` (exclude `\.md$`), `check-yaml`, `check-toml` and `check-merge-conflict`.
  - A `local` hook `mypy`: `entry: python -m mypy`, `language: system`, `pass_filenames: false`, `types: [python]`.
  - Do not run `pre-commit install`: it writes into `.git/hooks`, which is the owner's call. Do run `pre-commit run --all-files` once to check the config works. If the hygiene hooks change files, include those changes only if they are whitespace/EOF fixes in tracked text files, and list them.
- In `ci.yml`, add a `Type check (mypy)` step after the format check, only on `ubuntu-latest` with Python `3.12` (`if:`), running `python -m mypy`.
- In `AGENTS.md` "Testing changes", add the commands `python -m mypy` and `pre-commit install` (once, optional), and a sentence on the CI legs: bounds-checked, coverage floor, weekly JIT-free kernels and deep fuzzing. Keep it short.

**Steps:**
- [ ] Install the lint extras and run `python -m mypy`. Record the error count.
- [ ] Fix the errors module by module. Run the suite after each module.
- [ ] Add the config, pre-commit, CI and AGENTS.md changes. Run `pre-commit run --all-files`.
- [ ] Commit: `typing: mypy runs clean on the package and in CI; pre-commit hooks for ruff, mypy and file hygiene`

### Task 4: known-answer tests for the randomness contract

**Files:**
- Create: `tests/_reference_rng.py`
- Create: `tests/test_randomness_contract.py`
- Modify: `tests/test_replay.py` (fold `test_contract_constant` into the new file or keep it; do not duplicate)

**Interfaces produced:** `tests/_reference_rng.py`, for Tasks 5 and 15:

```python
MASK64 = (1 << 64) - 1
def splitmix64(x: int) -> tuple[int, int]: ...            # (new state, output)
def stream_state(key: tuple[int, int], index: int) -> list[int]: ...  # 4 words, contract-1 mixing
class Xoshiro:
    def __init__(self, state: list[int]): ...
    def next_u64(self) -> int: ...
    def next_double(self) -> float: ...
    def next_below(self, bound: int) -> int: ...          # Lemire, 1 <= bound < 2**32
    def next_bit(self) -> int: ...
```

**Requirements:**
- Write `_reference_rng.py` from the published algorithms: xoshiro256** (Blackman and Vigna), splitmix64, and Lemire's nearly divisionless bounded integers. Use Python ints with explicit `& MASK64`.
  - Its `stream_state` documents the contract-1 key mixing exactly as the kernel does it today: `x = key0 ^ ((index * 0xD1B54A32D192ED03) & MASK64) ^ key1`, then four splitmix64 outputs, with all zeros mapped to `[1, 0, 0, 0]`.
  - Note in the module docstring that this mixing ignores part of the 128-bit key (kernel finding S1), and that Task 15 changes it under contract 2.
- **Tests:**
  1. **Published vector:** xoshiro256** from state `[1, 2, 3, 4]` gives first outputs `11520, 0, 1509978240, 1215971899390074240`. Assert this on the reference, and assert the kernel `next_u64` (via `sparsegf2._kernel.rng`) gives the same.
  2. **Kernel equals reference:** for 1000 draws from 3 seeded states, `next_u64`, `next_double`, `next_bit` and `next_below` agree. Use bounds 1, 2, 3, 11520 and 2**31-1. Use separate state copies per function.
  3. **Kernel `stream_state(key, index, out)`** equals the reference for 50 (key, index) pairs, including index 0 and 2**40.
  4. **Pinned literals tied to `RANDOMNESS_CONTRACT`:**
     - `EXPECTED = {1: {...}}` holds:
       - the 4 words of `stream_state((0x0123456789ABCDEF, 0xFEDCBA9876543210), 7)`;
       - sha256 hex digests of `str(spec.circuit(t, seed=...))` for three specs and t in {0, 3};
       - the sha256 of a small run's integer outputs.
     - Fill these values from the reference and the current package. The test asserts `mipt.RANDOMNESS_CONTRACT in EXPECTED` and that the current values equal `EXPECTED[RANDOMNESS_CONTRACT]`.
     - A comment says: "A changed value means the random draws changed. Bump `RANDOMNESS_CONTRACT` and add a new entry. Never edit an existing one."
  - **Specs:** build each with `MonitoredCircuit` directly and keep them small (n <= 12, layers <= 4).
    - S1: cycle n=8, `schedule="matching"`, purification, p=0.2, `scramble_gates=3`.
    - S2: a collection of 2 Watts-Strogatz graphs (`graphs.watts_strogatz(10, 4, 0.3, seed=(1, i))`), `circuits_per_graph=2`, `random_edges` with explicit edge weights, `random_orientation=True`, single reference with a random partner, `scramble_matchings=1`, `scramble_layers=1`.
    - S3: `all_edges` on a path of 6, with a restricted `gate_set` and no reference.
  - **The run digest:** take `run(6, seed=11)` of S1 with probes `code_dimension`, `erasure_first_failure(samples=8)` and `uncorrectable_region_counts(max_size=2)` at checkpoints `[0, 2, 4]`. Hash the bytes of each `values[name].astype("<f8")` (sorted names), then `purification_layer` and `num_measurements` as `<i8`. Use only integer-valued outputs, so platforms agree.
- The digests must be identical on Linux, macOS and Windows. Use only integer-valued data and text.

**Steps:**
- [ ] Write the reference module and tests 1-3. They pass against the kernel. If they fail, the reference transcription is wrong; fix it, never the kernel.
- [ ] Compute and paste the pinned literals (test 4).
- [ ] Temporarily change the kernel's `stream_state` multiplier and confirm test 4 fails. Revert, and report the check.
- [ ] Commit: `tests: known-answer values pin the randomness contract (xoshiro256**, stream keys, circuits, a small run)`

### Task 5: an independent protocol oracle for the ensemble engine

**Files:**
- Create: `tests/_protocol_oracle.py`
- Create: `tests/test_protocol.py`
- Modify: `tests/test_mipt.py` (the `SPECS` used by `test_engine_matches_exported_circuits_exactly`)

**Interfaces:**
- Consumes `tests/_reference_rng.py` (Task 4).
- Produces `oracle_circuit_text(spec: MonitoredCircuit, trajectory: int, seed) -> str`, which is equal to `str(spec.circuit(trajectory, seed=seed))`.

**Requirements:**
- **The oracle.** `_protocol_oracle.py` transcribes the documented draw order: the `_kernel/ensemble.py` module docstring, items 1-4, plus the `MonitoredCircuit` docstring for setup and the schedules. It uses only `_reference_rng.py`, numpy's `SeedSequence` (to derive the keys exactly as `mipt._keys` does; read it), and plain Python. It must not import `sparsegf2._kernel` or call `MonitoredCircuit.circuit`. It may read the spec's public fields and graph arrays.
  - It must cover:
    - every schedule (`matching`, including the color cycle continuing from scramble layers into the dynamics; `random_edges` unweighted and weighted, with the cumulative search; `all_edges`);
    - `gates_per_layer`;
    - scramble pairs, matchings (Fisher-Yates) and layers;
    - every reference mode (none, single fixed, single random, purification);
    - `random_orientation` and `gate_set`;
    - graph collections (trajectory t on graph `t // circuits_per_graph`);
    - `first_trajectory`-style indices (any `t`).
  - The text must be produced in the same format as `Circuit.__str__`: compare by building the instruction list and formatting it the way `MonitoredCircuit.circuit` names gates; read that function for naming only.
- **Where the oracle and the engine disagree,** find out which is wrong. If the engine or its docstring is wrong, stop and report it as `DONE_WITH_CONCERNS`, with the evidence. Do not change engine code in this task.
- **`tests/test_protocol.py`:**
  - The oracle equals `spec.circuit(t, seed)` for t in {0, 1, 7}, over a grid of at least 12 specs covering every option above (small n, few layers).
  - The laws of the draws, with fixed seeds and bands of at least 4.5 sigma:
    - **Per-qubit measured fraction:** n=12, p=0.3, 400 layers over 5 trajectories. Each qubit's count, including qubit n-1, is within 4.5 sigma of the binomial mean, and so is the total.
    - **Gates per layer:** `random_edges` gives exactly `gates_per_layer`; `all_edges` gives |E|; `matching` gives exactly one color class per layer, and each class is a matching.
    - **Gate index:** a chi-square over 64 buckets (`index % 64`) and over 64 buckets (`index // 180`) for 20 000 draws gives p-value > 1e-6.
    - **Unweighted edge choice:** chi-square uniform. **Random partner:** uniform. **Orientation flip fraction:** 1/2 within 4.5 sigma.
- **Additions to `tests/test_mipt.py` SPECS** (engine against its exported circuits):
  - purification n=40 (80 qubits);
  - single n=70;
  - purification at n=63, 64 and 65 (layers <= 6);
  - single at n=127, 128 and 129 (layers <= 4).

  Each uses probes `code_dimension`, `lost_information` on a region that crosses qubit 64, `uncorrectable_region_counts(max_size=2)`, `contiguous_distance` and `code_snapshot`. Keep the trajectories per spec small, so the parametrization adds less than about 15 s.
- **Mutant check:** temporarily change `gen_layer` to measure `range(n_sys - 1)`, and confirm `test_protocol.py` fails. Revert, and report it.

**Steps:**
- [ ] Write the oracle and its equality test. Iterate until it passes or a real discrepancy is found.
- [ ] Write the law tests.
- [ ] Extend SPECS.
- [ ] Run the mutant check.
- [ ] Commit: `tests: an independent transcription of the ensemble draw order, the laws of every draw, and engine-vs-circuit checks with references above 64 qubits`

### Task 6: one chunk, thread and replay invariance test with every feature

**Files:**
- Create: `tests/test_invariance.py`

**Requirements:**
- **Spec:**
  - a collection of 4 Newman-Watts graphs with *different* edge counts, built with `graphs.newman_watts(20, ...)` from different seeds; assert that the edge counts differ;
  - `circuits_per_graph=3` (12 trajectories);
  - single reference with a random partner;
  - `stop_when_purified=True`;
  - `layers=12`, checkpoints every 2;
  - p chosen so that some, but not all, trajectories purify; assert both.
- **Probes:** `half_cut_entropy`, `entropy_profile` (its own checkpoints, stride 4), `code_dimension`, `lost_information` on 3 qubits, `contiguous_distance`, `erasure_first_failure(samples=6)`, `erasure_lost_profile(samples=3)`, `uncorrectable_region_counts(max_size=2)`, and `code_snapshot` (its own checkpoints, `"final"`).
- **Assertions** (`np.testing.assert_array_equal`, which treats NaN as equal):
  - `run(12, seed=5)` equals the concatenation (`EnsembleResult.concatenate`) of four chunks cut at 1, 7 and 11: `run(1)`, `run(6, first_trajectory=1)`, `run(4, first_trajectory=7)`, `run(1, first_trajectory=11)`. Compare every `values` array, `purification_layer`, `final_layer`, `num_measurements`, `trajectories` and `snapshots`.
  - `num_threads=1` equals `num_threads=min(4, numba.config.NUMBA_NUM_THREADS)`.
  - `replay(t)` with the same probes and checkpoints equals row t, for t in {0, 5, 11}.
- **Mutant check:** temporarily apply `graph_index_chunk` (the engine graph index `t // c` instead of `(first + t) // c`) and `probe_stream_chunk` (the probe stream `stream_state(key_probe, t)` instead of `first + t`). Each must fail this test. Revert, and report it.
- **Collections (DESIGN 8.1):**
  - A collection of one graph gives the same result as that graph given alone.
  - Appending graphs to a collection leaves the earlier graphs' trajectories unchanged (compare the common rows).
- **Edge cases:** each must run and round-trip.
  - A zero-trajectory run saved, loaded and concatenated with a non-empty run.
  - `layers=0` with `stop_when_purified`.
  - A checkpoint stride larger than `layers`.
  - A per-probe schedule disjoint from the spec's under `stop_when_purified`.

  The reviewer's `scripts/ensemble/bank_append.py` and `edge_specs.py` show these cases.

**Steps:**
- [ ] Write the test. It passes on the current code.
- [ ] Run the mutant checks.
- [ ] Commit: `tests: chunked, threaded and replayed runs of a full-featured spec equal one run`

### Task 7: erasure fixtures spanning several words, with a brute-force oracle

**Files:**
- Create: `tests/_gf2_oracle.py`
- Create: `tests/test_erasure_oracle.py`
- Modify: `tests/test_codes.py`

**Interfaces produced:** `tests/_gf2_oracle.py`, for Tasks 8 and 10. Paulis are `(x_bits, z_bits)` pairs of Python ints, or numpy uint8 arrays.

```python
def rank(rows: list[int]) -> int                      # GF(2) rank of bit-rows (Python ints)
def stabilizer_dim_on(gens, n, region) -> int         # dim of the stabilizer group restricted to elements supported in region
def normalizer_dim_on(gens, n, region) -> int         # dim of N(S) elements supported in region
def lost_information(gens, n, region) -> int          # = normalizer_dim_on - stabilizer_dim_on  (cleaning lemma; = I(A:R) in bits)
def uncorrectable_counts(gens, n, w) -> list[int]     # for m = 1..w: number of m-subsets A with lost_information > 0
def entropy(gens, n, region) -> int                   # pure state (n independent generators): S(A) = rank(gens restricted to A) - |A|
```

**Requirements:**
- Implement the oracle from the definitions above, with symplectic null spaces built by Gaussian elimination. Use no sparsegf2 code. `.superpowers/hardening/review/scripts/erasure/oracle.py` and `codegen.py` are a tested reference you may adapt. Document the identities used and cite the cleaning lemma in a docstring.
  - Check the oracle on [[4,2,2]] (`l({0}) = 0`, `l({0,1}) = 2`, counts `[0, 6]`), on [[5,1,3]] (counts `[0, 0, 10]`), and on a repetition code.
- **Fixtures.** Build codes with logical qubits (k > 0) and with small, known distance, so that the counts are not trivial:
  - a random code from `stim.Tableau.random(n)` (the images of Z_0..Z_{s-1} give s generators);
  - combined with a few [[4,2,2]] blocks and frozen qubits on chosen qubits, then a random permutation of the qubits.
  - Required (n, s) pairs: (70, 18), (70, 63), (70, 64), (70, 65), (130, 64) and (130, 128).
  - Every fixture asserts `k > 0`. Across the fixtures, both zero and nonzero counts at m = 2 must occur. Assert that too.
- **`tests/test_erasure_oracle.py`:**
  - `StabilizerCode.uncorrectable_counts(2)` equals the oracle exactly on every fixture. `uncorrectable_counts(3)` equals it on the n = 70, s = 64 fixture.
  - `lost_information` and `is_correctable` equal the oracle on 200 random regions per fixture (sizes 1..8 and n-8..n).
  - **The first-failure law on block codes with closed forms.** Take b copies of [[4,2,2]] plus f frozen qubits, with n in {63, 64, 65, 127, 128, 129} and at least one case with s a multiple of 64. 20 000 sampled D values (`first_failure_sizes`) match the exact law within 4.5 sigma per m (or pass a chi-square with p > 1e-6).
    - Derive the exact law in the test. D is the first m at which the erased set contains two qubits of the same block. The reviewer's `scripts/erasure/t05_first_failure_law.py` derives it.
- **Mutant check:** temporarily drop the `s % 64 == 0` case of the first-failure last-word mask in `_kernel/entropy.py` (the reviewer's mutant `fmask_word_first_failure`). The new tests must fail. Revert, and report it.
- **Fix the vacuous tests in `tests/test_codes.py`:**
  - `test_hybrid_estimator_uses_exact_counts_for_small_regions` uses a k = 0 code. Use a fixture with k > 0 and nonzero counts, and assert that it is not trivial.
  - `test_counts_on_codes_wider_than_one_word` uses a 0.15 sampling tolerance. Replace it with the exact oracle comparison, or delete it in favor of the new file, and say which.
  - Where `_engine_codes` fixtures can be k = 0, assert that at least one used code has k > 0 and nonzero counts.

**Steps:**
- [ ] Write the oracle and its self-tests.
- [ ] Build the fixtures and write the comparison tests.
- [ ] Run the mutant check.
- [ ] Fix the vacuous tests.
- [ ] Commit: `tests: brute-force GF(2) oracle for erasure counts and the first-failure law on codes spanning several words; no vacuous fixtures`

### Task 8: lock-step extensions

**Files:**
- Modify: `tests/_lockstep.py`
- Modify: `tests/test_simulator.py`

**Interfaces:**
- Consumes `tests/_gf2_oracle.py` `entropy` (Task 7).

**Requirements:**
- **Invariants.** Whenever `check_state` runs, also check the phased simulator's full tableau (DESIGN 10.3):
  - stabilizers pairwise commute;
  - destabilizer i anticommutes with stabilizer i and commutes with every other stabilizer;
  - destabilizers pairwise commute.

  Use the simulator's public tableau export if one exists (look for `to_tableau`/`tableau` in `simulator.py`). Rebuilding a `stim.Tableau` with `from_conjugated_generators`, which validates the pairing, is acceptable. Otherwise check the commutation relations directly on the exported rows. Never read private kernel arrays if a public path exists.
- **Random-state lock-step sizes** (`test_lockstep_from_random_states`): add 63, 127, 128, 129 and 200. Use fewer ops and a larger `check_every` at large n, so the additions take under about 10 s.
- **New operations in `Lockstep.random_op`:**
  - `MR`, `MRX`, `MRY`: measure then reset, with the outcome forced on Stim by postselection when it is random, as `measure` does.
  - `MXX`, `MYY`, `MZZ` on two distinct qubits, with and without inversion of either target: compare with Stim through `peek_observable_expectation` and `postselect_observable`, as `measure_pauli` does.
  - Keep the existing probabilities roughly as they are, carving out about 8% for these operations.
- **GF(2)-rank entropy oracle:** for n in {63, 64, 65, 127, 128, 129, 200}, build random states from `stim.Tableau.random(n)` (from_tableau). For random regions A, `Simulator.entropy(A)` equals `_gf2_oracle.entropy`. Use sizes 1, 2, n//2-1, n//2+1 and n-1; those above n/2 exercise the complement path. Check both phase modes.
- **Whole-circuit replay fuzz** (the random circuit generator in `tests/test_run.py` behind `test_replaying_stim_records_reproduces_everything`). Extend it to:
  - `MPP` with up to 40 factors, including repeated qubits and inverted factors, and several products per line;
  - broadcast `M`/`MX`/`MY`/`MR*` over up to 7 targets, and gates over several pairs;
  - `CZ q rec[-k]` (the record as the second target);
  - sizes 127, 128, 129 and 192.

  The kernel reviewer's `scripts/kernel/fuzz_run.py` does this; adapt it. Keep the additions under about 10 s.
- **Reference records:** `Program.reference_record` equals `stim.Circuit.reference_sample()` on 100 random Z-basis/`MR`/`R`/`MZZ`/`MPP` circuits (`scripts/kernel/fuzz_reference.py`). X/Y-basis multi-target instructions are excluded, because their documented ordering differs.

**Steps:**
- [ ] Add the invariant check. Run the existing lock-step tests, which must still pass.
- [ ] Add the operations and sizes.
- [ ] Add the entropy oracle test.
- [ ] Commit: `tests: lock-step checks the destabilizer pairing, covers MR*/MXX/MYY/MZZ and random states at the word boundaries; GF(2)-rank entropy oracle`

### Task 9: Watts-Strogatz and the other graph families, checked against their models

**Files:**
- Create: `tests/test_graph_models.py`

**Requirements** (fixed seeds; compare sparsegf2 graphs with networkx graphs of the same parameters):
- **Watts-Strogatz:**
  - For beta in {0, 0.1, 0.5, 1}, n in {12, 101}, k in {2, 4, 6}: exactly `n*k/2` edges, no self-loops and no duplicates.
  - At beta = 0, the graph is the ring lattice with total degree k.
  - Every node has degree at least k/2 (each node keeps its own k/2 forward edges in the networkx algorithm).
  - For n=100, k=4, beta=0.2, compare 300 sparsegf2 graphs with 300 `networkx.watts_strogatz_graph` graphs:
    - the mean shortcut fraction (`graphs.shortcut_fraction`, also applied to the networkx edges) and the mean average clustering (`networkx.average_clustering`) agree within 4.5 pooled standard errors;
    - the degree histograms agree by a chi-square two-sample test with p > 1e-6.
  - Edge cases: n = k + 1 and n = k + 2. Odd k and k >= n behave as documented. If the docs say nothing, add the documented behavior to the docstring and test it.
- **`random_regular`:** every degree equals d, the graph is simple, and the mean clustering is within 4.5 sigma of networkx's `random_regular_graph` over 200 samples.
- **`newman_watts` and `erdos_renyi`:** the mean edge count is within 4.5 sigma of its expectation (compute the expectation from the model).
- **`lattice_2d`:** degrees are 4 (periodic), with the documented open-boundary behavior.
- **`graphs.stats` against independent computations on 10 graphs,** including a disconnected one and one with an isolated node:
  - `mean_path_length` against `networkx.average_shortest_path_length` (connected graphs);
  - `clustering` against whichever networkx quantity matches the docstring definition (`average_clustering` or `transitivity`; read the docstring);
  - degrees against networkx;
  - `lambda2` against `numpy.linalg.eigvalsh` of a Laplacian built in the test.
  - For the disconnected graph, assert `lambda2 < 1e-12` here. Task 18 tightens this to `== 0.0`.

**Steps:**
- [ ] Write the tests. If a family disagrees with its model beyond the band, report it as `DONE_WITH_CONCERNS` with the numbers. Do not change `graphs.py` in this task.
- [ ] Commit: `tests: Watts-Strogatz and the other graph families checked against networkx and their models`

### Task 10: property-based tests for the estimators and codes

**Files:**
- Modify: `tests/conftest.py` (hypothesis profiles)
- Create: `tests/test_properties.py`
- Modify: `.github/workflows/weekly.yml` (a deep-profile job)

**Interfaces:**
- Consumes `tests/_gf2_oracle.py` (Task 7).

**Requirements:**
- **Profiles in `conftest.py`:**
  - `ci`: `max_examples=60`, `derandomize=True`, `deadline=None`, `print_blob=True`;
  - `deep`: `max_examples=2000`, `deadline=None`, `print_blob=True`.
  - Load `os.environ.get("HYPOTHESIS_PROFILE", "ci")`.
- **Analysis properties.** Draw first-failure arrays and counts from small random codes: n <= 10, codes from `stim.Tableau.random`, D sampled with `StabilizerCode.first_failure_sizes`, counts from `_gf2_oracle.uncorrectable_counts`. Include codes with k = 0.
  - `failure_by_size`: P(0) = 0; non-decreasing; for codes with k > 0, `P[n] == 1.0` exactly. Under ruling R-e, Task 14 (finding E2) has already landed. Add the [[10,4]] histogram from Task 14's regression test as an `@example`.
  - `erasure_failure_probability`:
    - Phi(0) = 0;
    - Phi(1) equals the fraction of codes with k > 0;
    - non-decreasing in e on a grid;
    - equal to the binomial mixture computed with `fractions.Fraction` and `math.comb` (n <= 12), within 1e-12.
  - **The hybrid estimator.** With `exact_counts` from the oracle, and samples that include D <= w:
    - `erasure_tolerance(delta)` agrees with a fine-grid inversion of `erasure_failure_probability(..., exact_counts=...)` within the grid resolution;
    - `Phi(e_delta) <= delta + 1e-9`.

    This catches the mutant `tolerance_hybrid_junction`. The current code is believed correct here. If it fails today, that is a new finding: report it with the shrunk example, and mark it `xfail(strict=True, reason="... fixed in Task 14")` only with an `@example` that fails deterministically.
  - `mean_first_failure` equals `sum_m (1 - hybrid P(m))` for m = 0..n.
  - `distance_bounds`: `1 <= low <= high <= n + 1`. If some count m <= w is positive, `low == high ==` the first such m. Codes with k = 0 give `(n + 1, n + 1)`.
  - `kaplan_meier` against a direct implementation written in the test on random (time, observed) data with ties: equal survival values. Without censoring, it equals the empirical survival.
- **Code properties** (n <= 8; s drawn so that k > 0 sometimes and k = 0 sometimes):
  - `lost_information` is monotone under inclusion;
  - `l(A) + l(complement A) == 2k` (purity of system + reference);
  - `is_correctable(A)` holds iff `l(A) == 0`;
  - `lost_information` equals `_gf2_oracle.lost_information`;
  - `uncorrectable_counts(w)` equals the oracle for w <= 3.
- **`weekly.yml`:** add job `deep-fuzz` (ubuntu-latest, Python 3.12, env `HYPOTHESIS_PROFILE: deep`, `SPARSEGF2_NO_SKIPS: "1"`, `timeout-minutes: 90`), running `python -m pytest tests/test_properties.py tests/test_simulator.py -k "lockstep or properties"`. Adjust the selection so that it runs the properties plus the lock-step fuzz.
- Keep the `ci` profile run of `tests/test_properties.py` under about 15 s.

**Steps:**
- [ ] Write the profiles and properties. Run them with the `ci` and `deep` profiles locally. Report the deep runtime and any failure. A deep failure with a shrunk example is a finding: report it.
- [ ] Commit: `tests: hypothesis properties of the erasure estimators, Kaplan-Meier and stabilizer codes; deep profile weekly`

### Task 11: the Python in AGENTS.md and LLM_INTERFACE.md runs

**Files:**
- Modify: `tests/test_docs.py`
- Modify: `AGENTS.md`
- Modify: `docs/LLM_INTERFACE.md` (only to make blocks runnable or to mark them `<!-- not run -->`)

**Requirements:**
- Add `"AGENTS.md"` and `"docs/LLM_INTERFACE.md"` to the documents whose Python blocks run.
- AGENTS.md's "Erasure analysis in Python" block loads `runs/erasure/result.npz`, which the JSON run just above it produces. Before running AGENTS.md's blocks, the test runs that documented JSON (find the JSON block that contains `"output_dir": "runs/erasure"` and pass it to `sparsegf2.api.run_ensemble(**args)`) in the temporary working directory. Treat any other block that depends on a JSON run the same way, through a small mapping in the test, rather than by editing the docs.
- **Schematic blocks** (literal `...` placeholders such as `mipt.MonitoredCircuit(...)`):
  - make them concrete and short when that keeps the guide readable;
  - otherwise mark them `<!-- not run -->`.
  - Make at least the "Distances, graphs, saved states and cross-checks" block of AGENTS.md runnable: replace `sim` and `mc` with tiny concrete objects defined in the block.
- The test fails, naming the block, if a block raises.
- Do not change the Kaplan-Meier recipe here; Task 12 owns it.

**Steps:**
- [ ] Extend the test and run it. Fix the docs until every block runs or is deliberately marked.
- [ ] Commit: `tests: the Python examples in AGENTS.md and LLM_INTERFACE.md run in the suite`

---

# Phase 2: fixes (each starts with a failing test)

### Task 12: N1, the right censoring time for Kaplan-Meier

**Files:**
- Modify: `src/sparsegf2/mipt.py` (`EnsembleResult`)
- Modify: `src/sparsegf2/analysis.py` (the `kaplan_meier` docstring and its error message for `-1`)
- Modify: `AGENTS.md`
- Modify: `docs/GETTING_STARTED.md`
- Modify: `examples/purification_time.py`
- Modify: `docs/LLM_INTERFACE.md` (if it carries the recipe)
- Modify: `CHANGELOG.md` (Unreleased)
- Test: `tests/test_mipt.py`

**Interfaces produced:**

```python
class EnsembleResult:
    @property
    def censoring_layer(self) -> np.ndarray: ...
        # int64, one per trajectory: the last *spec* checkpoint at which purification was tested,
        # i.e. the largest checkpoint_layers entry <= final_layer[t]. ValueError without a reference.
    def purification_times(self) -> tuple[np.ndarray, np.ndarray]: ...
        # (time, observed): time = tau where tau >= 0, else censoring_layer; observed = tau >= 0.
        # Ready for analysis.kaplan_meier(*result.purification_times()).
```

**Requirements (R2):**
- `final_layer` keeps its meaning (the last layer simulated).
- Every recipe that builds Kaplan-Meier input becomes `time, observed = result.purification_times()` followed by `analysis.kaplan_meier(time, observed)`. That covers AGENTS.md (the Pitfalls bullet), the `analysis.kaplan_meier` docstring, the error message raised when `-1` is passed (it should point to `purification_times()`), GETTING_STARTED, `examples/purification_time.py` and LLM_INTERFACE.
- In the example, a median beyond the data is reported as "> last checkpoint (<value>)", not "> layers".
- **Regression test (from `scripts/ensemble/n_km_censoring.py`):**
  - Use the same spec and seed with checkpoints `list(range(11))` and with `"every_layer"`, and `layers=40`.
  - The Kaplan-Meier survival from `purification_times()` of the first run equals the second run's exact `survival()` at layers 0..10.
  - Its event or censoring times never exceed 10.
  - The old recipe (`np.where(tau >= 0, tau, final_layer)`) gives a different (wrong) value at layer 20. Assert the inequality, to document the bug.
- **Unit tests:** `censoring_layer` with `stop_when_purified` (a purified trajectory's censoring layer is irrelevant but defined), with an explicit checkpoint list, and without a reference (raises).
- Add a CHANGELOG Unreleased bullet.

**Steps:**
- [ ] Write the failing tests. They fail because the attribute is missing.
- [ ] Implement them, and update the recipes.
- [ ] Run the full suite, including the doc tests from Task 11.
- [ ] Commit: `mipt: purification_times() censors at the last purification test, not the last layer simulated; recipes updated`

### Task 13: summaries condition P_C(m) on a code, and the other summary fixes

**Files:**
- Modify: `src/sparsegf2/api.py`
- Modify: `docs/LLM_INTERFACE.md`
- Modify: `AGENTS.md` (the summary sentences)
- Modify: `CHANGELOG.md`
- Test: `tests/test_api.py`, or a new `tests/test_api_summaries.py`

**Findings:** A1, A3, A10, A11, A15, A16 (`.superpowers/hardening/review/api.md`). Also the summary-field oracle tests that the audit found missing (`tests.md` A12 and A21, mutants `censored_tau_mean`, `tau_median_all_threshold` and `pc_sem_not_divided`).

**Requirements:**
- **A1 (R4): `erasure_first_failure`.**
  - At each shown evaluation, a trajectory "holds a code" if its smallest sample is <= n.
  - P_C(m) `mean` and `stderr` are taken over those trajectories. Add `count_with_code`, also in `vs_depth`.
  - The `what` line says "among trajectories that hold a code (k > 0); count_with_code says how many".
  - With no code at an evaluation, the mean is `null`.
- **A1 (R4): `uncorrectable_region_counts`.**
  - If the spec has a `code_dimension` probe evaluated at the same layers, condition on k > 0 in the same way, with `count_with_code`.
  - Otherwise, `what` and a note say that trajectories without a code count as P_C = 0, and suggest adding `code_dimension`.
- **A3:** `tau_median_all` only when strictly more than half purified within the horizon (`surviving[-1] < 0.5`). It is then the first layer with `surviving <= 0.5`.
- **A10:** no note is dropped when truncating. The truncation note is appended once.
- **A11:** automatic `#k` names in `run_circuit` are unique, also against user names that contain `#k`.
- **A15:** `truncated: true` and a note whenever an ensemble vector curve or a `vs_depth` grid is subsampled.
- **A16:** the default and the user-given `tick_stride` never evaluate more than `MAX_TICK_ROWS` ticks, counting the last tick.
- **Oracle tests against numpy on a saved run** (`output_dir`, then `EnsembleResult.load`):
  - every scalar `mean`, `stderr` and `count`;
  - the vector `stderr`;
  - `tau_mean_purified`;
  - `tau_quantiles_purified` (read the code for its quantile method and use the same method in numpy);
  - `tau_median_all` at a checkpoint where S = 1/2 exactly (even trajectory count);
  - the P_C(m) `stderr` of both erasure probes.

  Each of the three named mutants must make at least one of these tests fail. Check this by applying them temporarily, and report it.
- Add the review repros `e15`, `e06`, `e07`, `e11` and `e20` as small regression tests.
- Update the summary descriptions in LLM_INTERFACE and AGENTS.md. Add CHANGELOG bullets.

**Steps:**
- [ ] Write the failing tests (A1, A3, A10, A11, A15, A16), then the oracle tests (which may pass already).
- [ ] Implement.
- [ ] Run the mutant checks.
- [ ] Update the docs.
- [ ] Commit: `api: P_C(m) summaries condition on a code; tau_median_all needs a majority; notes survive truncation; unique names; truncation flagged; tick limit exact`

### Task 14: exact erasure counts and P_C(m) (E1, E2), and the e_delta junction

**Files:**
- Modify: `src/sparsegf2/analysis.py`
- Modify: `src/sparsegf2/_kernel/entropy.py` (the counts accumulation)
- Modify: `src/sparsegf2/codes.py` (docstring)
- Test: `tests/test_codes.py` and `tests/test_analysis_erasure_stats.py`
- Modify: `CHANGELOG.md`

**Requirements:**
- **E2:**
  - `failure_by_size` forms P_C(m) as `cumsum(integer bin counts) / samples`, with one division. `_histogram` returns counts, or callers divide once.
  - `failure_by_size(...)[..., n] == 1.0` exactly for codes with k > 0.
  - Regression (`scripts/erasure/t14_junction_dip_rounding.py`): on the [[10,4]] code with D histogram `[13, 21, 30, 36]`, `junction_dip` is False and `failure_by_size[n] == 1.0`.
  - Add a unit test for exact equality (Task 10's property comes later and also asserts it).
- **E1:**
  - The exhaustive counts are exact wherever an exact integer fits. Accumulate in int64 while `C(n, m) < 2**63`, using an int64 comb table where it fits, and float64 beyond. Document the choice in the kernel and in the `uncorrectable_counts` docstring.
  - `exact_failure_by_size` accepts float counts up to `C(n, m) * (1 + 1e-12)`, clips the ratio to [0, 1], and still rejects clearly wrong values.
  - Regression (`scripts/erasure/t06_float_counts.py`, `t06b_float_counts_engine.py`): the n=200, w=13 counts pass through `exact_failure_by_size`, `erasure_failure_probability` and `api.run_ensemble`.
- **The e_delta junction (audit mutant `tolerance_hybrid_junction`):**
  - `erasure_tolerance` must reuse the hybrid Phi of `erasure_failure_probability` instead of its own copy (`analysis.py` around lines 366-374), so there is one implementation.
  - Test: samples that contain D <= w, with nonzero `exact_counts`. `erasure_tolerance` agrees with the grid inversion of `erasure_failure_probability(..., exact_counts=...)`.
  - The mutant must fail this test. Check it temporarily against the old private copy, or by mutating the shared function, and report it.
  - Remove any `xfail` that Task 10 put on this.
- Add CHANGELOG bullets.

**Steps:**
- [ ] Write the failing tests.
- [ ] Implement.
- [ ] Run the full suite, including Task 10's properties with `HYPOTHESIS_PROFILE=deep` once.
- [ ] Commit: `analysis: P_C(m) from integer counts; exhaustive counts exact beyond 2**53 or tolerated; e_delta shares the hybrid estimator`

### Task 15: per-evaluation probe streams and randomness contract 2 (N4, kernel S1)

**Files:**
- Modify: `src/sparsegf2/_kernel/rng.py` (`stream_state`)
- Modify: `src/sparsegf2/_kernel/ensemble.py` (the sampled-probe streams)
- Modify: `src/sparsegf2/mipt.py` (`RANDOMNESS_CONTRACT = 2`, probe-name hashes, `EnsembleResult.randomness_contract`, checks)
- Modify: `src/sparsegf2/api.py` (record and check the contract in output dirs)
- Modify: `tests/_reference_rng.py`, `tests/_protocol_oracle.py`, `tests/test_randomness_contract.py`
- Test: `tests/test_replay.py`
- Modify: `AGENTS.md`, `docs/DESIGN.md` (sections 6.2 and 8.2, the `_kernel/ensemble.py` docstring), `CHANGELOG.md`

**Requirements (R3):**
- **`stream_state` mixes both key words.**
  - Replace `x = key0 ^ (index*C) ^ key1` with a mixing that depends on all 128 bits and on index. For example: run splitmix64 from `key0`, add `key1` to every step's increment, and xor index into the initial state through splitmix.
  - Any well-defined function in which distinct `(key0, key1, index)` give independent streams is acceptable. It must be documented exactly in the docstring, and transcribed in `_reference_rng.py` under contract 2.
  - It is used everywhere `stream_state` is used, including Simulator shot streams. `sample()` results with a fixed seed therefore change; the CHANGELOG says so.
- **Sampled probes.** Every evaluation of `erasure_first_failure` or `erasure_lost_profile` at layer L in trajectory t draws only from its own stream: `stream_state(key_probe, mix(first + t, name_hash, L))`.
  - `name_hash` is the first 8 bytes of `hashlib.blake2b(name.encode("utf-8"), digest_size=8)`, little-endian, as uint64. It is computed in Python and passed to the kernel per probe.
  - `mix` is a fixed 64-bit function (for example chained splitmix64), documented in the module docstring.
  - Samples therefore no longer depend on other probes, on the checkpoints, or on k at earlier evaluations.
- **The contract.** `RANDOMNESS_CONTRACT = 2`.
  - `EnsembleResult` gains `randomness_contract: int`, defaulting to the current contract. It is saved, and loads as 1 for files without it.
  - `EnsembleResult.concatenate` refuses mixed contracts.
  - The API records the contract in `provenance.json` and `spec.json` handling. Resuming or extending an output_dir written under another contract is refused: "this run was made under randomness contract 1; this version uses 2; start a new output_dir".
  - The `RANDOMNESS_CONTRACT` docstring states what it covers.
- **Tests:**
  1. `replay(t, probes=[erasure_first_failure(...)], checkpoints=[L])` reproduces the original run's samples at layer L exactly, where the original had other probes and checkpoints. The same holds for `erasure_lost_profile`. Use the repro from `scripts/ensemble/n_replay_samples.py`.
  2. Adding a second sampled probe does not change the first probe's samples.
  3. The first-failure law tests and the invariance test (Task 6) still pass.
  4. In `test_randomness_contract.py`, add `EXPECTED[2]`, computed from the updated reference and package. Keep `EXPECTED[1]` with a comment that it documents contract 1. The reference RNG implements both mixings, selected by contract.
  5. Save/load round-trips the contract. A file without the field loads as 1. `concatenate` refuses mixing. An output_dir written with contract 1 (craft one by editing `provenance.json`, or by monkeypatching) is refused on resume.
  6. `_protocol_oracle.py` is updated, and its equality test passes.
- Update the AGENTS.md replay bullet (sampled probes are reproduced too), DESIGN 8.2, and the CHANGELOG. The CHANGELOG gets a "Breaking" bullet: results saved before this version cannot be extended; shot streams of `sample()` changed.

**Steps:**
- [ ] Write the failing tests (1, 2, 5).
- [ ] Implement the rng change, the probe streams and the contract plumbing.
- [ ] Update the known answers and the oracle.
- [ ] Run the full suite, plus the lock-step and noise tests.
- [ ] Commit: `ensemble: sampled probes draw per evaluation, so replay reproduces them; stream keys use all 128 bits; randomness contract 2`

### Task 16: N2, derived specs re-derive their defaults

**Files:**
- Modify: `src/sparsegf2/mipt.py`
- Test: `tests/test_mipt.py`
- Modify: `CHANGELOG.md`

**Requirements:**
- **The bug.** `__post_init__` writes resolved defaults back into public fields: `gates_per_layer` (None becomes `n // 2`), `system_order` (None becomes `arange(n)`) and the default tripartite regions. `dataclasses.replace` then copies stale values. See `scripts/ensemble/n_replace_stale.py`.
- **The fix:** keep the user's input in the public field, None or empty for "default", and put resolved values in private `init=False` fields, the way `_checkpoint_layers` already works.
  - This includes probes whose size comes from a budget or a per-size mapping (Task 26). `replace(spec, num_system=m)` must resolve them again for m.
  - Add read-only properties for the resolved values where code or users need them. For example `resolved_gates_per_layer` and `resolved_system_order`. Choose names that fit `mipt`'s style and report them.
  - Update every internal reader: `_arrays`, `circuit`, `_protocol_difference`, the API's spec building and summaries, and `EnsembleResult.save`/`load`.
- **Saved files.** Files written before this change store the resolved values. They must still load and compare equal to a spec built with those explicit values.
- **Tests:**
  - `replace(spec, num_system=m, edges=new_edges)` equals a fresh spec with the same arguments: compare `_protocol_difference(...) is None` and the resolved values.
  - `replace(spec, system_order=reversed_order)` re-derives the half-cut and tripartite defaults.
  - An explicit `gates_per_layer` survives `replace`.
  - Old-format files load.
- Add a CHANGELOG bullet.

**Steps:**
- [ ] Write the failing tests.
- [ ] Implement.
- [ ] Run the full suite.
- [ ] Commit: `mipt: specs keep their inputs and resolve defaults privately, so dataclasses.replace re-derives them`

### Task 17: integers are integers, and edges are pairs (N3, A8, K4, E3)

**Files:**
- Modify: `src/sparsegf2/graphs.py` (one validator)
- Modify: `src/sparsegf2/mipt.py` (`_graph_list`)
- Modify: `src/sparsegf2/api.py` (edges)
- Modify: `src/sparsegf2/circuit.py` (`_as_target`)
- Modify: `src/sparsegf2/codes.py`
- Modify: `src/sparsegf2/observables.py`
- Tests: in the matching test files
- Modify: `CHANGELOG.md`

**Requirements:**
- **One validator, `graphs._edge_array(raw, what: str) -> np.ndarray`.** It is used by `from_edges`, `mipt._graph_list`, `stats`, `shortcut_fraction`, `graphs._checked` (and so `load`), and the API's `edges`.
  - Convert without a dtype first. Require `ndim == 2` and `shape[1] == 2`; `(0,)` and `(0, 2)` are allowed as empty.
  - A `(2, m)` array with `m != 2` raises `ValueError` with a hint to transpose.
  - Booleans and strings are refused. Whole floats are accepted; any other non-integral value is refused, the way `mipt._int_array` does it.
  - `_graph_list` treats only `ndim == 3` arrays or lists of graphs as collections. Every other shape is refused, never reshaped.
- **`Circuit.append` / `_as_target`:** use `operator.index`. A float target raises `TypeError`, as in Stim.
- **`codes.py`:** use `operator.index` for qubits (`_qubits`), `order` in `contiguous_distance`, the reference in `from_simulator`, and `samples` in `first_failure_sizes`. `_pauli_text` refuses a sign followed by `i` (`"+iXX"`, `"-iXX"`); a bare lowercase `ixx` keeps its documented reading.
- **`observables.py`:** use `operator.index` for `sizes` in `random_erasure_statistics`, and for `samples` in it and in `erasure_profile`.
- **Tests:** at every entry point above, the review repros (`n_reshape_edges.py`, `e02_validation.py` edges part, `repro_small_api.py` (b), `t12_inputs.py`) are refused with a message that names the argument. Valid inputs (numpy ints, whole floats where the rule allows them) still work.
- Add CHANGELOG bullets.

**Steps:**
- [ ] Write the failing tests.
- [ ] Implement.
- [ ] Run the full suite.
- [ ] Commit: `inputs: fractional indices and malformed edge arrays are errors everywhere, never truncated or re-paired`

### Task 18: graph files and statistics (N5, N6, ensemble S1)

**Files:**
- Modify: `src/sparsegf2/graphs.py`
- Modify: `src/sparsegf2/mipt.py` (the cumulative edge weights)
- Tests: `tests/test_graph_files.py`, `tests/test_graph_stats.py`, `tests/test_graph_models.py`
- Modify: `CHANGELOG.md`

**Requirements:**
- **N5:** `stats` computes `connected` first, and returns `lambda2 = 0.0` exactly when the graph is disconnected. Tighten Task 9's `< 1e-12` assertion to `== 0.0`.
- **N6:**
  - NPZ `ptr` must be 1-D integer, start at 0, be non-decreasing, and end at `len(edges)`.
  - JSON weights need one list per graph, and NPZ weights need `len(weights) == len(edges)`.
  - Violations raise `ValueError` naming the file and the field. Use the repro `n_graph_files.py`.
- **Ensemble S1:**
  - Build the cumulative weight table so that zero-weight edges can never be drawn: `cum` reaches exactly 1.0 at the last positive weight, and stays at 1.0 after it, while a draw `u` lies in [0, 1).
  - Test the table directly: weights `[1, 0, 0]` and `[0, 1, 0]`.
  - Test with many draws that a zero-weight edge never appears.
- Add CHANGELOG bullets.

**Steps:**
- [ ] Write the failing tests.
- [ ] Implement.
- [ ] Run the full suite.
- [ ] Commit: `graphs: lambda2 is 0 when disconnected; graph files are validated; zero-weight edges are never drawn`

### Task 19: resume and extend refuse a different run up front (A2, A9, A14, A12)

**Files:**
- Modify: `src/sparsegf2/api.py`
- Modify: `src/sparsegf2/cli.py`
- Modify: `docs/LLM_INTERFACE.md`
- Tests: `tests/test_api.py` and `tests/test_cli.py`
- Modify: `CHANGELOG.md`

**Requirements:**
- **A2:**
  - Before any computation on resume or extend, build the protocol from the given spec and compare it with the saved one (`mipt._protocol_difference`, or the saved `EnsembleResult.spec`). Refuse, naming the differing field.
  - For `{"family": "file"}`, `spec.json` records the resolved absolute path and the sha256 of the file's bytes. A changed file is refused up front.
  - If a merge still fails, the error names the chunk files that do not belong, so the earlier data can be recovered by deleting them.
  - Regression: `scripts/api/e04_resume_graph_file.py`.
- **A9:**
  - `load_results` returns the stored spec exactly, with no float rounding. Exempt it from `_bounded` rounding.
  - The "different spec" refusal names the first differing key path and both values.
  - Regression: `e08_load_results_spec_rounded.py`. The spec returned by `load_results` must be accepted by `start_ensemble_job` to extend the run.
- **A14:** resuming without `trajectories` resumes to the target stored in `spec.json`. Document this in LLM_INTERFACE and the tool description text in `mcp_server.py`, if it has that text.
- **A12:** the CLI passes `trajectories=None` through for any graph collection, including a file with several graphs. The API keeps its message for a single graph. Regression: `e05_cli.py` A12 part.
- Add CHANGELOG bullets.

**Steps:**
- [ ] Write the failing tests.
- [ ] Implement.
- [ ] Run the full suite.
- [ ] Commit: `api: resume and extend compare the protocol and graph file contents before computing; load_results returns the exact spec; resume needs no trajectories`

### Task 20: one writer per output directory, and merges that survive readers (A4, A5, api S3)

**Files:**
- Modify: `src/sparsegf2/api.py`
- Tests: `tests/test_api.py`
- Modify: `CHANGELOG.md`

**Requirements:**
- **A4, the lock.**
  - Writers hold `output_dir/.lock`, created with `O_CREAT | O_EXCL`, containing JSON `{"pid", "host", "started"}`. That covers background jobs, `run_ensemble` with `output_dir`, and the CLI.
  - A second writer is refused: "another process (pid P on host H, started T) is writing to DIR; wait for it to finish or cancel it".
  - A stale lock (same host, process not alive) is taken over. On POSIX, check liveness with `os.kill(pid, 0)`. On Windows, use `ctypes` `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` plus `GetExitCodeProcess`; no new dependencies.
  - The lock is released in a `finally`.
- **A4, overlapping chunks.** `_load_parts_once` tolerates them. Trajectories are deterministic, so a part's trajectories that are already covered are dropped instead of failing. Parts from different runs are still refused.
- **A5 and S3:**
  - `os.replace` and `unlink` during the merge are retried on `PermissionError`, with backoff of at most about 5 s in total.
  - If the replace still fails, the job ends `completed`, with a note: "merge pending: result.npz is in use; the chunk files hold the data, and the next load_results or start_ensemble_job merges them".
  - `_load_directory` retries `PermissionError` the same way.
- **Tests:**
  - A held lock from a live process (this process's pid in a hand-written lock file, with another job) is refused.
  - A lock from a dead pid is taken over.
  - Overlapping chunk files written by hand load correctly.
  - The merge retries: monkeypatch `os.replace` to raise `PermissionError` twice, then succeed. When it always raises, the job ends `completed` with the note, and a later load merges.
  - Keep the tests platform-neutral with monkeypatching.
- Add CHANGELOG bullets.

**Steps:**
- [ ] Write the failing tests.
- [ ] Implement.
- [ ] Run the full suite.
- [ ] Commit: `api: one writer per output directory; overlapping chunks merge; merges retry while a reader holds the file`

### Task 21: errors that say what to fix, and honest estimates (A6, A13, A7)

**Files:**
- Modify: `src/sparsegf2/api.py`
- Modify: `src/sparsegf2/cli.py`
- Modify: `src/sparsegf2/mcp_server.py`
- Tests: `tests/test_api.py`, `tests/test_cli.py`, `tests/test_mcp_server.py`
- Modify: `CHANGELOG.md`

**Requirements:**
- **A6, API:**
  - `isinstance(..., str)` checks before every lookup keyed by a user value: graph `family`, probe and observable `kind`, `capabilities` `section`, and the edge-weights family. Failures raise `ValueError` naming the field.
  - `OSError` from creating `output_dir` or writing files becomes `ValueError("output_dir '...' is not writable: ...")`.
  - `num_system` is bounded by `MAX_QUBITS` before any graph generation, also in `estimate_ensemble`.
- **A6, CLI:**
  - Check `isinstance(spec.get("graph"), dict)`.
  - `OSError` on user-given paths exits 2.
  - Regression: `e05_cli.py`.
- **A6, MCP (R5):** `mcp_server.call` converts any other exception into a `ToolError` that carries `f"{type(exc).__name__}: {exc}"`, plus "(internal error: please report it)" for exceptions that are not input errors. Regression: `e03_mcp_args.py` and `e16_mcp_oserror.py`.
- **A13:** MCP integer and boolean tool parameters are strict (pydantic `StrictInt`, `StrictBool`, or `Field(strict=True)`). `true` and `"3"` for `trajectories`, `seed` or `shots` are refused with a message.
- **API S1:** fresh seeds (`seed=None`) are drawn with 53 bits, not 63 (`secrets.randbits(53)`). JavaScript MCP clients, including Claude Code, parse JSON numbers as doubles and would round a larger seed, so the reported seed would not reproduce the run. Test that a fresh seed is below 2**53.
- **A7:**
  - Refit the `entropy_profile` and `erasure_lost_profile` cost models with an `n * n * words` term, measured on this machine up to n = 2048 with both references. Keep the documented "deliberately pessimistic" property with a margin of at least 1.5x.
  - Test: the estimate is >= the measured time at n = 512 for every probe kind, with one trajectory per thread. Keep it fast.
  - Report the n = 2048 numbers.
- Add CHANGELOG bullets.

**Steps:**
- [ ] Write the failing tests.
- [ ] Implement.
- [ ] Run the full suite.
- [ ] Commit: `api, mcp: every failure carries a message; strict MCP argument types; profile estimates stay pessimistic at large n`

### Task 22: the Stim cross-check is right on every noise-free circuit (K1, K2)

**Files:**
- Modify: `src/sparsegf2/crosscheck.py`
- Test: `tests/test_crosscheck.py`
- Modify: `AGENTS.md` (the crosscheck bullet)
- Modify: `CHANGELOG.md`

**Requirements:**
- Refuse measurement instructions with a nonzero flip probability, with a message that says why.
- **Resets of qubits that are not in an eigenstate of the reset basis** must compare correctly. Either:
  - force Stim's collapse onto SparseGF2's coin, predicted from the simulator's stream the way `tests/_lockstep.py` `reset` does; or
  - drive both simulators instruction by instruction.

  Choose one, and document the method in the module docstring.
- **Detectors and observables** are compared as raw parities of the identical records (`compile_m2d_converter(skip_reference_sample=True)`), not as events relative to two different reference samples.
- **`compare_stabilizers`** always compares the final state as well, reported as `"final"` in `first_mismatch["tick"]`. A circuit without `TICK` therefore compares something.
- **Tests:**
  - `M(0.5)` is refused.
  - `R` on a Bell pair matches over 20 seeds.
  - The `MY` detector circuit from `repro_crosscheck_detectors.py` matches over 20 seeds.
  - A circuit without `TICK` compares its final state, and a deliberately broken comparison is detected. For that, monkeypatch the SparseGF2 side to flip a sign.
  - A fuzz over 50 random noise-free circuits with resets on entangled qubits, MPP and feedback matches.
- Add CHANGELOG bullets.

**Steps:**
- [ ] Write the failing tests.
- [ ] Implement.
- [ ] Run the full suite.
- [ ] Commit: `crosscheck: refuses flip noise, forces reset collapses, compares parities and the final state`

### Task 23: small simulator fixes (K3, K5, K6, K7)

**Files:**
- Modify: `src/sparsegf2/_compile.py`
- Modify: `src/sparsegf2/simulator.py`
- Modify: `src/sparsegf2/result.py`
- Modify: `src/sparsegf2/_kernel/program.py` (docstring)
- Tests: `tests/test_run.py` and `tests/test_simulator.py`
- Modify: `CHANGELOG.md`

**Requirements:**
- **K3:** the `max_ops` unrolling guard counts every flattened instruction, including `DETECTOR`, `OBSERVABLE_INCLUDE`, `SHIFT_COORDS` and `QUBIT_COORDS`. `REPEAT 1000000000000 { DETECTOR rec[-1] }` (after one `M 0`) raises the documented error within a second.
- **K5:** the `Simulator.seed` docstring says to pass the `SeedSequence` back as `seed=` (or its `entropy` and `spawn_key`). Test that `Simulator(n, seed=sim.seed)` reproduces a spawned-seed simulator.
- **K6:** the `coins` docstring of `Simulator.run` and the `program.py` module docstring say "the outcome to collapse to (the value before measurement-flip noise)".
- **K7:**
  - `RunResult.detectors` and `RunResult.observables` document that events are relative to a noiseless run of the circuit from |0...0>.
  - `Simulator` tracks whether it is still in its initial state. `run()` from any other state, on a circuit with detectors or observables, emits a `UserWarning` once per run that says so.
  - Test that the warning fires, using `pytest.warns`, since warnings are errors in the suite. Test that a fresh simulator does not warn.
- Add CHANGELOG bullets.

**Steps:**
- [ ] Write the failing tests.
- [ ] Implement.
- [ ] Run the full suite.
- [ ] Commit: `simulator: the unrolling guard counts annotations; detection events warn when a run did not start from |0...0>; docstrings corrected`

---

# Phase 2b: research tooling (requested by the owner on 2026-10-14)

**Execution order (ruling R-e in the ledger):**
1. After Task 5: 25, 6, 7, 14, 15, 26, 27, 28.
2. Then 12, 13, 29, 30, 31.
3. Then 8, 9, 10, 11, 16-23 and 24.

Consequences:
- Task 10 asserts P_C(n) == 1.0 exactly, because Task 14 lands first.
- Task 16's "resolve defaults privately" covers the budget and per-size probe forms of Task 26.
- The observable docs (Tasks 28-31) are in the doc tests (`tests/test_docs.py`).

### Task 25: multithreading works, and how it scales

**Files:**
- Create: `benchmarks/thread_scaling.py`
- Create: `benchmarks/results/thread_scaling.json` and `benchmarks/results/thread_scaling.md`
- Modify: `benchmarks/README.md` (a table row and the command)
- Modify: `AGENTS.md` (a short "Threads" bullet under Conventions)
- Modify: `docs/GETTING_STARTED.md` (a short "Threads" subsection)
- Test: `tests/test_benchmarks.py`, a smoke test of `--quick`

**Requirements:**
- **Record the environment:** CPU model, physical and logical cores, `numba.config.NUMBA_NUM_THREADS`, and `numba.threading_layer()` (read after a parallel call). Say whether that layer is safe for concurrent use from several Python threads, as MCP background jobs do (the Numba docs say `tbb` and `omp` are, and `workqueue` is not).
- **Workloads,** each at thread counts 1, 2, 4, 8, 16 and `NUMBA_NUM_THREADS`:
  - W1: Watts-Strogatz n=64, k=4, beta=0.1; purification; p=0.1; 256 layers; `random_edges`. Probes `code_dimension` every 32 layers, `erasure_first_failure(samples=512)` final, and `uncorrectable_region_counts(max_size=2)` final.
  - W2: the same at n=256.
  - W3: the purification-time workload. Cycle n=128, single reference, p=0.1, `layers=8*n`, `stop_when_purified`, `code_dimension` every layer.
  - W4: `Simulator.sample` (or the documented shot API) of a small noisy repetition-code memory circuit, 200 000 shots.
- **Sizing and measurement:** choose trajectory and shot counts so that the largest thread count takes at least about 5 s. Report wall time as the median of 3 runs (after a warm-up compile), throughput, speedup over 1 thread, and parallel efficiency.
- **Identity across thread counts:** check that results are identical at every thread count (arrays equal; samples equal for a fixed seed). Fail loudly if not.
- **Options:** `--quick` (tiny sizes, under about 30 s including compilation) and `--out <dir>`. By default results go to `benchmarks/results/`.
- **Run it on this machine** and commit both result files. In the report, state whether 32 threads beat 16 (SMT) for each workload. In AGENTS.md and GETTING_STARTED, recommend a `num_threads` / `NUMBA_NUM_THREADS` setting based on these measurements, and say that every result is identical at any thread count.

**Steps:**
- [ ] Write the script and the smoke test.
- [ ] Run the full benchmark on an otherwise idle machine; no other test run may be going on at the same time.
- [ ] Commit the script, the results and the docs: `benchmarks: thread scaling of ensembles and sampling; results identical at every thread count`

### Task 26: the exhaustive cutoff m*(n), set by a budget or per system size

**Files:**
- Modify: `src/sparsegf2/analysis.py`
- Modify: `src/sparsegf2/mipt.py` (`uncorrectable_region_counts`, probe resolution)
- Modify: `src/sparsegf2/api.py` (JSON probe forms, estimate)
- Modify: `AGENTS.md`, `docs/LLM_INTERFACE.md`, `CHANGELOG.md`
- Tests: `tests/test_analysis_erasure_stats.py`, `tests/test_mipt.py`, `tests/test_api.py`

**Interfaces produced:**

```python
# analysis.py
def exhaustive_region_count(num_qubits: int, max_size: int) -> int: ...
    # N_exact(n, w) = sum_{m=1}^{w} C(n, m), exact Python int
def exhaustive_cutoff(num_qubits: int, budget: int) -> int: ...
    # m*(n) = max{w in 0..n : N_exact(n, w) <= budget}
# mipt.py
def uncorrectable_region_counts(
    max_size: int | Mapping[int, int] | None = None,
    name: str | None = None,
    *,
    budget: int | Mapping[int, int] | None = None,
    checkpoints=None,
) -> Probe: ...
```

**Requirements:**
- **Definitions** (the owner's): the exhaustive regime is 0 <= m <= m*(n) (every m-subset is enumerated), and the sampled regime is m*(n) < m <= n. The budget form is m*(n) = max{w : sum_{m=1}^{w} C(n, m) <= B}.
- **Arguments:**
  - Give exactly one of `max_size` and `budget`. With neither, `max_size=2` (the current default). With both, raise `ValueError`.
  - Either may be an int, or a mapping from system size n to a value, so one probe definition can serve a sweep over n: `max_size={64: 3, 128: 2, 256: 2}` or `budget={64: 20_000, 256: 50_000}`.
  - Resolution happens when a `MonitoredCircuit` resolves its probes, for its own `num_system`.
  - A mapping without an entry for n raises `ValueError` that lists the sizes given.
  - A resolved w must satisfy 1 <= w <= n. A budget that allows no size raises `ValueError`: "budget B allows no exhaustive region size at n = N (C(N, 1) = N > B); raise the budget or drop the probe".
- **Naming and storage:** the default probe name stays `uncorrectable_region_counts[w]`, with the resolved w. Document passing `name=` to keep one key across a sweep over n. The resolved w is what the engine uses and what saved results store. Saving a result and loading it back gives the same key and values.
- **JSON (API, CLI, MCP):**
  - `{"kind": "uncorrectable_region_counts", "budget": 50000}`;
  - `{"kind": "uncorrectable_region_counts", "max_size": {"64": 3, "128": 2}}`;
  - `{"kind": "uncorrectable_region_counts", "budget": {"64": 20000, "256": 50000}}`.

  JSON object keys are strings, so accept integer-valued keys and validate them. Validation errors name the field. `estimate_ensemble` uses the resolved w. `capabilities` documents the forms.
- **Tests:**
  - The owner's examples: `exhaustive_cutoff(n, 50_000)` is 3, 2, 2 and 1 for n = 64, 128, 256 and 512, and `exhaustive_region_count(256, 2) == 32_896`.
  - Edge cases: budget 0 gives 0; a budget >= 2**n - 1 gives n.
  - A run with `budget=B` equals a run with `max_size=exhaustive_cutoff(n, B)`: same key, same values.
  - Mapping forms in Python and JSON; the errors for a missing n, a budget that is too small, both arguments, and non-integer keys or values.
  - A save/load round trip.
- **Docs:** update AGENTS.md (the probe line and the erasure section), LLM_INTERFACE.md and the CHANGELOG.

**Steps:**
- [ ] Write the failing tests.
- [ ] Implement.
- [ ] Run the full suite, ruff and mypy.
- [ ] Commit: `erasure: the exhaustive cutoff m*(n) can be set per system size or by an enumeration budget`

### Task 27: the cost of exhaustive enumeration and of sampling, measured

**Files:**
- Create: `benchmarks/erasure_cost.py`
- Create: `benchmarks/results/erasure_cost.json` and `benchmarks/results/erasure_cost.md`
- Modify: `benchmarks/README.md`
- Test: `tests/test_benchmarks.py` (a `--quick` smoke test)

**Requirements:**
- **Codes:** take them from real trajectories. Use n in {32, 64, 128, 256, 512}; Watts-Strogatz k=4, beta=0.1; purification; `random_edges`; p in {0.05, 0.1}; final layer 4n. Take `code_snapshot` at the final layer, about 8 codes per (n, p) with k > 0, and record their k and their distance bounds.
- **What to measure:**
  - The one-thread wall time of the exhaustive counts for w = 1, 2, ... until one evaluation exceeds 10 s or N_exact exceeds 1e9. Measure both the engine probe path (a run whose only probe is the counts at the final layer, minus the same run without it, per trajectory) and `StabilizerCode.uncorrectable_counts(w)`. Report both if they differ.
  - The time per first-failure ordering, for M in {100, 512, 2048}.
- **What to report:**
  - for each (n, p, w): the median and max seconds per code, N_exact(n, w), and the seconds per region test;
  - for budgets B in {1e4, 5e4, 1e5, 1e6, 1e7}: the resulting m*(n) and the expected seconds per code;
  - the expected total for a study: per-code time × trajectories × checkpoints ÷ the effective threads from Task 25.
- **Options:** `--quick` and `--out`. Run it on this machine and commit the results. The erasure README (Task 28) cites them.

**Steps:**
- [ ] Write the script and the smoke test.
- [ ] Run it on an otherwise idle machine.
- [ ] Commit: `benchmarks: the cost of exhaustive erasure enumeration and of sampled orderings, by n and budget`

### Tasks 28-31: one detailed document per observable

**Files:**
- Create: `docs/observables/README.md` (the index: observable, definition, probe/observable/function names, file)
- Create: one file per observable family (listed per task below)
- Modify: `tests/test_docs.py` (run their Python blocks)
- Modify: `README.md` and `AGENTS.md` (link the index)

**Each document must contain, in order:**
1. **Definition.** The quantity in equations (GitHub math, `$...$` and `$$...$$`) and the setting: the purification picture, the reference R, the [[n, k_C]] code.
2. **The formula the simulator evaluates, and why it equals the definition.** A short derivation or proof sketch. Examples:
   - S(A) = rank(G_A) - |A| for a pure stabilizer state;
   - formula (1), $\ell_C(A) = 2|A| - s - \operatorname{rank}(H_A) + \operatorname{rank}(H_{\bar A})$, via the cleaning lemma;
   - the prefix of a uniformly random ordering is a uniformly random m-subset, and I(A:R) is monotone under inclusion (data processing), so D <= m exactly when ell(A_m) > 0, and Pr[D <= m] = P_C(m) for every m at once.
3. **From the kernel to the code you run.**
   - The data layout (the tableau's bitset columns and rows, the reference rows).
   - The kernel routines, by file and function name, with what each computes and its cost.
   - How the compiled ensemble engine dispatches the probe at checkpoints.
   - The Python layers: `mipt` probes, `observables`, `StabilizerCode`, `analysis`.
   - The JSON/MCP forms.
   - Every statement about the code must match the source; cite `path::function`.
4. **Cost and accuracy:** complexity, measured costs (Task 27), statistical error bars, and the exact and sampled regimes.
5. **Worked examples on single trajectories.** These are runnable and run by the doc tests: small n, a few seconds.
   - Run one trajectory, then compute the observable through the engine probe, through the library on the replayed state, and through an independent brute force written in the example.
   - Show that they agree.
   - Show printed numbers only where they are deterministic (contract-independent): exact counts, k, entropies.
6. **Pitfalls.**

### Task 28: random-region erasure, lost information and the code dimension

**Files:**
- Create: `docs/observables/random-region-erasure.md`
- Create: `docs/observables/lost-information.md`
- Create: `docs/observables/code-dimension.md`
- Create: `docs/observables/README.md`

**Requirements** (beyond the common list above):
- **`random-region-erasure.md`** is the most detailed. It covers:
  - P_C(m) as a probability over uniform m-subsets (the owner's definition). It is not an independent per-qubit erasure channel. Explain that Phi(e) mixes P_C(m) with binomial weights, and how that relates to the per-qubit channel.
  - The two regimes and the cutoff m*(n), given directly or by a budget (Task 26), with the owner's table for n = 256 and the budget examples.
  - The exhaustive regime: the counts kernel's enumeration (its walk, its pruning, why it is exact, its integer accumulation after Task 14).
  - The sampled regime: uniform random orderings, so each region is drawn without replacement and the M regions at a given m are independent and identically distributed. The nested regions give every m at once. The estimates at different m are positively correlated, each is unbiased with a binomial error bar, and the curve is monotone in m. Cover the per-evaluation streams after Task 15, which make replay exact.
  - The hybrid estimator; `junction_dip`; Phi(e); e_delta; mu_C; the distance bounds; conditioning on k > 0 (and the summaries after Task 13).
  - Measured costs from Task 27, and how to choose a budget.
  - A single-trajectory example. Take a Watts-Strogatz n = 16 trajectory. Its exact P_C(m) for every m comes from brute force over all subsets with formula (1), written with numpy in the example. Compare that with `uncorrectable_region_counts`, with the first-failure estimate (M = 2000), and with the hybrid curve.
- **`lost-information.md`:** ell_C(A) = I(A:R), correctability, formula (1), complementarity (ell(A) + ell(complement A) = 2k), `lost_information` (fixed region), `erasure_lost_profile`, and `StabilizerCode.lost_information` / `is_correctable`.
- **`code-dimension.md`:** k = S(R), the reference-row invariant (k is a popcount), the purification and single pictures, `code_dimension`, and the link to purification (`purification.md`, Task 30).

**Steps:**
- [ ] Write the documents and run their examples.
- [ ] Add them to the doc tests.
- [ ] Run the full suite.
- [ ] Commit: `docs: random-region erasure, lost information and the code dimension, from definition to kernel to examples`

### Task 29: entanglement observables

**Files:**
- Create: `docs/observables/entanglement.md`

**Requirements:**
- Cover S(A), the half-cut entropy, I(A:B), I3(A:B:C) and the entropy profile.
- Give the GF(2) formulas, the kernel rank routines (the complement path, the single-qubit fast path, `prefix_entropies`), phased vs phase-free, and the NaN after `stop_when_purified`.
- Give single-trajectory examples checked against a state vector for n <= 8.

**Commit:** `docs: entanglement observables, from definition to kernel to examples`

### Task 30: purification and contiguous distance

**Files:**
- Create: `docs/observables/purification.md`
- Create: `docs/observables/contiguous-distance.md`

**Requirements:**
- **`purification.md`:** tau, checkpoints, censoring, `purification_times()` and Kaplan-Meier (after Task 12), `survival()`, and the summaries (after Task 13).
- **`contiguous-distance.md`:** the definition along an order, periodic and open boundaries, the kernel window scan, NaN without a code, and a brute-force example.

**Commit:** `docs: purification time and contiguous distance, from definition to kernel to examples`

### Task 31: codes after the run

**Files:**
- Create: `docs/observables/codes.md`

**Requirements:**
- Cover `code_snapshot`, the packed generator format, `EnsembleResult.code`, and `StabilizerCode` (`from_simulator`, `from_packed`, `from_strings`, every method).
- Cover re-analyzing a saved code with more orderings, replay to get a code at a new depth, and saving and loading.

**Commit:** `docs: codes after the run, from snapshot to StabilizerCode`

### Task 32: compiled kernels are never stale

**Files:**
- Modify: `src/sparsegf2/_kernel/__init__.py`, or create `src/sparsegf2/_kernel/_cache.py` imported first by it
- Create: `tests/test_kernel_cache.py`
- Modify: `AGENTS.md` (Pitfalls)
- Modify: `docs/DESIGN.md` (section 5, a short note)
- Modify: `CHANGELOG.md` (Unreleased)

**The problem** (found in Task 15): Numba's `cache=True` keys each function's cache on its own source file's timestamp and size. A jitted caller in file A that calls a jitted callee in file B compiles B's code into A's cache entry. Editing B does not invalidate A's cache, so A keeps running the old callee. With JIT caches in the editable checkout's `__pycache__` (shared by the owner's SmallWorldCodes environment), a later kernel edit could silently run stale code in production. Task 15 confirmed it: `sample()` ran the contract-1 `stream_state` from a warm cache.

**Requirements:**
- **On import of `sparsegf2._kernel`,** before any kernel submodule is imported, compute a fingerprint: sha256 over the numba version, the Python version, and the bytes of every package module that defines or calls jitted code. Determine the set statically: all of `_kernel/*.py`, plus any other module with `@njit`/`numba.njit` (grep; `_parallel.py` uses numba but may not define kernels).
- **Compare** it with a marker file in the directory where Numba actually writes this package's cache. That is `<module dir>/__pycache__` by default, or the layout under `NUMBA_CACHE_DIR` when that is set. Find the real location through numba's cache locator logic (for example, by asking a jitted dispatcher's cache for its cache path), not by guessing.
- **On mismatch,** delete this package's Numba cache files (`*.nbi` and `*.nbc` of the fingerprinted modules) in that directory, then write the marker atomically.
- **Robustness:** a read-only cache dir and concurrent processes are tolerated (ignore missing files; never crash the import). Add one `logging` debug line, not a warning, since warnings are errors in the suite.
- **The end-to-end test** (it must fail without the fix):
  1. Copy `src/sparsegf2` to a temp dir, and use a fresh `NUMBA_CACHE_DIR` per test.
  2. In a subprocess with `PYTHONPATH` set to the copy, run a shot `sample()` with a fixed seed and record the result.
  3. Change a constant inside `_kernel/rng.py::stream_state` in the copy, keeping the file's other content, and run again in a new subprocess.
  4. The result must change.
  5. Then restore the file and check that the original result comes back.

  Show that the test fails with the guard disabled (temporarily, in the report). Keep it under about 60 s.
- **Docs:** add an AGENTS.md pitfall line saying kernels are recompiled automatically when any kernel source changes, and that a first import after an update is slower. Add a DESIGN 5 note and a CHANGELOG bullet.

**Steps:**
- [ ] Write the failing end-to-end test.
- [ ] Implement.
- [ ] Run the full suite, ruff and mypy.
- [ ] Commit: `kernel: caches are cleared when any kernel source changes, so callers never run a stale callee`

---

# Phase 3: documentation, version, and the mutation re-run

### Task 24: documentation, version and the mutation re-run

**Files:**
- Modify: `CHANGELOG.md` (the Unreleased section becomes `## 3.0.0.dev12: hardening`, with an opening paragraph and the Breaking bullets first)
- Modify: `src/sparsegf2/_version.py` (`3.0.0.dev12`)
- Modify: `README.md` (status line)
- Modify: `AGENTS.md`:
  - "Testing changes": the gates, mypy, pre-commit, the weekly workflow and the no-skip switch;
  - randomness contract 2;
  - `purification_times`;
  - the conditioned P_C(m) summaries.

  Check every section against the code.
- Modify: `docs/DESIGN.md`:
  - section 10 becomes the testing strategy as it now exists (oracles, gates, the weekly workflow, the known-answer contract tests), replacing the plans;
  - section 11 milestone rows;
  - section 8.2 sampled-probe streams.
- Modify: `docs/LLM_INTERFACE.md` (consistency pass)
- Modify: `docs/specs/2026-10-14-hardening-design.md` (Status: done, with the date)

**Requirements:**
- **The mutation re-run.**
  - Adapt `.superpowers/hardening/review/scripts/mutate.py` to the current tree, since lines moved. Apply each of the audit's 15 mutants in a scratch copy outside the worktree (under the session scratchpad or `%TEMP%`), with its own `NUMBA_CACHE_DIR`, and run the full suite of the copy.
  - All 15 must be caught. Put the table (mutant, caught by) in your report. If one survives, add the missing test, run it again, and report it.
- Keep `docs/plans/2026-10-14-hardening.md` unchanged.

**Steps:**
- [ ] Update the docs and the version.
- [ ] Run the mutants.
- [ ] Run the full suite, ruff and mypy.
- [ ] Commit: `docs: hardening guarantees, randomness contract 2 and the new recipes; 3.0.0.dev12`
