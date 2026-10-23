# Config-driven studies: design

**Owner's request (2026-10-23):** "I would like us to further develop the system that we have here with the config json
file and the way that we are running these simulations because I think that this is probably the best way for running
these simulations possible. Furthermore, we should make sure we are generalizing this as much as possible and updating
all of this into the SparseGF2 simulator and pushing consistently to github. I want you to go ahead and work on this
way of launching simulations and studies as a major part of the SparseGF2 simulator."

The system in question is the survey framework of SmallWorldMIPT (`studies/2026-10-22-ws-survey`: about 8,800 lines,
built 2026-10-22/23): one JSON file says what to simulate, record, draw and analyze; `plan`, `run`, `plot`, `analyze`
and `all` do it, resume after any interruption, grow a study without redoing finished work, and refuse data made by a
different simulator.

**Owner's rulings on scope (2026-10-23, asked because they reverse the 2026-10-17 rule that SparseGF2 does simulation
and data only):** all three layers go into SparseGF2 (launching in the core; figures and analyses as an optional
layer), and the study system supersedes `sparsegf2 sweep`.

**Owner's ruling on the extent (2026-10-23, later):** "if this is going to take a lot of time, let's put this off and
make sure we are only designing it based on what we have built for the SmallWorldMIPT study we have been conducting so
far." So the port is deferred, and its scope is exactly what the survey framework does today (R3, R4, R8 below); any
generalization beyond it waits until the owner asks for it.

**Status:** design only (2026-10-23, by Claude); deferred. When resumed, built on branch `studies` (worktree
`../SparseGF2-studies`), pushed after each task, merged into `main` only when the owner says.

## 1. What a study is

- **A config** (JSON) names: the sizes `n`; the graph family and its points (for Watts-Strogatz, the rewiring
  probabilities `beta`); the measurement rates `p` per `(n, point)`; the depth law `T(n)`; the circuit (a
  `MonitoredCircuit` protocol); what is recorded at every layer; the seeds; the sampling design; and lists of plots and
  analyses. Every key has a default except the study name, the points, the sizes and the rates; an unknown key is an
  error that names the closest valid key.
- **A condition** is `(n, point, p)`. **A trajectory** is `(condition, g, c)`: graph `g` of the point at size `n` and
  circuit `c` on it. A trajectory depends only on `(condition, g, c, seeds, engine)`: never on the depth asked for,
  on what else is simulated, on the order, or on the number of threads.
- **The data folder** holds, per condition, segments of trajectories over layer ranges `[t0, t1]`, each with its rows,
  their first passages of `k` and the change-point runs of every other recorded observable; the graphs drawn; the
  resolved plan; and the engine record. A figure or analysis is redrawn only when its inputs (settings, code, data)
  changed.
- **Work** per condition is computed by comparing the config with the data: `new` (nothing stored), `rows` (more
  graphs or circuits), `extend` (a larger depth: only the trajectories still holding a code are continued, and the
  rerun is checked against the stored layers), `rerecord` (a new observable). A smaller depth, fewer graphs or fewer
  rates need no simulation.

## 2. Rulings

- **R1 (place).** A package `sparsegf2.study` grows new modules beside the existing `identity` and `sweep`:
  `options` (the option machinery: one declaration per key gives validation, defaults, documentation and the
  `options` command), `config`, `laws` (graph laws), `plan`, `store`, `engine`, `run`, `progress`, and, in the
  optional layer, `plots` (with kinds), `analyses` (with kinds) and `fss` (collapse fits). Command line:
  `sparsegf2 study plan|run|plot|analyze|all|verify|status|config|options`. Why: one name for one concept, and the
  existing seeding code (`identity`) is reused.
- **R2 (no engine change).** No edit to `mipt.py`, `graphs.py` or `_kernel/`: they are fingerprinted in every stored
  trajectory's engine record and in the randomness contract (now 2). New graph laws (the Watts-Strogatz bank's
  vertex-major law) live in `study/laws.py`. Why: existing data folders (sweeps, the survey, the bank) stay valid.
- **R3 (scope).** Exactly the survey's: Watts-Strogatz graphs drawn by the bank's laws (vertex-major, and
  offset-major for comparisons); points are rewiring probabilities `beta` (lists, theta grids, irrational anchors);
  rates as lists or decimal ranges per `(n, beta)`; depth laws `c x {1, log2 n, sqrt n, n, n^2}`; the circuit of the
  survey (purification picture, `random_edges` with `n/2` gates per layer, periodic, every layer a checkpoint, stopped
  at purification); recorded observables `k` (first passages) and `d_cont` (change-point runs). Other graph families
  and other observables are out of scope until the owner asks.
- **R4 (sampling).** The survey's two modes: direct (`G` graphs per `(n, beta)`) and the bank's mixture (fixed-m
  strata with exact reweighting, quotas, coverage), both as they are.
- **R5 (optional layer).** Plots and analyses import matplotlib (and CMasher for its colormaps) only when used; they
  are declared as kinds in registries (`heatmap`, `condition`, `curves`; `onset`), each with its options declared once.
  An install group `sparsegf2[study]` brings their dependencies. Why: the core stays lean and the layer is extensible.
- **R6 (analyses).** The survey's analysis kind `onset` as it is (Kaplan-Meier medians of the first depth with
  `d_cont >= 2`, the fit of `tau_0` applied to it with its landscape, every version of the collapse, crossings, the
  front extrapolations, the depth collapse), with its README, `summary.json` and `report.pdf`.
- **R7 (fingerprints).** The engine record of a data folder and the code fingerprint of each figure and analysis are
  taken when the code is loaded, so a long-lived process never stamps results with the fingerprint of files edited
  after it started (a bug found in the survey on 2026-10-23).
- **R8 (commands).** The survey's commands, as `sparsegf2 study plan|run|plot|analyze|all|verify|config|options`,
  with the same progress displays and `RUN.md`-style copy-and-paste guide. MCP tools are out of scope until asked.
- **R9 (supersede sweep).** `sparsegf2 sweep` keeps reading and growing its existing directories; the docs mark it as
  the older path; `sparsegf2 study adopt` can later convert a sweep or a survey data folder after checking a sample of
  its trajectories bit for bit.
- **R10 (compatibility with the survey).** For the same config, the study system reproduces the survey's
  trajectories (first passages and `d_cont` runs) bit for bit; a test checks this against a fixture recorded from the
  survey. The survey's data folders are adopted (R9) rather than read in place.
- **R11 (presentation).** The plot options keep the survey's: every color, colormap, legend, label, font size and the
  Physical Review style (`"style": "prl"`); text is typeset with LaTeX when it is on the PATH.
- **R12 (checks).** Every module is typed (mypy reports Success), formatted (ruff), spelled in American English, and
  tested (an end-to-end test runs a tiny study through plan, run, plot and analyze); the coverage floor holds.

## 3. Phases (when resumed)

1. Core, ported as is: options, config, laws, points, plan, store, engine, simulate (direct and mixture), progress,
   verify; `sparsegf2 study plan|run|verify|config|options`.
2. Figures: the survey's plot kinds (`heatmap`, `condition`, `curves`) and options; `study plot`.
3. Analyses: the survey's `onset` kind; `study analyze`, `study all`.
4. `study adopt` for the survey's data folders (R9, R10); sweep marked as the older path in the docs.

Estimated effort: several long sessions (about 8,800 lines to port to the repository's checks: typing, formatting,
tests, documentation).
