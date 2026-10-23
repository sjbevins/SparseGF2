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

**Status:** written 2026-10-23 by Claude. Built on branch `studies` (worktree `../SparseGF2-studies`), pushed after
each task; merged into `main` only when the owner says.

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
- **R3 (generality).** The axes are sizes x points x rates. The graph comes from a law: any family of
  `sparsegf2.graphs` (with its parameters) or a registered law (`laws.py`); the points are values of the law's main
  parameter. The circuit is any `MonitoredCircuit` protocol the config can describe (schedule, gates per layer,
  picture, periodicity, orientation, stopping at purification). Recorded observables: `k` (always; its first
  passages store it exactly) and any scalar per-layer probe (stored as change-point runs: `d_cont` first, then
  `code_dimension`-like probes as they are needed). Why: the survey's Watts-Strogatz choices become defaults, not
  assumptions.
- **R4 (sampling).** Direct sampling first: `G` graphs per `(n, point)`, `C` circuits per graph. The bank's mixture
  design (fixed-rewiring-count strata with exact reweighting, quotas, coverage) is a second sampling mode, ported after
  direct sampling works. Why: direct sampling is general; the mixture is Watts-Strogatz specific and larger.
- **R5 (optional layer).** Plots and analyses import matplotlib (and CMasher for its colormaps) only when used; they
  are declared as kinds in registries (`heatmap`, `condition`, `curves`; `onset`), each with its options declared once.
  An install group `sparsegf2[study]` brings their dependencies. Why: the core stays lean and the layer is extensible.
- **R6 (analyses).** The scaling machinery is general: the loss of Houdayer and Hartmann with linear parameters in
  closed form (prefactor exponents, corrections), loss landscapes on a grid refined by Nelder-Mead and confined to the
  rates fitted, bootstrap refits, systematics, crossings and their drift, extrapolations in `n` with model comparison.
  The `onset` kind (the first depth at which an observable passes a threshold, such as `d_cont >= 2`) is its first
  user; `tau_l` and `<T*>` follow. Each analysis writes numbered figure folders, a README with every table,
  `summary.json` and a printable `report.pdf`.
- **R7 (fingerprints).** The engine record of a data folder and the code fingerprint of each figure and analysis are
  taken when the code is loaded, so a long-lived process never stamps results with the fingerprint of files edited
  after it started (a bug found in the survey on 2026-10-23).
- **R8 (agents).** Every command has `--json`; the MCP server gets `study_plan`, `study_start` (a background job),
  `study_status`, `study_plot`, `study_analyze` and `study_results`, sharing `api.py` as the other tools do.
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

## 3. Phases

1. Core: options, config, laws, plan, store, engine, run (direct sampling), progress, verify; `sparsegf2 study
   plan|run|status|verify|config|options`; docs `docs/STUDIES.md`.
2. Figures: plot kinds and options; `study plot`.
3. Analyses: `fss` and the `onset` kind; `study analyze`, `study all`.
4. Agents: `api.py` functions, MCP tools, AGENTS.md.
5. Mixture sampling; `study adopt` for survey and sweep folders; sweep marked as the older path.
