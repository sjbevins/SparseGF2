# The study's service layer, command line, MCP tools and diagnostic figures

How a person, a script or an LLM agent drives the Watts–Strogatz study: one JSON service layer
(`src/sparsegf2/study/api.py`) that the `sparsegf2 study` commands (`src/sparsegf2/cli.py`) and
the MCP `study_*` tools (`src/sparsegf2/mcp_server.py`) both call, so arguments, results and
error messages are the same everywhere; and the diagnostic figures
(`src/sparsegf2/study/plots.py`). Statements about the code cite `file::name`.

The scientific specification is `docs/misc/watts_strogatz_study_spec_v1_4.md` ("v1.4 §x"), the
design `docs/specs/2026-10-17-ws-study-pipeline-design.md` ("design §x"). The earlier documents
of this folder define what this layer reports: the manifest (doc 02), the store and runner
(doc 05), the observables (doc 04), the statistics (doc 06) and the scaling fits (doc 07). The
Python blocks run as written, in order, in one namespace, in a few seconds.

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Notation](#2-notation)
3. [Definitions, as implemented](#3-definitions-as-implemented)
4. [Propositions](#4-propositions)
5. [Algorithms](#5-algorithms)
6. [From the mathematics to the code](#6-from-the-mathematics-to-the-code)
7. [Worked example: a tiny study end to end](#7-worked-example-a-tiny-study-end-to-end)
8. [Validation map](#8-validation-map)
9. [Limitations and departures](#9-limitations-and-departures)

## 1. Purpose and scope

The parts below this layer store trajectories (doc 05), define observables (doc 04) and
estimate ensemble quantities (docs 06, 07). This part makes them usable from three front ends
without three implementations:

* **The JSON service layer** (design §8, gap G-CLI-1): fourteen functions that take plain JSON
  values and return plain JSON dicts of at most `sparsegf2.api.MAX_OUTPUT_CHARS` = 8000
  characters, as the rest of `sparsegf2.api` does, with every input error raised as
  `ValueError` carrying a message that says what to change. An agent with a context of about
  45,000 tokens reads any result whole; long tables go to files when an `out` path is given.
* **`sparsegf2 study COMMAND`**: one CLI command per function, `--json` for one JSON object on
  stdout, the CLI's exit codes (0 ok, 2 invalid input, 1 a failed verification or an internal
  error, 130 interrupted).
* **Seven MCP tools** for agents: six read-only (`study_status`, `study_catalog`, `study_plan`,
  `study_aggregate`, `study_tableau`, `study_fss`) and `study_start_run`, a background run that
  the existing `job_status`, `cancel_job` and `list_jobs` tools follow.
* **Eight diagnostic figures** (requirements R349, R350), each written as SVG and PNG with the
  exact numbers it shows as CSV and JSON, titled as a diagnostic.

The v1.4 sections this layer implements or reports: the exact first passages and their
censoring (v1.4 §5.2), the zero extension of `d_cyc` (§5.5), the three peak-time statistics and
the horizon flag (§5.8), the finite-size-scaling report with its censoring and peak policies
(§5.9), the nested estimator, the graph-cluster error and paired rate differences (§6.3, §6.4),
replay verification (§7.4, validation requirement 5), the lineage of every record (§8) and the
labeling of exploratory analyses (§9).

## 2. Notation

| Symbol | Meaning | Type | Code name |
|---|---|---|---|
| $\xi = (n, \beta, p, g, c, r)$ | a trajectory key (protocol F unless said) | tuple | `TrajectoryKey` |
| $(n, \beta, p)$ | a condition; with the protocol, one row of `status.conditions` | tuple | `_Cond` |
| $G, C$ | graphs and circuits per graph of a condition | int | `trials.graphs`, `trials.circuits_per_graph` |
| $C_g$, $R_{gc}$ | circuits stored for graph $g$; replicas of circuit $(g, c)$ ($R = 1$, design D1) | int | `NestedMean.circuits_per_graph`, `replicas_per_circuit` |
| $F_{gcr}(t)$ | a stored scalar value of trajectory $(g, c, r)$ at depth $t$ | float | `observations.value` or a data field |
| $\hat f_g(t)$ | graph $g$'s mean over its circuits and replicas | float | `NestedMean.graph_means` |
| $\hat\mu(t)$, $\widehat{\mathrm{SE}}(t)$ | the nested mean and graph-cluster error (v1.4 §6.3, §6.4) | float | `mean`, `se` |
| $\Delta_g(t)$ | within-graph difference of two rates | float | `paired[].difference` per graph |
| $\tau_\ell$ | first passage $\min\{t : k_t \le \ell\}$ (v1.4 §5.2) | int or censored | `first_passage.tau[ell]` |
| $T_{\max}$, $T$ | the horizon of a condition | int | `horizon`, `layers` |
| $\hat S(t)$ | graph-weighted Kaplan–Meier survival of $\tau_\ell$ | float | `SurvivalCurve.survival` |
| $\mathrm{RM}$ | restricted mean $\mathbb E[\min(\tau_\ell, T_{\max})]$ | float | `restricted_mean` |
| $\phi$ | graph-weighted censored fraction | float | `censored_fraction` |
| $T^*$ | peak time (v1.4 §5.8) of one trajectory | float or missing | `t_star`, `_t_star` |
| $h_i$ | whether $T_{\max}$ is in trajectory $i$'s maximizing set | bool | `at_horizon` |
| $w_i$ | graph-equal weight of trajectory $i$, $\propto 1/(C_g R_{gc})$ | int | `stats.graph_equal_weights` |
| $z_{\mathrm{pl}}$ | the plateau width of the peak rule (manifest `analysis.peak.z`) | float | `plateau_z` |
| $(p_c, z, \nu)$ | scaling parameters; $z$ the dynamical exponent | floats | `fit.p_c`, `fit.z`, `fit.nu` |
| $e_u$ | engine seconds of unit $u$ estimated by `api.estimate_ensemble` | float | `engine_seconds` |
| $m_u$ | measured seconds of a unit (engine and ingestion) | float | `units.elapsed_seconds` |
| $\kappa$ | measured time factor $\sum m / \sum e$ | float | `factor.value` |
| $b$ | stored bytes per snapshot | float | `storage.bytes_per_snapshot` |

## 3. Definitions, as implemented

### 3.1 Conventions shared by every function

* **Stores** are named by their directory (a path string). Reading functions open the store
  without its lock (SQLite readers, doc 05 §3.4), so they work while a run writes; writing
  functions (`init`, `run`, `start_run_job`, `extend`, `compute`, `aggregate(record=True)`) take
  the store's writer lock and are refused, naming the holder, while another writer has it.
* **Results** are dicts of JSON values. Floats are rounded to 6 significant digits for display,
  NaN and infinities become `null`, long lists are cut (`truncated: true` and a note say so).
  At most 24 conditions and, per condition, 50 depths (evenly spaced, both ends kept) are shown;
  `out` writes everything at full precision.
* **Errors.** Every input error is a `ValueError` whose message names the field and what to
  change. A record the store does not hold (the store raises `KeyError`) is an input error too.
  In the CLI that is exit code 2 with the message; over MCP a tool error with the message.
* **`where`** selects conditions and depths: an object with any of `protocol`, `n`, `beta`,
  `p`, `t`, each a value or a list. Tokens are compared canonically (`"0.1"`, `0.1` and `"1/10"`
  are one rate). Absent keys match everything; a selection that matches nothing is refused with
  what the study has.
* **The plateau width** of the peak rule is reported as `plateau_z`; the letter `z` in the
  scaling fits is the dynamical exponent (gap G-FSS-1). The manifest key itself is
  `analysis.peak.z` (doc 02); see §9.

### 3.2 The functions and their results

| Function | CLI | MCP | Result fields |
|---|---|---|---|
| `catalog(observable=None)` | `study catalog [--observable ID]` | `study_catalog` | `count`, `levels`, `observables` (id, version, level, retrospective, inputs, params, definition; shortened definitions with a note when the list would pass 8000 characters); with `observable`: `versions` in full |
| `init(manifest, store)` | `study init MANIFEST STORE` | none (shell) | `store`, `experiment_id`, `label`, `revision`, `manifest_digest`, `protocols`, `conditions`, `rates`, `trajectories_per_condition`, `planned_units`, `planned_trajectories`, `warnings`, `next_step` |
| `status(store)` | `study status STORE` | `study_status` | `complete`; `trajectories` {expected, stored, complete, remaining, reruns}; `units` {done, failed, remaining, seconds, trajectories_per_second}; `snapshots` {total, and per status}; `storage` {megabytes, files, bytes_per_snapshot}; `observables` (stored keys); `peak_rule` (with `plateau_z`); `versions`; `conditions` (incomplete first); `failed_units`; `next_step` |
| `plan(store, threads=None, benchmark=False)` | `study plan STORE [--threads N] [--benchmark]` | `study_plan` | `units`, `trajectories`, `rerun_units`, `rerun_trajectories`, `threads`, `estimated_engine_seconds`, `factor` {value, source, units_measured} or null, `estimated_seconds` (null without a factor), `storage` {bytes_per_snapshot, source, new_snapshots, estimated_megabytes, store_megabytes}, `peak_memory_megabytes`, `largest_unit`, `units_shown`, `by_condition`, `trial` (with `benchmark`), `plan_seconds`, `notes` |
| `run(store, *, max_units, max_seconds, num_threads)` | `study run STORE [--max-units K] [--max-seconds S] [--threads N]` | none (use the job) | `stopped` (`complete`, `max_units`, `max_seconds`), `complete`, `units_run`, `trajectories_run`, `snapshots_written`, `elapsed_seconds`, `units_remaining`, `trajectories_remaining` |
| `start_run_job(store, *, max_units, max_seconds, num_threads)` | none (`study run` in a shell) | `study_start_run` | a job snapshot: `job_id`, `kind: "study_run"`, `status` (`queued`, `running`, then `completed`, `stopped`, `cancelled` or `failed`), `units_done`, `units_total`, `trajectories_done`, `trajectories_total`, `progress`, `elapsed_seconds`, `eta_seconds`, `trajectories_per_second`; at the end `stopped`, `units_remaining`, `trajectories_remaining`, `error` |
| `extend(store, manifest)` | `study extend STORE MANIFEST` | none (shell) | `revision`, `changed`, `manifest_digest`, `units_remaining`, `trajectories_remaining`, `rerun_units`, `rerun_trajectories`, `warnings` |
| `tableau(store, *, n, beta, p, g, c, t, protocol, replay, out)` | `study tableau STORE --n --beta --p --g --c --t [--replay] [--out FILE]` | `study_tableau` (no `out`) | `key`, `t`, `source` (`store` or `replay`), `snapshot_status`, `qubit_order`, `joint` {num_qubits, rank, pure, digest}, `k`, `physical` {num_qubits, rank, digest}, `verified` (replay: `digest_matches_store`, `k_trace_through_layer`), `generators` and `physical_generators` (at most 32 qubits), `file`, `file_format` |
| `compute(store, observable, params, *, protocol, n, beta, p, t)` | `study compute STORE OBSERVABLE [--params JSON]` | none | `observable`, `version`, `params`, `inserted`, `existing`, `snapshots`, `elapsed_seconds` |
| `aggregate(store, observable, params, where, ...)` | `study aggregate STORE OBSERVABLE ...` | `study_aggregate` (no `record`, `out`) | `observable`, `kind`, `estimand`, `estimator`, `conditions` (per kind, §3.3), `paired`, `notes`, `recorded`, `files` |
| `fss(store, observable, beta, ...)` | `study fss STORE --beta B --observable ID ...` | `study_fss` (no `out`; refused above 50 s of refits) | `schema`, `hypothesis`, `beta`, `label`, `status`, `settings`, `sizes`, `results` (per observable: `status`, `points_used`, `points_excluded`, `excluded` with reasons, `fit`, `bootstrap`, `leave_one_size_out`, `diagnostics`, `input_digest`), `shared`, `files` |
| `export(store, directory)` | `study export STORE DIR` | none | `directory`, `format`, `files`, `megabytes`, `conditions`, `top_level` |
| `verify(store, sample=8, seed=0, *, num_threads)` | `study verify STORE [--sample K / --all] [--seed S]` (exit 1 when not ok) | none | `ok`, `trajectories`, `snapshots`, `failures`, `failures_total`, `sample`, `seed`, `elapsed_seconds` |
| `plot(store, kind, out, options)` | `study plot KIND --store STORE --out DIR [--options JSON]` | none | `kind`, `title`, `caption`, `figures` {svg, png}, `data` {csv, json}, `points` |

`aggregate_arrays` (the same aggregation, uncapped, with numpy arrays and the estimator
objects) and `fss_report` (the full report) serve scripts and the figures; they are not JSON.

### 3.3 What `aggregate` computes, by kind

The kind follows the catalog (`study/api.py::_aggregate_kind`):

* **snapshot** (every snapshot-level observable: `k`, `code_rate`, `stabilizer_rank`, `d_cyc`,
  `D`, `half_cut_entropy`, `d_conn`, `mean_leakage`, and the data fields of the vector ones):
  for each selected condition and each stored depth $t$, the nested mean of v1.4 §6.3
  $$\hat\mu(t) = \frac1G\sum_g \hat f_g(t),\qquad \hat f_g(t) = \frac1{C_g}\sum_c \frac1{R_{gc}}\sum_r F_{gcr}(t),$$
  and the graph-cluster error of v1.4 §6.4,
  $\widehat{\mathrm{SE}}(t) = \bigl(\sum_g (\hat f_g - \hat\mu)^2 / (G(G-1))\bigr)^{1/2}$
  (`stats.nested_mean`; doc 06 §3, §4). The value is the stored `value` column, or the data
  entry `field` (entry `index` of a list field). A row counts as undefined when its value is not
  finite or, for the `value` column, when its status is not `ok` (the rule of
  `TableauStore.aggregate`): `d_cyc` without a code (`undefined_no_code`), a state absent after a
  purification stop, a vector row. Undefined rows are refused unless `nan_policy="omit"`, whose
  estimand (each graph's mean over its defined values, averaged over the graphs that have one)
  is stated in the result. For `d_cyc` the unconditional trace of v1.4 §5.5 is
  `field="zero_extended"` ($d_{\mathrm{cyc}} = 0$ where $k_t = 0$).
  With `paired_rates` $= (p_1, p_2)$: for every $(n, \beta)$ holding both rates and every common
  depth, $\Delta_g = \hat f_g(p_2) - \hat f_g(p_1)$, their mean and graph-cluster error
  (`stats.paired_difference`; doc 06 §5).
* **first_passage**: for each threshold $\ell$ (default 0 and 1; $\ell > n$ is skipped), with
  times $x_i = \tau_{\ell,i}$ where observed and $x_i = T_{\max}$ where censored: the censored
  count and the graph-weighted censored fraction $\phi$ with its error (`stats.censoring_summary`),
  the graph-weighted Kaplan–Meier median and `median_upper` (`stats.nested_kaplan_meier`; doc 06
  §6.2, §6.4), the restricted mean $\mathrm{RM} = \int_0^{h}\hat S(t)\,dt$ with its error at the
  largest time $h$, and the mean of $\tau_\ell$ **only when nothing is censored** (`mean` is
  `null` otherwise). `horizon` is $T_{\max}$.
* **peak** (`peak_d_cyc`, `peak_D`, `peak_D_J`, `peak_d_conn`): `stats.peak_time_summaries`
  (doc 06 §8) of the stored $T^*$: the arithmetic mean $\mathbb E[T^*]$, the geometric mean
  $\exp\mathbb E[\ln T^*]$ (each over trajectories with a peak), the argmax over the positive
  checkpoints of the nested mean trace $\mathbb E[d(t)]$ (the trace behind the peak: `d_cyc`
  zero-extended, `D`, the coupling's column of `D_J`), `missing_fraction` and
  `horizon_fraction`, under the stated `horizon_policy`: `refuse` (default: the means are
  withheld when any peak is at $T_{\max}$), `exclude` (means over interior peaks) or `include`
  (means that are lower bounds).
* **k_trace**: $\hat\mu$ and $\widehat{\mathrm{SE}}$ of $k_t$ (or $k_t/n$ with `normalized`)
  at every layer $0..T_{\max}$, from the decoded traces.

`params` picks a stored parameter set: an object whose entries must all equal a stored set
(so `{"J": 1}` picks the coupling of `peak_D_J` whatever the peak rule's settings), the stored
canonical text, or nothing when one set is stored (`study/api.py::_select`).

### 3.4 What `fss` fits

`fss(store, observable, beta, ...)` builds one scaling dataset per observable $\theta$ at one
$\beta$ (v1.4 §5.9) and fits it with `study/fss.py` (doc 07). The points are the conditions
$(n, \beta, p)$ of the study, restricted by `sizes`, `n_min` and `p_range`; a point is **used**
when all of the following hold, else it is listed in `excluded` with its reason:

1. its condition is complete under the latest revision (every $(g, c)$ has its records; else
   "incomplete");
2. for $\theta = \tau_\ell$ (`first_passage` with `ell`, or `ell_fraction` $f$ for
   $\ell = \lfloor f n\rfloor$): no trajectory is censored at $\ell$, and then
   $y = \hat\mu$ of $\tau_\ell$; or, with `censoring="restricted_mean"`, $\phi \le$
   `max_censored_fraction`, and then $y$ is the nested mean of $\min(\tau_\ell, T_{\max})$, a
   lower bound of $\mathbb E[\tau_\ell]$ (§4, Proposition 3; the estimator text says so);
3. for a peak time: the fraction without a peak is at most `max_missing_fraction` and the
   fraction with the peak at the horizon at most `max_horizon_fraction` (both 0 by default), and
   then $y$ is the arithmetic $\mathbb E[T^*]$ under `horizon_policy` (`exclude` by default);
4. at least two graphs gave a value and $\widehat{\mathrm{SE}} > 0$ (the collapse weights by
   $1/\mathrm{SE}^2$).

The default box is $p_c \in [\min p, \max p]$ over the used points, $z \in [0, 3]$,
$\nu \in [0.5, 5]$. The report holds the settings, the sample hierarchy, the depth schedule
each size holds (R322), every point with its counts and fractions, the fit, the graph-cluster
bootstrap (`bootstrap` replicates, paired across $p$), leave-one-size-out refits, the
diagnostics of `fss.collapse_diagnostics`, the SHA-256 of the input points (`input_digest`), and
the hypothesis line "unverified ansatz". With a list of observables, `shared` adds the shared
fit and `fss.compare_shared_independent`, worded as "consistent with" or "rejected at alpha"
(R341), never as a confirmation. With `out` the report is written as canonical JSON (sorted
keys, exact floats, no timing) and the points as CSV.

### 3.5 What `plan` estimates

For the remaining units $u$ (`runner.plan`, reruns first): $e_u$ is
`sparsegf2.api.estimate_ensemble` of the unit's engine spec (the manifest's keywords of v1.4
§8.1, the requirements' horizon and probes, one of the condition's graphs, the unit's
trajectories) on the chosen threads. The measured factor is the ratio estimator
$$\kappa = \frac{\sum_{u \in U} m_u}{\sum_{u \in U} e_u}$$
over the last 200 units $U$ this store ran (`source: "store"`), or over one trial unit run
twice in a scratch store, the second run timed (`source: "trial"`, with `benchmark=True`); the
estimate is $\kappa\sum_u e_u$, and `null` without a factor. Storage is the new snapshots times
$b$, the bytes per snapshot measured in the store (its file bytes over its snapshot rows), or
in the trial (the larger of its file growth and its stored payload, over its snapshots). The
memory of a unit is the engine's result (`result_megabytes`) plus one more copy of its
snapshot arrays, which ingestion holds as canonical tableaux beside them (a model).

### 3.6 Runs, jobs and stopping

`run` and `start_run_job` plan once and execute the units in order, in batches of
`runner.run(units=...)` calls that hold `sparsegf2.api._COMPUTE_LOCK` (one parallel region at a
time in a process, as for ensemble jobs). After each unit is committed the runner's progress
callback checks the budgets: a set cancel flag or a passed `max_seconds` deadline stops the run
**between units**; `max_units` truncates the plan. The job is a subclass of `sparsegf2.api`'s
`_Job` kept in its job table, so `job_status`, `cancel_job` and `list_jobs` work unchanged. Its
final status is `completed` when no unit remains, `stopped` when a budget ended it with units
left, `cancelled` after `cancel_job`, `failed` with the error otherwise.

## 4. Propositions

**Proposition 1 (stopping and resuming change nothing; from doc 05).** Let a study be run by
any sequence of `run` and `start_run_job` calls, with any budgets, cancellations, thread counts
and interruptions, until `plan` is empty. The store then holds exactly the records of one
uninterrupted `runner.run`; in particular its logical digest (`history=False`) is the same.

*Proof.* Each call plans the units without a completion under the current requirements and
hands them to `runner.run` in batches. A stop between units leaves only whole committed units
(doc 05 Proposition 1: a unit is all or nothing; an interrupt inside a unit rolls it back).
How the units are grouped into `runner.run` calls changes no record, because one trajectory's
records do not depend on its unit, the other units or the thread count (doc 05 Proposition 2).
The next call plans exactly the trajectories without a completion, and running them to the end
gives the uninterrupted store (doc 05 Proposition 3). ∎

**Proposition 2 (the cost of planning in the batched run).** Let one plan take $s$ seconds and
one unit $m$ seconds. With batches of $k = \min(256, \max(1, \lceil 20 s / m\rceil))$ units,
the planning that `runner.run` repeats at the end of each batch adds at most a fraction
$s / (k m) \le 1/20$ of the run's time once $k < 256$, and $s / (256\, m)$ otherwise.

*Proof.* Each `runner.run` call plans once at its end (to report the units left), so a batch of
$k$ units costs about $k m + s$. If $k = \lceil 20 s / m\rceil$ then $k m \ge 20 s$, and the
overhead $s/(k m) \le 1/20$. ∎ (Measured here: one plan of 12,000 units over 480 condition rates took 0.6 to 0.7 s; the batch
size is chosen after each batch from its measured time per unit.)

**Proposition 3 (a censored point does not identify $\mathbb E[\tau]$; derived here).** Let
$\tau \ge 0$ and $T > 0$ with $\Pr(\tau > T) = \phi > 0$. Then
$$\mathbb E[\tau] = \mathbb E[\min(\tau, T)] + \phi\,\mathbb E[\tau - T \mid \tau > T],$$
and for every $M \ge 0$ there is a law that agrees with the law of $\tau$ on $[0, T]$ (so every
trajectory observed up to $T$ has the same distribution) and has
$\mathbb E[\tau - T \mid \tau > T] \ge M$. Hence no estimator from data censored at $T$ is
consistent for $\mathbb E[\tau]$, however small $\phi$, without a model of the tail.

*Proof.* $\tau = \min(\tau, T) + (\tau - T)_+$ pointwise; take expectations and write
$\mathbb E[(\tau - T)_+] = \phi\,\mathbb E[\tau - T \mid \tau > T]$. For the second claim keep
the law on $[0, T]$ and move the mass $\phi$ of $(T, \infty)$ to the single point $T + M$:
the observed data (the exact $\tau$ where $\tau \le T$, the censoring indicator elsewhere) have
the same distribution under both laws, and the conditional excess is $M$. ∎

So `fss` excludes a point with any censoring by default, and admits
$\mathbb E[\min(\tau, T_{\max})]$ only when asked, named as the lower bound it is (§3.4). A
threshold on $\phi$ alone (gap proposal G-FSS-3: "E[τ] from the Kaplan–Meier curve where the
censored fraction is below 1%") bounds nothing: the bias $\phi\,\mathbb E[\tau - T\mid\tau>T]$
can be any size. In the mixed phase the conditional excess grows with $n$, so the bias would
also distort the fitted $z$.

**Proposition 4 (the end of the $T^*$ distribution is one minus the missing fraction; derived
here).** Let $w_i \propto 1/(C_g R_{gc})$ be the graph-equal weights of the trajectories of a
condition (`stats.graph_equal_weights`) and $D$ the set with a defined $T^*$. Then
$\sum_{i \in D} w_i / \sum_i w_i = 1 - \texttt{missing\_fraction}$, where `missing_fraction` is
the nested mean of the indicator of a missing peak.

*Proof.* With $w_i = L/(C_g R_{gc})$, $\sum_i w_i = \sum_g \sum_c \sum_r L/(C_g R_{gc}) = G L$,
and for any set $A$ of trajectories
$\frac{1}{G L}\sum_{i\in A} w_i = \frac1G\sum_g \frac1{C_g}\sum_c \frac1{R_{gc}}\sum_r \mathbf 1[(g,c,r)\in A]$,
the nested mean of $\mathbf 1_A$ (v1.4 §6.3). Take $A = D$ and use
$\mathbf 1_D = 1 - \mathbf 1_{\text{missing}}$. ∎

The `peak_distribution` figure plots the cumulative sums of these normalized weights in the
order of $T^*$, so its curve ends at height $1 - $ `missing_fraction`; the same argument with
$A = D \cap \{h_i\}$ gives the `horizon_fraction`.

**Proposition 5 (the measured factor).** If every unit's measured time is proportional to its
engine estimate, $m_u = \kappa^\star e_u$, then $\kappa = \kappa^\star$ and the estimate
$\kappa \sum_u e_u$ is the true remaining time. If instead $m_u = \kappa_u e_u$ with
$\kappa_u$ depending on the condition, $\kappa$ is the $e$-weighted mean of the $\kappa_u$ of
the measured units, and the estimate is exact only when the remaining units have the same
$e$-weighted mean.

*Proof.* $\sum m_u / \sum e_u = \sum \kappa_u e_u / \sum e_u$. ∎

Ingestion (canonical forms, observables, one transaction) is Python and scales with $n$
differently from the engine; the measurements of doc 05 §5.8 show its share falling as $n$
grows. A trial on the most expensive condition within 5 estimated engine seconds, or a store's
own history, is therefore the better guide; the result says which (`factor.source`).

**Proposition 6 (the report file is a function of the data and the settings).** Two `fss` calls
on stores with the same records and the same settings write byte-identical report files.

*Proof.* The points are deterministic functions of the stored records (the estimators of
`stats.py` are bitwise invariant under row order, doc 06 §9.1). The fit, the bootstrap draws
(`graph_bootstrap` with seed $[s, n]$), the multi-start points and the refits are deterministic
given the seed (doc 07 §6, §10). The report holds no time and no path; it is written with sorted
keys, exact floats and a fixed layout (`study/api.py::_write_json`). ∎

## 5. Algorithms

**`plan`** (`study/api.py::plan`): (1) `runner.plan` lists the units; (2) for each, the
requirements of its condition (`runner.requirements`, cached per condition) and its JSON engine
spec (`_unit_spec`, `_probes_json`); (3) `api.estimate_ensemble` on one graph of the condition,
cached by requirements digest, rate, trajectory count and threads; (4) the factor from the
store's units or a trial (`_history_factor`, `_trial`); (5) totals, the largest unit, the first
12 units and the first 24 conditions. Cost: one plan plus one estimate per distinct
(requirements, rate, trajectory count), each building a `MonitoredCircuit` (milliseconds), plus
the trial's two unit runs with `benchmark`.

**`run` and the job** (`_execute`): (1) plan once; truncate to `max_units`; (2) set the
deadline; (3) while units remain: stop if canceled or past the deadline; run the next batch of
$k$ units (§4 Proposition 2) under the compute lock with a progress callback that counts each
committed unit and raises a private `_Stop` when the cancel flag is set or the deadline has
passed; (4) plan again for the units left. A job runs the same loop in a daemon thread that opens
the store as its writer first; the starting call waits until the store is open, so a refused
lock is reported synchronously and no job is listed for it.

**`aggregate`** (`_aggregate`): (1) resolve the kind, the stored parameter set and the
conditions; (2) per condition one indexed query per needed field (`TableauStore.observations`);
(3) group by depth and apply `stats.nested_mean`, `nested_kaplan_meier`, `censoring_summary`,
`peak_time_summaries` or `paired_difference`; (4) render (`_render`): drop private keys,
subsample depths beyond 50 (`sparsegf2.api._subsample`: evenly spaced, both ends), keep 24
conditions, then `sparsegf2.api._bounded`. Cost: linear in the selected rows; the peak kind also
reads the trace observable once per condition.

**`fss`** (`fss_report`): (1) validate the settings; (2) the incomplete conditions from the plan;
(3) the points (§3.4) and a `ScalingData` with per-graph values for the bootstrap; (4)
`fit_collapse`; if `max_seconds` is given and the measured fit time times the number of refits
still to come (bootstrap + 1, and sizes + 1 more with leave-one-size-out) exceeds it, refuse;
(5) `bootstrap_fit`, `leave_one_size_out`, `collapse_diagnostics`; (6) with several observables
`compare_shared_independent`; (7) the summary, and with `out` the files.

**Figures** (`study/plots.py`): (1) the numbers from `_aggregate` (or the report); (2) a
`matplotlib.figure.Figure` drawn without pyplot under one `rc_context` style; (3) the fixed-order
categorical palette, at most 8 series (more are refused: narrow `where`); (4) bands of one
graph-cluster standard error where a standard error exists; (5) `.svg` (no date, a fixed hash
salt, so reruns give identical files) and `.png` at 150 dpi; (6) the plotted numbers as `.csv`
(one row per point) and `.json` (with the title, caption and settings).

| Figure | What is drawn | Data columns |
|---|---|---|
| `k` | mean $k_t$ or $k_t/n$ at every layer, band ±1 SE | protocol, n, beta, p, t, mean, se |
| `survival` | Kaplan–Meier step curve of $\tau_\ell$ to $T_{\max}$, band ±1 SE, a tick at the censoring depth | protocol, n, beta, p, ell, t, survival, se |
| `d_cyc` | zero-extended mean $d_{\mathrm{cyc}}$ versus depth (symmetric log axis), band ±1 SE | protocol, n, beta, p, t, mean, se, has_code_fraction |
| `peak_distribution` | graph-weighted cumulative share with $T^* \le t$ (log axis), hollow markers for horizon peaks | protocol, n, beta, p, g, c, r, t_star, at_horizon, weight, cumulative |
| `peak_vs_tau1` | $T^*$ against $\tau_1$ (log axes), the diagonal, hollow horizon peaks | protocol, n, beta, p, g, c, r, tau_1, t_star, at_horizon |
| `vs_p` | an observable versus $p$, one line per $(n, \beta)$ | protocol, n, beta, p, p_float, value, se |
| `graph_measures` | $q_G$, mean distance, $\lambda_2$, degree variance versus $\beta$ | n, beta, beta_float, measure, mean, se, graphs |
| `collapse` | scaled points $x = (p-p_c)n^{1/\nu}$, $Y = y n^{-z}$, $\pm \mathrm{SE}\,n^{-z}$, the polynomial master curve | observable, n, p, p_float, x, Y, dY |

## 6. From the mathematics to the code

| Definition or step | Code |
|---|---|
| $\hat\mu(t)$, $\widehat{\mathrm{SE}}(t)$ per condition and depth | `study/api.py::_snapshot_kind` → `stats.nested_mean` |
| undefined rows and `nan_policy` | `study/api.py::_snapshot_kind` (rule of `store.py::TableauStore.aggregate`) |
| paired differences $\Delta_g$ | `study/api.py::_paired` → `stats.paired_difference` |
| $\phi$, Kaplan–Meier, RM, mean only uncensored | `study/api.py::_first_passage_kind`, `_survival` → `stats.censoring_summary`, `stats.nested_kaplan_meier` |
| peak statistics and the horizon policy | `study/api.py::_peak_kind`, `_peak_traces`, `_trace_matrix` → `stats.peak_time_summaries` |
| $k_t$ at every layer | `study/api.py::_k_trace_kind` |
| stored parameter sets | `study/api.py::_select`, `_matches`, `_stored_keys` |
| FSS points and exclusions (§3.4) | `study/api.py::_fss_points`, `_dataset`, `_estimator_text` |
| the fit, bootstrap, leave-one-size-out, diagnostics, shared test | `study/api.py::fss_report` → `fss.fit_collapse`, `bootstrap_fit`, `leave_one_size_out`, `collapse_diagnostics`, `compare_shared_independent` |
| `input_digest` | `study/api.py::_input_digest` |
| $e_u$, memory | `study/api.py::_Estimator.estimate`, `_unit_spec`, `_probes_json` → `api.estimate_ensemble` |
| $\kappa$ from the store, from a trial; $b$ | `study/api.py::_history_factor`, `_trial`, `_storage` |
| batched execution and stopping | `study/api.py::_execute`, `_Progress`, `_Stop` |
| the background job | `study/api.py::_StudyJob`, `_study_job`, `start_run_job` |
| rendering and capping | `study/api.py::_render`, `_finish` (`api._bounded`) |
| figures | `study/plots.py::k_trace`, `survival`, `d_cyc`, `peak_distribution`, `peak_vs_tau1`, `versus_p`, `graph_measures`, `collapse`; `_save` |
| CLI commands | `cli.py::_add_study`, `_STUDY_COMMANDS` |
| MCP tools | `mcp_server.py::build_server` (`study_*`) |

## 7. Worked example: a tiny study end to end

Six conditions ($n \in \{6, 8\}$, $\beta = 0.2$, three rates), three graphs of two circuits,
horizon $4n$, a geometric schedule of five points: 36 trajectories, a second to run. Every
block checks a claim. The shell forms of the same steps are

```sh
sparsegf2 study init manifest.json studies/doc09
sparsegf2 study plan studies/doc09 --threads 1
sparsegf2 study run studies/doc09 --max-units 2 --threads 1    # then again, to finish
sparsegf2 study status studies/doc09
sparsegf2 study aggregate studies/doc09 first_passage --where '{"n": 8}' --ell 0
sparsegf2 study fss studies/doc09 --beta 0.2 --observable first_passage --ell 0 --out fits
sparsegf2 study tableau studies/doc09 --n 6 --beta 0.2 --p 0.1 --g 0 --c 0 --t 5 --replay
sparsegf2 study plot survival --store studies/doc09 --out figures --options '{"ell": 0}'
sparsegf2 study verify studies/doc09 --sample 2
```

**(a) Create the store and look at the plan.** Nothing runs; the engine estimate has no factor
yet, so `estimated_seconds` is null and a note says how to measure one.

```python
import math
import tempfile
from pathlib import Path

import numpy as np

from sparsegf2.study import api, stats
from sparsegf2.study.store import TableauStore

root = Path(tempfile.mkdtemp())
store = str(root / "store")
manifest = {
    "experiment_id": "doc09-demo",
    "conditions": [{"n": [6, 8], "beta": ["0.2"], "p": ["0.1", "0.2", "0.3"]}],
    "depth": {"layers": {"per_n": 4}, "checkpoints": {"count": 5}},
    "trials": {"graphs": 3, "circuits_per_graph": 2},
    "seeds": {"master_graph_seed": 5, "master_circuit_seed": 6},
    "observables": {
        "tableau_snapshots": True,
        "contiguous_distance": True,
        "uniform_erasure": {"samples": 12, "exact_budget": 0},
    },
    "analysis": {"peak": {"bootstrap": 0}},
}
made = api.init(manifest, store)
assert made["planned_trajectories"] == 2 * 3 * 3 * 2  # sizes x rates x graphs x circuits
before = api.plan(store, threads=1)
assert before["units"] == made["planned_units"] and before["estimated_seconds"] is None
print(before["units"], "units;", before["units_shown"][0]["unit"])
```

**(b) Run with a budget, resume, and compare with a run in one go** (Proposition 1). After the
first units the plan's factor is measured from them.

```python
first = api.run(store, max_units=2, num_threads=1)
assert first["stopped"] == "max_units" and first["units_run"] == 2
measured = api.plan(store, threads=1)["factor"]
assert measured["source"] == "store" and measured["units_measured"] == 2
rest = api.run(store, num_threads=1)
assert rest["stopped"] == "complete" and api.status(store)["complete"]

twin = str(root / "twin")
api.init(manifest, twin)
api.run(twin, num_threads=1)
with TableauStore.open(store) as a, TableauStore.open(twin) as b:
    assert a.logical_digest(history=False) == b.logical_digest(history=False)
```

**(c) A snapshot aggregate equals the nested mean and cluster error written out by hand** (v1.4
§6.3, §6.4): three graphs, two circuits each, so each graph's mean is the mean of its two values.

```python
out = api.aggregate(store, "k", where={"n": 8, "p": "0.2"})
cond = out["conditions"][0]
i = cond["depths"].index(6)  # depths 0, 1, 2, 6, 13, 32
with TableauStore.open(store) as s:
    rows = s.observations("k", n=8, beta="0.2", p="0.2", t=6)
graph_means = [float(rows["value"][rows["g"] == g].mean()) for g in range(3)]
mu = sum(graph_means) / 3
se = math.sqrt(sum((f - mu) ** 2 for f in graph_means) / (3 * 2))
assert math.isclose(cond["mean"][i], mu, rel_tol=1e-5)
assert math.isclose(cond["se"][i], se, rel_tol=1e-5, abs_tol=1e-12)
print("k at t = 6:", cond["mean"][i], "+-", cond["se"][i])
```

**(d) First passages: no mean of censored times.** Where a trajectory is censored the mean is
null (Proposition 3); where none is, the mean equals the restricted mean.

```python
fp = api.aggregate(store, "first_passage", where={"n": 8}, ell=[0])
for cond in fp["conditions"]:
    (entry,) = cond["ells"]
    if entry["censored"]:
        assert entry["mean"] is None and entry["restricted_mean"] <= entry["horizon"]
    else:
        assert math.isclose(entry["mean"], entry["restricted_mean"], rel_tol=1e-5)
    print(cond["p"], "censored", entry["censored"], "of", entry["trajectories"],
          "median", entry["median"], "RM", entry["restricted_mean"])
```

**(e) Peaks, and Proposition 4.** The weighted share of trajectories with a defined $T^*$ is one
minus the missing fraction.

```python
peaks = api.aggregate_arrays(store, "peak_d_cyc", where={"n": 8, "p": "0.3"},
                             horizon_policy="include")
cond = peaks["conditions"][0]
w = stats.graph_equal_weights(cond["_g"], cond["_c"], cond["_r"]).astype(float)
defined = np.isfinite(cond["_t_star"])
assert math.isclose(w[defined].sum() / w.sum(), 1.0 - cond["missing_fraction"])
print("E[T*] =", cond["arithmetic_mean"], "horizon fraction", cond["horizon_fraction"])
```

**(f) A scaling report.** Points with censoring are excluded and listed; the report file is the
same bytes when the call is repeated (Proposition 6). With two sizes the parameters are not
constrained, and the report says so.

```python
fits = api.fss(store, "first_passage", "0.2", ell=0, starts=1, bootstrap=4,
               out=str(root / "fits"))
res = fits["results"][0]
assert all(e["reason"].startswith("censored") for e in res["excluded"])
again = api.fss(store, "first_passage", "0.2", ell=0, starts=1, bootstrap=4,
                out=str(root / "fits2"))
name = Path(fits["files"]["report"]).name
assert (root / "fits" / name).read_bytes() == (root / "fits2" / name).read_bytes()
print(res["status"], res["points_used"], "points used;", res.get("notes"))
```

**(g) A stored tableau and its replay.** The replayed joint state has the stored digest.

```python
t5 = api.tableau(store, n=6, beta="0.2", p="0.1", g=0, c=0, t=5)
replayed = api.tableau(store, n=6, beta="0.2", p="0.1", g=0, c=0, t=5, replay=True)
assert replayed["joint"]["digest"] == t5["joint"]["digest"]
assert replayed["verified"]["digest_matches_store"] is True
print("k =", t5["k"], t5["physical_generators"])
```

**(h) A figure, an export and a verification.**

```python
figure = api.plot(store, "survival", str(root / "figures"), {"ell": 0, "where": {"n": 8}})
assert figure["title"].startswith("Diagnostic:") and Path(figure["figures"]["svg"]).is_file()
exported = api.export(store, str(root / "export"))
assert exported["conditions"] == 6
assert api.verify(store, sample=2, num_threads=1)["ok"]
print(figure["caption"][:120], "...")
```

## 8. Validation map

| Test | What it checks |
|---|---|
| `tests/test_study_api.py::test_snapshot_means_equal_an_independent_nested_mean` | means, errors and contributing graphs equal loops over the stored rows (§3.3), and the store's own aggregate; exact to the display rounding |
| `tests/test_study_api.py::test_d_cyc_without_a_code_is_never_averaged_silently` | undefined rows refused with the fix named; `zero_extended` and `nan_policy="omit"` equal their loop references |
| `tests/test_study_api.py::test_paired_rates_are_differences_within_graphs` | paired differences and their errors equal per-graph differences written out |
| `tests/test_study_api.py::test_first_passage_summaries_never_average_censored_times` | censored counts and fractions, the Kaplan–Meier curve and median against an independent product-limit estimator, RM, mean only without censoring |
| `tests/test_study_api.py::test_peak_summaries_state_their_horizon_policy` | means withheld under `refuse`; arithmetic, geometric, horizon fraction and argmax against loops |
| `tests/test_study_api.py::test_k_trace_is_the_mean_of_the_stored_traces` | $k_t/n$ means and errors at every layer |
| `tests/test_study_api.py::test_the_estimate_spec_has_the_runners_engine_probes` | the JSON spec priced by `plan` builds the runner's engine run (keywords and probes), for F and P and a correlated probe |
| `tests/test_study_api.py::test_plan_estimates_each_unit_with_estimate_ensemble`, `test_plan_takes_its_factor_from_the_units_run`, `test_plan_benchmark_measures_a_trial_and_leaves_the_study_untouched` | $\sum e_u$ recomputed; $\kappa$ recomputed from the store's units; the trial writes nothing to the study (equal logical digest) |
| `tests/test_study_api.py::test_run_stops_between_units_and_resumes_to_the_same_store`, `test_batched_execution_gives_the_same_records` | Proposition 1: budgets and batching give the logical digest of a run in one go |
| `tests/test_study_api.py::test_a_study_job_is_followed_by_the_job_tools`, `test_a_study_job_can_be_canceled_between_units`, `test_a_store_with_another_writer_is_refused_by_name` | job statuses, counts and listing; cancellation; a held lock refused synchronously, naming the holder |
| `tests/test_study_api.py::test_tableau_from_the_store`, `test_tableau_replay_reproduces_the_store_and_reaches_any_depth`, `test_tableau_out_npz_and_lookup_errors` | digests, $k$, Pauli strings written letter by letter, replay verification, the SGF2RS1 and npz files |
| `tests/test_study_api.py::test_fss_report_lists_points_exclusions_and_settings`, `test_fss_rerun_writes_identical_bytes`, `test_fss_censoring_and_peak_policies`, `test_fss_shared_test_and_its_wording`, `test_fss_refuses_a_report_beyond_max_seconds` | §3.4 point by point (used points equal loop means, censored ones excluded with reasons); Proposition 6; the restricted-mean option; R341 wording; the time guard |
| `tests/test_study_api.py::test_aggregate_errors_say_what_to_change`, `test_fss_errors_say_what_to_change`, `test_run_arguments_are_checked` | every input error is a `ValueError` naming the field |
| `tests/test_study_cli.py` (all) | every command's `--json` equals the API's result; invalid input exits 2 with the API's message; a failed verification exits 1; progress on stderr |
| `tests/test_mcp_server.py::test_the_study_tools_are_annotated_and_described`, `test_the_study_tools_return_the_api_results`, `test_study_errors_are_messages_that_say_what_to_change`, `test_a_study_run_through_the_job_tools` | tool list, read-only annotations, descriptions under 2000 characters with an example, strict arguments, parity with the API, the job lifecycle |
| `tests/test_study_plots.py` (all) | every figure draws headless and writes SVG, PNG, CSV, JSON; titles start "Diagnostic:"; the plotted numbers equal loop references (nested means, product-limit, per-graph measures, scaled coordinates); the drawn lines equal the CSV; reruns give identical files; more than 8 series refused |

All checks are exact (equalities, to the 6 significant digits of the JSON display where a
displayed number is compared); none is statistical, so there is no false-alarm rate. Test
stores run on one thread: results do not depend on the thread count (doc 05 Proposition 2), and
on a loaded machine one thread avoids waiting for the others.

## 9. Limitations and departures

* **S12. The FSS censoring rule** (departure from gap proposal G-FSS-3, justified by
  Proposition 3). G-FSS-3 proposed $\mathbb E[\tau_\ell]$ "from the Kaplan–Meier curve where the
  censored fraction is below a declared threshold (default 1%)". That quantity is not
  identified from censored data. `fss` excludes a point with any censoring by default and lists
  it; `censoring="restricted_mean"` admits $\mathbb E[\min(\tau, T_{\max})]$ explicitly as a
  lower bound. Longer horizons (`extend` with a larger $T_{\max}$) are the remedy v1.4 §5.9
  names first.
* **S13. FSS settings are arguments, not yet a manifest block.** Gap G-FSS-1 proposes an
  `analysis.fss` block and renaming `analysis.peak.z` to `plateau_z`. The manifest
  (`study/manifest.py`, another part) has neither yet, and refuses unknown keys. Until it does,
  `fss` takes the settings as arguments with stated defaults and echoes every resolved setting in
  its report (so a report is reproducible from itself), and every output of this layer names the
  plateau width `plateau_z`. The manifest change is recorded as an open item.
* **S14. A new job status.** A study job that a budget ended with units left reports `stopped`
  (ensemble jobs have no budgets and never do); `completed` always means no work is left.
* **S15. The MCP subset.** `init`, `extend`, `compute`, `export`, `verify`, `plot` and the
  synchronous `run` are not MCP tools: they create or change files, and an agent with a shell
  runs them as `sparsegf2 study ...` (the same functions). `study_tableau`, `study_fss` and
  `study_aggregate` have no `out` or `record` over MCP, so the six read-only tools write nothing;
  `study_fss` refuses a call whose refits would take more than 50 s (`sparsegf2 study fss` has
  no limit).
* **Estimates are models.** The engine estimate is `estimate_ensemble`'s pessimistic model;
  the factor is a ratio over measured units (Proposition 5) and is exact only when every unit's
  time is proportional to its engine estimate. The bytes per snapshot of a small store include
  its fixed overhead (the SQLite schema and file pages), so early estimates of storage are high.
  The memory per unit is a model (the engine's result and one copy of its snapshots), not a
  measurement.
* **What is not shown.** `peak_d_conn` is aggregated when `d_conn` peaks are stored; the runner
  does not record them by itself. A figure draws at most 8 series. The geometric-mean peak time
  $\exp\mathbb E[\ln T^*]$ is not the mean of per-graph values, which the graph bootstrap needs
  (`fss.ScalingData`), so `fss` fits the arithmetic $\mathbb E[T^*]$.
* **Jobs live in their process.** After a restart the job ids are gone and the store holds every
  finished unit; `start_run_job` again resumes (Proposition 1).
* **matplotlib** is needed for the figures only (it is in the `test` extra); `import sparsegf2`
  and every other function work without it.
