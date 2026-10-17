# Identity, seeds and the experiment manifest: tokens, keys, the seed hierarchy, stream indices, graph records and revisions

How the Watts–Strogatz study names things so that every graph, circuit and trajectory can be
rebuilt from one frozen description: exact rate tokens, trajectory keys, the seeds of graphs and
of circuits, the trajectory index $u = g\,s + c$ and why extending the study never renumbers it, the
graph record with its edge digest, canonical JSON and digests, the experiment manifest field by
field, the checkpoint rule of v1.4 §9 stated exactly with its rounding (and its list and table
forms), the rules under which a later manifest may extend an earlier one, and the schedule that
results across revisions (ruling R-CK: the union of the schedules of all revisions, which never
drops a stored depth).

This document is written for a reader fluent in the stabilizer formalism and for language
models. Every symbol is defined where it first appears, every formula is the one the code
computes, and every statement about the code names the function it describes as
`path::function` (paths relative to `src/sparsegf2/`, or `tests/` where marked). The Python
blocks run as written, in order, in one namespace, in a few seconds once the kernels are
compiled.

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Notation](#2-notation)
3. [Definitions](#3-definitions)
4. [Propositions and proofs](#4-propositions-and-proofs)
5. [Algorithms](#5-algorithms)
6. [From the mathematics to the code](#6-from-the-mathematics-to-the-code)
7. [Worked examples](#7-worked-examples)
8. [Validation map](#8-validation-map)
9. [Limitations and departures](#9-limitations-and-departures)

---

## 1. Purpose and scope

The study (`docs/misc/watts_strogatz_study_spec_v1_4.md`, cited as "v1.4") asks for one immutable
experiment manifest with explicit master seeds for graphs and circuits, integer labels $g$, $c$, $r$,
exact decimal or rational tokens for the rates plus the float actually passed to the simulator, and
the trajectory index $u = gC + c$ (v1.4 §7.2); records keyed by $(n, \beta, g)$, $(n, \beta, g, c)$ and
$(n, \beta, g, c, p, r)$ (v1.4 §8); and a fixed deterministic checkpoint schedule (v1.4 §9). This part
implements those identities and the manifest, following the design (`docs/specs/2026-10-17-ws-study-pipeline-design.md`,
§3, cited as "the design"). It does not store anything and does not run anything: the store, the runner,
the observables and the statistics read these identities.

It implements

* **tokens** for $\beta$ and $p$ (`study/identity.py::Token`), and the trajectory key (`TrajectoryKey`);
* **seeds and indices**: `graph_seed`, `run_seed` for the two gate modes, `rate_hash`, `stream_index`;
* **graph records**: `GraphRecord`, `build_graph`, `validate_graph`, `edge_digest`;
* **canonical bytes**: `canonical_json`, `sha256_hex`, `json_digest`, and the version constants;
* **the manifest** (`study/manifest.py::ExperimentManifest`): strict reading, the canonical form and its digest,
  the resolved conditions, $T_{\max}$, the checkpoint rules (geometric, list, table), the `MonitoredCircuit` keywords of v1.4 §8.1,
  `check_extension` for revisions and `effective_checkpoints` for the schedule across them.

It changes no probability law and no random stream (the sampling it names is documented in
[01-ensemble-and-sampling.md](01-ensemble-and-sampling.md)). Departures from v1.4 and the design are listed in §9
(D1, D2, D4 of the design and the new rules N1–N6 introduced here).

## 2. Notation

| symbol | meaning | type | in the code |
|---|---|---|---|
| $x$ | an exact probability in $[0,1]$ (a rate $p$ or a rewiring probability $\beta$) | rational | `Token.value` |
| $\operatorname{tok}(x)$ | the canonical string of $x$ | string | `Token.token` |
| $\operatorname{fl}(x)$ | the double nearest to $x$ | float64 | `Token.float64`, `Token.hex` |
| $n$, $\beta$, $p$ | system size, rewiring probability, measurement rate | int, tokens | `n`, `beta`, `p` |
| $g$, $c$, $r$ | graph, circuit and measurement-replica labels ($r = 0$ only) | integers $\ge 0$ | `g`, `c`, `r` |
| $G_*$, $C$ | graphs per condition, circuits per graph | integers $\ge 1$ | `trials.graphs`, `trials.circuits_per_graph` |
| $s$ | the circuit index stride, $s \ge C$, frozen | integer | `trials.circuit_index_stride`, `manifest.stride` |
| $u$ | the trajectory index $u = g s + c$ | integer | `stream_index(g, c, stride)` |
| $M_g$, $M_c$ | the master graph seed and the master circuit seed | integers in $[0, 2^{32})$ | `seeds.master_graph_seed`, `seeds.master_circuit_seed` |
| $\sigma_p$ | the run seed for rate $p$ | integer or pair | `run_seed(M_c, gate_mode, p)` |
| $h(x)$ | the rate hash: first 8 bytes of BLAKE2b ($\texttt{digest\_size}=8$) of $\operatorname{tok}(x)$, little-endian | $[0, 2^{64})$ | `rate_hash` |
| $T_{\max}(n)$ | the horizon in layers | integer $\ge 1$ | `manifest.layers(n, beta)` |
| $N_T$ | the requested geometric checkpoint count | integer in $[1, 1000]$ | `depth.checkpoints.count` |
| $a_j$ | $T_{\max}^{\,j/(N_T - 1)}$, $j = 0, \dots, N_T - 1$ | real | |
| $S_r(n,\beta)$, also $\mathcal T(n,\beta)$ | the sparse checkpoints of revision $r$ from its own rule and horizon: a sorted set of integers containing $0, 1, T_{\max,r}$ | set | `manifest.checkpoints(n, beta)` |
| $E_r(n,\beta)$ | the effective schedule used for work under revision $r$: $(S_1 \cup \dots \cup S_r) \cap [0, T_{\max,r}]$ (ruling R-CK) | set | `manifest.effective_checkpoints(revisions, n, beta)` |
| $\mathcal P(n,\beta)$ | the rates run at $(n, \beta)$ | set of tokens | `manifest.rates(n, beta)` |
| $\operatorname{dig}$ | SHA-256, written as 64 lowercase hexadecimal digits | string | `sha256_hex` |

## 3. Definitions

### 3.1 Tokens (v1.4 §7.2, §9: "rational/decimal tokens plus actual float bits")

A **token** is an exact rational number $x \in [0,1]$ (`Token`, a frozen dataclass holding a
`fractions.Fraction`). It is built by `Token.parse` from

* a decimal string (`"0.1"`, `".25"`, `"1e-2"`; at most a four-digit exponent), or a rational string
  (`"1/7"`): read exactly; surrounding white space is ignored;
* a JSON number: an `int` as is; a `float` through its shortest round-trip decimal $\texttt{repr}$ (so `0.1` is $1/10$, not the double
  $0.1000000000000000055\ldots$); booleans, NaN and infinities are refused;
* a `Fraction` or another token.

Out-of-range values, malformed strings and wrong types raise `ValueError` with the field name.

The **canonical string** $\operatorname{tok}(x)$ is: if the denominator of $x$ in lowest terms has no prime factor besides 2 and 5,
the decimal expansion of $x$ with no exponent, no trailing zeros and no trailing point (`"0"`, `"1"`,
`"0.05"`, `"0.375"`); else `"num/den"` in lowest terms (`"1/3"`, `"2/3"`). Two tokens are equal exactly when their canonical strings are
(Proposition 1). The float handed to the simulator is $\operatorname{fl}(x)$ = `float(x)`, and the bit-exact
record is `float.hex()` of it. `Token.is_exact` says whether $\operatorname{fl}(x) = x$.

### 3.2 Experiment identifiers and trajectory keys (v1.4 §8)

An `experiment_id` is 1 to 64 characters from `A-Z a-z 0-9 . _ -`, starting with a letter or digit
(`identity.check_experiment_id`). A **trajectory key** (`TrajectoryKey`, frozen and hashable) is
$(\texttt{experiment\_id}, n, \beta, p, g, c, r)$ with $\beta, p$ tokens and $r = 0$ (design D1; the field keeps its place
for a future versioned extension). Its canonical string is
`"<id>:n=16:beta=0.1:p=1/3:g=3:c=1:r=0"` and `TrajectoryKey.parse` inverts it; the colon and the equals sign occur in no
token and no identifier, so the string is unambiguous.

### 3.3 Seeds (v1.4 §7.2; design §3)

* **Graph seed.** $\texttt{graph\_seed}(M_g, g) = (M_g, g)$, a pair of unsigned 32-bit words. This is the native collection convention
  ("`realizations` builds graph $i$ from `[seed, i]`", `api.py::_build_graph`) and the example of v1.4 §8.1. It does not depend
  on $p$, so one collection per $(n,\beta)$ serves every rate. It does not depend on $n$ or $\beta$ either: graphs of different
  sizes and rewiring probabilities share random numbers (common random numbers; comparisons across $\beta$ or $n$ are paired).
* **Run seed.** $\sigma_p = M_c$ in `gate_mode="shared"` (the native coupling: trajectory $u$ draws the same gates and the same mask
  uniforms at every $p$, document 01 Proposition 12), and $\sigma_p = (M_c, h(p))$ in `gate_mode="independent"` (design D4: gates and
  masks are independent across rates; nothing is nested). $h(p)$ is the first 8 bytes of
  `blake2b(token.encode("ascii"), digest_size=8)`, read little-endian: the convention the engine uses for probe names, not a
  truncation of the 64-byte digest.
* **Range.** Seeds and labels are in $[0, 2^{32})$, so that $(M, g)$ is a two-word `SeedSequence` entropy.

### 3.4 The stream index (v1.4 §7.2; design D2)

$u(g, c) = g\,s + c$ with the stride $s$ of the manifest and $0 \le c < s$ (`identity.stream_index`; the inverse is
`split_stream_index`). The engine runs a condition with `circuits_per_graph = s` and `indices = [u(g, c) for the wanted pairs]`, so trajectory $u$
runs on graph $\lfloor u/s\rfloor = g$ and draws its circuit from stream $u$ of the run seed. The design's default stride is $C$; a
larger frozen $s$ lets circuits $C, \dots, s-1$ be added later without renumbering. The engine looks the graph up by position in the list of
graphs it is given (`MonitoredCircuit.run(indices=...)`: each index must be below `len(edges) * circuits_per_graph`), so a run of the
trajectories of the graph labels $g_0, \dots, g_1 - 1$ must be given the graphs $0, \dots, g_1 - 1$ in that order; the earlier ones are not simulated.

### 3.5 Graph records and the edge digest (v1.4 §2.2 "Tracking", §8)

`build_graph(n, beta, g, M_g)` calls `graphs.watts_strogatz(n, 4, float(beta), seed=(M_g, g))`, validates the result
(`validate_graph`: canonical sorted form, $2n$ edges, minimum degree 2, $C(n,2)$ at $\beta = 0$; for $n = 5$ the checks leave $K_5$) and returns a
`GraphRecord`: $n$, the token $\beta$, $g$, the seed pair, the `generator` version string
(`GRAPH_GENERATOR_VERSION = "ws-offset-major-reject-1"`), the `edge_order` (`"offset-major"`), the read-only edge array of shape $(2n, 2)$
and the **edge digest**: the SHA-256 of the bytes `b"SGF2EDGES1\x00"` (11 bytes), then $n$ as an unsigned 32-bit little-endian integer,
then the edge rows $a_1, b_1, a_2, b_2, \dots$ as signed 32-bit little-endian integers, in the canonical order ($a < b$, rows increasing).
`GraphRecord.verify` re-checks the digest and the graph; `regenerate` rebuilds the graph from the seed and requires the same record.

The other version constants of `study/identity.py`: `CLIFFORD_ENUMERATION_VERSION` (the numbering $j = 16 s + b$), `DYNAMICS_VERSION`,
`ROWSPACE_FORMAT_VERSION` (`"SGF2RS1"`), `STORE_SCHEMA_VERSION`; `version_record()` collects them with `mipt.RANDOMNESS_CONTRACT` and the package version.

### 3.6 Canonical JSON and digests

`identity.canonical_json(obj)` writes sorted keys, no white space, ASCII only, finite numbers (`-0.0` as `0.0`); tokens as
their canonical strings (never floats); numpy scalars and arrays as plain numbers and lists; tuples as lists.
NaN, infinities, fractions, bytes and non-string keys are refused. `json_digest(obj)` is $\operatorname{dig}$ of its UTF-8 bytes.

### 3.7 The manifest (design §3)

`ExperimentManifest.from_json(dict | str | Path)` reads a manifest strictly: unknown keys are refused with the allowed ones listed, duplicate
JSON keys and the constants NaN and Infinity are refused, types are checked (an integer field refuses `16.0` and `true`), ranges are checked, and every
message names the field (`manifest.trials.graphs: must be at least 1, got 0`). A string is JSON text and must start with `{`; a
`pathlib.Path` is read as a file. The sections:

| key | type, default | constraints and meaning |
|---|---|---|
| `schema` | string, optional | must equal `"sparsegf2.study.manifest/1"` |
| `experiment_id` | string, required | §3.2 |
| `label` | `"exploratory"` (default) or `"confirmatory"` | keeps exploratory and confirmatory runs apart (v1.4 §9) |
| `conditions` | list of blocks, required | each block `{"n": [...], "beta": [...], "p": [...]}`: the product of its lists; $n \ge 5$ integers; $\beta$, $p$ tokens; no list repeats a value |
| `depth.layers` | exactly one of `{"per_n": a}`, `{"fixed": T}`, `{"table": {"16": 128, ...}}` | $T_{\max} = a n$, $T$, or the table (keys are the sizes in use, exactly); every value an integer $\ge 1$ |
| `depth.checkpoints` | exactly one of `{"count": N}`, `{"list": [...]}`, `{"table": {"16": [...], ...}}` | $1 \le N \le 1000$ (§3.9); a list: one non-empty list of integers $\ge 0$ for every size, the entries above a size's $T_{\max}$ dropped for that size; a table: one list (possibly empty) of integers $\ge 0$ per size, the keys exactly the sizes in use, every entry $\le T_{\max}(n)$ |
| `trials.graphs`, `trials.circuits_per_graph` | integers $\ge 1$, required | $G_*$, $C$ |
| `trials.circuit_index_stride` | integer $\ge C$, default $C$ | the frozen stride $s$ |
| `trials.measurement_replicas` | must be 1 | design D1: the native engine draws gates and masks from one stream |
| `seeds.master_graph_seed`, `seeds.master_circuit_seed` | integers in $[0, 2^{32})$, required | must differ (N1) |
| `seeds.gate_mode` | `"shared"` (default) or `"independent"` | §3.3 |
| `dynamics.stop_when_purified` | bool, default false | passed to the engine |
| `observables.tableau_snapshots` | bool, default true | `tableau_snapshot` probe at $\mathcal T(n,\beta)$ |
| `observables.contiguous_distance` | bool, default true | `contiguous_distance` probe at $\mathcal T(n,\beta)$ |
| `observables.uniform_erasure` | `null`, or `{"samples": 200, "exact_budget": 5000}` (the default) | `erasure_first_failure(samples)`; `uncorrectable_region_counts(budget=exact_budget)`, with `exact_budget` $= 0$ (no exact counts) or at least the largest $n$ |
| `observables.correlated_erasure` | list, default empty | each entry: `couplings` (required), `sizes` (`null` = every $m = 1..n-1$, a list, or a range string `"a:b[:step]"`; all within $[1, n-1]$ for every $n$), `samples` (100), `thin` (1), `geometry` (`"interaction"`), `method` (`"auto"`), `burn_in` (`null`), `tempering` (true), `ladder` (`"auto"`), `max_step` (`null`), `name` (`null`); validated by building the engine probe |
| `storage` | `{"graphs_per_unit": 8, "compression_level": 6}` | compression level $0..9$ |
| `resources` | `{"num_threads": null, "max_unit_trajectories": 4096}` | threads $\ge 1$ or `null` |
| `analysis` | `{"peak": {"rule": "plateau", "z": 1.0, "sensitivity_z": [0, 1, 2], "bootstrap": 200}, "bootstrap": 1000, "censoring": "survival"}` | the predeclared peak and censoring rules (v1.4 §5.8, §6); `peak.rule` must be `"plateau"` and `censoring` `"survival"` (`peak.z` and every `sensitivity_z` entry $\ge 0$, bootstraps $\ge 0$) |

`to_json()` writes **every** key, defaults included, in canonical JSON; `digest()` is its SHA-256. Reading also refuses what no key
alone can show: a duplicate condition, a layer table or a checkpoint table that does not cover exactly the sizes in use, a checkpoint-table entry beyond its size's
$T_{\max}$, correlated sizes outside $[1, n-1]$, and two correlated-erasure probes with the same name. `manifest.warnings` lists what is allowed but doubtful: $n = 5$ (the complete graph),
snapshots with `stop_when_purified`, a work unit larger than `max_unit_trajectories`, and checkpoint-list entries above $T_{\max}$ for every size (they are dropped everywhere and have no effect).

### 3.8 Conditions and their canonical factorization

`conditions()` resolves the blocks into `Condition(n, beta, rates)` triples sorted by $n$, then $\beta$, with the rates of each pair sorted and
merged over the blocks. A repeated $(n, \beta, p)$ is refused, and so are two different tokens with the same float64 (they would build identical
graphs or run identical trajectories under two names). The stored blocks are the **canonical factorization** of the resolved set: group the
pairs $(n,\beta)$ by their rate set $P$; within one $P$, group the sizes $n$ by their set of $\beta$ values; each group is one block
$\lbrace n\rbrace_{\text{group}} \times B \times P$. The factorization is a function of the set of triples alone (Proposition 9), so equivalent manifests have one
canonical text and one digest.

### 3.9 Depth: $T_{\max}$, the checkpoint rule and the effective schedule (v1.4 §9; design §3, ruling R-CK)

$T_{\max}(n)$ is $a n$, the fixed $T$, or the table entry; it is the same for every rate (v1.4 §1). The global checkpoints of the engine are every layer
(`checkpoints="every_layer"`, so $k_t$ and the exact first-passage depths are observed everywhere); the **sparse checkpoints** $S(n, \beta)$ of a revision
(written $\mathcal T(n,\beta)$ in v1.4) are where the expensive probes run. The rule of a revision has three forms. In each, $T = T_{\max}(n)$ and $S$ is sorted, without repeats, contained in $[0, T]$ and contains $0, 1, T$.

* **`{"count": N_T}`, the geometric rule.** With $q = N_T - 1$:
  1. if $N_T \ge 2$, for $j = 0, \dots, q$ let $a_j = \exp(j \ln T / q) = T^{j/q}$ and $k_j = \lfloor a_j + \tfrac12\rfloor$ (nearest integer, **ties upward**), clamped to $[0, T]$;
  2. $S = \lbrace 0, 1, T\rbrace \cup \lbrace k_j\rbrace$. For $N_T = 1$ it is $\lbrace 0, 1, T\rbrace$.

  The rounding is computed exactly in integers (Proposition 10, `study/manifest.py::round_half_up_power`).
* **`{"list": [d_1, ..., d_k]}`, one list for every size.** $S(n) = \lbrace 0, 1, T\rbrace \cup \lbrace d_i : d_i \le T\rbrace$. An entry above this size's horizon is **dropped** for this size; it is not an error, because one list has to serve sizes with different horizons. An entry above every horizon has no effect, which `manifest.warnings` says.
* **`{"table": {"16": [...], "24": [...]}}`, one list per size.** $S(n) = \lbrace 0, 1, T\rbrace \cup \lbrace\text{the entries of the list of } n\rbrace$. The keys must be exactly the sizes in use (a missing or an extra size is refused), the list of a size may be empty (just $0, 1, T$), and every entry must lie in $[0, T_{\max}(n)]$: a list written for one size that reaches beyond its horizon is a mistake, so it is refused rather than dropped.

**The effective schedule (ruling R-CK).** A study is revised (§3.11), and a revision may raise $T_{\max}$ and change the rule. If the work under revision $r$ used only its own schedule $S_r$, then raising the horizon under the geometric rule would move every checkpoint: the depths stored for the trajectories of the earlier revisions would no longer be asked for. The ruling therefore fixes the schedule **used for work** under revision $r$ as
$$
E_r(n,\beta) \;=\; \big(S_1(n,\beta) \cup \dots \cup S_r(n,\beta)\big) \cap [0, T_{\max,r}(n,\beta)],
$$
where $S_i$ is the own schedule of revision $i$ at $(n,\beta)$ and a revision that does not contain the condition contributes nothing. `manifest.checkpoints(n, beta)` is $S_r$ of one manifest; `effective_checkpoints(revisions, n, beta)` is $E_r$ for a list of revisions, oldest first (for one revision it is that revision's own schedule). Along a valid chain of revisions the intersection removes nothing, $E_r = S_1 \cup \dots \cup S_r$, and $E_r \supseteq E_{r-1}$ (Proposition 12): a depth, once scheduled, stays scheduled.

### 3.10 The engine keywords (v1.4 §8.1)

`monitored_circuit_kwargs(n, beta, p)` returns, for a condition of the manifest, the `MonitoredCircuit` keywords
`num_system=n`, `circuits_per_graph=s`, `layers=`$T_{\max}(n)$, `measurement_rate=`$\operatorname{fl}(p)$, `reference="purification"`,
`schedule="random_edges"`, `gates_per_layer=n // 2`, `random_orientation=False`, `scramble_gates=0`, `scramble_matchings=0`,
`scramble_layers=0`, `periodic=True`, `checkpoints="every_layer"` and `stop_when_purified` from `dynamics`. The caller adds `edges` (the graph records of the unit,
`[record.edges, ...]`) and `probes`.

### 3.11 Revisions (design §3 "Revisions" and ruling R-CK)

A store keeps every revision. `check_extension(old, new)` returns `None` when `new` only extends `old`, else a message naming the **first** forbidden change, checked in this order:

1. `schema`, `experiment_id`, `label` unchanged;
2. the seeds and `gate_mode` unchanged (the pairs of §3.3 fix every recorded graph and trajectory);
3. the stride unchanged (it fixes every $u$); `dynamics` and `measurement_replicas` unchanged;
4. `trials.graphs` and `trials.circuits_per_graph` not smaller;
5. every old $(n, \beta, p)$ still present;
6. $T_{\max}$ not smaller, for every old $(n, \beta)$ (a larger one is allowed, size by size, whatever the checkpoint rule becomes);
7. every observable switched on stays on with its settings (a `correlated_erasure` entry is identified by its full settings; adding entries is allowed).

The checkpoint rule is **not** checked: `count`, `list` and `table` may replace one another and a count may fall as well as rise, because the schedule used for work is the union of the schedules of all revisions (§3.9, Proposition 12), which never drops a depth that an earlier revision scheduled.
Changes of `storage`, `resources` and `analysis` are allowed: they touch no recorded value.

`effective_checkpoints(revisions, n, beta)` first requires a valid chain: at least one revision, every entry a manifest, and `check_extension(revisions[k-1], revisions[k])` equal to `None` for every consecutive pair. Otherwise it raises `ValueError("revision k does not extend revision k-1: <message>")` naming the first pair that fails and its first forbidden change (so a list whose horizon decreases is refused with the lowered $T_{\max}$ in the message). Pairwise validity suffices because extensions compose (Proposition 11).

## 4. Propositions and proofs

### 4.1 Tokens

**Proposition 1 (canonical strings).** Let $x \in [0,1]$ be rational, $x = a/d$ in lowest terms. If $d = 2^\alpha 5^\gamma$ put $k = \max(\alpha,\gamma)$. Then
$10^k x$ is an integer, not divisible by 10 when $k \ge 1$, so the decimal expansion of $x$ has exactly $k$ fractional digits and its last digit is nonzero. Hence the canonical
string is the unique decimal string without trailing zeros, the canonical string of a rational with a prime factor of $d$ outside $\lbrace2,5\rbrace$ is `"a/d"`,
the two kinds never coincide, `Token.parse(tok(x)) = x`, and $x \mapsto \operatorname{tok}(x)$ is injective.

*Proof.* $10^k x = a\,2^{k-\alpha}5^{k-\gamma}$ is an integer because $k \ge \alpha, \gamma$. Suppose $k \ge 1$. If $\alpha \ge \gamma$ then $k = \alpha \ge 1$, so $d$ is even and
$a$, coprime to $d$, is odd; $10^kx = a\,5^{k-\gamma}$ is odd, so not divisible by 10. If $\gamma > \alpha$ then $k = \gamma \ge 1$ and $5 \mid d$, so $5 \nmid a$ and
$10^kx = a\,2^{k-\alpha}$ (the power of 5 is $5^{k-\gamma} = 1$) is not divisible by 5. In both cases the last digit of $10^kx$ is not 0, so no
shorter decimal string represents $x$. A decimal string and a rational string never denote the same string; a decimal string is parsed exactly by `Fraction`, and a rational string
`"a/d"` in lowest terms with a prime factor of $d$ other than 2, 5 is not the value of any decimal string. Injectivity: equal strings parse to equal values, and the value
determines the string. ∎

**Proposition 2 (JSON numbers).** If a JSON author types a decimal with at most 15 significant digits and a parser turns it into a double, then reading that double through
$\texttt{repr}$ returns the typed decimal, so `Token.parse(float(s)) = Token.parse(s)`.

*Proof.* `repr(x)` of a double is the shortest decimal string that converts back to $x$. A decimal $s$ with at most 15 significant digits converts to a double $x$ and back to a 15-digit decimal as $s$
again (a double carries 15.95 decimal digits: $\mathrm{DBL\_DIG} = 15$). If some shorter decimal $s'$ also mapped to $x$, then $s'$, having at most 15 digits, would round-trip to itself as well, so
$s' = s$ as numbers. The shortest string is therefore $s$ (up to the form of the exponent and trailing zeros, which `Token.parse` normalizes). ∎ (For 16 or 17 digits the shortest round-trip decimal
may differ from the typed one; the manifest example and the tests use at most 15, and exact rational strings avoid the question.)

**Proposition 3 (the float).** For a token $x = a/d$, `Token.float64` is `float(Fraction(a, d))`, the true quotient $a/d$ correctly rounded to the nearest double (ties to even): CPython's
`int.__truediv__` is correctly rounded. Hence $\lvert\operatorname{fl}(x) - x\rvert \le 2^{-54}$ for $x \in [0,1]$, `float.fromhex(Token.hex) == Token.float64`, and the token (not the float) is the identifier:
different tokens can share a double (for instance `"0.1"` and `"0.1000000000000000055511151231257827"`), which `conditions()` refuses.

*Proof.* The correct rounding of integer division is documented in CPython (`long_true_divide`); the bound is half a unit in the last place of a double in $[0,1]$ ($2^{-53}$ spacing at most,
below 1). The test `test_float64_is_the_nearest_double_and_hex_round_trips` checks that the neighboring doubles are no closer. ∎

### 4.2 Seeds and indices

**Proposition 4 (graph seeds).** (a) For $M, g \in [0,2^{32})$ the seed $(M, g)$ is the entropy `[M, g]` of two `uint32` words, the same as the native `[seed, i]`.
(b) `SeedSequence([M, 0])` and `SeedSequence(M)` generate the same state, because numpy pads the entropy with zero words; `SeedSequence([M, g])` with $g \ge 1$ differs. (c) Consequently, if the master circuit
seed equalled the master graph seed, graph 0 would be built from the very state words that key the circuit streams of the run; the manifest refuses equal master seeds (N1).

*Proof.* (a) is the definition. (b) is checked numerically in `test_graph_zero_shares_its_seed_with_the_plain_master_seed` (and follows from the zero padding of the entropy pool). (c) is (b) with $g = 0$. ∎
That distinct pairs give distinct generated states is a property of the hash behind `SeedSequence`; it is checked on a grid (`test_graph_seed_gives_the_stream_the_graph_generator_uses`), not proved.

**Proposition 5 (the stream index).** For a stride $s \ge 1$ the map $(g, c) \mapsto g\,s + c$ is a bijection from $\mathbb N \times \lbrace 0, \dots, s-1\rbrace$ onto $\mathbb N$ with inverse
$u \mapsto (\lfloor u/s\rfloor, u \bmod s)$. An engine run with `circuits_per_graph = s` gives trajectory $u$ the graph $\lfloor u/s\rfloor$ and the circuit stream $u$. Therefore the graph and
the circuit of the pair $(g,c)$ depend on $(g, c, s)$, the seeds and the protocol only: enlarging $G_*$, or $C$ up to $s$, never changes the index, the graph or the circuit of an existing pair. Changing
$s$ changes $u$ for every pair with $g \ge 1$, which is why $s$ is frozen.

*Proof.* Division with remainder. For the engine statement see `_kernel/ensemble.py` (module docstring: trajectory $t$ runs on graph $t$ // `circuits_per_graph`; `stream_state(key, index)`). If
$s' \ne s$ and $g \ge 1$ then $g s' + c \ne g s + c$. ∎ The test `test_the_engine_runs_trajectory_u_on_graph_u_over_stride` checks the graph assignment, and
`tests/test_study_sampling.py::test_unrelated_choices_leave_the_circuit_unchanged` checks that appended graphs leave trajectory $u$ unchanged.

**Proposition 6 (the two gate modes).** In `shared` mode all rates of a condition use the seed $M_c$; by document 01 (Proposition 12) trajectory $u$ then has the same graph, gates and mask uniforms at every rate, and masks are nested in $p$.
In `independent` mode the seeds $(M_c, h(p))$ of two different tokens differ unless $h(p) = h(p')$, a collision of a 64-bit hash (probability $2^{-64}$ for a given pair, not zero); then
the streams are independent, nothing is nested and each rate is a separate ensemble.

*Proof.* The first statement is Proposition 12 of document 01. The second follows from the definition of $h$ and from the independence of `SeedSequence` streams of different entropy (§9, ideal). ∎

### 4.3 Records and canonical text

**Proposition 7 (the edge digest).** The payload of the edge digest is a fixed 15-byte header (the 11-byte magic and the 4-byte $n$) followed by $8m$ bytes of edge rows, so equal payloads
imply equal $n$ and equal edge arrays, and for canonical edge lists equal graphs give equal payloads. Two different graphs have different digests unless SHA-256 collides.

*Proof.* The header has fixed width, so $n$ and the number of rows $m$ are read from the payload without ambiguity, and each row is two 4-byte signed integers. The edge list is canonical (each edge as $a<b$, rows
strictly increasing, enforced by `edge_digest`), so it is a function of the edge set. ∎

**Proposition 8 (canonical JSON).** For data built from `None`, booleans, integers, finite floats, strings, lists and dictionaries with string keys, $\operatorname{json.loads}(\texttt{canonical\_json}(x)) = x$ (tuples
read back as lists). Hence `canonical_json` is injective on such data, and equal data give equal text whatever the key order of the source and the white space of the file.

*Proof.* Floats are written with Python's shortest round-trip $\texttt{repr}$, integers exactly, strings with ASCII escapes; the separators and the sorted keys are deterministic. The parse inverts each token. ∎
`-0.0` is normalized to `0.0`, the one value where equal numbers have different text.

**Proposition 9 (the manifest digest).** `digest()` is a function of the *content* of the manifest: it is invariant under the order of keys, white space, the spelling of tokens (`"0.1"`, `0.1`, `"1/10"`), the order of the entries of the lists `n`, `beta`, `p`,
explicit defaults versus omitted ones, repeated entries of a checkpoint list, and the way the conditions are factored into blocks; it changes with every change of a value.

*Proof.* Reading produces a manifest object in which tokens are exact, defaults are filled in, the lists of a block are sorted, the checkpoint list is sorted and deduplicated, and the blocks are replaced by the canonical
factorization of §3.8; `to_dict` is a function of that object and `digest` of its canonical text (Proposition 8). The canonical factorization depends on the set of triples only: it is built from `conditions()`, which is defined by the set. It
represents the same set: each pair $(n,\beta)$ has exactly one rate set $P$, so lies in exactly one group; within the group of that $P$, each $n$ has exactly one $\beta$-set $B$; the blocks $\lbrace n \text{'s with } B\rbrace \times B \times P$ are therefore pairwise
disjoint and their union is the set. A changed value changes the object and so, by injectivity (Proposition 8), the text; the hash distinguishes them except for a SHA-256 collision. ∎

### 4.4 The checkpoint rule

**Proposition 10 (exact rounding).** For integers $T \ge 1$, $q \ge 1$ and $0 \le j \le q$ let $a = T^{j/q}$, $S = 2^q T^j$ and $r = \lfloor S^{1/q}\rfloor$ (the largest integer with $r^q \le S$). Then
$$
\lfloor a + \tfrac12\rfloor = \Big\lfloor \frac{r + 1}{2} \Big\rfloor .
$$
Moreover $a$ is never a half-integer (so a tie never occurs for an integer horizon), $a_0 = 1$ and $a_q = T$ exactly, $a_j < a_{j+1}$ for $T \ge 2$, and the schedule satisfies:

* (a) $\mathcal T$ is sorted, without repeats, contained in $[0, T]$, contains $0, 1, T$, and has at most $\min(N_T + 1, T + 1)$ elements for $N_T \ge 2$ (3 for $N_T = 1$);
* (b) every $a_j$ is within $\tfrac12$ of an element of $\mathcal T$;
* (c) (refinement) for an integer $k \ge 1$ the count $N_T' = k(N_T - 1) + 1$ gives a schedule that contains the one of $N_T$.

*Proof.* $S^{1/q} = 2a$, so $r = \lfloor 2a\rfloor$. Write $2a = r + f$, $0 \le f < 1$. Then $a + \tfrac12 = (r + 1 + f)/2$. If $r + 1$ is even this is $(r+1)/2 + f/2$ with $f/2 < \tfrac12$, so the floor is $(r+1)/2$.
If $r + 1$ is odd, $r + 1 = 2i + 1$, it is $i + (1 + f)/2$ with $(1+f)/2 \in [\tfrac12, 1)$, so the floor is $i = \lfloor (r+1)/2\rfloor$. No tie: $a = k + \tfrac12$ would give $2^q T^j = (2k+1)^q$,
an even number equal to an odd one (for $q \ge 1$). $a_0 = T^0 = 1$ and $a_q = T^{q/q} = T$ exactly, and $T^{j/q}$ is increasing in $j$ for $T \ge 2$, so the rounded values are non-decreasing. (a) follows
from the definition: the set is $\lbrace0\rbrace \cup \lbrace k_j\rbrace \cup \lbrace 1, T\rbrace$ with $k_0 = 1$ and $k_q = T$ already among the $k_j$. (b): $\lvert k_j - a_j\rvert \le \tfrac12$. (c): $a'_{kj} = T^{kj/(kq)} = T^{j/q} = a_j$, the same real
number rounded by the same rule. ∎ The computation in `study/manifest.py::_half_up_root` uses a floating-point estimate of $r$ and then corrects it exactly with integer powers, so no value can cross a rounding boundary by floating-point error.

### 4.5 Revisions

**Proposition 11 (what an accepted revision guarantees).** (a) If `check_extension(old, new)` is `None`, then for every trajectory $(n, \beta, p, g, c)$ of `old` (with $g < G_*^{\mathrm{old}}$, $c < C^{\mathrm{old}}$):
the graph record (same $n$, $\beta$, seed pair, generator), the stream index $u$, the run seed and the float $\operatorname{fl}(p)$ are the same in `new`; the condition still exists in `new` with $T_{\max}^{\mathrm{new}} \ge T_{\max}^{\mathrm{old}}$ and the old
trajectory is the prefix of its extension (document 01, Proposition 13); every observable recorded under `old` has the same settings in `new`. (b) Extension is reflexive and transitive: `check_extension(a, a)` is `None`, and if `check_extension(a, b)` and `check_extension(b, c)` are `None` then so is `check_extension(a, c)`.
The checkpoint rule is not part of the guarantee: it may change freely (Proposition 12 says what survives).

*Proof.* (a) Each statement is one of the checks of §3.11 in contrapositive: seeds and mode equal, stride equal (Proposition 5), `dynamics` equal (the engine keywords are those of `old`), conditions a superset, horizons not smaller,
observables unchanged. (b) Each check is reflexive and transitive: equality of fields, $\ge$ between integers, inclusion of sets of conditions, "if on then on", "if recorded then equal" and "every old entry is among the new ones"
are preorders; the horizon check compares one condition across the manifests and that condition is present in all three by the inclusion of conditions. ∎ Not guaranteed: that the new trajectories have been run, and that `storage`, `resources` or `analysis` are unchanged (they are free to change).

**Proposition 12 (the effective schedule).** Let $m_1, \dots, m_r$ be revisions with `check_extension(`$m_{i-1}, m_i$`)` equal to `None` for $i = 2, \dots, r$, and let $(n, \beta)$ be a condition of $m_r$; write $S_i$ for the own schedule of $m_i$ at $(n,\beta)$ (`m_i.checkpoints(n, beta)`) and $T_i$ for its horizon,
and let $m_a$ be the first revision that contains $(n,\beta)$. Then

* (a) every $m_i$ with $i \ge a$ contains $(n,\beta)$ and no earlier one does, and $T_a \le T_{a+1} \le \dots \le T_r$;
* (b) $E_r = S_a \cup \dots \cup S_r$: the intersection with $[0, T_r]$ in the definition removes nothing. It is sorted, lies in $[0, T_r]$ and contains $0, 1, T_r$;
* (c) $E_r \supseteq S_i$ for every $i \ge a$, and for $r > a$ the effective schedule of the chain $m_1, \dots, m_{r-1}$ at $(n,\beta)$ is contained in $E_r$: $E_r = E_{r-1} \cup S_r \supseteq E_{r-1}$;
* (d) if only $m_r$ contains $(n,\beta)$ ($a = r$, in particular for one revision), $E_r = S_r$.

*Proof.* (a) Check 5 of §3.11 says that every condition $(n,\beta,p)$ of $m_i$ is a condition of $m_{i+1}$, so the presence of $(n,\beta)$ passes from each revision to the next: the revisions that contain it are $m_a, \dots, m_r$.
Check 6 says $T_i \le T_{i+1}$ for the old condition $(n,\beta)$, for each $i \ge a$. (b) Every own schedule lies in $[0, T_i]$ by construction (§3.9: geometric points are clamped to $[0, T]$, list entries above $T$ are dropped, table entries above $T$ are refused),
and $T_i \le T_r$ by (a), so $S_i \subseteq [0, T_r]$ and $E_r = \bigcup_{i \ge a} S_i \cap [0, T_r] = \bigcup_{i \ge a} S_i$. Since $0, 1, T_r \in S_r$, they belong to $E_r$. (c) A union contains each of its terms; and with $E_{r-1} = \bigcup_{a \le i < r} S_i$ (part (b) applied to the chain $m_1, \dots, m_{r-1}$,
which is also valid), $E_r = E_{r-1} \cup S_r$. (d) is the definition with a single term. ∎ *Consequences.* A depth stored under any earlier revision is a depth of every later effective schedule (it is never dropped), whatever the later rule is: raising $T_{\max}$ under the geometric rule moves the own schedule but not the union.
The set $E_r$ is also never above the latest horizon. Because of check 5 and check 6 a list that is not a valid chain is refused rather than reinterpreted: with a falling horizon, "never dropped" and "never above the latest $T_{\max}$" would contradict each other, with a removed condition the union would silently forget it,
and with changed seeds the stored depths would belong to other trajectories. Not guaranteed: that a trajectory run under an earlier revision has been observed at the depths that only a later revision added; a trajectory is deterministic (document 01, Proposition 13), so the runner can rerun it to the longer horizon and observe it at all of $E_r$.

## 5. Algorithms

### 5.1 Parsing a token

1. A `Token` returns itself; a `Fraction`, an `int` (not `bool`) and a finite `float` are converted ($\texttt{Fraction}(\texttt{repr}(x))$ for floats); other types are refused.
2. A string is stripped; it must match `[+-]?[0-9]+/[0-9]+` (rational; denominator 0 refused) or `[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]{1,4})?` (decimal; exact via `Fraction`). ASCII digits only.
3. The value must lie in $[0,1]$.
4. `token` is computed by factoring the denominator into 2s and 5s (Proposition 1); `float64 = float(value)`; `hex = float64.hex()`.

Cost: negligible. Exactness: exact.

### 5.2 Reading a manifest

1. Text: `json.loads` with a hook that refuses duplicate keys and the constants NaN and Infinity; a file is read as UTF-8.
2. Each object is checked against its allowed and required keys; each value against its type and range (`_object`, `_int`, `_real`, `_flag`, `_choice`, `_list`, `_token`); the message carries the path of the field.
3. Sections are built (frozen dataclasses); the engine validates each correlated-erasure entry by constructing its probe.
4. The constructor resolves the conditions (duplicates refused), replaces the blocks by the canonical factorization and checks the cross-field rules of §3.7.

### 5.3 The exact checkpoints

For $j = 0, \dots, q$: $S = 2^q T^j$; estimate $r \approx \lfloor 2 e^{j \ln T / q}\rfloor$ in floating point; decrease $r$ while $r^q > S$ and increase it while $(r+1)^q \le S$ (exact big-integer powers, usually zero steps); the checkpoint is $\lfloor (r+1)/2\rfloor$,
clamped to $[0, T]$. Cost $O(N_T)$ big-integer powers of $O(q + j\log_2 T)$ bits; $N_T \le 1000$.
For the list form (`CheckpointRule.resolve`) take the entries not above $T$ and add $0, 1, T$; for the table form look up the list of the size $n$ (refusing a size without an entry and an entry above $T$) and add $0, 1, T$; sort and remove repeats.

### 5.4 `check_extension`

The checks of §3.11, in order, return the message of the first failure; the condition checks compare sets of $(n, \beta, p)$, cost linear in the number of conditions. No checkpoint set is compared.

### 5.5 `effective_checkpoints`

1. Refuse an empty list of revisions and any entry that is not an `ExperimentManifest` (`TypeError`).
2. For $k = 1, \dots, r-1$ run `check_extension(revisions[k-1], revisions[k])` and refuse a failure with `revision k does not extend revision k-1: <message>`.
3. Read the horizon $T_r$ of the last revision at $(n, \beta)$ (a `ValueError` unless $(n,\beta)$ is one of its conditions).
4. Collect `checkpoints(n, beta)` of every revision that contains the condition, keep the points in $[0, T_r]$ (all of them, by Proposition 12; the filter is the definition and costs nothing) and sort.

Cost: $r - 1$ extension checks plus $r$ schedule computations.

## 6. From the mathematics to the code

| mathematics | code |
|---|---|
| token $x$, $\operatorname{tok}(x)$, $\operatorname{fl}(x)$, hex (§3.1) | `study/identity.py::Token` (`parse`, `token`, `float64`, `hex`, `is_exact`) |
| rate hash $h(x)$ (§3.3) | `study/identity.py::rate_hash` |
| experiment id (§3.2) | `study/identity.py::check_experiment_id` |
| trajectory key (§3.2) | `study/identity.py::TrajectoryKey` (`parse`, `stream_index`) |
| graph seed, run seed (§3.3) | `study/identity.py::graph_seed`, `run_seed`; `ExperimentManifest.graph_seed`, `run_seed` |
| $u = g s + c$ and inverse (§3.4) | `study/identity.py::stream_index`, `split_stream_index`; `ExperimentManifest.stream_index` |
| $C(n,2)$, $L_n$ (document 01) | `study/identity.py::circulant_edges`, `original_edge_list` |
| graph record (§3.5) | `study/identity.py::GraphRecord`, `build_graph`, `validate_graph`, `edge_digest` |
| versions (§3.5) | `study/identity.py::GRAPH_GENERATOR_VERSION`, `CLIFFORD_ENUMERATION_VERSION`, `DYNAMICS_VERSION`, `ROWSPACE_FORMAT_VERSION`, `STORE_SCHEMA_VERSION`, `version_record` |
| canonical JSON, digests (§3.6) | `study/identity.py::canonical_json`, `sha256_hex`, `json_digest` |
| the manifest (§3.7) | `study/manifest.py::ExperimentManifest` and its sections `ConditionBlock`, `Layers`, `CheckpointRule`, `Depth`, `Trials`, `Seeds`, `Dynamics`, `UniformErasure`, `CorrelatedErasure`, `Observables`, `Storage`, `Resources`, `Peak`, `Analysis` |
| conditions, canonical factorization (§3.8) | `ExperimentManifest.conditions`, `rates`, `study/manifest.py::_canonical_blocks` |
| $T_{\max}$, own schedule $S$ (§3.9) | `ExperimentManifest.layers`, `checkpoints`, `study/manifest.py::CheckpointRule.resolve`; `study/manifest.py::geometric_checkpoints`, `round_half_up_power`, `_half_up_root` |
| effective schedule $E_r$ (§3.9, §5.5) | `study/manifest.py::effective_checkpoints`, `ExperimentManifest.has_condition` |
| engine keywords (§3.10) | `ExperimentManifest.monitored_circuit_kwargs` |
| probes of the observables | `CorrelatedErasure.probe`; the other probes are built by the runner from `Observables` |
| revisions (§3.11) | `study/manifest.py::check_extension` |

## 7. Worked examples

### (a) Tokens

```python
import hashlib
import json
import random
import struct
from fractions import Fraction

import numpy as np

from sparsegf2 import graphs, mipt
from sparsegf2.study import identity as ident
from sparsegf2.study import manifest as mf
from sparsegf2.study.identity import Token

for source in ("0.1", 0.1, "1/10", ".1", "1e-1", Fraction(1, 10), "1/3", "0.5", 0, 1):
    t = Token.parse(source)
    print(repr(source).ljust(16), "->", str(t).ljust(8), t.float64, t.hex, "exact" if t.is_exact else "rounded")
assert len({Token.parse(s) for s in ("0.1", 0.1, "1/10", ".1", "1e-1", Fraction(1, 10))}) == 1
print("float(0.1) - 1/10 =", Fraction(Token.parse("0.1").float64) - Fraction(1, 10))  # the token is the identifier

rng = random.Random(1)  # decimals of up to 15 digits survive the trip through a JSON number
for _ in range(20_000):
    s = "0." + "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 15)))
    assert Token.parse(float(s)) == Token.parse(s)
```

### (b) Seeds, the stream index and the engine

```python
print(ident.graph_seed(17, 3), ident.run_seed(29, "shared", "0.1"), ident.run_seed(29, "independent", "0.1"))
assert ident.rate_hash("0.1") == int.from_bytes(hashlib.blake2b(b"0.1", digest_size=8).digest(), "little")
# (s, 0) is the SeedSequence of s itself: the reason the manifest refuses equal master seeds
zero = np.random.SeedSequence([17, 0]).generate_state(4)
assert np.array_equal(zero, np.random.SeedSequence(17).generate_state(4))
assert not np.array_equal(np.random.SeedSequence([17, 1]).generate_state(4), zero)

stride = 8
for g, c in [(0, 0), (0, 7), (1, 0), (3, 1), (19, 7)]:
    u = ident.stream_index(g, c, stride)
    assert ident.split_stream_index(u, stride) == (g, c)
    print((g, c), "->", u)
key = ident.TrajectoryKey("ws-pilot-001", 16, "0.1", "0.15", 3, 1)
print(key, "u =", key.stream_index(stride))
assert ident.TrajectoryKey.parse(str(key)) == key

# the engine runs trajectory u on graph u // stride (three graphs, four circuits each)
records = [ident.build_graph(12, "0.3", g, 17) for g in range(3)]
spec = mipt.MonitoredCircuit(
    num_system=12, edges=[r.edges for r in records], circuits_per_graph=4, layers=40,
    measurement_rate=0.1, schedule="random_edges", reference="purification",
)
text = str(spec.circuit(ident.stream_index(1, 3, 4), seed=29))  # graph 1, circuit 3
pairs = set()
for line in text.splitlines():
    if line.startswith("CLIFFORD_2Q"):
        flat = list(map(int, line.split(") ")[1].split()))
        pairs |= {(a, b) for a, b in zip(flat[0::2], flat[1::2])}
assert pairs <= {tuple(e) for e in records[1].edges.tolist()}  # all gates are edges of graph 1
assert not pairs <= {tuple(e) for e in records[0].edges.tolist()}  # and not all of graph 0's

# a run on explicit indices: the list of graphs must reach the largest label
result = spec.run(indices=[ident.stream_index(1, 3, 4)], seed=29)
assert result.trajectories.tolist() == [7]  # graph 1, circuit 3, trajectory 1 * 4 + 3
try:
    spec.run(indices=[ident.stream_index(3, 0, 4)], seed=29)  # graph 3: only graphs 0, 1, 2 were given
except ValueError as error:
    print("refused:", str(error)[:90])
else:
    raise AssertionError("an index beyond the given graphs was accepted")
```

### (c) A graph record and its digest, computed by hand

```python
rec = ident.build_graph(16, "0.1", 3, 17)
print(rec.to_dict())
payload = b"SGF2EDGES1\x00" + struct.pack("<I", 16) + b"".join(struct.pack("<ii", int(a), int(b)) for a, b in rec.edges)
assert hashlib.sha256(payload).hexdigest() == rec.edge_digest and len(payload) == 11 + 4 + 8 * 32
assert np.array_equal(rec.edges, graphs.watts_strogatz(16, 4, 0.1, seed=(17, 3)))
rec.verify()
assert rec.regenerate() == rec
```

### (d) Canonical JSON

```python
doc = {"p": [Token.parse("1/3"), 0.5], "b": {"y": None, "x": True}, "a": np.int64(3), "z": -0.0}
print(ident.canonical_json(doc))
assert ident.canonical_json(doc) == '{"a":3,"b":{"x":true,"y":null},"p":["1/3",0.5],"z":0.0}'
assert ident.sha256_hex(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
```

### (e) A manifest

```python
MANIFEST = {
    "experiment_id": "ws-demo",
    "conditions": [{"n": [16, 24], "beta": ["0", "0.1"], "p": ["0.1", "0.15", "0.2"]}],
    "depth": {"layers": {"per_n": 8}, "checkpoints": {"count": 12}},
    "trials": {"graphs": 20, "circuits_per_graph": 2, "circuit_index_stride": 8},
    "seeds": {"master_graph_seed": 17, "master_circuit_seed": 29},
    "observables": {"correlated_erasure": [{"couplings": [0, 1, 2], "sizes": "1:15:2", "samples": 100, "thin": 20}]},
}
m = mf.ExperimentManifest.from_json(MANIFEST)
print(m.digest())
assert mf.ExperimentManifest.from_json(m.to_json()) == m and m.to_json() == mf.ExperimentManifest.from_json(m.to_json()).to_json()
assert json.loads(m.to_json())["trials"]["measurement_replicas"] == 1  # every default is written out
print([(c.n, str(c.beta), [str(p) for p in c.rates]) for c in m.conditions()])
print("T_max(16) =", m.layers(16, "0.1"), " checkpoints(16) =", m.checkpoints(16, "0.1"))
kwargs = m.monitored_circuit_kwargs(16, "0.1", "0.15")
print(kwargs)
spec = mipt.MonitoredCircuit(**kwargs, edges=[ident.build_graph(16, "0.1", g, 17).edges for g in range(3)])
assert spec.gates_per_layer == 8 and spec.circuits_per_graph == 8 and spec.num_trajectories == 24
key = m.trajectory_key(16, "0.1", "0.15", 2, 5)
assert m.stream_index(2, 5) == 21 and str(key) == "ws-demo:n=16:beta=0.1:p=0.15:g=2:c=5:r=0"
assert m.run_seed("0.1") == 29 and m.graph_seed(2) == (17, 2) and m.warnings == ()

# strict reading: every message names the field
for bad in (
    {**MANIFEST, "trials": {"graphs": 20, "circuits_per_graph": 2, "measurement_replicas": 2}},
    {**MANIFEST, "seeds": {"master_graph_seed": 17, "master_circuit_seed": 17}},
    {**MANIFEST, "conditions": [{"n": [16, 4], "beta": ["0"], "p": ["0.1"]}]},
    {**MANIFEST, "layout": {}},
):
    try:
        mf.ExperimentManifest.from_json(bad)
    except ValueError as error:
        print("refused:", str(error)[:110])
    else:
        raise AssertionError("a malformed manifest was accepted")

# the digest does not depend on how the conditions are factored or spelled
spelled = {**MANIFEST, "conditions": [
    {"n": [24], "beta": [0.1, "0"], "p": [0.2, "3/20", "1/10"]},
    {"n": [16], "beta": ["0.0", "1/10"], "p": ["0.10", "0.15", "2/10"]},
]}
assert mf.ExperimentManifest.from_json(spelled).digest() == m.digest()
```

### (f) The checkpoint rule, exactly

```python
from decimal import Decimal, localcontext


def decimal_checkpoints(t_max, count):
    """The rule of v1.4 section 9 in 120-digit decimal arithmetic."""
    points = {0, 1, t_max}
    if count >= 2:
        with localcontext() as ctx:
            ctx.prec = 120
            for j in range(count):
                a = (Decimal(j) * Decimal(t_max).ln() / Decimal(count - 1)).exp()
                points.add(min(max(int((a + Decimal("0.5")).to_integral_value(rounding="ROUND_FLOOR")), 0), t_max))
    return tuple(sorted(points))


for t_max in (1, 2, 10, 128, 1000, 4096):
    for count in (1, 2, 3, 12, 50):
        assert mf.geometric_checkpoints(t_max, count) == decimal_checkpoints(t_max, count)
print(mf.geometric_checkpoints(128, 12))  # a_j = 128^(j / 11)
assert mf.geometric_checkpoints(10, 4) == (0, 1, 2, 5, 10)  # 10^(1/3) = 2.15, 10^(2/3) = 4.64
assert mf.geometric_checkpoints(64, 4) == (0, 1, 4, 16, 64)  # 64^(1/3) = 4 exactly
for t_max in (5, 100, 4096):  # refinement: N_T -> k (N_T - 1) + 1 keeps every old checkpoint
    assert set(mf.geometric_checkpoints(t_max, 7)) <= set(mf.geometric_checkpoints(t_max, 2 * 6 + 1))
assert mf._half_up_root(25, 2) == 3  # a tie (x = 5/2) rounds up; python's round(2.5) would give 2

# the list form serves every size: entries above a size's horizon are dropped for that size
listed = mf.CheckpointRule.from_dict({"list": [70, 5, 20, 64, 130, 400]}, "list")
print(listed.resolve(64, 16), listed.resolve(200, 24))
assert listed.resolve(64, 16) == (0, 1, 5, 20, 64)  # 70, 130 and 400 dropped
assert listed.resolve(200, 24) == (0, 1, 5, 20, 64, 70, 130, 200)  # 400 dropped, T_max added
# the table form gives one list per size; an entry above its size's horizon is refused
table = mf.CheckpointRule.from_dict({"table": {"16": [5, 64], "24": [3, 100]}}, "table")
assert table.resolve(128, 16) == (0, 1, 5, 64, 128) and table.resolve(192, 24) == (0, 1, 3, 100, 192)
try:
    table.resolve(50, 16)
except ValueError as error:
    print("refused:", error)
else:
    raise AssertionError("a table entry above its horizon was accepted")
```

### (g) Revisions

```python
def revised(**changes):
    data = json.loads(m.to_json())
    for path, value in changes.items():
        *parents, last = path.split(".")
        node = data
        for part in parents:
            node = node[part]
        node[last] = value
    return mf.ExperimentManifest.from_json(data)


assert mf.check_extension(m, revised(**{"trials.graphs": 40})) is None  # more graphs
assert mf.check_extension(m, revised(**{"depth.checkpoints": {"count": 23}})) is None  # a refined schedule
assert mf.check_extension(m, revised(**{"depth.checkpoints": {"list": [3, 50]}})) is None  # any other rule
assert mf.check_extension(m, revised(**{"depth.layers": {"per_n": 16}})) is None  # a larger T_max (ruling R-CK)
assert mf.check_extension(m, revised(**{"resources.num_threads": 4})) is None  # free to change
for changes in ({"seeds.master_graph_seed": 18}, {"trials.graphs": 10}, {"depth.layers": {"per_n": 4}}):
    print(mf.check_extension(m, revised(**changes)))
```

### (h) The schedule across revisions (ruling R-CK)

```python
first = revised(**{"depth.layers": {"fixed": 16}, "depth.checkpoints": {"count": 5}})  # 16^(j/4) = 2^j
second = revised(**{"depth.layers": {"fixed": 64}, "depth.checkpoints": {"count": 4}})  # 64^(j/3) = 4^j
third = revised(**{"depth.layers": {"fixed": 100}, "depth.checkpoints": {"list": [3, 50, 200]}})  # 200 > 100: dropped
print([r.checkpoints(16, "0.1") for r in (first, second, third)])
assert first.checkpoints(16, "0.1") == (0, 1, 2, 4, 8, 16)
assert second.checkpoints(16, "0.1") == (0, 1, 4, 16, 64)
assert third.checkpoints(16, "0.1") == (0, 1, 3, 50, 100)
assert mf.check_extension(first, second) is None and mf.check_extension(second, third) is None

after_two = mf.effective_checkpoints([first, second], 16, "0.1")
after_three = mf.effective_checkpoints([first, second, third], 16, "0.1")
print(after_two, after_three)
assert after_two == (0, 1, 2, 4, 8, 16, 64)
assert after_three == (0, 1, 2, 3, 4, 8, 16, 50, 64, 100)
assert 8 not in second.checkpoints(16, "0.1") and 8 in after_two  # a stored depth survives a rule that skips it
assert set(after_two) <= set(after_three) and max(after_three) == third.layers(16, "0.1")
assert mf.effective_checkpoints([first], 16, "0.1") == first.checkpoints(16, "0.1")  # one revision: its own

try:
    mf.effective_checkpoints([second, first], 16, "0.1")  # the horizon would fall from 64 to 16
except ValueError as error:
    print("refused:", str(error)[:100])
else:
    raise AssertionError("a falling horizon was accepted")
```

## 8. Validation map

Files: `tests/test_study_identity.py` (identity), `tests/test_study_manifest.py` (manifest). "Exact" means equality; "property" means a hypothesis test with a fixed derandomized database.

| claim | test | what is checked |
|---|---|---|
| Proposition 1 (canonical strings) | `test_canonical_token_strings`, `test_equivalent_spellings_give_one_token`, `test_every_token_string_parses_back_to_itself_in_lowest_terms` (property) | exact: 27 listed spellings; parse inverts the string for all fractions with denominators up to 2000; the decimal expansion equals `Decimal`'s |
| Proposition 2 (JSON numbers) | `test_a_json_number_is_read_through_its_shortest_decimal`, `test_a_decimal_of_up_to_15_digits_survives_the_trip_through_a_json_number` (property) | exact for 8 floats; property over decimals of 1–15 digits at several scales |
| Proposition 3 (float) | `test_float64_is_the_nearest_double_and_hex_round_trips` (property), `test_a_third_stays_rational_and_records_float_and_hex` | the neighboring doubles are no closer; `fromhex(hex)` returns the float |
| malformed tokens | `test_malformed_tokens_are_refused_with_a_message_naming_the_field`, `test_token_constructor_guards_its_value` | 18 bad inputs, message starts with the field name |
| trajectory key (§3.2) | `test_trajectory_key_*` (5 tests, one a property) | canonical string, parse round trip, frozen and hashable, bad fields refused |
| Proposition 4 (seeds) | `test_graph_seed_*`, `test_graph_zero_shares_its_seed_with_the_plain_master_seed`, `test_run_seed_*` | `(master, g)` equals the native `[seed, i]` of `api._build_graph`; distinct pairs on a grid give distinct states; `(s, 0)` equals `s`; the rate hash against `hashlib` and a pinned value |
| Proposition 5 (stream index) | `test_stream_index_values_and_inverse`, `test_stream_index_is_one_to_one_and_ignores_the_totals` (property), `test_the_engine_runs_trajectory_u_on_graph_u_over_stride`; `tests/test_study_sampling.py::test_unrelated_choices_leave_the_circuit_unchanged` | exact; bijection and independence of the totals; the engine assignment; appended graphs leave $u$ unchanged |
| Proposition 6 (modes) | `tests/test_study_sampling.py::test_masks_are_uniform_below_the_float_threshold_and_nested_across_rates`, `test_nesting_does_not_hold_when_the_gate_mode_is_independent` | exact |
| Proposition 7, graph record (§3.5) | `test_build_graph_returns_the_validated_record_of_the_native_graph`, `test_the_digest_is_pinned_and_agrees_with_the_independent_reference`, `test_graph_record_identity_comes_from_the_digest_and_the_provenance`, `test_verify_detects_a_tampered_graph`, `test_edge_digest_is_strict_about_canonical_form`, `test_validate_graph_*` | the digest against `struct` and `hashlib` computed in the test and against the reference rewiring of `tests/_ws_reference.py`; a pinned vector; tampering detected |
| Proposition 8 (canonical JSON) | `test_canonical_json_*`, `test_sha256_hex_matches_the_published_vectors` | sorted keys, tokens as strings, numpy and tuples, refusals with the path of the value, published SHA-256 vectors |
| Proposition 9 (manifest digest) | `test_digest_is_stable_across_key_order_whitespace_and_spelling`, `test_the_canonical_blocks_group_sizes_that_share_rates_and_betas`, `test_digest_changes_with_every_scientific_setting`, `test_the_canonical_text_of_a_minimal_manifest_is_the_documented_form`, `test_the_digest_of_the_design_example_is_pinned` | 10 shuffled spellings and 4 factorizations give one digest; 13 changes give 13 new digests; a literal canonical text and a `hashlib` digest |
| strict reading (§3.7) | `test_malformed_manifests_are_refused_naming_the_field` (130 cases), `test_bad_json_text_is_refused`, `test_duplicate_conditions_across_blocks_and_equal_floats_are_refused`, `test_from_json_reads_text_dicts_and_files` | each case refused with a message that names the field |
| conditions, depth (§3.8, §3.9) | `test_conditions_are_resolved_in_a_stable_order`, `test_blocks_with_the_same_size_and_beta_merge_their_rates`, `test_layers_rules`, `test_unknown_conditions_are_refused` | exact |
| Proposition 10 (checkpoints) | `test_geometric_checkpoints_hand_computed` (23 schedules), `test_geometric_checkpoints_equal_high_precision_arithmetic` (140 cases against 120-digit `Decimal`), `test_checkpoint_invariants`, `test_refining_the_count_keeps_every_old_checkpoint`, `test_round_half_up_power_is_the_nearest_integer` (properties), `test_ties_round_upward_and_never_arise_for_integer_horizons` | exact equality with hand-computed and high-precision values; sorted, unique, contains $0,1,T$, size bound, within $\tfrac12$ of $a_j$; refinement; ties |
| list and table forms (§3.9) | `test_explicit_checkpoint_lists_always_include_zero_one_and_horizon`, `test_a_checkpoint_list_serves_every_size_and_drops_what_is_above_its_horizon`, `test_a_checkpoint_table_gives_one_list_per_size`, `test_the_list_form_is_the_list_below_the_horizon_with_zero_one_and_the_horizon` (property), `test_the_table_form_is_the_size_entry_with_zero_one_and_the_horizon` (property); the refusals are among the 130 malformed cases | exact schedules at several sizes and horizons; the list form equals the entries not above $T$ plus $0, 1, T$ for random lists and horizons; the table form equals the entries of the size plus $0, 1, T$; a missing or an extra size and an entry above $T_{\max}(n)$ are refused |
| engine keywords (§3.10) | `test_monitored_circuit_kwargs_are_the_v14_protocol`, `test_the_kwargs_build_the_study_protocol_for_even_and_odd_sizes` | the exact dictionary; a `MonitoredCircuit` built for $n = 5, \dots, 25$ has $\lfloor n/2\rfloor$ gates per layer and every field of v1.4 §8.1 |
| correlated-erasure entries | `test_correlated_erasure_settings_build_the_engine_probe`, `test_correlated_erasure_sizes_forms`, `test_several_correlated_erasure_probes_need_distinct_names` | the probe equals the direct `mipt.correlated_erasure` call; sizes forms; unique names |
| Proposition 11 (revisions) | `test_allowed_revisions` (18 changes), `test_forbidden_revisions_name_the_first_forbidden_change` (19 changes), `test_the_first_forbidden_change_in_the_documented_order_is_reported`, `test_adding_observables_and_a_larger_horizon_are_extensions`, `test_a_chain_of_revisions_is_checked_pairwise`, `test_check_extension_requires_manifests` | each allowed change (among them a larger $T_{\max}$ under the geometric rule and every change of the rule) returns `None`; each forbidden one (among them a lower $T_{\max}$) returns a message with the named field; the order of the checks; composition |
| Proposition 12 (effective schedule) | `test_the_effective_schedule_is_the_union_of_the_schedules_of_the_revisions`, `test_a_size_or_beta_absent_from_an_earlier_revision_contributes_nothing_from_it`, `test_effective_checkpoints_refuses_what_is_not_an_extension_chain`, `test_any_change_of_the_checkpoint_rule_is_an_extension` (16 pairs of rule kinds), `test_effective_checkpoints_properties` and `test_a_chain_in_the_wrong_order_is_refused_exactly_when_it_is_not_an_extension` (properties over random valid chains) | exact unions at three horizons; a new size or beta contributes nothing from the revisions that lack it; refusals name the pair and the first forbidden change; over random chains of 1 to 4 revisions (horizons that rise, sizes that are added, every rule kind): the effective schedule is sorted, equals the union of the own schedules, never exceeds the latest $T_{\max}$, contains $0, 1, T_{\max}$, contains every own schedule and every earlier effective schedule, equals the own schedule for one revision, and every pair of the chain is an extension |

## 9. Limitations and departures

Departures from v1.4 and the design (D1, D2, D4 are the design's; N1–N6 are introduced here, each with its reason):

* **D1.** `measurement_replicas` must be 1: the native engine draws gates and masks from one stream, so independent mask replicates on the same gates do not exist (v1.4 §7.2, §8.1). The key keeps $r$.
* **D2.** The frozen stride $s \ge C$ (§3.4, Proposition 5).
* **D4.** `gate_mode="independent"` derives the run seed from the rate token (§3.3).
* **N1.** The master graph seed and the master circuit seed must differ, because graph 0 would otherwise share its seed words with the circuit streams (Proposition 4). This refuses a manifest the design would accept.
* **N2.** Seeds and graph labels are below $2^{32}$ (§3.3), so $(M, g)$ is a two-word entropy. Larger seeds are refused rather than reduced.
* **N3.** The digest is computed from the canonical factorization of the conditions, not from the blocks as written (§3.8, Proposition 9). Two manifests that describe the same conditions have one digest; the design's example, which is a single product block, is unchanged.
* **N4.** `label` is restricted to `exploratory` and `confirmatory` (v1.4 §9), and a revision may not change it. The manifest does not enforce v1.4's rule that the rate grid is frozen before confirmatory runs: a confirmatory manifest may still gain conditions, as for an exploratory one.
* **N5 (superseded by ruling R-CK).** The first version of this part let a revision only *add* checkpoints, which made a larger $T_{\max}$ under the geometric rule impossible without an explicit list. Ruling R-CK replaced that: a revision may raise $T_{\max}$ and change the checkpoint rule in any way, and the schedule used for work is the union of the schedules of all revisions (§3.9, Proposition 12). The costs: the union of a long chain is as large as the sum of its own schedules (the points of the geometric rule of every horizon that was ever used), and a trajectory run under an earlier revision holds only that revision's depths until the runner reruns it to the longer horizon.
* **N6.** `monitored_circuit_kwargs` leaves out `edges` and `probes`: the graph records of a work unit and the probes are the runner's. Two tokens with the same float64 are refused as a duplicate condition (§3.8).

Other limitations:

* **Hash-based claims.** That distinct seeds give distinct streams, that two different graphs have different digests, and that two different tokens have different rate hashes are claims about SHA-256, BLAKE2b and the `SeedSequence` mixing; they are checked on grids and not proved.
* **JSON numbers.** A float is read through its shortest round-trip decimal, which is the typed decimal only for at most 15 significant digits (Proposition 2); the exact forms (decimal or rational strings) have no such limit.
* **Analysis settings can change in a revision.** They touch no stored value; values derived under them record the settings used. The `peak` rule is "predeclared" (v1.4 §5.8) only if the label is confirmatory and the manifest is not revised, which the code does not enforce (N4).
* **No store.** Nothing here persists a manifest, a record or a key; document 05 describes the store that does.
