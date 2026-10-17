# Nested ensemble statistics

How the Watts–Strogatz study turns trajectory-level values into ensemble estimates with honest
errors: the equal-graph-weight nested mean and its graph-cluster standard error, paired
differences across measurement rates, censored first-passage times (weighted Kaplan–Meier with
graph-equal weights, its variance, the median and the restricted mean), the bootstrap over
graphs, and the three peak-time summaries of the specification. Everything here describes
`src/sparsegf2/study/stats.py`; statements about the code cite `stats.py::name`.

The scientific specification is `docs/misc/watts_strogatz_study_spec_v1_4.md` ("v1.4 §x"), the
design `docs/specs/2026-10-17-ws-study-pipeline-design.md` ("design §x"). The Python blocks run
as written, in order, in one namespace, in a few seconds.

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Notation](#2-notation)
3. [The nested estimator](#3-the-nested-estimator)
4. [The graph-cluster standard error](#4-the-graph-cluster-standard-error)
5. [Paired differences across rates](#5-paired-differences-across-rates)
6. [Censored first-passage times](#6-censored-first-passage-times)
7. [The bootstrap over graphs](#7-the-bootstrap-over-graphs)
8. [Peak-time summaries](#8-peak-time-summaries)
9. [Algorithms](#9-algorithms)
10. [From the mathematics to the code](#10-from-the-mathematics-to-the-code)
11. [Worked examples](#11-worked-examples)
12. [Validation map](#12-validation-map)
13. [Limitations and departures](#13-limitations-and-departures)

## 1. Purpose and scope

The study samples nested random objects (v1.4 §1, §6.1): interaction graphs $G_g$ drawn
independently from the Watts–Strogatz law $\mu_{n,\beta}$, gate sequences $c$ drawn given their
graph, and measurement realizations $r$ drawn given their gate sequence (the native engine has
$r = 0$ only, design §9 D1). Samples that share a graph are correlated, and the graphs are the
independent units (v1.4 §6.4). This part implements:

* the nested estimator of v1.4 §6.3, which gives every graph the same weight, with counts per
  level and an explicit policy for undefined values;
* the graph-cluster standard error of v1.4 §6.4 and Student-$t$ intervals;
* paired differences across rates on the same graphs (v1.4 §6.4, §7.3, §7.4 "Analysis" row);
* right-censored first-passage depths $\tau_\ell$ (v1.4 §5.2): a weighted Kaplan–Meier estimate
  with graph-equal weights, a graph-cluster variance, the median, the restricted mean to the
  horizon and a censoring summary, and never a plain mean of observed depths when some are
  censored (v1.4 §5.8, §5.9);
* the bootstrap over graphs, whose draws are shared by every rate and observable so that pairing
  survives (v1.4 §5.8, §9 "Finite-size scaling" row);
* the three peak-time statistics of v1.4 §5.8, kept apart.

The functions take plain arrays (one row per trajectory with its labels $g, c, r$), so they do
not depend on how the values were stored. Finite-size scaling, which consumes these estimates,
is [07-finite-size-scaling.md](07-finite-size-scaling.md).

## 2. Notation

| symbol | meaning | type | in the code |
|---|---|---|---|
| $g, c, r$ | graph, gate-sequence and replica labels of a trajectory | integers $\ge 0$ | `g`, `c`, `r` |
| $G$ | number of graphs (v1.4's $G_\ast$) | integer | `GraphMean.num_graphs` |
| $C_g$ | gate sequences on graph $g$ | integer | `NestedMean.circuits_per_graph` |
| $R_{gc}$ | replicas of gate sequence $(g, c)$ | integer | `NestedMean.replicas_per_circuit` |
| $F_{gcr}$ | the observable of one trajectory (one checkpoint, or a vector of them) | real or NaN | rows of `values` |
| $\hat f_{gc}$, $\hat f_g$ | gate-sequence and graph means | real | `NestedMean.graph_means` ($\hat f_g$) |
| $\hat\mu$ | the nested estimate | real | `GraphMean.mean` |
| $\widehat{\mathrm{SE}}$ | the graph-cluster standard error | real | `GraphMean.se` |
| $m(g)$ | $\mathbb{E}[F \mid G = g]$, the mean over gates, masks and outcomes on graph $g$ | real | (target) |
| $\mu$ | $\mathbb{E}_{G}[m(G)]$, the ensemble mean (v1.4 §6.2) | real | (target) |
| $D_g$ | per-graph difference between two rates | real | `paired_difference(...).graph_means` |
| $\tau$, $T_{\max}$ | a first-passage depth and the horizon (censoring depth) | integers | `times`, `observed`; `SurvivalCurve.horizon` |
| $w_i$ | weight of trajectory $i$ | positive | `graph_equal_weights` |
| $t_j$; $Y_j$, $D_j$ | event depths; weighted numbers at risk and of events | reals | `SurvivalCurve.times`; `at_risk`, `events` (as shares of $W = \sum_i w_i$) |
| $\hat S(t)$ | Kaplan–Meier estimate of $S(t) = \Pr(\tau > t)$ | in $[0, 1]$ | `SurvivalCurve.survival` |
| $U_g(t)$ | influence of cluster $g$ on $\hat S(t)$ | real | `stats.py::_km_influence_chunks` |
| $\mathrm{RM}(h)$ | restricted mean $\int_0^h \hat S(t)\,dt$ | real | `SurvivalCurve.restricted_mean` |
| $d(t)$, $\mathcal A$, $T^\ast$ | a robustness trace, its maximizing set, the peak depth (v1.4 §5.8) | | `traces`, `peak_times`, `at_horizon` |
| $B$ | bootstrap replicates | integer | `replicates` |
| $t_{q,\nu}$ | the $q$-quantile of Student's $t$ with $\nu$ degrees of freedom | real | `student_t_quantile(q, nu)` |

## 3. The nested estimator

### 3.1 Definition

For one condition $(n, \beta, p)$ and one depth (or a vector of depths, handled column by
column), with $C_g \ge 1$ gate sequences on graph $g$ and $R_{gc} \ge 1$ replicas of $(g, c)$
(v1.4 §6.3):

$$
\hat f_{gc} = \frac{1}{R_{gc}}\sum_{r} F_{gcr}, \qquad
\hat f_{g} = \frac{1}{C_g}\sum_{c} \hat f_{gc}, \qquad
\hat\mu = \frac{1}{G}\sum_{g} \hat f_{g}.
$$

Equivalently $\hat\mu = \sum_i w_i F_i$ with trajectory weights
$w_{gcr} = 1/(G\,C_g\,R_{gc})$: every graph has total weight $1/G$, and every gate sequence of a
graph an equal share of it. The code (`stats.py::nested_mean`) first puts the rows in canonical
order (by $g$, then $c$, then $r$) and refuses a repeated $(g, c, r)$; every sum then runs in
that order, so the result is bitwise the same for any order or chunking of the input rows.

### 3.2 Unbiasedness

**Proposition 1 (v1.4 §6.3; proof derived here).** Let $G_1, \dots, G_G$ be identically
distributed graphs, let each graph's trajectories be independent of the other graphs and of
their trajectories (each trajectory has its own stream), and let the allocation
$(C_g, R_{gc})$ be *nonadaptive*: fixed in advance, or a function of the graphs only, never of
observed values. Then $\mathbb{E}\hat\mu = \mu$.

*Proof.* Condition on all graphs $\mathbf G$ (the allocation is then fixed). Given $G_g$ and
its gate sequence $U_{gc}$, the replicas are identically distributed, so
$\mathbb{E}[F_{gcr} \mid \mathbf G, U_{gc}] = h(G_g, U_{gc})$, and the gate sequences of a graph
are identically distributed given it, so $\mathbb{E}[h(G_g, U_{gc}) \mid \mathbf G] = m(G_g)$.
Averages of quantities with conditional mean $m(G_g)$ have conditional mean $m(G_g)$:
$\mathbb{E}[\hat f_{gc} \mid \mathbf G] = \mathbb{E}[\hat f_{g} \mid \mathbf G] = m(G_g)$. Hence
$\mathbb{E}[\hat\mu \mid \mathbf G] = \frac1G\sum_g m(G_g)$ and
$\mathbb{E}\hat\mu = \mathbb{E}[m(G)] = \mu$. ∎

The conditional statement is the useful one: given the graphs that were drawn,
$\hat\mu$ estimates their *equal-weight* average $\frac1G\sum_g m(G_g)$, whatever the allocation.

### 3.3 Why pooling weights graphs by their row counts

The pooled mean of all $N$ rows is $\hat\mu_{\mathrm{pool}} = \sum_g n_g \bar F_g / \sum_g n_g$,
with $n_g = \sum_c R_{gc}$ the rows of graph $g$ and $\bar F_g$ their plain average.

**Proposition 2 (derived here).** Under the assumptions of Proposition 1,
$\mathbb{E}[\hat\mu_{\mathrm{pool}} \mid \mathbf G] = \sum_g n_g\, m(G_g) / \sum_g n_g$: each graph
weighs in proportion to its rows. If the $n_g$ are constants, $\hat\mu_{\mathrm{pool}}$ is also
unbiased for $\mu$; if they depend on the graphs, it converges to
$\mathbb{E}[n(G)m(G)]/\mathbb{E}[n(G)] = \mu + \operatorname{Cov}(n(G), m(G))/\mathbb{E}[n(G)]$
as $G \to \infty$, which differs from $\mu$ whenever the allocation is correlated with the graph
means.

*Proof.* $\bar F_g$ is an average of rows with conditional mean $m(G_g)$, so the first claim
follows as in Proposition 1. With constant $n_g$, $\mathbb{E}$ of the weighted average of
identically distributed $m(G_g)$ is $\mu$. With $n_g = n(G_g)$, numerator and denominator divided
by $G$ converge almost surely to $\mathbb{E}[n(G) m(G)]$ and $\mathbb{E}[n(G)]$ (strong law), and
$\mathbb{E}[nm] = \mathbb{E}[n]\mu + \operatorname{Cov}(n, m)$. ∎

So the equal graph weight of v1.4 §6.3 is a statement about the target: conditionally on the
graphs, the nested estimator weighs each graph $1/G$ and the pooled one $n_g/N$. Example
§11 (b) shows the bias of pooling when the allocation follows the graph type. When the allocation
is balanced ($C_g = C$, $R_{gc} = R$) the two coincide (v1.4 §6.3): $\hat\mu$ is then the plain
mean of the $GCR$ rows, though those rows remain correlated within graphs.

### 3.4 Undefined values and the change of target

Some observables are undefined on some trajectories: the raw contiguous distance is NaN when
$k_t = 0$ (v1.4 §5.5), a peak time is missing when $d^\ast = 0$ (v1.4 §5.8). `nested_mean`
refuses NaN unless `nan_policy="omit"`, which leaves out a NaN row from its gate sequence, a gate
sequence with no defined row from its graph, and a graph with no defined value from the mean.
`contributing_graphs` and `contributing_trajectories` count what entered.

**What "omit" estimates (derived here).** With $\mathcal D$ the event "defined", the omitted
estimator is, conditionally on the graphs and the pattern of defined rows, the equal-weight mean
over the graphs with at least one defined row of each graph's equal-weight mean of its defined
rows. It targets neither $\mu = \mathbb{E}[F]$ (undefined rows have no value) nor
$\mathbb{E}[F \mid \mathcal D]$, which weighs graph $g$ by $\Pr(\mathcal D \mid G = g)$; it also
depends on the allocation, because a graph enters only when one of its rows is defined. Example
(test `test_undefined_values_are_refused_unless_omitted_which_changes_the_target`): graph 0 has
one defined circuit of value 1 among four, graph 1 four defined circuits of value 3; the omitted
estimate is $2$ and $\mathbb{E}[F \mid \mathcal D]$'s plug-in estimate is $13/5$. Use omission
for statistics that are meant conditionally (for example on $k_t > 0$) and report
`contributing_graphs`; for an unconditional statistic use the zero extension instead (v1.4 §5.5).

## 4. The graph-cluster standard error

### 4.1 Definition and what it captures

With $G > 1$ graphs (v1.4 §6.4),

$$
\widehat{\mathrm{SE}}(\hat\mu) = \sqrt{\frac{1}{G(G-1)}\sum_{g}\big(\hat f_g - \hat\mu\big)^2}
$$

(`stats.py::graph_cluster_se`, `GraphMean.se`). It uses only the graph means, which contain all
the within-graph randomness (gate sequences, masks, outcomes) of their graph, so it estimates the
variance of the whole nested estimator; treating rows or gate sequences as independent draws
would ignore their correlation within a graph. With $G = 1$ it is NaN: the graph-ensemble
uncertainty cannot be inferred from one graph.

**Proposition 3 (derived here).** If the $\hat f_g$ are independent with a common mean $\mu$ and
arbitrary variances $v_g$, then $\operatorname{Var}\hat\mu = \frac{1}{G^2}\sum_g v_g$ and
$\mathbb{E}\big[\widehat{\mathrm{SE}}^2\big] = \operatorname{Var}\hat\mu$ exactly.

*Proof.* $\operatorname{Var}\hat\mu = G^{-2}\sum_g v_g$ by independence. Writing
$\hat f_g - \hat\mu = (\hat f_g - \mu) - (\hat\mu - \mu)$,
$\mathbb{E}\sum_g(\hat f_g - \hat\mu)^2 = \sum_g v_g - G\operatorname{Var}\hat\mu = (1 - 1/G)\sum_g v_g$.
Divide by $G(G-1)$. ∎

The graph means of different graphs are independent under the study's design (independent graph
draws, and per-trajectory streams that depend only on the trajectory's index, v1.4 §7.2–7.3), and
they share the mean $\mu$ (Proposition 1, with a nonadaptive allocation), but their variances
differ when $C_g$ or $R_{gc}$ differ. In the variance-components model
$F_{gcr} = \mu + a_g + b_{gc} + e_{gcr}$ (independent, variances $\sigma_a^2, \sigma_b^2, \sigma_e^2$),

$$
v_g = \sigma_a^2 + \frac{\sigma_b^2}{C_g} + \frac{\sigma_e^2}{C_g^2}\sum_c \frac{1}{R_{gc}},
$$

which example §11 (c) and the tests check by simulation.

### 4.2 Intervals

`GraphMean.interval(level)` returns
$\hat\mu \mp t_{(1+\mathrm{level})/2,\,G'-1}\,\widehat{\mathrm{SE}}$, with $G'$ the contributing
graphs. Its coverage is exact when the $\hat f_g$ are independent normal with equal variances; with
unequal variances it is approximate (measured 0.948 at level 0.95 for $G = 20$ in
`test_the_estimator_is_unbiased_and_its_cluster_error_estimates_the_exact_variance`).

**The $t$ quantile without SciPy** (`stats.py::student_t_quantile`). With
$\theta = \arctan(t/\sqrt\nu)$, the two-sided probability $A(t) = \Pr(\lvert T\rvert \le t)$ is a
finite sum (Abramowitz and Stegun, 26.7.3 and 26.7.4):

$$
\nu = 1:\ A = \tfrac{2\theta}{\pi};\qquad
\nu\ \text{odd}:\ A = \tfrac{2}{\pi}\Big[\theta + \sin\theta\cos\theta\sum_{k=0}^{(\nu-3)/2} a_k\cos^{2k}\theta\Big];\qquad
\nu\ \text{even}:\ A = \sin\theta\sum_{k=0}^{(\nu-2)/2} b_k\cos^{2k}\theta,
$$

with $a_0 = b_0 = 1$, $a_k = a_{k-1}\,2k/(2k+1)$, $b_k = b_{k-1}(2k-1)/(2k)$. $A$ increases from
0 to 1 on $[0, \pi/2)$, so the quantile for $q > 1/2$ is $\sqrt\nu\tan\theta^\ast$ with
$A(\theta^\ast) = 2q - 1$, found by bisection until no double lies strictly inside the bracket;
$t_{1-q} = -t_q$. The tests compare it with the closed forms for $\nu = 1, 2$, with tables and
with a numerical integral of the density.

## 5. Paired differences across rates

The rates of one $(n, \beta)$ are measured on the same graphs, and under the shared gate mode the
same gate sequences and mask uniforms (v1.4 §7.3, §9): estimates at two rates are positively
correlated. v1.4 §6.4: form the differences within each graph first,
$D_g = \hat f_g(p_2) - \hat f_g(p_1)$, and take their mean and cluster error
(`stats.py::paired_difference`, which returns a `GraphMean` of the $D_g$).

**Proposition 4 (derived here).** The $D_g$ are independent across graphs with mean
$\mu(p_2) - \mu(p_1)$ and variance
$\operatorname{Var}D_g = v_g(p_1) + v_g(p_2) - 2\operatorname{Cov}\big(\hat f_g(p_1), \hat f_g(p_2)\big)$,
so Proposition 3 applies to them: $\widehat{\mathrm{SE}}(\bar D)^2$ is unbiased for
$\operatorname{Var}\bar D$. Combining the two errors as if independent,
$\sqrt{\widehat{\mathrm{SE}}_1^2 + \widehat{\mathrm{SE}}_2^2}$, estimates the variance without the
covariance term and overstates it when the covariance is positive.

*Proof.* $D_g$ depends only on graph $g$'s trajectories at the two rates, which are independent of
other graphs'; linearity of expectation and the variance of a difference give the rest. ∎

The function requires the same graph labels at both rates (`graphs="same"`) or uses those both
have (`graphs="common"`); a graph undefined at either rate is left out. Example §11 (d) and
`test_paired_intervals_cover_and_unpaired_ones_overcover` show both effects.

## 6. Censored first-passage times

### 6.1 Data

For a threshold $\ell$, $\tau_\ell = \min\lbrace t : k_t \le \ell\rbrace$ (v1.4 §5.2). A trajectory
whose $k_t$ never reaches $\ell$ by $T_{\max}$ is right-censored: it is recorded with
`observed = False` and time $T_{\max}$ (the censoring depth, never a sentinel such as $-1$ or
$\infty$). In one run every censored trajectory has the same censoring depth, the horizon.

### 6.2 The weighted Kaplan–Meier estimate

With positive weights $w_i$, the distinct observed depths $t_1 < \dots < t_J$, and

$$
Y_j = \sum_{i:\,T_i \ge t_j} w_i, \qquad D_j = \sum_{i:\,T_i = t_j,\ \text{observed}} w_i,
$$

the estimate of $S(t) = \Pr(\tau > t)$ is right-continuous,

$$
\hat S(t) = \prod_{j:\,t_j \le t}\Big(1 - \frac{D_j}{Y_j}\Big)
$$

(`stats.py::weighted_kaplan_meier`). A trajectory censored at $t_j$ is still at risk at $t_j$.
For the study the weights are graph-equal (`stats.py::graph_equal_weights`):
$w_{gcr} = L/(C_g R_{gc})$ with $L = \operatorname{lcm}_{g,c}(C_g R_{gc})$, integers whose total
is $L$ for every graph. They are proportional to the nested weights of §3.1, and $\hat S$ only
uses their ratios. `stats.py::nested_kaplan_meier(g, c, r, times, observed)` uses them with the
graphs as clusters (§6.3).

**Proposition 5 (one horizon; derived here).** Suppose every censored trajectory has
$T_i = h$ and every observed depth is $\le h$. Then for every $t \in [0, h]$,

$$
\hat S(t) = \frac{\sum_i w_i\,\mathbf 1[\tau_i > t]}{\sum_i w_i},
$$

and with graph-equal weights this is the nested mean (§3.1) of $\mathbf 1[\tau > t]$, so by
Proposition 1 it is unbiased for $S(t)$.

*Proof.* Every trajectory has $T_i \ge t_1$, so $Y_1 = W = \sum_i w_i$. Censoring happens only at
$h \ge t_J$, so between consecutive event depths only events leave the risk set:
$Y_{j+1} = Y_j - D_j$ for $j < J$. The product telescopes,
$\hat S(t_j) = \prod_{k \le j} Y_{k+1}/Y_k = (Y_j - D_j)/W$ (with $Y_{J+1} := Y_J - D_J$), and
$Y_j - D_j$ is the weight of the trajectories with $T_i > t_j$ or censored at $t_j$, which are
exactly those with $\tau_i > t_j$ (a censored trajectory has $\tau_i > h \ge t_j$). Between event
depths both sides are constant, and before $t_1$ both equal 1. The nested mean is
$\sum_i F_i w_i/W$ with these weights (§3.1). ∎

So in the study's single-horizon runs the Kaplan–Meier curve is no model: it is the nested mean
of the survival indicator, depth by depth. With several horizons (runs of different $T_{\max}$
combined) it is the usual product-limit estimate, consistent under independent censoring.

### 6.3 The variance: graph-cluster delta method

The graphs are independent and the trajectories of one graph correlated, so the variance treats
each cluster (`clusters = g`) as one unit. Write $Y_{gj}$ and $D_{gj}$ for the at-risk and event
weights of cluster $g$, so $Y_j = \sum_g Y_{gj}$ and $D_j = \sum_g D_{gj}$.

**Proposition 6 (the influence; derived here).** Scale the weights of cluster $g$ by
$1 + \varepsilon$. Then

$$
U_g(t_j) := \frac{d\hat S(t_j)}{d\varepsilon}\Big\rvert_{\varepsilon=0}
= \hat S(t_j)\sum_{k \le j}\frac{Y_{gk}D_k - D_{gk}Y_k}{Y_k\,(Y_k - D_k)}
$$

as long as $Y_k > D_k$ for $k \le j$, and $U_g(t_j) = 0$ from the first $k$ with $Y_k = D_k$ on.
The influences sum to zero: $\sum_g U_g(t) = 0$.

*Proof.* $\log\hat S(t_j) = \sum_{k\le j}\log(1 - D_k/Y_k)$ and
$\frac{d}{d\varepsilon}\log(1 - D_k/Y_k) = \frac{-D_{gk}Y_k + D_kY_{gk}}{Y_k(Y_k - D_k)}$ because
$dD_k/d\varepsilon = D_{gk}$ and $dY_k/d\varepsilon = Y_{gk}$; multiply by $\hat S$. If
$Y_k = D_k$, then $\hat S = 0$ from $t_k$ on, and every trajectory at risk at $t_k$ has its event
there, so $Y_{gk} = D_{gk}$ for every $g$ and $1 - D_k/Y_k$ stays 0 under the perturbation: the
derivative is 0. Finally $\hat S$ is homogeneous of degree 0 in all the weights, so scaling every
cluster at once changes nothing: $\sum_g U_g = 0$ (Euler's identity). ∎

The variance estimate is the cluster sandwich (first-order delta method on the cluster totals):

$$
\widehat{\operatorname{Var}}\,\hat S(t) = \frac{G}{G-1}\sum_g U_g(t)^2 .
$$

It is a first-order approximation; the factor $G/(G-1)$ is the usual small-sample correction
(the mean influence is exactly 0). `stats.py::KM_VARIANCE` records this description with every
curve.

**Proposition 7 (reduction to v1.4 §6.4; derived here).** Under the assumptions of
Proposition 5 with graph-equal weights,
$U_g(t) = \frac1G\big(\hat S_g(t) - \hat S(t)\big)$, where $\hat S_g(t)$ is graph $g$'s nested
fraction with $\tau > t$, so $\widehat{\operatorname{Var}}\hat S(t)$ is exactly the squared
graph-cluster error of §4 of the nested mean of $\mathbf 1[\tau > t]$.

*Proof.* With one horizon, $Y_k - D_k = Y_{k+1}$ and $Y_{gk} - D_{gk} = Y_{g,k+1}$ for every
cluster. Then $Y_{gk}D_k - D_{gk}Y_k = Y_{g,k+1}Y_k - Y_{gk}Y_{k+1}$, and the summand is
$Y_{g,k+1}/Y_{k+1} - Y_{gk}/Y_k$, which telescopes to $Y_{g,j+1}/Y_{j+1} - W_g/W$ ($W_g$ the
total weight of cluster $g$). Multiplying by $\hat S(t_j) = Y_{j+1}/W$ gives
$U_g = Y_{g,j+1}/W - \hat S\,W_g/W = (W_g/W)(\hat S_g - \hat S)$ with
$\hat S_g = Y_{g,j+1}/W_g$; graph-equal weights have $W_g/W = 1/G$. Then
$\frac{G}{G-1}\sum_g U_g^2 = \frac{1}{G(G-1)}\sum_g(\hat S_g - \hat S)^2$. ∎

The code accumulates $\sum_g U_g^2$ over chunks of clusters (§9.2), so memory stays bounded; the
tests check the formula against finite differences of the estimate itself and Proposition 7 to
rounding.

### 6.4 The median

`SurvivalCurve.median` is $\min\lbrace t : \hat S(t) \le 1/2\rbrace$ (None when the curve stays
above 1/2 within the follow-up), the lower median of the estimated law. `median_upper` is
$\min\lbrace t : \hat S(t) < 1/2\rbrace$. When $\hat S$ sits exactly at 1/2 from `median` on,
every depth in $[\texttt{median}, \texttt{median\_upper}]$ is a median of the estimated law, and
`median_upper = None` says that interval reaches past the follow-up. (The JSON service's
`tau_median_all` reports the median only when the curve drops strictly below 1/2 before the
horizon; that is `median` when `median_upper` is not None.)

**Exactness at 1/2.** A product of float ratios can land one unit in the last place on the wrong
side of $1/2$ (example §11 (e): the factors give $0.5000000000000001$ for an exact $1/2$). With
integer weights (graph-equal weights are integers, unit weights too) the code decides exactly:
each factor and product of the float curve is rounded at most four times per step, so
$\lvert \hat S_{\mathrm{float}}(t_j) - \hat S(t_j)\rvert \le 2(j+1)u$ near $1/2$
($u = 2^{-53}$); only the $j$ with $\lvert\hat S_{\mathrm{float}}(t_j) - 1/2\rvert \le (4j+8)u$
are checked, by comparing $2\prod_{k\le j}(Y_k - D_k)$ with $\prod_{k\le j}Y_k$ in Python
integers. An exact $1/2$ becomes $0.5$; a value on the wrong side moves by one unit in the last
place (`stats.py::_snap_half`). Non-integer weights give the plain float curve (`exact_weights`
says which).

### 6.5 The restricted mean, and what it means under censoring at the horizon

**Proposition 8 (derived here).** For a non-negative $\tau$ and $h \ge 0$,
$\mathbb{E}[\min(\tau, h)] = \int_0^h S(t)\,dt$; for integer depths this is
$\sum_{t=0}^{h-1}\Pr(\tau > t)$.

*Proof.* $\min(\tau, h) = \int_0^h \mathbf 1[\tau > t]\,dt$ pointwise; take expectations and
exchange them with the integral (Tonelli, the integrand is non-negative). For integer $\tau$,
$S$ is constant on $[t, t+1)$. ∎

`SurvivalCurve.restricted_mean` is $\int_0^h \hat S(t)\,dt$ (the area under the step curve, with
$\hat S = 1$ before $t_1$) at `restricted_horizon` $= h$ (default: the largest time). Under
censoring at the horizon $T_{\max}$, Propositions 5 and 8 give its exact meaning:
$\mathrm{RM}(T_{\max})$ *is* the nested mean of $\min(\tau, T_{\max})$, the depth capped at the
horizon, an unbiased estimate of $\mathbb{E}[\min(\tau, T_{\max})]$ with no assumption about the
censored trajectories. It is a lower bound on $\mathbb{E}[\tau]$, equal to it exactly when
$\Pr(\tau > T_{\max}) = 0$. Its error is the cluster delta method with influence
$\sum_j \ell_j U_g(t_j)$ ($\ell_j$ the length of the step at $t_j$ inside $[0, h]$), which under one
horizon is the §6.4 error of the nested mean of $\min(\tau, T_{\max})$ (same telescoping as
Proposition 7).

A horizon above the largest follow-up depth is refused, because $\hat S$ is unknown there, unless
the curve has reached 0 (then $\hat S = 0$ beyond, and the extra area is 0).

**The mean.** `SurvivalCurve.mean` is the weighted mean of the depths, given only when no
trajectory is censored (it then equals $\mathrm{RM}$ at the largest depth); otherwise it is None.
A mean of the observed depths when some are censored is the conditional mean
$\mathbb{E}[\tau \mid \tau \le T_{\max}]$, not the ensemble mean, and nothing here returns it
(v1.4 §5.8, §5.9).

### 6.6 The censoring summary

`stats.py::censoring_summary(g, c, r, times, observed)` reports the counts, the graph-equal-weight
censored fraction (the nested mean of the censoring indicator) with its cluster error, the plain
fraction of rows, the horizon (largest depth), the distinct censoring depths, and how many graphs
are censored in every or in some trajectory (v1.4 §5.9: report censoring by condition).

## 7. The bootstrap over graphs

### 7.1 Draws

`stats.py::graph_bootstrap(graphs, B, seed)` returns a $(B, G)$ array of graph labels: replicate
$b$ draws $G$ labels uniformly with replacement. The draws come from the raw output of numpy's
PCG64 bit generator seeded by `SeedSequence(seed)`; numpy keeps the raw bit streams and
`SeedSequence` stable across versions (unlike `Generator` methods), so a seed gives the same
draws. A raw word $x$ is mapped to $x \bmod G$ after rejecting the $2^{64} \bmod G$ largest words:
the accepted words are uniform on $[0, L)$ with $L = 2^{64} - (2^{64} \bmod G)$ a multiple of $G$,
so every residue has $L/G$ preimages and is equally likely.

### 7.2 Applying the draws, and pairing

`GraphMean.bootstrap(draws)` returns each replicate's estimate: the mean of the drawn graphs'
values (a graph drawn twice counts twice; graph means undefined under `nan_policy="omit"` are left
out of the replicate). The finite-size-scaling bootstrap also recomputes each replicate's
cluster error from the drawn graph means (drawn copies count as separate graphs).

**Proposition 9 (pairing survives; derived here).** If the same draws are applied to two
estimates on the same graphs, the replicate differences are the replicates of the paired
difference: $\hat\mu_2^{\ast b} - \hat\mu_1^{\ast b} = \bar D^{\ast b}$.

*Proof.* Each replicate is the mean of the drawn graphs' values, a linear function of the graph
values with the same coefficients for both estimates. ∎

So one draw matrix per $(n, \beta)$, shared by every rate, observable and checkpoint, keeps all
of their correlations in the replicates (v1.4 §5.8, §9). Different sizes have different graph
collections; the finite-size-scaling bootstrap draws them from `seed=[seed, n]`.

**Proposition 10 (derived here).** Over all $G^G$ equally likely resamples, the replicate mean has
mean $\hat\mu$ and variance $\frac{G-1}{G}\widehat{\mathrm{SE}}^2$.

*Proof.* A replicate is the mean of $G$ independent draws from the empirical law of the $\hat f_g$,
whose mean is $\hat\mu$ and variance $\hat\sigma^2 = \frac1G\sum_g(\hat f_g - \hat\mu)^2$; the
variance of the replicate mean is $\hat\sigma^2/G = \frac{G-1}{G}\widehat{\mathrm{SE}}^2$. ∎

Example §11 (f) checks Proposition 10 exactly by enumerating all $4^4$ resamples of four graphs.

### 7.3 Percentile intervals

`stats.py::percentile_interval(replicates, level)` returns the $(1-\mathrm{level})/2$ and
$(1+\mathrm{level})/2$ quantiles along the replicate axis, each the linear interpolation between
order statistics (Hyndman and Fan's type 7, numpy's default method), computed in the module so
it does not change with the numpy version; NaN replicates are ignored.

## 8. Peak-time summaries

For a robustness trace $d(t)$ on the positive checkpoints $\mathcal T_+$, v1.4 §5.8 defines
$d^\ast = \max_{t\in\mathcal T_+} d(t)$, the maximizing set
$\mathcal A = \lbrace t \in \mathcal T_+ : d(t) = d^\ast\rbrace$ and
$T^\ast = \exp\big(\frac{1}{\lvert\mathcal A\rvert}\sum_{t\in\mathcal A}\ln t\big)$, missing when
$d^\ast = 0$. It names three different statistics, which `stats.py::peak_time_summaries` keeps
apart (all with graph-equal weights):

* the arithmetic mean $\mathbb{E}[T^\ast]$: `PeakTimeSummary.arithmetic`, the nested mean of $T^\ast$;
* the geometric mean $\exp(\mathbb{E}[\ln T^\ast])$: `geometric`, from the nested mean of
  $\ln T^\ast$ (`log_mean`), with the delta-method error
  $\exp(\hat m)\,\widehat{\mathrm{SE}}(\hat m)$;
* $\operatorname{argmax}_{t\in\mathcal T_+}\mathbb{E}[d(t)]$: `argmax`, the maximizer of the nested
  mean trace `mean_trace` over the positive checkpoints (the first one; `argmax_set` lists every
  exact maximizer of the averaged values and `argmax_at_horizon` says whether $T_{\max}$ is one).

**Proposition 11 (derived here).** For positive values and any positive weights summing to 1,
$\exp\big(\sum_i w_i \ln T_i\big) \le \sum_i w_i T_i$, with equality only when all $T_i$ with
$w_i > 0$ are equal. Hence `geometric` $\le$ `arithmetic` on the same trajectories.

*Proof.* Jensen's inequality for the concave $\ln$:
$\sum_i w_i\ln T_i \le \ln\sum_i w_i T_i$, strict unless the $T_i$ are equal; exponentiate. The
nested mean is such a weighted average (§3.1). ∎

**Proposition 12 (derived here).** $\operatorname{argmax}_t\mathbb{E}[d(t)]$ is not a function of
the law of $T^\ast$: two ensembles with the same law of $T^\ast$ can have different maximizers of
the mean trace.

*Proof.* Checkpoints $\lbrace 1, 2, 4, 8\rbrace$; two trajectories with traces
$(0, 5, 0, 0)$ and $(0, 0, 0, 4)$ have $T^\ast \in \lbrace 2, 8\rbrace$ and a mean trace
$(0, 2.5, 0, 2)$ maximized at 2. Replacing the second trace by $(0, 0, 0, 6)$ keeps
$T^\ast \in \lbrace 2, 8\rbrace$ and moves the maximizer of $(0, 2.5, 0, 3)$ to 8. ∎

**Missing and horizon peaks.** Missing $T^\ast$ (NaN) are left out of the means, which are then
conditional on a defined peak; `missing_fraction` is the graph-equal-weight fraction left out. A
peak whose $\mathcal A$ contains $T_{\max}$ may continue beyond the horizon, so its $T^\ast$ is a
lower bound. `horizon_policy` makes the choice explicit (v1.4 §9 "Finite-size scaling" row):
`"refuse"` (the default) withholds the means when any defined peak is at the horizon,
`"exclude"` averages the interior peaks only (conditional on an interior peak), `"include"` keeps
them and says the means are lower bounds. `horizon_fraction` is the graph-equal-weight fraction
of trajectories with a defined peak at the horizon. `argmax_replicates(draws)` gives the bootstrap
distribution of the argmax.

## 9. Algorithms

### 9.1 The nested mean (`nested_mean`)

1. Check the labels (non-negative integers, one per row) and the values (one row per trajectory,
   any trailing shape, finite or NaN).
2. Sort the rows by $(g, c, r)$ (`np.lexsort`); refuse a repeated triple.
3. Find the first row of each $(g, c)$ and the first gate sequence of each graph.
4. Per column: sums and counts of the defined values per gate sequence (`np.add.reduceat`), their
   means; sums and counts of the defined gate-sequence means per graph, their means $\hat f_g$.
5. $\hat\mu$, $\widehat{\mathrm{SE}}$ and the contributing graphs from the defined $\hat f_g$.

Cost $O(N\log N + NK)$ for $N$ rows and $K$ values per row; exact up to floating-point sums in a
fixed order, so the result is bitwise invariant under any permutation or chunking of the rows.

### 9.2 Kaplan–Meier (`weighted_kaplan_meier`, `nested_kaplan_meier`)

1. Check the depths (finite, $\ge 0$), the flags and the positive weights; sort the rows by
   depth, events before censorings at a depth, cluster and weight.
2. Suffix sums of the weights give $Y_j$ (from the first row at $t_j$) and $Y_j - D_j$ (from the
   first censoring at $t_j$, or the next depth): both are partial sums of one sequence, so
   $Y_j - D_j = 0$ exactly when nobody survives $t_j$. With integer weights these sums are exact
   integers.
3. $\hat S(t_j)$ = cumulative product of $(Y_j - D_j)/Y_j$; with integer weights, the exact check
   at $1/2$ of §6.4.
4. Median, upper median; the restricted mean as the area under the steps up to $h$.
5. Influences by chunks of clusters (about $2^{20}$ entries per $(\text{clusters}, J)$ array): the
   clusters' at-risk weights from a reverse cumulative sum of each row's exit position, their
   event weights at their event index, then Proposition 6 in cumulative form; accumulate
   $\sum_g U_g^2$ and the squared influences on the restricted mean.

Cost $O(N\log N + GJ)$ time, bounded memory; the curve is exact for integer weights at $1/2$ and a
float64 product elsewhere.

### 9.3 Bootstrap (`graph_bootstrap`, `GraphMean.bootstrap`)

1. `SeedSequence(seed)` → PCG64 raw words → residues with rejection (§7.1) → labels.
2. A replicate's estimate is the mean of the drawn graph values (NaN left out).

Cost $O(BG)$ per estimate.

## 10. From the mathematics to the code

| definition or algorithm | code |
|---|---|
| nested estimator §3.1, counts per level, NaN policy §3.4 | `stats.py::nested_mean` → `NestedMean` |
| mean and error from per-graph values | `stats.py::mean_over_graphs` → `GraphMean` |
| graph-cluster error §4.1 | `stats.py::graph_cluster_se`, `GraphMean.se` |
| Student-$t$ interval §4.2 | `GraphMean.interval`, `stats.py::student_t_quantile` |
| paired difference §5 | `stats.py::paired_difference` |
| graph-equal integer weights §6.2 | `stats.py::graph_equal_weights` |
| weighted Kaplan–Meier §6.2, variance §6.3 | `stats.py::weighted_kaplan_meier`, `stats.py::_km_influence_chunks` → `SurvivalCurve` |
| the study's curve (graph-equal weights, clusters = graphs) | `stats.py::nested_kaplan_meier` |
| median, upper median, exact 1/2 §6.4 | `SurvivalCurve.median`, `.median_upper`, `stats.py::_snap_half` |
| restricted mean §6.5, mean without censoring | `SurvivalCurve.restricted_mean`, `.restricted_mean_se`, `.mean`, `.mean_se` |
| censoring summary §6.6 | `stats.py::censoring_summary` → `CensoringSummary` |
| bootstrap draws §7.1 | `stats.py::graph_bootstrap` (`_bounded_integers`) |
| replicates §7.2, percentile interval §7.3 | `GraphMean.bootstrap`, `stats.py::percentile_interval` |
| peak-time statistics §8 | `stats.py::peak_time_summaries` → `PeakTimeSummary` (`argmax_replicates`) |

| field | formula |
|---|---|
| `NestedMean.mean`, `.se` | $\hat\mu$ (§3.1), $\widehat{\mathrm{SE}}$ (§4.1) |
| `NestedMean.graph_means` | $\hat f_g$ |
| `NestedMean.circuits_per_graph`, `.replicas_per_circuit`, `.trajectories_per_graph` | $C_g$, $R_{gc}$, $\sum_c R_{gc}$ |
| `NestedMean.contributing_graphs`, `.contributing_trajectories` | graphs and rows with defined values |
| `SurvivalCurve.survival`, `.se` | $\hat S(t_j)$, $\big(\frac{G}{G-1}\sum_g U_g(t_j)^2\big)^{1/2}$ |
| `SurvivalCurve.at_risk`, `.events` | $Y_j/W$, $D_j/W$ |
| `SurvivalCurve.restricted_mean` | $\int_0^h\hat S$ |
| `CensoringSummary.censored_fraction` | nested mean of $\mathbf 1[\text{censored}]$ |
| `PeakTimeSummary.geometric`, `.geometric_se` | $e^{\hat m}$, $e^{\hat m}\widehat{\mathrm{SE}}(\hat m)$, $\hat m$ the nested mean of $\ln T^\ast$ |

## 11. Worked examples

### (a) Equal graph weight

Graph 0 has ten gate sequences measuring 0, graph 1 one measuring 1: the nested estimate is 1/2,
the pooled mean 1/11.

```python
import itertools
import math
from fractions import Fraction

import numpy as np

from sparsegf2 import analysis
from sparsegf2.study import stats

g = np.array([0] * 10 + [1])
c = np.array(list(range(10)) + [0])
r = np.zeros(11, dtype=int)
values = np.array([0.0] * 10 + [1.0])
est = stats.nested_mean(g, c, r, values)
assert float(est.mean) == 0.5 and math.isclose(values.mean(), 1 / 11)
assert est.circuits_per_graph.tolist() == [10, 1] and float(est.se) == 0.5
# the rows' order and chunking do not matter (bitwise)
perm = np.random.default_rng(0).permutation(11)
assert float(stats.nested_mean(g[perm], c[perm], r[perm], values[perm]).mean) == 0.5
```

### (b) Pooling is biased when the allocation follows the graph

Half the graphs (at random) have mean 1 and get four gate sequences, the others mean 0 and one
(an allocation that depends on the graph, not on observed values). Over 2000 ensembles of 40
graphs the nested estimate averages to $\mu = 1/2$ and the pooled mean to about
$\mathbb{E}[nm]/\mathbb{E}[n] = 0.8$ (Proposition 2):

```python
rng = np.random.default_rng(3)
nested, pooled = [], []
for _ in range(2000):
    kind = rng.random(40) < 0.5
    circuits = np.where(kind, 4, 1)
    g = np.repeat(np.arange(40), circuits)
    c = np.concatenate([np.arange(k) for k in circuits])
    v = kind[g] + rng.normal(0, 0.1, g.size)
    nested.append(float(stats.nested_mean(g, c, np.zeros(g.size, dtype=int), v).mean))
    pooled.append(v.mean())
nested, pooled = np.array(nested), np.array(pooled)
assert abs(nested.mean() - 0.5) < 5 * nested.std() / math.sqrt(nested.size)
assert pooled.mean() > 0.75
print(f"nested {nested.mean():.3f}, pooled {pooled.mean():.3f}")
```

### (c) The cluster error estimates the exact variance

Variance components $\sigma_a = 0.4$, $\sigma_b = 0.2$, $\sigma_e = 0.6$, an unbalanced allocation
($C_g = 1, 2, 3$, $R_{gc} = 1, 2$), 20,000 replicates as the columns of one call: the mean of
$\hat\mu$ is $\mu$, its variance the formula of §4.1, and the mean of $\widehat{\mathrm{SE}}^2$ the
same (Proposition 3):

```python
rng = np.random.default_rng(7)
graphs, reps = 12, 20_000
rows = [(gi, ci, ri) for gi in range(graphs) for ci in range(1 + gi % 3)
        for ri in range(1 + (gi + ci) % 2)]
g, c, r = (np.array(col) for col in zip(*rows))
sa, sb, se_leaf = 0.4, 0.2, 0.6
F = (1.0 + rng.normal(0, sa, (graphs, reps))[g] + rng.normal(0, sb, (graphs, 3, reps))[g, c]
     + rng.normal(0, se_leaf, (g.size, reps)))
est = stats.nested_mean(g, c, r, F)
var = sum(
    sa**2 + sb**2 / (1 + gi % 3)
    + se_leaf**2 * sum(1 / (1 + (gi + ci) % 2) for ci in range(1 + gi % 3)) / (1 + gi % 3) ** 2
    for gi in range(graphs)
) / graphs**2
assert abs(est.mean.mean() - 1.0) < 5 * math.sqrt(var / reps)
assert abs(est.mean.var() / var - 1) < 5 * math.sqrt(2 / reps)
assert abs((est.se**2).mean() / var - 1) < 0.05
lo, hi = est.interval(0.95)
print(f"coverage of the t-interval: {np.mean((lo <= 1.0) & (1.0 <= hi)):.3f}")
```

### (d) Paired rates

A graph effect shared by two rates cancels in the paired difference; combining the two errors as
if independent ignores that:

```python
rng = np.random.default_rng(22)
g = np.repeat(np.arange(30), 3)
c = np.tile(np.arange(3), 30)
r = np.zeros(90, dtype=int)
effect = rng.normal(0, 1.0, 30)[g]
low = stats.nested_mean(g, c, r, 1.0 + effect + rng.normal(0, 0.2, 90))
high = stats.nested_mean(g, c, r, 1.5 + effect + rng.normal(0, 0.2, 90))
diff = stats.paired_difference(low, high)
assert math.isclose(float(diff.mean), float(high.mean) - float(low.mean), rel_tol=1e-12)
assert float(diff.se) < 0.3 * math.hypot(float(low.se), float(high.se))
print(f"paired SE {float(diff.se):.3f}, unpaired {math.hypot(float(low.se), float(high.se)):.3f}")
```

### (e) Kaplan–Meier: exact products, one horizon, and the median at 1/2

The weighted curve against exact fractions; with unit weights it is `analysis.kaplan_meier`; with
one horizon it is the nested mean of the indicator with the §6.4 error (Propositions 5 and 7), and
the restricted mean is the nested mean of the capped depth (Proposition 8):

```python
rng = np.random.default_rng(1)
t = rng.integers(0, 12, 30).astype(float)
e = rng.random(30) < 0.6
w = rng.integers(1, 9, 30)
km = stats.weighted_kaplan_meier(t, e, w)
exact, product = [], Fraction(1)
for tj in km.times:
    at_risk = Fraction(int(w[t >= tj].sum()))
    events = Fraction(int(w[(t == tj) & e].sum()))
    product *= 1 - events / at_risk
    exact.append(float(product))
assert np.allclose(km.survival, exact, rtol=1e-14, atol=0)
ones = stats.weighted_kaplan_meier(t, e, np.ones(30, dtype=int))
assert np.allclose(ones.survival, analysis.kaplan_meier(t, e)["survival"], rtol=1e-13)

# one horizon: graph frailty, censoring only at T_max = 12
g = np.repeat(np.arange(25), 2)
c = np.tile([0, 1], 25)
r = np.zeros(50, dtype=int)
tau = rng.geometric(0.08 * np.exp(rng.normal(0, 0.5, 25))[g])
observed = tau <= 12
times = np.where(observed, tau, 12).astype(float)
km = stats.nested_kaplan_meier(g, c, r, times, observed)
for j, tj in enumerate(km.times):
    indicator = stats.nested_mean(g, c, r, (tau > tj).astype(float))
    assert math.isclose(km.survival[j], float(indicator.mean), rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(km.se[j], float(indicator.se), rel_tol=1e-9, abs_tol=1e-15)
capped = stats.nested_mean(g, c, r, np.minimum(tau, 12).astype(float))
assert math.isclose(km.restricted_mean, float(capped.mean), rel_tol=1e-12)
assert math.isclose(km.restricted_mean_se, float(capped.se), rel_tol=1e-9)
assert km.mean is None  # some trajectories are censored

# a survival of exactly 1/2 that the plain float product misses
t = np.array([3, 4, 2, 2, 1, 2, 5, 5.0])
e = np.array([1, 1, 1, 0, 1, 0, 1, 0], dtype=bool)
w = np.array([4, 3, 5, 4, 4, 3, 5, 2])
plain = 1.0
for tj in (1.0, 2.0, 3.0):
    plain *= (w[t >= tj].sum() - w[(t == tj) & e].sum()) / w[t >= tj].sum()
km = stats.weighted_kaplan_meier(t, e, w)
assert plain > 0.5 and km.survival[2] == 0.5 and (km.median, km.median_upper) == (3.0, 4.0)
print(f"plain float product {plain!r}; exact curve {km.survival[2]!r}, median {km.median}")
```

The variance is checked against the derivative of the estimate itself (Proposition 6), by
central differences along a scaling of each cluster's weights, with censoring at several depths:

```python
rng = np.random.default_rng(45)
t = rng.integers(1, 15, 80).astype(float)
e = rng.random(80) < 0.6
w = rng.uniform(0.5, 2.0, 80)
cl = rng.integers(0, 9, 80)
km = stats.weighted_kaplan_meier(t, e, w, clusters=cl, horizon=13.0)
eps, fd_s, fd_rm = 1e-6, [], []
for k in range(9):
    up = stats.weighted_kaplan_meier(t, e, w * np.where(cl == k, 1 + eps, 1), clusters=cl, horizon=13.0)
    down = stats.weighted_kaplan_meier(t, e, w * np.where(cl == k, 1 - eps, 1), clusters=cl, horizon=13.0)
    fd_s.append((up.survival - down.survival) / (2 * eps))
    fd_rm.append((up.restricted_mean - down.restricted_mean) / (2 * eps))
fd_s, fd_rm = np.array(fd_s), np.array(fd_rm)
assert np.allclose(fd_s.sum(axis=0), 0, atol=1e-7)  # the influences sum to zero
assert np.allclose(km.se, np.sqrt(9 / 8 * (fd_s**2).sum(axis=0)), rtol=1e-5, atol=1e-9)
assert math.isclose(km.restricted_mean_se, math.sqrt(9 / 8 * (fd_rm**2).sum()), rel_tol=1e-5)
```

### (f) The bootstrap: exact enumeration, and pairing

All $4^4$ resamples of four graphs: the replicates have mean $\hat\mu$ and variance
$\frac34\widehat{\mathrm{SE}}^2$ (Proposition 10). Shared draws keep two rates paired
(Proposition 9):

```python
est = stats.mean_over_graphs(np.arange(4), [1.0, 2.0, 4.0, 7.0])
every = np.array(list(itertools.product(range(4), repeat=4)))
replicates = est.bootstrap(every)
assert math.isclose(replicates.mean(), float(est.mean), rel_tol=1e-14)
assert math.isclose(replicates.var(), 3 / 4 * float(est.se) ** 2, rel_tol=1e-12)

draws = stats.graph_bootstrap(low.graphs, 500, seed=9)
assert np.array_equal(draws, stats.graph_bootstrap(low.graphs, 500, seed=9))
assert np.allclose(high.bootstrap(draws) - low.bootstrap(draws), diff.bootstrap(draws),
                   rtol=0, atol=1e-13)
lo, hi = stats.percentile_interval(diff.bootstrap(draws), 0.95)
print(f"paired difference {float(diff.mean):.3f}, 95% percentile interval [{lo:.3f}, {hi:.3f}]")
```

### (g) Three peak statistics

The counterexample of Proposition 12, and Jensen's inequality:

```python
g, c, r = np.array([0, 1]), np.zeros(2, dtype=int), np.zeros(2, dtype=int)
checkpoints = np.array([1.0, 2.0, 4.0, 8.0])
for second, expected in (([0, 0, 0, 4.0], 2.0), ([0, 0, 0, 6.0], 8.0)):
    traces = np.array([[0, 5, 0, 0.0], second])
    s = stats.peak_time_summaries(g, c, r, [2.0, 8.0], at_horizon=[False, True],
                                  traces=traces, checkpoints=checkpoints, horizon_policy="include")
    assert float(s.arithmetic.mean) == 5.0 and math.isclose(s.geometric, 4.0)
    assert s.argmax == expected
refused = stats.peak_time_summaries(g, c, r, [2.0, 8.0], at_horizon=[False, True])
assert refused.arithmetic is None and "horizon" in refused.note
```

## 12. Validation map

All in `tests/test_study_stats.py`.

| test | what it checks | how |
|---|---|---|
| `test_every_graph_has_equal_weight_whatever_its_allocation`, `test_replicas_are_averaged_before_circuits` | §3.1 weights and counts | exact values |
| `test_nested_mean_matches_a_loop_reference_on_unbalanced_hierarchies` | §3.1, §4.1 | against dictionaries and loops, relative $10^{-13}$ |
| `test_a_balanced_allocation_gives_the_plain_mean` | §3.3 balanced case | exact to rounding |
| `test_the_estimator_is_unbiased_and_its_cluster_error_estimates_the_exact_variance` | Propositions 1 and 3, interval coverage | 4000 normal replicates: $\lvert z\rvert \le 4.5$ for the mean (false alarm $7\times10^{-6}$), variance ratio within 5 standard deviations ($\approx 10^{-6}$), mean of $\widehat{\mathrm{SE}}^2$ within 4.5 standard errors, coverage in $[0.93, 0.97]$ (measured 0.948, 5.1 binomial standard deviations from the edge) |
| `test_the_result_does_not_depend_on_the_order_or_chunking_of_the_rows` | bitwise invariance (§3.1) | permuted and re-chunked rows, `np.array_equal` |
| `test_trailing_axes_are_independent_columns` | vector values | equal to column-by-column calls |
| `test_undefined_values_are_refused_unless_omitted_which_changes_the_target` | §3.4 | exact values and counts |
| `test_one_graph_has_no_graph_ensemble_error` | $G = 1$ | NaN error and interval |
| `test_paired_differences_are_formed_within_graphs_first` | §5 | against loops; paired error below 0.3 of the unpaired |
| `test_paired_intervals_cover_and_unpaired_ones_overcover` | Proposition 4 | 3000 replicates: coverage 0.944 in $[0.93, 0.97]$ (false alarm about $2\times10^{-4}$); unpaired coverage above 0.99 |
| `test_graph_bootstrap_*`, `test_the_bounded_sampler_rejects_the_top_words` | §7.1 | determinism, pinned raw stream, Pearson test (p-value $> 10^{-6}$), the rejection branch |
| `test_bootstrap_replicates_equal_the_estimator_on_resampled_rows`, `test_the_same_draws_keep_rates_paired` | §7.2, Proposition 9 | exact to rounding |
| `test_the_bootstrap_variance_matches_the_cluster_error` | Proposition 10 | 4000 replicates, ratio within 0.1 (4.5 standard deviations) |
| `test_percentile_interval_is_numpys_linear_quantile_without_nan` | §7.3 | against `np.quantile(method="linear")` |
| `test_student_t_quantile_*` | §4.2 | closed forms ($\nu = 1, 2$), tables, Simpson integral of the density |
| `test_kaplan_meier_on_a_hand_computed_example`, `test_weighted_kaplan_meier_against_exact_fractions` | §6.2 | hand values; exact Fractions |
| `test_a_survival_of_exactly_one_half_decides_the_median_exactly`, `test_near_half_values_are_decided_by_the_exact_products` | §6.4 | exact |
| `test_equal_weights_reproduce_the_unweighted_estimator` | §6.2 | against `analysis.kaplan_meier` |
| `test_with_one_horizon_the_curve_is_the_nested_mean_of_indicators`, `test_without_censoring_the_mean_is_given_and_is_the_nested_mean` | Propositions 5, 7, 8 | to rounding |
| `test_cluster_influences_are_the_derivatives_of_the_estimate` | Proposition 6 | central differences, relative $10^{-5}$; chunking invariance |
| `test_censored_geometric_times_give_the_known_median_and_restricted_mean`, `test_exponential_times_give_the_known_median` | §6.4, §6.5 on known laws | median exact (false alarm below $10^{-9}$); restricted mean within 4.5 errors; median within 4.4 standard deviations |
| `test_kaplan_meier_errors_cover_under_graph_clustering` | §6.3 under clustering and mixed censoring | 2000 replicates of 60 graphs: coverage 0.946 ($\hat S(5)$) and 0.953 (restricted mean) in $[0.93, 0.97]$ (false alarm about $10^{-3}$) |
| `test_censoring_summary_uses_graph_equal_weights` | §6.6 | exact |
| `test_the_three_peak_statistics_are_different_numbers`, `test_horizon_peaks_follow_the_declared_policy`, `test_missing_peaks_are_left_out_and_counted`, `test_argmax_replicates_follow_the_graph_draws` | §8, Propositions 11 and 12 | exact |

## 13. Limitations and departures

* **Replicas $r > 0$.** The statistics accept any $R_{gc}$; the engine produces $r = 0$ only
  (design §9 D1).
* **First-order variances.** The Kaplan–Meier and restricted-mean errors are delta-method
  (sandwich) approximations; the $t$-intervals are exact only for normal graph means of equal
  variance. The tests measure their coverage for 20 to 60 graphs; with few graphs (below about 10)
  intervals of either kind are unreliable, and the percentile bootstrap too.
* **Undefined values** (`nan_policy="omit"`) change the target (§3.4); peak means leave out
  missing peaks (a conditional target, §8).
* **Exactness at 1/2** holds for integer weights only (§6.4).
* **The median convention** is the lower median $\min\lbrace t : \hat S(t) \le 1/2\rbrace$ with
  `median_upper` for the tie interval; v1.4 leaves the convention open (§6.4 relates it to the
  JSON service's `tau_median_all`).
* **The peak horizon policy** (`refuse`, `exclude`, `include`) is a convention introduced here to
  make v1.4 §9's "explicit peak-at-horizon policy" a required choice; the default refuses.
* **Graph seeds shared across conditions** (design §3): within one $(n, \beta, p)$ the graphs are
  independent, which is all this part assumes; comparisons across $n$ or $\beta$ share random
  numbers and are paired, not independent.
* **No SciPy**: Student's $t$ quantile is computed here (§4.2) and tested against closed forms
  and tables.
