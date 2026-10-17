# Dynamic finite-size scaling

How the Watts–Strogatz study fits the dynamic scaling form of a time observable, measures the
quality of a data collapse, compares a shared set of critical parameters with independent ones,
and checks a fit by leaving sizes out, by resampling graphs and by diagnostics. Everything here
describes `src/sparsegf2/study/fss.py`; statements about the code cite `fss.py::name`.

The scientific specification is `docs/misc/watts_strogatz_study_spec_v1_4.md` ("v1.4 §x"), the
design `docs/specs/2026-10-17-ws-study-pipeline-design.md` ("design §x"). The estimates that
enter a fit (nested means, survival-based estimates, graph-cluster errors, the bootstrap over
graphs) are [06-statistics.md](06-statistics.md). The Python blocks run as written, in order, in
one namespace, in a few seconds; blocks marked `<!-- not run -->` reproduce the calibration of
§10 and take about a minute.

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Notation](#2-notation)
3. [The scaling hypothesis](#3-the-scaling-hypothesis)
4. [The collapse quality](#4-the-collapse-quality)
5. [The bounded Nelder–Mead](#5-the-bounded-neldermead)
6. [Fits and their statuses](#6-fits-and-their-statuses)
7. [Shared versus independent exponents](#7-shared-versus-independent-exponents)
8. [Leaving one size out](#8-leaving-one-size-out)
9. [The bootstrap of a fit](#9-the-bootstrap-of-a-fit)
10. [Calibration on synthetic data](#10-calibration-on-synthetic-data)
11. [Diagnostics, and what they can and cannot tell apart](#11-diagnostics-and-what-they-can-and-cannot-tell-apart)
12. [Algorithms](#12-algorithms)
13. [From the mathematics to the code](#13-from-the-mathematics-to-the-code)
14. [Worked examples](#14-worked-examples)
15. [Validation map](#15-validation-map)
16. [Limitations and departures](#16-limitations-and-departures)

## 1. Purpose and scope

v1.4 §5.9 asks, for each fixed $\beta$ and each time observable $\theta$ (the first-passage
depths $\tau_\ell$, the peak times $T^\ast$), to test a near-critical dynamic scaling hypothesis
with explicit accounting for censoring and boundary-limited peaks; v1.4 §5.8 asks to compare
exponents per fixed $\beta$ with a graph-cluster bootstrap paired across rates and with
held-out-size collapse diagnostics, and to test a common exponent set as a hypothesis against
independent fits; v1.4 §9 ("Finite-size scaling") fixes that fits are per $\beta$, with
graph-cluster uncertainty paired across $p$, explicit censoring and horizon-peak policies, and a
shared-versus-separate comparison. Design §7 names the method: the collapse quality of Houdayer
and Hartmann, a bounded Nelder–Mead written here (no SciPy), multi-start, leave-one-size-out, the
paired bootstrap, diagnostics, and exploratory and confirmatory runs told apart by a recorded
label. This part implements exactly that.

## 2. Notation

| symbol | meaning | type | in the code |
|---|---|---|---|
| $n$, $p$, $\beta$ | system size, measurement rate, rewiring probability (fixed in a fit) | integer, reals | `ScalingData.n`, `.p`, `.beta` |
| $y$, $s$ | the estimate of $\mathbb{E}[\theta \mid n, \beta, p]$ at a point and its standard error | reals, $s > 0$ | `ScalingData.y`, `.se` |
| $P$ | number of points $(n, p)$ | integer | `ScalingData.size` |
| $\boldsymbol\theta = (p_c, z, \nu)$ | critical rate, dynamic exponent, correlation-length exponent | reals, $\nu > 0$ | `CollapseFit.p_c`, `.z`, `.nu`; `PARAMETERS` |
| $x$, $Y$, $dY$ | scaled abscissa $(p - p_c)n^{1/\nu}$, ordinate $y\,n^{-z}$, error $s\,n^{-z}$ | reals | `fss.py::scaled_coordinates` |
| $F_\theta$ | the scaling function of observable $\theta$ (never parametrized) | | |
| $\mathcal B_i$ | the bracketing points of point $i$ (from the other sizes) | set | `fss.py::_Collapse.chi2` |
| $\bar Y_i$, $d\bar Y_i^2$ | master-curve estimate at $x_i$ and its variance | reals | (inside `chi2`) |
| $\chi^2_i$ | $(Y_i - \bar Y_i)^2/(dY_i^2 + d\bar Y_i^2)$ | real | `CollapseCost.chi2` |
| $S(\boldsymbol\theta)$ | the collapse quality, the mean of $\chi^2_i$ over the $N$ bracketed points | real | `CollapseCost.cost`, `CollapseFit.cost` |
| $N_{\min}$ | bracketed points a parameter vector needs to be admissible | integer | `min_points` |
| $[\ell_k, h_k]$ | the box of parameter $k$ | reals | `bounds` |
| $\varphi$ | the unbounded variables of the optimizer | reals | (inside `minimize_bounded`) |
| $\Delta$ | shared cost minus pooled independent cost | real | `SharedComparison.delta` |
| $B$ | bootstrap replicates | integer | `replicates` |

## 3. The scaling hypothesis

At fixed $\beta$, for a time observable $\theta$ with ensemble mean
$y(n, p) = \mathbb{E}[\theta \mid n, \beta, p]$ (v1.4 §5.9),

$$
y(n, p) \simeq n^{z}\, F\big((p - p_c)\, n^{1/\nu}\big),
$$

the dynamic finite-size-scaling form of Hashizume, Bentsen and Daley (PRR 2022, Eq. (5)) for the
single-reference purification time, which v1.4 generalizes *as an unverified ansatz* to the
full-reference $\tau_\ell$ and to the robustness peak times. The scaling function $F$ is unknown
and is never parametrized: a collapse asks only that the points of all sizes fall on one curve in
the coordinates

$$
x = (p - p_c)\,n^{1/\nu}, \qquad Y = y\,n^{-z}, \qquad dY = s\,n^{-z}.
$$

Each observable has its own $F$; the parameters may or may not be shared (§7). Small-world
geometry can add a crossover scale (the number of rewired edges is of order $n\beta$, v1.4 §5.9),
so a single power-law collapse may hold only in a window of sizes and rates: choose the window
before fitting (`ScalingData.subset`) and record it.

**What $y$ must be.** v1.4 §5.9 forbids fitting the mean of the observed depths when censoring is
appreciable. `ScalingData.estimator` records which estimate was used: a nested mean when nothing
is censored; otherwise a survival-based estimate, such as the restricted mean
$\mathbb{E}[\min(\tau, T_{\max})]$ or the Kaplan–Meier median ([06-statistics.md](06-statistics.md)
§6), whose own scaling is then what the fit tests. The censored and horizon-peak fractions of each
point travel with the data (`censored_fraction`, `horizon_fraction`) into the diagnostics (§11).

## 4. The collapse quality

### 4.1 Definition (Houdayer and Hartmann, PRB 70, 014418 (2004), as implemented)

For point $i$ of size $n_i$ and every *other* size $n' \ne n_i$, let $x'_0 \le \dots \le x'_{m-1}$
be the scaled abscissas of size $n'$ (in the order of their rates: at fixed $n$, $x$ increases
with $p$ for every $\boldsymbol\theta$). Size $n'$ contributes to $\mathcal B_i$:

* its two consecutive points with $x'_j \le x_i < x'_{j+1}$, $j$ the last index with
  $x'_j \le x_i$;
* when $x_i = x'_{m-1}$ exactly (the top of $n'$'s range), its last two points;
* when $n'$ has a single point, that point if $x'_0 = x_i$;
* nothing otherwise: no extrapolation beyond a size's range.

Through the points $(u_l, Y_l)$ of $\mathcal B_i$, $u_l = x_l - x_i$, with weights
$w_l = 1/dY_l^2$, the weighted least-squares line gives at $u = 0$

$$
\bar Y_i = \bar y - b\,\bar u, \qquad d\bar Y_i^{2} = \frac1K + \frac{\bar u^{2}}{S_{uu}},
$$

with $K = \sum_l w_l$, $\bar u = \sum_l w_l u_l / K$, $\bar y = \sum_l w_l Y_l / K$,
$S_{uu} = \sum_l w_l(u_l - \bar u)^2$, $b = \sum_l w_l (u_l - \bar u)(Y_l - \bar y)/S_{uu}$; when
every $u_l = 0$ (so $S_{uu} = 0$) the line degenerates and $\bar Y_i = \bar y$, $d\bar Y_i^2 = 1/K$.
Then

$$
\chi^2_i = \frac{(Y_i - \bar Y_i)^2}{dY_i^2 + d\bar Y_i^2}, \qquad
S(\boldsymbol\theta) = \frac1N\sum_{i:\,\mathcal B_i \ne \emptyset}\chi^2_i ,
$$

over the $N$ points with a nonempty $\mathcal B_i$ (`fss.py::collapse_cost`). Houdayer and
Hartmann write the same line with the raw sums $K, K_x, K_y, K_{xx}, K_{xy}$; the code uses the
centered two-pass form, which is algebraically identical (Proposition 1) and avoids the
cancellation of $K K_{xx} - K_x^2$ when the abscissas are large compared with their spread.

**Proposition 1 (derived here).** $\bar Y_i$ and $d\bar Y_i^2$ are the value at $x_i$ of the
weighted least-squares line through $\mathcal B_i$ and its variance when the $Y_l$ are independent
with variances $dY_l^2$; they equal Houdayer and Hartmann's
$\bar Y = \big[K_{xx}K_y - K_xK_{xy} + x_i(KK_{xy} - K_xK_y)\big]/\Delta$ and
$d\bar Y^2 = (K_{xx} - 2x_iK_x + x_i^2K)/\Delta$, $\Delta = KK_{xx} - K_x^2$.

*Proof.* In centered form the fitted line is $\bar y + b(u - \bar u)$; $\bar y$ and $b$ are
uncorrelated with $\operatorname{Var}\bar y = \sum_l w_l^2 dY_l^2/K^2 = 1/K$ and
$\operatorname{Var} b = 1/S_{uu}$, so the value at $u = 0$ has variance $1/K + \bar u^2/S_{uu}$.
With $x = u + x_i$: $\Delta = K S_{uu}$ (both equal $K\sum_l w_l(x_l - \bar x)^2$), and expanding
the raw expressions in $\bar x = \bar u + x_i$ gives the centered ones. ∎

`test_collapse_cost_matches_the_houdayer_hartmann_formulas` compares the code with a loop
implementation of the raw formulas.

### 4.2 Properties

**Proposition 2 (units; derived here).** $S$ does not change when every $y$ and $s$ is multiplied
by the same constant $\lambda > 0$.

*Proof.* $Y$, $dY$, $\bar Y$ and $d\bar Y$ all scale by $\lambda$ (the weights by $\lambda^{-2}$,
which cancels in $\bar u$, $\bar y$, $b$), so each $\chi^2_i$ is unchanged. ∎

**Proposition 3 (about 1 for a collapse within the errors; derived here).** If at the true
$\boldsymbol\theta$ the points are $Y_l = F(x_l) + \epsilon_l$ with independent, mean-zero errors of
variance $dY_l^2$, and $F$ is linear across each bracket, then $\mathbb{E}\chi^2_i = 1$.

*Proof.* $\bar Y_i$ uses only other sizes' points, so it is independent of $\epsilon_i$; for a
linear $F$ it is unbiased for $F(x_i)$ with variance $d\bar Y_i^2$ (Proposition 1). So
$\mathbb{E}(Y_i - \bar Y_i)^2 = dY_i^2 + d\bar Y_i^2$. ∎

Curvature of $F$ across a bracket adds a bias (dense rates make it small; §14 (a)). The
$\chi^2_i$ of different points share bracketing points, and in the study the errors of one size
are correlated across rates (the rates share graphs, [06-statistics.md](06-statistics.md) §5), so
$S$ is not a calibrated chi-square statistic: its *minimum* over $\boldsymbol\theta$ can fall well
below 1 when correlated shifts of a size are absorbed by the parameters (§10). Uncertainties come
from the bootstrap (§9), not from the curvature of $S$.

**Proposition 4 (piecewise continuity; derived here).** $S$ is continuous in $\boldsymbol\theta$
except where some $\mathcal B_i$ changes, that is where some $x_i$ crosses an abscissa of another
size or the end of a size's range. Where only one other size brackets $x_i$, $\bar Y_i$ is the
linear interpolation of that size and stays continuous through a switch of its pair; with two or
more bracketing sizes, switching one size's pair changes the fitted line, and $S$ jumps.

*Proof.* Between changes of the $\mathcal B_i$ every quantity is a continuous function of
$\boldsymbol\theta$. With one bracketing size, at a switch $x_i$ equals one of its abscissas and
both adjacent pairs interpolate to that point's value, with the same variance. With two or more,
the least-squares line through the new set of points differs in general from the old one. ∎

At $p_c$ equal to a measured rate, every size's point at that rate has $x = 0$, so many brackets
switch together: the fitted $p_c$ often sits at the edge of such a jump, at a measured rate to
within rounding (§14 (b) shows a jump of 0.025 in $S$ across $p_c = 0.18$ and a fit landing on
0.18 to within $10^{-9}$). This is a property of Houdayer and Hartmann's quality, kept as published;
multi-start (§5) and the bootstrap (§9) treat the resulting roughness. The Monte Carlo of §10
finds no bias in $\hat p_c$ from it.

### 4.3 Admissibility

A parameter vector is admissible only when at least $N_{\min}$ points are bracketed
(`min_points`, default $\max(3, \lceil P/2\rceil)$, a convention chosen here): the mean of a few
$\chi^2_i$ says little, and parameters that push most sizes' scaled ranges apart would otherwise
win on a handful of points. Inadmissible vectors count as $+\infty$. For several observables
(§7) each needs its own $N_{\min}$.

## 5. The bounded Nelder–Mead

### 5.1 The box and the reparametrization

Each free parameter ($\ell_k < h_k$) is written

$$
\theta_k = \ell_k + (h_k - \ell_k)\,\frac{1 + \sin\varphi_k}{2}, \qquad \varphi_k \in \mathbb{R},
$$

and a parameter with $\ell_k = h_k$ is held fixed (`fss.py::minimize_bounded`). This is MINUIT's
transformation for doubly bounded parameters.

**Proposition 5 (derived here).** The map is smooth and maps $\mathbb R$ onto $[\ell_k, h_k]$,
reaching both bounds at finite $\varphi_k$ ($\pm\pi/2$); hence
$\inf_{\boldsymbol\varphi} f(\boldsymbol\theta(\boldsymbol\varphi)) = \inf_{\text{box}} f$, and a
minimizer on a bound corresponds to a stationary point in $\varphi$.

*Proof.* $\sin$ maps $\mathbb R$ onto $[-1, 1]$ and is smooth; the affine map takes $[-1, 1]$ onto
$[\ell_k, h_k]$. Surjectivity gives the equality of infima. At $\varphi_k = \pm\pi/2$ the
derivative $\cos\varphi_k$ vanishes. ∎

The optimizer can thus reach a bound exactly (tests: a minimum outside the box lands on its
corner), and Nelder–Mead runs unconstrained in $\varphi$. The map is periodic, which is harmless.

### 5.2 Nelder–Mead

`fss.py::_nelder_mead` is the method of Lagarias, Reeds, Wright and Wright (SIAM J. Optim. 9,
112 (1998)) with reflection 1, expansion 2, contraction 1/2 and shrink 1/2:

1. Sort the $d+1$ vertices by value (a stable sort: ties keep their order).
2. Stop when every vertex is within `xatol` ($10^{-10}$ in $\varphi$) of the best in every
   coordinate and every value within $\texttt{fatol}\cdot\max(1, \lvert f_{\mathrm{best}}\rvert)$
   ($10^{-12}$), or when the budget (`max_evaluations`, 2000 per start) is spent.
3. Reflect the worst vertex through the centroid of the others: $x_r = \bar x + (\bar x - x_w)$.
   If $f_r$ beats the best, try the expansion $\bar x + 2(x_r - \bar x)$ and keep the better of the
   two; if $f_r$ beats the second worst, keep $x_r$.
4. Otherwise contract outside ($\bar x + (x_r - \bar x)/2$, kept if not worse than $f_r$) when
   $f_r$ beats the worst, inside ($\bar x + (x_w - \bar x)/2$, kept if it beats the worst) when it
   does not; if the contraction fails, shrink every vertex halfway to the best.

Non-finite values count as $+\infty$. Starts: the rows of `initial`, or the center of the box, then
`starts` points (8 by default) uniform in the box, from the stable stream of `seed`
([06-statistics.md](06-statistics.md) §7.1). Each start's simplex has steps of 0.25 in $\varphi$
toward the interior. The best start wins (the first on ties) and is polished: a simplex with steps
0.05 is rebuilt at its best vertex, up to `restarts` (2) times while that improves the value.
Everything is deterministic given the seed. A coordinate within $10^{-6}$ of its box width from a
bound is reported `at_bound`.

## 6. Fits and their statuses

`fss.py::fit_collapse(data, bounds=...)` minimizes $S$ over the box and returns a `CollapseFit`:
the parameters, $S$ at them, the points used and the settings (bounds, $N_{\min}$ resolved and as
given, random starts, seed, budget), with the data's `observable`, `beta`, `label` and
`estimator`. It never raises for lack of data; its `status` says what happened:

| status | parameters | meaning |
|---|---|---|
| `ok` | yes | converged inside the box |
| `at_bound` | yes | some parameter on a box bound (named in `at_bound`): the box is too small, or the data do not constrain that parameter; widen and refit |
| `not_converged` | yes | the evaluation budget ran out before the tolerances were met |
| `too_few_points` | no | fewer points than $N_{\min}$, or no admissible vector in the box (the sizes' scaled ranges never overlap enough) |
| `too_few_sizes` | no | fewer than two sizes: there is nothing to collapse |

Two sizes give a fit, but three parameters are then weakly constrained (a collapse of two curves
leaves a family of near-equivalent parameter sets); use at least three sizes, and more to see a
drift (§11). `CollapseFit.usable` is true for the first three statuses.

## 7. Shared versus independent exponents

### 7.1 The shared fit

`fss.py::fit_shared(datasets, bounds=...)` fits one $\boldsymbol\theta$ to several observables at
one $\beta$ (refusing data from different $\beta$), each with its own $F$: the quality is evaluated
per observable and pooled,

$$
S_{\mathrm{shared}}(\boldsymbol\theta) = \frac{\sum_o\sum_{i \in o}\chi^2_{oi}(\boldsymbol\theta)}{\sum_o N_o(\boldsymbol\theta)},
$$

with each observable needing its own $N_{\min}$. This is v1.4's shared-criticality hypothesis.

### 7.2 The statistic

With independent fits $\hat{\boldsymbol\theta}_o$ and their pooled quality
$S_{\mathrm{ind}} = \sum_o N_o S_o(\hat{\boldsymbol\theta}_o)/\sum_o N_o$,

$$
\Delta = S_{\mathrm{shared}}(\hat{\boldsymbol\theta}_s) - S_{\mathrm{ind}}
$$

measures how much the collapse worsens when the parameters are shared (`SharedComparison.delta`).
$\Delta$ is near 0 when the observables' best parameters agree within their noise and large when
they do not; but $S$ is not a calibrated chi-square (§4.2), so $\Delta$ needs a null distribution.

### 7.3 The bootstrap null

`fss.py::compare_shared_independent` builds data that satisfy the shared hypothesis exactly and
resamples them. Each observable $o$ is mapped point by point to

$$
p' = \hat p_c^{\,s} + (p - \hat p_c^{\,o})\,n^{1/\hat\nu_o - 1/\hat\nu_s}, \qquad
y' = y\,n^{\hat z_s - \hat z_o}, \qquad s' = s\,n^{\hat z_s - \hat z_o},
$$

and its graph means by the same factor.

**Proposition 6 (derived here).** The mapped data have, at $\hat{\boldsymbol\theta}_s$, exactly
the scaled coordinates that observable $o$ has at $\hat{\boldsymbol\theta}_o$; hence
$S'_o(\hat{\boldsymbol\theta}_s) = S_o(\hat{\boldsymbol\theta}_o)$, and along
$p_c = \hat p_c^{\,s}$ the map $(z, \nu) \mapsto (z + \hat z_o - \hat z_s,\ \nu')$ with
$1/\nu' = 1/\nu + 1/\hat\nu_o - 1/\hat\nu_s$ carries $S'_o$ onto $S_o$.

*Proof.* $x' = (p' - \hat p_c^{\,s})\,n^{1/\hat\nu_s} = (p - \hat p_c^{\,o})\,n^{1/\hat\nu_o} = x$ and
$Y' = y'\,n^{-\hat z_s} = y\,n^{-\hat z_o} = Y$, likewise $dY' = dY$; the quality depends on the
data only through these. For the second claim repeat the computation with $(\hat p_c^{\,s}, z, \nu)$. ∎

So the mapped data keep each observable's collapse quality and noise, and the shared parameters
are exactly right for them: the independent refits of the mapped data land on
$\hat{\boldsymbol\theta}_s$ (checked in
`test_the_null_mapping_moves_each_observable_onto_the_shared_parameters`). Each of $B$ replicates
then resamples the graphs once per size (`graph_bootstrap(labels_n, B, seed=[seed, n])`, applied
to every rate and every observable at that size, so the pairing across rates and observables
survives, [06-statistics.md](06-statistics.md) §7.2), recomputes every point's mean and error from
the drawn graphs, refits the shared and the independent parameters from $\hat{\boldsymbol\theta}_s$
(no random starts), and records $\Delta^\ast$. With $B'$ usable replicates

$$
\text{p-value} = \frac{1 + \#\lbrace\Delta^\ast \ge \Delta\rbrace}{1 + B'},
$$

and `prefers` is `"independent"` when the p-value is at most `alpha` (0.05), `"shared"` otherwise,
`"undetermined"` when a fit gave no parameters or every replicate failed. This follows the usual
rule for bootstrap tests: resample from a distribution that satisfies the null, and keep the
statistic's form. It is an approximation: the noise is the resampled noise of the observed data,
the null center is built from the fitted parameters, and the refits start at
$\hat{\boldsymbol\theta}_s$. §10 compares it with the true null distribution on synthetic data.

## 8. Leaving one size out

`fss.py::leave_one_size_out` refits without each size in turn (from the full fit and `starts`
random starts) and reports the refits' parameters, the `spread` (max minus min) of each parameter
over them, and the *held-out cost*: the mean $\chi^2_i$ of the left-out size's points against the
master curve of the remaining sizes at the refit's parameters (each such point is bracketed only
by the other sizes, which are exactly the training sizes). On data that obey the scaling form
within their errors the refits agree and the held-out costs are of order 1 (on exact data with
1% errors they are below 0.05, §14 (e)); a held-out cost far above 1, or parameters that move
systematically with the left-out size, flag corrections to scaling, a crossover or a size outside
the scaling window (v1.4 §5.8: held-out-size collapse diagnostics). With fewer than three sizes
the refits have status `too_few_sizes`.

## 9. The bootstrap of a fit

`fss.py::bootstrap_fit` resamples graphs once per $(n, \beta)$: for each size, one draw matrix
(`graph_bootstrap(labels_n, B, seed=[seed, n])`) applied to every rate's graph means, so each
replicate's means and errors at that size come from the same drawn graphs and the coupling of the
rates (shared graphs, gates and mask uniforms, v1.4 §7.3) is kept. Every point of a size must have
the same graph labels (the rates were measured on the same graphs); the data must carry the
per-graph values (`fss.py::scaling_data`). Each replicate is refitted from the full fit with no
random starts; a point whose replicate error is 0 or undefined is left out of that replicate. The
result has the replicate parameters, their standard deviations (`se`), percentile intervals
(`lower`, `upper`), and counts of failed and at-bound replicates.

## 10. Calibration on synthetic data

The synthetic ensemble below has 20 graphs per point, a graph effect of 5% shared by every rate of
a size (as the coupled rates of the study share their graphs) and 3% rate noise, four sizes
16–128 and 21 rates in $[0.1, 0.3]$, with $(p_c, z, \nu) = (0.2, 1, 1.5)$. Over 40 such datasets
the fits are unbiased and the bootstrap errors (60 replicates, one dataset) are 0.85–0.91 of the
fits' true standard deviations:

| | $p_c$ | $z$ | $\nu$ |
|---|---|---|---|
| mean of 40 fits | 0.1991 | 1.0011 | 1.5200 |
| standard deviation of 40 fits | 0.0163 | 0.0432 | 0.1095 |
| bootstrap error (one dataset) | 0.0139 | 0.0377 | 0.0993 |

For the comparison, 12 fresh pairs of observables with the *same* parameters (and different $F$)
gave $\Delta$ between 0.03 and 0.56 (median 0.19); the bootstrap null of one pair (19 replicates)
ranged from 0.06 to 0.83 (median 0.24), and that pair's $\Delta = 0.39$ had p-value 0.25. Under the
alternative ($\nu = 1.2$ against $2.5$, 2% graph effects) $\Delta$ was 5.1 against a null whose
largest of 39 replicates was 0.52. The code (about a minute):

<!-- not run -->

```python
import numpy as np
from sparsegf2.study import fss
from sparsegf2.study import stats as st


def F1(x):
    return 2.0 / (1.0 + np.exp(x)) + 0.5


def F2(x):
    return np.exp(-0.5 * x) + 0.2


def synthetic(params, F, sizes=(16, 32, 64, 128), rates=np.linspace(0.1, 0.3, 21), graphs=20,
              sg=0.05, sp=0.03, seed=0, name="y"):
    pc, z, nu = params
    rng = np.random.default_rng(seed)
    means, n_, p_ = [], [], []
    for n in sizes:
        effect = rng.normal(0, sg, graphs)  # shared by every rate of the size
        for p in rates:
            truth = n**z * F((p - pc) * n ** (1 / nu))
            values = truth * (1 + effect + rng.normal(0, sp, graphs))
            means.append(st.mean_over_graphs(np.arange(graphs), values))
            n_.append(n)
            p_.append(p)
    return fss.scaling_data(np.array(n_), np.array(p_), means, observable=name, beta="0.1")


box = {"p_c": (0.1, 0.3), "z": (0.0, 2.0), "nu": (0.5, 3.0)}
fits = np.array([fss.fit_collapse(synthetic((0.2, 1.0, 1.5), F1, seed=100 + s), bounds=box,
                                  starts=4).params for s in range(40)])
boot = fss.bootstrap_fit(synthetic((0.2, 1.0, 1.5), F1, seed=100), bounds=box, replicates=60,
                         seed=1, starts=4)
print("mean", fits.mean(axis=0), "sd", fits.std(axis=0, ddof=1))
print("bootstrap se", boot.se, "ratio", boot.se / fits.std(axis=0, ddof=1))

deltas = []
for s in range(12):
    a = synthetic((0.2, 1.0, 1.5), F1, seed=300 + s, name="a")
    b = synthetic((0.2, 1.0, 1.5), F2, seed=400 + s, name="b")
    fa, fb = (fss.fit_collapse(d, bounds=box, starts=3) for d in (a, b))
    shared = fss.fit_shared([a, b], bounds=box, starts=3)
    pooled = (fa.cost * fa.points_used + fb.cost * fb.points_used) / (fa.points_used + fb.points_used)
    deltas.append(shared.cost - pooled)
cmp = fss.compare_shared_independent([synthetic((0.2, 1.0, 1.5), F1, seed=300, name="a"),
                                      synthetic((0.2, 1.0, 1.5), F2, seed=400, name="b")],
                                     bounds=box, replicates=19, seed=5, starts=3)
print("true null of delta", np.round(np.sort(deltas), 3), "median", np.median(deltas))
print("bootstrap null", np.round(np.sort(cmp.null), 3), "median", np.nanmedian(cmp.null))
print("delta", cmp.delta, "p", cmp.p_value)
```

These are measurements on one synthetic ensemble, not guarantees: the size of $\Delta$ and its
null depend on the noise, the window and the scaling functions. In development one pair generated
with the same parameters (different seeds from these) gave $\Delta = 1.2$ against a bootstrap null
whose median was 0.135, a rejection at the minimum p-value of 39 replicates; with 12 fresh pairs
spanning 0.03–0.56, that pair lies far in the tail of the true null as well, so it was a rare draw
rather than a broken null, but it shows that a single rejection at the 5% level is weak evidence.
Report the p-value with the fits' parameter differences and their bootstrap errors.

## 11. Diagnostics, and what they can and cannot tell apart

`fss.py::collapse_diagnostics(data, fit, bootstrap=None)` reports, at the fit:

* **bracketing coverage**: which points enter $S$ (`bracketed`, their $\chi^2$), overall and per
  size (`coverage`, `coverage_by_size`). Unbracketed points are the ends of the scaled range: the
  largest sizes' extreme rates. Low coverage means the fit rests on few points of some sizes.
* **drift with the smallest size removed**: a refit without the smallest size (same settings,
  started from the fit), its parameters minus the fit's (`drift`), and in units of a bootstrap's
  errors when one is given (`drift_in_se`).
* **censoring and horizon peaks**, passed through per point (`censored_fraction`,
  `horizon_fraction`) with their maxima and a note naming the estimator, since a point with
  appreciable censoring is only as good as its censoring-aware estimate.
* `label` (exploratory or confirmatory) and `estimator`, recorded only.

What they can and cannot distinguish:

| situation | what shows it | what it cannot tell |
|---|---|---|
| a crossover or corrections to scaling (small-world scale $n\beta$, v1.4 §5.9) | drift with the smallest size removed, and leave-one-size-out parameters that move with the size, beyond their bootstrap errors; held-out costs well above 1 | whether the cause is a crossover, corrections to scaling or a window too wide: each moves the parameters with $n$; only more sizes (and a restricted window) separate them |
| too few sizes | status `too_few_sizes` (one size); with two sizes no drift and no leave-one-size-out refits | the true parameters: with two sizes a collapse can be nearly as good for a family of parameters |
| censoring | `censored_fraction` per point, carried from [06-statistics.md](06-statistics.md) §6.6 | whether the chosen estimator's scaling is the scaling of $\mathbb{E}[\theta]$: a restricted mean scales like $\mathbb{E}[\min(\tau, T_{\max})]$, which saturates at the horizon; a longer horizon (design §9 D3) is the remedy |
| horizon peaks | `horizon_fraction` per point | the true peak depth beyond the horizon: peak times there are lower bounds |
| a poor collapse | $S$ well above 1, large $\chi^2$ at particular points | which assumption failed (the ansatz, the window, the errors) |
| an unconstrained parameter | status `at_bound`, wide bootstrap intervals | whether widening the box would find a valid optimum |

None of these diagnostics proves the scaling hypothesis: a good collapse with stable parameters
is consistent with it within the sizes and rates measured.

## 12. Algorithms

### 12.1 The quality (`_Collapse.chi2`)

1. Once per data set: sort each size's points by rate (ties by value and error) into padded arrays
   (sizes $\times$ points per size).
2. Per evaluation: $n^{1/\nu}$ and $n^{-z}$ by `np.power` (exact for integer powers); if anything
   overflows, the quality is undefined (every $\chi^2$ NaN).
3. For each size, `searchsorted` of every $x_i$ in that size's sorted abscissas gives the last
   index with $x' \le x_i$; the rules of §4.1 turn it into a pair (or a single point, or nothing).
4. Gather the pairs into $(P, 2\times\text{sizes})$ arrays with a mask, and compute $K$, $\bar u$,
   $\bar y$, $S_{uu}$, $b$, $\bar Y$, $d\bar Y^2$ and $\chi^2$ in two passes.

Cost $O(P\times\text{sizes})$ per evaluation (about 60–90 µs for 84–164 points and 4 sizes on the
development machine).

### 12.2 A fit (`fit_collapse`)

1. Statuses before searching: `too_few_sizes`, `too_few_points` (fewer than $N_{\min}$ points).
2. `minimize_bounded` on the admissible quality (§4.3) from the center (or `initial`) and `starts`
   random points, polish the winner.
3. Recount the bracketed points at the result and set the status.

About 6000 evaluations for 9 starts on 84 points (0.6 s).

### 12.3 The bootstrap and the comparison

1. Per size, check that every point (of every observable) has the same graph labels; draw
   `graph_bootstrap(labels_n, B, seed=[seed, n])`.
2. Replicate means and errors of every point from its graph means at the drawn positions (one
   vectorized pass per point).
3. Refit each replicate from the full fit (comparison: map first, §7.3; refit shared and
   independent from $\hat{\boldsymbol\theta}_s$).

Cost: $B$ (or $B\times(\text{observables} + 1)$) refits of about 0.1 s each.

## 13. From the mathematics to the code

| definition or algorithm | code |
|---|---|
| points of one observable at one $\beta$ (§3) | `fss.py::ScalingData`, `fss.py::scaling_data` (from `GraphMean` results), `ScalingData.subset` |
| scaled coordinates | `fss.py::scaled_coordinates` |
| the quality §4.1 | `fss.py::collapse_cost` → `CollapseCost`; `fss.py::_Collapse.chi2` |
| the bounded optimizer §5 | `fss.py::minimize_bounded` → `MinimizeResult`; `fss.py::_nelder_mead` |
| a fit and its status §6 | `fss.py::fit_collapse` → `CollapseFit`; `STATUSES` |
| the shared fit §7.1 | `fss.py::fit_shared` → `SharedFit` |
| the comparison and its null §7.2–7.3 | `fss.py::compare_shared_independent` → `SharedComparison` |
| leave one size out §8 | `fss.py::leave_one_size_out` → `LeaveOneSizeOut` |
| the bootstrap of a fit §9 | `fss.py::bootstrap_fit` → `BootstrapFit`; draws `fss.py::_draws_by_size` |
| diagnostics §11 | `fss.py::collapse_diagnostics` → `CollapseDiagnostics` |

| field | formula |
|---|---|
| `CollapseCost.chi2`, `.cost`, `.points_used` | $\chi^2_i$ (NaN if unbracketed), $S$, $N$ |
| `CollapseFit.p_c`, `.z`, `.nu`, `.cost` | $\hat{\boldsymbol\theta} = \arg\min S$ over admissible vectors in the box, $S(\hat{\boldsymbol\theta})$ |
| `CollapseFit.min_points` | $N_{\min}$ for these data |
| `SharedFit.cost`, `.costs` | $S_{\mathrm{shared}}(\hat{\boldsymbol\theta}_s)$; each observable's own mean $\chi^2$ there |
| `SharedComparison.delta`, `.null`, `.p_value` | $\Delta$, the $\Delta^\ast$, $(1 + \#\lbrace\Delta^\ast \ge \Delta\rbrace)/(1 + B')$ |
| `LeaveOneSizeOut.held_out_cost` | mean $\chi^2$ of the left-out size at its refit |
| `BootstrapFit.se`, `.lower`, `.upper` | standard deviation and percentile interval of the usable replicates |
| `CollapseDiagnostics.drift` | parameters without the smallest size minus the fit's |

## 14. Worked examples

### (a) The quality on exact data

Exact points of $y = n^z F((p - p_c)n^{1/\nu})$ with 1% errors: $S$ is near 0 at the truth (the
curvature of $F$ across the brackets is all that is left) and large away from it; it does not
depend on the units of $y$; and the code agrees with Houdayer and Hartmann's raw formulas on a
small data set:

```python
import math

import numpy as np

from sparsegf2.study import fss
from sparsegf2.study import stats as st


def F(x):
    return 2.0 / (1.0 + np.exp(x)) + 0.5


sizes = np.array([16, 32, 64, 128])
rates = np.linspace(0.1, 0.3, 21)
n = np.repeat(sizes, rates.size)
p = np.tile(rates, sizes.size)
y = n**1.0 * F((p - 0.2) * n ** (1 / 1.5))
exact = fss.ScalingData(n=n, p=p, y=y, se=0.01 * y, observable="tau_1", beta="0.1")
at_truth = fss.collapse_cost(exact, 0.2, 1.0, 1.5)
print(f"S at the truth {at_truth.cost:.2e} over {at_truth.points_used} of {exact.size} points")
assert at_truth.cost < 0.01 and fss.collapse_cost(exact, 0.21, 1.0, 1.5).cost > 1
scaled = fss.ScalingData(n=n, p=p, y=1e6 * y, se=1e4 * y)
assert math.isclose(fss.collapse_cost(scaled, 0.21, 1.1, 1.3).cost,
                    fss.collapse_cost(exact, 0.21, 1.1, 1.3).cost, rel_tol=1e-10)


def raw_hh(data, pc, z, nu):
    """Houdayer and Hartmann with the raw sums, by loops (an independent reference)."""
    x, Y, dY = fss.scaled_coordinates(data, pc, z, nu)
    out = []
    for i in range(data.size):
        chosen = []
        for other in np.unique(data.n):
            if other == data.n[i]:
                continue
            idx = np.flatnonzero(data.n == other)
            idx = idx[np.argsort(x[idx], kind="stable")]
            pairs = [j for j in range(idx.size - 1) if x[idx[j]] <= x[i] <= x[idx[j + 1]]]
            if pairs:
                chosen += [idx[pairs[-1]], idx[pairs[-1] + 1]]
        if not chosen:
            out.append(math.nan)
            continue
        w = 1 / dY[chosen] ** 2
        K, Kx, Ky = w.sum(), (w * x[chosen]).sum(), (w * Y[chosen]).sum()
        Kxx, Kxy = (w * x[chosen] ** 2).sum(), (w * x[chosen] * Y[chosen]).sum()
        delta = K * Kxx - Kx**2
        master = (Kxx * Ky - Kx * Kxy + x[i] * (K * Kxy - Kx * Ky)) / delta
        var = (Kxx - 2 * x[i] * Kx + x[i] ** 2 * K) / delta
        out.append((Y[i] - master) ** 2 / (dY[i] ** 2 + var))
    return np.array(out)


rng = np.random.default_rng(4)
small = fss.ScalingData(n=np.repeat([8, 16, 32], 6), p=np.tile(np.linspace(0.1, 0.3, 6), 3),
                        y=rng.uniform(0.5, 2, 18), se=rng.uniform(0.05, 0.2, 18))
ours = fss.collapse_cost(small, 0.19, 0.4, 1.2).chi2
ref = raw_hh(small, 0.19, 0.4, 1.2)
assert np.array_equal(np.isnan(ours), np.isnan(ref))
assert np.allclose(ours[~np.isnan(ref)], ref[~np.isnan(ref)], rtol=1e-9)
```

### (b) A jump of the quality at a measured rate

Synthetic data with a graph hierarchy (§10's generator, written out). Holding $z$ and $\nu$, the
quality jumps where $p_c$ crosses the measured rate 0.18 (Proposition 4), and the fit sits at the
edge of the jump:

```python
def synthetic(params, F, sizes=(16, 32, 64, 128), rates=np.linspace(0.1, 0.3, 21), graphs=20,
              sg=0.05, sp=0.03, seed=0, name="y"):
    pc, z, nu = params
    rng = np.random.default_rng(seed)
    means, n_, p_ = [], [], []
    for size in sizes:
        effect = rng.normal(0, sg, graphs)  # shared by every rate of the size
        for rate in rates:
            truth = size**z * F((rate - pc) * size ** (1 / nu))
            values = truth * (1 + effect + rng.normal(0, sp, graphs))
            means.append(st.mean_over_graphs(np.arange(graphs), values))
            n_.append(size)
            p_.append(rate)
    return fss.scaling_data(np.array(n_), np.array(p_), means, observable=name, beta="0.1",
                            label="exploratory")


box = {"p_c": (0.1, 0.3), "z": (0.0, 2.0), "nu": (0.5, 3.0)}
noisy = synthetic((0.2, 1.0, 1.5), F, seed=1)
fit = fss.fit_collapse(noisy, bounds=box)
below = fss.collapse_cost(noisy, 0.18 - 1e-7, fit.z, fit.nu).cost
above = fss.collapse_cost(noisy, 0.18 + 1e-7, fit.z, fit.nu).cost
print(f"fit p_c = {fit.p_c!r}; S just below 0.18: {below:.4f}, just above: {above:.4f}")
assert below - above > 0.01 and abs(fit.p_c - 0.18) < 1e-9 and fit.status == "ok"
```

### (c) The bounded optimizer

Rosenbrock's function in a box, and a minimum outside the box that lands on its corner:

```python
res = fss.minimize_bounded(lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2,
                           [(-2, 2), (-1, 3)], seed=1)
assert np.allclose(res.x, [1, 1], atol=1e-6) and res.converged
corner = fss.minimize_bounded(lambda v: (v[0] - 3) ** 2 + (v[1] + 2) ** 2, [(0, 1), (0, 1)])
assert np.allclose(corner.x, [1, 0], atol=1e-8) and corner.at_bound.tolist() == [True, True]
```

### (d) Fits, statuses and the shared fit

```python
fit = fss.fit_collapse(exact, bounds=box)
assert fit.status == "ok" and abs(fit.p_c - 0.2) < 1e-3 and abs(fit.nu - 1.5) < 2e-2
print(f"exact data: p_c = {fit.p_c:.5f}, z = {fit.z:.5f}, nu = {fit.nu:.4f}, S = {fit.cost:.2e}")
narrow = fss.fit_collapse(exact, bounds={"p_c": (0.1, 0.3), "z": (0.0, 2.0), "nu": (2.0, 3.0)})
assert narrow.status == "at_bound" and narrow.at_bound == ("nu",)
one_size = fss.fit_collapse(exact.subset(exact.n == 32), bounds=box)
assert one_size.status == "too_few_sizes" and math.isnan(one_size.p_c)

other = fss.ScalingData(n=n, p=p, y=n * (np.exp(-0.5 * (p - 0.2) * n ** (1 / 1.5)) + 0.2),
                        se=0.01 * n, observable="T_cyc", beta="0.1")
shared = fss.fit_shared([exact, other], bounds=box)
assert shared.status == "ok" and abs(shared.p_c - 0.2) < 1e-3 and abs(shared.nu - 1.5) < 2e-2
```

### (e) The null mapping, leaving sizes out, diagnostics and the bootstrap

Proposition 6 at generic parameters; leave-one-size-out on exact data; the diagnostics of a fit;
and a (short) paired bootstrap:

```python
own, target = np.array([0.1937, 1.031, 1.462]), np.array([0.21, 0.95, 1.4])
nn = noisy.n.astype(float)
mapped = fss.ScalingData(n=noisy.n, p=target[0] + (noisy.p - own[0]) * nn ** (1 / own[2] - 1 / target[2]),
                         y=noisy.y * nn ** (target[1] - own[1]), se=noisy.se * nn ** (target[1] - own[1]))
assert math.isclose(fss.collapse_cost(mapped, *target).cost, fss.collapse_cost(noisy, *own).cost,
                    rel_tol=1e-9)

loso = fss.leave_one_size_out(exact, bounds=box, starts=2)
assert np.allclose(loso.params[:, 0], 0.2, atol=2e-3) and np.all(loso.held_out_cost < 0.05)
print("held-out costs", np.round(loso.held_out_cost, 4))

diag = fss.collapse_diagnostics(exact, fss.fit_collapse(exact, bounds=box, starts=2))
print(f"coverage {diag.coverage:.3f} by size {diag.coverage_by_size}; drift {np.round(diag.drift, 5)}")

boot = fss.bootstrap_fit(noisy, bounds=box, replicates=8, seed=3, starts=2)
assert boot.failed == 0 and np.all(boot.se > 0)
print("bootstrap errors (8 replicates)", np.round(boot.se, 4))
```

## 15. Validation map

All in `tests/test_study_fss.py`.

| test | what it checks | how |
|---|---|---|
| `test_collapse_cost_matches_the_houdayer_hartmann_formulas` | §4.1, Proposition 1 | random data and parameters against a loop implementation of the raw formulas, relative $10^{-8}$, same bracketed set |
| `test_the_bracketing_rule_at_the_edges` | §4.1 edge rules (top of a range, single points, no extrapolation) | hand-computed $\chi^2$ on binary-exact abscissas |
| `test_a_linear_scaling_function_collapses_perfectly` | Proposition 3's exact case | $S < 10^{-20}$ at the truth |
| `test_the_quality_does_not_depend_on_the_units_of_y` | Proposition 2 | relative $10^{-10}$ |
| `test_extreme_parameters_give_no_quality_instead_of_overflow`, `test_overflowing_scaled_values_give_no_quality` | §12.1 step 2 | NaN quality |
| `test_the_optimizer_finds_known_minima`, `test_a_minimum_outside_the_box_lands_on_its_bounds`, `test_fixed_coordinates_and_infinite_regions`, `test_the_optimizer_is_deterministic_given_its_seed` | §5, Proposition 5 | Rosenbrock, Himmelblau, Booth, a quadratic; corner minimum; fixed coordinates; $+\infty$ regions; bitwise determinism |
| `test_the_fit_recovers_the_parameters_of_exact_data` | §6 | $\lvert\hat p_c - 0.2\rvert < 10^{-3}$, $\lvert\hat z - 1\rvert < 10^{-2}$, $\lvert\hat\nu - 1.5\rvert < 2\times10^{-2}$ |
| `test_the_fit_recovers_the_parameters_within_its_bootstrap_errors` | §9, §10 | each parameter within 4 bootstrap errors (false alarm about $2\times10^{-3}$) |
| `test_the_bootstrap_resamples_graphs_once_per_size_for_every_rate`, `test_bootstrap_points_must_share_their_graphs` | §9 pairing | the draws are `graph_bootstrap(labels, B, [seed, n])`; a constant offset between two rates survives every replicate exactly |
| `test_fits_are_deterministic_given_the_seed`, `test_the_fit_does_not_depend_on_the_units_or_on_duplicated_points` | determinism, scaled inputs, duplicated points | bitwise; relative $10^{-6}$; duplicated data refit near the truth |
| `test_a_fixed_parameter_stays_fixed`, `test_a_truth_outside_the_box_is_reported_at_its_bound`, `test_too_few_sizes_or_points_give_a_status_not_a_crash`, `test_min_points_is_respected`, `test_a_spent_budget_is_reported_as_not_converged` | §6 statuses | each status, no exception |
| `test_the_shared_fit_recovers_a_common_set` | §7.1 | two observables with different $F$, same parameters |
| `test_the_comparison_prefers_independent_exponents_for_different_nu` | §7.2–7.3 | $\nu = 1.2$ against 2.5: p-value $1/20$ with 19 replicates, $\Delta$ above 4 times the largest null replicate |
| `test_the_null_mapping_moves_each_observable_onto_the_shared_parameters` | Proposition 6 | equal coordinates and quality at generic parameters; the refit of mapped data lands on the shared set |
| `test_leaving_one_size_out_of_exact_data_gives_the_same_parameters` | §8 | every refit near the truth; held-out costs below 0.05 |
| `test_diagnostics_report_coverage_drift_and_the_fractions_passed_through`, `test_drift_in_units_of_the_bootstrap_errors`, `test_a_failed_drift_refit_is_noted` | §11 | coverage equal to the quality's bracketed set; drift equal to the refit's difference; fractions passed through; notes |
| `test_scaling_data_*`, `test_fit_bounds_are_checked`, `test_fit_settings_are_checked`, `test_shared_tools_check_their_inputs` | input checks | messages that say what to change |

## 16. Limitations and departures

* **The ansatz is unverified** (v1.4 §5.9): a fit measures how well the data collapse, not
  whether the hypothesis is true.
* **The quality is piecewise continuous** (Proposition 4): fits can sit at the edge of a jump, at
  a measured rate to within rounding. This is Houdayer and Hartmann's quality as published;
  a smoother variant (averaging each size's linear interpolation) would change the method and is
  not used.
* **$S$ is not a calibrated chi-square**: points share brackets and the errors of a size are
  correlated across rates (shared graphs). Parameter uncertainties come from the graph bootstrap
  (§9), which on the synthetic ensemble of §10 gave 0.85–0.91 of the true standard deviations
  with 20 graphs; with fewer graphs expect smaller ratios.
* **The bootstrap null** of the comparison is an approximation (§7.3); a single rejection at 5% is
  weak evidence (§10). Report it with the parameter differences and their errors.
* **Sizes are not strictly independent**: graph seeds `(master, g)` and circuit streams `u` are
  shared across sizes (design §3), so the same labels at different $n$ use common random numbers.
  The bootstrap resamples graphs once per $(n, \beta)$, as v1.4 §5.8 and the design ask, and treats
  sizes as independent; the residual correlation through shared seeds is not modeled.
* **The scaling window is chosen by the user** (`ScalingData.subset`); nothing here selects it.
* **Conventions introduced here**: the admissibility floor $N_{\min} = \max(3, \lceil P/2\rceil)$
  by default; the optimizer's tolerances, steps and polishing (§5.2); the bracketing tie rules
  (§4.1); the drift diagnostic removes the smallest size only (leave-one-size-out covers the
  others). The `label` (exploratory or confirmatory) is recorded and changes nothing.
