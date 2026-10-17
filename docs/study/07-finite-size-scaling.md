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
§11 and the benchmark of §5.8, and take from minutes to about two hours.

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Notation](#2-notation)
3. [The scaling hypothesis](#3-the-scaling-hypothesis)
4. [The collapse quality](#4-the-collapse-quality)
5. [The polynomial master curve (ruling R-FSS)](#5-the-polynomial-master-curve-ruling-r-fss)
6. [The bounded Nelder–Mead](#6-the-bounded-neldermead)
7. [Fits and their statuses](#7-fits-and-their-statuses)
8. [Shared versus independent exponents](#8-shared-versus-independent-exponents)
9. [Leaving one size out](#9-leaving-one-size-out)
10. [The bootstrap of a fit](#10-the-bootstrap-of-a-fit)
11. [Calibration on synthetic data](#11-calibration-on-synthetic-data)
12. [Diagnostics, and what they can and cannot tell apart](#12-diagnostics-and-what-they-can-and-cannot-tell-apart)
13. [Algorithms](#13-algorithms)
14. [From the mathematics to the code](#14-from-the-mathematics-to-the-code)
15. [Worked examples](#15-worked-examples)
16. [Validation map](#16-validation-map)
17. [Limitations and departures](#17-limitations-and-departures)

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
label. Ruling R-FSS (design §11) adds a continuous objective, a polynomial master curve fitted
jointly with the parameters, as the `method="polynomial"` of every fitting function, with the two
methods' agreement as a diagnostic (§5, §12). Houdayer and Hartmann's quality stays the default
(§5.8 says why). This part implements exactly that.

## 2. Notation

| symbol | meaning | type | in the code |
|---|---|---|---|
| $n$, $p$, $\beta$ | system size, measurement rate, rewiring probability (fixed in a fit) | integer, reals | `ScalingData.n`, `.p`, `.beta` |
| $\theta$ | a time observable ($\tau_\ell$, $T^\ast$) | random variable | `ScalingData.observable` (its name) |
| $y$, $s$ | the estimate of $\mathbb{E}[\theta \mid n, \beta, p]$ at a point and its standard error | reals, $s > 0$ | `ScalingData.y`, `.se` |
| $P$ | number of points $(n, p)$ | integer | `ScalingData.size` |
| $O$ | number of observables in a shared fit or a comparison | integer | `len(datasets)` |
| $\boldsymbol\theta = (p_c, z, \nu)$ | critical rate, dynamic exponent, correlation-length exponent | reals, $\nu > 0$ | `CollapseFit.p_c`, `.z`, `.nu`; `PARAMETERS` |
| $[\ell_k, h_k]$ | the box of parameter $k$ ($\ell_k = h_k$ fixes it) | reals | `bounds` |
| $N_{\mathrm{free}}$ | number of free parameters, those with $\ell_k < h_k$ (3 unless some are fixed) | integer | `fss.py::_free` |
| $x$, $Y$, $dY$ | scaled abscissa $(p - p_c)n^{1/\nu}$, ordinate $y\,n^{-z}$, error $s\,n^{-z}$ | reals | `fss.py::scaled_coordinates` |
| $F$ | an observable's scaling function (never parametrized) | function | |
| $\mathcal B_i$ | the bracketing points of point $i$ (from the other sizes) | set | `fss.py::_Collapse.chi2` |
| $u_l$, $w_l$ | a bracketing point's abscissa relative to $x_i$, $u_l = x_l - x_i$, and its weight $1/dY_l^2$ | reals | (inside `chi2`) |
| $K$, $\bar u$, $\bar y$, $S_{uu}$, $b$ | $\sum_l w_l$; the weighted means of the $u_l$ and of the bracketing ordinates $Y_l$; the weighted spread of the $u_l$; the slope of the weighted line | reals | (inside `chi2`) |
| $K_x$, $K_y$, $K_{xx}$, $K_{xy}$, $\Delta_{\mathrm{HH}}$ | Houdayer and Hartmann's raw weighted sums and their determinant $KK_{xx} - K_x^2$ | reals | (the tests' reference) |
| $\bar Y_i$, $d\bar Y_i^2$ | master-curve estimate at $x_i$ and its variance | reals | (inside `chi2`) |
| $\chi^2_i$ | $(Y_i - \bar Y_i)^2/(dY_i^2 + d\bar Y_i^2)$ | real | `CollapseCost.chi2` |
| $S(\boldsymbol\theta)$, $N$ | the collapse quality, the mean of $\chi^2_i$ over the $N$ bracketed points | real, integer | `CollapseCost.cost`, `.points_used`; `CollapseFit.cost` |
| $e_l$, $\lambda$ | a point's error $Y_l - F(x_l)$ (Proposition 3); a units factor (Proposition 2) | reals | |
| $N_{\min}$ | bracketed points a parameter vector needs to be admissible | integer | `min_points` |
| $\varphi$ | the unbounded variables of the optimizer | reals | (inside `minimize_bounded`) |
| $\mathbf v$, $\bar{\mathbf v}$ | Nelder–Mead vertices and the centroid of all but the worst | vectors | (inside `_nelder_mead`) |
| $\hat{\boldsymbol\theta}$, $\hat{\boldsymbol\theta}_o$, $\hat{\boldsymbol\theta}_s$ | a fit; observable $o$'s own fit; the shared fit | vectors | `CollapseFit.params`, `SharedFit.params` |
| $S_o$, $N_o$; $S_{\mathrm{shared}}$, $S_{\mathrm{ind}}$ | observable $o$'s quality and bracketed points; the pooled shared and independent qualities | reals, integer | `SharedFit.costs`, `.cost`; `SharedComparison.independent_cost` |
| $\Delta$ | shared cost minus pooled independent cost (polynomial: shared $\chi^2$ minus the independent $\chi^2$ sum) | real | `SharedComparison.delta` |
| $p'$, $y'$, $s'$; $\nu'$; $S'_o$ | the data mapped onto the shared parameters (§8.3); the mapped exponent along $p_c = \hat p_c^{\,s}$; the mapped data's quality | reals | (inside `compare_shared_independent`) |
| $\Delta^\ast$, $B'$ | a null replicate of $\Delta$; the usable null replicates | real, integer | `SharedComparison.null` |
| $B$ | bootstrap replicates | integer | `replicates` |
| $d$, $F_{\mathbf a}$, $\mathbf a$ | degree, polynomial master curve $\sum_{k\le d} a_k x^k$, its coefficients | integer, function, vector | `degree`, `CollapseFit.coefficients` |
| $A$, $\boldsymbol\eta$, $\mathbf r$ | whitened Vandermonde matrix $A_{ik} = x_i^k/dY_i$, whitened data $\eta_i = Y_i/dY_i$, residual $\boldsymbol\eta - A\mathbf a$ | matrix, vectors | (inside `_Polynomial.solve`) |
| $\xi$; $U\Sigma V^{\mathsf T}$; $\epsilon$ | the rescaled abscissa $x/\max_i\lvert x_i\rvert$; the singular value decomposition of $A$; $2^{-52}$ | | (inside `_Polynomial.solve`) |
| $W(\boldsymbol\theta)$, $x_{\max}$ | the window map (the points it selects at $\boldsymbol\theta$); the half-width of a fixed window | set, real | `fss.py::window_points`, `window` |
| $\mathcal S$; $\mathcal S_j$, $\boldsymbol\theta_j$ | the polynomial method's fixed point set (its window); its iterates during a fit | sets, vectors | `CollapseFit.selected`; (inside `_window_fit`) |
| $\chi^2(\boldsymbol\theta)$, $\mathrm{dof}$ | the polynomial objective on $\mathcal S$; $\lvert\mathcal S\rvert - (d+1) - N_{\mathrm{free}}$ | real, integer | `CollapseFit.chi2`, `.dof`, `fss.py::polynomial_cost` |
| $\zeta$ | the two methods' standardized difference, difference over its paired bootstrap error | real | `MethodComparison.z` |
| $\rho$ | a calibration ratio: bootstrap error over the true spread of the fits (§11) | real | (tests and §11) |

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

Each observable has its own $F$; the parameters may or may not be shared (§8). Small-world
geometry can add a crossover scale (the number of rewired edges is of order $n\beta$, v1.4 §5.9),
so a single power-law collapse may hold only in a window of sizes and rates: choose the window
before fitting (`ScalingData.subset`) and record it.

**What $y$ must be.** v1.4 §5.9 forbids fitting the mean of the observed depths when censoring is
appreciable. `ScalingData.estimator` records which estimate was used: a nested mean when nothing
is censored; otherwise a survival-based estimate, whose own scaling is then what the fit tests.
The supported one is the restricted mean $\mathbb{E}[\min(\tau, T_{\max})]$: with censoring only
at $T_{\max}$ it *is* the nested mean of $\min(\tau, T_{\max})$
([06-statistics.md](06-statistics.md) §6.5), so it has per-graph values and goes through
`scaling_data`, the bootstrap (§10) and the comparison (§8) like any nested mean. The
Kaplan–Meier median (06 §6.4) has neither a standard error here nor per-graph values whose mean
it is: it can be fitted only with an error the user supplies (`ScalingData` without `graphs` and
`graph_means`), and then `bootstrap_fit`, `compare_shared_independent` and `compare_methods`
refuse it. A `ScalingData` that carries `graph_means` must have each `y` equal to the
equal-weight mean of its point's defined graph means (to a relative $10^{-9}$, else it is
refused), because every bootstrap replicate is recomputed as that mean: a median with per-graph
medians attached would be fitted as one estimator and resampled as another. The censored and
horizon-peak fractions of each point travel with the data (`censored_fraction`,
`horizon_fraction`) into the diagnostics (§12).

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
$\bar Y = \big[K_{xx}K_y - K_xK_{xy} + x_i(KK_{xy} - K_xK_y)\big]/\Delta_{\mathrm{HH}}$ and
$d\bar Y^2 = (K_{xx} - 2x_iK_x + x_i^2K)/\Delta_{\mathrm{HH}}$, with $K_x = \sum_l w_lx_l$,
$K_y = \sum_l w_lY_l$, $K_{xx} = \sum_l w_lx_l^2$, $K_{xy} = \sum_l w_lx_lY_l$ and
$\Delta_{\mathrm{HH}} = KK_{xx} - K_x^2$.

*Proof.* In centered form the fitted line is $\bar y + b(u - \bar u)$; $\bar y$ and $b$ are
uncorrelated with $\operatorname{Var}\bar y = \sum_l w_l^2 dY_l^2/K^2 = 1/K$ and
$\operatorname{Var} b = 1/S_{uu}$, so the value at $u = 0$ has variance $1/K + \bar u^2/S_{uu}$.
With $x = u + x_i$: $\Delta_{\mathrm{HH}} = K S_{uu}$ (both equal $K\sum_l w_l(x_l - \bar x)^2$,
$\bar x = K_x/K$ the weighted mean abscissa), and expanding the raw expressions in
$\bar x = \bar u + x_i$ gives the centered ones. ∎

`test_collapse_cost_matches_the_houdayer_hartmann_formulas` compares the code with a loop
implementation of the raw formulas.

### 4.2 Properties

**Proposition 2 (units; derived here).** $S$ does not change when every $y$ and $s$ is multiplied
by the same constant $\lambda > 0$.

*Proof.* $Y$, $dY$, $\bar Y$ and $d\bar Y$ all scale by $\lambda$ (the weights by $\lambda^{-2}$,
which cancels in $\bar u$, $\bar y$, $b$), so each $\chi^2_i$ is unchanged. ∎

**Proposition 3 (about 1 for a collapse within the errors; derived here).** If at the true
$\boldsymbol\theta$ the points are $Y_l = F(x_l) + e_l$ with independent, mean-zero errors of
variance $dY_l^2$, and $F$ is linear on the smallest interval containing $x_i$ and the abscissas
of $\mathcal B_i$, then $\mathbb{E}\chi^2_i = 1$.

*Proof.* $\bar Y_i$ uses only other sizes' points, so it is independent of $e_i$. On that
interval $F(x) = F(x_i) + c\,(x - x_i)$ for some $c$, so $\mathbb{E}Y_l = F(x_i) + c\,u_l$ for
every $l \in \mathcal B_i$; the weighted least-squares line through points whose means lie on a
line is unbiased for that line, so $\mathbb{E}\bar Y_i = F(x_i)$, with variance $d\bar Y_i^2$
(Proposition 1). So $\mathbb{E}(Y_i - \bar Y_i)^2 = dY_i^2 + d\bar Y_i^2$. ∎

Linearity across each size's pair separately is not enough. The rule of §4.1 can give one size
the pair $[x_i, x'_{j+1}]$ (when $x'_j = x_i$) and another the pair $[x''_{m-2}, x_i]$ (when $x_i$
is the top of its range); the two pairs then meet only at $x_i$, and an $F$ with a kink there,
linear on each pair, biases the line. With equal weights and $u = 0, 1, -1, 0$ the bias at $u = 0$
is $(c_+ - c_-)/4$ for slopes $c_+$ to the right and $c_-$ to the left. Linearity on each pair
does imply linearity on their hull when some pair has $x_i$ in its interior (it then overlaps
every other pair on an interval of positive length), when every pair lies on one side of $x_i$,
or when $F$ is differentiable at $x_i$ (every pair's line then has slope $F'(x_i)$).

Curvature of $F$ across a bracket adds a bias (dense rates make it small; §15 (a)). The
$\chi^2_i$ of different points share bracketing points, and in the study the errors of one size
are correlated across rates (the rates share graphs, [06-statistics.md](06-statistics.md) §5), so
$S$ is not a calibrated chi-square statistic: its *minimum* over $\boldsymbol\theta$ can fall well
below 1 when correlated shifts of a size are absorbed by the parameters (§11). Uncertainties come
from the bootstrap (§10), not from the curvature of $S$.

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
within rounding (§15 (b) shows a jump of 0.025 in $S$ across $p_c = 0.18$ and a fit landing on
0.18 to within $10^{-9}$). This is a property of Houdayer and Hartmann's quality, kept as published;
multi-start (§6) and the bootstrap (§10) treat the resulting roughness. The Monte Carlo of §11
finds no bias in $\hat p_c$ from it.

### 4.3 Admissibility

A parameter vector is admissible only when at least $N_{\min}$ points are bracketed
(`min_points`, default $\max(3, \lceil P/2\rceil)$, a convention chosen here): the mean of a few
$\chi^2_i$ says little, and parameters that push most sizes' scaled ranges apart would otherwise
win on a handful of points. Inadmissible vectors count as $+\infty$. For several observables
(§8) each needs its own $N_{\min}$.

## 5. The polynomial master curve (ruling R-FSS)

Ruling R-FSS (design §11) adds a continuous objective beside the Houdayer–Hartmann quality: a
weighted least-squares fit of a polynomial master curve, jointly with $(p_c, z, \nu)$, on the
points of a window in $x$. It is `method="polynomial"` in `fit_collapse`, `fit_shared`,
`compare_shared_independent`, `leave_one_size_out` and `bootstrap_fit`; `collapse_diagnostics`
follows the method of the fit it is given. The propositions of this section are numbered P1–P3.

### 5.1 The objective

For a fixed point set $\mathcal S$ (§5.3) and parameters $\boldsymbol\theta$, with $x_i$, $Y_i$,
$dY_i$ as in §3,

$$
\chi^2(\boldsymbol\theta) = \min_{\mathbf a \in \mathbb{R}^{d+1}} \sum_{i \in \mathcal S}
\frac{\big(Y_i - F_{\mathbf a}(x_i)\big)^2}{dY_i^2}, \qquad
F_{\mathbf a}(x) = \sum_{k=0}^{d} a_k x^k .
$$

Since $(Y_i - F(x_i))/dY_i = (y_i - n_i^{z}F(x_i))/s_i$, each point is compared with the master
curve in the data's own units and errors. `fss.py::polynomial_cost` evaluates $\chi^2$ at given
parameters; `fit_collapse(..., method="polynomial")` minimizes it over the box with the optimizer
of §6.

### 5.2 The closed form: three parameters searched, the coefficients solved

**Proposition P1 (variable projection; Golub and Pereyra, SIAM J. Numer. Anal. 10, 413 (1973)).**
With $A_{ik} = x_i^k/dY_i$ and $\eta_i = Y_i/dY_i$ ($i \in \mathcal S$, $k = 0, \dots, d$),

$$
\chi^2(\boldsymbol\theta) = \big\lvert \boldsymbol\eta - A A^{+} \boldsymbol\eta\big\rvert^2
= \boldsymbol\eta^{\mathsf T}(I - P_A)\,\boldsymbol\eta ,
$$

with $A^{+}$ the pseudoinverse and $P_A$ the orthogonal projection onto the column space of $A$;
and $\min_{\boldsymbol\theta, \mathbf a}$ of the weighted sum is $\min_{\boldsymbol\theta}\chi^2$.
So only $(p_c, z, \nu)$ are searched.

*Proof.* For fixed $\boldsymbol\theta$ the sum is $\lvert \boldsymbol\eta - A\mathbf a\rvert^2$, a
linear least squares problem: its minimum over $\mathbf a$ is attained at
$\mathbf a = A^{+}\boldsymbol\eta$ and is the squared distance from $\boldsymbol\eta$ to the
column space, $\lvert(I - P_A)\boldsymbol\eta\rvert^2$. A joint minimum over
$(\boldsymbol\theta, \mathbf a)$ is the minimum over $\boldsymbol\theta$ of the minimum over
$\mathbf a$. ∎

The computation (`fss.py::_Polynomial.solve`) works in the basis $\xi = x/\max_i\lvert x_i\rvert$,
which spans the same column space (so $\chi^2$ is unchanged) and is better conditioned. The
singular value decomposition $A = U\Sigma V^{\mathsf T}$ gives
$\mathbf a = V\Sigma^{-1}U^{\mathsf T}\boldsymbol\eta$, dropping singular values below
$\sigma_{\max}\max(\lvert\mathcal S\rvert, d+1)\,\epsilon$ ($\sigma_{\max}$ the largest), then
$\mathbf r = \boldsymbol\eta - A\mathbf a$ and $\chi^2 = \lvert \mathbf r\rvert^2$. The reported `coefficients` are
$a_k/(\max\lvert x\rvert)^k$, in powers of $x$. $V\Sigma^{-2}V^{\mathsf T}$ is their covariance for
unit-variance whitened errors; it gives the variance of the fitted curve at a new abscissa, used for
held-out points (§9).

### 5.3 The window rule, exactly

The window $W(\boldsymbol\theta)$ (`fss.py::window_points`; only $p_c$ and $\nu$ enter) is

* `window="overlap"` (the default): the points with
  $x_i \in [\max_n \min_{j \in n} x_j,\ \min_n \max_{j \in n} x_j]$, every size's points in the
  range that every size covers (empty when the ranges do not meet);
* `window=x_max` ($> 0$): the points with $\lvert x_i\rvert \le x_{\max}$;

both ends included. A fit (`fss.py::_window_fit`):

1. draws the first window at $\boldsymbol\theta_0$: the first row of `initial`, else the
   Houdayer–Hartmann fit with the same settings, else the box center; $\mathcal S_0 = W(\boldsymbol\theta_0)$;
2. minimizes $\chi^2$ on $\mathcal S_0$ from `initial` (or $\boldsymbol\theta_0$) and `starts`
   random starts, giving $\boldsymbol\theta_1$;
3. draws $\mathcal S_j = W(\boldsymbol\theta_j)$; stops if $\mathcal S_j = \mathcal S_{j-1}$
   (`window_stable`), else minimizes $\chi^2$ on $\mathcal S_j$ from $\boldsymbol\theta_j$ (no
   random starts) and repeats; at most `window_iterations` (5) fits, the last one kept.

The set never changes during a search, so the optimizer sees a continuous objective (P2). Two
alternatives were rejected: a hard window redrawn at every $\boldsymbol\theta$ makes $\chi^2$ jump
when a point enters or leaves; a smooth taper keeps it continuous but lets the number of effective
points change with $\boldsymbol\theta$, so the search could lower $\chi^2$ by pushing points out
of the window. A fixed set has a fixed number of degrees of freedom.

A set is admissible when it holds points of at least two sizes, some size has $d + 1$ distinct
rates in it, and $\mathrm{dof} = \lvert\mathcal S\rvert - (d + 1) - N_{\mathrm{free}} \ge 1$;
otherwise the status is `too_few_points` and the message names the condition.

The window map can have more than one fixed point. In development the hierarchical example of
§15 settled on 48 points from a start near the reference and on 49 points with more random
starts; each fit was the optimum on its own window and $\nu$ differed by 0.027, a quarter of its
bootstrap error. The window is part of the estimator, so its variability belongs in the error:
by ruling R-FSS (design §11) every bootstrap replicate (§10) and every null replicate of the
comparison (§8.3) runs this whole procedure on its own data, first window at its own
Houdayer–Hartmann fit included (`reselect_window=True`, the default; `False` keeps the data
fit's window, cheaper but without that variability). Leave-one-size-out and the drift refit keep
the fit's window minus the left-out size by design (§5.7), and report `window_stable = None`:
their set is never redrawn, so whether it would settle is not checked.

### 5.4 Continuity

**Proposition P2 (derived here).** On an admissible fixed set, $\chi^2(\boldsymbol\theta)$ is
infinitely differentiable wherever $n^{1/\nu}$ and $n^{-z}$ are finite.

*Proof.* $x_i$, $Y_i$ and $dY_i > 0$ are smooth in $\boldsymbol\theta$. A size with $d + 1$
distinct rates in $\mathcal S$ has $d + 1$ distinct abscissas for every $\boldsymbol\theta$ ($x$
increases strictly with $p$ at fixed $n$), so $A$ has $d + 1$ linearly independent rows
(Vandermonde rows at distinct points, scaled by positive factors) and full column rank. Then
$P_A = A(A^{\mathsf T}A)^{-1}A^{\mathsf T}$ is smooth, and so is
$\boldsymbol\eta^{\mathsf T}(I - P_A)\boldsymbol\eta$. ∎

Contrast with Proposition 4: across the measured rate 0.18, §15 (f) shows $\chi^2$ changing by
about $4\times10^{-4}$ per $10^{-7}$ of $p_c$, steadily, where the Houdayer–Hartmann quality jumps
by 0.023.

### 5.5 $\chi^2/\mathrm{dof}$, and its caveat

`CollapseFit.cost` is $\chi^2(\hat{\boldsymbol\theta})/\mathrm{dof}$ (also `chi2_per_dof`). With
independent Gaussian errors of the stated sizes and a polynomial $F$ adequate on the window,
$\chi^2(\hat{\boldsymbol\theta})$ follows a chi-square law with dof degrees of freedom to first
order in the three nonlinear parameters (the linear least-squares result), so
$\chi^2/\mathrm{dof} \approx 1$; over 20 synthetic datasets with independent 2% errors the mean
was $0.94 \pm 0.06$ (`test_chi2_per_dof_is_about_one_for_independent_errors`). In the study the
rates of a size share graphs, gates and mask uniforms, so a size's errors are correlated across
$p$ and $\chi^2$ does not follow a chi-square law: it often falls below dof when the parameters
absorb a coherent shift of a size (0.57 on the hierarchical example of §15). Read
$\chi^2/\mathrm{dof} \gg 1$ as a failure of the polynomial or of the scaling form on the window,
and take the uncertainties from the bootstrap (§10).

### 5.6 The degree's effect

The degree must let $F$ follow the master curve across the window. Too low a degree biases the
parameters, and $\chi^2/\mathrm{dof}$ shows it: on a steep master curve over $\lvert x\rvert \le 2$
(§15 (g)), degrees 2, 3, 5 and 9 give $\nu = 3.00$ (the box bound), 2.51, 1.84 and 1.5007 with
$\chi^2/\mathrm{dof} = 322$, 67.6, 16.1 and 0.044 (true $\nu = 1.5$). Too high a degree costs
variance and degrees of freedom. `fss.py::polynomial_degree_scan` fits a list of degrees from the
same first window and reports their parameters, $\chi^2/\mathrm{dof}$ and spread; report it with
any polynomial fit. The default, degree 3 (degrees 1 to 10 are accepted), suits the default
overlap window, which is narrow (the smallest size's range when the sizes' ranges are nested).

### 5.7 Refits, the shared fit and the comparison

Bootstrap replicates (§10) and null replicates (§8.3) run the whole windowed fit on their own
data by default (§5.3). Leave-one-size-out and the drift refit (§9, §12) keep $\mathcal S$
without the left-out size (from the full fit's parameters and the random starts): they ask how the
parameters move when a size is removed from the same collapse, so the point set is held fixed;
the held-out points inside the $x$ range the refit's polynomial was fitted on are compared with
it, with its prediction variance added to their own. Their `window_stable` is None. The shared fit
(§8) uses each observable's window from its own fit.

**Proposition P3 (derived here).** On these fixed sets, with every minimum exact,
$\Delta = \chi^2_{\mathrm{shared}}(\hat{\boldsymbol\theta}_s) - \sum_o \chi^2_o(\hat{\boldsymbol\theta}_o) \ge 0$.
The optimizer's minima are approximate, so a computed $\Delta$ can be slightly negative.

*Proof.* The minimum of $\sum_o\chi^2_o(\boldsymbol\theta_o)$ over independent
$\boldsymbol\theta_o$ includes the choice $\boldsymbol\theta_o = \boldsymbol\theta$ for every
$o$, so it is at most the shared minimum. ∎

With independent Gaussian errors of known size, $\Delta$ is minus twice the log likelihood ratio
of the shared against the independent model (the master-curve coefficients are free in both), so
by Wilks's theorem (S. S. Wilks, Ann. Math. Statist. 9, 60 (1938)) it would be asymptotically
chi-square with $N_{\mathrm{free}}(O - 1)$ degrees of freedom, the number of parameters the
shared hypothesis removes (a parameter fixed through $\ell_k = h_k$ is fixed in both models and
removes none). With the study's correlated errors this law does not hold, and the bootstrap null
of §8.3 is used. Each null replicate fits every observable alone with its own window, then the
shared fit on those windows, exactly as for the data, so P3 holds within each replicate as well.

`fss.py::compare_methods` fits the data with both methods (the polynomial window drawn at the
Houdayer–Hartmann fit) and, with `replicates > 0`, fits every graph-bootstrap replicate with both,
by the same procedure as `bootstrap_fit` (§10: the replicate's polynomial window is drawn at the
replicate's own Houdayer–Hartmann fit). The standard deviation of the paired replicate differences
is the error of the difference, and the standardized difference
$\zeta = \texttt{difference}/\texttt{se\_difference}$ (`MethodComparison.z`) measures the
disagreement in units of what resampling the graphs explains (§12). On the hierarchical example
the two methods differed by $(0.0008, 0.0005, 0.0076)$ in $(p_c, z, \nu)$ with
$\zeta = (0.23, 0.06, 0.16)$ (§15 (g)). On
synthetic data this paired error was conservative, about 1.4 to 1.6 times the true spread of the
difference (§11), so $\lvert\zeta\rvert$ errs on the small side.

### 5.8 Which method, and why the default stays Houdayer and Hartmann's

A benchmark in development used 40 synthetic datasets per scenario, made with §11's generator (20
graphs, a 5% graph effect shared by the rates of a size, 3% rate noise, sizes 16–128, 21 rates).
Its master curves were the logistic $F_1$ and a steeper one, $F_3(x) = 2/(1 + e^{3x}) + 0.5$. The
polynomial fits started at the Houdayer–Hartmann fit. Root-mean-square errors of
$(p_c, z, \nu)$ (true $(0.2, 1, 1.5)$):

| method | $F_1$ | $F_3$ (steep) |
|---|---|---|
| Houdayer–Hartmann | 0.0161, 0.0427, 0.110 | 0.0009, 0.0114, 0.0219 |
| polynomial, overlap window, degree 3 (the defaults) | 0.0178, 0.0440, 0.143 | 0.0029, 0.0219, 0.0436 |
| polynomial, $\lvert x\rvert \le 1$, degree 3 | 0.0151, 0.0394, 0.109 | 0.0018, 0.0139, 0.0340 |
| polynomial, $\lvert x\rvert \le 1$, degree 5 | 0.0158, 0.0407, 0.117 | 0.0017, 0.0126, 0.0230 |
| polynomial, $\lvert x\rvert \le 2$, degree 5 | 0.0159, 0.0406, 0.108 | 0.0021, 0.0252, 0.294 |
| polynomial, $\lvert x\rvert \le 2$, degree 7 | 0.0160, 0.0420, 0.105 | 0.0006, 0.0085, 0.0409 |
| polynomial, $\lvert x\rvert \le 2.5$, degree 7 | 0.0157, 0.0409, 0.104 | 0.0010, 0.0095, 0.126 |

Houdayer and Hartmann's quality uses every bracketed point (about 76 of 84 here) and needs no
model of $F$. The polynomial with the parameter-free overlap window uses about 46–49 points, the
smallest size's range, and was less efficient here (the error of $\nu$ 30% larger for $F_1$ and
twice as large for $F_3$; for $F_1$ the difference came from a few datasets, and over 200 others
the two methods' standard deviations were close, §11: the polynomial's window choice gives it a
heavier tail). Wider windows with higher degrees come close to Houdayer and Hartmann,
or beat them on some parameters ($F_3$, $\lvert x\rvert \le 2$, degree 7: smaller errors in $p_c$
and $z$, a larger one in $\nu$). They need $x_{\max}$ and $d$ chosen for the data, and they bias the
parameters when the degree is too low for the window: $\nu$ off by 0.29 at $\lvert x\rvert \le 2$,
degree 5, with $\chi^2/\mathrm{dof} = 10.7$ there. So no polynomial setting was better on both
benchmarks, and the default stays Houdayer and Hartmann's.

When to prefer which:

* **Houdayer–Hartmann** (the default): no model of $F$; every bracketed point; robust without
  tuning. Its objective jumps (Proposition 4), so fits can sit on the edge of a jump, and $S$ is no
  calibrated statistic.
* **Polynomial**: a smooth objective (P2), hence precise optimization and smooth parameter scans;
  an explicit master curve (its coefficients, for plots and for comparing observables);
  $\chi^2/\mathrm{dof}$ readable with the caveat of §5.5; nearly exact on clean data (§15 (f):
  $p_c$ to $10^{-7}$, against $10^{-4}$ for Houdayer–Hartmann). Use it as the agreement check of
  `compare_methods` (§12), and as a primary method once a window and degree have been validated on
  the data: a degree scan whose parameters settle, $\chi^2/\mathrm{dof}$ consistent with the
  bootstrap, and agreement with Houdayer and Hartmann.

## 6. The bounded Nelder–Mead

### 6.1 The box and the reparametrization

Each free parameter ($\ell_k < h_k$) is written

$$
\theta_k = \ell_k + (h_k - \ell_k)\,\frac{1 + \sin\varphi_k}{2}, \qquad \varphi_k \in \mathbb{R},
$$

and a parameter with $\ell_k = h_k$ is held fixed (`fss.py::minimize_bounded`). The
transformation is a choice made here; Proposition 5 proves the properties the optimizer relies
on.

**Proposition 5 (derived here).** The map is smooth and maps $\mathbb R$ onto $[\ell_k, h_k]$,
reaching both bounds at finite $\varphi_k$ ($\pm\pi/2$); hence
$\inf_{\boldsymbol\varphi} f(\boldsymbol\theta(\boldsymbol\varphi)) = \inf_{\text{box}} f$, and a
minimizer on a bound corresponds to a stationary point in $\varphi$.

*Proof.* $\sin$ maps $\mathbb R$ onto $[-1, 1]$ and is smooth; the affine map takes $[-1, 1]$ onto
$[\ell_k, h_k]$. Surjectivity gives the equality of infima. At $\varphi_k = \pm\pi/2$ the
derivative $\cos\varphi_k$ vanishes. ∎

The optimizer can thus reach a bound exactly (tests: a minimum outside the box lands on its
corner), and Nelder–Mead runs unconstrained in $\varphi$. The map is periodic, which is harmless.

### 6.2 Nelder–Mead

`fss.py::_nelder_mead` is the method of Lagarias, Reeds, Wright and Wright (SIAM J. Optim. 9,
112 (1998)) with reflection 1, expansion 2, contraction 1/2 and shrink 1/2:

1. Sort the $N_{\mathrm{free}}+1$ vertices of the simplex in $\varphi$ by value (a stable sort:
   ties keep their order).
2. Stop when every vertex is within `xatol` ($10^{-10}$ in $\varphi$) of the best in every
   coordinate and every value within $\texttt{fatol}\cdot\max(1, \lvert f_{\mathrm{best}}\rvert)$
   ($10^{-12}$), or when the budget (`max_evaluations`, 2000 per start) is spent.
3. Reflect the worst vertex $\mathbf v_w$ through the centroid $\bar{\mathbf v}$ of the others:
   $\mathbf v_r = \bar{\mathbf v} + (\bar{\mathbf v} - \mathbf v_w)$, with value $f_r$. If $f_r$
   beats the best, try the expansion $\bar{\mathbf v} + 2(\mathbf v_r - \bar{\mathbf v})$ and keep
   the better of the two; if $f_r$ beats the second worst, keep $\mathbf v_r$.
4. Otherwise contract outside ($\bar{\mathbf v} + (\mathbf v_r - \bar{\mathbf v})/2$, kept if not
   worse than $f_r$) when $f_r$ beats the worst, inside
   ($\bar{\mathbf v} + (\mathbf v_w - \bar{\mathbf v})/2$, kept if it beats the worst) when it
   does not; if the contraction fails, shrink every vertex halfway to the best.

Non-finite values count as $+\infty$. Starts: the rows of `initial`, or the center of the box, then
`starts` points (8 by default) uniform in the box, from the stable stream of `seed`
([06-statistics.md](06-statistics.md) §7.1). Each start's simplex has steps of 0.25 in $\varphi$
toward the interior. The best start wins (the first on ties) and is polished: a simplex with steps
0.05 is rebuilt at its best vertex, up to `restarts` (2) times while that improves the value.
Everything is deterministic given the seed. A coordinate within $10^{-6}$ of its box width from a
bound is reported `at_bound`.

## 7. Fits and their statuses

`fss.py::fit_collapse(data, bounds=...)` minimizes $S$ over the box (or, with
`method="polynomial"`, $\chi^2$ on its window, §5) and returns a `CollapseFit`: the parameters,
the objective at them (`cost`: $S$, or $\chi^2/\mathrm{dof}$), the points used and the settings
(bounds, $N_{\min}$ resolved and as given, random starts, seed, budget; for the polynomial method
the degree, the window rule, the point set and whether it settled), with the data's `observable`,
`beta`, `label` and `estimator`. It never raises for lack of data; its `status` says what happened:

| status | parameters | meaning |
|---|---|---|
| `ok` | yes | converged inside the box |
| `at_bound` | yes | some parameter on a box bound (named in `at_bound`): the box is too small, or the data do not constrain that parameter; widen and refit |
| `not_converged` | yes | the evaluation budget ran out before the tolerances were met |
| `too_few_points` | no | fewer points than $N_{\min}$, or no admissible vector in the box (the sizes' scaled ranges never overlap enough); for the polynomial method, a window that is not admissible (§5.3) or an objective undefined everywhere the search went |
| `too_few_sizes` | no | fewer than two sizes: there is nothing to collapse |

Two sizes give a fit, but three parameters are then weakly constrained (a collapse of two curves
leaves a family of near-equivalent parameter sets); use at least three sizes, and more to see a
drift (§12). `CollapseFit.usable` is true for the first three statuses.

## 8. Shared versus independent exponents

### 8.1 The shared fit

`fss.py::fit_shared(datasets, bounds=...)` fits one $\boldsymbol\theta$ to several observables at
one $\beta$ (refusing data from different $\beta$), each with its own $F$: the quality is evaluated
per observable and pooled,

$$
S_{\mathrm{shared}}(\boldsymbol\theta) = \frac{\sum_o\sum_{i \in o}\chi^2_{oi}(\boldsymbol\theta)}{\sum_o N_o(\boldsymbol\theta)},
$$

with each observable needing its own $N_{\min}$. This is v1.4's shared-criticality hypothesis.

### 8.2 The statistic

With independent fits $\hat{\boldsymbol\theta}_o$ and their pooled quality
$S_{\mathrm{ind}} = \sum_o N_o S_o(\hat{\boldsymbol\theta}_o)/\sum_o N_o$,

$$
\Delta = S_{\mathrm{shared}}(\hat{\boldsymbol\theta}_s) - S_{\mathrm{ind}}
$$

measures how much the collapse worsens when the parameters are shared (`SharedComparison.delta`;
for the polynomial method $\Delta$ is the $\chi^2$ difference of §5.7).
$\Delta$ is near 0 when the observables' best parameters agree within their noise and large when
they do not; but $S$ is not a calibrated chi-square (§4.2), so $\Delta$ needs a null distribution.

### 8.3 The bootstrap null

`fss.py::compare_shared_independent` maps the data onto the shared hypothesis and resamples them.
Each observable $o$ is mapped point by point to

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

So the mapped data keep each observable's collapse quality and noise, and at
$\hat{\boldsymbol\theta}_s$ (and, for the polynomial method, on the window it chose) each has
exactly the fit it had at its own parameters. Proposition 6 does not say that
$\hat{\boldsymbol\theta}_s$ *minimizes* $S'_o$. Off the slice $p_c = \hat p_c^{\,s}$ the mapped
abscissa at $(p_c, z, \nu)$ is
$x' = (\hat p_c^{\,s} - p_c)\,n^{1/\nu} + (p - \hat p_c^{\,o})\,n^{1/\nu'}$: its shift and its slope
in $p$ grow with different powers of $n$ unless $\hat\nu_o = \hat\nu_s$, so it is the abscissa of
the original data at no parameter vector, and $S'_o$ is no reparametrization of $S_o$ there
(derived here). For a smooth objective (the polynomial $\chi^2$ on a fixed set) the consequence is
explicit: with $g_i$ the derivative of the objective with respect to $x_i$ at the common
coordinates, stationarity of $S_o$ at $\hat{\boldsymbol\theta}_o$ gives
$\sum_i g_i\,n_i^{1/\hat\nu_o} = 0$, while
$\partial S'_o/\partial p_c = -\sum_i g_i\,n_i^{1/\hat\nu_s}$ at $\hat{\boldsymbol\theta}_s$, which is
not 0 in general; the $z$ and $\nu$ derivatives vanish by the slice equivalence. So the mapped
data's own optimum lies *near* $\hat{\boldsymbol\theta}_s$, exactly on it only without noise (data
that follow the scaling form exactly at $\hat{\boldsymbol\theta}_o$ map to data that follow it
exactly at $\hat{\boldsymbol\theta}_s$) or when $\hat\nu_o = \hat\nu_s$. In development the shift
was about $10^{-4}$ in $p_c$ and $10^{-3}$ in $\nu$, against bootstrap errors near 0.013 and 0.09
(`test_the_null_mapping_moves_each_observable_onto_the_shared_parameters` checks the slice
equivalence to $10^{-9}$ and the refit to $2\times10^{-3}$).

Each of $B$ replicates then resamples the graphs once per size
(`graph_bootstrap(labels_n, B, seed=[seed, n])`, applied to every rate and every observable at
that size, so the pairing across rates and observables survives,
[06-statistics.md](06-statistics.md) §7.2), recomputes every point's mean and error from the drawn
graphs, and computes $\Delta^\ast$ *by the procedure that gave* $\Delta$: the shared and the
independent fits from `initial` (or the box center) and `replicate_starts` random starts (by
default as many as `starts`, the same seed), and for the polynomial method each observable's own
window drawn for the replicate (`reselect_window=True`, the default; `False` keeps the windows the
observables' fits chose). With $B'$ usable replicates

$$
\text{p-value} = \frac{1 + \#\lbrace\Delta^\ast \ge \Delta\rbrace}{1 + B'},
$$

and `prefers` is `"independent"` when the p-value is at most `alpha` (0.05), `"shared"` otherwise,
`"undetermined"` when a fit gave no parameters or every replicate failed. If $\Delta$ and the
$\Delta^\ast$ were exchangeable under the hypothesis, the p-value would be uniform on
$\lbrace 1/(B'+1), \dots, 1\rbrace$, so $\Pr(\text{p-value} \le \alpha) \le \alpha$. The
construction follows the first of Hall and Wilson's guidelines for bootstrap tests (P. Hall and
S. R. Wilson, Biometrics 47, 757 (1991)): resample in a way that reflects the null hypothesis.
Their second, a pivotal statistic, is not met: $\Delta$ is not studentized. Computing $\Delta^\ast$
by the same procedure as $\Delta$ matters: a local refit from $\hat{\boldsymbol\theta}_s$ searches
less than the multi-start fits of the data do, which pushes the null down. The null is an
approximation in four ways: the noise is the resampled noise of the observed data, its center is
built from fitted parameters, the mapped data's optimum is near $\hat{\boldsymbol\theta}_s$ rather
than on it (above), and $\Delta$ is not pivotal. §11 measures its size on synthetic data;
`test_null_replicates_are_the_comparison_on_mapped_resampled_data` checks each $\Delta^\ast$
against this construction written out in the test, and
`test_the_comparison_keeps_its_size_when_the_parameters_are_shared` guards the size.

## 9. Leaving one size out

`fss.py::leave_one_size_out` refits without each size in turn (from the full fit and `starts`
random starts) and reports the refits' parameters, the `spread` (max minus min) of each parameter
over them, and the *held-out cost*: the mean $\chi^2_i$ of the left-out size's points against the
master curve of the remaining sizes at the refit's parameters (each such point is bracketed only
by the other sizes, which are exactly the training sizes). On data that obey the scaling form
within their errors the refits agree and the held-out costs are of order 1 (on exact data with
1% errors they are below 0.05, §15 (e)); a held-out cost far above 1, or parameters that move
systematically with the left-out size, flag corrections to scaling, a crossover or a size outside
the scaling window (v1.4 §5.8: held-out-size collapse diagnostics). With fewer than three sizes
the refits have status `too_few_sizes`.

## 10. The bootstrap of a fit

`fss.py::bootstrap_fit` resamples graphs once per $(n, \beta)$: for each size, one draw matrix
(`graph_bootstrap(labels_n, B, seed=[seed, n])`) applied to every rate's graph means, so each
replicate's means and errors at that size come from the same drawn graphs and the coupling of the
rates (shared graphs, gates and mask uniforms, v1.4 §7.3) is kept. Every point of a size must have
the same graph labels (the rates were measured on the same graphs); the data must carry the
per-graph values (`fss.py::scaling_data`), and each `y` must be their mean (§3). A point whose
replicate error is 0 or undefined is left out of that replicate.

**Each replicate is fitted by the procedure that fitted the data.** The bootstrap estimates the
sampling spread of the *estimator*, so every replicate goes through `fit_collapse` with the same
method, bounds, `initial`, seed and settings, and with `replicate_starts` random starts (by
default as many as `starts`; fewer is cheaper and only approximates the procedure). For the
polynomial method each replicate also draws its own window (§5.3: first at its own
Houdayer–Hartmann fit, redrawn until it settles; ruling R-FSS, design §11), unless
`reselect_window=False`, which refits on the data fit's window. A replicate's random starts are
the data fit's (same seed), so the replicate is the same function of its data as the estimate is
of the data. `test_every_bootstrap_replicate_is_fitted_like_the_data` and
`test_polynomial_replicates_draw_their_own_windows` check this replicate by replicate.

Why it matters: a local search started at the data's fit stays near it on the rough
Houdayer–Hartmann quality (Proposition 4), so replicates refitted that way vary less than the
estimator does. With such refits (the first implementation) the errors on §11's ensemble were
0.86, 0.85 and 0.87 of the true spread of $(p_c, z, \nu)$ and the $\pm 1.96$ error intervals
covered the truth in 33, 35 and 30 of 40 datasets; refitting by the procedure gave 1.04, 0.98 and
1.11, and 36, 36 and 34 of 40 (§11). The price is time: a replicate costs a whole fit.

The result has the replicate parameters, their standard deviations (`se`), percentile intervals
(`lower`, `upper`), counts of failed and at-bound replicates, `replicate_starts`,
`reselect_window`, and for the polynomial method each replicate's point set (`selected`, a mask
over the data's points) and whether its window settled (`window_stable`).

## 11. Calibration on synthetic data

The synthetic ensemble below has 20 graphs per point, a graph effect of 5% shared by every rate of
a size (as the coupled rates of the study share their graphs) and 3% rate noise, four sizes
16–128 and 21 rates in $[0.1, 0.3]$, with $(p_c, z, \nu) = (0.2, 1, 1.5)$. The true spread of each
estimator is the standard deviation of its fits to 200 independent datasets (4 random starts);
its relative uncertainty is about $1/\sqrt{2 \cdot 199} = 5\%$. The bootstrap was run on 40 other
datasets ($B = 40$ replicates, 4 random starts, every replicate fitted by the data's procedure,
§10). Per parameter: the mean of the 40 bootstrap errors over the true spread (the ratio $\rho$,
with its standard error over the 40 datasets), and how many of the 40 intervals contain the truth:

| | $p_c$ | $z$ | $\nu$ |
|---|---|---|---|
| Houdayer–Hartmann: mean of 200 fits | 0.1998 | 0.9984 | 1.5107 |
| standard deviation of 200 fits | 0.0132 | 0.0375 | 0.0859 |
| mean bootstrap error; $\rho$ | 0.0137; $1.04 \pm 0.03$ | 0.0368; $0.98 \pm 0.03$ | 0.0950; $1.11 \pm 0.07$ |
| estimate $\pm 1.96$ errors covers | 36/40 | 36/40 | 34/40 |
| 95% percentile interval covers | 34/40 | 35/40 | 32/40 |
| polynomial (defaults): mean of 200 fits | 0.1995 | 0.9994 | 1.5113 |
| standard deviation of 200 fits | 0.0127 | 0.0376 | 0.0837 |
| mean bootstrap error; $\rho$ | 0.0149; $1.17 \pm 0.06$ | 0.0381; $1.01 \pm 0.02$ | 0.1196; $1.43 \pm 0.15$ |
| estimate $\pm 1.96$ errors covers | 38/40 | 37/40 | 37/40 |
| 95% percentile interval covers | 33/40 | 36/40 | 32/40 |

The fits are unbiased within these errors. The percentile intervals' baseline is not 38/40: if
the truth's deviation from the estimate were exchangeable with the $B$ replicates' deviations from
it, the truth's rank among the $B + 1$ values would be uniform, and the type-7 2.5% and 97.5%
quantiles of 40 replicates sit at order positions 1.975 and 39.025, so the interval would cover
with probability about $(39.025 - 1.975)/41 = 0.90$ (derived here), 36 of 40; the counts above
are 0 to 2.2 binomial standard deviations (1.9) below that. With
$B = 200$ the baseline is 0.94. The Houdayer–Hartmann errors are calibrated on average; the
polynomial ones are conservative, most for $\nu$, and heavy-tailed (the median ratios are 1.08,
1.00 and 1.15): the replicates' windows move (§5.3), and a few replicates whose window jumps
widen the spread. That is the window's own variability, which ruling R-FSS puts in the error.

The first implementation refitted each replicate locally from the data's fit (no random starts)
and kept the polynomial window fixed. On the same 40 datasets its Houdayer–Hartmann errors were
low, $\rho = 0.86$, 0.85 and 0.87, covering in 33, 35 and 30 of 40 (§10); its polynomial errors
had $\rho = 1.10$, 1.03 and 1.08. A single dataset (seed 100) had given a polynomial $\nu$ error of
0.68 of the spread of 40 fits, which prompted the ruling; over 40 datasets the fixed window was not
low on average, but it leaves the window's variability out of each replicate by construction.

A smaller design runs in the tests (`test_the_bootstrap_errors_match_the_true_spread_of_the_fits`:
sizes 16–64, 9 rates, no random starts, 80 fits and 12 bootstraps of 19 replicates). Over 24
independent repeats of that test, $\log\rho$ had means $(-0.15, -0.13, -0.11)$ for
Houdayer–Hartmann, $(-0.07, -0.07, 0.07)$ for the polynomial and $(0.37, 0.47, 0.40)$ for
`compare_methods`' paired error of the difference, with standard deviations 0.08–0.26. So on
this coarser design the Houdayer–Hartmann errors run 10–14% low, and the paired error of the
two methods' difference is conservative by a factor of about 1.5.

**The comparison** (§8.3), with Houdayer–Hartmann's quality: 40 fresh pairs of observables with
the *same* parameters (and different $F$; $B = 19$, 3 random starts) gave $\Delta$ from 0.015 to
1.10 (5%, 50%, 95% quantiles 0.02, 0.18, 0.72), while their pooled bootstrap nulls had quantiles
0.04, 0.25 and 0.95. One pair of 40 was rejected at 0.05 and two at 0.10, and the mean p-value
was 0.62 (0.525 for an exact null with $B = 19$): the test kept its size and was somewhat
conservative. With the polynomial method, 20 pairs gave $\Delta$ quantiles 3.3, 9.2 and 30 against
bootstrap-null quantiles 1.6, 14.5 and 59, no rejection, and a mean p-value of 0.61. Under the
alternative ($\nu = 1.2$ against $2.5$, 2% graph effects) $\Delta$ was 5.1 against a null whose
median of 39 replicates was 0.08 and largest 0.52. The code (on one core, about two hours; the
loops over datasets and pairs are independent and can run in separate processes):

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
truth = np.array([0.2, 1.0, 1.5])
for method in ("houdayer_hartmann", "polynomial"):
    fits = np.array([fss.fit_collapse(synthetic(truth, F1, seed=2000 + s), bounds=box,
                                      method=method, starts=4).params for s in range(200)])
    spread = fits.std(axis=0, ddof=1)
    boots = [fss.bootstrap_fit(synthetic(truth, F1, seed=100 + s), bounds=box, method=method,
                               replicates=40, seed=1, starts=4) for s in range(40)]
    se = np.array([b.se for b in boots])
    estimate = np.array([b.fit.params for b in boots])
    lower, upper = np.array([b.lower for b in boots]), np.array([b.upper for b in boots])
    rho = se / spread
    print(method, "mean", fits.mean(axis=0), "sd", spread, "mean se", se.mean(axis=0))
    print("  rho", rho.mean(axis=0), "+-", rho.std(axis=0, ddof=1) / np.sqrt(40),
          "median", np.median(rho, axis=0))
    print("  +-1.96 se covers", (np.abs(estimate - truth) <= 1.96 * se).sum(axis=0),
          "percentile covers", ((lower <= truth) & (truth <= upper)).sum(axis=0))

for method, pairs in (("houdayer_hartmann", 40), ("polynomial", 20)):
    deltas, nulls, pvalues = [], [], []
    for s in range(pairs):
        a = synthetic(truth, F1, seed=300 + s, name="a")
        b = synthetic(truth, F2, seed=400 + s, name="b")
        cmp = fss.compare_shared_independent([a, b], bounds=box, method=method, replicates=19,
                                             seed=5, starts=3)
        deltas.append(cmp.delta)
        nulls.extend(cmp.null[~np.isnan(cmp.null)])
        pvalues.append(cmp.p_value)
    pvalues = np.array(pvalues)
    print(method, "true null of delta", np.quantile(deltas, [0.05, 0.5, 0.95]),
          "bootstrap null", np.quantile(nulls, [0.05, 0.5, 0.95]))
    print("  rejected at 0.05:", (pvalues <= 0.05).sum(), "at 0.10:", (pvalues <= 0.10).sum(),
          "mean p", pvalues.mean())

rates = np.linspace(0.12, 0.28, 17)
a = synthetic((0.2, 1.0, 1.2), F1, rates=rates, sg=0.02, sp=0.02, seed=11, name="a")
b = synthetic((0.2, 1.0, 2.5), F2, rates=rates, sg=0.02, sp=0.02, seed=12, name="b")
alt = fss.compare_shared_independent([a, b], bounds=box, replicates=39, seed=5, starts=4)
print("alternative: delta", alt.delta, "null median", np.nanmedian(alt.null),
      "max", np.nanmax(alt.null))
```

(`synthetic` draws the same numbers as the tests' `_hierarchical`, so the alternative is the pair
of `test_the_comparison_prefers_independent_exponents_for_different_nu`.) These are measurements
on one synthetic ensemble, not guarantees: the size of $\Delta$ and its null depend on the noise,
the window and the scaling functions, and a single rejection at the 5% level is weak evidence.
Report the p-value with the fits' parameter differences and their bootstrap errors.

**The polynomial method's efficiency.** On the 40 datasets of the benchmark of §5.8 (seeds
100–139) the polynomial defaults had standard deviations $(0.0176, 0.0438, 0.137)$, against
$(0.0163, 0.0432, 0.110)$ for Houdayer and Hartmann; over the 200 datasets above they were
$(0.0127, 0.0376, 0.0837)$ against $(0.0132, 0.0375, 0.0859)$. The difference on the 40 came from
a few datasets: the polynomial's two largest $\nu$ errors there were 0.44 and 0.56, while its
median absolute $\nu$ error (0.052) was below Houdayer and Hartmann's (0.057). The polynomial's
window choice gives it a heavier tail, not a larger typical error. The benchmark of §5.8 is this
code with the polynomial settings added (about three minutes):

<!-- not run -->

```python
def F3(x):
    return 2.0 / (1.0 + np.exp(3.0 * x)) + 0.5


configs = [("overlap", 3), (1.0, 3), (1.0, 5), (2.0, 5), (2.0, 7), (2.5, 7)]
for label, F in (("F1", F1), ("F3", F3)):
    estimates = {"houdayer_hartmann": []} | {c: [] for c in configs}
    for s in range(40):
        d = synthetic(tuple(truth), F, seed=100 + s)
        hh = fss.fit_collapse(d, bounds=box, starts=4)
        estimates["houdayer_hartmann"].append(hh.params)
        for window, degree in configs:
            poly = fss.fit_collapse(d, bounds=box, method="polynomial", degree=degree,
                                    window=window, initial=hh.params, starts=4)
            estimates[(window, degree)].append(poly.params)
    for key, values in estimates.items():
        error = np.array(values) - truth
        print(label, key, "rmse", np.round(np.sqrt(np.nanmean(error**2, axis=0)), 4),
              "sd", np.round(np.nanstd(values, axis=0, ddof=1), 4))
```

## 12. Diagnostics, and what they can and cannot tell apart

`fss.py::collapse_diagnostics(data, fit, bootstrap=None)` reports, at the fit:

* **bracketing coverage**: which points enter $S$ (`bracketed`, their $\chi^2$), overall and per
  size (`coverage`, `coverage_by_size`). Unbracketed points are the ends of the scaled range: the
  largest sizes' extreme rates. Low coverage means the fit rests on few points of some sizes. For
  a polynomial fit these are the points in its window and their squared whitened residuals, and a
  note says when the window did not settle.
* **drift with the smallest size removed**: a refit without the smallest size (same settings,
  started from the fit), its parameters minus the fit's (`drift`), and in units of a bootstrap's
  errors when one is given (`drift_in_se`, as reliable as those errors, §11). A polynomial drift
  refit keeps the fit's window minus that size (`window_stable = None`, §5.7).
* **censoring and horizon peaks**, passed through per point (`censored_fraction`,
  `horizon_fraction`) with their maxima and a note naming the estimator, since a point with
  appreciable censoring is only as good as its censoring-aware estimate.
* `label` (exploratory or confirmatory) and `estimator`, recorded only.

Two more diagnostics go with a fit: `fss.py::compare_methods`, the two methods side by side
against the paired bootstrap spread of their difference (§5.7), and, for a polynomial fit,
`fss.py::polynomial_degree_scan` (§5.6).

What they can and cannot distinguish:

| situation | what shows it | what it cannot tell |
|---|---|---|
| a crossover or corrections to scaling (small-world scale $n\beta$, v1.4 §5.9) | drift with the smallest size removed, and leave-one-size-out parameters that move with the size, beyond their bootstrap errors; held-out costs well above 1 | whether the cause is a crossover, corrections to scaling or a window too wide: each moves the parameters with $n$; only more sizes (and a restricted window) separate them |
| too few sizes | status `too_few_sizes` (one size); with two sizes no drift and no leave-one-size-out refits | the true parameters: with two sizes a collapse can be nearly as good for a family of parameters |
| censoring | `censored_fraction` per point, carried from [06-statistics.md](06-statistics.md) §6.6 | whether the chosen estimator's scaling is the scaling of $\mathbb{E}[\theta]$: a restricted mean scales like $\mathbb{E}[\min(\tau, T_{\max})]$, which saturates at the horizon; a longer horizon (design §9 D3) is the remedy |
| horizon peaks | `horizon_fraction` per point | the true peak depth beyond the horizon: peak times there are lower bounds |
| a poor collapse | $S$ well above 1, large $\chi^2$ at particular points | which assumption failed (the ansatz, the window, the errors) |
| an unconstrained parameter | status `at_bound`, wide bootstrap intervals | whether widening the box would find a valid optimum |
| the two methods disagree | `compare_methods`: $\lvert\zeta\rvert$ of several (its paired error was conservative on synthetic data, §11, so a large $\lvert\zeta\rvert$ is no artifact of a small error) | which one is right: a polynomial too low in degree, a window too wide, or Houdayer and Hartmann's fit on the edge of a jump each produce it; read the degree scan and $\chi^2/\mathrm{dof}$ |
| a polynomial degree too low | `polynomial_degree_scan`: parameters that move with the degree, $\chi^2/\mathrm{dof} \gg 1$ | whether the scaling form itself fails on the window (that also raises $\chi^2/\mathrm{dof}$) |
| an unsettled polynomial window | `window_stable = False`, a note (a refit that keeps its set by design reports None: not checked); in a bootstrap, the replicates' `window_stable` and the spread of their `selected` sets | which of the fixed points is preferable; refit from other starts and compare |

None of these diagnostics proves the scaling hypothesis: a good collapse with stable parameters
is consistent with it within the sizes and rates measured.

## 13. Algorithms

### 13.1 The quality (`_Collapse.chi2`)

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

### 13.2 A fit (`fit_collapse`)

1. Statuses before searching: `too_few_sizes`, `too_few_points` (fewer than $N_{\min}$ points).
2. `minimize_bounded` on the admissible quality (§4.3) from the center (or `initial`) and `starts`
   random points, polish the winner.
3. Recount the bracketed points at the result and set the status.

About 6000 evaluations for 9 starts on 84 points (0.6 s).

### 13.3 The bootstrap and the comparison

1. Per size, check that every point (of every observable) has the same graph labels; draw
   `graph_bootstrap(labels_n, B, seed=[seed, n])`.
2. Replicate means and errors of every point from its graph means at the drawn positions (one
   vectorized pass per point).
3. Keep each replicate's usable points (finite mean, positive error) as a data set of their own
   and fit it by the data's procedure (`fss.py::_refit`: `fit_collapse` with the same settings
   and `replicate_starts` random starts; polynomial: its own window unless
   `reselect_window=False`). The comparison maps first (§8.3) and fits shared and independent
   parameters the same way (`fss.py::_null_replicate`).

Cost: $B$ (or $B\times(O + 1)$) whole fits, about 0.6 s each for 9 starts on 84 points (a
polynomial fit about 0.7 s with its reference fit), so $B = 200$ at the default 8 starts takes
2–3 minutes per observable on one core; `replicate_starts` trades fidelity for time.

### 13.4 The polynomial method (`_Polynomial.solve`, `_window_fit`)

1. Per evaluation, on the fixed set: $x$, $Y$, $dY$ by `np.power`; undefined (the objective
   $+\infty$) if anything overflows; the whitened Vandermonde matrix in the basis
   $x/\max\lvert x\rvert$; its singular value decomposition;
   $\chi^2 = \lvert \boldsymbol\eta - A\mathbf a\rvert^2$.
2. A fit: the first window at the reference (§5.3); the multi-start search; the window redrawn at
   the result until it no longer changes (usually one or two searches).

About 35 µs per evaluation for 46 points and degree 3, about 3000 evaluations for 9 starts, and
0.7 s for a fit including the Houdayer–Hartmann reference fit. Exact up to floating point (the
pseudoinverse discards singular values below the rank tolerance).

## 14. From the mathematics to the code

| definition or algorithm | code |
|---|---|
| points of one observable at one $\beta$ (§3) | `fss.py::ScalingData`, `fss.py::scaling_data` (from `GraphMean` results), `ScalingData.subset`; `y` against its graph means `fss.py::_check_graph_mean` |
| scaled coordinates | `fss.py::scaled_coordinates` |
| the quality §4.1 | `fss.py::collapse_cost` → `CollapseCost`; `fss.py::_Collapse.chi2` |
| the bounded optimizer §6 | `fss.py::minimize_bounded` → `MinimizeResult`; `fss.py::_nelder_mead` |
| a fit and its status §7 | `fss.py::fit_collapse` → `CollapseFit`; `STATUSES` |
| the shared fit §8.1 | `fss.py::fit_shared` → `SharedFit` |
| the comparison and its null §8.2–8.3 | `fss.py::compare_shared_independent` → `SharedComparison` |
| leave one size out §9 | `fss.py::leave_one_size_out` → `LeaveOneSizeOut` |
| the bootstrap of a fit §10 | `fss.py::bootstrap_fit` → `BootstrapFit`; draws `fss.py::_draws_by_size`; replicate values `fss.py::_replicate_values`; a replicate fitted by the data's procedure `fss.py::_refit` (`replicate_starts`, `reselect_window`) |
| the null replicates §8.3 | `fss.py::_null_replicate` |
| diagnostics §12 | `fss.py::collapse_diagnostics` → `CollapseDiagnostics` |
| the polynomial objective §5.1–5.2 | `fss.py::polynomial_cost` → `PolynomialCost`; `fss.py::_Polynomial.solve` |
| the window §5.3 | `fss.py::window_points`; the fit's procedure `fss.py::_window_fit` |
| polynomial fits §5 | `method="polynomial"` (with `degree`, `window`, `window_iterations`) of `fit_collapse`, `fit_shared`, `compare_shared_independent`, `leave_one_size_out`, `bootstrap_fit`; `METHODS`, `DEFAULT_DEGREE`, `MAX_DEGREE` |
| the two methods side by side §5.7 | `fss.py::compare_methods` → `MethodComparison` |
| the degree's effect §5.6 | `fss.py::polynomial_degree_scan` → `DegreeScan` |

| field | formula |
|---|---|
| `CollapseCost.chi2`, `.cost`, `.points_used` | $\chi^2_i$ (NaN if unbracketed), $S$, $N$ |
| `CollapseFit.p_c`, `.z`, `.nu`, `.cost` | $\hat{\boldsymbol\theta} = \arg\min S$ over admissible vectors in the box, $S(\hat{\boldsymbol\theta})$ |
| `CollapseFit.min_points` | $N_{\min}$ for these data |
| `SharedFit.cost`, `.costs` | $S_{\mathrm{shared}}(\hat{\boldsymbol\theta}_s)$; each observable's own mean $\chi^2$ there |
| `SharedComparison.delta`, `.null`, `.p_value` | $\Delta$, the $\Delta^\ast$, $(1 + \#\lbrace\Delta^\ast \ge \Delta\rbrace)/(1 + B')$ |
| `LeaveOneSizeOut.held_out_cost` | mean $\chi^2$ of the left-out size at its refit |
| `BootstrapFit.se`, `.lower`, `.upper` | standard deviation and percentile interval of the usable replicates |
| `BootstrapFit.replicate_starts`, `.reselect_window`, `.selected`, `.window_stable` | the replicates' random starts; whether polynomial replicates drew their own windows (None for Houdayer–Hartmann); each replicate's $\mathcal S$ as a mask over the data's points; whether it settled (None with `reselect_window=False`) |
| `SharedComparison.replicate_starts`, `.reselect_window` | the same, for the null replicates |
| `CollapseDiagnostics.drift` | parameters without the smallest size minus the fit's |
| `CollapseFit.chi2`, `.dof`, `.cost` (polynomial) | $\chi^2(\hat{\boldsymbol\theta})$, $\lvert\mathcal S\rvert - (d+1) - N_{\mathrm{free}}$, their ratio |
| `CollapseFit.selected`, `.window_range`, `.window_stable`, `.window_iterations`, `.coefficients` | $\mathcal S$; $(\min, \max)$ of $x$ on $\mathcal S$ at the fit; whether $W(\hat{\boldsymbol\theta}) = \mathcal S$ (None for a refit that keeps its set by design: leave-one-size-out, the drift, a bootstrap replicate with `reselect_window=False`); fits made; $a_k$ |
| `SharedFit.chi2`, `.dof`, `.selected` (polynomial) | $\sum_o\chi^2_o(\hat{\boldsymbol\theta}_s)$, $\sum_o(\lvert\mathcal S_o\rvert - d - 1) - N_{\mathrm{free}}$, the $\mathcal S_o$ |
| `MethodComparison.difference`, `.se_difference`, `.z` | polynomial minus Houdayer–Hartmann parameters; standard deviation of the paired replicate differences; their ratio $\zeta$ |
| `MethodComparison.se_houdayer_hartmann`, `.se_polynomial`, `.replicate_starts`, `.reselect_window` | each method's bootstrap error over the paired replicates (equal to `bootstrap_fit`'s without failures); the replicates' settings |
| `DegreeScan.params`, `.chi2_per_dof`, `.spread` | per degree; range over the usable fits |

## 15. Worked examples

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

Synthetic data with a graph hierarchy (§11's generator, written out). Holding $z$ and $\nu$, the
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

### (f) The polynomial master curve: closed form, exactness, continuity

The closed form at fixed parameters against a direct solve of the normal equations (P1); the fit of
exact data; and the objective across the measured rate 0.18 on the fit's fixed window (P2), where
the Houdayer–Hartmann quality jumps:

```python
cost = fss.polynomial_cost(noisy, 0.19, 1.0, 1.5, degree=3)
sel = cost.selected
x, Y, dY = fss.scaled_coordinates(noisy, 0.19, 1.0, 1.5)
A = np.vander(x[sel], 4, increasing=True) / dY[sel, None]
eta = Y[sel] / dY[sel]
a = np.linalg.solve(A.T @ A, A.T @ eta)
assert math.isclose(cost.chi2, float(np.sum((eta - A @ a) ** 2)), rel_tol=1e-9)
assert np.allclose(cost.coefficients, a, rtol=1e-7)

poly = fss.fit_collapse(exact, bounds=box, method="polynomial")
assert poly.window_stable and np.allclose(poly.params, [0.2, 1.0, 1.5], atol=[1e-5, 1e-4, 1e-3])
print(f"polynomial: p_c = {poly.p_c:.7f}, z = {poly.z:.6f}, nu = {poly.nu:.5f}, "
      f"chi2/dof = {poly.cost:.1e} on {poly.points_used} points")

pfit = fss.fit_collapse(noisy, bounds=box, method="polynomial")
steps = (-1e-7, 0.0, 1e-7)
chi2 = [fss.polynomial_cost(noisy, 0.18 + d, pfit.z, pfit.nu, selected=pfit.selected).chi2
        for d in steps]
quality = [fss.collapse_cost(noisy, 0.18 + d, pfit.z, pfit.nu).cost for d in steps]
print("polynomial chi2", np.round(chi2, 6), "Houdayer-Hartmann S", np.round(quality, 6))
assert abs(chi2[2] - chi2[0]) < 1e-3 and abs(quality[2] - quality[0]) > 1e-3
```

### (g) The degree's effect, and the two methods side by side

A steep master curve over $\lvert x\rvert \le 2$: low degrees bias $\nu$ and say so through
$\chi^2/\mathrm{dof}$ (§5.6). Then `compare_methods` on the hierarchical data (§5.7):

```python
steep = fss.ScalingData(n=n, p=p, se=0.01 * n, observable="steep",
                        y=n * (2.0 / (1.0 + np.exp(3.0 * (p - 0.2) * n ** (1 / 1.5))) + 0.5))
scan = fss.polynomial_degree_scan(steep, bounds=box, degrees=(2, 3, 5, 9), window=2.0,
                                  initial=[0.2, 1.0, 1.5], starts=2)
for d, f in zip(scan.degrees, scan.fits):
    print(f"degree {d}: p_c = {f.p_c:.4f}, nu = {f.nu:.4f}, chi2/dof = {f.chi2_per_dof:.3g}")
assert scan.chi2_per_dof[0] > 100 * scan.chi2_per_dof[-1] and abs(scan.params[-1, 2] - 1.5) < 0.01

both = fss.compare_methods(noisy, bounds=box, replicates=10, seed=3, starts=2)
print("polynomial - Houdayer-Hartmann", np.round(both.difference, 4), "z", np.round(both.z, 2))
assert both.failed == 0 and np.all(np.abs(both.z) < 4)
```

## 16. Validation map

All in `tests/test_study_fss.py`.

| test | what it checks | how |
|---|---|---|
| `test_collapse_cost_matches_the_houdayer_hartmann_formulas` | §4.1, Proposition 1 | random data and parameters against a loop implementation of the raw formulas, relative $10^{-8}$, same bracketed set |
| `test_the_bracketing_rule_at_the_edges` | §4.1 edge rules (top of a range, single points, no extrapolation) | hand-computed $\chi^2$ on binary-exact abscissas |
| `test_a_linear_scaling_function_collapses_perfectly` | Proposition 3's exact case | $S < 10^{-20}$ at the truth |
| `test_the_quality_does_not_depend_on_the_units_of_y` | Proposition 2 | relative $10^{-10}$ |
| `test_extreme_parameters_give_no_quality_instead_of_overflow`, `test_overflowing_scaled_values_give_no_quality` | §13.1 step 2 | NaN quality |
| `test_the_optimizer_finds_known_minima`, `test_a_minimum_outside_the_box_lands_on_its_bounds`, `test_fixed_coordinates_and_infinite_regions`, `test_the_optimizer_is_deterministic_given_its_seed` | §6, Proposition 5 | Rosenbrock, Himmelblau, Booth, a quadratic; corner minimum; fixed coordinates; $+\infty$ regions; bitwise determinism |
| `test_the_fit_recovers_the_parameters_of_exact_data` | §7 | $\lvert\hat p_c - 0.2\rvert < 10^{-3}$, $\lvert\hat z - 1\rvert < 10^{-2}$, $\lvert\hat\nu - 1.5\rvert < 2\times10^{-2}$ |
| `test_the_fit_recovers_the_parameters_within_its_bootstrap_errors` | §10, §11 | each parameter within 4 bootstrap errors (false alarm about $2\times10^{-3}$, from the measured per-dataset error ratios of §11) |
| `test_the_bootstrap_errors_match_the_true_spread_of_the_fits` | §10, §11, v1.4 §7.4 (coverage on synthetic controls) | 80 synthetic fits per method give the true spread, 12 bootstraps of 19 replicates the errors; $0.6 \le \rho \le 2$ for each parameter of each method and $\rho \ge 0.6$ for the paired error of the difference (false alarm about $3\times10^{-3}$, from 24 repeats of the test; errors 2 times too small fail in every repeat for either method) |
| `test_every_bootstrap_replicate_is_fitted_like_the_data`, `test_polynomial_replicates_draw_their_own_windows`, `test_compare_methods_bootstraps_like_bootstrap_fit`, `test_bootstrap_settings_are_checked` | §10, §5.3 (ruling R-FSS) | every replicate equals `fit_collapse` on the replicate's data with the same settings (parameters, cost, status, window and `window_stable`, bitwise); a settled replicate window equals `window_points` at its own fit, and some replicate's window differs from the data fit's; `reselect_window=False` keeps the data fit's window; `compare_methods`' errors equal `bootstrap_fit`'s; `replicate_starts` and `reselect_window` checked |
| `test_the_bootstrap_resamples_graphs_once_per_size_for_every_rate`, `test_bootstrap_points_must_share_their_graphs` | §10 pairing | the draws are `graph_bootstrap(labels, B, [seed, n])`; a constant offset between two rates survives every replicate exactly |
| `test_fits_are_deterministic_given_the_seed`, `test_the_fit_does_not_depend_on_the_units_or_on_duplicated_points` | determinism, scaled inputs, duplicated points | bitwise; relative $10^{-6}$; duplicated data refit near the truth |
| `test_a_fixed_parameter_stays_fixed`, `test_a_truth_outside_the_box_is_reported_at_its_bound`, `test_too_few_sizes_or_points_give_a_status_not_a_crash`, `test_min_points_is_respected`, `test_a_spent_budget_is_reported_as_not_converged` | §7 statuses | each status, no exception |
| `test_the_shared_fit_recovers_a_common_set` | §8.1 | two observables with different $F$, same parameters |
| `test_the_comparison_prefers_independent_exponents_for_different_nu` | §8.2–8.3 (power) | $\nu = 1.2$ against 2.5: p-value $1/20$ with 19 replicates, $\Delta$ above 4 times the largest null replicate |
| `test_the_comparison_keeps_its_size_when_the_parameters_are_shared` | §8.3 (size), both methods | 4 pairs generated with the same parameters, 9 replicates each: mean p-value at least 0.15; under a valid null each p-value is at least uniform, so by the Irwin–Hall law the false alarm is at most $0.6^4/4! = 5.4\times10^{-3}$; a degenerate null (every $\Delta^\ast = 0$) puts every p-value at $1/10$ and fails (milder shrinkage passes it: see the next row) |
| `test_null_replicates_are_the_comparison_on_mapped_resampled_data` | §8.3 (the null replicates), both methods | each $\Delta^\ast$ equals the comparison recomputed in the test from the mapping formula, `graph_bootstrap(labels_n, B, [seed, n])` draws and the public `fit_collapse`/`fit_shared` (relative $10^{-12}$): a null scaled down, frozen at the shared parameters or zero fails |
| `test_the_null_mapping_moves_each_observable_onto_the_shared_parameters` | Proposition 6 | equal coordinates and quality at generic parameters; the slice equivalence at three $(z, \nu)$ for both objectives (relative $10^{-9}$); the refit of mapped data within $2\times10^{-3}$ of the shared set (near it, not on it, §8.3) |
| `test_leaving_one_size_out_of_exact_data_gives_the_same_parameters` | §9 | every refit near the truth; held-out costs below 0.05 |
| `test_diagnostics_report_coverage_drift_and_the_fractions_passed_through`, `test_drift_in_units_of_the_bootstrap_errors`, `test_a_failed_drift_refit_is_noted` | §12 | coverage equal to the quality's bracketed set; drift equal to the refit's difference; fractions passed through; notes |
| `test_scaling_data_*`, `test_fit_bounds_are_checked`, `test_fit_settings_are_checked`, `test_shared_tools_check_their_inputs` | input checks | messages that say what to change; `test_scaling_data_refuses_a_y_that_is_not_the_mean_of_its_graph_means`: a median carrying graph means is refused (§3), rounding and NaN-omitting nested means accepted |
| `test_polynomial_cost_is_the_exact_weighted_least_squares` | P1, §5.2 | $\chi^2$, coefficients and residuals against normal equations solved in exact rationals (degrees 1–3), relative $10^{-10}$ ($\chi^2$) and $10^{-8}$ |
| `test_the_window_rules` | §5.3 | the overlap and $\lvert x\rvert \le x_{\max}$ windows against their definitions; an empty overlap; refused rules |
| `test_the_polynomial_fit_recovers_exact_parameters` | §5 | both windows: $(p_c, z, \nu)$ within $(10^{-5}, 10^{-4}, 10^{-3})$; a settled, self-consistent window; $\mathrm{dof}$ and $\chi^2/\mathrm{dof}$ |
| `test_the_polynomial_objective_is_continuous_where_the_quality_jumps` | P2 against Proposition 4 | steps of $10^{-9}$ in $p_c$ across 0.18 change $\chi^2$ by at most twice the local slope times the step, while $S$ jumps by more than $10^{-3}$; smooth in $z$ and $\nu$ |
| `test_the_degree_changes_the_answer_until_it_is_enough` | §5.6 | steep master curve over $\lvert x\rvert \le 2$: degree 2 biased with $\chi^2/\mathrm{dof}$ over 100 times that of degree 9, which recovers the truth |
| `test_chi2_per_dof_is_about_one_for_independent_errors` | §5.5 | 20 datasets with independent errors: mean $\chi^2/\mathrm{dof}$ within 4 standard errors of 1 (measured $0.94 \pm 0.06$; the error is estimated from the 20 values, so the false alarm is the Student-$t_{19}$ tail, $7.7\times10^{-4}$) |
| `test_the_two_methods_agree_on_clean_and_on_noisy_data` | §5.7, §12 | exact data: differences below $(10^{-3}, 10^{-2}, 2\times10^{-2})$; noisy data: $\lvert\zeta\rvert < 4$ (each $\zeta$ about Student-$t_{19}$: false alarm $7.7\times10^{-4}$ per parameter, $2.3\times10^{-3}$ for the three; a bias difference of one paired error raises the $\nu$ term to about $6\times10^{-3}$), paired error below the unpaired $\sqrt{\mathrm{se}_{\mathrm{HH}}^2 + \mathrm{se}_{\mathrm{poly}}^2}$ |
| `test_too_few_points_for_the_polynomial_give_a_status`, `test_polynomial_paths_without_a_usable_answer`, `test_an_unsettled_window_is_reported` | §5.3 admissibility, statuses | distinct rates, one size in the window, $\mathrm{dof} = 1$ accepted and 0 refused, overflow everywhere, failed observables in a shared fit, an unsettled window reported and a settled one from the same start |
| `test_the_polynomial_method_runs_through_every_tool`, `test_polynomial_settings_are_checked` | §5.7 | shared fit, leave-one-size-out (refits report `window_stable = None`, also in JSON), diagnostics (the drift refit too), and the comparison with `method="polynomial"` (P3's $\Delta$, re-selected windows); input checks |

## 17. Limitations and departures

* **The ansatz is unverified** (v1.4 §5.9): a fit measures how well the data collapse, not
  whether the hypothesis is true.
* **The quality is piecewise continuous** (Proposition 4): fits can sit at the edge of a jump, at
  a measured rate to within rounding. This is Houdayer and Hartmann's quality as published:
  continuous between the changes of bracketing and discontinuous at them, as design §11 now says.
  The continuous alternative of ruling R-FSS is the polynomial method (§5).
* **The polynomial method** needs a window and a degree that suit the data (§5.6, §5.8): too low a
  degree biases the parameters, and the default overlap window, being narrow, gives the estimates a
  heavier tail than Houdayer and Hartmann's (§11). Its window map can have several fixed points.
  Its bootstrap replicates draw their own windows (ruling R-FSS), which made its errors
  conservative on §11's ensemble (ratio 1.43 on average for $\nu$, 1.15 at the median). Its
  $\chi^2/\mathrm{dof}$ is calibrated only for independent errors (§5.5).
* **$S$ is not a calibrated chi-square**: points share brackets and the errors of a size are
  correlated across rates (shared graphs). Parameter uncertainties come from the graph bootstrap
  (§10), whose errors on the synthetic ensemble of §11 were 1.04, 0.98 and 1.11 of the true
  standard deviations of $(p_c, z, \nu)$ with 20 graphs, and 10–14% low on a coarser design (3
  sizes, 9 rates, no random starts); with fewer graphs or coarser designs expect lower ratios.
  These are measurements on synthetic ensembles, not guarantees, and a bootstrap of a
  discontinuous estimator has no general consistency theorem behind it.
* **Each bootstrap replicate costs a whole fit** (§13.3): $B = 200$ with 8 random starts takes
  minutes per observable. `replicate_starts` lowers the cost and makes the replicates only
  approximate the estimator; the calibration of §11 used as many starts as the fits.
* **The bootstrap null** of the comparison is an approximation in four ways (§8.3); on §11's
  ensemble it kept its size (1 rejection of 40 at 5%) and was somewhat conservative. A single
  rejection at 5% is weak evidence. Report it with the parameter differences and their errors.
* **Survival-based $y$**: the restricted mean to a common horizon is fully supported (it is a
  nested mean); a Kaplan–Meier median can be fitted only with an error the user supplies, and
  cannot be bootstrapped or compared here (§3).
* **Sizes and $\beta$ values are independent samples** by ruling R-SEED (design §3, §11): graph
  and run seeds are domain-tagged tuples that depend on $(n, \beta)$, so only the rates of one
  $(n, \beta)$ share graphs, gates and mask uniforms. The bootstrap's one resampling per
  $(n, \beta)$, applied to every rate and observable, is therefore the right unit, and treating the
  sizes as independent is exact.
* **The scaling window is chosen by the user** (`ScalingData.subset`); nothing here selects it.
* **Conventions introduced here**: the admissibility floor $N_{\min} = \max(3, \lceil P/2\rceil)$
  by default; the optimizer's tolerances, steps and polishing (§6.2); the bracketing tie rules
  (§4.1); the drift diagnostic removes the smallest size only (leave-one-size-out covers the
  others); for the polynomial method, the fixed window redrawn until it settles, its reference
  (the Houdayer–Hartmann fit) and the default degree 3 (§5.3); bootstrap and null replicates use
  the data fit's seed for their random starts (§10); the agreement of $y$ with its graph means to
  $10^{-9}$ of the largest absolute graph mean (§3). The `label` (exploratory or confirmatory) is
  recorded and changes nothing.
