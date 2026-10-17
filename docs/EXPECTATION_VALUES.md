# Expectation values for depth scans

What each average in a depth scan means, how to estimate it without bias, and how to get its
uncertainty. The simulator produces per-trajectory values (`docs/SWEEPS.md`); everything here
happens after that, in your analysis. The definitions follow the study specification
(`docs/misc/watts_strogatz_study_spec_v1_4.md`, "v1.4") §5–§6. Section numbers below are this
document's unless they say v1.4 or SWEEPS.

## 0. The rules

| # | Rule | Section |
|---|---|---|
| 1 | Compute every observable **per trajectory** first (k, τ_ℓ, d_cyc, D, T*), then average. Never average states or curves before a nonlinear step unless that is the declared target. | 2, 7 |
| 2 | **Equal weight per graph**: average the circuits of a graph, then average the graphs. | 2.3 |
| 3 | Uncertainty comes from **graphs**: SE = sd(graph means)/√N_G; for anything nonlinear or combining depths or rates, resample whole graphs (bootstrap). | 2.5, 8 |
| 4 | First passages: use the **survival curve** P(τ_ℓ > t), the **restricted mean** E[min(τ_ℓ, T_max)] or quantiles; never average only the observed τ. | 5 |
| 5 | Observables undefined at k = 0 (d_cyc, D): choose **zero-extended** or **conditional on a code**, and use the matching estimator. Convert the engine's k = 0 values first. | 6 |
| 6 | T* is a per-trajectory random variable; E[T*] is not the peak of the mean curve. Record undefined and horizon-limited peaks. | 7 |
| 7 | Compare rates **within** the same graph and circuit (paired), then average graphs. | 3.1 |
| 8 | For the ensemble mean, many graphs beat many circuits per graph. | 2.7 |

## 1. The ensemble

**Notation.**

| symbol | meaning | in the data |
|---|---|---|
| $n$, $\beta$, $p$ | size, rewiring probability, measurement rate (fixed, not random) | condition folder |
| $G$ | interaction graph, $G\sim\mathrm{WS}(n,\beta)$ | `graph_index` = $g$ |
| $g$, $c$ | graph label, circuit label within the graph | `rows()["g"]`, `["c"]` |
| $u=g\cdot s+c$ | stream index, $s$ the stride | `trajectories` |
| $U$, $M$ | gate schedule, measurement masks (all layers) | from the run seed and $u$ |
| $t$, $T$ | depth in completed layers; $T_{\max}$ the horizon | checkpoints |
| $\mathcal T$, $\mathcal T_+$ | checkpoint set, and its positive part | `res.layers(name)` |
| $k_t$ | logical qubits, $k_t=S(R)$ at depth $t$ | `code_dimension` |
| $\tau_\ell$ | first depth with $k_t\le\ell$ | first passages |
| $h_t$ | has a code: $\mathbf 1\{k_t>0\}$ | from $k_t$ |
| $N_G$, $C_g$ | graphs per condition, circuits on graph $g$ | sweep spec |

**1.1 Graphs.** $C(n,2)$ is the circulant ring with edges $\{i,i\pm1\}$ and $\{i,i\pm2\}$ (mod
$n$): degree 4 ($K=2$ neighbors per side), $2n$ edges for $n\ge5$. $\mathrm{WS}(n,\beta)$ is the
law of the sequential rewiring of $C(n,2)$ (v1.4 §2; SWEEPS §3). At $\beta=0$ every graph is
$C(n,2)$.

**1.2 Circuits given a graph.** Layer $t$ applies $h(n)=\lfloor n/2\rfloor$ gates, each on an edge
drawn uniformly from $E(G)$ with replacement, each a uniform two-qubit Clifford (11,520 up to
phase; phase-free runs use only its symplectic class, exactly uniform on 720: SWEEPS §2). Then
every system qubit is measured in $Z$ independently with probability $p$. Gates and masks are
independent of each other given $G$, and independent across layers.

**1.3 The joint law** of one trajectory to depth $T$ is
$\mu^{\mathrm{WS}}_{n,\beta}(G)\,\kappa_{G,T}(U)\,\nu_{p,T}(M)$ (v1.4 §6.1). Born outcomes do not
enter: every phase-free quantity (k, entropies, erasure correctability, so τ, d, D, T*) is the
same for every outcome history (v1.4 §6.2, the induction on operations), so there is nothing to
average over them.

**1.4 Prefix lemma.** The first $T$ layers of a trajectory run to $T_{\max}\ge T$ have exactly the
law of a trajectory run to depth $T$.

*Proof.* $\kappa_{G,T_{\max}}$ and $\nu_{p,T_{\max}}$ are products over layers, so
$\kappa_{G,T_{\max}}=\kappa_{G,T}\otimes\kappa_{G,T_{\max}-T}$ and likewise for $\nu$. Summing out
layers $T+1,\dots,T_{\max}$ leaves $\kappa_{G,T}\otimes\nu_{p,T}$. ∎

So one run to $T_{\max}$ gives an exact sample of $F_T$ at every checkpoint $T$. Values at
different depths of one trajectory share layers and are dependent. That does not affect
pointwise means. It does affect the uncertainty of anything that combines depths (§3.2, §8).

## 2. Targets and the nested estimator

**2.1 Target.** For an observable $F_T$ at depth $T$:
$$\mu_F(n,\beta,p,T)=\mathbb E_G\bigl[m_F(G,T)\bigr],\qquad m_F(G,T)=\mathbb E[F_T\mid G].$$

**2.2 Design.** Graphs $g=0,\dots,N_G-1$ are independent draws from $\mathrm{WS}(n,\beta)$ (distinct
graph seeds). Circuits $c=0,\dots,C_g-1$ on graph $g$ are independent given the graph (distinct
streams $u$). The counts $C_g$ are fixed in advance, never chosen from the values. Seeded streams
approximate these ideal independent draws (v1.4 §7).

**2.3 Estimator** (v1.4 §6.3):
$$\widehat m_g=\frac1{C_g}\sum_{c=0}^{C_g-1}F_{g,c},\qquad
\widehat\mu_F=\frac1{N_G}\sum_{g=0}^{N_G-1}\widehat m_g .$$

*Unbiased.* $\mathbb E[\widehat m_g\mid G_g]=m_F(G_g)$, so
$\mathbb E\widehat\mu_F=\mathbb E_G\,m_F(G)=\mu_F$. ∎

The pooled mean over all rows, $\sum_{g,c}F_{g,c}/\sum_g C_g$, weights graph $g$ by $C_g$. It
equals $\widehat\mu_F$ when every $C_g$ is equal and is biased otherwise, for example in an
interrupted condition. Excluding rows by their values (censored, NaN) changes the target (§5, §6).

**2.4 Variance.** Let $\sigma_b^2=\operatorname{Var}_G m_F(G)$ (between graphs) and
$\sigma_w^2=\mathbb E_G\operatorname{Var}(F_T\mid G)$ (within a graph, everything else: gates,
masks, and Monte Carlo region sampling if any). With equal $C_g=C$:
$$\operatorname{Var}\widehat\mu_F=\frac1{N_G}\Bigl(\sigma_b^2+\frac{\sigma_w^2}{C}\Bigr).$$

**2.5 Graph-cluster standard error** (v1.4 §6.4):
$$\widehat{\mathrm{SE}}^2=\frac{1}{N_G(N_G-1)}\sum_g(\widehat m_g-\widehat\mu_F)^2 .$$

*Unbiased for* $\operatorname{Var}\widehat\mu_F$, even with unequal $C_g$. The $\widehat m_g$ are
independent with common mean $\mu_F$ and variances $v_g$. Then
$\mathbb E\sum_g(\widehat m_g-\widehat\mu_F)^2=\sum_g v_g-N_G\operatorname{Var}\widehat\mu_F=\frac{N_G-1}{N_G}\sum_g v_g$,
and $\operatorname{Var}\widehat\mu_F=\sum_g v_g/N_G^2$. ∎

It needs $N_G\ge2$. Treating the $N_GC$ rows as independent gives
$(\sigma_b^2+\sigma_w^2)/(N_GC)$. That is too small by the factor
$(C\sigma_b^2+\sigma_w^2)/(\sigma_b^2+\sigma_w^2)$ whenever graphs differ ($\sigma_b^2>0$).

**2.6 Variance components** (equal $C\ge2$): with $s_g^2$ the within-graph sample variance
(ddof 1), $\widehat\sigma_w^2=\frac1{N_G}\sum_g s_g^2$ and
$\widehat\sigma_b^2=\frac1{N_G-1}\sum_g(\widehat m_g-\widehat\mu_F)^2-\widehat\sigma_w^2/C$. Both are
unbiased. $\widehat\sigma_b^2$ can come out negative: report it, and read it as "consistent with
0".

**2.7 Allocation.** With cost $c_G$ per graph and $c_C$ per circuit, minimizing
$\operatorname{Var}\widehat\mu_F$ at a fixed budget gives
$$C^*=\sqrt{\frac{c_G}{c_C}\,\frac{\sigma_w^2}{\sigma_b^2}} .$$
Here a graph costs microseconds and a trajectory to $10^6$ layers costs seconds, so
$c_G/c_C\approx10^{-6}$ and $C^*<1$. **For the ensemble mean at a fixed number of trajectories
$N=N_GC$**, $\operatorname{Var}=(C\sigma_b^2+\sigma_w^2)/N$ is smallest at $C=1$.

Many circuits per graph pay only when the per-graph mean $m_F(G)$ is itself the object:
graph-to-graph fluctuations, correlation with graph measures, or a nonlinear statistic of a
per-graph curve. They also pay when you want $\sigma_b^2$ and $\sigma_w^2$ separately, for which
$C=2$ is enough.

Recommendation: $C=2$ (or 4) and as many graphs as the budget allows. A sweep can add circuits
per graph later, up to the stride, without changing a saved trajectory (SWEEPS §10). "Many
circuits per graph, then many graphs" is unbiased; it is only less efficient for $\mu_F$.

## 3. Across rates, depths, sizes

**3.1 Rates.** For one $(n,\beta)$ every rate uses the same graphs (the graph seed has no $p$),
the same gates, and nested masks (SWEEPS §5). Each rate's marginal law is exact. Estimates at
different $p$ are dependent, usually positively. Pointwise errors follow §2. For a difference or
slope in $p$, difference within the same row $u$ first, then use §2 on the differences.

**3.2 Depths.** Values at different $t$ of one trajectory are dependent (§1.4). Any statistic
that combines several depths or several rates needs the graph bootstrap (§8). That covers T*,
quantiles, crossing points, areas under curves and collapse fits. A joint fit that treats the
points $(n,p,t)$ as independent understates its errors.

**3.3 Sizes and β.** Seeds differ across $n$ and $\beta$ (SWEEPS §4), so their estimates are
independent.

**3.4 Same labels, same objects.** The same $(n,\beta,g)$ is the same graph at every $p$, and the
same $u$ gives the same gates and nested masks at every $p$.

## 4. Logical qubits and code rate

$k_t=n-\dim W_t^{\mathrm{phys}}=S(R)$, and the code rate is $R_t=k_t/n$ (v1.4 §5.2). Both are
linear in $k_t$, so §2 applies as is. $k_t$ is an integer, nonincreasing in $t$ (v1.4 §5.2,
citing Li–Fisher Eqs. (9)–(11)): $k_0=n$ in the purification picture and $k_0=1$ in the single
picture. Stored: `code_dimension` at the checkpoints, and $k$ at every layer through the
first passages (§5).

## 5. First-passage depths τ_ℓ

**5.1 Definition** (v1.4 §5.2): $\tau_\ell=\min\{t\ge0:k_t\le\ell\}$ for $\ell=0,\dots,n$. It
is nonincreasing in $\ell$, and $\tau_\ell=0$ for $\ell\ge k_0$. $\tau_0$ is the purification
time. The engine records every $\tau_\ell$ online at every layer (exact, not on the checkpoint
grid), or $-1$ when $k_t>\ell$ through the last simulated layer: right-censored at
$T_{\max}$.

**5.2 Event identity.** For every $t\ge0$: $\{\tau_\ell>t\}=\{k_t>\ell\}$.

*Proof.* If $k_t>\ell$ then $k_s\ge k_t>\ell$ for all $s\le t$, so $\tau_\ell>t$. If
$k_t\le\ell$ then $\tau_\ell\le t$. ∎

**5.3 Survival curve:**
$$S_\ell(t)=\Pr(\tau_\ell>t)=\mathbb E\,\mathbf 1\{k_t>\ell\}=\mathbb E_G\,S_\ell(t\mid G).$$
At each $t$ this is the expectation of an indicator, so §2 applies verbatim: per-graph mean of
$\mathbf 1\{\tau_\ell>t\}$, mean over graphs, cluster SE. From the first passages it is exact
at every $t\le T_{\max}$, not only at checkpoints.

**5.4 Censoring is administrative and common.** Every trajectory of a condition runs to the same
$T_{\max}$, or stops at $\tau_0$ (`stop_when_purified`), by which point every threshold has been
reached. So for $t<T_{\max}$ the indicator $\mathbf 1\{\tau_\ell>t\}$ is observed for every
trajectory, and the graph-averaged empirical survival is exact. Kaplan–Meier is not needed: it
reduces to this when all censoring is at the horizon. Trajectories with different horizons
would need it; do not mix horizons in one condition.

**5.5 Means.** For an integer $\tau\ge0$, $\tau=\sum_{t\ge0}\mathbf 1\{\tau>t\}$, so
$$\mathbb E\tau_\ell=\sum_{t\ge0}S_\ell(t),\qquad
\mathbb E\min(\tau_\ell,T)=\sum_{t=0}^{T-1}S_\ell(t)\quad(T\le T_{\max}).$$
The **restricted mean** $\mathbb E\min(\tau_\ell,T_{\max})$ is always identified. Estimate it by
the nested mean of $\min(\tau_\ell,T_{\max})$, with censored entries set to $T_{\max}$.

If $S_\ell(T_{\max})>0$, $\mathbb E\tau_\ell$ is **not identified**. Two laws that agree on
$[0,T_{\max}]$ and place the remaining mass at different depths beyond it give the same data and
different means. The average of the observed $\tau$ only estimates
$\mathbb E[\tau_\ell\mid\tau_\ell\le T_{\max}]$, which is smaller. Report the censored fraction
$S_\ell(T_{\max})$ per condition. Fit $\mathbb E\tau_\ell$ (v1.4 §5.9) only where it is
negligible; elsewhere use the restricted mean (a lower bound, say so) or quantiles.

**5.6 Quantiles.** $t_q=\min\{t:S_\ell(t)\le1-q\}$, for example the median $q=1/2$. It is
identified when $S_\ell(T_{\max})\le1-q$. Plug in the graph-averaged $\widehat S_\ell$, and get
the uncertainty from §8. The mean of per-graph medians is a different target.

**5.7 Checkpoint-only times** $\widehat\tau^{\mathrm{chk}}_\ell\ge\tau_\ell$ (v1.4 §5.2) are interval
bounds. Never substitute them for $\tau_\ell$.

**5.8 The family.** All $\tau_\ell$ of a trajectory are dependent, so joint statements over
$\ell$ use §8. Common choices for scaling are a fixed $\ell$ (0, 1) or $\ell=\lfloor fn\rfloor$.

## 6. Distances and robustness at one depth

**6.1** $h_t=\mathbf 1\{k_t>0\}=\mathbf 1\{\tau_0>t\}$.

**6.2 Ring contiguous distance** $d_{\mathrm{cyc}}(t)$ (v1.4 §5.5): the smallest $m$ such that
some cyclic interval $\{a,\dots,a+m-1\}$ of the **original ring order** $0,\dots,n-1$ leaks
reference information. It is exact (no sampling) and not graph-intrinsic at $\beta>0$. The
engine stores `NaN` when $k_t=0$. The zero extension is $d^0_t=d_t h_t$ (`NaN`→0).

**6.3 Two estimands; pick one and use its estimator.**

- **Zero-extended:** $\mathbb E\,d^0_t$, linear, so §2 applies to $d^0$. v1.4 uses this for
  peak traces. It mixes two effects: $\mathbb E d^0_t=\mathbb E[d_t\mid k_t>0]\Pr(k_t>0)$.
- **Conditional on a code:** $r_t=\mathbb E[d_t\mid k_t>0]=\mathbb E d^0_t/\mathbb E h_t$, a ratio
  of two linear targets. Estimate it with $\widehat r=\widehat\mu_{d^0}/\widehat\mu_h$ (both nested,
  equal graph weights). It is consistent, with bias $O(1/N_G)$. Its SE by linearization is
  $\widehat{\mathrm{SE}}(\widehat r)=\widehat\mu_h^{-1}\sqrt{\sum_g e_g^2/(N_G(N_G-1))}$ with
  $e_g=\widehat m^{d^0}_g-\widehat r\,\widehat m^{h}_g$, or use §8.

  With equal $C_g$, $\widehat r$ equals the plain mean over the rows that hold a code. The mean
  of per-graph conditional means, $\mathbb E_G\,\mathbb E[d\mid k>0,G]$, is a different target:
  it is undefined for a graph with no code left, and it weights graphs regardless of survival.

**6.4 Uniform robustness** $D_t(0)$ (v1.4 §5.7):
$D_t=\sum_{m=0}^{n-1}(1-P_t(m))$ if $k_t>0$, and $0$ if $k_t=0$.

*Identity* $D_t=\mathbb E_\pi M_\pi$, where $M_\pi$ is the first failing prefix size of a uniform
ordering $\pi$. A prefix of size $m$ is a uniform $m$-subset, and failure is monotone under
inclusion, so $\Pr(M_\pi>m)=1-P_t(m)$. Then $\mathbb E M_\pi=\sum_{m\ge0}\Pr(M_\pi>m)$, with
$P_t(n)=1$ when $k_t>0$. ∎

The probe `erasure_first_failure` stores $S$ independent $M_\pi$ per evaluation, and **$n+1$
when $k_t=0$**. Convert: $\widehat D_t=\overline{M_\pi}$ if $k_t>0$, else 0. Unconverted, every
purified trajectory would count as $n+1$. $\widehat D_t$ is unbiased for $D_t$ given the
trajectory. Its sampling variance $\operatorname{Var}(M_\pi)/S$ joins $\sigma_w^2$: harmless for
means, but it matters for T* (§7.5). Bounds when $k_t>0$: $1\le D_t\le(n+1)/2$ (v1.4 §5.8).

**6.5 Graph-correlated robustness** $D_t(J)$, $J>0$ (v1.4 §5.7). This is the
"graph-correlated" analogue as implemented: regions drawn with probability
$\propto e^{-Jb_G(A)}$, so they favor few boundary edges on the trajectory's own graph.
$D_t(J)=\sum_{m=0}^{n-1}(1-P_{t,G,J}(m))$ with $P(0)=0$, and $0$ when $k_t=0$. The probe
`correlated_erasure` stores $\widehat P(m)$ = failures/$S$ per size and **records 0 when
$k_t=0$**. Summed unconverted, that gives $D=n$ for a purified trajectory. Convert:
$\widehat D_t(J)=1+\sum_{m=1}^{n-1}(1-\widehat P(m))$ if $k_t>0$, else 0.

- **Exact samplers** (a ring or path geometry, including $\beta=0$): $\widehat P(m)$ is unbiased
  with variance $P(1-P)/S$.
- **On rewired graphs**, a Metropolis swap chain: consistent after burn-in. Check mixing with
  `sample_erasure_regions` on the campaign's graphs before a long run (AGENTS.md "Correlated
  erasures").
- **Sizes not sampled** are interpolated, which adds bias. Sample every size, or the binomial
  window you need.

$D(J)$ at $J>0$ is not an expected first-failure size (v1.4 §5.7). The $J\to\infty$ limit, regions
of minimum boundary, has exact routes for saved codes. $d_{\mathrm{conn}}$ (v1.4 §5.6) is not
implemented.

**6.6 Anything else from snapshots.** Any phase-free quantity of the state at a saved checkpoint
is a per-trajectory value. Compute it per trajectory, then average by §2.

## 7. The peak depth T*

**7.1 Definition** (v1.4 §5.8), for a trajectory's trace $d(t)$ on $\mathcal T_+$ ($d^0$, or
$D$ with 0 at $k=0$): $d^*=\max d$, $\mathcal A=\{t\in\mathcal T_+:d(t)=d^*\}$, and
$T^*=\exp\bigl(\tfrac1{|\mathcal A|}\sum_{t\in\mathcal A}\ln t\bigr)$. It is undefined when
$d^*=0$.

**7.2 Targets.** $T^*$ is a random variable of the trajectory. The targets are $\mathbb E T^*$,
the geometric mean $\exp\mathbb E\ln T^*$, and its quantiles, each estimated per trajectory and
then by §2.

For $d^0$ and $D$, $d^*=0$ exactly when $k_1=0$ ($k$ never increases), so $T^*$ is undefined
only for trajectories purified within the first layer. Report that fraction. Averaging over the
defined ones conditions on $\tau_0>1$, which is a different target unless the fraction is 0.

**7.3 Not the same statistics:** $\mathbb E T^*\ne T^*(\mathbb E d)$, the peak of the mean curve,
and $\ne\mathbb E_G T^*(m_d(G,\cdot))$, the mean of per-graph peaks. $T^*$ is a nonlinear
functional of the whole curve (§12 has an example). v1.4 §5.9 scales $\mathbb E\theta$ with
$\theta$ per trajectory. Use the others only as declared diagnostics.

**7.4 Grid resolution.** $T^*$ lives on log-midpoints of checkpoint subsets. The geometric rule
has ratio $\rho\approx T_{\max}^{1/(N_T-1)}$ between large checkpoints. At $T_{\max}=10^6$,
$N_T=61$ gives $\ln\rho=0.23$ and $N_T=121$ gives $0.115$, while the grid is integer-dense at
small $t$. Choose $N_T$ so that $\ln\rho$ is small against the change of $\ln T^*$ across your
sizes; for example $z\ln2\approx0.69$ per doubling of $n$ if $z\approx1$. The grid statistic is
the defined target. Its distance from the all-depth peak is about $\ln\rho$ for a trace unimodal
at the grid scale, with no bound in general. $k$ and $d_{\mathrm{cyc}}$ are cheap per checkpoint
(see the depth-scan benchmark), so dense grids are affordable for them.

**7.5 Ties and noise.** $k$ and $d_{\mathrm{cyc}}$ are exact integers, so ties are real and the
log-midpoint handles them exactly.

For Monte Carlo traces ($\widehat D$), noise breaks ties and moves the argmax:
$\mathbb E\,T^*(\widehat D)\ne\mathbb E\,T^*(D)$. Remedies:
- more samples $S$;
- a predeclared plateau rule, $\mathcal A_z=\{t:\widehat D(t)\ge\max\widehat D-z\,\mathrm{se}(t)\}$,
  reporting the sensitivity to $z$ (v1.4 requires one);
- exact values for small $n$ (saved codes, exact counts).

An implementation of the plateau rule exists on branch `ws-study-full-archive`
(`study/observables.py`).

**7.6 Horizon.** If $T_{\max}\in\mathcal A$, the peak may lie beyond the horizon. Record a flag
per trajectory. $\mathbb E T^*$ is meaningful only when that fraction is small, as with
censoring.

**7.7 With τ.** Joint per-trajectory statements ($T^*/\tau_1$, whether $t\ge\tau_1$ on
$\mathcal A$) use the same trajectory's values and §8 for errors (v1.4 §5.9).

## 8. The graph bootstrap

For a statistic $\Theta$ that is nonlinear (a ratio, quantile, T* summary or fitted exponent) or
that combines depths or rates:

1. For $b=1,\dots,B$: draw $N_G$ graph labels with replacement, separately for each $(n,\beta)$
   (they are independent, §3.3).
2. Keep **all** rows of each drawn graph: every circuit, every rate, every depth. The same draw
   serves every $p$ of that $(n,\beta)$, which keeps the coupling of §3.1.
3. Recompute $\Theta_b$ from the resampled rows, with the estimators of §2–§7.

Then $\widehat{\mathrm{SE}}=\operatorname{sd}_b\Theta_b$, with percentile intervals; use
$B\ge1000$. Graphs are the independent units (§2.2). Resampling circuits or checkpoints instead
would understate the error.

## 9. A depth-scan recipe

| what | how |
|---|---|
| checkpoints | geometric, $N_T\approx100$ to $T_{\max}$ (§7.4); $k$ at every layer through first passages |
| probes on the checkpoints | `code_dimension`, `contiguous_distance`; `erasure_first_failure` ($S\ge100$) for $D(0)$; `correlated_erasure` where affordable; `tableau_snapshot` at a few depths (disk) |
| stopping | `stop_when_purified: true`. After $\tau_0$ every code value is known exactly ($k=0$, $d^0=0$, $D=0$), so stopping is not adaptive-sampling bias. Apply the conversions of §6. |
| allocation | $C=2$ circuits per graph, as many graphs as affordable, the same master seeds for every $p$ |
| per trajectory | $k_t$, $\mathbf 1\{\tau_\ell>t\}$, $\min(\tau_\ell,T_{\max})$, $d^0_t$, $h_t$, $\widehat D_t$, $T^*$, horizon flag |
| average | per graph, then over graphs (§2); ratios by §6.3; nonlinear and multi-depth by §8 |
| report | censored fraction per $(n,\beta,p,\ell)$; undefined and horizon-limited $T^*$ fractions; $N_G$, $C$; which estimand (§6.3, §7.2) |

## 10. Exact replication

- A saved value is fixed by the sweep spec (`sweep.json`, with its digest), the seeds, the
  stream index and the randomness contract (2):
  - graph seed $(\mathtt{WSG1},\text{master}_G,n,w(\beta),g)$;
  - run seed $(\mathtt{WSC1},\text{master}_C,n,w(\beta))$;
  - $u=g\,s+c$ (SWEEPS §4).
- `sparsegf2 sweep replay DIR n beta p u` (`sweep.verify`) recomputes a row and compares
  everything stored, including snapshot digests.
- A new probe or an unsaved depth needs a replay from layer 0. The stream is sequential per
  trajectory, so a replay to depth $T$ costs $T$ layers.
- Sampled probes draw from streams keyed by $(u,\text{probe name},\text{layer})$, so the same name
  and sample count give the same draws.
- Growing a sweep (graphs, circuits, rates, sizes) never changes a saved trajectory. A later
  package version may grow it only after its replay check (SWEEPS §10).

## 11. Pitfalls

- Engine values at $k=0$: `contiguous_distance` NaN, `erasure_first_failure` $n+1$,
  `correlated_erasure` 0 (so $D=n$ if summed). Convert to v1.4's $d^0=0$ and $D=0$ (§6).
- Pooling rows when $C_g$ differ (§2.3). Treating circuits or checkpoints as independent for
  errors (§2.5).
- Averaging only the observed τ (§5.5). Taking the mean curve's peak for $\mathbb E T^*$ (§7.3).
- At $\beta=0$ all graphs are equal, so $\sigma_b^2=0$. §2 still holds: the graph means are then
  i.i.d. circuit averages.

## 12. Checks (synthetic data; they run in `tests/test_docs.py`)

```python
import numpy as np

rng = np.random.default_rng(0)

# 1. the nested estimator, the pooled mean and the graph-cluster error (section 2)
G = 40
C = rng.integers(1, 6, size=G)                  # unequal circuits per graph (an interrupted run)
m_true = rng.normal(10.0, 2.0, size=G)          # E[F | G_g]
rows_g = np.repeat(np.arange(G), C)
F = m_true[rows_g] + rng.normal(0.0, 3.0, size=rows_g.size)
graph_means = np.array([F[rows_g == g].mean() for g in range(G)])
nested = graph_means.mean()                     # equal weight per graph
pooled = F.mean()                               # weights graph g by C_g
assert np.isclose(pooled, np.sum(C * graph_means) / C.sum())
se = graph_means.std(ddof=1) / np.sqrt(G)

# 2. first passages: the event identity {tau_ell > t} = {k_t > ell} (section 5.2)
n, T_max, rows = 12, 1000, 50
t = np.arange(T_max + 1)
drops = np.concatenate([np.zeros((rows, 1), int), (rng.random((rows, T_max)) < 0.01)], axis=1)
k = n - np.minimum(np.cumsum(drops, axis=1), n)  # a nonincreasing k_t, t = 0..T_max
for ell in range(n + 1):
    reached = k <= ell
    tau_ell = np.where(reached.any(axis=1), reached.argmax(axis=1), -1)   # -1: censored
    above = (tau_ell[:, None] < 0) | (tau_ell[:, None] > t[None, :])
    assert np.array_equal(above, k > ell)

# 3. survival and the restricted mean (section 5.5)
tau = rng.geometric(1 / 400, size=500)
tau_obs = np.where(tau > T_max, -1, tau)        # what a run stores
S = ((tau_obs[:, None] < 0) | (tau_obs[:, None] > t[None, :])).mean(axis=0)   # P(tau > t)
restricted = np.where(tau_obs < 0, T_max, tau_obs).mean()                      # E[min(tau, T_max)]
assert np.isclose(restricted, S[:T_max].sum())
assert tau_obs[tau_obs >= 0].mean() < restricted  # observed-only mean: biased low

# 4. the conditional distance as a ratio of nested means (section 6.3)
G, C = 30, 4
h = rng.random((G, C)) < 0.7
d = np.where(h, rng.integers(1, 6, size=(G, C)), np.nan)   # the engine: NaN without a code
d0 = np.nan_to_num(d)                                       # zero extension
mu_d0, mu_h = d0.mean(axis=1).mean(), h.mean(axis=1).mean()
ratio = mu_d0 / mu_h
assert np.isclose(ratio, np.nanmean(d))                     # equal C: the mean over rows with a code
e = d0.mean(axis=1) - ratio * h.mean(axis=1)
se_ratio = np.sqrt(np.sum(e**2) / (G * (G - 1))) / mu_h

# 5. the engine's k = 0 values converted to v1.4's conventions (sections 6.4, 6.5)
n = 8
k_t = np.array([3, 0])
M = np.array([[2, 3, 2, 4], [n + 1] * 4])                   # erasure_first_failure
D0 = np.where(k_t > 0, M.mean(axis=1), 0.0)
assert D0.tolist() == [2.75, 0.0]
P = np.array([[0.0, 0.1, 0.5, 0.9, 1.0, 1.0, 1.0], [0.0] * 7])  # correlated_erasure, m = 1..n-1
DJ = np.where(k_t > 0, 1 + (1 - P).sum(axis=1), 0.0)
assert np.allclose(DJ, [3.5, 0.0]) and 1 + (1 - P[1]).sum() == n   # unconverted: D = n


# 6. T*: the log-midpoint of the maximizing checkpoints; E[T*] is not T*(E[d]) (section 7)
def t_star(grid, trace):
    pos = grid > 0
    top = trace[pos].max()
    if top == 0:
        return None
    return float(np.exp(np.log(grid[pos][trace[pos] == top]).mean()))


grid = np.array([0, 1, 2, 4, 8, 16, 32])
a = np.array([1, 2, 2, 3, 3, 1, 0])             # a plateau at t = 4 and 8: T* = sqrt(32)
b = np.array([1, 1, 1, 2, 2, 4, 1])             # a peak at t = 16
assert np.isclose(t_star(grid, a), np.sqrt(32)) and np.isclose(t_star(grid, b), 16)
mean_of_t_star = (t_star(grid, a) + t_star(grid, b)) / 2       # 10.83
t_star_of_mean = t_star(grid, (a + b) / 2)                     # mean curve: plateau at 4, 8, 16
assert np.isclose(t_star_of_mean, 8.0) and not np.isclose(mean_of_t_star, t_star_of_mean)
```
