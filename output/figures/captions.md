# Figure captions

Lifted verbatim by the manuscript; every number quoted below is read from
`evidence/` by `scripts/make_figures.py`.  Common design: `X ~ N(1,1)`,
`beta0 = 1`, regression error `eps` centred gamma with `m2 = 1` (or `N(0,1)`),
measurement error `U` centred gamma, reliability
`lambda = sigma_x^2 / (sigma_x^2 + nu_2)`; `M = 2000` Monte Carlo replications
per cell, common random numbers across estimators within a replication.

## fig1_are_theory_vs_mc.pdf

**Asymptotic relative efficiency against corrected least squares, as a function
of the reliability.** Lines are the closed-form ARE of eq. (4.1) evaluated on
the theoretical grid `lambda in [0.600, 0.999]`; points are Monte Carlo
efficiencies measured at `n = 1000`, `M = 2000`, with paired bootstrap 95%
intervals (B = 2000, resampling unit = replication). Rows are the two
measurement-error designs -- the Woolsey--Huang gamma, whose shape
`4(1-lambda)/lambda` ties the shape of `U` to the reliability, and a gamma of
fixed shape `k = 4` that decouples them -- and columns are the two signs of the
slope; the regression error is skewed gamma with `c_3 = 1.0`, so
`g_2 = 0.7143`. The horizontal solid line is break-even against TC-LS and the
dotted line is `g_2`, the variance reduction PMM2 would deliver with no
measurement error. Transplanting the PMM2 weight ratio into the EIV score
(dash-dot) sends the ARE **above** break-even over a wide band of realistic
reliabilities -- 1.597 at `lambda = 0.80` in the Woolsey--Huang design -- so the
"improved" estimator is up to 60% *less* efficient than the corrected least
squares it was meant to improve. Re-optimising the ratio (solid) keeps the ARE
at or below one everywhere, as the Cauchy--Schwarz bound
`ARE* = 1 - Corr^2(Phi_L, G) <= 1` requires. Simulation reproduces the
closed form in both directions, including the predicted *loss*: the agreement
of the measured points with the theoretical curve on the losing branch is
evidence that the implementation and the algebra agree, since a coding error
would have to reproduce a penalty nobody wanted. In the two `beta_1 = -1`
panels the measured efficiencies sit slightly above the closed-form curve at
the higher reliabilities. This is the finite-sample deviation quantified by
verification gate G2, which requires agreement within two Monte Carlo standard
errors and records 11 exceedances out of 144 sub-checks (largest `|z| = 3.14`,
five of them at `n = 250` against two at `n = 1000`); panel (b) shows the same
estimator at `n = 250`, 500 and 1000, and the offset contracts towards the
curve as the sample grows. The closed form is an asymptotic quantity and is
quoted as such: it should not be read as a prediction of finite-sample MSE at
`n <= 500`.

## fig2_bias_mse_iqr.pdf

**Bias, mean squared error and interquartile range against reliability, in the
layout of Woolsey--Huang (their Figures 1--2).** Rows are the intercept and the
slope, columns the three summaries; `U` is the Woolsey--Huang gamma, of shape
`4(1-lambda)/lambda` and scale `1/2`, so that the shape of the measurement
error is itself tied to the reliability; the regression error has
`c_3 = 1.0`, `beta_1 = +1`, `n = 1000`, `M = 2000`. Five
estimators are shown: uncorrected least squares, corrected least squares
(TC-LS), TC-PMM2 with the transplanted PMM2 weight, TC-PMM2* with the
re-optimised weight, and least squares on the unobserved true regressor as an
error-free reference. MSE is on a logarithmic scale. Uncorrected least squares
carries the textbook attenuation, `bias(beta_1) = -(1-lambda)beta_1`, which the
tessarine-debiased correction removes at every reliability for all three
corrected estimators; the three corrected curves are indistinguishable in the
bias panels and separate only in dispersion. The MSE and IQR panels are where
the two weightings part: at `lambda = 0.80` the transplanted weight sits above
TC-LS, the re-optimised weight below it, and both approach the error-free
reference as `lambda -> 1`.

## fig3_weight_ratio.pdf

**Why the transplanted weights fail: the optimal weight ratio changes sign.**
The ARE of eq. (4.1) is a quadratic in the single weight ratio `t = b/a`,
minimised at `t_opt = E[Phi_L G] / E[G^2]`. Curves are the closed-form
`t_opt(lambda)` for the two slope signs; the dash-dot horizontal line is the
value PMM2 transplants, `t = m_3/(m_4 - m_2^2) = 0.2857` for `c_3 = 1.0`;
markers are the mean weight actually used by TC-PMM2* over `M = 2000`
replications at `n = 1000`. The left panel is the Woolsey--Huang measurement
error, a centred gamma of shape `4(1-lambda)/lambda` whose skewness therefore
grows as the reliability improves; the right panel holds the shape fixed at
`k = 4`. The shaded band marks reliabilities at which
`t_opt` is negative. In the Woolsey--Huang design at `lambda = 0.80` the
optimum is `t_opt = -0.0610` while PMM2 supplies `+0.2857`: the transplanted
weight is not merely off-optimum, it is on the **wrong side of zero**. The
skewness of the measurement error enters the working residual as `-beta_1 U`
and inverts the sign of the skewness signal `E[Phi_L G]` that PMM2 exploits, so
the transplanted weight actively works against the estimator. For
`beta_1 = -1` the same mechanism adds to the signal instead, `t_opt` stays near
the transplanted value, and the fixed weights do no harm -- which is why the
failure is invisible unless both signs are simulated.

## fig4_sensitivity.pdf

**Robustness to misspecified measurement-error moments.** Each moment
`nu_2, ..., nu_6` supplied to the estimators is multiplied by an independent
`Uniform[1-delta, 1+delta]` factor; `delta = 0` is the correctly specified case.
Rows are the two `U` designs at reliability 0.80 and 0.95; columns are bias,
MSE and IQR of the slope (`c_3 = 1.0`, `beta_1 = +1`, `n = 1000`, `M = 2000`).
All three corrected estimators degrade together: the correction itself needs
`nu_2`, so bias returns as `delta` grows, and MSE rises by roughly a factor of
five at `delta = 0.6`. The point of the figure is the *absence* of a gap.
TC-PMM2* consumes two more moments of `U` (`nu_5` and `nu_6`) than TC-LS, yet
its curves track TC-LS rather than diverging from it, so the extra moment
requirement is not an extra fragility. Its efficiency advantage is preserved
under misspecification, and the single cell where it falls marginally behind
TC-LS lies at the extreme `delta = 0.6`, where the optimality guarantee -- which
is proved for correctly specified `U` moments -- no longer applies.

## fig5_gaussian_eps.pdf

**Gaussian regression error: the classical method has nothing to exploit, the
EIV version still does.** With `m_3 = 0` the classical PMM2 weight is exactly
zero, `g_2 = 1`, and the fixed-weight TC-PMM2 collapses identically onto TC-LS
-- its closed-form ARE is identically one, so its dash-dot line *is* the
break-even level (no separate reference rule is drawn), and every Monte Carlo
point for it is exactly 1.0000, in all 36 Gaussian cells. TC-PMM2* nonetheless
gains: the closed-form ARE* falls to 0.8125 at `lambda = 0.80` in the
Woolsey--Huang design (a centred gamma of shape `4(1-lambda)/lambda`; the
right panel holds the shape fixed at `k = 4`), reproduced by simulation as
0.8085 with bootstrap interval
[0.779, 0.839]. The correlation between the linear and quadratic parts of the
score is generated here by the skewness of the *measurement* error rather than
of the regression error, through the term
`E[Phi_L G] = -beta_1^3 (nu_3 sigma_x^2 + nu_5 - 4 nu_2 nu_3) - 2 beta_1 m_2 nu_3`,
which is non-zero even when `m_3 = 0`. This is a source of efficiency with no
counterpart in the error-free theory; filled and open markers are the two slope
signs, whose theoretical ARE* coincide exactly.

## fig6_lognormal.pdf

**Heavy-tailed skewed measurement error: the transplanted weights do not lose,
they fail.** Lognormal `U` with `sigma = 1` (skewness 6.18, excess kurtosis
110.9) -- the regime in which the tessarine system (13) of Woolsey and Huang
admits no real solution, so their construction degrades to an approximate
minimisation while the closed-form corrected score remains exactly defined.
Columns are the two slope signs; `c_3(eps) = 1.0`, `n = 1000`, `M = 2000`.
Top row: the share of replications in which each estimator converged, together
with the rate at which the cumulant-deconvolution guard clipped -- a per-cell
diagnostic shared by both adaptive variants, reaching roughly 78% in the cells
drawn here and 82.1% in the worst cell of the design (`n = 250`). Corrected least
squares converged in every replication of every cell and is omitted from the
top row. Middle row: bias of the slope. Bottom row: the Monte Carlo MSE
ratio against corrected least squares with paired bootstrap 95% intervals, on a
**logarithmic** ordinate; the plotted quantity is a ratio, and only a log scale
places "twice as good" and "twice as bad" at equal distances from the
break-even line, so neither estimator is flattered by the choice.

With the transplanted weight the estimator stops being usable rather than
merely inefficient: convergence falls to 97.7% at `lambda = 0.80` (and to 81.0%
at `n = 250`), the bias reaches `+0.130` where corrected least squares carries
`+0.009`, and the MSE ratio reaches 13.7. Gate G6 records this as 16 of 36
cells passed. The re-optimised weight remains defined in 99.95% of replications,
holds a bias of `+0.0067`, passes 36 of 36 cells, and attains an MSE ratio of
0.78.

Two qualifications belong with the figure and are visible in it. First, the
adaptive -- that is, feasible -- branch of the transplanted-weight estimator is
destroyed here: the deconvolution guard clips in most replications, convergence
falls to 40.8%, and the MSE ratio degrades to 21.6, worse than its own oracle;
the adaptive branch of TC-PMM2* tracks its oracle despite the same clipping,
but the result that stands in this regime is the oracle one. Second, the
advantage of TC-PMM2* is not universal even here: at `lambda = 0.95` with
`beta_1 = +1` its measured ratio is 1.02 with bootstrap interval [1.006, 1.037]
under `c_3 = 1.5`, that is, slightly *worse* than corrected least squares, even
though the asymptotic guarantee `ARE* <= 1` holds for these cells (0.973). The
claim this figure supports is that the transplanted weights fail in this
regime, not that the re-optimised estimator dominates everywhere in it.

## fig7_application.pdf

**The NHANES case study in three views.** NHANES 2013-2014, `n = 5731` subjects
in 30 primary sampling units nested in 15 strata; every interval comes from a
stratified cluster bootstrap over PSUs within strata (`B = 2000`), and the point
estimates are unweighted. **(a)** The within-subject deviations
`d_ij = W_ij - Wbar_i`, whose law identifies the moments of the measurement
error, drawn against their own reflection about zero on a logarithmic density
scale; the left tail is the heavier one, and the recovered standardised skewness
is `gamma_3(U) = -0.371` with cluster-bootstrap interval [-0.626, -0.097]. The
deviations live on multiples of 2/3 mmHg because the readings are recorded as
even integers. **(b)** The closed-form efficiency as a quadratic in the single
weight ratio, evaluated at the recovered moments. The transplanted value
`t_PMM2 = 0.3765` (open circle) and the optimum `t_opt = 0.3697` (filled
diamond) are almost the same point, so the negative result of the paper is
inactive in this regime; the dotted rules are `g_2 = 0.6956` and the minimum
`g_2^EIV = 0.7066`, which differ by less than the sampling uncertainty of
either because the signal ratio `beta_1^2 sigma_x^2 / m_2` is 0.079, and the
shaded band is the observed variance ratio 0.546 with its delete-one-stratum
jackknife interval [0.255, 0.837] -- reported, but not interpretable as an
efficiency, because the two estimators being compared fail an
over-identification contrast at z = 4.18. **(c)** The two conditional-moment
assumptions, each tested against the pressure level and each refuted: the measurement-error variance `nu_2`, estimated from readings 2 and 3
alone and therefore statistically independent of the level proxy on the
abscissa, falls by a factor of two across quintiles of the first reading
(`t = -4.74`), while the variance `m_2` of the naive regression residual rises
(`t = 3.30`). Corrected least squares tolerates both; the degree-two score does
not.
