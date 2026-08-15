"""Estimators for the linear EIV model with doubly non-Gaussian errors (SPEC 3).

Nine fits: the six of SPEC 3.3 plus the two TC-PMM2* variants added by the
2026-08-14 coordinator amendment.

  1. naive-LS              OLS of Y on (1, W)                  -- lower bound
  2. naive-PMM2            PMM2 of Y on (1, W), empirical moments -- PMM alone does not fix EIV bias
  3. TC-LS                 corrected LS / Fuller               -- main competitor
  4. TC-PMM2-oracle        SPEC 3.2, true m2,m3,m4 of eps      -- FIXED (transplanted) weights
  5. TC-PMM2-adaptive      SPEC 3.2, moments from the cumulant-deconvolution lemma (SPEC 3.4)
  6. TC-PMM2*-oracle       weight ratio re-optimised IN the EIV context (primary estimator)
  7. TC-PMM2*-adaptive     same, with deconvolved moments
  8/9. reference           LS and PMM2 on the error-free X     -- unattainable benchmark

Estimators 4-5 transplant Kunchenko's PMM2 weights t = m3/(m4-m2^2), which are
optimal only when U vanishes; in an EIV context that choice is off-optimum and
CAN LOSE to TC-LS (ARE > 1).  Estimators 6-7 minimise the same quadratic in t
and therefore satisfy ARE = 1 - Corr^2(Phi_L, G) <= 1 by Cauchy-Schwarz --
this is g2_EIV, the correct EIV analogue of Kunchenko's g2.

TC-PMM2 is implemented in the CLOSED FORM of SPEC 3.2 (moment-corrected powers
of Wt), NOT with tessarine arithmetic.  The closed form was re-derived
independently of the SPEC before implementation and agrees with it exactly:

    t1 = Wt,  t2 = Wt^2 - nu2,  t3 = Wt^3 - 3*nu2*Wt - nu3
    (defined by  E[t_k(Wt) | X] = X^k )

    Psi0 = a*(A - b1*t1) - m3*(A^2 - 2*b1*A*t1 + b1^2*t2 - m2)
    Psi1 = a*(A*t1 - b1*t2) - m3*(A^2*t1 - 2*b1*A*t2 + b1^2*t3 - m2*t1)
    with a = m4 - m2^2 and A = Y - b0.

Setting nu2 = nu3 = 0 reduces Psi to the plain PMM2 score
s2(e) = (m4-m2^2)*e - m3*(e^2-m2), which is the EstemPMM oracle (gate G0).
Plain PMM2 is therefore never re-implemented separately: it is the nu = 0
special case of the new estimator, and G0 checks it against EstemPMM.

Standard errors are empirical sandwiches V = M^-1 Q M^-T / n (SPEC 4, Thm 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NEWTON_MAX_ITER = 100
NEWTON_TOL = 1e-11
G_MIN = 0.05          # adaptive-weight clip: enforce g2 = 1 - c3^2/(2+c4) >= G_MIN
MIN_M2 = 1e-8


@dataclass
class Fit:
    name: str
    beta: np.ndarray                 # (b0, b1)
    se: np.ndarray                   # sandwich standard errors
    converged: bool = True
    iterations: int = 0
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _sandwich(psi: np.ndarray, jac: np.ndarray) -> np.ndarray:
    """psi: (n,2) score rows;  jac: (2,2) SUM of d psi_r / d beta^T.

    M = -(1/n) * jac,  Q = (1/n) psi^T psi,  V = M^-1 Q M^-T / n."""
    n = psi.shape[0]
    m_mat = -jac / n
    q_mat = psi.T @ psi / n
    try:
        m_inv = np.linalg.inv(m_mat)
    except np.linalg.LinAlgError:
        return np.full(2, np.nan)
    v = m_inv @ q_mat @ m_inv.T / n
    d = np.diag(v)
    return np.sqrt(np.where(d > 0, d, np.nan))


def central_moments(e: np.ndarray) -> tuple[float, float, float]:
    """Empirical central moments about the sample mean is NOT used: PMM works
    with moments about zero of the residuals, matching EstemPMM::compute_moments
    (mean(e^2), mean(e^3), mean(e^4))."""
    return float(np.mean(e ** 2)), float(np.mean(e ** 3)), float(np.mean(e ** 4))


def pmm_weights_valid(m2: float, m3: float, m4: float) -> bool:
    return m2 > MIN_M2 and (m4 - m2 ** 2) > MIN_M2 and (m2 * (m4 - m2 ** 2) - m3 ** 2) > 0.0


# --------------------------------------------------------------------------
# least squares family
# --------------------------------------------------------------------------
def fit_ls(y: np.ndarray, z: np.ndarray, name: str) -> Fit:
    """OLS of y on (1, z) with a heteroskedasticity-robust sandwich."""
    n = y.size
    des = np.column_stack([np.ones(n), z])
    beta, *_ = np.linalg.lstsq(des, y, rcond=None)
    resid = y - des @ beta
    psi = des * resid[:, None]
    jac = -des.T @ des
    return Fit(name, beta, _sandwich(psi, jac), True, 1)


def fit_tc_ls(y: np.ndarray, wt: np.ndarray, nu2: float, name: str = 'tc_ls') -> Fit:
    """Corrected LS (Fuller).  Score  Psi = (1, Wt)^T (Y - b0 - b1 Wt) + (0, b1 nu2)^T."""
    n = y.size
    sw, sw2 = wt.sum(), (wt ** 2).sum()
    s_mat = np.array([[float(n), sw], [sw, sw2 - n * nu2]])
    rhs = np.array([y.sum(), float(wt @ y)])
    try:
        beta = np.linalg.solve(s_mat, rhs)
    except np.linalg.LinAlgError:
        return Fit(name, np.full(2, np.nan), np.full(2, np.nan), False, 0)
    resid = y - beta[0] - beta[1] * wt
    psi = np.column_stack([resid, wt * resid + beta[1] * nu2])
    jac = np.array([[-float(n), -sw], [-sw, -(sw2 - n * nu2)]])
    return Fit(name, beta, _sandwich(psi, jac), True, 1)


# --------------------------------------------------------------------------
# TC-PMM2 (closed form, SPEC 3.2)
# --------------------------------------------------------------------------
def _tc_psi_jac(beta, y, wt, m2, m3, m4, nu2, nu3, weights=None):
    """Score rows (n,2) and the SUM of their Jacobians (2,2).

    ``weights`` = (a, b) is the free weight pair of the corrected-score family

        Psi0 = a*(A - b1*t1) - b*(A^2 - 2*b1*A*t1 + b1^2*t2 - m2)
        Psi1 = a*(A*t1 - b1*t2) - b*(A^2*t1 - 2*b1*A*t2 + b1^2*t3 - m2*t1)

    Exact unbiasedness holds for EVERY (a, b) -- Theorem 1 constrains nothing
    here -- so the pair may be re-optimised inside the EIV context.
      * transplanted PMM2 weights (default): a = m4 - m2^2, b = m3
      * TC-PMM2*:                            a = 1,         b = t_opt
    Only the ratio t = b/a matters (the score is scale-invariant)."""
    b0, b1 = beta
    a, m3 = (m4 - m2 ** 2, m3) if weights is None else weights
    aa = y - b0                                   # A
    t1 = wt
    t2 = wt ** 2 - nu2
    t3 = wt ** 3 - 3.0 * nu2 * wt - nu3

    psi0 = a * (aa - b1 * t1) - m3 * (aa ** 2 - 2.0 * b1 * aa * t1 + b1 ** 2 * t2 - m2)
    psi1 = (a * (aa * t1 - b1 * t2)
            - m3 * (aa ** 2 * t1 - 2.0 * b1 * aa * t2 + b1 ** 2 * t3 - m2 * t1))
    psi = np.column_stack([psi0, psi1])

    # dA/db0 = -1, dA/db1 = 0
    d00 = -a + 2.0 * m3 * (aa - b1 * t1)
    d01 = -a * t1 - m3 * (-2.0 * aa * t1 + 2.0 * b1 * t2)
    d10 = -a * t1 - m3 * (-2.0 * aa * t1 + 2.0 * b1 * t2)
    d11 = -a * t2 - m3 * (-2.0 * aa * t2 + 2.0 * b1 * t3)
    jac = np.array([[d00.sum(), d01.sum()], [d10.sum(), d11.sum()]])
    return psi, jac


def fit_tc_pmm2(y, wt, m2, m3, m4, nu2, nu3, beta_init, name='tc_pmm2', info=None,
                weights=None) -> Fit:
    """Newton-Raphson on the closed-form corrected PMM2 score, started at TC-LS.

    Step-halving line search on ||sum psi|| guards against the second root of
    the (quadratic in b1) system."""
    beta = np.asarray(beta_init, float).copy()
    if not np.all(np.isfinite(beta)):
        return Fit(name, np.full(2, np.nan), np.full(2, np.nan), False, 0, dict(info or {}))
    converged = False
    it = 0
    psi, jac = _tc_psi_jac(beta, y, wt, m2, m3, m4, nu2, nu3, weights)
    g = psi.sum(axis=0)
    for it in range(1, NEWTON_MAX_ITER + 1):
        if abs(np.linalg.det(jac)) < 1e-12 * max(1.0, np.abs(jac).max() ** 2):
            break
        try:
            step = np.linalg.solve(jac, g)
        except np.linalg.LinAlgError:
            break
        norm0 = np.linalg.norm(g)
        ok = False
        t = 1.0
        for _ in range(30):                       # step halving
            cand = beta - t * step
            psi_c, jac_c = _tc_psi_jac(cand, y, wt, m2, m3, m4, nu2, nu3, weights)
            g_c = psi_c.sum(axis=0)
            if np.all(np.isfinite(g_c)) and np.linalg.norm(g_c) < norm0:
                ok = True
                break
            t *= 0.5
        if not ok:                                # cannot decrease: stationary
            converged = np.linalg.norm(g) / max(1, y.size) < 1e-8
            break
        move = np.linalg.norm(cand - beta)
        beta, psi, jac, g = cand, psi_c, jac_c, g_c
        if move < NEWTON_TOL * (1.0 + np.linalg.norm(beta)):
            converged = True
            break
    se = _sandwich(psi, jac) if converged else np.full(2, np.nan)
    return Fit(name, beta, se, converged, it, dict(info or {}))


# --------------------------------------------------------------------------
# adaptive weights: cumulant deconvolution (SPEC 3.4)
# --------------------------------------------------------------------------
def deconvolve_eps_moments(resid, b1_hat, nu2, nu3, nu4):
    """kappa_j(eps) = kappa_j(e_obs) - (-b1)^j kappa_j(U),  e_obs = eps - b1*(U - EU).

    Returns (m2, m3, m4, flags) with a documented deterministic fallback:
      * m2 <= 0                       -> TC-LS weights (m3 = 0)          [flag 'fallback']
      * implied g2 < G_MIN (incl. m4 - m2^2 <= 0) -> clip c4 to the
        boundary c4 = c3^2/(1-G_MIN) - 2                                  [flag 'clip']
    Both flags are recorded per replication as diagnostics, never silently masked."""
    k2_e, k3_e, k4_e = _cumulants_from_residuals(resid)
    k2_u, k3_u, k4_u = nu2, nu3, nu4 - 3.0 * nu2 ** 2
    k2 = k2_e - b1_hat ** 2 * k2_u
    k3 = k3_e + b1_hat ** 3 * k3_u
    k4 = k4_e - b1_hat ** 4 * k4_u

    fallback = clip = 0
    if not np.isfinite(k2) or k2 <= MIN_M2:
        fallback = 1
        m2 = k2_e if k2_e > MIN_M2 else 1.0
        return m2, 0.0, 3.0 * m2 ** 2, dict(adapt_fallback=1, adapt_clip=0)

    m2 = k2
    m3 = k3 if np.isfinite(k3) else 0.0
    m4 = (k4 if np.isfinite(k4) else 0.0) + 3.0 * m2 ** 2
    c3 = m3 / m2 ** 1.5
    c4 = m4 / m2 ** 2 - 3.0
    g2 = 1.0 - c3 ** 2 / (2.0 + c4) if (2.0 + c4) > 0 else -np.inf
    if not np.isfinite(g2) or g2 < G_MIN:
        clip = 1
        c4 = c3 ** 2 / (1.0 - G_MIN) - 2.0
        m4 = (c4 + 3.0) * m2 ** 2
    return m2, m3, m4, dict(adapt_fallback=fallback, adapt_clip=clip)


def _cumulants_from_residuals(e):
    """Cumulants of the naive residual, from moments about zero (its mean is
    ~0 by construction of the fitted model)."""
    mu2, mu3, mu4 = central_moments(e)
    return mu2, mu3, mu4 - 3.0 * mu2 ** 2


# --------------------------------------------------------------------------
# TC-PMM2*: weight ratio re-optimised inside the EIV context
# --------------------------------------------------------------------------
def are_blocks(sx2, b1, m2, m3, m4, nu2, nu3, nu4, nu5, nu6):
    """The three Proposition-3 building blocks (theory/ARE_derivation.md 3.1-3.3):

        E[Phi_L^2] = m2(sx2+nu2) + b1^2 (nu2 sx2 + nu4 - nu2^2)
        E[Phi_L G] = m3(sx2+nu2) - b1^3 (nu3 sx2 + nu5 - 4 nu2 nu3) - 2 b1 m2 nu3
        E[G^2]     = (m4-m2^2)(sx2+nu2) + 4 b1^2 m2 (nu4 - nu2^2 + nu2 sx2)
                     - 4 b1 m3 nu3
                     + b1^4 (nu6 + 9 nu2^3 - 6 nu2 nu4 - nu3^2 - nu2^2 sx2 + nu4 sx2)

    Typed here independently of theory/are_formulas.py; verify.py asserts the
    two implementations agree (cross-check C1)."""
    ell = m2 * (sx2 + nu2) + b1 ** 2 * (nu2 * sx2 + nu4 - nu2 ** 2)
    elg = (m3 * (sx2 + nu2)
           - b1 ** 3 * (nu3 * sx2 + nu5 - 4.0 * nu2 * nu3)
           - 2.0 * b1 * m2 * nu3)
    egg = ((m4 - m2 ** 2) * (sx2 + nu2)
           + 4.0 * b1 ** 2 * m2 * (nu4 - nu2 ** 2 + nu2 * sx2)
           - 4.0 * b1 * m3 * nu3
           + b1 ** 4 * (nu6 + 9.0 * nu2 ** 3 - 6.0 * nu2 * nu4 - nu3 ** 2
                        - nu2 ** 2 * sx2 + nu4 * sx2))
    return ell, elg, egg


def are_of_t(t, sx2, b1, m2, m3, m4, nu2, nu3, nu4, nu5, nu6):
    """ARE(t) = 1 - 2t E[Phi_L G]/E[Phi_L^2] + t^2 E[G^2]/E[Phi_L^2]   (eq. 4.1)."""
    ell, elg, egg = are_blocks(sx2, b1, m2, m3, m4, nu2, nu3, nu4, nu5, nu6)
    if not np.isfinite(ell) or ell <= 0:
        return np.nan
    return 1.0 - 2.0 * t * elg / ell + t ** 2 * egg / ell


def optimal_t(sx2, b1, m2, m3, m4, nu2, nu3, nu4, nu5, nu6):
    """t_opt = E[Phi_L G]/E[G^2]  and  g2_EIV = 1 - Corr^2(Phi_L, G) <= 1.

    Falls back to t = 0 (i.e. exactly TC-LS) when E[G^2] is not usable; the
    fallback is recorded, never silent."""
    ell, elg, egg = are_blocks(sx2, b1, m2, m3, m4, nu2, nu3, nu4, nu5, nu6)
    if not (np.isfinite(ell) and np.isfinite(elg) and np.isfinite(egg)) \
            or egg <= MIN_M2 or ell <= MIN_M2:
        return 0.0, 1.0, 1
    t = elg / egg
    return float(t), float(1.0 - elg ** 2 / (ell * egg)), 0


def fit_tc_pmm2_star(y, wt, m2, m3, m4, nu, sx2, beta_init, name, info=None) -> Fit:
    """Corrected score with the EIV-optimal weight ratio.  b1 inside t_opt is
    plugged in from the TC-LS first step (root-n consistent), exactly as SPEC
    3.4 plugs it into the cumulant deconvolution."""
    t, g2_eiv, fb = optimal_t(sx2, beta_init[1], m2, m3, m4,
                              nu['nu2'], nu['nu3'], nu['nu4'], nu['nu5'], nu['nu6'])
    inf = dict(info or {})
    inf.update(star_fallback=fb, star_t=t, star_g2_eiv=g2_eiv)
    return fit_tc_pmm2(y, wt, m2, m3, m4, nu['nu2'], nu['nu3'], beta_init,
                       name, inf, weights=(1.0, t))


# --------------------------------------------------------------------------
# full estimator suite for one replication (CRN: all see the same data)
# --------------------------------------------------------------------------
ESTIMATOR_NAMES = ('naive_ls', 'naive_pmm2', 'tc_ls',
                   'tc_pmm2_oracle', 'tc_pmm2_adaptive',
                   'tc_pmm2star_oracle', 'tc_pmm2star_adaptive',
                   'ref_ls', 'ref_pmm2')


def fit_all(y, w, x_true, eps_moments, nu_assumed, sx2_true=1.0) -> dict[str, Fit]:
    """eps_moments: dict(m2,m3,m4) TRUE moments of eps (oracle / reference).
    nu_assumed: dict(nu2..nu6) moments of U handed to the estimators
    (true in the oracle branch, perturbed in the sensitivity branch).
    sx2_true: var(X), used by the oracle TC-PMM2* weight; the adaptive variant
    estimates it as var(Wt) - nu2."""
    m2, m3, m4 = eps_moments['m2'], eps_moments['m3'], eps_moments['m4']
    nu2, nu3, nu4 = nu_assumed['nu2'], nu_assumed['nu3'], nu_assumed['nu4']
    out: dict[str, Fit] = {}

    # 1. naive LS on the mismeasured covariate
    f_naive = fit_ls(y, w, 'naive_ls')
    out['naive_ls'] = f_naive

    # 2. naive PMM2: empirical moments of the naive residuals, nu = 0
    r = y - f_naive.beta[0] - f_naive.beta[1] * w
    n2, n3, n4 = central_moments(r)
    if pmm_weights_valid(n2, n3, n4):
        out['naive_pmm2'] = fit_tc_pmm2(y, w, n2, n3, n4, 0.0, 0.0,
                                        f_naive.beta, 'naive_pmm2')
    else:
        out['naive_pmm2'] = Fit('naive_pmm2', f_naive.beta.copy(), f_naive.se.copy(),
                                False, 0, dict(weights_invalid=1))

    # 3. TC-LS (Fuller) -- the main competitor and the Newton start
    f_tcls = fit_tc_ls(y, w, nu2)
    out['tc_ls'] = f_tcls

    # 4. TC-PMM2 oracle
    out['tc_pmm2_oracle'] = fit_tc_pmm2(y, w, m2, m3, m4, nu2, nu3,
                                        f_tcls.beta, 'tc_pmm2_oracle')

    # 5. TC-PMM2 adaptive (cumulant deconvolution off the TC-LS fit)
    r_tc = y - f_tcls.beta[0] - f_tcls.beta[1] * w
    am2, am3, am4, flags = deconvolve_eps_moments(r_tc, f_tcls.beta[1], nu2, nu3, nu4)
    out['tc_pmm2_adaptive'] = fit_tc_pmm2(y, w, am2, am3, am4, nu2, nu3,
                                          f_tcls.beta, 'tc_pmm2_adaptive', flags)

    # 5a/5b. TC-PMM2*: same corrected-score family, EIV-optimal weight ratio
    out['tc_pmm2star_oracle'] = fit_tc_pmm2_star(
        y, w, m2, m3, m4, nu_assumed, sx2_true, f_tcls.beta, 'tc_pmm2star_oracle')
    sx2_hat = max(float(np.var(w, ddof=1)) - nu2, MIN_M2)
    out['tc_pmm2star_adaptive'] = fit_tc_pmm2_star(
        y, w, am2, am3, am4, nu_assumed, sx2_hat, f_tcls.beta,
        'tc_pmm2star_adaptive', flags)

    # 6. references on the error-free data
    f_ref = fit_ls(y, x_true, 'ref_ls')
    out['ref_ls'] = f_ref
    out['ref_pmm2'] = fit_tc_pmm2(y, x_true, m2, m3, m4, 0.0, 0.0,
                                  f_ref.beta, 'ref_pmm2')
    return out


# --------------------------------------------------------------------------
# asymptotic ARE (Proposition 3): sandwich evaluated at the truth by a single
# very large sample -- used as a theoretical benchmark for gate G2, not as data.
# --------------------------------------------------------------------------
def theory_are(cell):
    """Closed-form asymptotic ARE for BOTH estimators in a cell (eq. 4.1).

    Returns the fixed-weight ARE (transplanted PMM2 t = m3/(m4-m2^2)) and the
    optimal ARE* = g2_EIV = 1 - Corr^2(Phi_L, G), which is <= 1 by
    Cauchy-Schwarz.  These are the per-cell targets of the revised gate G2."""
    from dgp import VAR_X
    em, um = cell.eps.moments, cell.u.moments
    args = (VAR_X, cell.b1, em['m2'], em['m3'], em['m4'],
            um['nu2'], um['nu3'], um['nu4'], um['nu5'], um['nu6'])
    t_fixed = em['m3'] / (em['m4'] - em['m2'] ** 2)
    t_opt, g2_eiv, fb = optimal_t(*args)
    ell, elg, egg = are_blocks(*args)
    return dict(are_fixed=float(are_of_t(t_fixed, *args)),
                are_star=float(g2_eiv), g2=em['g2'],
                t_fixed=float(t_fixed), t_opt=float(t_opt),
                e_phil2=float(ell), e_philg=float(elg), e_g2=float(egg),
                star_fallback=fb)


def lambda_star(eps_key, u_family, b1, lo=0.5, hi=0.99999, tol=1e-6):
    """Break-even reliability: the lam at which the FIXED-weight ARE crosses 1.

    Returns None when the fixed-weight estimator never loses (or never wins) on
    the bracket -- i.e. no crossing exists."""
    from dgp import make_cell

    def f(lam):
        return theory_are(make_cell(eps_key, u_family, lam, 1000, b1=b1))['are_fixed'] - 1.0

    flo, fhi = f(lo), f(hi)
    if max(abs(flo), abs(fhi)) < 1e-12:      # ARE == 1 identically (m3 = 0): no crossing
        return None
    if not np.isfinite(flo) or not np.isfinite(fhi) or flo * fhi > 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def asymptotic_are(cell, n_big=4_000_000, seed=7):
    """ARE(TC-PMM2 : TC-LS) for the slope, from M^-1 Q M^-T at Theta*."""
    from dgp import MEAN_X, VAR_X
    rng = np.random.default_rng(seed)
    x = MEAN_X + np.sqrt(VAR_X) * rng.standard_normal(n_big)
    eps = cell.eps.draw(n_big, rng)
    u = cell.u.draw(n_big, rng)
    y = cell.b0 + cell.b1 * x + eps
    wt = x + u
    em, um = cell.eps.moments, cell.u.moments
    beta = np.array([cell.b0, cell.b1])

    psi_p, jac_p = _tc_psi_jac(beta, y, wt, em['m2'], em['m3'], em['m4'],
                               um['nu2'], um['nu3'])
    v_p = _sandwich(psi_p, jac_p) ** 2 * n_big          # per-observation variance

    resid = y - beta[0] - beta[1] * wt
    psi_l = np.column_stack([resid, wt * resid + beta[1] * um['nu2']])
    des = np.column_stack([np.ones(n_big), wt])
    jac_l = -np.array([[float(n_big), wt.sum()],
                       [wt.sum(), (wt ** 2).sum() - n_big * um['nu2']]])
    v_l = _sandwich(psi_l, jac_l) ** 2 * n_big
    return dict(are_b1=float(v_p[1] / v_l[1]), are_b0=float(v_p[0] / v_l[0]),
                g2=em['g2'], n_big=n_big)


# --------------------------------------------------------------------------
# gate G0 fixture: TC-PMM2 with U == 0 must reproduce EstemPMM::lm_pmm2
# --------------------------------------------------------------------------
def emit_g0_fixture(outdir):
    """Write datasets + the TC-PMM2(nu=0) fit that scripts/oracle_check.R checks
    against EstemPMM::lm_pmm2.  The moments handed to TC-PMM2 are the ones
    EstemPMM uses internally: mean(e^k) of the OLS residuals, held fixed."""
    import csv as _csv
    import os as _os
    from dgp import EPS_SPECS, MEAN_X, VAR_X

    _os.makedirs(outdir, exist_ok=True)
    exp_rows = []
    for i, (ek, n) in enumerate([('g_c3_1.0', 200), ('g_c3_1.5', 500),
                                 ('g_c3_0.5', 1000), ('gauss', 300)]):
        rng = np.random.default_rng(4200 + i)
        x = MEAN_X + np.sqrt(VAR_X) * rng.standard_normal(n)
        y = 1.0 + 1.0 * x + EPS_SPECS[ek].draw(n, rng)
        with open(_os.path.join(outdir, f'g0_data_{i}.csv'), 'w', newline='') as fh:
            wr = _csv.writer(fh)
            wr.writerow(['y', 'x'])
            wr.writerows(zip(y, x))
        f_ols = fit_ls(y, x, 'ols')
        r = y - f_ols.beta[0] - f_ols.beta[1] * x
        m2, m3, m4 = central_moments(r)
        f = fit_tc_pmm2(y, x, m2, m3, m4, 0.0, 0.0, f_ols.beta, 'tc_pmm2_nu0')
        exp_rows.append(dict(dataset=i, eps_key=ek, n=n, m2=m2, m3=m3, m4=m4,
                             b0_py=f.beta[0], b1_py=f.beta[1],
                             converged=int(f.converged), iterations=f.iterations))
    with open(_os.path.join(outdir, 'g0_expected.csv'), 'w', newline='') as fh:
        wr = _csv.DictWriter(fh, fieldnames=list(exp_rows[0].keys()))
        wr.writeheader()
        wr.writerows(exp_rows)
    return exp_rows


if __name__ == '__main__':
    import sys as _sys
    if len(_sys.argv) > 2 and _sys.argv[1] == '--g0-fixture':
        for row in emit_g0_fixture(_sys.argv[2]):
            print(row)
    else:
        print('usage: estimators.py --g0-fixture <outdir>')
