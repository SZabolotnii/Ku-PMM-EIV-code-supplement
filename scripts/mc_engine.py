"""Monte Carlo replication loop, metrics and paired comparisons (SPEC 6.2).

Common random numbers: within one replication every estimator sees exactly the
same (Y, W, X), so estimator differences are paired.  MC replications are
genuine i.i.d. runs -- the one exception allowed by portfolio Verification
Standard 9 -- so the resampling unit is the replication and nothing else.
"""
from __future__ import annotations

import numpy as np

from dgp import Cell, draw_replication, assumed_u_moments
from estimators import ESTIMATOR_NAMES, fit_all

Z95 = 1.959963984540054
PAIRS = (
    ('tc_pmm2_oracle', 'tc_ls'),
    ('tc_pmm2_adaptive', 'tc_ls'),
    ('tc_pmm2_adaptive', 'tc_pmm2_oracle'),
    ('tc_pmm2star_oracle', 'tc_ls'),          # revised G2 / K1 headline
    ('tc_pmm2star_adaptive', 'tc_ls'),
    ('tc_pmm2star_adaptive', 'tc_pmm2star_oracle'),
    ('tc_pmm2star_oracle', 'tc_pmm2_oracle'),  # value added by re-optimising t
    ('naive_pmm2', 'naive_ls'),
    ('ref_pmm2', 'ref_ls'),
)


def run_cell(cell: Cell, n_rep: int) -> dict:
    """Returns raw per-replication estimates plus aggregated metrics."""
    eps_mom = cell.eps.moments
    truth = np.array([cell.b0, cell.b1])
    k = len(ESTIMATOR_NAMES)
    beta = np.full((n_rep, k, 2), np.nan)
    se = np.full((n_rep, k, 2), np.nan)
    conv = np.zeros((n_rep, k), dtype=bool)
    # adapt_fallback, adapt_clip, star_fallback(oracle), star_fallback(adaptive)
    flags = np.zeros((n_rep, 4), dtype=int)
    star_t = np.full((n_rep, 2), np.nan)         # t_opt used by oracle / adaptive

    from dgp import VAR_X
    for r in range(n_rep):
        d = draw_replication(cell, r)
        nu = assumed_u_moments(cell, r)
        fits = fit_all(d['y'], d['w'], d['x'], eps_mom, nu, sx2_true=VAR_X)
        for j, nm in enumerate(ESTIMATOR_NAMES):
            f = fits[nm]
            beta[r, j] = f.beta
            se[r, j] = f.se
            conv[r, j] = f.converged
        inf = fits['tc_pmm2_adaptive'].info
        so, sa = fits['tc_pmm2star_oracle'].info, fits['tc_pmm2star_adaptive'].info
        flags[r] = (inf.get('adapt_fallback', 0), inf.get('adapt_clip', 0),
                    so.get('star_fallback', 0), sa.get('star_fallback', 0))
        star_t[r] = (so.get('star_t', np.nan), sa.get('star_t', np.nan))

    return dict(cell=cell, n_rep=n_rep, beta=beta, se=se, conv=conv, flags=flags,
                star_t=star_t, truth=truth,
                summary=_summarise(cell, beta, se, conv, flags, star_t, truth, n_rep),
                paired=_paired(cell, beta, conv, truth, n_rep))


# --------------------------------------------------------------------------
def _summarise(cell, beta, se, conv, flags, star_t, truth, m):
    from estimators import theory_are
    th = theory_are(cell)
    rows = []
    for j, nm in enumerate(ESTIMATOR_NAMES):
        for p, pname in ((0, 'b0'), (1, 'b1')):
            est = beta[:, j, p]
            ok = np.isfinite(est)
            err = est - truth[p]
            sq = err ** 2
            s = se[:, j, p]
            cov_ok = np.isfinite(s) & ok
            covered = (np.abs(err) <= Z95 * s) & cov_ok
            n_cov = int(cov_ok.sum())
            p_cov = float(covered.sum() / n_cov) if n_cov else np.nan
            q1, q3 = (np.nanpercentile(est[ok], [25, 75]) if ok.any() else (np.nan, np.nan))
            rows.append(dict(
                cell_id=cell.cell_id, eps_key=cell.eps_key, u_family=cell.u_family,
                lam=cell.lam, n=cell.n, b1_true=cell.b1, delta=cell.delta,
                g2_theory=cell.eps.moments['g2'], m_rep=m,
                estimator=nm, param=pname,
                n_finite=int(ok.sum()),
                conv_rate=float(conv[:, j].mean()),
                bias=float(np.nanmean(err)),
                se_bias=float(np.nanstd(err, ddof=1) / np.sqrt(max(ok.sum(), 1))),
                mse=float(np.nanmean(sq)),
                se_mse=float(np.nanstd(sq, ddof=1) / np.sqrt(max(ok.sum(), 1))),
                rmse=float(np.sqrt(np.nanmean(sq))),
                var=float(np.nanvar(est, ddof=1)),
                iqr=float(q3 - q1),
                median_est=float(np.nanmedian(est)),
                coverage=p_cov,
                se_coverage=(float(np.sqrt(p_cov * (1 - p_cov) / n_cov)) if n_cov else np.nan),
                n_cov_usable=n_cov,
                adapt_fallback_rate=float(flags[:, 0].mean()),
                adapt_clip_rate=float(flags[:, 1].mean()),
                star_fallback_rate_oracle=float(flags[:, 2].mean()),
                star_fallback_rate_adaptive=float(flags[:, 3].mean()),
                star_t_oracle_mean=float(np.nanmean(star_t[:, 0])),
                star_t_adaptive_mean=float(np.nanmean(star_t[:, 1])),
                are_fixed_theory=th['are_fixed'], are_star_theory=th['are_star'],
                t_fixed_theory=th['t_fixed'], t_opt_theory=th['t_opt'],
            ))
    return rows


def _paired(cell, beta, conv, truth, m):
    from estimators import theory_are
    th = theory_are(cell)
    idx = {nm: j for j, nm in enumerate(ESTIMATOR_NAMES)}
    rows = []
    rng = np.random.default_rng(991)
    for a, b in PAIRS:
        for p, pname in ((0, 'b0'), (1, 'b1')):
            ea = beta[:, idx[a], p] - truth[p]
            eb = beta[:, idx[b], p] - truth[p]
            ok = np.isfinite(ea) & np.isfinite(eb)
            sa, sb = ea[ok] ** 2, eb[ok] ** 2
            d = sa - sb
            n_ok = d.size
            dm = float(d.mean()) if n_ok else np.nan
            dse = float(d.std(ddof=1) / np.sqrt(n_ok)) if n_ok > 1 else np.nan
            re = float(sa.mean() / sb.mean()) if n_ok and sb.mean() > 0 else np.nan
            # paired bootstrap CI for the MSE ratio (resampling unit = replication)
            if n_ok > 1:
                bs = rng.integers(0, n_ok, size=(2000, n_ok))
                ratios = sa[bs].mean(axis=1) / sb[bs].mean(axis=1)
                re_lo, re_hi = np.percentile(ratios, [2.5, 97.5])
            else:
                re_lo = re_hi = np.nan
            rows.append(dict(
                cell_id=cell.cell_id, eps_key=cell.eps_key, u_family=cell.u_family,
                lam=cell.lam, n=cell.n, b1_true=cell.b1, delta=cell.delta,
                g2_theory=cell.eps.moments['g2'], m_rep=m, n_paired=n_ok,
                estimator_a=a, estimator_b=b, param=pname,
                mse_a=float(sa.mean()) if n_ok else np.nan,
                mse_b=float(sb.mean()) if n_ok else np.nan,
                mse_diff=dm, se_mse_diff=dse,
                ci_lo=dm - Z95 * dse if n_ok > 1 else np.nan,
                ci_hi=dm + Z95 * dse if n_ok > 1 else np.nan,
                re=re, re_boot_lo=float(re_lo), re_boot_hi=float(re_hi),
                # MC SE of the RE ratio (delta method on the paired ratio) --
                # gate G2 compares RE against the closed-form target within 2 SE
                se_re=(float(np.std(sa - re * sb, ddof=1) / np.sqrt(n_ok) / sb.mean())
                       if n_ok > 1 and sb.mean() > 0 else np.nan),
                are_fixed_theory=th['are_fixed'], are_star_theory=th['are_star'],
            ))
    return rows
