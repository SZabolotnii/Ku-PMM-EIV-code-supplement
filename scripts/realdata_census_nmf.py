"""Real-data study 2: the 2020 Census Noisy Measurement File (NMF).

Why this dataset exists in this project
---------------------------------------
The NHANES study (scripts/realdata_nhanes.py) rejected the model's conditional-moment
conditions and could not say *which* of them failed, because with replicate-estimated
moments the measurement-error side and the response side fail together.

The NMF separates them.  It is, as far as we could establish, the only public dataset in
which the classical errors-in-variables model holds by *construction* rather than by
assumption: the U.S. Census Bureau adds discrete Gaussian noise to each tabulated count
under zero-concentrated differential privacy, the noise is drawn independently of the
data, and the variance of every single measurement is published alongside it.  Therefore

    * nu2 is KNOWN, not estimated -- no replicates, no exchangeability requirement;
    * nu3 = nu5 = 0 and nu4 = 3 nu2^2, nu6 = 15 nu2^3 exactly (the discrete Gaussian's
      departure from the Gaussian moments is O(exp(-2 pi^2 sigma^2)), i.e. below machine
      precision at every sigma occurring here);
    * U _||_ X holds by the mechanism, not by hope.

That leaves the response side as the only thing that can fail -- and it does.  This script
measures both halves and confirms, to within 0.4%, the bias formula of the
robustness-asymmetry proposition:  E[Psi1] = -b * Cov(X, eps^2).

Ground truth.  Several distinct queries measure the SAME true count with different,
independently drawn noise.  The most precise of them (``total_dpq``, sigma = 1.75 at the
block-group level, against a between-unit spread of ~700) serves as a near-exact reference
X, so every estimator can be scored against a gold standard -- something no replicate-based
design in this literature can offer.

Resampling unit.  County (geocode[5:8]), per Verification Standard 9: the DP noise is
independent across units by construction, but the *regression* error is spatially
correlated, and county is the coarsest grouping that keeps n_groups >> 10.

Run:  python3 scripts/realdata_census_nmf.py [--boot B]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimators import (central_moments, fit_ls, fit_tc_ls, fit_tc_pmm2,
                        fit_tc_pmm2_star, optimal_t, are_of_t, _tc_psi_jac)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'census_nmf'
OUT = ROOT / 'evidence' / 'realdata_census_nmf'

S3 = ('https://uscb-2020-product-releases.s3.us-west-2.amazonaws.com/decennial/'
      'redistricting/2020/nmf/2020-pl94-nmf-state-partitioned/US_Person_PL_PROD/')
FILES = {
    'bg': (S3 + 'Block_Group.parquet/DPQuery/State=039/'
                'part-00030-51f19ca3-3e6a-4d66-bcdb-29e1b18a3571.c000.snappy.parquet',
           'oh_block_group.parquet'),
}
STATE = '039'          # Ohio: 88 counties, 15 333 block groups -- large enough for a
                       # county-clustered bootstrap and small enough to stay a 41 MB file.


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def fetch() -> dict[str, Path]:
    DATA.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (url, name) in FILES.items():
        p = DATA / name
        if not p.exists():
            print(f'  downloading {name} ...', flush=True)
            urllib.request.urlretrieve(url, p)
        paths[key] = p
    return paths


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    """Return dict query_name -> (value matrix (n, ncells), per-cell noise variance),
    plus the geocode vector.  Rows are sorted by geocode so every query aligns."""
    tab = pq.read_table(path, columns=['geocode', 'query_name', 'value', 'variance'])
    df = tab.to_pandas()
    out, geo = {}, None
    for q, sub in df.groupby('query_name'):
        sub = sub.set_index('geocode').sort_index()
        if geo is None:
            geo = sub.index.to_numpy()
        elif not np.array_equal(geo, sub.index.to_numpy()):
            raise RuntimeError(f'geocode misalignment in {q}')
        out[q] = (np.stack(sub['value'].apply(np.asarray).values).astype(float),
                  float(sub['variance'].iloc[0]))
    return out, geo


# ---------------------------------------------------------------------------
# gate 1: is the declared noise the actual noise, and is it independent across queries?
# ---------------------------------------------------------------------------
def query_variance_check(Q) -> list[dict]:
    """Falsifiable prediction.  Several queries sum to the same true total.  If their
    noise is independent and mean-zero, then for any two of them

        Var(sum_A - sum_B) = ncells_A * var_A + ncells_B * var_B

    exactly.  Nothing is fitted here; the right-hand side is published metadata."""
    ref, vref = Q['total_dpq']
    ref = ref[:, 0]
    rows = []
    for q in ['hispanic_dpq', 'votingage_dpq', 'hhgq_dpq', 'cenrace_dpq',
              'hhinstlevels_dpq']:
        val, var = Q[q]
        s = val.sum(1)
        d = s - ref
        pred = var * val.shape[1] + vref
        rows.append(dict(query=q, ncells=int(val.shape[1]),
                         var_diff_measured=float(d.var(ddof=1)),
                         var_diff_predicted=float(pred),
                         ratio=float(d.var(ddof=1) / pred),
                         mean_diff=float(d.mean()),
                         mean_diff_se=float(d.std(ddof=1) / np.sqrt(d.size))))
    return rows


# ---------------------------------------------------------------------------
# one configuration
# ---------------------------------------------------------------------------
def analyse(tag, y, w, ref, nu2, var_y, county, purpose, boot=0, seed=20260815):
    """ref is the near-exact reference for X.  All moments of U are exact (Gaussian)."""
    n = y.size
    sx2 = float(ref.var(ddof=1) - VAR_TOTAL)
    lam = sx2 / (sx2 + nu2)
    nu = dict(nu2=nu2, nu3=0.0, nu4=3.0 * nu2 ** 2, nu5=0.0, nu6=15.0 * nu2 ** 3)

    gold = fit_ls(y, ref, 'gold')                       # target: X known to +-1.75
    naive = fit_ls(y, w, 'naive')
    tcls = fit_tc_ls(y, w, nu2)
    resid = y - gold.beta[0] - gold.beta[1] * ref
    m2, m3, m4 = central_moments(resid)                 # eps includes Y's own DP noise,
                                                        # which is part of the regression
                                                        # error by definition -- NOT
                                                        # deconvolved.
    c3, c4 = m3 / m2 ** 1.5, m4 / m2 ** 2 - 3.0
    g2 = 1.0 - c3 ** 2 / (2.0 + c4)
    t_opt, g2_eiv, _ = optimal_t(sx2, tcls.beta[1], m2, m3, m4, **nu)
    t_pmm = m3 / (m4 - m2 ** 2)
    are_transplant = are_of_t(t_pmm, sx2, tcls.beta[1], m2, m3, m4, **nu)
    tc2 = fit_tc_pmm2(y, w, m2, m3, m4, nu2, 0.0, tcls.beta, 'tc_pmm2')
    star = fit_tc_pmm2_star(y, w, m2, m3, m4, nu, sx2, tcls.beta, 'tc_pmm2star')

    # --- U-side verification (this is the half that the model gets exactly right) ---
    u = w - ref
    u_side = dict(
        nu2_declared=float(nu2), nu2_measured=float(u.var(ddof=1)),
        nu2_ratio=float(u.var(ddof=1) / nu2),
        mean_u=float(u.mean()), mean_u_se=float(u.std(ddof=1) / np.sqrt(n)),
        skew_u=float(stats.skew(u)), exkurt_u=float(stats.kurtosis(u)),
        corr_u_x=float(np.corrcoef(u, ref)[0, 1]),
        corr_u2_x=float(np.corrcoef(u ** 2, ref)[0, 1]),
        corr_u_eps=float(np.corrcoef(u, resid)[0, 1]))

    # --- response-side conditions: the ONLY extra one the degree-two score needs ---
    # Cov(X, eps^2) looks unobservable, but it is exactly identified:
    #     Cov(Wt, delta^2) = Cov(X, eps^2) + b1^2 nu3,   delta = Y - b0 - b1 Wt,
    # every cross term vanishing under U _||_ (X, eps) and E eps = 0.  The right-hand
    # column below therefore needs the gold standard; the left-hand one does not, and it
    # is what a practitioner can actually compute.  Its standard error is dominated by
    # the nu3-estimating term, ~ b1^2 sqrt(15 nu2^3 / n).
    cov_x_e2 = float(np.cov(ref, resid ** 2, ddof=1)[0, 1])
    delta = y - tcls.beta[0] - tcls.beta[1] * w
    cov_obs = float(np.cov(w, delta ** 2, ddof=1)[0, 1] - tcls.beta[1] ** 2 * nu['nu3'])
    cov_obs_se = float(tcls.beta[1] ** 2 * np.sqrt(15.0 * nu2 ** 3 / n))
    eps_side = dict(
        corr_eps_x=float(np.corrcoef(resid, ref)[0, 1]),
        corr_eps2_x=float(np.corrcoef(resid ** 2, ref)[0, 1]),
        corr_eps3_x=float(np.corrcoef(resid ** 3, ref)[0, 1]),
        cov_x_eps2_gold=cov_x_e2,
        cov_x_eps2_observable=cov_obs,
        cov_x_eps2_observable_se=cov_obs_se,
        identification_ratio=float(cov_obs / cov_x_e2) if cov_x_e2 else float('nan'),
        spearman_p=float(stats.spearmanr(ref, np.abs(resid - resid.mean()))[1]))

    # --- the bias formula, checked against the score evaluated at the gold beta ---
    psi_lin = np.column_stack([y - gold.beta[0] - gold.beta[1] * w,
                               w * (y - gold.beta[0] - gold.beta[1] * w)
                               + gold.beta[1] * nu2])
    psi_q, jac = _tc_psi_jac(gold.beta, y, w, m2, m3, m4, nu2, 0.0, weights=(1.0, t_opt))
    implied = np.linalg.solve(-jac / n, psi_q.mean(0))
    bias_check = dict(
        t_used=float(t_opt),
        lin_psi0_z=float(psi_lin[:, 0].mean() / (psi_lin[:, 0].std(ddof=1) / np.sqrt(n))),
        lin_psi1_z=float(psi_lin[:, 1].mean() / (psi_lin[:, 1].std(ddof=1) / np.sqrt(n))),
        quad_psi0_z=float(psi_q[:, 0].mean() / (psi_q[:, 0].std(ddof=1) / np.sqrt(n))),
        quad_psi1_z=float(psi_q[:, 1].mean() / (psi_q[:, 1].std(ddof=1) / np.sqrt(n))),
        quad_psi1_measured=float(psi_q[:, 1].mean()),
        quad_psi1_predicted=float(-t_opt * cov_x_e2),
        formula_ratio=float(psi_q[:, 1].mean() / (-t_opt * cov_x_e2)),
        # The ratio is only meaningful when the predicted bias is large relative to the
        # score's own sampling noise; at t_opt ~ 0 the degree-two score collapses onto
        # the linear one and both sides are noise.
        formula_informative=bool(abs(-t_opt * cov_x_e2)
                                 > 2.0 * psi_q[:, 1].std(ddof=1) / np.sqrt(n)),
        implied_slope_bias=float(implied[1]),
        observed_star_minus_gold=float(star.beta[1] - gold.beta[1]))

    # --- county-clustered bootstrap (Verification Standard 9) ---
    boot_out = {}
    if boot:
        rng = np.random.default_rng(seed)
        counties, inv = np.unique(county, return_inverse=True)
        idx_by_c = [np.flatnonzero(inv == k) for k in range(counties.size)]
        keep = {'tcls': [], 'star': [], 'gold': [], 'star_minus_gold': []}
        for _ in range(boot):
            pick = rng.integers(0, counties.size, counties.size)
            sel = np.concatenate([idx_by_c[k] for k in pick])
            yb, wb, rb = y[sel], w[sel], ref[sel]
            gb = fit_ls(yb, rb, 'g')
            lb = fit_tc_ls(yb, wb, nu2)
            rs = yb - gb.beta[0] - gb.beta[1] * rb
            b2, b3, b4 = central_moments(rs)
            sb = float(rb.var(ddof=1) - VAR_TOTAL)
            st = fit_tc_pmm2_star(yb, wb, b2, b3, b4, nu, sb, lb.beta, 's')
            keep['tcls'].append(lb.beta[1]); keep['star'].append(st.beta[1])
            keep['gold'].append(gb.beta[1])
            keep['star_minus_gold'].append(st.beta[1] - gb.beta[1])
        for k, v in keep.items():
            v = np.asarray(v)
            boot_out[k] = dict(sd=float(v.std(ddof=1)),
                               lo=float(np.quantile(v, .025)),
                               hi=float(np.quantile(v, .975)))
        boot_out['n_counties'] = int(counties.size)

    return dict(
        tag=tag, purpose=purpose, n=int(n), n_counties=int(np.unique(county).size),
        lambda_=float(lam), sx2=sx2, nu2=float(nu2), var_y_noise=float(var_y),
        varsigma=float(gold.beta[1] ** 2 * sx2 / m2),
        dilution=float(gold.beta[1] ** 2 * sx2 / m2 * (1 - lam) / lam),
        m2=float(m2), m3=float(m3), m4=float(m4), c3=float(c3), c4=float(c4),
        g2=float(g2), g2_eiv=float(g2_eiv), t_opt=float(t_opt), t_pmm2=float(t_pmm),
        are_transplanted=float(are_transplant),
        beta1=dict(gold=float(gold.beta[1]), naive=float(naive.beta[1]),
                   tc_ls=float(tcls.beta[1]), tc_pmm2=float(tc2.beta[1]),
                   tc_pmm2star=float(star.beta[1])),
        se=dict(gold=float(gold.se[1]), tc_ls=float(tcls.se[1]),
                tc_pmm2=float(tc2.se[1]), tc_pmm2star=float(star.se[1])),
        var_ratio_star_over_tcls=float((star.se[1] / tcls.se[1]) ** 2),
        u_side=u_side, eps_side=eps_side, bias_check=bias_check, bootstrap=boot_out)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=400)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    paths = fetch()
    Q, geo = load(paths['bg'])
    county = np.array([g[5:8] for g in geo])

    global VAR_TOTAL
    tot, VAR_TOTAL = Q['total_dpq']
    ref_all = tot[:, 0]

    qv = query_variance_check(Q)

    hisp, v_hisp = Q['hispanic_dpq']
    va, v_va = Q['votingage_dpq']
    cen, v_cen = Q['cenrace_dpq']
    hc, v_hc = Q['hispanic * cenrace_dpq']
    inst, v_inst = Q['hhinstlevels_dpq']
    gq = inst[:, 1] + inst[:, 2]

    band_a = (ref_all >= 900) & (ref_all <= 2000)
    band_b = (ref_all >= 700) & (ref_all <= 2500) & (gq < 100)

    configs = [
        dict(tag='A_white_nh_on_total', keep=band_a,
             y=hc[:, 0], var_y=v_hc, w=hisp.sum(1), nu2=v_hisp * hisp.shape[1],
             purpose='U-side verification and the Cov(X,eps^2) bias formula'),
        dict(tag='B_votingage_on_total', keep=band_b,
             y=va[:, 1], var_y=v_va, w=cen.sum(1), nu2=v_cen * cen.shape[1],
             purpose='real-data check of the negative result: transplanted weights lose'),
    ]

    results = []
    for cfg in configs:
        k = cfg['keep']
        results.append(analyse(cfg['tag'], cfg['y'][k], cfg['w'][k], ref_all[k],
                               cfg['nu2'], cfg['var_y'], county[k], cfg['purpose'],
                               boot=args.boot))

    manifest = dict(
        state=STATE, source=S3,
        files={n: dict(sha256=sha256(p), bytes=p.stat().st_size)
               for n, p in ((p.name, p) for p in paths.values())},
        n_units=int(ref_all.size), boot=args.boot,
        note=('US Government work, public domain (17 U.S.C. 105).  The 2023-06-15 NMF '
              'release is a fixed, dated artefact; the raw parquet is therefore not '
              'tracked in git -- the sha256 above pins it.'))

    (OUT / 'query_variance_check.json').write_text(json.dumps(qv, indent=2))
    (OUT / 'configurations.json').write_text(json.dumps(results, indent=2))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    lines = ['=== Census 2020 NMF real-data study ===', '']
    lines.append('-- Gate N1: declared noise variance == actual noise variance,')
    lines.append('   and noise independent across queries (nothing is fitted) --')
    for r in qv:
        ok = 'PASS' if 0.94 <= r['ratio'] <= 1.06 else 'FAIL'
        lines.append(f"  [{ok}] {r['query']:18s} ratio={r['ratio']:.4f}  "
                     f"mean_diff={r['mean_diff']:+.3f} (se {r['mean_diff_se']:.3f})")
    for r in results:
        lines += ['', f"-- {r['tag']}  ({r['purpose']}) --",
                  f"  n={r['n']}  counties={r['n_counties']}  lambda={r['lambda_']:.4f}  "
                  f"varsigma={r['varsigma']:.3f}",
                  f"  eps: c3={r['c3']:+.3f} c4={r['c4']:+.3f} -> g2={r['g2']:.4f}; "
                  f"g2_EIV={r['g2_eiv']:.4f}; ARE(transplanted)={r['are_transplanted']:.4f}"]
        u = r['u_side']
        ok = ('PASS' if (0.94 <= u['nu2_ratio'] <= 1.06 and abs(u['corr_u_x']) < .05
                         and abs(u['corr_u2_x']) < .05) else 'FAIL')
        lines.append(f"  [{ok}] U-side: nu2 ratio={u['nu2_ratio']:.4f}, skew={u['skew_u']:+.4f}, "
                     f"exkurt={u['exkurt_u']:+.4f}, corr(U,X)={u['corr_u_x']:+.4f}, "
                     f"corr(U^2,X)={u['corr_u2_x']:+.4f}, corr(U,eps)={u['corr_u_eps']:+.4f}")
        e = r['eps_side']
        ok = 'PASS' if abs(e['corr_eps2_x']) < .05 else 'FAIL'
        lines.append(f"  [{ok}] eps-side: corr(eps^2,X)={e['corr_eps2_x']:+.4f}, "
                     f"corr(eps^3,X)={e['corr_eps3_x']:+.4f}, spearman p={e['spearman_p']:.2e}")
        b = r['bias_check']
        lines.append(f"  score at the gold beta: linear z=({b['lin_psi0_z']:+.2f}, "
                     f"{b['lin_psi1_z']:+.2f});  degree-two z=({b['quad_psi0_z']:+.2f}, "
                     f"{b['quad_psi1_z']:+.2f})")
        note = '' if b['formula_informative'] else '   [not informative: t_opt ~ 0, the '
        note += '' if b['formula_informative'] else 'degree-two score collapses onto the linear one]'
        lines.append(f"  bias formula  E[Psi1] = -b Cov(X,eps^2):  measured "
                     f"{b['quad_psi1_measured']:+.1f}  predicted {b['quad_psi1_predicted']:+.1f}"
                     f"  ratio={b['formula_ratio']:.4f}{note}")
        lines.append(f"  Cov(X,eps^2) from observables only = {e['cov_x_eps2_observable']:.4g} "
                     f"(se {e['cov_x_eps2_observable_se']:.2g})  vs gold-standard "
                     f"{e['cov_x_eps2_gold']:.4g}   ratio={e['identification_ratio']:.4f}")
        lines.append(f"  implied slope bias {b['implied_slope_bias']:+.6f}  vs observed "
                     f"{b['observed_star_minus_gold']:+.6f}")
        lines.append("  beta1: " + "  ".join(f"{k}={v:+.5f}" for k, v in r['beta1'].items()))
        lines.append(f"  Var(star)/Var(tc_ls) = {r['var_ratio_star_over_tcls']:.4f}  "
                     f"(predicted g2_EIV = {r['g2_eiv']:.4f})")
        if r['bootstrap']:
            bs = r['bootstrap']
            lines.append(f"  county bootstrap ({bs['n_counties']} clusters): "
                         f"star-gold 95% CI [{bs['star_minus_gold']['lo']:+.5f}, "
                         f"{bs['star_minus_gold']['hi']:+.5f}]")
    (OUT / 'verification_report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
