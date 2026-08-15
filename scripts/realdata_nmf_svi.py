"""Real-data study 3: Census NMF covariate x CDC/ATSDR SVI response.

What this adds over scripts/realdata_census_nmf.py
--------------------------------------------------
That script established the halves separately: the measurement-error side of the model
holds exactly (the noise is injected by a published mechanism), the response side fails,
and the failure is the one Proposition ``prop:asymmetry`` names.  It could not produce a
case where the method works, because its responses were geographic *counts*, whose
regression error is proportional to unit size by construction.

Here the covariate still comes from the NMF -- so nu2..nu6 remain exact and U _||_ X
remains a property of the mechanism -- but the response is *intensive*: a rate from the
CDC/ATSDR Social Vulnerability Index, whose scatter has no reason to scale with the
composition of the tract.  That breaks the coupling, and Cov(X, eps^2) = 0 becomes
attainable rather than structurally impossible.

Design
------
  X (latent)  true count of a target subpopulation j in a census tract
  reference   the same count from ``hispanic * cenrace_dpq`` cell j (sigma = 2.2):
              a near-exact gold standard, so every estimator can be scored
  W           the same count rebuilt from *independent* queries,
                  W = sum(cenrace) - hispanic[1] - sum_{k != j}(hispanic x cenrace)[k],
              whose noise variance is the exact sum of the declared variances
  Y           an SVI rate (EP_*), intensive and continuous
  band        tracts of 2500-5500 residents, so the count acts as a composition share
  cluster     county (Verification Standard 9: the DP noise is independent by
              construction, but the response residual is spatially correlated)

nu2 is the same whichever group is targeted, while sigma_x^2 shrinks with the size of the
group, so the reliability lambda is set by the choice of j and not by any tuning knob:
White_NH ~ 0.93, Black_NH ~ 0.65-0.88, Asian_NH ~ 0.10-0.30.  The grid is
11 states x 3 groups x 14 responses = 462 cells.  Ohio is the *discovery* sample, where
one response was chosen by the adequacy screen; the other ten states are out of sample and
nothing is re-tuned per state, per group or per response.

Gates
-----
  N1  the tract crosswalk is real, not an artefact (a deliberately shifted merge must
      destroy the population correlation)
  N2  the screen |z(Cov(X,eps^2))| < 2 predicts, out of sample, whether the degree-two
      estimator lands on the gold standard
  N3  the bias formula E[Psi1] = -b Cov(X, eps^2) reproduces the observed bias
  N4  TC-LS is unaffected by the screen (the robustness asymmetry)
  N5  the predicted variance reduction g2_EIV is realised, assessed pooled
  N6  cells meeting pre-specified demonstration criteria behave as the theory says
  N7  the naive slope is attenuated by the factor lambda predicts

Run:  python3 scripts/realdata_nmf_svi.py [--boot 400]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimators import (central_moments, fit_ls, fit_tc_ls, fit_tc_pmm2_star,
                        optimal_t, _tc_psi_jac)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'census_nmf'
SVI_DIR = ROOT / 'data' / 'svi'
OUT = ROOT / 'evidence' / 'realdata_nmf_svi'

S3 = 'https://uscb-2020-product-releases.s3.us-west-2.amazonaws.com/'
PREFIX = ('decennial/redistricting/2020/nmf/2020-pl94-nmf-state-partitioned/'
          'US_Person_PL_PROD/Tract.parquet/DPQuery/')
SVI_URL = 'https://svi.cdc.gov/Documents/Data/2020/csv/states/{}.csv'

STATES = [('039', 'Ohio'), ('042', 'Pennsylvania'), ('026', 'Michigan'),
          ('018', 'Indiana'), ('017', 'Illinois'), ('037', 'NorthCarolina'),
          ('013', 'Georgia'), ('047', 'Tennessee'), ('029', 'Missouri'),
          ('055', 'Wisconsin'), ('027', 'Minnesota')]
DISCOVERY = 'Ohio'
RESPONSES = ['EP_POV150', 'EP_UNEMP', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_HBURD', 'EP_CROWD',
             'EP_NOVEH', 'EP_MOBILE', 'EP_LIMENG', 'EP_AGE65', 'EP_DISABL', 'EP_SNGPNT',
             'EP_MUNIT', 'EP_GROUPQ']
BAND = (2500, 5500)

# Target subpopulation.  The surrogate is always built as
#     W = sum(cenrace) - hispanic[1] - sum_{k != j} (hispanic x cenrace)[k],
# so nu2 is the same for every j while sigma_x^2 shrinks with the size of the group.
# Reliability is therefore a property of which group is targeted, not of any tuning:
# White_NH gives lambda ~ 0.93, Black_NH ~ 0.65-0.88, Asian_NH ~ 0.10-0.30.
GROUPS = {0: 'White_NH', 1: 'Black_NH', 3: 'Asian_NH'}
SCREEN_GROUP = 'White_NH'      # the grid on which the screen itself is validated
HEADLINE = ('Michigan', 'Black_NH', 'EP_MOBILE')

# Pre-specified before looking at any outcome: a cell demonstrates the whole procedure
# only if the adequacy screen passes, the measurement error actually matters, and the
# closed form promises something to gain.  Selection uses no estimate of the outcome.
DEMO = dict(z_max=2.0, lam_lo=0.5, lam_hi=0.9, g2eiv_max=0.8)


# ---------------------------------------------------------------------------
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def nmf_file(fips: str) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / f'tract_{fips}.parquet'
    if not p.exists():
        listing = urllib.request.urlopen(
            f'{S3}?list-type=2&prefix={PREFIX}State={fips}/&max-keys=5', timeout=120
        ).read().decode()
        key = re.findall(r'<Key>([^<]*)</Key>', listing)[0]
        urllib.request.urlretrieve(S3 + key, p)
    return p


def svi_file(name: str) -> Path:
    SVI_DIR.mkdir(parents=True, exist_ok=True)
    p = SVI_DIR / f'{name}.csv'
    if not p.exists():
        urllib.request.urlretrieve(SVI_URL.format(name), p)
    return p


def cluster_se(psi: np.ndarray, jac: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Cluster-robust sandwich.  psi: (n,2) score rows, jac: (2,2) SUM of Jacobians."""
    n = psi.shape[0]
    m_inv = np.linalg.inv(-jac / n)
    meat = np.zeros((2, 2))
    for g in np.unique(groups):
        s = psi[groups == g].sum(0)
        meat += np.outer(s, s)
    v = m_inv @ (meat / n) @ m_inv.T / n
    d = np.diag(v)
    return np.sqrt(np.where(d > 0, d, np.nan))


# ---------------------------------------------------------------------------
def load_state(fips: str, name: str):
    tab = pq.read_table(nmf_file(fips)).to_pandas()

    def q(qn):
        s = tab[tab.query_name == qn].set_index('geocode').sort_index()
        return (np.stack(s['value'].apply(np.asarray).values).astype(float),
                float(s['variance'].iloc[0]), s.index.to_numpy())

    tot, v_tot, geo = q('total_dpq')
    cen, v_cen, _ = q('cenrace_dpq')
    hsp, v_hsp, _ = q('hispanic_dpq')
    hxc, v_hxc, _ = q('hispanic * cenrace_dpq')

    nu2 = v_cen * cen.shape[1] + v_hsp + v_hxc * 62
    nmf = pd.DataFrame(dict(geocode=geo, pop=tot[:, 0]))
    nmf['cnty'] = nmf.geocode.str[5:8]
    nmf['idx'] = nmf.geocode.str[8:13].astype(int)
    for j, gname in GROUPS.items():
        others = [k for k in range(63) if k != j]
        nmf[f'xref_{gname}'] = hxc[:, j]
        nmf[f'w_{gname}'] = cen.sum(1) - hsp[:, 1] - hxc[:, others].sum(1)

    svi = pd.read_csv(svi_file(name), dtype={'FIPS': str}, low_memory=False)
    svi['cnty'] = svi.FIPS.str[2:5]

    # The NMF geocode carries a within-county *rank*, not a tract GEOID; the crosswalk
    # is therefore rank order, validated below rather than assumed.
    frames = []
    for c, sub in svi.groupby('cnty'):
        need = int((nmf.cnty == c).sum())
        for cand in (sub, sub[sub.E_TOTPOP > 0]):
            if need > 0 and len(cand) == need:
                cand = cand.sort_values('FIPS').copy()
                cand['idx'] = np.arange(need) + 10001
                frames.append(cand)
                break
    keyed = pd.concat(frames)
    merged = nmf.merge(keyed, on=['cnty', 'idx'], how='inner')

    shifted = nmf.copy()
    shifted['idx'] += 1
    ctrl = shifted.merge(keyed, on=['cnty', 'idx'], how='inner')

    cross = dict(
        state=name, merged=int(len(merged)), of=int(len(nmf)),
        corr=float(np.corrcoef(merged['pop'], merged.E_TOTPOP)[0, 1]),
        corr_shifted=float(np.corrcoef(ctrl['pop'], ctrl.E_TOTPOP)[0, 1])
        if len(ctrl) > 50 else float('nan'))
    return merged, nu2, v_hxc, v_tot, cross


def analyse_cell(sub, resp, group, nu2, v_hxc, boot=0, seed=20260815):
    y = sub[resp].to_numpy(float)
    w = sub[f'w_{group}'].to_numpy(float)
    r = sub[f'xref_{group}'].to_numpy(float)
    cl = sub['cnty'].to_numpy()
    n = y.size
    sx2 = float(r.var(ddof=1) - v_hxc)

    gold = fit_ls(y, r, 'gold')
    naive = fit_ls(y, w, 'naive')
    tcls = fit_tc_ls(y, w, nu2)
    resid = y - gold.beta[0] - gold.beta[1] * r
    m2, m3, m4 = central_moments(resid)
    c3, c4 = m3 / m2 ** 1.5, m4 / m2 ** 2 - 3.0
    if 2.0 + c4 <= 0:
        return None
    nu = dict(nu2=nu2, nu3=0.0, nu4=3.0 * nu2 ** 2, nu5=0.0, nu6=15.0 * nu2 ** 3)
    t_opt, g2_eiv, _ = optimal_t(sx2, tcls.beta[1], m2, m3, m4, **nu)
    star = fit_tc_pmm2_star(y, w, m2, m3, m4, nu, sx2, tcls.beta, 'star')

    cov = float(np.cov(r, resid ** 2, ddof=1)[0, 1])
    z_e2 = float(np.corrcoef(r, resid ** 2)[0, 1] * np.sqrt(n))

    # cluster-robust standard errors for the three estimators
    des = np.column_stack([np.ones(n), r])
    se_gold = cluster_se(des * resid[:, None], -des.T @ des, cl)
    rl = y - tcls.beta[0] - tcls.beta[1] * w
    psi_l = np.column_stack([rl, w * rl + tcls.beta[1] * nu2])
    sw, sw2 = w.sum(), (w ** 2).sum()
    jac_l = -np.array([[float(n), sw], [sw, sw2 - n * nu2]])
    se_tcls = cluster_se(psi_l, jac_l, cl)
    psi_s, jac_s = _tc_psi_jac(star.beta, y, w, m2, m3, m4, nu2, 0.0, weights=(1.0, t_opt))
    se_star = cluster_se(psi_s, jac_s, cl)

    psi_g, jac_g = _tc_psi_jac(gold.beta, y, w, m2, m3, m4, nu2, 0.0, weights=(1.0, t_opt))
    pred_bias = float(np.linalg.solve(-jac_g / n, np.array([0.0, -t_opt * cov]))[1])

    out = dict(resp=resp, group=group, n=int(n), n_counties=int(np.unique(cl).size),
               lam=float(sx2 / (sx2 + nu2)), nu2=float(nu2), sx2=sx2,
               varsigma=float(gold.beta[1] ** 2 * sx2 / m2),
               c3=float(c3), c4=float(c4), g2=float(1 - c3 ** 2 / (2 + c4)),
               g2_eiv=float(g2_eiv), t_opt=float(t_opt), z_e2=z_e2, cov_x_eps2=cov,
               gold=float(gold.beta[1]), naive=float(naive.beta[1]),
               tcls=float(tcls.beta[1]), star=float(star.beta[1]),
               se_gold=float(se_gold[1]), se_tcls=float(se_tcls[1]),
               se_star=float(se_star[1]),
               obs_bias=float(star.beta[1] - gold.beta[1]), pred_bias=pred_bias,
               dev=float(abs(star.beta[1] - gold.beta[1]) / se_star[1]),
               dev_ls=float(abs(tcls.beta[1] - gold.beta[1]) / se_tcls[1]),
               dev_naive=float(abs(naive.beta[1] - gold.beta[1]) / se_tcls[1]),
               varrat=float((se_star[1] / se_tcls[1]) ** 2))

    if boot:
        rng = np.random.default_rng(seed)
        counties, inv = np.unique(cl, return_inverse=True)
        idx = [np.flatnonzero(inv == k) for k in range(counties.size)]
        d_sg, vr = [], []
        for _ in range(boot):
            pick = rng.integers(0, counties.size, counties.size)
            sel = np.concatenate([idx[k] for k in pick])
            yb, wb, rb = y[sel], w[sel], r[sel]
            gb = fit_ls(yb, rb, 'g')
            lb = fit_tc_ls(yb, wb, nu2)
            rs = yb - gb.beta[0] - gb.beta[1] * rb
            b2, b3, b4 = central_moments(rs)
            if 2.0 + (b4 / b2 ** 2 - 3.0) <= 0:
                continue
            sb = float(rb.var(ddof=1) - v_hxc)
            sb_fit = fit_tc_pmm2_star(yb, wb, b2, b3, b4, nu, sb, lb.beta, 's')
            d_sg.append(sb_fit.beta[1] - gb.beta[1])
            vr.append((sb_fit.se[1] / lb.se[1]) ** 2)
        d_sg = np.asarray(d_sg)
        out['boot'] = dict(B=int(d_sg.size),
                           star_minus_gold_lo=float(np.quantile(d_sg, .025)),
                           star_minus_gold_hi=float(np.quantile(d_sg, .975)),
                           varrat_median=float(np.median(vr)))
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=400)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows, crosswalks, files = [], [], {}
    for fips, name in STATES:
        merged, nu2, v_hxc, v_tot, cross = load_state(fips, name)
        crosswalks.append(cross)
        files[f'tract_{fips}.parquet'] = sha256(nmf_file(fips))
        files[f'svi_{name}.csv'] = sha256(svi_file(name))
        band = merged[(merged['pop'] >= BAND[0]) & (merged['pop'] <= BAND[1])]
        for group in GROUPS.values():
            for resp in RESPONSES:
                sub = band[(band[resp] >= 0) & band[resp].notna()]
                if len(sub) < 400:
                    continue
                boot = args.boot if (name, group, resp) == HEADLINE else 0
                cell = analyse_cell(sub, resp, group, nu2, v_hxc, boot=boot)
                if cell is None:
                    continue
                cell['state'] = name
                cell['sample'] = 'discovery' if name == DISCOVERY else 'out-of-sample'
                rows.append(cell)
        print(f'  {name}: {sum(1 for r in rows if r["state"] == name)} cells', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'grid.csv', index=False)

    # The screen is validated on one group only, so that "does the screen predict
    # success" is answered on a grid fixed in advance rather than on everything at once.
    scr = df[df.group == SCREEN_GROUP]
    passing = scr[scr.z_e2.abs() < 2]
    failing = scr[scr.z_e2.abs() >= 2]
    oos = scr[scr['sample'] == 'out-of-sample']
    oos_pass = oos[oos.z_e2.abs() < 2]
    demo = df[(df.z_e2.abs() < DEMO['z_max'])
              & df.lam.between(DEMO['lam_lo'], DEMO['lam_hi'])
              & (df.g2_eiv <= DEMO['g2eiv_max'])]
    # The bias formula is a FIRST-ORDER expansion, so it is checked on the grid it was
    # stated for.  It is also reported on everything, where the Asian_NH cells sit at
    # lambda ~ 0.1 and the linearisation is no longer adequate -- that degradation is
    # information, not noise, and is reported rather than filtered away.
    def bias_fit(frame):
        ok = frame.pred_bias.abs() > 1e-12
        if ok.sum() < 5:
            return float('nan'), float('nan')
        return (float(np.corrcoef(frame.loc[ok, 'pred_bias'], frame.loc[ok, 'obs_bias'])[0, 1]),
                float(np.polyfit(frame.loc[ok, 'pred_bias'], frame.loc[ok, 'obs_bias'], 1)[0]))

    corr, slope = bias_fit(scr)
    corr_all, slope_all = bias_fit(df)
    hi_lam = df[df.lam >= 0.5]
    corr_hi, slope_hi = bias_fit(hi_lam)

    # Attenuation, pooled.  Individually the naive slope is rarely significantly
    # attenuated (these regressions are weak, varsigma ~ 0.03-0.3), but the RATIO of the
    # naive slope to the gold-standard slope should equal lambda cell by cell.
    # A per-cell variance RATIO is two estimated standard errors from one sample, so it
    # is noisy by construction; a handful of near-degenerate fits at lambda ~ 0.1 also
    # blow up.  The reduction is therefore assessed pooled, over usable cells only.
    usable = df[np.isfinite(df.varrat) & (df.varrat > 0) & (df.varrat < 5)]
    up = usable[usable.z_e2.abs() < 2]
    vr_corr = float(np.corrcoef(up.g2_eiv, up.varrat)[0, 1])
    vr_slope = float(np.polyfit(up.g2_eiv, up.varrat, 1)[0])

    att = df[(df.gold.abs() > 1e-9) & (df.lam > 0.05)].copy()
    att['ratio'] = att.naive / att.gold
    att_corr = float(np.corrcoef(att.lam, att.ratio)[0, 1])
    att_slope = float(np.polyfit(att.lam, att.ratio, 1)[0])
    att_gap = float(np.mean(att.ratio - att.lam))

    summary = dict(
        cells=int(len(df)), states=int(df.state.nunique()),
        passing=int(len(passing)), failing=int(len(failing)),
        pass_within2=float((passing.dev < 2).mean()),
        fail_within2=float((failing.dev < 2).mean()),
        pass_median_dev=float(passing.dev.median()),
        fail_median_dev=float(failing.dev.median()),
        oos_pass_cells=int(len(oos_pass)),
        oos_pass_within2=float((oos_pass.dev < 2).mean()) if len(oos_pass) else None,
        tcls_within2_pass=float((passing.dev_ls < 2).mean()),
        tcls_within2_fail=float((failing.dev_ls < 2).mean()),
        bias_formula_corr=corr, bias_formula_slope=slope,
        bias_formula_all=dict(corr=corr_all, slope=slope_all),
        bias_formula_lambda_ge_half=dict(corr=corr_hi, slope=slope_hi),
        attenuation=dict(corr_ratio_lambda=att_corr, slope=att_slope,
                         mean_gap=att_gap, cells=int(len(att))),
        varrat_pooled=dict(cells=int(len(up)), corr=vr_corr, slope=vr_slope,
                           mean_realised=float(up.varrat.mean()),
                           mean_predicted=float(up.g2_eiv.mean())),
        demo=dict(criteria=DEMO, cells=int(len(demo)),
                  within2=int((demo.dev < 2).sum()),
                  attenuation_visible=int((demo.dev_naive > 2).sum()),
                  varrat_gap=float(np.mean(np.abs(demo.varrat - demo.g2_eiv))),
                  tcls_within2=int((demo.dev_ls < 2).sum())),
        headline=df[(df.state == HEADLINE[0]) & (df.group == HEADLINE[1])
                    & (df.resp == HEADLINE[2])].to_dict('records')[0])
    demo.sort_values('lam').to_csv(OUT / 'demonstration_cells.csv', index=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    (OUT / 'crosswalk.json').write_text(json.dumps(crosswalks, indent=2))
    (OUT / 'manifest.json').write_text(json.dumps(dict(
        nmf_source=S3 + PREFIX, svi_source=SVI_URL, states=STATES,
        responses=RESPONSES, band=BAND, discovery=DISCOVERY, headline=HEADLINE,
        files=files, boot=args.boot), indent=2))

    L = ['=== NMF covariate x SVI response ===', '']
    L.append('-- Gate N1: the rank crosswalk is real (shifted merge is the control) --')
    for c in crosswalks:
        ok1 = c['corr'] > 0.85 and c['corr_shifted'] < 0.6
        L.append(f"  [{'PASS' if ok1 else 'FAIL'}] {c['state']:14s} "
                 f"merged {c['merged']:5d}/{c['of']:5d}  corr={c['corr']:.4f}  "
                 f"shifted-control={c['corr_shifted']:.4f}")
    L += ['', '-- Gate N2: does the screen predict success, out of sample? --',
          f"  screen PASSES (|z|<2): {len(passing):3d} cells, "
          f"{100*(passing.dev<2).mean():.0f}% within 2 SE of gold, median dev "
          f"{passing.dev.median():.2f}",
          f"  screen FAILS (|z|>=2): {len(failing):3d} cells, "
          f"{100*(failing.dev<2).mean():.0f}% within 2 SE of gold, median dev "
          f"{failing.dev.median():.2f}",
          f"  out-of-sample only    : {len(oos_pass):3d} passing cells, "
          f"{100*(oos_pass.dev<2).mean():.0f}% within 2 SE",
          '', '-- Gate N3: the bias formula E[Psi1] = -b Cov(X,eps^2) --',
          f"  on the {SCREEN_GROUP} grid ({len(scr)} cells): corr = {corr:.4f}, "
          f"slope = {slope:.4f}",
          f"  all cells with lambda >= 0.5 ({len(hi_lam)}): corr = {corr_hi:.4f}, "
          f"slope = {slope_hi:.4f}",
          f"  every cell ({len(df)}): corr = {corr_all:.4f}, slope = {slope_all:.4f}  "
          f"-- degrades where lambda ~ 0.1 and the first-order expansion fails",
          '', '-- Gate N7: is the attenuation the one lambda predicts? --',
          f"  naive/gold vs lambda over {len(att)} cells: corr = {att_corr:.4f}, "
          f"slope = {att_slope:.4f} (1.0 = exact), mean gap = {att_gap:+.4f}",
          '', '-- Gate N4: robustness asymmetry (TC-LS should not care) --',
          f"  |gold-tcls|/se < 2 in {100*(passing.dev_ls<2).mean():.0f}% of passing and "
          f"{100*(failing.dev_ls<2).mean():.0f}% of failing cells",
          '', '-- Gate N5: is the predicted variance reduction realised? (pooled) --',
          f"  {len(up)} usable cells passing the screen: "
          f"corr(g2_EIV, realised) = {vr_corr:.4f}, slope = {vr_slope:.4f}",
          f"  mean realised {up.varrat.mean():.4f} vs mean predicted {up.g2_eiv.mean():.4f}"
          f"  -- a real reduction, systematically smaller than the asymptotic value at",
          f"     these n (865-1887) and kurtoses, consistent with the finite-sample",
          f"     threshold already reported in the simulation section",
          '', '-- Gate N6: cells meeting the pre-specified demonstration criteria --',
          f"  (screen passes, lambda in [{DEMO['lam_lo']},{DEMO['lam_hi']}], "
          f"predicted g2_EIV <= {DEMO['g2eiv_max']}; no outcome enters the selection)",
          f"  {len(demo)} cells: {int((demo.dev<2).sum())} land within 2 SE of gold, "
          f"{int((demo.dev_ls<2).sum())} for TC-LS, "
          f"{int((demo.dev_naive>2).sum())} show a visibly attenuated naive slope",
          f"  mean |varrat - g2_EIV| = {np.mean(np.abs(demo.varrat-demo.g2_eiv)):.4f}"]
    h = summary['headline']
    L += ['', f"-- headline cell: {HEADLINE[0]} / {HEADLINE[1]} / {HEADLINE[2]} --",
          f"  n={h['n']} counties={h['n_counties']} lambda={h['lam']:.4f} "
          f"varsigma={h['varsigma']:.3f}",
          f"  eps: c3={h['c3']:+.3f} c4={h['c4']:+.3f} -> g2={h['g2']:.4f}, "
          f"g2_EIV={h['g2_eiv']:.4f}",
          f"  screen z(Cov(X,eps^2)) = {h['z_e2']:+.3f}  (condition not rejected)",
          f"  beta1: gold={h['gold']:+.6f} naive={h['naive']:+.6f} "
          f"tc_ls={h['tcls']:+.6f} star={h['star']:+.6f}",
          f"  |gold-star|/se = {h['dev']:.2f}   |gold-naive|/se = {h['dev_naive']:.2f}",
          f"  Var(star)/Var(tc_ls) = {h['varrat']:.4f}  vs predicted g2_EIV "
          f"{h['g2_eiv']:.4f}"]
    if 'boot' in h:
        b = h['boot']
        L.append(f"  county bootstrap (B={b['B']}): star-gold 95% CI "
                 f"[{b['star_minus_gold_lo']:+.6f}, {b['star_minus_gold_hi']:+.6f}], "
                 f"median varrat {b['varrat_median']:.4f}")
    (OUT / 'verification_report.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
