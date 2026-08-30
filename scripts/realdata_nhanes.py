#!/usr/bin/env python3
"""Real-data application: NHANES 2013--2014 diastolic blood pressure and GGT.

    Y = log(serum gamma-glutamyl transferase),  W = first diastolic reading,
    X = the subject's latent within-visit diastolic level.

The three consecutive diastolic readings of the NHANES blood-pressure examination
(BPXDI1..3, k = 3) identify the central moments nu_2..nu_6 of the measurement
error from the within-subject deviations, which is the information the corrected
degree-two score of this paper needs. A two-replicate design recovers the odd
moments too, but only through the paired cross-moments (Remark on replicate
designs, corrected 2026-08-29; theory/verify_replicate_identification_k2.py):
in the within-subject deviations the coefficient of nu_3 carries a factor k-2.

Everything below is computed from the two public NHANES files; nothing is typed
by hand.  Outputs go to evidence/realdata_nhanes/ and are never hand-edited.

Design facts that govern the inference and are declared in the manuscript:

  * NHANES is a stratified multistage probability sample.  The resampling unit is
    the PSU within stratum (SDMVPSU within SDMVSTRA), never the row -- portfolio
    Verification Standard 9.  Every interval reported here comes from a
    stratified cluster bootstrap over the 30 PSUs in 15 strata.  The point
    estimates are UNWEIGHTED: this is an illustration of the estimator, not a
    population estimate for the United States.
  * nu_3 < 0 here.  The simulation study of this paper never ran a cell with a
    left-skewed measurement error directly; it is covered only through an exact
    sign symmetry of the closed form, which this script verifies numerically.
  * U is not instrument error.  Three readings minutes apart contain short-term
    physiological variation, so X is "diastolic pressure during this visit", not
    "usual blood pressure".
  * Terminal digits are non-uniform (all readings even: a 2 mmHg grid).  The
    script quantifies the share of nu_2, nu_4 and nu_6 attributable to the grid.
  * U independent of X is tested rather than assumed, and it fails.

Usage:  .venv/bin/python scripts/realdata_nhanes.py [--boot 2000]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / 'theory'))

from estimators import (  # noqa: E402
    are_blocks, are_of_t, central_moments, deconvolve_eps_moments, fit_ls,
    fit_tc_ls, fit_tc_pmm2, fit_tc_pmm2_star, optimal_t,
)

DATA = ROOT / 'data' / 'nhanes'
OUT = ROOT / 'evidence' / 'realdata_nhanes'
BASE_URL = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/'

# sha256 recorded in source_notes/REALDATA_CANDIDATES_2026-08-14.md at the time
# of the source scouting.  NHANES occasionally reissues a file; a mismatch means
# a new revision, not a corrupt download, and is reported rather than swallowed.
EXPECTED_SHA256 = {
    'BPX_H.xpt': 'b6641efc30ac9b9de328dfbf4110d0d120439aadca2c2bc89f2e77310d1a0c8e',
    'BIOPRO_H.xpt': 'b114fde7fc46933e6b4e3f620b8e7b61ea5b46f82121c0dc05a7b8c1fbdd0548',
    # DEMO_H carries the survey design variables (SDMVSTRA, SDMVPSU, WTMEC2YR).
    # It was not part of the original scouting note, so no prior checksum exists;
    # the value observed on first download is recorded in the manifest.
}
FILES = ('BPX_H.xpt', 'BIOPRO_H.xpt', 'DEMO_H.xpt')

K_REP = 3                 # three diastolic readings per subject
GRID_MM_HG = 2.0          # recorded resolution of the readings (all even)
Z95 = 1.959963984540054


# ==========================================================================
# 1.  SAS XPORT v5 reader (self-contained: the project depends on numpy+scipy
#     only, and pulling in pandas for one file format is not worth it)
# ==========================================================================
def _ibm_to_ieee(raw: np.ndarray) -> np.ndarray:
    """IBM-360 hexadecimal doubles (n, 8) uint8 -> IEEE float64."""
    b = raw.astype(np.uint64)
    sign = (b[:, 0] >> np.uint64(7)).astype(np.float64)
    expo = (b[:, 0] & np.uint64(0x7F)).astype(np.int64) - 64
    mant = np.zeros(b.shape[0], dtype=np.uint64)
    for i in range(1, 8):
        mant = (mant << np.uint64(8)) | b[:, i]
    val = mant.astype(np.float64) / float(1 << 56) * np.power(16.0, expo.astype(np.float64))
    return np.where(sign > 0, -val, val)


def read_xport(path: Path) -> dict[str, np.ndarray]:
    """Read a single-member SAS XPORT v5 file into {name: array}.

    Numeric SAS missing values (first byte '.', 'A'-'Z' or '_', remaining bytes
    zero) become NaN; a genuine zero is all eight bytes zero and is preserved."""
    data = path.read_bytes()
    if not data.startswith(b'HEADER RECORD*******LIBRARY HEADER RECORD'):
        raise ValueError(f'{path.name}: not a SAS XPORT v5 file')
    i_mem = data.find(b'HEADER RECORD*******MEMBER  HEADER RECORD')
    i_nam = data.find(b'HEADER RECORD*******NAMESTR HEADER RECORD', i_mem)
    nvar = int(data[i_nam + 54:i_nam + 58])
    pos = i_nam + 80
    fields, off = [], 0
    for _ in range(nvar):
        ns = data[pos:pos + 140]
        ntype, _nhfun, nlng, _nvar0 = struct.unpack('>hhhh', ns[0:8])
        fields.append(dict(name=ns[8:16].decode('ascii').strip(), type=ntype,
                           length=nlng, pos=off))
        off += nlng
        pos += 140
    if (140 * nvar) % 80:
        pos += 80 - (140 * nvar) % 80
    pos = data.find(b'HEADER RECORD*******OBS     HEADER RECORD', pos) + 80
    reclen = off
    body = data[pos:]
    nobs = len(body) // reclen
    arr = np.frombuffer(body[:nobs * reclen], dtype=np.uint8).reshape(nobs, reclen)
    while nobs and np.all(arr[nobs - 1] == 0x20):      # trailing padding record
        nobs -= 1
    arr = arr[:nobs]
    out = {}
    for f in fields:
        sl = arr[:, f['pos']:f['pos'] + f['length']]
        if f['type'] == 1:
            pad = np.zeros((nobs, 8), dtype=np.uint8)
            pad[:, :f['length']] = sl
            v = _ibm_to_ieee(pad)
            tail0 = np.all(sl[:, 1:] == 0, axis=1)
            code = ((sl[:, 0] == 0x2E) | ((sl[:, 0] >= 0x41) & (sl[:, 0] <= 0x5A))
                    | (sl[:, 0] == 0x5F))
            out[f['name']] = np.where(tail0 & code, np.nan, v)
        else:
            out[f['name']] = np.array([bytes(r).decode('latin1').strip() for r in sl])
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(fname: str, log: list) -> tuple[Path, str, bool]:
    """Download once and cache; verify the checksum on every run."""
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / fname
    cached = path.exists()
    if not cached:
        with urllib.request.urlopen(BASE_URL + fname, timeout=120) as r, \
                open(path, 'wb') as fh:
            fh.write(r.read())
    digest = sha256(path)
    expected = EXPECTED_SHA256.get(fname)
    ok = expected is None or digest == expected
    log.append(dict(file=fname, url=BASE_URL + fname, bytes=path.stat().st_size,
                    sha256=digest, sha256_expected=expected or '(none recorded)',
                    checksum_ok=int(ok), from_cache=int(cached)))
    return path, digest, ok


# ==========================================================================
# 2.  Replicate identification: E[d^p] as linear forms in nu_2..nu_6
# ==========================================================================
def cum_from_central(mu: dict):
    """Cumulants 2..6 of a zero-mean variable from its central moments.

    Polynomial in its arguments, so it is used with floats and with sympy
    symbols alike."""
    m2, m3, m4, m5, m6 = (mu[r] for r in range(2, 7))
    return {2: m2, 3: m3, 4: m4 - 3 * m2 ** 2, 5: m5 - 10 * m3 * m2,
            6: m6 - 15 * m4 * m2 - 10 * m3 ** 2 + 30 * m2 ** 3}


def central_from_cum(kap: dict):
    """Central moments 0..6 of a zero-mean variable from its cumulants."""
    k2, k3, k4, k5, k6 = (kap[r] for r in range(2, 7))
    return {0: 1, 1: 0, 2: k2, 3: k3, 4: k4 + 3 * k2 ** 2,
            5: k5 + 10 * k3 * k2,
            6: k6 + 15 * k4 * k2 + 10 * k3 ** 2 + 15 * k2 ** 3}


def nu_of_mean(nu: dict, k: int) -> dict:
    """Moments of the error attached to the AVERAGE of k replicate readings:
    kappa_r(Ubar) = kappa_r(U)/k^{r-1}."""
    kap = cum_from_central({r: nu[f'nu{r}'] for r in range(2, 7)})
    cen = central_from_cum({r: kap[r] / k ** (r - 1) for r in range(2, 7)})
    return {f'nu{r}': float(cen[r]) for r in range(2, 7)}


def replicate_system(k: int):
    """Symbolic E[d^p], p = 2..6, for d = W_ij - Wbar_i with k i.i.d. replicates.

    Returns (coef[p], rest_fn[p]) with  E[d^p] = coef[p]*nu_p + rest_p(nu_2..nu_{p-1}),
    so the moments are recovered by forward substitution.  The p <= 5 forms are
    asserted equal to theory/replicate_identification.py, which was written
    independently of this script; p = 6 is the extension this application needs,
    because the optimal weight requires nu_6.
    """
    import sympy as sp
    nu = sp.symbols('nu2 nu3 nu4 nu5 nu6')
    muU = {0: 1, 1: 0, 2: nu[0], 3: nu[1], 4: nu[2], 5: nu[3], 6: nu[4]}
    kapS = {r: (k - 1) * cum_from_central(muU)[r] for r in range(2, 7)}
    muS = central_from_cum(kapS)
    a = sp.Rational(k - 1, k)
    exprs = {}
    for p in range(2, 7):
        exprs[p] = sp.expand(sum(
            sp.binomial(p, r) * a ** (p - r) * (-sp.Rational(1, k)) ** r
            * muU[p - r] * muS[r] for r in range(p + 1)))

    # cross-check against the independently written theory module (p = 2..5)
    import replicate_identification as ri
    ref = dict(ri.symbolic_coefficients(max(k, 6)))[k]
    xcheck = {p: bool(sp.simplify(exprs[p] - ref[p]) == 0) for p in (2, 3, 4, 5)}

    coef, rest = {}, {}
    for p in range(2, 7):
        c = sp.expand(exprs[p]).coeff(nu[p - 2], 1)
        coef[p] = float(c)
        rest[p] = sp.lambdify(nu[:p - 2], sp.expand(exprs[p] - c * nu[p - 2]), 'numpy')
    return coef, rest, xcheck, {p: str(exprs[p]) for p in exprs}


def recover_nu(dev: np.ndarray, coef, rest) -> dict:
    """Forward-substitute nu_2..nu_6 from the empirical moments of the deviations."""
    emp = {p: float((dev ** p).mean()) for p in range(2, 7)}
    nu = []
    for p in range(2, 7):
        nu.append((emp[p] - (rest[p](*nu) if p > 2 else rest[p]())) / coef[p])
    return dict(zip(('nu2', 'nu3', 'nu4', 'nu5', 'nu6'), (float(v) for v in nu)))


# ==========================================================================
# 3.  Sample construction
# ==========================================================================
def build_sample(attrition: list):
    bpx = read_xport(DATA / 'BPX_H.xpt')
    bio = read_xport(DATA / 'BIOPRO_H.xpt')
    dem = read_xport(DATA / 'DEMO_H.xpt')

    seqn = bpx['SEQN']
    rd = np.column_stack([bpx['BPXDI1'], bpx['BPXDI2'], bpx['BPXDI3']])
    step = lambda name, mask, note: attrition.append(          # noqa: E731
        dict(step=name, n=int(mask.sum()), note=note))

    m = np.ones(len(seqn), bool)
    step('BPX_H records', m, 'blood-pressure examination file, one row per subject')
    m1 = m & np.all(np.isfinite(rd), axis=1)
    step('three non-missing diastolic readings', m1,
         'BPXDI1..3 all recorded (k = 3)')
    zero = np.any(rd == 0, axis=1)
    m2 = m1 & ~zero
    step('no zero diastolic reading', m2,
         'a recorded 0 means no fifth Korotkoff sound was heard; it is not a '
         'measurement of diastolic pressure and is dropped')
    idx_bio = {int(s): i for i, s in enumerate(bio['SEQN'])}
    pos = np.array([idx_bio.get(int(s), -1) for s in seqn])
    ggt = np.where(pos >= 0, bio['LBXSGTSI'][np.clip(pos, 0, None)], np.nan)
    m3 = m2 & np.isfinite(ggt)
    step('merged with BIOPRO_H on SEQN, GGT present', m3,
         'LBXSGTSI (serum GGT, U/L); BIOPRO_H covers ages 12+')
    idx_dem = {int(s): i for i, s in enumerate(dem['SEQN'])}
    pd_ = np.array([idx_dem.get(int(s), -1) for s in seqn])
    m4 = m3 & (pd_ >= 0)
    step('survey design variables present', m4,
         'SDMVSTRA, SDMVPSU, WTMEC2YR from DEMO_H')

    d = dict(
        seqn=seqn[m4], readings=rd[m4], y=np.log(ggt[m4]),
        strata=dem['SDMVSTRA'][pd_[m4]].astype(int),
        psu=dem['SDMVPSU'][pd_[m4]].astype(int),
        wt=dem['WTMEC2YR'][pd_[m4]],
        age=dem['RIDAGEYR'][pd_[m4]],
    )
    # the replicate-only sample: every subject with three usable readings,
    # whether or not GGT was measured.  The source-scouting note quoted its
    # moments; keeping it lets this run reproduce those numbers explicitly.
    d['rep_only_readings'] = rd[m2]
    d['rep_only_strata'] = dem['SDMVSTRA'][pd_[m2]].astype(int)
    d['rep_only_psu'] = dem['SDMVPSU'][pd_[m2]].astype(int)
    # two counterfactual subsets, kept only so that the exclusions above can be
    # quantified in evidence rather than asserted in prose
    keep_zero = m1 & np.isfinite(ggt) & (pd_ >= 0)
    d['zeros_kept_readings'] = rd[keep_zero]
    d['dropped_by_merge_readings'] = rd[m2 & ~np.isfinite(ggt)]
    return d


# ==========================================================================
# 4.  One analysis pass (used for the point estimate and for every bootstrap
#     resample, so the resample propagates the moment recovery as well)
# ==========================================================================
def analyse(readings: np.ndarray, y: np.ndarray, coef, rest,
            nu_override: dict | None = None, adjust_reading: bool = True,
            w_col: str = 'first') -> dict:
    """Full pass: reading effect -> nu -> reliability -> five estimators.

    ``w_col='mean'`` replaces the surrogate by the average of the k readings; the
    caller must then pass the matching ``nu_override`` from :func:`nu_of_mean`."""
    if adjust_reading:
        alpha = readings.mean(axis=0) - readings.mean()
    else:
        alpha = np.zeros(readings.shape[1])
    adj = readings - alpha
    dev = (adj - adj.mean(axis=1, keepdims=True)).ravel()
    nu = recover_nu(dev, coef, rest) if nu_override is None else dict(nu_override)

    w = adj.mean(axis=1) if w_col == 'mean' else adj[:, 0]
    var_w = float(np.var(w, ddof=1))
    sx2 = var_w - nu['nu2']
    lam = sx2 / var_w

    out = dict(alpha1=float(alpha[0]), var_w=var_w, sx2=sx2, lam=lam, **nu)
    out['gamma3_u'] = nu['nu3'] / nu['nu2'] ** 1.5 if nu['nu2'] > 0 else np.nan
    out['gamma4_u'] = nu['nu4'] / nu['nu2'] ** 2 - 3.0 if nu['nu2'] > 0 else np.nan

    fits = {}
    fits['naive_ls'] = fit_ls(y, w, 'naive_ls')
    f_tc = fit_tc_ls(y, w, nu['nu2'])
    fits['tc_ls'] = f_tc
    if not np.all(np.isfinite(f_tc.beta)) or sx2 <= 0:
        return out | dict(fits=fits, ok=False)

    r_tc = y - f_tc.beta[0] - f_tc.beta[1] * w
    m2, m3, m4, flags = deconvolve_eps_moments(r_tc, f_tc.beta[1],
                                               nu['nu2'], nu['nu3'], nu['nu4'])
    raw2, raw3, raw4 = central_moments(r_tc)
    c3 = m3 / m2 ** 1.5
    c4 = m4 / m2 ** 2 - 3.0
    out.update(m2=m2, m3=m3, m4=m4, c3=c3, c4=c4,
               g2=1.0 - c3 ** 2 / (2.0 + c4),
               m2_raw=raw2, m3_raw=raw3, m4_raw=raw4, **flags)

    args = (sx2, f_tc.beta[1], m2, m3, m4,
            nu['nu2'], nu['nu3'], nu['nu4'], nu['nu5'], nu['nu6'])
    ell, elg, egg = are_blocks(*args)
    t_opt, g2_eiv, star_fb = optimal_t(*args)
    t_fixed = m3 / (m4 - m2 ** 2)
    out.update(e_phil2=ell, e_philg=elg, e_g2=egg, t_fixed=t_fixed, t_opt=t_opt,
               g2_eiv=g2_eiv, are_fixed=are_of_t(t_fixed, *args),
               star_fallback=star_fb,
               signal_ratio=float(f_tc.beta[1] ** 2 * sx2 / m2))

    fits['tc_pmm2'] = fit_tc_pmm2(y, w, m2, m3, m4, nu['nu2'], nu['nu3'],
                                  f_tc.beta, 'tc_pmm2', flags)
    fits['tc_pmm2star'] = fit_tc_pmm2_star(y, w, m2, m3, m4, nu, sx2,
                                           f_tc.beta, 'tc_pmm2star', flags)
    # "adaptive": a second pass that re-deconvolves the regression-error moments
    # at the TC-PMM2* fit and re-optimises the weight there.  On real data no
    # oracle exists for m_2, m_3, m_4, so this is the honest feasible variant;
    # the pass above is its one-step counterpart, matching what the simulation
    # calls the oracle branch in the sense that the weight is formed once.
    b_st = fits['tc_pmm2star'].beta
    if np.all(np.isfinite(b_st)):
        r2 = y - b_st[0] - b_st[1] * w
        am2, am3, am4, fl2 = deconvolve_eps_moments(r2, b_st[1], nu['nu2'],
                                                    nu['nu3'], nu['nu4'])
        fits['tc_pmm2star_adaptive'] = fit_tc_pmm2_star(
            y, w, am2, am3, am4, nu, sx2, b_st, 'tc_pmm2star_adaptive', fl2)
        out['t_opt_adaptive'] = fits['tc_pmm2star_adaptive'].info.get('star_t', np.nan)
        out['g2_eiv_adaptive'] = fits['tc_pmm2star_adaptive'].info.get('star_g2_eiv',
                                                                      np.nan)
    else:
        fits['tc_pmm2star_adaptive'] = fits['tc_pmm2star']
        out['t_opt_adaptive'] = np.nan
        out['g2_eiv_adaptive'] = np.nan
    return out | dict(fits=fits, ok=True)


EST_ORDER = ('naive_ls', 'tc_ls', 'tc_pmm2', 'tc_pmm2star', 'tc_pmm2star_adaptive')


# ==========================================================================
# 5.  Stratified cluster bootstrap over PSUs within strata
# ==========================================================================
def cluster_bootstrap(d, coef, rest, b_reps: int, seed: int):
    """Resample PSUs with replacement within each stratum (Verification
    Standard 9: the resampling unit is the unit that generates the dependence).

    NHANES 2013--2014 has 15 strata with 2 PSUs each; the design degrees of
    freedom are therefore 30 - 15 = 15, and every interval below inherits that.
    """
    rng = np.random.default_rng(seed)
    keys = np.stack([d['strata'], d['psu']], axis=1)
    strata = np.unique(d['strata'])
    members = {}
    for h in strata:
        psus = np.unique(d['psu'][d['strata'] == h])
        members[h] = [np.flatnonzero((d['strata'] == h) & (d['psu'] == p))
                      for p in psus]
    rows = []
    t0 = time.time()
    for b in range(b_reps):
        take = []
        for h in strata:
            ms = members[h]
            pick = rng.integers(0, len(ms), size=len(ms))
            take.extend(ms[j] for j in pick)
        idx = np.concatenate(take)
        res = analyse(d['readings'][idx], d['y'][idx], coef, rest)
        row = dict(rep=b, n=idx.size, lam=res['lam'], sx2=res['sx2'],
                   nu2=res['nu2'], nu3=res['nu3'], nu4=res['nu4'],
                   nu5=res['nu5'], nu6=res['nu6'],
                   gamma3_u=res['gamma3_u'],
                   g2=res.get('g2', np.nan), g2_eiv=res.get('g2_eiv', np.nan),
                   t_fixed=res.get('t_fixed', np.nan),
                   t_opt=res.get('t_opt', np.nan),
                   star_fallback=res.get('star_fallback', -1),
                   adapt_clip=res.get('adapt_clip', -1))
        for nm in EST_ORDER:
            f = res['fits'].get(nm)
            row[f'b0_{nm}'] = f.beta[0] if f else np.nan
            row[f'b1_{nm}'] = f.beta[1] if f else np.nan
            row[f'conv_{nm}'] = int(f.converged) if f else 0
        rows.append(row)
    return rows, time.time() - t0, int(len(strata)), int(sum(len(v) for v in members.values())), keys


# ==========================================================================
# 6.  Diagnostics
# ==========================================================================
def reading_effects(readings, strata, psu):
    """Test the reading-number fixed effect with PSU-clustered standard errors."""
    out = []
    for a, b in ((0, 1), (1, 2), (0, 2)):
        diff = readings[:, a] - readings[:, b]
        se_iid = diff.std(ddof=1) / np.sqrt(diff.size)
        se_cl = _cluster_se_mean(diff, strata, psu)
        out.append(dict(contrast=f'reading {a + 1} - reading {b + 1}',
                        mean=float(diff.mean()),
                        se_iid=float(se_iid), t_iid=float(diff.mean() / se_iid),
                        se_cluster=float(se_cl),
                        t_cluster=float(diff.mean() / se_cl)))
    return out


def _cluster_se_mean(v, strata, psu):
    """SE of a mean under stratified two-PSU-per-stratum sampling (linearised)."""
    tot = 0.0
    n = v.size
    for h in np.unique(strata):
        mh = strata == h
        psus = np.unique(psu[mh])
        sums = np.array([v[mh & (psu == p)].sum() for p in psus])
        cnt = len(psus)
        if cnt < 2:
            continue
        tot += cnt / (cnt - 1.0) * ((sums - sums.mean()) ** 2).sum()
    return float(np.sqrt(tot) / n)


def digit_diagnostics(readings):
    v = readings.ravel()
    digits = np.bincount((v % 10).astype(int), minlength=10)
    share = digits / digits.sum()
    even = share[::2]
    chi2 = float(((even - 0.2) ** 2 / 0.2).sum() * digits.sum())
    # additive-grid model: an independent rounding error R, uniform on the
    # 2 mmHg grid, contributes nu2(R) = h^2/12, nu4(R) = h^4/80, nu6(R) = h^6/448
    h = GRID_MM_HG
    return dict(share=share, chi2_even_uniform=chi2,
                nu2_R=h ** 2 / 12.0, nu4_R=h ** 4 / 80.0, nu6_R=h ** 6 / 448.0,
                odd_share=float(share[1::2].sum()))


def independence_checks(readings, y, age, nu2_hat):
    """Is the measurement-error variance free of the pressure level?

    Two views.  (a) A level proxy independent of the error being measured:
    E[(W_2 - W_3)^2 | X] = 2 nu_2, and reading 1 is independent of U_2, U_3, so
    regressing the pair variance on reading 1 is contaminated only through the
    attenuation of the proxy, not through the statistic itself.  (b) Age, which
    is exogenous to the measurement process altogether.
    """
    w1 = readings[:, 0]
    pair_var = (readings[:, 1] - readings[:, 2]) ** 2 / 2.0
    des = np.column_stack([np.ones(w1.size), w1 - w1.mean()])
    beta, *_ = np.linalg.lstsq(des, pair_var, rcond=None)
    resid = pair_var - des @ beta
    xtx_inv = np.linalg.inv(des.T @ des)
    v = xtx_inv @ (des * resid[:, None]).T @ (des * resid[:, None]) @ xtx_inv
    slope, slope_se = float(beta[1]), float(np.sqrt(v[1, 1]))

    rows = []
    q = np.quantile(w1, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        sel = (w1 >= q[i]) & (w1 < q[i + 1]) if i < 4 else (w1 >= q[i])
        pv = pair_var[sel]
        rows.append(dict(stratum=f'reading-1 quintile {i + 1}', n=int(sel.sum()),
                         level=float(w1[sel].mean()),
                         nu2_hat=float(pv.mean()),
                         se=float(pv.std(ddof=1) / np.sqrt(sel.sum()))))
    for lo, hi in ((12, 29), (30, 44), (45, 59), (60, 100)):
        sel = (age >= lo) & (age <= hi)
        if sel.sum() < 50:
            continue
        pv = pair_var[sel]
        rows.append(dict(stratum=f'age {lo}-{min(hi, 80)}', n=int(sel.sum()),
                         level=float(w1[sel].mean()),
                         nu2_hat=float(pv.mean()),
                         se=float(pv.std(ddof=1) / np.sqrt(sel.sum()))))
    return dict(slope=slope, slope_se=slope_se, slope_t=slope / slope_se,
                rows=rows, nu2_hat_overall=float(pair_var.mean()),
                nu2_replicate=nu2_hat)


def epsilon_checks(y, w):
    """Is the regression error free of the covariate?

    The degree-two score needs $\\E[e^2\\mid X] = m_2$ and $\\E[e^3\\mid X] = m_3$;
    only the first moment condition is needed by least squares.  We regress the
    squared and cubed naive residual on the surrogate with heteroskedasticity-
    robust standard errors, and tabulate the residual moments by level."""
    des = np.column_stack([np.ones(w.size), w - w.mean()])
    beta = np.linalg.lstsq(des, y, rcond=None)[0]
    r = y - des @ beta
    out = {}
    xtx = np.linalg.inv(des.T @ des)
    for p, key in ((2, 'sq'), (3, 'cube')):
        v = r ** p
        b = np.linalg.lstsq(des, v, rcond=None)[0]
        e = v - des @ b
        m = des * e[:, None]
        vv = xtx @ (m.T @ m) @ xtx
        out[key] = dict(slope=float(b[1]), se=float(np.sqrt(vv[1, 1])),
                        t=float(b[1] / np.sqrt(vv[1, 1])), mean=float(v.mean()))
    rows = []
    q = np.quantile(w, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        sel = (w >= q[i]) & (w < q[i + 1]) if i < 4 else (w >= q[i])
        rr = r[sel]
        m2, m3, m4 = central_moments(rr)
        c3, c4 = m3 / m2 ** 1.5, m4 / m2 ** 2 - 3.0
        rows.append(dict(quintile=i + 1, n=int(sel.sum()),
                         level=float(w[sel].mean()), m2=float(m2),
                         c3=float(c3), c4=float(c4),
                         g2=float(1.0 - c3 ** 2 / (2.0 + c4))))
    out['by_level'] = rows
    return out


def weighted_slopes(y, w, wt, nu2):
    """Design-weighted naive LS and corrected LS, for the declaration that the
    unweighted analysis is an illustration rather than a population estimate."""
    sw = wt.sum()
    des = np.column_stack([np.ones(w.size), w])
    a = des.T @ (des * wt[:, None])
    b = des.T @ (y * wt)
    beta_n = np.linalg.solve(a, b)
    a_c = a - sw * np.diag([0.0, nu2])
    beta_c = np.linalg.solve(a_c, b)
    return dict(naive_ls_weighted=float(beta_n[1]), tc_ls_weighted=float(beta_c[1]))


def sign_symmetry_check(res):
    """(beta1, nu3, nu5) -> (-beta1, -nu3, -nu5) leaves the closed-form efficiency
    invariant.  This is what makes the beta1 = -1 cells of the simulation, run
    with a right-skewed U, informative about the left-skewed U met here."""
    b1 = res['fits']['tc_ls'].beta[1]
    a = (res['sx2'], b1, res['m2'], res['m3'], res['m4'],
         res['nu2'], res['nu3'], res['nu4'], res['nu5'], res['nu6'])
    a_flip = (res['sx2'], -b1, res['m2'], res['m3'], res['m4'],
              res['nu2'], -res['nu3'], res['nu4'], -res['nu5'], res['nu6'])
    g_a, g_b = optimal_t(*a)[1], optimal_t(*a_flip)[1]
    f_a = are_of_t(res['t_fixed'], *a)
    f_b = are_of_t(res['t_fixed'], *a_flip)
    return dict(g2_eiv=float(g_a), g2_eiv_flipped=float(g_b),
                abs_diff_star=float(abs(g_a - g_b)),
                are_fixed=float(f_a), are_fixed_flipped=float(f_b),
                abs_diff_fixed=float(abs(f_a - f_b)))


# ==========================================================================
# 7.  Output helpers
# ==========================================================================
def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def git_hash() -> str:
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                       text=True).strip()
    except Exception:
        return 'unavailable'


def pct(x, lo=2.5, hi=97.5):
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, lo)), float(np.percentile(a, hi))) if a.size \
        else (np.nan, np.nan)


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20260815)
    args = ap.parse_args()
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    gates = []

    def gate(name, ok, detail):
        gates.append((name, bool(ok), detail))

    # ---- data -----------------------------------------------------------
    dl = []
    checks = [fetch(f, dl)[2] for f in FILES]
    gate('R0 file checksums', all(checks),
         '; '.join(f"{r['file']}: {'match' if r['checksum_ok'] else 'MISMATCH'}"
                   for r in dl))
    attrition = []
    d = build_sample(attrition)
    n = d['y'].size
    write_csv(OUT / 'attrition.csv', attrition)

    # ---- replicate identification system --------------------------------
    coef, rest, xcheck, exprs = replicate_system(K_REP)
    gate('R1 replicate system vs theory/replicate_identification.py',
         all(xcheck.values()),
         'symbolic identity of E[d^p] for p = 2,3,4,5: '
         + ', '.join(f'p={p}: {"ok" if v else "DIFFERS"}' for p, v in xcheck.items())
         + f'; p=6 is the extension needed for nu_6, coefficient {coef[6]:.6f}')
    gate('R2 all six moments identified at k = 3',
         all(abs(coef[p]) > 1e-12 for p in range(2, 7)),
         'coefficients of nu_p in E[d^p]: '
         + ', '.join(f'{p}: {coef[p]:.4f}' for p in range(2, 7))
         + ' (every one non-zero, so nu_2..nu_6 are all identified; at k = 2 the '
           'coefficients of nu_3 and nu_5 are exactly zero)')

    # ---- point analysis --------------------------------------------------
    res_adj = analyse(d['readings'], d['y'], coef, rest, adjust_reading=True)
    res_raw = analyse(d['readings'], d['y'], coef, rest, adjust_reading=False)
    # the replicate-only sample quoted by the source-scouting note
    rep_dev_adj = d['rep_only_readings'] - (d['rep_only_readings'].mean(axis=0)
                                            - d['rep_only_readings'].mean())
    rep_dev_adj = (rep_dev_adj - rep_dev_adj.mean(axis=1, keepdims=True)).ravel()
    rep_dev_raw = (d['rep_only_readings']
                   - d['rep_only_readings'].mean(axis=1, keepdims=True)).ravel()
    nu_rep_raw = recover_nu(rep_dev_raw, coef, rest)
    nu_rep_adj = recover_nu(rep_dev_adj, coef, rest)
    vw_rep = float(np.var(d['rep_only_readings'][:, 0], ddof=1))

    gate('R3 TC-PMM2* weight fallback never fired',
         res_adj['star_fallback'] == 0 and res_adj.get('adapt_clip', 0) == 0,
         f"star_fallback = {res_adj['star_fallback']}, cumulant-deconvolution "
         f"clip = {res_adj.get('adapt_clip')}")
    gate('R4 Cauchy-Schwarz bound g2_EIV <= 1', res_adj['g2_eiv'] <= 1.0,
         f"g2_EIV = {res_adj['g2_eiv']:.6f}")
    sym = sign_symmetry_check(res_adj)
    gate('R5 sign symmetry (beta1, nu3, nu5) -> (-beta1, -nu3, -nu5)',
         sym['abs_diff_star'] < 1e-12 and sym['abs_diff_fixed'] < 1e-12,
         f"|difference| = {sym['abs_diff_star']:.2e} (optimal weight), "
         f"{sym['abs_diff_fixed']:.2e} (transplanted weight)")

    # ---- bootstrap -------------------------------------------------------
    boot, wall_boot, n_strata, n_psu, _ = cluster_bootstrap(
        d, coef, rest, args.boot, args.seed)
    with gzip.open(OUT / 'bootstrap_raw.csv.gz', 'wt', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(boot[0].keys()))
        wr.writeheader()
        wr.writerows(boot)
    B = np.array([[r[f'b1_{nm}'] for nm in EST_ORDER] for r in boot], float)
    conv = np.array([[r[f'conv_{nm}'] for nm in EST_ORDER] for r in boot], int)
    gate('R6 bootstrap convergence', bool((conv == 1).all()),
         f'all {args.boot} resamples converged for all {len(EST_ORDER)} estimators'
         if (conv == 1).all() else 'some resamples failed to converge')

    # a deliberately wrong interval, for the record: resampling rows instead of
    # PSUs.  Reported to show how much Verification Standard 9 is worth here.
    rng_row = np.random.default_rng(args.seed + 1)
    row_boot = []
    for _ in range(min(args.boot, 500)):
        idx = rng_row.integers(0, n, n)
        f = fit_tc_ls(d['y'][idx], (d['readings'] - (d['readings'].mean(axis=0)
                                                     - d['readings'].mean()))[idx, 0],
                      res_adj['nu2'])
        row_boot.append(f.beta[1])

    # ---- estimator table -------------------------------------------------
    est_rows = []
    for j, nm in enumerate(EST_ORDER):
        f = res_adj['fits'][nm]
        col = B[:, j]
        lo, hi = pct(col)
        est_rows.append(dict(
            estimator=nm, beta0=float(f.beta[0]), beta1=float(f.beta[1]),
            se_sandwich=float(f.se[1]), se_cluster=float(np.nanstd(col, ddof=1)),
            ci_lo=lo, ci_hi=hi,
            ci_lo_t=float(f.beta[1] - Z95 * np.nanstd(col, ddof=1)),
            ci_hi_t=float(f.beta[1] + Z95 * np.nanstd(col, ddof=1)),
            converged=int(f.converged), iterations=int(f.iterations),
            var_ratio_sandwich=float((f.se[1] / res_adj['fits']['tc_ls'].se[1]) ** 2),
            var_ratio_cluster=float(np.nanvar(col, ddof=1)
                                    / np.nanvar(B[:, 1], ddof=1)),
            resampling_unit='PSU within stratum',
            n_psu=n_psu, n_strata=n_strata, n_obs=n, b_reps=args.boot))
    write_csv(OUT / 'estimates.csv', est_rows)

    # paired efficiency of TC-PMM2* against TC-LS, same resamples
    j_star, j_tcls, j_fix = EST_ORDER.index('tc_pmm2star'), 1, EST_ORDER.index('tc_pmm2')
    ratio_star = float(np.nanvar(B[:, j_star], ddof=1) / np.nanvar(B[:, j_tcls], ddof=1))
    ratio_fix = float(np.nanvar(B[:, j_fix], ddof=1) / np.nanvar(B[:, j_tcls], ddof=1))
    # Two different uncertainties attach to a variance ratio and they must not be
    # confused.  (i) Monte Carlo noise of the bootstrap variances themselves,
    # obtained by resampling the resamples -- small and uninteresting.  (ii) The
    # sampling uncertainty of the ratio, which with 15 design degrees of freedom
    # is what actually limits the claim.  For (ii) we delete one stratum at a
    # time and repeat the whole cluster bootstrap on the remaining 14 strata.
    rng2 = np.random.default_rng(args.seed + 2)
    idxs = rng2.integers(0, args.boot, size=(1000, args.boot))
    ratio_star_mc = pct(np.nanvar(B[:, j_star][idxs], axis=1, ddof=1)
                        / np.nanvar(B[:, j_tcls][idxs], axis=1, ddof=1))
    ratio_fix_mc = pct(np.nanvar(B[:, j_fix][idxs], axis=1, ddof=1)
                       / np.nanvar(B[:, j_tcls][idxs], axis=1, ddof=1))
    jack_star, jack_fix = [], []
    strata_all = np.unique(d['strata'])
    b_jack = max(200, args.boot // 10)
    for h in strata_all:
        keep = d['strata'] != h
        dj = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[0] == n else v)
              for k, v in d.items()}
        bj, _, _, _, _ = cluster_bootstrap(dj, coef, rest, b_jack,
                                           args.seed + 100 + int(h))
        Bj = np.array([[r[f'b1_{nm}'] for nm in EST_ORDER] for r in bj], float)
        jack_star.append(np.nanvar(Bj[:, j_star], ddof=1)
                         / np.nanvar(Bj[:, j_tcls], ddof=1))
        jack_fix.append(np.nanvar(Bj[:, j_fix], ddof=1)
                        / np.nanvar(Bj[:, j_tcls], ddof=1))

    def jack_ci(vals, point):
        v = np.asarray(vals, float)
        g = len(v)
        se = float(np.sqrt((g - 1) / g * ((v - v.mean()) ** 2).sum()))
        return (point - Z95 * se, point + Z95 * se, se)

    ratio_star_ci = jack_ci(jack_star, ratio_star)[:2]
    ratio_fix_ci = jack_ci(jack_fix, ratio_fix)[:2]
    ratio_star_se = jack_ci(jack_star, ratio_star)[2]

    # ---- moments table ---------------------------------------------------
    def nu_row(tag, sample, nn, src, vw, lam, extra=None):
        row = dict(sample=tag, n_subjects=nn, adjustment=src, var_w1=vw, lam=lam)
        row.update({k: float(sample[k]) for k in
                    ('nu2', 'nu3', 'nu4', 'nu5', 'nu6')})
        row['sd_u'] = float(np.sqrt(sample['nu2']))
        row['gamma3_u'] = float(sample['nu3'] / sample['nu2'] ** 1.5)
        row['gamma4_u'] = float(sample['nu4'] / sample['nu2'] ** 2 - 3.0)
        row['nu5_std'] = float(sample['nu5'] / sample['nu2'] ** 2.5)
        row['nu6_std'] = float(sample['nu6'] / sample['nu2'] ** 3)
        row.update(extra or {})
        return row

    mom_rows = [
        nu_row('regression sample', res_raw, n, 'raw (readings as recorded)',
               res_raw['var_w'], res_raw['lam']),
        nu_row('regression sample', res_adj, n,
               'reading-number means removed', res_adj['var_w'], res_adj['lam']),
        nu_row('replicate-only sample', nu_rep_raw,
               int(d['rep_only_readings'].shape[0]), 'raw (readings as recorded)',
               vw_rep, 1.0 - nu_rep_raw['nu2'] / vw_rep),
        nu_row('replicate-only sample', nu_rep_adj,
               int(d['rep_only_readings'].shape[0]),
               'reading-number means removed', vw_rep,
               1.0 - nu_rep_adj['nu2'] / vw_rep),
    ]
    write_csv(OUT / 'moments.csv', mom_rows)

    # bootstrap intervals for the moments (PSU resampling, regression sample)
    mom_ci = []
    for key in ('nu2', 'nu3', 'nu4', 'nu5', 'nu6', 'gamma3_u', 'lam', 'g2',
                'g2_eiv', 't_fixed', 't_opt'):
        col = np.array([r[key] for r in boot], float)
        lo, hi = pct(col)
        mom_ci.append(dict(quantity=key, point=float(res_adj[key]),
                           boot_mean=float(np.nanmean(col)),
                           boot_sd=float(np.nanstd(col, ddof=1)),
                           ci_lo=lo, ci_hi=hi,
                           excludes_zero=int(lo * hi > 0),
                           rel_width=float((hi - lo) / abs(res_adj[key]))
                           if res_adj[key] != 0 else np.nan))
    write_csv(OUT / 'moment_intervals.csv', mom_ci)

    # ---- diagnostics -----------------------------------------------------
    rd_eff = reading_effects(d['readings'], d['strata'], d['psu'])
    dig = digit_diagnostics(d['readings'])
    ind = independence_checks(d['readings'], d['y'], d['age'], res_adj['nu2'])
    w_adj = (d['readings'] - (d['readings'].mean(axis=0)
                              - d['readings'].mean()))[:, 0]
    eps_chk = epsilon_checks(d['y'], w_adj)
    wsl = weighted_slopes(d['y'], w_adj, d['wt'], res_adj['nu2'])

    # grid (digit) sensitivity: subtract the 2 mmHg rounding contribution
    nu_g = {
        'nu2': res_adj['nu2'] - dig['nu2_R'],
        'nu3': res_adj['nu3'],
        'nu4': res_adj['nu4'] - 6.0 * (res_adj['nu2'] - dig['nu2_R']) * dig['nu2_R']
               - dig['nu4_R'],
        'nu5': res_adj['nu5'],
    }
    nu_g['nu6'] = (res_adj['nu6']
                   - 15.0 * nu_g['nu4'] * dig['nu2_R']
                   - 15.0 * nu_g['nu2'] * dig['nu4_R'] - dig['nu6_R'])
    res_grid = analyse(d['readings'], d['y'], coef, rest, nu_override=nu_g)
    res_rep = analyse(d['readings'], d['y'], coef, rest, nu_override=nu_rep_adj)
    # surrogate = average of the three readings: the same data with the
    # measurement-error variance divided by k, hence with any conditional-moment
    # misspecification of U scaled down by the same factor
    nu_bar = nu_of_mean(dict(nu2=res_adj['nu2'], nu3=res_adj['nu3'],
                             nu4=res_adj['nu4'], nu5=res_adj['nu5'],
                             nu6=res_adj['nu6']), K_REP)
    res_bar = analyse(d['readings'], d['y'], coef, rest, nu_override=nu_bar,
                      w_col='mean')

    # the two corrected estimators are consistent for the same beta_1 under
    # correct specification, so their difference is an over-identification test
    gap = B[:, EST_ORDER.index('tc_ls')] - B[:, EST_ORDER.index('tc_pmm2star')]
    gap_pt = (res_adj['fits']['tc_ls'].beta[1]
              - res_adj['fits']['tc_pmm2star'].beta[1])
    gap_sd = float(np.nanstd(gap, ddof=1))
    gap_bar = (res_bar['fits']['tc_ls'].beta[1]
               - res_bar['fits']['tc_pmm2star'].beta[1])
    gate('R8 measurement error independent of the covariate',
         abs(ind['slope_t']) < 3.0,
         f"the pair variance (W_2-W_3)^2/2 has slope {ind['slope']:+.4f} "
         f"(SE {ind['slope_se']:.4f}, t = {ind['slope_t']:.2f}) on reading 1; "
         'nu_2 falls from '
         + f"{ind['rows'][0]['nu2_hat']:.1f} to {ind['rows'][4]['nu2_hat']:.1f} "
         'across reading-1 quintiles. Reading 1 is independent of the errors in '
         'readings 2 and 3, so this is not a mechanical artefact')
    gate('R9 regression error independent of the covariate',
         abs(eps_chk['sq']['t']) < 3.0,
         f"the squared naive residual has slope {eps_chk['sq']['slope']:+.5f} "
         f"(t = {eps_chk['sq']['t']:.2f}) on the surrogate; the cubed residual "
         f"t = {eps_chk['cube']['t']:.2f}. Least squares tolerates this; the "
         'quadratic moment condition of the degree-two score does not')
    gate('R7 corrected LS and TC-PMM2* agree on the slope',
         abs(gap_pt / gap_sd) < 3.0,
         f'difference {gap_pt:.6f} with cluster-bootstrap SE {gap_sd:.6f}, '
         f'z = {gap_pt / gap_sd:.2f}. Under the model of this paper both are '
         'consistent for beta_1, so a large z is a specification signal, not an '
         'efficiency result. It is reported, not suppressed.')

    diag_rows = [dict(check=r['contrast'], value=r['mean'],
                      se=r['se_cluster'], stat=r['t_cluster'],
                      note=f"t = {r['t_iid']:.3f} treating rows as independent; "
                           f"t = {r['t_cluster']:.3f} with PSU clustering")
                 for r in rd_eff]
    diag_rows += [
        dict(check='terminal digit share, even digits 0/2/4/6/8',
             value=float(dig['share'][::2].sum()), se=np.nan,
             stat=dig['chi2_even_uniform'],
             note='shares ' + '/'.join(f'{100 * s:.2f}' for s in dig['share'][::2])
                  + f" per cent; odd digits {100 * dig['odd_share']:.2f} per cent; "
                    'chi-square against a uniform distribution over the five even '
                    'digits, 4 df'),
        dict(check='nu_2 attributable to the 2 mmHg recording grid',
             value=dig['nu2_R'], se=np.nan,
             stat=100 * dig['nu2_R'] / res_adj['nu2'],
             note='additive uniform rounding model; per cent of nu_2 in the stat '
                  'column'),
        dict(check='nu_4 attributable to the grid',
             value=6.0 * res_adj['nu2'] * dig['nu2_R'] + dig['nu4_R'], se=np.nan,
             stat=100 * (6.0 * res_adj['nu2'] * dig['nu2_R'] + dig['nu4_R'])
                  / res_adj['nu4'], note='per cent of nu_4 in the stat column'),
        dict(check='nu_6 attributable to the grid',
             value=15.0 * res_adj['nu4'] * dig['nu2_R']
                   + 15.0 * res_adj['nu2'] * dig['nu4_R'] + dig['nu6_R'],
             se=np.nan,
             stat=100 * (15.0 * res_adj['nu4'] * dig['nu2_R']
                         + 15.0 * res_adj['nu2'] * dig['nu4_R'] + dig['nu6_R'])
                  / res_adj['nu6'], note='per cent of nu_6 in the stat column'),
        dict(check='slope of grid-corrected moments on beta1(TC-PMM2*)',
             value=res_grid['fits']['tc_pmm2star'].beta[1], se=np.nan,
             stat=res_grid['fits']['tc_pmm2star'].beta[1]
                  - res_adj['fits']['tc_pmm2star'].beta[1],
             note='TC-PMM2* refitted with the grid contribution removed from '
                  'nu_2, nu_4, nu_6; stat column is the change in the slope'),
        dict(check='pair variance (W2-W3)^2/2 on reading 1',
             value=ind['slope'], se=ind['slope_se'], stat=ind['slope_t'],
             note='mmHg^2 per mmHg; a non-zero slope means the measurement-error '
                  'variance depends on the pressure level, contradicting U '
                  'independent of X'),
        dict(check='naive LS slope, design-weighted',
             value=wsl['naive_ls_weighted'], se=np.nan,
             stat=wsl['naive_ls_weighted'] - res_adj['fits']['naive_ls'].beta[1],
             note='WTMEC2YR weights; stat column is the difference from the '
                  'unweighted estimate'),
        dict(check='corrected LS slope, design-weighted',
             value=wsl['tc_ls_weighted'], se=np.nan,
             stat=wsl['tc_ls_weighted'] - res_adj['fits']['tc_ls'].beta[1],
             note='WTMEC2YR weights; stat column is the difference from the '
                  'unweighted estimate'),
        dict(check='row-level bootstrap SE of the TC-LS slope', value=np.nan,
             se=float(np.std(row_boot, ddof=1)),
             stat=float(np.std(row_boot, ddof=1)
                        / np.nanstd(B[:, 1], ddof=1)),
             note='WRONG resampling unit, reported only for contrast; stat column '
                  'is its ratio to the PSU cluster bootstrap SE'),
        dict(check='sign symmetry of the closed-form efficiency',
             value=sym['g2_eiv'], se=np.nan, stat=sym['abs_diff_star'],
             note='g2_EIV under (beta1, nu3, nu5) and under '
                  '(-beta1, -nu3, -nu5); stat column is the absolute difference'),
        dict(check='signal ratio beta1^2 sigma_x^2 / m_2',
             value=res_adj['signal_ratio'], se=np.nan, stat=np.nan,
             note='the dimensionless quantity that controls how far g2_EIV can '
                  'depart from g2; it is the regression signal-to-noise ratio, '
                  'not the reliability'),
        dict(check='TC-LS minus TC-PMM2* slope (over-identification contrast)',
             value=float(gap_pt), se=gap_sd, stat=float(gap_pt / gap_sd),
             note='both are consistent for beta_1 under the model, so this is a '
                  'specification test; PSU cluster bootstrap, B = %d' % args.boot),
        dict(check='same contrast with the surrogate = mean of the three readings',
             value=float(gap_bar), se=np.nan,
             stat=float(gap_bar / gap_pt) if gap_pt else np.nan,
             note='averaging the replicates divides nu_2 by k = 3 and so scales '
                  'down any conditional-moment failure of U; stat column is the '
                  'ratio of the two contrasts. Reliability rises to %.4f'
                  % res_bar['lam']),
        dict(check='TC-PMM2* slope with the surrogate = mean of three readings',
             value=float(res_bar['fits']['tc_pmm2star'].beta[1]), se=np.nan,
             stat=float(res_bar['g2_eiv']),
             note='stat column is g_2^EIV for that surrogate'),
        dict(check='TC-PMM2* slope with the replicate-only-sample moments',
             value=res_rep['fits']['tc_pmm2star'].beta[1], se=np.nan,
             stat=res_rep['lam'],
             note='nu_2..nu_6 taken from all subjects with three usable readings '
                  'rather than from the regression sample; stat column is the '
                  'implied reliability'),
    ]
    def nu_of_readings(rr):
        a = rr - (rr.mean(axis=0) - rr.mean())
        return recover_nu((a - a.mean(axis=1, keepdims=True)).ravel(), coef, rest)

    nu_zeros = nu_of_readings(d['zeros_kept_readings'])
    nu_dropped = nu_of_readings(d['dropped_by_merge_readings'])
    diag_rows += [
        dict(check='nu_2 if zero diastolic readings were retained',
             value=nu_zeros['nu2'], se=np.nan,
             stat=nu_zeros['nu3'] / nu_zeros['nu2'] ** 1.5,
             note='n = %d; stat column is the implied gamma_3(U). A recorded 0 '
                  'means no fifth Korotkoff sound was audible, so retaining '
                  'those subjects would manufacture almost the whole skewness '
                  'signal' % d['zeros_kept_readings'].shape[0]),
        dict(check='nu_2 among the subjects removed by the GGT merge',
             value=nu_dropped['nu2'], se=np.nan,
             stat=nu_dropped['nu3'] / nu_dropped['nu2'] ** 1.5,
             note='n = %d, mostly ages 8-11 (the biochemistry file starts at 12); '
                  'their measurement error is larger than the regression '
                  "sample's, which is why nu is recovered from the regression "
                  'sample itself' % d['dropped_by_merge_readings'].shape[0]),
        dict(check='squared naive residual regressed on the surrogate',
             value=eps_chk['sq']['slope'], se=eps_chk['sq']['se'],
             stat=eps_chk['sq']['t'],
             note='the degree-two score needs E[e^2|X] constant; a non-zero slope '
                  'means the regression error is not independent of the covariate, '
                  'which least squares tolerates and the quadratic moment '
                  'condition does not'),
        dict(check='cubed naive residual regressed on the surrogate',
             value=eps_chk['cube']['slope'], se=eps_chk['cube']['se'],
             stat=eps_chk['cube']['t'],
             note='the same for E[e^3|X]'),
    ]
    for r in eps_chk['by_level']:
        diag_rows.append(dict(check=f"m_2 of the naive residual, quintile {r['quintile']}",
                              value=r['m2'], se=np.nan, stat=r['level'],
                              note=f"c_3 = {r['c3']:.4f}, c_4 = {r['c4']:.4f}, "
                                   f"implied g_2 = {r['g2']:.4f}; stat column is "
                                   'the mean surrogate value in the quintile'))
    for r in ind['rows']:
        diag_rows.append(dict(check=f"nu_2 within {r['stratum']}",
                              value=r['nu2_hat'], se=r['se'], stat=r['level'],
                              note='half the mean squared difference of readings 2 '
                                   'and 3; stat column is the mean of reading 1'))
    write_csv(OUT / 'diagnostics.csv', diag_rows)

    # ---- independence profile (figure panel) -----------------------------
    ind_rows = []
    for r_nu, r_eps in zip(ind['rows'][:5], eps_chk['by_level']):
        ind_rows.append(dict(quintile=r_eps['quintile'], n=r_eps['n'],
                             level=r_eps['level'],
                             nu2=r_nu['nu2_hat'], nu2_se=r_nu['se'],
                             nu2_rel=r_nu['nu2_hat'] / ind['nu2_hat_overall'],
                             m2=r_eps['m2'], m2_rel=r_eps['m2'] / res_adj['m2'],
                             c3=r_eps['c3'], g2=r_eps['g2']))
    write_csv(OUT / 'independence.csv', ind_rows)

    # ---- efficiency comparison ------------------------------------------
    eff_rows = [
        dict(quantity='g_2 (error-free Kunchenko coefficient)',
             value=res_adj['g2'], ci_lo=pct([r['g2'] for r in boot])[0],
             ci_hi=pct([r['g2'] for r in boot])[1],
             note='1 - c_3^2/(2+c_4) with the regression-error moments recovered '
                  'by cumulant deconvolution'),
        dict(quantity='g_2^EIV (closed-form EIV analogue)',
             value=res_adj['g2_eiv'], ci_lo=pct([r['g2_eiv'] for r in boot])[0],
             ci_hi=pct([r['g2_eiv'] for r in boot])[1],
             note='1 - Corr^2(Phi_L, G) at the recovered moments'),
        dict(quantity='ARE of the transplanted weight',
             value=res_adj['are_fixed'], ci_lo=np.nan, ci_hi=np.nan,
             note='closed-form efficiency at t = m_3/(m_4 - m_2^2)'),
        dict(quantity='observed variance ratio TC-PMM2* : TC-LS (cluster bootstrap)',
             value=ratio_star, ci_lo=ratio_star_ci[0], ci_hi=ratio_star_ci[1],
             note='ratio of bootstrap variances over the same PSU resamples; the '
                  'interval is a delete-one-stratum jackknife (15 replicates, '
                  f'SE {ratio_star_se:.4f}). Monte Carlo interval of the '
                  f'bootstrap variances alone: [{ratio_star_mc[0]:.4f}, '
                  f'{ratio_star_mc[1]:.4f}]'),
        dict(quantity='observed variance ratio TC-PMM2 : TC-LS (cluster bootstrap)',
             value=ratio_fix, ci_lo=ratio_fix_ci[0], ci_hi=ratio_fix_ci[1],
             note='transplanted weight; delete-one-stratum jackknife interval. '
                  f'Monte Carlo interval: [{ratio_fix_mc[0]:.4f}, '
                  f'{ratio_fix_mc[1]:.4f}]'),
        dict(quantity='observed variance ratio TC-PMM2* : TC-PMM2 (re-optimisation)',
             value=float(np.nanvar(B[:, j_star], ddof=1)
                         / np.nanvar(B[:, j_fix], ddof=1)),
             ci_lo=np.nan, ci_hi=np.nan,
             note='what re-optimising the weight buys on this dataset. The '
                  f"transplanted ratio is t = {res_adj['t_fixed']:.4f} and the "
                  f"optimum is {res_adj['t_opt']:.4f}, so the answer is nothing: "
                  'the negative half of the paper is inactive in this regime'),
        dict(quantity='observed variance ratio TC-PMM2* : TC-LS (sandwich)',
             value=float((res_adj['fits']['tc_pmm2star'].se[1]
                          / res_adj['fits']['tc_ls'].se[1]) ** 2),
             ci_lo=np.nan, ci_hi=np.nan,
             note='squared ratio of model-based sandwich standard errors, which '
                  'ignore the survey design'),
    ]
    write_csv(OUT / 'efficiency.csv', eff_rows)

    # ---- ARE(t) curve for the figure ------------------------------------
    b1 = res_adj['fits']['tc_ls'].beta[1]
    aargs = (res_adj['sx2'], b1, res_adj['m2'], res_adj['m3'], res_adj['m4'],
             res_adj['nu2'], res_adj['nu3'], res_adj['nu4'], res_adj['nu5'],
             res_adj['nu6'])
    tgrid = np.linspace(-0.2, 0.95, 231)
    write_csv(OUT / 'are_curve.csv',
              [dict(t=float(t), are=float(are_of_t(t, *aargs))) for t in tgrid])

    # How far g_2^EIV can sit from g_2 is governed by the dimensionless signal
    # ratio beta_1^2 sigma_x^2 / m_2, not by the reliability: the reliability
    # enters the three blocks only through terms carried by powers of beta_1.
    # The curve below holds every standardised moment of this dataset fixed and
    # scales beta_1, so it answers how strong the regression would have to be
    # for the distinction between g_2 and g_2^EIV to become visible here.
    sig_rows = []
    for mult in np.exp(np.linspace(np.log(0.25), np.log(20.0), 60)):
        bb = b1 * mult
        aa = (res_adj['sx2'], bb, res_adj['m2'], res_adj['m3'], res_adj['m4'],
              res_adj['nu2'], res_adj['nu3'], res_adj['nu4'], res_adj['nu5'],
              res_adj['nu6'])
        t_o, g_e, _ = optimal_t(*aa)
        sig_rows.append(dict(beta1_multiplier=float(mult), beta1=float(bb),
                             signal_ratio=float(bb ** 2 * res_adj['sx2']
                                                / res_adj['m2']),
                             t_opt=float(t_o), g2_eiv=float(g_e),
                             are_fixed=float(are_of_t(res_adj['t_fixed'], *aa)),
                             g2=float(res_adj['g2'])))
    write_csv(OUT / 'signal_ratio_curve.csv', sig_rows)

    # Where does this dataset sit relative to the break-even reliability of the
    # transplanted weight?  Hold the standardised shape of U and every moment of
    # the regression error fixed, and sweep the reliability by rescaling U.
    shape_u = {r: res_adj[f'nu{r}'] / res_adj['nu2'] ** (r / 2.0)
               for r in range(3, 7)}
    lam_rows = []
    for lam_g in np.round(np.arange(0.50, 0.9951, 0.005), 4):
        nu2_l = res_adj['sx2'] * (1.0 - lam_g) / lam_g
        nul = {'nu2': nu2_l}
        for r in range(3, 7):
            nul[f'nu{r}'] = shape_u[r] * nu2_l ** (r / 2.0)
        aa = (res_adj['sx2'], b1, res_adj['m2'], res_adj['m3'], res_adj['m4'],
              nul['nu2'], nul['nu3'], nul['nu4'], nul['nu5'], nul['nu6'])
        t_o, g_e, _ = optimal_t(*aa)
        lam_rows.append(dict(lam=float(lam_g), nu2=float(nu2_l),
                             t_opt=float(t_o), g2_eiv=float(g_e),
                             are_fixed=float(are_of_t(res_adj['t_fixed'], *aa)),
                             g2=float(res_adj['g2'])))
    write_csv(OUT / 'lambda_sweep.csv', lam_rows)
    max_are_fixed = max(r['are_fixed'] for r in lam_rows)
    gate('R10 the transplanted weight never loses at this shape of U',
         max_are_fixed <= 1.0,
         'closed-form efficiency of the transplanted weight over '
         'lambda in [0.50, 0.995] with the standardised shape of U and the '
         f'regression-error moments held at their estimates: maximum '
         f'{max_are_fixed:.4f}, attained at lambda = '
         f"{min(lam_rows, key=lambda r: -r['are_fixed'])['lam']:.3f}. With "
         'beta_1 nu_3 < 0 no break-even reliability exists, so this dataset '
         'lies in the benign part of the parameter space, not the adversarial '
         'part')

    # Sensitivity of the optimal weight to the higher moments, which are the
    # least reliably estimated things in the whole analysis.
    def variant(tag, nu_v, note):
        r = analyse(d['readings'], d['y'], coef, rest, nu_override=nu_v)
        return dict(variant=tag, nu4=nu_v['nu4'], nu5=nu_v['nu5'],
                    nu6=nu_v['nu6'], t_opt=r.get('t_opt', np.nan),
                    g2_eiv=r.get('g2_eiv', np.nan),
                    beta1=float(r['fits']['tc_pmm2star'].beta[1]),
                    d_beta1=float(r['fits']['tc_pmm2star'].beta[1]
                                  - res_adj['fits']['tc_pmm2star'].beta[1]),
                    note=note)

    base_nu = {k: res_adj[k] for k in ('nu2', 'nu3', 'nu4', 'nu5', 'nu6')}
    ci = {r['quantity']: r for r in mom_ci}
    sens = [variant('point estimates', base_nu,
                    'nu_2..nu_6 as recovered from the three readings')]
    for key in ('nu4', 'nu5', 'nu6'):
        for end in ('ci_lo', 'ci_hi'):
            v = dict(base_nu)
            v[key] = ci[key][end]
            sens.append(variant(f'{key} at its {end[3:]} bootstrap limit', v,
                                f'{key} = {v[key]:.4g}, the others at their '
                                'point estimates'))
    v0 = dict(base_nu)
    v0['nu5'] = 0.0
    sens.append(variant('nu_5 set to zero', v0,
                        'its bootstrap interval covers zero, so the sign of '
                        'nu_5 is not established by k = 3 replicates'))
    vg = dict(base_nu)
    vg['nu4'] = 3.0 * base_nu['nu2'] ** 2
    vg['nu5'] = 0.0
    vg['nu6'] = 15.0 * base_nu['nu2'] ** 3
    sens.append(variant('Gaussian nu_4, nu_5, nu_6', vg,
                        'a deliberately wrong but common default: the higher '
                        'moments of a normal error with the same variance'))
    sens.append(variant('2 mmHg grid contribution removed', nu_g,
                        'digit-heaping correction under the additive uniform '
                        'rounding model'))
    sens.append(variant('replicate-only sample moments', nu_rep_adj,
                        'nu recovered from all 6751 subjects with three usable '
                        'readings rather than from the 5731 in the regression'))
    write_csv(OUT / 'moment_sensitivity.csv', sens)

    # ---- deviation histogram for the figure -----------------------------
    adj = d['readings'] - (d['readings'].mean(axis=0) - d['readings'].mean())
    dev = (adj - adj.mean(axis=1, keepdims=True)).ravel()
    # the readings are even integers, so d = W_ij - Wbar_i lives on multiples
    # of 2/3 mmHg; bins of exactly that width, centred on the support, avoid
    # the alternating empty bins that any other width produces
    step_bin = 2.0 / 3.0
    edges = (np.arange(-31, 31.1) - 0.5) * step_bin
    cnt, _ = np.histogram(dev, bins=edges)
    write_csv(OUT / 'deviation_hist.csv',
              [dict(left=float(edges[i]), right=float(edges[i + 1]),
                    count=int(cnt[i])) for i in range(len(cnt))])

    # ---- summary.csv (one row per headline quantity) ---------------------
    summ = []
    for r in est_rows:
        summ.append(dict(block='estimator', key=r['estimator'],
                         value=r['beta1'], lo=r['ci_lo'], hi=r['ci_hi'],
                         note=f"cluster-bootstrap SE {r['se_cluster']:.6g}, "
                              f"sandwich SE {r['se_sandwich']:.6g}"))
    for k in ('lam', 'nu2', 'nu3', 'nu4', 'nu5', 'nu6', 'gamma3_u', 'gamma4_u',
              'c3', 'c4', 'g2', 'g2_eiv', 't_fixed', 't_opt', 'signal_ratio',
              'sx2', 'var_w', 'm2'):
        ci = [c for c in mom_ci if c['quantity'] == k]
        summ.append(dict(block='quantity', key=k, value=float(res_adj[k]),
                         lo=ci[0]['ci_lo'] if ci else np.nan,
                         hi=ci[0]['ci_hi'] if ci else np.nan,
                         note='reading-adjusted, regression sample, n = %d' % n))
    summ.append(dict(block='efficiency', key='observed_ratio_star_tcls',
                     value=ratio_star, lo=ratio_star_ci[0], hi=ratio_star_ci[1],
                     note='cluster-bootstrap variance ratio'))
    summ.append(dict(block='efficiency', key='observed_ratio_fixed_tcls',
                     value=ratio_fix, lo=ratio_fix_ci[0], hi=ratio_fix_ci[1],
                     note='cluster-bootstrap variance ratio'))
    summ.append(dict(block='specification', key='gap_tcls_minus_star',
                     value=float(gap_pt),
                     lo=float(np.percentile(gap, 2.5)),
                     hi=float(np.percentile(gap, 97.5)),
                     note=f'z = {gap_pt / gap_sd:.2f}; over-identification '
                          'contrast, not an efficiency result'))
    summ.append(dict(block='specification', key='gap_z',
                     value=float(gap_pt / gap_sd), lo=np.nan, hi=np.nan,
                     note='cluster-bootstrap z of the previous row'))
    summ.append(dict(block='design', key='n_obs', value=float(n), lo=np.nan,
                     hi=np.nan, note='subjects'))
    summ.append(dict(block='design', key='n_psu', value=float(n_psu), lo=np.nan,
                     hi=np.nan, note='PSUs, the resampling unit'))
    summ.append(dict(block='design', key='n_strata', value=float(n_strata),
                     lo=np.nan, hi=np.nan, note='strata'))
    write_csv(OUT / 'summary.csv', summ)

    # ---- verification report --------------------------------------------
    lines = ['NHANES 2013-2014 real-data application -- verification report',
             '=' * 70, '']
    for name, ok, detail in sorted(gates, key=lambda g: int(g[0][1:].split()[0])):
        lines.append(f'[{"PASS" if ok else "FAIL"}] {name}')
        lines.append(f'       {detail}')
    n_ok = sum(1 for _, ok, _ in gates if ok)
    lines += ['', f'{n_ok}/{len(gates)} checks passed', '',
              'Resampling unit: PSU within stratum '
              f'({n_psu} PSUs in {n_strata} strata, design df = {n_psu - n_strata}). '
              'Row-level resampling is never used for an interval.',
              'The point estimates are unweighted: this is an illustration of the '
              'estimator, not a population estimate.']
    (OUT / 'verification_report.txt').write_text('\n'.join(lines) + '\n')

    # ---- manifest --------------------------------------------------------
    import scipy
    import sympy
    manifest = dict(
        run='realdata_nhanes', timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        git_hash=git_hash(), python_executable=sys.executable,
        python_version=sys.version, platform=platform.platform(),
        numpy=np.__version__, scipy=scipy.__version__, sympy=sympy.__version__,
        env_KU_PMM_EIV_PYTHON=os.environ.get('KU_PMM_EIV_PYTHON', '(unset)'),
        source_files=dl,
        data_cache=str(DATA.relative_to(ROOT)),
        citation='NCHS, National Health and Nutrition Examination Survey '
                 '2013-2014: Blood Pressure (BPX_H), Standard Biochemistry '
                 'Profile (BIOPRO_H), Demographics (DEMO_H). Work of the United '
                 'States government, public domain (17 U.S.C. 105); no data-use '
                 'agreement.',
        variables=dict(W='BPXDI1 (first diastolic reading, mmHg)',
                       replicates='BPXDI1..3, k = 3',
                       Y='log(LBXSGTSI), serum gamma-glutamyl transferase, U/L',
                       design='SDMVSTRA, SDMVPSU, WTMEC2YR'),
        n_obs=n, n_psu=n_psu, n_strata=n_strata,
        design_df=n_psu - n_strata,
        resampling_unit='PSU within stratum (Verification Standard 9)',
        weighting='point estimates unweighted; design-weighted naive LS and '
                  'corrected LS reported in diagnostics.csv as a check',
        bootstrap=dict(kind='stratified cluster bootstrap over PSUs within strata',
                       b_reps=args.boot, seed=args.seed,
                       wall_sec=round(wall_boot, 2),
                       propagates='reading-number adjustment, moment recovery '
                                  'nu_2..nu_6, cumulant deconvolution, weight '
                                  'optimisation and all five fits'),
        replicate_system=dict(k=K_REP, coefficients={str(p): coef[p] for p in coef},
                              expressions=exprs, cross_check_vs_theory=xcheck),
        estimators=list(EST_ORDER),
        gates=[dict(name=nm, passed=ok, detail=dt) for nm, ok, dt in gates],
        attrition=attrition,
        wall_sec=round(time.time() - t_start, 2),
        outputs=sorted(p.name for p in OUT.iterdir()),
    )
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str)
                                       + '\n')

    # ---- console ---------------------------------------------------------
    print(f'n = {n} subjects, {n_psu} PSUs in {n_strata} strata')
    print(f"lambda = {res_adj['lam']:.4f}  nu2 = {res_adj['nu2']:.3f}  "
          f"nu3 = {res_adj['nu3']:.2f}  gamma3(U) = {res_adj['gamma3_u']:+.4f}")
    print(f"c3(eps) = {res_adj['c3']:.4f}  g2 = {res_adj['g2']:.4f}  "
          f"g2_EIV = {res_adj['g2_eiv']:.4f}  observed ratio = {ratio_star:.4f}")
    for r in est_rows:
        print(f"  {r['estimator']:22s} b1 = {r['beta1']:.6f}  "
              f"cluster SE {r['se_cluster']:.6f}  CI [{r['ci_lo']:.6f}, "
              f"{r['ci_hi']:.6f}]")
    print(f'{n_ok}/{len(gates)} checks passed  ->  {OUT}')


if __name__ == '__main__':
    main()
