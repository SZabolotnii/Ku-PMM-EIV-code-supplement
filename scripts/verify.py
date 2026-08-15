"""Verification gates G0-G6 over evidence/<run>/ (SPEC 7).

Run BEFORE citing any number anywhere.  A failing gate means "re-run / revisit
the theory", never "edit the table" (CLAUDE.md, hygiene rules).

    python scripts/verify.py --run confirmatory

Gating convention (stated explicitly because the SPEC does not fix it): all
gates are evaluated on the SLOPE b1, the parameter of interest.  The intercept
b0 is reported alongside as a non-gating diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / 'evidence'
sys.path.insert(0, str(Path(__file__).resolve().parent))


# --------------------------------------------------------------------------
def load(run):
    d = EVIDENCE / run
    with open(d / 'summary.csv') as fh:
        summ = list(csv.DictReader(fh))
    with open(d / 'paired.csv') as fh:
        pair = list(csv.DictReader(fh))
    for rows in (summ, pair):
        for r in rows:
            for k, v in list(r.items()):
                if k in ('cell_id', 'eps_key', 'u_family', 'estimator', 'param',
                         'estimator_a', 'estimator_b'):
                    continue
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = np.nan
    return d, summ, pair


def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r[k] == v]
    return out


C3 = {'gauss': 0.0, 'g_c3_0.5': 0.5, 'g_c3_1.0': 1.0, 'g_c3_1.5': 1.5,
      'real_dbp': 1.284}          # NHANES application regime (2026-08-15)

# Gamma-type U designs the main gates (G1-G5) apply to.  The '_neg' mirrors and
# the NHANES-calibrated 'gamma_real' were added on 2026-08-15; the heavy-tailed
# 'lognormal' stays out of these gates and is handled separately by G6.
MAIN_U = ('gamma_wh', 'gamma_fixed',
          'gamma_wh_neg', 'gamma_fixed_neg',
          'gamma_real', 'gamma_real_neg')
ALL_U = MAIN_U + ('lognormal',)
# mirrored family -> family it mirrors (used by the mirror-symmetry gate G7)
MIRROR_OF = {'gamma_wh_neg': 'gamma_wh', 'gamma_fixed_neg': 'gamma_fixed',
             'gamma_real_neg': 'gamma_real'}


class Report:
    def __init__(self):
        self.gates = []

    def gate(self, name, desc, checks, note=''):
        """checks: list of (label, bool, detail)"""
        if not checks:                      # nothing to test -> never a silent PASS
            self.skip(name, desc, 'no applicable cells in this run')
            return
        failed = [c for c in checks if not c[1]]
        ok = not failed
        self.gates.append((name, ok, len(checks) - len(failed), len(checks)))
        print(f'\n[{"PASS" if ok else "FAIL"}] {name}  {desc}')
        if note:
            print(f'       {note}')
        for lab, good, detail in checks:
            if not good or len(checks) <= 12:
                print(f'       {"ok " if good else "FAIL"} {lab:52s} {detail}')
        if failed and len(checks) > 12:
            print(f'       ({len(checks) - len(failed)}/{len(checks)} sub-checks passed)')

    def skip(self, name, desc, why):
        self.gates.append((name, None, 0, 0))
        print(f'\n[SKIPPED-ENVIRONMENT] {name}  {desc}\n       {why}')

    def finish(self):
        run_g = [g for g in self.gates if g[1] is not None]
        npass = sum(1 for g in run_g if g[1])
        print('\n' + '=' * 78)
        for name, ok, a, b in self.gates:
            tag = 'SKIPPED' if ok is None else ('PASS' if ok else 'FAIL')
            print(f'  {name:4s} {tag:8s} {a}/{b} sub-checks')
        print(f'\n{npass}/{len(run_g)} checks passed'
              + (f'  ({len(self.gates) - len(run_g)} skipped)'
                 if len(run_g) != len(self.gates) else ''))
        return npass == len(run_g)


# --------------------------------------------------------------------------
def g0(rep: Report):
    f = EVIDENCE / 'g0_oracle' / 'g0_result.csv'
    if not f.exists():
        rep.skip('G0', 'TC-PMM2(U=0) == EstemPMM::lm_pmm2',
                 'evidence/g0_oracle/g0_result.csv absent; run Rscript scripts/oracle_check.R')
        return
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    checks = [(f'dataset {r["dataset"]} ({r["eps"]}, n={r["n"]})',
               float(r['max_dbeta']) < 1e-6, f'max|dbeta| = {float(r["max_dbeta"]):.3e}')
              for r in rows]
    rep.gate('G0', 'reduction to the EstemPMM oracle at U == 0 (max|dbeta| < 1e-6)', checks)


def g0b_python_reduction(rep: Report):
    """Supplementary, solver-independent: TC-PMM2 with nu = 0 must equal a plain
    PMM2 fit obtained by a completely different root finder (scipy)."""
    import scipy.optimize as so
    import dgp
    from estimators import fit_ls, fit_tc_pmm2, central_moments

    checks = []
    for i, (ek, n) in enumerate([('g_c3_1.0', 300), ('g_c3_1.5', 800), ('gauss', 400)]):
        rng = np.random.default_rng(9100 + i)
        x = dgp.MEAN_X + rng.standard_normal(n)
        y = 1.0 + x + dgp.EPS_SPECS[ek].draw(n, rng)
        f0 = fit_ls(y, x, 'ols')
        m2, m3, m4 = central_moments(y - f0.beta[0] - f0.beta[1] * x)
        mine = fit_tc_pmm2(y, x, m2, m3, m4, 0.0, 0.0, f0.beta, 'tc')

        def score(b):                     # plain PMM2 score, independent coding
            e = y - b[0] - b[1] * x
            s = (m4 - m2 ** 2) * e - m3 * (e ** 2 - m2)
            return np.array([s.sum(), float(x @ s)])

        ref = so.root(score, f0.beta, method='hybr', tol=1e-13)
        d = float(np.max(np.abs(mine.beta - ref.x)))
        checks.append((f'{ek}, n={n}', ref.success and d < 1e-8, f'max|dbeta| = {d:.3e}'))
    rep.gate('G0b', 'pure-Python cross-check of the same reduction (independent solver)',
             checks)


TC_ESTS = ('tc_pmm2_oracle', 'tc_pmm2_adaptive',
           'tc_pmm2star_oracle', 'tc_pmm2star_adaptive')


def g1(rep: Report, summ, u_family=None, name='G1', extra=''):
    checks = []
    for est in TC_ESTS:
        rows = sel(summ, estimator=est, param='b1')
        if u_family:
            rows = [r for r in rows if r['u_family'] in u_family]
        for r in rows:
            nv = sel(summ, cell_id=r['cell_id'], estimator='naive_pmm2', param='b1')[0]
            thr = max(0.25 * abs(nv['bias']), 2.0 * r['se_bias'])
            ok = abs(r['bias']) <= thr
            checks.append((f'{est} {r["cell_id"]}', ok,
                           f'|bias|={abs(r["bias"]):.5f} <= max(0.25|bias_naive|, 2SE)={thr:.5f}'))
    rep.gate(name, f'unbiasedness vs naive-PMM2 (H1){extra}', checks)


def g2_theory_crosscheck(rep: Report):
    """C1: are_blocks (typed by hand here) == theory/are_formulas.scalars."""
    sys.path.insert(0, str(ROOT / 'theory'))
    import dgp
    from estimators import are_blocks
    try:
        import are_formulas as TH
    except ImportError:
        rep.skip('C1', 'closed-form blocks vs theory/are_formulas.py',
                 'theory/are_formulas.py not importable')
        return
    worst, worst_lab = 0.0, ''
    for ek in dgp.EPS_SPECS:
        for fam in ALL_U:
            for lam in (0.80, 0.90, 0.95, dgp.REALDATA_LAM):
                for b1 in (1.0, -1.0):
                    c = dgp.make_cell(ek, fam, lam, 1000, b1=b1)
                    em, um = c.eps.moments, c.u.moments
                    a = (dgp.VAR_X, b1, em['m2'], em['m3'], em['m4'],
                         um['nu2'], um['nu3'], um['nu4'], um['nu5'], um['nu6'])
                    mine = are_blocks(*a)
                    theirs = TH.scalars(dgp.MEAN_X, *a)
                    d = max(abs(x - y) / max(1.0, abs(y)) for x, y in zip(mine, theirs))
                    if d > worst:
                        worst, worst_lab = d, f'{ek}/{fam}/lam={lam}/b1={b1:+.0f}'
    rep.gate('C1', 'independent typing of blocks (3.1)-(3.3) == theory module',
             [('max relative difference over all configurations', worst < 1e-12,
               f'{worst:.3e} (worst at {worst_lab})')])


def g2(rep: Report, paired):
    """REVISED (coordinator amendment): measured RE must match the closed-form
    ARE for that cell within +-2 MC SE -- for BOTH estimators.  Reproducing the
    *predicted loss* of fixed-weight TC-PMM2 is itself a passing check."""
    checks = []
    for est_a, key, tag in (('tc_pmm2star_oracle', 'are_star_theory', 'TC-PMM2*'),
                            ('tc_pmm2_oracle', 'are_fixed_theory', 'fixed  ')):
        rows = [r for r in sel(paired, estimator_a=est_a, estimator_b='tc_ls',
                               param='b1')
                if C3[r['eps_key']] >= 1.0 and r['u_family'] in MAIN_U
                and r['delta'] == 0.0]
        for r in sorted(rows, key=lambda z: (z['eps_key'], z['u_family'],
                                             -z['b1_true'], z['lam'], z['n'])):
            tgt, se = r[key], r['se_re']
            z = (r['re'] - tgt) / se if se > 0 else np.nan
            ok = abs(z) <= 2.0
            checks.append((f'{tag} {r["eps_key"]} {r["u_family"]} '
                           f'b1{r["b1_true"]:+.0f} lam={r["lam"]:.2f} n={int(r["n"])}', ok,
                           f'RE={r["re"]:.4f} vs ARE={tgt:.4f}  z={z:+.2f}'))
    rep.gate('G2', 'measured RE matches closed-form ARE within 2 SE, c3 >= 1.0 (H2, revised)',
             checks)


def g2b_star_dominance(rep: Report, paired):
    """TC-PMM2* must never lose materially to TC-LS (ARE* <= 1 by Cauchy-Schwarz)."""
    rows = [r for r in sel(paired, estimator_a='tc_pmm2star_oracle', estimator_b='tc_ls',
                           param='b1')
            if r['u_family'] in MAIN_U and r['delta'] == 0.0]
    checks = []
    for r in sorted(rows, key=lambda z: (z['eps_key'], z['u_family'],
                                         -z['b1_true'], z['lam'], z['n'])):
        ok = r['re'] <= 1.0 + 2.0 * (r['se_re'] if np.isfinite(r['se_re']) else 0.0)
        checks.append((f'{r["eps_key"]} {r["u_family"]} b1{r["b1_true"]:+.0f} '
                       f'lam={r["lam"]:.2f} n={int(r["n"])}', ok,
                       f'RE={r["re"]:.4f} (+-{2*r["se_re"]:.4f})'))
    rep.gate('G2b', 'TC-PMM2* never materially worse than TC-LS (ARE* <= 1)', checks)


def g3(rep: Report, paired):
    checks = []
    for a, b, tag in (('tc_pmm2_adaptive', 'tc_pmm2_oracle', 'fixed'),
                      ('tc_pmm2star_adaptive', 'tc_pmm2star_oracle', 'star ')):
        rows = [r for r in sel(paired, estimator_a=a, estimator_b=b, param='b1')
                if C3[r['eps_key']] >= 1.0 and r['n'] >= 500 and r['delta'] == 0.0
                and r['u_family'] in MAIN_U]
        checks += [(f'{tag} {r["eps_key"]} {r["u_family"]} b1{r["b1_true"]:+.0f} '
                    f'lam={r["lam"]:.2f} n={int(r["n"])}',
                    r['re'] <= 1.10, f'MSE(adapt)/MSE(oracle) = {r["re"]:.4f}')
                   for r in sorted(rows, key=lambda z: (z['eps_key'], z['u_family'],
                                                        z['lam'], z['n']))]
    rep.gate('G3', 'adaptive loss bounded by 10% at n >= 500 (H3)', checks)


def g4(rep: Report, summ):
    checks = []
    for est in TC_ESTS + ('tc_ls',):
        for r in sorted(sel(summ, estimator=est, param='b1'),
                        key=lambda z: (z['eps_key'], z['u_family'], z['lam'], z['n'])):
            if r['n'] < 500 or r['u_family'] not in MAIN_U \
                    or r['delta'] != 0.0:
                continue
            ok = 0.93 <= r['coverage'] <= 0.97
            checks.append((f'{est} {r["eps_key"]} {r["u_family"]} b1{r["b1_true"]:+.0f} '
                           f'lam={r["lam"]:.2f} n={int(r["n"])}', ok,
                           f'coverage={r["coverage"]*100:.1f}% (+-{r["se_coverage"]*100:.1f})'))
    rep.gate('G4', 'sandwich 95% CI coverage in [93,97]% at n >= 500 (H4)', checks)


def g5(rep: Report, paired):
    checks = []
    for est, tag in (('tc_pmm2_oracle', 'fixed'), ('tc_pmm2star_oracle', 'star ')):
        rows = [r for r in sel(paired, estimator_a=est, estimator_b='tc_ls',
                               param='b1') if r['eps_key'] == 'gauss' and r['delta'] == 0.0]
        checks += [(f'{tag} gauss {r["u_family"]} b1{r["b1_true"]:+.0f} '
                    f'lam={r["lam"]:.2f} n={int(r["n"])}',
                    r['re'] <= 1.05, f'MSE ratio = {r["re"]:.4f}')
                   for r in sorted(rows, key=lambda z: (z['u_family'], z['lam'], z['n']))]
    rep.gate('G5', 'Gaussian-eps control: MSE(TC-PMM2)/MSE(TC-LS) <= 1.05 (H0)', checks)


def g6(rep: Report, summ):
    rows = [r for r in summ if r['param'] == 'b1' and r['u_family'] == 'lognormal'
            and r['estimator'] in ('tc_pmm2_oracle', 'tc_pmm2star_oracle')]
    if not rows:
        rep.skip('G6', 'lognormal U', 'no lognormal cells in this run')
        return
    checks = []
    for r in sorted(rows, key=lambda z: (z['estimator'], z['eps_key'], z['lam'], z['n'])):
        defined = r['conv_rate'] >= 0.99 and np.isfinite(r['bias'])
        nv = sel(summ, cell_id=r['cell_id'], estimator='naive_pmm2', param='b1')[0]
        thr = max(0.25 * abs(nv['bias']), 2.0 * r['se_bias'])
        checks.append((f'{r["estimator"]} {r["eps_key"]} b1{r["b1_true"]:+.0f} '
                       f'lam={r["lam"]:.2f} n={int(r["n"])}',
                       defined and abs(r['bias']) <= thr,
                       f'conv={r["conv_rate"]*100:.1f}% |bias|={abs(r["bias"]):.5f} thr={thr:.5f}'))
    rep.gate('G6', 'estimator defined + G1 holds under heavy-tailed lognormal U', checks)


def g7p_pathwise(rep: Report):
    """G7p (2026-08-15): the mirror equivalence is EXACT in finite samples, not
    only asymptotically -- and that is what makes gate G7e a pure MC-noise test.

    Take any dataset (X, U, eps) and reflect it: X -> -X, U -> -U, b1 -> -b1.
    Then Y is unchanged, Wt -> -Wt, and the corrected powers transform as
    t1 -> -t1, t2 -> t2, t3 -> -t3 (because nu3 -> -nu3), so Psi0 is invariant
    and Psi1 flips sign: the estimating system is solved by (b0_hat, -b1_hat).
    The slope is additionally invariant to a location shift of X (the score is
    equivariant under the design map (1,X) -> (1,X+c)), so the mirrored cell may
    keep X ~ N(+1,1) instead of N(-1,1) without breaking the identity.

    Consequence: MSE(mirrored cell) == MSE(b1-flipped cell) EXACTLY at every n,
    so any G7e discrepancy is Monte Carlo noise or a DGP bug, never an O(1/n)
    finite-sample term.  Checked here on all nine estimators."""
    import dgp
    from estimators import ESTIMATOR_NAMES, fit_all

    cfgs = [('g_c3_1.0', 'gamma_wh', 0.80, 300, 1.0),
            ('g_c3_1.5', 'gamma_fixed', 0.95, 500, -1.0),
            ('real_dbp', 'gamma_real', dgp.REALDATA_LAM, 400, 1.0),
            ('gauss', 'gamma_wh', 0.90, 250, 1.0)]
    checks = []
    for ek, fam, lam, n, b1 in cfgs:
        cell = dgp.make_cell(ek, fam, lam, n, b1=b1)
        mir = dgp.make_cell(ek, fam + '_neg', lam, n, b1=-b1)
        d = dgp.draw_replication(cell, 0)
        em = cell.eps.moments
        a = fit_all(d['y'], d['w'], d['x'], em,
                    {k: cell.u.moments[k] for k in dgp.NU_KEYS}, dgp.VAR_X)
        # reflected data: same eps, same Y, X -> -X, U -> -U, b1 -> -b1
        y_m = mir.b0 + mir.b1 * (-d['x']) + d['eps']
        b = fit_all(y_m, -d['w'], -d['x'], em,
                    {k: mir.u.moments[k] for k in dgp.NU_KEYS}, dgp.VAR_X)
        worst = max(abs(a[nm].beta[1] + b[nm].beta[1]) for nm in ESTIMATOR_NAMES)
        dy = float(np.max(np.abs(y_m - d['y'])))
        checks.append((f'{ek} {fam} lam={lam:g} n={n} b1{b1:+.0f}',
                       worst < 1e-9 and dy < 1e-12,
                       f'max|b1_hat(mirror) + b1_hat(orig)| = {worst:.3e} '
                       f'over {len(ESTIMATOR_NAMES)} estimators'))
    rep.gate('G7p', 'pathwise mirror identity: reflecting (X,U,b1) negates every '
                    'slope estimate exactly', checks)


def g7_mirror(rep: Report, paired, ref_run='confirmatory'):
    """G7 (2026-08-15, declared BEFORE the negative-nu3 run was executed).

    Theory: the sign of the whole effect is carried by sign(b1 * gamma3(U) * m3),
    and in blocks (3.1)-(3.3) the odd U-moments always appear multiplied by an
    odd power of b1.  Mirroring U (nu3, nu5 -> -nu3, -nu5) must therefore be
    EXACTLY equivalent to flipping the sign of b1.  Two sub-gates:

      G7t (closed form)  : |ARE(mirror cell) - ARE(b1-flipped original)| < 1e-12,
                           read off the are_*_theory columns already stored in
                           evidence, for both estimators.  A failure here would
                           be a formula-level problem affecting the whole paper.
      G7e (Monte Carlo)  : the two cells' MEASURED RE agree within MC error,
                           |z| <= 3 with z = (RE_a - RE_b)/sqrt(SE_a^2 + SE_b^2).
                           The two cells use independent U streams, so the SEs
                           add.  Threshold 3 (not 2) because ~290 sub-checks are
                           made; at 2 SE one expects ~14 failures by chance.
    """
    pool = list(paired)
    if ref_run:
        try:
            _, _, ref_pair = load(ref_run)
            pool += ref_pair
        except FileNotFoundError:
            pass
    index = {}
    for r in pool:
        if r['param'] != 'b1' or r['estimator_b'] != 'tc_ls':
            continue
        index[(r['estimator_a'], r['eps_key'], r['u_family'], r['lam'],
               r['n'], r['b1_true'], r['delta'])] = r

    checks_t, checks_e, missing = [], [], 0
    for (est, ek, fam, lam, n, b1, dl), r in sorted(index.items()):
        if fam not in MIRROR_OF or dl != 0.0:
            continue
        if est not in ('tc_pmm2_oracle', 'tc_pmm2star_oracle'):
            continue
        key = (est, ek, MIRROR_OF[fam], lam, n, -b1, dl)
        ref = index.get(key)
        if ref is None:
            missing += 1
            continue
        tag = ('star ' if 'star' in est else 'fixed') + \
              f' {ek} {MIRROR_OF[fam]} lam={lam:.3f} n={int(n)} b1{b1:+.0f}'
        thkey = 'are_star_theory' if 'star' in est else 'are_fixed_theory'
        dt = abs(r[thkey] - ref[thkey])
        checks_t.append((tag, dt < 1e-12, f'|dARE| = {dt:.3e}'))
        se = float(np.hypot(r['se_re'], ref['se_re']))
        d_re = r['re'] - ref['re']
        z = d_re / se if se > 0 else np.nan
        # degeneracy guard: with m3 = 0 the fixed-weight score IS the TC-LS
        # score, so RE == 1 exactly in both cells and z is a 0/0 artefact.
        checks_e.append((tag, abs(z) <= 3.0 or abs(d_re) <= 1e-6,
                         f'RE(mirror)={r["re"]:.4f} vs RE(b1-flip)={ref["re"]:.4f}  z={z:+.2f}'))
    note = (f'{missing} mirrored cells had no partner in this run or in '
            f'evidence/{ref_run}' if missing else
            f'partner cells taken from this run and from evidence/{ref_run}')
    rep.gate('G7t', 'closed-form ARE is invariant under (nu3,nu5 -> -) <=> (b1 -> -b1)',
             checks_t, note)
    rep.gate('G7e', 'measured RE of a mirrored cell == that of the b1-flipped cell '
                    '(|z| <= 3)', checks_e, note)
    if checks_e:
        zs = [float(c[2].split('z=')[1]) for c in checks_e]
        print(f'       mirror z-statistics: mean {np.mean(zs):+.3f}, '
              f'sd {np.std(zs, ddof=1):.3f}, max|z| {max(abs(z) for z in zs):.2f}, '
              f'#|z|>2 = {sum(1 for z in zs if abs(z) > 2)}/{len(zs)}')


def k1_report(paired):
    """K1 REVISED (coordinator amendment, supersedes SPEC 8): track A dies only
    if TC-PMM2* fails to beat TC-LS (paired MSE difference not significantly < 0)
    in EVERY cell with c3 >= 1.0, across both U-designs and both signs of b1.
    Fixed-weight TC-PMM2 losing does NOT kill the track."""
    print('\n' + '-' * 78)
    print('KILL CRITERION K1 (revised): does TC-PMM2* beat TC-LS anywhere with c3 >= 1.0?')
    rows = [r for r in sel(paired, estimator_a='tc_pmm2star_oracle',
                           estimator_b='tc_ls', param='b1')
            if C3[r['eps_key']] >= 1.0 and r['delta'] == 0.0
            and r['u_family'] in MAIN_U]
    if not rows:
        print('  no applicable cells in this run')
        return
    wins = [r for r in rows if r['mse_diff'] < 0 and r['ci_hi'] < 0]
    print(f'  cells examined              : {len(rows)}')
    print(f'  significant wins (CI < 0)   : {len(wins)}')
    print(f'  best cell  : RE={min(r["re"] for r in rows):.4f}')
    print(f'  worst cell : RE={max(r["re"] for r in rows):.4f}')
    by = {}
    for r in rows:
        k = (r['u_family'], r['b1_true'])
        by.setdefault(k, []).append(r)
    for (fam, b1), rs in sorted(by.items()):
        w = sum(1 for r in rs if r['mse_diff'] < 0 and r['ci_hi'] < 0)
        print(f'    {fam:12s} b1={b1:+.0f}: {w}/{len(rs)} significant wins, '
              f'RE range [{min(r["re"] for r in rs):.4f}, {max(r["re"] for r in rs):.4f}]')
    print(f'  => K1 {"FIRES: track A dead" if not wins else "does NOT fire"}')


def lambda_star_report(run):
    if (EVIDENCE / 'lambda_star' / 'paired.csv').exists():
        run = 'lambda_star'          # dense lam grid gives a far better crossing
    """Empirical break-even reliability lam* for FIXED-weight TC-PMM2 (RE crosses 1),
    per (c3, U-design, sign b1), against the closed-form value."""
    d = EVIDENCE / run
    if not (d / 'paired.csv').exists():
        return
    _, _, pair = load(run)
    theo = {}
    f = EVIDENCE / 'asymptotics' / 'lambda_star_theory.csv'
    if f.exists():
        with open(f) as fh:
            for r in csv.DictReader(fh):
                theo[(r['eps_key'], r['u_family'], float(r['b1']))] = r['lambda_star_theory']
    print('\n' + '-' * 78)
    print('BREAK-EVEN RELIABILITY lambda* (fixed-weight TC-PMM2, RE crosses 1), '
          f'slope b1   [source: evidence/{run}]')
    print(f'  {"eps":9s} {"U-design":12s} {"b1":>3s} {"lam*_emp":>9s} {"lam*_theory":>12s}')
    groups = {}
    for r in sel(pair, estimator_a='tc_pmm2_oracle', estimator_b='tc_ls', param='b1'):
        if r['delta'] != 0.0 or r['eps_key'] == 'gauss':
            continue                 # ARE == 1 identically when m3 = 0: no crossing
        groups.setdefault((r['eps_key'], r['u_family'], r['b1_true']), []).append(r)
    for key in sorted(groups):
        rs = sorted(groups[key], key=lambda z: z['lam'])
        emp = ''
        for a, b in zip(rs, rs[1:]):
            if (a['re'] - 1.0) * (b['re'] - 1.0) < 0:      # linear interpolation
                emp = f"{a['lam'] + (b['lam'] - a['lam']) * (1.0 - a['re']) / (b['re'] - a['re']):.4f}"
                break
        if not emp:
            emp = 'none' if all(r['re'] < 1 for r in rs) else '>max' if all(
                r['re'] > 1 for r in rs) else 'n/a'
        t = theo.get(key, '')
        print(f'  {key[0]:9s} {key[1]:12s} {key[2]:+3.0f} {emp:>9s} '
              f'{(t if t else "none"):>12s}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='confirmatory')
    ap.add_argument('--mirror-ref', default='confirmatory',
                    help="run holding the positive-nu3 partner cells for gate G7 "
                         "('' disables the cross-run lookup)")
    ap.add_argument('--no-assert', action='store_true')
    args = ap.parse_args()

    d, summ, pair = load(args.run)
    with open(d / 'manifest.json') as fh:
        man = json.load(fh)
    print('=' * 78)
    print(f'VERIFICATION REPORT  --  Ku-PMM-EIV, TC-PMM2 linear EIV')
    print(f'run       : {args.run}   ({d})')
    print(f'mode      : {man["mode"]}   M = {int(man["m_rep"])}'
          f'{"  (REDUCED: " + man["reduction_reason"] + ")" if man.get("m_reduced") else ""}')
    print(f'cells     : {int(man["n_cells"])}   wall = {man["wall_time_sec"]}s')
    print(f'git       : {man["git_hash"][:12]}   python = {man["python_executable"]}')
    print(f'numpy     : {man["numpy"]}   scipy = {man["scipy"]}')
    print('=' * 78)

    rep = Report()
    g0(rep)
    g0b_python_reduction(rep)
    g2_theory_crosscheck(rep)
    g1(rep, summ, u_family=MAIN_U, extra=' [all gamma U designs present in the run]')
    g2(rep, pair)
    g2b_star_dominance(rep, pair)
    g3(rep, pair)
    g4(rep, summ)
    g5(rep, pair)
    g6(rep, summ)
    if any(r['u_family'] in MIRROR_OF for r in pair):
        g7p_pathwise(rep)
        g7_mirror(rep, pair, args.mirror_ref)
    ok = rep.finish()
    k1_report(pair)
    lambda_star_report(args.run)

    if not args.no_assert:
        assert ok, 'one or more verification gates FAILED -- re-run, do not edit tables'


if __name__ == '__main__':
    main()
