#!/usr/bin/env python3
"""Check every headline number of the manuscript against the stored evidence.

    python scripts/verify_reported_values.py

This is the fast entry point of the supplement: it runs in seconds, needs no
simulation and no network, and it fails loudly if a number printed in the paper
cannot be reproduced from `evidence/`.

How to read a failure.  The expected values below are transcribed *from the
manuscript*, at the precision the manuscript prints them.  The actual values are
recomputed from the stored evidence files.  A mismatch therefore means one of
two things, and never "adjust the expected value":

  * the manuscript drifted from the evidence -- fix the manuscript; or
  * the evidence was regenerated and the numbers moved -- re-run the gates
    (`python scripts/verify.py --run confirmatory`) and update the manuscript.

The complementary check is `scripts/verify.py`, which evaluates the
pre-registered gates G0-G6 rather than the printed numbers.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / 'evidence'

FAILURES: list[str] = []
CHECKED = 0


# ---------------------------------------------------------------- comparison
def check(label: str, where: str, actual, printed, decimals: int) -> None:
    """Assert that `actual`, rounded to the manuscript's precision, is `printed`."""
    global CHECKED
    CHECKED += 1
    if actual is None:
        FAILURES.append(f'{label}  [{where}]\n      no value found in evidence')
        print(f'  FAIL  {label}  (not found)')
        return
    got = round(float(actual), decimals)
    want = round(float(printed), decimals)
    if got == want:
        print(f'  ok    {label:<58} {got:>12}')
    else:
        FAILURES.append(
            f'{label}  [{where}]\n      manuscript prints {want}, evidence gives {got} '
            f'(unrounded {float(actual)!r})')
        print(f'  FAIL  {label:<58} {got:>12}  != {want}')


# ------------------------------------------------------------------- loaders
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def rows(path: Path) -> list[dict]:
    with open(path, newline='') as fh:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def pick(table: list[dict], **where):
    """The single row matching `where`, or None.  Floats compared with a tolerance."""
    out = []
    for r in table:
        ok = True
        for k, v in where.items():
            got = r.get(k)
            if isinstance(v, float):
                if got is None or not isinstance(got, float) or abs(got - v) > 1e-9:
                    ok = False
                    break
            elif got != v:
                ok = False
                break
        if ok:
            out.append(r)
    if len(out) != 1:
        return None
    return out[0]


SUMMARY = rows(EV / 'confirmatory' / 'summary.csv')
PAIRED = rows(EV / 'confirmatory' / 'paired.csv')
ASY = rows(EV / 'asymptotics' / 'asymptotic_are.csv')
LSTAR_T = rows(EV / 'asymptotics' / 'lambda_star_theory.csv')


def report_counts(run: str) -> dict[str, str]:
    """Read gate outcomes out of a machine-written verification report.

    The report marks each gate `[PASS]`/`[FAIL]` and prints a
    `(NN/MM sub-checks passed)` line only for the gates that failed, so the
    overall count is derived from the markers rather than read off a line.
    """
    out: dict[str, str] = {}
    status: dict[str, str] = {}
    current = None
    for line in (EV / run / 'verification_report.txt').read_text().splitlines():
        m = re.match(r'\[(PASS|FAIL)\]\s+(G\d[a-z]?|C\d)\b', line)
        if m:
            current = m.group(2)
            status[current] = m.group(1)
            continue
        m = re.search(r'\((\d+/\d+) sub-checks passed\)', line)
        if m and current:
            out[current] = m.group(1)
    if status:
        out['TOTAL'] = f'{sum(v == "PASS" for v in status.values())}/{len(status)}'
    return out


# =========================================================================
print(__doc__.splitlines()[0])
print(f'evidence root: {EV}\n')

# -- Section 5: the negative result ---------------------------------------
print('Section 5  the transplanted weights lose')

r = pick(PAIRED, eps_key='g_c3_1.0', u_family='lognormal', lam=0.8, n=1000.0,
         b1_true=1.0, delta=0.0, estimator_a='tc_pmm2_oracle',
         estimator_b='tc_ls', param='b1')
check('MSE ratio, lognormal U, lambda=0.80 (abstract: 13.7)',
      'abstract; Table lognormal', r and r['re'], 13.7, 1)

a = pick(ASY, eps_key='g_c3_1.0', u_family='gamma_wh', b1=1.0, lam=0.8)
check('closed-form ARE, transplanted weight, W&H, lambda=0.80',
      'Sec 5.3, Table are-theory', a and a['are_fixed'], 1.597, 3)
check('transplanted weight ratio t_PMM2', 'Sec 5.3, Table weights',
      a and a['t_fixed'], 0.2857, 4)
check('optimal weight ratio t_opt (wrong side of zero)', 'Sec 5.3, Table weights',
      a and a['t_opt'], -0.0610, 4)

# -- Section 6: the constructive half -------------------------------------
print('\nSection 6  re-optimisation restores the guarantee')

check('closed-form ARE*, W&H, lambda=0.80', 'Sec 6.4', a and a['are_star'], 0.9809, 4)

r = pick(PAIRED, eps_key='g_c3_1.0', u_family='gamma_wh', lam=0.8, n=1000.0,
         b1_true=1.0, delta=0.0, estimator_a='tc_pmm2star_oracle',
         estimator_b='tc_ls', param='b1')
check('measured RE*, W&H, lambda=0.80', 'Table main-re', r and r['re'], 0.9742, 4)

# -- Section 8.6: the Gaussian surprise -----------------------------------
print('\nSection 8.6  gain under a Gaussian regression error')

g = pick(ASY, eps_key='gauss', u_family='gamma_wh', b1=1.0, lam=0.8)
check('closed-form ARE*, Gaussian eps, lambda=0.80', 'Sec 8.6',
      g and g['are_star'], 0.8125, 4)

r = pick(PAIRED, eps_key='gauss', u_family='gamma_wh', lam=0.8, n=1000.0,
         b1_true=1.0, delta=0.0, estimator_a='tc_pmm2star_oracle',
         estimator_b='tc_ls', param='b1')
check('measured RE*, Gaussian eps, lambda=0.80', 'Sec 8.6', r and r['re'], 0.8085, 4)
check('paired bootstrap lower end', 'Sec 8.6', r and r['re_boot_lo'], 0.779, 3)
check('paired bootstrap upper end', 'Sec 8.6', r and r['re_boot_hi'], 0.839, 3)

fixed_gauss = [x for x in PAIRED
               if x['eps_key'] == 'gauss' and x['delta'] == 0.0
               and x['estimator_a'] == 'tc_pmm2_oracle'
               and x['estimator_b'] == 'tc_ls' and x['param'] == 'b1']
worst = max((abs(x['re'] - 1.0) for x in fixed_gauss), default=None)
check('fixed weight is identically TC-LS in every Gaussian cell (max |RE-1|)',
      'Sec 8.6', worst, 0.0, 4)

# -- Section 8.2: the gates -----------------------------------------------
print('\nSection 8.2  pre-registered verification gates')

counts = report_counts('confirmatory')
for gate, printed in (('TOTAL', '7/10'), ('G2', '133/144'),
                      ('G3', '94/96'), ('G6', '52/72')):
    CHECKED += 1
    got = counts.get(gate)
    if got == printed:
        print(f'  ok    {"gate " + gate:<58} {got:>12}')
    else:
        FAILURES.append(f'gate {gate}  [Sec 8.2, Table gates]\n'
                        f'      manuscript prints {printed}, report gives {got}')
        print(f'  FAIL  {"gate " + gate:<58} {str(got):>12}  != {printed}')

# -- Section 5.4: break-even reliability ----------------------------------
print('\nSection 5.4  break-even reliability, closed form')

for c3, printed in (('g_c3_0.5', 0.9691), ('g_c3_1.0', 0.9431), ('g_c3_1.5', 0.9148)):
    row = pick(LSTAR_T, eps_key=c3, u_family='gamma_wh', b1=1.0)
    check(f'lambda* at {c3}', 'Sec 5.4, Table lambda-star',
          row and row['lambda_star_theory'], printed, 4)

# -- Section 9: NHANES diagnostic -----------------------------------------
print('\nSection 9  NHANES case study (a diagnostic: the model is rejected)')

eff = {r['quantity']: r for r in rows(EV / 'realdata_nhanes' / 'efficiency.csv')}

for key, printed, dec in (
        ('g_2 (error-free Kunchenko coefficient)', 0.6956, 4),
        ('g_2^EIV (closed-form EIV analogue)', 0.7066, 4)):
    check(key.split(' (')[0], 'Sec 9.7', eff.get(key, {}).get('value'), printed, dec)

est = {r['estimator']: r for r in rows(EV / 'realdata_nhanes' / 'estimates.csv')}
check('TC-LS slope', 'Sec 9.6, Table application',
      est.get('tc_ls', {}).get('beta1'), 0.01553, 5)
check('naive LS slope', 'Sec 9.6, Table application',
      est.get('naive_ls', {}).get('beta1'), 0.01358, 5)

# -- Section 9.9: Census NMF ----------------------------------------------
print('\nSection 9.9  2020 Census noisy measurement file')

cfgs = json.loads((EV / 'realdata_census_nmf' / 'configurations.json').read_text())
by_tag = {c.get('tag') or c.get('purpose'): c for c in cfgs}
neg = next((c for c in cfgs if c.get('g2_eiv') is not None
            and round(float(c['g2_eiv']), 4) == 1.0), None)
check('negative-result configuration: no gain available (g2_EIV)',
      'Sec 9.9', neg and neg['g2_eiv'], 1.0000, 4)
check('negative-result configuration: transplanted weight would cost',
      'Sec 9.9', neg and neg['are_transplanted'], 1.028, 3)
check('negative-result configuration: reliability', 'Sec 9.9',
      neg and neg['lambda_'], 0.659, 3)

# -- Section 9.10: the validation grid ------------------------------------
print('\nSection 9.10  NMF covariate x SVI response: the validation')

svi = json.loads((EV / 'realdata_nmf_svi' / 'summary.json').read_text())
check('screen passes -> within 2 SE of gold standard', 'Sec 9.10',
      100 * svi['pass_within2'], 92, 0)
check('screen fails  -> within 2 SE of gold standard', 'Sec 9.10',
      100 * svi['fail_within2'], 43, 0)
check('out-of-sample states only, screen passes', 'Sec 9.10',
      100 * svi['oos_pass_within2'], 92, 0)
check('TC-LS unaffected, screen passes', 'Sec 9.10',
      100 * svi['tcls_within2_pass'], 100, 0)
check('TC-LS unaffected, screen fails', 'Sec 9.10',
      100 * svi['tcls_within2_fail'], 100, 0)
check('bias formula: correlation with observed bias', 'Sec 9.10',
      svi['bias_formula_corr'], 0.9935, 4)
check('bias formula: slope', 'Sec 9.10', svi['bias_formula_slope'], 1.012, 3)

h = svi['headline']
check('headline cell reliability', 'Sec 9.10', h['lam'], 0.677, 3)
check('headline cell predicted g2_EIV', 'Sec 9.10', h['g2_eiv'], 0.577, 3)
check('headline cell realised variance ratio (clustered sandwich)', 'Sec 9.10',
      h['varrat'], 0.634, 3)
check('headline cell realised variance ratio (county bootstrap)', 'Sec 9.10',
      h['boot']['varrat_median'], 0.639, 3)
check('headline cell deviation from gold standard, in SE', 'Sec 9.10',
      h['dev'], 0.41, 2)
check('headline cell gold-standard slope', 'Sec 9.10', h['gold'], -0.00385, 5)
check('headline cell naive slope', 'Sec 9.10', h['naive'], -0.00248, 5)
check('headline cell TC-LS slope', 'Sec 9.10', h['tcls'], -0.00370, 5)
check('headline cell TC-PMM2* slope', 'Sec 9.10', h['star'], -0.00357, 5)
check('pre-specified demonstration set: TC-PMM2* within 2 SE', 'Sec 9.10',
      svi['demo']['within2'], 31, 0)
check('pre-specified demonstration set: TC-LS within 2 SE', 'Sec 9.10',
      svi['demo']['tcls_within2'], 36, 0)

# the same screen computed from the observables only (Lemma "testable"), 2026-08-15
so = svi['screen_obs']
check('observable-only screen: passing cells', 'Sec 9.10', so['passing'], 24, 0)
check('observable-only screen: failing cells', 'Sec 9.10', so['failing'], 130, 0)
check('observable-only screen passes -> within 2 SE', 'Sec 9.10',
      100 * so['pass_within2'], 96, 0)
check('observable-only screen fails  -> within 2 SE', 'Sec 9.10',
      100 * so['fail_within2'], 43, 0)
check('observable-only screen: median dev, passing', 'Sec 9.10',
      so['pass_median_dev'], 0.30, 2)
check('observable-only screen: median dev, failing', 'Sec 9.10',
      so['fail_median_dev'], 2.49, 2)
check('observable-only screen: out-of-sample passing cells', 'Sec 9.10',
      so['oos_pass_cells'], 23, 0)
check('observable-only screen: out-of-sample within 2 SE', 'Sec 9.10',
      100 * so['oos_pass_within2'], 96, 0)
check('observable-only screen: TC-LS within 2 SE (pass)', 'Sec 9.10',
      100 * so['tcls_within2_pass'], 100, 0)
check('observable-only screen: TC-LS within 2 SE (fail)', 'Sec 9.10',
      100 * so['tcls_within2_fail'], 100, 0)
check('verdict agreement of the two screens, 154 cells (%)', 'Sec 9.10',
      100 * so['verdict_agreement'], 99.4, 1)
check('verdict agreement of the two screens, all 462 cells (%)', 'Sec 9.10',
      100 * so['verdict_agreement_all_cells'], 85.7, 1)
check('correlation of the two screen statistics', 'Sec 9.10', so['z_corr'], 0.997, 3)
check('headline cell observable-only screen z', 'Sec 9.10', h['z_e2_obs'], -0.232, 3)

# -- Section 8.9: robustness branch (latent-covariate law, SIMEX baseline) --------
print('\nSection 8.9  robustness branch')
ROB_P = rows(EV / 'robustness' / 'paired.csv')
ROB_S = rows(EV / 'robustness' / 'summary.csv')
star = [r for r in ROB_P if r['param'] == 'b1' and r['estimator_a'] == 'tc_pmm2star_oracle'
        and r['estimator_b'] == 'tc_ls']
check('robustness: cells with TC-PMM2* vs TC-LS', 'Sec 8.9', len(star), 12, 0)
spread = max(max(r['re'] for r in star if (r['u_family'], r['lam']) == key)
             - min(r['re'] for r in star if (r['u_family'], r['lam']) == key)
             for key in {(r['u_family'], r['lam']) for r in star})
check('robustness: max spread of RE* across X laws within a cell', 'Sec 8.9',
      spread, 0.0114, 4)
check('robustness: max |RE* - ARE*|', 'Sec 8.9',
      max(abs(r['re'] - r['are_star_theory']) for r in star), 0.0182, 4)
check('robustness: max |z| of RE* against ARE*', 'Sec 8.9',
      max(abs(r['re'] - r['are_star_theory']) / r['se_re'] for r in star), 2.18, 2)
ao = [r for r in ROB_P if r['param'] == 'b1' and r['estimator_a'] == 'tc_pmm2star_adaptive'
      and r['estimator_b'] == 'tc_pmm2star_oracle']
check('robustness: adaptive/oracle MSE ratio, min', 'Sec 8.9', min(r['re'] for r in ao), 0.9918, 4)
check('robustness: adaptive/oracle MSE ratio, max', 'Sec 8.9', max(r['re'] for r in ao), 1.0075, 4)
sx = [r for r in ROB_P if r['param'] == 'b1' and r['estimator_a'] == 'simex_ls'
      and r['estimator_b'] == 'tc_ls']
check('robustness: SIMEX RE vs TC-LS at lam=0.80, min', 'Sec 8.9',
      min(r['re'] for r in sx if r['lam'] == 0.8), 1.311, 3)
check('robustness: SIMEX RE vs TC-LS at lam=0.80, max', 'Sec 8.9',
      max(r['re'] for r in sx if r['lam'] == 0.8), 1.430, 3)
check('robustness: SIMEX RE vs TC-LS at lam=0.95, min', 'Sec 8.9',
      min(r['re'] for r in sx if r['lam'] == 0.95), 1.0022, 4)
check('robustness: SIMEX RE vs TC-LS at lam=0.95, max', 'Sec 8.9',
      max(r['re'] for r in sx if r['lam'] == 0.95), 1.0096, 4)
vs = [r for r in ROB_P if r['param'] == 'b1' and r['estimator_a'] == 'tc_pmm2star_oracle'
      and r['estimator_b'] == 'simex_ls']
check('robustness: TC-PMM2*/SIMEX MSE at lam=0.80, min', 'Sec 8.9',
      min(r['re'] for r in vs if r['lam'] == 0.8), 0.680, 3)
check('robustness: TC-PMM2*/SIMEX MSE at lam=0.80, max', 'Sec 8.9',
      max(r['re'] for r in vs if r['lam'] == 0.8), 0.746, 3)
check('robustness: TC-PMM2*/SIMEX MSE at lam=0.95, min', 'Sec 8.9',
      min(r['re'] for r in vs if r['lam'] == 0.95), 0.768, 3)
check('robustness: TC-PMM2*/SIMEX MSE at lam=0.95, max', 'Sec 8.9',
      max(r['re'] for r in vs if r['lam'] == 0.95), 0.916, 3)
sxb = [r for r in ROB_S if r['param'] == 'b1' and r['estimator'] == 'simex_ls' and r['lam'] == 0.8]
check('robustness: SIMEX bias at lam=0.80 (about -0.03), max |bias|', 'Sec 8.9',
      max(abs(r['bias']) for r in sxb), 0.03, 2)
corr = [r for r in ROB_S if r['param'] == 'b1' and r['estimator'] in
        ('tc_ls', 'tc_pmm2_oracle', 'tc_pmm2star_oracle', 'tc_pmm2star_adaptive')]
check('robustness: max |bias| of corrected estimators (< 0.004)', 'Sec 8.9',
      max(abs(r['bias']) for r in corr), 0.0037, 4)
check('robustness: max MC SE of those biases', 'Sec 8.9',
      max(r['se_bias'] for r in corr), 0.0013, 4)
cov = [r for r in ROB_S if r['param'] == 'b1' and r['estimator'] in
       ('tc_ls', 'tc_pmm2star_oracle', 'tc_pmm2star_adaptive')]
check('robustness: sandwich coverage, min (%)', 'Sec 8.9', 100 * min(r['coverage'] for r in cov), 94.2, 1)
check('robustness: sandwich coverage, max (%)', 'Sec 8.9', 100 * max(r['coverage'] for r in cov), 95.6, 1)

# =========================================================================
print('\n' + '=' * 72)
if FAILURES:
    print(f'{len(FAILURES)} of {CHECKED} reported values could NOT be reproduced:\n')
    for f in FAILURES:
        print('  * ' + f)
    print('\nDo not edit the expected values to make this pass; see the module '
          'docstring.')
    sys.exit(1)
print(f'all {CHECKED} reported values reproduced from stored evidence')
