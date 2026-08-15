"""Run the TC-PMM2 EIV Monte Carlo study (SPEC 6.2).

Modes
  quick          M = 200,  c3 = 1.0, both U-designs, both signs of b1     (K1 check)
  confirmatory   M = 2000, full grid                                       (all paper numbers)
  lambda_star    M = 2000, dense lam grid at n = 1000, to locate break-even lam*
  sensitivity    M = 2000, U-moment misspecification, delta in 0..0.6
  negative_nu3   M = 2000, LEFT-skewed U (nu3 < 0): mirror of the confirmatory
                 main grid + the NHANES application regime                (2026-08-15)
  asymptotics    closed-form ARE(lam) + lambda* (no replications)
  robustness     M = 2000, c3 = 1.0, b1 = +1, n = 1000, both gamma U-designs,
                 lam in {0.80, 0.95}, latent covariate X in {Gaussian, skewed
                 gamma, Student t(5)} (same mean and variance), plus a SIMEX
                 baseline; adaptive TC-PMM2* estimates sigma_x^2 as usual
                 (2026-08-15, PaperMentor round-4 requests)

Design factors (2026-08-14 amendment).  The U family is a MAIN factor, not a
nuisance: 'gamma_wh' ties U's standardised shape to lam (the W&H 6.1 design --
kept as the adversarial cell), while 'gamma_fixed' holds the shape constant so
the method's behaviour is separable from that design artefact.  The sign of b1
is likewise a main factor: it enters E[Phi_L G] through b1 and b1^3 and flips
the verdict for the fixed-weight estimator.

Writes evidence/<run_name>/ : estimates_raw.csv.gz, summary.csv, paired.csv,
manifest.json.  Nothing under evidence/ is ever hand-edited.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dgp import EPS_SPECS, Cell, make_cell            # noqa: E402
from estimators import ESTIMATOR_NAMES, asymptotic_are  # noqa: E402
from mc_engine import run_cell                        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / 'evidence'

LAM_FULL = (0.80, 0.90, 0.95)
N_FULL = (250, 500, 1000)
EPS_FULL = ('gauss', 'g_c3_0.5', 'g_c3_1.0', 'g_c3_1.5')
U_MAIN = ('gamma_wh', 'gamma_fixed')     # adversarial (shape tied to lam) + decoupled
U_MIRROR = ('gamma_wh_neg', 'gamma_fixed_neg')   # nu3 < 0 mirrors of the two above
B1_FULL = (1.0, -1.0)                    # sign of b1 is a MAIN factor, not sensitivity
LAM_STAR = (0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98)


def build_grid(mode: str) -> list[Cell]:
    cells: list[Cell] = []
    if mode == 'quick':
        for fam in U_MAIN:
            for b1 in B1_FULL:
                for lam in (0.80, 0.95):
                    for n in (250, 1000):
                        cells.append(make_cell('g_c3_1.0', fam, lam, n, b1=b1))
    elif mode == 'confirmatory':
        for ek in EPS_FULL:                       # main grid, both U-designs, both signs
            for fam in U_MAIN:
                for b1 in B1_FULL:
                    for lam in LAM_FULL:
                        for n in N_FULL:
                            cells.append(make_cell(ek, fam, lam, n, b1=b1))
        for ek in ('g_c3_1.0', 'g_c3_1.5'):       # G6: heavy-tailed lognormal U
            for b1 in B1_FULL:
                for lam in LAM_FULL:
                    for n in N_FULL:
                        cells.append(make_cell(ek, 'lognormal', lam, n, b1=b1))
    elif mode == 'negative_nu3':
        # 2026-08-15 coverage gap: every cell above uses a RIGHT-skewed U.
        # (a) exact mirror of the confirmatory main grid, U -> -U (nu3, nu5 < 0);
        #     each cell here pairs with the confirmatory cell that has the same
        #     (eps, base U family, lam, n) and the OPPOSITE sign of b1.
        for ek in EPS_FULL:
            for fam in U_MIRROR:
                for b1 in B1_FULL:
                    for lam in LAM_FULL:
                        for n in N_FULL:
                            cells.append(make_cell(ek, fam, lam, n, b1=b1))
        # (b) the application regime itself: NHANES 2013-14 diastolic BP
        #     (lam = 0.879, gamma3(U) = -0.378, c3(eps) = +1.284, c4 = 3.42),
        #     with its own positive-skew mirror partner in the same run.
        from dgp import REALDATA_LAM
        for fam in ('gamma_real_neg', 'gamma_real'):
            for b1 in B1_FULL:
                for n in N_FULL:
                    cells.append(make_cell('real_dbp', fam, REALDATA_LAM, n, b1=b1))
    elif mode == 'lambda_star':                   # dense lam grid to locate break-even
        for ek in ('g_c3_0.5', 'g_c3_1.0', 'g_c3_1.5'):
            for fam in U_MAIN:
                for b1 in B1_FULL:
                    for lam in LAM_STAR:
                        cells.append(make_cell(ek, fam, lam, 1000, b1=b1))
    elif mode == 'sensitivity':
        for delta in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            for fam in U_MAIN:
                for lam in (0.80, 0.95):
                    cells.append(make_cell('g_c3_1.0', fam, lam, 1000, delta=delta))
    elif mode == 'robustness':
        from dgp import X_FAMILIES
        for xf in X_FAMILIES:
            for fam in U_MAIN:
                for lam in (0.80, 0.95):
                    cells.append(make_cell('g_c3_1.0', fam, lam, 1000, b1=1.0,
                                           x_family=xf))
    else:
        raise ValueError(mode)
    return cells


def git_hash() -> str:
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                       text=True).strip()
    except Exception:
        return 'unavailable'


def _worker(args):
    cell, m, extra = args
    t0 = time.time()
    res = run_cell(cell, m, extra=extra)
    res['wall'] = time.time() - t0
    return res


def write_run(run_name, results, meta):
    out = EVIDENCE / run_name
    out.mkdir(parents=True, exist_ok=True)

    with gzip.open(out / 'estimates_raw.csv.gz', 'wt', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['cell_id', 'eps_key', 'u_family', 'lam', 'n', 'b1_true', 'delta',
                     'rep', 'estimator', 'b0_hat', 'b1_hat', 'se_b0', 'se_b1',
                     'converged', 'adapt_fallback', 'adapt_clip',
                     'star_fallback_oracle', 'star_fallback_adaptive',
                     'star_t_oracle', 'star_t_adaptive'])
        for res in results:
            c = res['cell']
            for r in range(res['n_rep']):
                for j, nm in enumerate(res.get('names', ESTIMATOR_NAMES)):
                    wr.writerow([c.cell_id, c.eps_key, c.u_family, c.lam, c.n, c.b1,
                                 c.delta, r, nm,
                                 f"{res['beta'][r, j, 0]:.10g}", f"{res['beta'][r, j, 1]:.10g}",
                                 f"{res['se'][r, j, 0]:.10g}", f"{res['se'][r, j, 1]:.10g}",
                                 int(res['conv'][r, j]),
                                 int(res['flags'][r, 0]), int(res['flags'][r, 1]),
                                 int(res['flags'][r, 2]), int(res['flags'][r, 3]),
                                 f"{res['star_t'][r, 0]:.10g}",
                                 f"{res['star_t'][r, 1]:.10g}"])

    for fname, key in (('summary.csv', 'summary'), ('paired.csv', 'paired')):
        rows = [row for res in results for row in res[key]]
        with open(out / fname, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

    with open(out / 'manifest.json', 'w') as fh:
        json.dump(meta, fh, indent=2, default=str)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['quick', 'confirmatory', 'sensitivity', 'asymptotics',
                             'lambda_star', 'negative_nu3', 'robustness'])
    ap.add_argument('--reps', type=int, default=None,
                    help='override M (any reduction is recorded in the manifest)')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--reduction-reason', default='')
    args = ap.parse_args()

    if args.mode == 'asymptotics':
        from estimators import theory_are, lambda_star as lam_star_fn
        lam_dense = [round(x, 4) for x in np.arange(0.60, 0.9951, 0.005)] + [0.999]
        rows, star_rows = [], []
        for ek in EPS_FULL:
            for fam in U_MAIN + ('lognormal',):
                for b1 in (1.0, -1.0):
                    for lam in lam_dense:
                        c = make_cell(ek, fam, lam, 1000, b1=b1)
                        rows.append(dict(eps_key=ek, u_family=fam, b1=b1, lam=lam,
                                         **theory_are(c)))
                    ls = lam_star_fn(ek, fam, b1)
                    star_rows.append(dict(eps_key=ek, u_family=fam, b1=b1,
                                          lambda_star_theory=('' if ls is None
                                                              else round(ls, 5))))
                    print(f'{ek:10s} {fam:12s} b1={b1:+.0f}  lambda* = '
                          f'{"none (never crosses 1)" if ls is None else f"{ls:.5f}"}')
        # independent validation of the closed form at a few cells
        val = []
        for ek in ('g_c3_1.0', 'g_c3_1.5'):
            for fam in U_MAIN:
                for lam in LAM_FULL:
                    c = make_cell(ek, fam, lam, 1000)
                    th, mc = theory_are(c), asymptotic_are(c, 2_000_000)
                    val.append(dict(eps_key=ek, u_family=fam, lam=lam,
                                    are_fixed_closed_form=th['are_fixed'],
                                    are_fixed_big_sample=mc['are_b1'],
                                    abs_diff=abs(th['are_fixed'] - mc['are_b1'])))
        out = EVIDENCE / (args.run_name or 'asymptotics')
        out.mkdir(parents=True, exist_ok=True)
        for fname, data in (('asymptotic_are.csv', rows),
                            ('lambda_star_theory.csv', star_rows),
                            ('closed_form_validation.csv', val)):
            with open(out / fname, 'w', newline='') as fh:
                wr = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
                wr.writeheader()
                wr.writerows(data)
        with open(out / 'manifest.json', 'w') as fh:
            json.dump(dict(mode='asymptotics', git_hash=git_hash(),
                           python=sys.executable, numpy=np.__version__,
                           n_rows=len(rows),
                           note='closed-form ARE (eq. 4.1, blocks 3.1-3.3), typed '
                                'independently in estimators.are_blocks and asserted '
                                'equal to theory/are_formulas.scalars by verify.py; '
                                'closed_form_validation.csv compares it against a '
                                '2e6-sample empirical sandwich'),
                      fh, indent=2)
        print(f'\nwrote {out}  ({len(rows)} rows)')
        return

    default_m = {'quick': 200, 'confirmatory': 2000, 'sensitivity': 2000,
                 'lambda_star': 2000, 'negative_nu3': 2000, 'robustness': 2000}[args.mode]
    m = args.reps or default_m
    cells = build_grid(args.mode)
    run_name = args.run_name or args.mode
    extra = ('simex_ls',) if args.mode == 'robustness' else ()

    print(f'mode={args.mode}  cells={len(cells)}  M={m}  jobs={args.jobs}')
    t0 = time.time()
    tasks = [(c, m, extra) for c in cells]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_worker, tasks))
    else:
        results = [_worker(t) for t in tasks]
    wall = time.time() - t0

    import scipy
    meta = dict(
        mode=args.mode, run_name=run_name, m_rep=m, m_rep_default=default_m,
        m_reduced=(m != default_m),
        reduction_reason=args.reduction_reason or ('none' if m == default_m else 'UNDOCUMENTED'),
        n_cells=len(cells), cells=[c.describe() for c in cells],
        estimators=list(ESTIMATOR_NAMES) + list(extra),
        base_entropy=20260814,
        seeding='SeedSequence([base, key(stream), key(cfg), n, rep]); CRN across '
                'estimators within a replication; X,eps shared across lam',
        wall_time_sec=round(wall, 2),
        per_cell_wall_sec={r['cell'].cell_id: round(r['wall'], 2) for r in results},
        git_hash=git_hash(), python_executable=sys.executable,
        python_version=sys.version, platform=platform.platform(),
        numpy=np.__version__, scipy=scipy.__version__,
        env_KU_PMM_EIV_PYTHON=os.environ.get('KU_PMM_EIV_PYTHON', '(unset)'),
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    )
    out = write_run(run_name, results, meta)
    print(f'wall={wall:.1f}s  ->  {out}')


if __name__ == '__main__':
    main()
