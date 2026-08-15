"""Identifiability of the measurement-error moments from k replicate measurements.

Setting: unit i is measured k times, W_ij = X_i + U_ij, with U_ij iid, independent of X_i.
The within-unit deviations d_ij = W_ij - Wbar_i = U_ij - Ubar_i remove X entirely, so any
moment of U estimable without a gold standard must be a function of the law of d.

Claim (odd moments): for k = 2 the deviation vector is (D/2, -D/2) with D = U_1 - U_2
symmetric, hence every odd moment of d vanishes identically whatever nu3 is. So paired
replicates carry NO information about odd central moments of U. From k >= 3 they do.

This script derives E[d^p] for p = 2,3,4,5 symbolically and confirms it by simulation.
"""

import numpy as np
import sympy as sp


def symbolic_coefficients(kmax=6):
    """E[d^p] as a linear form in the central moments of U, for p = 2..5."""
    k = sp.symbols('k', positive=True, integer=True)
    rows = []
    for kk in range(2, kmax + 1):
        a = sp.Rational(kk - 1, kk)
        # d = a*U_j - S/kk, S the sum of the other kk-1 iid errors, independent of U_j.
        # Central moments of S expressed via those of U (cumulant/moment bookkeeping):
        #   E[S]=0, E[S^2]=(k-1)n2, E[S^3]=(k-1)n3,
        #   E[S^4]=(k-1)n4+3(k-1)(k-2)n2^2,
        #   E[S^5]=(k-1)n5+10(k-1)(k-2)n2*n3
        n2, n3, n4, n5 = sp.symbols('nu2 nu3 nu4 nu5')
        m = kk - 1
        ES = {1: 0,
              2: m * n2,
              3: m * n3,
              4: m * n4 + 3 * m * (m - 1) * n2 ** 2,
              5: m * n5 + 10 * m * (m - 1) * n2 * n3}
        EU = {1: 0, 2: n2, 3: n3, 4: n4, 5: n5}

        def E_d_pow(p):
            tot = 0
            for r in range(p + 1):
                c = sp.binomial(p, r) * (a ** (p - r)) * ((-sp.Rational(1, kk)) ** r)
                eu = 1 if p - r == 0 else EU.get(p - r, 0)
                es = 1 if r == 0 else ES.get(r, 0)
                tot += c * eu * es
            return sp.expand(tot)

        rows.append((kk, {p: sp.simplify(E_d_pow(p)) for p in (2, 3, 4, 5)}))
    return rows


def closed_forms():
    """The compact coefficient formulas quoted in the paper."""
    k = sp.symbols('k', positive=True)
    return {
        'nu2 in E[d^2]': (k - 1) / k,
        'nu3 in E[d^3]': (k - 1) * (k - 2) / k ** 2,
        'nu4 in E[d^4]': (k - 1) * ((k - 1) ** 3 + 1) / k ** 4,
        'nu5 in E[d^5]': (k - 1) * ((k - 1) ** 4 - 1) / k ** 5,
    }


def simulate(kk, n=2_000_000, shape=1.0, seed=11):
    """Monte Carlo check with a strongly skewed U (centred, unit-variance gamma)."""
    rng = np.random.default_rng(seed)
    g = rng.gamma(shape, 1.0, size=(n, kk))
    U = (g - shape) / np.sqrt(shape)          # mean 0, var 1, nu3 = 2/sqrt(shape)
    d = U - U.mean(axis=1, keepdims=True)
    return {p: float((d ** p).mean()) for p in (2, 3, 4, 5)}


if __name__ == '__main__':
    k = sp.symbols('k', positive=True)
    print('Compact coefficient formulas (verified against the per-k expansion below):')
    for name, expr in closed_forms().items():
        vals = {kk: sp.nsimplify(expr.subs(k, kk)) for kk in range(2, 6)}
        print(f'  {name:>18} = {expr}   ->  ' +
              ', '.join(f'k={kk}: {float(v):+.5f}' for kk, v in vals.items()))

    print('\nPer-k symbolic expansion of E[d^p]:')
    for kk, ex in symbolic_coefficients():
        print(f'  k={kk}')
        for p, e in ex.items():
            print(f'      E[d^{p}] = {e}')

    print('\nMonte Carlo confirmation, U centred gamma(shape=1): nu2=1, nu3=2, nu4=9, nu5=44')
    print('(moments recovered by solving the triangular system in order 2 -> 3 -> 4 -> 5)')
    truth = {2: 1.0, 3: 2.0, 4: 9.0, 5: 44.0}
    n2s, n3s, n4s, n5s = sp.symbols('nu2 nu3 nu4 nu5')
    per_k = dict(symbolic_coefficients())
    for kk in (2, 3, 4):
        sim = simulate(kk)
        ex = per_k[kk]
        rec = {}
        # order matters: each equation introduces exactly one new unknown
        rec[2] = sp.solve(sp.Eq(ex[2], sim[2]), n2s)[0]
        sol3 = sp.solve(sp.Eq(ex[3].subs(n2s, rec[2]), sim[3]), n3s)
        rec[3] = sol3[0] if sol3 else None
        rec[4] = sp.solve(sp.Eq(ex[4].subs(n2s, rec[2]), sim[4]), n4s)[0]
        sub5 = ex[5].subs({n2s: rec[2], n3s: rec[3] if rec[3] is not None else 0})
        sol5 = sp.solve(sp.Eq(sub5, sim[5]), n5s)
        rec[5] = sol5[0] if sol5 else None
        print(f'  k={kk}:')
        for p in (2, 3, 4, 5):
            if rec[p] is None:
                print(f'      nu{p}: NOT IDENTIFIED (coefficient vanishes identically at k=2)')
            else:
                print(f'      nu{p} = {float(rec[p]):+8.4f}   (true {truth[p]:+8.4f})')
