#!/usr/bin/env python3
"""Symbolic derivation of the TC-PMM2 asymptotics.

Theorem 1 (exact unbiasedness), Theorem 2 (sandwich M, Q), Proposition 3
(ARE(lambda) vs TC-LS), Lemma 4 (cumulant deconvolution).

Model:  Y = b0 + b1*X + eps,   W = X + U,   U _||_ (X, eps),  eps _||_ X.
Write  xi = X - mu_x,  u = U - E(U),  Wt = W - E(U) = mu_x + xi + u.

Everything is a polynomial in the three independent, mean-zero variables
(xi, eps, u); the expectation operator factorises over them.

Side effects:
  * prints the full derivation
  * writes are_symbolic_out.json  (results, design tables)
  * writes are_formulas.py        (generated numpy closed forms, imported by
                                   are_numeric_check.py)

Run:  .venv/bin/python theory/are_symbolic.py
"""

import json
import os

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
SEP = "=" * 78

# ----------------------------------------------------------------------------
# 0. symbols
# ----------------------------------------------------------------------------
xi, ep, u = sp.symbols("xi eps u")                       # independent, mean 0
mux, sx2 = sp.symbols("mu_x sigma_x2", positive=True)
b0, b1 = sp.symbols("beta_0 beta_1", real=True)          # free argument Theta
b0s, b1s = sp.symbols("beta_0s beta_1s", real=True)      # truth Theta*

m = {j: sp.Symbol(f"m_{j}", real=True) for j in range(2, 9)}    # eps moments
nu = {j: sp.Symbol(f"nu_{j}", real=True) for j in range(2, 9)}  # U moments
mx = {j: sp.Symbol(f"mx_{j}", real=True) for j in range(2, 9)}  # X central mom
mx[2] = sx2
GAUSS_X = {mx[3]: 0, mx[4]: 3 * sx2**2, mx[5]: 0, mx[6]: 15 * sx2**3,
           mx[7]: 0, mx[8]: 105 * sx2**4}

# generic PMM2 weights:  s(e) = a*e - b*(e^2 - c);  the estimator uses
# a = m4 - m2^2, b = m3, c = m2.  Keeping them generic lets us separate what
# unbiasedness needs from what only efficiency needs.
aw, bw, cw = sp.symbols("a b c", real=True)
PMM2_W = {aw: m[4] - m[2] ** 2, bw: m[3], cw: m[2]}
LS_W = {aw: 1, bw: 0, cw: 0}


def _mom(tab, k):
    if k == 0:
        return sp.Integer(1)
    if k == 1:
        return sp.Integer(0)
    return tab[k]


def E(expr):
    """Expectation over (xi, eps, u): independent, mean zero."""
    poly = sp.Poly(sp.expand(expr), xi, ep, u)
    out = sp.Integer(0)
    for (i, j, k), coeff in poly.terms():
        out += coeff * _mom(mx, i) * _mom(m, j) * _mom(nu, k)
    return sp.expand(out)


def E_u(expr):
    """Expectation over u only (conditional on X and eps)."""
    poly = sp.Poly(sp.expand(expr), u)
    return sp.expand(sum(coeff * _mom(nu, k) for (k,), coeff in poly.terms()))


# ----------------------------------------------------------------------------
# 1. moment-corrected powers Re(T^k) -- must satisfy E_u[Re(T^k)] = X^k
# ----------------------------------------------------------------------------
X = mux + xi
Wt = X + u

ReT = {
    1: Wt,
    2: Wt**2 - nu[2],
    3: Wt**3 - 3 * nu[2] * Wt - nu[3],
    4: Wt**4 - 6 * nu[2] * Wt**2 - 4 * nu[3] * Wt - nu[4] + 6 * nu[2] ** 2,
}
RET_CHECK = {k: sp.simplify(E_u(v) - X**k) for k, v in ReT.items()}

# ----------------------------------------------------------------------------
# 2. the corrected score
# ----------------------------------------------------------------------------
# A = Y - beta_0 is OBSERVED: during the correction step it is an opaque
# symbol.  Only the X's appearing explicitly in the score get corrected --
# correcting the latent X hidden inside Y would be a modelling error.
Asym, Xs = sp.symbols("A X_sym")
A_at_truth = (b0s - b0) + b1s * X + ep


def s_gen(e):
    return aw * e - bw * (e**2 - cw)


Psi_sym = sp.Matrix([s_gen(Asym - b1 * Xs), Xs * s_gen(Asym - b1 * Xs)])


def correct_in_X(expr):
    poly = sp.Poly(sp.expand(expr), Xs)
    return sp.expand(sum(coeff * (ReT[k] if k > 0 else 1)
                         for (k,), coeff in poly.terms()))


Psi_sub = sp.Matrix([correct_in_X(Psi_sym[i]).subs({Asym: A_at_truth})
                     for i in range(2)])
Psi_true = sp.Matrix([Psi_sym[i].subs({Asym: A_at_truth, Xs: X})
                      for i in range(2)])

A = A_at_truth
# --- the closed form written out in SPEC 3.2 -------------------------------
a_spec, b_spec = m[4] - m[2] ** 2, m[3]
Psi_spec = sp.Matrix([
    a_spec * (A - b1 * Wt)
    - b_spec * (A**2 - 2 * b1 * A * Wt + b1**2 * (Wt**2 - nu[2]) - m[2]),
    a_spec * (A * Wt - b1 * (Wt**2 - nu[2]))
    - b_spec * (A**2 * Wt - 2 * b1 * A * (Wt**2 - nu[2])
                + b1**2 * (Wt**3 - 3 * nu[2] * Wt - nu[3]) - m[2] * Wt),
])
SPEC_CHECK = sp.expand(Psi_sub.subs(PMM2_W) - Psi_spec)

# --- the "residual form": the algebraically transparent rewriting ----------
# delta = A - b1*Wt is the observable naive residual; at Theta* it equals
# eps - b1*(U - EU).
dsym = sp.Symbol("delta")
Psi_res = sp.Matrix([
    aw * dsym - bw * (dsym**2 - cw - b1**2 * nu[2]),
    aw * (Wt * dsym + b1 * nu[2])
    - bw * (Wt * dsym**2 + 2 * b1 * nu[2] * Asym - 3 * b1**2 * nu[2] * Wt
            - b1**2 * nu[3] - cw * Wt),
])
RES_CHECK = sp.expand(
    Psi_res.subs({dsym: Asym - b1 * Wt}).subs({Asym: A_at_truth}) - Psi_sub)

# ----------------------------------------------------------------------------
# 3. Theorem 1
# ----------------------------------------------------------------------------
AT = {b0: b0s, b1: b1s}
EPsi_generic = sp.Matrix([sp.factor(E(Psi_sub[i].subs(AT))) for i in range(2)])
EPsi_pmm2 = sp.Matrix([sp.simplify(x.subs(PMM2_W)) for x in EPsi_generic])

# what happens if nu_2 / nu_3 are themselves misspecified?
d2, d3 = sp.symbols("Delta_2 Delta_3", real=True)      # nu_j^used - nu_j^true
Psi_missnu = Psi_sub.subs({nu[2]: nu[2] + d2, nu[3]: nu[3] + d3})
EPsi_missnu = sp.Matrix([sp.factor(E(Psi_missnu[i].subs(AT)).subs(PMM2_W))
                         for i in range(2)])

# ----------------------------------------------------------------------------
# 4. Theorem 2:  M and Q
# ----------------------------------------------------------------------------
Mmat = sp.Matrix(2, 2, lambda i, j: -sp.simplify(
    E(sp.diff(Psi_sub[i], [b0, b1][j]).subs(AT))))
Psi0 = sp.Matrix([Psi_sub[i].subs(AT) for i in range(2)])
Qmat = sp.Matrix(2, 2, lambda i, j: sp.expand(E(Psi0[i] * Psi0[j])))

Sigma_x = sp.Matrix([[1, mux], [mux, mux**2 + sx2]])
M_STRUCT_CHECK = sp.simplify(Mmat - aw * Sigma_x)

# ----------------------------------------------------------------------------
# 5. Proposition 3
# ----------------------------------------------------------------------------
Phi = sp.expand(Psi0[1] - mux * Psi0[0])          # = sigma_x^2 * (v . Psi)
Phi_L = sp.expand(Phi.subs(LS_W))
G = sp.expand(sp.simplify((aw * Phi_L - Phi) / bw)).subs(cw, m[2])
DECOMP_CHECK = sp.expand(Phi.subs(cw, m[2]) - (aw * Phi_L - bw * G))

EPL2 = sp.expand(E(Phi_L**2))
EPLG = sp.expand(E(Phi_L * G))
EG2 = sp.expand(E(G**2))
EG1 = sp.expand(E(G))

AsyVar_b1_generic = sp.expand(E(Phi**2) / (aw**2 * sx2**2))
AsyVar_b1_PMM2 = sp.expand(AsyVar_b1_generic.subs(PMM2_W))
AsyVar_b1_LS = sp.expand(AsyVar_b1_generic.subs(LS_W))
ARE = sp.simplify(AsyVar_b1_PMM2 / AsyVar_b1_LS)

t_sym = sp.Symbol("t", real=True)
ARE_decomp = 1 - 2 * t_sym * EPLG / EPL2 + t_sym**2 * EG2 / EPL2
ARE_DECOMP_CHECK = sp.simplify(
    ARE_decomp.subs(t_sym, m[3] / (m[4] - m[2] ** 2)) - ARE)
t_opt = EPLG / EG2
ARE_opt = 1 - EPLG**2 / (EG2 * EPL2)

ZERO_NU = {nu[j]: 0 for j in nu}
g2 = 1 - m[3] ** 2 / (m[2] * (m[4] - m[2] ** 2))
ARE_LIMIT = sp.simplify(ARE.subs(ZERO_NU))
ARE_LIMIT_CHECK = sp.simplify(ARE_LIMIT - g2)

# --- transcription cross-check: the formulas as they appear in the .md -----
HAND = {
    "EPhiL2": m[2] * (sx2 + nu[2])
    + b1s**2 * (nu[2] * sx2 + nu[4] - nu[2] ** 2),
    "EPhiLG": m[3] * (sx2 + nu[2])
    - b1s**3 * (nu[3] * sx2 + nu[5] - 4 * nu[2] * nu[3])
    - 2 * b1s * m[2] * nu[3],
    "EG2": (m[4] - m[2] ** 2) * (sx2 + nu[2])
    + 4 * b1s**2 * m[2] * (nu[4] - nu[2] ** 2 + nu[2] * sx2)
    - 4 * b1s * m[3] * nu[3]
    + b1s**4 * (nu[6] + 9 * nu[2] ** 3 - 6 * nu[2] * nu[4] - nu[3] ** 2
                - nu[2] ** 2 * sx2 + nu[4] * sx2),
}
HAND_CHECK = {k: sp.simplify(HAND[k] - {"EPhiL2": EPL2, "EPhiLG": EPLG,
                                        "EG2": EG2}[k]) for k in HAND}

# ----------------------------------------------------------------------------
# 6. Lemma 4
# ----------------------------------------------------------------------------
def lemma4():
    z = ep - b1 * u
    mu2, mu3, mu4 = E(z**2), E(z**3), E(z**4)
    k2, k3, k4 = sp.expand(mu2), sp.expand(mu3), sp.expand(mu4 - 3 * mu2**2)
    ke = {2: m[2], 3: m[3], 4: m[4] - 3 * m[2] ** 2}
    ku = {2: nu[2], 3: nu[3], 4: nu[4] - 3 * nu[2] ** 2}
    checks = {j: sp.simplify(kk - (ke[j] + (-b1) ** j * ku[j]))
              for j, kk in [(2, k2), (3, k3), (4, k4)]}
    return {"k2": k2, "k3": k3, "k4": k4, "checks": checks}


# ----------------------------------------------------------------------------
# 7. designs
# ----------------------------------------------------------------------------
def central_from_cumulants(kap, jmax):
    """Central moments 2..jmax from cumulants kappa_2..kappa_jmax."""
    kap = dict(kap)
    kap[1] = 0
    mu = {0: sp.Integer(1), 1: sp.Integer(0)}
    for n in range(2, jmax + 1):
        s = kap[n]
        for k in range(1, n):
            s += sp.binomial(n - 1, k - 1) * kap[k] * mu[n - k]
        mu[n] = sp.expand(s)
    return mu


def gamma_moments(shape, scale, jmax=6):
    kap = {j: sp.factorial(j - 1) * shape * scale**j for j in range(1, jmax + 1)}
    return central_from_cumulants(kap, jmax)


def normal_moments(var, jmax=6):
    kap = {j: (var if j == 2 else 0) for j in range(1, jmax + 1)}
    return central_from_cumulants(kap, jmax)


def lognormal_central_moments(sig, var_target, jmax=6):
    """Centred lognormal exp(mu_l + sig*Z) scaled to have variance var_target."""
    s = sp.exp(sig**2)
    raw_std = {j: sp.exp(sp.Rational(j * j, 2) * sig**2) for j in range(1, jmax + 1)}
    # central moments of the *unscaled* lognormal
    cen = {}
    for n in range(2, jmax + 1):
        cen[n] = sp.expand(sum(sp.binomial(n, k) * raw_std[k] * (-raw_std[1]) ** (n - k)
                               for k in range(1, n + 1))
                           + (-raw_std[1]) ** n)
    v0 = cen[2]
    scl = sp.sqrt(var_target / v0)
    return {n: sp.simplify(cen[n] * scl**n) for n in range(2, jmax + 1)}


def u_design(name, lam, sx2v=1):
    """Central moments nu_2..nu_6 of U for reliability lam."""
    n2 = sx2v * (1 - lam) / lam
    if name == "wh_gamma":            # W&H 6.1: shape 4(1-lam)/lam, scale 1/2
        return gamma_moments(4 * (1 - lam) / lam, sp.Rational(1, 2), 6)
    if name.startswith("gamma_k"):    # fixed standardised shape
        k = sp.Rational(name.split("gamma_k")[1])
        return gamma_moments(k, sp.sqrt(n2 / k), 6)
    if name == "normal":
        return normal_moments(n2, 6)
    if name == "lognormal":           # W&H 6.3 style
        return lognormal_central_moments(sp.Rational(1, 2), n2, 6)
    raise ValueError(name)


def eps_design(c3):
    """Centred gamma eps standardised to m2 = 1 with skewness c3."""
    if c3 == 0:
        return normal_moments(1, 4)
    k = 4 / c3**2
    return gamma_moments(k, 1 / sp.sqrt(k), 4)


def main():
    out = {}

    print(SEP)
    print("1. moment-corrected powers Re(T^k):  E_u[Re(T^k)] - X^k")
    for k, d in RET_CHECK.items():
        print(f"   k={k}: {d}")
        assert d == 0
    print("   -> verified for k = 1..4")

    print(SEP)
    print("2. SPEC 3.2 closed form vs the substitution rule (must be zero)")
    print(f"   component 0: {SPEC_CHECK[0]}")
    print(f"   component 1: {SPEC_CHECK[1]}")
    spec_ok = bool(SPEC_CHECK[0] == 0 and SPEC_CHECK[1] == 0)
    print(f"   -> SPEC 3.2 algebra correct: {spec_ok}")
    print(f"   residual form (Psi in terms of delta) check: "
          f"{[RES_CHECK[0], RES_CHECK[1]]}")
    out["spec_3_2_ok"] = spec_ok
    out["residual_form_ok"] = bool(RES_CHECK[0] == 0 and RES_CHECK[1] == 0)

    print(SEP)
    print("3. Theorem 1.  E[Psi(Theta*)] with GENERIC weights (a, b, c):")
    print(f"   E[Psi^(0)] = {EPsi_generic[0]}")
    print(f"   E[Psi^(1)] = {EPsi_generic[1]}")
    print(f"   with PMM2 weights: {EPsi_pmm2.T.tolist()}")
    print("   with MISSPECIFIED U moments nu_2 -> nu_2+D2, nu_3 -> nu_3+D3:")
    print(f"   E[Psi^(0)] = {EPsi_missnu[0]}")
    print(f"   E[Psi^(1)] = {EPsi_missnu[1]}")
    out["unbiased_generic"] = [str(EPsi_generic[0]), str(EPsi_generic[1])]
    out["unbiased_missnu"] = [str(EPsi_missnu[0]), str(EPsi_missnu[1])]

    print(SEP)
    print("4. Theorem 2.  M = -E[dPsi/dTheta^T]:")
    sp.pprint(Mmat)
    print(f"   M - a*Sigma_x = {M_STRUCT_CHECK.tolist()}")
    print("   Q = E[Psi Psi^T]:")
    for i in range(2):
        for j in range(i, 2):
            print(f"   Q[{i},{j}] = {sp.expand(Qmat[i, j])}")
    print("   NOTE: only mu_x and sigma_x^2 appear -- no distributional")
    print("   assumption on X is needed (mx_3, mx_4 cancel).")
    out["M"] = [[str(Mmat[i, j]) for j in range(2)] for i in range(2)]
    out["Q_generic"] = [[str(sp.expand(Qmat[i, j])) for j in range(2)]
                        for i in range(2)]
    out["Q_needs_only_mu_sigma"] = not any(
        Qmat[i, j].has(mx[3]) or Qmat[i, j].has(mx[4])
        for i in range(2) for j in range(2))

    print(SEP)
    print("5. Proposition 3.")
    print(f"   E[Phi_L^2] = {EPL2}")
    print(f"   E[Phi_L G] = {EPLG}")
    print(f"   E[G]       = {EG1}")
    print(f"   E[G^2]     = {EG2}")
    print(f"   Phi - (a Phi_L - b G) = {DECOMP_CHECK}")
    print("   hand-transcribed formulas minus sympy (must all be 0):")
    for k, v in HAND_CHECK.items():
        print(f"      {k}: {v}")
        assert v == 0, f"transcription error in {k}"
    print(f"   ARE(nu->0) = {ARE_LIMIT}   (minus g2: {ARE_LIMIT_CHECK})")
    print(f"   ARE = 1 - 2 t C + t^2 V check: {ARE_DECOMP_CHECK}")
    print(f"   t_opt = C/V (weight ratio that would be optimal at lambda<1)")
    out.update({"EPhiL2": str(EPL2), "EPhiLG": str(EPLG), "EG2": str(EG2),
                "ARE_limit_ok": bool(ARE_LIMIT_CHECK == 0),
                "ARE_decomp_ok": bool(ARE_DECOMP_CHECK == 0),
                "EG_zero": bool(EG1 == 0)})

    print(SEP)
    print("6. Lemma 4.  e_obs = eps - beta1*(U - EU)")
    lem = lemma4()
    for j in (2, 3, 4):
        print(f"   kappa{j}(e_obs) = {lem[f'k{j}']}")
    for j, c in lem["checks"].items():
        print(f"   check kappa_j(e_obs) - [kappa_j(eps) + (-b1)^j kappa_j(U)] "
              f"= {c}   (j={j})")
        assert c == 0
    out["lemma4_ok"] = True

    print(SEP)
    print("7. designs")
    lam = sp.Symbol("lambda", positive=True)
    muU = u_design("wh_gamma", lam)
    print("   W&H gamma design, shape=4(1-lam)/lam, scale=1/2:")
    for j in range(2, 7):
        e = sp.simplify(muU[j])
        print(f"     nu_{j} = {e}")
        out[f"nu{j}_wh_gamma"] = str(e)
    print("   standardised skewness/excess-kurtosis of U in that design:")
    g3u = sp.simplify(muU[3] / muU[2] ** sp.Rational(3, 2))
    g4u = sp.simplify(muU[4] / muU[2] ** 2 - 3)
    print(f"     gamma3(U) = {g3u},  gamma4(U) = {g4u}")
    out["wh_gamma_skew"] = str(g3u)
    out["wh_gamma_exkurt"] = str(g4u)

    # ---- numeric ARE tables ----
    def are_num(lamv, c3v, b1v, uname, sx2v=1.0):
        me = eps_design(sp.nsimplify(c3v))
        mu_u = u_design(uname, sp.Float(lamv), sp.Float(sx2v))
        sub = {mux: 1, sx2: sx2v, b1s: b1v}
        sub.update({m[j]: me[j] for j in (2, 3, 4)})
        sub.update({nu[j]: mu_u[j] for j in range(2, 7)})
        are = float(sp.N(ARE.subs(sub)))
        aopt = float(sp.N(ARE_opt.subs(PMM2_W).subs(sub)))
        topt = float(sp.N(t_opt.subs(PMM2_W).subs(sub)))
        g2v = float(sp.N(g2.subs({m[j]: me[j] for j in (2, 3, 4)})))
        tuse = float(sp.N((m[3] / (m[4] - m[2] ** 2)).subs(
            {m[j]: me[j] for j in (2, 3, 4)})))
        return dict(ARE=are, ARE_opt=aopt, t_opt=topt, t_used=tuse, g2=g2v)

    tables = []
    print("\n   ARE(lambda) tables. 'ARE' = TC-PMM2 / TC-LS for beta_1;")
    print("   'ARE_opt' = the same score family with the weight ratio t")
    print("   re-optimised at that lambda (lower envelope).")
    for uname in ["wh_gamma", "gamma_k4", "normal", "lognormal"]:
        for b1v in [1.0, -1.0]:
            print(f"\n   --- U = {uname}, beta_1 = {b1v:+.0f} ---")
            print("     c3   lambda      ARE       g2   ARE_opt    verdict")
            for c3v in [0.5, 1.0, 1.5]:
                for lamv in [0.80, 0.90, 0.95, 0.99]:
                    r = are_num(lamv, c3v, b1v, uname)
                    r.update(dict(c3=c3v, lam=lamv, beta1=b1v, U=uname))
                    tables.append(r)
                    verdict = ("LOSS" if r["ARE"] > 1 else
                               ("below g2" if r["ARE"] < r["g2"] else "gain"))
                    print(f"    {c3v:4.2f}  {lamv:5.2f}  {r['ARE']:8.4f} "
                          f"{r['g2']:8.4f}  {r['ARE_opt']:8.4f}   {verdict}")
    out["ARE_tables"] = tables

    # break-even reliability lambda*(c3) for the W&H gamma design
    print("\n   break-even reliability lambda* (ARE = 1), W&H gamma U, beta1=+1:")
    brk = {}
    for c3v in [0.5, 1.0, 1.5]:
        lo, hi = 0.5, 0.999999
        f = lambda L: are_num(L, c3v, 1.0, "wh_gamma")["ARE"] - 1.0
        if f(lo) * f(hi) < 0:
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if f(lo) * f(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            brk[c3v] = 0.5 * (lo + hi)
            print(f"     c3={c3v:4.2f}:  lambda* = {brk[c3v]:.4f}  "
                  f"(TC-PMM2 loses for lambda < lambda*)")
        else:
            print(f"     c3={c3v:4.2f}:  no crossing in [0.5, 1)")
    out["breakeven_lambda_wh_gamma"] = brk

    # first-order expansion in nu_2 for the W&H design
    print("\n   expansion around lambda = 1 (W&H gamma design, beta1=+1, c3=1):")
    for lamv in [0.999, 0.9999]:
        r = are_num(lamv, 1.0, 1.0, "wh_gamma")
        print(f"     lambda={lamv}:  ARE={r['ARE']:.6f}, g2={r['g2']:.6f}, "
              f"ARE-g2={r['ARE'] - r['g2']:.6f}, (1-lambda)={1 - lamv:.1e}")

    print(SEP)
    print("8. explicit ARE(lambda) for the SPEC 6.1 gamma/gamma design")
    c3s = sp.Symbol("c_3", positive=True)
    me_s = eps_design(c3s)
    mu_u_s = u_design("wh_gamma", lam)
    sub_s = {mux: 1, sx2: 1, b1s: 1}
    sub_s.update({m[j]: me_s[j] for j in (2, 3, 4)})
    sub_s.update({nu[j]: mu_u_s[j] for j in range(2, 7)})
    ARE_wh = sp.cancel(sp.simplify(ARE.subs(sub_s)))
    print("   X ~ N(1,1), beta_1 = +1, eps ~ centred gamma (m2=1, skew c3),")
    print("   U ~ centred gamma(4(1-lam)/lam, 1/2):")
    print(f"   ARE(lambda, c3) = {ARE_wh}")
    out["ARE_wh_gamma_rational"] = str(ARE_wh)
    num, den = sp.fraction(sp.together(ARE_wh))
    out["ARE_wh_gamma_num"] = str(sp.expand(num))
    out["ARE_wh_gamma_den"] = str(sp.expand(den))
    print(f"   numerator   = {sp.factor(sp.expand(num))}")
    print(f"   denominator = {sp.factor(sp.expand(den))}")

    print("\n   expansion of ARE around lambda = 1 (nu_2 = (1-lambda)/lambda):")
    eps_l = sp.Symbol("varepsilon", positive=True)          # = 1 - lambda
    ser = sp.series(ARE_wh.subs(lam, 1 - eps_l), eps_l, 0, 2).removeO()
    ser = sp.expand(sp.simplify(ser))
    print(f"   ARE = {sp.collect(ser, eps_l)}")
    out["ARE_wh_gamma_series"] = str(sp.collect(ser, eps_l))
    K = sp.simplify(sp.diff(ARE_wh.subs(lam, 1 - eps_l), eps_l).subs(eps_l, 0))
    print(f"   slope K = dARE/d(1-lambda)|_{{lambda=1}} = {sp.simplify(K)}")
    out["ARE_wh_gamma_slope"] = str(sp.simplify(K))
    for c3v in [0.5, 1.0, 1.5]:
        print(f"     c3={c3v}:  K = {float(sp.N(K.subs(c3s, c3v))):.4f}")

    print("\n   generic 'regular' U family (standardised shape fixed):")
    print("   nu_3 = k3*nu_2^{3/2}, nu_4 = k4*nu_2^2, nu_5 = k5*nu_2^{5/2},")
    print("   nu_6 = k6*nu_2^3;  expand ARE in nu_2:")
    k3s, k4s, k5s, k6s = sp.symbols("k_3 k_4 k_5 k_6", real=True)
    v = sp.Symbol("varv", positive=True)                    # = nu_2
    reg = {nu[2]: v, nu[3]: k3s * v ** sp.Rational(3, 2), nu[4]: k4s * v**2,
           nu[5]: k5s * v ** sp.Rational(5, 2), nu[6]: k6s * v**3}
    ARE_reg = ARE.subs(reg)
    s2 = sp.series(ARE_reg, v, 0, 2).removeO()
    s2 = sp.collect(sp.expand(sp.simplify(s2)), v)
    print(f"   ARE = {s2}")
    out["ARE_regular_series"] = str(s2)

    print("\n   condition for TC-PMM2 to beat TC-LS (assuming m_3 > 0):")
    print("   ARE < 1  <=>  m_3 * E[G^2]  <  2 (m_4 - m_2^2) * E[Phi_L G]")

    # ------------------------------------------------------------------
    # code generation for the numeric cross-check
    # ------------------------------------------------------------------
    ren = {mux: sp.Symbol("mux"), sx2: sp.Symbol("sx2"), b1s: sp.Symbol("b1"),
           b0s: sp.Symbol("b0")}
    ren.update({m[j]: sp.Symbol(f"m{j}") for j in m})
    ren.update({nu[j]: sp.Symbol(f"n{j}") for j in nu})

    def src(expr):
        return sp.pycode(sp.expand(expr.subs(PMM2_W).subs(GAUSS_X).subs(ren)))

    def src_ls(expr):
        return sp.pycode(sp.expand(expr.subs(LS_W).subs(GAUSS_X).subs(ren)))

    lines = [
        '"""AUTO-GENERATED by are_symbolic.py -- do not edit by hand."""',
        "",
        "",
        "def M_pmm2(mux, sx2, b1, m2, m3, m4, n2, n3, n4, n5, n6):",
        f"    return [[{src(Mmat[0, 0])}, {src(Mmat[0, 1])}],",
        f"            [{src(Mmat[1, 0])}, {src(Mmat[1, 1])}]]",
        "",
        "",
        "def Q_pmm2(mux, sx2, b1, m2, m3, m4, n2, n3, n4, n5, n6):",
        f"    return [[{src(Qmat[0, 0])}, {src(Qmat[0, 1])}],",
        f"            [{src(Qmat[1, 0])}, {src(Qmat[1, 1])}]]",
        "",
        "",
        "def M_ls(mux, sx2, b1, m2, m3, m4, n2, n3, n4, n5, n6):",
        f"    return [[{src_ls(Mmat[0, 0])}, {src_ls(Mmat[0, 1])}],",
        f"            [{src_ls(Mmat[1, 0])}, {src_ls(Mmat[1, 1])}]]",
        "",
        "",
        "def Q_ls(mux, sx2, b1, m2, m3, m4, n2, n3, n4, n5, n6):",
        f"    return [[{src_ls(Qmat[0, 0])}, {src_ls(Qmat[0, 1])}],",
        f"            [{src_ls(Qmat[1, 0])}, {src_ls(Qmat[1, 1])}]]",
        "",
        "",
        "def asyvar_b1_pmm2(mux, sx2, b1, m2, m3, m4, n2, n3, n4, n5, n6):",
        f"    return {src(AsyVar_b1_generic)}",
        "",
        "",
        "def asyvar_b1_ls(mux, sx2, b1, m2, m3, m4, n2, n3, n4, n5, n6):",
        f"    return {src_ls(AsyVar_b1_generic)}",
        "",
        "",
        "def scalars(mux, sx2, b1, m2, m3, m4, n2, n3, n4, n5, n6):",
        '    """E[Phi_L^2], E[Phi_L G], E[G^2] (Proposition 3 building blocks)."""',
        f"    return ({sp.pycode(sp.expand(EPL2.subs(ren)))},",
        f"            {sp.pycode(sp.expand(EPLG.subs(ren)))},",
        f"            {sp.pycode(sp.expand(EG2.subs(ren)))})",
        "",
    ]
    with open(os.path.join(HERE, "are_formulas.py"), "w") as fh:
        fh.write("\n".join(lines))
    print(SEP)
    print("wrote are_formulas.py (generated closed forms)")

    with open(os.path.join(HERE, "are_symbolic_out.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote are_symbolic_out.json")


if __name__ == "__main__":
    main()
