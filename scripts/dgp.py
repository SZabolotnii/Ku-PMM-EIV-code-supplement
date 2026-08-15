"""Data-generating processes for the TC-PMM2 linear EIV study (SPEC 6.1).

Model:  Y = b0 + b1*X + eps ,  W = X + U ,  X ~ N(1,1),  U _||_ (X, eps).

Moment notation is frozen (CLAUDE.md 3): ``m_j`` = central moments of the
regression error eps, ``nu_j`` = central moments of the measurement error U.
Both error generators are centred (E eps = E U = 0), so ``Wt = W`` and the
raw/central moments of U coincide for j >= 2.

Every configuration exposes its EXACT theoretical moments; the oracle
estimators and the verification gates read them from here, never from data.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, asdict, field

import numpy as np

VAR_X = 1.0          # X ~ N(MEAN_X, VAR_X); reliability lam = VAR_X/(VAR_X+nu2)
MEAN_X = 1.0
LOGNORMAL_SIGMA = 1.0   # shape of the heavy-tail U cell (documented choice)
GAMMA_FIXED_SHAPE = 4.0  # fixed-shape U design: skew 1.0, excess kurtosis 1.5,
                         # INDEPENDENT of lam (decouples U's shape from reliability)

# --------------------------------------------------------------------------
# NEGATIVE-nu3 EXTENSION (2026-08-15).  Every cell run before this date used a
# right-skewed U (nu3 > 0).  The theory says the sign of the whole effect is
# carried by sign(b1 * gamma3(U) * m3), so mirroring U (U -> -U, which flips
# the ODD central moments nu3, nu5 and leaves nu2, nu4, nu6 untouched) must be
# exactly equivalent to flipping the sign of b1.  Families ending in '_neg'
# are those mirrors; nothing about the existing families is changed.
#
# 'gamma_real' is calibrated to the NHANES 2013-14 diastolic-BP replicates
# (source_notes/REALDATA_CANDIDATES_2026-08-14.md 3.2): nu2 = 20.04,
# nu3 = -33.90, i.e. gamma3(U) = -0.3779.  Only the DIMENSIONLESS invariants
# are matched -- the study keeps its own scale (sx2 = 1, m2 = 1) -- so the
# calibration transfers through (lam, gamma3(U), c3(eps), c4(eps)).
NEG_SUFFIX = '_neg'
REALDATA_NU2 = 20.04          # NHANES 2013-14 DBP, within-visit replicates
REALDATA_NU3 = -33.90
REALDATA_U_SKEW = REALDATA_NU3 / REALDATA_NU2 ** 1.5        # -0.377897...
GAMMA_REAL_SHAPE = 4.0 / REALDATA_U_SKEW ** 2               # 28.011: |skew| match
REALDATA_LAM = 0.879          # var(X)/(var(X)+nu2) in the NHANES design
REALDATA_C3_EPS = 1.284       # skewness of the regression error there
REALDATA_C4_EPS = 3.42        # excess kurtosis (=> g2 = 0.696)

# fixed-shape gamma designs: shape constant in lam, scale set to hit nu2
FIXED_SHAPES = {'gamma_fixed': GAMMA_FIXED_SHAPE, 'gamma_real': GAMMA_REAL_SHAPE}


def central_from_cumulants(k2, k3, k4, k5, k6):
    """Central moments nu_2..nu_6 of a zero-mean variable from its cumulants."""
    return (k2,
            k3,
            k4 + 3 * k2 ** 2,
            k5 + 10 * k3 * k2,
            k6 + 15 * k4 * k2 + 10 * k3 ** 2 + 15 * k2 ** 3)


# --------------------------------------------------------------------------
# regression error eps
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EpsSpec:
    """Distribution of the regression error, standardised to m2 = 1.

    'gamma'     -- centred gamma, c4 = 1.5*c3^2 is then forced by the family.
    'gamma_mix' -- w*(centred standardised gamma) + sqrt(1-w^2)*N(0,1), which
                   frees c4 from c3 (c4 = 1.5*c3^2/w^2 >= 1.5*c3^2) and lets a
                   cell reproduce a MEASURED (c3, c4) pair.  Used only by the
                   NHANES-calibrated cell (c3 = 1.284, c4 = 3.42 => g2 = 0.696).
    """
    key: str
    family: str          # 'gauss' | 'gamma' | 'gamma_mix'
    c3: float            # target skewness
    c4: float = float('nan')   # target excess kurtosis ('gamma_mix' only)

    @property
    def shape(self) -> float:
        """Gamma shape k giving the required skewness of the gamma component."""
        if self.family == 'gamma':
            return 4.0 / self.c3 ** 2
        if self.family == 'gamma_mix':
            return 4.0 / (self.c3 / self.mix_w ** 3) ** 2
        return float('nan')

    @property
    def mix_w(self) -> float:
        """Weight of the gamma component: c4 = 1.5*c3^2/w^2 (needs c4 >= 1.5c3^2)."""
        if self.family != 'gamma_mix':
            return float('nan')
        return float(np.sqrt(1.5 * self.c3 ** 2 / self.c4))

    @property
    def moments(self) -> dict:
        """Exact central moments (m2, m3, m4) and coefficients (c3, c4, g2)."""
        if self.family == 'gauss':
            m2, m3, m4 = 1.0, 0.0, 3.0
            c3, c4 = 0.0, 0.0
        elif self.family == 'gamma':
            # eps = (Gamma(k,1) - k)/sqrt(k):  m2 = 1, m3 = 2/sqrt(k), m4 = 3 + 6/k
            k = self.shape
            m2 = 1.0
            m3 = 2.0 / np.sqrt(k)
            m4 = 3.0 + 6.0 / k
            c3 = m3
            c4 = m4 - 3.0
        else:                                   # 'gamma_mix'
            m2, c3, c4 = 1.0, self.c3, self.c4
            m3, m4 = c3, c4 + 3.0
        g2 = 1.0 - c3 ** 2 / (2.0 + c4)
        return dict(m2=m2, m3=m3, m4=m4, c3=c3, c4=c4, g2=g2)

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.family == 'gauss':
            return rng.standard_normal(n)
        k = self.shape
        g = (rng.gamma(k, 1.0, n) - k) / np.sqrt(k)      # m2 = 1, skew = 2/sqrt(k)
        if self.family == 'gamma':
            return g
        w = self.mix_w
        return w * g + np.sqrt(1.0 - w ** 2) * rng.standard_normal(n)


EPS_SPECS = {
    'gauss':  EpsSpec('gauss',  'gauss', 0.0),
    'g_c3_0.5': EpsSpec('g_c3_0.5', 'gamma', 0.5),
    'g_c3_1.0': EpsSpec('g_c3_1.0', 'gamma', 1.0),
    'g_c3_1.5': EpsSpec('g_c3_1.5', 'gamma', 1.5),
    # NHANES 2013-14 DBP application regime (2026-08-15): both (c3, c4) matched
    'real_dbp': EpsSpec('real_dbp', 'gamma_mix', REALDATA_C3_EPS, REALDATA_C4_EPS),
}


# --------------------------------------------------------------------------
# measurement error U
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class USpec:
    """Distribution of the measurement error, scaled to nu2 = (1-lam)/lam.

    Three designs:
      'gamma_wh'   -- W&H 6.1: shape 4(1-lam)/lam, scale 0.5.  Its standardised
                      shape is TIED to lam (skew 2.0 / 3.0 / 4.36 at lam =
                      .80/.90/.95), which is exactly the confound the theory
                      stream identified; kept as the adversarial cell.
      'gamma_fixed'-- shape fixed at GAMMA_FIXED_SHAPE, scale set to hit nu2.
                      Standardised shape constant in lam -> separates the
                      method's behaviour from the W&H design artefact.
      'gamma_real' -- fixed shape 28.011, i.e. |gamma3(U)| = 0.3779, the NHANES
                      2013-14 diastolic-BP value.
      'lognormal'  -- heavy tail, W&H 6.3 (tessarine system (13) unsolvable).

    Any family may be suffixed '_neg' (2026-08-15): the centred draw is
    negated, so nu3 and nu5 flip sign while nu2, nu4, nu6 are untouched.  This
    is the LEFT-SKEWED mirror needed by the real-data regime, and by
    construction it isolates the sign of the odd moments as the only change.
    """
    key: str
    family: str          # 'gamma_wh' | 'gamma_fixed' | 'gamma_real' | 'lognormal' (+ '_neg')
    lam: float

    @property
    def mirrored(self) -> bool:
        return self.family.endswith(NEG_SUFFIX)

    @property
    def base_family(self) -> str:
        return self.family[:-len(NEG_SUFFIX)] if self.mirrored else self.family

    @property
    def nu2_target(self) -> float:
        return (1.0 - self.lam) / self.lam * VAR_X

    @property
    def params(self) -> dict:
        fam = self.base_family
        if fam == 'gamma_wh':
            # nu2 = nu3 = (1-lam)/lam
            return dict(shape=4.0 * (1.0 - self.lam) / self.lam, scale=0.5)
        if fam in FIXED_SHAPES:
            k = FIXED_SHAPES[fam]
            return dict(shape=k, scale=np.sqrt(self.nu2_target / k))
        s = LOGNORMAL_SIGMA
        raw_var = (np.exp(s ** 2) - 1.0) * np.exp(s ** 2)
        return dict(sigma=s, scale=np.sqrt(self.nu2_target / raw_var))

    @property
    def moments(self) -> dict:
        """Exact central moments nu2..nu6 plus the standardised shape."""
        fam = self.base_family
        if fam == 'gamma_wh' or fam in FIXED_SHAPES:
            k, th = self.params['shape'], self.params['scale']
            # gamma cumulants: kappa_j = (j-1)! * k * theta^j
            nu2, nu3, nu4, nu5, nu6 = central_from_cumulants(
                k * th ** 2, 2 * k * th ** 3, 6 * k * th ** 4,
                24 * k * th ** 5, 120 * k * th ** 6)
        else:
            s, sc = LOGNORMAL_SIGMA, self.params['scale']
            ez = np.exp(s ** 2 / 2.0)
            raw = [np.exp(r ** 2 * s ** 2 / 2.0) for r in range(7)]   # E[Z^r]
            from math import comb
            cen = [sum(comb(r, j) * raw[j] * (-ez) ** (r - j) for j in range(r + 1))
                   for r in range(7)]
            nu2, nu3, nu4, nu5, nu6 = (sc ** r * cen[r] for r in range(2, 7))
        if self.mirrored:                      # U -> -U: odd central moments flip
            nu3, nu5 = -nu3, -nu5
        return dict(nu2=nu2, nu3=nu3, nu4=nu4, nu5=nu5, nu6=nu6,
                    skew=nu3 / nu2 ** 1.5 if nu2 > 0 else 0.0,
                    exkurt=nu4 / nu2 ** 2 - 3.0 if nu2 > 0 else 0.0)

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        p = self.params
        fam = self.base_family
        if fam == 'gamma_wh' or fam in FIXED_SHAPES:
            g = rng.gamma(p['shape'], p['scale'], n)
            out = g - p['shape'] * p['scale']
        else:
            z = rng.lognormal(0.0, p['sigma'], n)
            out = (z - np.exp(p['sigma'] ** 2 / 2.0)) * p['scale']
        return -out if self.mirrored else out


def u_spec(family: str, lam: float) -> USpec:
    return USpec(f'{family}_lam{lam:g}', family, lam)


# --------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    cell_id: str
    eps_key: str
    u_family: str
    lam: float
    n: int
    b0: float = 1.0
    b1: float = 1.0
    delta: float = 0.0      # U-moment misspecification amplitude (sensitivity)

    @property
    def eps(self) -> EpsSpec:
        return EPS_SPECS[self.eps_key]

    @property
    def u(self) -> USpec:
        return u_spec(self.u_family, self.lam)

    def describe(self) -> dict:
        d = asdict(self)
        d.update({f'eps_{k}': v for k, v in self.eps.moments.items()})
        d.update({f'u_{k}': v for k, v in self.u.moments.items()})
        return d


def make_cell(eps_key, u_family, lam, n, b0=1.0, b1=1.0, delta=0.0) -> Cell:
    cid = f'{eps_key}|{u_family}|lam{lam:g}|n{n}|b1{b1:g}'
    if delta > 0:
        cid += f'|d{delta:g}'
    return Cell(cid, eps_key, u_family, lam, n, b0, b1, delta)


# --------------------------------------------------------------------------
# seeding: deterministic, CRN across estimators (same data) and across cells
# --------------------------------------------------------------------------
BASE_ENTROPY = 20260814


def _key(s: str) -> int:
    return int(zlib.crc32(s.encode()))


def draw_replication(cell: Cell, rep: int) -> dict:
    """One replication.  X and eps depend only on (eps_key, n, rep) so that
    cells sharing those share the same latent data (variance-reduced grids);
    U depends additionally on (u_family, lam)."""
    rng_core = np.random.default_rng(
        np.random.SeedSequence([BASE_ENTROPY, _key('core'), _key(cell.eps_key), cell.n, rep]))
    rng_u = np.random.default_rng(
        np.random.SeedSequence([BASE_ENTROPY, _key('u'), _key(cell.u_family),
                                int(round(cell.lam * 1000)), cell.n, rep]))
    x = MEAN_X + np.sqrt(VAR_X) * rng_core.standard_normal(cell.n)
    eps = cell.eps.draw(cell.n, rng_core)
    u = cell.u.draw(cell.n, rng_u)
    y = cell.b0 + cell.b1 * x + eps
    return dict(x=x, y=y, w=x + u, u=u, eps=eps)


NU_KEYS = ('nu2', 'nu3', 'nu4', 'nu5', 'nu6')


def assumed_u_moments(cell: Cell, rep: int) -> dict:
    """U-moments handed to the estimators.  delta = 0 -> true moments (oracle);
    delta > 0 -> W&H 6.4 misspecification: independent Unif[1-d, 1+d] multipliers
    on every moment the estimators use (nu2..nu6; TC-PMM2* needs nu5, nu6)."""
    tm = cell.u.moments
    nu = {k: tm[k] for k in NU_KEYS}
    if cell.delta > 0:
        rng = np.random.default_rng(np.random.SeedSequence(
            [BASE_ENTROPY, _key('miss'), int(round(cell.delta * 1000)), cell.n, rep]))
        f = rng.uniform(1.0 - cell.delta, 1.0 + cell.delta, len(NU_KEYS))
        nu = {k: nu[k] * f[i] for i, k in enumerate(NU_KEYS)}
    return nu


if __name__ == '__main__':   # quick self-report of the theoretical moments
    print(f'{"eps":10s} {"c3":>7s} {"c4":>7s} {"m3":>7s} {"m4":>7s} {"g2":>7s}')
    for k, s in EPS_SPECS.items():
        m = s.moments
        print(f'{k:10s} {m["c3"]:7.4f} {m["c4"]:7.4f} {m["m3"]:7.4f} {m["m4"]:7.4f} {m["g2"]:7.4f}')
    print()
    print(f'{"U":22s} {"nu2":>9s} {"nu3":>9s} {"nu4":>9s} {"nu5":>10s} {"nu6":>11s} '
          f'{"skew":>8s} {"exkurt":>9s}')
    for fam in ('gamma_wh', 'gamma_wh_neg', 'gamma_fixed', 'gamma_fixed_neg',
                'gamma_real', 'gamma_real_neg', 'lognormal'):
        for lam in (0.80, 0.90, 0.95, REALDATA_LAM):
            s = u_spec(fam, lam)
            m = s.moments
            print(f'{s.key:22s} {m["nu2"]:9.5f} {m["nu3"]:9.5f} {m["nu4"]:9.5f} '
                  f'{m["nu5"]:10.5f} {m["nu6"]:11.5f} {m["skew"]:8.3f} {m["exkurt"]:9.3f}')
