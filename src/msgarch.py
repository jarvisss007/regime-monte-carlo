"""
msgarch.py -- Markov-switching GARCH(1,1), the Haas-Mittnik-Paolella (2004)
specification, implemented from scratch. K=2 regimes.

This is the hybrid the HMM paper's discussion points to: persistent hidden
states (like the HMM) where each state carries its OWN GARCH variance
recursion (like GARCH), rather than a constant within-state variance. It
targets both failure modes found in the paper at once:
  * regime structure   -> a single-regime GARCH cannot represent it;
  * adaptive variance   -> the HMM's constant state ceilings broke in 2008.

Haas et al. avoid the path-dependence problem (an exact MS-GARCH likelihood
sums over K^T variance paths) by letting all K variance processes run in
parallel off the common observed return, so each regime's variance is a
deterministic function of the data:

    h_{k,t} = omega_k + alpha_k * r_{t-1}^2 + beta_k * h_{k,t-1}      (all k)

The latent chain (transition matrix P) only selects which regime's density
generates r_t. Innovations are Normal or standardised Student-t(nu).
Estimation is MLE via the Hamilton filter; the K=2 filter is scalar-unrolled
so a full walk-forward is fast.

Convention (matches src/hmm_t.py and src/evaluate2.py): a component with
conditional variance h and Student-t(nu) innovations is represented as a
RAW t with scale s = sqrt(h*(nu-2)/nu) = _t_scale(h, nu), so its variance is
exactly h. This lets the mixture VaR/ES helpers in evaluate2 be reused.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats

SC = 100.0          # fit in percent-return units for numerical health
_A_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# parameter (un)packing
# ---------------------------------------------------------------------------
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _unpack(theta, student):
    """theta -> dict with per-regime (omega,alpha,beta) in PERCENT units,
    2x2 transition matrix P, and nu (inf for normal)."""
    w1, a1, b1, w2, a2, b2, p00, p11 = theta[:8]
    om = np.array([np.exp(w1), np.exp(w2)])
    al = np.array([_sig(a1), _sig(a2)])
    be = (1 - al) * np.array([_sig(b1), _sig(b2)])   # ensures alpha+beta<1
    P = np.array([[_sig(p00), 1 - _sig(p00)],
                  [1 - _sig(p11), _sig(p11)]])
    nu = 2.0 + np.exp(theta[8]) if student else np.inf
    return om, al, be, P, nu


def _stationary(P):
    p00, p11 = P[0, 0], P[1, 1]
    d = 2 - p00 - p11
    return np.array([(1 - p11) / d, (1 - p00) / d]) if d > 1e-9 else np.array([.5, .5])


# ---------------------------------------------------------------------------
# fast scalar K=2 negative log-likelihood (returns in percent)
# ---------------------------------------------------------------------------
def _nll(theta, rp, student):
    om, al, be, P, nu = _unpack(theta, student)
    o1, o2 = om; a1, a2 = al; b1, b2 = be
    P00, P01 = P[0, 0], P[0, 1]
    P10, P11 = P[1, 0], P[1, 1]
    pi = _stationary(P)
    v0 = rp.var()
    h1 = h2 = v0
    p1, p2 = pi[0], pi[1]

    if student:
        c = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
             - 0.5 * np.log(np.pi * nu))
        sc2 = (nu - 2) / nu                      # standardise to unit variance

        def dens(r2, h):
            s2 = h * sc2
            return np.exp(c - 0.5 * np.log(s2) - 0.5 * (nu + 1) * np.log1p(r2 / (s2 * nu)))
    else:
        def dens(r2, h):
            return np.exp(-0.5 * (np.log(2 * np.pi * h) + r2 / h))

    ll = 0.0
    for r in rp:
        r2 = r * r
        # predictive state probs (through t-1)
        q1 = p1 * P00 + p2 * P10
        q2 = p1 * P01 + p2 * P11
        e1 = dens(r2, h1); e2 = dens(r2, h2)
        m = q1 * e1 + q2 * e2
        if m <= 1e-300 or not np.isfinite(m):
            return 1e10
        ll += np.log(m)
        # filter update
        p1 = q1 * e1 / m; p2 = q2 * e2 / m
        # advance variances
        h1 = o1 + a1 * r2 + b1 * h1
        h2 = o2 + a2 * r2 + b2 * h2
    return -ll


def fit(r, student=True, warm=None, maxiter=1500):
    """MLE of the K=2 MS-GARCH. r in raw return units. Returns dict of
    params in RAW-return units, ordered so regime 0 is the calmer one."""
    rp = np.asarray(r, float) * SC
    if warm is not None:
        th0 = np.array(warm, float)
    else:
        # init two distinct regimes from vol terciles of |r|
        q = np.quantile(np.abs(rp), [1 / 3, 2 / 3])
        lo = rp[np.abs(rp) <= q[0]]; hi = rp[np.abs(rp) >= q[1]]
        vlo = max(lo.var(), 1e-4); vhi = max(hi.var(), vlo * 3)
        # omega = uncond_var*(1-a-b), with a=.05,b=.9
        th0 = [np.log(vlo * 0.05), _logit(0.05), _logit(0.9 / 0.95),
               np.log(vhi * 0.05), _logit(0.08), _logit(0.88 / 0.92),
               _logit(0.97), _logit(0.95)]
        if student:
            th0.append(np.log(6.0))
    res = optimize.minimize(_nll, th0, args=(rp, student), method="Nelder-Mead",
                            options=dict(maxiter=maxiter, xatol=1e-5, fatol=1e-6))
    om, al, be, P, nu = _unpack(res.x, student)
    uncond = om / np.maximum(1 - al - be, _A_FLOOR)
    order = np.argsort(uncond)                 # calm regime first
    om, al, be = om[order], al[order], be[order]
    P = P[np.ix_(order, order)]
    # convert omega back to raw-return variance units (percent^2 -> raw^2)
    return dict(omega=om / SC**2, alpha=al, beta=be, P=P, nu=float(nu),
                loglik=float(-res.fun),
                theta=res.x.tolist(), order=order.tolist())


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


# ---------------------------------------------------------------------------
# online filter for out-of-sample prediction (no look-ahead)
# ---------------------------------------------------------------------------
class MSGarchFilter:
    """Maintains filtered state probs p (through t-1) and current per-regime
    variances h (h_{k,t}, applicable to the next return). All in RAW units.
    Mirrors the TFilter API so it slots into the evaluation harness."""

    def __init__(self, params, v0):
        self.om = np.asarray(params["omega"], float)
        self.al = np.asarray(params["alpha"], float)
        self.be = np.asarray(params["beta"], float)
        self.P = np.asarray(params["P"], float)
        self.nu = params["nu"]
        self.student = np.isfinite(self.nu)
        self.p = _stationary(self.P)
        self.h = np.full(2, float(v0))

    def _dens(self, r):
        if self.student:
            s = np.sqrt(self.h * (self.nu - 2) / self.nu)
            return stats.t.pdf(r / s, df=self.nu) / s
        s2 = self.h
        return np.exp(-0.5 * (np.log(2 * np.pi * s2) + r * r / s2))

    def predict_state_probs(self):
        return self.p @ self.P

    def predict_next_var(self):
        q = self.predict_state_probs()
        return float((q * self.h).sum())        # zero-mean mixture variance

    def _mix(self):
        """(q, sigma, nu) for the predictive mixture, sigma in the raw-t /
        normal scale convention used by evaluate2.mixture_*."""
        q = self.predict_state_probs()
        if self.student:
            sigma = np.sqrt(self.h * (self.nu - 2) / self.nu)
            return q, sigma, self.nu
        return q, np.sqrt(self.h), None

    def logscore(self, r):
        q = self.predict_state_probs()
        return float(np.log(max((q * self._dens(r)).sum(), 1e-300)))

    def pit(self, r):
        q = self.predict_state_probs()
        if self.student:
            s = np.sqrt(self.h * (self.nu - 2) / self.nu)
            return float((q * stats.t.cdf(r / s, df=self.nu)).sum())
        return float((q * stats.norm.cdf(r / np.sqrt(self.h))).sum())

    def update(self, r):
        q = self.predict_state_probs()
        e = self._dens(r)
        m = (q * e).sum()
        self.p = (q * e) / m if m > 0 else np.array([.5, .5])
        r2 = r * r
        self.h = self.om + self.al * r2 + self.be * self.h
        return self.p
