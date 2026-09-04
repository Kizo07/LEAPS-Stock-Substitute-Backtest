"""Vectorised building blocks for the Monte Carlo study.

Everything is numpy-array-in / numpy-array-out so the simulation loop can price
thousands of paths x dozens of legs in one call.

Contents
--------
* `bs_call` / `bs_put` / `bs_delta_call` / `bs_delta_put` / `bs_vega` -- closed
  form Black-Scholes-Merton with a continuous dividend yield.
* `crr_american_vec` -- Cox-Ross-Rubinstein tree run over *arrays* of contracts
  at once, returning both the American and the European value on the same grid
  (so the early-exercise premium is grid-bias free).
* `EEGrid` -- precomputes the early-exercise premium on a (moneyness, tau) grid
  once and interpolates it during the simulation.  Prices are homogeneous of
  degree 1 in (S, K) under a continuous dividend yield, so a single grid with
  S = 1 covers every contract.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

# --------------------------------------------------------------------- BSM
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _d1(S, K, T, r, q, sig):
    return (np.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * np.sqrt(T))


def bs_call(S, K, T, r, q, sig):
    d1 = _d1(S, K, T, r, q, sig)
    d2 = d1 - sig * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, q, sig):
    d1 = _d1(S, K, T, r, q, sig)
    d2 = d1 - sig * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_delta_call(S, K, T, r, q, sig):
    return np.exp(-q * T) * norm.cdf(_d1(S, K, T, r, q, sig))


def bs_delta_put(S, K, T, r, q, sig):
    return -np.exp(-q * T) * norm.cdf(-_d1(S, K, T, r, q, sig))


def bs_vega(S, K, T, r, q, sig):
    """Per 1.00 = 100 vol points."""
    return S * np.exp(-q * T) * np.exp(-0.5 * _d1(S, K, T, r, q, sig) ** 2) / _SQRT2PI * np.sqrt(T)


def bs_theta_call(S, K, T, r, q, sig):
    """dV/dt (calendar time), per year."""
    d1 = _d1(S, K, T, r, q, sig)
    d2 = d1 - sig * np.sqrt(T)
    return (-S * np.exp(-q * T) * np.exp(-0.5 * d1 ** 2) / _SQRT2PI * sig / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1))


def bs_theta_put(S, K, T, r, q, sig):
    d1 = _d1(S, K, T, r, q, sig)
    d2 = d1 - sig * np.sqrt(T)
    return (-S * np.exp(-q * T) * np.exp(-0.5 * d1 ** 2) / _SQRT2PI * sig / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1))


def strike_for_delta(S, T, r, q, sig, target, kind="C", iters=80):
    """Vectorised bisection for the strike giving |delta| = target."""
    S = np.asarray(S, float)
    lo = np.full(np.shape(S), 1e-6)
    hi = np.asarray(S, float) * 20.0
    f = (lambda K: bs_delta_call(S, K, T, r, q, sig) - target) if kind == "C" \
        else (lambda K: -bs_delta_put(S, K, T, r, q, sig) - target)
    flo = f(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if np.all(np.abs(hi - lo) < 1e-8):
            break
        neg = flo * f(mid) <= 0
        hi = np.where(neg, mid, hi)
        lo = np.where(neg, lo, mid)
        flo = np.where(neg, flo, f(lo))
    return 0.5 * (lo + hi)


# ---------------------------------------------------- vectorised CRR tree
def crr_american_vec(S, K, T, r, q, sig, n=200, kind="C"):
    """American and European value on the same tree, vectorised over contracts.

    S, K, T may be arrays (broadcast together).  Returns (american, european).
    """
    S = np.asarray(S, float)
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    S, K, T = np.broadcast_arrays(S, K, T)
    T = np.maximum(T, 1e-6)
    shape = S.shape
    Sf, Kf, Tf = S.ravel(), K.ravel(), T.ravel()
    m = Sf.size

    h = Tf / n
    u = np.exp(sig * np.sqrt(h))
    d = 1.0 / u
    disc = np.exp(-r * h)
    p = (np.exp((r - q) * h) - d) / (u - d)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    is_call = kind.upper().startswith("C")

    # terminal stock prices: (n+1, m)
    j = np.arange(n + 1)[:, None]
    ST = Sf[None, :] * (u ** j) * (d ** (n - j))
    V = np.maximum(ST - Kf[None, :], 0.0) if is_call else np.maximum(Kf[None, :] - ST, 0.0)

    for i in range(n - 1, -1, -1):
        V = disc[None, :] * (p[None, :] * V[1:] + (1 - p[None, :]) * V[:-1])
        if i > 0:
            jj = np.arange(i + 1)[:, None]
            Si = Sf[None, :] * (u ** jj) * (d ** (i - jj))
            ex = np.maximum(Si - Kf[None, :], 0.0) if is_call else np.maximum(Kf[None, :] - Si, 0.0)
            V = np.maximum(V, ex)

    american = V[0].reshape(shape)

    j = np.arange(n + 1)[:, None]
    ST = Sf[None, :] * (u ** j) * (d ** (n - j))
    Ve = np.maximum(ST - Kf[None, :], 0.0) if is_call else np.maximum(Kf[None, :] - ST, 0.0)
    for _ in range(n):
        Ve = disc[None, :] * (p[None, :] * Ve[1:] + (1 - p[None, :]) * Ve[:-1])
    european = Ve[0].reshape(shape)
    return american, european


# ------------------------------------------------- early-exercise premium
class EEGrid:
    """Interpolated early-exercise premium, in units of spot.

    eep(S, K, tau) = S * eep(1, K/S, tau)   [homogeneity under a yield dividend]
    """

    def __init__(self, r, q, sig, k_lo=-2.0, k_hi=2.0, nk=81,
                 t_lo=0.02, t_hi=2.6, nt=52, nsteps=140):
        self.r, self.q, self.sig = r, q, sig
        # both axes must be UNIFORM in the transformed variable, otherwise the
        # index arithmetic in _bilinear is wrong (an earlier version took the log
        # after the linspace, which silently collapsed the whole tau axis)
        self.lk = np.linspace(k_lo, k_hi, nk)                   # ln(K/S)
        self.lt = np.linspace(math.log(t_lo), math.log(t_hi), nt)  # ln(tau)
        LK, LT = np.meshgrid(self.lk, self.lt, indexing="ij")
        K = np.exp(LK).ravel()
        T = np.exp(LT).ravel()
        self.ec = np.zeros(LK.shape)
        self.ep = np.zeros(LK.shape)
        chunk = 4000
        for s in range(0, K.size, chunk):
            ks, ts = K[s:s + chunk], T[s:s + chunk]
            ac, ec = crr_american_vec(1.0, ks, ts, r, q, sig, nsteps, "C")
            ap, ep = crr_american_vec(1.0, ks, ts, r, q, sig, nsteps, "P")
            ix, iy = np.unravel_index(np.arange(s, min(s + chunk, K.size)), LK.shape)
            self.ec[ix, iy] = ac - ec
            self.ep[ix, iy] = ap - ep
        self.dlk = self.lk[1] - self.lk[0]
        self.dlt = self.lt[1] - self.lt[0]

    def _bilinear(self, grid, lk, lt):
        x = np.clip((lk - self.lk[0]) / self.dlk, 0, len(self.lk) - 1.001)
        y = np.clip((lt - self.lt[0]) / self.dlt, 0, len(self.lt) - 1.001)
        i, j = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - i, y - j
        g = grid
        return (g[i, j] * (1 - fx) * (1 - fy) + g[i + 1, j] * fx * (1 - fy)
                + g[i, j + 1] * (1 - fx) * fy + g[i + 1, j + 1] * fx * fy)

    def call(self, S, K, tau):
        tau = np.maximum(np.asarray(tau, float), 1e-3)
        return np.asarray(S, float) * self._bilinear(self.ec, np.log(np.asarray(K, float) / np.asarray(S, float)),
                                                     np.log(tau))

    def put(self, S, K, tau):
        tau = np.maximum(np.asarray(tau, float), 1e-3)
        return np.asarray(S, float) * self._bilinear(self.ep, np.log(np.asarray(K, float) / np.asarray(S, float)),
                                                     np.log(tau))
