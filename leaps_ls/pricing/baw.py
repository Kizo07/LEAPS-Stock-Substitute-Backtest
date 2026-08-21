"""Barone-Adesi & Whaley (1987) American approximation — sensitivity use only (PLAN §5.1).

Works on the dividend-escrowed spot S*, whose drift carries no further payouts,
so cost of carry b = r. Consequences:
- American call on S* equals the European call (no dividends left to exercise for);
  call early exercise is handled by the ex-date rules in pricing.exercise instead.
- American put gets the standard BAW quadratic early-exercise premium.

Fallbacks: T<=0 -> intrinsic; sigma<=0 -> discounted intrinsic; r<=0 -> European
price (BAW's quadratic root degenerates at r=0; the put premium is small there).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from . import black_scholes as bs


def american_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """American call on payout-free S* == European call."""
    return float(bs.call_price(S, K, T, r, sigma))


def _critical_put_price(K: float, T: float, r: float, sigma: float, q2: float) -> float:
    """Solve the BAW value-matching condition for the put exercise boundary S**."""

    def f(Sc: float) -> float:
        d1 = (np.log(Sc / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
        p = float(bs.put_price(Sc, K, T, r, sigma))
        a2 = -(Sc / q2) * (1.0 - ndtr(-d1))  # exp((b-r)T) = 1 for b = r
        return K - Sc - p - a2

    return brentq(f, 1e-10, K, xtol=1e-12, rtol=1e-12)


def american_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BAW American put price on S* (b = r)."""
    S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
    if T <= 0.0:
        return max(K - S, 0.0)
    if sigma <= 0.0:
        return max(K * np.exp(-r * T) - S, 0.0)
    if r <= 0.0:
        return float(bs.put_price(S, K, T, r, sigma))  # documented fallback
    m = 2.0 * r / (sigma * sigma)  # N = M = 2r/sigma^2 when b = r
    k_t = 1.0 - np.exp(-r * T)
    q2 = (-(m - 1.0) - np.sqrt((m - 1.0) ** 2 + 4.0 * m / k_t)) / 2.0
    s_crit = _critical_put_price(K, T, r, sigma, q2)
    if S <= s_crit:
        return K - S  # immediate exercise
    d1 = (np.log(s_crit / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    a2 = -(s_crit / q2) * (1.0 - ndtr(-d1))
    return float(bs.put_price(S, K, T, r, sigma)) + a2 * (S / s_crit) ** q2


def american_price(S: float, K: float, T: float, r: float, sigma: float, cp: str = "C") -> float:
    """Dispatch on option type."""
    return american_call(S, K, T, r, sigma) if cp.upper() == "C" else american_put(S, K, T, r, sigma)
