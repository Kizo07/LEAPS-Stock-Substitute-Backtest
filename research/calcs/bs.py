"""Self-contained Black-Scholes-Merton analytics used by the research notes.

Written independently of leaps_ls.pricing so the research results are an
independent check on the project engine rather than a restatement of it.

Conventions
-----------
* continuous compounding everywhere
* spot S, strike K, tenor T (years), risk-free r (cont.), dividend yield q (cont.)
* all prices per share, per 1 unit of underlying
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

__all__ = [
    "N", "npdf", "d1_d2", "call", "put", "delta_call", "delta_put",
    "gamma", "vega", "theta_call", "theta_put", "rho_call",
    "bs_price", "imp_vol",
]


def N(x):
    return norm.cdf(x)


def npdf(x):
    return norm.pdf(x)


def d1_d2(S, K, T, r, q, sig):
    if T <= 0:
        raise ValueError("T must be > 0")
    if sig <= 0:
        raise ValueError("sig must be > 0")
    d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return d1, d2


def call(S, K, T, r, q, sig):
    d1, d2 = d1_d2(S, K, T, r, q, sig)
    return S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)


def put(S, K, T, r, q, sig):
    d1, d2 = d1_d2(S, K, T, r, q, sig)
    return K * math.exp(-r * T) * N(-d2) - S * math.exp(-q * T) * N(-d1)


def bs_price(S, K, T, r, q, sig, kind="C"):
    return call(S, K, T, r, q, sig) if kind.upper().startswith("C") else put(S, K, T, r, q, sig)


def delta_call(S, K, T, r, q, sig):
    d1, _ = d1_d2(S, K, T, r, q, sig)
    return math.exp(-q * T) * N(d1)


def delta_put(S, K, T, r, q, sig):
    d1, _ = d1_d2(S, K, T, r, q, sig)
    return -math.exp(-q * T) * N(-d1)


def gamma(S, K, T, r, q, sig):
    d1, _ = d1_d2(S, K, T, r, q, sig)
    return math.exp(-q * T) * npdf(d1) / (S * sig * math.sqrt(T))


def vega(S, K, T, r, q, sig):
    """Per 1.00 = 100 vol points."""
    d1, _ = d1_d2(S, K, T, r, q, sig)
    return S * math.exp(-q * T) * npdf(d1) * math.sqrt(T)


def theta_call(S, K, T, r, q, sig):
    """Per year (continuous)."""
    d1, d2 = d1_d2(S, K, T, r, q, sig)
    return (
        -S * math.exp(-q * T) * npdf(d1) * sig / (2 * math.sqrt(T))
        - r * K * math.exp(-r * T) * N(d2)
        + q * S * math.exp(-q * T) * N(d1)
    )


def theta_put(S, K, T, r, q, sig):
    d1, d2 = d1_d2(S, K, T, r, q, sig)
    return (
        -S * math.exp(-q * T) * npdf(d1) * sig / (2 * math.sqrt(T))
        + r * K * math.exp(-r * T) * N(-d2)
        - q * S * math.exp(-q * T) * N(-d1)
    )


def rho_call(S, K, T, r, q, sig):
    """Per 1.00 = 100bp."""
    _, d2 = d1_d2(S, K, T, r, q, sig)
    return K * T * math.exp(-r * T) * N(d2) / 100.0


def imp_vol(price, S, K, T, r, q, kind="C", lo=1e-4, hi=5.0, tol=1e-10, iters=200):
    """Brent-free bisection on total (undiscounted) vega-monotone price."""
    f = lambda s: bs_price(S, K, T, r, q, s, kind) - price  # noqa: E731
    # intrinsic floors/ceilings
    if kind.upper().startswith("C"):
        lo_i, hi_i = max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T)), S * math.exp(-q * T)
    else:
        lo_i, hi_i = max(0.0, K * math.exp(-r * T) - S * math.exp(-q * T)), K * math.exp(-r * T)
    if price <= lo_i + 1e-12:
        return lo
    if price >= hi_i - 1e-12:
        return hi
    a, b = lo, hi
    fa = f(a)
    for _ in range(iters):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol:
            return m
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)
