"""Vectorized Black-Scholes prices and greeks on dividend-escrowed spot S* (PLAN §5.1).

All functions take (S, K, T, r, sigma) where S is the dividend-escrowed spot
(S* = S - PV of dividends ex-dating before expiry; see pricing.dividends),
T is in years, r and sigma are decimals. Dividends are handled outside via S*,
so there is no continuous-yield argument here. T->0 and sigma->0 degrade to
(discounted) intrinsic value without NaNs.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_EPS = 1e-12


def _npdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _bcast(*xs) -> list[np.ndarray]:
    return list(np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in xs]))


def _d1d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def _live_inputs(S, K, T, r, sigma):
    """Inputs made safe for d1/d2 where the live (T>0, sigma>0) mask is False."""
    live = (T > _EPS) & (sigma > _EPS) & (S > 0.0) & (K > 0.0)
    Ts = np.where(live, T, 1.0)
    ss = np.where(live, sigma, 0.2)
    Ss = np.where(live, S, 1.0)
    Ks = np.where(live, K, 1.0)
    return live, Ss, Ks, Ts, ss


def price(S, K, T, r, sigma, cp: str = "C"):
    """BS call/put price. ``cp`` is "C" or "P" (scalar); numeric args broadcast."""
    S, K, T, r, sigma = _bcast(S, K, T, r, sigma)
    T = np.maximum(T, 0.0)
    disc = np.exp(-r * T)
    live, Ss, Ks, Ts, ss = _live_inputs(S, K, T, r, sigma)
    d1, d2 = _d1d2(Ss, Ks, Ts, r, ss)
    call_bs = np.clip(Ss * ndtr(d1) - Ks * disc * ndtr(d2), 0.0, Ss)
    put_bs = np.clip(Ks * disc * ndtr(-d2) - Ss * ndtr(-d1), 0.0, Ks * disc)
    call_deg = np.where(T <= _EPS, np.maximum(S - K, 0.0), np.maximum(S - K * disc, 0.0))
    put_deg = np.where(T <= _EPS, np.maximum(K - S, 0.0), np.maximum(K * disc - S, 0.0))
    call = np.where(live, call_bs, call_deg)
    put = np.where(live, put_bs, put_deg)
    return call if cp.upper() == "C" else put


def call_price(S, K, T, r, sigma):
    """BS call price (see :func:`price`)."""
    return price(S, K, T, r, sigma, "C")


def put_price(S, K, T, r, sigma):
    """BS put price (see :func:`price`)."""
    return price(S, K, T, r, sigma, "P")


def delta(S, K, T, r, sigma, cp: str = "C"):
    """BS delta; degenerate inputs give the intrinsic subgradient (0.5 at the money)."""
    S, K, T, r, sigma = _bcast(S, K, T, r, sigma)
    live, Ss, Ks, Ts, ss = _live_inputs(S, K, T, r, sigma)
    d1, _ = _d1d2(Ss, Ks, Ts, r, ss)
    call_live = ndtr(d1)
    itm = (S > K).astype(float) + 0.5 * (S == K).astype(float)
    call = np.where(live, call_live, itm)
    return call if cp.upper() == "C" else call - 1.0


def gamma(S, K, T, r, sigma):
    """BS gamma (same for calls and puts); 0 for degenerate inputs."""
    S, K, T, r, sigma = _bcast(S, K, T, r, sigma)
    live, Ss, Ks, Ts, ss = _live_inputs(S, K, T, r, sigma)
    d1, _ = _d1d2(Ss, Ks, Ts, r, ss)
    g = _npdf(d1) / (Ss * ss * np.sqrt(Ts))
    return np.where(live, g, 0.0)


def vega(S, K, T, r, sigma):
    """BS vega per unit (1.00) of vol; 0 for degenerate inputs."""
    S, K, T, r, sigma = _bcast(S, K, T, r, sigma)
    live, Ss, Ks, Ts, ss = _live_inputs(S, K, T, r, sigma)
    d1, _ = _d1d2(Ss, Ks, Ts, r, ss)
    v = Ss * _npdf(d1) * np.sqrt(Ts)
    return np.where(live, v, 0.0)


def theta(S, K, T, r, sigma, cp: str = "C"):
    """BS theta per year; 0 for degenerate inputs."""
    S, K, T, r, sigma = _bcast(S, K, T, r, sigma)
    disc = np.exp(-r * np.maximum(T, 0.0))
    live, Ss, Ks, Ts, ss = _live_inputs(S, K, T, r, sigma)
    d1, d2 = _d1d2(Ss, Ks, Ts, r, ss)
    common = -(Ss * _npdf(d1) * ss) / (2.0 * np.sqrt(Ts))
    th_call = common - r * Ks * disc * ndtr(d2)
    th_put = common + r * Ks * disc * ndtr(-d2)
    th = th_call if cp.upper() == "C" else th_put
    return np.where(live, th, 0.0)
