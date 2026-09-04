"""Cox-Ross-Rubinstein binomial pricer with known dollar dividends.

Escrowed-spot construction: the tree is built on the prepaid forward
S* = S - PV(all dividends ex-dated before T).  At node time t the actual
stock price is S*_node + PV(dividends ex-dating after t), discounted from
their ex-date back to t at rate r.  Early exercise is tested against that
actual price.  This is the standard treatment for known dollar dividends.

Also returns the European value on the same tree, so the early-exercise
premium is measured on an identical discretisation (no grid bias).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["crr_american", "ee_premium"]


def _div_pv_to(sched, t: float, r: float) -> float:
    """PV at time t of dividends with ex-date strictly after t."""
    return sum(d * math.exp(-r * (td - t)) for td, d in sched if td > t + 1e-12)


def crr_american(S, K, T, r, sig, sched, n: int = 400, kind: str = "C"):
    """(american, european, exercise_nodes) via CRR.

    sched: list of (t_exdate, dollar_amount), all with 0 < t < T.
    """
    sched = sorted(sched)
    pv_all = sum(d * math.exp(-r * t) for t, d in sched)
    Sstar = S - pv_all
    if Sstar <= 0:
        Sstar = 1e-6

    h = T / n
    u = math.exp(sig * math.sqrt(h))
    d = 1.0 / u
    disc = math.exp(-r * h)
    p = (math.exp(r * h) - d) / (u - d)
    p = min(max(p, 1e-9), 1 - 1e-9)

    is_call = kind.upper().startswith("C")

    # terminal actual stock prices
    j = np.arange(n + 1)
    Sstar_T = Sstar * (u ** j) * (d ** (n - j))
    S_T = Sstar_T + _div_pv_to(sched, T, r)          # 0 after last ex-date
    V = np.maximum(S_T - K, 0.0) if is_call else np.maximum(K - S_T, 0.0)

    for i in range(n - 1, -1, -1):
        t_i = i * h
        V = disc * (p * V[1:] + (1 - p) * V[:-1])
        if i > 0:
            Sstar_i = Sstar * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
            S_i = Sstar_i + _div_pv_to(sched, t_i, r)
            exer = np.maximum(S_i - K, 0.0) if is_call else np.maximum(K - S_i, 0.0)
            V = np.maximum(V, exer)

    american = float(V[0])

    # European on the same grid: no exercise test, terminal only
    j = np.arange(n + 1)
    Sstar_T = Sstar * (u ** j) * (d ** (n - j))
    S_T = Sstar_T + 0.0
    Ve = np.maximum(S_T - K, 0.0) if is_call else np.maximum(K - S_T, 0.0)
    for _ in range(n):
        Ve = disc * (p * Ve[1:] + (1 - p) * Ve[:-1])
    european = float(Ve[0])
    return american, european, american - european


def ee_premium(S, K, T, r, sig, sched, n: int = 400, kind: str = "C"):
    """Early-exercise premium in dollars (american - european)."""
    a, e, d = crr_american(S, K, T, r, sig, sched, n, kind)
    return d
