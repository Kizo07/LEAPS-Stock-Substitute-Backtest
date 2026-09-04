"""Structural risks of a LEAPS-implemented long/short book.

All numbers here are friction-free: no bid-ask, no borrow, no financing
spread.  Whatever damage shows up is pure structure.

A. EXPOSURE DRIFT
   Long-call delta RISES with S; long-put |delta| FALLS with S.  A book that
   is delta-neutral when struck becomes net long after a rally and net short
   after a selloff.  Measured as a function of the market move AND of the
   remaining tenor (the drift is mild far from expiry and violent near it).

B. RUIN
   A stock position loses everything only if the company goes to zero.  A
   LEAPS call loses everything if S_T < K.  Measured under the risk-neutral
   measure and under the empirical distribution of the project's own universe.

C. CRASH / REBOUND ASYMMETRY
   Momentum crashes are rebounds out of a drawdown.  At the trough the LEAPS
   book is maximally short (longs have lost delta, shorts have gained it), so
   it is positioned worst exactly at the turning point.

D. BREAK-EVEN THRESHOLDS
   For a given set of frictions, at what borrow rate / dividend yield /
   volatility does the LEAPS implementation stop being worth it?

NOTE ON SCOPE
    Section C (the endpoint-only crash comparison) and section D (put-vs-cash-short)
    were removed.  Both priced the deep-ITM put European, which is wrong: a deep-ITM
    American put trades near intrinsic, and the European value understates it by 6.8
    points of spot at delta = -0.80 / T = 2y.  The European put also has POSITIVE theta
    (because K e^{-rT} accretes), which made a buy-and-hold LEAPS book look profitable
    in drawdowns when it is not.  The correct American treatment is in
    04_drift_and_capital.py, which supersedes both.

Outputs: research/results/risk_*.csv
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import bs

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

S0, SIG, R, Q = 100.0, 0.25, 0.042, 0.015
DELTA_TGT = 0.80
NSTEPS_MC = 200_000


def strike_for_delta(S, T, r, q, sig, target, kind="C"):
    """Strike giving |delta| = target."""
    lo, hi = (1e-4, 20.0 * S) if kind == "C" else (1e-4, 20.0 * S)
    f = (lambda K: bs.delta_call(S, K, T, r, q, sig) - target) if kind == "C" \
        else (lambda K: -bs.delta_put(S, K, T, r, q, sig) - target)
    a, b = lo, hi
    for _ in range(200):
        m = 0.5 * (a + b)
        if f(a) * f(m) <= 0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)


# ------------------------------------------------------------------ A
def exposure_drift():
    """Net delta exposure ($ per $1 of initial leg exposure) vs market move."""
    T0 = 2.0
    Kc = strike_for_delta(S0, T0, R, Q, SIG, DELTA_TGT, "C")
    Kp = strike_for_delta(S0, T0, R, Q, SIG, DELTA_TGT, "P")
    # number of contracts per $1 of initial delta exposure on each leg
    n = 1.0 / (DELTA_TGT * S0)

    rows = []
    for tau in (2.0, 1.5, 1.0, 0.5, 0.25, 0.10, 0.02):
        for x in (-0.60, -0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.50, 1.00):
            Sp = S0 * (1 + x)
            dc = bs.delta_call(Sp, Kc, min(tau, T0), R, Q, SIG)
            dp = bs.delta_put(Sp, Kp, min(tau, T0), R, Q, SIG)
            net = n * (dc + dp) * Sp                  # dp<0 ; net $ per $1 leg
            rows.append(dict(tau=tau, mkt_move=round(100 * x, 1), S=round(Sp, 1),
                             delta_call=round(dc, 3), delta_put=round(dp, 3),
                             net_exposure=round(net, 3),
                             gross_exposure=round(n * (dc - dp) * Sp, 3)))
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="tau", columns="mkt_move", values="net_exposure")
    return df, piv, Kc, Kp


# ------------------------------------------------------------------ B
def ruin_table():
    """P(S_T < K) and expected recovery for a delta-targeted LEAPS call."""
    rows = []
    for T in (0.5, 1.0, 2.0):
        for tgt in (0.70, 0.80, 0.90, 0.95):
            K = strike_for_delta(S0, T, R, Q, SIG, tgt, "C")
            C = bs.call(S0, K, T, R, Q, SIG)
            rn_prob = bs.N(-(math.log(S0 / K) + (R - Q - 0.5 * SIG ** 2) * T)
                           / (SIG * math.sqrt(T)))
            # expected payoff / expected forward, and breakeven at expiry
            fwd = S0 * math.exp((R - Q) * T)
            be = K + C * math.exp(R * T)
            rows.append(dict(T=T, delta_tgt=tgt, K=round(K, 2), prem=round(C, 2),
                             prem_pct_spot=round(100 * C / S0, 1),
                             leverage_x=round(S0 * tgt / C, 2),
                             P_expire_worthless=round(100 * rn_prob, 2),
                             breakeven_S_T=round(be, 2),
                             breakeven_vs_fwd_pct=round(100 * (be / fwd - 1), 2)))
    return pd.DataFrame(rows)


def ruin_empirical():
    """Historical 2y drawdown distribution of the project universe."""
    rows = []
    for p in sorted((ROOT / "data").glob("hist_*.parquet")):
        tkr = p.stem.replace("hist_", "")
        if tkr in ("VIX",):
            continue
        h = pd.read_parquet(p)
        px = h["Adj Close"].astype(float).dropna()
        if len(px) < 1500:
            continue
        r2y = (px / px.shift(504) - 1.0).dropna()
        r1y = (px / px.shift(252) - 1.0).dropna()
        rows.append(dict(ticker=tkr,
                         w2y_p05=round(100 * r2y.quantile(0.05), 1),
                         w2y_p01=round(100 * r2y.quantile(0.01), 1),
                         w2y_min=round(100 * r2y.min(), 1),
                         w2y_frac_lt_m30=round(100 * (r2y < -0.30).mean(), 1),
                         w2y_frac_lt_m50=round(100 * (r2y < -0.50).mean(), 1),
                         w1y_frac_lt_m30=round(100 * (r1y < -0.30).mean(), 1)))
    return pd.DataFrame(rows)


# Sections C and D removed -- see the module docstring.
# Their correct American-priced replacements live in 04_drift_and_capital.py.


def main():
    print("=" * 78)
    print("A. EXPOSURE DRIFT  (net delta exposure per $1 of initial leg exposure)")
    print("   strikes fixed at inception (T0=2y) so that |delta| = 0.80")
    print("=" * 78)
    df, piv, Kc, Kp = exposure_drift()
    print(f"\nK_call = {Kc:.2f}   K_put = {Kp:.2f}   (S0 = {S0})\n")
    print(piv.round(3).to_string())
    print("\npositive = net LONG, negative = net SHORT.  Columns are market moves in %.")
    drift = df[df.mkt_move == -30.0].set_index("tau").net_exposure
    print("\nnet exposure after a -30% market, by remaining tenor:")
    print(drift.round(3).to_string())

    print("\n" + "=" * 78)
    print("B. RUIN  (delta-targeted LEAPS call)")
    print("=" * 78)
    rt = ruin_table()
    print(rt.to_string(index=False))

    print("\n" + "=" * 78)
    print("B2. empirical 2-year drawdowns, project universe")
    print("=" * 78)
    re_ = ruin_empirical()
    print(re_.to_string(index=False))
    print("\nmedian across names: w2y_p01 = %.1f%%, frac(2y < -30%%) = %.1f%%, "
          "frac(2y < -50%%) = %.1f%%"
          % (re_.w2y_p01.median(), re_.w2y_frac_lt_m30.median(), re_.w2y_frac_lt_m50.median()))

    df.to_csv(OUT / "risk_exposure_drift.csv", index=False)
    rt.to_csv(OUT / "risk_ruin.csv", index=False)
    re_.to_csv(OUT / "risk_ruin_empirical.csv", index=False)
    print("\nwrote risk_*.csv")


if __name__ == "__main__":
    main()
