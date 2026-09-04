"""Path P&L of a LEAPS-implemented L/S book, and the capital question.

The short leg MUST be priced American.  A deep-ITM European put is worth
K e^{-rT} - S + ..., which understates reality: the holder can exercise now
and receive K - S.  For delta = -0.80 at T = 2y that gap is worth ~8 points of
spot, and it also flips the sign of theta (a deep-ITM European put has
POSITIVE theta, an American one has essentially none).  European put maths
therefore makes the short side look far cheaper and far more profitable than
it is.  Everything below uses a binomial American valuation.

Q1  Total P&L of the LEAPS book over a market path, by actual repricing.
    The stock book is dollar-neutral throughout, so its P&L on a common
    market move is 0 by construction: whatever the LEAPS book earns or loses
    is the cost of the structure.

Q2  Capital required to hold $1 long delta + $1 short delta.

Outputs: research/results/drift_*.csv
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import binomial as bn
import bs
from _common import DELTA_TGT, OUT, Q, R, S0, SIG, strike_for_delta

NSTEPS = 400
Q_DISC = [0.25 * k for k in range(1, 9)]      # quarterly dividend dates


def sched(T: float, q: float, r: float):
    """Flat quarterly dividend schedule of a total of q*S0 per year."""
    return [(t, q * S0 / 4.0) for t in Q_DISC if t <= T + 1e-9]


def amer_call(S, K, T, sig):
    return bn.crr_american(S, K, T, R, sig, sched(T, Q, R), NSTEPS, "C")[0]


def amer_put(S, K, T, sig):
    return bn.crr_american(S, K, T, R, sig, sched(T, Q, R), NSTEPS, "P")[0]


def path_pnl(path, T0=2.0, vol=SIG, n_sub=24):
    Kc = strike_for_delta(S0, T0, R, Q, vol, DELTA_TGT, "C")
    Kp = strike_for_delta(S0, T0, R, Q, vol, DELTA_TGT, "P")
    n = 1.0 / (DELTA_TGT * S0)

    cap = n * amer_call(S0, Kc, T0, vol) + n * amer_put(S0, Kp, T0, vol)

    ts, ss = [], []
    for i in range(len(path) - 1):
        t0, m0 = path[i]
        t1, m1 = path[i + 1]
        for k in range(n_sub):
            ts.append(t0 + (t1 - t0) * k / n_sub)
            ss.append(m0 * (m1 / m0) ** (k / n_sub))
    ts.append(path[-1][0])
    ss.append(path[-1][1])

    V = []
    net, gross = [], []
    for t, m in zip(ts, ss):
        S = S0 * m
        tau = max(T0 - t, 1e-3)
        V.append(n * (amer_call(S, Kc, tau, vol) + amer_put(S, Kp, tau, vol)))
        dc = bs.delta_call(S, Kc, tau, R, Q, vol)
        dp = bs.delta_put(S, Kp, tau, R, Q, vol)
        net.append(n * (dc + dp) * S)
        gross.append(n * (dc - dp) * S)
    V = np.array(V)
    net = np.array(net)
    return dict(Kc=Kc, Kp=Kp, cap=cap, V=V, net=net, gross=np.array(gross),
                t=np.array(ts), S=S0 * np.array(ss),
                pnl_pct=100 * (V[-1] - V[0]) / cap,
                max_dd_pct=100 * (V.min() - V[0]) / cap,
                net_at_min=net[np.argmin(V)])


def capital_table():
    rows = []
    for T in (0.5, 1.0, 2.0):
        for tgt in (0.70, 0.80, 0.90):
            Kc = strike_for_delta(S0, T, R, Q, SIG, tgt, "C")
            Kp = strike_for_delta(S0, T, R, Q, SIG, tgt, "P")
            C_a = amer_call(S0, Kc, T, SIG)
            C_e = bs.call(S0, Kc, T, R, Q, SIG)
            P_a = amer_put(S0, Kp, T, SIG)
            P_e = bs.put(S0, Kp, T, R, Q, SIG)
            n = 1.0 / (tgt * S0)
            rows.append(dict(T=T, delta_tgt=tgt,
                             K_call_pct_spot=round(100 * Kc / S0, 1),
                             K_put_pct_spot=round(100 * Kp / S0, 1),
                             call_prem_pct=round(100 * C_a / S0, 1),
                             put_prem_A_pct=round(100 * P_a / S0, 1),
                             put_prem_E_pct=round(100 * P_e / S0, 1),
                             put_EE_prem_pct=round(100 * (P_a - P_e) / S0, 1),
                             cap_long=round(100 * n * C_a, 1),
                             cap_short_A=round(100 * n * P_a, 1),
                             cap_total_A=round(100 * n * (C_a + P_a), 1),
                             cap_total_E=round(100 * n * (C_e + P_e), 1)))
    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print("Q1  LEAPS-BOOK P&L OVER MARKET PATHS  (American valuation, no frictions)")
    print("    stock benchmark = 0.0% on every path (dollar-neutral by construction)")
    print("=" * 78)
    scen = {
        "momentum crash: -50% then +90% (net -5%)": [(0.0, 1.00), (0.75, 0.50), (2.0, 0.95)],
        "V-shaped: -35% then +54% (net  0%)":       [(0.0, 1.00), (1.0, 0.65), (2.0, 1.00)],
        "grind down: -30% then +43% (net  0%)":     [(0.0, 1.00), (1.5, 0.70), (2.0, 1.00)],
        "melt up: +40% straight":                   [(0.0, 1.00), (2.0, 1.40)],
        "crash only: -50% and stay":                [(0.0, 1.00), (0.75, 0.50), (2.0, 0.50)],
        "flat market":                              [(0.0, 1.00), (2.0, 1.00)],
        "grind up: +20%":                           [(0.0, 1.00), (2.0, 1.20)],
    }
    rows, store = [], {}
    for name, p in scen.items():
        r = path_pnl(p)
        store[name] = r
        rows.append(dict(scenario=name, capital=round(r["cap"], 3),
                         net_exp_at_trough=round(r["net"].min(), 3),
                         net_exp_end=round(r["net"][-1], 3),
                         pnl_pct_of_capital=round(r["pnl_pct"], 2),
                         max_dd_pct_of_capital=round(r["max_dd_pct"], 2)))
    res = pd.DataFrame(rows)
    print()
    print(res.to_string(index=False))

    mc = store["momentum crash: -50% then +90% (net -5%)"]
    idx = np.linspace(0, len(mc["t"]) - 1, 11).astype(int)
    print("\n--- detail: momentum crash path ---")
    print(pd.DataFrame(dict(t=mc["t"][idx].round(2), S=mc["S"][idx].round(1),
                            net_exposure=mc["net"][idx].round(3),
                            gross_exposure=mc["gross"][idx].round(3),
                            book_value=mc["V"][idx].round(3))).to_string(index=False))
    print(f"\ncapital at risk = {mc['cap']:.3f} for $1 long + $1 short delta")

    print("\n" + "=" * 78)
    print("Q2  CAPITAL FOR $1 LONG DELTA + $1 SHORT DELTA  (% of that leg's exposure)")
    print("=" * 78)
    cap = capital_table()
    print()
    print(cap.to_string(index=False))
    print("\n'put_EE_prem_pct' is the early-exercise premium the European formula misses.")

    alts = pd.DataFrame([
        dict(route="stock, Reg T (long paid in full + 50% short margin)", capital=1.50),
        dict(route="stock, portfolio margin (~20% each leg)", capital=0.40),
        dict(route="stock, portfolio margin (~30% each leg)", capital=0.60),
        dict(route="stock, fully funded both legs", capital=2.00),
    ])
    print("\nStock route (total capital for the two-legged book):\n")
    print(alts.to_string(index=False))

    b = cap[(cap["T"] == 2.0) & (cap["delta_tgt"] == 0.80)].iloc[0]
    lev = b.cap_total_A / 100.0
    print(f"\nAt T=2y, delta=0.80, American: LEAPS book needs {lev:.2f} of capital")
    for _, a in alts.iterrows():
        print(f"   vs {a['route']:<52s} {a['capital']:.2f}   -> LEAPS uses "
              f"{100*lev/a['capital']:.0f}% of it")
    print(f"\n(European put maths would have said {b.cap_total_E/100:.2f} -- "
          f"i.e. it understates the true capital by "
          f"{100*(b.cap_total_A-b.cap_total_E)/b.cap_total_E:.0f}%)")

    res.to_csv(OUT / "drift_scenarios.csv", index=False)
    cap.to_csv(OUT / "drift_capital_leaps.csv", index=False)
    alts.to_csv(OUT / "drift_capital_stock.csv", index=False)
    print("\nwrote drift_*.csv")


if __name__ == "__main__":
    main()
