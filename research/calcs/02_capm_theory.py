"""Does CAPM hold for a LEAPS-constructed long/short book?

PART 1 -- theory, verified numerically.
    Ito:  dC = (Theta + mu S Delta + 1/2 sig^2 S^2 Gamma) dt + sig S Delta dW
    so the option's INSTANTANEOUS beta is
        beta_C = (S Delta / C) beta_S = Omega * beta_S        [Omega = elasticity]
    and the BS PDE  Theta + r S Delta + 1/2 sig^2 S^2 Gamma = r C  makes
        E[dC]/C dt = r + beta_C (mu_S - r)
    hold EXACTLY.  CAPM is not merely approximately true for options -- in
    continuous time it is an identity.

PART 2 -- why it fails over any horizon you can actually trade.
    (a) Omega is not constant: it moves with S, tau, sigma.
    (b) Gamma convexity: the option return is quadratic in the market return.
    (c) The variance risk premium: you buy at IV, the world delivers RV < IV.
        This is a pure negative alpha -- the empirical content of
        Coval-Shumway (2001) and Bakshi-Kapadia (2003).

PART 3 -- exposure drift, the channel that actually hurts a L/S book.
    Long-call delta RISES with S; long-put |delta| FALLS with S.  A
    delta-targeted LEAPS book is therefore more long after rallies and more
    short after selloffs: its net exposure is a positively-sloped function of
    the market, unlike a dollar-neutral stock book whose exposure is constant.
    No frictions required -- this is pure structure.

Outputs: research/results/capm_*.csv
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
BETA_S, ERP = 1.0, 0.055          # stock beta 1, equity risk premium 5.5%
RHO = 0.85                        # corr(stock, market): R^2 of stock on mkt = 0.72
# beta_S = sig_S * rho / sig_M  =>  sig_M = sig_S * rho / beta_S
SIG_M = SIG * RHO / BETA_S
MU_TOT = R + BETA_S * ERP         # stock TOTAL-return drift
MU_PX = MU_TOT - Q                # stock PRICE drift (option sees the price)
NPATHS = 200_000


def omega_call(S, K, T, r, q, sig):
    return S * bs.delta_call(S, K, T, r, q, sig) / bs.call(S, K, T, r, q, sig)


def omega_put(S, K, T, r, q, sig):
    return S * bs.delta_put(S, K, T, r, q, sig) / bs.put(S, K, T, r, q, sig)


# --------------------------------------------------------------- part 1
def verify_instantaneous_capm() -> pd.DataFrame:
    rows = []
    for K in (60.0, 80.0, 100.0, 120.0):
        for T in (0.5, 1.0, 2.0):
            C = bs.call(S0, K, T, R, Q, SIG)
            d = bs.delta_call(S0, K, T, R, Q, SIG)
            g = bs.gamma(S0, K, T, R, Q, SIG)
            th = bs.theta_call(S0, K, T, R, Q, SIG)
            # Ito uses the PRICE drift; CAPM prices off the TOTAL return
            drift = (th + MU_PX * S0 * d + 0.5 * SIG ** 2 * S0 ** 2 * g) / C
            om = S0 * d / C
            pred = R + om * BETA_S * (MU_TOT - R)
            rows.append(dict(K=K, T=T, C=round(C, 4), delta=round(d, 4),
                             omega=round(om, 4), beta_inst=round(om * BETA_S, 4),
                             E_drift_ann=round(drift, 6), capm_pred_ann=round(pred, 6),
                             resid=drift - pred))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- part 2
def simulate(K, T_opt, n_steps, T_hold, vrp=0.0, kind="C", seed=7):
    """Buy an option at IV = SIG + vrp, hold T_hold, reprice at that same IV."""
    rng = np.random.default_rng(seed)
    dt = T_hold / n_steps
    iv = SIG + vrp
    S = np.full(NPATHS, S0)
    M = np.full(NPATHS, S0)
    C0 = bs.call(S0, K, T_opt, R, Q, iv) if kind == "C" else bs.put(S0, K, T_opt, R, Q, iv)
    for _ in range(n_steps):
        z1 = rng.standard_normal(NPATHS)
        z2 = rng.standard_normal(NPATHS)
        zm = RHO * z1 + math.sqrt(max(1 - RHO ** 2, 0.0)) * z2
        S = S * np.exp((MU_PX - 0.5 * SIG ** 2) * dt + SIG * math.sqrt(dt) * z1)
        M = M * np.exp((MU_TOT - 0.5 * SIG_M ** 2) * dt + SIG_M * math.sqrt(dt) * zm)
    tau = T_opt - T_hold
    if tau <= 1e-4:
        tau = 1e-4
    px = bs.call if kind == "C" else bs.put
    C1 = np.array([px(s, K, tau, R, Q, iv) for s in S])
    return (S / S0 - 1.0), (C1 / C0 - 1.0), (M / S0 - 1.0)


def horizon_capm_table() -> pd.DataFrame:
    rows = []
    for K, T_opt in [(80.0, 2.0), (90.0, 2.0), (100.0, 2.0), (110.0, 2.0), (100.0, 0.25)]:
        C0 = bs.call(S0, K, T_opt, R, Q, SIG)
        d0 = bs.delta_call(S0, K, T_opt, R, Q, SIG)
        om = S0 * d0 / C0
        for vrp in (0.0, 0.03):
            for hd, nst in [(1, 1), (5, 1), (21, 2), (63, 4)]:
                Th = hd / 252.0
                _, ro, rm = simulate(K, T_opt, nst, Th, vrp=vrp)
                x = rm - R * Th
                y = ro - R * Th
                A = np.column_stack([np.ones_like(x), x])
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                r2 = 1 - np.var(y - A @ coef) / np.var(y)
                rows.append(dict(
                    K=K, T_opt=T_opt, horizon_days=hd, vrp_volpts=round(100 * vrp, 0),
                    delta0=round(d0, 3), omega=round(om, 2),
                    beta_inst=round(om * BETA_S, 2),
                    beta_hat=round(coef[1], 2),
                    beta_bias_pct=round(100 * (coef[1] / (om * BETA_S) - 1), 1),
                    alpha_ann=round(100 * coef[0] * 252 / hd, 2),
                    r2=round(r2, 4),
                    E_ret_ann=round(100 * np.mean(ro) * 252 / hd, 2),
                    capm_pred_ann=round(100 * (R + om * BETA_S * ERP), 2)))
    return pd.DataFrame(rows)


def vrp_table() -> pd.DataFrame:
    """Annualised drag from buying at IV when the world delivers RV = IV - vrp."""
    rows = []
    for K in (60.0, 80.0, 90.0, 100.0, 110.0, 120.0):
        for T in (0.25, 1.0, 2.0):
            C = bs.call(S0, K, T, R, Q, SIG)
            V = bs.vega(S0, K, T, R, Q, SIG) / 100.0     # per 1 vol point
            for vrp in (2.0, 5.0):
                rows.append(dict(K=K, T=T, mny=round(K / S0, 2), prem=round(C, 2),
                                 vega_volpt=round(V, 4), vrp_volpts=vrp,
                                 drag_pct_of_premium=round(-100 * V * vrp / C, 2),
                                 drag_pct_of_spot=round(-100 * V * vrp / S0, 3)))
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 78)
    print("PART 1  instantaneous CAPM is an identity under Black-Scholes")
    print("=" * 78)
    t1 = verify_instantaneous_capm()
    print(t1.to_string(index=False))
    print("\nmax |resid| = %.3e  -> CAPM holds exactly in continuous time\n"
          % t1.resid.abs().max())

    print("=" * 78)
    print("PART 2  what happens over a tradeable horizon (regression of realised")
    print("        option returns on market returns; alpha/E-ret in %/yr)")
    print("=" * 78)
    t2 = horizon_capm_table()
    print(t2.to_string(index=False))

    print("\n" + "=" * 78)
    print("PART 2c  variance-risk-premium drag (buy IV, realise RV)")
    print("=" * 78)
    t3 = vrp_table()
    piv = t3[t3.vrp_volpts == 2.0].pivot_table(index="mny", columns="T",
                                               values="drag_pct_of_premium")
    print("\ndrag, % of option premium per year, VRP = 2 vol points:\n")
    print(piv.round(2).to_string())
    piv2 = t3[t3.vrp_volpts == 5.0].pivot_table(index="mny", columns="T",
                                                values="drag_pct_of_premium")
    print("\nsame, VRP = 5 vol points:\n")
    print(piv2.round(2).to_string())

    t1.to_csv(OUT / "capm_instantaneous.csv", index=False)
    t2.to_csv(OUT / "capm_horizon.csv", index=False)
    t3.to_csv(OUT / "capm_vrp_drag.csv", index=False)
    print("\nwrote capm_instantaneous.csv, capm_horizon.csv, capm_vrp_drag.csv")


if __name__ == "__main__":
    main()
