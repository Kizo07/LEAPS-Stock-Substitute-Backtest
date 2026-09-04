"""Empirical CAPM / factor tests on the project's own backtest output.

Reads results/nav_portfolios.parquet (V0 stock .. V3 hybrid) and the Ken-French
daily factors, then asks the questions the theory says to ask:

  1. Static FF3+MOM attribution (replication check).
  2. Is the market beta STABLE?  Rolling 252-day betas -- the theory predicts
     a LEAPS book has a state-dependent beta, a stock book does not.
  3. Treynor-Mazuy curvature test:  r = a + b x + g x^2.  The theory predicts
     a LEAPS book picks up curvature in the market factor.
  4. Does the LEAPS variant load on a volatility factor it should not?  We use
     the VIX level change as a proxy.
  5. Down-market vs up-market beta asymmetry.

Outputs: research/results/emp_*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def ols(y, X):
    X = np.column_stack([np.ones(len(X))] + [np.asarray(c, float) for c in X.T]) \
        if X.ndim > 1 else np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    resid = np.asarray(y, float) - X @ beta
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / np.where(se > 0, se, np.nan)
    r2 = 1 - (resid @ resid) / (np.asarray(y, float) - np.mean(y)) @ 0 \
        if False else 1 - (resid @ resid) / np.sum((np.asarray(y, float) - np.mean(y)) ** 2)
    return beta, t, r2


def main():
    nav = pd.read_parquet(RES / "nav_portfolios.parquet")
    print("nav_portfolios.parquet columns:", nav.columns.tolist())
    print(nav.head(3).to_string())
    nav = nav.sort_index()
    ret = nav.pct_change().dropna()
    if ret.abs().median().max() > 0.5:        # looks like a level series of 1e6
        pass
    print("\nreturn series summary (%/day):\n", (100 * ret.describe()).round(4).to_string())

    ff = pd.read_parquet(ROOT / "data" / "french_factors.parquet")
    print("\nfrench_factors columns:", ff.columns.tolist())
    print(ff.tail(2).to_string())

    vix = pd.read_parquet(ROOT / "data" / "hist_VIX.parquet")["Adj Close"].astype(float)
    dvix = vix.pct_change()

    j = ret.join(ff, how="inner").dropna()
    j["dvix"] = dvix.reindex(j.index)
    print("\njoined", j.shape, j.index.min().date(), "->", j.index.max().date())

    variants = [c for c in nav.columns if c.upper().startswith("V")]
    print("variants:", variants)

    # ---- 1. static FF3 + MOM -------------------------------------------
    rows = []
    fac = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"] \
        if "RMW" in ff.columns else [c for c in ff.columns if c != "RF"]
    for v in variants:
        y = j[v] - j["RF"]
        X = j[fac].to_numpy(float)
        b, t, r2 = ols(y, X)
        rows.append(dict(series=v, alpha_ann=round(100 * 252 * b[0], 2),
                         alpha_t=round(t[0], 2), **{f"b_{f}": round(b[i + 1], 3)
                                                    for i, f in enumerate(fac)},
                         r2=round(r2, 3)))
    st = pd.DataFrame(rows)
    print("\n=== 1. static FF attribution (daily, excess returns) ===")
    print(st.to_string(index=False))

    # ---- 2. rolling market beta ----------------------------------------
    print("\n=== 2. rolling 252-day market beta ===")
    rb = {}
    for v in variants:
        y = j[v] - j["RF"]
        x = j["Mkt-RF"] if "Mkt-RF" in j.columns else j[fac[0]]
        cov = y.rolling(252).cov(x)
        var = x.rolling(252).var()
        rb[v] = cov / var
    roll = pd.DataFrame(rb).dropna()
    print(roll.describe().round(3).to_string())
    print("\nrolling-beta dispersion is the CAPM-stability diagnostic:")
    print(pd.DataFrame(dict(mean=roll.mean(), std=roll.std(),
                            p05=roll.quantile(0.05), p95=roll.quantile(0.95),
                            range=roll.quantile(0.95) - roll.quantile(0.05))).round(3).to_string())

    # ---- 3. Treynor-Mazuy curvature ------------------------------------
    print("\n=== 3. Treynor-Mazuy: r = a + b x + g x^2  (market timing / convexity) ===")
    x = j["Mkt-RF"].to_numpy(float) if "Mkt-RF" in j.columns else j[fac[0]].to_numpy(float)
    rows = []
    for v in variants:
        y = (j[v] - j["RF"]).to_numpy(float)
        X2 = np.column_stack([np.ones_like(x), x, x ** 2])
        b, t, r2 = ols(y, X2[:, 1:])
        rows.append(dict(series=v, alpha_ann=round(100 * 252 * b[0], 2),
                         beta=round(b[1], 3), gamma=round(b[2], 2),
                         gamma_t=round(t[2], 2), r2=round(r2, 3)))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\ngamma is in units of 'per unit of squared daily market excess return'.")
    print("A gamma of 100 means: on a +-2% market day (x^2 = 4e-4) the curvature term")
    print("contributes 100 * 4e-4 = 4bp of extra return.")

    # ---- 4. VIX loading -------------------------------------------------
    print("\n=== 4. does the LEAPS book load on vol? (adding dVIX to FF+MOM) ===")
    rows = []
    for v in variants:
        y = j[v] - j["RF"]
        X0 = j[fac].to_numpy(float)
        b0, t0, r20 = ols(y, X0)
        k = j.dropna(subset=["dvix"])
        yk = (k[v] - k["RF"]).to_numpy(float)
        X1 = np.column_stack([k[fac].to_numpy(float), k["dvix"].to_numpy(float)])
        b1, t1, r21 = ols(yk, X1)
        rows.append(dict(series=v, r2_ff=round(r20, 3), r2_ff_vix=round(r21, 3),
                         b_dvix=round(b1[-1], 4), t_dvix=round(t1[-1], 2)))
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- 5. up/down beta ------------------------------------------------
    print("\n=== 5. up-market vs down-market beta ===")
    mkt = j["Mkt-RF"] if "Mkt-RF" in j.columns else j[fac[0]]
    up = mkt > 0
    dn = mkt < 0
    rows = []
    for v in variants:
        y = j[v] - j["RF"]
        bu, tu, _ = ols(y[up], mkt[up].to_numpy(float))
        bd, td, _ = ols(y[dn], mkt[dn].to_numpy(float))
        rows.append(dict(series=v, beta_up=round(bu[1], 3), t_up=round(tu[1], 1),
                         beta_down=round(bd[1], 3), t_down=round(td[1], 1),
                         asym=round(bd[1] - bu[1], 3)))
    print(pd.DataFrame(rows).to_string(index=False))

    st.to_csv(OUT / "emp_ff_static.csv", index=False)
    roll.to_csv(OUT / "emp_rolling_beta.csv")
    pd.DataFrame(rows).to_csv(OUT / "emp_updown_beta.csv", index=False)
    print("\nwrote emp_*.csv")


if __name__ == "__main__":
    main()
