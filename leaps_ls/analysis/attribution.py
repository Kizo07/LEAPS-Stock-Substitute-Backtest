"""Factor attribution (RQ3) and rate-regime sub-periods (PLAN §5.10).

Daily-return OLS of V0-V3 on Fama-French 3 + Momentum (Ken French daily, inner
join — the factor series ends ~2 months before the NAV series). Left-hand side
is the portfolio EXCESS return (ret - RF). Reports annualized alpha with t-stat,
betas with t-stats, and R^2. Also emits a sub-period table (CAGR/Sharpe) for the
portfolio variants and the SPY lanes across rate regimes.
Runs offline from results/ plus the cached French factors.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from leaps_ls import config  # noqa: E402
from leaps_ls.data import french  # noqa: E402
from leaps_ls.analysis import metrics  # noqa: E402

FACTORS = ["Mkt-RF", "SMB", "HML", "MOM"]
SUBPERIODS = [
    ("2007-2009", "2007-01-01", "2009-12-31"),
    ("2010-2019", "2010-01-01", "2019-12-31"),
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-present", "2022-01-01", None),
]


def ols(y: np.ndarray, X: np.ndarray) -> dict:
    """Plain OLS with homoskedastic standard errors. X includes a constant column."""
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - k, 1)
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * s2)
    t = beta / se
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else float("nan")
    return {"beta": beta, "se": se, "t": t, "r2": r2, "n": n}


def factor_regression(ret: pd.Series, factors: pd.DataFrame) -> dict:
    """Excess-return regression of a daily return series on FF3+MOM."""
    df = pd.concat([ret.rename("ret"), factors], axis=1, join="inner").dropna()
    y = (df["ret"] - df["RF"]).to_numpy()
    X = np.column_stack([np.ones(len(df)), df[FACTORS].to_numpy()])
    fit = ols(y, X)
    return {
        "alpha_ann": fit["beta"][0] * metrics.TRADING_DAYS,
        "alpha_t": fit["t"][0],
        **{f"beta_{name}": fit["beta"][i + 1] for i, name in enumerate(FACTORS)},
        **{f"t_{name}": fit["t"][i + 1] for i, name in enumerate(FACTORS)},
        "r2": fit["r2"], "n": fit["n"],
    }


def subperiod_table(navs: dict[str, pd.Series]) -> pd.DataFrame:
    """CAGR/Sharpe per series per rate-regime window."""
    rows = []
    for label, start, end in SUBPERIODS:
        for name, nav in navs.items():
            sub = nav.loc[start:pd.Timestamp(end) if end else None].dropna()
            if len(sub) < 30:
                continue
            rows.append({
                "window": label, "series": name,
                "cagr": metrics.cagr(sub / sub.iloc[0]),
                "sharpe": metrics.sharpe(metrics.daily_returns(sub)),
                "maxdd": metrics.max_drawdown(sub / sub.cummax()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    pf = pd.read_parquet(config.RESULTS_DIR / "nav_portfolios.parquet")
    lanes = pd.read_parquet(config.RESULTS_DIR / "nav_lanes.parquet")
    factors = french.load_factors()

    rows = []
    for v in pf.columns:
        rows.append({"series": v, **factor_regression(pf[v].pct_change().dropna(), factors)})
    attr = pd.DataFrame(rows)
    attr.to_csv(config.RESULTS_DIR / "ff_attribution.csv", index=False)

    series = {v: pf[v] for v in pf.columns}
    for lane in ("stock", "call", "synth_long", "short_stock", "put", "synth_short"):
        series[f"SPY:{lane}"] = lanes[f"SPY:{lane}"]
    sub = subperiod_table(series)
    sub.to_csv(config.RESULTS_DIR / "subperiods.csv", index=False)

    show = ["series", "alpha_ann", "alpha_t", "beta_Mkt-RF", "t_Mkt-RF", "beta_MOM",
            "t_MOM", "beta_HML", "beta_SMB", "r2"]
    print(attr[show].to_string(index=False, formatters={
        "alpha_ann": "{:+.3f}".format, "alpha_t": "{:+.1f}".format,
        "beta_Mkt-RF": "{:+.3f}".format, "t_Mkt-RF": "{:+.1f}".format,
        "beta_MOM": "{:+.3f}".format, "t_MOM": "{:+.1f}".format,
        "beta_HML": "{:+.3f}".format, "beta_SMB": "{:+.3f}".format,
        "r2": "{:.3f}".format}))
    print("\nWrote ff_attribution.csv, subperiods.csv")


if __name__ == "__main__":
    main()
