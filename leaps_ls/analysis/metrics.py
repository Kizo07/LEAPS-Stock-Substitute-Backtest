"""Headline performance metrics (PLAN §5.10 subset used by script 03)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(nav: pd.Series) -> pd.Series:
    """Simple daily returns of a NAV series."""
    return nav.pct_change().dropna()


def cagr(nav: pd.Series) -> float:
    """Compound annual growth rate from the NAV index."""
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)


def ann_vol(ret: pd.Series) -> float:
    """Annualized volatility of daily returns."""
    return float(ret.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sharpe(ret: pd.Series) -> float:
    """Annualized Sharpe ratio of daily returns (no RF subtraction)."""
    s = ret.std(ddof=0)
    return float(ret.mean() / s * np.sqrt(TRADING_DAYS)) if s > 0 else float("nan")


def sortino(ret: pd.Series) -> float:
    """Annualized Sortino ratio of daily returns (downside deviation vs 0)."""
    down = ret[ret < 0.0]
    d = float(down.std(ddof=0)) if len(down) > 1 else 0.0
    return float(ret.mean() / d * np.sqrt(TRADING_DAYS)) if d > 0 else float("nan")


def max_drawdown(nav: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative fraction)."""
    return float((nav / nav.cummax() - 1.0).min())


def calmar(nav: pd.Series) -> float:
    """CAGR divided by the absolute maximum drawdown."""
    dd = abs(max_drawdown(nav))
    return float(cagr(nav) / dd) if dd > 0 else float("nan")


def ann_tracking_diff(ret_a: pd.Series, ret_b: pd.Series) -> float:
    """Annualized mean daily return difference (a - b), aligned on common days."""
    a, b = ret_a.align(ret_b, join="inner")
    return float((a - b).mean() * TRADING_DAYS)


def headline(nav: pd.Series) -> dict:
    """CAGR, vol, Sharpe, maxDD for a NAV series."""
    ret = daily_returns(nav)
    return {
        "cagr": cagr(nav),
        "vol": ann_vol(ret),
        "sharpe": sharpe(ret),
        "sortino": sortino(ret),
        "maxdd": max_drawdown(nav),
        "calmar": calmar(nav),
    }
