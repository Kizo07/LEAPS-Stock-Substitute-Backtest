"""12-1 total-return momentum signal (PLAN §5.9).

Momentum at date t = TRI(t - MOMENTUM_SKIP) / TRI(t - MOMENTUM_LOOKBACK -
MOMENTUM_SKIP) - 1 over config.STOCK_UNIVERSE (indices excluded), using only
closes on or before t (the signal is computed at t and traded at t's close).
Top/bottom quintile equal weight, dollar-neutral (longs sum +1, shorts sum -1);
names with insufficient history are excluded from the sort.
"""
from __future__ import annotations

import pandas as pd

from .. import config


def momentum_frame(md) -> pd.DataFrame:
    """Momentum per stock-universe ticker aligned to the MarketData calendar."""
    cols = [tk for tk in config.STOCK_UNIVERSE if tk in md.tri.columns]
    tri = md.tri[cols]
    return tri.shift(config.MOMENTUM_SKIP) / tri.shift(
        config.MOMENTUM_LOOKBACK + config.MOMENTUM_SKIP) - 1.0


def momentum_targets(date, mom: pd.DataFrame, quintile: int = config.QUINTILE) -> dict[str, float]:
    """Equal-weight top/bottom quintile targets (fractions of NAV) at ``date``."""
    row = mom.loc[pd.Timestamp(date)].dropna()
    n = len(row) // quintile
    if n == 0:
        return {}
    ranked = row.sort_values()
    targets: dict[str, float] = {}
    for tk in ranked.index[-n:]:
        targets[tk] = 1.0 / n
    for tk in ranked.index[:n]:
        targets[tk] = -1.0 / n
    return targets
