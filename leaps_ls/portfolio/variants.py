"""Strategy variants and single-instrument lanes (PLAN §5.9).

Portfolios (monthly momentum L/S, dollar-neutral):
- V0 all-stock baseline (shorts charged borrow);
- V1 LEAPS replacement (longs -> 0.80-delta calls, shorts -> -0.80-delta puts);
- V2 synthetic (longs -> synthetic long, shorts -> synthetic short);
- V3 hybrid (longs LEAPS calls, shorts cash stock).

Single-instrument lanes: constant fully-invested target (1x NAV delta-notional)
rebalanced monthly, per ticker: stock / call / synth_long on the long side,
short-stock / put / synth_short on the short side. Deviation from the plan text:
lanes are constant-exposure monthly-rebalanced rather than literal buy-and-hold
(the engine is plan-driven); the difference is small monthly adjustment trades,
which are costed.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from . import signals

# variant -> (mode_long, mode_short)
VARIANT_MODES: dict[str, tuple[str, str]] = {
    "V0": ("stock", "stock"),
    "V1": ("call", "put"),
    "V2": ("synth_long", "synth_short"),
    "V3": ("call", "stock"),
}

# lane -> (signed weight, option/stock mode for that side)
LANES: dict[str, tuple[float, str]] = {
    "stock": (1.0, "stock"),
    "call": (1.0, "call"),
    "synth_long": (1.0, "synth_long"),
    "short_stock": (-1.0, "stock"),
    "put": (-1.0, "put"),
    "synth_short": (-1.0, "synth_short"),
}
LONG_LANES = ["stock", "call", "synth_long"]
SHORT_LANES = ["short_stock", "put", "synth_short"]


def rebalance_dates(days: pd.DatetimeIndex, start, end) -> list[pd.Timestamp]:
    """First trading day of each calendar month within [start, end]."""
    days = pd.DatetimeIndex(days)
    s = pd.Series(days, index=days)
    firsts = s.groupby([s.index.year, s.index.month]).min()
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return [pd.Timestamp(d) for d in firsts if lo <= d <= hi]


def portfolio_plan(md, start, end, mom: pd.DataFrame | None = None) -> pd.DataFrame:
    """Momentum L/S target weights at each monthly rebalance date."""
    dates = rebalance_dates(md.days, start, end)
    if mom is None:
        mom = signals.momentum_frame(md)
    rows = {d: signals.momentum_targets(d, mom) for d in dates}
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()


def lane_plan(md, ticker: str, lane: str, start, end) -> tuple[pd.DataFrame, str]:
    """Constant-weight plan for one ticker/lane plus its mode."""
    weight, mode = LANES[lane]
    dates = rebalance_dates(md.days, start, end)
    valid = md.valid_start(ticker)
    dates = [d for d in dates if d >= valid]
    plan = pd.DataFrame({ticker: weight}, index=pd.DatetimeIndex(dates))
    return plan, mode


def lane_modes(lane: str) -> tuple[str, str]:
    """(mode_long, mode_short) engine arguments for a lane."""
    weight, mode = LANES[lane]
    return (mode, "stock") if weight > 0 else ("stock", mode)
