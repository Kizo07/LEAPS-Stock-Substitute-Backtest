"""Discrete dividend schedules and escrowed-dividend adjusted spot S* (PLAN §5.1).

Two schedule kinds:
- :func:`realized_dividends` — actual ex-dated dividends (engine cash ledger, ex-date handling).
- :func:`project_dividends` — ex-ante schedule estimated as of a trade date using ONLY
  dividends ex-dated on or before that date (strict no-look-ahead).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

_FREQ_STEPS = (1, 2, 4, 12)  # annual, semiannual, quarterly, monthly
_SCHEDULE_COLS = ["ex_date", "amount", "t"]


def realized_dividends(hist: pd.DataFrame) -> pd.DataFrame:
    """Actual dividend schedule (``ex_date``, ``amount``) from a Yahoo history frame."""
    div = hist["Dividends"].fillna(0.0).astype(float)
    div = div[div > 0.0]
    return pd.DataFrame({"ex_date": div.index, "amount": div.to_numpy()})


def project_dividends(
    hist: pd.DataFrame, as_of_date, horizon_years: float
) -> pd.DataFrame:
    """Projected discrete dividend schedule as of ``as_of_date`` (ex-ante).

    Frequency = number of ex-dates in the trailing 12 months, snapped to
    {1, 2, 4, 12} per year (default quarterly if the name pays but nothing
    ex-dated in the trailing year). Amount = most recent per-share dividend.
    Anchored to the last actual ex-date and stepped forward; names whose last
    ex-date is stale (> 1.5 payment intervals old) are treated as non-payers.
    Dividends ex-dated after ``as_of_date`` are NEVER used.

    Returns a frame with ``ex_date``, ``amount`` and ``t`` (years from as_of).
    """
    as_of = pd.Timestamp(as_of_date)
    actual = realized_dividends(hist)
    actual = actual[actual["ex_date"] <= as_of]
    if actual.empty:
        return pd.DataFrame(columns=_SCHEDULE_COLS)
    trailing = actual[actual["ex_date"] > as_of - pd.Timedelta(days=365)]
    n = len(trailing)
    freq = min(_FREQ_STEPS, key=lambda f: (abs(f - n), -f)) if n > 0 else 4
    step_days = 365.25 / freq
    last_ex = actual["ex_date"].iloc[-1]
    if (as_of - last_ex).days > 1.5 * step_days:
        return pd.DataFrame(columns=_SCHEDULE_COLS)  # stale payer -> no projection
    amount = float(actual["amount"].iloc[-1])
    rows: list[tuple] = []
    k = 1
    while True:
        d = last_ex + pd.Timedelta(days=step_days * k)
        t = (d - as_of).days / 365.25
        if t <= 0.0:
            k += 1
            continue
        if t > horizon_years + 1e-9:
            break
        rows.append((d, amount, t))
        k += 1
    return pd.DataFrame(rows, columns=_SCHEDULE_COLS)


def adjusted_spot(
    S: float, schedule: pd.DataFrame, rate_fn: Callable[[object, float], float], T: float
) -> float:
    """S* = S − PV(dividends ex-dating before expiry), i.e. schedule rows with t ≤ T.

    ``rate_fn(ex_date, t_years)`` returns the decimal discount rate for a cash flow
    ``t_years`` out (e.g. ``lambda d, t: curve.rate(as_of, t)``).
    """
    if schedule is None or len(schedule) == 0:
        return float(S)
    sch = schedule[schedule["t"] <= T + 1e-12]
    if sch.empty:
        return float(S)
    pv = 0.0
    for ex_date, amount, t in sch.itertuples(index=False):
        r = float(rate_fn(ex_date, float(t)))
        pv += float(amount) * np.exp(-r * float(t))
    return float(S) - pv
