"""Delta-targeted LEAPS contract selection (PLAN §5.4).

Expiry: third Friday of the January whose DTE in [TENOR_MIN_DAYS, TENOR_MAX_DAYS]
is closest to TENOR_TARGET_DAYS. Strikes for the stock-substitute legs are solved
so |delta| = DELTA_TARGET exactly (Brent root-find; delta is monotone in K).
Listed-strike discreteness is abstracted away — the execution cost of real strike
granularity is covered by the calibrated half-spread model (PLAN §5.7). Synthetic
lanes use the forward strike S* e^{rT} (continuous-strike abstraction), which puts
net premium at zero before costs by put-call parity.

All pricing uses the Phase-2 model: escrowed spot S* from dividends projected as
of the trade date (strict no-look-ahead), the calibrated IV proxy, and the FRED
curve. Prices are trade-date close model prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .. import config
from ..pricing import black_scholes as bs, dividends as dvd, vol

LEG_SIGNS: dict[str, list[dict]] = {
    "call": [{"cp": "C", "sign": 1}],
    "put": [{"cp": "P", "sign": 1}],
    "synth_long": [{"cp": "C", "sign": 1}, {"cp": "P", "sign": -1}],
    "synth_short": [{"cp": "P", "sign": 1}, {"cp": "C", "sign": -1}],
}


def third_friday_january(year: int) -> pd.Timestamp:
    """Third Friday of January (standard January LEAPS expiry)."""
    d = pd.Timestamp(year=year, month=1, day=1)
    first_friday = d + pd.Timedelta(days=(4 - d.weekday()) % 7)
    return first_friday + pd.Timedelta(days=14)


def select_expiry(date, target_days: int | None = None) -> tuple[pd.Timestamp, int]:
    """January expiry with DTE in [TENOR_MIN, TENOR_MAX] closest to the tenor target."""
    date = pd.Timestamp(date)
    target = config.TENOR_TARGET_DAYS if target_days is None else int(target_days)
    cands = []
    for y in range(date.year + 1, date.year + 4):
        exp = third_friday_january(y)
        cands.append((exp, (exp - date).days))
    in_range = [c for c in cands if config.TENOR_MIN_DAYS <= c[1] <= config.TENOR_MAX_DAYS] or cands
    return min(in_range, key=lambda c: abs(c[1] - target))


def solve_strike_for_delta(
    S_star: float, T: float, r: float, iv_atm: float, slope: float,
    target_abs: float, cp: str,
) -> tuple[float, float, float, bool]:
    """Strike with |delta| = ``target_abs`` (skew-aware), monotone root-find in K.

    Returns (K, delta, iv_at_K, fallback). If the root bracket fails (extreme
    rates/dividends), falls back to the grid K with |delta| closest to target
    inside DELTA_BAND (closest overall if the band is empty), flagged.
    """
    target = target_abs if cp.upper() == "C" else -target_abs

    def f(K: float) -> float:
        iv = float(vol.apply_skew(iv_atm, K, S_star, slope))
        return float(bs.delta(S_star, K, T, r, iv, cp)) - target

    lo, hi = 0.3 * S_star, 2.5 * S_star
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo, float(bs.delta(S_star, lo, T, r, float(vol.apply_skew(iv_atm, lo, S_star, slope)), cp)), float(vol.apply_skew(iv_atm, lo, S_star, slope)), False
    if flo * fhi < 0:
        K = brentq(f, lo, hi, xtol=1e-10, rtol=1e-12)
        iv = float(vol.apply_skew(iv_atm, K, S_star, slope))
        return float(K), float(bs.delta(S_star, K, T, r, iv, cp)), iv, False
    # fallback: scan a grid
    grid = np.linspace(lo, hi, 400)
    ivs = vol.apply_skew(iv_atm, grid, S_star, slope)
    deltas = np.asarray(bs.delta(S_star, grid, T, r, ivs, cp), dtype=float)
    absd = np.abs(deltas)
    band = (absd >= config.DELTA_BAND[0]) & (absd <= config.DELTA_BAND[1])
    pool = np.where(band)[0] if band.any() else np.arange(len(grid))
    j = int(pool[np.argmin(np.abs(absd[pool] - target_abs))])
    return float(grid[j]), float(deltas[j]), float(ivs[j]), True


def select_contract(date, ticker: str, kind: str, md,
                    delta_target: float | None = None,
                    tenor_target_days: int | None = None,
                    iv_mult_adj: float = 1.0) -> dict:
    """Contract spec for ``kind`` in {call, put, synth_long, synth_short} as of ``date``.

    The spec carries everything the engine needs to open and mark the position:
    per-leg mids/deltas/IVs and the entry-frozen projected dividend schedule.
    The optional overrides drive the PLAN §7 sensitivity grid.
    """
    if kind not in LEG_SIGNS:
        raise ValueError(f"unknown contract kind {kind!r}")
    date = pd.Timestamp(date)
    i = int(md.days.get_loc(date))
    j = md.j[ticker]
    expiry, dte = select_expiry(date, tenor_target_days)
    T = dte / 365.0
    S = float(md.close_arr[i, j])
    r = md.rate_ord(md.day_ordinals[i], T)
    schedule = dvd.project_dividends(md.hist(ticker), date, horizon_years=T + 0.5)
    rate_fn = lambda d, t: md.rate_ord(md.day_ordinals[i], max(float(t), 1e-6))  # noqa: B023
    S_star = max(dvd.adjusted_spot(S, schedule, rate_fn, T), 0.01 * S)  # floor vs stale projections
    iv_atm = md.iv_atm(ticker, i) * iv_mult_adj
    if not np.isfinite(iv_atm) or iv_atm <= 0.0:
        raise ValueError(f"no valid IV proxy for {ticker} on {date.date()}")
    slope = md.slope(ticker)

    fallback = False
    if kind in ("call", "put"):
        cp = "C" if kind == "call" else "P"
        target = config.DELTA_TARGET if delta_target is None else float(delta_target)
        K, dl, iv, fallback = solve_strike_for_delta(
            S_star, T, r, iv_atm, slope, target, cp)
        px = float(bs.price(S_star, K, T, r, iv, cp))
        legs = [{"cp": cp, "sign": 1, "premium": px, "delta": dl, "iv": iv}]
    else:
        # forward strike: net premium ~0 by parity, net delta ~ +-1
        K = float(S_star * np.exp(r * T))
        iv = float(vol.apply_skew(iv_atm, K, S_star, slope))
        pc = float(bs.price(S_star, K, T, r, iv, "C"))
        pp = float(bs.price(S_star, K, T, r, iv, "P"))
        dc = float(bs.delta(S_star, K, T, r, iv, "C"))
        dp = float(bs.delta(S_star, K, T, r, iv, "P"))
        prem = {"C": pc, "P": pp}
        dl_map = {"C": dc, "P": dp}
        legs = [
            {"cp": leg["cp"], "sign": leg["sign"], "premium": prem[leg["cp"]],
             "delta": dl_map[leg["cp"]], "iv": iv}
            for leg in LEG_SIGNS[kind]
        ]

    premium = float(sum(l["sign"] * l["premium"] for l in legs))
    delta_net = float(sum(l["sign"] * l["delta"] for l in legs))
    if len(schedule):
        sched_days = np.array([d.toordinal() for d in schedule["ex_date"]], dtype=np.int64)
        sched_amts = schedule["amount"].to_numpy(dtype=float)
    else:
        sched_days = np.empty(0, dtype=np.int64)
        sched_amts = np.empty(0, dtype=float)
    return {
        "kind": kind, "ticker": ticker,
        "expiry_date": expiry, "expiry_ord": expiry.toordinal(), "dte": int(dte),
        "K": float(K), "premium": premium, "delta": delta_net,
        "S_star": float(S_star), "iv": float(iv), "r": float(r), "spot": S,
        "legs": legs, "sched_days": sched_days, "sched_amts": sched_amts,
        "fallback": fallback,
    }
