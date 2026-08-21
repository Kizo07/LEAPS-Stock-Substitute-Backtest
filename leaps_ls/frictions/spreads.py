"""Bucketed half-spread model for option trades, calibrated from live chains (PLAN §5.7).

Half-spread per share = max($floor, s_bucket x premium), with s_bucket by |delta|:
itm_deep (|Δ| >= 0.70), itm_shallow (0.50-0.70), atm_otm (< 0.50). Calibrated tiers
(median (ask-bid)/2 / mid over LEAPS contracts with DTE >= 365 across sample tickers)
are persisted to data/calibrated_spreads.json and used when present.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..data import cache, fred
from ..pricing import black_scholes as bs

CAL_PATH = config.DATA_DIR / "calibrated_spreads.json"
BUCKETS = (  # name, |Δ| lower bound (inclusive), |Δ| upper bound (exclusive)
    ("itm_deep", 0.70, np.inf),
    ("itm_shallow", 0.50, 0.70),
    ("atm_otm", 0.0, 0.50),
)
MIN_OBS = 30  # minimum contracts for a calibrated bucket to override the default
MIN_DTE = 365  # LEAPS only, per plan


def bucket_for_delta(abs_delta: float) -> str:
    """Spread bucket name for an absolute delta."""
    a = abs(float(abs_delta))
    for name, lo, hi in BUCKETS:
        if lo <= a < hi:
            return name
    return "atm_otm"


def load_calibration(path: Path | None = None) -> dict | None:
    path = path or CAL_PATH
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def get_spread_tiers(path: Path | None = None) -> dict[str, float]:
    """Active tier fractions: calibrated values when present, else config defaults."""
    tiers = dict(config.SPREAD_TIERS)
    cal = load_calibration(path)
    if cal:
        for name, rec in (cal.get("tiers") or {}).items():
            if name in tiers and rec.get("value") is not None:
                tiers[name] = float(rec["value"])
    return tiers


def get_spread_floor(path: Path | None = None) -> float:
    cal = load_calibration(path)
    if cal and cal.get("floor_usd") is not None:
        return float(cal["floor_usd"])
    return float(config.SPREAD_FLOOR_USD)


def half_spread(
    premium: float, delta: float, tiers: dict[str, float] | None = None,
    floor: float | None = None, mult: float = 1.0,
) -> float:
    """Half-spread in $ per share: max(floor, s_bucket x premium) x stress multiplier."""
    tiers = tiers or get_spread_tiers()
    floor = get_spread_floor() if floor is None else floor
    s = tiers[bucket_for_delta(delta)]
    return float(mult) * max(float(floor), s * float(premium))


def calibrate_from_chains(
    chains: dict[str, pd.DataFrame],
    spots: dict[str, float] | None = None,
    min_dte: int = MIN_DTE,
    min_obs: int = MIN_OBS,
    path: Path | None = None,
) -> dict:
    """Calibrate tier fractions from live chains and persist them.

    ``chains``: {ticker: stacked chain frame from yahoo.fetch_option_chain}.
    Market deltas are approximated from each contract's own IV via Black-Scholes
    (spot = ticker's last cached close, rate = FRED curve at contract tenor).
    Buckets with fewer than ``min_obs`` contracts keep their config default.
    """
    try:
        curve = fred.load_curve()
        as_of = pd.Timestamp.today().normalize()

        def rate_fn(t: float) -> float:
            return curve.rate(as_of, max(t, 1e-6))
    except Exception:  # noqa: BLE001 - offline without FRED cache
        def rate_fn(t: float) -> float:
            return 0.0

    rows: list[pd.DataFrame] = []
    for tk, df in chains.items():
        if spots and tk in spots:
            spot = float(spots[tk])
        else:
            spot = float(cache.load_df(f"hist_{tk}")["Close"].iloc[-1])
        d = df[(df["dte"] >= min_dte) & (df["bid"] > 0) & (df["ask"] > df["bid"])].copy()
        d["mid"] = 0.5 * (d["bid"] + d["ask"])
        d = d[d["mid"] > 0.05]
        if d.empty:
            continue
        T = d["dte"].to_numpy() / 365.0
        r = np.array([rate_fn(t) for t in T])
        iv = d["impliedVolatility"].clip(0.01, 3.0).to_numpy()
        K = d["strike"].to_numpy()
        is_call = (d["type"] == "C").to_numpy()
        d["abs_delta"] = np.where(
            is_call,
            bs.delta(spot, K, T, r, iv, "C"),
            np.abs(bs.delta(spot, K, T, r, iv, "P")),
        )
        d["half_rel"] = ((d["ask"] - d["bid"]) / 2.0) / d["mid"]
        d = d[(d["half_rel"] > 0.0) & (d["half_rel"] < 0.5)]
        d["bucket"] = d["abs_delta"].map(bucket_for_delta)
        d["ticker"] = tk
        rows.append(d[["ticker", "bucket", "half_rel", "abs_delta", "mid"]])
    if not rows:
        raise ValueError("no usable contracts for spread calibration")
    allc = pd.concat(rows, ignore_index=True)

    tiers: dict[str, dict] = {}
    for name, _, _ in BUCKETS:
        sub = allc[allc["bucket"] == name]
        med = float(sub["half_rel"].median()) if len(sub) else None
        use = med is not None and len(sub) >= min_obs
        tiers[name] = {
            "value": med if use else None,  # None -> loader falls back to default
            "median_calibrated": med,
            "n": int(len(sub)),
            "default": config.SPREAD_TIERS[name],
        }
    payload = {
        "calibrated_on": str(pd.Timestamp.today().date()),
        "floor_usd": float(config.SPREAD_FLOOR_USD),
        "min_dte": int(min_dte),
        "min_obs": int(min_obs),
        "n_contracts": int(len(allc)),
        "tickers": sorted(chains.keys()),
        "tiers": tiers,
    }
    out = path or CAL_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    return payload
