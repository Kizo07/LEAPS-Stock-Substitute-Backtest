"""Volatility model (PLAN §5.2): EWMA realized vol -> IV proxy, live-chain
calibration of the per-name IV/RV multiplier and skew slope, VIX anchor for index lanes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..data import cache

CAL_PATH = config.DATA_DIR / "calibrated_vol.json"
CAL_DTE = (300, 800)  # calibration window, days to expiry
NEAR_ATM_LOGMONEY = 0.10  # |ln(K/S*)| treated as near-ATM for the multiplier
SLOPE_MAX_LOGMONEY = 0.60  # regression support for the skew slope
# |slope| bound so that the skew adjustment at |ln(K/S*)| = 0.5 cannot exceed SKEW_CLAMP
SLOPE_BOUND = config.SKEW_CLAMP / 0.5
MIN_OBS_MULT = 8
MIN_OBS_SLOPE = 20


# ------------------------------------------------------------------ realized vol
def ewma_realized_vol(
    close: pd.Series,
    lam: float = config.EWMA_LAMBDA,
    min_window: int = config.EWMA_MIN_WINDOW,
) -> pd.Series:
    """Annualized EWMA (RiskMetrics) volatility of daily log close-to-close returns."""
    logret = np.log(close.astype(float)).diff()
    var = logret.pow(2).ewm(alpha=1.0 - lam, min_periods=min_window, adjust=False).mean()
    return np.sqrt(var * 252.0).rename("rv")


def rv_as_of(hist: pd.DataFrame, date, **ewma_kw) -> float:
    """EWMA realized vol using only closes on or before ``date`` (no look-ahead)."""
    close = hist.loc[: pd.Timestamp(date), "Close"]
    rv = ewma_realized_vol(close, **ewma_kw)
    return float(rv.iloc[-1])


# ------------------------------------------------------------------ calibration io
def load_vol_calibration(path: Path | None = None) -> dict:
    """Calibrated multipliers/slopes; empty structure if not calibrated yet."""
    path = path or CAL_PATH
    if not path.exists():
        return {"tickers": {}, "default": {"mult": None, "slope": None}}
    with open(path) as fh:
        return json.load(fh)


def save_vol_calibration(cal: dict, path: Path | None = None) -> Path:
    path = path or CAL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(cal, fh, indent=2, default=float)
    return path


def clamp_mult(m: float) -> float:
    """Clamp the IV/RV multiplier to the config bounds [1.00, 1.35]."""
    return float(np.clip(m, config.IV_MULT_LO, config.IV_MULT_HI))


def clamp_slope(s: float) -> float:
    return float(np.clip(s, -SLOPE_BOUND, SLOPE_BOUND))


def _default_mult(cal: dict) -> float:
    m = (cal.get("default") or {}).get("mult")
    return float(m) if m else 1.0


def _default_slope(cal: dict) -> float:
    s = (cal.get("default") or {}).get("slope")
    return float(s) if s is not None else 0.0


def get_iv_multiplier(ticker: str, cal: dict | None = None) -> float:
    """Per-name clamped IV/RV multiplier; cross-sectional median default."""
    cal = cal if cal is not None else load_vol_calibration()
    rec = (cal.get("tickers") or {}).get(ticker) or {}
    m = rec.get("mult")
    return clamp_mult(m) if m else _default_mult(cal)


def get_skew_slope(ticker: str, cal: dict | None = None) -> float:
    """Per-name skew slope (dIV per unit ln(K/S*)); cross-sectional median default."""
    cal = cal if cal is not None else load_vol_calibration()
    rec = (cal.get("tickers") or {}).get(ticker) or {}
    s = rec.get("slope")
    return clamp_slope(s) if s is not None else _default_slope(cal)


def get_term_adj(ticker: str, cal: dict | None = None) -> float:
    """VIX term adjustment for index lanes (default 1.0, calibrated in [0.9, 1.1])."""
    cal = cal if cal is not None else load_vol_calibration()
    rec = (cal.get("tickers") or {}).get(ticker) or {}
    return float(rec.get("term_adj") or 1.0)


# ------------------------------------------------------------------ IV proxy
def _vix_close(date, vix: pd.Series | None = None) -> float:
    if vix is None:
        vix = cache.load_df("hist_VIX")["Close"]
    return float(vix.loc[: pd.Timestamp(date)].iloc[-1])


def iv_proxy(
    date, ticker: str, hist: pd.DataFrame | None = None, cal: dict | None = None,
    vix: pd.Series | None = None,
) -> float:
    """ATM IV proxy as of ``date``: clamp(m_name) x EWMA RV; VIX anchor for index lanes."""
    if ticker in config.INDEX_TICKERS_VIX_ANCHOR:
        return min(_vix_close(date, vix) / 100.0 * get_term_adj(ticker, cal), config.IV_CAP)
    if hist is None:
        hist = cache.load_df(f"hist_{ticker}")
    rv = rv_as_of(hist, date)
    if not np.isfinite(rv) or rv <= 0.0:
        return float("nan")
    return min(get_iv_multiplier(ticker, cal) * rv, config.IV_CAP)


def apply_skew(iv_atm, K, S_star, slope: float):
    """IV(K) = IV_ATM + slope x ln(K/S*), total adjustment clamped to ±SKEW_CLAMP."""
    adj = np.clip(
        slope * np.log(np.asarray(K, dtype=float) / np.asarray(S_star, dtype=float)),
        -config.SKEW_CLAMP,
        config.SKEW_CLAMP,
    )
    return np.asarray(iv_atm, dtype=float) + adj


def implied_vol(
    date, ticker: str, K, S_star: float, hist: pd.DataFrame | None = None,
    cal: dict | None = None, vix: pd.Series | None = None,
):
    """Full IV proxy for strike(s) ``K``: ATM proxy plus clamped linear skew."""
    cal = cal if cal is not None else load_vol_calibration()
    atm = iv_proxy(date, ticker, hist=hist, cal=cal, vix=vix)
    return apply_skew(atm, K, S_star, get_skew_slope(ticker, cal))


# ------------------------------------------------------------------ calibration from a live chain
def calibrate_from_chain(
    ticker: str,
    chain: pd.DataFrame,
    rv: float,
    vix_now: float | None = None,
) -> dict:
    """Calibrate IV/RV multiplier (or VIX term_adj for index lanes) and skew slope.

    ``chain`` must have columns: strike, dte, s_star and a market-IV column —
    ``iv_mkt`` (BS-implied from the market mid) when available, else Yahoo's
    ``impliedVolatility``. Uses contracts with 300-800 DTE; multiplier = median
    market IV over near-ATM contracts / current EWMA RV; slope = OLS of market IV
    on ln(K/S*). Records fall back to ``None`` (i.e. cross-sectional default) when thin.
    """
    iv_col = "iv_mkt" if "iv_mkt" in chain.columns else "impliedVolatility"
    df = chain[(chain["dte"] >= CAL_DTE[0]) & (chain["dte"] <= CAL_DTE[1])].copy()
    df = df.rename(columns={iv_col: "mkt_iv"})
    df = df[(df["mkt_iv"] > 0.005) & (df["mkt_iv"] < 3.0)]
    df["log_money"] = np.log(df["strike"] / df["s_star"])
    rec: dict = {"n_cal": int(len(df)), "rv": float(rv) if np.isfinite(rv) else None}

    near = df[df["log_money"].abs() <= NEAR_ATM_LOGMONEY]
    med_iv = float(near["mkt_iv"].median()) if len(near) else float("nan")
    rec["median_mkt_iv"] = med_iv if np.isfinite(med_iv) else None
    rec["n_near_atm"] = int(len(near))

    if ticker in config.INDEX_TICKERS_VIX_ANCHOR:
        if np.isfinite(med_iv) and vix_now and vix_now > 0 and len(near) >= MIN_OBS_MULT:
            rec["term_adj"] = float(np.clip(med_iv / (vix_now / 100.0), 0.9, 1.1))
        else:
            rec["term_adj"] = None
        rec["anchor"] = "vix"
    else:
        ok = np.isfinite(med_iv) and np.isfinite(rv) and rv > 0 and len(near) >= MIN_OBS_MULT
        rec["mult_raw"] = float(med_iv / rv) if ok else None
        rec["mult"] = clamp_mult(med_iv / rv) if ok else None

    wide = df[df["log_money"].abs() <= SLOPE_MAX_LOGMONEY]
    if len(wide) >= MIN_OBS_SLOPE and wide["log_money"].std() > 1e-6:
        slope = float(np.polyfit(wide["log_money"], wide["mkt_iv"], 1)[0])
        rec["slope_raw"] = slope
        rec["slope"] = clamp_slope(slope)
    else:
        rec["slope_raw"] = None
        rec["slope"] = None
    return rec
