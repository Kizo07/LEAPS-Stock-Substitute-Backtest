"""Precomputed market-data bundle for backtest runs.

Loads every ticker's cached history once and precomputes the daily series the
engine and selector need (close, causal EWMA realized vol, total-return index,
VIX, FRED rate curve as fast numpy arrays, realized dividend calendars, and the
calibrated vol/spread parameters). All series are causal: the value at day t
uses only data on or before t (strict no-look-ahead, PLAN §8).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..data import fred, yahoo
from ..frictions import spreads as spreads_mod
from ..pricing import dividends as dvd, vol


class MarketData:
    """Daily series and fast accessors shared by the selector and the engine."""

    def __init__(self, tickers: list[str], start, end):
        self.tickers = list(tickers)
        self.j = {tk: k for k, tk in enumerate(self.tickers)}
        spy = yahoo.load_history("SPY")
        self.days = pd.DatetimeIndex(spy.loc[pd.Timestamp(start): pd.Timestamp(end)].index)
        self.day_ordinals = np.array([d.toordinal() for d in self.days], dtype=np.int64)
        self.ord_of_day = {d: o for d, o in zip(self.days, self.day_ordinals)}

        self.hist_map: dict[str, pd.DataFrame] = {}
        closes, rvs, tris = {}, {}, {}
        for tk in self.tickers:
            h = yahoo.load_history(tk)
            self.hist_map[tk] = h
            closes[tk] = h["Close"].reindex(self.days).ffill()
            rvs[tk] = vol.ewma_realized_vol(h["Close"]).reindex(self.days)
            tris[tk] = yahoo.total_return_series(h).reindex(self.days)
        self.close = pd.DataFrame(closes)
        self.rv = pd.DataFrame(rvs)
        self.tri = pd.DataFrame(tris)
        # cumulative FUTURE split factor vs the historical (as-traded) share basis:
        # split_factor(t) = product of split ratios ex-dated after t. Dollar-denominated
        # frictions (per-share floors/commissions) are quoted in as-traded terms, so the
        # engine divides them by this factor on split-adjusted prices.
        splits = {}
        for tk in self.tickers:
            sf = pd.Series(1.0, index=self.days)
            sp = self.hist_map[tk]["Stock Splits"].fillna(0.0)
            sp = sp[sp > 0.0].sort_index()
            for d, ratio in sp.items():
                sf.loc[sf.index < d] *= float(ratio)
            splits[tk] = sf
        self.split_factor = pd.DataFrame(splits)
        self.split_arr = self.split_factor.to_numpy(dtype=float)
        # numpy fast paths (rows aligned to self.days)
        self.close_arr = self.close.to_numpy(dtype=float)
        self.rv_arr = self.rv.to_numpy(dtype=float)
        vix_h = yahoo.load_history(config.VIX_TICKER)
        self.vix_arr = vix_h["Close"].reindex(self.days).ffill().to_numpy(dtype=float)

        self.curve = fred.load_curve()
        self._rate_dates_ord = np.array([d.toordinal() for d in self.curve.dates], dtype=np.int64)
        self._rate_yields = self.curve.yields
        self._rate_tenors = self.curve.tenors
        i3m = int(np.argmin(np.abs(self._rate_tenors - 0.25)))
        self.short_rate_arr = pd.Series(
            self._rate_yields[:, i3m], index=self.curve.dates
        ).reindex(self.days).ffill().to_numpy(dtype=float)

        cal = vol.load_vol_calibration()
        self._mult = {tk: vol.get_iv_multiplier(tk, cal) for tk in self.tickers}
        self._slope = {tk: vol.get_skew_slope(tk, cal) for tk in self.tickers}
        self._term_adj = {tk: vol.get_term_adj(tk, cal) for tk in self.tickers}
        self._is_index = {tk: tk in config.INDEX_TICKERS_VIX_ANCHOR for tk in self.tickers}

        self.spread_tiers = spreads_mod.get_spread_tiers()
        self.spread_floor = spreads_mod.get_spread_floor()

        self._divs: dict[str, dict[int, float]] = {}
        for tk in self.tickers:
            dd = dvd.realized_dividends(self.hist_map[tk])
            self._divs[tk] = {d.toordinal(): float(a) for d, a in zip(dd["ex_date"], dd["amount"])}

    # ---------------------------------------------------------------- accessors
    def hist(self, ticker: str) -> pd.DataFrame:
        return self.hist_map[ticker]

    def ex_dividend_ord(self, ticker: str, day_ord: int) -> float:
        """Actual per-share dividend ex-dating on ``day_ord`` (0 if none)."""
        return self._divs.get(ticker, {}).get(day_ord, 0.0)

    def rate_ord(self, day_ord: int, T: float) -> float:
        """FRED curve rate (decimal) in force on ``day_ord`` for tenor ``T`` years."""
        i = int(np.searchsorted(self._rate_dates_ord, day_ord, side="right")) - 1
        i = max(i, 0)
        return float(np.interp(float(T), self._rate_tenors, self._rate_yields[i]))

    def iv_atm(self, ticker: str, i: int) -> float:
        """ATM IV proxy at day index ``i``: clamped multiplier x EWMA RV, VIX for index lanes."""
        if self._is_index[ticker]:
            iv = float(self.vix_arr[i]) / 100.0 * self._term_adj[ticker]
        else:
            iv = self._mult[ticker] * float(self.rv_arr[i, self.j[ticker]])
        return min(iv, config.IV_CAP)

    def slope(self, ticker: str) -> float:
        return self._slope[ticker]

    def valid_start(self, ticker: str) -> pd.Timestamp:
        """First calendar day with finite close and realized vol."""
        mask = np.isfinite(self.rv_arr[:, self.j[ticker]]) & np.isfinite(self.close_arr[:, self.j[ticker]])
        return self.days[int(np.argmax(mask))] if mask.any() else self.days[-1]
