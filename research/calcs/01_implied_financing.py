"""Effective financing rate embedded in quoted LEAPS — vol-free parity estimator.

WHY THIS ESTIMATOR
------------------
Put-call parity with discrete dividends and an *effective* financing rate f:

        C - P = (S - PV(D)) - K e^{-f T}

The time value of a call and a put at the same strike is IDENTICAL, so
differencing removes it.  The estimator is therefore completely free of any
volatility or vol-surface assumption — the single biggest source of error in
the usual approach.  The one remaining term is the American early-exercise
premium on the call (dividend capture), which is removed explicitly:

        f = -(1/T) ln( (S - PV(D) - (C - eep_call - P)) / K )

Validity: requires the put to be deep OTM (so its own American premium is
zero).  We enforce that contract-by-contract with a binomial check.

VALIDATION (see 01_validate_estimator.py): on synthetic chains built from a
known model this estimator recovers the true f to < 1 bp for every strike and
every volatility from 15% to 70%.  The un-adjusted version is biased by
+30 to +137 bps on SPY-like dividends, i.e. the correction matters.

f - r is the ALL-IN cost of the leverage embedded in the LEAPS: dealer funding
spread + balance-sheet charge + whatever the market charges for the forward.
The ask-side variant adds the cost of crossing the quote.

Outputs: research/results/financing_leaps.csv, financing_summary.csv
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import binomial as bn

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "JPM", "XOM", "KO"]
AS_OF = pd.Timestamp("2026-07-31")
DTE_LO, DTE_HI = 300, 900
MONEY_LO, MONEY_HI = 0.20, 0.80      # K/S : deep-ITM calls only
MIN_OI, MAX_SPREAD_BPS = 5, 2500
EE_VOLS = (0.15, 0.30, 0.50)         # robustness grid for the eep correction
NSTEPS = 800


def load_rates() -> dict[str, float]:
    """FRED DGS* are stored as DECIMALS (0.0426 = 4.26%), not percent.

    Guard anyway: if a value looks like it is in percent, rescale, and hard-fail
    outside a plausible range.  An earlier version divided by 100 a second time,
    which silently shifted every financing-spread estimate by ~390 bps.
    """
    f = pd.read_parquet(DATA / "fred_curve.parquet").dropna(how="all").ffill()
    row = f.loc[:AS_OF].iloc[-1].astype(float)
    out = {}
    for k, v in row.items():
        if v > 1.5:          # clearly percent-encoded
            v = v / 100.0
        if not 0.0 < v < 0.25:
            raise ValueError(f"implausible rate for {k}: {v}")
        out[k] = float(v)
    return out


def r_cont(T: float, rates) -> float:
    x = [0.0, 1.0, 2.0]
    y = [rates["DGS3MO"], rates["DGS1"], rates["DGS2"]]
    rq = float(np.interp(T, x, y)) if T <= 2.0 else rates["DGS2"]
    return math.log1p(rq)


def div_info(tkr: str):
    h = pd.read_parquet(DATA / f"hist_{tkr}.parquet").loc[:AS_OF]
    d = h["Dividends"]
    d = d[d > 0]
    last4 = d.tail(4)
    ann = float(last4.sum()) if len(last4) else 0.0
    return ann, float(h["Close"].iloc[-1])


def sched_and_pv(ann: float, T: float, r: float, n: int = 4):
    if ann <= 0:
        return [], 0.0
    step = 1.0 / n
    s = [(k * step, ann / n) for k in range(1, int(T / step) + 1) if k * step <= T + 1e-9]
    return s, sum(d * math.exp(-r * t) for t, d in s)


def main() -> None:
    rates = load_rates()
    print("Treasury quoted on %s: %s" % (AS_OF.date(),
          {k: round(v * 100, 2) for k, v in rates.items()}))

    rows = []
    for tkr in TICKERS:
        ch = pd.read_parquet(DATA / f"chain_{tkr}.parquet")
        ch = ch[(ch.dte >= DTE_LO) & (ch.dte <= DTE_HI)].copy()
        S = float(ch.underlying_price.iloc[0])
        ann_div, S_close = div_info(tkr)

        ok = (ch.bid > 0) & (ch.ask > 0) & (ch.ask >= ch.bid)
        ch = ch[ok]
        ch["mid"] = 0.5 * (ch.bid + ch.ask)
        ch["spread_bps"] = 1e4 * (ch.ask - ch.bid) / ch.mid

        cs = ch[ch.type == "C"].set_index(["expiry", "strike"]).sort_index()
        ps = ch[ch.type == "P"].set_index(["expiry", "strike"]).sort_index()
        common = cs.index.intersection(ps.index)

        for expiry, K in common:
            c, p = cs.loc[(expiry, K)], ps.loc[(expiry, K)]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[0]
            if isinstance(p, pd.DataFrame):
                p = p.iloc[0]
            T = float(c.dte) / 365.25
            r = r_cont(T, rates)
            sched, pvD = sched_and_pv(ann_div, T, r)
            mny = K / S
            if not (MONEY_LO <= mny <= MONEY_HI):
                continue
            if c.spread_bps > MAX_SPREAD_BPS or c.openInterest < MIN_OI:
                continue

            cmid, pmid = float(c.mid), float(p.mid)

            # --- early-exercise premia, robust across a wide vol grid -------
            eep_c = np.mean([bn.crr_american(S, K, T, r, v, sched, NSTEPS, "C")[2] for v in EE_VOLS])
            eep_p = np.mean([bn.crr_american(S, K, T, r, v, sched, NSTEPS, "P")[2] for v in EE_VOLS])
            if eep_p > 0.005 * S:            # put is NOT deep OTM -> parity invalid
                continue

            def f_of(cp_leg: float, put_leg: float):
                den = S - pvD - (cp_leg - eep_c - put_leg)
                return -math.log(den / K) / T if 0 < den < K else float("nan")

            f_mid = f_of(cmid, pmid)
            f_ask = f_of(float(c.ask), float(p.bid))     # all-in: pay offer, hit bid
            if not np.isfinite(f_mid):
                continue

            # bid-ask induced uncertainty: df/d(cash) = 1 / (K T e^{-fT})
            KTe = K * T * math.exp(-f_mid * T)
            ba_cash = 0.5 * (float(c.ask) - float(c.bid)) + 0.5 * (float(p.ask) - float(p.bid))
            f_err_bps = 1e4 * ba_cash / KTe if KTe > 0 else float("nan")

            rows.append(dict(
                ticker=tkr, expiry=str(pd.Timestamp(expiry).date()), dte=int(round(T * 365.25)),
                T=round(T, 4), strike=float(K), moneyness=round(mny, 4), S=S, r=round(r, 5),
                ann_div=round(ann_div, 4), q_yield=round(ann_div / S_close, 5), pv_div=round(pvD, 4),
                c_bid=float(c.bid), c_ask=float(c.ask), c_mid=cmid,
                p_bid=float(p.bid), p_ask=float(p.ask), p_mid=pmid,
                c_spread_bps=round(float(c.spread_bps), 1), c_oi=float(c.openInterest),
                eep_call=round(eep_c, 4), eep_put=round(eep_p, 4),
                f_mid=f_mid, f_ask=f_ask,
                fs_bps=1e4 * (f_mid - r), fs_ask_bps=1e4 * (f_ask - r) if np.isfinite(f_ask) else np.nan,
                fs_err_bps=f_err_bps,
            ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "financing_leaps.csv", index=False)
    print(f"\nwrote financing_leaps.csv  ({len(df)} usable deep-ITM call/put pairs)")

    print("\n=== embedded financing spread over Treasuries (bps/yr) ===")
    print("    mid = mid-quote entry | ask = crossing the offer | err = +-1 half-spread\n")
    g = df.groupby("ticker").agg(
        n=("fs_bps", "size"), dte=("dte", "median"), mny=("moneyness", "median"),
        r_pct=("r", lambda s: round(100 * s.median(), 2)),
        q_pct=("q_yield", lambda s: round(100 * s.median(), 2)),
        fs_mid=("fs_bps", "median"), fs_iqr=("fs_bps", lambda s: s.quantile(.75) - s.quantile(.25)),
        fs_ask=("fs_ask_bps", "median"),
        err=("fs_err_bps", "median"),
        c_spread_bps=("c_spread_bps", "median"),
        eep_call_pct_mid=("eep_call", lambda s: round(100 * (s / df.loc[s.index, "c_mid"]).median(), 2)),
    ).round(1)
    print(g.to_string())

    print("\nPOOLED: fs_mid = %.0f bps | fs_ask = %.0f bps | +-half-spread error = %.0f bps"
          % (df.fs_bps.median(), df.fs_ask_bps.median(), df.fs_err_bps.median()))
    print("POOLED ex-SPY: fs_mid = %.0f bps" % df[df.ticker != "SPY"].fs_bps.median())
    print("SPY only    : fs_mid = %.0f bps" % df[df.ticker == "SPY"].fs_bps.median())

    print("\n=== by moneyness (pooled) ===")
    df["bucket"] = pd.cut(df.moneyness, [0.2, 0.35, 0.5, 0.65, 0.8],
                          labels=["0.20-0.35", "0.35-0.50", "0.50-0.65", "0.65-0.80"])
    print(df.groupby("bucket", observed=True).agg(
        n=("fs_bps", "size"), fs_mid=("fs_bps", "median"), fs_ask=("fs_ask_bps", "median"),
        err=("fs_err_bps", "median"), c_spread_bps=("c_spread_bps", "median"), oi=("c_oi", "median"),
    ).round(0).to_string())

    g.to_csv(OUT / "financing_summary.csv")
    print("\nwrote financing_summary.csv")


if __name__ == "__main__":
    main()
