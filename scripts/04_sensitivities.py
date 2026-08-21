#!/usr/bin/env python
"""PLAN §7 sensitivity grid: one-at-a-time around the base case.

Portfolios: full grid on V0-V3 (15 distinct settings incl. base x 4 variants).
Lanes: reduced grid (spread_mult, iv_mult_adj, delta_target, tenor_target_days)
on SPY, QQQ, AAPL, JNJ, JPM, XOM — long call lane and short put lane, measured
as annualized tracking difference vs the invariant stock-lane baseline.
All runs reuse the engine and one shared MarketData.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from leaps_ls import config  # noqa: E402
from leaps_ls.analysis import metrics  # noqa: E402
from leaps_ls.portfolio import engine as eng_mod, variants  # noqa: E402
from leaps_ls.portfolio.market_data import MarketData  # noqa: E402

BASE = {
    "spread_mult": 1.0,
    "borrow_bps": config.BORROW_BPS_BASE,
    "delta_target": config.DELTA_TARGET,
    "tenor_target_days": config.TENOR_TARGET_DAYS,
    "roll_dte": config.ROLL_DTE_THRESHOLD,
    "iv_mult_adj": 1.0,
    "sizing": config.SIZING,
}
GRID = {
    "spread_mult": config.SENSITIVITIES["spread_mult"],
    "borrow_bps": config.SENSITIVITIES["borrow_bps"],
    "delta_target": config.SENSITIVITIES["delta_target"],
    "tenor_target_days": config.SENSITIVITIES["tenor_target_days"],
    "roll_dte": config.SENSITIVITIES["roll_dte"],
    "iv_mult_adj": config.SENSITIVITIES["iv_mult_adj"],
    "sizing": config.SENSITIVITIES["sizing"],
}
LANE_GRID_KEYS = ["spread_mult", "iv_mult_adj", "delta_target", "tenor_target_days"]
LANE_TICKERS = ["SPY", "QQQ", "AAPL", "JNJ", "JPM", "XOM"]
LANE_KINDS = {"call": ("stock",), "put": ("short_stock",)}  # option lane -> baseline lane


def settings(grid_keys: list[str]) -> list[tuple[str, object]]:
    """("base","base") + every non-base (param, value) once."""
    out = [("base", "base")]
    for p in grid_keys:
        for v in GRID[p]:
            if v != BASE[p]:
                out.append((p, v))
    return out


def rcfg_for(param: str, value) -> eng_mod.RunConfig:
    kw = dict(BASE)
    if param != "base":
        kw[param] = value
    return eng_mod.RunConfig(**kw)


def _costs(res: eng_mod.BacktestResult) -> dict:
    last = res.daily.iloc[-1]
    cap = res.rcfg.capital
    return {
        "spread_pct": last["spread_cum"] / cap, "comm_pct": last["comm_cum"] / cap,
        "borrow_pct": last["borrow_cum"] / cap, "financing_pct": last["financing_cum"] / cap,
        "dividends_pct": last["dividends_cum"] / cap,
    }


def run_portfolio_grid(md: MarketData, plan: pd.DataFrame, end) -> pd.DataFrame:
    rows = []
    todo = settings(list(GRID.keys()))
    for k, (param, val) in enumerate(todo, start=1):
        t0 = time.time()
        rcfg = rcfg_for(param, val)
        for v, (ml, ms) in variants.VARIANT_MODES.items():
            res = eng_mod.run_backtest(plan, ml, ms, md, rcfg, end=end)
            nav = res.daily["nav"] / res.daily["nav"].iloc[0]
            rows.append({"param": param, "value": str(val), "variant": v,
                         **metrics.headline(nav), **_costs(res),
                         "bankruptcies": res.events["bankruptcies"]})
        print(f"  [{k}/{len(todo)}] portfolios {param}={val} ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def run_lane_grid(md: MarketData, end) -> pd.DataFrame:
    # invariant baselines (stock lanes do not depend on the option params)
    base_nav: dict[tuple[str, str], pd.Series] = {}
    for tk in LANE_TICKERS:
        for baseline in ("stock", "short_stock"):
            plan, _ = variants.lane_plan(md, tk, baseline, config.BACKTEST_START, end)
            ml, ms = variants.lane_modes(baseline)
            res = eng_mod.run_backtest(plan, ml, ms, md, eng_mod.RunConfig(), end=end)
            base_nav[(tk, baseline)] = res.daily["nav"] / res.daily["nav"].iloc[0]
    rows = []
    todo = settings(LANE_GRID_KEYS)
    for k, (param, val) in enumerate(todo, start=1):
        t0 = time.time()
        rcfg = rcfg_for(param, val)
        for tk in LANE_TICKERS:
            for lane, (baseline,) in LANE_KINDS.items():
                plan, _ = variants.lane_plan(md, tk, lane, config.BACKTEST_START, end)
                ml, ms = variants.lane_modes(lane)
                res = eng_mod.run_backtest(plan, ml, ms, md, rcfg, end=end)
                nav = res.daily["nav"] / res.daily["nav"].iloc[0]
                td = metrics.ann_tracking_diff(
                    metrics.daily_returns(nav), metrics.daily_returns(base_nav[(tk, baseline)]))
                rows.append({"param": param, "value": str(val), "ticker": tk, "lane": lane,
                             "td_ann": td, "cagr": metrics.cagr(nav),
                             **_costs(res)})
        print(f"  [{k}/{len(todo)}] lanes {param}={val} ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def main() -> int:
    t_start = time.time()
    end = pd.Timestamp.today().normalize()
    print(f"Building MarketData ({len(config.UNIVERSE)} tickers)...")
    md = MarketData(list(config.UNIVERSE), config.DATA_START, end)
    plan = variants.portfolio_plan(md, config.BACKTEST_START, end)
    print(f"  plan: {len(plan)} rebalance dates\n")

    print("Portfolio grid (15 settings x V0-V3):")
    pf = run_portfolio_grid(md, plan, end)
    pf.to_csv(config.RESULTS_DIR / "sensitivities_portfolios.csv", index=False)

    print("\nLane grid (9 settings x 6 tickers x 2 lanes):")
    ln = run_lane_grid(md, end)
    ln.to_csv(config.RESULTS_DIR / "sensitivities_lanes.csv", index=False)

    # headline swings
    print("\nLargest swings (max-min across values, per param):")
    base = pf[pf["param"] == "base"].set_index("variant")
    swings = []
    for param in GRID:
        sub = pf[(pf["param"] == param) | (pf["param"] == "base")]
        g = sub.groupby("variant")["cagr"].agg(lambda s: s.max() - s.min())
        swings.append({"param": param, **{v: g.get(v) for v in base.index}})
    sw = pd.DataFrame(swings)
    print(sw.to_string(index=False, formatters={v: "{:.1%}".format for v in base.index}))
    print(f"\nTotal runtime: {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
