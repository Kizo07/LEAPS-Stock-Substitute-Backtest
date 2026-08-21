#!/usr/bin/env python
"""Phase 3: smoke tests (NAV conservation, no-look-ahead) + full backtest run.

(a) Smoke: 3-ticker fixed plan over a 2-year window for every variant mode pair;
    asserts the daily NAV reconciliation identity and that shifting the plan one
    trading day later changes results (no-look-ahead plumbing check).
(b) Full run: single-instrument lanes for every ticker in config.UNIVERSE
    (stock / call / synth_long / short_stock / put / synth_short) and the V0-V3
    momentum portfolios over config.BACKTEST_START -> latest. Writes NAV series,
    ledgers, summaries and portfolio trade logs to results/; prints headline stats.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from leaps_ls import config  # noqa: E402
from leaps_ls.analysis import metrics  # noqa: E402
from leaps_ls.portfolio import engine as eng_mod, variants  # noqa: E402
from leaps_ls.portfolio.market_data import MarketData  # noqa: E402

SMOKE_TICKERS = ["AAPL", "XOM", "JPM"]
SMOKE_START, SMOKE_END = "2010-01-01", "2011-12-30"
RECON_TOL = 1e-4


# ---------------------------------------------------------------------- smoke
def _smoke_plan(md: MarketData) -> pd.DataFrame:
    dates = variants.rebalance_dates(md.days, SMOKE_START, SMOKE_END)
    rows = {d: {"AAPL": 0.5, "XOM": 0.1, "JPM": -0.6} for d in dates}
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()


def smoke_test(md: MarketData) -> None:
    print("=" * 78)
    print("SMOKE TESTS — NAV conservation (3 tickers, 2010-2011) + no-look-ahead shift")
    print("=" * 78)
    plan = _smoke_plan(md)
    for v, (ml, ms) in variants.VARIANT_MODES.items():
        rcfg = eng_mod.RunConfig(check_recon=True)
        res = eng_mod.run_backtest(plan, ml, ms, md, rcfg, end=SMOKE_END)
        nav = res.daily["nav"]
        err = float(np.nanmax(np.abs(res.daily["recon_err"].to_numpy())))
        assert np.isfinite(nav).all() and (nav > 0).all(), f"{v}: bad NAV path"
        assert err < RECON_TOL, f"{v}: recon err {err}"
        print(f"  {v} ({ml}/{ms}): recon max|err|={err:.2e}  final NAV={nav.iloc[-1]:,.0f}  "
              f"events={res.events}")
    # no-look-ahead: shifting the plan one trading day later must change results
    base = eng_mod.run_backtest(plan, "call", "put", md, eng_mod.RunConfig(), end=SMOKE_END)
    shifted_index = [md.days[min(int(md.days.get_loc(d)) + 1, len(md.days) - 1)] for d in plan.index]
    plan_shift = plan.copy()
    plan_shift.index = pd.DatetimeIndex(shifted_index)
    moved = eng_mod.run_backtest(plan_shift, "call", "put", md, eng_mod.RunConfig(), end=SMOKE_END)
    diff = float((base.daily["nav"] - moved.daily["nav"]).abs().max())
    assert diff > 1e-6, "shift test failed: results identical after shifting signals"
    print(f"  shift test: max |NAV diff| after +1 day signal shift = {diff:,.4f} (> 0)")
    print("SMOKE: PASS\n")


# ---------------------------------------------------------------------- summaries
def _cost_row(res: eng_mod.BacktestResult) -> dict:
    last = res.daily.iloc[-1]
    cap = res.rcfg.capital
    return {
        "spread_pct": last["spread_cum"] / cap,
        "comm_pct": last["comm_cum"] / cap,
        "borrow_pct": last["borrow_cum"] / cap,
        "financing_pct": last["financing_cum"] / cap,
        "dividends_pct": last["dividends_cum"] / cap,
    }


def _norm_nav(res: eng_mod.BacktestResult) -> pd.Series:
    nav = res.daily["nav"]
    return nav / nav.iloc[0]


# ---------------------------------------------------------------------- full run
def run_lanes(md: MarketData, end) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    navs: dict[str, pd.Series] = {}
    ledgers: dict[str, pd.Series] = {}
    rows: list[dict] = []
    tickers = list(config.UNIVERSE)
    for n, tk in enumerate(tickers, start=1):
        t0 = time.time()
        lane_nav: dict[str, pd.Series] = {}
        for lane in variants.LANES:
            plan, mode = variants.lane_plan(md, tk, lane, config.BACKTEST_START, end)
            if plan.empty:
                continue
            ml, ms = variants.lane_modes(lane)
            res = eng_mod.run_backtest(plan, ml, ms, md, eng_mod.RunConfig(), end=end)
            nav = _norm_nav(res)
            lane_nav[lane] = nav
            navs[f"{tk}:{lane}"] = nav
            for col in ("spread_cum", "comm_cum", "borrow_cum", "financing_cum", "dividends_cum"):
                ledgers[f"{tk}:{lane}:{col}"] = res.daily[col]
            base_lane = "stock" if variants.LANES[lane][0] > 0 else "short_stock"
            row = {"ticker": tk, "lane": lane, **metrics.headline(nav),
                   **_cost_row(res), **{f"ev_{k}": v for k, v in res.events.items()}}
            row["track_diff_ann"] = (
                metrics.ann_tracking_diff(metrics.daily_returns(nav),
                                          metrics.daily_returns(lane_nav[base_lane]))
                if lane != base_lane and base_lane in lane_nav else 0.0)
            rows.append(row)
        print(f"  [{n}/{len(tickers)}] {tk} lanes done in {time.time() - t0:.1f}s")
    nav_df = pd.DataFrame(navs)
    return nav_df, pd.DataFrame(ledgers), pd.DataFrame(rows)


def run_portfolios(md: MarketData, end) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    plan = variants.portfolio_plan(md, config.BACKTEST_START, end)
    print(f"  momentum plan: {len(plan)} rebalance dates x {int((plan != 0).any(axis=0).sum())} tickers")
    navs, ledgers, rows, trades = {}, {}, [], {}
    for v, (ml, ms) in variants.VARIANT_MODES.items():
        t0 = time.time()
        res = eng_mod.run_backtest(plan, ml, ms, md, eng_mod.RunConfig(), end=end)
        nav = _norm_nav(res)
        navs[v] = nav
        d = res.daily
        for col in ("spread_cum", "comm_cum", "borrow_cum", "financing_cum", "dividends_cum"):
            ledgers[f"{v}:{col}"] = d[col]
        rows.append({"variant": v, "modes": f"{ml}/{ms}", **metrics.headline(nav),
                     **_cost_row(res), **{f"ev_{k}": val for k, val in res.events.items()}})
        trades[v] = res.trades
        print(f"  {v} ({ml}/{ms}) done in {time.time() - t0:.1f}s: "
              f"CAGR {rows[-1]['cagr']:.2%}, Sharpe {rows[-1]['sharpe']:.2f}")
    return pd.DataFrame(navs), pd.DataFrame(ledgers), pd.DataFrame(rows), trades


# ---------------------------------------------------------------------- main
def main() -> int:
    t_start = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.today().normalize()
    print(f"Building MarketData ({len(config.UNIVERSE)} tickers)...")
    md = MarketData(list(config.UNIVERSE), config.DATA_START, end)
    print(f"  calendar: {len(md.days)} trading days "
          f"{md.days[0].date()} -> {md.days[-1].date()}\n")

    smoke_test(md)

    print("=" * 78)
    print("FULL RUN — single-instrument lanes")
    print("=" * 78)
    lane_navs, lane_ledgers, lane_summary = run_lanes(md, end)
    lane_navs.to_parquet(config.RESULTS_DIR / "nav_lanes.parquet")
    lane_ledgers.to_parquet(config.RESULTS_DIR / "lane_ledgers.parquet")
    lane_summary.to_csv(config.RESULTS_DIR / "lane_summary.csv", index=False)

    print("\n" + "=" * 78)
    print("FULL RUN — V0-V3 momentum portfolios")
    print("=" * 78)
    pf_navs, pf_ledgers, pf_summary, pf_trades = run_portfolios(md, end)
    pf_navs.to_parquet(config.RESULTS_DIR / "nav_portfolios.parquet")
    pf_ledgers.to_parquet(config.RESULTS_DIR / "portfolio_ledgers.parquet")
    pf_summary.to_csv(config.RESULTS_DIR / "portfolio_summary.csv", index=False)
    for v, tr in pf_trades.items():
        tr.to_csv(config.RESULTS_DIR / f"trades_{v}.csv", index=False)

    # ---------------- headline prints
    print("\n" + "=" * 78)
    print("HEADLINE — SPY lanes (normalized NAV, tracking vs stock lane, annualized)")
    print("=" * 78)
    spy = lane_summary[lane_summary["ticker"] == "SPY"].set_index("lane")
    for lane in ("stock", "call", "synth_long"):
        r = spy.loc[lane]
        print(f"  {lane:<12} CAGR {r['cagr']:>7.2%}  vol {r['vol']:>6.2%}  "
              f"Sharpe {r['sharpe']:>5.2f}  maxDD {r['maxdd']:>7.2%}  "
              f"track {r['track_diff_ann']:>+7.2%}")
    for lane in ("short_stock", "put", "synth_short"):
        r = spy.loc[lane]
        print(f"  {lane:<12} CAGR {r['cagr']:>7.2%}  vol {r['vol']:>6.2%}  "
              f"Sharpe {r['sharpe']:>5.2f}  maxDD {r['maxdd']:>7.2%}  "
              f"track {r['track_diff_ann']:>+7.2%}")

    print("\n" + "=" * 78)
    print("HEADLINE — V0-V3 portfolios")
    print("=" * 78)
    cols = ["variant", "cagr", "vol", "sharpe", "maxdd", "spread_pct", "comm_pct",
            "borrow_pct", "financing_pct", "dividends_pct"]
    print(pf_summary[cols].to_string(index=False,
          formatters={c: "{:.2%}".format for c in cols if c not in ("variant", "sharpe")}
          | {"sharpe": "{:.2f}".format}))
    ev_cols = ["variant", "ev_rolls", "ev_call_exercise_reopens", "ev_put_parity_reopens",
               "ev_expiry_settlements"]
    print("\nEvent counts:")
    print(pf_summary[ev_cols].to_string(index=False))
    ev_num = [c for c in lane_summary.columns if c.startswith("ev_") and c != "ev_bankruptcy_date"]
    lane_events = lane_summary[ev_num].sum()
    print("\nLane event totals across all tickers:")
    print(lane_events.to_string())
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
