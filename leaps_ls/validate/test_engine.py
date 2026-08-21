"""Offline engine tests on a synthetic, deterministic MarketData (no network/disk).

Run directly:  python leaps_ls/validate/test_engine.py
Or via pytest: python -m pytest leaps_ls/validate -q
"""
from __future__ import annotations

import contextlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from leaps_ls import config  # noqa: E402
from leaps_ls.data import fred, yahoo  # noqa: E402
from leaps_ls.frictions import spreads as spreads_mod  # noqa: E402
from leaps_ls.pricing import dividends as dvd, vol  # noqa: E402
from leaps_ls.portfolio import engine as eng_mod, signals, variants  # noqa: E402
from leaps_ls.portfolio.market_data import MarketData  # noqa: E402

TICKERS = ["AAA", "BBB", "CCC"]
IDX = pd.bdate_range("2007-01-03", "2011-06-30")
SPLIT_DATE = IDX[len(IDX) // 2]


def _gbm(rng: np.random.Generator, s0: float, mu: float, sig: float) -> np.ndarray:
    n = len(IDX)
    steps = rng.normal(mu / 252.0 - sig * sig / (2.0 * 252.0), sig / np.sqrt(252.0), n)
    return s0 * np.exp(np.cumsum(steps))


def _hist(close: np.ndarray, divs: pd.Series | None = None,
          splits: pd.Series | None = None) -> pd.DataFrame:
    d = pd.Series(0.0, index=IDX)
    if divs is not None:
        d.loc[divs.index] = divs
    s = pd.Series(0.0, index=IDX)
    if splits is not None:
        s.loc[splits.index] = splits
    return pd.DataFrame({"Close": close, "Dividends": d, "Stock Splits": s}, index=IDX)


def _synthetic_histories() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    aaa_div_dates = IDX[40::63]
    bbb_div_dates = IDX[55::126]
    return {
        "SPY": _hist(_gbm(rng, 120.0, 0.05, 0.12)),
        "^VIX": pd.DataFrame({"Close": pd.Series(20.0, index=IDX),
                              "Dividends": 0.0, "Stock Splits": 0.0}, index=IDX),
        "AAA": _hist(_gbm(rng, 100.0, 0.06, 0.15),
                     divs=pd.Series(0.25, index=aaa_div_dates)),
        "BBB": _hist(_gbm(rng, 80.0, -0.04, 0.30),
                     divs=pd.Series(1.00, index=bbb_div_dates)),
        "CCC": _hist(_gbm(rng, 50.0, 0.08, 0.25),
                     splits=pd.Series(2.0, index=[SPLIT_DATE])),
        "DDD": _hist(_gbm(rng, 60.0, 0.10, 0.20),
                     divs=pd.Series(0.40, index=IDX[30::63])),
        "EEE": _hist(_gbm(rng, 40.0, 0.02, 0.35)),
    }


@contextlib.contextmanager
def _data_env():
    """Swap loaders for synthetic deterministic ones; restore on exit."""
    hists = _synthetic_histories()

    def fake_load_history(ticker: str) -> pd.DataFrame:
        return hists[ticker]

    curve_df = pd.DataFrame({"DGS3MO": 0.03, "DGS1": 0.035, "DGS2": 0.04},
                            index=[IDX[0], IDX[-1]])
    curve = fred.RateCurve(curve_df)
    saved = (
        (yahoo, "load_history", yahoo.load_history),
        (fred, "load_curve", fred.load_curve),
        (vol, "load_vol_calibration", vol.load_vol_calibration),
        (spreads_mod, "get_spread_tiers", spreads_mod.get_spread_tiers),
        (spreads_mod, "get_spread_floor", spreads_mod.get_spread_floor),
    )
    yahoo.load_history = fake_load_history
    fred.load_curve = lambda: curve
    vol.load_vol_calibration = lambda: {}
    spreads_mod.get_spread_tiers = lambda: dict(config.SPREAD_TIERS)
    spreads_mod.get_spread_floor = lambda: config.SPREAD_FLOOR_USD
    try:
        yield hists
    finally:
        for mod, name, fn in saved:
            setattr(mod, name, fn)


def make_md() -> MarketData:
    md = MarketData(TICKERS, IDX[0], IDX[-1])
    return md


def _month_starts(md: MarketData, start: str, end: str) -> list[pd.Timestamp]:
    return variants.rebalance_dates(md.days, start, end)


# ------------------------------------------------------------------ fixture sanity
def test_synth_md_construction() -> None:
    with _data_env():
        md = make_md()
        j = md.j["CCC"]
        pre = int(np.argmax(md.days >= SPLIT_DATE)) - 1
        assert float(md.split_arr[pre, j]) == 2.0
        assert float(md.split_arr[-1, j]) == 1.0
        assert float(md.split_arr[-1, md.j["AAA"]]) == 1.0
        iv = md.iv_atm("BBB", 400)
        assert 0.01 < iv <= config.IV_CAP + 1e-12
        assert np.isfinite(md.short_rate_arr).all()
        assert md.rate_ord(int(md.day_ordinals[300]), 1.5) > 0


# ------------------------------------------------------------------ full runs
def test_engine_recon_and_events_all_variants() -> None:
    with _data_env():
        md = make_md()
        dates = _month_starts(md, "2008-01-01", "2010-06-01")
        plan = pd.DataFrame.from_dict(
            {d: {"AAA": 0.5, "BBB": -0.5} for d in dates}, orient="index"
        ).fillna(0.0).sort_index()
        for v, (ml, ms) in variants.VARIANT_MODES.items():
            res = eng_mod.run_backtest(plan, ml, ms, md,
                                       eng_mod.RunConfig(check_recon=True),
                                       end=md.days[-1])
            err = float(np.nanmax(np.abs(res.daily["recon_err"].to_numpy())))
            nav = res.daily["nav"]
            assert err < 1e-6, (v, err)
            assert np.isfinite(nav).all() and (nav > 0).all(), v
            if ml != "stock":
                assert res.events["option_trades"] > 0, v


def test_shift_plan_changes_results() -> None:
    with _data_env():
        md = make_md()
        dates = _month_starts(md, "2008-01-01", "2009-12-31")
        plan = pd.DataFrame.from_dict(
            {d: {"AAA": 1.0} for d in dates}, orient="index").fillna(0.0).sort_index()
        base = eng_mod.run_backtest(plan, "call", "put", md, eng_mod.RunConfig(),
                                    end=md.days[-1])
        shifted = plan.copy()
        pos = [int(md.days.get_loc(d)) for d in plan.index]
        shifted.index = pd.DatetimeIndex([md.days[min(p + 1, len(md.days) - 1)] for p in pos])
        moved = eng_mod.run_backtest(shifted, "call", "put", md, eng_mod.RunConfig(),
                                     end=md.days[-1])
        diff = float((base.daily["nav"] - moved.daily["nav"]).abs().max())
        assert diff > 1e-6, diff


# ------------------------------------------------------------------ event mechanics
def _open_manual_group(md: MarketData, tk: str, kind: str, K: float, i: int,
                       contracts: float = 10.0) -> tuple[eng_mod.Engine, dict]:
    mode_long, mode_short = variants.lane_modes(
        {"call": "call", "synth_long": "synth_long", "put": "put"}[kind])
    eng = eng_mod.Engine(pd.DataFrame({tk: [1.0]}, index=[md.days[i]]), mode_long,
                         mode_short, md, eng_mod.RunConfig())
    t_ord = int(md.day_ordinals[i])
    sched = dvd.project_dividends(md.hist(tk), md.days[i], horizon_years=2.0)
    if len(sched):
        sd = np.array([d.toordinal() for d in sched["ex_date"]], dtype=np.int64)
        sa = sched["amount"].to_numpy(dtype=float)
    else:
        sd = np.empty(0, dtype=np.int64)
        sa = np.empty(0, dtype=float)
    pos = eng._open_group(tk, kind, K, t_ord + 730, contracts, i, "open", (sd, sa))
    return eng, pos


def test_call_exercise_reopen_mechanics() -> None:
    with _data_env():
        md = make_md()
        i = 500
        S = float(md.close_arr[i, md.j["AAA"]])
        eng, pos = _open_manual_group(md, "AAA", "call", K=0.70 * S, i=i)
        tom_ord = int(md.day_ordinals[i + 1])
        eng.md._divs["AAA"][tom_ord] = 6.0
        eng._mark_all(i, int(md.day_ordinals[i]))
        eng._exercise_checks(i, int(md.day_ordinals[i]))
        assert "AAA" not in eng.opt
        assert len(eng.pending_reopen) == 1 and eng.pending_reopen[0]["ticker"] == "AAA"
        assert eng.events["call_exercise_reopens"] == 1
        assert eng.trade_rows[-1]["action"] == "close_exercise_reopen"


def test_put_parity_reopen_mechanics() -> None:
    with _data_env():
        md = make_md()
        i = 500
        S = float(md.close_arr[i, md.j["BBB"]])
        eng, pos = _open_manual_group(md, "BBB", "put", K=2.0 * S, i=i)
        eng._mark_all(i, int(md.day_ordinals[i]))
        px = pos["last_legs"][0]["px"]
        assert exercise_violates(px, pos["K"], pos["last_S_star"])
        eng._parity_checks(i, int(md.day_ordinals[i]))
        assert "BBB" not in eng.opt
        assert eng.events["put_parity_reopens"] == 1
        assert eng.trade_rows[-1]["action"] == "close_parity_reopen"


def exercise_violates(px: float, K: float, s_star: float) -> bool:
    return bool(px < K - s_star)


def test_expiry_settlement_mechanics() -> None:
    with _data_env():
        md = make_md()
        i = 500
        S = float(md.close_arr[i, md.j["AAA"]])
        eng = eng_mod.Engine(pd.DataFrame({"AAA": [1.0]}, index=[md.days[i]]), "call",
                             "stock", md, eng_mod.RunConfig())
        t_ord = int(md.day_ordinals[i])
        eng._open_group("AAA", "call", 0.9 * S, t_ord, 10.0, i, "open",
                        (np.empty(0, dtype=np.int64), np.empty(0, dtype=float)))
        spread_before = eng.ledgers["spread"]
        eng._settle_expiries(i, t_ord)
        assert "AAA" not in eng.opt
        assert eng.events["expiry_settlements"] == 1
        assert eng.ledgers["spread"] == spread_before
        assert eng.trade_rows[-1]["action"] == "close_expiry_settle"


def test_ruin_writeoff_reconciliation() -> None:
    with _data_env():
        md = make_md()
        eng = eng_mod.Engine(pd.DataFrame({"BBB": [-6.0]}, index=[md.days[300]]),
                             "stock", "stock", md, eng_mod.RunConfig(check_recon=True))
        i = 320
        t_ord = int(md.day_ordinals[i])
        eng.shares["BBB"] = -200_000.0
        eng.cash += 11_000_000.0
        eng._mark_all(i - 1, int(md.day_ordinals[i - 1]))
        eng._prev_i, eng._prev_nav = i - 1, eng._nav(i - 1)
        eng._fin_today = eng._div_today = eng._borrow_today = 0.0
        eng._spread_today = eng._comm_today = eng._writeoff_today = 0.0
        eng._mtm_today = 0.0
        eng._mark_all(i, t_ord)
        assert eng._nav(i) < 0.0
        eng._liquidate_bankrupt(i, t_ord)
        nav = eng._nav(i)
        expect = (eng._prev_nav + eng._mtm_today + eng._fin_today + eng._div_today
                  - eng._borrow_today - eng._spread_today - eng._comm_today
                  + eng._writeoff_today)
        assert abs(nav) < 1e-9 and abs(nav - expect) < 1e-6
        assert eng._bankrupt and eng.events["bankruptcies"] == 1
        assert not eng.opt and not eng.shares and not eng.pending_reopen


def test_split_scaled_frictions() -> None:
    with _data_env():
        md = make_md()
        j = md.j["CCC"]
        split_i = int(np.argmax(md.days >= SPLIT_DATE))
        legs = [{"cp": "C", "sign": 1.0, "px": 1.0, "delta": 0.85, "iv": 0.20}]
        eng = eng_mod.Engine(pd.DataFrame({"CCC": [1.0]}, index=[md.days[split_i]]),
                             "call", "stock", md, eng_mod.RunConfig())
        sp_pre, cm_pre = eng._option_trade_costs("CCC", split_i - 1, legs, 1.0)
        sp_post, cm_post = eng._option_trade_costs("CCC", split_i, legs, 1.0)
        sf = float(md.split_arr[split_i - 1, j])
        assert sf == 2.0
        assert abs(sp_post / sp_pre - sf) < 1e-9, (sp_pre, sp_post)
        assert abs(cm_post / cm_pre - sf) < 1e-9, (cm_pre, cm_post)


# ------------------------------------------------------------------ signals
def test_momentum_targets_dollar_neutral() -> None:
    with _data_env():
        md = MarketData(["AAA", "BBB", "DDD", "EEE"], IDX[0], IDX[-1])
        saved = config.STOCK_UNIVERSE
        config.STOCK_UNIVERSE = ["AAA", "BBB", "DDD", "EEE"]
        try:
            mom = signals.momentum_frame(md)
            k = 400
            targets = signals.momentum_targets(md.days[k], mom, quintile=2)
        finally:
            config.STOCK_UNIVERSE = saved
        assert set(targets) <= {"AAA", "BBB", "DDD", "EEE"}
        pos_w = sum(w for w in targets.values() if w > 0)
        neg_w = sum(w for w in targets.values() if w < 0)
        assert abs(pos_w - 1.0) < 1e-9 and abs(neg_w + 1.0) < 1e-9
        longs = {t for t, w in targets.items() if w > 0}
        shorts = {t for t, w in targets.items() if w < 0}
        assert not (longs & shorts)


ALL_TESTS = [
    test_synth_md_construction,
    test_engine_recon_and_events_all_variants,
    test_shift_plan_changes_results,
    test_call_exercise_reopen_mechanics,
    test_put_parity_reopen_mechanics,
    test_expiry_settlement_mechanics,
    test_ruin_writeoff_reconciliation,
    test_split_scaled_frictions,
    test_momentum_targets_dollar_neutral,
]


def run_all(verbose: bool = True) -> bool:
    """Run every test; return True when all pass."""
    failures: list[str] = []
    for fn in ALL_TESTS:
        try:
            fn()
            if verbose:
                print(f"  PASS {fn.__name__}")
        except Exception:  # noqa: BLE001 - report and continue
            failures.append(fn.__name__)
            if verbose:
                print(f"  FAIL {fn.__name__}")
                traceback.print_exc()
    if verbose:
        print(f"\n{len(ALL_TESTS) - len(failures)}/{len(ALL_TESTS)} engine tests passed")
    return not failures


if __name__ == "__main__":
    ok = run_all()
    print("ENGINE TESTS:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
