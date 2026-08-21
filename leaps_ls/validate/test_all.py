"""V2/V3 unit tests (PLAN §6) as a plain runnable script with asserts.

Run directly:  python leaps_ls/validate/test_all.py
Or via scripts/02_validate_pricing.py (imports run_all).

Covers: BS put-call parity and textbook values, degenerate T/sigma handling,
escrowed-dividend spot math on a hand-computed example, EWMA vol two-period
closed form, no-after-date guarantee of project_dividends, spread model
monotonicity, BAW American bounds, exercise-rule logic, FRED tenor
interpolation, total-return series, skew/multiplier clamps.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from leaps_ls import config  # noqa: E402
from leaps_ls.data import fred, yahoo  # noqa: E402
from leaps_ls.frictions import spreads  # noqa: E402
from leaps_ls.pricing import baw, black_scholes as bs, dividends as dvd, exercise, vol  # noqa: E402

TOL = 1e-8


# ---------------------------------------------------------------- Black-Scholes
def test_bs_textbook_values() -> None:
    # Textbook case (Haug): S=K=100, T=1, r=5%, sigma=20% -> C=10.4506, P=5.5735, d=0.6368
    c = float(bs.call_price(100.0, 100.0, 1.0, 0.05, 0.2))
    p = float(bs.put_price(100.0, 100.0, 1.0, 0.05, 0.2))
    d = float(bs.delta(100.0, 100.0, 1.0, 0.05, 0.2, "C"))
    assert abs(c - 10.4506) < 1e-3, c
    assert abs(p - 5.5735) < 1e-3, p
    assert abs(d - 0.6368) < 1e-3, d


def test_put_call_parity() -> None:
    rng = np.random.default_rng(7)
    S = rng.uniform(20, 400, 50)
    K = rng.uniform(20, 400, 50)
    T = rng.uniform(0.05, 3.0, 50)
    r = rng.uniform(0.0, 0.06, 50)
    sig = rng.uniform(0.1, 0.8, 50)
    lhs = bs.call_price(S, K, T, r, sig) - bs.put_price(S, K, T, r, sig)
    rhs = S - K * np.exp(-r * T)
    assert np.max(np.abs(lhs - rhs)) < 1e-8, np.max(np.abs(lhs - rhs))


def test_bs_degenerate_inputs() -> None:
    S = np.array([100.0, 100.0, 100.0])
    K = np.array([90.0, 100.0, 110.0])
    # T = 0 -> intrinsic
    assert np.allclose(bs.call_price(S, K, 0.0, 0.05, 0.2), [10.0, 0.0, 0.0])
    assert np.allclose(bs.put_price(S, K, 0.0, 0.05, 0.2), [0.0, 0.0, 10.0])
    # sigma = 0 -> discounted forward intrinsic
    disc = np.exp(-0.05 * 1.5)
    assert np.allclose(bs.call_price(S, K, 1.5, 0.05, 0.0), np.maximum(S - K * disc, 0.0))
    assert np.allclose(bs.put_price(S, K, 1.5, 0.05, 0.0), np.maximum(K * disc - S, 0.0))
    # no NaNs anywhere in greeks at degenerate inputs
    for fn, kw in ((bs.delta, {"cp": "C"}), (bs.gamma, {}), (bs.vega, {}), (bs.theta, {"cp": "P"})):
        out = fn(S, K, 0.0, 0.05, 0.0, **kw)
        assert np.all(np.isfinite(out)), fn.__name__


# ---------------------------------------------------------------- dividends / escrowed spot
def _toy_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ex_date": pd.to_datetime(["2026-10-01", "2027-01-01"]),
            "amount": [1.0, 1.0],
            "t": [0.25, 0.50],
        }
    )


def test_escrowed_spot_hand_example() -> None:
    # S=100, $1 at t=0.25 and $1 at t=0.50, flat 10% -> S* = 100 - e^-0.025 - e^-0.05
    s_star = dvd.adjusted_spot(100.0, _toy_schedule(), lambda d, t: 0.10, T=1.0)
    assert abs(s_star - (100.0 - np.exp(-0.025) - np.exp(-0.05))) < 1e-9, s_star
    # horizon cut: T=0.3 excludes the t=0.50 dividend
    s_star = dvd.adjusted_spot(100.0, _toy_schedule(), lambda d, t: 0.10, T=0.3)
    assert abs(s_star - (100.0 - np.exp(-0.025))) < 1e-9, s_star
    # parity holds on the escrowed spot (V2 by construction)
    K, T, r, sig = 95.0, 1.0, 0.10, 0.25
    c = float(bs.call_price(s_star, K, T, r, sig))
    p = float(bs.put_price(s_star, K, T, r, sig))
    assert abs((c - p) - (s_star - K * np.exp(-r * T))) < TOL


def _toy_hist_with_divs() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", "2025-06-30")
    divs = pd.Series(0.0, index=dates)
    # all ex-dates are business days (weekend dates would be dropped by the bdate index)
    for d in ("2023-03-15", "2023-06-15", "2023-09-15", "2023-12-15",
              "2024-03-15", "2024-06-14", "2024-09-13", "2024-12-13"):
        divs[pd.Timestamp(d)] = 0.20
    for d in ("2025-03-14", "2025-06-13"):  # raise AFTER as_of: must never leak in
        divs[pd.Timestamp(d)] = 0.50
    return pd.DataFrame({"Close": 100.0, "Dividends": divs, "Stock Splits": 0.0}, index=dates)


def test_project_dividends_no_after_date() -> None:
    hist = _toy_hist_with_divs()
    as_of = pd.Timestamp("2024-12-31")
    full = dvd.project_dividends(hist, as_of, horizon_years=2.0)
    trunc = dvd.project_dividends(hist.loc[:as_of], as_of, horizon_years=2.0)
    assert not full.empty
    # identical whether or not post-as_of rows exist: strict no-look-ahead
    pd.testing.assert_frame_equal(full.reset_index(drop=True), trunc.reset_index(drop=True))
    # amount = last pre-as_of dividend (0.20, NOT the later 0.50)
    assert (full["amount"] == 0.20).all()
    # all projected dates strictly after as_of, roughly quarterly
    assert (full["ex_date"] > as_of).all()
    gaps = full["ex_date"].diff().dt.days.dropna()
    assert gaps.between(85, 97).all(), gaps
    assert abs(full["ex_date"].iloc[0] - pd.Timestamp("2025-03-14")).days <= 7
    # non-payer -> empty schedule
    zero_hist = hist.assign(Dividends=0.0)
    assert dvd.project_dividends(zero_hist, as_of, 2.0).empty


def test_project_dividends_freq_tie_prefers_quarterly() -> None:
    # exactly 3 trailing ex-dates (mid-year as_of) must snap UP to quarterly (~91d),
    # not down to semiannual
    hist = _toy_hist_with_divs()
    as_of = pd.Timestamp("2024-10-31")
    sched = dvd.project_dividends(hist, as_of, horizon_years=2.0)
    assert not sched.empty
    gaps = sched["ex_date"].diff().dt.days.dropna()
    assert gaps.between(85, 97).all(), gaps
    assert (sched["amount"] == 0.20).all()


# ---------------------------------------------------------------- vol
def test_ewma_vol_two_period() -> None:
    close = pd.Series([100.0, 102.0, 101.0], index=pd.bdate_range("2026-01-01", periods=3))
    lam = config.EWMA_LAMBDA
    r1 = np.log(102.0 / 100.0)
    r2 = np.log(101.0 / 102.0)
    var2 = lam * r1**2 + (1.0 - lam) * r2**2  # closed form, adjust=False EWMA
    expected = np.sqrt(var2 * 252.0)
    got = vol.ewma_realized_vol(close, min_window=1).iloc[-1]
    assert abs(got - expected) < 1e-10, (got, expected)


def test_vol_clamps() -> None:
    assert vol.clamp_mult(2.0) == config.IV_MULT_HI
    assert vol.clamp_mult(0.5) == config.IV_MULT_LO
    assert vol.clamp_slope(-10.0) == -vol.SLOPE_BOUND
    assert vol.clamp_slope(10.0) == vol.SLOPE_BOUND
    # skew adjustment clamped to ±SKEW_CLAMP however extreme the slope
    iv = vol.apply_skew(0.25, K=80.0, S_star=100.0, slope=-5.0)
    assert abs(iv - (0.25 + config.SKEW_CLAMP)) < 1e-12, iv
    iv = vol.apply_skew(0.25, K=[80.0, 100.0, 120.0], S_star=100.0, slope=0.0)
    assert np.allclose(iv, 0.25)
    # array-valued S* must broadcast too (used by the live-chain validation)
    iv = vol.apply_skew(0.25, K=[80.0, 120.0], S_star=[100.0, 100.0], slope=-0.10)
    assert np.allclose(iv, 0.25 - 0.10 * np.log([0.8, 1.2]))


# ---------------------------------------------------------------- BAW / exercise
def test_baw_bounds() -> None:
    for S, K, T, r, sig in ((100, 100, 1, 0.05, 0.2), (60, 100, 1, 0.08, 0.3),
                            (150, 100, 0.5, 0.03, 0.4), (80, 120, 2, 0.06, 0.25)):
        amer = baw.american_put(S, K, T, r, sig)
        euro = float(bs.put_price(S, K, T, r, sig))
        assert amer >= euro - 1e-9, (amer, euro)
        assert amer >= K - S - 1e-6 or S > 0  # >= intrinsic whenever exercised
    # call on payout-free S* equals European
    assert abs(baw.american_call(100, 100, 1, 0.05, 0.2) - float(bs.call_price(100, 100, 1, 0.05, 0.2))) < 1e-12
    # deep-ITM put with positive rates carries an exercise premium
    assert baw.american_put(60, 100, 1, 0.08, 0.3) > float(bs.put_price(60, 100, 1, 0.08, 0.3)) + 0.5
    # absolute accuracy vs binomial references (CRR N=4000): ATM 1y and moderate cases
    assert abs(baw.american_put(100, 100, 1, 0.05, 0.2) - 6.0902) < 0.02
    assert abs(baw.american_put(90, 100, 0.5, 0.03, 0.4) - 15.6038) < 0.06


def test_exercise_rules() -> None:
    # deep-ITM call: big dividend beats remaining time value, small one does not
    assert exercise.should_exercise_call(5.0, 150.0, 100.0, 0.9, 0.04, 0.25) is True
    assert exercise.should_exercise_call(3.0, 150.0, 100.0, 0.9, 0.04, 0.25) is False
    assert exercise.should_exercise_call(0.0, 150.0, 100.0, 0.9, 0.04, 0.25) is False
    assert exercise.should_assign_short_call(5.0, 150.0, 100.0, 0.9, 0.04, 0.25) is True
    assert exercise.put_parity_violation(20.0, 100.0, 75.0) is True
    assert exercise.put_parity_violation(30.0, 100.0, 75.0) is False


# ---------------------------------------------------------------- frictions
def test_spread_monotonicity() -> None:
    tiers = dict(config.SPREAD_TIERS)
    floor = config.SPREAD_FLOOR_USD
    # within a bucket, half-spread is non-decreasing in premium
    hs = [spreads.half_spread(p, 0.80, tiers=tiers, floor=floor) for p in (1, 5, 20, 60)]
    assert all(b >= a for a, b in zip(hs, hs[1:]))
    # floor binds for tiny premiums
    assert spreads.half_spread(0.50, 0.80, tiers=tiers, floor=floor) == floor
    # default tier ordering: deep ITM cheapest
    assert tiers["itm_deep"] <= tiers["itm_shallow"] <= tiers["atm_otm"]
    # bucket edges
    assert spreads.bucket_for_delta(0.85) == "itm_deep"
    assert spreads.bucket_for_delta(0.70) == "itm_deep"
    assert spreads.bucket_for_delta(0.69) == "itm_shallow"
    assert spreads.bucket_for_delta(0.49) == "atm_otm"
    # stress multiplier scales linearly
    assert spreads.half_spread(10, 0.80, tiers=tiers, floor=floor, mult=2.0) == 2.0 * max(floor, 0.01 * 10)


# ---------------------------------------------------------------- rates / total return
def test_rate_curve_interpolation() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"DGS3MO": [0.05, 0.05], "DGS1": [0.06, 0.06], "DGS2": [0.07, 0.07]}, index=idx)
    curve = fred.RateCurve(df)
    assert abs(curve.rate("2024-01-03", 0.25) - 0.05) < 1e-12
    assert abs(curve.rate("2024-01-03", 0.625) - 0.055) < 1e-12  # linear between 3MO and 1Y
    assert abs(curve.rate("2024-01-03", 5.0) - 0.07) < 1e-12     # flat extrapolation long end
    assert abs(curve.rate("2024-01-03", 0.10) - 0.05) < 1e-12    # flat extrapolation short end
    assert abs(curve.rate("2024-01-04", 0.625) - 0.055) < 1e-12  # ffill to last observation


def test_total_return_series() -> None:
    idx = pd.bdate_range("2026-01-01", periods=3)
    hist = pd.DataFrame(
        {"Close": [100.0, 102.0, 101.0], "Dividends": [0.0, 1.0, 0.0], "Stock Splits": 0.0},
        index=idx,
    )
    tri = yahoo.total_return_series(hist)
    assert abs(tri.iloc[0] - 1.0) < 1e-12
    assert abs(tri.iloc[1] - 1.03) < 1e-12
    assert abs(tri.iloc[2] - 1.03 * (101.0 / 102.0)) < 1e-12


# ---------------------------------------------------------------- runner
ALL_TESTS = [
    test_bs_textbook_values,
    test_put_call_parity,
    test_bs_degenerate_inputs,
    test_escrowed_spot_hand_example,
    test_project_dividends_no_after_date,
    test_project_dividends_freq_tie_prefers_quarterly,
    test_ewma_vol_two_period,
    test_vol_clamps,
    test_baw_bounds,
    test_exercise_rules,
    test_spread_monotonicity,
    test_rate_curve_interpolation,
    test_total_return_series,
]


def run_all(verbose: bool = True) -> bool:
    """Run every test; return True when all pass."""
    config.validate()
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
        print(f"\n{len(ALL_TESTS) - len(failures)}/{len(ALL_TESTS)} unit tests passed")
    return not failures


if __name__ == "__main__":
    ok = run_all()
    print("UNIT TESTS:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
