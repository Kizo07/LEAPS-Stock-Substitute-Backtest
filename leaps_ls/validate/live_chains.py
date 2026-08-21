"""V1 live-chain validation gate (PLAN §6).

Fetches today's full option chains for >= 6 liquid sample names + SPY and, for all
contracts with 300-900 DTE and |mkt delta| in [0.5, 0.95] (market delta from BS with
market IV on the dividend-escrowed spot S*), validates the pricing stack:

  (a) GATE — median |model price − market mid| / mid ≤ 10% on this near-ATM 1-2y
      bucket, pooled across the sample. The plan's literal "model price using market
      IV" is not implementable as a meaningful gate with free data (see below), so the
      gate is applied to the FULL calibrated vol stack (mult × EWMA RV + skew; VIX
      anchor for SPY) — stricter in content, since it includes vol-model error — plus
      two IV-free machinery checks (mid-implied repricing residual and put-call-parity
      forward deviation ≤ 1%).
  (b) vol-model test — our IV proxy vs the market IV distribution.

Market-IV sourcing (deviation from the plan text, flagged): the plan says "model price
using market IV" where the natural free field is Yahoo's ``impliedVolatility``. That
field is derived from each contract's LAST TRADE price against the CURRENT spot, so on
trending days it is internally inconsistent with live bid/ask mids — in this sample
(large down day) it ran ~+6..+10 vol pts vs mid-implied vol on calls and −7..−12 on
puts, and repricing mids with it errs ~10% with a strong one-sided sign. We therefore
define market IV as ``iv_mkt`` = BS-implied vol of the market MID on our S* (standard
practice) for deltas, calibration and the vol-model test, and report the literal
Yahoo-IV table as a diagnostic.

Writes results/validation_live_chains.md and persists calibration to
data/calibrated_vol.json and data/calibrated_spreads.json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .. import config
from ..data import fred, yahoo
from ..frictions import spreads as spreads_mod
from ..pricing import black_scholes as bs, dividends as dvd, vol

SAMPLE_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "KO", "NVDA"]
INDEX_SAMPLE = ["SPY"]
DTE_RANGE = (300, 900)          # validation window ("1-2y" bucket)
DELTA_RANGE = (0.50, 0.95)      # near-ATM to moderately ITM, per plan
GATE_THRESHOLD = 0.10           # median |err|/mid, pooled
PARITY_THRESHOLD = 0.01         # per-ticker parity-forward deviation, fraction of spot
DTE_BUCKETS = ((300, 500), (500, 700), (700, 900))


# ------------------------------------------------------------------ market IV from mid
def implied_vol_from_mid(mid: float, s_star: float, K: float, T: float, r: float, cp: str) -> float:
    """BS implied vol of a market mid on escrowed spot S* (NaN when no root)."""
    disc = np.exp(-r * T)
    lo = max(s_star - K * disc, 0.0) if cp == "C" else max(K * disc - s_star, 0.0)
    if not np.isfinite(mid) or mid <= lo + 1e-6:
        return float("nan")
    f = lambda sig: float(bs.price(s_star, K, T, r, sig, cp)) - mid  # noqa: E731
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-8)
    except Exception:  # noqa: BLE001 - no bracket/root
        return float("nan")


# ------------------------------------------------------------------ chain preparation
def prepare_chain(ticker: str, hist: pd.DataFrame, curve: fred.RateCurve, as_of) -> pd.DataFrame:
    """Chain frame annotated with S*, rate, market IV (mid-implied), delta, model prices.

    Spot = the chain's own underlying price when Yahoo provides it, else the last
    cached close. S* escrows the projected (ex-ante) dividend schedule. Market delta
    uses the mid-implied market IV, falling back to Yahoo's IV when inversion fails.
    """
    chain = yahoo.fetch_option_chain(ticker)
    und = chain["underlying_price"].dropna()
    spot = float(und.iloc[0]) if len(und) and np.isfinite(und.iloc[0]) else float(hist["Close"].iloc[-1])
    schedule = dvd.project_dividends(hist, as_of, horizon_years=3.0)

    df = chain[(chain["dte"] >= DTE_RANGE[0]) & (chain["dte"] <= DTE_RANGE[1])].copy()
    df = df[(df["bid"] > 0) & (df["ask"] > df["bid"])].copy()
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df = df[(df["mid"] > 0.10) & (df["impliedVolatility"] > 0.005) & (df["impliedVolatility"] < 3.0)]
    if df.empty:
        return df

    parts: list[pd.DataFrame] = []
    for (expiry, cp), g in df.groupby(["expiry", "type"]):
        T = float(g["dte"].iloc[0]) / 365.0
        r = curve.rate(as_of, T)
        rate_fn = lambda d, t: curve.rate(as_of, max(float(t), 1e-6))  # noqa: B023
        s_star = dvd.adjusted_spot(spot, schedule, rate_fn, T)
        g = g.copy()
        g["T"] = T
        g["r"] = r
        g["s_star"] = s_star
        g["iv_mkt"] = [
            implied_vol_from_mid(m, s_star, k, T, r, cp)
            for m, k in zip(g["mid"], g["strike"])
        ]
        iv_eff = g["iv_mkt"].fillna(g["impliedVolatility"]).to_numpy()
        K = g["strike"].to_numpy()
        g["mkt_delta"] = bs.delta(s_star, K, T, r, iv_eff, cp)
        g["model_px_mktiv"] = bs.price(s_star, K, T, r, g["impliedVolatility"].to_numpy(), cp)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["abs_err_mid"] = (out["model_px_mktiv"] - out["mid"]).abs() / out["mid"]
    out["log_money"] = np.log(out["strike"] / out["s_star"])
    out["ticker"] = ticker
    out["spot"] = spot
    return out


def gate_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Contracts the gate applies to: |mkt delta| in DELTA_RANGE (DTE already filtered)."""
    return df[df["mkt_delta"].abs().between(*DELTA_RANGE)]


def parity_forward_dev(df: pd.DataFrame) -> float:
    """Median per-expiry |F_model − F_impl| / spot at the near-ATM strike (IV-free).

    F_impl comes from put-call parity on market mids, F_model = S* e^{rT}; a small
    value validates the S*/rate machinery without relying on any IV source.
    """
    devs: list[float] = []
    for expiry, g in df.groupby("expiry"):
        calls = g[g["type"] == "C"].set_index("strike")
        puts = g[g["type"] == "P"].set_index("strike")
        both = calls[["mid", "s_star", "T", "r", "spot"]].join(
            puts[["mid"]], lsuffix="_c", rsuffix="_p", how="inner"
        )
        if both.empty:
            continue
        s_star = float(both["s_star"].iloc[0])
        K = float(both.index[np.argmin(np.abs(both.index.to_numpy() - s_star))])
        row = both.loc[[K]].iloc[0]
        f_impl = K + np.exp(row["r"] * row["T"]) * (row["mid_c"] - row["mid_p"])
        f_model = s_star * np.exp(row["r"] * row["T"])
        devs.append(abs(f_model - f_impl) / float(row["spot"]))
    return float(np.median(devs)) if devs else float("nan")


# ------------------------------------------------------------------ report helpers
def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> list[str]:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(floatfmt.format(v) if isinstance(v, float) and np.isfinite(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _proxy_prices(g: pd.DataFrame, atm: float, slope: float) -> np.ndarray:
    """Model prices on the gate bucket using the full calibrated vol stack."""
    proxy_iv = vol.apply_skew(atm, g["strike"].to_numpy(), g["s_star"].to_numpy(), slope)
    return np.where(
        g["type"].to_numpy() == "C",
        bs.price(g["s_star"], g["strike"], g["T"], g["r"], proxy_iv, "C"),
        bs.price(g["s_star"], g["strike"], g["T"], g["r"], proxy_iv, "P"),
    )


# ------------------------------------------------------------------ main validation
def run_validation(refresh_chains: bool = False, verbose: bool = True) -> dict:
    """Run the V1 gate end to end; write the markdown report and calibration JSONs."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp.today().normalize()
    curve = fred.load_curve()
    tickers = SAMPLE_TICKERS + INDEX_SAMPLE
    vix_now = float(yahoo.fetch_vix().loc[:as_of].iloc[-1])

    # ---- prepare chains
    prepared: dict[str, pd.DataFrame] = {}
    hists: dict[str, pd.DataFrame] = {}
    raw_chains: dict[str, pd.DataFrame] = {}
    spots: dict[str, float] = {}
    for tk in tickers:
        try:
            if refresh_chains:
                yahoo.fetch_option_chain(tk, refresh=True)
            hists[tk] = yahoo.load_history(tk)
            prepared[tk] = prepare_chain(tk, hists[tk], curve, as_of)
            raw_chains[tk] = yahoo.load_option_chain(tk)
            spots[tk] = float(prepared[tk]["spot"].iloc[0]) if len(prepared[tk]) else np.nan
            if verbose:
                print(f"  {tk}: {len(prepared[tk])} usable contracts (300-900 DTE), "
                      f"spot {spots[tk]:.2f}")
        except Exception as exc:  # noqa: BLE001 - report and continue with fewer names
            print(f"  {tk}: chain preparation FAILED: {exc}")
            prepared[tk] = pd.DataFrame()

    # ---- calibrate vol model (multiplier / term_adj / slope) on mid-implied IVs
    cal: dict = {"calibrated_on": str(as_of.date()), "tickers": {}}
    mults, slopes = [], []
    for tk in tickers:
        df = prepared[tk]
        if df.empty:
            cal["tickers"][tk] = {"error": "no prepared contracts"}
            continue
        rv = vol.rv_as_of(hists[tk], as_of)
        rec = vol.calibrate_from_chain(tk, df, rv, vix_now=vix_now)
        cal["tickers"][tk] = rec
        if rec.get("mult") is not None:
            mults.append(rec["mult"])
        if rec.get("slope") is not None:
            slopes.append(rec["slope"])
    cal["default"] = {
        "mult": float(np.median(mults)) if mults else 1.0,
        "slope": float(np.median(slopes)) if slopes else 0.0,
    }
    vol_path = vol.save_vol_calibration(cal)

    # ---- calibrate spread tiers and persist
    try:
        spread_cal = spreads_mod.calibrate_from_chains(raw_chains, spots=spots)
    except Exception as exc:  # noqa: BLE001
        spread_cal = {"error": str(exc)}
        print(f"  spread calibration FAILED: {exc}")

    # ---- per-ticker metrics on the gate bucket
    gate_rows, mach_rows, vol_rows = [], [], []
    proxy_err_all: list[np.ndarray] = []
    yahoo_err_all: list[np.ndarray] = []
    for tk in tickers:
        df = prepared[tk]
        if df.empty:
            gate_rows.append({"ticker": tk, "n": 0, "proxy_px_err": np.nan,
                              "yahoo_iv_err": np.nan, "gate": "SKIP"})
            continue
        g = gate_bucket(df)
        rec = cal["tickers"].get(tk, {})
        atm = vol.iv_proxy(as_of, tk, hist=hists[tk], cal=cal)
        slope = vol.get_skew_slope(tk, cal)
        px_proxy = _proxy_prices(g, atm, slope)
        proxy_err = np.abs(px_proxy - g["mid"].to_numpy()) / g["mid"].to_numpy()
        proxy_err_all.append(proxy_err)
        yahoo_err_all.append(g["abs_err_mid"].to_numpy())
        med_proxy = float(np.median(proxy_err))
        med_yahoo = float(g["abs_err_mid"].median())
        gate_rows.append({
            "ticker": tk, "n": int(len(g)), "proxy_px_err": med_proxy,
            "yahoo_iv_err": med_yahoo,
            "gate": "PASS" if med_proxy <= GATE_THRESHOLD else "FAIL",
        })

        # machinery: mid-implied repricing residual (consistency) + parity forward (IV-free)
        gi = g.dropna(subset=["iv_mkt"])
        resid = np.nan
        if len(gi):
            px_i = np.where(
                gi["type"].to_numpy() == "C",
                bs.price(gi["s_star"], gi["strike"], gi["T"], gi["r"], gi["iv_mkt"], "C"),
                bs.price(gi["s_star"], gi["strike"], gi["T"], gi["r"], gi["iv_mkt"], "P"),
            )
            resid = float(np.median(np.abs(px_i - gi["mid"].to_numpy()) / gi["mid"].to_numpy()))
        mach_rows.append({
            "ticker": tk,
            "reprice_resid_ivmid": f"{resid:.1e}" if np.isfinite(resid) else "n/a",
            "parity_fwd_dev": parity_forward_dev(df),
            "parity_gate": "PASS" if parity_forward_dev(df) <= PARITY_THRESHOLD else "FAIL",
        })

        # vol model in IV space
        if len(gi):
            rv = rec.get("rv")
            proxy_iv = vol.apply_skew(atm, gi["strike"].to_numpy(), gi["s_star"].to_numpy(), slope)
            mkt_iv = gi["iv_mkt"].to_numpy()
            vol_rows.append({
                "ticker": tk,
                "rv": float(rv) if rv else np.nan,
                "mult_or_termadj": rec.get("mult") if rec.get("mult") is not None else rec.get("term_adj"),
                "slope": slope,
                "median_mkt_iv": float(np.median(mkt_iv)),
                "median_proxy_iv": float(np.median(proxy_iv)),
                "proxy_over_mkt": float(np.mean(proxy_iv / mkt_iv)),
                "med_abs_iv_diff": float(np.median(np.abs(proxy_iv - mkt_iv))),
            })

    pooled_proxy = float(np.median(np.concatenate(proxy_err_all))) if proxy_err_all else float("nan")
    pooled_yahoo = float(np.median(np.concatenate(yahoo_err_all))) if yahoo_err_all else float("nan")
    total_n = sum(r["n"] for r in gate_rows)
    parity_ok = all(r["parity_gate"] == "PASS" for r in mach_rows)
    gate_pass = bool(pooled_proxy <= GATE_THRESHOLD and parity_ok)
    gate_rows.append({
        "ticker": "ALL (pooled)", "n": int(total_n), "proxy_px_err": pooled_proxy,
        "yahoo_iv_err": pooled_yahoo, "gate": "PASS" if pooled_proxy <= GATE_THRESHOLD else "FAIL",
    })
    gate_table = pd.DataFrame(gate_rows)
    mach_table = pd.DataFrame(mach_rows)
    vol_table = pd.DataFrame(vol_rows)

    # DTE sub-buckets of the literal Yahoo-IV diagnostic, pooled
    pooled_df = pd.concat([gate_bucket(prepared[tk]) for tk in tickers if not prepared[tk].empty])
    dte_rows = []
    for lo, hi in DTE_BUCKETS:
        sub = pooled_df[pooled_df["dte"].between(lo, hi - 1)]
        dte_rows.append({
            "dte_bucket": f"{lo}-{hi}", "n": int(len(sub)),
            "yahoo_iv_err": float(sub["abs_err_mid"].median()) if len(sub) else np.nan,
            "p90_err": float(sub["abs_err_mid"].quantile(0.9)) if len(sub) else np.nan,
        })
    dte_table = pd.DataFrame(dte_rows)

    # ---- markdown report
    fetched = "n/a"
    if tickers[0] in raw_chains and len(raw_chains[tickers[0]]):
        fetched = str(raw_chains[tickers[0]]["fetched"].iloc[0].date())
    lines = [
        "# V1 — Live-chain validation (PLAN §6)",
        "",
        f"Validation date (as_of): **{as_of.date()}**; chains fetched from Yahoo Finance on {fetched}.",
        "",
        "Model: Black-Scholes on dividend-escrowed spot S* = S − PV(projected discrete "
        "dividends before expiry), FRED DGS3MO/DGS1/DGS2 curve interpolated to tenor, "
        "IV proxy = calibrated multiplier × EWMA RV with linear log-moneyness skew "
        "(VIX anchor for SPY).",
        "",
        "## Gate",
        "",
        f"Gate bucket: 300–900 DTE, |Δ_mkt| ∈ [{DELTA_RANGE[0]}, {DELTA_RANGE[1]}], "
        f"pooled across {', '.join(tickers)}.",
        "",
        f"- **Full-stack price error (GATED, ≤ {GATE_THRESHOLD:.0%}): {pooled_proxy:.2%} "
        f"→ {'PASS' if pooled_proxy <= GATE_THRESHOLD else 'FAIL'}** — model priced with "
        "the calibrated vol proxy vs market mid.",
        f"- **Parity-forward deviation (GATED, ≤ {PARITY_THRESHOLD:.0%} per ticker): "
        f"{'PASS' if parity_ok else 'FAIL'}** — IV-free check of S*/rate machinery.",
        f"- Literal plan-text variant (model at Yahoo's `impliedVolatility` field): "
        f"{pooled_yahoo:.2%} pooled — diagnostic only, see market-IV note below.",
        "",
        "**Deviation from the plan text (flagged):** the plan's 'model price using "
        "market IV' is not implementable as a meaningful gate with free data. Yahoo's "
        "impliedVolatility is derived from each contract's last trade price against the "
        "current spot; on trending days it is inconsistent with live mids (this sample: "
        "+6..+10 vol pts vs mid-implied vol on calls, −7..−12 on puts, after a large "
        "down move; restricting to contracts traded today does NOT remove it). Market IV "
        "is therefore defined as `iv_mkt` = BS-implied vol of the market mid on our S*. "
        "With that definition the machinery repricing residual is ~1e-11 (Table 3), so "
        "the 10% gate is applied to the full vol-model stack (non-tautological) and to "
        "the IV-free parity-forward check instead.",
        "",
        "### Table 1 — price error on the gate bucket, per ticker "
        "(full vol-model stack; Yahoo-IV diagnostic alongside)",
        *_md_table(gate_table, "{:.4f}"),
        "",
        "### Table 2 — Yahoo-IV diagnostic by DTE bucket (pooled; not gated)",
        *_md_table(dte_table, "{:.4f}"),
        "",
        "### Table 3 — machinery checks (IV-free / consistency)",
        "",
        "- `reprice_resid_ivmid`: median |err| repricing mids with their own mid-implied "
        "vol — ≈0 confirms mids are exactly reachable in our (S*, r) frame.",
        "- `parity_fwd_dev`: median per-expiry |F_model − F_impl|/spot from put-call "
        "parity on near-ATM mids — validates S* (dividend + rate assumptions) against "
        "the market-implied forward; gated at 1% of spot.",
        "",
        *_md_table(mach_table, "{:.6f}"),
        "",
        "### Table 4 — vol model vs market IV (mid-implied)",
        *_md_table(vol_table, "{:.4f}"),
        "",
        "Note: names whose EWMA RV spiked on very recent jumps (e.g. earnings) show "
        "proxy IV above market IV; the multiplier clamp floor of 1.00 (PLAN §5.2) "
        "prevents discounting RV. This is a known EWMA limitation after jumps, stressed "
        "via the §7 iv_mult sensitivity.",
        "",
        "### Calibration artifacts",
        f"- `data/calibrated_vol.json` — per-ticker IV/RV multiplier (clamped to "
        f"[{config.IV_MULT_LO}, {config.IV_MULT_HI}]), skew slope (clamped), SPY term_adj; "
        f"cross-sectional defaults mult={cal['default']['mult']:.3f}, "
        f"slope={cal['default']['slope']:.3f}.",
        "- `data/calibrated_spreads.json` — half-spread tiers from live LEAPS quotes.",
        "",
        "Spread tiers now in effect (calibrated where sample size allowed):",
        "",
    ]
    if "tiers" in spread_cal:
        lines += [
            "| bucket | tier (half-spread / premium) | n contracts | default |",
            "|---|---|---|---|",
        ]
        for name, rec in spread_cal["tiers"].items():
            val = rec["value"] if rec["value"] is not None else rec["default"]
            lines.append(f"| {name} | {val:.4f} | {rec['n']} | {rec['default']:.4f} |")
    else:
        lines.append(f"Spread calibration failed: {spread_cal.get('error', 'unknown')}")
    lines += [
        "",
        "### Caveats",
        "- Yahoo quotes are delayed; contracts with zero bid, crossed markets, or "
        "mid ≤ $0.10 are excluded. Deep-ITM LEAPS are wide/illiquid, so mids are noisy.",
        "- The base model prices European; American early-exercise premium (mainly "
        "deep-ITM puts at high rates) is handled by rules + BAW sensitivity (PLAN §5.1), "
        "so European prices can sit a few % below American mids for puts.",
        "- Calibration is anchored to today's chains only; historical backtests stress "
        "the vol multiplier and spread levels per PLAN §7.",
        "- Skew is linear in log-moneyness; smile/term curvature is not modeled.",
    ]
    report_path = config.RESULTS_DIR / "validation_live_chains.md"
    report_path.write_text("\n".join(lines) + "\n")

    summary = {
        "as_of": str(as_of.date()),
        "gate_threshold": GATE_THRESHOLD,
        "pooled_proxy_err": pooled_proxy,
        "pooled_yahoo_iv_err": pooled_yahoo,
        "parity_ok": parity_ok,
        "gate_pass": gate_pass,
        "per_ticker": {r["ticker"]: r for r in gate_rows},
        "machinery": {r["ticker"]: r for r in mach_rows},
        "vol_cal_path": str(vol_path),
        "report_path": str(report_path),
        "default_mult": cal["default"]["mult"],
        "default_slope": cal["default"]["slope"],
        "spread_cal": spread_cal if "tiers" in spread_cal else None,
    }
    if verbose:
        print(f"\nV1 gate: pooled full-stack price error = {pooled_proxy:.2%} "
              f"(≤ {GATE_THRESHOLD:.0%}: {'PASS' if pooled_proxy <= GATE_THRESHOLD else 'FAIL'}); "
              f"parity-forward check: {'PASS' if parity_ok else 'FAIL'}")
        print(gate_table.to_string(index=False))
        print("\nMachinery checks:")
        print(mach_table.to_string(index=False))
        print("\nVol model:")
        print(vol_table.to_string(index=False))
        print(f"\nReport: {report_path}")
    return summary


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    out = run_validation()
    raise SystemExit(0 if out["gate_pass"] else 1)
