"""Tracking-difference decomposition (PLAN §5.10; RQ1 long side, RQ2 short side).

For every ticker and each lane comparison (call vs stock, synth_long vs stock,
put vs short_stock, synth_short vs short_stock) the annualized tracking
difference TD (option-lane daily-return minus stock-lane, x252) is decomposed
into ledger-driven buckets plus a residual, with the EXACT identity

    TD  =  ddiv + dfin - dcost + dborrow + resid

where each bucket is the difference between the two lanes' DAILY ledger flows
scaled to same-day NAV, annualized (mean(flow_t / NAV_{t-1}) x 252 — the same
return-equivalent basis as TD):
- ddiv    = dividends flow (option lane) - (stock lane)   [drag on long lanes]
- dfin    = financing flow (option lane) - (stock lane)   [embedded financing benefit]
- dcost   = (spread+comm) flow (option lane) - (stock lane)  [extra trading cost]
- dborrow = borrow flow (stock lane) - (option lane)      [borrow saved by puts]
- resid   = TD minus the above = "vol premium + theta + model" (vol-proxy vs
            realized vol, time-value decay, skew, projection/model error).

Runs offline from results/ (nav_lanes.parquet + lane_ledgers.parquet).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from leaps_ls import config  # noqa: E402
from leaps_ls.analysis import metrics  # noqa: E402

LONG_COMPS = [("call", "stock"), ("synth_long", "stock")]
SHORT_COMPS = [("put", "short_stock"), ("synth_short", "short_stock")]
BUCKETS = ["ddiv", "dfin", "dcost", "dborrow", "resid"]


def _lane_navs() -> pd.DataFrame:
    return pd.read_parquet(config.RESULTS_DIR / "nav_lanes.parquet")


def _lane_ledgers() -> pd.DataFrame:
    return pd.read_parquet(config.RESULTS_DIR / "lane_ledgers.parquet")


def _lane_summary() -> pd.DataFrame:
    return pd.read_csv(config.RESULTS_DIR / "lane_summary.csv")


def _ann_flow(ledgers: pd.DataFrame, navs: pd.DataFrame, key: str,
              cum_cols: tuple[str, ...]) -> pd.Series:
    """Annualized return-equivalent of cumulative ledger flows: mean(dflow/NAV_prev) x 252.

    Ledger flows are raw dollars; lane NAVs are normalized to 1 at start, so the
    same-day NAV is rescaled by config.CAPITAL_BASE (error < day-1 costs, negligible).
    """
    flow = None
    for col in cum_cols:
        f = ledgers[f"{key}:{col}"].diff().fillna(ledgers[f"{key}:{col}"].iloc[0:1].squeeze())
        flow = f if flow is None else flow + f
    nav = navs[key] * config.CAPITAL_BASE
    return (flow / nav.shift(1)).dropna() * metrics.TRADING_DAYS


def decompose(comparisons: list[tuple[str, str]], navs: pd.DataFrame,
              ledgers: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker decomposition rows for the given (option_lane, stock_lane) pairs."""
    s = summary.set_index(["ticker", "lane"])
    rows: list[dict] = []
    for tk in summary["ticker"].unique():
        for opt_lane, stk_lane in comparisons:
            ko, ks = f"{tk}:{opt_lane}", f"{tk}:{stk_lane}"
            if ko not in navs.columns or ks not in navs.columns:
                continue
            td = metrics.ann_tracking_diff(
                metrics.daily_returns(navs[ko]), metrics.daily_returns(navs[ks]))
            div = _ann_flow(ledgers, navs, ko, ("dividends_cum",)) - _ann_flow(
                ledgers, navs, ks, ("dividends_cum",))
            fin = _ann_flow(ledgers, navs, ko, ("financing_cum",)) - _ann_flow(
                ledgers, navs, ks, ("financing_cum",))
            cost = _ann_flow(ledgers, navs, ko, ("spread_cum", "comm_cum")) - _ann_flow(
                ledgers, navs, ks, ("spread_cum", "comm_cum"))
            borr = _ann_flow(ledgers, navs, ks, ("borrow_cum",)) - _ann_flow(
                ledgers, navs, ko, ("borrow_cum",))
            ddiv, dfin, dcost, dborrow = (float(div.mean()), float(fin.mean()),
                                          float(cost.mean()), float(borr.mean()))
            resid = td - (ddiv + dfin - dcost + dborrow)
            idx = navs[ko].dropna().index
            rows.append({
                "ticker": tk, "comparison": f"{opt_lane} vs {stk_lane}",
                "option_lane": opt_lane, "td_ann": td, "ddiv": ddiv, "dfin": dfin,
                "dcost": dcost, "dborrow": dborrow, "resid": resid,
                "years": (idx[-1] - idx[0]).days / 365.25,
                "bankruptcies": int(s.loc[(tk, opt_lane), "ev_bankruptcies"])
                if (tk, opt_lane) in s.index else 0,
            })
    df = pd.DataFrame(rows)
    # cross-sectional median/mean rows per comparison
    agg_rows = []
    for comp, g in df.groupby("comparison"):
        for stat, fn in (("MEDIAN", "median"), ("MEAN", "mean")):
            r = {"ticker": stat, "comparison": comp,
                 "option_lane": g["option_lane"].iloc[0],
                 **{c: float(getattr(g[c], fn)()) for c in
                    ["td_ann", "ddiv", "dfin", "dcost", "dborrow", "resid"]},
                 "years": np.nan, "bankruptcies": int(g["bankruptcies"].sum())}
            agg_rows.append(r)
    return pd.concat([df, pd.DataFrame(agg_rows)], ignore_index=True)


def _md_table(df: pd.DataFrame, cols: list[str], pct_cols: set[str]) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in pct_cols and isinstance(v, float) and np.isfinite(v):
                cells.append(f"{v:+.2%}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    navs = _lane_navs()
    ledgers = _lane_ledgers()
    summary = _lane_summary()
    long_df = decompose(LONG_COMPS, navs, ledgers, summary)
    short_df = decompose(SHORT_COMPS, navs, ledgers, summary)
    long_df.to_csv(config.RESULTS_DIR / "decomposition_long.csv", index=False)
    short_df.to_csv(config.RESULTS_DIR / "decomposition_short.csv", index=False)

    both = pd.concat([long_df, short_df])
    med = both[both["ticker"] == "MEDIAN"]
    pct = {"td_ann", "ddiv", "dfin", "dcost", "dborrow", "resid"}
    cols = ["comparison", "td_ann", "ddiv", "dfin", "dcost", "dborrow", "resid"]

    def hit_rate(df: pd.DataFrame, lane: str, thr: float) -> float:
        g = df[(df["option_lane"] == lane) & (df["ticker"] != "MEDIAN") & (df["ticker"] != "MEAN")]
        return float((g["td_ann"] >= thr).mean()) if len(g) else float("nan")

    lines = [
        "# Tracking-difference decomposition — cross-sectional medians (annualized)",
        "",
        "Identity: `TD = ddiv + dfin − dcost + dborrow + resid` (see "
        "leaps_ls/analysis/decomposition.py docstring for bucket definitions). "
        "resid = vol premium + theta + model (vol proxy vs realized vol, time-value "
        "decay, skew, projection/model error, compounding effects).",
        "",
        _md_table(med, cols, pct)[0],
        *_md_table(med, cols, pct)[1:],
        "",
        "## Hit rates (fraction of names)",
        "",
        f"- call lane TD ≥ −1%/yr: {hit_rate(both, 'call', -0.01):.0%}; "
        f"≥ −2%/yr: {hit_rate(both, 'call', -0.02):.0%}",
        f"- synth_long lane TD ≥ −1%/yr: {hit_rate(both, 'synth_long', -0.01):.0%}; "
        f"≥ −2%/yr: {hit_rate(both, 'synth_long', -0.02):.0%}",
        f"- put lane TD ≥ −3%/yr: {hit_rate(both, 'put', -0.03):.0%}; "
        f"≥ −5%/yr: {hit_rate(both, 'put', -0.05):.0%}",
        f"- synth_short lane TD ≥ −3%/yr: {hit_rate(both, 'synth_short', -0.03):.0%}; "
        f"≥ −5%/yr: {hit_rate(both, 'synth_short', -0.05):.0%}",
        "",
        "Files: `decomposition_long.csv` (call/synth_long vs stock), "
        "`decomposition_short.csv` (put/synth_short vs short_stock); per ticker + "
        "MEDIAN/MEAN rows.",
    ]
    out = config.RESULTS_DIR / "decomposition_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(med[cols].to_string(index=False,
                             formatters={c: "{:+.2%}".format for c in pct}))
    print(f"\nHit rates: call ≥-1%: {hit_rate(both, 'call', -0.01):.0%}, "
          f"≥-2%: {hit_rate(both, 'call', -0.02):.0%}; "
          f"put ≥-3%: {hit_rate(both, 'put', -0.03):.0%}, "
          f"≥-5%: {hit_rate(both, 'put', -0.05):.0%}")
    print(f"Wrote decomposition_long.csv, decomposition_short.csv, {out.name}")


if __name__ == "__main__":
    main()
