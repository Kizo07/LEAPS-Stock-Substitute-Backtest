#!/usr/bin/env python
"""Assemble results/tables_manifest.json and a skeleton REPORT.md (Phase 4/5 glue).

The report skeleton contains section headers (methods, RQ1-RQ4, validation,
limitations), auto-inserted key tables (portfolio summary, decomposition
medians, FF attribution, top sensitivity swings), and explicit prose
placeholders. Narrative prose is written separately — this script only
scaffolds. Reads everything from results/ (offline).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from leaps_ls import config  # noqa: E402

CAPTIONS = {
    "data_coverage.csv": "raw-data coverage per ticker/source (Phase 1)",
    "validation_live_chains.md": "V1 live-chain validation gate report (Phase 2)",
    "nav_lanes.parquet": "daily normalized NAV, all single-instrument lanes",
    "lane_ledgers.parquet": "daily cumulative cost ledgers per lane",
    "lane_summary.csv": "per-lane headline stats, cost buckets, event counts, tracking diff",
    "nav_portfolios.parquet": "daily normalized NAV, V0-V3 portfolio variants",
    "portfolio_ledgers.parquet": "daily cumulative cost ledgers per variant",
    "portfolio_summary.csv": "V0-V3 headline stats + cost buckets + event counts",
    "trades_V0.csv": "trade log, V0 all-stock",
    "trades_V1.csv": "trade log, V1 LEAPS replacement",
    "trades_V2.csv": "trade log, V2 synthetic",
    "trades_V3.csv": "trade log, V3 hybrid",
    "decomposition_long.csv": "per-ticker TD decomposition, long lanes (call/synth_long vs stock)",
    "decomposition_short.csv": "per-ticker TD decomposition, short lanes (put/synth_short vs short_stock)",
    "decomposition_summary.md": "median decomposition table + hit rates",
    "ff_attribution.csv": "FF3+Momentum daily-return regressions, V0-V3",
    "subperiods.csv": "CAGR/Sharpe by rate-regime window, variants + SPY lanes",
    "sensitivities_portfolios.csv": "PLAN §7 one-at-a-time grid, V0-V3",
    "sensitivities_lanes.csv": "reduced grid on 6 names, call/put lanes",
    "fig_portfolio_nav.png": "V0-V3 log NAV vs SPY and equal-weight benchmark",
    "fig_spy_lanes.png": "SPY long and short lanes, log NAV panels",
    "fig_tracking_cross.png": "cross-sectional tracking-difference distributions",
    "fig_decomposition.png": "median decomposition bars (long call, short put)",
    "fig_tornado_portfolios.png": "V1 Sharpe one-at-a-time sensitivity tornado",
    "fig_tornado_spy_call.png": "SPY call-lane TD sensitivity tornado",
    "fig_drawdown.png": "V0-V3 drawdown curves",
}

PROSE = "_[Prose to be written in the final report pass.]_"


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(floatfmt.format(v) if isinstance(v, float) and np.isfinite(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_manifest() -> dict:
    entries = []
    for name in sorted(CAPTIONS):
        path = config.RESULTS_DIR / name
        entries.append({"file": name, "caption": CAPTIONS[name], "exists": path.exists()})
    manifest = {
        "results_dir": str(config.RESULTS_DIR),
        "environment": _env_stamp(),
        "entries": entries,
    }
    (config.RESULTS_DIR / "tables_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _env_stamp() -> dict:
    """Record the interpreter and key package versions that produced these results."""
    import platform

    from importlib.metadata import version

    pkgs = {}
    for dist in ("numpy", "pandas", "scipy", "matplotlib", "pyarrow", "yfinance"):
        try:
            pkgs[dist] = version(dist)
        except Exception:  # noqa: BLE001 - stamp what is known
            pkgs[dist] = "unknown"
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": pkgs}


def top_swings(pf: pd.DataFrame, variant: str, metric: str, n: int = 3) -> pd.DataFrame:
    rows = []
    for param in pf["param"].unique():
        if param == "base":
            continue
        sub = pf[(pf["param"] == param) & (pf["variant"] == variant)]
        g = sub.groupby("value")[metric].agg(["min", "max"])
        rows.append({"param": param, "lo": g["min"].min(), "hi": g["max"].max(),
                     "span": g["max"].max() - g["min"].min()})
    return pd.DataFrame(rows).sort_values("span", ascending=False).head(n)


def build_report(manifest: dict) -> None:
    r = config.RESULTS_DIR
    pf_sum = pd.read_csv(r / "portfolio_summary.csv")
    dec_long = pd.read_csv(r / "decomposition_long.csv")
    dec_short = pd.read_csv(r / "decomposition_short.csv")
    attr = pd.read_csv(r / "ff_attribution.csv")
    sens_pf = pd.read_csv(r / "sensitivities_portfolios.csv")
    sub = pd.read_csv(r / "subperiods.csv")

    pct = {"cagr": "{:.1%}", "vol": "{:.1%}", "maxdd": "{:.1%}", "spread_pct": "{:.1%}",
           "comm_pct": "{:.1%}", "borrow_pct": "{:.1%}", "financing_pct": "{:.1%}",
           "dividends_pct": "{:.1%}", "sharpe": "{:.2f}"}

    def fmt(df: pd.DataFrame) -> str:
        d = df.copy()
        for c, f in pct.items():
            if c in d.columns:
                d[c] = d[c].map(lambda v: f.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else v)
        return _md(d)

    med_long = dec_long[dec_long["ticker"] == "MEDIAN"].copy()
    med_short = dec_short[dec_short["ticker"] == "MEDIAN"].copy()
    buckets = ["td_ann", "ddiv", "dfin", "dcost", "dborrow", "resid"]
    for d in (med_long, med_short):
        for c in buckets:
            d[c] = d[c].map(lambda v: f"{v:+.2%}")

    swings_v1 = top_swings(sens_pf, "V1", "cagr")
    swings_v1_s = top_swings(sens_pf, "V1", "sharpe")

    pf_cols = ["variant", "cagr", "vol", "sharpe", "maxdd", "spread_pct", "comm_pct",
               "borrow_pct", "financing_pct", "dividends_pct"]
    attr_cols = ["series", "alpha_ann", "alpha_t", "beta_Mkt-RF", "beta_MOM", "beta_HML", "r2"]
    sub_pf = sub[~sub["series"].str.contains(":")].pivot_table(
        index="window", columns="series", values="cagr").reset_index()

    lines = [
        "# LEAPS as Stock Substitutes in Long/Short Equity — Report",
        "",
        "_Generated from `results/` (see tables_manifest.json). Plan: PLAN.md. "
        "All historical option prices are model-synthesized and calibrated/validated "
        "against live chains (results/validation_live_chains.md); every limitation "
        "of PLAN §11 applies._",
        "",
        "## 1. Methods (summary)",
        "",
        "Universe, data, pricing model, frictions, selection/roll rules, sizing, "
        "and portfolio construction per PLAN.md §3-§5. Validation per PLAN §6 "
        "(results/validation_live_chains.md).",
        "",
        PROSE,
        "",
        "## 2. RQ1 — long side: deep-ITM LEAPS calls as long-stock substitutes",
        "",
        "Cross-sectional median tracking-difference decomposition (annualized; "
        "identity TD = ddiv + dfin − dcost + dborrow + resid):",
        "",
        _md(med_long[["comparison", *buckets]]),
        "",
        "See `decomposition_long.csv`, `decomposition_summary.md`, `fig_spy_lanes.png`, "
        "`fig_tracking_cross.png`.",
        "",
        PROSE,
        "",
        "## 3. RQ2 — short side: deep-ITM puts and synthetic shorts",
        "",
        _md(med_short[["comparison", *buckets]]),
        "",
        "See `decomposition_short.csv`, `fig_decomposition.png`.",
        "",
        PROSE,
        "",
        "## 4. RQ3 — portfolio level: V0-V3 momentum L/S",
        "",
        fmt(pf_sum[pf_cols]),
        "",
        "Factor attribution (FF3 + Momentum, daily, excess returns):",
        "",
        _md(attr[attr_cols], "{:.3f}"),
        "",
        "Sub-period CAGR by rate regime:",
        "",
        _md(sub_pf, "{:.1%}"),
        "",
        "See `fig_portfolio_nav.png`, `fig_drawdown.png`, `subperiods.csv`.",
        "",
        PROSE,
        "",
        "## 5. RQ4 — robustness (PLAN §7 grid)",
        "",
        f"Largest one-at-a-time swings on V1 CAGR (top 3 of 7 params):",
        "",
        _md(swings_v1, "{:.1%}"),
        "",
        f"Largest swings on V1 Sharpe:",
        "",
        _md(swings_v1_s, "{:.2f}"),
        "",
        "Full tables: `sensitivities_portfolios.csv`, `sensitivities_lanes.csv`; "
        "figures `fig_tornado_portfolios.png`, `fig_tornado_spy_call.png`.",
        "",
        PROSE,
        "",
        "## 6. Validation",
        "",
        "See `results/validation_live_chains.md` (V1 live-chain gate), unit tests "
        "(leaps_ls/validate/test_all.py), smoke tests in scripts/03_run_backtests.py.",
        "",
        PROSE,
        "",
        "## 7. Limitations",
        "",
        "Per PLAN §11, plus the deviations documented in the phase summaries "
        "(market-IV sourcing in the gate, IV cap, bankruptcy guard, split-scaled "
        "dollar frictions, dividend-projection lag, parity-rule scoping, lane "
        "rebalancing convention, $1M capital base).",
        "",
        PROSE,
    ]
    (config.ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    manifest = build_manifest()
    missing = [e["file"] for e in manifest["entries"] if not e["exists"]]
    build_report(manifest)
    print(f"tables_manifest.json written ({len(manifest['entries'])} entries)")
    if missing:
        print(f"WARNING missing files: {missing}")
    print("REPORT.md skeleton written")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
