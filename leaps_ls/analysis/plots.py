"""Figures for the LEAPS-substitute study (PLAN §5.10 outputs).

Reads results/*.parquet/csv plus the cached Yahoo histories (equal-weight
benchmark) and writes 150-dpi PNGs to results/. Matplotlib only, no seaborn.
Run directly: python leaps_ls/analysis/plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from leaps_ls import config  # noqa: E402
from leaps_ls.data import yahoo  # noqa: E402
from leaps_ls.analysis import metrics  # noqa: E402

DPI = 150
VARIANT_COLORS = {"V0": "#444444", "V1": "#1f77b4", "V2": "#d62728", "V3": "#2ca02c"}
BUCKET_COLORS = {"ddiv": "#d62728", "dfin": "#2ca02c", "dcost": "#ff7f0e",
                 "dborrow": "#9467bd", "resid": "#7f7f7f"}
BUCKET_LABELS = {"ddiv": "dividends foregone", "dfin": "financing benefit",
                 "dcost": "spread+comm (drag)", "dborrow": "borrow saved",
                 "resid": "vol prem+theta+model"}


def _out(name: str) -> Path:
    return config.RESULTS_DIR / name


def _save(fig: plt.Figure, name: str) -> Path:
    path = _out(name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def ew_benchmark() -> pd.Series:
    """Equal-weight long-only total-return index of the stock universe (context)."""
    tris = {}
    for tk in config.STOCK_UNIVERSE:
        h = yahoo.load_history(tk)
        tris[tk] = yahoo.total_return_series(h)
    tri = pd.DataFrame(tris).sort_index().ffill()
    tri = tri.loc[config.BACKTEST_START:]
    ew = tri.pct_change().mean(axis=1)
    ew = (1.0 + ew.fillna(0.0)).cumprod()
    return ew / ew.iloc[0]


def fig_portfolio_nav(pf: pd.DataFrame, lanes: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for v in pf.columns:
        ax.plot(pf.index, pf[v], label=v, color=VARIANT_COLORS[v], lw=1.2)
    spy = lanes["SPY:stock"].dropna()
    ax.plot(spy.index, spy / spy.iloc[0], label="SPY (stock)", color="#888888",
            lw=1.0, ls="--")
    ew = ew_benchmark()
    ax.plot(ew.index, ew, label="EW universe (long-only)", color="#bbbbbb", lw=1.0, ls=":")
    ax.set_yscale("log")
    ax.set_title("Momentum L/S portfolio variants vs context benchmarks (log NAV, start = 1)")
    ax.set_ylabel("NAV (log scale)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, "fig_portfolio_nav.png")


def fig_spy_lanes(lanes: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    for lane, c in zip(("stock", "call", "synth_long"), ("#444444", "#1f77b4", "#2ca02c")):
        s = lanes[f"SPY:{lane}"].dropna()
        axes[0].plot(s.index, s / s.iloc[0], label=lane, color=c, lw=1.1)
    axes[0].set_yscale("log")
    axes[0].set_title("SPY long lanes (log NAV)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    for lane, c in zip(("short_stock", "put", "synth_short"), ("#444444", "#d62728", "#9467bd")):
        s = lanes[f"SPY:{lane}"].dropna()
        axes[1].plot(s.index, s / s.iloc[0], label=lane, color=c, lw=1.1)
    axes[1].set_yscale("log")
    axes[1].set_title("SPY short lanes (log NAV)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    return _save(fig, "fig_spy_lanes.png")


def fig_tracking_cross(long_df: pd.DataFrame, short_df: pd.DataFrame) -> Path:
    df = pd.concat([long_df, short_df])
    comps = ["call vs stock", "synth_long vs stock",
             "put vs short_stock", "synth_short vs short_stock"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, comp in zip(axes.flat, comps):
        g = df[(df["comparison"] == comp) & (~df["ticker"].isin(["MEDIAN", "MEAN"]))]
        g = g.sort_values("td_ann")
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in g["td_ann"]]
        ax.barh(g["ticker"], g["td_ann"] * 100.0, color=colors, alpha=0.85)
        med = g["td_ann"].median() * 100.0
        ax.axvline(med, color="k", ls="--", lw=0.9)
        ax.set_title(f"{comp}  (median {med:+.1f}%/yr)", fontsize=9)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(alpha=0.25, axis="x")
    fig.suptitle("Annualized tracking difference vs stock implementation, per ticker (%/yr)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "fig_tracking_cross.png")


def fig_decomposition(long_df: pd.DataFrame, short_df: pd.DataFrame) -> Path:
    df = pd.concat([long_df, short_df])
    comps = ["call vs stock", "put vs short_stock"]
    buckets = ["ddiv", "dfin", "dcost", "dborrow", "resid"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ypos = np.arange(len(comps))
    for y, comp in zip(ypos, comps):
        row = df[(df["comparison"] == comp) & (df["ticker"] == "MEDIAN")].iloc[0]
        left = 0.0
        for b in buckets:
            # plot signed contributions: TD = ddiv + dfin - dcost + dborrow + resid
            v = float(row[b]) * 100.0 * (-1.0 if b == "dcost" else 1.0)
            ax.barh(y, v, left=left, color=BUCKET_COLORS[b], alpha=0.9, height=0.55,
                    label=BUCKET_LABELS[b] if comp == comps[0] else None)
            left += v
        ax.plot([float(row["td_ann"]) * 100.0], [y], marker="D", color="k", ms=6)
    ax.set_yticks(ypos, comps)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("annualized contribution to tracking difference (%/yr)")
    ax.set_title("Median tracking-difference decomposition (diamond = total TD)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25, axis="x")
    return _save(fig, "fig_decomposition.png")


def _tornado(ax: plt.Axes, ranges: pd.DataFrame, base_val: float, title: str,
             xlabel: str) -> None:
    """ranges: DataFrame[param, lo, hi] sorted by span."""
    ranges = ranges.sort_values("span")
    y = np.arange(len(ranges))
    ax.barh(y, (ranges["hi"] - ranges["lo"]), left=ranges["lo"], color="#1f77b4",
            alpha=0.75, height=0.6)
    ax.axvline(base_val, color="k", ls="--", lw=1.0)
    ax.set_yticks(y, [f"{p}" for p in ranges["param"]])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25, axis="x")


def fig_tornado_portfolios(pf: pd.DataFrame) -> Path:
    variant, metric = "V1", "sharpe"
    base_val = float(pf[(pf["param"] == "base") & (pf["variant"] == variant)][metric].iloc[0])
    rows = []
    for param in pf["param"].unique():
        if param == "base":
            continue
        sub = pf[(pf["param"] == param) & (pf["variant"] == variant)][metric]
        lo, hi = min(sub.min(), base_val), max(sub.max(), base_val)  # range includes base
        rows.append({"param": param, "lo": lo, "hi": hi, "span": hi - lo})
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _tornado(ax, pd.DataFrame(rows), base_val,
             f"{variant} portfolio {metric} — one-at-a-time sensitivity (dashed = base)",
             "Sharpe ratio")
    return _save(fig, "fig_tornado_portfolios.png")


def fig_tornado_spy_call(lanes: pd.DataFrame) -> Path:
    sub0 = lanes[(lanes["ticker"] == "SPY") & (lanes["lane"] == "call")]
    base_val = float(sub0[sub0["param"] == "base"]["td_ann"].iloc[0]) * 100.0
    rows = []
    for param in sub0["param"].unique():
        if param == "base":
            continue
        sub = sub0[sub0["param"] == param]["td_ann"] * 100.0
        lo, hi = min(sub.min(), base_val), max(sub.max(), base_val)  # range includes base
        rows.append({"param": param, "lo": lo, "hi": hi, "span": hi - lo})
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    _tornado(ax, pd.DataFrame(rows), base_val,
             "SPY call-lane tracking difference — sensitivity (dashed = base)",
             "annualized tracking difference (%/yr)")
    return _save(fig, "fig_tornado_spy_call.png")


def fig_drawdown(pf: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for v in pf.columns:
        dd = pf[v] / pf[v].cummax() - 1.0
        ax.plot(dd.index, dd * 100.0, label=v, color=VARIANT_COLORS[v], lw=1.0)
    ax.set_title("Drawdown curves — portfolio variants")
    ax.set_ylabel("drawdown (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, "fig_drawdown.png")


def main() -> None:
    pf = pd.read_parquet(config.RESULTS_DIR / "nav_portfolios.parquet")
    lanes = pd.read_parquet(config.RESULTS_DIR / "nav_lanes.parquet")
    long_df = pd.read_csv(config.RESULTS_DIR / "decomposition_long.csv")
    short_df = pd.read_csv(config.RESULTS_DIR / "decomposition_short.csv")
    sens_pf = pd.read_csv(config.RESULTS_DIR / "sensitivities_portfolios.csv")
    sens_ln = pd.read_csv(config.RESULTS_DIR / "sensitivities_lanes.csv")

    made = [
        fig_portfolio_nav(pf, lanes),
        fig_spy_lanes(lanes),
        fig_tracking_cross(long_df, short_df),
        fig_decomposition(long_df, short_df),
        fig_tornado_portfolios(sens_pf),
        fig_tornado_spy_call(sens_ln),
        fig_drawdown(pf),
    ]
    for p in made:
        print(f"  wrote {p.name}")


if __name__ == "__main__":
    main()
