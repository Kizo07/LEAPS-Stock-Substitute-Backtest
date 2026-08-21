#!/usr/bin/env python
"""Phase 1: download and cache all raw data; write results/data_coverage.csv."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from leaps_ls import config  # noqa: E402
from leaps_ls.data import cache, fred, french, yahoo  # noqa: E402


def _coverage_row(ticker: str, df: pd.DataFrame, status: str = "ok", note: str = "") -> dict:
    div = df["Dividends"] if "Dividends" in df.columns else pd.Series(dtype=float)
    spl = df["Stock Splits"] if "Stock Splits" in df.columns else pd.Series(dtype=float)
    gaps = df.index.to_series().diff().dt.days.dropna()
    return {
        "ticker": ticker,
        "status": status,
        "first_date": df.index.min().date() if len(df) else None,
        "last_date": df.index.max().date() if len(df) else None,
        "n_rows": len(df),
        "n_dividends": int((div > 0).sum()),
        "n_splits": int((spl > 0).sum()),
        "max_gap_days": int(gaps.max()) if len(gaps) else None,
        "n_gaps_gt10d": int((gaps > 10).sum()),
        "note": note,
    }


def main() -> int:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tickers = config.UNIVERSE + [config.VIX_TICKER]
    rows: list[dict] = []
    failed: list[str] = []
    for i, tk in enumerate(tickers, start=1):
        try:
            df = yahoo.fetch_history(tk)  # retry-once inside; cached immediately
            rows.append(_coverage_row(tk, df))
            print(
                f"[{i}/{len(tickers)}] {tk}: {len(df)} rows "
                f"{df.index.min().date()} -> {df.index.max().date()}"
            )
        except Exception as exc:  # noqa: BLE001 - record and continue per plan
            failed.append(tk)
            rows.append(
                {
                    "ticker": tk, "status": "failed", "first_date": None, "last_date": None,
                    "n_rows": 0, "n_dividends": 0, "n_splits": 0, "max_gap_days": None,
                    "n_gaps_gt10d": None, "note": str(exc)[:200],
                }
            )
            print(f"[{i}/{len(tickers)}] {tk}: FAILED after retry: {exc}")
        time.sleep(0.5)

    fred_df = fred.fetch_curve()
    print(f"FRED curve: {len(fred_df)} days {fred_df.index.min().date()} -> {fred_df.index.max().date()}")
    rows.append(_coverage_row("FRED(DGS3MO/1/2)", fred_df))
    fr_df = french.fetch_factors()
    print(f"French factors: {len(fr_df)} days {fr_df.index.min().date()} -> {fr_df.index.max().date()}")
    rows.append(_coverage_row("FRENCH(FF3+MOM)", fr_df))

    cov = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "data_coverage.csv"
    cov.to_csv(out, index=False)
    print(f"\nCoverage report written to {out}\n")
    print(cov.to_string(index=False))
    if failed:
        print(f"\nWARNING: failed tickers (recorded, continuing): {failed}")
    else:
        print("\nAll tickers downloaded and cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
