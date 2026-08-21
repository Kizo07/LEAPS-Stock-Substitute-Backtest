#!/usr/bin/env python
"""Phase 2 validation driver (PLAN §6): V2/V3 unit tests, V4-style parity sanity, V1 live gate."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from leaps_ls.pricing import black_scholes as bs, dividends as dvd  # noqa: E402
from leaps_ls.validate import live_chains, test_all  # noqa: E402


def v4_synthetic_parity(verbose: bool = True) -> bool:
    """V4-style sanity: model prices on a synthetic escrowed-dividend example satisfy parity.

    S=150, K=120, T=1.5y, flat 4% rate, sigma=28%, two projected $1.20 dividends:
    C(S*) - P(S*) must equal S* - K e^{-rT} exactly (by construction of the model),
    and both prices must be arbitrage-consistent (call <= S*, put <= K e^{-rT}).
    """
    S, K, T, r, sig = 150.0, 120.0, 1.5, 0.04, 0.28
    schedule = dvd.project_dividends(
        _synthetic_hist(), as_of_date="2026-07-31", horizon_years=2.0
    )
    s_star = dvd.adjusted_spot(S, schedule, lambda d, t: r, T)
    c = float(bs.call_price(s_star, K, T, r, sig))
    p = float(bs.put_price(s_star, K, T, r, sig))
    parity_err = abs((c - p) - (s_star - K * np.exp(-r * T)))
    lower_c = s_star - K * np.exp(-r * T)  # escrowed-spot lower bound for the call
    ok = (
        parity_err < 1e-8
        and lower_c - 1e-9 <= c <= s_star
        and 0.0 <= p <= K * np.exp(-r * T)
    )
    if verbose:
        print(f"  synthetic example: S=150, K=120, T=1.5, r=4%, sigma=28%, "
              f"{len(schedule)} projected dividends")
        print(f"  S* = {s_star:.4f}; C = {c:.4f}; P = {p:.4f}; parity |err| = {parity_err:.2e}")
    return bool(ok)


def _synthetic_hist():
    """A quarterly $1.20 payer for the parity example."""
    import pandas as pd

    dates = pd.bdate_range("2025-01-02", "2026-07-31")
    divs = pd.Series(0.0, index=dates)
    for d in ("2025-09-15", "2025-12-15", "2026-03-16", "2026-06-15"):
        divs[pd.Timestamp(d)] = 1.20
    return pd.DataFrame({"Close": 150.0, "Dividends": divs, "Stock Splits": 0.0}, index=dates)


def main() -> int:
    results: dict[str, bool] = {}

    print("=" * 72)
    print("V2/V3 — unit tests (BS, parity, escrowed dividends, EWMA, frictions)")
    print("=" * 72)
    results["unit_tests"] = test_all.run_all()

    print()
    print("=" * 72)
    print("V4-style — synthetic escrowed-dividend parity sanity")
    print("=" * 72)
    try:
        results["synthetic_parity"] = v4_synthetic_parity()
        print("  ->", "PASS" if results["synthetic_parity"] else "FAIL")
    except Exception as exc:  # noqa: BLE001
        results["synthetic_parity"] = False
        print(f"  -> FAIL ({exc})")

    print()
    print("=" * 72)
    print("V1 — live-chain gate (median |model - mid|/mid <= 10% on 1-2y near-ATM)")
    print("=" * 72)
    try:
        summary = live_chains.run_validation()
        results["live_gate"] = summary["gate_pass"]
    except Exception as exc:  # noqa: BLE001
        results["live_gate"] = False
        print(f"V1 gate ERROR: {exc}")

    print()
    print("=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)
    for name, ok in results.items():
        print(f"  {name:<18} {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
