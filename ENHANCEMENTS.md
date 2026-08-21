# Enhancement Plan — LEAPS Stock-Substitute Backtest

Scope note: every item below is verifiable offline (cached data / synthetic fixtures),
keeps the PLAN.md conventions intact, and avoids forcing a full pipeline re-run.
Status column reflects the implementation pass in this repository.

| # | Area | Enhancement | Why | Status |
|---|------|-------------|-----|--------|
| P1 | Testing | Synthetic deterministic `MarketData` fixture (no Yahoo/FRED/network) + engine unit tests: NAV reconciliation for all variants, call-exercise reopen, put-parity reopen, expiry settlement, ruin/writeoff, split-scaled dollar frictions | Engine accounting was only covered by script 03 smoke tests that require cached parquet data; the suite must run anywhere, fast | done |
| P2 | Testing | Root `conftest.py` so `python -m pytest` collects `leaps_ls/validate/test_all.py` and the new engine tests out of the box | Tests were runnable only as a hand-written script | done |
| P3 | Robustness | `config.validate()` sanity gate (delta target inside roll band, tenor ordering, quintile ≥ 2, sizing names valid, spread-tier monotonicity, EWMA bounds) called by `Engine.__init__`, scripts 03/04, and the test runner | Bad sensitivity-grid values currently fail deep inside runs with opaque errors | done |
| P4 | Correctness | `_exercise_checks`: measure remaining time value from tomorrow's ex-date instead of today (matches the `exercise.py` docstring "T_rem from ex-date") | One-day convention inconsistency between rule and evaluation | done |
| P5 | Reproducibility | Pin `requirements.txt`; stamp Python + package versions into `results/tables_manifest.json` (script 05) | yfinance API churn broke this env once already; result files should record what produced them | done |
| P6 | Ops | `scripts/run_all.sh` orchestrator for phases 1→5 with per-stage gating and clear failure output | Pipeline stages were only documented implicitly across README/PLAN | done |
| P7 | Analysis | Add Sortino ratio and Calmar ratio to `metrics.headline` (additive columns; downstream readers use named columns so nothing breaks) | Downside-risk view of option-replacement strategies was missing from summaries | done |
| P8 | Docs | README updates: how to run tests (script + pytest), pipeline usage, summary of the 2026-08 bug fixes and enhancements | Keep entry points discoverable | done |

## Deliberately deferred (documented, not implemented here)

- **Day-count unification** (selector/live-chains use ACT/365, engine marking uses ACT/365.25).
  Changing it silently invalidates cached vol/spread calibrations; keep as a flagged
  convention until the next full recalibration cycle.
- **Vectorized pricing path** for the engine hot loop — large refactor, modest payoff at
  current universe size (~35 tickers); revisit if the universe grows.
- **Announced-dividend ingestion** to replace the ex-date projection approximation —
  needs a paid data source; out of scope for the free-data constraint.
- **GARCH(1,1) vol alternative** alongside EWMA — would require recalibrating the IV
  multiplier against live chains.

## Verification performed

- `python leaps_ls/validate/test_all.py` → all unit tests pass (incl. new BAW/dividend regressions).
- `python -m pytest leaps_ls/validate -q` → full offline suite green on a clean machine path.
- Engine smoke over synthetic data reconciles to < 1e-9 for V0–V3 incl. ruin path.
