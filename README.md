# LEAPS as Stock Substitutes in Long/Short Equity — Empirical Backtest

Implements the research plan in [`PLAN.md`](PLAN.md), which operationalizes the literature
review in `~/Documents/Deep-Research-Ideas/deep-research-report-leaps-equity-portfolio.md`
using only publicly available data (Yahoo Finance, FRED, Ken French library).

## Setup

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements.txt
```

## Run (in order)

```bash
scripts/run_all.sh            # full pipeline: data -> validate -> backtests -> sensitivities -> report
scripts/run_all.sh backtests  # or resume from a single stage (data|validate|backtests|sensitivities|report)
```

Stage by stage:

```bash
.venv/bin/python scripts/01_download_data.py     # cache all raw data into data/
.venv/bin/python scripts/02_validate_pricing.py  # V1–V4 validation gates
.venv/bin/python scripts/03_run_backtests.py     # single-instrument lanes + V0–V3 portfolios
.venv/bin/python scripts/04_sensitivities.py     # sensitivity grid
.venv/bin/python scripts/05_make_report.py       # assemble REPORT.md (+ environment stamp in tables_manifest.json)
```

## Tests

Offline and fast (no network; the engine tests use synthetic deterministic data):

```bash
.venv/bin/python leaps_ls/validate/test_all.py    # pricing/data/frictions unit tests
.venv/bin/python leaps_ls/validate/test_engine.py # engine accounting & event mechanics
python -m pytest leaps_ls/validate -q             # everything at once
```

Engine coverage: daily NAV reconciliation for V0–V3, call-exercise reopen, put-parity
reopen, expiry settlement, ruin/writeoff reconciliation, split-scaled dollar frictions,
no-look-ahead signal-shift check, momentum dollar-neutrality.

All parameters live in `leaps_ls/config.py` (validated at engine construction —
inconsistent settings fail fast). Raw downloads are cached under `data/`; results
(tables, figures, validation reports) land in `results/`.

See [`ENHANCEMENTS.md`](ENHANCEMENTS.md) for the enhancement log and deliberately
deferred items (day-count unification, vectorized pricing, announced-dividend ingestion,
GARCH vol alternative).

### Notable fixes (2026-08)

- BAW American-put premium used the wrong normal-CDF argument (`1−N(d1)` instead of
  `1−N(−d1)`), understating deep-ITM American puts by up to ~15%.
- Dividend-frequency snapping tied 3 trailing ex-dates to semiannual instead of quarterly,
  halving projected dividends for mid-year as-of dates.
- `Engine.run(end=...)` crashed on non-trading end dates (weekends/holidays).
