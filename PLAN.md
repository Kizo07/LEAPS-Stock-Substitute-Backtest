# LEAPS as Stock Substitutes in Long/Short Equity — Empirical Backtest Plan

**Status:** approved working plan · **Created:** 2026-07-31
**Source idea:** `~/Documents/Deep-Research-Ideas/deep-research-report-leaps-equity-portfolio.md`

## 1. Background and research gap

The source report reviews 56 official, academic, and practitioner sources on deep in-the-money
(ITM) LEAPS calls/puts as capital-efficient substitutes for long and short stock exposure. Its
central finding: the conceptual case is well documented (long side cleaner than short side), but
**no tightly targeted empirical study tests deep-ITM LEAPS replacement of both legs of equity
long/short portfolios across a long sample, net of all relevant frictions**. The report specifies
the "minimum defensible backtest": delta-targeted strike selection, explicit roll rules, bid-ask
and slippage, dividend handling, hard-to-borrow and financing assumptions, early-exercise/
assignment scenarios for American-style options, and separate reporting for long and short legs.

This project implements exactly that backtest using **only publicly available (free) data**.

## 2. Research questions

- **RQ1 (long side):** Net of frictions, how closely does a deep-ITM LEAPS call (Δ ≈ 0.80)
  replicate a cash long stock position? Quantify tracking difference, its decomposition
  (dividends foregone vs. embedded financing benefit vs. time value vs. transaction costs),
  and capital efficiency.
- **RQ2 (short side):** Net of frictions, can (a) a deep-ITM long put (Δ ≈ −0.80) and
  (b) a synthetic short (long put + short call) substitute for a cash short sale charged
  realistic borrow fees? Which dominates, and when?
- **RQ3 (portfolio level):** In a monthly-rebalanced long/short momentum portfolio, how does
  LEAPS replacement of both legs compare to an all-stock implementation on net return, risk,
  Sharpe, drawdown, factor exposures, and return on deployed capital?
- **RQ4 (robustness):** How sensitive are RQ1–RQ3 conclusions to spread levels, borrow fees,
  delta target, tenor, roll timing, and volatility estimation?

## 3. Data sources (public only)

| Source | What | Access | Use in project |
|---|---|---|---|
| Yahoo Finance (via `yfinance`) | Daily OHLCV (unadjusted + corporate actions), dividends, splits for universe | Free API | Primary market data, 2005–present |
| Yahoo Finance option chains | **Current** live chains (strikes, bid/ask, last, IV) | Free API | Validation & calibration of the pricing model (IV/RV multiplier, spread tiers, skew slope) |
| FRED (St. Louis Fed) | `DGS3MO`, `DGS1`, `DGS2` daily Treasury yields | Free CSV (`fredgraph.csv`) | Risk-free rate curve for pricing and cash financing |
| Yahoo Finance `^VIX` | Daily VIX | Free API | Systematic IV anchor for SPY/QQQ index lanes and IV/RV scaling sanity checks |
| Ken French Data Library | Fama-French 3 factors + Momentum (daily) | Free zip/CSV | Factor exposure attribution of portfolio variants |
| Published borrow-fee literature (D'Avolio 2002; Geczy, Musto & Reed 2002; Drechsler & Drechsler 2016) | Stylized borrow-fee distribution (general collateral ≈ 0.25–0.50%/yr; small hard-to-borrow tail) | Papers cited in source report | Scenario assumptions for cash-short borrow costs (no free borrow-fee time series exists) |

**Considered and rejected (not free / not exportable):** OptionMetrics IvyDB US, CBOE DataShop,
ThetaData, ORATS, Polygon.io options (paid); QuantConnect free cloud (data usable only inside
their platform — noted as an optional external cross-check, not part of the pipeline); Kaggle
option-chain snapshots (spotty coverage, unusable for a 20-year panel).

**Consequence:** there is no free 20-year historical OPRA option-quote panel. The project
therefore prices LEAPS **synthetically** — an American-aware Black-Scholes model with discrete
dividends, fed by estimated implied volatility — and **validates/calibrates the model against
today's live chains** (Section 6). This trade-off is stated plainly in every output.

## 4. Universe and sample

- **Universe:** 33 highly liquid, long-listed, optionable US names across sectors, plus SPY and
  QQQ as index lanes:
  `AAPL MSFT GOOGL AMZN META NVDA INTC CSCO IBM ORCL TXN QCOM JPM BAC WFC GS MS V MA JNJ UNH LLY
  ABBV MRK PFE PG KO PEP WMT MCD HD XOM CVX CAT` + `SPY QQQ`.
- **Data window:** 2005-01-01 → latest available (≥ 12 months of lookback for the signal).
- **Backtest window:** 2007-01-02 → latest.
- **Known bias:** universe is selected with hindsight (today's liquid survivors). Treated as a
  stated limitation, not silently; conclusions are framed per-name and cross-sectionally, not as
  investable-performance claims.

## 5. Methodology

### 5.1 Option pricing model
- European core: Black-Scholes on dividend-escrowed spot `S* = S − PV(dividends ex-dated before
  expiry)`, using discrete actual historical/announced dividends (not a yield guess).
- American feature handled by rules (equity LEAPS are American):
  - *Calls:* before each ex-date, compare remaining time value vs. dividend; if dividend exceeds
    time value, close the option the day before ex-date and re-establish exposure the next day
    (economically equivalent to exercise-and-replace without modeling share delivery). Event
    counted and costed.
  - *Puts:* deep-ITM put early-exercise premium is small at low rates; approximate via
    Barone-Adesi-Whaley adjustment as a sensitivity; base case prices European and tracks how
    often the parity bound `P < K − S*` is violated (flagged, costed as exercise-reopen when hit).
- Short options inside synthetic lanes: assignment mirrors the same rules (short call assigned
  before ex-dates when time value < dividend).

### 5.2 Volatility estimation (IV proxy)
- Realized vol: EWMA (λ = 0.94) of daily log returns, min 63-day window, annualized.
- IV proxy per name/day: `IV = clamp(m_name, 1.00, 1.35) × RV`, where `m_name` is a per-name
  IV/RV multiplier calibrated on the live-chain validation date (median market IV ÷ EWMA RV over
  near-ATM LEAPS contracts), defaulted to the cross-sectional median when a name's chain is thin.
- Index lanes (SPY/QQQ): IV anchored to VIX (`IV = VIX/100 × term_adj`, `term_adj ≈ 1.0`,
  sensitivity 0.9–1.1).
- Skew: linear in log-moneyness, `IV(K) = IV_ATM + slope_name × ln(K/S*)`, `slope_name`
  calibrated on the validation date (cross-sectional median default), total adjustment clamped
  to ±15%.

### 5.3 Interest rates
- Daily constant-maturity Treasury curve from FRED (`DGS3MO`, `DGS1`, `DGS2`), linearly
  interpolated to option tenor; missing days forward-filled.

### 5.4 Contract selection rules
- **Tenor:** January expiry with DTE ∈ [365, 1100], closest to 730 (mirrors LEAPS listing cycle).
- **Long-stock substitute:** call with Δ closest to +0.80 (accept band [0.75, 0.85]).
- **Short-stock substitute (lane A):** put with Δ closest to −0.80 (band [−0.85, −0.75]).
- **Synthetic lane (lane B):** strike nearest the forward `S*·e^{rT}`; long synthetic = long call
  + short put; short synthetic = long put + short call, same strike/expiry.
- All selection uses information available on the trade date only (strict no-look-ahead).

### 5.5 Roll rules
- Base: at each monthly rebalance, any option leg with DTE < 180 is rolled to a fresh contract
  per Section 5.4. Sensitivities: DTE thresholds {90, 365}.
- Delta-band roll (sensitivity only): also roll if |Δ| leaves [0.60, 0.95].

### 5.6 Dividends and corporate actions
- Cash stock legs receive dividends (cash reinvested into the leg at next rebalance).
- Option legs receive no dividends; the ex-date drop enters through spot. This asymmetry is a
  first-class output (cost decomposition, Section 5.10).
- Splits: back-adjusted share counts; option contract ratio adjustments noted but not needed
  under synthetic pricing (contracts re-struck at each roll).

### 5.7 Frictions (all explicit, itemized, reported separately)
- **Commissions:** $0.65/contract (base), $0 (institutional sensitivity). Stock: $0.005/share.
- **Bid-ask spread:** pay half-spread on every option trade; half-spread modeled as
  `max($0.05, s_bucket × premium)` with `s_bucket` calibrated from live chains in moneyness ×
  tenor buckets (base tiers expected ≈ 1%/2%/3%); stress multipliers {0.5×, 1×, 2×, 3×}.
  Stock legs: 1 bp of notional.
- **Stock borrow (cash shorts):** base = general collateral 0.30%/yr on short market value,
  accrued daily; scenarios {0%, 1%, 3%}; stress mix: 25% of names at 5%/yr.
- **Financing:** cash credit at 3M T-bill − 0.25%; debit (if ever negative) at 3M + 1.50%.
- **Margin/capital accounting:** fully funded base case (premiums paid in cash, no margin loan);
  capital-efficiency reported as premium outlay vs. stock purchase cost for identical delta
  exposure (Reg-T/PM implications discussed, not simulated).

### 5.8 Sizing conventions
- Base: **delta-equivalent** — option position sized so Δ-exposure equals the stock leg's shares
  (`contracts = target_shares / (100 × |Δ|)`). Isolates cost/carry differences at equal market
  exposure.
- Sensitivity: **1:1 share-equivalent** (1 contract per 100 target shares).

### 5.9 Portfolio construction and strategy variants
- Signal: 12-1 total-return momentum, monthly, cross-sectional ranks over the stock universe
  (indices excluded from the L/S portfolio; SPY/QQQ run as separate single-instrument lanes).
- Portfolios: top-quintile long / bottom-quintile short, equal weight, monthly rebalance,
  $1 initial NAV, fully funded, dollar-neutral notional.
- Variants per leg and whole portfolio:
  - **V0 all-stock** (baseline; shorts charged borrow);
  - **V1 LEAPS replacement** (longs → 0.80Δ calls; shorts → −0.80Δ puts);
  - **V2 synthetic** (longs → synthetic long; shorts → synthetic short);
  - **V3 hybrid** (longs LEAPS calls, shorts cash) — isolates the cleaner long side.
- Single-instrument lanes for RQ1/RQ2: buy-and-hold-with-rolls on each name and on SPY/QQQ:
  stock vs. LEAPS call vs. synthetic long (long side); cash short vs. LEAPS put vs. synthetic
  short (short side).

### 5.10 Metrics and decomposition
- Per leg/variant/portfolio: CAGR, ann. vol, Sharpe, Sortino, max drawdown, turnover.
- Tracking difference vs. the stock implementation (ann., bps) and information ratio.
- **Cost decomposition (bps/yr):** spread paid, commissions, borrow, financing, dividends
  foregone (option legs), time-value decay net of financing benefit (parity carry).
- Capital usage: mean/median premium outlay vs. stock cost; implied capital released.
- Factor attribution: daily-return regression on Fama-French 3 + Momentum (alpha, betas, R²).

## 6. Validation strategy

- **V1 live-chain validation (mandatory gate):** pull today's full chains for ≥ 6 sample names +
  SPY; compare model price vs. market mid and model IV vs. market IV across strikes/tenors.
  Gate: median |price error| / mid ≤ 10% for near-ATM 1–2y contracts; report full table.
- **V2 parity/consistency:** model prices satisfy put-call parity to tolerance by construction;
  unit tests assert it (guards implementation bugs).
- **V3 unit tests:** BS vs. known reference values; escrowed-dividend spot math; EWMA vol vs.
  closed-form two-period case; accounting conservation (cash + positions = NAV) every day of a
  smoke backtest; no-look-ahead test (signals/prices shifted by one day must change results).
- **V4 economic sanity:** SPY LEAPS-call replacement tracking difference ≈ (financing benefit −
  dividend yield − costs) within tolerance; deep-ITM call Δ P&L attribution R² vs. stock ≥ 0.95.

## 7. Sensitivity grid

One-at-a-time around base case: spread {0.5×, 1×, 2×, 3×}; borrow {0, 30, 100, 300} bps/yr;
delta target {0.70, 0.80, 0.90}; tenor target {1y, 2y}; roll DTE {90, 180, 365}; IV multiplier
{0.9, 1.0, 1.1}×; sizing {delta-equiv, 1:1}. Reported as tornado-style tables for RQ1–RQ3
headline metrics.

## 8. Technical architecture

Python 3.13, project-local venv (`.venv/`, `--system-site-packages` reusing system
numpy/pandas/scipy/matplotlib; `yfinance` installed in-venv). All raw data cached under `data/`
on first download; every script re-runnable offline from cache. Strict no-look-ahead: all
selection/signal functions receive data truncated to the trade date.

```
LEAPS-Stock-Substitute-Backtest/
├── PLAN.md                     # this file
├── README.md                   # how to run
├── requirements.txt
├── leaps_ls/
│   ├── config.py               # universe, dates, all parameters (single source of truth)
│   ├── data/
│   │   ├── cache.py            # parquet/csv cache helpers
│   │   ├── yahoo.py            # OHLCV, dividends, splits, live chains (yfinance)
│   │   ├── fred.py             # DGS3MO/DGS1/DGS2 → daily curve interp
│   │   └── french.py           # FF3 + Momentum daily factors
│   ├── pricing/
│   │   ├── black_scholes.py    # BS price + greeks (vectorized)
│   │   ├── dividends.py        # escrowed-dividend adjusted spot S*
│   │   ├── baw.py              # Barone-Adesi-Whaley American approx (sensitivity)
│   │   ├── exercise.py         # ex-date exercise/assignment decision rules
│   │   └── vol.py              # EWMA RV → IV proxy, VIX anchor, skew
│   ├── instruments/
│   │   ├── selector.py         # delta-targeted LEAPS selection (Section 5.4)
│   │   └── rolls.py            # roll scheduling (Section 5.5)
│   ├── frictions/
│   │   ├── spreads.py          # bucketed half-spread model + calibration from live chains
│   │   ├── borrow.py           # borrow-fee scenarios
│   │   └── financing.py        # cash credit/debit rates
│   ├── portfolio/
│   │   ├── signals.py          # 12-1 momentum
│   │   ├── engine.py           # daily accounting backtester (cash, positions, NAV, costs)
│   │   └── variants.py         # V0–V3 builders, single-instrument lanes
│   ├── analysis/
│   │   ├── metrics.py          # Section 5.10 metrics + cost decomposition
│   │   ├── attribution.py      # FF3+Momentum regressions
│   │   └── plots.py            # equity curves, tracking-diff, tornado tables→figures
│   └── validate/
│       ├── live_chains.py      # V1 gate
│       └── test_all.py         # V2/V3 tests (plain asserts, run as script)
├── scripts/
│   ├── 01_download_data.py     # cache all raw data
│   ├── 02_validate_pricing.py  # V1–V4 gates, writes results/validation_*.md
│   ├── 03_run_backtests.py     # single-instrument lanes + V0–V3 portfolios
│   ├── 04_sensitivities.py     # Section 7 grid
│   └── 05_make_report.py       # assemble REPORT.md from results/
├── data/                       # cached raw data (not hand-edited)
├── results/                    # tables (.csv) + figures (.png) + validation reports
└── REPORT.md                   # final findings (RQ1–RQ4, limitations)
```

**Conventions:** prices in actual dollars (unadjusted OHLC) with dividends/splits handled
explicitly; all returns daily; costs booked as separate ledger lines in the engine so the
decomposition is exact, not estimated; every figure/table regenerable from scripts.

## 9. Execution phases and checkpoints

| Phase | Content | Checkpoint |
|---|---|---|
| 0 | venv, deps, folder scaffold | `python -c "import yfinance"` OK in `.venv` |
| 1 | data pipeline (Yahoo/FRED/French, cached) | `01_download_data.py` completes; cache files present; date ranges & missing-data report |
| 2 | pricing + vol + frictions models; V1–V4 validation | `02_validate_pricing.py` passes gates; `results/validation_live_chains.md` written |
| 3 | selection/rolls + engine + variants | smoke backtest conserves NAV; V0–V3 run end-to-end |
| 4 | full experiments + sensitivity grid | `results/` tables & figures complete |
| 5 | REPORT.md (findings, decomposition, limitations) | report answers RQ1–RQ4 with numbers |

## 10. Deliverables

1. This plan (`PLAN.md`).
2. Reproducible codebase + cached public data (`data/`).
3. Validation report (live-chain model error tables).
4. Results: per-lane and per-variant tables, cost decomposition, factor attribution, figures.
5. `REPORT.md`: empirical answers to RQ1–RQ4, with every limitation stated.

## 11. Limitations and threats to validity (stated up front)

- Historical option prices are **model-synthesized**, not market quotes; conclusions are
  conditional on the vol/skew/spread calibration, which is anchored to today's live chains and
  stressed in sensitivity.
- Universe suffers survivorship/hindsight bias by construction.
- Borrow fees are literature-based scenarios, not name-specific time series.
- Early exercise/assignment modeled by decision rules, not full American pricing (BAW check as
  sensitivity); deep-ITM put exercise premium may be understated in high-rate sub-periods.
- Skew is linear-in-log-moneyness; smiles/term-structure curvature not modeled.
- No margin calls, forced liquidations, or Reg SHO locate failures simulated (fully funded base).
- Taxes ignored.

## 12. Traceability to the source report's "minimum defensible backtest"

| Report requirement | Plan element |
|---|---|
| Delta-targeted strike selection | §5.4 (Δ ≈ ±0.80, bands, sensitivities) |
| Explicit roll rules | §5.5 |
| Bid-ask and slippage | §5.7 (calibrated bucket model, stress multipliers) |
| Dividend handling | §5.6 + cost decomposition §5.10 |
| Hard-to-borrow and financing assumptions | §5.7 (borrow scenarios, financing spreads) |
| Early-exercise/assignment scenarios | §5.1 (ex-date rules, BAW sensitivity, parity-bound flags) |
| Separate long/short leg reporting | §5.9–5.10 (per-leg lanes and decomposition) |
| Portfolio-level, net-of-friction evidence | RQ3, V0–V3, §5.10 |
