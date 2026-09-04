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

## Research dossier (`research/`)

Independent investigation run alongside (not inside) the engine — 8 notes,
reproducible calculators, and result CSVs, snapshotted 2026-09-02 against
chains/Treasury curve as of 2026-07-31:

| Note | Question | Answer in one line |
|---|---|---|
| `01_parity_identity` | What is a "stock substitute"? | Put-call parity: you pay American, collect European (−4.3%/yr forfeited early-exercise premium) |
| `02_embedded_financing` | What financing is embedded in quotes? | 61–131 bp over Treasuries on the deferred strike |
| `03_capm` | Does CAPM hold for LEAPS? | Instantaneously exact (residual 1e-15); empirically no — TM γ = +6.8, t = 15.4 |
| `04_structural_risks` | What breaks structurally? | Static book is a short strangle: −9.3% of capital over 2 yrs inside [−27%, +64%] |
| `05_risk_taxonomy` | Full risk list? | 12 quantified risks; drift, short-strangle, short-leg capital, liquidity dominate |
| `06_verdict` | Bottom line? | Not cheaper same-book — different strategy: V1 −30.1% vs V0 −6.7% CAGR, turnover the largest term |
| `07_monte_carlo` | Attribution in a CAPM-true world? | Simulated P&L decomposition, E1–E8 battery incl. ruin + sensitivity sweeps |
| `08_all_combinations` | Does any variant survive? | Full factorial over the design space |

Headline: the LEAPS book is long convexity and long volatility with
state-dependent beta (swings 0.77 up- vs down-market) and −36.4% FF3+MOM
alpha — a linear factor model cannot absorb it. Reproduce any note from
`research/calcs/` (closed-form/binomial maths verified against
Black-Scholes) or the project's cached `data/`; result tables in
`research/results/`.

## Findings: LEAPS book vs stock book

The engine backtest (2007–2026, 36 names, monthly-rebalanced 2-yr LEAPS)
and the independent dossier agree on the shape of the answer, and the
Monte Carlo + 810-world factorial say which parts are policy vs physics.

**Backtest head-to-head (project engine).** V0 = stock long/short,
V1 = LEAPS long/short on the same names and signals:

| | Stock book (V0) | LEAPS book (V1) | Gap |
|---|---|---|---|
| CAGR | −6.7%/yr | −30.1%/yr | **−23.4 pp** |
| Annualised vol | 26.0% | 29.1% | +3.1 pp |
| Sharpe | −0.14 | −1.09 | −0.95 |
| FF3+MOM alpha | −8.7% (t −2.1) | **−36.4%** (t −6.8) | −27.7 pp |
| Beta, down days | −0.13 | **−0.64** | regime flip |
| Beta, up days | −0.34 | **+0.13** | regime flip |
| Treynor-Mazuy γ | −1.3 | **+6.8** (t 15.4) | convex |
| Loading on ΔVIX | −0.012 | **+0.024** (t 6.2) | long vol |

Read the beta rows first: the stock book is roughly market-neutral in both
regimes; the LEAPS book is net **short** after selloffs and net **long**
after rallies — positioned worst exactly at turning points. That drift,
plus long vega into a negative variance premium, is why a linear factor
model prints −36% alpha: beta is not a sufficient statistic for this book.

**Where the −23.4 pp goes (per year, per unit of book capital).**
Forfeited early-exercise premium −4.3% (you pay American, collect
European); variance risk premium −0.7 to −1.7%; embedded financing spread
−0.4 to −1.1% (61–131 bp over Treasuries measured from live chains);
dividend give-up net of carry ≈ −1.5% at sample-average rates (rate
dependent — at today's 4.2% short rate the carry largely offsets it on
sub-1.9% yielders); and monthly re-striking turnover, the largest single
term (≈82%/yr of spread cost).

**Structure vs frictions (Monte Carlo, 4,000 paths × 20 yrs, CAPM true by
construction).** With all frictions off, the books differ by −0.03%/yr —
nothing. Of the −9.8%/yr simulated gap: bid-ask/turnover 6.67, unexercised
American premium 4.11, dividends 0.27; drift/gamma/theta net out. Two
corrections to the analytic notes: the early-exercise premium and drift
are *policy* costs (never-exercise, roll-monthly), not physics — policies
can change; and Δ ≥ 0.90 is wrong at realistic spreads (deeper delta =
more premium for the spread to eat): Δ=0.70 → −6.4%/yr vs Δ=0.95 →
−18.1%/yr. Δ ≈ 0.75–0.80 is the sweet spot.

**How much the answer moves (810-world factorial).** LEAPS-minus-stock gap:
median −4.7%/yr, mean −7.4, P(win) 18%, P(dominated < −10%) 27%. Two
parameters decide it — roll horizon (range 17.7 pp; 24-mo rolls −1.3 vs
1-mo −19.0) and half-spread (10.9 pp), 44% of variance with their
interaction. Funding spread, dividends, and stock-picking strategy are
noise (ΔR² < 0.002). Churning at 200 bp spreads costs 21.6 pp/yr; at 12-mo
rolls it costs 2.9.

**When substitution is defensible:** capital-constrained accounts with no
margin access, non-dividend names, low-vol names, Δ ≈ 0.75–0.80, held
12–24 months — not the monthly-rebalanced 36-name backtest, which is
dominated. The static structure is otherwise a short strangle (−9.3% of
capital over two years inside [−27%, +64%]), the short leg needs 70% cash
vs 20–50% stock margin, and deep-ITM quotes (210–286 bp wide, 0–8 lots)
often don't trade at all.

### Notable fixes (2026-08)

- BAW American-put premium used the wrong normal-CDF argument (`1−N(d1)` instead of
  `1−N(−d1)`), understating deep-ITM American puts by up to ~15%.
- Dividend-frequency snapping tied 3 trailing ex-dates to semiannual instead of quarterly,
  halving projected dividends for mid-year as-of dates.
- `Engine.run(end=...)` crashed on non-trading end dates (weekends/holidays).
