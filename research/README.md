# Research: LEAPS-implemented long/short vs a stock long/short

Independent investigation run alongside (not inside) the main `LEAPS-Stock-Substitute-Backtest`
engine. Everything here is reproducible from `calcs/` and uses either the project's own cached
market data or closed-form / binomial mathematics verified against Black-Scholes.

**Date:** 2026-09-02 · **Data snapshot:** chains and Treasury curve as of 2026-07-31 (project `data/`)

---

## Headline answers

**1. What is the empirical difference between a stock L/S book and a LEAPS L/S book?**

| | Stock L/S (V0) | LEAPS L/S (V1) |
|---|---|---|
| CAGR (project backtest, 2007-2026) | −6.7% | −30.1% |
| Annualised vol | 26.0% | 29.1% |
| Sharpe | −0.14 | −1.09 |
| FF3+MOM alpha (ann.) | −8.7% (t = −2.1) | −36.4% (t = −6.8) |
| Market beta on down days | −0.13 | **−0.64** |
| Market beta on up days | −0.34 | **+0.13** |
| Treynor-Mazuy curvature γ | −1.3 | **+6.8 (t = 15.4)** |
| Loading on ΔVIX | −0.012 | **+0.024 (t = 6.2)** |

The LEAPS book is not "the same strategy with cheaper capital". It is a different asset: it is
long convexity and long volatility, its market beta is strongly state-dependent, and it earns a
large negative alpha that a linear factor model cannot absorb.

**2. Does CAPM hold?**

- **Instantaneously, exactly.** Under Black-Scholes and CAPM for the underlying, the identity
  `E[dC]/C·dt = r + Ω·(μ_total − r)` with `Ω = S·Δ/C` holds to machine precision
  (verified: max residual 1.1e-15, `03_capm.md`).
- **Over any tradeable horizon, for deep-ITM LEAPS, nearly.** Regression beta is within
  +0.1% (1 day) to +4.3% (63 days) of the instantaneous beta, and the spurious alpha is
  ≤ 0.6%/yr. Deep-ITM LEAPS are *close to linear*, which is the one thing in their favour.
- **Once the variance risk premium is real, no.** Buying at IV and realising RV costs
  −0.7%/yr of notional at Δ≈0.86 with a 2-vol-point VRP (−1.7%/yr at 5 points).
- **Empirically, decisively no.** V1 has a Treynor-Mazuy γ of +6.8 with t = 15.4: a linear
  beta is not a sufficient statistic for its risk. Its beta swings by 0.77 between up and
  down markets.

**3. What are the risks?**

Twelve of them, quantified, in `05_risk_taxonomy.md`. The four that dominate:

1. **Exposure drift** — the book is net **short** after selloffs and net **long** after rallies.
   On a −30% market the net exposure is −0.43 to −0.88 (per $1 of a leg) depending on remaining
   tenor. Positioned worst exactly at the turning point.
2. **The static structure is a short strangle.** Holding a Δ=0.80 call plus a Δ=−0.80 put to
   expiry pays a *constant* `K_put − K_call` for any `S_T` between the strikes. You pay 1.039 of
   capital to receive 0.943 with certainty in that range: **−9.3% of capital over two years**,
   and you only profit outside `S_T ∈ [−27%, +64%]`.
3. **The short side kills the capital-efficiency case.** American put maths (which is the correct
   maths) puts the short leg at **70% of its exposure in cash**, versus 20–50% margin for a cash
   short. The LEAPS book needs **1.04** of capital per $1 long + $1 short delta — 69% of Reg-T
   stock capital, but **173–260% of portfolio-margin capital**.
4. **Liquidity.** Median quoted bid-ask on the deep-ITM LEAPS in the cached chains is
   **210–286 bps of mid**, p90 545–688 bps. Deep strikes have open interest of 0–8 contracts and
   quote *below intrinsic*. The instruments this strategy needs do not trade.

**4. Does LEAPS substitution ever make sense?**

Yes, narrowly — see `06_verdict.md` and `07_monte_carlo.md`. The defensible cases are:
capital-constrained accounts that cannot use margin at all, non-dividend-paying names, low-volatility
names, Δ ≈ 0.75–0.80, and **holding 12–24 months rather than re-striking monthly**. The strategy as
backtested (monthly rebalance of 2-year LEAPS across 36 names) is dominated.

## The Monte Carlo changed two conclusions

`07_monte_carlo.md` simulates 4,000 paths × 20 years in a single-factor world where **CAPM holds
exactly by construction**, so any failure measured is caused by the option layer. Two things moved:

1. **The gap is ~100% frictions, not structure.** With every friction switched off, the LEAPS book
   and the stock book differ by **−0.03%/yr** — statistically nothing. Of the −9.8%/yr gap:
   bid-ask/turnover **6.67%/yr**, the American premium you never exercise **4.11%/yr**, dividends
   0.27%/yr, and the exposure drift / gamma / theta all net out. Notes 04–05 classified
   the early-exercise premium and the drift as "structural, surviving perfect execution"; that was
   wrong. They are real costs but they are *policy* costs (the policy being "never exercise", and
   "roll this often"), and policies can be changed.
2. **Rule 5 in note 06 was backwards.** It recommended Δ ≥ 0.90 to cut ruin and the variance
   premium. At realistic spreads that is wrong: deeper delta means more premium, and the bid-ask is
   charged on premium. Measured gap: Δ=0.70 → −6.4%/yr, Δ=0.95 → −18.1%/yr. With frictions off all
   four are ≈ 0. Δ ≈ 0.75–0.80 is the right choice.

---

## Files

| File | Contents |
|---|---|
| `01_parity_identity.md` | The put-call-parity identity that organises everything: what a "stock substitute" actually is, and why "call vs stock" is a category error |
| `02_embedded_financing.md` | Measuring the real financing rate inside quoted LEAPS from the project's cached chains. Includes the estimator's validation and a units bug that nearly produced a 400bp phantom |
| `03_capm.md` | Does CAPM hold? Instantaneous identity, discrete-horizon breakdown, variance risk premium, and empirical tests on the project's own backtest |
| `04_structural_risks.md` | Exposure drift, ruin, capital efficiency, the short-strangle result. All friction-free |
| `05_risk_taxonomy.md` | Every risk, quantified, in a decision-useful order |
| `06_verdict.md` | When LEAPS substitution works, when it does not, and what to change in the backtest |
| **`07_monte_carlo.md`** | **Monte Carlo in a world where CAPM is exactly true: isolates structure from frictions, and corrects two claims in notes 04–06** |
| **`08_all_combinations.md`** | **810 simulated worlds: a 360-cell complete factorial plus a 450-cell Latin hypercube over all 14 assumptions. Which assumptions decide the answer, and which results are universal** |

Raw outputs: `results/` (CSVs). Code: `calcs/`.

## Reproducing

```bash
cd research/calcs
../../.venv/bin/python 01_implied_financing.py   # ~1 min  -> embedded financing from live chains
../../.venv/bin/python 02_capm_theory.py         # ~6 min  -> CAPM identity, horizon bias, VRP drag
../../.venv/bin/python 03_structural_risks.py    # ~10 sec -> exposure drift, ruin
../../.venv/bin/python 04_drift_and_capital.py   # ~5 sec  -> path P&L, capital requirements
../../.venv/bin/python 05_empirical_capm.py      # ~10 sec -> factor tests on nav_portfolios.parquet
../../.venv/bin/python 07_monte_carlo.py         # ~14 min -> the Monte Carlo study (E1-E8)
```

`08_factorial.py` then `09_factorial_analysis.py` run the 810-combination sweep (~40 min for the
sweep, seconds for the analysis). `07_monte_carlo.py` depends on `sim_lib.py` (vectorised Black-Scholes, a vectorised CRR tree, and an
interpolated early-exercise-premium grid) and `sim_engine.py`. Its grid build takes ~2 s and is
validated against the scalar binomial in `binomial.py` to <0.5%.

`calcs/bs.py` (Black-Scholes + greeks) and `calcs/binomial.py` (CRR, American, known dollar
dividends) were written independently of `leaps_ls.pricing` so that this is a check on the engine
rather than a restatement of it. `binomial.py`'s European branch reproduces Black-Scholes to
1e-10, and its American branch returns intrinsic value for a deep-ITM put as it must.

## Caveats

- The financing measurement uses a **single day's** chain snapshot (2026-07-31), which was a
  heavy idiosyncratic day (AAPL −7.4%, MSFT +3.0%).
- The CAPM Monte Carlo and the structural calculations use a **single representative name**
  (S=100, σ=25%, r=4.2%, q=1.5%). They are comparative, not forecasts.
- The empirical factor tests use the project's existing backtest, so they inherit every
  modelling choice in `PLAN.md`. They are evidence about *that* backtest.
