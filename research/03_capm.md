# 3. Does CAPM hold?

*Code: `calcs/02_capm_theory.py` (theory + Monte Carlo), `calcs/05_empirical_capm.py` (tests on the project's own backtest) · Output: `results/capm_*.csv`, `results/emp_*.csv`*

Short answer, in three parts: **instantaneously yes and exactly; over a tradeable horizon for
deep-ITM LEAPS almost; empirically on the project's own backtest, no.**

## 3.1 Instantaneously, CAPM is an identity

Under Black-Scholes, Itô gives

```
dC = (Θ + μ_price·S·Δ + ½σ²S²·Γ) dt + σS·Δ dW
```

so the option's instantaneous beta is the **Cox-Rubinstein elasticity** times the underlying's beta:

```
β_C = Ω · β_S ,      Ω ≡ S·Δ / C
```

Substituting the Black-Scholes PDE `Θ + (r−q)S·Δ + ½σ²S²Γ = rC` and writing `μ_total = μ_price + q`:

```
E[dC]/C · dt  =  r + Ω · (μ_total − r)
```

which is CAPM, exactly. Verified numerically over K ∈ {60, 80, 100, 120} and T ∈ {0.5, 1, 2}:

```
max | E_drift − CAPM_prediction | = 1.11e-15
```

**CAPM is not approximately true for options. In continuous time it is a theorem**, provided you
use the *total*-return drift of the underlying (an earlier version compared against the price
drift and produced a spurious residual of exactly `q·S·Δ/C`).

The elasticities are large, which is the whole point of the exercise:

| strike | Δ (T=2y) | premium | Ω = SΔ/C | β_C (β_S = 1) |
|---|---|---|---|---|
| 60 (Δ≈0.93) | 0.934 | 42.5 | 2.20 | 2.20 |
| **80 (Δ≈0.81)** | 0.807 | 27.2 | **2.97** | **2.97** |
| 100 (ATM) | 0.611 | 16.0 | 3.82 | 3.82 |
| 120 (OTM) | 0.414 | 8.8 | 4.68 | 4.68 |

A Δ=0.80 two-year LEAPS call carries **2.97× the market beta of the stock per dollar of premium**.
That is the "leverage", and it is what any risk report must show.

## 3.2 Over a tradeable horizon: deep-ITM LEAPS are nearly linear

Monte Carlo (200k paths, GBM, stock β=1, ρ=0.85 so R²(stock|market)=0.72, ERP 5.5%). Regression of
realised option excess returns on market excess returns, no variance risk premium:

| option | horizon | Ω | β (instantaneous) | β̂ (OLS) | bias | apparent α | R² |
|---|---|---|---|---|---|---|---|
| K=80, T=2y (Δ=0.81) | 1 d | 2.97 | 2.97 | 2.97 | +0.1% | +0.22%/yr | 0.723 |
| | 21 d | 2.97 | 2.97 | 3.00 | +0.9% | +0.42%/yr | 0.722 |
| | 63 d | 2.97 | 2.97 | 3.02 | +1.8% | +0.10%/yr | 0.716 |
| K=100, T=2y (ATM) | 63 d | 3.82 | 3.82 | 3.96 | +3.4% | −0.07%/yr | 0.703 |
| K=100, **T=0.25y** (ATM, short-dated) | 63 d | 10.29 | 10.29 | **11.45** | **+11.3%** | −2.60%/yr | **0.577** |

**This is the one genuine virtue of deep-ITM LEAPS.** Two-year, Δ≈0.80 contracts track their
instantaneous beta to within 2% even at a 63-day horizon, and the misspecification alpha is under
0.5%/yr. A three-month ATM option — the instrument most option research is written about — is
broken: an 11% beta bias, −2.6%/yr of spurious alpha, and R² collapsing from 0.72 to 0.58.

So the "options violate CAPM" literature (Coval & Shumway 2001; Bakshi & Kapadia 2003; and the
wider variance-premium literature) is largely a statement about **short-dated and at-the-money**
options. Deep-ITM LEAPS are the least-bad way to hold optionality, and the linear factor model is
nearly adequate for them — *if* there is no variance risk premium.

## 3.3 The variance risk premium is where CAPM actually dies

You buy at implied vol and the world delivers realised vol. The drag is `−vega × (IV − RV)`:

**Annualised drag, % of the option premium** (S=100, σ=25%, r=4.2%, q=1.5%):

| K/S | T=0.25 | T=1.0 | T=2.0 |
|---|---|---|---|
| 0.6 | 0.00 | −0.14 | −0.53 |
| **0.8** | −0.32 | −1.77 | **−2.54** |
| 0.9 | −2.13 | −3.83 | −4.28 |
| 1.0 (ATM) | −7.46 | −6.93 | **−6.50** |
| 1.2 | −32.77 | −16.09 | −12.17 |

(VRP = 2 vol points. At 5 vol points, multiply by 2.5.)

Restated **per unit of notional exposure**, which is the comparison that matters against a stock
portfolio (T = 2y):

| K/S | VRP = 2 pts | VRP = 5 pts |
|---|---|---|
| 0.6 | −0.23%/yr | −0.57%/yr |
| **0.8** | **−0.69%/yr** | **−1.73%/yr** |
| 1.0 (ATM) | −1.04%/yr | −2.59%/yr |

Deep ITM roughly halves the variance-premium drag relative to ATM — worth having, but at 0.7–1.7%
of notional per year it is the same order as the financing spread and the dividend give-up, and it
is a pure negative alpha that no factor model will attribute correctly.

In the Monte Carlo with a 3-vol-point VRP, the Δ=0.80 two-year call shows **α = −1.5 to −1.8%/yr**
and a beta biased −4 to −7%. The three-month ATM call shows **α = −38 to −43%/yr**.

## 3.4 Empirical tests on the project's own backtest

`results/nav_portfolios.parquet` (V0 … V3) against the cached Ken-French daily factors,
2007-01-04 → 2026-05-29 (4,881 overlapping days).

### (a) Static attribution — successfully replicates the project

| variant | α (ann.) | t(α) | β Mkt-RF | β SMB | β HML | β MOM | R² |
|---|---|---|---|---|---|---|---|
| V0 stock | −8.69% | −2.14 | 0.087 | 0.087 | −0.245 | 0.987 | 0.515 |
| V1 LEAPS | −36.38% | −6.85 | 0.044 | 0.088 | −0.246 | 0.895 | 0.347 |
| V2 synthetic | −21.92% | −4.79 | 0.070 | 0.114 | −0.261 | 1.019 | 0.475 |
| V3 hybrid | −13.12% | −2.94 | 0.053 | 0.061 | −0.260 | 0.959 | 0.462 |

Matches `REPORT.md` to the digit — the pipeline is reproducible. R² is the first warning sign:
**0.515 for V0 but 0.347 for V1.** The factor model explains a third less of the LEAPS book.

### (b) Treynor-Mazuy curvature — the decisive test

`r − rf = a + b·x + γ·x²` where `x` is the market excess return:

| variant | α (ann.) | β | **γ** | **t(γ)** | R² |
|---|---|---|---|---|---|
| V0 stock | +2.25% | −0.148 | −1.32 | −3.28 | 0.015 |
| **V1 LEAPS** | −58.74% | −0.152 | **+6.83** | **+15.42** | 0.059 |
| V2 synthetic | −8.84% | −0.172 | −1.81 | −4.16 | 0.018 |
| **V3 hybrid** | −20.27% | −0.170 | **+3.15** | **+7.57** | 0.029 |

Read this carefully, because the internal consistency is the evidence:

- V0 and V2 — the two variants that are **not** net long options — both show γ ≈ −1.3 to −1.8.
  That is the known concavity of momentum itself (momentum is short the rebound; it is concave in
  the market). Call it the baseline.
- V1 and V3 — the two variants that **are** net long options — flip to γ = **+6.8** and **+3.2**,
  with t-statistics of 15.4 and 7.6. The LEAPS overlay adds roughly **+5 to +8 units of
  convexity** on top of the momentum baseline.

**This is the direct empirical answer to "does CAPM hold?": no.** For V1 the market beta is not a
sufficient statistic — a squared market term is significant at t = 15.4. Magnitude: γ = 6.83 means
a ±2% market day contributes 27 bp of extra return from curvature alone, and a ±5% day
contributes 1.7%. Any risk report that quotes a single beta for the LEAPS book is misdescribing
it.

### (c) Volatility loading — the same story, independently

Adding ΔVIX to FF3+MOM:

| variant | R² (FF) | R² (+ΔVIX) | β_ΔVIX | t |
|---|---|---|---|---|
| V0 stock | 0.515 | 0.517 | **−0.0116** | −3.95 |
| **V1 LEAPS** | 0.347 | 0.352 | **+0.0237** | **+6.18** |
| V2 synthetic | 0.475 | 0.477 | −0.0129 | −3.90 |
| V3 hybrid | 0.462 | 0.463 | +0.0051 | +1.59 |

V0 and V2 load **negatively** on VIX changes, like any equity book. V1 loads **positively and
significantly**: a +10% VIX day adds 24 bp. V1 is **long volatility**, because it is net long
options. V2 is not, because synthetics are forwards and have no vega.

This is not a bug in the backtest — it is the backtest correctly reporting that the LEAPS book is
a different asset. But it means V1's return stream contains a short-vol-premium-harvesting-inverse
(i.e. long-vol) component with a *negative* expected return, which is part of the −36%/yr alpha.

### (d) Up-market vs down-market beta — the state dependence

| variant | β (up days) | β (down days) | asymmetry |
|---|---|---|---|
| V0 stock | −0.335 | −0.130 | +0.205 |
| **V1 LEAPS** | **+0.126** | **−0.643** | **−0.769** |
| V2 synthetic | −0.392 | −0.120 | +0.272 |
| V3 hybrid | −0.113 | −0.415 | −0.302 |

V1's market beta swings by **0.77** between up and down markets — and in the opposite direction to
V0's. On down days V1 has a beta of −0.64, i.e. it *rises* when the market falls. This is the
empirical signature of the exposure-drift mechanism derived in `04_structural_risks.md`: the book
accumulates short exposure as the market falls, so it is short at the bottom.

### (e) One test that was uninformative

Rolling 252-day market betas: V0 σ = 0.522 (range 1.649), V1 σ = 0.502 (range 1.620). The LEAPS
book's beta is **not** more unstable in this metric. The reason is that the rolling beta of a
momentum L/S book is dominated by the momentum factor's own time-varying beta, which swamps the
LEAPS effect. Reported for completeness; it should not be used as evidence either way.

## 3.5 Synthesis

| claim | status |
|---|---|
| CAPM holds instantaneously for options | **True, exactly** (β_C = Ω·β_S) |
| CAPM holds over a month for deep-ITM LEAPS | **Nearly** (β bias ≤ 2%, α ≤ 0.5%/yr) |
| CAPM holds over a month for short-dated ATM options | **False** (β bias 11%, R² 0.58) |
| CAPM holds once the variance risk premium is real | **False** (−0.7 to −1.7%/yr of notional for deep ITM) |
| A single beta describes the LEAPS L/S book | **False** — γ = +6.8, t = 15.4; β swings 0.77 between up and down markets |
| The LEAPS book has the same factor exposures as the stock book | **False** — it is long volatility (β_ΔVIX = +0.024 vs −0.012) |

**Practical consequence.** Do not risk-manage a LEAPS-implemented book with a linear beta. Report
(a) delta-notional gross and net exposure, (b) the Treynor-Mazuy γ, (c) the ΔVIX loading, and
(d) the exposure-drift profile as a function of the market move and of remaining tenor
(`04_structural_risks.md` §4.1 gives the table to paste into a risk pack).
