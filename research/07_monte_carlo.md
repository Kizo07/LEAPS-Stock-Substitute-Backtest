# 7. Monte Carlo: stock L/S vs LEAPS L/S in a world where CAPM is true

*Code: `calcs/sim_lib.py` (vectorised BSM + CRR + early-exercise grids), `calcs/sim_engine.py`, `calcs/07_monte_carlo.py` · Output: `results/mc_*.csv`*

## Why simulate

The real-data tests in `03_capm.md` cannot separate three things: the strategy's own negative
alpha, the LEAPS structure, and the frictions. The Monte Carlo separates them, because **CAPM
holds exactly for every stock by construction**. Whatever we then measure failing is caused by the
option layer.

## The world

Arithmetic (not log) returns, so CAPM is exact rather than approximate:

```
R_M = mu_M·dt + sig_M·sqrt(dt)·z_M
R_i = mu_i·dt + beta_i·sig_M·sqrt(dt)·z_M + sig_eps,i·sqrt(dt)·z_i
mu_i = r + beta_i·(mu_M − r)                      <- CAPM, by construction
```

`R_i` is the **total** return. The price return is `R_i − q·dt` and the option is written on the
price. Betas are drawn `U(0.7, 1.3)`; `sig_M = 16%`, total stock vol 30%, `r = 4.2%`, `q = 1.5%`,
ERP 5.5%. 30 names, 10 per side, monthly steps, 20 years, 4,000 paths for the headline.

**Both books get the same capital and the same target delta exposure (+1 long, −1 short per unit),
and both are credited `r` on all posted capital** — the stock book via the rebate on short proceeds
and margin, the LEAPS book via interest on surplus cash plus the rate embedded in the options. So
`r` is the common baseline and the *gap* is the number that means something.

## E1 Headline

| book | CAGR | vol | Sharpe | max DD | terminal wealth p05 / p50 / p95 |
|---|---|---|---|---|---|
| stock L/S | **+3.85%** | 7.48% | 0.54 | −18.5% | 1.19 / **2.13** / 3.72 |
| LEAPS L/S | **−5.99%** | 9.06% | −0.64 | −71.8% | 0.15 / **0.29** / 0.56 |
| **gap** | **−9.84%/yr** | ×1.21 | | | |

Capital per $1+$1 of delta exposure: LEAPS **1.277** vs stock **1.500** (Reg T). So the LEAPS book
really does use 15% less capital — and still loses by 9.8%/yr.

## E2 Attribution — the answer

Each row switches **one** friction off. 2,000 paths, 20 years, quarterly rolls.

| variant | stock CAGR | LEAPS CAGR | gap |
|---|---|---|---|
| all frictions on (baseline) | 3.84% | −5.99% | **−9.83%** |
| no bid-ask | 3.84% | +0.68% | −3.16% |
| no funding spread | 3.84% | −6.20% | −10.04% |
| no dividend | 3.84% | −5.72% | −9.56% |
| European pricing (no early-exercise premium) | 3.84% | −1.88% | −5.72% |
| no borrow on the stock short | 4.02% | −5.99% | −10.01% |
| **ZERO frictions** | 4.02% | 3.99% | **−0.03%** |

**Cost attributable to each friction** (change in the gap when it is switched off):

| friction | cost | share |
|---|---|---|
| **bid-ask / turnover** | **6.67 %/yr** | 68% |
| **American premium you never exercise** | **4.11 %/yr** | 42% |
| dividend forgone | 0.27 %/yr | 3% |
| embedded funding spread | **−0.21 %/yr** (a *benefit*) | −2% |
| borrow the stock book pays | **+0.18 %/yr** (LEAPS avoids it) | −2% |
| **pure structure** | **−0.03 %/yr** | ~0% |

### This corrects the earlier analytic work

In `04_structural_risks.md` and `05_risk_taxonomy.md` I classified the forfeited early-exercise
premium (−4.3%/yr) and the exposure drift as "structural, surviving perfect execution." The
simulation says otherwise, and the simulation is the better instrument:

- **With zero frictions the two books are statistically identical: −0.03%/yr.** The exposure drift,
  the gamma/theta profile and the convexity all net out. Drift helps on the way down and hurts on
  the way up; with symmetric normal returns that is a wash. Gamma earned offsets theta paid.
- The early-exercise premium is real and large (−4.1%/yr), but it is not "structure": it is the
  price of a **policy** (never exercise). Exercise and it disappears — and exercising the deep-ITM
  put simply converts it into the cash short you were trying to replace.
- **Every one of the −9.8% is a friction.** That is the useful conclusion, because frictions can be
  attacked; structure cannot.

## E3 P&L decomposition (%/yr of current equity; components are additive)

| roll every | delta | theta | gamma (residual) | interest | **spread** | sum | LEAPS CAGR |
|---|---|---|---|---|---|---|---|
| 1 mo | −0.03 | −0.65 | +0.81 | +0.58 | **−12.67** | −11.95 | −18.27% |
| 3 mo | −0.01 | −0.70 | +1.04 | +0.59 | **−5.37** | −4.45 | −6.06% |
| 6 mo | +0.04 | −0.76 | +1.23 | +0.60 | **−3.03** | −1.93 | −2.61% |
| 12 mo | +0.15 | −0.90 | +1.46 | +0.60 | **−1.63** | −0.32 | −0.74% |
| 24 mo | +0.34 | −1.17 | +1.69 | +0.62 | **−0.43** | +1.05 | +0.55% |

- `delta ≈ 0` — the book is delta-neutral, as intended.
- `theta + gamma > 0` (+0.16 to +0.52). This is not free money: it is the risk-free rate accruing
  on the net capital prepaid inside the options, and it is what makes the "pure structure" gap zero.
- `interest ≈ +0.60%/yr` = 14.3% of equity in surplus cash × 4.2%. This is the capital-efficiency
  benefit, measured. It is real but small.
- **`spread` is the whole story.** Rolling 2-year LEAPS every month costs 12.7%/yr of equity.
- "gamma" is the residual bucket: convexity P&L net of the early-exercise premium's decay and
  higher-order terms.

## E4 CAPM battery

Pooled over paths × months (4,000 × 240 observations). `d_*` columns are LEAPS minus stock — these
isolate the option layer.

| strategy | α stock | α LEAPS | **d_α** | β stock | β LEAPS | **d_β** | R² stock | R² LEAPS | **d_γ** | γ t-stat | asym stock | asym LEAPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | −0.16% | −9.32% | **−9.16%** | 0.0009 | −0.1206 | −0.122 | 0.000 | 0.045 | **+0.51** | 42.3 | −0.002 | **−0.104** |
| beta-tilted | −0.26% | −9.45% | **−9.19%** | 0.2624 | 0.1664 | −0.096 | 0.241 | 0.081 | **+0.51** | 41.5 | −0.001 | **−0.105** |

Four findings, and three of them reproduce on the real data in `03_capm.md`:

1. **The stock book's α is −0.16%/yr — i.e. zero.** CAPM holds in the simulation, as designed, and
   the harness is not leaking. This is the control that validates everything else.
2. **The LEAPS book loses 9.2%/yr of alpha** relative to the stock book — the same order as the
   −27.7%/yr the project's V1 loses versus V0, and now attributable to known frictions.
3. **d_γ = +0.51 with t = 42.** The LEAPS book acquires significant convexity in the market factor
   that the stock book does not have. Same sign and same story as the real-data result
   (V1 γ = +6.83 vs V0 γ = −1.32). A single beta is not a sufficient statistic.
4. **asym = −0.104 vs −0.002.** The LEAPS book's beta is ~0.10 lower in down markets than in up
   markets; the stock book's is flat. Same sign as the real data (V1 −0.769 vs V0 +0.205).

**New finding — the LEAPS book delivers only 63% of the intended factor exposure.** In the
beta-tilted variant the stock book measures β = 0.2624 and the LEAPS book only 0.1664. The target
delta is identical at each roll; between rolls the delta decays, so the *average realised* exposure
is materially below target. If you size a LEAPS book off its nominal delta you will be
systematically under-exposed, and the shortfall grows with the holding period.

**R² collapses** from 0.241 to 0.081 — the factor model explains a third as much of the LEAPS book.
Same as the real data (0.515 → 0.347).

## E5 Exposure drift (now conditional on the move since the last roll)

Net delta exposure, $ per $1 of a leg. The stock book reads exactly 0.000 in every bucket.

| market move since last roll | 3-month rolls | 12-month rolls |
|---|---|---|
| below −15% | **−0.024** | **−0.111** |
| −15% to −8% | −0.008 | −0.039 |
| −8% to −3% | −0.002 | −0.010 |
| −3% to +3% | +0.003 | +0.020 |
| +3% to +8% | +0.010 | +0.070 |
| above +8% | **+0.035** | **+0.206** |

Monotone in the market, and **three to six times larger at a 12-month holding period** than at
3 months. This confirms the analytic table in `04_structural_risks.md` §4.1 and adds the part that
table could not show: **the drift is a function of the holding period, so it is controllable.**
Roll quarterly and the book stays close to neutral; hold for a year and it is ±0.2 of a leg away
from neutral at the extremes.

Note this drift costs nothing in expectation (E2: pure structure ≈ 0). It costs you when it is
*correlated with your strategy's exposure* — which for momentum it is.

## E6 Ruin

| tenor | Δ | % of legs expiring worthless | worthless legs per path-year |
|---|---|---|---|
| 2y | 0.70 | 40.8% | 19.1 |
| **2y** | **0.80** | **8.8%** | **4.1** |
| 2y | 0.90 | 0.28% | 0.13 |
| 2y | 0.95 | 0.00% | 0.00 |
| 1y | 0.80 | 17.8% | 8.3 |

At the default Δ = 0.80 / 2-year, **roughly four legs per year expire worthless**. Each is a 100%
loss on that leg's premium. Δ = 0.90 all but eliminates it.

## E7 Sensitivity sweeps (gap = LEAPS − stock CAGR)

| parameter | values | gap |
|---|---|---|
| **roll horizon** | 1 / 3 / 6 / 12 / 24 mo | **−22.1 / −9.9 / −6.4 / −4.5 / −3.3 %/yr** |
| **delta target** | 0.70 / 0.80 / 0.90 / 0.95 | **−6.4 / −9.9 / −14.0 / −18.1 %/yr** |
| **stock vol** | 20% / 30% / 40% | **−5.3 / −9.9 / −15.8 %/yr** |
| **VRP** | 0 / 2 / 5 vol pts | **−9.9 / −11.7 / −14.2 %/yr** |
| dividend yield | 0 / 1.5% / 3% | −9.6 / −9.9 / −10.3 %/yr |
| funding spread | 0 / 61 / 200 bp | −10.1 / −9.9 / −9.4 %/yr |
| margin (stock book) | 20% / 35% / 50% | −12.2 / −10.9 / −9.9 %/yr |
| **rebate on short proceeds** | 0 / 3.2% / 4.2% | **−5.6 / −8.8 / −9.9 %/yr** |

Reading the four that matter:

- **Roll horizon is the dominant lever, by a factor of seven.** Monthly re-striking of 2-year LEAPS
  costs 22%/yr; holding them for two years costs 3.3%/yr. This is the single most actionable number
  in the entire project, and `rebalance` was never swept in the project's sensitivity grid.
- **Delta target reverses sign depending on frictions.** With frictions on, deeper is much worse
  (0.70 → −6.4%, 0.95 → −18.1%) because deeper options carry more premium and the bid-ask is
  charged on premium. With frictions off, all four are ≈ 0. **This corrects Rule 5 in
  `06_verdict.md`, which recommended Δ ≥ 0.90.** The right answer is a trade-off, and at realistic
  spreads Δ = 0.70–0.80 wins despite the higher ruin rate; the ruin insurance at Δ = 0.90 does not
  pay for itself.
- **Volatility is the main risk driver.** Doubling vol from 20% to 40% quadruples the gap. LEAPS
  substitution is a low-volatility technique.
- **The rebate assumption moves the answer by 4.3%/yr.** If you are credited `r` on short proceeds
  and margin (institutional), the LEAPS book loses 9.9%/yr. If you are credited nothing (retail
  Reg T), it loses 5.6%/yr. **Anyone quoting a single number for "how bad is LEAPS substitution"
  without stating this is not answering the question.**

## E8 Strategy independence

| strategy | stock CAGR | LEAPS CAGR | gap | net delta after a crash |
|---|---|---|---|---|
| random | 3.84% | −5.99% | **−9.83%** | +0.003 |
| momentum (12-1) | 3.78% | −6.04% | **−9.82%** | +0.003 |
| beta-tilted | 5.11% | −4.67% | **−9.79%** | +0.004 |

The gap is −9.8%/yr for all three. **The LEAPS penalty is a constant drag, additive and independent
of whatever alpha the underlying strategy has.** (Momentum has no alpha here — returns are iid —
which is exactly what makes this a clean test of additivity.) The momentum variant does not show an
extra crash penalty beyond the constant, because with iid returns there are no momentum crashes to
have.

## What the simulation changes

| earlier claim (analytic / real data) | simulation verdict |
|---|---|
| Exposure drift is a first-order structural risk | **Confirmed as a real effect** (E5, ±0.2 at 12-month holds) **but it nets to zero in expectation.** It matters through interaction with the strategy, not on its own. |
| Forfeited early-exercise premium ≈ −4.3%/yr, structural | **Magnitude confirmed (−4.1%/yr) but it is a policy cost, not structure.** Exercise and it goes away. |
| All the V1 − V0 gap is "the LEAPS structure" | **Wrong. ~0% is structure; 100% is frictions.** |
| Use Δ ≥ 0.90 to cut ruin and variance premium | **Wrong at realistic spreads.** Deeper delta costs more premium-proportional spread. Δ = 0.70–0.80 is better. |
| Capital efficiency favours LEAPS | **True but small**: 15% less capital, worth +0.60%/yr of surplus-cash interest. |
| Monthly rebalancing is a minor implementation detail | **It is the dominant term** — 12.7%/yr of equity, 68% of the entire gap. |

## Bottom line

In a world where CAPM is exactly true, a LEAPS-implemented long/short book and a stock
long/short book are **economically equivalent before frictions** (−0.03%/yr apart) and **9.8%/yr
apart after them**. The LEAPS book is not a different asset in expectation — it is the same asset
with a large, recurring, avoidable transaction cost, plus a modest capital saving and a large
increase in path risk (max drawdown −18% → −72%).

If you must use LEAPS: hold them for 12–24 months, re-hedge delta with stock rather than by
re-striking, restrict to low-volatility non-dividend payers, use Δ ≈ 0.75–0.80, and price the short
leg American. That combination removes most of the 9.8%.
