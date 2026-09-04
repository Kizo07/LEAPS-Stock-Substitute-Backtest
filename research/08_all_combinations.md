# 8. Every combination: 810 simulated worlds

*Code: `calcs/08_factorial.py` (design + run), `calcs/09_factorial_analysis.py` (analysis) ·
Data: `results/mc_factorial.csv` (810 cells), `mc_factorial_effects.csv`, `mc_factorial_anova.csv`*

Note 07 measured one baseline and swept one parameter at a time. That cannot see interactions. This
note runs **combinations**.

## The design

**Stage A — a complete factorial** over the six assumptions that interact most. Every combination
is run, no sampling: `roll horizon (5) × delta target (3) × stock vol (3) × American (2) ×
half-spread (2) × VRP (2)` = **360 cells**.

**Stage B — a Latin hypercube** over all **14** assumptions, so combinations outside the core grid
are covered: tenor, dividend, funding spread, borrow, margin, rebate, strategy, universe size.
**450 cells.**

**810 cells total**, 400 paths × 20 years each, all using **common random numbers** (a fixed seed).
Cells that differ only in an implementation parameter see *identical* price paths, which makes the
difference between two cells far more precise than the noise in either one. 26 cells blew up
(equity went negative in extreme corners) and are dropped, leaving **784**.

Everything else is held at the note-07 baseline: `r = 4.2%`, ERP 5.5%, σ_M 16%, 20 names, 6 per side.

## A1 The distribution of the answer

Gap = LEAPS CAGR − stock CAGR, over all 784 combinations.

| p1 | p5 | p10 | p25 | **p50** | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|
| −43.4 | −27.8 | −20.2 | −10.6 | **−4.65** | −0.97 | +1.47 | +2.93 | +4.96 |

mean −7.44 %/yr, sd 9.87.

| | |
|---|---|
| **P(gap > 0)** — LEAPS actually wins | **18.2%** |
| P(gap > −3%/yr) — competitive | 39.0% |
| P(gap < −10%/yr) — clearly dominated | 26.7% |

**This is the honest answer to "is LEAPS substitution good or bad?" — it is not a fact, it is a
distribution, and the answer depends on assumptions you have to state.** The median is −4.7%/yr,
but across the plausible parameter space it ranges from +5% to −43%.

## A2 Main effects (range = worst level minus best level, pp of CAGR)

| parameter | levels | range | best level | best gap | worst level | worst gap |
|---|---|---|---|---|---|---|
| **roll horizon** | 6 | **17.74** | 24 mo | −1.27 | 1 mo | −19.00 |
| **half-spread** | 4 | **10.90** | 25 bp | −1.21 | 200 bp | −12.11 |
| **stock vol** | 9 | **8.75** | 18% | −3.02 | 45% | −11.77 |
| **delta target** | 7 | **7.00** | 0.65 | −2.98 | 0.90 | −9.98 |
| **option tenor** | 3 | **5.32** | 0.5y | −3.44 | 2.0y | −8.76 |
| **VRP** | 4 | **5.31** | 0.02 | −5.47 | 0.04 | −10.78 |
| **short rebate** | 3 | **4.75** | 0 | −3.01 | 4.2% | −7.76 |
| American pricing | 2 | 2.78 | European | −6.07 | American | −8.85 |
| strategy | 3 | 2.74 | beta | −5.42 | random | −8.17 |
| borrow | 3 | 2.37 | 3% | −4.38 | 0.25% | −8.84 |
| dividend yield | 3 | 3.84 | 3% | −4.75 | 1.5% | −8.59 |
| funding spread | 3 | 3.50 | 200 bp | −5.21 | 61 bp | −8.71 |
| margin rate | 3 | 3.35 | 20% | −5.30 | 50% | −8.65 |
| names per side | 2 | 1.13 | 5 | −5.06 | 10 | −6.19 |

Note the two counter-intuitive signs, both of which are real:
- **A higher funding spread is mildly *good*** for the LEAPS book (−8.71 at 61 bp → −5.21 at
  200 bp). Because the Δ=0.80 put is struck 74% above spot, the short side is a large *lender*;
  pricing the pair at `r + spread` makes that leg cheaper and the benefit outweighs the cost on the
  call leg. Small effect, but the sign is not a bug.
- **A higher borrow fee narrows the gap.** Borrow is a cost to the *stock* book only.

## A3 Variance decomposition (incremental R², all other terms held in)

Model R² = 0.662 on 784 cells. Standardised numerics (`z_`), dummies (`d_`), interactions (`i_`).

| term | coef (pp per 1 sd) | ΔR² |
|---|---|---|
| **z_roll horizon** | +4.69 | **0.220** |
| **z_half-spread** | −3.96 | **0.156** |
| **i_roll × half-spread** | +2.51 | **0.064** |
| d_American | −3.71 | 0.035 |
| z_tenor | −2.12 | 0.027 |
| z_stock vol | −1.51 | 0.023 |
| z_VRP | −1.35 | 0.018 |
| z_delta target | −1.25 | 0.015 |
| z_rebate | −1.49 | 0.014 |
| i_tenor × roll | +1.12 | 0.013 |
| i_roll × vol | +0.97 | 0.009 |
| i_roll × delta | +0.94 | 0.009 |
| z_borrow | +0.90 | 0.006 |
| z_margin | +0.69 | 0.003 |
| z_dividend | +0.41 | 0.002 |
| z_funding spread | +0.20 | 0.0004 |
| strategy dummies | ≤0.42 | 0.0002 |

**Two parameters — how often you roll and how wide your spread is — explain 38% of the variance on
their own and 44% with their interaction.** The funding spread, the dividend and the strategy are
noise by comparison (ΔR² < 0.002). The residual 34% is Monte Carlo noise plus higher-order
interactions.

## A4 The interaction that matters

`gap, %/yr` — roll horizon (rows) × half-spread (cols):

| | 50 bp | 200 bp | spread cost of churning |
|---|---|---|---|
| **1 mo** | −13.5 | **−35.1** | **21.6 pp** |
| 3 mo | −6.2 | −17.1 | 10.9 pp |
| 6 mo | −4.3 | −10.0 | 5.7 pp |
| 12 mo | −3.3 | −6.2 | 2.9 pp |
| **24 mo** | −2.3 | −3.1 | **0.8 pp** |

**The bid-ask only matters if you trade.** At a 24-month hold, paying 200 bp instead of 50 bp costs
0.8 pp/yr. At monthly rolls the same widening costs 21.6 pp/yr — more than the entire gap in most
other configurations. This is the single most actionable result in the project: the spread and the
horizon are not independent dials, they multiply.

`gap, %/yr` — roll horizon × American pricing:

| roll | European | American | cost of not exercising |
|---|---|---|---|
| 1 mo | −15.3 | −18.8 | 3.5 pp |
| 6 mo | −2.0 | −5.8 | 3.8 pp |
| 12 mo | −0.9 | −4.4 | 3.5 pp |
| **24 mo** | **+0.1** | **−3.3** | **3.4 pp** |

The early-exercise cost is ~3.5 pp/yr **regardless of horizon** — and that means at long horizons
it becomes the *dominant* remaining cost. A book that holds LEAPS for two years and never exercises
loses 3.3%/yr purely to the American premium, and at that horizon it is essentially all that is
left.

## A5 The favourable region

**143 of 784 combinations (18%) have the LEAPS book winning outright.** Their profile:

| parameter | distribution among the winners |
|---|---|
| roll horizon | 24 mo 36%, 12 mo 26%, 6 mo 22%, 3 mo 6% — **nothing at 1 mo** |
| half-spread | 25 bp 45%, 100 bp 24%, 200 bp 21%, 50 bp 10% |
| American | **European 71%** |
| VRP | 0 52%, 0.02 30%, 0.05 17% |
| rebate | **0 52%** (i.e. the stock book gets no interest on short proceeds) |
| stock vol | spread across 0.18–0.38, no strong tilt |

Loosening to `gap > −2%/yr`: 247 of 784 (32%). Loosening to `gap > −5%/yr`: 410 of 784 (52%).

**The recipe, if you must use LEAPS:** hold 12–24 months, trade at ≤ 25–50 bp, expect no interest
credit on your short proceeds, and pick low-volatility names. Under those conditions LEAPS wins in
a majority of draws.

## A6 Robustness of the CAPM findings across the whole space

This is the most valuable output of the sweep — it says which results are properties of the world
and which are properties of one calibration.

| finding | median | p05 | p95 | **share of combinations with the same sign** |
|---|---|---|---|---|
| **Treynor-Mazuy γ gap (LEAPS − stock)** | +0.68 | +0.16 | +1.98 | **100.0% positive** |
| **up/down beta asymmetry gap** | −0.137 | −0.403 | −0.031 | **99.9% negative** |
| alpha gap (LEAPS − stock) | −4.23 %/yr | −30.5 | +3.0 | **81.4% negative** |
| beta gap (LEAPS − stock) | −0.008 | −0.147 | +0.037 | 68.1% — **not robust** |

- **Convexity: universal.** In all 784 simulated worlds the LEAPS book has *more* curvature in the
  market factor than the stock book, without a single exception. A linear beta is never a sufficient
  statistic for a LEAPS-implemented book. This is the strongest result in the whole study.
- **Down-market beta gap: universal.** In 99.9% of worlds the LEAPS book's beta is lower in down
  markets than in up markets. The exposure-drift mechanism is not a calibration artefact.
- **Negative alpha: robust but not universal.** 81% of worlds; in 19% the LEAPS book's alpha is
  actually higher (those are the favourable-region cells).
- **Beta dilution: not robust.** Median −0.008, and the sign flips in 32% of cells. The "LEAPS
  delivers only 63% of intended beta" result from note 07 depends on the book having a meaningful
  beta to dilute.

R² of the market regression, by strategy (the comparison is only meaningful where beta ≠ 0):

| strategy | n | R² stock | R² LEAPS | LEAPS lower in |
|---|---|---|---|---|
| beta-tilted | 146 | 0.165 | 0.105 | **100%** |
| momentum | 146 | 0.000 | 0.002 | 34% |
| random | 144 | 0.000 | 0.001 | 15% |

For the beta-tilted book — the only one with real market exposure — **the LEAPS book's R² is lower
in every single combination.** The factor model always explains less of it.

## A7 Extremes

**10 best** (all win by 6–8%/yr):

| gap | roll | Δ | σ | American | spread | VRP | tenor | strategy | rebate |
|---|---|---|---|---|---|---|---|---|---|
| +8.00 | 6 mo | 0.95 | 45% | **European** | **25 bp** | 0.05 | 2.0y | momentum | 0 |
| +7.13 | 6 mo | 0.65 | 26% | **European** | **25 bp** | 0.00 | 1.0y | beta | 0 |
| +6.53 | 24 mo | 0.75 | 38% | **European** | **25 bp** | 0.00 | 2.0y | random | 0 |
| +6.44 | 12 mo | 0.90 | 18% | **European** | 100 bp | 0.00 | 1.0y | random | 0 |
| +6.40 | 24 mo | 0.80 | 22% | **European** | **25 bp** | 0.02 | 1.0y | random | 0 |

**10 worst** (all lose by 43–49%/yr):

| gap | roll | Δ | σ | American | spread | tenor | rebate |
|---|---|---|---|---|---|---|---|
| −49.28 | **1 mo** | 0.80 | 40% | European | **200 bp** | 2.0y | 4.2% |
| −49.14 | **1 mo** | 0.90 | 30% | American | **200 bp** | 2.0y | 4.2% |
| −47.48 | **1 mo** | 0.90 | 30% | European | **200 bp** | 2.0y | 4.2% |
| −46.11 | **1 mo** | 0.80 | 34% | American | **200 bp** | 2.0y | 0 |
| −45.95 | **1 mo** | 0.70 | 40% | American | **200 bp** | 2.0y | 4.2% |

Every one of the ten worst cells has **monthly or bi-monthly rolls and a 200 bp spread**. Every one
of the ten best has **a ≤ 100 bp spread and a ≥ 6-month hold**. Not one exception in either
direction.

## The delta-target trade-off (all 784 cells)

| Δ | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|
| mean gap, %/yr | **−2.98** | −6.65 | −5.46 | −8.34 | −5.98 | **−9.98** | −8.43 |
| % legs expiring worthless | **69.4** | 51.9 | 36.0 | 19.7 | 12.4 | **2.5** | 0.8 |
| cells | 65 | 181 | 64 | 180 | 63 | 171 | 60 |

The gap is *not* monotone in Δ — the cell counts are uneven and the marginal means are noisy — but
the trend is clear (shallow ≈ −3, deep ≈ −10) and the ruin rate falls steeply and monotonically.

Priced at the endpoints: **moving from Δ = 0.65 to Δ = 0.90 costs 7.0 pp/yr of return and buys a
reduction in the worthless-leg rate from 69% to 2.5%.** That is far too expensive — a leg expiring
worthless costs that leg's premium, not the whole book's return. **Δ ≈ 0.75–0.80 is the balance
point**: you give up ~2–5 pp/yr versus the shallowest setting and cut the ruin rate to 20–36%.
Note 06's corrected rule (Δ ≈ 0.75–0.80, not ≥ 0.90) holds across the whole space.

## Marginal means for the three dominant parameters

| roll horizon | 1 mo | 2 | 3 | 6 | 12 | 24 |
|---|---|---|---|---|---|---|
| mean gap | **−19.00** | −10.22 | −8.85 | −4.65 | −2.60 | **−1.27** |

| half-spread | 25 bp | 50 | 100 | 200 |
|---|---|---|---|---|
| mean gap | **−1.21** | −5.96 | −5.79 | **−12.11** |

| stock vol | 18% | 20 | 22 | 26 | 30 | 34 | 38 | 40 | 45 |
|---|---|---|---|---|---|---|---|---|---|
| mean gap | −3.02 | −7.50 | −5.49 | −4.57 | −8.88 | −5.93 | −5.92 | **−11.77** | −7.91 |

(Vol is non-monotone because the marginal means confound vol with the tenor and delta draws; the
clean comparison is note 07's one-at-a-time sweep: 20% → −5.3, 30% → −9.9, 40% → −15.8.)

## What this changes

1. **"Does LEAPS substitution work?" has no single answer.** Across 784 plausible worlds: 18% it
   wins, 39% it is competitive, 27% it is dominated. Anyone quoting one number has picked a
   calibration, probably without saying so.
2. **Two parameters decide it: roll horizon and spread**, and they multiply. 44% of the variance
   sits in those two and their interaction. The dividend, the funding spread and the choice of
   strategy are noise.
3. **The CAPM failures are universal; the return penalty is not.** Convexity (100% of worlds) and
   down-market beta asymmetry (99.9%) are properties of the structure. The negative alpha is a
   property of the frictions (81%) and disappears in the favourable region.
4. **The early-exercise premium is the floor.** ~3.4 pp/yr at every horizon. Once you stop churning,
   it is what remains — and it is removed by exercising, which is free for a put (you just become
   short the stock).
