# 6. Verdict, decision rules, and what to change

## 6.1 The verdict

**LEAPS substitution is not a cheaper way to run the same long/short book. It is a different
strategy that happens to hold the same names.**

The stock book has: constant exposure, zero vega, linear market response, and margin-based capital.
The LEAPS book has: exposure that drifts with the market, long vega, convex market response
(γ = +6.8, t = 15.4), and 100%-cash capital at 1.04 per $1 long + $1 short.

That is not a criticism of the implementation — the project's engine is clean and its results are
reproducible (I replicated its FF attribution to the digit). It is a statement about the object
being measured. The −23.4%/yr of CAGR that V1 loses relative to V0 (−30.1% vs −6.7%) decomposes roughly as, per
year, per unit of the book's capital:

| term | size | source |
|---|---|---|
| forfeited early-exercise premium | −4.3% | `01` §1.3 — you pay American, you collect European |
| variance risk premium | −0.7 to −1.7% | `03` §3.3 — long vega into a negative premium |
| embedded financing spread | −0.4 to −1.1% | `02` — 61–131 bp over Treasuries on the deferred strike |
| dividend give-up, net of the carry benefit | −1.5% at 2007–26 average rates | Rule 2 below |
| **turnover (monthly re-striking)** | **the remainder, and the largest single term** | `05` R11 — 82%/yr of spread cost |

Note the first two are structural and survive perfect execution. Note also that the dividend term
is **rate-dependent**: at today's 4.2% short rate the carry benefit largely offsets the dividend on
names yielding under ~1.9%, whereas at the 1.5% average rate that prevailed over most of the
sample it does not. The backtest ran through the low-rate regime, which is the worst case for this
structure.

## 6.2 When it does make sense

Ranked by how defensible the case is:

**1. You have no margin line at all, and the name pays no dividend.**
This is the only case where the arithmetic is unambiguously favourable. Long leg via a Δ≥0.90
LEAPS at 40.7% of exposure versus 100% fully funded, on a zero-dividend name where you give up
nothing. Capital cost: 50–100 bps over Treasuries plus ~0.2–0.6%/yr of variance premium. Works.

**2. You need certainty of borrow cost on the short leg.**
The one genuine advantage the option route has: a synthetic short fixes your financing for the
full tenor and carries **no recall risk**. If you have been bought in, or you are short a name
whose borrow is volatile, that is worth something real. Note this is an argument for the
*synthetic* (V2), not for the naked put (V1) — and note that you pay for it in capital (70.2% of
exposure).

**3. You want long-dated convexity and you will say so.**
If the objective is explicitly "momentum exposure with a long-vol overlay and tail participation",
then V1 is a coherent product. γ = +6.8 with positive ΔVIX loading is a feature under that
mandate. Just size it as an option book, not as a 1× equity book.

**4. Tax.**
US equity options held more than 12 months get long-term capital-gains treatment. Buy-and-hold
LEAPS on a concentrated position is a legitimate tax-aware structure — but **rolling destroys the
benefit** by realising gains annually. This argues for buy-and-hold, not monthly re-striking.

## 6.3 When it does not

| condition | why |
|---|---|
| You have portfolio margin | LEAPS need **173–260%** of the capital a PM stock book needs |
| Dividend-paying names | You give up 1.0–2.6%/yr (SPY 1.0%, JPM 1.7%, KO 2.4%, XOM 2.6%) and get ~50–100 bps of financing benefit |
| You rebalance monthly | 82%/yr of spread cost. The single largest term in the whole analysis |
| Short leg, in general | 70.2% of exposure in cash, 6.8% of spot of forfeited American premium, and a BAW model will understate the price by 14% |
| You intend to hold to expiry | The pair is a short strangle: −9.3% of capital unless S_T exits [−27%, +64%] |
| Names without liquid deep-ITM LEAPS | 32 of the 35 names in the project universe. Median spread 210–286 bps, OI frequently single digits |

## 6.4 Decision rules

Compute these before putting on a LEAPS-implemented leg. All are closed-form.

**Rule 1 — Capital test.** Substitution only if
`premium_American / |Δ| < margin_rate_stock`.
At Δ=0.80, T=2y: long leg 33.7% < 50% (Reg T) → passes; < 20–30% (PM) → **fails**.
Short leg 70.2% → **fails against everything except Reg T on a hard-to-borrow name.**

**Rule 2 — Dividend test (long leg).** All per year, per unit of notional, before transaction
costs. The call route frees `(S − C)/S` of capital; valuing that at the risk-free rate gives the
benefit, and the embedded financing spread taxes the deferred strike `K·e^{−rT}/S`:

```
q*  =  r · (S − C)/S   −  (f − r) · K·e^{−rT}/S   −  variance_premium
```

| | Δ=0.80, T=2y | Δ=0.90, T=2y | Δ=0.80, T=1y |
|---|---|---|---|
| premium / notional | 0.266 | 0.366 | 0.199 |
| capital freed | 0.734 | 0.634 | 0.801 |
| deferred strike | 0.743 | 0.617 | 0.815 |
| **q\* at r = 1.5%, spread 61bp** | **−0.04%** | −0.12% | +0.01% |
| **q\* at r = 4.2%, spread 61bp** | **+1.94%** | +1.60% | +2.18% |
| **q\* at r = 4.2%, spread 131bp** | **+1.42%** | +1.16% | +1.61% |

Two readings. First, **the case is strongly rate-dependent**: at the 2007–2026 average short rate
of ~1.5% the break-even dividend yield is essentially zero, and at today's 4.2% it is ~1.4–1.9%.
Second, this reconciles the project's own RQ1 result exactly: over 2007–2026 the sample average
short rate was ~1.5% and the universe's median dividend yield ~1.5%, so the predicted tracking
difference is `q* − q ≈ −1.5%/yr`, and the project measures **−2.42%/yr** — the ~0.9% gap being
transaction costs. The model and the backtest agree.

Practical form: **the long leg only pays on names yielding less than ~1.5%/yr, and only when short
rates are high.**

**Rule 3 — Borrow test (short leg).** Put-call parity makes this cleaner than it looks: a synthetic
short *is* a short forward, so it is economically identical to a cash short once borrow is paid.
The decision variable is **not the level of the borrow** but the difference between your quoted
borrow and the borrow implied by the chain:
`q_loan = r − (1/T)·ln(F_chain / (S − PV(D)))`.
If your quoted borrow exceeds the option-implied borrow, the synthetic is cheaper; if not, it is
not. Do not compare a naked put to a cash short — they are different assets (`01` §1.4).

**Rule 4 — Horizon test.** Never re-strike more than twice a year. The dominant cost is
`(turnovers/yr) × (round-trip spread) × (capital/exposure)`. At 12 turnovers, 2.5% round-trip and
1.04 capital, that is 31%/yr of notional. At 2 turnovers it is 5%/yr.

**Rule 5 — Delta target: use ~0.75–0.80, NOT ≥ 0.90.** This rule was reversed by the Monte Carlo.
The analytic case for deeper delta is real (ruin halves, 28.1% → 13.5%; variance premium falls
~35%) but it ignores that **the bid-ask is charged on premium, and deeper options carry more
premium**. Measured in `07_monte_carlo.md` E7, the LEAPS − stock gap is:

| Δ | 0.70 | 0.80 | 0.90 | 0.95 |
|---|---|---|---|---|
| gap (with frictions) | **−6.4%** | −9.9% | −14.0% | −18.1% /yr |
| gap (frictions off) | −0.07% | −0.05% | −0.03% | −0.01% /yr |
| legs expiring worthless (T=2y) | 40.8% | 8.8% | 0.3% | 0.0% |

So deeper delta is *structurally* slightly better and *practically* much worse. Δ ≈ 0.75–0.80 is
the sweet spot: it cuts ruin from 41% to 9% without paying the premium-proportional spread. Note
also that the project's grid finds `delta_target` nearly irrelevant (1.2% CAGR span) — that is an
artefact of monthly turnover, where spread swamps the delta effect; at a realistic horizon delta
target is the second-largest lever after the roll horizon.

**Rule 6 — Risk reporting.** Never quote a single beta. Report:
(a) delta-notional gross and net, (b) Treynor-Mazuy γ, (c) β_ΔVIX, (d) the exposure-drift table
from `04` §4.1 as a function of market move and remaining tenor, (e) the strangle breakevens.

## 6.5 What I would change in the backtest

Ordered by expected impact on the answer:

1. **Sweep the implementation horizon.** `rebalance` is fixed at `ME` (monthly). Add `QE`, `2/Y`
   and `hold-to-expiry`. This is the dominant parameter and it is not in the grid. Expected effect:
   V1's −30.1% CAGR moves by tens of points.
2. **Price the short leg American.** `pricing/baw.py` understates the Δ=−0.80 two-year put by 6.8%
   of spot (14% of its price). Replace BAW with the CRR binomial in `calcs/binomial.py` — it is
   40 lines, validates against Black-Scholes to 1e-10, and is fast enough for the hot loop at this
   universe size if the tree is limited to ~200 steps. Expected effect: materially worse V1/V2.
3. **Restrict the universe to names with real LEAPS liquidity.** Filter on open interest at the
   target strike, not on the existence of a quote. This cuts 32 of 35 names and is the honest
   answer to "can this be traded".
4. **Report exposure drift as a first-class output.** Net delta-notional through time, and the
   conditional (up/down) betas. The current `portfolio_summary.csv` reports CAGR/vol/Sharpe/maxDD;
   none of these reveal that the book is −0.64 beta on down days.
5. **Add the Treynor-Mazuy regression to `05_make_report.py`.** γ = +6.8, t = 15.4 is the single
   most informative statistic in the whole study and it is not currently computed.
6. **Reframe RQ1/RQ2.** "Call vs stock" and "put vs short stock" compare assets with different
   payoff shapes. The defensible comparison is *synthetic vs stock*, whose tracking difference is
   purely the financing-spread differential. Keep the naked-instrument lanes, but label them as
   "convex overlay", not as "substitute".

## 6.6 Literature cross-check

| finding here | external support |
|---|---|
| Options earn far less than CAPM predicts | Coval & Shumway (2001), *Expected Option Returns*; Bakshi & Kapadia (2003) on delta-hedged gains |
| The variance risk premium is pervasive and negative | Carr & Wu (2009) and the wider VRP literature; magnitudes of 2–5 vol points for single names are standard |
| LEAPS calls are not preferred stock substitutes for risk-averse investors | *Equity LEAPS Calls vs. Stocks: An Empirical Study for Long-Term Speculation*, SSRN 1919066 — reaches the same conclusion by simulation |
| Borrow cost is priced into the option-implied forward | Muravyeva & Pearson, *Understanding Returns to Short Selling Using Option-Implied Stock Borrowing Fees*; the implied-fee extraction in `02` follows the same parity inversion |
| Deep-ITM American puts trade near intrinsic | Standard early-exercise result; the magnitude here (14% of price at Δ=−0.80, T=2y) is computed, not cited |

## 6.7 One paragraph

Put-call parity is the whole analysis. It says the only clean stock substitute built from options
is the synthetic — a forward — and that a naked call or a naked put is a forward plus an option
whose cost you can compute exactly in advance. The measured cost of that option is about 4.3%/yr of
forfeited early-exercise premium, 0.7–1.7%/yr of variance premium, and 0.6–1.5%/yr of financing
spread, on top of the dividend you no longer receive. Against that you save capital only if you
have no margin line, and you take on a book whose exposure drifts with the market, whose beta
swings by 0.77 between up and down days, and whose convexity is significant at t = 15.4. LEAPS
substitution is a real technique with a narrow, well-defined use case. A monthly-rebalanced,
both-legs-replaced, 35-name momentum book is not it.
