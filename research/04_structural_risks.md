# 4. Structural risks: what breaks with no frictions at all

*Code: `calcs/03_structural_risks.py`, `calcs/04_drift_and_capital.py` · Output: `results/risk_*.csv`, `results/drift_*.csv`*

Everything in this note is computed with **zero bid-ask, zero borrow, zero financing spread, zero
commissions**. Whatever damage appears here is pure structure — it cannot be engineered away by
getting better fills.

> **Read `07_monte_carlo.md` alongside this note.** The Monte Carlo (in a world where CAPM is
> exactly true) finds that **all of these effects net to zero in expectation** — the two books differ
> by −0.03%/yr with every friction switched off. The exposure drift in §4.1 and the gamma/theta
> profile in §4.2 are real and they are large *conditionally*, but they are symmetric and they
> cancel. What does not cancel is the cost of the policies you layer on top: how often you roll
> (§4.2's strangle is a hold-to-expiry artefact) and whether you ever exercise (§1.3). Treat the
> numbers below as **conditional risk**, not as expected cost.

Baseline: S = 100, σ = 25%, r = 4.2%, q = 1.5%, Δ target = 0.80, T = 2y. Long leg Δ=0.80 call
(K = 80.8); short leg Δ=−0.80 put (K = 156.2). Both legs sized to $1 of delta exposure, so the
book is delta-neutral at inception.

## 4.1 Exposure drift — the book is not neutral when you need it to be

Long-call delta **rises** with S. Long-put |delta| **falls** with S. So a book that is delta-neutral
when struck is not neutral afterwards.

Net delta exposure, in $ per $1 of an initial leg (strikes fixed at inception, only S and τ change):

| remaining tenor | −60% | −40% | −30% | −20% | −10% | 0% | +10% | +20% | +30% | +50% | +100% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **2.00y** | −0.46 | −0.50 | −0.43 | −0.31 | −0.17 | 0.00 | +0.19 | +0.39 | +0.60 | +1.03 | +2.05 |
| 1.50y | −0.48 | −0.55 | −0.48 | −0.36 | −0.22 | −0.05 | +0.13 | +0.33 | +0.55 | +1.01 | +2.11 |
| 1.00y | −0.49 | −0.61 | −0.55 | −0.41 | −0.25 | −0.10 | +0.07 | +0.25 | +0.47 | +0.97 | +2.19 |
| 0.50y | −0.50 | −0.70 | −0.64 | −0.45 | −0.24 | −0.10 | +0.01 | +0.13 | +0.31 | +0.88 | +2.34 |
| 0.25y | −0.50 | −0.74 | **−0.74** | −0.48 | −0.18 | −0.04 | 0.00 | +0.03 | +0.14 | +0.78 | +2.45 |
| 0.10y | −0.50 | −0.75 | **−0.84** | −0.52 | −0.09 | 0.00 | 0.00 | 0.00 | +0.02 | +0.62 | +2.50 |
| 0.02y | −0.50 | −0.75 | **−0.88** | −0.60 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.25 | +2.50 |

Positive = net long. A stock L/S book reads 0.000 in every cell.

Three readings:

1. **The book is net short after declines and net long after rallies.** Exposure is a
   positively-sloped function of the market. It is a momentum-following exposure profile: the book
   adds long exposure as the market rises and adds short exposure as it falls.
2. **It gets worse as expiry approaches.** A −30% market leaves the book −0.43 short with two
   years left but −0.88 short with two weeks left. Rolling LEAPS forward does not fix this — it
   resets the exposure but realises the drift as a loss each time.
3. **The book is worst positioned exactly at the turning point.** At the trough of a drawdown it
   carries maximum net short exposure. Momentum crashes are rebounds out of drawdowns.

The full path, momentum-crash shape (−50% over 9 months, then +90% over 15 months):

| t (y) | S | net exposure | gross exposure |
|---|---|---|---|
| 0.00 | 100.0 | 0.000 | 2.000 |
| 0.28 | 77.1 | −0.379 | 1.439 |
| 0.59 | 57.8 | −0.568 | 0.845 |
| **0.75 (trough)** | **50.0** | **−0.569** | **0.658** |
| 1.22 | 63.6 | **−0.636** | 0.935 |
| 1.74 | 83.1 | −0.380 | 1.689 |
| 2.00 | 95.0 | 0.000 | 2.375 |

Note gross exposure collapsing from 2.00 to 0.658 — **the book shrinks by two-thirds as it loses**,
which is why it cannot recover. And the net exposure is still −0.636 *after* the rebound has
started, so it is short into the recovery.

This is the mechanism behind the measured up/down beta asymmetry in `03_capm.md` §3.4(d): V1
β = −0.64 on down days, +0.13 on up days.

## 4.2 The static structure is a short strangle

The single most useful thing to know about a buy-and-hold LEAPS L/S position:

Payoff at expiry of (long call at K_c) + (long put at K_p):

```
S_T < K_c        :  K_p − S_T        (falls as S falls — wins)
K_c ≤ S_T ≤ K_p  :  K_p − K_c        CONSTANT
S_T > K_p        :  S_T − K_c        (rises as S rises — wins)
```

Between the two strikes the payoff is **flat**. That is a short strangle.

With our numbers: cost = **1.039** of capital; the constant middle payoff is
`K_p − K_c = 75.4` scaled by n = 0.0125 → **0.943**.

| | value |
|---|---|
| Capital deployed | 1.039 |
| Certain payoff if `S_T ∈ [80.8, 156.2]` | 0.943 |
| **Loss in the middle region** | **−9.3% of capital over 2 years (−4.6%/yr)** |
| PV of that payoff at r = 4.2% | 0.867 → **−16.6% in PV terms (−8.7%/yr)** |
| Lower breakeven (S_T) | 73.1 (**−26.9%** over 2 years) |
| Upper breakeven (S_T) | 163.9 (**+63.9%** over 2 years) |

**You only make money if the stock is down more than 27% or up more than 64% two years later.**
Everything in between loses money. And a lot of that −9.3% is the forfeited American
early-exercise premium from `01_parity_identity.md` §1.3 — you paid the American price and you
collect the European payoff.

Simulated over several two-year paths (American valuation, no frictions; the stock book earns
**0.0%** on every one of these by construction):

| scenario | net exposure at trough | LEAPS P&L (% of capital) |
|---|---|---|
| momentum crash: −50% then +90% | −0.636 | **−9.26%** |
| V-shaped: −35% then +54% (net 0%) | −0.590 | **−9.26%** |
| grind down: −30% then +43% (net 0%) | −0.644 | **−9.26%** |
| melt up: +40% straight | 0.000 | **−9.26%** |
| flat market | −0.107 | **−9.26%** |
| crash only: −50% and stay | −0.625 | +27.81% |

The five identical numbers are not a bug: every one of those paths ends between the strikes, so
they all receive the constant 0.943. **A flat market, a melt-up, and a V-shaped recovery are all
the same trade, and they all lose 9.3%.** Only the crash-that-does-not-recover makes money,
because that is the one path that exits the box.

## 4.3 Ruin: the LEAPS position can go to zero when the company does not

A stock position is wiped out only if the company is. A LEAPS call is wiped out if `S_T < K`.

Δ-targeted call, S = 100, σ = 25%, r = 4.2%, q = 1.5%:

| T | Δ target | K | premium (% spot) | leverage | **P(expire worthless)** | breakeven S_T | breakeven vs forward |
|---|---|---|---|---|---|---|---|
| 1.0y | 0.80 | 85.0 | 19.9% | 4.01× | 26.3% | 105.7 | +2.91% |
| **2.0y** | **0.80** | **80.8** | **26.6%** | **3.01×** | **28.1%** | **109.8** | **+3.98%** |
| 2.0y | 0.90 | 67.1 | 36.6% | 2.46× | 13.5% | 107.0 | +1.32% |
| 2.0y | 0.95 | 54.8 | 47.0% | 2.02× | 4.7% | 105.9 | +0.31% |

The Δ=0.80 two-year call needs the stock **up 9.8% at expiry just to break even** (3.98% above the
forward), and has a 28% risk-neutral chance of expiring worthless.

Empirically, using the project's own universe (overlapping 2-year windows, 2005–2026):

| | median across 35 tickers | worst names |
|---|---|---|
| 1st-pctile 2y return | **−38.8%** | BAC −83.7%, MS −73.4%, NVDA −66.1% |
| 2y windows below −30% | 2.6% | BAC 19.4%, MS 16.0%, UNH 13.3%, INTC 9.3% |
| 2y windows below −50% | 0.0% | BAC 12.9%, MS 6.1%, NVDA 5.8%, UNH 3.4% |

At the median name a Δ=0.80 call (K = 0.81·S, so a −19% two-year return kills it) is safe-ish.
But at the 10-name portfolio level independence does the damage: with a 10% per-name chance of
total loss over two years, **P(at least one leg expires worthless) = 65%**. The tail is not in the
median, it is in the cross-section — and the strategy holds the short leg too, which has its own
ruin region (S_T > 156% of spot).

## 4.4 Capital efficiency: the argument inverts on the short side

Capital required per $1 of delta exposure (American valuation — the European numbers are
optimistic by 9% on the total):

| T | Δ | K_call (% spot) | K_put (% spot) | long leg | **short leg** | **total** |
|---|---|---|---|---|---|---|
| 0.5y | 0.80 | 88.4 | 119.9 | 18.8% | 25.6% | 44.4% |
| 1.0y | 0.80 | 84.9 | 132.3 | 25.3% | 40.6% | 65.8% |
| **2.0y** | **0.80** | **80.8** | **156.2** | **33.7%** | **70.2%** | **103.9%** |
| 2.0y | 0.90 | 67.1 | 188.1 | 40.7% | 97.8% | 138.9% |

Against the alternatives, for a book with $1 long + $1 short delta:

| route | capital | LEAPS as % of it |
|---|---|---|
| **LEAPS, Δ=0.80, T=2y (American)** | **1.04** | — |
| stock, Reg T (long paid + 50% short margin) | 1.50 | **69%** — a saving |
| stock, portfolio margin (~30% each leg) | 0.60 | **173%** — a cost |
| stock, portfolio margin (~20% each leg) | 0.40 | **260%** — a cost |
| stock, fully funded both legs | 2.00 | 52% — a saving |

**The result depends entirely on which side of the margin regime you are on**, and the direction
is the opposite of the conventional pitch:

- **Long leg alone:** 33.7% of exposure in cash. Long options **cannot be margined** — it is all
  cash. Against Reg T (50%) that is a saving; against portfolio margin (20–30%) it is a wash, and
  you are paying 50–100 bps over Treasuries for the embedded financing (`02_embedded_financing.md`)
  instead of a margin spread.
- **Short leg alone:** 70.2% of exposure in cash. This is where the argument dies. A deep-ITM put
  is priced at roughly intrinsic (`K − S` = 56% of spot) because the holder can exercise and
  receive K. You are prepaying the intrinsic and it earns nothing while you hold.
- Note also that exercising the put — the only way to avoid forfeiting the American premium —
  converts it into a cash short, which is the thing it was supposed to replace.

## 4.5 What is *not* structural

For honesty, the things that are genuinely just frictions and would improve with better execution:

- The 50–100 bps financing spread (`02_embedded_financing.md`) — comparable to a margin loan.
- The −0.7%/yr variance-premium drag at Δ≈0.86 (`03_capm.md`) — small for deep ITM.
- The dividend give-up — real, but it is the price of the leverage, not a defect.

Everything in §4.1–4.4 survives perfect execution.
