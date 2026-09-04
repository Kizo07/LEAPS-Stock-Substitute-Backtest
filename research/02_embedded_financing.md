# 2. The financing rate actually embedded in quoted LEAPS

*Code: `calcs/01_implied_financing.py` · Data: cached chains `data/chain_*.parquet`, FRED `data/fred_curve.parquet`, snapshot 2026-07-31 · Output: `results/financing_leaps.csv`, `financing_summary.csv`*

## 2.1 The claim under test

The standard pitch is that a deep-ITM LEAPS call gives you stock exposure and frees capital. The
freed capital is not free: by buying the call instead of the stock you are implicitly borrowing
`K·e^{−fT}`. The question is what `f` is. If `f = r` the leverage is free and the pitch holds; if
`f = r + 300bp` you are running a margin loan at a bad rate.

## 2.2 An estimator with no volatility in it

Put-call parity with an unknown effective financing rate `f`:

```
C − P = (S − PV(D)) − K·e^{−fT}
```

The time value of a call and a put at the same strike is *identical*, so differencing removes it.
The estimator is therefore **completely free of any vol or vol-surface assumption**, which is the
dominant error source in the usual approach. The only remaining term is the American
early-exercise premium on the call, which we subtract explicitly using the binomial pricer:

```
f = −(1/T) · ln[ (S − PV(D) − (C − eep_call − P)) / K ]
```

Validity requires the put to be deep OTM so that its own American premium is zero; enforced
contract-by-contract with a binomial check (`eep_put > 0.5% of spot` → rejected).

### Validation

Two estimators were built first and both were wrong before this one. The record is worth keeping:

| estimator | failure |
|---|---|
| Call-only parity `f = −(1/T)ln((S − PV(D) − C)/K)` | ignores time value; biased +1.6 to +137 bps depending on moneyness |
| American-model refit at a fitted IV surface | IV surface fit R² = 0.31 and 56% of deep-ITM contracts required extrapolation; the answer moved by 100+ bps with the surface |
| **`C − P` with eep correction** | recovers the true `f` to **< 1 bp** for every strike and every volatility from 15% to 70% |

None of this caught the largest error, which was not in the estimator at all.

### A units bug worth recording

FRED's `DGS*` series are cached in `data/fred_curve.parquet` as **decimals** (`0.0426` = 4.26%),
not percent. Dividing by 100 a second time gave `r = 0.0004` instead of `0.0392` and inflated
every measured financing spread by ≈ **390 bps** — producing a clean, consistent, completely
fictitious result of "+400 bps/yr embedded financing" that survived sanity checks because it was
plausible-looking and stable across tickers. The loader now range-checks and hard-fails:

```python
if v > 1.5: v = v / 100.0
if not 0.0 < v < 0.25: raise ValueError(...)
```

Lesson applied throughout: **when a measurement produces a suspiciously round and suspiciously
stable number, suspect the units before the economics.**

## 2.3 Result

Treasuries on the snapshot date: 3M 3.83%, 1Y 4.04%, 2Y 4.22%. Deep-ITM pairs with 300–900 DTE.

**Restricted to the highest-quality contracts** (quoted spread ≤ 200 bps, open interest ≥ 100) —
this is SPY, MSFT and AAPL only, because no other name in the cached universe has liquid
deep-ITM LEAPS:

| ticker | n | median spread (bps) | median OI | **f − r at mid** | **f − r crossing the offer** | ±1 half-spread error |
|---|---|---|---|---|---|---|
| SPY | 43 | 161 | 290 | **+72 bps** | +102 bps | ±36 bps |
| MSFT | 40 | 181 | 220 | **+47 bps** | +136 bps | ±81 bps |
| AAPL | 25 | 184 | 267 | **+162 bps** | +303 bps | ±142 bps |
| pooled | 108 | 178 | 250 | **+61 bps** | +131 bps | ±69 bps |

Full universe (all quality levels, 858 pairs): pooled median **+55 bps**, +146 bps at the offer.
KO (−64 bps) and XOM (−67 bps) come out negative, but both have 600+ bps quoted spreads, n < 40,
and error bars of ±170–190 bps, so they are not informative.

## 2.4 Interpretation

**The embedded leverage costs roughly 50–100 bps over Treasuries at mid, and 130–300 bps once you
cross the spread.** That is:

- **Comparable to a margin loan**, not cheaper than one. Typical retail/institutional margin runs
  SOFR + 50–150 bps. LEAPS are in the same band.
- **An order of magnitude smaller than the dividend give-up** on a dividend payer (SPY 1.0%, JPM
  1.7%, KO 2.4%, XOM 2.6%/yr).
- **Smaller than the measurement error from the bid-ask alone.** The ±69 bps error bar from a
  single half-spread is as large as the effect. You cannot identify this number precisely from one
  snapshot, and more importantly: **the transaction cost is the same size as the financing cost.**

AAPL's +162 bps is the outlier. The snapshot day was a heavy idiosyncratic day (AAPL −7.4%,
MSFT +3.0%), so its chain is the least reliable; treat it as an upper bound.

## 2.5 Second finding: the instruments do not trade

This fell out of the same data and is more important than the financing number.

Bid-ask on deep-ITM LEAPS calls (all pairs, 858 contracts):

| moneyness K/S | n | median spread | p90 spread |
|---|---|---|---|
| 0.20–0.40 | 123 | 210 bps | 688 bps |
| 0.40–0.60 | 341 | 210 bps | 545 bps |
| 0.60–0.75 | 293 | 234 bps | 613 bps |
| 0.75–0.80 | 101 | 286 bps | 592 bps |

And at the strikes the Δ = 0.80 construction actually needs, open interest collapses. From the
SPY 2027-12-17 chain (505 DTE), spot 747:

```
strike  bid      ask      OI     note
 50     694.78   699.50     2    bid is BELOW intrinsic (697.03) - stale, untradeable
 85     668.00   672.50     0
 95       0.00     0.00     0    lastPrice 562.90 with no bid or ask
125       0.00     0.00     0
200     550.00   554.77   323    <- the deepest strike with real size
```

Yahoo's `impliedVolatility` field at these strikes is also unusable: values of `0.000010`,
`1.209477`, `5.453128` appear on contracts with zero open interest.

**Only SPY, MSFT and AAPL have deep-ITM LEAPS you could actually transact in.** The project's
universe is 33 names plus two ETFs. A backtest that assumes monthly re-striking at Δ = 0.80 in all
35 is assuming liquidity that does not exist — and this is before modelling market impact, which
on a 200–300 bps quoted spread with single-digit OI is unbounded.

## 2.6 What the project assumed

The project's sensitivity grid sweeps `spread_mult`, `iv_mult_adj`, `delta_target` and four
others, and finds `spread_mult` dominant (79 pp span on V1 CAGR). Given §2.5, that is the right
variable to worry about — but the grid varies it around a calibrated level derived from quoted
spreads. The calibration is measuring instruments that, in the deep-ITM region where the strategy
lives, have open interest of zero to eight contracts. The true spread for size is wider than the
calibration, not narrower.
