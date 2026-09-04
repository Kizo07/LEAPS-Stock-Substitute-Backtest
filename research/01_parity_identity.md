# 1. The organising identity: what a "stock substitute" actually is

## 1.1 Put-call parity is the whole story

For any strike `K` and maturity `T`, with `PV(D)` the present value of dividends ex-dated before
`T` and `r` the appropriate discount rate:

```
C − P = (S − PV(D)) − K·e^{−rT}                                   (European, no borrow)
```

This is model-free. No volatility, no distributional assumption, no pricing model. Every result
in this research note is a rearrangement of it.

Read it three ways:

**(a) The synthetic long.** `C − P` is the cost of a *forward*: pay `C`, receive `P`, and you own
`S_T − K` at maturity. Delta is ≈ 1. **This is the only clean stock substitute you can build from
options.** It is a forward, it has no gamma, no theta, no vega, and by construction it embeds
financing at the rate that the market charges for forwards.

**(b) The call alone is not a substitute.** Rearranging:

```
C = (S − PV(D)) − K·e^{−rT} + P
    └── stock ──┘   └─ loan ─┘   └ short put ┘
```

A long call is: **long the stock, funded by a loan of `K·e^{−rT}`, plus short a put.** You do not
own the dividend. Your delta is not 1. You are short the downside below `K`. Comparing "call vs
stock" — which is what RQ1 does — compares a *levered, convex, dividend-excluded, downside-short*
position to an unlevered one. The entire tracking difference lives in those four terms, and
calling the residual a "tracking error" rather than a different asset is the category error at
the centre of this project.

**(c) The put alone is not a short.** Rearranging the other way:

```
P = C − S + PV(D) + K·e^{−rT}
```

A long deep-ITM put is: **short the stock, plus a deposit of `K·e^{−rT}`, plus long a deep-OTM
call.** The deposit is the point. For Δ = −0.80 at T = 2y the strike is 156% of spot
(see §1.2), so the put embeds a deposit worth 56% of spot that earns nothing while you hold it.

## 1.2 The asymmetry nobody expects

"Both legs at Δ = 0.80" sounds symmetric. It is not. Solving `N(d1) = 0.80` for the call and
`N(−d1) = 0.80` for the put, with S=100, σ=25%, r=4.2%, q=1.5%:

| leg | strike | premium (American) | capital per $1 of exposure |
|---|---|---|---|
| call, Δ = +0.80 | **80.8** (19% below spot) | 26.9% of spot | **33.7%** |
| put, Δ = −0.80 | **156.2** (56% above spot) | 56.2% of spot | **70.2%** |

The log-moneyness is asymmetric by `2·(r − q + σ²/2)·T = 2 × 0.1165 = 0.233`, and `σ²T/2`
dominates that at a two-year tenor. Consequences:

- The short leg costs **2.1× more capital** than the long leg for the *same* delta magnitude.
- The short leg is deep ITM, so its American early-exercise premium is large (§1.3).
- At Δ = ±0.80 the deep-OTM call in the synthetic short is nearly worthless, so **"synthetic
  short" and "long put" are almost the same position** — the project's V2 and its put lane are
  not the independent alternatives they look like.

## 1.3 The American early-exercise premium — and who pays it

Equity LEAPS are American. The early-exercise premium on each leg, from a CRR binomial verified
against Black-Scholes (`calcs/binomial.py`), at S=100, σ=25%, T=2y:

| leg | European | American | premium | as % of spot |
|---|---|---|---|---|
| call K=80.8 | 26.61 | 26.9 | 0.3 | 0.3% |
| put K=156.2 | 49.4 | **56.2** | **6.8** | **6.8%** |

Two things follow, and both matter.

**The market prices a deep-ITM American put at roughly intrinsic value** (`K − S = 56.2`), because
the holder can exercise now and receive `K`. A European model — including the Barone-Adesi-Whaley
approximation the project uses — understates it by 6.8% of spot, i.e. by **14% of the put's
price**. This is the same region where the project's 2026-08 bug fix lived (the BAW normal-CDF
argument). Any RQ2 conclusion drawn from European or BAW put prices is materially optimistic
about the short side.

**A stock-substitute strategy never exercises.** Exercising the call converts it to stock;
exercising the put converts it to a cash short. Either way the "substitute" stops being a
substitute. So the strategy pays the American premium and never collects it. That forfeited
premium is a dead-weight cost of about **8.5% of capital over two years, ≈ 4.3%/yr**, and it is
almost entirely on the short leg.

Worked example (S=300, K=400, T=1.5, r=4%, σ=25%):

```
European put   90.27
American put  101.15     (<- what you actually pay)
intrinsic     100.00
```

You pay 101.15 for an instrument whose European value is 90.27. If you hold to expiry you receive
the European payoff. The 10.88 gap is gone.

## 1.4 What this means for the project's research questions

| RQ | Issue |
|---|---|
| **RQ1** (call vs stock) | Comparing a levered convex position to an unlevered one. The correct comparison is synthetic long (a forward) vs stock, whose tracking difference is *only* the financing-spread differential. The project's own numbers show this: call vs stock −2.42%/yr, synth_long vs stock −1.35%/yr. |
| **RQ2** (put vs short stock) | The −7.80%/yr is mostly the American premium plus the fact that a put is not a short. synth_short vs short_stock is −1.00%/yr — as parity says it should be, since a synthetic short *is* a short forward and carries the same economics as a cash short once borrow is paid. |
| **RQ3** (portfolio) | Valid as designed, but the variant labels hide the fact that V1/V3 are net long options (long vega, long convexity) while V0/V2 are not. See `03_capm.md`. |
| **RQ4** (robustness) | Sound, but `delta_target` is near-irrelevant (span 1.2% on CAGR) while the *implementation horizon* — which was never swept — is the dominant parameter. See `06_verdict.md`. |

## 1.5 The one sentence

> Put-call parity says there is exactly one friction-free stock substitute built from options: the
> synthetic (a forward). Everything else — a naked call, a naked put — is a forward *plus* an
> option position whose cost you can compute exactly in advance and whose value you should not
> expect to be zero.
