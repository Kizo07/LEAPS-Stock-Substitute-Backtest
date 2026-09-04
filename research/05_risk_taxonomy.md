# 5. Risk taxonomy

Twelve risks, ordered by how much capital they put at risk, with the number that sizes each one.
References are to the note where the calculation lives.

## Tier 1 — structural, survive perfect execution

### R1. Exposure drift
**The book's net delta is a function of the market, not of your decision.**
Net exposure per $1 of a leg: −0.43 at −30% (2y remaining) down to −0.88 at −30% (2 weeks
remaining); +0.60 at +30%. Gross exposure collapses from 2.00 to 0.66 on a −50% path.
→ `04_structural_risks.md` §4.1
**Mitigation that works:** re-hedge delta with the *cheap* instrument (shares, or a futures/ETF
overlay), not by re-striking the LEAPS.
**Mitigation that does not:** rolling the LEAPS. Rolling resets the exposure but realises the drift
as a loss each time.

### R2. Short-strangle payoff
**Buy-and-hold loses 9.3% of capital (−16.6% in PV) whenever S_T lands between the strikes.**
Breakevens at −26.9% and +63.9% over two years. A flat market, a melt-up and a V-recovery are all
the same trade and all lose.
→ `04_structural_risks.md` §4.2
**Mitigation:** do not hold the pair statically; close or roll well before expiry; or accept that
you are running a short-vol book and size it as one.

### R3. Forfeited early-exercise premium
**The American premium you pay and never collect: 8.5% of capital over 2 years (≈4.3%/yr), 80% of
it on the short leg.** A European or BAW model understates the Δ=−0.80 put by 6.8% of spot — 14%
of its price.
→ `01_parity_identity.md` §1.3
**Mitigation:** price the short leg American (CRR binomial, ~40 lines — see `calcs/binomial.py`).
Do not use BAW for deep-ITM puts.

### R4. Capital efficiency inverts on the short side
**Short leg needs 70.2% of its exposure in cash; long leg 33.7%; total 1.04 per $1+$1.** That is
69% of Reg-T stock capital but **173–260% of portfolio-margin capital.** Long options cannot be
margined.
→ `04_structural_risks.md` §4.4
**Mitigation:** only use LEAPS where you genuinely have no margin line, or use them on the long leg
only.

### R5. Ruin at a strike the company never reaches
**A Δ=0.80 two-year call has a 28% risk-neutral chance of expiring worthless and needs +9.8% at
expiry to break even.** At the median name in the project's universe, 2.6% of two-year windows lose
>30%; at BAC, 12.9% lose >50%. Across 10 names with a 10% per-name rate, P(at least one total
loss) = 65%.
→ `04_structural_risks.md` §4.3
**Mitigation:** Δ ≥ 0.90 halves the ruin probability (28.1% → 13.5%) at the cost of leverage
(3.01× → 2.46×) and capital (33.7% → 40.7%). Diversify the expiry ladder, not just the names.

## Tier 2 — market risks that the stock book does not carry

### R6. Long volatility
**V1 loads +0.0237 on ΔVIX (t = 6.2) where the stock book loads −0.0116.** A +10% VIX day is +24 bp;
a +50% VIX spike is +1.2%. Conversely, in quiet markets the book bleeds.
→ `03_capm.md` §3.4(c)
**Mitigation:** if you want equity replacement and not vol exposure, use synthetics (V2 loads
−0.013, statistically indistinguishable from the stock book).

### R7. Convexity the risk system will not see
**Treynor-Mazuy γ = +6.83 (t = 15.4) for V1, vs −1.32 for V0.** A single beta understates the
response to large moves by ~27 bp on a ±2% day and ~1.7% on a ±5% day.
→ `03_capm.md` §3.4(b)
**Mitigation:** report γ, not just β. Report delta-notional gross/net, not beta × NAV.

### R8. Variance risk premium
**−0.69%/yr of notional at Δ≈0.86 with a 2-vol-point VRP; −1.73%/yr at 5 points.** Deep ITM roughly
halves the ATM drag (−1.04%/yr) but does not remove it.
→ `03_capm.md` §3.3
**Mitigation:** Δ ≥ 0.90, longer tenor, low-vol names. Cannot be eliminated while long options.

### R9. Beta state-dependence
**V1 β = −0.643 on down days, +0.126 on up days — a 0.77 swing, opposite in sign to V0's.**
→ `03_capm.md` §3.4(d)
**Mitigation:** stress-test with conditional betas, not unconditional ones.

## Tier 3 — execution and operations

### R10. Liquidity — the instruments do not trade
**Median quoted bid-ask on deep-ITM LEAPS: 210–286 bps of mid; p90 545–688 bps.** At the strikes
Δ=0.80 needs, open interest is frequently 0–8 contracts and quotes appear *below intrinsic*. Only
SPY, MSFT and AAPL have transactable deep-ITM LEAPS in the cached universe — out of 35 names.
Yahoo's `impliedVolatility` field is unusable at these strikes (returns 0.00001 or 5.45).
→ `02_embedded_financing.md` §2.5
**Mitigation:** restrict the universe to names with real LEAPS OI; treat quoted spreads as a floor,
not an estimate; model market impact explicitly.

### R11. Turnover — the parameter nobody swept
**Monthly re-striking of 2-year LEAPS is the dominant cost and it is not in the sensitivity grid.**
The project reports V1 paying 82.1%/yr in spread costs. With 12 full turnovers of a book whose
round-trip cost is 2–3% of premium on ~104% of exposure, ~40–45%/yr is arithmetically expected;
commissions and the option legs push it higher. `spread_mult` is correctly identified as the
dominant sensitivity (79 pp CAGR span) — but the *horizon* is the lever, and it was never varied.
→ `06_verdict.md`
**Mitigation:** hold LEAPS 12–24 months; re-hedge delta monthly with stock, not by churning options.

### R12. Model, dividend and operations risks

| risk | detail |
|---|---|
| **American put mispricing** | BAW understates deep-ITM puts by ~14% of price. The project fixed a BAW CDF-argument bug in 2026-08; the approximation itself remains the limitation. |
| **Dividend projection** | Trailing-4-quarter projection feeds PV(D), which feeds parity. Errors here move the measured financing spread by tens of bps. Announced dividends are not free — noted as deferred in `ENHANCEMENTS.md`. |
| **Early exercise / assignment** | American calls on dividend payers are exercised when `D > K(1 − e^{−r(T−t)})` net of remaining time value. On a Δ=0.80 call at K=80.8 with quarterly $0.375 dividends, exercise becomes optimal only in the final months; on deeper strikes it is optimal from inception. Each event costs a spread and, for the call, forfeits remaining time value. |
| **Recall risk** | *Absent* on the option route for the short leg — this is its one genuine advantage over a cash short. The option route buys certainty of borrow cost for the full tenor. |
| **Margin** | Long options: no margin, but 100% cash. Short options inside a synthetic: standard option margin, and early assignment on the short call leg around ex-dates. |
| **Pin / expiry risk** | LEAPS expiries cluster on a few annual dates (third Fridays in Jan/Jun/Dec). Concentrated expiry = concentrated forced decision. |
| **Tax** | US equity options held > 12 months get long-term capital-gains treatment — a real advantage. But rolling realises gains annually, converting LTCG into STCG and destroying the benefit. Buy-and-hold-and-exercise preserves it. |
| **Counterparty** | Negligible: OCC-cleared, listed. |

## Important caveat: "structural" vs "policy"

`07_monte_carlo.md` runs both books in a world where CAPM is exactly true and finds that with
**every friction switched off the two books differ by −0.03%/yr** — i.e. the net expected cost of
R1, R2, R3, R6, R7, R8 and R9 together is zero. They are **conditional risks**, not expected costs:
they are large in some states and they change the shape of the return distribution, but they are
symmetric enough to cancel in expectation.

The costs that survive are **policy** costs — how often you roll (R11) and whether you ever
exercise (R3) — plus genuine frictions (R10, R12). So the right reading of the taxonomy below is
"what can hurt me and when", not "what will cost me per year on average."

## Summary table

| # | risk | size | structural? | mitigable? |
|---|---|---|---|---|
| R1 | exposure drift | −0.43 to −0.88 net at −30% | yes | hedge with stock |
| R2 | short-strangle payoff | −9.3% of capital / 2y | yes | don't hold static |
| R3 | forfeited EE premium | −4.3%/yr | yes | price American |
| R4 | capital inversion (short) | 173–260% of PM capital | yes | long leg only |
| R5 | ruin at the strike | 28% P(worthless) | yes | Δ ≥ 0.90, ladder |
| R6 | long vol | β_ΔVIX = +0.024 | yes | use synthetics |
| R7 | convexity | γ = +6.8, t = 15.4 | yes | report γ |
| R8 | variance premium | −0.7 to −1.7%/yr notional | yes | deeper Δ |
| R9 | beta state-dependence | 0.77 up/down swing | yes | conditional stress |
| R10 | liquidity | 210–286 bps median spread | no | restrict universe |
| R11 | turnover | 82%/yr at monthly re-strike | no | hold 12–24m |
| R12 | model / div / tax / ops | misc | mixed | mixed |

**Nine of twelve survive perfect execution.** That is the finding that should drive the decision.
