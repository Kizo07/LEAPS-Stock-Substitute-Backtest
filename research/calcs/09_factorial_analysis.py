"""Analysis of the full-combination sweep in results/mc_factorial.csv.

  A1  distribution of the gap across every combination tested
  A2  main effects (marginal means per parameter level)
  A3  variance decomposition: which assumptions actually decide the answer
  A4  two-way interaction tables
  A5  the favourable region: when (if ever) does LEAPS win
  A6  robustness of the CAPM findings across the whole space
  A7  best / worst combinations
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "results"

PARAMS = ["rebal_months", "delta_tgt", "sig_stock", "q", "vrp", "fspread_bps",
          "half_spread_bps", "american", "margin_rate", "rebate", "borrow",
          "opt_tenor", "strategy", "n_leg"]
NUM = ["rebal_months", "delta_tgt", "sig_stock", "q", "vrp", "fspread_bps",
       "half_spread_bps", "margin_rate", "rebate", "borrow", "opt_tenor", "n_leg"]


# Stage A (the core factorial) varies six parameters and holds the rest at these
# Cfg baselines; record them so all 784 cells can enter the same regression.
BASELINE = dict(q=0.015, fspread_bps=61.0, margin_rate=0.50, rebate=0.042,
                borrow=0.0025, opt_tenor=2.0, n_leg=6)
BASELINE_CAT = dict(strategy="random")


def load():
    df = pd.read_csv(OUT / "mc_factorial.csv")
    df = df.dropna(subset=["gap"])
    for c in NUM:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c, v in BASELINE.items():
        if c in df:
            df[c] = df[c].fillna(v)
    for c, v in BASELINE_CAT.items():
        if c in df:
            df[c] = df[c].fillna(v)
    return df


def main_effects(df, by_stage=None):
    rows = []
    for p in PARAMS:
        if p not in df:
            continue
        sub = df.dropna(subset=[p])
        if by_stage:
            sub = sub[sub.stage == by_stage]
        if sub[p].nunique() < 2:
            continue
        g = sub.groupby(p)["gap"].agg(["size", "mean"])
        rows.append(dict(param=p, n_levels=int(sub[p].nunique()),
                         range_pp=round(g["mean"].max() - g["mean"].min(), 2),
                         best_level=g["mean"].idxmax(),
                         best_gap=round(g["mean"].max(), 2),
                         worst_level=g["mean"].idxmin(),
                         worst_gap=round(g["mean"].min(), 2)))
    return pd.DataFrame(rows).sort_values("range_pp", ascending=False), None


def anova(df, target="gap", interactions=True):
    """OLS on standardised numerics + dummies; returns a variance decomposition."""
    d = df.dropna(subset=[target]).copy()
    # every column used in the design must be numeric and finite, or the SVD
    # will not converge (this bit me once via a mis-aligned CSV column)
    for c in NUM + [target]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    X = pd.DataFrame(index=d.index)
    terms = []
    for p in NUM:
        if p in d and d[p].nunique() > 1:
            v = d[p].astype(float)
            v = (v - v.mean()) / (v.std() + 1e-12)
            X[f"z_{p}"] = v
            terms.append(f"z_{p}")
    for p in ["american", "strategy"]:
        if p in d and d[p].nunique() > 1:
            dm = pd.get_dummies(d[p].astype(str), prefix=f"d_{p}", drop_first=True, dtype=float)
            X = pd.concat([X, dm], axis=1)
            terms += list(dm.columns)
    if interactions:
        pairs = [("rebal_months", "half_spread_bps"), ("delta_tgt", "half_spread_bps"),
                 ("sig_stock", "american"), ("rebal_months", "delta_tgt"),
                 ("sig_stock", "half_spread_bps"), ("rebal_months", "sig_stock"),
                 ("opt_tenor", "rebal_months"), ("delta_tgt", "sig_stock"),
                 ("american", "half_spread_bps"), ("margin_rate", "rebate")]
        for a, b in pairs:
            if f"z_{a}" in X and f"z_{b}" in X:
                X[f"i_{a}_{b}"] = X[f"z_{a}"] * X[f"z_{b}"]
                terms.append(f"i_{a}_{b}")

    Xm = X[terms].to_numpy(float)
    y = d[target].to_numpy(float)
    ok = np.isfinite(Xm).all(axis=1) & np.isfinite(y)
    Xm, y = Xm[ok], y[ok]
    print(f"   (ANOVA on {len(y)} of {len(d)} cells; "
          f"{len(d)-len(y)} dropped for non-finite design entries)")
    A = np.column_stack([np.ones(len(y)), Xm])
    b, res, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ b
    r2_full = 1 - np.var(y - fit) / np.var(y)

    # incremental R^2 of each term (one at a time, over a base of all others)
    rows = []
    for i, t in enumerate(terms):
        keep = [j for j in range(len(terms)) if j != i]
        Ab = np.column_stack([np.ones(len(y)), Xm[:, keep]])
        bb, *_ = np.linalg.lstsq(Ab, y, rcond=None)
        r2_no = 1 - np.var(y - Ab @ bb) / np.var(y)
        rows.append(dict(term=t, coef=round(b[i + 1], 3),
                         dR2=round(r2_full - r2_no, 4)))
    out = pd.DataFrame(rows).sort_values("dR2", ascending=False)
    return out, r2_full


def main():
    df = load()
    print("=" * 100)
    print(f"A1  DISTRIBUTION OF THE GAP OVER ALL {len(df)} COMBINATIONS TESTED")
    print("=" * 100)
    g = df["gap"]
    q = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(pd.DataFrame({"percentile": [f"p{p}" for p in q],
                        "gap_%/yr": [round(np.percentile(g, p), 2) for p in q]}).to_string(index=False))
    print(f"\nmean {g.mean():+.2f} %/yr   sd {g.std():.2f}")
    print(f"P(gap > 0)  i.e. LEAPS beats stock      : {100*(g>0).mean():.1f}%")
    print(f"P(gap > -3%/yr)  i.e. competitive       : {100*(g>-3).mean():.1f}%")
    print(f"P(gap < -10%/yr) i.e. clearly dominated : {100*(g<-10).mean():.1f}%")
    print(f"\nby stage:")
    print(df.groupby("stage")["gap"].agg(["size", "mean", "std", "min", "max"]).round(2).to_string())

    print("\n" + "=" * 100)
    print("A2  MAIN EFFECTS  (range = worst level minus best level, in percentage points of CAGR)")
    print("=" * 100)
    me, _ = main_effects(df)
    print(me.to_string(index=False))

    print("\n" + "=" * 100)
    print("A3  VARIANCE DECOMPOSITION  (incremental R^2 of each term, all others held in)")
    print("=" * 100)
    av, r2 = anova(df)
    print(av.to_string(index=False))
    print(f"\nmodel R^2 = {r2:.3f}")
    print("\n'z_' terms are standardised: the coefficient is the effect of a 1-sd move.")
    print("'i_' terms are interactions. 'd_american_True' / 'd_strategy_*' are dummies.")

    print("\n" + "=" * 100)
    print("A4  INTERACTION TABLES  (gap, %/yr)")
    print("=" * 100)
    core = df[(df.stage == "A_core")]
    for a, b in [("rebal_months", "delta_tgt"), ("sig_stock", "american"),
                 ("rebal_months", "half_spread_bps"), ("delta_tgt", "half_spread_bps")]:
        if a in core and b in core:
            print(f"\n--- {a} (rows) x {b} (cols) ---")
            print(core.pivot_table(index=a, columns=b, values="gap", aggfunc="mean").round(1).to_string())

    print("\n" + "=" * 100)
    print("A5  THE FAVOURABLE REGION")
    print("=" * 100)
    for thr in (0, -2, -5):
        s = df[df["gap"] > thr]
        if len(s) == 0:
            print(f"\ngap > {thr}: no combinations qualify")
            continue
        print(f"\ngap > {thr} %/yr  ({len(s)} of {len(df)} combinations, {100*len(s)/len(df):.0f}%):")
        for p in PARAMS:
            if p in s and s[p].nunique() > 1 and df[p].nunique() > 1:
                vc = s[p].value_counts(normalize=True)
                print(f"   {p:<18s} " + "  ".join(f"{k}:{100*v:.0f}%" for k, v in vc.head(4).items()))

    print("\n" + "=" * 100)
    print("A6  ROBUSTNESS OF THE CAPM FINDINGS ACROSS THE WHOLE SPACE")
    print("=" * 100)
    for col, lab in [("d_alpha", "alpha gap (LEAPS - stock), %/yr"),
                     ("d_gamma", "Treynor-Mazuy gamma gap"),
                     ("d_beta", "beta gap (LEAPS - stock)"),
                     ("d_asym", "up/down beta asymmetry gap")]:
        if col in df:
            v = df[col].dropna()
            print(f"\n{lab}:")
            print(f"   median {v.median():+.3f}   p05 {v.quantile(0.05):+.3f}   p95 {v.quantile(0.95):+.3f}")
            print(f"   share of combinations with the same sign as the median: "
                  f"{100*max((v>0).mean(),(v<0).mean()):.1f}%")
    if {"r2_stock", "r2_leaps"} <= set(df.columns):
        r = df[["r2_stock", "r2_leaps"]].dropna()
        print(f"\nR^2 of the market regression: stock {r['r2_stock'].median():.3f} vs "
              f"LEAPS {r['r2_leaps'].median():.3f}; LEAPS lower in "
              f"{100*(r['r2_leaps']<r['r2_stock']).mean():.0f}% of combinations")

    print("\n" + "=" * 100)
    print("A7  BEST AND WORST COMBINATIONS")
    print("=" * 100)
    show = ["gap", "cagr_stock", "cagr_leaps", "rebal_months", "delta_tgt", "sig_stock",
            "american", "half_spread_bps", "vrp", "q", "opt_tenor", "strategy",
            "margin_rate", "rebate", "borrow", "n_leg", "fspread_bps"]
    show = [c for c in show if c in df.columns]
    print("\n--- 10 best ---")
    print(df.nlargest(10, "gap")[show].round(2).to_string(index=False))
    print("\n--- 10 worst ---")
    print(df.nsmallest(10, "gap")[show].round(2).to_string(index=False))

    me.to_csv(OUT / "mc_factorial_effects.csv", index=False)
    av.to_csv(OUT / "mc_factorial_anova.csv", index=False)
    print("\nwrote mc_factorial_effects.csv, mc_factorial_anova.csv")


if __name__ == "__main__":
    main()
