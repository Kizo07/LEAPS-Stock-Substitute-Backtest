"""Monte Carlo study: stock L/S vs LEAPS L/S in a world where CAPM is true.

  E1  baseline headline
  E2  attribution of the LEAPS - stock gap, one friction at a time
  E3  P&L decomposition of the LEAPS book (delta / theta / gamma / interest)
  E4  CAPM battery
  E5  exposure drift, conditioned on the market move since the last roll
  E6  ruin
  E7  sensitivity sweeps
  E8  momentum variant

Outputs: research/results/mc_*.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sim_engine as se
import sim_lib as sl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

R, Q, SIG0 = 0.042, 0.015, 0.30
EE = sl.EEGrid(R, Q, SIG0, nsteps=140)


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X.T])
    b, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    res = np.asarray(y, float) - X @ b
    n, k = X.shape
    se_ = np.sqrt(np.diag((res @ res / (n - k)) * np.linalg.pinv(X.T @ X)))
    r2 = 1 - (res @ res) / np.sum((np.asarray(y, float) - np.mean(y)) ** 2)
    return b, b / np.where(se_ > 0, se_, np.nan), r2


def stats(ret, yrs):
    eq = np.prod(1.0 + ret, axis=1)
    vol = np.mean(ret.std(axis=1) * np.sqrt(12.0))
    mu = np.mean(ret.mean(axis=1) * 12.0)
    path = np.cumprod(1.0 + ret, axis=1)
    peak = np.maximum.accumulate(path, axis=1)
    return dict(cagr=np.mean(eq ** (1.0 / yrs) - 1.0), vol=vol, mu=mu,
                sharpe=mu / max(vol, 1e-9),
                maxdd=np.mean((path / peak - 1.0).min(axis=1)),
                eq_p05=np.percentile(eq, 5), eq_p50=np.median(eq), eq_p95=np.percentile(eq, 95))


def run(cfg):
    o = se.simulate(cfg, EE)
    return o, stats(o["ret_s"], cfg.n_months / 12.0), stats(o["ret_l"], cfg.n_months / 12.0)


# ---------------------------------------------------------------- E1
def e1():
    cfg = se.Cfg(n_paths=4000, n_months=240, n_stocks=30, n_leg=10)
    o, ss, ls = run(cfg)
    print("=" * 100)
    print("E1  BASELINE  20y, 4000 paths, 30 names, 10 per side, monthly steps, quarterly rolls")
    print("=" * 100)
    print(pd.DataFrame([dict(book="stock L/S", **{k: round(v, 4) for k, v in ss.items()}),
                        dict(book="LEAPS L/S", **{k: round(v, 4) for k, v in ls.items()})]).to_string(index=False))
    print(f"\ncapital per $1+$1 of delta exposure: LEAPS {o['prem_ratio']:.3f}  vs stock {1+cfg.margin_rate:.3f} (Reg T)")
    print(f"gap (LEAPS - stock)   : {100*(ls['cagr']-ss['cagr']):+.2f} %/yr")
    print(f"vol ratio             : {ls['vol']/ss['vol']:.3f}")
    print(f"median terminal wealth: LEAPS {ls['eq_p50']:.3f}  vs stock {ss['eq_p50']:.3f}")
    print(f"both books are credited r on all posted capital, so r = {100*R:.1f}% is the common")
    print(f"baseline; the gap is the number that matters.")
    return cfg, o, ss, ls


# ---------------------------------------------------------------- E2
def e2():
    print("\n" + "=" * 100)
    print("E2  ATTRIBUTION  (each row switches ONE friction off; gap = LEAPS - stock CAGR)")
    print("=" * 100)
    v = {
        "all frictions on (baseline)": dict(),
        "no bid-ask": dict(half_spread_bps=0.0),
        "no funding spread": dict(fspread_bps=0.0),
        "no dividend": dict(q=0.0),
        "European pricing (no EE premium)": dict(american=False),
        "no borrow on stock short": dict(borrow=0.0),
        "ZERO frictions": dict(half_spread_bps=0.0, fspread_bps=0.0, q=0.0, american=False, borrow=0.0),
    }
    rows = []
    for nm, ov in v.items():
        cfg = se.Cfg(n_paths=2000, n_months=240, n_stocks=30, n_leg=10, **ov)
        o, ss, ls = run(cfg)
        rows.append(dict(variant=nm, cagr_stock=round(100 * ss["cagr"], 2),
                         cagr_leaps=round(100 * ls["cagr"], 2),
                         gap=round(100 * (ls["cagr"] - ss["cagr"]), 2),
                         vol_leaps=round(100 * ls["vol"], 2)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    b = df.iloc[0]["gap"]
    print("\ncost attributable to each friction (= change in the gap when it is switched off):")
    for _, r in df.iloc[1:6].iterrows():
        print(f"   {r['variant']:<34s} {r['gap']-b:+6.2f} %/yr")
    print(f"\n   pure structure (all frictions off) : {df.iloc[-1]['gap']:+.2f} %/yr")
    print(f"   total friction load                : {df.iloc[-1]['gap']-b:+.2f} %/yr")
    return df


# ---------------------------------------------------------------- E3
def e3():
    print("\n" + "=" * 100)
    print("E3  P&L DECOMPOSITION OF THE LEAPS BOOK  (mean monthly contribution, % of equity)")
    print("=" * 100)
    rows = []
    for rm in (1, 3, 6, 12, 24):
        cfg = se.Cfg(n_paths=2000, n_months=240, n_stocks=30, n_leg=10, rebal_months=rm)
        o, ss, ls = run(cfg)
        f = lambda a: round(100 * o[a].mean(), 4)
        rows.append(dict(roll_months=min(rm, 24), delta=f("pl_delta"), theta=f("pl_theta"),
                         gamma=f("pl_gamma"), interest=f("pl_interest"),
                         spread=round(-100 * o["spread_cost"].mean(), 4),
                         total=round(100 * o["ret_l"].mean(), 4),
                         cagr_leaps=round(100 * ls["cagr"], 2), cagr_stock=round(100 * ss["cagr"], 2)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nspread is negative (a cost); delta/gamma/interest/theta are gross contributions.")
    print("'total' is the mean monthly arithmetic return of the LEAPS book.")
    return df


# ---------------------------------------------------------------- E4
def e4():
    print("\n" + "=" * 100)
    print("E4  CAPM BATTERY  (pooled paths x months; x = market excess return)")
    print("=" * 100)
    rows = []
    for strat in ("random", "beta"):
        cfg = se.Cfg(n_paths=2000, n_months=240, n_stocks=30, n_leg=10, strategy=strat)
        o, ss, ls = run(cfg)
        x = (o["ret_mkt"] - cfg.r * cfg.dt).ravel()
        rec = dict(strategy=strat)
        for nm, key in (("stock", "ret_s"), ("LEAPS", "ret_l")):
            y = o[key].ravel()
            b, t, r2 = ols(y, x[:, None])
            b2, t2, _ = ols(y, np.column_stack([x, x ** 2]))
            up, dn = x > 0, x < 0
            bu, _, _ = ols(y[up], x[up][:, None])
            bd, _, _ = ols(y[dn], x[dn][:, None])
            rec.update({
                f"a_{nm}": round(100 * 12 * (b[0] - cfg.r * cfg.dt), 2), f"at_{nm}": round(t[0], 1),
                f"b_{nm}": round(b[1], 4), f"r2_{nm}": round(r2, 4),
                f"g_{nm}": round(b2[2], 2), f"gt_{nm}": round(t2[2], 1),
                f"bu_{nm}": round(bu[1], 4), f"bd_{nm}": round(bd[1], 4)})
        rec["d_alpha"] = round(rec["a_LEAPS"] - rec["a_stock"], 2)
        rec["d_beta"] = round(rec["b_LEAPS"] - rec["b_stock"], 4)
        rec["d_gamma"] = round(rec["g_LEAPS"] - rec["g_stock"], 2)
        rec["asym_stock"] = round(rec["bd_stock"] - rec["bu_stock"], 4)
        rec["asym_leaps"] = round(rec["bd_LEAPS"] - rec["bu_LEAPS"], 4)
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\na_ = annualised intercept in EXCESS of the risk-free rate (so a zero-beta,")
    print("zero-cost book reads 0.00).  d_* columns are LEAPS minus stock -- the numbers")
    print("that isolate the effect of the option layer.")
    return df


# ---------------------------------------------------------------- E5
def e5():
    print("\n" + "=" * 100)
    print("E5  EXPOSURE DRIFT  (net delta exposure vs the market move since the last roll)")
    print("=" * 100)
    for rm in (3, 12):
        cfg = se.Cfg(n_paths=2000, n_months=240, n_stocks=30, n_leg=10, rebal_months=rm)
        o, _, _ = run(cfg)
        m = o["net_delta"].ravel()
        g = o["gross_delta"].ravel()
        msr = o["mkt_since_reset"].ravel()
        b = pd.qcut(msr, 6, labels=["< -15%", "-15:-8", "-8:-3", "-3:+3", "+3:+8", "> +8%"],
                    duplicates="drop")
        d = pd.DataFrame(dict(bucket=np.asarray(b), net=m, gross=g))
        t = d.groupby("bucket", observed=True).agg(
            n=("net", "size"), net_delta=("net", "mean"), net_sd=("net", "std"),
            gross_delta=("gross", "mean")).round(4)
        print(f"\n--- rolls every {rm} months ---")
        print(t.to_string())
    print("\nThe STOCK book reads exactly 0.000 net in every bucket (rebalanced monthly).")
    print("Positive net = the book has become net long.  Exposure rises with the market.")
    return None


# ---------------------------------------------------------------- E6
def e6():
    print("\n" + "=" * 100)
    print("E6  RUIN  (fraction of LEAPS legs finishing out of the money)")
    print("=" * 100)
    rows = []
    for tenor in (1.0, 2.0):
        for d in (0.70, 0.80, 0.90, 0.95):
            cfg = se.Cfg(n_paths=1500, n_months=120, n_stocks=20, n_leg=6, opt_tenor=tenor, delta_tgt=d)
            o, _, _ = run(cfg)
            frac = o["worthless"].sum() / max(o["n_legs"].sum(), 1)
            # legs per path-year
            per_yr = o["n_legs"].sum() / (1500 * 10)
            rows.append(dict(tenor=tenor, delta=d,
                             frac_worthless=round(100 * frac, 2),
                             worthless_per_path_yr=round(frac * per_yr, 2),
                             p_ge1_worthless_pa=round(100 * (o["worthless"] >= 1).mean(), 1)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\n'per path yr' = expected number of legs per year that expire worthless.")
    return df


# ---------------------------------------------------------------- E7
def e7():
    print("\n" + "=" * 100)
    print("E7  SENSITIVITY SWEEPS  (gap = LEAPS - stock CAGR, %/yr)")
    print("=" * 100)
    frames = []
    for key, vals in [("rebal_months", [1, 3, 6, 12, 24]),
                      ("delta_tgt", [0.70, 0.80, 0.90, 0.95]),
                      ("sig_stock", [0.20, 0.30, 0.40]),
                      ("q", [0.0, 0.015, 0.030]),
                      ("vrp", [0.0, 0.02, 0.05]),
                      ("fspread_bps", [0.0, 61.0, 200.0]),
                      ("margin_rate", [0.20, 0.35, 0.50]),
                      ("rebate", [0.0, 0.032, 0.042])]:
        rows = []
        for v in vals:
            cfg = se.Cfg(n_paths=1500, n_months=240, n_stocks=30, n_leg=10, **{key: v})
            o, ss, ls = run(cfg)
            rows.append(dict(param=key, value=v,
                             cagr_stock=round(100 * ss["cagr"], 2),
                             cagr_leaps=round(100 * ls["cagr"], 2),
                             gap=round(100 * (ls["cagr"] - ss["cagr"]), 2),
                             vol_leaps=round(100 * ls["vol"], 2),
                             eq_leaps=round(ls["eq_p50"], 3), eq_stock=round(ss["eq_p50"], 3)))
        f = pd.DataFrame(rows)
        frames.append(f)
        print(f"\n--- {key} ---")
        print(f.to_string(index=False))

    # delta target with frictions OFF (isolates the structural trade-off)
    print("\n--- delta_tgt with ZERO frictions ---")
    rows = []
    for d in (0.70, 0.80, 0.90, 0.95):
        cfg = se.Cfg(n_paths=1500, n_months=240, n_stocks=30, n_leg=10, delta_tgt=d,
                     half_spread_bps=0.0, fspread_bps=0.0, q=0.0, american=False, borrow=0.0)
        o, ss, ls = run(cfg)
        rows.append(dict(delta_tgt=d, gap=round(100 * (ls["cagr"] - ss["cagr"]), 2),
                         vol_leaps=round(100 * ls["vol"], 2),
                         frac_worthless=round(100 * o["worthless"].sum() / max(o["n_legs"].sum(), 1), 2)))
    f = pd.DataFrame(rows)
    print(f.to_string(index=False))
    frames.append(f.assign(param="delta_tgt_nofric"))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- E8
def e8():
    print("\n" + "=" * 100)
    print("E8  MOMENTUM VARIANT  (12-1 momentum; 'after crash' = month after a worst-5% 3m move)")
    print("=" * 100)
    rows = []
    for strat in ("random", "momentum", "beta"):
        cfg = se.Cfg(n_paths=2000, n_months=240, n_stocks=30, n_leg=10, strategy=strat)
        o, ss, ls = run(cfg)
        trail = pd.DataFrame(o["ret_mkt"]).rolling(3, min_periods=3).sum().to_numpy()
        m = ~np.isnan(trail)
        w = (trail[m] <= np.nanpercentile(trail[m], 5)).ravel()
        rows.append(dict(strategy=strat,
                         cagr_stock=round(100 * ss["cagr"], 2), cagr_leaps=round(100 * ls["cagr"], 2),
                         gap=round(100 * (ls["cagr"] - ss["cagr"]), 2),
                         maxdd_stock=round(100 * ss["maxdd"], 1), maxdd_leaps=round(100 * ls["maxdd"], 1),
                         net_delta_after_crash=round(o["net_delta"][m][w].mean(), 4),
                         ret_leaps_after_crash=round(100 * o["ret_l"][m][w].mean(), 3),
                         ret_stock_after_crash=round(100 * o["ret_s"][m][w].mean(), 3)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def main():
    _, o, ss, ls = e1()
    a = e2(); d = e3(); c = e4(); e5(); ru = e6(); sw = e7(); mo = e8()
    pd.DataFrame([dict(book="stock", **ss), dict(book="leaps", **ls)]).to_csv(OUT / "mc_baseline.csv", index=False)
    a.to_csv(OUT / "mc_attribution.csv", index=False)
    d.to_csv(OUT / "mc_decomposition.csv", index=False)
    c.to_csv(OUT / "mc_capm.csv", index=False)
    ru.to_csv(OUT / "mc_ruin.csv", index=False)
    sw.to_csv(OUT / "mc_sweeps.csv", index=False)
    mo.to_csv(OUT / "mc_momentum.csv", index=False)
    print("\nwrote mc_*.csv")


if __name__ == "__main__":
    main()
