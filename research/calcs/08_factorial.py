"""Full-combination test: core factorial + space-filling design over every assumption.

Stage A  COMPLETE FACTORIAL over the six assumptions that interact most
         (roll horizon x delta target x vol x American x spread x VRP).
         Every combination is run -- 5 x 3 x 3 x 2 x 2 x 2 = 360 cells.

Stage B  LATIN HYPERCUBE over ALL THIRTEEN assumptions, so that combinations
         outside the core grid (momentum vs random, tenor, borrow, margin,
         rebate, universe size ...) are covered too.

Both stages use COMMON RANDOM NUMBERS (a fixed seed), so cells that differ only
in an implementation parameter see the identical price paths.  That makes the
difference between two cells far more precise than the noise in either cell.

Writes results/mc_factorial.csv incrementally.
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sim_engine as se
import sim_lib as sl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "results"
OUTC = ROOT / "research" / "results"
OUTC.mkdir(parents=True, exist_ok=True)

R, Q = 0.042, 0.015
_GRID_CACHE: dict[tuple, sl.EEGrid] = {}


def grid_for(sig_px: float, r_px: float, q: float) -> sl.EEGrid:
    key = (round(float(sig_px), 2), round(float(r_px), 4), round(float(q), 4))
    g = _GRID_CACHE.get(key)
    if g is None:
        g = sl.EEGrid(r_px, q, key[0], nk=41, nt=32, nsteps=80)
        _GRID_CACHE[key] = g
    return g


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X.T])
    b, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    res = np.asarray(y, float) - X @ b
    n, k = X.shape
    s = np.sqrt(np.diag((res @ res / (n - k)) * np.linalg.pinv(X.T @ X)))
    r2 = 1 - (res @ res) / np.sum((np.asarray(y, float) - np.mean(y)) ** 2)
    return b, b / np.where(s > 0, s, np.nan), r2


def evaluate(cfg: se.Cfg) -> dict:
    """Run one cell and return the headline metrics."""
    r_px = cfg.r + cfg.fspread_bps / 1e4
    sig_px = cfg.sig_stock + cfg.vrp
    ee = grid_for(sig_px, r_px, cfg.q)
    o = se.simulate(cfg, ee)
    yrs = cfg.n_months / 12.0

    def st(ret):
        eq = np.prod(1.0 + ret, axis=1)
        path = np.cumprod(1.0 + ret, axis=1)
        peak = np.maximum.accumulate(path, axis=1)
        return dict(cagr=np.mean(eq ** (1.0 / yrs) - 1.0),
                    vol=np.mean(ret.std(axis=1) * np.sqrt(12.0)),
                    maxdd=np.mean((path / peak - 1.0).min(axis=1)),
                    eq50=np.median(eq))

    ss, ls = st(o["ret_s"]), st(o["ret_l"])

    # CAPM battery on the pooled panel
    x = (o["ret_mkt"] - cfg.r * cfg.dt).ravel()
    row = {}
    for nm, key in (("stock", "ret_s"), ("leaps", "ret_l")):
        y = o[key].ravel()
        b, t, r2 = ols(y, x[:, None])
        b2, t2, _ = ols(y, np.column_stack([x, x ** 2]))
        up, dn = x > 0, x < 0
        bu, _, _ = ols(y[up], x[up][:, None])
        bd, _, _ = ols(y[dn], x[dn][:, None])
        row[f"alpha_{nm}"] = 100 * 12 * (b[0] - cfg.r * cfg.dt)
        row[f"beta_{nm}"] = b[1]
        row[f"gamma_{nm}"] = b2[2]
        row[f"asym_{nm}"] = bd[1] - bu[1]
        row[f"r2_{nm}"] = r2
    row["d_alpha"] = row["alpha_leaps"] - row["alpha_stock"]
    row["d_beta"] = row["beta_leaps"] - row["beta_stock"]
    row["d_gamma"] = row["gamma_leaps"] - row["gamma_stock"]
    row["d_asym"] = row["asym_leaps"] - row["asym_stock"]

    return dict(
        cagr_stock=100 * ss["cagr"], cagr_leaps=100 * ls["cagr"],
        gap=100 * (ls["cagr"] - ss["cagr"]),
        vol_stock=100 * ss["vol"], vol_leaps=100 * ls["vol"],
        maxdd_stock=100 * ss["maxdd"], maxdd_leaps=100 * ls["maxdd"],
        eq50_stock=ss["eq50"], eq50_leaps=ls["eq50"],
        frac_worthless=100 * o["worthless"].sum() / max(o["n_legs"].sum(), 1),
        prem_ratio=o["prem_ratio"],
        **row)


# ------------------------------------------------------------------ designs
METRICS = ["cagr_stock", "cagr_leaps", "gap", "vol_stock", "vol_leaps",
           "maxdd_stock", "maxdd_leaps", "eq50_stock", "eq50_leaps",
           "frac_worthless", "prem_ratio"]
CAPM_COLS = ["alpha_stock", "beta_stock", "gamma_stock", "asym_stock", "r2_stock",
             "alpha_leaps", "beta_leaps", "gamma_leaps", "asym_leaps", "r2_leaps",
             "d_alpha", "d_beta", "d_gamma", "d_asym"]


def core_design():
    rows = []
    for (rm, dt_, sg, am, hs, vrp) in itertools.product(
            [1, 3, 6, 12, 24], [0.70, 0.80, 0.90], [0.20, 0.30, 0.40],
            [True, False], [50.0, 200.0], [0.0, 0.04]):
        rows.append(dict(stage="A_core", rebal_months=rm, delta_tgt=dt_, sig_stock=sg,
                         american=am, half_spread_bps=hs, vrp=vrp))
    return rows


LEVELS = dict(
    rebal_months=[1, 2, 3, 6, 12, 24],
    delta_tgt=[0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    sig_stock=[0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.45],
    q=[0.0, 0.015, 0.030],
    vrp=[0.0, 0.02, 0.05],
    fspread_bps=[0.0, 61.0, 200.0],
    half_spread_bps=[25.0, 100.0, 200.0],
    american=[True, False],
    margin_rate=[0.20, 0.35, 0.50],
    rebate=[0.0, 0.021, 0.042],
    borrow=[0.0025, 0.01, 0.03],
    opt_tenor=[0.5, 1.0, 2.0],
    strategy=["random", "momentum", "beta"],
    n_leg=[5, 10],
)


CANON = METRICS + CAPM_COLS + ["stage"] + list(LEVELS)


def lhs_design(n: int, seed: int = 7):
    """Latin hypercube on the rank scale, then snapped to that parameter's levels."""
    rng = np.random.default_rng(seed)
    keys = list(LEVELS)
    U = np.empty((n, len(keys)))
    for j in range(len(keys)):
        perm = rng.permutation(n)
        U[:, j] = (perm + rng.random(n)) / n
    rows = []
    for i in range(n):
        row = dict(stage="B_lhs")
        for j, k in enumerate(keys):
            lv = LEVELS[k]
            idx = min(int(U[i, j] * len(lv)), len(lv) - 1)
            row[k] = lv[idx]
        rows.append(row)
    return rows


# ------------------------------------------------------------------ runner
def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_lhs = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    design = []
    if which in ("all", "a"):
        design += core_design()
    if which in ("all", "b"):
        design += lhs_design(n_lhs)

    path = OUTC / "mc_factorial.csv"
    done = 0
    if path.exists():
        done = len(pd.read_csv(path))
        print(f"resuming: {done} rows already written")

    print(f"design: {len(design)} cells   (~{len(design)*2.6/60:.0f} min at ~2.6 s/cell)")
    t0 = time.time()
    buf = []
    for i, spec in enumerate(design):
        if i < done:
            continue
        sp = dict(spec)
        stage = sp.pop("stage")
        try:
            base = dict(n_paths=400, n_months=240, n_stocks=20, n_leg=6)
            base.update(sp)
            cfg = se.Cfg(**base)
            rec = evaluate(cfg)
        except Exception as e:                       # keep the sweep alive
            print(f"  cell {i} FAILED: {e}")
            rec = {k: np.nan for k in
                   ["cagr_stock", "cagr_leaps", "gap", "vol_stock", "vol_leaps",
                    "maxdd_stock", "maxdd_leaps", "eq50_stock", "eq50_leaps",
                    "frac_worthless", "prem_ratio", "d_alpha", "d_beta", "d_gamma",
                    "d_asym", "r2_stock", "r2_leaps"]}
        rec.update(stage=stage, **sp)
        buf.append(rec)
        if (i + 1) % 25 == 0:
            batch = pd.DataFrame(buf)
            # canonical column order: the two stages carry different parameter
            # sets, so every batch must be reindexed or the CSV goes ragged
            batch = batch.reindex(columns=CANON)
            batch.to_csv(path, mode="a", header=not path.exists(), index=False)
            el = time.time() - t0
            print(f"  {i+1}/{len(design)}  {el:.0f}s  eta {el/(i+1-done)*(len(design)-i-1)/60:.1f} min"
                  f"   last gap {rec['gap']:+.2f}")
            buf = []
    if buf:
        pd.DataFrame(buf).reindex(columns=CANON).to_csv(
            path, mode="a", header=not path.exists(), index=False)
    print(f"done: {len(design)} cells in {(time.time()-t0)/60:.1f} min -> {path}")


if __name__ == "__main__":
    main()
