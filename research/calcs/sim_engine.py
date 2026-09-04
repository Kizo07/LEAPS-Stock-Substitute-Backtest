"""Monte Carlo engine: a single-factor equity world, stock L/S vs LEAPS L/S.

THE WORLD
---------
One market factor plus idiosyncratic noise, ARITHMETIC (not log) returns so that
CAPM holds EXACTLY for the underlying stocks:

    R_M = mu_M dt + sig_M sqrt(dt) z_M
    R_i = mu_i dt + beta_i sig_M sqrt(dt) z_M + sig_eps,i sqrt(dt) z_i
    mu_i = r + beta_i (mu_M - r)          <- CAPM, exact by construction

`R_i` is the TOTAL return; the price return is `R_i - q dt` and the price index
the options are written on follows the price return.

CAPM IS TRUE IN THIS WORLD BY CONSTRUCTION.  Anything we measure failing in the
LEAPS book is caused by the option layer, not by the data-generating process.
That is the whole point of doing this on simulated rather than real data.

ACCOUNTING
----------
Both books start with the same equity E0 and are compared on two conventions:

  equal_exposure : both run the same delta exposure per unit of equity,
                   x = 1/(1 + margin_rate) per side -- the most the stock book
                   can support.  The LEAPS book pays less premium, so it holds
                   the surplus in cash earning r.  This measures the capital
                   efficiency benefit directly and holds risk constant.
  equal_capital  : both use all of E0.  The LEAPS book gets more exposure
                   because options are cheaper than funded stock.

Every period:
  stock book : P&L = sum(w_long * R_total) - sum(w_short * R_total)
                     - borrow * short_notional * dt
  LEAPS book : P&L = change in the mark of the option book
                     + r * cash * dt
                     - half-spread * premium on every trade

Delta, gamma, theta and the embedded financing at the risk-free rate all fall
out of the repricing, so nothing is double counted.

FRICTION KNOBS (defaults are the values measured on real data in notes 02/03)
----------------------------------------------------------------------------
  fspread_bps      dealer funding spread.  Implemented by PRICING options at
                   r + spread while cash earns r: the spread is therefore
                   amortised into the mark rather than charged separately.
                   61 bp measured from live chains.
  half_spread_bps  one-way cost as a fraction of premium. ~100 bp one-way.
  vrp              vol points added to the pricing vol vs realised vol.
  american         mark at American value (adds the early-exercise premium).
  borrow           stock-loan fee on the stock book's short notional.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

import sim_lib as sl


@dataclass
class Cfg:
    n_paths: int = 5000
    n_months: int = 240
    n_stocks: int = 30
    n_leg: int = 10
    seed: int = 20260902

    sig_mkt: float = 0.16
    sig_stock: float = 0.30
    beta_lo: float = 0.7
    beta_hi: float = 1.3
    erp: float = 0.055
    r: float = 0.042
    q: float = 0.015

    delta_tgt: float = 0.80
    opt_tenor: float = 2.0
    rebal_months: int = 3

    margin_rate: float = 0.50
    borrow: float = 0.0025
    rebate: float = -1.0          # rate earned on the stock book's posted capital
                                  # (short proceeds + margin).  -1 -> use r.

    half_spread_bps: float = 100.0
    fspread_bps: float = 61.0
    vrp: float = 0.0
    american: bool = True
    mode: str = "equal_exposure"     # equal_exposure | equal_capital

    strategy: str = "random"         # random | momentum | beta
    mom_lookback: int = 12
    mom_skip: int = 1

    dt: float = 1.0 / 12.0


def _price(S, K, tau, cfg, kind, ee, r_px, sig_px):
    # unheld legs carry K = 0 and n = 0; clip so log(S/K) stays finite (the
    # result is multiplied by n = 0 so it cannot affect the mark)
    K = np.maximum(K, 1e-8)
    p = sl.bs_call(S, K, tau, r_px, cfg.q, sig_px) if kind == "C" \
        else sl.bs_put(S, K, tau, r_px, cfg.q, sig_px)
    if cfg.american and ee is not None:
        p = p + (ee.call(S, K, tau) if kind == "C" else ee.put(S, K, tau))
    return p


def _select(cfg, rng, beta, ret_buf, buf_ix, P):
    """(long_ix, short_ix), each (P, n_leg)."""
    N, K = cfg.n_stocks, cfg.n_leg
    if cfg.strategy == "random" or ret_buf is None:
        perm = np.argsort(rng.random((P, N)), axis=1)
        return perm[:, :K], perm[:, N - K:]
    if cfg.strategy == "beta":
        score = np.broadcast_to(beta[None, :], (P, N)).copy()
    else:
        L = ret_buf.shape[2]
        order = [(buf_ix - 1 - k) % L for k in range(L)]
        stacked = ret_buf[:, :, order]
        score = stacked[:, :, cfg.mom_skip:].sum(2)
    perm = np.argsort(score, axis=1)
    return perm[:, N - K:], perm[:, :K]


def simulate(cfg: Cfg, ee: sl.EEGrid | None = None):
    rng = np.random.default_rng(cfg.seed)
    P, N, K, T = cfg.n_paths, cfg.n_stocks, cfg.n_leg, cfg.n_months
    dt, sdt = cfg.dt, np.sqrt(cfg.dt)

    beta = rng.uniform(cfg.beta_lo, cfg.beta_hi, N)
    var_eps = np.maximum(cfg.sig_stock ** 2 - (beta * cfg.sig_mkt) ** 2, 1e-4)
    sig_eps = np.sqrt(var_eps)
    mu_i = cfg.r + beta * cfg.erp
    mu_m = cfg.r + cfg.erp

    r_px = cfg.r + cfg.fspread_bps / 1e4
    sig_px = cfg.sig_stock + cfg.vrp
    hs = cfg.half_spread_bps / 1e4

    # exposure scale per side, as a fraction of equity
    x_stock = 1.0 / (1.0 + cfg.margin_rate)
    kc0 = sl.strike_for_delta(np.array([1.0]), cfg.opt_tenor, r_px, cfg.q, sig_px, cfg.delta_tgt, "C")[0]
    kp0 = sl.strike_for_delta(np.array([1.0]), cfg.opt_tenor, r_px, cfg.q, sig_px, cfg.delta_tgt, "P")[0]
    prem_c = sl.bs_call(np.array([1.0]), np.array([kc0]), np.array([cfg.opt_tenor]), r_px, cfg.q, sig_px)[0]
    prem_p = sl.bs_put(np.array([1.0]), np.array([kp0]), np.array([cfg.opt_tenor]), r_px, cfg.q, sig_px)[0]
    if cfg.american and ee is not None:
        prem_c += ee.call(np.array([1.0]), np.array([kc0]), np.array([cfg.opt_tenor]))[0]
        prem_p += ee.put(np.array([1.0]), np.array([kp0]), np.array([cfg.opt_tenor]))[0]
    # capital required per $1 long + $1 short of delta exposure
    prem_ratio = (prem_c + prem_p) / cfg.delta_tgt
    x_leaps = x_stock if cfg.mode == "equal_exposure" else 1.0 / prem_ratio

    # ---------------- state ----------------
    S = np.ones((P, N))
    eq_s = np.ones(P)
    cash = np.ones(P) * 1.0
    eq0 = 1.0

    Kc = np.zeros((P, N)); nc = np.zeros((P, N))
    Kp = np.zeros((P, N)); npu = np.zeros((P, N))
    tau = np.zeros(P)
    vintage = np.full(P, -10**9, dtype=int)

    Lbuf = cfg.mom_lookback + cfg.mom_skip
    ret_buf = np.zeros((P, N, Lbuf)); buf_ix = 0
    long_ix, short_ix = _select(cfg, rng, beta, None, 0, P)

    out = dict(
        ret_s=np.zeros((P, T)), ret_l=np.zeros((P, T)), ret_mkt=np.zeros((P, T)),
        net_delta=np.zeros((P, T)), gross_delta=np.zeros((P, T)),
        spread_cost=np.zeros((P, T)), cash_share=np.zeros((P, T)),
        pl_delta=np.zeros((P, T)), pl_theta=np.zeros((P, T)),
        pl_gamma=np.zeros((P, T)), pl_interest=np.zeros((P, T)),
        mkt_since_reset=np.zeros((P, T)),
        worthless=np.zeros(P), n_legs=np.zeros(P),
        prem_ratio=prem_ratio, x_stock=x_stock, x_leaps=x_leaps,
        beta=beta,
    )
    optval_prev = np.zeros(P)
    eq_l_prev = np.ones(P)
    mkt_idx = np.ones(P)
    mkt_at_vintage = np.ones(P)

    roll_every = max(1, int(round(cfg.opt_tenor / dt)))
    roll_every = min(cfg.rebal_months, roll_every)

    for t in range(T):
        zM = rng.standard_normal(P)
        zI = rng.standard_normal((P, N))
        Rm = mu_m * dt + cfg.sig_mkt * sdt * zM
        Ri = mu_i[None, :] * dt + (beta * cfg.sig_mkt * sdt)[None, :] * zM[:, None] \
            + (sig_eps * sdt)[None, :] * zI
        Rp = Ri - cfg.q * dt

        # ---------------- stock book ----------------
        # constant-fraction: x_stock of equity per side, rebalanced every period
        wl = np.zeros((P, N)); ws = np.zeros((P, N))
        wl[np.arange(P)[:, None], long_ix] = 1.0 / K
        ws[np.arange(P)[:, None], short_ix] = 1.0 / K
        r_long = (wl * Ri).sum(1)
        r_short = (ws * Ri).sum(1)
        reb = cfg.r if cfg.rebate < 0 else cfg.rebate
        ret_s = x_stock * (r_long - r_short - cfg.borrow * dt) + reb * dt
        out["ret_s"][:, t] = ret_s
        eq_s = eq_s * (1.0 + ret_s)

        # ---------------- LEAPS book: roll if due ----------------
        S_prev = S.copy()
        due = (t == 0) | ((t - vintage) >= roll_every) | (tau <= 1e-9)
        if due.any():
            j = np.where(due)[0]
            # legs that have expired settle at intrinsic, the rest are sold
            if np.any(j):
                jh = j
                tv = tau[jh][:, None]
                expd = tau[jh] <= 1e-9
                pay_c = np.maximum(S[jh] - Kc[jh], 0.0)
                pay_p = np.maximum(Kp[jh] - S[jh], 0.0)
                vc = np.where(expd[:, None], pay_c,
                              _price(S[jh], Kc[jh], np.maximum(tv, 1e-4), cfg, "C", ee, r_px, sig_px))
                vp = np.where(expd[:, None], pay_p,
                              _price(S[jh], Kp[jh], np.maximum(tv, 1e-4), cfg, "P", ee, r_px, sig_px))
                val = (nc[jh] * vc + npu[jh] * vp).sum(1)
                # only charge the exit half-spread on legs that are actually sold
                sold = np.where(expd[:, None], 0.0, 1.0)
                val_sold = (nc[jh] * vc * sold + npu[jh] * vp * sold).sum(1)
                cash[jh] += val - val_sold * hs
                out["spread_cost"][jh, t] += val_sold * hs
                out["worthless"][jh] += ((Kc[jh] > 0) & (S[jh] <= Kc[jh])).sum(1) \
                    + ((Kp[jh] > 0) & (S[jh] >= Kp[jh])).sum(1)
                out["n_legs"][jh] += (nc[jh] > 0).sum(1) + (npu[jh] > 0).sum(1)
                nc[jh] = 0.0; npu[jh] = 0.0; Kc[jh] = 0.0; Kp[jh] = 0.0

            # equity available to size the new vintage (constant-fraction)
            tv = np.maximum(tau[j], 1e-4)[:, None]
            eq_now = cash[j] + (nc[j] * _price(S[j], Kc[j], tv, cfg, "C", ee, r_px, sig_px)
                                + npu[j] * _price(S[j], Kp[j], tv, cfg, "P", ee, r_px, sig_px)).sum(1)
            eq_now = np.maximum(eq_now, 1e-6)

            # new vintage
            tau[j] = cfg.opt_tenor
            vintage[j] = t
            mkt_at_vintage[j] = mkt_idx[j]
            kcv = sl.strike_for_delta(S[j], cfg.opt_tenor, r_px, cfg.q, sig_px, cfg.delta_tgt, "C")
            kpv = sl.strike_for_delta(S[j], cfg.opt_tenor, r_px, cfg.q, sig_px, cfg.delta_tgt, "P")
            ncv = np.zeros_like(kcv); npv = np.zeros_like(kpv)
            rows = np.arange(len(j))[:, None]
            expo = (x_leaps * eq_now / K)[:, None]
            ncv[rows, long_ix[j]] = expo / (cfg.delta_tgt * S[j][rows, long_ix[j]])
            npv[rows, short_ix[j]] = expo / (cfg.delta_tgt * S[j][rows, short_ix[j]])
            Kc[j] = kcv; Kp[j] = kpv; nc[j] = ncv; npu[j] = npv
            prem = (ncv * _price(S[j], kcv, cfg.opt_tenor, cfg, "C", ee, r_px, sig_px)
                    + npv * _price(S[j], kpv, cfg.opt_tenor, cfg, "P", ee, r_px, sig_px)).sum(1)
            cash[j] -= prem * (1.0 + hs)
            out["spread_cost"][j, t] += prem * hs / np.maximum(np.abs(eq_l_prev[j]), 1e-9)

        # advance spot and tenor
        S = S * (1.0 + Rp)
        tau = np.maximum(tau - dt, 0.0)

        # exposure diagnostics measured at the START of the month
        tv = np.maximum(tau + dt, 1e-4)[:, None]
        dc = sl.bs_delta_call(S_prev, Kc, tv, r_px, cfg.q, sig_px)
        dp = sl.bs_delta_put(S_prev, Kp, tv, r_px, cfg.q, sig_px)
        out["net_delta"][:, t] = (nc * dc * S_prev + npu * dp * S_prev).sum(1)
        out["gross_delta"][:, t] = (nc * dc * S_prev - npu * dp * S_prev).sum(1)

        # mark at the start of the month (post-trade) so the decomposition is clean
        optval_start = (nc * _price(S_prev, Kc, tv, cfg, "C", ee, r_px, sig_px)
                        + npu * _price(S_prev, Kp, tv, cfg, "P", ee, r_px, sig_px)).sum(1)

        # mark, accrue interest
        tv2 = np.maximum(tau, 1e-4)[:, None]
        optval = (nc * _price(S, Kc, tv2, cfg, "C", ee, r_px, sig_px)
                  + npu * _price(S, Kp, tv2, cfg, "P", ee, r_px, sig_px)).sum(1)

        # ---- attribution of the month's option P&L ----
        dS = S - S_prev
        d_pl = (nc * dc * dS + npu * dp * dS).sum(1)
        th_c = sl.bs_theta_call(S_prev, Kc, tv, r_px, cfg.q, sig_px)
        th_p = sl.bs_theta_put(S_prev, Kp, tv, r_px, cfg.q, sig_px)
        t_pl = (nc * th_c + npu * th_p).sum(1) * dt
        dv = optval - optval_start
        # normalise by start-of-month equity so the components are additive
        # returns and sum to the book's return (excluding trade spread costs)
        sc = 1.0 / np.maximum(np.abs(eq_l_prev), 1e-9)
        out["pl_delta"][:, t] = d_pl * sc
        out["pl_theta"][:, t] = t_pl * sc
        out["pl_gamma"][:, t] = (dv - d_pl - t_pl) * sc
        out["pl_interest"][:, t] = np.maximum(cash, 0.0) * cfg.r * dt * sc

        mkt_idx = mkt_idx * (1.0 + Rm)
        out["mkt_since_reset"][:, t] = mkt_idx / mkt_at_vintage - 1.0
        cash = cash * (1.0 + cfg.r * dt)
        eq_l = cash + optval
        out["ret_l"][:, t] = eq_l / eq_l_prev - 1.0
        out["cash_share"][:, t] = np.maximum(cash, 0.0) / np.maximum(eq_l, 1e-9)
        eq_l_prev = eq_l.copy()
        optval_prev = optval

        out["ret_mkt"][:, t] = Rm

        # buffer + next selection
        ret_buf[:, :, buf_ix] = Ri
        buf_ix = (buf_ix + 1) % Lbuf
        if cfg.strategy != "random":
            long_ix, short_ix = _select(cfg, rng, beta, ret_buf, buf_ix, P)

    out["eq_s"] = eq_s
    out["eq_l"] = eq_l
    out["cfg"] = cfg
    return out


def summarise(o, name=""):
    """Headline stats from a simulate() output."""
    rs, rl = o["ret_s"], o["ret_l"]
    yrs = o["cfg"].n_months / 12.0

    def stats(r):
        eq = np.prod(1.0 + r, axis=1)
        cagr = eq ** (1.0 / yrs) - 1.0
        vol = r.std(axis=1) * np.sqrt(12.0)
        mu = r.mean(axis=1) * 12.0
        sharpe = mu / np.maximum(vol, 1e-9)
        # max drawdown
        path = np.cumprod(1.0 + r, axis=1)
        peak = np.maximum.accumulate(path, axis=1)
        dd = (path / peak - 1.0).min(axis=1)
        return dict(cagr=cagr, vol=vol, mu=mu, sharpe=sharpe, maxdd=dd, eq=eq)

    ss, sl_ = stats(rs), stats(rl)
    return dict(name=name,
                cagr_s=ss["cagr"].mean(), cagr_l=sl_["cagr"].mean(),
                vol_s=ss["vol"].mean(), vol_l=sl_["vol"].mean(),
                sharpe_s=ss["sharpe"].mean(), sharpe_l=sl_["sharpe"].mean(),
                maxdd_s=ss["maxdd"].mean(), maxdd_l=sl_["maxdd"].mean(),
                eq_s_med=np.median(ss["eq"]), eq_l_med=np.median(sl_["eq"]),
                eq_s_p05=np.percentile(ss["eq"], 5), eq_l_p05=np.percentile(sl_["eq"], 5),
                eq_s_p95=np.percentile(ss["eq"], 95), eq_l_p95=np.percentile(sl_["eq"], 95),
                _ss=ss, _sl=sl_)
