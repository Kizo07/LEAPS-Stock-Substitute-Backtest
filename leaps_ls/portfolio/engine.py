"""Daily accounting backtest engine (PLAN §5.6–5.10), variant-agnostic.

Conventions
-----------
- Trades execute at the SAME-DAY close: stock at the close price, options at the
  model mid; every trade (buy or sell) pays the calibrated half-spread plus
  commissions. Costs are booked to exact cumulative cash ledgers.
- Options are marked to the model daily. Each position's projected dividend
  schedule (ex-ante) is refreshed at every monthly rebalance AND on each actual
  ex-date of the underlying — the day new dividend information enters the
  projection (closest feasible approximation to PLAN §5.1's "announced"
  dividends with free data; a cut like BAC's 2009 $0.32->$0.01 is absorbed the
  day it goes ex rather than up to a month later). Schedule-change revaluation
  is booked as mark-to-market P&L. Between refreshes, marking re-escrows the
  remaining projected ex-dates at the current spot and rate. As a numerical
  guard S* is floored at 1% of S so log terms stay finite when projected
  dividends temporarily exceed the price.
- Actual ex-date calendar and announced amounts are treated as publicly known
  before the ex-date (as in reality): stock legs receive/pay actual dividends in
  cash; option legs apply the PLAN §5.1 exercise rules around them.
- Early-exercise handling: calls are closed the day before an ex-date and
  re-established the next day when the dividend exceeds remaining time value
  (assignment of short calls, including inside synthetic lanes, mirrors this);
  puts are closed-and-reopened on the FIRST day of each parity-bound (P < K -
  S*) violation streak — costed per violation episode, not per day, to avoid
  degenerate daily churn in persistent high-rate regimes (flagged deviation
  from a literal per-day reading). Per PLAN §5.1 the parity-bound rule is
  scoped to the standalone deep-ITM put lane; inside synthetics the plan
  specifies assignment handling only for the short call, so put legs of
  synthetic groups are exempt (their European mispricing vs American is the
  model's documented base-case understatement, not an exercise trigger).
- Bankruptcy guard (deviation, flagged): the plan's fully-funded world has no
  margin calls, so NAV <= 0 is outside its conventions — yet a monthly-sized
  short book can be carried past zero by a violent intramonth rally (position
  notionals reset only at rebalances). When mark-to-market NAV <= 0 we liquidate
  the whole book at that day's close (paying costs), halt trading permanently,
  and floor NAV at zero; any residual debit is written off into a dedicated
  ledger so the reconciliation identity stays exact. Reported as ruin.
- Fully funded: option premiums are paid in cash, no margin loan; negative cash
  is charged the debit rate. Borrow on short stock accrues per trading day at
  bps/252 of short market value (previous close). Cash financing accrues over
  calendar days at credit/debit rates (frictions.financing).
- Fractional contracts and shares are allowed (standard backtest abstraction).
- Dollar-denominated frictions ($0.05/share spread floor, $0.65/contract and
  $0.005/share commissions) are quoted in as-traded share terms and divided by
  the cumulative future split factor before application, since prices are
  split-adjusted to today's share basis; percentage frictions (spread tiers,
  stock bps, borrow, financing) are scale-free.
- Fixed-dollar frictions need a capital scale: sizing runs on RunConfig.capital
  dollars (default config.CAPITAL_BASE); NAV is reported in those dollars and
  normalized downstream. Commission impact scales inversely with capital.

Reconciliation (exact cash bookkeeping): with mtm_t the mark-to-market of
positions held from t-1 to t,  NAV_t = NAV_{t-1} + mtm_t + financing_t
+ dividends_t - borrow_t - spread_t - commissions_t.  The engine asserts this
daily when RunConfig.check_recon is set (smoke tests).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from ..frictions import financing, spreads as spreads_mod
from ..instruments import rolls as rolls_mod, selector
from ..pricing import black_scholes as bs, dividends as dvd, exercise, vol


@dataclass
class RunConfig:
    """Per-run knobs (sensitivities vary these one at a time, PLAN §7)."""

    sizing: str = config.SIZING
    spread_mult: float = 1.0
    borrow_bps: float | dict = config.BORROW_BPS_BASE  # scalar or {ticker: bps}
    commission_contract: float = config.COMMISSION_PER_CONTRACT
    commission_share: float = config.COMMISSION_STOCK_PER_SHARE
    stock_spread_bps: float = config.STOCK_SPREAD_BPS
    roll_dte: int = config.ROLL_DTE_THRESHOLD
    delta_band_roll: bool = False
    capital: float = config.CAPITAL_BASE
    check_recon: bool = False
    # one-at-a-time sensitivity overrides (None -> config value)
    delta_target: float | None = None      # overrides config.DELTA_TARGET at selection
    tenor_target_days: int | None = None   # overrides config.TENOR_TARGET_DAYS at selection
    iv_mult_adj: float = 1.0               # multiplies the IV proxy (PLAN §7 iv_mult_adj)


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    events: dict
    rcfg: RunConfig


class Engine:
    """Plan-driven daily accounting engine. One option group per ticker at most."""

    def __init__(self, plan: pd.DataFrame, mode_long: str, mode_short: str, md, rcfg: RunConfig):
        config.validate()
        if mode_long not in ("stock", "call", "synth_long") or \
                mode_short not in ("stock", "put", "synth_short"):
            raise ValueError(f"invalid modes: {mode_long}/{mode_short}")
        missing = [c for c in plan.columns if c not in md.j]
        if missing:
            raise ValueError(f"plan tickers missing from MarketData: {missing}")
        self.plan = plan.sort_index()
        self.mode_long = mode_long
        self.mode_short = mode_short
        self.md = md
        self.rcfg = rcfg

        self.cash = float(rcfg.capital)
        self.shares: dict[str, float] = {}
        self.opt: dict[str, dict] = {}
        self.pending_reopen: list[dict] = []
        self._parity_active: dict[str, bool] = {}

        self.ledgers = {"spread": 0.0, "comm": 0.0, "borrow": 0.0,
                        "financing": 0.0, "dividends": 0.0, "writeoff": 0.0}
        self.events = {"rolls": 0, "call_exercise_reopens": 0, "put_parity_reopens": 0,
                       "expiry_settlements": 0, "stock_trades": 0, "option_trades": 0,
                       "bankruptcies": 0, "bankruptcy_date": None}
        self.trade_rows: list[dict] = []
        self.daily_rows: list[dict] = []
        self._bankrupt = False

        self._prev_i: int | None = None
        self._prev_nav: float | None = None
        self._mtm_today = 0.0
        self._fin_today = 0.0
        self._div_today = 0.0
        self._borrow_today = 0.0
        self._spread_today = 0.0
        self._comm_today = 0.0
        self._writeoff_today = 0.0

    # ------------------------------------------------------------------ helpers
    def _borrow_bps(self, ticker: str) -> float:
        b = self.rcfg.borrow_bps
        return float(b[ticker]) if isinstance(b, dict) else float(b)

    def _log_trade(self, i: int, ticker: str, action: str, kind: str, K, expiry_ord,
                   qty: float, px_per_share: float, spread_cost: float, comm: float) -> None:
        self.trade_rows.append({
            "date": self.md.days[i], "ticker": ticker, "action": action, "kind": kind,
            "K": K, "expiry": pd.Timestamp.fromordinal(int(expiry_ord)) if expiry_ord else None,
            "qty": qty, "px_per_share": px_per_share,
            "spread_cost": spread_cost, "comm_cost": comm,
        })

    def _book_costs(self, spread_cost: float, comm: float) -> None:
        self.ledgers["spread"] += spread_cost
        self.ledgers["comm"] += comm
        self._spread_today += spread_cost
        self._comm_today += comm

    def _leg_infos_from_spec(self, spec: dict) -> list[dict]:
        return [{"cp": l["cp"], "sign": l["sign"], "px": l["premium"], "delta": l["delta"], "iv": l["iv"]}
                for l in spec["legs"]]

    def _option_trade_costs(self, ticker: str, i: int, leg_infos: list[dict],
                            n_contracts: float) -> tuple[float, float]:
        """Half-spread + commission for trading ``n_contracts`` of a group.

        Dollar frictions (per-share floor, per-contract commission) are quoted in
        as-traded share terms, so they are divided by the cumulative split factor:
        on split-adjusted historical prices an unscaled $0.05 floor or $0.65
        commission would be overstated by up to the full split ratio.
        """
        sf = float(self.md.split_arr[i, self.md.j[ticker]])
        sf = sf if sf > 0.0 else 1.0
        floor = self.md.spread_floor / sf
        sp = 0.0
        for li in leg_infos:
            sp += spreads_mod.half_spread(
                li["px"], li["delta"], self.md.spread_tiers, floor,
                self.rcfg.spread_mult)
        spread_cost = abs(n_contracts) * 100.0 * sp
        comm = abs(n_contracts) * len(leg_infos) * (self.rcfg.commission_contract / sf)
        return spread_cost, comm

    # ------------------------------------------------------------------ marking
    def _mark_group(self, pos: dict, t_ord: int, i: int):
        """Per-share group mid, net delta, S*, r and per-leg marks at day ``i``."""
        tk = pos["ticker"]
        S = float(self.md.close_arr[i, self.md.j[tk]])
        t_days = int(pos["expiry_ord"]) - t_ord
        T = max(t_days, 0) / 365.25
        r = self.md.rate_ord(t_ord, max(T, 1e-6)) if t_days > 0 else 0.0
        pv = 0.0
        sd = pos["sched_days"]
        if t_days > 0 and sd.size:
            rem = (sd > t_ord) & (sd <= int(pos["expiry_ord"]))
            if rem.any():
                t_div = (sd[rem] - t_ord) / 365.25
                pv = float(np.sum(pos["sched_amts"][rem] * np.exp(-r * t_div)))
        S_star = max(S - pv, 0.01 * S)  # floor keeps log terms finite (stale projections)
        iv_atm = self.md.iv_atm(tk, i) * self.rcfg.iv_mult_adj
        slope = self.md.slope(tk)
        value, delta_net, leg_infos = 0.0, 0.0, []
        for leg in pos["legs"]:
            K = float(pos["K"])
            iv = float(vol.apply_skew(iv_atm, K, S_star, slope))
            px = float(bs.price(S_star, K, T, r, iv, leg["cp"]))
            dl = float(bs.delta(S_star, K, T, r, iv, leg["cp"]))
            value += leg["sign"] * px
            delta_net += leg["sign"] * dl
            leg_infos.append({"cp": leg["cp"], "sign": leg["sign"], "px": px, "delta": dl, "iv": iv})
        return value, delta_net, S_star, r, leg_infos

    def _mark_all(self, i: int, t_ord: int) -> None:
        """Mark every open option group at day ``i``; accumulate overnight mtm P&L."""
        mtm = 0.0
        for pos in self.opt.values():
            value, delta_net, S_star, r, leg_infos = self._mark_group(pos, t_ord, i)
            if pos.get("last_value") is not None:
                mtm += pos["contracts"] * 100.0 * (value - pos["last_value"])
            pos["last_value"] = value
            pos["last_delta"] = delta_net
            pos["last_S_star"] = S_star
            pos["last_r"] = r
            pos["last_legs"] = leg_infos
        if self._prev_i is not None:
            for tk, sh in self.shares.items():
                j = self.md.j[tk]
                mtm += sh * (float(self.md.close_arr[i, j]) - float(self.md.close_arr[self._prev_i, j]))
        self._mtm_today = mtm

    def _nav(self, i: int) -> float:
        v = self.cash
        for tk, sh in self.shares.items():
            v += sh * float(self.md.close_arr[i, self.md.j[tk]])
        for pos in self.opt.values():
            v += pos["contracts"] * 100.0 * pos["last_value"]
        return v

    # ------------------------------------------------------------------ option trades
    def _open_group(self, ticker: str, kind: str, K: float, expiry_ord: int, contracts: float,
                    i: int, action: str, sched, leg_infos: list[dict] | None = None,
                    legs_value: float | None = None) -> dict:
        t_ord = int(self.md.day_ordinals[i])
        pos = {"ticker": ticker, "kind": kind, "K": float(K), "expiry_ord": int(expiry_ord),
               "contracts": float(contracts), "legs": [dict(l) for l in selector.LEG_SIGNS[kind]],
               "sched_days": sched[0], "sched_amts": sched[1], "entry_date": self.md.days[i]}
        if leg_infos is None or legs_value is None:
            legs_value, _, _, _, leg_infos = self._mark_group(pos, t_ord, i)
        spread_cost, comm = self._option_trade_costs(ticker, i, leg_infos, contracts)
        self.cash -= contracts * 100.0 * legs_value + spread_cost + comm
        self._book_costs(spread_cost, comm)
        pos["last_value"] = legs_value
        pos["last_delta"] = float(sum(li["sign"] * li["delta"] for li in leg_infos))
        pos["last_legs"] = leg_infos
        self.opt[ticker] = pos
        if action != "reopen":
            self._parity_active[ticker] = False  # a fresh contract starts a new violation streak
        self.events["option_trades"] += 1
        self._log_trade(i, ticker, action, kind, K, expiry_ord, contracts,
                        legs_value, spread_cost, comm)
        return pos

    def _close_group(self, ticker: str, i: int, reason: str, settle: bool = False) -> dict:
        pos = self.opt.pop(ticker)
        n = pos["contracts"]
        value = pos["last_value"]
        if settle:  # expiry settlement: no spread/commission
            spread_cost = comm = 0.0
        else:
            spread_cost, comm = self._option_trade_costs(ticker, i, pos["last_legs"], n)
        self.cash += n * 100.0 * value - spread_cost - comm
        self._book_costs(spread_cost, comm)
        self.events["option_trades"] += 1
        self._log_trade(i, ticker, f"close_{reason}", pos["kind"], pos["K"],
                        pos["expiry_ord"], -n, value, spread_cost, comm)
        return pos

    def _adjust_group(self, pos: dict, target_contracts: float, i: int) -> None:
        d_n = target_contracts - pos["contracts"]
        if abs(d_n) < 1e-12:
            return
        leg_infos = pos["last_legs"]
        value = pos["last_value"]
        spread_cost, comm = self._option_trade_costs(pos["ticker"], i, leg_infos, d_n)
        self.cash -= d_n * 100.0 * value + spread_cost + comm
        self._book_costs(spread_cost, comm)
        pos["contracts"] = float(target_contracts)
        self.events["option_trades"] += 1
        self._log_trade(i, pos["ticker"], "adjust", pos["kind"], pos["K"],
                        pos["expiry_ord"], d_n, value, spread_cost, comm)

    # ------------------------------------------------------------------ events
    def _stock_dividends(self, t_ord: int) -> None:
        for tk, sh in self.shares.items():
            amt = self.md.ex_dividend_ord(tk, t_ord)
            if amt and sh:
                d = sh * amt
                self.cash += d
                self.ledgers["dividends"] += d
                self._div_today += d

    def _exercise_checks(self, i: int, t_ord: int) -> None:
        """Call exercise/assignment rule on the day before an (announced) ex-date."""
        tom_ord = int(self.md.day_ordinals[i + 1]) if i + 1 < len(self.md.days) else None
        if tom_ord is None:
            return
        for tk, pos in list(self.opt.items()):
            amt = self.md.ex_dividend_ord(tk, tom_ord)
            if not amt or not any(l["cp"] == "C" for l in pos["legs"]):
                continue
            T = max(int(pos["expiry_ord"]) - tom_ord, 0) / 365.25
            fire = False
            for li in pos["last_legs"]:
                if li["cp"] != "C":
                    continue
                if exercise.should_exercise_call(amt, pos["last_S_star"], pos["K"], T,
                                                 pos["last_r"], li["iv"]):
                    fire = True
            if fire:
                old = self._close_group(tk, i, "exercise_reopen")
                self.events["call_exercise_reopens"] += 1
                self.pending_reopen.append(old)

    def _parity_checks(self, i: int, t_ord: int) -> None:
        """Put parity-bound rule: close-and-reopen on the first day of a violation streak.

        Scoped to standalone put positions (PLAN §5.1); synthetic put legs are exempt.
        """
        for tk, pos in list(self.opt.items()):
            if pos["kind"] != "put":
                continue
            violated = any(
                li["cp"] == "P" and exercise.put_parity_violation(li["px"], pos["K"], pos["last_S_star"])
                for li in pos["last_legs"]
            )
            was = self._parity_active.get(tk, False)
            if violated and not was:
                old = self._close_group(tk, i, "parity_reopen")
                self.events["put_parity_reopens"] += 1
                self.pending_reopen.append(old)
            self._parity_active[tk] = violated

    def _settle_expiries(self, i: int, t_ord: int) -> None:
        """Safety net: settle any group at/after expiry at intrinsic (no trade costs).

        Monthly rolls at DTE < ROLL_DTE_THRESHOLD make this rare in practice.
        """
        for tk, pos in list(self.opt.items()):
            if int(pos["expiry_ord"]) <= t_ord:
                self._close_group(tk, i, "expiry_settle", settle=True)
                self.events["expiry_settlements"] += 1

    def _liquidate_bankrupt(self, i: int, t_ord: int) -> None:
        """Ruin: close the whole book at today's close and halt trading permanently.

        Any residual debit balance is written off into the ``writeoff`` ledger so
        the daily reconciliation identity stays exact; NAV is floored at zero.
        """
        for tk in list(self.opt):
            self._close_group(tk, i, "bankruptcy")
        for tk in list(self.shares):
            self._trade_stock_to(tk, 0.0, i)
        self.pending_reopen.clear()
        w = -min(self.cash, 0.0)  # residual debit after liquidating
        if w > 0.0:
            self.cash += w
            self.ledgers["writeoff"] += w
            self._writeoff_today = w
        self._bankrupt = True
        self.events["bankruptcies"] += 1
        self.events["bankruptcy_date"] = str(self.md.days[i].date())

    # ------------------------------------------------------------------ rebalance
    def _trade_stock_to(self, ticker: str, target_shares: float, i: int) -> None:
        cur = self.shares.get(ticker, 0.0)
        d_sh = target_shares - cur
        if abs(d_sh) < 1e-10:
            return
        S = float(self.md.close_arr[i, self.md.j[ticker]])
        notional = abs(d_sh) * S
        sf = float(self.md.split_arr[i, self.md.j[ticker]])
        sf = sf if sf > 0.0 else 1.0
        spread_cost = notional * self.rcfg.stock_spread_bps / 1e4
        comm = abs(d_sh) * (self.rcfg.commission_share / sf)  # as-traded share terms
        self.cash -= d_sh * S + spread_cost + comm
        self._book_costs(spread_cost, comm)
        if abs(target_shares) < 1e-10:
            self.shares.pop(ticker, None)
        else:
            self.shares[ticker] = target_shares
        self.events["stock_trades"] += 1
        self._log_trade(i, ticker, "stock_adjust", "stock", None, None,
                        d_sh, S, spread_cost, comm)

    def _size_contracts(self, target_notional: float, delta_net: float, S: float) -> float:
        if self.rcfg.sizing == "share_equivalent" or abs(delta_net) < 1e-6:
            return abs(target_notional) / (100.0 * S)  # degenerate delta -> share-equivalent
        return abs(target_notional) / (100.0 * abs(delta_net) * S)  # delta_equivalent

    def _rebalance(self, targets: pd.Series, i: int, t_ord: int, nav: float) -> None:
        # A rebalance fully re-establishes exposure from targets, so any pending
        # exercise/parity reopen is superseded (avoids reopening over a position
        # the rebalance just opened, or recreating exposure the plan no longer wants).
        self.pending_reopen.clear()
        tickers = set(targets.index) | set(self.opt) | set(self.shares)
        for tk in tickers:
            w = float(targets.get(tk, 0.0))
            if not np.isfinite(w):
                w = 0.0
            S = float(self.md.close_arr[i, self.md.j[tk]])
            if not np.isfinite(S):
                continue  # no price (pre-IPO): cannot hold or open
            mode = self.mode_long if w > 0 else (self.mode_short if w < 0 else None)
            target_notional = w * nav
            if mode == "stock" or mode is None:
                if tk in self.opt:
                    self._close_group(tk, i, "rebalance")
                tgt_sh = target_notional / S if mode == "stock" else 0.0
                self._trade_stock_to(tk, tgt_sh, i)
                continue
            # option-implemented side
            if tk in self.shares:
                self._trade_stock_to(tk, 0.0, i)
            pos = self.opt.get(tk)
            if pos is not None:
                band = config.ROLL_DELTA_BAND if self.rcfg.delta_band_roll else None
                kind_switch = pos["kind"] != mode
                if kind_switch or rolls_mod.needs_roll(
                        pos, t_ord, self.rcfg.roll_dte, band, abs(pos.get("last_delta") or 1.0)):
                    if not kind_switch:
                        self.events["rolls"] += 1
                    self._close_group(tk, i, "roll" if not kind_switch else "switch")
                    pos = None
            if pos is None:
                spec = selector.select_contract(
                    self.md.days[i], tk, mode, self.md,
                    delta_target=self.rcfg.delta_target,
                    tenor_target_days=self.rcfg.tenor_target_days,
                    iv_mult_adj=self.rcfg.iv_mult_adj)
                n = self._size_contracts(target_notional, spec["delta"], S)
                self._open_group(tk, mode, spec["K"], spec["expiry_ord"], n, i, "open",
                                 (spec["sched_days"], spec["sched_amts"]),
                                 leg_infos=self._leg_infos_from_spec(spec),
                                 legs_value=spec["premium"])
            else:
                self._refresh_schedule(pos, i, t_ord)
                n = self._size_contracts(target_notional, pos["last_delta"], S)
                self._adjust_group(pos, n, i)

    def _refresh_schedule(self, pos: dict, i: int, t_ord: int) -> None:
        """Re-project the position's dividend schedule (ex-ante, data <= today) and
        re-mark; the revaluation from the schedule change is booked into today's
        mark-to-market P&L so the reconciliation identity stays exact."""
        horizon = max(int(pos["expiry_ord"]) - t_ord, 0) / 365.25 + 0.5
        sched = dvd.project_dividends(self.md.hist(pos["ticker"]), self.md.days[i],
                                      horizon_years=horizon)
        if len(sched):
            pos["sched_days"] = np.array([d.toordinal() for d in sched["ex_date"]],
                                         dtype=np.int64)
            pos["sched_amts"] = sched["amount"].to_numpy(dtype=float)
        else:
            pos["sched_days"] = np.empty(0, dtype=np.int64)
            pos["sched_amts"] = np.empty(0, dtype=float)
        prev_v = pos["last_value"]
        (pos["last_value"], pos["last_delta"], pos["last_S_star"], pos["last_r"],
         pos["last_legs"]) = self._mark_group(pos, t_ord, i)
        self._mtm_today += pos["contracts"] * 100.0 * (pos["last_value"] - prev_v)

    # ------------------------------------------------------------------ main loop
    def run(self, end=None) -> BacktestResult:
        days = self.md.days
        start_i = int(days.get_loc(self.plan.index[0]))
        if end is None:
            end_i = len(days) - 1
        else:
            end_i = int(days.searchsorted(pd.Timestamp(end), side="right")) - 1
            if end_i < start_i:
                raise ValueError(f"end {pd.Timestamp(end).date()} is before the plan start")
        plan_dates = set(self.plan.index)
        for i in range(start_i, end_i + 1):
            t = days[i]
            t_ord = int(self.md.day_ordinals[i])
            self._fin_today = self._div_today = self._borrow_today = 0.0
            self._spread_today = self._comm_today = self._writeoff_today = 0.0
            # (1) financing on overnight cash (calendar days)
            if self._prev_i is not None:
                gap = t_ord - int(self.md.day_ordinals[self._prev_i])
                r3m = float(self.md.short_rate_arr[self._prev_i])
                if self.cash >= 0.0:
                    fin = self.cash * financing.cash_credit_rate(r3m) * gap / 365.0
                else:
                    fin = self.cash * financing.cash_debit_rate(r3m) * gap / 365.0
                self.cash += fin
                self.ledgers["financing"] += fin
                self._fin_today = fin
                # (2) borrow on short stock value (previous close), per trading day
                cost = 0.0
                for tk, sh in self.shares.items():
                    if sh < 0.0:
                        v = -sh * float(self.md.close_arr[self._prev_i, self.md.j[tk]])
                        cost += v * (self._borrow_bps(tk) / 1e4) / 252.0
                if cost > 0.0:
                    self.cash -= cost
                    self.ledgers["borrow"] += cost
                    self._borrow_today = cost
            # (3) dividends, reopens from yesterday's event closes
            self._stock_dividends(t_ord)
            for old in self.pending_reopen:
                self._open_group(old["ticker"], old["kind"], old["K"], old["expiry_ord"],
                                 old["contracts"], i, "reopen",
                                 (old["sched_days"], old["sched_amts"]))
            self.pending_reopen.clear()
            # (4) mark everything at today's close
            self._mark_all(i, t_ord)
            # (4b) ruin check: liquidate and halt if the book is carried past zero
            if not self._bankrupt and self._nav(i) <= 0.0:
                self._liquidate_bankrupt(i, t_ord)
            # (5) event closes (exercise/parity/expiry), priced off today's marks
            if not self._bankrupt:
                # ex-date-driven schedule refresh: new dividend information enters the
                # projection the day it goes ex (approximates announced dividends)
                for tk, pos in list(self.opt.items()):
                    if self.md.ex_dividend_ord(tk, t_ord):
                        self._refresh_schedule(pos, i, t_ord)
                self._exercise_checks(i, t_ord)
                self._parity_checks(i, t_ord)
                self._settle_expiries(i, t_ord)
            # (6) rebalance at today's close
            if not self._bankrupt and t in plan_dates:
                nav_now = self._nav(i)
                if nav_now > 0.0:  # a dead book stops trading; NAV floored by positions
                    self._rebalance(self.plan.loc[t], i, t_ord, nav_now)
            # (7) books + reconciliation
            nav = self._nav(i)
            recon = np.nan
            if self._prev_nav is not None:
                expect = (self._prev_nav + self._mtm_today + self._fin_today + self._div_today
                          - self._borrow_today - self._spread_today - self._comm_today
                          + self._writeoff_today)
                recon = nav - expect
            stock_val = sum(sh * float(self.md.close_arr[i, self.md.j[tk]])
                            for tk, sh in self.shares.items())
            opt_val = sum(p["contracts"] * 100.0 * p["last_value"] for p in self.opt.values())
            gross = 0.0
            net = 0.0
            for tk, sh in self.shares.items():
                v = sh * float(self.md.close_arr[i, self.md.j[tk]])
                net += v
                gross += abs(v)
            for p in self.opt.values():
                v = p["contracts"] * 100.0 * p["last_delta"] * float(
                    self.md.close_arr[i, self.md.j[p["ticker"]]])
                net += v
                gross += abs(v)
            self.daily_rows.append({
                "date": t, "nav": nav, "cash": self.cash,
                "stock_value": stock_val, "option_value": opt_val,
                "net_delta_notional": net, "gross_delta_notional": gross,
                "spread_cum": self.ledgers["spread"], "comm_cum": self.ledgers["comm"],
                "borrow_cum": self.ledgers["borrow"], "financing_cum": self.ledgers["financing"],
                "dividends_cum": self.ledgers["dividends"],
                "writeoff_cum": self.ledgers["writeoff"],
                "n_stock": len(self.shares), "n_option": len(self.opt),
                "recon_err": recon,
            })
            self._prev_i = i
            self._prev_nav = nav
        daily = pd.DataFrame(self.daily_rows).set_index("date")
        trades = pd.DataFrame(self.trade_rows)
        if self.rcfg.check_recon:
            err = float(np.nanmax(np.abs(daily["recon_err"].to_numpy())))
            assert err < 1e-4, f"NAV reconciliation failed: max |err| = {err}"
        return BacktestResult(daily=daily, trades=trades, events=dict(self.events), rcfg=self.rcfg)


def run_backtest(plan: pd.DataFrame, mode_long: str, mode_short: str, md,
                 rcfg: RunConfig | None = None, end=None) -> BacktestResult:
    """Run the engine over ``plan`` (weights indexed by rebalance date)."""
    eng = Engine(plan, mode_long, mode_short, md, rcfg or RunConfig())
    return eng.run(end=end)
