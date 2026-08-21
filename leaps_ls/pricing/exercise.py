"""Early-exercise / assignment decision rules for American equity LEAPS (PLAN §5.1).

Calls: before each ex-date, exercise-and-replace (modeled as close the option the
day before ex-date and re-establish the next day) when the projected dividend
exceeds the call's remaining time value. Puts: track violations of the American
parity bound P < K - S*; hits are flagged and costed as exercise-reopen.
Short options mirror the same rules (short call assigned when time value < dividend).
"""
from __future__ import annotations

from . import black_scholes as bs


def call_time_value(S: float, K: float, T_rem: float, r: float, sigma: float) -> float:
    """Call price minus intrinsic, evaluated just before the ex-date.

    ``T_rem`` is the time from the ex-date to expiry and ``S`` the cum-dividend
    spot (consistent with the escrowed-spot convention used by the engine).
    """
    return float(bs.call_price(S, K, T_rem, r, sigma)) - max(S - K, 0.0)


def should_exercise_call(
    div_amount: float, S: float, K: float, T_rem: float, r: float, sigma: float
) -> bool:
    """True when the projected dividend before the ex-date exceeds remaining time value."""
    if div_amount <= 0.0 or T_rem <= 0.0:
        return False
    return div_amount > call_time_value(S, K, T_rem, r, sigma)


def should_assign_short_call(
    div_amount: float, S: float, K: float, T_rem: float, r: float, sigma: float
) -> bool:
    """Assignment of a short call mirrors the long-call exercise rule."""
    return should_exercise_call(div_amount, S, K, T_rem, r, sigma)


def put_parity_violation(P: float, K: float, S_star: float) -> bool:
    """True when the American put parity bound P >= K - S* is violated."""
    return P < K - S_star
