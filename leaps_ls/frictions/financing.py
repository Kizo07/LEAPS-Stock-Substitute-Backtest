"""Cash financing rates (PLAN §5.7): credit on positive cash, debit on negative."""
from __future__ import annotations

from .. import config


def cash_credit_rate(tbill_3m: float) -> float:
    """Rate earned on positive cash: 3M T-bill - 25bp, floored at zero.

    (Floor deviates slightly from the literal plan text: with the bill rate near
    zero the un-floored formula would charge interest on positive cash balances,
    which no broker does.)
    """
    return max(float(tbill_3m) - config.CASH_CREDIT_SPREAD_BPS / 1e4, 0.0)


def cash_debit_rate(tbill_3m: float) -> float:
    """Rate charged on negative cash: 3M T-bill + 150bp."""
    return float(tbill_3m) + config.CASH_DEBIT_SPREAD_BPS / 1e4
