"""Roll scheduling for open option positions (PLAN §5.5).

Base rule (checked at each monthly rebalance): roll when remaining DTE falls
below ROLL_DTE_THRESHOLD. Delta-band roll (|delta| outside ROLL_DELTA_BAND) is a
sensitivity option enabled per run.
"""
from __future__ import annotations

from .. import config


def needs_roll(
    position: dict,
    day_ord: int,
    roll_dte: int = config.ROLL_DTE_THRESHOLD,
    delta_band: tuple[float, float] | None = None,
    abs_delta: float | None = None,
) -> bool:
    """True when the position must be rolled at this rebalance date.

    ``position`` carries ``expiry_ord``; ``day_ord`` is the current day ordinal.
    The delta-band rule additionally requires the position's current |delta|.
    """
    dte = int(position["expiry_ord"]) - int(day_ord)
    if dte < roll_dte:
        return True
    if delta_band is not None and abs_delta is not None:
        lo, hi = delta_band
        if not (lo <= abs_delta <= hi):
            return True
    return False
