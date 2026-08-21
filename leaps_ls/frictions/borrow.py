"""Stock-borrow fee scenarios for cash shorts (PLAN §5.7).

No free borrow-fee time series exists; scenarios follow the literature
(D'Avolio 2002; Geczy/Musto/Reed 2002; Drechsler & Drechsler 2016):
general collateral ~30 bps/yr base, with stress scenarios.
"""
from __future__ import annotations

from .. import config

SCENARIOS_BPS: dict[str, float] = {
    "none": 0.0,
    "base": config.BORROW_BPS_BASE,  # general collateral
    "high": 100.0,
    "stress": 300.0,
}


def borrow_rate_bps(scenario: str = "base") -> float:
    """Annual borrow fee in bps for a named scenario."""
    return float(SCENARIOS_BPS[scenario])


def daily_borrow_cost(
    short_market_value: float, bps: float | None = None, scenario: str = "base", days: int = 1
) -> float:
    """Borrow cost in $ accrued over ``days`` calendar days on the short market value."""
    rate_bps = borrow_rate_bps(scenario) if bps is None else float(bps)
    return float(short_market_value) * (rate_bps / 1e4) * days / 365.0


def stress_mix_bps(
    names: list[str], fraction: float = 0.25, stress_bps: float = 500.0,
    base_bps: float | None = None,
) -> dict[str, float]:
    """Stress mix: a deterministic ``fraction`` of names at ``stress_bps``, rest at base."""
    base = config.BORROW_BPS_BASE if base_bps is None else float(base_bps)
    ordered = sorted(names)
    n_stress = int(round(len(ordered) * fraction))
    stressed = set(ordered[:: max(1, round(1.0 / fraction))][:n_stress])
    return {n: (stress_bps if n in stressed else base) for n in names}
