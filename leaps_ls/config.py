"""Single source of truth for universe, dates, and all model parameters (see PLAN.md)."""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# ---------------------------------------------------------------- universe & sample
STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "INTC", "CSCO", "IBM", "ORCL",
    "TXN", "QCOM", "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "JNJ", "UNH", "LLY",
    "ABBV", "MRK", "PFE", "PG", "KO", "PEP", "WMT", "MCD", "HD", "XOM", "CVX", "CAT",
]
INDEX_UNIVERSE = ["SPY", "QQQ"]
UNIVERSE = STOCK_UNIVERSE + INDEX_UNIVERSE

DATA_START = "2005-01-01"          # raw data download start
BACKTEST_START = "2007-01-02"      # portfolio start (after momentum lookback)
MOMENTUM_LOOKBACK = 252            # 12 months
MOMENTUM_SKIP = 21                 # skip most recent month
REBALANCE_FREQ = "MS"              # monthly (month start)

# ---------------------------------------------------------------- vol model (PLAN §5.2)
EWMA_LAMBDA = 0.94
EWMA_MIN_WINDOW = 63
IV_MULT_LO = 1.00                  # clamp bounds for per-name IV/RV multiplier
IV_MULT_HI = 1.35
IV_CAP = 1.00                      # Phase-3 addition: absolute cap on the IV proxy (decimal).
                                   # The plan clamps the multiplier, not IV; uncapped EWMA
                                   # priced 2y LEAPS at 150-250% vol in 2008-09, which no
                                   # market traded at — the cap bounds degenerate marks.
SKEW_CLAMP = 0.15                  # max |skew adjustment| to IV
INDEX_TICKERS_VIX_ANCHOR = INDEX_UNIVERSE  # IV anchored to VIX for these

# ---------------------------------------------------------------- contract selection (PLAN §5.4)
DELTA_TARGET = 0.80                # |delta| target for stock-substitute legs
DELTA_BAND = (0.75, 0.85)
TENOR_TARGET_DAYS = 730
TENOR_MIN_DAYS = 365
TENOR_MAX_DAYS = 1100
LEAPS_MONTH = 1                    # January expiry cycle
SYNTH_STRIKE_MODE = "forward"      # strike nearest S*exp(rT)

# ---------------------------------------------------------------- rolls (PLAN §5.5)
ROLL_DTE_THRESHOLD = 180           # roll when remaining DTE < this
ROLL_DELTA_BAND = (0.60, 0.95)     # sensitivity only

# ---------------------------------------------------------------- frictions (PLAN §5.7)
COMMISSION_PER_CONTRACT = 0.65
COMMISSION_STOCK_PER_SHARE = 0.005
SPREAD_TIERS = {                   # half-spread as fraction of premium, by |delta|
    "itm_deep": 0.01,              # |Δ| >= 0.70  (base values; recalibrated from live chains)
    "itm_shallow": 0.02,           # 0.50 <= |Δ| < 0.70
    "atm_otm": 0.03,               # |Δ| < 0.50
}
SPREAD_FLOOR_USD = 0.05            # minimum half-spread per share
STOCK_SPREAD_BPS = 1.0             # half-spread on stock legs, bps of notional
BORROW_BPS_BASE = 30.0             # general-collateral borrow, bps/yr on short value
CASH_CREDIT_SPREAD_BPS = 25.0      # cash earns tbill - this
CASH_DEBIT_SPREAD_BPS = 150.0      # negative cash charged tbill + this

# ---------------------------------------------------------------- portfolio (PLAN §5.9)
QUINTILE = 5                       # top/bottom quintile sorts
INITIAL_NAV = 1.0
SIZING = "delta_equivalent"        # or "share_equivalent" (sensitivity)

# ---------------------------------------------------------------- sensitivity grid (PLAN §7)
SENSITIVITIES = {
    "spread_mult": [0.5, 1.0, 2.0, 3.0],
    "borrow_bps": [0.0, 30.0, 100.0, 300.0],
    "delta_target": [0.70, 0.80, 0.90],
    "tenor_target_days": [365, 730],
    "roll_dte": [90, 180, 365],
    "iv_mult_adj": [0.9, 1.0, 1.1],
    "sizing": ["delta_equivalent", "share_equivalent"],
}

# ---------------------------------------------------------------- misc
CAPITAL_BASE = 1_000_000.0         # notional $ capital for fixed-$ frictions (NAV reported normalized to 1)
FRED_SERIES = ["DGS3MO", "DGS1", "DGS2"]
FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
FRENCH_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)
VIX_TICKER = "^VIX"


def validate() -> None:
    """Fail fast on internally inconsistent parameters (sensitivity grids, typos)."""
    errs: list[str] = []

    def _chk(cond: bool, msg: str) -> None:
        if not cond:
            errs.append(msg)

    _chk(0.0 < DELTA_TARGET < 1.0, f"DELTA_TARGET={DELTA_TARGET} must be in (0, 1)")
    lo, hi = DELTA_BAND
    _chk(lo <= DELTA_TARGET <= hi,
         f"DELTA_TARGET={DELTA_TARGET} outside selection band {DELTA_BAND}")
    rlo, rhi = ROLL_DELTA_BAND
    _chk(0.0 < rlo < rhi <= 1.0, f"ROLL_DELTA_BAND={ROLL_DELTA_BAND} must be ordered in (0, 1]")
    _chk(TENOR_MIN_DAYS < TENOR_TARGET_DAYS <= TENOR_MAX_DAYS,
         f"TENOR_TARGET_DAYS={TENOR_TARGET_DAYS} outside [{TENOR_MIN_DAYS}, {TENOR_MAX_DAYS}]")
    _chk(TENOR_MAX_DAYS > TENOR_MIN_DAYS, "tenor bounds must be increasing")
    _chk(ROLL_DTE_THRESHOLD < TENOR_MIN_DAYS,
         f"ROLL_DTE_THRESHOLD={ROLL_DTE_THRESHOLD} must be below TENOR_MIN_DAYS "
         "(positions would otherwise roll before reaching target tenor)")
    _chk(QUINTILE >= 2 and isinstance(QUINTILE, int),
         f"QUINTILE={QUINTILE} must be an integer >= 2")
    _chk(SIZING in ("delta_equivalent", "share_equivalent"), f"SIZING={SIZING!r} unknown")
    _chk(1.0 - 0.99 <= EWMA_LAMBDA < 1.0, f"EWMA_LAMBDA={EWMA_LAMBDA} must be in [0.01, 1)")
    _chk(0.0 < IV_MULT_LO <= IV_MULT_HI, "IV multiplier clamp bounds must be ordered positive")
    _chk(IV_CAP > IV_MULT_LO * 0.10, f"IV_CAP={IV_CAP} implausibly low vs multiplier floor")
    _chk(SKEW_CLAMP >= 0.0, "SKEW_CLAMP must be non-negative")
    tiers = list(SPREAD_TIERS.values())
    _chk(all(b >= a for a, b in zip(tiers, tiers[1:])),
         f"SPREAD_TIERS must be non-decreasing itm_deep -> atm_otm, got {SPREAD_TIERS}")
    _chk(set(SPREAD_TIERS) == {"itm_deep", "itm_shallow", "atm_otm"},
         f"unexpected spread tier names: {sorted(SPREAD_TIERS)}")
    _chk(SPREAD_FLOOR_USD > 0.0 and STOCK_SPREAD_BPS >= 0.0
         and COMMISSION_PER_CONTRACT >= 0.0 and COMMISSION_STOCK_PER_SHARE >= 0.0,
         "friction parameters must be non-negative (floor strictly positive)")
    _chk(CAPITAL_BASE > 0.0, "CAPITAL_BASE must be positive")
    _chk(len(set(UNIVERSE)) == len(UNIVERSE), "duplicate tickers in UNIVERSE")
    if errs:
        raise ValueError("config validation failed:\n  " + "\n  ".join(errs))
