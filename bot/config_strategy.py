"""Strategy configuration — scoring weights, selection tiers, exit rules, timing."""

# ═══════════════════════════════════════════════════
# Composite scoring weights (3:50 PM signal model)
# ═══════════════════════════════════════════════════
# vs_sector is zeroed until a real sector mapping exists.
# Its weight is redistributed to vs_market.
SCORE_WEIGHT_INTRADAY_RETURN = 0.20
SCORE_WEIGHT_PROXIMITY_HIGH = 0.15
SCORE_WEIGHT_VOLUME_VS_AVG = 0.20
SCORE_WEIGHT_VOLUME_TREND = 0.10
SCORE_WEIGHT_VS_MARKET = 0.25   # absorbs the old 0.15 sector weight
SCORE_WEIGHT_ATR_PCT = -0.10    # negative = penalty for high volatility

# ═══════════════════════════════════════════════════
# Account-tier selection presets
# ═══════════════════════════════════════════════════
STRATEGY_TIERS = [
    {"max_equity": 25_000,  "min_bucket": 4, "max_positions": 25},
    {"max_equity": 100_000, "min_bucket": 4, "max_positions": 25},
    {"max_equity": None,    "min_bucket": 4, "max_positions": 25},
]

# ═══════════════════════════════════════════════════
# Position sizing — HEAD / TAIL allocation
# ═══════════════════════════════════════════════════
MAX_LEVERAGE = 1.0          # 1.0 = no margin (cash account)
ADV_CAP_PCT = 0.003         # 0.3% of ADV max position size
MAX_POSITION_DOLLARS = 50_000  # Legacy: absolute dollar cap (used by position_manager)
MIN_SHARES = 25             # Minimum share count per position

HEAD_COUNT = 10             # Fixed number of equal-weight HEAD positions
TAIL_MAX_POSITIONS = 15     # Max tail candidates after head symbols removed
MAX_TOTAL_POSITIONS = 25    # Hard cap: HEAD_COUNT + TAIL_MAX_POSITIONS
MAX_HEAD_POSITIONS = HEAD_COUNT   # Compat alias used by SelectionConfig

# ═══════════════════════════════════════════════════
# Exit rules (morning of T+1)
# ═══════════════════════════════════════════════════
# Simple rule: market sell ALL positions at 9:40 AM, no conditions.
V2_FAILSAFE_TIME = "09:45"   # Post-exit failsafe verification

# ═══════════════════════════════════════════════════
# Execution safety — buying power buffer + mop-up
# ═══════════════════════════════════════════════════
ENTRY_BP_BUFFER_PCT = 0.98       # Size each order to 98% of reported buying power
ENTRY_MIN_DEPLOY_PCT = 0.95      # If first pass deploys <95%, run mop-up pass
ENTRY_MOPUP_MAX_POSITIONS = 5    # Max extra candidates to try in mop-up

# ═══════════════════════════════════════════════════
# Afternoon timeline (T-1 entry day)
# ═══════════════════════════════════════════════════
DATA_COLLECTION_TIME = "15:30"   # Begin universe pipeline
SCORING_TIME = "15:48"           # Fetch signal bars + score
ENTRY_TIME = "15:50"             # Execute entries (market orders)

# ═══════════════════════════════════════════════════
# Morning timeline (T+1 exit day)
# ═══════════════════════════════════════════════════
MARKET_OPEN_TIME = "09:30"
EXIT_940_TIME = "09:40"          # Market sell ALL positions

# ═══════════════════════════════════════════════════
# Sector ETFs — kept for future use when sector mapping is added
# ═══════════════════════════════════════════════════
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
}
MARKET_BENCHMARK = "SPY"

