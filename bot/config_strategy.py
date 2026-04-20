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
    {"max_equity": 25_000,  "selection_mode": "top30",  "min_bucket": 4, "max_positions": 30},
    {"max_equity": 100_000, "selection_mode": "top30",  "min_bucket": 4, "max_positions": 30},
    {"max_equity": None,    "selection_mode": "top30",  "min_bucket": 4, "max_positions": 30},
]

# ═══════════════════════════════════════════════════
# Position sizing — HEAD / TAIL allocation
# ═══════════════════════════════════════════════════
MAX_LEVERAGE = 1.0          # 1.0 = no margin (cash account)
ADV_CAP_PCT = 0.003         # 0.3% of ADV max position size
MAX_POSITION_DOLLARS = 50_000  # Legacy: absolute dollar cap (used by position_manager)
MIN_SHARES = 25             # Minimum share count per position

HEAD_PCT = 0.70             # 70% of capital to top-ranked positions
TAIL_PCT = 0.30             # 30% of capital to remaining candidates
MAX_HEAD_POSITIONS = 10     # Equal-weight top N
MAX_TOTAL_POSITIONS = 30    # Hard cap including head + tail

# ═══════════════════════════════════════════════════
# Exit rules (morning of T+1)
# ═══════════════════════════════════════════════════
HARD_STOP_PCT = -0.05       # -5% from entry price -> exit at 9:30 open

# Exit rule: ret_open_to_935 > 0.5% -> exit at 9:35, else -> exit at 11:30
# Threshold lives in exit_classifier.py (UP_MOVE_PCT = 0.5)
V2_FAILSAFE_TIME = "11:35"   # Post-exit failsafe verification

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
V2_CLASSIFY_TIME = "09:35"       # Exit classification + immediate 9:35 exits
EXIT_BUCKET_1130_TIME = "11:30"  # Hold bucket exit

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

# Compat: legacy position_manager methods reference MAX_POSITIONS directly.
MAX_POSITIONS = MAX_TOTAL_POSITIONS
