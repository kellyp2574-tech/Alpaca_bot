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
    {"max_equity": 25_000,  "selection_mode": "top10",  "min_bucket": 4, "max_positions": 10},
    {"max_equity": 100_000, "selection_mode": "top20",  "min_bucket": 4, "max_positions": 20},
    {"max_equity": None,    "selection_mode": "bucket",  "min_bucket": 4, "max_positions": 100},
]

# ═══════════════════════════════════════════════════
# Position sizing
# ═══════════════════════════════════════════════════
MAX_LEVERAGE = 1.0          # 1.0 = no margin (cash account)
ADV_CAP_PCT = 0.003         # 0.3% of ADV max position size
MAX_POSITION_DOLLARS = 50_000  # Absolute dollar cap per position

# ═══════════════════════════════════════════════════
# Exit rules (morning of T+1)
# ═══════════════════════════════════════════════════
HARD_STOP_PCT = -0.05       # -5% from entry price → exit at 9:30 open
DROP_STOP_PCT = 0.06         # 6% drop from open-to-9:35 high → exit at 9:35
EXIT_TIME = "11:00"          # Exit ALL remaining positions at this time

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
FIRST_CHECKPOINT_TIME = "09:35"  # 6% drop-stop check

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
# Resolved from the largest tier so old code doesn't crash if invoked.
MAX_POSITIONS = max(t["max_positions"] for t in STRATEGY_TIERS)
