"""Strategy configuration — combined MR_WIDE + GDP_BASE sleeves."""

# ═══════════════════════════════════════════════════
# Combined sleeve mode
# ═══════════════════════════════════════════════════
ENABLE_COMBINED_SLEEVES = True

# Starting live-paper allocation (60/40 MR/GDP)
MR_ALLOCATION_PCT = 0.60
GDP_ALLOCATION_PCT = 0.40

MR_WEIGHT = 0.70                 # MR score weight in combined ranking
MOM_WEIGHT = 0.30                # Momentum score weight in combined ranking

COMBINED_MAX_POSITIONS = 20  # Hard cap across both sleeves

# ═══════════════════════════════════════════════════
# Position sizing (waterfall allocation)
# ═══════════════════════════════════════════════════
MAX_LEVERAGE = 1.0               # 1.0 = no margin (cash account)
ADV_CAP_PCT = 0.003              # 0.3% of ADV max position size
MAX_SINGLE_POSITION_PCT = 0.10 # 10% max of portfolio in one stock
MIN_POSITION_DOLLARS = 50        # Min order notional (skip if ADV cap < this)
MIN_SHARES = 25                  # Minimum share count per position
MAX_POSITION_DOLLARS = 50_000    # Absolute dollar cap per position (legacy)

# ═══════════════════════════════════════════════════
# Sleeve 1: Mean Reversion — MR_WIDE
# ═══════════════════════════════════════════════════
MR_MIN_PRICE = 1.00
MR_MAX_PRICE = 5.00
MR_DAY_RET_MAX = -0.03          # day return <= -3%
MR_VOLUME_RATIO_MIN = 1.5       # today volume / expected volume >= 1.5x
MR_CLOSE_POSITION_MAX = 0.20    # close in bottom 20% of day range
MR_LATE_DROP_MAX = None         # optional: set -0.01 for stricter late-drop filter

MR_MAX_POSITIONS = 12           # target slots for MR sleeve (60% of 20)
MR_USE_RANDOM_SELECTION = False # deterministic first-N selection for live trading

# ═══════════════════════════════════════════════════
# Sleeve 2: Green-Day Pullback — GDP_BASE
# ═══════════════════════════════════════════════════
GDP_MIN_PRICE = 1.00
GDP_MAX_PRICE = 10.00
GDP_DAY_RET_MIN = 0.01          # day return >= +1%
GDP_DAY_RET_MAX = 0.10          # day return <= +10%
GDP_REQUIRE_BELOW_VWAP = True   # price must be below intraday VWAP
GDP_LATE_MOM_MAX = 0.0          # late momentum 15:30->signal must be <= 0 (decelerating)
GDP_MAX_CLOSE_POSITION = None   # optional: limit close_position (None = no limit)

GDP_MAX_POSITIONS = 8           # target slots for GDP sleeve (40% of 20)

# ═══════════════════════════════════════════════════
# Execution safety — buying power buffer + mop-up
# ═══════════════════════════════════════════════════
ENTRY_BP_BUFFER_PCT = 0.98       # Size each order to 98% of reported buying power
ENTRY_MIN_DEPLOY_PCT = 0.95      # If first pass deploys <95%, run mop-up pass
ENTRY_MOPUP_MAX_POSITIONS = 0    # 0 = mop-up disabled (paper trading phase)

# ═══════════════════════════════════════════════════
# Afternoon timeline (T-1 entry day)
# ═══════════════════════════════════════════════════
DATA_COLLECTION_TIME = "15:30"   # Begin universe pipeline
SCORING_TIME = "15:55"           # Score using latest available bars/snapshot
ENTRY_TIME = "15:50"             # Execute immediately after scoring

# ═══════════════════════════════════════════════════
# Morning timeline (T+1 exit day)
# ═══════════════════════════════════════════════════
MARKET_OPEN_TIME = "09:30"
GDP_EXIT_TIME = "09:30"          # GDP sleeve exits at market open
MR_EXIT_TIME = "09:30"           # MR sleeve exits at market open
V2_FAILSAFE_TIME = "09:45"       # Post-exit failsafe verification

# ═══════════════════════════════════════════════════
# Sector ETFs — kept for future use
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

