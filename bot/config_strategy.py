"""Strategy configuration — static 70/30 combined overnight sleeves.

Production default from combined-cache research:
- Entry/signal: 15:50
- Exit: 09:30
- Static weights: 70% MR / 30% momentum pullback (GDP)
- Max single-name exposure: 10% of equity
"""

# ═══════════════════════════════════════════════════
# Combined sleeve mode
# ═══════════════════════════════════════════════════
ENABLE_COMBINED_SLEEVES = True

# Static production allocation from robustness sweep.
# Note: live "GDP" sleeve is the momentum-pullback sleeve from the backtests.
MR_WEIGHT = 0.70
MOM_WEIGHT = 0.30

# Backward-compatible names used by the live allocator.
MR_ALLOCATION_PCT = MR_WEIGHT
GDP_ALLOCATION_PCT = MOM_WEIGHT

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

MR_MAX_POSITIONS = 12           # max MR slots; dollar budget is controlled by MR_ALLOCATION_PCT

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

GDP_MAX_POSITIONS = 8           # max GDP/MOM slots; dollar budget is controlled by GDP_ALLOCATION_PCT

# ═══════════════════════════════════════════════════
# Execution safety — buying power buffer
# ═══════════════════════════════════════════════════
ENTRY_BP_BUFFER_PCT = 0.98       # Size each order to 98% of reported buying power

# Daily loss circuit breaker — abort 15:50 entries if today's drawdown
# (equity vs yesterday's close equity) is worse than this threshold.
# 0.0 disables the check. 0.05 = abort if today's PnL < -5%.
DAILY_LOSS_LIMIT_PCT = 0.05

# ═══════════════════════════════════════════════════
# Afternoon timeline (T-1 entry day)
# ═══════════════════════════════════════════════════
DATA_COLLECTION_TIME = "15:30"   # Begin universe pipeline
SCORING_TIME = "15:50"           # Score using 9:30-15:50 bars
ENTRY_TIME = "15:50"             # Execute immediately after scoring

# ═══════════════════════════════════════════════════
# Morning timeline (T+1 exit day)
# ═══════════════════════════════════════════════════
MARKET_OPEN_TIME = "09:30"
GDP_EXIT_TIME = "09:30"          # GDP sleeve exits at market open
MR_EXIT_TIME = "09:30"           # MR sleeve exits at market open
V2_FAILSAFE_TIME = "09:45"       # Post-exit failsafe verification

# Cancel any resting overnight limits before the 09:30 market exit logic.
MORNING_CANCEL_OPEN_ORDERS_TIME = "09:25"

# ═══════════════════════════════════════════════════
# Rolling premarket dynamic limit management (05:00 → 06:00)
# ═══════════════════════════════════════════════════
# At 15-minute checkpoints, only "decisive" symbols are acted on; unclear
# symbols wait for the final 06:00 checkpoint. Any remaining limits are
# canceled at MORNING_CANCEL_OPEN_ORDERS_TIME before the 09:30 exit path.
ENABLE_PREMARKET_DYNAMIC_LIMIT_SELLS = True
PREMARKET_DYNAMIC_START_TIME = "05:00"
PREMARKET_DYNAMIC_FINAL_TIME = "06:00"
PREMARKET_DYNAMIC_CHECK_INTERVAL_MINUTES = 15
PREMARKET_DYNAMIC_DATA_FEED = "iex"
PREMARKET_DYNAMIC_LIMIT_TIME_IN_FORCE = "day"
PREMARKET_DYNAMIC_LIMIT_EXTENDED_HOURS = True
PREMARKET_DYNAMIC_MAX_STALE_MINUTES = 60

# Dynamic classification thresholds. These are intentionally lenient for IEX:
# runners should usually show at least some IEX activity, but sparse IEX bars
# should not automatically invalidate the signal.
PREMARKET_DYNAMIC_DEFAULT_LIMIT_PCT = 0.05
PREMARKET_DYNAMIC_SPARSE_HIGH_RETURN_LIMIT_PCT = 0.10
PREMARKET_DYNAMIC_VERY_HIGH_RETURN_NO_CAP_PCT = 0.10
PREMARKET_DYNAMIC_HIGH_RETURN_NO_CAP_PCT = 0.05
PREMARKET_DYNAMIC_MODERATE_RETURN_PCT = 0.02

# ═══════════════════════════════════════════════════
# Fast open market exit (streamlined 09:30 liquidation)
# ═══════════════════════════════════════════════════
# When enabled, at 09:30 the bot submits all market sells in batch using
# the frozen 09:25 broker-position plan. No green/red decisions, no trailing stops,
# just simple market sell for everything. This is faster and more reliable.
ENABLE_FAST_OPEN_MARKET_EXIT = True

# ═══════════════════════════════════════════════════
# Paper research exit: red-open trailing stop
# ═══════════════════════════════════════════════════
# Mutually exclusive with ENABLE_FAST_OPEN_MARKET_EXIT — the fast-exit branch
# wins in integrated_main when both are True, but having both enabled also
# pushes the bot onto the 10:00 failsafe schedule and blocks early-completion,
# producing a confusing hybrid. Keep this False whenever fast exit is on.
ENABLE_RED_OPEN_TRAIL_EXIT = False
RED_OPEN_TRAIL_PCT = 1.0              # Alpaca trail_percent value, e.g. 1.0 = 1%
RED_OPEN_TRAIL_FAILSAFE_TIME = "10:00"
RED_OPEN_TRAIL_PRICE_BUFFER_PCT = 0.0    # match backtest: any open/current price below entry is red


