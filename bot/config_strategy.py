"""Strategy configuration — paper test: clean overnight MR sleeve.

Current paper-test candidate from the clean no-lookahead cache research:
- Entry/signal: 15:45
- Exit: 09:30
- Sleeve: MR only, cheap late-day washout
- Filters: entry price $1-$2, return vs prior close <= -5%, close-location <= 0.25, ADV >= $1M
- Rank: lowest close-location first
- Max positions: 3
- Regime sizing: full size when 3-ETF basket is red before 15:45, half size when positive
"""

import os

# ═══════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "combined_overnight_bot.log")
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ═══════════════════════════════════════════════════
# ETF Router Configuration (9:30-10:00 AM)
# ═══════════════════════════════════════════════════
# Master switches
ETF_ROUTER_ENABLED = True
MR_OVERNIGHT_ENABLED = True
MR_PERMISSION_MODE = "skip_if_router_traded"  # "skip_if_router_traded" or "all_regimes"

# ETF symbols for tape measurement
ETF_ROUTER_SYMBOLS = ["QQQ", "SPY", "IWM", "XLK", "VXX", "SQQQ", "UVXY", "TQQQ"]

# Capital allocation for ETF router (separate from MR capital)
ETF_ROUTER_CAPITAL_PCT = 0.90  # 90% of equity for ETF router (single daily position, maximize deployment)

# Tape recording cadence. The main loop ticks every 1 s during the
# 09:24-10:02 hot window; calling get_snapshots on 8 ETFs every tick is
# ~2,400 API calls per session for no benefit (the router decision only
# cares about returns vs 09:30 open and the 09:45 continuation marker,
# not sub-second granularity). Throttle to once every N seconds.
ETF_TAPE_UPDATE_INTERVAL_SECONDS = 5

# ═══════════════════════════════════════════════════
# Combined sleeve mode
# ═══════════════════════════════════════════════════
ENABLE_COMBINED_SLEEVES = True

# Paper test: MR-only. GDP/MOM is disabled until the next sleeve is tested.
MR_WEIGHT = 1.00
MOM_WEIGHT = 0.00

# Backward-compatible names used by the live allocator.
MR_ALLOCATION_PCT = MR_WEIGHT
GDP_ALLOCATION_PCT = MOM_WEIGHT

COMBINED_MAX_POSITIONS = 3   # Paper MR test: top 3 only

# ═══════════════════════════════════════════════════
# Position sizing (waterfall allocation)
# ═══════════════════════════════════════════════════
MAX_LEVERAGE = 1.0               # 1.0 = no margin (cash account)
ADV_CAP_PCT = 0.003              # 0.3% of ADV max position size
MAX_SINGLE_POSITION_PCT = 0.34 # 34% max per position allows 3 positions to use full 100% sleeve budget
MIN_POSITION_DOLLARS = 50        # Min order notional (skip if ADV cap < this)
MIN_SHARES = 25                  # Minimum share count per position
MAX_POSITION_DOLLARS = 50_000    # Absolute dollar cap per position (legacy)

# ADV multiplier for IEX data (IEX reports lower volume than composite)
ADV_DOLLAR_MULTIPLIER = 50.0     # Multiply raw IEX ADV by this for sizing

# Leftover redeployment: redeploy unused sleeve budget to other candidates
ENABLE_LEFTOVER_REDEPLOYMENT = True

# ═══════════════════════════════════════════════════
# Entry execution (concurrent submission with client_order_id)
# ═══════════════════════════════════════════════════
ENTRY_SUBMIT_TIMEOUT_SECONDS = 2       # Timeout per order submission (short to avoid blocking)
ENTRY_RECONCILE_TIMEOUT_SECONDS = 3    # Timeout for client_order_id reconciliation
ENTRY_SUBMIT_MAX_WORKERS = 8           # Max concurrent workers for buy submission
ENTRY_BP_BUFFER_PCT = 0.98             # 2% buying power buffer
ENTRY_MAX_SPREAD_PCT = 0.05            # Max spread for entry execution gate

# Marketable-limit entry slippage cap. The 15:45 MR entries submit
# limit buys at ``ask * (1 + ENTRY_MAX_SLIPPAGE_PCT)`` instead of pure
# market orders, so a sudden quote dislocation cannot pay more than this
# fraction above the prevailing ask. Falls back to a market order when
# the ask is missing.
ENTRY_MAX_SLIPPAGE_PCT = 0.02          # 2% above ask for low-priced MR names

# ETF router 10:00 entry uses tighter caps because the universe is
# highly liquid (SPY/QQQ/IWM/TQQQ/SQQQ/UVXY/VXX/XLK).
ETF_ENTRY_MAX_SPREAD_PCT = 0.005       # 0.5% spread gate for ETF entry
ETF_ENTRY_MAX_SLIPPAGE_PCT = 0.005     # 0.5% above ask for ETF marketable limit
ETF_ENTRY_MAX_STALE_SECONDS = 10.0     # Quote freshness gate for ETF entry

# ═══════════════════════════════════════════════════
# Sleeve 1: Mean Reversion — MR_WIDE
# ═══════════════════════════════════════════════════
MR_MIN_PRICE = 1.00
MR_MAX_PRICE = 2.00
MR_DAY_RET_MAX = -0.04          # clean-cache candidate: return vs prior/entry-day signal <= -4%
MR_VOLUME_RATIO_MIN = 0.0       # no relative-volume requirement in finalist test
MR_CLOSE_POSITION_MAX = 0.25    # close in bottom 25% of day range
MR_LATE_DROP_MAX = None         # optional, off
MR_MIN_AVG_DOLLAR_VOLUME = 1_000_000

# Finalist rank is close_location ascending, top 3.
MR_RANK_BY_CLOSE_LOCATION_ONLY = True
MR_MIN_CANDIDATES = 2
MR_MAX_POSITIONS = 3            # top 3 only

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

GDP_MAX_POSITIONS = 0           # GDP disabled for this paper MR test

# ═══════════════════════════════════════════════════
# MR regime sizing for this paper test
# ═══════════════════════════════════════════════════
ENABLE_MR_ETF_REGIME_SIZING = True
MR_ETF_REGIME_SYMBOLS = ["SPY", "IWM", "QQQ"]
MR_ETF_NEGATIVE_SIZE_MULT = 1.0
MR_ETF_POSITIVE_SIZE_MULT = 0.5

# Daily loss circuit breaker — abort 15:50 entries if today's drawdown
# (equity vs yesterday's close equity) is worse than this threshold.
# 0.0 disables the check. 0.05 = abort if today's PnL < -5%.
DAILY_LOSS_LIMIT_PCT = 0.05

# ═══════════════════════════════════════════════════
# Afternoon timeline (T-1 entry day)
# ═══════════════════════════════════════════════════
DATA_COLLECTION_TIME = "15:30"   # Begin universe pipeline
SCORING_TIME = "15:45"           # Score late-day MR sleeve
ENTRY_TIME = "15:45"             # Execute immediately after scoring

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
# ETF Router timeline (09:00 - 16:00)
# ═══════════════════════════════════════════════════
BOT_START_TIME = "09:00"           # Bot startup/pre-market prep
ROUTER_DECISION_TIME = "10:00"     # 10:00 AM ETF router decision
ROUTER_ENTRY_AFTER = "10:00"       # Enter ETF strictly after 10:00

# ETF exit checkpoints (branch-specific)
UVXY_EXIT_TIME = "11:00"           # UVXY crash branch exit
SQQQ_EXIT_TIME = "14:00"           # SQQQ Goldilocks exit
TQQQ_EXIT_TIME = "15:00"           # All TQQQ branches exit

# ═══════════════════════════════════════════════════
# Rolling premarket dynamic limit management (05:00 → 06:00)
# ═══════════════════════════════════════════════════
# At 15-minute checkpoints, only "decisive" symbols are acted on; unclear
# symbols wait for the next checkpoint.
ENABLE_PREMARKET_DYNAMIC_LIMIT_SELLS = True

# ═══════════════════════════════════════════════════
# Premarket dynamic limit classification
# ═══════════════════════════════════════════════════
# Rolling checkpoints at 05:00, 05:15, 05:30, 05:45, and final at 06:00.
# Uses delayed SIP historical bars ending at decision_dt - 16 minutes.
PREMARKET_DYNAMIC_START_TIME = "05:00"
PREMARKET_DYNAMIC_FINAL_TIME = "06:00"
PREMARKET_DYNAMIC_CHECK_INTERVAL_MINUTES = 15
PREMARKET_SIP_DELAY_MINUTES = 16  # Delay minutes for SIP historical bars (must be >= 15)
PREMARKET_DYNAMIC_LIMIT_TIME_IN_FORCE = "day"
PREMARKET_DYNAMIC_LIMIT_EXTENDED_HOURS = True
PREMARKET_DYNAMIC_MAX_STALE_MINUTES = 60

# Dynamic classification thresholds. These are intentionally lenient for the
# delayed-SIP feed: runners should usually show meaningful premarket activity
# in the historical SIP bars, but a sparse stretch (e.g. illiquid ticker with
# few prints in the last 30 minutes before the 16-minute delay window) should
# not automatically invalidate the signal.
PREMARKET_DYNAMIC_DEFAULT_LIMIT_PCT = 0.05
PREMARKET_DYNAMIC_SPARSE_HIGH_RETURN_LIMIT_PCT = 0.10
PREMARKET_DYNAMIC_VERY_HIGH_RETURN_NO_CAP_PCT = 0.10
PREMARKET_DYNAMIC_HIGH_RETURN_NO_CAP_PCT = 0.05
PREMARKET_DYNAMIC_MODERATE_RETURN_PCT = 0.02

# ═══════════════════════════════════════════════════
# Legacy SIP-snapshot backup knobs (DISABLED — not wired at runtime)
# ═══════════════════════════════════════════════════
# The active premarket classifier in `bot/premarket_classifier.py` calls the
# historical-bars endpoint with `feed=sip` and an end timestamp ~16 minutes
# behind decision time (delayed SIP). It does NOT use a live snapshot feed
# and these knobs are not read anywhere. Left in place purely as a placeholder
# for a future SIP-snapshot cross-check; toggle the BACKUP flag and re-wire
# the classifier if you bring that path back online.
USE_SIP_SNAPSHOT_PREMARKET_BACKUP = False
SIP_SNAPSHOT_MAX_SPREAD_PCT = 0.02       # max spread for SIP midpoint to be usable
SIP_IEX_CONFIRM_DIFF_PCT = 0.0075        # SIP-vs-IEX agreement threshold
SIP_ALLOW_HIGH_CORRECTION = True         # let SIP raise the premarket high

# DEPRECATED: No-data fallback limit is no longer used.
# New behavior: No data = no order. This config is kept for reference only.
# PREMARKET_DYNAMIC_NO_DATA_FALLBACK_LIMIT_PCT = 0.03

# ═══════════════════════════════════════════════════
# Fast open market exit (streamlined 09:30 liquidation)
# ═══════════════════════════════════════════════════
# When enabled, at 09:30 the bot submits all market sells in batch using
# the frozen 09:25 broker-position plan. No green/red decisions, no trailing stops,
# just simple market sell for everything. This is faster and more reliable.
ENABLE_FAST_OPEN_MARKET_EXIT = True

# ═══════════════════════════════════════════════════
# MOO/Open-Auction exit mode (09:25 OPG orders) - DISABLED
# ═══════════════════════════════════════════════════
# DISABLED: MOO/Open-Auction proved unreliable. Using batched open market exits instead.
# When enabled, at 09:25 the bot submits MOO (market-on-open) orders using
# time_in_force="opg" for all overnight positions. If orders are canceled/expired/rejected,
# the bot immediately submits market sells for remaining shares. Fallback to market
# sell at 09:30:30 for any remaining positions.
ENABLE_OPEN_AUCTION_EXIT = False
OPEN_AUCTION_FALLBACK_TIME = "09:30:30"
OPEN_AUCTION_TIF = "opg"

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


