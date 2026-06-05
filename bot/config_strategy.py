"""Strategy configuration — paper test: clean overnight MR sleeve + intraday ETF router/V2/P1.

Live constants (single source of truth):

    INTRADAY_ETF_ALLOCATION_PCT  = 0.90    # 90% of equity for the intraday ETF sleeve
    MR_ALLOC_PER_POSITION_PCT    = 0.30    # 30% of equity per MR position
    MR_MAX_PRIMARY_POSITIONS     = 3       # Top 3 MR candidates only
    MR_MAX_TOTAL_ALLOCATION_PCT  = 0.90    # Max 90% of equity in the MR sleeve
    MR_ADV_CAP_PCT               = 0.003   # 0.3% of 20-day ADV per symbol

Paper-test sleeve (CLEAN_OVERNIGHT_MR):
- Entry/signal: 15:45
- Exit: 09:30
- Filters: entry price $1-$2, return vs prior close <= -4%, close-location <= 0.25, ADV >= $1M
- Rank: lowest close-location first
- Max positions: 3
- Regime sizing: full size when SPY/IWM/QQQ avg open->signal return is red, half size when positive
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
# Master switches
# ═══════════════════════════════════════════════════
ETF_ROUTER_ENABLED = True
MR_OVERNIGHT_ENABLED = True
MR_PERMISSION_MODE = "skip_if_router_position"  # "skip_if_router_position" = only block when ETF filled; "all_regimes" = never block

# ═══════════════════════════════════════════════════
# Capital allocation — LIVE CONSTANTS
# ═══════════════════════════════════════════════════
# 90% of equity for intraday ETF sleeve (Router > V2 > P1, only one fills per day)
INTRADAY_ETF_ALLOCATION_PCT = 0.90

# MR sleeve sizing
MR_ALLOC_PER_POSITION_PCT = 0.30   # 30% of equity per MR position
MR_MAX_PRIMARY_POSITIONS = 3       # Top 3 MR candidates only
MR_MAX_TOTAL_ALLOCATION_PCT = 0.90 # Max 90% of equity across the MR sleeve
MR_ADV_CAP_PCT = 0.003             # 0.3% of 20-day ADV per symbol

# ═══════════════════════════════════════════════════
# ETF Router Configuration (9:30-10:00 AM)
# ═══════════════════════════════════════════════════
# ETF symbols for tape measurement
ETF_ROUTER_SYMBOLS = ["QQQ", "SPY", "IWM", "XLK", "VXX", "SQQQ", "UVXY", "TQQQ"]

# Tape recording cadence — once every N seconds.
ETF_TAPE_UPDATE_INTERVAL_SECONDS = 5

# How many minutes after the 10:00 decision to actually submit the entry order.
# Keeps entry in a predictable place for logging and scheduling.
ROUTER_ENTRY_DELAY_MINUTES = 5  # 10:00 decision → 10:05 entry

# ═══════════════════════════════════════════════════
# Position sizing (waterfall allocation)
# ═══════════════════════════════════════════════════
MAX_LEVERAGE = 1.0               # 1.0 = no margin (cash account)
MIN_POSITION_DOLLARS = 50        # Min order notional (skip if ADV cap < this)
MIN_SHARES = 25                  # Minimum share count per position
MAX_POSITION_DOLLARS = 50_000    # Absolute dollar cap per position (legacy hard cap)

# ADV multiplier for IEX data (IEX reports lower volume than composite)
ADV_DOLLAR_MULTIPLIER = 50.0

# Leftover redeployment: redeploy unused per-position budget to other ranked candidates
ENABLE_LEFTOVER_REDEPLOYMENT = True
MR_MAX_WATERFALL_POSITIONS = 6  # Absolute max including overflow ranks

# ═══════════════════════════════════════════════════
# Entry execution (concurrent submission with client_order_id)
# ═══════════════════════════════════════════════════
ENTRY_SUBMIT_TIMEOUT_SECONDS = 2       # Per-order submission timeout
ENTRY_RECONCILE_TIMEOUT_SECONDS = 3    # client_order_id reconciliation timeout
ENTRY_SUBMIT_MAX_WORKERS = 8           # Concurrent workers for buy submission AND fill monitoring
ENTRY_BP_BUFFER_PCT = 0.98             # 2% buying-power buffer
ENTRY_MAX_SPREAD_PCT = 0.05            # Max spread for MR entry execution gate
ENTRY_MAX_SLIPPAGE_PCT = 0.02          # Marketable-limit cap above ask for low-priced MR names

# ETF router 10:00 entry uses tighter caps (highly-liquid universe)
ETF_ENTRY_MAX_SPREAD_PCT = 0.005
ETF_ENTRY_MAX_SLIPPAGE_PCT = 0.005
# Staleness threshold for router/V2/P1 ETF entries (IEX can be 10-30s behind SIP).
# A 12-second-old quote on UVXY is normal — do not reject liquid ETFs on IEX latency.
ETF_ENTRY_MAX_STALE_SECONDS = 60.0
# Log a warning when quote age exceeds this but still allow entry.
ETF_ENTRY_WARN_STALE_SECONDS = 30.0

# ═══════════════════════════════════════════════════
# V2 Fallback (Router > V2 > P1 priority)
# ═══════════════════════════════════════════════════
ENABLE_V2_FALLBACK = True
V2_LONG_ENTRY_WINDOW_START = "10:10"
V2_LONG_ENTRY_WINDOW_END = "10:30"
V2_SHORT_ENTRY_WINDOW_START = "10:15"
V2_SHORT_ENTRY_WINDOW_END = "10:30"
V2_HYBRID_CHECKPOINT_TIME = "11:30"
V2_FINAL_EXIT_TIME = "15:30"
V2_QQQ_RANGE_BREAKOUT_THRESHOLD = 0.0045  # 0.45% QQQ range for breakout signal

# V2 subtypes that allow V2 fallback evaluation (router no-trade subtypes)
V2_ALLOWED_SUBTYPES = [
    "LIVE_FLAT_CHOP",
    "LIVE_WEAK_GREEN",
    "LIVE_MILD_RISK_OFF",
    "LIVE_BULLISH_MESSY",
    "LIVE_VOL_WARNING",
]

# V2 polling cadence. The main loop ticks every 1s in the hot window,
# but V2 does not need a fresh REST snapshot every second.
V2_EVAL_INTERVAL_SECONDS = 10

# MR subtypes that allow MR entries (router no-trade subtypes)
# MR only runs on these subtypes that were profitable in backtest
# LIVE_FLAT_CHOP included: router logs "MR allowed at 15:45" on no-trade days,
# and LIVE_FLAT_CHOP is the default no-trade subtype on flat/range-bound days.
MR_ALLOWED_SUBTYPES = [
    "LIVE_VOL_WARNING",
    "LIVE_MILD_RISK_OFF",
    "LIVE_BULLISH_MESSY",
    "LIVE_CRASH_WARNING",
    "LIVE_FLAT_CHOP",
]

V2_TRAIL_PCT = 1.50                # 1.5% trailing stop attached after V2 entry
V2_LONG_VEHICLE = "TQQQ"
V2_SHORT_VEHICLE = "SPXU"

# ═══════════════════════════════════════════════════
# Per-branch Stop-Loss / Take-Profit (bracket orders)
# ═══════════════════════════════════════════════════
# All values are fractions of fill price (e.g. 0.06 = 6%).
# None = that leg is omitted (use OTO with only one side, or plain market).
# Alpaca requires take_profit.limit_price > stop_loss.stop_price for buy orders.
#
# Sleeve          SL       TP     Notes
# A++ long       -6%      +20%   Highest conviction; wide TP
# A long         -6%      None   TP hurt this sleeve; stop only
# A- long        None     None   Only 28 trades, leave baseline
# A- weak        None     None   Only 17 trades, leave baseline
# SQQQ Goldilocks None    +3%    TP only; no stop
# P1 Sleeve      -4%      +5%    Strongest branch-specific result
# V2 Long        None     None   Already has trailing stop; leave unchanged
# V2 Short       -2%      +3%    Improve only if paper test confirms
ETF_SL_TP: dict = {
    "A_PLUS_PLUS_LONG":  {"sl": 0.06,  "tp": 0.20},
    "A_LONG":            {"sl": 0.06,  "tp": None},
    "A_MINUS_LONG":      {"sl": None,  "tp": None},
    "A_MINUS_WEAK":      {"sl": None,  "tp": None},
    "SQQQ_GOLDILOCKS":   {"sl": None,  "tp": 0.03},
    "UVXY_CRASH":        {"sl": None,  "tp": None},
    "P1_FALLBACK":       {"sl": 0.04,  "tp": 0.05},
    "V2_LONG":           {"sl": None,  "tp": None},
    "V2_SHORT":          {"sl": 0.02,  "tp": 0.03},
}

# ═══════════════════════════════════════════════════
# P1 Fallback (lowest priority intraday)
# ═══════════════════════════════════════════════════
ENABLE_P1_FALLBACK = True
P1_ENTRY_TIME = "10:15"
P1_EXIT_TIME = "15:00"
P1_VEHICLE = "TQQQ"
P1_REQUIRED_SUBTYPE = "LIVE_BULLISH_MESSY"
P1_REQUIRED_XLK_POSITIVE = True

# ═══════════════════════════════════════════════════
# Sleeve 1: Mean Reversion — MR_WIDE
# ═══════════════════════════════════════════════════
MR_MIN_PRICE = 1.00
MR_MAX_PRICE = 2.00
MR_DAY_RET_MAX = -0.04
MR_VOLUME_RATIO_MIN = 0.0
MR_CLOSE_POSITION_MAX = 0.25
MR_LATE_DROP_MAX = None
MR_MIN_AVG_DOLLAR_VOLUME = 1_000_000

# Finalist rank is close_location ascending, top 3.
MR_RANK_BY_CLOSE_LOCATION_ONLY = True
MR_MIN_CANDIDATES = 2

# ═══════════════════════════════════════════════════
# Early Close / Trading Calendar Handling
# ═══════════════════════════════════════════════════
SKIP_MR_ON_EARLY_CLOSE = True
SKIP_INTRADAY_ETF_ON_EARLY_CLOSE = True

# ═══════════════════════════════════════════════════
# MR regime sizing
# ═══════════════════════════════════════════════════
ENABLE_MR_ETF_REGIME_SIZING = True
MR_ETF_REGIME_SYMBOLS = ["SPY", "IWM", "QQQ"]
MR_ETF_NEGATIVE_SIZE_MULT = 1.0
MR_ETF_POSITIVE_SIZE_MULT = 0.5

# Daily-loss circuit breaker (vs yesterday's close equity).
# 0.05 = abort 15:45 entries AND 10:00 ETF entry if today's PnL < -5%.
DAILY_LOSS_LIMIT_PCT = 0.05

# ═══════════════════════════════════════════════════
# Afternoon timeline (T-1 entry day)
# ═══════════════════════════════════════════════════
DATA_COLLECTION_TIME = "15:30"
SCORING_TIME = "15:45"
ENTRY_TIME = "15:45"  # MR entries at 15:45 (not 15:50)
ENTRY_HARD_CUTOFF_TIME = "15:58:30"   # entry_executor refuses new submits after this

# ═══════════════════════════════════════════════════
# Morning timeline (T+1 exit day)
# ═══════════════════════════════════════════════════
MARKET_OPEN_TIME = "09:30"
MORNING_EXIT_TIME = "09:30"          # MR sleeve flat by this time
V2_FAILSAFE_TIME = "09:45"           # Post-exit failsafe verification

# Cancel any resting overnight limits before the 09:30 market exit logic.
MORNING_CANCEL_OPEN_ORDERS_TIME = "09:25"

# ═══════════════════════════════════════════════════
# Unified System Timeline (05:00 - 16:00)
# ═══════════════════════════════════════════════════
# 05:00 - Bot startup, load state, verify calendar
# 05:00-06:00 - Premarket MR monitoring and dynamic limit decisions
# 09:25 - Cancel premarket limits, prepare for open
# 09:30 - Batch market sell remaining MR positions
# 09:30-10:00 - Build ETF tape snapshot
# 10:00 - Router decision (priority 1)
# 10:05 - Router entry (if fired)
# 10:10-10:30 - V2 fallback window (priority 2)
# 10:15 - P1 fallback entry (priority 3)
# 11:30 - V2 hybrid checkpoint
# 14:00 - SQQQ exit
# 15:00 - Router/P1 TQQQ exit
# 15:30 - V2 final exit / hard flatten all intraday ETF
# 15:45 - MR scoring + entry (entries at 15:45, not 15:50)
# 16:00 - Market close, final reconciliation

BOT_START_TIME = "05:00"
ROUTER_DECISION_TIME = "10:00"
ROUTER_ENTRY_TIME = "10:05"

# Intraday ETF exit checkpoints
SQQQ_EXIT_TIME = "14:00"
TQQQ_EXIT_TIME = "15:00"
INTRADAY_ETF_HARD_FLATTEN_TIME = "15:30"

# ═══════════════════════════════════════════════════
# Rolling premarket dynamic limit management (05:00 → 06:00)
# ═══════════════════════════════════════════════════
ENABLE_PREMARKET_DYNAMIC_LIMIT_SELLS = True
PREMARKET_DYNAMIC_START_TIME = "05:00"
PREMARKET_DYNAMIC_FINAL_TIME = "06:00"
PREMARKET_DYNAMIC_CHECK_INTERVAL_MINUTES = 15
PREMARKET_SIP_DELAY_MINUTES = 16   # SIP historical-bars delay (must be >= 15)
PREMARKET_DYNAMIC_LIMIT_TIME_IN_FORCE = "day"
PREMARKET_DYNAMIC_LIMIT_EXTENDED_HOURS = True
PREMARKET_DYNAMIC_MAX_STALE_MINUTES = 60

# Dynamic classification thresholds (lenient for delayed-SIP feed).
PREMARKET_DYNAMIC_DEFAULT_LIMIT_PCT = 0.05
PREMARKET_DYNAMIC_SPARSE_HIGH_RETURN_LIMIT_PCT = 0.10
PREMARKET_DYNAMIC_VERY_HIGH_RETURN_NO_CAP_PCT = 0.10
PREMARKET_DYNAMIC_HIGH_RETURN_NO_CAP_PCT = 0.05
PREMARKET_DYNAMIC_MODERATE_RETURN_PCT = 0.02
