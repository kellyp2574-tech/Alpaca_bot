"""Strategy configuration — 3-sleeve optimized portfolio.

Intraday ETF sleeve (priority 1, one trade per day):
    5 strategies checked in order at 10:00 AM (1-3) and 10:10 AM (4-5):
    1. VXX Spike Recovery  — VXX 30min >= +2.5%, QQQ range 0.3-0.8%  → BUY TQQQ, exit 15:30
    2. VXX Collapse        — VXX 30min <= -2.0%, QQQ 30min >= -1.0%   → BUY TQQQ, exit 15:30
    3. Momentum Sleeve     — QQQ 30min >= +0.5%                         → BUY TQQQ, exit 15:00 (TP+2%, SL-1%)
    4. Router Long         — QQQ-SPY 40min spread >= +0.2%              → BUY TQQQ at 10:10, exit 15:00 (TP+3%, SL-1%)
    5. SVIX Long           — SVIX 40min >= +0.2%                        → BUY SVIX at 10:10, exit 15:00 (TP+3%, no SL)

Overnight ETF sleeve (priority 2, only if NO intraday trade today):
    Checked at 15:45 in order:
    A. VXX Mean Reversion  — VXX day return >= +2.5%                    → BUY SVIX overnight, sell 09:30
    B. Overnight Quality   — SPY day > +0.5% OR VXX day < -2.0%,
                             AND VXX day < +2.0%                        → BUY TQQQ overnight, sell 09:30
    C. Gap Bounce          — QQQ day < -0.5%, morning down, VXX down    → BUY TQQQ overnight, sell 09:30

Overnight single-stock MR sleeve (priority 3, fallback):
    Only runs at 15:45 if NO intraday trade AND no overnight ETF trade fired.
    Filters: entry price $1-$2, return <= -4%, close-location <= 0.25, ADV >= $1M
    Max 3 positions, 30% equity each.

Live constants (single source of truth):
    INTRADAY_ETF_ALLOCATION_PCT  = 0.90    # 90% of equity for the intraday ETF sleeve
    MR_ALLOC_PER_POSITION_PCT    = 0.30    # 30% of equity per MR position
    MR_MAX_PRIMARY_POSITIONS     = 3       # Top 3 MR candidates only
    MR_MAX_TOTAL_ALLOCATION_PCT  = 0.90    # Max 90% of equity in the MR sleeve
    MR_ADV_CAP_PCT               = 0.003   # 0.3% of 20-day ADV per symbol
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
OVERNIGHT_ETF_ENABLED = True   # A/B/C overnight ETF strategies
MR_OVERNIGHT_ENABLED = True    # Single-stock MR fallback

# ═══════════════════════════════════════════════════
# Capital allocation — LIVE CONSTANTS
# ═══════════════════════════════════════════════════
# 90% of equity for intraday ETF sleeve (only one strategy fills per day)
INTRADAY_ETF_ALLOCATION_PCT = 0.90

# MR sleeve sizing
MR_ALLOC_PER_POSITION_PCT = 0.30   # 30% of equity per MR position
MR_MAX_PRIMARY_POSITIONS = 3       # Top 3 MR candidates only
MR_MAX_TOTAL_ALLOCATION_PCT = 0.90 # Max 90% of equity across the MR sleeve
MR_ADV_CAP_PCT = 0.003             # 0.3% of 20-day ADV per symbol

# ═══════════════════════════════════════════════════
# Intraday ETF Router Configuration (9:30-10:10 AM)
# ═══════════════════════════════════════════════════
# ETF symbols for tape measurement (9:30 opens + 10:00/10:10 snapshots)
ETF_ROUTER_SYMBOLS = ["QQQ", "SPY", "VXX", "SVIX", "TQQQ"]

# Tape recording cadence — once every N seconds.
ETF_TAPE_UPDATE_INTERVAL_SECONDS = 5

# ── Strategy 1: VXX Spike Recovery (check at 10:00) ──
VXX_SPIKE_MIN_RETURN_PCT = 2.5       # VXX 30min return >= +2.5%
VXX_SPIKE_QQQ_RANGE_MIN_PCT = 0.3    # QQQ 9:30-10:00 range >= 0.3%
VXX_SPIKE_QQQ_RANGE_MAX_PCT = 0.8    # QQQ 9:30-10:00 range <= 0.8%
VXX_SPIKE_VEHICLE = "TQQQ"
VXX_SPIKE_EXIT_TIME = "15:30"

# ── Strategy 2: VXX Collapse (check at 10:00) ──
VXX_COLLAPSE_VXX_MAX_RETURN_PCT = -2.0   # VXX 30min <= -2.0%
VXX_COLLAPSE_QQQ_MIN_RETURN_PCT = -1.0   # QQQ 30min >= -1.0%
VXX_COLLAPSE_VEHICLE = "TQQQ"
VXX_COLLAPSE_EXIT_TIME = "15:30"

# ── Strategy 3: Momentum Sleeve (check at 10:00) ──
MOMENTUM_QQQ_MIN_RETURN_PCT = 0.5    # QQQ 30min return >= +0.5%
MOMENTUM_VEHICLE = "TQQQ"
MOMENTUM_EXIT_TIME = "15:00"
MOMENTUM_TAKE_PROFIT_PCT = 0.02      # +2% TP
MOMENTUM_STOP_LOSS_PCT = 0.01        # -1% SL

# ── Strategy 4: Router Long (check at 10:10) ──
ROUTER_LONG_SPREAD_MIN_PCT = 0.2     # QQQ-SPY 40min spread >= +0.2%
ROUTER_LONG_VEHICLE = "TQQQ"
ROUTER_LONG_EXIT_TIME = "15:00"
ROUTER_LONG_TAKE_PROFIT_PCT = 0.03   # +3% TP
ROUTER_LONG_STOP_LOSS_PCT = 0.01     # -1% SL

# ── Strategy 5: SVIX Long (check at 10:10) ──
SVIX_LONG_MIN_RETURN_PCT = 0.2       # SVIX 40min return >= +0.2%
SVIX_LONG_VEHICLE = "SVIX"
SVIX_LONG_EXIT_TIME = "15:00"
SVIX_LONG_TAKE_PROFIT_PCT = 0.03     # +3% TP
# No stop loss for SVIX Long

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
# Staleness threshold for intraday ETF entries (IEX can be 10-30s behind SIP).
# A 12-second-old quote is normal — do not reject liquid ETFs on IEX latency.
ETF_ENTRY_MAX_STALE_SECONDS = 60.0
# Log a warning when quote age exceeds this but still allow entry.
ETF_ENTRY_WARN_STALE_SECONDS = 30.0

# ═══════════════════════════════════════════════════
# Overnight ETF Strategies (A/B/C — at 15:45 if no intraday trade)
# ═══════════════════════════════════════════════════
# These are checked at 15:45 in priority order (A > B > C).
# If any fires, the single-stock MR sleeve does NOT run that day.

# ── Strategy A: VXX Mean Reversion → BUY SVIX ──
OVERNIGHT_VXX_MR_TRIGGER_PCT = 2.5      # VXX day return >= +2.5%
OVERNIGHT_VXX_MR_VEHICLE = "SVIX"

# ── Strategy B: Overnight Quality → BUY TQQQ ──
OVERNIGHT_QUALITY_SPY_MIN_PCT = 0.5     # SPY day return > +0.5%  (OR condition)
OVERNIGHT_QUALITY_VXX_COLLAPSE_PCT = -2.0  # VXX day return < -2.0%  (OR condition)
OVERNIGHT_QUALITY_VXX_EXCLUSION_PCT = 2.0  # VXX day return must be < +2.0% (AND exclusion)
OVERNIGHT_QUALITY_VEHICLE = "TQQQ"

# ── Strategy C: Gap Bounce → BUY TQQQ ──
OVERNIGHT_GAP_BOUNCE_QQQ_MAX_PCT = -0.5    # QQQ day return < -0.5% (down day)
# Also requires: QQQ 9:30-10:00 return < 0% AND VXX day return < 0%
OVERNIGHT_GAP_BOUNCE_VEHICLE = "TQQQ"

# All overnight ETF positions are sold at 09:30 AM the next day
OVERNIGHT_ETF_EXIT_TIME = "09:30"

# ═══════════════════════════════════════════════════
# Per-strategy Stop-Loss / Take-Profit (bracket orders)
# ═══════════════════════════════════════════════════
# All values are fractions of fill price (e.g. 0.02 = 2%).
# None = that leg is omitted.
#
# Strategy              SL       TP     Notes
# VXX Spike Recovery   None     None   No SL/TP — hard time exit at 15:30
# VXX Collapse         None     None   No SL/TP — hard time exit at 15:30
# Momentum Sleeve      -1%      +2%    Time exit 15:00 is fallback
# Router Long          -1%      +3%    Time exit 15:00 is fallback
# SVIX Long            None     +3%    No SL per strategy rules
ETF_SL_TP: dict = {
    "VXX_SPIKE_RECOVERY": {"sl": None,  "tp": None},
    "VXX_COLLAPSE":       {"sl": None,  "tp": None},
    "MOMENTUM_SLEEVE":    {"sl": 0.01,  "tp": 0.02},
    "ROUTER_LONG":        {"sl": 0.01,  "tp": 0.03},
    "SVIX_LONG":          {"sl": None,  "tp": 0.03},
}

# ═══════════════════════════════════════════════════
# Sleeve: Single-stock Mean Reversion (fallback overnight)
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
MORNING_EXIT_TIME = "09:30"          # All overnight positions sold at open
MORNING_FAILSAFE_TIME = "09:45"      # Post-exit failsafe verification

# Cancel any resting orders before the 09:30 market exit logic.
MORNING_CANCEL_OPEN_ORDERS_TIME = "09:25"

# ═══════════════════════════════════════════════════
# Unified System Timeline (09:00 - 16:00)
# ═══════════════════════════════════════════════════
# 09:00 - Bot startup, load state, verify calendar
# 09:25 - Cancel any open orders, prepare for open
# 09:30 - Sell any overnight positions (SVIX, TQQQ, or single-stock MR)
# 09:30 - Begin tape recording (9:30 opens)
# 09:31 - Broker rescue pass for remaining positions
# 09:45 - Post-exit failsafe
# 10:00 - Intraday check: strategies 1-3 (VXX Spike, VXX Collapse, Momentum)
# 10:10 - Intraday check: strategies 4-5 (Router Long, SVIX Long)
# 15:00 - Time exit for strategies 3/4/5
# 15:30 - Time exit for strategies 1/2 + hard flatten
# 15:45 - Overnight decision (ETF A/B/C → then single-stock MR fallback)
# 16:00 - Market close, final reconciliation

BOT_START_TIME = "09:00"
ROUTER_DECISION_TIME = "10:00"       # Strategies 1-3 evaluated here
ROUTER_1010_TIME = "10:10"           # Strategies 4-5 evaluated here

# Intraday ETF exit checkpoints
INTRADAY_EXIT_1500 = "15:00"         # Momentum, Router Long, SVIX Long
INTRADAY_ETF_HARD_FLATTEN_TIME = "15:30"  # VXX Spike, VXX Collapse + hard flatten
