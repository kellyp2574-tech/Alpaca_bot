"""Strategy configuration — 3-sleeve optimized portfolio.

Intraday ETF sleeve (priority 1, one trade per day):
    5 strategies checked in order at 10:00 AM (1-3) and 10:10 AM (4-5):
    1. VXX Spike Recovery  — VXX 30min >= +2.5%, QQQ range 0.3-0.8%  → BUY TQQQ, exit 15:30
    2. VXX Collapse        — VXX 30min <= -2.0%, QQQ 30min >= -1.0%   → BUY TQQQ, exit 15:30
    3. Momentum Sleeve     — QQQ 30min >= +0.5%                         → BUY TQQQ, exit 15:00 (TP+2%, SL-1%)
    4. Router Long         — QQQ-SPY 40min spread >= +0.2%              → BUY TQQQ at 10:10, exit 15:00 (TP+3%, SL-1%)
    5. SVIX Long           — SVIX 40min >= +0.2%                        → BUY SVIX at 10:10, exit 15:00 (TP+3%, no SL)

Overnight sleeve (at 15:45 — runs every day, independent of intraday trades):
    - Single-stock MR ALWAYS runs (top 3 candidates).
    - Conditional TQQQ is added ON TOP when favorable (see
      overnight_etf_runner_conditional.py). It does NOT block MR; instead MR
      capacity is reduced so combined exposure stays within
      OVERNIGHT_COMBINED_MAX_ALLOCATION_PCT (default 90%).
    Conditional TQQQ fires when both MR and TQQQ signals are positive, OR when the
    TQQQ expected return exceeds TQQQ_STRONG_RETURN_THRESHOLD.

Overnight single-stock MR sleeve:
    Filters: entry price $1-$2, return <= -4%, close-location <= 0.25, ADV >= $1M
    Max 3 positions, 30% equity each.

Live constants (single source of truth):
    INTRADAY_ETF_ALLOCATION_PCT  = 1.00    # Max router allocation = up to 100% of equity minus morning-MR deployment
    MR_ALLOC_PER_POSITION_PCT    = 0.30    # 30% of equity per MR position
    MR_MAX_PRIMARY_POSITIONS     = 3       # Top 3 MR candidates only
    MR_MAX_TOTAL_ALLOCATION_PCT  = 0.60    # Base 60% of equity for single-stock MR
    TQQQ_CONDITIONAL_ALLOCATION_PCT = 0.30 # 30% reserved for conditional TQQQ
    OVERNIGHT_COMBINED_MAX_ALLOCATION_PCT = 0.90  # Max 90% combined MR+TQQQ
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
# ═══════════════════════════════════════════════════
# Intraday ETF Router Dynamic Allocation (10:00/10:10 AM)
# ═══════════════════════════════════════════════════
# Router uses remaining capacity after morning MR deployment, capped by buying power.
# INTRADAY_ETF_ALLOCATION_PCT now serves as the MAXIMUM router allocation when
# morning MR has deployed nothing (i.e., router can use up to 100% of equity).
# When morning MR has deployed capital, router uses remaining equity - deployed.
INTRADAY_ETF_ALLOCATION_PCT = 1.00   # Maximum 100% of equity (was 0.50)

# Buying power buffer for ETF entries (default 98% to leave room for price movement)
ETF_ENTRY_BP_BUFFER_PCT = 0.98

# ═══════════════════════════════════════════════════
# Overnight Single-Stock MR Sizing (15:45 entry)
# ═══════════════════════════════════════════════════
# Base allocation for single-stock MR (when no conditional TQQQ fires): 60%.
# When overnight TQQQ fires, the remaining MR budget is combined_max - TQQQ
# allocation, capped at 60%. The combined overnight sleeve is capped at 90%.
MR_ALLOC_PER_POSITION_PCT = 0.30   # 30% of equity per MR position
MR_MAX_PRIMARY_POSITIONS = 3       # Top 3 MR candidates only
MR_MAX_TOTAL_ALLOCATION_PCT = 0.60 # Base 60% of equity for single-stock MR
MR_ADV_CAP_PCT = 0.003             # 0.3% of 20-day ADV per symbol

# Conditional TQQQ allocation (overnight, 15:45)
# When TQQQ fires, MR cap is reduced to leave room for TQQQ position
TQQQ_CONDITIONAL_ALLOCATION_PCT = 0.30  # 30% when TQQQ signal is positive
OVERNIGHT_COMBINED_MAX_ALLOCATION_PCT = 0.90  # Max 90% combined MR+TQQQ

# ═══════════════════════════════════════════════════
# Intraday ETF Router Configuration (9:30-10:10 AM)
# ═══════════════════════════════════════════════════
# ETF symbols for tape measurement (9:30 opens + 10:00/10:10 snapshots)
ETF_ROUTER_SYMBOLS = ["QQQ", "SPY", "VXX", "SVIX", "TQQQ", "SQQQ"]

# Tape recording cadence — once every N seconds.
# Reduced to 10s to avoid operating near the 60/60 API rate limit during
# the 9:30-10:10 execution window, when order/account calls are also needed.
ETF_TAPE_UPDATE_INTERVAL_SECONDS = 10

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
MOMENTUM_VEHICLE = "TQQQ"            # Normal regime: long TQQQ
MOMENTUM_ANTI_VEHICLE = "SQQQ"       # HIGH_RISK regime: short via SQQQ (anti-momentum)
MOMENTUM_EXIT_TIME = "15:00"
MOMENTUM_TAKE_PROFIT_PCT = 0.02      # +2% TP
MOMENTUM_STOP_LOSS_PCT = 0.01        # -1% SL

# ── VXX Regime Classification (used by Momentum + Overnight filtering) ──
# HIGH_RISK if VXX 30min return >= this threshold OR VXX price >= price threshold.
# In HIGH_RISK: Momentum fires SQQQ (anti-momentum), Overnight TQQQ skipped.
VXX_HIGH_RISK_RETURN_PCT = 2.0       # VXX 30min return >= +2.0% → HIGH_RISK
VXX_HIGH_RISK_PRICE = 400.0          # VXX price >= $400 → HIGH_RISK (absolute level)

# ── Strategy 4: Router Long (check at 10:10) ──
ROUTER_LONG_SPREAD_MIN_PCT = 0.2     # QQQ-SPY 40min spread >= +0.2%
ROUTER_LONG_VEHICLE = "TQQQ"
# Operative exit is ETF_SL_TP["ROUTER_LONG"]["exit_time"] (15:30); this value only
# feeds RouterDecision.exit_time used for audit logging. Kept in sync to avoid drift.
ROUTER_LONG_EXIT_TIME = "15:30"
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
# Staleness threshold for intraday ETF entries (IEX can be 30-90s behind SIP).
# Observed: SVIX quote 86s old at 10:10 was rejected at 60s — raised to 120s.
ETF_ENTRY_MAX_STALE_SECONDS = 120.0
# Log a warning when quote age exceeds this but still allow entry.
ETF_ENTRY_WARN_STALE_SECONDS = 45.0

# ═══════════════════════════════════════════════════
# Overnight ETF positions are sold at 09:30 AM the next day
# ═══════════════════════════════════════════════════
OVERNIGHT_ETF_EXIT_TIME = "09:30"

# ═══════════════════════════════════════════════════
# Per-strategy Stop-Loss / Take-Profit (timed exits, SL armed later)
# ═══════════════════════════════════════════════════
# SL/TP values are fractions of fill price (e.g. 0.005 = 0.5%).
# SL_ARM_TIME: When to submit the stop-loss order (not a bracket at entry).
# EXIT_TIME: Hard timed exit regardless of PnL.
#
# Strategy              SL       SL_ARM_TIME   EXIT_TIME   Notes
# Momentum_Sleeve      0.5%     13:00         15:00       Cut losers at 1pm, ride winners
# Router_Long          0.5%     13:30         15:30       Cut losers at 1:30pm, late exit
# SVIX_Long            None     None          15:00       15:00 timed exit only (no 13:30 flat/red gate)
# VXX_Collapse         1.0%     13:00         15:30       Catastrophe stop, ride vol crush
# VXX_Spike_Recovery   None     None          15:30       No SL — hard time exit only
ETF_SL_TP: dict = {
    # blocks_overnight_etf   = prevents 15:45 overnight ETF sleeve from firing
    # blocks_single_stock_mr = single-stock overnight MR is always independent; all branches False
    "VXX_SPIKE_RECOVERY":       {"sl": None,  "sl_arm_time": None,    "exit_time": "15:30", "blocks_overnight_etf": True,  "blocks_single_stock_mr": False},
    "VXX_COLLAPSE":             {"sl": 0.01,   "sl_arm_time": "13:00", "exit_time": "15:30", "blocks_overnight_etf": True,  "blocks_single_stock_mr": False},
    "MOMENTUM_SLEEVE":          {"sl": 0.005,  "sl_arm_time": "13:00", "exit_time": "15:00", "blocks_overnight_etf": True,  "blocks_single_stock_mr": False},
    "MOMENTUM_SLEEVE_ANTI":     {"sl": 0.005,  "sl_arm_time": "13:00", "exit_time": "15:00", "blocks_overnight_etf": False, "blocks_single_stock_mr": False},
    "ROUTER_LONG":              {"sl": 0.005,  "sl_arm_time": "13:30", "exit_time": "15:30", "blocks_overnight_etf": True,  "blocks_single_stock_mr": False},
    "SVIX_LONG":                {"sl": None,    "sl_arm_time": None,   "exit_time": "15:00", "blocks_overnight_etf": False, "blocks_single_stock_mr": False},
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
MR_LATE_DROP_MIN = -0.20  # Reject stocks with late drop worse than -20% (falling knives)
MR_MIN_AVG_DOLLAR_VOLUME = 1_000_000

# Finalist rank is close_location ascending, top 3.
MR_RANK_BY_CLOSE_LOCATION_ONLY = True
MR_MIN_CANDIDATES = 2

# ═══════════════════════════════════════════════════
# MR free-data pipeline (replaces paid Massive live data)
# ═══════════════════════════════════════════════════
USE_FREE_MR_PIPELINE = True   # ENABLED for paper test: Massive free + Alpaca snapshots

# Stage 1: Massive previous-day grouped daily filters (broad watchlist)
MR_FREE_PREV_MIN_PRICE = 0.75
MR_FREE_PREV_MAX_PRICE = 3.00
MR_FREE_PREV_MIN_DOLLAR_VOLUME = 500_000

# Prior-day return filter (T-2 to T-1) - morning pre-filter for overnight MR
# Based on backtest: stocks with prior-day returns in [-20%, +5%] perform best
# (0-5% bucket: +0.60% avg, 5-10% bucket: -0.66% avg, so boundary is +5%)
MR_FREE_PRIOR_RET_MIN = -0.20   # Reject stocks that crashed >20% prior day
MR_FREE_PRIOR_RET_MAX = 0.05    # Reject stocks that surged >5% prior day

# If True, fail closed when T-2 data is unavailable (return empty watchlist)
# If False, proceed without prior_ret filtering when T-2 data is missing
MR_FREE_PRIOR_RET_REQUIRE_DATA = False

# Stage 2: Alpaca live snapshot scan settings
MR_FREE_ALPACA_BATCH_SIZE = 200
MR_FREE_ALPACA_BATCH_SLEEP_SECONDS = 0.25

# ═══════════════════════════════════════════════════
# MR regime sizing
# ═══════════════════════════════════════════════════
ENABLE_MR_ETF_REGIME_SIZING = True
MR_ETF_REGIME_SYMBOLS = ["SPY", "IWM", "QQQ"]
MR_ETF_NEGATIVE_SIZE_MULT = 1.0
MR_ETF_POSITIVE_SIZE_MULT = 0.75

# Daily-loss circuit breaker (vs yesterday's close equity).
# 0.05 = abort 15:45 entries AND 10:00 ETF entry if today's PnL < -5%.
DAILY_LOSS_LIMIT_PCT = 0.05

# ═══════════════════════════════════════════════════
# Sleeve: Intraday Mean Reversion (Morning Momentum)
# ═══════════════════════════════════════════════════
# Gap-reversal longs, entered 9:32-9:47, flat same day.
# Validated: 26.4x equity, Sharpe 2.39, 641/1263 active days.
# Capital: 50% of equity split equally across candidates.
# Router exit rule at 10:00: if SHORT → keep Theme A only, exit B/C/D/UL.

INTRADAY_MR_ENABLED = True             # Paper testing enabled.

# Regime classification (VIX >= 15 validated vs VIX >= 20 original)
INTRADAY_MR_VIX_THRESHOLD = 15.0      # Active day if VIX >= this
INTRADAY_MR_GAP_THRESHOLD = 0.01      # Active day if |SPY gap| > 1% AND |QQQ gap| > 1%

# Candidate caps (validated min=1, max=8)
INTRADAY_MR_MIN_CANDIDATES = 1
INTRADAY_MR_MAX_CANDIDATES = 8

# ADV filter — previous day dollar volume minimum
INTRADAY_MR_MIN_ADV_DOLLARS = 1_000_000

# Capital allocation: 50% of equity split equally across all candidates
INTRADAY_MR_BUDGET_PCT = 0.50

# Universe pre-filter for pre-market snapshot fetch
# Price range intentionally wider than Theme bins to allow all themes
INTRADAY_MR_UNIVERSE_MIN_PRICE = 2.00
INTRADAY_MR_UNIVERSE_MAX_PRICE = 100.0

# Two-stage build timing
INTRADAY_MR_STAGE1_TIME      = "09:00"  # Universe + T-1/T-2 bar cache
INTRADAY_MR_STAGE2_TIME      = "09:30"  # Official opens + VIX + finalize candidates

# Maximum seconds past a candidate's scheduled entry_time before skipping it.
# Extended to 15 minutes (900s) to accommodate Alpaca paper delays.
# Backtest used exact entry times; entries delayed beyond this are stale.
INTRADAY_MR_MAX_ENTRY_DELAY_S = 900

# Entry cutoff time — no new entries after this time regardless of delay
INTRADAY_MR_ENTRY_CUTOFF = "10:00"

# Minimum remaining hold time in seconds required to enter a position
# Prevents entering positions too close to exit time
INTRADAY_MR_MIN_REMAINING_HOLD_S = 120

# Hard flatten time (failsafe — reconciles against broker positions)
# MUST be before 15:45 overnight entry time to avoid position conflicts
INTRADAY_MR_HARD_FLATTEN_TIME = "15:40"

# ── Allocation note ──────────────────────────────────────────────────────────
# On overlap days (MR + ETF router both active):
#   INTRADAY_MR_BUDGET_PCT      = 0.50  (MR buys up to 50% equity pre-10:00)
#   INTRADAY_ETF_ALLOCATION_PCT = 1.00  (router maximum = 100% of equity,
#                                        less actual morning-MR deployment)
# Combined max = 100% equity (no double-spend).
# On non-router days: at 10:10 the router bucket is reallocated to open MR
# positions as add-on buys (see INTRADAY_REALLOC_* settings below).
# ─────────────────────────────────────────────────────────────────────────────
# ── Conditional TQQQ Overnight Strategy ─────────────────────────────────────
# Replaces A/B/C priority system with conditional approach
# Individual MR always runs (60% base), TQQQ added conditionally (30% max)

INDIVIDUAL_MR_SIGNAL_THRESHOLD = 0.5      # Minimum average score for individual MR signal
INDIVIDUAL_MR_MIN_CANDIDATES = 1          # Minimum candidates for positive signal

TQQQ_VIX_LOW_THRESHOLD = 15.0            # VIX below this = trending (good for TQQQ)
TQQQ_VIX_HIGH_THRESHOLD = 25.0           # VIX above this = risk-off (bad for TQQQ)
TQQQ_STRONG_RETURN_THRESHOLD = 0.015     # >1.5% expected return triggers TQQQ regardless

# ─────────────────────────────────────────────────────────────────────────────

# ── Router-budget reallocation to MR (10:10 add-on) ─────────────────────────
# Fires once at ~10:10 when the router did NOT enter a position.
# The unused router bucket (50% equity) is redistributed to open MR winners.
# Weighting: equal weight among positions with ret >= 0 at 10:10.
# Disable here without touching the rest of the sleeve.
INTRADAY_REALLOC_ENABLED      = True
INTRADAY_REALLOC_TIME         = "10:10"   # window start
INTRADAY_REALLOC_CUTOFF       = "10:11"   # abandon after this (no more retries)
INTRADAY_REALLOC_MODE         = "equal_positive"   # equal weight among ret>=0 winners
INTRADAY_REALLOC_MAX_PCT      = 0.50      # max fraction of equity to redeploy
INTRADAY_REALLOC_BP_BUFFER    = 0.98      # apply to buying_power before sizing

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
