"""
Combined Overnight Rebound Bot — Unified config re-export.

Sleeve 1: MR_WIDE (Mean Reversion)
  - Buy 15:55, $1–5, day_ret <= -3%, vol_ratio >= 1.5x
  - Exit 09:40

Sleeve 2: GDP_BASE (Green-Day Pullback)
  - Buy 15:55, $1–10, day_ret +1% to +10%, below VWAP
  - Exit 09:35

All settings live in their dedicated files:
  config_broker.py   — API credentials, endpoints, data feed
  config_runtime.py  — state paths, logging
  config_universe.py — price/ADV filters, presets, lookback periods
  config_strategy.py — sleeve configs, exit rules, timing

This file re-exports everything so that existing code using
``from bot import config; config.X`` continues to work.
"""

# Broker
from bot.config_broker import (          # noqa: F401
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER,
    ALPACA_BASE_URL, ALPACA_DATA_URL, DATA_FEED,
    MASSIVE_API_KEY, MASSIVE_BASE_URL,
)

# Runtime
from bot.config_runtime import (         # noqa: F401
    STATE_DIR, STATE_FILE, LOG_DIR, LOG_FILE, TRADE_LOG_FILE,
    LOG_LEVEL, LOG_FORMAT,
    POSITIONS_FILE, DAILY_LOG_FILE,
)

# Universe
from bot.config_universe import (        # noqa: F401
    UNIVERSE_PRESET, UNIVERSE_PRESETS,
    MIN_PRICE, MAX_PRICE, MIN_ADV_DOLLARS,
    ADV_LOOKBACK_DAYS, ATR_LOOKBACK_DAYS,
    UNIVERSE_MAX_RETRIES,
)

# Strategy
from bot.config_strategy import (        # noqa: F401
    # Combined sleeve mode
    ENABLE_COMBINED_SLEEVES,
    MR_ALLOCATION_PCT, GDP_ALLOCATION_PCT, COMBINED_MAX_POSITIONS,
    # Position sizing
    MAX_LEVERAGE, ADV_CAP_PCT, MAX_POSITION_DOLLARS, MIN_SHARES,
    # MR sleeve
    MR_MIN_PRICE, MR_MAX_PRICE, MR_DAY_RET_MAX, MR_VOLUME_RATIO_MIN,
    MR_CLOSE_POSITION_MAX, MR_LATE_DROP_MAX, MR_MAX_POSITIONS, MR_USE_RANDOM_SELECTION,
    # GDP sleeve
    GDP_MIN_PRICE, GDP_MAX_PRICE, GDP_DAY_RET_MIN, GDP_DAY_RET_MAX,
    GDP_REQUIRE_BELOW_VWAP, GDP_LATE_MOM_MAX, GDP_MAX_CLOSE_POSITION, GDP_MAX_POSITIONS,
    # Execution safety
    ENTRY_BP_BUFFER_PCT, ENTRY_MIN_DEPLOY_PCT, ENTRY_MOPUP_MAX_POSITIONS,
    # Timing
    DATA_COLLECTION_TIME, SCORING_TIME, ENTRY_TIME,
    MARKET_OPEN_TIME, GDP_EXIT_TIME, MR_EXIT_TIME, V2_FAILSAFE_TIME,
    # Benchmarks
    SECTOR_ETFS, MARKET_BENCHMARK,
)

