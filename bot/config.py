"""
Overnight Momentum Bot — Unified config re-export.

All settings live in their dedicated files:
  config_broker.py   — API credentials, endpoints, data feed
  config_runtime.py  — state paths, logging
  config_universe.py — price/ADV filters, presets, lookback periods
  config_strategy.py — scoring weights, selection tiers, exit rules, timing

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
    SCORE_WEIGHT_INTRADAY_RETURN, SCORE_WEIGHT_PROXIMITY_HIGH,
    SCORE_WEIGHT_VOLUME_VS_AVG, SCORE_WEIGHT_VOLUME_TREND,
    SCORE_WEIGHT_VS_MARKET, SCORE_WEIGHT_ATR_PCT,
    STRATEGY_TIERS,
    MAX_LEVERAGE, ADV_CAP_PCT, MAX_POSITION_DOLLARS, MIN_SHARES,
    HEAD_COUNT, TAIL_MAX_POSITIONS, MAX_HEAD_POSITIONS, MAX_TOTAL_POSITIONS,
    MR_MIN_PRICE, MR_MAX_PRICE, MR_DAY_RET_MAX, MR_VOLUME_RATIO_MIN,
    MR_CLOSE_POSITION_MAX, MR_LATE_DROP_MAX, MR_MAX_POSITIONS, MR_USE_RANDOM_SELECTION,
    V2_FAILSAFE_TIME,
    ENTRY_BP_BUFFER_PCT, ENTRY_MIN_DEPLOY_PCT, ENTRY_MOPUP_MAX_POSITIONS,
    DATA_COLLECTION_TIME, SCORING_TIME, ENTRY_TIME,
    MARKET_OPEN_TIME, EXIT_940_TIME,
    SECTOR_ETFS, MARKET_BENCHMARK,
)

