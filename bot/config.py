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
    HEAD_PCT, TAIL_PCT, MAX_HEAD_POSITIONS, MAX_TOTAL_POSITIONS, MAX_POSITIONS,
    HARD_STOP_PCT, EXIT_UP_MOVE_PCT, V2_FAILSAFE_TIME,
    DATA_COLLECTION_TIME, SCORING_TIME, ENTRY_TIME,
    MARKET_OPEN_TIME, V2_CLASSIFY_TIME, EXIT_BUCKET_1130_TIME,
    SECTOR_ETFS, MARKET_BENCHMARK,
)

