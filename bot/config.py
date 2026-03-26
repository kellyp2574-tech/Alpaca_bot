"""
Gap Momentum Bot Configuration — API keys, state paths, logging, strategy parameters.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════
# Alpaca API
# ═══════════════════════════════════════════════════
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# ═══════════════════════════════════════════════════
# Massive API (for universe reduction)
# ═══════════════════════════════════════════════════
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY", "")

# ═══════════════════════════════════════════════════
# State & Logging
# ═══════════════════════════════════════════════════
STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state")
STATE_FILE = os.path.join(STATE_DIR, "bot_state.json")
LOG_DIR = os.path.join(STATE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "bot.log")
TRADE_LOG_FILE = os.path.join(LOG_DIR, "trades.log")

# Logging config
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# State files
POSITIONS_FILE = os.path.join(STATE_DIR, "positions.json")
DAILY_LOG_FILE = os.path.join(STATE_DIR, "daily_log.json")

# ═══════════════════════════════════════════════════
# Gap Momentum Strategy Parameters
# ═══════════════════════════════════════════════════

# Trading hours (Eastern Time)
START_TIME = "09:00"
SNAPSHOT_TIME = "09:29"
ENTRY_TIME = "09:30"

# Price filters (Step 1: Universe reduction via Massive)
MIN_PRICE = 0.50
MAX_PRICE = 5.00

# Gap filters (Step 2: Signal engine via Alpaca)
MIN_GAP_PCT = 3.0
MAX_GAP_PCT = 50.0

# Volume filter
MIN_ADV_DOLLARS = 5_000_000  # $5M minimum average daily dollar volume

# Position sizing
LIQUIDITY_CAP_PCT = 0.003  # 0.3% of ADV max position size
MAX_POSITIONS = 12  # Maximum positions to hold at once
POSITION_SIZE_DOLLARS = 10000  # Base position size per trade

# Price bucket position sizing multipliers
PRICE_BUCKET_LOW_MAX = 1.00      # Upper bound of low price bucket ($0.50-$1.00)
PRICE_BUCKET_MID_MAX = 2.00      # Upper bound of mid price bucket ($1.00-$2.00)
PRICE_BUCKET_LOW_MULTIPLIER = 0.5   # 0.5x for $0.50-$1.00
PRICE_BUCKET_MID_MULTIPLIER = 1.0     # 1.0x baseline for $1.00-$2.00
PRICE_BUCKET_HIGH_MULTIPLIER = 1.2  # 1.2x for $2.00-$5.00
PRICE_BUCKET_LOW_EQUITY_CAP = 0.12    # 12% max equity for ALL low price bucket positions combined
VIX_LOW_THRESHOLD = 12.0
VIX_HIGH_THRESHOLD = 22.0
EXIT_TIME_LOW_VIX = "14:30"  # 2:30 PM
EXIT_TIME_MIDDLE_VIX = "15:30"  # 3:30 PM (VIX 12-22)
EXIT_TIME_HIGH_VIX = "15:30"  # 3:30 PM

# Exit rules (VIX-conditioned)
# Trailing stop (for middle VIX regime: 12-22)
TRAILING_STOP_ACTIVATION = 0.15  # 15% gain to activate
TRAILING_STOP_PCT = 0.03  # 3% trail

# API endpoints
ALPACA_BASE_URL = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

# Massive API (for universe reduction)
MASSIVE_BASE_URL = "https://api.massive.com"

# Data feed (IEX for free tier)
DATA_FEED = "iex"

