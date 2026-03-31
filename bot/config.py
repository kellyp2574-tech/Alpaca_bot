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
MAX_POSITIONS = 100  # Maximum positions to hold at once
MAX_POSITION_DOLLARS = 50_000  # Maximum dollars per position (absolute cap)
MAX_POSITION_SHARES = 50_000  # Maximum shares per position (absolute cap)

# Exit rules (VIX-conditioned)
VIX_LOW_THRESHOLD = 12.0
VIX_HIGH_THRESHOLD = 22.0
EXIT_TIME_LOW_VIX = "14:30"  # 2:30 PM
EXIT_TIME_MIDDLE_VIX = "15:30"  # 3:30 PM (VIX 12-22)
EXIT_TIME_HIGH_VIX = "15:30"  # 3:30 PM

# Trailing stop (for middle VIX regime: 12-22)
TRAILING_STOP_ACTIVATION = 0.15  # 15% gain to activate
TRAILING_STOP_PCT = 0.03  # 3% trail

# API endpoints
ALPACA_BASE_URL = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

# Massive API (for universe reduction)
MASSIVE_BASE_URL = "https://api.massive.com"
UNIVERSE_MAX_RETRIES = 3  # Max retries for Massive universe building before Alpaca fallback

# Data feed (IEX for free tier)
DATA_FEED = "iex"

# ═══════════════════════════════════════════════════
# Staged Entry Execution Model (MOO + Post-Open Rescue)
# ═══════════════════════════════════════════════════

# Enable staged entry: submit partial MOO before open, aggressive fill remainder after
USE_STAGED_OPEN_ENTRY = True

# Percent of target size sent as MOO before the open (0.25 = 25%)
MOO_ENTRY_PCT = 0.25

# Timing for post-open rescue passes
POST_OPEN_ENTRY_TIME_1 = "09:30:10"
POST_OPEN_ENTRY_TIME_2 = "09:30:30"

# Aggressive marketable limit buffer for buy orders (0.005 = 50 bps)
POST_OPEN_BUY_LIMIT_BUFFER = 0.005

# Optionally avoid chasing if price runs too far from expected open (0.03 = 3%)
MAX_CHASE_FROM_OPEN_PCT = 0.03

# Skip tiny leftovers
MIN_RESCUE_NOTIONAL = 100.0
MIN_RESCUE_SHARES = 1

