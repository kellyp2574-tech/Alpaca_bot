"""
Bot Configuration — All strategy parameters and settings.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════
# Alpaca API
# ═══════════════════════════════════════════════════
ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# ═══════════════════════════════════════════════════
# Strategy A: MA Crossover
# ═══════════════════════════════════════════════════
MA_SIGNAL_GROWTH = "QQQ"       # 1x signal ticker for growth
MA_SIGNAL_SAFE = "TLT"        # 1x signal ticker for safety
MA_TRADE_GROWTH = "QLD"        # 2x leveraged growth ETF
MA_TRADE_SAFE = "UBT"         # 2x leveraged bond ETF
MA_TRADE_ALT = "DBMF"         # fallback when both signals fail
MA_PERIOD = 100                # SMA lookback period (days)
MA_BUFFER_PCT = 0.03           # 3% hysteresis band
MA_CONFIRM_ENTRY = 2           # days above/below to confirm entry
MA_CONFIRM_EXIT = 5            # days to confirm exit (asymmetric)
MA_ALLOC_PCT = 0.50            # 50% of equity for MA positions

# ═══════════════════════════════════════════════════
# State & Logging
# ═══════════════════════════════════════════════════
STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state")
STATE_FILE = os.path.join(STATE_DIR, "bot_state.json")
LOG_DIR = os.path.join(STATE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "bot.log")
TRADE_LOG_FILE = os.path.join(LOG_DIR, "trades.log")

# ═══════════════════════════════════════════════════
# All tickers the bot needs data for
# ═══════════════════════════════════════════════════
ALL_TICKERS = list(set([
    MA_SIGNAL_GROWTH, MA_SIGNAL_SAFE,
    MA_TRADE_GROWTH, MA_TRADE_SAFE, MA_TRADE_ALT,
]))
