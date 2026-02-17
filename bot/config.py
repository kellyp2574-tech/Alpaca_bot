"""
Bot Configuration — All strategy parameters and settings.
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
# Strategy B: Monday Dip
# ═══════════════════════════════════════════════════
DIP_SIGNAL_TICKER = "SPY"      # signal ticker for dip detection
DIP_TRADE_TICKER = "UPRO"     # 3x leveraged SPY
DIP_MIN_PCT = 0.005            # minimum 0.5% dip to trigger
DIP_ALLOC_CAP = 0.30           # max 30% of equity
DIP_HOLD_DAYS = 2              # hold for 2 trading days
DIP_TAKE_PROFIT = 0.025        # 2.5% take profit
# No stop loss for Monday Dip

# ═══════════════════════════════════════════════════
# Strategy C: BB Reversion
# ═══════════════════════════════════════════════════
BB_SIGNAL_TICKER = "SPY"       # signal ticker for BB calculation
BB_TRADE_TICKER = "UPRO"      # 3x leveraged SPY
BB_PERIOD = 20                 # Bollinger Band SMA period
BB_SIGMA = 2.0                 # standard deviations for lower band
BB_ALLOC_CAP = 0.20            # max 20% of equity
BB_STOP_LOSS = 0.10            # 10% stop loss
BB_TAKE_PROFIT = 0.05          # 5% take profit
# Exit: when SPY returns to 20-day SMA

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
    DIP_SIGNAL_TICKER, DIP_TRADE_TICKER,
    BB_SIGNAL_TICKER, BB_TRADE_TICKER,
]))
