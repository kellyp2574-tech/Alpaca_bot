"""Runtime configuration — state paths, logging, directories."""
import os

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
