"""Broker configuration — API credentials and endpoints."""
import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════
# Alpaca API
# ═══════════════════════════════════════════════════
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

ALPACA_BASE_URL = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

# Data feed (IEX for free tier, "sip" for paid)
DATA_FEED = "iex"

# ═══════════════════════════════════════════════════
# Massive API (Polygon-compatible, for universe reduction)
# ═══════════════════════════════════════════════════
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY", "")
MASSIVE_BASE_URL = "https://api.massive.com"
