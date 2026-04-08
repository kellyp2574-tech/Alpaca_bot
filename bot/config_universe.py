"""Universe configuration — filters, presets, and data lookback periods."""

# ═══════════════════════════════════════════════════
# Universe presets — pick one via UNIVERSE_PRESET
# ═══════════════════════════════════════════════════
UNIVERSE_PRESET = "core_smallcap"

UNIVERSE_PRESETS = {
    "core_smallcap": {
        "min_price": 1.0,
        "max_price": 5.0,
        "min_adv_dollars": 3_000_000,
    },
    "expanded_smallcap": {
        "min_price": 1.0,
        "max_price": 10.0,
        "min_adv_dollars": 3_000_000,
    },
    "midcap": {
        "min_price": 5.0,
        "max_price": 50.0,
        "min_adv_dollars": 5_000_000,
    },
    "higher_capacity": {
        "min_price": 5.0,
        "max_price": 100.0,
        "min_adv_dollars": 10_000_000,
    },
}

# Resolved at import time — other modules read these directly
_preset = UNIVERSE_PRESETS[UNIVERSE_PRESET]
MIN_PRICE = _preset["min_price"]
MAX_PRICE = _preset["max_price"]
MIN_ADV_DOLLARS = _preset["min_adv_dollars"]

# ═══════════════════════════════════════════════════
# Historical data lookback
# ═══════════════════════════════════════════════════
ADV_LOOKBACK_DAYS = 20   # Days for average daily volume
ATR_LOOKBACK_DAYS = 14   # Days for ATR

# Maximum retries for Massive snapshot before Alpaca fallback
UNIVERSE_MAX_RETRIES = 3
