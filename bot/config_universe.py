"""Universe configuration — filters, presets, and data lookback periods."""

# ═══════════════════════════════════════════════════
# Universe presets — pick one via UNIVERSE_PRESET
# ═══════════════════════════════════════════════════
UNIVERSE_PRESET = "expanded_smallcap"

UNIVERSE_PRESETS = {
    "core_smallcap":            {"min_price": 1.0, "max_price": 5.0},
    "expanded_smallcap":        {"min_price": 1.0, "max_price": 10.0},
    "mean_reversion_smallcap":  {"min_price": 1.0, "max_price": 3.0},
    "midcap":                   {"min_price": 5.0, "max_price": 50.0},
    "higher_capacity":          {"min_price": 5.0, "max_price": 100.0},
}

# Resolved at import time — other modules read these directly.
_preset = UNIVERSE_PRESETS[UNIVERSE_PRESET]
MIN_PRICE = _preset["min_price"]
MAX_PRICE = _preset["max_price"]

# ═══════════════════════════════════════════════════
# Historical data lookback
# ═══════════════════════════════════════════════════
ADV_LOOKBACK_DAYS = 20   # Days for average daily volume
