"""Premarket overnight-exit limit classification (05:00 -> 06:00).

Pure-ish functions used by the 05:00/05:15/05:30/05:45/06:00 checkpoint
runner to decide whether a resting limit-sell should be placed on each
overnight position and at what percent above entry.

Inputs:
    - Position objects (read-only, just for ``sleeve`` and entry price)
    - Delayed-SIP 1-minute bars (15+ minute delay to satisfy the
      non-realtime-SIP entitlement)
    - Bot config thresholds

All HTTP I/O is concentrated in ``fetch_delayed_sip_premarket_bars``;
everything else is computation.
"""
from __future__ import annotations

import logging
from datetime import datetime, time as dt_time, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from bot import config

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


# ──────────────────────────────────────────────────────────────
# Bar helpers
# ──────────────────────────────────────────────────────────────

def bar_dt(bar: dict) -> Optional[datetime]:
    """Parse Alpaca bar timestamp into America/New_York datetime."""
    raw = bar.get("t") or bar.get("timestamp") or bar.get("time")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_ET)
        return parsed.astimezone(_ET)
    except Exception:
        return None


def bar_float(bar: dict, *keys: str) -> Optional[float]:
    """Read a float from Alpaca bar keys, supporting both short and long names."""
    for key in keys:
        if key in bar and bar.get(key) is not None:
            try:
                return float(bar.get(key))
            except (TypeError, ValueError):
                continue
    return None


# ──────────────────────────────────────────────────────────────
# Data fetch
# ──────────────────────────────────────────────────────────────

def fetch_delayed_sip_premarket_bars(
    session,
    symbols: List[str],
    decision_dt: datetime,
) -> Dict[str, List[dict]]:
    """Fetch delayed SIP 1-min bars from 04:00 through ``decision_dt - delay``.

    The end parameter is deliberately offset by at least 15 minutes to
    satisfy the non-realtime-SIP entitlement. Caller supplies a
    requests.Session-like object so we don't have to import the
    PositionManager here.
    """
    delay_minutes = getattr(config, "PREMARKET_SIP_DELAY_MINUTES", 16)
    start_dt = datetime.combine(decision_dt.date(), dt_time(4, 0), tzinfo=_ET)
    end_dt = decision_dt - timedelta(minutes=delay_minutes)

    data_url = getattr(config, "ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
    url = f"{data_url}/v2/stocks/bars"

    params = {
        "symbols": ",".join(symbols),
        "timeframe": "1Min",
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "adjustment": "raw",
        "feed": "sip",
        "limit": 10000,
    }

    try:
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        bars_by_symbol: Dict[str, List[dict]] = {}
        for symbol, bars_data in payload.get("bars", {}).items():
            bars_by_symbol[symbol] = bars_data if isinstance(bars_data, list) else []
        logger.info(
            f"Delayed SIP premarket bars: fetched {len(bars_by_symbol)} symbols, end={end_dt.strftime('%H:%M')}"
        )
        return bars_by_symbol
    except Exception as e:
        logger.warning(
            f"Delayed SIP premarket bars: failed to fetch for {len(symbols)} symbols: {e}",
            exc_info=True,
        )
        return {}


# ──────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────

def compute_delayed_sip_premarket_metrics(
    session,
    symbol: str,
    buy_price: float,
    decision_dt: datetime,
    pre_fetched_bars: Optional[Dict[str, List[dict]]] = None,
) -> Dict[str, Any]:
    """Compute premarket metrics from delayed SIP historical bars.

    Returns a dict with at minimum ``has_data`` and (when True)
    ``current_return``, ``distance_from_high``, ``trend_from_first_bar``,
    ``premarket_minutes``, ``last_bar_age_minutes``, etc.
    """
    if pre_fetched_bars is not None:
        bars_raw = pre_fetched_bars.get(symbol, [])
    else:
        bars_by_symbol = fetch_delayed_sip_premarket_bars(session, [symbol], decision_dt)
        bars_raw = bars_by_symbol.get(symbol, [])

    normalized = []
    for bar in bars_raw:
        dt_val = bar_dt(bar)
        close = bar_float(bar, "c", "close")
        high = bar_float(bar, "h", "high")
        low = bar_float(bar, "l", "low")
        volume = bar_float(bar, "v", "volume") or 0.0
        if dt_val and close and high and low:
            normalized.append({
                "dt": dt_val,
                "close": close,
                "high": high,
                "low": low,
                "volume": volume,
            })

    normalized.sort(key=lambda b: b["dt"])

    if not normalized:
        return {
            "has_data": False,
            "reason": "no_delayed_sip_bars",
            "premarket_minutes": 0,
            "current_return": None,
        }

    first = normalized[0]
    last = normalized[-1]
    sip_stale_minutes = (decision_dt - last["dt"]).total_seconds() / 60.0
    sip_high = max(b["high"] for b in normalized)
    sip_low = min(b["low"] for b in normalized)
    premarket_volume = sum(b["volume"] for b in normalized)
    sip_current = last["close"]
    first_price = first["close"]

    current_return = sip_current / buy_price - 1.0 if buy_price > 0 else 0.0
    distance_from_high = sip_current / sip_high - 1.0 if sip_high > 0 else 0.0
    return_from_low = sip_current / sip_low - 1.0 if sip_low > 0 else 0.0
    trend_from_first_bar = sip_current / first_price - 1.0 if first_price > 0 else 0.0

    logger.info(
        "PREMARKET DELAYED SIP %s: entry=%.4f current=%.4f ret=%+.2f%% "
        "high=%.4f low=%.4f bars=%d stale=%.0fm",
        symbol,
        buy_price,
        sip_current,
        current_return * 100.0,
        sip_high,
        sip_low,
        len(normalized),
        sip_stale_minutes,
    )

    return {
        "has_data": True,
        "reason": "delayed_sip_bars",
        "price_source": "delayed_sip",
        "first_premarket_time": first["dt"],
        "first_premarket_price": first_price,
        "current_time": last["dt"],
        "current_price": sip_current,
        "iex_current_price": None,
        "sip_current_price": sip_current,
        "sip_snapshot_reason": None,
        "premarket_high": sip_high,
        "iex_premarket_high": sip_high,
        "premarket_low": sip_low,
        "premarket_minutes": len(normalized),
        "premarket_volume": premarket_volume,
        "last_bar_age_minutes": sip_stale_minutes,
        "iex_last_bar_age_minutes": sip_stale_minutes,
        "snapshot_spread_pct": None,
        "current_return": current_return,
        "distance_from_high": distance_from_high,
        "return_from_low": return_from_low,
        "trend_from_first_bar": trend_from_first_bar,
    }


# ──────────────────────────────────────────────────────────────
# Decision logic
# ──────────────────────────────────────────────────────────────

def classify_premarket_limit(pos, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Lenient delayed-SIP dynamic limit decision.

    Sparse bars are treated as usable signal rather than requiring dense
    coverage. The fallback for unclear signals is a normal 5% harvest
    limit rather than no decision.
    """
    fallback_limit = getattr(config, "PREMARKET_DYNAMIC_DEFAULT_LIMIT_PCT", 0.05)
    sparse_wide_limit = getattr(config, "PREMARKET_DYNAMIC_SPARSE_HIGH_RETURN_LIMIT_PCT", 0.10)
    very_high = getattr(config, "PREMARKET_DYNAMIC_VERY_HIGH_RETURN_NO_CAP_PCT", 0.10)
    high = getattr(config, "PREMARKET_DYNAMIC_HIGH_RETURN_NO_CAP_PCT", 0.05)
    moderate = getattr(config, "PREMARKET_DYNAMIC_MODERATE_RETURN_PCT", 0.02)
    stale_max = getattr(config, "PREMARKET_DYNAMIC_MAX_STALE_MINUTES", 60)

    if not metrics.get("has_data"):
        return {"action": "NO_ACTION", "limit_pct": None, "reason": "data_unavailable"}

    bars = int(metrics.get("premarket_minutes", 0) or 0)
    stale = float(metrics.get("last_bar_age_minutes", 999) or 999)
    current_return = metrics.get("current_return")

    if current_return is None:
        return {
            "action": "NO_ACTION",
            "limit_pct": None,
            "reason": "data_unavailable_in_classifier",
        }

    current_return = float(current_return) if current_return is not None else 0.0
    distance_from_high = float(metrics.get("distance_from_high", 0.0) or 0.0)
    trend = float(metrics.get("trend_from_first_bar", 0.0) or 0.0)
    sleeve = str(getattr(pos, "sleeve", "UNKNOWN") or "UNKNOWN").upper()
    fresh_enough = stale <= stale_max
    data_source = metrics.get("reason", "")
    is_snapshot = data_source == "snapshot_data"

    if is_snapshot:
        if current_return >= very_high and fresh_enough:
            return {"action": "NO_CAP", "limit_pct": None, "reason": "snapshot_very_high_return_no_cap"}
        elif current_return >= high:
            if sleeve == "MR" and current_return < very_high:
                return {"action": "PLACE_LIMIT", "limit_pct": fallback_limit, "reason": "snapshot_high_return_mr_harvest_5pct"}
            return {"action": "NO_CAP", "limit_pct": None, "reason": "snapshot_high_return_no_cap"}
        elif current_return >= moderate:
            return {"action": "PLACE_LIMIT", "limit_pct": 0.06, "reason": "snapshot_moderate_return_6pct"}
        elif current_return >= 0:
            return {"action": "PLACE_LIMIT", "limit_pct": 0.04, "reason": "snapshot_small_winner_4pct"}
        else:
            return {"action": "PLACE_LIMIT", "limit_pct": 0.03, "reason": "snapshot_negative_pop_harvest_3pct"}

    # Bar-based decision (delayed SIP).
    if current_return >= very_high and bars >= 1 and fresh_enough:
        return {"action": "NO_CAP", "limit_pct": None, "reason": "sip_very_high_return_no_cap"}

    if current_return >= high:
        if bars >= 2 and fresh_enough:
            if sleeve == "MR" and current_return < very_high:
                return {"action": "PLACE_LIMIT", "limit_pct": fallback_limit, "reason": "sip_high_return_mr_harvest_5pct"}
            return {"action": "NO_CAP", "limit_pct": None, "reason": "sip_high_return_no_cap"}
        return {"action": "PLACE_LIMIT", "limit_pct": sparse_wide_limit, "reason": "sip_high_return_sparse_wide_10pct"}

    if current_return >= moderate:
        if distance_from_high > -0.01 and trend > 0 and fresh_enough:
            return {"action": "PLACE_LIMIT", "limit_pct": 0.07, "reason": "sip_moderate_near_high_7pct"}
        if distance_from_high < -0.03 or trend < 0:
            return {"action": "PLACE_LIMIT", "limit_pct": 0.05, "reason": "sip_moderate_fading_5pct"}
        return {"action": "PLACE_LIMIT", "limit_pct": 0.06, "reason": "sip_moderate_default_6pct"}

    if current_return >= 0:
        return {"action": "PLACE_LIMIT", "limit_pct": 0.04, "reason": "sip_small_winner_4pct"}

    return {"action": "PLACE_LIMIT", "limit_pct": 0.03, "reason": "sip_negative_pop_harvest_3pct"}


def is_decisive_premarket_signal(
    decision_time: str,
    final_time: str,
    current_return: float,
    distance_from_high: float,
    trend_from_first_bar: float,
    minutes_traded: int,
    last_bar_age_minutes: float = 999,
    sleeve: str = "UNKNOWN",
    data_source: str = "",
) -> Tuple[bool, str]:
    """Decisive = act now. Not decisive = leave unresolved for next checkpoint.

    Snapshot data is treated specially: no bar count requirements, only
    freshness and return thresholds. Red/weak signals are decisive
    regardless of source to allow early lower-limit placement.
    """
    if decision_time >= final_time:
        return True, "final_checkpoint"

    fresh = last_bar_age_minutes <= getattr(config, "PREMARKET_DYNAMIC_MAX_STALE_MINUTES", 60)
    is_snapshot = data_source == "snapshot_data"

    if fresh and current_return <= -0.01:
        return True, "decisive_red_lower_limit"

    if is_snapshot and fresh:
        if current_return >= 0.10:
            return True, "decisive_snapshot_very_high_return"
        if current_return >= 0.05:
            return True, "decisive_snapshot_high_return"
        if sleeve.upper() == "MR" and current_return >= 0.03:
            return True, "decisive_snapshot_mr_harvest"

    if current_return >= 0.10 and minutes_traded >= 1 and fresh:
        return True, "decisive_very_high_return"

    if current_return >= 0.05 and minutes_traded >= 2 and fresh:
        return True, "decisive_high_return"

    if (
        current_return >= 0.02
        and distance_from_high > -0.01
        and trend_from_first_bar > 0
        and minutes_traded >= 2
        and fresh
    ):
        return True, "decisive_moderate_near_high_building"

    if (
        current_return >= 0.02
        and distance_from_high < -0.03
        and minutes_traded >= 2
        and fresh
    ):
        return True, "decisive_moderate_fading"

    if (
        sleeve.upper() == "MR"
        and current_return >= 0.03
        and minutes_traded >= 2
        and fresh
    ):
        return True, "decisive_mr_harvest"

    return False, "not_decisive_wait"
