"""Shared utilities for the MR and GDP scorers.

Both scorers consume 9:30→signal-time minute bars and need the same
primitives (anchor bar near 15:30, intraday VWAP, base intraday metrics).
Centralising them here keeps the two scorer files focused on their unique
filter/scoring logic.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class IntradayBaseMetrics:
    """Common intraday metrics derived from 9:30 → signal-time minute bars."""
    open_price: float
    signal_price: float
    high_price: float
    low_price: float
    total_volume: int
    day_return: float
    volume_ratio: float
    close_position: float
    price_1530: float
    late_return_1530_signal: float


def bar_near_1530(bars: List[dict]) -> Optional[dict]:
    """Return the last bar at or before 15:30 ET.

    Tries timestamp-based lookup first (handles gaps in IEX data). Falls back
    to an index approximation only if timestamps are absent.
    """
    if not bars:
        return None

    # Timestamp-based: walk backwards to find last bar at or before 15:30.
    for bar in reversed(bars):
        ts = bar.get("t", "")
        if not ts:
            break  # no timestamps on any bar — fall through to index fallback
        # Timestamps are ISO-8601 e.g. "2026-05-02T15:31:00-04:00".
        try:
            time_part = ts[11:16]  # "HH:MM"
            if time_part <= "15:30":
                return bar
        except (IndexError, TypeError):
            break

    # Index fallback: assume bars are continuous and bars[-1] is the signal.
    # 15:30 is ~20 minutes before 15:50 → bars[-21] when len>=21.
    if len(bars) >= 21:
        return bars[-21]
    return bars[0]


def calc_intraday_vwap(bars: List[dict]) -> float:
    """Volume-weighted average price across the supplied minute bars.

    Prefers per-bar VWAP (``vw``) when supplied by the data feed, otherwise
    approximates with the bar close.
    """
    dollar_volume = 0.0
    volume = 0.0

    for b in bars:
        v = b.get("v", 0) or 0
        p = b.get("vw") or b.get("c") or 0
        if p > 0 and v > 0:
            dollar_volume += p * v
            volume += v

    return dollar_volume / volume if volume > 0 else 0.0


def compute_intraday_base_metrics(
    bars: List[dict],
    adv_shares: float,
    partial_day_factor: float = 0.70,
) -> Optional[IntradayBaseMetrics]:
    """Compute the metrics shared by every intraday scorer.

    Args:
        bars: Minute bars from 9:30 to the signal time, sorted ascending.
        adv_shares: 20-day average daily share volume from the ADV cache.
        partial_day_factor: Fraction of full-day ADV that the bar window
            should equal at full activity. 0.70 ≈ 9:30 → 15:50 (~82% of RTH
            adjusted for tail-of-day skew).

    Returns:
        IntradayBaseMetrics on success, or None if the bars are unusable
        (zero/missing OHLCV).
    """
    if not bars:
        return None

    open_price = bars[0].get("o", 0) or 0
    signal_price = bars[-1].get("c", 0) or 0
    high_price = max((b.get("h", 0) for b in bars), default=0)
    low_price = min((b.get("l", 0) for b in bars if b.get("l", 0) > 0), default=0)
    total_volume = sum(b.get("v", 0) for b in bars)

    if open_price <= 0 or signal_price <= 0 or high_price <= 0 or low_price <= 0:
        return None

    day_return = (signal_price / open_price) - 1.0

    if adv_shares > 0 and partial_day_factor > 0:
        volume_ratio = total_volume / (adv_shares * partial_day_factor)
    else:
        volume_ratio = 0.0

    day_range = high_price - low_price
    close_position = (signal_price - low_price) / day_range if day_range > 0 else 1.0

    bar_1530 = bar_near_1530(bars)
    price_1530 = bar_1530.get("c", 0) if bar_1530 else 0
    late_return = (signal_price / price_1530) - 1.0 if price_1530 > 0 else 0.0

    return IntradayBaseMetrics(
        open_price=open_price,
        signal_price=signal_price,
        high_price=high_price,
        low_price=low_price,
        total_volume=int(total_volume),
        day_return=day_return,
        volume_ratio=volume_ratio,
        close_position=close_position,
        price_1530=price_1530,
        late_return_1530_signal=late_return,
    )
