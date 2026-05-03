"""Mean Reversion Scorer — overnight mean reversion candidate selection.

Filters the universe to stocks that had a large intraday decline with elevated
volume and closed near the session low, characteristics associated with
overnight mean reversion edge in the $1-$3 price range.

Pipeline:
  1. build_mean_reversion_candidates() — compute raw metrics from minute bars
  2. filter_mean_reversion_candidates() — apply MR filter thresholds + score
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from bot import config

logger = logging.getLogger(__name__)


@dataclass
class MeanReversionCandidate:
    """Overnight mean reversion candidate with intraday metrics."""
    symbol: str
    signal_price: float
    open_price_930: float
    high_930_to_signal: float
    low_930_to_signal: float
    volume_930_to_signal: int
    adv_20d: float
    adv_dollars: float

    day_return: float = 0.0
    volume_ratio: float = 0.0
    close_position: float = 0.0        # 0.0 = at low, 1.0 = at high
    late_drop_1530_1550: float = 0.0   # return from ~15:30 bar to signal
    selection_score: float = 0.0


def _bar_near_1530(bars: List[dict]) -> Optional[dict]:
    """Return the last bar at or before 15:30 ET.

    Tries timestamp-based lookup first (handles gaps in IEX data).
    Falls back to index approximation only if timestamps are absent.
    """
    if not bars:
        return None

    # Timestamp-based: walk backwards to find last bar at or before 15:30
    for bar in reversed(bars):
        ts = bar.get("t", "")
        if not ts:
            break  # no timestamps on any bar — fall through to index fallback
        # Timestamps are ISO-8601 e.g. "2026-05-02T15:31:00-04:00"
        # Extract HH:MM from the time portion
        try:
            time_part = ts[11:16]  # "HH:MM"
            if time_part <= "15:30":
                return bar
        except (IndexError, TypeError):
            break

    # Index fallback: assume continuous bars ending at 15:50
    if len(bars) >= 21:
        return bars[-21]
    return bars[0]


def build_mean_reversion_candidates(
    symbols: List[str],
    minute_bars: Dict[str, List[dict]],
    adv_cache: Dict[str, Tuple[float, float]],
    min_bars: int = 30,
) -> List[MeanReversionCandidate]:
    """Build MeanReversionCandidate objects from 9:30-3:50 minute bars.

    Args:
        symbols:      Universe symbols that passed pipeline filters.
        minute_bars:  {symbol: [bar, ...]} with bars from 9:30-3:50.
        adv_cache:    {symbol: (adv_shares, adv_dollars)}.
        min_bars:     Minimum minute bars required (data-quality gate).

    Returns:
        List of candidates with raw metrics populated.
    """
    candidates = []
    skipped_bars = 0
    skipped_price = 0

    for symbol in symbols:
        bars = minute_bars.get(symbol, [])
        if len(bars) < min_bars:
            skipped_bars += 1
            continue

        open_price = bars[0].get("o", 0)
        signal_price = bars[-1].get("c", 0)
        high_price = max((b.get("h", 0) for b in bars), default=0)
        low_price = min((b.get("l", 0) for b in bars if b.get("l", 0) > 0), default=0)
        total_volume = sum(b.get("v", 0) for b in bars)

        if open_price <= 0 or signal_price <= 0 or high_price <= 0 or low_price <= 0:
            skipped_price += 1
            continue

        adv_shares, adv_dollars = adv_cache.get(symbol, (0.0, 0.0))

        c = MeanReversionCandidate(
            symbol=symbol,
            signal_price=signal_price,
            open_price_930=open_price,
            high_930_to_signal=high_price,
            low_930_to_signal=low_price,
            volume_930_to_signal=total_volume,
            adv_20d=adv_shares,
            adv_dollars=adv_dollars,
        )

        c.day_return = (signal_price / open_price) - 1.0

        # Volume ratio: today partial-day vs expected partial-day volume.
        # 0.70 factor adjusts ADV (full day) to 9:30-3:50 window (~82% of RTH).
        if adv_shares > 0:
            c.volume_ratio = total_volume / (adv_shares * 0.70)

        # Close position within today's range: 0.0 = at low, 1.0 = at high
        day_range = high_price - low_price
        if day_range > 0:
            c.close_position = (signal_price - low_price) / day_range
        else:
            c.close_position = 1.0

        # Late-day momentum: return from ~15:30 bar to signal close
        bar_1530 = _bar_near_1530(bars)
        price_1530 = bar_1530.get("c", 0) if bar_1530 else 0
        if price_1530 > 0:
            c.late_drop_1530_1550 = (signal_price / price_1530) - 1.0

        candidates.append(c)

    logger.info(
        f"build_mean_reversion_candidates: {len(candidates)} built, "
        f"{skipped_bars} skipped (bars), {skipped_price} skipped (price)"
    )
    return candidates


def filter_mean_reversion_candidates(
    candidates: List[MeanReversionCandidate],
) -> List[MeanReversionCandidate]:
    """Apply MR filter thresholds and rank survivors by selection_score.

    Filter criteria (from config):
      - signal_price in [MR_MIN_PRICE, MR_MAX_PRICE]
      - day_return <= MR_DAY_RET_MAX            (e.g. <= -3%)
      - volume_ratio >= MR_VOLUME_RATIO_MIN     (e.g. >= 1.5x)
      - close_position <= MR_CLOSE_POSITION_MAX (e.g. <= 0.20)
      - late_drop_1530_1550 <= MR_LATE_DROP_MAX if set

    Selection score (higher is better):
      - 50% weight on low close_position (closed near low = strongest signal)
      - 30% weight on volume_ratio capped at 3x
      - 20% weight on magnitude of day_return capped at 10%

    Returns:
        Filtered candidates sorted by selection_score descending.
    """
    filtered = []
    rejected = {"price": 0, "day_ret": 0, "vol_ratio": 0, "close_pos": 0, "late_drop": 0}

    for c in candidates:
        if not (config.MR_MIN_PRICE <= c.signal_price <= config.MR_MAX_PRICE):
            rejected["price"] += 1
            continue
        if c.day_return > config.MR_DAY_RET_MAX:
            rejected["day_ret"] += 1
            continue
        if c.volume_ratio < config.MR_VOLUME_RATIO_MIN:
            rejected["vol_ratio"] += 1
            continue
        if c.close_position > config.MR_CLOSE_POSITION_MAX:
            rejected["close_pos"] += 1
            continue
        if config.MR_LATE_DROP_MAX is not None and c.late_drop_1530_1550 > config.MR_LATE_DROP_MAX:
            rejected["late_drop"] += 1
            continue

        c.selection_score = (
            (1.0 - c.close_position) * 0.50
            + min(c.volume_ratio / 3.0, 1.0) * 0.30
            + min(abs(c.day_return) / 0.10, 1.0) * 0.20
        )

        filtered.append(c)

    rejected_str = ", ".join(f"{k}={v}" for k, v in rejected.items() if v > 0)
    logger.info(
        f"filter_mean_reversion_candidates: {len(candidates)} in -> "
        f"{len(filtered)} passed (rejected: {rejected_str or 'none'})"
    )

    filtered.sort(key=lambda c: c.selection_score, reverse=True)
    return filtered
