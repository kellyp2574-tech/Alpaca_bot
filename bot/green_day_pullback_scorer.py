"""Green-Day Pullback Rebound Scorer.

GDP_BASE sleeve:
- Green day: day_return +1% to +10%
- Below VWAP: price_vs_vwap < 0 (pullback from intraday VWAP)
- Decelerating late momentum: late_mom_1530->signal <= 0
- Buy the pullback, not the chase
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from bot import config
from bot.scorer_utils import calc_intraday_vwap, compute_intraday_base_metrics

logger = logging.getLogger(__name__)


@dataclass
class GreenDayPullbackCandidate:
    """Green-day pullback rebound candidate with intraday metrics."""
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
    vwap_930_to_signal: float = 0.0
    price_vs_vwap: float = 0.0
    close_position: float = 0.0
    late_mom_1530_signal: float = 0.0
    selection_score: float = 0.0


def build_green_day_pullback_candidates(
    symbols: List[str],
    minute_bars: Dict[str, List[dict]],
    adv_cache: Dict[str, Tuple[float, float]],
    min_bars: int = 30,
) -> List[GreenDayPullbackCandidate]:
    """Build GreenDayPullbackCandidate objects from 9:30-signal minute bars.

    Args:
        symbols:      Universe symbols that passed pipeline filters.
        minute_bars:  {symbol: [bar, ...]} with bars from 9:30 to signal time.
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

        adv_shares, adv_dollars = adv_cache.get(symbol, (0.0, 0.0))
        m = compute_intraday_base_metrics(bars, adv_shares)
        if m is None:
            skipped_price += 1
            continue

        vwap = calc_intraday_vwap(bars)
        price_vs_vwap = (m.signal_price / vwap) - 1.0 if vwap > 0 else 0.0

        candidates.append(GreenDayPullbackCandidate(
            symbol=symbol,
            signal_price=m.signal_price,
            open_price_930=m.open_price,
            high_930_to_signal=m.high_price,
            low_930_to_signal=m.low_price,
            volume_930_to_signal=m.total_volume,
            adv_20d=adv_shares,
            adv_dollars=adv_dollars,
            day_return=m.day_return,
            volume_ratio=m.volume_ratio,
            vwap_930_to_signal=vwap,
            price_vs_vwap=price_vs_vwap,
            close_position=m.close_position,
            late_mom_1530_signal=m.late_return_1530_signal,
        ))

    logger.info(
        f"build_green_day_pullback_candidates: {len(candidates)} built, "
        f"{skipped_bars} skipped (bars), {skipped_price} skipped (price)"
    )
    return candidates


def filter_green_day_pullback_candidates(
    candidates: List[GreenDayPullbackCandidate],
) -> List[GreenDayPullbackCandidate]:
    """Apply GDP filter thresholds and rank survivors by selection_score.

    Filter criteria (from config):
      - signal_price in [GDP_MIN_PRICE, GDP_MAX_PRICE]
      - day_return in [GDP_DAY_RET_MIN, GDP_DAY_RET_MAX]
      - price_vs_vwap < 0 if GDP_REQUIRE_BELOW_VWAP is True
      - late_mom_1530_signal <= GDP_LATE_MOM_MAX
      - close_position <= GDP_MAX_CLOSE_POSITION if set

    Selection score (higher is better):
      - 35% weight on VWAP pullback depth (deeper = better signal)
      - 25% weight on deceleration (more negative late_mom = better)
      - 20% weight on volume ratio
      - 20% weight on lower close_position

    Returns:
        Filtered candidates sorted by selection_score descending.
    """
    filtered = []
    rejected = {
        "price": 0,
        "day_ret": 0,
        "vwap": 0,
        "late_mom": 0,
        "close_pos": 0,
    }

    for c in candidates:
        if not (config.GDP_MIN_PRICE <= c.signal_price <= config.GDP_MAX_PRICE):
            rejected["price"] += 1
            continue

        if not (config.GDP_DAY_RET_MIN <= c.day_return <= config.GDP_DAY_RET_MAX):
            rejected["day_ret"] += 1
            continue

        if config.GDP_REQUIRE_BELOW_VWAP and not (c.price_vs_vwap < 0):
            rejected["vwap"] += 1
            continue

        if c.late_mom_1530_signal > config.GDP_LATE_MOM_MAX:
            rejected["late_mom"] += 1
            continue

        if config.GDP_MAX_CLOSE_POSITION is not None and c.close_position > config.GDP_MAX_CLOSE_POSITION:
            rejected["close_pos"] += 1
            continue

        # Score: prefer deeper-but-controlled VWAP pullback, deceleration, moderate volume,
        # and lower close_position. Keep it simple for live paper.
        vwap_pullback = min(abs(c.price_vs_vwap), 0.05) / 0.05
        decel_score = min(abs(min(c.late_mom_1530_signal, 0.0)), 0.05) / 0.05
        vol_score = min(c.volume_ratio, 2.5) / 2.5 if c.volume_ratio > 0 else 0.0
        close_score = 1.0 - max(0.0, min(c.close_position, 1.0))

        c.selection_score = (
            0.35 * vwap_pullback +
            0.25 * decel_score +
            0.20 * vol_score +
            0.20 * close_score
        )

        filtered.append(c)

    filtered.sort(key=lambda c: c.selection_score, reverse=True)

    logger.info(
        f"GDP filter: {len(candidates)} raw -> {len(filtered)} passed. "
        f"Rejected: {rejected}"
    )
    return filtered
