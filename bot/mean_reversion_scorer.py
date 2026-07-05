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
from bot.scorer_utils import compute_intraday_base_metrics

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
    adv_multiplier: float = 1.0      # source-specific multiplier (e.g. 1.0 for Massive, 50.0 for IEX)
    adv_source: str = "unknown"    # e.g. "massive_grouped_daily" or "alpaca_iex"

    day_return: float = 0.0
    volume_ratio: float = 0.0
    volume_ratio_available: bool = True  # False if source cannot compute this metric
    close_position: float = 0.0        # 0.0 = at low, 1.0 = at high
    late_drop_1530_1550: float = 0.0   # return from ~15:30 bar to signal
    late_drop_available: bool = True   # False if source cannot compute this metric
    selection_score: float = 0.0
    prior_ret: Optional[float] = None            # T-2 to T-1 return from Massive
    prior_ret_filter_passed: Optional[bool] = None  # True if prior_ret passed configured bounds


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

        adv_shares, adv_dollars = adv_cache.get(symbol, (0.0, 0.0))
        m = compute_intraday_base_metrics(bars, adv_shares)
        if m is None:
            skipped_price += 1
            continue

        candidates.append(MeanReversionCandidate(
            symbol=symbol,
            signal_price=m.signal_price,
            open_price_930=m.open_price,
            high_930_to_signal=m.high_price,
            low_930_to_signal=m.low_price,
            volume_930_to_signal=m.total_volume,
            adv_20d=adv_shares,
            adv_dollars=adv_dollars,
            adv_multiplier=float(getattr(config, "ADV_DOLLAR_MULTIPLIER", 1.0)),
            adv_source="alpaca_iex",
            day_return=m.day_return,
            volume_ratio=m.volume_ratio,
            close_position=m.close_position,
            late_drop_1530_1550=m.late_return_1530_signal,
        ))

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

    late_drop_min = getattr(config, "MR_LATE_DROP_MIN", None)
    late_drop_max = getattr(config, "MR_LATE_DROP_MAX", None)

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
        if late_drop_max is not None and c.late_drop_1530_1550 > late_drop_max:
            rejected["late_drop"] += 1
            continue
        if late_drop_min is not None and c.late_drop_1530_1550 < late_drop_min:
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
