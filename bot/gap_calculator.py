"""Gap calculation and candidate filtering logic"""
import logging
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass
from bot import config

logger = logging.getLogger(__name__)


@dataclass
class GapCandidate:
    """Represents a gap momentum candidate"""
    symbol: str
    open_price: float
    prev_close: float
    gap_pct: float
    volume: int
    adv_estimate: float  # Estimated ADV based on recent data

    @property
    def price(self) -> float:
        """Alias for open_price - used by position sizing and state serialization"""
        return self.open_price


class GapCalculator:
    """Calculates overnight gaps and filters candidates"""

    def __init__(self):
        self.min_gap = config.MIN_GAP_PCT
        self.max_gap = config.MAX_GAP_PCT
        self.min_adv = config.MIN_ADV_DOLLARS

    def calculate_gap(self, open_price: float, prev_close: float) -> float:
        """Calculate overnight gap percentage"""
        if not prev_close or prev_close == 0:
            return 0.0
        return ((open_price - prev_close) / prev_close) * 100

    def estimate_adv(self, snapshots: Dict[str, dict]) -> Dict[str, float]:
        """
        Estimate ADV from available data.
        Uses 20-day average if available, otherwise uses recent volume.
        """
        adv_estimates = {}

        for symbol, data in snapshots.items():
            # Use previous day's volume as ADV estimate
            prev_volume = data.get("prev_volume", 0)
            close_price = data.get("prev_close", data.get("close", 0))

            if prev_volume and close_price:
                adv_dollars = prev_volume * close_price
                adv_estimates[symbol] = adv_dollars
            else:
                # Fallback: use today's volume if available
                volume = data.get("volume", 0)
                price = data.get("close", data.get("last_price", 0))
                if volume and price:
                    adv_estimates[symbol] = volume * price * 5  # Estimate: today * 5

        return adv_estimates

    def find_candidates(
        self, snapshots: Dict[str, dict], min_adv: Optional[float] = None
    ) -> List[GapCandidate]:
        """
        Find gap momentum candidates from Alpaca snapshots.
        Filters by gap % and ADV requirements.
        """
        if min_adv is None:
            min_adv = self.min_adv

        candidates = []
        adv_estimates = self.estimate_adv(snapshots)
        
        # Debug counters
        missing_data = 0
        low_adv = 0
        gap_too_small = 0
        gap_too_large = 0

        for symbol, data in snapshots.items():
            open_price = data.get("open")
            prev_close = data.get("prev_close")
            volume = data.get("volume", 0)
            adv = adv_estimates.get(symbol, 0)

            # Skip if missing required data
            if not open_price or not prev_close:
                missing_data += 1
                continue

            # Check ADV filter
            if adv < min_adv:
                low_adv += 1
                # Log first few with GOOD GAPS but low ADV
                test_gap = self.calculate_gap(open_price, prev_close)
                if low_adv <= 5 and abs(test_gap) >= 3:
                    logger.info(f"ADV reject (good gap): {symbol} - gap={test_gap:.1f}%, adv=${adv/1e6:.2f}M")
                continue

            # Calculate gap
            gap_pct = self.calculate_gap(open_price, prev_close)

            # Check gap range
            if gap_pct < self.min_gap:
                gap_too_small += 1
                continue
            if gap_pct > self.max_gap:
                gap_too_large += 1
                if gap_too_large <= 3:
                    logger.info(f"Gap too large: {symbol} - gap={gap_pct:.1f}% (max={self.max_gap}%)")
                continue

            # Log validation for top candidates to diagnose data issues
            if len(candidates) < 10:
                logger.info(
                    f"{symbol} | open={open_price} prev_close={prev_close} "
                    f"gap={gap_pct:+.2f}% volume={volume} adv=${adv/1e6:.2f}M"
                )

            candidate = GapCandidate(
                symbol=symbol,
                open_price=open_price,
                prev_close=prev_close,
                gap_pct=gap_pct,
                volume=volume,
                adv_estimate=adv,
            )
            candidates.append(candidate)

        # Sort by gap percentage (descending)
        candidates.sort(key=lambda x: abs(x.gap_pct), reverse=True)
        
        # Debug summary
        total = len(snapshots)
        logger.info(f"Gap filter summary: {total} total, {missing_data} missing data, {low_adv} low ADV, {gap_too_small} gap too small, {gap_too_large} gap too large, {len(candidates)} passed")

        logger.info(f"Gap candidates: {len(candidates)} (gap {self.min_gap}%-{self.max_gap}%)")
        return candidates

    def select_by_liquidity_and_gap(
        self, candidates: List[GapCandidate], max_positions: int = 20
    ) -> List[GapCandidate]:
        """
        Select top candidates by liquidity (ADV) first, then gap quality.
        Sorts by ADV descending (most liquid first), then takes top max_positions.
        """
        # Sort by liquidity (ADV) descending - most liquid first
        sorted_by_liq = sorted(candidates, key=lambda x: x.adv_estimate, reverse=True)

        # Take top N
        selected = sorted_by_liq[:max_positions]

        logger.info(f"Liquidity selection: {len(selected)}/{len(candidates)} (top {max_positions} by ADV)")
        return selected
