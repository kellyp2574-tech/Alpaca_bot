"""Overnight Momentum Scoring Engine — 3:50 PM signal snapshot.

All functions and field names use the "350" suffix to mark them as the
validated, live-trading signal model.  Do not mix with earlier prototypes.

Pipeline:
1. build_signal_candidates_350()  — construct candidates from bars
2. compute_raw_metrics_350()      — populate 7 raw metrics
3. normalize_and_score_350()      — z-score + weighted composite
4. assign_buckets()               — decile 1-10
5. select_positions()             — pick + size using SelectionConfig
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from bot import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Selection config (runtime, chosen by account tier)
# ──────────────────────────────────────────────────────────────

@dataclass
class SelectionConfig:
    """Runtime selection parameters — chosen based on account equity."""
    selection_mode: str = "top10"   # "top10", "top20", "bucket"
    min_bucket: int = 4
    max_positions: int = 10
    max_leverage: float = 1.0
    adv_cap_pct: float = 0.003
    max_position_dollars: float = 50_000


def get_selection_config(equity: float) -> SelectionConfig:
    """Pick the right tier based on current account equity."""
    for tier in config.STRATEGY_TIERS:
        cap = tier["max_equity"]
        if cap is None or equity <= cap:
            cfg = SelectionConfig(
                selection_mode=tier["selection_mode"],
                min_bucket=tier["min_bucket"],
                max_positions=tier["max_positions"],
                max_leverage=config.MAX_LEVERAGE,
                adv_cap_pct=config.ADV_CAP_PCT,
                max_position_dollars=config.MAX_POSITION_DOLLARS,
            )
            logger.info(
                f"Account ${equity:,.0f} → tier {tier['selection_mode']} "
                f"(max_positions={cfg.max_positions})"
            )
            return cfg
    # Fallback
    return SelectionConfig()


# ──────────────────────────────────────────────────────────────
# Candidate dataclass — 3:50 PM signal snapshot
# ──────────────────────────────────────────────────────────────

@dataclass
class MomentumCandidate:
    """Scored candidate for overnight momentum entry.

    All price/volume fields reflect data through 3:50 PM (signal time).
    """
    symbol: str
    signal_price: float          # Last close at signal time (~3:50 PM)
    open_price_930: float        # Opening price at 9:30 AM
    high_930_to_signal: float    # Max high from 9:30 to signal time
    volume_930_to_signal: int    # Volume accumulated 9:30 to signal time
    adv_20d: float               # 20-day average daily volume (shares)
    adv_dollars: float           # 20-day average daily dollar volume
    atr_14d: float               # 14-period ATR

    # Raw metrics (computed by compute_raw_metrics_350)
    intraday_return: float = 0.0
    proximity_to_high: float = 0.0
    volume_vs_avg: float = 0.0
    volume_trend: float = 0.0    # last-60min volume / avg-60min volume
    vs_market: float = 0.0
    atr_percent: float = 0.0

    # Scored
    composite_score: float = 0.0
    bucket: int = 0


# ──────────────────────────────────────────────────────────────
# Step 1: Build candidates from pre-fetched bars
# ──────────────────────────────────────────────────────────────

def build_signal_candidates_350(
    symbols: List[str],
    minute_bars: Dict[str, List[dict]],
    adv_cache: Dict[str, Tuple[float, float]],
    atr_cache: Dict[str, float],
    min_bars: int = 30,
) -> List[MomentumCandidate]:
    """Build MomentumCandidate objects from 9:30-3:50 minute bars.

    Args:
        symbols: Universe symbols that passed all filters.
        minute_bars: {symbol: [bar, ...]} with bars from 9:30-3:50.
        adv_cache: {symbol: (adv_shares, adv_dollars)}
        atr_cache: {symbol: atr_14d}
        min_bars: Minimum minute bars required (data-quality gate).
    """
    candidates = []
    skipped_no_bars = 0
    skipped_bad_price = 0

    for symbol in symbols:
        bars = minute_bars.get(symbol, [])
        if len(bars) < min_bars:
            skipped_no_bars += 1
            continue

        open_price = bars[0].get("o", 0)
        signal_price = bars[-1].get("c", 0)
        high_intraday = max((b.get("h", 0) for b in bars), default=0)
        total_volume = sum(b.get("v", 0) for b in bars)

        if not open_price or open_price <= 0 or not signal_price or signal_price <= 0:
            skipped_bad_price += 1
            continue

        adv_shares, adv_dollars = adv_cache.get(symbol, (0.0, 0.0))
        atr = atr_cache.get(symbol, 0.0)

        candidates.append(MomentumCandidate(
            symbol=symbol,
            signal_price=signal_price,
            open_price_930=open_price,
            high_930_to_signal=high_intraday,
            volume_930_to_signal=total_volume,
            adv_20d=adv_shares,
            adv_dollars=adv_dollars,
            atr_14d=atr,
        ))

    logger.info(
        f"build_signal_candidates_350: {len(candidates)} built, "
        f"{skipped_no_bars} skipped (bars), {skipped_bad_price} skipped (price)"
    )
    return candidates


# ──────────────────────────────────────────────────────────────
# Step 2: Compute raw metrics (3:50 PM signal)
# ──────────────────────────────────────────────────────────────

def compute_raw_metrics_350(
    candidates: List[MomentumCandidate],
    spy_return: float,
    volume_last_60min: Dict[str, int],
    volume_avg_60min: Dict[str, float],
) -> List[MomentumCandidate]:
    """Compute raw metrics using data available at 3:50 PM.

    vs_sector is NOT populated — weight is zeroed in config until a real
    sector mapping exists.  This avoids fake diversification in the signal.

    Args:
        candidates: Candidates with base fields populated.
        spy_return: SPY intraday return (open to signal time).
        volume_last_60min: {symbol: volume in last 60 minutes}.
        volume_avg_60min: {symbol: average 60-min bucket volume for the day}.
    """
    for c in candidates:
        # Price metrics
        if c.open_price_930 > 0:
            c.intraday_return = (c.signal_price - c.open_price_930) / c.open_price_930
        if c.high_930_to_signal > 0:
            c.proximity_to_high = c.signal_price / c.high_930_to_signal

        # Volume: today / (ADV × 0.70) — 0.70 adjusts for 9:30-3:50 = ~82% of RTH
        if c.adv_20d > 0 and c.volume_930_to_signal > 0:
            c.volume_vs_avg = c.volume_930_to_signal / (c.adv_20d * 0.70)

        # Volume trend: last 60 min vs average 60 min
        sym_vol = volume_last_60min.get(c.symbol, 0)
        sym_avg = volume_avg_60min.get(c.symbol, 0)
        if sym_avg > 0:
            c.volume_trend = sym_vol / sym_avg

        # Relative strength vs market
        if spy_return and spy_return != 0:
            c.vs_market = c.intraday_return / spy_return if c.intraday_return else 0.0

        # Volatility
        if c.signal_price > 0 and c.atr_14d > 0:
            c.atr_percent = c.atr_14d / c.signal_price

    return candidates


# ──────────────────────────────────────────────────────────────
# Step 3: Normalize + score
# ──────────────────────────────────────────────────────────────

def _zscore(values: np.ndarray) -> np.ndarray:
    """Compute z-scores. Returns zeros if std is ~0."""
    std = np.std(values)
    if std < 1e-10:
        return np.zeros_like(values)
    return (values - np.mean(values)) / std


def normalize_and_score_350(candidates: List[MomentumCandidate]) -> List[MomentumCandidate]:
    """Z-score raw metrics and compute weighted composite score."""
    if not candidates:
        return candidates

    n = len(candidates)
    if n < 3:
        logger.warning(f"Only {n} candidates — z-scores unreliable with small sample")

    z_intraday = _zscore(np.array([c.intraday_return for c in candidates]))
    z_proximity = _zscore(np.array([c.proximity_to_high for c in candidates]))
    z_vol_avg = _zscore(np.array([c.volume_vs_avg for c in candidates]))
    z_vol_trend = _zscore(np.array([c.volume_trend for c in candidates]))
    z_market = _zscore(np.array([c.vs_market for c in candidates]))
    z_atr = _zscore(np.array([c.atr_percent for c in candidates]))

    for i, c in enumerate(candidates):
        c.composite_score = (
            config.SCORE_WEIGHT_INTRADAY_RETURN * z_intraday[i] +
            config.SCORE_WEIGHT_PROXIMITY_HIGH * z_proximity[i] +
            config.SCORE_WEIGHT_VOLUME_VS_AVG * z_vol_avg[i] +
            config.SCORE_WEIGHT_VOLUME_TREND * z_vol_trend[i] +
            config.SCORE_WEIGHT_VS_MARKET * z_market[i] +
            config.SCORE_WEIGHT_ATR_PCT * z_atr[i]
        )

    return candidates


# ──────────────────────────────────────────────────────────────
# Step 4: Bucket assignment
# ──────────────────────────────────────────────────────────────

def assign_buckets(candidates: List[MomentumCandidate]) -> List[MomentumCandidate]:
    """Assign decile buckets (1-10) based on composite score percentile."""
    if not candidates:
        return candidates

    scores = np.array([c.composite_score for c in candidates])
    n = len(scores)

    for c in candidates:
        rank_pct = np.sum(scores <= c.composite_score) / n
        c.bucket = max(1, min(10, int(np.ceil(rank_pct * 10))))

    return candidates


# ──────────────────────────────────────────────────────────────
# Step 5: Selection + sizing
# ──────────────────────────────────────────────────────────────

def select_positions(
    candidates: List[MomentumCandidate],
    equity: float,
    sel: Optional[SelectionConfig] = None,
) -> Tuple[List[MomentumCandidate], Dict[str, int]]:
    """Select positions and compute share quantities.

    Args:
        candidates: Scored + bucketed candidates.
        equity: Current account equity.
        sel: Runtime selection config (from get_selection_config). Falls back
             to default if None.

    Returns:
        (selected_candidates, {symbol: num_shares})
    """
    if sel is None:
        sel = get_selection_config(equity)

    # Filter by minimum bucket
    eligible = [c for c in candidates if c.bucket >= sel.min_bucket]
    if not eligible:
        logger.warning(f"No candidates with bucket >= {sel.min_bucket}")
        return [], {}

    # Sort by composite score descending
    eligible.sort(key=lambda c: c.composite_score, reverse=True)

    # Select based on mode
    if sel.selection_mode == "top10":
        selected = eligible[:10]
    elif sel.selection_mode == "top20":
        selected = eligible[:20]
    elif sel.selection_mode == "bucket":
        selected = eligible
    else:
        logger.error(f"Unknown selection mode: {sel.selection_mode}")
        selected = eligible[:10]

    selected = selected[:sel.max_positions]
    if not selected:
        return [], {}

    # Position sizing: equal weight, ADV-capped
    deployable = equity * sel.max_leverage
    weight = 1.0 / len(selected)
    dollar_per_position = deployable * weight

    sizing: Dict[str, int] = {}
    for c in selected:
        alloc = min(dollar_per_position, sel.max_position_dollars)
        if c.signal_price <= 0:
            continue
        desired_shares = int(alloc / c.signal_price)

        # ADV cap
        if c.adv_dollars > 0:
            adv_dollar_cap = c.adv_dollars * sel.adv_cap_pct
            max_shares_by_adv = int(adv_dollar_cap / c.signal_price)
            desired_shares = min(desired_shares, max_shares_by_adv)

        if desired_shares > 0:
            sizing[c.symbol] = desired_shares

    logger.info(
        f"Selection: {len(sizing)} positions from {len(eligible)} eligible "
        f"(mode={sel.selection_mode}, equity=${equity:,.0f}, deployable=${deployable:,.0f})"
    )
    return selected, sizing
