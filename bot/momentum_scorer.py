"""Overnight Momentum Scoring Engine — 3:50 PM signal snapshot.

All functions and field names use the "350" suffix to mark them as the
validated, live-trading signal model.  Do not mix with earlier prototypes.

Pipeline:
1. build_signal_candidates_350()  — construct candidates from bars
2. compute_raw_metrics_350()      — populate 7 raw metrics
3. normalize_and_score_350()      — z-score + weighted composite
4. assign_buckets()               — decile 1-10
5. allocate_head_tail()            — HEAD/TAIL position sizing
"""
import logging
import math
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
    min_bucket: int = 4
    max_positions: int = 10
    max_head_positions: int = 10
    max_leverage: float = 1.0
    adv_cap_pct: float = 0.003
    max_position_dollars: float = 50_000
    min_shares: int = 25


def get_selection_config(equity: float) -> SelectionConfig:
    """Pick the right tier based on current account equity."""
    for tier in config.STRATEGY_TIERS:
        cap = tier["max_equity"]
        if cap is None or equity <= cap:
            cfg = SelectionConfig(
                min_bucket=tier["min_bucket"],
                max_positions=tier["max_positions"],
                max_head_positions=min(tier["max_positions"], config.MAX_HEAD_POSITIONS),
                max_leverage=config.MAX_LEVERAGE,
                adv_cap_pct=config.ADV_CAP_PCT,
                max_position_dollars=config.MAX_POSITION_DOLLARS,
                min_shares=config.MIN_SHARES,
            )
            logger.info(
                f"Account ${equity:,.0f} -> tier "
                f"(min_bucket={cfg.min_bucket}, max_positions={cfg.max_positions})"
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

        # Relative strength vs market (difference, not ratio — stable when SPY near 0)
        c.vs_market = c.intraday_return - spy_return

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
# Step 5: HEAD / TAIL position allocation
# ──────────────────────────────────────────────────────────────

@dataclass
class Allocation:
    """Single position allocation produced by the HEAD/TAIL allocator."""
    symbol: str
    shares: int
    rank: int
    alloc_bucket: str            # "HEAD" or "TAIL"
    candidate: MomentumCandidate # reference to scored candidate


def allocate_head_tail(
    candidates: List[MomentumCandidate],
    total_capital: float,
    sel: Optional[SelectionConfig] = None,
) -> List[Allocation]:
    """HEAD/TAIL position allocator — runs once per day before entry.

    Decision tree:
      1. Rank candidates by composite_score descending.
      2. Split: head = top max_head_positions, tail = rest.
      3. HEAD (HEAD_PCT of capital): equal-weight, ADV-capped, MIN_SHARES gate.
         Unspent capital rolls into tail.
      4. TAIL (TAIL_PCT + leftover): waterfall until capital or position cap exhausted.
      5. Combine and return.

    Args:
        candidates: Scored + bucketed candidates.
        total_capital: Deployable cash (equity × leverage).
        sel: Runtime selection config (from get_selection_config). Uses
             global defaults when None.

    Returns:
        List[Allocation] ordered head-first then tail by rank.
    """
    min_bucket = sel.min_bucket if sel else 4
    adv_cap_pct = sel.adv_cap_pct if sel else config.ADV_CAP_PCT
    min_shares = sel.min_shares if sel else config.MIN_SHARES
    max_head = sel.max_head_positions if sel else config.MAX_HEAD_POSITIONS
    max_total = sel.max_positions if sel else config.MAX_TOTAL_POSITIONS
    head_pct = config.HEAD_PCT
    tail_pct = config.TAIL_PCT

    # ── Step 1: filter + rank ────────────────────────────
    eligible = [c for c in candidates if c.bucket >= min_bucket]
    if not eligible:
        logger.warning(f"No candidates with bucket >= {min_bucket}")
        return []

    eligible.sort(key=lambda c: c.composite_score, reverse=True)

    # ── Step 2: split HEAD / TAIL ────────────────────────
    head_candidates = eligible[:max_head]
    tail_candidates = eligible[max_head:]

    # ── Step 3: allocate HEAD (equal-weight) ─────────────
    head_capital = total_capital * head_pct
    slot_size = head_capital / len(head_candidates) if head_candidates else 0

    head_allocations: List[Allocation] = []
    head_leftover = 0.0

    for rank, c in enumerate(head_candidates, start=1):
        if c.signal_price <= 0:
            head_leftover += slot_size
            continue

        max_dollars = min(slot_size, c.adv_dollars * adv_cap_pct) if c.adv_dollars > 0 else slot_size
        shares = math.floor(max_dollars / c.signal_price)

        if shares < min_shares:
            head_leftover += slot_size      # roll full slot forward
            logger.info(
                f"HEAD skip {c.symbol}: {shares} shares < {min_shares} min "
                f"(price={c.signal_price:.2f}, slot=${slot_size:,.0f}, "
                f"adv_cap=${c.adv_dollars * adv_cap_pct:,.0f})"
            )
            continue

        cost = shares * c.signal_price
        head_allocations.append(Allocation(
            symbol=c.symbol,
            shares=shares,
            rank=rank,
            alloc_bucket="HEAD",
            candidate=c,
        ))
        head_leftover += (slot_size - cost)

    # ── Step 4: allocate TAIL (spread waterfall) ─────────
    tail_capital = total_capital * tail_pct + head_leftover
    remaining_cash = tail_capital
    positions_count = len(head_allocations)

    tail_allocations: List[Allocation] = []

    # Compute per-position slice: spread tail across target_slots,
    # also cap at head_slot * TAIL_MAX_POSITION_FACTOR to prevent
    # any single tail name from absorbing the whole tail budget.
    target_slots = config.TAIL_TARGET_SLOTS
    base_tail_slice = tail_capital / target_slots if target_slots > 0 else tail_capital
    head_slot_size = slot_size if head_candidates else tail_capital
    tail_position_cap = min(
        base_tail_slice,
        head_slot_size * config.TAIL_MAX_POSITION_FACTOR,
    )

    deployed_so_far = sum(a.shares * a.candidate.signal_price for a in head_allocations)

    skip_reasons = {
        "max_positions": 0,
        "insufficient_cash": 0,
        "min_shares": 0,
        "price_zero": 0,
    }

    for rank_offset, c in enumerate(tail_candidates, start=max_head + 1):
        if c.signal_price <= 0:
            skip_reasons["price_zero"] += 1
            continue

        if positions_count >= max_total:
            skip_reasons["max_positions"] += 1
            logger.info(f"TAIL: reached max positions ({max_total}), stopping")
            break

        min_cost_needed = c.signal_price * min_shares
        if remaining_cash < min_cost_needed:
            skip_reasons["insufficient_cash"] += 1
            logger.debug(
                f"TAIL skip {c.symbol}: remaining ${remaining_cash:,.0f} "
                f"< min cost ${min_cost_needed:,.0f}"
            )
            continue

        # Cap: slice target, ADV cap, position cap, remaining cash
        adv_cap = c.adv_dollars * adv_cap_pct if c.adv_dollars > 0 else tail_position_cap
        max_dollars = min(tail_position_cap, adv_cap, remaining_cash)
        shares = math.floor(max_dollars / c.signal_price)

        if shares < min_shares:
            skip_reasons["min_shares"] += 1
            logger.debug(
                f"TAIL skip {c.symbol}: {shares} shares < {min_shares} min "
                f"(price=${c.signal_price:.2f}, max_dollars=${max_dollars:,.0f})"
            )
            continue        # skip, do NOT consume cash

        cost = shares * c.signal_price
        tail_allocations.append(Allocation(
            symbol=c.symbol,
            shares=shares,
            rank=rank_offset,
            alloc_bucket="TAIL",
            candidate=c,
        ))
        remaining_cash -= cost
        deployed_so_far += cost
        positions_count += 1

        deploy_pct = deployed_so_far / total_capital * 100
        logger.debug(
            f"TAIL allocated {c.symbol}: ${cost:,.0f} "
            f"(slice=${max_dollars:,.0f}, cap=${tail_position_cap:,.0f}), "
            f"{deploy_pct:.1f}% total deployed"
        )

    # ── Step 5: combine ──────────────────────────────────
    final = head_allocations + tail_allocations

    # ── Step 6: safety ───────────────────────────────────
    total_cost = sum(a.shares * a.candidate.signal_price for a in final)
    if total_cost > total_capital * 1.001:  # tiny float tolerance
        logger.error(
            f"Allocation exceeds capital: ${total_cost:,.2f} > ${total_capital:,.2f}"
        )

    # ── Logging ──────────────────────────────────────────
    head_cost = sum(a.shares * a.candidate.signal_price for a in head_allocations)
    tail_cost = sum(a.shares * a.candidate.signal_price for a in tail_allocations)
    deploy_pct = total_cost / total_capital * 100
    
    logger.info(
        f"HEAD/TAIL allocation: {len(final)} positions from {len(eligible)} eligible "
        f"(capital=${total_capital:,.0f})"
    )
    logger.info(
        f"  HEAD: {len(head_allocations)} positions, ${head_cost:,.0f} deployed "
        f"({head_cost / total_capital * 100:.1f}%)"
    )
    logger.info(
        f"  TAIL: {len(tail_allocations)} positions, ${tail_cost:,.0f} deployed "
        f"({tail_cost / total_capital * 100:.1f}%) "
        f"[slice_cap=${tail_position_cap:,.0f}, target_slots={target_slots}]"
    )
    logger.info(
        f"  Total: ${total_cost:,.0f} deployed "
        f"({deploy_pct:.1f}%), "
        f"${total_capital - total_cost:,.0f} undeployed"
    )

    skipped_head = len(head_candidates) - len(head_allocations)
    if skipped_head > 0:
        logger.info(f"HEAD: {skipped_head} candidates skipped (price=0, ADV caps, or min_shares={min_shares})")

    total_tail_skips = sum(skip_reasons.values())
    if total_tail_skips > 0:
        reasons_str = ", ".join(f"{k}: {v}" for k, v in skip_reasons.items() if v > 0)
        logger.info(f"TAIL skip reasons: {reasons_str}")

    if deploy_pct < 80.0:
        logger.warning(
            f"Deployment below 80% target: {deploy_pct:.1f}% deployed "
            f"(${total_cost:,.0f} of ${total_capital:,.0f})"
        )

    return final
