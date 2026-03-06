"""
Multi-stage premarket scanning pipeline with feed-specific data collection.

Timeline:
- 8:30-8:40 AM: Broad filter using delayed_sip (4,000 → 800)
- 9:05 AM: First IEX refinement (800 → 300)
- 9:15 AM: Second IEX refinement (300 → final ranking)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .morning_config import Config
from .premarket_scan import (
    CandidateLedger,
    Drop,
    seed_universe_massive,
    _safe_float,
)
from .storage import Candidate

logger = logging.getLogger(__name__)


@dataclass
class StagedScanResult:
    """Result from a staged scanning pass."""
    candidates: List[Candidate]
    ledger: CandidateLedger
    stage: str
    feed_used: str


def _build_candidates_from_snapshots(
    cfg: Config,
    snapshots: Dict,
    prev_close_map: Dict[str, float],
    seed_symbols: List[str],
    max_candidates: int,
    stage_name: str,
    ledger: Optional[CandidateLedger] = None,
) -> List[Candidate]:
    """
    Build candidates from snapshot data (shared logic for all stages).
    
    Args:
        cfg: Configuration
        snapshots: Dict of symbol -> snapshot from Alpaca
        prev_close_map: Dict of symbol -> prev_close from Massive
        seed_symbols: Symbols to process
        max_candidates: Max candidates to return
        stage_name: Stage name for drop tracking
        ledger: Optional ledger for tracking drops
    
    Returns:
        List of Candidate objects
    """
    drops: List[Drop] = []
    out: List[Candidate] = []
    
    logger.info(f"{stage_name}: Snapshot coverage {len(snapshots)}/{len(seed_symbols)} symbols")
    
    for sym in seed_symbols:
        snap = snapshots.get(sym)
        if snap is None:
            drops.append(Drop(sym, stage_name, "no_snapshot", {}))
            continue
        
        # Get prev close from Massive
        prev_close = prev_close_map.get(sym, 0.0)
        if prev_close <= 0:
            drops.append(Drop(sym, stage_name, "no_prev_close", {}))
            continue
        
        # Get current price (prefer quote mid)
        lq = getattr(snap, "latest_quote", None)
        lt = getattr(snap, "latest_trade", None)
        
        price_now = 0.0
        if lq is not None:
            bp = _safe_float(getattr(lq, "bp", 0) or getattr(lq, "bid_price", 0))
            ap = _safe_float(getattr(lq, "ap", 0) or getattr(lq, "ask_price", 0))
            if bp > 0 and ap > 0:
                price_now = (bp + ap) / 2.0
        
        if price_now <= 0 and lt is not None:
            price_now = _safe_float(getattr(lt, "p", 0) or getattr(lt, "price", 0))
        
        if price_now <= 0:
            drops.append(Drop(sym, stage_name, "no_price", {}))
            continue
        
        # Exclusion filter
        if sym in cfg.excluded_symbols:
            drops.append(Drop(sym, stage_name, "excluded_instrument", {}))
            continue
        
        # Price filter
        if not (cfg.min_price <= price_now <= cfg.max_price):
            drops.append(Drop(sym, stage_name, "price_out_of_range", {"price": price_now}))
            continue
        
        # Gap filter
        gap_pct = (price_now - prev_close) / prev_close
        if not (cfg.min_gap_pct <= gap_pct <= cfg.max_gap_pct):
            drops.append(Drop(sym, stage_name, "gap_out_of_range", {"gap_pct": gap_pct}))
            continue
        
        out.append(
            Candidate(
                symbol=sym,
                price=price_now,
                prev_close=prev_close,
                pm_last=price_now,
                pm_high=price_now,
                pm_volume=0,
                avg_vol_30d=0,
                float_shares=0,
                gap_pct=gap_pct,
                pm_vol_float=0,
                relvol=0,
                score=gap_pct,
            )
        )
        
        if len(out) >= max_candidates:
            break
    
    # Sort by gap %
    out.sort(key=lambda c: c.gap_pct, reverse=True)
    
    if ledger is not None:
        ledger.drops.extend(drops)
    
    return out


def stage1_broad_filter_delayed_sip(
    cfg: Config,
    alpaca,
    massive_client,
    date: datetime,
) -> StagedScanResult:
    """
    Stage 1: 8:30-8:40 AM - Broad filter using delayed_sip.
    
    Build 4,000-symbol universe from Massive, then filter using delayed_sip snapshots.
    Target: 800 candidates after broad filter.
    
    Returns:
        StagedScanResult with ~800 candidates
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: Broad Filter (delayed_sip) - 8:30-8:40 AM")
    logger.info("=" * 80)
    
    run_date = date.date().isoformat()
    ledger = CandidateLedger(run_date=run_date)
    
    # Build 4,000-symbol universe from Massive
    max_seed = cfg.max_seed_universe
    logger.info(f"Building {max_seed}-symbol universe from Massive...")
    
    seed_symbols, prev_close_map, meta, seed_drops = seed_universe_massive(
        massive_client,
        max_seed=max_seed,
        include_otc=False,
    )
    
    ledger.snapshot_count_seen = int(meta.get("snapshot_count_seen", 0))
    ledger.snapshot_count_with_prev_obj = int(meta.get("snapshot_count_with_prev_obj", 0))
    ledger.seed_total = int(meta.get("usable_snapshot_items", 0))
    ledger.seed_selected = len(seed_symbols)
    ledger.drops.extend(seed_drops)
    
    logger.info(f"Universe built: {len(seed_symbols)} symbols")
    
    # Get delayed_sip snapshots with fallback to IEX
    feed_used = cfg.universe_filter_feed
    logger.info(f"Fetching snapshots with feed={feed_used}...")
    
    try:
        snapshots = alpaca.get_snapshots(seed_symbols, feed=feed_used)
        logger.info(f"Successfully fetched {len(snapshots)} snapshots using {feed_used}")
    except Exception as e:
        logger.warning(f"delayed_sip unavailable ({e}), falling back to iex")
        feed_used = "iex"
        snapshots = alpaca.get_snapshots(seed_symbols, feed=feed_used)
        logger.info(f"Fetched {len(snapshots)} snapshots using fallback {feed_used}")
    
    # Build candidates from snapshots
    max_first_pool = cfg.first_filter_pool_size
    candidates = _build_candidates_from_snapshots(
        cfg,
        snapshots,
        prev_close_map,
        seed_symbols,
        max_first_pool,
        "stage1_delayed_sip",
        ledger,
    )
    
    ledger.validated = len(candidates)
    ledger.final = len(candidates)
    
    logger.info(f"Stage 1 complete: {len(candidates)} candidates (target: {max_first_pool})")
    logger.info(f"Feed used: {feed_used}")
    
    return StagedScanResult(
        candidates=candidates,
        ledger=ledger,
        stage="stage1_broad_filter",
        feed_used=feed_used,
    )


def stage2_first_iex_refinement(
    cfg: Config,
    alpaca,
    prev_close_map: Dict[str, float],
    stage1_candidates: List[Candidate],
) -> StagedScanResult:
    """
    Stage 2: 9:05 AM - First live IEX refinement.
    
    Take ~800 candidates from stage 1, refresh with live IEX data.
    Target: 300 candidates after refinement.
    
    Returns:
        StagedScanResult with ~300 candidates
    """
    logger.info("=" * 80)
    logger.info("STAGE 2: First IEX Refinement - 9:05 AM")
    logger.info("=" * 80)
    
    symbols = [c.symbol for c in stage1_candidates]
    logger.info(f"Refining {len(symbols)} candidates with live IEX data...")
    
    feed_used = cfg.preopen_refine_feed
    snapshots = alpaca.get_snapshots(symbols, feed=feed_used)
    logger.info(f"Fetched {len(snapshots)} snapshots using {feed_used}")
    
    # Build candidates from fresh IEX snapshots
    max_candidates = cfg.max_candidates_returned
    candidates = _build_candidates_from_snapshots(
        cfg,
        snapshots,
        prev_close_map,
        symbols,
        max_candidates,
        "stage2_iex_refine",
        ledger=None,
    )
    
    logger.info(f"Stage 2 complete: {len(candidates)} candidates (target: {max_candidates})")
    
    return StagedScanResult(
        candidates=candidates,
        ledger=CandidateLedger(run_date=""),  # Ledger not updated in refinement stages
        stage="stage2_first_refinement",
        feed_used=feed_used,
    )


def stage3_second_iex_refinement(
    cfg: Config,
    alpaca,
    prev_close_map: Dict[str, float],
    stage2_candidates: List[Candidate],
) -> StagedScanResult:
    """
    Stage 3: 9:15 AM - Second live IEX refinement.
    
    Take ~300 candidates from stage 2, refresh with latest IEX data.
    Final ranking and selection of top candidates.
    
    Returns:
        StagedScanResult with final ranked candidates
    """
    logger.info("=" * 80)
    logger.info("STAGE 3: Second IEX Refinement - 9:15 AM")
    logger.info("=" * 80)
    
    symbols = [c.symbol for c in stage2_candidates]
    logger.info(f"Final refinement of {len(symbols)} candidates with latest IEX data...")
    
    feed_used = cfg.final_preopen_refresh_feed
    snapshots = alpaca.get_snapshots(symbols, feed=feed_used)
    logger.info(f"Fetched {len(snapshots)} snapshots using {feed_used}")
    
    # Build final candidate list (no max limit, return all that pass filters)
    candidates = _build_candidates_from_snapshots(
        cfg,
        snapshots,
        prev_close_map,
        symbols,
        max_candidates=len(symbols),  # No limit, keep all that pass
        stage_name="stage3_iex_final",
        ledger=None,
    )
    
    logger.info(f"Stage 3 complete: {len(candidates)} final candidates")
    logger.info(f"Top candidates: {', '.join(c.symbol for c in candidates[:12])}")
    
    return StagedScanResult(
        candidates=candidates,
        ledger=CandidateLedger(run_date=""),
        stage="stage3_final_refinement",
        feed_used=feed_used,
    )


def build_candidates_staged(
    cfg: Config,
    alpaca,
    massive_client,
    date: datetime,
    *,
    stage: int = 3,
) -> Tuple[List[Candidate], CandidateLedger, Dict[str, float]]:
    """
    Multi-stage candidate building with feed-specific data collection.
    
    Args:
        cfg: Configuration
        alpaca: Alpaca data adapter
        massive_client: Massive API client
        date: Current date
        stage: Which stage to run up to (1, 2, or 3). Default 3 = full pipeline.
    
    Returns:
        (candidates, ledger, prev_close_map)
    """
    # Stage 1: Broad filter with delayed_sip (8:30-8:40)
    result1 = stage1_broad_filter_delayed_sip(cfg, alpaca, massive_client, date)
    
    if stage == 1:
        return result1.candidates, result1.ledger, {}
    
    # Extract prev_close_map from stage 1 candidates
    prev_close_map = {c.symbol: c.prev_close for c in result1.candidates}
    
    # Stage 2: First IEX refinement (9:05)
    result2 = stage2_first_iex_refinement(cfg, alpaca, prev_close_map, result1.candidates)
    
    if stage == 2:
        return result2.candidates, result1.ledger, prev_close_map
    
    # Stage 3: Second IEX refinement (9:15)
    result3 = stage3_second_iex_refinement(cfg, alpaca, prev_close_map, result2.candidates)
    
    return result3.candidates, result1.ledger, prev_close_map
