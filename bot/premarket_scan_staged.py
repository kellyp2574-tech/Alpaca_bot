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
    seed_symbols: List[str],
    max_candidates: int,
    stage_name: str,
    ledger: Optional[CandidateLedger] = None,
) -> List[Candidate]:
    """
    Build candidates from snapshot data (shared logic for all stages).
    
    Extracts prev_close from Alpaca snapshot.prev_daily_bar (no Massive dependency).
    
    Args:
        cfg: Configuration
        snapshots: Dict of symbol -> snapshot from Alpaca
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
        
        # Get prev close from Alpaca snapshot.prev_daily_bar (no Massive dependency)
        prev_close = 0.0
        if snap.prev_daily_bar is not None:
            prev_close = _safe_float(snap.prev_daily_bar.c)
        
        if prev_close <= 0:
            drops.append(Drop(sym, stage_name, "no_prev_close_from_snapshot", {}))
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
    seed_symbols: List[str],
    date: datetime,
) -> StagedScanResult:
    """
    Stage 1: 8:30-8:40 AM - Broad filter using delayed_sip.
    
    Takes pre-built universe and filters using delayed_sip snapshots.
    Target: 800 candidates after broad filter.
    
    Args:
        cfg: Configuration
        alpaca: AlpacaDataAdapter
        seed_symbols: Pre-built universe (from Alpaca Assets API)
        date: Current date
    
    Returns:
        StagedScanResult with ~800 candidates
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: Broad Filter (delayed_sip) - 8:30-8:40 AM")
    logger.info("=" * 80)
    
    run_date = date.date().isoformat()
    ledger = CandidateLedger(run_date=run_date)
    
    ledger.seed_total = len(seed_symbols)
    ledger.seed_selected = len(seed_symbols)
    
    logger.info(f"Universe built: {len(seed_symbols)} symbols (from local sources)")
    
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
    
    # Build candidates from snapshots (prev_close extracted from snapshots)
    max_first_pool = cfg.first_filter_pool_size
    candidates = _build_candidates_from_snapshots(
        cfg,
        snapshots,
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
    stage1_candidates: List[Candidate],
) -> StagedScanResult:
    """
    Stage 2: 9:05 AM - First live IEX refinement.
    
    Take ~800 candidates from stage 1, refresh with live IEX data.
    Target: 300 candidates after refinement.
    
    Args:
        cfg: Configuration
        alpaca: AlpacaDataAdapter
        stage1_candidates: Candidates from stage 1
    
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
    
    # Build candidates from fresh IEX snapshots (prev_close from snapshots)
    max_candidates = cfg.max_candidates_returned
    candidates = _build_candidates_from_snapshots(
        cfg,
        snapshots,
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
    stage2_candidates: List[Candidate],
) -> StagedScanResult:
    """
    Stage 3: 9:15 AM - Second live IEX refinement.
    
    Take ~300 candidates from stage 2, refresh with latest IEX data.
    Final ranking and selection of top candidates.
    
    Args:
        cfg: Configuration
        alpaca: AlpacaDataAdapter
        stage2_candidates: Candidates from stage 2
    
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
    
    # Build final candidate list (prev_close from snapshots, no max limit)
    candidates = _build_candidates_from_snapshots(
        cfg,
        snapshots,
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


# REMOVED: build_candidates_staged() - Stale function with outdated signatures
# integrated_main.py now calls stage1/stage2/stage3 functions directly
# If you need a helper function, use the individual stage functions:
#   result1 = stage1_broad_filter_delayed_sip(cfg, alpaca, date)
#   result2 = stage2_first_iex_refinement(cfg, alpaca, result1.candidates)
#   result3 = stage3_second_iex_refinement(cfg, alpaca, result2.candidates)
