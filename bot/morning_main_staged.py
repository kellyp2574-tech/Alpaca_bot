"""
Staged candidate fetching for new timeline.

This module provides the orchestration for the multi-stage candidate selection:
- Stage 1 (8:30-8:40): delayed_sip broad filter
- Stage 2 (9:05): IEX first refinement  
- Stage 3 (9:15): IEX second refinement
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from .clock import market_now
from .morning_config import Config
from .premarket_scan_staged import build_candidates_staged
from .storage import Candidate

logger = logging.getLogger(__name__)


def fetch_candidates_staged(
    cfg: Config,
    data,
    *,
    current_time: datetime,
) -> Tuple[List[Candidate], Dict[str, any], Dict[str, float]]:
    """
    Fetch candidates using staged timeline with appropriate feeds.
    
    Timeline:
    - Before 8:40: Run stage 1 only (delayed_sip broad filter)
    - 8:40-9:05: Wait for 9:05
    - 9:05-9:15: Run stages 1+2 (add IEX first refinement)
    - 9:15+: Run all 3 stages (add IEX second refinement)
    
    Args:
        cfg: Configuration
        data: DataStack with alpaca and massive clients
        current_time: Current market time
    
    Returns:
        (candidates, stats, prev_close_map)
    """
    logger.info("=" * 80)
    logger.info("STAGED CANDIDATE FETCHING - NEW TIMELINE")
    logger.info("=" * 80)
    
    now_time = current_time.time()
    
    # Determine which stage to run based on current time
    stage1_time = datetime.strptime(cfg.broad_filter_end, "%H:%M").time()
    stage2_time = datetime.strptime(cfg.first_refinement, "%H:%M").time()
    stage3_time = datetime.strptime(cfg.second_refinement, "%H:%M").time()
    
    if now_time < stage1_time:
        # Before 8:40: Run stage 1 only
        logger.info(f"Current time {now_time.strftime('%H:%M')} - Running Stage 1 only (delayed_sip)")
        stage = 1
    elif now_time < stage2_time:
        # 8:40-9:05: Still run stage 1 (we'll wait for 9:05 in orchestrator)
        logger.info(f"Current time {now_time.strftime('%H:%M')} - Running Stage 1 (waiting for 9:05)")
        stage = 1
    elif now_time < stage3_time:
        # 9:05-9:15: Run stages 1+2
        logger.info(f"Current time {now_time.strftime('%H:%M')} - Running Stages 1+2 (IEX refinement)")
        stage = 2
    else:
        # 9:15+: Run all 3 stages
        logger.info(f"Current time {now_time.strftime('%H:%M')} - Running all 3 stages (full pipeline)")
        stage = 3
    
    # Run staged candidate building
    candidates, ledger, prev_close_map = build_candidates_staged(
        cfg,
        data.alpaca,
        data.massive,
        current_time,
        stage=stage,
    )
    
    # Save audit file
    try:
        BASE_DIR = Path(__file__).resolve().parents[1]
        report_path = BASE_DIR / "state" / "candidates" / f"{current_time.date().isoformat()}_stage{stage}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        ledger.save(report_path)
        logger.info(f"Saved candidate ledger (stage {stage}) → {report_path}")
    except Exception as e:
        logger.exception(f"Failed to save candidate ledger: {e}")
    
    # Build stats dict
    stats = {
        c.symbol: {
            "prev_close": c.prev_close,
            "gap_pct": c.gap_pct,
            "price": c.price,
            "pm_last": c.pm_last,
            "float_shares": c.float_shares,
            "pm_vol_float": c.pm_vol_float,
            "relvol": c.relvol,
            "score": c.score,
        }
        for c in candidates
    }
    
    logger.info(f"Staged fetch complete: {len(candidates)} candidates from stage {stage}")
    logger.info("=" * 80)
    
    return candidates, stats, prev_close_map


def wait_for_timeline_stage(cfg: Config, target_stage: str) -> None:
    """
    Wait until the specified timeline stage.
    
    Args:
        cfg: Configuration with timeline constants
        target_stage: One of 'first_refinement', 'second_refinement', 'candidate_freeze', 'stream_start'
    """
    stage_times = {
        'first_refinement': cfg.first_refinement,
        'second_refinement': cfg.second_refinement,
        'candidate_freeze': cfg.candidate_freeze,
        'stream_start': cfg.stream_start,
    }
    
    if target_stage not in stage_times:
        logger.warning(f"Unknown timeline stage: {target_stage}")
        return
    
    target_time_str = stage_times[target_stage]
    target_time = datetime.strptime(target_time_str, "%H:%M").time()
    
    while True:
        now = market_now()
        current_time = now.time()
        
        if current_time >= target_time:
            logger.info(f"Timeline stage '{target_stage}' reached at {current_time.strftime('%H:%M')}")
            break
        
        # Calculate wait time
        target_dt = now.replace(hour=target_time.hour, minute=target_time.minute, second=0, microsecond=0)
        wait_seconds = (target_dt - now).total_seconds()
        
        if wait_seconds > 60:
            logger.info(f"Waiting for {target_stage} at {target_time_str} ({wait_seconds/60:.1f} minutes)")
            time.sleep(60)  # Check every minute
        elif wait_seconds > 0:
            logger.info(f"Waiting {wait_seconds:.0f} seconds for {target_stage} at {target_time_str}")
            time.sleep(wait_seconds)
        else:
            break
