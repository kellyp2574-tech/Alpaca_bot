"""
Timeline stage helpers for the morning momentum bot.

Provides wait_for_timeline_stage() used by EntryLoop to pause
until the IEX stream start time (9:28 AM).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from .clock import market_now
from .morning_config import Config

logger = logging.getLogger(__name__)


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
