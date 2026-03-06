"""
Daily liquidity ranking generator.

Runs after market close (4:05 PM) to generate liquidity_ranking.json
for next day's universe selection.

Fetches prior-day bars from Alpaca and calculates dollar volume for ranking.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def generate_liquidity_ranking(
    broker,
    alpaca_data,
    output_path: Path,
) -> bool:
    """
    Generate liquidity ranking file from prior-day market data.
    
    Fetches daily bars for all tradable symbols and calculates:
    - prev_close: Prior day's closing price
    - prev_volume: Prior day's volume
    - dollar_volume: prev_close × prev_volume
    
    Args:
        broker: Alpaca TradingClient for getting asset list
        alpaca_data: AlpacaDataAdapter for getting bars
        output_path: Path to save liquidity_ranking.json
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("GENERATING LIQUIDITY RANKING (4:05 PM)")
    logger.info("=" * 80)
    
    try:
        # Get all tradable US equities from Alpaca Assets
        logger.info("Fetching tradable assets from Alpaca...")
        assets = broker.get_all_assets()
        
        # Filter to US equities that are active and tradable
        eligible_symbols = []
        for asset in assets:
            if (getattr(asset, 'asset_class', '') == 'us_equity' and
                getattr(asset, 'status', '') == 'active' and
                getattr(asset, 'tradable', False)):
                eligible_symbols.append(asset.symbol)
        
        logger.info(f"Found {len(eligible_symbols)} eligible US equity symbols")
        
        # Get recent bars for all symbols (batched)
        # Fetch last 5 calendar days to handle weekends/holidays
        logger.info("Fetching recent daily bars from Alpaca...")
        
        # Calculate date range (last 5 calendar days to ensure we get last trading day)
        today = datetime.now().date()
        start_date = today - timedelta(days=5)
        
        # Fetch bars in batches (Alpaca limits to ~100 symbols per request)
        batch_size = 100
        all_bars = {}
        
        for i in range(0, len(eligible_symbols), batch_size):
            batch = eligible_symbols[i:i+batch_size]
            logger.info(f"Fetching bars for batch {i//batch_size + 1}/{(len(eligible_symbols)-1)//batch_size + 1}")
            
            try:
                bars = alpaca_data.get_bars(
                    batch,
                    start=start_date,
                    end=today,
                    timeframe='1Day',
                )
                
                if bars:
                    all_bars.update(bars)
            
            except Exception as e:
                logger.warning(f"Failed to fetch bars for batch: {e}")
                continue
        
        logger.info(f"Fetched bars for {len(all_bars)} symbols")
        
        # Calculate dollar volume for each symbol
        liquidity_data = {}
        
        for symbol, bar_list in all_bars.items():
            if not bar_list:
                continue
            
            # Get most recent completed bar (last trading day, not calendar yesterday)
            # This handles weekends and holidays correctly
            bar = bar_list[-1]
            
            prev_close = float(bar.c) if hasattr(bar, 'c') else 0.0
            prev_volume = float(bar.v) if hasattr(bar, 'v') else 0.0
            
            if prev_close > 0 and prev_volume > 0:
                dollar_volume = prev_close * prev_volume
                
                liquidity_data[symbol] = {
                    'prev_close': prev_close,
                    'prev_volume': prev_volume,
                    'dollar_volume': dollar_volume,
                }
        
        logger.info(f"Calculated liquidity for {len(liquidity_data)} symbols")
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(liquidity_data, f, indent=2)
        
        logger.info(f"Saved liquidity ranking to {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        # Log top 10 by dollar volume
        sorted_symbols = sorted(
            liquidity_data.items(),
            key=lambda x: x[1]['dollar_volume'],
            reverse=True
        )
        
        logger.info("Top 10 by dollar volume:")
        for i, (symbol, data) in enumerate(sorted_symbols[:10], 1):
            logger.info(f"  {i}. {symbol}: ${data['dollar_volume']/1e9:.2f}B")
        
        logger.info("=" * 80)
        logger.info("LIQUIDITY RANKING GENERATION COMPLETE")
        logger.info("=" * 80)
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to generate liquidity ranking: {e}", exc_info=True)
        return False
