"""Morning Pipeline Tester - Test universe build and gap calculation without placing orders.

This script exercises Steps 1 and 2 of the gap momentum bot:
1. Build universe from Massive (with Alpaca fallback)
2. Calculate gaps and find candidates

It shows what the bot WOULD buy at 9:30, without actually placing orders.
"""
import logging
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import config
from bot.massive_client import MassiveClient
from bot.market_data import AlpacaDataClient
from bot.gap_calculator import GapCalculator
from bot.vix_fetcher import VIXFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def test_universe_build():
    """Step 1: Test universe building from Massive with Alpaca fallback."""
    logger.info("=" * 60)
    logger.info("STEP 1: Testing Universe Build")
    logger.info("=" * 60)
    
    massive = MassiveClient()
    
    # Try Massive first
    logger.info("Fetching market snapshot from Massive...")
    snapshots = massive.get_full_market_snapshot()
    
    if not snapshots:
        logger.error("Massive snapshot failed - would fall back to Alpaca")
        # Try Alpaca fallback
        alpaca = AlpacaDataClient()
        logger.info("Fetching tradable assets from Alpaca...")
        assets = alpaca.get_tradable_assets()
        logger.info(f"Got {len(assets)} assets from Alpaca")
        
        # Get snapshots for first 2000 assets
        logger.info("Fetching snapshots for price filtering...")
        snapshots = alpaca.get_snapshots(assets[:2000])
        
        if not snapshots:
            logger.error("FAILED: Could not get any snapshots from Massive or Alpaca")
            return None
    
    logger.info(f"Total snapshots retrieved: {len(snapshots)}")
    
    # Filter by price range
    logger.info(f"Filtering by price range: ${config.MIN_PRICE:.2f} - ${config.MAX_PRICE:.2f}")
    universe = massive.filter_by_price_range(snapshots, config.MIN_PRICE, config.MAX_PRICE)
    
    logger.info(f"Universe size after price filter: {len(universe)} symbols")
    
    if universe:
        logger.info(f"Sample universe symbols: {universe[:10]}")
    else:
        logger.error("WARNING: Universe is empty!")
    
    return universe


def test_gap_calculation(universe):
    """Step 2: Test gap calculation and candidate selection."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Testing Gap Calculation")
    logger.info("=" * 60)
    
    if not universe:
        logger.error("Cannot calculate gaps - no universe available")
        return []
    
    alpaca = AlpacaDataClient()
    gap_calc = GapCalculator()
    vix_fetcher = VIXFetcher()
    
    # Get Alpaca snapshots for universe
    logger.info(f"Fetching Alpaca snapshots for {len(universe)} universe symbols...")
    snapshots = alpaca.get_snapshots(universe)
    
    if not snapshots:
        logger.error("FAILED: Could not get Alpaca snapshots")
        return []
    
    logger.info(f"Retrieved {len(snapshots)} snapshots")
    
    # Find gap candidates
    logger.info("Calculating gaps and finding candidates...")
    candidates = gap_calc.find_candidates(snapshots)
    
    logger.info(f"Raw candidates found: {len(candidates)}")
    
    # Select top candidates
    selected = gap_calc.select_by_liquidity_and_gap(candidates, max_positions=20)
    
    logger.info(f"Selected candidates: {len(selected)}")
    
    # Get VIX level
    vix_level = vix_fetcher.get_vix_level()
    if vix_level:
        logger.info(f"VIX level: {vix_level:.2f}")
    else:
        logger.warning("Could not fetch VIX, using default")
        vix_level = 15.0
    
    return selected, vix_level


def show_what_would_be_bought(candidates, vix_level):
    """Display what the bot would buy at 9:30."""
    logger.info("\n" + "=" * 60)
    logger.info("WHAT WOULD BE BOUGHT AT 9:30")
    logger.info("=" * 60)
    
    if not candidates:
        logger.warning("No candidates to display - bot would not buy anything")
        return
    
    logger.info(f"VIX Level: {vix_level:.2f}")
    logger.info(f"Max Positions: {config.MAX_POSITIONS}")
    logger.info(f"Position Sizing: Base ${config.POSITION_SIZE_DOLLARS:,.0f} per position")
    logger.info("")
    
    # Group by gap direction
    longs = [c for c in candidates if c.gap_pct > 0]
    shorts = [c for c in candidates if c.gap_pct < 0]
    
    logger.info(f"LONG candidates (positive gap): {len(longs)}")
    logger.info(f"SHORT candidates (negative gap): {len(shorts)}")
    logger.info("")
    
    # Show top candidates
    logger.info("Top candidates by rank:")
    logger.info(f"{'Rank':<6} {'Symbol':<8} {'Gap%':>8} {'Price':>8} {'ADV($M)':>10} {'Action':<8}")
    logger.info("-" * 60)
    
    for i, c in enumerate(candidates[:config.MAX_POSITIONS], 1):
        action = "BUY" if c.gap_pct > 0 else "SELL"
        logger.info(
            f"{i:<6} {c.symbol:<8} {c.gap_pct:>+7.1f}% "
            f"${c.current_price:>7.2f} ${c.adv_estimate/1e6:>9.1f}M {action:<8}"
        )
    
    # Calculate approximate position sizing
    if candidates:
        logger.info("\n" + "-" * 60)
        logger.info("Estimated position sizing (portfolio-level allocation):")
        
        # Simple estimation
        target_dollars = config.POSITION_SIZE_DOLLARS
        total_target = target_dollars * min(len(candidates), config.MAX_POSITIONS)
        
        logger.info(f"Target per position: ${target_dollars:,.0f}")
        logger.info(f"Total estimated allocation: ${total_target:,.0f}")
        logger.info(f"Max positions allowed: {config.MAX_POSITIONS}")
    
    logger.info("\n" + "=" * 60)
    logger.info("DRY RUN COMPLETE - No orders were placed")
    logger.info("=" * 60)


def main():
    """Run the morning pipeline tester."""
    logger.info("\n" + "=" * 60)
    logger.info("MORNING PIPELINE TESTER")
    logger.info("Testing universe build + gap calculation (no orders)")
    logger.info("=" * 60)
    
    # Step 1: Build universe
    universe = test_universe_build()
    
    if not universe:
        logger.error("\nFAILED: Could not build universe - check API connections")
        return 1
    
    # Step 2: Calculate gaps
    candidates, vix_level = test_gap_calculation(universe)
    
    # Show results
    show_what_would_be_bought(candidates, vix_level)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Universe size: {len(universe)} symbols")
    logger.info(f"Candidates found: {len(candidates)}")
    logger.info(f"Would enter: {min(len(candidates), config.MAX_POSITIONS)} positions")
    logger.info("")
    
    if candidates:
        logger.info("Status: READY for market open")
    else:
        logger.info("Status: NO CANDIDATES - bot would sit idle")
    
    return 0


if __name__ == "__main__":
    exit(main())
