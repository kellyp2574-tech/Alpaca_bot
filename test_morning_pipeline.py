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
        return None, {}
    
    logger.info(f"Total snapshots retrieved: {len(snapshots)}")
    
    # Filter by price range
    logger.info(f"Filtering by price range: ${config.MIN_PRICE:.2f} - ${config.MAX_PRICE:.2f}")
    universe = massive.filter_by_price_range(snapshots, config.MIN_PRICE, config.MAX_PRICE)
    
    logger.info(f"Universe size after price filter: {len(universe)} symbols")
    
    if universe:
        logger.info(f"Sample universe symbols: {universe[:10]}")
    else:
        logger.error("WARNING: Universe is empty!")
    
    # Return both universe list and full Massive data for ADV
    return universe, snapshots


def test_gap_calculation(universe, massive_data):
    """Step 2: Test gap calculation and candidate selection."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Testing Gap Calculation")
    logger.info("=" * 60)
    
    if not universe:
        logger.error("Cannot calculate gaps - no universe available")
        return [], 15.0
    
    alpaca = AlpacaDataClient()
    gap_calc = GapCalculator()
    vix_fetcher = VIXFetcher()
    
    # Get Alpaca snapshots for universe
    logger.info(f"Fetching Alpaca snapshots for {len(universe)} universe symbols...")
    snapshots = alpaca.get_snapshots(universe)
    
    if not snapshots:
        logger.error("FAILED: Could not get Alpaca snapshots")
        return [], 15.0  # Return empty candidates and default VIX, 15.0  # Return empty candidates and default VIX
    
    # Merge Massive data (prev_volume, prev_close) with Alpaca snapshots
    logger.info("Merging Massive ADV data with Alpaca price data...")
    for symbol in snapshots:
        if symbol in massive_data:
            snapshots[symbol]["prev_volume"] = massive_data[symbol].get("prev_volume", 0)
            snapshots[symbol]["prev_close"] = massive_data[symbol].get("prev_close", 0)
            # Fallback: if Alpaca doesn't have prev_close, use Massive's
            if not snapshots[symbol].get("prev_close"):
                snapshots[symbol]["prev_close"] = massive_data[symbol].get("prev_close", 0)
    
    logger.info(f"Retrieved {len(snapshots)} snapshots with Massive ADV data")
    
    # Debug: Detailed gap analysis
    logger.info("\n--- Detailed Gap Analysis ---")
    
    # Sample some symbols to check their data
    sample_symbols = list(snapshots.keys())[:5]  # Just check 5
    gaps_calculated = []
    
    for symbol in sample_symbols:
        data = snapshots[symbol]
        open_price = data.get("open")
        prev_close = data.get("prev_close")
        prev_volume = data.get("prev_volume", 0)
        volume = data.get("volume", 0)
        close = data.get("close")
        last_price = data.get("last_price")
        
        # Show ALL fields
        logger.info(f"{symbol} raw data: open={open_price}, prev_close={prev_close}, prev_vol={prev_volume}, vol={volume}, close={close}, last={last_price}")
        
        # Calculate ADV
        if prev_volume and prev_close:
            adv = prev_volume * prev_close
        elif volume and open_price:
            adv = volume * open_price * 5
        else:
            adv = 0
        
        if open_price and prev_close and prev_close > 0:
            gap = ((open_price - prev_close) / prev_close) * 100
            gaps_calculated.append((symbol, gap, open_price, prev_close, volume, adv))
            logger.info(f"{symbol}: gap={gap:+.1f}%, adv=${adv/1e6:.2f}M")
        else:
            logger.info(f"{symbol}: MISSING DATA")
    
    # Show available fields from first snapshot
    if snapshots:
        first_sym = list(snapshots.keys())[0]
        logger.info(f"\nAvailable fields in snapshot: {list(snapshots[first_sym].keys())}")
    
    logger.info("--- End Detailed Analysis ---\n")
    
    # Find gap candidates
    logger.info("Calculating gaps and finding candidates...")
    candidates = gap_calc.find_candidates(snapshots)
    
    logger.info(f"Raw candidates found: {len(candidates)}")
    
    # Debug: Show gap distribution
    all_gaps = []
    for symbol, data in snapshots.items():
        open_price = data.get("open")
        prev_close = data.get("prev_close")
        if open_price and prev_close and prev_close > 0:
            gap = ((open_price - prev_close) / prev_close) * 100
            all_gaps.append(gap)
    
    if all_gaps:
        all_gaps.sort(key=abs, reverse=True)
        logger.info(f"Gap distribution (top 10): {[f'{g:+.1f}%' for g in all_gaps[:10]]}")
        logger.info(f"Max gap seen: {max(all_gaps, key=abs):+.1f}%")
        logger.info(f"Gaps >3%: {sum(1 for g in all_gaps if abs(g) >= 3)}")
        logger.info(f"Gaps >5%: {sum(1 for g in all_gaps if abs(g) >= 5)}")
    
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


def show_what_would_be_bought(candidates, vix_level, account_cash):
    """Display what the bot would buy at 9:30."""
    logger.info("\n" + "=" * 60)
    logger.info("WHAT WOULD BE BOUGHT AT 9:30")
    logger.info("=" * 60)
    
    if not candidates:
        logger.warning("No candidates to display - bot would not buy anything")
        return
    
    # Calculate dynamic position sizing
    num_positions = min(len(candidates), config.MAX_POSITIONS)
    position_size = account_cash / num_positions if num_positions > 0 else 0
    
    logger.info(f"VIX Level: {vix_level:.2f}")
    logger.info(f"Max Positions: {config.MAX_POSITIONS}")
    logger.info(f"Account Cash: ${account_cash:,.2f}")
    logger.info(f"Positions to Enter: {num_positions}")
    logger.info(f"Position Size (Cash / Positions): ${position_size:,.2f}")
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
        
        total_target = position_size * num_positions
        
        logger.info(f"Position Size: ${position_size:,.2f} (account cash / {num_positions} positions)")
        logger.info(f"Total Allocation: ${total_target:,.2f}")
        logger.info(f"Remaining Cash: ${account_cash - total_target:,.2f}")
    
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
    universe, massive_data = test_universe_build()
    
    if not universe:
        logger.error("\nFAILED: Could not build universe - check API connections")
        return 1
    
    # Step 2: Calculate gaps
    candidates, vix_level = test_gap_calculation(universe, massive_data)
    
    # Get Alpaca account cash for position sizing
    alpaca = AlpacaDataClient()
    account = alpaca.get_account()
    account_cash = float(account.get("cash", 0)) if account else 100000.0  # Fallback for testing
    
    # Show results
    show_what_would_be_bought(candidates, vix_level, account_cash)
    
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
