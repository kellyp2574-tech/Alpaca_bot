"""
Morning Candidate Tester - Replicates the bot's candidate selection logic
without placing any actual orders.

This script performs Steps 1 and 2 of the gap momentum strategy:
1. Build universe from Massive (price filter: $0.50-$5.00)
2. Find gap candidates (3%+ gaps, $5M+ ADV) and split into core/filler

Outputs a detailed report of candidates that would be selected for trading.
"""
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import config
from bot.massive_client import MassiveClient
from bot.gap_calculator import GapCalculator, GapCandidate
from bot.vix_fetcher import VIXFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.LOG_DIR, "candidate_tester.log"))
    ]
)
logger = logging.getLogger(__name__)


def format_dollar_amount(amount: float) -> str:
    """Format dollar amount in millions or thousands."""
    if amount >= 1_000_000:
        return f"${amount/1e6:.2f}M"
    elif amount >= 1_000:
        return f"${amount/1e3:.1f}K"
    else:
        return f"${amount:.2f}"


def run_candidate_test() -> Tuple[List[GapCandidate], List[GapCandidate], List[GapCandidate]]:
    """
    Run the full candidate selection pipeline.
    
    Returns:
        Tuple of (all_candidates, core_candidates, filler_candidates)
    """
    logger.info("=" * 80)
    logger.info("MORNING CANDIDATE TESTER - Starting candidate selection pipeline")
    logger.info("=" * 80)
    
    # Initialize components
    massive = MassiveClient()
    gap_calc = GapCalculator()
    vix_fetcher = VIXFetcher()
    
    # Get current VIX
    vix_level = vix_fetcher.get_vix_level() or 15.0
    logger.info(f"Current VIX level: {vix_level:.2f}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # STEP 1: Build Universe
    # ═══════════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Building universe from Massive")
    logger.info("=" * 80)
    
    snapshots = massive.get_full_market_snapshot()
    if not snapshots:
        logger.error("Failed to fetch Massive snapshot")
        return [], [], []
    
    logger.info(f"Fetched {len(snapshots)} symbols from Massive")
    
    # Filter by price range ($0.50 - $5.00)
    universe = massive.filter_by_price_range(
        snapshots, config.MIN_PRICE, config.MAX_PRICE
    )
    logger.info(f"Universe after price filter (${config.MIN_PRICE}-${config.MAX_PRICE}): {len(universe)} symbols")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # STEP 2: Find Candidates
    # ═══════════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Finding gap candidates")
    logger.info("=" * 80)
    
    # Refresh Massive snapshot (critical - don't use stale data)
    logger.info("Refreshing Massive snapshot for candidate selection...")
    fresh_snapshots = massive.get_full_market_snapshot()
    if not fresh_snapshots:
        logger.error("Failed to refresh Massive snapshot")
        return [], [], []
    
    # Filter to universe only
    filtered_snapshots = {
        symbol: fresh_snapshots[symbol]
        for symbol in universe
        if symbol in fresh_snapshots
    }
    logger.info(f"Filtered to {len(filtered_snapshots)} universe symbols from Massive")
    
    # Find all candidates (gap >= 3%, ADV >= $5M)
    all_candidates = gap_calc.find_candidates(filtered_snapshots)
    
    if not all_candidates:
        logger.warning("No candidates found matching criteria")
        return [], [], []
    
    # Split candidates: core (4%+) vs filler (3-4%)
    core_candidates = [c for c in all_candidates if c.gap_pct >= 4.0]
    filler_candidates = [c for c in all_candidates if 3.0 <= c.gap_pct < 4.0]
    
    logger.info(f"\nCandidate split:")
    logger.info(f"  Total: {len(all_candidates)}")
    logger.info(f"  Core (4%+): {len(core_candidates)}")
    logger.info(f"  Filler (3-4%): {len(filler_candidates)}")
    
    # Apply liquidity filter separately
    logger.info(f"\nApplying liquidity filter (max_positions={config.MAX_POSITIONS})...")
    
    core_candidates = gap_calc.select_by_liquidity_and_gap(
        core_candidates, max_positions=config.MAX_POSITIONS
    )
    filler_candidates = gap_calc.select_by_liquidity_and_gap(
        filler_candidates, max_positions=config.MAX_POSITIONS
    )
    
    # Final combined list
    final_candidates = core_candidates + filler_candidates
    
    logger.info(f"\nFinal candidates after liquidity filter:")
    logger.info(f"  Core (4%+): {len(core_candidates)}")
    logger.info(f"  Filler (3-4%): {len(filler_candidates)}")
    logger.info(f"  Total: {len(final_candidates)}")
    
    return final_candidates, core_candidates, filler_candidates


def print_candidate_report(
    all_candidates: List[GapCandidate],
    core_candidates: List[GapCandidate],
    filler_candidates: List[GapCandidate]
):
    """Print a formatted candidate report."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build report
    report_lines = []
    report_lines.append("\n" + "=" * 80)
    report_lines.append(f"CANDIDATE SELECTION REPORT - {timestamp}")
    report_lines.append("=" * 80)
    report_lines.append(f"\nSUMMARY:")
    report_lines.append(f"  Total Candidates: {len(all_candidates)}")
    report_lines.append(f"  Core (4%+ gaps): {len(core_candidates)}")
    report_lines.append(f"  Filler (3-4% gaps): {len(filler_candidates)}")
    report_lines.append(f"  Max Positions: {config.MAX_POSITIONS}")
    report_lines.append(f"  Min Gap: {config.MIN_GAP_PCT}%")
    report_lines.append(f"  Max Gap: {config.MAX_GAP_PCT}%")
    report_lines.append(f"  Min ADV: {format_dollar_amount(config.MIN_ADV_DOLLARS)}")
    
    # Core candidates table
    if core_candidates:
        report_lines.append("\n" + "-" * 80)
        report_lines.append(f"CORE CANDIDATES (4%+ gaps) - Sorted by Liquidity (ADV):")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Rank':<6} {'Symbol':<8} {'Gap %':<10} {'Open $':<10} {'Prev $':<10} {'ADV':<12} {'Volume':<12}")
        report_lines.append("-" * 80)
        
        for i, c in enumerate(core_candidates, 1):
            report_lines.append(
                f"{i:<6} {c.symbol:<8} {c.gap_pct:>+8.2f}%  "
                f"${c.open_price:<8.2f} ${c.prev_close:<8.2f} "
                f"{format_dollar_amount(c.adv_estimate):<12} {c.volume:>10,}"
            )
    
    # Filler candidates table
    if filler_candidates:
        report_lines.append("\n" + "-" * 80)
        report_lines.append(f"FILLER CANDIDATES (3-4% gaps) - Sorted by Liquidity (ADV):")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Rank':<6} {'Symbol':<8} {'Gap %':<10} {'Open $':<10} {'Prev $':<10} {'ADV':<12} {'Volume':<12}")
        report_lines.append("-" * 80)
        
        for i, c in enumerate(filler_candidates, 1):
            report_lines.append(
                f"{i:<6} {c.symbol:<8} {c.gap_pct:>+8.2f}%  "
                f"${c.open_price:<8.2f} ${c.prev_close:<8.2f} "
                f"{format_dollar_amount(c.adv_estimate):<12} {c.volume:>10,}"
            )
    
    # Position sizing preview
    report_lines.append("\n" + "-" * 80)
    report_lines.append("POSITION SIZING PREVIEW (if entering today):")
    report_lines.append("-" * 80)
    
    # Estimate total positions that would be entered
    total_core = len(core_candidates)
    total_slots = config.MAX_POSITIONS
    remaining_slots = max(0, total_slots - total_core)
    filler_used = min(len(filler_candidates), remaining_slots)
    
    report_lines.append(f"  Core positions: {total_core}")
    report_lines.append(f"  Remaining slots: {remaining_slots}")
    report_lines.append(f"  Filler positions (conditional): {filler_used}")
    report_lines.append(f"  Total positions (estimated): {total_core + filler_used}")
    
    # Deployment logic preview
    report_lines.append("\n" + "-" * 80)
    report_lines.append("PHASED DEPLOYMENT LOGIC:")
    report_lines.append("-" * 80)
    report_lines.append("  Phase 1: Deploy to ALL core candidates (4%+ gaps)")
    report_lines.append("  Phase 2: Check deployment ratio vs total capital")
    report_lines.append("  Phase 3: If deployment < 80% and slots remain, deploy to filler candidates")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Print and save report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save to file
    report_file = os.path.join(
        config.STATE_DIR, 
        f"candidate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Report saved to: {report_file}")


def save_candidates_json(
    all_candidates: List[GapCandidate],
    core_candidates: List[GapCandidate],
    filler_candidates: List[GapCandidate]
):
    """Save candidates to JSON for programmatic access."""
    import json
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_candidates": len(all_candidates),
            "core_count": len(core_candidates),
            "filler_count": len(filler_candidates),
            "max_positions": config.MAX_POSITIONS
        },
        "core_candidates": [
            {
                "symbol": c.symbol,
                "open_price": c.open_price,
                "prev_close": c.prev_close,
                "gap_pct": c.gap_pct,
                "volume": c.volume,
                "adv_estimate": c.adv_estimate
            }
            for c in core_candidates
        ],
        "filler_candidates": [
            {
                "symbol": c.symbol,
                "open_price": c.open_price,
                "prev_close": c.prev_close,
                "gap_pct": c.gap_pct,
                "volume": c.volume,
                "adv_estimate": c.adv_estimate
            }
            for c in filler_candidates
        ]
    }
    
    json_file = os.path.join(
        config.STATE_DIR,
        f"candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Candidates JSON saved to: {json_file}")


def main():
    """Main entry point."""
    logger.info("Morning Candidate Tester - Starting...")
    
    # Check for API keys
    if not config.MASSIVE_API_KEY:
        logger.error("MASSIVE_API_KEY not set in environment!")
        print("\nERROR: Please set MASSIVE_API_KEY in your .env file or environment variables.")
        sys.exit(1)
    
    try:
        # Run candidate selection
        all_candidates, core_candidates, filler_candidates = run_candidate_test()
        
        # Print report
        print_candidate_report(all_candidates, core_candidates, filler_candidates)
        
        # Save JSON
        save_candidates_json(all_candidates, core_candidates, filler_candidates)
        
        logger.info("\nCandidate test complete!")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error in candidate tester: {e}")
        raise


if __name__ == "__main__":
    main()
