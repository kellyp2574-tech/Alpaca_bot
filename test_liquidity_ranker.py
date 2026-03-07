"""
Test script to manually run liquidity ranking generation.
"""

import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("TESTING LIQUIDITY RANKING GENERATION")
    logger.info("=" * 80)
    
    # Import dependencies
    from bot import alpaca_client as broker
    from bot.data_alpaca import AlpacaDataAdapter
    from bot.liquidity_ranker import generate_liquidity_ranking
    
    # Initialize data adapter
    logger.info("Initializing Alpaca data adapter...")
    alpaca_data = AlpacaDataAdapter()
    
    # Set output path
    output_path = Path(__file__).resolve().parent / "state" / "universe" / "liquidity_ranking.json"
    logger.info(f"Output path: {output_path}")
    
    # Generate ranking
    logger.info("Starting liquidity ranking generation...")
    success = generate_liquidity_ranking(
        broker.get_trading_client(),
        alpaca_data,
        output_path,
    )
    
    if success:
        logger.info("✅ Liquidity ranking generation SUCCESSFUL")
        logger.info(f"Output file: {output_path}")
        logger.info(f"File exists: {output_path.exists()}")
        if output_path.exists():
            logger.info(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    else:
        logger.error("❌ Liquidity ranking generation FAILED")
    
    logger.info("=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
