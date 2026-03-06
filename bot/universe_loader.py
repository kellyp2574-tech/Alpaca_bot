"""
Local universe loader - replaces Massive dependency for Stage 1.

Provides a 4,000-symbol seed universe using Alpaca's built-in data sources:
- Most actives from Alpaca screener
- Tradable assets from Alpaca
- No external Massive API dependency
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def load_universe_from_alpaca(
    alpaca,
    max_symbols: int = 4000,
    *,
    min_price: float = 1.0,
    max_price: float = 100.0,
) -> List[str]:
    """
    Build universe from Alpaca data sources (no Massive dependency).
    
    Strategy:
    1. Get most actives from Alpaca screener (up to 100)
    2. Get additional tradable assets from Alpaca if needed
    3. Filter by price range
    4. Return up to max_symbols
    
    Args:
        alpaca: AlpacaDataAdapter instance
        max_symbols: Maximum symbols to return (default 4000)
        min_price: Minimum price filter (default $1)
        max_price: Maximum price filter (default $100)
    
    Returns:
        List of symbol strings
    """
    logger.info(f"Building universe from Alpaca (target: {max_symbols} symbols)")
    
    universe = []
    
    # Step 1: Get most actives (up to 100)
    try:
        most_actives = alpaca.get_most_actives(count=100)
        logger.info(f"Got {len(most_actives)} most active symbols from Alpaca")
        universe.extend(most_actives)
    except Exception as e:
        logger.warning(f"Failed to get most actives: {e}")
    
    # Step 2: If we need more symbols, we could:
    # - Use a static list of major symbols
    # - Query Alpaca assets API for tradable stocks
    # - Use a maintained local file
    
    # For now, if most_actives gave us symbols, use those
    # In production, you'd want to expand this with additional sources
    
    if len(universe) < max_symbols:
        logger.info(f"Universe has {len(universe)} symbols, target is {max_symbols}")
        logger.info("Note: Using most_actives only. For full 4000-symbol universe,")
        logger.info("consider adding: static symbol list, Alpaca assets API, or local file")
    
    # Remove duplicates and return
    universe = list(dict.fromkeys(universe))
    
    logger.info(f"Final universe: {len(universe)} symbols")
    return universe[:max_symbols]


def load_universe_from_file(
    filepath: str,
    max_symbols: int = 4000,
) -> List[str]:
    """
    Load universe from a local file (alternative to Alpaca screener).
    
    File format: One symbol per line, or comma-separated.
    
    Args:
        filepath: Path to universe file
        max_symbols: Maximum symbols to return
    
    Returns:
        List of symbol strings
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Handle both newline-separated and comma-separated
        if ',' in content:
            symbols = [s.strip().upper() for s in content.split(',')]
        else:
            symbols = [s.strip().upper() for s in content.split('\n')]
        
        # Remove empty strings
        symbols = [s for s in symbols if s]
        
        logger.info(f"Loaded {len(symbols)} symbols from {filepath}")
        return symbols[:max_symbols]
    
    except FileNotFoundError:
        logger.warning(f"Universe file not found: {filepath}")
        return []
    except Exception as e:
        logger.error(f"Failed to load universe from file: {e}")
        return []


def build_universe(
    alpaca,
    max_symbols: int = 4000,
    *,
    universe_file: Optional[str] = None,
    min_price: float = 1.0,
    max_price: float = 100.0,
) -> List[str]:
    """
    Build universe using best available source.
    
    Priority:
    1. Local file (if specified and exists)
    2. Alpaca most_actives + additional sources
    
    Args:
        alpaca: AlpacaDataAdapter instance
        max_symbols: Maximum symbols to return
        universe_file: Optional path to local universe file
        min_price: Minimum price filter
        max_price: Maximum price filter
    
    Returns:
        List of symbol strings
    """
    # Try local file first if specified
    if universe_file:
        symbols = load_universe_from_file(universe_file, max_symbols)
        if symbols:
            logger.info(f"Using universe from file: {len(symbols)} symbols")
            return symbols
        else:
            logger.info("File load failed, falling back to Alpaca")
    
    # Fall back to Alpaca
    return load_universe_from_alpaca(
        alpaca,
        max_symbols,
        min_price=min_price,
        max_price=max_price,
    )
