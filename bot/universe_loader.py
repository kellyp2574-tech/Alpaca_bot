"""
Local universe loader using Alpaca Assets API.

Proper architecture:
1. Build base universe from Alpaca /v2/assets (master asset catalog)
2. Cache asset list locally for fast startup
3. Filter to ~4,000 tradable US equities
4. Use market data (snapshots/quotes) only for ranking, not discovery

Separation of concerns:
- Assets API: Defines what symbols exist and are tradable
- Market Data API: Ranks and filters those symbols
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


def refresh_master_alpaca_assets(broker) -> List[Dict[str, Any]]:
    """
    Fetch master asset list from Alpaca /v2/assets API.
    
    This is the authoritative source for what symbols Alpaca supports.
    Call this once per day (or week) and cache the results.
    
    Args:
        broker: Alpaca TradingClient instance
    
    Returns:
        List of asset dicts with keys: symbol, name, exchange, status, tradable, etc.
    """
    logger.info("Fetching master asset list from Alpaca /v2/assets...")
    
    try:
        # Get all assets from Alpaca
        assets = broker.get_all_assets()
        
        # Convert to dicts for caching
        asset_dicts = []
        for asset in assets:
            asset_dicts.append({
                'symbol': asset.symbol,
                'name': getattr(asset, 'name', ''),
                'exchange': getattr(asset, 'exchange', ''),
                'status': getattr(asset, 'status', ''),
                'tradable': getattr(asset, 'tradable', False),
                'asset_class': getattr(asset, 'asset_class', ''),
            })
        
        logger.info(f"Fetched {len(asset_dicts)} assets from Alpaca")
        return asset_dicts
    
    except Exception as e:
        logger.error(f"Failed to fetch Alpaca assets: {e}")
        return []


def load_master_assets_cache(cache_path: Path) -> List[Dict[str, Any]]:
    """
    Load cached Alpaca asset list from local file.
    
    Args:
        cache_path: Path to cached asset JSON file
    
    Returns:
        List of asset dicts, or empty list if cache missing/invalid
    """
    try:
        if not cache_path.exists():
            logger.info(f"Asset cache not found: {cache_path}")
            return []
        
        with open(cache_path, 'r') as f:
            data = json.load(f)
        
        assets = data.get('assets', [])
        cached_date = data.get('cached_date', '')
        
        logger.info(f"Loaded {len(assets)} assets from cache (date: {cached_date})")
        return assets
    
    except Exception as e:
        logger.error(f"Failed to load asset cache: {e}")
        return []


def save_master_assets_cache(cache_path: Path, assets: List[Dict[str, Any]]) -> None:
    """
    Save Alpaca asset list to local cache file.
    
    Args:
        cache_path: Path to cache file
        assets: List of asset dicts to save
    """
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'cached_date': datetime.now().isoformat(),
            'asset_count': len(assets),
            'assets': assets,
        }
        
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(assets)} assets to cache: {cache_path}")
    
    except Exception as e:
        logger.error(f"Failed to save asset cache: {e}")


def is_cache_stale(cache_path: Path, max_age_days: int = 7) -> bool:
    """
    Check if asset cache is stale and needs refresh.
    
    Args:
        cache_path: Path to cache file
        max_age_days: Maximum age in days before cache is stale
    
    Returns:
        True if cache is stale or missing
    """
    if not cache_path.exists():
        return True
    
    try:
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        is_stale = age > timedelta(days=max_age_days)
        
        if is_stale:
            logger.info(f"Asset cache is {age.days} days old (stale)")
        else:
            logger.info(f"Asset cache is {age.days} days old (fresh)")
        
        return is_stale
    
    except Exception as e:
        logger.warning(f"Failed to check cache age: {e}")
        return True


def _is_preferred_or_weird_class(symbol: str) -> bool:
    """
    Check if symbol is a preferred share or weird share class.
    
    Common patterns:
    - Ends with .P, .PR, -P, -PR (preferred)
    - Has multiple dots or dashes
    - Contains /WS, /WT, /U (warrants, units)
    - Ends with share class beyond A/B/C (e.g., .D, .E)
    
    Args:
        symbol: Stock symbol to check
    
    Returns:
        True if symbol looks like preferred/weird class
    """
    symbol_upper = symbol.upper()
    
    # Preferred patterns
    if any(symbol_upper.endswith(suffix) for suffix in ['.P', '.PR', '-P', '-PR', '_P', '_PR']):
        return True
    
    # Warrants, units, rights
    if any(pattern in symbol_upper for pattern in ['/WS', '/WT', '/U', '/R', '.WS', '.WT', '.U', '.R']):
        return True
    
    # Multiple dots or dashes (usually weird classes)
    if symbol.count('.') > 1 or symbol.count('-') > 1:
        return True
    
    # Weird share classes (beyond common A/B/C)
    if any(symbol_upper.endswith(suffix) for suffix in ['.D', '.E', '.F', '-D', '-E', '-F']):
        return True
    
    return False


def load_liquidity_cache(cache_dir: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """
    Load cached prior-day liquidity data for ranking.
    
    Expected file format:
    {
        "AAPL": {"prev_close": 172.23, "prev_volume": 53400000, "dollar_volume": 9197082000},
        "MSFT": {"prev_close": 418.56, "prev_volume": 22100000, "dollar_volume": 9240176000}
    }
    
    Args:
        cache_dir: Directory containing liquidity cache (default: state/universe/)
    
    Returns:
        Dict mapping symbol to liquidity stats, or empty dict if unavailable
    """
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parents[1] / "state" / "universe"
    
    cache_path = cache_dir / "liquidity_ranking.json"
    
    try:
        if not cache_path.exists():
            logger.info(f"Liquidity cache not found: {cache_path}")
            return {}
        
        with open(cache_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded liquidity data for {len(data)} symbols")
        return data
    
    except Exception as e:
        logger.warning(f"Failed to load liquidity cache: {e}")
        return {}


def build_daily_universe(
    master_assets: List[Dict[str, Any]],
    target_size: int = 4000,
    *,
    liquidity_cache: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[str]:
    """
    Build daily trading universe from master Alpaca asset list.
    
    Filters:
    - US equities only (asset_class = 'us_equity')
    - Active and tradable
    - Not OTC
    - Not preferred shares or weird share classes
    
    Selection:
    - Ranks by prior-day dollar volume (prev_close × prev_volume)
    - Selects top N by liquidity
    - Falls back to alphabetical if liquidity data unavailable
    
    Args:
        master_assets: Full Alpaca asset list
        target_size: Target universe size (default 4000)
        liquidity_cache: Optional liquidity data for ranking
    
    Returns:
        List of symbol strings for daily universe, ranked by liquidity
    """
    logger.info(f"Building daily universe from {len(master_assets)} master assets (target: {target_size})")
    
    filtered = []
    
    for asset in master_assets:
        symbol = asset.get('symbol', '')
        asset_class = asset.get('asset_class', '').lower()
        status = asset.get('status', '').lower()
        tradable = asset.get('tradable', False)
        exchange = asset.get('exchange', '').upper()
        
        # Filter 1: US equities only
        if asset_class != 'us_equity':
            continue
        
        # Filter 2: Active and tradable
        if status != 'active' or not tradable:
            continue
        
        # Filter 3: No OTC
        if 'OTC' in exchange or exchange == 'OTC':
            continue
        
        # Filter 4: No preferred shares or weird share classes
        if _is_preferred_or_weird_class(symbol):
            continue
        
        filtered.append(symbol)
    
    logger.info(f"After filtering: {len(filtered)} eligible US equities")
    
    # Rank by liquidity if data available
    if liquidity_cache:
        logger.info("Ranking symbols by prior-day dollar volume...")
        
        # Create list of (symbol, dollar_volume) tuples
        ranked = []
        for symbol in filtered:
            liq_data = liquidity_cache.get(symbol, {})
            dollar_volume = liq_data.get('dollar_volume', 0.0)
            ranked.append((symbol, dollar_volume))
        
        # Sort by dollar volume descending (highest liquidity first)
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        # Extract symbols
        result = [symbol for symbol, _ in ranked[:target_size]]
        
        # Log ranking stats
        if len(ranked) > target_size:
            top_symbol, top_vol = ranked[0]
            cutoff_symbol, cutoff_vol = ranked[target_size - 1]
            logger.info(f"Selected top {len(result)} by liquidity from {len(filtered)} eligible")
            logger.info(f"Top: {top_symbol} (${top_vol/1e9:.2f}B), Cutoff: {cutoff_symbol} (${cutoff_vol/1e9:.2f}B)")
        else:
            logger.info(f"Selected all {len(result)} eligible symbols (ranked by liquidity)")
    
    else:
        # Fallback to alphabetical if no liquidity data
        logger.warning("No liquidity data available, falling back to alphabetical selection")
        logger.warning("For better universe quality, populate state/universe/liquidity_ranking.json")
        filtered.sort()
        result = filtered[:target_size]
        
        if len(filtered) > target_size:
            logger.info(f"Selected {len(result)} symbols from {len(filtered)} eligible (alphabetical fallback)")
        else:
            logger.info(f"Selected all {len(result)} eligible symbols")
    
    return result


def get_or_refresh_master_assets(
    broker,
    cache_path: Path,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """
    Get master Alpaca asset list, using cache when available.
    
    Logic:
    1. If cache exists and fresh: load from cache
    2. If cache stale or missing: refresh from Alpaca API
    3. If API fails: use stale cache as fallback
    
    Args:
        broker: Alpaca TradingClient
        cache_path: Path to asset cache file
        force_refresh: Force refresh even if cache is fresh
    
    Returns:
        List of asset dicts
    """
    # Check cache freshness
    cache_is_stale = is_cache_stale(cache_path) or force_refresh
    
    if not cache_is_stale:
        # Cache is fresh, use it
        assets = load_master_assets_cache(cache_path)
        if assets:
            return assets
        else:
            logger.warning("Cache load failed, will refresh from API")
    
    # Cache is stale or missing, refresh from API
    logger.info("Refreshing master asset list from Alpaca API...")
    assets = refresh_master_alpaca_assets(broker)
    
    if assets:
        # Save fresh assets to cache
        save_master_assets_cache(cache_path, assets)
        return assets
    else:
        # API failed, try to use stale cache as fallback
        logger.warning("API refresh failed, attempting to use stale cache...")
        assets = load_master_assets_cache(cache_path)
        if assets:
            logger.warning(f"Using stale cache with {len(assets)} assets (degraded mode)")
            return assets
        else:
            logger.error("No assets available (API failed and no cache)")
            return []


def build_universe(
    broker,
    target_size: int = 4000,
    *,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> List[str]:
    """
    Build daily trading universe from Alpaca Assets API.
    
    This is the main entry point for universe building.
    
    Note:
        Price filtering is NOT done here (Assets API doesn't include prices).
        Price filtering happens in Stage 1 snapshot filtering.
    
    Args:
        broker: Alpaca TradingClient
        target_size: Target universe size (default 4000)
        cache_dir: Directory for asset cache (default: state/universe/)
        force_refresh: Force refresh even if cache is fresh
    
    Returns:
        List of symbol strings
    """
    # Set up cache path
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parents[1] / "state" / "universe"
    
    cache_path = cache_dir / "alpaca_assets_us_equity.json"
    
    # Get master asset list (from cache or API)
    master_assets = get_or_refresh_master_assets(broker, cache_path, force_refresh)
    
    if not master_assets:
        logger.error("Failed to get master assets, universe will be empty")
        return []
    
    # Load liquidity cache for ranking
    liquidity_cache = load_liquidity_cache(cache_dir)
    
    # Build daily universe from master assets
    universe = build_daily_universe(
        master_assets,
        target_size=target_size,
        liquidity_cache=liquidity_cache,
    )
    
    return universe
