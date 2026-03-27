"""Alpaca Market Data client for IEX snapshots (Step 2: Signal Engine)"""
import logging
from typing import Dict, List, Optional, Set
import requests
from bot import config

logger = logging.getLogger(__name__)


class AlpacaDataClient:
    """Client for Alpaca Market Data API (IEX feed for free tier)"""

    def __init__(self):
        self.data_url = config.ALPACA_DATA_URL
        self.api_key = config.ALPACA_API_KEY
        self.secret_key = config.ALPACA_SECRET_KEY
        self.feed = config.DATA_FEED

        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        })

    def get_snapshots(self, symbols: List[str]) -> Dict[str, dict]:
        """
        Fetch IEX snapshots for given symbols.
        Max 1000 symbols per request (Alpaca limit).
        """
        if not symbols:
            return {}

        all_snapshots = {}
        batch_size = 1000

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            symbols_param = ",".join(batch)

            url = f"{self.data_url}/v2/stocks/snapshots"
            params = {
                "symbols": symbols_param,
                "feed": self.feed,
            }

            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Check if response is paginated (has 'NEXT' key) or direct snapshots
                if "snapshots" in data:
                    snapshots_map = data.get("snapshots", {})
                elif "NEXT" in data:
                    # Paginated response - data might be directly under symbol keys
                    logger.warning(f"Paginated response detected. Raw first keys: {list(data.keys())[:5]}")
                    snapshots_map = {k: v for k, v in data.items() if k != "NEXT" and isinstance(v, dict)}
                else:
                    # Direct symbol:snapshot mapping
                    snapshots_map = {k: v for k, v in data.items() if isinstance(v, dict)}
                
                for symbol, snapshot in snapshots_map.items():
                    if snapshot and isinstance(snapshot, dict):
                        parsed = self._parse_snapshot(symbol, snapshot)
                        if parsed:
                            all_snapshots[symbol] = parsed
                    else:
                        logger.debug(f"Empty/invalid snapshot for {symbol}: {type(snapshot)}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Alpaca snapshot error for batch {i//batch_size + 1}: {e}")

        logger.info(f"Alpaca snapshots: {len(all_snapshots)} symbols")
        return all_snapshots

    def _parse_snapshot(self, symbol: str, data: dict) -> dict:
        """Parse Alpaca snapshot into standardized format"""
        daily_bar = data.get("dailyBar", {})
        prev_daily_bar = data.get("prevDailyBar", {})
        latest_quote = data.get("latestQuote", {})
        latest_trade = data.get("latestTrade", {})

        # Extract fields with detailed logging for diagnostics
        open_price = daily_bar.get("o")
        prev_close = prev_daily_bar.get("c")
        last_price = latest_trade.get("p")
        
        # DIAGNOSTIC: Log suspicious data for key symbols or when fields are missing
        if symbol in ["SNAP", "FLY", "AAPL", "TSLA"] or open_price is None or prev_close is None:
            logger.info(
                f"SNAPSHOT DIAGNOSTIC {symbol}: "
                f"dailyBar={daily_bar}, "
                f"prevDailyBar={prev_daily_bar}, "
                f"latestTrade={latest_trade}"
            )
        
        # VALIDATION: Warn if dailyBar appears stale (open == close with zero volume)
        daily_volume = daily_bar.get("v", 0) if daily_bar else 0
        daily_close = daily_bar.get("c") if daily_bar else None
        if open_price and daily_close and open_price == daily_close and daily_volume == 0:
            logger.warning(
                f"STALE BAR WARNING {symbol}: open={open_price}, close={daily_close}, volume=0 - "
                f"dailyBar may not be populated yet (premarket or no trades)"
            )

        return {
            "symbol": symbol,
            "open": open_price,
            "high": daily_bar.get("h"),
            "low": daily_bar.get("l"),
            "close": daily_close,
            "volume": daily_volume,
            "vwap": daily_bar.get("vw"),
            "prev_close": prev_close,
            "prev_volume": prev_daily_bar.get("v"),
            "bid": latest_quote.get("bp"),
            "ask": latest_quote.get("ap"),
            "bid_size": latest_quote.get("bs"),
            "ask_size": latest_quote.get("as"),
            "last_price": last_price,
            "last_size": latest_trade.get("s"),
            "timestamp": latest_trade.get("t"),
        }

    def get_latest_trade(self, symbol: str) -> Optional[dict]:
        """Get latest trade for a single symbol"""
        url = f"{self.data_url}/v2/stocks/{symbol}/trades/latest"
        params = {"feed": self.feed}

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            trade = data.get("trade", {})

            return {
                "symbol": symbol,
                "price": trade.get("p"),
                "size": trade.get("s"),
                "timestamp": trade.get("t"),
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting latest trade for {symbol}: {e}")
            return None

    def get_regular_session_open(self, symbol: str) -> Optional[float]:
        """
        Get the true regular-session (9:30 AM ET) opening price for a symbol.
        Uses Alpaca's quotes endpoint with limit=1 to get the first RTH quote.
        This is more reliable than dailyBar.o at 9:30 which may include pre-market data.
        """
        url = f"{self.data_url}/v2/stocks/{symbol}/quotes"
        params = {
            "feed": self.feed,
            "limit": 1,
        }
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            quotes = data.get("quotes", [])
            if quotes:
                # First quote after 9:30 should be the opening quote
                first_quote = quotes[0]
                # Use ask price (offer) as the opening price - what you'd pay to enter
                open_price = first_quote.get("ap")
                logger.debug(f"Regular session open for {symbol}: ${open_price} (from first quote)")
                return open_price
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting regular session open for {symbol}: {e}")
            return None

    def get_tradable_assets(self) -> List[str]:
        """
        Get list of tradable assets from Alpaca API.
        Used as fallback when Massive universe build fails.
        """
        # Use Alpaca Trading API for assets (not Market Data API)
        base_url = config.ALPACA_BASE_URL
        url = f"{base_url}/v2/assets"
        params = {
            "status": "active",
            "tradable": "true",
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Filter for US equities only
            symbols = []
            for asset in data:
                if asset.get("class") == "us_equity" and asset.get("tradable") and asset.get("status") == "active":
                    symbols.append(asset.get("symbol"))

            logger.info(f"Alpaca assets: {len(symbols)} tradable US equities")
            return symbols

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Alpaca assets: {e}")
            return []
