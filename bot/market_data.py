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

                # Alpaca wraps snapshots under top-level "snapshots" key
                snapshots_map = data.get("snapshots", {})
                for symbol, snapshot in snapshots_map.items():
                    if snapshot:
                        all_snapshots[symbol] = self._parse_snapshot(symbol, snapshot)

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

        return {
            "symbol": symbol,
            "open": daily_bar.get("o"),
            "high": daily_bar.get("h"),
            "low": daily_bar.get("l"),
            "close": daily_bar.get("c"),
            "volume": daily_bar.get("v"),
            "vwap": daily_bar.get("vw"),
            "prev_close": prev_daily_bar.get("c"),
            "prev_volume": prev_daily_bar.get("v"),
            "bid": latest_quote.get("bp"),
            "ask": latest_quote.get("ap"),
            "bid_size": latest_quote.get("bs"),
            "ask_size": latest_quote.get("as"),
            "last_price": latest_trade.get("p"),
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
