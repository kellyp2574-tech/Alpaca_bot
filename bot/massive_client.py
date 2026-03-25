"""Massive.com API client for full market snapshot (Step 1: Universe Reduction)"""
import logging
from typing import Dict, List, Optional
import requests
from bot import config

logger = logging.getLogger(__name__)


class MassiveClient:
    """Client for Massive.com full market snapshot API"""

    def __init__(self):
        self.base_url = config.MASSIVE_BASE_URL
        self.api_key = config.MASSIVE_API_KEY
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        })

    def get_full_market_snapshot(self) -> Dict[str, dict]:
        """
        Fetch full market snapshot from Massive.
        Returns: Dict mapping symbol -> snapshot data with last trade price
        """
        url = f"{self.base_url}/v1/stocks/snapshots"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            snapshots = {}
            for item in data.get("snapshots", []):
                symbol = item.get("symbol")
                if not symbol:
                    continue

                last_trade = item.get("last_trade", {})
                price = last_trade.get("price")

                if price is not None:
                    snapshots[symbol] = {
                        "symbol": symbol,
                        "price": price,
                        "volume": item.get("daily_volume", 0),
                        "timestamp": last_trade.get("timestamp"),
                    }

            logger.info(f"Massive snapshot: {len(snapshots)} symbols")
            return snapshots

        except requests.exceptions.RequestException as e:
            logger.error(f"Massive API error: {e}")
            return {}

    def filter_by_price_range(
        self, snapshots: Dict[str, dict], min_price: float, max_price: float
    ) -> List[str]:
        """
        Filter symbols by price range.
        Returns: List of symbols within price range
        """
        filtered = []
        for symbol, data in snapshots.items():
            price = data.get("price", 0)
            if min_price <= price <= max_price:
                filtered.append(symbol)

        logger.info(f"Price filter (${min_price}-${max_price}): {len(filtered)} symbols")
        return filtered
