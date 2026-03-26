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
        # Correct v2 endpoint for Massive (Polygon) API
        url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            snapshots = {}
            # v2 endpoint returns tickers array
            for item in data.get("tickers", []):
                symbol = item.get("ticker")
                if not symbol:
                    continue

                # v2 endpoint structure: try lastTrade first, then day close, then prevDay close
                last_trade = item.get("lastTrade", {})
                day_data = item.get("day", {})
                prev_day = item.get("prevDay", {})
                
                price = (
                    last_trade.get("p")  # last trade price
                    or day_data.get("c")  # daily close
                    or prev_day.get("c")  # previous day close
                )

                if price is not None:
                    day_data = item.get("day", {})
                    prev_day = item.get("prevDay", {})
                    snapshots[symbol] = {
                        "symbol": symbol,
                        "price": price,
                        "volume": day_data.get("v", 0),  # today's volume
                        "prev_volume": prev_day.get("v", 0),  # yesterday's volume for ADV
                        "prev_close": prev_day.get("c", 0),  # yesterday's close for gap calc
                        "timestamp": last_trade.get("t"),
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
