"""VIX data fetcher for exit regime determination"""
import logging
import requests
from typing import Optional
from bot import config

logger = logging.getLogger(__name__)


class VIXFetcher:
    """Fetch VIX level for exit regime determination"""

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

    def get_vix_level(self) -> Optional[float]:
        """
        Get current VIX level from Alpaca.
        VIX is available as index data through Alpaca's API.
        """
        # Try to get VIX via Alpaca's snapshot API
        url = f"{self.data_url}/v2/stocks/VIX/snapshot"

        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                latest_trade = data.get("latestTrade", {})
                vix = latest_trade.get("p")
                if vix:
                    logger.info(f"VIX level: {vix:.2f}")
                    return float(vix)
        except requests.exceptions.RequestException:
            pass

        # Fallback: try VIXY (VIX ETF proxy) if VIX not available
        url = f"{self.data_url}/v2/stocks/VIXY/snapshot"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # VIXY price roughly VIX/10, so estimate VIX
                vixy_price = data.get("latestTrade", {}).get("p", 0)
                if vixy_price:
                    estimated_vix = vixy_price * 10
                    logger.info(f"VIX estimate (via VIXY): {estimated_vix:.2f}")
                    return estimated_vix
        except requests.exceptions.RequestException:
            pass

        logger.warning("Could not fetch VIX, using default middle regime (12-22)")
        return 15.0  # Default to middle regime
