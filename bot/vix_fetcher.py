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
        Get current intraday VIX level via yfinance.
        Priority: fast_info live price > 1-minute bar close > daily bar close.
        Falls back to 15.0 (middle regime) if unavailable.
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker("^VIX")

            # Primary: fast_info gives the live/latest price during market hours
            try:
                last_price = ticker.fast_info.get("lastPrice")
                if last_price and float(last_price) > 0:
                    vix = float(last_price)
                    logger.info(f"VIX level (live): {vix:.2f}")
                    return vix
            except Exception:
                pass

            # Fallback: 1-minute bars for current session
            try:
                bars = ticker.history(period="1d", interval="1m")
                if not bars.empty:
                    vix = float(bars["Close"].iloc[-1])
                    if vix > 0:
                        logger.info(f"VIX level (1m bar): {vix:.2f}")
                        return vix
            except Exception:
                pass

            # Last resort: daily bar (may be previous close, not intraday)
            hist = ticker.history(period="1d")
            if not hist.empty:
                vix = float(hist["Close"].iloc[-1])
                if vix > 0:
                    logger.info(f"VIX level (daily close, may be stale): {vix:.2f}")
                    return vix

            logger.warning("yfinance returned no usable VIX data")
        except ImportError:
            logger.warning("yfinance not installed - cannot fetch VIX")
        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}")

        logger.warning("Using default middle regime (VIX=15.0)")
        return 15.0
