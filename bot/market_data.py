"""Alpaca Market Data client for snapshots, minute bars, and historical data"""
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Set, Tuple
import requests
from bot import config
from bot.rate_limiter import create_alpaca_session

logger = logging.getLogger(__name__)


class AlpacaDataClient:
    """Client for Alpaca Market Data API (IEX feed for free tier)"""

    def __init__(self):
        self.data_url = config.ALPACA_DATA_URL
        self.api_key = config.ALPACA_API_KEY
        self.secret_key = config.ALPACA_SECRET_KEY
        self.feed = config.DATA_FEED

        self.session = create_alpaca_session()
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
        
        # DIAGNOSTIC: Log when critical fields are missing (DEBUG to avoid log flood)
        if open_price is None or prev_close is None:
            logger.debug(
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

    # get_regular_session_open() removed - was unused and misleading
    # The implementation used limit=1 without time filtering, which doesn't reliably
    # return the first RTH quote after 9:30. If needed in future, reimplement with
    # proper start/end time parameters to filter for RTH session.

    def get_tradable_assets(self) -> List[str]:
        """
        Get list of tradable asset symbols from Alpaca API.
        Used as fallback when Massive universe build fails.
        """
        full = self.get_tradable_assets_full()
        return [a.get("symbol") for a in full if a.get("symbol")]

    def get_tradable_assets_full(self) -> List[dict]:
        """
        Get full asset dicts from Alpaca API (symbol, name, exchange, class, etc.).
        Used by universe_builder for asset-type filtering.
        """
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
            logger.info(f"Alpaca assets: {len(data)} returned")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Alpaca assets: {e}")
            return []

    # ═══════════════════════════════════════════════════
    # Overnight Momentum — Intraday minute bars
    # ═══════════════════════════════════════════════════

    def get_minute_bars(
        self, symbols: List[str], start: str, end: str, limit: int = 10000
    ) -> Dict[str, List[dict]]:
        """Fetch 1-minute bars for multiple symbols between start and end (RFC3339).

        Returns: {symbol: [{"t": timestamp, "o": open, "h": high, "l": low, "c": close, "v": volume}, ...]}
        Batches symbols in groups of 200 to stay within API limits.
        """
        all_bars: Dict[str, List[dict]] = {}
        batch_size = 200

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            symbols_param = ",".join(batch)
            url = f"{self.data_url}/v2/stocks/bars"
            params = {
                "symbols": symbols_param,
                "timeframe": "1Min",
                "start": start,
                "end": end,
                "limit": limit,
                "feed": self.feed,
                "adjustment": "raw",
            }

            try:
                page_token = None
                while True:
                    if page_token:
                        params["page_token"] = page_token
                    response = self.session.get(url, params=params, timeout=60)
                    response.raise_for_status()
                    data = response.json()

                    bars_map = data.get("bars", {})
                    for sym, bars in bars_map.items():
                        if sym not in all_bars:
                            all_bars[sym] = []
                        all_bars[sym].extend(bars)

                    page_token = data.get("next_page_token")
                    if not page_token:
                        break

            except requests.exceptions.RequestException as e:
                logger.error(f"Minute bars error for batch {i // batch_size + 1}: {e}")

        logger.info(f"Minute bars: {len(all_bars)} symbols, {sum(len(v) for v in all_bars.values())} total bars")
        return all_bars

    # ═══════════════════════════════════════════════════
    # Overnight Momentum — Historical daily bars (ADV, ATR)
    # ═══════════════════════════════════════════════════

    def get_daily_bars(
        self, symbols: List[str], days: int = 30
    ) -> Dict[str, List[dict]]:
        """Fetch daily bars for the last N calendar days for multiple symbols.

        Returns: {symbol: [{"t": date, "o", "h", "l", "c", "v"}, ...]}
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=int(days * 1.6))  # pad for weekends/holidays

        all_bars: Dict[str, List[dict]] = {}
        batch_size = 200

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            symbols_param = ",".join(batch)
            url = f"{self.data_url}/v2/stocks/bars"
            params = {
                "symbols": symbols_param,
                "timeframe": "1Day",
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "limit": 10000,
                "feed": self.feed,
                "adjustment": "raw",
            }

            try:
                page_token = None
                while True:
                    if page_token:
                        params["page_token"] = page_token
                    response = self.session.get(url, params=params, timeout=60)
                    response.raise_for_status()
                    data = response.json()

                    bars_map = data.get("bars", {})
                    for sym, bars in bars_map.items():
                        if sym not in all_bars:
                            all_bars[sym] = []
                        all_bars[sym].extend(bars)

                    page_token = data.get("next_page_token")
                    if not page_token:
                        break

            except requests.exceptions.RequestException as e:
                logger.error(f"Daily bars error for batch {i // batch_size + 1}: {e}")

        logger.info(f"Daily bars: {len(all_bars)} symbols")
        return all_bars

    @staticmethod
    def calculate_adv(daily_bars: List[dict], days: int = 20) -> Tuple[float, float]:
        """Calculate average daily volume (shares) and ADV dollars from daily bars.

        Uses the most recent `days` bars.
        Returns: (adv_shares, adv_dollars)
        """
        if not daily_bars:
            return 0.0, 0.0
        recent = daily_bars[-days:] if len(daily_bars) >= days else daily_bars
        volumes = [b.get("v", 0) for b in recent]
        closes = [b.get("c", 0) for b in recent]
        avg_vol = sum(volumes) / len(volumes) if volumes else 0.0
        avg_close = sum(closes) / len(closes) if closes else 0.0
        return avg_vol, avg_vol * avg_close

    @staticmethod
    def calculate_atr(daily_bars: List[dict], period: int = 14) -> float:
        """Calculate Average True Range from daily bars.

        ATR = SMA of True Range over `period` bars.
        True Range = max(H-L, |H-prev_C|, |L-prev_C|)
        """
        if len(daily_bars) < 2:
            return 0.0
        recent = daily_bars[-(period + 1):] if len(daily_bars) > period + 1 else daily_bars
        true_ranges = []
        for j in range(1, len(recent)):
            h = recent[j].get("h", 0)
            l = recent[j].get("l", 0)
            prev_c = recent[j - 1].get("c", 0)
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            true_ranges.append(tr)
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

    # ═══════════════════════════════════════════════════
    # Overnight Momentum — Benchmark & sector ETF returns
    # ═══════════════════════════════════════════════════

    def get_intraday_etf_returns(
        self, etf_symbols: List[str], bar_date: str
    ) -> Dict[str, float]:
        """Get intraday returns (open to ~3:30 PM) for benchmark/sector ETFs.

        Args:
            etf_symbols: List of ETF symbols (e.g., ["SPY", "XLK", "XLF", ...])
            bar_date: Date string "YYYY-MM-DD"

        Returns: {symbol: intraday_return_pct}
        """
        start = f"{bar_date}T09:30:00-04:00"
        end = f"{bar_date}T15:30:00-04:00"

        returns = {}
        # Fetch snapshots for simplicity (open and current price)
        snapshots = self.get_snapshots(etf_symbols)
        for sym, snap in snapshots.items():
            open_p = snap.get("open")
            last_p = snap.get("last_price") or snap.get("close")
            if open_p and last_p and open_p > 0:
                returns[sym] = (last_p - open_p) / open_p
            else:
                returns[sym] = 0.0

        logger.info(f"ETF returns: {returns}")
        return returns

    def get_volume_profile_60min(
        self, minute_bars: List[dict]
    ) -> Tuple[int, float]:
        """Compute volume in last 60 minutes and average 60-min bucket volume.

        Used by the 3:50 PM signal model — last 60 minutes = 2:50-3:50 PM.

        Args:
            minute_bars: List of 1-minute bars for a single symbol, sorted by time.

        Returns: (volume_last_60min, avg_60min_volume)
        """
        if not minute_bars or len(minute_bars) < 2:
            return 0, 0.0

        last_60 = minute_bars[-60:] if len(minute_bars) >= 60 else minute_bars
        vol_last_60 = sum(b.get("v", 0) for b in last_60)

        total_vol = sum(b.get("v", 0) for b in minute_bars)
        num_60min_buckets = max(1, len(minute_bars) / 60)
        avg_60min_vol = total_vol / num_60min_buckets

        return vol_last_60, avg_60min_vol

    # ═══════════════════════════════════════════════════
    # Overnight Momentum — Dedicated signal bar fetch
    # ═══════════════════════════════════════════════════

    def get_intraday_bars_for_signal(
        self,
        symbols: List[str],
        bar_date: str,
        start: str = "09:30",
        end: str = "15:50",
    ) -> Dict[str, List[dict]]:
        """Fetch 1-minute bars for the signal window (default 9:30-3:50 PM).

        Convenience wrapper around get_minute_bars() using RFC3339 timestamps.

        Args:
            symbols: List of symbols.
            bar_date: "YYYY-MM-DD"
            start: "HH:MM" (ET, 24h).  Default "09:30".
            end:   "HH:MM" (ET, 24h).  Default "15:50".

        Returns: {symbol: [bar_dict, ...]}
        """
        start_rfc = f"{bar_date}T{start}:00-04:00"
        end_rfc = f"{bar_date}T{end}:00-04:00"
        return self.get_minute_bars(symbols, start_rfc, end_rfc)
