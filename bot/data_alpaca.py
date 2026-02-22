"""Thin Alpaca market data adapter used by the morning momentum bot."""

from __future__ import annotations

import logging
import os
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence

from dotenv import load_dotenv

try:  # Run-time dependency on alpaca-py
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.screener import ScreenerClient
    from alpaca.data.requests import MostActivesBy, MostActivesRequest
    from alpaca.data.live import StockDataStream
    from alpaca.data.requests import (
        StockBarsRequest,
        StockLatestQuoteRequest,
    )
    from alpaca.data.timeframe import TimeFrame
except (
    ImportError
) as exc:  # pragma: no cover - surfaced when module imported without deps
    raise ImportError("alpaca-py must be installed to use data_alpaca") from exc


load_dotenv()  # loads ALPACA_* variables from .env if present

from .clock import MARKET_TZ, market_now

logger = logging.getLogger(__name__)


@dataclass
class MinuteBar:
    symbol: str
    timestamp: datetime
    o: float
    h: float
    l: float
    c: float
    v: float


@dataclass
class DailyStats:
    prev_close: float
    avg_vol_30d: float


@dataclass
class Quote:
    bid_price: float
    ask_price: float


class AlpacaDataAdapter:
    """Restricted surface area around Alpaca's market data endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        feed: Optional[str] = None,
    ) -> None:
        api_key = api_key or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        secret_key = secret_key or os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        feed = feed or os.getenv("ALPACA_DATA_FEED", "sip")

        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca API key/secret must be provided via args or environment"
            )

        self.feed = feed
        self._historical = StockHistoricalDataClient(api_key, secret_key)
        self._screener = ScreenerClient(api_key, secret_key)
        self._api_key = api_key
        self._secret_key = secret_key

        self._stream: Optional[StockDataStream] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._bar_queue: "queue.Queue[MinuteBar]" = queue.Queue(maxsize=5000)
        self._subscribed_symbols: set[str] = set()

    # ------------------------------------------------------------------
    # Historical endpoints

    def get_most_actives(self, count: int = 50) -> List[str]:
        """Return the most-active symbols (by volume)."""
        by = getattr(MostActivesBy, "DOLLAR_VOLUME", None) or MostActivesBy.VOLUME
        req = MostActivesRequest(top=count, by=by)
        resp = self._screener.get_most_actives(req)
        items = resp.most_actives or []
        symbols = []
        for row in items:
            sym = None
            if isinstance(row, dict):
                sym = row.get("symbol")
            else:
                sym = getattr(row, "symbol", None)
            if sym:
                symbols.append(str(sym))
        return symbols

    def get_bars(
        self,
        symbols: Sequence[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Dict[str, List[MinuteBar]]:
        """Fetch minute bars for symbols within [start, end]."""
        if timeframe not in ["1Min", "5Min"]:
            raise ValueError("Only 1Min and 5Min timeframes are supported")

        start_utc = self._ensure_utc(start)
        end_utc = self._ensure_utc(end)

        # Map timeframe to Alpaca TimeFrame
        timeframe_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame.Minute * 5,
        }
        alpaca_timeframe = timeframe_map[timeframe]

        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=alpaca_timeframe,
            start=start_utc,
            end=end_utc,
            feed=self.feed,
        )
        response = self._historical.get_stock_bars(request)
        out: Dict[str, List[MinuteBar]] = {}
        for symbol, barset in response.data.items():
            bars = sorted(barset, key=lambda bar: bar.timestamp)
            minute_bars = [self._to_minute_bar(symbol, bar) for bar in bars]
            out[symbol] = minute_bars
        return out

    def get_daily_bars(
        self,
        symbols: Sequence[str],
        lookback_days: int = 35,
        end_dt: Optional[datetime] = None,
    ) -> MutableMapping[str, DailyStats]:
        """Return prev close + 30-day average volume for each symbol."""
        if not symbols:
            return {}
        if end_dt is None:
            end_dt = datetime.now(timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(days=lookback_days * 2)

        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame.Day,
            start=self._ensure_utc(start_dt),
            end=self._ensure_utc(end_dt),
            feed=self.feed,
        )
        response = self._historical.get_stock_bars(request)

        stats: Dict[str, DailyStats] = {}
        today = market_now().date()
        for symbol, barset in response.data.items():
            if not barset:
                continue
            bars = sorted(barset, key=lambda bar: bar.timestamp)
            vols = [bar.volume for bar in bars]
            last_bar = bars[-1]
            last_dt = last_bar.timestamp
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            last_date = last_dt.astimezone(MARKET_TZ).date()
            if last_date == today and len(bars) >= 2:
                prev_close = bars[-2].close
            else:
                prev_close = last_bar.close
            vol_window = vols[-30:] if len(vols) >= 30 else vols
            avg_vol = sum(vol_window) / len(vol_window)
            stats[symbol] = DailyStats(prev_close=float(prev_close), avg_vol_30d=float(avg_vol))
        return stats

    def get_latest_quote(self, symbol: str) -> Quote:
        """Fetch the best bid/ask for *symbol*."""

        request = StockLatestQuoteRequest(symbol_or_symbols=[symbol], feed=self.feed)
        response = self._historical.get_stock_latest_quote(request)
        try:
            quote = response[symbol]
        except Exception:
            logger.exception("Quote missing for %s", symbol)
            return Quote(bid_price=0.0, ask_price=0.0)
        bid = getattr(quote, "bid_price", None) or 0.0
        ask = getattr(quote, "ask_price", None) or 0.0
        return Quote(bid_price=float(bid), ask_price=float(ask))

    def get_latest_quotes(self, symbols: Sequence[str]) -> Dict[str, Quote]:
        """Fetch best bid/ask for multiple symbols with a single request."""

        symbols = list(dict.fromkeys(symbols))
        if not symbols:
            return {}
        request = StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=self.feed)
        response = self._historical.get_stock_latest_quote(request)
        quotes: Dict[str, Quote] = {}
        for symbol in symbols:
            try:
                data = response[symbol]
            except Exception:
                logger.warning("Quote missing for %s", symbol)
                continue
            bid = getattr(data, "bid_price", None) or 0.0
            ask = getattr(data, "ask_price", None) or 0.0
            quotes[symbol] = Quote(bid_price=float(bid), ask_price=float(ask))
        return quotes

    # ------------------------------------------------------------------
    # Live stream

    def subscribe_stream(self, symbols: Iterable[str]) -> None:
        """Subscribe to live 1m bars for the provided symbols."""
        symbols = list(dict.fromkeys(symbols))
        if not symbols:
            return
        if self._stream is None:
            self._stream = StockDataStream(
                self._api_key, self._secret_key, feed=self.feed
            )

        new_symbols = [sym for sym in symbols if sym not in self._subscribed_symbols]
        if not new_symbols:
            return

        for symbol in new_symbols:
            self._stream.subscribe_bars(self._on_stream_bar, symbol)
            self._subscribed_symbols.add(symbol)

        if self._stream_thread is None or not self._stream_thread.is_alive():
            self._stream_thread = threading.Thread(target=self._stream.run, daemon=True)
            self._stream_thread.start()

    def next_bar(self, timeout: Optional[float] = None) -> Optional[MinuteBar]:
        """Blocking read for the next streamed bar."""
        try:
            return self._bar_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close_stream(self) -> None:
        if self._stream:
            self._stream.stop()
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=1)
        self._stream = None
        self._stream_thread = None
        self._subscribed_symbols.clear()

    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _to_minute_bar(symbol: str, bar) -> MinuteBar:
        return MinuteBar(
            symbol=symbol,
            timestamp=bar.timestamp,
            o=bar.open,
            h=bar.high,
            l=bar.low,
            c=bar.close,
            v=bar.volume,
        )

    def _on_stream_bar(self, bar) -> None:
        minute_bar = MinuteBar(
            symbol=bar.symbol,
            timestamp=bar.timestamp,
            o=bar.open,
            h=bar.high,
            l=bar.low,
            c=bar.close,
            v=bar.volume,
        )
        try:
            self._bar_queue.put_nowait(minute_bar)
        except queue.Full:
            try:
                self._bar_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._bar_queue.put_nowait(minute_bar)
            except queue.Full:
                logger.warning("Dropping bar for %s due to full queue", bar.symbol)
