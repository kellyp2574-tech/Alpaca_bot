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
        StockSnapshotRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
except (
    ImportError
) as exc:  # pragma: no cover - surfaced when module imported without deps
    raise ImportError("alpaca-py must be installed to use data_alpaca") from exc


load_dotenv()  # loads ALPACA_* variables from .env if present

from .clock import MARKET_TZ, market_now
from .monitoring import get_session_monitor

# TimeFrame mapping for alpaca-py compatibility
_TIMEFRAME_MAP = {
    "1Min": TimeFrame(1, TimeFrameUnit.Minute),
    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
    "1Day": TimeFrame(1, TimeFrameUnit.Day),
}

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


@dataclass
class SnapshotBar:
    c: float
    v: float


@dataclass
class SnapshotTrade:
    p: float


@dataclass
class SnapshotQuote:
    bp: float
    ap: float


@dataclass
class Snapshot:
    latest_trade: Optional[SnapshotTrade]
    latest_quote: Optional[SnapshotQuote]
    daily_bar: Optional[SnapshotBar]
    prev_daily_bar: Optional[SnapshotBar]
    minute_bar: Optional[SnapshotBar] = None


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
        raw_feed = (feed or os.getenv("ALPACA_DATA_FEED", "iex")).lower()

        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca API key/secret must be provided via args or environment"
            )

        self.feed = self._parse_feed(raw_feed)
        self._historical = StockHistoricalDataClient(api_key, secret_key)
        self._screener = ScreenerClient(api_key, secret_key)
        self._api_key = api_key
        self._secret_key = secret_key

        self._stream: Optional[StockDataStream] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._bar_queue: "queue.Queue[MinuteBar]" = queue.Queue(maxsize=5000)
        self._subscribed_symbols: set[str] = set()

    def _parse_feed(self, feed_str: str) -> DataFeed:
        """Parse feed string to DataFeed enum, supporting delayed_sip, iex, and sip."""
        feed_lower = feed_str.lower()
        if feed_lower == "sip":
            return DataFeed.SIP
        elif feed_lower == "iex":
            return DataFeed.IEX
        elif feed_lower == "delayed_sip":
            # delayed_sip is a valid feed option for snapshot/quote endpoints
            # Return as string since DataFeed enum may not have it
            return "delayed_sip"  # type: ignore
        else:
            raise ValueError(f"Unsupported feed={feed_str!r} (use 'iex', 'sip', or 'delayed_sip')")
    
    # ------------------------------------------------------------------
    # Historical endpoints

    def get_most_actives(self, count: int = 50) -> List[str]:
        """Return the most-active symbols (by volume)."""
        # Alpaca screener "most-actives" supports only up to 100
        capped = max(1, min(int(count), 100))
        if capped != count:
            logger.warning(f"most_active_count={count} capped to {capped} (Alpaca limit is 100)")
        
        by = getattr(MostActivesBy, "DOLLAR_VOLUME", None) or MostActivesBy.VOLUME
        req = MostActivesRequest(top=capped, by=by)
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
        if timeframe not in _TIMEFRAME_MAP:
            raise ValueError(f"Supported timeframes: {list(_TIMEFRAME_MAP.keys())}")

        start_utc = self._ensure_utc(start)
        end_utc = self._ensure_utc(end)

        # Map timeframe to Alpaca TimeFrame
        alpaca_timeframe = _TIMEFRAME_MAP[timeframe]

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
            timeframe=_TIMEFRAME_MAP["1Day"],
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

    def get_latest_quotes(self, symbols: Sequence[str], feed: Optional[str] = None) -> Dict[str, Quote]:
        """Fetch latest quotes for multiple symbols with optional feed override."""
        symbols = list(dict.fromkeys(symbols))
        if not symbols:
            return {}
        
        # Use provided feed or fall back to instance default
        feed_to_use = self._parse_feed(feed) if feed else self.feed
        
        import time as _time
        _t0 = _time.monotonic()
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=feed_to_use)
            response = self._historical.get_stock_latest_quote(request)
        except Exception:
            try:
                get_session_monitor().record_api_call(False)
            except Exception:
                pass
            raise
        _latency_ms = (_time.monotonic() - _t0) * 1000
        try:
            mon = get_session_monitor()
            mon.record_api_call(True)
            mon.record_refresh_latency(quote_ms=_latency_ms)
        except Exception:
            pass
        
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

    def get_snapshots(self, symbols: Sequence[str], feed: Optional[str] = None) -> Dict[str, "Snapshot"]:
        """
        Fetch Alpaca snapshots for multiple symbols with optional feed override.
        Returns a dict: symbol -> Snapshot with normalized fields the scanner uses.
        """
        symbols = list(dict.fromkeys(symbols))
        if not symbols:
            return {}
        
        # Use provided feed or fall back to instance default
        feed_to_use = self._parse_feed(feed) if feed else self.feed

        # Alpaca snapshot endpoint supports multiple symbols, but keep batches sane.
        # (Even if Alpaca allows more, batching avoids request-size surprises.)
        def chunk(seq, n):
            for i in range(0, len(seq), n):
                yield seq[i:i+n]

        out: Dict[str, Snapshot] = {}
        
        # Aggregate normalization counters
        _n_raw = 0
        _n_has_lt = 0
        _n_has_lq = 0
        _n_has_db = 0
        _n_has_pdb = 0
        _n_has_mb = 0
        _n_lt_ok = 0
        _n_lq_ok = 0
        _n_db_ok = 0
        _n_pdb_ok = 0
        _n_mb_ok = 0
        _diag_logged = 0  # how many raw snapshots we've dumped so far

        for batch in chunk(symbols, 200):
            try:
                req = StockSnapshotRequest(symbol_or_symbols=batch, feed=feed_to_use)
                resp = self._historical.get_stock_snapshot(req)
                try:
                    get_session_monitor().record_api_call(True)
                except Exception:
                    pass
            except Exception:
                try:
                    get_session_monitor().record_api_call(False)
                except Exception:
                    pass
                raise

            # alpaca-py returns a mapping-like object: resp[symbol] -> snapshot
            for sym in batch:
                try:
                    s = resp[sym]
                except Exception:
                    logger.warning("Snapshot missing for %s", sym)
                    continue
                
                _n_raw += 1
                
                # ── Diagnostic: dump first 3 raw snapshots ────────────
                if _diag_logged < 3:
                    _diag_logged += 1
                    try:
                        logger.info(
                            "SNAPSHOT DIAG [%s] type=%s repr=%.500s",
                            sym, type(s).__name__, repr(s),
                        )
                        for attr_name in ("prev_daily_bar", "daily_bar", "latest_trade",
                                          "latest_quote", "minute_bar"):
                            raw_attr = getattr(s, attr_name, "MISSING_ATTR")
                            if raw_attr == "MISSING_ATTR" and isinstance(s, dict):
                                raw_attr = s.get(attr_name, "MISSING_KEY")
                            logger.info(
                                "  DIAG %s.%s type=%s repr=%.300s",
                                sym, attr_name,
                                type(raw_attr).__name__ if raw_attr not in ("MISSING_ATTR", "MISSING_KEY") else "N/A",
                                repr(raw_attr),
                            )
                    except Exception as diag_err:
                        logger.warning("SNAPSHOT DIAG error for %s: %s", sym, diag_err)

                # ── Helper: extract sub-object supporting both attr and dict ──
                def _sub(parent, *names):
                    """Get a sub-object from parent trying attr access then dict access."""
                    if parent is None:
                        return None
                    for n in names:
                        v = getattr(parent, n, None)
                        if v is not None:
                            return v
                    if isinstance(parent, dict):
                        for n in names:
                            v = parent.get(n)
                            if v is not None:
                                return v
                    return None
                
                def _num(obj, *names):
                    """Extract a numeric value from obj trying multiple attr/key names."""
                    if obj is None:
                        return None
                    for n in names:
                        v = getattr(obj, n, None)
                        if v is not None:
                            try:
                                return float(v)
                            except (TypeError, ValueError):
                                continue
                    if isinstance(obj, dict):
                        for n in names:
                            v = obj.get(n)
                            if v is not None:
                                try:
                                    return float(v)
                                except (TypeError, ValueError):
                                    continue
                    return None

                # ── latest trade ──
                lt = _sub(s, "latest_trade", "latestTrade")
                latest_trade = None
                if lt is not None:
                    _n_has_lt += 1
                    p = _num(lt, "price", "p")
                    if p is not None:
                        latest_trade = SnapshotTrade(p=p)
                        _n_lt_ok += 1

                # ── latest quote ──
                lq = _sub(s, "latest_quote", "latestQuote")
                latest_quote = None
                if lq is not None:
                    _n_has_lq += 1
                    bp = _num(lq, "bid_price", "bp", "bidPrice")
                    ap = _num(lq, "ask_price", "ap", "askPrice")
                    if bp is not None and ap is not None:
                        latest_quote = SnapshotQuote(bp=bp, ap=ap)
                        _n_lq_ok += 1

                # ── daily bar ──
                db = _sub(s, "daily_bar", "dailyBar")
                daily_bar = None
                if db is not None:
                    _n_has_db += 1
                    c = _num(db, "close", "c")
                    v = _num(db, "volume", "v") or 0
                    if c is not None:
                        daily_bar = SnapshotBar(c=c, v=v)
                        _n_db_ok += 1

                # ── prev daily bar ──
                pdb = _sub(s, "prev_daily_bar", "prevDailyBar", "previous_daily_bar")
                prev_daily_bar = None
                if pdb is not None:
                    _n_has_pdb += 1
                    c = _num(pdb, "close", "c")
                    v = _num(pdb, "volume", "v") or 0
                    if c is not None:
                        prev_daily_bar = SnapshotBar(c=c, v=v)
                        _n_pdb_ok += 1

                # ── minute bar ──
                mb = _sub(s, "minute_bar", "minuteBar")
                minute_bar = None
                if mb is not None:
                    _n_has_mb += 1
                    c = _num(mb, "close", "c")
                    v = _num(mb, "volume", "v") or 0
                    if c is not None:
                        minute_bar = SnapshotBar(c=c, v=v)
                        _n_mb_ok += 1

                out[sym] = Snapshot(
                    latest_trade=latest_trade,
                    latest_quote=latest_quote,
                    daily_bar=daily_bar,
                    prev_daily_bar=prev_daily_bar,
                    minute_bar=minute_bar,
                )

        # ── Log aggregate normalization results ──
        logger.info(
            "Snapshot normalization: raw=%d | "
            "lt(has=%d ok=%d) lq(has=%d ok=%d) "
            "db(has=%d ok=%d) pdb(has=%d ok=%d) mb(has=%d ok=%d)",
            _n_raw,
            _n_has_lt, _n_lt_ok, _n_has_lq, _n_lq_ok,
            _n_has_db, _n_db_ok, _n_has_pdb, _n_pdb_ok,
            _n_has_mb, _n_mb_ok,
        )

        return out

    # ------------------------------------------------------------------
    # Live stream

    def subscribe_stream(self, symbols: Iterable[str], feed: Optional[str] = None) -> None:
        """Subscribe to live 1m bars for the provided symbols with optional feed override."""
        symbols = list(dict.fromkeys(symbols))
        if not symbols:
            return
        
        # Use provided feed or fall back to instance default
        feed_to_use = self._parse_feed(feed) if feed else self.feed
        
        # If stream exists but with different feed, close and recreate
        if self._stream is not None:
            # Check if we need to switch feeds (stream object doesn't expose feed, so track it)
            if not hasattr(self, '_current_stream_feed'):
                self._current_stream_feed = self.feed
            
            if feed_to_use != self._current_stream_feed:
                logger.info(f"Switching stream feed from {self._current_stream_feed} to {feed_to_use}")
                self.close_stream()
        
        if self._stream is None:
            self._stream = StockDataStream(
                self._api_key, self._secret_key, feed=feed_to_use
            )
            self._current_stream_feed = feed_to_use

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
        
        # Drain bar queue to prevent stale bars from previous session
        while not self._bar_queue.empty():
            try:
                self._bar_queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset stream feed tracking
        if hasattr(self, '_current_stream_feed'):
            delattr(self, '_current_stream_feed')
    
    def unsubscribe_all(self) -> None:
        """Alias for close_stream() for compatibility with existing code."""
        self.close_stream()

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

    async def _on_stream_bar(self, bar) -> None:
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
