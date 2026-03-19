"""
Market Data — SPY, QQQ, VIX price tracking for the condor strategy.

Uses Alpaca market data for SPY/QQQ and yfinance for VIX prev close.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import requests
import yfinance as yf

from bot.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER

MARKET_TZ = ZoneInfo("America/New_York")

logger = logging.getLogger("bot.market_data")

# Alpaca data base URLs
DATA_BASE = "https://data.alpaca.markets"


def _data_headers() -> dict:
    return {
        "accept": "application/json",
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }


@dataclass
class MorningTracker:
    """
    Tracks intraday price data for morning assessment and defense monitoring.

    IMPORTANT:
    - spy_high / spy_low / qqq_high / qqq_low are sourced from Alpaca's
      session-wide daily bar, NOT only from when the bot started tracking.
      This is intentional: it gives the true session extremes even on a
      late start.  The QQQ morning range filter and logging reflect the
      full session range as reported by the exchange.
    - For condor defense, we maintain *separate* post-anchor high/low
      fields (spy_post_anchor_high / spy_post_anchor_low) that are only
      updated after the anchor is set.  This lets check_defense() catch
      intra-interval breaches that the current last price may have
      recovered from.
    """
    # SPY tracking (session-wide from Alpaca daily bar)
    spy_open: Optional[float] = None
    spy_high: float = 0.0
    spy_low: float = float("inf")
    spy_last: float = 0.0

    # SPY post-anchor tracking (only populated after condor entry)
    spy_post_anchor_high: float = 0.0
    spy_post_anchor_low: float = float("inf")

    # QQQ tracking (session-wide from Alpaca daily bar)
    qqq_open: Optional[float] = None
    qqq_high: float = 0.0
    qqq_low: float = float("inf")
    qqq_last: float = 0.0

    # VIX previous close
    vix_prev_close: Optional[float] = None

    # Condor anchor price (SPY at 11:30)
    condor_anchor: Optional[float] = None

    def update_spy(self, price: float, high: Optional[float] = None, low: Optional[float] = None):
        """
        Update SPY tracking.

        - spy_high / spy_low use session-wide daily bar extremes (from the
          ``high`` / ``low`` params supplied by refresh_tracker).  These
          are suitable for the QQQ morning-range filter and general
          session stats.
        - spy_post_anchor_high / spy_post_anchor_low use ONLY the
          last-trade ``price``, deliberately ignoring the daily-bar
          high/low.  This prevents pre-anchor session extremes from
          contaminating post-entry defense monitoring.
        """
        if self.spy_open is None:
            self.spy_open = price
        self.spy_last = price
        # Session-wide extremes (from daily bar where available)
        if high is not None:
            self.spy_high = max(self.spy_high, high)
        else:
            self.spy_high = max(self.spy_high, price)
        if low is not None:
            self.spy_low = min(self.spy_low, low)
        else:
            self.spy_low = min(self.spy_low, price)
        # Post-anchor tracking: last-trade only, never daily bar extremes.
        # This ensures that a large pre-anchor session move does not
        # instantly trigger defense on a freshly-entered condor.
        if self.condor_anchor is not None:
            self.spy_post_anchor_high = max(self.spy_post_anchor_high, price)
            self.spy_post_anchor_low = min(self.spy_post_anchor_low, price)

    def update_qqq(self, price: float, high: Optional[float] = None, low: Optional[float] = None):
        """Update QQQ tracking with latest price."""
        if self.qqq_open is None:
            self.qqq_open = price
        self.qqq_last = price
        if high is not None:
            self.qqq_high = max(self.qqq_high, high)
        else:
            self.qqq_high = max(self.qqq_high, price)
        if low is not None:
            self.qqq_low = min(self.qqq_low, low)
        else:
            self.qqq_low = min(self.qqq_low, price)

    @property
    def qqq_morning_range_pct(self) -> float:
        """QQQ morning range as percentage: (high - low) / open."""
        if not self.qqq_open or self.qqq_open == 0:
            return 0.0
        return (self.qqq_high - self.qqq_low) / self.qqq_open

    @property
    def qqq_morning_direction_pct(self) -> float:
        """QQQ morning directional move: (last - open) / open. Positive = up."""
        if not self.qqq_open or self.qqq_open == 0:
            return 0.0
        return (self.qqq_last - self.qqq_open) / self.qqq_open

    @property
    def qqq_direction(self) -> str:
        """'up' or 'down' based on morning move."""
        return "up" if self.qqq_morning_direction_pct > 0 else "down"

    def spy_move_from_anchor_pct(self) -> Optional[float]:
        """Current SPY move from condor anchor price as absolute percentage."""
        if self.condor_anchor is None or self.condor_anchor == 0:
            return None
        return abs(self.spy_last - self.condor_anchor) / self.condor_anchor

    def spy_max_move_from_anchor_pct(self) -> Optional[float]:
        """
        Max SPY move from anchor using post-anchor high/low.

        Uses only extremes recorded *after* the anchor was set, so it
        catches intra-interval breaches even if the current last-trade
        has recovered.
        """
        if self.condor_anchor is None or self.condor_anchor == 0:
            return None
        if self.spy_post_anchor_high == 0.0:
            # No post-anchor data yet; fall back to current last
            return abs(self.spy_last - self.condor_anchor) / self.condor_anchor
        move_high = abs(self.spy_post_anchor_high - self.condor_anchor) / self.condor_anchor
        move_low = abs(self.spy_post_anchor_low - self.condor_anchor) / self.condor_anchor
        return max(move_high, move_low)

    def reset_post_anchor_tracking(self):
        """Call when the condor anchor is set to start fresh post-anchor extremes."""
        seed = self.spy_last if self.spy_last > 0 else (self.condor_anchor or 0.0)
        self.spy_post_anchor_high = seed
        self.spy_post_anchor_low = seed


# ─── Data fetching ────────────────────────────────────────────────────────────

def get_latest_quote(symbol: str) -> Optional[float]:
    """Get the latest trade price for a symbol via Alpaca."""
    url = f"{DATA_BASE}/v2/stocks/{symbol}/trades/latest"
    try:
        resp = requests.get(url, headers=_data_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        price = float(data.get("trade", {}).get("p", 0))
        if price > 0:
            return price
    except Exception as e:
        logger.warning("Failed to get latest quote for %s: %s", symbol, e)
    return None


@dataclass
class SnapshotData:
    """Snapshot data for a single symbol from Alpaca."""
    last_price: float = 0.0
    bar_high: float = 0.0
    bar_low: float = 0.0
    bar_open: float = 0.0
    daily_open: float = 0.0
    daily_high: float = 0.0
    daily_low: float = 0.0


def get_latest_snapshots(symbols: list[str]) -> dict[str, SnapshotData]:
    """Get full snapshot data for multiple symbols including bar high/low."""
    results = {}
    url = f"{DATA_BASE}/v2/stocks/snapshots"
    params = {"symbols": ",".join(symbols), "feed": "iex"}
    try:
        resp = requests.get(url, headers=_data_headers(), params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for sym, snapshot in data.items():
            trade = snapshot.get("latestTrade", {})
            minute_bar = snapshot.get("minuteBar", {})
            daily_bar = snapshot.get("dailyBar", {})
            results[sym] = SnapshotData(
                last_price=float(trade.get("p", 0)),
                bar_high=float(minute_bar.get("h", 0)),
                bar_low=float(minute_bar.get("l", 0)),
                bar_open=float(minute_bar.get("o", 0)),
                daily_open=float(daily_bar.get("o", 0)),
                daily_high=float(daily_bar.get("h", 0)),
                daily_low=float(daily_bar.get("l", 0)),
            )
    except Exception as e:
        logger.warning("Failed to get snapshots for %s: %s", symbols, e)
        # Fallback: individual trade lookups with no bar data
        for sym in symbols:
            price = get_latest_quote(sym)
            if price:
                results[sym] = SnapshotData(last_price=price)
    return results


def get_latest_quotes(symbols: list[str]) -> dict[str, float]:
    """Get latest trade prices for multiple symbols (convenience wrapper)."""
    snapshots = get_latest_snapshots(symbols)
    return {sym: s.last_price for sym, s in snapshots.items() if s.last_price > 0}


def get_vix_previous_close() -> Optional[float]:
    """
    Get the *prior trading day's* VIX close using yfinance.

    Before market open, the last row in a 5d history IS yesterday's close.
    After today's daily bar updates, the last row is today's close, so we
    need the second-to-last.  We guard against both cases.
    """
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if hist.empty:
            logger.warning("No VIX history returned")
            return None

        today = datetime.now(MARKET_TZ).date()
        if len(hist) >= 2:
            last_date = hist.index[-1].date()
            if last_date >= today:
                # Today's bar already present — use second-to-last
                prev_close = float(hist["Close"].iloc[-2])
            else:
                # Most recent bar is from a prior day (pre-open)
                prev_close = float(hist["Close"].iloc[-1])
        else:
            prev_close = float(hist["Close"].iloc[-1])

        logger.info("VIX previous close: %.2f (rows=%d)", prev_close, len(hist))
        return prev_close
    except Exception as e:
        logger.error("Failed to get VIX previous close: %s", e)
        return None


def refresh_tracker(tracker: MorningTracker) -> MorningTracker:
    """
    Refresh the morning tracker with latest SPY and QQQ data.

    High/low values come from Alpaca's session-wide daily bar, giving
    the exchange-reported intraday extremes (not just our sampled max/min
    of last-trade prices).  This is intentional: it means qqq_high/low
    reflect the true session range, which is what the morning-range
    filter should evaluate.
    """
    snapshots = get_latest_snapshots(["SPY", "QQQ"])

    spy = snapshots.get("SPY")
    if spy and spy.last_price > 0:
        # Use daily bar high/low if available, otherwise fall back to last price
        high = spy.daily_high if spy.daily_high > 0 else None
        low = spy.daily_low if spy.daily_low > 0 else None
        # Sanity check: discard extended-hours extremes (>3% from last trade)
        if high is not None and abs(high - spy.last_price) / spy.last_price > 0.03:
            logger.warning(
                "SPY daily_high=%.2f deviates >3%% from last=%.2f — likely extended hours; ignoring",
                high, spy.last_price,
            )
            high = None
        if low is not None and low > 0 and abs(low - spy.last_price) / spy.last_price > 0.03:
            logger.warning(
                "SPY daily_low=%.2f deviates >3%% from last=%.2f — likely extended hours; ignoring",
                low, spy.last_price,
            )
            low = None
        tracker.update_spy(spy.last_price, high=high, low=low)
        # Backfill open on first successful refresh
        if tracker.spy_open is None and spy.daily_open > 0:
            tracker.spy_open = spy.daily_open

    qqq = snapshots.get("QQQ")
    if qqq and qqq.last_price > 0:
        high = qqq.daily_high if qqq.daily_high > 0 else None
        low = qqq.daily_low if qqq.daily_low > 0 else None
        # Sanity check: daily bar high/low can include extended-hours data
        # which distorts intraday range calculations. If > 3% from last
        # trade, discard and use last price only.
        if high is not None and abs(high - qqq.last_price) / qqq.last_price > 0.03:
            logger.warning(
                "QQQ daily_high=%.2f deviates >3%% from last=%.2f — likely extended hours; ignoring",
                high, qqq.last_price,
            )
            high = None
        if low is not None and low > 0 and abs(low - qqq.last_price) / qqq.last_price > 0.03:
            logger.warning(
                "QQQ daily_low=%.2f deviates >3%% from last=%.2f — likely extended hours; ignoring",
                low, qqq.last_price,
            )
            low = None
        tracker.update_qqq(qqq.last_price, high=high, low=low)
        if tracker.qqq_open is None and qqq.daily_open > 0:
            tracker.qqq_open = qqq.daily_open

    return tracker
