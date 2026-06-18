"""
MR free-data pipeline.

Purpose:
    Replace paid Massive live universe dependency with:
      1. Massive free previous-day/grouped daily data for broad watchlist.
      2. Alpaca live snapshots for actual 15:30/15:45 MR signal.

This should NOT be used to make final buy decisions from stale Massive data.
Massive is only for narrowing the universe.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from bot import config
from bot.rate_limiter import create_alpaca_session

logger = logging.getLogger(__name__)


@dataclass
class MRWatchSymbol:
    symbol: str
    prev_close: float
    prev_volume: int
    prev_dollar_volume: float
    prior_ret: Optional[float] = None  # T-2 to T-1 return (prior day return)


@dataclass
class LiveMRCandidate:
    symbol: str
    signal_price: float
    day_return: float
    close_position: float
    adv_dollars: float
    bid: Optional[float]
    ask: Optional[float]
    spread_pct: Optional[float]
    volume_today: Optional[float]
    selection_score: float
    prior_ret: Optional[float] = None  # T-2 to T-1 return for audit/diagnostics


def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def previous_weekday(d: Optional[date] = None) -> date:
    """Simple previous weekday helper. Replace with market calendar if you have one."""
    d = d or date.today()
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d = d - timedelta(days=1)
    return d


def _fetch_massive_grouped(
    trade_date: date,
    massive_api_key: str,
    timeout: int = 30,
) -> Optional[List[dict]]:
    """Fetch grouped daily data from Massive for a specific date."""
    url = f"https://api.massive.com/v2/aggs/grouped/locale/us/market/stocks/{trade_date.isoformat()}"
    params = {"adjusted": "true", "apiKey": massive_api_key}
    
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("results") or []
    except requests.RequestException as e:
        logger.warning(f"Massive request failed for {trade_date}: {e}")
        return None


def _fetch_latest_grouped_on_or_before(
    start_date: date,
    massive_api_key: str,
    max_lookback: int = 7,
    timeout: int = 30,
) -> Tuple[Optional[date], Optional[List[dict]]]:
    """
    Find the latest available grouped session on or before start_date.
    Returns (actual_date, rows) or (None, None) if no data found.
    """
    d = start_date
    for _ in range(max_lookback):
        rows = _fetch_massive_grouped(d, massive_api_key, timeout)
        if rows:
            return d, rows
        d = previous_weekday(d)
    return None, None


def build_massive_prevday_watchlist(
    massive_api_key: str,
    trade_date: Optional[date] = None,
    min_prev_close: float = 0.75,
    max_prev_close: float = 3.00,
    min_prev_dollar_volume: float = 500_000,
    apply_prior_ret_filter: bool = False,  # True for overnight MR, False for intraday MR
    prior_ret_min: Optional[float] = None,  # Defaults to config.MR_FREE_PRIOR_RET_MIN
    prior_ret_max: Optional[float] = None,   # Defaults to config.MR_FREE_PRIOR_RET_MAX
    require_t2_data: Optional[bool] = None,  # Defaults to config.MR_FREE_PRIOR_RET_REQUIRE_DATA
    timeout: int = 30,
) -> List[MRWatchSymbol]:
    """
    Build broad MR watchlist from previous-day Massive grouped daily data.
    
    Args:
        apply_prior_ret_filter: If True, fetches T-2 and filters by prior-day return.
                               If False, returns all T-1 symbols (for intraday MR).
        
    NOTE: T-1 and T-2 are fetched SEQUENTIALLY to ensure correct pairing:
      1. Find actual T-1 (latest available session before today)
      2. Find actual T-2 (latest available session before actual T-1)
    This prevents holiday misalignment issues.
    """
    # Use config defaults if not provided
    if prior_ret_min is None:
        prior_ret_min = float(getattr(config, "MR_FREE_PRIOR_RET_MIN", -0.20))
    if prior_ret_max is None:
        prior_ret_max = float(getattr(config, "MR_FREE_PRIOR_RET_MAX", 0.05))  # +5% per backtest
    if require_t2_data is None:
        require_t2_data = bool(getattr(config, "MR_FREE_PRIOR_RET_REQUIRE_DATA", False))
    
    # Step 1: Find actual T-1 (latest available grouped session)
    search_start = trade_date or previous_weekday()
    actual_t1, results_t1 = _fetch_latest_grouped_on_or_before(search_start, massive_api_key, timeout=timeout)
    
    if not results_t1 or not actual_t1:
        logger.error("Massive prevday watchlist failed: no T-1 grouped data found")
        return []
    
    # Step 2: Find actual T-2 (session before actual T-1) - ONLY if applying filter
    actual_t2 = None
    results_t2 = None
    t2_close_by_symbol = {}
    
    if apply_prior_ret_filter:
        # Start searching from the day before actual T-1
        t2_search_start = previous_weekday(actual_t1)
        actual_t2, results_t2 = _fetch_latest_grouped_on_or_before(t2_search_start, massive_api_key, timeout=timeout)
        
        if not results_t2:
            msg = f"No T-2 data available (T-1={actual_t1}, searched back from {t2_search_start})"
            if require_t2_data:
                logger.error(f"Massive prevday watchlist: {msg}; failing closed (require_t2_data=True)")
                return []
            else:
                logger.warning(f"Massive prevday watchlist: {msg}; prior_ret filtering disabled")
        else:
            # Build T-2 close lookup
            for row in results_t2:
                symbol = row.get("T") or row.get("ticker")
                close = row.get("c")
                if symbol and close is not None:
                    try:
                        t2_close_by_symbol[symbol] = float(close)
                    except (TypeError, ValueError):
                        pass
            logger.info(f"T-2 data found: actual_t2={actual_t2}, symbols={len(t2_close_by_symbol)}")
    
    # Process T-1 results and optionally apply prior_ret filter
    watchlist: List[MRWatchSymbol] = []
    filtered_crashed = 0   # prior_ret < -20%
    filtered_surged = 0    # prior_ret > +5%
    filtered_missing_t2 = 0  # T-2 data unavailable for symbol
    
    for row in results_t1:
        symbol = row.get("T") or row.get("ticker")
        close = row.get("c")
        volume = row.get("v")

        if not symbol or close is None or volume is None:
            continue

        try:
            prev_close = float(close)
            prev_volume = int(float(volume))
        except (TypeError, ValueError):
            continue

        if prev_close <= 0 or prev_volume <= 0:
            continue

        prev_dollar_volume = prev_close * prev_volume

        # Broad watchlist filter, intentionally wider than final MR.
        if not (min_prev_close <= prev_close <= max_prev_close):
            continue

        if prev_dollar_volume < min_prev_dollar_volume:
            continue

        # Lightweight symbol exclusions.
        if should_exclude_symbol_basic(symbol):
            continue

        # Calculate prior-day return (T-2 to T-1) if applying filter and T-2 data available
        prior_ret = None
        if apply_prior_ret_filter and t2_close_by_symbol:
            t2_close = t2_close_by_symbol.get(symbol)
            if t2_close and t2_close > 0:
                prior_ret = (prev_close - t2_close) / t2_close
                
                # Apply filter: reject crashed stocks (< -20%)
                if prior_ret < prior_ret_min:
                    filtered_crashed += 1
                    continue
                
                # Apply filter: reject surged stocks (> +5%)
                if prior_ret > prior_ret_max:
                    filtered_surged += 1
                    continue
            else:
                # Symbol in T-1 but not in T-2 (e.g., new listing)
                filtered_missing_t2 += 1
                if require_t2_data:
                    continue

        watchlist.append(
            MRWatchSymbol(
                symbol=symbol,
                prev_close=prev_close,
                prev_volume=prev_volume,
                prev_dollar_volume=prev_dollar_volume,
                prior_ret=prior_ret,
            )
        )

    # Detailed logging
    total_rejected = filtered_crashed + filtered_surged + filtered_missing_t2
    if apply_prior_ret_filter:
        logger.info(
            f"Massive prevday watchlist (WITH prior_ret filter): {len(watchlist)} symbols "
            f"from T-1={actual_t1} (T-2={actual_t2}), "
            f"rejected: crashed={filtered_crashed}, surged={filtered_surged}, no_t2={filtered_missing_t2}"
        )
    else:
        logger.info(
            f"Massive prevday watchlist (NO prior_ret filter): {len(watchlist)} symbols from T-1={actual_t1}"
        )
    
    return watchlist


def should_exclude_symbol_basic(symbol: str) -> bool:
    """
    Basic ticker hygiene.

    Keep this conservative. More precise ETF/common-stock filtering can come
    from Alpaca assets or your existing universe filters.
    """
    s = symbol.upper()

    # Common non-common-stock suffix patterns.
    bad_markers = [
        ".WS", ".W", ".U", ".R", ".P", ".PR",
        "-WS", "-W", "-U", "-R", "-P", "-PR",
    ]

    if any(marker in s for marker in bad_markers):
        return True

    # Avoid weird test/temporary symbols.
    if len(s) > 5 and "." not in s and "-" not in s:
        return True

    return False


def fetch_alpaca_snapshots_batched(
    symbols: List[str],
    alpaca_data_key: str,
    alpaca_data_secret: str,
    feed: str = "iex",
    batch_size: int = 200,
    sleep_seconds: float = 0.25,
    timeout: int = 20,
) -> Dict[str, dict]:
    """
    Fetch Alpaca stock snapshots for many symbols.

    Alpaca snapshots include latest trade, latest quote, minute bar,
    daily bar, and previous daily bar for requested symbols.
    """
    base_url = "https://data.alpaca.markets/v2/stocks/snapshots"
    # Use the shared rate-limited session so these snapshot calls count against
    # the global Alpaca budget (60/min) instead of bypassing it with raw requests.
    session = create_alpaca_session()
    session.headers.update({
        "APCA-API-KEY-ID": alpaca_data_key,
        "APCA-API-SECRET-KEY": alpaca_data_secret,
    })

    out: Dict[str, dict] = {}

    for batch in chunked(symbols, batch_size):
        params = {
            "symbols": ",".join(batch),
            "feed": feed,
        }

        try:
            resp = session.get(base_url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            # Alpaca generally returns {"snapshots": {...}} on v2 docs/examples,
            # but keep fallback in case your wrapper returns direct mapping.
            snapshots = data.get("snapshots", data)
            if isinstance(snapshots, dict):
                out.update(snapshots)

        except requests.RequestException as e:
            logger.warning(f"Alpaca snapshot batch failed: {batch[:3]}... size={len(batch)} err={e}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    logger.info(f"Alpaca snapshots fetched: {len(out)}/{len(symbols)} symbols")
    return out


def _bar_get(bar: Optional[dict], key: str) -> Optional[float]:
    if not isinstance(bar, dict):
        return None
    val = bar.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _quote_get(quote: Optional[dict], key: str) -> Optional[float]:
    if not isinstance(quote, dict):
        return None
    val = quote.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _trade_price(snapshot: dict) -> Optional[float]:
    trade = snapshot.get("latestTrade") or snapshot.get("latest_trade") or {}
    price = trade.get("p") or trade.get("price")
    if price is None:
        return None
    try:
        return float(price)
    except (TypeError, ValueError):
        return None


def parse_live_mr_fields(symbol: str, snapshot: dict, fallback_prev_close: Optional[float] = None) -> Optional[dict]:
    """
    Convert Alpaca raw snapshot into normalized MR fields.
    """
    latest_price = _trade_price(snapshot)

    daily_bar = snapshot.get("dailyBar") or snapshot.get("daily_bar") or {}
    prev_daily_bar = snapshot.get("prevDailyBar") or snapshot.get("prev_daily_bar") or {}
    quote = snapshot.get("latestQuote") or snapshot.get("latest_quote") or {}

    day_open = _bar_get(daily_bar, "o")
    day_high = _bar_get(daily_bar, "h")
    day_low = _bar_get(daily_bar, "l")
    day_close = _bar_get(daily_bar, "c")
    volume_today = _bar_get(daily_bar, "v")

    prev_close = _bar_get(prev_daily_bar, "c") or fallback_prev_close

    # Use latest trade if available; fallback to daily close from snapshot.
    signal_price = latest_price or day_close

    if not signal_price or not prev_close or prev_close <= 0:
        return None

    if not day_high or not day_low or day_high <= day_low:
        return None

    day_return = (signal_price / prev_close) - 1.0
    close_position = (signal_price - day_low) / (day_high - day_low)
    close_position = max(0.0, min(1.0, close_position))

    bid = _quote_get(quote, "bp") or _quote_get(quote, "bid_price")
    ask = _quote_get(quote, "ap") or _quote_get(quote, "ask_price")

    spread_pct = None
    if bid and ask and bid > 0 and ask >= bid:
        mid = (bid + ask) / 2.0
        if mid > 0:
            spread_pct = (ask - bid) / mid

    return {
        "symbol": symbol,
        "signal_price": signal_price,
        "prev_close": prev_close,
        "day_return": day_return,
        "close_position": close_position,
        "bid": bid,
        "ask": ask,
        "spread_pct": spread_pct,
        "volume_today": volume_today,
        "day_open": day_open,
        "day_high": day_high,
        "day_low": day_low,
    }


def build_live_mr_candidates_from_free_pipeline(
    watchlist: List[MRWatchSymbol],
    alpaca_data_key: str,
    alpaca_data_secret: str,
    feed: str = "iex",
) -> List[LiveMRCandidate]:
    """
    Final MR candidate generation:
      - Massive previous-day watchlist narrows symbols.
      - Alpaca snapshots decide today's live MR filters.
    """
    batch_size = int(getattr(config, "MR_FREE_ALPACA_BATCH_SIZE", 200))
    sleep_seconds = float(getattr(config, "MR_FREE_ALPACA_BATCH_SLEEP_SECONDS", 0.25))

    symbols = [w.symbol for w in watchlist]
    prev_by_symbol = {w.symbol: w for w in watchlist}

    snapshots = fetch_alpaca_snapshots_batched(
        symbols=symbols,
        alpaca_data_key=alpaca_data_key,
        alpaca_data_secret=alpaca_data_secret,
        feed=feed,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
    )

    candidates: List[LiveMRCandidate] = []

    for symbol, snap in snapshots.items():
        prev = prev_by_symbol.get(symbol)
        if not prev:
            continue

        fields = parse_live_mr_fields(
            symbol=symbol,
            snapshot=snap,
            fallback_prev_close=prev.prev_close,
        )
        if not fields:
            continue

        price = fields["signal_price"]
        day_return = fields["day_return"]
        close_position = fields["close_position"]
        spread_pct = fields["spread_pct"]

        # Your live MR filters.
        if price < float(getattr(config, "MR_MIN_PRICE", 1.00)):
            continue
        if price > float(getattr(config, "MR_MAX_PRICE", 2.00)):
            continue
        if day_return > float(getattr(config, "MR_DAY_RET_MAX", -0.04)):
            continue
        if close_position > float(getattr(config, "MR_CLOSE_POSITION_MAX", 0.25)):
            continue

        # Execution spread sanity. For MR low-priced stocks you currently allow wider than ETFs.
        max_spread = float(getattr(config, "ENTRY_MAX_SPREAD_PCT", 0.05))
        if spread_pct is not None and spread_pct > max_spread:
            continue

        # ADV estimate from previous-day Massive, adjusted by IEX multiplier only if you need to
        # compensate for IEX-derived volume elsewhere. Since Massive grouped daily is composite-ish,
        # I would NOT multiply Massive volume by 50. Keep multiplier for IEX-only ADV.
        adv_dollars = prev.prev_dollar_volume

        min_adv = float(getattr(config, "MR_MIN_AVG_DOLLAR_VOLUME", 1_000_000))
        if adv_dollars < min_adv:
            continue

        # Lower close_position is better; more negative day_return is usually better,
        # but don't overfit this. Keep it simple.
        selection_score = close_position + max(day_return, -0.50)

        candidates.append(
            LiveMRCandidate(
                symbol=symbol,
                signal_price=price,
                day_return=day_return,
                close_position=close_position,
                adv_dollars=adv_dollars,
                bid=fields["bid"],
                ask=fields["ask"],
                spread_pct=spread_pct,
                volume_today=fields["volume_today"],
                selection_score=selection_score,
                prior_ret=prev.prior_ret,  # Propagate from Massive watchlist for audit
            )
        )

    # Your newest rank preference: close location ascending.
    if getattr(config, "MR_RANK_BY_CLOSE_LOCATION_ONLY", True):
        candidates.sort(key=lambda c: (c.close_position, c.day_return))
    else:
        candidates.sort(key=lambda c: c.selection_score)

    logger.info(f"Live MR candidates from free pipeline: {len(candidates)}")
    return candidates


def convert_live_to_existing_mr_candidate(c: LiveMRCandidate):
    """
    Adapter to your existing MeanReversionCandidate class.

    Adjust fields to match your actual dataclass constructor.
    """
    from bot.mean_reversion_scorer import MeanReversionCandidate

    result = MeanReversionCandidate(
        symbol=c.symbol,
        signal_price=c.signal_price,
        open_price_930=c.signal_price,  # placeholder - use day_open if available
        high_930_to_signal=c.signal_price,  # placeholder - use day_high if available
        low_930_to_signal=c.signal_price,  # placeholder - use day_low if available
        volume_930_to_signal=int(c.volume_today or 0),
        adv_20d=c.adv_dollars,      # proxy ADV from previous-day Massive dollar volume
        adv_dollars=c.adv_dollars,  # do NOT multiply by 50 when source is Massive
        day_return=c.day_return,
        volume_ratio=1.0,  # placeholder unless you compute today's vol / ADV
        close_position=c.close_position,
        late_drop_1530_1550=0.0,
        selection_score=c.selection_score,
    )

    # Preserve prior_ret for audit/diagnostics (may not be in original dataclass)
    result.prior_ret = c.prior_ret
    return result
