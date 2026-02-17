"""
Market Data Module — Fetches prices, computes SMA, Bollinger Bands.
Primary: Alpaca Market Data API. Fallback: Yahoo Finance (yfinance).
"""
import logging
import math
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
import yfinance as yf
from bot.config import ALPACA_API_KEY, ALPACA_SECRET_KEY

logger = logging.getLogger("bot.data")


def get_data_client():
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def _fetch_alpaca(tickers, lookback_days=150):
    """Fetch daily bars from Alpaca. Returns dict of ticker -> {dates, opens, closes}."""
    client = get_data_client()
    end = datetime.now()
    start = end - timedelta(days=int(lookback_days * 1.6))

    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(request)

    result = {}
    bar_data = bars.data if hasattr(bars, 'data') else {}
    for ticker in tickers:
        ticker_bars = bar_data.get(ticker, [])
        dates = []
        opens = []
        closes = []
        for bar in ticker_bars:
            dates.append(bar.timestamp.strftime("%Y-%m-%d"))
            opens.append(float(bar.open))
            closes.append(float(bar.close))
        result[ticker] = {"dates": dates, "opens": opens, "closes": closes}

    return result


def _fetch_yahoo(tickers, lookback_days=150):
    """Fetch daily bars from Yahoo Finance as fallback."""
    end = datetime.now()
    start = end - timedelta(days=int(lookback_days * 1.6))

    result = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                           end=end.strftime("%Y-%m-%d"), progress=False)
            if df.empty:
                result[ticker] = {"dates": [], "opens": [], "closes": []}
                continue
            dates = [d.strftime("%Y-%m-%d") for d in df.index]
            opens = [float(v) for v in df["Open"]]
            closes = [float(v) for v in df["Close"]]
            result[ticker] = {"dates": dates, "opens": opens, "closes": closes}
        except Exception as e:
            logger.error(f"Yahoo fallback failed for {ticker}: {e}")
            result[ticker] = {"dates": [], "opens": [], "closes": []}

    return result


def fetch_daily_bars(tickers, lookback_days=150):
    """Fetch daily bars — Alpaca primary, Yahoo Finance fallback.
    Any ticker that returns empty from Alpaca is retried via Yahoo."""
    # Try Alpaca first
    try:
        result = _fetch_alpaca(tickers, lookback_days)
    except Exception as e:
        logger.warning(f"Alpaca data fetch failed: {e} -- falling back to Yahoo")
        return _fetch_yahoo(tickers, lookback_days)

    # Check for any empty tickers and fill from Yahoo
    missing = [t for t in tickers if not result.get(t, {}).get("closes")]
    if missing:
        logger.warning(f"Alpaca returned no data for {missing} -- trying Yahoo fallback")
        yahoo_data = _fetch_yahoo(missing, lookback_days)
        for t in missing:
            if yahoo_data.get(t, {}).get("closes"):
                result[t] = yahoo_data[t]
                logger.info(f"Yahoo fallback OK for {t}: {len(yahoo_data[t]['closes'])} bars")
            else:
                logger.error(f"No data from either source for {t}")

    return result


def compute_sma(closes, period):
    """Compute SMA of the last `period` closes."""
    if len(closes) < period:
        return closes[-1] if closes else 0
    return sum(closes[-period:]) / period


def compute_std(closes, period):
    """Compute population std dev of the last `period` closes."""
    if len(closes) < period:
        return 0
    window = closes[-period:]
    mean = sum(window) / period
    return math.sqrt(sum((x - mean) ** 2 for x in window) / period)


def compute_bollinger_lower(closes, period=20, sigma=2.0):
    """Compute lower Bollinger Band."""
    sma = compute_sma(closes, period)
    std = compute_std(closes, period)
    return sma - sigma * std


def fetch_live_prices(tickers):
    """Get real-time prices via Alpaca latest trade. Fallback to yfinance.
    Use this for intraday checks — daily bars only give yesterday's close."""
    prices = {}

    # Try Alpaca latest trade
    try:
        client = get_data_client()
        request = StockLatestTradeRequest(symbol_or_symbols=tickers)
        trades = client.get_stock_latest_trade(request)
        for ticker in tickers:
            if ticker in trades and trades[ticker]:
                prices[ticker] = float(trades[ticker].price)
                logger.debug(f"Live price {ticker}: ${prices[ticker]:.2f} (Alpaca latest trade)")
    except Exception as e:
        logger.warning(f"Alpaca latest trade failed: {e}")

    # Fallback: yfinance for any missing tickers
    missing = [t for t in tickers if t not in prices]
    if missing:
        for ticker in missing:
            try:
                tk = yf.Ticker(ticker)
                info = tk.fast_info
                price = getattr(info, 'last_price', None) or getattr(info, 'previous_close', None)
                if price and price > 0:
                    prices[ticker] = float(price)
                    logger.debug(f"Live price {ticker}: ${prices[ticker]:.2f} (yfinance fallback)")
            except Exception as e:
                logger.warning(f"yfinance live price failed for {ticker}: {e}")

    return prices


def get_current_prices(tickers):
    """Get the most recent close price for each ticker."""
    bars = fetch_daily_bars(tickers, lookback_days=5)
    prices = {}
    for ticker in tickers:
        if ticker in bars and bars[ticker]["closes"]:
            prices[ticker] = bars[ticker]["closes"][-1]
    return prices


def get_historical_closes(ticker, lookback_days=150):
    """Get list of historical close prices for a single ticker."""
    bars = fetch_daily_bars([ticker], lookback_days=lookback_days)
    if ticker in bars:
        return bars[ticker]["dates"], bars[ticker]["closes"]
    return [], []


def is_first_trading_day_of_week(bars_dates):
    """Check if today is the first trading day of this week.
    Compare the weekday of the last date vs the second-to-last date.
    If last weekday <= previous weekday, it's a new week."""
    if len(bars_dates) < 2:
        return False
    last = datetime.strptime(bars_dates[-1], "%Y-%m-%d").weekday()
    prev = datetime.strptime(bars_dates[-2], "%Y-%m-%d").weekday()
    return last <= prev
