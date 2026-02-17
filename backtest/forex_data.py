"""
Forex Historical Data Fetcher

Uses yfinance to pull free daily OHLCV data for forex pairs.
Yahoo Finance tickers use format: EURUSD=X, GBPUSD=X, etc.
Data available back to ~2003 for most major pairs.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import yfinance as yf
import pandas as pd


@dataclass
class ForexPair:
    symbol: str          # e.g. "EURUSD"
    yahoo_ticker: str    # e.g. "EURUSD=X"
    base: str            # e.g. "EUR"
    quote: str           # e.g. "USD"


# Major and minor pairs — all quoted vs USD for consistency
PAIR_MAP = {
    # Majors
    "EURUSD": ForexPair("EURUSD", "EURUSD=X", "EUR", "USD"),
    "GBPUSD": ForexPair("GBPUSD", "GBPUSD=X", "GBP", "USD"),
    "USDJPY": ForexPair("USDJPY", "JPY=X", "USD", "JPY"),
    "USDCHF": ForexPair("USDCHF", "CHF=X", "USD", "CHF"),
    "AUDUSD": ForexPair("AUDUSD", "AUDUSD=X", "AUD", "USD"),
    "NZDUSD": ForexPair("NZDUSD", "NZDUSD=X", "NZD", "USD"),
    "USDCAD": ForexPair("USDCAD", "CAD=X", "USD", "CAD"),
    # Commodity / EM
    "USDSEK": ForexPair("USDSEK", "SEK=X", "USD", "SEK"),
    "USDNOK": ForexPair("USDNOK", "NOK=X", "USD", "NOK"),
    "USDMXN": ForexPair("USDMXN", "MXN=X", "USD", "MXN"),
    "USDZAR": ForexPair("USDZAR", "ZAR=X", "USD", "ZAR"),
    "USDSGD": ForexPair("USDSGD", "SGD=X", "USD", "SGD"),
    # Crosses (non-USD)
    "EURGBP": ForexPair("EURGBP", "EURGBP=X", "EUR", "GBP"),
    "EURJPY": ForexPair("EURJPY", "EURJPY=X", "EUR", "JPY"),
    "GBPJPY": ForexPair("GBPJPY", "GBPJPY=X", "GBP", "JPY"),
    "AUDJPY": ForexPair("AUDJPY", "AUDJPY=X", "AUD", "JPY"),
    "NZDJPY": ForexPair("NZDJPY", "NZDJPY=X", "NZD", "JPY"),
    "EURCHF": ForexPair("EURCHF", "EURCHF=X", "EUR", "CHF"),
}


@dataclass
class PairData:
    symbol: str
    dates: list[str]       # ISO date strings
    timestamps: list[int]  # ms since epoch
    opens: list[float]
    highs: list[float]
    lows: list[float]
    closes: list[float]
    volumes: list[float]


def fetch_pair(pair: ForexPair, start: str = "2003-01-01",
               end: str = "2025-01-01") -> PairData:
    """Fetch daily OHLCV from Yahoo Finance."""
    ticker = yf.Ticker(pair.yahoo_ticker)
    df = ticker.history(start=start, end=end, interval="1d")

    if df.empty:
        raise ValueError(f"No data returned for {pair.symbol} ({pair.yahoo_ticker})")

    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]

    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    timestamps = [int(d.timestamp() * 1000) for d in df.index]

    return PairData(
        symbol=pair.symbol,
        dates=dates,
        timestamps=timestamps,
        opens=df["Open"].tolist(),
        highs=df["High"].tolist(),
        lows=df["Low"].tolist(),
        closes=df["Close"].tolist(),
        volumes=df["Volume"].tolist(),
    )


def load_pair_cached(pair: ForexPair, cache_dir: str = "state/forex_cache",
                     start: str = "2003-01-01", end: str = "2025-01-01") -> PairData:
    """Load from cache or fetch and cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{pair.symbol}_daily.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            data = json.load(f)
        return PairData(**data)

    print(f"  Fetching {pair.symbol} from Yahoo Finance...")
    pair_data = fetch_pair(pair, start=start, end=end)
    print(f"    {len(pair_data.closes)} daily bars fetched")

    with open(cache_file, "w") as f:
        json.dump({
            "symbol": pair_data.symbol,
            "dates": pair_data.dates,
            "timestamps": pair_data.timestamps,
            "opens": pair_data.opens,
            "highs": pair_data.highs,
            "lows": pair_data.lows,
            "closes": pair_data.closes,
            "volumes": pair_data.volumes,
        }, f)

    return pair_data
