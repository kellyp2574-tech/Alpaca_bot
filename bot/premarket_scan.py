"""Premarket scanning pipeline for the gap momentum strategy."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List

from .clock import window_from_strings
from .morning_config import Config
from .storage import Candidate


def build_candidates(
    cfg: Config,
    alpaca,
    symbols: Iterable[str],
    date: datetime,
) -> List[Candidate]:
    """Build candidates based on gap strategy filters.

    Args:
        cfg: Strategy configuration.
        alpaca: Adapter exposing Alpaca data helpers.
        symbols: Iterable of symbols from most-actives list.
        date: Trading date.
    """

    scan_window = window_from_strings(
        reference=date,
        start_str=cfg.scan_start,
        end_str=cfg.scan_end,
    )
    
    # Get daily bars for gap calculation
    daily = alpaca.get_daily_bars(
        list(symbols), lookback_days=35, end_dt=scan_window.start
    )
    
    # Get all premarket bars at once (more efficient than one-by-one)
    pm_bars_all = alpaca.get_bars(
        list(symbols),
        timeframe="1Min",
        start=scan_window.start,
        end=scan_window.end,
    )
    
    # Get 1-min bars for 9:30-9:35 and aggregate for 5-minute checks
    market_open = window_from_strings(
        reference=date,
        start_str="09:30",
        end_str="09:35",
    )
    min1_bars = alpaca.get_bars(
        list(symbols),
        timeframe="1Min",
        start=market_open.start,
        end=market_open.end,
    )

    candidates: List[Candidate] = []
    for symbol in symbols:
        daily_stats = daily.get(symbol)
        if not daily_stats:
            continue

        prev_close = daily_stats.prev_close
        if prev_close <= 0:
            continue

        # Get latest premarket price from pre-fetched bars
        pm_bars = pm_bars_all.get(symbol, [])
        
        if not pm_bars:
            continue
            
        pm_last = pm_bars[-1].c
        pm_volume = sum(bar.v for bar in pm_bars)
        
        # Price filter
        price = pm_last
        if not (cfg.min_price <= price <= cfg.max_price):
            continue
            
        # Dollar volume filter
        avg_vol_30d = daily_stats.avg_vol_30d
        dollar_volume = avg_vol_30d * prev_close if avg_vol_30d else 0
        if dollar_volume < cfg.min_dollar_volume:
            continue
            
        # First 5-min volume filter (require exactly 5 bars for 9:30-9:35 window)
        first_1min = min1_bars.get(symbol, [])
        if len(first_1min) < 5:
            continue  # Need complete 5-minute window
        first_1min = first_1min[:5]  # Use exactly first 5 minutes
        
        # Aggregate 1-min bars for 5-minute volume
        vol_5min = sum(bar.v for bar in first_1min)
        dollar_vol_5min = sum(bar.v * bar.c for bar in first_1min)  # True dollar volume
        if dollar_vol_5min < cfg.min_5min_volume:
            continue

        # Gap calculation
        gap_pct = (pm_last - prev_close) / prev_close
        
        # Gap range filter
        if not (cfg.min_gap_pct <= gap_pct <= cfg.max_gap_pct):
            continue

        # Opening strength filter: 9:35 close > 9:30 open
        if cfg.opening_strength and first_1min:
            open_930 = first_1min[0].o  # First bar's open (9:30)
            close_935 = first_1min[-1].c  # Last bar's close (9:35)
            if close_935 <= open_930:
                continue

        candidates.append(
            Candidate(
                symbol=symbol,
                price=price,
                prev_close=prev_close,
                pm_last=pm_last,
                pm_high=max(bar.h for bar in pm_bars) if pm_bars else pm_last,
                pm_volume=pm_volume,
                avg_vol_30d=avg_vol_30d,
                float_shares=daily_stats.float_shares if hasattr(daily_stats, 'float_shares') else 0,
                gap_pct=gap_pct,
                pm_vol_float=0,
                relvol=pm_volume / avg_vol_30d if avg_vol_30d > 0 else 0,
                score=gap_pct,  # Score by gap size
            )
        )

    # Sort by gap size (largest gaps first)
    candidates.sort(key=lambda c: c.gap_pct, reverse=True)
    return candidates
