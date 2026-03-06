"""Core configuration for the morning momentum bot - Gap Strategy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Strategy parameters and guardrails - Gap Momentum Strategy."""

    # Strategy window - New Timeline
    universe_build_time: str = "08:30"     # Build 4,000-symbol universe
    broad_filter_start: str = "08:30"      # Start delayed_sip broad filter
    broad_filter_end: str = "08:40"        # End delayed_sip broad filter
    first_pool_save: str = "08:45"         # Save first filtered pool
    first_refinement: str = "09:05"        # First live IEX refinement
    second_refinement: str = "09:15"       # Second live IEX refinement
    candidate_freeze: str = "09:25"        # Freeze candidates, build watchlist
    stream_start: str = "09:28"            # Start IEX live stream
    market_open: str = "09:30"             # Market open
    entry_start: str = "09:40"             # Entry time (after 10-min confirmation)
    entry_cutoff: str = "14:30"            # Last entry allowed
    hard_exit: str = "14:30"               # Exit all positions
    
    # Data feed strategy
    universe_filter_feed: str = "delayed_sip"  # Broad filter (8:30-8:40)
    preopen_refine_feed: str = "iex"           # First refinement (9:05)
    final_preopen_refresh_feed: str = "iex"    # Second refinement (9:15)
    live_stream_feed: str = "iex"              # Live stream (9:28 onward)
    live_quote_refresh_feed: str = "iex"       # Live quote refresh (9:30 onward)

    # Universe filters (gap strategy - WIDER NET)
    min_price: float = 1.0  # $1 min price
    max_price: float = 100.0  # $100 max price
    min_dollar_volume: float = 10_000_000  # $10M ADV floor (20-day average daily volume)
    min_5min_volume: float = 250_000  # $250K first 5-min dollar volume (enforced in EntryLoop)
    min_gap_pct: float = 0.05  # 5% min gap
    max_gap_pct: float = 0.25  # 25% max gap
    opening_breakout: bool = True  # Enter only if price > first 1-min bar high
    max_seed_universe: int = 4000  # Max symbols from Massive snapshot (8:30 AM universe build)
    first_filter_pool_size: int = 800  # Max after delayed_sip broad filter (8:30-8:40)
    max_candidates_returned: int = 300  # Max after IEX refinements (9:05, 9:15)
    max_subscribe_symbols: int = 25    # Top symbols to subscribe to stream (9:28)
    max_candidates_monitored: int = 12  # Top watchlist for active trading (9:25)

    # Position sizing (85% of available cash)
    daily_deploy_pct: float = 0.85  # 85% of equity deployed daily
    max_per_ticker_pct: float = 0.25  # 25% max per position (concentration cap)
    max_position_pct_of_5min_vol: float = 0.01  # Max 1% of morning 5-min volume (volume participation cap)
    max_adv_pct: float = 0.01  # 1% of 20-day ADV cap
    min_order_dollars: float = 25.0  # Minimum order size in dollars
    use_smaller_of_sizing_or_vol_cap: bool = True  # Hard cap by smaller of risk sizing or volume cap

    # Risk guardrails
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_concurrent: int = 25  # Max open positions
    max_trades_per_day: int = 25  # Max trades per day
    daily_kill_r: float = -3.0  # Stop trading if R <= -3R

    # Exit rules
    breakeven_at_pct: float = 0.006  # Move stop to breakeven after +0.6%
    take_profit_pct: float = 0.05  # Enable trailing once gains reach 5%
    trail_pct: float = 0.015  # 1.5% trailing stop
    stop_loss_pct: float = 0.05  # 5% hard stop
    stop_atr_mult: float = 2.0  # ATR multiplier for initial stop
    stop_min_pct: float = 0.02  # 2% minimum stop
    stop_max_pct: float = 0.05  # 5% maximum stop

    # Excluded instruments (leveraged ETFs, vol ETPs, commodity leveraged)
    excluded_symbols: tuple = (
        # Leveraged equity ETFs
        "TQQQ", "SQQQ", "UPRO", "SPXU", "SSO", "SDS", "QLD", "QID",
        "SOXL", "SOXS", "TNA", "TZA", "FAS", "FAZ",
        # Volatility ETPs
        "UVXY", "VXX", "SVXY", "VIXY",
        # Commodity leveraged ETFs
        "UCO", "SCO", "BOIL", "KOLD",
        "NUGT", "DUST", "JNUG", "JDST",
        "GUSH", "DRIP",
        "ERX", "ERY",
    )

    # Execution
    slippage_pct: float = 0.001  # 10 bps slippage
    exec_slippage_buy_pct: float = 0.001
    exec_slippage_sell_pct: float = 0.001
    exit_ack_timeout_seconds: float = 5.0  # Seconds to wait before exit reconciliation
    
    # Performance tracking & adaptive allocation
    slippage_threshold_pct: float = 0.015  # 1.5% sustained slippage triggers reduction
    reduced_allocation_pct: float = 0.30  # Reduce to 30% if slippage exceeds threshold
    slippage_lookback_trades: int = 20  # Rolling window for slippage calculation
    metrics_log_file: str = "state/performance_metrics.jsonl"  # Daily metrics log
