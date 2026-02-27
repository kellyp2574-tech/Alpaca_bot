"""Core configuration for the morning momentum bot - Gap Strategy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Strategy parameters and guardrails - Gap Momentum Strategy."""

    # Strategy window
    scan_start: str = "09:00"      # Start scanning at 9am
    scan_end: str = "09:29"        # End scanning at 9:29 (same as final_scan)
    scan_interval_minutes: int = 5  # Scan every 5 min
    final_scan: str = "09:29"       # Final candidate build
    entry_start: str = "09:35"      # Entry time
    entry_cutoff: str = "10:30"     # Last entry allowed
    hard_exit: str = "10:30"        # Exit all positions
    market_open: str = "09:30"

    # Universe filters (gap strategy - WIDER NET)
    min_price: float = 2.0
    max_price: float = 100.0
    min_dollar_volume: float = 0  # Daily volume baseline OFF (EntryLoop enforces real constraint)
    min_5min_volume: float = 250_000  # $250K first 5-min dollar volume (enforced in EntryLoop)
    min_gap_pct: float = 0.05  # 5% min gap (wider bucket)
    max_gap_pct: float = 0.25  # 25% max gap (wider bucket)
    opening_strength: bool = True  # First 5-min candle must be green
    max_seed_universe: int = 600  # Max symbols from Massive snapshot
    max_candidates_returned: int = 300  # Max candidates returned from scanner (wide funnel)
    max_candidates_monitored: int = 35  # Max candidates to actively monitor/trade (focused)

    # Position sizing (50% of available cash)
    daily_deploy_pct: float = 0.50  # 50% of cash per day (no fixed hard cap, scales with account)
    max_position_pct_of_5min_vol: float = 0.01  # Max 1% of morning 5-min volume
    use_smaller_of_sizing_or_vol_cap: bool = True  # Hard cap by smaller of risk sizing or volume cap

    # Risk guardrails
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_concurrent: int = 25  # Max open positions
    max_trades_per_day: int = 25  # Max trades per day
    daily_kill_r: float = -3.0  # Stop trading if R <= -3R

    # Exit rules
    breakeven_at_pct: float = 0.006  # Move stop to breakeven after +0.6%
    take_profit_pct: float = 0.012  # Enable trailing once gains reach 1.2%
    trail_pct: float = 0.01  # 1% trailing stop
    stop_loss_pct: float = 0.05  # 5% hard stop
    stop_atr_mult: float = 2.0  # ATR multiplier for initial stop
    stop_min_pct: float = 0.02  # 2% minimum stop
    stop_max_pct: float = 0.05  # 5% maximum stop

    # Execution
    slippage_pct: float = 0.005  # 0.5% base slippage
    exec_slippage_buy_pct: float = 0.002
    exec_slippage_sell_pct: float = 0.005
    exit_ack_timeout_seconds: float = 5.0  # Seconds to wait before exit reconciliation
    
    # Performance tracking & adaptive allocation
    slippage_threshold_pct: float = 0.015  # 1.5% sustained slippage triggers reduction
    reduced_allocation_pct: float = 0.30  # Reduce to 30% if slippage exceeds threshold
    slippage_lookback_trades: int = 20  # Rolling window for slippage calculation
    metrics_log_file: str = "state/performance_metrics.jsonl"  # Daily metrics log
