"""
Comprehensive Monitoring System for the Gap Momentum Trading Bot.

Collects metrics across 11 sections:
1. Daily top-line dashboard
2. Funnel metrics (candidate pipeline)
3. Entry execution quality
4. Exit execution quality
5. Trade outcome stats
6. Running tallies (intraday / daily / weekly / MTD / all-time / rolling)
7. Risk and exposure stats
8. Data integrity stats
9. Broker/order integrity stats
10. Strategy drift diagnostics
11. Alerts

Data flows into a global SessionMonitor singleton that any module can import.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from zoneinfo import ZoneInfo

from .state_manager import _atomic_write_json

logger = logging.getLogger("monitoring")

MARKET_TZ = ZoneInfo("America/New_York")

# ═══════════════════════════════════════════════════════════════
# Section 1 – Daily Dashboard
# ═══════════════════════════════════════════════════════════════

@dataclass
class DashboardMetrics:
    date: str = ""
    bot_start_time: str = ""
    bot_stop_time: str = ""
    strategy_modules_run: List[str] = field(default_factory=list)
    market_regime: str = ""
    market_open_status: bool = False

    account_equity_start: float = 0.0
    account_equity_current: float = 0.0
    cash_start: float = 0.0
    cash_current: float = 0.0

    realized_pnl_today: float = 0.0
    unrealized_pnl_current: float = 0.0
    total_pnl_today: float = 0.0

    trades_opened: int = 0
    trades_closed: int = 0
    symbols_scanned: int = 0
    candidates_found: int = 0
    entries_attempted: int = 0
    entries_filled: int = 0
    partial_fills: int = 0
    canceled_entries: int = 0
    forced_exits: int = 0
    current_open_positions: int = 0


# ═══════════════════════════════════════════════════════════════
# Section 2 – Funnel Metrics
# ═══════════════════════════════════════════════════════════════

@dataclass
class FunnelMetrics:
    total_starting_universe: int = 0
    symbols_with_valid_data: int = 0
    passing_price_filter: int = 0
    passing_gap_filter: int = 0
    passing_liquidity_filter: int = 0
    passing_relvol_filter: int = 0
    passing_all_filters: int = 0
    final_ranked_candidates: int = 0
    selected_for_sizing: int = 0

    skipped_capital_constraints: int = 0
    skipped_volume_cap: int = 0
    skipped_day_lockout: int = 0
    skipped_missing_prev_close: int = 0
    skipped_stale_minute_bars: int = 0

    # Drop detail from CandidateLedger
    drop_reasons: Dict[str, int] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Section 3 – Entry Execution Quality
# ═══════════════════════════════════════════════════════════════

@dataclass
class EntryOrderMetric:
    symbol: str = ""
    client_order_id: str = ""  # Unique identifier for lifecycle tracking
    signal_ts: str = ""
    submit_ts: str = ""
    intended_shares: float = 0.0
    intended_notional: float = 0.0
    filled_shares: float = 0.0
    filled_notional: float = 0.0
    intended_price: float = 0.0
    submitted_limit: float = 0.0
    avg_fill_price: float = 0.0
    slippage_dollars: float = 0.0
    slippage_bps: float = 0.0
    fill_pct: float = 0.0
    time_to_first_fill_s: float = 0.0
    time_to_full_fill_s: float = 0.0
    status: str = ""  # filled / partial / canceled / expired / rejected / unknown
    cancel_reason: str = ""  # time / price / manual / broker / other
    is_fractional: bool = False
    tif: str = ""  # IOC / DAY


@dataclass
class EntryAggregateMetrics:
    avg_slippage_bps: float = 0.0
    median_slippage_bps: float = 0.0
    p95_slippage_bps: float = 0.0
    worst_slippage_bps: float = 0.0
    avg_fill_pct: float = 0.0
    avg_time_to_first_fill: float = 0.0
    avg_time_to_full_fill: float = 0.0
    pct_fully_filled: float = 0.0
    pct_partially_filled: float = 0.0
    pct_canceled_unfilled: float = 0.0
    pct_canceled_60s: float = 0.0
    pct_canceled_1pct: float = 0.0


# ═══════════════════════════════════════════════════════════════
# Section 4 – Exit Execution Quality
# ═══════════════════════════════════════════════════════════════

@dataclass
class ExitOrderMetric:
    symbol: str = ""
    entry_ts: str = ""
    exit_signal_ts: str = ""
    exit_submit_ts: str = ""
    planned_exit_reason: str = ""
    actual_exit_reason: str = ""
    exit_order_type: str = ""
    intended_exit_price: float = 0.0
    avg_exit_price: float = 0.0
    exit_slippage_bps: float = 0.0
    time_to_fill_s: float = 0.0
    partial_exit: bool = False
    force_flat: bool = False
    force_flat_reason: str = ""


@dataclass
class ExitAggregateMetrics:
    avg_slippage_bps: float = 0.0
    median_slippage_bps: float = 0.0
    p95_slippage_bps: float = 0.0
    pct_force_flat: float = 0.0
    pct_partial_before_completion: float = 0.0
    avg_signal_to_exit_delay_s: float = 0.0


# ═══════════════════════════════════════════════════════════════
# Section 5 – Trade Outcome Stats (per-trade)
# ═══════════════════════════════════════════════════════════════

@dataclass
class TradeOutcome:
    symbol: str = ""
    entry_time: str = ""
    exit_time: str = ""
    holding_time_s: float = 0.0
    gross_return_pct: float = 0.0
    net_return_pct: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    dollars_won_lost: float = 0.0
    exit_reason: str = ""
    gap_at_entry: float = 0.0
    first_5min_volume: float = 0.0
    fill_pct: float = 0.0
    entry_slippage_bps: float = 0.0


@dataclass
class TradeAggregateStats:
    win_rate: float = 0.0
    loss_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    expectancy_pct: float = 0.0
    profit_factor: float = 0.0
    avg_holding_time_s: float = 0.0
    median_holding_time_s: float = 0.0
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    avg_return_full_fills: float = 0.0
    avg_return_partial_fills: float = 0.0


# ═══════════════════════════════════════════════════════════════
# Section 6 – Running Tallies
# ═══════════════════════════════════════════════════════════════

@dataclass
class RunningTally:
    period_label: str = ""  # "intraday", "daily", "weekly", "mtd", "all_time", "rolling_20d", etc.
    cumulative_realized_pnl: float = 0.0
    cumulative_unrealized_pnl: float = 0.0
    cumulative_net_return: float = 0.0
    cumulative_deployed_dollars: float = 0.0
    cumulative_gross_traded_notional: float = 0.0
    cumulative_fees: float = 0.0
    cumulative_slippage_dollars: float = 0.0
    cumulative_slippage_bps_avg: float = 0.0
    cumulative_entries_attempted: int = 0
    cumulative_entries_filled: int = 0
    cumulative_partial_fills: int = 0
    cumulative_canceled_entries: int = 0
    cumulative_force_flat_exits: int = 0
    cumulative_win_rate: float = 0.0
    cumulative_expectancy: float = 0.0
    cumulative_profit_factor: float = 0.0
    cumulative_max_drawdown: float = 0.0
    cumulative_avg_daily_return: float = 0.0
    cumulative_avg_trade_return: float = 0.0


# ═══════════════════════════════════════════════════════════════
# Section 7 – Risk & Exposure
# ═══════════════════════════════════════════════════════════════

@dataclass
class RiskExposureMetrics:
    current_cash: float = 0.0
    current_equity: float = 0.0
    total_deployed: float = 0.0
    pct_deployed: float = 0.0
    open_positions_count: int = 0
    largest_position_pct: float = 0.0
    smallest_position_pct: float = 0.0
    avg_position_pct: float = 0.0
    concentration_top3: float = 0.0
    concentration_top5: float = 0.0
    total_notional_at_stops: float = 0.0
    estimated_loss_to_stops: float = 0.0
    intraday_drawdown_pct: float = 0.0
    drawdown_from_ath: float = 0.0
    capital_reserved_pending: float = 0.0
    pending_order_notional: float = 0.0

    # Daily risk summary
    max_capital_deployed_today: float = 0.0
    max_simultaneous_positions: int = 0
    largest_single_exposure: float = 0.0
    max_intraday_drawdown: float = 0.0
    max_open_risk: float = 0.0


# ═══════════════════════════════════════════════════════════════
# Section 8 – Data Integrity
# ═══════════════════════════════════════════════════════════════

@dataclass
class DataIntegrityMetrics:
    symbols_missing_prev_close: int = 0
    symbols_missing_open: int = 0
    symbols_stale_quotes: int = 0
    symbols_missing_minute_bars: int = 0
    symbols_zero_volume: int = 0
    api_request_count: int = 0
    api_failure_count: int = 0
    api_timeout_count: int = 0
    rate_limit_hits: int = 0
    broker_reconnect_count: int = 0
    quote_refresh_latency_ms: float = 0.0
    bar_refresh_latency_ms: float = 0.0
    time_since_last_good_update_s: float = 0.0
    last_good_update_ts: str = ""


# ═══════════════════════════════════════════════════════════════
# Section 9 – Broker / Order Integrity
# ═══════════════════════════════════════════════════════════════

@dataclass
class BrokerIntegrityMetrics:
    rejected_orders: int = 0
    canceled_orders: int = 0
    expired_orders: int = 0
    unknown_order_statuses: int = 0
    orphan_fills: int = 0
    pending_entries: int = 0
    pending_exits: int = 0
    reconciliation_corrections: int = 0
    position_mismatches: int = 0
    cash_mismatches: int = 0
    bot_recoveries: int = 0


# ═══════════════════════════════════════════════════════════════
# Section 10 – Strategy Drift Diagnostics
# ═══════════════════════════════════════════════════════════════

@dataclass
class DriftDiagnostics:
    returns_by_day_of_week: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    returns_by_gap_bucket: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    returns_by_fill_time_bucket: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    returns_by_slippage_bucket: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    returns_by_partial_vs_full: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    returns_by_entry_time_bucket: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    returns_by_candidate_count_bucket: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))


# ═══════════════════════════════════════════════════════════════
# Section 11 – Alerts
# ═══════════════════════════════════════════════════════════════

@dataclass
class AlertThresholds:
    missing_prev_close_rate: float = 0.10    # 10% of universe
    candidate_count_low: int = 3
    fill_rate_low: float = 0.30              # 30%
    avg_slippage_bps_high: float = 50.0      # 50 bps
    partial_fill_rate_high: float = 0.50     # 50%
    rejection_count_high: int = 3
    force_flat_count_high: int = 3
    daily_drawdown_pct_high: float = 0.03    # 3%
    stale_data_seconds: float = 120.0        # 2 minutes
    idle_during_market_seconds: float = 300.0  # 5 minutes


@dataclass
class Alert:
    timestamp: str
    severity: str  # "WARNING", "CRITICAL"
    category: str
    message: str


# ═══════════════════════════════════════════════════════════════
# SessionMonitor – The Central Collector
# ═══════════════════════════════════════════════════════════════

class SessionMonitor:
    """
    Central metrics collector for a single trading session.

    Usage:
        monitor = get_session_monitor()
        monitor.record_session_start(equity, cash)
        monitor.record_funnel(...)
        monitor.record_entry_order(...)
        ...
        monitor.generate_eod_report()
    """

    def __init__(self, state_dir: str = None):
        self.state_dir = Path(state_dir or "state")
        self.reports_dir = self.state_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Section instances
        self.dashboard = DashboardMetrics()
        self.funnel = FunnelMetrics()
        self.entry_orders: List[EntryOrderMetric] = []
        self.exit_orders: List[ExitOrderMetric] = []
        self.trade_outcomes: List[TradeOutcome] = []
        self.risk = RiskExposureMetrics()
        self.data_integrity = DataIntegrityMetrics()
        self.broker_integrity = BrokerIntegrityMetrics()
        self.drift = DriftDiagnostics()
        self.alerts: List[Alert] = []
        self.alert_thresholds = AlertThresholds()

        # Intraday equity tracking for drawdown
        self._equity_high_water: float = 0.0
        self._equity_snapshots: List[Tuple[str, float]] = []
        self._last_activity_ts: float = time.monotonic()

        # Rolling stats persistence
        self._rolling_stats_file = self.reports_dir / "rolling_stats.json"
        self._rolling_stats: Dict[str, Any] = self._load_rolling_stats()

        # Daily trade tracking for tallies
        self._daily_entries_attempted: int = 0
        self._daily_slippage_dollars: float = 0.0
        self._daily_gross_notional: float = 0.0

    # ───────────────────────────────────────────────────
    # Session lifecycle
    # ───────────────────────────────────────────────────

    def record_session_start(
        self,
        equity: float,
        cash: float,
        *,
        strategy_modules: Optional[List[str]] = None,
    ) -> None:
        now = datetime.now(MARKET_TZ)
        self.dashboard.date = now.strftime("%Y-%m-%d")
        self.dashboard.bot_start_time = now.strftime("%H:%M:%S")
        self.dashboard.account_equity_start = equity
        self.dashboard.account_equity_current = equity
        self.dashboard.cash_start = cash
        self.dashboard.cash_current = cash
        self.dashboard.strategy_modules_run = strategy_modules or ["morning_momentum"]
        self._equity_high_water = equity
        self._last_activity_ts = time.monotonic()
        logger.info("Session monitor started: equity=$%.2f cash=$%.2f", equity, cash)

    def record_session_stop(self) -> None:
        now = datetime.now(MARKET_TZ)
        self.dashboard.bot_stop_time = now.strftime("%H:%M:%S")

    def update_account(self, equity: float, cash: float) -> None:
        self.dashboard.account_equity_current = equity
        self.dashboard.cash_current = cash
        self._equity_snapshots.append(
            (datetime.now(MARKET_TZ).strftime("%H:%M:%S"), equity)
        )
        if equity > self._equity_high_water:
            self._equity_high_water = equity
        # Intraday drawdown
        if self._equity_high_water > 0:
            dd = (self._equity_high_water - equity) / self._equity_high_water
            self.risk.intraday_drawdown_pct = dd
            if dd > self.risk.max_intraday_drawdown:
                self.risk.max_intraday_drawdown = dd
        self._last_activity_ts = time.monotonic()

    def update_market_status(self, is_open: bool, regime: str = "") -> None:
        self.dashboard.market_open_status = is_open
        self.dashboard.market_regime = regime

    # ───────────────────────────────────────────────────
    # Section 2 – Funnel
    # ───────────────────────────────────────────────────

    def record_funnel(
        self,
        *,
        starting_universe: int = 0,
        valid_data: int = 0,
        pass_price: int = 0,
        pass_gap: int = 0,
        pass_liquidity: int = 0,
        pass_relvol: int = 0,
        pass_all: int = 0,
        final_ranked: int = 0,
        selected_for_sizing: int = 0,
        drop_reasons: Optional[Dict[str, int]] = None,
    ) -> None:
        self.funnel.total_starting_universe = starting_universe
        self.funnel.symbols_with_valid_data = valid_data
        self.funnel.passing_price_filter = pass_price
        self.funnel.passing_gap_filter = pass_gap
        self.funnel.passing_liquidity_filter = pass_liquidity
        self.funnel.passing_relvol_filter = pass_relvol
        self.funnel.passing_all_filters = pass_all
        self.funnel.final_ranked_candidates = final_ranked
        self.funnel.selected_for_sizing = selected_for_sizing
        self.dashboard.symbols_scanned = starting_universe
        self.dashboard.candidates_found = final_ranked
        if drop_reasons:
            self.funnel.drop_reasons = dict(drop_reasons)

    def record_funnel_skip(self, reason: str) -> None:
        if reason == "capital_constraints":
            self.funnel.skipped_capital_constraints += 1
        elif reason == "volume_cap":
            self.funnel.skipped_volume_cap += 1
        elif reason == "day_lockout":
            self.funnel.skipped_day_lockout += 1
        elif reason == "missing_prev_close":
            self.funnel.skipped_missing_prev_close += 1
        elif reason == "stale_minute_bars":
            self.funnel.skipped_stale_minute_bars += 1

    # ───────────────────────────────────────────────────
    # Section 3 – Entry Execution
    # ───────────────────────────────────────────────────

    def record_entry_order(
        self,
        symbol: str,
        intended_qty: float,
        intended_price: float,
        submitted_limit: float,
        filled_qty: float,
        avg_fill_price: float,
        status: str,
        *,
        client_order_id: str = "",
        signal_ts: str = "",
        submit_ts: str = "",
        time_to_first_fill_s: float = 0.0,
        time_to_full_fill_s: float = 0.0,
        cancel_reason: str = "",
        is_fractional: bool = False,
        tif: str = "IOC",
    ) -> None:
        intended_notional = intended_qty * intended_price
        filled_notional = filled_qty * avg_fill_price
        slip_dollars = (avg_fill_price - intended_price) * filled_qty if filled_qty > 0 else 0.0
        slip_bps = ((avg_fill_price - intended_price) / intended_price * 10000) if intended_price > 0 and filled_qty > 0 else 0.0
        fill_pct = (filled_qty / intended_qty * 100) if intended_qty > 0 else 0.0

        metric = EntryOrderMetric(
            symbol=symbol,
            client_order_id=client_order_id,
            signal_ts=signal_ts,
            submit_ts=submit_ts or datetime.now(MARKET_TZ).isoformat(),
            intended_shares=intended_qty,
            intended_notional=intended_notional,
            filled_shares=filled_qty,
            filled_notional=filled_notional,
            intended_price=intended_price,
            submitted_limit=submitted_limit,
            avg_fill_price=avg_fill_price,
            slippage_dollars=slip_dollars,
            slippage_bps=slip_bps,
            fill_pct=fill_pct,
            time_to_first_fill_s=time_to_first_fill_s,
            time_to_full_fill_s=time_to_full_fill_s,
            status=status,
            cancel_reason=cancel_reason,
            is_fractional=is_fractional,
            tif=tif,
        )
        self.entry_orders.append(metric)

        # Update dashboard (only for new entries, not updates)
        if status != "unknown":  # Don't count unknown/pending as attempted yet
            self._daily_entries_attempted += 1
            self.dashboard.entries_attempted = self._daily_entries_attempted
        if status == "filled":
            self.dashboard.entries_filled += 1
            self.dashboard.trades_opened += 1
        elif status == "partial":
            self.dashboard.partial_fills += 1
            self.dashboard.trades_opened += 1
        elif status in ("canceled", "unfilled", "expired"):
            self.dashboard.canceled_entries += 1

        # Accumulate daily slippage and notional
        self._daily_slippage_dollars += abs(slip_dollars)
        self._daily_gross_notional += filled_notional
        self._last_activity_ts = time.monotonic()

    def update_entry_order(
        self,
        client_order_id: str,
        *,
        filled_qty: float = None,
        avg_fill_price: float = None,
        status: str = None,
        time_to_first_fill_s: float = None,
        time_to_full_fill_s: float = None,
        cancel_reason: str = None,
    ) -> bool:
        """
        Update an existing entry order record by client_order_id.
        Returns True if found and updated, False otherwise.
        """
        for metric in self.entry_orders:
            if metric.client_order_id == client_order_id:
                # Update fields if provided
                if filled_qty is not None:
                    old_filled = metric.filled_shares
                    metric.filled_shares = filled_qty
                    metric.filled_notional = filled_qty * metric.avg_fill_price if metric.avg_fill_price > 0 else 0.0
                    metric.fill_pct = (filled_qty / metric.intended_shares * 100) if metric.intended_shares > 0 else 0.0
                    
                if avg_fill_price is not None:
                    metric.avg_fill_price = avg_fill_price
                    metric.filled_notional = metric.filled_shares * avg_fill_price
                    metric.slippage_dollars = (avg_fill_price - metric.intended_price) * metric.filled_shares if metric.filled_shares > 0 else 0.0
                    metric.slippage_bps = ((avg_fill_price - metric.intended_price) / metric.intended_price * 10000) if metric.intended_price > 0 and metric.filled_shares > 0 else 0.0
                    
                old_status = metric.status
                if status is not None:
                    metric.status = status
                    
                if time_to_first_fill_s is not None:
                    metric.time_to_first_fill_s = time_to_first_fill_s
                    
                if time_to_full_fill_s is not None:
                    metric.time_to_full_fill_s = time_to_full_fill_s
                    
                if cancel_reason is not None:
                    metric.cancel_reason = cancel_reason
                
                # Update dashboard counters based on status change
                if status is not None and status != old_status:
                    # Remove old status count
                    if old_status == "unknown":
                        # Now it's resolved, count as attempted
                        self._daily_entries_attempted += 1
                        self.dashboard.entries_attempted = self._daily_entries_attempted
                    elif old_status == "filled":
                        self.dashboard.entries_filled -= 1
                        self.dashboard.trades_opened -= 1
                    elif old_status == "partial":
                        self.dashboard.partial_fills -= 1
                        self.dashboard.trades_opened -= 1
                    elif old_status in ("canceled", "unfilled", "expired"):
                        self.dashboard.canceled_entries -= 1
                    
                    # Add new status count
                    if status == "filled":
                        self.dashboard.entries_filled += 1
                        self.dashboard.trades_opened += 1
                    elif status == "partial":
                        self.dashboard.partial_fills += 1
                        self.dashboard.trades_opened += 1
                    elif status in ("canceled", "unfilled", "expired"):
                        self.dashboard.canceled_entries += 1
                
                return True
        return False

    def compute_entry_aggregates(self) -> EntryAggregateMetrics:
        agg = EntryAggregateMetrics()
        if not self.entry_orders:
            return agg

        total = len(self.entry_orders)
        slippages = [o.slippage_bps for o in self.entry_orders if o.filled_shares > 0]
        fill_pcts = [o.fill_pct for o in self.entry_orders]
        ttff = [o.time_to_first_fill_s for o in self.entry_orders if o.time_to_first_fill_s > 0]
        ttfull = [o.time_to_full_fill_s for o in self.entry_orders if o.time_to_full_fill_s > 0]
        filled = sum(1 for o in self.entry_orders if o.status == "filled")
        partial = sum(1 for o in self.entry_orders if o.status == "partial")
        canceled = sum(1 for o in self.entry_orders if o.status in ("canceled", "unfilled", "expired"))
        cancel_60s = sum(1 for o in self.entry_orders if o.cancel_reason == "time")
        cancel_1pct = sum(1 for o in self.entry_orders if o.cancel_reason == "price")

        if slippages:
            agg.avg_slippage_bps = statistics.mean(slippages)
            agg.median_slippage_bps = statistics.median(slippages)
            sorted_s = sorted(slippages)
            idx_95 = min(int(len(sorted_s) * 0.95), len(sorted_s) - 1)
            agg.p95_slippage_bps = sorted_s[idx_95]
            agg.worst_slippage_bps = sorted_s[-1]
        if fill_pcts:
            agg.avg_fill_pct = statistics.mean(fill_pcts)
        if ttff:
            agg.avg_time_to_first_fill = statistics.mean(ttff)
        if ttfull:
            agg.avg_time_to_full_fill = statistics.mean(ttfull)

        agg.pct_fully_filled = filled / total * 100 if total else 0
        agg.pct_partially_filled = partial / total * 100 if total else 0
        agg.pct_canceled_unfilled = canceled / total * 100 if total else 0
        agg.pct_canceled_60s = cancel_60s / total * 100 if total else 0
        agg.pct_canceled_1pct = cancel_1pct / total * 100 if total else 0
        return agg

    # ───────────────────────────────────────────────────
    # Section 4 – Exit Execution
    # ───────────────────────────────────────────────────

    def record_exit_order(
        self,
        symbol: str,
        intended_price: float,
        avg_exit_price: float,
        status: str,
        *,
        entry_ts: str = "",
        exit_signal_ts: str = "",
        exit_submit_ts: str = "",
        planned_reason: str = "",
        actual_reason: str = "",
        exit_order_type: str = "limit_ioc",
        time_to_fill_s: float = 0.0,
        partial_exit: bool = False,
        force_flat: bool = False,
        force_flat_reason: str = "",
    ) -> None:
        slip_bps = ((intended_price - avg_exit_price) / intended_price * 10000) if intended_price > 0 and avg_exit_price > 0 else 0.0

        metric = ExitOrderMetric(
            symbol=symbol,
            entry_ts=entry_ts,
            exit_signal_ts=exit_signal_ts,
            exit_submit_ts=exit_submit_ts or datetime.now(MARKET_TZ).isoformat(),
            planned_exit_reason=planned_reason,
            actual_exit_reason=actual_reason,
            exit_order_type=exit_order_type,
            intended_exit_price=intended_price,
            avg_exit_price=avg_exit_price,
            exit_slippage_bps=slip_bps,
            time_to_fill_s=time_to_fill_s,
            partial_exit=partial_exit,
            force_flat=force_flat,
            force_flat_reason=force_flat_reason,
        )
        self.exit_orders.append(metric)

        if status in ("filled", "dry_run"):
            self.dashboard.trades_closed += 1
        if force_flat:
            self.dashboard.forced_exits += 1
        self._last_activity_ts = time.monotonic()

    def compute_exit_aggregates(self) -> ExitAggregateMetrics:
        agg = ExitAggregateMetrics()
        if not self.exit_orders:
            return agg

        total = len(self.exit_orders)
        slippages = [o.exit_slippage_bps for o in self.exit_orders if o.avg_exit_price > 0]
        ff_count = sum(1 for o in self.exit_orders if o.force_flat)
        partial_count = sum(1 for o in self.exit_orders if o.partial_exit)
        delays = [o.time_to_fill_s for o in self.exit_orders if o.time_to_fill_s > 0]

        if slippages:
            agg.avg_slippage_bps = statistics.mean(slippages)
            agg.median_slippage_bps = statistics.median(slippages)
            sorted_s = sorted(slippages)
            idx_95 = min(int(len(sorted_s) * 0.95), len(sorted_s) - 1)
            agg.p95_slippage_bps = sorted_s[idx_95]

        agg.pct_force_flat = ff_count / total * 100 if total else 0
        agg.pct_partial_before_completion = partial_count / total * 100 if total else 0
        if delays:
            agg.avg_signal_to_exit_delay_s = statistics.mean(delays)
        return agg

    # ───────────────────────────────────────────────────
    # Section 5 – Trade Outcomes
    # ───────────────────────────────────────────────────

    def record_trade_outcome(
        self,
        symbol: str,
        entry_time: str,
        exit_time: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        *,
        exit_reason: str = "",
        gap_at_entry: float = 0.0,
        first_5min_volume: float = 0.0,
        max_favorable_excursion: float = 0.0,
        max_adverse_excursion: float = 0.0,
        fill_pct: float = 100.0,
        entry_slippage_bps: float = 0.0,
    ) -> None:
        buy_value = qty * entry_price
        sell_value = qty * exit_price
        pnl = sell_value - buy_value
        gross_ret = (pnl / buy_value * 100) if buy_value > 0 else 0.0
        hold_s = 0.0
        try:
            et = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            xt = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
            hold_s = (xt - et).total_seconds()
        except Exception:
            pass

        outcome = TradeOutcome(
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            holding_time_s=hold_s,
            gross_return_pct=gross_ret,
            net_return_pct=gross_ret,  # fees not tracked separately yet
            max_favorable_excursion=max_favorable_excursion,
            max_adverse_excursion=max_adverse_excursion,
            dollars_won_lost=pnl,
            exit_reason=exit_reason,
            gap_at_entry=gap_at_entry,
            first_5min_volume=first_5min_volume,
            fill_pct=fill_pct,
            entry_slippage_bps=entry_slippage_bps,
        )
        self.trade_outcomes.append(outcome)
        self.dashboard.realized_pnl_today += pnl
        self.dashboard.total_pnl_today = (
            self.dashboard.realized_pnl_today + self.dashboard.unrealized_pnl_current
        )

        # Drift diagnostics
        self._classify_for_drift(outcome)

    def compute_trade_aggregates(self) -> TradeAggregateStats:
        agg = TradeAggregateStats()
        if not self.trade_outcomes:
            return agg

        wins = [t for t in self.trade_outcomes if t.dollars_won_lost > 0]
        losses = [t for t in self.trade_outcomes if t.dollars_won_lost <= 0]
        total = len(self.trade_outcomes)

        agg.win_rate = len(wins) / total * 100 if total else 0
        agg.loss_rate = len(losses) / total * 100 if total else 0
        if wins:
            agg.avg_win_pct = statistics.mean([t.gross_return_pct for t in wins])
        if losses:
            agg.avg_loss_pct = statistics.mean([t.gross_return_pct for t in losses])

        # Expectancy
        if total > 0:
            avg_ret = statistics.mean([t.gross_return_pct for t in self.trade_outcomes])
            agg.expectancy_pct = avg_ret

        # Profit factor
        gross_wins = sum(t.dollars_won_lost for t in wins)
        gross_losses = abs(sum(t.dollars_won_lost for t in losses))
        agg.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0

        holds = [t.holding_time_s for t in self.trade_outcomes if t.holding_time_s > 0]
        if holds:
            agg.avg_holding_time_s = statistics.mean(holds)
            agg.median_holding_time_s = statistics.median(holds)

        mfes = [t.max_favorable_excursion for t in self.trade_outcomes]
        maes = [t.max_adverse_excursion for t in self.trade_outcomes]
        if mfes:
            agg.avg_mfe = statistics.mean(mfes)
        if maes:
            agg.avg_mae = statistics.mean(maes)

        full = [t for t in self.trade_outcomes if t.fill_pct >= 99.9]
        partial = [t for t in self.trade_outcomes if t.fill_pct < 99.9]
        if full:
            agg.avg_return_full_fills = statistics.mean([t.gross_return_pct for t in full])
        if partial:
            agg.avg_return_partial_fills = statistics.mean([t.gross_return_pct for t in partial])

        return agg

    # ───────────────────────────────────────────────────
    # Section 7 – Risk & Exposure (point-in-time update)
    # ───────────────────────────────────────────────────

    def update_risk_exposure(
        self,
        positions: Dict[str, Any],
        equity: float,
        cash: float,
        *,
        pending_notional: float = 0.0,
    ) -> None:
        self.risk.current_cash = cash
        self.risk.current_equity = equity
        self.risk.open_positions_count = len(positions)
        self.dashboard.current_open_positions = len(positions)

        if not positions:
            self.risk.total_deployed = 0.0
            self.risk.pct_deployed = 0.0
            self.risk.pending_order_notional = pending_notional
            return

        position_values = []
        total_at_stop = 0.0
        total_loss_to_stop = 0.0

        for sym, pos in positions.items():
            qty = getattr(pos, "qty", 0)
            entry_p = getattr(pos, "entry_price", 0)
            stop_p = getattr(pos, "stop_price", 0)
            peak_p = getattr(pos, "peak_price", entry_p)
            value = qty * entry_p
            position_values.append(value)
            stop_value = qty * stop_p
            total_at_stop += stop_value
            total_loss_to_stop += (value - stop_value)

        deployed = sum(position_values)
        self.risk.total_deployed = deployed
        self.risk.pct_deployed = (deployed / equity * 100) if equity > 0 else 0.0
        self.risk.total_notional_at_stops = total_at_stop
        self.risk.estimated_loss_to_stops = total_loss_to_stop
        self.risk.pending_order_notional = pending_notional

        if position_values:
            sorted_vals = sorted(position_values, reverse=True)
            total_val = sum(sorted_vals)
            if equity > 0:
                self.risk.largest_position_pct = sorted_vals[0] / equity * 100
                self.risk.smallest_position_pct = sorted_vals[-1] / equity * 100
                self.risk.avg_position_pct = statistics.mean(sorted_vals) / equity * 100
                self.risk.concentration_top3 = sum(sorted_vals[:3]) / equity * 100
                self.risk.concentration_top5 = sum(sorted_vals[:5]) / equity * 100

        # Track daily highs
        if deployed > self.risk.max_capital_deployed_today:
            self.risk.max_capital_deployed_today = deployed
        if len(positions) > self.risk.max_simultaneous_positions:
            self.risk.max_simultaneous_positions = len(positions)
        if position_values:
            largest = max(position_values)
            if largest > self.risk.largest_single_exposure:
                self.risk.largest_single_exposure = largest

    # ───────────────────────────────────────────────────
    # Section 8 – Data Integrity
    # ───────────────────────────────────────────────────

    def record_api_call(self, success: bool, *, is_timeout: bool = False) -> None:
        self.data_integrity.api_request_count += 1
        if not success:
            self.data_integrity.api_failure_count += 1
        if is_timeout:
            self.data_integrity.api_timeout_count += 1

    def record_rate_limit_hit(self) -> None:
        self.data_integrity.rate_limit_hits += 1

    def record_broker_reconnect(self) -> None:
        self.data_integrity.broker_reconnect_count += 1

    def record_data_quality(
        self,
        *,
        missing_prev_close: int = 0,
        missing_open: int = 0,
        stale_quotes: int = 0,
        missing_bars: int = 0,
        zero_volume: int = 0,
    ) -> None:
        self.data_integrity.symbols_missing_prev_close = missing_prev_close
        self.data_integrity.symbols_missing_open = missing_open
        self.data_integrity.symbols_stale_quotes = stale_quotes
        self.data_integrity.symbols_missing_minute_bars = missing_bars
        self.data_integrity.symbols_zero_volume = zero_volume

    def record_refresh_latency(self, quote_ms: float = 0.0, bar_ms: float = 0.0) -> None:
        if quote_ms > 0:
            self.data_integrity.quote_refresh_latency_ms = quote_ms
        if bar_ms > 0:
            self.data_integrity.bar_refresh_latency_ms = bar_ms
        self.data_integrity.last_good_update_ts = datetime.now(MARKET_TZ).isoformat()

    # ───────────────────────────────────────────────────
    # Section 9 – Broker Integrity
    # ───────────────────────────────────────────────────

    def record_broker_event(self, event_type: str, count: int = 1) -> None:
        attr = {
            "rejected": "rejected_orders",
            "canceled": "canceled_orders",
            "expired": "expired_orders",
            "unknown_status": "unknown_order_statuses",
            "orphan_fill": "orphan_fills",
            "pending_entry": "pending_entries",
            "pending_exit": "pending_exits",
            "reconciliation": "reconciliation_corrections",
            "position_mismatch": "position_mismatches",
            "cash_mismatch": "cash_mismatches",
            "bot_recovery": "bot_recoveries",
        }.get(event_type)
        if attr and hasattr(self.broker_integrity, attr):
            current = getattr(self.broker_integrity, attr)
            setattr(self.broker_integrity, attr, current + count)

    # ───────────────────────────────────────────────────
    # Section 10 – Drift Diagnostics
    # ───────────────────────────────────────────────────

    def _classify_for_drift(self, outcome: TradeOutcome) -> None:
        ret = outcome.gross_return_pct

        # Day of week
        try:
            dt = datetime.fromisoformat(outcome.entry_time.replace("Z", "+00:00"))
            dow = dt.strftime("%A")
            self.drift.returns_by_day_of_week[dow].append(ret)
        except Exception:
            pass

        # Gap bucket
        gap = abs(outcome.gap_at_entry * 100) if outcome.gap_at_entry else 0
        if gap < 5:
            bucket = "<5%"
        elif gap < 10:
            bucket = "5-10%"
        elif gap < 15:
            bucket = "10-15%"
        elif gap < 20:
            bucket = "15-20%"
        else:
            bucket = "20%+"
        self.drift.returns_by_gap_bucket[bucket].append(ret)

        # Slippage bucket
        slip = abs(outcome.entry_slippage_bps)
        if slip < 10:
            sb = "<10bps"
        elif slip < 25:
            sb = "10-25bps"
        elif slip < 50:
            sb = "25-50bps"
        else:
            sb = "50bps+"
        self.drift.returns_by_slippage_bucket[sb].append(ret)

        # Fill type
        fill_type = "full" if outcome.fill_pct >= 99.9 else "partial"
        self.drift.returns_by_partial_vs_full[fill_type].append(ret)

        # Entry time bucket
        try:
            dt = datetime.fromisoformat(outcome.entry_time.replace("Z", "+00:00"))
            hour_min = dt.strftime("%H:%M")
            if hour_min < "09:45":
                tb = "09:30-09:45"
            elif hour_min < "10:00":
                tb = "09:45-10:00"
            elif hour_min < "10:30":
                tb = "10:00-10:30"
            elif hour_min < "11:00":
                tb = "10:30-11:00"
            else:
                tb = "11:00+"
            self.drift.returns_by_entry_time_bucket[tb].append(ret)
        except Exception:
            pass

    # ───────────────────────────────────────────────────
    # Section 11 – Alerts
    # ───────────────────────────────────────────────────

    def check_alerts(self) -> List[Alert]:
        new_alerts: List[Alert] = []
        now_str = datetime.now(MARKET_TZ).isoformat()
        thresholds = self.alert_thresholds

        # Missing prev close rate
        if self.funnel.total_starting_universe > 0:
            rate = self.funnel.skipped_missing_prev_close / self.funnel.total_starting_universe
            if rate > thresholds.missing_prev_close_rate:
                new_alerts.append(Alert(now_str, "WARNING", "data_quality",
                    f"Missing prev close rate {rate:.1%} exceeds threshold {thresholds.missing_prev_close_rate:.1%}"))

        # Low candidate count
        if 0 < self.funnel.final_ranked_candidates < thresholds.candidate_count_low:
            new_alerts.append(Alert(now_str, "WARNING", "funnel",
                f"Only {self.funnel.final_ranked_candidates} candidates found (threshold: {thresholds.candidate_count_low})"))

        # Low fill rate
        if self.dashboard.entries_attempted > 0:
            fill_rate = (self.dashboard.entries_filled + self.dashboard.partial_fills) / self.dashboard.entries_attempted
            if fill_rate < thresholds.fill_rate_low:
                new_alerts.append(Alert(now_str, "WARNING", "execution",
                    f"Fill rate {fill_rate:.1%} below threshold {thresholds.fill_rate_low:.1%}"))

        # High slippage
        entry_agg = self.compute_entry_aggregates()
        if entry_agg.avg_slippage_bps > thresholds.avg_slippage_bps_high:
            new_alerts.append(Alert(now_str, "WARNING", "execution",
                f"Avg entry slippage {entry_agg.avg_slippage_bps:.1f}bps exceeds threshold {thresholds.avg_slippage_bps_high:.1f}bps"))

        # High partial fill rate
        if self.dashboard.entries_attempted > 0:
            partial_rate = self.dashboard.partial_fills / self.dashboard.entries_attempted
            if partial_rate > thresholds.partial_fill_rate_high:
                new_alerts.append(Alert(now_str, "WARNING", "execution",
                    f"Partial fill rate {partial_rate:.1%} exceeds threshold {thresholds.partial_fill_rate_high:.1%}"))

        # High rejection count
        if self.broker_integrity.rejected_orders > thresholds.rejection_count_high:
            new_alerts.append(Alert(now_str, "CRITICAL", "broker",
                f"Rejected orders ({self.broker_integrity.rejected_orders}) exceeds threshold ({thresholds.rejection_count_high})"))

        # Force-flat exits
        if self.dashboard.forced_exits > thresholds.force_flat_count_high:
            new_alerts.append(Alert(now_str, "WARNING", "execution",
                f"Force-flat exits ({self.dashboard.forced_exits}) exceeds threshold ({thresholds.force_flat_count_high})"))

        # Daily drawdown
        if self.risk.max_intraday_drawdown > thresholds.daily_drawdown_pct_high:
            new_alerts.append(Alert(now_str, "CRITICAL", "risk",
                f"Intraday drawdown {self.risk.max_intraday_drawdown:.2%} exceeds threshold {thresholds.daily_drawdown_pct_high:.2%}"))

        # Stale data
        if self.data_integrity.last_good_update_ts:
            try:
                last_update = datetime.fromisoformat(self.data_integrity.last_good_update_ts)
                age = (datetime.now(MARKET_TZ) - last_update).total_seconds()
                if age > thresholds.stale_data_seconds and self.dashboard.market_open_status:
                    new_alerts.append(Alert(now_str, "CRITICAL", "data_quality",
                        f"No good data update for {age:.0f}s (threshold: {thresholds.stale_data_seconds:.0f}s)"))
            except Exception:
                pass

        # Broker/local mismatch
        if self.broker_integrity.position_mismatches > 0:
            new_alerts.append(Alert(now_str, "CRITICAL", "broker",
                f"Position mismatch detected ({self.broker_integrity.position_mismatches} mismatches)"))

        # Idle during market hours
        if self.dashboard.market_open_status:
            idle = time.monotonic() - self._last_activity_ts
            if idle > thresholds.idle_during_market_seconds:
                new_alerts.append(Alert(now_str, "WARNING", "system",
                    f"Bot idle for {idle:.0f}s during market hours (threshold: {thresholds.idle_during_market_seconds:.0f}s)"))

        self.alerts.extend(new_alerts)
        return new_alerts

    # ───────────────────────────────────────────────────
    # Section 6 – Rolling Stats Persistence
    # ───────────────────────────────────────────────────

    def _load_rolling_stats(self) -> Dict[str, Any]:
        if self._rolling_stats_file.exists():
            try:
                with self._rolling_stats_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                logger.warning("Could not load rolling stats, starting fresh")
        return {
            "all_time": self._empty_tally(),
            "daily_history": [],  # list of {date, ...tally fields}
        }

    def _empty_tally(self) -> Dict[str, Any]:
        return asdict(RunningTally())

    def update_rolling_stats(self) -> None:
        today = datetime.now(MARKET_TZ).strftime("%Y-%m-%d")
        trade_agg = self.compute_trade_aggregates()
        entry_agg = self.compute_entry_aggregates()

        # Build today's tally
        today_tally = {
            "period_label": today,
            "cumulative_realized_pnl": self.dashboard.realized_pnl_today,
            "cumulative_unrealized_pnl": self.dashboard.unrealized_pnl_current,
            "cumulative_net_return": (
                (self.dashboard.realized_pnl_today / self.dashboard.account_equity_start * 100)
                if self.dashboard.account_equity_start > 0 else 0.0
            ),
            "cumulative_deployed_dollars": self.risk.max_capital_deployed_today,
            "cumulative_gross_traded_notional": self._daily_gross_notional,
            "cumulative_fees": 0.0,
            "cumulative_slippage_dollars": self._daily_slippage_dollars,
            "cumulative_slippage_bps_avg": entry_agg.avg_slippage_bps,
            "cumulative_entries_attempted": self.dashboard.entries_attempted,
            "cumulative_entries_filled": self.dashboard.entries_filled,
            "cumulative_partial_fills": self.dashboard.partial_fills,
            "cumulative_canceled_entries": self.dashboard.canceled_entries,
            "cumulative_force_flat_exits": self.dashboard.forced_exits,
            "cumulative_win_rate": trade_agg.win_rate,
            "cumulative_expectancy": trade_agg.expectancy_pct,
            "cumulative_profit_factor": trade_agg.profit_factor if trade_agg.profit_factor != float("inf") else 999.0,
            "cumulative_max_drawdown": self.risk.max_intraday_drawdown * 100,
            "cumulative_avg_daily_return": 0.0,
            "cumulative_avg_trade_return": trade_agg.expectancy_pct,
        }

        # Update daily history (replace today if exists, else append)
        history = self._rolling_stats.get("daily_history", [])
        history = [d for d in history if d.get("period_label") != today]
        history.append(today_tally)
        # Keep last 365 days
        if len(history) > 365:
            history = history[-365:]
        self._rolling_stats["daily_history"] = history

        # Update all-time cumulative
        at = self._rolling_stats.get("all_time", self._empty_tally())
        at["cumulative_realized_pnl"] = sum(d.get("cumulative_realized_pnl", 0) for d in history)
        at["cumulative_gross_traded_notional"] = sum(d.get("cumulative_gross_traded_notional", 0) for d in history)
        at["cumulative_slippage_dollars"] = sum(d.get("cumulative_slippage_dollars", 0) for d in history)
        at["cumulative_entries_attempted"] = sum(d.get("cumulative_entries_attempted", 0) for d in history)
        at["cumulative_entries_filled"] = sum(d.get("cumulative_entries_filled", 0) for d in history)
        at["cumulative_partial_fills"] = sum(d.get("cumulative_partial_fills", 0) for d in history)
        at["cumulative_canceled_entries"] = sum(d.get("cumulative_canceled_entries", 0) for d in history)
        at["cumulative_force_flat_exits"] = sum(d.get("cumulative_force_flat_exits", 0) for d in history)

        total_filled = at["cumulative_entries_filled"] + at["cumulative_partial_fills"]
        if total_filled > 0:
            daily_returns = [d.get("cumulative_net_return", 0) for d in history if d.get("cumulative_net_return", 0) != 0]
            if daily_returns:
                at["cumulative_avg_daily_return"] = statistics.mean(daily_returns)

        # Compute rolling windows
        self._rolling_stats["all_time"] = at
        self._rolling_stats["rolling_20d"] = self._compute_rolling_window(history, 20)
        self._rolling_stats["rolling_50d"] = self._compute_rolling_window(history, 50)
        self._rolling_stats["weekly"] = self._compute_rolling_window(history, 5)
        self._rolling_stats["mtd"] = self._compute_mtd(history)

        # Persist
        try:
            _atomic_write_json(self._rolling_stats_file, self._rolling_stats)
        except Exception:
            logger.exception("Failed to save rolling stats")

    def _compute_rolling_window(self, history: List[Dict], days: int) -> Dict[str, Any]:
        window = history[-days:] if len(history) >= days else history
        tally = self._empty_tally()
        tally["period_label"] = f"rolling_{days}d"
        if not window:
            return tally
        for key in [
            "cumulative_realized_pnl", "cumulative_gross_traded_notional",
            "cumulative_slippage_dollars", "cumulative_entries_attempted",
            "cumulative_entries_filled", "cumulative_partial_fills",
            "cumulative_canceled_entries", "cumulative_force_flat_exits",
        ]:
            tally[key] = sum(d.get(key, 0) for d in window)
        rets = [d.get("cumulative_net_return", 0) for d in window]
        if rets:
            tally["cumulative_avg_daily_return"] = statistics.mean(rets)
        win_rates = [d.get("cumulative_win_rate", 0) for d in window if d.get("cumulative_entries_filled", 0) > 0]
        if win_rates:
            tally["cumulative_win_rate"] = statistics.mean(win_rates)
        dds = [d.get("cumulative_max_drawdown", 0) for d in window]
        if dds:
            tally["cumulative_max_drawdown"] = max(dds)
        return tally

    def _compute_mtd(self, history: List[Dict]) -> Dict[str, Any]:
        today = datetime.now(MARKET_TZ)
        first_of_month = today.replace(day=1).strftime("%Y-%m-%d")
        mtd_days = [d for d in history if d.get("period_label", "") >= first_of_month]
        return self._compute_rolling_window(mtd_days, len(mtd_days)) if mtd_days else self._empty_tally()

    # ───────────────────────────────────────────────────
    # Snapshot for serialization
    # ───────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        return {
            "dashboard": asdict(self.dashboard),
            "funnel": asdict(self.funnel),
            "entry_aggregates": asdict(self.compute_entry_aggregates()),
            "exit_aggregates": asdict(self.compute_exit_aggregates()),
            "trade_aggregates": asdict(self.compute_trade_aggregates()),
            "risk_exposure": asdict(self.risk),
            "data_integrity": asdict(self.data_integrity),
            "broker_integrity": asdict(self.broker_integrity),
            "alerts": [asdict(a) for a in self.alerts],
            "entry_orders_count": len(self.entry_orders),
            "exit_orders_count": len(self.exit_orders),
            "trade_outcomes_count": len(self.trade_outcomes),
        }

    def get_rolling_stats(self) -> Dict[str, Any]:
        return dict(self._rolling_stats)


# ═══════════════════════════════════════════════════════════════
# Global Singleton
# ═══════════════════════════════════════════════════════════════

_session_monitor: Optional[SessionMonitor] = None


def get_session_monitor() -> SessionMonitor:
    global _session_monitor
    if _session_monitor is None:
        _session_monitor = SessionMonitor()
    return _session_monitor


def reset_session_monitor() -> SessionMonitor:
    global _session_monitor
    _session_monitor = SessionMonitor()
    return _session_monitor
