"""
Report Generation for the Monitoring System.

Layer 1: Live console summary (compact, during market hours)
Layer 2: End-of-day report (full daily summary)
Layer 3: Trade ledger CSV (one row per order event)
Layer 4: Rolling stats JSON (persisted by SessionMonitor)
"""

from __future__ import annotations

import csv
import io
import logging
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo

from .monitoring import (
    SessionMonitor,
    get_session_monitor,
)
from .state_manager import _atomic_write_json

logger = logging.getLogger("monitor_reports")

MARKET_TZ = ZoneInfo("America/New_York")


# ═══════════════════════════════════════════════════════════════
# Layer 1 – Live Console Summary
# ═══════════════════════════════════════════════════════════════

def live_console_summary(monitor: Optional[SessionMonitor] = None) -> str:
    """
    Compact, human-readable summary for display during market hours.
    Designed to fit in ~20 lines.
    """
    m = monitor or get_session_monitor()
    d = m.dashboard
    r = m.risk
    entry_agg = m.compute_entry_aggregates()
    trade_agg = m.compute_trade_aggregates()

    # Compute fill rate
    fill_rate = 0.0
    if d.entries_attempted > 0:
        fill_rate = (d.entries_filled + d.partial_fills) / d.entries_attempted * 100

    lines = [
        f"{'═' * 60}",
        f"  LIVE DASHBOARD  {d.date}  {datetime.now(MARKET_TZ).strftime('%H:%M:%S')}",
        f"{'═' * 60}",
        f"  Equity: ${d.account_equity_current:,.2f}  (start: ${d.account_equity_start:,.2f})",
        f"  Cash:   ${d.cash_current:,.2f}  |  Deployed: {r.pct_deployed:.1f}%",
        f"  P&L:    realized ${d.realized_pnl_today:+,.2f}  |  unrealized ${d.unrealized_pnl_current:+,.2f}  |  total ${d.total_pnl_today:+,.2f}",
        f"{'─' * 60}",
        f"  Scanned: {d.symbols_scanned}  →  Candidates: {d.candidates_found}  →  Sized: {m.funnel.selected_for_sizing}",
        f"  Entries: {d.entries_attempted} attempted  |  {d.entries_filled} filled  |  {d.partial_fills} partial  |  {d.canceled_entries} canceled",
        f"  Fill rate: {fill_rate:.0f}%  |  Avg slippage: {entry_agg.avg_slippage_bps:.1f}bps",
        f"{'─' * 60}",
        f"  Positions: {d.current_open_positions} open  |  {d.trades_closed} closed today  |  {d.forced_exits} force-flat",
        f"  Win rate: {trade_agg.win_rate:.0f}%  |  Expectancy: {trade_agg.expectancy_pct:+.2f}%  |  PF: {_fmt_pf(trade_agg.profit_factor)}",
        f"  Drawdown: {r.intraday_drawdown_pct:.2%} (max today: {r.max_intraday_drawdown:.2%})",
        f"{'─' * 60}",
        f"  Data errors: API fail={m.data_integrity.api_failure_count}  timeout={m.data_integrity.api_timeout_count}  stale={m.data_integrity.symbols_stale_quotes}",
        f"  Broker: reject={m.broker_integrity.rejected_orders}  mismatch={m.broker_integrity.position_mismatches}  orphan={m.broker_integrity.orphan_fills}",
    ]

    # Active alerts
    recent_alerts = m.alerts[-3:] if m.alerts else []
    if recent_alerts:
        lines.append(f"{'─' * 60}")
        lines.append("  ALERTS:")
        for a in recent_alerts:
            lines.append(f"    [{a.severity}] {a.category}: {a.message}")

    lines.append(f"{'═' * 60}")
    return "\n".join(lines)


def print_live_summary(monitor: Optional[SessionMonitor] = None) -> None:
    summary = live_console_summary(monitor)
    for line in summary.split("\n"):
        logger.info(line)


# ═══════════════════════════════════════════════════════════════
# Layer 2 – End-of-Day Report
# ═══════════════════════════════════════════════════════════════

def generate_eod_report(monitor: Optional[SessionMonitor] = None) -> str:
    """Generate comprehensive end-of-day report text."""
    m = monitor or get_session_monitor()
    d = m.dashboard
    f = m.funnel
    r = m.risk
    di = m.data_integrity
    bi = m.broker_integrity
    entry_agg = m.compute_entry_aggregates()
    exit_agg = m.compute_exit_aggregates()
    trade_agg = m.compute_trade_aggregates()

    lines = []

    # ── Header ──
    lines.append("=" * 80)
    lines.append(f"  END-OF-DAY REPORT  —  {d.date}")
    lines.append(f"  Generated: {datetime.now(MARKET_TZ).strftime('%Y-%m-%d %H:%M:%S ET')}")
    lines.append("=" * 80)

    # ── Section 1: Daily Dashboard ──
    lines.append("\n┌─ 1. SESSION SUMMARY ─────────────────────────────────────────┐")
    lines.append(f"  Date:           {d.date}")
    lines.append(f"  Bot start:      {d.bot_start_time}")
    lines.append(f"  Bot stop:       {d.bot_stop_time}")
    lines.append(f"  Strategies:     {', '.join(d.strategy_modules_run)}")
    lines.append(f"  Market regime:  {d.market_regime or 'N/A'}")
    lines.append(f"  Market open:    {d.market_open_status}")
    lines.append(f"  Equity start:   ${d.account_equity_start:,.2f}")
    lines.append(f"  Equity end:     ${d.account_equity_current:,.2f}")
    lines.append(f"  Cash start:     ${d.cash_start:,.2f}")
    lines.append(f"  Cash end:       ${d.cash_current:,.2f}")
    lines.append(f"  Realized P&L:   ${d.realized_pnl_today:+,.2f}")
    lines.append(f"  Unrealized P&L: ${d.unrealized_pnl_current:+,.2f}")
    lines.append(f"  Total P&L:      ${d.total_pnl_today:+,.2f}")
    lines.append(f"  Trades opened:  {d.trades_opened}")
    lines.append(f"  Trades closed:  {d.trades_closed}")
    lines.append(f"  Entries attempted: {d.entries_attempted}")
    lines.append(f"  Entries filled:    {d.entries_filled}")
    lines.append(f"  Partial fills:     {d.partial_fills}")
    lines.append(f"  Canceled entries:  {d.canceled_entries}")
    lines.append(f"  Forced exits:      {d.forced_exits}")
    lines.append(f"  Open positions:    {d.current_open_positions}")
    lines.append("└─────────────────────────────────────────────────────────────┘")

    # ── Section 2: Funnel ──
    lines.append("\n┌─ 2. CANDIDATE FUNNEL ────────────────────────────────────────┐")
    lines.append(f"  Starting universe:      {f.total_starting_universe}")
    lines.append(f"  With valid data:        {f.symbols_with_valid_data}")
    lines.append(f"  Pass price filter:      {f.passing_price_filter}")
    lines.append(f"  Pass gap filter:        {f.passing_gap_filter}")
    lines.append(f"  Pass liquidity filter:  {f.passing_liquidity_filter}")
    lines.append(f"  Pass relvol filter:     {f.passing_relvol_filter}")
    lines.append(f"  Pass ALL filters:       {f.passing_all_filters}")
    lines.append(f"  Final ranked:           {f.final_ranked_candidates}")
    lines.append(f"  Selected for sizing:    {f.selected_for_sizing}")
    lines.append(f"  ─── Skipped reasons ───")
    lines.append(f"  Capital constraints:    {f.skipped_capital_constraints}")
    lines.append(f"  Volume cap:             {f.skipped_volume_cap}")
    lines.append(f"  Day lockout:            {f.skipped_day_lockout}")
    lines.append(f"  Missing prev close:     {f.skipped_missing_prev_close}")
    lines.append(f"  Stale minute bars:      {f.skipped_stale_minute_bars}")
    if f.drop_reasons:
        lines.append(f"  ─── Drop breakdown ───")
        for reason, count in sorted(f.drop_reasons.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {reason:<30s} {count:>6d}")
    lines.append("└─────────────────────────────────────────────────────────────┘")

    # ── Section 3: Entry Execution Quality ──
    lines.append("\n┌─ 3. ENTRY EXECUTION QUALITY ─────────────────────────────────┐")
    lines.append(f"  Avg slippage:       {entry_agg.avg_slippage_bps:+.1f} bps")
    lines.append(f"  Median slippage:    {entry_agg.median_slippage_bps:+.1f} bps")
    lines.append(f"  95th pct slippage:  {entry_agg.p95_slippage_bps:+.1f} bps")
    lines.append(f"  Worst slippage:     {entry_agg.worst_slippage_bps:+.1f} bps")
    lines.append(f"  Avg fill %:         {entry_agg.avg_fill_pct:.1f}%")
    lines.append(f"  Avg time to 1st fill: {entry_agg.avg_time_to_first_fill:.2f}s")
    lines.append(f"  Avg time to full fill:{entry_agg.avg_time_to_full_fill:.2f}s")
    lines.append(f"  % fully filled:     {entry_agg.pct_fully_filled:.1f}%")
    lines.append(f"  % partially filled: {entry_agg.pct_partially_filled:.1f}%")
    lines.append(f"  % canceled unfilled:{entry_agg.pct_canceled_unfilled:.1f}%")
    lines.append(f"  % canceled by 60s:  {entry_agg.pct_canceled_60s:.1f}%")
    lines.append(f"  % canceled by +1%:  {entry_agg.pct_canceled_1pct:.1f}%")
    if m.entry_orders:
        lines.append(f"  ─── Per-order detail (last 10) ───")
        lines.append(f"  {'Symbol':<6} {'Status':<10} {'Intended':>9} {'Filled':>9} {'Slip bps':>9} {'Fill%':>6} {'TIF':<4}")
        for o in m.entry_orders[-10:]:
            lines.append(
                f"  {o.symbol:<6} {o.status:<10} "
                f"${o.intended_notional:>8,.0f} ${o.filled_notional:>8,.0f} "
                f"{o.slippage_bps:>+8.1f} {o.fill_pct:>5.0f}% {o.tif:<4}"
            )
    lines.append("└─────────────────────────────────────────────────────────────┘")

    # ── Section 4: Exit Execution Quality ──
    lines.append("\n┌─ 4. EXIT EXECUTION QUALITY ──────────────────────────────────┐")
    lines.append(f"  Avg slippage:       {exit_agg.avg_slippage_bps:+.1f} bps")
    lines.append(f"  Median slippage:    {exit_agg.median_slippage_bps:+.1f} bps")
    lines.append(f"  95th pct slippage:  {exit_agg.p95_slippage_bps:+.1f} bps")
    lines.append(f"  % force-flat:       {exit_agg.pct_force_flat:.1f}%")
    lines.append(f"  % partial before done: {exit_agg.pct_partial_before_completion:.1f}%")
    lines.append(f"  Avg signal→exit:    {exit_agg.avg_signal_to_exit_delay_s:.2f}s")
    if m.exit_orders:
        lines.append(f"  ─── Per-exit detail (last 10) ───")
        lines.append(f"  {'Symbol':<6} {'Reason':<18} {'Intended':>9} {'Filled':>9} {'Slip bps':>9} {'FF':>3}")
        for o in m.exit_orders[-10:]:
            ff_flag = "YES" if o.force_flat else ""
            lines.append(
                f"  {o.symbol:<6} {o.actual_exit_reason:<18} "
                f"${o.intended_exit_price:>8.2f} ${o.avg_exit_price:>8.2f} "
                f"{o.exit_slippage_bps:>+8.1f} {ff_flag:>3}"
            )
    lines.append("└─────────────────────────────────────────────────────────────┘")

    # ── Section 5: Trade Outcome Stats ──
    lines.append("\n┌─ 5. TRADE OUTCOME STATS ─────────────────────────────────────┐")
    lines.append(f"  Win rate:           {trade_agg.win_rate:.1f}%")
    lines.append(f"  Loss rate:          {trade_agg.loss_rate:.1f}%")
    lines.append(f"  Avg win %:          {trade_agg.avg_win_pct:+.2f}%")
    lines.append(f"  Avg loss %:         {trade_agg.avg_loss_pct:+.2f}%")
    lines.append(f"  Expectancy %:       {trade_agg.expectancy_pct:+.2f}%")
    lines.append(f"  Profit factor:      {_fmt_pf(trade_agg.profit_factor)}")
    lines.append(f"  Avg hold time:      {_fmt_duration(trade_agg.avg_holding_time_s)}")
    lines.append(f"  Median hold time:   {_fmt_duration(trade_agg.median_holding_time_s)}")
    lines.append(f"  Avg MFE:            {trade_agg.avg_mfe:.2f}%")
    lines.append(f"  Avg MAE:            {trade_agg.avg_mae:.2f}%")
    lines.append(f"  Avg return (full):  {trade_agg.avg_return_full_fills:+.2f}%")
    lines.append(f"  Avg return (partial):{trade_agg.avg_return_partial_fills:+.2f}%")
    if m.trade_outcomes:
        lines.append(f"  ─── Per-trade detail (last 10) ───")
        lines.append(f"  {'Symbol':<6} {'P&L $':>9} {'Ret %':>7} {'Hold':>8} {'Exit Reason':<18} {'Fill%':>6}")
        for t in m.trade_outcomes[-10:]:
            lines.append(
                f"  {t.symbol:<6} ${t.dollars_won_lost:>+8.2f} "
                f"{t.gross_return_pct:>+6.2f}% {_fmt_duration(t.holding_time_s):>8} "
                f"{t.exit_reason:<18} {t.fill_pct:>5.0f}%"
            )
    lines.append("└─────────────────────────────────────────────────────────────┘")

    # ── Section 6: Running Tallies ──
    rolling = m.get_rolling_stats()
    lines.append("\n┌─ 6. RUNNING TALLIES ─────────────────────────────────────────┐")
    lines.append(f"  {'Period':<14} {'P&L':>10} {'Entries':>8} {'Filled':>8} {'WinRate':>8} {'DD':>8} {'Slip$':>8}")
    for period_key in ["all_time", "mtd", "weekly", "rolling_20d", "rolling_50d"]:
        t = rolling.get(period_key, {})
        if not t:
            continue
        label = period_key.replace("_", " ").title()
        lines.append(
            f"  {label:<14} "
            f"${t.get('cumulative_realized_pnl', 0):>+9,.0f} "
            f"{t.get('cumulative_entries_attempted', 0):>8} "
            f"{t.get('cumulative_entries_filled', 0):>8} "
            f"{t.get('cumulative_win_rate', 0):>7.1f}% "
            f"{t.get('cumulative_max_drawdown', 0):>7.2f}% "
            f"${t.get('cumulative_slippage_dollars', 0):>7,.0f}"
        )
    lines.append("└─────────────────────────────────────────────────────────────┘")

    # ── Section 7: Risk & Exposure ──
    lines.append("\n┌─ 7. RISK & EXPOSURE ─────────────────────────────────────────┐")
    lines.append(f"  Cash:               ${r.current_cash:,.2f}")
    lines.append(f"  Equity:             ${r.current_equity:,.2f}")
    lines.append(f"  Deployed:           ${r.total_deployed:,.2f} ({r.pct_deployed:.1f}%)")
    lines.append(f"  Open positions:     {r.open_positions_count}")
    lines.append(f"  Largest pos %:      {r.largest_position_pct:.1f}%")
    lines.append(f"  Smallest pos %:     {r.smallest_position_pct:.1f}%")
    lines.append(f"  Avg pos %:          {r.avg_position_pct:.1f}%")
    lines.append(f"  Top 3 concentration:{r.concentration_top3:.1f}%")
    lines.append(f"  Top 5 concentration:{r.concentration_top5:.1f}%")
    lines.append(f"  Notional at stops:  ${r.total_notional_at_stops:,.2f}")
    lines.append(f"  Est loss to stops:  ${r.estimated_loss_to_stops:,.2f}")
    lines.append(f"  Intraday DD:        {r.intraday_drawdown_pct:.2%}")
    lines.append(f"  Max intraday DD:    {r.max_intraday_drawdown:.2%}")
    lines.append(f"  Pending notional:   ${r.pending_order_notional:,.2f}")
    lines.append(f"  ─── Daily risk summary ───")
    lines.append(f"  Max deployed today: ${r.max_capital_deployed_today:,.2f}")
    lines.append(f"  Max positions:      {r.max_simultaneous_positions}")
    lines.append(f"  Max single exposure:${r.largest_single_exposure:,.2f}")
    lines.append("└─────────────────────────────────────────────────────────────┘")

    # ── Section 8: Data Integrity ──
    lines.append("\n┌─ 8. DATA INTEGRITY ──────────────────────────────────────────┐")
    lines.append(f"  Missing prev close: {di.symbols_missing_prev_close}")
    lines.append(f"  Missing open:       {di.symbols_missing_open}")
    lines.append(f"  Stale quotes:       {di.symbols_stale_quotes}")
    lines.append(f"  Missing min bars:   {di.symbols_missing_minute_bars}")
    lines.append(f"  Zero volume:        {di.symbols_zero_volume}")
    lines.append(f"  API requests:       {di.api_request_count}")
    lines.append(f"  API failures:       {di.api_failure_count}")
    lines.append(f"  API timeouts:       {di.api_timeout_count}")
    lines.append(f"  Rate limit hits:    {di.rate_limit_hits}")
    lines.append(f"  Broker reconnects:  {di.broker_reconnect_count}")
    lines.append(f"  Quote latency:      {di.quote_refresh_latency_ms:.0f}ms")
    lines.append(f"  Bar latency:        {di.bar_refresh_latency_ms:.0f}ms")
    lines.append(f"  Last good update:   {di.last_good_update_ts or 'N/A'}")
    lines.append("└─────────────────────────────────────────────────────────────┘")

    # ── Section 9: Broker Integrity ──
    lines.append("\n┌─ 9. BROKER / ORDER INTEGRITY ────────────────────────────────┐")
    lines.append(f"  Rejected orders:    {bi.rejected_orders}")
    lines.append(f"  Canceled orders:    {bi.canceled_orders}")
    lines.append(f"  Expired orders:     {bi.expired_orders}")
    lines.append(f"  Unknown statuses:   {bi.unknown_order_statuses}")
    lines.append(f"  Orphan fills:       {bi.orphan_fills}")
    lines.append(f"  Pending entries:    {bi.pending_entries}")
    lines.append(f"  Pending exits:      {bi.pending_exits}")
    lines.append(f"  Reconciliation:     {bi.reconciliation_corrections}")
    lines.append(f"  Position mismatches:{bi.position_mismatches}")
    lines.append(f"  Cash mismatches:    {bi.cash_mismatches}")
    lines.append(f"  Bot recoveries:     {bi.bot_recoveries}")
    lines.append("└─────────────────────────────────────────────────────────────┘")

    # ── Section 10: Drift Diagnostics ──
    drift = m.drift
    lines.append("\n┌─ 10. STRATEGY DRIFT DIAGNOSTICS ─────────────────────────────┐")
    _append_drift_table(lines, "Returns by Day of Week", drift.returns_by_day_of_week)
    _append_drift_table(lines, "Returns by Gap Bucket", drift.returns_by_gap_bucket)
    _append_drift_table(lines, "Returns by Slippage Bucket", drift.returns_by_slippage_bucket)
    _append_drift_table(lines, "Returns by Fill Type", drift.returns_by_partial_vs_full)
    _append_drift_table(lines, "Returns by Entry Time", drift.returns_by_entry_time_bucket)
    lines.append("└─────────────────────────────────────────────────────────────┘")

    # ── Section 11: Alerts ──
    lines.append("\n┌─ 11. ALERTS ─────────────────────────────────────────────────┐")
    if m.alerts:
        for a in m.alerts:
            ts = a.timestamp[:19] if len(a.timestamp) > 19 else a.timestamp
            lines.append(f"  [{a.severity:8s}] {ts} {a.category}: {a.message}")
    else:
        lines.append("  No alerts triggered this session.")
    lines.append("└─────────────────────────────────────────────────────────────┘")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def save_eod_report(monitor: Optional[SessionMonitor] = None) -> Path:
    """Generate and save the EOD report to state/reports/."""
    m = monitor or get_session_monitor()
    report_text = generate_eod_report(m)
    today = datetime.now(MARKET_TZ).strftime("%Y-%m-%d")
    report_path = m.reports_dir / f"eod_report_{today}.txt"
    try:
        report_path.write_text(report_text, encoding="utf-8")
        logger.info("EOD report saved to %s", report_path)
    except Exception:
        logger.exception("Failed to save EOD report")

    # Also save full snapshot as JSON
    snapshot_path = m.reports_dir / f"eod_snapshot_{today}.json"
    try:
        _atomic_write_json(snapshot_path, m.snapshot())
        logger.info("EOD snapshot saved to %s", snapshot_path)
    except Exception:
        logger.exception("Failed to save EOD snapshot")

    return report_path


# ═══════════════════════════════════════════════════════════════
# Layer 3 – Trade Ledger CSV
# ═══════════════════════════════════════════════════════════════

_CSV_ENTRY_HEADERS = [
    "date", "time", "type", "symbol", "status",
    "intended_qty", "filled_qty", "intended_price", "limit_price", "avg_fill_price",
    "slippage_dollars", "slippage_bps", "fill_pct",
    "time_to_first_fill_s", "time_to_full_fill_s",
    "cancel_reason", "is_fractional", "tif",
]

_CSV_EXIT_HEADERS = [
    "date", "time", "type", "symbol", "status",
    "intended_price", "avg_exit_price", "exit_slippage_bps",
    "time_to_fill_s", "exit_reason", "force_flat", "force_flat_reason",
    "partial_exit",
]

_CSV_TRADE_HEADERS = [
    "date", "symbol", "entry_time", "exit_time",
    "holding_time_s", "gross_return_pct", "net_return_pct",
    "dollars_won_lost", "max_favorable_excursion", "max_adverse_excursion",
    "exit_reason", "gap_at_entry", "first_5min_volume",
    "fill_pct", "entry_slippage_bps",
]


def append_trade_ledger_csv(monitor: Optional[SessionMonitor] = None) -> None:
    """Append today's order events and trade outcomes to CSV files."""
    m = monitor or get_session_monitor()
    today = datetime.now(MARKET_TZ).strftime("%Y-%m-%d")

    # Entry orders CSV
    _append_entry_csv(m, today)

    # Exit orders CSV
    _append_exit_csv(m, today)

    # Trade outcomes CSV
    _append_trade_csv(m, today)


def _append_entry_csv(m: SessionMonitor, today: str) -> None:
    csv_path = m.reports_dir / "entry_ledger.csv"
    write_header = not csv_path.exists()
    try:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(_CSV_ENTRY_HEADERS)
            for o in m.entry_orders:
                ts = o.submit_ts or ""
                time_part = ts[11:19] if len(ts) > 18 else ts
                writer.writerow([
                    today, time_part, "ENTRY", o.symbol, o.status,
                    f"{o.intended_shares:.4f}", f"{o.filled_shares:.4f}",
                    f"{o.intended_price:.4f}", f"{o.submitted_limit:.4f}", f"{o.avg_fill_price:.4f}",
                    f"{o.slippage_dollars:.4f}", f"{o.slippage_bps:.2f}", f"{o.fill_pct:.1f}",
                    f"{o.time_to_first_fill_s:.3f}", f"{o.time_to_full_fill_s:.3f}",
                    o.cancel_reason, o.is_fractional, o.tif,
                ])
    except Exception:
        logger.exception("Failed to append entry ledger CSV")


def _append_exit_csv(m: SessionMonitor, today: str) -> None:
    csv_path = m.reports_dir / "exit_ledger.csv"
    write_header = not csv_path.exists()
    try:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(_CSV_EXIT_HEADERS)
            for o in m.exit_orders:
                ts = o.exit_submit_ts or ""
                time_part = ts[11:19] if len(ts) > 18 else ts
                writer.writerow([
                    today, time_part, "EXIT", o.symbol, "",
                    f"{o.intended_exit_price:.4f}", f"{o.avg_exit_price:.4f}",
                    f"{o.exit_slippage_bps:.2f}",
                    f"{o.time_to_fill_s:.3f}", o.actual_exit_reason,
                    o.force_flat, o.force_flat_reason, o.partial_exit,
                ])
    except Exception:
        logger.exception("Failed to append exit ledger CSV")


def _append_trade_csv(m: SessionMonitor, today: str) -> None:
    csv_path = m.reports_dir / "trade_ledger.csv"
    write_header = not csv_path.exists()
    try:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(_CSV_TRADE_HEADERS)
            for t in m.trade_outcomes:
                writer.writerow([
                    today, t.symbol, t.entry_time, t.exit_time,
                    f"{t.holding_time_s:.1f}", f"{t.gross_return_pct:.4f}",
                    f"{t.net_return_pct:.4f}", f"{t.dollars_won_lost:.2f}",
                    f"{t.max_favorable_excursion:.4f}", f"{t.max_adverse_excursion:.4f}",
                    t.exit_reason, f"{t.gap_at_entry:.4f}", f"{t.first_5min_volume:.0f}",
                    f"{t.fill_pct:.1f}", f"{t.entry_slippage_bps:.2f}",
                ])
    except Exception:
        logger.exception("Failed to append trade ledger CSV")


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _fmt_pf(pf: float) -> str:
    if pf == float("inf") or pf > 999:
        return "∞"
    return f"{pf:.2f}"


def _fmt_duration(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _append_drift_table(lines: List[str], title: str, data: Dict[str, List[float]]) -> None:
    if not data:
        lines.append(f"  {title}: No data")
        return
    lines.append(f"  ─── {title} ───")
    lines.append(f"  {'Bucket':<18} {'N':>5} {'Avg%':>7} {'Med%':>7} {'WinR':>6}")
    for bucket, returns in sorted(data.items()):
        if not returns:
            continue
        n = len(returns)
        avg = statistics.mean(returns)
        med = statistics.median(returns)
        wins = sum(1 for r in returns if r > 0)
        wr = wins / n * 100 if n else 0
        lines.append(f"  {bucket:<18} {n:>5} {avg:>+6.2f}% {med:>+6.2f}% {wr:>5.0f}%")
