"""Overnight Momentum Bot — Main Orchestrator

Daily Schedule (ET) — bot starts at 9:00 AM:

MORNING (T+1 exits — positions from yesterday's 3:50 PM entries):
  09:00  Start, detect overnight positions from broker
  09:30  Market open — check hard stops (entry_price × 0.95)
  09:35  First checkpoint — check 6% drop-stop from open high
  11:00  Exit ALL remaining positions at market price
  11:05  Post-exit failsafe verification

AFTERNOON (T-1 entries — new positions for tomorrow's exits):
  15:30  Build universe (Massive + Alpaca asset filter + daily bars + ADV)
  15:48  Fetch 9:30-3:50 minute bars → build & score candidates (350 model)
  15:50  Select positions (account-tier), size, EXECUTE ENTRIES (market)
  16:00  Confirm positions held overnight, save state, done
"""
import logging
import os
import sys
import time
from datetime import datetime, time as dt_time, timedelta, date
from typing import List, Optional, Dict, Any, Tuple

from bot import config
from bot.massive_client import MassiveClient
from bot.market_data import AlpacaDataClient
from bot.momentum_scorer import (
    MomentumCandidate,
    SelectionConfig,
    get_selection_config,
    build_signal_candidates_350,
    compute_raw_metrics_350,
    normalize_and_score_350,
    assign_buckets,
    select_positions,
)
from bot.position_manager_overnight import PositionManager, Position
from bot.rate_limiter import get_api_call_count
from bot.state_manager import StateManager
from bot.universe_builder import (
    build_universe,
    filter_minute_data_quality,
    filter_execution_ready,
    save_universe_audit,
    save_candidates_audit,
    save_run_health,
    save_execution_audit,
    UniverseDiagnostics,
    ExecutionDiagnostics,
)

# Setup logging
os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class OvernightMomentumBot:
    """Main bot orchestrator for overnight momentum strategy"""

    def __init__(self):
        self.massive = MassiveClient()
        self.alpaca = AlpacaDataClient()
        self.position_mgr = PositionManager()
        self.state_mgr = StateManager()

        # Universe & candidates
        self.universe: List[str] = []
        self.scored_candidates: List[MomentumCandidate] = []
        self._universe_diag: Optional[UniverseDiagnostics] = None

        # Stage flags
        self.morning_exits_done = False   # All overnight positions exited
        self.data_collected = False       # Universe + daily bars ready
        self.scoring_done = False         # 3:48 PM scoring complete
        self.entries_done = False         # 3:50 PM entries executed

        # Morning stop tracking
        self.hard_stops_checked = False
        self.drop_stops_checked = False
        self.final_exit_done = False

        # Open prices captured at market open (for drop-stop calculation)
        self.open_prices: Dict[str, float] = {}

        # Failsafe
        self.post_exit_failsafe_done = False

        # Retry counters
        self.universe_retry_count = 0

        # Data collection results (stored between steps)
        self._minute_bars: Dict[str, List[dict]] = {}
        self._daily_bars: Dict[str, List[dict]] = {}
        self._etf_returns: Dict[str, float] = {}
        self._adv_cache: Dict[str, Tuple[float, float]] = {}
        self._atr_cache: Dict[str, float] = {}
        self._exec_stats: Dict[str, Any] = {}
        self._exec_diag: Optional[ExecutionDiagnostics] = None

    def run(self):
        """Main bot loop - runs from 9:00 AM until after market close"""
        logger.info("=" * 60)
        logger.info("Overnight Momentum Bot Starting")
        logger.info("=" * 60)

        # Load any saved state and detect mode
        self._load_state()

        # Check if we have overnight positions to manage
        broker_positions = self.position_mgr.get_broker_positions()
        if broker_positions is None:
            logger.error("Cannot reach broker API at startup — will retry in main loop")
        elif broker_positions:
            logger.info(f"Detected {len(broker_positions)} overnight positions — morning exit mode")
            for pos in broker_positions:
                logger.info(f"  Overnight: {pos.get('symbol')} qty={pos.get('qty')} avg_entry={pos.get('avg_entry_price')}")
            # Reconcile local state with broker
            self.position_mgr.reconcile_local_positions_from_broker()
            self._save_state()
        else:
            logger.info("No overnight positions — skipping morning exits")
            self.morning_exits_done = True

        # If starting after 11:00 AM with positions, flatten immediately
        current_time = datetime.now().time()
        if current_time >= dt_time(11, 5) and self.position_mgr.get_position_count() > 0:
            logger.warning("Started after 11:05 AM with positions — flattening immediately")
            self._run_failsafe_flatten("late-start flatten")
            self.morning_exits_done = True

        # If starting after 4:00 PM, nothing to do
        if current_time >= dt_time(16, 0):
            logger.error("Started after market close — nothing to do")
            return

        # Main event loop
        while True:
            now = datetime.now()
            current_time = now.time()

            # ════════════════════════════════════════════
            # MORNING: Manage overnight position exits
            # ════════════════════════════════════════════

            if not self.morning_exits_done:
                has_positions = self.position_mgr.get_position_count() > 0

                if not has_positions:
                    # Verify with broker
                    bc = self.position_mgr.broker_position_count()
                    if bc == 0:
                        logger.info("Morning exits complete — no positions remaining")
                        self.morning_exits_done = True
                    elif bc > 0:
                        # Broker has positions we don't know about locally
                        logger.warning(f"Local empty but broker has {bc} positions — reconciling")
                        self.position_mgr.reconcile_local_positions_from_broker()
                        self._save_state()

                if has_positions and not self.morning_exits_done:
                    # 9:30 AM — Hard stop check at market open
                    if not self.hard_stops_checked and current_time >= dt_time(9, 30):
                        self._check_hard_stops()
                        self.hard_stops_checked = True
                        self._save_state()

                    # 9:35 AM — Drop-stop check
                    if not self.drop_stops_checked and current_time >= dt_time(9, 35):
                        self._check_drop_stops()
                        self.drop_stops_checked = True
                        self._save_state()

                    # 11:00 AM — Exit ALL remaining positions
                    if not self.final_exit_done and current_time >= dt_time(11, 0):
                        self._exit_all_positions("11:00 AM scheduled exit")
                        self.final_exit_done = True
                        self.morning_exits_done = True
                        self._save_state()

                    # 11:05 AM — Post-exit failsafe
                    if not self.post_exit_failsafe_done and current_time >= dt_time(11, 5):
                        bc = self.position_mgr.broker_position_count()
                        if bc > 0:
                            logger.warning(f"Post-exit failsafe: broker still has {bc} positions")
                            self._run_failsafe_flatten("11:05 AM post-exit failsafe")
                        elif bc == 0:
                            logger.info("Post-exit failsafe: broker confirmed flat")
                        self.post_exit_failsafe_done = True
                        self.morning_exits_done = True
                        self._save_state()

            # ════════════════════════════════════════════
            # AFTERNOON: Score universe and enter new positions
            # ════════════════════════════════════════════

            # 3:30 PM — Data collection
            if not self.data_collected and current_time >= dt_time(15, 30):
                if current_time < dt_time(15, 45):
                    self._step_collect_data()
                else:
                    logger.warning("Past 3:45 PM without data collection — attempting now")
                    self._step_collect_data()

            # 3:48 PM — Score and rank (requires data collection)
            if self.data_collected and not self.scoring_done and current_time >= dt_time(15, 48):
                self._step_score_and_rank()

            # 3:50 PM — Execute entries (requires scoring)
            if self.scoring_done and not self.entries_done and current_time >= dt_time(15, 50):
                self._step_execute_entries()

            # ════════════════════════════════════════════
            # Day completion check
            # ════════════════════════════════════════════

            if current_time >= dt_time(16, 0):
                if self.entries_done:
                    logger.info("Market closed — positions held overnight. Day complete.")
                    self._save_end_of_day_reports()
                    self._save_state()
                    break
                elif self.position_mgr.get_position_count() > 0:
                    logger.info("Market closed with positions — holding overnight as intended.")
                    self._save_end_of_day_reports()
                    self._save_state()
                    break
                else:
                    logger.info("Market closed — no entries made today.")
                    self._finalize_day()
                    break

            time.sleep(1)

    # ════════════════════════════════════════════════════════════
    # MORNING EXIT METHODS
    # ════════════════════════════════════════════════════════════

    def _check_hard_stops(self):
        """9:30 AM: Check if any position opened below hard stop level (entry × 0.95)."""
        logger.info("HARD STOP CHECK: checking opening prices against entry stops")

        positions = list(self.position_mgr.positions.items())
        if not positions:
            return

        # Get current prices (opening prints)
        symbols = [s for s, _ in positions]
        snapshots = self.alpaca.get_snapshots(symbols)

        exits_triggered = []
        for symbol, position in positions:
            snap = snapshots.get(symbol, {})
            open_price = snap.get("open") or snap.get("last_price")
            if not open_price:
                logger.warning(f"No opening price for {symbol} — skipping hard stop")
                continue

            # Record opening price for drop-stop calculation later
            self.open_prices[symbol] = open_price
            position.current_price = open_price

            # Hard stop: entry_price × (1 + HARD_STOP_PCT)
            stop_level = position.entry_price * (1.0 + config.HARD_STOP_PCT)
            if open_price <= stop_level:
                logger.warning(
                    f"HARD STOP TRIGGERED: {symbol} open={open_price:.4f} "
                    f"<= stop={stop_level:.4f} (entry={position.entry_price:.4f})"
                )
                exits_triggered.append(symbol)

        # Execute exits for triggered stops
        for symbol in exits_triggered:
            self._exit_single_position(symbol, "hard stop at open")

        if exits_triggered:
            logger.info(f"Hard stops: exited {len(exits_triggered)} positions")
        else:
            logger.info(f"Hard stops: no triggers ({len(positions)} positions checked)")

    def _check_drop_stops(self):
        """9:35 AM: Check for 6% drop from open high."""
        logger.info("DROP STOP CHECK: checking 9:35 prices for 6% drop from open high")

        positions = list(self.position_mgr.positions.items())
        if not positions:
            return

        symbols = [s for s, _ in positions]
        snapshots = self.alpaca.get_snapshots(symbols)

        exits_triggered = []
        for symbol, position in positions:
            snap = snapshots.get(symbol, {})
            price_935 = snap.get("last_price") or snap.get("close")
            if not price_935:
                logger.warning(f"No 9:35 price for {symbol} — skipping drop stop")
                continue

            position.current_price = price_935

            # open_high = max(open_price, price_at_935)
            open_price = self.open_prices.get(symbol, position.entry_price)
            open_high = max(open_price, price_935)

            if open_high > 0:
                drop_from_high = (open_high - price_935) / open_high
                if drop_from_high >= config.DROP_STOP_PCT:
                    logger.warning(
                        f"DROP STOP TRIGGERED: {symbol} drop={drop_from_high:.2%} "
                        f"(open_high={open_high:.4f}, price_935={price_935:.4f})"
                    )
                    exits_triggered.append(symbol)

        for symbol in exits_triggered:
            self._exit_single_position(symbol, "6% drop-stop at 9:35")

        if exits_triggered:
            logger.info(f"Drop stops: exited {len(exits_triggered)} positions")
        else:
            logger.info(f"Drop stops: no triggers ({len(positions)} positions checked)")

    def _exit_single_position(self, symbol: str, reason: str):
        """Exit a single position with market sell."""
        position = self.position_mgr.positions.get(symbol)
        if not position:
            return

        qty = position.quantity
        sell_resp = self.position_mgr._submit_sell_order(symbol, qty)
        if not sell_resp:
            # Fallback: limit sell
            last_price = self.position_mgr._get_last_price(symbol)
            if last_price and last_price > 0:
                limit_price = round(last_price * 0.97, 4)
                sell_resp = self.position_mgr._submit_sell_order(symbol, qty, "limit", limit_price)

        if not sell_resp:
            logger.error(f"Failed to exit {symbol} ({reason})")
            return

        order_id = sell_resp.get("id")
        if order_id:
            fill = self.position_mgr.get_order_fill(order_id, max_wait=30)
            if fill:
                filled_qty = int(fill["filled_qty"])
                exit_price = fill["filled_avg_price"]
                pnl = (exit_price - position.entry_price) * filled_qty
                pnl_pct = ((exit_price / position.entry_price) - 1) * 100
                logger.info(
                    f"EXIT {symbol}: {filled_qty} @ {exit_price:.4f} "
                    f"(P&L: {pnl:+.2f}, {pnl_pct:+.1f}%) — {reason}"
                )
                if filled_qty >= position.quantity:
                    self.position_mgr.positions.pop(symbol, None)
                else:
                    position.quantity -= filled_qty
                    logger.warning(f"Partial exit {symbol}: {filled_qty}/{qty}, {position.quantity} remaining")
            else:
                logger.error(f"No fill confirmation for {symbol} exit")

    def _exit_all_positions(self, reason: str):
        """Exit ALL remaining positions with market sells."""
        positions = list(self.position_mgr.positions.keys())
        if not positions:
            logger.info(f"{reason}: no positions to exit")
            return

        logger.info(f"{reason}: exiting {len(positions)} positions")
        for symbol in positions:
            self._exit_single_position(symbol, reason)

        # Check what's left
        remaining = self.position_mgr.get_position_count()
        if remaining > 0:
            logger.warning(f"{reason}: {remaining} positions still remaining after exit attempt")
        else:
            logger.info(f"{reason}: all positions exited successfully")

    # ════════════════════════════════════════════════════════════
    # AFTERNOON DATA & SCORING METHODS
    # ════════════════════════════════════════════════════════════

    def _step_collect_data(self):
        """~3:30 PM: Build base universe (Stages A+B+D). Stage C runs at 3:48."""
        logger.info("=" * 50)
        logger.info("DATA COLLECTION: Building base universe (staged pipeline)")
        logger.info("=" * 50)

        try:
            final, diag, adv_cache, atr_cache = build_universe(
                self.massive, self.alpaca,
            )

            self.universe = final
            self._universe_diag = diag
            self._adv_cache = adv_cache
            self._atr_cache = atr_cache

            if not self.universe:
                logger.error("Empty universe after pipeline — cannot proceed")
                return

            save_universe_audit(diag, final)

            self.data_collected = True
            self._save_state()
            logger.info(f"Base universe ready: {len(self.universe)} symbols (Stage C deferred to 3:48)")

        except Exception as e:
            logger.exception(f"Error in data collection: {e}")

    def _step_score_and_rank(self):
        """~3:48 PM: Fetch 9:30-3:50 bars, build 350-model candidates, score."""
        logger.info("=" * 50)
        logger.info("SCORING (350 model): Fetching signal bars and scoring")
        logger.info("=" * 50)

        try:
            today = date.today().isoformat()

            # 1. Fetch 9:30-3:50 minute bars for the full base universe
            logger.info(f"Fetching 9:30-3:50 minute bars for {len(self.universe)} symbols...")
            self._minute_bars = self.alpaca.get_intraday_bars_for_signal(
                self.universe, today, start="09:30", end="15:50",
            )

            # 2. Stage C: minute-bar data quality filter
            pre_c_count = len(self.universe)
            quality_passed = filter_minute_data_quality(
                self.universe,
                self._minute_bars,
                min_minute_bars=30,
                diag=self._universe_diag,
            )
            logger.info(f"Stage C data quality: {pre_c_count} → {len(quality_passed)}")
            self.universe = quality_passed

            if not self.universe:
                logger.error("Empty universe after Stage C data quality — cannot score")
                self.scoring_done = True
                return

            # 3. Fetch SPY return (open to current)
            spy_snap = self.alpaca.get_snapshots([config.MARKET_BENCHMARK])
            spy_data = spy_snap.get(config.MARKET_BENCHMARK, {})
            spy_open = spy_data.get("open") or 0
            spy_last = spy_data.get("last_price") or spy_data.get("close") or 0
            spy_return = (spy_last - spy_open) / spy_open if spy_open > 0 else 0.0
            logger.info(f"SPY return: {spy_return:.4f} (open={spy_open}, last={spy_last})")

            # 4. Build volume profiles (60-min)
            volume_last_60min: Dict[str, int] = {}
            volume_avg_60min: Dict[str, float] = {}
            for symbol in self.universe:
                bars = self._minute_bars.get(symbol, [])
                vol_60, avg_60 = self.alpaca.get_volume_profile_60min(bars)
                volume_last_60min[symbol] = vol_60
                volume_avg_60min[symbol] = avg_60

            # 5. Build candidates from minute bars
            candidates = build_signal_candidates_350(
                self.universe, self._minute_bars,
                self._adv_cache, self._atr_cache,
            )

            if not candidates:
                logger.error("No valid candidates after build_signal_candidates_350")
                self.scoring_done = True
                return

            # 5. Compute raw metrics
            candidates = compute_raw_metrics_350(
                candidates, spy_return,
                volume_last_60min, volume_avg_60min,
            )

            # 6. Normalize, score, bucket
            candidates = normalize_and_score_350(candidates)
            candidates = assign_buckets(candidates)
            candidates.sort(key=lambda c: c.composite_score, reverse=True)

            self.scored_candidates = candidates
            self.scoring_done = True
            self._save_state()

            # Log top 10
            logger.info(f"Scoring complete: {len(candidates)} scored")
            for c in candidates[:10]:
                logger.info(
                    f"  {c.symbol}: score={c.composite_score:.3f} bucket={c.bucket} "
                    f"ret={c.intraday_return:.2%} prox={c.proximity_to_high:.3f} "
                    f"vol_vs_avg={c.volume_vs_avg:.2f} atr%={c.atr_percent:.3f}"
                )

            # Save candidates audit artifact
            top_20_dicts = [
                {
                    "symbol": c.symbol, "score": round(c.composite_score, 4),
                    "bucket": c.bucket, "intraday_return": round(c.intraday_return, 4),
                    "proximity_to_high": round(c.proximity_to_high, 4),
                    "volume_vs_avg": round(c.volume_vs_avg, 2),
                    "volume_trend": round(c.volume_trend, 2),
                    "vs_market": round(c.vs_market, 4),
                    "atr_percent": round(c.atr_percent, 4),
                    "signal_price": round(c.signal_price, 4),
                    "adv_dollars": round(c.adv_dollars, 0),
                }
                for c in candidates[:20]
            ]
            save_candidates_audit(top_20_dicts)

            # Also update universe audit with top 20
            if self._universe_diag:
                save_universe_audit(self._universe_diag, self.universe, scored_top20=top_20_dicts)

        except Exception as e:
            logger.exception(f"Error in scoring: {e}")
            self.scoring_done = True

    def _step_execute_entries(self):
        """3:50 PM: Select positions via account tier, size, execute market buys."""
        logger.info("=" * 50)
        logger.info("ENTRY EXECUTION: Submitting market buy orders")
        logger.info("=" * 50)

        exec_diag = ExecutionDiagnostics()
        self._exec_diag = exec_diag

        try:
            if not self.scored_candidates:
                logger.warning("No scored candidates — skipping entries")
                self.entries_done = True
                return

            # Get account equity and choose tier
            equity = self.position_mgr.get_account_equity()
            if not equity or equity <= 0:
                logger.error("Cannot determine account equity — skipping entries")
                self.entries_done = True
                return

            logger.info(f"Account equity: ${equity:,.2f}")
            sel = get_selection_config(equity)

            # Select and size positions
            selected, sizing = select_positions(self.scored_candidates, equity, sel)

            if not sizing:
                logger.warning("No positions selected after sizing — skipping entries")
                self.entries_done = True
                return

            exec_diag.selected_symbols = [c.symbol for c in selected if sizing.get(c.symbol, 0) > 0]

            # Execution eligibility gate — fetch fresh snapshots, reject unorderable
            fresh_snaps = self.alpaca.get_snapshots(exec_diag.selected_symbols)
            orderable, exec_rejected = filter_execution_ready(
                exec_diag.selected_symbols, fresh_snaps,
                max_spread_pct=0.05, require_quote=True,
            )
            exec_diag.orderable_symbols = list(orderable)
            exec_diag.rejected_symbols = dict(exec_rejected)
            orderable_set = set(orderable)

            if exec_rejected:
                for sym, reason in exec_rejected.items():
                    logger.warning(f"Execution reject {sym}: {reason}")
                    sizing.pop(sym, None)

            # Submit market buy orders
            total_deployed = 0.0

            for candidate in selected:
                if candidate.symbol not in orderable_set:
                    continue
                symbol = candidate.symbol
                qty = sizing.get(symbol, 0)
                if qty <= 0:
                    continue

                buy_resp = self.position_mgr.submit_buy_order(symbol, qty)
                if not buy_resp:
                    logger.error(f"Failed to submit buy for {symbol} x{qty}")
                    exec_diag.failed_submissions[symbol] = "submit_failed"
                    continue

                order_id = buy_resp.get("id")
                if not order_id:
                    exec_diag.failed_submissions[symbol] = "no_order_id"
                    continue

                exec_diag.submitted_symbols.append(symbol)

                fill = self.position_mgr.get_order_fill(order_id, max_wait=30)
                if fill and int(fill["filled_qty"]) > 0:
                    filled_qty = int(fill["filled_qty"])
                    fill_price = fill["filled_avg_price"]

                    position = Position(
                        symbol=symbol,
                        entry_price=fill_price,
                        quantity=filled_qty,
                        entry_time=datetime.now(),
                        entry_gap_pct=0.0,
                        adv_estimate=candidate.adv_dollars,
                        peak_price=fill_price,
                        current_price=fill_price,
                    )
                    self.position_mgr.positions[symbol] = position
                    total_deployed += fill_price * filled_qty
                    exec_diag.filled_symbols.append(symbol)
                    exec_diag.fill_details[symbol] = {
                        "qty": filled_qty, "price": round(fill_price, 4),
                        "score": round(candidate.composite_score, 4),
                        "bucket": candidate.bucket,
                    }

                    logger.info(
                        f"ENTRY {symbol}: {filled_qty} @ {fill_price:.4f} "
                        f"(score={candidate.composite_score:.3f}, bucket={candidate.bucket})"
                    )
                else:
                    logger.warning(f"No fill for {symbol} buy order")
                    exec_diag.failed_submissions[symbol] = "no_fill"

            self.entries_done = True
            self._save_state()

            # Store execution stats for health report
            self._exec_stats = {
                "selected": len(exec_diag.selected_symbols),
                "orderable": len(exec_diag.orderable_symbols),
                "exec_rejected": len(exec_diag.rejected_symbols),
                "exec_rejected_reasons": exec_diag.rejected_symbols,
                "orders_submitted": len(exec_diag.submitted_symbols),
                "entries_filled": len(exec_diag.filled_symbols),
                "total_deployed": total_deployed,
                "equity": equity,
            }

            logger.info(
                f"Entry execution complete: {len(exec_diag.filled_symbols)} filled, "
                f"{len(exec_diag.rejected_symbols)} rejected at execution gate, "
                f"${total_deployed:,.2f} deployed "
                f"({total_deployed / equity * 100:.1f}% of equity)"
            )

        except Exception as e:
            logger.exception(f"Error in entry execution: {e}")
            self.entries_done = True

    # ════════════════════════════════════════════════════════════
    # INFRASTRUCTURE (failsafe, state, etc.)
    # ════════════════════════════════════════════════════════════

    def _run_failsafe_flatten(self, label: str):
        """Broker-based catch-all flatten with multi-layer retry."""
        logger.warning(f"{label}: starting broker-based failsafe flatten")

        summary = self.position_mgr.force_flatten_broker_positions(label)

        logger.warning(
            f"{label}: failsafe flatten complete | "
            f"positions_seen={summary['positions_seen']} | "
            f"closes_submitted={summary['closes_submitted']} | "
            f"fills_confirmed={summary['fills_confirmed']} | "
            f"errors={len(summary['errors'])}"
        )

        manual = summary.get("manual_required", [])
        if manual:
            for item in manual:
                logger.critical(f"{label}: {item}")

        remaining = self.position_mgr.broker_position_count()
        if remaining == 0:
            self.position_mgr.positions.clear()
            logger.warning(f"{label}: broker confirmed flat — local state cleared")
        elif remaining < 0:
            logger.error(f"{label}: broker API unreachable after failsafe — cannot confirm flat")
        else:
            logger.error(f"{label}: broker still shows {remaining} open positions after failsafe")

        self._save_state()

    def _save_end_of_day_reports(self):
        """Write all daily diagnostic artifacts. Called on EVERY completed market day."""
        try:
            stats = self._exec_stats
            save_run_health(
                diag=self._universe_diag,
                scored_count=len(self.scored_candidates),
                selected_count=stats.get("selected", 0),
                orderable_count=stats.get("orderable", 0),
                filled_count=stats.get("entries_filled", 0),
                total_deployed=stats.get("total_deployed", 0.0),
                equity=stats.get("equity", 0.0),
                exec_rejected=stats.get("exec_rejected_reasons"),
                extra={"api_calls_total": get_api_call_count()},
            )
        except Exception as e:
            logger.error(f"Failed to save health report: {e}")

        try:
            if self._universe_diag:
                save_universe_audit(self._universe_diag, self.universe)
        except Exception as e:
            logger.error(f"Failed to save universe audit: {e}")

        try:
            if self.scored_candidates:
                top_20 = [
                    {
                        "symbol": c.symbol, "score": round(c.composite_score, 4),
                        "bucket": c.bucket, "intraday_return": round(c.intraday_return, 4),
                    }
                    for c in self.scored_candidates[:20]
                ]
                save_candidates_audit(top_20)
        except Exception as e:
            logger.error(f"Failed to save candidates audit: {e}")

        try:
            if self._exec_diag:
                save_execution_audit(self._exec_diag)
        except Exception as e:
            logger.error(f"Failed to save execution audit: {e}")

    def _finalize_day(self, clear_state: bool = True):
        """End-of-day: write reports, optionally clear state.

        When clear_state=True (no-entry day), we clear bot flags so tomorrow
        starts fresh, and only persist positions (which should be empty).
        We deliberately do NOT call _save_state() after clearing, because
        _save_state() would re-write the bot flags we just cleared.
        """
        logger.info("Finalizing trading day")
        self._save_end_of_day_reports()
        if clear_state:
            self.state_mgr.clear_bot_state()
            # Only persist positions (should be empty); do NOT re-save bot flags
            self.state_mgr.save_positions(self.position_mgr.positions)
        else:
            self._save_state()

    def _save_state(self):
        """Persist current state to disk."""
        try:
            # Save positions
            self.state_mgr.save_positions(self.position_mgr.positions)

            # Save bot state
            bot_state = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "morning_exits_done": self.morning_exits_done,
                "hard_stops_checked": self.hard_stops_checked,
                "drop_stops_checked": self.drop_stops_checked,
                "final_exit_done": self.final_exit_done,
                "data_collected": self.data_collected,
                "scoring_done": self.scoring_done,
                "entries_done": self.entries_done,
                "open_prices": self.open_prices,
            }
            self.state_mgr.save_bot_state(bot_state)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def _load_state(self):
        """Load state from previous run (same-day recovery only)."""
        today = datetime.now().strftime("%Y-%m-%d")
        bot_state = self.state_mgr.load_bot_state()

        if not bot_state or bot_state.get("date") != today:
            logger.info("No same-day state to restore — fresh start")
            # Load positions from file (may have overnight holds from yesterday's entries)
            saved = self.state_mgr.load_positions()
            if saved:
                self.position_mgr.load_positions(saved)
                logger.info(f"Loaded {len(saved)} saved positions")
            return

        # Same-day state: restore flags
        logger.info("Restoring same-day bot state")
        self.morning_exits_done = bot_state.get("morning_exits_done", False)
        self.hard_stops_checked = bot_state.get("hard_stops_checked", False)
        self.drop_stops_checked = bot_state.get("drop_stops_checked", False)
        self.final_exit_done = bot_state.get("final_exit_done", False)
        self.data_collected = bot_state.get("data_collected", False)
        self.scoring_done = bot_state.get("scoring_done", False)
        self.entries_done = bot_state.get("entries_done", False)
        self.open_prices = bot_state.get("open_prices", {})

        # Load positions
        saved = self.state_mgr.load_positions()
        if saved:
            self.position_mgr.load_positions(saved)
            logger.info(f"Loaded {len(saved)} saved positions")


def main():
    bot = OvernightMomentumBot()
    bot.run()


if __name__ == "__main__":
    main()
