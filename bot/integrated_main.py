"""Combined Overnight Rebound Bot — Main Orchestrator

Sleeve 1: MR_WIDE (Mean Reversion)
  - Buy 15:50, $1–5, day_ret <= -3%, vol_ratio >= 1.5x, close_position <= 0.20
  - Exit 09:30

Sleeve 2: GDP_BASE / MOM_CLEAN (Green-Day Pullback)
  - Buy 15:50, $1–10, day_ret +1% to +10%, below VWAP, late_mom <= 0
  - Exit 09:30

Production allocation: static 70/30 MR/GDP, 10% max single-name cap.

Daily Schedule (ET) — morning bot starts around 05:00 AM:

MORNING (T+1 exits — positions from yesterday's 15:50 entries):
  05:00  Start, detect overnight positions from broker
  05:00, 05:15, 05:30, 05:45  Rolling premarket dynamic limit classification (decisive symbols only)
  06:00  Final premarket classification for all unresolved symbols
  09:25  Cancel any remaining premarket limits before open exits
  09:30  Green/flat positions sell immediately; red positions get 1% trailing-stop orders
  10:00  Cancel any remaining trailing orders; force-flatten anything still open

AFTERNOON (T-1 entries — new positions for tomorrow's exits):
  15:30  Build universe (Massive + Alpaca, $1–10, ADV sizing cap protects)
  15:50  Fetch latest 9:30-15:50 minute bars, build both MR and GDP candidates
  15:50  Execute entries immediately after scoring
  16:00  Confirm positions held overnight, save state, done
"""
import logging
import math
import os
import sys
import time
from datetime import datetime, time as dt_time, date
from typing import List, Optional, Dict, Any, Tuple
from zoneinfo import ZoneInfo

from bot import config
from bot.massive_client import MassiveClient
from bot.market_data import AlpacaDataClient
from bot.mean_reversion_scorer import (
    MeanReversionCandidate,
    build_mean_reversion_candidates,
    filter_mean_reversion_candidates,
)
from bot.green_day_pullback_scorer import (
    GreenDayPullbackCandidate,
    build_green_day_pullback_candidates,
    filter_green_day_pullback_candidates,
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

_ET = ZoneInfo("America/New_York")


def _parse_config_time(time_str: str) -> dt_time:
    """Parse an 'HH:MM' config string into a datetime.time object."""
    parts = time_str.split(":")
    return dt_time(int(parts[0]), int(parts[1]))


class CombinedOvernightReboundBot:
    """Main bot orchestrator for combined MR_WIDE + GDP_BASE strategy"""

    def __init__(self):
        self.massive = MassiveClient()
        self.alpaca = AlpacaDataClient()
        self.position_mgr = PositionManager()
        self.state_mgr = StateManager()

        # Universe & candidates
        self.universe: List[str] = []
        self.mr_candidates: List[MeanReversionCandidate] = []
        self.gdp_candidates: List[GreenDayPullbackCandidate] = []
        self._universe_diag: Optional[UniverseDiagnostics] = None

        # Stage flags
        self.morning_exits_done = False   # All overnight positions exited
        self.data_collected = False       # Universe + daily bars ready
        self.scoring_done = False         # 3:50 PM scoring complete
        self.entries_done = False         # 3:50 PM entries executed

        # Sleeve-specific exit flags
        self.gdp_exits_done = False
        self.mr_exits_done = False
        self.red_trail_exit_submitted = False
        self.red_trail_order_ids: Dict[str, str] = {}
        self.red_trail_symbols: set = set()

        # Morning/overnight order management
        self.morning_open_orders_cancelled = False
        self.premarket_dynamic_limits_done = False
        self.premarket_limit_order_ids: Dict[str, str] = {}
        self.premarket_decided_symbols: set = set()  # Track symbols already decided in rolling checks
        # Legacy 20:00 overnight limit fields kept only for backward-compatible state loading.
        self.overnight_limit_sells_done = False
        self.overnight_limit_order_ids: Dict[str, str] = {}
        self.end_of_day_reports_done = False

        # Failsafe
        self.post_exit_failsafe_done = False

        # PDT guard: symbols sold today (no same-day re-entry when equity < $50k)
        self.sold_today: set = set()

        # Data collection results (stored between steps)
        self._minute_bars: Dict[str, List[dict]] = {}
        self._adv_cache: Dict[str, Tuple[float, float]] = {}
        self._exec_stats: Dict[str, Any] = {}
        self._exec_diag: Optional[ExecutionDiagnostics] = None

    def run(self):
        """Main bot loop - runs from 9:00 AM until after market close"""
        try:
            self._run()
        except Exception:  # noqa: BLE001
            logger.critical("UNHANDLED EXCEPTION in run() — bot terminated", exc_info=True)
            try:
                self._save_state()
            except Exception:
                logger.critical("State save also failed after crash", exc_info=True)
            raise

    def _validate_config(self):
        """Fail fast on config combinations that contradict the researched setup."""
        total_alloc = config.MR_ALLOCATION_PCT + config.GDP_ALLOCATION_PCT
        if abs(total_alloc - 1.0) > 1e-6:
            raise ValueError(
                f"Allocation must sum to 1.0, got MR={config.MR_ALLOCATION_PCT}, "
                f"GDP={config.GDP_ALLOCATION_PCT}, total={total_alloc}"
            )

        if _parse_config_time(config.SCORING_TIME) > _parse_config_time(config.ENTRY_TIME):
            raise ValueError(
                f"SCORING_TIME must be <= ENTRY_TIME, got "
                f"{config.SCORING_TIME} > {config.ENTRY_TIME}"
            )

        if config.GDP_EXIT_TIME != config.MR_EXIT_TIME:
            logger.warning(
                f"GDP_EXIT_TIME ({config.GDP_EXIT_TIME}) != MR_EXIT_TIME ({config.MR_EXIT_TIME}); "
                f"current bot exits both at GDP_EXIT_TIME"
            )
        if getattr(config, "ENABLE_RED_OPEN_TRAIL_EXIT", False):
            if _parse_config_time(config.RED_OPEN_TRAIL_FAILSAFE_TIME) <= _parse_config_time(config.GDP_EXIT_TIME):
                raise ValueError(
                    "RED_OPEN_TRAIL_FAILSAFE_TIME must be after the 09:30 exit decision"
                )
            if config.RED_OPEN_TRAIL_PCT <= 0:
                raise ValueError("RED_OPEN_TRAIL_PCT must be positive")

    def _run(self):
        """Inner run — called by run() which wraps it with top-level error handling."""
        logger.info("=" * 60)
        logger.info("Combined Overnight Rebound Bot Starting")
        logger.info("=" * 60)

        self._validate_config()

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

        # Pre-compute schedule times from config
        t_exit_all     = _parse_config_time(config.GDP_EXIT_TIME)           # 09:30 (both sleeves)
        t_failsafe     = _parse_config_time(
            config.RED_OPEN_TRAIL_FAILSAFE_TIME
            if getattr(config, "ENABLE_RED_OPEN_TRAIL_EXIT", False)
            else config.V2_FAILSAFE_TIME
        )
        t_data_collect = _parse_config_time(config.DATA_COLLECTION_TIME)    # 15:30
        t_scoring      = _parse_config_time(config.SCORING_TIME)            # 15:50
        t_entry        = _parse_config_time(config.ENTRY_TIME)              # 15:50
        t_cancel_orders = _parse_config_time(getattr(config, "MORNING_CANCEL_OPEN_ORDERS_TIME", "09:25"))
        t_premarket_start = _parse_config_time(getattr(config, "PREMARKET_DYNAMIC_START_TIME", "05:00"))
        t_premarket_final = _parse_config_time(getattr(config, "PREMARKET_DYNAMIC_FINAL_TIME", "06:00"))
        t_premarket_interval = getattr(config, "PREMARKET_DYNAMIC_CHECK_INTERVAL_MINUTES", 15)
        t_market_close = dt_time(16, 0)

        # If starting after failsafe time with positions, flatten immediately
        # Only late-flatten during the morning exit window, not after market close.
        current_time = datetime.now(_ET).time()
        if (
            current_time >= t_failsafe
            and current_time < t_market_close
            and self.position_mgr.get_position_count() > 0
        ):
            logger.warning("Started after morning failsafe during regular session — flattening immediately")
            self._run_failsafe_flatten("late-start flatten")
            self.morning_exits_done = True

        # After market close there is no 20:00 limit workflow anymore.
        # The bot should be restarted around 05:00 for the 05:00-06:00 rolling premarket dynamic limits.
        if current_time >= t_market_close:
            logger.info("Started after market close — nothing to do until the 05:00-06:00 premarket run")
            return

        # Main event loop
        while True:
            now = datetime.now(_ET)
            current_time = now.time()

            # ════════════════════════════════════════════
            # MORNING: Manage overnight position exits
            # ════════════════════════════════════════════

            # Gate morning exits to prevent after-market execution on restart
            if current_time < t_market_close and not self.morning_exits_done:
                # Rolling premarket dynamic limit classification (05:00 → 06:00)
                # At each 15-minute checkpoint, classify only "decisive" symbols.
                # Leave unclear symbols unresolved for the next checkpoint.
                if (getattr(config, "ENABLE_PREMARKET_DYNAMIC_LIMIT_SELLS", False)
                        and not self.premarket_dynamic_limits_done
                        and current_time >= t_premarket_start
                        and current_time < t_cancel_orders):
                    # Calculate which checkpoint we're at
                    minutes_since_start = (current_time.hour * 60 + current_time.minute) - (t_premarket_start.hour * 60 + t_premarket_start.minute)
                    checkpoint_num = minutes_since_start // t_premarket_interval
                    checkpoint_time = dt_time(
                        t_premarket_start.hour + checkpoint_num // 60,
                        t_premarket_start.minute + checkpoint_num % 60
                    )
                    # Only trigger if we're at a checkpoint time (within 1 minute tolerance)
                    if abs((current_time.hour * 60 + current_time.minute) - (checkpoint_time.hour * 60 + checkpoint_time.minute)) <= 1:
                        checkpoint_str = checkpoint_time.strftime("%H:%M")
                        logger.info(f"PREMARKET CHECKPOINT: {checkpoint_str} - running dynamic limit classification")
                        self._place_premarket_dynamic_limit_sells(decision_time_str=checkpoint_str)

                # 09:25 — cancel any resting premarket limit/trailing orders before normal exits.
                # Run regardless of local position state for safety.
                if (not self.morning_open_orders_cancelled
                        and current_time >= t_cancel_orders):
                    logger.warning("09:25 order cleanup: canceling all open orders before 09:30 exits")
                    self.position_mgr.cancel_all_open_orders()
                    self.premarket_limit_order_ids.clear()
                    self.overnight_limit_order_ids.clear()
                    self.red_trail_order_ids.clear()
                    self.red_trail_symbols.clear()
                    self.morning_open_orders_cancelled = True
                    self._save_state()

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
                        # Update has_positions so exits can run immediately in this loop
                        has_positions = self.position_mgr.get_position_count() > 0

                if has_positions and not self.morning_exits_done:
                    # 09:30 — either run the red-open trailing experiment or fixed exit all.
                    if not self.gdp_exits_done and current_time >= t_exit_all:
                        if getattr(config, "ENABLE_RED_OPEN_TRAIL_EXIT", False):
                            self._submit_red_open_trail_or_sell_green()
                        else:
                            self._exit_sleeve_positions("GDP", "09:30 all positions (GDP)")
                            self._exit_sleeve_positions("MR", "09:30 all positions (MR)")
                        self.gdp_exits_done = True
                        self.mr_exits_done = True
                        self._save_state()

                    # Failsafe: 10:00 if red-open trail is enabled, otherwise V2_FAILSAFE_TIME.
                    if (self.gdp_exits_done and self.mr_exits_done
                            and not self.post_exit_failsafe_done and current_time >= t_failsafe):
                        bc = self.position_mgr.broker_position_count()
                        if bc > 0:
                            logger.warning(f"Post-exit failsafe: broker still has {bc} positions")
                            self._run_failsafe_flatten(f"{t_failsafe.strftime('%H:%M')} post-exit failsafe")
                        elif bc == 0:
                            logger.info("Post-exit failsafe: broker confirmed flat")
                            if self.red_trail_order_ids:
                                logger.warning("Broker flat at failsafe; canceling remembered trailing-stop orders")
                                self.position_mgr.cancel_all_open_orders()
                                self.red_trail_order_ids.clear()
                                self.red_trail_symbols.clear()
                            self.position_mgr.reconcile_local_positions_from_broker()
                        self.post_exit_failsafe_done = True
                        self.morning_exits_done = True
                        self._save_state()

                    # Early completion: only allowed before failsafe when no trailing orders are active.
                    if (not getattr(config, "ENABLE_RED_OPEN_TRAIL_EXIT", False)
                            and self.gdp_exits_done and self.mr_exits_done
                            and self.position_mgr.get_position_count() == 0
                            and not self.morning_exits_done):
                        logger.info("All exits complete — no positions remaining")
                        self.morning_exits_done = True
                        self._save_state()

            # ════════════════════════════════════════════
            # AFTERNOON: Score universe and enter new positions
            # ════════════════════════════════════════════

            # 3:30 PM — Data collection
            if not self.data_collected and current_time >= t_data_collect:
                if current_time < dt_time(15, 50):
                    self._step_collect_data()
                else:
                    logger.warning("Past 3:50 PM without data collection — attempting now")
                    self._step_collect_data()

            # 3:50 PM — Score and rank using 9:30-15:50 bars
            if self.data_collected and not self.scoring_done and current_time >= t_scoring:
                self._step_score_and_rank()

            # 3:50 PM — Execute entries (requires scoring)
            if self.scoring_done and not self.entries_done and current_time >= t_entry:
                self._step_execute_entries()

            # ════════════════════════════════════════════
            # Day completion check
            # ════════════════════════════════════════════

            if current_time >= t_market_close:
                has_eod_positions = self.position_mgr.get_position_count() > 0

                if not self.end_of_day_reports_done:
                    if self.entries_done or has_eod_positions:
                        logger.info("Market closed — positions held overnight.")
                        self._save_end_of_day_reports()
                        self.end_of_day_reports_done = True
                        self._save_state()
                    else:
                        logger.info("Market closed — no entries made today.")
                        self._finalize_day()
                        break

                if self.entries_done or has_eod_positions:
                    logger.info("Market closed — day complete. Restart around 05:55 for 06:00 dynamic limits.")
                    self._save_state()
                break

            time.sleep(1)

    # ════════════════════════════════════════════════════════════
    # MORNING EXIT METHODS
    # ════════════════════════════════════════════════════════════

    def _exit_sleeve_positions(self, sleeve: str, reason: str):
        """Exit all positions of a specific sleeve at market.

        Batch behavior: submit every sell order first, then monitor fills.
        This prevents one slow/partial fill from delaying the next symbol's
        sell order by minutes at the open.

        Args:
            sleeve: "MR" or "GDP"
            reason: log description
        """
        # For GDP exit, also include UNKNOWN sleeve positions (safer early exit)
        if sleeve == "GDP":
            positions = [
                symbol for symbol, pos in self.position_mgr.positions.items()
                if getattr(pos, "sleeve", "MR") == sleeve
                or getattr(pos, "sleeve", "MR") == "UNKNOWN"
            ]
            unknown_positions = [
                symbol for symbol, pos in self.position_mgr.positions.items()
                if getattr(pos, "sleeve", "MR") == "UNKNOWN"
            ]
            if unknown_positions:
                logger.warning(f"EXIT {sleeve}: including {len(unknown_positions)} UNKNOWN sleeve positions: {unknown_positions}")
        else:
            positions = [
                symbol for symbol, pos in self.position_mgr.positions.items()
                if getattr(pos, "sleeve", "MR") == sleeve
            ]

        if not positions:
            logger.info(f"EXIT {sleeve}: no positions to sell")
            return

        logger.info(f"EXIT {sleeve}: market selling {len(positions)} positions: {positions}")

        # Pass 1: submit all sell orders
        submitted_orders = []  # List of (order_id, symbol, qty)

        for symbol in positions:
            position = self.position_mgr.positions.get(symbol)
            if not position:
                continue

            # Verify broker actually holds this position
            broker_pos = self.position_mgr.get_broker_position(symbol)
            if broker_pos is None:
                logger.warning(f"EXIT DEFER {symbol}: broker API error — skipping for retry")
                continue
            if broker_pos is self.position_mgr.BROKER_NOT_FOUND:
                logger.warning(f"EXIT SKIP {symbol}: broker confirmed no position — removing local")
                self.position_mgr.positions.pop(symbol, None)
                continue

            broker_qty = abs(int(float(broker_pos.get("qty", 0))))
            if broker_qty <= 0:
                logger.warning(f"EXIT SKIP {symbol}: broker qty=0 — removing local")
                self.position_mgr.positions.pop(symbol, None)
                continue

            qty = min(position.quantity, broker_qty)
            if qty != position.quantity:
                logger.warning(f"EXIT {symbol}: local qty={position.quantity} but broker qty={broker_qty} — selling {qty}")
                position.quantity = qty

            # Market sell
            sell_resp = self.position_mgr._submit_sell_order(symbol, qty)
            if not sell_resp:
                # Limit fallback
                last_price = self.position_mgr._get_last_price(symbol)
                if last_price and last_price > 0:
                    limit_price = self.position_mgr.round_limit_price(last_price * 0.97)
                    logger.warning(f"Market sell failed for {symbol}, trying limit @ {limit_price}")
                    sell_resp = self.position_mgr._submit_sell_order(symbol, qty, "limit", limit_price)

            if sell_resp and sell_resp.get("id"):
                order_id = sell_resp["id"]
                submitted_orders.append((order_id, symbol, qty))
            else:
                logger.error(f"Failed to submit sell for {symbol} x{qty}")

        # Pass 2: monitor fills
        for order_id, symbol, qty in submitted_orders:
            position = self.position_mgr.positions.get(symbol)
            if not position:
                continue

            fill = self.position_mgr.get_order_fill(order_id, max_wait=30)
            if fill and int(fill.get("filled_qty", 0)) > 0:
                filled_qty = int(fill["filled_qty"])
                exit_price = fill["filled_avg_price"]

                # PDT guard
                if filled_qty > 0:
                    self.sold_today.add(symbol)

                remaining = position.quantity - filled_qty
                if remaining > 0:
                    position.quantity = remaining
                    # Immediate resubmit for residual (same logic as old _exit_position)
                    slice_residual = qty - filled_qty
                    fill_status = fill.get("status", "unknown")
                    if slice_residual > 0 and fill_status != "filled":
                        logger.warning(f"EXIT PARTIAL {symbol}: filled {filled_qty}/{qty} — resubmitting {slice_residual}")
                        resub_resp = self.position_mgr._submit_sell_order(symbol, slice_residual)
                        if resub_resp and resub_resp.get("id"):
                            resub_id = resub_resp["id"]
                            resub_fill = self.position_mgr.get_order_fill(resub_id, max_wait=15)
                            if resub_fill:
                                resub_qty = int(resub_fill.get("filled_qty", 0))
                                remaining -= resub_qty
                                position.quantity = remaining
                                if resub_qty > 0:
                                    self.sold_today.add(symbol)
                                logger.info(f"EXIT RESUBMIT {symbol}: filled additional {resub_qty}, now {remaining} remaining")
                            else:
                                logger.warning(f"EXIT RESUBMIT {symbol}: no fill on resubmit order")
                        else:
                            logger.error(f"EXIT RESUBMIT {symbol}: resubmit failed")
                else:
                    self.position_mgr.positions.pop(symbol, None)
                    logger.info(f"EXIT FILLED {symbol}: qty={filled_qty}, price={exit_price:.4f}")
            else:
                logger.warning(f"EXIT NO FILL {symbol}: keeping for retry")

        # Reconcile local state with broker
        actions = self.position_mgr.reconcile_local_positions_from_broker()
        if actions:
            logger.info(f"EXIT {sleeve}: post-exit reconciliation adjustments: {actions}")

        # Count remaining positions (include UNKNOWN in GDP count for safety)
        if sleeve == "GDP":
            remaining = sum(
                1 for p in self.position_mgr.positions.values()
                if getattr(p, "sleeve", "MR") in ("GDP", "UNKNOWN")
            )
        else:
            remaining = sum(
                1 for p in self.position_mgr.positions.values()
                if getattr(p, "sleeve", "MR") == sleeve
            )
        logger.info(f"EXIT {sleeve}: done — {remaining} positions remaining")

    def _exit_single_position(self, symbol: str, reason: str):
        """Exit a single position — delegates to position_mgr._exit_position().

        That method handles: broker position check -> market sell -> limit
        fallback -> partial fill resubmit -> local state cleanup.
        We check whether the symbol is fully gone afterwards and persist state.
        """
        if symbol not in self.position_mgr.positions:
            return

        result = self.position_mgr._exit_position(symbol, reason)

        # PDT guard: only mark as sold if shares actually changed hands
        if result.get("filled_qty", 0) > 0:
            self.sold_today.add(symbol)

        still_held = symbol in self.position_mgr.positions
        if still_held:
            remaining = self.position_mgr.positions[symbol].quantity
            logger.warning(
                f"EXIT INCOMPLETE {symbol}: {remaining} shares still held — "
                f"failsafe will catch at {config.RED_OPEN_TRAIL_FAILSAFE_TIME if getattr(config, 'ENABLE_RED_OPEN_TRAIL_EXIT', False) else config.V2_FAILSAFE_TIME}"
            )
        self._save_state()

    # ════════════════════════════════════════════════════════════
    # AFTERNOON DATA & SCORING METHODS
    # ════════════════════════════════════════════════════════════

    def _step_collect_data(self):
        """~3:30 PM: Build universe (price/ADV/daily-bar filters). Stage C minute-quality runs at 3:50."""
        logger.info("=" * 50)
        logger.info("DATA COLLECTION: Building base universe (staged pipeline)")
        logger.info("=" * 50)

        try:
            final, diag, adv_cache, _atr_cache = build_universe(
                self.massive, self.alpaca,
            )

            self.universe = final
            self._universe_diag = diag
            self._adv_cache = adv_cache

            if not self.universe:
                logger.error("Empty universe after pipeline — cannot proceed")
                return

            save_universe_audit(diag, final)

            self.data_collected = True
            self._save_state()
            logger.info(f"Base universe ready: {len(self.universe)} symbols (Stage C minute-quality runs at 3:50)")

        except Exception as e:
            logger.exception(f"Error in data collection: {e}")

    def _step_score_and_rank(self):
        """~3:50 PM: Fetch 9:30-15:50 bars, run Stage C quality filter, build MR and GDP candidates."""
        logger.info("=" * 50)
        logger.info("SCORING (combined): Building MR and GDP candidates")
        logger.info("=" * 50)

        try:
            today = date.today().isoformat()
            signal_end = config.ENTRY_TIME  # 15:50

            # 1. Fetch 9:30-15:50 minute bars for the full base universe
            logger.info(f"Fetching 9:30-{signal_end} minute bars for {len(self.universe)} symbols...")
            self._minute_bars = self.alpaca.get_intraday_bars_for_signal(
                self.universe, today, start="09:30", end=signal_end,
            )

            # Log signal bar timestamps to verify data recency in live trading
            sample_last_times = []
            for sym, bars in list(self._minute_bars.items())[:20]:
                if bars:
                    sample_last_times.append((sym, bars[-1].get("t")))
            logger.info(f"Signal bar timestamp samples: {sample_last_times[:10]}")

            # 2. Stage C: minute-bar data quality filter
            pre_c_count = len(self.universe)
            quality_passed = filter_minute_data_quality(
                self.universe,
                self._minute_bars,
                min_minute_bars=30,
                diag=self._universe_diag,
            )
            logger.info(f"Stage C data quality: {pre_c_count} -> {len(quality_passed)}")
            self.universe = quality_passed

            if not self.universe:
                logger.error("Empty universe after Stage C data quality — cannot score")
                self.scoring_done = True
                return

            # 3. Build raw MR candidates from minute bars
            raw_mr = build_mean_reversion_candidates(
                self.universe,
                self._minute_bars,
                self._adv_cache,
            )
            filtered_mr = filter_mean_reversion_candidates(raw_mr)
            self.mr_candidates = filtered_mr  # Keep ALL passed candidates for allocator

            # CRITICAL DIAGNOSTIC: MR pipeline counts
            logger.info(
                f"MR pipeline: "
                f"universe={len(self.universe)}, "
                f"raw={len(raw_mr)}, "
                f"passed={len(filtered_mr)}"
            )

            # 4. Build raw GDP candidates from minute bars
            raw_gdp = build_green_day_pullback_candidates(
                self.universe,
                self._minute_bars,
                self._adv_cache,
            )
            filtered_gdp = filter_green_day_pullback_candidates(raw_gdp)

            # 5. Remove GDP candidates that are already MR candidates (MR takes priority)
            mr_symbols = {c.symbol for c in filtered_mr}  # Use full filtered list
            filtered_gdp = [c for c in filtered_gdp if c.symbol not in mr_symbols]
            self.gdp_candidates = filtered_gdp  # Keep ALL passed candidates for allocator

            # CRITICAL DIAGNOSTIC: GDP pipeline counts
            logger.info(
                f"GDP pipeline: "
                f"universe={len(self.universe)}, "
                f"raw={len(raw_gdp)}, "
                f"passed={len(filtered_gdp)}"
            )

            self.scoring_done = True
            self._save_state()

            logger.info(
                f"Combined scoring: "
                f"MR raw={len(raw_mr)} passed={len(filtered_mr)} top_slots={min(len(filtered_mr), config.MR_MAX_POSITIONS)} | "
                f"GDP raw={len(raw_gdp)} passed={len(filtered_gdp)} top_slots={min(len(filtered_gdp), config.GDP_MAX_POSITIONS)}"
            )

            # Log top MR candidates (only up to max positions)
            for c in self.mr_candidates[:config.MR_MAX_POSITIONS]:
                logger.info(
                    f"MR TOP {c.symbol}: price={c.signal_price:.2f}, "
                    f"day_ret={c.day_return:.2%}, vol_ratio={c.volume_ratio:.2f}x, "
                    f"close_pos={c.close_position:.2f}, "
                    f"late_drop={c.late_drop_1530_1550:.2%}, "
                    f"score={c.selection_score:.3f}"
                )

            # Log top GDP candidates (only up to max positions)
            for c in self.gdp_candidates[:config.GDP_MAX_POSITIONS]:
                logger.info(
                    f"GDP TOP {c.symbol}: price={c.signal_price:.2f}, "
                    f"day_ret={c.day_return:.2%}, price_vs_vwap={c.price_vs_vwap:.2%}, "
                    f"late_mom={c.late_mom_1530_signal:.2%}, "
                    f"close_pos={c.close_position:.2f}, vol_ratio={c.volume_ratio:.2f}x, "
                    f"score={c.selection_score:.3f}"
                )

            # Save candidates audit artifact
            def _mr_dict(c):
                return {
                    "symbol": c.symbol,
                    "sleeve": "MR",
                    "selection_score": round(c.selection_score, 4),
                    "signal_price": round(c.signal_price, 4),
                    "day_return": round(c.day_return, 4),
                    "volume_ratio": round(c.volume_ratio, 2),
                    "close_position": round(c.close_position, 3),
                    "late_drop_1530_1550": round(c.late_drop_1530_1550, 4),
                    "adv_dollars": round(c.adv_dollars, 0),
                }

            def _gdp_dict(c):
                return {
                    "symbol": c.symbol,
                    "sleeve": "GDP",
                    "selection_score": round(c.selection_score, 4),
                    "signal_price": round(c.signal_price, 4),
                    "day_return": round(c.day_return, 4),
                    "price_vs_vwap": round(c.price_vs_vwap, 4),
                    "late_mom_1530_signal": round(c.late_mom_1530_signal, 4),
                    "volume_ratio": round(c.volume_ratio, 2),
                    "close_position": round(c.close_position, 3),
                    "adv_dollars": round(c.adv_dollars, 0),
                }

            audit_dicts = {
                "mr_selected": [_mr_dict(c) for c in self.mr_candidates[:config.MR_MAX_POSITIONS]],
                "mr_all_passed": [_mr_dict(c) for c in self.mr_candidates],
                "gdp_selected": [_gdp_dict(c) for c in self.gdp_candidates[:config.GDP_MAX_POSITIONS]],
                "gdp_all_passed": [_gdp_dict(c) for c in self.gdp_candidates],
            }
            save_candidates_audit(audit_dicts)

            if self._universe_diag:
                top_mr = [_mr_dict(c) for c in self.mr_candidates[:20]]
                top_gdp = [_gdp_dict(c) for c in self.gdp_candidates[:20]]
                save_universe_audit(
                    self._universe_diag, self.universe,
                    scored_top20=top_mr + top_gdp,
                )

        except Exception as e:
            logger.exception(f"Error in scoring: {e}")
            self.scoring_done = True

    def _step_execute_entries(self):
        """3:50 PM: Dual-sleeve allocation (70/30 MR/GDP budget) -> execution-gate -> market buys."""
        logger.info("=" * 50)
        logger.info("ENTRY EXECUTION: Dual-sleeve market buy orders")
        logger.info("=" * 50)

        exec_diag = ExecutionDiagnostics()
        self._exec_diag = exec_diag

        # Sleeve allocation dataclass (local, simple)
        from dataclasses import dataclass
        from typing import Any

        @dataclass
        class SleeveAllocation:
            symbol: str
            shares: int
            target_dollars: float
            rank: int
            sleeve: str
            candidate: Any

        def allocate_waterfall(candidates, sleeve_budget: float, equity: float, sleeve_name: str, max_positions: int) -> list:
            """Two-pass waterfall allocator with effective minimum sizing.

            Pre-filters: cap checks, effective min (MIN_POSITION_DOLLARS vs MIN_SHARES*price)
            Pass 1: Equal share respecting caps (single name 10%, ADV 0.3%, MAX_POSITION_DOLLARS)
            Pass 2: Redistribute leftover to candidates with capacity (highest ADV first)

            Returns list of dicts with symbol, target_dollars, cap_dollars, adv_dollars.
            """
            if not candidates or sleeve_budget <= 0:
                return []

            max_single_dollars = equity * config.MAX_SINGLE_POSITION_PCT

            # Pre-filter: skip candidates where cap is below effective minimum
            viable = []
            skips = {}
            for c in candidates:
                adv_cap = c.adv_dollars * config.ADV_CAP_PCT if c.adv_dollars > 0 else 0.0
                cap = min(
                    adv_cap,
                    max_single_dollars,
                    config.MAX_POSITION_DOLLARS,
                )

                # Effective minimum considers both MIN_POSITION_DOLLARS and MIN_SHARES requirement
                effective_min = max(
                    config.MIN_POSITION_DOLLARS,
                    config.MIN_SHARES * c.signal_price,
                )

                if cap < effective_min:
                    skips[c.symbol] = (
                        f"cap_below_effective_min: cap=${cap:.2f}, "
                        f"effective_min=${effective_min:.2f}, adv=${c.adv_dollars:.0f}, price=${c.signal_price:.2f}"
                    )
                    continue

                viable.append(c)

            # Log skips if any
            if skips:
                logger.warning(f"{sleeve_name}: skipped {len(skips)} candidates (cap below effective min)")
                for sym, reason in list(skips.items())[:5]:  # Log first 5
                    logger.warning(f"  {sym}: {reason}")

            # Apply max_positions cap while preserving score order
            viable = viable[:max_positions]

            if not viable:
                logger.warning(f"{sleeve_name}: no viable candidates after cap/min-share filters")
                return []

            # Calculate per-candidate caps
            caps = {}
            for c in viable:
                adv_cap = c.adv_dollars * config.ADV_CAP_PCT
                caps[c.symbol] = min(
                    adv_cap,
                    max_single_dollars,
                    config.MAX_POSITION_DOLLARS,
                )

            allocations = {c.symbol: 0.0 for c in viable}
            base_target = sleeve_budget / len(viable)

            # Pass 1: Low ADV first - give everyone their capped equal share
            for c in sorted(viable, key=lambda x: x.adv_dollars):
                effective_min = max(config.MIN_POSITION_DOLLARS, config.MIN_SHARES * c.signal_price)
                alloc = min(base_target, caps[c.symbol])
                if alloc >= effective_min:
                    allocations[c.symbol] = alloc

            leftover = sleeve_budget - sum(allocations.values())

            # Pass 2: High ADV first - push leftover into names with room
            for c in sorted(viable, key=lambda x: x.adv_dollars, reverse=True):
                if leftover < config.MIN_POSITION_DOLLARS:
                    break
                room = caps[c.symbol] - allocations[c.symbol]
                if room <= 0:
                    continue
                add = min(leftover, room)
                allocations[c.symbol] += add
                leftover -= add

            # Build result list (filter by effective min one more time)
            result = []
            for c in viable:
                effective_min = max(config.MIN_POSITION_DOLLARS, config.MIN_SHARES * c.signal_price)
                if allocations[c.symbol] >= effective_min:
                    result.append({
                        "symbol": c.symbol,
                        "target_dollars": allocations[c.symbol],
                        "cap_dollars": caps[c.symbol],
                        "adv_dollars": c.adv_dollars,
                        "candidate": c,
                    })

            # Log allocation summary
            total_allocated = sum(r["target_dollars"] for r in result)

            # Warn about zero allocations (viable but got nothing)
            zero_alloc = [
                c.symbol for c in viable
                if allocations.get(c.symbol, 0.0) <= 0
            ]
            if zero_alloc:
                logger.warning(f"{sleeve_name}: zero allocations after waterfall: {zero_alloc[:10]}")

            logger.info(
                f"{sleeve_name} waterfall: "
                f"candidates={len(candidates)}, viable={len(viable)}, selected={len(result)}, "
                f"budget=${sleeve_budget:,.2f}, allocated=${total_allocated:,.2f}, leftover=${leftover:,.2f}"
            )

            return result

        try:
            # Get account equity and buying power
            equity = self.position_mgr.get_account_equity()
            if not equity or equity <= 0:
                logger.error("Cannot determine account equity — skipping entries")
                self.entries_done = True
                return

            buying_power = self.position_mgr.get_total_capital()
            if not buying_power or buying_power <= 0:
                logger.warning("Cannot determine buying power — falling back to equity")
                buying_power = equity

            deployable = min(buying_power, equity * config.MAX_LEVERAGE)
            logger.info(
                f"Account equity: ${equity:,.2f}, buying_power: ${buying_power:,.2f}, "
                f"deployable: ${deployable:,.2f}"
            )

            # PDT filter: remove recently-sold symbols from both sleeves
            if equity < 50_000 and self.sold_today:
                before_mr = len(self.mr_candidates)
                before_gdp = len(self.gdp_candidates)
                self.mr_candidates = [c for c in self.mr_candidates if c.symbol not in self.sold_today]
                self.gdp_candidates = [c for c in self.gdp_candidates if c.symbol not in self.sold_today]
                blocked_mr = before_mr - len(self.mr_candidates)
                blocked_gdp = before_gdp - len(self.gdp_candidates)
                if blocked_mr or blocked_gdp:
                    logger.warning(
                        f"PDT guard: filtered MR={blocked_mr}, GDP={blocked_gdp} "
                        f"same-day re-entry candidates (equity ${equity:,.0f} < $50k)"
                    )

            # Calculate sleeve budgets and target slots
            mr_budget = deployable * config.MR_ALLOCATION_PCT
            gdp_budget = deployable * config.GDP_ALLOCATION_PCT
            logger.info(
                f"Sleeve budgets: MR ${mr_budget:,.2f} ({config.MR_ALLOCATION_PCT:.0%}) | "
                f"GDP ${gdp_budget:,.2f} ({config.GDP_ALLOCATION_PCT:.0%})"
            )

            # Execution eligibility gate FIRST (before allocation)
            # This prevents budget from being assigned to names that fail spread check
            # Cap pool size to avoid large snapshot calls (3x max positions gives replacement depth)
            EXEC_POOL_MULTIPLIER = 3
            mr_pool = self.mr_candidates[:config.MR_MAX_POSITIONS * EXEC_POOL_MULTIPLIER]
            gdp_pool = self.gdp_candidates[:config.GDP_MAX_POSITIONS * EXEC_POOL_MULTIPLIER]
            candidate_symbols = [c.symbol for c in mr_pool + gdp_pool]
            fresh_snaps = self.alpaca.get_snapshots(candidate_symbols)
            orderable, exec_rejected = filter_execution_ready(
                candidate_symbols, fresh_snaps,
                max_spread_pct=0.05, require_quote=True,
            )
            orderable_set = set(orderable)

            if exec_rejected:
                for sym, reason in exec_rejected.items():
                    logger.warning(f"Execution reject {sym}: {reason}")

            # Filter candidates to only orderable symbols (from the capped pools)
            mr_orderable = [c for c in mr_pool if c.symbol in orderable_set]
            gdp_orderable = [c for c in gdp_pool if c.symbol in orderable_set]

            logger.info(
                f"Post-spread-filter: MR {len(self.mr_candidates)} -> {len(mr_orderable)} orderable, "
                f"GDP {len(self.gdp_candidates)} -> {len(gdp_orderable)} orderable"
            )

            # Allocate ONLY from orderable candidates (budget flows to clean names)
            mr_results = allocate_waterfall(
                mr_orderable, mr_budget, equity, "MR", config.MR_MAX_POSITIONS
            )
            gdp_results = allocate_waterfall(
                gdp_orderable, gdp_budget, equity, "GDP", config.GDP_MAX_POSITIONS
            )

            # Build SleeveAllocation list with shares calculated from target dollars
            allocations: List[SleeveAllocation] = []
            for rank, r in enumerate(mr_results, start=1):
                c = r["candidate"]
                shares = math.floor(r["target_dollars"] / c.signal_price) if c.signal_price > 0 else 0
                allocations.append(SleeveAllocation(
                    symbol=c.symbol,
                    shares=shares,
                    target_dollars=r["target_dollars"],
                    rank=rank,
                    sleeve="MR",
                    candidate=c
                ))

            for rank, r in enumerate(gdp_results, start=1):
                c = r["candidate"]
                shares = math.floor(r["target_dollars"] / c.signal_price) if c.signal_price > 0 else 0
                allocations.append(SleeveAllocation(
                    symbol=c.symbol,
                    shares=shares,
                    target_dollars=r["target_dollars"],
                    rank=rank,
                    sleeve="GDP",
                    candidate=c
                ))

            # Enforce combined position cap (prioritizes MR as appended first)
            if len(allocations) > config.COMBINED_MAX_POSITIONS:
                logger.warning(
                    f"Combined cap trimming allocations {len(allocations)} -> {config.COMBINED_MAX_POSITIONS}"
                )
                allocations = allocations[:config.COMBINED_MAX_POSITIONS]

            if not allocations:
                logger.warning("No positions sized across both sleeves — skipping entries")
                self.entries_done = True
                return

            exec_diag.selected_symbols = [a.symbol for a in allocations]
            exec_diag.orderable_symbols = [a.symbol for a in allocations]  # Only the ones we actually sized
            exec_diag.rejected_symbols = dict(exec_rejected)
            total_target = sum(a.target_dollars for a in allocations)
            logger.info(
                f"Selected {len(allocations)} allocations: "
                f"{sum(1 for a in allocations if a.sleeve=='MR')} MR + "
                f"{sum(1 for a in allocations if a.sleeve=='GDP')} GDP, "
                f"total_target=${total_target:,.2f}"
            )
            logger.info(
                f"Execution pool metrics: pool_size={len(candidate_symbols)}, "
                f"orderable={len(orderable_set)}, rejected_spread={len(exec_rejected)}"
            )
            # Submit market buy orders
            total_deployed = 0.0

            def _adaptive_qty(alloc: SleeveAllocation, bp_buffer: float = config.ENTRY_BP_BUFFER_PCT) -> int:
                """Return shares clamped to current buying power using target_dollars."""
                bp = self.position_mgr.get_total_capital()
                if not bp or bp <= 0:
                    return alloc.shares
                max_notional = bp * bp_buffer
                # Use target_dollars as the primary constraint, but cap at buying power
                target = min(alloc.target_dollars, max_notional)
                price_ref = alloc.candidate.signal_price
                if price_ref <= 0:
                    return 0
                return math.floor(target / price_ref)

            # Hard cutoff for new buy submissions (don't chase too close to close)
            entry_cutoff = dt_time(15, 58, 30)

            # Pass 1: submit all buy orders
            submitted_orders = []  # List of (order_id, alloc, qty, candidate)

            for alloc in allocations:
                # Check cutoff before processing
                if datetime.now(_ET).time() >= entry_cutoff:
                    logger.warning("ENTRY CUTOFF reached (15:58:30) — stopping new buy submissions")
                    break

                symbol = alloc.symbol
                if symbol not in orderable_set:
                    continue

                candidate = alloc.candidate
                price_ref = candidate.signal_price
                qty = _adaptive_qty(alloc)

                # Pre-submit logging
                planned_notional = qty * price_ref
                bp_before = self.position_mgr.get_total_capital() or 0.0
                logger.info(
                    f"ENTRY PLANNED {symbol}: qty={qty}, price_ref={price_ref:.4f}, "
                    f"notional={planned_notional:,.2f}, bp_before={bp_before:,.2f}, "
                    f"sleeve={alloc.sleeve}, rank={alloc.rank}"
                )

                if qty < config.MIN_SHARES:
                    logger.warning(
                        f"ENTRY SKIP {symbol}: adaptive qty {qty} < {config.MIN_SHARES} min shares "
                        f"(bp=${bp_before:,.2f}, price={price_ref:.4f})"
                    )
                    exec_diag.failed_submissions[symbol] = "bp_resize_below_min"
                    continue

                buy_resp = self.position_mgr.submit_buy_order(symbol, qty)
                if not buy_resp:
                    # Retry once with fresh BP
                    fresh_bp = self.position_mgr.get_total_capital()
                    if fresh_bp and fresh_bp > 0 and price_ref > 0:
                        retry_qty = math.floor((fresh_bp * config.ENTRY_BP_BUFFER_PCT) / price_ref)
                        if retry_qty >= config.MIN_SHARES and retry_qty < qty:
                            logger.warning(
                                f"ENTRY RETRY {symbol}: resizing {qty} -> {retry_qty} "
                                f"after submit failure (fresh_bp=${fresh_bp:,.2f})"
                            )
                            buy_resp = self.position_mgr.submit_buy_order(symbol, retry_qty)
                            if buy_resp:
                                qty = retry_qty

                if not buy_resp:
                    logger.error(f"Failed to submit buy for {symbol} x{qty}")
                    exec_diag.failed_submissions[symbol] = "submit_failed"
                    continue

                order_id = buy_resp.get("id")
                if not order_id:
                    exec_diag.failed_submissions[symbol] = "no_order_id"
                    continue

                exec_diag.submitted_symbols.append(symbol)
                submitted_orders.append((order_id, alloc, qty, candidate))

            # Pass 2: monitor fills for all submitted orders
            for order_id, alloc, qty, candidate in submitted_orders:
                symbol = alloc.symbol
                fill = self.position_mgr.get_order_fill(order_id, max_wait=10)
                if fill and int(fill["filled_qty"]) > 0:
                    filled_qty = int(fill["filled_qty"])
                    fill_price = fill["filled_avg_price"]

                    position = Position(
                        symbol=symbol,
                        entry_price=fill_price,
                        quantity=filled_qty,
                        entry_time=datetime.now(_ET),
                        adv_estimate=candidate.adv_dollars,
                        sleeve=alloc.sleeve,
                        current_price=fill_price,
                    )
                    self.position_mgr.positions[symbol] = position
                    total_deployed += fill_price * filled_qty
                    exec_diag.filled_symbols.append(symbol)
                    exec_diag.fill_details[symbol] = {
                        "qty": filled_qty, "price": round(fill_price, 4),
                        "score": round(candidate.selection_score, 4),
                        "day_return": round(candidate.day_return, 4),
                        "sleeve": alloc.sleeve,
                        "rank": alloc.rank,
                    }

                    logger.info(
                        f"ENTRY FILLED {symbol}: sleeve={alloc.sleeve}, "
                        f"qty={filled_qty}, avg={fill_price:.4f}, "
                        f"score={candidate.selection_score:.3f}"
                    )
                else:
                    self.position_mgr._cancel_order(order_id)
                    logger.warning(f"No fill for {symbol} buy order (order canceled)")
                    exec_diag.failed_submissions[symbol] = "no_fill"

            # Mop-up disabled (ENTRY_MOPUP_MAX_POSITIONS = 0)
            self.entries_done = True
            self._save_state()

            # Execution stats
            mr_filled = sum(1 for s in exec_diag.filled_symbols
                             if exec_diag.fill_details.get(s, {}).get("sleeve") == "MR")
            gdp_filled = sum(1 for s in exec_diag.filled_symbols
                              if exec_diag.fill_details.get(s, {}).get("sleeve") == "GDP")

            self._exec_stats = {
                "selected": len(exec_diag.selected_symbols),
                "orderable": len(exec_diag.orderable_symbols),
                "exec_rejected": len(exec_diag.rejected_symbols),
                "exec_rejected_reasons": exec_diag.rejected_symbols,
                "orders_submitted": len(exec_diag.submitted_symbols),
                "entries_filled": len(exec_diag.filled_symbols),
                "mr_filled": mr_filled,
                "gdp_filled": gdp_filled,
                "total_deployed": total_deployed,
                "equity": equity,
                "deployable": deployable,
            }

            deployment_pct = total_deployed / deployable * 100 if deployable > 0 else 0.0
            logger.info(
                f"Entry execution complete: {len(exec_diag.filled_symbols)} filled "
                f"({mr_filled} MR + {gdp_filled} GDP), "
                f"{len(exec_diag.rejected_symbols)} rejected at execution gate, "
                f"${total_deployed:,.2f} deployed "
                f"({deployment_pct:.1f}% of deployable)"
            )

            # Shortfall diagnostics
            if deployment_pct < 80.0:
                logger.warning("=== DEPLOYMENT SHORTFALL DIAGNOSTICS ===")
                if equity < 50_000 and self.sold_today:
                    logger.warning(f"PDT guard active: sold_today={self.sold_today}")
                if exec_diag.rejected_symbols:
                    logger.warning(f"Execution gate rejected: {len(exec_diag.rejected_symbols)} symbols")
                if exec_diag.failed_submissions:
                    logger.warning(f"Failed submissions: {len(exec_diag.failed_submissions)} symbols")

                # Sizing/rounding issues (candidates too small)
                planned_symbols = set([a.symbol for a in allocations])
                filled_symbols = set(exec_diag.filled_symbols)
                not_filled = planned_symbols - filled_symbols
                if not_filled:
                    logger.warning(f"Symbols not filled: {len(not_filled)} (e.g., {list(not_filled)[:5]})")

                # Target vs actual deployment
                target_deploy_pct = deployable / equity * 100
                logger.warning(
                    f"Target deployment: ${deployable:,.2f} ({target_deploy_pct:.1f}% of equity) "
                    f"-> Actual: ${total_deployed:,.2f} ({deployment_pct:.1f}%) "
                    f"= Gap of ${deployable - total_deployed:,.2f}"
                )
                logger.warning("=== END SHORTFALL DIAGNOSTICS ===")

        except Exception as e:
            logger.exception(f"Error in entry execution: {e}")
            self.entries_done = True

    # ════════════════════════════════════════════════════════════
    # INFRASTRUCTURE (failsafe, state, etc.)
    # ════════════════════════════════════════════════════════════

    def _position_reference_price(self, broker_pos: dict, symbol: str) -> Optional[float]:
        """Best available 09:30 decision price for red-open trailing logic.

        Prefer a fresh snapshot/last price at decision time — the broker
        position endpoint's current_price can lag around the open. Fall back
        to broker position fields only if the snapshot call fails.
        """
        try:
            px = self.position_mgr._get_last_price(symbol)
            if px and px > 0:
                return float(px)
        except Exception:
            logger.warning(f"RED TRAIL {symbol}: failed to fetch last price", exc_info=True)

        for key in ("current_price", "lastday_price"):
            raw = broker_pos.get(key) if broker_pos else None
            try:
                px = float(raw)
                if px > 0:
                    return px
            except (TypeError, ValueError):
                pass

        return None

    def _submit_red_open_trail_or_sell_green(self):
        """09:30 exit decision for the red-open trailing-stop paper test.

        Batch behavior:
        - Trailing-stop orders are submitted without waiting.
        - Green/flat market exits are all submitted first, then monitored.
        - This prevents one slow/partial fill from delaying the next symbol's
          sell order by minutes at the open.

        - If a position is green/flat versus its stored entry price, sell it at market.
        - If a position is red versus its stored entry price, submit an Alpaca
          trailing-stop sell order using config.RED_OPEN_TRAIL_PCT.
        - Any bad/missing price or qty data falls back to an immediate market sell.
        - The failsafe at RED_OPEN_TRAIL_FAILSAFE_TIME cancels any remaining
          trailing orders and force-flattens.
        """
        if self.red_trail_exit_submitted:
            logger.info("RED TRAIL: 09:30 exit decision already submitted; skipping duplicate call")
            return

        broker_positions = self.position_mgr.get_broker_positions()
        if broker_positions is None:
            logger.error("RED TRAIL: cannot read broker positions; falling back to local market exits")
            for symbol in list(self.position_mgr.positions.keys()):
                self._exit_single_position(symbol, "red-trail broker read failed fallback")
            self.red_trail_exit_submitted = True
            self._save_state()
            return

        broker_by_symbol = {
            str(p.get("symbol", "")).upper(): p
            for p in broker_positions
            if p.get("symbol")
        }

        # Reconcile first so UNKNOWN broker-only positions are included in this decision.
        self.position_mgr.reconcile_local_positions_from_broker()

        submitted_trails = 0
        market_exits = 0
        fallbacks = 0
        market_orders: Dict[str, Dict[str, Any]] = {}

        # Pass 1: submit trailing stops and collect market exit symbols
        for symbol, pos in list(self.position_mgr.positions.items()):
            sym = symbol.upper()
            broker_pos = broker_by_symbol.get(sym)

            if not broker_pos:
                logger.warning(f"RED TRAIL {symbol}: broker no longer holds symbol; removing local position")
                self.position_mgr.positions.pop(symbol, None)
                continue

            try:
                broker_qty = int(abs(float(broker_pos.get("qty", pos.quantity))))
            except (TypeError, ValueError):
                broker_qty = int(pos.quantity or 0)

            entry_price = float(getattr(pos, "entry_price", 0.0) or broker_pos.get("avg_entry_price", 0.0) or 0.0)
            decision_price = self._position_reference_price(broker_pos, symbol)

            if broker_qty <= 0:
                logger.warning(f"RED TRAIL {symbol}: broker qty <= 0; removing local position")
                self.position_mgr.positions.pop(symbol, None)
                continue

            if entry_price <= 0 or not decision_price or decision_price <= 0:
                logger.warning(
                    f"RED TRAIL {symbol}: bad decision data "
                    f"entry={entry_price}, price={decision_price}; submitting market fallback"
                )
                resp = self.position_mgr._submit_sell_order(symbol, broker_qty)
                if resp and resp.get("id"):
                    market_orders[symbol] = {"order_id": resp["id"], "qty": broker_qty}
                    market_exits += 1
                else:
                    fallbacks += 1
                continue

            # Optional tiny buffer. Default 0.0 means any price below entry is red.
            red_trigger_price = entry_price * (1.0 - config.RED_OPEN_TRAIL_PRICE_BUFFER_PCT)

            if decision_price < red_trigger_price:
                resp = self.position_mgr.submit_trailing_stop_sell_order(
                    symbol=symbol,
                    qty=broker_qty,
                    trail_percent=config.RED_OPEN_TRAIL_PCT,
                )
                if resp and resp.get("id"):
                    order_id = resp["id"]
                    self.red_trail_order_ids[symbol] = order_id
                    self.red_trail_symbols.add(symbol)
                    self.sold_today.add(symbol)
                    submitted_trails += 1
                    logger.info(
                        f"RED TRAIL {symbol}: price={decision_price:.4f} < entry={entry_price:.4f}; "
                        f"submitted {config.RED_OPEN_TRAIL_PCT:.2f}% trailing-stop sell "
                        f"qty={broker_qty} order_id={order_id}"
                    )
                else:
                    logger.error(f"RED TRAIL {symbol}: trailing-stop submit failed; submitting market fallback")
                    resp = self.position_mgr._submit_sell_order(symbol, broker_qty)
                    if resp and resp.get("id"):
                        market_orders[symbol] = {"order_id": resp["id"], "qty": broker_qty}
                        market_exits += 1
                    else:
                        fallbacks += 1
            else:
                logger.info(
                    f"GREEN/FLAT EXIT {symbol}: price={decision_price:.4f} >= entry={entry_price:.4f}; "
                    f"submitting market sell"
                )
                resp = self.position_mgr._submit_sell_order(symbol, broker_qty)
                if resp and resp.get("id"):
                    market_orders[symbol] = {"order_id": resp["id"], "qty": broker_qty}
                    market_exits += 1
                else:
                    fallbacks += 1

        # Pass 2: monitor green/flat market exits only after all orders are in
        for symbol, meta in market_orders.items():
            pos = self.position_mgr.positions.get(symbol)
            if not pos:
                continue
            fill = self.position_mgr.get_order_fill(meta["order_id"], max_wait=20)
            if fill and int(fill.get("filled_qty", 0)) > 0:
                filled_qty = int(fill["filled_qty"])
                self.sold_today.add(symbol)
                remaining = int(pos.quantity) - filled_qty
                if remaining > 0:
                    pos.quantity = remaining
                    logger.warning(f"GREEN/FLAT EXIT {symbol}: partial fill {filled_qty}; {remaining} remaining")
                else:
                    logger.info(f"GREEN/FLAT EXIT {symbol}: filled {filled_qty}; removing local position")
                    self.position_mgr.positions.pop(symbol, None)
            else:
                logger.warning(f"GREEN/FLAT EXIT {symbol}: no confirmed fill yet — failsafe will retry")

        self.position_mgr.reconcile_local_positions_from_broker()
        self.red_trail_exit_submitted = True
        logger.warning(
            f"RED TRAIL DECISION COMPLETE: trailing_orders={submitted_trails}, "
            f"market_exits={market_exits}, fallbacks={fallbacks}, "
            f"open_trail_symbols={list(self.red_trail_order_ids.keys())}"
        )
        self._save_state()

    def _run_failsafe_flatten(self, label: str):
        """Broker-based catch-all flatten with multi-layer retry."""
        logger.warning(f"{label}: starting broker-based failsafe flatten")
        if self.red_trail_order_ids:
            logger.warning(
                f"{label}: canceling open trailing-stop orders before force flatten: "
                f"{list(self.red_trail_order_ids.keys())}"
            )
            self.position_mgr.cancel_all_open_orders()
            self.sold_today.update(self.red_trail_order_ids.keys())

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
            self.red_trail_order_ids.clear()
            self.red_trail_symbols.clear()
            logger.warning(f"{label}: broker confirmed flat — local state cleared")
        elif remaining < 0:
            logger.error(f"{label}: broker API unreachable after failsafe — cannot confirm flat")
        else:
            logger.error(f"{label}: broker still shows {remaining} open positions after failsafe")

        self._save_state()

    def _fetch_live_premarket_bars(self, symbol: str, decision_dt: datetime) -> List[dict]:
        """Fetch today's live IEX 1-minute bars from 04:00 through the decision time.

        This intentionally uses the Alpaca data API directly so the 06:00 live
        decision does not depend on historical cache files. With IEX, premarket
        bars can be sparse; the classifier treats sparse bars as usable signal
        rather than requiring dense coverage.
        """
        start_dt = datetime.combine(decision_dt.date(), dt_time(4, 0), tzinfo=_ET)
        end_dt = decision_dt
        data_url = getattr(config, "ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
        feed = getattr(config, "PREMARKET_DYNAMIC_DATA_FEED", getattr(config, "DATA_FEED", "iex"))
        url = f"{data_url}/v2/stocks/{symbol}/bars"
        params = {
            "timeframe": "1Min",
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "adjustment": "raw",
            "feed": feed,
            "limit": 1000,
        }
        try:
            resp = self.position_mgr.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            bars = payload.get("bars", [])
            return bars if isinstance(bars, list) else []
        except Exception:
            logger.warning(f"06:00 LIMIT {symbol}: failed to fetch live premarket bars", exc_info=True)
            return []

    @staticmethod
    def _bar_dt(bar: dict) -> Optional[datetime]:
        """Parse Alpaca bar timestamp into America/New_York datetime."""
        raw = bar.get("t") or bar.get("timestamp") or bar.get("time")
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=_ET)
            return parsed.astimezone(_ET)
        except Exception:
            return None

    @staticmethod
    def _bar_float(bar: dict, *keys: str) -> Optional[float]:
        """Read a float from Alpaca bar keys, supporting both short and long names."""
        for key in keys:
            if key in bar and bar.get(key) is not None:
                try:
                    return float(bar.get(key))
                except (TypeError, ValueError):
                    continue
        return None

    def _compute_live_premarket_metrics(self, symbol: str, buy_price: float, decision_dt: datetime) -> Dict[str, Any]:
        """Compute lenient IEX premarket metrics for the 06:00 dynamic classifier."""
        bars_raw = self._fetch_live_premarket_bars(symbol, decision_dt)
        normalized = []
        for bar in bars_raw:
            dt_val = self._bar_dt(bar)
            close = self._bar_float(bar, "c", "close")
            high = self._bar_float(bar, "h", "high")
            low = self._bar_float(bar, "l", "low")
            volume = self._bar_float(bar, "v", "volume") or 0.0
            if dt_val and close and high and low:
                normalized.append({
                    "dt": dt_val,
                    "close": close,
                    "high": high,
                    "low": low,
                    "volume": volume,
                })

        normalized.sort(key=lambda b: b["dt"])
        if not normalized:
            return {
                "has_data": False,
                "reason": "no_iex_premarket_bars",
                "premarket_minutes": 0,
            }

        first = normalized[0]
        last = normalized[-1]
        stale_minutes = (decision_dt - last["dt"]).total_seconds() / 60.0
        premarket_high = max(b["high"] for b in normalized)
        premarket_low = min(b["low"] for b in normalized)
        premarket_volume = sum(b["volume"] for b in normalized)
        current_price = last["close"]
        first_price = first["close"]

        return {
            "has_data": True,
            "reason": "iex_premarket_data",
            "first_premarket_time": first["dt"],
            "first_premarket_price": first_price,
            "current_time": last["dt"],
            "current_price": current_price,
            "premarket_high": premarket_high,
            "premarket_low": premarket_low,
            "premarket_minutes": len(normalized),
            "premarket_volume": premarket_volume,
            "last_bar_age_minutes": stale_minutes,
            "current_return": current_price / buy_price - 1.0,
            "distance_from_high": current_price / premarket_high - 1.0 if premarket_high > 0 else 0.0,
            "return_from_low": current_price / premarket_low - 1.0 if premarket_low > 0 else 0.0,
            "trend_from_first_bar": current_price / first_price - 1.0 if first_price > 0 else 0.0,
        }

    def _classify_iex_premarket_limit(self, pos: Position, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Lenient IEX-aware dynamic limit decision.

        Absence of IEX prints is not treated as proof there is no market-wide
        premarket activity, but it does make it less likely that this is one of
        the true runners we are trying not to cap. Therefore the fallback is a
        normal 5% harvest limit rather than no decision.
        """
        fallback_limit = getattr(config, "PREMARKET_DYNAMIC_DEFAULT_LIMIT_PCT", 0.05)
        sparse_wide_limit = getattr(config, "PREMARKET_DYNAMIC_SPARSE_HIGH_RETURN_LIMIT_PCT", 0.10)
        very_high = getattr(config, "PREMARKET_DYNAMIC_VERY_HIGH_RETURN_NO_CAP_PCT", 0.10)
        high = getattr(config, "PREMARKET_DYNAMIC_HIGH_RETURN_NO_CAP_PCT", 0.05)
        moderate = getattr(config, "PREMARKET_DYNAMIC_MODERATE_RETURN_PCT", 0.02)
        stale_max = getattr(config, "PREMARKET_DYNAMIC_MAX_STALE_MINUTES", 60)

        if not metrics.get("has_data"):
            return {
                "action": "PLACE_LIMIT",
                "limit_pct": fallback_limit,
                "reason": metrics.get("reason", "no_iex_data_default_5pct"),
            }

        bars = int(metrics.get("premarket_minutes", 0) or 0)
        stale = float(metrics.get("last_bar_age_minutes", 999) or 999)
        current_return = float(metrics.get("current_return", 0.0) or 0.0)
        distance_from_high = float(metrics.get("distance_from_high", 0.0) or 0.0)
        trend = float(metrics.get("trend_from_first_bar", 0.0) or 0.0)
        sleeve = str(getattr(pos, "sleeve", "UNKNOWN") or "UNKNOWN").upper()
        fresh_enough = stale <= stale_max

        # True runner: even sparse IEX activity is enough not to choke it.
        if current_return >= very_high and bars >= 1 and fresh_enough:
            return {"action": "NO_CAP", "limit_pct": None, "reason": "iex_very_high_return_no_cap"}

        # Strong runner: needs at least a little IEX confirmation, but not dense tape.
        if current_return >= high:
            if bars >= 2 and fresh_enough:
                # MR can still harvest below +10%; continuation/unknown stays uncapped.
                if sleeve == "MR" and current_return < very_high:
                    return {"action": "PLACE_LIMIT", "limit_pct": fallback_limit, "reason": "iex_high_return_mr_harvest_5pct"}
                return {"action": "NO_CAP", "limit_pct": None, "reason": "iex_high_return_no_cap"}
            return {"action": "PLACE_LIMIT", "limit_pct": sparse_wide_limit, "reason": "iex_high_return_sparse_wide_10pct"}

        # Moderate winner: if it is building and near highs, give it more room.
        if current_return >= moderate:
            if distance_from_high > -0.01 and trend > 0 and fresh_enough:
                return {"action": "PLACE_LIMIT", "limit_pct": 0.07, "reason": "iex_moderate_near_high_7pct"}
            if distance_from_high < -0.03 or trend < 0:
                return {"action": "PLACE_LIMIT", "limit_pct": 0.05, "reason": "iex_moderate_fading_5pct"}
            return {"action": "PLACE_LIMIT", "limit_pct": 0.06, "reason": "iex_moderate_default_6pct"}

        if current_return >= 0:
            return {"action": "PLACE_LIMIT", "limit_pct": 0.04, "reason": "iex_small_winner_4pct"}

        return {"action": "PLACE_LIMIT", "limit_pct": 0.03, "reason": "iex_negative_pop_harvest_3pct"}

    def _is_decisive_premarket_signal(
        self,
        decision_time: str,
        final_time: str,
        current_return: float,
        distance_from_high: float,
        trend_from_first_bar: float,
        minutes_traded: int,
        last_bar_age_minutes: float = 999,
        sleeve: str = "UNKNOWN",
    ) -> tuple[bool, str]:
        """
        Decisive = act now.
        Not decisive = leave unresolved and check again in 15 minutes.
        """
        # Final checkpoint: no more waiting.
        if decision_time >= final_time:
            return True, "final_checkpoint"

        fresh = last_bar_age_minutes <= getattr(config, "PREMARKET_DYNAMIC_MAX_STALE_MINUTES", 60)

        # 1. Obvious monster runner.
        # Even sparse IEX is enough here.
        if current_return >= 0.10 and minutes_traded >= 1 and fresh:
            return True, "decisive_very_high_return"

        # 2. Strong runner.
        # Needs a little confirmation, but not dense tape.
        if current_return >= 0.05 and minutes_traded >= 2 and fresh:
            return True, "decisive_high_return"

        # 3. Moderate runner candidate: already up, near high, building.
        if (
            current_return >= 0.02
            and distance_from_high > -0.01
            and trend_from_first_bar > 0
            and minutes_traded >= 2
            and fresh
        ):
            return True, "decisive_moderate_near_high_building"

        # 4. Clear harvest / fade signal.
        # It had strength but is already meaningfully below its premarket high.
        if (
            current_return >= 0.02
            and distance_from_high < -0.03
            and minutes_traded >= 2
            and fresh
        ):
            return True, "decisive_moderate_fading"

        # 5. MR-specific harvest signal.
        # MR does not need as much upside continuation evidence to justify taking profit.
        if (
            sleeve.upper() == "MR"
            and current_return >= 0.03
            and minutes_traded >= 2
            and fresh
        ):
            return True, "decisive_mr_harvest"

        return False, "not_decisive_wait"

    def _place_premarket_dynamic_limit_sells(self, decision_time_str: str = None):
        """Rolling premarket dynamic limit classification (05:00 → 06:00).

        At each 15-minute checkpoint (05:00, 05:15, 05:30, 05:45), classify only
        "decisive" symbols - those with clear enough signals to act now. Leave unclear
        symbols unresolved for the next checkpoint. At 06:00 (final checkpoint),
        classify all remaining unresolved symbols with the normal dynamic rule.

        This replaces the old 20:00 blanket limit workflow. Orders are submitted
        in a non-blocking batch style, then the normal 09:25 cleanup cancels any
        remaining open limits before the 09:30 exit logic.
        """
        if self.premarket_dynamic_limits_done:
            logger.info("PREMARKET LIMITS: already completed; skipping duplicate call")
            return

        self.position_mgr.reconcile_local_positions_from_broker()
        symbols = list(self.position_mgr.positions.keys())
        if not symbols:
            logger.info("PREMARKET LIMITS: no overnight positions to classify")
            return

        broker_positions = self.position_mgr.get_broker_positions()
        if broker_positions is None:
            logger.error("PREMARKET LIMITS: broker position read failed; skipping limit placement")
            return
        broker_by_symbol = {
            str(p.get("symbol", "")).upper(): p
            for p in broker_positions
            if p.get("symbol")
        }

        # Determine decision time
        start_time = _parse_config_time(getattr(config, "PREMARKET_DYNAMIC_START_TIME", "05:00"))
        final_time = _parse_config_time(getattr(config, "PREMARKET_DYNAMIC_FINAL_TIME", "06:00"))
        interval_min = getattr(config, "PREMARKET_DYNAMIC_CHECK_INTERVAL_MINUTES", 15)

        current_time = datetime.now(_ET).time()
        decision_time_str = decision_time_str or current_time.strftime("%H:%M")
        decision_time_dt = _parse_config_time(decision_time_str)

        if decision_time_str is None:
            # Auto-detect which checkpoint we're at
            minutes_since_start = (current_time.hour * 60 + current_time.minute) - (start_time.hour * 60 + start_time.minute)
            checkpoint_num = minutes_since_start // interval_min
            decision_time_dt = dt_time(start_time.hour + checkpoint_num // 60, start_time.minute + checkpoint_num % 60)
            decision_time_str = decision_time_dt.strftime("%H:%M")

        decision_dt = datetime.combine(datetime.now(_ET).date(), decision_time_dt, tzinfo=_ET)
        if datetime.now(_ET) < decision_dt:
            decision_dt = datetime.now(_ET)

        final_time_str = final_time.strftime("%H:%M")

        placed = 0
        no_cap = 0
        skipped = 0
        waited = 0

        for symbol in symbols:
            # Skip symbols already decided in previous checkpoints
            if symbol in self.premarket_decided_symbols:
                continue

            pos = self.position_mgr.positions.get(symbol)
            broker_pos = broker_by_symbol.get(symbol.upper())
            if not pos or not broker_pos:
                skipped += 1
                logger.warning(f"PREMARKET LIMIT {symbol}: missing local/broker position; skipping")
                continue

            try:
                qty = min(int(pos.quantity), abs(int(float(broker_pos.get("qty", 0)))))
            except (TypeError, ValueError):
                qty = int(pos.quantity or 0)
            if qty <= 0:
                skipped += 1
                logger.warning(f"PREMARKET LIMIT {symbol}: qty <= 0; skipping")
                continue

            entry_price = float(getattr(pos, "entry_price", 0.0) or broker_pos.get("avg_entry_price", 0.0) or 0.0)
            if entry_price <= 0:
                skipped += 1
                logger.warning(f"PREMARKET LIMIT {symbol}: missing entry price; skipping")
                continue

            sleeve = str(getattr(pos, "sleeve", "UNKNOWN") or "UNKNOWN").upper()

            metrics = self._compute_live_premarket_metrics(symbol, entry_price, decision_dt)

            # Check if signal is decisive (act now) or should wait
            is_decisive, decisive_reason = self._is_decisive_premarket_signal(
                decision_time=decision_time_str,
                final_time=final_time_str,
                current_return=metrics.get("current_return", 0.0),
                distance_from_high=metrics.get("distance_from_high", 0.0),
                trend_from_first_bar=metrics.get("trend_from_first_bar", 0.0),
                minutes_traded=int(metrics.get("premarket_minutes", 0) or 0),
                last_bar_age_minutes=float(metrics.get("last_bar_age_minutes", 999) or 999),
                sleeve=sleeve,
            )

            if not is_decisive:
                waited += 1
                logger.info(
                    f"PREMARKET LIMIT {symbol}: NOT DECISIVE at {decision_time_str} | "
                    f"reason={decisive_reason}, ret={metrics.get('current_return', 0.0):+.2%}, "
                    f"will check again in 15 minutes"
                )
                continue

            # Symbol is decisive - classify and act
            decision = self._classify_iex_premarket_limit(pos, metrics)

            log_metrics = (
                f"bars={metrics.get('premarket_minutes', 0)}, "
                f"ret={metrics.get('current_return', 0.0):+.2%}, "
                f"dist_high={metrics.get('distance_from_high', 0.0):+.2%}, "
                f"trend={metrics.get('trend_from_first_bar', 0.0):+.2%}, "
                f"stale={metrics.get('last_bar_age_minutes', 999):.0f}m"
            )

            # Mark as decided so we don't reclassify in future checkpoints
            self.premarket_decided_symbols.add(symbol)

            if decision["action"] == "NO_CAP":
                no_cap += 1
                logger.warning(
                    f"PREMARKET LIMIT {symbol} [{decision_time_str}]: NO CAP | entry={entry_price:.4f}, "
                    f"{log_metrics}, decisive={decisive_reason}, action={decision['reason']}"
                )
                continue

            limit_pct = decision.get("limit_pct")
            if limit_pct is None:
                no_cap += 1
                logger.warning(
                    f"PREMARKET LIMIT {symbol} [{decision_time_str}]: NO LIMIT | entry={entry_price:.4f}, "
                    f"{log_metrics}, decisive={decisive_reason}, action={decision['reason']}"
                )
                continue

            limit_price = self.position_mgr.round_limit_price(entry_price * (1.0 + float(limit_pct)))
            resp = self.position_mgr._submit_sell_order(
                symbol=symbol,
                qty=qty,
                order_type="limit",
                limit_price=limit_price,
                time_in_force=getattr(config, "PREMARKET_DYNAMIC_LIMIT_TIME_IN_FORCE", "day"),
                extended_hours=getattr(config, "PREMARKET_DYNAMIC_LIMIT_EXTENDED_HOURS", True),
            )
            if resp and resp.get("id"):
                self.premarket_limit_order_ids[symbol] = resp["id"]
                placed += 1
                logger.info(
                    f"PREMARKET LIMIT {symbol} [{decision_time_str}]: placed qty={qty}, entry={entry_price:.4f}, "
                    f"limit_pct={limit_pct:.2%}, limit={limit_price:.4f}, {log_metrics}, "
                    f"decisive={decisive_reason}, action={decision['reason']}, order_id={resp['id']}"
                )
            else:
                skipped += 1
                logger.error(f"PREMARKET LIMIT {symbol} [{decision_time_str}]: submit failed | {log_metrics}, decisive={decisive_reason}, action={decision['reason']}")

        # Check if we're at the final checkpoint
        is_final = decision_time_str >= final_time_str

        logger.warning(
            f"PREMARKET LIMITS [{decision_time_str}] COMPLETE: placed={placed}, no_cap={no_cap}, "
            f"skipped={skipped}, waited={waited}, final_checkpoint={is_final}"
        )

        # If final checkpoint, mark as done
        if is_final:
            self.premarket_dynamic_limits_done = True
            self._save_state()
        else:
            # Save state after each checkpoint to track decided symbols
            self._save_state()

    def _place_overnight_limit_sells(self):
        """20:00: place resting limit sells for all overnight positions.

        Limit formula:
            base = max(today_open_price, entry_price * (1 + target_gain_pct))
            if current_price > base: limit = current_price * (1 + current_price_premium_pct)
        The 09:25 cleanup cancels these orders before the normal 09:30 exit path.
        """
        if self.overnight_limit_sells_done:
            logger.info("OVERNIGHT LIMITS: already placed; skipping duplicate call")
            return

        self.position_mgr.reconcile_local_positions_from_broker()
        symbols = list(self.position_mgr.positions.keys())
        if not symbols:
            logger.info("OVERNIGHT LIMITS: no positions to place limits for")
            return

        broker_positions = self.position_mgr.get_broker_positions()
        if broker_positions is None:
            logger.error("OVERNIGHT LIMITS: broker position read failed; skipping limit placement")
            return
        broker_by_symbol = {
            str(p.get("symbol", "")).upper(): p
            for p in broker_positions
            if p.get("symbol")
        }

        snapshots = self.alpaca.get_snapshots(symbols)
        placed = 0
        skipped = 0

        target_gain = getattr(config, "OVERNIGHT_LIMIT_TARGET_GAIN_PCT", 0.025)
        current_premium = getattr(config, "OVERNIGHT_LIMIT_CURRENT_PRICE_PREMIUM_PCT", 0.005)
        tif = getattr(config, "OVERNIGHT_LIMIT_TIME_IN_FORCE", "gtc")
        extended = getattr(config, "OVERNIGHT_LIMIT_EXTENDED_HOURS", False)

        for symbol in symbols:
            pos = self.position_mgr.positions.get(symbol)
            broker_pos = broker_by_symbol.get(symbol.upper())
            if not pos or not broker_pos:
                skipped += 1
                logger.warning(f"OVERNIGHT LIMIT {symbol}: missing local/broker position; skipping")
                continue

            try:
                qty = min(int(pos.quantity), abs(int(float(broker_pos.get("qty", 0)))))
            except (TypeError, ValueError):
                qty = int(pos.quantity or 0)
            if qty <= 0:
                skipped += 1
                logger.warning(f"OVERNIGHT LIMIT {symbol}: qty <= 0; skipping")
                continue

            snap = snapshots.get(symbol, {}) if snapshots else {}
            today_open = snap.get("open") or 0.0
            current_price = snap.get("last_price") or snap.get("close") or 0.0
            try:
                today_open = float(today_open or 0.0)
                current_price = float(current_price or 0.0)
            except (TypeError, ValueError):
                today_open = 0.0
                current_price = 0.0

            entry_price = float(getattr(pos, "entry_price", 0.0) or broker_pos.get("avg_entry_price", 0.0) or 0.0)
            if entry_price <= 0:
                skipped += 1
                logger.warning(f"OVERNIGHT LIMIT {symbol}: missing entry price; skipping")
                continue

            base_limit = entry_price * (1.0 + target_gain)
            if today_open > 0:
                base_limit = max(base_limit, today_open)

            limit_price = base_limit
            if current_price > base_limit:
                limit_price = current_price * (1.0 + current_premium)

            limit_price = self.position_mgr.round_limit_price(limit_price)
            resp = self.position_mgr._submit_sell_order(
                symbol=symbol,
                qty=qty,
                order_type="limit",
                limit_price=limit_price,
                time_in_force=tif,
                extended_hours=extended,
            )
            if resp and resp.get("id"):
                self.overnight_limit_order_ids[symbol] = resp["id"]
                placed += 1
                logger.info(
                    f"OVERNIGHT LIMIT {symbol}: qty={qty}, entry={entry_price:.4f}, "
                    f"open={today_open:.4f}, current={current_price:.4f}, limit={limit_price:.4f}, "
                    f"tif={tif}, order_id={resp['id']}"
                )
            else:
                skipped += 1
                logger.error(f"OVERNIGHT LIMIT {symbol}: submit failed")

        logger.warning(f"OVERNIGHT LIMITS COMPLETE: placed={placed}, skipped={skipped}")

    def _save_end_of_day_reports(self):
        """Write all daily diagnostic artifacts. Called on EVERY completed market day."""
        try:
            stats = self._exec_stats
            total_candidates = len(self.mr_candidates) + len(self.gdp_candidates)
            save_run_health(
                diag=self._universe_diag,
                scored_count=total_candidates,
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
            def _mr_dict(c):
                return {
                    "symbol": c.symbol,
                    "sleeve": "MR",
                    "selection_score": round(c.selection_score, 4),
                    "signal_price": round(c.signal_price, 4),
                    "day_return": round(c.day_return, 4),
                    "volume_ratio": round(c.volume_ratio, 2),
                    "close_position": round(c.close_position, 3),
                    "late_drop_1530_1550": round(c.late_drop_1530_1550, 4),
                    "adv_dollars": round(c.adv_dollars, 0),
                }
            def _gdp_dict(c):
                return {
                    "symbol": c.symbol,
                    "sleeve": "GDP",
                    "selection_score": round(c.selection_score, 4),
                    "signal_price": round(c.signal_price, 4),
                    "day_return": round(c.day_return, 4),
                    "price_vs_vwap": round(c.price_vs_vwap, 4),
                    "late_mom_1530_signal": round(c.late_mom_1530_signal, 4),
                    "volume_ratio": round(c.volume_ratio, 2),
                    "close_position": round(c.close_position, 3),
                    "adv_dollars": round(c.adv_dollars, 0),
                }
            audit_dicts = {
                "mr_selected": [_mr_dict(c) for c in self.mr_candidates[:config.MR_MAX_POSITIONS]],
                "mr_all_passed": [_mr_dict(c) for c in self.mr_candidates],
                "gdp_selected": [_gdp_dict(c) for c in self.gdp_candidates[:config.GDP_MAX_POSITIONS]],
                "gdp_all_passed": [_gdp_dict(c) for c in self.gdp_candidates],
            }
            if audit_dicts["mr_selected"] or audit_dicts["gdp_selected"]:
                save_candidates_audit(audit_dicts)
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
                "date": datetime.now(_ET).strftime("%Y-%m-%d"),
                "morning_exits_done": self.morning_exits_done,
                "gdp_exits_done": self.gdp_exits_done,
                "mr_exits_done": self.mr_exits_done,
                "red_trail_exit_submitted": self.red_trail_exit_submitted,
                "red_trail_order_ids": self.red_trail_order_ids,
                "red_trail_symbols": list(self.red_trail_symbols),
                "morning_open_orders_cancelled": self.morning_open_orders_cancelled,
                "premarket_dynamic_limits_done": self.premarket_dynamic_limits_done,
                "premarket_limit_order_ids": self.premarket_limit_order_ids,
                "premarket_decided_symbols": list(self.premarket_decided_symbols),
                # Legacy 20:00 fields kept for backward compatibility
                "overnight_limit_sells_done": self.overnight_limit_sells_done,
                "overnight_limit_order_ids": self.overnight_limit_order_ids,
                "end_of_day_reports_done": self.end_of_day_reports_done,
                "post_exit_failsafe_done": self.post_exit_failsafe_done,
                "data_collected": self.data_collected,
                "scoring_done": self.scoring_done,
                "entries_done": self.entries_done,
                "sold_today": list(self.sold_today),
            }
            self.state_mgr.save_bot_state(bot_state)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def _load_state(self):
        """Load state from previous run (same-day recovery only)."""
        today = datetime.now(_ET).strftime("%Y-%m-%d")
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
        # Handle backward compatibility: old v2_classified flag maps to both new flags
        if "v2_classified" in bot_state and "gdp_exits_done" not in bot_state:
            v2_done = bot_state.get("v2_classified", False)
            self.gdp_exits_done = v2_done
            self.mr_exits_done = v2_done
        else:
            self.gdp_exits_done = bot_state.get("gdp_exits_done", False)
            self.mr_exits_done = bot_state.get("mr_exits_done", False)
        self.post_exit_failsafe_done = bot_state.get("post_exit_failsafe_done", False)
        self.data_collected = bot_state.get("data_collected", False)
        self.scoring_done = bot_state.get("scoring_done", False)
        self.entries_done = bot_state.get("entries_done", False)
        self.sold_today = set(bot_state.get("sold_today", []))
        self.red_trail_exit_submitted = bot_state.get("red_trail_exit_submitted", False)
        self.red_trail_order_ids = bot_state.get("red_trail_order_ids", {})
        self.red_trail_symbols = set(bot_state.get("red_trail_symbols", []))
        self.morning_open_orders_cancelled = bot_state.get("morning_open_orders_cancelled", False)
        # New premarket fields
        self.premarket_dynamic_limits_done = bot_state.get("premarket_dynamic_limits_done", False)
        self.premarket_limit_order_ids = bot_state.get("premarket_limit_order_ids", {})
        self.premarket_decided_symbols = set(bot_state.get("premarket_decided_symbols", []))
        # Legacy 20:00 fields kept for backward compatibility
        self.overnight_limit_sells_done = bot_state.get("overnight_limit_sells_done", False)
        self.overnight_limit_order_ids = bot_state.get("overnight_limit_order_ids", {})
        self.end_of_day_reports_done = bot_state.get("end_of_day_reports_done", False)

        # Load positions
        saved = self.state_mgr.load_positions()
        if saved:
            self.position_mgr.load_positions(saved)
            logger.info(f"Loaded {len(saved)} saved positions")


def main():
    try:
        bot = CombinedOvernightReboundBot()
    except Exception:
        logging.critical("UNHANDLED EXCEPTION during bot initialisation", exc_info=True)
        raise
    bot.run()


if __name__ == "__main__":
    main()
