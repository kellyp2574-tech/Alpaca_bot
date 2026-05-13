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
  06:00  Final premarket classification for all unresolved symbols (runs once,
         within the 06:00–06:02 cutoff window)
  09:25  Cancel any remaining premarket limits, freeze broker exit plan
  09:30  Submit batched market sells against the frozen plan
         (ENABLE_FAST_OPEN_MARKET_EXIT=True; the alternative red-trail mode
         is mutually exclusive and currently disabled)
  09:45  V2 failsafe — verify broker is flat or force-flatten any stragglers

AFTERNOON (T-1 entries — new positions for tomorrow's exits):
  15:30  Build universe (Massive + Alpaca, $1–10, ADV sizing cap protects)
  15:50  Fetch latest 9:30-15:50 minute bars, build both MR and GDP candidates
  15:50  Daily-loss circuit breaker check, then execute entries
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


# Hot windows where the main loop should tick at 1s for prompt action.
# Outside these windows the loop sleeps up to 30s to save CPU and rate-limit
# budget. Each entry is (HH, MM_start, HH, MM_end) in ET.
_HOT_WINDOWS_HHMM = (
    (5, 0, 6, 2),     # Premarket dynamic-limit checkpoints (incl. final 06:00 trip)
    (9, 24, 10, 5),   # Order cancel, 09:30 batch sells, fills, failsafe
    (15, 29, 16, 1),  # Universe build, scoring, entries, EOD
)


def _is_hot_window(now_t: dt_time) -> bool:
    """True if ``now_t`` falls inside any pre-defined hot window."""
    cur = now_t.hour * 60 + now_t.minute
    for sh, sm, eh, em in _HOT_WINDOWS_HHMM:
        if (sh * 60 + sm) <= cur < (eh * 60 + em):
            return True
    return False


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
        self.open_exit_plan: List[Dict[str, Any]] = []  # Frozen broker-position sell plan built after 09:25 cleanup

        # Morning/overnight order management
        self.morning_open_orders_cancelled = False
        self.premarket_dynamic_limits_done = False
        self.premarket_limit_order_ids: Dict[str, str] = {}
        self.premarket_decided_symbols: set = set()  # Track symbols already decided in rolling checks
        self.premarket_checkpoints_done: set = set()  # Per-checkpoint completion guard (HH:MM strings)
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
        finally:
            # Always tear down the trade_updates websocket cleanly so the
            # daemon thread doesn't keep the process alive on shutdown.
            try:
                stream = getattr(self.position_mgr, "fill_stream", None)
                if stream is not None:
                    stream.stop(timeout=2.0)
            except Exception:
                logger.warning("FillStream stop failed on shutdown", exc_info=True)

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
            if getattr(config, "ENABLE_FAST_OPEN_MARKET_EXIT", False):
                raise ValueError(
                    "ENABLE_FAST_OPEN_MARKET_EXIT and ENABLE_RED_OPEN_TRAIL_EXIT are "
                    "mutually exclusive — fast-exit wins at 09:30 but red-trail still "
                    "forces the 10:00 failsafe schedule. Set exactly one to True."
                )
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
                # Rolling premarket dynamic limit classification (05:00 → 06:00).
                # At each 15-minute checkpoint we classify only "decisive"
                # symbols; unclear ones wait for the next checkpoint. The
                # final 06:00 checkpoint must run too — that is when any
                # remaining unresolved symbol gets a normal harvest limit.
                #
                # We run this block while ``current_time < t_premarket_cutoff``
                # (06:02) so the 06:00 minute itself is included exactly once.
                # Any checkpoint slot computed past 06:00 is clamped down to
                # the 06:00 string so the dedup set still works correctly.
                t_premarket_cutoff = dt_time(t_premarket_final.hour, t_premarket_final.minute + 2)
                if (getattr(config, "ENABLE_PREMARKET_DYNAMIC_LIMIT_SELLS", False)
                        and not self.premarket_dynamic_limits_done
                        and current_time >= t_premarket_start
                        and current_time < t_premarket_cutoff):
                    minutes_since_start = (current_time.hour * 60 + current_time.minute) - (t_premarket_start.hour * 60 + t_premarket_start.minute)
                    checkpoint_num = minutes_since_start // t_premarket_interval
                    checkpoint_minutes = (t_premarket_start.hour * 60 + t_premarket_start.minute) + checkpoint_num * t_premarket_interval

                    # Clamp anything beyond the configured final to the final
                    # checkpoint so 06:01 / 06:02 ticks still trigger the
                    # canonical "06:00" run rather than inventing a new slot.
                    final_minutes = t_premarket_final.hour * 60 + t_premarket_final.minute
                    checkpoint_minutes = min(checkpoint_minutes, final_minutes)
                    checkpoint_time = dt_time(checkpoint_minutes // 60, checkpoint_minutes % 60)
                    checkpoint_str = checkpoint_time.strftime("%H:%M")

                    # Per-checkpoint dedup: each HH:MM runs at most once per session.
                    if checkpoint_str not in self.premarket_checkpoints_done:
                        is_final = (checkpoint_minutes >= final_minutes)
                        logger.info(
                            f"PREMARKET CHECKPOINT: {checkpoint_str} - running dynamic limit classification"
                            f"{' (FINAL)' if is_final else ''}"
                        )
                        self._place_premarket_dynamic_limit_sells(decision_time_str=checkpoint_str)
                        self.premarket_checkpoints_done.add(checkpoint_str)
                        # Once the final 06:00 slot has run, mark the whole
                        # premarket window done so we don't keep re-entering.
                        if is_final:
                            self.premarket_dynamic_limits_done = True
                            self._save_state()

                # Bot started after the final-checkpoint cutoff: skip premarket entirely.
                if (getattr(config, "ENABLE_PREMARKET_DYNAMIC_LIMIT_SELLS", False)
                        and not self.premarket_dynamic_limits_done
                        and current_time >= t_premarket_cutoff
                        and current_time < t_cancel_orders):
                    logger.info(
                        f"Bot started after {t_premarket_cutoff.strftime('%H:%M')} cutoff "
                        f"- skipping premarket limits, proceeding to morning exits"
                    )
                    self.premarket_dynamic_limits_done = True
                    self._save_state()

                # 09:25 — cancel any resting premarket limit/trailing orders before normal exits.
                # Run regardless of local position state for safety.
                if (not self.morning_open_orders_cancelled
                        and current_time >= t_cancel_orders):
                    logger.warning("09:25 order cleanup: canceling all open orders before 09:30 exits")
                    self.position_mgr.cancel_all_open_orders()
                    self.premarket_limit_order_ids.clear()
                    self.red_trail_order_ids.clear()
                    self.red_trail_symbols.clear()
                    self.morning_open_orders_cancelled = True
                    self._build_open_exit_plan_from_broker(reason="09:25 post-cancel broker snapshot")
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
                    # 09:30 — fast open liquidation. Use the frozen 09:25 broker-position
                    # plan and submit every market sell before monitoring fills.
                    if not self.gdp_exits_done and current_time >= t_exit_all:
                        if getattr(config, "ENABLE_FAST_OPEN_MARKET_EXIT", True):
                            self._submit_open_exit_market_sells()
                        elif getattr(config, "ENABLE_RED_OPEN_TRAIL_EXIT", False):
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

            # Adaptive sleep: 1s during hot windows (open, close, premarket
            # checkpoints) so transitions are prompt; 30s otherwise so the bot
            # spends ~1500 ticks/day instead of ~36000 — drastically lower CPU
            # and shared rate-limit pressure.
            time.sleep(1 if _is_hot_window(current_time) else 30)

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

    def _build_open_exit_plan_from_broker(self, reason: str = "broker snapshot") -> List[Dict[str, Any]]:
        """Freeze the remaining broker positions into a simple 09:30 market-sell plan.

        This is intentionally called before the open, after canceling premarket
        limit orders. The 09:30 path should not fetch prices or make
        green/red decisions; it should only submit these prepared market sells.
        """
        broker_positions = self.position_mgr.get_broker_positions()
        if broker_positions is None:
            logger.error("OPEN_EXIT_PLAN: broker position read failed during %s", reason)
            self.open_exit_plan = []
            return self.open_exit_plan

        plan: List[Dict[str, Any]] = []
        for broker_pos in broker_positions:
            symbol = str(broker_pos.get("symbol", "")).upper().strip()
            if not symbol:
                continue
            try:
                qty = int(abs(float(broker_pos.get("qty", 0))))
            except (TypeError, ValueError):
                logger.warning("OPEN_EXIT_PLAN: bad qty for %s: %s", symbol, broker_pos.get("qty"))
                continue
            if qty <= 0:
                continue

            plan.append({
                "symbol": symbol,
                "qty": qty,
                "avg_entry_price": broker_pos.get("avg_entry_price"),
                "source": reason,
            })

        self.open_exit_plan = plan
        logger.warning(
            "OPEN_EXIT_PLAN_READY reason=%s count=%d symbols=%s",
            reason,
            len(plan),
            [p["symbol"] for p in plan],
        )
        return plan

    def _submit_open_exit_market_sells(self):
        """09:30 fast liquidation: submit all market sells first, then reconcile.

        No snapshots. No green/red branch. No trailing stops. No fill-wait inside
        the submit loop. This keeps the 09:30 minute focused purely on order
        submission.
        """
        if not self.open_exit_plan:
            logger.warning("OPEN_EXIT: no frozen 09:25 plan found; building one from broker now")
            self._build_open_exit_plan_from_broker(reason="09:30 fallback broker snapshot")

        if not self.open_exit_plan:
            logger.warning("OPEN_EXIT: no broker positions to submit")
            self.position_mgr.reconcile_local_positions_from_broker()
            return

        logger.warning(
            "OPEN_EXIT_BATCH_START count=%d symbols=%s",
            len(self.open_exit_plan),
            [p["symbol"] for p in self.open_exit_plan],
        )

        submitted_orders: List[Tuple[str, str, int]] = []

        # Phase 1: submit every market sell as quickly as possible.
        for item in list(self.open_exit_plan):
            symbol = item.get("symbol")
            qty = int(item.get("qty", 0) or 0)
            if not symbol or qty <= 0:
                continue

            submit_start = datetime.now(_ET)
            t0 = time.perf_counter()
            try:
                resp = self.position_mgr._submit_sell_order(
                    symbol,
                    qty,
                    order_type="market",
                    time_in_force="day",
                    extended_hours=False,
                )
            except TypeError:
                # Backward compatibility with older PositionManager signatures.
                resp = self.position_mgr._submit_sell_order(symbol, qty)
            except Exception:
                logger.exception("OPEN_EXIT_SUBMIT_FAILED symbol=%s qty=%s", symbol, qty)
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            order_id = resp.get("id") if resp else None
            if order_id:
                submitted_orders.append((order_id, symbol, qty))
                self.sold_today.add(symbol)
                logger.warning(
                    "OPEN_EXIT_SUBMITTED symbol=%s qty=%s order_id=%s start=%s elapsed_ms=%.1f type=market tif=day ext=false",
                    symbol,
                    qty,
                    order_id,
                    submit_start.isoformat(),
                    elapsed_ms,
                )
            else:
                logger.error("OPEN_EXIT_SUBMIT_NO_ORDER_ID symbol=%s qty=%s resp=%s", symbol, qty, resp)

        logger.warning(
            "OPEN_EXIT_BATCH_ALL_SUBMITTED submitted=%d planned=%d",
            len(submitted_orders),
            len(self.open_exit_plan),
        )

        # Phase 2: do not block the opening submissions. Reconcile once after all
        # orders are out; the normal failsafe can catch anything still held.
        actions = self.position_mgr.reconcile_local_positions_from_broker()
        if actions:
            logger.info("OPEN_EXIT: post-submit reconciliation adjustments: %s", actions)

        remaining = self.position_mgr.broker_position_count()
        if remaining == 0:
            logger.warning("OPEN_EXIT: broker confirmed flat after batch submit")
            self.position_mgr.positions.clear()
            self.open_exit_plan = []
        elif remaining > 0:
            logger.warning("OPEN_EXIT: broker still shows %d positions after batch submit; failsafe will retry", remaining)
        else:
            logger.error("OPEN_EXIT: broker position count unavailable after batch submit")

    # ════════════════════════════════════════════════════════════
    # AFTERNOON DATA & SCORING METHODS
    # ════════════════════════════════════════════════════════════

    def _step_collect_data(self):
        """~3:30 PM: Build universe (price/ADV/daily-bar filters). Stage C minute-quality runs at 3:50."""
        logger.info("=" * 50)
        logger.info("DATA COLLECTION: Building base universe (staged pipeline)")
        logger.info("=" * 50)

        try:
            final, diag, adv_cache = build_universe(
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
            # Single account fetch — was previously 2 calls here plus 1 per
            # allocation in _adaptive_qty (~22 account calls per entry pass).
            account = self.position_mgr.get_account()
            if not account:
                logger.error("Cannot fetch account — skipping entries")
                self.entries_done = True
                return
            try:
                equity = float(account.get("equity") or 0.0)
                buying_power = float(account.get("buying_power") or 0.0)
            except (TypeError, ValueError):
                equity = 0.0
                buying_power = 0.0
            if equity <= 0:
                logger.error("Cannot determine account equity — skipping entries")
                self.entries_done = True
                return
            if buying_power <= 0:
                logger.warning("Cannot determine buying power — falling back to equity")
                buying_power = equity

            # Daily loss circuit breaker — abort entries if today's PnL is worse
            # than DAILY_LOSS_LIMIT_PCT. account['last_equity'] is yesterday's
            # market-close equity, so today's drawdown is a clean comparison.
            loss_limit = float(getattr(config, "DAILY_LOSS_LIMIT_PCT", 0.0) or 0.0)
            if loss_limit > 0:
                try:
                    last_equity = float(account.get("last_equity") or 0.0)
                except (TypeError, ValueError):
                    last_equity = 0.0
                if last_equity > 0:
                    day_ret = (equity - last_equity) / last_equity
                    if day_ret <= -loss_limit:
                        logger.critical(
                            f"DAILY LOSS CIRCUIT BREAKER TRIPPED — equity ${equity:,.2f} "
                            f"vs last_equity ${last_equity:,.2f} = {day_ret:+.2%}; "
                            f"limit -{loss_limit:.0%}. SKIPPING all entries today."
                        )
                        self.entries_done = True
                        return
                    logger.info(
                        f"Daily PnL check OK: {day_ret:+.2%} (limit -{loss_limit:.0%})"
                    )
                else:
                    logger.warning(
                        "Daily loss check skipped — last_equity unavailable from account API"
                    )

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

            # Calculate leftover from sleeve allocations
            mr_allocated = sum(r["target_dollars"] for r in mr_results)
            gdp_allocated = sum(r["target_dollars"] for r in gdp_results)
            mr_leftover = mr_budget - mr_allocated
            gdp_leftover = gdp_budget - gdp_allocated
            total_leftover = mr_leftover + gdp_leftover

            # Log sleeve allocation results
            logger.info(
                f"MR sleeve: budget=${mr_budget:,.2f}, allocated=${mr_allocated:,.2f}, leftover=${mr_leftover:,.2f}, "
                f"positions={len(mr_results)}"
            )
            logger.info(
                f"GDP sleeve: budget=${gdp_budget:,.2f}, allocated=${gdp_allocated:,.2f}, leftover=${gdp_leftover:,.2f}, "
                f"positions={len(gdp_results)}"
            )

            # Global leftover redeployment pass
            # Fallback order: MR leftover → GDP candidates → all remaining orderable candidates
            if total_leftover > config.MIN_POSITION_DOLLARS:
                # Build set of already allocated symbols
                allocated_symbols = {r["symbol"] for r in mr_results + gdp_results}

                # Fallback 1: MR leftover → GDP candidates
                if mr_leftover > config.MIN_POSITION_DOLLARS and gdp_orderable:
                    gdp_unallocated = [c for c in gdp_orderable if c.symbol not in allocated_symbols]
                    if gdp_unallocated:
                        gdp_fallback = allocate_waterfall(
                            gdp_unallocated, mr_leftover, equity, "MR_FALLBACK_GDP", config.GDP_MAX_POSITIONS
                        )
                        if gdp_fallback:
                            gdp_fallback_allocated = sum(r["target_dollars"] for r in gdp_fallback)
                            logger.info(
                                f"MR fallback to GDP: budget=${mr_leftover:,.2f}, allocated=${gdp_fallback_allocated:,.2f}, "
                                f"positions={len(gdp_fallback)}"
                            )
                            # Add to GDP results with original sleeve label
                            for r in gdp_fallback:
                                r["fallback"] = True
                            gdp_results.extend(gdp_fallback)
                            allocated_symbols.update(r["symbol"] for r in gdp_fallback)
                            mr_leftover -= gdp_fallback_allocated
                            total_leftover -= gdp_fallback_allocated

                # Fallback 2: Remaining leftover → all remaining orderable candidates sorted by score
                if total_leftover > config.MIN_POSITION_DOLLARS:
                    # Combine all orderable candidates, remove already allocated
                    all_orderable = mr_orderable + gdp_orderable
                    overflow_pool = [c for c in all_orderable if c.symbol not in allocated_symbols]
                    # Sort by selection score (highest first)
                    overflow_pool.sort(key=lambda x: getattr(x, "selection_score", 0.0), reverse=True)
                    
                    if overflow_pool:
                        # Build symbol sets for sleeve detection
                        mr_orderable_symbols = {c.symbol for c in mr_orderable}
                        gdp_orderable_symbols = {c.symbol for c in gdp_orderable}
                        
                        # Use combined max positions for overflow
                        current_positions = len(mr_results) + len(gdp_results)
                        remaining_slots = max(0, config.COMBINED_MAX_POSITIONS - current_positions)
                        
                        if remaining_slots > 0:
                            overflow_fallback = allocate_waterfall(
                                overflow_pool, total_leftover, equity, "OVERFLOW", remaining_slots
                            )
                            if overflow_fallback:
                                overflow_allocated = sum(r["target_dollars"] for r in overflow_fallback)
                                logger.info(
                                    f"Overflow fallback: budget=${total_leftover:,.2f}, allocated=${overflow_allocated:,.2f}, "
                                    f"positions={len(overflow_fallback)}"
                                )
                                # Assign sleeve based on original candidate source using symbol membership
                                for r in overflow_fallback:
                                    r["fallback"] = True
                                    if r["candidate"].symbol in mr_orderable_symbols:
                                        mr_results.append(r)
                                    else:
                                        gdp_results.append(r)
                                total_leftover -= overflow_allocated

                # Log final leftover
                if total_leftover > config.MIN_POSITION_DOLLARS:
                    logger.warning(
                        f"Final leftover after fallback: ${total_leftover:,.2f} (no more orderable candidates)"
                    )

            # Build SleeveAllocation list with shares calculated from target dollars
            allocations: List[SleeveAllocation] = []
            for rank, r in enumerate(mr_results, start=1):
                c = r["candidate"]
                shares = math.floor(r["target_dollars"] / c.signal_price) if c.signal_price > 0 else 0
                sleeve_label = "MR" if not r.get("fallback") else "MR_FALLBACK"
                allocations.append(SleeveAllocation(
                    symbol=c.symbol,
                    shares=shares,
                    target_dollars=r["target_dollars"],
                    rank=rank,
                    sleeve=sleeve_label,
                    candidate=c
                ))

            for rank, r in enumerate(gdp_results, start=1):
                c = r["candidate"]
                shares = math.floor(r["target_dollars"] / c.signal_price) if c.signal_price > 0 else 0
                sleeve_label = "GDP" if not r.get("fallback") else "GDP_FALLBACK"
                allocations.append(SleeveAllocation(
                    symbol=c.symbol,
                    shares=shares,
                    target_dollars=r["target_dollars"],
                    rank=rank,
                    sleeve=sleeve_label,
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
                f"{sum(1 for a in allocations if a.sleeve.startswith('MR'))} MR + "
                f"{sum(1 for a in allocations if a.sleeve.startswith('GDP'))} GDP, "
                f"total_target=${total_target:,.2f}"
            )
            logger.info(
                f"Execution pool metrics: pool_size={len(candidate_symbols)}, "
                f"orderable={len(orderable_set)}, rejected_spread={len(exec_rejected)}"
            )
            # Submit market buy orders
            total_deployed = 0.0

            # Track buying power locally to avoid hitting /v2/account once per
            # symbol. Decremented after each submission by the planned notional;
            # reconciles with broker on submit failure (one fresh fetch then).
            bp_remaining = buying_power

            def _adaptive_qty(alloc: SleeveAllocation, bp_avail: float,
                              bp_buffer: float = config.ENTRY_BP_BUFFER_PCT) -> int:
                """Return shares clamped to current buying power using target_dollars."""
                if bp_avail <= 0:
                    return 0
                max_notional = bp_avail * bp_buffer
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
                qty = _adaptive_qty(alloc, bp_remaining)

                # Pre-submit logging
                planned_notional = qty * price_ref
                logger.info(
                    f"ENTRY PLANNED {symbol}: qty={qty}, price_ref={price_ref:.4f}, "
                    f"notional={planned_notional:,.2f}, bp_remaining={bp_remaining:,.2f}, "
                    f"sleeve={alloc.sleeve}, rank={alloc.rank}"
                )

                if qty < config.MIN_SHARES:
                    logger.warning(
                        f"ENTRY SKIP {symbol}: adaptive qty {qty} < {config.MIN_SHARES} min shares "
                        f"(bp=${bp_remaining:,.2f}, price={price_ref:.4f})"
                    )
                    exec_diag.failed_submissions[symbol] = "bp_resize_below_min"
                    continue

                buy_resp = self.position_mgr.submit_buy_order(symbol, qty)
                if not buy_resp:
                    # Submit failed — refresh BP from broker once and retry smaller.
                    fresh_bp = self.position_mgr.get_total_capital()
                    if fresh_bp and fresh_bp > 0 and price_ref > 0:
                        bp_remaining = fresh_bp  # resync local tracker
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

                # Decrement local BP tracker by the submitted notional.
                bp_remaining = max(0.0, bp_remaining - qty * price_ref)
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
            mr_filled = sum(
                1 for s in exec_diag.filled_symbols
                if exec_diag.fill_details.get(s, {}).get("sleeve", "").startswith("MR")
            )
            gdp_filled = sum(
                1 for s in exec_diag.filled_symbols
                if exec_diag.fill_details.get(s, {}).get("sleeve", "").startswith("GDP")
            )

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

    def _compute_live_premarket_metrics(self, symbol: str, buy_price: float, decision_dt: datetime, pre_fetched_snapshots: Optional[Dict[str, dict]] = None) -> Dict[str, Any]:
        """Compute premarket metrics with IEX bars + SIP snapshot backup.

        IEX bars remain the primary source for the premarket *shape*:
        first print, high/low, volume, distance from high, and trend.

        SIP snapshot data is used as a current-price cross-check even when IEX
        bars exist. This protects the 05:00-06:00 limit classifier from thin,
        stale, or understated IEX premarket prints. If there are no usable IEX
        bars, the same SIP snapshot path becomes the full fallback.
        """
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

        # Always try the SIP snapshot when enabled. If bars exist, this is a
        # cross-check/current-price correction. If bars are missing, this is the
        # full fallback.
        use_sip_backup = getattr(config, "USE_SIP_SNAPSHOT_PREMARKET_BACKUP", True)
        snapshot_metrics = {}
        if use_sip_backup:
            snapshot_metrics = self._compute_snapshot_metrics(symbol, buy_price, decision_dt, pre_fetched_snapshots=pre_fetched_snapshots)

        if normalized:
            first = normalized[0]
            last = normalized[-1]
            iex_stale_minutes = (decision_dt - last["dt"]).total_seconds() / 60.0
            iex_high = max(b["high"] for b in normalized)
            iex_low = min(b["low"] for b in normalized)
            premarket_volume = sum(b["volume"] for b in normalized)
            iex_current = last["close"]
            first_price = first["close"]

            resolved_current = iex_current
            resolved_source = "iex_only"
            resolved_stale_minutes = iex_stale_minutes
            effective_high = iex_high
            sip_current = None
            sip_spread_pct = None
            sip_stale_minutes = None
            sip_reason = snapshot_metrics.get("reason") if snapshot_metrics else None

            if snapshot_metrics.get("has_data"):
                sip_current = float(snapshot_metrics.get("current_price", 0.0) or 0.0)
                sip_spread_pct = snapshot_metrics.get("snapshot_spread_pct")
                sip_stale_minutes = float(snapshot_metrics.get("last_bar_age_minutes", 999) or 999)
                max_spread = getattr(config, "SIP_SNAPSHOT_MAX_SPREAD_PCT", 0.02)
                confirm_diff = getattr(config, "SIP_IEX_CONFIRM_DIFF_PCT", 0.0075)

                spread_ok = (
                    sip_spread_pct is None
                    or sip_spread_pct <= 0
                    or float(sip_spread_pct) <= max_spread
                )

                if sip_current > 0 and spread_ok and iex_current > 0:
                    diff_pct = abs(sip_current - iex_current) / iex_current
                    if diff_pct <= confirm_diff:
                        resolved_current = sip_current
                        resolved_source = "sip_confirmed"
                        resolved_stale_minutes = sip_stale_minutes
                    elif sip_current > iex_current:
                        resolved_current = sip_current
                        resolved_source = "sip_higher_than_iex"
                        resolved_stale_minutes = sip_stale_minutes
                    else:
                        # SIP lower than IEX: use the lower/conservative mark.
                        resolved_current = min(iex_current, sip_current)
                        resolved_source = "conservative_min_iex_sip"
                        resolved_stale_minutes = min(iex_stale_minutes, sip_stale_minutes)
                elif sip_current > 0 and not spread_ok:
                    resolved_source = "iex_only_sip_wide_spread"

                if (
                    getattr(config, "SIP_ALLOW_HIGH_CORRECTION", True)
                    and sip_current
                    and sip_current > effective_high
                    and (sip_spread_pct is None or sip_spread_pct <= 0 or sip_spread_pct <= getattr(config, "SIP_SNAPSHOT_MAX_SPREAD_PCT", 0.02))
                ):
                    effective_high = sip_current

            current_return = resolved_current / buy_price - 1.0 if buy_price > 0 else 0.0
            distance_from_high = resolved_current / effective_high - 1.0 if effective_high > 0 else 0.0
            return_from_low = resolved_current / iex_low - 1.0 if iex_low > 0 else 0.0
            trend_from_first_bar = resolved_current / first_price - 1.0 if first_price > 0 else 0.0

            logger.info(
                "PREMARKET PRICE RESOLVE %s: entry=%.4f iex_latest=%.4f sip_price=%s "
                "sip_spread=%s resolved=%.4f source=%s ret=%+.2f%% iex_high=%.4f effective_high=%.4f "
                "iex_stale=%.0fm resolved_stale=%.0fm sip_reason=%s",
                symbol,
                buy_price,
                iex_current,
                f"{sip_current:.4f}" if sip_current else "None",
                f"{sip_spread_pct:.2%}" if sip_spread_pct is not None else "None",
                resolved_current,
                resolved_source,
                current_return * 100.0,
                iex_high,
                effective_high,
                iex_stale_minutes,
                resolved_stale_minutes,
                sip_reason,
            )

            return {
                "has_data": True,
                "reason": "iex_premarket_data",
                "price_source": resolved_source,
                "first_premarket_time": first["dt"],
                "first_premarket_price": first_price,
                "current_time": last["dt"],
                "current_price": resolved_current,
                "iex_current_price": iex_current,
                "sip_current_price": sip_current,
                "sip_snapshot_reason": sip_reason,
                "premarket_high": effective_high,
                "iex_premarket_high": iex_high,
                "premarket_low": iex_low,
                "premarket_minutes": len(normalized),
                "premarket_volume": premarket_volume,
                "last_bar_age_minutes": resolved_stale_minutes,
                "iex_last_bar_age_minutes": iex_stale_minutes,
                "snapshot_spread_pct": sip_spread_pct,
                "current_return": current_return,
                "distance_from_high": distance_from_high,
                "return_from_low": return_from_low,
                "trend_from_first_bar": trend_from_first_bar,
            }

        # No bars - use snapshot fallback if available.
        if snapshot_metrics.get("has_data"):
            return snapshot_metrics

        # Both failed - return true no data with current_return=None to distinguish from zero return
        return {
            "has_data": False,
            "reason": "no_bars_and_no_snapshot",
            "premarket_minutes": 0,
            "current_return": None,
        }

    def _compute_snapshot_metrics(self, symbol: str, buy_price: float, decision_dt: datetime, pre_fetched_snapshots: Optional[Dict[str, dict]] = None) -> Dict[str, Any]:
        """Compute premarket metrics from Alpaca SIP snapshot/quote data.

        This method is intentionally reusable in two modes:
        1. Full fallback when IEX has no premarket bars.
        2. Current-price backup when IEX bars exist but may be stale/thin.

        Price priority:
        - Fresh NBBO midpoint when spread is sane.
        - Fresh latest trade when midpoint is unavailable/stale.
        Wide spreads reject the midpoint but do not automatically reject a fresh
        latest trade; the caller may still decide whether to use it.
        """
        try:
            # Use pre-fetched snapshots if provided (batch mode), otherwise fetch individually
            if pre_fetched_snapshots is not None:
                snapshots = pre_fetched_snapshots
            else:
                # Use SIP feed for snapshot backup when enabled
                use_sip_feed = getattr(config, "USE_SIP_SNAPSHOT_PREMARKET_BACKUP", True)
                feed = "sip" if use_sip_feed else None
                snapshots = self.alpaca.get_snapshots([symbol], feed=feed)
            
            if not snapshots or symbol not in snapshots:
                return {
                    "has_data": False,
                    "reason": "snapshot_not_available",
                    "premarket_minutes": 0,
                    "current_return": None,
                }

            snap = snapshots[symbol]
            if not snap:
                return {
                    "has_data": False,
                    "reason": "snapshot_empty",
                    "premarket_minutes": 0,
                    "current_return": None,
                }

            latest_trade = snap.get("latestTrade") or snap.get("last_trade") or snap.get("latest_trade") or {}
            latest_quote = snap.get("latestQuote") or snap.get("quote") or snap.get("latest_quote") or {}

            def _float_from(container: dict, *keys: str) -> float:
                for key in keys:
                    try:
                        val = container.get(key)
                        if val is not None:
                            return float(val)
                    except (TypeError, ValueError, AttributeError):
                        continue
                return 0.0

            # Support both raw Alpaca nested snapshots and parsed/flattened snapshots
            last_trade_price = (
                _float_from(latest_trade, "p", "price")
                or _float_from(snap, "last_price", "price", "current_price")
            )

            bid = (
                _float_from(latest_quote, "bp", "bid_price", "bid")
                or _float_from(snap, "bid")
            )

            ask = (
                _float_from(latest_quote, "ap", "ask_price", "ask")
                or _float_from(snap, "ask")
            )

            quote_timestamp = (
                latest_quote.get("t")
                or latest_quote.get("timestamp")
                if latest_quote else None
            )

            trade_timestamp = (
                latest_trade.get("t")
                or latest_trade.get("timestamp")
                if latest_trade else None
            )

            # Flattened parser stores the latest trade timestamp here
            if not trade_timestamp:
                trade_timestamp = snap.get("timestamp")

            def _parse_snap_time(raw) -> Optional[datetime]:
                if not raw:
                    return None
                try:
                    parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=_ET)
                    return parsed.astimezone(_ET)
                except Exception:
                    return None

            quote_time = _parse_snap_time(quote_timestamp)
            trade_time = _parse_snap_time(trade_timestamp)

            stale_max = getattr(config, "PREMARKET_DYNAMIC_MAX_STALE_MINUTES", 60)
            quote_stale_minutes = (decision_dt - quote_time).total_seconds() / 60.0 if quote_time else 999.0
            trade_stale_minutes = (decision_dt - trade_time).total_seconds() / 60.0 if trade_time else 999.0
            quote_fresh = quote_stale_minutes <= stale_max
            trade_fresh = trade_stale_minutes <= stale_max

            max_spread = getattr(config, "SIP_SNAPSHOT_MAX_SPREAD_PCT", 0.02)
            midpoint = 0.0
            spread_pct = None
            midpoint_usable = False
            if bid > 0 and ask > 0 and ask >= bid:
                midpoint = (bid + ask) / 2.0
                spread_pct = (ask - bid) / midpoint if midpoint > 0 else None
                midpoint_usable = quote_fresh and spread_pct is not None and spread_pct <= max_spread

            current_price = 0.0
            source = ""
            used_time = None
            stale_minutes = 999.0

            if midpoint_usable:
                current_price = midpoint
                source = "snapshot_mid"
                used_time = quote_time
                stale_minutes = quote_stale_minutes
            elif last_trade_price > 0 and trade_fresh:
                current_price = last_trade_price
                source = "snapshot_last"
                used_time = trade_time
                stale_minutes = trade_stale_minutes
            elif midpoint > 0 and quote_fresh:
                # Quote exists but spread is too wide. Do not use it for actual
                # pricing, but log why it was rejected.
                logger.warning(
                    "PREMARKET SNAPSHOT MID REJECTED: symbol=%s reason=wide_spread spread=%s bid=%.4f ask=%.4f max=%.2f%%",
                    symbol,
                    f"{spread_pct:.2%}" if spread_pct is not None else "None",
                    bid,
                    ask,
                    max_spread * 100.0,
                )

            if current_price <= 0:
                logger.warning(
                    "PREMARKET SNAPSHOT REJECTED: symbol=%s reason=no_fresh_usable_price "
                    "quote_fresh=%s trade_fresh=%s quote_stale=%.0fm trade_stale=%.0fm bid=%.4f ask=%.4f last=%.4f spread=%s",
                    symbol,
                    quote_fresh,
                    trade_fresh,
                    quote_stale_minutes,
                    trade_stale_minutes,
                    bid,
                    ask,
                    last_trade_price,
                    f"{spread_pct:.2%}" if spread_pct is not None else "None",
                )
                return {
                    "has_data": False,
                    "reason": "snapshot_stale_or_no_fresh_data",
                    "premarket_minutes": 0,
                    "snapshot_bid": bid,
                    "snapshot_ask": ask,
                    "snapshot_spread_pct": spread_pct,
                }

            current_return = current_price / buy_price - 1.0 if buy_price > 0 else 0.0

            logger.info(
                "PREMARKET SNAPSHOT PRICE %s: source=%s bid=%.4f ask=%.4f mid=%.4f last=%.4f "
                "used=%.4f entry=%.4f ret=%+.2f%% spread=%s stale=%.0fm",
                symbol,
                source,
                bid,
                ask,
                midpoint,
                last_trade_price,
                current_price,
                buy_price,
                current_return * 100.0,
                f"{spread_pct:.2%}" if spread_pct is not None else "None",
                stale_minutes,
            )

            return {
                "has_data": True,
                "reason": "snapshot_data",
                "price_source": source,
                "current_time": used_time or decision_dt,
                "current_price": current_price,
                "premarket_high": current_price,
                "premarket_low": current_price,
                "premarket_minutes": 0,
                "premarket_volume": 0,
                "last_bar_age_minutes": stale_minutes,
                "current_return": current_return,
                "distance_from_high": 0.0,
                "return_from_low": 0.0,
                "trend_from_first_bar": 0.0,
                "snapshot_bid": bid,
                "snapshot_ask": ask,
                "snapshot_mid": midpoint,
                "snapshot_last": last_trade_price,
                "snapshot_spread_pct": spread_pct,
                "snapshot_quote_stale_minutes": quote_stale_minutes,
                "snapshot_trade_stale_minutes": trade_stale_minutes,
            }
        except Exception:
            logger.warning(f"PREMARKET SNAPSHOT FALLBACK FAILED for {symbol}", exc_info=True)
            return {
                "has_data": False,
                "reason": "snapshot_not_available",
                "premarket_minutes": 0,
                "current_return": None,
            }

    def _classify_iex_premarket_limit(self, pos: Position, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Lenient IEX-aware dynamic limit decision.

        Absence of IEX prints is not treated as proof there is no market-wide
        premarket activity, but it does make it less likely that this is one of
        the true runners we are trying not to cap. Therefore the fallback is a
        normal 5% harvest limit rather than no decision.
        
        Now includes snapshot fallback when bars are unavailable. Snapshot data
        is treated as a single price point without bar count requirements.
        """
        fallback_limit = getattr(config, "PREMARKET_DYNAMIC_DEFAULT_LIMIT_PCT", 0.05)
        no_data_fallback_limit = getattr(config, "PREMARKET_DYNAMIC_NO_DATA_FALLBACK_LIMIT_PCT", 0.03)
        sparse_wide_limit = getattr(config, "PREMARKET_DYNAMIC_SPARSE_HIGH_RETURN_LIMIT_PCT", 0.10)
        very_high = getattr(config, "PREMARKET_DYNAMIC_VERY_HIGH_RETURN_NO_CAP_PCT", 0.10)
        high = getattr(config, "PREMARKET_DYNAMIC_HIGH_RETURN_NO_CAP_PCT", 0.05)
        moderate = getattr(config, "PREMARKET_DYNAMIC_MODERATE_RETURN_PCT", 0.02)
        stale_max = getattr(config, "PREMARKET_DYNAMIC_MAX_STALE_MINUTES", 60)

        if not metrics.get("has_data"):
            # Check if data is truly unavailable (current_return is None)
            current_return = metrics.get("current_return")
            if current_return is None:
                return {
                    "action": "NO_ACTION",
                    "limit_pct": None,
                    "reason": "data_unavailable",
                }
            # Data available but no bars/snapshot - use conservative fallback
            return {
                "action": "PLACE_LIMIT",
                "limit_pct": no_data_fallback_limit,
                "reason": metrics.get("reason", "no_bars_and_no_snapshot_default_3pct"),
            }

        bars = int(metrics.get("premarket_minutes", 0) or 0)
        stale = float(metrics.get("last_bar_age_minutes", 999) or 999)
        current_return = metrics.get("current_return")
        
        # If current_return is None, data is unavailable - should have been caught above
        if current_return is None:
            return {
                "action": "NO_ACTION",
                "limit_pct": None,
                "reason": "data_unavailable_in_classifier",
            }
        
        current_return = float(current_return) if current_return is not None else 0.0
        distance_from_high = float(metrics.get("distance_from_high", 0.0) or 0.0)
        trend = float(metrics.get("trend_from_first_bar", 0.0) or 0.0)
        sleeve = str(getattr(pos, "sleeve", "UNKNOWN") or "UNKNOWN").upper()
        fresh_enough = stale <= stale_max
        data_source = metrics.get("reason", "")
        
        # Snapshot data: treat as single price point without bar count requirements
        is_snapshot = data_source == "snapshot_data"
        
        if is_snapshot:
            # Snapshot-based decision: simpler logic based on current_return and staleness
            if current_return >= very_high and fresh_enough:
                return {"action": "NO_CAP", "limit_pct": None, "reason": "snapshot_very_high_return_no_cap"}
            elif current_return >= high:
                if sleeve == "MR" and current_return < very_high:
                    return {"action": "PLACE_LIMIT", "limit_pct": fallback_limit, "reason": "snapshot_high_return_mr_harvest_5pct"}
                return {"action": "NO_CAP", "limit_pct": None, "reason": "snapshot_high_return_no_cap"}
            elif current_return >= moderate:
                return {"action": "PLACE_LIMIT", "limit_pct": 0.06, "reason": "snapshot_moderate_return_6pct"}
            elif current_return >= 0:
                return {"action": "PLACE_LIMIT", "limit_pct": 0.04, "reason": "snapshot_small_winner_4pct"}
            else:
                return {"action": "PLACE_LIMIT", "limit_pct": 0.03, "reason": "snapshot_negative_pop_harvest_3pct"}

        # Bar-based decision: original logic with bar count requirements
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
        data_source: str = "",
    ) -> tuple[bool, str]:
        """
        Decisive = act now.
        Not decisive = leave unresolved and check again in 15 minutes.
        
        Snapshot data is treated specially: no bar count requirements, only freshness and return thresholds.
        Red/weak signals are decisive regardless of source to allow early lower-limit placement.
        """
        # Final checkpoint: no more waiting.
        if decision_time >= final_time:
            return True, "final_checkpoint"

        fresh = last_bar_age_minutes <= getattr(config, "PREMARKET_DYNAMIC_MAX_STALE_MINUTES", 60)
        is_snapshot = data_source == "snapshot_data"
        
        # Red/weak signal should be decisive even when source is IEX+SIP-resolved
        # This allows red names to place lower limits earlier instead of waiting until 06:00
        if fresh and current_return <= -0.01:
            return True, "decisive_red_lower_limit"
        
        # Snapshot data: decisive based on return thresholds without bar count requirements
        if is_snapshot and fresh:
            if current_return >= 0.10:
                return True, "decisive_snapshot_very_high_return"
            if current_return >= 0.05:
                return True, "decisive_snapshot_high_return"
            if sleeve.upper() == "MR" and current_return >= 0.03:
                return True, "decisive_snapshot_mr_harvest"

        # Bar-based decisive logic: requires bar count confirmation
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

        decision_dt = datetime.combine(datetime.now(_ET).date(), decision_time_dt, tzinfo=_ET)
        if datetime.now(_ET) < decision_dt:
            decision_dt = datetime.now(_ET)

        final_time_str = final_time.strftime("%H:%M")

        # Batch fetch all snapshots and bars before looping through symbols
        # This avoids making separate API calls per symbol and handles SIP fallback at the batch level
        use_sip_feed = getattr(config, "USE_SIP_SNAPSHOT_PREMARKET_BACKUP", True)
        feed = "sip" if use_sip_feed else None
        
        # Batch fetch snapshots with automatic SIP->IEX fallback
        all_snapshots = {}
        try:
            all_snapshots = self.alpaca.get_snapshots(symbols, feed=feed)
            logger.info(f"PREMARKET LIMITS: batch snapshot fetch returned {len(all_snapshots)} symbols (feed={feed})")
        except Exception as e:
            logger.warning(f"PREMARKET LIMITS: batch snapshot fetch failed: {e}")
        
        # Batch fetch bars (can't easily batch bars due to per-symbol API, so we'll keep per-symbol for bars)
        # But we've at least batched the snapshots which were the main 403 failure point
        
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

            metrics = self._compute_live_premarket_metrics(symbol, entry_price, decision_dt, pre_fetched_snapshots=all_snapshots)

            # Check if signal is decisive (act now) or should wait
            current_return = metrics.get("current_return")
            
            # If data is unavailable, skip this checkpoint
            if current_return is None:
                waited += 1
                logger.warning(
                    f"PREMARKET LIMIT {symbol}: DATA UNAVAILABLE at {decision_time_str} | "
                    f"source={metrics.get('reason')}, will check again in 15 minutes"
                )
                continue
            
            is_decisive, decisive_reason = self._is_decisive_premarket_signal(
                decision_time=decision_time_str,
                final_time=final_time_str,
                current_return=current_return,
                distance_from_high=metrics.get("distance_from_high", 0.0),
                trend_from_first_bar=metrics.get("trend_from_first_bar", 0.0),
                minutes_traded=int(metrics.get("premarket_minutes", 0) or 0),
                last_bar_age_minutes=float(metrics.get("last_bar_age_minutes", 999) or 999),
                sleeve=sleeve,
                data_source=metrics.get("reason", ""),
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
                f"source={metrics.get('reason')}, "
                f"bars={metrics.get('premarket_minutes', 0)}, "
                f"ret={metrics.get('current_return', 0.0):+.2%}, "
                f"dist_high={metrics.get('distance_from_high', 0.0):+.2%}, "
                f"trend={metrics.get('trend_from_first_bar', 0.0):+.2%}, "
                f"stale={metrics.get('last_bar_age_minutes', 999):.0f}m"
            )

            # Mark as decided so we don't reclassify in future checkpoints
            self.premarket_decided_symbols.add(symbol)

            if decision["action"] == "NO_ACTION":
                skipped += 1
                logger.error(
                    f"PREMARKET LIMIT {symbol} [{decision_time_str}]: NO ACTION (DATA UNAVAILABLE) | "
                    f"entry={entry_price:.4f}, {log_metrics}, decisive={decisive_reason}, action={decision['reason']}"
                )
                continue

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

        # If final checkpoint and data was unavailable for any symbols, log severe warning
        if is_final:
            data_unavailable_count = sum(
                1 for s in symbols
                if s not in self.premarket_decided_symbols
                and s not in self.premarket_limit_order_ids
            )
            if data_unavailable_count > 0:
                logger.error(
                    f"PREMARKET LIMITS FINAL CHECKPOINT: {data_unavailable_count} symbols had no data available "
                    f"and were not classified. These will be handled by regular morning exit logic."
                )

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
                "open_exit_plan": self.open_exit_plan,
                "morning_open_orders_cancelled": self.morning_open_orders_cancelled,
                "premarket_dynamic_limits_done": self.premarket_dynamic_limits_done,
                "premarket_limit_order_ids": self.premarket_limit_order_ids,
                "premarket_decided_symbols": list(self.premarket_decided_symbols),
                "premarket_checkpoints_done": list(self.premarket_checkpoints_done),
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
        self.open_exit_plan = bot_state.get("open_exit_plan", [])
        self.morning_open_orders_cancelled = bot_state.get("morning_open_orders_cancelled", False)
        # New premarket fields
        self.premarket_dynamic_limits_done = bot_state.get("premarket_dynamic_limits_done", False)
        self.premarket_limit_order_ids = bot_state.get("premarket_limit_order_ids", {})
        self.premarket_decided_symbols = set(bot_state.get("premarket_decided_symbols", []))
        self.premarket_checkpoints_done = set(bot_state.get("premarket_checkpoints_done", []))
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
