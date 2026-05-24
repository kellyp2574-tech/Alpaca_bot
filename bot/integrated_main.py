"""Combined Overnight Rebound Bot — Main Orchestrator

Paper test sleeve: CLEAN_OVERNIGHT_MR
  - Buy 15:45, $1–$2, day_ret <= -5%, close_position <= 0.25, ADV >= $1M
  - Rank by lowest close_position, top 3, min 2 candidates
  - ETF regime sizing: full size when SPY/IWM/QQQ avg is red before 15:45, half size otherwise
  - Exit 09:30

GDP/MOM is disabled for this paper test; the 09:30 batched sell/failsafe stack is unchanged.

Daily Schedule (ET) — morning bot starts around 05:00 AM:

MORNING (T+1 exits — positions from yesterday's 15:45 entries):
  05:00  Start, detect overnight positions from broker
  05:00, 05:15, 05:30, 05:45  Rolling premarket dynamic limit classification (decisive symbols only)
  06:00  Final premarket classification for all unresolved symbols (runs once,
         within the 06:00–06:02 cutoff window)
  09:25  Cancel any remaining premarket limits, freeze broker exit plan
  09:30  Submit batched market sells for all remaining broker positions
  09:31  Broker-native rescue pass for any remaining positions
  09:45  V2 failsafe — verify broker is flat or force-flatten any stragglers

AFTERNOON (T-1 entries — new positions for tomorrow's exits):
  15:30  Build universe (Massive + Alpaca, $1–10, ADV sizing cap protects)
  15:45  Fetch latest 9:30-15:45 minute bars, build both MR and GDP candidates
  15:45  Daily-loss circuit breaker check, then execute entries
  16:00  Confirm positions held overnight, save state, done
"""
import logging
import math
import os
import requests
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as dt_time, date, timedelta
from typing import List, Optional, Dict, Any, Tuple
from zoneinfo import ZoneInfo

from bot import config
from bot import premarket_classifier
from bot import state_io
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
from bot.etf_router import ETFRouter, RouterDecision
from bot.position_manager_overnight import PositionManager, Position
from bot.state_manager import StateManager
from bot.universe_builder import (
    build_universe,
    filter_minute_data_quality,
    filter_execution_ready,
    save_universe_audit,
    save_candidates_audit,
    UniverseDiagnostics,
    ExecutionDiagnostics,
)

# Setup logging with safe fallbacks
_default_log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
LOG_DIR = getattr(config, "LOG_DIR", _default_log_dir)
LOG_FILE = getattr(config, "LOG_FILE", os.path.join(LOG_DIR, "combined_overnight_bot.log"))
LOG_LEVEL = getattr(config, "LOG_LEVEL", "INFO")
LOG_FORMAT = getattr(
    config,
    "LOG_FORMAT",
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ],
    force=True,
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
    (9, 24, 10, 2),   # Order cancel, 09:30 batch sells, 09:31 rescue + ETF router
    (10, 59, 11, 2),  # UVXY exit
    (13, 59, 14, 2),  # SQQQ exit
    (14, 59, 15, 2),  # TQQQ exit
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
        self.etf_router = ETFRouter(config)

        # Universe & candidates
        self.universe: List[str] = []
        self.mr_candidates: List[MeanReversionCandidate] = []
        self.gdp_candidates: List[GreenDayPullbackCandidate] = []
        self._universe_diag: Optional[UniverseDiagnostics] = None

        # ETF Router state
        self.router_decision: Optional[RouterDecision] = None
        self.router_traded_today = False
        self.router_branch: Optional[str] = None
        self.mr_blocked_today = False
        self.etf_position: Optional[Dict[str, Any]] = None  # Current ETF position if any
        self.etf_opens_930: Dict[str, float] = {}  # 9:30 opens for tape
        self.tape_recording_active = False
        # Throttle for _update_tape (API efficiency #1). Monotonic seconds
        # of the last snapshot fetch; 0.0 means "never run yet".
        self._tape_last_update_monotonic: float = 0.0

        # Stage flags
        self.startup_done = False         # 9:00-9:25 startup phase
        self.tape_initialized = False     # 9:30 opens recorded
        self.router_decision_made = False  # 10:00 decision complete
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
        
        # Open market exit state
        self.open_market_rescue_done = False
        
        self.end_of_day_reports_done = False

        # Failsafe
        self.post_exit_failsafe_done = False

        # PDT guard: symbols sold today (no same-day re-entry when equity < $50k)
        self.sold_today: set = set()

        # Daily-loss kill switch: when tripped, blocks BOTH the 10:00 ETF
        # router entry and the 15:45 MR entries for the rest of the session.
        # Persisted in state so a mid-day restart respects yesterday-was-bad
        # is irrelevant (state is reset across dates) but a same-day restart
        # after a crash still honors the breaker.
        self.kill_switch_tripped: bool = False
        self.kill_switch_reason: Optional[str] = None

        # Graceful shutdown flag set by SIGINT/SIGTERM handlers installed in
        # main(). The main loop polls this between ticks and exits cleanly:
        # save state + stop fill stream. NEVER bypass exits-in-progress.
        self._shutdown_requested: bool = False
        self._shutdown_signal: Optional[str] = None

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
                            self.gdp_exits_done = True
                            self.mr_exits_done = True
                            self._save_state()
                        elif getattr(config, "ENABLE_RED_OPEN_TRAIL_EXIT", False):
                            self._submit_red_open_trail_or_sell_green()
                            self.gdp_exits_done = True
                            self.mr_exits_done = True
                            self._save_state()
                        else:
                            self._exit_sleeve_positions("GDP", "09:30 all positions (GDP)")
                            self._exit_sleeve_positions("MR", "09:30 all positions (MR)")
                            self.gdp_exits_done = True
                            self.mr_exits_done = True
                            self._save_state()

                    # 09:31 — broker-native rescue pass for any remaining positions.
                    # Issue #7: this used to require gdp_exits_done AND mr_exits_done
                    # to flip True at 09:30 — but if the 09:30 submit block raised
                    # mid-way (e.g. broker outage), the flags never flipped and the
                    # rescue+failsafe were silently skipped while positions remained
                    # open. Gate on TIME ALONE; the position-count check inside is
                    # what decides whether to act.
                    t_rescue = dt_time(t_exit_all.hour, t_exit_all.minute + 1)
                    if (not self.open_market_rescue_done
                            and current_time >= t_rescue):
                        bc = self.position_mgr.broker_position_count()
                        if bc > 0:
                            logger.warning(f"09:31 rescue: broker still has {bc} positions, running broker-native rescue")
                            self._run_broker_native_rescue()
                        elif bc == 0:
                            logger.info("09:31 rescue: broker confirmed flat")
                        self.open_market_rescue_done = True
                        self._save_state()

                    # Failsafe at V2_FAILSAFE_TIME (or RED_OPEN_TRAIL_FAILSAFE_TIME).
                    # Same hardening as the rescue: time + broker-not-flat is the
                    # only requirement so a partial-failure 09:30 path still gets
                    # the safety net.
                    if (not self.post_exit_failsafe_done
                            and current_time >= t_failsafe):
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

                    # Early completion: allowed after the 09:30 submit time when no
                    # trailing orders are active and the broker is confirmed flat.
                    # No longer requires gdp_exits_done/mr_exits_done flags.
                    if (not getattr(config, "ENABLE_RED_OPEN_TRAIL_EXIT", False)
                            and current_time >= t_exit_all
                            and not self.morning_exits_done):
                        bc = self.position_mgr.broker_position_count()
                        if bc == 0:
                            logger.info("All exits complete — broker confirmed flat")
                            self.position_mgr.positions.clear()
                            self.morning_exits_done = True
                            self._save_state()

            # ════════════════════════════════════════════
            # INTRADAY ETF ROUTER (9:00 - 15:45)
            # ════════════════════════════════════════════

            if getattr(config, "ETF_ROUTER_ENABLED", False):
                t_startup = _parse_config_time(getattr(config, "BOT_START_TIME", "09:00"))
                t_market_open = _parse_config_time(getattr(config, "MARKET_OPEN_TIME", "09:30"))
                t_router_decision = _parse_config_time(getattr(config, "ROUTER_DECISION_TIME", "10:00"))
                t_uvxy_exit = _parse_config_time(getattr(config, "UVXY_EXIT_TIME", "11:00"))
                t_sqqq_exit = _parse_config_time(getattr(config, "SQQQ_EXIT_TIME", "14:00"))
                t_tqqq_exit = _parse_config_time(getattr(config, "TQQQ_EXIT_TIME", "15:00"))

                # 09:00 startup - pre-market prep (allow up to 10:00 for late starts)
                if (not self.startup_done
                        and current_time >= t_startup
                        and current_time < t_router_decision):
                    self._run_startup_phase()

                # 09:30 initialize ETF tape (once market opens)
                if (self.startup_done
                        and not self.tape_initialized
                        and not self.router_decision_made
                        and current_time >= t_market_open
                        and current_time < t_router_decision):
                    # Guard: if starting after 09:31, tape will be incomplete (missing 09:30-09:xx data)
                    # For safety, disable router on late starts until backfill is implemented
                    if current_time > dt_time(9, 31):
                        logger.warning(f"ETF router late-start at {current_time.strftime('%H:%M')} without 09:30 tape; disabling router for today")
                        logger.warning("Router disabled to prevent false signals from incomplete range calculations")
                        self.router_decision_made = True
                        self.router_traded_today = False
                        self.mr_blocked_today = False
                        self.router_branch = "Late start - router disabled"
                        self._save_state()
                    else:
                        self._initialize_tape_recording()

                # 09:30-10:00 update tape with prices
                if (self.tape_initialized
                        and not self.router_decision_made
                        and current_time >= t_market_open
                        and current_time < t_router_decision):
                    self._update_tape()

                # 10:00 make router decision
                if (self.tape_initialized
                        and not self.router_decision_made
                        and current_time >= t_router_decision):
                    self._update_tape(force=True)  # Final tape update bypasses throttle
                    self._make_router_decision()

                # ETF exit checkpoints (11:00 UVXY, 14:00 SQQQ, 15:00 TQQQ)
                if self.router_traded_today and self.etf_position:
                    symbol = self.etf_position.get("symbol")
                    if symbol == "UVXY" and current_time >= t_uvxy_exit:
                        self._check_etf_exits(current_time)
                    elif symbol == "SQQQ" and current_time >= t_sqqq_exit:
                        self._check_etf_exits(current_time)
                    elif symbol == "TQQQ" and current_time >= t_tqqq_exit:
                        self._check_etf_exits(current_time)

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

            # Graceful shutdown check (SIGINT / SIGTERM).
            if self._shutdown_requested:
                logger.warning(
                    f"Shutdown requested ({self._shutdown_signal}); saving state and exiting loop"
                )
                try:
                    self._save_state()
                except Exception:
                    logger.warning("Shutdown: state save failed", exc_info=True)
                break

            # Adaptive sleep: 1s during hot windows (open, close, premarket
            # checkpoints) so transitions are prompt; 30s otherwise so the bot
            # spends ~1500 ticks/day instead of ~36000 — drastically lower CPU
            # and shared rate-limit pressure. The sleep is broken into 1s
            # chunks so a SIGINT during a 30s idle is honored within ~1s
            # instead of waiting out the full interval.
            sleep_total = 1 if _is_hot_window(current_time) else 30
            slept = 0
            while slept < sleep_total:
                if self._shutdown_requested:
                    break
                time.sleep(1)
                slept += 1

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

    def _run_broker_native_rescue(self):
        """Broker-native rescue pass at 09:31 for any remaining positions.
        
        This fetches broker positions directly and market-sells whatever remains.
        Submits all orders first, then reconciles with a small sleep for Alpaca to update state.
        Only marks exits complete after broker confirms flat.
        """
        broker_positions = self.position_mgr.get_broker_positions()
        if broker_positions is None:
            logger.error("09:31 rescue: broker position read failed; skipping")
            return
        
        if not broker_positions:
            logger.info("09:31 rescue: broker confirmed flat")
            return
        
        logger.warning(f"09:31 rescue: broker has {len(broker_positions)} positions, selling all")
        
        submitted = []
        
        for broker_pos in broker_positions:
            symbol = str(broker_pos.get("symbol", "")).upper()
            try:
                qty = abs(int(float(broker_pos.get("qty", 0))))
            except (TypeError, ValueError):
                logger.warning(f"09:31 rescue: bad qty for {symbol}: {broker_pos.get('qty')}")
                continue
            
            if qty <= 0:
                continue
            
            logger.info(f"09:31 rescue: selling {symbol} x{qty}")
            resp = self.position_mgr._submit_sell_order(
                symbol=symbol,
                qty=qty,
                order_type="market",
                time_in_force="day",
                extended_hours=False,
            )
            
            if resp and resp.get("id"):
                submitted.append((symbol, qty, resp["id"]))
                logger.info(f"09:31 rescue: submitted market sell for {symbol} x{qty}, order_id={resp['id']}")
            else:
                logger.error(f"09:31 rescue: failed to submit market sell for {symbol} x{qty}")
        
        logger.warning(f"09:31 rescue: submitted={len(submitted)}")
        
        # Small sleep to allow Alpaca to update position state
        time.sleep(3)
        
        # Reconcile after rescue
        self.position_mgr.reconcile_local_positions_from_broker()
        remaining = self.position_mgr.broker_position_count()
        if remaining == 0:
            logger.info("09:31 rescue: broker confirmed flat after rescue")
        else:
            logger.warning(f"09:31 rescue: broker still has {remaining} positions after rescue")

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

            # 1. Fetch 9:30-15:50 minute bars for the full base universe AND
            # the MR ETF regime symbols in a single batched request (API #2:
            # avoids a separate get_intraday_bars_for_signal call inside
            # _compute_mr_etf_regime_size_multiplier).
            regime_symbols = []
            if getattr(config, "ENABLE_MR_ETF_REGIME_SIZING", False):
                regime_symbols = list(
                    getattr(config, "MR_ETF_REGIME_SYMBOLS", ["SPY", "IWM", "QQQ"])
                )
            fetch_symbols = list(self.universe)
            for s in regime_symbols:
                if s not in fetch_symbols:
                    fetch_symbols.append(s)

            logger.info(
                f"Fetching 9:30-{signal_end} minute bars for {len(fetch_symbols)} symbols "
                f"({len(self.universe)} universe + {len(regime_symbols)} regime ETFs)..."
            )
            self._minute_bars = self.alpaca.get_intraday_bars_for_signal(
                fetch_symbols, today, start="09:30", end=signal_end,
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

            # Paper-test: enforce min ADV filter explicitly (config may not be wired into scorer)
            min_adv = getattr(config, "MR_MIN_AVG_DOLLAR_VOLUME", 0)
            if min_adv > 0:
                filtered_mr = [
                    c for c in filtered_mr
                    if getattr(c, "adv_dollars", 0.0) >= min_adv
                ]

            # Paper-test: rank by lowest close_position when configured
            if getattr(config, "MR_RANK_BY_CLOSE_LOCATION_ONLY", False):
                filtered_mr = sorted(
                    filtered_mr,
                    key=lambda c: (
                        getattr(c, "close_position", 999.0),
                        getattr(c, "day_return", 0.0),
                    )
                )

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

    def _check_daily_loss_kill_switch(self, account: Optional[dict] = None) -> bool:
        """Evaluate the global daily-loss kill switch.

        Trips ``self.kill_switch_tripped`` (and persists state) when
        ``(equity - last_equity) / last_equity <= -DAILY_LOSS_LIMIT_PCT``.
        Once tripped, the flag remains True for the rest of the session and
        blocks both the 10:00 ETF router entry and the 15:45 MR entries.

        Args:
            account: Optional pre-fetched account dict (avoids redundant
                API calls when the caller already has one).

        Returns:
            True if the kill switch is tripped (either previously or just
            now), False otherwise.
        """
        if self.kill_switch_tripped:
            return True

        loss_limit = float(getattr(config, "DAILY_LOSS_LIMIT_PCT", 0.0) or 0.0)
        if loss_limit <= 0:
            return False

        try:
            acct = account if account is not None else self.position_mgr.get_account()
        except Exception:
            logger.warning("kill-switch: account fetch failed; cannot evaluate", exc_info=True)
            return False
        if not acct:
            return False

        try:
            equity = float(acct.get("equity") or 0.0)
            last_equity = float(acct.get("last_equity") or 0.0)
        except (TypeError, ValueError):
            return False
        if equity <= 0 or last_equity <= 0:
            return False

        day_ret = (equity - last_equity) / last_equity
        if day_ret <= -loss_limit:
            self.kill_switch_tripped = True
            self.kill_switch_reason = (
                f"day_ret={day_ret:+.2%} <= -{loss_limit:.0%} "
                f"(equity=${equity:,.2f} vs last_equity=${last_equity:,.2f})"
            )
            logger.critical(
                f"DAILY LOSS KILL SWITCH TRIPPED — {self.kill_switch_reason}. "
                f"All new entries (ETF router + MR) BLOCKED for the rest of the session."
            )
            try:
                self._save_state()
            except Exception:
                logger.warning("kill-switch: state save failed", exc_info=True)
            return True

        return False

    def _compute_mr_etf_regime_size_multiplier(self) -> tuple[float, dict]:
        """Return MR size multiplier from SPY/IWM/QQQ avg return vs open.

        Uses the same late-day timing as the paper-test signal. If ETF data is
        unavailable, fail open at full size but log the reason so the paper test
        does not silently skip otherwise valid MR candidates.
        """
        if not getattr(config, "ENABLE_MR_ETF_REGIME_SIZING", False):
            return 1.0, {"enabled": False, "reason": "disabled"}

        symbols = list(getattr(config, "MR_ETF_REGIME_SYMBOLS", ["SPY", "IWM", "QQQ"]))
        today = date.today().isoformat()
        end_time = getattr(config, "ENTRY_TIME", "15:45")

        # Reuse the minute bars already fetched for the universe at 15:45
        # (API #2). _step_score_and_rank now prepends the regime ETFs to the
        # batched fetch, so SPY/IWM/QQQ should already be in self._minute_bars.
        # Fall back to a separate fetch only if the cache is missing data
        # (e.g. _step_score_and_rank was skipped or returned early).
        cached = self._minute_bars or {}
        missing = [s for s in symbols if not cached.get(s)]
        if missing:
            logger.info(
                f"MR ETF regime sizing: {len(symbols) - len(missing)} cached, "
                f"fetching {missing} from API"
            )
            try:
                extra = self.alpaca.get_intraday_bars_for_signal(
                    missing, today, start="09:30", end=end_time,
                )
                bars_by_symbol = dict(cached)
                bars_by_symbol.update(extra)
            except Exception:
                logger.warning("MR ETF regime sizing: failed to fetch ETF bars; using full size", exc_info=True)
                return 1.0, {"enabled": True, "reason": "fetch_failed"}
        else:
            bars_by_symbol = cached

        returns = {}

        def _bar_val(bar: dict, *keys: str) -> Optional[float]:
            for key in keys:
                try:
                    val = bar.get(key)
                    if val is not None:
                        return float(val)
                except (TypeError, ValueError, AttributeError):
                    continue
            return None

        for sym in symbols:
            bars = bars_by_symbol.get(sym, []) if bars_by_symbol else []
            if not bars:
                continue
            first = bars[0]
            last = bars[-1]
            open_px = _bar_val(first, "o", "open")
            close_px = _bar_val(last, "c", "close")
            if open_px and close_px and open_px > 0:
                returns[sym] = close_px / open_px - 1.0

        if not returns:
            logger.warning("MR ETF regime sizing: no usable ETF bars; using full size")
            return 1.0, {"enabled": True, "reason": "no_usable_bars"}

        avg_ret = sum(returns.values()) / len(returns)
        is_negative = avg_ret < 0
        mult = (
            float(getattr(config, "MR_ETF_NEGATIVE_SIZE_MULT", 1.0))
            if is_negative
            else float(getattr(config, "MR_ETF_POSITIVE_SIZE_MULT", 0.5))
        )
        info = {
            "enabled": True,
            "reason": "ok",
            "avg_return": avg_ret,
            "is_negative": is_negative,
            "returns": returns,
            "multiplier": mult,
        }
        logger.warning(
            "MR ETF REGIME SIZE: avg=%+.2f%% multiplier=%.2f returns=%s",
            avg_ret * 100.0,
            mult,
            {k: f"{v:+.2%}" for k, v in returns.items()},
        )
        return mult, info

    def _step_execute_entries(self):
        """15:45: clean MR-only paper allocation -> execution-gate -> market buys."""
        logger.info("=" * 50)
        logger.info("ENTRY EXECUTION: Clean MR paper-test market buy orders")
        logger.info("=" * 50)

        # Check MR permission - blocked if ETF router signal fired today
        # Note: MR is blocked even if ETF entry failed (regime protection)
        if self.mr_blocked_today or self.router_traded_today:
            branch = self.router_branch or "unknown"
            has_etf_position = self.etf_position is not None
            logger.info("MR entries BLOCKED - ETF router signal fired today (branch=%s, has_position=%s)", branch, has_etf_position)
            logger.info("MR blocked because router signal fired; ETF position may or may not have filled")
            logger.info("Skipping MR candidate scan and entry")
            self.entries_done = True
            return

        # Check if MR is enabled in config
        if not getattr(config, "MR_OVERNIGHT_ENABLED", True):
            logger.info("MR entries DISABLED in config - skipping")
            self.entries_done = True
            return

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

            # Daily loss kill switch — global flag set by
            # _check_daily_loss_kill_switch(). Trips if today's PnL is worse
            # than DAILY_LOSS_LIMIT_PCT. account['last_equity'] is yesterday's
            # close equity. Once tripped, also blocks any future entries this
            # session (and the 10:00 ETF router entry checks the same flag).
            if self._check_daily_loss_kill_switch(account=account):
                logger.critical(
                    f"MR entries BLOCKED by daily-loss kill switch — {self.kill_switch_reason}"
                )
                self.entries_done = True
                return
            try:
                last_equity = float(account.get("last_equity") or 0.0)
                if last_equity > 0:
                    day_ret = (equity - last_equity) / last_equity
                    logger.info(
                        f"Daily PnL check OK: {day_ret:+.2%} (limit "
                        f"-{float(getattr(config, 'DAILY_LOSS_LIMIT_PCT', 0.0)):.0%})"
                    )
            except (TypeError, ValueError):
                pass

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

            # ETF-regime sizing from the clean-cache finalist:
            # full size when 3-ETF avg is negative before entry, half size otherwise.
            mr_size_mult, mr_regime_info = self._compute_mr_etf_regime_size_multiplier()

            # Calculate sleeve budgets and target slots. GDP/MOM is intentionally zero for this paper test.
            mr_budget = deployable * config.MR_ALLOCATION_PCT * mr_size_mult
            gdp_budget = deployable * config.GDP_ALLOCATION_PCT
            logger.info(
                f"Sleeve budgets: MR ${mr_budget:,.2f} "
                f"({config.MR_ALLOCATION_PCT:.0%} * regime_mult={mr_size_mult:.2f}) | "
                f"GDP ${gdp_budget:,.2f} ({config.GDP_ALLOCATION_PCT:.0%}) | "
                f"regime={mr_regime_info}"
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
                max_spread_pct=getattr(config, "ENTRY_MAX_SPREAD_PCT", 0.05), require_quote=True,
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

            # Paper-test guard: require min candidates AFTER execution gate (spread/quote check)
            mr_min_candidates = int(getattr(config, "MR_MIN_CANDIDATES", 1) or 1)
            if len(mr_orderable) < mr_min_candidates:
                logger.warning(
                    "MR paper test: only %d orderable candidates after execution gate, below min_candidates=%d — skipping entries",
                    len(mr_orderable),
                    mr_min_candidates,
                )
                self.entries_done = True
                return

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

            # MR-only paper test: no fallback/redeployment. Leftover stays as cash.
            # This ensures selection remains exactly the researched top-N close-location sleeve.
            if getattr(config, "GDP_MAX_POSITIONS", 0) <= 0 or getattr(config, "GDP_ALLOCATION_PCT", 0.0) <= 0:
                if total_leftover > config.MIN_POSITION_DOLLARS:
                    logger.info(
                        "MR-only paper test: skipping global leftover redeployment. "
                        "Leftover $%.2f remains as cash.",
                        total_leftover,
                    )
            elif total_leftover > config.MIN_POSITION_DOLLARS:
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
            # Submit market buy orders concurrently with short timeout.
            total_deployed = 0.0

            # Track buying power locally to avoid hitting /v2/account once per symbol
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

            # Create deterministic client_order_id for each allocation
            # Format: BOT-YYYYMMDD-HHMMSS-SYMBOL
            timestamp = datetime.now(_ET).strftime("%Y%m%d-%H%M%S")
            submission_plans = []  # List of (alloc, qty, client_order_id, price_ref, limit_price)

            # Issue #6: marketable-limit slippage cap for MR entries.
            mr_slippage_pct = float(getattr(config, "ENTRY_MAX_SLIPPAGE_PCT", 0.02))

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

                if qty < config.MIN_SHARES:
                    logger.warning(
                        f"ENTRY SKIP {symbol}: adaptive qty {qty} < {config.MIN_SHARES} min shares "
                        f"(bp=${bp_remaining:,.2f}, price={price_ref:.4f})"
                    )
                    exec_diag.failed_submissions[symbol] = "bp_resize_below_min"
                    continue

                # Marketable-limit price (issue #6): cap at ask * (1 + slippage).
                # Falls back to market order (limit_price=None) when the ask is
                # missing from the snapshot, preserving prior behavior on
                # degraded data instead of refusing to enter.
                snap = (fresh_snaps or {}).get(symbol, {}) or {}
                ask = snap.get("ask")
                if ask and float(ask) > 0:
                    limit_price = float(ask) * (1.0 + mr_slippage_pct)
                else:
                    limit_price = None

                # Create deterministic client_order_id
                client_order_id = f"BOT-{timestamp}-{symbol}"
                planned_notional = qty * (limit_price if limit_price else price_ref)

                logger.info(
                    f"ENTRY PLAN {symbol}: qty={qty}, price_ref={price_ref:.4f}, "
                    f"ask={ask}, limit={f'{limit_price:.4f}' if limit_price else 'MARKET'}, "
                    f"notional={planned_notional:,.2f}, bp_remaining={bp_remaining:,.2f}, "
                    f"sleeve={alloc.sleeve}, rank={alloc.rank}, client_id={client_order_id}"
                )

                submission_plans.append((alloc, qty, client_order_id, price_ref, limit_price))
                # Decrement local BP tracker by the planned notional
                bp_remaining = max(0.0, bp_remaining - planned_notional)

            # Batch submit all orders concurrently with short timeout.
            submitted_orders = []      # List of (order_id, alloc, qty, candidate, client_order_id)
            submission_timeouts = []   # List of (alloc, qty, client_order_id, price_ref)

            def _submit_entry_order(plan):
                """Submit one buy order. Runs inside ThreadPoolExecutor."""
                alloc, qty, client_order_id, price_ref, limit_price = plan
                symbol = alloc.symbol
                submit_start = datetime.now(_ET)
                t0 = time.perf_counter()

                try:
                    if limit_price is not None:
                        buy_resp, error_type = self.position_mgr.submit_buy_order(
                            symbol,
                            qty,
                            client_order_id=client_order_id,
                            timeout=getattr(config, "ENTRY_SUBMIT_TIMEOUT_SECONDS", 2),
                            order_type="limit",
                            limit_price=limit_price,
                        )
                    else:
                        buy_resp, error_type = self.position_mgr.submit_buy_order(
                            symbol,
                            qty,
                            client_order_id=client_order_id,
                            timeout=getattr(config, "ENTRY_SUBMIT_TIMEOUT_SECONDS", 2),
                        )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    return {
                        "symbol": symbol,
                        "alloc": alloc,
                        "qty": qty,
                        "client_order_id": client_order_id,
                        "price_ref": price_ref,
                        "buy_resp": buy_resp,
                        "error_type": error_type,
                        "elapsed_ms": elapsed_ms,
                        "submit_start": submit_start,
                        "exception": None,
                    }

                except Exception as e:
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return {
                        "symbol": symbol,
                        "alloc": alloc,
                        "qty": qty,
                        "client_order_id": client_order_id,
                        "price_ref": price_ref,
                        "buy_resp": None,
                        "error_type": "exception",
                        "elapsed_ms": elapsed_ms,
                        "submit_start": submit_start,
                        "exception": e,
                    }

            max_workers = min(
                len(submission_plans),
                int(getattr(config, "ENTRY_SUBMIT_MAX_WORKERS", 8)),
            )

            logger.warning(
                "ENTRY CONCURRENT SUBMIT START: orders=%d workers=%d timeout=%ss",
                len(submission_plans),
                max_workers,
                getattr(config, "ENTRY_SUBMIT_TIMEOUT_SECONDS", 2),
            )

            # Collect per-order submission latencies so we can publish a
            # p50/p95/avg summary in the run_health artifact (observability).
            submit_latencies_ms: List[float] = []

            if max_workers <= 0:
                logger.warning("ENTRY CONCURRENT SUBMIT: no submission plans")
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(_submit_entry_order, plan): plan
                        for plan in submission_plans
                    }

                    for future in as_completed(futures):
                        result = future.result()

                        symbol = result["symbol"]
                        alloc = result["alloc"]
                        qty = result["qty"]
                        client_order_id = result["client_order_id"]
                        price_ref = result["price_ref"]
                        buy_resp = result["buy_resp"]
                        error_type = result["error_type"]
                        elapsed_ms = result["elapsed_ms"]
                        submit_latencies_ms.append(float(elapsed_ms))

                        if buy_resp and buy_resp.get("id"):
                            order_id = buy_resp["id"]
                            submitted_orders.append((order_id, alloc, qty, alloc.candidate, client_order_id))
                            exec_diag.submitted_symbols.append(symbol)
                            logger.info(
                                "ENTRY SUBMITTED %s: order_id=%s client_id=%s elapsed_ms=%.1f",
                                symbol,
                                order_id,
                                client_order_id,
                                elapsed_ms,
                            )

                        elif error_type in ("timeout", "network_error", "exception"):
                            submission_timeouts.append((alloc, qty, client_order_id, price_ref))
                            logger.warning(
                                "ENTRY TIMEOUT/NETWORK %s: error=%s elapsed_ms=%.1f; will reconcile client_id=%s",
                                symbol,
                                error_type,
                                elapsed_ms,
                                client_order_id,
                            )

                        else:
                            exec_diag.failed_submissions[symbol] = f"submit_failed_{error_type}"
                            logger.error(
                                "ENTRY REJECTED %s: error=%s elapsed_ms=%.1f; no reconciliation",
                                symbol,
                                error_type,
                                elapsed_ms,
                            )

            logger.info(
                "ENTRY SUBMISSION SUMMARY: %d immediate success, %d timeout/error needing reconciliation",
                len(submitted_orders),
                len(submission_timeouts),
            )

            # Reconcile timeouts by querying orders by client_order_id
            if submission_timeouts:
                logger.info(f"ENTRY RECONCILING {len(submission_timeouts)} timeout/error submissions...")
                # Give Alpaca a moment to process orders
                time.sleep(1.0)
                
                for alloc, qty, client_order_id, price_ref in submission_timeouts:
                    symbol = alloc.symbol
                    try:
                        # Query orders by client_order_id using the correct endpoint
                        base_url = getattr(config, "ALPACA_BASE_URL", "https://api.alpaca.markets").rstrip("/")
                        url = f"{base_url}/v2/orders:by_client_order_id"
                        params = {"client_order_id": client_order_id}
                        resp = self.position_mgr.session.get(
                            url,
                            params=params,
                            timeout=getattr(config, "ENTRY_RECONCILE_TIMEOUT_SECONDS", 3),
                        )
                        resp.raise_for_status()
                        order_data = resp.json()  # Returns a single order object, not a list
                        
                        if order_data and order_data.get("id"):
                            # Order exists - add to submitted list
                            order_id = order_data.get("id")
                            submitted_orders.append((order_id, alloc, qty, alloc.candidate, client_order_id))
                            exec_diag.submitted_symbols.append(symbol)
                            logger.info(f"ENTRY RECONCILED {symbol}: order_id={order_id}, client_id={client_order_id}")
                        else:
                            # Order does not exist - treat as failed
                            exec_diag.failed_submissions[symbol] = "reconciliation_no_order"
                            logger.warning(f"ENTRY RECONCILE FAILED {symbol}: no order found for client_id={client_order_id}")
                    except requests.exceptions.HTTPError as e:
                        # 404 means order doesn't exist - treat as failed
                        if e.response and e.response.status_code == 404:
                            exec_diag.failed_submissions[symbol] = "reconciliation_no_order"
                            logger.warning(f"ENTRY RECONCILE FAILED {symbol}: 404 no order for client_id={client_order_id}")
                        else:
                            exec_diag.failed_submissions[symbol] = f"reconciliation_http_{e.response.status_code if e.response else 'unknown'}"
                            logger.error(f"ENTRY RECONCILE ERROR {symbol}: HTTP {e}")
                    except Exception as e:
                        exec_diag.failed_submissions[symbol] = f"reconciliation_error: {str(e)}"
                        logger.error(f"ENTRY RECONCILE ERROR {symbol}: {e}")
            
            # Pass 2: monitor fills for all submitted orders (with longer wait)
            for order_id, alloc, qty, candidate, client_order_id in submitted_orders:
                symbol = alloc.symbol
                fill = self.position_mgr.get_order_fill(order_id, max_wait=30)  # Increased wait for reconciliation
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
                        f"score={candidate.selection_score:.3f}, client_id={client_order_id}"
                    )
                else:
                    # Cancel unfilled orders
                    try:
                        self.position_mgr._cancel_order(order_id)
                        logger.warning(f"ENTRY NO FILL {symbol}: order canceled (order_id={order_id}, client_id={client_order_id})")
                    except Exception as e:
                        logger.error(f"ENTRY CANCEL ERROR {symbol}: {e}")
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

            # Latency summary (observability). p50/p95/avg of per-order
            # POST /v2/orders elapsed time. Surfaces a sluggish broker
            # endpoint into the run_health artifact instead of being buried
            # in per-order log lines.
            latency_summary: Dict[str, Any] = {"count": len(submit_latencies_ms)}
            if submit_latencies_ms:
                sorted_lat = sorted(submit_latencies_ms)
                n = len(sorted_lat)
                p50 = sorted_lat[n // 2]
                p95 = sorted_lat[min(n - 1, int(n * 0.95))]
                latency_summary.update({
                    "avg_ms": round(sum(sorted_lat) / n, 1),
                    "p50_ms": round(p50, 1),
                    "p95_ms": round(p95, 1),
                    "max_ms": round(sorted_lat[-1], 1),
                })

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
                "submit_latency_ms": latency_summary,
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

    # ────────────────────────────────────────────────────────────
    # Premarket classifier — delegated to bot.premarket_classifier
    # ────────────────────────────────────────────────────────────

    def _fetch_delayed_sip_premarket_bars(self, symbols: List[str], decision_dt: datetime) -> Dict[str, List[dict]]:
        return premarket_classifier.fetch_delayed_sip_premarket_bars(
            self.position_mgr.session, symbols, decision_dt,
        )

    @staticmethod
    def _bar_dt(bar: dict) -> Optional[datetime]:
        return premarket_classifier.bar_dt(bar)

    @staticmethod
    def _bar_float(bar: dict, *keys: str) -> Optional[float]:
        return premarket_classifier.bar_float(bar, *keys)

    def _compute_delayed_sip_premarket_metrics(self, symbol: str, buy_price: float, decision_dt: datetime, pre_fetched_bars: Optional[Dict[str, List[dict]]] = None) -> Dict[str, Any]:
        return premarket_classifier.compute_delayed_sip_premarket_metrics(
            self.position_mgr.session, symbol, buy_price, decision_dt, pre_fetched_bars,
        )

    def _classify_premarket_limit(self, pos: Position, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return premarket_classifier.classify_premarket_limit(pos, metrics)

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
        return premarket_classifier.is_decisive_premarket_signal(
            decision_time=decision_time,
            final_time=final_time,
            current_return=current_return,
            distance_from_high=distance_from_high,
            trend_from_first_bar=trend_from_first_bar,
            minutes_traded=minutes_traded,
            last_bar_age_minutes=last_bar_age_minutes,
            sleeve=sleeve,
            data_source=data_source,
        )

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

        # Batch fetch delayed SIP bars for all symbols before looping
        # This uses the historical bars endpoint with feed=sip and end=decision_dt - 16 minutes
        all_bars = self._fetch_delayed_sip_premarket_bars(symbols, decision_dt)
        logger.info(f"PREMARKET LIMITS: batch delayed SIP bars fetched {len(all_bars)} symbols")
        
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

            metrics = self._compute_delayed_sip_premarket_metrics(symbol, entry_price, decision_dt, pre_fetched_bars=all_bars)

            # Check if signal is decisive (act now) or should wait
            current_return = metrics.get("current_return")
            
            # If data is unavailable, skip this checkpoint
            if current_return is None:
                if decision_time_str >= final_time_str:
                    skipped += 1
                    logger.error(
                        f"PREMARKET LIMIT {symbol}: DATA UNAVAILABLE at FINAL {decision_time_str}; "
                        f"no premarket limit placed; regular 09:30/MOO exit will handle it"
                    )
                else:
                    waited += 1
                    logger.warning(
                        f"PREMARKET LIMIT {symbol}: DATA UNAVAILABLE at {decision_time_str}; "
                        f"will check again in 15 minutes"
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
            decision = self._classify_premarket_limit(pos, metrics)

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

        # Append this checkpoint's outcome to the daily premarket-limits
        # artifact (observability). One file per session with one entry
        # per 05:00 / 05:15 / 05:30 / 05:45 / 06:00 checkpoint, so the
        # entire premarket decision sequence is reviewable as JSON.
        try:
            self._append_premarket_limits_artifact(
                decision_time_str=decision_time_str,
                placed=placed,
                no_cap=no_cap,
                skipped=skipped,
                waited=waited,
                is_final=is_final,
                symbol_count=len(symbols),
                limit_order_ids=dict(self.premarket_limit_order_ids),
            )
        except Exception:
            logger.warning("premarket_limits artifact append failed", exc_info=True)

        # If final checkpoint, mark as done
        if is_final:
            self.premarket_dynamic_limits_done = True
            self._save_state()
        else:
            # Save state after each checkpoint to track decided symbols
            self._save_state()

    def _append_premarket_limits_artifact(
        self,
        decision_time_str: str,
        placed: int,
        no_cap: int,
        skipped: int,
        waited: int,
        is_final: bool,
        symbol_count: int,
        limit_order_ids: Dict[str, str],
    ):
        """Append one checkpoint outcome to state/logs/premarket_limits_YYYY-MM-DD.json.

        File schema:
          {
            "date": "YYYY-MM-DD",
            "checkpoints": [
              {"time": "05:00", "placed": 2, "no_cap": 0, "skipped": 1, "waited": 3, ...},
              ...
            ],
            "limit_order_ids_at_end": {"AAPL": "<order_id>", ...}
          }

        Idempotent at the checkpoint level: appending the same HH:MM twice
        replaces the prior entry rather than duplicating it (defensive
        against any future retry path).
        """
        import json
        today = date.today().isoformat()
        path = os.path.join(config.LOG_DIR, f"premarket_limits_{today}.json")

        existing: Dict[str, Any] = {"date": today, "checkpoints": []}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    existing = json.load(f) or existing
                if not isinstance(existing, dict) or "checkpoints" not in existing:
                    existing = {"date": today, "checkpoints": []}
            except Exception:
                logger.warning(f"premarket_limits artifact: could not read {path}; overwriting", exc_info=True)
                existing = {"date": today, "checkpoints": []}

        # Replace prior entry for this checkpoint if present.
        checkpoints = [c for c in existing.get("checkpoints", []) if c.get("time") != decision_time_str]
        checkpoints.append({
            "time": decision_time_str,
            "is_final": is_final,
            "symbols_considered": symbol_count,
            "placed": placed,
            "no_cap": no_cap,
            "skipped": skipped,
            "waited": waited,
        })
        existing["checkpoints"] = sorted(checkpoints, key=lambda c: c["time"])
        existing["limit_order_ids_at_end"] = limit_order_ids
        existing["decided_symbols"] = sorted(self.premarket_decided_symbols)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        logger.info(f"Premarket-limits artifact updated for {decision_time_str}: {path}")

    def _build_etf_router_summary(self) -> Dict[str, Any]:
        """Compact ETF router summary for run_health + standalone artifact.

        Captures what was decided, whether an entry actually fired, and
        the realized fill price so a post-day review can be done from a
        single JSON file. Returns an empty dict when the router is
        disabled (e.g. ``ETF_ROUTER_ENABLED=False``).
        """
        if not getattr(config, "ETF_ROUTER_ENABLED", False):
            return {"enabled": False}

        decision = self.router_decision
        summary: Dict[str, Any] = {
            "enabled": True,
            "tape_initialized": self.tape_initialized,
            "decision_made": self.router_decision_made,
            "branch": self.router_branch,
            "mr_blocked_today": self.mr_blocked_today,
            "router_traded_today": self.router_traded_today,
        }
        if decision is not None:
            try:
                summary["decision"] = decision.to_dict()
            except Exception:
                logger.warning("ETF router decision.to_dict() failed", exc_info=True)
        if self.etf_position:
            # The exit path nulls out etf_position on a successful exit, so
            # a non-null value here means we still have an open ETF
            # position at EOD (which would be a bug — log it loud).
            logger.warning(f"ETF router EOD: still holding {self.etf_position.get('symbol')}")
            summary["open_position_at_eod"] = self.etf_position
        return summary

    def _save_etf_router_artifact(self):
        """Write state/logs/etf_router_YYYY-MM-DD.json for forensic review."""
        import json
        today = date.today().isoformat()
        path = os.path.join(config.LOG_DIR, f"etf_router_{today}.json")
        payload = {"date": today, **self._build_etf_router_summary()}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info(f"ETF router artifact saved: {path}")

    # ────────────────────────────────────────────────────────────
    # State persistence + EOD reports — delegated to bot.state_io
    # ────────────────────────────────────────────────────────────

    def _save_end_of_day_reports(self):
        state_io.save_end_of_day_reports(self)

    def _finalize_day(self, clear_state: bool = True):
        state_io.finalize_day(self, clear_state=clear_state)

    def _save_state(self):
        state_io.save_state(self)

    def _load_state(self):
        state_io.load_state(self)

    # ═══════════════════════════════════════════════════════════════════════════
    # ETF Router Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_startup_phase(self):
        """9:00-9:25 AM: Startup and pre-market preparation."""
        logger.info("=" * 60)
        logger.info("STARTUP PHASE (9:00-9:25)")
        logger.info("=" * 60)

        # Load and log config
        logger.info(f"ETF Router enabled: {getattr(config, 'ETF_ROUTER_ENABLED', False)}")
        logger.info(f"MR Overnight enabled: {getattr(config, 'MR_OVERNIGHT_ENABLED', True)}")
        logger.info(f"MR Permission mode: {getattr(config, 'MR_PERMISSION_MODE', 'skip_if_router_traded')}")

        # Reconcile account state
        try:
            account = self.position_mgr.get_account()
            if account:
                cash = float(account.get("cash", 0))
                buying_power = float(account.get("buying_power", 0))
                logger.info(f"Account state - Cash: ${cash:,.2f}, BP: ${buying_power:,.2f}")
            else:
                logger.warning("Could not fetch account state")
        except Exception as e:
            logger.error(f"Error fetching account: {e}")

        # Check broker positions
        broker_positions = self.position_mgr.get_broker_positions()
        if broker_positions is not None:
            logger.info(f"Broker positions: {len(broker_positions)}")
            for pos in broker_positions:
                logger.info(f"  - {pos.get('symbol')}: {pos.get('qty')} shares")
        else:
            logger.warning("Could not fetch broker positions")

        # Cancel stale orders (ETF router symbols)
        self._cancel_stale_etf_orders()

        self.startup_done = True
        self._save_state()
        logger.info("Startup phase complete")

    def _cancel_stale_etf_orders(self):
        """Cancel only stale ETF router orders from a previous session.

        IMPORTANT: must NOT touch overnight premarket limit sells placed
        between 05:00 and 06:00 — those are on equity positions, never
        on ETF router symbols. Earlier versions used
        ``cancel_all_open_orders()`` here which wiped those limits before
        they had any chance to fill in the 09:00-09:25 pre-open window.
        """
        etf_symbols = getattr(
            config,
            "ETF_ROUTER_SYMBOLS",
            ["QQQ", "SPY", "IWM", "XLK", "VXX", "SQQQ", "UVXY", "TQQQ"],
        )
        logger.info(f"Cancelling any stale ETF-only orders for: {etf_symbols}")
        try:
            cancelled = self.position_mgr.cancel_orders_for_symbols(etf_symbols)
            logger.info(f"Stale ETF order cleanup complete: {cancelled} canceled")
        except Exception as e:
            logger.error(f"Error cancelling stale ETF orders: {e}")

    def _initialize_tape_recording(self):
        """9:30 AM: Record canonical 09:30 opens and start tape recording.

        Source priority (highest to lowest fidelity):
          1. First 1-min bar of today via get_intraday_bars_for_signal — this
             is the canonical 09:30 opening print.
          2. snapshot.daily_bar.o, but ONLY if daily_bar.t parses to today.
             At ~09:30:00 the daily_bar field often still holds the prior
             trading day's bar; we must reject those.
          3. snapshot.last_trade.p, but ONLY if last_trade.t is within the
             last 60s. This guards against stale pre-market prints (e.g.
             a 09:29:50 IEX trade being treated as the 09:30 open).

        If none of the sources are usable for a given symbol, that symbol is
        skipped. ``ETFTapeSnapshot.is_valid()`` will then be False for it and
        ``ETFRouter.make_decision`` will gracefully resolve to NO_TRADE.

        If NO symbols produce a usable open (the most common case when this
        runs at exactly 09:30:00.x and the 1-min bar hasn't formed yet), we
        leave ``tape_initialized = False`` so the next 1s tick retries.
        """
        logger.info("=" * 60)
        logger.info("TAPE INITIALIZATION (9:30)")
        logger.info("=" * 60)

        etf_symbols = getattr(
            config,
            "ETF_ROUTER_SYMBOLS",
            ["QQQ", "SPY", "IWM", "XLK", "VXX", "SQQQ", "UVXY", "TQQQ"],
        )
        now = datetime.now(_ET)
        today_iso = now.date().isoformat()

        try:
            # 1) Canonical source: first 1-min bar of today, 09:30 -> 09:31.
            minute_bars: Dict[str, List[dict]] = {}
            try:
                minute_bars = self.alpaca.get_intraday_bars_for_signal(
                    etf_symbols, today_iso, start="09:30", end="09:31",
                )
            except Exception:
                logger.warning("Tape init: 09:30 minute-bar fetch failed; will rely on snapshot fallbacks", exc_info=True)

            # 2) Snapshot fallbacks (also captured for "latest" logging).
            snapshots = self.alpaca.get_snapshots(etf_symbols) or {}

            opens: Dict[str, float] = {}
            sources: Dict[str, str] = {}

            for symbol in etf_symbols:
                open_price: Optional[float] = None
                source = "none"

                # ── Source 1: today's first 1-min bar ──────────────────────
                bars = minute_bars.get(symbol) or []
                if bars:
                    first_bar = bars[0]
                    bar_dt = self._bar_dt(first_bar)
                    bar_open = self._bar_float(first_bar, "o", "open")
                    if (
                        bar_open is not None
                        and bar_open > 0
                        and bar_dt is not None
                        and bar_dt.date() == now.date()
                        and bar_dt.hour == 9
                        and bar_dt.minute == 30
                    ):
                        open_price = bar_open
                        source = "minute_bar_0930"

                snap = snapshots.get(symbol, {}) or {}

                # ── Source 2: parsed snapshot last_price if fresh (<= 60s) ──
                # NOTE: get_snapshots returns flattened dicts; the raw
                # latestTrade/dailyBar fields are not exposed as nested
                # objects. There is no exposed daily_bar timestamp, so we
                # cannot safely use the daily_bar open as a fallback — it
                # may be the prior trading day's bar. Rely on a fresh
                # last_price instead.
                if open_price is None:
                    lt_p = snap.get("last_price")
                    lt_t = snap.get("timestamp")
                    if lt_p and lt_t:
                        try:
                            lt_dt = datetime.fromisoformat(str(lt_t).replace("Z", "+00:00"))
                            age_s = (now - lt_dt.astimezone(_ET)).total_seconds()
                        except (TypeError, ValueError):
                            age_s = 9999.0
                        if 0 <= age_s <= 60:
                            open_price = float(lt_p)
                            source = f"last_price_fresh_{age_s:.0f}s"
                        else:
                            logger.warning(
                                f"Tape init {symbol}: rejecting last_price — age={age_s:.0f}s (limit 60s)"
                            )

                if open_price is not None and open_price > 0:
                    opens[symbol] = open_price
                    self.etf_opens_930[symbol] = open_price
                    sources[symbol] = source
                    latest = snap.get("last_price", "N/A")
                    logger.info(
                        f"ETF open {symbol}: ${open_price:.2f} (source={source}, "
                        f"latest={latest}, ts={now.strftime('%H:%M:%S')})"
                    )
                else:
                    logger.warning(
                        f"Tape init {symbol}: no usable 09:30 open yet — will retry on next tick"
                    )

            logger.info(f"9:30 Opens summary: {opens}")
            logger.info(f"Open sources: {sources}")

            # If we couldn't seed ANY symbol yet, defer init so the next 1s
            # tick retries (the 1-min bar typically lands within ~30s of the
            # open with the IEX feed).
            if not opens:
                logger.warning("Tape init: no symbols had a usable open; retrying next tick")
                return

            # If we got SOME but not all, still proceed — the missing ones
            # will get populated by subsequent _update_tape() calls (their
            # open_930 will then be the first tape print, which is acceptable
            # for the symbols that couldn't be sourced cleanly).
            self.etf_router.start_recording(opens, datetime.now(_ET))
            self.tape_recording_active = True
            self.tape_initialized = True
            self._save_state()

        except Exception as e:
            logger.error(f"Error initializing tape: {e}", exc_info=True)

    def _update_tape(self, force: bool = False):
        """Update tape with latest prices during 9:30-10:00 window.

        Uses the parsed snapshot ``last_price`` field.

        Throttled (API efficiency #1) to one snapshot fetch per
        ``ETF_TAPE_UPDATE_INTERVAL_SECONDS`` (default 5s). The main loop
        ticks every 1s in this hot window, but the router only needs the
        9:30 open, the 9:45 continuation print, and the 10:00 final
        price — sub-second granularity is wasted budget.

        Pass ``force=True`` to bypass the throttle (used for the final
        tape update right before the 10:00 decision).
        """
        if not self.tape_recording_active:
            return

        # Throttle: skip if the previous fetch was too recent.
        interval = float(getattr(config, "ETF_TAPE_UPDATE_INTERVAL_SECONDS", 5))
        now_mono = time.monotonic()
        if not force and interval > 0 and (now_mono - self._tape_last_update_monotonic) < interval:
            return

        etf_symbols = getattr(
            config,
            "ETF_ROUTER_SYMBOLS",
            ["QQQ", "SPY", "IWM", "XLK", "VXX", "SQQQ", "UVXY", "TQQQ"],
        )

        try:
            snapshots = self.alpaca.get_snapshots(etf_symbols)
            now = datetime.now(_ET)

            for symbol in etf_symbols:
                snap = snapshots.get(symbol, {}) or {}
                price = snap.get("last_price")
                if price:
                    self.etf_router.update_tape(symbol, float(price), now)

            self._tape_last_update_monotonic = now_mono

        except Exception as e:
            logger.error(f"Error updating tape: {e}")

    def _make_router_decision(self):
        """10:00 AM: Make ETF router decision and (maybe) enter.

        The daily-loss kill switch is checked BEFORE the ETF entry. The
        decision and ``mr_blocked_today`` flag are still recorded so the
        EOD report reflects what the router would have done; we just
        suppress the order submission.
        """
        logger.info("=" * 60)
        logger.info("ROUTER DECISION (10:00)")
        logger.info("=" * 60)

        self.tape_recording_active = False
        now = datetime.now(_ET)

        # Make decision
        decision = self.etf_router.make_decision(now)
        self.router_decision = decision
        self.router_decision_made = True

        # Update state
        self.router_branch = decision.branch.value
        self.router_traded_today = decision.mr_blocked()
        self.mr_blocked_today = decision.mr_blocked()

        logger.info(f"Router decision: {decision.branch.value}")
        logger.info(f"MR blocked today: {self.mr_blocked_today}")

        if decision.symbol:
            logger.info(f"Selected ETF: {decision.symbol}")
            logger.info(f"Entry: {decision.entry_time}, Exit: {decision.exit_time}")
            # Kill-switch gate (issue #5): block 10:00 ETF entry if the
            # daily-loss circuit breaker has tripped.
            if self._check_daily_loss_kill_switch():
                logger.critical(
                    f"ETF entry BLOCKED by daily-loss kill switch — {self.kill_switch_reason}; "
                    f"branch={decision.branch.value} would have bought {decision.symbol}"
                )
            else:
                self._execute_etf_entry(decision)
        else:
            logger.info("No ETF trade - MR allowed at 15:45")

        self._save_state()

    def _execute_etf_entry(self, decision: RouterDecision):
        """Execute ETF entry after 10:00 decision.

        Adds (issue #4) a spread + staleness execution gate via
        ``filter_execution_ready`` and a marketable-limit order (issue #6)
        at ``ask * (1 + ETF_ENTRY_MAX_SLIPPAGE_PCT)`` so a momentary wide
        quote cannot cost more than the configured slippage cap.
        """
        if not decision.symbol:
            return

        symbol = decision.symbol
        logger.info(f"Executing ETF entry: {symbol}")

        try:
            # Calculate position size
            account = self.position_mgr.get_account()
            if not account:
                logger.error("Cannot fetch account for ETF sizing")
                return

            equity = float(account.get("equity", 0))
            etf_capital_pct = getattr(config, "ETF_ROUTER_CAPITAL_PCT", 0.30)
            etf_budget = equity * etf_capital_pct

            # Fresh snapshot for sizing AND for the execution gate.
            snapshots = self.alpaca.get_snapshots([symbol]) or {}
            snap = snapshots.get(symbol, {}) or {}

            # Execution gate (issue #4): tight spread, fresh quote, require quote.
            orderable, exec_rejected = filter_execution_ready(
                [symbol], snapshots,
                max_spread_pct=float(getattr(config, "ETF_ENTRY_MAX_SPREAD_PCT", 0.005)),
                require_quote=True,
                max_stale_seconds=float(getattr(config, "ETF_ENTRY_MAX_STALE_SECONDS", 10.0)),
            )
            if symbol not in orderable:
                reason = exec_rejected.get(symbol, "unknown")
                logger.warning(f"ETF entry REJECTED for {symbol}: {reason}")
                return

            ask = snap.get("ask")
            bid = snap.get("bid")
            last_price = snap.get("last_price")
            if not last_price:
                logger.error(f"ETF entry {symbol}: no last_price after gate (should not happen)")
                return

            # Size off ask (where we'd actually fill) if available, else last.
            sizing_price = float(ask) if ask else float(last_price)
            qty = int(etf_budget / sizing_price)

            if qty <= 0:
                logger.warning(
                    f"ETF qty <= 0, skipping entry: budget=${etf_budget:.2f}, "
                    f"sizing_price=${sizing_price:.2f}"
                )
                return

            # Marketable limit (issue #6): bound slippage above the ask.
            slippage_pct = float(getattr(config, "ETF_ENTRY_MAX_SLIPPAGE_PCT", 0.005))
            if ask:
                limit_price = float(ask) * (1.0 + slippage_pct)
                order_type = "limit"
                logger.info(
                    f"ETF entry {symbol}: qty={qty}, bid={bid}, ask={ask}, "
                    f"last={last_price}, marketable_limit={limit_price:.4f} "
                    f"(ask + {slippage_pct:.2%})"
                )
                order, error_type = self.position_mgr.submit_buy_order(
                    symbol, qty, order_type="limit", limit_price=limit_price,
                )
            else:
                # No ask available — falls back to market order. Should be
                # rare after the execution gate, kept as a defensive path.
                logger.warning(f"ETF entry {symbol}: no ask after gate — using market order")
                order, error_type = self.position_mgr.submit_buy_order(symbol, qty)
                limit_price = None
                order_type = "market"

            price = float(last_price)

            if order and order.get("id"):
                # Wait for fill confirmation (max 10 seconds for liquid ETFs)
                fill = self.position_mgr.get_order_fill(order["id"], max_wait=10)
                if fill and int(fill.get("filled_qty", 0)) > 0:
                    filled_qty = int(fill["filled_qty"])
                    fill_price = float(fill.get("filled_avg_price", price))
                    self.etf_position = {
                        "symbol": symbol,
                        "qty": filled_qty,
                        "entry_price": fill_price,
                        "entry_time": datetime.now(_ET).isoformat(),
                        "branch": decision.branch.value,
                        "planned_exit_time": decision.exit_time.isoformat() if decision.exit_time else None,
                        "order_id": order.get("id"),
                    }
                    logger.info(f"ETF position opened: {symbol} {filled_qty} shares @ ${fill_price:.2f}")
                else:
                    logger.warning(f"ETF entry not filled for {symbol}; canceling and leaving flat")
                    self.position_mgr._cancel_order(order["id"])
            else:
                logger.error(f"Failed to submit ETF buy order for {symbol}: {error_type}")

        except Exception as e:
            logger.error(f"Error executing ETF entry: {e}", exc_info=True)

    def _check_etf_exits(self, current_time: dt_time):
        """Check ETF exit checkpoints at 11:00, 14:00, 15:00."""
        if not self.etf_position:
            return

        symbol = self.etf_position.get("symbol")
        planned_exit = self.etf_position.get("planned_exit_time")

        if not planned_exit:
            return

        # Parse planned exit time
        try:
            if isinstance(planned_exit, str):
                # Handle HH:MM:SS format
                parts = planned_exit.split(":")
                exit_h, exit_m = int(parts[0]), int(parts[1])
            else:
                exit_h, exit_m = planned_exit.hour, planned_exit.minute
        except Exception:
            return

        exit_time = dt_time(exit_h, exit_m)

        # Check if it's time to exit
        if current_time >= exit_time:
            logger.info(f"ETF exit time reached: {symbol} at {current_time}")
            self._execute_etf_exit()

    def _execute_etf_exit(self):
        """Execute ETF exit order with fill confirmation and duplicate guard."""
        if not self.etf_position:
            return

        symbol = self.etf_position.get("symbol")
        qty = self.etf_position.get("qty", 0)

        # Check if exit order already submitted (duplicate guard)
        existing_exit_id = self.etf_position.get("exit_order_id")
        exit_submitted_at = self.etf_position.get("exit_submitted_at")

        if existing_exit_id:
            logger.info(f"ETF exit order already exists for {symbol}; checking fill status")
            fill = self.position_mgr.get_order_fill(existing_exit_id, max_wait=2)

            if fill and int(fill.get("filled_qty", 0)) > 0:
                filled_qty = int(fill["filled_qty"])
                if filled_qty >= qty:
                    logger.info(f"ETF exit filled: {symbol} {filled_qty} shares")
                    self.etf_position = None
                    self._save_state()
                else:
                    remaining = qty - filled_qty
                    self.etf_position["qty"] = remaining
                    self.etf_position["exit_order_id"] = None  # Clear to allow retry
                    logger.warning(f"ETF exit partial fill: {symbol} {filled_qty}/{qty}, {remaining} remaining")
                    self._save_state()
                return

            # If exit has been pending too long, cancel and retry once
            if exit_submitted_at:
                submitted_dt = datetime.fromisoformat(exit_submitted_at)
                age_seconds = (datetime.now(_ET) - submitted_dt).total_seconds()

                if age_seconds > 60:
                    logger.warning(
                        f"ETF exit order {existing_exit_id} pending {age_seconds:.0f}s; canceling and retrying"
                    )
                    try:
                        self.position_mgr._cancel_order(existing_exit_id)
                    except Exception:
                        logger.warning("ETF exit cancel failed", exc_info=True)

                    self.etf_position["exit_order_id"] = None
                    self.etf_position["exit_submitted_at"] = None
                    self._save_state()
                    return

            logger.info(f"ETF exit order {existing_exit_id} still pending; waiting for fill")
            return

        logger.info(f"Executing ETF exit: {symbol} {qty} shares")

        try:
            # Submit sell order
            order = self.position_mgr._submit_sell_order(
                symbol,
                qty,
                order_type="market",
                time_in_force="day",
                extended_hours=False,
            )

            if order and order.get("id"):
                # Record exit order id for duplicate guard
                self.etf_position["exit_order_id"] = order.get("id")
                self.etf_position["exit_submitted_at"] = datetime.now(_ET).isoformat()
                self._save_state()

                # Wait for fill confirmation
                fill = self.position_mgr.get_order_fill(order["id"], max_wait=10)
                if fill and int(fill.get("filled_qty", 0)) > 0:
                    filled_qty = int(fill["filled_qty"])
                    if filled_qty >= qty:
                        logger.info(f"ETF exit filled: {symbol} {filled_qty} shares")
                        self.etf_position = None
                        self._save_state()
                    else:
                        # Partial fill - update remaining qty
                        remaining = qty - filled_qty
                        self.etf_position["qty"] = remaining
                        self.etf_position["exit_order_id"] = None
                        logger.warning(f"ETF exit partial fill: {symbol} {filled_qty}/{qty}, {remaining} remaining")
                        self._save_state()
                else:
                    logger.warning(f"ETF exit submitted but not filled yet for {symbol}; will retry on next tick")
            else:
                logger.error(f"Failed to submit ETF sell order for {symbol}")

        except Exception as e:
            logger.error(f"Error executing ETF exit: {e}", exc_info=True)


def main():
    try:
        bot = CombinedOvernightReboundBot()
    except Exception:
        logging.critical("UNHANDLED EXCEPTION during bot initialisation", exc_info=True)
        raise

    # Install signal handlers for graceful shutdown. The handler only flips
    # a flag; the main loop polls it between ticks so we never interrupt a
    # broker submission / fill check mid-flight. Both SIGINT (Ctrl+C) and
    # SIGTERM (kill / systemd stop) are honored. SIGTERM is not available
    # on Windows so we guard the install.
    def _signal_handler(signum, _frame):
        try:
            sig_name = signal.Signals(signum).name
        except (ValueError, AttributeError):
            sig_name = str(signum)
        if not bot._shutdown_requested:
            bot._shutdown_requested = True
            bot._shutdown_signal = sig_name
            logger.warning(f"Received {sig_name}; requesting graceful shutdown")
        else:
            logger.warning(f"Received {sig_name} again; shutdown already in progress")

    try:
        signal.signal(signal.SIGINT, _signal_handler)
    except (ValueError, OSError):
        logger.warning("Could not install SIGINT handler", exc_info=True)
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _signal_handler)
        except (ValueError, OSError):
            logger.warning("Could not install SIGTERM handler", exc_info=True)

    bot.run()


if __name__ == "__main__":
    main()
