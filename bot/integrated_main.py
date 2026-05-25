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
import os
import signal
import sys
import time
from datetime import datetime, time as dt_time, date, timedelta
from typing import List, Optional, Dict, Any, Tuple
from zoneinfo import ZoneInfo

from bot import config
from bot import entry_executor
from bot import etf_router_runtime
from bot import morning_exits
from bot import premarket_classifier
from bot import premarket_runner
from bot import scoring
from bot import state_io
from bot.massive_client import MassiveClient
from bot.market_data import AlpacaDataClient
from bot.mean_reversion_scorer import MeanReversionCandidate
from bot.green_day_pullback_scorer import GreenDayPullbackCandidate
from bot.etf_router import ETFRouter, RouterDecision
from bot.position_manager_overnight import PositionManager
from bot.state_manager import StateManager
from bot.universe_builder import (
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
        return morning_exits.exit_sleeve_positions(self, sleeve, reason)

    def _exit_single_position(self, symbol: str, reason: str):
        return morning_exits.exit_single_position(self, symbol, reason)

    def _build_open_exit_plan_from_broker(self, reason: str = "broker snapshot") -> List[Dict[str, Any]]:
        return morning_exits.build_open_exit_plan_from_broker(self, reason)

    def _submit_open_exit_market_sells(self):
        return morning_exits.submit_open_exit_market_sells(self)

    def _run_broker_native_rescue(self):
        return morning_exits.run_broker_native_rescue(self)


    # ════════════════════════════════════════════════════════════
    # AFTERNOON DATA & SCORING METHODS
    # ════════════════════════════════════════════════════════════

    def _step_collect_data(self):
        return scoring.step_collect_data(self)

    def _step_score_and_rank(self):
        return scoring.step_score_and_rank(self)

    def _check_daily_loss_kill_switch(self, account: Optional[dict] = None) -> bool:
        return scoring.check_daily_loss_kill_switch(self, account)

    def _compute_mr_etf_regime_size_multiplier(self) -> tuple[float, dict]:
        return scoring.compute_mr_etf_regime_size_multiplier(self)

    def _step_execute_entries(self):
        return entry_executor.step_execute_entries(self)


    # ════════════════════════════════════════════════════════════
    # INFRASTRUCTURE (failsafe, state, etc.)
    # ════════════════════════════════════════════════════════════

    def _position_reference_price(self, broker_pos: dict, symbol: str) -> Optional[float]:
        return morning_exits.position_reference_price(self, broker_pos, symbol)

    def _submit_red_open_trail_or_sell_green(self):
        return morning_exits.submit_red_open_trail_or_sell_green(self)

    def _run_failsafe_flatten(self, label: str):
        return morning_exits.run_failsafe_flatten(self, label)


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

    def _classify_premarket_limit(self, pos, metrics: Dict[str, Any]) -> Dict[str, Any]:
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

    # ────────────────────────────────────────────────────────────
    # Premarket runner — delegated to bot.premarket_runner
    # ────────────────────────────────────────────────────────────

    def _place_premarket_dynamic_limit_sells(self, decision_time_str: str = None):
        return premarket_runner.place_premarket_dynamic_limit_sells(self, decision_time_str)

    def _append_premarket_limits_artifact(
        self,
        decision_time_str: str,
        placed: int,
        no_cap: int,
        skipped: int,
        waited: int,
        is_final: bool,
        symbol_count: int,
        limit_order_ids,
    ):
        return premarket_runner.append_premarket_limits_artifact(
            self,
            decision_time_str=decision_time_str,
            placed=placed,
            no_cap=no_cap,
            skipped=skipped,
            waited=waited,
            is_final=is_final,
            symbol_count=symbol_count,
            limit_order_ids=limit_order_ids,
        )

    def _build_etf_router_summary(self) -> Dict[str, Any]:
        return etf_router_runtime.build_etf_router_summary(self)

    def _save_etf_router_artifact(self):
        return etf_router_runtime.save_etf_router_artifact(self)


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
        return etf_router_runtime.run_startup_phase(self)

    def _cancel_stale_etf_orders(self):
        return etf_router_runtime.cancel_stale_etf_orders(self)

    def _initialize_tape_recording(self):
        return etf_router_runtime.initialize_tape_recording(self)

    def _update_tape(self, force: bool = False):
        return etf_router_runtime.update_tape(self, force=force)

    def _make_router_decision(self):
        return etf_router_runtime.make_router_decision(self)

    def _execute_etf_entry(self, decision):
        return etf_router_runtime.execute_etf_entry(self, decision)

    def _check_etf_exits(self, current_time):
        return etf_router_runtime.check_etf_exits(self, current_time)

    def _execute_etf_exit(self):
        return etf_router_runtime.execute_etf_exit(self)



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
